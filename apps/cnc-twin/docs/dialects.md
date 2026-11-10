# Controller dialects

## Structure of the controller

The controller turns program text into the `Command` stream motion executes, exactly as the
control would: in the control's dialect, with its modal rules, offsets, macro language, alarms and
timing. It is a domain without a bus record (it reads and writes none, architecture 7); it talks
to motion through `cnctwin/controller/ir.py` and the lock-step API below.

### Modules

| module | role |
|---|---|
| `ir.py` | `Command`, `Feedback`, `ProbeResult`, `SetterResult`, `OperatorPanel` (the interface to motion) |
| `lexer.py`, `program.py` | ISO word-address text -> `Program` / `Block` / `Word` (immutable); `ProgramStore` |
| `expr.py` | custom macro B expressions and statements: parser (AST tuples) and `Evaluator` |
| `executor.py` | `PathExecutor`: the dialect-neutral state machine of one path, block-time model, coordinate pipeline, emission |
| `pipeline.py` | programmable transforms (mirror, scaling, rotation, feature frames) and arc geometry |
| `controller.py` | `Controller`: one executor per path, path-wait coordination, binding of a restored state |
| `state.py` | `ControllerState` / `PathState` / `Frame` ...: all executor state as data, `to_arrays()` / `from_arrays()` |
| `hooks.py` | extension registries (cycles, compensation, transforms, probing, tool life, restart) and their data (`Step`, `CycleCall`, `PathItem`) |
| `machine.py` | `MachineModel`: the machine definition as the control sees it (axes per path, counts, devices, table kinematics) |
| `alarms.py` | `AlarmTable`: condition key -> number and text (controller database first) |
| `dialects/` | registry (`base.py`), FANUC 0i-F (`fanuc.py`, `fanuc_tables.py`, `fanuc_vars.py`), Haas NGC (`haas.py`) |
| `haas_cycles.py`, `haas_font.py`, `contour.py` | Haas G12/G13, G47, G150 cycle providers, engraving font, 2-D contour offsets |
| `measure.py` | default G37 / lathe G36-G37 tool measurement provider |
| `kernels.py`, `abi.py` | Numba kernels (G150 raster intervals, link clearance) with Python references |
| `schema.py` | `physics.controller` settings (schema + `ControllerSettings`) |
| `domain.py`, `channels.py` | `ControllerDomain` (core.domain.Domain) and its channels |
| `dryrun.py` | `dry_run()`: the program against an ideal machine (check harness, tests, cycle-time lower bound) |
| `serial.py` | JSON form of dataclasses by type hints (checkpoints, buffered commands) |

### Data flow

```
program text --lexer(LexerRules of the dialect)--> ProgramStore (programs, blocks; immutable)
      |
PathExecutor._step(): pending cycle Step | pending call | block at frames[-1].pc
      |-- dialect handler.execute(block): words -> values (expressions, <vacant>), G/M validation,
      |   modal groups, F S T, one-shot codes, motion or cycle, M codes, flow control
      |-- core services: move()/arc_move()/machine_move()/dwell()/emit(), goto/loops/calls/ret,
      |   variables, alarms, waits
      v
PathItem (work frame: after mirror/scale/rotation/feature, before offsets)
      |-- compensation filter (hooks.COMPENSATION, look-ahead) when G41/G42/... is active
      v
final stage: default (machine = round(work/q) + offset counts) | transform provider (TCP, DWO, ...)
      v
Command (ready_tick, path, block_seq) -> look-ahead buffer -> produce() returns it to motion
```

### The state machine (architecture 6)

All mutable state is in `PathState` (one per path) and `ControllerState` (paths, shared commons,
the program digest, the wait sequence): the call stack `frames` (each `Frame` has the program
index, the program counter `pc`, its local-variable level, remaining repetitions and entry block,
its macro arguments and its own DO-loop stack `loops`), locals per macro level, the common
variable tables, modal codes, positions as integer counts (`pos` program frame, `mach` commanded
machine position, `applied` tool offsets contained in `mach`), transforms, offsets, spindle and
coolant state, the canned-cycle modal data, the look-ahead `buffer` of produced and not retired
commands, the `pending` cycle expansion (a list of `hooks.Step` data), the compensation filter
data (`comp_state`), what the path waits for (`waiting`), fresh live values for the current
statement, the block-time clock `interp_tick`, events (alarms, warnings) and the operator panel.
Blocks are never interpreted by Python recursion or generators: a call pushes a `Frame`, a return
pops it, a loop records `LoopEntry(m, do_pc, end_pc)`, a buffer-stopping read leaves `pc` on the
block and records `waiting`. The executor object itself holds only the immutable context and
per-step caches derived from the state. `ControllerState.to_arrays()` returns the variable tables
as float64 arrays (NaN = `<vacant>`) and everything else as JSON (`serial.dump`; floats round-trip
exactly); `from_arrays()` restores it, and `Controller(setup, state)` checks the program digest
(SHA-256 of the program texts) before continuing. `tests/unit/controller/test_checkpoint.py`
checkpoints in the middle of a WHILE loop (also with a half-emitted Haas cycle and while waiting
for a probe), restores into a fresh controller and requires the continued command stream to equal
the uninterrupted one exactly.

### Lock-step interface (what core.loop and motion do)

`ControllerDomain` (and `Controller`, without the run context) offers:

| call | when | effect |
|---|---|---|
| `produce(ctx, state, path, now_tick) -> [Command]` | start of each macro step, after every retire/feedback | interprets until the buffer holds `capacity()` commands (machine `lookahead_blocks`; Haas G103 P lowers it), a buffer-stopping block is reached, an alarm stops the path, the program ends, or `interp_tick > now_tick + horizon` (`horizon_s`, default 10 ms); returns the commands it appended |
| `retire(ctx, state, path, count, tick)` | motion finished the oldest `count` commands | frees buffer room; a blocked interpreter resumes at `tick` |
| `feedback(ctx, state, path, Feedback, tick)` | the buffer-stopping command ending the buffer is done | see table below; the interpreter resumes at `tick` |
| `cycle_start(ctx, state, path, tick)` | operator CYCLE START | continues after M00/M01/#3006/single block, or runs the next part after M02/M30/M99 |
| `reset(ctx, state, path, tick)` | operator RESET | clears an alarm, rewinds the main program (modal state kept, FANUC) |
| `operator(state, path) -> OperatorPanel`, `set_operator(state, path, **kw)` | every macro step | feed/rapid/spindle overrides, feed hold, single block, optional stop, block delete switches |
| `done(ctx, state)` | | every path ended or stopped by an alarm |

Buffer-stopping commands and the feedback motion owes for each:

| command | reason | feedback |
|---|---|---|
| `SYNC` `sync.reason == "live_read"` | a macro read of #5021-, #5041-, #5101- (encoder positions) | `Feedback.machine_pos` = actual axis positions (commanded minus following error) when in position |
| `PROBE` (G31, probing steps) | skip move | `Feedback.probe = ProbeResult(triggered, latched machine_pos, time)`, `Feedback.machine_pos` = where the axes stopped |
| `TOOL_SETTER` (G37/G36) | measurement | `ProbeResult` (latched positions) or `SetterResult(length_m)` |
| `SYNC` `"path_wait"` | multi-path wait code | any `Feedback` once in position (the controller releases all paths of the wait together) |
| `SYNC` `"exact_stop_check"` (G04 without time) | in-position check | any `Feedback` |
| `STOP` (`M00`, `M01`, `single_block`, `M02`, `M30`, `M99-end`), `MESSAGE` (#3006) | stop | `cycle_start` (no feedback needed) |
| `ALARM` | the control stopped | none: motion decelerates to a stop at once (the real control raises PS alarms when it pre-reads the block) |

A machine-side alarm (`Feedback.alarm = {code, text}`) stops the path with that alarm;
`Feedback.tool_life_expired` is passed to the tool-life provider.

### Commands (conventions on top of ir.py)

* `target`: every axis of the path, absolute machine values in SI (metres, radians; an X axis
  programmed in diameter is the radius); continuous rotary axes are unwrapped (FANUC roll-over:
  G90 takes the shortest way, G91 the increment). `Space.TIP` (Haas G234/G254, TCP providers):
  `tip_target_m` is the tool tip in the control's work frame (program frame after the programmable
  transforms and G52/G92, without the work offset), `tip_axis` the tool axis in that frame,
  `modal["work_origin_M0_m"]` the work zero in machine coordinates at rotary zero,
  `modal["tip_interpolation"]` `"tip"` (TCPC: keep the tip on the path) or `"joint"` (DWO: end
  points only); `target` still holds the commanded machine axes of the end point.
* `arc`: `centre` in the same space as `target`, with the start point's value along the normal;
  `normal` the rotation axis (+Z for G17, +Y for G18, +X for G19, rotated by G68/G268);
  `sweep_rad` signed (CW negative; full circles +-2 pi); `radius_start_m != radius_end_m` only within
  the arc tolerance (a spiral).
* `feed`: `FeedMode.RAPID` with value 0 (= machine rapid), per minute (m/s), per revolution (m/rev),
  inverse time (1/s); `override_allowed` is false for threading (G32/G33), G63 and #3004 bit 1.
* `smoothing.corner`: `exact_stop` (G61, G09, cycle steps that ask for it) | `blend` | `tapping` (G63);
  `tolerance_m` and `mode_code` from G187 / G05.1 / G08 and the machine's smoothing table.
* `modal` snapshot: `plane, units, distance, feed_mode, work_offset, work_offset_m (per axis), tool,
  h, d, length_offset_m, radius_offset_m, comp, motion, cycle, css, exact_stop` (+ `offset_number` and
  `program_offset_m` on lathes: work + tool offsets per path axis, program = machine - it, X as a
  radius, the radius constant surface speed is computed at; `feed_hold_inhibit` with #3004, `rigid_tap`
  after M29, `thread` for G32/G33, `reference_point`, `machine_coordinates`, `tool_preselect` ...).
  TCP commands (Haas G234/G254, FANUC G43.4/G43.5/G68.2) add `tip_start_m`, the tip at the start of the
  move in the frame of `tip_target_m` (the tip path's length is the distance to `tip_target_m`).
* `spindle`: `{id, omega_rad_s (M03 < 0), css_m_s, max_rad_s, gear, orient_rad, direction, css_axis}`;
  `coolant`: `{pumps: {id: on}, nozzles: {id: position}, air, code}` (the full state after the change);
  `tool`: `{number, pocket|station, slot (core ToolTable, -1 unknown), length_offset, radius_offset,
  offset (lathe), turret, probe}` (`probe`: the T number calls a spindle probe of the project's tool
  list, named after the machine probe it is; it is not a tool-table slot); `probe` (PROBE commands):
  `{skip, stylus (the active probe's name, else spindle_probe), expect, start_m}` with `start_m` the
  machine position the skip move starts from (an ideal probe needs it); devices (`PALLET`, `DOOR`,
  `CONVEYOR`, `CHUCK`, brakes, outputs as `MODE`): `modal["device"] = {kind, action, code, ...}`.
* `ready_tick`, `path`, `block_seq` (commands of one NC block share it), `sync` (above).

### Block-time model (data starvation)

The interpreter has its own clock (base ticks). Block k is ready at

    t_ready(k) = max(t_ready(k-1), t_free(k), t_fb(k)) + max(c(k), 1 tick)

where `t_free` is the retire tick that gave a full buffer room again, `t_fb` the feedback tick after
a buffer stop, and the cost `c(k)` is

    NC block:        c = t_block + n_terms * t_term          (t_block = machine controller.block_processing_ms)
    macro statement: c = t_stmt  + n_terms * t_term
    + t_call per M98/M97/G65/G66/code call and M99 return, + t_goto * (blocks searched) per GOTO / M99 P

(`n_terms` = operators, functions and variable accesses of the block's expressions). Every command
of the block carries `ready_tick = t_ready(k)`; cycle steps carry the tick at which the core turns
them into commands. The planner never starts a command, nor looks past it, before its ready tick:
a sequence of blocks each shorter than `v * t_block` cannot be run at feed v, which is the
data-starvation limit of CNC look-ahead (Altintas 2012, ch. 6; Erkorkmaz & Altintas 2001). The
interpreter stops when the buffer is full and resumes at the retire tick, so the look-ahead depth
is real too.

| parameter | value | source |
|---|---|---|
| `t_block` | machine file `controller.block_processing_ms` (1 ms for the Haas/FANUC machines in db/) | est (machine file notes) |
| `t_stmt` `macro_statement_s` | 0.05 ms | est, db/models/controller-block-time.yaml |
| `t_term` `macro_term_s` | 0.005 ms | est |
| `t_call` `call_s` | 0.2 ms | est |
| `t_goto` `goto_scan_s` | 0.002 ms per block | est |
| floor `min_block_s` | 1 us | modelling floor (a loop without motion advances time) |
| `horizon_s` | 10 ms | numerical: how far ahead one produce() may interpret; results do not depend on it |

Validity and error: the queueing model is exact for a control that processes blocks sequentially
at a fixed cost; its error is that of the parameters. None is published by FANUC or Haas: the
block time is an order-of-magnitude estimate (0.5-2 ms for current controls), so a program that
starves has a cycle-time error proportional to the error of `t_block` (estimated +-50 %); a program
whose blocks take longer to execute than to prepare is unaffected. Macro costs (est, x/÷ 3) matter
only for macro loops without motion. `tests/unit/controller/test_timing.py` checks the recurrence
(ready ticks 1 ms apart, resumption at the retire tick, macro statement cost, starvation of 0.05 mm
segments at 10 m/min).

### Coordinate pipeline

Positions are integer counts of the axis quantum `q` (machine `least_input_increment`; for an X
axis programmed in diameter `q/2` of radius = one diameter increment), so incremental chains, full
circles (start = end in counts) and reads are exact. A program value is rounded half away from
zero to the least input increment of its unit (0.001 mm, 0.0001 in, 0.001 deg for IS-B), then to
counts. Program frame -> work frame (FANUC order):

    p1 = mirror(p)          p1_a = 2 c_a - p_a                 for mirrored axes (G51.1 / Haas G101)
    p2 = scale(p1)          p2_a = c_a + s_a (p1_a - c_a)      (G51; s_a < 0 mirrors)
    p3 = rotate(p2)         [a b] = c + R(theta) ([a b] - c)    in the active plane (G68)
    w  = feature(p3)        w = o + M(I, J, K; Q) p3            (Haas G268)
    machine counts = round(w / q) + WCS + EXT + G52 + G92 + tool offsets

All stages are affine, so the inverse (G92, #5041, skip positions, a work offset change) is exact;
under TCP/DWO the provider's `work_from_machine` inverts the table kinematics. Error bound: q/2
per rounding (0.5 um on a 0.001 mm machine); the transforms are exact in double precision.

### Arcs

From the start S, end E (plane axes a, b) and direction d (+1 G03, -1 G02):
I/J/K give the centre C = S + (I, J, K) (always incremental); R gives the chord midpoint M, the
half chord h_c = |E - S| / 2 and C = M + side * sqrt(R^2 - h_c^2) * n_left, side = d * sign(R), so
R > 0 is the arc of at most 180 deg and R < 0 the larger one. Sweep = angle from S to E about C in
the sense d, in (0, 2 pi]; S = E with I/J/K is a full circle, with R no motion (FANUC). The end-point
check: |r_E - r_S| > tolerance (FANUC parameter 3410, default 0.02 mm; Haas 0.0254 mm, est) ->
PS0020 / Haas 304, an R shorter than the half chord by more than the tolerance -> PS0020 / Haas 302;
within it the arc is a spiral from r_S to r_E. Axes outside the plane move linearly (helical
interpolation). Through the transforms the arc's normal is `L a x L b` (L the linear part), so a
mirror reverses the turning sense and rotations keep it; with unequal in-plane scaling FANUC does
not draw an ellipse: R is scaled by the larger in-plane magnification and the I/J/K arc is rebuilt
from the scaled points (a spiral); Haas alarms (467).

### Hooks (for the cycles/compensation/TCP/probing/tool-life/restart agents)

Providers register at import; `hooks.load_plugins()` imports, in this order, the packages
`cnctwin.controller.{cycles, comp, probing, toollife, restart, tcp, dialects.siemens840d,
dialects.heidenhain}` when they exist (a package that exists but fails to import raises).
Registering an existing key raises unless `replace=True`.

* **Cycles** - `register_cycle(code, provider, dialect="")`; key `"<dialect>:<code>"` wins over
  `"<code>"`. `provider.expand(ex, call: CycleCall) -> list[Step]`. Canned cycles (group 09 on the M
  series, group 10 on the T series): the core keeps the modal cycle code and data (`R Q I J E` as
  lengths in SI, `P` dwell in s, `K`/`L` repetitions, `F` the SI feed, the drilling-axis word in SI
  with `data["absolute"]`), the initial level (drilling-axis value when the cycle mode began) and
  G98/G99; a block with any axis word or R executes the cycle `K` times (K0: data only), each at
  the next incremental position in G91; `call.position` is the hole position (program frame,
  absolute SI), `call.extra["drilling_axis"]` the plane normal (Z/X on lathes for face/side cycles).
  A group-01 code cancels the cycle mode. One-shot cycles (lathe G70-G76, Haas G12/G13/G47/G150...)
  and the T series box cycles G90/G92/G94 (modal data merged block by block) get every word in
  `call.data` (axis words and `I J K R Q E U V W` as SI lengths, `F` in SI, others as written) and
  the exact literal form in `call.extra["word_values"][letter] = [value, has_point, from_variable]`
  (apply `ex.handler.lin(WordValue)` etc. for context rules) plus `call.extra["comment"]`.
  Lathe cycles that need their finishing shape read the program directly:
  `ex.store.programs[ex.s.frames[-1].program]`, `.find_n(n)`, `ex.handler.evaluate(block.words)`,
  and continue after the shape with `ex.goto(n)`/frame pc. Steps are executed in order by the core
  through the full pipeline (`comp=False` bypasses cutter compensation, `machine=True` is G53-like,
  `exact_stop`, `override`, `feed`/`feed_mode`); kind `command` emits a prebuilt `Command`, `dwell`,
  `mode` (`handler.apply_mode`), `probe`/`setter` (buffer-stopping, continued by the provider named
  in `step.provider` with `resume(ex, step.data, feedback)`), `resume` (continuation without motion).
  Missing provider: TW0001 (`cycle_unavailable`).
* **Compensation** - `register_compensation(kind, factory)`, kinds `cutter` (G41/G42 milling), `nose`
  (G41/G42 lathes), `3d` (Haas G141), `corner` (`,C` `,R`). G41/G42 call `ex.start_comp(kind)`
  (TW0002 without a provider); from then on every item (motions and the non-motion commands between
  them, in order) goes through `filter.push(ex, item) -> released items`; `ex.flush_comp(reason)`
  is called at buffer stops, program stops, tool changes, G53, G28 and path waits; the filter
  reads `item.comp` (G40 ends compensation: release everything and call `ex.end_comp()`),
  `item.comp_radius_m`, `item.plane`, `item.bypass` (cycle positioning) and modifies `end`, `start`,
  `centre`, `sweep_rad`, radii in the work frame; its whole state is `to_data()` (JSON types),
  restored by `factory.from_data(ex, data)`. A FANUC block with `,C` / `,R` / `,A` gives its item
  `item.corner = {",C": m}` (or `",R"` m, `",A"` deg); with no compensation active the core starts the
  `corner` filter for it, which ends itself (`ex.end_comp()`) once the corner is resolved with the
  next block; an active cutter/nose filter receives the `corner` field itself.
* **Transforms** - `register_transform(key, provider)`: a G code of kind `tcp` (FANUC G43.4/G43.5,
  G68.2, G07.1, G12.1; Haas G143, G107, G112, G253) sets `ex.s.tcp = key`; every motion item then
  goes to `provider.to_command(ex, item) -> Command` (optional `machine_counts(ex, item)` and
  `work_from_machine(ex, machine_counts, applied)` keep the executor's positions and reads exact).
  Missing provider: TW0003. Haas G234/G254 are implemented this way (`TableTcp`).
* **Probing** - `register_probe(key, provider)`: `start(ex, call) -> [Step]` (queued ahead of the
  next block), `resume(ex, data, feedback) -> [Step]`. After a probe/setter step the core has set
  `ex.s.skip` (program-frame counts of the latch, or the end point without trigger),
  `ex.s.skip_triggered`, and the position where the axes stopped. Default providers: `G37` (FANUC and
  Haas), `G36` (lathe), `measure.ToolMeasure`. Missing: TW0004.
* **Tool life** - `register_tool_life(provider)`: `select(ex, number) -> number used` at every M06 (mill)
  and turret T call (lathe), `expired(ex, numbers)` on feedback. Without one, a call of an expired
  tool is a warning.
* **Restart** - `register_restart(provider)`: `reconstruct(ex, target) -> [Step]`. The core offers
  `ex.search(predicate)`: interprets without producing commands until `predicate(ex)` holds, keeping
  modal state, offsets, positions and `ex.s.restart_summary` (the latest SPINDLE/COOLANT/TOOL_CHANGE/
  MODE commands to re-establish).

Core services a handler or provider may call (executor): `move(kind, target_counts, feed, ...)`,
`arc_move(...)`, `machine_move(...)`, `dwell(s)`, `emit(Command)`, `queue_steps(steps, front)`,
`feed_spec(...)`, `smoothing_spec()`, `counts(letter, si)`, `si(letter, counts)`, `transform()`,
`offset_counts()`, `program_from_machine(...)`, `resync_program_position()`, `invalidate_transform()`,
`goto(n)`, `loop_while(cond, m)`, `loop_end(m)`, `request_call(kind, program, args, repeat, entry_pc)`,
`ret(n)`, `end_program(code)`, `wait(kind, **data)`, `read(n)`/`write(n, v)`, `note(kind, message)`,
`line_ref()`, `start_comp/flush_comp/end_comp`, `search(...)`.

### Alarms

The executor raises `ControllerAlarm(key, detail)`; `AlarmTable.resolve` gives, in this order, the
controller database entry whose `kind` is the key, else the dialect's number for the key with the
database's text for that number, else the dialect's built-in text (`source: builtin`). The path
stops, an `ALARM` command ends the buffer and the event becomes a finding (category alarm, severity
error, at the block that raised it). Twin alarms (`TW0001`-`TW0021`, `alarms.TWIN_ALARMS`) mark what
the real control has no number for: features of other agents not yet present, M codes the machine
definition does not declare, variables this build does not model, an unreleasable path wait; a
database entry with that `kind` replaces them (e.g. a builder's PMC alarm for an undefined M code).

### Multi-path

Axes are grouped into paths by `MachineModel` (trailing digit of the names on `swiss` machines or
with `controller.two_path`/`paths`; `physics.controller.paths` overrides). Each path has its own
executor, program (`$1`/`$2` sections of the operation's file, or `program_texts`), spindle
(path 1 main, path 2 sub), offsets and variables (commons shared per `shared_common_100/500`).
A wait code emits `SYNC {reason: path_wait, code, paths, id}` and waits; `Controller.arrive` releases
all named paths at the latest arrival tick; different codes -> PS0160 on both; a P naming a path
that does not exist -> PS0161; waiting for a path that ended or alarmed -> TW0008. Other dialects
(Siemens WAITM) use the same `ex.wait("path_wait", code=..., paths=[...])`.

### Determinism, bus, channels, findings

No random numbers (no RNG streams). Iteration only over lists and insertion-ordered dicts; all
caches in the executor object are per step and derived from the state. Bus: reads none, writes none
(lock-step API only). Channels (macro tier, per path): `controller.buffer_fill`,
`controller.interp_lead_s`, `controller.block_rate_Hz`, `controller.waiting`, `controller.alarm`
(`ControllerDomain.last_channels` after each macro step). Findings: every controller event
(alarm -> error, warning, info) at its interpretation tick with the program, line and block of the
block that caused it. `ControllerDomain.finalize` reports blocks, commands, alarms, parts and DPRNT
output per path.

### Dry run

`dry_run(controller)` executes the commands on an ideal machine (constant speed, no acceleration,
no following error; rapids per axis at the rapid rate, the slowest axis decides; tool change =
machine `tool_change_s`) and answers buffer stops itself (probe/setter callbacks). Feeds follow the
control's semantics: per minute over the combined machine axes (rotary degrees count as
millimetres), except under tool centre point control with tip interpolation (Haas G234, FANUC
G43.4/G43.5), where F is the tip's speed along the tip path (`tip_start_m` -> `tip_target_m`, the
arc's length for an arc; a block that only turns the tool about its tip moves the rotary axes at F
read as degrees per minute); per revolution with the commanded speed, or under constant surface
speed with n = min(v / (2 pi r), n_max) at the program radius of the CSS axis integrated exactly along
the move (lines in closed form, arcs over 1 degree chords); inverse time as 1 / F. A probe that
triggers ends its skip move at the latch (the move's time is cut there; the PROBE payload gets
`latched_m`). `dryrun.BoxProbe(boxes, radius_m, ...)` is an ideal ball-stylus probe against axis-aligned
boxes in machine coordinates (the ball's bottom at the applied length offset, contact found to 0.1 um
by bisection); `domain.stock_probe(project, op_index)` builds one against the stock as set up (3-axis
convention: the table frame is the machine frame), with the stylus ball of each probe T number.
`PathExecutor.block_log` (a list, off by default, not state) records every executed block
`{seq, depth, program, line, words}` with its evaluated address values: program checks
(cnctwin.sanity) read what the executor evaluated from it. The cycle time is a lower bound of the
real one; a feed per revolution with the spindle stopped ends it with a stall message (the real axes
would not move).

### Assumptions and limitations

* Canned cycles, cutter compensation, TCP/tilted planes beyond Haas G234/G254/G268, probing cycles
  beyond G31/G37 and tool life management are providers (section "Cycles and compensation"); a code
  without one raises TW0001-TW0004 instead of guessing.
* The machine model reads the axis definitions of the machine's `axes` mapping and skips the source
  keys a sourced mapping carries beside them (`src`, `note`, `notes`, `page`, ...); motion's machine model
  does the same.
* A work offset the project writes as a list (`setup.work_offsets_mm`) is the work zero's machine
  coordinates X, Y, Z, then the path's other axes, on every machine type; `setup_from_project`
  (`domain.offset_row`) puts it in the executor's axis order (X, Z, Y, C on lathes). Lists handed to
  `ControllerSetup` directly are in the executor's order. Consequence: a project list written in the
  lathe's executor order (X, Z, ...) puts the work zero on the wrong axes.
* Tool numbers: a `kind: probe` entry of the project's tool list is a probe T number (no "not in the
  tool list" warning; tool changes carry `tool.probe`); the setup takes the lathe geometry offsets
  (X as set, Z, nose radius, tip, X/Z wear) and each entry's 1-based `path`.
* Single block and operator switches act at the next block the interpreter processes (up to the
  look-ahead depth later than on the control when switched during a run).
* The block-time parameters and the arc tolerances are estimates (above). Consequence: data starvation
  on short-segment programs and the arc end-point alarms are as good as those estimates (block_time can
  be calibrated from machine data, docs/models/calibration.md). FANUC alarm texts are built-in until
  db/controllers/fanuc-0i-mf.yaml lists them (Haas texts come from the database where present).
* #4201-#4330 (modal data of the executing block), interface signals without `interface_inputs`,
  and the variables listed per dialect below are not modelled and raise TW0006/TW0007.

## Fanuc (0i-MF / 0i-TF, custom macro B)

`cnctwin/controller/dialects/fanuc.py` (handler), `fanuc_tables.py` (G codes, alarm numbers and
built-in texts, address formats), `fanuc_vars.py` (variables). The variant follows the machine type:
`vertical_mill`, `horizontal_mill`, `five_axis_mill` use the M series G codes (0i-MF); `lathe`,
`mill_turn`, `swiss` the T series G code system A (0i-TF). Parameter defaults are in
`db/models/controller-fanuc.yaml` (each with its FANUC parameter number and source); every one can
be set in the project under `physics.controller` (`cnctwin/controller/schema.py`). The controller
database (`db/controllers/fanuc-0i-mf.yaml`, `-tf.yaml`) gates the G codes it lists and supplies
alarm texts when it lists them (it lists none yet, so the built-in texts are used and marked).

### Program format

ISO 6983 word address: `%` ... `%` tape marks, `Onnnn` programs (several per file, 4 digits),
`Nnnnnn` sequence numbers (5 digits), `( )` comments, `;` end of block (text after it is a comment),
`/` and `/1`-`/9` block delete at the start of a block or from the middle of a block to its end.
Literal formats: at most 9 significant digits for lengths, 5 for G, 8 for M/T/P/L, 3 for H/D, 5 for S
(PS0003). Errors found while reading (no data after an address PS0005, two minus signs PS0006, two
decimal points PS0007, an unknown character PS0009) are raised when the block is executed, so a
program fails at the block that is wrong, not at load.

**Decimal-point rule.** A value without a decimal point is a count of least input increments for
the lengths and angles X Y Z U V W A B C I J K Q R (X10 = 0.010 mm, A90000 = 90 deg), and also in
G04 X (0.001 s); with a decimal point it is in mm/inch/deg/s. Values of macro variables count as
written with a decimal point. F without a decimal point is in units of its least increment:
1 mm/min, 0.01 in/min, 0.01 mm/rev, 0.0001 in/rev (unverified against the 0i-F manual; FANUC
programs write F with a point in practice). Addresses that take no decimal point (M, S, T, H, D, P,
L, N, O and G04 P) raise PS0007 with one. "Pocket calculator" input (parameter 3401#0 DPI = 1:
values without a point are mm/inch/deg) is off by default (`pocket_calculator: true` turns it on).
Every value is rounded half away from zero to the least input increment of its unit (IS-B: 0.001 mm,
0.0001 in, 0.001 deg; the machine's `least_input_increment_mm/deg`) and then to counts; inch values
on a metric machine are rounded again to 0.001 mm (X1.00005 in -> 1.0001 in -> 25.403 mm).

### Order of execution in a block

1. A macro statement (assignment, IF, WHILE, DO, END, GOTO, DPRNT/POPEN/PCLOS) is the whole block
   (NC words with it: PS0127).
2. Word values are evaluated (expressions, `<vacant>`); a vacant lone variable drops its address.
3. G codes are validated (not in the table or not listed by the database: PS0010; listed but not
   modelled: TW0012); G codes of one group: the last one is used (FANUC rule) and a warning is
   recorded; two group-00 codes that take axis words: PS0010; macro-call codes and G65 make the
   block a call. M codes: at most `max_m_codes_per_block` (3: parameter 3404#7 M3B = 1, est), each
   defined by the dialect for this machine's devices, the project's `m_codes`, the wait range or a
   macro-call code (else TW0005, or the database's `undefined_m_code` alarm).
4. Modal groups in group order (units first), then F, S, T, H, D.
5. M codes that start something (M03, M04, M07, M08, M29 ...) before the motion; the rest
   (M05, M09, M06, M19, M00, M01, M02, M30, M98, M99, waits) after it (the common PMC convention;
   machine builders may differ).
6. One-shot codes, or the motion of group 01 / one canned-cycle execution, then the G66 call.
7. Any word no function used: PS0009 (an axis address the path does not have, Q with G01 ...).

### G codes (M series)

| group | codes | behaviour |
|---|---|---|
| 01 | G00 G01 G02 G03 G33 | rapid, linear, arcs (R / I J K, helical), threading (F = lead, override off); cancels a canned cycle |
| 00 | G04 | dwell: P ms (integer), X s; alone: exact-stop check (SYNC) |
| 00 | G05.1 Q1/Q0, G08 P1/P0 | AI contour / preview on/off: MODE with the machine's smoothing tolerance |
| 00 | G09 | exact stop for the block |
| 00 | G10 L2 P0-6, L20 P1-300, L10-L13 P R (L1, the memory A form, sets the geometry) | work offsets EXT/G54-G59, G54.1 Pn; H/D geometry/wear; G91 adds |
| 00 | G27, G28, G29, G30 P2-4 | reference check (PS0092), return via the intermediate point (rapid, cutter comp bypassed), from reference, 2nd-4th reference points (`reference_points_mm`, default machine zero as parameters 1241-1243) |
| 00 | G31 (P = skip signal) | skip move: PROBE, buffer stop, #5061-; comp active: PS0035 |
| 00 | G37 | tool length measurement (default provider; see Hooks) |
| 00 | G52, G53, G92, G92.1, G60 | local shift (G91 incremental), machine coordinates (rapid; ignored in G91 with a warning), coordinate setting (cancels G52), preset, single-direction positioning (`g60_overrun_mm`) |
| 00 | G65 | macro call |
| 02 | G17 G18 G19 | planes (change in cutter comp: PS0037) |
| 03 | G90 G91 | absolute / incremental |
| 04 | G22 G23 | stored stroke check (MODE with the limits) |
| 05 | G93 G94 G95 | inverse time (F in every cutting block), per minute, per revolution |
| 06 | G20 G21 | inch / mm input |
| 07 | G40 G41 G42 | cutter compensation (provider) |
| 08 | G43 G44 G49, G43.4 G43.5 | length compensation along Z (type A, `tool_length_axis: C` for the block's axis), TCP (provider) |
| 09 | G73 G74 G76 G80 G81-G89 G81.1 | canned cycles (provider), cancel |
| 10 | G98 G99 | return to initial level / R level |
| 11 | G50 G51 | scaling: centre X Y Z, P (all axes) or I J K (per axis), without a point in 0.001 (parameter 5400#7), negative = mirror; 0: PS0142 |
| 12 | G66 G67 | modal macro call |
| 13 | G96 G97 | constant surface speed (also on mills) |
| 14 | G54-G59, G54.1 P1-P300 | work offsets |
| 15 | G61 G62 G63 G64 | exact stop, corner override, tapping (override off), cutting |
| 16 | G68 G69, G68.2 | rotation (centre X Y in the plane, current position if omitted; R absolute, parameter 5400#0 RIN = 0), tilted plane (provider) |
| 17 | G15 G16 | polar coordinates (G90: radius from the work zero; G91: from the current point, angle added to the last polar angle; an omitted radius/angle keeps the current one) |
| 19 | G25 G26 | spindle speed fluctuation detection (MODE) |
| 21 | G12.1 G13.1 | polar coordinate interpolation (provider) |
| 22 | G50.1 G51.1 | programmable mirror image |
| 00 | G07.1, G39 | cylindrical interpolation (provider), corner arc (compensation provider) |

### G codes (T series, system A)

Absolute X/Z/Y/C, incremental U/W/V/H; X and U in diameter (`diameter_programming_X`), I and R in
radius. G00 G01 G02 G03 G32 (thread) and the box cycles G90 G92 G94 (group 01, modal, provider) and
G34; G04 (P, X, U); G10 (P1-64 wear, P10001-10064 geometry; L1 geometry, L11 wear; X Z Y R Q, U W V C
incremental; offsets in diameter, parameter 5004#1 ORC = 0) and G10 L2/L20; G17 G18 (power-on) G19
(group 16); G20 G21; G22 G23; G25 G26; G27 G28 (G28 U0 W0) G29 G30; G31; G36 G37 (automatic tool
offset X/Z, default provider); G40 G41 G42 (nose radius compensation provider); G50 (S = maximum
spindle speed for G96, axis words = coordinate setting), G50.3; G52 G53; G54-G59; G65 G66 G67;
G70-G76 (one-shot, provider); G80 and G83-G89 (group 10; face cycles drill along Z, side cycles along
X; provider); G96 G97 (group 02); G98 G99 (group 05, G99 at power-on). The T word Tttoo indexes the
turret to station tt (TOOL_CHANGE; above the turret's stations: TW0013) and selects offset oo
(geometry + wear X, Z, Y and nose radius), applied with the next motion (parameter 5002#6 LWM = 1);
oo = 00 cancels it. Modelled: the offset applies with the next motion (LWM = 1) and the tool offset memory is type C (separate geometry and wear for H and D); LWM = 0 and memory A are not modelled.

### Spindle, coolant, tools, program end

S is rpm (G97) or surface speed in m/min or ft/min (G96, `css_m_s`, the spindle speed follows the
X radius, clamped by G50 S / G92 S and the spindle maximum); M03 gives omega < 0 (architecture 4),
M04 omega > 0, M05 stop, M19 orientation (R angle), M41-M44 gears when the spindle has them. S above
the spindle maximum is clamped with a warning. M07 mist/MQL pumps, M08 flood pumps, M09 all off
(machine pumps by kind). Mill: T preselects (MODE `tool_preselect`; above the magazine capacity
TW0013), M06 changes (TOOL_CHANGE with the core ToolTable slot; T not in the project: warning).
M00 stops; M01 stops when the operator's optional stop is on; M02/M30 end the program, count a part
(#3901, `part_count_codes`) and wait for the next CYCLE START (the main program then restarts; modal
state is kept, comp/cycle/modal call/rotation/scaling/mirror/polar are cancelled as by RESET); M99
in the main program loops (STOP `M99-end`, then from the start or N of M99 P). M29 marks rigid
tapping for the cycle provider. Builder M codes come from `physics.controller.m_codes`
(`{M60: {kind: pallet}, M10: {kind: chuck, action: clamp}, M13: {kind: spindle, spindle: live,
direction: cw}, M57: {kind: output, output: 7, on: true}, M77: {kind: ignore, note: ...}}`) and
builder macro calls from `macro_call_codes` (`{G100: 9010, M06: 9001, T: 9000}`: G codes and M codes
9020-9029 call macros with the block's words as arguments, other M codes and T call subprograms, T
with #149 = the T value; inside the called program the code acts normally; FANUC parameters
6050-6089, 6001#5).

### Custom macro B

Variables (program units: mm/inch by G20/G21, deg, diameter on diameter axes):

| variables | meaning |
|---|---|
| #0 | always `<vacant>`, read only (write: PS0116) |
| #1-#33 | locals of the current macro level (main program = level 0; G65/G66/macro-call codes open a level; M98 does not) |
| #100-#199, #500-#999 | commons (per path; `shared_common_100/500` share the first n between paths) |
| #1000-#1035, #1100-#1135 | interface signals: inputs from `interface_inputs`, else TW0007; outputs recorded |
| #2001-#2200 / #2201-#2400 / #2401-#2600 / #2601-#2800 | memory C, offsets 1-200: H wear / H geometry / D wear / D geometry |
| #10001- / #11001- / #12001- / #13001- (999 each) | the same for offsets 1-999 |
| T series #2001-#2064 / #2101 / #2201 / #2301 / #2701 / #2801 / #2901 / #2401 / #2451 | X wear, Z wear, R wear, tip T, X geometry, Z geometry, R geometry, Y wear, Y geometry |
| #3000 | `#3000 = n (TEXT)`: alarm 3000 + n with the text (n 0-200), the path stops |
| #3001, #3002 | ms clock and hour clock of the simulated time (writable: they continue from the value set) |
| #3003, #3004 | single-block suppression (bit 0), feed hold / feed override / exact stop inhibit (bits 0-2, in `modal`) |
| #3006 | `#3006 = n (TEXT)`: stop with message (MESSAGE), CYCLE START continues |
| #3007 | setting mirror image: 0 (not modelled as a switch) |
| #3011, #3012 | date YYYYMMDD, time HHMMSS from `clock_start` plus the simulated time (deterministic) |
| #3901, #3902 | parts machined (counted by M02/M30), parts required |
| #4001-#4030 | modal G code of each group (G01 -> 1, G54.1 -> 54.1) |
| #4102-#4130 | B D E F H M N O S T P of the preceding block (F/S/T/H/D modal; #4130 = G54.1 P) |
| #5001-#5020 | block end point, program frame (no buffer stop) |
| #5021-#5040, #5041-#5060 | current position, machine / workpiece (encoder = commanded - following error): buffer stop |
| #5061-#5080 | skip position (program frame of the latch; the end point when G31 did not trigger) |
| #5081-#5100 | tool offset in effect per axis |
| #5101-#5120 | servo deviation (commanded - actual): buffer stop |
| #5201-#5220, #5221-#5340 | EXT and G54-G59 work offsets (20 per system) |
| #7001-#7960, #14001-#19980 | G54.1 P1-P48 and P1-P300 |

Not modelled (TW0006): #4201-#4330 (modal data of the block being executed). Other numbers: PS0115.

`<vacant>`: a lone variable keeps it (`#2 = #1`), any operation treats it as 0 (`#1 * 5`, `#1 + #1`,
`-#1` in a statement); EQ/NE distinguish vacant from 0, GT/LT/GE/LE treat it as 0; an address whose
value is a vacant lone variable (or its negation) is ignored (`G00 X#1` moves nothing in X).

Expressions: brackets `[ ]` (5 levels, PS0118); priority functions > `* / AND MOD` > `+ - OR XOR` >
`EQ NE GT LT GE LE` (so `[#1 GT 0] AND [#2 LT 5]` needs the brackets); SIN COS TAN in degrees; ASIN
(270..90 deg, NAT = 0; -90..90 with `nat: true`), ACOS (0..180), ATAN[a]/[b] or ATAN[a, b]
(0..360 deg, -180..180 with NAT), one-argument ATAN; SQRT, LN (argument <= 0, ASIN/ACOS outside
[-1, 1], SQRT < 0, BIN of a non-BCD value: PS0119), EXP, POW[a, b], ABS, ROUND (half away from zero;
to the least input increment inside an NC address), FIX (toward zero), FUP (away from zero), BIN/BCD
(32-bit), ADP[#n] (the argument as if passed with a decimal point), MOD (operands rounded, sign of
the dividend); AND OR XOR bit by bit on 32-bit integers; division by zero PS0112; overflow PS0111.
Comparisons are exact (no tolerance).

Control flow: `IF [c] GOTO n`, `IF [c] THEN #i = e`, `GOTO n` (search forward, then from the top;
cost per block searched; n outside 0-99999: PS0128; not found: PS0078), `WHILE [c] DO m` ... `END m`
and `DO m` ... `END m` (m 1-3, three levels, the same m not nested; missing END: PS0124; END without
its DO or a wrong m: PS0126); a GOTO out of a loop drops it. Calls: `M98 Pnnnn Lr` (or the 8-digit
P = repetitions x 10000 + program), `M99` (return; `M99 Pn` to Nn of the caller; at the end of a
repetition back to the subprogram start), `G65 Pp Lr <arguments>` (argument specification I: A B C I
J K D E F H M Q R S T U V W X Y Z = #1 #2 #3 #4 #5 #6 #7 #8 #9 #11 #13 #17-#26; specification II:
repeated I J K sets = #4-#33; the later of two assignments of one variable wins; a value without a
decimal point is converted by its address' rule (X10 -> 0.010); G in the arguments: PS0129),
`G66 Pp Lr <arguments>` (the macro is called after every block that moves, at the level that
commanded it, not for the macro's own moves; G67 cancels; 4 nested: PS0122). Nesting: 5 macro levels,
15 call levels including macros (PS0077) - `macro_nesting_max`/`sub_nesting_max`. Missing programs:
PS0076. A program that ends without M02/M30/M99: PS5010. `DPRNT[text #v[ab]]` writes to
`PathState.printed` (a = integer digits, b = decimals; `*` prints a space); POPEN/PCLOS are recorded;
BPRNT, SETVN and variable names are not supported (TW0012 / PS0125).

### Parameters (db/models/controller-fanuc.yaml)

| name | value | stands for | source |
|---|---|---|---|
| arc_tolerance_m | 0.02 mm | 3410 arc radius error limit | est |
| macro_nesting_levels / sub_nesting_levels | 5 / 15 | macro and call nesting | est |
| max_m_codes_per_block | 3 | 3404#7 M3B | est |
| first_wait_code / last_wait_code | 100 / 999 | 8110 / 8111 | est |
| macro_alarm_base / max_macro_alarm | 3000 / 200 | #3000 alarm numbers | fanuc-cnc-functions-catalog-2024 p. 504 |
| g37_feed_m_s, g37_approach_m, g37_tolerance_m | 100 mm/min, 5 mm, 2 mm | 6241, 6251, 6254 | est |
| same_group_alarm | false | last G code of a group wins | est |
| feed_mode_change_clears_f | true | F needed again after G93/G94/G95 | est (chosen so a per-minute F is never read as per-revolution) |
| g92_cancels_g52, lathe_offsets_in_diameter | true | G92 and G52; 5004#1 ORC = 0 | est |

### Validity and error

The interpretation is exact for the documented semantics (tests: tests/unit/controller). Where the
0i-F manual could not be read (alarm texts, some parameter defaults, the F scaling without decimal
point, the M-code timing within a block, the order of G92 and G52 cancellation) the choice is the
most common FANUC behaviour and is listed above; positions are exact to the least increment.

### References

FANUC CNC Functions catalogue (fanuc-cnc-functions-catalog-2024); Smid, *Fanuc CNC Custom
Macros* (smid-2005-fanuc-macros) and *CNC Programming Handbook* (smid-2008-cnc-handbook); ISO
6983-1:2009 (iso-6983-1-2009); db/controllers/fanuc-0i-mf.yaml and fanuc-0i-tf.yaml.

## Haas NGC

`cnctwin/controller/dialects/haas.py` (handler `HaasHandler` on top of the FANUC handler, variables
`HaasVars`, tables, alarm numbers, the TCPC/DWO stage `TableTcp`), `haas_cycles.py` (G12/G13, G47,
G150), `haas_font.py`, `contour.py`, `kernels.py`. Mill and lathe NGC share the dialect; the variant
follows the machine type as for FANUC. Defaults: `db/models/controller-haas.yaml`. The controller
database `db/controllers/haas-ngc.yaml` (read from the Haas NGC operator's manuals) gates the G and M
codes (mill lists, or its `lathe` lists) and gives the alarm texts it has (312, 314, 332, 370, 467,
1000-1999 ...); the others are built-in (`source: builtin`).

### Differences from FANUC

* One M code per block (alarm 339 "Multiple Codes"); every M code acts at the end of its block.
* Two G codes of one group, or one address twice, in a block: alarm 339 (FANUC uses the last).
* F is as written (no least-increment scaling without a decimal point, Setting 77 default); S may
  carry a decimal point (rounded).
* G04 P: with a decimal point seconds, without milliseconds (P2.5 = P2500); G04 without P: 326.
* G68 takes the centre in A (first plane axis) and B (second) with R the angle.
* G100/G101 cancel/enable mirror image (as G50.1/G51.1); G103 Pn limits the look-ahead to n blocks
  (0-15; G103 / P0 = machine look-ahead), which the executor's buffer capacity follows.
* Work offsets G54-G59, G110-G129 (= G154 P1-P20), G154 P1-P99; G52 is the FANUC-style local shift
  (Setting 33 = FANUC) and is also the offset entry #5201-#5206 / G10 L2 P0; G10 L20 P1-P99.
* G10 L10/L11/L12/L13: length geometry / length wear / diameter-radius geometry / wear
  (`offsets_diameter: true` = Setting 40 diameter; default radius).
* G187 P1/P2/P3 [E]: smoothness level and maximum corner rounding E (haas-ngc-mill-om-2020 p. 393). Without
  an E the corner rounding is Setting 85 (p. 456): the machine smoothing table's entry for "G187 Pn" (the db
  machines give Setting 85 for every level), and G187 Pn leaves an active E alone; G187 E leaves P alone.
  G187 alone cancels E and returns to the defaults: Setting 191 (medium) and Setting 85, the machine file's
  "G187 P2" entry (on a mill without a smoothing table the controller database's 0.0250 in factory
  default), never 0. The same defaults hold at power-on and after RESET, M30 and M02. Setting 85 and E are the
  largest radius of a rounded corner ("the control keeps the radii of corners no bigger than" the value, p. 456):
  the Smoothing's `tolerance_kind` is `radius` and motion rounds a corner only as far as the axes need at the feed,
  up to that radius, slowing the corner where the feed needs more (docs/models/motion.md 2.2; until 2026-11-09 the
  value was taken as the deviation from the corner: a 1.53 mm radius at a square corner for the 0.635 mm default).
* #3000 = n: alarm 1000 + n (n 1-999) with the comment as its text.
* M97 Pn Lr: local subprogram from block Nn of the same program (337 when Nn is missing); M98 P with
  5-digit program numbers; M99 P n.
* M codes by device (TW0010 when the machine definition lacks the device): M07 shower, M08/M09
  flood (M09 leaves TSC alone), M88/M89 through-spindle coolant, M83/M84 air gun (lathe M12/M13),
  M34/M35 programmable coolant nozzle +-1 position (the nozzle moves to the tool's stored position,
  #3401+T or the project's `nozzle_positions`, at every tool change), M31/M33 chip conveyor,
  M50 pallet change, M36 part ready, M80/M81 automatic door (lathe M85/M86), M10-M13 4th/5th-axis
  brakes, M51-M56/M61-M66/M59/M69 outputs, M16 = M06, M41/M42 only with a gearbox (else no effect,
  noted, as Haas), M78/M79 alarm if the last G31 did / did not trigger (378/379), lathe M10/M11 and
  M110/M111 chucks, M21/M22 tailstock, M146/M147 steady rest, M36/M37 parts catcher, M133-M135 live
  tool and M143-M145 sub spindle (P = rpm); M109 (operator input), M21-M29/M121-M129 (wait for M-FIN),
  M95/M96/M46/M48 (inputs) are not modelled (TW0011/TW0007/TW0012). M199 with a pallet changer is a
  pallet change, otherwise program end.
* G41 in a G150 block is the pocket's finishing side, not modal compensation.
* `,C` chamfer and `,R` corner rounding between two G01 blocks (haas-ngc-mill-om-2020 p. 309-310);
  on a G02/G03 block, or followed by one: TW0031.
* Lathes: D in a G71/G72 block is the depth of cut (radius), in G73 the number of passes, in G74/G75 the
  tool clearance, in G76 the first pass depth (haas-ngc-lathe-om-2020): cycle data, never an offset
  number (offsets come from the T word). The face drilling cycles G81 G82 G83 G84 G85 G86 G89 and
  G184 (reverse tapping, the control turns the spindle CCW itself) drill along Z, the radial cycles
  G241 G242 G243 G245 G246 G249 along X; R is the position of the R plane (a diameter for the radial
  ones), without R the cycle starts at the starting plane, to which it returns.
* Mill G77 back bore and the G70/G71/G72 bolt hole patterns (see "Drilling cycles").

### Variables (NGC numbering)

#1-#33 locals; #10000-#10999 general purpose with the legacy aliases #100-#699 and #800-#999
(= #10100-..., #700-#749 reserved: 331); #1000-#1063 / #11000-#11063 inputs (`interface_inputs`);
mill tool offsets #2001-#2200 length geometry, #2201-#2400 length wear, #2401-#2600 diameter/radius
geometry, #2601-#2800 wear; lathe #2001-#2050 X shift, #2051 Y shift, #2101 Z shift, #2201 nose
radius, #2301 tip direction, #2701 X wear, #2751 Y wear, #2801 Z wear, #2901 nose radius wear;
#3000/#3001/#3002/#3003/#3004/#3006 as FANUC; #3011/#3012 date/time; #3020/#3021 power-on and cycle
timers (simulated time), #3023/#3024/#3025 present/last/previous part times; #3026 tool in spindle;
#3027 commanded spindle rpm; #3030 single block (rw), #3032 block delete, #3033 optional stop;
#3401-#3600 programmable coolant positions; #3901/#3902 M30 counters (both count M30); #4001-#4021
modal groups; #4101-#4126 address data A-Z of the last block (F S T H D modal); #5001-#5006 last
target, #5021-#5026 machine and #5041-#5046 work positions (buffer stop), #5061-#5069 skip positions,
#5081-#5086 tool length compensation; #5201-#5206 G52, #5221-#5326 G54-G59, #7001-#7386 G110-G129,
#14001-#15966 G154 P1-P99; #6198 = 1000000. Not modelled (TW0006): #1064-#1098 loads and analog
inputs, #1601-#2000 flutes/vibration, #3022, #3028, #3034, #3201-#3400, #5401-#6000 tool timers and
monitors, #8500-#8564 ATM, #20000-#39999 settings and parameters.

### G12 / G13 circular pocket

At the current X Y (centre c), G12 clockwise, G13 counter-clockwise: radii R_k = I, I + Q, ... up to
K (K needs Q > 0: 305), path radii rho_k = R_k - r_tool (D offset; rho_1 <= 0: 369). Each loop: feed
to Z (absolute in G90; in G91 Z is the increment per loop), a semicircle of radius rho_1/2 from the
centre to (c_x + rho_1, c_y) that enters tangentially, the full circle, then for each further radius
a semicircle of radius (rho_k - rho_(k-1))/2 to (c_x + rho_k, c_y) and the full circle, and a feed
back to the centre; L loops. The transition semicircles reverse direction at their start (Haas does
not publish the exact path; area, depth, radii and cutting length are those of the cycle).

### G47 engraving

P0: the comment text; P1: serial number from #599 with as many digits as `#` in the comment, then
#599 + 1; P32-P126: one character. From the lower-left start X Y, text height J (default 1 inch),
angle I about the start, return plane R and depth Z (canned-cycle rule in G91), feed F, plunge E.
Each stroke: rapid at R to its start, feed to Z at E, engrave at F, rapid back to R (connected
strokes stay down). Font: the twin's single-stroke font, cap height J, character advance 1.0 x J
(`haas_font.py`; Haas's own glyphs are not published, so positions within a character differ from
the machine's by up to the character size, the engraved length and depth are representative);
characters outside it: TW0014.

### G150 general pocket

1. Contour: subprogram P read from the start point X Y with its own G90/G91 and G01/G02/G03 (R or
   I J); macro statements, M codes other than M99, other G codes or other axes: 370. Closed when its
   last point is the start point (contour = all moves) or the end of its first move (contour = the
   moves after the first: the Haas example form); otherwise 370. The start point must lie inside
   the contour (370).
2. Orientation by the signed area (arcs included: sum of 1/2 (x0 y1 - x1 y0) + r^2/2 (phi - sin phi));
   the roughing region is the contour eroded by r_tool + K: on each raster line, the even-odd inside
   intervals minus the union of the capsule intervals of every boundary segment (kernel
   `raster_intervals`, exact for the polyline; arcs as chords with sagitta <= 1 um, so the region is
   accurate to 1 um).
3. Raster: lines parallel to X stepping by J (or parallel to Y stepping by I), from y_min + d to
   y_max - d, evenly spaced with spacing <= the stepover; alternate directions (zig-zag); passes are
   ordered by following the raster and preferring the next pass reachable by a straight link that
   keeps the clearance (kernel `segment_clearance`), otherwise retract to R, rapid and feed down.
4. Depths from R down by Q (in G91 R is relative to the current Z and Z to R), the last at Z.
5. Finishing at each depth: the exact inward offset of the contour by r_tool (lines shifted, arcs
   r -+ r_tool, convex corners trimmed to the intersection, concave corners rounded with an arc of
   radius r_tool; vanishing elements removed; a remaining self-intersection: 370), traversed
   counter-clockwise for G41 (climb with M03), clockwise for G42, entered from the last raster pass
   (link rule above); then retract to R and rapid to the start point.
The finishing pass is made at every depth even with K = 0 (it cleans the scallops the zig-zag ends
leave); the raster leaves at most the stepover-dependent cusp between passes. The zig-zag link rule
and pass order are the twin's; Haas does not publish them.

### G254 / G255 dynamic work offset, G234 tool centre point control

On machines whose rotary axes are all in the workpiece chain (UMC, EC-400 B table; otherwise TW0003):
the work offset is the part zero in machine coordinates at rotary zero, W. A program point p (work
frame, after G268) is the table point p0 = W + p; the table moves it to p = T(B, C) p0, where
T = product over the rotary links of `Trans(a) Rot(d, angle) Trans(-a)` with the axis points a
(machine coordinates at zero, from the kinematic chains: for the UMC-750 they reproduce
`kinematics.mrzp_mm`) and directions d of the machine file; the commanded linear axes are
p + L t (L the H tool length, t the machine tool axis). The command is in `Space.TIP` with
`tip_target_m = p`, `tip_axis = T^-1 t` (the tool axis in the part frame), `modal["work_origin_M0_m"]
= W` and `tip_interpolation` "joint" (G254: end points only) or "tip" (G234: the tip follows the
path while B/C move; G234 takes H like G43 and G49 cancels it). Reads (#5041, skip) invert the same
transform. Error: exact for the rigid kinematic chain of the machine file (the machine's own MRZP
settings may differ from the drawing values the file holds).

### G268 / G269 feature coordinate system

`G268 X Y Z I J K Q`: origin (X Y Z, current work coordinates) and rotation about X, Y, Z by I, J, K
deg applied intrinsically in the order of the digits of Q (default Q321: first Z, then the new Y, then
the new X); program points are mapped p_work = o + M p, arcs keep their sense in the feature plane.
G269 cancels. With G254 the table rotation is applied after it (above). The default Q is an
assumption (the manual text read for the database gives no default).

### Parameters (db/models/controller-haas.yaml)

| name | value | source |
|---|---|---|
| arc_tolerance_m | 0.0254 mm | est |
| macro_nesting_levels / sub_nesting_levels | 9 / 9 | est |
| max_m_codes_per_block | 1 | est (Haas manuals: one M code per block) |
| macro_alarm_base / max_macro_alarm | 1000 / 999 | haas-ngc-mill-om-2020 p. 266 |
| g37_* | 100 mm/min, 5 mm, 2 mm | est |
| same_group_alarm, feed_mode_change_clears_f, g92_cancels_g52 | true | est |
| offsets_diameter | false | est (Setting 40 radius) |

Built-in alarm numbers (used when the database lacks them): 302 R arc, 303 plane axes, 304 I J K
radius, 305 Q, 307 nesting, 308 tool offset, 310 G code, 311 unknown code (lexical errors and
unused addresses), 312 program end, 313 no P, 314 subprogram not in memory, 315 invalid P, 320 feed,
324/326 dwell, 329 M code, 331 range error (macro value errors), 337 GOTO/P line, 338 I J K and X Y Z,
339 multiple codes, 344 plane change in comp, 369 tool too big, 370 pocket definition, 376 comp in
skip, 378/379 skip found/not found, 467 arc with per-axis scaling, 1000 + n program alarms.

### References

Haas NGC mill and lathe operator's manuals (haas-ngc-mill-om-2020, haas-ngc-lathe-om-2020)
as read into db/controllers/haas-ngc.yaml; Smid 2005/2008 for the FANUC-compatible parts.

## Cycles and compensation

Packages `cnctwin.controller.cycles`, `comp`, `probing`, `toollife`, `restart`, `tcp` and the macro library
`cnctwin.controller.macrolib` extend the core through its hooks (FANUC semantics; Haas where it differs).
Everything runs inside the state machine: cycle expansions are `Step` lists in `PathState.pending`, the
compensation filter state is JSON in `PathState.comp_state`, probing / tool life / restart state lives in
`PathState.sys`, so a checkpoint may fall inside a cycle, a compensated contour or a probe macro. Parameters:
`db/models/controller-cycles.yaml` (each entry names the FANUC/Haas parameter it stands for and its
source; `est` values are the twin's assumption), overridden per project by
`physics.controller.cycle_parameters` (keys with their unit suffix, e.g. `g73_retract_mm: 0.2`). Sources:
`db/bibliography/models-controller-cycles.yaml` (FANUC 30i-B Plus manuals for machining centres
`fanuc-30ibplus-om-mc-2019`, lathes `fanuc-30ibplus-om-lathe-2019`, parameters `fanuc-30ibplus-param-2019`;
the common manual `fanuc-30ibplus-om-2019`; `murray-li-sastry-1994`). No randomness (no RNG stream); no
bus fields are read or written (the physics talks to these functions only through `Feedback`).

### Drilling cycles (G73-G89, G84.2/G84.3; T series G83-G89; Haas)

Every hole is FANUC's six operations (mc p. 45): positioning, rapid to R, machining, bottom operation,
retraction, rapid to the initial level. Along the drilling axis: R level = R (G90) or initial + R (G91,
T series); Z level = Z (G90) or R + Z (G91); return level = initial (G98, T series, G87) or R (G99). With
q = |Q|, d73 = parameter 5114 (`g73_retract_m`), d83 = 5115 (`g83_clearance_m`):

| cycle | sequence |
|---|---|
| G81 / G82 | feed to Z (G82 dwell P), rapid out |
| G73 | pecks of q, rapid retract d73 after each |
| G83 | pecks of q, rapid to R, rapid back to d83 above the last depth (Q 0/missing: PS0045) |
| G85 / G89 | feed in, (G89 dwell), feed out to R |
| G86 / G88 | M05 at Z; G86 rapid out, G88 dwell + STOP (manual retract), spindle restarted |
| G76 / G87 | dwell, M05 (5101#7), M19, shift q (5148 direction), rapid out / back boring from below |
| G84 / G74 | floating: feed in (override off), dwell, spindle reversed, feed out |
| rigid (M29, G84.2/G84.3, Haas) | spindle as a servo axis: lead = F (G95) or F/S (G94); LINEAR moves with `Feed(PER_REV, lead, override off)`, modal `rigid_tap`, `tap {lead_m, direction, phase}`; motion reverses the spindle on the outward move; peck tapping Q with 5213; extraction override 5211 |

K/L repeat a hole at the next incremental position; Haas G73/G83 take I/J/K peck schedules, G76/G87
Haas variants. The Haas cycle data are modal until G80 (haas-ngc-mill-om-2020 canned cycles): the I/J/K
peck schedule (or Q) and the G76/G77 shift by Q (Setting 27) or I/J stay with the cycle for the next
holes; the block that gives one of them chooses. Alarms: PS0200-PS0206 rigid tapping, PS0549 plane
change in the cycle mode.

Haas NGC additions:

* **G77 back bore** (mill, haas-ngc-mill-om-2020 p. 353-355): FANUC's G87 sequence with the Haas shift:
  M05 + M19 and shift at the initial level, rapid down to R (past the bottom of the part), shift back,
  spindle on, feed up to Z, M19 and shift, rapid to the initial level (always: the control assumes
  G98), shift back, spindle on. The spindle runs in the last commanded direction, M03 when the program
  has commanded none, not at all after M05 (the manual's "no-spin" cycle).
* **G70 / G71 / G72 bolt hole patterns** (mill, p. 348-349; `cycles.patterns.BoltPattern`): the active
  canned cycle (G73, G74, G76, G77, G81-G89, with its modal data and G98/G99 return) at every
  position of the pattern, in the active plane, about the tool's position: G70 I radius, J start angle
  (degrees CCW from 3 o'clock), L holes evenly on the circle; G71 I, J, K angular spacing, L; G72 I
  spacing, J angle of the line, L holes. The first G72 hole lies at the tool's position
  (`haas_g72_first_hole_at_start`, an assumption: the manual does not say). No active cycle: TW0037;
  I or L missing, L < 1 or axis words in the block: TW0030.
* **Lathe** face cycles G81 G82 G83 G84 G85 G86 G89 (mill sequences along Z) and G184 (rigid, CCW),
  radial cycles G241 G242 G243 G245 G246 G249 (the G81 G82 G83 G85 G86 G89 sequences along X); R is
  absolute (a diameter on X), the return is the starting plane. Live-tooling taps (G95, G186, G195,
  G196), broaching (G156) and the mill G153/G174/G184 non-vertical and five-axis cycles have no provider
  (TW0001).

### Lathe cycles (system A) and multiple repetitive cycles

* G90 / G92 / G94 box cycles (lathe 4.1): modal X(U) Z(W) R; a block without a move repeats the cycle.
  G92 thread: lead F along the major axis, start angle Q, chamfer over `thread_chamfer_leads` x L at
  `thread_chamfer_angle_rad` (5130/5131), override off. Under tool nose compensation the four moves are
  compensated with start-up from the start point and cancel on the return (4.1.5).
* G32 threading: `thread {lead_m, lead_major_m, major_axis, start_angle_rad}` on the item, synchronised
  (PER_REV, override off).
* G96/G97 with the G50 clamp: SPINDLE commands carry `css_m_s` and `max_rad_s`.
* G71 / G72 stock removal (lathe 4.2.1-4.2.4): the figure N(ns)..N(nf) shifted by the finishing
  allowances (U/2 radius, W). Type I: levels of depth dd from A toward A', each level: infeed, cut to
  the figure, 45 deg escape by e, rapid back; then the rough pass along the figure. Type II (pockets):
  regions cut depth-first, nearest first, following entry and exit walls. G73 pattern repeat: d passes
  from offset (di + du/2, dk + dw) down to (du/2, dw). G70 runs ns..nf as written and returns to A.
  Errors: PS0061-PS0069, PS0322, PS0325, PS0329; levels are found on arc chords of at most 1 um sagitta.
* G74 / G75 peck grooving: pecks dk, return e (5139), relief at the bottom, columns di.
* G76 multiple threading (two blocks P m r a, Q dmin, R d; X Z R(i) P(k) Q(dd) F(L)): pass n at depth
  `d_n = dd * sqrt(n)` from the crest (constant cutting area, lathe p. 71), at least dmin deeper than the
  previous pass, until k - d; then m finishing passes at k. One-edge infeed: the start shifts along the
  flank by `(d_n - k) tan(a/2)`. Haas G76: K D A I P1-P4 Q.

### Cutter radius compensation (G41/G42/G40, FANUC type C) and tool nose radius compensation

A block-stream filter (`comp.cutter.CompFilter`) holds up to N blocks (`crc_lookahead_blocks`, parameter
19625, 3 by default; mc p. 211) and releases compensated items. Elements live in the plane's (u, v) frame
of the program, so mirror/rotation keep the side. Offsets: a line moves by d along its left normal
(d > 0 for G41, the D sign swaps the side); an arc's radius becomes `R - d sign(sweep)` (<= 0:
interference). Corners at the programmed point P with the turn angle phi toward the tool:

* inner (phi > 0): the intersection of both offset elements (none: PS0033);
* outer, 90 <= alpha < 180 deg: the intersection; outer alpha < 90: two vectors, each offset path
  extended by |d| along its tangent (`crc_outer_corner: linear`, 19607#2 CCC = 0); `circular`: an arc
  of radius |d| about P;
* start-up types A/B/C and cancel types A/B/C (5003 SUP/SUV, `crc_startup_type`): A ends on the normal of
  the next block, B adds the corner vectors, C shifts a start-up block without motion; arcs in start-up
  or cancel: PS0034; at most N - 2 blocks without motion in the plane keep the corner, otherwise the
  vector is normal (mc p. 211); D changes take effect at the end of their block; G39 inserts a corner
  arc; I J K in G00/G01 give the IJ type vector; G40 I J K the cancel direction.
* interference check (5008, 19607#5): the direction check (a compensated element reversed against its
  programmed direction by 90 deg or more) and the circular angle check (the compensated arc's angle
  differs by 180 deg or more) on the next block; an interfering vector is erased when the remaining
  path is still valid (vector erasure), otherwise PS0041 before the block that would overcut.
* tool nose (lathes): the commanded point is the imaginary tip `P = C + r t(tip)`, with tip directions
  (first, second plane axis; G18 Z, X): 1 (+1,+1), 2 (-1,+1), 3 (-1,-1), 4 (+1,-1), 5 (+1,0), 6 (0,+1),
  7 (-1,0), 8 (0,-1), 0 and 9 the nose centre (lathe 5.2.2, as tabulated in `smid-2008-cnc-handbook` ch. 30). The cycles G70-G73,
  G90/G94 compensate their own paths (`cycles.nose`).
* `,C` chamfer / `,R` corner rounding (mc 5.5): points at distance c on both elements / the arc of radius
  r tangent to both; PS0050-PS0055.

Error bound: compensated points are computed in double precision and rounded to counts (q/2).

### Five-axis (G43.4 / G43.5, G68.2 / G68.3 / G68.4, G53.1, G69; Haas G234 / G254 / G268)

Kinematics from the machine file's chains: `Rel(q) = Wp(q)^-1 Tl(q)` (products of the link transforms);
the linear axes for a tip p with length L along the tool axis a solve `Rel(q) (-L a) = Z_T + p` (3x3
linear solve); the rotary values for a tool direction come from the Paden-Kahan subproblems 1 and 2 on
the product of exponentials (murray-li-sastry-1994 ch. 3), the solution nearest the current rotary values
within travel (TW0022 none, TW0023 outside travel). Rotary values are rounded to counts before the
linear solve, so the tip error is at most half a linear count; the model is exact for the drawing
values of the chains (no kinematic errors). Commands are `Space.TIP` with `tip_target_m`, `tip_axis`,
machine `target`, `modal tip_interpolation` ("tip"; "joint" for the G43.4 start-up block and under
G68.2) and `tcp_type` "vector" (G43.5, I J K normalised; kept until the next vector).
G68.2: Euler Z X' Z'' (`R = Rz(I) Rx(J) Rz(K)`), P1 roll-pitch-yaw in the Q order, P2 three points
(I J K all zero: PS5457); G68.3 by the tool axis; G68.4 composes with the active frame; G53.1 turns the
rotary axes so the tool is normal to the plane (PS5458 without G68.2); G69 in compensation: PS5462;
P3/P4 forms: TW0012. Haas G234 (`TableTcp` of the core) equals G43.4, G268 (rotation about X, Y, Z in the
Q order) equals G68.2 with the same rotation, G254 keeps the tip on the work offset.

### Probing and tool setting

* G31 (`probing:G31`, replaces the core's): `G31 [P1-P4] IP F` is one buffer-stopping PROBE command
  `{"skip": P, "stylus": "spindle_probe", "expect": "trigger"}` with `Feed.override_allowed = False`
  (override, dry run and acceleration are disabled in skip moves unless 6200#7 SKF,
  `skip_feed_override`; fanuc-30ibplus-om-2019 p. 133); PS0035 in compensation, PS0005 without an
  axis, PS0011 without F, P outside 1-4: TW0034, P90 / P98 / P99: TW0012. #5061-#5066 are the latched
  encoder positions of the feedback in the work frame (the end point when the signal did not come).
* The skip number of the tool setter (`setter_skip_signal`, P2) makes the move a TOOL_SETTER command
  `{"measure": "skip", "skip", "tool", "offset"}`. A `SetterResult` without latched positions gives the
  contact at machine tool-axis value `z_s + L` (`setter_position_m`); outside the move, or `broken`,
  is no signal. The last result is in `PathState.sys["setter_result"]`.
* G37 / G36 (`SetterMeasure`): the core's measurement (window epsilon 6254, PS0080-PS0083, offset +=
  latched error) plus: `SetterResult.length_m` sets the length geometry to `L - wear` when
  `|L - H| <= epsilon` (otherwise PS0080, a signal outside the window); `broken` without a latch: PS0080.
* Built-in macros (FANUC custom macro B, `macrolib/O98xx.nc`, loaded as library programs; Renishaw-like,
  the Renishaw programs were not used): O9810 protected positioning, O9811 single surface (S work
  offset, T length wear, H tolerance, Q overtravel), O9812 web/pocket (X or Y width, Z for a web),
  O9814 bore/boss (D, Z for a boss; X pair first, the Y pair through the measured X centre gives the
  diameter), O9851 tool length on the setter (G37; a zero offset is first searched on the setter
  signal), O9853 broken-tool check (no contact at the stored length less H: BROKEN TOOL; contact
  above the clearance plane or more than H early: OUT OF TOLERANCE). Stylus ball radius = D offset of the
  probe's tool number (geometry + wear; <= 0: PROBE NOT CALIBRATED). Equations: measured surface =
  skip + s r (s = direction of travel), error = measured - nominal, work offset += error, length wear
  -= error; width = p+ - p-, centre = (p+ + p-)/2. Results #135-#140. Alarms (#3000, base
  `probe_alarm_base` 86): 86 PATH OBSTRUCTED, 87 PROBE OPEN, 88 PROBE NOT CALIBRATED, 89 FORMAT ERROR,
  90 OUT OF TOLERANCE, 91 BROKEN TOOL (the twin's numbering). A skip position within one count of the end
  point counts as no trigger. `ControllerSettings.builtin_macros`: None installs the probe macros with a
  spindle probe and the setter macros with a tool setter of known position; a program number the project
  defines is left out.

### Tool life management

FANUC basic tool life management (fanuc-30ibplus-om-2019 10.5, pp. 265-282): groups from
`physics.controller.tool_life` (`{"groups": [{"group": 1, "life_uses": 50, "tools": [10, 11, {"t": 12,
"h": 12, "d": 32, "used_uses": 3}]}, {"group": 2, "life_min": 30, "tools": [...]}]}`; lathe tools as T codes
`0101`) or G10 L3 / G10 L3 P1 / G10 L3 P2 ... G11 (P L Q, T H D; PS0149-PS0156). Group call `T(6810 +
group)` (`tool_life_ignore_number` 9900: `T99nn`), Haas `T1000`-`T2999`, lathes `T(group)99` /
`T(group)88`. Sister selection searches from the group's current tool toward the last and around; all
expired: PS5317 (`tool_life_all_expired: alarm`) or the last tool (`last`, FANUC's own behaviour with the
TLCH signal). Uses count at M06 (lathe: the T call) once per group per run (Haas: every change); time
counts the LINEAR/ARC commands from their retirement ticks while a managed tool is in the spindle
(error: when several commands retire in one call the interval is shared equally; feed hold inside a
cutting move is counted). H99 / D99 select the registered offsets (PS0154 without a group tool).
`Feedback.tool_life_expired` (bus `wear.expired` via the physics) marks tools expired in every group.

### Program restart and the dangerous-restart analysis

`restart.restart(controller, n=|line=, frequency=1, kind="P"|"Q", path=0)` (fanuc-30ibplus-om-2019 4.9,
pp. 1206-1208): the path is interpreted from its head without output (`ex.search`), then the M/S/T state
is restored by `restart_output` (7300 MOU/MOA: `none`, `last`, `all`) and the tool approaches the restart
point axis by axis in `restart_axis_order` (7310) at `restart_feed_m_s` (dry-run feed) in machine
coordinates; the program continues at the target. P: from where the tool is; Q: after a RESET from the
reference position. PS0060 (Haas TW0036) when the block is not reached; TW0024 inside rigid tapping.
`restart.restart_analysis(setup, path=0, lines=None)` returns `[{"line", "program", "n", "issues":
[{"group", "value", "fresh", "severity", "first_move_line", "message"}]}]` for every tool change (or the
given lines): each restart-relevant group (motion, plane, distance, feed mode, units, cutter / length
compensation, cycle, return level, scaling, rotation / tilted plane, mirror, polar, CSS, work offset,
TCP, G52 shift, macro modal call, spindle, coolant, tool) whose value when the program reaches the line
differs from a power-on start there and that the section does not restate up to its first move (a
lexical test) is an issue; `cnctwin.sanity.rules` reads the records (`restart_analysis=`).

### Core hooks added (additive)

`hooks.GCODES` / `register_gcode(code, spec, variant=, dialect=)` (G codes merged into the dialect
tables), `GCODE_HANDLERS` (GSpec kind `ext`: `execute(ex, code, spec, bw) -> moved`), `MOTION_WORDS`
(`motion_words(ex, code, bw) -> modal entries`), `LIBRARIES` (`sources(machine, dialect, settings,
programs) -> [ProgramSource]`), `DATA_MODES` (`"G10L3"`: `begin/block/end`), cycle providers'
`cancel(ex, code)`, `RESUMERS` `after_block(ex, data)` (`PathState.sys["after_block"]`), tool-life
`t_code(ex, number)`, `offset_code(ex, kind, number)`, `retired(ex, commands, tick)`, `reset(ex)`;
`alarms.register_alarm_codes(dialect, key_codes, texts)`; settings `cycle_parameters`, `tool_life`,
`builtin_macros`.

### Limitations

`,A` angle chamfers (TW0012) and `,D`; the small-hole peck cycle and CRC interference avoidance
(19607#5 CAV = 1); G68.2 P3/P4; Haas G35 / G36 / G136 probing cycles and M75 (TW0004 / TW0012); multi-step
skip signal selection beyond the setter number; FANUC tool management (10.2), life count override,
remaining-life notice and the restart M code; time counting includes feed holds; restart analysis is
lexical; the setter is a plane on the tool axis (no diameter measurement).

## Siemens 840D sl

`cnctwin/controller/dialects/siemens840d/` registers the dialect `siemens840d` (the machine file's
`controller.dialect`). Modules: `syntax.py` (program reader: files -> programs -> immutable blocks with
the control-structure index), `values.py` (expressions, R parameters, local user data, system variables,
frame values), `handler.py` (`SiemensHandler`, the block semantics), `cycles.py` (drilling, hole
patterns, pockets and slots, CYCLE800, CYCLE832, the step resumer `siemens840d:action`), `milling.py`
(pocket, slot and helix toolpaths, shared with the Heidenhain dialects), `turning.py` (CYCLE95, CYCLE97,
CYCLE93), `tables.py` (G functions and groups, keywords, alarm numbers with built-in texts, the word
list), `kit.py` (runtime shared with the Heidenhain dialects: `NativeHandler`, alarm-text binding, the
frame -> core-stage mapping, two-axis table kinematics, the TCP stages). Defaults are in
`db/models/controller-siemens840d.yaml` (plus the block-time model `controller-block-time`); each can be
set under `physics.controller`. Alarm texts come from `db/controllers/siemens-840d-sl.yaml` when it lists
the number (`source: db:...`), otherwise from built-in copies (`source: builtin`). The variant follows
the machine type: power-on G0 G17 G40 G500 G71 G90 G94 DIAMOF (turning machines G18 G95 DIAMON), G601
CFC NORM G450 BRISK FFWOF COMPOF WALIMON, G60 or G64 by the machine's corner default. No random
numbers are used and nothing is written to the bus: the dialect only produces Commands. All dialect
state is in `PathState` (`sys`, `variables["R"]`) and checkpoints with it.

### Program format

* One program per file, named after the file (`PART.MPF` -> `PART`, `SUB.SPF` -> `SUB`); archive form
  with `%_N_NAME_MPF` / `%_N_NAME_SPF` header lines; a library file may hold several `PROC NAME`
  programs (a `PROC` line after other blocks starts the next program). `;$PATH` lines are skipped.
* One line is one block: `[/n] [Nnnn | :nnn] [LABEL:] items [; comment]`. Labels start with two letters
  or underscores (`MA1:`) and are unique per program (12450). Text is case-insensitive outside strings.
  Skip levels `/0`-`/8` follow the operator's block-delete switches.
* Items: G functions (`G1`, `CIP`, `G641`), addresses with a number or a value (`X10.5`, `S1=1000`,
  `X=R1+2`, `X=AC(10)`, `CR=-5`), assignments (`R1=5`, `NAME=..`, `R[R2]=..`, `$P_UIFR[1]=..`),
  keywords (`TRANS`, `STOPRE`, `TRAORI`), calls (`CYCLE81(10,0,2,-20)`, `SUB1 P3`, `SUB2(1,,R1)`,
  `MCALL CYCLE81(..)`, `CALL "SUB"`, `MSG("..")`) and jumps (`GOTOF LABEL`). IF / ELSE / ENDIF, WHILE,
  FOR, LOOP, REPEAT / UNTIL, REPEAT label, REPEATB, DEF, PROC and EXTERN are whole-block statements.
* Reading never fails. A block that cannot be read carries its alarm and raises it when control reaches
  it: 12080 syntax error with the text from the error on, 12550 unknown name, 14500 DEF / PROC after
  the first NC instruction, 12640 / 12641 control-structure nesting / depth (`structure_nesting_levels`),
  12450 label twice. Reaching the end of a program without M30, M2, M17 or RET is 14000.

### Order of execution in a block

1. The items are evaluated left to right against an overlay (an assignment is visible to the items
   after it) and committed when the block has been read completely. A main-run variable (`$AA_IW`,
   `$AA_IM`) stops the block buffer first; the core runs the block again with the live positions.
2. Checks: an address twice 12000, two G functions of one group 12060, an unknown G function 12470,
   a function the build does not model TW0012 (`tables.UNMODELLED`: splines, POLY, TRANSMIT, TRACYL,
   CUT3D..., ORIWKS / ORIMKS / ORIAXES / ORIVECT, CASE, POS / POSA, ...).
3. Modal G functions and keywords; F; S, LIMS, T, D, SPOS and the M functions that start something
   (M3 M4 M7 M8; M6 before them, so `T1 M6 S1000 M3` changes the tool first); a frame instruction, the
   pole, G4 / G74 / G75 or the motion of group 1 (with G40 / G41 / G42 and CHF= / CHR= / RND= /
   RNDM=); the modal call (MCALL) after a path motion; subprogram and cycle calls; MSG, SETAL, WAITM,
   STOPRE, TRAORI / TRAFOOF; the remaining M functions (M5 M9 M0 M1 M2 M30 M17); a jump.
4. Words left unused: 12090 (an address the block cannot use), 16410 (an axis the channel does not
   have).

### Motion and arcs

G0 / G1 / G2 / G3 / CIP / CT (group 1, modal), G33 thread (lead I / J / K, override off), G331 / G332
rigid tapping, G4 F (s) / S (spindle revolutions), G74 reference point, G75 FP=n fixed point
(`reference_points_mm`), G90 / G91 with the per-word `=AC()` / `=IC()`, `=DC() =ACP() =ACN()` on modulo
rotary axes (a plain G90 value on a modulo axis moves directly, not the shortest way), G110 / G111 /
G112 pole (last point, work zero, last pole) and the polar end point `AP=` (deg) / `RP=`, `TURN=` extra
full turns of a helix.

Arcs in the active plane (axes a, b; start s, end e, centre c, direction d = -1 for G2, +1 for G3):

* centre `I J K`: incremental from s by default, `=AC()` absolute;
* `CR=r`: c on the perpendicular bisector of s and e at sqrt(r^2 - (|e - s| / 2)^2) from the midpoint,
  on the side that gives a sweep <= 180 deg for r > 0 and > 180 deg for r < 0; r more than the
  tolerance below the half chord: 14095 (within it: the half circle); CR= with a centre or AR=: 14900;
  a full circle needs the centre (14040);
* `AR=a` (0 < a < 360 deg, else 14910): with the end point r = |e - s| / (2 sin(a/2)), c on the
  bisector by d; with the centre, e is s turned by d a about c; centre and end point both: 14900;
* `CIP` through the intermediate point `I1= J1= K1=` (G90 / G91 as the end point, or `=AC()` /
  `=IC()`): the circumcentre of the three points, direction from their order; collinear points: 14920;
  a circle in a plane inclined to the working plane: TW0012;
* `CT`: tangential to the previous element (its end tangent t in this plane, lines and arcs, through
  frame changes it is dropped): c = s + r n with n the unit normal of t toward e and
  r = |e - s|^2 / (2 n.(e - s)); no previous element: 14045; e on the tangent: 14045.

End point (all arcs): |r_e - r_s| <= max(`circle_error_const_m`, `circle_error_factor` r_s)
(MD 21000 / 21010), else 14040. Within the tolerance the centre is moved to the nearest point of the
perpendicular bisector of s and e, so the commanded arc is an exact circle through both points (the
radius changes by at most the tolerance). A circle whose plane axes have different active scale factors
would be an ellipse: TW0018 (in POCKET3 / POCKET4 / SLOT1 / SLOT2 and the turning cycles 61011).

Corners: `RND=` rounding, `CHR=` chamfer leg, `CHF=` chamfer length (leg CHF / (2 cos(delta / 2)),
delta the change of direction to the next block's start), `RNDM=` modal rounding of every following
corner (0 ends it; may stand alone), through the core's corner provider (as FANUC ,R / ,C). Non-modal
CHF / CHR / RND in a block without motion: 10882. Radius compensation G41 / G42 / G40 through the core's
cutter (mill) or tool-nose (lathe) provider: selected and cancelled only in G0 / G1 blocks (10753 /
10754), plane change while active 10757, without a cutting edge (D0) 10750; G450 / G451 are kept
modal, the provider's corner rule applies.

Block-preparation checks of the end point of every G0 / G1 / G33 / G331 / G332 block, before anything
is emitted: software limit switches (the machine position of each axis against its travel, 10720 with
the axis and + / -; continuous rotary axes and moves under TRAORI / CYCLE800 are exempt) and, while
WALIMON is active (power-on), the working area of G25 / G26 with axes (the tool tip in the basic
coordinate system: work point plus work offsets, tool offsets excluded; 10730). WALIMOF ends the check;
G25 / G26 S are the spindle speed limits.

### Frames

Settable frames G500 (none), G54-G57, G505-G599 are the core's work offsets of the same names
(`work_offsets_m`); `$P_UIFR[n]` reads and writes them as frames (translations only: a rotation, scale
or mirror in a settable frame is TW0012). G53 / G153 / SUPA suppress the frames for one block.

The programmable frame (`sys["frame"]`) has the Siemens components x_parent = T + R S M x
(translation, rotation, scale, mirror): TRANS / ATRANS (rotary axes: an angle offset), ROT / AROT (`RPL=`
in the active plane; `X Y Z` angles as RPY, R = R_z(c) R_y(b) R_x(a)), SCALE / ASCALE (0: 12160),
MIRROR / AMIRROR. An absolute instruction replaces the whole frame (TRANS alone clears it); an A-form
is chained in the frame's own coordinates (ATRANS shifted by R S M, AROT post-multiplied, ASCALE and
AMIRROR per component). Frame variables: `CTRANS() CROT() CSCALE() CMIRROR()` and the chain operator `:`,
`$P_PFRAME`, `$P_IFRAME`, `$P_ACTFRAME`; G58 / G59 set / add axial translations. The frame maps exactly
onto the core's mirror -> scale -> feature stages (`kit.apply_frame`: mirror and scale about the
program zero, then the orthonormal R and T as the feature frame), so tool offsets and arcs follow it.

**CYCLE800** (swivel plane; machines with two rotary table axes, `machine.work_rotaries`): the swivel
frame x_parent = X0 + R_sw (X1 + x) in front of the programmable frame. R_sw from `_MODE` by
`cnctwin.geom.transform.siemens_cycle800` (docs/models/geom.md 2.1): bits 7-6 = 00 axis by axis
(bits 1-0, 3-2, 5-4 the first, second, third axis, 01 X 10 Y 11 Z, each about the already rotated axes;
57 = X Y Z; `_MODE` 0 only for the zero plane, the default of `CYCLE800()`), 01 solid angle
R_z(_A) R_y(_B) R_z(_C), 10 projection angles, 11 rotary axes direct: _A and _B are the positions of
rotary axes 1 and 2 (the base-side and part-side table axis), the table turns the part by
R_t = R_1(_A) R_2(_B) and R_sw = R_t^T (the swivelled axes stay on the machine axes; 61153 without two
table axes). The rotary positions that bring the new Z onto the tool axis are the closed-form solution
of R_1(q1) R_2(q2) n = t (Paden-Kahan subproblem 2, murray-li-sastry-1994 ch. 3; two solutions, `_DIR`
-1 / +1 picks the sign of the first rotary axis, 0 swivels the frame only; positions brought into the
axis travel, continuous axes to the nearest turn; none reachable: 61184, a direct position outside the
travel: 61189). `_FR` retracts first (1 and 4: Z to the top of its travel, which is the tool direction
on a table machine; 2: also X Y to their home; 5: `_FR_I` along Z, 0 is 61190; others 61183); `_ST` units
digit 1 adds to the active swivel frame; `_TC` "0" deselects. Afterwards the linear axes follow the table
kinematics (stage `siemens840d:CYCLE800`, Space.TIP commands with `tip_interpolation` "joint").

**TRAORI / TRAFOOF**: tool centre point control on the table kinematics (stage `siemens840d:TRAORI`,
Space.TIP, "tip"): the programmed point is the tool tip in the workpiece; the orientation is given by the
rotary axes or by the tool vector `A3= B3= C3=` in the workpiece, turned into rotary positions by the
same subproblem-2 solution (least rotary motion). Both stages are `TableTcp` from the Haas dialect.

### Programming

* R0-R99 (`r_parameters`), `R[expr]`; DEF INT / REAL / BOOL / CHAR / STRING[n] / AXIS / FRAME local
  user data per program level (`sys["lud"]`), PROC with value and VAR (by reference; a constant for a
  VAR parameter: 12320) parameters, SAVE (modal G functions and frames restored at the subprogram end),
  EXTERN.
* Expressions (`values.py`): degrees; ATAN2(a, b) is the angle of the vector (b, a); POT(x) = x^2; TRUNC
  toward zero; ROUND half away from zero; DIV integer division toward zero; MOD the remainder with the
  dividend's sign; priority NOT B_NOT, * / DIV MOD, + -, B_AND, B_XOR, B_OR, AND, XOR, OR, << (string
  concatenation), then the comparisons (a combined condition needs brackets, as on the control). INT is
  rounded half away from zero on assignment. Division by 0 and arguments out of range: 14051.
* Control structures are matched at load: IF / ELSE / ENDIF, WHILE / ENDWHILE, FOR / TO / ENDFOR (the
  end value is evaluated once at the start, `sys["for"]`), LOOP / ENDLOOP, REPEAT / UNTIL. Jumps GOTOF
  (forward), GOTOB (backward), GOTO (forward, then backward), GOTOC (no alarm when not found) to labels, N
  numbers or a string expression; not found 14080.
* Program section repetition: `REPEATB label P=n` repeats one line; `REPEAT label P=n` the lines from
  the label to the next `ENDLABEL:` before the statement, or else up to the line before the statement;
  `REPEAT label1 label2 P=n` the lines from label1 to label2 (both included, before or after the
  statement). The section runs n more times (P omitted: 1, P=0: none), then the block after the
  statement follows. Sections nest (`sys["sections"]`, at most `structure_nesting_levels`) and may call
  subprograms; a section containing its own statement is 12640, a label not found 14080.
* Subprograms by name (parameters `SUB(1,,R1)`, repetitions `P=n` / `Pn`), `Lnnn`, `CALL "NAME"`,
  `MCALL` (modal call after every positioning; MCALL alone ends it); ended by M17, RET, or M2 / M30 in
  the main program; `sub_nesting_levels` program levels (14012 beyond).
* MSG("text") (MSG() clears), SETAL(n) raises cycle alarm n (60000-69999, else 14020),
  STOPRE (SYNC `exact_stop_check` and a buffer wait: the following blocks are prepared after the motion
  before them has ended), WAITM(marker, channels) (SYNC `path_wait`, then `ex.wait("path_wait")` until
  every named channel has reached the marker; a channel the machine does not have: 14020).

### Tools, spindle, coolant

T n preselects (MODE `tool_preselect`), M6 changes (TOOL_CHANGE), D n selects the cutting edge
(`cutting_edge_default` D1 after M6; the twin's tools have one edge, so D2-D9: 17181; D0 no offsets).
Milling tools: the length acts along the axis normal to the active plane (G17 Z, G18 Y, G19 X) and the
radius for G41 / G42; turning tools: the X and Z lengths and the nose radius. Offsets act with the next
motion. `$TC_DP1/3/4/5/6/12/15` read and write the offset rows. S is rpm, or m/min with G96 (LIMS=
upper speed); M3 / M4 / M5 (one per block: 14760), SPOS= oriented stop, M40-M45 gears; M7 mist and M8
flood when the machine has such pumps (else TW0005), M9 off; M0, M1 (optional stop), M2 / M30 (end,
part count), M17 (subprogram end). More than `max_m_codes_per_block` M functions: 14770. Machine-builder
M codes come from `physics.controller.m_codes`; other M codes: TW0005.

### Technology cycles

Called like subprograms with positional parameters (omitted trailing ones are 0, lengths in program
units, angles in degrees); the parameter lists are those of the 840D sl Job Planning manual
(siemens-840dsl-jobplan-ru-2018 ch. 20.1) and, for the cycles it lists only by name, the classic 840D
lists (cycles.py docstring). The expansion is a list of core Steps queued after the block (through
mirror, scale and rotation like programmed moves); spindle, stop, frame and nose-compensation actions
inside it run in sequence through the resumer `siemens840d:action`, which also expands the MCALL cycle
after each positioning where the tool then is. The drilling axis is the plane normal (`_DMODE` / `_AXN`
select another); the retraction side is that of RTP relative to RFP; DPR wins over DP; DTB > 0 dwells
seconds, < 0 spindle revolutions.

* CYCLE81 drilling, CYCLE82 with dwell, CYCLE83 deep hole (FDEP / FDPR first depth, `_DAM` degression
  (< 0 in % of the first depth), `_MDEP` minimum, VARI 0 chip breaking by `_VRT`
  (`cycle83_chip_break_retract_m` when 0), 1 chip removal: out to the reference plane and back to the
  last depth minus the advance clearance `_DIS1` (when 0: `cycle83_anticipation_m`, above
  `cycle83_anticipation_limit_depth_m` depth / 50, at most `cycle83_max_anticipation_m`), FRF feed
  factor of the first infeed, DTS / DTB / `_DTD` dwells), CYCLE84 rigid tapping (oriented start POSS,
  SST in / SST1 out, PIT with `_PITA` units or MPIT metric coarse size, `_VARI` 1 / 2 chip breaking /
  removal by `_DAM`, SDAC spindle after the cycle), CYCLE840 floating holder (SDR reversal, ENC 0 with
  encoder at the pitch per revolution, 1 at F), CYCLE85 (FFR in, RFF out), CYCLE86 (oriented stop
  POSS, lift RPA / RPO / RPAP, rapid out, SDIR on again), CYCLE87 / CYCLE88 (spindle stop and M0 at the
  bottom, the operator retracts; then rapid out), CYCLE89 (dwell, out at F). No F: 10860; a reference
  plane that contradicts the depth: 61101.
* HOLES1 (row: SPCA SPCO STA1 FDIS DBH NUM, started from the end nearer the tool) and HOLES2 (circle: CPA
  CPO RAD STA1 INDA NUM, INDA 0: 360 deg / NUM) move to every hole in the plane and run the MCALL drilling
  cycle there (NUM 0: 61103; a modally called user subprogram: TW0012).
* POCKET3 / POCKET4 / SLOT1 / SLOT2 (`milling.py`): contour-parallel from the inside out. For a rounded
  rectangle of length L, width W, corner radius rc (circles and obround slots as special cases) the
  tool-centre path at wall distance e is the rounded rectangle with half sizes L/2 - e, W/2 - e and
  corner radius max(rc - e, 0), the exact offset of the wall; roughing passes are evenly spaced between
  the innermost path (e = W/2) and e = r + allowance with a spacing <= `_MIDA` (when 0:
  `pocket_stepover_fraction` of the diameter), depth layers equal and <= `_MID`; `_VARI` units digit 1
  roughs (leaving `_FAL` / `_FALD`), 2 finishes (the sides at e = r, the bottom when `_FALD` > 0). Climb
  milling with M3 is the counter-clockwise loop (G41 sense), M4 swaps. Plunging vertical, helical
  (`_RAD1` radius, `_DP1` depth per turn, the last turn closed by a full circle; missing: 61108) or
  oscillating at the ramp angle; tool radius too large 61105, slot geometry impossible (shorter than
  wide, overlapping) 61104. SLOT2 circumferential slots use the annular-sector family.
* CYCLE95 / CYCLE97 / CYCLE93 (`turning.py`; lathe in G18, else 61004): CYCLE95 roughing parallel to Z
  or X between the contour subprogram NPP (shifted by FALZ / FALX and offset by FAL) and its envelope, in
  equal infeeds <= MID, undercuts at FF2, chip breaking DAM / DT, lift `_VRT` (`cycle95_lift_m`), and
  finishing along the contour at FF3 with tool-nose compensation; CYCLE97 thread passes (NRC roughing
  cuts with constant infeed or constant chip cross-section depth_k = (TDEP - FAL) sqrt(k / NRC), NID idle
  passes, NUMT starts, IANG flank infeed); CYCLE93 grooves with flank angles, radii / chamfers, pecking
  IDEP and finishing (tool width = twice the radius offset: the control takes it from the second cutting
  edge, which the twin's tools do not have; wider than the groove 61602).
* CYCLE832(_TOL, _TOLM, _OTOL): `_OFF` G64 COMPOF; `_FINISH` / `_SEMIFIN` / `_ROUGH` G642 COMPCAD SOFT
  FFWON with the tolerance (when 0: the machine's smoothing table entry finish / medium / rough), as
  MODE `smoothing` for the motion model.

### Alarms

Condition keys map to 840D sl numbers (`tables.ALARM_CODES`); texts are the database's when it lists the
number (`source: db:siemens-840d-sl`), else the built-in copy. The text parameters are filled as the
control shows them: `[Channel %1: ]` only on machines with more than one path; `Block %n` is the N
number, the label or `line k`; the other `%k` are the alarm's own values (axis, sign, parameter). Twin
codes: TW0005 M code not defined for the machine, TW0006 system variable not modelled, TW0008 WAITM
markers inconsistent, TW0012 function not modelled, TW0018 circle under unequal scale factors.

### Parameters (db/models/controller-siemens840d.yaml)

| name | value | stands for | source |
|---|---|---|---|
| circle_error_const_m / circle_error_factor | 0.01 mm / 0.001 | MD 21000 / 21010, circle end-point tolerance max(const, factor r) | est |
| sub_nesting_levels | 16 | program levels with the main program (14012) | est (802D sl: 8, siemens-802dsl-diag-2011 p. 101) |
| structure_nesting_levels | 16 | IF / WHILE / FOR / LOOP / REPEAT and section repetitions per level (12641) | est (802D sl: 8, p. 96) |
| r_parameters | 100 | MD 28050 MM_NUM_R_PARAM | est |
| max_m_codes_per_block | 5 | M functions per block (14770) | est |
| cutting_edge_default | 1 | MD 20270 CUTTING_EDGE_DEFAULT | est |
| pocket_stepover_fraction | 0.8 | width of infeed when `_MIDA` = 0 | est |
| cycle83_chip_break_retract_m | 1 mm | CYCLE83 / CYCLE84 retraction when `_VRT` = 0 | est |
| cycle83_anticipation_m, cycle83_anticipation_limit_depth_m, cycle83_max_anticipation_m | 0.6 mm, 30 mm, 7 mm | CYCLE83 advance clearance when `_DIS1` = 0 | est |
| cycle95_lift_m | 1 mm | turning cycles' lift-off when `_VRT` = 0 | est |

### Words (`tables.WORD_LIST`)

* motion: G0 G1 G2 G3 CIP CT G33 G331 G332, I J K (=AC() =IC()), I1 J1 K1, CR=, AR=, TURN=, AP= RP=
* plane, units, distance: G17 G18 G19, G70 G71 G700 G710, G90 G91, AC() IC() DC() ACP() ACN(), DIAMON
  DIAMOF DIAM90
* feed and spindle: F, G93 G94 G95 G96 G97, LIMS=, S, M3 M4 M5, SPOS=, M40-M45, G25 / G26 S, CFC CFTCP
  CFIN
* frames: G500 G54-G57 G505-G599, G53 G153 SUPA, TRANS ATRANS ROT AROT (RPL=) SCALE ASCALE MIRROR
  AMIRROR, G58 G59, G110 G111 G112, $P_PFRAME $P_UIFR[n] $P_IFRAME $P_ACTFRAME, CTRANS CROT CSCALE
  CMIRROR ':', G25 / G26 with axes (working area limitation) WALIMON WALIMOF
* path control: G60 G64 G641 ADIS= ADISPOS= G642 G643 G644 G645, G9, G601 G602 G603, SOFT BRISK DRIVE,
  FFWON FFWOF, COMPOF COMPON COMPCURV COMPCAD COMPSURF, CYCLE832, CHF= CHR= RND= RNDM=. G641 smooths by a distance
  criterion: ADIS is the rounding distance along the blocks from the transition (Smoothing `tolerance_kind`
  `distance`; until 2026-11-09 taken as a deviation, a rounding 2.4 x ADIS long at a square corner); CYCLE832's
  S_TOL and G642 bound the deviation from the contour; leaving G641 restores CYCLE832's contour tolerance
  (docs/models/motion.md 2.2)
* tools: T, M6, D0-D9, G40 G41 G42, G450 G451, NORM
* other M: M0 M1 M2 M30 M17 M7 M8 M9, machine-builder M codes (`physics.controller.m_codes`)
* programming: R0-R99, R[n], DEF INT / REAL / BOOL / CHAR / STRING[n] / AXIS / FRAME, PROC (VAR) SAVE,
  EXTERN, subprogram calls by name with P= and parameters, CALL, MCALL, M17, RET, IF / ELSE / ENDIF,
  WHILE / ENDWHILE, FOR / TO / ENDFOR, LOOP / ENDLOOP, REPEAT / UNTIL, REPEAT label [label] P=, REPEATB
  label P=, ENDLABEL, GOTOF GOTOB GOTO GOTOC (labels, N numbers), MSG(), SETAL(), STOPRE, WAITM(), G4
  F / S, G74, G75 FP=
* functions and operators: SIN COS TAN ASIN ACOS ATAN2 SQRT POT ABS TRUNC ROUND ROUNDUP LN EXP MINVAL
  MAXVAL BOUND NUMBER TOSTR STRLEN, + - * / DIV MOD == <> > < >= <= AND OR XOR NOT B_AND B_OR B_XOR
  B_NOT <<
* system variables: $P_EP $P_F $P_S $P_TOOLNO $P_TOOL $P_TOOLP $P_GG $P_DRYRUN $P_SEARCH $P_AXN1-3
  $P_UIFR $P_PFRAME $P_IFRAME $P_ACTFRAME $AA_IW $AA_IM $A_YEAR $A_MONTH $A_DAY $A_HOUR $A_MINUTE
  $A_SECOND $TC_DP1 $TC_DP3-$TC_DP6 $TC_DP12 $TC_DP15 (others: TW0006)
* cycles: CYCLE81 CYCLE82 CYCLE83 CYCLE84 CYCLE840 CYCLE85 CYCLE86 CYCLE87 CYCLE88 CYCLE89, HOLES1
  HOLES2, POCKET3 POCKET4 SLOT1 SLOT2, CYCLE800 (axis by axis, solid angle, projection angle, rotary
  axes direct), CYCLE832, CYCLE95 CYCLE97 CYCLE93, TRAORI TRAFOOF (rotary axes or A3= B3= C3=)

### Validity and error

Positions and arcs are exact to the axis quantum (the arc end-point correction moves the centre by at
most the circle tolerance); frames, CYCLE800 and TRAORI are exact rigid transforms (subproblem 2 in
closed form, residual ~1e-15). The cycle sequences follow the parameter tables of the job-planning
manual; their fixed internal distances (retractions, advance clearances, stepover) are the generally
documented defaults marked est above and can differ by machine data and cycle version. Not modelled
(TW0012, never silent): spline and polynomial interpolation, TRANSMIT / TRACYL / TRAANG, 3-D tool
compensation, orientation interpolation modes (ORIWKS ...), CASE, positioning axes (POS, POSA, WAITP),
synchronized actions, settable frames with rotations, scale or mirror, CYCLE800 on head kinematics,
the other standard cycles (`tables.STANDARD_CYCLES`: CYCLE60-64, CYCLE70-79, CYCLE90 92 94 96 98 99,
CYCLE801 802 830 899 930 940 950-952, LONGHOLE; a program of that name in the project is called
instead), tool management by name (`T="name"`: 17190), more than one cutting edge per tool. A program
that is not there is 14011.

Tests: tests/unit/controller/test_siemens.py (motion, arcs, frames, language,
tools, alarms), test_siemens_cycles.py (expected command streams of the cycles, CYCLE800, TRAORI,
turning), test_siemens_checkpoint.py (resume inside a FOR loop and inside a cycle expansion
bit-identical), test_siemens_heidenhain_same_part.py (one part in FANUC, Siemens, Klartext and ISO: the
same toolpath).

### References

SINUMERIK 840D sl Programming Manual Fundamentals (siemens-840dsl-fund-ru-2018: G
functions, groups, modal effect) and Job Planning (siemens-840dsl-jobplan-ru-2018: cycle parameter
lists, CYCLE800, CYCLE832); alarm texts and causes: SINUMERIK 802D sl Diagnostics Manual
(siemens-802dsl-diag-2011, the same NCK and cycle alarms) and the 840D sl alarm list
(siemens-840dsl-diag-zh-2022, numbers); Murray, Li and Sastry 1994 (subproblem 2); db/controllers/
siemens-840d-sl.yaml.

## Heidenhain TNC (Klartext and ISO)

`cnctwin/controller/dialects/heidenhain/` registers `heidenhain_klartext` and `heidenhain_iso` (TNC 640,
NC software 34059x-18). Both names build the same dialect: a program file that starts with `%` is read
as ISO, one with `BEGIN PGM` as Klartext, whatever the machine's dialect, and one handler executes both.
Modules: `klartext.py` (Klartext reader, the shared block form `HBlock(kind, fields)`), `iso.py` (ISO
reader), `formula.py` (Q-parameter values, formulas, string functions), `handler.py`
(`HeidenhainHandler`), `isoexec.py` (ISO modal G codes onto the Klartext semantics), `contour.py`
(APPR / DEP, FK), `cycles.py` (fixed cycles, PATTERN DEF, program cycles, the step resumer
`heidenhain:action`), `plane.py` (PLANE), `tables.py` (error numbers, built-in texts, M function lists,
the word list); the runtime (`NativeHandler`, table kinematics, the TCP stages) and the pocket toolpaths
are shared from `siemens840d/kit.py` and `siemens840d/milling.py`. Defaults:
`db/models/controller-heidenhain.yaml` (plus `controller-block-time`); error texts:
`db/controllers/heidenhain-tnc640.yaml` (`source: db:...`), else built-in copies. Power-on: G17 (lathes
G18), R0, absolute, feed per minute, no tool. No random numbers; nothing on the bus; all state is in
`PathState` (`sys`, `variables["Q" | "QR" | "QL<level>"]`, `sys["qs"]`) and checkpoints with it.

### Program format

* Klartext: one program per file, `0 BEGIN PGM NAME MM|INCH` ... `n END PGM NAME MM`; every block
  starts with its block number; a line ending in `~` continues on the next (cycle definitions with one Q
  parameter per line, continuation lines without numbers); `;` starts a comment (also between a value
  and `~`); `*` blocks are structure items. Words are separated by spaces: `X+10`, `IX-5` (incremental),
  `PR` `PA` `IPR` `IPA` (polar), values may be Q parameters (`X+Q1`, `Y-QL3`). Text outside strings is
  upper-cased.
* ISO: `%NAME G71 *` (G70 inch) ... `N99999999 %NAME G71 *`; blocks `N<n> words *`; several G codes per
  block; G00-G16 are modal (a block without a motion code continues the active one; a block without an
  end point does not repeat it).
* Reading never fails: a block that cannot be read carries its error (160-0003 block format incorrect,
  160-00CF NC command not available, 160-000B CYCL DEF not defined ...) raised when control reaches it.

### Order inside a positioning block

Coordinates and values are evaluated (an undefined Q parameter as a lone coordinate drops that axis; in
a calculation it is 160-0028), the radius compensation (RL / RR / R0 modal, R+ / R- paraxial lengthening /
shortening by the radius), the feed (F modal, FMAX for this block, FAUTO = the TOOL CALL feed, FU per
revolution; FZ needs the tool table's teeth: 160-00E0; none: 160-0069), the M functions that act at the
block start (M3 M4 M8 M13 M14 M126 M127), the motion (RND / CHF of the next block as its corner), then
the M functions at the block end (M5 M9 M0 M1 M2 M30, M99 / M89 cycle calls, M128 / M129, M140). More
than `max_m_codes_per_block` M functions, or M3 with M4, M8 with M9, M128 with M129: 160-0122.

### Paths

* `L` line; `CC` pole (alone: the current position); `C` circle about CC (DR+ / DR-); `CR` circle by
  radius (R > 0 at most 180 deg, R < 0 more; radius below the half chord beyond the tolerance 1A0-0083);
  `CT` tangential circle (1A0-0082 without a preceding element or with the end point on the tangent);
  `LP` / `CP` / `CTP` polar about CC (`IPA` beyond 360 deg with `IZ` is a helix: the full turns are
  passed as turns); `RND R` rounding and `CHF` chamfer between the neighbouring elements (the core's
  corner provider); M91 coordinates in the machine system, M92 relative to the machine's home position:
  the machine axes go there and "the tool length is not compensated" in the block (TNC 640 Klartext
  manual, M91/M92), so `L Z-10 R0 FMAX M91` puts the spindle 10 mm below the machine datum whatever the
  tool; the called tool's length stays active, and the next block programmed in the workpiece system
  starts from its tip.
* Arc end point: the end point must lie on the start circle within `arc_end_point_tolerance_m`
  (0.016 mm, circleDeviation), else 1A0-0084 "Circle incorrectly programmed"; within it the centre is
  moved onto the perpendicular bisector (exact circle). ISO: G02 / G03 about I J K (G90 absolute, G91
  from the last position) or with R, G05 / G15 with the last programmed direction (G02 / G03 / G12 / G13
  / C / CP set it; none yet: 1A0-000C), G06 / G16 tangential, G10-G16 polar (R radius, H angle).
* Endless rotary axes (machine `continuous`) are shown modulo 360 deg: an absolute value moves from the
  shown position to it (from 350 to 10: -340 deg); with M126 the shortest way (at most half a turn, +180
  on a tie); M127 ends M126. Limited rotary axes move to the value.
* Approach and departure (`contour.py`; the compensation itself is the core's cutter provider):
  `APPR LT` (PH = PA - LEN t, t the start tangent of the next element), `APPR LN` (PH = PA + (LEN + r) n,
  along the normal), `APPR CT` (arc tangent at PA, centre PA + |R| n on the tool side for R > 0, centre
  angle CCA), `APPR LCT` (the same arc entered by the tangent from PS), the polar `APPR PLT / PLN / PCT /
  PLCT`, and `DEP LT / LN / CT / LCT` (and polar) mirrored at the end point PE; without RL / RR 1A0-0071.
* FK free contour (`FPOL`, `FL` / `FLT`, `FC` / `FCT`): every element is solved from the known start point
  and tangent with its own data (end point X Y / IX IY / PR PA about FPOL, direction AN / IAN /
  tangential, LEN, centre CCX CCY / ICCX ICCY / CCPR CCPA, R, DR). One solution is executed; several
  (coinciding solutions count once) need the following `FSELECT n` (ordered by the end point's first,
  then second plane coordinate). `CLSD+` marks the first element of a closed contour, `CLSD-` the last
  (its missing end coordinates are the CLSD+ start point). Not determined or several without FSELECT:
  TW0040; contradictory: TW0041 (the TNC's own FK error numbers were not in the error list read; a
  database entry with kind fk_undetermined / fk_contradiction replaces them). Data relative to other
  blocks (RX RY RAN PAR DP PDX PDY RPR RPA): TW0012.
* Software limit switches: the end point of every L / LP / FK / ISO line in machine coordinates against
  the axis travel (1A1-0010) before the motion is emitted (not under PLANE / TCPM).

### Coordinate transformations

The TNC applies them in a fixed order whatever the order they were programmed in: the tilted plane
(PLANE) about the active datum, the datum shift (cycle 7 / ISO G54, given in the system active when it
is defined), mirroring (8 / G28), rotation (10 / G73) and scaling (11 / G72, 26):

x_machine = preset + O + R_tilt (M R(rot) S (x - c) + c)

with O the datum shift turned by the tilt active at its definition, M the mirror, S the scale factors
about the centre c (26: per axis, CCX CCY CCZ). The chain maps exactly onto the core's mirror -> scale ->
rotation -> feature stages, using M R(theta) = R(-theta) M for an odd number of mirrored plane axes and
M S(c) = S(Mc) M. Cycle 247 (Q339) / the ISO preset select `P<n>` of `work_offsets_m` (missing:
280-03F1); a datum or preset table entry `#n` and ISO G53: TW0012. A circle whose plane axes have
different scale factors (26) would be an ellipse: TW0018 (the TNC requires equal factors on the axes of
a circle).

**PLANE** (`plane.py`; machines with two rotary table axes, else 1A0-0100 / transform unavailable): the
tilt R (columns = the tilted X', Y', Z' in the untilted frame) from SPATIAL SPA SPB SPC
(R = R_z(SPC) R_y(SPB) R_x(SPA), fixed axes; as `geom.transform.heidenhain_plane_spatial`), PROJECTED
PROPR PROMIN ROT (X' along the plane's trace in ZX at PROPR about +Y, the trace in YZ at PROMIN about +X
also in the plane, Z' = their cross product, then ROT about Z'; equal to SPATIAL when one angle is 0),
EULER (R_z(EULPR) R_x(EULNU) R_z(EULROT)), VECTOR (Z' along N, X' the base vector made normal), POINTS
(X' = P1 -> P2, Z' = (P2 - P1) x (P3 - P1)), RELATIV (the active R turned about its own X, Y or Z), AXIAL
(the plane the rotary axes give at those positions, R = R_table^T) and RESET. The rotary positions are
the closed-form subproblem-2 solution (as CYCLE800): SEQ+ / SEQ- the sign of the first rotary axis of
the chain, else the least rotary motion; unreachable 1A0-00FC. MOVE positions with compensating linear
motion (the point DIST above the tip stays fixed on the part; TCPM interpolation), TURN moves the rotary
axes only, STAY moves nothing (Q120-Q122 hold the angles for a later positioning block); F / FMAX for
the positioning. With MOVE and TURN, MB (move back: `MB MAX`, `MB 50`, `MB+50`, `MB 0`) first retracts the
tool along the current tool axis, by the incremental length or to the end of the traverse range, at the
positioning feed (TNC 640 Klartext manual, PLANE automatic positioning; the M140 MB retraction: beyond the
range 1A1-0010); MB with STAY, or without a length, is 160-0003. Afterwards the program frame is the
tilted one and the linear axes follow the table kinematics (stage `heidenhain:PLANE`, Space.TIP, "joint").

**TCPM**: M128 / `FUNCTION TCPM` (options F TCP, AXIS POS, PATHCTRL AXIS, REFPNT TIP-TIP; F CONT, AXIS
SPAT, PATHCTRL VECTOR and the other reference points: TW0012) keep the tool tip on the programmed path
(stage `heidenhain:TCPM`, Space.TIP, "tip"); M129 / `FUNCTION RESET TCPM` end it; with radius
compensation active 160-00D0.

### Tools, spindle, coolant

`TOOL CALL n | "name" X|Y|Z S F DL DR DR2`: n from the project's tools (name: the tool table's
`tool_id`; unknown 160-00C5), the tool axis (its letter must be on the machine), the spindle speed, the
feed (the current feed until a positioning block programs another one, and the FAUTO value), the length
and radius deltas DL / DR (DR2 is kept for 3-D compensation, which is not modelled). The length acts
along the tool axis, the radius for RL / RR; both from the next motion. `TOOL DEF n L R` sets the offsets
(with neither: preselection, MODE `tool_preselect`); `TOOL DEF n` alone preselects. M3 / M4 / M5, M13 /
M14 (spindle and coolant), M8 / M9; M0, M1 (optional stop), M2 / M30 (spindle and coolant off, part
count; inside a called program 160-0160). `M140 MB d [F]` retracts d along the tool axis (the TOOL CALL
axis, positive direction; the tool does not tilt on table kinematics) at the end of the block, at F or
rapid; `MB MAX` to the end of the travel range; beyond it: 1A1-0010. Other M functions the TNC lists
(M94 M97 M98 M101-M103 M107-M111 M116-M118 M120 M130 M136-M138 M141 M143-M145 M148 M149 M197): TW0012;
unknown: 160-00C3; machine-builder M codes from `physics.controller.m_codes`.

### Q parameters and program flow

* Q0-Q1999 (Q100-Q199 preassigned, read only: Q108 tool radius, Q109 tool axis, Q110 spindle state,
  Q111 coolant, Q113 unit, Q114 tool length, Q115-Q119 last touch point, Q120-Q122 PLANE angles; writing
  them 160-0028), QL0-QL499 local to each program level (CALL PGM), QR0-QR499 remanent, QS0-QS1999
  strings. `FN 0`-`FN 8` (assignment, + - * /, SQRT, SIN, COS, root of a sum of squares LEN), `FN 9`-`FN 12`
  conditional jumps (EQU NE GT LT, IS DEFINED / IS UNDEFINED), `FN 13` angle (ANG, 0..360), `FN 14 ERROR=`
  (1000-1199: the predefined errors 280-03E8 + (n - 1000), hexadecimal; other numbers 280-0064 "FN 14:
  error code n"), `FN 16 F-PRINT` (the text to `PathState.printed`; SCREEN: also stops with the message),
  `FN 17 SYSWRITE ID50 NR1/2/4/5` (tool length, radius, DL, DR), `FN 18 SYSREAD` (ID10 NR3, ID20
  NR1-5/8/9, ID30, ID35, ID50, ID60, ID210 NR2-4/6, ID220 NR2, ID230 NR2/3, ID240 NR1, ID270 NR1, ID280 NR1;
  others TW0006); `FN 15 PRINT` does not exist on the TNC 640 (160-00CF). Formulas: + - * / ^ (right
  associative) % ( ), SIN COS TAN ASIN ACOS ATAN (degrees), SQ SQRT LN LOG EXP ABS INT (toward zero) FRAC
  SGN NEG, PI; `SET UNDEFINED`; division by 0 and arguments out of range 160-0032.
* Strings: `QS = "text"`, `QS = QSm`, `STRING FORMULA QSa || QSb || "text"`, `TOCHAR ( DAT+value
  DECIMALSn )` (fixed point, DECIMALS default 3, an assumption), `SUBSTR ( SRC_QSm BEGk LENn )` (from
  position k counted from 0; outside the string 160-0028), and in formulas `TONUMB ( SRC_QSm )`,
  `STRLEN ( SRC_QSm )` (-1 for an undefined string), `INSTR ( SRC_QSm SEA_QSn BEGk )` (the position from
  0, the length of QSm when not found), `STRCOMP ( SRC_QSm SEA_QSn )` (0 equal, -1 first, +1 after);
  SYSSTR: 160-00CF; an undefined string 160-00AF.
* `LBL n` / `LBL "name"`, `LBL 0` (end of a label subprogram); `CALL LBL n` runs the label subprogram;
  `CALL LBL n REP m` with the label before the call repeats the section between them m more times
  (counter in `sys["rep"]`); recursion 160-00A8, nesting beyond `sub_nesting_levels` 160-00AA, label not
  found 160-001B, twice 160-001C. `CALL PGM name` (path and extension ignored; a different unit gives a
  warning; recursion 160-00AB), `SEL PGM name` + `CALL SELECTED PGM`, END PGM returns. STOP / M0.

### Cycles

Call-active cycles take effect at `CYCL CALL`, `M99` (per block), `M89` (modal until the next CYCL DEF),
`CYCL CALL POS X Y Z` (positioning logic: in the plane first when the tool is above the 2nd set-up
clearance, else up first; Z adds to Q203) and `CYCL CALL PAT` (every point of the last PATTERN DEF at the
2nd set-up clearance, its Z adding to Q203). With POS and PAT the cycle is expanded when the tool has
reached each position. Along the tool axis every drilling cycle moves at rapid to Q203 + Q200, works,
and returns to Q203 + Q200 or the larger 2nd set-up clearance Q203 + Q204; Q201 > 0 is 280-0435, Q201 = 0
does nothing (in the pockets 280-0488); feeds Q206 / Q207 / Q208 / Q253 / Q254 / Q385 in mm/min.
Spindle and stop actions run in sequence through `heidenhain:action`.

* 200 DRILLING (pecks Q202, dwell Q210 at the top, Q211 at the bottom), 201 REAMING (Q208 out), 202
  BORING (oriented stop Q336, disengaging `boring_disengage_m` toward Q214, out at Q208), 203 UNIVERSAL
  DRILLING (decrement Q212 down to Q205, chip breaks Q213 by Q256 or `chip_break_retract_m`), 204 BACK
  BORING (rapid to Q203 + Q200, oriented stop Q336, off centre by Q251 toward Q214 (1 -a, 2 -b, 3 +a, 4 +b;
  0: 280-03F9), down at Q253 until the cutting edge (Q252 above the tip) is Q200 under the underside
  Q203 - Q250, back to the centre, spindle on, up at Q254 until the edge is Q249 above the underside, dwell
  Q255, oriented stop, off centre, out at Q253, 2nd set-up clearance, centre, spindle as before), 205
  UNIVERSAL PECKING (advance stop distances Q258 / Q259, chip breaking Q257 / Q256, deepened start Q379 at
  Q253), 206 TAPPING with a floating holder (spindle reversal at depth; spindle stopped 280-03E8), 207
  RIGID TAPPING (pitch Q239, sign = hand) and 209 with chip breaking (Q257, Q256, Q336, retract speed
  factor Q403), 208 BORE MILLING (helix of pitch Q334 to the diameter Q335, climb / up-cut Q351), 240
  CENTERING (depth; Q343 = 1 by diameter needs the point angle: TW0012), 251 RECTANGULAR POCKET, 252
  CIRCULAR POCKET, 253 SLOT MILLING (Q215 0 roughing and finishing, 1 roughing, 2 finishing; allowances
  Q368 / Q369, stepover Q370 times the tool radius, plunging Q366 0 vertical, 1 helical, 2 reciprocating at
  `plunge_angle_rad`, helix radius `helix_radius_fraction` of the tool radius; the contour-parallel paths of
  `milling.py`, see Siemens POCKET3), CYCL DEF 12 PGM CALL and SEL CYCLE (a program called like CALL PGM;
  its own positioning blocks do not repeat M89; POS / PAT of it: TW0012).
* DEF-active: 7 DATUM SHIFT, 8 MIRRORING (no axis: off), 9 DWELL TIME, 10 ROTATION (ROT / IROT), 11
  SCALING (SCL; 0: 1A0-0086), 13 ORIENTATION, 26 AXIS-SPECIFIC SCALING (with CCX CCY CCZ), 32 TOLERANCE
  (T, HSC-MODE, TA: MODE smoothing), 247 PRESETTING.
* `PATTERN DEF` POS (single points, IX / IY / IZ from the previous point), ROW (D spacing, NUM, ROT), PAT
  and FRAME (DX DY NUMX NUMY ROT; ROTX / ROTY turn only the main / secondary axis: a sheared grid; FRAME
  without the inner points), CIRC (D diameter, START, NUM over the full circle) and PITCHCIRC (STEP); Z adds
  to the surface.

### ISO mapping (`isoexec.py`)

G00 / G01 line (both modal), G02 / G03 / G05 circle, G06 tangential circle, G10 / G11 polar line, G12 /
G13 / G15 polar circle, G16 tangential polar circle, G24 / G25 chamfer / rounding R, G28 mirroring (axes
named), G29 the last position as pole, G30 / G31 blank, G36 S spindle orientation, G38 stop, G04 F dwell,
G40-G42, G51 T preselection, G54 datum shift, G62 T tolerance, G70 / G71, G72 F scaling, G73 H rotation,
G79 cycle call, G90 / G91, G98 L label, `L n,m` label call, `%NAME` program call, G99 T L R tool
definition, G200-G209 G240 G251-G253 with their Q parameters, T tool call (axis by G17 / G18 / G19), D00-D18
with P01 P02 P03 (= FN 0-18), Q formulas. G26 / G27 (tangential approach / departure), G53 and the codes
not listed: TW0012 or 160-0003.

### Errors

Condition keys map to TNC error numbers (`tables.ALARM_CODES`, heidenhain-nc-errors-2023), texts from the
database or the built-in copies. Twin codes: TW0005 M code not defined for the machine, TW0006 system data
not modelled, TW0012 function not modelled, TW0018 circle under unequal scale factors, TW0040 / TW0041 FK
element not determined / contradictory.

### Parameters (db/models/controller-heidenhain.yaml)

| name | value | stands for | source |
|---|---|---|---|
| arc_end_point_tolerance_m | 0.016 mm | circleDeviation (MP 200901), circle end-point tolerance | heidenhain-tnc640-klartext-2023 p. 166 |
| sub_nesting_levels | 19 | label subprograms, repeats and CALL PGM nesting (160-00AA) | est |
| max_m_codes_per_block | 4 | M functions in one block | est |
| boring_disengage_m | 0.2 mm | cycle 202 disengaging distance | est |
| chip_break_retract_m | 0.2 mm | cycles 203 / 205 chip-breaking retraction when Q256 = 0 | est |
| plunge_angle_rad | 5 deg | cycles 251-253 plunging angle (the tool table's ANGLE) | est |
| helix_radius_fraction | 0.75 | cycles 251 / 252 helix radius / tool radius | est |

### Words (`tables.WORD_LIST`)

* program: BEGIN PGM / END PGM name MM | INCH, BLK FORM 0.1 / 0.2, * structure blocks, ; comments, ~
  continuation
* tools: TOOL CALL n | "name" X | Y | Z S F DL DR DR2, TOOL DEF n L R (or preselection), FAUTO
* paths: L, CC, C (DR+ DR-), CR (R, DR), CT, RND R, CHF, LP / CP / CTP (PR PA IPR IPA, helix with IZ), R0
  RL RR R+ R-, F FMAX FAUTO FU FZ, I-prefixed incremental coordinates, M91 M92
* approach: APPR LT / LN / CT / LCT and the polar APPR PLT / PLN / PCT / PLCT (LEN CCA R), DEP LT / LN /
  CT / LCT
* FK: FPOL, FL FLT FC FCT with X Y IX IY PR PA AN IAN LEN DR R CCX CCY ICCX ICCY CCPR CCPA (determined
  elements), CLSD+ / CLSD-, FSELECT
* cycles: CYCL DEF 200 201 202 203 204 205 206 207 208 209 240 251 252 253 (Q parameters), 7 (datum
  shift), 8 (mirroring), 9 (dwell), 10 (rotation), 11 (scaling), 12 (PGM CALL), 13 (orientation), 26
  (axis-specific scaling), 32 (tolerance), 247 (presetting); SEL CYCLE; CYCL CALL, CYCL CALL POS, CYCL
  CALL PAT, M99, M89; PATTERN DEF POS ROW PAT FRAME CIRC PITCHCIRC (ROTX ROTY shear)
* parameters: Q0-Q1999 (Q100-Q199 read only), QL0-QL499, QR0-QR499, QS0-QS1999; FN 0 - FN 18 (FN 9-12
  jumps, FN 14 ERROR, FN 16 F-PRINT, FN 17 SYSWRITE ID50, FN 18 SYSREAD), formulas with + - * / ^ ( ) SIN
  COS TAN ASIN ACOS ATAN SQ SQRT LN LOG EXP ABS INT FRAC SGN NEG PI %; QS = "text" / QS / STRING FORMULA
  || / TOCHAR / SUBSTR, TONUMB INSTR STRLEN STRCOMP
* program flow: LBL n / "name" / LBL 0, CALL LBL n REP m, CALL PGM name, SEL PGM / CALL SELECTED PGM,
  SEL CYCLE and CYCL DEF 12 PGM CALL (program as cycle), STOP, M0 M1 M2 M30
* tilting: PLANE SPATIAL / PROJECTED / EULER / VECTOR / POINTS / RELATIV / AXIAL / RESET with MOVE (DIST
  F) / TURN (F) / STAY, MB (MAX / length, with MOVE and TURN) and SEQ+ / SEQ-; M128 F, M129, FUNCTION TCPM,
  FUNCTION RESET TCPM
* spindle and coolant: S, M3 M4 M5 M8 M9 M13 M14; M140 MB distance / MB MAX [F]; M126 / M127; other
  listed M functions TW0012
* ISO: %name G71 / G70 header, N blocks with *, G00 G01 G02 G03 G05 G06 G10-G16 (polar), G17-G19, G24 G25
  (chamfer, round), G28 (mirror), G29 (pole), G30 G31 (blank), G36, G38 (stop), G04, G40-G42, G51, G54
  (datum shift), G62, G72 (scaling), G73 (rotation), G79 (cycle call), G90 G91, G98 L (label), L n,m
  (label call), G99 (tool definition), G200-G209 G240 G251-G253 (cycles), I J K (pole), T, D00-D18, M;
  G26 / G27 and G53: TW0012

### Validity and error

Paths and transformations are exact (the arc end-point correction moves the centre by at most 0.016 mm;
the subproblem-2 solution of PLANE / TCPM is closed form). The cycle sequences follow the TNC 640 cycle
manual's descriptions; the internal distances that manual does not give numerically (disengaging and
chip-breaking distances, plunging angle, helix radius) are the estimates above. Choices: the FK solver
works element by element (an element the TNC determines only from later blocks is TW0040 here);
APPR LN to an inner corner follows the core provider's start-up (FANUC type A); PLANE PROJECTED's X' is
the ZX trace (not in the shared geometry model, which leaves PROJECTED to the controller); the shown
position of an endless rotary axis is taken modulo 360 deg. Not modelled (TW0012, never silent): the
other TNC 640 machining cycles (`tables.TNC640_CYCLES`: SL contour cycles 14, 20-25, 27-29, 39, 270-277,
patterns 220 / 221, 224, 225, 232, 233, 238, 239, 241, 254, 256-258, 262-267, 285-287, 291, 292,
1271-1282, cycle 19 (use PLANE); numbers the control does not have, such as the probing cycles of the
other manual, are 160-000B), FK data relative to other blocks, 3-D radius compensation (LN vectors, DR2),
M120 look-ahead, datum / preset tables, PLANE on head kinematics, FUNCTION TCPM options other than the
defaults, ISO G26 / G27 / G53.

Tests: tests/unit/controller/test_heidenhain.py
(paths, Q parameters, strings, flow, APPR / DEP, FK, PLANE, TCPM, M140, M126, ISO, errors),
test_heidenhain_cycles.py (expected command streams of the cycles, 204, PATTERN DEF, transformations),
test_heidenhain_checkpoint.py (resume inside a CALL LBL REP section bit-identical),
test_siemens_heidenhain_same_part.py.

### References

TNC 640 User's Manuals Klartext (heidenhain-tnc640-klartext-2023), ISO
(heidenhain-tnc640-iso-2023) and Machining Cycles (heidenhain-tnc640-cycles-2023), NC Error Messages
(heidenhain-nc-errors-2023), Setup (heidenhain-tnc640-setup-2023); Murray, Li and Sastry 1994
(subproblem 2); db/controllers/heidenhain-tnc640.yaml.

## Adding a dialect

1. **Package.** `cnctwin/controller/dialects/<name>/__init__.py` (or a module). Add it to
   `hooks.PLUGIN_PACKAGES` only if it is not one of the listed packages (`dialects.siemens840d`,
   `dialects.heidenhain` are already loaded when present).
2. **Builder.** `register_dialect("<name>", build)` with `build(machine: MachineModel, settings:
   ControllerSettings, controller_db: dict | None, db_id: str) -> Dialect`. The name is the machine
   file's `controller.dialect` (`siemens840d`, `heidenhain_klartext`, `heidenhain_iso`).
3. **Dialect fields.** `lexer` (`LexerRules` for ISO-like syntax; a dialect with its own syntax
   parses blocks itself into `Block`s (its own payload in `Block.extra`) and builds the programs with
   `lexer.make_program(file, number, blocks)` and a `ProgramStore`: set `Dialect.parser` to a function
   `(sources, lexer_rules) -> (ProgramStore, problems)` that the Controller calls in place of
   `lexer.build_store`), `gcodes`
   (`GSpec(group, kind, arg, cycle, axis_words)`: the handler dispatches on `kind`; `cycle=True` routes
   to `hooks.CYCLES`), `power_on` (group -> code), `alarms` (`AlarmTable(name, key -> code,
   code -> built-in text, controller_db, db_id)`; use the executor's condition keys listed in
   `fanuc_tables.ALARM_CODES` and `alarms.TWIN_ALARMS`), `sysvars` (an object with `read(ex, n)`,
   `write(ex, n, value)`, `adp(ex, n, value)`; for R parameters or named variables keep the values in
   `PathState.variables[<table>]` lists so they checkpoint), `handler_class`, `params` (resolved
   defaults: start from `fanuc_params(settings, "controller-<name>")` with a
   `db/models/controller-<name>.yaml` holding the same parameter names, or provide every key the
   executor reads: `macro_statement_s macro_term_s call_s goto_scan_s min_block_s horizon_s
   arc_tolerance_m macro_nesting_max sub_nesting_max max_m_codes_per_block first_wait_code
   last_wait_code macro_alarm_base max_macro_alarm same_group_alarm ...`), `supported_g/m` (from the
   controller database's `words`, `base.word_set`).
4. **Handler.** Either subclass `FanucHandler` (ISO word address; override what differs, as
   `HaasHandler` does) or write a class with the methods the core calls: `init_state()`, `on_reset()`,
   `execute(block)`, `plane() -> "XY"|"ZX"|"YZ"`, `feed_mode() -> "per_min"|"per_rev"|"inverse_time"`,
   `absolute() -> bool`, `tool_offset_counts(letter) -> int`, `comp_radius() -> float`,
   `modal_group_value(group) -> float`, `modal_snapshot() -> dict`, `apply_mode(step)`,
   `deferred_m(step)`, and the value helpers providers use (`lin`, `rot`, `integer`, `feed_value`,
   `evaluate(words)`, `find_program(number)`). Keep every piece of state in `ex.s` (`PathState`
   fields, `ex.s.sys` for small dialect items, `ex.s.variables` for tables) and nothing in the handler
   object: it is rebuilt on resume.
5. **Semantics through the core.** Motion via `ex.move/arc_move/machine_move` with program-frame
   counts (`ex.counts(letter, si)`), offsets in `ex.s.offsets`, `ex.s.g52`, `ex.s.g92`, transforms by
   setting `ex.s.mirror/scale/rotation/feature` and calling `ex.invalidate_transform()` and
   `ex.resync_program_position()` (Siemens TRANS/ROT/SCALE/MIRROR frames map onto these; a frame
   chain the four stages cannot express is a transform provider), non-motion actions with
   `ex.emit(Command(...))`, flow with `ex.goto/loop_while/loop_end/request_call/ret/end_program`,
   buffer stops with `ex.wait(kind, ...)` after emitting the SYNC/STOP/PROBE command, path waits with
   `ex.wait("path_wait", code=marker, paths=[...])` after `SYNC {reason: "path_wait", ...}` (Siemens
   WAITM), alarms by raising `ControllerAlarm(key, detail)`, warnings with `ex.note`.
6. **Tests** in `tests/unit/controller/` with inline machine and controller-database fixtures
   (`tests/unit/controller/harness.py` shows the minimal machine definitions), including a
   checkpoint/resume test of the dialect's own state.
7. **The name outside the controller package.** Each of these holds the list of dialect names and
   refuses or mishandles a name it does not have:
   * `cnctwin/db/schemas.py` `DIALECTS`: the machine schema's `controller.dialect` and the controller
     schema's `dialect` accept only these names (`twin db check` rejects a machine or controller file
     with a new name until it is added; `db/_templates/machine.yaml` and `controller.yaml` list the
     allowed values in their comments);
   * `cnctwin/controller/hooks.py` `PLUGIN_PACKAGES`: a new package (not `dialects.siemens840d` or
     `dialects.heidenhain`) must be listed, or its `register_dialect` never runs (step 1);
   * `cnctwin/project/new.py` `program_skeleton` and `PROGRAM_SUFFIX`: the shortest complete program and
     the file extension `twin new` writes for the dialect (without an entry `twin new --machine <a
     machine of the dialect>` fails with "no program skeleton for dialect ...");
   * `cnctwin/api/views.py` `TEXT_PROGRAMS`: the program extensions the API serves as text (the UI shows
     them in place);
   * the static program checks (`cnctwin.sanity`) scan only FANUC and Haas syntax themselves: a program of
     any other dialect gets one `scan-limits` finding and no other rule unless the controller's command
     stream is passed (docs/models/sanity.md section 3).

   [howto-add-dialect.md](howto-add-dialect.md) walks through these steps with the checks to run.
