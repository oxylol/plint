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
  h, d, length_offset_m, radius_offset_m, comp, motion, cycle, css, exact_stop` (+ `offset_number` on
  lathes, `feed_hold_inhibit` with #3004, `rigid_tap` after M29, `thread` for G32/G33,
  `reference_point`, `machine_coordinates`, `tool_preselect` ...).
* `spindle`: `{id, omega_rad_s (M03 < 0), css_m_s, max_rad_s, gear, orient_rad, direction, css_axis}`;
  `coolant`: `{pumps: {id: on}, nozzles: {id: position}, air, code}` (the full state after the change);
  `tool`: `{number, pocket|station, slot (core ToolTable, -1 unknown), length_offset, radius_offset,
  offset (lathe), turret}`; devices (`PALLET`, `DOOR`, `CONVEYOR`, `CHUCK`, brakes, outputs as `MODE`):
  `modal["device"] = {kind, action, code, ...}`.
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
no following error; rapids per axis at the rapid rate, the slowest axis decides; per-revolution
feed with the commanded spindle speed; tool change = machine `tool_change_s`) and answers buffer
stops itself (probe/setter callbacks). Its cycle time is a lower bound of the real one; a feed per
revolution with the spindle stopped ends it with a stall message (the real axes would not move).

### Limitations

* Canned cycles, cutter compensation, TCP/tilted planes beyond Haas G234/G254/G268, probing cycles
  beyond G31/G37 and tool life management belong to the providers of the next agents; until they
  exist the core raises TW0001-TW0004 instead of guessing.
* Single block and operator switches act at the next block the interpreter processes (up to the
  look-ahead depth later than on the control when switched during a run).
* The block-time parameters and the arc tolerances are estimates (above); FANUC alarm texts are
  built-in until db/controllers/fanuc-0i-mf.yaml lists them (Haas texts come from the database
  where present).
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

References: FANUC CNC Functions catalogue (fanuc-cnc-functions-catalog-2024); Smid, *Fanuc CNC Custom
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
* G187 P1/P2/P3 [E]: smoothness level (machine smoothing table tolerance for "G187 Pn") and maximum
  corner rounding E; G187 alone returns to the default (Setting 191, medium; E cancelled).
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

References: Haas NGC mill and lathe operator's manuals (haas-ngc-mill-om-2020, haas-ngc-lathe-om-2020)
as read into db/controllers/haas-ngc.yaml; Smid 2005/2008 for the FANUC-compatible parts.

## Cycles and compensation

(written by the owner of `cnctwin/controller/cycles`, `comp`, `probing`, `toollife`, `restart`, `tcp`)

## Siemens 840D sl

(written by the owner of `cnctwin/controller/dialects/siemens840d`)

## Heidenhain TNC (Klartext and ISO)

(written by the owner of `cnctwin/controller/dialects/heidenhain`)

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
