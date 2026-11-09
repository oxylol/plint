# The run loop (`cnctwin/core/loop.py`, `loop_wiring.py`, `loop_outputs.py`, `check.py`, `warmup.py`)

The whole simulation of a project: every domain stepped by architecture 5.3, the operations of the project in
sequence with their setups and transfers, the parts of its production batch, the run directory of architecture
16 and its derived outputs. `twin run` and the API's `simulate` job call `run_project`; `twin check` calls
`check_project`, `twin warmup` calls `warmup.main`, `twin replay --window` calls `replay_window`.

| file | what |
|---|---|
| `loop.py` | `run_project`, `replay_window`, `check_project` (re-export), `Run` (set-up, operations and parts, the macro step, checkpoints, resume, the end of the run), `LoopParticipant`, `CouplingMonitor` |
| `loop_wiring.py` | the domain objects in dependency order, the protocol adapters between them (`LiveField`, `LiveEdges`, `LiveCatalog`, `LiveBricks`, `SurfaceBurrs`, `LoopResults`) and the run's ledger (`RunLedger`) |
| `loop_outputs.py` | frames.bin layout, scene.json and its meshes, the keyframe payloads, chips' emission rows |
| `check.py` | `check_project`, the L0 stride rule, the ring sizing, the controller's dry run, the planner's cycle time, the cost model |
| `warmup.py` | `twin warmup`: every kernel module compiled in its own process, then two tiny built-in projects through the whole loop |
| `drive.py` | `Feeder` (controller <-> motion protocol), `fixture_pose_F`, `project_variant` (shared); `OperationDriver` (one operation with the hot-loop domains only: the hot-loop tests) |

## 1. Entry points

`run_project(project_path, run_dir, options, progress_cb) -> dict` (the API worker's contract, docs/api.md):
`run_dir` may hold the API's `status.json`, which the loop never writes (it reports only through `progress_cb`,
once per macro step: `sim_time_s`, `est_total_s`, `op`, `part`, `phase`, `message`, and `checkpoint {tick, file}`
after each checkpoint). The progress callback may raise (cancel): the loop lets any exception pass, keeping
`findings.json`, `timeline.json` and the closed series as evidence.

Options: `resume_from` (a committed checkpoint marker: the run continues bit-identically), `replay`
(`{tick0, tick1, checkpoint}`: section 8), `max_sim_s` (stop after this much simulated time: the operation
running is ended through every domain's `end_operation` except inspect's (the unfinished part is not inspected
against the CAD and gets no part files: meshing a fine field at its end is millions of triangles, more than a
1.8 GB process holds next to the run), the remaining ones are not run, run.json says `complete: false` and
`summary.truncated {at_s, reason, op, op_index, part}`), `parts` (overrides
`production.parts`), `checkpoint_interval_s` (default `recording.checkpoint_interval_s` = 10 s), `outputs`
(cnctwin.core.outputs settings for `finalize_outputs`, `false`: none), `sanity` (`false`: no static program
checks at operation start). The return value: `{run_dir, run_id, verdict, complete, summary, cpu_s, outputs}`
(`cpu_s`: process CPU seconds of the hot loop, the slow domains (each, and split in / out of the cut) and the
recording; volatile, never in run.json).

## 2. Wiring

Domains are built and initialised in dependency order (`loop_wiring.INIT_ORDER`): tooling (the tool table and
geometries every other domain sizes by), the recorder (it needs the tool teeth to size its ring), controller,
motion (per operation, below), cutting, removal, dynamics, structure, holding, then the adapters that need an
initialised state (`late_wiring`: wear's and surface's tool geometries, wear's project tolerances, chips' locator
faces of the next setups from holding), thermal, coolant, chips, wear, surface, and inspect when the project has a
CAD. Every adapter reads `run.state` at call time, so a restored state is followed without rewiring.

| consumer | protocol | wired to |
|---|---|---|
| thermal, structure | WorkGeometry | `LiveField`: removal's FieldGeometry; its change tracking (`changed_box_S` per caller) is checkpointed |
| chips | ChipGeometry, ToolCatalog, LocatorFaces | `LiveField`, `LiveCatalog` (tooling.ToolingCatalog: gullet area of all flutes), holding's `locator_faces` |
| surface | WorkpieceQuery, BrickGrid, ToolInfo | `LiveField`, `LiveBricks` (removal's level-0 brick numbering of the field as it is now: a transfer moves the lattice), surface's GeometryToolInfo |
| thermal | ToolEdges | `LiveEdges` (tooling's effective edges) |
| wear | ToolInfo, ToolTolerances | wear's GeometryToolInfo, ProjectTolerances |
| holding | BurrSource | `SurfaceBurrs`: surface's surviving burrs on each locating patch |
| inspect | ResultSource, SurfaceResults | `LoopResults` (removal's mesh of the field = X, structure's grids as cut and released, thermal's unload temperatures), surface's document |
| chips | hot-loop emission rows | cutting's pass rows (`CuttingContact.emissions`: `PASS_COLUMNS` = chips' `EMISSION_COLUMNS`) in chips' EventBuffer |

Defaults kept (each domain records its fallback in its finalize document): holding's PartGeometry (the stock's
shape and mass falling with the removed volume), coolant's WorkpieceBricks (the stock box) and machine meshes,
wear's response (its MechanicsResponse), chips' formation (MechanicsFormation).

## 3. One macro step (`Run.step`, architecture 5.3)

1. `bus_next` = a fresh Bus; physics.resolved strides applied to the paths' current tools (section 9).
2. `HotLoop.run_step(feeder)`: the controller produces commands for every path (the Feeder), motion plans,
   tooling / dynamics / cutting / removal build their frozen kernel inputs, `run_macro` runs in chunks between
   the sync ticks (the controller answers there: retire, feedback, cycle start), drains and field growth at
   splits; motion absorbs after every chunk, cutting, removal and dynamics after the step. The Feeder hands the
   tool numbers whose life the wear domain ended (`bus_prev.wear.expired`) to each path's control as
   `Feedback.tool_life_expired` (inside the next feedback motion sends, or alone when the path is not waiting
   for one: a feedback answers a wait), so its tool-life management calls the sister tool. The hot loop stamps
   each fast ring row with the end tick of its L0 step; the loop moves the step's rows to the last base tick
   inside their L0 step (end - 1), so every row lies in the macro step and L1 tick it was computed in (the
   recorder takes the rows of a step in [tick0, tick1)).
3. chips' absorb of the step's emission rows.
4. The slow domains' macro steps in a fixed order (`SLOW_ORDER`: motion, controller, tooling, cutting, removal,
   dynamics, chips, thermal, coolant, wear, holding, structure, surface). Each reads `bus_prev` and writes its own
   record of `bus_next`; cadences (`every_n_macro`) are each domain's own: thermal's FV step, structure's refresh
   jobs (latency 50 steps), coolant's occlusion jobs, cutting's coefficient pipeline; chips, wear, holding,
   surface, dynamics and motion run every step by their schemas. `physics.lag_macro: 2` is computed as lag 1 (the
   slow domains of a step run after its hot loop; nothing runs concurrently).
5. The coupling-lag monitor (section 6), the conservation ledger (`Ledger.check`), the exact-volume check, the
   cutting energy identity (reported).
6. Recording: every domain's channel values split by their registered tier into `record_tick` (one row per L1
   tick stamped with the tick's start: a row covers [t, t + ipo) and its fast rows are its envelope; motion's
   PathFrame ticks are the ticks' ends) / `record_macro` (at the step's end), `end_step` (fast windows cut from the
   ring, tick envelopes), `frames.bin`, the timeline (motion's
   records with operation-local indices copied into the run's timeline with global ones), `findings.json` when
   the findings changed.
7. The clock advances, the bus swaps, a keyframe is written when due, a checkpoint when due.

## 4. Operations and setups

`start_operation`: for every operation after the first, the controller's `start_operation` (a new program;
commons #500+ and part counters carried), a new motion domain with the operation's fixture and stock poses (the
fixture pose of a mill converted from machine coordinates by `drive.fixture_pose_F`) that starts where the last
program left the machine (`Run._continue_axes`: the drives at rest at the axis positions the last operation
ended with, the planners starting there, the spindles at rest; the operator turns the part, the axes do not
move. Started at home instead,
the axes would jump in one tick, which holding reads as a table acceleration of about 1e5 m/s2: the part
"slips" by the jump and comes free), a new hot loop and feeder; the timeline offset, the exact-volume baseline and `clock.op_start_tick`; sanity's
`check_operation` of the operation (first part only; static findings dated at the operation's start); an
operation-end checkpoint of the previous operation.

`end_operation` calls every domain's `end_operation` in `END_ORDER`: motion (closes the open timeline records),
controller (carries commons), cutting, dynamics, chips (books the last window, drops flying chips, measures the
debris on the next setup's locators, cleans the part), coolant (tank idles for the time between operations),
wear, holding (unclamp), thermal (unload temperatures, cool-down for `environment.time_between_ops`), structure
(the released state: unclamped, 3-2-1 support, spring-back and distortion; the clamped grids are captured first
for inspection), surface (burr survival, finish findings, threads), inspect (the part meshes `part_<op>.glb|stl|3mf`
at 20 degC and `part_<op>_unload.stl` at the unload temperature, inspection records, the next stock), removal
(the transfer to the next operation by `from_previous`, docs/formats.md, or the remaining-material check after
the last), tooling. Without a CAD the loop writes the part meshes itself through `LoopResults`. Then the next
operation, the next part, or the end of the run.

An operation ends when every path has run its program (and the parts of a one-operation batch) or stopped with
an alarm (`Feeder.finished`): an alarm stops its path's motion as on the machine, commands queued before it do not
execute (one of them may wait for the control, which answers nothing in alarm; examples/sanity-mistakes stops at
its line 16 after 7 ms). The run stops there too: no next operation or part runs on a machine in alarm (run.json
`summary.stopped {at_s, reason: alarm, op, op_index, part, alarms [{path, alarm}]}`, trace `loop:stopped_by_alarm`;
`complete` stays true: the run reached the machine's stop). Before, examples/batch-50 ended and restarted its
operation in every macro step after its probe's alarm, writing the part files 50 times.

The time between operations is not simulated time: the next operation starts at the tick the previous one ended
(thermal's and coolant's cool-down run on their own clocks; production's operator model books the handling).

## 5. Production batches

`production.parts` (or `options.parts`) parts, each the sequence of operations. A one-operation batch runs on
one control and machine as the real one does: after M30 the operator loads the next blank and presses cycle start
(`ControllerDomain.cycle_start`): the control keeps its work and tool offsets, variables and tool-life counters,
the axes stay where the program left them, motion's timeline goes on. The rest of the macro step after M30 is
the machine waiting for that cycle start: it is booked as `program_stop` into the next part's first block
(motion books waits into the block that follows them), so the path's records stay contiguous and the time
balance holds over the parts. A batch of several operations loads each
operation's program (commons and part counters carried). A new part starts from a fresh blank (removal re-initialised from the project stock; the field version continues past the last part's so
chips, thermal and surface see the change) and carries everything a real batch carries: the tools' wear, damage,
counters and life (wear, tooling's effective edges, sister tools), the controller's commons and part counters,
the machine, spindle, axis, structure and coolant-tank temperatures (thermal's and coolant's states; the
workpiece grid is reset by thermal when `clock.part_index` changes), the chips conveyed and in the pan, holding's
tool clamps. production.json is composed at the end from the run directory (`production_from_run_dir`: the
timeline per part, frames.bin, tools.json with its per-part rows, the removed mass per part from the ledger).

## 6. Checks of the run

* **Conservation** (`core.ledger.Ledger`, architecture 12), every macro step: mass (removed + detached = six chip
  inventories + displaced + detached, at chips' booked tick, `tol_mass`), energy (shaft + feed = heat by zone +
  ploughing + modal change + damping + chip kinetic + friction + cold work + unbalance work, `tol_energy`), time
  (block durations = each path's span, integer ticks; motion's timeline of the operation), thermal solvers.
  A violation adds a fatal finding and fails the run. After the last operation the whole run's mass balance is
  checked once more (`summary.final_mass`): every chip is booked then. A run that stops before chips books its
  first window (an alarm in the first steps) checks no energy interval: with no energy booked at all its summary
  says so and counts as balanced.
* **The run's ledger** (`loop_wiring.RunLedger`: the core Ledger, its alignment, failure and checkpoint
  behaviour, with two terms of the integrated run):
  * `modal_quadrature_J` (out): cutting books the work of its forces on the deflection rates by its own
    quadrature (impulse x predicted rate = dynamics' `cutting_work_quadrature_J`), the modes receive the exact
    first-order-hold work (`cutting_work_J`, inside the modal energy change, the damping and the unbalance term).
    They differ by 0.1-3 % of the vibration work (docs/models/dynamics.md "Ledger note"); at the tiny project's
    entry the vibration work is 0.36 % of the shaft work and the difference 2.2e-6 of the interval's terms, above
    the 1e-6 tolerance. Dynamics computes both; their difference is carried as its own term (dynamics' documented
    remedy), so the balance closes to round-off.
  * chip kinetic energy: chips books the kinetic energy of the chips at separation (`chips.kinetic_energy_J`).
    cutting's force model has no momentum flux, so its shaft work does not contain that energy (about 1e-5 of
    the shaft work at aluminium speeds) and the out term alone would break the 1e-6 balance by that much. With
    `loop.chip_kinetic_in_balance: 0` (db/models/loop.yaml) it is reported (`ledger.chip_kinetic_reported_J`,
    run.json `summary.conservation.energy.chip_kinetic_reported_J`) and kept out of the balance; 1 restores the
    architecture's out term once cutting's torque carries the chip momentum force.
* **Friction**: cutting computes the work its chip recut impulses and packing loads take from the tool
  (`CuttingContact.recut_J`, `packing_J`, terms of its own energy identity). Cutting's `_ledger` books them into
  `SimState.ledger.friction_J` itself since 2026-11-09; earlier builds wrote no ledger field for them. The loop
  books them only when the friction term did not change over the hot loop (`Run._book_friction`): both together
  counted them twice, 0.4 % of examples/two-setup-flip's facing entry (0.02 J of 5.3 J, a fatal energy finding).
  Holding's slip friction stays reported only (holding's documented choice).
* **Exact volume**: per workpiece and operation the removal records sum to the field's own volume change
  (integer quanta, 1e-9): a fatal `mass_balance` finding otherwise.
* **Cutting's energy identity** (shaft + feed = heat by zone + ploughing + recut + packing + work on the
  deflection rates) is reported as `summary.cutting_identity_rel`.
* **Coupling-lag monitor** (architecture 5.3): frozen quantities handed across the one-step stagger (tool rake
  temperatures, coolant friction factors, holder pull-out, spindle growth) whose successive changes alternate in
  sign with a ratio above 0.5 for 3 steps (each change above 1e-3 of the value; db/models/loop.yaml) raise a
  warning (category sanity, domain loop) with a `loop:coupling_lag` trace event.

## 7. Run directory

`RunDirectory.prepare()` (status.json kept: the job runner owns it), `trace.jsonl` (TraceLog), `channels.json`
(every domain's channels, the ledger's, the hot loop's fast channels), `scene.json` + `scene/*.glb` (the stock as
set up in S: a block as its exact 12-triangle box, other stocks as the field's mesh; the fixture bodies of every
operation in S, each rotating tool's envelope in T, gravity, the
frames.bin header, the brick layout), `frames.bin` (axis origins: the values that put the spindle nose on the
stock origin), `keyframes/` (keyframe 0 at tick 0 and every `recording.keyframe_period_s`: removal's dirty bricks
re-meshed, coarse fields `work_temp_K`, chip pile, wetted cells, coolant jets, tool deflection, part pose,
`pose_M_S`; chips' particles in `chips/kf_<k>.npz`; a full snapshot at the end), `series/` (the recorder),
`timeline.json`, `findings.json`, `checkpoints/`, the part files of each operation (a keyframe's bricks are
meshed 256 at a time and kept as float32 / int32 until the writer quantises them: a full snapshot of
examples/pocket-islands-al6061's 0.075 mm field is about 9.5 M triangles), and at the end
`inspection.json` (inspect's document, or a note when the project has no CAD), `surface.json`, `tools.json`
(wear's document with per-part rows), `production.json`, `sld/T<n>.json` (dynamics), `run.json`
(`run_document`: resolved project, versions, input hashes, numba cache key, threads, clock with the L0 strides
each operation used (`l0_ticks: {op: ticks}`, a list when it used several: the report's, the UI's and the
synthetic run fixture's form), verdict, summary, every domain's finalize document under `domains`, the database estimates of the objects
used, the output index), then `finalize_outputs` (docs/outputs.md).

## 8. Checkpoints, resume, replay

A checkpoint is due every `checkpoint_interval_s` at a macro boundary, and is taken at every operation end
(when the next operation is set up, and after the last one; kept by the pruning). Participants: the recorder,
frames.bin, keyframes, the run's timeline, the run directory (late outputs), the ledger, inspect, and the loop
(`LoopParticipant`: operation and part, the feeder, the exact-volume baseline, the timeline offset, the hot
loop's arrays, `LiveField`'s change tracking, the controller's carried commons, the coupling monitor, the per-part
masses, dynamics' trace ids of its last modes events). cutting's coefficient-model caches go into its state
(`model_checkpoint` / `model_restore`).

Resume (`options.resume_from`): everything is built as for a fresh run on a scratch state whose trace goes
nowhere (each domain's init sets up its domain object), then `Checkpointer.resume` restores the state and every
participant and truncates every output after the cut; the operation's motion domain is rebuilt with its poses
and attached, the controller rebinds, the hot loop restores its arrays. The result set of a resumed run equals
the uninterrupted run's byte for byte (test).

Replay (`options.replay`, `replay_window`): the run is restored from the checkpoint without truncation (or
started from the beginning), domains write no files (`ctx.run_dir` None, an in-memory trace), the ledger is off,
the recorder in replay mode writes only `series/fast/replay-<tick0>-<tick1>.parquet` and its index line.

## 9. `twin check`

`check_project(request, out_dir, progress_cb)`: per operation the controller's dry run (cycle-time lower bound,
alarms, the tools and spindle speeds of the feed moves) and the planner's cycle time (motion's MotionRunner); the
L0 stride of every tool by architecture 5.1 at the programmed speed times the machine's largest spindle override
(the CSS clamp for G96), with f_max the highest frequency among structure's tool modes and the machine's frame and
table modes carrying >= 1 % of the tip compliance (`stride_compliance_share`); the lattice rule from sanity's
engagement estimate; the estimated CPU seconds per simulated second (cost model in db/models/loop.yaml: per L0
step a fixed part, per engaged element, per mode; per L1 tick; per macro step in and out of the cut; in-cut time =
the dry run's feed-move time, an upper bound) and the disk `fast: all` would take; sanity's findings. `--write`
(no errors only) stores `physics.resolved` with `project.write_resolved_physics` (each segment: tool, path,
l0_ticks, teeth, the binding rule `angle` | `tooth` | `mode` | `ipo` of the project schema, max_rpm and f_max_Hz
when positive); a run then uses those strides (`Run._apply_resolved_strides`). `check.json` in `out_dir`; exit
code 1 on errors. The API's `check` job calls it through `cnctwin.project.check`.

## 10. `twin warmup`

Imports every kernel module, each in its own Python process (explicit signatures compile at definition; Numba
keeps the machine code and LLVM state of all it compiled resident, and every module in one process outgrew the
1.8 GB cap of bin/tsafe: killed during the first cold warmup), then runs two tiny built-in projects through the
whole loop, each in its own process (a side step 0.5 mm wide and 1 mm deep along 12 mm of a 30 x 20 x 10 mm Al block on the
VF-2SS, led out sideways before the rapid retract; a 3 mm OD pass on a d30 bar on the ST-20Y), so the first real
run finds every kernel in the cache. The slow domains' heavy L2 analyses run on coarse settings
(`warmup.WARM_PHYSICS`: 64-point lobe scans, two process-damping iterations, structure's voxel model at 3000
DOF, cutting's coefficient pipeline every 50 steps, thermal 4 mm cells): warmup compiles kernels, it does not need
their answers. The mill project is a side step, not a slot: a full slot 1.5 mm deep is about 30 times the D10's
predicted stability limit (0.044 mm at 10 000 rpm) and the simulated chatter grows until the tool breaks
(section 15).

## 11. Changes in domain packages (additive, documented)

* `surface/params.py` `table_value`: a Sourced scalar (`{value, src}`) was taken for a table because Sourced is
  callable; the material's density made surface's init fail on every real material. The Sourced's own value is
  tested now.
* `thermal/ledger.py` `SolverLedger.book(..., content_J=0)` and `thermal/workpiece.py` `fv_step`: the workpiece
  solver books internal energies relative to 20 degC computed from absolute temperatures; a workpiece sitting at
  20 degC (scale near zero) failed its ledger on round-off (3.5e-13 J against 1.6e-15 J in the first FV step of
  every run). The heat content above 0 K (mass x cp x mean temperature) is added to the scale, so the check does
  not depend on the reference temperature.

* `sound/inputs.py` `_sourced`: run.json's resolved project writes property tables as `{x, x_unit, values}`;
  sound read only `{value}` and `{table}` forms and failed on the work material's Young's modulus (every audio
  output of a real run). Such a table now gives its first point.
* `project/check.py` (new): the API's `check` job entry of docs/api.md (`cnctwin.project.check:check_project`)
  did not exist; it re-exports `cnctwin.core.check.check_project`.
* `core/drive.py` `Feeder` (the loop's and the hot-loop tests' controller-motion protocol): expired tools are
  handed to the control as `Feedback.tool_life_expired`; `finished` ends an operation whose paths stopped in
  alarm without waiting for motion's queue (section 4).

* Adapters for interface mismatches found in integration, kept in the loop (no change in the owners' files):
  chips reads `ToolInfo.hole_making`, a property of chips' own ToolInfo that tooling's ToolInfoView lacks
  (`LiveCatalog` hands chips its own record built from the view); cutting's `channels.macro_values` reads
  `CuttingContact.h_max_m` as a number while the bus field is the per-element array (the loop hands it the
  contacts with the array's peak when the call fails: `Run._cutting_values`).

## 12. Tests

`tests/integration/test_loop_*.py` (one process per file, `bin/tsafe`, `TSAFE_MEM=1800M`; projects in
`tests/integration/loopfix.py`):

| test | what |
|---|---|
| `test_loop_run.py` | the tiny project (warmup's: a side step 0.5 mm wide, 1 mm deep along 12 mm of a 30 x 20 x 10 mm Al block, D10 at 10 000 rpm) through `run_project` with every domain on and small outputs: complete, safe, every ledger passes (mass 1e-9, energy 1e-6, time exact, end-of-run mass), the removed volume is the volume the tool path recorded in frames.bin sweeps (3 %), every file of the run directory exists and is indexed, report / pictures / video / sound made, frames / timeline / production / tools agree, every domain wrote its channels and trace events |
| `test_loop_determinism.py` | two runs give the same `result_set_digest`; a run cancelled from its progress callback after its second checkpoint (2 s, before the cut) and resumed from it gives the same digest; a replay of a recorded fast window reproduces its rows exactly (every column) |
| `test_loop_flip.py` | (xfail on model errors, section 15) examples/two-setup-flip with one side step per operation: OP20 cuts the OP10 part turned 180 deg about X (one removal transfer, the OP10 step ends up at the bottom of OP20's S), both steps' volumes against each operation's recorded path, timeline and ledgers over both operations, operation-end checkpoints, part files per operation |
| `test_loop_batch.py` | (xfail on model errors, section 15) three parts of the tiny project: fresh blanks, the same removed mass, tools.json per-part rows with rising cut time, VB and life used, production.json with three parts, thermal unload records per part, the time balance over the parts |
| `test_loop_check.py` | `check_project`: strides by the 5.1 rules, cycle times, estimates; `--write` stores physics.resolved and a run uses those strides (and `max_sim_s` truncates it); a project with errors is not written |
| `test_loop_parts.py` | the loop's pieces without a simulation: the coupling monitor, the ring restamping, the peak view, the run ledger's kinetic-energy report, quadrature term, energy scale floor and the summary of a run without energy, friction booked once, the adapters' change tracking through a checkpoint |
| `test_outputs_studies.py` | the simulate entry resolves to this loop; replay windows and comparisons through cnctwin.core.outputs |

`tests/regression/test_examples.py` (marked slow, `bin/test --all regression`): one test per example (section 14).

## 13. Measurements

This box (4 cores, 7.7 GB, shared with two other agents' jobs and the home server's services), Python 3.12,
`bin/tsafe` (nice 10, one thread, 1.8 GB cap), the Numba cache warm unless said. "Live tree": a frozen copy of
the working tree of 2026-11-09 03:39 (every agent's files at that moment) with the loop's files; "HEAD tree":
git HEAD d91a7ee with the loop's files.

**warmup's mill project** (the tiny project of the tests; 6.06 s simulated, 0.51 s of it in cut, 606 macro
steps), live tree, outputs off:

| | CPU s | per step |
|---|---|---|
| whole run (wall 45.6 s) | 45.1 | |
| hot loop, steps in cut (51) | 3.0 | 59 ms (2000 L0 steps of 5 ticks: 29 us each) |
| hot loop, other steps (555) | 5.2 | 9.4 ms |
| slow domains, steps in cut | 10.2 | 199 ms |
| slow domains, other steps | 23.0 | 41 ms |
| recording (recorder, frames, timeline, findings) | 1.0-1.2 | 2 ms |
| set-up (database, resolve, domain inits, kernel loading, scene, keyframe 0, dry-run estimate) | about 5 | |

Slow domains over the run: dynamics 15.4 s, coolant 5.0, cutting 3.8, structure 3.3, chips 2.3, thermal 1.5,
holding 0.6, surface 0.6, removal 0.2, wear 0.15, tooling 0.08, controller 0.02, motion 0.01. In the HEAD tree
dynamics' process-damped stability limit cost 25 s per evaluation (the fixed point does not converge in its 30
iterations: 127 lobe scans for 6 evaluations, 154 s of a 200 s run); warmup's physics caps it at two iterations,
and the live tree's dynamics (cached tooth means, stacked eigenvalues, diagrams deferred to the operation's end)
is about ten times faster. `twin check` estimated 2.3 CPU s per simulated s with the cost model as first written
(the domains' own cost notes): the run took 7.4. The cost model is now the measurement above
(db/models/loop.yaml): about 9 CPU s per simulated s for this project (the dry run's feed time, 1.06 s, bounds
the time in cut from above).

Conservation over that run: mass residual 4.5e-16 (largest step), energy 4.3e-9 (largest interval, 605 checked;
the modal quadrature term, -1.8e-4 J over the run, closes the entry; the chips' kinetic energy reported, 1.8e-4 J),
time exact; cutting's own identity 4e-10. Removed volume 6.106 mm3: the swept volume of the path frames.bin
recorded, sampled on the field's rays, is 6.105 mm3; cutting's analytic chip volume 5.43 mm3 (section 15).

**Tests** (wall, one process each): live tree `test_loop_parts` 0.7 s (9 passed), `test_loop_check` 41 s (3),
`test_loop_run` 2 min 28 s (6; with every derived output: report 76 PDF pages, pictures, a 192 x 108 video,
sound), `test_loop_determinism` 3 min 2 s (3; four runs and a replay), `test_outputs_studies` 5 s (3),
`test_loop_flip` 64 s and `test_loop_batch` 53 s (expected failures, section 15); the same in the working tree of
07:10 (every agent's files then); HEAD tree `test_loop_run` 3 min 54 s, `test_loop_determinism` 8 min (its slower
dynamics). The hot loop's tests that use `drive.py` (test_slot, test_turning, test_drill, test_chatter,
test_determinism_checkpoint, test_free_dynamics, test_signatures: 19 tests) pass on the live tree.

**Cold compile**: the first `twin warmup` in an empty cache compiled every kernel module in 13 min 59 s in one
process and was killed by the 1.8 GB cap in the mill run that followed (hence one process per module now); the
second, with the cache filled, took 2 min 56 s (mill run 51.6 s; the lathe run stops, section 15).

## 14. Regression

`tests/regression/test_examples.py` runs every example (and a no-probe variant of the two that probe) through
`run_project` with every domain on and outputs off, `max_sim_s` 8 s (the set-up, the first tool change and the
first cut of each program: the whole loop costs 7-60 CPU s per simulated second on these projects, so 20 s per
example, the first choice, would have taken the box most of a day), one pytest process per case under
`bin/tsafe` with `TSAFE_MEM=1800M`; it asserts that the run
completes and conserves, and records each EXPECTED.md item it can read as passed, failed or not reached
(tests/regression/expected.py; `<basetemp>/regression/<case>.json`). Results on the live tree (section 13), by
case:

| case | wall s | sim s | run | verdict | findings | EXPECTED passed | failed | not reached | conservation | error |
|---|---|---|---|---|---|---|---|---|---|---|
| batch-50 | 511 | 4.94 | stopped by alarm at 4.94 s | not safe | error 1, info 2, warning 2 (sanity 3, alarm 1, remaining_material 1) | no spindle overload finding | - | T1 roughing power | ok |  |
| batch-50 (no-probe) | 367 | 4.22 | stopped by the error at 4.22 s | - | error 2, fatal 1, info 1, warning 5 (sanity 2, chatter 1, thermal_damage 1, coolant_shadowing 1) | no spindle overload finding | - | T1 roughing power | {} | ConservationError: energy balance violated at t=4.220000 s: expected 46.7058515, got 46.7061528 (relative tolerance 1e-06) |
| blade-5axis-ti | 666 | 8.00 | first 8.0 s of 113 s (dry run) | safe | info 2, warning 1 (sanity 2, remaining_material 1) | no spindle overload finding; machine travel inside the UMC-750 limits: no overtravel finding; no pass through B = 0 (no C singularity finding) | - | - | ok |  |
| die-h13-ballnose | 891 | 8.00 | first 8.0 s of 1155 s (dry run) | not safe | error 4, info 3, warning 1 (part_slip 3, sanity 1, program_error 1, part_free 1) | no spindle overload finding; travel inside the limits (Z margin 6 mm): no overtravel finding | - | T1 roughing cutting power; T1 mean tangential force | ok |  |
| drill-tap-plate-1045 | 310 | 2.51 | stopped by the error at 2.51 s | - | info 2 (sanity 2) | no spindle overload finding | - | drilling torque, D6.8; drilling thrust, D6.8 | {} | ValueError: cannot convert float NaN to integer |
| pocket-islands-al6061 | 698 | 6.11 | stopped by alarm at 6.11 s | not safe | error 1, info 1, warning 3 (sanity 3, alarm 1, remaining_material 1) | no spindle overload finding | N10 probe Z set: G31 triggers without alarm (1) | peak spindle cutting power in the trochoid loops | ok |  |
| pocket-islands-al6061 (no-probe) | 393 | 4.50 | stopped by the error at 4.50 s | - | fatal 1, info 1, warning 5 (sanity 4, chatter 1, coolant_shadowing 1, energy_balance 1) | no spindle overload finding; N10 probe Z set: G31 triggers without alarm | - | peak spindle cutting power in the trochoid loops | {} | ConservationError: energy balance violated at t=4.500000 s: expected 0.0923155028, got 0.0923164222 (relative tolerance 1e-06) |
| sanity-mistakes | 163 | 0.01 | stopped by alarm at 0.01 s | not safe | error 14, warning 11 (sanity 23, alarm 1, remaining_material 1) | the 23 sanity findings of expected_findings.yaml; the controller stops with alarm 320 at line 16 | - | - | ok |  |
| thin-wall-al7075 | 432 | 4.46 | stopped by the error at 4.46 s | - | error 3, fatal 1, info 1, warning 4 (tool_pullout 2, sanity 1, chatter 1, coolant_shadowing 1) | no spindle overload finding | - | peel cutting power (T1) | {} | ConservationError: energy balance violated at t=4.460000 s: expected 17.6317711, got 17.6313611 (relative tolerance 1e-06) |
| ti-slot-throughtool | 10 | 0.00 | stopped by the error at 0.00 s | - |  | - | - | - | {} | AttributeError: 'str' object has no attribute 'get' |
| two-setup-flip | 353 | 3.82 | stopped by the error at 3.82 s | - | error 2, fatal 1, info 1, warning 3 (sanity 1, chatter 1, coolant_shadowing 1, tool_end_of_life 1) | no spindle overload finding | - | OP20 runs on the OP10 part turned about X | {} | ConservationError: energy balance violated at t=3.820000 s: expected 9.11978149, got 9.11993034 (relative tolerance 1e-06) |

lathe-shaft-4140: stopped by hand after 28 min of wall time at 2.4 s simulated, before its first contact (budget
4 s). In those 2.4 s structure set the work model up 55 times and dynamics rebuilt its modes 53 times (trace
`structure:work_setup`, `dynamics:basis_change`): the rotating bar in the chuck is set up again every few macro
steps, each an FE solve of the bar and a new modal basis. Warmup's lathe pass blows up at its first contact
(section 15).

Where the runs end (11 of 12 cases recorded):

* **Stopped by the program's alarm, conserving** (3): examples/sanity-mistakes at line 16 after 7 ms (both
  EXPECTED items pass: the 23 sanity findings exactly, alarm 320 at line 16), examples/pocket-islands-al6061 and
  examples/batch-50 at their probe (alarm 379, M79 after a G31 that did not trigger: the probe's EXPECTED item
  fails, the hot loop has no stylus model).
* **The budget ran, conserving** (2): examples/blade-5axis-ti (8 of 113 s, safe, its three EXPECTED items pass:
  no spindle overload, no overtravel on the UMC-750, no pass through B = 0) and examples/die-h13-ballnose (8 of
  1155 s; the part slips three times in its holding and comes free before the first cut; its two "no finding"
  items pass, the power and force items are not reached).
* **A model error at the first cut or the set-up** (6): an energy imbalance of 3e-6 to 1.2e-5 at the entry of the
  first cut (examples/thin-wall-al7075 after its holder turns 792 deg and the tool pulls out, two-setup-flip after
  its part slips 38 mm in the vise, the no-probe variants of pocket-islands and batch-50; the terms are in each
  record's fatal finding, the residual is about 0.1 % of the modes' work); thermal's NaN flank temperatures into
  cutting's coefficient pipeline (drill-tap-plate-1045 at its first tool, 2.5 s); thermal's machine model at the
  set-up of the EC-400 (ti-slot-throughtool: an axis entry of the machine file is a string, `axis.get`).

No case reaches a second operation or part within its budget, so the regression does not show the transfers and
carry-overs (tests/integration/test_loop_flip.py and test_loop_batch.py do, section 12). EXPECTED items over the
11 recorded cases: 15 passed, 1 failed (pocket-islands' probe: alarm 379), 10 not reached (beyond the budget, or
a band the run missed because it stopped at the cut's entry: thin-wall's peel power was 1.76 kW against 2-4 kW,
batch-50's roughing power 4.6 kW against 9-15 kW, pocket-islands' trochoid power 13 W, each when its run
stopped). The no-probe variant's "probe triggers without alarm" passes trivially: its probing section is cut.

## 15. Open problems

* **Cost.** The tiny project costs 7.4 CPU s per simulated second (section 13): the hot loop 6 s per second in
  contact, the slow domains 20 s per second in contact (199 ms per macro step) and 4 s per second out of it
  (41 ms per step; dynamics 25 ms of it on average): a real program of minutes takes hours. The regression tests
  run a representative first part of the long examples (section 14).
* **Probe.** The hot loop's G31 sweep needs a stylus profile no domain builds (docs/models/hotloop.md "Open
  problems"): programs that probe (examples/pocket-islands-al6061 N10, examples/batch-50 on every part) stop at
  the first probing alarm (the probe macro's or M79), and the run stops there (section 4). The regression runs
  them as written and as variants without the probing section (the work offset set to what the probe would have
  measured).
* **A turned part is set up again and again** (structure, holding, dynamics): in examples/lathe-shaft-4140 the
  bar's work model is set up 55 times in 2.4 s with as many new modal bases (section 14): the lathe's loop runs
  at 0.02-0.4 simulated s per wall minute.
* **Memory at an operation's end.** inspect's part meshes come from the field's mesh at its lattice
  (`LoopResults.result_part`: removal's `mesh_level`, then inspect's displacement grids per vertex and the STL,
  GLB and 3MF files): about 9.5 M triangles for examples/pocket-islands-al6061 (0.075 mm on 100 x 80 x 30 mm).
  That end killed its first regression runs at the 1.8 GB cap (exit 137) when its probe alarm ended the
  operation; unfinished operations are not inspected now (section 1), but a finished operation of a
  fine-lattice example still needs more than 1.8 GB there.
* **Chip kinetic energy** is reported, not balanced, until cutting's torque carries the chip momentum force
  (section 6).
* **Transfers and the slow domains' grids.** removal transfers the material to the next operation's stock frame
  (`from_previous`); thermal's workpiece grid and structure's work grid stay in the frame they were built in
  (thermal refreshes its fill fractions from the field, so its cells see the turned part's material, but the
  temperatures are not turned with it; structure rebuilds its setup when holding re-places the part). After a
  flip the workpiece temperatures of the first operation are misplaced by the turn (the part has cooled for
  `time_between_ops` by then).
* **Defaults where no adapter exists yet**: holding's PartGeometry (the stock's shape; mass falling with the
  removed volume), coolant's WorkpieceBricks (the stock box), inspect's FieldColumns (MeshColumns dexelises the
  result mesh on a 0.25 mm lattice instead of reading removal's columns).
* **Channel names other readers expect.** cnctwin.sound and cnctwin.render read `cutting.force_N` from the fast
  tier; the loop's fast channels are the hot loop's (`hotloop.force_T0_N`, `hotloop.torque_Nm`) and dynamics':
  the sound's cutting source is silent (its notes say so). cnctwin.production reads `cutting.spindle_power_W`
  from the tick tier; cutting records it at the macro tier: production.json lists the spindle cutting power as
  an assumption (0).
* **lag_macro 2** is computed as lag 1 (nothing runs concurrently).
* **Numba cache.** The cache directory is keyed by every kernel source: a change in any kernel file costs the
  next run a cold compile of every kernel (12-14 minutes on this box under load).
* **Found in integration, for the domains' owners** (each reproduced with the tiny project, a variant of it, or
  an example of the regression; the loop works around none of them):
  * *Cuts go unstable at entry in the whole loop* (dynamics, structure, cutting, holding): the time-domain
    chatter detector reports a low frequency at the first contact (65 Hz on the tiny project's side step and
    slot, 67 Hz on examples/thin-wall-al7075, 85 Hz on examples/two-setup-flip's facing), then the forces run
    away: 16 800 N m spindle torque on the tiny slot, 169 N m on thin-wall's D10 (its holder turns 792 deg and
    pulls out 0.23 mm), the flip's part slips 38 mm in the vise, the lathe's tool breaks. The energy balance then
    fails by 1e-5 to 1e-3 (the slip and pull-out work is in no ledger term) or a NaN stops a domain. The hot
    loop's own tests (test_slot, test_chatter, test_turning: dynamics with its own tool and spindle modes, no
    structure, holding or thermal) are stable on the same tree, so the coupling with the slow domains' modes and
    loads is the suspect.
  * *Full-slot blow-up* (dynamics, cutting, structure): warmup's first program, a 14 mm full slot 1.5 mm deep
    with the D10 (32 mm stickout) at 10 000 rpm in Al 6061, is about 30 times the stability limit dynamics
    predicts (0.044 mm). The chatter (detected at 65 Hz at entry) grows without the saturation a tool leaving
    the cut gives: within 0.1 s spindle torque 52 N m, then 16 800 N m, tool breakage, holder
    pull-out, part free, a surface temperature of 4.6e7 degC, tool heat 5.8e5 J in one step, and thermal's tool
    solver ledger fails (relative 2e-8). The tests and warmup cut a side step instead (0.5 mm x 1 mm, stable
    enough to finish). Structure's tool modes for that tool look odd too: the 5339 Hz pair has a larger tip
    modal compliance (1.4e-7 m/N) than the 697 Hz fundamental pair (3.6e-8 m/N).
  * *Cutting's analytic chip volume* is 11 % below the volume the tool path sweeps on the side step (ae/D 0.05,
    ap 1 mm): 5.43 mm3 against 6.05-6.11 mm3. Removal's field agrees with the swept volume of the recorded path
    to 0.02 % (its rays sampled on the same lattice), so removal's pass check ("field volume and analytic chip
    volume differ by 12.5 %, allowed 4.6 %") reports cutting's shortfall.
  * *Rapid into material at zero depth* (removal): a G00 retract straight up from the end of a side cut raises
    the error `rapid_into_material` with "0.000 mm deep, 0.000 mm^3" (any removal during a rapid counts, here
    the servo lag of the feed axis catching up while Z already rises). Real programs retract like that; the
    tests lead out sideways first.
  * *Slivers* (removal): the side step leaves 4-quantum (40 nm) pieces of material floating in rays it cut (ray
    segments like (972732, 972736) at z 9.602 mm above a floor at 8.993 mm); `surface_height` then reports the
    sliver's top as the surface.
  * *Staggered thermo-mechanical coupling* (cutting, thermal): the tool's rake temperature alternates by 20-37 K
    from one macro step to the next while cutting; the loop's monitor and cutting's own check both report it.
  * *Turning blows up in the whole loop* (cutting, dynamics, structure, thermal; the live tree; the hot loop's own
    tests/integration/test_turning.py, without the slow domains, passes there):
    warmup's OD pass on the d30 4140 bar (CNMG, 1400 rpm, ap 1.5 mm, f 0.25 mm/rev, and also ap 0.5 mm, f 0.15)
    gives a surface temperature of 1700-3600 degC at the first contact, then within 50 ms axis drive forces of
    20-200 kN, tool breakage (bending 20 GPa at the shank), part slip of 66 m, part free, and dynamics stops the
    run ("path 0 was stepped by 0 base ticks"; or a NaN conversion). `twin warmup` reports the lathe run as
    stopped (the kernels it reached are cached) and exits 1.
  * *Thermal's tool field after a tool change* (thermal): in a batch of the tiny project the cutting-region field
    of T1, built in part 1's cut, diverges to NaN in its first solve after part 2's T1 M06 (the same tool;
    "tool.0 field solve, residual nan after 5000 iterations"), with finite inputs (the smallest cell capacity is
    2.8e-10 J/K: 2 um rings); the same in the flip test at OP20's T2 M06 (T2 cut in OP10).
  * *NaN states across part and operation transitions* (thermal, structure, cutting, dynamics): with the tool
    field off the batch and the flip stop further on: in the tree of 03:39 thermal's flank temperatures of T1
    (`flank_temp_q_K`) are NaN for elements in cut at the second part's first coefficient update (cutting's
    `quantised_state`: "cannot convert float NaN to integer"; with or without the tool field), in the git-HEAD
    tree structure's work modes of the re-seated part carry NaN rows into dynamics' `build_system` (the part,
    mislocated by the chips below, is not seated). `test_loop_batch.py` and `test_loop_flip.py` mark a model
    error (TwinError, ValueError) as an expected failure (`xfail`, not strict) until these are fixed; a wrong
    loop result still fails them.
  * *Residual spindle speed* (motion; cutting and thermal read it): after M05 the spindle's speed loop leaves
    5.8e-12 rad/s, never exactly 0. cutting's `_stale_ticks` turns a positive speed into one revolution in base
    ticks, 1e22 for that speed: an `OverflowError` into int64 (seen when a new operation's motion carried the
    spindle state; the loop now starts spindles at rest there, but a tool near the material while a spindle
    coasts down would meet it too).
  * *Chips under the next blank's locators* (chips, holding): in the batch every next part is mislocated by the
    chips the last part left on the vise ("chips or burrs under a locating face put the part 0.211 mm off its
    nominal location", an error): the verdict of every batch is unsafe.
* **Corner rounding and test references.** The controller's look-ahead rounds program corners within the
  machine's tolerance (the VF-2SS: about 0.7 mm at the side step's 90 degree corner at F1600). The tests take
  the removed-volume reference from the path recorded in frames.bin, not from the program.
