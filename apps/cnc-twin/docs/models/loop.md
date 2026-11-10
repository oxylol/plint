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
| holding | BurrSource | `SurfaceBurrs`: surface's surviving burrs on each locating patch, formed since the part in the machine started its first operation (`Run.part_start_tick`: a batch's next blank carries none of the last part's burrs; before 2026-11-09 they mislocated every next blank) |
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
  * the slip of tools in their holders and of parts in their workholding (2026-11-09): holding's ratchets return
    the work their loads did on the slip, the dissipation and the change of the slip motions' kinetic (and lift
    potential) energy, with `in = dissipation + change` exact for the discrete scheme (docs/models/holding.md
    2, 5). Holding books the dissipation into `SimState.ledger.friction_J` (architecture 12: slip friction); the
    run's ledger books `holding_slip_in_J` (in: `slip_in_J` of every tool and part record) and
    `holding_slip_mech_J` (out: `slip_mech_change_J`), `summary.conservation.energy.holding_slip_in_J`. Before,
    the slip friction was reported only and the loads' work on a slipping part or tool appeared nowhere.
* **Friction**: cutting computes the work its chip recut loads and packing loads dissipate in the chips
  (`CuttingContact.recut_J`, `packing_J`, terms of its own energy identity: their rigid-body work less their work
  on the deflection rates, which is in `deflection_work_J` with the element forces' since 2026-11-09, so it equals
  dynamics' quadrature of the modal work). Cutting's `_ledger` books them into
  `SimState.ledger.friction_J` itself since 2026-11-09; earlier builds wrote no ledger field for them. The loop
  books them only when the friction term did not change over the hot loop (`Run._book_friction`): both together
  counted them twice, 0.4 % of examples/two-setup-flip's facing entry (0.02 J of 5.3 J, a fatal energy finding).
  Holding books its slip dissipation after the hot loop (the slow domains), outside that comparison.
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
table modes carrying >= 1 % of the tip compliance (`stride_compliance_share`); the surface-memory rule (2026-11-09):
for every rotating tool the spindle angle of an L0 step at the stride the run takes at the programmed speed
(physics.resolved's when the project stores one, else the stride rule at that speed; entries `angle_per_step_deg`,
`memory_bins_max`, `l0_run_ticks`) and a warning when `physics.memory_bins` exceeds 360 deg / that angle (finer bins
than one per step add an error: docs/convergence.md 4.2, 2880 bins at 0.48 deg per step moved the mean force by
+1.5 %); the lattice rule (2026-11-09: one warning per pass with thin radial engagement, a finishing pass or a
trochoid's crescents; at most 20 named, the rest counted) from sanity's engagement estimate re-run on a height map
of lattice / 4 (`check.fine_engagement`; on sanity's own 0.5 mm map a pass thinner than a cell was found only where a
cell centre fell into it, its ae came out a cell wider, and a block whose size was a whole number of cells up to
round-off got a phantom row of material: the 0.3 mm finish of docs/convergence.md 2.1 read 1.0 mm = 4 x 0.25 mm and
was not warned. examples/pocket-islands-al6061's trochoid arcs, ae 0.17-0.22 mm on its 0.075 mm lattice, now are;
the finer map costs 12 s of that example's 22 s check): every cutting move with ae below 4 x the
lattice it cuts on (lattice / factor inside a refine box that contains the move's whole swept region; the former
rule read a `lattice_m` that refine entries do not have, and never checked coverage), grouped into passes
(consecutive moves of one tool) with their lines, the smallest ae, the lattice in force, the one the rule asks for,
the refine factor that meets it and the ledger's removed-volume error bound lattice / (2 ae) (`lattice.passes`; the
estimate is within one cell of the true ae, so the rule is decided on it and the recommendation rests on its lower
bound `ae_min_m`, never too coarse);
the estimated CPU seconds per simulated second (cost model in db/models/loop.yaml: per L0
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

* `production/energy.py` `axis_models` (2026-11-09): the machine file's `axes` block holds its own `src` (haas-ec400);
  the axis loop took it for an axis ("axis src: the machine file gives no moving_mass_kg") and every run of
  examples/ti-slot-throughtool failed at its end (production.json). Non-mapping entries are skipped, as in thermal's
  `axis_models` (the same crash at that example's set-up).
* `core/bus.py` (additive): `ToolHeat.flank_entry_K`, `ToolHeat.rake_contact_mean_K` (thermal; `rake_temp_mean_K` is
  the rake contact's entry temperature now, docs/models/thermal.md 2.2), `ToolHolding.slip_in_J`, `slip_loss_J`,
  `slip_mech_change_J`, `PartHolding.slip_in_J`, `slip_mech_change_J` (holding's slip energy, section 6).
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
| `test_loop_flip.py` | examples/two-setup-flip with one side step per operation: OP20 cuts the OP10 part turned 180 deg about X (one removal transfer, the OP10 step ends up at the bottom of OP20's S), both steps' volumes against each operation's recorded path, timeline and ledgers over both operations, operation-end checkpoints, part files per operation |
| `test_loop_batch.py` | three parts of the tiny project: fresh blanks, the same removed mass, every ledger over the three parts, tools.json per-part rows (each part's own cutting time, equal for equal parts, adding up to the tool's; VB and life used never decreasing: the tool comes in used), production.json with three parts and each part's share of the tool life, thermal unload records per part, the time balance over the parts |
| `test_loop_check.py` | `check_project`: strides by the 5.1 rules, cycle times, estimates; `--write` stores physics.resolved and a run uses those strides (and `max_sim_s` truncates it); a project with errors is not written |
| `test_loop_memory.py` | memory per macro step (2026-11-09): the tiny project with one tick per macro step (every tool change on a step's last tick) through its tool change: no tick's feed velocity above the machine's tool speed, chips' occupancy grid the size of a tool's sweep, no step adding 64 MB and the 450 steps after the tool change within 16 MB (slope <= 20 kB per step) |
| `test_check_rules.py` (tests/integration) | `twin check` on a side finishing pass (ae 0.3 mm): a warning naming the pass at lattice 0.25 mm, none at 0.0625 mm or with a refine box (factor 4) over it, a memory-bins warning for 2880 bins (0.48 deg per step) and none for 720 |
| `test_loop_probe.py` | G31 through the whole loop: the tiny project's block with the pocket example's spindle probe; a skip move from 10 mm above the top to 5 mm below stops at the top (the program checks #5063 and raises #3000 otherwise: the run must end without an alarm), the gauge line comes to rest within the braking distance (25 um) below 125 mm above the top and the retract goes straight up |
| `test_loop_parts.py` | the loop's pieces without a simulation: the coupling monitor, the ring restamping, the peak view, the run ledger's kinetic-energy report, quadrature term, holding's slip energy pair, energy scale floor and the summary of a run without energy, friction booked once, the adapters' change tracking through a checkpoint |
| `test_outputs_studies.py` | the simulate entry resolves to this loop; replay windows and comparisons through cnctwin.core.outputs |

`tests/regression/test_examples.py` (marked slow, `bin/test --all regression`): one test per example (section 14).

Unit tests of the fixes of section 15 in the domains' own suites (2026-11-09): tests/unit/core/test_hot_probe.py
(the stylus profile, probe entries of a raw and of a resolved project, the probe-only contact, removal's sweep
finding a block's top), tests/unit/motion/test_features.py `test_the_move_after_a_skip_starts_where_the_axes_stopped`,
tests/unit/holding/test_kernels.py (the tooth-period pulse and its exact sub-step mean, stall and part-speed
bounds, the discrete energy identities of slip and separation) and test_domain.py (the chuck rebuild rule),
tests/unit/thermal/test_kernels.py and test_tool.py (the entry temperatures), test_conservation.py
`test_a_turning_tools_shear_heat_goes_to_its_engaged_elements`, tests/unit/structure/test_domain.py (supports placed
again keep the model, `test_the_next_part_of_a_batch_gets_its_own_setup`), tests/unit/surface/test_domain.py
`test_a_batchs_next_part_opens_its_own_patches`, tests/unit/chips/test_domain.py (no debris under locators the
operator cleans), tests/unit/dynamics/test_chatter_detection.py (free ringing is no chatter) and test_domain.py
`test_a_turning_contact_takes_its_operating_point_from_cuttings_engaged_elements`,
tests/integration/test_turning_split.py (a polar overflow repeats only the refused removal: the pass with one inline
segment per ray finishes balanced, its booking exact; its equality with the pass with sixteen is open, section 15),
tests/unit/dynamics/test_domain.py `test_a_semi_discretisation_too_large_to_build_is_recorded_not_built` and
test_stability.py (the state size of the check).

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

2026-11-09 11:39, the frozen tree of section 14 (one process each): the unit suites of the domains the loop fix
changed pass (core 70, thermal 97, holding 73, structure 55, surface 84, chips 59, coolant 90, dynamics 158,
motion 140: 826 tests, 4 min 30 s together), and test_loop_parts (10), test_loop_probe (1, 19 s), test_turning (5),
test_slot (3), test_drill (3), test_signatures (2); test_loop_flip (4, 78 s) and test_loop_batch (3, 144 s) run
through both operations and all three parts with every ledger passing, for the first time (until then expected
failures on the NaN states at the transitions); the batch's per-part tool rows then failed an expectation of the
test (cumulative per-part cutting time; wear's rows hold each part's own), corrected (section 12).

**Cold compile**: the first `twin warmup` in an empty cache compiled every kernel module in 13 min 59 s in one
process and was killed by the 1.8 GB cap in the mill run that followed (hence one process per module now); the
second, with the cache filled, took 2 min 56 s (mill run 51.6 s; the lathe run stops, section 15). 2026-11-09:
`twin warmup --only mill` on the frozen tree of section 14, cold cache, one process per module: 12 min 46 s, its
mill run 41.0 s (safe).

**Longer runs, 2026-11-09 evening** (frozen copy of the working tree with this task's fixes, `run_project` with
every domain on and outputs off, one at a time under `bin/tsafe` with `TSAFE_MEM=1800M`, warm cache;
/tmp/cfix/scripts/longrun.py, the regression's record):

| run | wall s | CPU s | simulated | end | verdict | findings / EXPECTED |
|---|---|---|---|---|---|---|
| lathe-shaft-4140, M04 (the mirror of the real M03 set-up, section 15) | 1323 | 1308 | 5.6 s (2.4 s of the G71's first pass) | budget | not safe: gouges (the polar sync's sampling of the centre hole, fixed after this run, section 15), white layer 834 degC (thermal's open estimate) | roughing spindle power 6.41 kW (5.4-6.8): passed, the first time the example's power item is reached; no spindle overload: passed; 5049 mm3 removed; mass, energy and time balanced |
| drill-tap-plate-1045 | 512 | ~510 | 5.33 s | general protection fault in the hot loop | - | the D6.8 cuts its first hole steadily from 4.99 s; a recut strike of flute chips kicks it into growing vibration (section 15) |
| thin-wall-al7075 | 715 | ~700 | 5.9 s (1.5 s into its first cut) | killed by the 1.8 GB cap (anon RSS 1.83 GB) | - | not measured further (section 15) |
| pocket-islands-al6061 | 964 | 954 | 12.0 s | budget (probing, the T1 change; no cut yet) | safe | the N10 probing item passes; structure took 818 of the 954 s (section 15) |

At these costs (240 CPU s per simulated second in the lathe's cut, structure's refresh alone 60-70 s/s) an
operation of 150-265 s takes 10-17 hours of CPU per example: no example was run through a whole operation.

**Runs of 2026-11-10** (worktree with section 15's fixes of that day, `run_project` with every domain on and
outputs off, one at a time under `bin/tsafe` with `TSAFE_MEM=1800M`; /tmp/runsfix/scripts/longrun.py; resumes read
checkpoints of the same tree with the build check bypassed):

| run | wall s | CPU s | simulated | end | findings |
|---|---|---|---|---|---|
| thin-wall-al7075, resumed at 4.3 s | 3694 | 3675 | 4.3 -> 56.68 s | fatal divergence at 56.68 s (chip thickness 185.7 mm, Y axis 23.6 kN demanded against 8874 N; trochoid between the fins, line 120) | memory flat: anon RSS 1.68-1.69 GB from 6 s on, checkpoints at 10-50 s; gouges 0.603 mm with breakthrough at 21.8, 24.5, 44.7, 47.4 s; melt smear 478-518 degC repeatedly; chatter warnings (81-5057 Hz); 3-16 % recut chips |
| pocket-islands-al6061, resumed at 12.5 s | 255 | 249 | 12.5 -> 14.5 s (0.8 s of T1's helix) | budget | verdict safe, no end-of-life finding (was tool_end_of_life at 13.80 s); a 205 um witness mark on a finished surface at 14.49 s and servo marks (not examined) |
| pocket-islands-al6061, finish formula fixed but not yet the per-feature assignment | - | - | 10 -> 16 s | killed by the cap at its end (the end-of-run keyframe: anon RSS 1.38 -> 1.85 GB) | T1's finish estimate 0.90 of its limit at 16 s with the hone at 14.3 um, 42 elements held to the floor's Ra 1.6 um (the reason for the per-feature assignment, section 15) |
| drill-tap-plate-1045, resumed at 6.79 s | 1914 | 1908 | 6.79 -> 8.5 s | budget (the spot drill's first spot, dwell and retract) | no divergence (was 6.80-6.95 s); thrust 1.10-1.36 kN (EXPECTED 0.8-1.4 kN); gouges 0.008-0.072 mm on the first countersink; packed flutes (fill up to 2.5) |

## 14. Regression

`tests/regression/test_examples.py` runs every example (and a no-probe variant of the two that probe) through
`run_project` with every domain on and outputs off, `max_sim_s` 8 s (the set-up, the first tool change and the
first cut of each program: the whole loop costs 7-60 CPU s per simulated second on these projects, so 20 s per
example, the first choice, would have taken the box most of a day), one pytest process per case under
`bin/tsafe` with `TSAFE_MEM=1800M`; it asserts that the run
completes and conserves, and records each EXPECTED.md item it can read as passed, failed or not reached
(tests/regression/expected.py; `<basetemp>/regression/<case>.json`). Results of 2026-11-09 11:39 on a frozen copy
of the working tree of 09:44 (every agent's files then, the loop-fix changes of section 15 included; the Numba
cache compiled cold by `twin warmup --only mill` in 12.8 min, its mill run 41.0 s, safe), by case:

| case | wall s | sim s | run | verdict | findings | EXPECTED passed | failed | not reached | conservation |
|---|---|---|---|---|---|---|---|---|---|
| sanity-mistakes | 122 | 0.01 | stopped by its alarm 320 at line 16 | not safe (the program's mistakes) | error 14, warning 11 (sanity 23, alarm 1, remaining_material 1) | the 23 sanity findings; alarm 320 at line 16 | - | - | ok |
| ti-slot-throughtool | 412 | 8.00 | budget (30 ms into the slot); first run killed by the 1.8 GB cap at its end (358 s), fixed (section 15) and run again at 12:31 | safe | info 2, warning 4 (thermal_damage 887 degC, tool_end_of_life, burr, air_cutting) | no spindle overload; cutting-data rule silent for T1 | slotting torque 9.7 N m (10-24) and power 0.09 kW (0.6-1.4), both at the slot's entry | - | ok |
| drill-tap-plate-1045 | 768 | 8.00 | budget | not safe (the D6.8's entry blows up at 4.99 s: tool_breakage, thermal_damage) | error 2, warning 5, info 2 | no spindle overload | drilling torque 0.86 N m (2.6-4.6); thrust 12.3 kN (0.8-1.4) | - | ok |
| two-setup-flip | 344 | 3.81 | stopped by the energy balance at the first cut (1.01e-6) | - | fatal 1, error 1 (gouge 6 um), warning 2 | no spindle overload | - | OP20 on the turned part | stopped |
| thin-wall-al7075 | 393 | 4.46 | stopped by the energy balance at the first cut (1.66e-5) | - | fatal 1, warning 4 (chip_recutting 3.9 %, tool_end_of_life) | no spindle overload | - | peel power | stopped |
| pocket-islands-al6061 | 692 | 8.00 | budget (N10's probing, the T1 change, before the helix) | safe | info 1, warning 3 | no spindle overload; **N10 probe Z: G31 triggers without alarm** | - | trochoid power | ok |
| pocket-islands-al6061 (no-probe) | 399 | 4.50 | stopped by the energy balance at the helix entry (5.4e-6) | - | fatal 1, warning 6 (part_slip margin, chip_recutting) | no spindle overload; the probe item (trivially) | - | trochoid power | stopped |
| batch-50 | 619 | 8.00 | budget (N10's probing, before the first cut) | safe | info 2, warning 2 | no spindle overload | - | T1 roughing power | ok |
| batch-50 (no-probe) | 368 | 4.22 | stopped by the energy balance at the first cut (1.67e-6) | - | fatal 1, error 2 (gouge 6 um, melt smear 589 degC), warning 4 | no spindle overload | - | T1 roughing power | stopped |
| lathe-shaft-4140 | 517 | 8.00 | budget (before the first cut; 7 work set-ups, 55 in 2.4 s before) | safe | info 1, warning 1 | no spindle overload | - | roughing shaft power | ok |
| die-h13-ballnose | 889 | 8.00 | budget | not safe (the unclamped insert slips and comes free, section 15) | error 4, warning 1, info 3 | no spindle overload; no overtravel | - | T1 power, T1 force | ok |
| blade-5axis-ti | 598 | 8.00 | budget | safe | info 2, warning 1 | no spindle overload; no overtravel; no pass through B = 0 | - | - | ok |

Compared with the regression before this task's fixes (same budgets): 8 of 12 cases now run to their budget or
their program's own alarm conserving (before 5), the probing examples probe (pocket-islands' EXPECTED probe
item passes; before alarm 379 at the first G31 on both), drill-tap passes the coefficient pipeline's NaN (cutting's
owner's fix) and ti-slot its set-up (thermal's and production's machine models) and its end (dynamics' bounded
semi-discretisation check), lathe-shaft-4140 runs 8 s in 8.6 min of wall time (before stopped by hand after 28 min
at 2.4 s). EXPECTED items over the 12 cases: 19 passed, 4 failed (drill-tap's torque and thrust at its blown-up
entry; ti-slot's torque and power in the slot's first 30 ms, below the full slot's bands), 9 not reached (beyond
the budget or a run that stopped at its first cut).

Where the runs end:

* **Budget or the program's alarm, conserving** (8): sanity-mistakes (its alarm), pocket-islands and batch-50
  (their probing done, no cut yet), lathe-shaft-4140 (no cut yet), blade-5axis-ti, ti-slot-throughtool (into its
  slot by 30 ms), die-h13-ballnose (its insert
  slips before the first cut: no clamping of the zero-point plate, section 15) and drill-tap-plate-1045, whose
  D6.8 blows up at its entry at 4.99 s (lateral vibration 38 um within 0.1 ms of the first contact, then forces of
  10-30 kN: tool_breakage, white layer, edge chipping; the run conserves through it; bisection in section 15).
* **The energy balance at the first cut's entry** (4): thin-wall-al7075, two-setup-flip and the no-probe variants
  of pocket-islands and batch-50, with residuals of 1.0e-6 to 1.7e-5 of the interval's terms; each interval holds
  the recut and packing loads (`friction_J` 0.02-1.05 J): cutting's recut impulse and the recut / packing work on
  the deflection rates missing from `e_def` (section 15, items 1 and 2 of "Cuts went unstable at entry";
  the pocket run balances with the recut frozen).
* **Memory** (fixed): ti-slot-throughtool was killed by the 1.8 GB cap at the end of its run, in dynamics'
  end-of-run semi-discretisation check (section 15); run again with the check bounded it completes as above (its
  T1 chipped and "worn out" in its first 4 ms by wear's edge chipping at 829 N, for wear's owner; the 887 degC
  surface is thermal's flash estimate on a fresh edge's hone-wide flank contact, not validated, section 15).

A longer run to take one example through a complete operation: examples/batch-50 with one part and 30 s
(`/tmp/lf/scripts/longrun.py`, the regression's record) probed, changed to T1 and stopped at 13.88 s at T1's first
cut on the energy balance (5.98e-6; friction 0.28 J of recut and packing in the interval: the same cause), after
940 s of wall time; a part slip margin of 0.41 and a 6 um gouge on the way in. No example therefore completes an
operation until cutting's recut terms are fixed; the complete operations and transitions run on small variants:
tests/integration/test_loop_flip.py (both operations of examples/two-setup-flip with one side step each) and
test_loop_batch.py (three complete parts of the tiny project) pass on the frozen tree, every ledger balanced
(section 12).

## 15. Open problems

* **Cutting's energy identity with a lubricated contact** (2026-11-10): run.json's `cutting_identity_rel` on the
  tiny project is 5.7e-11 dry and 1.2e-9 once the hand-set flood nozzle reaches the contact and the friction
  factor varies from step to step (0.93-0.95). The cause is not isolated; it is three orders below the run's
  energy contract (1e-6), and tests/integration/test_loop_run.py bounds it at 1e-8.
* **Cost.** The tiny project costs 7.4 CPU s per simulated second (section 13): the hot loop 6 s per second in
  contact, the slow domains 20 s per second in contact (199 ms per macro step) and 4 s per second out of it
  (41 ms per step; dynamics 25 ms of it on average): a real program of minutes takes hours. The regression tests
  run a representative first part of the long examples (section 14).
* **Probe** (fixed 2026-11-09). The hot loop builds a probe-only contact for a path whose spindle holds a probe
  (core/hot_probe.py, docs/models/hotloop.md 9a): removal's `sweep_contact` latches G31 moves and the controller
  gets the encoder position. Before, no domain built the stylus profile and every probing program stopped at its
  first probing alarm (pocket-islands N10, batch-50 on every part). Two further causes surfaced on the way: the
  resolved project lists its probes in `tools.probes` (the loader moves the `kind: probe` entries out of
  `tools.list`), which hot_probe now reads with each entry's own resolved stylus; and motion's planner planned the
  move after a skip from the G31 block's programmed end point (`Planner._skip_stop` braked the axes but left
  `tail_q`, the queue's end, at the programmed end): the command jumped 5 mm past the touch and the stylus
  followed it 3 mm into the part before the retract came back. `_skip_stop` now sets the queue's end to where the
  axes come to rest (one line in cnctwin/motion/planner.py, test in tests/unit/motion/test_features.py; nothing
  is queued past a skip move, which stops the block buffer). tests/integration/test_loop_probe.py: the G31 stops at
  the block's top (#5063 within 0.05 mm, the gauge line never more than the 25 um braking distance below the
  touch). The regression keeps the no-probe variants.
* **A turned part was set up again and again** (fixed 2026-11-09): holding rebuilt a chuck's setup whenever
  omega^2 changed by 2 % (a spindle run-up from rest: dozens of rebuilds for grip changes of a fraction of a
  percent) and structure set its work model up at every new `supports_version` (55 FE solves and modal bases in
  2.4 s of examples/lathe-shaft-4140). Holding now rebuilds when the centrifugal loss moved the grip by 2 % of the
  static force, structure keeps its model when the re-placed supports are within `support_change_rel` (2 %)
  (docs/models/holding.md 4, docs/models/structure.md).
* **Memory at an operation's end** (fixed 2026-11-09: a part whose surface bricks could hold more than 3 M
  triangles is meshed with its flat patches as fans, the same surface; docs/models/removal.md 10). inspect's part
  meshes come from the field's mesh at its lattice (`LoopResults.result_part`: removal's `mesh_level`, then
  inspect's displacement grids per vertex and the STL, GLB and 3MF files): about 9.5 M triangles for
  examples/pocket-islands-al6061 (0.075 mm on 100 x 80 x 30 mm). That end killed its first regression runs at the
  1.8 GB cap (exit 137) when its probe alarm ended the operation; unfinished operations are not inspected (section
  1), and a finished operation of a fine-lattice example needed more than 1.8 GB there.
* **Memory at the end of a run** (fixed, dynamics): examples/ti-slot-throughtool ran its set-up, ICS coolant start
  and approach to its budget (the first contact at 7.97 s) and was killed by the 1.8 GB cap; the harness run of
  drill-tap with process damping off the same at 5.03 s. Both ended right after their first operating point: the
  end-of-run semi-discretisation check sized its intervals to resolve the highest checked mode over the delay,
  r = 24 x f_max x tau, which at 573 rpm (4 teeth, tau 26 ms) or 830 rpm (2 teeth, 36 ms) against modes of
  10-20 kHz is tens of thousands, and built a monodromy matrix of (2 n + r n)^2 entries. A point whose check would
  exceed `sd_max_states` (2000) now keeps its frequency-domain verdict and is recorded as not checked
  (docs/models/dynamics.md 5.4).
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
* **lag_macro 2** is computed as lag 1 (nothing runs concurrently). fixed (2026-11-10, the advice): the coupling
  warnings of cutting and of the loop asked to "compare physics.lag_macro 1 and 2", a comparison that cannot differ;
  both now advise halving `physics.macro_ipo`, the coupling interval itself (docs/convergence.md 9 item 7). The
  monitors were fixed with it: cutting's watched the main row's coefficient with a 1e-6 threshold (the quantised
  state's +-0.15 % flips and unused rows flagged), the loop's the largest element's rake temperature (switching
  elements flagged); both now watch the in-cut-time weighted quantity (the coefficient averaged over the rows the cut
  uses, the rake temperature the rows read) under the loop's rule (1e-3 of the value, three alternations), and a
  step without cut leaves them as they were (`CuttingDomain._lag_monitor`, `loop._cut_weighted`).
* **Peaks were step means; the element at the depth-of-cut line was whole or nothing** (docs/convergence.md 9 items
  1 and 2). fixed (2026-11-10, cutting and the hot loop): the step's peak is the instantaneous wrench at its start
  or end where larger than the mean (stride_slot at the default 0.5 deg: 259 N against 258 N at 0.0625 deg, was
  251 against 264), and the boundary element carries its engaged width, measured by `hotloop.edge_reach` with the
  gating's point test along the edge after every interval refresh (element_flat: 196.7 N at every element size
  from 1 to 0.0625 mm, was 186.8-200.2 N). Both first versions failed in the integrated slot and were corrected the
  same day: the end forces extrapolated the table segment of the centroid chip (850 N peaks), and the boundary
  was a ray cast along the chord between the stations, which at a flat end mill's corner runs through the cleared
  inside of the cut (the lowest flank element lost half its width, -6 % at 0.5 mm). A slot is the case measured; an
  inclined boundary is taken at one angle per station side.
* **Numba cache.** The cache directory is keyed by every kernel source: a change in any kernel file costs the
  next run a cold compile of every kernel (12-14 minutes on this box under load).
* **Found in integration** (each reproduced with the tiny project, a variant of it, or an example; 2026-11-09:
  the root causes, what this task fixed in its domains, and what it hands to the owners of cnctwin/cutting,
  cnctwin/removal, cnctwin/mechanics, cnctwin/tooling, cnctwin/motion and cnctwin/wear, which it did not edit but
  for the one-line skip fix in motion's planner above):
  * *Cuts went unstable at entry* - three causes, bisected by freezing each slow feedback in turn on the tiny
    project's full slot (structure modes -> machine and spindle modes only, holding -> none, thermal -> ambient,
    chips' recut grid and packing -> none, coolant -> dry): only freezing the chips' **recut** removed the
    blow-up (the slot then cut its 14 mm at 110 N mean, 140 N peak, 530 W, stable).
    1. **Cutting's recut impulse (fixed 2026-11-09: docs/models/cutting.md "Recutting": the swept share of a
       voxel's chips, cut like material at the chips' section over the crossing time)** - `cnctwin/cutting/kernels.py` `recut_scan` (from
       `finish_step`): every chip voxel of chips' frozen occupancy grid (2 mm pile cells) whose centre is inside
       the tool's envelope gives its whole remaining mass m to one strike per tooth period, with the impulse
       `J = m V + E_cut / V`, `E_cut = u(h_chip) m / rho` (the specific cutting energy of the voxel's whole chip
       volume), applied to the tool within the one L0 step of the scan (5 us at 10 000 rpm). The tiny slot at
       4.12 s: one voxel of 1.2 mg, 0.43 J, 0.09 N s: 17 kN over the step, 8.9 MN in the next after the tool's
       response (tool broken, holder pulled out, part free); examples/thin-wall-al7075 at 4.444 s: 169 N m on the
       D10 (the ER collet turned 792 deg in the old holding model); examples/two-setup-flip at 3.81 s: a 17.3 kN /
       914 N m peak on the part (it slipped 38 mm in the old holding model). The physics: a tooth meeting a loose
       chip knocks it away (impulse m V: momentum, no cutting energy) or, trapped against a wall, shears through
       its section while the edge crosses it; the force is bounded by the chip section's shear strength (about
       tau_s x width x thickness: ~100 N for a 40 um x 1.5 mm Al chip) and lasts the crossing time (size / V),
       the energy is that force over the path, not u times the voxel's volume; and a tooth meets only the chips in
       its swept volume of the step, not a 2 mm voxel's whole mass. Fix: spread each strike over the crossing
       time with the section-bounded force, take the swept fraction of the voxel's mass.
    2. **The recut and packing loads' work on the deflection rates (fixed 2026-11-09: both act at an engaged
       station, their work on the deflection rates is in e_def, equal to dynamics' quadrature)** - the same
       `finish_step` adds the recut impulse `rj` and the packing force to the step's impulse on the tool after
       `e_def` (CA_DEFLECTION_J: the element forces' work on the station deflection rates) was accumulated:
       dynamics' modes receive and integrate them (its quadrature `cutting_work_quadrature_J` includes their work
       on the deflection rate), `CuttingContact.deflection_work_J` does not, so the run's energy balance misses it:
       the entry residuals of the regression (pocket-islands no-probe 9.2e-7 J of 0.092 J at 4.5 s, its friction
       term (recut + packing) 0.046 J; thin-wall 4.1e-4 J of 17.6 J; the flip 1.5e-4 J of 9.1 J). With the recut
       frozen the same pocket run passes 4.5 s balanced. Fix: add their work on the deflection rate of the struck
       station (the same predicted rate the element forces use) to e_def.
    3. **Holding turned the spikes into run-away displacements (fixed, holding)**: the tooth harmonic stretched a
       microsecond shock over a third of every tooth period of the step, and above the friction capacity the
       slip accelerated the small tool inertia or the part freely (792 deg, 38 mm per step). Now the demand's
       tooth-period pulse follows each tick's mean / peak, every sub-step takes its exact impulse, the tool stalls
       at the spindle speed and a part moves no faster than the cut can push it, and the slip's energy is in the
       ledger (docs/models/holding.md 2, 5; section 6). The flip's spike now slips the part 2.8 um.
    4. **The "low-frequency chatter" was the machine frame ringing (fixed, dynamics)**: the 55-130 Hz frame and
       table modes of the VF-2SS, excited by the base excitation of rapids and plunges, rang with 1-3 um while the
       broad phase ran L0 steps near the stock before the first chip; the detector analysed those samples and
       called the free vibration "regenerative chatter" at 65-91 Hz. The window now takes only steps that cut,
       and a vibration decaying at its structure's damping is a transient (docs/models/dynamics.md 4).
    The hot loop's own tests were stable because they cut no slot with chips: OperationDriver runs no chips
    domain, its occupancy grid is empty.
  * *Structure's 5339 Hz pair* is the D10's own bending mode on its collet, not an anomaly: hand check in
    docs/models/structure.md 10 (rigid-clamp cantilever 13 N/um at 7.2 kHz; with the collet 7.2 N/um at 5.3 kHz,
    modal mass 6.4 g; the 697 Hz pair is the spindle flange mode with holder and tool). Work modes of the tiny
    project's 30 x 20 x 10 mm block in the vise: 34-71 kHz with modal compliances of 1e-9 m/N (a stubby Al block:
    Euler-Bernoulli 58 kHz for its 10 mm bending, Timoshenko about 30 % lower), the machine's frame modes
    m = k / omega^2 = 120-210 kg with 1/k at the tip: units and normalisation consistent.
  * *Staggered thermo-mechanical coupling* (fixed at its source, thermal): the rake temperature cutting's
    coefficient pipeline reads was the in-cut mean, which already holds the rise of the step's own heat that
    mechanics' partition adds again (docs/models/thermal.md 2.2): the explicit stagger had a loop gain near -1
    and alternated by 20-37 K. thermal now publishes the temperature at the entry into the cut (the background
    of mechanics' stationary-source partition): the tiny project's alternation fell to about 2 K. What remains
    is a limit cycle of cutting's quantised slow state (fixed 2026-11-09: hysteresis of 0.75 cell, and the flank
    partition reads `flank_entry_K`; docs/models/cutting.md 1, 2): `cnctwin/cutting/coefficients.py`
    `quantised_state` snaps the tool temperature to `tol_temp_dK` (10 K) by rounding, so a temperature at a cell
    boundary flips the row between two cells from one step to the next (the loop's monitor still flags 2.4 K /
    1.9 K changes, ratio 1.2, at the tiny project's entry); a hysteresis of the snapping (keep the cell while the
    value stays within 0.75 cell of its centre) removes it. Also for cutting's owner: `_row_state` reads
    `flank_temp_q_K` (the in-cut flank temperature, which wear needs) for mechanics' flank partition, the same
    double count on the flank; thermal publishes `flank_entry_K` for it.
  * *NaN flank temperature in the coefficient pipeline (for cutting's owner)* - not thermal's:
    `cnctwin/cutting/domain.py` `CuttingDomain._row_state` computes
    `flank_T = float(np.mean((q @ w)[members] / w.sum()))` without the `members.any()` guard `member_mean` has:
    for a row none of whose elements is a member (examples/drill-tap-plate-1045's D6.8 drill at 2.58 s, row 0 of
    slot 0: members (2, 33) all false) it is the mean of an empty array, NaN, and `quantised_state`'s `_snap`
    raises "cannot convert float NaN to integer"; thermal's `flank_temp_q_K` there is finite (2 x 33 x 3, no NaN).
    Fix: `flank_T = None` for an empty row, as for the tool temperature (done by cutting's owner on 2026-11-09:
    `_row_state` now returns None for a row without members and for a non-finite mean).
  * *Cutting's analytic chip volume* (fixed 2026-11-09, two causes: removal's classification turned the top cell of
    a cut along the stock's top edge into air, so its station had no engaged interval (-17 %); cutting counted a
    ploughed layer below h_min again at the pass that removed it (+3.7 %); now 1.026 of the field on the side step,
    docs/models/removal.md 5, cutting.md 4) was 11 % below the volume the tool path sweeps on the
    side step (ae/D 0.05, ap 1 mm): 5.43 mm3 against 6.05-6.11 mm3. Removal's field agrees with the swept volume
    of the recorded path to 0.02 % (its rays sampled on the same lattice), so removal's pass check ("field volume
    and analytic chip volume differ by 12.5 %, allowed 4.6 %") reports cutting's shortfall.
  * *Rapid into material at zero depth* (fixed 2026-11-09: the event carries the intrusion's depth, a crash is a
    depth beyond the collision tolerance; docs/models/removal.md "Rapids"): a G00 retract straight up from the end of a side
    cut raises the error `rapid_into_material` with "0.000 mm deep, 0.000 mm^3" (any removal during a rapid
    counts, here the servo lag of the feed axis catching up while Z already rises). Real programs retract like
    that; the tests lead out sideways first.
  * *Slivers* (fixed 2026-11-09: a remainder under 1 um touching a removal goes with it; docs/models/removal.md
    "Slivers"): the side step leaves 4-quantum (40 nm) pieces of material floating in rays it
    cut (ray segments like (972732, 972736) at z 9.602 mm above a floor at 8.993 mm); `surface_height` then
    reports the sliver's top as the surface.
  * *Residual spindle speed* (for motion's owner; cutting and thermal read it): after M05 the spindle's speed loop
    leaves 5.8e-12 rad/s, never exactly 0. cutting's `_stale_ticks` turns a positive speed into one revolution in
    base ticks, 1e22 for that speed: an `OverflowError` into int64 (seen when a new operation's motion carried the
    spindle state; the loop starts spindles at rest there, but a tool near the material while a spindle coasts
    down would meet it too). cutting's owner bounded `_stale_ticks` like a stopped spindle on 2026-11-09; the
    residual speed itself is still motion's.
  * *Chips and burrs under the next blank's locators* (fixed, chips and the loop): every next part of a batch was
    mislocated ("chips or burrs under a locating face put the part 0.211 mm off its nominal location", an error:
    every batch unsafe). Two causes: chips measured the chips on the next setup's locating faces as if nobody
    cleaned them, while the operator blows fixture and locators clean before loading (production.operator
    clean_s, now modelled: docs/models/chips.md 4); and the loop's `SurfaceBurrs` handed holding the burrs of the
    batch's previous part, which left the machine with it (now only the burrs since the part in the machine
    started, section 2). With the mislocation gone the re-seated part's NaN work-mode rows of the git-HEAD tree
    (structure's model of a part not seated on its datum) no longer arise in the batch; structure was not changed
    for them.
  * *A batch's later parts* (fixed, structure and surface): a one-operation batch runs every part as operation 0.
    Structure's released check keyed on the operation only, so once the first part was unclamped its released
    state (no clamp loads, minimal support) stayed published for every later part, for which no model was ever set
    up (and with the supports placed as before, the kept-model rule of the lathe fix would have kept the machined
    first part's modes for the fresh blank): the work model is now set up per part (`WorkStructState.part_index`,
    docs/models/structure.md, tests/unit/structure/test_domain.py). Surface matched a later part's events to the
    first part's face patches (same workpiece index, brick and plane), compared the two parts' tracks as overlapping
    passes and reported witness marks of 2-143 um (finish_out_of_spec warnings) on every second part of the tiny
    batch: patches now belong to the operation that opened them (docs/models/surface.md 7).
  * *Turning in the full loop.* Warmup's lathe pass (d30 4140 bar, CNMG 120408, ap 1.5 mm, f 0.25 mm/rev,
    1400 rpm) first churned set-ups (above) and then stopped or exploded at its first contact for three loop
    causes, fixed: a polar overflow split repeated the whole L0 step (dynamics stepped by 0 base ticks, the step's
    work booked twice: only its removal is repeated now, docs/models/hotloop.md 6); thermal placed the turning heat
    source at removal's contact point, which turning contacts leave at zero (the spindle axis at the chuck face:
    3627 degC), now at the tool tip; dynamics found no operating point for turning (removal gives no depth) and ran
    without its relaxation estimate, now from cutting's engaged elements (docs/models/dynamics.md 5.4). The pass
    then cuts stably for one revolution and more, but the hot loop's force alternates from one L0 step to the next
    from the first revolution on (Fz 739 / 857 N, Fx -74 / -158 N at 5 us steps, displacements smooth), and at
    2.707 s, 1.3 revolutions in, the alternation grows within 0.5 ms (the high steps 865 N -> 1.9 kN -> 30 kN ->
    115 kN) until the tool breaks and the part comes free. **Root cause (fixed 2026-11-09: process damping from
  each element's wave memory of the surface it generated, docs/models/cutting.md "Process damping")**: process damping.
    `cnctwin/cutting/kernels.py` `turn_step` (and `cut_step` likewise) adds per element
    `fpd = pd[e,0] * flank_interference_area(vnd / Vc, land, tan_clear, pd_max_flank)` with `vnd` the normal
    velocity of the step's predicted vibration state. The model linearises the surface the edge left behind as a
    straight line of slope vnd / Vc under the whole flank (indentation depth slope x x up to 0.5 mm behind the
    edge), which holds for waves much longer than that contact length (here frequencies well below
    Vc / 0.5 mm = 4 kHz). The hot loop's L0 steps carry the response of the work's lightly damped bending modes
    (32-48 kHz, zeta 0.1-0.16 %, 40-110 g at the contact, docs/models/structure.md) and of the predictor's
    extrapolated force: a step's normal velocity of about 0.2 m/s at nanometre amplitude gives a slope above the
    clearance (about 6 deg for the CNMG in its holder) and the whole 0.5 mm flank "indents" (about 10 N per
    element) although the real interference is bounded by twice the vibration amplitude (tests/integration/
    test_turning.py, whose OperationDriver runs dynamics without structure's work modes, turns the same kind of
    pass stably with process damping on); the force is one-sided (vnd > 0 only), so it switches
    on and off from step to step, the predictor's linear extrapolation of that switching force overshoots the
    predicted velocity, and the pair is a self-sustained relay that diverges once the cut has grown. Bisected: the
    same run with K_sp = 0 (process damping off, harness only) cuts stably to 2.75 s (Fm 1139 N tangential,
    811 N feed, 30 um peak-to-peak vibration, no step-to-step alternation), with it it explodes at 2.707 s
    (deterministic: lathe3 and lathe_pd identical). The relaxation of dynamics' corrector already uses the
    operating point's cutting stiffness (2.6e6 N/m: K_m G1 ~ 2e-4, negligible) and could take the linear process
    damping of the same model, but no linearisation covers a force that switches between nothing and the
    0.5 mm flank. Fix: evaluate the interference from the surface the edge actually generated - the element's
    normal displacement over the last pd_max_flank / Vc (a ring of u_n per element: delta(x) = u_n(t) -
    u_n(t - x / Vc) less the clearance), or equivalently low-pass the slope with the contact time L / Vc - so that
    vibration much shorter than the flank gives no indentation while chatter keeps its process damping.
    The same cause breaks examples/drill-tap-plate-1045's D6.8 at its entry (regression, section 14): at 4.99 s the
    first contact's lateral vibration reaches 38 um within 0.1 ms and the forces 10-30 kN (edge chipping, a 530 GPa
    bending stress, 3296 degC). Bisected on the frozen tree to 5.03 s: with the chips' recut grid frozen the entry
    blows up identically; with K_sp = 0 it does not (that run reached its budget; its end then needed the bounded
    semi-discretisation check of section 15). Near a drill's centre the cutting speed goes to zero, so the slope
    vnd / Vc saturates the flank interference for any vibration at all.
  * *Heat of a turning pass in the work* (fixed, thermal): thermal spread the shear-zone heat over the edge
    elements with weights h db r, r = `elem_r_m`, which tooling defines for non-rotating tools as the x_T
    coordinate (<= 0); every weight was clipped to zero, the share fell back to an even split over all 254
    elements of the CNMG and the idle elements' share went into the work (their chip fraction read at h = 0): 75 %
    of warmup's turning pass (2.3 kW) went into the bar, 20 % into the chip. The work's surface speed is the
    same along a turning edge, the weight is now h db: 75 % chip, 5 % insert, 19 % bar at 2.72 s (docs/models/
    thermal.md 2, tests/unit/thermal/test_conservation.py). The surface temperature thermal reports for that pass
    is still 981 degC (white layer at 740 degC) with a fresh insert: its flank and ploughing heat (350 W, 15 % of
    the power) enters through a flank contact one hone radius wide; that estimate is not validated.
  * *Removal's turning contacts carry no point and no depth* (fixed 2026-11-09: polar_cut accumulates them;
    docs/models/removal.md 11): `RemovalContact.geometry`
    of a TK_TURNING contact has `point_S_m` zero and `ap_m` 0 (`polar_cut` books the removed volume but not the
    engaged centroid FA_CX..FA_CZ / FA_W nor the depth range FA_ZREL_MIN / FA_ZREL_MAX that the helix and field
    cuts accumulate). Every consumer of the contact point read the S origin (thermal's source, dynamics' reference
    station and operating point, above). Thermal and dynamics now take the tool tip and cutting's engaged
    elements; fix in removal: accumulate the turning step's centroid and z range in polar_cut as the other cuts do.
* **Polar capacity changes the pass** (fixed 2026-11-09: the cause was the polar rows' memory budget, reckoned with
  the configured polar_kp: kp 16 coarsened the d45 bar's rows from 6.25 um to 75 um; the budget now uses a fixed
  capacity and the split pass equals the pass without overflow to round-off, test_turning_split): tests/integration/
  test_turning_split.py turns the same OD pass (d45 bar, ap 1.5 mm, f 0.25 mm/rev, 1400 rpm) with one and with
  sixteen inline segments per polar ray (`physics.removal.polar_kp`). With one, the hot loop splits once on a polar
  overflow at the first revolution's feed marks and resumes with only the refused removal (the fix of this task:
  the pass finishes, its ledgers balance and the booked removal is the field's change, the first test). The pass
  should not depend on the field's capacity, but it removes 1.1 % less (5.89e-7 against 5.96e-7 m3, rigid; 5.86e-7
  against 5.93e-7 with the modes), and the rigid runs' step forces differ by 36 N already in the step before the
  first split (the first contact step, nothing booked yet in either), the removal per step then 10-60 % lower for
  a few steps and about 8 % lower at full engagement. Not isolated: a part of removal's polar cut or of the field's
  growth that depends on `PI_KP` before an overflow is reported, or the hot loop's view of the refused step. The
  exactness test is marked xfail with this note.
* **The lathe-shaft example cut air** (fixed 2026-11-09, removal): its stock is a STEP mesh (the d45 x 230 bar with
  the A3.15 centre hole at the free end) and removal built a polar field only for cylinder and tube stocks: the G71
  ran its passes with the tool 2 mm inside the bar and removed nothing (air_cutting, 11 s). A mesh stock on a lathe
  now gets `polar.polar_from_mesh` (exact ray-triangle crossings, the tri-dexel tie rule; docs/models/removal.md
  11). Two more causes surfaced with the first chips: the polar volume summed `q^2` in int64 and wrapped (the bar
  read 20 % of its volume; `polar.q2_sum` is exact now), and cutting's chip of the pass was 1.06 x the polar field's
  removal (22.25 against 20.97 mm3 per macro step): `h db` on the CNMG's 0.8 mm nose overstates the strip behind a
  curved edge by h / 2 r_eps (the strips now narrow, `abi.edge_curvature`), and `polar_depth`'s linear root sat near
  the outside sample where the depth ray enters the previous revolution's air gap (now a sign bisection; both
  docs/models/cutting.md "Turning family"). With the bar cutting, removal's polar -> tri-dexel sync rebuilt the whole
  disc of the dirty rows at every revolution and booked its polar sampling of the unturned centre hole as removed
  part material (gouge errors of 7-292 um at r 2-3 mm): the sync now rebuilds only the annulus the insert reached
  (docs/models/removal.md 11). And the example's M03 turns the bar against the CNMG's rake face in this model
  (hotloop.md "Lathe spindle direction": tooling's right-hand turning tool sits upright at +X, where a Haas ST's
  turret holds it inverted); the runs of section 13 use M04, the mirror image of the real set-up.
* **Recut strikes of flute chips kick a drill into a growing vibration** (open, cutting and chips; found 2026-11-09
  in examples/drill-tap-plate-1045, whose run then died with a general protection fault at 5.33 s). The D6.8 cuts
  its first hole steadily from 4.99 s (213 N, 4.6 um station deflection, 0.01 m/s), until a recut strike of 0.39 mg
  at 5.136 s: the voxel's chips inside the drill's envelope cylinder (cutting's `recut_scan`), cut by the lip
  across the whole 2 mm voxel (h 60 um x 2.2 mm: 431 N) at the speed of the struck centroid (r near the axis, 0.17
  m/s, so the load lasts milliseconds). The drill then vibrates with growing amplitude (about 6 kHz, 130 rad/s
  rotation rate and 0.4 mm within 4 ms; logged step by step with every kernel in Python from a checkpoint),
  decays, is kicked again, and at 5.33 s an element reads a chip of 649 km: `int((h - h_min) inv_dhi)` overflows
  and the table index is out of bounds (pure-Python resume: OverflowError at cutting/kernels.py cut_step's segment
  index). The chips the strike takes are in the frozen occupancy grid inside the drill's envelope, i.e. mostly in
  its flutes, which turn with the tool and are never met by a lip; the recut model takes every chip of the
  envelope cylinder. To be settled between cutting (which chips a tooth can meet: the swept layer in front of the
  edges, not the flute volume) and chips (flute chips are carried, not frozen); and cut_step should refuse a chip
  thickness beyond the tool's reach (a finding, not a silent wrap of the index).
  fixed (2026-11-10, cutting; root causes measured on the 2026-11-10 tree, every run from the start of the example,
  logged per macro step from 4.95 s): the tool cutting from 4.99 s is T1, N10's D10 90 deg spot drill (G82 to
  Z-4.5 at 830 rpm; this entry and section 14 called it the D6.8). With the tree as found it cut steadily from 4.99 s (138-214 N, station
  deflection 4 um) until a recut strike at 5.13 s (0.39 mg, 431 N for 2.2 ms: 0.39 mm and 2.2 m/s), decayed, was
  struck again at 5.21 s (0.54 mg), went into a sustained 2-3 kHz vibration of 0.1-0.4 mm (process damping
  dissipating 0.6 J per macro step, the cutting forces' work on the deflection rates +1.2 J), and at 5.29 s read a
  chip of 1.65 mm (forces 90 kN, 284 evaluations above the table) and died with exit 139 (the general protection
  fault). Bisection by switches in the run script, everything else equal: with the chips' occupancy grid emptied
  (no recut) the drill cut smoothly to 5.39 s (station deflection 4-10 um, forces rising 190 -> 512 N as the point
  engaged); with process damping off but the strikes on it survived the strikes with 1-5 m/s bursts that decayed; the
  L0 stride edit (0.125 deg) changed nothing (dynamics' rule already steps 2 us for the drill's modes). Three causes,
  fixed: (1) **which chips a tooth meets**: every chip inside the envelope cylinder was cut once per tooth period, a
  drill's own flute chips included; now only the layer the edges advance into through air (cutting.md "Recutting");
  (2) **the strike's force**: even a few micrograms took the section force of whole chips, 431 N for 50 us from
  13 ug near the axis, which the drill's light modes turned into 0.16 mm; the chips are spread over the voxel, so the
  edges meet them over the time they sweep the band's arc: the mean force (about 6 N for that strike), bounded by the
  section force; (3) **a diverged state read outside the coefficient table**: `int((h - h_min) inv_dhi)` of a
  1.65 mm (later 649 km) chip wrapped; the index is compared before the conversion and a chip beyond the tool's
  diameter stops the run with a fatal finding (`CA_H_BEYOND`). With the three: the first spot cuts steadily
  from 4.99 to 6.49 s, station deflection 4-21 um, thrust rising to 1.03 kN as the point engages fully (EXPECTED
  0.8-1.4 kN), strikes of 10-30 ug without a kick. Still open (cutting): process damping makes the drill's response
  to a large disturbance self-sustained (stable at small amplitude, a 0.1-0.4 mm vibration after a 0.4 mm kick);
  a likely cause is the wave memory's projection near a drill's axis, where the flank memory spans x / r of up to
  pi rad of the element's circle but is projected on the current normal (cutting.md 10); and the drill's analytic
  chip volume is 1.4-5 times below removal's field volume in its countersink entry (removal's mass-balance warning,
  36 times in 1.5 s; not isolated). The example end to end (run_project, budget 8 s, 1925 s wall, 1.94 GB peak RSS
  of the tsafe scope at 1.8 GB + swap): the spot cuts from 4.99 s, chip packing flags the flutes full at 6.39 s
  (fill 1.00: a spot drill's G82 without pecking), the Z axis is asked for 27 kN at 6.877 s (limit 13.7 kN) and at
  6.88 s the divergence finding stops the run (a chip of 41 mm, 4948 evaluations); thrust peaked at 1.18 kN (EXPECTED
  0.8-1.4 kN, passed), torque 2.51 N m when it stopped (2.6-4.6, not reached). What drives the last 10 ms (the
  packing load, the self-sustained vibration above, or the axis) is open.
  fixed (2026-11-10, dynamics; isolated on the tree of 8d51cc1 from checkpoints at 6.6 and 6.79 s, each variant
  resumed from the same state): none of the three. The run diverged at 6.80 s (6.88 s in the run above: the same
  kind of event, the tree had changed); resumed without chip packing it diverged at 6.80 s as before, without the
  recut grid at 6.87 s, without process damping it cut steadily past 6.9 s (packing loads at the divergence: 1.6 N,
  0.01 N m; the 27 kN on the Z axis are the consequence). Logged L0 step by L0 step (run_macro and contact_steps in
  Python around the compiled kernels, from 6.79 s): steady 1.10 kN thrust, then from 6.798 s the lateral force
  alternates in sign from one 2 us step to the next and grows by 8 % per step (+-270 N -> +-1.4 kN in 12 steps, the
  station rotation rates 23 -> 87 rad/s), a numerical relay, not a vibration of the drill. Three steps, each
  measured on the same state:
  (1) process damping's flank contact (K_sp dS L per element with the remembered surface held, 5.4e7 N/m on the
  drill) was missing from the predictor-corrector's stiffness estimate: added (docs/models/dynamics.md 2.3), the
  divergence moved to 6.95 s;
  (2) a contact-stiffened L0 stride rule (2 pi f_c dt <= 1; f_c 19.3 kHz, 0.24 at the 2 us step): not binding, the
  run still diverged at 6.95 s; and the estimate existed only in the steps where removal subtracted (the spot
  drill's teeth subtract every 36 ms against 10 ms steps): dynamics now keeps the last subtracting step's contact
  while cutting reports the tool in cut (`_engaged_contact`); with it alone the divergence came at 6.84 s;
  (3) the cause: the residual flexibilities take the last evaluated forces (one L0 step behind), and the tool's
  residual at the contact station is 1.1-1.5e-8 m/N, so against the contact the explicit iteration's gain is of
  order one (the alternation's -1.08: a restoring sensitivity of about -8e7 N/m laterally), whatever the step. The
  corrector now stores the residuals' forces at the fixed point F_(n-1) + (I - K C)^-1 (F_n - F_(n-1)) with the
  flank contact's K (docs/models/dynamics.md 2.3, `modal.residual_relaxation`). Resumed from 6.79 s the spot drill
  cuts its first spot to Z-4.5, dwells and retracts (7.8-8.0 s) and the run reaches its 8.5 s budget without a
  divergence (wall 1914 s for 1.71 s; thrust 1.10 -> 1.36 kN at 7.5 s against EXPECTED 0.8-1.4 kN, vibration 2-7 um
  peak to peak in the cut). New on the way: gouges of 0.008-0.072 mm (0.002-1.95 mm^3) once per tooth period from
  7.55 s to 7.93 s on the countersink of the first hole (CAD 9.0 mm, +-0.1 mm; not examined: the tool's tip geometry,
  the spindle's thermal growth or the countersink's CAD), and the flutes packed (fill up to 2.5, packing force
  60-70 N). With the cut's linearised sensitivity in K as well the first relaxed step blew up: the delay model's mean
  on this drill is not restoring (trace +4e8 N/m, open, dynamics.md 2.3).
  The drill's mass balance (field volume 1.4-5 x the analytic chip volume) was removal's pass bookkeeping, not
  missing chips: a pass ended at the first macro step without a subtraction, and the spot drill's teeth subtract
  every 36 ms (830 rpm, two lips) against 10 ms steps, so each piece compared one tooth's field volume with the chips
  of one or two steps (docs/models/removal.md "Volume consistency"; the operation driver's totals of the same drill,
  summed over every step, agree within 3 %). The wave memory's projection on the current normal (cutting.md 10) is
  still open: near the axis it spans up to 1.3 rad of the circle for the inner lip elements of this drill (the chisel
  elements' normals are axial and do not turn).

* **Structure's refresh dominates every run's cost** (open, structure; measured 2026-11-09 with the loop's own CPU
  split): pocket-islands' first 12 s (probing, no cut) took 954 CPU s, 818 s of it in structure; the lathe-shaft
  example's first 5.6 s (2.5 s of it in the G71 cut) 1308 s: structure 445, dynamics 220, cutting 200, recording
  119, hot loop 93. Structure re-solves its work model (a Ritz solve of 49 824 DOF, about 31 s) every 50 macro steps
  although nothing changed (identical first frequency every time): its input key hashes the thermal field's bytes
  exactly, and micro-kelvin changes make every refresh a new job. With it a whole operation of these examples
  costs about 10 hours of CPU (240 s per simulated second in the lathe's cut).
  fixed (structure; 2026-11-09 and 2026-11-10): an unchanged model is no longer solved again (`work.model_key`: the
  cells, supports, clamped state and material; only the thermal displacement is solved, 0.9-1.1 s per job instead
  of 27-31 s at 47 000 DOF, docs/performance.md 11), and a refresh after cells flipped starts with block steps
  preconditioned by the multigrid V-cycle, so the exact check passes after one block solve (9.1-12.6 s for 5-300
  removed cells at 46 662 DOF, was 12.9-30.5 s; frequencies within 7e-8 of a full solve; docs/models/structure.md
  3.5). The refresh's cadence and latency are in run time (500 ms each; they were 50 macro steps, so at 50 ms steps
  the work model was refreshed every 2.5 s and applied 2.5 s late, docs/convergence.md 9 item 13).
* **Removal's ledger volume and contact geometry** (fixed 2026-11-10, removal; docs/convergence.md 9 items 4 and 12):
  the booked removed volume of a thin finishing layer came from the z family (whole node rows: -18 % at 0.25 mm) and
  is now the field's ledger volume (per dual cell the family most normal to the surface; the side finish's removed
  volume within 0.28 % at every lattice, the pocket corner within 0.06 %); the loop's and the driver's exact-volume
  checks read the same volume (`_volumes` in core/loop.py and core/drive.py: one line each). The contact point,
  normal and depth of cut of a milling step came from the z family and the rows' ends: in the convergence case's
  side step at 5 ms steps 37 of 142 steps published the point at the S origin and ap alternated 0.75 / 0.88 /
  1.01 mm; now no step has a zero point, ap is 1.007 mm in every step and the point moves by 0.03 mm across the
  steady steps (was 0.25 / 0.50 mm in y / z); thermal's preheat band, the surface temperature and the work deposits
  read them.
* **Surface temperatures of fresh edges** (open, thermal): thermal's surface temperature (the feed-band rise plus
  Jaeger's flash rise of the work's heat over the flank contact) reports 887 degC at ti-slot's entry with 22 W of
  work heat (flash rise 797 K), 981 degC in warmup's turning pass, 589-700 degC (melt smear of 6061) in batch-50's
  roughing entry. With VB 0 the flank contact is `flank_contact_edge_factor` x the hone radius wide (tens of um),
  which concentrates the flank and ploughing heat; the estimate has not been compared with measured machined-
  surface temperatures and its damage findings (burn, white layer, melt smear) are not trusted until it is.
* **Wear from the entry spikes** (fixed 2026-11-09: the spikes are gone at their source in cutting; wear's own
  bounds: the edge no hotter than the work's melting, Basquin's curve ending at one reversal, the sliding distance
  travelled; docs/models/wear.md 2, 8, 9): the regression's runs that stop at their first cut's entry
  report wear that a few milliseconds cannot give: examples/thin-wall-al7075 at 4.43 s `VB_max_m` 146 (metres;
  notch 22.6 m, "dimensional drift 6334.931 mm"), ti-slot's T1 "worn out" after 4 ms (edge chipping at 829 N),
  drill-tap's D6.8 with a fatigue damage of 1.7e24 after its blown-up entry. The loads are the spikes of the entry
  (cutting's recut impulse, the process-damping relay, above); `cnctwin/wear` integrates them without a bound of
  its own (a VB larger than the insert, a wear rate from one step's peak). The tiny batch's tool, which comes in
  used (720 s, VB 131 um), wears by nm per part, as it should.
* **Wear, thermal and coolant of the flooded aluminium side step (fixed 2026-11-10; docs/convergence.md 9 item 9).**
  (1) wear's Taylor response computed the flank temperature with Blok's share of the land friction alone while the
  run partitions the flank friction and the edge's ploughing work over VB + arc: it now calls the run's partition
  (`mechanics.coefficients.flank_contact_heat`, docs/models/wear.md 3) and the database laws are recalibrated on it;
  (2) chipping took a shock row's force per unit width over the tooth's elements in cut in the macro step, so a pass
  split by a step boundary chipped more at short steps: the width is now over the last revolution (wear.md 7); (3)
  continuous elements of the cutting-region field published their own in-cut temperature as their entry temperature,
  a double count in mechanics' partitions: now the field less their own steady rise, bounded by the body temperature
  (thermal.md 2.2); (4) the flood never reached the cut (friction factor 1.0 at every step): the coolant -> contact
  feedback was right, the VF-2SS's flex nozzles kept an aim that missed the D10 by 26.6 mm; hand-set nozzles are now
  aimed at the tool in the spindle (`physics.coolant.aim_hand_nozzles`, coolant.md 3): reach 0.09-0.11, friction
  factor 0.93-0.95. Five side steps in the whole loop (about 29 s of cutting) with thermal's own temperatures: VB_max
  20.0 -> 24.1 um, no chipping, about 2.5 um of it at the pass entries where thermal's flank spikes for 20-40 ms
  (open, thermal), about 50 nm per second of cutting between them. For cutting's owner: with the contact lubricated
  (friction factors below 1 that change from step to step) cutting's own energy identity on the tiny mill project
  closes to 1.22e-9 of its terms (tests/integration/test_loop_run.py `test_every_ledger_passes` asks < 1e-9) where
  it closed to 5.7e-11 unlubricated (`aim_hand_nozzles: 0`); the terms are cutting's (hot loop heat by zone against
  shaft and feed work), not the coolant's.
* **A fresh roughing end mill at its end of life 0.08 s into its first cut** (fixed 2026-11-10, wear;
  examples/pocket-islands-al6061, T1 D10 3FL at 12 000 rpm, helix at f_z 0.047 mm, sharp corner, hone 4 um, the
  example's 0.02 mm initial land): two errors in the criteria, measured on the run itself (wear's inputs logged per
  macro step from the first contact at 13.72 s). (1) The finish estimate read Brammertz's formula beyond its range:
  R_t = f^2/8r + (h_min/2)(1 + r h_min/f^2) holds for f >= sqrt(2 r h_min) (140 um here: r is the tool radius,
  5 mm, for a square end mill), and below it grows as r h_min^2/2f^2; with the worn hone rising from 8.5 to 9.5 um it
  gave Ra 1.38 -> 1.68 um, "1.05 of 1" against the floor's Ra 1.6 um at 13.80 s. Below that feed the profile is the
  sawtooth of height h_min that the surface domain's kinematic profile already gives (docs/models/surface.md 2):
  Ra 0.49 -> 0.55 um there (`wear.life.roughness_m`). (2) The limits were the part's: every element that wrote a
  surface event "finished", and every tool took the part's tightest tolerance and finish, so the roughing tool,
  which leaves 0.2 mm on the floor and the walls, was held to the floor's Ra and the islands' 0.02 mm flatness;
  with (1) alone it reached its end of life by dimensional drift at 14.04 s (10.0 um of recession, the initial land
  growing, against 10 um). The tolerances now apply where a tool generates the toleranced feature's final surface
  (surface events on the feature's plane within 0.05 mm, docs/models/wear.md 10). Resumed from 12.5 s with both
  fixes the run cuts to its 14.5 s budget with no end-of-life finding and verdict safe (T1: no finishing elements on
  a toleranced face, Ra estimate 0.80 um at 14.5 s). Open (wear's owner): the hone radius of this tool grows from
  8.5 to 14 um in 0.3 s of cutting (edge rounding; at that rate a finishing tool reaches Ra 1.6 um, h_min = 0.23 x
  28 um, within seconds).
* **A zero-point plate does not clamp** (holding's devices and the workholding database, open): examples/
  die-h13-ballnose holds its insert on a Lang Quick-Point 52 grid plate, "pulled down by two d16 clamping studs"
  (the entry's label). db/workholding/lang-quick-point-52.yaml types the plate `fixture_plate` with its size, grid
  and mass and no clamping data, and holding's `build_fixture_plate` gives a plate only its base contacts with
  friction: the insert stands on the plate by its weight alone, and the rapids' accelerations slide it
  (part_slip three times, part_free, before the first cut). The fix needs the system's pull-down force per stud
  (Lang's catalogue) in the database and a device that turns the studs into clamp contacts pulling the part onto
  the plate; not done here (no sourced clamping force at hand), so that example's EXPECTED power and force items
  stay out of reach.
* **Corner rounding and test references.** The controller's look-ahead rounds program corners within the
  machine's tolerance. Since 2026-11-09 Haas's Setting 85 is the corner radius it documents (docs/models/motion.md
  2.2): at the VF-2SS's 90 degree corner at F1600 the radius the axes need, 0.436 mm (deviation 0.18 mm, actual
  path 0.19 mm), where it was a 1.53 mm blend passing the corner at 0.635 mm (actual 0.646 mm). The tests take the
  removed-volume reference from the path recorded in frames.bin, not from the program.
* **Memory per macro step** (fixed 2026-11-09, motion and chips): with 2 ms macro steps a run exceeded the 1.8 GB cap
  at its tool change (docs/convergence.md 9 item 10): motion booked the tip's jump to the new tool as one tick's
  feed velocity (102 m/s) and chips sized its occupancy grid by the travel at it over the lag (8e7 voxels). Measured
  after the fix on the convergence case's air-cut variant at 2 ms: 2500 macro steps through the tool change, RSS
  944-983 MB, trend -13.5 kB per step after it (allocator noise), the largest step +36 MB at the tool change itself
  (the new tool's tables). tests/integration/test_loop_memory.py.
* **Checkpoints and resumes doubled the material field** (fixed 2026-11-10, removal and the loop; measured on
  examples/thin-wall-al7075, whose two refine boxes give 8.3 M rays: field 0.57 GB, its CAD target 0.15 GB). Nothing
  grows in that run: per-owner array sizes logged every 0.1 s from 4.2 to 6.3 s stay flat (removal 734 MB, structure
  19 -> 26 MB at its first refresh in cut, dynamics 6 MB, surface 2 MB, chips 2 MB); the process's anonymous memory
  is 1.48 GB before the first cut, 1.65 GB at 6.3 s: the field, about 0.6 GB of compiled kernels (522 Numba functions
  loaded from the cache: hot loop 0.2 GB alone, the domain imports 0.3 GB, tooling's 0.3 GB) and the domains. What
  killed the earlier runs was a copy: `removal.field.to_arrays` copied every field array for the checkpoint
  (`np.array`), so the end-of-run checkpoint (and every periodic one) asked for another 0.72 GB (the 2.1 GB peak); and a
  resume built the scratch state's field before reading the checkpoint's (1.9 GB, killed before its first step).
  The checkpoint now writes the field's own arrays (they do not change while it is written; the operation
  driver's in-memory `checkpoint()` copies them, it is held while the run goes on), and a resume drops
  the scratch state's fields before reading (`Run._release_scratch_fields`: 1.37 -> 0.65 GB before the read). With
  both, the run checkpoints at 4.4 and 6.0 s and ends at its budget (anonymous memory 1.66 GB at 6.5 s); resumed
  from 4.3 s with a 60 s budget it cut to 56.68 s with anonymous memory flat at 1.68-1.69 GB and a checkpoint every
  10 s, and stopped there on a divergence (Y axis 23.6 kN demanded, chip thickness 185.7 mm; section 13, open). Still at
  the edge of the 1.8 GB cap: the end-of-run full keyframe (every surface brick meshed and held until written) took
  examples/pocket-islands-al6061 from 1.38 GB to 1.85 GB (with 0.25 GB in swap) at 14.5 s and killed its run at 16 s;
  the compiled kernels' 0.6 GB is the largest item after the field (open).
* **`run_project` ignores `options.operator` (for the loop's owner; 2026-11-09).** The loop's `run_project` does
  not read the operator overrides. `twin run` and the API's default `simulate` entry now send a run whose options
  carry them through `cnctwin.sweep.runner.run_project`, which applies them and refuses overrides the machine or the
  tools cannot take (docs/howto-run.md section 3, docs/api.md), but a direct caller of
  `cnctwin.core.loop.run_project` (an uncertainty study or a calibration with the loop as runner, a test) still gets
  the plain run without a word. Exact change in `cnctwin/core/loop.py`, at the top of `run_project` right after
  `options = dict(options or {})`:

  ```python
      if options.get("operator") and not options.get("replay"):
          from ..sweep.operator import normalise
          if normalise(options["operator"]) is not None:
              from ..sweep.runner import run_project as with_overrides     # applies them (OverrideRun)
              return with_overrides(project_path, run_dir, options, progress_cb)
  ```

  It does not recurse: the runner's non-replay path builds its `Run` subclass itself and calls `loop.run_project`
  only after dropping an operator option at 100 %, and its replay path (which calls `loop.run_project` with the
  option under the patched `Run`) is excluded by the `replay` test. Test: `run_project(project, run_dir,
  {"operator": {"feed_override": 1.2}})` returns `operator.batches > 0` and run.json `options.operator`.
* **Calibration overlays the project includes (for structure's and the loop's owners; 2026-11-09).** A project's
  `calibration: <file>` is applied by `cnctwin.project.resolve` (docs/formats.md "Project file"): machine, material
  and tool values arrive in `ctx.project` already calibrated, so `Run` needs nothing for them. Two kinds of entry
  have no database object and wait in `ctx.project.calibration_overlay` (None without the key), each reported as a
  `calibration` warning by the resolver until a domain applies it:
  * `factors.tool_stiffness[<T>]` (`calibration_overlay["factors"]["tool_stiffness"][str(T)]`, the calibrate
    target `tool_stiffness_factor`): exact change in `cnctwin/structure/toolmodel.py` `tool_modes(...)`, a keyword
    `stiffness_factor: float = 1.0` applied before `solve_modes`: `sys.K = sys.K * stiffness_factor`,
    `sys.K_full = sys.K_full * stiffness_factor`, every `(name, eta, K)` of `sys.parts` and `sys.parts_full` with
    `K * stiffness_factor` (the masses and the base modes unchanged: the static tip stiffness and the modal
    stiffness scale by the factor, the frequencies by its square root); and in `cnctwin/structure/domain.py`
    `_build_tool`, the call `TM.tool_modes(..., stiffness_factor=_tool_stiffness_factor(ctx, slot_obj.number))`
    with `_tool_stiffness_factor` reading `((getattr(ctx.project, "calibration_overlay", None) or
    {}).get("factors") or {}).get("tool_stiffness", {}).get(str(number), 1.0)`. Then the resolver's warning for
    `factors.tool_stiffness` can go (`cnctwin/project/loader.py` `_calibration_record`: skip the entries a domain
    reads).
  * `models.<model>.<name>` (`calibration_overlay["models"]`, e.g. `production-energy.axis_seal_drag_N`): the
    readers of `cnctwin.db.models.params(model)` for a run (production's energy model) should merge them over the
    file's values; `cnctwin.production` is not a loop file, the hook it needs is the project it is given.
  * The calibrated values are `src: est` values of the overlay file, not of a database file, so `_estimates` in
    `cnctwin/core/loop.py` does not list them in run.json `estimates` (the report's "Estimated database values
    used"). Exact change at its end: `calibration = getattr(project, "calibration_overlay", None)`; when set,
    append for every `path, entry` of `calibration["set"].items()` a `cnctwin.db.problems.Estimate(file=
    calibration["file"], line=0, path=path, value=entry["value"], note=f"calibrated ({calibration['name']})" +
    (f", parameter {entry['parameter']}" if entry.get("parameter") else ""), obj="calibration")` to the list it
    returns (`_run_document` writes file, path, value and note of each).

## 16. Parameters and settings

The loop composes the domains; its own values are settings of its checks, its ring and the cost model of
`twin check` (db/models/loop.yaml, read by `cnctwin.db.models.params("loop")`), and project settings of the run.

| name | value | source |
|---|---|---|
| `coupling_ratio` | 0.5 | cnctwin-architecture-2026 (5.3: two successive changes of opposite sign with a magnitude ratio above 0.5) |
| `coupling_rel_change` | 1e-3 of the quantity's value (range 1e-9-0.1) | est: below it an alternation is round-off or the noise of a converged quantity (tool temperatures settle to 1e-6 between steps) |
| `coupling_alternations` | 3 (range 1-100) | est: one alternation follows any single disturbance (a tool entering the cut) |
| `chip_kinetic_in_balance` | 0 (reported, not balanced) | est: cutting's force model has no momentum flux (section 6) |
| `energy_scale_floor_J` | 1e-6 J | est: the smallest scale an interval's relative residual is taken against (out of the cut the modal terms are picojoules; the lathe warmup: 6.7e-18 J residual on 3.0e-12 J terms) |
| `probe_stem_diameter_ratio`, `probe_ball_diameter_m`, `probe_stylus_length_m` | 0.75, 6 mm, 50 mm | est (docs/models/hotloop.md 9a) |
| `ring_f_max_Hz` | 20 kHz | est: tool bending modes of short carbide tools reach 5-10 kHz, spindle flange modes 1-3 kHz |
| cost model `cost_l0_fixed_s`, `cost_l0_element_s`, `cost_l0_mode_s` | 7 us, 0.29 us per engaged element, 0.4 us per mode (per L0 step) | est, measured on the D10 slot (docs/models/hotloop.md 13, cutting.md, dynamics.md) |
| cost model `cost_l1_tick_s`, `cost_l2_step_s`, `cost_l2_idle_step_s` | 0.9 ms per L1 tick out of contact, 0.2 s per macro step in the cut, 41 ms out of it | est, measured on warmup's mill project (section 13) |
| `physics.tol_mass`, `physics.tol_energy` | 1e-9, 1e-6 (relative) | project template; cnctwin-architecture-2026 (12) |
| `checkpoint_interval_s` | 10 s of simulated time | db/models/recording.yaml (cnctwin-architecture-2026, 10) |
| `recording.keyframe_period_s` | 0.5 s | project template |
| `physics.lag_macro` | 1 (2 is accepted and computed as 1) | cnctwin-architecture-2026 (5.3); section 15 |

## 17. Assumptions, validity and error bounds

| assumption | consequence | validity | error bound (how established) |
|---|---|---|---|
| The slow domains run after the step's hot loop and read `bus_prev` (one-step stagger; `lag_macro: 2` is computed as lag 1) | every slow coupling closes one macro step (10 ms) late | `physics.macro_ipo` 5-20 ms | tool state and the 0.5 s means of the coupling temperatures within 0.2-1.5 % from 5 to 20 ms; values at one instant of the rake temperature and the preheat scatter 1-2 %, the workpiece bulk rise and the edge peak's 0.5 s mean move 2-6 % (docs/convergence.md 5); lag 1 and 2 identical |
| The time between operations is not simulated time; thermal and coolant cool down on their own clocks, production's operator model books the handling | the run's clock is machine time only; `timeline.json` has no gap between operations | any project | exact (time balance in integer ticks) |
| An operation starts where the last program left the axes, drives at rest | no jump of the axes between operations (which holding would read as ~1e5 m/s2) | mills and lathes, one machine per run | - |
| A new part starts from a fresh blank and carries tool wear, controller state, machine and tank temperatures; the workpiece grid is reset | a batch is one continuous machine history | `production.parts` of one or more operations | three-part batch: equal removed mass, every ledger over the parts (test_loop_batch) |
| An alarm stops the run; `max_sim_s` ends the running operation without inspecting the unfinished part | a stopped run has no later operations, parts or part files of the unfinished part | any project | - |
| The chips' kinetic energy at separation is reported, not balanced | the energy balance omits a term of about 1e-5 of the shaft work at aluminium speeds | until cutting's torque carries the chip momentum force | below the 1e-6 energy tolerance by construction |
| Cutting's quadrature of the work on the deflection rates and the modes' exact work differ; the difference is its own ledger term | the balance closes to round-off | runs with dynamics | the difference is 0.1-3 % of the vibration work (docs/models/dynamics.md "Ledger note") |
| Thermal's and structure's workpiece grids stay in the frame they were built in after a removal transfer | after a flip the first operation's workpiece temperatures are misplaced by the turn | multi-setup projects | not bounded (the part has cooled for `time_between_ops` by then) |
| Defaults where no adapter exists: holding's PartGeometry (the stock's shape), coolant's WorkpieceBricks (the stock box), inspect's MeshColumns (the result mesh dexelised at 0.25 mm) | those domains see the stock, not the cut part, where these protocols act | until the adapters exist | not bounded |
| `twin check`'s cost model: per-step costs measured on one project; in-cut time = the dry run's feed-move time | an estimate of CPU s per simulated s, the in-cut part an upper bound | projects like warmup's mill (about 9 CPU s per simulated s) | the examples cost 7-60 CPU s per simulated s (section 14); the first model (the domains' own cost notes) said 2.3 where the run took 7.4 |
| `twin check`'s lattice rule decides on sanity's engagement estimate at lattice / 4 | a pass is warned about from its estimated ae, recommended from its lower bound | cutting moves with ae < 4 x the lattice in force | the estimate is within one cell of the true ae; the ledger's removed-volume error bound is lattice / (2 ae) |

Checks of every run (section 6) and their tolerances: mass relative to the removed mass <= `tol_mass` (1e-9;
measured 4.5e-16 on warmup's mill project), energy per interval <= `tol_energy` (1e-6 against at least the
1e-6 J scale floor; measured 4.3e-9), time exact in integer ticks, exact volume (integer quanta) to 1e-9,
cutting's identity reported (5.7e-11 dry, 1.2e-9 lubricated; bounded at 1e-8 by test_loop_run). Removed
volume against the swept path of frames.bin within 3 % (test_loop_run; 6.106 against 6.105 mm3 measured).
Resumed runs equal the uninterrupted run byte for byte (test_loop_determinism).

Validated range of the whole loop: the tiny projects of the tests (a side step, a 3 mm OD pass, a two-setup
flip with one side step per operation, a three-part batch) run complete and balanced; the long examples run to
an 8 s budget, and none completes an operation until cutting's recut terms are fixed (section 14).

## References

* cnctwin-architecture-2026: the contract the loop implements (5.1 clock, 5.3 macro step and coupling-lag
  monitor, 10 recording and checkpoints, 12 ledgers, 16 run directory).
* renishaw-omp40-2-2025: the spindle probe the probe defaults stand for.
* The domains' own references for every model the loop runs: docs/models/*.md.
