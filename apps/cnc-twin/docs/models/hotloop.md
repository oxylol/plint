# Hot loop (`cnctwin/core/hotloop.py`, `hot_abi.py`, `hot_kernels.py`, `drive.py`)

The compiled composition of the per-tick and L0 kernels of motion, cutting, removal and dynamics
(architecture 5.1-5.5, 7, 9, 12): one Numba function, `run_macro`, advances the L1 ticks of one macro step
with every contact's L0 steps inside; a Python wrapper, `HotLoop`, builds its frozen inputs from each domain's
`kernel_inputs`, calls it in chunks between the controller's sync ticks and hands the results to every
domain's `absorb`. `core.drive.OperationDriver` runs one operation of a real project through it (controller
-> motion -> hot loop -> absorb -> the L2 macro steps of motion, cutting, removal and dynamics) and checks the
ledgers every macro step.

| file | what |
|---|---|
| `hot_abi.py` | layouts of the loop (`CutArgs`, `ContactGeo`, `ContactState`, `HotContact`, `HotGlobal`), index sets, split codes, Numba types |
| `hot_kernels.py` | compiled helpers: Hermite angles, axis interpolation, FK frames, step poses, displacement grid, deflections, slab points, refill planes, polar views |
| `hotloop.py` | `headroom`, `do_pass`, `update_gamma`, `regate`, `tooth_events`, `contact_steps`, `run_macro`; `HotLoop`, `StepReport`, fast channels |
| `drive.py` | `OperationDriver`, `Feeder`, `StepRecord`, `DriveResult`, `fixture_pose_F`, `project_variant` |

## 1. Layouts (`hot_abi.py`)

Every layout is a NamedTuple of C-contiguous arrays (architecture 5.5); `run_macro` has one explicit
signature `i4(MOTION_T, REMOVAL_T, CONTACTS_T, RING_T, GLOBAL_T)`.

* `MOTION_T`: motion's `MotionInputs` (one per macro step, `motion.kernel_inputs`).
* `REMOVAL_T`: removal's `RemovalInputs` with the per-workpiece tuples padded to `MAX_WORK = 2` and the
  per-path tuples to `MAX_PATHS = 2` (homogeneous tuples: runtime indexing).
* `CONTACTS_T`: a homogeneous `UniTuple(HotContact, MAX_CONTACTS = 2)`, one contact per controller path with a
  tool; inactive contacts are padding of the same types (`empty_contact`, made once per axis count and reused:
  `CI_ACTIVE = 0`, no kernel reads or writes them).
* `HotContact`: `ipar` [`CI_*`] (path, slot, workpiece, tool kind, rotating, tool / work spindle, teeth,
  elements, stations, L0 stride, dynamics on, recorder columns, polar, stale-before, probe), `fpar` [`CF_*`]
  (hand, cutting radius, refill reach, lattice, work-spindle sign, gating parameters, feed fraction, polar
  signs, tool mount), `cut: CutArgs` (cutting's tool, table, memory, intervals, refill, step scalars, pose,
  deflection, occupancy, packing, axes, state, accumulators, outputs, event buffers, polar view), `sys`,
  `base`, `grid` (dynamics' system, axis base, work grid), `geo: ContactGeo` (stations, slab -> station map,
  slab lags, the path's machine axes and their direction in S), `cst: ContactState` (the loop's own state:
  generation angle per tooth and slab, recorded slab points, slab done flags, last pass tick, polar view
  window, counters `CS_*`, angles `CSF_*`), removal's `SubtractSpec`, gate parameters, `work` [128] (frames
  and vectors of the step and the loop's scratch vectors, `hot_kernels.WK_*`), base-excitation inputs, FK scratch,
  polar step, refill plane list, probe profile, `teeth` [nt, `TW_*`] (per tooth: engaged and recorded slab counts
  and the window of the slabs' phases; derived from `ContactState`, not checkpointed, section 4).
* Every kernel of the loop (`run_macro`, `contact_steps`, `tooth_events`, the helpers) allocates nothing (scratch
  vectors are views of `work`) and is compiled without reference counting (`RAW`, `_nrt=False`, as removal's hot
  kernels): the loop body takes members of large NamedTuples, which otherwise costs an atomic incref / decref per
  member array. The per-step helpers of `hot_kernels` take the few layouts they read, never the whole `CutArgs`
  or `HotContact` (a call that is not inlined copies every member array descriptor: ~1 us for a `CutArgs`).
* `HotGlobal`: `ipar` [`GI_*`] (tick0, ipo ticks, resume point, buffer row bounds), `fpar`, the carried L1
  loads per axis and spindle, the tick-start axis values and spindle angles, the tool tips (probe sweep), the
  structure displacement grid d(X), `result` [`RES_*`], counters `CNT_*`, per-tick in-cut flags.

## 2. One macro step (`run_macro`)

Per L1 tick k (from motion's `k_from`):

1. resume handling (a split returned inside this tick: jump to the contact and L0 step of `GI_RESUME_*`);
   motion's `replan_before`; save q0, spindle angles and speeds; `motion_tick` (setpoint, servo with the
   previous tick's axis loads, spindle loops with the previous torque, FK, out.tip_S);
2. G31 probe sweep (`sweep_contact`) of every probe-only contact (`CI_PROBE`, section 9a) while its path runs a
   G31 move (`setpoint.probe`): the first contact of the stylus with the field sets `out.probe_hit` and returns
   `SYNC_PROBE_TRIGGER` at that tick (motion latches the encoder position plus the pre-travel);
3. per active contact the broad phase: removal's `l1_contact` and `collide_envelope` (rotating tools), the
   polar field's exact block maxima (`polar_max_radius`) for turning;
4. in contact: at the start of an engagement the engaged intervals (`engaged_intervals`), the displacement
   d(X) at the tip, dynamics' `refresh_work_shapes`, then `contact_steps` (section 3); out of contact
   dynamics' `modal_free` (exact free propagation with unbalance and base excitation);
5. dynamics' `record_tick`; the tick's mean cutting load per machine axis (the force in S projected on each
   axis direction of `geo.cut_axes`) and the spindle torque (`-tq`; turning `-tq * work_sign`) become the
   next tick's `dist_force` / `dist_torque`; `out.in_cut`; tip0;
6. a motion sync code returns (the controller runs in Python, `HotLoop.run_step` calls again).

## 3. L0 steps of a contact (`contact_steps`)

At the contact's stride `l0` (dynamics' `l0_ticks_rule`: largest divisor of ipo_ticks with |omega| dt <=
max_deg_per_step, steps_per_tooth per tooth period, 2 pi f_max dt <= 0.3; `set_l0_ticks` tells dynamics):

* headroom check of every event buffer (section 6);
* spindle angles by cubic Hermite between the tick's end values (angle and speed), axis values interpolated
  the same way, FK at the step middle and end (`fk_frame`, the tool mount applied), cutting's `StepPose`
  (pose at the middle, velocities from the end frames);
* dynamics' predictor `modal_begin` (unbalance, base excitation from the path's axis accelerations,
  predicted deflections at the step's stations, `DynSystem.act`); the deflections (read in place by cutting
  when the tool mount is the identity, `CI_DEFL_SHARED`) and the workpiece displacement (+ d(X)) in cutting's
  `Deflection`;
* rotating tools: refill planes (section 5), `cut_step`, `clear_planes`; non-rotating tools: `turn_step`;
* dynamics' corrector `modal_correct` with the step's station loads (first-order hold);
* the recorder ring row (fast channels `hotloop.force_T0_N`, `hotloop.torque_Nm`, dynamics' columns);
* removal: tooth events (section 4) for envelopes, `helix_subtract` per step for taps, `polar_cut` per step
  and the delayed view update for turning (section 7).

## 4. Tooth passes (generation instants)

Removal subtracts each tooth's envelope at the axis points `X_e + u_rel` the slabs had when they generated
the surface (architecture 4.1, 5.4). The generation angle of a slab is the exit of its station's main engaged
interval in the direction of rotation (`update_gamma`; NaN when the station is not engaged). After every L0
step the stateless crossing test `d0 = sigma (gamma - (theta0 + a0)) mod 2 pi < |dtheta|` (a0 the slab's
angular lag on the helix) finds the slabs that passed their angle in the step; the slab's point is taken at
that fraction of the step (`slab_point`: tip, axis and u_rel interpolated between the step's end frames). A tooth
is visited only when the step's swept angle reaches the arc holding its engaged slabs' phases sigma (gamma - a0)
(`HotContact.teeth`, `tooth_windows`, rebuilt whenever gamma or the recorded flags change and once per call; margin
1e-8 rad): the exact per-slab test runs for that tooth only, and the slab counts of the pass test are kept with the
slabs' states, so a step without a generation instant costs O(teeth) (13.8 us -> 0.2 us per step in the slot).
A tooth's pass (`do_pass` -> removal's `tooth_pass`: engaged intervals refreshed against the field, sector box,
`subtract_envelope` per slab at the recorded points) runs once every engaged slab of the tooth is done, or is
forced when the tooth had no pass for `SI_STALE_TICKS` (a slab missed its angle while the engagement changed:
undone slabs take the step end; `CS_FORCED` counts them). Without a pass for a tooth period the engaged
intervals are refreshed alone (`regate`). The plunge flag of a pass comes from the feed direction in T0
(`WK_FEED` against `RQ_TAN_RAMP_MAX`); its surface event id from removal's per-workpiece counter.

The earlier rule (a pass when the tooth's last slab of the whole flute passed) left the field up to a flute
length of rotation behind the cut: the next tooth then saw material the previous tooth had already cut
(chip thickness up to 0.7 mm in the slot test, peak force +56 %).

## 5. Refill planes (stale memory bins)

Cutting reads the surface the previous tooth left from its surface memory; a bin older than
`SI_STALE_TICKS` (or an element that reads the field directly) needs the field (architecture 5.4). Before
each `cut_step`, `refill_planes` covers the elements of the engaged intervals (margin: half the step's
rotation plus one bin) and the four bins around each: for every stale bin without a plane in this step it
asks removal's interpolated boundary of the dexel family most aligned with the chip-thickness normal
(`surface_plane`: D_f, `gradient`, exact for planes, a^2 kappa / 4 on curved boundaries); when that is not
valid (the family's rays do not all see the boundary, the boundary does not face the element: `m . n > -0.3`)
the bin is refilled at the element itself (q = circle point, m = -n: zero chip until a tooth writes it). A
plane lives for one L0 step: its (row, bin) is listed in `HotContact.planes` and `clear_planes` removes it
after the force kernel, so no later pose reads it. Ray casts were tried first: they misclassify within a cell
of concave edges (the slot's floor corner) and fed chips of the full lattice step.

## 6. Splits (deterministic, no rollback)

Before every L0 step `headroom` checks that every event buffer (cutting's shocks, surface events, deposits,
recut, passes, refills; removal's events) has room for the rows one step may write (`GI_ROWS_*`, set by
`HotLoop._rows` from the tool: shocks 4 nt + 4, surface 3 n_el + 8, deposits 2 nt + 4, recut n_el + 4 nt + 4,
passes 2 nt + 4, refills 2 n_el + 4, removal 8 nt + 8; buffers hold at least 4 x the largest). If not, it
returns `HOT_SPLIT` / `SPLIT_DRAIN` with the resume point; the wrapper moves the rows into its own lists
(`_drain`) and calls again: the loop continues at that L0 step. A removal pool overflow in a pass
(`SPLIT_POOL`), in a helix subtraction (`SPLIT_POOL`) or a polar inline-segment overflow (`SPLIT_POLAR`) returns
too: the wrapper grows the field (`prepare_replay`), and only the refused removal is repeated at the resume, the
step itself is not: the pass (`CS_OVERFLOW_TOOTH`), the polar cut or the helix subtraction (`CS_OVERFLOW_REMOVAL`
with `CSF_PENDING_ANGLE`: `pending_removal` at the start of the resumed `contact_steps`, with the step's frames
`WK_A0` / `WK_A1`, its polar step and its angles, all untouched since; the turning view window after it,
`polar_window`), then the loop continues at the next L0 step. Before 2026-11-09 a polar or helix overflow
resumed at the same L0 step and repeated all of it: the step's forces, work integrals and axis loads were booked
twice and dynamics was stepped by 0 base ticks (the TwinError that stopped warmup's lathe pass and
examples/lathe-shaft-4140 at their first contact). No row is lost and nothing is rolled back, so the results do
not depend on buffer capacities (test f runs with small buffers; tests/integration/test_turning_split.py runs an OD
pass with one inline segment per polar ray, which overflows on the feed marks' scallops: it finishes balanced with
its booking exact; that it equals the pass with 16 segments is still open, docs/models/loop.md 15).

## 7. Turning

`turn_step` reads cutting's `PolarView` (outer / inner radius per bin and row) built on the polar field's grid
from removal's radial rays (`polar_view_update`). The polar field is cut every L0 step (`polar_cut`, the
insert region between the step's end poses); the view of a bin is refreshed from the field (the pending
window `polar_win`) once no element can read it any more: cutting interpolates between the two bin centres
around an element's angle, so a bin is refreshed when the lagging one of the reference plane and the deflected
tip has passed the centre of the next bin in the direction of motion; the bins under the tool still show the
previous revolution (cutting's requirement). Work speed for cutting: `omega_S = omega_work x work_sign`
(the tool turns by `-omega_S` about S z; `work_sign` from the chain). The polar step takes the deflected tip
(`tip - u_work + R_S_T0 u_tool(tip)`, the translation and rotation of the station nearest the tip; the insert
as a rigid body), as cutting composes its elements: with the modes on, the previous revolution is cut with the
same static deflection and the steady chip is the rigid one (without it the steady force came out 7 % low).

**Rays of several segments.** A radial ray of the polar field may hold several material segments. Behind
a major edge that leans back (kappa_r > 90 deg, e.g. the 95 deg PCLNR) each revolution leaves a thin ring
above its cut in the rows just behind its front (the outer part of the edge reaches those rows only at the
next pass, when the ring is part of that chip). Cutting's `PolarView` held one interval per cell: with the hull
(first start, last end) the major-edge elements' rays crossed the gap under the ring (chips up to 57 um, 23 %,
too thick near the nose; force +10 %); with the outermost segment the nose elements' rays lost the core under
the ring (chips down to zero). The view now carries one air gap per cell as well (`gap_lo_m`, `gap_hi_m`, an
additive field of cutting's layout, nearest cell in `polar_inside`): the hull and the gap between the two
segments the tool meets first (the outermost two for external tools, the innermost for internal ones; `side`
= the polar step's sx). Every element's chip then follows the exact surface of the earlier revolutions
(test c). A third segment on one ray is not represented (none in a steady pass).

**Turning tool frame.** Motion's chain ends in the turret station's frame (the machine's axes at home);
tooling's non-rotating tools have x_T into the work (toward the axis for external tools, shank along -x_T;
away from it for internal ones) and z_T from the tip into the tool body along the work axis (+S z for right-
hand and neutral tools, -S z for left-hand ones). The hot loop scales the chain's T0 axes by the tool mount
`(mx, my, mz)` (`CF_MOUNT_*`; a proper rotation: identity or a half turn, decided by `_turning_signs` where the
tip is off the axis and kept per path and slot): cutting's pose, removal's polar step (`sx`, `sz`) and the
deflections exchanged with dynamics (which works in motion's T0: the 6-vectors are sign-flipped both ways)
all see tooling's frame. Without it the insert region was placed inside the bar (x_T outward: removal took it
for a boring tool) and the view never showed the cut.

## 8. Field resolution

The tri-dexel field decides a column's top to a lattice cell: an element within one cell of an edge of the
stock (the stock top at the depth of cut, the cut front) sees a chip of the field's discretisation, not of the
feed. With lattice 0.25 mm and ap = 3.0 mm the slot's top station sat 0.04 mm below the top: a 6 % force
deficit. The tests use lattice 0.1 mm and a depth midway between two peripheral stations
(`hotfix.station_ap_mm`). The analytic chip volume of cutting's stand-alone driver counts the top element
fully (+2.2 % at ap 3.0); the field's volume is exact for its columns.

## 9. Python wrapper (`HotLoop`)

`run_step(feeder)`: the controller's commands for every path (`feeder.feed`), motion's `kernel_inputs`, the
global layout (once; restored after a checkpoint), `_build` (strides -> `dynamics.set_l0_ticks`; tooling,
dynamics, cutting, removal `kernel_inputs`; padded removal inputs; contacts, a rigid contact with dynamics
placeholders made once per shape (`CI_DYN = 0`: no kernel touches them); buffer rows), then `run_macro`
chunks: at a split `_handle_split`, at a sync motion's `absorb`, the controller actions, the next commands,
until `step_complete`. `_absorb`: the drained rows merged into new event buffers for `cutting.absorb`,
removal's event buffer swapped in for `removal.absorb`, `dynamics.absorb` with the step's accumulators (or an
empty frame for rigid runs). `StepReport` carries chunks, splits, syncs, strides and CPU times (`cpu_s`,
`cpu_macro_s`, `cpu_build_s`, `cpu_absorb_s`, `l0_steps`). `to_arrays` / `from_arrays`: contact states, base
inputs, polar views, tool mounts, the carried loads, tips and counters.

## 9a. Probe (`hot_probe.py`)

A path whose spindle holds a probe (the planner's tool number names a `kind: probe` entry of the project's tool
list, which names one of the machine's `probes`; the resolved project the loop runs keeps these entries in
`tools.probes`, each with its probe resolved against the machine, a raw project in `tools.list`) gets a
probe-only contact after the cutting contacts:
`CI_ACTIVE` 0 (no cutting, removal or dynamics loop touches it), `CI_PROBE` 1, `CI_PATH`, `CI_WORK` (the path's
workpiece) and the stylus as removal's `Profile`: the ruby ball (`stylus_ball_mm` of the entry's resolved probe,
else of `<probe>` or `<probe>_extra`), its upper half down to the stem, the stem (`probe_stem_diameter_ratio` x the ball,
db/models/loop.yaml: the machine files give no stem) to `stylus_length_mm` above the ball's centre. The probe's
tool control point is the bottom of the ball (its length offset, the calibrated electrical trigger point, runs to
it); motion's tip of a path is its tool slot's length below the gauge line and a probe has no slot (it is not in
the tool table: the tip is the gauge line), so the profile is placed `length offset - motion's tip length` below
the tip (negative z). The contact is a copy of the padding contact with its own `ipar`, `work` scratch and
`probe_out`; it is built at every step start while the probe is in the spindle (the kernel sweeps only on G31
ticks). The sweep is removal's `sweep_contact` (the stylus' leading half sampled, cast along the tick's move
against the interpolated field boundary, refined): the trigger is the first contact with the material, the
trigger force and the stylus' deflection before it are motion's pre-travel (`_probe_feedback`).

## 10. Driver (`drive.py`)

`OperationDriver(project, op_index=0, db=None, use_dynamics=True, recorder=None, l0_override=None, seed=None,
check_energy=True, dynamics_l2=True, cutting_model_factory=None)`: project resolve, tooling, controller,
motion (`poses_F_W` from the converted fixture pose, `poses_W_S`), cutting, removal, dynamics, HotLoop, the
core `Ledger` (time; motion, removal, dynamics). `step()`: fresh bus, `hot.run_step`, the L2 macro steps,
publishers, ledger, energy, record, exact volume, observers, clock, bus swap. Per step it checks:

* the core ledger's time balance (block times sum to each path's span in integer ticks);
* cutting's energy identity per step from the step's own records (shaft + feed = heat by zone + ploughing +
  recut + packing + the work on the deflection rates), cumulative residual <= 1e-9 of the term magnitudes;
  dynamics' exact modal work is reported next to it (`modal_cutting_work_J`);
* the exact removed volume per workpiece: the booked removal equals the field's own change (the polar field
  for a turned workpiece, the tri-dexel's ledger volume otherwise: docs/models/removal.md 1.1) to 1e-9. A turned workpiece milled by live tools
  in the same operation is not covered (removal mills the tri-dexel, the polar field does not see it).

Frames: `fixture_pose` is written in machine coordinates; for mills `F_W = T(F(0)^-1 N(0)) fixture_pose`
(`fixture_pose_F`), lathes use it as it is. `checkpoint()` / `restore()`: the core codec of SimState plus the
loop's arrays, the ledger and the driver's counters (bit-identical continuation, test g).
`project_variant(src, dst, program_text=, program_name=, edits=)` writes a project copy with its own program.

## 11. Changes in domain packages (additive, documented)

* `dynamics/abi.py` `DynSystem.act` [`ACT_*`]: the station range of the step. The loop sets it before
  `modal_begin` to the engaged stations (engaged intervals) with `CI_ST_MARGIN` neighbours on each side (2 + the
  most stations of one slab: a slab point interpolates between the stations around the slab's mid height);
  `modal_begin` computes the deflections there only and `modal_correct` projects the loads of those stations
  (every other station has no load; `DynState.st_force` is updated over the union with the previous step's range).
  Unset (`ACT_LO` < 0, `all_stations()`): every station (dynamics' own drivers and tests; turning contacts).
  Deflections of stations outside the range keep their last values: only a forced pass's slab that is not engaged
  (its stations have no engaged interval) can read them.
* `cutting/abi.py` `CutTool.pk` / `pki`: the element constants packed per element in station-list order
  (`pack_elements`, built at L2 by `build_cut_tool`; not stored in checkpoints, rebuilt on restore).
* `removal/domain.py` `density_20C`: falls back to `material.physical.density_kg_m3` when the material has no
  density table (the pocket example's al6061-t6 failed at init).
* `removal/collision.py` `device_body_box` (used by `fixture_meshes` and `RemovalDomain._held_from_setup`): a
  workholding body may be the catalogue's round form `{outer_diameter, length}` (chucks; its bounding box on
  the device's +z) as well as the template's two corners; the lathe projects failed at init.
* `cutting/abi.py` `PolarView.gap_lo_m`, `gap_hi_m` (`empty_polar` builds them empty) and `cutting/kernels.py`
  `polar_inside` subtracting that gap: the polar field's radial rays hold two segments behind a leaning major
  edge (section 7); with one interval per cell no view gives every element its chip. Cutting's own tests build
  views without gaps (unchanged results).
* `cutting/state.py` `CuttingState.last_paths` (persisted as meta `last_paths`, default `[]` for older
  checkpoints) and `cutting/domain.py` reading it instead of the domain object's `_last_paths`: the paths of
  the last step must survive a checkpoint for a bit-identical continuation (test g).

## 12. Tests (`tests/integration`, run with `bin/tsafe`, `TSAFE_MEM=1700M`)

One process per file (`for f in tests/integration/test_*.py; do TSAFE_MEM=1700M bin/tsafe python -m pytest -q
$f; done`): the whole directory in one process on a cold Numba cache went over the 1.5 GB cap (compilation of
every domain's kernels plus the module fixtures; the compilation alone peaks at 1.44 GB). A cold cache costs
about 12 CPU minutes in the first file (the hot loop's own functions 365 s).

| test | what | result |
|---|---|---|
| `test_slot.py` (a) | full slot Al 6061, D10 4FL, 10000 rpm, f_z 0.04, ap between stations | mean F within 0.2 % of cutting's stand-alone kernel (of its magnitude), peak +9 %; removed = lattice slot volume 1e-5; chip mass = rho V |
| `test_drill.py` (b) | G83 D6.8, 4 pecks, Kienzle table | thrust, torque within 0.2 %; hole = lattice volume 2e-7 |
| `test_turning.py` (c) | OD pass 4140 d45, CNMG120408 / PCLNR, 1400 rpm M04, f 0.25, ap 1.5, polar field; rigid and with the tool / turret modes | mean F within 0.15 % of cutting's stand-alone turning driver over the exact earlier-revolution surface; every element's chip within 3.7 %; removal rate vs ring less scallops +0.01 %; removed = polar change exactly; with the modes the steady force is the rigid one |
| `test_free_dynamics.py` (d) | spindle unbalance and base excitation out of contact | x amplitude vs closed form 0.3 %; base excitation vs quasi-static 3 % |
| `test_chatter.py` (e) | 1-DOF regenerative slot (900 Hz, zeta 0.02, 5 N/um) | stable at 0.4 x the ZOA limit, chatter (index > 0.5, 1066 Hz) at 2 x |
| `test_determinism_checkpoint.py` (f, g) | two runs; checkpoint between macro steps | bit-identical (dynamics on) |
| `test_signatures.py` (h) | every dispatcher reached from run_macro | one signature each |
| `test_performance.py` (i) | CPU s per simulated s: slot (rigid, with the machine's modes), pocket-islands helical entry (0.3 s of cut, and the whole run with the default L2 settings) | section 13; regression bounds about 1.5-2 x the measured costs; the 0.45 s/s budget and the 1 s/s whole-run target are strict expected failures (removal's tooth pass and connectivity check) |

## 13. Measurements

One core of the i5-4690 (shared box, `bin/tsafe`, nice 10, a load of 3-6 from other jobs: +-15 %), process CPU
time; docs/performance.md has the per-kernel, per-level, whole-run and memory tables and how the costs were reached.

| case | numbers |
|---|---|
| slot, rigid (D10 4FL, 119 stations, 15 engaged, 27 engaged elements, 10000 rpm, stride 8 us) | run_macro 49-59 us per L0 step = 6.2-7.4 s per simulated s in cut (2026-11-04: 53-80 us); instrumented: removal's tooth passes 33.5 us (6.1 ms per pass), regate 2.4-5.2 us, cut_step 8.3 us, pose (motion FK x 2) 4.4 us, refill planes 4.4 us (4.4 per step), slab events 0.2 us |
| slot, the machine's 8 modes (4 tool / machine, 4 workpiece; deflections at 61 stations, loads at 15) | dynamics' kernels 3.5 us per L0 step (modal_begin 2.2 us, modal_correct 1.3 us; 7.6 us with the station ranges alone, ~55 us on 2026-11-04: RAW kernels, contiguous station runs, docs/performance.md "Dynamics"); the whole slot with the modes 2.8 s/s, of which dynamics' L2 1.0 s/s (fast-L2 settings) |
| pocket-islands helical entry, rigid (D10 3FL, 12000 rpm, stride 5 us) | run_macro 39-45 us per L0 step = 7.8-9.0 s/s in cut (61-71 us); removal's tooth passes 37 us (12.4 ms per pass), cut_step 2.8 us, pose 4.5 us, refill planes 0.9 us |
| whole pocket helical-entry run, default L2 | 3.9-4.7 CPU s per simulated s (2026-11-04: 6.6); L2 1.9-2.2 s/s, of which removal's connectivity check 1.7-1.9 |
| turning pass, rigid (CNMG, stride 50 us) | ~4 s/s in cut (polar cut and view per step; not re-measured) |
| L2 between macro steps | dynamics: the limit at the operating point only, cached by engagement class and speed (stacked frequency scan, batched bisection, cached cut-model means); lobe diagrams and SD checks once per class at the end of the operation. Cutting's coefficient pipeline: end-mill run-up 7.0 s -> 0.83 s, drill run-up 211 s (27 s re-measured with the old rule) -> 7.0 s (provisional rows before the speed settles, quantised slow state, lazy process maps, cheaper continuations) |
| cold compile | ~700 CPU s for every kernel the loop calls with an empty cache (the hot loop's own functions 365 s), peak RSS 1.44 GB (docs/performance.md "Compile time") |
| slot forces | mean [-103.60, 163.31, -38.28] N vs stand-alone [-103.60, 163.20, -38.41] (rows at the quantised slow state since 2026-11-09; before [-103.96, 163.85, -38.33] vs [-104.07, 163.90, -38.49]); peak 237.8 vs 218.7 N |
| slot volume | removed 9.718233e-7 m^3, lattice 9.718317e-7, continuous 9.722590e-7; floor 7 um below the programmed depth (machine geometric errors) |
| drill (Kienzle) | thrust 493.67 vs 494.04 N, torque 1.4864 vs 1.4884 N m; hole 3.4003130e-7 vs lattice 3.4003207e-7 m^3 |
| turning | mean F T0 [-138.1, 1067.0, 452.7] vs stand-alone [-138.7, 1065.8, 452.3] N; every element's chip within 3.7 %; removal 1.192717e-6 vs 1.192580e-6 m^3/s |
| turning with 4 modes | steady F [-138.0, 1067.1, 452.6] N = rigid to 1e-4 (the axial static deflection shortens the cut: volume -0.5 % on a 3 mm pass); energy identity 7e-16; without the deflected polar cut -6.7 %, without the stencil refresh -2.8 % |
| free dynamics | unbalance x 1.77350e-7 vs closed form 1.76891e-7 m; base excitation -59.39 vs quasi-static -57.46 um at 4.98 m/s^2 |
| chatter (1-DOF, 900 Hz, 5 N/um) | ZOA limit 1.4-2.0 mm; ap 0.62 mm: vibration 2.9 um pp, index 0.004-0.007; ap 2.86 mm: 114 um pp, index 0.99, 1066 Hz |

## 14. Open problems

* **Cost.** `run_macro` costs 49-59 us per L0 step for the D10 4FL slot (rigid) and 39-45 us for the pocket's
  helical entry: 6-9 CPU s per simulated second in cut, 14-20 x the 0.45 s/s budget; the whole pocket run 3.9-4.7
  s/s against the 1 s/s target. Removal's tooth pass (`tooth_pass`: engaged intervals and the subtraction of the
  tooth's envelope, once per tooth per revolution) is 33-37 us of every L0 step (6.1 ms per pass at lattice 0.1 mm,
  12.4 ms at 0.075 mm): alone 4-7.5 s/s; its connectivity check (`find_bodies`) is 1.9 s/s of the whole run's L2.
  The rest: cut_step 2.8-8.3 us (about 300 ns per engaged element), the two FK frames of a step (motion's
  `fk_poses`) ~4 us, refill planes 1-4 us. docs/performance.md "Open problems".
* **L2 costs between macro steps.** The first coefficient build of a tool (lazy process maps; continuations of
  map rows without an admissible Oxley solution) is 0.3-0.8 s for the end mills and ~7 s for the drill's 18 rows;
  the end-of-operation lobe diagrams and SD checks cost seconds per engagement class at the default grids; with
  dynamics, the time-domain chatter detection costs ~26 ms per analysed window and the limit at a new operating
  point ~0.13 s (1.0 s/s of L2 in the slot with the machine's modes); removal's connectivity check 1.7-1.9 s/s in
  the whole pocket run.
* **Turning tool frame.** The mount belongs in motion's chain (a tool's mounting rotation on its station);
  the bus (`pose_S_T`) and dynamics' L2 operating point still use the chain's frame for turning tools.
  Left-hand turning tools also reverse z_T: dynamics' station positions along z_T would then be mirrored.
* **Lathe spindle direction.** Under motion's ISO 841 sign (M03 turns the work counter-clockwise seen from
  the tailstock) tooling's right-hand turning tool (hand sign -1, intended cutting velocity h y_T) on a
  turret at +X cuts along its rake face only in M04; the lathe example's M03 with the CNMG roughing tool
  turns the work against it (cutting's unit test uses the work turning clockwise seen from the tailstock).
  One of tooling's turning hand sign, the machine's C direction or the example's program is off; the
  turning test uses M04.
* **Probe.** The G31 sweep runs with the probe-only contacts of section 9a (2026-11-09; before, no stylus
  profile was built and every probing program stopped at its probe alarm). Only spindle probes on a path's
  tool spindle are built; a tool setter keeps motion's own trigger rule.
* **Turning force alternation** (fixed 2026-11-09, docs/models/loop.md 15): with cutting's process damping on, a
  turning contact's L0 force alternated from step to step (the linearised flank indentation switching with the
  predicted normal velocity) and diverged once the cut had grown. Process damping now comes from each element's
  wave memory of the surface it generated (docs/models/cutting.md "Process damping"): nanometres of vibration
  indent the flank by nanometres, and the force is a function of the displacement, not of the predicted velocity.
* **Peak force** of the slot is 9 % above the stand-alone kernel with a beat every 120 ms (not explained:
  the variable pitch against the 0.5 deg stride and the refill planes are suspects).
* **Field resolution** (section 8): within a cell of a stock edge the field, not the feed, sets the chip.
* **Helical ramp, analytic chip volume.** In the pocket example's helical entry cutting's analytic chip volume
  is 54 % of the field's (removal's mass-balance warning, 85 % over the pass); the slot agrees to 2 %. Likely
  cause (not verified): the end edges' axial chip of the ramp (~2 um per tooth) is below cutting's minimum
  chip thickness, booked as ploughing, while removal removes the layer geometrically.
* **Zero-depth rapid** (fixed 2026-11-09: the rapid's event carries its depth, a crash is a depth beyond the
  collision tolerance; docs/models/removal.md "Rapids"). Removal flagged `rapid_into_material` (error) with 0.000 mm depth and volume when the
  pocket variant retracts with G00 from the helix floor; the rule needs a tolerance.

## 15. Parameters and constants

The hot loop has no physical coefficients of its own: the forces, the removal, the modes and the drives are
the domains' (their docs and `db/models` files). Its settings and constants:

| name | value | source |
|---|---|---|
| `MAX_CONTACTS`, `MAX_PATHS`, `MAX_WORK` (`hot_abi.py`) | 2, 2, 2 | one contact per controller path, main and sub spindle (the scope of cnctwin-architecture-2026, section 1) |
| L0 stride rule (`physics.max_deg_per_step`, `physics.steps_per_tooth`, modal rule) | 0.5 deg, 96 steps per tooth period, `2 pi f_max dt <= 0.3` for modes carrying >= 1 % of the tip compliance | cnctwin-architecture-2026 (5.1); docs/convergence.md 1 |
| macro step (`physics.macro_ipo`) | 10 L1 ticks (10 ms at a 1 ms IPO period) | cnctwin-architecture-2026 (5.1); docs/convergence.md 5 |
| dexel lattice (`physics.lattice_mm`) | 0.25 mm (template); `twin check` asks for <= ae / 4 | docs/convergence.md 2 |
| surface-memory bins (`physics.memory_bins`) | 720 per revolution | docs/convergence.md 4 |
| stale memory bin | older than one revolution plus one tooth period (cutting's `SI_STALE_TICKS` entry) | cnctwin-architecture-2026 (5.4) |
| station margin of the step's range (`CI_ST_MARGIN` entry) | 2 + the most stations of one slab | a slab point interpolates between the stations around the slab's mid height (section 11) |
| tooth-window margin (`WIN_EPS`) | 1e-8 rad | round-off margin; the exact per-slab test decides inside the window |
| refill plane validity | the boundary faces the element when `m . n <= -0.3`; covered: the engaged intervals plus half the step's rotation and one bin, four bins around each element | est (section 5) |
| event buffer rows per L0 step (`HotLoop._rows`) | shocks 4 nt + 4, surface 3 n_el + 8, deposits 2 nt + 4, recut n_el + 4 nt + 4, passes 2 nt + 4, refills 2 n_el + 4, removal 8 nt + 8; buffers >= 4 x the largest | upper bounds of what one step writes (section 6); the results do not depend on them |
| `probe_stem_diameter_ratio`, `probe_ball_diameter_m`, `probe_stylus_length_m` | 0.75, 6 mm, 50 mm | est, db/models/loop.yaml (Renishaw M4 / M2 styli and the OMP40-2 / Haas WIPS kits in its notes; renishaw-omp40-2-2025) |
| `ring_f_max_Hz` (fast ring sizing) | 20 kHz | est, db/models/loop.yaml |

## 16. Assumptions, validity and error bounds

| assumption | consequence | validity | error bound (how established) |
|---|---|---|---|
| Kernel inputs are frozen for a macro step (coefficients, temperatures, structure grids from the L2 domains of the previous step, architecture 5.3) | the slow couplings close one macro step late | `physics.macro_ipo` 5-20 ms | tool state and the 0.5 s means of the coupling temperatures within 0.2-1.5 % from 5 to 20 ms; values at one instant scatter 1-2 % (docs/convergence.md 5) |
| A tick's mean cutting load per axis and the spindle torque act on the drives in the next L1 tick | the servo sees the cutting disturbance one IPO period late | drive velocity bandwidths far below the IPO frequency (default 50 Hz, est, against 1 kHz at a 1 ms IPO period) | not measured separately; inside the cycle-time and following-error bounds of docs/models/motion.md 9 |
| Spindle angles and axis values inside a tick are cubic Hermite interpolants of the tick's end values and speeds | L0 poses follow a cubic between L1 ticks; the servo's variation within a tick is not seen | any L1 period | exact for motion of constant jerk within a tick (a cubic Hermite interpolant reproduces cubics) |
| Generation instant of a slab: the step's crossing fraction, the slab point interpolated linearly between the step's end frames; a tooth that missed an angle while the engagement changed is forced after the stale age, its undone slabs at the step end (`CS_FORCED`) | the field is cut where the surface was generated, not at a step end | L0 strides by the rule above | removed volume = the field's change to 1e-9 (every macro step); slot: removed = lattice slot volume to 1e-5, drill hole to 2e-7 (tests a, b) |
| Stale memory bins are refilled from removal's interpolated boundary plane of the best-aligned dexel family, else at the element (zero chip until a tooth writes the bin); a plane lives for one L0 step | chips at a bin the memory cannot give come from the field | boundaries that face the element (`m . n <= -0.3`) | exact for planes, `a^2 kappa / 4` on curved boundaries (a the lattice, kappa the curvature) |
| The field decides a column's top to a lattice cell | within one cell of a stock edge the chip is the field's, not the feed's | lattice <= ae / 4 (`twin check`) | 6 % force deficit at lattice 0.25 mm and ap 3.0 mm in the slot (section 8); steady peak force +31 %, mean 1.5-1.9 % at 0.25 mm in a side finish (docs/convergence.md 2) |
| Station deflections are computed over the engaged stations plus the margin; others keep their last values; static (residual) deflections lag one L0 step | only a forced pass's slab that is not engaged can read an old deflection | statically stable cuts (`c_res K_c < 1`, docs/models/dynamics.md 1) | slot with 8 modes against the full range: not separately bounded; free dynamics 0.3 % against the closed form, base excitation 3 % against quasi-static (test d) |
| Turning: the insert is a rigid body at the deflected tip (translation and rotation of the station nearest the tip); the polar view carries the hull and one air gap per cell | a third material segment on one radial ray is not represented | steady external and internal passes (none has a third segment) | mean force within 0.15 % of the stand-alone turning driver over the exact earlier-revolution surface, every element's chip within 3.7 % (test c); without the deflected polar cut -6.7 % |
| Peak forces are the instantaneous wrench at a step's start or end where larger than the step mean (2026-11-10) | peaks resolve to the step ends, not within the step | default stride 0.5 deg | slot: 259 N at 0.5 deg against 258 N at 0.0625 deg (docs/models/loop.md 15); the integrated slot's peak is 9 % above the stand-alone kernel's (unexplained, section 14) |
| Probe: the stylus is a ball and a stem of 0.75 x the ball diameter; only spindle probes on a path's tool spindle get a contact | a G31 move that meets material with the stem first triggers at the estimated stem | spindle work probes (tool setters keep motion's trigger rule) | trigger within the 25 um braking distance (tests/integration/test_loop_probe.py) |
| At most 2 contacts, 2 paths and 2 workpieces per step; a turned workpiece milled by live tools in the same operation is cut in the tri-dexel only | the exact-volume check does not cover that case | mills, lathes with one or two paths | - |

Overall, against the domains' stand-alone kernels: slot mean force within 0.2 % (of its magnitude), drill thrust
and torque within 0.2 %, cutting's energy identity cumulative residual <= 1e-9 of the term magnitudes, runs and
checkpoint continuations bit-identical (tests a-g, section 12).

## References

* cnctwin-architecture-2026: the contract the loop implements (5.1 clock and strides, 5.3 macro step, 5.4 hot loop, 5.5 layouts, 12 ledgers).
* altintas-budak-1995-zoa: the zero-order stability limit the chatter test is set against (test e).
* kienzle-1952: the drill test's force table (test b).
* renishaw-omp40-2-2025: spindle probe the probe defaults stand for (section 9a).
* The domains' own references for every model the loop composes: docs/models/motion.md, cutting.md, removal.md,
  dynamics.md (15, 12, 19, 11).
