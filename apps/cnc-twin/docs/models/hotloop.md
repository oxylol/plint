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
  tool; inactive contacts are padding of the same types (`empty_contact`).
* `HotContact`: `ipar` [`CI_*`] (path, slot, workpiece, tool kind, rotating, tool / work spindle, teeth,
  elements, stations, L0 stride, dynamics on, recorder columns, polar, stale-before, probe), `fpar` [`CF_*`]
  (hand, cutting radius, refill reach, lattice, work-spindle sign, gating parameters, feed fraction, polar
  signs, tool mount), `cut: CutArgs` (cutting's tool, table, memory, intervals, refill, step scalars, pose,
  deflection, occupancy, packing, axes, state, accumulators, outputs, event buffers, polar view), `sys`,
  `base`, `grid` (dynamics' system, axis base, work grid), `geo: ContactGeo` (stations, slab -> station map,
  slab lags, the path's machine axes and their direction in S), `cst: ContactState` (the loop's own state:
  generation angle per tooth and slab, recorded slab points, slab done flags, last pass tick, polar view
  window, counters `CS_*`, angles `CSF_*`), removal's `SubtractSpec`, gate parameters, `work` [112] (frames
  and vectors of the step, `hot_kernels.WK_*`), base-excitation inputs, FK scratch, polar step, refill plane
  list, probe profile.
* `HotGlobal`: `ipar` [`GI_*`] (tick0, ipo ticks, resume point, buffer row bounds), `fpar`, the carried L1
  loads per axis and spindle, the tick-start axis values and spindle angles, the tool tips (probe sweep), the
  structure displacement grid d(X), `result` [`RES_*`], counters `CNT_*`, per-tick in-cut flags.

## 2. One macro step (`run_macro`)

Per L1 tick k (from motion's `k_from`):

1. resume handling (a split returned inside this tick: jump to the contact and L0 step of `GI_RESUME_*`);
   motion's `replan_before`; save q0, spindle angles and speeds; `motion_tick` (setpoint, servo with the
   previous tick's axis loads, spindle loops with the previous torque, FK, out.tip_S);
2. G31 probe sweep (`sweep_contact`) when a probe profile is set (`CI_PROBE`; see open problems);
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
  predicted deflections); the deflections and the workpiece displacement (+ d(X)) copied into cutting's
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
that fraction of the step (`slab_point`: tip, axis and u_rel interpolated between the step's end frames).
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
(`SPLIT_POOL`) or a polar inline-segment overflow (`SPLIT_POLAR`) returns too: the wrapper grows the field
(`prepare_replay`), and the refused pass (`CS_OVERFLOW_TOOTH`) or polar cut is repeated at the resume (both
idempotent). No row is lost and nothing is rolled back, so the results do not depend on buffer capacities
(test f runs with small buffers).

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
dynamics, cutting, removal `kernel_inputs`; padded removal inputs; contacts; buffer rows), then `run_macro`
chunks: at a split `_handle_split`, at a sync motion's `absorb`, the controller actions, the next commands,
until `step_complete`. `_absorb`: the drained rows merged into new event buffers for `cutting.absorb`,
removal's event buffer swapped in for `removal.absorb`, `dynamics.absorb` with the step's accumulators (or an
empty frame for rigid runs). `StepReport` carries chunks, splits, syncs, strides and CPU times (`cpu_s`,
`cpu_macro_s`, `cpu_build_s`, `cpu_absorb_s`, `l0_steps`). `to_arrays` / `from_arrays`: contact states, base
inputs, polar views, tool mounts, the carried loads, tips and counters.

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
  for a turned workpiece, the tri-dexel z family otherwise) to 1e-9. A turned workpiece milled by live tools
  in the same operation is not covered (removal mills the tri-dexel, the polar field does not see it).

Frames: `fixture_pose` is written in machine coordinates; for mills `F_W = T(F(0)^-1 N(0)) fixture_pose`
(`fixture_pose_F`), lathes use it as it is. `checkpoint()` / `restore()`: the core codec of SimState plus the
loop's arrays, the ledger and the driver's counters (bit-identical continuation, test g).
`project_variant(src, dst, program_text=, program_name=, edits=)` writes a project copy with its own program.

## 11. Changes in domain packages (additive, documented)

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

## 12. Tests (`tests/integration`, run with `bin/tsafe`, `TSAFE_MEM=1500M`)

One process per file (`for f in tests/integration/test_*.py; do TSAFE_MEM=1500M bin/tsafe python -m pytest -q
$f; done`): the whole directory in one process on a cold Numba cache went over the 1.5 GB cap (compilation of
every domain's kernels plus the module fixtures). A cold cache costs minutes in the first file (the hot loop
alone ~283 s).

| test | what | result |
|---|---|---|
| `test_slot.py` (a) | full slot Al 6061, D10 4FL, 10000 rpm, f_z 0.04, ap between stations | mean F within 0.2 % of cutting's stand-alone kernel, peak +8 %; removed = lattice slot volume 1e-5; chip mass = rho V |
| `test_drill.py` (b) | G83 D6.8, 4 pecks, Kienzle table | thrust, torque within 0.2 %; hole = lattice volume 2e-7 |
| `test_turning.py` (c) | OD pass 4140 d45, CNMG120408 / PCLNR, 1400 rpm M04, f 0.25, ap 1.5, polar field; rigid and with the tool / turret modes | mean F within 0.15 % of cutting's stand-alone turning driver over the exact earlier-revolution surface; every element's chip within 3.7 %; removal rate vs ring less scallops +0.01 %; removed = polar change exactly; with the modes the steady force is the rigid one |
| `test_free_dynamics.py` (d) | spindle unbalance and base excitation out of contact | x amplitude vs closed form 0.3 %; base excitation vs quasi-static 3 % |
| `test_chatter.py` (e) | 1-DOF regenerative slot (900 Hz, zeta 0.02, 5 N/um) | stable at 0.4 x the ZOA limit, chatter (index > 0.5, 1063 Hz) at 2 x |
| `test_determinism_checkpoint.py` (f, g) | two runs; checkpoint between macro steps | bit-identical (dynamics on) |
| `test_signatures.py` (h) | every dispatcher reached from run_macro | one signature each |
| `test_performance.py` (i) | CPU s per simulated s, slot and pocket-islands helical entry | section 13; the 0.45 s/s budget is a strict expected failure |

## 13. Measurements

One core of the i5-4690 (shared box, `bin/tsafe`, nice 10), process CPU time.

| case | numbers |
|---|---|
| slot, rigid (D10 4FL, 119 stations, 10000 rpm, stride 8 us) | run_macro 53 us per L0 step = 6.6 s per simulated s in cut; cut_step ~50 us of it; refill planes 4.4 per step |
| slot, 8 dynamics modes | 108 us per L0 step = 13.5 s/s |
| pocket-islands helical entry, rigid (D10 3FL, 12000 rpm, stride 5 us) | 61 us per L0 step = 19 s/s in cut; whole run 6.5 s simulated in 42.7 s CPU (hot 18.9, L2 21.0); removed 173.5 mm^3 = field change exactly |
| turning pass, rigid (CNMG, stride 50 us) | ~4 s/s in cut (polar cut and view per step) |
| L2 between macro steps | dynamics' lobes ~200 s per new engagement at the default grids (1.3 s coarse); cutting's coefficient pipeline in a run-up 16 s (end mill), 211 s (drill) |
| cold compile of the hot loop | ~283 s (all domains' kernels: longer) |
| slot forces | mean [-103.96, 163.85, -38.33] N vs stand-alone [-104.07, 163.90, -38.49]; peak 237.2 vs 219.6 N |
| slot volume | removed 9.718006e-7 m^3, lattice 9.718077e-7, continuous 9.72235e-7; floor 7 um below the programmed depth (machine geometric errors) |
| drill (Kienzle) | thrust 493.67 vs 494.04 N, torque 1.4864 vs 1.4884 N m; hole 3.4003130e-7 vs lattice 3.4003207e-7 m^3 |
| turning | mean F T0 [-138.4, 1068.8, 453.8] vs stand-alone [-139.0, 1067.5, 453.4] N; removal 1.192717e-6 vs 1.192580e-6 m^3/s |
| turning with 4 modes | steady F [-138.4, 1068.9, 453.8] N = rigid to 1e-4 (the axial static deflection shortens the cut: volume -0.6 % on a 3 mm pass); energy identity 7e-16; without the deflected polar cut -6.7 %, without the stencil refresh -2.8 % |
| free dynamics | unbalance x 1.77350e-7 vs closed form 1.76891e-7 m; base excitation -59.39 vs quasi-static -57.46 um at 4.98 m/s^2 |
| chatter (1-DOF, 900 Hz, 5 N/um) | ZOA limit 1.4-1.7 mm; ap 0.61 mm: vibration 2.9 um pp, index 0.005; ap 2.86 mm: 113 um pp, index 0.97, 1063 Hz |

## 14. Open problems

* **Cost.** `run_macro` costs about 53 us per L0 step for the D10 4FL slot (119 stations, rigid) and 108 us
  with 8 dynamics modes, i.e. 6.6 and 13.5 CPU s per simulated second at the 8 us stride of 0.5 deg per step
  at 10000 rpm: 15-30 x the 0.45 s/s budget. 94 % of it is cutting's `cut_step` (about 50 us per step) and
  dynamics' `modal_begin` / `modal_correct`; the loop's own work (FK, refills, tooth events) is about 3 us
  per step. The budget needs cheaper kernels or a coarser stride (steps_per_tooth, max_deg_per_step).
* **L2 costs between macro steps.** dynamics' lobe diagram per new engagement (minutes at the default grids;
  the tests use coarse grids) and cutting's coefficient pipeline during a spindle run-up (16 s for the end
  mill, 211 s for the drill; the tests run it every 50 macro steps).
* **Turning tool frame.** The mount belongs in motion's chain (a tool's mounting rotation on its station);
  the bus (`pose_S_T`) and dynamics' L2 operating point still use the chain's frame for turning tools.
  Left-hand turning tools also reverse z_T: dynamics' station positions along z_T would then be mirrored.
* **Lathe spindle direction.** Under motion's ISO 841 sign (M03 turns the work counter-clockwise seen from
  the tailstock) tooling's right-hand turning tool (hand sign -1, intended cutting velocity h y_T) on a
  turret at +X cuts along its rake face only in M04; the lathe example's M03 with the CNMG roughing tool
  turns the work against it (cutting's unit test uses the work turning clockwise seen from the tailstock).
  One of tooling's turning hand sign, the machine's C direction or the example's program is off; the
  turning test uses M04.
* **Probe.** The G31 sweep runs only with a stylus profile (`CI_PROBE`), which no domain builds yet: the
  pocket example's N10 probing cannot run; the driver tests use a variant without it.
* **Peak force** of the slot is 8 % above the stand-alone kernel with a beat every 120 ms (not explained:
  the variable pitch against the 0.5 deg stride and the refill planes are suspects).
* **Field resolution** (section 8): within a cell of a stock edge the field, not the feed, sets the chip.
* **Helical ramp, analytic chip volume.** In the pocket example's helical entry cutting's analytic chip volume
  is 54 % of the field's (removal's mass-balance warning, 85 % over the pass); the slot agrees to 2 %. Likely
  cause (not verified): the end edges' axial chip of the ramp (~2 um per tooth) is below cutting's minimum
  chip thickness, booked as ploughing, while removal removes the layer geometrically.
* **Zero-depth rapid.** Removal flags `rapid_into_material` (error) with 0.000 mm depth and volume when the
  pocket variant retracts with G00 from the helix floor; the rule needs a tolerance.
