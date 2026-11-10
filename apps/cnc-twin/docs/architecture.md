# CNC Twin architecture (contract v2)

This document is the contract every part of the simulator is built against: the physics loop,
the shared state, units, frames, the kernel ABI and the interfaces between domains. A module
may add fields to the interfaces (additively, documented in its model doc); it may not change
the meaning of an existing one. Version 2 incorporates the architecture review of 2026-11-03.

## 1. Stack, and why

Python 3.12 with NumPy and SciPy for the numerics, Numba for every per-step kernel (and for
the whole compiled L1/L0 loop), FastAPI/uvicorn for the HTTP API and its OpenAPI document,
declared schemas for files, PyYAML, PyArrow (Parquet), Matplotlib (report plots), WeasyPrint
(PDF), Pillow and FFmpeg (images, video), OpenCASCADE through OCP (STEP AP242 with PMI),
asyncua (OPC UA), three.js vendored for the UI, optional CUDA through numba-cuda. All open
source, all offline.

Why: the work is dominated by a broad set of physical models that need dense linear algebra,
sparse solvers, optimisation and root finding (SciPy covers all of it), and a few hot loops that
must run at compiled speed. Numba compiles those loops from the same source, deterministically
(IEEE-754 double, no fast-math, fixed reduction order), so one language serves models, kernels,
API and tests. The cost is JIT warm-up on first run (cached on disk; pre-warmed in the image).

**Scope decisions** (each with its consequence, also in docs/approximations.md):
* Multi-path controls and simultaneous cutting are in scope: several controller paths
  (channels) with wait synchronisation, several active cutting contacts in the same step, several
  workpieces (main and sub spindle). Single-path mills are the common case of the same structure.
* Out-of-contact tool dynamics include forced (unbalance) and base (axis acceleration) excitation.

## 2. Repository layout and ownership

```
apps/cnc-twin/
  pyproject.toml  Dockerfile  README.md  bin/twin  bin/tsafe
  cnctwin/
    units.py   core/ (clock, rng, state, bus, abi, loop, hotloop, findings, trace, channels,
               recorder, ledger, checkpoint, rundir, keyframes, errors, domain)
    project/   db/   geom/
    controller/ motion/ tooling/ removal/ cutting/ mechanics/ chips/ thermal/ coolant/ wear/
    holding/ structure/ dynamics/ surface/ inspect/ production/ sanity/ sound/ render/
    report/ uncertainty/ mdata/ gpu/ api/   cli.py
  web/ db/ examples/ tests/ validation/ docs/
```

Libraries (importable by every domain): `units`, `core`, `geom`, `db`, `project`, `tooling`
(edge models, envelopes, tool table), `mechanics` (Oxley / Johnson-Cook chip formation, heat
generation and partition fractions, chip form and colour: pure functions and tables). Domains
(`controller motion removal cutting chips thermal coolant wear holding structure dynamics
surface`) never import each other's internals; they exchange data through the bus and the
kernel ABI. Each domain package has `state.py` (its SimState field), `channels.py`, `schema.py`
(its `physics.<domain>` project settings), `domain.py` (its Domain object), `kernels.py` (Numba),
`abi.py` (its own kernel layouts), and a model doc `docs/models/<domain>.md`.

## 3. Units, names, constants

* **SI inside**: m, kg, s, N, Pa, J, W, K, rad. Temperatures are kelvin inside, degC outside.
* **Names carry units**: `force_N`, `torque_Nm`, `pos_m`, `vel_m_s`, `acc_m_s2`, `omega_rad_s`,
  `temp_K`, `power_W`, `energy_J`, `mass_kg`, `time_s`, `freq_Hz`, `stiffness_N_per_m`, `kc_Pa`,
  `edge_coeff_N_per_m`, `h_W_m2K`, `k_W_mK`, `cp_J_kgK`, `rho_kg_m3`, `alpha_1_K`; arrays the same.
* **Interface keys declare their unit as the last token** (`travel_mm`, `power_kW`, `max_rpm`,
  `growth_at_max_speed_um`). Rules: the unit is always last; minimum/maximum are prefixes
  (`min_abs_pressure_kPa`, never `pressure_kPa_abs_min`); temperature differences use `_dK`
  (`_degC` is only for absolute temperatures); a per-length unit is `_per_mm`; poses are
  `{translate_mm: [x, y, z], rotate_deg: [a, b, c], order: xyz}`; tables carry their x unit
  (`{table: [[x, v], ...], x: degC}`, degC being the default only for temperature-dependent
  properties); two-column curves are declared by schema path (`machine.spindles[].gear_ranges[].power_kW`
  is `[[rpm, kW], ...]`). The schemas in `cnctwin/db` declare every field's unit; a key whose
  suffix is not the declared unit is an error. Older key forms are migrated by a documented
  rename table (the "migrate forward" rule), never guessed.
* **Kienzle and Taylor carry their reference values**: `kc = kc11 (h/h_ref)^-mc` with
  `h_ref_mm: 1`, `b_ref_mm: 1` (and kf11/mf, kp11/mp for feed and passive forces when published);
  `vc (T/T_ref)^n f^a ap^b = C` with `T_ref_min: 1`. Usui-type constants carry their units in the
  key (`A_per_MPa`, the rate unit stated).
* **No magic numbers outside the databases.** Physical constants live in `cnctwin.units`; model
  coefficients in `db/models/<model>.yaml` with citations (`cnctwin.db.models.params(name)`);
  numerical settings in the project's `physics:` section (declared in `<domain>/schema.py`).

## 4. Frames, signs and configurations

| frame | fixed to | notes |
|---|---|---|
| M | machine base (inertial) | chains start here; chips, jets, UI positions are in M |
| F | fixture mounting face | end of the workpiece chain (table, pallet, spindle face) |
| W | workpiece | `F * fixture_pose * part_pose`; part_pose changes if the part slips |
| W0 | non-rotating workpiece | for lathes: W without the spindle C rotation |
| S | stock lattice | `W * stock_pose`; dexel field, thermal grid, displacement grids live here |
| N | spindle nose / turret station | end of the tool chain |
| T | tool (rotating with the spindle) | origin at the tool control point (tip), +z_T toward the spindle |
| T0 | tool, non-rotating | T without the spindle angle; forces on the tool, surface-memory bins |

**Signs.** Every angle and angular velocity is right-handed about its named axis; for the
spindle that axis is +z_T, so **M03 gives omega < 0** (on the bus and in `ir.Command.spindle`).
Kernels use `phi_cut = sign(omega) * phi` internally. Arcs are right-handed (CW negative).

### 4.1 Configurations (where displacements enter, once)

* The dexel field, the polar field, the thermal grid, the surface memory and the surface events
  are stored in the **material configuration X** of S: 20 degC, unclamped, before
  machining-induced stress redistribution.
* The current position of a material point is `x = X + d(X, t)`, with
  `d = d_thermal + d_clamp + d_residual` (published by structure on a grid in S,
  `StructureFeedback.displacement`) plus the instantaneous elastic deflection of the work under
  the cutting load `u_work` (dynamics, in the hot loop).
* Each edge element's position is mapped to X by one fixed-point step `X_e = x_e - d(x_e)`;
  removal subtracts at X_e, with `u_rel = u_tool(e) - u_work(e)` sampled at the element's own
  generation instant.
* **Composition of an edge element's position** (in this order): nominal chain pose (axes incl.
  per-link geometric and thermal error transforms) + spindle growth + tool length and radial
  thermal growth + holder growth + pull-out (axial) + holder slip (as an angular phase) + wear
  (radial/axial per element) + runout and tilt + tool deflection `u_tool(e)`.
* Reported geometry: inspection uses `X + d_residual(unclamped, final)` at 20 degC;
  `part_<op>_unload` is `X + d` at the unload temperature. Mass uses rho at 20 degC times the
  material-configuration volume.

## 5. Time and the loop

### 5.1 Clock

* Integer **base clock**: `tick_s = 1e-6` s. `t_s = tick * tick_s`, never accumulated.
* **L1 tick** = `ipo_ticks` base ticks, from the machine's controller IPO period (machine file;
  the controller database only as a fallback); must be an integer number of base ticks.
* **L0 stride** `l0_ticks`: per operation (and re-chosen at each tool change or spindle speed
  range change), the largest divisor of `ipo_ticks` that satisfies all of:
  `max |omega| * dt <= physics.max_deg_per_step` (default 0.5 deg; max speed = programmed S x
  max spindle override, and the CSS clamp), `2 pi f_max dt <= 0.3` for every mode carrying at
  least 1 % of the tip compliance, and at least `physics.steps_per_tooth` (default 96) steps per
  tooth period. `twin check --write` stores the resolved strides in the project file under
  `physics.resolved` (so the timestep is recorded in the project); a run uses those when present.
* **Macro step** = `macro_ipo` L1 ticks (default 10 ms). Per-domain cadence:
  `physics.<domain>.every_n_macro`; long jobs use a fixed latency `k` macro steps (contract:
  structure mode refresh k = 50, workpiece FV k = 1, coolant occlusion k = 1): a job started at
  macro n is applied at n + k, the loop waits if it is not done, never applies it early. The structure
  refresh states its cadence and latency in run time (500 ms each, 50 steps at the default 10 ms;
  docs/models/structure.md 3.5) so that they do not move with `macro_ipo`.

### 5.2 Levels

* **L0** (stride `l0_ticks`): spindle angle, element positions, chip thickness, forces, modal
  dynamics, tooth-period events (removal, chip emission, surface events, recut), only while in
  contact (broad phase at L1).
* **L1** (`ipo_ticks`): servo (following error, backlash, reversal, cutting-load disturbance,
  spindle speed/position loop), kinematics, travel and collision checks, contact broad phase,
  per-tick recording. **Compiled**: `core.hotloop.run_macro` advances a whole macro step (servo,
  error transforms, kinematics, broad phase, L0) in one Numba call; Python is entered once per
  macro step or at a sync tick.
* **L2** (macro): the slow domains (coolant, thermal, chips, wear, holding, structure, surface
  records, mechanics/coefficient tables), findings rules, ledgers, recording, checkpoints.

### 5.3 One macro step

1. `controller` runs until its look-ahead buffer holds the machine's `lookahead_blocks` or it
   reaches a buffer-stopping block (G31, macro read of a live position/sensor, M00/M01, tool
   setter, path wait). Each Command carries `ready_tick` from the control's block-time model
   (block processing time + macro statement cost): the planner never starts or looks ahead past
   a block before its ready_tick (data starvation is modelled here).
2. `motion` plans the step's setpoints (commanded axis values per L1 tick, per path; per-rev feed,
   threading and tapping setpoints are computed inside the compiled loop from the actual spindle
   angle) and builds its kernel inputs.
3. Every domain's `kernel_inputs()` (frozen for the step) -> `core.hotloop.run_macro(...)`.
   It returns early at a **sync tick** (queue empty, in-position for SYNC/STOP/PROBE/TOOL_SETTER,
   probe trigger, machine alarm, path wait): the controller runs there (feedback with the encoder
   position at the latch time) and the loop resumes at the next tick. Macro boundaries stay fixed.
4. Every domain's `absorb()` turns accumulators and event buffers into its record in `bus_next`.
5. The slow domains' `macro_step(prev=bus_prev, nxt=bus_next)` (may run concurrently: they read
   only `bus_prev` and write only their own record), then findings rules, `ledger.check()`,
   recorder, checkpoint if due, `state.swap_bus()`.

Fast thermo-mechanical coupling (temperature -> flow stress -> forces) is inside
ProcessCoefficients, built self-consistently at L2 for the next step (5.4), so the one-step
stagger never closes that loop by itself. The coupling-lag monitor flags staggered instability:
two successive changes of a frozen quantity with opposite sign and magnitude ratio > 0.5.
`physics.lag_macro: 2` allows the slow domains of step n to run concurrently with the hot loop
of step n+1 (two-stage pipeline); docs/convergence.md compares lag 1 and 2.

### 5.4 The hot loop

Per L0 step, for every active contact (tool slot, workpiece, path):

* **Pose**: linear axes interpolated in axis space, the spindle/C angle from the spindle servo
  state, then the compiled kinematics kernel. For a rotating workpiece:
  `pose_S_T(t) = R_C(theta_C(t))^-1 * pose_W0_T(t)`. Never a matrix lerp.
* **Edge elements** (cnctwin.tooling.EdgeModel, core/abi.py): each tooth carries elements along
  its cutting edge by arc length (`physics.edge_element_mm`), each with z, r, lag, kappa, rake,
  inclination, clearance, edge radius, length, chip width db (finite at kappa = 0), kind
  (peripheral, end, corner, lip, chisel, margin, thread flank, wiper, insert, nose) and a shared
  station index. The element cutting velocity is the relative velocity
  `v_e = omega x r_e + v_feed,rel + du_rel/dt`; below `v_min` or `h_min` (ProcessCoefficients)
  the element uses the indentation/ploughing model. Forces of each element are integrated in
  closed form over the angle it sweeps in the step, clipped to its engaged intervals, so the
  impulse does not depend on step alignment (no helix-lag stair steps).
* **Chip thickness (vector form)**: `h_e = (p_e(t) - M[station, bin(phi)]) . n_e(t)` where `p_e`
  is the element position in S per 4.1 (X frame), `n_e` the element normal in the plane normal to
  the local cutting velocity pointing into the material, and **M the surface memory**
  (abi.SurfaceMemory): per station and angle bin in T0, the point (in S) the last effective edge
  passage left. After an element passes a bin with `h >= h_min` the bin is set to `p_e(t)`
  (partially, by the elastic-recovery model, between h_min and 2 h_min); below h_min it is left
  unchanged (the next tooth sees the accumulated thickness). This one dot product covers feed
  per tooth, axial feed (drilling, ramping, plunging), runout skipping, variable pitch and helix,
  spindle-speed variation, deflection and vibration at generation (regeneration) and ploughing.
  A bin is stale when older than one revolution plus one tooth period, or after a tool change or
  rapid; stale bins are refilled from the field (the surface point along n_e).
* **Outer-boundary gating** (abi.EngagedIntervals): the field never decides the freshly cut inner
  boundary (that is the memory); it clips to outer boundaries (stock surface, previous-pass walls
  and floors, holes). Per station, float intervals of the T0 angle where the element circle lies
  in material, found once per tooth period by root finding along the circle against the field,
  warm-started from the previous intervals, with boundaries located by interpolating the exact
  endpoints of the dexel family most normal to the boundary (error about lattice^2 x curvature / 8).
  `twin check` requires `lattice_mm <= min(ae) / 4` on every pass, or a refine box covering the
  finishing passes. Only elements inside engaged intervals are evaluated.
* **Forces**: per element from CoeffTable interpolation (no pow/exp at L0); sin/cos by angle
  addition from per-element tables; process damping and ploughing per element; unbalance
  `F = U omega^2 e(theta + psi_U) + U alpha e_t` (vector and couple, HoldingFeedback); chip packing
  torque and force (additive, ChipFeedback); recut impulses where the swept envelope meets the
  frozen chip-occupancy grid (recut rows applied immediately).
* **Dynamics** (abi.ModalModel): exact first-order-hold discretisation (Phi, G0, G1) with a
  predictor-corrector (state predicted with F_n, force re-evaluated at the predicted state) so the
  cutting stiffness is not delayed by dt/2 (no numerical negative damping); tool modes with
  6-component mass-normalised shapes at the stations, workpiece modes evaluated at the current
  contact points from the shape grid, machine modes with base excitation `-m_eff a_axis(t)`;
  residual (static) flexibility added quasi-statically. Out of contact: exact propagation with
  the harmonic particular solution for unbalance and first-order-hold base excitation.
* **Removal** (once per tooth period per tooth, at the element's own generation instant): the
  tooth's swept envelope is subtracted at X_e per axial slab with that slab's u_rel (bent tool,
  bent wall). Taps and threading tools are subtracted by sweeping the actual tooth profile along
  the relative helical path (never a revolved body). Rotating single-point tools on mills (boring
  heads, fly cutters) use this scheme with N = 1.
* **Turning** uses the polar field `r(theta_b, z_i)` (rotating S; at least 720 theta bins,
  dz <= min(f_min/8, r_nose/16)): updated every L0 step for the bins the deflected elements swept;
  the chip area per element is read at the current (theta, z); it persists across passes
  (threading). The tri-dexel field of the same workpiece is synced once per revolution (revolved
  swept profile) for collisions, coolant and meshing; the polar field is authoritative for the
  removed volume while turning; at operation end the tri-dexel is rebuilt exactly from it.
* **Mass**: emission mass per tooth period = rho x analytic chip volume (integral of h db r dphi
  from the force kernel); the field's removed volume is the exact integer sum of its own interval
  deltas (for the tri-dexel the ledger volume: per dual cell the family most normal to the surface,
  docs/models/removal.md 1.1, the Z family before 2026-11-10; the radial family for polar). Once per macro step emissions are
  scaled by `V_field / V_analytic`; the ratio is the channel `removal.volume_consistency`, and a
  finding is raised if it leaves 1 +/- 2 % over a pass. Displaced volume (forming taps, burrs,
  side flow, re-added to the field) and freed bodies (connectivity check after each subtraction:
  a disconnected component becomes a loose body with part_free / chip_hazard) are ledger terms.
* **Surface events**: one row per element where a pass leaves the final surface (X, normal, tick,
  tooth, element, h, edge radius, VB, u_rel; bus.SURFACE_EVENT_COLUMNS) for the surface domain.
* **Loads to L1**: the tick's mean force per machine axis and the spindle torque go to the servo
  and spindle loops as disturbances on the next tick (one-tick lag).

### 5.5 Kernel ABI and rules

* `cnctwin/core/abi.py` fixes the shared layouts (EdgeModel, CoeffTable, SurfaceMemory,
  EngagedIntervals, ModalModel, ModalState, Accum, EventBuffer); owners define their internal
  layouts in `<domain>/abi.py` (removal: DexelField, PolarField, Envelope; motion: Chain, Servo).
* Every kernel: `@njit(<explicit signature>, cache=True, nogil=True, error_model="numpy")`;
  arguments only writable C-contiguous arrays, scalars and NamedTuples of them (no jitclass,
  typed containers, closures or first-class functions in the hot path); accumulators are checked
  for finite values at L2; each kernel has a plain-Python reference and a test comparing them;
  an integration test asserts `len(kernel.signatures) == 1`; one CI run uses NUMBA_BOUNDSCHECK=1.
* Numba's cache does not notice a changed callee in another file: the cache directory is keyed
  by a hash of every kernel source file, the numba/llvmlite versions and the CPU (`bin/tsafe`,
  `bin/twin`, the job runner and the image all set it; run.json records it; `twin warmup` builds it).
* Kernel cost rules: loop only over the engaged stations; sin/cos by angle addition; linearise
  every h-dependent coefficient at L2; subtract only within the bounding box of the engaged
  sector (the trailing half of the envelope is air).

## 6. Shared state (core/state.py)

`SimState` has core bookkeeping (clock, ledger, findings, tool table, `bus_prev`/`bus_next`,
RNG counters) and **exactly one field per domain**, owned by it: controller, motion, tooling,
removal, cutting, chips, thermal, coolant, wear, holding, structure, dynamics, surface. Per-tool
data is kept by each owner in arrays indexed by the tool slot of the immutable `ToolTable`.
The surface memory and engaged intervals are cutting's state; the dexel/polar fields removal's;
the thermal grid thermal's; modes and displacement grids structure's; modal states dynamics'.
The controller executor is an explicit state machine whose whole state is data (no Python
generators or recursion across blocks), so `ControllerState.to_arrays()` is complete.

## 7. The bus (core/bus.py)

One record per writer, double-buffered (`bus_prev` read-only, `bus_next` written, swapped at the
end of the step): `motion` (PathFrame per path, per-tick arrays), `removal` (per contact volume,
contact geometry, events, loose bodies), `cutting` (per contact wrenches per tick, loads, work,
heat generation by zone, per-element integrals, shocks, work-heat deposits, surface events,
section moments, recut rows; and ProcessCoefficients per tool for the next step), `chips`
(chip formation per contact with emissions, flute fill, packing torque/force, frozen occupancy
grid, pile heights, blocked nozzles/probe, wraps, dust, chip heat to part/fixture/coolant,
debris under locators), `dynamics` (vibration, chatter index/frequency, modal energy and damping
terms), `thermal` (per tool element temperatures at quadrature points, growths, spindle growth,
per-axis 6-component drift transforms, work temperature grids, surface/preheat/bulk temperatures,
heat to coolant), `coolant` (reach, h, friction factor, boiling, jets in M, shadows, wetted and
flush-pressure grids, fluid temperature: coolant owns the tank), `wear` (per tool [teeth, elem]
arrays, broken, expired tools), `structure` (tool modes and sections, work modes on grids,
displacement grids with components, fixture compliance), `holding` (per tool pull-out, slip,
unbalance vector and couple; per part pose delta, free, seated, location error, supports).
A test fails if two domains assign to the same record.

## 8. Domain interface (core/domain.py)

`channels`, `init`, `kernel_inputs` (NamedTuple for the compiled loop, frozen for the step),
`absorb` (accumulators and event buffers -> own bus record + trace events), `macro_step(prev,
nxt)`, `end_operation`, `finalize`. `RunContext` carries the resolved project, the databases,
the run directory, base clock, IPO and macro sizes, channel registry, trace log, recorder and
timeline. Findings are raised with `ctx.finding(state, ..., tick=...)`: line, block and
positions are resolved at that tick (pos_W = stock_pose * tip_S), and they go to
`state.findings` (the only list).

## 9. Determinism

* No stateful RNG in the loop: every draw is `hash_uniform(seed, stream_key("<domain>.<purpose>"),
  a, b, c)` with counters kept in the owner's state; `ctx.init_rng()` only at init and in outer
  studies. Each model doc lists its stream names.
* Threads: OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1 and PYTHONHASHSEED=0 in bin/twin, bin/tsafe, the
  job runner and the Dockerfile (run.json records them). No `parallel=True` reductions; prange
  only with disjoint writes reduced serially. No iteration over sets or dicts whose order is not
  defined by insertion of sorted keys.
* Iterative eigen/SVD solves pass a deterministic v0 (ones/sqrt(n)); eigenvectors are
  sign-normalised (largest-magnitude component positive); degenerate clusters (|df|/f < 1e-6) are
  rotated to a canonical basis (projection on machine axes, Gram-Schmidt in fixed order).
* The determinism test hashes the declared result set: findings.json, trace.jsonl, series/,
  frames.bin, keyframes/, part_*, inspection.json, surface.json, tools.json, production.json.
  Volatile fields (progress, wall time) live in status.json. Reports, PDF, SVG, MP4 use fixed
  metadata (SOURCE_DATE_EPOCH, Matplotlib metadata CreationDate None, ffmpeg -fflags +bitexact).
* Same project + same build + same hardware = bit-identical result set, including across a
  checkpoint/resume. Across platforms: within docs/determinism.md tolerances.

## 10. Recording

* Channels have unit, tier, owner, doc, `max_shape` and dtype; columns `<name>[i]` padded with
  NaN; tick-tier columns `<name>.min/.mean/.max`.
* `fast` (L0): a 0.5 s ring buffer; windows (first revolutions of each engagement, capped by
  minimum spacing and maximum per second; around findings raised in the step; periodic) are cut
  from it at L2. Findings raised later (end-of-operation inspection) get their window by
  deterministic replay from the nearest checkpoint (`twin replay --window`). `fast: all` stores
  float32 and `twin check` prints the size estimate.
* `tick` (L1) and `macro` (L2) tiers as described in the channels.
* Series: one Parquet file per tier per checkpoint interval (<= 10 s simulated, closed at each
  checkpoint) under `series/<tier>/`, listed in an append-only `series/index.jsonl`.
* Memory budget (docs/performance.md): the simulation process stays <= 1.2 GB for the reference
  part 300 x 200 x 100 mm: dexel field <= 300 MB (int32 fixed-point endpoints in 10 nm units,
  K = 4 inline segments + overflow pool), thermal <= 60 MB, structure <= 200 MB (FE <= 50k DOF),
  recording buffers <= 50 MB, trace index <= 20 MB, keyframe meshing per brick. STEP import,
  reports/PDF and video run in separate processes before or after the simulation.

## 11. Findings and the chain of cause

`Finding(id, time_s, tick, op, program, line, block, pos_W_m, pos_M_m, severity, category,
message, quantities, cause_chain, domain)`; categories in core/findings.py. `core.trace.record(
time_s, tick, domain, event, quantities, parents)` writes one JSON line; a finding's
`cause_chain` lists the trace ids from the finding back to its root causes. The verdict is
**safe to run** when no finding has severity error or fatal.

## 12. Conservation (checked every macro step; a violation above tolerance fails the run)

Every term is computed independently; nothing is closed by difference.

* **Mass**: removed field volume x rho(20 degC) = chips in flight + settled + conveyed + adhered +
  in flutes + dust; plus displaced volume and freed bodies as their own terms. Exact integer
  volumes make it exact (tolerance `physics.tol_mass`, 1e-9 relative).
* **Energy** per step: in = spindle shaft work (integral T omega) + feed work (integral F . v).
  Out = shear + rake + flank heat (forces x velocities by zone) + ploughing + change in modal
  energy + structural and process damping dissipation (booked as heat) + chip kinetic energy +
  packing/recut/slip friction + stored cold work + unbalance work. Tolerance `physics.tol_energy`
  (1e-6). Each thermal solver separately: change in internal energy = sources - losses - enthalpy
  of removed material.
* **Single owners**: the hot loop reports heat generation by zone; ProcessCoefficients fractions
  split it; thermal distributes it; coolant owns the tank and receives heat_to_coolant from
  thermal and chips.
* **Time**: sum of block times = cycle time, exactly (integer ticks).

## 13. Files and formats

Project `project.yaml` (format `cnctwin-project`, versioned, migrated forward); databases
`db/**/*.yaml` (templates in `db/_templates/`); run directory (16). Format docs in
docs/formats.md.

## 14. Model documentation

One per domain (docs/models/<domain>.md): equations, parameters and sources, assumptions,
validity range, error bound (and how it was established), references, RNG stream names, bus
fields read/written. docs/approximations.md collects every approximation and its error.

## 15. Code discipline

Small functions; quantities named with their units; no magic numbers outside the databases;
comments only where the why is not obvious; every finding traceable; unit tests per model
(`tests/unit/<domain>/`), run through `bin/tsafe` (memory-capped, nice 10); no stubs, no TODOs,
no "simplified for now".

## 16. Run directory (what every run writes; the API and UI read only this)

```
runs/<run_id>/
  run.json              resolved project, versions, input hashes, numba cache key, thread settings,
                        verdict {safe, reasons[]}, summary numbers, output index (no volatile fields)
  status.json           status, progress {sim_time_s, est_total_s, wall_s, eta_s} (volatile)
  findings.json  trace.jsonl  channels.json  timeline.json (per executed block)
  series/<tier>/part-<n>.parquet  series/index.jsonl
  scene.json + scene/*.glb   static scene: machine components and links, chain, axes, fixture,
                        stock, tool envelopes, nozzles
  frames.bin            per L1 tick: int64 tick, float32 axis values (relative to the stock origin
                        for linear axes), angles wrapped to [0, 2 pi) + int32 revolution count, tool
                        tip in S and tool axis per path (the UI never re-implements kinematics);
                        header in scene.json, layout in docs/formats.md
  keyframes/bricks.idx  brick -> sorted [(k, file, offset, length)]; one ranged read per brick to seek
  keyframes/kf_<k>.glb  changed IPW bricks (32^3 lattice cells, quantised positions); a full snapshot
                        every 20 keyframes; keyframes/index.jsonl (append-only)
  keyframes/kf_<k>.npz  coarse fields: surface temperatures (or thermal at <= 8 mm, float16), chip pile
                        heights, coolant wetted cells, tool deflection, part pose
  keyframes/attr/       per-brick final attributes written at the end (deviation, finish, wear)
  chips/kf_<k>.npz      particles in flight, pile summary
  part_<op>.glb|stl|3mf + part_<op>_unload.stl
  inspection.json surface.json tools.json production.json sld/<tool>.json
  report.html report.pdf setup_sheet.html setup_sheet.pdf images/*.png video.mp4 audio.wav
  checkpoints/ck_<tick>.npz + .json   (atomic: .tmp, fsync, rename, .json last as commit marker;
                        keep the last two + one per operation end). A checkpoint records the byte
                        lengths of trace.jsonl and frames.bin, the keyframe and timeline counts and
                        the closed series files; resume truncates everything after the cut.
```

## 17. HTTP API (FastAPI; OpenAPI exported to docs/openapi.json)

```
GET  /api/health
GET  /api/projects | POST /api/projects (zip) | GET /api/projects/{id} | GET /api/projects/{id}/files/{path}
POST /api/projects/{id}/check
POST /api/runs {project_id, options} | GET /api/runs | GET /api/runs/{id} (run.json + status.json)
GET  /api/runs/{id}/events (SSE: progress, finding, keyframe, done) | POST /api/runs/{id}/cancel | /resume
GET  /api/runs/{id}/findings | /trace/{event_id} | /channels
GET  /api/runs/{id}/series?tier=&channels=a,b&t0=&t1=&max_points=   (min/max-preserving downsampling)
GET  /api/runs/{id}/program/{op} | /timeline | /scene | /frames?t0=&t1= | /keyframes | /keyframes/{k}.glb
GET  /api/runs/{id}/keyframes/{k}/fields | /bricks/{brick}?t= | /chips?t= | /jets?t= | /sld/{tool}
GET  /api/runs/{id}/inspection | /surface | /tools | /production | /files/{path} (range requests)
GET  /api/runs/{id}/export?format=csv|parquet&tier=&channels=
GET  /api/compare?a=&b=  | POST /api/runs/{id}/uncertainty | POST /api/calibrate
GET  /api/db/{kind} | /api/db/{kind}/{id}
```

The UI (web/) is served at `/`; everything the CLI reports is reachable through it.

## 18. Reading project and database objects

`cnctwin.db` and `cnctwin.project` return files as `SIMap` objects (cnctwin/db/simap.py):
attribute and item access, keys converted to SI by their declared unit (`travel_mm` ->
`travel_m`), lists of mappings as lists of `SIMap`, absent optional keys as `None`, `.raw` for
the file's own values. A sourced scalar `{value, src}` is a `Sourced` (`float(x)` is SI,
`.src`, `.note`, `.estimated`); a table `{table: [[x, v]], x: degC}` is a `Table` (call with the
SI x, e.g. kelvin). `ctx.project` (ResolvedProject) mirrors db/_templates/project.yaml the same
way with database ids resolved (`ctx.project.machine`, `.material`, `.tools.list[i].tool`,
`.holder`, `.coolant.fluid`) and programs read (`.operations[i].program_text`).

## 19. Every physical domain ships a Domain object

`cnctwin/<domain>/domain.py` defines `<Name>Domain` implementing `core.domain.Domain`, reading
and writing exactly the bus records in its model doc, tested with a fake bus and state.

## 20. Performance budget (single core of this box, per simulated second, docs/performance.md)

Hot loop <= 0.45 s, L1 <= 0.1 s, L2 <= 0.25 s, recording <= 0.05 s for the reference pocket
(D10 4-flute at 10 000 rpm); `tests/perf/` fails above 0.9 s/s. `twin check` prints the estimated
CPU seconds per simulated second from element, step and query counts.

## 21. Paths

Code resolves data through `cnctwin.paths` (`ROOT`, `DB`, `WEB`, `EXAMPLES`, `VALIDATION`, `DOCS`,
`data_dir()`), overridable with `CNCTWIN_ROOT` / `DATA_DIR`; never a hard-coded /opt/plint path.
