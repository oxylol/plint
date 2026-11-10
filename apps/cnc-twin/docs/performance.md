# Performance (architecture 20)

Budget per simulated second on one core of this box (i5-4690, 3.5 GHz, 6 MB L3 shared by 4 cores): hot loop
<= 0.45 s, L1 <= 0.1 s, L2 <= 0.25 s, recording <= 0.05 s; the performance task adds a whole-run target of 1 CPU s
per simulated second for the pocket-islands example's helical entry including L2. Process CPU time throughout
(`time.process_time`; the per-section split by the time-stamp counter), measured through `bin/tsafe` (nice 10) on
the shared box: the other agents' jobs keep the load at 4-6 on the 4 cores, which moves single timings by
+-15 % (the memory-bound removal kernels more than the rest). `tests/integration/test_performance.py` holds the
regression bounds and the budget assertions (expected failures, strict).

Cases (`tests/integration/hotfix.py`): the **slot** (D10 4FL MHSV, 10000 rpm, f_z 0.04 mm, a_p 2.9 mm between two
stations, 40 x 30 x 10 mm stock at lattice 0.1 mm, stride 8 us = 0.48 deg), the **pocket** (pocket-islands-al6061,
the helical entry of its D10 3FL roughing end mill, 12000 rpm, lattice 0.075 mm, stride 5 us), both rigid unless
stated, and the slot with the VF-2SS machine's modes (4 tool / machine modes and 4 workpiece modes, 119 stations).

**Status.** Met: cut_step <= 3 us per L0 step for the pocket's helical entry (2.8 us); L1 (0.03 s/s); the
end-of-operation stability analysis and the quantised, cached coefficient pipeline (L2); removal's tooth pass
<= 8 us per L0 step for the pocket (7.3 us, from 32.5), removal's connectivity check <= 0.1 s/s (0.02 s/s in the
whole pocket run, from 1.7-1.9) and motion's FK <= 1.5 us per L0 step (two calls of 0.49 us, from 1.8 us each)
(section 10). Missed: cut_step for the slot (8.3 us, from 16.9); dynamics <= 1 us per L0 step for 8 modes (3.5 us
in the slot's loop, from 7.6; 1.8 us in dynamics' harness with the same shape); removal's tooth pass for the slot
(8.3 us per L0 step, from 29.4; plus 2.1 us of engaged-interval refreshes without a pass, from 4.1); the 0.45 s/s
hot-loop budget (2.6-2.8 s/s in cut for the slot, 2.5 s/s for the pocket, from 6-9) and the 1 s/s whole-run target
(1.43 s/s, from 4.41; section 9).

## 1. Where the time goes (per L0 step in cut)

Instrumented copy of `run_macro` (time-stamp counter around each section of `contact_steps`, a `/tmp` harness, not
part of the package); removal's tooth pass split into its gating (`engaged_intervals`, contact geometry, sector
box) and the envelope subtraction.

| section | slot before | slot now | slot after removal / FK | pocket before | pocket now | pocket after removal / FK | owner |
|---|---|---|---|---|---|---|---|
| removal tooth passes (`do_pass` -> `tooth_pass`) | 31.7 | 33.5 | 8.3 | 34.7 | 37.3 | 7.3 | removal |
| slab generation instants and pass test (`tooth_events` without the passes) | 13.8 | 0.2 | 0.1 | 12.2 | 0.1 | 0.1 | hot loop |
| engaged-interval refresh without a pass (`regate`) | 2.3 | 5.2 (2.4 at a load of 3) | 2.1 | 0.0 | 0.0 | 0.0 | removal (call: hot loop) |
| `cut_step` | 16.9 | 8.3 | 6.7 | 8.5 | 2.8 | 2.3 | cutting |
| pose: Hermite angles, axis interpolation, FK x 2, StepPose | 11.3 | 4.4 | 1.4 | 9.8 | 4.5 | 1.4 | motion FK, hot loop |
| refill planes (`refill_planes` + removal's `surface_plane`) | 9.5 | 4.4 | 3.1 | 4.1 | 0.9 | 0.7 | hot loop, removal |
| `clear_planes`, headroom, ring, scalars | 0.8 | 0.07 | 0.04 | 0.7 | 0.04 | 0.04 | hot loop |
| **run_macro per L0 step** (instrumented) | **87.5** | **57.0** | **22.6** | **70.7** | **45.1** | **12.4** | |
| run_macro per L0 step (plain build: test_performance, harness) | 53-80 | 49-59 | 21.0 | 61-71 | 39-45 | 12.2 | |

Microseconds. "Before" is the state of 2026-11-04 measured with the same instrumentation on 2026-11-09 at a load of
about 3 (the documented 53 us / 61 us were taken on a quieter box); "now" is the final code at a load of 5-6 (an
earlier run of the same rigid code at a load of 3: 50.5 us slot, 42.5 us pocket, removal's passes 32.6 / 33.9 us).
Removal's tooth pass costs 6.1 ms per pass in the slot (gating 2.0 ms, subtraction 4.1 ms; 5.5 passes per 1000
steps) and 12.4 ms in the pocket (gating 2.1 ms, subtraction 10.4 ms; 3.0 per 1000 steps, lattice 0.075 mm): it
was then three fifths to four fifths of the loop and alone 4.2 s/s (slot) and 7.5 s/s (pocket), 9-17 times the
whole budget (after section 10: 1.0 and 1.5 s/s). The architecture's claim that cut_step and the modal kernels were 94 % of the loop held for cutting's
own benchmark, not for the integrated loop: there the composition (reference counting, argument copies, the slab
loop) and removal's passes dominated.

"After removal / FK" (section 10) is the code of 2026-11-09 10:07 at a load of 2.3, with the day's changes of the
other domains (cut_step 6.8 / 2.6 us just before this work); the same harness on the code just before it (09:20, a
load of 3.3) gave run_macro 49.1 / 40.6 us, removal's passes 29.4 / 32.5 us (5.37 / 10.83 ms per pass: gating 1.51,
subtraction 3.85 / 9.32 ms), regate 4.1 us, pose 4.1 / 4.2 us, refill 3.8 / 0.8 us. Removal's tooth pass now costs
1.51 ms per pass in the slot (gating 0.78, subtraction 0.73) and 2.45 ms in the pocket (gating 0.46, subtraction
1.98): 8.3 and 7.3 us per L0 step, the loop's largest section in the pocket, second to cut_step in the slot.

## 2. Per kernel

| kernel | case | before | now | how |
|---|---|---|---|---|
| `cut_step` | slot, 27.4 engaged elements of 60 visited (15 engaged stations) | 16.9 us | 8.3 us (7.5 us cache-warm) | section 6 |
| `cut_step` | pocket helical entry | 8.5 us | 2.8 us | |
| `cut_step` | cutting's benchmark: 4 x 80 elements, 160 engaged (`test_milling_kernel_cost`) | 48.6 us, 303 ns/elem | 33-36 us, 210-225 ns/elem | |
| `modal_begin` + `modal_correct` | in the slot's loop: 8 modes, deflections at 61 stations (15 engaged + 2 x 26), loads at 15 | ~55 us (2026-11-04: residual, every station); 7.6 us (5.0 + 2.6) with the station ranges alone | 3.5 us (2.2 + 1.3) | RAW, contiguous runs (section 6) |
| `modal_begin` + `modal_correct` | dynamics' harness, the slot's shape (4 + 4 modes, 119 stations, act 61 / 15), cache-warm | 5.2 us | 1.8 us | bit-identical results |
| `modal_begin` + `modal_correct` | dynamics' harness, 5 modes, 1 station / 13 modes, 20 stations (`test_cost_and_settings`) | 1.3-1.7 / 5.1-7.9 us | 0.76 / 4.5 us | |
| `tooth_events` (no pass) | slot | 13.8 us | 0.2 us | tooth windows, counts kept with the slab states |
| `refill_planes` + `clear_planes` | slot, 4.4 planes per step | 10.3 us | 4.4 us | lean arguments, RAW |
| `set_step_pose` | per step | ~2 us | ~0.1 us | no allocation, StepPose only |
| removal `tooth_pass` | slot / pocket, in the loop | 5.37 / 10.83 ms per pass (gating 1.51 / 1.51, subtraction 3.85 / 9.32) | 1.51 / 2.45 ms (gating 0.78 / 0.46, subtraction 0.73 / 1.98) | section 10; results bit-identical |
| removal `engaged_intervals` | slot / pocket, before each replayed pass (section 10) | 2.56 / 1.88 ms | 0.71 / 0.34 ms | section 10; identical intervals |
| motion `fk_poses` | VF-2SS chain (3 + 2 links, error transforms) | 1.80 us | 0.49 us | one RAW function, no calls; bit-identical |
| removal connectivity per macro step | pocket's 0.075 mm field (1.4 M z columns) | 67-69 ms (`find_bodies`) | 0.7-1.0 ms (`local_connected`, 27-32 k columns) | section 10 |
| mechanics `continuation_reference` | per node of a map row without an admissible solution | 18-22 ms | 11-12 ms | the row's own solved nodes skip their t1 trials |
| mechanics `build_process_map` (lazy) | 108-node map (adiabatic) | 0.95 s eager | 0.09 s + refinement of the rows read | `lazy=True` |

## 3. Per level

| level | what is in it | slot | pocket | budget |
|---|---|---|---|---|
| L0, in cut | `run_macro`'s L0 steps (plain build) | 21 us per step = 2.6 s/s (49-59 us = 6.2-7.4 s/s before section 10) | 12.2 us per step = 2.4 s/s (39-45 us = 7.8-9.0 s/s) | 0.45 s/s (with L1) |
| L1, per 1 ms tick | `motion_tick` 20 us, broad phase (removal's `l1_contact`, `collide_envelope`) 10 us, tick loads 3 us | 33 us = 0.033 s/s | 34 us = 0.034 s/s | 0.1 s/s |
| L1, out of contact | the same plus dynamics' `modal_free` (exact free propagation) with the modes | +5-11 us per tick | | |
| hot-loop wrapper, per 10 ms macro step | `_build` (every domain's `kernel_inputs`, contacts, padding), `_absorb` | 2.3 ms = 0.23 s/s | 2.9 ms = 0.29 s/s | |
| L2, per 10 ms macro step | the domains' macro steps (whole runs, section 4) | 1.0 ms = 0.10 s/s (removal 0.02, cutting 0.08; 2.3 ms before section 10) | 1.7 ms = 0.17 s/s (removal 0.03, cutting 0.14; 20 ms = 2.0 s/s before) | 0.25 s/s |
| recording | not exercised by these runs (no recorder) | | | 0.05 s/s |

The wrapper's padding contacts and the rigid contacts' dynamics placeholders are made once per shape (they were
rebuilt every macro step). The wrapper's remaining cost is mostly cutting's and removal's `kernel_inputs` (fresh
step buffers per the kernel_inputs / absorb contract) and the contacts' geometry.

## 4. Whole runs

| run | simulated (in cut) | CPU | s/s | hot loop (run_macro + wrapper) | L2 by domain | before |
|---|---|---|---|---|---|---|
| pocket helical entry, two turns, default L2 settings, rigid | 6.54 s (0.86 s) | 9.3 s | 1.43 | 5.1 s (3.1 + 1.7) | 1.1 s: removal 0.18, cutting 0.91 | 4.41 s/s (28.8 s: hot 11.8, L2 13.4, removal 12.3) before section 10; 6.6 s/s (42.7 s) before that |
| slot, rigid | 5.24 s (0.78 s) | 5.05 s | 0.96 | 4.2 s (2.8 + 1.1) | 0.50 s: removal 0.10, cutting 0.40 | 1.80 s/s (9.4 s) before section 10; 2.5-2.8 s/s before that |
| slot with the machine's modes, fast-L2 test settings | 5.24 s (0.77 s) | 14.6 s | 2.78 | 8.1 s (6.3 + 1.6) | 6.1 s: dynamics 5.0, removal 0.7, cutting 0.4 | dynamics' L2 alone ~19 s (profiled) |

The pocket run's L2 was removal's connectivity check: `release_loose` -> `find_bodies` (a labelling of the whole
0.075 mm field) at every macro step once a column had changed its segment count (the counter was never reset while
the material stayed one body), 162 calls of 67 ms = 10.8 s. The incremental check (section 10) makes 19 checks in
the run, 18 of them on the changed region alone: 0.15 s = 0.023 s/s (the slot run: 78 checks, 77 local, 0.015
s/s). What remains of the whole pocket run: the hot loop 5.1 s (0.77 s/s; run_macro 3.1 s, the wrapper's build and
absorb 1.7 s), cutting's coefficient pipeline 0.9 s, the rest of the driver step (motion, the controller, the
publishers, ledger and checks) 3.2 s.

## 5. L2

* **Dynamics' stability analysis** (docs/models/dynamics.md 5.4): lobe diagrams and the semi-discretisation check
  are report artefacts and an input to nothing in the loop. During the run only the limit at the operating speed
  is computed (the bus `depth_limit_m`, the stability finding), once per engagement class and speed (cached by
  key in the checkpointed state); its chatter-frequency scan evaluates all frequencies' characteristic matrices at
  once, the sign changes of all frequency intervals are bisected together (one stacked eigenvalue call per
  bisection level instead of one call per interval and level: 60880 scalar calls in a 5 s slot run), and the cut
  model caches its per-tooth means (they were recomputed at every frequency: most of the ~200 s per engagement of
  the default grids). The diagrams (at most `sld_max_diagrams` per tool, smallest margins first; uniform pitch:
  the analytic ZOA lobes, variable pitch: a coarse speed and frequency grid) and the SD checks run once at the
  end of the operation. Chatter detection stays the time-domain index of every step. In the slot with the
  machine's modes (fast-L2 test settings, 5.24 s simulated) dynamics' L2 is 5.0 s: the limits at the operating
  points ~2.2 s (19 engagement classes, ~0.13 s each; 15.9 s with the scalar bisection) and the time-domain
  chatter detection ~2.8 s (108 windows of ~26 ms, section 9).
* **Cutting's coefficient pipeline**: built per tool once; rows at the quantised slow state (speed and friction
  factor on log grids, temperatures, flank wear, BUE height and edge radius on uniform grids of the rebuild
  tolerances), recomputed only when a row's cell changes, equal cells from a cache keyed
  by the quantised state (512 rows); a tool that is not cutting while its spindle changes speed gets provisional
  rows and waits for the speed to settle; the mechanics process maps are solved coarse and refined per row when a
  lookup first reads it (coarse to fine). Run-up of 100 macro steps then 20 steady steps, mechanics model, default
  cadence (every macro step): D10 end mill 7.0 s -> 0.83 s CPU (rows computed 71 -> 6), D6.8 drill 211 s (task
  statement; 27 s re-measured with the old refresh rule on this box) -> 7.0 s (52 row computations of 18 rows, 6.2
  s of it the lazy process maps). What remains is the first build of each row.
* **Mechanics' table building**: lazy process maps (a row's fallback refinement when a lookup first reads it) and
  continuations that do not re-solve the thickness trials next to the row's own solved nodes (section 2): the
  drill run-up's continuations 1404 calls / 16.1 s -> 108 calls / 0.3 s.

## 6. Hot loop composition (how the costs were reached)

### Reference counting

`run_macro`, `contact_steps`, `tooth_events`, the hot-kernel helpers, `cut_step` and dynamics' `modal_begin` /
`modal_correct` allocate nothing (scratch vectors are views of `HotContact.work`) and are compiled with
`_nrt=False` (`RAW`, as removal's hot kernels). Numba increfs every member array of a NamedTuple taken from a
larger one when it cannot prove the reference dead; in a loop with branches that was ~80 atomic operations per
access of a 40-array tuple (457 ns, measured); for a DynSystem about 1.2 us per L0 step.

### Lean arguments

A call that LLVM does not inline copies every member array descriptor of its arguments onto the stack (CutArgs:
~150 arrays, ~1000 words, ~1 us; `clear_planes(cut, ...)` cost 0.75 us for a few stores). The per-step helpers take
the arrays they read (`refill_planes`, `clear_planes`, `set_step_pose`, `slab_point`); the step's layouts are taken
from CutArgs once per call, not per step; `set_step_pose` and `rotvec` no longer allocate. The two dynamics calls
per step still pass the whole DynSystem (~0.15 us each, section 9).

### Tooth windows

A tooth's slabs are visited only when the step's swept angle reaches the arc holding their phases; the pass test
keeps its counts with the slab states (docs/models/hotloop.md section 4): 13.8 us -> 0.2 us per step in the slot.

### cut_step

Only stations with engaged intervals (or loads written in the last two steps) are visited, within the range of
such stations; an element's swept arc is classified against a station's single interval with two comparisons (the
general `gate` within 1e-12 rad of a boundary: identical results) and interior elements skip the boundary tests of
the memory stencil; every element constant comes from one packed row (`CutTool.pk` / `pki`, 48 + 10 columns,
built at L2, rows in station-list order); the small helpers are inlined at the IR level; the memory crossing uses a
multiplication by 1 / bins-per-step and a conditional wrap instead of a division and a modulo. Per element and step
there are no transcendental functions (sin / cos of the spindle angle by angle addition in the caller, table lines
with a segment index from a multiplication). What it costs now (perf counters, cutting's benchmark): ~1200
instructions per engaged element at an IPC of 1.3-1.6; the hottest instructions wait on the memory bins (point,
tick and valid of the shared SurfaceMemory are three arrays, three cache lines per bin pair) and on the per-element
state, and the ~50 frame and station constants live across the element loop spill (16 vector registers).

### Dynamics

`modal_begin` computes deflections only for the step's stations (`DynSystem.act`: the engaged ones and
`CI_ST_MARGIN` = 2 + the most stations of one slab on each side, for the slab interpolation: 26 in the slot) and
`modal_correct` projects only the engaged stations' loads (every other station has none); with the identity tool
mount of rotating tools cutting reads dynamics' deflection arrays in place (no copy). Both kernels are RAW, and
their station loops run over the stations' 6 components as one contiguous run (`u_st`, `v_st`, a mode's
`tool_shape` row and the station loads are C-contiguous [station, 6]): the deflection update is vectorised (every
element still sums the modes in their order), the load projection sums four modes side by side without a branch
in the loop (each mode's sum in its order). Results are bit-identical to the former kernels (checked on the
slot's shape and on systems with a dense residual flexibility). The tool's static residual (when a tool has one:
not the slot's) adds a dense 3 x 3 block per loaded and deflected station pair: +2.6 us for 15 x 27 stations.

## 7. Memory

| run | peak RSS (VmHWM) |
|---|---|
| slot rigid, whole run (40 x 30 x 10 mm stock, lattice 0.1 mm) | 0.66 GB |
| slot with the machine's modes, whole run | 0.67 GB |
| pocket-islands helical entry, whole run (lattice 0.075 mm, default L2) | 1.16 GB |
| cold compile of every kernel the loop calls (`compile_time` harness) | 1.44 GB |

The integration tests run one file per process with `TSAFE_MEM=1700M` (a cold cache compiles in the first file).

## 8. Compile time

| modules (in import order) | cold compile, CPU |
|---|---|
| cutting kernels (`cut_step`, `turn_step` and their helpers) | 57 s |
| dynamics kernels | 26 s |
| motion, removal (`k_*` modules), recorder | 243 s |
| `hot_kernels` | 9 s |
| `hotloop` (`contact_steps`, `run_macro`, `tooth_events`, ...) | 365 s |
| **total** | **700 s (11.7 min), peak RSS 1.44 GB** |

An empty `NUMBA_CACHE_DIR` (`compile_time` harness: the imports in that order, every kernel with its explicit
signature compiles at import), process CPU at a load of 4-6. Numba links each callee's IR into its caller's
module and optimises the whole tree again at -O3, so the top of the tree (`run_macro` over `contact_steps` over
cut_step, removal's passes and the dynamics kernels) costs the most; the RAW kernels and the IR-inlined helpers
add to it (the hot loop's own functions were noted at ~283 s before this work, not under the same conditions).
No easy reduction was found that keeps the explicit signatures and -O3: a lower optimisation level or fewer
inlined helpers trade it against the loop's cost. The cold cache is paid once per change of any kernel file:
`bin/tsafe` keys the cache by all of them, because Numba does not notice a changed callee in another file.

## 9. Open problems

* **Removal's tooth pass** (1.5 ms per pass at lattice 0.1 mm, 2.4 ms at 0.075 mm: 8.3 / 7.3 us per L0 step, about
  1 s/s in cut; section 10) is still the largest section of the pocket's loop. In the slot: the gating (0.78 ms)
  classifies about 2600 samples near the boundary in full (the front half of the peripheral stations, within a
  feed per tooth of the previous cut: ~400 cycles each) and refines ~85 boundaries (~6500 cycles each, the Illinois
  iterations with their sine and cosine); the subtraction (0.73 ms) books 4400 changed rays (the ray against its
  one slab: a quadratic's square root and divisions, ~170 cycles; the write and the brick booking). In the pocket
  the subtraction (1.98 ms) books 15 000 z columns lowered by 2.2 um per pass (the ramp's floor: every column
  under the end edges). Exact further steps: fewer gatings (the hot loop's refresh without a pass, 2.1 us per
  step in the slot), a classification that shares more between the stations of one circle.
* **Removal's connectivity check at L2**: incremental since section 10 (0.02 s/s). A through cut whose two sides
  join only outside the changed region falls back to the whole field (67 ms on the pocket's field) at every macro
  step with a changed segment count until the body separates.
* **Motion's FK**: 0.49 us per call since section 10 (the pose section 1.4 us per step).
* **cut_step for the slot**: ~300 ns per engaged element (~1200 instructions at IPC ~1.3, register spills of the
  ~50 live frame and station constants, L2 misses on the memory bins); the 3 us target is met for the pocket's
  helical entry, not for the slot's 27 engaged elements (8.3 us).
* **Dynamics**: 3.5 us per L0 step in the slot's loop (1.8 us in dynamics' harness with the same shape,
  cache-warm) against the 1 us target for 8 modes. What remains: the deflections at 61 stations (the margin of 26
  stations on each side covers the slab interpolation of the slot's 24-station slabs; the stations the step's
  slab points actually read would be about 30), the two calls' argument passing (~0.3 us: inlining them into
  `contact_steps` measured that much, but an IR-inlined body takes its caller's reference counting into
  dynamics' own drivers), cache misses on `tool_shape` (45 KB) next to cut_step and removal, the energy
  integrals. At L2 the time-domain chatter detection costs ~26 ms per analysed window (the forced-line masks over
  ~375 spindle harmonics in a Python loop, three detrending fits per window, the FFTs): 0.5 s/s in the slot with
  the modes, twice the whole L2 budget.
* **Compile time**: about 12 CPU minutes and 1.44 GB with an empty cache (section 8), paid again after any edit
  of a kernel file.

## 10. Removal and FK (second pass, 2026-11-09)

Removal's tooth pass, its connectivity check at L2 and motion's FK, with exact results throughout: every field
array, accumulator, event and engaged interval of 200 tooth passes of the slot and 120 of the pocket's helical
entry bit-identical (a replay harness: the captured in-cut states continued by synthetic passes along their
paths), `tests/unit/gpu` parity of the GPU dexel path, and the whole slot and pocket (two-turn helical entry)
driver runs bit-identical before and after (the wrench of every tick, the removed volumes, the final field,
the findings). Each step measured with the replay harness (process CPU and the time-stamp counter, ms per pass,
slot / pocket; gating alone on a copy of the engaged intervals before each pass; a load of 2-4: +-15 %):

| change | tooth pass | of it gating |
|---|---|---|
| before | 7.81 / 11.54 | 2.56 / 1.88 |
| subtraction: only the slabs that can reach the ray's material (the slab window: the slab axis points' spread and the ray's material extent in the scratch, `_env_ray` reads them); no division for rays along the axis; quick reject without the square root; a ray whose quantised intervals meet no material left without load / store; one-segment trims written in place; pieces booked in batches of 512, one brick directly, brick indices without integer division, sums in registers. Gating: a valid zero-spread family decides at once, the other families followed only while they can still be chosen (validity first; the smallest-spread choice among invalid families in full only when none is valid), the lattice coordinates of a sample computed once for the empty-brick test and the three families, circles beyond the lattice skipped | 1.90 / 2.82 | 0.97 / 0.69 |
| per-axis envelope box (the slab points' extent along each axis, not their 3-D spread: the columns cut to the floor under the tool fall outside it), the cylinder test on rays along an axis-aligned tool axis | 1.72 / 2.86 | 0.90 / 0.57 |
| gating: a sample of a horizontal circle whose four z columns end at or below it (all far, or all at one height within SPREAD_CELLS) is air without classifying (the end-edge stations above the floor, the air behind the cut) | 1.55 / 2.47 | 0.74 / 0.33 |
| a piece strictly inside the last piece's brick booked into it directly | 1.49 / 2.32 | 0.71 / 0.34 |

In the loop (the instrumented harness of section 1): 5.37 -> 1.51 ms per pass in the slot, 10.83 -> 2.45 ms in the
pocket; per L0 step 29.4 -> 8.3 us and 32.5 -> 7.3 us; the refresh without a pass (`regate`, the same gating) 4.1 ->
2.1 us per step in the slot. The replay's passes run back to back (warm caches); in the loop they are 180 (slot)
and 330 (pocket) L0 steps apart.

**Connectivity** (docs/models/removal.md section 9): while the material is one body and only subtractions changed
it, each check labels the column rectangle of the bricks changed since the last one and a ring of one column around
it (`k_connect.local_connected`): one component reaching the ring proves the material still one body; otherwise
the whole field is labelled, so a body separates at exactly the macro step it would with the full labelling
(`test_06`: a trepanned core at the step that closes the ring through the stock, a part on two tabs at the step
that cuts the second, a blind groove never). The change marks and the segment-count counter restart from zero
after every check (the counter was never reset while the material stayed one body: the full labelling ran at
every macro step after the first split column). Pocket's field: 0.7-1.0 ms per local check (27-32 k columns of the
1.4 M), 67-69 ms per full labelling. Whole runs: pocket 19 checks (18 local) 0.15 s = 0.023 s/s (162 full labellings
= 10.8 s before), slot 78 checks (77 local) 0.076 s = 0.015 s/s.

**FK** (`fk_poses`): the chain product written out in one function compiled without reference counting (the
helpers it called per link, each with a copy of the chain's array descriptors and a reference count per array,
cost more than the arithmetic): 1.80 -> 0.49 us per call on the VF-2SS chain, the same operations in the same
order (bit-identical poses, 635 pose sets: the captured chains and 300 random chains with rotary links, error
polynomials, thermal strain and work-spindle links). The pose section of an L0 step: 4.1 -> 1.4 us.

**Refill planes**: removal's `k_sd.level_at`, `dexel_sd` and `gradient` (every stale memory bin's plane) compiled
without reference counting, `gradient` without its work arrays: the refill section 3.8 -> 3.1 us per step in the
slot, 0.8 -> 0.7 us in the pocket.

## 11. Structure's workpiece model at 50 000 DOF (2026-11-09)

The work model's default (`physics.structure.max_dof` 50 000) cost about 1 s per macro step on a side-step or pocket
stock (docs/convergence.md 5 capped it at 3000 DOF). Measured on the convergence study's macro-step case (a side
step 0.5 x 1 mm along the 150 x 20 x 15 mm Al block in the vise, every domain on, 7 s simulated: 700 macro steps,
1.5 s of them in cut; 46 662 DOF on 1.5 mm cells; process CPU through bin/tsafe on the shared box, a load of 2-3),
with every part of a job timed (a `/tmp` harness: each job's inputs pickled with the material fractions frozen and
replayed through the old and the new code):

| part of the cost | before | after |
|---|---|---|
| structure's macro steps, total (700 steps) | 503 s = 0.719 s per step | 63.9 s = 0.091 s per step |
| set-up at the run's first step (full eigen solve, 20 modes: ARPACK shift-invert, about 50 multigrid-CG solves of 14 iterations) | 57.0 s, unclamped (holding had not published yet) | waits one step |
| set-up, clamped | 61.5 s | 47.9 s (the same solve; load) |
| refresh job, 13 of them (every 50 steps: the temperatures changed, no cell removed) | 27-31 s: Ritz 24-27 s (two block solves of 20 columns in chunks of 6), clamping system and solve 1.1-1.3 s, a multigrid set-up 0.4 s, thermal solve 0.8 s | 0.86-1.1 s: the thermal solve on the kept system, the published grids |
| idle macro step (fixture compliance, channel values) | 1.6 ms | 1.5 ms |
| release at the operation's end (full eigen solve of the part on a 3-2-1 support; in end_operation, not in the steps above) | 45.7 s | 57.5 s (the same solve) |
| whole run | 679.5 s | 239.4 s |

Amortised over an operation (set-up and release once), a refresh every 50 steps costs 0.56 s per macro step before
and 0.022 s after (x 25). What changed (docs/models/structure.md 3.4-3.5): an unchanged model (the same cells,
supports, clamped state and material: `model_key`) is not solved again (its eigenpairs, clamping and
residual-stress displacements are the latest solution's; the held system with its multigrid hierarchy stays in a
runtime cache), the first step waits for holding's record of the part, and block solves run all 20 Ritz columns in
one block. Exactness, replaying the 13 refresh jobs: displacement grids (thermal, clamping, residual, total) and the
clamp loads at the supports bit-identical, frequencies within 7e-12, MAC 1, residual compliance within 5e-12 of
the former refresh (which re-polished the same eigenpairs).

**When material removal flips cells** (a pocket's roughing: at about 2.5 mm cells a D10 removing 250 mm3 per 0.5 s
flips 15-30 cells per refresh) the model changes and the refresh is a Ritz solve. Measured by removing a trench of
5 / 30 / 300 cells along the front top edge of the same 46 600-DOF model, against a full eigen solve of the changed
model:

| cells removed | refresh before | after | full eigen solve | published frequencies vs full | residual compliance vs full |
|---|---|---|---|---|---|
| 5 | - | 21.4 s | 61.4 s | 3e-10 | 1.4 % |
| 30 | 27.9 s | 19.9 s | 57.6 s | 6e-8 (MAC 1 - 1.4e-8) | 1.7 % |
| 300 | 50.2 s | 39.2 s | 54.5 s | 2e-13 | 4.4 % |

The gain is the block width (one block of 20 columns: 7.7 s per block solve against 11.1 s, bit-identical
solutions) and the Ritz start on the previous modes' own Ritz pairs (a change that hardly moves the modes is
accepted after one block solve; none of these three was). The residual compliance differs from the full solve by the
former refresh's own amount (residual modes accepted at 10 x `ritz_tol`), the displacement grids not at all. Each
Ritz iteration is one block solve (about 7.7 s here: 20 columns of 14 CG iterations). Since 2026-11-10 the exact
iteration starts after block steps preconditioned by the V-cycle (docs/models/structure.md 3.5): on the same stock
clamped on its bottom face (a different support from the table above) the three trenches cost 9.2 / 9.1 / 12.6 s
against 12.9 / 13.0 / 30.5 s for the former refresh (one block solve instead of 2 / 2 / 5), against a full eigen
solve frequencies within 7e-8 and the residual compliance as before; the start from the old modes plus static
corrections from a local patch (below) was not needed for this. Open before that change: a start that passes the
first check (the previous modes plus their static corrections from a direct solve on a patch of a few cells around
the removed ones: the residual of the old modes in the new model is the removed cells' element forces, local and
for the stiffness part self-equilibrated; the one global block solve then checks it), which would halve a changed
refresh; a looser inner tolerance for the Ritz basis (1e-8 instead of 1e-10: 15 % less, the residual estimate
changes by 4e-16); and the set-up and release eigen solves (45-60 s each at 47 000 DOF, once per operation).

