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
end-of-operation stability analysis and the quantised, cached coefficient pipeline (L2). Missed: cut_step for the
slot (8.3 us, from 16.9); dynamics <= 1 us per L0 step for 8 modes (3.5 us in the slot's loop, from 7.6; 1.8 us in
dynamics' harness with the same shape); the 0.45 s/s hot-loop budget (6-9 s/s in cut, from 10-15) and the 1 s/s
whole-run target (3.9-4.7 s/s, from 6.6): removal's tooth pass alone is 4-7.5 s/s in cut and its connectivity
check 1.7-1.9 s/s at L2 (section 9).

## 1. Where the time goes (per L0 step in cut)

Instrumented copy of `run_macro` (time-stamp counter around each section of `contact_steps`, a `/tmp` harness, not
part of the package); removal's tooth pass split into its gating (`engaged_intervals`, contact geometry, sector
box) and the envelope subtraction.

| section | slot before | slot now | pocket before | pocket now | owner |
|---|---|---|---|---|---|
| removal tooth passes (`do_pass` -> `tooth_pass`) | 31.7 | 33.5 | 34.7 | 37.3 | removal |
| slab generation instants and pass test (`tooth_events` without the passes) | 13.8 | 0.2 | 12.2 | 0.1 | hot loop |
| engaged-interval refresh without a pass (`regate`) | 2.3 | 5.2 (2.4 at a load of 3) | 0.0 | 0.0 | removal (call: hot loop) |
| `cut_step` | 16.9 | 8.3 | 8.5 | 2.8 | cutting |
| pose: Hermite angles, axis interpolation, FK x 2, StepPose | 11.3 | 4.4 | 9.8 | 4.5 | motion FK, hot loop |
| refill planes (`refill_planes` + removal's `surface_plane`) | 9.5 | 4.4 | 4.1 | 0.9 | hot loop, removal |
| `clear_planes`, headroom, ring, scalars | 0.8 | 0.07 | 0.7 | 0.04 | hot loop |
| **run_macro per L0 step** (instrumented) | **87.5** | **57.0** | **70.7** | **45.1** | |
| run_macro per L0 step (plain build: test_performance, harness) | 53-80 | 49-59 | 61-71 | 39-45 | |

Microseconds. "Before" is the state of 2026-11-04 measured with the same instrumentation on 2026-11-09 at a load of
about 3 (the documented 53 us / 61 us were taken on a quieter box); "now" is the final code at a load of 5-6 (an
earlier run of the same rigid code at a load of 3: 50.5 us slot, 42.5 us pocket, removal's passes 32.6 / 33.9 us).
Removal's tooth pass costs 6.1 ms per pass in the slot (gating 2.0 ms, subtraction 4.1 ms; 5.5 passes per 1000
steps) and 12.4 ms in the pocket (gating 2.1 ms, subtraction 10.4 ms; 3.0 per 1000 steps, lattice 0.075 mm): **it
is now three fifths to four fifths of the loop and alone 4.2 s/s (slot) and 7.5 s/s (pocket)**, 9-17 times the
whole budget. The architecture's claim that cut_step and the modal kernels were 94 % of the loop held for cutting's
own benchmark, not for the integrated loop: there the composition (reference counting, argument copies, the slab
loop) and removal's passes dominated.

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
| removal `tooth_pass` | slot / pocket | 5.9 / 11.3 ms per pass | unchanged (6.1 / 12.4 ms at a load of 5-6) | removal's kernel (open problem) |
| mechanics `continuation_reference` | per node of a map row without an admissible solution | 18-22 ms | 11-12 ms | the row's own solved nodes skip their t1 trials |
| mechanics `build_process_map` (lazy) | 108-node map (adiabatic) | 0.95 s eager | 0.09 s + refinement of the rows read | `lazy=True` |

## 3. Per level

| level | what is in it | slot | pocket | budget |
|---|---|---|---|---|
| L0, in cut | `run_macro`'s L0 steps (plain build) | 49-59 us per step = 6.2-7.4 s/s | 39-45 us per step = 7.8-9.0 s/s | 0.45 s/s (with L1) |
| L1, per 1 ms tick | `motion_tick` 20 us, broad phase (removal's `l1_contact`, `collide_envelope`) 10 us, tick loads 3 us | 33 us = 0.033 s/s | 34 us = 0.034 s/s | 0.1 s/s |
| L1, out of contact | the same plus dynamics' `modal_free` (exact free propagation) with the modes | +5-11 us per tick | | |
| hot-loop wrapper, per 10 ms macro step | `_build` (every domain's `kernel_inputs`, contacts, padding), `_absorb` | 2.3 ms = 0.23 s/s | 2.9 ms = 0.29 s/s | |
| L2, per 10 ms macro step | the domains' macro steps (whole runs, section 4) | 2.3 ms = 0.23 s/s | 20 ms = 2.0 s/s (removal 1.9, cutting 0.16) | 0.25 s/s |
| recording | not exercised by these runs (no recorder) | | | 0.05 s/s |

The wrapper's padding contacts and the rigid contacts' dynamics placeholders are made once per shape (they were
rebuilt every macro step). The wrapper's remaining cost is mostly cutting's and removal's `kernel_inputs` (fresh
step buffers per the kernel_inputs / absorb contract) and the contacts' geometry.

## 4. Whole runs

| run | simulated (in cut) | CPU | s/s | hot loop (run_macro + wrapper) | L2 by domain | before |
|---|---|---|---|---|---|---|
| pocket helical entry, two turns, default L2 settings, rigid | 6.54 s (0.86 s) | 28.8 s | 4.41 (test_performance: 3.88-4.68) | 11.8 s (9.4 + 1.9) | 13.4 s: removal 12.3, cutting 1.0 | 6.6 s/s (42.7 s: hot 18.9, L2 21.0) |
| slot, rigid | 5.24 s (0.78 s) | 9.4 s | 1.80 | 7.8 s (6.4 + 1.2) | 1.2 s: removal 0.8, cutting 0.4 | 2.5-2.8 s/s |
| slot with the machine's modes, fast-L2 test settings | 5.24 s (0.77 s) | 14.6 s | 2.78 | 8.1 s (6.3 + 1.6) | 6.1 s: dynamics 5.0, removal 0.7, cutting 0.4 | dynamics' L2 alone ~19 s (profiled) |

The pocket run's L2 is removal's connectivity check: `release_loose` -> `find_bodies` (a labelling of the whole
0.075 mm field) every few macro steps, 162 calls of 67 ms = 10.8 s. Without it the run would be ~2.5 s/s, of which
removal's tooth passes in cut are ~1 s/s of the run.

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

* **Removal's tooth pass** (6 ms per pass at lattice 0.1 mm, 12 ms at 0.075 mm: about 33-37 us per L0 step,
  4-7.5 s/s) keeps the loop 9-17 times above the 0.45 s/s budget and the whole pocket run above 1 s/s. Its cost
  scales with the rays of the engaged sector box (1 / lattice^2 per area, the full stock height of the box), the
  subtraction two thirds of it in the slot and five sixths in the pocket. Not in this task's ownership
  (cnctwin/removal).
* **Removal's connectivity check at L2** (`release_loose` -> `find_bodies`, 67 ms per call on the pocket's field,
  every few macro steps): 1.7-1.9 s/s of the whole pocket run, more than all other L2 work together. An
  incremental check (only the cells the step's sweep touched) would remove it. Not in this task's ownership.
* **Motion's FK** (`fk_poses`, two per L0 step: mid and end frames, ~2 us each with the error transforms of every
  link) is most of the 4.4 us pose section.
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
