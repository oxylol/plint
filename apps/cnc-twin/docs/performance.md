# Performance (architecture 20)

Budget per simulated second on one core of this box (i5-4690, 3.5 GHz, shared): hot loop <= 0.45 s, L1 <= 0.1 s,
L2 <= 0.25 s, recording <= 0.05 s; the performance task adds a whole-run target of 1 CPU s per simulated second
for the pocket-islands example's helical entry including L2. Process CPU time throughout (`time.process_time`),
measured through `bin/tsafe` (nice 10) on the shared box: the other agents' jobs keep the load at 4-6 on 4 cores,
which moves single timings by +-30 %. `tests/integration/test_performance.py` holds the regression bounds and the
budget assertions (expected failures, strict).

Cases (tests/integration/hotfix.py): the **slot** (D10 4FL MHSV, 10000 rpm, f_z 0.04 mm, ap 2.9 mm between two
stations, lattice 0.1 mm, stride 8 us = 0.48 deg), the **pocket** (pocket-islands-al6061, its D10 3FL roughing end
mill's helical entry, 12000 rpm, lattice 0.075 mm, stride 5 us), both rigid unless stated, and the slot with the
VF-2SS machine's modes.

## 1. Where the time goes (per L0 step in cut)

Instrumented copy of `run_macro` (rdtsc around each section of `contact_steps`, `/tmp` harness, not part of the
package); removal's tooth pass split into its gating (`engaged_intervals`, contact geometry, sector box) and the
envelope subtraction.

| section | slot before | slot now | pocket before | pocket now | owner |
|---|---|---|---|---|---|
| removal tooth passes (`do_pass` -> `tooth_pass`) | 31.7 | 32.6 | 34.7 | 33.9 | removal |
| slab generation instants and pass test (`tooth_events` without the passes) | 13.8 | 0.2 | 12.2 | 0.1 | hot loop |
| engaged-interval refresh without a pass (`regate`) | 2.3 | 2.4 | 0.0 | 0.0 | removal (call: hot loop) |
| `cut_step` | 16.9 | 8.2 | 8.5 | 3.0 | cutting |
| pose: Hermite angles, axis interpolation, FK x 2, StepPose | 11.3 | 4.6 | 9.8 | 4.2 | motion FK, hot loop |
| refill planes (`refill_planes` + removal's `surface_plane`) | 9.5 | 4.2 | 4.1 | 0.8 | hot loop, removal |
| `clear_planes`, headroom, ring, scalars | 0.8 | 0.04 | 0.7 | 0.03 | hot loop |
| **run_macro per L0 step** (instrumented) | **87.5** | **50.5** | **70.7** | **42.5** | |
| run_macro per L0 step (plain build, test_performance) | 53-80 | 40-46 | 61-71 | 38-43 | |

Microseconds. "Before" is the state of 2026-11-04 measured with the same instrumentation on 2026-11-09 (the
documented 53 us / 61 us were taken on a quieter box). Removal's tooth pass costs 5.9 ms per pass in the slot
(5.5 passes per 1000 steps) and 11.3 ms in the pocket (3.0 per 1000 steps, lattice 0.075 mm): **it is now two
thirds to four fifths of the loop and alone 4.1 s/s (slot) and 6.8 s/s (pocket)**, 9-15 times the whole budget.
The architecture's claim that cut_step and the modal kernels were 94 % of the loop held for cutting's own
benchmark, not for the integrated loop: there the composition (refcounting, argument copies, the slab loop) and
removal's passes dominated.

## 2. Per kernel

| kernel | case | before | now | how |
|---|---|---|---|---|
| `cut_step` | slot, 27.4 engaged elements of 60 visited (15 engaged stations) | 16.9 us | 8.2 us (7.5 us cache-warm) | see section 5 |
| `cut_step` | pocket helical entry | 8.5 us | 3.0 us | |
| `cut_step` | cutting's benchmark: 4 x 80 elements, 160 engaged (`test_milling_kernel_cost`) | 48.6 us, 303 ns/elem | 33-36 us, 210-225 ns/elem | |
| `modal_begin` + `modal_correct` | slot, 8 modes (see section 1 note) | ~55 us (residual and all 119 stations) | see table 3 | engaged stations only |
| `tooth_events` (no pass) | slot | 13.8 us | 0.2 us | tooth windows, counts kept with the slab states |
| `refill_planes` + `clear_planes` | slot, 4.4 planes per step | 10.3 us | 4.2 us | lean arguments, RAW |
| `set_step_pose` | per step | ~2 us | ~0.1 us | no allocation, StepPose only |
| removal `tooth_pass` | slot / pocket | 5.9 / 11.3 ms per pass | unchanged | removal's kernel (open problem) |
| mechanics `continuation_reference` | per node of a map row without an admissible solution | 18-22 ms | 11-12 ms | the row's own solved nodes skip their t1 trials |
| mechanics `build_process_map` (lazy) | 108-node map (adiabatic) | 0.95 s eager | 0.09 s + refinement of the rows read | `lazy=True` |

## 3. Per level and whole runs

| run | simulated | whole run | hot loop in cut | L2 | notes |
|---|---|---|---|---|---|
| slot rigid, fast-L2 test project | 5.24 s (0.69 s in cut) | 3.2 s/s | 6.6 s/s (hot), 6.3 s/s run_macro | 1.1 s/s in the cut steps | before: 2.5-2.8 s/s, 10-11 s/s |
| pocket helical entry rigid, first 0.3 s of cut | 4.77 s | 1.6 s/s | 8.8 s/s | 1.2 s/s | before: 4.3-4.6 s/s, 14.5 s/s |

## 4. L2

* **Dynamics' stability analysis** (docs/models/dynamics.md 5.4): lobe diagrams and the semi-discretisation check
  are report artefacts and an input to nothing in the loop. During the run only the limit at the operating speed
  is computed (the bus `depth_limit_m`, the stability finding), once per engagement class and speed (cached by
  key in the checkpointed state); its chatter-frequency scan evaluates all frequencies' characteristic matrices at
  once and the cut model caches its per-tooth means (they were recomputed at every frequency: most of the ~200 s
  per engagement of the default grids). The diagrams (at most `sld_max_diagrams` per tool, smallest margins
  first) and the SD checks run once at the end of the operation.
* **Cutting's coefficient pipeline**: rows at the quantised slow state, recomputed only when its grid cell changes,
  equal cells from a cache; a tool that is not cutting while its spindle changes speed waits for the speed to
  settle. Run-up of 100 macro steps then 20 steady steps, mechanics model, default cadence (every macro step):
  D10 end mill 7.0 s -> 2.6 s CPU (rows computed 71 -> 6), D6.8 drill 211 s (task statement; 27 s measured with
  the old refresh rule on this box) -> 21 s. What remains is the first build of each row (continuations).
* **Mechanics' table building**: lazy process maps (a row's fallback refinement when a lookup first reads it) and
  continuations that do not re-solve the thickness trials next to the row's own solved nodes (section 2).

## 5. Hot loop composition (how the costs were reached)

### Reference counting

`run_macro`, `contact_steps`, `tooth_events`, the hot-kernel helpers and `cut_step` allocate nothing (scratch
vectors are views of `HotContact.work`) and are compiled with `_nrt=False` (`RAW`, as removal's hot kernels).
Numba increfs every member array of a NamedTuple taken from a larger one when it cannot prove the reference dead;
in a loop with branches that was ~80 atomic operations per access of a 40-array tuple (457 ns, measured).

### Lean arguments

A call that LLVM does not inline copies every member array descriptor of its arguments onto the stack (CutArgs:
~150 arrays, ~1000 words, ~1 us; `clear_planes(cut, ...)` cost 0.75 us for a few stores). The per-step helpers take
the arrays they read (`refill_planes`, `clear_planes`, `set_step_pose`, `slab_point`); the step's layouts are taken
from CutArgs once per call, not per step; `set_step_pose` and `rotvec` no longer allocate.

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
`CI_ST_MARGIN` = 2 + the most stations of one slab on each side, for the slab interpolation) and `modal_correct`
projects only the engaged stations' loads (every other station has none), the station sums of all modes side by
side (each mode's sum keeps its order: the same result, without the dependency chains); with the identity tool
mount of rotating tools cutting reads dynamics' deflection arrays in place (no copy).

## 6. Memory

| run | peak RSS |
|---|---|
| slot (40 x 30 x 10 mm stock, lattice 0.1 mm) | see test logs |
| pocket-islands (lattice 0.075 mm) | see test logs |

## 7. Compile time

Cold cache: see section "Compile time" below (measured with an empty NUMBA_CACHE_DIR).

## 8. Open problems

* **Removal's tooth pass** (5.9 ms per pass at lattice 0.1 mm, 11.3 ms at 0.075 mm: about 33 us per L0 step,
  4-7 s/s) keeps the loop 9-15 times above the 0.45 s/s budget and the whole pocket run above 1 s/s. Its cost
  scales with the rays of the engaged sector box (1 / lattice^2 per area, the full stock height of the box).
  Not in this task's ownership (cnctwin/removal).
* **Motion's FK** (`fk_poses`, two per L0 step: mid and end frames, ~2 us each with the error transforms of every
  link) is most of the 4.2-4.6 us pose section.
* **cut_step for the slot**: ~270 ns per engaged element (~1200 instructions at IPC ~1.3, register spills of the
  ~50 live frame and station constants, L2 misses on the memory bins); the 3 us target is met for the pocket's
  helical entry, not for the slot's 27 engaged elements (8 us).
