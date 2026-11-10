# GPU acceleration (`cnctwin/gpu/`)

Optional CUDA versions of three CPU kernels, chosen per call by the project switch `physics.gpu`, with an
automatic fall back to the CPU kernels whenever no usable device is present or a GPU call fails. The GPU
paths are numerical equivalents of the CPU kernels, not models of their own: they have no coefficients,
no database entries and no RNG streams, and they read and write no bus record (they work below the
domains, on the domains' own kernel layouts).

| GPU path | CPU kernel it replaces | parallel over | equality with the CPU kernel |
|---|---|---|---|
| dexel subtraction | `cnctwin.removal.k_subtract.subtract_envelope` | candidate rays | bit-identical (field arrays, accumulators, events, status) |
| thermal FV step | `cnctwin.thermal.kernels.work_step` | cells, lattice lines | bit-identical (grid, scratch, ledger terms) |
| particle flight | `cnctwin.chips.kernels.advance_particles` (K1) | particles | bit-identical in the simulator; on a device up to the libm/libdevice difference of `exp`, `expm1`, `pow` (section 2) |

| file | what |
|---|---|
| `schema.py` | `physics.gpu` (switch and `auto` thresholds): `settings`, `from_project`, `schema_node` |
| `device.py` | detection in a child process (`probe`, `info`, `available`), `exact_compile` (FMA off), launch helpers |
| `twins.py` | one source, two targets: CPU (`numba.njit`) and CUDA device twins of a function and everything it calls |
| `accel.py` | `Accelerator`: backend choice, fall back, usage record (`run_info` -> run.json `gpu`), `for_context` |
| `k_dexel.py` | candidates (`gather`), per-ray intervals (`k_rays`), commit through removal's own kernel |
| `k_thermal.py` | `k_cells`, `k_sweep_x/y/z`, `k_update`; the host `ledger` in the CPU kernel's summation order |
| `k_particles.py` | `flight` (K1 for one particle), `k_flight`, the ordered host `commit`, packing |
| `bench.py`, `cases.py` | CPU vs GPU benchmark on seeded synthetic cases (also the tests' inputs) |
| `ptxcheck.py` | compiles every kernel to PTX for sm_52 without a device, checks the FMA setting |
| `__main__.py` | `python -m cnctwin.gpu info | bench [smoke|default] | ptx [5.2] | warmup` |

## 1. How the GPU paths equal the CPU kernels

### 1.1 One source

Wherever a CPU kernel's work is per item, the GPU runs **the CPU kernel's own Python source**:
`twins.Builder("cuda")` compiles a domain's njit function (`py_func`) as a CUDA device function and
rebinds every global it calls to that callee's twin, recursively and across modules (a module global
whose attributes are called is replaced by a proxy module holding the twins). removal's envelope geometry
`_env_ray` (quadratics, the torus quartic, root isolation, interval merge), thermal's `prop_interp`,
`face_fraction`, `face_conductances`, `coolant_h`, `temperature`, and chips' `propagate`, `phi12`,
`drag_rate`, `nusselt_h`, `cp_at`, `enthalpy`, `impulse`, `cyl_hit`, `sph_hit`, `box_exit`, `tick_index`,
`lerp3`, `lerp1` reach the GPU this way, so a change in a domain's per-item arithmetic reaches the GPU path
without anyone porting it. Functions of this package written for both targets are plain Python tagged
`@twin` and compiled with `numba.njit` (CPU, cached) or as device functions from the same source (the
particle flight, the commits' helpers).

NumPy scalar functions that numba-cuda does not lower (`np.floor`, `np.ceil`, `np.trunc`, `np.sqrt`,
`np.fabs`, `np.isnan`, `np.isinf`, `np.isfinite`) are replaced inside device code by libdevice twins; all
are exact (IEEE), so values are unchanged.

### 1.2 Exact arithmetic

Bit-identity needs the device to round every operation as the host does:

* **FMA contraction.** NVVM fuses `a * b + c` into one fused multiply-add by default (`-fma=1`), which rounds
  once where the CPU rounds twice. numba-cuda passes no option for it, so `device.exact_compile()` compiles
  this package's kernels with `-fma=0` (it wraps `numba.cuda.cudadrv.nvvm.compile_ir` while they are
  compiled, under a lock). Verified offline by `ptxcheck.fma_hook_works` (a pure `x * y + z` kernel has one
  `fma.rn.f64` without the setting and none with it) and at run time by the device probe (section 4).
* **Division and square root** are IEEE correctly rounded on both sides (`fastmath` off: NVVM
  `-prec-div=1 -prec-sqrt=1`; x86-64 `divsd`, `sqrtsd`); `floor`, comparisons and integer arithmetic are
  exact. Numba on the CPU does not contract either (no `contract` flags without fastmath).
* **Transcendental functions** (`exp`, `expm1`, `pow`) are not correctly rounded on either side and the
  implementations differ (glibc vs libdevice); only the particle flight uses them (section 2).
* **Summation order.** Sums over items are never reduced on the device: each GPU path returns per-item
  results and the host adds them in the CPU kernel's order.

### 1.3 Dexel subtraction

The CPU kernel walks the rays of the engaged-sector box level by level, family by family, (iu, iv) in
order. Per ray: quick rejects, the exact envelope intervals, quantisation to 10 nm, subtraction from the
ray's segments (int32 endpoints, K = 4 inline plus the overflow pool, architecture 5.5 and 10), the write
back with pool blocks from a shared free list, and the piece records that `post_pieces` books into bricks,
target checks (gouge, breakthrough), the non-cutting centre and the contact's sums. Only the interval
geometry is independent per ray. The GPU path:

1. `gather` (CPU): the candidate rays in the CPU kernel's order, with its rejects (no segments, farther than
   the quick-reject radius from the axis, no material in the box) and the ray origins computed with the CPU
   kernel's own expressions.
2. `k_rays` (GPU, one thread per candidate): removal's `_env_ray` as a device twin, merged as the CPU kernel
   asks for it. More than `MAX_RAW` = 64 raw or `MAX_OUT` = 16 merged intervals: the ray is marked SPILL.
3. Commit (CPU): **removal's own `subtract_envelope`**, compiled from its current source with one global
   rebound, `_env_ray` -> `env_ray_lookup`. The lookup returns the GPU's intervals of the ray it is asked
   about, matched exactly on (family, origin, range), searching forward from the last match; a ray it has
   no result for (spilled, or one `gather` did not foresee) is computed there with the real `_env_ray` and
   the CPU kernel's own capacity (`MAX_IV`). The table (4 key rows per candidate, then the intervals packed,
   built by compiled code) reaches it through two extra regions of the scratch arrays `ivt` / `ivp`, which
   only `_env_ray` uses; the lookup is inlined like removal's own helpers (a call with 13 array arguments
   would cost about as much as the geometry it saves).

Everything after the geometry is therefore the CPU kernel itself (pool, bricks, `post_pieces` and its flush
points, events, status), whatever removal changes there. Consequences: if `gather`'s rejects drift from
removal's, results stay exact and only the speed changes (the call's stats count the rays the host
computed: `host_items`); if removal stops calling `_env_ray` by that name, `commit_kernel` raises GpuError
and the accelerator falls back to the CPU (test `test_lookup_survives_a_gather_that_drifted`). The commit
clone is compiled once per process and not cached on disk (it shares its source file and line with
removal's cached kernel).

### 1.4 Thermal finite-volume step

`work_step` is a locally one-dimensional backward-Euler splitting (thermal.md). The GPU runs:

| kernel | threads | CPU kernel part |
|---|---|---|
| `k_cells` | one per cell | pass 0: capacity, half-cell resistances (float32 storage), exposure bits |
| `k_sweep_x` | one per line (j, k) | x sub-step: forward elimination with the faces' implicit conductances, back substitution |
| `k_sweep_y` | one per line (i, k) | y sub-step |
| `k_sweep_z` | one per line (i, j) | z sub-step |
| `k_update` | one per cell | h += C dT / m, T = T(h) |

Each line is solved in the CPU kernel's operation order, so every cell value is bit-identical. The ledger
terms (`L_SRC` ... `L_LOST`, Neumaier-compensated where the CPU compensates) are sums over cells whose
order the CPU kernel fixes (for the losses: x sub-step j up, i down, k up; y sub-step i up, j down; z
sub-step k down): the host recomputes them in exactly that order from the GPU's per-cell results with
thermal's own njit helpers (`k_thermal.ledger`, about a third of the CPU kernel's face work plus one pass
of sums). `WorkScratch.line_c`, `line_d` and `gprev` are not written by the GPU path (the CPU kernel
leaves the last line's coefficients there; nothing reads them). The kernel's inputs (`g.h`, `g.T`,
`s.src`) change only after every kernel and transfer succeeded; the scratch arrays the CPU kernel only
writes (`cap`, `rx`, `ry`, `rz`, `bits`, `a`) are filled straight from the device.

### 1.5 Particle flight

K1 advances the particles one after the other; its only couplings between particles are shared state:
the heat and count accumulators, the event buffer, and the pile, which a particle that settles raises
under the particles after it. The GPU path keeps K1's per-particle arithmetic and moves the couplings to
an ordered host commit:

* `flight(i, ...)` is K1's loop body for particle i, operation for operation (jets and the wall film,
  Haider-Levenspiel drag, the exact exponential integrator, cooling and radiation, contacts with the
  enclosure, the pile height field (DDA), rotating cylinders, nozzles; welding, wrapping, hooking), against
  the pile as it was at the start of the call. Instead of touching shared state it records the heat
  addends of every sub-step (a log of up to `CAP_LOG` = 128 pairs), its sub-step and hit counts, at most
  one event (K1 writes at most one per particle per call), the terminal action (settle into a cell, or the
  pan) and the box of pile cells it read.
* `k_flight`: one GPU thread per particle runs `flight`.
* `commit` (host, the CPU twin): in particle order, writes the particle back, replays its heat addends
  into `out` (K1's order of additions), adds its counts (exact integers), writes its event and applies its
  settle / pan with chips' own `settle` / `to_pan`. A particle whose read box contains a cell an earlier
  particle of this call settled into, or whose log overflowed, is recomputed right there by the CPU twin
  of `flight` against the live pile, which is exactly K1's sequential semantics (test
  `test_pile_coupling_is_resolved_in_particle_order` with a negative control showing the frozen-pile
  result differs). The read box is conservative (cells inside it that the particle never read count as
  read): how many particles the host recomputes depends on how many settle per call; measured on the
  synthetic case (24 x 20 pile cells): 94 of 256, 519 of 1024, 2780 of 4096 particles, with 28, 117 and
  471 settles. Dense settling in one window therefore moves most of the work back to the host (results
  stay exact).

## 2. Validity range and tolerances (how they were established)

**Validity range.** The equalities below hold for CUDA devices of compute capability 5.2 (sm_52) and
newer with double-precision arithmetic, kernels compiled through `device.exact_compile` (`-fma=0`,
`-prec-div=1`, `-prec-sqrt=1`) with numba-cuda 0.30 and the CUDA 12.x toolkit (12.9 checked offline; a
driver of at least 12.2 through nvJitLink), and inputs inside the CPU kernels' own layouts: dexel rays of
up to `MAX_RAW` = 64 raw and `MAX_OUT` = 16 merged intervals on the device (more: the host computes the ray),
particles with up to `CAP_LOG` = 128 heat addends per call (more: the host recomputes the particle). Outside
that range nothing changes in the results: the host computes the item or the whole call (section 5); only
the speed does. Not covered: devices on which the probe's arithmetic check fails (no GPU path runs there),
CUDA 13 toolkits (no sm_52), and a run on real hardware has not been made here (section 4).

* **Dexel and thermal: bit-identical** on any device whose kernels are compiled with `exact_compile`
  (they use only +, -, *, /, sqrt, floor, comparisons and integer operations). Established by the parity
  tests in the CUDA simulator (every array of the field / grid / scratch and every accumulator and event
  compared with `np.array_equal`, signed zeros and NaN included, over torus and ball envelopes with
  refine levels, a target with gouge and breakthrough, plunge dead zones, rapids, pool overflow, spilled
  rays, a drifted candidate list; cut cells, fixture contact, coolant zones and the wetted grid, repeated
  steps) and by the offline PTX compile for sm_52 with `-fma=0` (section 7).
* **Particles: bit-identical in the simulator** (Python uses the host's glibc for `exp`, `expm1`, `pow`,
  as Numba's CPU code does). On a device libdevice's double-precision `exp` and `expm1` are within 1 ulp and
  `pow` within 2 ulp of the exact result (CUDA C++ Programming Guide, "Mathematical Functions",
  double-precision table), glibc's within about 0.5 ulp (Szabolcs Nagy's 2018 implementations; glibc
  manual "Errors in Math Functions"), so one call may differ by up to about 2.5 ulp (relative 5.6e-16).
  Each sub-step makes a fixed number of such calls (drag `Re ** b`, `phi12` via `expm1`, cooling `exp`,
  the jet profile `exp`, Ranz-Marshall `Re ** m Pr ** n`), so position, velocity and temperature differ
  by a few times 1e-16 relative per sub-step and the differences add up roughly linearly over the window
  (the drag makes the flight contractive, nothing amplifies them) until a discrete decision (which surface
  is hit first, settle or bounce) falls the other way for a particle that grazes a threshold; that
  particle then takes the other branch, as it would for a 1e-16 change of its initial state on the CPU.
  The real-device test asserts identical statuses and positions, velocities, temperatures within 1e-9
  relative. Not measured on hardware here (no device, section 4).

## 3. The switch: `physics.gpu`

```yaml
physics:
  gpu: auto                  # auto | on | off   (true/false read as on/off)
  # or: gpu: {mode: auto, device: 0, min_rays: 2048, min_slabs: 24, min_cells: 32768, min_particles: 1024}
```

* `off`: CPU kernels only; the device is never probed.
* `on`: the GPU whenever a CUDA device, or with `NUMBA_ENABLE_CUDASIM=1` the simulator, is usable, at any
  size; without one the CPU kernels run and the reason is recorded. A run never fails for want of an
  accelerator.
* `auto` (default): the GPU on a real CUDA device for calls of at least `min_rays` candidate rays and
  `min_slabs` envelope slabs (dexel; rays counted as those of the subtraction's node ranges), `min_cells`
  cells (thermal), `min_particles` particles in use (chips); never the simulator.
* `CNCTWIN_GPU=off` in the environment disables the GPU path without probing (hosts, CI).

Defaults of the thresholds (numerical settings, not model coefficients): the size above which a GTX
950-class card (sm_52, 6 SMs, 2 GB, FP64 at 1/32 of FP32, about 55 GFLOP/s; PCIe 3.0 x16, about 12 GB/s)
is expected to beat one core of this box, from the launch latency (about 10 us per kernel and per
transfer), the bytes moved per item (dexel 88 B in and 264 B out per candidate; thermal about 100 B per
cell; particles about 2.5 KB) and the measured CPU cost per item and host share of each path (section 6):
`min_slabs` 24 because the dexel path's host part costs as much as the whole CPU kernel up to about 12
slabs. They are estimates until `python -m cnctwin.gpu bench default` has run on a real device; tune
them from it.

### 3.1 Parameters and settings

No model coefficients (no `db/models` file): the GPU paths compute what the CPU kernels compute. The numerical
settings and constants that decide where a call runs:

| name | value | source |
|---|---|---|
| `physics.gpu.mode` | auto | this package's choice: the GPU only on a real device and above the thresholds |
| `device` | 0 | CUDA_VISIBLE_DEVICES order |
| `min_rays` | 2048 | est: launch latency (~10 us per kernel and transfer) and bytes per candidate (88 B in, 264 B out) at ~12 GB/s against 0.31 + 0.022 x slabs us per candidate on the CPU (section 6) |
| `min_slabs` | 24 | measured: the dexel path's host part costs as much as the whole CPU kernel up to about 12 slabs on this box; at 24 its host share is about 0.7 (0.58 us against 0.31 + 0.022 x 24 = 0.84 us per candidate, section 6) |
| `min_cells` | 32768 | est: ~100 B per cell moved, 160 ns per cell on the CPU, host share 0.28 (section 6) |
| `min_particles` | 1024 | est: ~2.5 KB per particle moved, 38.7 us per particle on the CPU, host share 0.21 (section 6) |
| `MAX_RAW`, `MAX_OUT` (`k_dexel.py`) | 64, 16 | a thread's own buffers (k_rays keeps its 64-interval buffer in local memory, section 7); a ray beyond them is computed on the host with removal's capacity (`MAX_IV` = 512) |
| `CAP_LOG` (`k_particles.py`) | 128 | heat addends a thread logs; more: host recompute |
| `PROBE_TIMEOUT_S` (`device.py`) | 120 s | the probe's compile and launch take 1-10 s |
| PTX check target (`ptxcheck.py`) | sm_52, 900 s timeout | the GTX 950 of this host (section 4) |

The thresholds are estimates until `python -m cnctwin.gpu bench default` has run on a real device.

## 4. Device detection

`device.probe(index)` runs `python -I -c PROBE_CODE` in a child process (a broken driver can crash or hang
a process; the simulation process never touches the driver before the probe said yes) and returns a
`DeviceInfo`; `info()` caches it per process and device; `available() -> (bool, reason)`. Steps, each with
its reason when it fails: numba-cuda importable; the CUDA driver initialises (on Linux the reason also
says when `/dev/nvidiactl` is missing: the kernel module is not loaded); the device index exists and
opens; the toolkit compiles for its compute capability (`nvrtc.get_supported_ccs`); NVVM loads; a test
kernel compiles with `-fma=0`, runs, and its double-precision results (`x y - z` with `x = y = 1 + 2^-30`,
`(1 + 2^-27)(1 - 2^-27) - 1`, a square root, a division, a sum) equal the host's bit for bit. A timeout
(120 s) or a crash of the child is a reason too. `probe(simulate=True)` runs the same steps in the
simulator (its tests).

This host (2026-11-03): GeForce GTX 950 (GM206, sm_52) with the 535 driver libraries installed but the
kernel module not loaded: `CUDA_ERROR_NO_DEVICE; the NVIDIA kernel driver is not loaded (/dev/nvidiactl is
missing)`. Every call therefore runs on the CPU (and the GPU tests in the simulator).

Installation: `pip install 'numba-cuda[cu12]'` (numba-cuda 0.30.4 with the CUDA 12.9 toolkit wheels:
NVVM, NVRTC, runtime, nvJitLink; the plain `numba-cuda` wheel pulls CUDA 13 bindings, whose simulator
import already needs a CUDA runtime, and CUDA 13 no longer compiles for sm_52). The 535 driver supports
CUDA 12.2: the 12.9 NVVM emits PTX ISA 8.8, which that driver cannot JIT itself; numba-cuda links through
nvJitLink to a cubin for sm_52, which a 12.x driver loads (CUDA minor-version compatibility). If the probe
reports a PTX or link error on such a host, install the toolkit matching the driver
(`cuda-toolkit[nvcc,nvrtc,cudart]==12.2.*`) or a driver of at least the toolkit's version.

## 5. Fall back and the run record

`Accelerator` (one per run: `cnctwin.gpu.for_context(ctx)`, from `ctx.project.physics.gpu`) has the three CPU
kernels' signatures. A GPU path changes the caller's arrays only after its device phase succeeded, so on
any GpuError (CUDA, NVVM, driver, a GPU module that no longer imports because a mirrored domain kernel
changed shape) the CPU kernel runs on untouched inputs; after one failure the accelerator stays on the
CPU for the rest of the run (no repeated failures) and records why. `run_info()` is run.json's `gpu`
entry (no wall times):

```
{mode, device_index, thresholds {min_rays, min_slabs, min_cells, min_particles},
 device {kind none|cuda|simulator, usable, reason, index, name, compute_capability, driver_version,
         runtime_version, nvvm_version, memory_bytes, multiprocessors, numba_cuda_version,
         fma_contraction, exact_arithmetic, supported_ccs},
 kernels {dexel_subtract|thermal_work_step|chips_advance: {gpu_calls, cpu_calls, gpu_items, cpu_items,
          host_items, fallbacks, last_backend, last_reason, parity}},
 disabled_reason, errors [<= 20]}
```

`host_items` counts the items of GPU calls the host computed (spilled or unforeseen rays, recomputed
particles). Same project, build and hardware give the same record; the synthetic run directory
(`tests/fixtures/make_synthetic_run.py`, `gpu_record`) carries a CPU-only one.

## 6. Benchmark and measured costs

`cnctwin.gpu.benchmark(sizes="default" | "smoke" | {...}, repeat=3)` times each CPU kernel and, when a
CUDA device is usable, its GPU path on the same seeded inputs (best of `repeat` on fresh copies; the first
GPU call, which compiles, separately), checks that both give the same result (`identical`,
`max_rel_diff`) and otherwise says why there are CPU times only. The simulator is timed only on request
(`include_simulator=True`): its times are Python execution. Reference sizes (`default`): dexel 20 x 16 x
10 mm block at 0.1 mm with a 2:1 refine box, a tilted bull-nose of radius 4.8 mm, 8 subtractions; thermal
96 x 64 x 32 cells of 1 mm; chips 4096 particles.

Measured on this host (one core of the 4-core box, nice 10 under bin/tsafe, other jobs running:
about +-20 %), CPU kernels and the host part of each GPU path on the same inputs, no device:

| path | case | CPU kernel | host part of the GPU path | host share (bounds the speed-up) |
|---|---|---|---|---|
| thermal | 96 x 64 x 32 cells of 1 mm (196 608), pocket, coolant, fixture | 160 ns per cell | ledger 44 ns per cell, plus copies | 0.28: at most 3.6x |
| chips | 4096 parcels, 10 ms window | 38.7 us per particle | commit 8.2 us per particle (2780 recomputed: dense settling), packing 0.08 us | 0.21: at most 4.7x here |
| dexel, 3 slabs | 20 x 16 x 10 mm at 0.1 mm, refine box, tilted bull-nose R 4.8 mm, 3 cuts (131 013 candidates) | 0.38 us per candidate | 0.58 us per candidate | 1.53: slower than the CPU |
| dexel, 12 slabs | same (131 332 candidates) | 0.58 us | 0.57 us | 0.98: break-even |
| dexel, 48 slabs | same (131 584 candidates) | 1.35 us | 0.61 us | 0.45: at most 2.2x |

The dexel numbers deserve a word. The GPU path's host part (gather 0.03-0.05 us, table 0.03-0.05 us,
the commit) is about constant per candidate, while the CPU kernel's geometry grows with the slabs
(fit: 0.31 + 0.022 x slabs us per candidate). Most of the commit's cost is not the bookkeeping itself but
a Numba effect measured here: removal's kernel compiled when its module is imported has 31 NRT reference
count calls left after Numba's refcount pruning, while the same function compiled any time later in the
process, including a plain copy with no change at all, a deep copy with every callee recompiled, fresh
code objects, cache on or off, has 917 and runs 2.4 times slower (0.077 s vs 0.187 s for the 3-cut,
12-slab case). The commit must be such a later compile (it rebinds `_env_ray`), so it inherits that
factor; if Numba's pruning of later compiles is fixed, the host part drops to about 0.25 us per candidate
and the dexel path pays from a few slabs on. Hence `min_slabs` (section 3).

## 7. Checks without a device

`ptxcheck.check((5, 2), groups)` compiles every kernel to PTX for sm_52 with the installed NVVM in a child
process without the simulator (numba-cuda asks the current device for its compute capability when it
compiles device functions; the check answers for it), with `exact_compile`. Result on this host: all 7
kernels compile for sm_52 (k_rays 4.6k PTX lines, uses local memory for its 64-interval buffer; k_flight
7.8k lines, no local memory; the thermal kernels 0.5k-1.4k lines) and the FMA setting works
(`fma_default` 1, `fma_exact` 0). Not checked here: loading and running them on a device (needs the
kernel driver).

## 8. Tests (`tests/unit/gpu/`)

`NUMBA_ENABLE_CUDASIM=1 bin/tsafe python -m pytest -q tests/unit/gpu` (the conftest selects the simulator
by itself when no device is usable; on a real device the same tests run on it). Parity:
`test_dexel_parity.py`, `test_thermal_parity.py`, `test_particles_parity.py`; detection and settings:
`test_device.py`, `test_settings.py`; twins: `test_twins.py`; dispatch and fall back: `test_accel.py`;
benchmark: `test_bench.py`; offline compile: `test_ptx_thermal_dexel.py`, `test_ptx_particles.py`; run record:
`test_run_record.py`. With a warm Numba cache a file takes about 1 to 45 s (the simulator executes the
kernels as Python; the PTX checks compile with NVVM); the cache directory is keyed by every kernel source
of the project (bin/tsafe), so after any kernel file changes the first run recompiles the domains' kernels
it imports (minutes on this box while other jobs run).

## 9. Assumptions, limitations and their consequences

* **One device per process** (`physics.gpu.device`); kernels are compiled for it on first use (a few
  seconds each; `Accelerator.warmup()` compiles them up front). CUDA kernels are not cached on disk
  (consequence: the warm-up repeats per process; avoids a stale cache keyed by NVVM options numba-cuda
  does not record).
* **Transfers per call**: the GPU paths copy their inputs to the device and the results back on every call
  (the domains' arrays live on the host, where their other kernels run); the host shares of section 6
  bound the speed-up (Amdahl).
* **Calls from inside the compiled hot loop** (`core.hotloop.run_macro`) cannot launch CUDA kernels: the
  dexel path serves calls made from Python (batched subtractions at L2, replays, rebuilds); a subtraction
  inside the hot loop stays on the CPU unless the integration splits the loop there.
* **The thermal and particle paths mirror** their CPU kernels' loop structure (the per-cell and
  per-particle helpers are the domains' own code); the dexel commit is removal's code itself. A change in
  `work_step`'s sweep structure or in K1's loop body must be mirrored in `k_thermal` / `k_particles`; the
  parity tests fail until it is.
* **Particle pile conflicts are found by a box** of the cells a particle read: conservative (a large
  trajectory box can trigger a needless host recompute: speed, not results).
* **The dexel path only pays with many slabs** (section 6): below about 12 slabs its host part alone costs
  as much as the CPU kernel on this box, because the commit is a later compile of removal's kernel that
  Numba prunes worse (917 vs 31 refcount calls). Consequence: `auto` leaves envelopes under `min_slabs`
  on the CPU; `on` still uses the GPU path (identical results).
* **Dense settling** moves the particle path's work back to the host (section 1.5).
* **auto thresholds are estimates** (section 3) until benchmarked on a device.

## 10. What the integration has to wire

* Project schema: `PHYSICS` in `cnctwin/project/schema.py` needs `"gpu": cnctwin.gpu.schema.schema_node()`
  (the Struct is strict); the template `db/_templates/project.yaml` may show `gpu: auto` under `physics`.
* Domains: call the accelerator where they call the kernels, with the same arguments:
  `gpu.for_context(ctx).work_step(...)` (thermal `workpiece.fv_step`),
  `gpu.for_context(ctx).advance_particles(...)` (chips `domain.macro_step`),
  `gpu.for_context(ctx).subtract_envelope(...)` (removal, for subtractions made from Python).
* run.json: `run["gpu"] = gpu.for_context(ctx).run_info()` at the end of the run.
* Job runner / image: `python -m cnctwin.gpu warmup` (or `Accelerator.warmup()`) when a device is present;
  `CNCTWIN_GPU=off` where the probe should not run.

## References

* NVIDIA, *CUDA C++ Programming Guide*, Appendix "Mathematical Functions" (double-precision maximum ulp
  errors) and "Floating Point Standard" (FMA contraction, `-fmad`), CUDA 12.9, 2025.
* NVIDIA, *libNVVM API*, `nvvmCompileProgram` options (`-fma`, `-prec-div`, `-prec-sqrt`), CUDA 12.9, 2025.
* The GNU C Library manual, "Errors in Math Functions" (glibc 2.39), and S. Nagy, new `exp`/`log`/`pow`
  implementations, glibc 2.28 (2018).
* IEEE 754-2019, *Standard for Floating-Point Arithmetic* (correct rounding of +, -, *, /, sqrt, fma).
* Numba and numba-cuda documentation 0.68 / 0.30: CUDA simulator (`NUMBA_ENABLE_CUDASIM`), kernel and device
  function compilation.
