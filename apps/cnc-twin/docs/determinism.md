# Determinism

(owner: the run infrastructure, `cnctwin/core/rundir.py`, `checkpoint.py`, `recorder.py`, `keyframes.py`,
`bin/twin`; architecture 9)

**Same project + same build + same hardware = bit-identical result set**, whether the run goes through
in one process or is stopped and resumed from a checkpoint in another, and whatever the process's hash
seed or thread variables. Across platforms (another CPU, C library, NumPy or Numba build) results agree
within the tolerances below, not bit for bit.

## 1. What is bit-identical, and how it is hashed

The **result set** (architecture 9) is every file of `findings.json`, `trace.jsonl`, `series/`,
`frames.bin`, `keyframes/`, `part_*`, `inspection.json`, `surface.json`, `tools.json` and
`production.json` (dot files, i.e. temporary files, excluded). `cnctwin.core.rundir.result_set_digest(run_dir)`
returns `{relative path: sha256}`; two runs are bit-identical when these dictionaries are equal.

Not in the result set, and why:

* `status.json`: volatile by design (progress, wall time, ETA). Everything that depends on wall time
  lives here and nowhere else.
* `run.json`: deterministic (no wall time, no dates), but it records the environment (`project_dir`,
  `platform`, `threads`, `numba_cache_key`) so a run can be reproduced; two machines legitimately write
  different run.json files for identical results.
* `checkpoints/`: deterministic for the same run, but which ones exist depends on how the run was split
  (retention keeps the last two); they are a means to the result set, not part of it.
* reports, PDF, SVG, images, video, audio: their owners write them with fixed metadata
  (`SOURCE_DATE_EPOCH`, Matplotlib `CreationDate: None`, ffmpeg `-fflags +bitexact`); they are
  derived from the result set.

**How it is established.** `tests/unit/core/test_checkpoint.py::test_resume_in_a_fresh_process_is_bit_identical`
runs a toy run of three domains (counter-based draws, growing arrays, a NumPy view and an alias in a
domain state, fresh bus records each step, a timeline, trace events, findings, the fast ring with
engagement and finding windows, all three series tiers, frames.bin, keyframes with full snapshots and
an emptied brick, the ledger, checkpoints) for 60 macro steps in one process, and again in a process that
is killed (`os._exit`, nothing flushed) three steps after its checkpoint at step 20 and a fresh process
that resumes from it: the 45 files of the result set, `timeline.json`, `channels.json` and the final
state are identical. The same toy run with `PYTHONHASHSEED=1`, `OMP_NUM_THREADS=4` and an empty Numba
cache gives the same 45 digests as with the settings of `bin/tsafe` (measured 2026-11-03 on this box).

## 2. Sources of non-determinism and how each is removed

| source | rule (where it is enforced) |
|---|---|
| random numbers | no stateful generator in the loop: every draw is `hash_uniform(seed, stream_key(name), a, b, c)` with counters in the owner's state (core/rng.py, architecture 9); stream keys are CRC-32 of the name, the same on every platform. Generators (`stream`) only at initialisation and in outer studies. |
| hash randomisation, set order | `PYTHONHASHSEED=0` (bin/twin, bin/tsafe, the job runner, the Dockerfile; recorded in run.json) and no result may depend on the iteration order of a set or of a dict not built in sorted insertion order. The infrastructure sorts every set before it iterates (keyframe bricks, file lists). |
| threads | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS` = 1 (same places): a threaded BLAS or OpenMP reduction splits sums by thread count. No `parallel=True` reductions; `prange` only with disjoint writes. |
| compiled code | every kernel `@njit` with one explicit signature, `cache=True`, `error_model="numpy"`, no `fastmath` (so LLVM may not reassociate, contract into FMA or assume no NaN), reductions written as loops in a fixed order (architecture 5.5). |
| stale compiled code | Numba's cache does not notice a changed callee in another file: `NUMBA_CACHE_DIR` is keyed by sha256 over the `sha256sum` lines of every kernel source, the numba/llvmlite versions and the CPU model (`bin/twin` and `bin/tsafe` compute it in bash, `cnctwin.core.rundir.numba_cache_key()` and the API's job runner in Python: a test checks they agree). run.json records the key. |
| code and libraries | `cnctwin.core.rundir.build_id()`: sha256 over every `cnctwin/**/*.py` (path and content) and the Python, NumPy, SciPy, Numba, llvmlite and PyArrow versions. A checkpoint records it and is refused by another build (CheckpointError): a continuation under other code could not be bit-identical. |
| iterative solvers | deterministic start vector `ones/sqrt(n)`, eigenvectors sign-normalised, degenerate clusters rotated to a canonical basis (architecture 9; their owners). |
| wall clock | only status.json; zip members are dated 1980-01-01 (`npz_bytes`), GLB/JSON carry no dates, Parquet parts have no timestamps; row groups are cut by row count, never by time. |
| file system order | every listing the run reads is sorted; JSON keeps insertion order (deterministic) or is sorted where the format says so (GLB JSON, index lines). |
| float formatting | JSON numbers are Python's shortest round-trip `repr`; non-finite floats become null in JSON (findings, run.json), and keep their exact bits in checkpoints (`{"$f": <bits>}`). |
| containers | `.npz` written member by member with fixed dates and sorted names, GLB with sorted keys and fixed separators, Parquet by PyArrow with zstd: the same bytes for the same input **and the same library versions** (zlib, zstd and Parquet writers may change their output between versions; the build id covers PyArrow, the zlib of the Python build is part of "same build"). |
| resume | a checkpoint is a consistent cut between two macro steps: the whole SimState (bus_prev and bus_next included), every participant's state (the recorder's ring and windows, the keyframe index, the timeline, the ledger's history), the byte lengths of every append-only file; resume truncates everything after the cut. Floats keep their bits, NumPy scalars their type, shared objects and arrays stay shared and views stay views of their base, so in-place updates after the resume reach the same places. |
| environment | run.json records versions, thread variables, numba cache key, platform; the job runner also fixes `TZ=UTC`, `LC_ALL=C.UTF-8`, `SOURCE_DATE_EPOCH`. |

Known limits of the cut (documented consequences, not silent): an array of a domain state that was
not C-contiguous is restored C-contiguous (`Restored.converted_to_c` counts them): its values are
identical, but a later NumPy reduction over a multi-dimensional array can sum in another order and
round differently in the last bit, so domain states keep C-contiguous arrays, as the kernel ABI already
requires. An array viewing memory the state does not otherwise hold (`as_strided` over a foreign buffer)
is restored as a copy. Instance attributes listed in a class's `__checkpoint_exclude__` are not stored:
they must be caches the owner rebuilds.

## 3. Across platforms: tolerances, and why

Bit identity across platforms is not promised, because three things below the simulator legitimately
differ:

* **transcendental functions**: Numba calls the C library's `sin`, `exp`, `log`, `pow`; glibc lists errors
  of up to about one ulp for them (its manual's table of known maximum errors), other C libraries round
  some arguments differently. NumPy dispatches SIMD implementations by CPU at run time (on AVX-512 CPUs,
  SVML-based float64 `exp`, `log`, `sin`, `cos`), which can differ in the last bits from the scalar ones.
  Measured on this box (Intel i5-4690, AVX2, no AVX-512): NumPy, glibc through Python's `math` and Numba
  give identical results (0 ulp, 10^6 arguments each) for sin, exp, log and x^0.37.
* **dense linear algebra**: OpenBLAS/MKL pick kernels and blocking per CPU, so a matrix product or a
  factorisation can round differently (single-threaded everywhere, but not the same kernel).
* **code generation**: Numba compiles for the host CPU; without fast-math LLVM keeps IEEE semantics
  (results of element-wise code do not change), but a different CPU can still change library calls.

Each operation then differs by at most a few ulp (2.2e-16 relative each). What that does to a result
depends on the dynamics it passes through:

| quantity | tolerance across platforms | why |
|---|---|---|
| integer time (block times, cycle time, timeline) | identical structure; a block time within one L1 tick | integer arithmetic; only a float-decided event (a probe trigger, an in-position test) can move by one tick when its value lies within a few ulp of the threshold |
| ledger residuals | below `tol_mass` / `tol_energy` on every platform | they are checks, computed in each run |
| stable (damped) regimes: forces, positions, temperatures, wear, volumes | relative 1e-9 of the channel's range | errors of a few ulp per operation do not grow in a damped system; measured: a 1-ulp change of the initial state of a damped forced oscillator (Duffing, gamma 0.1) stays within 2e-16 of the range over 380 forcing periods; summed rounding over the up to 10^9 operations of a long run grows like sqrt(N) ulp (about 1e-11), and 1e-9 leaves two orders of margin for ill-conditioned steps |
| unstable or chaotic regimes (chatter above its threshold, chip flight, bird-nesting) | statistics only: RMS within 2 %, dominant frequency within one FFT bin, findings' categories and lines the same | a perturbation grows exponentially until it reaches the size of the attractor: measured, the chaotic forced Duffing oscillator (gamma 0.5) differs by 0.1 % of its range 28 forcing periods after a 1-ulp change, and the RMS of the two runs over the last 190 forcing periods differs by 1.4 % |
| findings | same categories, severities and lines; times within one macro step | unless the deciding quantity lies within its stated model error of the rule's threshold: then either outcome is a correct prediction |
| frames.bin | within one float32 ulp (6e-8 m at 0.5 m, 4.8e-7 rad at 2 pi) | float64 sources within a few ulp round to the same or adjacent float32 values |
| keyframe vertices | within one quantisation step (0.12 um for an 8 mm brick) | adjacent quantisation steps |
| removed volume (10 nm fixed-point dexel endpoints) | relative 1e-9 | an endpoint flips by one quantum only where a float position lies within a few ulp of a quantum boundary |
| compressed files (Parquet, npz, GLB) | compare decoded values, not bytes | other zstd/zlib versions may encode the same values differently |

To compare two runs across platforms, read both result sets (`cnctwin.core.recorder.iter_series`,
`read_frames`, `KeyframeReader`, findings.json) and compare values with these tolerances; within one
platform and build, compare `result_set_digest`.

## 4. How to keep it

* Run through `bin/twin` (or `bin/tsafe` for tests); in containers the image sets the same variables.
* Draw random numbers only with `hash_uniform` and counters in the domain's state; record a stream name
  in the domain's model doc.
* Keep domain state as data (dataclasses, NumPy arrays, lists, dicts): the checkpoint codec refuses
  functions, open files, generators and local classes by name, and classes outside `cnctwin`.
* Give each macro step fresh bus records; write only your own record (architecture 7).
* Never iterate a set to produce a result; sort it.
* Never put wall time in anything but status.json.

## References

* IEEE 754-2019, Standard for Floating-Point Arithmetic (binary16/32/64 precision, rounding).
* The GNU C Library Reference Manual, "Errors in Math Functions" (known maximum errors per function).
* Steele, G.L., Lea, D., Flood, C.H. (2014), Fast splittable pseudorandom number generators, OOPSLA 2014,
  doi:10.1145/2660193.2660195 (the SplitMix64 finaliser of `hash_uniform`).
* Jugel, U. et al. (2014), M4: A Visualization-Oriented Time Series Data Aggregation, PVLDB 7(10) 797-808
  (min/max-preserving downsampling of the series reader).
* docs/architecture.md sections 5.5, 9, 10, 16; docs/formats.md "Run directory".
