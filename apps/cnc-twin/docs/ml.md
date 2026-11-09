# Training data from the twin: run cache, sweeps, datasets

The twin does not have to run in real time: a run is computed once, kept in the **run cache** and played back
(the UI, the API and the reports read only its run directory). That makes it a deterministic, scriptable data
generator: a **sweep** runs many variants of a project (feeds, speeds, overrides, stick-outs, programs written
by a generator ...) through the cache, and **`twin dataset`** turns the finished runs into tables a model can
train on (a run-level metrics table and a per-block feature/label table, Parquet and CSV, with every column's
unit and provenance). The side goal they serve is a local neural network that proposes toolpaths, feeds and
speeds; read [Limits](#limits) before trusting what it learns.

| part | module | command | API |
|---|---|---|---|
| run cache | `cnctwin/cache.py` | `twin run` (cached), `twin run --no-cache`, `twin cache list\|key\|verify\|evict\|compact` | `POST /api/runs` answers a cached run unless `no_cache` |
| sweeps | `cnctwin/sweep/` | `twin sweep SPEC [--out DIR] [--workers N] [--dataset [DIR]] ...` | `POST /api/sweeps`, `GET /api/sweeps`, `GET /api/sweeps/{id}` |
| datasets | `cnctwin/dataset/` | `twin dataset SOURCE... --out DIR` | a sweep job writes `dataset/` into its folder |

Contents: [quick start](#quick-start) · [run cache](#run-cache) · [sweeps](#sweeps) · [datasets](#datasets) ·
[columns](#columns) · [determinism](#determinism) · [cost](#cost) · [limits](#limits)

## Quick start

```
bin/twin warmup                                         # compile the kernels once (minutes)
bin/twin sweep my-sweep.yaml --dataset                  # every point, then DATA_DIR/sweeps/<name>/dataset/
bin/twin sweep my-sweep.yaml --dataset                  # again: every finished point is a cache hit, only new points run
bin/twin dataset DATA_DIR/sweeps/<name> other-run-dir --out training/   # any mix of sweeps and runs
bin/twin cache list                                     # what the cache holds
```

A sweep spec (full format: [docs/formats.md "Sweep"](formats.md#sweep)):

```yaml
format: cnctwin-sweep
version: 1
name: pocket roughing feeds
base: ../examples/pocket-islands-al6061        # the project (folder or project.yaml), relative to this file
seed: 1                                        # seeds the design only
design: {kind: sobol, points: 32}
parameters:
  - {name: feed, kind: override, target: feed, tool: 1, range: [0.6, 1.4]}           # T1's feed override
  - {name: speed, kind: override, target: spindle, tool: 1, range: [0.7, 1.0]}       # T1's spindle override
  - {name: stickout, kind: project, path: "tools.list[number=1].stickout_mm", range: [28, 45], unit: mm}
  - {name: stepover, kind: generator, generator: rough, argument: TROCHOID_STEP, range: [0.6, 1.6], unit: mm}
generators:
  rough: {script: make_program.py, output: op10.nc}     # examples/pocket-islands-al6061/make_program.py
options: {outputs: false}                      # run options of every point (the default: no report or video)
execution: {workers: 1, memory: 1536M}
```

## Run cache

`cnctwin.cache.run_cached(project_path, options, runs_dir, progress_cb)` returns the existing complete run of a
request when its key and its result set verify, else runs it and registers it.

**The key** is the sha256 of the canonical JSON of everything that decides a run's results:

| component | what | why |
|---|---|---|
| `project_hash` | cnctwin.project's project hash: the migrated project data, every file it names (programs, CAD, stock mesh, tool library), the database files of the objects it uses (with `extends` chains, tool materials, coatings, controllers) and every `db/models/*.yaml` | the inputs |
| `databases` | sha256 of every file under `db/` | a run reads database files beyond the objects it names (the class representative tool materials of db/models/tooling.yaml, the template defaults in db/_templates): conservative, any database edit invalidates every cached run |
| `physics` | the project's resolved physics settings (SI) | the numerical settings (also in the project hash; named for the record) |
| `options` | every run option except `resume_from` and `no_cache` | `max_sim_s`, `parts`, `outputs`, `sanity`, `checkpoint_interval_s`, `operator`, `uncertainty` ... change the run directory |
| `runner` | the simulation entry: the loop, and cnctwin.sweep.runner without operator overrides (which calls the loop unchanged), are one; any other runner is its spec with the sha256 of its source | a test or custom runner never matches a real run |
| `build` | the build id of cnctwin.core.rundir (every `cnctwin/**/*.py` and the numpy/scipy/numba/llvmlite/pyarrow/python versions), recomputed for every key (file hashes memoised by size and mtime) | never reuse a run of other code |
| `libraries` | versions of OCP (STEP import), matplotlib, Pillow, WeasyPrint, imageio-ffmpeg, PyYAML, pydantic | inspection and the derived outputs depend on them |
| `platform` | machine, system, CPU model | Numba compiles for the CPU; bit-identity holds on the same hardware (architecture 9) |
| `environment` | OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS = 1, PYTHONHASHSEED = 0 (the contract of bin/twin, bin/tsafe and the API's jobs), SOURCE_DATE_EPOCH | a process outside the contract is not deterministic: its runs are not cached at all |

**Registration.** A run is registered when it wrote run.json (complete or cut by `max_sim_s`: both are results of
their options) and the key computed after it equals the key computed before it: if the source tree, the
databases or an input changed while it ran, the run is a mix and is not cached. A run.json that names another
build than the key is refused. `twin run --resume` and the API's resumes register the finished run (a resume is
bit-identical to an uninterrupted run, architecture 9); `run_cached` itself runs resumes and replays without the
cache.

**Lookup and verification.** The newest entry of the key is checked: its run.json is there and names the
entry's build, and the sha256 over (path, sha256) of its result set (cnctwin.core.rundir.result_set_files:
findings, trace, series, frames, keyframes, inspection, surface, tools, production, part files) equals the
registered digest. The fast-tier parts a `twin replay` adds (and their lines of series/index.jsonl) are left out:
they add rows of a window the run already determines. An entry that does not verify is dropped and the next one
is tried; with none, the run is simulated again.

**Index.** `DATA_DIR/cache/index.jsonl` (or `CNCTWIN_CACHE_DIR`): append-only JSON lines (`put` with the run,
digest, byte counts, build, project hash and the key's components; `hit`; `drop` with the reason), each appended
with one write under an exclusive flock of `cache/.lock` and fsynced. A torn last line (a crash while appending)
is skipped when read and terminated before the next append. `twin cache compact` (and eviction, when the index
holds more than 1000 lines beyond four per live entry) rewrites it atomically with one line per live entry. While a key is computed its lock file
`cache/locks/<key>.lock` is held (flock, released by the kernel if the process dies), so a second request for
the same key waits and then finds the run instead of computing it twice.

**Where runs live.** `run_cached` without `runs_dir` writes into `DATA_DIR/cache/runs/` (runs the cache owns:
sweep points); `twin run` writes into `DATA_DIR/runs/` as before and registers the run there (a reference); the
API registers its runs in `DATA_DIR/runs/`. `twin run --out DIR` with a cache hit copies the cached run into DIR.

**Eviction policy** (after every registration and `twin cache evict [--max-gb N] [--max-age-days D] [--dry-run]`):

1. only runs the cache owns are deleted; a reference (a `twin run` or API run) is never deleted by the cache and
   leaves the index only when it no longer verifies;
2. an owned entry not used (registered or hit) for more than `CNCTWIN_CACHE_MAX_AGE_DAYS` (default 90) is evicted;
3. then, while the owned runs take more than `CNCTWIN_CACHE_MAX_GB` (default 10 GiB; the whole run folder counts,
   checkpoints included), the least recently used owned entry is evicted (ties by key);
4. an entry whose key lock is held (a run being computed or read) is never evicted.

A sweep's points are owned runs: when one is evicted, `twin dataset` lists it as skipped and running the sweep
again recomputes it. `CNCTWIN_CACHE=off` switches the cache off.

**API.** `POST /api/runs` computes the key of the job it would start (its `simulate` entry as runner, the jobs'
environment) and answers a verified run with **200** and `cache: {hit: true, key, run_id}` instead of queuing a
simulation; a run held outside the server's runs folder (a sweep's) is copied into a new run folder first
(`copied_from`). Otherwise it answers 201 as before with `cache: {hit: false, key}`, and the worker registers the
finished run. `{"no_cache": true}` skips the lookup; the new run is registered and supersedes. A problem of the
cache is a miss with a `note`, never a failed request.

## Sweeps

`twin sweep SPEC` (`cnctwin.sweep.run_sweep`) reads the spec (format, every key and the parameter kinds:
[docs/formats.md "Sweep"](formats.md#sweep)), draws the design, writes one project variant per point and runs
every point through the run cache.

### Parameter kinds

| kind | what it changes | how |
|---|---|---|
| `override` | the feed, spindle or rapid key of the control panel, globally or for one tool (`tool: 1`, `path: 2`) | what the operator's overrides do on the machine (below); the variant's project file is unchanged, the point's run options get `operator` |
| `program_template` | values in the program text | `{{NAME}}` placeholders of a template (a program file of the base project) replaced by the value, written with `format` (default: integers as written, floats with a decimal point: `1200.`, so no control reads `X10` as 10 increments) |
| `project` | any key of the project file | a key path: `tools.list[number=1].stickout_mm`, `tools.list[number=1].runout_um`, `coolant.fluid`, `material.id`, `operations[name=OP10].setup.workholding[0].input_torque_Nm`, `physics.lattice_mm` ... |
| `generator` | the whole program of an operation | a make_program.py style script (examples/*/make_program.py) run with the value of one of its constants (`call: constants`: its top-level assignments are replaced before it runs, so derived constants follow) or keyword arguments (`call: kwargs`), in a copy of the project folder, in its own process |

**Overrides** (`cnctwin.sweep.operator`, applied by cnctwin.sweep.runner, the sweep's default runner): they act
where the control applies them, on the commands the controller hands to motion, with the tool active at each
block (the controller's modal `tool`):

* feed: the programmed feed of every feed move whose override the control allows (not tapping, threading, G63,
  skip moves without the skip-feed override) is multiplied; per minute, per revolution and inverse-time feeds
  alike; the planner's limits (maximum cutting feed, axis and curvature limits) apply afterwards;
* spindle: the commanded speed (and a constant surface speed) of every spindle command, held within the spindle's
  maximum and the command's clamp; taps and threading inserts stay at 100 % (controls hold rigid tapping and
  threading at the programmed speed);
* rapid: the panel's rapid key (motion applies it); only the machine's `rapid_override_steps_pct` or 100 %.

Values are checked against the machine's `feed_override_range_pct` and `spindle_override_range_pct` and the
project's tools before anything runs. A run with overrides records them in run.json `options.operator`, and
`twin replay` of such a run applies them again. Not modelled: an operator turning the knob during a cut; a
per-tool spindle override is applied to the spindle commands issued while that tool is active (an `S` word
programmed before the tool change belongs to the previous tool).

**Override or program?** An override changes the feed of a whole tool by one factor, exactly as on the machine:
the right parameter for "how fast can this program run". A template or generator parameter changes the program
itself (different feeds per move, different step-overs or depths): the right parameter for "which program".
Both are recorded per block in the dataset (`feed_override`, and the realised `feed_per_tooth_m`).

### Designs

| design | points | use |
|---|---|---|
| `grid` | the product of every parameter's `values` (or `levels` over its `range`) | few parameters, a few levels each; plots |
| `lhs` | `points`, one in each of n equal strata of every parameter (scipy.stats.qmc.LatinHypercube; `optimization: random-cd` lowers the discrepancy) | the default for training data: uniform marginals with few points |
| `sobol` | `points`, a scrambled Sobol' sequence (a power of two keeps its balance; otherwise a note says so) | space filling in many dimensions; extending a sweep later |
| `random` | `points` independent uniform draws | baselines |
| `list` | the given points (a parameter a point omits takes its `default`) | re-running chosen cases |

Every design is seeded (`seed`, through cnctwin.core.rng.stream): the same spec gives the same points. A value is
mapped from the unit draw by the parameter's `range` (`scale: log`, `step`, `integer`) or picks one of its
`values` (categorical).

### Execution

* **One point at a time by default** (this box runs one simulation at a time). Each point runs in its own
  process (`isolation: process`, `python -m cnctwin.sweep.point`) with the deterministic environment and a
  memory cap like bin/tsafe (`memory_mode: auto`: `sudo -n systemd-run --scope -p MemoryMax=<memory>
  -p MemorySwapMax=256M` when available, else RLIMIT_DATA; `memory` default 1536M). A fresh process runs the
  source tree as it is on disk, so its run's key is that build's. `workers: N` runs N points at once, each in its
  own cap: N x `memory` must fit the machine. `isolation: inline` runs the points in the sweep's own process
  (tests, tiny runs).
* **Resumable.** The points' keys are content keys: running the same spec into the same folder again (or another
  sweep over the same values) takes every finished point from the cache. A point that failed (an exception: an
  input error, a conservation failure) is recorded with its error and not run again under the same key unless
  `--retry-failed`; under another build or other inputs it is retried.
* **Progress and ETA.** The progress callback (the API job's, or the command's lines) gets `done`, `total`,
  `cached`, `simulated`, `failed`, `running` and `eta_s` (the mean wall time of the simulated points times the
  remaining points, plus each running point's own remainder from its `sim_time_s` / `est_total_s`, divided by the
  workers); `status.json` of the sweep folder holds the same.
* **Cancel.** An exception from the progress callback (the API's cancel), SIGTERM or Ctrl-C terminates the running
  points (SIGTERM, then SIGKILL and `systemctl kill` of their scopes), records what finished and re-raises; a
  point process also stops by itself when the sweep process is gone (it watches its standard input).
* **API.** `POST /api/sweeps {project_id, spec, options}` (the project is the base; generator scripts must lie in
  its folder, and only the shipped example projects may use generators unless the server allows uploaded ones with
  CNCTWIN_ALLOW_UPLOAD_GENERATORS=1; a request cannot name a runner) starts a heavy job: points one after another inside the job's memory cap, then the dataset into the
  job's `dataset/` folder (`options.dataset: false` skips it). `GET /api/sweeps/{id}` gives the job, the progress,
  the summary and every point; the files are at `/api/jobs/{id}/files/...`.

The sweep folder (default `DATA_DIR/sweeps/<name>`, the job's output folder in the API): `sweep.json` (the
resolved spec and its sha256, the design's notes), `plan.json` (every point: values, unit draws, variant, run
options), `points/p<nnnnn>/project.yaml` (+ the program a template or generator wrote, `result.json`, and for
point processes `job.json`, `progress.json`, `outcome.json`, `log.txt`), `status.json` (volatile), `summary.json`
(counts, verdicts, every point's status, key and run; no wall times).

## Datasets

`twin dataset SOURCE... --out DIR [--format parquet,csv] [--no-blocks]` (`cnctwin.dataset.export`) reads sweep
folders (every finished point, with its parameters and cache key; points that failed or whose run was evicted
are listed in the manifest's `skipped`), run directories and folders of run directories, and writes:

* `runs.parquet`, `runs.csv`: one row per run (identity, the sweep's parameters as `param.<name>`, the run's
  overrides, and the run-level results below);
* `blocks.parquet`, `blocks.csv`: one row per executed NC block of every run (timeline.json): features and
  labels;
* `schema.json` (format `cnctwin-dataset-schema`): every column of both tables with its type, unit, provenance and
  meaning (the Parquet fields carry the same unit, provenance and doc as metadata);
* `manifest.json` (format `cnctwin-dataset`): the runs exported (run dir, run id, sweep, point, cache key,
  build), the points skipped and why, the builds the runs come from.

Every row is a reduction of the run's own files; nothing is simulated again. The time series are read in one
streaming pass per tier (cnctwin.core.recorder.iter_series, part by part), so a long run does not have to fit in
memory. A tick row belongs to the block of its path that holds its tick; a macro row (one per macro step,
10 ms by default) summarises its step and counts for every block of its path the step overlaps: **macro labels of
a block shorter than a macro step are the values of the step(s) it falls in, shared with its neighbours**. A
channel a run did not record reads NaN, never a made-up number.

Types: `float64` (NaN, an empty CSV cell, when the run does not have the value), `int64`, `int8` (flags: 1 true,
0 false, -1 not known) and `string`. Units are SI, as the name's last token says (`_m`, `_N`, `_K`, `_rad_s`;
`-` dimensionless).

## Columns

Every column is in `schema.json` beside the data, with the same unit, provenance and meaning as here (cnctwin/dataset/schema.py).

### runs: one row per run

| column | type | unit | source | meaning |
|---|---|---|---|---|
| `sweep` | string | - | sweep.json name | name of the sweep the run belongs to ('' for a run given directly) |
| `point` | int64 | - | plan.json points[].index | index of the sweep point (-1 for a run given directly) |
| `run_id` | string | - | run.json run_id | the run's id |
| `cache_key` | string | - | points/p<n>/result.json key | run cache key of the point ('' when unknown) |
| `project_name` | string | - | run.json project.name | project name (a sweep variant's carries its values) |
| `project_hash` | string | - | run.json project_hash | sha256 of the project's content and inputs |
| `build` | string | - | run.json versions.build | build id of the code that made the run |
| `seed` | int64 | - | run.json seed | the project's seed (stochastic models) |
| `machine` | string | - | run.json project.machine.id | machine id |
| `material` | string | - | run.json project.material.db_id (else .id) | workpiece material id |
| `complete` | int8 | - | run.json complete | 1: the run simulated the whole program (0: max_sim_s cut it) |
| `safe` | int8 | - | run.json verdict.safe | 1: no finding of severity error or fatal (architecture 11) |
| `feed_override` | float64 | - | run.json options.operator.feed_override['*'] | feed override of every tool not named per tool (1 = 100 %, cnctwin.sweep.operator) |
| `spindle_override` | float64 | - | run.json options.operator.spindle_override['*'] | spindle override of every tool not named per tool (1 = 100 %) |
| `rapid_override` | float64 | - | run.json options.operator.rapid_override | rapid override (1 = 100 %) |
| `cycle_time_s` | float64 | s | run.json summary.cycle_time_s | machine cycle time (longest path) |
| `sim_time_s` | float64 | s | run.json summary.sim_time_s | simulated time |
| `takt_s` | float64 | s | production.json per_part_s | time per part including the operator model |
| `removed_mass_kg` | float64 | kg | run.json summary.removed_mass_kg | removed material (exact field volume x rho) |
| `removed_volume_m3` | float64 | m3 | run.json summary.removed_volume_m3 | removed material volume |
| `chip_mass_kg` | float64 | kg | run.json summary.chips_kg (sum of the six inventories) | chip mass booked by the chips domain |
| `cutting_work_J` | float64 | J | run.json summary.shaft_work_J + feed_work_J | work of the cut: spindle shaft work plus feed work |
| `energy_per_part_J` | float64 | J | production.json per_part[simulated].energy.total_J (mean) | machine energy per part (spindle, axes, pumps, base load, air) |
| `tool_cost_per_part_eur` | float64 | eur | production.json per_part[simulated].tool_cost_eur (mean) | tool cost per part from the wear model and tool prices |
| `force_peak_max_N` | float64 | N | series/macro cutting.force_peak_N [path]: maximum | largest instantaneous tool force (L0 peak of every macro step) |
| `force_max_N` | float64 | N | series/tick cutting.force_x_N, force_y_N, force_z_N [path]: \|F\| per tick, maximum | largest tick-mean tool force |
| `force_mean_N` | float64 | N | series/tick cutting.force_x_N, force_y_N, force_z_N [path]: \|F\| per tick, mean over the ticks in cut (\|F\| > 0) | mean tool force while cutting |
| `torque_max_Nm` | float64 | N m | series/tick cutting.torque_Nm [path]: \|T\| maximum | largest tick-mean cutting torque |
| `torque_mean_Nm` | float64 | N m | series/tick cutting.torque_Nm [path]: \|T\| mean over the ticks in cut | mean cutting torque while cutting |
| `spindle_power_max_W` | float64 | W | series/macro cutting.spindle_power_W [path]: maximum | largest step-mean spindle cutting power |
| `spindle_power_mean_W` | float64 | W | series/macro cutting.spindle_power_W [path]: mean over the steps in cut (cutting.in_cut_fraction > 0) | mean spindle cutting power while cutting |
| `spindle_load_max` | float64 | - | series/macro cutting.spindle_load_fraction [path]: maximum | largest cutting torque over the available torque at speed and gear |
| `spindle_thermal_load_max` | float64 | - | series/macro cutting.spindle_thermal_load [path]: maximum | largest spindle motor thermal load (1 = S1 limit) |
| `mrr_mean_m3_s` | float64 | m3/s | series/macro cutting.chip_volume_rate_m3_s [path]: mean over the steps in cut | mean material removal rate while cutting |
| `tool_deflection_max_m` | float64 | m | series/tick dynamics.tool_deflection_x_m, tool_deflection_y_m [path]: lateral magnitude, maximum | largest tool deflection at the reference station |
| `vibration_pp_max_m` | float64 | m | series/macro dynamics.vibration_pp_path_m [path]: maximum | largest peak-to-peak relative tool-work vibration of a step |
| `chatter_index_max` | float64 | - | series/macro dynamics.chatter_index_path [path]: maximum | largest non-forced share of the vibration energy (chatter) |
| `stability_margin_min` | float64 | - | series/macro dynamics.stability_margin [path]: minimum (finite, > 0) | smallest depth limit over depth of cut at the operating point (below 1: chatter predicted) |
| `tool_edge_temp_max_K` | float64 | K | series/macro thermal.tool_edge_max_K: maximum | hottest cutting edge |
| `tool_flank_temp_max_K` | float64 | K | series/macro thermal.tool_flank_max_K: maximum | hottest flank contact |
| `work_temp_max_K` | float64 | K | series/macro thermal.work_bulk_max_K [workpiece]: maximum | hottest finite volume of the workpiece |
| `work_surface_temp_max_K` | float64 | K | series/macro thermal.work_surface_K [workpiece]: maximum | hottest surface temperature at the contact |
| `spindle_temp_max_K` | float64 | K | series/macro thermal.spindle_temp_K [spindle]: maximum | hottest spindle |
| `chip_temp_max_K` | float64 | K | series/macro chips.max_temp_K: maximum | hottest chip |
| `coolant_tank_temp_max_K` | float64 | K | series/macro coolant.tank_temp_K: maximum | warmest coolant tank |
| `holder_margin_min` | float64 | - | series/macro holding.tool_margin: minimum (finite) | smallest holder friction capacity over peak demand (below 1: the tool slips or pulls out) |
| `part_slip_max_m` | float64 | m | series/macro holding.part_slip_m [workpiece]: maximum | largest persistent slip of a workpiece in its setup |
| `burr_max_m` | float64 | m | surface.json burrs[].height_m: maximum | highest exit burr |
| `burr_count` | int64 | - | surface.json burrs | exit burrs recorded |
| `face_Ra_max_m` | float64 | m | surface.json faces[].Ra_m: maximum | roughest machined face not named by a roughness tolerance |
| `face_Ra_mean_m` | float64 | m | surface.json faces[].Ra_m: mean | mean Ra of those faces |
| `mass_balance_ok` | int8 | - | run.json summary.conservation.mass.ok | 1: the mass balance held every step |
| `energy_balance_ok` | int8 | - | run.json summary.conservation.energy.ok | 1: the energy balance held every step |
| `findings_count` | int64 | - | findings.json | findings of the run |
| `findings[info]` | int64 | - | findings.json severity == info | findings of severity info |
| `findings[warning]` | int64 | - | findings.json severity == warning | findings of severity warning |
| `findings[error]` | int64 | - | findings.json severity == error | findings of severity error |
| `findings[fatal]` | int64 | - | findings.json severity == fatal | findings of severity fatal |


Columns made from the data (one per parameter, finding category, tool, inspected characteristic and named surface feature; the run table lists them after the fixed ones):

| column | type | unit | source | meaning |
|---|---|---|---|---|
| `param.<name>` | float64 | <unit> | plan.json points[].values | the sweep parameter (its spec's unit and doc) |
| `findings[<category>]` | int64 | - | findings.json category == <category> | findings of category <category> |
| `VB_max_m[T<n>]` | float64 | m | tools.json tools[number=T<n>].VB_max_m | largest flank wear of T<n> |
| `KT_max_m[T<n>]` | float64 | m | tools.json tools[number=T<n>].KT_max_m | deepest crater of T<n> |
| `life_used_fraction[T<n>]` | float64 | - | tools.json tools[number=T<n>].life_used_fraction | life used of T<n> (first criterion) |
| `cut_time_s[T<n>]` | float64 | s | tools.json tools[number=T<n>].cut_time_s | cutting time of T<n> |
| `edge_temp_max_K[T<n>]` | float64 | K | tools.json tools[number=T<n>].max_edge_temp_K | hottest edge of T<n> |
| `broken[T<n>]` | int8 | - | tools.json tools[number=T<n>].broken | 1: T<n> broke |
| `measured_m[<id>]` | float64 | m | inspection.json tolerances[id=<id>].measured_m | value of characteristic <id> as the virtual CMM measures it (20 degC, unclamped) |
| `deviation_m[<id>]` | float64 | m | inspection.json tolerances[id=<id>].deviation_m | deviation of <id> from its nominal |
| `pass[<id>]` | int8 | - | inspection.json tolerances[id=<id>].pass | 1: <id> is in tolerance |
| `Ra_m[<id>]` | float64 | m | surface.json features[id=<id>].Ra_m | Ra of feature <id> |
| `Rz_m[<id>]` | float64 | m | surface.json features[id=<id>].Rz_m | Rz of feature <id> |


Finding categories are every category of cnctwin.core.findings (`collision`, `overtravel`, `spindle_overload`, ..., `time_balance`) plus any other a run raised; tools are labelled `T<n>` (`T<n>@<path>` on paths other than 1). Faces of surface.json that no roughness tolerance names (`face-<k>`) are grouped per run and their numbering is not stable between runs, so they enter only as `face_Ra_max_m` and `face_Ra_mean_m`.

### blocks: one row per executed NC block

| column | type | unit | source | meaning |
|---|---|---|---|---|
| `sweep` | string | - | sweep.json name | sweep of the run ('' for a run given directly) |
| `point` | int64 | - | plan.json points[].index | sweep point (-1 for a run given directly) |
| `run_id` | string | - | run.json run_id | the run's id |
| `block` | int64 | - | timeline.json blocks[].index | position of the block in the run's timeline |
| `op` | string | - | timeline.json blocks[].op | operation |
| `program` | string | - | timeline.json blocks[].program (file name) | program file name |
| `line` | int64 | - | timeline.json blocks[].line | line of the block in its program |
| `path` | int64 | - | timeline.json blocks[].path | controller path (0-based) |
| `part` | int64 | - | timeline.json blocks[].part | part of the batch (0-based) |
| `kind` | string | - | timeline.json blocks[].kind | path kind: rapid, linear, arc, dwell, tool_change, ... |
| `tool` | int64 | - | timeline.json blocks[].tool | T number active (-1: none) |
| `t0_s` | float64 | s | timeline.json blocks[].tick0 x tick_s | start of the block |
| `duration_s` | float64 | s | timeline.json blocks[] (tick1 - tick0) x tick_s | duration of the block |
| `cutting_s` | float64 | s | timeline.json blocks[].ticks.cutting x tick_s | time in cut |
| `air_cutting_s` | float64 | s | timeline.json blocks[].ticks.air_cutting x tick_s | feed time out of contact |
| `tool_type` | string | - | run.json project.tools.list[number].tool.type | tool type (database template) |
| `tool_diameter_m` | float64 | m | run.json project.tools.list[number].tool.geometry.diameter_m | cutting diameter |
| `tool_flutes` | int64 | - | run.json project.tools.list[number].tool.geometry.flutes | flutes (teeth) |
| `tool_corner_radius_m` | float64 | m | run.json project.tools.list[number].tool.geometry.corner_radius_m | corner radius |
| `tool_helix_rad` | float64 | rad | run.json project.tools.list[number].tool.geometry.helix_rad | helix angle |
| `tool_rake_rad` | float64 | rad | run.json project.tools.list[number].tool.geometry.rake_rad | rake angle |
| `tool_material_class` | string | - | run.json project.tools.list[number].tool.tool_material.class | tool substrate class |
| `tool_coating` | string | - | run.json project.tools.list[number].tool.coating.id | coating id |
| `tool_stickout_m` | float64 | m | run.json project.tools.list[number].stickout_m | stick-out from the holder |
| `tool_runout_m` | float64 | m | run.json project.tools.list[number].runout_m | runout (TIR) |
| `material` | string | - | run.json project.material.db_id (else .id) | workpiece material id |
| `material_hardness_HB` | float64 | HB | run.json project.material.physical.hardness.HB | Brinell hardness |
| `feed_override` | float64 | - | run.json options.operator.feed_override [the block's tool, else '*'] | feed override of the block's tool (1 = 100 %) |
| `spindle_override` | float64 | - | run.json options.operator.spindle_override [the block's tool, else '*'] | spindle override of the block's tool (1 = 100 %) |
| `spindle_speed_rad_s` | float64 | rad/s | series/tick motion.spindle_omega_rad_s [spindle of the path]: \|omega\|, mean over the block's ticks | spindle speed (actual, from the spindle servo) |
| `feed_speed_m_s` | float64 | m/s | series/tick motion.path_speed_m_s [path]: mean over the block's ticks | mean tool-tip path speed |
| `feed_speed_cut_m_s` | float64 | m/s | series/tick motion.path_speed_m_s [path]: mean over the block's ticks in cut (\|F\| > 0) | mean path speed while cutting |
| `feed_per_tooth_m` | float64 | m | feed_speed_cut_m_s / (spindle_speed_rad_s / 2 pi x tool_flutes) | feed per tooth while cutting |
| `cutting_speed_m_s` | float64 | m/s | spindle_speed_rad_s x tool_diameter_m / 2 | cutting speed at the diameter |
| `ae_m` | float64 | m | series/macro removal.ae_m [path]: maximum over the steps overlapping the block | radial engagement |
| `ap_m` | float64 | m | series/macro removal.ap_m [path]: maximum over the steps overlapping the block | axial engagement |
| `engaged_angle_rad` | float64 | rad | series/macro removal.engaged_angle_rad [path]: maximum | engaged angle |
| `mrr_m3_s` | float64 | m3/s | series/macro cutting.chip_volume_rate_m3_s [path]: mean over the overlapping steps in cut | material removal rate |
| `coolant_on` | float64 | - | series/macro coolant.on: mean over the overlapping steps | share of the block with coolant, MQL or air at the outlets |
| `coolant_flow_m3_s` | float64 | m3/s | series/macro coolant.flow_m3_s: mean | coolant delivered |
| `force_mean_N` | float64 | N | series/tick cutting.force_x_N, force_y_N, force_z_N [path]: \|F\| per tick, mean over the block's ticks in cut | mean tool force while cutting |
| `force_max_N` | float64 | N | series/tick cutting.force_x_N, force_y_N, force_z_N [path]: \|F\| per tick, maximum over the block | largest tick-mean force |
| `force_peak_N` | float64 | N | series/macro cutting.force_peak_N [path]: maximum over the overlapping steps | largest instantaneous force |
| `torque_mean_Nm` | float64 | N m | series/tick cutting.torque_Nm [path]: \|T\| mean over the block's ticks in cut | mean cutting torque |
| `torque_max_Nm` | float64 | N m | series/tick cutting.torque_Nm [path]: \|T\| maximum over the block | largest cutting torque |
| `spindle_power_mean_W` | float64 | W | series/macro cutting.spindle_power_W [path]: mean over the overlapping steps in cut | spindle cutting power |
| `spindle_power_max_W` | float64 | W | series/macro cutting.spindle_power_W [path]: maximum | largest spindle power |
| `spindle_load_max` | float64 | - | series/macro cutting.spindle_load_fraction [path]: maximum | largest spindle load |
| `tool_edge_temp_max_K` | float64 | K | series/macro thermal.tool_edge_max_K: maximum over the overlapping steps | hottest edge |
| `work_temp_max_K` | float64 | K | series/macro thermal.work_bulk_max_K [workpiece]: maximum | hottest workpiece |
| `VB_increment_m` | float64 | m | series/macro wear.VB_max_m: maximum - minimum over the overlapping steps | flank wear grown during the block (steps shared with neighbouring blocks included) |
| `life_used_increment` | float64 | - | series/macro wear.life_used_slot [the tool's slot]: maximum - minimum | tool life used during the block |
| `deflection_max_m` | float64 | m | series/tick dynamics.tool_deflection_x_m, tool_deflection_y_m [path]: lateral magnitude, maximum over the block | largest tool deflection |
| `vibration_pp_max_m` | float64 | m | series/macro dynamics.vibration_pp_path_m [path]: maximum | largest peak-to-peak vibration |
| `chatter_index_max` | float64 | - | series/macro dynamics.chatter_index_path [path]: maximum | chatter index |
| `stability_margin_min` | float64 | - | series/macro dynamics.stability_margin [path]: minimum (finite, > 0) | stability margin at the operating point |
| `Ra_m` | float64 | m | surface.json features[] whose op and lines name the block: maximum Ra_m | roughness of the named surfaces this block generated |
| `burr_max_m` | float64 | m | surface.json burrs[block == block].height_m: maximum | highest burr the block left |
| `findings_count` | int64 | - | findings.json: tick in [tick0, tick1), same op and line | findings raised in the block |
| `findings_severe_count` | int64 | - | findings.json as findings_count, severity error or fatal | errors and fatal findings raised in the block |
| `findings_categories` | string | - | findings.json as findings_count: categories, sorted, ';'-separated | categories of the block's findings |

Features (what a toolpath optimiser chooses or knows before the cut): the tool and material columns, `feed_override`, `spindle_override`, `spindle_speed_rad_s`, `feed_speed_m_s`, `feed_speed_cut_m_s`, `feed_per_tooth_m`, `cutting_speed_m_s`, `ae_m`, `ap_m`, `engaged_angle_rad`, `mrr_m3_s`, `kind`, `coolant_on`, `coolant_flow_m3_s`. Labels (what the twin predicts): forces, torque, power, spindle load, temperatures, `VB_increment_m`, `life_used_increment`, deflection, vibration, chatter index, stability margin, `Ra_m`, `burr_max_m` and the findings.

## Determinism

* The same spec and seed give the same points, variants and keys; a variant's project hash does not depend on
  where the sweep folder is (generated programs are written beside the variant and named relatively; the other
  files are named by absolute path into the base project, the same for every point).
* A run is bit-identical for the same key on the same hardware (architecture 9): that is what makes a cached run
  as good as a fresh one.
* `twin dataset` writes the same bytes for the same runs: rows in source order (sweep points by index, blocks by
  timeline index), columns in schema order, sums accumulated in time order, Parquet with fixed settings and no
  timestamps, CSV with shortest round-trip floats (tests/unit/dataset compares two exports byte for byte).

## Cost

Measured on this box (4 cores shared with other work, under bin/tsafe at nice 10, 2026-11-09):

| what | cost |
|---|---|
| a simulation | the tiny mill slot of cnctwin.core.warmup, 3.9 s simulated (cut at `max_sim_s`): 650 s wall the first time (nearly all of it compiling every Numba kernel for a changed source tree), 21 s once compiled (databases, domain initialisation, air moves); cutting costs 10-14 CPU s per simulated second at the hot loop's present state, so a 60 s program is 10-15 CPU minutes per point, about 100-140 points a day per worker |
| the cache key | 2.2-2.5 s for the first key in a process (loading the databases), then 0.13-0.16 s (resolving the project 0.08 s, the build id 0.06 s with file hashes memoised by size and mtime, the database digest 0.03 s) |
| verifying a hit | full (sha256 of the result set): 0.063 s for a real run (58 files, 12 MB; the folder is 61 MB with checkpoints), about 200 MB/s; quick (file count and bytes): 0.009 s |
| a cache hit in a sweep | 0.13 s per point (4 cached points in 0.52 s, planning included) |
| a point process | about 5-6 s per point (a fresh interpreter, imports, the databases, the key before and after the run) against 0.25 s inline: nothing next to a simulation |
| planning | about 50 ms per variant; a generator costs what its script costs (examples/pocket-islands-al6061/make_program.py about 10 s: it computes its own engagement map) |
| `twin dataset` | 0.04 s for the real 3.9 s run (2 s with the first imports of the process); 3.1-4.4 s for a 979 s run with 608 blocks (98 000 macro and 979 000 tick rows): about 3-4 ms per simulated second, streaming, under 0.5 GB |
| disk | a run folder: 61 MB for the 3.9 s run (49 MB of checkpoints); its rows in the dataset: kilobytes |

## Limits

**A model trained on the twin learns the twin, errors included.** The twin's models are physics models with
parameters from published sources; nothing is tuned to a reference. The validation suite
([docs/validation-report.md](validation-report.md), 2026-11-03, 93 published experiments) currently misses its
error budget for every quantity; the median absolute relative errors over the independent cases are:

| quantity | median error | p90 error | budget | what it means for training data |
|---|---|---|---|---|
| cycle time | 7.0 % | 21.9 % | 5 % | usable: block times, feed limits and accelerations are close |
| stability lobes (limit depth) | 12.0 % | 44.8 % | 25 % | usable for where chatter starts, with margin |
| drilling torque | 24.7 % | 82.3 % | 15 % | trends |
| chip compression ratio | 26.4 % | 49.1 % | 20 % | trends |
| cutting force | 31.0 % | 181.6 % | 15-30 % | relative comparisons at a fixed tool and material; absolute limits need calibration |
| shear angle | 35.1 % | 48.1 % | 20 % | trends |
| chip temperature | 20.2 % | 44.4 % | 20 % | one case only |
| roughness (Ra, Rz) | 55.6 % | 188.8 % | 30 % | not for optimising finish |
| coolant reach | 64.5 % | 105.5 % | 15 % | not for coolant decisions |
| tool temperature | 65.3 % | 120.6 % | 15 % | not as a limit |
| wall error | 71.1 % | 163.4 % | 25 % | not for tolerances of thin walls |
| drilling thrust | 90.3 % | 118.6 % | 20 % | not trustworthy |
| spindle growth | 92.8 % | 95.5 % | 30 % | not trustworthy |
| tool life (Taylor) | 99.5 % | 213.6 % | 50 % | not trustworthy: wear increments and life used are not a basis for choosing speeds |
| chip form | 5 of 32 classes | | 25 % | not trustworthy |

Some of these misses are model defects being fixed; the report gives every case. Read them as the state of the
twin when a dataset was made (`manifest.json` names the builds): a dataset of an older build keeps that build's
errors.

**Which outputs to optimise on.**

* *Trustworthy for optimisation*: what follows from geometry and kinematics: the removed volume and mass (the
  removal field is exact; the ledger checks it every step), the material removal rate, the engagement (`ae_m`,
  `ap_m`, `engaged_angle_rad`), the programmed and realised feeds and speeds, cycle time; collisions, overtravel
  and gouges (findings of geometry); the stability margin as a relative indicator.
* *Usable for ranking, not as absolute limits*: forces, torque, spindle power and load, deflection (forces
  through the tool's compliance), chatter index; compare candidates at the same tool, holder and material, and
  keep the catalogue's limits as the constraint.
* *Not trustworthy for optimisation*: tool wear and life, tool and chip temperatures, roughness, burrs, thermal
  growth, coolant reach, chip form. A model optimising feeds and speeds will exploit these errors (if wear is
  under-predicted at high cutting speed, the optimum moves to high speed). Constrain them with the tool maker's
  data instead of the twin's.

**Calibration reduces them.** `twin calibrate` (docs/models/calibration.md) fits model coefficients to recorded
data of the real machine (spindle power or load, positions, following error, temperatures; MTConnect, OPC UA,
FOCAS logs or CSV): `kc_scale` of the material's cutting coefficients, the spindle's idle and bearing power, the
axes' servo dynamics, the thermal model. The result is a calibration overlay the project includes
(`calibration: calibration.yaml` in project.yaml, applied when the project is resolved; docs/formats.md "Project
file"): it is an input of the project, so every cached run of that project is out of date by its key and a sweep
run again simulates every point with the calibrated twin (its variants keep the key, with the overlay's path made
absolute). Calibrate on the machine, tools and materials the model will be
used for, and validate on cuts it was not calibrated on.

**Further limits of the data.**

* Ranges: the data covers the sweep's ranges and the base project's tools, materials and machine; a model says
  nothing reliable outside them. A program generator changes toolpaths only within what its script can make.
* Scatter: a run uses the project's seed; the stochastic models (edge chipping, chip scatter) draw the same in
  every point (common random numbers), so differences between points are the parameters', not noise. To train on
  the scatter, vary the seed (a `project` parameter over `seed`) or use the uncertainty study's factors as
  parameters.
* Block labels from the macro tier are shared by blocks within one macro step (above); the tick tier (1 ms) gives
  forces, torque and deflection per block exactly.
* Findings attributed to a block are those raised during it on its line; findings of the end of an operation
  (inspection) are in the run table only.
* A dataset mixes builds when its runs come from several (`manifest.json` `builds`, a note): train on one build.
