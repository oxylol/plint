# How to run CNC Twin

Everything the twin does can be done from the command line (`bin/twin`), the HTTP API or the browser UI;
the three call the same code and write the same files. This page is the user's guide to all of it; the
references behind it are [api.md](api.md), [ui.md](ui.md), [ml.md](ml.md), [formats.md](formats.md) and
[models/loop.md](models/loop.md).

Contents: [environment](#1-environment) · [check](#2-check-a-project) · [run](#3-run-a-project) ·
[results](#4-read-the-results) · [examples](#5-the-examples) · [checkpoints, resume, replay](#6-checkpoints-resume-and-replay) ·
[run cache](#7-the-run-cache) · [sweeps](#8-sweeps) · [datasets](#9-datasets) · [uncertainty](#10-uncertainty-studies) ·
[calibration](#11-calibration-from-machine-data) · [compare](#12-comparing-two-runs) · [validation and benchmarks](#13-validation-and-benchmarks) ·
[the service](#14-the-service-api-and-ui) · [exit codes](#15-exit-codes)

## 1. Environment

* `bin/twin <command>` runs `python -m cnctwin.cli` from the virtualenv `~/.venvs/cnc-twin` (`CNCTWIN_VENV`) on
  this source tree, or, without a virtualenv, from the image `plint/cnc-twin:local` (`CNCTWIN_IMAGE`) with the
  current directory mounted at the same path (paths outside it are not visible there). It sets the
  deterministic environment: one thread for OpenMP/OpenBLAS/MKL/Numba, `PYTHONHASHSEED=0`,
  `SOURCE_DATE_EPOCH`, and a Numba cache folder keyed by a hash of every kernel source file, the
  numba/llvmlite versions and the CPU (docs/determinism.md). A run made outside this environment is not
  cached.
* `bin/tsafe <command>` runs any command at nice 10 in a memory-capped cgroup (`TSAFE_MEM`, default 900M) with
  the same environment; tests and ad-hoc Python go through it (`bin/tsafe python -m pytest ...`).
* `DATA_DIR` is where runs, sweeps and the cache go: default `runs-data/` in this folder, `/data` in the
  container (`data/cnc-twin` on Plint). `CNCTWIN_ROOT` points the code at another source tree (the image uses
  `/app`).
* `bin/twin warmup` compiles and caches every kernel (each kernel module in its own process, then two tiny
  projects through the loop): about 12 CPU minutes and 1.4 GB the first time, and again after any change to
  a file that holds a kernel. The image runs it at build time. Without it the first run compiles on the fly.
* `twin --traceback <command>` (or `CNCTWIN_TRACEBACK=1`) shows the Python traceback of an error.

## 2. Check a project

```
bin/twin check examples/pocket-islands-al6061 --out /tmp/pocket-check
```

`twin check PROJECT [--write] [--json] [--out DIR] [--set KEY=VALUE] [--quiet]` runs, per operation, the
controller's dry run on an ideal machine (alarms with their lines, the tools and spindle speeds used, the
cycle-time lower bound), the planner's cycle time, the L0 time step of every tool by architecture 5.1, the
lattice rule (`lattice_mm <= min(ae) / 4` on every cut, or a refine box), the estimated CPU seconds per
simulated second and the disk a full fast recording would take, and the static program checks (sanity
findings). It writes `check.json` (format `cnctwin-check`) to `--out` (default `<project>/.twin-check`, which
writes into the project folder: give `--out` for the examples). `--write` stores the resolved time steps under
`physics.resolved` in project.yaml (only that entry is changed, comments kept); a run then uses them. Exit
status 1 when the check found errors.

A new project: `bin/twin new DIR --machine <id> [--material <id>] [--coolant <id>] [--name ..]` writes a
`project.yaml` from the template and the shortest program of the machine's dialect; fill in the stock, tools,
workholding and program (formats.md "Project file").

## 3. Run a project

```
bin/twin run examples/pocket-islands-al6061/project.yaml --set max_sim_s=8
```

`twin run PROJECT [--out RUN_DIR | --runs DIR] [--resume [MARKER]] [--set KEY=VALUE ...] [--no-cache] [--quiet]`

* The run directory is a new numbered folder under `--runs` (default `DATA_DIR/runs`), named
  `<nnnnn>-<project folder>-<8 hex digits of the project and options hash>`, or the folder `--out` names.
* Progress goes to standard error (simulated time of the expected total, operation, phase, wall time) and into
  the run's `status.json`; at the end the verdict: `SAFE TO RUN` (exit 0) or `NOT SAFE TO RUN` and the reasons
  (exit 3).
* The run goes through the [run cache](#7-the-run-cache): when a finished run with the same key exists and
  verifies, it is reported (copied to `--out` when given) instead of simulated again; `--no-cache` runs anyway.

Run options (`--set KEY=VALUE`, the value read as JSON when it parses, dotted keys nest; the API takes the same
keys in `options`):

| option | effect |
|---|---|
| `max_sim_s` | stop after this much simulated time: the running operation is ended (not inspected), later ones are skipped, run.json says `complete: false` and `summary.truncated` |
| `parts` | override `production.parts` |
| `checkpoint_interval_s` | simulated seconds between checkpoints (default 10, db/models/recording.yaml) |
| `outputs` | settings of the derived outputs (pictures, video, sound, report; docs/outputs.md); `false` makes none, which saves minutes on a short test |
| `sanity` | `false`: no static program checks at operation start |
| `resume_from` | set by `--resume` (section 6) |

Operator overrides (`operator`: feed, spindle and rapid override per tool, fractions with 1 = 100 %; format in
formats.md "Operator option"), for example `--set operator.feed_override=1.2` (every tool) or
`--set 'operator.spindle_override={"T3": 0.9}'`: the loop's run_project does not read them, so `twin run` sends a
run whose options carry them through cnctwin.sweep.runner, which applies them (and refuses, before the run, an
override outside the machine's override ranges or rapid keys, for a tool the project does not have, or of a tap's
spindle speed). The option is normalised first (entries at 100 % dropped; nothing left: the plain run) and the
run's cache key names that runner, so it is the run a sweep point with the same overrides makes. The API does the
same (its default `simulate` entry is cnctwin.sweep.runner); a sweep (section 8) varies the overrides as
parameters.

A long program takes long: with every domain on, the loop costs several CPU seconds per simulated second
(docs/models/loop.md section 13, docs/performance.md). Simulate the start with `max_sim_s`, check the cost with
`twin check` first, and keep `fast: windows` recording (the project's `recording` section).

The same from Python, as the API worker calls it (run it with `bin/tsafe python` for the deterministic
environment):

```python
from pathlib import Path
from cnctwin.core.loop import run_project
result = run_project(Path("examples/pocket-islands-al6061/project.yaml"), Path("/tmp/run1"),
                     {"max_sim_s": 8, "outputs": False}, progress_cb=None)
print(result["verdict"])          # {safe, reasons}; run_dir, run_id, complete, summary, cpu_s, outputs
```

## 4. Read the results

Every result is a file of the run directory (formats.md "Run directory"). The main ones:

| file | what |
|---|---|
| `run.json` | the verdict `{safe, reasons}`, summary numbers (cycle time, removed mass, energy, conservation), the resolved project, versions, input hashes, every domain's final document |
| `findings.json`, `trace.jsonl` | the findings, and the trace events their cause chains point to |
| `timeline.json` | every executed NC block with its tick range and time categories |
| `series/` | the time series of every channel (Parquet; tiers fast, tick, macro) |
| `inspection.json`, `surface.json`, `tools.json`, `production.json`, `sld/T<n>.json` | inspection against the CAD, finish and burrs, wear and life, cycle time and costs, stability lobes |
| `part_<op>.glb/.stl/.3mf` | the machined part of each operation |
| `report.html`, `report.pdf`, `setup_sheet.*`, `images/`, `video.mp4`, `audio.wav` | the derived outputs |

```
bin/twin explain RUN_DIR                     # every finding: id, severity, category, time, line, message
bin/twin explain RUN_DIR 4                   # finding 4 and its chain of cause (trace.jsonl), root causes last
bin/twin explain RUN_DIR 4 --json
bin/twin export RUN_DIR --tier tick --channels cutting.spindle_power_W --format csv --out power.csv
bin/twin export RUN_DIR --tier macro --t0 1.0 --t1 2.0 --format parquet --out window.parquet
bin/tsafe python -m cnctwin.core.outputs RUN_DIR   # make the derived outputs again (pictures, video, sound, report)
```

`twin export RUN [--tier fast|tick|macro] [--channels a,b] [--format csv|parquet] [--t0 S] [--t1 S] [--out FILE]`
writes full-resolution series (CSV to standard output without `--out`). In the UI, the run page shows all of
it (docs/ui.md "The run page").

## 5. The examples

`examples/<name>/` are complete projects with a README (the job, setup, tools, what the run should show) and
`EXPECTED.md` (hand-calculated bands): `pocket-islands-al6061`, `thin-wall-al7075`, `two-setup-flip`,
`drill-tap-plate-1045`, `batch-50`, `sanity-mistakes`, `blade-5axis-ti`, `die-h13-ballnose`,
`ti-slot-throughtool`, `lathe-shaft-4140` (one line each in the README).

```
bin/twin check examples/thin-wall-al7075 --out /tmp/thin-wall-check
bin/twin run examples/thin-wall-al7075 --set max_sim_s=8 --set outputs=false
```

In the UI they are listed as `example-<folder>` (read-only: upload a copy to change one). The regression test
`tests/regression/test_examples.py` runs the first 8 simulated seconds of each with every domain on and
compares what it reaches with EXPECTED.md (docs/models/loop.md section 14 has the latest results;
`bin/test --all` runs it, slowly). Several examples have a generator (`make_program.py`, `make_cad.py`); run
them with `bin/tsafe python examples/<name>/make_program.py` from this folder.

## 6. Checkpoints, resume and replay

* A run writes a checkpoint every `checkpoint_interval_s` of simulated time (10 s) and at every operation end
  (`checkpoints/ck_<tick>.npz` with its `.json` commit marker; the last two and every operation end are kept).
* `bin/twin run PROJECT --out RUN_DIR --resume` continues a cancelled, killed or failed run from its newest
  committed checkpoint (`--resume checkpoints/ck_<tick>.json` from a given one): everything after the
  checkpoint is truncated and simulated again, and the result set is byte for byte the uninterrupted run's
  (docs/determinism.md). A checkpoint of another project or another build is refused: start the run again.
* `Ctrl-C` stops a command-line run (status `cancelled`, exit 130); the API cancels with
  `POST /api/runs/{id}/cancel` and resumes with `POST /api/runs/{id}/resume`; the UI has both buttons.
* `bin/twin replay RUN_DIR --window T0 T1` re-simulates the window from the nearest checkpoint before T0 and
  writes only the fast (L0) tier of that window (`series/fast/replay-<tick0>-<tick1>.parquet`): the way to get
  full-rate data around a finding raised after its rows left the ring buffer. A run made with operator
  overrides is replayed with them.

## 7. The run cache

`twin run`, the API's `POST /api/runs` and every sweep point look a run up by its **key**, the sha256 of
everything that decides its results: the project and every file it names, the database files it uses and a
digest of all of `db/`, the run options, the runner, the build (every `cnctwin/**/*.py` and the library
versions), the platform and the deterministic environment (docs/ml.md "Run cache"). A finished run whose result
set still verifies is answered at once; any change to an input, a database file or the code makes a new key.

```
bin/twin cache list                          # the cached runs, most recently used last
bin/twin cache key PROJECT [--set K=V]       # the key of a project with options, and its components
bin/twin cache verify [--quick]              # re-hash every entry, drop the ones that changed
bin/twin cache evict [--max-gb N] [--max-age-days D] [--dry-run]
bin/twin cache compact                       # rewrite the index with one line per live entry
```

The cache lives in `DATA_DIR/cache` (`CNCTWIN_CACHE_DIR`); it deletes only runs it owns (sweep points), oldest
use first, beyond `CNCTWIN_CACHE_MAX_GB` (10) or `CNCTWIN_CACHE_MAX_AGE_DAYS` (90); `CNCTWIN_CACHE=off`
switches it off. `twin run --no-cache` and the API's `no_cache: true` run anyway and register the new run.

## 8. Sweeps

A sweep runs many variants of a project through the cache: overrides, any project key, values in a program
template, or a program generator's constants (docs/ml.md "Sweeps"; the spec format: formats.md "Sweep").

```yaml
format: cnctwin-sweep
version: 1
name: pocket roughing feeds
base: ../examples/pocket-islands-al6061        # relative to this file
seed: 1
design: {kind: sobol, points: 32}              # grid | lhs | sobol | random | list
parameters:
  - {name: feed, kind: override, target: feed, tool: 1, range: [0.6, 1.4]}
  - {name: stickout, kind: project, path: "tools.list[number=1].stickout_mm", range: [28, 45], unit: mm}
options: {outputs: false, max_sim_s: 20}
execution: {workers: 1, memory: 1536M}
```

```
bin/twin sweep my-sweep.yaml --plan-only          # write the variants, run nothing
bin/twin sweep my-sweep.yaml --dataset            # run every point, then DATA_DIR/sweeps/<name>/dataset/
bin/twin sweep my-sweep.yaml                      # again: finished points are cache hits, only new ones run
bin/twin sweep --show DATA_DIR/sweeps/<name>      # what the folder says now
```

Options: `--out DIR` (default `DATA_DIR/sweeps/<slug of name>`), `--workers N` (each point in its own process
and memory cap: N x `--memory` must fit), `--isolation process|inline`, `--memory 1536M`, `--memory-mode`,
`--runner`, `--no-cache`, `--retry-failed`, `--dataset [DIR]`, `--json`, `--quiet`. A point that failed is not
retried under the same key without `--retry-failed`. Ctrl-C stops the running points and keeps the finished
ones. **One project at an override**: a sweep whose only parameter is
`{name: feed, kind: override, target: feed, tool: 1, values: [0.8]}` with `design: {kind: grid}` has exactly
one point. The API: `POST /api/sweeps {project_id, spec, options}` (docs/api.md "Sweeps").

## 9. Datasets

```
bin/twin dataset DATA_DIR/sweeps/<name> other-run-dir --out training/ [--format parquet,csv] [--no-blocks]
```

Reads sweep folders, run directories and folders of run directories and writes `runs.parquet/.csv` (one row
per run), `blocks.parquet/.csv` (one row per executed NC block: features and labels), `schema.json` (every
column's type, unit, provenance and meaning) and `manifest.json` (the runs, the points skipped and why, the
builds). Nothing is simulated again. Every column: docs/ml.md "Columns"; what the data is good for:
docs/ml.md "Limits".

## 10. Uncertainty studies

How the results scatter over the tolerances the project states (stock size, tool offsets, runout, material
batch, clamping force; the project's `uncertainty` and `material.batch` sections; docs/models/uncertainty.md):

```
bin/twin uncertainty PROJECT --out STUDY_DIR [--run RUN_DIR] [--runs N] [--method lhs|monte_carlo] [--seed S] [--vary-seed] [--plan-only] [--json]
```

It writes `plan.json`, one variant per sample under `variants/v<nnnn>/` (project and run), `study.json`
(format `cnctwin-uncertainty`: the distribution and capability of every inspected dimension, the probability
of each finding and of "not safe", the sensitivity ranking) and `uncertainty.html`. `--run` names the run the
study is made around: the study is attached to it (`RUN_DIR/uncertainty/`) and its report gets the section.
Running it again continues where it stopped. Each sample is a full simulation: `--runs` times the cost of one
run. API: `POST /api/runs/{id}/uncertainty {options}`; UI: the run's Uncertainty tab.

## 11. Calibration from machine data

Fits model coefficients to recordings of the real machine running the same program (docs/models/calibration.md):

```
bin/twin calibrate --project PROJECT --run RUN_DIR --kind csv --path log.csv \
    --map "time=t [s]" --map "spindle.power_W=P_spindle [kW]" --target spindle --out CAL_DIR
bin/twin calibrate --project PROJECT --run RUN_DIR --data machine_data.yaml --target servo --target block_time --out CAL_DIR
```

* Machine data: `--kind mtconnect | opcua | focas | csv | log` with `--path` (a file, a folder, an MTConnect
  agent URL or an `opc.tcp://` server) and `--map "CHANNEL=<column or node> [<unit>]"` entries (the unit in
  brackets, as written; channels such as `spindle.power_W`, `spindle.load_frac`, `axis.X.pos_machine_m`, and
  `time` for the time column), or `--data` a YAML/JSON file
  `{kind, path, mapping, start, end}`. Import alone: `bin/tsafe python -m cnctwin.mdata.importers --kind K --path P
  [--map ...] [--out DIR]`.
* `--target`: `spindle` (`kc_scale` of the material's cutting coefficients and the spindle's idle or bearing
  power, from spindle power or load), `servo`, `axis_friction`, `block_time`, `thermal_growth`, `tool_life`,
  `usui`, `deflection`, `kienzle`, `mechanistic`; default every target whose data are present.
  `--simulator auto | surrogate | loop` (`loop` re-simulates the program about 10 (p + 1) times),
  `--offset` a known time offset, `--options` further settings.
* Outputs in `--out`: `machine_log.json/.parquet`, `alignment.json`, `calibration.yaml` (the overlay, format
  `cnctwin-calibration`), `calibration.json` (parameters, intervals, correlations, metrics before and after),
  `overlay.json` (measured, simulated and calibrated series) and **`project.calibrated.yaml`**, a standalone
  project with the calibrated values in place (inline `extends` objects): run that file to simulate with the
  calibrated twin. In this build `project.yaml` has no key that includes a `calibration.yaml` (the project
  schema refuses one).
* The intervals cover random measurement error only; check the residual metrics in `calibration.json`, and
  validate on cuts that were not used for the fit.

API: `POST /api/calibrate {project_id, run_id, machine_data, options}`; UI: the run's Calibration tab.

## 12. Comparing two runs

```
bin/twin compare RUN_A RUN_B --out /tmp/compare [--json]
```

Writes `compare.html` and `compare.json` (format `cnctwin-compare`): the verdicts, the findings matched,
resolved and new, every inspected characteristic, the cycle time by category, line and tool, tools, energy and
cost per part, surface finish and which input files differ (docs/report.md section 8). API:
`GET /api/compare?a=<run>&b=<run>` (computed once, then served); UI: pick two runs on the runs page.

## 13. Validation and benchmarks

* `bin/twin validate` runs every published experiment in `validation/cases/` through its harness and rewrites
  docs/validation-report.md and validation/results/*.json; `--quantity Q`, `--case ID`, `--list`, `--no-write`,
  `--json` (docs/validation.md). Exit 1 when a budgeted quantity fails.
* `bin/twin bench [examples ...] [--all] [--max-sim-s S] [--outputs]` measures CPU seconds per simulated second
  per level of the loop. It **updates docs/performance.md** unless `--no-doc` is given.
* `bin/twin db check [--strict] [--estimates] [--migrations] [--rules] [--json]` validates the databases (the
  how-tos for adding data use it).

## 14. The service: API and UI

```
bin/twin serve --host 127.0.0.1 --port 5990 --data-dir /tmp/twin-data      # a local server (python -m cnctwin.api)
```

Options and environment: `--max-jobs` (`CNCTWIN_MAX_JOBS`, simulations at once, default 1), `--job-memory`
(`CNCTWIN_JOB_MEMORY`, 1536M), `--memory-mode auto|systemd|rlimit|none`, `--log-level`; every setting in
docs/api.md "Running the server". One server per `DATA_DIR`. On Plint the container runs it on port 5210 with
`DATA_DIR=/data` and one job at a time; `https://twin.<domain>` is the UI.

Every simulation, check, comparison, uncertainty study, calibration and sweep is a job in its own process
under the memory cap. A session with curl:

```
cd examples/pocket-islands-al6061 && zip -r /tmp/pocket.zip . && cd -
curl -s -F file=@/tmp/pocket.zip -F name=pocket localhost:5990/api/projects        # upload: project id "pocket"
curl -s -X POST localhost:5990/api/projects/pocket/check                             # twin check as a job
curl -s -X POST -H 'Content-Type: application/json' \
     -d '{"project_id": "pocket", "options": {"max_sim_s": 8}}' localhost:5990/api/runs   # 201, or 200 from the cache
curl -sN localhost:5990/api/runs/<run_id>/events                                     # progress, finding, keyframe, done
curl -s 'localhost:5990/api/runs/<run_id>/findings?severity=error'
curl -s localhost:5990/api/runs/<run_id>/trace/<event_id>                            # a cause chain
curl -s -X POST localhost:5990/api/runs/<run_id>/cancel                              # and .../resume
```

The UI (served at `/`) does the same with buttons: the projects page (examples and uploads, upload a zip), a
project (files, Check, Start a run with options), the runs list, the run page with its tabs (Overview,
Playback, Inspection, Surface, Tools, Production, Stability, Coolant, Chips, Thermal, Checks, Uncertainty,
Calibration, Reports & files, Run info) and the comparison page (docs/ui.md). A zip must have `project.yaml`
at its top (or inside one folder); uploads are at most 256 MB (`CNCTWIN_MAX_UPLOAD_MB`; Caddy allows 500 MB).

## 15. Exit codes

| code | `twin` command |
|---|---|
| 0 | success; `run`: the program is safe to run |
| 1 | an error (input error, failed run, `check` found errors, `validate` found a failing quantity) |
| 2 | a usage error |
| 3 | `run` finished and the program is **not** safe to run |
| 4 | the command's module is not in this build |
| 130 | interrupted (Ctrl-C) |
