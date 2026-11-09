# CNC Twin HTTP API

The API of architecture 17 (`cnctwin/api/`): projects, simulation runs (queued as jobs, followed
with Server-Sent Events, cancelled and resumed from checkpoints), everything a run writes, studies
(compare, uncertainty, calibration) and the databases. The UI (`web/`) is served at `/` and uses
nothing else. The machine-readable description is [`docs/openapi.json`](openapi.json), also served
at `/api/openapi.json`.

* SI inside and out: every quantity is SI and its name carries the unit (`sim_time_s`,
  `force_N`, `pos_W_m`); times are simulated seconds unless the name says wall (`wall_s`).
* JSON has no NaN: non-finite numbers (channel padding, undefined values) are `null`.
* Errors are JSON `{"detail": "what went wrong"}` with the HTTP status: 400 bad input (an invalid
  id or path), 403 a path that leads outside its folder, 404 not there (or not yet: a running run
  has no `inspection.json`), 405 wrong method, 409 not in a state for this, 413 too large, 415 wrong
  body type, 422 a body or project file that does not validate, 500 a failed job, 501 a job whose
  entry point is not in this build, 503 the job manager is not running.
* Ids (runs, projects, jobs, bricks, tools) are `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`.

Contents: [running the server](#running-the-server) · [endpoints](#endpoints) ·
[examples](#examples) · [server-sent events](#server-sent-events) · [series and M4](#series-and-m4-downsampling) ·
[files and ranges](#files-range-requests-and-safety) · [jobs](#the-job-manager) ·
[entry points](#entry-points-what-the-worker-calls) · [run directory](#run-directory-as-read-by-the-api) ·
[run cache](#run-cache) · [sweeps](#sweeps) · [decisions](#decisions-and-their-consequences) · [OpenAPI](#openapi)

## Running the server

```
python -m cnctwin.api --host 127.0.0.1 --port 5210 --data-dir /data     # or: cnctwin.api.serve(argv)
bin/tsafe python -m cnctwin.api --port 5990 --data-dir /tmp/twin-data   # a test server on this box
```

| option | environment | default |
|---|---|---|
| `--data-dir` | `DATA_DIR` | `cnctwin.paths.data_dir()` (`<root>/runs-data`) |
| `--host` | `CNCTWIN_HOST` | `127.0.0.1` |
| `--port` | `CNCTWIN_PORT` (else `PORT`) | `5210` |
| `--max-jobs` | `CNCTWIN_MAX_JOBS` | `1` simulation (heavy job) at a time |
| | `CNCTWIN_MAX_LIGHT_JOBS` | `1` check or comparison at a time |
| `--job-memory` | `CNCTWIN_JOB_MEMORY` | `1536M` per heavy job (`CNCTWIN_LIGHT_JOB_MEMORY`: `1024M`) |
| | `CNCTWIN_JOB_SWAP` | `256M` (systemd mode, as bin/tsafe) |
| `--memory-mode` | `CNCTWIN_JOB_MEMORY_MODE` | `auto`: `systemd` when available, else `rlimit` (or `none`) |
| | `CNCTWIN_CANCEL_GRACE_S`, `CNCTWIN_KILL_GRACE_S` | `60`, `10` |
| | `CNCTWIN_MAX_UPLOAD_MB`, `CNCTWIN_MAX_EXTRACT_MB` | `256`, `1024` |
| | `CNCTWIN_EXAMPLES_DIR`, `CNCTWIN_WEB_DIR`, `CNCTWIN_DB_DIR` | `examples/`, `web/`, `db/` of the root |
| | `CNCTWIN_ENTRY_<KIND>` | entry point of a job kind ([below](#entry-points-what-the-worker-calls)) |
| | `CNCTWIN_NUMBA_CACHE_BASE` | `<root>/.numba-cache` (`DATA_DIR/numba-cache` if read-only) |
| | `SOURCE_DATE_EPOCH` | `315532800` (1980-01-01) passed to every job |
| `--log-level` | `CNCTWIN_LOG_LEVEL` | `info` |

The server is one uvicorn process (the job manager lives in it). One server per DATA_DIR (an flock
on `DATA_DIR/.api.lock`; a second one refuses to start). On shutdown open event streams are
closed after 5 s; running jobs keep running and are adopted by the next server (or, if their
process died meanwhile, marked failed and resumable). DATA_DIR holds:

```
DATA_DIR/projects/<id>/      uploaded projects (examples stay in examples/)
DATA_DIR/runs/<run_id>/      run directories (architecture 16)
DATA_DIR/jobs/<job_id>/      job.json (manager), worker.json (worker), cancel, log.txt, result.json, out/
DATA_DIR/tmp/                uploads and exports in progress
```

There are no `/docs` or `/redoc` pages: they load scripts from a CDN and the box runs offline.

## Endpoints

| method and path | what | answers |
|---|---|---|
| `GET /api/health` | version, data folder, memory mode, job counts and caps, which entry points exist, series reader, database backend, numba cache key, UI installed | 200 |
| `GET /api/projects` | examples (always, read-only) then uploads | 200 |
| `POST /api/projects?name=` | upload a zipped project (body: the zip, or multipart field `file`) | 201, 400, 413, 415, 422 |
| `GET /api/projects/{id}` | project.yaml as written, operations, files | 200, 404 |
| `GET /api/projects/{id}/files/{path}` | a project file (ranges) | 200/206, 400, 403, 404 |
| `POST /api/projects/{id}/check?wait_s=20` | `twin check` as a job; body `{options}` | 200 result, 202 job, 409, 500, 501 |
| `POST /api/runs` | queue a simulation: `{project_id, options, no_cache}`; a run the [run cache](#run-cache) holds is answered instead | 201 (Location), 200 cached run, 404, 422 |
| `GET /api/runs?project_id=&status=&limit=&offset=` | runs, newest first | 200 |
| `GET /api/runs/{id}` | `{id, run: run.json or null, status: status.json, job, checkpoint, links}` | 200, 404 |
| `GET /api/runs/{id}/events?timeout_s=` | SSE: `progress`, `finding`, `keyframe`, `done` | 200 stream |
| `POST /api/runs/{id}/cancel` | stop a queued or running simulation | 202, 409 |
| `POST /api/runs/{id}/resume` | continue a cancelled or failed one from its last checkpoint | 202, 409 |
| `GET /api/runs/{id}/findings?severity=&category=&op=` | findings (filtered), worst severity, verdict | 200 |
| `GET /api/runs/{id}/trace/{event_id}?max_events=` | a trace event and its chain of causes | 200, 404 |
| `GET /api/runs/{id}/channels?tier=` | channels.json | 200 |
| `GET /api/runs/{id}/series?tier=&channels=&t0=&t1=&max_points=` | channel values, M4 above max_points | 200, 400, 404 |
| `GET /api/runs/{id}/export?format=csv\|parquet&tier=&channels=&t0=&t1=` | full resolution | 200 file |
| `GET /api/runs/{id}/program/{op}` | the program text of an operation (name or index) | 200, 404, 409 |
| `GET /api/runs/{id}/timeline?op=&t0=&t1=` | executed blocks with tick ranges | 200 |
| `GET /api/runs/{id}/scene` | scene.json (incl. the frames.bin header) | 200 |
| `GET /api/runs/{id}/frames?t0=&t1=&format=bin\|json&max_frames=` | frames.bin records | 200 |
| `GET /api/runs/{id}/keyframes` | keyframes/index.jsonl | 200 |
| `GET /api/runs/{id}/keyframes/{k}.glb` | changed in-process-workpiece bricks of keyframe k | 200/206 |
| `GET /api/runs/{id}/keyframes/{k}/fields?names=&format=json\|npz` | coarse fields of keyframe k | 200 |
| `GET /api/runs/{id}/bricks/{brick}?t=` | one brick at time t (one ranged read) | 200, 404 |
| `GET /api/runs/{id}/chips?t=&names=&format=` | chips in flight and pile at the keyframe at or before t | 200 |
| `GET /api/runs/{id}/jets?t=` | coolant jets at the keyframe at or before t, and the scene's nozzles | 200 |
| `GET /api/runs/{id}/sld` · `/sld/{tool}` | tools with a stability lobe diagram · one (`1` finds `T1`) | 200, 404 |
| `GET /api/runs/{id}/inspection` · `/surface` · `/tools` · `/production` | the result files | 200, 404 |
| `GET /api/runs/{id}/files` · `/files/{path}` | list · any file of the run (ranges, HEAD) | 200/206, 400, 403, 404, 416 |
| `GET /api/compare?a=&b=&wait_s=&retry=` | compare two finished runs (job, result kept) | 200, 202, 409, 500, 501 |
| `POST /api/runs/{id}/uncertainty` | uncertainty study around a run: `{options}` | 202 job, 409 |
| `POST /api/calibrate` | `{project_id?, run_id?, machine_data?, options}` | 202 job, 400, 404 |
| `POST /api/sweeps` | a parameter sweep of a project as a job: `{project_id, spec, options}` ([sweeps](#sweeps)) | 202 job, 400, 403, 404, 422 |
| `GET /api/sweeps` · `/api/sweeps/{id}` | sweeps, newest first · one: job, progress, summary, every point | 200, 404 |
| `GET /api/jobs?kind=&state=&limit=` · `/api/jobs/{id}` | jobs, newest first · one, with its result | 200, 404 |
| `POST /api/jobs/{id}/cancel` · `GET /api/jobs/{id}/events` | cancel · SSE `progress`, `done` | 202, 409 · stream |
| `GET /api/jobs/{id}/files` · `/files/{path}` · `/log` | what the job wrote · its output (ranges) | 200, 404 |
| `GET /api/db` · `/api/db/{kind}` · `/api/db/{kind}/{id}` | kinds · items · one item with SI view, problems, sources | 200, 404 |
| `GET /api/docs/{path}` | a Markdown document of the build's `docs/` (`approximations.md`, `models/loop.md` ...): the report links the full approximation table here | 200, 403, 404 |
| `GET /` · `GET /{path}` | the UI: `web/` files, `index.html` for client-side routes | 200, 404 |

Added to architecture 17 (additive): `/api/jobs/*` (checks and studies are jobs that need a
handle), `/api/runs/{id}/files` and `/api/runs/{id}/sld` listings, `/api/db` (the kinds), the
`timeout_s` and `Last-Event-ID` stream options, `max_frames`, `format=json|npz` on arrays, `t0/t1`
on export and timeline, the HEAD variants of file routes, `/api/docs/{path}`, the run cache of `POST /api/runs`
(`no_cache`, the 200 answer, `cache` in the record) and `/api/sweeps`.

## Run cache

`POST /api/runs` looks the run up in the run cache (`DATA_DIR/cache`, cnctwin.cache, docs/ml.md "Run cache")
before it queues a simulation: the key of the job it would start (the project, its inputs and databases, the
options, the `simulate` entry, the build, the platform, the jobs' environment and SOURCE_DATE_EPOCH). A verified
finished run in the runs folder is answered with **200** (`Location` the run, `cache: {hit: true, key, run_id}`);
one held elsewhere (a sweep's point in the cache's own store) is copied into a new run folder first (`cache:
{hit: true, key, run_id, copied_from}`, the copy is registered too). Otherwise the answer is 201 as before with
`cache: {hit: false, key}` (or a `note` saying why there is no key: a project that does not resolve, the cache
switched off with CNCTWIN_CACHE=off) and the simulation's worker registers the finished run when its key did not
change while it ran (the job's result `cache: {registered, key, note}`). `no_cache: true` skips the lookup; the
new run is registered and supersedes. A cache problem never fails the request.

## Sweeps

`POST /api/sweeps {project_id, spec, options}` starts a sweep (cnctwin.sweep, docs/ml.md "Sweeps", the spec's
format in docs/formats.md "Sweep") as a heavy job (kind `sweep`, one at a time with the simulations): the
project is the base (a `base` in the spec is ignored), generator scripts and templates must lie in the project's
folder (403/400 otherwise), a spec that does not validate is 422. A request may not name `execution.runner` (422:
the points run the server's own simulation) and may use generators (Python from the project's folder) only for
the example projects shipped with the server (403 for an uploaded project, unless the server runs with
CNCTWIN_ALLOW_UPLOAD_GENERATORS=1). The job's output folder is the sweep folder:
the points run one after another through the run cache (each in its own process inside the job's memory cap;
`options.isolation: inline` runs them in the job's process), then the dataset is exported into `dataset/`
(`options.dataset: false` skips it; `options.no_cache`, `options.retry_failed` as `twin sweep`). The points'
runner is the server's `simulate` entry when CNCTWIN_ENTRY_SIMULATE overrides the default loop, else
cnctwin.sweep.runner (the loop plus the operator overrides). `GET /api/sweeps/{id}` (the job id): `{id, name, project_id, job,
status (status.json: progress, ETA), summary (summary.json), notes, points [{index, values, status, key, run_id,
run_dir, safe, error}], result, links}`; every file at `/api/jobs/{id}/files/...` (`links.dataset`,
`links.blocks`, `links.schema`). Cancel with `POST /api/jobs/{id}/cancel`: running points stop, finished ones
stay cached, a new `POST` with the same spec resumes.

## Examples

Responses below are from the synthetic run (`tests/fixtures/make_synthetic_run.py`, seed 4),
shortened where marked `...`.

### Health

```
$ curl -s localhost:5210/api/health
{"status": "ok", "version": "1.0.0", "format_version": 1, "data_dir": "/data", "memory_mode": "systemd",
 "jobs": {"queued": 0, "running": 0, "cancelling": 0, "max_heavy": 1, "max_light": 1,
          "heavy_memory_bytes": 1610612736, "light_memory_bytes": 1073741824},
 "entry_points": {"simulate": {"entry": "cnctwin.sweep.runner:run_project", "module_found": true}, ...},
 "series_reader": "pyarrow", "db_backend": "cnctwin.db.database", "numba_cache_key": "f47d5bf4d6c1c2c7",
 "ui": true}
```

### Upload a project, check it, run it

```
$ (cd pocket && zip -r ../pocket.zip .)
$ curl -s -X POST --data-binary @pocket.zip -H 'Content-Type: application/zip' localhost:5210/api/projects
{"id": "synthetic-pocket-al-6061-t6", "source": "upload", "read_only": false,
 "name": "Synthetic pocket, Al 6061-T6", "machine": "haas-vf2ss", "material": "al6061-t6",
 "operations": [{"name": "OP10", "program": "op10.nc", "program_found": true}], "tools": 1, "error": null,
 "project": {...project.yaml...}, "files": [{"path": "op10.nc", "bytes": 330}, {"path": "project.yaml", ...}],
 "skipped": [], "problems": []}
$ curl -s -F file=@pocket.zip -F name=pocket localhost:5210/api/projects          # from a form: id "pocket"

$ curl -s -X POST localhost:5210/api/projects/pocket/check
{"job": {"id": "j000002", "kind": "check", "state": "done", "pool": "light", ...}, "result": {...}}

$ curl -s -X POST -H 'Content-Type: application/json' -d '{"project_id": "pocket", "options": {}}' \
       localhost:5210/api/runs
{"id": "00002-pocket-3f1c9a2e", "run": null,
 "status": {"format": "cnctwin-run-status", "version": 1, "status": "queued",
            "progress": {"sim_time_s": 0.0, "est_total_s": null, "wall_s": 0.0, "eta_s": null},
            "job_id": "j000001", "attempt": 1, "error": null, "checkpoint": null, "resume_from": null,
            "queue_position": 0},
 "job": {"id": "j000001", "kind": "simulate", "state": "queued", "pool": "heavy", "run_id": "00002-pocket-3f1c9a2e",
         "project_id": "pocket", "queue_position": 0, "attempt": 1, "memory_mode": "systemd",
         "memory_bytes": 1610612736, "links": {"self": "/api/jobs/j000001", ...}, ...},
 "checkpoint": null, "links": {"self": "/api/runs/00002-pocket-3f1c9a2e", "events": "/api/runs/00002-pocket-3f1c9a2e/events", ...}}
```

### Follow it

```
$ curl -sN localhost:5210/api/runs/00002-pocket-3f1c9a2e/events
retry: 2000

event: progress
id: f0k0
data: {"format":"cnctwin-run-status","version":1,"status":"running","progress":{"sim_time_s":0.01,"est_total_s":2.32,"op":"OP10","checkpoint":null,"wall_s":0.153,"eta_s":null},...}

event: keyframe
id: f0k1
data: {"bricks":[0,1,2,3,4,5],"chips":"chips/kf_0.npz","fields":"keyframes/kf_0.npz","full":true,"glb":"keyframes/kf_0.glb","k":0,"tick":0,"time_s":0.0}

event: finding
id: f1k1
data: {"category":"air_cutting","severity":"info","message":"Feed motion out of contact for 0.100 s ...","time_s":0.442,"tick":442000,"op":"OP10","program":"op10.nc","line":10,"block":"G01 Z-2. F1200.","pos_W_m":[...],"pos_M_m":[...],"quantities":{...},"cause_chain":[2,0],"id":0}
...
event: done
id: f5k5
data: {"status":"done","verdict":{"safe":false,"reasons":["out_of_tolerance (error) at OP10 line 17: pocket-depth 2.062 mm, tolerance 2 +/- 0.05 mm (tool growth 12 um)"]},"error":null,"findings":5,"keyframes":5}
```

In a browser: `new EventSource("/api/runs/00002-pocket-3f1c9a2e/events")` with listeners for `progress`,
`finding`, `keyframe` and `done` (close it on `done`; EventSource reconnects by itself otherwise,
sending `Last-Event-ID`).

### Cancel and resume

```
$ curl -s -X POST localhost:5210/api/runs/00002-pocket-3f1c9a2e/cancel      # 202: status cancelling, then cancelled
$ curl -s -X POST localhost:5210/api/runs/00002-pocket-3f1c9a2e/resume      # 202: attempt 2, resume_from the last checkpoint
{"id": "00002-pocket-3f1c9a2e", "status": {"status": "queued", ...},
 "job": {"id": "j000003", "attempt": 2, "resume_from": "/data/runs/00002-pocket-3f1c9a2e/checkpoints/ck_1000000.json", ...},
 "checkpoint": {"tick": 1000000, "file": "checkpoints/ck_1000000.json", "time_s": 1.0, "op": "OP10"}, ...}
```

### Findings and their causes

```
$ curl -s 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/findings?severity=error'
{"run_id": "00002-pocket-3f1c9a2e", "total": 5, "count": 1, "worst": "error", "safe": false,
 "findings": [{"category": "out_of_tolerance", "severity": "error",
               "message": "pocket-depth 2.062 mm, tolerance 2 +/- 0.05 mm (tool growth 12 um)",
               "time_s": 2.161, "tick": 2161000, "domain": "inspect", "op": "OP10", "program": "op10.nc",
               "line": 17, "block": "G01 X0.", "pos_W_m": [0.01005, 0.021, 0.013], "pos_M_m": [-0.38095, -0.1912, -0.4084],
               "quantities": {"deviation_m": 6.2e-05, "measured_m": 0.002062, "nominal_m": 0.002},
               "cause_chain": [30, 29, 10, 28, 8, 25, 3, 0], "id": 4}]}
$ curl -s localhost:5210/api/runs/00002-pocket-3f1c9a2e/trace/30
{"event": {"domain": "inspect", "event": "finding:out_of_tolerance", "id": 30, "line": 17, "parents": [29], ...},
 "chain": [{"id": 30, ...}, {"id": 29, "event": "floor_position", "parents": [10, 28], ...},
           {"id": 10, "event": "thermal_growth", "domain": "thermal", ...}, ...]}
```

### Series

```
$ curl -s 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/series?tier=tick&channels=cutting.spindle_power_W&t0=1.0&t1=1.002'
{"run_id": "00002-pocket-3f1c9a2e", "tier": "tick", "tick_s": 1e-06, "t0_s": 1.0, "t1_s": 1.002, "rows": 3,
 "max_points": 2000, "downsampled": false, "method": "raw", "buckets": null,
 "series": [{"column": "cutting.spindle_power_W", "channel": "cutting.spindle_power_W", "unit": "W", "n": 3,
             "t_s": [1.0, 1.001, 1.002], "v": [1120.68, 1153.60, 1185.13]}]}
$ curl -s 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/series?tier=tick&channels=cutting.spindle_power_W&max_points=8'
{..., "rows": 2320, "max_points": 8, "downsampled": true, "method": "m4", "buckets": 2,
 "series": [{"column": "cutting.spindle_power_W", ..., "n": 7,
             "t_s": [0.0, 0.982, 1.159, 1.16, 1.373, 2.162, 2.319],
             "v": [0.0, 1192.0, 506.9, 525.1, 1191.8, 0.0, 0.0]}]}
$ curl -s -o tick.csv 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/export?format=csv&tier=tick&channels=cutting.force_N'
```

### Scene, frames, keyframes

```
$ curl -s 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/frames?t0=1.0&t1=1.001&format=json'
{"run_id": "00002-pocket-3f1c9a2e", "count": 2, "stride": 1, "tick_s": 1e-06,
 "header": {"format": "cnctwin-frames", "record_bytes": 52, "byte_order": "little", "tick_s": 1e-06, "ipo_ticks": 1000,
            "axes": [{"name": "X", "kind": "linear", "unit": "m"}, ...],
            "fields": [{"name": "tick", "dtype": "<i8", "shape": [], "offset": 0}, {"name": "axis", "dtype": "<f4", "shape": [3], "offset": 8}, ...]},
 "fields": {"tick": [1000000, 1001000], "axis": [[0.026, 0.0109, 0.013], [0.026, 0.01095, 0.013]], ...}}
$ curl -s -D- -o frames.bin 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/frames?t0=0.5&t1=0.6'
x-frame-count: 101  x-frame-stride: 1  x-record-bytes: 52  x-first-tick: 500000  x-tick-s: 1e-06
$ curl -s -D- -o b0 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/bricks/0?t=1.2'
x-keyframe: 2  x-keyframe-time-s: 1.0  x-brick: 0  x-source-file: kf_2.glb  x-source-offset: 1236  content-length: 236
$ curl -s 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/keyframes/3/fields?names=tool_deflection_m,jet_kind'
{"k": 3, "time_s": 1.5, "file": "keyframes/kf_3.npz",
 "arrays": {"tool_deflection_m": {"dtype": "float64", "shape": [3], "data": [2.795e-06, 4.985e-06, 7.73e-07]},
            "jet_kind": {"dtype": "str", "shape": [1], "data": ["flood"]}}}
$ curl -s 'localhost:5210/api/runs/00002-pocket-3f1c9a2e/jets?t=1.2'
{"k": 2, "time_s": 1.0, "jets": [{"nozzle_id": "n1", "kind": "flood", "origin_M_m": [0.08, 0.05, -0.25],
  "direction_M": [-0.6, -0.4, -0.69], "speed_m_s": 3.2, "flow_m3_s": 0.00025, "half_angle_rad": 0.12,
  "reach_fraction": 0.82}], "nozzles": [{"id": "n1", "kind": "flood", ...}]}
```

### Video with a range request, and the database

```
$ curl -s -H 'Range: bytes=0-1023' -o head.mp4 -w '%{http_code}\n' localhost:5210/api/runs/00002-pocket-3f1c9a2e/files/video.mp4
206
$ curl -s localhost:5210/api/db/holders/haimer-a63-000-16
{"backend": "cnctwin.db.database", "kind": "holder", "id": "haimer-a63-000-16", "file": "db/holders/haimer-weldon.yaml",
 "files": ["db/holders/haimer-weldon.yaml"], "data": {...as written...}, "si": {"bore_m": 0.016, ...},
 "si_error": null, "problems": [],
 "sources": {"haimer-shop-a63-000-16": {"type": "catalogue", "authors": "HAIMER GmbH", ...}}, "unresolved_sources": []}
```

## Server-Sent Events

`GET /api/runs/{id}/events` tails the run directory every `sse_poll_s` (0.25 s):

| event | when | data |
|---|---|---|
| `progress` | status.json changed (first event: the current status) | status.json (+ `queue_position` while queued) |
| `finding` | a finding appeared in findings.json | the finding (Finding.to_dict) |
| `keyframe` | a line appeared in keyframes/index.jsonl | the keyframe entry |
| `done` | status is done, failed or cancelled (after a last read of findings and keyframes) | `{status, verdict, error, findings, keyframes}` |

Each event has `id: f<findings sent>k<keyframes sent>`; reconnecting with `Last-Event-ID` (or
`?last_event_id=`) sends only what is new. After a resume the run rewrites what came after its
checkpoint (identically), so a client that reconnects may see a finding again: key findings by
their `id`. A `: keep-alive` comment goes out after 15 s of silence; `X-Accel-Buffering: no`
keeps proxies from buffering. `?timeout_s=` ends the stream after that long (reconnect to go
on). `GET /api/jobs/{id}/events` sends `progress` (the job record) and `done`.

## Series and M4 downsampling

`channels=` takes channel names or columns, comma-separated. A channel name selects every column
of it: `cutting.force_N` gives `cutting.force_N[0]`, `[1]`, `[2]` and, in the tick tier, each
column's `.min`, `.mean`, `.max`; an exact column (`cutting.force_N[1].max`) selects that one. At
most 256 columns per request. `t0`, `t1` (simulated seconds, inclusive) become ticks by rounding
`t / tick_s`; times in the response are `tick / ticks_per_second` (exact decimals).

Up to `max_points` rows (default 2000, 4 to 200 000) come back as recorded. Above that, every
column is downsampled with **M4** (Jugel, Jerzak, Hackenbroich and Markl, *M4: A
Visualization-Oriented Time Series Data Aggregation*, PVLDB 7(10), 797-808, 2014,
doi:10.14778/2732951.2732953): the time range is split into `max_points / 4` equal buckets, and
each bucket keeps its first, last, minimum and maximum sample with their own times (at most 4
points, in time order). Validity and error: drawn as a line with one bucket per pixel column
the result rasterises exactly like the full series (the paper's error-free line rendering); with
fewer buckets than pixels every bucket's extremes are still exact, so peaks (a force spike, a
temperature maximum) are never smoothed away; every point returned is a recorded sample. NaN
samples are left out of the extremes; a bucket with samples but none finite gives one NaN
(`null`) point, so a plot breaks the line over gaps. The fast tier holds samples only inside its
recording windows: break lines where the tick step jumps. The accumulator is streaming (one
Parquet part at a time, O(buckets) memory), so an hour of tick data is one pass.

`/export` returns full resolution: CSV (streamed part by part; `tick,time_s,<columns>`, header
quoted, NaN as an empty field, floats at their shortest round-trip digits) or one Parquet file
(zstd; NaN as null). Without `channels` every column of the tier.

## Files, range requests and safety

Files of runs, projects and jobs go out through Starlette's `FileResponse`: `Accept-Ranges:
bytes`, single ranges (`bytes=100-199`, `bytes=-50`, `bytes=500-`), multiple ranges
(`multipart/byteranges`), `If-Range` with the ETag, 416 for a range past the end, HEAD. Media
types are set for `.glb .gltf .stl .3mf .obj .step .npz .parquet .jsonl .wav .mp4 .png .svg .pdf
.html .csv .yaml` and the CNC program extensions (text/plain). HTML and SVG from runs, projects
and jobs get `Content-Security-Policy: sandbox` (displayable, but no scripts and no same-origin
access), everything `X-Content-Type-Options: nosniff`; the UI's own files are served as they are.

A relative path is refused (400) if it is empty or absolute, has a drive letter, a backslash, a NUL
byte, or an empty, `.` or `..` segment; after that it is resolved with its symbolic links and
refused (403) if it lands outside its folder. Listings leave such links out.

Uploads are streamed to disk with a size cap (413 above `max_upload_bytes`). Extraction refuses,
before writing anything: absolute names, `..`, drive letters, symbolic links, encrypted members,
duplicate names, a name that is both a file and a folder, more than 20 000 members, more than
1 GiB in total, members above 1 MiB that expand more than 200 times; while writing it counts again
(a header can lie). `__MACOSX/`, `.DS_Store` and `._*` are dropped and listed in `skipped`. A zip
of one folder is unpacked from inside that folder. The archive must have `project.yaml` (format
`cnctwin-project`) at its top; it is parsed with cnctwin.db.yamlio (422 with the line on a YAML
error). Ids: `?name=` (or a multipart field `name`), else the project's `name:`, else the zip's
file name, slugged and made unique (`-2`, `-3`); an upload id never starts with `example-` (a
`p-` is put in front), so it cannot shadow an example.

## The job manager

Every simulation, check, comparison, uncertainty study and calibration is a job in its own
process: `python -m cnctwin.api.worker DATA_DIR/jobs/<id>`.

* **Pools.** `heavy` (simulate, uncertainty, calibrate, sweep; default 1 at a time) and `light` (check,
  compare; default 1). Within a pool jobs start in submission order; `queue_position` counts the
  jobs ahead. A job that has reported its end keeps its slot until its process exits.
* **Memory cap.** `systemd` (when systemd runs and root or passwordless sudo may call it):
  `sudo -n systemd-run --scope -q -p MemoryMax=<cap> -p MemorySwapMax=256M --unit=cnctwin-job-<hash>-<id>
  --uid=.. --gid=.. /usr/bin/env -i <environment> python -m cnctwin.api.worker <dir>`, a cgroup cap on
  the job and everything it starts, exactly as bin/tsafe. `rlimit` (containers): the worker sets
  RLIMIT_DATA to the cap before importing anything, so an allocation beyond it raises MemoryError
  and the job fails saying so ("the job ran out of memory (cap 1536 MiB)"). `none`: no cap.
* **Environment** (built from nothing; only HOME is taken over): `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS` = 1, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`,
  `NUMBA_CACHE_DIR` keyed as bin/tsafe keys it (sha256 over the `sha256sum` lines of every cnctwin
  source file with a kernel, C-sorted, the numba/llvmlite versions and the CPU model; first 16 hex;
  the server's own NUMBA_CACHE_DIR when it has one, as under bin/tsafe), `SOURCE_DATE_EPOCH`,
  `TZ=UTC`, `LC_ALL=C.UTF-8`, `MPLBACKEND=Agg`, `DATA_DIR`, `CNCTWIN_ROOT`, `CNCTWIN_JOB_DIR`,
  `PYTHONPATH` (the source tree the server runs), and passed through if set:
  `NUMBA_ENABLE_CUDASIM`, `NUMBA_BOUNDSCHECK`, `CUDA_VISIBLE_DEVICES`. The worker runs at nice 10
  in its own session (so a signal reaches everything it starts).
* **States.** `queued` -> `running` -> `done` | `failed` | `cancelled` (`cancelling` between a
  cancel request and the end). A queued job is cancelled at once. A running job gets a `cancel`
  file; the worker notices within `status_interval_s` and its progress callback raises
  [`Cancelled`](#entry-points-what-the-worker-calls); after `cancel_grace_s` (60 s) the process
  group gets SIGTERM (the worker turns it into Cancelled too), after `kill_grace_s` (10 s) more
  SIGKILL (and in systemd mode `systemctl kill` on the scope).
* **status.json of a simulation** is written by the manager while queued and when a job ended
  without reporting (killed, crashed: `WorkerDied` with the exit status or signal, e.g. "the job
  process was killed by SIGKILL (it may have reached its memory cap of 1536 MiB)", plus the log
  tail), and by the worker while it runs: `status` (running, cancelling, done, failed, cancelled),
  `progress` (whatever the loop reports, `wall_s` measured by the worker and summed over attempts,
  `eta_s` from the rate of this attempt unless the loop gives one), `op`, `phase`, `message`,
  `checkpoint`, `error` (`{type, message, traceback}`), `job_id`, `attempt`, `resume_from`. The
  worker writes its `worker.json` (final) before the last status.json, so when a run reads as
  ended its job has ended too.
* **Resume** (`POST /api/runs/{id}/resume`, for cancelled and failed runs): a new job (attempt
  n + 1) with the same project and options and `resume_from` = the newest committed checkpoint
  (`checkpoints/ck_<tick>.json` with its `.npz`), or none (the run starts over) when there is no
  checkpoint. A finished run is not resumed (409).
* **Restart.** job.json and worker.json survive the server: queued jobs are queued again; a
  running job whose process is still alive (`/proc/<pid>/cmdline` names the worker and its folder)
  is adopted; one whose process is gone gets its final report if it wrote one, else it is marked
  failed ("the server stopped while the job was running") and can be resumed.
* **Compare** results are kept: the job key is the two run ids and the sha256 of both run.json
  files, so asking again returns the stored result at once (`retry=true` re-runs a failed one).

## Entry points (what the worker calls)

| kind | default | call |
|---|---|---|
| simulate | `cnctwin.sweep.runner:run_project` (the loop's `run_project`, with the operator overrides of `options.operator` applied) | `run_project(project_path: Path, run_dir: Path, options: dict, progress_cb)` |
| check | `cnctwin.project.check:check_project` | `check_project(request: dict, out_dir: Path, progress_cb) -> dict` |
| compare | `cnctwin.report.compare:compare_runs` | `compare_runs(request, out_dir, progress_cb) -> dict` |
| uncertainty | `cnctwin.uncertainty:run_study` | `run_study(request, out_dir, progress_cb) -> dict` |
| calibrate | `cnctwin.mdata.calibrate:calibrate` | `calibrate(request, out_dir, progress_cb) -> dict` |
| sweep | `cnctwin.sweep:run_sweep_job` | `run_sweep_job(request, out_dir, progress_cb) -> dict` |

Override any with `CNCTWIN_ENTRY_<KIND>=module:function` or `/path/to/file.py:function`. When the
module or function is not in the build, the job fails with type `EntryPointMissing` and a message
such as `cnctwin.core.loop.run_project is not available in this build (no module
'cnctwin.core.loop')`; the run's status.json says `failed` with that error, job endpoints answer
501. `/api/health` shows for each kind whether its module exists (without importing it).

* `project_path` is the project's `project.yaml`; `run_dir` exists and holds the API's
  status.json (the loop must not delete it; it may overwrite everything else);
  `options` is the client's `options` object, unchanged except `operator` (normalised by
  `POST /api/runs`: entries at 100 % dropped, nothing overridden: no option; a malformed value is a 422),
  plus `resume_from` (absolute path of the checkpoint marker `.json`) when resuming. The default entry
  applies `operator` (the loop's run_project does not read it) and refuses overrides the machine or the
  project's tools cannot take (the job fails with that message); a custom `CNCTWIN_ENTRY_SIMULATE` gets
  the option as it is. The return value (JSON-able) is kept as the job's `result.json` (format
  `cnctwin-job-result`, "API data folder" in docs/formats.md).
* `request` of the other kinds: `check` {project_id, project_path, project_dir, read_only, options}
  (`options.write` is refused for examples); `compare` {run_a, run_b, run_a_dir, run_b_dir, options};
  `uncertainty` {run_id, run_dir, project_id, project_path, project_dir, options};
  `calibrate` {project_id, project_path, project_dir, run_id, run_dir, machine_data, options};
  `sweep` {project_id, project_path, project_dir, spec, options, simulate_entry}.
  Every request also has `out_dir` (the job's output folder, served at `/api/jobs/{id}/files`).
* `progress_cb(progress: dict | None = None, **fields) -> None` takes `sim_time_s`, `est_total_s`,
  `eta_s`, `op`, `phase`, `message`, `checkpoint` ({tick, file}) and any other JSON-able fields
  (kept in `progress`). After a cancel request it raises `cnctwin.api.worker.Cancelled`, a
  **BaseException**: an `except Exception` in the loop does not catch it; `finally` blocks run.
  Call it once per macro step (or more often); a loop that never calls it can only be stopped by
  SIGTERM/SIGKILL after the grace periods.

## Run directory as read by the API

All reading is in `cnctwin/api/runfiles.py` and `cnctwin/api/series.py`. These are the layouts the
API expects (tests/fixtures/make_synthetic_run.py writes them, deterministically from a seed):

* A folder of `DATA_DIR/runs/` is a run when it has `run.json` or `status.json`; without
  status.json a run with run.json reads as `done` (`synthesized: true`).
* `run.json`: `verdict {safe, reasons}`, `summary`, `project` (`name`; `operations[]` with `name`,
  `program`, `program_text`, `program_sha256`), `inputs {file: {sha256, bytes}}`, `project_dir`,
  `clock.tick_s`. `/program/{op}` uses `program_text`, else reads `project_dir/<program>` (or the
  job's project folder) and refuses (409) if its sha256 differs from the recorded one.
* `status.json`: `{status, progress {sim_time_s, est_total_s, wall_s, eta_s}, ...}`.
* `findings.json`: `{"findings": [...]}` or a bare list, in id order.
* `trace.jsonl`: one TraceEvent per line, `id` equal to the line number (cnctwin.core.trace);
  `/trace/{id}` bisects byte offsets (no index in memory) and ignores a line still being written;
  the chain is breadth first, each event once, like TraceLog.chain.
* `channels.json`: `{"channels": [...]}` or a list; `timeline.json`: `{"blocks": [...]}` or a list.
* `series/index.jsonl` and `series/<tier>/part-<n>.parquet` with an int64 `tick` column (parts
  without an index are read in part-number order; a part's tick range from the index, else from
  its Parquet statistics).
* `scene.json["frames"]`: `{record_bytes, fields: [{name, dtype, shape, offset}], tick_s, ...}` with
  a `tick` field; `frames.bin` records in tick order (a record still being written is ignored).
* `keyframes/index.jsonl`: `{k, tick, time_s, glb, fields, chips, ...}`; missing names default to
  `keyframes/kf_<k>.glb`, `keyframes/kf_<k>.npz`, `chips/kf_<k>.npz`. `keyframes/bricks.idx`: JSON
  `{"bricks": {"<brick>": [[k, file, offset, length], ...]}}` (or JSON lines
  `{brick, k, file, offset, length}`), file relative to `keyframes/`. `/bricks/{brick}?t=` takes
  the newest entry whose keyframe time is at or before t and returns those bytes as they are.
* Coolant jets are the `jet_*` arrays of `keyframes/kf_<k>.npz` (`jet_nozzle_id`, `jet_kind`,
  `jet_origin_M_m`, `jet_direction_M`, `jet_speed_m_s`, `jet_flow_m3_s`, `jet_half_angle_rad`,
  `jet_reach_fraction`; one row per jet): `/jets` returns one object per row with the prefix
  dropped. `.npz` files are read with `allow_pickle=False`.
* `checkpoints/ck_<tick>.json` (commit marker) next to `ck_<tick>.npz`: committed checkpoints.
* `sld/<tool>.json`, `inspection.json`, `surface.json`, `tools.json`, `production.json` are passed
  through.

## Decisions and their consequences

* **Run ids** are the ids `twin run` gives its runs (cnctwin.core.rundir.run_id): `<nnnnn>-<slug of the
  project id>-<h8>`, the number one above the highest numbered run and the first 8 hex digits of
  sha256(project hash, canonical options without `resume_from`): readable, in creation order, and
  the same project with the same options gives the same suffix whether the CLI or the API ran it.
  The project hash is the one of the run cache key (or, with `no_cache`, of the project resolved
  the same way; of the file alone when it does not resolve). Consequence: an id says nothing about
  the result; runs of the older API form `<nnnnn>-<project id>` and runs made elsewhere keep their
  folder names, are listed and read, and count for the next number.
* **Examples are `example-<folder>`**, read-only, always listed; their check may not write.
  Consequence: to change an example, upload a copy.
* **A GET that starts a job** (`/api/compare`): the comparison is a deterministic function of two
  finished runs, so the GET names a resource that is computed once and then served; 202 while it
  is being made. Consequence: the first request may take `wait_s` to answer.
* **Jobs for checks and studies, with a `wait_s`** (default 20 s): short checks answer in one
  request, long ones hand back a job. Consequence: clients handle 200 and 202.
* **Entry points as `module:function` with one documented call each**: the API never imports
  the simulator, so a broken or missing module costs one failed job (501), not the server.
  Consequence: the modules of the other parts must provide these names, or the integration sets
  `CNCTWIN_ENTRY_<KIND>`.
* **Cancel by exception from the progress callback**: the loop stops at a macro-step boundary and
  resumes from its last checkpoint; work since that checkpoint (at most a checkpoint interval,
  <= 10 s simulated) is done again, deterministically.
* **rlimit instead of a cgroup without systemd**: RLIMIT_DATA counts private writable mappings,
  not resident memory; a job that maps much more than it touches can fail early. Consequence: in
  containers give the cap headroom (the container's own memory limit is the hard one).
* **Media types and sandboxing**: HTML and SVG from runs and uploads cannot run scripts when
  opened directly; the UI embeds them as documents or images.
* **No interactive API pages** (Swagger UI and ReDoc fetch from a CDN; the box is offline):
  docs/openapi.json and this document are the reference.

## OpenAPI

`cnctwin.api.export_openapi(path=None)` builds the document from the app (no files touched) and
writes it to `docs/openapi.json` (`python -m cnctwin.api.openapi [path]`); the output is
deterministic, and a test fails when docs/openapi.json is stale. Request bodies are typed
(RunCreate, OptionsBody, CalibrateRequest); responses name the fields the API adds and allow the
rest (run files pass through as written).
