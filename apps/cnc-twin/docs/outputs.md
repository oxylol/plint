# Outputs of a run and the studies around runs (`cnctwin/core/outputs.py`)

What a finished run directory (architecture 16) gets after the simulation, and the studies built on whole runs.
The run loop (docs/models/loop.md) calls `finalize_outputs(run_dir, options)` as its last step; `python -m
cnctwin.core.outputs RUN_DIR` makes them again on any finished run directory.

## 1. finalize_outputs

`finalize_outputs(run_dir, options=None, progress_cb=None) -> dict` runs four generators, each in its own
subprocess (`python -m cnctwin.core.outputs --generate NAME ...`) in this order:

| generator | module | files |
|---|---|---|
| images | cnctwin.render | `images/*.png`, `render.json` (what each picture shows): the standard pictures (setup, cut, part per operation with its section, temperature, chip pile, coolant, deflection, deviation, finish, wear), the finished part with every colour layer the run has data for, a findings overview and one picture per finding of severity warning or worse where it was raised |
| video | cnctwin.render | `video.mp4` (libx264, bit-exact), its part of `render.json` |
| audio | cnctwin.sound | `audio.wav` (44.1 kHz mono, 20 Pa full scale), `audio.json` |
| report | cnctwin.report | `report.html`, `report.pdf`, `setup_sheet.html`, `setup_sheet.pdf` (embeds the pictures: runs last) |

Each subprocess runs with the deterministic environment (one thread, `PYTHONHASHSEED=0`, `SOURCE_DATE_EPOCH`,
`TZ=UTC`) under a memory cap (a systemd scope like `bin/tsafe` where systemd-run is available, else
RLIMIT_DATA times `rlimit_data_factor`), a time limit (`timeout_base_s + timeout_wall_per_sim x simulated s`) and
nice. A generator that fails or times out is recorded (`run.json` `generators.<name> = {status, reason, files}`),
its partial files are removed and the others go on; a cancel (the progress callback raising) stops the
generator's process (scope) and releases the run directory's lock (`.outputs.lock`). Every file is registered in
run.json's output index (`outputs[]: {path, bytes, sha256, generator}`; a hidden hash cache avoids re-reading
unchanged series and frames).

## 2. Settings

The template's `outputs` section, under the project's (`run.json project.outputs`), under the run's options
(`run.json options.outputs`), under the caller's `options["outputs"]`: `report` (formats, `[]` none),
`setup_sheet`, `images`, `findings`, `layers`, `image_size` (`[w, h, supersampling]`), `video` (`false` or `{fps,
speed, size}`), `audio` (`false` or `{listener, sources}`), `only` (generators to make, the others kept),
`memory_mode` (auto, systemd, rlimit, none), `memory_bytes` and `timeout_s` (one value or per generator).
`outputs: false` makes nothing. Defaults, caps and limits: `db/models/outputs.yaml`.

## 3. Studies around runs

* `uncertainty_study(run_dir, options)`: cnctwin.uncertainty plans samples from the project's stated tolerances and
  runs every variant through `run_project` (`resolve_entry`, default `cnctwin.core.loop:run_project`, overridden
  by `CNCTWIN_ENTRY_SIMULATE`); the result is attached to the run (`uncertainty/study.json`,
  `uncertainty/uncertainty.html`, `uncertainty.json`) and the report is made again with its Uncertainty section.
* `compare_study(run_a, run_b)`: cnctwin.report.compare (`compare.json`, `compare.html`).
* `calibration_study(run_dir, machine_data)`: cnctwin.mdata.calibrate; its loop simulator re-runs `run_project`
  with the parameters as an overlay.
* `replay_window(run_dir, tick0, tick1)` (`twin replay --window`): `run_project` with `options["replay"] =
  {tick0, tick1, checkpoint}` replays the window from the nearest committed checkpoint at or before tick0 and
  writes only `series/fast/replay-<tick0>-<tick1>.parquet` and its index line (`replay: true`).
