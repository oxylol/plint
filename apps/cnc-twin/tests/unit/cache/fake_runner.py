"""Stand-ins for cnctwin.core.loop.run_project in the cache tests (``run_project(project_path, run_dir, options,
progress_cb)``).

* :func:`run_project` writes a small but complete result set in a few milliseconds (findings.json, trace.jsonl,
  timeline.json, a series part with its index, frames.bin, inspection/surface/tools/production.json, a part file
  and run.json with this build's id), deterministic from the project file's text and the options.
* :func:`synthetic_run_project` writes the full synthetic run of tests/fixtures/make_synthetic_run.py (keyframes,
  checkpoints, scene, ...), seeded the same way (about 7 s).
* :func:`edit_program_run_project` changes the project's program while it runs (the key changes during the run).

Every call appends a line to the file named by FAKE_RUNNER_LOG, so a test can count the simulations.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (str(ROOT), str(ROOT / "tests" / "fixtures")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _seed(project_path, options) -> int:
    from cnctwin.cache import canonical_options
    text = Path(project_path).read_text(encoding="utf-8") + json.dumps(canonical_options(options), sort_keys=True)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 100000


def _log(project_path, run_dir) -> None:
    path = os.environ.get("FAKE_RUNNER_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{project_path}\t{run_dir}\n")


def run_project(project_path, run_dir, options, progress_cb):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cnctwin import cache
    from cnctwin.core.rundir import RunDirectory, Timeline, result_set_digest, run_document, write_json_atomic
    _log(project_path, run_dir)
    seed = _seed(project_path, options)
    rng = np.random.default_rng(seed)
    run_dir = Path(run_dir)
    rd = RunDirectory(run_dir, own_status=False)
    rd.prepare()
    if progress_cb is not None:
        progress_cb({"sim_time_s": 0.0, "est_total_s": 1.0}, phase="init")
    (run_dir / "trace.jsonl").write_text(json.dumps({"id": 0, "time_s": 0.0, "tick": 0, "domain": "loop",
                                                     "event": "start", "quantities": {}, "parents": []}) + "\n")
    rd.write_findings([])
    timeline = Timeline(run_dir)
    for i in range(3):
        index = timeline.open(path=0, op="OP10", program="op.nc", line=10 + i, block=f"G01 X{i}", kind="linear",
                              tick0=i * 10_000)
        timeline.close(index, (i + 1) * 10_000, ticks={"cutting": 10_000})
    timeline.write()
    ticks = np.arange(0, 30_000, 10_000, dtype=np.int64) + 10_000
    table = pa.table({"tick": ticks, "time_s": ticks / 1e6, "x.value_m": rng.normal(size=3)})
    (run_dir / "series" / "macro").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, run_dir / "series" / "macro" / "part-0.parquet")
    (run_dir / "series" / "index.jsonl").write_text(json.dumps(
        {"tier": "macro", "part": 0, "file": "series/macro/part-0.parquet", "tick0": 10_000, "tick1": 30_000,
         "t0_s": 0.01, "t1_s": 0.03, "rows": 3}) + "\n")
    (run_dir / "frames.bin").write_bytes(rng.bytes(64))
    (run_dir / "part_OP10.stl").write_bytes(rng.bytes(128))
    measured = 0.002 + float(rng.normal(0.0, 1e-5))
    write_json_atomic(run_dir / "inspection.json", {"format": "cnctwin-inspection", "version": 1, "tolerances": [
        {"id": "depth", "kind": "distance", "nominal_m": 0.002, "plus_m": 5e-5, "minus_m": 5e-5, "measured_m": measured,
         "deviation_m": measured - 0.002, "pass": abs(measured - 0.002) <= 5e-5}]})
    write_json_atomic(run_dir / "surface.json", {"format": "cnctwin-surface", "version": 1, "features": [], "burrs": []})
    write_json_atomic(run_dir / "tools.json", {"tools": [{"number": 1, "VB_max_m": 2e-5, "life_used_fraction": 0.01}]})
    write_json_atomic(run_dir / "production.json", {"format": "cnctwin-production", "cycle_time_s": 0.03})
    if progress_cb is not None:
        progress_cb({"sim_time_s": 0.03, "est_total_s": 0.03}, phase="done")
    document = run_document(run_id=run_dir.name, project={"name": "fake", "seed": seed}, options=options,
                            clock={"tick_s": 1e-6}, verdict_={"safe": True, "reasons": []},
                            summary={"sim_time_s": 0.03, "cycle_time_s": 0.03}, outputs=rd.outputs_index(), seed=seed)
    document["versions"]["build"] = cache.build_id()   # the key's build (tests pin it)
    rd.write_run_json(document)
    return {"seed": seed, "digest": result_set_digest(run_dir)}


def synthetic_run_project(project_path, run_dir, options, progress_cb):
    import make_synthetic_run as synthetic
    _log(project_path, run_dir)
    run = synthetic.SyntheticRun(run_dir, _seed(project_path, options), write_status=False, media=False,
                                 project_path=project_path)
    run.begin()
    while not run.finished:
        run.step()
        if progress_cb is not None:
            progress_cb({"sim_time_s": run.time_s, "est_total_s": run.total_s, "op": "OP10"})
    return {"verdict": run.verdict(), "end_tick": run.end_tick}


def edit_program_run_project(project_path, run_dir, options, progress_cb):
    result = run_project(project_path, run_dir, options, progress_cb)
    program = Path(project_path).parent / "warmup.nc"
    program.write_text(program.read_text(encoding="utf-8") + "\n(EDITED WHILE RUNNING)\n", encoding="utf-8")
    return result


def slow_run_project(project_path, run_dir, options, progress_cb):
    import time
    time.sleep(float(os.environ.get("CNCTWIN_TEST_DELAY_S", "0.5")))
    return run_project(project_path, run_dir, options, progress_cb)
