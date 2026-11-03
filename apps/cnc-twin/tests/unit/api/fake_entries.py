"""Stand-ins for the job entry points, run by the real worker subprocess in the API tests.

``run_project`` behaves like cnctwin.core.loop.run_project: it writes the synthetic run
(tests/fixtures/make_synthetic_run.py) one macro step at a time, reports progress, honours
``resume_from`` and stops when progress_cb raises. Options steer it: ``step_delay_s`` (wall time per
macro step), ``fail_at_s`` (raise), ``crash_at_s`` (os._exit), ``hang_at_s`` (sleep, ignoring
SIGTERM), ``hog_mb`` (allocate), ``media`` (write video, audio, reports).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

import make_synthetic_run as synthetic  # noqa: E402


def run_project(project_path, run_dir, options, progress_cb):
    seed = synthetic.seed_from_project(project_path)
    run = synthetic.SyntheticRun(run_dir, seed, write_status=False, media=bool(options.get("media", False)),
                                 project_path=project_path)
    if options.get("resume_from"):
        run.resume(options["resume_from"])
    else:
        run.begin()
    if options.get("hog_mb"):
        hog = bytearray(int(options["hog_mb"]) * 1024 * 1024)
        hog[::4096] = b"x" * len(hog[::4096])
    delay = float(options.get("step_delay_s", 0.0))
    while not run.finished:
        run.step()
        t = run.time_s
        progress_cb({"sim_time_s": t, "est_total_s": run.total_s, "op": "OP10"}, checkpoint=run.last_checkpoint)
        if options.get("fail_at_s") is not None and t >= float(options["fail_at_s"]):
            raise RuntimeError(f"synthetic failure at {t:.3f} s")
        if options.get("crash_at_s") is not None and t >= float(options["crash_at_s"]):
            os._exit(3)
        if options.get("hang_at_s") is not None and t >= float(options["hang_at_s"]):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(3600)
        if delay:
            time.sleep(delay)
    return {"verdict": run.verdict(), "end_tick": run.end_tick}


def check_project(request, out_dir, progress_cb):
    options = request.get("options") or {}
    if options.get("fail"):
        raise ValueError("synthetic check failure")
    progress_cb(phase="checking")
    text = Path(request["project_path"]).read_text(encoding="utf-8")
    return {"ok": "format: cnctwin-project" in text, "project_id": request["project_id"], "problems": [],
            "read_only": request.get("read_only"), "options": options}


def compare_runs(request, out_dir, progress_cb):
    out = {}
    for side in ("a", "b"):
        run_dir = Path(request[f"run_{side}_dir"])
        document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        findings = json.loads((run_dir / "findings.json").read_text(encoding="utf-8"))["findings"]
        out[side] = {"run": request[f"run_{side}"], "safe": document["verdict"]["safe"], "findings": len(findings)}
    out["findings_delta"] = out["b"]["findings"] - out["a"]["findings"]
    (Path(out_dir) / "compare.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def run_study(request, out_dir, progress_cb):
    runs = int((request.get("options") or {}).get("runs", 3))
    for i in range(runs):
        progress_cb(done=i + 1, total=runs)
        time.sleep(0.02)
    (Path(out_dir) / "study.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return {"runs": runs, "base_run": request["run_id"]}


def calibrate(request, out_dir, progress_cb):
    progress_cb(phase="fitting")
    return {"project_id": request.get("project_id"), "run_id": request.get("run_id"),
            "machine_data": request.get("machine_data"), "parameters": {"axis_X.kv_1_s": 33.3}}
