"""Job entry points of the UI test server (docs/api.md "Entry points"), run by the real API worker.

``run_project`` writes the synthetic run of tests/fixtures/make_synthetic_run.py one macro step at
a time, reporting progress and honouring ``resume_from``, like cnctwin.core.loop.run_project
(options: ``step_delay_s``, ``media``). ``check_project`` returns a check result in the shape
``twin check`` prints; ``run_study`` and ``calibrate`` write the study and calibration documents of
ui_fixtures.py. Comparisons use the real cnctwin.report.compare:compare_runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ui_fixtures  # noqa: E402
from ui_fixtures import synthetic  # noqa: E402


def run_project(project_path, run_dir, options, progress_cb):
    seed = synthetic.seed_from_project(project_path)
    run = synthetic.SyntheticRun(run_dir, seed, write_status=False, media=bool(options.get("media", False)),
                                 project_path=project_path, media_channels=True)
    if options.get("resume_from"):
        run.resume(options["resume_from"])
    else:
        run.begin()
    delay = float(options.get("step_delay_s", 0.0))
    while not run.finished:
        run.step()
        progress_cb({"sim_time_s": run.time_s, "est_total_s": run.total_s, "op": "OP10"}, checkpoint=run.last_checkpoint)
        if delay:
            time.sleep(delay)
    return {"verdict": run.verdict(), "end_tick": run.end_tick}


def check_project(request, out_dir, progress_cb):
    progress_cb(phase="checking")
    text = Path(request["project_path"]).read_text(encoding="utf-8")
    ok = "format: cnctwin-project" in text
    return {"ok": ok, "summary": "project checked: 1 operation, 1 tool, timestep resolved",
            "warnings": ["tool T1: no measured runout phase (0 deg assumed)"], "errors": [],
            "project_id": request.get("project_id"), "read_only": bool(request.get("read_only")),
            "estimated_cpu_s_per_sim_s": 0.62, "estimated_memory_MB": 410,
            "resolved": {"ipo_ticks": 1000, "macro_ipo": 10, "operations": {"OP10": [{"tool": 1, "l0_ticks": 8, "rule": "max_deg_per_step"}]}}}


def run_study(request, out_dir, progress_cb):
    options = request.get("options") or {}
    runs = int(options.get("runs", 12))
    for i in range(runs):
        progress_cb(phase=f"variant {i + 1}/{runs}", done=i + 1, total=runs)
        time.sleep(0.01)
    progress_cb(phase="aggregating")
    study = ui_fixtures.synthetic_study(seed=int(options.get("seed", 9)), runs=max(runs, 4), base_run=request.get("run_id"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps(study, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if options.get("attach", True) and request.get("run_dir"):
        ui_fixtures.attach_study(Path(request["run_dir"]), study)
    results = study["results"]
    return {"format": "cnctwin-uncertainty", "study": "study.json", "page": None, "samples_ok": results["samples_ok"],
            "samples_failed": results["samples_failed"], "verdict": results["verdict"], "factors": [f["name"] for f in study["factors"]]}


def calibrate(request, out_dir, progress_cb):
    progress_cb(phase="spindle: fit")
    if not request.get("run_dir"):
        raise ValueError("the UI test calibration needs a run")
    return ui_fixtures.write_calibration(Path(out_dir), Path(request["run_dir"]))
