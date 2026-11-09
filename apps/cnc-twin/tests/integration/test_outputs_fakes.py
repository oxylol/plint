"""Stand-ins for cnctwin.core.loop.run_project used by the outputs tests (no tests here; the job worker and the bench
subprocess load this file by path as their entry point).

* :func:`fake_run_project` writes a synthetic run directory (tests/fixtures/make_synthetic_run.py) the way a run
  would: a copy of a template run made once (``options["template_run"]``), its run.json naming the project it was
  given and the options, the pocket depth of inspection.json following the variant project's stock height (the
  known response of the uncertainty test: depth = 2.011 mm + (stock height - 15 mm)), an out-of-tolerance error
  finding when that depth leaves 2 +/- 0.05 mm, progress reported per macro step; with ``options["replay"]`` it
  writes the replay part of the window instead (a Parquet file of the fast tier and its index line).
* :func:`make_template_runs` writes the synthetic runs and their project folder; :func:`templates` once per session.
* :func:`bench_runner` stands in for run_project in the benchmark test: macro steps that spend known CPU time in
  functions of known levels (:data:`BENCH_TABLE`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
for entry in (str(ROOT), str(FIXTURES)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import make_synthetic_run as synthetic  # noqa: E402

DEPTH_M = 0.002
DEPTH_TOL_M = 5e-5
BASE_DEPTH_M = 0.002011
BASE_HEIGHT_M = 0.015
CALLS_FILE = "fake_calls.jsonl"         # one line per call, beside the template (the tests read it)


# small pictures and a short video keep the tests fast; the pipeline is the same at full size
SMALL = {"image_size": [320, 180, 1], "video": {"size": [192, 108, 1], "speed": 2, "fps": 10},
         "memory_mode": "rlimit"}


def make_template_runs(base: Path) -> dict:
    """Two finished synthetic runs (seed 5: safe, seed 4: not safe), no media, each with its project folder."""
    out = {}
    for key, seed in (("safe", 5), ("unsafe", 4)):
        project = synthetic.write_synthetic_project(base / f"project-{key}", seed=seed)
        run_dir = base / f"template-{key}" / "00001-template"
        synthetic.make_synthetic_run(run_dir, seed=seed, media=False, write_status=False, project_path=project,
                                     media_channels=True)
        out[key] = run_dir
        out[f"{key}_project"] = project
    return out


def templates(tmp_path_factory) -> dict:
    """The template runs, made once per session (in pytest's base temporary folder, shared by the test files)."""
    base = tmp_path_factory.getbasetemp() / "outputs-templates"
    marker = base / "made.json"
    if not marker.is_file():
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
        runs = make_template_runs(base)
        marker.write_text(json.dumps({k: str(v) for k, v in runs.items()}), encoding="utf-8")
    return {k: Path(v) for k, v in json.loads(marker.read_text(encoding="utf-8")).items()}


def calls(template: Path, prefix) -> list[dict]:
    """The fake's calls whose run directory lies under ``prefix``, in call order."""
    path = Path(template).parent / CALLS_FILE
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["run_dir"].startswith(str(prefix))]


def _stock_height_m(project_path: Path) -> float:
    import yaml
    data = yaml.safe_load(Path(project_path).read_text(encoding="utf-8"))
    return float(data["stock"]["size_mm"][2]) * 1e-3


def _record(template: Path, entry: dict) -> None:
    with open(template.parent / CALLS_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _replay(project_path: Path, run_dir: Path, options: dict, progress_cb) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    window = options["replay"]
    tick0, tick1 = int(window["tick0"]), int(window["tick1"])
    ticks = np.arange(tick0, tick1, 8, dtype=np.int64)
    table = pa.table({"tick": ticks, "time_s": ticks * 1e-6, "cutting.force_N[0]": np.sin(ticks * 1e-3)})
    name = f"series/fast/replay-{tick0}-{tick1}.parquet"
    (run_dir / "series" / "fast").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, run_dir / name)
    with open(run_dir / "series" / "index.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tier": "fast", "part": -1, "file": name, "tick0": tick0, "tick1": tick1,
                                 "t0_s": tick0 * 1e-6, "t1_s": tick1 * 1e-6, "rows": len(ticks), "replay": True})
                     + "\n")
    progress_cb({"sim_time_s": tick1 * 1e-6, "phase": "replay"})
    return {"replay": [tick0, tick1], "checkpoint": window.get("checkpoint"), "rows": len(ticks)}


def fake_run_project(project_path, run_dir, options, progress_cb):
    project_path, run_dir = Path(project_path), Path(run_dir)
    if options.get("replay"):
        return _replay(project_path, run_dir, options, progress_cb)
    template = Path(options["template_run"])
    _record(template, {"project": str(project_path), "run_dir": str(run_dir),
                       "sample": (options.get("uncertainty") or {}).get("sample"),
                       "project_sha256": hashlib.sha256(project_path.read_bytes()).hexdigest()})
    for item in template.iterdir():
        target = run_dir / item.name
        if item.name == "status.json":
            continue                                    # the job runner owns status.json
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    for step in range(1, 5):
        progress_cb({"sim_time_s": 0.58 * step, "est_total_s": 2.32, "op": "OP10"})
    depth = BASE_DEPTH_M + (_stock_height_m(project_path) - BASE_HEIGHT_M)
    inspection = json.loads((run_dir / "inspection.json").read_text(encoding="utf-8"))
    record = next(t for t in inspection["tolerances"] if t["id"] == "pocket-depth")
    record.update({"measured_m": depth, "deviation_m": depth - DEPTH_M, "pass": abs(depth - DEPTH_M) <= DEPTH_TOL_M})
    (run_dir / "inspection.json").write_text(json.dumps(inspection, indent=1), encoding="utf-8")
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    findings = json.loads((run_dir / "findings.json").read_text(encoding="utf-8"))
    if not record["pass"]:
        message = f"pocket-depth {depth * 1e3:.4f} mm, tolerance 2 +/- 0.05 mm"
        findings["findings"].append({"id": len(findings["findings"]), "category": "out_of_tolerance",
                                     "severity": "error", "message": message, "time_s": 2.161, "tick": 2161000,
                                     "domain": "inspect", "op": "OP10", "program": "op10.nc", "line": 17,
                                     "block": "G01 X0.", "pos_W_m": [0.01005, 0.021, 0.013],
                                     "pos_M_m": [-0.38095, -0.1912, -0.4084], "quantities": {"measured_m": depth},
                                     "cause_chain": []})
        document["verdict"] = {"safe": False,
                               "reasons": [f"out_of_tolerance (error) at OP10 line 17: {message}"]}
        (run_dir / "findings.json").write_text(json.dumps(findings, indent=1), encoding="utf-8")
    document.update({"project_dir": str(project_path.parent), "project_file": project_path.name,
                     "options": {k: v for k, v in options.items() if k not in ("resume_from", "template_run")}})
    (run_dir / "run.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    return {"depth_m": depth, "safe": document["verdict"]["safe"]}


# ----------------------------------------------------------------------------- the benchmark's stand-in
BENCH_STEP_S = 0.01            # simulated seconds per fake macro step
BURN_S = {"hot": 0.004, "l1": 0.002, "l2": 0.003, "record": 0.001}
BENCH_TABLE = {"hot_loop": ["tests.integration.test_outputs_fakes:hot_step"],
               "l1": ["tests.integration.test_outputs_fakes:l1_step"],
               "l2": ["tests.integration.test_outputs_fakes:l2_step"],
               "recording": ["tests.integration.test_outputs_fakes:record_step"], "op_end": []}


def burn(seconds: float) -> None:
    """Spend ``seconds`` of process CPU time."""
    import time
    end = time.process_time() + seconds
    x = 0
    while time.process_time() < end:
        x += 1


def l1_step() -> None:
    burn(BURN_S["l1"])


def hot_step() -> None:
    burn(BURN_S["hot"])
    l1_step()                  # nested: booked to L1, not to the hot loop


def l2_step() -> None:
    burn(BURN_S["l2"])


def record_step() -> None:
    burn(BURN_S["record"])


def bench_runner(project_path, run_dir, options, progress_cb):
    """Macro steps of known cost until ``options['steps']`` (the benchmark stops it earlier with --max-sim-s)."""
    module = sys.modules[__name__]
    for step in range(1, int(options.get("steps", 1000)) + 1):
        module.hot_step()
        module.l2_step()
        module.record_step()
        progress_cb({"sim_time_s": step * BENCH_STEP_S})
    return {"steps": int(options.get("steps", 1000))}
