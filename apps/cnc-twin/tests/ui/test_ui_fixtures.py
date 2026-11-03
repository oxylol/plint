"""The data of the UI tests (ui_fixtures.py, fake_jobs.py): the data folder, the uncertainty study
and the calibration overlay have the documented structure and are internally consistent; and,
with CNCTWIN_UI_TESTS=1, the browser tests themselves (run_ui_tests.sh, Playwright in docker)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fake_jobs  # noqa: E402
import ui_fixtures  # noqa: E402


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    data = tmp_path_factory.mktemp("ui") / "data"
    ui_fixtures.build_data_dir(data)
    return data


def test_data_dir_has_the_runs_and_examples(data_dir):
    runs = sorted(p.name for p in (data_dir / "runs").iterdir())
    assert runs == sorted([ui_fixtures.DONE_RUN, ui_fixtures.SAFE_RUN, ui_fixtures.PARTIAL_RUN])
    done = json.loads((data_dir / "runs" / ui_fixtures.DONE_RUN / "run.json").read_text())
    safe = json.loads((data_dir / "runs" / ui_fixtures.SAFE_RUN / "run.json").read_text())
    assert done["verdict"]["safe"] is False and safe["verdict"]["safe"] is True
    assert not (data_dir / "runs" / ui_fixtures.PARTIAL_RUN / "run.json").exists()
    status = json.loads((data_dir / "runs" / ui_fixtures.PARTIAL_RUN / "status.json").read_text())
    assert status["status"] == "running"
    for name in ui_fixtures.EXAMPLES:
        assert "format: cnctwin-project" in (data_dir / "examples" / name / "project.yaml").read_text()
    channels = json.loads((data_dir / "runs" / ui_fixtures.DONE_RUN / "channels.json").read_text())["channels"]
    assert any(c["name"] == "dynamics.chatter_freq_Hz" for c in channels)       # media channels on


def test_attached_study_follows_the_format(data_dir):
    study = json.loads((data_dir / "runs" / ui_fixtures.DONE_RUN / "uncertainty" / "study.json").read_text())
    assert study["format"] == "cnctwin-uncertainty" and study["version"] == 1
    results = study["results"]
    assert set(results) >= {"dimensions", "factor_samples", "failed", "findings", "samples_failed", "samples_ok", "scalars",
                            "sensitivity", "verdict"}
    n = study["settings"]["runs"]
    assert results["samples_ok"] == n == len(study["sample_results"]) == len(results["factor_samples"]["values"])
    assert (data_dir / "runs" / ui_fixtures.DONE_RUN / "uncertainty" / "uncertainty.html").is_file()


def test_study_numbers_are_consistent():
    study = ui_fixtures.synthetic_study(seed=5, runs=20)
    results = study["results"]
    depth = next(d for d in results["dimensions"] if d["id"] == "pocket-depth")
    samples = np.array(depth["samples"])
    out = int(np.sum(np.abs(samples - 0.002) > 5e-5))
    assert depth["out_of_tolerance"]["k"] == out
    unsafe = results["verdict"]["unsafe"]
    assert unsafe["k"] == sum(1 for r in study["sample_results"] if not r["safe"]) == out
    assert unsafe["low"] <= unsafe["p"] <= unsafe["high"]
    s = samples.std(ddof=1)
    cpk = min((0.00205 - samples.mean()) / (3 * s), (samples.mean() - 0.00195) / (3 * s))
    assert math.isclose(depth["capability"]["Cpk"], cpk, rel_tol=1e-12)
    assert math.isclose(depth["measured"]["mean"], samples.mean(), rel_tol=1e-12)
    for entry in results["sensitivity"]:
        rhos = [abs(r["rho"]) for r in entry["ranking"]]
        assert rhos == sorted(rhos, reverse=True)
        assert all(-1 <= r["rho"] <= 1 for r in entry["ranking"])
    depth_rank = next(e for e in results["sensitivity"] if e["output"] == "dimension pocket-depth")["ranking"]
    assert depth_rank[0]["factor"] in ("stock_size[2]", "length_offset[T1]")    # the response is built from these
    assert json.dumps(study) == json.dumps(ui_fixtures.synthetic_study(seed=5, runs=20))   # deterministic


def test_calibration_overlay(data_dir, tmp_path):
    summary = ui_fixtures.write_calibration(tmp_path, data_dir / "runs" / ui_fixtures.DONE_RUN)
    overlay = json.loads((tmp_path / "overlay.json").read_text())
    assert overlay["format"] == "cnctwin-calibration-overlay" and overlay["time_base"] == "simulation"
    series = overlay["series"][0]
    assert len(series["t_s"]) == len(series["measured"]) == len(series["simulated"]) == len(series["calibrated"]) > 100
    assert series["metrics"]["calibrated"]["rmse"] < series["metrics"]["simulated"]["rmse"]
    report = json.loads((tmp_path / "calibration.json").read_text())
    assert report["format"] == "cnctwin-calibration-result"
    kc = next(p for p in report["targets"][0]["params"] if p["name"] == "kc_scale")
    assert abs(kc["value"] - 1.08) < 0.02                                     # the synthetic truth
    assert summary["files"]["series"] == "overlay.json"


def test_fake_jobs(tmp_path):
    project = tmp_path / "project"
    ui_fixtures.synthetic.write_synthetic_project(project, seed=8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    seen = []
    result = fake_jobs.run_project(project / "project.yaml", run_dir, {}, lambda progress=None, **kw: seen.append(progress))
    assert result["verdict"]["safe"] is False and (run_dir / "run.json").is_file() and seen
    check = fake_jobs.check_project({"project_path": str(project / "project.yaml"), "project_id": "p"}, tmp_path, lambda **kw: None)
    assert check["ok"] is True
    out = tmp_path / "study"
    study = fake_jobs.run_study({"run_id": "run", "run_dir": str(run_dir), "options": {"runs": 6}}, out, lambda **kw: None)
    assert study["samples_ok"] == 6 and (out / "study.json").is_file() and (run_dir / "uncertainty" / "study.json").is_file()
    cal = fake_jobs.calibrate({"run_dir": str(run_dir)}, tmp_path / "cal", lambda **kw: None)
    assert (tmp_path / "cal" / "overlay.json").is_file() and cal["targets"][0]["target"] == "spindle"


@pytest.mark.skipif(os.environ.get("CNCTWIN_UI_TESTS") != "1" or shutil.which("docker") is None,
                    reason="browser tests: set CNCTWIN_UI_TESTS=1 (needs docker and the Playwright image)")
def test_browser_suite():
    completed = subprocess.run([str(HERE / "run_ui_tests.sh")], capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-2000:]
