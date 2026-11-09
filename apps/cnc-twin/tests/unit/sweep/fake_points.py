"""Runners of the sweep tests beside tests/unit/dataset/realrun.py: one that fails for long stick-outs, one that
waits (CNCTWIN_TEST_DELAY_S) before writing its run."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import yaml

DATASET_TESTS = Path(__file__).resolve().parents[1] / "dataset"
if str(DATASET_TESTS) not in sys.path:
    sys.path.insert(0, str(DATASET_TESTS))

import realrun  # noqa: E402


def failing_run_project(project_path, run_dir, options, progress_cb):
    data = yaml.safe_load(Path(project_path).read_text(encoding="utf-8"))
    if float(data["tools"]["list"][0]["stickout_mm"]) >= 40.0:
        raise RuntimeError("synthetic failure for a long stick-out")
    return realrun.write_run(project_path, run_dir, options, progress_cb)


def slow_run_project(project_path, run_dir, options, progress_cb):
    time.sleep(float(os.environ.get("CNCTWIN_TEST_DELAY_S", "1")))
    return realrun.write_run(project_path, run_dir, options, progress_cb)
