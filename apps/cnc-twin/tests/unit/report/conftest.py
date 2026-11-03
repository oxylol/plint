"""Synthetic run directories for the report tests (tests/fixtures/make_synthetic_run.py), made once."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
HERE = Path(__file__).resolve().parent
for folder in (FIXTURES, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import make_synthetic_run as synthetic  # noqa: E402


@pytest.fixture(scope="session")
def runs(tmp_path_factory):
    """Seed 4 (an out-of-tolerance error: not safe) with images, seed 5 (safe), and a copy of the seed-4
    run in another folder of the same name (the report must not depend on where the run is)."""
    base = tmp_path_factory.mktemp("report-runs")
    made = {}
    for key, seed, media in (("unsafe", 4, True), ("safe", 5, False)):
        folder = base / key / "run-0001"
        synthetic.make_synthetic_run(folder, seed=seed, media=media)
        made[key] = folder
    made["twin"] = base / "twin" / "run-0001"           # the same run in another place
    shutil.copytree(made["unsafe"], made["twin"])
    return made


@pytest.fixture()
def run_copy(runs, tmp_path):
    """A writable copy of the unsafe run (tests that edit files)."""
    def copy(key: str = "unsafe") -> Path:
        target = tmp_path / key / "run-0001"
        shutil.copytree(runs[key], target)
        return target
    return copy
