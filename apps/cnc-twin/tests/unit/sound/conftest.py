"""Fixtures of the sound tests: synthetic inputs and a synthetic run directory (tests/fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "fixtures"
for folder in (FIXTURES, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import make_synthetic_run as synthetic  # noqa: E402
from soundkit import make_inputs  # noqa: E402


@pytest.fixture()
def inputs():
    return make_inputs()


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("sound-run") / "run-0001"
    synthetic.make_synthetic_run(folder, seed=4, media=False, media_channels=True)
    return folder
