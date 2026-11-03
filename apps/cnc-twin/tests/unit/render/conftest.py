"""A synthetic run directory (tests/fixtures/make_synthetic_run.py) made once for the render tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

import make_synthetic_run as synthetic  # noqa: E402


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("render-run") / "run-0001"
    synthetic.make_synthetic_run(folder, seed=4, media=False, media_channels=True)
    return folder
