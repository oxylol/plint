"""Fixtures of the API tests: synthetic run directories, a data folder and a client."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_support import DONE_RUN, PARTIAL_RUN, make_settings, synthetic  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from cnctwin.api.app import create_app  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_runs(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("synthetic")
    synthetic.make_synthetic_run(base / "runs" / DONE_RUN, seed=2)
    synthetic.make_synthetic_run(base / "runs" / PARTIAL_RUN, seed=3, until_s=1.2, media=False)
    return base


@pytest.fixture
def data_dir(tmp_path, synthetic_runs) -> Path:
    data = tmp_path / "data"
    shutil.copytree(synthetic_runs / "runs", data / "runs")
    synthetic.write_synthetic_project(data / "examples" / "pocket", seed=4)
    return data


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(make_settings(data_dir))) as test_client:
        yield test_client
