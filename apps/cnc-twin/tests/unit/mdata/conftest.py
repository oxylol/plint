import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mdatakit  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_run(tmp_path_factory):
    """The synthetic run directory (tests/fixtures/make_synthetic_run.py, seed 7, no media)."""
    return mdatakit.make_run(tmp_path_factory.mktemp("mdata-run") / "run")


@pytest.fixture(scope="session")
def project_file(tmp_path_factory):
    """A project on the real database (haas-vf2ss, al6061-t6) the calibration targets read."""
    return mdatakit.write_project(tmp_path_factory.mktemp("mdata-project"))


@pytest.fixture(scope="session")
def resolved_project(project_file):
    from cnctwin.project import resolve
    return resolve(project_file)
