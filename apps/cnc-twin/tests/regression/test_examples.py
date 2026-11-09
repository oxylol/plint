"""Regression: every example project through cnctwin.core.loop.run_project (marked slow; docs/models/loop.md
"Regression"). Each test copies the example, runs it with every domain on and checks that the run completes, that
conservation holds (the per-step ledgers and the end-of-run mass check) and the behaviours of its EXPECTED.md that
the part of the program run can show. Long programs are run for a representative first part (``BUDGET_S``
simulated seconds: the hot loop still costs 10-30 CPU seconds per simulated second); the record says exactly what
was run. Every test writes its record (wall time, simulated time, verdict, findings, each EXPECTED item with
passed / failed / not reached) to ``<pytest basetemp>/regression/<example>.json`` and prints it; the EXPECTED items
are recorded, not asserted (several depend on models still being calibrated), the run and conservation are.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from cnctwin.core.loop import run_project
from cnctwin.paths import EXAMPLES
from tests.regression import expected as EX

pytestmark = pytest.mark.slow

NAMES = sorted(p.name for p in Path(EXAMPLES).iterdir() if (p / "project.yaml").is_file())


def _copy(name: str, target: Path) -> Path:
    shutil.copytree(Path(EXAMPLES) / name, target)
    return target / "project.yaml"


CASES = [(name, None) for name in NAMES] + [(name, EX.VARIANTS[name][0]) for name in NAMES if name in EX.VARIANTS]


@pytest.mark.parametrize("name,variant", CASES, ids=[f"{n}-{v}" if v else n for n, v in CASES])
def test_example(name, variant, tmp_path, tmp_path_factory):
    project = _copy(name, tmp_path / "project")
    if variant is not None:
        EX.VARIANTS[name][1](project.parent)
    run_dir = tmp_path / "run"
    budget = EX.VARIANT_BUDGET_S.get(name, EX.BUDGET_S.get(name)) if variant else EX.BUDGET_S.get(name)
    options = {"outputs": False}
    if budget is not None:
        options["max_sim_s"] = float(budget)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    error = None
    try:
        run_project(project, run_dir, options, None)
    except Exception as failure:          # recorded, then re-raised below (the record says where it stopped)
        error = failure
    record = EX.record(name, run_dir, options, error, time.perf_counter() - wall0, time.process_time() - cpu0)
    record["variant"] = variant
    out = tmp_path_factory.getbasetemp() / "regression"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}{'-' + variant if variant else ''}.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str))
    if error is not None:
        raise error
    assert record["completed_run"], record
    assert all(record["conservation"].values()), record["conservation"]


def test_records_are_complete():
    """Every example has a budget entry or runs in full, and an EXPECTED checker."""
    for name in NAMES:
        assert name in EX.CHECKS, name
    assert set(EX.BUDGET_S) <= set(NAMES)
    assert np.all([b > 0 for b in EX.BUDGET_S.values()])
