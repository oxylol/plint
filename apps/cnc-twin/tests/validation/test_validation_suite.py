"""The validation suite (slow): every case through its harness; a quantity whose independent cases miss
its budget (validation/budgets.yaml rule) fails here. Quantities without a harness are skipped until one
is registered; quantities without a budget are run (every case must simulate) and reported as skipped
with their statistics.

Run: bin/tsafe python -m pytest -m slow tests/validation
"""

from __future__ import annotations

import pytest

from cnctwin.db import Database
from cnctwin.db.schemas import VALIDATION_QUANTITIES
from cnctwin.validation import report as RP
from cnctwin.validation import runner as RN

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def db():
    return Database.load()


@pytest.fixture(scope="module")
def results(db):
    return {q.quantity: q for q in RN.run(db)}


def _stats(q) -> str:
    s = q.independent
    if s is None:
        return "no statistics"
    classes = f", classes {s.class_matches}/{s.class_values} match" if s.class_values else ""
    return (f"{s.passing if s.passing is not None else '-'}/{s.cases_with_values} independent cases within budget, "
            f"median {RP.pct(s.median_abs_rel)}, p90 {RP.pct(s.p90_abs_rel)}{classes}")


@pytest.mark.parametrize("quantity", VALIDATION_QUANTITIES)
def test_quantity_meets_its_budget(results, quantity):
    q = results.get(quantity)
    if q is None:
        pytest.skip(f"no case for {quantity}")
    if q.verdict == RN.NO_HARNESS:
        pytest.skip(f"no harness registered for {quantity} yet")
    errors = [f"{r.id}: {r.error}" for r in q.cases if r.status == "error"]
    assert not errors, "cases that could not be simulated:\n" + "\n".join(errors)
    if q.verdict in (RN.NO_BUDGET, RN.NO_CASES):
        pytest.skip(f"{quantity}: {q.verdict} ({_stats(q)})")
    assert q.verdict == RN.PASS, f"{quantity}: {q.verdict_reason} ({_stats(q)})"


def test_a_case_gives_the_same_result_twice(db):
    first = RN.run(db, ["chip_ratio"])
    second = RN.run(db, ["chip_ratio"])
    assert RP.dumps([RP.quantity_json(q) for q in first]) == RP.dumps([RP.quantity_json(q) for q in second])
