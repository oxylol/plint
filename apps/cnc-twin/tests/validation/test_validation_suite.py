"""The validation suite (slow): every case through its harness; a quantity whose independent cases miss
its budget (validation/budgets.yaml rule) fails here. Quantities without a harness are skipped until one
is registered; quantities without a budget are run (every case must simulate) and reported as skipped
with their statistics.

Run: bin/tsafe python -m pytest -m slow tests/validation
"""

from __future__ import annotations

import numpy as np
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


def test_orthogonal_kernel_reproduces_the_closed_form_of_its_rows(results):
    """The orthogonal force values are the turn_step kernel over the polar field; each lists the rows' closed form
    (K h + Ke) b at the same h. They must agree to the kernel's stated accuracy (strips and table, 1-2 %): a larger
    gap is an integration error of the harness, not a model result."""
    q = results.get("cutting_force")
    if q is None:
        pytest.skip("no cutting_force case")
    checked = 0
    for r in q.cases:
        if r.status != "ok" or r.case.operation != "orthogonal_turning":
            continue
        for c in r.comparisons:
            closed = c.extra.get("closed_form") if c.extra else None
            if c.status == "compared" and isinstance(closed, (int, float)) and closed:
                assert abs(float(c.simulated) / float(closed) - 1.0) < 0.03, (r.id, c.name, c.simulated, closed)
                checked += 1
    assert checked > 50


def test_stability_benchmarks_find_the_published_lobe_minimum(results):
    """The deepest point of every analytical benchmark curve (the stability minimum of the published
    semi-discretisation) is reproduced within the curve's digitisation uncertainty plus 15 %: the accuracy
    docs/models/dynamics.md 5.3 states for these charts (troughs within the digitisation + 3 % at a/D = 0.5 and
    2-DOF slotting, the a/D = 0.05 chart reading 7-15 % above the converged solution)."""
    q = results.get("stability_lobe")
    if q is None:
        pytest.skip("no stability_lobe case")
    checked = 0
    for r in q.cases:
        if r.status != "ok" or r.case.measured:
            continue
        pts = [c for c in r.comparisons if c.where == "curve" and c.status == "compared"]
        if not pts:
            continue
        c = min(pts, key=lambda p: float(p.reference))
        unc = float(c.uncertainty or 0.0)
        assert abs(float(c.simulated) - float(c.reference)) <= unc + 0.15 * float(c.reference), (r.id, c.x)
        checked += 1
    assert checked >= 6
