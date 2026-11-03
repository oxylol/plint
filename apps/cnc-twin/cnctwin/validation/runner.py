"""Running the cases through their harnesses and aggregating per quantity (docs/validation.md).

:func:`run` loads the cases (cnctwin.db), calls the registered harness of each case's quantity,
compares every reference value with the prediction (:mod:`metrics`), marks calibration-dependent
cases (:mod:`provenance`) and aggregates per quantity:

* point level, over the counted values of the independent cases: median and 90th percentile of
  |relative error| (temperatures: of the rise; classes: the mismatch rate), the share of values within
  their uncertainty, values without a prediction;
* case level: cases within budget (every compared value inside its tolerance), the largest miss in
  tolerances; the verdict by the rule of validation/budgets.yaml (:mod:`budgets`).

Calibration-dependent cases are aggregated the same way but separately; they never enter the
verdict. Everything is deterministic: cases in (quantity, id) order, no time stamps.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field

from . import budgets as B
from . import metrics as M
from . import provenance as PV
from .cases import Case, load_cases
from .registry import COMPARED, NO_PREDICTION, HarnessOutput, get, load_harnesses

PASS = "pass"
FAIL = "fail"
NO_BUDGET = "no budget"
NO_CASES = "no independent cases"
NO_HARNESS = "no harness"


@dataclass
class CaseResult:
    case: Case
    status: str                                   # ok | error | no_harness
    output: HarnessOutput | None = None
    comparisons: list = field(default_factory=list)
    stats: M.CaseStats | None = None
    dependencies: list = field(default_factory=list)
    error: str = ""

    @property
    def calibration_dependent(self) -> bool:
        return bool(self.dependencies)

    @property
    def id(self) -> str:
        return self.case.id


@dataclass
class GroupStats:
    cases: int
    cases_with_values: int
    passing: int | None
    values: int
    compared: int
    no_prediction: int
    median_abs_rel: float | None
    p90_abs_rel: float | None
    within_uncertainty: int
    uncertainty_known: int
    worst_case: str | None
    worst_miss_ratio: float | None
    median_values_abs_rel: float | None = None      # reference values only
    median_curves_abs_rel: float | None = None      # curve points only
    class_matches: int = 0
    class_values: int = 0


@dataclass
class QuantityResult:
    quantity: str
    budget: B.Budget | None
    harness: str
    cases: list[CaseResult]
    independent: GroupStats | None = None
    dependent: GroupStats | None = None
    verdict: str = NO_HARNESS
    verdict_reason: str = ""


def _group_stats(results: list[CaseResult], budgeted: bool) -> GroupStats:
    comps = [c for r in results if r.status == "ok" for c in r.comparisons if c.counted]
    errs = M.abs_errors(comps)
    with_unc = [c for c in comps if c.within_uncertainty is not None]
    ok = [r for r in results if r.status == "ok" and r.stats is not None and (r.stats.compared + r.stats.no_prediction)]
    passing = sum(1 for r in ok if r.stats.passes) if budgeted else None
    worst, worst_ratio = None, None
    for r in ok:
        ratio = r.stats.max_miss_ratio
        if ratio is not None and (worst_ratio is None or ratio > worst_ratio):
            worst, worst_ratio = r.id, ratio
    matches, n_class = M.class_counts(comps)
    return GroupStats(cases=len(results), cases_with_values=len(ok), passing=passing, values=len(comps),
                      compared=sum(1 for c in comps if c.status == COMPARED),
                      no_prediction=sum(1 for c in comps if c.status == NO_PREDICTION),
                      median_abs_rel=M.percentile(errs, 50.0), p90_abs_rel=M.percentile(errs, 90.0),
                      within_uncertainty=sum(1 for c in with_unc if c.within_uncertainty),
                      uncertainty_known=len(with_unc), worst_case=worst, worst_miss_ratio=worst_ratio,
                      median_values_abs_rel=M.percentile(M.abs_errors(comps, "value"), 50.0),
                      median_curves_abs_rel=M.percentile(M.abs_errors(comps, "curve"), 50.0),
                      class_matches=matches, class_values=n_class)


def verdict(stats: GroupStats | None, budget: B.Budget | None, failed_runs: int) -> tuple[str, str]:
    """(verdict, reason) by the rule of validation/budgets.yaml applied to the independent cases."""
    if failed_runs:
        return FAIL, f"{failed_runs} case(s) could not be simulated (harness error)"
    if stats is None or stats.cases_with_values == 0:
        return NO_CASES, "every case is calibration-dependent or has no compared value"
    if budget is None:
        return NO_BUDGET, "validation/budgets.yaml defines no budget for this quantity"
    share = stats.passing / stats.cases_with_values
    worst = stats.worst_miss_ratio
    if share >= B.QUANTITY_PASS_FRACTION and (worst is None or worst <= B.MAX_MISS_FACTOR):
        return PASS, (f"{stats.passing} of {stats.cases_with_values} cases within budget ({share:.0%} >= "
                      f"{B.QUANTITY_PASS_FRACTION:.0%}); largest miss {_ratio(worst)} tolerances")
    parts = []
    if share < B.QUANTITY_PASS_FRACTION:
        parts.append(f"only {stats.passing} of {stats.cases_with_values} cases within budget ({share:.0%} < "
                     f"{B.QUANTITY_PASS_FRACTION:.0%})")
    if worst is not None and worst > B.MAX_MISS_FACTOR:
        parts.append(f"{stats.worst_case} misses by {_ratio(worst)} tolerances (> {B.MAX_MISS_FACTOR:g})")
    return FAIL, "; ".join(parts)


def _ratio(value) -> str:
    if value is None:
        return "-"
    return "inf" if not math.isfinite(value) else f"{value:.2f}"


def run_case(case: Case, db, works: PV.Works, budget: B.Budget | None) -> CaseResult:
    harness = get(case.quantity)
    if harness is None:
        return CaseResult(case, "no_harness", error=f"no harness registered for {case.quantity!r}")
    try:
        output = harness.function(case, db)
        comps = M.compare_case(case, output, budget)
    except Exception as error:                  # reported per case; the suite goes on with the others
        lines = traceback.format_exception(type(error), error, error.__traceback__)
        return CaseResult(case, "error", error="".join(lines[-6:]).strip())
    stats = M.case_stats(comps, budget is not None)
    deps = PV.dependencies(case.src, output.provenance, works)
    return CaseResult(case, "ok", output, comps, stats, deps)


def run(db=None, quantities=None, case_ids=None, budgets: dict | None = None, works: PV.Works | None = None
        ) -> list[QuantityResult]:
    """Run the selected cases (all by default) and aggregate per quantity, in quantity order."""
    if db is None:
        from ..db import Database
        db = Database.load()
    load_harnesses()
    budgets = budgets if budgets is not None else B.load_budgets()
    works = works if works is not None else PV.works_from_db(db)
    cases = load_cases(db, quantities, case_ids)
    by_quantity: dict[str, list[Case]] = {}
    for case in cases:
        by_quantity.setdefault(case.quantity, []).append(case)
    out = []
    for quantity in sorted(by_quantity):
        budget = budgets.get(quantity)
        harness = get(quantity)
        results = [run_case(c, db, works, budget) for c in by_quantity[quantity]]
        q = QuantityResult(quantity, budget, f"{harness.function.__module__}.{harness.function.__name__}"
                           if harness else "", results)
        if harness is None:
            q.verdict, q.verdict_reason = NO_HARNESS, "no harness registered yet"
        else:
            indep = [r for r in results if r.status == "ok" and not r.calibration_dependent]
            dep = [r for r in results if r.status == "ok" and r.calibration_dependent]
            failed = sum(1 for r in results if r.status == "error")
            q.independent = _group_stats(indep, budget is not None)
            q.dependent = _group_stats(dep, budget is not None) if dep else None
            q.verdict, q.verdict_reason = verdict(q.independent, budget, failed)
        out.append(q)
    return out


__all__ = ["run", "run_case", "verdict", "CaseResult", "QuantityResult", "GroupStats", "PASS", "FAIL", "NO_BUDGET",
           "NO_CASES", "NO_HARNESS"]
