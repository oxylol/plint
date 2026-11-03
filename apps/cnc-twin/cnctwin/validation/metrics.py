"""Error metrics of the validation suite (docs/validation.md "Metrics").

Per compared value (a reference value or one point of a reference curve):

* scalars: relative error e = (simulated - reference) / |reference|;
* temperatures (a reference in a temperature unit): the same relative to the rise above ambient,
  e = (T_sim - T_ref) / (T_ref - T_ambient) (the budget rule of validation/budgets.yaml);
* bounds: a reference whose note says "lower bound" ("upper bound") is a one-sided limit: a simulated
  value on the allowed side has e = 0, otherwise e as above;
* classes (chip forms, grades, stability points): e = 0 when the simulated class equals the
  reference class, 1 otherwise.

Curves: the points are compared one by one at the reference x (the harness simulates at the
reference x, so x is always matched); the curve error is the RMS and the maximum of |e| over its
compared points.

Uncertainty of a reference value: the prediction's propagated uncertainty, else the value's (or the
curve's) ``uncertainty``, else the case's ``uncertainty_pct`` of |reference| as written. A value is
within its uncertainty when |simulated - reference| <= uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import budgets as B
from .cases import Case, is_temperature_unit, to_si
from .registry import CLASS, COMPARED, NO_PREDICTION, NOT_COMPARED, HarnessOutput, Prediction

LOWER = "lower"
UPPER = "upper"
NONE = "none"


@dataclass
class Comparison:
    where: str                      # value | curve
    index: int                      # index of the value, or of the curve
    point: int | None               # index of the curve point
    name: str
    x: float | None
    x_name: str
    x_unit: str
    reference: float | str
    unit: str
    simulated: float | str | None
    status: str
    kind: str = "scalar"
    bound: str = NONE
    temperature: bool = False
    scale: float | None = None      # |reference| or the rise above ambient (reference unit)
    error_rel: float | None = None
    uncertainty: float | None = None
    tolerance: float | None = None
    budget_pct: float | None = None
    component: str | None = None
    within_tolerance: bool | None = None
    within_uncertainty: bool | None = None
    miss_ratio: float | None = None  # |sim - ref| / tolerance (0 when inside a bound)
    note: str = ""
    reference_note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def counted(self) -> bool:
        return self.status in (COMPARED, NO_PREDICTION)


def bound_of(note: str) -> str:
    text = (note or "").lower()
    if "lower bound" in text:
        return LOWER
    if "upper bound" in text:
        return UPPER
    return NONE


def _uncertainty(pred: Prediction, entry_unc, curve_unc, case_pct, reference) -> float | None:
    if pred.uncertainty is not None:
        return abs(float(pred.uncertainty))
    if entry_unc is not None:
        return abs(float(entry_unc))
    if curve_unc is not None:
        return abs(float(curve_unc))
    if case_pct is not None and isinstance(reference, (int, float)):
        return abs(float(reference)) * float(case_pct) / 100.0
    return None


def compare_one(*, where: str, index: int, point: int | None, name: str, reference, unit: str, pred: Prediction,
                x=None, x_name: str = "", x_unit: str = "", entry_unc=None, curve_unc=None, case_pct=None,
                ref_note: str = "", ambient_K: float | None = None, budget: B.Budget | None = None) -> Comparison:
    """Comparison of one simulated value with its reference (see the module doc)."""
    temperature = is_temperature_unit(unit) if pred.kind != CLASS else False
    comp = Comparison(where=where, index=index, point=point, name=name, x=x, x_name=x_name or "", x_unit=x_unit or "",
                      reference=reference, unit=unit or "", simulated=pred.value, status=pred.status, kind=pred.kind,
                      bound=bound_of(ref_note), temperature=temperature, component=pred.component, note=pred.note,
                      reference_note=ref_note or "", extra=dict(pred.extra))
    if budget is not None:
        comp.budget_pct = budget.pct_for(pred.component)
    if pred.kind == CLASS:
        ref_class = pred.reference_class if pred.reference_class is not None else str(reference)
        comp.reference = ref_class if pred.reference_class is not None else reference
        if pred.status == COMPARED:
            match = str(pred.value) == str(ref_class)
            comp.error_rel = 0.0 if match else 1.0
            comp.within_tolerance = match
            comp.miss_ratio = 0.0 if match else math.inf
        elif pred.status == NO_PREDICTION:
            comp.error_rel = 1.0
            comp.within_tolerance = False
            comp.miss_ratio = math.inf
        return comp
    ref = float(reference)
    comp.uncertainty = _uncertainty(pred, entry_unc, curve_unc, case_pct, ref)
    if temperature:
        if ambient_K is None:
            raise ValueError(f"{name}: a temperature reference needs the ambient temperature of the case")
        rise = to_si(ref, unit) - ambient_K          # kelvin = degC differences
        comp.scale = abs(rise)
    else:
        comp.scale = abs(ref)
    if comp.budget_pct is not None:
        comp.tolerance = B.tolerance(comp.budget_pct, comp.scale, comp.uncertainty)
    if pred.status == NO_PREDICTION:
        comp.within_tolerance = False if comp.tolerance is not None else None
        comp.within_uncertainty = False if comp.uncertainty is not None else None
        comp.miss_ratio = math.inf if comp.tolerance is not None else None
        return comp
    if pred.status == NOT_COMPARED or pred.value is None:
        return comp
    sim = float(pred.value)
    delta = sim - ref
    if comp.bound == LOWER and delta >= 0.0 or comp.bound == UPPER and delta <= 0.0:
        delta_eff = 0.0
    else:
        delta_eff = delta
    comp.error_rel = delta_eff / comp.scale if comp.scale and comp.scale > 0.0 else (0.0 if delta_eff == 0.0 else
                                                                                      math.copysign(math.inf, delta_eff))
    if comp.uncertainty is not None:
        comp.within_uncertainty = abs(delta_eff) <= comp.uncertainty
    if comp.tolerance is not None:
        comp.within_tolerance = abs(delta_eff) <= comp.tolerance
        comp.miss_ratio = abs(delta_eff) / comp.tolerance if comp.tolerance > 0.0 else (
            0.0 if delta_eff == 0.0 else math.inf)
    return comp


def compare_case(case: Case, out: HarnessOutput, budget: B.Budget | None) -> list[Comparison]:
    """Every reference value and curve point of the case against the harness output."""
    values = case.reference_values
    curves = case.reference_curves
    if len(out.values) != len(values):
        raise ValueError(f"{case.id}: the harness returned {len(out.values)} predictions for {len(values)} values")
    if len(out.curves) != len(curves):
        raise ValueError(f"{case.id}: the harness returned {len(out.curves)} curves for {len(curves)} curves")
    comps: list[Comparison] = []
    for i, (entry, pred) in enumerate(zip(values, out.values)):
        comps.append(compare_one(where="value", index=i, point=None, name=str(entry.get("name")),
                                 reference=entry.get("value"), unit=str(entry.get("unit") or ""), pred=pred,
                                 x=entry.get("x"), x_name=str(entry.get("x_name") or ""),
                                 x_unit=str(entry.get("x_unit") or ""), entry_unc=entry.get("uncertainty"),
                                 case_pct=case.uncertainty_pct, ref_note=str(entry.get("note") or ""),
                                 ambient_K=out.ambient_K, budget=budget))
    for c, (curve, preds) in enumerate(zip(curves, out.curves)):
        points = curve.get("points") or []
        if len(preds) != len(points):
            raise ValueError(f"{case.id}: curve {curve.get('name')}: {len(preds)} predictions for {len(points)} points")
        for k, (pt, pred) in enumerate(zip(points, preds)):
            comps.append(compare_one(where="curve", index=c, point=k, name=f"{curve.get('name')}",
                                     reference=pt[1], unit=str(curve.get("y_unit") or ""), pred=pred, x=pt[0],
                                     x_name=str(curve.get("x_name") or ""), x_unit=str(curve.get("x_unit") or ""),
                                     curve_unc=curve.get("uncertainty"), case_pct=case.uncertainty_pct,
                                     ref_note=str(curve.get("note") or ""), ambient_K=out.ambient_K, budget=budget))
    return comps


# ------------------------------------------------------------------------------------------ statistics
def abs_errors(comps: list[Comparison], where: str | None = None) -> np.ndarray:
    """|e| of the compared scalar values with a finite error (classes and no-prediction values excluded;
    ``where`` 'value' or 'curve' selects reference values or curve points)."""
    vals = [abs(c.error_rel) for c in comps if c.status == COMPARED and c.kind != CLASS and c.error_rel is not None
            and math.isfinite(c.error_rel) and (where is None or c.where == where)]
    return np.array(vals, dtype=float)


def class_counts(comps: list[Comparison]) -> tuple[int, int]:
    """(matches, counted classifications)."""
    cls = [c for c in comps if c.kind == CLASS and c.counted]
    return sum(1 for c in cls if c.error_rel == 0.0), len(cls)


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def curve_errors(comps: list[Comparison]) -> list[dict]:
    """RMS and max |e| of each curve over its compared points."""
    out = []
    keys = sorted({c.index for c in comps if c.where == "curve"})
    for key in keys:
        pts = [c for c in comps if c.where == "curve" and c.index == key]
        errs = abs_errors(pts)
        out.append({"curve": pts[0].name if pts else "", "index": key, "points": len(pts),
                    "compared": int(errs.size), "rms_rel": float(np.sqrt(np.mean(errs ** 2))) if errs.size else None,
                    "max_rel": float(errs.max()) if errs.size else None})
    return out


@dataclass
class CaseStats:
    compared: int
    no_prediction: int
    not_compared: int
    median_abs_rel: float | None
    max_abs_rel: float | None
    rms_rel: float | None
    within_tolerance: int | None
    within_uncertainty: int | None
    uncertainty_known: int
    passes: bool | None            # None without a budget
    max_miss_ratio: float | None
    classification: bool
    class_matches: int = 0
    class_values: int = 0


def case_stats(comps: list[Comparison], budgeted: bool) -> CaseStats:
    counted = [c for c in comps if c.counted]
    errs = abs_errors(counted)
    n_comp = sum(1 for c in counted if c.status == COMPARED)
    n_none = sum(1 for c in counted if c.status == NO_PREDICTION)
    n_skip = sum(1 for c in comps if c.status == NOT_COMPARED)
    with_unc = [c for c in counted if c.within_uncertainty is not None]
    classification = bool(counted) and all(c.kind == CLASS for c in counted)
    if budgeted and counted:
        tol_known = [c for c in counted if c.within_tolerance is not None]
        within = sum(1 for c in tol_known if c.within_tolerance)
        passes = len(tol_known) == len(counted) and within == len(counted)
        ratios = [c.miss_ratio for c in counted if c.miss_ratio is not None]
        max_miss = max(ratios) if ratios else None
    else:
        within = None
        passes = None
        max_miss = None
    matches, n_class = class_counts(counted)
    return CaseStats(compared=n_comp, no_prediction=n_none, not_compared=n_skip,
                     median_abs_rel=percentile(errs, 50.0), max_abs_rel=float(errs.max()) if errs.size else None,
                     rms_rel=float(np.sqrt(np.mean(errs ** 2))) if errs.size else None,
                     within_tolerance=within, within_uncertainty=sum(1 for c in with_unc if c.within_uncertainty),
                     uncertainty_known=len(with_unc), passes=passes, max_miss_ratio=max_miss,
                     classification=classification, class_matches=matches, class_values=n_class)


__all__ = ["Comparison", "compare_one", "compare_case", "case_stats", "CaseStats", "curve_errors", "abs_errors",
           "class_counts", "percentile", "bound_of", "LOWER", "UPPER", "NONE"]
