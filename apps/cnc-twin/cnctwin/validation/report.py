"""Validation report: markdown (docs/validation-report.md) and JSON (validation/results/<quantity>.json).

Both list every case with its source and location, every reference value with the simulated value,
the error, the tolerance and whether it is within budget, the inputs that were assumed (those the case
marks ``src: est`` and those the harness had to add), the model chain and settings, and per quantity a
verdict; the markdown starts with a summary table that says plainly which quantities pass. Output is
deterministic (no dates, floats rounded to 6 significant digits, keys sorted).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import budgets as B
from .metrics import Comparison
from .registry import CLASS, NO_PREDICTION, NOT_COMPARED
from .runner import FAIL, NO_BUDGET, NO_CASES, NO_HARNESS, PASS, CaseResult, GroupStats, QuantityResult

FORMAT = "cnctwin-validation-results"
VERSION = 1


# ------------------------------------------------------------------------------------------ helpers
def sig(value, digits: int = 6):
    """Float rounded to ``digits`` significant digits (JSON/markdown stability); others unchanged."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
        if value == 0.0:
            return 0.0
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {str(k): sig(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sig(v, digits) for v in value]
    if hasattr(value, "item"):
        return sig(value.item(), digits)
    return value


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}g}"
    return str(value)


def pct(value, signed: bool = False) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "inf"
    return f"{value * 100:+.1f} %" if signed else f"{value * 100:.1f} %"


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


# ------------------------------------------------------------------------------------------ JSON
def comparison_json(c: Comparison) -> dict:
    return {"where": c.where, "index": c.index, "point": c.point, "name": c.name, "x": c.x, "x_name": c.x_name,
            "x_unit": c.x_unit, "reference": c.reference, "unit": c.unit, "simulated": c.simulated,
            "status": c.status, "kind": c.kind, "bound": c.bound, "temperature": c.temperature, "scale": c.scale,
            "error_rel": c.error_rel, "uncertainty": c.uncertainty, "tolerance": c.tolerance,
            "budget_pct": c.budget_pct, "component": c.component, "within_tolerance": c.within_tolerance,
            "within_uncertainty": c.within_uncertainty, "miss_ratio": c.miss_ratio, "note": c.note,
            "reference_note": c.reference_note, "extra": c.extra}


def _stats_json(s: GroupStats | None) -> dict | None:
    if s is None:
        return None
    return dict(s.__dict__)


def case_json(r: CaseResult) -> dict:
    case = r.case
    out = {"id": case.id, "file": case.file, "quantity": case.quantity, "src": case.src, "location": case.location,
           "measured": case.measured, "digitised": case.digitised, "uncertainty_pct": case.uncertainty_pct,
           "operation": case.operation, "material": case.material_id, "status": r.status, "error": r.error,
           "calibration_dependent": r.calibration_dependent,
           "dependencies": [d.__dict__ for d in r.dependencies],
           "case_assumed_inputs": [{"input": a.path, "value": a.value, "src": a.src, "note": a.note}
                                   for a in case.assumed_inputs()]}
    if r.output is not None:
        o = r.output
        out.update({"model_chain": list(o.model_chain), "settings": o.settings, "ambient_K": o.ambient_K,
                    "harness_assumptions": [a.__dict__ for a in o.assumptions],
                    "provenance": [p.__dict__ for p in o.provenance], "notes": list(o.notes)})
    if r.stats is not None:
        out["stats"] = dict(r.stats.__dict__)
        out["values"] = [comparison_json(c) for c in r.comparisons if c.where == "value"]
        out["curves"] = [comparison_json(c) for c in r.comparisons if c.where == "curve"]
        from .metrics import curve_errors
        out["curve_errors"] = curve_errors(r.comparisons)
    return out


def quantity_json(q: QuantityResult) -> dict:
    b = q.budget
    return sig({"format": FORMAT, "version": VERSION, "quantity": q.quantity, "harness": q.harness,
                "budget": None if b is None else {"metric": b.metric, "pct": b.pct, "components": b.components,
                                                  "src": b.src, "note": b.note},
                "rule": B.RULE_TEXT, "verdict": q.verdict, "verdict_reason": q.verdict_reason,
                "independent": _stats_json(q.independent), "calibration_dependent": _stats_json(q.dependent),
                "cases": [case_json(r) for r in q.cases]})


def dumps(document) -> str:
    return json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_results(results: list[QuantityResult], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for q in results:
        path = directory / f"{q.quantity}.json"
        path.write_text(dumps(quantity_json(q)), encoding="utf-8")
        paths.append(path)
    return paths


# ------------------------------------------------------------------------------------------ markdown
VERDICT_TEXT = {PASS: "**pass**", FAIL: "**fail**", NO_BUDGET: "not judged (no budget)",
                NO_CASES: "not judged (no independent case)", NO_HARNESS: "no harness yet"}


def _frac(a, b) -> str:
    return f"{a}/{b}" if b else "-"


def _summary_row(q: QuantityResult) -> str:
    s = q.independent
    dep = q.dependent.cases if q.dependent else 0
    n_cases = len(q.cases)
    budget = q.budget.describe() if q.budget else "-"
    if s is None:
        return f"| {q.quantity} | {n_cases} | - | - | - | - | - | - | - | {budget} | {VERDICT_TEXT[q.verdict]} |"
    within = f"{s.passing}/{s.cases_with_values}" if s.passing is not None else "-"
    return (f"| {q.quantity} | {n_cases} | {s.cases} | {dep} | {within} | {pct(s.median_abs_rel)} | "
            f"{pct(s.p90_abs_rel)} | {_frac(s.class_matches, s.class_values)} | "
            f"{_frac(s.within_uncertainty, s.uncertainty_known)} | {budget} | {VERDICT_TEXT[q.verdict]} |")


PLAIN_VERDICT = {PASS: "pass", FAIL: "fail", NO_BUDGET: "not judged (no budget)",
                 NO_CASES: "not judged (no independent case)", NO_HARNESS: "no harness"}


def _plain_row(q: QuantityResult) -> str:
    s = q.independent
    dep = q.dependent.cases if q.dependent else 0
    if s is None:
        return f"| {q.quantity} | {len(q.cases)} | - | - | - | - | {PLAIN_VERDICT[q.verdict]} |"
    within = f"{s.passing}/{s.cases_with_values}" if s.passing is not None else "-"
    median = pct(s.median_abs_rel)
    if s.class_values and s.class_values == s.values:
        median = f"{s.class_matches}/{s.class_values} classes match"
    return (f"| {q.quantity} | {len(q.cases)} | {s.cases} | {dep} | {within} | {median} | "
            f"{PLAIN_VERDICT[q.verdict]} |")


def _stats_table(q: QuantityResult) -> list[str]:
    lines = ["| group | cases | cases within budget | counted values (no prediction) | median abs. error (all / values / "
             "curve points) | p90 abs. error | class matches | values within uncertainty | largest miss (tolerances) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, s in (("independent", q.independent), ("calibration-dependent", q.dependent)):
        if s is None:
            continue
        within = f"{s.passing}/{s.cases_with_values}" if s.passing is not None else "-"
        worst = f"{fmt(s.worst_miss_ratio, 3)} ({s.worst_case})" if s.worst_miss_ratio is not None else "-"
        lines.append(f"| {label} | {s.cases} | {within} | {s.values} ({s.no_prediction}) | {pct(s.median_abs_rel)} / "
                     f"{pct(s.median_values_abs_rel)} / {pct(s.median_curves_abs_rel)} | {pct(s.p90_abs_rel)} | "
                     f"{_frac(s.class_matches, s.class_values)} | {_frac(s.within_uncertainty, s.uncertainty_known)} | "
                     f"{worst} |")
    return lines


def _within_text(c: Comparison) -> str:
    if c.status == NOT_COMPARED:
        return "not compared"
    if c.status == NO_PREDICTION:
        return "**no prediction**"
    if c.within_tolerance is None:
        return "-" if c.within_uncertainty is None else ("in unc." if c.within_uncertainty else "outside unc.")
    return "yes" if c.within_tolerance else "**no**"


def _value_rows(comps: list[Comparison]) -> list[str]:
    lines = ["| reference | x | ref. | sim. | unit | error | tolerance | within | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        x = "-" if c.x is None else f"{fmt(c.x)} {c.x_unit}".strip()
        err = "-" if c.error_rel is None else (("match" if c.error_rel == 0 else "mismatch") if c.kind == CLASS
                                               else pct(c.error_rel, signed=True))
        tol = "-" if c.tolerance is None else fmt(c.tolerance, 3)
        bound = f" ({c.bound} bound)" if c.bound != "none" else ""
        note = c.note
        lines.append(f"| {_cell(c.name)}{bound} | {_cell(x)} | {fmt(c.reference)} | {fmt(c.simulated)} | "
                     f"{_cell(c.unit)} | {err} | {tol} | {_within_text(c)} | {_cell(note)} |")
    return lines


def _case_section(r: CaseResult, discussion: dict) -> list[str]:
    case = r.case
    label = "calibration-dependent" if r.calibration_dependent else "independent"
    lines = [f"### {case.id}", ""]
    kind = "measured" if case.measured else "analytical/numerical benchmark (measured: false)"
    lines.append(f"* Source: `{case.src}`, {case.location} ({kind}{', digitised' if case.digitised else ''}"
                 f"{', stated uncertainty ' + fmt(case.uncertainty_pct) + ' %' if case.uncertainty_pct is not None else ''}).")
    lines.append(f"* File: `{case.file}`; operation {case.operation}; material `{case.material_id}`.")
    if r.status != "ok":
        lines.append(f"* **{r.status}**: `{_cell(r.error)}`")
        lines.append("")
        return lines
    s = r.stats
    o = r.output
    lines.append(f"* Validation class: **{label}**" + (": " + "; ".join(
        f"{d.what} from `{d.src}` ({'source' if d.via == 'src' else 'named in its note'})" for d in r.dependencies)
        if r.dependencies else "") + ".")
    lines.append(f"* Model chain: {' -> '.join(o.model_chain)}.")
    if o.settings:
        settings = ", ".join(f"{k} = {fmt(v) if not isinstance(v, (list, dict)) else json.dumps(sig(v))}"
                             for k, v in sorted(o.settings.items()))
        lines.append(f"* Settings: {_cell(settings)}.")
    assumed = case.assumed_inputs()
    if assumed:
        lines.append("* Inputs the case marks as assumed: " + "; ".join(
            f"`{a.path}` = {_cell(json.dumps(sig(a.value)) if not isinstance(a.value, str) else a.value)}"
            + (f" ({_cell(a.note)})" if a.note else "") for a in assumed) + ".")
    if o.assumptions:
        lines.append("* Inputs the harness assumed: " + "; ".join(
            f"{_cell(a.input)} = {_cell(a.value)} ({_cell(a.reason)}{', ' + a.source if a.source else ''})"
            for a in o.assumptions) + ".")
    if o.provenance:
        lines.append("* Database values used: " + "; ".join(f"{_cell(p.what)} (`{p.src or 'none'}`)"
                                                             for p in o.provenance) + ".")
    for note in o.notes:
        lines.append(f"* {_cell(note)}")
    result = "not judged (no budget)" if s.passes is None else ("within budget" if s.passes else "**outside budget**")
    parts = [f"{s.compared} compared, {s.no_prediction} without prediction, {s.not_compared} not compared"]
    if s.median_abs_rel is not None:
        parts.append(f"median abs. error {pct(s.median_abs_rel)}, max {pct(s.max_abs_rel)}")
    if s.class_values:
        parts.append(f"{s.class_matches}/{s.class_values} classes match")
    if s.uncertainty_known:
        parts.append(f"{s.within_uncertainty}/{s.uncertainty_known} within uncertainty")
    miss = f"; largest miss {fmt(s.max_miss_ratio, 3)} tolerances" if s.max_miss_ratio is not None else ""
    lines.append(f"* Result: {'; '.join(parts)}; case {result}{miss}.")
    extra = discussion.get(case.id)
    if extra:
        lines.append(f"* Likely cause: {_cell(extra)}")
    lines.append("")
    values = [c for c in r.comparisons if c.where == "value"]
    if values:
        lines.extend(_value_rows(values))
        lines.append("")
    curves = [c for c in r.comparisons if c.where == "curve"]
    if curves:
        from .metrics import curve_errors
        for ce in curve_errors(r.comparisons):
            pts = [c for c in curves if c.index == ce["index"]]
            lines.append(f"Curve `{ce['curve']}` ({pts[0].x_name} [{pts[0].x_unit}] -> {pts[0].unit}): "
                         f"RMS abs. error {pct(ce['rms_rel'])}, max {pct(ce['max_rel'])} over {ce['compared']} "
                         f"compared points.")
            lines.append("")
            lines.extend(_value_rows(pts))
            lines.append("")
    return lines


def markdown(results: list[QuantityResult], discussions: dict | None = None, intro: dict | None = None) -> str:
    """The whole report. ``discussions[quantity]`` = {"summary": text, case_id: likely cause};
    ``intro[quantity]`` = text describing the harness."""
    discussions = discussions or {}
    intro = intro or {}
    lines = ["# CNC Twin validation report", "",
             "Generated by `twin validate` (`cnctwin.validation`, see docs/validation.md) from the cases in "
             "`validation/cases/` against the budgets of `validation/budgets.yaml`. Every case is simulated by "
             "its quantity's harness with the domain libraries at the level the experiment measured; models take "
             "every parameter from the databases and the case's stated inputs (nothing is tuned to a reference). "
             "Errors are relative to the reference (temperatures: to the rise above ambient). Cases whose "
             "reference also built a database value the model used are **calibration-dependent** and are "
             "reported separately; verdicts use the independent cases only.", "",
             "## Summary", "",
             "Plain summary (verdicts use the independent cases only; 'within budget' counts independent cases whose "
             "every value lies inside its tolerance; the median is the median absolute relative error over the "
             "independent cases' compared values):", "",
             "| quantity | cases | independent | calibration-dependent | within budget | median error | verdict |",
             "|---|---|---|---|---|---|---|"]
    for q in results:
        lines.append(_plain_row(q))
    lines += ["", "Detailed statistics:", "",
              "| quantity | cases | independent | calibration-dependent | independent cases within budget | "
              "median abs. error | p90 abs. error | class matches | values within uncertainty | budget | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in results:
        lines.append(_summary_row(q))
    lines.append("")
    groups = {PASS: [], FAIL: [], NO_BUDGET: [], NO_CASES: [], NO_HARNESS: []}
    for q in results:
        groups[q.verdict].append(q.quantity)
    lines.append(f"* **Pass**: {', '.join(groups[PASS]) or 'none'}.")
    lines.append(f"* **Fail**: {', '.join(groups[FAIL]) or 'none'}.")
    lines.append(f"* Not judged, no budget in validation/budgets.yaml: {', '.join(groups[NO_BUDGET]) or 'none'}.")
    if groups[NO_CASES]:
        lines.append(f"* Not judged, no independent case: {', '.join(groups[NO_CASES])}.")
    lines.append(f"* No harness yet: {', '.join(groups[NO_HARNESS]) or 'none'}.")
    lines.append("")
    lines.append(f"Budget rule (validation/budgets.yaml, applied as written): {B.RULE_TEXT}. Values without a "
                 "prediction count as misses; values marked 'not compared' are listed with the reason but do not "
                 "count. The error statistics are over the compared scalar values of the independent cases (reference "
                 "values and curve points; temperatures as rises above ambient); classifications are counted as "
                 "matches.")
    lines.append("")
    dependent = [(q.quantity, r) for q in results for r in q.cases if r.status == "ok" and r.calibration_dependent]
    if dependent:
        lines.append("Calibration-dependent cases (excluded from the verdicts):")
        lines.append("")
        for quantity, r in dependent:
            lines.append(f"* `{r.id}` ({quantity}): " + "; ".join(f"{d.what} from `{d.src}`" for d in r.dependencies))
        lines.append("")
    for q in results:
        lines.append(f"## {q.quantity}")
        lines.append("")
        budget = "none in validation/budgets.yaml" if q.budget is None else (
            f"{q.budget.describe()} ({q.budget.metric})")
        lines.append(f"* Budget: {_cell(budget)}.")
        lines.append(f"* Verdict: {VERDICT_TEXT[q.verdict]} - {q.verdict_reason}.")
        if q.harness:
            lines.append(f"* Harness: `{q.harness}`.")
        if intro.get(q.quantity):
            lines.append(f"* {intro[q.quantity]}")
        lines.append("")
        if q.independent is not None:
            lines.extend(_stats_table(q))
            lines.append("")
        summary = (discussions.get(q.quantity) or {}).get("summary")
        if summary:
            lines.append(f"**Failures and likely causes.** {summary}")
            lines.append("")
        for r in q.cases:
            lines.extend(_case_section(r, discussions.get(q.quantity) or {}))
    return "\n".join(lines).rstrip() + "\n"


def summary_text(results: list[QuantityResult]) -> str:
    """Plain-text summary for the terminal."""
    rows = [("quantity", "cases", "indep.", "dep.", "within", "median", "p90", "verdict")]
    for q in results:
        s = q.independent
        dep = q.dependent.cases if q.dependent else 0
        if s is None:
            rows.append((q.quantity, str(len(q.cases)), "-", "-", "-", "-", "-", q.verdict))
            continue
        within = f"{s.passing}/{s.cases_with_values}" if s.passing is not None else "-"
        rows.append((q.quantity, str(len(q.cases)), str(s.cases), str(dep), within, pct(s.median_abs_rel),
                     pct(s.p90_abs_rel), q.verdict))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows) + "\n"


__all__ = ["markdown", "summary_text", "quantity_json", "case_json", "write_results", "dumps", "sig", "FORMAT",
           "VERSION"]
