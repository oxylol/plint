"""Fast tests of the validation framework: cases, metrics, budgets, provenance, runner and report
(no harness physics; the slow suite in tests/validation runs the harnesses)."""

from __future__ import annotations

import json
import math

import pytest

from cnctwin.validation import budgets as B
from cnctwin.validation import metrics as M
from cnctwin.validation import provenance as PV
from cnctwin.validation import registry as R
from cnctwin.validation import report as RP
from cnctwin.validation import runner as RN
from cnctwin.validation.cases import Case, difference_to_si, find_test, from_si, per_test_table, to_si, unit_of


def make_case(**over) -> Case:
    raw = {
        "format": "cnctwin-validation-case", "version": 1, "id": "fake-case", "quantity": "fake_quantity",
        "reference": {"src": "paper-a", "location": "Table 1", "measured": True, "uncertainty_pct": 5},
        "setup": {"operation": "orthogonal_turning", "material": {"db": "aisi1045"},
                  "tool": {"rake_deg": 0, "edge_radius_um": {"value": 20, "src": "est", "note": "assumed hone"},
                           "clearance_deg": 7, "edge_radius_per_test_um": [30, 40], "assumed_inputs": ["clearance_deg"]},
                  "conditions": {"test_ids": ["T1", "T2"], "vc_per_test_m_min": [60, 120], "feed_mm_rev": 0.1,
                                 "ambient_temp_degC": 25}},
        "reference_values": [{"name": "chip_ratio T1 (derived)", "value": 0.5, "unit": ""},
                             {"name": "T_chip T2", "value": 360, "unit": "degC", "note": "top of the table: lower bound"}],
        "reference_curves": [{"name": "c", "x_name": "vc", "x_unit": "m/min", "y_name": "T", "y_unit": "degC",
                              "uncertainty": 4, "points": [[60, 300], [120, 400]]}],
    }
    raw.update(over)
    return Case(id=raw["id"], quantity=raw["quantity"], file="validation/cases/fake.yaml", raw=raw)


# ------------------------------------------------------------------------------------------ cases and units
def test_reference_units_convert_with_the_database_unit_parser():
    assert to_si(150, "m/min") == pytest.approx(2.5)
    assert to_si(37.5, "um") == pytest.approx(37.5e-6)
    assert to_si(20, "degC") == pytest.approx(293.15)
    assert from_si(293.15, "degC") == pytest.approx(20.0)
    assert difference_to_si(8, "degC") == pytest.approx(8.0)          # a difference has no offset
    assert unit_of("index") is None and unit_of("") is None and unit_of(None) is None
    assert to_si(3, "index") == 3.0


def test_leaves_unwrap_values_and_mark_assumed_inputs():
    case = make_case()
    leaf = case.leaf("tool", "edge_radius_um")
    assert leaf.value == 20 and leaf.assumed and leaf.src == "est"
    assert case.si("tool", "edge_radius_um") == pytest.approx(20e-6)
    assumed = {a.path for a in case.assumed_inputs()}
    assert assumed == {"setup.tool.edge_radius_um", "setup.tool.clearance_deg"}
    assert case.leaf("tool", "missing_mm") is None


def test_per_test_lists_spread_over_named_tests_and_match_references():
    case = make_case()
    rows = per_test_table(case)
    assert [r.test_id for r in rows] == ["T1", "T2"]
    assert rows[1].si("vc_m_min") == pytest.approx(2.0)
    assert rows[0].si("edge_radius_um") == pytest.approx(30e-6)        # per-test list overrides the shared value
    assert rows[0].si("feed_mm_rev") == pytest.approx(1e-4)
    assert find_test(rows, {"name": "chip_ratio T2 (derived)"}).test_id == "T2"
    assert find_test(rows, {"name": "x", "x": 120, "x_unit": "m/min"}, ("vc_m_min",)).test_id == "T2"
    assert find_test(rows, {"name": "phi test 2", "x": 1}, ()) is None


# ------------------------------------------------------------------------------------------ metrics
def test_relative_error_and_the_budget_tolerance_rule():
    budget = B.Budget("q", "m", 20.0)
    c = M.compare_one(where="value", index=0, point=None, name="r", reference=0.5, unit="", pred=R.Prediction(0.6),
                      budget=budget, entry_unc=0.06)
    assert c.error_rel == pytest.approx(0.2)
    assert c.tolerance == pytest.approx(max(0.2 * 0.5, 2 * 0.06))     # max(pct x |ref|, 2 u)
    assert c.within_tolerance and not c.within_uncertainty
    assert c.miss_ratio == pytest.approx(0.1 / 0.12)


def test_temperature_error_is_relative_to_the_rise_above_ambient():
    budget = B.Budget("t", "m", 15.0)
    c = M.compare_one(where="value", index=0, point=None, name="T", reference=520.0, unit="degC",
                      pred=R.Prediction(470.0), ambient_K=293.15, budget=budget)
    assert c.temperature and c.scale == pytest.approx(500.0)
    assert c.error_rel == pytest.approx(-0.1)
    assert c.tolerance == pytest.approx(75.0) and c.within_tolerance
    with pytest.raises(ValueError):
        M.compare_one(where="value", index=0, point=None, name="T", reference=520.0, unit="degC",
                      pred=R.Prediction(470.0), ambient_K=None)


def test_lower_bound_references_accept_values_above_them():
    above = M.compare_one(where="value", index=0, point=None, name="T", reference=360.0, unit="degC",
                          pred=R.Prediction(400.0), ambient_K=293.15, ref_note="top of the table: lower bound")
    below = M.compare_one(where="value", index=0, point=None, name="T", reference=360.0, unit="degC",
                          pred=R.Prediction(300.0), ambient_K=293.15, ref_note="lower bound")
    assert above.bound == M.LOWER and above.error_rel == 0.0
    assert below.error_rel == pytest.approx(-60.0 / 340.0)


def test_classes_match_or_miss_and_missing_predictions_count_as_misses():
    hit = M.compare_one(where="value", index=0, point=None, name="g", reference=1, unit="",
                        pred=R.Prediction("+", kind=R.CLASS, reference_class="+"))
    miss = M.compare_one(where="value", index=0, point=None, name="g", reference=1, unit="",
                         pred=R.Prediction("-", kind=R.CLASS, reference_class="+"))
    none = M.compare_one(where="value", index=0, point=None, name="g", reference=0.5, unit="",
                         pred=R.no_prediction("no chip"), budget=B.Budget("q", "m", 20.0))
    assert hit.error_rel == 0.0 and hit.within_tolerance
    assert miss.error_rel == 1.0 and not miss.within_tolerance
    assert none.within_tolerance is False and math.isinf(none.miss_ratio)


def test_uncertainty_priority_prediction_entry_curve_case():
    pred = R.Prediction(1.0, uncertainty=0.3)
    assert M._uncertainty(pred, 0.2, 0.1, 5, 2.0) == 0.3
    assert M._uncertainty(R.Prediction(1.0), 0.2, 0.1, 5, 2.0) == 0.2
    assert M._uncertainty(R.Prediction(1.0), None, 0.1, 5, 2.0) == 0.1
    assert M._uncertainty(R.Prediction(1.0), None, None, 5, 2.0) == pytest.approx(0.1)
    assert M._uncertainty(R.Prediction(1.0), None, None, None, 2.0) is None


def test_case_comparison_and_statistics():
    case = make_case()
    out = R.HarnessOutput([R.Prediction(0.55), R.Prediction(380.0)],
                          [[R.Prediction(310.0), R.not_compared("excluded")]], ["fake"], ambient_K=293.15)
    budget = B.Budget("fake_quantity", "m", 20.0)
    comps = M.compare_case(case, out, budget)
    assert len(comps) == 4 and comps[3].status == R.NOT_COMPARED
    stats = M.case_stats(comps, True)
    assert stats.compared == 3 and stats.not_compared == 1 and stats.passes
    curves = M.curve_errors(comps)
    assert curves[0]["compared"] == 1 and curves[0]["rms_rel"] == pytest.approx(10.0 / 280.0)
    with pytest.raises(ValueError):
        M.compare_case(case, R.HarnessOutput([R.Prediction(0.5)], [[]], ["fake"]), budget)


# ------------------------------------------------------------------------------------------ budgets
def test_budgets_are_read_from_the_file_with_their_rule():
    budgets = B.load_budgets()
    assert budgets["tool_temperature"].pct == 15
    assert budgets["chip_ratio"].pct == 20
    assert budgets["cutting_force"].components["thrust_force_pct"] == 30
    assert budgets["cutting_force"].pct_for(None) is None
    assert "taylor_life" not in budgets
    header = B.header_text().lower()
    assert "at least 80 %" in header and "twice its tolerance" in header and "2 * uncertainty" in header
    assert B.tolerance(None, 1.0, 0.1) is None
    assert B.tolerance(10.0, 2.0, 0.15) == pytest.approx(0.3)


def test_quantity_verdict_follows_the_budget_rule():
    budget = B.Budget("q", "m", 20.0)
    stats = RN.GroupStats(cases=5, cases_with_values=5, passing=4, values=10, compared=10, no_prediction=0,
                          median_abs_rel=0.1, p90_abs_rel=0.2, within_uncertainty=3, uncertainty_known=10,
                          worst_case="c5", worst_miss_ratio=1.9)
    assert RN.verdict(stats, budget, 0)[0] == RN.PASS
    stats.worst_miss_ratio = 2.5
    assert RN.verdict(stats, budget, 0)[0] == RN.FAIL
    stats.worst_miss_ratio, stats.passing = 1.0, 3
    assert RN.verdict(stats, budget, 0)[0] == RN.FAIL
    assert RN.verdict(stats, None, 0)[0] == RN.NO_BUDGET
    assert RN.verdict(stats, budget, 1)[0] == RN.FAIL


# ------------------------------------------------------------------------------------------ provenance
def test_works_group_keys_by_doi_url_title_note_and_the_data_table():
    entries = {
        "a-2020": {"title": "A study", "year": 2020, "doi": "10.1/abc"},
        "a-2020-copy": {"title": "Another title", "year": 2020, "doi": "10.1/ABC"},
        "b-2021": {"title": "B data", "year": 2021, "url": "https://x.org/b"},
        "b-2021-copy": {"title": "B, data!", "year": 2021},
        "c-2022": {"title": "C", "year": 2022, "note": "Same work as c-2022-other in another file."},
        "c-2022-other": {"title": "C prime", "year": 2022},
        "d-2023": {"title": "D", "year": 2023, "note": "compares with c-2022 (not the same)"},
        "klippel-2022-arxiv-orthogonal-data": {"title": "x", "year": 2022},
        "klippel-2024-ck45-orthogonal": {"title": "y", "year": 2024},
    }
    works = PV.build_works(entries)
    assert works.same("a-2020", "a-2020-copy")
    assert works.same("b-2021", "b-2021-copy")
    assert works.same("c-2022", "c-2022-other")
    assert not works.same("d-2023", "c-2022")
    assert works.same("klippel-2022-arxiv-orthogonal-data", "klippel-2024-ck45-orthogonal")


def test_dependencies_count_sources_and_estimates_fitted_to_the_reference():
    works = PV.build_works({"paper-a": {"title": "A", "year": 2000}, "paper-b": {"title": "B", "year": 2001}})
    used = [R.Provenance("flow stress", "paper-a"), R.Provenance("friction", "paper-b"),
            R.Provenance("Taylor fit", "est", "FITTED HERE to the tables of paper-a"),
            R.Provenance("conductivity", "paper-b", "paper-a gives another value")]
    deps = PV.dependencies("paper-a", used, works)
    assert [(d.what, d.via) for d in deps] == [("flow stress", "src"), ("Taylor fit", "note")]


# ------------------------------------------------------------------------------------------ registry, runner, report
def test_registry_refuses_a_second_harness_for_a_quantity():
    def one(case, db):
        return None

    def two(case, db):
        return None
    R.register("unit_test_quantity", one)
    try:
        with pytest.raises(ValueError):
            R.register("unit_test_quantity", two)
        R.register("unit_test_quantity", two, replace=True)
        assert R.HARNESSES["unit_test_quantity"].function is two
    finally:
        R.HARNESSES.pop("unit_test_quantity", None)
    with pytest.raises(ValueError):
        R.Prediction(None, R.COMPARED)
    with pytest.raises(ValueError):
        R.Prediction(1.0, "unknown")


def _fake_result(passes: bool, dependent: bool) -> RN.CaseResult:
    case = make_case(id="dep" if dependent else ("pass" if passes else "fail"))
    sim = 0.52 if passes else 0.9
    out = R.HarnessOutput([R.Prediction(sim), R.Prediction(370.0)], [[R.Prediction(300.0), R.Prediction(410.0)]],
                          ["fake model"], {"setting": 1.0}, [R.Assumption("x", "1", "because")],
                          [R.Provenance("value", "paper-a" if dependent else "paper-z")], ["a note"], 293.15)
    budget = B.Budget("fake_quantity", "m", 20.0)
    comps = M.compare_case(case, out, budget)
    deps = PV.dependencies(case.src, out.provenance, PV.build_works({"paper-a": {"title": "A", "year": 1}}))
    return RN.CaseResult(case, "ok", out, comps, M.case_stats(comps, True), deps)


def test_report_lists_every_case_and_says_plainly_what_passes():
    results = [_fake_result(True, False), _fake_result(False, False), _fake_result(True, True)]
    budget = B.Budget("fake_quantity", "metric text", 20.0)
    indep = [r for r in results if not r.calibration_dependent]
    q = RN.QuantityResult("fake_quantity", budget, "tests.fake", results, RN._group_stats(indep, True),
                          RN._group_stats([results[2]], True))
    q.verdict, q.verdict_reason = RN.verdict(q.independent, budget, 0)
    assert q.verdict == RN.FAIL and q.independent.passing == 1
    md = RP.markdown([q], {"fake_quantity": {"summary": "because", "fail": "cause"}})
    assert md.startswith("# CNC Twin validation report")
    assert "| fake_quantity | 3 | 2 | 1 | 1/2 |" in md and "| independent | 2 | 1/2 |" in md
    assert "**Fail**: fake_quantity." in md
    for case_id in ("pass", "fail", "dep"):
        assert f"### {case_id}" in md
    assert "calibration-dependent" in md and "Likely cause: cause" in md
    doc = RP.quantity_json(q)
    text = RP.dumps(doc)
    assert RP.dumps(json.loads(text)) == text                       # stable, sorted, plain JSON
    assert doc["verdict"] == RN.FAIL and len(doc["cases"]) == 3
    assert doc["cases"][2]["calibration_dependent"] is True


def test_significant_digit_rounding_is_stable():
    assert RP.sig(1.23456789) == 1.23457
    assert RP.sig(float("inf")) == "inf"
    assert RP.sig({"a": [0.1 + 0.2]}) == {"a": [0.3]}
