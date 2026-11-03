"""Uncertainty studies end to end with a fast surrogate of known response: the factors from the stated
tolerances, variants that carry every sampled value in the project file (inline material included),
the distribution and capability of each dimension against the analytic values, finding probabilities,
the sensitivity ranking, failures, caching, determinism, the API entry, the CLI and the report section."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import write_project
from htmlcheck import check
from scipy import stats as sps

from cnctwin.db.models import params
from cnctwin.db.yamlio import load_file
from cnctwin.project import load_project, resolve
from cnctwin.uncertainty.driver import RunnerMissing, plan_study, resolve_runner, run_study, study
from cnctwin.uncertainty.factors import factors_for
from cnctwin.uncertainty.sampling import TruncatedNormal, Uniform

SURROGATE = f"{Path(__file__).resolve().parent / 'surrogate.py'}:run"


def _factors(tmp_path, **kwargs):
    project = load_project(write_project(tmp_path / "p", **kwargs))
    return factors_for(project.data, project.dir, params("uncertainty"))


def test_factors_from_the_stated_tolerances(tmp_path):
    factors, skipped = _factors(tmp_path)
    by_name = {f.name: f for f in factors}
    assert sorted(by_name) == ["clamping_force[OP10.0]", "hardness_scale", "kc_scale", "length_offset[T1]",
                               "radius_offset[T1]", "runout[T1]", "stock_size[0]", "stock_size[1]", "stock_size[2]"]
    assert by_name["stock_size[2]"].distribution == Uniform(pytest.approx(0.0149), pytest.approx(0.0151))
    assert by_name["length_offset[T1]"].distribution == Uniform(pytest.approx(-5e-6), pytest.approx(5e-6))
    assert by_name["runout[T1]"].distribution == Uniform(pytest.approx(2e-6), pytest.approx(8e-6))
    assert by_name["kc_scale"].distribution == TruncatedNormal(1.0, 0.04, pytest.approx(0.88), pytest.approx(1.12))
    assert by_name["hardness_scale"].distribution.sigma == pytest.approx(0.03)
    assert by_name["kc_scale"].stated_by == "material.batch.kc_scale_sigma = 0.04"
    assert by_name["clamping_force[OP10.0]"].applies == "project" and by_name["kc_scale"].applies == "material"
    assert by_name["clamping_force[OP10.0]"].target.endswith("input_torque_Nm") and not skipped
    factors, _ = _factors(tmp_path / "b", vary="stock_size: false, tool_offsets_um: 5, coolant_reach_rel: 0.2, "
                                                "kc_scale_sigma: 0")
    names = {f.name: f for f in factors}
    assert "kc_scale" not in names and not any(n.startswith("stock_size") for n in names)
    assert names["coolant_reach_scale"].applies == "options" and names["coolant_reach_scale"].distribution == \
        Uniform(pytest.approx(0.8), pytest.approx(1.2))


def test_variants_carry_every_value(tmp_path, db, base_material):
    project_path = write_project(tmp_path / "p")
    plan = plan_study(project_path, tmp_path / "study", options={"runs": 6}, db=db)
    assert len(plan["samples"]) == 6 and (tmp_path / "study" / "plan.json").is_file()
    for sample in plan["samples"]:
        path = tmp_path / "study" / sample["project"]
        data = load_file(path).data
        values = sample["values"]
        assert data["stock"]["size_mm"][2] == pytest.approx(values["stock_size[2]"] * 1e3)
        tool = data["tools"]["list"][0]
        assert tool["length_offset_mm"] == pytest.approx(120.0 + values["length_offset[T1]"] * 1e3)
        assert tool["radius_offset_mm"] == pytest.approx(1.0 + values["radius_offset[T1]"] * 1e3)
        assert tool["runout_um"] == pytest.approx(values["runout[T1]"] * 1e6)
        assert data["operations"][0]["setup"]["workholding"][0]["input_torque_Nm"] == pytest.approx(
            27.0 * values["clamping_force[OP10.0]"])
        material = data["material"]["id"]
        assert material["extends"] == "al6061-t6" and "uncertainty sample" in material["notes"]
        assert material["flow_stress"]["A_MPa"] == pytest.approx(base_material["A_MPa"] * values["kc_scale"])
        assert material["physical"]["hardness"]["HB"] == pytest.approx(base_material["HB"] * values["hardness_scale"])
        assert all("kc11_N_mm2" in entry for entry in material["cutting"]["kienzle"])
        assert Path(data["operations"][0]["program"]).is_absolute() and Path(data["operations"][0]["program"]).is_file()
        assert data["seed"] == 77 and data["material"]["batch"] == {"kc_scale_sigma": 0.04, "hardness_sigma_pct": 3}
    first = tmp_path / "study" / plan["samples"][0]["project"]
    resolved = resolve(first, db)                         # the inline material resolves against the database
    kc = plan["samples"][0]["values"]["kc_scale"]
    assert float(resolved.material.flow_stress.A_Pa) == pytest.approx(base_material["A_MPa"] * kc * 1e6)
    kienzle = resolved.material.cutting.kienzle[0]
    base_kc11 = db.raw("material", "al6061-t6")["cutting"]["kienzle"][0]["kc11_N_mm2"]
    assert float(kienzle.kc11_Pa) == pytest.approx(float(base_kc11) * kc * 1e6)


def test_a_csv_tool_library_is_inlined(tmp_path, db):
    plan = plan_study(write_project(tmp_path / "p", library=True), tmp_path / "study", options={"runs": 2}, db=db)
    data = load_file(tmp_path / "study" / plan["samples"][1]["project"]).data
    assert data["tools"]["library"] is None and len(data["tools"]["list"]) == 1
    assert data["tools"]["list"][0]["length_offset_mm"] == pytest.approx(
        120.0 + plan["samples"][1]["values"]["length_offset[T1]"] * 1e3)


def _analytic_depth_std() -> float:
    kc = sps.truncnorm(-3.0, 3.0, loc=1.0, scale=0.04)
    return math.sqrt((2e-4) ** 2 / 12.0 + (1e-5) ** 2 / 12.0 + (5e-4 * kc.std()) ** 2)


@pytest.fixture(scope="module")
def surrogate_study(tmp_path_factory, base_material):
    out = tmp_path_factory.mktemp("study")
    project = write_project(out / "project", runs=96)
    calls = []
    document = study(project, out / "s", options={"runner": SURROGATE, "simulate_options": {"base": base_material}},
                     progress_cb=lambda progress=None, **fields: calls.append(fields))
    return {"doc": document, "out": out, "project": project, "calls": calls}


def test_distribution_and_capability_match_the_known_response(surrogate_study):
    results = surrogate_study["doc"]["results"]
    assert results["samples_ok"] == 96 and results["samples_failed"] == 0
    depth = next(d for d in results["dimensions"] if d["id"] == "pocket-depth")
    measured = depth["measured"]
    std = _analytic_depth_std()
    assert abs(measured["mean"] - 0.002) < 3.0 * std / math.sqrt(96)
    assert measured["std"] == pytest.approx(std, rel=0.12)
    capability = depth["capability"]
    assert capability["Cp"] == pytest.approx(1e-4 / (6.0 * measured["std"]))
    assert capability["Cpk"] <= capability["Cp"] and capability["capable"] is False
    outside = sum(1 for v in depth["samples"] if abs(v - 0.002) > 5e-5)
    assert depth["out_of_tolerance"]["k"] == outside and depth["out_of_tolerance"]["n"] == 96
    flat = next(d for d in results["dimensions"] if d["id"] == "floor-flatness")
    assert flat["one_sided"] is True and flat["capability"]["Cp"] is None
    assert flat["capability"]["Cpk"] == flat["capability"]["Cpu"]


def test_finding_probabilities_and_verdict(surrogate_study):
    results = surrogate_study["doc"]["results"]
    depth = next(d for d in results["dimensions"] if d["id"] == "pocket-depth")
    categories = {r["category"]: r for r in results["findings"]["by_category"]}
    assert categories["out_of_tolerance"]["probability"]["k"] == depth["out_of_tolerance"]["k"]
    assert results["verdict"]["unsafe"]["k"] == depth["out_of_tolerance"]["k"]
    locations = {(r["category"], r["location"]): r for r in results["findings"]["by_location"]}
    assert ("out_of_tolerance", "OP10 op10.nc:17") in locations and ("chatter", "OP10 op10.nc:13") in locations
    chatter = categories["chatter"]["probability"]
    exact = (sps.norm.cdf(3.0) - sps.norm.cdf(1.0)) / (sps.norm.cdf(3.0) - sps.norm.cdf(-3.0))   # P(kc > 1.04)
    assert chatter["low"] <= exact <= chatter["high"]


def test_sensitivity_ranking(surrogate_study):
    by_output = {r["output"]: r for r in surrogate_study["doc"]["results"]["sensitivity"]}
    depth = by_output["dimension pocket-depth"]["ranking"]
    assert depth[0]["factor"] == "stock_size[2]" and depth[0]["rho"] > 0.85 and depth[0]["significant"]
    order = [r["factor"] for r in depth]
    assert order.index("kc_scale") < order.index("length_offset[T1]")
    assert depth[0]["prcc"] > 0.9 and by_output["dimension pocket-depth"]["prcc_dof"] == 96 - 2 - 8
    flat = {r["factor"]: r for r in by_output["dimension floor-flatness"]["ranking"]}
    assert by_output["dimension floor-flatness"]["ranking"][0]["factor"] == "runout[T1]"
    assert flat["runout[T1]"]["rho"] > 0.9
    # the clamping effect is small next to runout: the rank correlation hardly sees it, the partial one does
    assert flat["clamping_force[OP10.0]"]["prcc"] < -0.5
    assert max(abs(r["prcc"]) for name, r in flat.items() if name not in ("runout[T1]", "clamping_force[OP10.0]")) < 0.4
    assert by_output["VB_max_m[T1]"]["ranking"][0]["factor"] in ("kc_scale", "hardness_scale")


def test_outputs_progress_caching_and_determinism(surrogate_study, base_material):
    doc, out = surrogate_study["doc"], surrogate_study["out"]
    folder = out / "s"
    stored = json.loads((folder / "study.json").read_text())
    assert stored["format"] == "cnctwin-uncertainty" and stored["results"] == json.loads(json.dumps(doc["results"]))
    page = (folder / "uncertainty.html").read_text(encoding="utf-8")
    check(page)
    assert "pocket-depth" in page and "Cpk" in page and "stock_size[2]" in page
    assert any(c.get("phase") == "aggregating" for c in surrogate_study["calls"])
    assert any(c.get("phase") == "variant 96/96" for c in surrogate_study["calls"])

    def refuse(*args, **kwargs):
        raise AssertionError("a cached variant was run again")

    again = study(surrogate_study["project"], folder, runner=refuse,
                  options={"simulate_options": {"base": base_material}})
    assert again["results"] == doc["results"]
    fresh = study(surrogate_study["project"], out / "fresh",
                  options={"runner": SURROGATE, "simulate_options": {"base": base_material}})
    assert fresh["results"] == doc["results"] and fresh["samples"] == doc["samples"]


def test_failed_samples_are_reported(tmp_path, base_material):
    project = write_project(tmp_path / "p", runs=12)
    doc = study(project, tmp_path / "s", options={"runner": SURROGATE, "simulate_options":
                                                  {"base": base_material, "fail_samples": [1, 3]}})
    results = doc["results"]
    assert results["samples_ok"] == 10 and results["samples_failed"] == 2
    assert [f["index"] for f in results["failed"]] == [1, 3]
    assert all("surrogate failure on request" in f["error"] for f in results["failed"])
    assert "Failed samples" in (tmp_path / "s" / "uncertainty.html").read_text(encoding="utf-8")


def test_monte_carlo_and_vary_seed(tmp_path, base_material):
    project = write_project(tmp_path / "p", runs=8, method="monte_carlo")
    plan = study(project, tmp_path / "s", options={"plan_only": True, "vary_seed": True})
    seeds = [s["seed"] for s in plan["samples"]]
    assert plan["settings"]["method"] == "monte_carlo" and len(set(seeds)) == 8
    assert not (tmp_path / "s" / "study.json").exists()


def test_runner_resolution():
    with pytest.raises(RunnerMissing):
        resolve_runner("cnctwin.module_that_does_not_exist:run_project")
    with pytest.raises(RunnerMissing):
        resolve_runner("/nonexistent/file.py:run")
    with pytest.raises(RunnerMissing):
        resolve_runner("no-colon")
    assert callable(resolve_runner(SURROGATE))


def test_api_entry_attaches_to_the_run_report(tmp_path, base_material):
    import make_synthetic_run as synthetic
    run_dir = tmp_path / "runs" / "00001-study"
    synthetic.make_synthetic_run(run_dir, seed=4, media=False)
    project = write_project(tmp_path / "p", runs=12)
    request = {"run_id": "00001-study", "run_dir": str(run_dir), "project_id": "p", "project_path": str(project),
               "project_dir": str(project.parent),
               "options": {"runner": SURROGATE, "simulate_options": {"base": base_material}}}
    result = run_study(request, tmp_path / "job", lambda *a, **k: None)
    assert result["samples_ok"] == 12 and Path(result["study"]).is_file()
    attached = json.loads((run_dir / "uncertainty" / "study.json").read_text())
    depth = next(d for d in attached["results"]["dimensions"] if d["id"] == "pocket-depth")
    assert depth["nominal_run_m"] == pytest.approx(0.002062)           # the run the study is made around
    from cnctwin.report.report import report_html
    text, _ = report_html(run_dir)
    check(text)
    section = text[text.index('id="uncertainty"'):text.index('id="identity"')]
    assert "pocket-depth" in section and "Cpk" in section and "not safe to run" in section
    with pytest.raises(Exception):
        run_study({"options": {}}, tmp_path / "job2", None)


def test_cli(tmp_path, capsys):
    from cnctwin.uncertainty import main
    project = write_project(tmp_path / "p", runs=6)
    assert main([str(project), "--out", str(tmp_path / "plan"), "--plan-only"]) == 0
    assert "planned 6 samples of 9 factors" in capsys.readouterr().out
    assert main([str(project), "--out", str(tmp_path / "s"), "--runner", SURROGATE, "--runs", "5", "--seed", "3"]) == 0
    assert "5 samples ok, 0 failed" in capsys.readouterr().out
    stored = json.loads((tmp_path / "s" / "study.json").read_text())
    assert stored["settings"]["seed"] == 3 and len(stored["samples"]) == 5
    values = np.array([s["values"]["stock_size[0]"] for s in stored["samples"]])
    assert sorted(np.floor((values - 0.0399) / 2e-4 * 5).astype(int).tolist()) == [0, 1, 2, 3, 4]   # LHS strata
    assert main([str(project), "--out", str(tmp_path / "x"), "--runner", "cnctwin.nothing:run"]) == 1
