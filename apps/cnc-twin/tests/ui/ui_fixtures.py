"""Data of the UI tests: a DATA_DIR with synthetic runs and example projects, an uncertainty study
document and a calibration overlay, all deterministic from a seed.

The runs come from tests/fixtures/make_synthetic_run.py (architecture 16). The study follows
docs/models/uncertainty.md section 10 (``cnctwin-uncertainty``) and the calibration outputs
docs/models/calibration.md "Running a calibration" (``overlay.json`` format
``cnctwin-calibration-overlay``, ``calibration.json`` format ``cnctwin-calibration-result``).
Their numbers are synthetic, with the right structure and internally consistent (the capability
indices, Wilson intervals and rank correlations are computed from the samples written), not model
output.

    python tests/ui/ui_fixtures.py DATA_DIR      # writes DATA_DIR/{runs,examples}
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
for entry in (str(ROOT), str(FIXTURES)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import make_synthetic_run as synthetic  # noqa: E402

DONE_RUN = "00001-pocket-unsafe"      # seed 2: finished, an error finding (not safe), media channels, study attached
SAFE_RUN = "00002-pocket-safe"        # seed 3: finished, safe
PARTIAL_RUN = "00003-pocket-partial"  # seed 5: stopped at 1.2 s, status running, no run.json
EXAMPLES = {"pocket": 4, "pocket-safe": 7}
CONFIDENCE_Z = 1.959963984540054      # Phi^-1(0.975), the 95 % level of db/models/uncertainty.yaml
CAPABILITY_TARGET = 1.33
PERCENTILES = (0.00135, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99865)


def _rng(seed: int, name: str) -> np.random.Generator:
    key = sum(ord(c) * (i + 1) for i, c in enumerate(name))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed & 0xFFFFFFFF, key])))


def _wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"p": None, "low": None, "high": None, "k": 0, "n": 0}
    p = k / n
    z = CONFIDENCE_Z
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return {"high": high, "k": k, "low": low, "n": n, "p": p}


def _distribution(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    from scipy import stats
    t = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    quantiles = {str(q): float(np.quantile(values, q)) for q in PERCENTILES}
    return {"max": float(values.max()), "mean": mean, "mean_ci": [mean - t * std / math.sqrt(n), mean + t * std / math.sqrt(n)],
            "min": float(values.min()), "n": n, "quantiles": quantiles, "std": std}


def _capability(values: np.ndarray, lower: float | None, upper: float) -> dict:
    n = len(values)
    mean = float(values.mean())
    s = float(values.std(ddof=1))
    cpu = (upper - mean) / (3 * s)
    cpl = (mean - lower) / (3 * s) if lower is not None else None
    cpk = min(cpu, cpl) if cpl is not None else cpu
    half = CONFIDENCE_Z * math.sqrt(1 / (9 * n) + cpk * cpk / (2 * (n - 1)))
    cp = (upper - lower) / (6 * s) if lower is not None else None
    from scipy import stats
    out = 1 - stats.norm.cdf((upper - mean) / s) + (stats.norm.cdf((lower - mean) / s) if lower is not None else 0.0)
    return {"Cp": cp, "Cp_ci": [cp * math.sqrt(stats.chi2.ppf(0.025, n - 1) / (n - 1)), cp * math.sqrt(stats.chi2.ppf(0.975, n - 1) / (n - 1))] if cp else None,
            "Cp_q": None, "Cpk": cpk, "Cpk_ci": [cpk - half, cpk + half], "Cpk_q": None, "Cpl": cpl, "Cpu": cpu,
            "capable": cpk >= CAPABILITY_TARGET, "expected_out_frac": float(out), "lower": lower, "n": n,
            "one_sided": lower is None, "reported": n >= 10, "target": CAPABILITY_TARGET, "upper": upper}


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0, 1.0
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    from scipy import stats
    if abs(rho) >= 1:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return rho, float(2 * stats.t.sf(abs(t), n - 2))


FACTORS = (
    ("stock_size[2]", "stock height", "m", 0.015, {"kind": "uniform", "low": 0.0149, "high": 0.0151}, "stock.tolerance_mm[2] = 0.1"),
    ("length_offset[T1]", "length offset error of T1", "m", 0.0, {"kind": "uniform", "low": -5e-6, "high": 5e-6}, "uncertainty.vary.tool_offsets_um = 5"),
    ("runout[T1]", "runout of T1", "m", 5e-6, {"kind": "uniform", "low": 2e-6, "high": 8e-6}, "uncertainty.vary.runout_um = 3"),
    ("kc_scale", "specific cutting force scale", "-", 1.0, {"kind": "truncated_normal", "mean": 1.0, "std": 0.04, "low": 0.88, "high": 1.12}, "material.batch.kc_scale_sigma = 0.04"),
)


def synthetic_study(seed: int = 2, runs: int = 16, base_run: str | None = None, nominal_depth_m: float = 0.002062) -> dict:
    """A cnctwin-uncertainty document around the synthetic pocket (pocket depth and floor flatness)."""
    rng = _rng(seed, "ui-study")
    n = runs
    values = np.empty((n, len(FACTORS)))
    for j, (_, _, _, _, dist, _) in enumerate(FACTORS):
        strata = (rng.permutation(n) + rng.uniform(size=n)) / n
        if dist["kind"] == "uniform":
            values[:, j] = dist["low"] + strata * (dist["high"] - dist["low"])
        else:
            from scipy import stats
            a, b = (dist["low"] - dist["mean"]) / dist["std"], (dist["high"] - dist["mean"]) / dist["std"]
            values[:, j] = stats.truncnorm.ppf(strata, a, b, loc=dist["mean"], scale=dist["std"])
    depth = nominal_depth_m + (values[:, 0] - 0.015) - values[:, 1] + 4e-4 * (values[:, 3] - 1.0)
    flatness = 4e-6 + 0.5 * values[:, 2] + 1e-6 * rng.normal(size=n)
    cycle = np.full(n, 2.32)
    vb = 2.7e-5 * values[:, 3]
    energy = 1.777e5 * (0.97 + 0.03 * values[:, 3])
    depth_pass = np.abs(depth - 0.002) <= 5e-5
    flat_pass = flatness <= 2e-5
    unsafe = ~depth_pass
    names = [f[0] for f in FACTORS]
    samples, sample_results = [], []
    for i in range(n):
        samples.append({"index": i, "options_factors": {}, "project": f"variants/v{i:04d}/project.yaml",
                        "project_sha256": f"{(seed * 7919 + i * 104729) % (1 << 64):016x}" * 4, "seed": seed,
                        "units": {name: float((values[i, j] - values[:, j].min()) / (np.ptp(values[:, j]) or 1)) for j, name in enumerate(names)},
                        "values": {name: float(values[i, j]) for j, name in enumerate(names)}, "written": []})
        findings = []
        if unsafe[i]:
            findings.append({"category": "out_of_tolerance", "severity": "error", "op": "OP10", "program": "op10.nc", "line": 17})
        if values[i, 3] > 1.04:
            findings.append({"category": "chatter", "severity": "warning", "op": "OP10", "program": "op10.nc", "line": 13})
        sample_results.append({
            "dimensions": {
                "pocket-depth": {"deviation_m": float(depth[i] - 0.002), "kind": "distance", "lower_m": 0.00195, "measured_m": float(depth[i]),
                                 "nominal_m": 0.002, "one_sided": False, "pass": bool(depth_pass[i]), "upper_m": 0.00205},
                "floor-flatness": {"deviation_m": None, "kind": "flatness", "lower_m": None, "measured_m": float(flatness[i]),
                                   "nominal_m": None, "one_sided": True, "pass": bool(flat_pass[i]), "upper_m": 2e-5}},
            "error": None, "findings": findings, "index": i, "ok": True, "run_dir": f"variants/v{i:04d}/run", "safe": bool(not unsafe[i]),
            "scalars": {"VB_max_m[T1]": float(vb[i]), "cycle_time_s": float(cycle[i]), "energy_per_part_J": float(energy[i])}})
    dims = []
    for ident, kind, measured, lower, upper, nominal, passes in (
            ("pocket-depth", "distance", depth, 0.00195, 0.00205, 0.002, depth_pass),
            ("floor-flatness", "flatness", flatness, None, 2e-5, None, flat_pass)):
        dims.append({"capability": _capability(measured, lower, upper),
                     "deviation": _distribution(measured - nominal) if nominal is not None else {"max": None, "mean": None, "mean_ci": None, "min": None, "n": 0, "quantiles": {}, "std": None},
                     "id": ident, "kind": kind, "lower_m": lower, "measured": _distribution(measured), "nominal_m": nominal,
                     "nominal_run_m": float(nominal_depth_m) if ident == "pocket-depth" else 8e-6, "one_sided": lower is None,
                     "out_of_tolerance": _wilson(int((~passes).sum()), n), "samples": [float(v) for v in measured], "upper_m": upper})
    scalars = [{**_distribution(arr), "name": name} for name, arr in (("VB_max_m[T1]", vb), ("cycle_time_s", cycle), ("energy_per_part_J", energy))]
    outputs = [("dimension pocket-depth", "dimension", depth), ("dimension floor-flatness", "dimension", flatness),
               ("VB_max_m[T1]", "scalar", vb), ("energy_per_part_J", "scalar", energy), ("verdict", "verdict", unsafe.astype(float))]
    sensitivity = []
    for output, kind, y in outputs:
        if np.ptp(y) == 0:
            continue
        ranking = []
        for j, name in enumerate(names):
            rho, p = _spearman(values[:, j], y)
            ranking.append({"factor": name, "n": n, "p": p, "prcc": None, "prcc_p": None, "rho": rho, "significant": p < 0.05})
        ranking.sort(key=lambda r: -abs(r["rho"]))
        sensitivity.append({"kind": kind, "output": output, "prcc_dof": n - 2 - (len(names) - 1), "ranking": ranking})
    chatter = np.array([values[i, 3] > 1.04 for i in range(n)])
    by_category = [{"category": "chatter", "mean_count": float(chatter.mean()), "probability": _wilson(int(chatter.sum()), n), "worst": "warning"},
                   {"category": "out_of_tolerance", "mean_count": float(unsafe.mean()), "probability": _wilson(int(unsafe.sum()), n), "worst": "error"}]
    by_location = [{**row, "location": f"OP10 op10.nc:{13 if row['category'] == 'chatter' else 17}"} for row in by_category]
    factors = []
    for j, (name, label, unit, nominal, dist, stated) in enumerate(FACTORS):
        d = dict(dist)
        if d["kind"] == "uniform":
            d.update({"mean": (d["low"] + d["high"]) / 2, "std": (d["high"] - d["low"]) / math.sqrt(12)})
        factors.append({"applies": "project", "distribution": d, "index": {"factor": j}, "label": label, "name": name, "nominal": nominal,
                        "stated_by": stated, "target": name, "unit": unit})
    return {
        "base_run": base_run, "factors": factors, "format": "cnctwin-uncertainty", "nominal": None,
        "not_varied": [{"key": "clamping_force", "reason": "no clamping torque or pressure in the project"}],
        "options_only": [], "parameters": {"capability_target": CAPABILITY_TARGET, "confidence_level": 0.95, "min_samples_capability": 10,
                                           "normal_truncation_sigma": 3, "percentile_levels": list(PERCENTILES), "prcc_min_dof": 3},
        "project": {"name": "Synthetic pocket, Al 6061-T6", "path": "project.yaml", "sha256": "0" * 64},
        "results": {"dimensions": dims, "factor_samples": {"indexes": list(range(n)), "names": names, "values": values.tolist()},
                    "failed": [], "findings": {"by_category": by_category, "by_location": by_location}, "samples_failed": 0,
                    "samples_ok": n, "scalars": scalars, "sensitivity": sensitivity,
                    "verdict": {"safe_samples": int((~unsafe).sum()), "unsafe": _wilson(int(unsafe.sum()), n)}},
        "sample_results": sample_results, "samples": samples,
        "settings": {"keep_runs": True, "method": "lhs", "runs": n, "seed": seed, "vary_seed": False},
        "vary": {"runout_um": 3, "stock_size": True, "tool_offsets_um": 5}, "vary_origin": {"runout_um": "project", "stock_size": "project", "tool_offsets_um": "project"},
        "version": 1}


def attach_study(run_dir: Path, study: dict) -> None:
    folder = Path(run_dir) / "uncertainty"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "study.json").write_text(json.dumps(study, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    verdict = study["results"]["verdict"]["unsafe"]
    (folder / "uncertainty.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Uncertainty study</title></head><body>"
        f"<h1>Uncertainty study</h1><p>P(not safe to run) = {verdict['p']:.3f} over {verdict['n']} samples.</p></body></html>\n",
        encoding="utf-8")


def _metrics(measured: np.ndarray, predicted: np.ndarray) -> dict:
    ok = np.isfinite(measured) & np.isfinite(predicted)
    m, q = measured[ok], predicted[ok]
    e = q - m
    rmse = float(np.sqrt(np.mean(e ** 2)))
    span = float(np.ptp(m))
    ss = float(np.sum((m - m.mean()) ** 2))
    return {"bias": float(e.mean()), "mae": float(np.mean(np.abs(e))), "max_abs": float(np.max(np.abs(e))), "n": int(len(m)),
            "nrmse": rmse / span if span > 0 else None, "r": float(np.corrcoef(m, q)[0, 1]) if np.std(m) > 0 and np.std(q) > 0 else None,
            "r2": 1 - float(np.sum(e ** 2)) / ss if ss > 0 else None, "rmse": rmse}


def write_calibration(out_dir: Path, run_dir: Path, seed: int = 11) -> dict:
    """overlay.json, calibration.json and alignment.json of a spindle-power calibration against a run."""
    import pyarrow.parquet as pq
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted((Path(run_dir) / "series" / "tick").glob("part-*.parquet"), key=lambda p: int(p.stem.split("-")[1]))
    t, power = [], []
    for part in parts:
        table = pq.read_table(part, columns=["time_s", "cutting.spindle_power_W"])
        t.extend(table.column("time_s").to_pylist())
        power.extend(v if v is not None else float("nan") for v in table.column("cutting.spindle_power_W").to_pylist())
    t = np.asarray(t[::10], float)
    simulated = np.asarray(power[::10], float) + 900.0
    rng = _rng(seed, "ui-calibration")
    measured = 1.08 * (simulated - 900.0) + 612.0 + 25.0 * rng.normal(size=len(t))
    a = np.vstack([simulated - 900.0, np.ones(len(t))]).T
    coef, *_ = np.linalg.lstsq(a, measured, rcond=None)
    calibrated = a @ coef
    alignment = {"method": "markers", "offset_s": 0.0123, "quality": {"matched": 18, "residual_s": 0.0004}, "scale": 1.0}
    series = [{"target": "spindle", "name": "spindle.power_W", "unit": "W", "doc": "electrical spindle power",
               "x": "t_s", "t_s": t.tolist(), "measured": measured.tolist(), "simulated": simulated.tolist(), "calibrated": calibrated.tolist(),
               "reduced": "none", "metrics": {"simulated": _metrics(measured, simulated), "calibrated": _metrics(measured, calibrated)}}]
    (out_dir / "overlay.json").write_text(json.dumps({"format": "cnctwin-calibration-overlay", "version": 1, "time_base": "simulation",
                                                       "alignment": alignment, "series": series}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    params = [{"name": "kc_scale", "unit": "-", "prior": 1.0, "value": float(coef[0]), "ci": [float(coef[0]) - 0.01, float(coef[0]) + 0.01],
               "std_error": 0.005, "free": True, "identifiable": True, "at_bound": False, "written": True, "remark": ""},
              {"name": "spindle_idle_power_W", "unit": "W", "prior": 900.0, "value": float(coef[1]), "ci": [float(coef[1]) - 15, float(coef[1]) + 15],
               "std_error": 7.5, "free": True, "identifiable": True, "at_bound": False, "written": True, "remark": ""},
              {"name": "spindle_efficiency", "unit": "-", "prior": 0.85, "value": 0.85, "ci": None, "std_error": None, "free": False,
               "identifiable": None, "at_bound": False, "written": False, "remark": "fixed"}]
    report = {"format": "cnctwin-calibration-result", "version": 1, "name": "calibration of spindle, haas-vf2ss", "machine": "haas-vf2ss",
              "material": "al6061-t6", "inputs": {"run": {"run_id": Path(run_dir).name}}, "alignment": alignment,
              "targets": [{"target": "spindle", "status": "fitted", "simulator": "surrogate", "params": params, "confounded": [],
                           "metrics": series[0]["metrics"], "notes": ["kc_scale and the efficiency enter only as their ratio in power data"]}],
              "set": {"models.production-energy.spindle_idle_power_W": {"value": float(coef[1]), "parameter": "spindle_idle_power_W"},
                      "material.cutting.kienzle[*].kc11_N_mm2": {"value": float(coef[0]), "parameter": "kc_scale"}}}
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "alignment.json").write_text(json.dumps(alignment, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"format": "cnctwin-calibration-result", "version": 1, "name": report["name"], "out_dir": str(out_dir),
            "files": {"series": "overlay.json", "report": "calibration.json", "alignment": "alignment.json"},
            "alignment": alignment, "set": report["set"], "targets": [{k: v for k, v in report["targets"][0].items() if k != "notes"}]}


def build_data_dir(data_dir: Path) -> dict:
    """DATA_DIR/runs (three synthetic runs, the first with a study) and DATA_DIR/examples (two projects)."""
    data_dir = Path(data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    (data_dir / "runs").mkdir(parents=True)
    done = synthetic.make_synthetic_run(data_dir / "runs" / DONE_RUN, seed=2, media_channels=True)
    safe = synthetic.make_synthetic_run(data_dir / "runs" / SAFE_RUN, seed=3, media_channels=True)
    partial = synthetic.make_synthetic_run(data_dir / "runs" / PARTIAL_RUN, seed=5, until_s=1.2, media=False)
    attach_study(data_dir / "runs" / DONE_RUN, synthetic_study(seed=2, base_run=DONE_RUN))
    for name, seed in EXAMPLES.items():
        synthetic.write_synthetic_project(data_dir / "examples" / name, seed=seed,
                                          name=f"Synthetic pocket {name}, Al 6061-T6")
    return {"runs": [DONE_RUN, SAFE_RUN, PARTIAL_RUN], "examples": [f"example-{n}" for n in EXAMPLES],
            "verdicts": {DONE_RUN: done["verdict"], SAFE_RUN: safe["verdict"]}, "partial_finished": partial["finished"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir")
    args = parser.parse_args(argv)
    print(json.dumps(build_data_dir(Path(args.data_dir)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
