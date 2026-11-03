"""Aggregation of a study's samples: distributions of every inspected dimension with capability,
the probability of every finding (by category and by program location), the verdict, the scalar
results, and the sensitivity ranking of the factors (rank correlation) for each of them."""

from __future__ import annotations

import numpy as np

from . import stats

SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "fatal": 3}


def _limits(samples, ident):
    for sample in samples:
        record = sample.dimensions.get(ident)
        if record is not None:
            return record
    return {}


def dimensions(samples, params: dict, nominal=None) -> list[dict]:
    levels = list(params["percentile_levels"])
    confidence = float(params["confidence_level"])
    out = []
    for ident in sorted({i for s in samples for i in s.dimensions}):
        ref = _limits(samples, ident)
        values = [s.dimensions[ident]["measured_m"] for s in samples if ident in s.dimensions]
        present = [s for s in samples if ident in s.dimensions]
        judged = [s.dimensions[ident]["pass"] for s in present if s.dimensions[ident]["pass"] is not None]
        if judged:
            failed = sum(1 for p in judged if p is False)
            observed = stats.wilson(failed, len(judged), confidence)
        else:
            lower, upper = ref.get("lower_m"), ref.get("upper_m")
            outside = [v for v in stats.finite(values) if (lower is not None and v < lower) or (upper is not None and v > upper)]
            observed = stats.wilson(len(outside), len(stats.finite(values)), confidence)
        record = {"id": ident, "kind": ref.get("kind"), "nominal_m": ref.get("nominal_m"), "lower_m": ref.get("lower_m"),
                  "upper_m": ref.get("upper_m"), "one_sided": bool(ref.get("one_sided")),
                  "measured": stats.describe(values, levels, confidence),
                  "deviation": stats.describe([s.dimensions[ident]["deviation_m"] for s in present], levels, confidence),
                  "capability": stats.capability(values, ref.get("lower_m"), ref.get("upper_m"),
                                                 one_sided=bool(ref.get("one_sided")), confidence=confidence,
                                                 target=float(params["capability_target"]),
                                                 min_samples=int(params["min_samples_capability"]), levels=levels),
                  "out_of_tolerance": observed, "samples": [s.dimensions[ident]["measured_m"] for s in present],
                  "nominal_run_m": (nominal.dimensions.get(ident) or {}).get("measured_m") if nominal else None}
        out.append(record)
    return out


def _location(finding: dict) -> str:
    if finding["line"] > 0:
        return f"{finding['op']} {finding['program']}:{finding['line']}".strip()
    return f"{finding['op']} (no line)".strip()


def findings(samples, params: dict) -> dict:
    confidence = float(params["confidence_level"])
    n = len(samples)
    categories: dict[str, dict] = {}
    locations: dict[tuple, dict] = {}
    for sample in samples:
        seen_categories, seen_locations = set(), set()
        for finding in sample.findings:
            category = finding["category"]
            key = (category, _location(finding))
            for table, name, seen in ((categories, category, seen_categories), (locations, key, seen_locations)):
                record = table.setdefault(name, {"samples": 0, "count": 0, "worst": "info"})
                record["count"] += 1
                if SEVERITY_RANK.get(finding["severity"], 0) > SEVERITY_RANK.get(record["worst"], 0):
                    record["worst"] = finding["severity"]
                if name not in seen:
                    seen.add(name)
                    record["samples"] += 1
    by_category = [{"category": c, "probability": stats.wilson(r["samples"], n, confidence),
                    "mean_count": r["count"] / n if n else None, "worst": r["worst"]}
                   for c, r in sorted(categories.items())]
    by_location = [{"category": c, "location": loc, "probability": stats.wilson(r["samples"], n, confidence),
                    "mean_count": r["count"] / n if n else None, "worst": r["worst"]}
                   for (c, loc), r in sorted(locations.items())]
    return {"by_category": by_category, "by_location": by_location}


def scalars(samples, params: dict) -> list[dict]:
    levels = list(params["percentile_levels"])
    confidence = float(params["confidence_level"])
    names = sorted({k for s in samples for k in s.scalars})
    return [{"name": name, **stats.describe([s.scalars.get(name) for s in samples], levels, confidence)}
            for name in names]


def outputs(samples, dims: list[dict], scal: list[dict], finding_stats: dict) -> list[tuple[str, str, np.ndarray]]:
    """(name, kind, values per sample) of every output the sensitivity analysis ranks factors for."""
    out = []
    for record in dims:
        out.append((f"dimension {record['id']}", "dimension",
                    np.array([s.dimensions.get(record["id"], {}).get("measured_m", np.nan) for s in samples], float)))
    for record in scal:
        out.append((record["name"], "scalar", np.array([s.scalars.get(record["name"], np.nan) for s in samples], float)))
    for record in finding_stats["by_category"]:
        p = record["probability"]["p"]
        if p is not None and 0.0 < p < 1.0:
            out.append((f"finding {record['category']}", "probability",
                        np.array([float(any(f["category"] == record["category"] for f in s.findings)) for s in samples])))
    unsafe = np.array([0.0 if s.safe else 1.0 for s in samples])
    if 0.0 < unsafe.mean() < 1.0:
        out.append(("not safe to run", "probability", unsafe))
    return out


def sensitivity(factor_names: list[str], X: np.ndarray, outs, params: dict) -> list[dict]:
    result = []
    min_dof = int(params["prcc_min_dof"])
    confidence = float(params["confidence_level"])
    for name, kind, values in outs:
        partial = stats.prcc(X, values, min_dof) if len(factor_names) else []
        rows = []
        for j, factor in enumerate(factor_names):
            rank = stats.spearman(X[:, j], values)
            rows.append({"factor": factor, "rho": rank["rho"], "p": rank["p"], "n": rank["n"],
                         "significant": rank["p"] is not None and rank["p"] < 1.0 - confidence,
                         "prcc": partial[j]["prcc"] if partial else None, "prcc_p": partial[j]["p"] if partial else None})
        rows.sort(key=lambda r: (-(abs(r["rho"]) if r["rho"] is not None else -1.0), r["factor"]))
        result.append({"output": name, "kind": kind, "ranking": rows,
                       "prcc_dof": partial[0]["dof"] if partial else None})
    return result


def aggregate(plan: dict, samples: list, params: dict, nominal=None) -> dict:
    """The study's results (JSON-able) from the plan and the samples' results."""
    ok = [s for s in samples if s.ok]
    failed = [{"index": s.index, "error": s.error, "run_dir": s.run_dir} for s in samples if not s.ok]
    names = [f["name"] for f in plan["factors"]]
    rows = {int(r["index"]): r for r in plan["samples"]}
    X = np.array([[float(rows[s.index]["values"][n]) for n in names] for s in ok], float).reshape(len(ok), len(names))
    dims = dimensions(ok, params, nominal)
    scal = scalars(ok, params)
    finding_stats = findings(ok, params)
    confidence = float(params["confidence_level"])
    unsafe = sum(1 for s in ok if not s.safe)
    return {"samples_ok": len(ok), "samples_failed": len(failed), "failed": failed,
            "verdict": {"unsafe": stats.wilson(unsafe, len(ok), confidence),
                        "safe_samples": len(ok) - unsafe},
            "dimensions": dims, "scalars": scal, "findings": finding_stats,
            "sensitivity": sensitivity(names, X, outputs(ok, dims, scal, finding_stats), params),
            "factor_samples": {"names": names, "values": X.tolist(), "indexes": [s.index for s in ok]}}


__all__ = ["aggregate", "dimensions", "findings", "scalars", "sensitivity", "outputs"]
