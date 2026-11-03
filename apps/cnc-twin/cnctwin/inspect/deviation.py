"""Deviation map of the result surface against the CAD (docs/models/inspect.md "Deviation map").

Every result vertex has the signed distance to the CAD surface (BVH closest point; + where there is material
outside the CAD: excess, - where CAD material is missing) and the B-rep face of its closest point. Statistics are
area-weighted (each vertex carries a third of its triangles' area): per face and for the whole surface the mean,
RMS, standard deviation, minimum, maximum, the 1 % / 99 % quantiles, the largest |d| with its location, and the
areas in excess / missing beyond a tolerance. ``usable`` restricts a face's statistics to the points the face
assignment accepts (cnctwin.inspect.sampling), so a neighbouring wall's points near an edge do not appear as a
huge deviation of the floor; the statistics over every point of the face are given too.
"""

from __future__ import annotations

import numpy as np

from .cadref import CadReference
from .sampling import Assignment


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    total = cum[-1]
    if total <= 0.0:
        return float(np.quantile(values, q))
    idx = int(np.searchsorted(cum, q * total, side="left"))
    return float(v[min(idx, len(v) - 1)])


def statistics(d: np.ndarray, w: np.ndarray, points: np.ndarray, tolerance_m: float) -> dict:
    """Area-weighted statistics of deviations (empty -> counts only)."""
    finite = np.isfinite(d)
    d, w, points = d[finite], w[finite], points[finite]
    out = {"points": int(len(d)), "area_m2": float(w.sum())}
    if not len(d):
        return out
    wsum = float(w.sum())
    weights = w if wsum > 0.0 else np.ones_like(w)
    total = float(weights.sum())
    mean = float(np.sum(weights * d) / total)
    rms = float(np.sqrt(np.sum(weights * d * d) / total))
    std = float(np.sqrt(max(np.sum(weights * (d - mean) ** 2) / total, 0.0)))
    k = int(np.argmax(np.abs(d)))
    out.update({"mean_m": mean, "rms_m": rms, "std_m": std, "min_m": float(d.min()), "max_m": float(d.max()),
                "p01_m": _weighted_quantile(d, weights, 0.01), "p99_m": _weighted_quantile(d, weights, 0.99),
                "max_abs_m": float(abs(d[k])), "max_abs_point_S_m": [float(v) for v in points[k]],
                "excess_area_m2": float(w[d > tolerance_m].sum()), "missing_area_m2": float(w[d < -tolerance_m].sum())})
    return out


def deviation_map(assignment: Assignment, cad: CadReference, tolerance_m: float) -> dict:
    """Per-face and whole-surface deviation statistics of an assignment (see the module documentation)."""
    a = assignment
    faces = []
    for f in cad.faces:
        mask = a.face == f.id
        if not mask.any():
            faces.append({"face": f.id, "surface": f.surface, "name": f.name, "feature": f.feature, "points": 0})
            continue
        usable = mask & a.usable
        entry = {"face": f.id, "surface": f.surface, "name": f.name, "feature": f.feature,
                 "usable": statistics(a.deviation_m[usable], a.weights[usable], a.points[usable], tolerance_m),
                 "all": statistics(a.deviation_m[mask], a.weights[mask], a.points[mask], tolerance_m)}
        entry["points"] = entry["all"]["points"]
        machined = a.machined_fraction([f.id])
        if machined is not None:
            entry["machined_fraction"] = machined
        faces.append(entry)
    return {"surface": statistics(a.deviation_m, a.weights, a.points, tolerance_m),
            "usable": statistics(a.deviation_m[a.usable], a.weights[a.usable], a.points[a.usable], tolerance_m),
            "tolerance_m": float(tolerance_m), "sign": "+ material outside the CAD (excess), - CAD material missing",
            "faces": faces}


def vertex_deviation(cad: CadReference, vertices) -> np.ndarray:
    """Signed deviation of arbitrary points (brick mesh vertices) from the CAD."""
    d, _, _, _ = cad.closest(np.asarray(vertices, dtype=np.float64).reshape(-1, 3))
    return d


def brick_attributes(cad: CadReference, bricks: dict, displace=None) -> dict:
    """``{brick key: {"deviation_m": [nv] float32}}`` for keyframe brick meshes ``key -> (vertices, faces)``
    in the material configuration; ``displace(X) -> d`` moves them to the inspected geometry first."""
    out = {}
    for key in sorted(bricks):
        vertices = np.asarray(bricks[key][0], dtype=np.float64).reshape(-1, 3)
        P = vertices + displace(vertices) if displace is not None else vertices
        out[key] = {"deviation_m": vertex_deviation(cad, P).astype(np.float32)}
    return out


def write_brick_attributes(run_dir, attributes: dict) -> list[str]:
    """Merge ``deviation_m`` into ``keyframes/attr/brick_<key>.npz`` (other writers' arrays kept)."""
    from pathlib import Path

    from ..core.rundir import write_npz

    folder = Path(run_dir) / "keyframes" / "attr"
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for key in sorted(attributes):
        path = folder / f"brick_{key}.npz"
        merged = {}
        if path.is_file():
            with np.load(path, allow_pickle=False) as old:
                merged = {name: np.asarray(old[name]) for name in old.files}
        merged.update(attributes[key])
        write_npz(path, merged)
        written.append(f"keyframes/attr/brick_{key}.npz")
    return written


__all__ = ["deviation_map", "statistics", "vertex_deviation", "brick_attributes", "write_brick_attributes"]
