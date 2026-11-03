"""CAD face selectors: ``{face, normal | axis, near_mm, radius_mm, id}`` -> B-rep face ids
(docs/models/geom.md "Face selectors"; project ``cad.datums[].feature`` and ``cad.tolerances[].feature``).

Matching, in this order:

1. ``id``: ``"face:<n>"`` (or ``"<n>"``) names B-rep face n directly; any other text matches the faces of
   that name in the STEP file.
2. ``face``: the surface type (``plane cylinder cone sphere torus``; ``freeform`` for every other surface;
   ``any``).
3. ``normal``: the outward normal of a plane, or of a curved face at the point nearest to ``near_mm``,
   within ``geom.selector_angle_tolerance_rad``; ``axis``: the axis of a cylinder, cone or torus (either
   sign), or a plane's normal up to sign. A direction the face type has no notion of does not match.
4. ``radius_mm``: a cylinder's or sphere's radius, a torus's minor radius, a cone's radius at the point
   nearest to ``near_mm``, within ``geom.selector_radius_tolerance_m``.
5. Best match: the smallest distance from ``near_mm`` to the face (exact point-triangle distance on the
   tessellation, within its chord tolerance of the surface), ties by angle error, then face id; without
   ``near_mm`` the smallest angle error, then the largest area.

The result is every face of the best face's feature (adjacent faces on the same analytic surface, such as
the two halves of a cylinder split at its seam).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from . import transform as tf
from .cad import ANALYTIC, CadModel
from .settings import resolve


@dataclass
class Selection:
    faces: list = field(default_factory=list)      # face ids of the matched feature, sorted
    best: int = -1
    distance_m: float = math.nan                   # near point to the best face (NaN without near_mm)
    angle_rad: float = 0.0                         # direction error of the best face
    candidates: int = 0                            # faces that passed the type/direction/radius filters
    reason: str = ""                               # why nothing matched (empty on success)

    @property
    def ok(self) -> bool:
        return self.best >= 0


def _get(sel, key):
    if isinstance(sel, Mapping):
        return sel.get(key)
    return getattr(sel, key, None)


def _vec_or_none(value):
    if value is None:
        return None
    value = getattr(value, "value", value)
    return np.asarray([float(getattr(v, "value", v)) for v in value], dtype=np.float64)


def _length(sel, key: str):
    """SI value of ``<key>_m`` (SI view) or ``<key>_mm`` (file form)."""
    v = _get(sel, key + "_m")
    if v is not None:
        return _vec_or_none(v) if isinstance(getattr(v, "value", v), (list, tuple, np.ndarray)) else float(v)
    v = _get(sel, key + "_mm")
    if v is None:
        return None
    raw = getattr(v, "value", v)
    if isinstance(raw, (list, tuple, np.ndarray)):
        return _vec_or_none(raw) * 1e-3
    return float(raw) * 1e-3


def closest_points_on_triangles(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Closest point of each triangle ``[m, 3, 3]`` to the point p (vectorised Voronoi-region test,
    Ericson 2005 5.1.5)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    out = np.empty_like(a)
    done = np.zeros(len(a), bool)

    def put(mask, value):
        m = mask & ~done
        out[m] = value[m] if value.ndim == 2 else value
        done[m] = True

    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_ab = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + s_ab[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        s_ac = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + s_ac[:, None] * ac)
        den_bc = (d4 - d3) + (d5 - d6)
        s_bc = np.where(den_bc != 0, (d4 - d3) / den_bc, 0.0)
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + s_bc[:, None] * (c - b))
        den = va + vb + vc
        v = np.where(den != 0, vb / den, 0.0)
        w = np.where(den != 0, vc / den, 0.0)
    put(np.ones(len(a), bool), a + v[:, None] * ab + w[:, None] * ac)
    return out


def _face_distances(model: CadModel, near: np.ndarray):
    """Per face: (distance from near, closest point, normal of the closest triangle)."""
    tris = model.mesh.triangles()
    q = closest_points_on_triangles(near, tris)
    d = np.linalg.norm(q - near, axis=1)
    normals = model.mesh.face_normals()
    n_faces = len(model.faces)
    best_d = np.full(n_faces, np.inf)
    best_t = np.full(n_faces, -1, np.int64)
    g = model.mesh.face_group.astype(np.int64)
    order = np.lexsort((np.arange(len(d)), d, g))
    first = np.ones(len(order), bool)
    first[1:] = g[order][1:] != g[order][:-1]
    for t in order[first]:
        if 0 <= g[t] < n_faces:
            best_d[g[t]] = d[t]
            best_t[g[t]] = t
    points = np.where(best_t[:, None] >= 0, q[np.maximum(best_t, 0)], np.nan)
    nrm = np.where(best_t[:, None] >= 0, normals[np.maximum(best_t, 0)], np.nan)
    return best_d, points, nrm


def _angle(u: np.ndarray, v: np.ndarray, signed: bool) -> float:
    c = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))
    if not signed:
        c = abs(c)
    return math.acos(c)


def select_faces(model: CadModel, selector, pose_C_X=None, angle_tol_rad: float | None = None,
                 radius_tol_m: float | None = None) -> Selection:
    """Faces of ``model`` that ``selector`` names (see the module doc). ``pose_C_X`` maps a selector written
    in another frame X (the stock frame) into the CAD frame C."""
    ang_tol = float(resolve(angle_tol_rad, "selector_angle_tolerance_rad"))
    rad_tol = float(resolve(radius_tol_m, "selector_radius_tolerance_m"))
    T = None if pose_C_X is None else np.asarray(pose_C_X, dtype=np.float64)
    kind = str(_get(selector, "face") or "any")
    normal = _vec_or_none(_get(selector, "normal"))
    axis = _vec_or_none(_get(selector, "axis"))
    near = _length(selector, "near")
    radius = _length(selector, "radius")
    if T is not None:
        normal = tf.apply_dir(T, normal) if normal is not None else None
        axis = tf.apply_dir(T, axis) if axis is not None else None
        near = tf.apply(T, near) if near is not None else None
    ident = _get(selector, "id")
    faces = model.faces
    candidates = list(range(len(faces)))
    if ident is not None:
        text = str(ident).strip()
        num = text[5:] if text.startswith("face:") else text
        if num.isdigit():
            candidates = [int(num)] if int(num) < len(faces) else []
            if not candidates:
                return Selection(reason=f"no face {num} (the model has {len(faces)})")
        else:
            candidates = [f.id for f in faces if f.name == text]
            if not candidates:
                return Selection(reason=f"no face named {text!r}")
    if kind not in ("any",):
        if kind == "freeform":
            candidates = [i for i in candidates if faces[i].surface not in ANALYTIC]
        else:
            candidates = [i for i in candidates if faces[i].surface == kind]
    if not candidates:
        return Selection(reason=f"no {kind} face")
    dist = pts = nrm = None
    if near is not None:
        dist, pts, nrm = _face_distances(model, np.asarray(near, dtype=np.float64))
    kept: list[tuple[float, float, float, int]] = []
    for i in candidates:
        f = faces[i]
        err = 0.0
        if normal is not None:
            if f.surface == "plane":
                err = max(err, _angle(np.asarray(f.params["normal"]), normal, True))
            elif nrm is not None and np.all(np.isfinite(nrm[i])):
                err = max(err, _angle(nrm[i], normal, True))
            else:
                continue
        if axis is not None:
            if f.surface in ("cylinder", "cone", "torus"):
                err = max(err, _angle(np.asarray(f.params["axis"]), axis, False))
            elif f.surface == "plane":
                err = max(err, _angle(np.asarray(f.params["normal"]), axis, False))
            else:
                continue
        if err > ang_tol:
            continue
        if radius is not None:
            r_face = _face_radius(f, pts[i] if pts is not None else None)
            if r_face is None or abs(r_face - radius) > rad_tol:
                continue
        d = float(dist[i]) if dist is not None else math.nan
        kept.append((d if dist is not None else 0.0, err, -f.area_m2 if dist is None else 0.0, i))
    if not kept:
        return Selection(candidates=0, reason=f"no {kind} face matches the direction/radius given")
    kept.sort()
    d, err, _, best = kept[0]
    return Selection(sorted(model.feature_faces(best)), best, float(dist[best]) if dist is not None else math.nan,
                     float(err), len(kept), "")


def _face_radius(f, point) -> float | None:
    p = f.params
    if f.surface in ("cylinder", "sphere"):
        return float(p["radius_m"])
    if f.surface == "torus":
        return float(p["minor_radius_m"])
    if f.surface == "cone" and point is not None and np.all(np.isfinite(point)):
        a = np.asarray(p["apex_m"])
        u = np.asarray(p["axis"])
        v = np.asarray(point) - a
        return float(np.linalg.norm(v - np.dot(v, u) * u))
    return None


def _clean(values, digits: int) -> list[float]:
    return [float(round(float(v), digits)) + 0.0 for v in values]


def selector_for_face(model: CadModel, face_id: int) -> dict:
    """The project-form selector (mm) that names ``face_id``: ``{face, normal | axis, near_mm, radius_mm,
    id: "face:<n>"}``."""
    f = model.faces[int(face_id)]
    out: dict = {"face": f.selector_type}
    if f.surface == "plane":
        out["normal"] = _clean(f.params["normal"], 12)
    elif f.surface in ("cylinder", "cone", "torus"):
        out["axis"] = _clean(f.params["axis"], 12)
    out["near_mm"] = _clean(np.asarray(f.point_m) * 1e3, 9)
    if f.surface in ("cylinder", "sphere"):
        out["radius_mm"] = float(round(f.params["radius_m"] * 1e3, 9))
    elif f.surface == "torus":
        out["radius_mm"] = float(round(f.params["minor_radius_m"] * 1e3, 9))
    out["id"] = f"face:{f.id}"
    return out


__all__ = ["Selection", "select_faces", "selector_for_face", "closest_points_on_triangles"]
