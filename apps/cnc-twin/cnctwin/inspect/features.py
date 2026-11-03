"""Features of the CAD part and their measured size, location and orientation on the result
(docs/models/inspect.md "Features").

**Recognition** (B-rep faces, cnctwin.geom.cad features: adjacent faces of one analytic surface):

* single-surface features: plane, cylinder (hole when concave, boss otherwise), cone, sphere, torus, freeform;
* **width features** (ISO 14405 size between opposite parallel planes): two plane features whose outward
  normals are antiparallel within ``pair_angle_rad`` and whose outlines overlap in projection, each the other's
  nearest such face: a **slot** when the normals face each other (material outside), a **tab** otherwise;
* **pockets**: a plane (the floor) all of whose neighbours meet it at concave edges and rise perpendicular to it
  (the walls; a neighbour at a convex edge makes the pocket open on that side); its depth is measured to the
  faces adjacent to the walls that are parallel to the floor with the same outward sense (the top).

**Measurement**: least-squares association on the feature's usable result points (cnctwin.inspect.sampling), and
the deviations from the CAD: size (diameter, width, radii, half angle), location (distance of the measured
plane, axis or centre from the nominal at the feature's middle), orientation (angle between the measured and the
nominal normal or axis), form (minimum zone).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geom import transform as tf
from . import fitting as F
from .cadref import CadReference
from .schema import InspectSettings


@dataclass
class Feature:
    id: str
    kind: str                       # plane cylinder cone sphere torus freeform slot tab pocket
    faces: list
    internal: bool = False
    nominal: dict = field(default_factory=dict)
    members: list = field(default_factory=list)        # sub-feature ids (slot: the two planes; pocket: floor, walls)

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "faces": [int(f) for f in self.faces], "internal": self.internal,
                "nominal": _plain(self.nominal), "members": list(self.members)}


def _plain(d):
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    if isinstance(d, np.ndarray):
        return [float(v) for v in d.reshape(-1)]
    if isinstance(d, (list, tuple)):
        return [_plain(v) for v in d]
    if isinstance(d, (np.floating, float)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def _plane_outline(cad: CadReference, faces, n) -> tuple[np.ndarray, np.ndarray]:
    """2-D bounding box of faces in the plane frame of normal n."""
    V = np.concatenate([cad.face_vertices(f) for f in faces])
    R = tf.frame_from_z(n)
    uv = V @ R[:, :2]
    return uv.min(axis=0), uv.max(axis=0)


def recognise(cad: CadReference, settings: InspectSettings) -> list[Feature]:
    """Every feature of the CAD part (module documentation), in a deterministic order."""
    groups: dict[int, list] = {}
    for f in cad.faces:
        groups.setdefault(int(f.feature) if f.feature >= 0 else -1 - f.id, []).append(f.id)
    out: list[Feature] = []
    planes = []
    for key in sorted(groups, key=lambda k: min(groups[k])):
        faces = sorted(groups[key])
        f0 = cad.face(faces[0])
        kind = f0.surface if f0.analytic else "freeform"
        nominal = dict(f0.params)
        if kind == "cylinder":
            nominal["diameter_m"] = 2.0 * float(f0.params["radius_m"])
        if kind == "sphere":
            nominal["diameter_m"] = 2.0 * float(f0.params["radius_m"])
        if kind in ("cylinder", "cone"):
            u = tf.unit(f0.params["axis"])
            origin = np.asarray(f0.params["axis_point_m"] if kind == "cylinder" else f0.params["apex_m"])
            z = np.concatenate([(cad.face_vertices(f) - origin) @ u for f in faces])
            nominal["axial_range_m"] = [float(z.min()), float(z.max())]
        feat = Feature(f"F{len(out) + 1}", kind, faces, bool(f0.concave), nominal)
        out.append(feat)
        if kind == "plane":
            planes.append(feat)
    out += _width_features(cad, planes, settings, len(out))
    out += _pockets(cad, planes, settings, len(out))
    return out


def _width_features(cad, planes, settings, start) -> list[Feature]:
    """Mutually nearest pairs of antiparallel, overlapping planes: facing pairs (air between) are slots, back-to-back
    pairs (material between) are tabs; a face may be in one slot and one tab."""
    cos_lim = math.cos(settings.pair_angle_rad)
    best = {"slot": {}, "tab": {}}
    gaps = {}
    for a in planes:
        na = tf.unit(a.nominal["normal"])
        pa = np.asarray(a.nominal["point_m"])
        lo_a, hi_a = _plane_outline(cad, a.faces, na)
        for b in planes:
            if b.id == a.id or float(na @ tf.unit(b.nominal["normal"])) > -cos_lim:
                continue
            gap = float((np.asarray(b.nominal["point_m"]) - pa) @ na)
            if gap == 0.0:
                continue
            lo_b, hi_b = _plane_outline(cad, b.faces, na)
            if np.any(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b) <= 0.0):
                continue
            kind = "slot" if gap > 0.0 else "tab"
            if a.id not in best[kind] or abs(gap) < best[kind][a.id][0]:
                best[kind][a.id] = (abs(gap), b.id)
            gaps[(a.id, b.id)] = gap
    out = []
    by_id = {p.id: p for p in planes}
    for kind in ("slot", "tab"):
        seen = set()
        for a_id in sorted(best[kind], key=lambda s: int(s[1:])):
            b_id = best[kind][a_id][1]
            if best[kind].get(b_id, (None, None))[1] != a_id or b_id in seen:
                continue
            seen.add(a_id)
            a, b = by_id[a_id], by_id[b_id]
            out.append(Feature(f"F{start + len(out) + 1}", kind, sorted(a.faces + b.faces), kind == "slot",
                               {"width_m": abs(gaps[(a_id, b_id)]), "normal": tf.unit(a.nominal["normal"]),
                                "sides": [a.faces, b.faces]}, [a_id, b_id]))
    return out


def edge_is_concave(cad: CadReference, a: int, b: int) -> bool | None:
    """Whether the edge between faces a and b is concave (b rises on a's outward side at the edge, judged on the
    triangles that meet there, the majority of the edge's mesh segments); None if not adjacent."""
    share = cad.edge_concavity(a, b)
    return None if share is None else bool(share > 0.5)


def _pockets(cad, planes, settings, start) -> list[Feature]:
    sin_lim = math.sin(settings.pair_angle_rad)
    out = []
    plane_faces = {f: p for p in planes for f in p.faces}
    for floor in planes:
        n = tf.unit(floor.nominal["normal"])
        neighbours = sorted({b for f in floor.faces for b in cad.adjacency.get(f, []) if b not in floor.faces})
        if not neighbours:
            continue
        walls, open_side = [], False
        for b in neighbours:
            concave = any(edge_is_concave(cad, f, b) for f in floor.faces if b in cad.adjacency.get(f, []))
            fb = cad.face(b)
            if fb.surface == "plane":
                upright = abs(float(tf.unit(fb.params["normal"]) @ n)) < sin_lim
            elif fb.surface == "cylinder":
                upright = abs(abs(float(tf.unit(fb.params["axis"]) @ n)) - 1.0) < 1.0 - math.cos(settings.pair_angle_rad)
            else:
                upright = False
            if concave and upright:
                walls.append(b)
            else:
                open_side = True
        if len(walls) < 1 or (open_side and len(walls) < 2):
            continue
        tops = set()
        for w in walls:
            for c in cad.adjacency.get(w, []):
                if c in floor.faces or c in walls:
                    continue
                fc = cad.face(c)
                if fc.surface == "plane" and float(tf.unit(fc.params["normal"]) @ n) > math.cos(settings.pair_angle_rad):
                    tops.add(c)
        if not tops:
            continue                     # no face around the opening: a wall or a step, not a pocket floor
        p0 = np.asarray(floor.nominal["point_m"])
        depth = max(float((np.asarray(cad.face(c).params["point_m"]) - p0) @ n) for c in sorted(tops))
        out.append(Feature(f"F{start + len(out) + 1}", "pocket", sorted(floor.faces + walls), True,
                           {"floor": floor.faces, "walls": walls, "top": sorted(tops), "depth_m": depth,
                            "open": open_side, "normal": n},
                           [floor.id] + sorted({plane_faces[w].id for w in walls if w in plane_faces},
                                               key=lambda s: int(s[1:]))))
    return out


# ============================================================================ measurement

def _angle(u, v, signed: bool = True) -> float:
    c = float(np.clip(tf.unit(u) @ tf.unit(v), -1.0, 1.0))
    return math.acos(c if signed else abs(c))


def measure(feature: Feature, cad: CadReference, points_of, settings: InspectSettings) -> dict:
    """Measured association and deviations of a feature (``points_of(faces) -> PointSet``)."""
    out = {"id": feature.id, "kind": feature.kind}
    try:
        if feature.kind in ("slot", "tab"):
            return out | _measure_width(feature, cad, points_of, settings)
        if feature.kind == "pocket":
            return out | _measure_pocket(feature, cad, points_of, settings)
        ps = points_of(feature.faces)
        out["points"] = len(ps)
        if len(ps) == 0:
            out["status"] = "no usable result points"
            return out
        P = ps.points
        nom = feature.nominal
        d, _ = cad.face_distance(feature.faces[0], P)
        out["mean_deviation_m"] = float(np.mean(d))
        if feature.kind == "plane":
            ls = F.fit_plane(P, nom["normal"])
            mz = F.zone_plane(P, nom["normal"], settings)
            n0 = tf.unit(nom["normal"])
            out.update({"fit": ls.to_json(), "location_deviation_m": float((ls.params["point_m"] - nom["point_m"]) @ n0),
                        "orientation_deviation_rad": _angle(ls.params["normal"], n0), "flatness_m": mz.zone_m,
                        "flatness_ls_m": ls.zone_m})
        elif feature.kind == "cylinder":
            u0 = tf.unit(nom["axis"])
            ls = F.fit_cylinder(P, nom["axis_point_m"], u0, float(nom["radius_m"]), settings)
            a, u = ls.params["axis_point_m"], ls.params["axis"]
            w = a - np.asarray(nom["axis_point_m"])
            out.update({"fit": ls.to_json(), "diameter_m": 2.0 * ls.params["radius_m"],
                        "diameter_deviation_m": 2.0 * (ls.params["radius_m"] - float(nom["radius_m"])),
                        "location_deviation_m": float(np.linalg.norm(w - (w @ u0) * u0)),
                        "orientation_deviation_rad": _angle(u, u0, signed=False), "form_m": ls.zone_m})
        elif feature.kind == "sphere":
            ls = F.fit_sphere(P, nom["centre_m"], float(nom["radius_m"]), settings)
            out.update({"fit": ls.to_json(), "diameter_m": 2.0 * ls.params["radius_m"],
                        "diameter_deviation_m": 2.0 * (ls.params["radius_m"] - float(nom["radius_m"])),
                        "location_deviation_m": float(np.linalg.norm(ls.params["centre_m"] - nom["centre_m"])),
                        "form_m": ls.zone_m})
        elif feature.kind == "cone":
            ls = F.fit_cone(P, nom["apex_m"], nom["axis"], float(nom["semi_angle_rad"]), settings)
            out.update({"fit": ls.to_json(), "semi_angle_deviation_rad": float(ls.params["semi_angle_rad"] -
                                                                                 float(nom["semi_angle_rad"])),
                        "apex_deviation_m": float(np.linalg.norm(ls.params["apex_m"] - nom["apex_m"])),
                        "orientation_deviation_rad": _angle(ls.params["axis"], nom["axis"], signed=False),
                        "form_m": ls.zone_m})
        elif feature.kind == "torus":
            ls = F.fit_torus(P, nom["centre_m"], nom["axis"], float(nom["major_radius_m"]),
                             float(nom["minor_radius_m"]), settings)
            out.update({"fit": ls.to_json(),
                        "major_radius_deviation_m": float(ls.params["major_radius_m"] - float(nom["major_radius_m"])),
                        "minor_radius_deviation_m": float(ls.params["minor_radius_m"] - float(nom["minor_radius_m"])),
                        "location_deviation_m": float(np.linalg.norm(ls.params["centre_m"] - nom["centre_m"])),
                        "form_m": ls.zone_m})
        else:
            out.update({"form_m": float(d.max() - d.min())})
        out["status"] = "measured"
    except F.FitError as error:
        out["status"] = f"not measured: {error}"
    return out


def width_two_point(side_a, side_b, n) -> tuple[np.ndarray, float]:
    """Local two-point widths (ISO 14405 LP) of a width feature: from every point of each side, the distance along
    the normal to the other side's least-squares plane; and the least-squares width."""
    la = F.fit_plane(side_a, n)
    lb = F.fit_plane(side_b, -n)
    na, ca = la.params["normal"], la.params["point_m"]
    nb, cb = lb.params["normal"], lb.params["point_m"]
    wa = np.abs((side_a - cb) @ nb)
    wb = np.abs((side_b - ca) @ na)
    mid = 0.5 * (ca + cb)
    ls = 0.5 * (abs(float((mid - ca) @ na)) + abs(float((mid - cb) @ nb))) * 2.0
    return np.concatenate([wa, wb]), ls


def _measure_width(feature, cad, points_of, settings) -> dict:
    sides = feature.nominal["sides"]
    A = points_of(sides[0])
    B = points_of(sides[1])
    if len(A) < 3 or len(B) < 3:
        return {"status": "too few usable points on the two sides", "points": len(A) + len(B)}
    n = tf.unit(feature.nominal["normal"])
    two, ls = width_two_point(A.points, B.points, n)
    nominal = float(feature.nominal["width_m"])
    return {"points": len(A) + len(B), "width_m": ls, "width_deviation_m": ls - nominal,
            "two_point_min_m": float(two.min()), "two_point_max_m": float(two.max()), "status": "measured"}


def _measure_pocket(feature, cad, points_of, settings) -> dict:
    nom = feature.nominal
    floor = points_of(nom["floor"])
    out = {"points": len(floor)}
    if len(floor) < 3:
        out["status"] = "too few usable points on the floor"
        return out
    n = tf.unit(nom["normal"])
    lf = F.fit_plane(floor.points, n)
    out["floor_flatness_m"] = F.zone_plane(floor.points, n, settings).zone_m
    if nom["top"]:
        top = points_of(nom["top"])
        if len(top) >= 3:
            lt = F.fit_plane(top.points, n)
            depth = float((lt.params["point_m"] - lf.params["point_m"]) @ lf.params["normal"])
            out["depth_m"] = depth
            if nom["depth_m"] is not None:
                out["depth_deviation_m"] = depth - float(nom["depth_m"])
    out["status"] = "measured"
    return out


__all__ = ["Feature", "recognise", "measure", "edge_is_concave", "width_two_point"]
