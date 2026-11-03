"""Evaluation of every characteristic of the project's tolerance kinds on the result (docs/models/inspect.md
"Characteristics"; ISO 1101 / ASME Y14.5 zones, ISO 14405 / ASME Rule #1 sizes).

The same evaluation runs on two point sources (``EvalContext.points_of``): the dense result surface (every
usable vertex of the inspected geometry, cnctwin.inspect.sampling) and the virtual CMM's compensated probe points
(cnctwin.inspect.cmm). Datum-related characteristics map the points into the nominal datum reference frame
(cnctwin.inspect.datums) and evaluate against the CAD there; degrees of freedom the datums leave free are
optimised (linear programs on the small-displacement torsor, re-linearised on the exact geometry).

========================  =========================================================================================
kind                      measured value
========================  =========================================================================================
flatness                  minimum-zone width of the plane points (ISO 12781 MZPL); LS width reported
straightness              plane: max MZ width of line profiles along the face's long direction; cylinder: max over
                          generator lines, or the derived median line's minimum enclosing cylinder diameter when
                          the modifier names the axis (``axis`` / ``DML``)
circularity               max over cross sections of the MZ circle zone (ISO 12181)
cylindricity              MZ cylinder zone (ISO 12180)
profile_surface           2 max |d| from the nominal surface in the datum frame; without datums the zone floats (all
                          six degrees of freedom; ISO 1660)
profile_line              the same per cross section; without datums each section floats in its plane
perpendicularity,         plane: zone width along the nominal normal (in the datum frame, free rotations optimised);
parallelism, angularity   axis: derived median line - a cylindrical zone (diameter) when the nominal axis lies along
                          the primary datum direction, else a planar zone
position                  hole / boss: 2 x the largest distance of the axis of the unrelated actual mating envelope
                          from the true position axis over the feature length (ASME Y14.5 10.3), MMC / LMC bonus from
                          the envelope size; sphere: 2 x centre distance; slot / tab: 2 x the largest centre-plane
                          offset; plane: 2 max |d| from the true plane (ISO)
concentricity (coaxial)   2 x the largest distance of the section centres from the datum axis
symmetry                  2 x the largest deviation of the median plane's offset from the datum plane (the CAD's
                          offset subtracted)
runout / total_runout     range of the normal deviation from the nominal surface coaxial with the datum axis, per
                          cross section (circular) or over the surface (total)
distance, angle           between the least-squares features (plane-plane at the second's centroid, axis-axis,
                          plane-axis, centres); angle between normals / axes
diameter, radius          least-squares size (GG); two-point local sizes (LP) and the envelope (GX hole / GN boss)
                          checked against the limits (ASME Rule #1); limits from plus/minus or an ISO 286 class
roughness                 surface's Ra / Rz / Rt of the feature (surface.json)
thread                    a size whose fit is a thread class (6g, 6H, 2A): ISO 965 / ASME B1.1 limits, GO / NOT GO
========================  =========================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..geom import transform as tf
from . import datums as D
from . import fitting as F
from . import fits286
from . import threads as TH
from .cadref import CadReference
from .sampling import PointSet
from .schema import InspectSettings
from .specs import SizeSpec, ToleranceSpec

ZONE_KINDS = ("flatness", "straightness", "circularity", "cylindricity", "profile_surface", "profile_line",
              "perpendicularity", "parallelism", "angularity", "position", "concentricity", "symmetry", "runout",
              "total_runout")
SIZE_KINDS = ("diameter", "radius", "distance")


class EvalError(Exception):
    """A characteristic that cannot be evaluated (the reason goes to inspection.json)."""


@dataclass
class EvalContext:
    cad: CadReference
    points_of: object                       # faces -> PointSet
    settings: InspectSettings
    datum_specs: dict
    features: list = field(default_factory=list)       # cnctwin.inspect.features.Feature
    surface: object = None                  # cnctwin.inspect.protocols.SurfaceResults
    sizes: dict = field(default_factory=dict)          # frozenset(faces) -> SizeSpec (feature size limits)
    method: str = "surface"
    frames: dict = field(default_factory=dict)

    def with_points(self, points_of) -> "EvalContext":
        return replace(self, points_of=points_of, frames={})


@dataclass
class Evaluation:
    measured: float | None
    passed: bool | None
    record: dict
    points: PointSet | None = None
    deviations: np.ndarray | None = None


# ============================================================================ helpers

def _faces(ctx: EvalContext, selector, what: str = "feature") -> list:
    if selector is None:
        raise EvalError(f"no {what} selector")
    faces = ctx.cad.select(selector)
    if not faces:
        raise EvalError(f"the {what} selector matches no CAD face ({ctx.cad.select_reason(selector)})")
    return faces


def _points(ctx: EvalContext, faces, minimum: int = 1) -> PointSet:
    ps = ctx.points_of(faces)
    if len(ps) < minimum:
        raise EvalError(f"{len(ps)} usable points on faces {list(faces)} ({minimum} needed)")
    return ps


def _nominal(ctx: EvalContext, P, faces_of_points) -> tuple[np.ndarray, np.ndarray]:
    d = np.zeros(len(P))
    N = np.zeros((len(P), 3))
    for f in np.unique(faces_of_points).tolist():
        sel = np.flatnonzero(faces_of_points == f)
        d[sel], N[sel] = ctx.cad.face_distance(int(f), P[sel])
    return d, N


def _frame(ctx: EvalContext, labels):
    key = tuple(labels)
    if key not in ctx.frames:
        def points_of(faces):
            return ctx.points_of(faces).points
        ctx.frames[key] = D.drf_pair(ctx.cad, ctx.datum_specs, list(labels), points_of, ctx.settings)
    return ctx.frames[key]


def _registered(ps: PointSet, T) -> PointSet:
    return ps.transformed(T)


def _surface(ctx, faces) -> str:
    return ctx.cad.face(faces[0]).surface


def _apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def _sections(P, origin, u, count: int, min_points: int):
    """Point index groups of cross sections perpendicular to u (equal bins over the axial range)."""
    z = (P - origin) @ u
    lo, hi = float(z.min()), float(z.max())
    if hi - lo <= 0.0:
        return [np.arange(len(P))], np.array([lo])
    edges = np.linspace(lo, hi, count + 1)
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, count - 1)
    groups, centres = [], []
    for k in range(count):
        sel = np.flatnonzero(idx == k)
        if len(sel) >= min_points:
            groups.append(sel)
            centres.append(0.5 * (edges[k] + edges[k + 1]))
    return groups, np.array(centres)


def _angular_span(P, origin, u) -> float:
    e1, e2 = tf.frame_from_z(u)[:, 0], tf.frame_from_z(u)[:, 1]
    w = P - origin
    ang = np.sort(np.mod(np.arctan2(w @ e2, w @ e1), 2 * math.pi))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi]))
    return float(2 * math.pi - gaps.max())


def derived_median_line(P, axis_point, axis, settings: InspectSettings):
    """Section centres (least-squares circles) of cylinder points along an axis: [k, 3]."""
    u = tf.unit(axis)
    groups, _ = _sections(P, axis_point, u, settings.sections, settings.section_min_points)
    centres = []
    for g in groups:
        if _angular_span(P[g], axis_point, u) < math.pi:
            continue
        c0 = axis_point + float((P[g].mean(axis=0) - axis_point) @ u) * u
        fit = F.fit_circle(P[g], c0, u, settings)
        centres.append(fit.params["centre_m"])
    if len(centres) < 2:
        raise EvalError("fewer than two cross sections cover half a turn: no derived median line")
    return np.array(centres)


def _radial(points, axis_point, u):
    """2-D offsets of points from an axis in a fixed frame perpendicular to it."""
    R = tf.frame_from_z(u)
    w = np.asarray(points) - axis_point
    w = w - np.outer(w @ u, u)
    return np.column_stack([w @ R[:, 0], w @ R[:, 1]]), R


def _radial_free(frame: D.Frame, Q, u, R) -> np.ndarray:
    """d(2-D radial offset)/d(free displacement) of points Q under the frame's free motions: [m, 2, k]."""
    cols = []
    for j in range(3):
        if frame.free_translation[j]:
            v = np.tile(frame.axes[:, j], (len(Q), 1))
            cols.append(v)
    for j in range(3):
        if frame.free_rotation[j]:
            cols.append(np.cross(frame.axes[:, j], Q - frame.origin))
    if not cols:
        return np.zeros((len(Q), 2, 0))
    G = np.zeros((len(Q), 2, len(cols)))
    for k, v in enumerate(cols):
        v = v - np.outer(v @ u, u)
        G[:, 0, k] = v @ R[:, 0]
        G[:, 1, k] = v @ R[:, 1]
    return G


def _deviation_points(P, d) -> list[dict]:
    return [{"x_S_m": float(p[0]), "y_S_m": float(p[1]), "z_S_m": float(p[2]), "deviation_m": float(v)}
            for p, v in zip(P, d)]


def _zone_record(spec: ToleranceSpec, measured: float, method: str, extra: dict | None = None,
                 allowed: float | None = None) -> tuple[bool | None, dict]:
    tol = spec.tolerance_m
    limit = allowed if allowed is not None else tol
    passed = None if limit is None else bool(measured <= limit * (1.0 + 1e-12) + 1e-15)
    rec = {"tolerance_m": tol, "measured_m": float(measured), "method": method}
    if allowed is not None and tol is not None and allowed != tol:
        rec["allowed_m"] = float(allowed)
    rec.update(extra or {})
    return passed, rec


# ============================================================================ size limits

def size_limits(size: SizeSpec, internal: bool | None) -> tuple[float, float, str]:
    """(lower, upper, source) of a size (plus/minus, or an ISO 286 class)."""
    if size.plus_m is not None or size.minus_m is not None:
        return size.nominal_m - abs(size.minus_m or 0.0), size.nominal_m + (size.plus_m or 0.0), "plus/minus"
    if size.fit:
        lo, hi, hole = fits286.limits_m(size.nominal_m, size.fit)
        if internal is not None and hole != bool(internal):
            raise EvalError(f"ISO 286 class {size.fit} is for a {'hole' if hole else 'shaft'} but the feature is "
                            f"{'internal' if internal else 'external'}")
        return lo, hi, f"ISO 286 {size.fit}"
    raise EvalError("the size has neither plus/minus nor a tolerance class")


def _spec_limits(spec: ToleranceSpec, nominal: float | None) -> tuple[float, float, float]:
    """(nominal, lower, upper) of a size characteristic written with nominal_m and plus/minus."""
    nom = spec.nominal_m if spec.nominal_m is not None else nominal
    if nom is None:
        raise EvalError("no nominal value")
    plus = spec.plus_m if spec.plus_m is not None else 0.0
    minus = spec.minus_m if spec.minus_m is not None else 0.0
    if spec.plus_m is None and spec.minus_m is None:
        raise EvalError("no plus / minus limits")
    return nom, nom - abs(minus), nom + plus


# ============================================================================ form

def _flatness(spec, ctx, faces):
    ps = _points(ctx, faces, 3)
    n = ctx.cad.face(faces[0]).params.get("normal")
    if n is None:
        raise EvalError("flatness needs a plane feature")
    mz = F.zone_plane(ps.points, n, ctx.settings)
    ls = F.fit_plane(ps.points, n)
    d, _ = _nominal(ctx, ps.points, ps.faces)
    passed, rec = _zone_record(spec, mz.zone_m, f"{ctx.method}: {len(ps)} points, minimum zone (ISO 12781 MZPL)",
                               {"least_squares_m": ls.zone_m})
    return Evaluation(mz.zone_m, passed, rec, ps, d)


def _straightness(spec, ctx, faces):
    ps = _points(ctx, faces, 4)
    face = ctx.cad.face(faces[0])
    s = ctx.settings
    worst, worst_ls, lines = 0.0, 0.0, 0
    if face.surface == "plane":
        n = tf.unit(face.params["normal"])
        ls = F.fit_plane(ps.points, n)
        R = tf.frame_from_z(ls.params["normal"])
        uv = (ps.points - ls.params["point_m"]) @ R[:, :2]
        _, _, vt = np.linalg.svd(uv - uv.mean(axis=0), full_matrices=False)
        along, across = vt[0], vt[1]
        a = uv @ along
        c = uv @ across
        h = (ps.points - ls.params["point_m"]) @ ls.params["normal"]
        edges = np.linspace(c.min(), c.max(), s.sections + 1)
        for k in range(s.sections):
            sel = (c >= edges[k]) & (c <= edges[k + 1])
            if sel.sum() >= max(3, s.section_min_points // 2):
                mz, lsz, _ = F.zone_line(np.column_stack([a[sel], h[sel]]), s)
                worst, worst_ls = max(worst, mz), max(worst_ls, lsz)
                lines += 1
        method = f"{ctx.method}: {lines} line profiles along the face, minimum zone (ISO 12780 MZLI)"
    elif face.surface in ("cylinder", "cone"):
        u0 = tf.unit(face.params["axis"])
        a0 = np.asarray(face.params["axis_point_m"] if face.surface == "cylinder" else face.params["apex_m"])
        if "AXIS" in spec.modifier.upper() or "DML" in spec.modifier.upper():
            ls = F.fit_cylinder(ps.points, a0, u0, float(face.params.get("radius_m", 0.0)) or float(np.mean(
                F.cylinder_distance(ps.points, a0, u0))), s)
            C = derived_median_line(ps.points, ls.params["axis_point_m"], ls.params["axis"], s)
            fit = F.zone_cylinder(C, ls.params["axis_point_m"], ls.params["axis"], "MCC", s) if len(C) >= 5 else None
            if fit is None:
                xy, _ = _radial(C, ls.params["axis_point_m"], ls.params["axis"])
                _, r = F.min_enclosing_circle(xy)
                worst = 2.0 * r
            else:
                worst = 2.0 * fit.params["radius_m"]
            passed, rec = _zone_record(spec, worst, f"{ctx.method}: derived median line of {len(C)} sections, "
                                                    f"cylindrical zone")
            return Evaluation(worst, passed, rec, ps, None)
        w = ps.points - a0
        z = w @ u0
        R = tf.frame_from_z(u0)
        ang = np.mod(np.arctan2(w @ R[:, 1], w @ R[:, 0]), 2 * math.pi)
        rho = np.linalg.norm(w - np.outer(z, u0), axis=1)
        n_lines = max(4, s.sections)
        for k in range(n_lines):
            centre = 2 * math.pi * k / n_lines
            sel = np.abs(np.angle(np.exp(1j * (ang - centre)))) <= math.pi / n_lines
            if sel.sum() >= max(3, s.section_min_points // 2):
                mz, lsz, _ = F.zone_line(np.column_stack([z[sel], rho[sel]]), s)
                worst, worst_ls = max(worst, mz), max(worst_ls, lsz)
                lines += 1
        method = f"{ctx.method}: {lines} generator lines, minimum zone (ISO 12780 MZLI)"
    else:
        raise EvalError(f"straightness of a {face.surface} face is not defined here (plane, cylinder, cone)")
    if lines == 0:
        raise EvalError("no line profile has enough points")
    passed, rec = _zone_record(spec, worst, method, {"least_squares_m": worst_ls, "lines": lines})
    return Evaluation(worst, passed, rec, ps, None)


def _circularity(spec, ctx, faces):
    ps = _points(ctx, faces, 8)
    face = ctx.cad.face(faces[0])
    s = ctx.settings
    p = face.params
    if face.surface == "cylinder":
        a0, u0 = np.asarray(p["axis_point_m"]), tf.unit(p["axis"])
        ls = F.fit_cylinder(ps.points, a0, u0, float(p["radius_m"]), s)
        a0, u0 = ls.params["axis_point_m"], ls.params["axis"]
    elif face.surface == "cone":
        ls = F.fit_cone(ps.points, p["apex_m"], p["axis"], float(p["semi_angle_rad"]), s)
        a0, u0 = ls.params["apex_m"], ls.params["axis"]
    elif face.surface in ("sphere", "torus"):
        a0 = np.asarray(p["centre_m"])
        u0 = tf.unit(p.get("axis", [0.0, 0.0, 1.0]))
    else:
        raise EvalError(f"circularity of a {face.surface} face is not defined (cylinder, cone, sphere, torus)")
    groups, _ = _sections(ps.points, a0, u0, s.sections, s.section_min_points)
    worst, worst_ls, used = 0.0, 0.0, 0
    for g in groups:
        if _angular_span(ps.points[g], a0, u0) < math.pi:
            continue
        c0 = a0 + float((ps.points[g].mean(axis=0) - a0) @ u0) * u0
        mz = F.zone_circle(ps.points[g], c0, u0, "MZ", s)
        lsc = F.fit_circle(ps.points[g], c0, u0, s)
        worst, worst_ls = max(worst, mz.zone_m), max(worst_ls, lsc.zone_m)
        used += 1
    if used == 0:
        raise EvalError("no cross section has enough points over half a turn")
    passed, rec = _zone_record(spec, worst, f"{ctx.method}: {used} cross sections, minimum zone (ISO 12181 MZCI)",
                               {"least_squares_m": worst_ls, "sections": used})
    return Evaluation(worst, passed, rec, ps, None)


def _cylindricity(spec, ctx, faces):
    ps = _points(ctx, faces, 8)
    face = ctx.cad.face(faces[0])
    if face.surface != "cylinder":
        raise EvalError("cylindricity needs a cylinder feature")
    p = face.params
    mz = F.zone_cylinder(ps.points, p["axis_point_m"], p["axis"], "MZ", ctx.settings)
    ls = F.fit_cylinder(ps.points, p["axis_point_m"], p["axis"], float(p["radius_m"]), ctx.settings)
    d, _ = _nominal(ctx, ps.points, ps.faces)
    passed, rec = _zone_record(spec, mz.zone_m, f"{ctx.method}: {len(ps)} points, minimum zone (ISO 12180 MZCY)",
                               {"least_squares_m": ls.zone_m})
    return Evaluation(mz.zone_m, passed, rec, ps, d)


# ============================================================================ profile

ALL_FREE = D.Frame((), np.zeros(3), np.eye(3))


def _floating_zone(ctx, ps: PointSet, frame: D.Frame, rotations=True, translations=True, iterations: int = 4):
    """2 min max |d| over the frame's free displacements, re-linearised on the exact geometry; (zone, d, P)."""
    P = ps.points.copy()
    d, N = _nominal(ctx, P, ps.faces)
    free = int(np.count_nonzero(frame.free_translation)) * translations + \
        int(np.count_nonzero(frame.free_rotation)) * rotations
    if free == 0:
        return 2.0 * float(np.max(np.abs(d))), d, P
    origin = P.mean(axis=0) if frame is ALL_FREE else frame.origin
    fr = D.Frame(frame.labels, origin, frame.axes, frame.free_translation, frame.free_rotation)
    best = float(np.max(np.abs(d)))
    for _ in range(iterations):
        J = D.free_columns(fr, P, N, translations, rotations)
        _, xi = F.symmetric_zone(d, J)
        T = D.free_motion(fr, xi, translations, rotations)
        P2 = _apply(T, P)
        d2, N2 = _nominal(ctx, P2, ps.faces)
        value = float(np.max(np.abs(d2)))
        if value > best * (1 + 1e-12):
            break
        gain = best - value
        P, d, N, best = P2, d2, N2, value
        if gain <= ctx.settings.lp_tolerance_m:
            break
    return 2.0 * best, d, P


def _profile_surface(spec, ctx, faces):
    ps = _points(ctx, faces, 3)
    if spec.datums:
        nominal, _, T = _frame(ctx, spec.datums)
        reg = _registered(ps, T)
        zone, d, _ = _floating_zone(ctx, reg, nominal)
        method = f"{ctx.method}: {len(ps)} points in the datum frame {'|'.join(spec.datums)}"
    else:
        zone, d, _ = _floating_zone(ctx, ps, ALL_FREE)
        method = f"{ctx.method}: {len(ps)} points, zone floating (no datums: form and size)"
    passed, rec = _zone_record(spec, zone, method)
    return Evaluation(zone, passed, rec, ps, d)


def _profile_line(spec, ctx, faces):
    ps = _points(ctx, faces, 6)
    s = ctx.settings
    if spec.datums:
        nominal, _, T = _frame(ctx, spec.datums)
        ps = _registered(ps, T)
    P = ps.points
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    direction = vt[0]                                        # sections across the face's long direction
    z = (P - c) @ direction
    edges = np.linspace(z.min(), z.max(), s.sections + 1)
    worst, used, all_d = 0.0, 0, np.zeros(len(P))
    for k in range(s.sections):
        sel = np.flatnonzero((z >= edges[k]) & (z <= edges[k + 1]))
        if len(sel) < max(3, s.section_min_points // 2):
            continue
        sub = ps.subset(sel)
        if spec.datums:
            zone, d, _ = _floating_zone(ctx, sub, nominal)
        else:
            frame = D.Frame((), sub.points.mean(axis=0), tf.frame_from_z(direction),
                            np.array([True, True, False]), np.array([False, False, True]))
            zone, d, _ = _floating_zone(ctx, sub, frame)
        all_d[sel] = d
        worst = max(worst, zone)
        used += 1
    if used == 0:
        raise EvalError("no cross section has enough points")
    passed, rec = _zone_record(spec, worst, f"{ctx.method}: {used} sections" +
                               (f" in the datum frame {'|'.join(spec.datums)}" if spec.datums else
                                ", each floating in its plane"), {"sections": used})
    return Evaluation(worst, passed, rec, ps, all_d)


# ============================================================================ orientation

def _orientation(spec, ctx, faces):
    if not spec.datums:
        raise EvalError(f"{spec.kind} needs datums")
    ps = _points(ctx, faces, 3)
    nominal, _, T = _frame(ctx, spec.datums)
    reg = _registered(ps, T)
    face = ctx.cad.face(faces[0])
    e3 = nominal.axes[:, 2]
    s = ctx.settings
    if face.surface == "plane":
        n = tf.unit(face.params["normal"])
        h = reg.points @ n
        J = D.free_columns(nominal, reg.points, np.tile(n, (len(reg), 1)), translations=False)
        width, _ = F.band_zone(h, J)
        passed, rec = _zone_record(spec, width, f"{ctx.method}: {len(ps)} points, planar zone in the datum frame "
                                                f"{'|'.join(spec.datums)}")
        d, _ = _nominal(ctx, reg.points, reg.faces)
        return Evaluation(width, passed, rec, ps, d)
    if face.surface != "cylinder":
        raise EvalError(f"{spec.kind} of a {face.surface} face is evaluated for planes and cylinders")
    p = face.params
    u = tf.unit(p["axis"])
    ls = F.fit_cylinder(reg.points, p["axis_point_m"], u, float(p["radius_m"]), s)
    C = derived_median_line(reg.points, ls.params["axis_point_m"], ls.params["axis"], s)
    if abs(float(u @ e3)) > 1.0 - 1e-6:
        xy, _ = _radial(C, p["axis_point_m"], u)
        _, r = F.min_enclosing_circle(xy)
        value = 2.0 * r
        zone = "cylindrical zone along the primary datum direction"
    else:
        m = tf.unit(e3 - (e3 @ u) * u)
        h = C @ m
        J = D.free_columns(nominal, C, np.tile(m, (len(C), 1)), translations=False)
        value, _ = F.band_zone(h, J)
        zone = "planar zone"
    passed, rec = _zone_record(spec, value, f"{ctx.method}: derived median line ({len(C)} sections), {zone}, datum "
                                            f"frame {'|'.join(spec.datums)}")
    return Evaluation(value, passed, rec, ps, None)


# ============================================================================ location

def _true_axis(spec, ctx, face, nominal: D.Frame | None):
    p = face.params
    point = np.asarray(p["axis_point_m"] if face.surface == "cylinder" else p.get("apex_m", p.get("centre_m")),
                       dtype=float)
    u = tf.unit(p["axis"]) if "axis" in p else (nominal.axes[:, 2] if nominal is not None else np.array([0, 0, 1.0]))
    if spec.basic_m is not None and nominal is not None:
        # basic_mm: coordinates (e1, e2[, e3]) of a point of the true axis in the datum frame
        local = nominal.to_local(point)
        for k, value in enumerate(list(spec.basic_m)[:3]):
            local[k] = value
        point = nominal.to_S(local)
        if abs(float(u @ nominal.axes[:, 2])) > 1.0 - 1e-6:
            u = nominal.axes[:, 2] * (1.0 if u @ nominal.axes[:, 2] > 0 else -1.0)
    return point, u


def _feature_size(ctx, faces, spec) -> SizeSpec | None:
    if spec.size is not None:
        return spec.size
    return ctx.sizes.get(frozenset(ctx.cad.feature_faces(faces[0])))


def _width_feature(ctx, faces):
    fset = set(ctx.cad.feature_faces(faces[0]))
    for f in ctx.features:
        if f.kind in ("slot", "tab") and any(set(side) <= set(f.faces) and fset <= set(f.faces)
                                             for side in f.nominal["sides"]):
            return f
    return None


def _position(spec, ctx, faces):
    face = ctx.cad.face(faces[0])
    s = ctx.settings
    nominal = T = None
    if spec.datums:
        nominal, _, T = _frame(ctx, spec.datums)
    ps = _points(ctx, faces, 3)
    reg = _registered(ps, T) if T is not None else ps
    frame = nominal if nominal is not None else D.Frame((), np.zeros(3), np.eye(3), np.zeros(3, bool),
                                                          np.zeros(3, bool))
    where = f"datum frame {'|'.join(spec.datums)}" if spec.datums else "the stock frame (no datums)"
    if face.surface == "cylinder":
        t0, u = _true_axis(spec, ctx, face, nominal)
        internal = bool(face.concave)
        mode = "MIC" if internal else "MCC"
        ame = F.zone_cylinder(reg.points, face.params["axis_point_m"], face.params["axis"], mode, s)
        note = ""
        if not ame.converged:
            ame = F.fit_cylinder(reg.points, face.params["axis_point_m"], face.params["axis"],
                                 float(face.params["radius_m"]), s)
            note = "the envelope is not bounded: the least-squares axis is used"
        a, ua = ame.params["axis_point_m"], ame.params["axis"]
        z = (reg.points - t0) @ u
        ends = []
        for zk in (float(z.min()), float(z.max())):
            sk = (zk - float((a - t0) @ u)) / float(ua @ u)
            ends.append(a + sk * ua)
        Q = np.array(ends)
        o, R = _radial(Q, t0, u)
        G = _radial_free(frame, Q, u, R)
        radius, _ = F.radial_zone(o, G, s.zone_polygon_sides, s)
        measured = 2.0 * radius
        extra = {"envelope_diameter_m": 2.0 * ame.params["radius_m"], "true_position_S_m": [float(v) for v in t0],
                 "axis_S": [float(v) for v in u]}
        allowed = spec.tolerance_m
        mc = spec.material_condition
        if mc != "RFS" and spec.tolerance_m is not None:
            size = _feature_size(ctx, faces, spec)
            if size is None:
                extra["bonus_note"] = f"{mc} without the feature's size limits: evaluated without bonus"
            else:
                lo, hi, _ = size_limits(size, internal)
                envelope = 2.0 * ame.params["radius_m"]
                if mc == "MMC":
                    mmc = lo if internal else hi
                    bonus = max(0.0, (envelope - mmc) if internal else (mmc - envelope))
                    extra.update({"mmc_m": mmc, "bonus_m": bonus})
                else:
                    other = F.zone_cylinder(reg.points, a, ua, "MCC" if internal else "MIC", s)
                    lmc = hi if internal else lo
                    mme = 2.0 * other.params["radius_m"]
                    bonus = max(0.0, (lmc - mme) if internal else (mme - lmc))
                    extra.update({"lmc_m": lmc, "bonus_m": bonus, "minimum_material_envelope_m": mme})
                allowed = spec.tolerance_m + bonus
        if note:
            extra["note"] = note
        passed, rec = _zone_record(spec, measured, f"{ctx.method}: axis of the actual mating envelope ({mode}) in "
                                                   f"{where}", extra, allowed)
        return Evaluation(measured, passed, rec, ps, None)
    if face.surface == "sphere":
        p = face.params
        t0 = np.asarray(p["centre_m"], float)
        if spec.basic_m is not None and nominal is not None:
            local = nominal.to_local(t0)
            local[:len(spec.basic_m)] = spec.basic_m[:3]
            t0 = nominal.to_S(local)
        env = F.zone_sphere(reg.points, t0, "MIC" if face.concave else "MCC", s)
        o = env.params["centre_m"] - t0
        cols = [frame.axes[:, j] for j in range(3) if frame.free_translation[j]]
        cols += [np.cross(frame.axes[:, j], env.params["centre_m"] - frame.origin) for j in range(3)
                 if frame.free_rotation[j]]
        if cols:
            G = np.column_stack(cols)
            xi, *_ = np.linalg.lstsq(G, -o, rcond=None)
            o = o + G @ xi
        measured = 2.0 * float(np.linalg.norm(o))
        passed, rec = _zone_record(spec, measured, f"{ctx.method}: centre of the envelope sphere in {where}")
        return Evaluation(measured, passed, rec, ps, None)
    if face.surface == "plane":
        width = _width_feature(ctx, faces)
        if width is not None:
            return _centre_plane_location(spec, ctx, width, nominal, T, frame, where)
        n = tf.unit(face.params["normal"])
        p0 = np.asarray(face.params["point_m"], float)
        if spec.basic_m is not None and nominal is not None:
            local = nominal.to_local(p0)
            local[:len(spec.basic_m)] = spec.basic_m[:3]
            p0 = nominal.to_S(local)
        d = (reg.points - p0) @ n
        J = D.free_columns(frame, reg.points, np.tile(n, (len(reg), 1)))
        half, _ = F.symmetric_zone(d, J)
        passed, rec = _zone_record(spec, 2.0 * half, f"{ctx.method}: location of the plane in {where} (ISO 1101)")
        return Evaluation(2.0 * half, passed, rec, ps, d)
    raise EvalError(f"position of a {face.surface} face is evaluated for cylinders, spheres, planes and slots")


def _centre_plane(ctx, width, T):
    """(points of the two sides registered, LS planes, nominal centre point, normal, corner points)."""
    sides = width.nominal["sides"]
    A = _points(ctx, sides[0], 3)
    B = _points(ctx, sides[1], 3)
    if T is not None:
        A, B = A.transformed(T), B.transformed(T)
    n = tf.unit(width.nominal["normal"])
    la, lb = F.fit_plane(A.points, n), F.fit_plane(B.points, -n)
    pa = np.asarray(ctx.cad.face(sides[0][0]).params["point_m"], float)
    pb = np.asarray(ctx.cad.face(sides[1][0]).params["point_m"], float)
    nominal_mid = 0.5 * (pa + pb)
    R = tf.frame_from_z(n)
    uv = np.concatenate([A.points, B.points]) @ R[:, :2]
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    corners2 = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    h_mid = float(nominal_mid @ n)
    corners = corners2 @ R[:, :2].T + h_mid * n
    # median-plane height at each corner: halfway between the two least-squares planes {p . m = d}, each reached
    # from the corner along n: h = c . n + (d - c . m) / (n . m)
    heights = []
    for c in corners:
        hs = []
        for fit in (la, lb):
            m = fit.params["normal"]
            hs.append(float(c @ n) + (float(fit.params["point_m"] @ m) - float(c @ m)) / float(n @ m))
        heights.append(0.5 * (hs[0] + hs[1]))
    return A, B, nominal_mid, n, corners, np.array(heights)


def _centre_plane_location(spec, ctx, width, nominal, T, frame, where):
    A, B, mid, n, corners, heights = _centre_plane(ctx, width, T)
    target = mid
    if spec.basic_m is not None and nominal is not None:
        local = nominal.to_local(mid)
        local[:len(spec.basic_m)] = spec.basic_m[:3]
        target = nominal.to_S(local)
    offsets = heights - float(target @ n)
    Qc = corners + (heights - corners @ n)[:, None] * n
    J = D.free_columns(frame, Qc, np.tile(n, (len(Qc), 1)))
    half, _ = F.symmetric_zone(offsets, J)
    passed, rec = _zone_record(spec, 2.0 * half, f"{ctx.method}: centre plane of the {width.kind} in {where}",
                               {"width_feature": width.id})
    return Evaluation(2.0 * half, passed, rec, PointSet.concat([A, B]), None)


def _datum_axis(spec, ctx):
    """(nominal axis point, direction, registration) of the datum axis of runout / coaxiality: one axis datum, the
    common axis of several axis datums (through their envelope axes' mid points), or the frame's e3 axis."""
    specs = [ctx.datum_specs.get(l) for l in spec.datums]
    if any(s_ is None for s_ in specs):
        raise EvalError(f"datum {[l for l, s_ in zip(spec.datums, specs) if s_ is None][0]} is not defined")
    noms = [D.nominal_datum(ctx.cad, s_) for s_ in specs]
    if all(n.kind == "axis" for n in noms):
        meas = [D.measured_datum(n, ctx.points_of(n.faces).points, ctx.settings) for n in noms]

        def mid(d, P):
            z = (P - d.point) @ d.direction
            return d.point + 0.5 * (z.min() + z.max()) * d.direction

        if len(noms) == 1:
            a_n, u_n = noms[0].point, noms[0].direction
            a_m, u_m = meas[0].point, meas[0].direction
        else:
            pts_n = [mid(n, ctx.cad.face_vertices(n.faces[0])) for n in noms]
            pts_m = [mid(m, ctx.points_of(m.faces).points) for m in meas]
            a_n, u_n = pts_n[0], tf.unit(pts_n[-1] - pts_n[0])
            a_m, u_m = pts_m[0], tf.unit(pts_m[-1] - pts_m[0])
            if u_m @ u_n < 0:
                u_m = -u_m
        R = _align(u_m, u_n)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = a_n - R @ a_m
        return a_n, u_n, T
    nominal, _, T = _frame(ctx, spec.datums)
    return nominal.origin, nominal.axes[:, 2], T


def _align(u, v) -> np.ndarray:
    """Rotation taking unit vector u onto v (the smallest one)."""
    u, v = tf.unit(u), tf.unit(v)
    axis = np.cross(u, v)
    s = float(np.linalg.norm(axis))
    c = float(u @ v)
    if s < 1e-15:
        return np.eye(3) if c > 0 else tf.axis_angle(tf.perpendicular(u), math.pi)
    return tf.axis_angle(axis / s, math.atan2(s, c))


def _runout(spec, ctx, faces, total: bool):
    if not spec.datums:
        raise EvalError(f"{spec.kind} needs a datum axis")
    a_n, u_n, T = _datum_axis(spec, ctx)
    ps = _points(ctx, faces, 4)
    reg = _registered(ps, T)
    d, _ = _nominal(ctx, reg.points, reg.faces)
    face = ctx.cad.face(faces[0])
    s = ctx.settings
    if total:
        value = float(d.max() - d.min())
        how = "range over the surface"
    else:
        w = reg.points - a_n
        z = w @ u_n
        rho = np.linalg.norm(w - np.outer(z, u_n), axis=1)
        coord = rho if (face.surface == "plane" and abs(float(tf.unit(face.params["normal"]) @ u_n)) > 0.9) else z
        edges = np.linspace(coord.min(), coord.max(), s.sections + 1)
        value, used = 0.0, 0
        for k in range(s.sections):
            sel = (coord >= edges[k]) & (coord <= edges[k + 1])
            if sel.sum() >= s.section_min_points:
                value = max(value, float(d[sel].max() - d[sel].min()))
                used += 1
        if used == 0:
            raise EvalError("no measuring circle has enough points")
        how = f"range per measuring circle ({used} circles)"
    passed, rec = _zone_record(spec, value, f"{ctx.method}: {len(ps)} points, {how}, datum axis "
                                            f"{'-'.join(spec.datums)}")
    return Evaluation(value, passed, rec, ps, d)


def _concentricity(spec, ctx, faces):
    if not spec.datums:
        raise EvalError("coaxiality needs a datum axis")
    a_n, u_n, T = _datum_axis(spec, ctx)
    ps = _points(ctx, faces, 8)
    reg = _registered(ps, T)
    face = ctx.cad.face(faces[0])
    if face.surface != "cylinder":
        raise EvalError("coaxiality is evaluated for cylinders")
    p = face.params
    ls = F.fit_cylinder(reg.points, p["axis_point_m"], p["axis"], float(p["radius_m"]), ctx.settings)
    C = derived_median_line(reg.points, ls.params["axis_point_m"], ls.params["axis"], ctx.settings)
    o, _ = _radial(C, a_n, u_n)
    value = 2.0 * float(np.max(np.linalg.norm(o, axis=1)))
    passed, rec = _zone_record(spec, value, f"{ctx.method}: {len(C)} section centres against the datum axis "
                                            f"{'-'.join(spec.datums)}")
    return Evaluation(value, passed, rec, ps, None)


def _symmetry(spec, ctx, faces):
    if not spec.datums:
        raise EvalError("symmetry needs a datum centre plane")
    width = _width_feature(ctx, faces)
    if width is None:
        raise EvalError("symmetry is evaluated for a slot or tab (two opposite planes)")
    dspec = ctx.datum_specs.get(spec.datums[0])
    if dspec is None:
        raise EvalError(f"datum {spec.datums[0]} is not defined")
    dfaces = _faces(ctx, dspec.feature, "datum")
    n = tf.unit(width.nominal["normal"])
    # the datum is the plane its selector names (its tangent-plane simulator); the median plane's offset from it
    # is compared with the CAD's
    ref = D.measured_datum(D.nominal_datum(ctx.cad, dspec), ctx.points_of(dfaces).points, ctx.settings)
    if ref.kind != "plane":
        raise EvalError(f"symmetry needs a planar datum, {spec.datums[0]} is an {ref.kind}")
    ref_point, ref_normal = ref.point, tf.unit(ref.direction)
    nominal_ref = np.asarray(ctx.cad.face(dfaces[0]).params["point_m"], float)
    A, B, mid, _, corners, heights = _centre_plane(ctx, width, None)
    if ref_normal @ n < 0:
        ref_normal = -ref_normal
    nominal_offset = float((mid - nominal_ref) @ n)
    Qc = corners + (heights - corners @ n)[:, None] * n
    offsets = (Qc - ref_point) @ ref_normal - nominal_offset
    value = 2.0 * float(np.max(np.abs(offsets)))
    passed, rec = _zone_record(spec, value, f"{ctx.method}: median plane of the {width.kind} against the datum centre "
                                            f"plane {spec.datums[0]}", {"width_feature": width.id})
    return Evaluation(value, passed, rec, PointSet.concat([A, B]), None)


# ============================================================================ sizes, distances, angles

def _two_point_cylinder(P, a, u, settings: InspectSettings) -> np.ndarray:
    """Local two-point diameters: per section, opposite angular bins (36 per turn) about the least-squares axis."""
    groups, _ = _sections(P, a, u, settings.sections, settings.section_min_points)
    R = tf.frame_from_z(u)
    out = []
    nb = 36
    for g in groups:
        w = P[g] - a
        z = w @ u
        radial = w - np.outer(z, u)
        rho = np.linalg.norm(radial, axis=1)
        ang = np.mod(np.arctan2(radial @ R[:, 1], radial @ R[:, 0]), 2 * math.pi)
        b = (ang / (2 * math.pi) * nb).astype(np.int64) % nb
        mean = np.full(nb, np.nan)
        for k in range(nb):
            sel = b == k
            if sel.any():
                mean[k] = float(rho[sel].mean())
        for k in range(nb // 2):
            if math.isfinite(mean[k]) and math.isfinite(mean[k + nb // 2]):
                out.append(mean[k] + mean[k + nb // 2])
    return np.array(out)


def _size(spec, ctx, faces):
    face = ctx.cad.face(faces[0])
    s = ctx.settings
    ps = _points(ctx, faces, 6)
    radius_kind = spec.kind == "radius"
    p = face.params
    if spec.size is not None and spec.size.thread_class:
        return _thread(spec, ctx, faces)
    if face.surface == "cylinder":
        internal = bool(face.concave)
        ls = F.fit_cylinder(ps.points, p["axis_point_m"], p["axis"], float(p["radius_m"]), s)
        env = F.zone_cylinder(ps.points, ls.params["axis_point_m"], ls.params["axis"], "MIC" if internal else "MCC", s)
        two = _two_point_cylinder(ps.points, ls.params["axis_point_m"], ls.params["axis"], s)
        ls_size, env_size, nominal_cad = 2.0 * ls.params["radius_m"], 2.0 * env.params["radius_m"], 2.0 * float(
            p["radius_m"])
    elif face.surface == "sphere":
        internal = bool(face.concave)
        ls = F.fit_sphere(ps.points, p["centre_m"], float(p["radius_m"]), s)
        env = F.zone_sphere(ps.points, ls.params["centre_m"], "MIC" if internal else "MCC", s)
        rho = np.linalg.norm(ps.points - ls.params["centre_m"], axis=1)
        two = np.array([2.0 * rho.min(), 2.0 * rho.max()])
        ls_size, env_size, nominal_cad = 2.0 * ls.params["radius_m"], 2.0 * env.params["radius_m"], 2.0 * float(
            p["radius_m"])
    elif face.surface == "torus" and radius_kind:
        internal = bool(face.concave)
        ls = F.fit_torus(ps.points, p["centre_m"], p["axis"], float(p["major_radius_m"]), float(p["minor_radius_m"]), s)
        ls_size = 2.0 * ls.params["minor_radius_m"]
        env_size, two, nominal_cad = None, np.zeros(0), 2.0 * float(p["minor_radius_m"])
    else:
        raise EvalError(f"a {spec.kind} of a {face.surface} face is evaluated for cylinders, spheres and tori")
    scale = 0.5 if radius_kind else 1.0
    if spec.size is not None:
        lo, hi, source = size_limits(spec.size, internal)
        nominal = spec.size.nominal_m
    else:
        nominal, lo, hi = _spec_limits(spec, None)
        lo, hi, nominal = lo / scale, hi / scale, nominal / scale
        source = "plus/minus"
    measured = ls_size
    checks = {}
    if len(two):
        checks["two_point"] = bool(two.min() >= lo - 1e-15 and two.max() <= hi + 1e-15)
    if env_size is not None:
        checks["envelope"] = bool(env_size >= lo - 1e-15) if internal else bool(env_size <= hi + 1e-15)
    checks["least_squares"] = bool(lo - 1e-15 <= ls_size <= hi + 1e-15)
    passed = all(checks.values())
    rec = {"nominal_m": nominal * scale, "lower_m": lo * scale, "upper_m": hi * scale,
           "measured_m": measured * scale, "deviation_m": (measured - nominal) * scale,
           "size": {"least_squares_m": ls_size * scale, "envelope_m": None if env_size is None else env_size * scale,
                    "two_point_min_m": float(two.min()) * scale if len(two) else None,
                    "two_point_max_m": float(two.max()) * scale if len(two) else None, "limits": source,
                    "fit": spec.size.fit if spec.size is not None else None, "checks": checks,
                    "nominal_cad_m": nominal_cad * scale,
                    "rule": "ASME Y14.5 Rule #1: two-point local sizes within the limits, the actual mating envelope "
                            "within maximum material; measured = least squares (ISO 14405 GG)"},
           "method": f"{ctx.method}: {len(ps)} points"}
    if spec.size is not None and spec.size.fit:
        rec["fit"] = spec.size.fit
    d, _ = _nominal(ctx, ps.points, ps.faces)
    return Evaluation(measured * scale, passed, rec, ps, d)


def _direction_and_point(ctx, faces, ps: PointSet):
    """(kind, point, direction) of the least-squares feature of a face set."""
    face = ctx.cad.face(faces[0])
    p = face.params
    s = ctx.settings
    if face.surface == "plane":
        ls = F.fit_plane(ps.points, p["normal"])
        return "plane", ls.params["point_m"], ls.params["normal"]
    if face.surface == "cylinder":
        ls = F.fit_cylinder(ps.points, p["axis_point_m"], p["axis"], float(p["radius_m"]), s)
        a, u = ls.params["axis_point_m"], ls.params["axis"]
        z = (ps.points - a) @ u
        return "axis", a + 0.5 * (z.min() + z.max()) * u, u
    if face.surface == "sphere":
        ls = F.fit_sphere(ps.points, p["centre_m"], float(p["radius_m"]), s)
        return "point", ls.params["centre_m"], None
    if face.surface == "cone":
        ls = F.fit_cone(ps.points, p["apex_m"], p["axis"], float(p["semi_angle_rad"]), s)
        return "axis", ls.params["apex_m"], ls.params["axis"]
    raise EvalError(f"a {face.surface} face has no point or direction for a distance or angle")


def _nominal_direction_and_point(ctx, faces):
    face = ctx.cad.face(faces[0])
    p = face.params
    if face.surface == "plane":
        return "plane", np.asarray(p["point_m"], float), tf.unit(p["normal"])
    if face.surface == "cylinder":
        u = tf.unit(p["axis"])
        a = np.asarray(p["axis_point_m"], float)
        z = (ctx.cad.face_vertices(faces[0]) - a) @ u
        return "axis", a + 0.5 * (z.min() + z.max()) * u, u
    if face.surface == "sphere":
        return "point", np.asarray(p["centre_m"], float), None
    if face.surface == "cone":
        return "axis", np.asarray(p["apex_m"], float), tf.unit(p["axis"])
    raise EvalError(f"a {face.surface} face has no point or direction for a distance or angle")


def _between_distance(f1, f2) -> float:
    (k1, p1, u1), (k2, p2, u2) = f1, f2
    if k1 == "plane":
        return abs(float((p2 - p1) @ u1))
    if k2 == "plane":
        return abs(float((p1 - p2) @ u2))
    if k1 == "point" and k2 == "point":
        return float(np.linalg.norm(p2 - p1))
    if k1 == "axis" and k2 == "point":
        w = p2 - p1
        return float(np.linalg.norm(w - (w @ u1) * u1))
    if k1 == "point" and k2 == "axis":
        w = p1 - p2
        return float(np.linalg.norm(w - (w @ u2) * u2))
    w = p2 - p1
    cross = np.cross(u1, u2)
    if float(np.linalg.norm(cross)) < 1e-9:
        return float(np.linalg.norm(w - (w @ u1) * u1))
    return abs(float(w @ tf.unit(cross)))


def _distance(spec, ctx):
    if len(spec.between) != 2:
        raise EvalError("a distance needs between: [feature, feature]")
    faces = [_faces(ctx, b, "between") for b in spec.between]
    sets = [_points(ctx, f, 3) for f in faces]
    meas = [_direction_and_point(ctx, f, ps) for f, ps in zip(faces, sets)]
    nom = [_nominal_direction_and_point(ctx, f) for f in faces]
    measured = _between_distance(*meas)
    nominal_cad = _between_distance(*nom)
    nominal, lo, hi = _spec_limits(spec, nominal_cad)
    passed = bool(lo - 1e-15 <= measured <= hi + 1e-15)
    rec = {"nominal_m": nominal, "plus_m": spec.plus_m, "minus_m": spec.minus_m, "lower_m": lo, "upper_m": hi,
           "measured_m": measured, "deviation_m": measured - nominal, "nominal_cad_m": nominal_cad,
           "method": f"{ctx.method}: least-squares {meas[0][0]} to {meas[1][0]} ({len(sets[0])} + {len(sets[1])} points)"}
    if abs(nominal - nominal_cad) > max(hi - nominal, nominal - lo, 1e-9):
        rec["note"] = f"the specified nominal differs from the CAD's ({nominal_cad * 1e3:.4f} mm) by more than the " \
                      f"tolerance"
    return Evaluation(measured, passed, rec, PointSet.concat(sets), None)


def _angle(spec, ctx):
    if len(spec.between) != 2:
        raise EvalError("an angle needs between: [feature, feature]")
    faces = [_faces(ctx, b, "between") for b in spec.between]
    sets = [_points(ctx, f, 3) for f in faces]
    meas = [_direction_and_point(ctx, f, ps) for f, ps in zip(faces, sets)]
    nom = [_nominal_direction_and_point(ctx, f) for f in faces]
    if meas[0][2] is None or meas[1][2] is None:
        raise EvalError("an angle needs two features with a direction")

    def ang(a, b):
        return math.acos(float(np.clip(tf.unit(a) @ tf.unit(b), -1.0, 1.0)))

    measured = ang(meas[0][2], meas[1][2])
    nominal_cad = ang(nom[0][2], nom[1][2])
    nominal = spec.nominal_rad if spec.nominal_rad is not None else nominal_cad
    if abs(nominal - (math.pi - nominal_cad)) < abs(nominal - nominal_cad):
        measured, nominal_cad = math.pi - measured, math.pi - nominal_cad   # the specification counts the supplement
    plus = spec.plus_rad if spec.plus_rad is not None else 0.0
    minus = spec.minus_rad if spec.minus_rad is not None else 0.0
    if spec.plus_rad is None and spec.minus_rad is None:
        raise EvalError("an angle needs plus_deg / minus_deg")
    lo, hi = nominal - abs(minus), nominal + plus
    passed = bool(lo - 1e-15 <= measured <= hi + 1e-15)
    rec = {"nominal_rad": nominal, "plus_rad": spec.plus_rad, "minus_rad": spec.minus_rad, "lower_rad": lo,
           "upper_rad": hi, "measured_rad": measured, "deviation_rad": measured - nominal,
           "nominal_cad_rad": nominal_cad, "method": f"{ctx.method}: least-squares directions"}
    return Evaluation(measured, passed, rec, PointSet.concat(sets), None)


# ============================================================================ roughness and threads

def _roughness(spec, ctx, faces):
    if ctx.surface is None:
        raise EvalError("no surface results (surface.json) are wired: roughness is not evaluated")
    entry = None
    for f in ctx.surface.features():
        if str(f.get("id")) == spec.id:
            entry = f
    if entry is None:
        raise EvalError(f"surface.json has no feature {spec.id!r}")
    for key, limit in (("Ra_m", spec.Ra_m), ("Rz_m", spec.Rz_m), ("Rt_m", spec.Rt_m)):
        if limit is None:
            continue
        value = entry.get(key)
        if value is None:
            raise EvalError(f"surface.json gives no {key} for {spec.id!r} ({entry.get('note', 'no machined surface')})")
        passed = entry.get("pass") if isinstance(entry.get("pass"), bool) else bool(float(value) <= limit)
        rec = {"tolerance_m": limit, "measured_m": float(value), "parameter": key[:-2], "method": "surface: kinematic "
               "profile simulation, ISO 4287 / 4288 (surface.json)", "lines": entry.get("lines"), "op": entry.get("op"),
               "tools": [entry["tool"]] if isinstance(entry.get("tool"), int) else entry.get("tool")}
        return Evaluation(float(value), passed, rec, None, None)
    raise EvalError("a roughness tolerance needs Ra_um, Rz_um or Rt_um")


def _thread(spec, ctx, faces):
    face = ctx.cad.face(faces[0])
    internal = bool(face.concave) if face.surface == "cylinder" else None
    size = spec.size
    if size is None or not size.fit:
        raise EvalError("a thread needs diameter_mm: {nominal, fit: <class>}")
    measured_threads = ctx.surface.threads() if ctx.surface is not None else []
    match = None
    for t in measured_threads:
        kind_ok = internal is None or (t.get("kind") == ("internal" if internal else "external"))
        pitch = float(t.get("pitch_nominal_m") or t.get("pitch_m") or 0.0)
        major = float(t.get("major_diameter_m") or 0.0)
        if kind_ok and pitch > 0 and abs(major - size.nominal_m) <= ctx.settings.thread_match_rel * size.nominal_m:
            match = t
            break
    pitch = float(match.get("pitch_nominal_m")) if match is not None else None
    if pitch is None and not TH.UTS_CLASS.match(size.fit.upper()) and not size.fit.upper().startswith("M"):
        try:
            pitch = TH.coarse_pitch_m(size.nominal_m)
        except TH.ThreadSpecError:
            pitch = None
    tspec = TH.parse(size.fit, internal, size.nominal_m, pitch)
    lim = TH.limits(tspec)
    if match is None:
        raise EvalError(f"no measured thread matches the feature (surface.json threads); {tspec.designation} limits: "
                        f"pitch diameter {lim.d2_min_m * 1e3:.4f}-{lim.d2_max_m * 1e3:.4f} mm")
    rec = TH.judge(lim, match)
    rec["kind"] = "thread"
    rec["measured_thread"] = match.get("id")
    rec["method"] = "surface: simulated thread flanks, three-wire / simple pitch diameter (surface.json), gauges per " \
                    "ISO 1502"
    return Evaluation(rec["pitch_diameter_m"]["measured"], rec["pass"], rec, None, None)


# ============================================================================ dispatch

def evaluate(spec: ToleranceSpec, ctx: EvalContext) -> Evaluation:
    """Evaluate one characteristic (raises EvalError with the reason when it cannot)."""
    kind = spec.kind
    if kind == "distance":
        return _distance(spec, ctx)
    if kind == "angle":
        return _angle(spec, ctx)
    faces = _faces(ctx, spec.feature)
    if kind == "roughness":
        return _roughness(spec, ctx, faces)
    if kind in ("diameter", "radius"):
        return _size(spec, ctx, faces)
    if kind in ZONE_KINDS and spec.tolerance_m is None:
        raise EvalError(f"a {kind} tolerance needs tolerance_mm")
    if kind == "flatness":
        return _flatness(spec, ctx, faces)
    if kind == "straightness":
        return _straightness(spec, ctx, faces)
    if kind == "circularity":
        return _circularity(spec, ctx, faces)
    if kind == "cylindricity":
        return _cylindricity(spec, ctx, faces)
    if kind == "profile_surface":
        return _profile_surface(spec, ctx, faces)
    if kind == "profile_line":
        return _profile_line(spec, ctx, faces)
    if kind in ("perpendicularity", "parallelism", "angularity"):
        return _orientation(spec, ctx, faces)
    if kind == "position":
        return _position(spec, ctx, faces)
    if kind == "concentricity":
        return _concentricity(spec, ctx, faces)
    if kind == "symmetry":
        return _symmetry(spec, ctx, faces)
    if kind in ("runout", "total_runout"):
        return _runout(spec, ctx, faces, total=kind == "total_runout")
    raise EvalError(f"tolerance kind {kind!r} is not known")


__all__ = ["EvalContext", "Evaluation", "EvalError", "evaluate", "size_limits", "derived_median_line", "ZONE_KINDS",
           "SIZE_KINDS"]
