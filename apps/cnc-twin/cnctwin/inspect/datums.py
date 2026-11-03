"""Datum features, datum feature simulators and datum reference frames (docs/models/inspect.md "Datums").

**Simulators** (ASME Y14.5-2018 7, Y14.5.1; ISO 5459 default associations) on the result points of a datum
feature, in precedence order:

* plane: the tangent plane (constrained minimax, the outer plane of the minimum zone); a secondary plane keeps
  its normal perpendicular to the primary direction (one rotation left), a tertiary plane has both fixed (it only
  translates to the outermost point);
* cylinder: maximum inscribed (hole) or minimum circumscribed (boss) cylinder; free when primary, its axis
  parallel to the primary direction when secondary or tertiary;
* sphere: maximum inscribed / minimum circumscribed sphere (a point); cone: least-squares axis and apex.

The same construction on the CAD's analytic faces gives the **nominal** frame, so a result point maps into the
nominal frame by ``registration`` = nominal frame o measured frame^-1, and every characteristic with datums is
evaluated against the CAD there.

**Frame rules** (axes e1, e2, e3; origin O): a primary plane sets e3 = -n (into the material) and O on the
plane; a primary axis sets e3 along it (canonical sign) and O on it; a primary point sets O. A secondary plane
sets e1 = -n made perpendicular to e3 and moves O onto it; a secondary axis parallel to e3 puts O on it (in the
primary plane); a secondary plane perpendicular to a primary axis puts O at their intersection. A tertiary plane
sets e2 (or clocks e1 when the secondary was an axis) and moves O onto it; a tertiary axis clocks e1 toward it.
``e2 = e3 x e1`` unless a tertiary plane's inward normal points the other way (then the frame is left-handed, so
basic dimensions are measured into the material from every datum). Degrees of freedom no datum locks stay
**free**: characteristics minimise over them (ASME Y14.5-2018 7.11: the zone may move in unconstrained degrees of
freedom).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geom import transform as tf
from . import fitting as F
from .cadref import CadReference
from .schema import InspectSettings
from .specs import DatumSpec


class DatumError(F.FitError):
    """A datum feature that cannot be simulated or a datum system that cannot be built."""


@dataclass
class DatumResult:
    label: str
    faces: list
    kind: str                       # plane | axis | point | axis_point
    point: np.ndarray               # a point of the plane, of the axis, or the point
    direction: np.ndarray | None    # outward normal of a plane, axis direction (canonical sign)
    simulator: str
    points: int = 0
    internal: bool = False
    note: str = ""

    def to_json(self) -> dict:
        return {"label": self.label, "faces": [int(f) for f in self.faces], "kind": self.kind,
                "point_S_m": [float(v) for v in self.point],
                "direction_S": None if self.direction is None else [float(v) for v in self.direction],
                "simulator": self.simulator, "points": int(self.points), "note": self.note}


@dataclass
class Frame:
    labels: tuple
    origin: np.ndarray
    axes: np.ndarray                                  # columns e1, e2, e3 (S)
    free_translation: np.ndarray = field(default_factory=lambda: np.ones(3, bool))
    free_rotation: np.ndarray = field(default_factory=lambda: np.ones(3, bool))
    datums: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_local(self, P) -> np.ndarray:
        return (np.asarray(P, dtype=np.float64) - self.origin) @ self.axes

    def to_S(self, local) -> np.ndarray:
        return self.origin + np.asarray(local, dtype=np.float64) @ self.axes.T

    def to_json(self) -> dict:
        return {"datums": list(self.labels), "origin_S_m": [float(v) for v in self.origin],
                "axes_S": [[float(v) for v in self.axes[:, k]] for k in range(3)],
                "free_translation": [bool(v) for v in self.free_translation],
                "free_rotation": [bool(v) for v in self.free_rotation], "notes": list(self.notes),
                "features": [d.to_json() for d in self.datums]}


IDENTITY_FRAME = Frame((), np.zeros(3), np.eye(3))


def _canonical(u) -> np.ndarray:
    u = tf.unit(u)
    k = int(np.argmax(np.abs(u)))
    return u if u[k] > 0 else -u


def _proj_perp(v, a) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v - (v @ a) * a


# ============================================================================ simulators

def nominal_datum(cad: CadReference, spec: DatumSpec) -> DatumResult:
    faces = cad.select(spec.feature)
    if not faces:
        raise DatumError(f"datum {spec.label}: the feature selector matches no face ({cad.select_reason(spec.feature)})")
    f = cad.face(faces[0])
    p = f.params
    if f.surface == "plane":
        return DatumResult(spec.label, faces, "plane", np.asarray(p["point_m"], float), tf.unit(p["normal"]), "CAD")
    if f.surface == "cylinder":
        return DatumResult(spec.label, faces, "axis", np.asarray(p["axis_point_m"], float), _canonical(p["axis"]),
                           "CAD", internal=bool(f.concave))
    if f.surface == "sphere":
        return DatumResult(spec.label, faces, "point", np.asarray(p["centre_m"], float), None, "CAD",
                           internal=bool(f.concave))
    if f.surface == "cone":
        return DatumResult(spec.label, faces, "axis_point", np.asarray(p["apex_m"], float), _canonical(p["axis"]),
                           "CAD", internal=bool(f.concave))
    raise DatumError(f"datum {spec.label}: a {f.surface} face is not a datum feature this inspection simulates "
                     f"(plane, cylinder, sphere, cone)")


def measured_datum(nominal: DatumResult, P: np.ndarray, settings: InspectSettings, fixed_axis=None,
                   fixed_normal=None) -> DatumResult:
    """Simulator of a datum feature on its result points. ``fixed_axis``: the higher datums leave a plane one
    rotation about this direction (its normal stays perpendicular to it) and hold a cylinder's axis parallel to
    it; ``fixed_normal``: the higher datums fix a plane's orientation completely (it only translates)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if not len(P):
        raise DatumError(f"datum {nominal.label}: no result points on the datum feature")
    if nominal.kind == "plane":
        if fixed_normal is not None:
            n = tf.unit(fixed_normal)
            top = float(np.max(P @ n))
            fit = F.Fit("plane", "tangent", {"point_m": top * n, "normal": n}, P @ n - top)
            sim = "outermost point, orientation fixed by the higher datums"
        elif fixed_axis is not None:
            fit = F.tangent_plane(P, nominal.direction, settings, fixed_axis=fixed_axis)
            sim = "tangent plane, one rotation left by the higher datums"
        else:
            fit = F.tangent_plane(P, nominal.direction, settings)
            sim = "tangent plane (constrained minimax)"
        return DatumResult(nominal.label, nominal.faces, "plane", np.asarray(fit.params["point_m"]),
                           tf.unit(fit.params["normal"]), sim, len(P))
    if nominal.kind == "axis":
        mode = "MIC" if nominal.internal else "MCC"
        fixed = fixed_axis is not None and abs(float(nominal.direction @ tf.unit(fixed_axis))) > 1 - 1e-6
        direction = (tf.unit(fixed_axis) if float(nominal.direction @ fixed_axis) > 0 else -tf.unit(fixed_axis)) \
            if fixed else nominal.direction
        try:
            fit = F.zone_cylinder(P, nominal.point, direction, mode, settings, fixed_direction=fixed)
            note = fit.note
            if not fit.converged and fit.note:
                raise F.FitError(fit.note)
            sim = ("maximum inscribed" if nominal.internal else "minimum circumscribed") + " cylinder" + \
                  (" parallel to the primary direction" if fixed else "")
        except F.FitError as error:
            fit = F.fit_cylinder(P, nominal.point, nominal.direction, float(np.mean(
                F.cylinder_distance(P, nominal.point, tf.unit(nominal.direction)))), settings)
            sim, note = "least-squares cylinder", f"envelope not available ({error}): least squares used"
        return DatumResult(nominal.label, nominal.faces, "axis", np.asarray(fit.params["axis_point_m"]),
                           _canonical(fit.params["axis"]), sim, len(P), nominal.internal, note)
    if nominal.kind == "point":
        fit = F.zone_sphere(P, nominal.point, "MIC" if nominal.internal else "MCC", settings)
        return DatumResult(nominal.label, nominal.faces, "point", np.asarray(fit.params["centre_m"]), None,
                           "maximum inscribed sphere" if nominal.internal else "minimum circumscribed sphere", len(P),
                           nominal.internal, fit.note)
    fit = F.fit_cone(P, nominal.point, nominal.direction, _cone_angle_guess(nominal, P), settings)
    return DatumResult(nominal.label, nominal.faces, "axis_point", np.asarray(fit.params["apex_m"]),
                       _canonical(fit.params["axis"]), "least-squares cone (axis and apex)", len(P), nominal.internal)


def constraints_for(nominal: DatumResult, partial: Frame | None) -> tuple:
    """(fixed_axis, fixed_normal) the measured higher datums impose on the next datum feature."""
    if partial is None or not partial.labels:
        return None, None
    e3 = partial.axes[:, 2]
    if partial.free_rotation[0] or partial.free_rotation[1]:      # only a point so far
        return None, None
    if nominal.kind == "plane":
        n = tf.unit(nominal.direction)
        if abs(float(n @ e3)) > 1.0 - 1e-6:
            return None, (e3 if float(n @ e3) > 0 else -e3)
        if partial.free_rotation[2]:
            return e3, None
        n_fixed = _proj_perp(_proj_perp(n, e3), partial.axes[:, 0])
        if float(np.linalg.norm(n_fixed)) < 1e-9:
            n_fixed = _proj_perp(n, e3)
        return None, tf.unit(n_fixed)
    return e3, None


def _cone_angle_guess(nominal: DatumResult, P) -> float:
    w = P - nominal.point
    z = w @ nominal.direction
    rho = np.linalg.norm(w - np.outer(z, nominal.direction), axis=1)
    ok = np.abs(z) > 0
    return float(np.median(np.arctan2(rho[ok], np.abs(z[ok])))) if ok.any() else 0.1


# ============================================================================ frames

def build_frame(datums: list[DatumResult], reference_axes: np.ndarray | None = None) -> Frame:
    """The frame of datum results in precedence order (module documentation). ``reference_axes`` (3x3) orient
    what the datums leave free (the nominal frame's axes; S's otherwise)."""
    ref = np.eye(3) if reference_axes is None else np.asarray(reference_axes, dtype=np.float64)
    origin = np.zeros(3)
    e1 = e2 = e3 = None
    free_t = np.ones(3, bool)
    free_r = np.ones(3, bool)
    handed = 1.0
    notes = []
    for order, d in enumerate(datums):
        if order == 0:
            origin = d.point.copy()
            if d.kind == "plane":
                e3 = -tf.unit(d.direction)
                free_t[2] = False
                free_r[:2] = False
            elif d.kind in ("axis", "axis_point"):
                e3 = tf.unit(d.direction)
                free_t[:2] = False
                free_r[:2] = False
                free_t[2] = d.kind != "axis_point"
            else:
                free_t[:] = False
            continue
        if e3 is None:                                       # the primary was a point: this datum orients e3
            if d.kind == "plane":
                e3 = -tf.unit(d.direction)
            elif d.direction is not None:
                e3 = tf.unit(d.direction)
            else:
                notes.append(f"datum {d.label}: a second point does not orient the frame")
                continue
            free_r[:2] = False
            continue
        if d.kind == "plane":
            n = tf.unit(d.direction)
            if abs(float(n @ e3)) > 1.0 - 1e-6:              # a face perpendicular to the primary axis
                if free_t[2]:
                    origin = origin + float((d.point - origin) @ e3) * e3
                    free_t[2] = False
                else:
                    notes.append(f"datum {d.label}: its translation along the primary direction is already locked")
                continue
            inward = -tf.unit(_proj_perp(n, e3))
            if free_r[2]:
                if free_t[0] or free_t[1]:
                    e1 = inward
                    origin = origin + float((d.point - origin) @ e1) * e1
                    free_t[0] = False
                else:                                        # the position is located: this plane only clocks
                    e2 = inward
                    e1 = np.cross(e2, e3)
                free_r[2] = False
            elif free_t[1]:
                e2 = tf.unit(_proj_perp(inward, e1))
                origin = origin + float((d.point - origin) @ e2) * e2
                free_t[1] = False
                if float(np.cross(e3, e1) @ e2) < 0.0:
                    handed = -1.0
            else:
                notes.append(f"datum {d.label}: every degree of freedom it could lock is already locked")
            continue
        q = d.point - float((d.point - origin) @ e3) * e3    # trace of the axis / point in the primary plane
        if d.direction is not None and abs(float(tf.unit(d.direction) @ e3)) < 1.0 - 1e-6:
            notes.append(f"datum {d.label}: its axis is not parallel to the primary direction; its trace at the "
                         f"origin's level is used")
        if free_t[0] and free_t[1]:
            origin = q
            free_t[:2] = False
        elif free_r[2]:
            v = _proj_perp(q - origin, e3)
            if float(np.linalg.norm(v)) <= 1e-12:
                raise DatumError(f"datum {d.label} lies on the frame's axis: it cannot clock the frame")
            e1 = tf.unit(v)
            free_r[2] = False
        else:
            notes.append(f"datum {d.label}: every degree of freedom it could lock is already locked")
    if e3 is None:
        e3 = ref[:, 2]
    if e1 is None:
        cand = _proj_perp(ref[:, 0], e3)
        if np.linalg.norm(cand) < 1e-9:
            cand = _proj_perp(ref[:, 1], e3)
        e1 = tf.unit(cand)
    e1 = tf.unit(_proj_perp(e1, e3))
    e2 = handed * np.cross(e3, e1)
    return Frame(tuple(d.label for d in datums), np.asarray(origin, dtype=np.float64),
                 np.column_stack([e1, e2, e3]), free_t, free_r, list(datums), notes)


def registration(nominal: Frame, measured: Frame) -> np.ndarray:
    """4x4 that maps result points (S) into the nominal frame: p_nom = O_n + A_n A_m^T (p - O_m)."""
    R = nominal.axes @ measured.axes.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = nominal.origin - R @ measured.origin
    return T


def free_columns(frame: Frame, P: np.ndarray, N: np.ndarray, translations: bool = True,
                 rotations: bool = True) -> np.ndarray:
    """d(normal deviation)/d(free small displacement) per point [n, k] (small-displacement torsor: a translation
    t along e_j changes n.p by n.e_j, a rotation w about e_j through O by ((p - O) x n).e_j)."""
    cols = []
    for j in range(3):
        if translations and frame.free_translation[j]:
            cols.append(N @ frame.axes[:, j])
    for j in range(3):
        if rotations and frame.free_rotation[j]:
            cols.append(np.cross(P - frame.origin, N) @ frame.axes[:, j])
    return np.column_stack(cols) if cols else np.zeros((len(P), 0))


def free_motion(frame: Frame, xi: np.ndarray, translations: bool = True, rotations: bool = True) -> np.ndarray:
    """4x4 rigid motion of the free displacement xi (order of :func:`free_columns`), exact rotation."""
    k = 0
    t = np.zeros(3)
    w = np.zeros(3)
    for j in range(3):
        if translations and frame.free_translation[j]:
            t += xi[k] * frame.axes[:, j]
            k += 1
    for j in range(3):
        if rotations and frame.free_rotation[j]:
            w += xi[k] * frame.axes[:, j]
            k += 1
    angle = float(np.linalg.norm(w))
    R = tf.axis_angle(w / angle, angle) if angle > 0 else np.eye(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = frame.origin - R @ frame.origin + t
    return T


def drf_pair(cad: CadReference, specs: dict, labels: list, points_of, settings: InspectSettings):
    """(nominal frame, measured frame, registration) of a datum system; ``points_of(faces) -> [n, 3]``."""
    nominals, measured = [], []
    for label in labels:
        spec = specs.get(label)
        if spec is None:
            raise DatumError(f"datum {label} is not defined (cad.datums or the PMI)")
        nominals.append(nominal_datum(cad, spec))
    nominal_frame = build_frame(nominals)
    partial = None
    for nom in nominals:
        fixed_axis, fixed_normal = constraints_for(nom, partial)
        if fixed_normal is not None and nom.direction is not None and float(fixed_normal @ nom.direction) < 0:
            fixed_normal = -fixed_normal
        meas = measured_datum(nom, points_of(nom.faces), settings, fixed_axis, fixed_normal)
        measured.append(meas)
        partial = build_frame(_aligned(measured, nominals[:len(measured)]), nominal_frame.axes)
    measured_frame = partial
    return nominal_frame, measured_frame, registration(nominal_frame, measured_frame)


def _aligned(measured: list[DatumResult], nominals: list[DatumResult]) -> list[DatumResult]:
    """Measured axes with the sign of the nominal ones (the frame rules use the sign of axis datums)."""
    out = []
    for m, n in zip(measured, nominals):
        if m.direction is not None and n.direction is not None and m.kind != "plane" and m.direction @ n.direction < 0:
            m = DatumResult(m.label, m.faces, m.kind, m.point, -m.direction, m.simulator, m.points, m.internal, m.note)
        out.append(m)
    return out


__all__ = ["DatumResult", "Frame", "DatumError", "nominal_datum", "measured_datum", "build_frame", "registration",
           "free_columns", "free_motion", "drf_pair", "IDENTITY_FRAME"]
