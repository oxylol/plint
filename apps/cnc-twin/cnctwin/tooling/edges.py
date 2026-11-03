"""Edge elements from a profile and its teeth: the generic part of every EdgeModel builder.

A builder (mills, holemaking, indexable, turning) describes a tool by

* a **layout**: the profile pieces (:class:`cnctwin.tooling.profile.Seg`) every tooth follows in
  the meridian plane, each with its edge kind, angular-placement rule and face model;
* its **teeth** (:class:`ToothDef`): cutting-sense phase, lag parameters (helix per flute, web
  offset ...), which layout elements the tooth has, and optional per-tooth profile changes
  (knuckle waves, tap lead shift) or a 3-D curve (inserts).

:func:`assemble` discretises the layout once by arc length (shared stations: element k of every
tooth sits at the same place on the layout, so teeth sweep the same circle there), evaluates
every element of every tooth exactly (sub-sampled curve), and derives kappa, chip width db
(meridian arc length), true edge length, inclination (ISO 3002-1 lambda_s), normal rake and
normal clearance from the face model, then groups the elements into coefficient groups and
packs the :class:`cnctwin.core.abi.EdgeModel`.

Conventions (docs/models/tooling.md): the cutting-sense frame C has x = r cos psi, y = r sin psi,
z; the cutting velocity of an element is +e_psi; T is C mirrored in y for a right-hand tool
(h = -1). The material lies on the +n side, n = sin(kappa) e_r - cos(kappa) e_z; the chip side of
the rake face is -n.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EdgeModel
from .abi import EdgeFrameT, EdgeTrig, GroupTable
from .profile import Seg
from .schema import ToolingSettings

SUBSAMPLES = 16             # chords per element for lengths and tangents (even): chord error < (dtheta/16)^2/24
SIZING_SAMPLES = 256        # samples per piece for the discretisation cost


# --------------------------------------------------------------------------- face models
@dataclass
class Face:
    """Rake and flank of a profile piece.

    mode ``radial``: rake and clearance measured in the local radial plane (perpendicular to the
    meridian tangent; for a peripheral edge the transverse plane of an end mill), converted with
    the local inclination (tan g_n = tan g_r cos l, tan a_n = tan a_r / cos l).
    mode ``normal``: normal rake and clearance given directly.
    mode ``helicoid``: the rake face is the helical flute of lead 2 pi R0 / tan(beta0) (drills);
    clearance as in ``radial``.
    mode ``insert``: an insert in its seat. ``N_rake`` is the seat's rake plane (tooth frame C);
    ``rake`` is the chip-breaker rake added normal to every edge, ``clear`` the insert's own normal
    clearance alpha_i (ISO 1832); the wedge is constant around the insert, so an element whose seat
    rake is g_s gets rake g_s + rake and clearance alpha_i - g_s. With ``lift`` the edge tangent is
    taken in the rake plane (non-rotating tools keep their elements in the reference plane, the
    inclination then comes from the plane).
    """

    mode: str = "radial"
    rake: float = 0.0
    clear: float = 0.0
    tan_beta0: float = 0.0
    R0: float = 1.0
    N_rake: tuple | None = None
    edge_radius: float = 0.0
    chamfer_width: float = 0.0
    chamfer_angle: float = 0.0
    forming: bool = False
    extra_incl: float = 0.0          # added to the geometric inclination (spiral-point taps)
    db_override: float = 0.0         # margins and pads: contact land width instead of the chip width
    lift: bool = False               # mode insert: edge tangent in the rake plane


# --------------------------------------------------------------------------- teeth
@dataclass
class ToothDef:
    """One tooth (flute, lip, insert, land, pad).

    ``params`` holds the numbers the placement rules use: ``tan_beta`` and ``R_helix`` (helix,
    constant lead), ``w`` and ``R_lip`` (drill lip web offset), ``z_margin`` (start of the margin
    helix), ``psi`` (constant). ``mask(seg_index, seg, z, r)`` says whether the tooth has the
    element whose midpoint (on this tooth's own profile) is (z, r). ``shift(seg, z, r) -> (z, r)``
    changes the tooth's own profile (knuckle wave, thread lead, chamfer truncation).
    ``curve(seg, u) -> (z, r, psi) | None`` replaces the profile and placement entirely for pieces
    it handles (3-D insert edges). ``kind_fn`` and ``face_fn`` ``(seg, z, r)`` override the
    piece's edge kind and face for this tooth's element at (z, r) (tap chamfer teeth).
    """

    phase: float
    params: dict = field(default_factory=dict)
    mask: Callable | None = None
    shift: Callable | None = None
    curve: Callable | None = None
    faces: dict = field(default_factory=dict)     # seg.tag -> Face overriding the piece's face
    kind_fn: Callable | None = None
    face_fn: Callable | None = None


def _lag(tooth: ToothDef, seg: Seg, z: np.ndarray, r: np.ndarray) -> np.ndarray:
    rule = seg.rule
    prm = tooth.params
    if rule == "helix":
        z0 = seg.extra.get("z0", prm.get("helix_z0", 0.0))
        return -(z - z0) * prm.get("tan_beta", 0.0) / prm.get("R_helix", 1.0) + seg.extra.get("psi0", 0.0)
    if rule == "radial":
        return np.zeros_like(z)
    if rule == "const":
        return np.full_like(z, seg.extra.get("psi", prm.get("psi", 0.0)))
    if rule == "lip":
        w = prm["w"]
        ratio = np.clip(w / np.maximum(r, w), -1.0, 1.0)
        return np.arcsin(ratio) - math.asin(min(w / prm["R_lip"], 1.0))
    if rule == "chisel":
        w = prm["w"]
        return np.full_like(z, math.asin(min(w / prm["r_chisel"], 1.0)) - math.asin(min(w / prm["R_lip"], 1.0)))
    if rule == "margin":
        z0 = prm.get("z_margin", 0.0)
        return -(z - z0) * prm.get("tan_beta", 0.0) / prm.get("R_helix", 1.0)
    raise ValueError(f"unknown placement rule {rule!r}")


def tooth_eval(tooth: ToothDef, seg: Seg, u) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z, r, psi) of the tooth's edge on piece ``seg`` at parameters ``u`` (psi relative to the
    tooth phase, cutting sense)."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if tooth.curve is not None:
        result = tooth.curve(seg, u)
        if result is not None:
            return result
    z, r = seg.point(u)
    z = np.atleast_1d(np.asarray(z, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if tooth.shift is not None:
        z, r = tooth.shift(seg, z, r)
    return z, r, _lag(tooth, seg, z, r)


# --------------------------------------------------------------------------- discretisation
@dataclass
class Layout:
    segs: list[Seg]
    seg_index: np.ndarray      # [ne] int
    u0: np.ndarray             # [ne]
    u1: np.ndarray             # [ne]


def discretise(segs: list[Seg], teeth: list[ToothDef], settings: ToolingSettings) -> Layout:
    """Element boundaries per piece, shared by all teeth.

    Each piece gets n elements with n = ceil(max cost), the cost density being the largest of
    meridian length / (edge_element x scale), true edge length / same (worst tooth), |d kappa| /
    max_kappa_step and |d psi| / max_lag_step (worst tooth); boundaries are equidistant in that
    cost, so every element satisfies every limit.
    """
    seg_index, u0s, u1s = [], [], []
    for index, seg in enumerate(segs):
        u = np.linspace(0.0, 1.0, SIZING_SAMPLES + 1)
        limit_len = settings.edge_element_m * seg.element_scale
        zs, rs = seg.point(u)
        cost = np.hypot(np.diff(zs), np.diff(rs)) / limit_len
        kap = np.unwrap(np.arctan2(*seg.deriv(u)))
        # a corner (kink) cannot be resolved by subdivision: it adds at most one element boundary
        cost = np.maximum(cost, np.minimum(np.abs(np.diff(kap)) / settings.max_kappa_step_rad, 1.0))
        for tooth in teeth:
            z, r, psi = tooth_eval(tooth, seg, u)
            x, y = r * np.cos(psi), r * np.sin(psi)
            d3 = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
            cost = np.maximum(cost, d3 / limit_len)
            cost = np.maximum(cost, np.abs(np.diff(psi)) / settings.max_lag_step_rad)
            kap_t = np.unwrap(np.arctan2(np.diff(z), np.diff(r)))
            if len(kap_t) > 1:
                dk = np.abs(np.diff(kap_t))
                cost[1:] = np.maximum(cost[1:], np.minimum(dk / settings.max_kappa_step_rad, 1.0))
        total = float(cost.sum())
        n = max(int(math.ceil(total - 1e-9)), seg.min_elements, 1)
        cumulative = np.concatenate([[0.0], np.cumsum(cost)])
        targets = np.linspace(0.0, cumulative[-1], n + 1)
        bounds = np.interp(targets, cumulative, u) if cumulative[-1] > 0 else np.linspace(0.0, 1.0, n + 1)
        bounds[0], bounds[-1] = 0.0, 1.0
        for k in range(n):
            seg_index.append(index)
            u0s.append(bounds[k])
            u1s.append(bounds[k + 1])
    if len(seg_index) > settings.max_elements_per_tooth:
        raise ValueError(f"{len(seg_index)} edge elements per tooth exceed physics.tooling.max_elements_per_tooth "
                         f"({settings.max_elements_per_tooth}); raise physics.edge_element_mm")
    return Layout(segs, np.asarray(seg_index, np.int64), np.asarray(u0s), np.asarray(u1s))


# --------------------------------------------------------------------------- element geometry
@dataclass
class ElementArrays:
    """Everything computed per (tooth, element) in the cutting-sense frame C."""

    z: np.ndarray
    r: np.ndarray
    lag: np.ndarray
    length: np.ndarray
    db: np.ndarray
    kappa: np.ndarray
    rake: np.ndarray
    incl: np.ndarray
    clear: np.ndarray
    edge_radius: np.ndarray
    kind: np.ndarray
    valid: np.ndarray
    tangent_C: np.ndarray        # [nt, ne, 3]
    normal_C: np.ndarray
    cutdir_C: np.ndarray
    rake_normal_C: np.ndarray
    chamfer_width: np.ndarray
    chamfer_angle: np.ndarray
    forming: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0.0 else v


def face_angles(face: Face, t3: np.ndarray, n: np.ndarray, e_v: np.ndarray,
                point: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(normal rake, normal clearance, rake-face normal) of an element from its face model.

    t3: unit edge tangent, n: chip-thickness normal into the material, e_v: unit cutting
    velocity, point: element midpoint (C frame). The rake face trace in the normal plane P_n is
    u = cos(g) (-n) - sin(g) m, the flank trace -cos(a) m - sin(a) n, with m the unit cutting
    direction in P_n (ISO 3002-1).
    """
    m = e_v - float(np.dot(e_v, t3)) * t3 - float(np.dot(e_v, n)) * n
    m = _unit(m)
    if face.mode == "normal":
        return face.rake, face.clear, _rake_normal_from_angle(face.rake, n, m)
    if face.mode == "insert":
        seat = _rake_from_normal(np.asarray(face.N_rake, dtype=float), t3, n, m)
        rake = seat + face.rake
        return rake, face.clear - seat, _rake_normal_from_angle(rake, n, m)
    if face.mode not in ("radial", "helicoid"):
        raise ValueError(f"unknown face mode {face.mode!r}")
    if face.mode == "radial":
        u_r = math.cos(face.rake) * (-n) - math.sin(face.rake) * _unit(e_v - float(np.dot(e_v, n)) * n)
        N_r = np.cross(t3, u_r)
    else:
        N_r = _helicoid_normal(point, t3, face.tan_beta0, face.R0)
    u_f = -math.cos(face.clear) * _unit(e_v - float(np.dot(e_v, n)) * n) - math.sin(face.clear) * n
    N_f = np.cross(t3, u_f)
    rake = _rake_from_normal(N_r, t3, n, m)
    clear = _clear_from_normal(N_f, t3, n, m)
    return rake, clear, _rake_normal_from_angle(rake, n, m)


def lift_tangent(t3: np.ndarray, e_v: np.ndarray, N_r: np.ndarray) -> np.ndarray:
    """The edge direction in the rake plane whose projection on the reference plane (normal to
    the cutting velocity) is ``t3``: t3 + c e_v with (t3 + c e_v) . N_r = 0."""
    denom = float(np.dot(e_v, N_r))
    if abs(denom) < 1e-12:
        return t3
    lifted = t3 - (float(np.dot(t3, N_r)) / denom) * e_v
    return _unit(lifted)


def _rake_normal_from_angle(rake: float, n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Outward rake-face normal in P_n for normal rake ``rake``: perpendicular to the face trace
    cos(g)(-n) - sin(g) m, pointing toward the oncoming chip (+m side)."""
    return _unit(math.cos(rake) * m - math.sin(rake) * n)


def _rake_from_normal(N_r, t3, n, m) -> float:
    trace = np.cross(N_r, t3)
    if np.linalg.norm(trace) == 0.0:
        return 0.0
    trace = _unit(trace)
    if float(np.dot(trace, -n)) < 0.0:
        trace = -trace
    return math.atan2(-float(np.dot(trace, m)), float(np.dot(trace, -n)))


def _clear_from_normal(N_f, t3, n, m) -> float:
    trace = np.cross(N_f, t3)
    if np.linalg.norm(trace) == 0.0:
        return 0.0
    trace = _unit(trace)
    if float(np.dot(trace, -m)) < 0.0:
        trace = -trace
    return math.atan2(float(np.dot(trace, -n)), float(np.dot(trace, -m)))


def _helicoid_normal(point: np.ndarray, t3: np.ndarray, tan_beta0: float, R0: float) -> np.ndarray:
    """Normal of the helical flute surface through the lip at ``point``: the surface contains the
    lip tangent and the flute helix of constant lead (tan beta(r) = r tan(beta0) / R0); going up
    the flute (+z) the helix trails (-e_psi), as the helix of an end mill does."""
    x, y = float(point[0]), float(point[1])
    r = math.hypot(x, y)
    if r == 0.0:
        return np.array([0.0, 0.0, 1.0])
    e_r = np.array([x / r, y / r, 0.0])
    e_psi = np.array([-e_r[1], e_r[0], 0.0])
    beta = math.atan(r * tan_beta0 / R0)
    helix = -math.sin(beta) * e_psi + math.cos(beta) * np.array([0.0, 0.0, 1.0])
    return np.cross(t3, helix)


def evaluate(layout: Layout, teeth: list[ToothDef]) -> ElementArrays:
    nt, ne = len(teeth), len(layout.seg_index)
    shape = (nt, ne)
    out = {name: np.zeros(shape) for name in ("z", "r", "lag", "length", "db", "kappa", "rake", "incl", "clear",
                                              "edge_radius", "chamfer_width", "chamfer_angle")}
    kind = np.zeros(shape, np.int32)
    valid = np.zeros(shape, np.int8)
    forming = np.zeros(shape, np.int8)
    frames = {name: np.zeros((nt, ne, 3)) for name in ("tangent_C", "normal_C", "cutdir_C", "rake_normal_C")}
    q = np.linspace(0.0, 1.0, SUBSAMPLES + 1)
    mid = SUBSAMPLES // 2
    for k in range(ne):
        seg = layout.segs[int(layout.seg_index[k])]
        u = layout.u0[k] + (layout.u1[k] - layout.u0[k]) * q
        for j, tooth in enumerate(teeth):
            z, r, psi = tooth_eval(tooth, seg, u)
            zm, rm = float(z[mid]), float(r[mid])
            has = True if tooth.mask is None else bool(tooth.mask(int(layout.seg_index[k]), seg, zm, rm))
            own_kind = tooth.kind_fn(seg, zm, rm) if tooth.kind_fn is not None else None
            kind[j, k] = own_kind if own_kind is not None else (seg.kind if seg.kind is not None else 0)
            pts = np.stack([r * np.cos(psi), r * np.sin(psi), z], axis=1)
            if seg.shape == "arc" and tooth.shift is None and tooth.curve is None:
                meridian = seg.rho * abs(seg.a1 - seg.a0) * (layout.u1[k] - layout.u0[k])     # exact
            else:
                meridian = float(np.sum(np.hypot(np.diff(z), np.diff(r))))
            length3 = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            pm = float(psi[mid])
            dz, dr = float(z[mid + 1] - z[mid - 1]), float(r[mid + 1] - r[mid - 1])
            if dz == 0.0 and dr == 0.0:
                dz, dr = float(z[-1] - z[0]), float(r[-1] - r[0])
            kappa = math.atan2(dz, dr)
            t3 = _unit(pts[mid + 1] - pts[mid - 1])
            if not np.any(t3):
                t3 = _unit(pts[-1] - pts[0])
            e_r = np.array([math.cos(pm), math.sin(pm), 0.0])
            e_v = np.array([-math.sin(pm), math.cos(pm), 0.0])
            e_z = np.array([0.0, 0.0, 1.0])
            n = math.sin(kappa) * e_r - math.cos(kappa) * e_z
            face = tooth.face_fn(seg, zm, rm) if tooth.face_fn is not None else None
            if face is None:
                face = tooth.faces.get(seg.tag, seg.face) or Face()
            if face.mode == "insert" and face.lift:
                t3 = lift_tangent(t3, e_v, np.asarray(face.N_rake, dtype=float))
            incl = math.asin(max(-1.0, min(1.0, -float(np.dot(t3, e_v))))) + face.extra_incl
            rake, clear, n_rake = face_angles(face, t3, n, e_v, pts[mid])
            out["z"][j, k], out["r"][j, k], out["lag"][j, k] = zm, rm, pm
            out["length"][j, k] = length3
            out["db"][j, k] = face.db_override if face.db_override > 0.0 else meridian
            out["kappa"][j, k] = kappa
            out["rake"][j, k], out["clear"][j, k], out["incl"][j, k] = rake, clear, incl
            out["edge_radius"][j, k] = face.edge_radius
            out["chamfer_width"][j, k], out["chamfer_angle"][j, k] = face.chamfer_width, face.chamfer_angle
            forming[j, k] = 1 if face.forming else 0
            valid[j, k] = 1 if (has and length3 > 0.0) else 0
            frames["tangent_C"][j, k] = t3
            frames["normal_C"][j, k] = n
            frames["cutdir_C"][j, k] = e_v
            frames["rake_normal_C"][j, k] = n_rake
    return ElementArrays(kind=kind, valid=valid, forming=forming, **out, **frames)


# --------------------------------------------------------------------------- groups
def assign_groups(el: ElementArrays, settings: ToolingSettings) -> tuple[np.ndarray, GroupTable]:
    """Coefficient group of every element: same edge kind, rake class, inclination class, edge
    preparation and forming flag. Groups are numbered in sorted key order (deterministic)."""
    nt, ne = el.kind.shape
    keys = {}
    elem_keys = np.empty((nt, ne), dtype=object)
    for j in range(nt):
        for k in range(ne):
            key = (int(el.kind[j, k]), int(round(el.rake[j, k] / settings.rake_class_rad)),
                   int(round(el.incl[j, k] / settings.incl_class_rad)),
                   int(round(el.chamfer_width[j, k] * 1e7)), int(round(el.chamfer_angle[j, k] * 1e4)),
                   int(el.forming[j, k]))
            elem_keys[j, k] = key
            if el.valid[j, k]:
                keys.setdefault(key, []).append((j, k))
    ordered = sorted(keys)
    index = {key: i for i, key in enumerate(ordered)}
    group = np.zeros((nt, ne), np.int32)
    for j in range(nt):
        for k in range(ne):
            key = elem_keys[j, k]
            group[j, k] = index.get(key, _nearest_group(key, ordered))
    ng = max(len(ordered), 1)
    table = {name: np.zeros(ng) for name in ("rake_rad", "incl_rad", "clear_rad", "edge_radius_m",
                                             "chamfer_width_m", "chamfer_angle_rad")}
    kinds = np.zeros(ng, np.int32)
    forming = np.zeros(ng, np.int8)
    count = np.zeros(ng, np.int32)
    for key, members in keys.items():
        g = index[key]
        js = np.array([m[0] for m in members])
        ks = np.array([m[1] for m in members])
        kinds[g] = key[0]
        forming[g] = key[5]
        count[g] = len(members)
        table["rake_rad"][g] = float(np.mean(el.rake[js, ks]))
        table["incl_rad"][g] = float(np.mean(el.incl[js, ks]))
        table["clear_rad"][g] = float(np.mean(el.clear[js, ks]))
        table["edge_radius_m"][g] = float(np.mean(el.edge_radius[js, ks]))
        table["chamfer_width_m"][g] = float(np.mean(el.chamfer_width[js, ks]))
        table["chamfer_angle_rad"][g] = float(np.mean(el.chamfer_angle[js, ks]))
    return group, GroupTable(kind=kinds, forming=forming, count=count, **table)


def _nearest_group(key, ordered) -> int:
    """Invalid elements (no member of their own key) borrow the closest group of the same kind."""
    if not ordered:
        return 0
    best, best_score = 0, None
    for i, other in enumerate(ordered):
        score = (0 if other[0] == key[0] else 1, abs(other[1] - key[1]) + abs(other[2] - key[2]))
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


# --------------------------------------------------------------------------- packing
@dataclass
class EdgeBuild:
    """Result of :func:`assemble`: the EdgeModel plus the tables and frames other code needs."""

    edge: EdgeModel
    groups: GroupTable
    trig: EdgeTrig
    frames: EdgeFrameT
    layout: Layout
    elements: ElementArrays


def assemble(segs: list[Seg], teeth: list[ToothDef], settings: ToolingSettings, hand_sign: float) -> EdgeBuild:
    if not segs or not teeth:
        raise ValueError("a tool needs at least one cutting piece and one tooth")
    layout = discretise(segs, teeth, settings)
    el = evaluate(layout, teeth)
    group, groups = assign_groups(el, settings)
    nt, ne = el.kind.shape
    station = np.tile(np.arange(ne, dtype=np.int32), (nt, 1))
    st_z = np.zeros(ne)
    st_r = np.zeros(ne)
    for k in range(ne):
        seg = layout.segs[int(layout.seg_index[k])]
        zk, rk = seg.point(layout.u0[k] + 0.5 * (layout.u1[k] - layout.u0[k]))
        st_z[k], st_r[k] = float(zk), float(rk)
    phases = np.array([_wrap_2pi(t.phase) for t in teeth])
    h = float(hand_sign)
    edge = EdgeModel(
        tooth_phase_rad=phases, elem_z_m=el.z.copy(), elem_r_m=el.r.copy(), elem_lag_rad=el.lag.copy(),
        elem_len_m=el.length.copy(), elem_db_m=el.db.copy(), elem_kappa_rad=el.kappa.copy(),
        elem_rake_rad=el.rake.copy(), elem_incl_rad=el.incl.copy(), elem_clear_rad=el.clear.copy(),
        elem_edge_radius_m=el.edge_radius.copy(), elem_kind=el.kind.copy(), elem_group=group,
        elem_station=station, elem_valid=el.valid.copy(), runout_T_m=np.zeros(3), tilt_rad=np.zeros(2),
        station_z_m=st_z, station_r_m=st_r, hand_sign=np.array([h]))
    trig = edge_trig(edge)
    frames = frames_T(el, phases, h)
    return EdgeBuild(edge, groups, trig, frames, layout, el)


def _wrap_2pi(angle: float) -> float:
    value = math.fmod(angle, 2.0 * math.pi) + 0.0
    if value < 0.0:
        value += 2.0 * math.pi
    if value >= 2.0 * math.pi:
        value -= 2.0 * math.pi
    return value


def edge_trig(edge: EdgeModel) -> EdgeTrig:
    """cos/sin of every element's right-handed angle at spindle angle 0: alpha = h (phase + lag)."""
    h = float(edge.hand_sign[0])
    psi = edge.tooth_phase_rad[:, None] + edge.elem_lag_rad
    return EdgeTrig(np.ascontiguousarray(np.cos(psi)), np.ascontiguousarray(h * np.sin(psi)))


def frames_T(el: ElementArrays, phases: np.ndarray, hand_sign: float) -> EdgeFrameT:
    """Element unit vectors in T at spindle angle 0: rotate the tooth-frame vectors by the tooth
    phase (cutting sense) and mirror y by the hand."""
    def to_T(vec):
        c = np.cos(phases)[:, None]
        s = np.sin(phases)[:, None]
        x = c * vec[..., 0] - s * vec[..., 1]
        y = s * vec[..., 0] + c * vec[..., 1]
        return np.ascontiguousarray(np.stack([x, hand_sign * y, vec[..., 2]], axis=-1))
    return EdgeFrameT(to_T(el.tangent_C), to_T(el.normal_C), to_T(el.cutdir_C), to_T(el.rake_normal_C))
