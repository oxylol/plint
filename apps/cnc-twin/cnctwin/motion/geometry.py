"""Path pieces for the look-ahead planner (docs/models/motion.md "Path geometry and limits").

A piece is a stretch of path with one geometry and one set of limits:

* **joint** pieces in machine-axis space: lines ``q = q0 + dq u`` and arcs (circular, helical, spiral
  within the control's radius tolerance) ``q = c + r(u)(cos(phi u) e1 + sin(phi u) e2) + w u`` in the
  path's X/Y/Z axes with the other axes linear in u (GEO_LINE / GEO_ARC, evaluated by the compiled
  ``geo_eval`` both here and in the loop for spindle-synchronous stretches);
* **tip** pieces (TCP): the tool tip moves on a line or arc in the workpiece frame W, the tool axis
  turns on the great circle between its end directions (slerp, Shoemake 1985) and the machine axes
  follow from inverse kinematics at every tick;
* **blend** pieces: circular arcs in the scaled joint space (1 deg = 1 mm for rotary axes) inserted at
  a corner within the tolerance band.

Path parameter (the metric the programmed feed applies to): FANUC/Haas: the vector of linear axes
(m) and rotary axes (1 deg = 1 mm), arcs along the plane arc (the helical axis follows); Siemens and
Heidenhain: the linear axes only when any moves (rotary axes follow), the helix length on helical
arcs; TCP: the tip path in W (the rotary angle (1 deg = 1 mm) when only the orientation changes).

Limits of a piece: v <= F (with override, the machine's max feed), v <= v_i / max|dq_i/ds| (axis
velocity limits), a <= A_i / max|dq_i/ds| and j <= J_i / max|dq_i/ds| (tangential), and on curved
pieces v <= sqrt(A_i / max|d2q_i/ds2|) and v <= (J_i / max|d3q_i/ds3|)^(1/3) (centripetal: on a circle
of radius R in the plane of axes i these are v <= sqrt(A R) and v <= (J R^2)^(1/3)).

Corner (a tangent discontinuity of angle theta between two pieces) in continuous-path mode with
tolerance eps: an arc tangent to both, R = eps cos(theta/2) / (1 - cos(theta/2)), tangent length
l = R tan(theta/2) (at most half of each adjacent piece, the radius shrinking accordingly so the
deviation from the corner stays <= eps); the arc's centripetal limits give the corner speed. Lines
are blended geometrically; at corners of arcs or TCP pieces the corner speed of the same arc bounds
the node and the commanded path keeps its corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .abi import GEO_ARC, GEO_LINE
from .kernels import geo_eval, geo_eval_many

PK_JOINT = 0
PK_TIP = 1
PK_BLEND = 2


@dataclass
class TipPath:
    """Tool tip path in W and the tool axis interpolation of a TCP piece."""

    kind: int                   # GEO_LINE / GEO_ARC
    p0: np.ndarray              # tip at u = 0 (W)
    p1: np.ndarray
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    e1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    e2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    r0: float = 0.0
    r1: float = 0.0
    sweep: float = 0.0
    helix: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))   # tool axis at u = 0 (W)
    a1: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    mode: str = "rotary"        # rotary: rotary axes linear in u (programmed B/C); vector: tool axis slerp
    rot_axes: list = field(default_factory=list)        # rotary axis indices interpolated linearly (rotary mode)
    rot0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rot1: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def tip(self, u: float) -> np.ndarray:
        if self.kind == GEO_ARC:
            r = self.r0 + (self.r1 - self.r0) * u
            ph = self.sweep * u
            return self.centre + r * (math.cos(ph) * self.e1 + math.sin(ph) * self.e2) + self.helix * u
        return self.p0 + (self.p1 - self.p0) * u

    def axis(self, u: float) -> np.ndarray:
        return slerp(self.a0, self.a1, u)


def slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Great-circle interpolation of unit vectors (shorter arc; a normalised lerp below 1e-4 rad)."""
    c = float(np.clip(a @ b, -1.0, 1.0))
    angle = math.acos(c)
    if angle < 1e-4:
        v = a + (b - a) * u
        return v / np.linalg.norm(v)
    if math.pi - angle < 1e-9:
        # antiparallel: turn about any vector normal to a (deterministic choice)
        n = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(n) < 1e-6:
            n = np.cross(a, np.array([0.0, 1.0, 0.0]))
        n = n / np.linalg.norm(n)
        ph = angle * u
        return math.cos(ph) * a + math.sin(ph) * np.cross(n, a)
    s = math.sin(angle)
    return (math.sin((1.0 - u) * angle) * a + math.sin(u * angle) * b) / s


@dataclass
class Piece:
    item: int                   # index of the owning command (PathPlanner.items)
    kind: int                   # PK_*
    geo: int                    # GEO_* of the joint geometry (joint and blend pieces)
    length: float               # path parameter length (m)
    vec: np.ndarray             # [4, NA] joint geometry
    par: np.ndarray             # [3]
    q0: np.ndarray              # axis values at the start (NA)
    q1: np.ndarray              # at the end
    t0: np.ndarray              # unit tangent at the start in the scaled joint space (NA)
    t1: np.ndarray              # at the end
    feed: str = "per_min"       # per_min | per_rev | inverse_time | rapid | thread | tap
    rapid: bool = False
    vmax: float = 0.0
    amax: float = 1.0
    jmax: float = 1.0
    ustart: float = 0.0
    ready: int = 0
    stop_end: bool = False
    tip: TipPath | None = None
    curvature_scaled: float = 0.0   # 1/R of an arc in the scaled joint space (0 for lines)
    samples_q: np.ndarray | None = None     # TCP: IK samples [n, NA] at u = linspace(0, 1, n)
    singular: bool = False
    blend: int = 0              # 0 programmed, 1 first half of a blend, 2 second half
    ideal_v: float = 0.0        # programmed speed (ledger: ideal time = length / ideal_v)
    feed_v: float = math.inf    # programmed path speed without override (inf for rapids)
    per_rev: float = 0.0        # feed per spindle revolution (m) of per-revolution pieces (speed set at plan time)
    limit_v: float = math.inf   # axis, curvature and machine limits at 100 % rapid override
    helix_m: float = 0.0        # rise of a helical joint arc along its normal over the whole arc (m)

    def eval(self, s: np.ndarray, axes: np.ndarray, out: np.ndarray) -> None:
        """Axis values at path positions s (joint and blend pieces) into out[:, axes]."""
        geo_eval_many(self.geo, self.length, self.vec, self.par, np.ascontiguousarray(s, dtype=float), axes, out)

    def eval1(self, s: float, axes: np.ndarray, out: np.ndarray) -> None:
        geo_eval(self.geo, self.length, self.vec, self.par, float(s), axes, out)


def scale_vector(model) -> np.ndarray:
    """Scaled joint metric weights: 1 for linear axes, m per rad for rotary axes (1 deg = 1 mm)."""
    return np.array([model.rot_len if ax.rotary else 1.0 for ax in model.axes])


def path_weights(model, dq: np.ndarray, axes: list[int]) -> np.ndarray:
    """Weights of the programmed-feed metric for a move dq (see module doc)."""
    w = np.zeros(len(model.axes))
    lin_moves = any(not model.axes[a].rotary and abs(dq[a]) > 0.0 for a in axes)
    linear_only = model.dialect.startswith("heidenhain") or model.dialect == "siemens840d"
    for a in axes:
        ax = model.axes[a]
        if ax.synthetic:
            continue
        if ax.rotary:
            w[a] = 0.0 if (linear_only and lin_moves) else model.rot_len
        else:
            w[a] = 1.0
    return w


def _unit_or_zero(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else np.zeros_like(v)


def line_piece(item: int, q0: np.ndarray, q1: np.ndarray, model, axes: list[int]) -> Piece:
    n_axes = len(model.axes)
    dq = q1 - q0
    w = path_weights(model, dq, axes)
    length = float(np.sqrt(np.sum((w * dq) ** 2)))
    vec = np.zeros((4, n_axes))
    vec[0] = q0
    vec[1] = dq
    sc = scale_vector(model)
    t = _unit_or_zero(sc * dq)
    return Piece(item, PK_JOINT, GEO_LINE, length, vec, np.zeros(3), q0.copy(), q1.copy(), t, t.copy())


def arc_piece(item: int, q0: np.ndarray, q1: np.ndarray, arc, model, path: int) -> Piece:
    """A machine-space G02/G03 (ArcSpec centre/normal in the path's X, Y, Z axes)."""
    n_axes = len(model.axes)
    letters = model.paths[path].letters
    xyz = [model.axis_index.get(letters.get(l)) if letters.get(l) else None for l in "XYZ"]
    p0 = np.array([q0[i] if i is not None else 0.0 for i in xyz])
    p1 = np.array([q1[i] if i is not None else 0.0 for i in xyz])
    n = np.asarray(arc.normal, dtype=float)
    n = n / np.linalg.norm(n)
    c = np.asarray(arc.centre, dtype=float).copy()
    c = c + n * float((p0 - c) @ n)               # centre in the plane of the start point
    rv = p0 - c
    r0 = float(np.linalg.norm(rv))
    e1 = rv / r0 if r0 > 0 else np.zeros(3)
    e2 = np.cross(n, e1)
    h = float((p1 - p0) @ n)
    end_in_plane = p1 - n * h - c
    r1 = float(np.linalg.norm(end_in_plane))
    sweep = float(arc.sweep_rad)
    vec = np.zeros((4, n_axes))
    for k, i in enumerate(xyz):
        if i is None:
            continue
        vec[0, i] = c[k]
        vec[1, i] = e1[k]
        vec[2, i] = e2[k]
        vec[3, i] = h * n[k]
    others = [a for a in model.axis_index.values() if a not in [x for x in xyz if x is not None]
              and model.axes[a].path == path and not model.axes[a].synthetic]
    dq = q1 - q0
    for a in others:
        vec[0, a] = q0[a]
        vec[3, a] = dq[a]
    w = path_weights(model, dq, [a for a in others])
    arc_len = abs(sweep) * 0.5 * (r0 + r1)
    linear_only = model.dialect.startswith("heidenhain") or model.dialect == "siemens840d"
    extra = float(np.sum((w * dq)[others] ** 2)) if others else 0.0
    length = math.sqrt(arc_len ** 2 + extra + (h * h if linear_only else 0.0))
    if length <= 0.0:
        length = abs(h)
    sc = scale_vector(model)
    # tangents in the scaled joint space at both ends
    t0 = np.zeros(n_axes)
    t1 = np.zeros(n_axes)
    d0 = r0 * sweep * e2 + h * n
    d1 = r1 * sweep * (math.cos(sweep) * e2 - math.sin(sweep) * e1) + h * n
    for k, i in enumerate(xyz):
        if i is not None:
            t0[i] = d0[k]
            t1[i] = d1[k]
    for a in others:
        t0[a] = dq[a]
        t1[a] = dq[a]
    piece = Piece(item, PK_JOINT, GEO_ARC, length, vec, np.array([r0, r1, sweep]), q0.copy(), q1.copy(),
                  _unit_or_zero(sc * t0), _unit_or_zero(sc * t1), helix_m=abs(h))
    r_mean = 0.5 * (r0 + r1)
    piece.curvature_scaled = 1.0 / r_mean if r_mean > 0 else 0.0
    return piece


def tip_piece(item: int, q0: np.ndarray, q1: np.ndarray, tip: TipPath, model) -> Piece:
    n_axes = len(model.axes)
    if tip.kind == GEO_ARC:
        arc_len = abs(tip.sweep) * 0.5 * (tip.r0 + tip.r1)
        length = math.sqrt(arc_len ** 2 + float(tip.helix @ tip.helix))
    else:
        length = float(np.linalg.norm(tip.p1 - tip.p0))
    if length < 1e-12:
        angle = math.acos(float(np.clip(tip.a0 @ tip.a1, -1.0, 1.0)))
        length = model.rot_len * angle
    sc = scale_vector(model)
    dq = q1 - q0
    t = _unit_or_zero(sc * dq)
    return Piece(item, PK_TIP, GEO_LINE, length, np.zeros((4, n_axes)), np.zeros(3), q0.copy(), q1.copy(), t,
                 t.copy(), tip=tip)


# ------------------------------------------------------------------------------- limits

def derivative_bounds(piece: Piece, model, axes: list[int], n_samples: int = 33) -> tuple[np.ndarray, ...]:
    """max |dq/ds|, max |d2q/ds2|, max |d3q/ds3| per axis over the piece (joint/blend: exact
    derivatives of the geometry sampled; tip: finite differences of the IK samples)."""
    n_axes = len(model.axes)
    L = piece.length
    d1 = np.zeros(n_axes)
    d2 = np.zeros(n_axes)
    d3 = np.zeros(n_axes)
    if L <= 0.0:
        return d1, d2, d3
    if piece.kind == PK_TIP:
        qs = piece.samples_q
        if qs is None or len(qs) < 2:
            return np.abs(piece.q1 - piece.q0) / L, d2, d3
        h = L / (len(qs) - 1)
        g1 = np.diff(qs, axis=0) / h
        d1 = np.max(np.abs(g1), axis=0)
        if len(qs) >= 3:
            g2 = np.diff(g1, axis=0) / h
            d2 = np.max(np.abs(g2), axis=0)
            if len(qs) >= 4:
                d3 = np.max(np.abs(np.diff(g2, axis=0) / h), axis=0)
        return d1, d2, d3
    if piece.geo == GEO_LINE:
        return np.abs(piece.vec[1]) / L, d2, d3
    r0, r1, ph = piece.par
    u = np.linspace(0.0, 1.0, n_samples)
    for a in axes:
        e1 = piece.vec[1, a]
        e2 = piece.vec[2, a]
        w = piece.vec[3, a]
        r = r0 + (r1 - r0) * u
        dr = (r1 - r0)
        cs = np.cos(ph * u)
        sn = np.sin(ph * u)
        # derivatives in u, then / L^k
        q1u = dr * (cs * e1 + sn * e2) + r * ph * (-sn * e1 + cs * e2) + w
        q2u = 2 * dr * ph * (-sn * e1 + cs * e2) - r * ph * ph * (cs * e1 + sn * e2)
        q3u = -3 * dr * ph * ph * (cs * e1 + sn * e2) - r * ph ** 3 * (-sn * e1 + cs * e2)
        d1[a] = float(np.max(np.abs(q1u))) / L
        d2[a] = float(np.max(np.abs(q2u))) / L ** 2
        d3[a] = float(np.max(np.abs(q3u))) / L ** 3
    return d1, d2, d3


def apply_limits(piece: Piece, model, axes: list[int], feed_v: float, rapid_scale: float = 1.0) -> None:
    """Set vmax, amax, jmax of a piece from the feed and the axis limits (module doc); keeps the
    programmed speed (feed_v) and the limit (limit_v, at 100 % rapid override) for the overrides the
    planner applies when it plans."""
    d1, d2, d3 = derivative_bounds(piece, model, axes)
    piece.feed_v = feed_v
    limit = math.inf
    amax = math.inf
    jmax = math.inf
    for a in axes:
        ax = model.axes[a]
        if ax.synthetic:
            continue
        v_ax = ax.rapid if piece.rapid else min(ax.max_feed, ax.rapid)
        if d1[a] > 1e-15:
            limit = min(limit, v_ax / d1[a])
            amax = min(amax, ax.accel / d1[a])
            jmax = min(jmax, ax.jerk / d1[a])
        if d2[a] > 1e-15:
            limit = min(limit, math.sqrt(ax.accel / d2[a]))
        if d3[a] > 1e-15:
            limit = min(limit, (ax.jerk / d3[a]) ** (1.0 / 3.0))
    if not math.isfinite(amax):
        amax = min((model.axes[a].accel for a in axes), default=1.0)
        jmax = min((model.axes[a].jerk for a in axes), default=1.0)
    piece.limit_v = limit
    piece.vmax = max(min(feed_v, limit * (rapid_scale if piece.rapid else 1.0)), 0.0)
    piece.amax = amax
    piece.jmax = jmax


# ------------------------------------------------------------------------------- tolerance-driven limits

def piece_radius(piece: Piece) -> float:
    """Radius of curvature of a piece's path (inf for lines): joint arcs and blends in the scaled joint space, a
    TCP arc on the tip path in W; a helix r(1 + (c/r)^2) with c the lead per radian (its curvature is
    r / (r^2 + c^2)), so a helical arc is no more limited than its own curve."""
    if piece.kind == PK_TIP:
        tip = piece.tip
        if tip is None or tip.kind != GEO_ARC or tip.sweep == 0.0:
            return math.inf
        r = 0.5 * (tip.r0 + tip.r1)
        c = float(np.linalg.norm(tip.helix)) / abs(tip.sweep)
    elif piece.geo == GEO_ARC:
        r0, r1, sweep = (float(x) for x in piece.par)
        if sweep == 0.0:
            return math.inf
        r = 0.5 * (r0 + r1)
        c = piece.helix_m / abs(sweep)
    else:
        return math.inf
    if not r > 0.0:
        return math.inf
    return r + c * c / r


def chord_speed(radius: float, tol: float, ipo_s: float) -> float:
    """Chord error of the linearised path at the interpolation period: the control hands the drives one point per
    IPO period T, and between two points the commanded path is their chord. On a curve of radius R a chord of
    length v T lies R - sqrt(R^2 - (v T / 2)^2) inside it; within the contour tolerance e that needs
    v <= 2 sqrt(2 R e - e^2) / T, about sqrt(8 R e) / T (Suh et al. 2008, sampled-data interpolation; Yeh & Hsu
    2002, confined chord error). e is capped at R (no chord is longer than the diameter)."""
    if not (radius > 0.0 and math.isfinite(radius)) or tol <= 0.0 or ipo_s <= 0.0:
        return math.inf
    e = min(tol, radius)
    return 2.0 * math.sqrt(max(2.0 * radius * e - e * e, 0.0)) / ipo_s


def filter_band_speed(radius: float, tol: float, filter_s: float) -> float:
    """Contour tolerance band of an axis setpoint filter of length T_f (a moving average: FANUC's linear
    acceleration/deceleration after interpolation, a nominal-position or jerk filter; Biagiotti & Melchiorri 2012):
    on a circle of radius R the filtered position lies v^2 T_f^2 / (24 R) inside the commanded one (the radius
    error of linear acc/dec after interpolation, FANUC 30i-B operator's manual, feedrate control), so the control
    holds it within the tolerance e with v <= sqrt(24 R e) / T_f."""
    if not (radius > 0.0 and math.isfinite(radius)) or tol <= 0.0 or filter_s <= 0.0:
        return math.inf
    return math.sqrt(24.0 * radius * tol) / filter_s


def tolerance_speed(piece: Piece, tol: float, ipo_s: float, filter_s: float = 0.0) -> float:
    """The tolerance-driven speed bound of a piece: chord error at the IPO period and, when the control filters
    its axis setpoints, the filter's tolerance band (inf on lines)."""
    r = piece_radius(piece)
    if not math.isfinite(r):
        return math.inf
    return min(chord_speed(r, tol, ipo_s), filter_band_speed(r, tol, filter_s))


def end_curvatures(piece: Piece, model) -> tuple[np.ndarray, np.ndarray]:
    """d2q/ds2 of every axis at the start and at the end of a joint or blend piece (zero for lines; TCP pieces
    are taken as zero: their IK samples do not resolve the end curvature)."""
    n_axes = len(model.axes)
    L = piece.length
    if L <= 0.0 or piece.kind == PK_TIP or piece.geo != GEO_ARC:
        return np.zeros(n_axes), np.zeros(n_axes)
    r0, r1, ph = (float(x) for x in piece.par)
    dr = r1 - r0
    v1, v2 = piece.vec[1], piece.vec[2]
    start = (2.0 * dr * ph * v2 - r0 * ph * ph * v1) / (L * L)
    c, s = math.cos(ph), math.sin(ph)
    end = (2.0 * dr * ph * (-s * v1 + c * v2) - r1 * ph * ph * (c * v1 + s * v2)) / (L * L)
    return start, end


def curvature_step_speed(prev: Piece, nxt: Piece, model, axes: list[int], filter_s: float = 0.0) -> float:
    """Highest speed at a junction where the path curvature changes (a line meeting an arc, two arcs): each axis'
    acceleration steps there by v^2 |dk_i| (dk_i the jump of d2q_i/ds2). The step may not exceed the axis'
    acceleration A_i (the same allowance as the velocity step of one interpolation period at a corner,
    jump_speed); a control that filters its setpoints over T_f spreads the step over the filter and keeps the
    jerk within J_i only up to J_i T_f, so min(A_i, J_i T_f) then."""
    _, k1 = end_curvatures(prev, model)
    k2, _ = end_curvatures(nxt, model)
    v = math.inf
    for a in axes:
        ax = model.axes[a]
        if ax.synthetic:
            continue
        step = abs(float(k2[a] - k1[a]))
        if step <= 1e-12:
            continue
        allowed = ax.accel if filter_s <= 0.0 else min(ax.accel, ax.jerk * filter_s)
        v = min(v, math.sqrt(allowed / step))
    return v


# ------------------------------------------------------------------------------- corners

def corner_angle(prev: Piece, nxt: Piece) -> float:
    c = float(np.clip(prev.t1 @ nxt.t0, -1.0, 1.0))
    return math.acos(c)


def blend_radius(tol: float, theta: float, l_avail: float) -> tuple[float, float]:
    """(R, l): the arc tangent to both segments with deviation <= tol from the corner and tangent length
    l <= l_avail."""
    half = 0.5 * theta
    one_minus = 1.0 - math.cos(half)
    if one_minus <= 0.0:
        return math.inf, 0.0
    R = tol * math.cos(half) / one_minus
    l = R * math.tan(half)
    if l > l_avail:
        l = l_avail
        R = l / math.tan(half) if math.tan(half) > 0 else 0.0
    return R, l


def end_derivatives(piece: Piece, model) -> tuple[np.ndarray, np.ndarray]:
    """dq/ds of every axis at the start and at the end of a piece (real units per metre of path)."""
    L = piece.length
    n_axes = len(model.axes)
    if L <= 0.0:
        return np.zeros(n_axes), np.zeros(n_axes)
    if piece.kind == PK_TIP:
        qs = piece.samples_q
        if qs is None or len(qs) < 2:
            d = (piece.q1 - piece.q0) / L
            return d, d.copy()
        h = L / (len(qs) - 1)
        return (qs[1] - qs[0]) / h, (qs[-1] - qs[-2]) / h
    if piece.geo == GEO_LINE:
        d = piece.vec[1] / L
        return d.copy(), d.copy()
    r0, r1, ph = piece.par
    dr = r1 - r0
    v1, v2, v3 = piece.vec[1], piece.vec[2], piece.vec[3]
    start = (dr * v1 + r0 * ph * v2 + v3) / L
    c, s = math.cos(ph), math.sin(ph)
    end = (dr * (c * v1 + s * v2) + r1 * ph * (-s * v1 + c * v2) + v3) / L
    return start, end


def jump_speed(prev: Piece, nxt: Piece, model, axes: list[int], dt: float) -> float:
    """Highest path speed at which a corner without geometric blending changes every axis' velocity by no
    more than its acceleration achieves in one interpolation period (the control's allowed speed difference
    at a corner)."""
    _, d1 = end_derivatives(prev, model)
    d2, _ = end_derivatives(nxt, model)
    v = math.inf
    for a in axes:
        ax = model.axes[a]
        if ax.synthetic:
            continue
        jump = abs(float(d2[a] - d1[a]))
        if jump > 1e-12:
            v = min(v, ax.accel * dt / jump)
    return v


def corner_speed(R: float, prev: Piece, nxt: Piece, model, axes: list[int]) -> float:
    """Speed through an arc of radius R (scaled joint space) joining prev and nxt (module doc)."""
    if R <= 0.0:
        return 0.0
    if not math.isfinite(R):
        return math.inf
    sc = scale_vector(model)
    # plane of the corner (unit vectors in the scaled space), per-axis amplitude of the normal direction
    u1 = prev.t1
    u2 = nxt.t0
    e1 = _unit_or_zero(u2 - u1)
    e2 = _unit_or_zero(u1 + u2)
    v = math.inf
    for a in axes:
        ax = model.axes[a]
        if ax.synthetic:
            continue
        rho = math.hypot(e1[a], e2[a]) / sc[a]
        if rho < 1e-15:
            continue
        v = min(v, math.sqrt(ax.accel * R / rho), (ax.jerk * R * R / rho) ** (1.0 / 3.0))
    return v


def blend_pieces(prev: Piece, nxt: Piece, tol: float, model, axes: list[int], prev_done: float,
                 frac: float) -> tuple[float, Piece, Piece, float] | None:
    """Arc blend of two joint lines in the scaled joint space.

    Returns (new length of prev, first half (item of prev), second half (item of nxt), length to cut
    from the start of nxt), or None when the corner cannot be blended (not two lines, no room)."""
    if prev.kind != PK_JOINT or nxt.kind != PK_JOINT or prev.geo != GEO_LINE or nxt.geo != GEO_LINE:
        return None
    theta = corner_angle(prev, nxt)
    if theta <= model.p["min_corner_angle_rad"] or theta >= math.pi - 1e-6:
        return None
    sc = scale_vector(model)
    L1 = float(np.linalg.norm(sc * (prev.q1 - prev.q0)))
    L2 = float(np.linalg.norm(sc * (nxt.q1 - nxt.q0)))
    if L1 <= 0 or L2 <= 0 or abs(L1 - prev.length) > 1e-9 * max(L1, 1e-3) or abs(L2 - nxt.length) > 1e-9 * max(L2,
                                                                                                         1e-3):
        return None           # the programmed-feed metric differs from the scaled metric (Siemens/Heidenhain rotary)
    avail1 = max(0.0, frac * L1 - 0.0) if prev_done <= 0.0 else max(0.0, min(frac * L1, L1 - prev_done))
    R, l = blend_radius(tol, theta, min(avail1, frac * L2))
    if l <= 1e-12 or not math.isfinite(R):
        return None
    n_axes = len(model.axes)
    u1 = prev.t1
    u2 = nxt.t0
    P = sc * prev.q1
    T1 = P - l * u1
    bis = _unit_or_zero(u2 - u1)
    C = P + (R / math.cos(0.5 * theta)) * bis
    e1 = (T1 - C) / R
    e2 = u1
    half = 0.5 * theta
    v = corner_speed(R, prev, nxt, model, axes)
    pieces = []
    for k, (a0, sweep) in enumerate(((0.0, half), (half, half))):
        f1 = math.cos(a0) * e1 + math.sin(a0) * e2
        f2 = -math.sin(a0) * e1 + math.cos(a0) * e2
        vec = np.zeros((4, n_axes))
        vec[0] = C / sc
        vec[1] = f1 / sc
        vec[2] = f2 / sc
        q_start = (C + R * f1) / sc
        q_end = (C + R * (math.cos(sweep) * f1 + math.sin(sweep) * f2)) / sc
        t_start = f2
        t_end = -math.sin(sweep) * f1 + math.cos(sweep) * f2
        side = prev if k == 0 else nxt
        piece = Piece(side.item, PK_BLEND, GEO_ARC, R * sweep, vec, np.array([R, R, sweep]),
                      q_start, q_end, t_start, t_end, feed=side.feed, rapid=side.rapid,
                      vmax=min(v, side.vmax), amax=min(prev.amax, nxt.amax), jmax=min(prev.jmax, nxt.jmax),
                      ready=side.ready, blend=k + 1, ideal_v=side.ideal_v, feed_v=side.feed_v,
                      per_rev=side.per_rev, limit_v=min(v, prev.limit_v, nxt.limit_v))
        piece.curvature_scaled = 1.0 / R
        pieces.append(piece)
    return L1 - l, pieces[0], pieces[1], l


def trim_line_end(piece: Piece, new_length_scaled: float, model) -> None:
    """Shorten a joint line to end new_length_scaled (scaled metric) from its start."""
    sc = scale_vector(model)
    L = float(np.linalg.norm(sc * (piece.q1 - piece.q0)))
    f = new_length_scaled / L if L > 0 else 0.0
    piece.q1 = piece.q0 + (piece.q1 - piece.q0) * f
    piece.vec[1] = piece.q1 - piece.q0
    piece.length = piece.length * f


def trim_line_start(piece: Piece, cut_scaled: float, model) -> None:
    """Drop the first cut_scaled (scaled metric) of a joint line."""
    sc = scale_vector(model)
    L = float(np.linalg.norm(sc * (piece.q1 - piece.q0)))
    f = cut_scaled / L if L > 0 else 0.0
    piece.q0 = piece.q0 + (piece.q1 - piece.q0) * f
    piece.vec[0] = piece.q0
    piece.vec[1] = piece.q1 - piece.q0
    piece.length = piece.length * (1.0 - f)
