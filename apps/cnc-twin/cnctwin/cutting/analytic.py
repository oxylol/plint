"""Closed forms for the cutting checks (docs/models/cutting.md "Verification") and ideal tools.

* :func:`milling_average_forces` - mean forces per tooth period of a linear edge-force model on a
  straight-path cut with immersion [phi_st, phi_ex] (Altintas 2012, ch. 2.3, eq. 2.42).
* :func:`milling_forces_at` - instantaneous forces of a straight-tooth cutter (the same model).
* :func:`drilling_lip_chisel` - thrust and torque of a two-lip drill as the lip and chisel integrals of
  a linear model (Chandrasekharan, Kapoor & DeVor 1998 structure; straight lips at a web offset).
* :func:`kienzle_turning` - cutting force of a straight-edge turning tool (Kienzle 1952).
* :func:`nose_chip_area`, :func:`colwell_angle`, :func:`equivalent_chip_thickness` - chip cross-section of
  a nose-radius tool, Colwell's (1954) chord angle and Woxen's equivalent chip thickness A / L
  (Young, Mathew & Oxley 1987).
* :func:`runout_chip_thickness` - effective feed per tooth with runout (Kline & DeVor 1983).
* ideal tools as EdgeModel + frames + a minimal geometry (:func:`cylinder_mill`, :func:`two_lip_drill`,
  :func:`straight_turning_tool`) for the kernels through :func:`cnctwin.cutting.geometry.build_cut_tool`.

Frame of the milling closed forms (Altintas): X along the feed, Y the feed turned by +90 deg in the
plane, Z along the tool axis toward the spindle; the cutter turns clockwise seen from the spindle (M03);
phi is measured clockwise from +Y, so an edge at phi sits at R (sin phi, cos phi) and h = c sin phi.
Forces are on the tool: F = -F_t c(phi) - F_r n(phi) - F_a e_z with c = (cos phi, -sin phi),
n = (sin phi, cos phi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EDGE_CHISEL, EDGE_LIP, EDGE_PERIPHERAL, EdgeModel


# ============================================================================ milling
def milling_average_forces(Ktc, Krc, Kac, Kte, Kre, Kae, n_teeth, a_m, c_m, phi_st, phi_ex):
    """Mean forces (X, Y, Z) on the tool over a tooth period (Altintas 2012 eq. 2.42), linear model
    F_t = Ktc h a + Kte a, F_r = Krc h a + Kre a, F_a = Kac h a + Kae a, h = c sin(phi)."""
    def fx(p):
        return (n_teeth * a_m * c_m / (8.0 * math.pi)) * (Ktc * math.cos(2.0 * p) - Krc * (2.0 * p - math.sin(2.0 * p))) \
            + (n_teeth * a_m / (2.0 * math.pi)) * (-Kte * math.sin(p) + Kre * math.cos(p))

    def fy(p):
        return (n_teeth * a_m * c_m / (8.0 * math.pi)) * (Ktc * (2.0 * p - math.sin(2.0 * p)) + Krc * math.cos(2.0 * p)) \
            - (n_teeth * a_m / (2.0 * math.pi)) * (Kte * math.cos(p) + Kre * math.sin(p))

    def fz(p):
        return (n_teeth * a_m / (2.0 * math.pi)) * (Kac * c_m * math.cos(p) - Kae * p)

    return fx(phi_ex) - fx(phi_st), fy(phi_ex) - fy(phi_st), fz(phi_ex) - fz(phi_st)


def milling_forces_at(Ktc, Krc, Kac, Kte, Kre, Kae, a_m, c_m, phi):
    """Forces (X, Y, Z) on the tool of one straight tooth at immersion phi (h = c sin phi > 0)."""
    h = c_m * math.sin(phi)
    if h <= 0.0:
        return 0.0, 0.0, 0.0
    ft = (Ktc * h + Kte) * a_m
    fr = (Krc * h + Kre) * a_m
    fa = (Kac * h + Kae) * a_m
    return (-ft * math.cos(phi) - fr * math.sin(phi), ft * math.sin(phi) - fr * math.cos(phi), -fa)


def runout_chip_thickness(c_m, radii_m, pitch_rad=None) -> np.ndarray:
    """Peak chip thickness (at phi = 90 deg) of every tooth with radial runout (Kline & DeVor 1983): tooth j
    cuts the surface of the last tooth m that cut, h_j = (j - m) c pitch-weighted + r_j - r_m; a tooth with
    h <= 0 skips and the next one sees the accumulated feed. Returned for the steady state."""
    r = np.asarray(radii_m, float)
    n = r.shape[0]
    pitch = np.full(n, 2.0 * math.pi / n) if pitch_rad is None else np.asarray(pitch_rad, float)
    feed = c_m * n * pitch / (2.0 * math.pi)       # feed travelled before tooth j (variable pitch)
    # steady state over many revolutions: iterate the surface left at phi = 90 deg
    surface = 0.0
    position = 0.0
    out = np.zeros(n)
    for _ in range(8):
        for j in range(n):
            position += feed[j]
            depth = position + r[j] - surface
            out[j] = max(depth, 0.0)
            if depth > 0.0:
                surface = position + r[j]
    return out


# ============================================================================ drilling
def drilling_lip_chisel(Ktc, Krc, Kte, Kre, f_rev_m, radius_m, half_web_m, half_point_rad, chisel_Et_N_per_m,
                        chisel_Er_N_per_m, n_lips: int = 2):
    """Thrust and torque of a drill whose straight lips run at the web offset w from the chisel corner
    (x = 0) to the periphery (x_R = sqrt(R^2 - w^2)) at the half point angle p, with a straight chisel
    through the axis (half length w per lip), as the lip and chisel integrals of the mechanistic drilling
    model (Chandrasekharan, Kapoor & DeVor 1998) with constant coefficients:

        h = (f / N) sin p,  lip element ds = dx / sin p at radius r = sqrt(x^2 + w^2)
        thrust = N (Krc h + Kre) sin p L_lip + N Er_ch w,        L_lip = x_R / sin p
        torque = N (Ktc h + Kte) int r ds + N (Krc h + Kre) w cos p L_lip + N Et_ch w^2 / 2

    with int r ds = I_r / sin p, I_r = int_0^x_R sqrt(x^2 + w^2) dx; the chisel row gives constant forces
    per unit length (indentation); the lip force acts along the cutting velocity (moment arm r) and along
    the lip normal, whose horizontal component cos p acts at the web offset w (moment arm w)."""
    h = f_rev_m / n_lips * math.sin(half_point_rad)
    w = half_web_m
    xR = math.sqrt(max(radius_m * radius_m - w * w, 0.0))

    def prim(x):
        if w == 0.0:
            return 0.5 * x * x
        return 0.5 * (x * math.sqrt(x * x + w * w) + w * w * math.asinh(x / w))

    int_r = prim(xR) - prim(0.0)
    L_lip = xR / math.sin(half_point_rad)
    thrust = n_lips * (Krc * h + Kre) * xR + n_lips * chisel_Er_N_per_m * w
    torque = (n_lips * (Ktc * h + Kte) * int_r / math.sin(half_point_rad)
              + n_lips * (Krc * h + Kre) * w * math.cos(half_point_rad) * L_lip
              + n_lips * chisel_Et_N_per_m * 0.5 * w * w)
    return thrust, torque


# ============================================================================ turning
def kienzle_turning(kc11_Pa, mc, f_m, ap_m, kappa_r_rad, h_ref_m=1e-3):
    """F_c = kc11 b h (h / h_ref)^-mc with h = f sin(kappa_r), b = ap / sin(kappa_r) (Kienzle 1952)."""
    h = f_m * math.sin(kappa_r_rad)
    b = ap_m / math.sin(kappa_r_rad)
    return kc11_Pa * b * h * (h / h_ref_m) ** (-mc)


def nose_profile_depth(z_m, ap_m, r_eps_m, kappa_r_rad, kappa_minor_rad):
    """Depth below the uncut surface reached by a nose-radius tool at axial position z (z = 0 under the
    lowest nose point, the tool feeding toward -z): major edge at the entering angle kappa_r on the feed
    side, nose arc of radius r_eps, minor edge at kappa_r' behind."""
    z = np.atleast_1d(np.asarray(z_m, float))
    zt_major = -r_eps_m * math.sin(kappa_r_rad)
    dt_major = ap_m - r_eps_m + r_eps_m * math.cos(kappa_r_rad)
    zt_minor = r_eps_m * math.sin(kappa_minor_rad)
    dt_minor = ap_m - r_eps_m + r_eps_m * math.cos(kappa_minor_rad)
    nose = ap_m - r_eps_m + np.sqrt(np.clip(r_eps_m * r_eps_m - z * z, 0.0, None))
    major = dt_major + (z - zt_major) * math.tan(kappa_r_rad)
    minor = dt_minor - (z - zt_minor) * math.tan(kappa_minor_rad)
    return np.where(z <= zt_major, major, np.where(z >= zt_minor, minor, nose))


def nose_chip_area(f_m, ap_m, r_eps_m, kappa_r_rad, kappa_minor_rad=math.radians(5.0), n: int = 400001):
    """(area, engaged edge length) of the uncut chip of one revolution of a nose-radius tool: the material
    between the surface the previous revolution left (the profile shifted by +f) and the current profile,
    both clipped to [0, ap], by quadrature (exact to the grid)."""
    z_lo = -(ap_m / math.tan(kappa_r_rad) + r_eps_m + f_m) if kappa_r_rad < math.pi / 2 else -(r_eps_m + 2.0 * f_m)
    z = np.linspace(z_lo, r_eps_m + 2.0 * f_m, n)
    cur = np.clip(nose_profile_depth(z, ap_m, r_eps_m, kappa_r_rad, kappa_minor_rad), 0.0, ap_m)
    prev = np.clip(nose_profile_depth(z - f_m, ap_m, r_eps_m, kappa_r_rad, kappa_minor_rad), 0.0, ap_m)
    gap = np.clip(cur - prev, 0.0, None)
    area = float(np.trapezoid(gap, z))
    slope = np.gradient(cur, z)
    engaged = gap > 0.0
    length = float(np.trapezoid(np.where(engaged, np.sqrt(1.0 + slope * slope), 0.0), z))
    return area, length


def colwell_angle(f_m, ap_m, r_eps_m, kappa_r_rad, kappa_minor_rad=math.radians(5.0)):
    """Angle from the feed axis of Colwell's chord joining the ends of the engaged edge: the minor end where
    the nose meets the previous revolution's nose (z = f / 2) and the major end where the major edge
    reaches the uncut surface; the chip (and the resultant thrust) is normal to it (Colwell 1954)."""
    z_m = 0.5 * f_m
    d_m = float(nose_profile_depth(z_m, ap_m, r_eps_m, kappa_r_rad, kappa_minor_rad)[0])
    zt = -r_eps_m * math.sin(kappa_r_rad)
    dt = ap_m - r_eps_m + r_eps_m * math.cos(kappa_r_rad)
    z_M = zt - dt / math.tan(kappa_r_rad) if dt > 0.0 else -math.sqrt(max(r_eps_m ** 2 - (r_eps_m - ap_m) ** 2, 0.0))
    return math.atan2(d_m, z_m - z_M)


def equivalent_chip_thickness(area_m2, engaged_length_m):
    """Woxen's equivalent chip thickness A / L."""
    return area_m2 / engaged_length_m if engaged_length_m > 0.0 else 0.0


# ============================================================================ ideal tools
@dataclass
class _Groups:
    kind: np.ndarray
    rake_rad: np.ndarray
    incl_rad: np.ndarray
    clear_rad: np.ndarray
    edge_radius_m: np.ndarray
    chamfer_width_m: np.ndarray
    chamfer_angle_rad: np.ndarray
    forming: np.ndarray
    count: np.ndarray


@dataclass
class _Props:
    diameter_m: float
    flutes: int
    sections: list = field(default_factory=list)
    stickout_m: float = 0.03


@dataclass
class _Frames:
    tangent: np.ndarray
    normal: np.ndarray
    cut_dir: np.ndarray
    rake_normal: np.ndarray


@dataclass
class IdealTool:
    """Minimal stand-in for tooling's ToolGeometry (what geometry.build_cut_tool reads)."""

    edge: EdgeModel
    frames: _Frames
    groups: _Groups
    props: _Props
    type: str = "square_endmill"
    rotating: bool = True
    holder: object = None
    slot: int = 0


def _groups(kinds, rakes, incls, clears, radii, forming=None) -> _Groups:
    n = len(kinds)
    return _Groups(np.array(kinds, np.int32), np.array(rakes, float), np.array(incls, float), np.array(clears, float),
                   np.array(radii, float), np.zeros(n), np.zeros(n), np.array(forming or [0] * n, np.int8),
                   np.ones(n, np.int32))


def cylinder_mill(R_m, n_teeth, ap_m, n_z, helix_rad=0.0, pitch_rad=None, hand=-1.0, rake_rad=0.1, clear_rad=0.15,
                  edge_radius_m=0.0, runout_T_m=(0.0, 0.0, 0.0), z0_m=0.0) -> IdealTool:
    """Peripheral teeth of a cylinder of radius R from z0 to z0 + ap (n_z elements per tooth, constant-lead
    helix), tooth j at the cutting-sense phase -sum(pitch[:j]) (equal pitch by default), hand -1 = RH."""
    pitch = np.full(n_teeth, 2.0 * math.pi / n_teeth) if pitch_rad is None else np.asarray(pitch_rad, float)
    phases = -np.concatenate([[0.0], np.cumsum(pitch[:-1])])
    dz = ap_m / n_z
    z = z0_m + (np.arange(n_z) + 0.5) * dz
    tb = math.tan(helix_rad)
    lag = -z * tb / R_m
    shape = (n_teeth, n_z)
    alpha = hand * (phases[:, None] + lag[None, :])
    ca, sa = np.cos(alpha), np.sin(alpha)
    normal = np.stack([ca, sa, np.zeros(shape)], axis=-1)
    cut = np.stack([-hand * sa, hand * ca, np.zeros(shape)], axis=-1)
    sb, cb = math.sin(helix_rad), math.cos(helix_rad)
    # profile tangent going up the flute: the helix trails in the cutting sense
    tangent = -np.stack([sb * cut[..., 0], sb * cut[..., 1], -np.full(shape, cb)], axis=-1)
    rake_normal = -cut
    f2 = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), shape).copy())
    i2 = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v), shape).astype(np.int32).copy())
    edge = EdgeModel(np.mod(phases, 2.0 * math.pi), f2(z[None, :]), f2(R_m), f2(lag[None, :]), f2(dz / cb), f2(dz),
                     f2(math.pi / 2.0), f2(math.atan(math.tan(rake_rad) * cb)), f2(helix_rad), f2(clear_rad),
                     f2(edge_radius_m), i2(EDGE_PERIPHERAL), i2(0), i2(np.arange(n_z)[None, :]),
                     np.ones(shape, np.int8), np.array(runout_T_m, float), np.zeros(2), z.copy(), np.full(n_z, R_m),
                     np.array([hand]))
    frames = _Frames(np.ascontiguousarray(tangent), np.ascontiguousarray(normal), np.ascontiguousarray(cut),
                     np.ascontiguousarray(rake_normal))
    groups = _groups([EDGE_PERIPHERAL], [math.atan(math.tan(rake_rad) * cb)], [helix_rad], [clear_rad], [edge_radius_m])
    return IdealTool(edge, frames, groups, _Props(2.0 * R_m, n_teeth))


def end_face_mill(R_m, n_teeth, n_r, r0_m=0.0, hand=-1.0, rake_rad=0.1, clear_rad=0.15) -> IdealTool:
    """Flat end teeth (kappa = 0, normal -z) from r0 to R, radial (no lag): the end edges of a plunging or
    ramping end mill."""
    pitch = 2.0 * math.pi / n_teeth
    phases = -np.arange(n_teeth) * pitch
    dr = (R_m - r0_m) / n_r
    r = r0_m + (np.arange(n_r) + 0.5) * dr
    shape = (n_teeth, n_r)
    alpha = hand * phases[:, None] + np.zeros(shape)
    ca, sa = np.cos(alpha), np.sin(alpha)
    normal = np.stack([np.zeros(shape), np.zeros(shape), -np.ones(shape)], axis=-1)
    cut = np.stack([-hand * sa, hand * ca, np.zeros(shape)], axis=-1)
    tangent = np.stack([ca, sa, np.zeros(shape)], axis=-1)       # outward along the end edge
    f2 = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), shape).copy())
    i2 = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v), shape).astype(np.int32).copy())
    from ..core.abi import EDGE_END
    edge = EdgeModel(np.mod(phases, 2.0 * math.pi), f2(0.0), f2(r[None, :]), f2(0.0), f2(dr), f2(dr), f2(0.0),
                     f2(rake_rad), f2(0.0), f2(clear_rad), f2(0.0), i2(EDGE_END), i2(0), i2(np.arange(n_r)[None, :]),
                     np.ones(shape, np.int8), np.zeros(3), np.zeros(2), np.zeros(n_r), r.copy(), np.array([hand]))
    frames = _Frames(np.ascontiguousarray(tangent), np.ascontiguousarray(normal), np.ascontiguousarray(cut),
                     np.ascontiguousarray(-cut))
    groups = _groups([EDGE_END], [rake_rad], [0.0], [clear_rad], [0.0])
    return IdealTool(edge, frames, groups, _Props(2.0 * R_m, n_teeth))


def two_lip_drill(R_m, half_point_rad, half_web_m, n_lip, n_chisel, hand=-1.0, rake_rad=0.3,
                  chisel_rake_rad=-0.95, clear_rad=0.17) -> IdealTool:
    """Two straight lips at the web offset w (lip j: points (x, -w) of its tooth frame rotated by the tooth
    phase, x from x_c to sqrt(R^2 - w^2), z = (x - x_c) / tan p) and a straight chisel through the axis
    joining the lip corners (half chisel from the axis to (x_c, -w)), no margins."""
    nt = 2
    x_c = 0.0
    x_R = math.sqrt(R_m * R_m - half_web_m * half_web_m)
    n = n_lip + n_chisel
    shape = (nt, n)
    P = np.zeros(shape + (3,))
    N_ = np.zeros(shape + (3,))
    C = np.zeros(shape + (3,))
    Tg = np.zeros(shape + (3,))
    db = np.zeros(shape)
    kind = np.zeros(shape, np.int32)
    group = np.zeros(shape, np.int32)
    incl = np.zeros(shape)
    rake = np.zeros(shape)
    sp, cp = math.sin(half_point_rad), math.cos(half_point_rad)
    ch_len = half_web_m        # chisel half: from the axis (0, 0) to the lip corner (0, -w) along -y
    for j in range(nt):
        phase = -j * math.pi
        a = hand * phase
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        # chisel half (tooth frame): from (0, 0, 0) to (0, -w, 0)
        for k in range(n_chisel):
            t = (k + 0.5) / n_chisel
            p = np.array([0.0, -hand * half_web_m * t, 0.0])
            P[j, k] = rot @ p
            N_[j, k] = [0.0, 0.0, -1.0]
            v = np.cross(np.array([0.0, 0.0, 1.0]), P[j, k])
            nv = np.linalg.norm(v)
            C[j, k] = hand * v / nv if nv > 0 else rot @ np.array([1.0, 0.0, 0.0]) * -hand
            Tg[j, k] = rot @ np.array([0.0, -hand, 0.0])
            db[j, k] = ch_len / n_chisel
            kind[j, k] = EDGE_CHISEL
            group[j, k] = 1
            rake[j, k] = chisel_rake_rad
        for k in range(n_lip):
            t = (k + 0.5) / n_lip
            x = x_c + t * (x_R - x_c)
            p = np.array([x, -hand * half_web_m, (x - x_c) / math.tan(half_point_rad)])
            q = rot @ p
            P[j, n_chisel + k] = q
            # lip normal: in the plane of the lip and the axis, normal to the lip line: (cos p e_x' - sin p e_z)
            lip_dir = rot @ np.array([sp, 0.0, cp])
            n_vec = rot @ np.array([cp, 0.0, -sp])
            N_[j, n_chisel + k] = n_vec
            v = np.cross(np.array([0.0, 0.0, 1.0]), q)
            c_vec = hand * v / np.linalg.norm(v)
            # the velocity of a lip point at the web offset is not normal to the lip: that is its inclination
            C[j, n_chisel + k] = c_vec
            Tg[j, n_chisel + k] = lip_dir
            incl[j, n_chisel + k] = math.asin(max(-1.0, min(1.0, -float(np.dot(lip_dir, c_vec)))))
            db[j, n_chisel + k] = (x_R - x_c) / sp / n_lip
            kind[j, n_chisel + k] = EDGE_LIP
            group[j, n_chisel + k] = 0
            rake[j, n_chisel + k] = rake_rad
    r = np.hypot(P[..., 0], P[..., 1])
    lag = np.zeros(shape)
    for j in range(nt):
        base = hand * (-j * math.pi)
        ang = np.arctan2(P[j, :, 1], P[j, :, 0])
        lag[j] = hand * (ang - base)
    station = np.tile(np.arange(n, dtype=np.int32), (nt, 1))
    f2 = lambda v: np.ascontiguousarray(np.asarray(v, float))
    edge = EdgeModel(np.mod(np.array([0.0, -math.pi]), 2.0 * math.pi), f2(P[..., 2]), f2(r), f2(lag), f2(db), f2(db),
                     f2(np.where(kind == EDGE_LIP, math.pi / 2.0 - half_point_rad, 0.0)), f2(rake), f2(incl),
                     f2(np.full(shape, clear_rad)), f2(np.zeros(shape)), np.ascontiguousarray(kind),
                     np.ascontiguousarray(group), station, np.ones(shape, np.int8), np.zeros(3), np.zeros(2),
                     f2(P[0, :, 2]), f2(r[0]), np.array([hand]))
    frames = _Frames(np.ascontiguousarray(Tg), np.ascontiguousarray(N_), np.ascontiguousarray(C), np.ascontiguousarray(-C))
    groups = _groups([EDGE_LIP, EDGE_CHISEL], [rake_rad, chisel_rake_rad], [0.0, 0.0], [clear_rad, clear_rad], [0.0, 0.0])
    return IdealTool(edge, frames, groups, _Props(2.0 * R_m, 2), type="twist_drill")


def straight_turning_tool(kappa_r_rad, edge_len_m, n, hand=-1.0, rake_rad=0.1, clear_rad=0.1) -> IdealTool:
    """Sharp straight major edge of a turning tool (no nose radius) through the tip: tool frame of tooling
    (x_T into the work, z_T from the tip into the tool along the work axis, the tool feeding along
    -z_T); the edge leaves the tip at the entering angle kappa_r from the feed direction, its normal
    n = sin(kappa) x_T - cos(kappa) z_T with kappa = 90 deg - kappa_r."""
    kappa = math.pi / 2.0 - kappa_r_rad
    shape = (1, n)
    # edge direction (x_T, z_T): from the tip (the trailing point at the machined diameter) away from the work
    # (x_T < 0) and ahead along the feed (-z_T), at kappa_r from the feed direction; perpendicular to n
    d = np.array([-math.sin(kappa_r_rad), 0.0, -math.cos(kappa_r_rad)])
    t = (np.arange(n) + 0.5) / n * edge_len_m
    P = t[:, None] * d[None, :]
    normal = np.array([math.sin(kappa), 0.0, -math.cos(kappa)])
    cut = np.array([0.0, hand, 0.0])
    tangent = d
    f2 = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), shape).copy())
    i2 = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v), shape).astype(np.int32).copy())
    from ..core.abi import EDGE_INSERT
    edge = EdgeModel(np.zeros(1), f2(P[None, :, 2]), f2(P[None, :, 0]), f2(0.0), f2(edge_len_m / n), f2(edge_len_m / n),
                     f2(kappa), f2(rake_rad), f2(0.0), f2(clear_rad), f2(0.0), i2(EDGE_INSERT), i2(0),
                     i2(np.arange(n)[None, :]), np.ones(shape, np.int8), np.zeros(3), np.zeros(2), P[:, 2].copy(),
                     P[:, 0].copy(), np.array([hand]))
    f3 = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), shape + (3,)).copy())
    frames = _Frames(f3(tangent), f3(normal), f3(cut), f3(-cut))
    groups = _groups([EDGE_INSERT], [rake_rad], [0.0], [clear_rad], [0.0])
    return IdealTool(edge, frames, groups, _Props(0.0, 1), type="turning_insert", rotating=False)


__all__ = ["milling_average_forces", "milling_forces_at", "runout_chip_thickness", "drilling_lip_chisel",
           "kienzle_turning", "nose_profile_depth", "nose_chip_area", "colwell_angle", "equivalent_chip_thickness",
           "IdealTool",
           "cylinder_mill", "end_face_mill", "two_lip_drill", "straight_turning_tool"]
