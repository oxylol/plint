"""Workholding devices as contact patches on the part (docs/models/holding.md section 4).

Every device entry of a setup (``operations[i].setup.workholding[j]``: a database item placed by ``pose``,
the device frame D in the workholding frame H) becomes patches: rectangles (or cylinder strips) on the
part's surface, each discretised by n x n Simpson nodes (weights exact for the force, first and second
moments of a uniform pressure, so stiffness and tipping edges are exact; the torsional friction of a
rectangle is within 1 % with 5 x 5 nodes). A patch has a contact law:

* ``locator`` - stiff unilateral support that reacts as needed (vise fixed jaw, parallels, plates, pins,
  jaw steps), preload from the clamping solve;
* ``locked`` - a clamp tightened force-controlled to its clamping force, then locked (screws, wedges,
  chuck jaws, steady-rest rollers, centres): a preloaded spring;
* ``bond`` - bilateral bonded area with tensile and shear strength (adhesive films, tabs).

Vacuum and magnetic holding are constant hold-down forces on the part (pressure x area at the area's
centroid) plus the plate as a locator patch. Device frame conventions (also in the model doc):

* vise / soft jaws: origin at the top edge of the fixed jaw face, centred on the jaw width; x_D along the
  jaw width, y_D the clamping direction (fixed jaw face y_D = 0 facing +y), z_D up (jaw tops at z_D = 0);
* clamp: origin at the pad's contact point, the clamp presses along -z_D;
* fixture plate, vacuum, magnetic, adhesive: origin at the centre of the plate top, z_D up;
* chucks and collets: origin at the chuck face centre, z_D along the spindle axis out of the chuck;
* tailstock centre: origin at the centre tip, z_D along the part axis toward the chuck;
* steady rest: origin on the part axis in the plane of the rollers, z_D along the axis, rollers at
  90 deg + 360 deg j / n from x_D;
* tabs: origin at the tab section centre, x_D normal to the section pointing into the held part, the
  section rectangle in (y_D, z_D) of the item's ``dims_mm`` width (y) and height (z);
* point locators: any entry whose ``label`` starts with ``locator``, ``stop`` or ``pin`` touches the part
  at its pose origin pushing along +x_D; ``support`` / ``rest`` labels push along +z_D.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from ..units import STANDARD_ATMOSPHERE_PA
from . import params as P
from .protocols import PartShape

LAW_LOCATOR = "locator"
LAW_LOCKED = "locked"
LAW_BOND = "bond"

LOCATOR_LABELS = ("locator", "stop", "pin")
SUPPORT_LABELS = ("support", "rest")
ROTATING_TYPES = ("chuck_3jaw", "chuck_4jaw", "collet_chuck", "collet")
AXIS_TOL = 0.01                       # sine of the largest angle between a chuck axis and the stock axis
VON_MISES_SHEAR = 1.0 / math.sqrt(3.0)   # distortion-energy shear strength ratio (Budynas & Nisbett sec. 5-5)


def mindlin_ratio(poisson: float) -> float:
    """Tangential / normal stiffness of an elastic contact, 2 (1 - nu) / (2 - nu) (Mindlin; Johnson 1985 7.2)."""
    return 2.0 * (1.0 - poisson) / (2.0 - poisson)


def simpson_weights(n: int) -> np.ndarray:
    """Composite Simpson weights of n (odd) equally spaced nodes on [-1, 1], normalised to sum 1."""
    if n == 1:
        return np.ones(1)
    if n < 3 or n % 2 == 0:
        raise InputError(f"holding: patch nodes must be 1 or an odd number >= 3 (got {n})")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / w.sum()


@dataclass
class Patch:
    """One contact area on the part (see the module doc); node arrays in H."""

    id: str
    role: str
    law: str
    device: str
    entry: int
    pos_H: np.ndarray
    n_H: np.ndarray
    t1_H: np.ndarray
    t2_H: np.ndarray
    weight: np.ndarray
    area_m2: float
    k_N_per_m: float
    kt_ratio: tuple = (0.0, 0.0)
    mu: float = 0.0
    force_N: float = 0.0
    gap_m: float = 0.0
    cap_t_Pa: float = 0.0
    cap_s_Pa: float = 0.0
    seal: np.ndarray | None = None
    datum: str = ""
    locating: bool = False
    centre_H: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal_H: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    axes_H: np.ndarray = field(default_factory=lambda: np.eye(3)[:2].copy())
    half_m: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rotating: bool = False
    slides_while_clamping: bool = False     # its actuation is not along its normal (a centre on its cone): no
                                            # tangential spring while the clamping force is applied

    @property
    def n_nodes(self) -> int:
        return int(self.pos_H.shape[0])


@dataclass
class DeviceOut:
    patches: list = field(default_factory=list)
    holddown: list = field(default_factory=list)      # dicts {force_H, point_H, kind, device, entry, area_m2, pressure_Pa}
    info: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


@dataclass
class BuildCtx:
    """What a device builder needs: the operation, the part placement, materials and settings."""

    op_name: str
    workpiece: int
    shape: PartShape
    T_H_S: np.ndarray
    n_side: int
    centre_nodes: int
    material_family: str = ""
    ultimate_Pa: float | None = None
    youngs_Pa: float | None = None
    poisson_work: float | None = None
    machine: object = None
    soft_jaws: object = None
    soft_entry: object = None         # the setup entry of the soft jaws (its grip_depth_mm applies to the chuck)
    omega_rad_s: float = 0.0          # work-spindle speed for the centrifugal jaw loss
    strict: bool = False              # the part has moved (slip): devices act only where they meet it
    T_H_S0: np.ndarray | None = None  # the part as programmed (before slip; None: T_H_S): where the devices are

    def corners_H(self, programmed: bool = False) -> np.ndarray:
        T = self.T_H_S0 if programmed and self.T_H_S0 is not None else self.T_H_S
        c = self.shape.corners_S()
        return c @ T[:3, :3].T + T[:3, 3]

    def aabb_D(self, T_H_D: np.ndarray, programmed: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box in D of the part now (or as programmed: the placement the devices were set to)."""
        T_D_H = np.linalg.inv(T_H_D)
        pts = self.corners_H(programmed) @ T_D_H[:3, :3].T + T_D_H[:3, 3]
        return pts.min(axis=0), pts.max(axis=0)

    def axis_D(self, T_H_D: np.ndarray):
        """(point at z0, unit direction, z-range along it) of a cylinder's axis in D."""
        T = np.linalg.inv(T_H_D) @ self.T_H_S
        p0 = T[:3, :3] @ np.array([0.0, 0.0, self.shape.z0_m]) + T[:3, 3]
        p1 = T[:3, :3] @ np.array([0.0, 0.0, self.shape.z1_m]) + T[:3, 3]
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        return p0, p1, u


def _k_joint(area_m2: float) -> float:
    return P.p(P.WORK, "joint_k_per_cm2_N_per_m") * area_m2 / 1e-4


def _series(*k) -> float:
    inv = sum(1.0 / v for v in k if v > 0.0 and math.isfinite(v))
    return 1.0 / inv if inv > 0.0 else 0.0


def _to_H(T_H_D, pts_D) -> np.ndarray:
    return np.asarray(pts_D, float) @ T_H_D[:3, :3].T + T_H_D[:3, 3]


def _dir_H(T_H_D, v_D) -> np.ndarray:
    return np.asarray(v_D, float) @ T_H_D[:3, :3].T


def rect_patch(pid, role, law, device, entry, T_H_D, centre_D, n_D, a1_D, a2_D, h1, h2, n_side, k, kt_ratio, mu,
               force=0.0, gap=0.0, datum="", locating=False, cap_t=0.0, cap_s=0.0, seal=False) -> Patch:
    """Rectangle on a plane of the part: centre, inward normal n (direction of the support force), in-plane
    axes a1, a2 (unit), half sizes h1, h2, n_side x n_side Simpson nodes (D frame inputs)."""
    n_D = np.asarray(n_D, float) / np.linalg.norm(n_D)
    a1_D = np.asarray(a1_D, float) - float(np.dot(a1_D, n_D)) * n_D
    if float(np.linalg.norm(a1_D)) < 1e-9:
        raise InputError(f"holding: patch {pid}: its first axis is parallel to its normal")
    a1_D = a1_D / np.linalg.norm(a1_D)
    a2_D = np.cross(n_D, a1_D)
    s = np.linspace(-1.0, 1.0, n_side) if n_side > 1 else np.zeros(1)
    w = simpson_weights(n_side)
    pts, wts, rim = [], [], []
    for i in range(len(s)):
        for j in range(len(s)):
            pts.append(np.asarray(centre_D, float) + s[i] * h1 * a1_D + s[j] * h2 * a2_D)
            wts.append(w[i] * w[j])
            rim.append(n_side > 1 and (i in (0, len(s) - 1) or j in (0, len(s) - 1)))
    m = len(pts)
    pos = _to_H(T_H_D, pts)
    nH = np.tile(_dir_H(T_H_D, n_D), (m, 1))
    t1 = np.tile(_dir_H(T_H_D, a1_D), (m, 1))
    t2 = np.tile(_dir_H(T_H_D, a2_D), (m, 1))
    area = 4.0 * h1 * h2
    return Patch(pid, role, law, device, entry, pos, nH, t1, t2, np.asarray(wts), area, float(k), tuple(kt_ratio),
                 float(mu), float(force), float(gap), float(cap_t), float(cap_s),
                 np.asarray(rim, bool) if seal else None, datum, locating, _to_H(T_H_D, [centre_D])[0],
                 -_dir_H(T_H_D, n_D), np.stack([_dir_H(T_H_D, a1_D), _dir_H(T_H_D, a2_D)]), np.array([h1, h2]))


def point_patch(pid, role, law, device, entry, T_H_D, point_D, n_D, k, kt_ratio, mu, force=0.0, gap=0.0, datum="",
                locating=False, half=0.0) -> Patch:
    n_D = np.asarray(n_D, float) / np.linalg.norm(n_D)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_D[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a1 = np.cross(helper, n_D)
    a1 /= np.linalg.norm(a1)
    return rect_patch(pid, role, law, device, entry, T_H_D, point_D, n_D, a1, np.cross(n_D, a1), max(half, 0.0),
                      max(half, 0.0), 1, k, kt_ratio, mu, force, gap, datum, locating)


def cylinder_strip(pid, role, law, device, entry, T_H_D, axis_p0_D, axis_u_D, radius, theta_c, half_theta, z_lo, z_hi,
                   n_side, k, kt_ratio, mu, force, inward=True, datum="", locating=False) -> Patch:
    """A jaw or segment on a cylinder of the given radius about the axis (p0 + s u), angles theta_c +- half_theta
    measured from the axis-normal frame (x_D projected), axial s in [z_lo, z_hi]."""
    u = np.asarray(axis_u_D, float)
    ex = np.array([1.0, 0.0, 0.0]) - u[0] * u
    if np.linalg.norm(ex) < 1e-9:
        ex = np.array([0.0, 1.0, 0.0]) - u[1] * u
    ex /= np.linalg.norm(ex)
    ey = np.cross(u, ex)
    s_ax = np.linspace(-1.0, 1.0, n_side) if n_side > 1 else np.zeros(1)
    w = simpson_weights(n_side)
    zc, hz = 0.5 * (z_lo + z_hi), 0.5 * (z_hi - z_lo)
    pts, nrm, t1, t2, wts = [], [], [], [], []
    for i, sa in enumerate(s_ax):
        for j, sb in enumerate(s_ax):
            th = theta_c + sb * half_theta
            er = math.cos(th) * ex + math.sin(th) * ey
            et = -math.sin(th) * ex + math.cos(th) * ey
            pts.append(np.asarray(axis_p0_D, float) + (zc + sa * hz) * u + radius * er)
            n = -er if inward else er
            nrm.append(n)
            t1.append(et)
            t2.append(np.cross(n, et))
            wts.append(w[i] * w[j])
    area = 2.0 * half_theta * radius * (z_hi - z_lo)
    centre = np.asarray(axis_p0_D, float) + zc * u + radius * (math.cos(theta_c) * ex + math.sin(theta_c) * ey)
    n_c = -(math.cos(theta_c) * ex + math.sin(theta_c) * ey) * (1.0 if inward else -1.0)
    et_c = -math.sin(theta_c) * ex + math.cos(theta_c) * ey
    return Patch(pid, role, law, device, entry, _to_H(T_H_D, pts), _dir_H(T_H_D, nrm), _dir_H(T_H_D, t1),
                 _dir_H(T_H_D, t2), np.asarray(wts), area, float(k), tuple(kt_ratio), float(mu), float(force), 0.0,
                 datum=datum, locating=locating, centre_H=_to_H(T_H_D, [centre])[0], normal_H=-_dir_H(T_H_D, n_c),
                 axes_H=np.stack([_dir_H(T_H_D, u), _dir_H(T_H_D, et_c)]),
                 half_m=np.array([hz, half_theta * radius]), rotating=True)


# --------------------------------------------------------------------------- clamping forces
def _force_from_torque(item, torque_Nm: float | None, what: str, notes: list) -> float | None:
    """Clamping force of an item at an input torque by its catalogue relation (force per torque, or the
    published force scaled linearly with torque); None when the item gives none."""
    fpt = P.num(item, "force_per_torque_N_per_Nm")
    f_ref = P.num(item, "clamping_force_N")
    t_ref = P.extra(item, "clamping_force_N", "at_handle_torque_Nm") or P.extra(item, "clamping_force_N",
                                                                                  "at_torque_Nm")
    if torque_Nm is None:
        if f_ref is not None:
            notes.append(f"{what}: no input torque given: the catalogue force {f_ref / 1e3:.2f} kN"
                         + (f" at {t_ref:.0f} Nm" if t_ref else "") + " is taken")
            return f_ref
        return None
    if fpt is not None:
        return fpt * torque_Nm
    if f_ref is not None and t_ref:
        return f_ref * torque_Nm / t_ref
    return None


def screw_force(torque_Nm: float, d_m: float, lead_m: float, half_angle_rad: float, mu_t: float, mu_c: float,
                r_collar_m: float) -> float:
    """Axial force of a power screw at a torque (Budynas & Nisbett eq. 8-5/8-6, pitch diameter d - lead/2)."""
    d_m_pitch = d_m - 0.5 * lead_m
    lam = math.atan(lead_m / (math.pi * d_m_pitch))
    rho = math.atan(mu_t / math.cos(half_angle_rad))
    lever = 0.5 * d_m_pitch * math.tan(lam + rho) + mu_c * r_collar_m
    return torque_Nm / lever


def vise_force(entry, item, notes: list) -> float:
    torque = P.num(entry, "input_torque_Nm")
    max_t = P.num(item, "max_input_torque_Nm")
    if torque is not None and max_t is not None and torque > max_t * (1.0 + 1e-9):
        notes.append(f"vise {P.text(item, 'id')}: input torque {torque:.0f} Nm above the maximum {max_t:.0f} Nm")
    force = _force_from_torque(item, torque, f"vise {P.text(item, 'id')}", notes)
    if force is not None:
        return float(force)
    if torque is None:
        torque = P.num(item, "recommended_torque_Nm")
    if torque is None:
        raise InputError(f"vise {P.text(item, 'id')}: neither an input torque nor a catalogue clamping force")
    notes.append(f"vise {P.text(item, 'id')}: force from the generic screw model (no catalogue relation)")
    return screw_force(torque, P.p(P.WORK, "vise_screw_d_m"), P.p(P.WORK, "vise_screw_lead_m"),
                       P.p(P.WORK, "vise_thread_half_angle_rad"), P.p(P.WORK, "vise_thread_friction"),
                       P.p(P.WORK, "vise_collar_friction"), P.p(P.WORK, "vise_collar_radius_m"))


def _stud_diameter_m(text: str) -> float | None:
    t = text.strip().upper()
    if t.startswith("M"):
        digits = ""
        for ch in t[1:]:
            if ch.isdigit() or ch == ".":
                digits += ch
            else:
                break
        if digits:
            return float(digits) * 1e-3
    return None


def clamp_force(entry, item, notes: list) -> float:
    torque = P.num(entry, "input_torque_Nm")
    ident = P.text(item, "id")
    force = _force_from_torque(item, torque, f"clamp {ident}", notes)
    if force is not None and (torque is not None or P.num(item, "stud_preload_N") is None):
        return float(force)
    lever = P.num(item, "lever_ratio") or 1.0
    t_rec = P.num(item, "recommended_torque_Nm")
    t_use = torque if torque is not None else t_rec
    preload = P.num(item, "stud_preload_N")
    if t_use is not None and preload is not None and t_rec:
        return preload * t_use / t_rec / lever
    d = _stud_diameter_m(P.text(item, "stud", "") or P.text(item, "screw", ""))
    if t_use is not None and d is not None:
        notes.append(f"clamp {ident}: stud preload from T = K F d (K = {P.p(P.WORK, 'nut_factor'):.2f})")
        return t_use / (P.p(P.WORK, "nut_factor") * d) / lever
    if force is not None:
        return float(force)
    raise InputError(f"clamp {ident}: no clamping force, stud preload or input torque to derive one")


def chuck_static_force(entry, item, notes: list) -> float:
    ident = P.text(item, "id")
    f_max = P.num(item, "gripping_force_N")
    if f_max is None:
        f_in = P.num(item, "max_input_force_N")
        ratio = P.num(item, "pull_back_ratio")
        if f_in is not None and ratio is not None:
            f_max = f_in * ratio
            notes.append(f"{ident}: gripping force = max input force x pull-back ratio (taper friction neglected)")
    torque = P.num(entry, "input_torque_Nm")
    if torque is not None and P.num(item, "force_per_torque_N_per_Nm") is not None:
        return P.num(item, "force_per_torque_N_per_Nm") * torque
    if f_max is None:
        raise InputError(f"{ident}: no gripping force in the database")
    pressure = P.num(entry, "pressure_Pa")
    p_max = P.num(item, "max_actuation_pressure_Pa")
    if p_max is None:
        p_max = _text_pressure_Pa(P.text(item, "matching_cylinder", ""))
        if p_max is not None and pressure is not None:
            notes.append(f"{ident}: maximum actuation pressure {p_max / 1e5:.1f} bar read from the matching cylinder "
                         f"text")
    if pressure is not None:
        if p_max is not None and p_max > 0.0:
            # wedge chucks: gripping force proportional to the cylinder thrust, i.e. to the pressure
            return f_max * min(pressure / p_max, 1.0)
        notes.append(f"{ident}: actuation pressure given but no maximum actuation pressure: the maximum gripping "
                     f"force is taken")
    return f_max


def _text_pressure_Pa(text: str) -> float | None:
    """'max pressure 2.65 MPa' / '... 26 bar' in a catalogue text -> Pa (None when absent)."""
    match = re.search(r"max(?:imum)?\.?\s*pressure\s*([0-9]+(?:\.[0-9]+)?)\s*(MPa|bar|kPa)", text or "", re.I)
    if not match:
        return None
    value = float(match.group(1))
    return value * {"mpa": 1e6, "bar": 1e5, "kpa": 1e3}[match.group(2).lower()]


def chuck_loss(item, omega_rad_s: float, static_N: float, n_jaws: int, notes: list) -> float:
    """Centrifugal loss of gripping force at a speed (VDI 3106: sum m r omega^2; or the item's model)."""
    w = abs(omega_rad_s)
    model = P.field_of(item, "centrifugal_loss_model")
    method = P.text(model, "method", "") if model is not None else ""
    if method == "quadratic_in_speed":
        loss, at = P.num(model, "loss_N"), P.num(model, "at_rad_s")
        if loss is not None and at:
            return loss * (w / at) ** 2
    if method == "measured":
        cur = P.curve(model, "gripping_force_curve_N")
        if cur is not None:
            x, y = cur
            f0 = float(np.interp(0.0, x, y))
            fw = float(np.interp(w, x, y))
            scale = static_N / f0 if f0 > 0.0 else 1.0
            return max(static_N - fw * scale, 0.0)
    coeff = P.num(item, "centrifugal_loss_coeff") or 1.0
    moment = P.num(item, "jaw_mass_moment_kg_m")
    if moment is not None:
        return coeff * moment * w * w
    m_jaw, r_jaw = P.num(item, "jaw_mass_kg"), P.num(item, "jaw_cg_radius_m")
    if m_jaw is not None and r_jaw is not None:
        return coeff * n_jaws * m_jaw * r_jaw * w * w
    w_max = P.num(item, "max_rad_s")
    if w_max:
        notes.append(f"{P.text(item, 'id')}: no centrifugal loss data: the speed rule (dynamic force 1/3 of static at "
                     f"the maximum speed) is taken")
        return P.p(P.WORK, "chuck_speed_rule_loss_fraction") * static_N * (w / w_max) ** 2
    notes.append(f"{P.text(item, 'id')}: no centrifugal loss data and no maximum speed: no loss taken")
    return 0.0


# --------------------------------------------------------------------------- device builders
def _jaw_friction(item, default_name="jaw_friction") -> float:
    mu = P.num(item, "jaw_friction")
    return float(mu) if mu is not None else P.p(P.WORK, default_name)


def _kt() -> tuple:
    """Tangential stiffness ratios of a dry elastic joint (Mindlin) along both tangents."""
    r = mindlin_ratio(P.p(P.WORK, "contact_poisson"))
    return (r, r)


def build_vise(bc: BuildCtx, j: int, entry, item, T_H_D: np.ndarray, label: str) -> DeviceOut:
    """Vise or soft jaws in a vise. D: origin on the seat (the top of the parallels, else the vise bed: where the
    part's bottom rests), centred on the jaw width; x along the jaw width, y the clamping direction, z up. The
    jaws rise from the bed (z = -parallels) to their tops at grip_depth_mm above the seat (else jaw height -
    parallels, else the part top); the jaw faces are where the part's y faces are (the convention of the project
    template, the examples and sanity's jaw boxes). A setup that does not place the part between the jaws is
    gripped over its bottom grip depth on the central jaw width (descriptive, noted). The jaws stay where the
    setup as programmed put them: a part that slipped is gripped where it still is between them."""
    out = DeviceOut()
    ident = P.text(item, "id", f"vise{j}")
    soft = bc.soft_jaws
    w = P.num(soft, "jaw_width_m") if soft is not None and P.num(soft, "jaw_width_m") else P.num(item, "jaw_width_m")
    h_j = P.num(soft, "jaw_height_m") if soft is not None and P.num(soft, "jaw_height_m") else \
        P.num(item, "jaw_height_m")
    mu_j = _jaw_friction(soft) if soft is not None and P.num(soft, "jaw_friction") is not None else _jaw_friction(item)
    if w is None:
        raise InputError(f"vise {ident}: jaw_width_mm missing")
    lo, hi = bc.aabb_D(T_H_D)
    lo0, hi0 = bc.aabb_D(T_H_D, programmed=True)
    part_h = hi0[2] - lo0[2]
    grip = P.num(entry, "grip_depth_m")
    parallels = P.num(entry, "parallels_m")
    par = parallels if parallels is not None else 0.0
    if grip is not None:
        top = grip
        if h_j is not None and abs(min(grip, part_h) - min(h_j - par, part_h)) > 5e-4:
            out.notes.append(f"vise {ident}: grip_depth_mm {grip * 1e3:.2f} and jaw height - parallels "
                             f"{(h_j - par) * 1e3:.2f} mm disagree: grip_depth is used")
    elif h_j is not None:
        top = h_j - par
    else:
        top = hi0[2]
        out.notes.append(f"vise {ident}: no jaw height and no grip depth: the jaws grip the whole part height")
    # the jaws (x range, z range) and the seat in D, from the setup as programmed
    if min(hi0[0], 0.5 * w) > max(lo0[0], -0.5 * w) + 1e-9 and min(hi0[2], top) > max(lo0[2], 0.0) + 1e-9:
        jaw_x, jaw_z, seat = (-0.5 * w, 0.5 * w), (0.0, top), 0.0
        if lo0[2] < -P.p(P.WORK, "seat_tolerance_m"):
            out.notes.append(f"vise {ident}: the part's contact shape reaches {-lo0[2] * 1e3:.3f} mm below the seat "
                             f"(a stock shape of a face machined earlier, or the pose): it rests on the seat")
    else:
        xc0 = 0.5 * (lo0[0] + hi0[0])
        depth = min(max(top if grip is not None or h_j is not None else part_h, 0.0), part_h)
        jaw_x, jaw_z, seat = (xc0 - 0.5 * w, xc0 + 0.5 * w), (lo0[2], lo0[2] + depth), lo0[2]
        out.notes.append(f"vise {ident}: the setup does not place the part between the jaws of the vise frame (origin "
                         f"on the seat): gripped over {depth * 1e3:.1f} mm from its bottom (grip_depth / jaw height - "
                         f"parallels)")
    # the part now between the jaws, on the seat
    x_lo, x_hi = max(lo[0], jaw_x[0]), min(hi[0], jaw_x[1])
    z_lo, z_hi = max(lo[2], jaw_z[0]), min(hi[2], jaw_z[1])
    if x_hi <= x_lo + 1e-9 or z_hi <= z_lo + 1e-9:
        out.info["no_grip"] = True
        out.notes.append(f"vise {ident}: " + ("the moved part is no longer between the jaws" if bc.strict else
                                              "the jaws do not reach the part"))
        return out
    gap = 0.0
    z_b = max(lo[2], seat)
    if lo[2] > seat + P.p(P.WORK, "seat_tolerance_m"):
        gap = -(lo[2] - seat)
        out.notes.append(f"vise {ident}: the part bottom is {(lo[2] - seat) * 1e3:.3f} mm above the "
                         f"{'parallels' if parallels is not None else 'vise bed'}: not seated")
    opening = P.num(entry, "jaw_opening_m")
    width = hi[1] - lo[1]
    if opening is not None and abs(opening - width) > 5e-4:
        out.notes.append(f"vise {ident}: jaw_opening {opening * 1e3:.2f} mm, part width between the jaws "
                         f"{width * 1e3:.2f} mm: the jaws close on the part")
    force = vise_force(entry, item, out.notes)
    k_j = P.num(item, "jaw_stiffness_N_per_m") or P.p(P.WORK, "vise_jaw_stiffness_N_per_m")
    xc, zc = 0.5 * (x_lo + x_hi), 0.5 * (z_lo + z_hi)
    h1, h2 = 0.5 * (x_hi - x_lo), 0.5 * (z_hi - z_lo)
    area = 4.0 * h1 * h2
    k_face = _series(k_j, _k_joint(area))
    base = f"{bc.op_name}/{j}:{ident}"
    kt = _kt()
    out.patches.append(rect_patch(f"{base}/fixed_jaw", "fixed_jaw", LAW_LOCATOR, ident, j, T_H_D, [xc, lo[1], zc],
                                  [0, 1, 0], [1, 0, 0], None, h1, h2, bc.n_side, k_face, kt, mu_j, datum="secondary",
                                  locating=True))
    out.patches.append(rect_patch(f"{base}/moving_jaw", "moving_jaw", LAW_LOCKED, ident, j, T_H_D, [xc, hi[1], zc],
                                  [0, -1, 0], [1, 0, 0], None, h1, h2, bc.n_side, k_face, kt, mu_j, force=force))
    # base: parallels under both jaws, else the vise bed, at the seat
    mu_b = P.p(P.WORK, "base_friction")
    if parallels is not None:
        w_p = min(P.p(P.WORK, "parallel_width_m"), 0.5 * width)
        for side, y0 in (("parallel_fixed", lo[1]), ("parallel_moving", hi[1] - w_p)):
            area_p = (x_hi - x_lo) * w_p
            out.patches.append(rect_patch(f"{base}/{side}", "parallel", LAW_LOCATOR, ident, j, T_H_D,
                                          [xc, y0 + 0.5 * w_p, z_b], [0, 0, 1], [1, 0, 0], None, h1, 0.5 * w_p,
                                          bc.n_side, _k_joint(area_p), kt, mu_b, gap=gap, datum="primary",
                                          locating=True))
    else:
        area_b = (x_hi - x_lo) * width
        out.patches.append(rect_patch(f"{base}/bed", "base", LAW_LOCATOR, ident, j, T_H_D,
                                      [xc, 0.5 * (lo[1] + hi[1]), z_b], [0, 0, 1], [1, 0, 0], None, h1, 0.5 * width,
                                      bc.n_side, _k_joint(area_b), kt, mu_b, gap=gap, datum="primary", locating=True))
    out.info.update({"clamp_force_N": force, "grip_depth_m": z_hi - z_lo, "grip_width_m": x_hi - x_lo,
                     "jaw_friction": mu_j, "seated_gap_m": -gap})
    return out


def build_clamp(bc: BuildCtx, j: int, entry, item, T_H_D: np.ndarray, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"clamp{j}")
    force = clamp_force(entry, item, out.notes)
    pad = P.p(P.WORK, "clamp_pad_m")
    k = _series(P.p(P.WORK, "clamp_stiffness_N_per_m"), _k_joint(pad * pad))
    mu = _jaw_friction(item)
    lo, hi = bc.aabb_D(T_H_D)
    lo0, hi0 = bc.aabb_D(T_H_D, programmed=True)
    # the pad presses along -z_D on the part's top (in D); where, in x and y, is set by the setup as programmed
    gap = 0.0
    if lo0[0] - 1e-9 <= 0.0 <= hi0[0] + 1e-9 and lo0[1] - 1e-9 <= 0.0 <= hi0[1] + 1e-9:
        px, py = 0.0, 0.0
        if abs(hi0[2]) > P.p(P.WORK, "seat_tolerance_m"):
            out.notes.append(f"clamp {ident}: pad {hi0[2] * 1e3:+.3f} mm from the part surface: placed on it")
    else:
        px, py = min(max(0.0, lo0[0]), hi0[0]), min(max(0.0, lo0[1]), hi0[1])
        out.notes.append(f"clamp {ident}: the pad is beside the part: moved onto its nearest edge")
    if not (lo[0] - 1e-9 <= px <= hi[0] + 1e-9 and lo[1] - 1e-9 <= py <= hi[1] + 1e-9):
        out.info["no_grip"] = True
        out.notes.append(f"clamp {ident}: the moved part is no longer under the pad")
        return out
    point = np.array([px, py, hi[2]])
    out.patches.append(rect_patch(f"{bc.op_name}/{j}:{ident}/pad", "clamp", LAW_LOCKED, ident, j, T_H_D, point,
                                  [0, 0, -1], [1, 0, 0], None, 0.5 * pad, 0.5 * pad, bc.n_side, k, _kt(), mu,
                                  force=force, gap=gap))
    out.info.update({"clamp_force_N": force, "push_H": _dir_H(T_H_D, [0, 0, -1])})
    return out


def _footprint(bc: BuildCtx, T_H_D: np.ndarray, plate_half=None):
    """(x_lo, x_hi, y_lo, y_hi, z_bottom, z_top) of the part footprint on a plate at z_D = 0."""
    lo, hi = bc.aabb_D(T_H_D)
    x_lo, x_hi, y_lo, y_hi = lo[0], hi[0], lo[1], hi[1]
    if plate_half is not None:
        x_lo, x_hi = max(x_lo, -plate_half[0]), min(x_hi, plate_half[0])
        y_lo, y_hi = max(y_lo, -plate_half[1]), min(y_hi, plate_half[1])
    return x_lo, x_hi, y_lo, y_hi, lo[2], hi[2]


def _plate_half(item) -> np.ndarray | None:
    plate = P.numlist(item, "plate_m")
    if plate and len(plate) >= 2:
        return 0.5 * np.array(plate[:2])
    return None


def _plate_top(item, z_bottom: float) -> float:
    """The plate face the part rests on, in D: z_D = 0 (the documented convention: origin on the plate top) or,
    when the item gives its thickness and the part sits there, z_D = thickness (origin on the plate bottom)."""
    plate = P.numlist(item, "plate_m")
    candidates = [0.0] + ([float(plate[2])] if plate and len(plate) >= 3 else [])
    return min(candidates, key=lambda z: (abs(z_bottom - z), z))


def _base_on_plate(bc, j, ident, role, T_H_D, item, mu, out: DeviceOut, seal=False, law=LAW_LOCATOR,
                   cap_t=0.0, cap_s=0.0, k_scale=1.0):
    x_lo, x_hi, y_lo, y_hi, z_b, z_t = _footprint(bc, T_H_D, _plate_half(item))
    if x_hi <= x_lo or y_hi <= y_lo:
        out.info["no_grip"] = True
        out.notes.append(f"{role} {ident}: the part is not on the plate")
        return None
    gap = 0.0
    top = _plate_top(item, z_b)
    if law != LAW_BOND and z_b - top > P.p(P.WORK, "seat_tolerance_m"):
        gap = -(z_b - top)
        out.notes.append(f"{role} {ident}: the part bottom is {(z_b - top) * 1e3:.3f} mm above the plate: not "
                         f"seated")
    area = (x_hi - x_lo) * (y_hi - y_lo)
    patch = rect_patch(f"{bc.op_name}/{j}:{ident}/{role}", role, law, ident, j, T_H_D,
                       [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), z_b], [0, 0, 1],
                       [1, 0, 0], None, 0.5 * (x_hi - x_lo), 0.5 * (y_hi - y_lo), bc.n_side,
                       _k_joint(area) * k_scale, _kt(), mu, gap=gap, datum="primary", locating=law != LAW_BOND,
                       cap_t=cap_t, cap_s=cap_s, seal=seal)
    out.patches.append(patch)
    return patch, area, (x_lo, x_hi, y_lo, y_hi, z_b, z_t)


def build_fixture_plate(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"plate{j}")
    lo, _ = bc.aabb_D(T_H_D)
    if lo[2] - _plate_top(item, lo[2]) > P.p(P.WORK, "mounting_gap_m"):
        out.notes.append(f"plate {ident}: {(lo[2] - _plate_top(item, lo[2])) * 1e3:.1f} mm below the part: it carries "
                         f"other devices, not the part")
        return out
    _base_on_plate(bc, j, ident, "plate", T_H_D, item, P.p(P.WORK, "base_friction"), out)
    return out


def build_point_locator(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    """A locator pin / stop (+x_D) or a rest button (+z_D) at the entry's pose origin."""
    out = DeviceOut()
    ident = P.text(item, "id", f"locator{j}")
    support = label.lower().startswith(SUPPORT_LABELS)
    n_D = np.array([0.0, 0.0, 1.0]) if support else np.array([1.0, 0.0, 0.0])
    lo, hi = bc.aabb_D(T_H_D)
    axis = 2 if support else 0
    # the locator touches the part face that faces it: the face at the minimum along n_D
    point = np.zeros(3)
    point[axis] = lo[axis]
    others = [k for k in range(3) if k != axis]
    for k in others:
        point[k] = min(max(0.0, lo[k]), hi[k])
    gap = 0.0
    if lo[axis] > P.p(P.WORK, "seat_tolerance_m"):
        gap = -lo[axis]
        out.notes.append(f"locator {label or ident}: {lo[axis] * 1e3:.3f} mm clear of the part")
    elif lo[axis] < -P.p(P.WORK, "seat_tolerance_m"):
        out.notes.append(f"locator {label or ident}: {-lo[axis] * 1e3:.3f} mm inside the part: taken at its surface")
    k = P.p(P.WORK, "locator_stiffness_N_per_m")
    out.patches.append(point_patch(f"{bc.op_name}/{j}:{label or ident}", "locator", LAW_LOCATOR, ident, j, T_H_D, point,
                                   n_D, k, _kt(), P.p(P.WORK, "base_friction"), gap=gap,
                                   datum="primary" if support else "secondary", locating=True))
    return out


def build_vacuum(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"vacuum{j}")
    got = _base_on_plate(bc, j, ident, "vacuum_plate", T_H_D, item, _jaw_friction(item, "base_friction"), out,
                         seal=True)
    if got is None:
        return out
    patch, area, (x_lo, x_hi, y_lo, y_hi, z_b, _) = got
    util = P.num(item, "area_utilisation")
    if util is None:
        util = P.p(P.WORK, "vacuum_area_utilisation")
    p_entry = P.num(entry, "pressure_Pa")
    if p_entry is not None:
        dp = STANDARD_ATMOSPHERE_PA - p_entry
        out.notes.append(f"vacuum {ident}: pressure_bar read as the absolute pressure reached")
    elif P.num(item, "min_abs_pressure_Pa") is not None:
        dp = STANDARD_ATMOSPHERE_PA - P.num(item, "min_abs_pressure_Pa")
    elif P.num(item, "vacuum_with_pump_Pa") is not None:
        dp = P.num(item, "vacuum_with_pump_Pa")
    elif P.num(item, "holding_pressure_Pa") is not None:
        dp = P.num(item, "holding_pressure_Pa")
    else:
        raise InputError(f"vacuum {ident}: no vacuum level in the database")
    dp = max(dp, 0.0)
    sealed = util * area
    force = dp * sealed
    centre = _to_H(T_H_D, [[0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), z_b]])[0]
    out.holddown.append({"force_H": force * _dir_H(T_H_D, [0, 0, -1]), "point_H": centre, "kind": "vacuum",
                         "device": ident, "entry": j, "area_m2": sealed, "pressure_Pa": dp, "patch": patch.id})
    out.info.update({"holddown_N": force, "pressure_Pa": dp, "sealed_area_m2": sealed})
    return out


def build_magnetic(bc: BuildCtx, j: int, entry, item, T_H_D, label: str, debris_m: float = 0.0) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"magnet{j}")
    got = _base_on_plate(bc, j, ident, "magnet_plate", T_H_D, item, _jaw_friction(item, "base_friction"), out)
    if got is None:
        return out
    patch, area, (x_lo, x_hi, y_lo, y_hi, z_b, z_t) = got
    thickness = z_t - z_b
    table = P.table(item, "holding_pressure_Pa")
    if table is not None:
        pressure = float(table(thickness))
    elif P.num(item, "max_holding_pressure_Pa") is not None:
        pressure = P.num(item, "max_holding_pressure_Pa")
        out.notes.append(f"magnetic {ident}: no thickness table: the maximum holding pressure is taken")
    elif P.num(item, "holding_pressure_Pa") is not None:
        pressure = P.num(item, "holding_pressure_Pa")
    else:
        raise InputError(f"magnetic {ident}: no holding pressure in the database")
    family = (bc.material_family or "").lower()
    factors = P.get(P.WORK)
    factor = float(factors.get(f"magnetic_factor_{family}", 0.0))
    if factor <= 0.0:
        out.notes.append(f"magnetic {ident}: the work material ({family or 'unknown'}) is not ferromagnetic: no "
                         f"magnetic holding")
    g_c = P.p(P.WORK, "magnetic_gap_m")
    gap_factor = 1.0 / (1.0 + max(debris_m, 0.0) / g_c) ** 2
    force = pressure * factor * gap_factor * area
    centre = _to_H(T_H_D, [[0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), z_b]])[0]
    out.holddown.append({"force_H": force * _dir_H(T_H_D, [0, 0, -1]), "point_H": centre, "kind": "magnetic",
                         "device": ident, "entry": j, "area_m2": area, "pressure_Pa": pressure * factor * gap_factor,
                         "patch": patch.id})
    out.info.update({"holddown_N": force, "pressure_Pa": pressure * factor * gap_factor, "thickness_m": thickness,
                     "material_factor": factor, "gap_factor": gap_factor})
    return out


def build_adhesive(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"adhesive{j}")
    family = (bc.material_family or "").lower()
    tau = P.num(item, "shear_strength_aluminium_Pa") if family == "aluminium" else None
    if tau is None:
        tau = P.num(item, "shear_strength_Pa")
    if tau is None:
        raise InputError(f"adhesive {ident}: no shear strength in the database")
    util = P.num(item, "bond_area_utilisation")
    util = 1.0 if util is None else util
    peel = P.num(item, "peel_strength_N_per_m")
    sigma_t = tau * P.p(P.WORK, "adhesive_tension_ratio")
    got = _base_on_plate(bc, j, ident, "bond", T_H_D, item, 0.0, out, law=LAW_BOND, cap_t=sigma_t * util,
                         cap_s=tau * util, k_scale=util)
    if got is None:
        return out
    patch, area, _ = got
    if peel is not None:
        out.info["peel_strength_N_per_m"] = peel
    out.info.update({"shear_strength_Pa": tau, "bonded_area_m2": area * util})
    return out


def build_tabs(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"tab{j}")
    dims = P.field_of(item, "dims_m")
    width = P.num(dims, "width") if dims is not None else None
    height = P.num(dims, "height") if dims is not None else None
    if width is None or height is None:
        raise InputError(f"tab {label or ident}: dims_mm {{width, height}} of the tab section are required")
    length = (P.num(dims, "length") if dims is not None else None) or width
    if bc.ultimate_Pa is None or bc.youngs_Pa is None:
        raise InputError("tabs: the work material needs physical.ultimate_MPa and youngs_modulus_GPa")
    area = width * height
    nu = bc.poisson_work if bc.poisson_work is not None else P.p(P.WORK, "contact_poisson")
    k_ax = bc.youngs_Pa * area / length
    shear_ratio = 1.0 / (2.0 * (1.0 + nu))          # G / E
    patch = rect_patch(f"{bc.op_name}/{j}:{label or ident}", "tab", LAW_BOND, ident, j, T_H_D, [0, 0, 0], [1, 0, 0],
                       [0, 1, 0], None, 0.5 * width, 0.5 * height, bc.n_side, k_ax, (shear_ratio, shear_ratio), 0.0,
                       cap_t=bc.ultimate_Pa, cap_s=VON_MISES_SHEAR * bc.ultimate_Pa)
    out.patches.append(patch)
    out.info.update({"tab_area_m2": area, "tab_length_m": length})
    return out


def _grip_side(entry) -> tuple[bool, bool]:
    """(inside gripping, axial stop) from the entry's jaws text: 'id'/'internal'/'inside' grip a bore,
    'no_stop'/'bar'/'through' remove the axial stop at the jaw steps."""
    text = (P.text(entry, "jaws", "") or "").lower().replace("-", "_")
    words = set(text.replace(",", " ").split())
    inside = bool(words & {"id", "internal", "inside", "bore"})
    stop = not bool(words & {"no_stop", "bar", "through", "nostop"})
    return inside, stop


def build_chuck(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"chuck{j}")
    kind = P.text(item, "type", "chuck_3jaw")
    collet = kind in ("collet_chuck", "collet")
    if bc.shape.kind != "cylinder":
        raise InputError(f"{ident}: a {kind} grips a cylindrical stock (the stock is {bc.shape.kind})")
    p0, p1, u = bc.axis_D(T_H_D)
    if np.linalg.norm(np.cross(u, [0.0, 0.0, 1.0])) > AXIS_TOL:
        raise InputError(f"{ident}: the stock axis is not the {kind} axis (angle "
                         f"{math.degrees(math.asin(min(np.linalg.norm(np.cross(u, [0, 0, 1])), 1.0))):.2f} deg)")
    if u[2] < 0.0:
        p0, p1, u = p1, p0, -u
    inside, stop = _grip_side(entry)
    radius = bc.shape.r_in_m if inside else bc.shape.radius_m
    if radius <= 0.0:
        raise InputError(f"{ident}: inside gripping needs a tube stock")
    n_jaws = int(P.num(item, "jaws") or (P.p(P.WORK, "collet_segments") if collet else 3))
    if collet:
        n_jaws = int(P.p(P.WORK, "collet_segments"))
        rng = P.numlist(item, "range_m")
        if rng and len(rng) == 2 and not (rng[0] - 1e-9 <= 2.0 * radius <= rng[1] + 1e-9):
            out.info["no_grip"] = True
            out.notes.append(f"{ident}: part diameter {2e3 * radius:.2f} mm outside the collet range "
                             f"{rng[0] * 1e3:.1f}-{rng[1] * 1e3:.1f} mm")
            return out
    soft = bc.soft_jaws
    grip = P.num(entry, "grip_depth_m") or (P.num(bc.soft_entry, "grip_depth_m") if bc.soft_entry is not None
                                            else None) or P.num(item, "grip_length_m") or \
        (P.num(soft, "jaw_length_m") if soft is not None else None)
    if grip is None:
        grip = P.p(P.WORK, "chuck_grip_length_m")
        out.notes.append(f"{ident}: grip length {grip * 1e3:.0f} mm taken (none given)")
    z_part0 = float(p0[2])
    z_start = max(z_part0, 0.0)
    z_end = min(z_start + grip, float(p1[2]))
    if z_end <= z_start:
        out.info["no_grip"] = True
        out.notes.append(f"{ident}: the part does not reach into the jaws")
        return out
    static = chuck_static_force(entry, item, out.notes)
    loss = 0.0 if collet else chuck_loss(item, bc.omega_rad_s, static, n_jaws, out.notes)
    dynamic = static + loss if inside else static - loss
    dynamic = max(dynamic, 0.0)
    mu = _jaw_friction(soft, "chuck_jaw_friction") if soft is not None and P.num(soft, "jaw_friction") is not None \
        else _jaw_friction(item, "chuck_jaw_friction")
    if collet:
        half_theta = 0.9 * math.pi / n_jaws
    else:
        jaw_w = (P.num(soft, "jaw_width_m") if soft is not None else None) or P.p(P.WORK, "chuck_jaw_width_m")
        half_theta = min(0.5 * jaw_w / radius, 0.9 * math.pi / n_jaws)
    k_jaw = P.p(P.WORK, "chuck_jaw_stiffness_N_per_m")
    axis_p = p0 - (p0[2]) * u            # axis point at z_D = 0
    base = f"{bc.op_name}/{j}:{ident}"
    for k in range(n_jaws):
        theta = 2.0 * math.pi * k / n_jaws
        out.patches.append(cylinder_strip(f"{base}/jaw{k}", "jaw", LAW_LOCKED, ident, j, T_H_D, axis_p, u, radius, theta,
                                          half_theta, z_start, z_end, bc.n_side, k_jaw, _kt(), mu,
                                          dynamic / n_jaws, inward=not inside, datum="secondary"))
    if stop and z_part0 >= -1e-9:
        r_stop = P.p(P.WORK, "chuck_stop_radius_ratio") * bc.shape.radius_m
        for k in range(n_jaws):
            theta = 2.0 * math.pi * k / n_jaws
            pt = axis_p + z_part0 * u + r_stop * np.array([math.cos(theta), math.sin(theta), 0.0])
            out.patches.append(point_patch(f"{base}/stop{k}", "stop", LAW_LOCATOR, ident, j, T_H_D, pt, u,
                                           P.p(P.WORK, "locator_stiffness_N_per_m"), _kt(), mu, datum="primary",
                                           locating=True))
    elif stop:
        out.notes.append(f"{ident}: the part reaches behind the chuck face: no axial stop at the jaw steps")
    out.info.update({"static_force_N": static, "loss_N": loss, "gripping_force_N": dynamic, "jaws": n_jaws,
                     "grip_length_m": z_end - z_start, "grip_radius_m": radius, "inside": inside, "friction": mu,
                     "axis_H": (_to_H(T_H_D, [axis_p])[0], _dir_H(T_H_D, u)), "max_rad_s": P.num(item, "max_rad_s"),
                     "jaw_stiffness_N_per_m": k_jaw, "z_face_D": 0.0})
    return out


def build_centre(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"centre{j}")
    if bc.shape.kind != "cylinder":
        raise InputError(f"{ident}: a tailstock centre supports a cylindrical stock")
    p0, p1, u = bc.axis_D(T_H_D)
    if np.linalg.norm(np.cross(u, [0.0, 0.0, 1.0])) > AXIS_TOL:
        raise InputError(f"{ident}: the stock axis is not the centre axis")
    if u[2] < 0.0:
        p0, p1, u = p1, p0, -u
    angle = P.num(item, "angle_rad") or math.radians(60.0)
    alpha_h = 0.5 * angle
    r_c = max(P.p(P.WORK, "centre_hole_radius_ratio") * bc.shape.radius_m, P.p(P.WORK, "centre_hole_min_radius_m"))
    thrust = None
    tail = P.field_of(bc.machine, "tailstock") if bc.machine is not None else None
    if tail is not None:
        rng = P.numlist(tail, "thrust_N")
        if rng:
            thrust = min(rng)
    if thrust is None:
        thrust = P.p(P.WORK, "tailstock_thrust_N")
        out.notes.append(f"{ident}: tailstock thrust {thrust / 1e3:.1f} kN taken (the machine gives none)")
    n_c = int(bc.centre_nodes)
    depth = r_c / math.tan(alpha_h)
    z_ring = float(p0[2]) + depth
    axis_p = p0 - p0[2] * u
    per_node = thrust / (n_c * math.sin(alpha_h))
    k_node = P.p(P.WORK, "centre_stiffness_N_per_m") / n_c
    pts, nrm, t1, t2 = [], [], [], []
    for k in range(n_c):
        th = 2.0 * math.pi * k / n_c
        er = np.array([math.cos(th), math.sin(th), 0.0])
        et = np.array([-math.sin(th), math.cos(th), 0.0])
        n = math.sin(alpha_h) * u + math.cos(alpha_h) * er
        pts.append(axis_p + z_ring * u + r_c * er)
        nrm.append(n)
        t1.append(et)
        t2.append(np.cross(n, et))
    w = np.full(n_c, 1.0 / n_c)
    pid = f"{bc.op_name}/{j}:{ident}/centre"
    patch = Patch(pid, "centre", LAW_LOCKED, ident, j, _to_H(T_H_D, pts), _dir_H(T_H_D, nrm), _dir_H(T_H_D, t1),
                  _dir_H(T_H_D, t2), w, math.pi * r_c * r_c, k_node * n_c,
                  (0.0, mindlin_ratio(P.p(P.WORK, "contact_poisson"))), P.p(P.WORK, "centre_friction"),
                  force_N=per_node * n_c, centre_H=_to_H(T_H_D, [axis_p + z_ring * u])[0], normal_H=-_dir_H(T_H_D, u),
                  half_m=np.array([r_c, r_c]), rotating=True, slides_while_clamping=True)
    out.patches.append(patch)
    out.info.update({"thrust_N": thrust, "contact_radius_m": r_c, "half_angle_rad": alpha_h,
                     "radial_stiffness_N_per_m": P.p(P.WORK, "centre_stiffness_N_per_m"),
                     "max_rad_s": P.num(item, "max_rad_s"), "live": bool(P.plain(P.field_of(item, "live"))),
                     "point_H": _to_H(T_H_D, [axis_p + float(p0[2]) * u])[0]})
    return out


def build_steady(bc: BuildCtx, j: int, entry, item, T_H_D, label: str) -> DeviceOut:
    out = DeviceOut()
    ident = P.text(item, "id", f"steady{j}")
    if bc.shape.kind != "cylinder":
        raise InputError(f"{ident}: a steady rest supports a cylindrical stock")
    p0, p1, u = bc.axis_D(T_H_D)
    if np.linalg.norm(np.cross(u, [0.0, 0.0, 1.0])) > AXIS_TOL:
        raise InputError(f"{ident}: the stock axis is not the steady-rest axis")
    if u[2] < 0.0:
        p0, p1, u = p1, p0, -u
    if not (min(p0[2], p1[2]) - 1e-9 <= 0.0 <= max(p0[2], p1[2]) + 1e-9):
        out.info["no_grip"] = True
        out.notes.append(f"{ident}: the rollers' plane is outside the part")
        return out
    radius = bc.shape.radius_m
    grip = P.numlist(item, "grip_range_m")
    if grip and len(grip) == 2 and not (grip[0] - 1e-9 <= 2.0 * radius <= grip[1] + 1e-9):
        out.notes.append(f"{ident}: part diameter {2e3 * radius:.2f} mm outside the grip range")
    n_r = int(P.num(item, "rollers") or 3)
    pressure = P.num(entry, "pressure_Pa")
    cur = P.curve(item, "clamp_force_per_roller_vs_pressure_N")
    f_max = P.num(item, "clamp_force_per_roller_N")
    if pressure is not None and cur is not None and len(cur[0]) >= 2:
        force = float(np.interp(pressure, cur[0], cur[1]))
    elif pressure is not None and f_max is not None and P.numlist(item, "pressure_Pa"):
        force = f_max * min(pressure / max(P.numlist(item, "pressure_Pa")), 1.0)
    elif f_max is not None:
        force = f_max
        if pressure is not None:
            out.notes.append(f"{ident}: no force-pressure relation: the maximum roller force is taken")
    else:
        raise InputError(f"{ident}: no roller clamping force in the database")
    k = P.num(item, "roller_stiffness_N_per_m") or P.p(P.WORK, "chuck_jaw_stiffness_N_per_m")
    axis_p = p0 - p0[2] * u
    for r in range(n_r):
        theta = 0.5 * math.pi + 2.0 * math.pi * r / n_r
        er = np.array([math.cos(theta), math.sin(theta), 0.0])
        pt = axis_p + radius * er
        patch = point_patch(f"{bc.op_name}/{j}:{ident}/roller{r}", "roller", LAW_LOCKED, ident, j, T_H_D, pt, -er, k,
                            (0.0, mindlin_ratio(P.p(P.WORK, "contact_poisson"))), P.p(P.WORK, "roller_axial_friction"),
                            force=force)
        # rolling direction e_theta is frictionless: t1 = e_theta, t2 = axial
        et = np.array([-math.sin(theta), math.cos(theta), 0.0])
        patch.t1_H = np.tile(_dir_H(T_H_D, et), (1, 1))
        patch.t2_H = np.tile(_dir_H(T_H_D, np.cross(-er, et)), (1, 1))
        patch.rotating = True
        out.patches.append(patch)
    out.info.update({"roller_force_N": force, "rollers": n_r, "roller_stiffness_N_per_m": k,
                     "point_H": _to_H(T_H_D, [axis_p])[0]})
    return out


BUILDERS = {
    "vise": build_vise, "clamp": build_clamp, "fixture_plate": build_fixture_plate, "vacuum": build_vacuum,
    "magnetic": build_magnetic, "adhesive": build_adhesive, "chuck_3jaw": build_chuck, "chuck_4jaw": build_chuck,
    "collet_chuck": build_chuck, "collet": build_chuck, "tailstock_centre": build_centre, "steady_rest": build_steady,
    "tabs": build_tabs,
}

__all__ = ["Patch", "DeviceOut", "BuildCtx", "BUILDERS", "rect_patch", "point_patch", "cylinder_strip",
           "simpson_weights", "mindlin_ratio", "screw_force", "vise_force", "clamp_force", "chuck_static_force",
           "chuck_loss", "build_point_locator", "LAW_LOCATOR", "LAW_LOCKED", "LAW_BOND", "LOCATOR_LABELS",
           "SUPPORT_LABELS", "ROTATING_TYPES"]
