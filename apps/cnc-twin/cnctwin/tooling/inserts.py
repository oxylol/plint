"""Insert geometry shared by indexable milling cutters, boring heads and turning tools.

Tool angles of an insert in its seat follow the classic relations between the axial (back) rake
gamma_p, the radial (side) rake gamma_f and the entering angle kappa_r (Armarego & Brown 1969;
ISO 3002-1):

    tan(lambda_s) = tan(gamma_p) sin(kappa_r) - tan(gamma_f) cos(kappa_r)
    tan(gamma_n)  = (tan(gamma_f) sin(kappa_r) + tan(gamma_p) cos(kappa_r)) cos(lambda_s)

The seat puts the insert's rake face on a plane through the corner; every element of the insert
(major edge, nose, minor edge, wiper) then gets its own normal rake, clearance and inclination
from that plane and its local edge direction (cnctwin.tooling.edges.face_angles). The insert's
own normal clearance alpha_i (ISO 1832 symbol 2) gives the flank: alpha_n = alpha_i - gamma_seat.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.errors import InputError
from . import params as P
from .spec import ToolSpec, num, sub, text

HALF_PI = 0.5 * math.pi


def seat_angles(gamma_p: float, gamma_f: float, kappa_r: float) -> tuple[float, float]:
    """(normal rake, inclination) of the major edge from axial rake, radial rake, entering angle."""
    lam = math.atan(math.tan(gamma_p) * math.sin(kappa_r) - math.tan(gamma_f) * math.cos(kappa_r))
    gam = math.atan((math.tan(gamma_f) * math.sin(kappa_r) + math.tan(gamma_p) * math.cos(kappa_r)) * math.cos(lam))
    return gam, lam


def face_planes(kappa: float, rake_n: float, incl: float, clear_n: float) -> tuple[np.ndarray, np.ndarray]:
    """Outward rake-face and flank-face normals (tooth frame C, element at psi = 0) of an edge whose
    meridian direction has angle ``kappa`` and which has normal rake ``rake_n``, inclination
    ``incl`` and normal clearance ``clear_n``."""
    e_r = np.array([1.0, 0.0, 0.0])
    e_v = np.array([0.0, 1.0, 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    t_m = math.cos(kappa) * e_r + math.sin(kappa) * e_z
    n = math.sin(kappa) * e_r - math.cos(kappa) * e_z
    t3 = math.cos(incl) * t_m - math.sin(incl) * e_v      # positive inclination: the edge trails going up
    m = e_v - float(np.dot(e_v, t3)) * t3
    m /= np.linalg.norm(m)
    u_rake = math.cos(rake_n) * (-n) - math.sin(rake_n) * m
    u_flank = -math.cos(clear_n) * m - math.sin(clear_n) * n
    N_r = np.cross(t3, u_rake)
    if float(np.dot(N_r, m)) < 0.0:
        N_r = -N_r
    N_f = np.cross(t3, u_flank)
    if float(np.dot(N_f, n)) < 0.0:
        N_f = -N_f
    return N_r / np.linalg.norm(N_r), N_f / np.linalg.norm(N_f)


def psi_on_plane(z: np.ndarray, r: np.ndarray, N_r: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Angular position psi (near 0) where the point at height z and radius r lies on the rake plane
    through Q with normal N_r: r (cos psi N_x + sin psi N_y) = Q.N_r - z N_z."""
    A = r * N_r[0]
    B = r * N_r[1]
    C = float(np.dot(Q, N_r)) - z * N_r[2]
    norm = np.hypot(A, B)
    out = np.zeros_like(r)
    ok = norm > 0.0
    base = np.arctan2(B[ok], A[ok])
    ratio = np.clip(C[ok] / norm[ok], -1.0, 1.0)
    spread = np.arccos(ratio)
    cand1 = base - spread
    cand2 = base + spread
    cand1 = np.arctan2(np.sin(cand1), np.cos(cand1))
    cand2 = np.arctan2(np.sin(cand2), np.cos(cand2))
    out[ok] = np.where(np.abs(cand1) <= np.abs(cand2), cand1, cand2)
    return out


def included_angle(spec: ToolSpec, insert) -> float | None:
    """Corner angle of the insert: insert.included_angle_deg, else from the shape symbol (ISO 1832);
    None for round inserts."""
    value = num(insert, "included_angle_rad")
    if value is not None:
        return value
    shape = text(insert, "shape") or _shape_from_code(text(insert, "iso_code"))
    if shape is None:
        return None
    shape = shape.upper()
    if shape == "R":
        return None
    symbols = P.p("insert_shape_symbols")
    if shape not in symbols:
        raise InputError(f"tool {spec.tool_id}: insert shape {shape!r} needs insert.included_angle_deg")
    return float(P.p("insert_included_angle_rad")[symbols.index(shape)])


def _shape_from_code(code: str | None) -> str | None:
    if code and code[0].isalpha():
        return code[0].upper()
    return None


def insert_clearance(spec: ToolSpec, insert) -> float:
    value = num(insert, "clearance_rad")
    if value is not None:
        return value
    code = text(insert, "iso_code")
    symbols = P.p("insert_clearance_symbols")
    values = P.p("insert_clearance_rad")
    if code and len(code) >= 2 and code[1].upper() in symbols:
        return float(values[symbols.index(code[1].upper())])
    value = float(values[symbols.index("C")])
    spec.notes.append(f"insert clearance not given: {math.degrees(value):.0f} deg (ISO symbol C)")
    return value


def edge_length(spec: ToolSpec, insert, corner: float | None) -> float:
    """Usable cutting-edge length: insert.edge_length_mm, else from the inscribed circle and the
    corner angle (rhombus side ic / sin(eps), triangle sqrt(3) ic, square ic, regular polygon
    ic tan(pi / n))."""
    value = num(insert, "edge_length_m")
    if value is not None:
        return value
    ic = num(insert, "ic_m")
    if ic is None:
        width = num(insert, "width_m")
        if width is not None:
            return width
        raise InputError(f"tool {spec.tool_id}: insert needs edge_length_mm or ic_mm")
    if corner is None:
        return ic
    eps = corner
    shape = (text(insert, "shape") or _shape_from_code(text(insert, "iso_code")) or "").upper()
    if shape in ("T", "W"):
        return math.sqrt(3.0) * ic
    if shape in ("S", "L"):
        return ic
    if shape in ("H", "O", "P"):
        sides = {"H": 6, "O": 8, "P": 5}[shape]
        return ic * math.tan(math.pi / sides)
    return ic / math.sin(eps)


def chip_breaker_rake(insert) -> float:
    breaker = sub(insert, "chip_breaker")
    if breaker is None:
        return 0.0
    return num(breaker, "rake_rad", 0.0) or 0.0


def wiper_length(spec: ToolSpec, insert, base: float, nose: float | None = None, turning: bool = False) -> float:
    value = num(insert, "wiper_flat_m")
    if value is not None:
        return value
    if turning and nose:
        value = P.pf("turning_wiper_flat_per_nose") * nose
    else:
        value = P.pf("wiper_flat_per_ic") * base
    spec.notes.append(f"wiper flat length not given: {value * 1e3:.2f} mm")
    return value
