"""The operating point of a cut and its linearised cut model (docs/models/dynamics.md section 5.4; L2).

From the previous step's bus records: the engaged edge elements (cutting's per-element in-cut time, else the
stations within the axial depth), the engaged arc of a rotating tool (removal's engaged angle around the
centroid direction of the engaged region, in T0), the feed per tooth (motion's feed velocity and the spindle
speed), the mean chip thickness over the arc, and the incremental cutting coefficients d(K h)/dh of each
element's group at that thickness (cutting's ProcessCoefficients). The element matrices follow
cnctwin.dynamics.sld: ``M_e = -(db_e / a_p)(K_t c_e + K_r n_e - K_a a_e) n_e^T`` in T at spindle angle 0, with the
element frames of cnctwin.tooling (EdgeFrameT); the process-damping matrices ``K_sp db L_w^2 / (4 a_p) n n^T``
per unit cutting speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .sld import TWO_PI, CutModel


@dataclass
class OperatingPoint:
    path: int
    slot: int
    tick: int
    speed_rad_s: float              # |omega| (step mean)
    ap_m: float
    ae_m: float
    arc_lo_rad: float
    arc_hi_rad: float
    feed_per_tooth_m: float
    h_mean_m: float
    n_teeth: int
    rotating: bool
    spin: float
    signature: tuple = ()           # what makes a new diagram (immersion, coefficients, structure)
    cut: CutModel | None = None
    notes: list = field(default_factory=list)


def incremental_coefficient(h_grid_m: np.ndarray, K_Pa: np.ndarray, h_m: float) -> float:
    """d(K h)/dh at h (K tabulated on a log-spaced ascending grid; clamped to the grid)."""
    h = np.asarray(h_grid_m, float)
    K = np.asarray(K_Pa, float)
    if len(h) == 0:
        return 0.0
    if len(h) == 1:
        return float(K[0])
    hm = min(max(float(h_m), float(h[0])), float(h[-1]))
    lo = max(hm * 0.95, float(h[0]))
    hi = min(hm * 1.05, float(h[-1]))
    if hi <= lo:
        return float(np.interp(hm, h, K))
    f_lo = float(np.interp(lo, h, K)) * lo
    f_hi = float(np.interp(hi, h, K)) * hi
    return (f_hi - f_lo) / (hi - lo)


def _arc_from_geometry(R_S_T0: np.ndarray, tip_S: np.ndarray, point_S: np.ndarray,
                       engaged_angle_rad: float) -> tuple[float, float]:
    d = R_S_T0.T @ (np.asarray(point_S, float) - np.asarray(tip_S, float))
    centre = math.atan2(d[1], d[0])
    half = 0.5 * min(max(float(engaged_angle_rad), 0.0), TWO_PI)
    if half >= math.pi - 1e-12:
        return 0.0, TWO_PI
    return centre - half, centre + half


def mean_chip_thickness(fz_m: float, feed_angle_rad: float, lo: float, hi: float) -> float:
    """Mean of f_z max(cos(alpha - alpha_feed), 0) over the arc [lo, hi] (straight-cut chip thickness)."""
    if hi <= lo or fz_m <= 0.0:
        return 0.0
    a = np.linspace(lo, hi, 721)
    return float(fz_m * np.mean(np.maximum(np.cos(a - feed_angle_rad), 0.0)))


def cut_model(edge, frames, coeffs, *, engaged: np.ndarray, ap_m: float, h_mean_m: float, arc: tuple,
              rotating: bool, spin: float, K_sp_scale: float, land_per_radius: float, VB_m: np.ndarray | None,
              work_radius_m: float) -> CutModel:
    """CutModel of the engaged elements (module doc). engaged [nt, ne] bool."""
    nt, ne = np.asarray(edge.elem_r_m).shape
    hand = float(np.asarray(edge.hand_sign).reshape(-1)[0]) if getattr(edge, "hand_sign", None) is not None else -1.0
    h_grid = np.asarray(coeffs.h_grid_m, float)
    n_groups = np.asarray(coeffs.Kt_Pa).shape[0]
    mats, damp, alpha0, tooth, radius = [], [], [], [], []
    for j in range(nt):
        for k in range(ne):
            if not bool(edge.elem_valid[j, k]) or not bool(engaged[j, k]):
                continue
            g = min(max(int(edge.elem_group[j, k]), 0), n_groups - 1)
            Kt = incremental_coefficient(h_grid, coeffs.Kt_Pa[g], h_mean_m)
            Kr = incremental_coefficient(h_grid, coeffs.Kr_Pa[g], h_mean_m)
            Ka = incremental_coefficient(h_grid, coeffs.Ka_Pa[g], h_mean_m)
            a0 = hand * (float(edge.tooth_phase_rad[j]) + float(edge.elem_lag_rad[j, k])) if rotating else 0.0
            if frames is not None:
                n = np.asarray(frames.normal[j, k], float)
                c = np.asarray(frames.cut_dir[j, k], float)
                t = np.asarray(frames.tangent[j, k], float)
                inc = float(edge.elem_incl_rad[j, k])
                ci = math.cos(inc)
                a = -(t + math.sin(inc) * c) / ci if abs(ci) > 1e-6 else np.zeros(3)
            else:
                kap = float(edge.elem_kappa_rad[j, k])
                n = np.array([math.sin(kap) * math.cos(a0), math.sin(kap) * math.sin(a0), -math.cos(kap)])
                c = hand * np.array([-math.sin(a0), math.cos(a0), 0.0])
                a = np.zeros(3)
                Ka = 0.0
            db = float(edge.elem_db_m[j, k])
            mats.append(-(db / ap_m) * np.outer(Kt * c + Kr * n - Ka * a, n))
            vb = float(VB_m[j, k]) if VB_m is not None and np.shape(VB_m) == (nt, ne) else 0.0
            L = vb + land_per_radius * float(edge.elem_edge_radius_m[j, k])
            Ksp = float(K_sp_scale[g]) if np.ndim(K_sp_scale) else float(K_sp_scale)
            damp.append(Ksp * db * L * L / (4.0 * ap_m) * np.outer(n, n))
            alpha0.append(a0)
            tooth.append(j)
            radius.append(float(edge.elem_r_m[j, k]))
    phases = np.asarray(edge.tooth_phase_rad, float)
    pitch = np.full(nt, TWO_PI / max(nt, 1))
    if rotating and nt > 1:
        # tooth j's delay: the angle (cutting sense) to the tooth that passes a point just before it, which is the
        # tooth with the next larger phase (phases are in the cutting sense)
        for j in range(nt):
            d = (phases - phases[j]) % TWO_PI
            d[j] = TWO_PI
            pitch[j] = float(np.min(d))
    M = np.array(mats) if mats else np.zeros((0, 3, 3))
    D = np.array(damp) if damp and np.any(np.array(damp) != 0.0) else np.zeros((0, 3, 3))
    return CutModel(M=M, alpha0_rad=np.array(alpha0), tooth=np.array(tooth, int), n_teeth=max(nt, 1) if rotating else 1,
                    pitch_rad=pitch if rotating else np.array([TWO_PI]), arc_lo_rad=float(arc[0]),
                    arc_hi_rad=float(arc[1]), spin=float(spin), rotating=bool(rotating), damping_M=D,
                    radius_m=np.array(radius), work_radius_m=float(work_radius_m))


__all__ = ["OperatingPoint", "incremental_coefficient", "mean_chip_thickness", "cut_model", "_arc_from_geometry"]
