"""Long turned parts: a stepped Timoshenko beam on chuck, tailstock and steady-rest supports
(docs/models/structure.md section 3.6).

A cylinder or tube stock held in a chuck (holding's ``PartHolding.lathe``) whose free length from the chuck face
exceeds ``lathe_slenderness`` diameters is modelled as a beam along S z instead of voxels (a slender shaft on a
coarse voxel grid has too few cells across its section). Sections come from removal's material fractions on a
fine grid (``lathe_radial_cells`` cells per radius): area, centroid and second moments of every slice (cell
moments ``f c^2 y^2 + f c^4 / 12``), the in-plane mean ``(I_1 + I_2) / 2`` for both planes (the part rotates:
the time-invariant mean stiffness in the non-rotating frame, Genta 2005), Cowper's hollow-circle shear
coefficient of the equivalent annulus; slices are grouped into elements (stiffness in series, mass averaged).
Supports: the chuck as distributed lateral springs over the grip length (holding's radial stiffness) plus the
tilt stiffness holding gives beyond that distribution, the axial stop at the chuck face and the torsional grip at
the grip centre; tailstock and steady rest as lateral springs (a rigid pin when holding gives no stiffness).

Displacements on the beam: free thermal strain and thermal curvature of every slice (Boley & Weiner 1960) and the
residual-stress eigenstrain ``-sigma0 / E`` of the remaining section (axial strain and curvature from its
unbalanced resultants), solved as held; mapped to the node grid as ``u(z) + theta(z) x r`` plus the thermal
radial expansion ``alpha dT r``. Clamping distortion of a long bar (local crushing under the jaws) is not
modelled (zero).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from ..core.errors import InputError
from . import beam as B
from . import modal as MO
from .fixture import skew

MAX_ELEMENTS = 80


@dataclass
class LatheSupports:
    chuck: dict | None = None
    tailstock: dict | None = None
    steady: list = field(default_factory=list)

    @classmethod
    def from_holding(cls, lathe: dict | None) -> "LatheSupports":
        lathe = lathe or {}
        return cls(lathe.get("chuck"), lathe.get("tailstock"), list(lathe.get("steady") or []))

    def to_dict(self) -> dict:
        return {"chuck": self.chuck, "tailstock": self.tailstock, "steady": list(self.steady)}


def free_length(lo, hi, sup: LatheSupports) -> tuple[float, float]:
    """(free length from the chuck face to the far end, outer diameter)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    D = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    if sup.chuck is None:
        return float(hi[2] - lo[2]), D
    zf = float(sup.chuck.get("z_face_S_m", lo[2]))
    return float(max(hi[2] - zf, zf - lo[2])), D


@dataclass
class Slices:
    z_m: np.ndarray            # slice centres
    dz_m: float
    A_m2: np.ndarray
    cx_m: np.ndarray
    cy_m: np.ndarray
    Ixx_m4: np.ndarray         # int (y - cy)^2 dA
    Iyy_m4: np.ndarray
    Ixy_m4: np.ndarray
    fill: np.ndarray           # [nx, ny, nz] fractions of the fine grid
    origin: np.ndarray
    cell_m: float


def section_slices(geometry, workpiece: int, lo, hi, radial_cells: int) -> Slices:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    R = 0.5 * float(max(hi[0] - lo[0], hi[1] - lo[1]))
    c = R / float(radial_cells)
    shape = tuple(int(max(1, math.ceil((hi[a] - lo[a]) / c - 1e-9))) for a in range(3))
    f = np.clip(np.asarray(geometry.fill_fraction(workpiece, lo, c, shape), dtype=float), 0.0, 1.0)
    x = lo[0] + (np.arange(shape[0]) + 0.5) * c
    y = lo[1] + (np.arange(shape[1]) + 0.5) * c
    da = c * c
    A = f.sum(axis=(0, 1)) * da
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(A > 0, np.einsum("ijk,i->k", f, x) * da / np.where(A > 0, A, 1.0), 0.0)
        cy = np.where(A > 0, np.einsum("ijk,j->k", f, y) * da / np.where(A > 0, A, 1.0), 0.0)
    X = x[:, None, None] - cx[None, None, :]
    Y = y[None, :, None] - cy[None, None, :]
    own = f.sum(axis=(0, 1)) * c ** 4 / 12.0
    Ixx = np.sum(f * Y * Y, axis=(0, 1)) * da + own
    Iyy = np.sum(f * X * X, axis=(0, 1)) * da + own
    Ixy = np.sum(f * X * Y, axis=(0, 1)) * da
    z = lo[2] + (np.arange(shape[2]) + 0.5) * c
    return Slices(z, c, A, cx, cy, Ixx, Iyy, Ixy, f, lo, c)


@dataclass
class LatheModel:
    node_z_m: np.ndarray
    K: np.ndarray
    M: np.ndarray
    K_springs: np.ndarray
    free: np.ndarray                     # free DOFs
    elements: list                       # (i, j, Section)
    axis_xy_m: np.ndarray                # beam axis (x, y) in S
    slices: Slices
    element_slices: list                 # slice indices of every element
    E_Pa: float
    nu: float
    rho_kg_m3: float
    notes: list = field(default_factory=list)


def _annulus(A: float, I: float) -> tuple[float, float]:
    """(outer, inner) diameter of the annulus with area A and second moment I (I = A (r_o^2 + r_i^2) / 4)."""
    s = 4.0 * I / A
    d = A / math.pi
    ro2 = max(0.5 * (s + d), 1e-30)
    ri2 = max(0.5 * (s - d), 0.0)
    return 2.0 * math.sqrt(ro2), 2.0 * math.sqrt(ri2)


def build_lathe(geometry, workpiece: int, lo, hi, mat, sup: LatheSupports, settings) -> LatheModel:
    sl = section_slices(geometry, workpiece, lo, hi, int(settings.lathe_radial_cells))
    present = np.nonzero(sl.A_m2 > 1e-12 * max(float(sl.A_m2.max()), 1e-300))[0]
    if len(present) < 2:
        raise InputError("structure: the turned part has fewer than two material slices")
    k0, k1 = int(present[0]), int(present[-1]) + 1
    n_sl = k1 - k0
    n_el = min(MAX_ELEMENTS, n_sl)
    bounds = np.round(np.linspace(k0, k1, n_el + 1)).astype(np.int64)
    G = B.shear_modulus(mat.E_Pa, mat.nu)
    z_nodes = sl.origin[2] + bounds * sl.cell_m
    elements = []
    el_slices = []
    asm = B.BeamAssembly(len(z_nodes))
    for e in range(n_el):
        idx = np.arange(bounds[e], bounds[e + 1])
        A = np.maximum(sl.A_m2[idx], 1e-12 * float(sl.A_m2.max()))
        I = np.maximum(0.5 * (sl.Ixx_m4[idx] + sl.Iyy_m4[idx]), 1e-30)
        A_s = 1.0 / np.mean(1.0 / A)
        I_s = 1.0 / np.mean(1.0 / I)
        d_o, d_i = _annulus(float(np.mean(A)), float(np.mean(I)))
        k = B.cowper_circle(mat.nu, d_i / d_o if d_o > 0.0 else 0.0)
        sec = B.Section(A_s, I_s, I_s, 2.0 * I_s, k, k, mat.E_Pa, G, mat.rho_kg_m3, mass_A_m2=float(np.mean(A)),
                        mass_Ix_m4=float(np.mean(I)), mass_Iy_m4=float(np.mean(I)))
        asm.add_element(e, e + 1, float(z_nodes[e + 1] - z_nodes[e]), sec)
        elements.append((e, e + 1, sec))
        el_slices.append(idx)
    n = asm.K.shape[0]
    Ks = np.zeros((n, n))
    fixed: list[int] = []
    notes: list[str] = []

    def nearest(z):
        return int(np.argmin(np.abs(z_nodes - z)))

    if sup.chuck is not None:
        ch = sup.chuck
        zc = float(ch.get("grip_centre_S_m", z_nodes[0]))
        Lg = float(ch.get("grip_length_m", 0.0)) or float(sl.cell_m)
        k_rad = float(ch.get("radial_stiffness_N_per_m", 0.0))
        k_tilt = float(ch.get("tilt_stiffness_Nm_per_rad", 0.0))
        a, b = zc - 0.5 * Lg, zc + 0.5 * Lg
        if k_rad > 0.0:
            kp = k_rad / Lg
            for e, (i, j, _s) in enumerate(elements):
                lo_e, hi_e = max(z_nodes[i], a), min(z_nodes[j], b)
                if hi_e <= lo_e:
                    continue
                share = (hi_e - lo_e) / (z_nodes[j] - z_nodes[i])
                Kf = B.foundation(float(z_nodes[j] - z_nodes[i]), kp * share, 0.0, 0.0)
                asm.add_ground_coupling(i, j, Kf, stiffness_into=Ks)
            extra = k_tilt - kp * Lg ** 3 / 12.0
            if extra > 0.0:
                g = nearest(zc)
                for d in (B.RX, B.RY):
                    asm.K[6 * g + d, 6 * g + d] += extra
                    Ks[6 * g + d, 6 * g + d] += extra
        else:
            for g in range(len(z_nodes)):
                if a - 1e-12 <= z_nodes[g] <= b + 1e-12:
                    fixed.extend(6 * g + d for d in (B.UX, B.UY, B.RX, B.RY))
            notes.append("chuck without stiffness: the grip is rigid")
        fixed.append(6 * nearest(float(ch.get("z_face_S_m", a))) + B.UZ)
        fixed.append(6 * nearest(zc) + B.RZ)
    else:
        fixed.extend(range(6))
        notes.append("no chuck: the part is clamped at its lower end (S z) for the beam model")
    for name, entry, k_key in (("tailstock", sup.tailstock, "radial_stiffness_N_per_m"),):
        if entry is None:
            continue
        g = nearest(float(entry.get("z_S_m", z_nodes[-1])))
        k = float(entry.get(k_key, 0.0))
        if k > 0.0:
            for d in (B.UX, B.UY):
                asm.K[6 * g + d, 6 * g + d] += k
                Ks[6 * g + d, 6 * g + d] += k
        else:
            fixed.extend(6 * g + d for d in (B.UX, B.UY))
            notes.append(f"{name} without stiffness: a rigid pin")
    for st in sup.steady:
        g = nearest(float(st.get("z_S_m", z_nodes[-1])))
        k = 0.5 * float(st.get("rollers", 3)) * float(st.get("stiffness_N_per_m", 0.0))
        if k > 0.0:
            for d in (B.UX, B.UY):
                asm.K[6 * g + d, 6 * g + d] += k
                Ks[6 * g + d, 6 * g + d] += k
        else:
            fixed.extend(6 * g + d for d in (B.UX, B.UY))
            notes.append("steady rest without stiffness: a rigid pin")
    free = np.setdiff1d(np.arange(n), np.unique(np.asarray(fixed, np.int64)))
    axis = np.array([float(np.average(sl.cx_m[k0:k1], weights=sl.A_m2[k0:k1])),
                     float(np.average(sl.cy_m[k0:k1], weights=sl.A_m2[k0:k1]))])
    return LatheModel(z_nodes, asm.K, asm.M, Ks, free, elements, axis, sl, el_slices, mat.E_Pa, mat.nu,
                      mat.rho_kg_m3, notes)


def lathe_modes(model: LatheModel, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """(lambda, full-length mass-normalised canonical vectors) of the held beam."""
    f = model.free
    K = model.K[np.ix_(f, f)]
    M = model.M[np.ix_(f, f)]
    axes = np.zeros((len(f), 3))
    for a in range(3):
        axes[:, a] = (f % 6 == a).astype(float)
    lam, V = MO.dense_modes(K, M, axes)
    keep = lam > 0.0
    lam, V = lam[keep][:n_modes], V[:, keep][:, :n_modes]
    full = np.zeros((model.K.shape[0], V.shape[1]))
    full[f] = V
    return lam, full


def static_compliance6(model: LatheModel) -> np.ndarray:
    """[n_nodes, 6, 6] static compliance of every node's DOFs (zero rows for fixed DOFs)."""
    f = model.free
    Kinv = sla.inv(model.K[np.ix_(f, f)])
    n = model.K.shape[0]
    full = np.zeros((n, n))
    full[np.ix_(f, f)] = 0.5 * (Kinv + Kinv.T)
    nn = len(model.node_z_m)
    return np.stack([full[6 * i:6 * i + 6, 6 * i:6 * i + 6] for i in range(nn)])


def interpolate_nodes(model: LatheModel, values: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear interpolation in z of node vectors values [n_nodes, ...] (held at the ends)."""
    zn = model.node_z_m
    t = np.clip(np.searchsorted(zn, z, side="right") - 1, 0, len(zn) - 2)
    w = np.clip((z - zn[t]) / (zn[t + 1] - zn[t]), 0.0, 1.0)
    shape = (-1,) + (1,) * (values.ndim - 1)
    return values[t] * (1.0 - w).reshape(shape) + values[t + 1] * w.reshape(shape)


def field_at(model: LatheModel, u_full: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Displacement u(z) + theta(z) x (r - axis) at points [n, 3] from a beam DOF vector."""
    nodes = u_full.reshape(-1, 6)
    vals = interpolate_nodes(model, nodes, points[:, 2])
    r = points - np.array([model.axis_xy_m[0], model.axis_xy_m[1], 0.0])
    r[:, 2] = 0.0
    return vals[:, :3] + np.cross(vals[:, 3:], r)


def compliance_at(model: LatheModel, C6_nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """[n, 3, 3] translational compliance at points from node 6 x 6 compliances."""
    C = interpolate_nodes(model, C6_nodes, points[:, 2])
    out = np.zeros((len(points), 3, 3))
    for p in range(len(points)):
        r = points[p] - np.array([model.axis_xy_m[0], model.axis_xy_m[1], points[p, 2]])
        A = np.hstack([np.eye(3), -skew(r)])
        out[p] = A @ C[p] @ A.T
    return out


def eigen_loads(model: LatheModel, strain: np.ndarray, curv_xz: np.ndarray, curv_yz: np.ndarray) -> np.ndarray:
    """Beam loads of element eigenstrains (axial strain, curvatures d2ux/dz2 and d2uy/dz2) [n_el]."""
    n = model.K.shape[0]
    f = np.zeros(n)
    for e, (i, j, sec) in enumerate(model.elements):
        EA = sec.E_Pa * sec.A_m2
        f[6 * i + B.UZ] -= EA * strain[e]
        f[6 * j + B.UZ] += EA * strain[e]
        EIy = sec.E_Pa * sec.Iy_m4
        EIx = sec.E_Pa * sec.Ix_m4
        f[6 * i + B.RY] -= EIy * curv_xz[e]
        f[6 * j + B.RY] += EIy * curv_xz[e]
        # y-z plane: the rotation DOF is -rx
        f[6 * i + B.RX] += EIx * curv_yz[e]
        f[6 * j + B.RX] -= EIx * curv_yz[e]
    return f


def slice_eigenstrains(model: LatheModel, eps_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Element (axial strain, curvature xz, curvature yz) from a free strain per fine-grid cell
    [nx, ny, nz]: area means and first moments over each element's slices (module doc)."""
    sl = model.slices
    c = sl.cell_m
    x = sl.origin[0] + (np.arange(sl.fill.shape[0]) + 0.5) * c
    y = sl.origin[1] + (np.arange(sl.fill.shape[1]) + 0.5) * c
    n_el = len(model.elements)
    eps0, kxz, kyz = np.zeros(n_el), np.zeros(n_el), np.zeros(n_el)
    for e, idx in enumerate(model.element_slices):
        fa = sl.fill[:, :, idx]
        ea = eps_cells[:, :, idx]
        A = float(fa.sum()) * c * c
        if A <= 0.0:
            continue
        cx = float(np.average(sl.cx_m[idx], weights=np.maximum(sl.A_m2[idx], 1e-300)))
        cy = float(np.average(sl.cy_m[idx], weights=np.maximum(sl.A_m2[idx], 1e-300)))
        X = (x - cx)[:, None, None]
        Y = (y - cy)[None, :, None]
        eps0[e] = float(np.sum(fa * ea)) * c * c / A
        Iyy = float(np.sum(fa * X * X)) * c * c
        Ixx = float(np.sum(fa * Y * Y)) * c * c
        kxz[e] = -float(np.sum(fa * ea * X)) * c * c / Iyy if Iyy > 0.0 else 0.0
        kyz[e] = -float(np.sum(fa * ea * Y)) * c * c / Ixx if Ixx > 0.0 else 0.0
    return eps0, kxz, kyz


def solve_held(model: LatheModel, f: np.ndarray) -> np.ndarray:
    fr = model.free
    u = np.zeros(model.K.shape[0])
    u[fr] = sla.solve(model.K[np.ix_(fr, fr)], f[fr], assume_a="pos", check_finite=False)
    return u


__all__ = ["LatheSupports", "LatheModel", "Slices", "free_length", "section_slices", "build_lathe", "lathe_modes",
           "static_compliance6", "interpolate_nodes", "field_at", "compliance_at", "eigen_loads",
           "slice_eigenstrains", "solve_held", "MAX_ELEMENTS"]
