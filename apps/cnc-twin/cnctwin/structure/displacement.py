"""Loads of the displacement grids (docs/models/structure.md section 5): temperatures on the structure grid,
free thermal strain, the stock's initial residual stress, and the Treuting-Read layer-removal relation the
residual-stress release is checked against.

* **Thermal**: thermal's work grid (cell centres origin + (i + 1/2) cell, NaN outside material) is filled
  outward by nearest values, sampled at 8 points of every structure cell (trilinear) and averaged; the free
  strain is ``alpha(T) (T - 293.15 K)`` with alpha the mean expansion coefficient from 20 degC of the material
  table (architecture 4.1: the material configuration is at 20 degC).
* **Residual stress** (the as-supplied stock, material ``residual_stress`` of the stock's form, the entry whose
  thickness is nearest the stock's): rolled plate and other plate forms: in-plane stresses through the stock
  thickness (S z), rolling direction L along the longer in-plane side of the stock, LT across it, ST ignored
  (free surfaces); every cell layer takes the profile's mean over its depth, then the profile is balanced over
  the stock's layers (its discrete force and moment through the thickness set to zero: the stock as supplied is
  in equilibrium and undistorted, d = 0 at the start). Bars and tubes: axial (L or axial), hoop and radial
  stress as functions of r / R (cylindrical components turned into S), the axial resultant balanced.
* **Treuting-Read** (1951), forward form: a plate keeping z in [0, h1] of a stress profile sigma_x(z),
  sigma_y(z) curves by ``kappa = D^-1 C^-1 M`` with ``M = -int sigma (z - h1/2) dz``, ``D = E h1^3 / 12``,
  ``C = [[1, nu], [nu, 1]] / (1 - nu^2)``.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.ndimage as ndi

from ..units import REFERENCE_TEMP_K
from .materials import WorkMaterial

PLATE_FORMS = ("rolled_plate", "forged", "cast", "hot_rolled_bar")
BAR_FORMS = ("extruded_bar", "cold_drawn_bar", "hot_rolled_bar")


# ------------------------------------------------------------------------------- temperatures
def cell_temperatures(grid, cells: np.ndarray, origin_T, cell_T: float, temp_T: np.ndarray | None,
                      ambient_K: float = REFERENCE_TEMP_K) -> np.ndarray:
    """Temperature (K) of each structure cell from thermal's work grid (module doc); the reference temperature
    when there is no grid."""
    n = len(cells)
    if temp_T is None or np.size(temp_T) == 0 or cell_T <= 0.0:
        return np.full(n, float(ambient_K))
    T = np.asarray(temp_T, dtype=float)
    bad = ~np.isfinite(T)
    if np.all(bad):
        return np.full(n, float(ambient_K))
    if np.any(bad):
        idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        T = T[tuple(idx)]
    off = np.array([(a, b, c) for a in (0.25, 0.75) for b in (0.25, 0.75) for c in (0.25, 0.75)])
    origin_T = np.asarray(origin_T, dtype=float)
    out = np.zeros(n)
    pts = grid.origin[None, None, :] + (cells[:, None, :] + off[None, :, :]) * grid.cell_m
    u = (pts - origin_T) / cell_T - 0.5
    shape = np.array(T.shape)
    lo = np.clip(np.floor(u).astype(np.int64), 0, np.maximum(shape - 2, 0))
    t = np.clip(u - lo, 0.0, 1.0)
    t = np.where(shape[None, None, :] > 1, t, 0.0)
    acc = np.zeros(u.shape[:2])
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = (np.where(di, t[..., 0], 1 - t[..., 0]) * np.where(dj, t[..., 1], 1 - t[..., 1])
                     * np.where(dk, t[..., 2], 1 - t[..., 2]))
                ii = np.minimum(lo[..., 0] + di, shape[0] - 1)
                jj = np.minimum(lo[..., 1] + dj, shape[1] - 1)
                kk = np.minimum(lo[..., 2] + dk, shape[2] - 1)
                acc += w * T[ii, jj, kk]
    out[:] = acc.mean(axis=1)
    return out


def thermal_strain(mat: WorkMaterial, T_cells: np.ndarray) -> np.ndarray:
    T = np.asarray(T_cells, dtype=float)
    return np.asarray(mat.alpha_1_K(T), dtype=float) * (T - REFERENCE_TEMP_K)


# ------------------------------------------------------------------------------- residual stress
def _profile_mean(x: np.ndarray, s: np.ndarray, a: float, b: float) -> float:
    """Mean of the piecewise-linear profile (held constant beyond its ends) over [a, b] (exact)."""
    if b <= a:
        return float(np.interp(a, x, s))
    pts = np.unique(np.concatenate([[a, b], x[(x > a) & (x < b)]]))
    vals = np.interp(pts, x, s)
    return float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(pts)) / (b - a))


def select_profiles(mat: WorkMaterial, form: str, thickness_m: float) -> dict:
    """direction -> profile entry of the stock's form, the entry with the thickness nearest the stock's."""
    out: dict = {}
    for entry in mat.residual:
        if entry["form"] != form:
            continue
        d = entry["direction"]
        t = entry.get("thickness_m") or thickness_m
        best = out.get(d)
        if best is None or abs(t - thickness_m) < abs((best.get("thickness_m") or thickness_m) - thickness_m):
            out[d] = entry
    return out


def plate_stress(mat: WorkMaterial, form: str, grid, cells: np.ndarray, lo, hi, layer_weight: np.ndarray | None = None):
    """Initial stress [n_cells, 6] of a plate stock (module doc) and the balanced layer profiles
    {direction: [n_layers]}."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    t = float(hi[2] - lo[2])
    profiles = select_profiles(mat, form, t)
    sigma = np.zeros((len(cells), 6))
    layers = {}
    if not profiles or t <= 0.0:
        return sigma, layers
    nz = grid.shape[2]
    z0 = grid.origin[2] + np.arange(nz) * grid.cell_m
    a = np.clip((z0 - lo[2]) / t, 0.0, 1.0)
    b = np.clip((z0 + grid.cell_m - lo[2]) / t, 0.0, 1.0)
    w = (b - a) if layer_weight is None else np.asarray(layer_weight, dtype=float)
    zeta = 0.5 * (a + b)
    size = hi - lo
    L_axis, LT_axis = (0, 1) if size[0] >= size[1] else (1, 0)
    for direction, comp in (("L", L_axis), ("LT", LT_axis)):
        entry = profiles.get(direction)
        if entry is None:
            continue
        x = np.asarray(entry["x"], dtype=float)
        s = np.asarray(entry["sigma_Pa"], dtype=float)
        lay = np.array([_profile_mean(x, s, a[k], b[k]) if b[k] > a[k] else 0.0 for k in range(nz)])
        lay = balance(lay, zeta, w)
        layers[direction] = lay
        sigma[:, comp] = lay[cells[:, 2]]
    return sigma, layers


def balance(values: np.ndarray, zeta: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """values - (c0 + c1 (zeta - zeta_mean)) with zero weighted resultant and first moment."""
    w = np.asarray(weight, dtype=float)
    if np.sum(w) <= 0.0:
        return values * 0.0
    zm = float(np.sum(w * zeta) / np.sum(w))
    dz = zeta - zm
    c0 = float(np.sum(w * values) / np.sum(w))
    m2 = float(np.sum(w * dz * dz))
    c1 = float(np.sum(w * values * dz) / m2) if m2 > 0.0 else 0.0
    return np.where(w > 0.0, values - c0 - c1 * dz, 0.0)


def bar_stress(mat: WorkMaterial, form: str, grid, cells: np.ndarray, radius_m: float):
    """Initial stress [n_cells, 6] of a bar or tube stock on S z (module doc)."""
    profiles = select_profiles(mat, form, 2.0 * radius_m)
    sigma = np.zeros((len(cells), 6))
    if not profiles or radius_m <= 0.0:
        return sigma
    c = grid.origin + (cells + 0.5) * grid.cell_m
    r = np.hypot(c[:, 0], c[:, 1])
    rho = np.clip(r / radius_m, 0.0, 1.0)
    cos = np.where(r > 0.0, c[:, 0] / np.where(r > 0.0, r, 1.0), 1.0)
    sin = np.where(r > 0.0, c[:, 1] / np.where(r > 0.0, r, 1.0), 0.0)

    def comp(names):
        for d in names:
            if d in profiles:
                e = profiles[d]
                return np.interp(rho, np.asarray(e["x"], float), np.asarray(e["sigma_Pa"], float))
        return np.zeros(len(cells))

    s_z = comp(("axial", "L"))
    s_t = comp(("hoop",))
    s_r = comp(("radial",))
    if np.any(s_z != 0.0):
        s_z = s_z - float(np.mean(s_z))          # zero axial resultant over the (equal-area) cells
    sigma[:, 0] = s_r * cos * cos + s_t * sin * sin
    sigma[:, 1] = s_r * sin * sin + s_t * cos * cos
    sigma[:, 2] = s_z
    sigma[:, 5] = (s_r - s_t) * sin * cos
    return sigma


# ------------------------------------------------------------------------------- Treuting-Read
def treuting_read_curvature(z_edges: np.ndarray, sigma_x: np.ndarray, sigma_y: np.ndarray, h1: float, E_Pa: float,
                            nu: float, plate: bool = True) -> np.ndarray:
    """(kappa_x, kappa_y) of the plate kept between z = 0 and h1 for layer-wise constant initial stresses
    (layers between z_edges, only z < h1 counted). ``plate=False``: a narrow beam (kappa_x = M_x 12 / (E h1^3),
    kappa_y = -nu kappa_x for a uniaxial sigma_x)."""
    z_edges = np.asarray(z_edges, dtype=float)
    c = 0.5 * h1
    Mx = My = 0.0
    for k in range(len(z_edges) - 1):
        a, b = z_edges[k], min(z_edges[k + 1], h1)
        if b <= a:
            continue
        first = 0.5 * ((b - c) ** 2 - (a - c) ** 2)          # int (z - c) dz over the layer
        Mx -= float(sigma_x[k]) * first
        My -= float(sigma_y[k]) * first
    if not plate:
        kx = 12.0 * Mx / (E_Pa * h1 ** 3)
        return np.array([kx, -nu * kx])
    D = E_Pa * h1 ** 3 / (12.0 * (1.0 - nu * nu))
    A = D * np.array([[1.0, nu], [nu, 1.0]])
    return np.linalg.solve(A, np.array([Mx, My]))


__all__ = ["cell_temperatures", "thermal_strain", "plate_stress", "bar_stress", "balance", "select_profiles",
           "treuting_read_curvature", "PLATE_FORMS", "BAR_FORMS"]
