"""Thin walls and floors thinner than ``thin_wall_cells`` cells (docs/models/structure.md section 4).

A voxel grid cannot represent a wall thinner than about two cells: its cells are in or out of the model by the
0.5 fill rule, so the wall is missing, or one or two cells thick. Such walls get an analytic plate model:

* **detection** on the structure grid from removal's material fractions f: along each axis, a run of
  consecutive cells with material between air cells is a wall across that axis when its material thickness
  ``T = sum f cell`` is below ``thin_wall_cells`` cells; face-connected cells of the same axis form one wall (at
  least two cells in each in-plane direction); its thickness is removal's ``wall_thickness`` at the wall's
  central cell when the geometry offers it, else the median T;
* **plate**: a Kirchhoff rectangle over the wall's in-plane extent (free edges refined by the fractional fill
  of their cells), each edge clamped where the wall meets thick material, free where it ends in air, simply
  supported where it meets another thin wall; Rayleigh-Ritz with Bhat's characteristic orthogonal polynomials
  (start function x^p (1 - x)^q satisfying the edge conditions, Gram-Schmidt on [0, 1]; Bhat 1985, Warburton
  1954, Leissa 1969); modes mass-normalised (rho t int w^2 dA = 1) and the static point compliance from the
  same Ritz space;
* **blend** with the FE, per wall cell, weight ``w = clip(c_thin - T / cell, 0, 1)`` with c_thin =
  ``thin_wall_cells`` (1 when the wall is a cell or less thick, 0 at the threshold; the FE's own representation
  takes 1 - w): FE mode shapes ``(1 - w) phi_FE(cell) + w phi_FE(root)`` (the wall is carried by the part at its
  root), plate modes appended with shapes ``w n phi_wall``, residual compliance
  ``(1 - w) C_FE(cell) + w (C_FE(root) + n n^T G_wall,res)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.ndimage as ndi
from numpy.polynomial import Polynomial
from numpy.polynomial import legendre as L

EDGE_POWER = {"F": 0, "S": 1, "C": 2}
GAUSS_POINTS = 32


# ------------------------------------------------------------------------------- 1-D Ritz functions
def ritz_functions(left: str, right: str, n: int) -> list[Polynomial]:
    """n polynomials on [0, 1] orthonormal in L2, each a multiple of x^p (1 - x)^q (p, q by the edge
    conditions F 0, S 1, C 2)."""
    p, q = EDGE_POWER[left], EDGE_POWER[right]
    base = Polynomial([0.0, 1.0]) ** p * Polynomial([1.0, -1.0]) ** q
    xg, wg = L.leggauss(GAUSS_POINTS + 2 * n)
    xg = 0.5 * (xg + 1.0)
    wg = 0.5 * wg
    out: list[Polynomial] = []
    for k in range(n):
        f = base * _shifted_legendre(k)
        for g in out:
            f = f - float(np.sum(wg * f(xg) * g(xg))) * g
        norm = math.sqrt(float(np.sum(wg * f(xg) ** 2)))
        out.append(f / norm)
    return out


def _shifted_legendre(k: int) -> Polynomial:
    coef = np.zeros(k + 1)
    coef[k] = 1.0
    leg = Polynomial(L.leg2poly(coef))
    return leg(Polynomial([-1.0, 2.0]))


def _integrals(funcs: list[Polynomial]):
    """Gram matrices on [0, 1]: (f_i f_j), (f_i'' f_j''), (f_i'' f_j), (f_i' f_j')."""
    n = len(funcs)
    xg, wg = L.leggauss(GAUSS_POINTS + 2 * n)
    xg = 0.5 * (xg + 1.0)
    wg = 0.5 * wg
    F0 = np.array([f(xg) for f in funcs])
    F1 = np.array([f.deriv(1)(xg) for f in funcs])
    F2 = np.array([f.deriv(2)(xg) for f in funcs])
    I00 = (F0 * wg) @ F0.T
    I22 = (F2 * wg) @ F2.T
    I20 = (F2 * wg) @ F0.T
    I11 = (F1 * wg) @ F1.T
    return I00, I22, I20, I11


@dataclass
class Plate:
    """Kirchhoff rectangle a x b, thickness t, edge conditions (u = 0, u = a, v = 0, v = b)."""

    a_m: float
    b_m: float
    t_m: float
    E_Pa: float
    nu: float
    rho_kg_m3: float
    edges: tuple = ("C", "F", "F", "F")
    terms: int = 6
    _fu: list = field(default_factory=list, repr=False)
    _fv: list = field(default_factory=list, repr=False)
    K: np.ndarray | None = field(default=None, repr=False)
    M: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if min(self.a_m, self.b_m, self.t_m) <= 0.0:
            raise ValueError("plate dimensions must be positive")
        n = int(self.terms)
        self._fu = ritz_functions(self.edges[0], self.edges[1], n)
        self._fv = ritz_functions(self.edges[2], self.edges[3], n)
        U00, U22, U20, U11 = _integrals(self._fu)
        V00, V22, V20, V11 = _integrals(self._fv)
        D = self.E_Pa * self.t_m ** 3 / (12.0 * (1.0 - self.nu ** 2))
        a, b = self.a_m, self.b_m
        K = (np.kron(U22, V00) / a ** 4 + np.kron(U00, V22) / b ** 4
             + self.nu * (np.kron(U20, V20.T) + np.kron(U20.T, V20)) / (a * a * b * b)
             + 2.0 * (1.0 - self.nu) * np.kron(U11, V11) / (a * a * b * b)) * D * a * b
        self.K = 0.5 * (K + K.T)
        M = self.rho_kg_m3 * self.t_m * a * b * np.kron(U00, V00)
        self.M = 0.5 * (M + M.T)

    @property
    def D_Nm(self) -> float:
        return self.E_Pa * self.t_m ** 3 / (12.0 * (1.0 - self.nu ** 2))

    def basis(self, u_m, v_m) -> np.ndarray:
        """[n_points, n_terms^2] Ritz functions at plate coordinates."""
        u = np.atleast_1d(np.asarray(u_m, dtype=float)) / self.a_m
        v = np.atleast_1d(np.asarray(v_m, dtype=float)) / self.b_m
        Fu = np.array([f(u) for f in self._fu]).T
        Fv = np.array([f(v) for f in self._fv]).T
        return np.einsum("pi,pj->pij", Fu, Fv).reshape(len(u), -1)

    def modes(self) -> tuple[np.ndarray, np.ndarray]:
        """(omega [n] ascending, coefficient vectors [n_terms^2, n] mass-normalised, sign-normalised)."""
        lam, Y = sla.eigh(self.K, self.M, check_finite=False)
        Y = Y / np.sqrt(np.einsum("ij,ij->j", Y, self.M @ Y))
        idx = np.argmax(np.abs(Y), axis=0)
        Y = Y * np.where(Y[idx, np.arange(Y.shape[1])] < 0.0, -1.0, 1.0)
        return np.sqrt(np.maximum(lam, 0.0)), Y

    def compliance(self, u_m, v_m) -> np.ndarray:
        """Static normal point compliance (m/N) at points (the Ritz solution of a unit point force)."""
        Phi = self.basis(u_m, v_m)
        X = sla.solve(self.K, Phi.T, assume_a="pos", check_finite=False)
        return np.einsum("ij,ji->i", Phi, X)

    def deflection(self, u_m, v_m, load_u_m, load_v_m, force_N) -> np.ndarray:
        """Static deflection at points under point forces."""
        Pl = self.basis(load_u_m, load_v_m)
        a = sla.solve(self.K, Pl.T @ np.atleast_1d(np.asarray(force_N, dtype=float)), assume_a="pos",
                      check_finite=False)
        return self.basis(u_m, v_m) @ a


# ------------------------------------------------------------------------------- detection on the grid
@dataclass
class Wall:
    axis: int                         # normal axis of the wall (0 x, 1 y, 2 z)
    cells: np.ndarray                 # [n, 3] cells of the wall
    weight: np.ndarray                # [n] blend weight of the analytic model
    thickness_m: float
    plate_axes: tuple                 # (u axis, v axis)
    u_range_m: tuple                  # plate extent in S along the u axis
    v_range_m: tuple
    edges: tuple                      # (u0, u1, v0, v1) in {"C", "S", "F"}
    root_cells: np.ndarray            # [n, 3] FE cells whose shape carries each wall cell
    omega_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shapes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))     # [m, n] normal displacement per mode
    residual_m_per_N: np.ndarray = field(default_factory=lambda: np.zeros(0))  # [n] after the published modes
    compliance_m_per_N: np.ndarray = field(default_factory=lambda: np.zeros(0))  # [n] static, all modes

    def to_dict(self) -> dict:
        return {"axis": int(self.axis), "thickness_m": float(self.thickness_m),
                "plate_axes": [int(a) for a in self.plate_axes], "u_range_m": [float(v) for v in self.u_range_m],
                "v_range_m": [float(v) for v in self.v_range_m], "edges": list(self.edges)}


def _runs(fill_line: np.ndarray, fmin: float):
    """(start, end) of runs of material along a line (end exclusive)."""
    mat = fill_line > fmin
    out = []
    i = 0
    n = len(mat)
    while i < n:
        if mat[i]:
            j = i
            while j < n and mat[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def thin_maps(fill: np.ndarray, cell_m: float, thin_cells: float, fmin: float):
    """Per axis: (thin [nx, ny, nz] bool, thickness [nx, ny, nz] in m of the run through each thin cell)."""
    out = []
    for axis in range(3):
        moved = np.moveaxis(fill, axis, -1)
        thin = np.zeros(moved.shape, bool)
        thick = np.zeros(moved.shape)
        for idx in np.ndindex(moved.shape[:-1]):
            line = moved[idx]
            for a, b in _runs(line, fmin):
                T = float(np.sum(line[a:b])) * cell_m
                if T < thin_cells * cell_m:
                    thin[idx + (slice(a, b),)] = True
                    thick[idx + (slice(a, b),)] = T
        out.append((np.moveaxis(thin, -1, axis), np.moveaxis(thick, -1, axis)))
    return out


def detect_walls(fill: np.ndarray, active: np.ndarray, origin: np.ndarray, cell_m: float, settings,
                 geometry=None, workpiece: int = 0) -> list[Wall]:
    """Thin walls of the current material (module doc), largest first, at most ``max_walls``."""
    fmin = float(settings.wall_min_fill)
    thin_c = float(settings.thin_wall_cells)
    if thin_c <= 0.0 or settings.max_walls <= 0:
        return []
    maps = thin_maps(fill, cell_m, thin_c, fmin)
    claimed = np.zeros(fill.shape, bool)
    walls: list[Wall] = []
    candidates = []
    for axis, (thin, thick) in enumerate(maps):
        others = [a for a in range(3) if a != axis]
        # cells thin along this axis but not along another (a wall, not a pin or an edge)
        mask = thin.copy()
        for a in others:
            mask &= ~maps[a][0]
        labels, n = ndi.label(mask)
        for lab in range(1, n + 1):
            cells = np.argwhere(labels == lab)
            ext = [int(cells[:, a].max() - cells[:, a].min() + 1) for a in others]
            if min(ext) < 2:
                continue
            candidates.append((-len(cells), axis, lab, cells, thick))
    candidates.sort(key=lambda c: (c[0], c[1], tuple(c[3][0])))
    for _neg, axis, _lab, cells, thick in candidates[: int(settings.max_walls)]:
        if np.any(claimed[tuple(cells.T)]):
            continue
        claimed[tuple(cells.T)] = True
        walls.append(_build_wall(axis, cells, thick, fill, active, maps, origin, cell_m, settings, geometry,
                                 workpiece))
    return walls


def _edge_kind(fill, active, maps, axis, cells_out, threshold) -> str:
    """C when the cells beyond the edge are mostly thick material, F when mostly air, S otherwise (another
    thin wall)."""
    shape = np.array(fill.shape)
    inside = np.all((cells_out >= 0) & (cells_out < shape[None, :]), axis=1)
    if not np.any(inside):
        return "F"
    c = cells_out[inside]
    f = fill[tuple(c.T)]
    thin_other = np.zeros(len(c), bool)
    for a in range(3):
        if a != axis:
            thin_other |= maps[a][0][tuple(c.T)]
    solid = (f > threshold) & ~thin_other & ~maps[axis][0][tuple(c.T)]
    air = f <= threshold * 0.2
    n = len(c)
    if np.count_nonzero(solid) >= 0.5 * n:
        return "C"
    if np.count_nonzero(air) >= 0.5 * n:
        return "F"
    return "S"


def _build_wall(axis, cells, thick, fill, active, maps, origin, cell_m, settings, geometry, workpiece) -> Wall:
    u_ax, v_ax = [a for a in range(3) if a != axis]
    T_cells = thick[tuple(cells.T)]
    T = float(np.median(T_cells))
    centre = cells[np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))][len(cells) // 2]
    if geometry is not None and hasattr(geometry, "wall_thickness"):
        n = np.zeros(3)
        n[axis] = 1.0
        side = origin + (centre + 0.5) * cell_m
        side[axis] = origin[axis] + (cells[:, axis].max() + 1.5) * cell_m       # in the air beyond the wall
        measured = float(geometry.wall_thickness(workpiece, side, n, 3.0 * cell_m * max(settings.thin_wall_cells, 1)))
        if 0.0 < measured < settings.thin_wall_cells * cell_m * 1.5:
            T = measured
    weight = np.clip(float(settings.thin_wall_cells) - T_cells / cell_m, 0.0, 1.0)
    thr = float(settings.fill_threshold)
    lo = cells.min(axis=0)
    hi = cells.max(axis=0)
    edges = []
    ranges = []
    for ax in (u_ax, v_ax):
        kinds = []
        rng = []
        for side, idx in ((0, lo[ax] - 1), (1, hi[ax] + 1)):
            boundary = cells[cells[:, ax] == (lo[ax] if side == 0 else hi[ax])].copy()
            boundary[:, ax] = idx
            kind = _edge_kind(fill, active, maps, axis, boundary, thr)
            kinds.append(kind)
            face = origin[ax] + (lo[ax] if side == 0 else hi[ax] + 1) * cell_m
            if kind == "F":
                edge_cells = cells[cells[:, ax] == (lo[ax] if side == 0 else hi[ax])]
                frac = float(np.clip(np.mean(fill[tuple(edge_cells.T)] * cell_m / np.maximum(
                    thick[tuple(edge_cells.T)], 1e-12)), 0.0, 1.0))
                face = origin[ax] + ((lo[ax] + 1 - frac) if side == 0 else (hi[ax] + frac)) * cell_m
            rng.append(face)
        edges.extend(kinds)
        ranges.append((min(rng), max(rng)))
    roots = _roots(cells, active, axis, edges, (u_ax, v_ax), lo, hi)
    return Wall(axis, cells, weight, T, (u_ax, v_ax), ranges[0], ranges[1], tuple(edges), roots)


def _roots(cells, active, axis, edges, plate_axes, lo, hi) -> np.ndarray:
    """For each wall cell the nearest active cell beyond a clamped edge (the wall's root), else itself."""
    shape = np.array(active.shape)
    targets = []
    for k, ax in enumerate(plate_axes):
        for side in (0, 1):
            if edges[2 * k + side] == "C":
                targets.append((ax, lo[ax] - 1 if side == 0 else hi[ax] + 1))
    out = cells.copy()
    if not targets:
        return out
    for i, c in enumerate(cells):
        best = None
        for ax, idx in targets:
            cand = c.copy()
            cand[ax] = idx
            if np.all(cand >= 0) and np.all(cand < shape) and active[tuple(cand)]:
                d = abs(int(c[ax]) - int(idx))
                if best is None or d < best[0]:
                    best = (d, cand)
        if best is not None:
            out[i] = best[1]
    return out


def solve_wall(wall: Wall, origin: np.ndarray, cell_m: float, E_Pa: float, nu: float, rho: float, settings) -> Wall:
    """Plate modes and compliances at the wall's cell centres (in place)."""
    a = wall.u_range_m[1] - wall.u_range_m[0]
    b = wall.v_range_m[1] - wall.v_range_m[0]
    plate = Plate(a, b, wall.thickness_m, E_Pa, nu, rho, wall.edges, int(settings.wall_terms))
    omega, Y = plate.modes()
    u_ax, v_ax = wall.plate_axes
    centres = origin + (wall.cells + 0.5) * cell_m
    u = np.clip(centres[:, u_ax] - wall.u_range_m[0], 0.0, a)
    v = np.clip(centres[:, v_ax] - wall.v_range_m[0], 0.0, b)
    Phi = plate.basis(u, v)
    m = int(min(settings.wall_modes, len(omega)))
    shapes = (Phi @ Y[:, :m]).T
    comp = plate.compliance(u, v)
    res = comp - np.sum(shapes ** 2 / np.maximum(omega[:m, None] ** 2, 1e-300), axis=0)
    wall.omega_rad_s = omega[:m].copy()
    wall.shapes = shapes
    wall.compliance_m_per_N = comp
    wall.residual_m_per_N = np.maximum(res, 0.0)
    return wall


__all__ = ["Plate", "Wall", "ritz_functions", "thin_maps", "detect_walls", "solve_wall", "EDGE_POWER"]
