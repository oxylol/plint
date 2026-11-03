"""Voxel finite elements of a workpiece (docs/models/structure.md section 3).

The grid is co-registered with the stock box in S: origin at the stock's lower corner, cubic cells of the
smallest size ``>= physics.structure.min_cell`` for which the full box has at most ``max_dof`` DOFs (so the
model never exceeds it as material is removed). A cell is in the model when removal's material fraction exceeds
``fill_threshold`` (0.5): the model's volume equals the material's to within the surface cells' rounding (half a
cell layer at worst, on average zero). Cells not face-connected to the supports (or to the largest component
without supports) are dropped: they are slivers the grid cannot resolve (their mass and stiffness are lost).

Elements: :mod:`cnctwin.structure.hexfe` (incompatible-mode hexahedra), assembled in CSR by the kernels of
:mod:`cnctwin.structure.kernels`; consistent mass. Supports (holding's ``supports``) are springs at their
contact points, spread to the cell's corners by trilinear weights: normal stiffness for closed contacts, the two
tangential stiffnesses for sticking and bonded ones. Modes by ARPACK shift-invert at sigma = 0 with v0 = ones /
sqrt(n) (eigsh), the inverse by SuperLU or multigrid CG (multigrid.py); refreshed by Rayleigh-Ritz on the previous
modes plus the static residual vectors ``K^-1 (K phi - lambda M phi)`` of the changed model, a full eigen solve
only when the Ritz residual exceeds ``ritz_tol``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..core.errors import InputError
from . import hexfe
from . import kernels as KN
from . import modal as MO
from .multigrid import SPDSolver


@dataclass(frozen=True)
class VoxelGrid:
    origin_S_m: tuple
    cell_m: float
    shape: tuple

    @property
    def node_shape(self) -> tuple:
        return tuple(n + 1 for n in self.shape)

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.origin_S_m, dtype=float)

    def cell_centres(self) -> np.ndarray:
        """[nx, ny, nz, 3] cell centres in S."""
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), axis=-1)
        return self.origin + (idx + 0.5) * self.cell_m

    def node_positions(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(ijk, dtype=float) * self.cell_m

    def to_dict(self) -> dict:
        return {"origin_S_m": [float(v) for v in self.origin_S_m], "cell_m": float(self.cell_m),
                "shape": [int(v) for v in self.shape]}

    @classmethod
    def from_dict(cls, d: dict) -> "VoxelGrid":
        return cls(tuple(float(v) for v in d["origin_S_m"]), float(d["cell_m"]), tuple(int(v) for v in d["shape"]))


def choose_grid(lo, hi, max_dof: int, min_cell_m: float) -> VoxelGrid:
    """Cubic grid over [lo, hi]: the smallest cell >= min_cell with 3 (nx+1)(ny+1)(nz+1) <= max_dof."""
    lo = np.asarray(lo, dtype=float)
    span = np.asarray(hi, dtype=float) - lo
    if np.any(span <= 0.0):
        raise InputError(f"structure: the stock box {lo} .. {hi} is empty")

    def dof(c):
        n = np.maximum(np.ceil(span / c - 1e-9), 1.0)
        return 3.0 * float(np.prod(n + 1.0))

    if dof(min_cell_m) <= max_dof:
        cell = float(min_cell_m)
    else:
        a, b = float(min_cell_m), float(np.max(span))
        if dof(b) > max_dof:
            raise InputError(f"structure: max_dof {max_dof} is too small for a single cell of the stock")
        for _ in range(80):
            m = 0.5 * (a + b)
            if dof(m) <= max_dof:
                b = m
            else:
                a = m
        cell = b
    shape = tuple(int(v) for v in np.maximum(np.ceil(span / cell - 1e-9), 1).astype(np.int64))
    return VoxelGrid(tuple(float(v) for v in lo), cell, shape)


def keep_connected(active: np.ndarray, seeds: np.ndarray | None) -> tuple[np.ndarray, int]:
    """(active cells face-connected to any seed cell (or the largest component without seeds), cells dropped)."""
    if not np.any(active):
        return active.copy(), 0
    labels, n = ndi.label(active)
    if n <= 1:
        return active.copy(), 0
    if seeds is not None and np.any(seeds & active):
        keep_labels = np.unique(labels[seeds & active])
        keep_labels = keep_labels[keep_labels > 0]
        out = np.isin(labels, keep_labels)
    else:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        out = labels == int(np.argmax(sizes))
    return out, int(np.count_nonzero(active) - np.count_nonzero(out))


class VoxelModel:
    """The assembled voxel model of one set of active cells."""

    def __init__(self, grid: VoxelGrid, active: np.ndarray, E_Pa: float, nu: float, rho_kg_m3: float):
        self.grid = grid
        self.active = np.ascontiguousarray(active, dtype=bool)
        self.E_Pa, self.nu, self.rho_kg_m3 = float(E_Pa), float(nu), float(rho_kg_m3)
        nx, ny, nz = grid.shape
        self.cells = np.ascontiguousarray(np.argwhere(self.active).astype(np.int64))
        if len(self.cells) == 0:
            raise InputError("structure: the workpiece has no cell in the model")
        ok = np.zeros(grid.node_shape, dtype=bool)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    ok[di:di + nx, dj:dj + ny, dk:dk + nz] |= self.active
        self.node_ok = ok
        self.node_ijk = np.ascontiguousarray(np.argwhere(ok).astype(np.int64))
        self.n_nodes = len(self.node_ijk)
        self.node_index = np.full(grid.node_shape, -1, np.int64)
        self.node_index[tuple(self.node_ijk.T)] = np.arange(self.n_nodes)
        self.nbr = np.zeros((self.n_nodes, 27), np.int64)
        KN.neighbour_table(np.ascontiguousarray(self.active.astype(np.int8)), self.node_index, self.node_ijk, self.nbr)
        counts = np.count_nonzero(self.nbr >= 0, axis=1).astype(np.int64)
        row_len = np.repeat(3 * counts, 3)
        self.indptr = np.zeros(3 * self.n_nodes + 1, np.int64)
        np.cumsum(row_len, out=self.indptr[1:])
        nnz = int(self.indptr[-1])
        if nnz >= 2 ** 31 - 1:
            raise InputError(f"structure: the voxel model has {nnz} matrix entries (beyond 32-bit indices)")
        self.indices = np.zeros(nnz, np.int32)
        KN.fill_indices(self.node_ijk, self.node_index, self.nbr, self.indptr, self.indices)
        Ke, Me, G0 = hexfe.matrices(grid.cell_m, self.E_Pa, self.nu, self.rho_kg_m3)
        self.Ke, self.Me, self.G0 = Ke, Me, G0
        ones = np.ones(len(self.cells))
        kd = np.zeros(nnz)
        md = np.zeros(nnz)
        KN.assemble_values(self.cells, self.node_index, self.nbr, self.indptr, np.ascontiguousarray(Ke), ones, kd)
        KN.assemble_values(self.cells, self.node_index, self.nbr, self.indptr, np.ascontiguousarray(Me), ones, md)
        n = 3 * self.n_nodes
        # int32 index arrays shared by every matrix of the model (K, M, springs, systems): one pattern in memory
        self.indptr32 = self.indptr.astype(np.int32)
        self.K = sp.csr_matrix((kd, self.indices, self.indptr32), shape=(n, n))
        self.M = sp.csr_matrix((md, self.indices, self.indptr32), shape=(n, n))
        self.volume_m3 = len(self.cells) * grid.cell_m ** 3

    # ------------------------------------------------------------------ geometry helpers
    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    def dof_ijk(self) -> tuple[np.ndarray, np.ndarray]:
        return np.repeat(self.node_ijk, 3, axis=0), np.tile(np.arange(3, dtype=np.int64), self.n_nodes)

    def node_xyz(self) -> np.ndarray:
        return self.grid.node_positions(self.node_ijk)

    def axes(self) -> np.ndarray:
        """[n_dof, 3]: rigid translations along x, y, z."""
        A = np.zeros((self.n_dof, 3))
        for a in range(3):
            A[a::3, a] = 1.0
        return A

    def rigid_modes(self) -> np.ndarray:
        """[n_dof, 6]: translations and rotations about the model's centroid."""
        x = self.node_xyz()
        c = x.mean(axis=0)
        r = x - c
        R = np.zeros((self.n_dof, 6))
        for a in range(3):
            R[a::3, a] = 1.0
        # rotation about x: (0, -z, y); about y: (z, 0, -x); about z: (-y, x, 0)
        R[1::3, 3], R[2::3, 3] = -r[:, 2], r[:, 1]
        R[0::3, 4], R[2::3, 4] = r[:, 2], -r[:, 0]
        R[0::3, 5], R[1::3, 5] = -r[:, 1], r[:, 0]
        return R

    def locate(self, points_S_m) -> tuple[np.ndarray, np.ndarray]:
        """(cell [n, 3], corner weights [n, 8]) of points: the containing active cell, else the nearest active
        cell within two cells (the point projected onto it); trilinear weights, normalised."""
        P = np.atleast_2d(np.asarray(points_S_m, dtype=float))
        g = self.grid
        cells = np.zeros((len(P), 3), np.int64)
        weights = np.zeros((len(P), 8))
        shape = np.array(g.shape)
        offs = np.array([(a, b, c) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)], np.int64)
        for p, x in enumerate(P):
            f = (x - g.origin) / g.cell_m
            ci = np.clip(np.floor(f).astype(np.int64), 0, shape - 1)
            if not self.active[tuple(ci)]:
                cand = ci[None, :] + offs
                ok = np.all((cand >= 0) & (cand < shape[None, :]), axis=1)
                cand = cand[ok]
                cand = cand[self.active[tuple(cand.T)]]
                if len(cand) == 0:
                    raise InputError(f"structure: support point {x} is more than two cells from the model")
                centres = g.origin + (cand + 0.5) * g.cell_m
                d2 = np.sum((centres - x) ** 2, axis=1)
                order = np.lexsort((cand[:, 2], cand[:, 1], cand[:, 0], d2))
                ci = cand[order[0]]
            t = np.clip(f - ci, 0.0, 1.0)
            for c in range(8):
                di, dj, dk = (c >> 2) & 1, (c >> 1) & 1, c & 1
                weights[p, c] = (t[0] if di else 1 - t[0]) * (t[1] if dj else 1 - t[1]) * (t[2] if dk else 1 - t[2])
            cells[p] = ci
        return cells, weights

    def spring_data(self, points_S_m, kmats) -> np.ndarray:
        """Springs to ground (3 x 3 each) at points: the CSR data of the model's pattern (the springs couple
        only corners of one cell, which the pattern contains)."""
        data = np.zeros(len(self.indices))
        P = np.atleast_2d(np.asarray(points_S_m, dtype=float))
        if P.shape[0] and np.size(kmats):
            cells, w = self.locate(P)
            KN.add_point_springs(self.node_index, self.nbr, self.indptr, np.ascontiguousarray(cells),
                                 np.ascontiguousarray(w), np.ascontiguousarray(np.asarray(kmats, dtype=float)), data)
        return data

    def pattern_matrix(self, data: np.ndarray) -> sp.csr_matrix:
        n = self.n_dof
        return sp.csr_matrix((data, self.indices, self.indptr32), shape=(n, n))

    def spring_matrix(self, points_S_m, kmats) -> sp.csr_matrix:
        return self.pattern_matrix(self.spring_data(points_S_m, kmats))

    def point_loads(self, points_S_m, forces_N) -> np.ndarray:
        f = np.zeros(self.n_dof)
        P = np.atleast_2d(np.asarray(points_S_m, dtype=float))
        if len(P) == 0:
            return f
        cells, w = self.locate(P)
        F = np.atleast_2d(np.asarray(forces_N, dtype=float))
        for p in range(len(P)):
            i, j, k = cells[p]
            for c in range(8):
                if w[p, c] == 0.0:
                    continue
                n = self.node_index[i + ((c >> 2) & 1), j + ((c >> 1) & 1), k + (c & 1)]
                f[3 * n:3 * n + 3] += w[p, c] * F[p]
        return f

    def point_operator(self, points_S_m) -> sp.csr_matrix:
        """[3 n_points, n_dof]: displacement at points (trilinear) from the DOF vector."""
        P = np.atleast_2d(np.asarray(points_S_m, dtype=float))
        cells, w = self.locate(P)
        rows, cols, vals = [], [], []
        for p in range(len(P)):
            i, j, k = cells[p]
            for c in range(8):
                if w[p, c] == 0.0:
                    continue
                n = self.node_index[i + ((c >> 2) & 1), j + ((c >> 1) & 1), k + (c & 1)]
                for a in range(3):
                    rows.append(3 * p + a)
                    cols.append(3 * n + a)
                    vals.append(w[p, c])
        return sp.csr_matrix((vals, (rows, cols)), shape=(3 * len(P), self.n_dof))

    def cell_loads(self, coef: np.ndarray, Ge: np.ndarray) -> np.ndarray:
        """Global vector of the element vectors Ge @ coef[c] over the model's cells (coef [n_cells, k])."""
        out = np.zeros(self.n_dof)
        KN.scatter_cell_vectors(self.cells, self.node_index, np.ascontiguousarray(coef, dtype=float),
                                np.ascontiguousarray(Ge, dtype=float), out)
        return out

    def thermal_load(self, strain: np.ndarray) -> np.ndarray:
        """Load of the free volumetric thermal strain alpha dT of every cell ([n_cells])."""
        g = hexfe.thermal_vector(self.grid.cell_m, self.E_Pa, self.nu)
        return self.cell_loads(np.asarray(strain, dtype=float)[:, None], g[:, None])

    def initial_stress_load(self, sigma0: np.ndarray) -> np.ndarray:
        """-int B^T sigma0 dV of a cell-wise initial stress ([n_cells, 6], Voigt xx yy zz yz xz xy)."""
        return self.cell_loads(-np.asarray(sigma0, dtype=float), self.G0)

    def to_lattice(self, u: np.ndarray) -> np.ndarray:
        """[nnx, nny, nnz, 3] node values of a DOF vector (zero at lattice nodes outside the model)."""
        out = np.zeros(self.grid.node_shape + (3,))
        out[tuple(self.node_ijk.T)] = np.asarray(u, dtype=float).reshape(self.n_nodes, 3)
        return out

    def from_lattice(self, field: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(field)[tuple(self.node_ijk.T)].reshape(-1))


# ------------------------------------------------------------------------------- supports
@dataclass
class Supports:
    """Contact springs and clamp loads in S (from holding's supports, docs/models/holding.md 1)."""

    points: np.ndarray                  # [n, 3]
    k_held: np.ndarray                  # [n, 3, 3]: every closed contact (modes, compliance, thermal, residual)
    k_locate: np.ndarray                # [n, 3, 3]: locators and bonds only (clamping solve)
    k_locked: np.ndarray                # [n, 3, 3]: force-controlled contacts (rigid-body completion)
    preload_N: np.ndarray               # [n, 3]: clamping forces of the force-controlled contacts

    @classmethod
    def empty(cls) -> "Supports":
        z3 = np.zeros((0, 3))
        z33 = np.zeros((0, 3, 3))
        return cls(z3, z33, z33.copy(), z33.copy(), z3.copy())

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def to_arrays(self, prefix: str) -> dict:
        return {prefix + "points": self.points, prefix + "k_held": self.k_held, prefix + "k_locate": self.k_locate,
                prefix + "k_locked": self.k_locked, prefix + "preload": self.preload_N}

    @classmethod
    def from_arrays(cls, a: dict, prefix: str) -> "Supports":
        return cls(*(np.array(a[prefix + k], dtype=float) for k in ("points", "k_held", "k_locate", "k_locked",
                                                                     "preload")))


CLOSED = (0, 1, 3)       # holding abi.ST_*: stick, slip, bonded carry a normal force
RITZ_ITERATIONS = 4      # Rayleigh-Ritz steps before a full eigen solve is preferred
STICKING = (0, 3)        # stick, bonded carry tangential force


def supports_from_holding(supports: list) -> Supports:
    pts, kh, kl, kk, pre = [], [], [], [], []
    for patch in supports or []:
        law = str(patch.get("law", "locator"))
        pos = np.atleast_2d(np.asarray(patch.get("pos_S_m"), dtype=float))
        if pos.size == 0:
            continue
        n = np.atleast_2d(np.asarray(patch.get("normal_S"), dtype=float))
        t1 = np.atleast_2d(np.asarray(patch.get("t1_S"), dtype=float))
        kn = np.atleast_1d(np.asarray(patch.get("stiffness_N_per_m"), dtype=float))
        kt = np.atleast_2d(np.asarray(patch.get("tangential_stiffness_N_per_m"), dtype=float))
        state = np.atleast_1d(np.asarray(patch.get("state", np.zeros(len(pos))), dtype=np.int64))
        preload = np.atleast_2d(np.asarray(patch.get("preload_S_N", np.zeros((len(pos), 3))), dtype=float))
        for i in range(len(pos)):
            s = int(state[i]) if i < len(state) else 0
            K = np.zeros((3, 3))
            if s in CLOSED:
                ni = n[i] / max(np.linalg.norm(n[i]), 1e-300)
                K += kn[i] * np.outer(ni, ni)
                if s in STICKING:
                    a = t1[i] - (t1[i] @ ni) * ni
                    a = a / max(np.linalg.norm(a), 1e-300)
                    b = np.cross(ni, a)
                    kti = kt[i] if kt.shape[0] > i else np.zeros(2)
                    K += kti[0] * np.outer(a, a) + kti[1] * np.outer(b, b)
            pts.append(pos[i])
            kh.append(K)
            locked = law == "locked"
            kl.append(np.zeros((3, 3)) if locked else K)
            kk.append(K if locked else np.zeros((3, 3)))
            pre.append(preload[i] if (locked and s in CLOSED) else np.zeros(3))
    if not pts:
        return Supports.empty()
    return Supports(np.array(pts), np.array(kh), np.array(kl), np.array(kk), np.array(pre))


def minimal_constraints(model: VoxelModel) -> np.ndarray:
    """Six DOFs that remove the rigid-body motions without restraining the part (3-2-1): A (x, y, z) at the
    first node of the lowest layer, B (y, z) at the farthest node of the layer on A's x line (else (x, z) on its
    y line: B constrains only the components normal to AB, so a free expansion is not restrained), C (z) at
    the node of the layer farthest from the line AB."""
    ijk = model.node_ijk
    xyz = model.node_xyz()
    bottom = np.nonzero(ijk[:, 2] == ijk[:, 2].min())[0]
    A = int(bottom[0])
    d = xyz[bottom] - xyz[A]
    on_x = bottom[(ijk[bottom, 1] == ijk[A, 1]) & (ijk[bottom, 0] > ijk[A, 0])]
    on_y = bottom[(ijk[bottom, 0] == ijk[A, 0]) & (ijk[bottom, 1] > ijk[A, 1])]
    if len(on_x):
        B = int(on_x[int(np.argmax(ijk[on_x, 0]))])
        compB = (1, 2)
    elif len(on_y):
        B = int(on_y[int(np.argmax(ijk[on_y, 1]))])
        compB = (0, 2)
    else:
        B = int(bottom[int(np.argmax(np.sum(d * d, axis=1)))])
        dB0 = xyz[B] - xyz[A]
        compB = (1, 2) if abs(dB0[0]) >= abs(dB0[1]) else (0, 2)
    dB = xyz[B] - xyz[A]
    line = dB / max(np.linalg.norm(dB), 1e-300)
    off = d - np.outer(d @ line, line)
    C = int(bottom[int(np.argmax(np.sqrt(np.sum(off * off, axis=1))))])
    dofs = np.array([3 * A, 3 * A + 1, 3 * A + 2, 3 * B + compB[0], 3 * B + compB[1], 3 * C + 2], np.int64)
    R = model.rigid_modes()[dofs]
    if np.linalg.matrix_rank(R, tol=1e-9 * max(np.abs(R).max(), 1.0)) < 6:
        raise InputError("structure: the model's lowest node layer cannot carry a 3-2-1 support (degenerate)")
    return dofs


# ------------------------------------------------------------------------------- systems, solves, modes
class System:
    """K x = f of a model with extra springs (CSR data of the model's pattern) and fixed DOFs. The pattern is
    shared: fixed DOFs keep their row and column with only the diagonal (x = 0 there), so one data array is the
    whole system."""

    def __init__(self, model: VoxelModel, extra_data: np.ndarray | None = None, fixed=None, settings=None):
        self.model = model
        n = model.n_dof
        self.fixed = np.zeros(0, np.int64) if fixed is None else np.unique(np.asarray(fixed, np.int64))
        data = model.K.data.copy()
        if extra_data is not None:
            data += extra_data
        if len(self.fixed):
            data = _constrain(model, data, self.fixed, keep_diagonal=True)
            self.M = _MaskedMass(model.M, self.fixed)
        else:
            self.M = model.M
        self.K = model.pattern_matrix(data)
        self.free_mask = np.ones(n, bool)
        self.free_mask[self.fixed] = False
        self.free = np.nonzero(self.free_mask)[0]
        ijk, comp = model.dof_ijk()
        self.solver = SPDSolver(self.K, model.grid.node_shape, ijk, comp, settings)

    def solve(self, f: np.ndarray) -> np.ndarray:
        """Solution(s) for load(s) (zero at fixed DOFs); blocks in chunks of BLOCK columns (memory)."""
        f = np.array(f, dtype=float, copy=True)
        f[self.fixed] = 0.0
        if f.ndim == 2 and f.shape[1] > BLOCK:
            x = np.empty_like(f)
            for a in range(0, f.shape[1], BLOCK):
                x[:, a:a + BLOCK] = self.solver.solve(np.ascontiguousarray(f[:, a:a + BLOCK]))
        else:
            x = self.solver.solve(f)
        x[self.fixed] = 0.0
        return x

    def eigen(self, n_modes: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
        """Lowest modes by ARPACK shift-invert (sigma = 0, v0 = ones / sqrt(n)); canonical, mass-normalised."""
        nf = len(self.free)
        k = int(min(n_modes, nf - 2))
        if k < 1:
            raise InputError("structure: too few DOFs for a modal solve")
        n = self.model.n_dof
        ncv = int(min(n, max(2 * k + 1, k + 16)))
        v0 = np.ones(n) / math.sqrt(n)
        v0[self.fixed] = 0.0
        M = self.M.as_operator() if isinstance(self.M, _MaskedMass) else self.M
        lam, V = spla.eigsh(self.K, k=k, M=M, sigma=0.0, which="LM", v0=v0, ncv=ncv, tol=tol,
                            OPinv=spla.LinearOperator((n, n), matvec=self.solve, dtype=float),
                            maxiter=max(100 * k, 1000))
        order = np.argsort(lam, kind="stable")
        lam, V = lam[order], V[:, order]
        V[self.fixed] = 0.0
        return self._finish(lam, V)

    def _finish(self, lam, V):
        V = MO.m_normalise(V, self.M)
        axes = self.model.axes()
        axes[self.fixed] = 0.0
        return lam, MO.canonical(MO.freq_Hz(lam), V, self.M, axes)

    def ritz(self, V_guess: np.ndarray, lam_guess: np.ndarray, n_modes: int, tol: float, n_strict: int | None = None,
             max_iter: int = RITZ_ITERATIONS) -> tuple:
        """Rayleigh-Ritz on the guess vectors plus their static residual vectors K^-1 (K v - lambda M v),
        repeated while the energy-norm residual ``sqrt(r^T K^-1 r / lambda)`` of the first n_strict pairs exceeds
        tol (10 tol for the others). Returns (lam, vectors, residual, converged)."""
        V = np.array(V_guess, dtype=float, copy=True)
        V[self.fixed] = 0.0
        lam = np.asarray(lam_guess, dtype=float)
        n_strict = n_modes if n_strict is None else int(n_strict)
        eps = np.full(n_modes, np.inf)
        Z_prev = None
        for it in range(max_iter + 1):
            R = self.K @ V - (self.M @ V) * lam[None, :]
            Z = self.solve(R)
            eps = np.sqrt(np.maximum(np.einsum("ij,ij->j", R, Z), 0.0) / np.where(lam > 0.0, lam, 1.0))
            if it > 0:
                worst = max(float(np.max(eps[:n_strict], initial=0.0)),
                            float(np.max(eps[n_strict:], initial=0.0)) / 10.0)
                if worst <= tol:
                    lam, V = self._finish(lam, V)
                    return lam, V, worst, True
                if it == max_iter:
                    break
            # block Krylov basis: current vectors, their static corrections and the previous corrections
            basis = np.hstack([V, Z] if Z_prev is None else [V, Z, Z_prev])
            del R
            Z_prev = Z
            lam, V = MO.rayleigh_ritz(self.K, self.M, basis, n_modes)
        worst = max(float(np.max(eps[:n_strict], initial=0.0)), float(np.max(eps[n_strict:], initial=0.0)) / 10.0)
        lam, V = self._finish(lam, V)
        return lam, V, worst, False

BLOCK = 6       # columns per block CG solve (memory of the block recurrences)


class _MaskedMass:
    """The mass matrix with the rows and columns of fixed DOFs zeroed, as an operator (no copy)."""

    def __init__(self, M, fixed: np.ndarray):
        self.M = M
        self.fixed = fixed
        self.shape = M.shape
        self.dtype = M.dtype

    def __matmul__(self, x):
        y = np.array(x, dtype=float, copy=True)
        y[self.fixed] = 0.0
        y = self.M @ y
        y[self.fixed] = 0.0
        return y

    def dot(self, x):
        return self @ x

    def as_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(self.shape, matvec=lambda v: self @ v, matmat=lambda V: self @ V, dtype=float)


def _constrain(model: VoxelModel, data: np.ndarray, fixed: np.ndarray, keep_diagonal: bool) -> np.ndarray:
    """Zero the rows and columns of fixed DOFs in a pattern data array (diagonal kept, or zero), in place."""
    indptr = model.indptr
    is_fixed = np.zeros(model.n_dof, bool)
    is_fixed[fixed] = True
    diag_pos = []
    for r in fixed:
        a, b = int(indptr[r]), int(indptr[r + 1])
        cols = model.indices[a:b]
        diag_pos.append(a + int(np.nonzero(cols == r)[0][0]))
        data[a:b] = 0.0
    diag_pos = np.asarray(diag_pos, np.int64)
    keep = model.K.data[diag_pos].copy() if keep_diagonal else None
    data[is_fixed[model.indices]] = 0.0
    if keep_diagonal:
        data[diag_pos] = keep
    return data


def support_fraction(model: VoxelModel, K_springs, lam: np.ndarray, V: np.ndarray) -> np.ndarray:
    """phi^T K_s phi / lambda per mode: the share of each mode's strain energy in the support springs."""
    if K_springs is None:
        return np.zeros(len(lam))
    KV = K_springs @ V
    return np.einsum("ij,ij->j", V, KV) / np.where(lam > 0.0, lam, 1.0)


__all__ = ["VoxelGrid", "VoxelModel", "Supports", "System", "choose_grid", "keep_connected", "supports_from_holding",
           "minimal_constraints", "support_fraction", "CLOSED", "STICKING"]
