"""Solvers of the voxel stiffness systems (docs/models/structure.md section 3.4).

A sparse factorisation of a 3-D voxel model of 50 000 DOFs fills to about 1e8 entries (1.1 GB with SuperLU's
minimum-degree ordering, measured), far above the structure budget (200 MB). Large systems are therefore solved
by conjugate gradients preconditioned with a geometric multigrid V-cycle on the node lattice (Briggs, Henson &
McCormick 2000): trilinear prolongation from every second lattice node, Galerkin coarse operators
``A_c = P^T A P`` (they keep holes, supports and thin features of the fine model), Chebyshev smoothing of
``D^-1 A`` on ``[lambda_max / 30, 1.1 lambda_max]`` (Adams et al. 2003; lambda_max by 20 power iterations from
ones / sqrt(n)), pre and post (a symmetric preconditioner), a dense Cholesky factorisation on the coarsest level.
Small systems (``direct_max_dof``) are factorised (SuperLU, symmetric mode, no pivoting). Everything runs in a
fixed order on one thread: results are bit-identical for the same input.

Block right-hand sides are solved by independent CG recurrences in one vectorised loop (each column stops
updating when it has converged).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

CHEB_RATIO = 30.0           # lambda_min = lambda_max / 30 for the smoother (Adams et al. 2003)
CHEB_SAFETY = 1.1           # lambda_max estimate x 1.1
POWER_ITERATIONS = 20
DENSE_COARSE_MAX = 6000     # coarsest level up to this size: dense Cholesky, above: SuperLU
MIN_REDUCTION = 0.7         # stop coarsening when a level keeps more than 70 % of the DOFs


def _axis_map(N: int):
    """Coarse lattice positions and, for every fine index, (q0, q1, w0, w1) of the linear interpolation."""
    if N <= 2:
        f = np.arange(N)
        return N, f, f, np.ones(N), np.zeros(N)
    Nc = (N - 1) // 2 + 1 + (N - 1) % 2          # every second node, and the last one
    pos = np.minimum(2 * np.arange(Nc), N - 1)
    f = np.arange(N)
    q0 = np.minimum(f // 2, Nc - 1)
    q1 = np.minimum(q0 + 1, Nc - 1)
    p0 = pos[q0]
    p1 = pos[q1]
    span = np.where(p1 > p0, p1 - p0, 1)
    w1 = np.where(p1 > p0, (f - p0) / span, 0.0)
    w0 = 1.0 - w1
    exact = f == p0
    w0 = np.where(exact, 1.0, w0)
    w1 = np.where(exact, 0.0, w1)
    return Nc, q0, q1, w0, w1


def prolongation(dims, dof_ijk: np.ndarray, dof_comp: np.ndarray):
    """(P [n_fine, n_coarse] csr, coarse dims, coarse dof ijk, coarse dof comp)."""
    maps = [_axis_map(int(N)) for N in dims]
    cdims = tuple(m[0] for m in maps)
    rows, keys, vals = [], [], []
    n = dof_ijk.shape[0]
    fine = np.arange(n)
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                q = []
                w = np.ones(n)
                for a, s in enumerate((sx, sy, sz)):
                    Nc, q0, q1, w0, w1 = maps[a]
                    idx = dof_ijk[:, a]
                    q.append(q1[idx] if s else q0[idx])
                    w = w * (w1[idx] if s else w0[idx])
                keep = w != 0.0
                key = ((q[0] * cdims[1] + q[1]) * cdims[2] + q[2]) * 3 + dof_comp
                rows.append(fine[keep])
                keys.append(key[keep])
                vals.append(w[keep])
    rows = np.concatenate(rows)
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)
    ukeys, cols = np.unique(keys, return_inverse=True)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(ukeys)))
    P.sum_duplicates()
    node = ukeys // 3
    comp = (ukeys % 3).astype(np.int64)
    cijk = np.stack([node // (cdims[1] * cdims[2]), (node // cdims[2]) % cdims[1], node % cdims[2]], axis=1)
    return P, cdims, cijk.astype(np.int64), comp


def _lambda_max(A, dinv: np.ndarray) -> float:
    v = np.ones(A.shape[0]) / math.sqrt(A.shape[0])
    lam = 1.0
    for _ in range(POWER_ITERATIONS):
        w = dinv * (A @ v)
        lam = float(np.linalg.norm(w))
        if lam == 0.0:
            return 1.0
        v = w / lam
    return lam


class _Level:
    def __init__(self, A):
        self.A = A
        d = A.diagonal()
        if np.any(d <= 0.0):
            raise ValueError("multigrid: a stiffness matrix with a non-positive diagonal entry")
        self.dinv = 1.0 / d
        self.lmax = CHEB_SAFETY * _lambda_max(A, self.dinv)
        self.lmin = self.lmax / CHEB_RATIO
        self.P = None


class Multigrid:
    """Geometric multigrid V-cycle (a symmetric preconditioner) for an SPD voxel stiffness matrix."""

    def __init__(self, A, dims, dof_ijk, dof_comp, coarse_dof: int = 1500, degree: int = 3):
        self.degree = int(degree)
        self.levels = [_Level(sp.csr_matrix(A))]
        ijk, comp, cur_dims = np.asarray(dof_ijk, np.int64), np.asarray(dof_comp, np.int64), tuple(dims)
        while self.levels[-1].A.shape[0] > coarse_dof:
            if all(N <= 2 for N in cur_dims):
                break
            P, cdims, cijk, ccomp = prolongation(cur_dims, ijk, comp)
            if P.shape[1] > MIN_REDUCTION * P.shape[0]:
                break
            AP = self.levels[-1].A @ P
            Ac = (P.T.tocsr() @ AP).tocsr()
            del AP
            Ac = (0.5 * (Ac + Ac.T)).tocsr()
            self.levels[-1].P = P.tocsr()
            self.levels.append(_Level(Ac))
            ijk, comp, cur_dims = cijk, ccomp, cdims
        Ac = self.levels[-1].A
        if Ac.shape[0] <= DENSE_COARSE_MAX:
            self._chol = sla.cho_factor(Ac.toarray(), lower=True, check_finite=False)
            self._lu = None
        else:
            self._chol = None
            self._lu = spla.splu(Ac.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                 options={"SymmetricMode": True})

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def _coarse(self, b):
        if self._chol is not None:
            return sla.cho_solve(self._chol, b, check_finite=False)
        return self._lu.solve(b)

    def _smooth(self, lv: _Level, b, x):
        theta = 0.5 * (lv.lmax + lv.lmin)
        delta = 0.5 * (lv.lmax - lv.lmin)
        sigma = theta / delta
        rho = 1.0 / sigma
        dinv = lv.dinv if b.ndim == 1 else lv.dinv[:, None]
        r = b - lv.A @ x
        d = dinv * r / theta
        for _ in range(self.degree):
            x = x + d
            r = r - lv.A @ d
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = (rho_new * rho) * d + (2.0 * rho_new / delta) * (dinv * r)
            rho = rho_new
        return x

    def vcycle(self, b, level: int = 0):
        lv = self.levels[level]
        if level == len(self.levels) - 1:
            return self._coarse(b)
        x = self._smooth(lv, b, np.zeros_like(b))
        r = b - lv.A @ x
        ec = self.vcycle(lv.P.T @ r, level + 1)
        x = x + lv.P @ ec
        return self._smooth(lv, b, x)

    def __call__(self, b):
        return self.vcycle(b)


def pcg(A, b, precond, rtol: float, max_iter: int):
    """Preconditioned CG for one or several right-hand sides (columns independent). Returns (x, iterations,
    relative residual per column)."""
    single = b.ndim == 1
    B = b[:, None] if single else b
    X = np.zeros_like(B)
    R = B.copy()
    bnorm = np.sqrt(np.einsum("ij,ij->j", B, B))
    bnorm[bnorm == 0.0] = 1.0
    done = np.sqrt(np.einsum("ij,ij->j", R, R)) <= rtol * bnorm
    Z = precond(R)
    Pd = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    it = 0
    while it < max_iter and not np.all(done):
        Q = A @ Pd
        pq = np.einsum("ij,ij->j", Pd, Q)
        alpha = np.where(done | (pq == 0.0), 0.0, rz / np.where(pq == 0.0, 1.0, pq))
        X += Pd * alpha
        R -= Q * alpha
        res = np.sqrt(np.einsum("ij,ij->j", R, R))
        done = done | (res <= rtol * bnorm)
        it += 1
        if np.all(done):
            break
        Z = precond(R)
        rz_new = np.einsum("ij,ij->j", R, Z)
        beta = np.where(done | (rz == 0.0), 0.0, rz_new / np.where(rz == 0.0, 1.0, rz))
        Pd = Z + Pd * beta
        rz = rz_new
    res = np.sqrt(np.einsum("ij,ij->j", R, R)) / bnorm
    return (X[:, 0] if single else X), it, res


class SPDSolver:
    """K x = b for a voxel model: SuperLU up to ``direct_max_dof``, else multigrid-preconditioned CG."""

    def __init__(self, A, dims, dof_ijk, dof_comp, settings):
        self.A = sp.csr_matrix(A)
        self.n = self.A.shape[0]
        self.rtol = float(settings.pcg_rtol)
        self.max_iter = int(settings.pcg_max_iter)
        self.iterations = 0
        self.worst_residual = 0.0
        if self.n <= int(settings.direct_max_dof):
            self.kind = "direct"
            self._lu = spla.splu(self.A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                 options={"SymmetricMode": True})
            self._mg = None
        else:
            self.kind = "multigrid"
            self._lu = None
            self._mg = Multigrid(self.A, dims, dof_ijk, dof_comp, int(settings.gmg_coarse_dof),
                                 int(settings.gmg_degree))

    def solve(self, b):
        b = np.asarray(b, dtype=float)
        if self._lu is not None:
            return self._lu.solve(b)
        x, it, res = pcg(self.A, b, self._mg, self.rtol, self.max_iter)
        self.iterations += it
        self.worst_residual = max(self.worst_residual, float(np.max(res)) if np.size(res) else 0.0)
        if np.any(res > max(1e3 * self.rtol, 1e-6)):
            raise ArithmeticError(f"structure: CG did not converge ({float(np.max(res)):.2e} after {it} iterations)")
        return x

    def operator(self) -> spla.LinearOperator:
        return spla.LinearOperator((self.n, self.n), matvec=self.solve, dtype=float)

    def preconditioner(self):
        """The multigrid V-cycle (a symmetric approximation of A^-1 at the cost of about one CG iteration), or None
        for a factorised system (its solve is cheap)."""
        return self._mg


__all__ = ["Multigrid", "pcg", "SPDSolver", "prolongation"]
