"""Deterministic modal algebra (architecture 9; docs/models/structure.md section 7).

* mass normalisation ``phi^T M phi = 1`` and sign normalisation (largest-magnitude component positive, the first
  such component when several are equal);
* degenerate clusters (``|f_i - f_j| / f < 1e-6``) rotated to a canonical basis: the M-projections of the rigid
  translations along the machine axes x, y, z (in that order) Gram-Schmidt orthonormalised in the M inner
  product, completed by the cluster's own vectors in their order;
* M-orthonormalisation of a basis by modified Gram-Schmidt (twice), dropping dependent columns in order;
* Rayleigh-Ritz on a basis and the relative eigen-residual of the Ritz pairs.

``M`` may be a dense array, a scipy sparse matrix or None (identity).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla

CLUSTER_REL = 1e-6          # architecture 9: |df| / f below this is one degenerate cluster
AXIS_REL = 1e-6             # axis projections below this fraction of the largest are numerical noise


def _mv(M, V):
    return V if M is None else M @ V


def m_normalise(V: np.ndarray, M=None) -> np.ndarray:
    V = np.array(V, dtype=float, copy=True)
    if V.ndim == 1:
        V = V[:, None]
    MV = _mv(M, V)
    norms = np.sqrt(np.maximum(np.einsum("ij,ij->j", V, MV), 0.0))
    norms[norms == 0.0] = 1.0
    return V / norms


def sign_normalise(V: np.ndarray) -> np.ndarray:
    """Columns flipped so that the component of largest magnitude is positive (argmax: the first one)."""
    V = np.array(V, dtype=float, copy=True)
    if V.ndim == 1:
        k = int(np.argmax(np.abs(V)))
        return -V if V[k] < 0.0 else V
    idx = np.argmax(np.abs(V), axis=0)
    signs = np.where(V[idx, np.arange(V.shape[1])] < 0.0, -1.0, 1.0)
    return V * signs


def clusters(freq: np.ndarray, rel: float = CLUSTER_REL) -> list[list[int]]:
    """Index groups of ascending frequencies whose neighbours differ by less than rel (relative)."""
    out: list[list[int]] = []
    for i, f in enumerate(np.asarray(freq, dtype=float)):
        if out and abs(f - freq[out[-1][-1]]) <= rel * max(abs(f), abs(freq[out[-1][-1]]), 1e-300):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _gram_schmidt_m(candidates: list[np.ndarray], M, need: int, rel_tol: float = 1e-8) -> list[np.ndarray]:
    """Modified Gram-Schmidt in the M inner product (two passes); M b is kept for every accepted b, so each
    candidate costs two products with M."""
    basis: list[np.ndarray] = []
    mbasis: list[np.ndarray] = []
    for c in candidates:
        if len(basis) >= need:
            break
        v = np.array(c, dtype=float, copy=True)
        n0 = math.sqrt(max(float(v @ _mv(M, v)), 0.0))
        if n0 == 0.0:
            continue
        for _ in range(2):
            for b, mb in zip(basis, mbasis):
                v -= float(mb @ v) * b
        mv = _mv(M, v)
        n1 = math.sqrt(max(float(v @ mv), 0.0))
        if n1 <= rel_tol * n0:
            continue
        basis.append(v / n1)
        mbasis.append(mv / n1)
    return basis


def canonical(freq: np.ndarray, V: np.ndarray, M=None, axes: np.ndarray | None = None,
              rel: float = CLUSTER_REL) -> np.ndarray:
    """Modes V (M-orthonormal columns, ascending freq) with every degenerate cluster rotated to the canonical
    basis and every column sign-normalised. ``axes`` [n, 3]: the rigid translation fields along x, y, z of the
    model (the projection basis); None: the cluster keeps its own vectors (only signs normalised)."""
    V = np.array(V, dtype=float, copy=True)
    for group in clusters(freq, rel):
        if len(group) < 2:
            continue
        W = V[:, group]
        cands: list[np.ndarray] = []
        if axes is not None:
            MW = _mv(M, W)
            proj = [MW.T @ axes[:, a] for a in range(axes.shape[1])]
            norms = [float(np.linalg.norm(c)) for c in proj]
            big = max(norms) if norms else 0.0
            for c, nc in zip(proj, norms):
                if big > 0.0 and nc > AXIS_REL * big:      # a projection at the noise level is no direction
                    cands.append(W @ c)
        cands.extend(W[:, k] for k in range(W.shape[1]))
        basis = _gram_schmidt_m(cands, M, len(group))
        if len(basis) == len(group):
            V[:, group] = np.stack(basis, axis=1)
    return sign_normalise(V)


def m_orthonormalise(V: np.ndarray, M=None, rel_tol: float = 1e-10) -> np.ndarray:
    """M-orthonormal basis of the columns of V (modified Gram-Schmidt twice, dependent columns dropped)."""
    cols = _gram_schmidt_m([V[:, k] for k in range(V.shape[1])], M, V.shape[1], rel_tol)
    if not cols:
        return np.zeros((V.shape[0], 0))
    return np.stack(cols, axis=1)


def rayleigh_ritz(K, M, V: np.ndarray, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(eigenvalues ascending, Ritz vectors) of K x = lambda M x on span(V) (V is M-orthonormalised here)."""
    Q = m_orthonormalise(V, M)
    Kr = Q.T @ (K @ Q)
    Kr = 0.5 * (Kr + Kr.T)
    Mr = Q.T @ _mv(M, Q)
    Mr = 0.5 * (Mr + Mr.T)
    lam, Y = sla.eigh(Kr, Mr, check_finite=False)
    if n is not None:
        lam, Y = lam[:n], Y[:, :n]
    return lam, Q @ Y


def residuals(K, M, lam: np.ndarray, V: np.ndarray) -> np.ndarray:
    """|K v - lambda M v| / |K v| per column."""
    KV = K @ V
    R = KV - _mv(M, V) * lam[None, :]
    num = np.sqrt(np.einsum("ij,ij->j", R, R))
    den = np.sqrt(np.einsum("ij,ij->j", KV, KV))
    den[den == 0.0] = 1.0
    return num / den


def freq_Hz(lam: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(lam, dtype=float), 0.0)) / (2.0 * math.pi)


def dense_modes(K: np.ndarray, M: np.ndarray, axes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """All modes of a small dense symmetric pencil (eigenvalues ascending, canonical M-normalised vectors)."""
    lam, V = sla.eigh(0.5 * (K + K.T), 0.5 * (M + M.T), check_finite=False)
    V = m_normalise(V, M)
    return lam, canonical(freq_Hz(lam), V, M, axes)


__all__ = ["CLUSTER_REL", "AXIS_REL", "m_normalise", "sign_normalise", "clusters", "canonical", "m_orthonormalise",
           "rayleigh_ritz", "residuals", "freq_Hz", "dense_modes"]
