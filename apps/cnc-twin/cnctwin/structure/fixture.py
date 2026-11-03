"""Fixture compliance of a held part (docs/models/structure.md section 6).

Holding publishes the stiffness of the part's contact set as a rigid body, ``support_stiffness_S`` (6 x 6 about
``centroid_S_m``, S axes: force and moment against translation and rotation). Its inverse is the compliance of
the part on its fixture, C6 (rigid-body directions without stiffness are left out: a part free in a direction
is holding's finding, not a compliance). The 3 x 3 translational compliance at a point p of the part is
``A C6 A^T`` with ``A = [I, -[r]x]``, r = p - centroid (a force F at p is the wrench (F, r x F) about the centroid;
the point moves by u + theta x r). The bus record gives it at the part datum: the origin of W expressed in S
(``ctx.project.stock_pose_W_S``, the work zero the program is written in).
"""

from __future__ import annotations

import numpy as np

REL_FREE = 1e-12          # eigenvalues below this fraction of the largest are free rigid-body directions


def skew(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def compliance6(K6: np.ndarray) -> np.ndarray:
    K = 0.5 * (np.asarray(K6, dtype=float) + np.asarray(K6, dtype=float).T)
    lam, V = np.linalg.eigh(K)
    if lam.size == 0 or lam[-1] <= 0.0:
        return np.zeros((6, 6))
    keep = lam > REL_FREE * lam[-1]
    return (V[:, keep] / lam[keep]) @ V[:, keep].T


def point_compliance(C6: np.ndarray, centroid_S_m, point_S_m) -> np.ndarray:
    r = np.asarray(point_S_m, dtype=float) - np.asarray(centroid_S_m, dtype=float)
    A = np.hstack([np.eye(3), -skew(r)])
    C = A @ C6 @ A.T
    return 0.5 * (C + C.T)


def datum_point_S(stock_pose_W_S) -> np.ndarray:
    """The W origin in S (stock_pose maps S to W: p_W = R p_S + t)."""
    T = np.eye(4) if stock_pose_W_S is None else np.asarray(stock_pose_W_S, dtype=float)
    return -T[:3, :3].T @ T[:3, 3]


__all__ = ["compliance6", "point_compliance", "datum_point_S", "skew"]
