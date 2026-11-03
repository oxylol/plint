"""The voxel element: a cube of side a, eight nodes, linear isotropic elasticity, with Wilson's incompatible
modes (1 - xi^2, 1 - eta^2, 1 - zeta^2 per displacement component, condensed at element level; Taylor,
Beresford & Wilson 1976). For cubes the Jacobian is constant, so the element passes the patch test and is exact
for pure bending (no shear locking with one or two cells through a thickness); docs/models/structure.md 3.2.

Corner c of a cell (i, j, k) is the lattice node (i + di, j + dj, k + dk) with c = 4 di + 2 dj + dk; element DOFs
are (ux, uy, uz) per corner in that order. Voigt order of strains and stresses: xx, yy, zz, yz, xz, xy
(engineering shear strains).
"""

from __future__ import annotations

import functools
import math

import numpy as np

CORNERS = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], dtype=np.int64)
_XI = 2.0 * CORNERS - 1.0                 # local coordinates of the corners
_G = 1.0 / math.sqrt(3.0)
_GAUSS = [(sx * _G, sy * _G, sz * _G) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]


def elasticity(E_Pa: float, nu: float) -> np.ndarray:
    lam = E_Pa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_Pa / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _shape(p):
    xi, eta, zeta = p
    N = (1.0 + _XI[:, 0] * xi) * (1.0 + _XI[:, 1] * eta) * (1.0 + _XI[:, 2] * zeta) / 8.0
    dN = np.stack([_XI[:, 0] * (1.0 + _XI[:, 1] * eta) * (1.0 + _XI[:, 2] * zeta),
                   _XI[:, 1] * (1.0 + _XI[:, 0] * xi) * (1.0 + _XI[:, 2] * zeta),
                   _XI[:, 2] * (1.0 + _XI[:, 0] * xi) * (1.0 + _XI[:, 1] * eta)]) / 8.0
    return N, dN                       # dN [3, 8] w.r.t. the local coordinates


def _strain_matrix(dx: np.ndarray) -> np.ndarray:
    """B [6, 3 n] from shape-function gradients dx [3, n] (x, y, z derivatives)."""
    n = dx.shape[1]
    B = np.zeros((6, 3 * n))
    for c in range(n):
        gx, gy, gz = dx[:, c]
        B[0, 3 * c] = gx
        B[1, 3 * c + 1] = gy
        B[2, 3 * c + 2] = gz
        B[3, 3 * c + 1] = gz
        B[3, 3 * c + 2] = gy
        B[4, 3 * c] = gz
        B[4, 3 * c + 2] = gx
        B[5, 3 * c] = gy
        B[5, 3 * c + 1] = gx
    return B


@functools.lru_cache(maxsize=32)
def matrices(a_m: float, E_Pa: float, nu: float, rho_kg_m3: float):
    """(K [24, 24] condensed, M [24, 24] consistent, G0 [24, 6] = int B^T dV) of a cube of side a."""
    C = elasticity(E_Pa, nu)
    J = 2.0 / a_m                       # d(local)/d(x)
    w = (a_m / 2.0) ** 3                # Gauss weight x det J
    Kuu = np.zeros((24, 24))
    Kua = np.zeros((24, 9))
    Kaa = np.zeros((9, 9))
    M = np.zeros((24, 24))
    G0 = np.zeros((24, 6))
    for p in _GAUSS:
        N, dN = _shape(p)
        Bu = _strain_matrix(dN * J)
        # incompatible modes: d(1 - xi^2)/dx = -2 xi J (one gradient component per mode)
        dP = np.zeros((3, 3))
        dP[0, 0] = -2.0 * p[0] * J
        dP[1, 1] = -2.0 * p[1] * J
        dP[2, 2] = -2.0 * p[2] * J
        Ba = _strain_matrix(dP)
        CBu = C @ Bu
        Kuu += w * Bu.T @ CBu
        Kua += w * Bu.T @ (C @ Ba)
        Kaa += w * Ba.T @ (C @ Ba)
        Nm = np.zeros((3, 24))
        for c in range(8):
            Nm[0, 3 * c] = Nm[1, 3 * c + 1] = Nm[2, 3 * c + 2] = N[c]
        M += w * rho_kg_m3 * Nm.T @ Nm
        G0 += w * Bu.T
    K = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)
    K = 0.5 * (K + K.T)
    return K, 0.5 * (M + M.T), G0


def thermal_vector(a_m: float, E_Pa: float, nu: float) -> np.ndarray:
    """int B^T C (1, 1, 1, 0, 0, 0) dV: the load of a unit free volumetric strain component (times alpha dT).
    The incompatible modes take no load from a constant strain (their gradients integrate to zero)."""
    _, _, G0 = matrices(a_m, E_Pa, nu, 1.0)
    return G0 @ (elasticity(E_Pa, nu) @ np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def centre_strain(a_m: float) -> np.ndarray:
    """B [6, 24] at the cell centre (mean strain of the element's compatible field)."""
    _, dN = _shape((0.0, 0.0, 0.0))
    return _strain_matrix(dN * (2.0 / a_m))


__all__ = ["CORNERS", "elasticity", "matrices", "thermal_vector", "centre_strain"]
