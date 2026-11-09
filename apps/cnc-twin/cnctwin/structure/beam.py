"""Timoshenko beam finite elements (docs/models/structure.md section 2.2).

Beam axis along local z; six DOFs per node in the order (ux, uy, uz, rx, ry, rz) (translations, then
right-handed small rotations). Bending in the x-z plane uses (ux, ry) with dux/dz = ry and the second moment
``Iy = int x^2 dA``; bending in the y-z plane uses (uy, rx) with duy/dz = -rx and ``Ix = int y^2 dA``.

Element matrices (Przemieniecki 1968, ch. 5 and 11): stiffness with shear deformation,
``Phi = 12 E I / (k G A L^2)``, exact for end loads of a prismatic Timoshenko beam (static deflection
``P L^3 / (3 E I) + P L / (k G A)`` of a cantilever reproduced exactly by one element); consistent mass
with translational and rotary inertia including Phi; axial and torsional bars with linear shape functions.
Shear coefficients by Cowper (1966): solid / hollow circle and rectangle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

DOF = 6
UX, UY, UZ, RX, RY, RZ = range(6)


@dataclass(frozen=True)
class Section:
    """Section of a beam element (SI). ``mass_*`` are the values for the inertia when they differ from the
    stiffness section (a fluted section whose stiffness is an equivalent circle)."""

    A_m2: float
    Ix_m4: float                 # int y^2 dA: deflection along y (uy, rx)
    Iy_m4: float                 # int x^2 dA: deflection along x (ux, ry)
    J_m4: float                  # Saint-Venant torsion constant
    kx: float                    # shear coefficient for shear along x
    ky: float                    # shear coefficient for shear along y
    E_Pa: float
    G_Pa: float
    rho_kg_m3: float
    mass_A_m2: float | None = None
    mass_Ix_m4: float | None = None
    mass_Iy_m4: float | None = None

    @property
    def rhoA_kg_m(self) -> float:
        return self.rho_kg_m3 * (self.A_m2 if self.mass_A_m2 is None else self.mass_A_m2)

    @property
    def rhoIx_kg_m(self) -> float:
        return self.rho_kg_m3 * (self.Ix_m4 if self.mass_Ix_m4 is None else self.mass_Ix_m4)

    @property
    def rhoIy_kg_m(self) -> float:
        return self.rho_kg_m3 * (self.Iy_m4 if self.mass_Iy_m4 is None else self.mass_Iy_m4)

    def scaled(self, stiffness: float = 1.0) -> "Section":
        """The same section with E and G multiplied (a stiffness change at the same mass)."""
        return replace(self, E_Pa=self.E_Pa * stiffness, G_Pa=self.G_Pa * stiffness)


def shear_modulus(E_Pa: float, nu: float) -> float:
    return E_Pa / (2.0 * (1.0 + nu))


def cowper_circle(nu: float, ratio: float = 0.0) -> float:
    """Shear coefficient of a solid (ratio 0) or hollow circle, ratio = inner / outer radius (Cowper 1966)."""
    m2 = ratio * ratio
    a = (1.0 + m2) ** 2
    return 6.0 * (1.0 + nu) * a / ((7.0 + 6.0 * nu) * a + (20.0 + 12.0 * nu) * m2)


def cowper_rectangle(nu: float) -> float:
    return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)


def circle(d_m: float, E_Pa: float, nu: float, rho_kg_m3: float, d_inner_m: float = 0.0) -> Section:
    """Solid or hollow circular section."""
    if d_m <= 0.0 or d_inner_m < 0.0 or d_inner_m >= d_m:
        raise ValueError(f"circular section needs 0 <= d_inner < d (got {d_inner_m}, {d_m})")
    A = math.pi * (d_m ** 2 - d_inner_m ** 2) / 4.0
    I = math.pi * (d_m ** 4 - d_inner_m ** 4) / 64.0
    k = cowper_circle(nu, d_inner_m / d_m)
    return Section(A, I, I, 2.0 * I, k, k, E_Pa, shear_modulus(E_Pa, nu), rho_kg_m3)


def rectangle(bx_m: float, hy_m: float, E_Pa: float, nu: float, rho_kg_m3: float) -> Section:
    """Rectangle of width bx (along x) and height hy (along y). Torsion constant of a solid rectangle
    J = a b^3 (1/3 - 0.21 (b/a) (1 - b^4 / (12 a^4))), a >= b (Young & Budynas 2012, Table 10.7 case 4)."""
    a, b = max(bx_m, hy_m), min(bx_m, hy_m)
    J = a * b ** 3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b ** 4 / (12.0 * a ** 4)))
    k = cowper_rectangle(nu)
    return Section(bx_m * hy_m, bx_m * hy_m ** 3 / 12.0, hy_m * bx_m ** 3 / 12.0, J, k, k, E_Pa,
                   shear_modulus(E_Pa, nu), rho_kg_m3)


def plane_matrices(L: float, EI: float, kGA: float, rhoA: float, rhoI: float) -> tuple[np.ndarray, np.ndarray]:
    """(K, M) of one bending plane, DOFs (w1, theta1, w2, theta2) with theta = dw/dz."""
    phi = 12.0 * EI / (kGA * L * L) if kGA > 0.0 else 0.0
    c = EI / ((1.0 + phi) * L ** 3)
    K = c * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L],
                      [6.0 * L, (4.0 + phi) * L * L, -6.0 * L, (2.0 - phi) * L * L],
                      [-12.0, -6.0 * L, 12.0, -6.0 * L],
                      [6.0 * L, (2.0 - phi) * L * L, -6.0 * L, (4.0 + phi) * L * L]])
    p2 = phi * phi
    s = rhoA * L / (1.0 + phi) ** 2
    a1 = 13.0 / 35.0 + 7.0 * phi / 10.0 + p2 / 3.0
    a2 = (11.0 / 210.0 + 11.0 * phi / 120.0 + p2 / 24.0) * L
    a3 = 9.0 / 70.0 + 3.0 * phi / 10.0 + p2 / 6.0
    a4 = (13.0 / 420.0 + 3.0 * phi / 40.0 + p2 / 24.0) * L
    a5 = (1.0 / 105.0 + phi / 60.0 + p2 / 120.0) * L * L
    a6 = (1.0 / 140.0 + phi / 60.0 + p2 / 120.0) * L * L
    Mt = s * np.array([[a1, a2, a3, -a4], [a2, a5, a4, -a6], [a3, a4, a1, -a2], [-a4, -a6, -a2, a5]])
    r = rhoI / ((1.0 + phi) ** 2 * L)
    b1 = 6.0 / 5.0
    b2 = (1.0 / 10.0 - phi / 2.0) * L
    b3 = (2.0 / 15.0 + phi / 6.0 + p2 / 3.0) * L * L
    b4 = (1.0 / 30.0 + phi / 6.0 - p2 / 6.0) * L * L
    Mr = r * np.array([[b1, b2, -b1, b2], [b2, b3, -b2, -b4], [-b1, -b2, b1, -b2], [b2, -b4, -b2, b3]])
    return K, Mt + Mr


_XZ = (UX, RY, DOF + UX, DOF + RY)
_YZ = (UY, RX, DOF + UY, DOF + RX)
_YZ_SIGN = np.array([1.0, -1.0, 1.0, -1.0])


def element(L: float, sec: Section) -> tuple[np.ndarray, np.ndarray]:
    """(K, M) 12 x 12 of a Timoshenko beam element of length L along local z."""
    if L <= 0.0:
        raise ValueError("beam element length must be positive")
    K = np.zeros((12, 12))
    M = np.zeros((12, 12))
    Kx, Mx = plane_matrices(L, sec.E_Pa * sec.Iy_m4, sec.kx * sec.G_Pa * sec.A_m2, sec.rhoA_kg_m, sec.rhoIy_kg_m)
    Ky, My = plane_matrices(L, sec.E_Pa * sec.Ix_m4, sec.ky * sec.G_Pa * sec.A_m2, sec.rhoA_kg_m, sec.rhoIx_kg_m)
    ix = np.array(_XZ)
    iy = np.array(_YZ)
    K[np.ix_(ix, ix)] += Kx
    M[np.ix_(ix, ix)] += Mx
    sy = np.outer(_YZ_SIGN, _YZ_SIGN)
    K[np.ix_(iy, iy)] += Ky * sy
    M[np.ix_(iy, iy)] += My * sy
    bar = np.array([[1.0, -1.0], [-1.0, 1.0]])
    lump = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    ia = np.array([UZ, DOF + UZ])
    K[np.ix_(ia, ia)] += sec.E_Pa * sec.A_m2 / L * bar
    M[np.ix_(ia, ia)] += sec.rhoA_kg_m * L * lump
    it = np.array([RZ, DOF + RZ])
    K[np.ix_(it, it)] += sec.G_Pa * sec.J_m4 / L * bar
    M[np.ix_(it, it)] += (sec.rhoIx_kg_m + sec.rhoIy_kg_m) * L * lump
    return K, M


def foundation(L: float, k_lat_N_per_m2: float, k_ax_N_per_m2: float, k_tor_N: float,
               k_lat_y_N_per_m2: float | None = None, k_rot_N: float = 0.0, k_rot_y_N: float | None = None) -> np.ndarray:
    """12 x 12 stiffness of a distributed elastic (Winkler) foundation over one element, acting on the element's
    DOFs (or on relative DOFs between two coincident beams): lateral stiffness per length k_lat (along x; along y
    ``k_lat_y``, default the same) with the cubic (Hermite) displacement interpolation; axial, torsional and
    bending-rotation stiffness per length (``k_rot`` against ry, ``k_rot_y`` against rx: the axial tangential
    contact of a gripped surface resisting the rotation of its section) with linear interpolation."""
    K = np.zeros((12, 12))
    base = L / 420.0 * np.array([[156.0, 22.0 * L, 54.0, -13.0 * L],
                                 [22.0 * L, 4.0 * L * L, 13.0 * L, -3.0 * L * L],
                                 [54.0, 13.0 * L, 156.0, -22.0 * L],
                                 [-13.0 * L, -3.0 * L * L, -22.0 * L, 4.0 * L * L]])
    ky = k_lat_N_per_m2 if k_lat_y_N_per_m2 is None else k_lat_y_N_per_m2
    ix = np.array(_XZ)
    iy = np.array(_YZ)
    K[np.ix_(ix, ix)] += k_lat_N_per_m2 * base
    K[np.ix_(iy, iy)] += ky * base * np.outer(_YZ_SIGN, _YZ_SIGN)
    lin = np.array([[2.0, 1.0], [1.0, 2.0]]) * L / 6.0
    ia = np.array([UZ, DOF + UZ])
    it = np.array([RZ, DOF + RZ])
    K[np.ix_(ia, ia)] += k_ax_N_per_m2 * lin
    K[np.ix_(it, it)] += k_tor_N * lin
    kry = k_rot_N if k_rot_y_N is None else k_rot_y_N
    iry = np.array([RY, DOF + RY])
    irx = np.array([RX, DOF + RX])
    K[np.ix_(iry, iry)] += k_rot_N * lin
    K[np.ix_(irx, irx)] += kry * lin
    return K


class BeamAssembly:
    """Dense global matrices of beam nodes (6 DOFs each) built from elements and foundation couplings."""

    def __init__(self, n_nodes: int):
        self.n_nodes = int(n_nodes)
        n = DOF * self.n_nodes
        self.K = np.zeros((n, n))
        self.M = np.zeros((n, n))

    def dofs(self, node: int) -> np.ndarray:
        return DOF * int(node) + np.arange(DOF)

    def add_element(self, i: int, j: int, L: float, sec: Section, R: np.ndarray | None = None,
                    stiffness_into: np.ndarray | None = None) -> None:
        """Element between nodes i and j along the local axis; R (3 x 3, columns = local axes in the global
        frame) rotates it into the global frame; ``stiffness_into`` (an n x n array) also receives the stiffness
        (the per-part matrices of the damping model)."""
        Ke, Me = element(L, sec)
        if R is not None:
            T = np.kron(np.eye(4), np.asarray(R, dtype=float))
            Ke = T @ Ke @ T.T
            Me = T @ Me @ T.T
        idx = np.concatenate([self.dofs(i), self.dofs(j)])
        self.K[np.ix_(idx, idx)] += Ke
        self.M[np.ix_(idx, idx)] += Me
        if stiffness_into is not None:
            stiffness_into[np.ix_(idx, idx)] += Ke

    def add_coupling(self, ia: int, ja: int, ib: int, jb: int, Kf: np.ndarray, R: np.ndarray | None = None,
                     stiffness_into: np.ndarray | None = None) -> None:
        """Foundation stiffness Kf (12 x 12 on relative DOFs) between beam a (nodes ia, ja) and beam b (ib, jb):
        energy 1/2 (u_a - u_b)^T Kf (u_a - u_b)."""
        if R is not None:
            T = np.kron(np.eye(4), np.asarray(R, dtype=float))
            Kf = T @ Kf @ T.T
        a = np.concatenate([self.dofs(ia), self.dofs(ja)])
        b = np.concatenate([self.dofs(ib), self.dofs(jb)])
        for (r, sr) in ((a, 1.0), (b, -1.0)):
            for (c, sc) in ((a, 1.0), (b, -1.0)):
                self.K[np.ix_(r, c)] += sr * sc * Kf
                if stiffness_into is not None:
                    stiffness_into[np.ix_(r, c)] += sr * sc * Kf

    def add_ground_coupling(self, i: int, j: int, Kf: np.ndarray, R: np.ndarray | None = None,
                            stiffness_into: np.ndarray | None = None) -> None:
        """Foundation stiffness to a fixed ground over the element (i, j)."""
        if R is not None:
            T = np.kron(np.eye(4), np.asarray(R, dtype=float))
            Kf = T @ Kf @ T.T
        a = np.concatenate([self.dofs(i), self.dofs(j)])
        self.K[np.ix_(a, a)] += Kf
        if stiffness_into is not None:
            stiffness_into[np.ix_(a, a)] += Kf


def subdivide(breakpoints, max_len: float) -> np.ndarray:
    """Sorted node positions: every breakpoint kept, intervals split into equal elements no longer than
    max_len."""
    pts = np.unique(np.round(np.asarray(sorted(float(b) for b in breakpoints)), 12))
    out = [float(pts[0])]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil((b - a) / max_len - 1e-9)))
        out.extend(float(a + (b - a) * (k + 1) / n) for k in range(n))
    return np.array(out)


__all__ = ["Section", "DOF", "UX", "UY", "UZ", "RX", "RY", "RZ", "shear_modulus", "cowper_circle",
           "cowper_rectangle", "circle", "rectangle", "plane_matrices", "element", "foundation", "BeamAssembly",
           "subdivide"]
