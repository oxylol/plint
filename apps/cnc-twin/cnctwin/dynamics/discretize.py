"""Exact discretisation of mass-normalised modes (docs/models/dynamics.md section 2).

Each mode obeys ``q'' + 2 zeta omega q' + omega^2 q = f(t)`` (mass-normalised coordinates, f in N kg^-1/2).
With the state x = (q, v):

* **first-order hold** over a step h (f linear between the samples f0 and f1): ``x1 = Phi x0 + G0 f0 + G1 f1``,
  from the augmented exponential of [[A, B, 0], [0, 0, 1/h], [0, 0, 0]] (Franklin, Powell & Workman 1998,
  section 6.3.2); exact for a force linear in time;
* **energy integrals** of one first-order-hold step, exact for the same trajectory: with z = [q0, v0, f0, s]
  (s = (f1 - f0)/h) the structural damping dissipation ``D = int 2 zeta omega v^2 dt = z^T M_D z`` and the work
  of a force part f_p(t) = f0_p + s_p t, ``W_p = int f_p v dt = [f0_p, s_p] N z``, both from Van Loan's block
  exponential (Van Loan 1978) for the exosystem-augmented state;
* **free interval** with harmonic plus linear forcing ``f(t) = Re(P e^(i W t)) + g0 + g1 t`` (unbalance rotating at
  W, base excitation by first-order hold): the exact solution (harmonic particular solution, polynomial
  particular solution, homogeneous part) and the closed-form work and dissipation integrals built from
  ``J0(k) = int_0^h e^(k t) dt`` and ``J1(k) = int_0^h t e^(k t) dt`` (``cnctwin.dynamics.kernels`` has the
  compiled twin of :func:`free_interval`).

Python functions here are used at L2 and as references of the kernels.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.linalg import expm

from ..core.errors import InputError


def check_mode(omega_rad_s: float, zeta: float, what: str = "mode") -> None:
    """A mode the kernels can propagate: omega > 0 and 0 < zeta < 1 (underdamped, as every structural mode)."""
    if not (math.isfinite(omega_rad_s) and omega_rad_s > 0.0):
        raise InputError(f"{what}: natural frequency must be positive (omega = {omega_rad_s!r} rad/s)")
    if not (math.isfinite(zeta) and 0.0 < zeta < 1.0):
        raise InputError(f"{what}: damping ratio must lie in (0, 1) (zeta = {zeta!r})")


def state_matrix(omega_rad_s: float, zeta: float) -> np.ndarray:
    return np.array([[0.0, 1.0], [-omega_rad_s * omega_rad_s, -2.0 * zeta * omega_rad_s]])


def foh(omega_rad_s: float, zeta: float, h_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Phi [2, 2], G0 [2], G1 [2]) of one first-order-hold step of length h."""
    A = state_matrix(omega_rad_s, zeta)
    M = np.zeros((4, 4))
    M[:2, :2] = A * h_s
    M[1, 2] = h_s           # B = e2
    M[2, 3] = 1.0           # f(t) = f0 + (f1 - f0) t / h: the ramp state integrates in units of h
    E = expm(M)
    Phi = E[:2, :2].copy()
    gam0 = E[:2, 2].copy()
    gam1 = E[:2, 3].copy()
    # x1 = Phi x0 + gam0 f0 + gam1 (f1 - f0)
    return Phi, gam0 - gam1, gam1.copy()


def _augmented(omega_rad_s: float, zeta: float) -> np.ndarray:
    """A_z of z = [q, v, f, s]: q' = v, v' = -w^2 q - 2 zeta w v + f, f' = s, s' = 0."""
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 0] = -omega_rad_s * omega_rad_s
    A[1, 1] = -2.0 * zeta * omega_rad_s
    A[1, 2] = 1.0
    A[2, 3] = 1.0
    return A


def van_loan(A: np.ndarray, Q: np.ndarray, h_s: float) -> np.ndarray:
    """int_0^h e^(A^T t) Q e^(A t) dt (Van Loan 1978, Theorem 1)."""
    n = A.shape[0]
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = -A.T
    C[:n, n:] = Q
    C[n:, n:] = A
    E = expm(C * h_s)
    return E[n:, n:].T @ E[:n, n:]


def energy_grams(omega_rad_s: float, zeta: float, h_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(M_D [4, 4], N_fs [2, 4]) of one first-order-hold step: D = z^T M_D z, W_p = [f0_p, s_p] N_fs z."""
    A = _augmented(omega_rad_s, zeta)
    QD = np.zeros((4, 4))
    QD[1, 1] = 2.0 * zeta * omega_rad_s
    MD = van_loan(A, QD, h_s)
    MD = 0.5 * (MD + MD.T)
    QW = np.zeros((4, 4))
    QW[2, 1] = 1.0           # (e3^T z_p)(e2^T z)
    N = van_loan(A, QW, h_s)
    return MD, np.ascontiguousarray(N[2:4, :])


def modal_discretisation(omega_rad_s: np.ndarray, zeta: np.ndarray, h_s: float):
    """Phi [nm, 2, 2], G0 [nm, 2], G1 [nm, 2], gram_D [nm, 4, 4], gram_W [nm, 2, 4] of every mode at step h."""
    nm = len(omega_rad_s)
    Phi = np.zeros((nm, 2, 2))
    G0 = np.zeros((nm, 2))
    G1 = np.zeros((nm, 2))
    gD = np.zeros((nm, 4, 4))
    gW = np.zeros((nm, 2, 4))
    for k in range(nm):
        Phi[k], G0[k], G1[k] = foh(float(omega_rad_s[k]), float(zeta[k]), h_s)
        gD[k], gW[k] = energy_grams(float(omega_rad_s[k]), float(zeta[k]), h_s)
    return Phi, G0, G1, gD, gW


# ------------------------------------------------------------------------------- free interval (reference)
def phi1(z: complex) -> complex:
    """(e^z - 1) / z, accurate for every z (series below |z| = 0.2, expm1 form above)."""
    if abs(z) < 0.2:
        total = 0.0 + 0.0j
        term = 1.0 + 0.0j
        for n in range(14):
            total += term / math.factorial(n + 1)
            term *= z
        return total
    x, y = z.real, z.imag
    em1 = complex(math.expm1(x) * math.cos(y) - 2.0 * math.sin(0.5 * y) ** 2, math.exp(x) * math.sin(y))
    return em1 / z


def psi1(z: complex) -> complex:
    """((z - 1) e^z + 1) / z^2 = sum_n z^n (n + 1) / (n + 2)!."""
    if abs(z) < 0.2:
        total = 0.0 + 0.0j
        term = 1.0 + 0.0j
        for n in range(14):
            total += term * (n + 1) / math.factorial(n + 2)
            term *= z
        return total
    return ((z - 1.0) * cmath.exp(z) + 1.0) / (z * z)


def J0(kappa: complex, h_s: float) -> complex:
    return h_s * phi1(kappa * h_s)


def J1(kappa: complex, h_s: float) -> complex:
    return h_s * h_s * psi1(kappa * h_s)


def free_interval(omega_rad_s: float, zeta: float, h_s: float, q0: float, v0: float, P: complex, Omega_rad_s: float,
                  g0: float, g1: float) -> tuple[float, float, float, float, float]:
    """Exact state after h and the work / dissipation integrals for f(t) = Re(P e^(i W t)) + g0 + g1 t.

    Returns (q1, v1, W_harmonic, W_linear, D)."""
    w = omega_rad_s
    sig = zeta * w
    wd = w * math.sqrt(1.0 - zeta * zeta)
    lam = complex(-sig, wd)
    W = Omega_rad_s
    H = 1.0 / complex(w * w - W * W, 2.0 * sig * W)
    Q = P * H
    VP = 1j * W * Q
    qL0 = g0 / (w * w) - 2.0 * zeta * g1 / (w ** 3)
    vL = g1 / (w * w)
    a = q0 - Q.real - qL0
    b = v0 - VP.real - vL
    Cq = complex(a, -(b + sig * a) / wd)
    Cv = lam * Cq
    El = cmath.exp(lam * h_s)
    EW = cmath.exp(1j * W * h_s)
    q1 = (Cq * El).real + (Q * EW).real + (g0 + g1 * h_s) / (w * w) - 2.0 * zeta * g1 / (w ** 3)
    v1 = (Cv * El).real + (VP * EW).real + vL
    iW = 1j * W
    Wh = 0.5 * (P * Cv * J0(iW + lam, h_s)).real + 0.5 * (P * Cv.conjugate() * J0(iW + lam.conjugate(), h_s)).real \
        + 0.5 * (P * VP * J0(2.0 * iW, h_s)).real + 0.5 * (P * VP.conjugate()).real * h_s \
        + vL * (P * J0(iW, h_s)).real
    Wl = (Cv * (g0 * J0(lam, h_s) + g1 * J1(lam, h_s))).real + (VP * (g0 * J0(iW, h_s) + g1 * J1(iW, h_s))).real \
        + vL * (g0 * h_s + 0.5 * g1 * h_s * h_s)
    I = 0.5 * (Cv * Cv * J0(2.0 * lam, h_s)).real + 0.5 * abs(Cv) ** 2 * J0(complex(-2.0 * sig, 0.0), h_s).real \
        + (Cv * VP * J0(lam + iW, h_s)).real + (Cv * VP.conjugate() * J0(lam - iW, h_s)).real \
        + 2.0 * vL * (Cv * J0(lam, h_s)).real + 0.5 * (VP * VP * J0(2.0 * iW, h_s)).real + 0.5 * abs(VP) ** 2 * h_s \
        + 2.0 * vL * (VP * J0(iW, h_s)).real + vL * vL * h_s
    D = 2.0 * sig * I
    return q1, v1, Wh, Wl, D


def modal_energy(x: np.ndarray, omega_rad_s: np.ndarray) -> float:
    """sum_k (v_k^2 + omega_k^2 q_k^2) / 2 (mass-normalised: joules)."""
    q = np.asarray(x)[:, 0]
    v = np.asarray(x)[:, 1]
    w = np.asarray(omega_rad_s)
    return float(0.5 * np.sum(v * v + w * w * q * q))


__all__ = ["check_mode", "state_matrix", "foh", "van_loan", "energy_grams", "modal_discretisation", "phi1", "psi1",
           "J0", "J1", "free_interval", "modal_energy"]
