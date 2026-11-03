"""Semi-discretisation (Insperger & Stepan 2002, 2004): the independent check of the lobe diagrams.

The linearised regenerative cut in modal coordinates q (cnctwin.dynamics.sld module doc)::

    q'' + 2 Z W q' + W^2 q = a Psi^T sum_j K_j(t) Psi (q(t) - q(t - tau_j))

is a linear delay-differential equation with T-periodic coefficients (T: the tooth period for a uniform pitch, the
revolution for a variable pitch or a turned part). The first-order (updated) semi-discretisation divides T into m
intervals of length dt; on interval i the periodic coefficients are replaced by their interval means
(analytic, sld.CutModel.interval_mean), the delayed displacement by the straight line between its samples at the
interval ends, and the resulting linear ODE is solved exactly (first-order-hold matrices from the augmented
exponential). The map over T (monodromy) acts on [x_i, q_(i-1) ... q_(i-r)]; the cut is stable when its spectral
radius is below 1. Delays that are not whole multiples of dt (variable pitch) are rounded to the nearest
interval (the per-revolution interval count is chosen so the rounding is below dt / 2).

The residual (static) flexibility of the FRF is not part of the semi-discretised model (modes only).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from .sld import TWO_PI, CutModel, ModalFRF


def _foh_identity(A: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, Gam0, Gam1) of x' = A x + u(t), u linear from u0 to u1 over dt: x1 = P x0 + Gam0 u0 + Gam1 u1."""
    n = A.shape[0]
    M = np.zeros((3 * n, 3 * n))
    M[:n, :n] = A * dt
    M[:n, n:2 * n] = np.eye(n) * dt
    M[n:2 * n, 2 * n:] = np.eye(n)
    E = expm(M)
    P = E[:n, :n]
    g0 = E[:n, n:2 * n]
    g1 = E[:n, 2 * n:]
    return P, g0 - g1, g1


def intervals(cut: CutModel, frf: ModalFRF, speed_rad_s: float, m: int, points_per_period: int) -> int:
    """Intervals per tooth period (per revolution for turning): at least m, and at least points_per_period per
    period of the highest mode over the delay (the delayed term is interpolated linearly; its error is second
    order in omega dt)."""
    w = abs(float(speed_rad_s))
    tau = TWO_PI / w if not cut.rotating else TWO_PI / (cut.n_teeth * w)
    f_max = float(np.max(frf.omega_rad_s)) / TWO_PI if len(frf.omega_rad_s) else 0.0
    return max(int(m), int(math.ceil(points_per_period * f_max * tau)))


def _setup(cut: CutModel, speed_rad_s: float, m: int):
    """(period T, intervals per period, dt, delays in intervals per tooth group, tooth groups); m intervals per
    tooth period (per revolution for turning)."""
    w = abs(float(speed_rad_s))
    if not cut.rotating:
        T = TWO_PI / w
        return T, int(m), T / m, [int(m)], [None]
    if cut.uniform_pitch:
        T = TWO_PI / (cut.n_teeth * w)
        return T, int(m), T / m, [int(m)], [None]
    T = TWO_PI / w
    n_int = int(m) * cut.n_teeth
    dt = T / n_int
    delays = [max(1, int(round(float(p) / w / dt))) for p in cut.pitch_rad]
    return T, n_int, dt, delays, list(range(cut.n_teeth))


def monodromy(cut: CutModel, frf: ModalFRF, speed_rad_s: float, depth_m: float, m: int,
              process_damping: bool = False) -> np.ndarray:
    """The semi-discretised map over one principal period (``process_damping``: the cut's linearised indentation
    damping a C(t) in the interval means as well)."""
    W = np.asarray(frf.omega_rad_s, float)
    Z = np.asarray(frf.zeta, float)
    Psi = np.asarray(frf.psi, float).T             # [3, n]
    n = len(W)
    T, n_int, dt, delays, groups = _setup(cut, speed_rad_s, m)
    r = max(delays)
    d = 2 * n + r * n
    w = abs(float(speed_rad_s))
    spin = cut.spin if cut.rotating else 1.0
    Phi = np.eye(d)
    a = float(depth_m)
    for i in range(n_int):
        th0 = spin * w * i * dt
        th1 = spin * w * (i + 1) * dt
        Kq_tot = Psi.T @ cut.interval_mean(th0, th1) @ Psi
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.diag(W * W) + a * Kq_tot
        A[n:, n:] = -np.diag(2.0 * Z * W)
        if process_damping:
            A[n:, n:] -= a * (Psi.T @ cut.damping_interval_mean(th0, th1, speed_rad_s) @ Psi)
        P, G0, G1 = _foh_identity(A, dt)
        new = np.empty_like(Phi)
        top = P @ Phi[:2 * n]
        for mj, g in zip(delays, groups):
            Kq = Kq_tot if g is None else Psi.T @ cut.interval_mean(th0, th1, tooth=g) @ Psi
            Bq = np.zeros((2 * n, n))
            Bq[n:, :] = -a * Kq
            rows_old = 2 * n + (mj - 1) * n
            q_old = Phi[rows_old:rows_old + n]                                  # q_(i - mj)
            q_new = Phi[0:n] if mj == 1 else Phi[2 * n + (mj - 2) * n:2 * n + (mj - 1) * n]   # q_(i + 1 - mj)
            top += G0 @ Bq @ q_old + G1 @ Bq @ q_new
        new[:2 * n] = top
        new[2 * n:3 * n] = Phi[0:n]
        if r > 1:
            new[3 * n:] = Phi[2 * n:2 * n + (r - 1) * n]
        Phi = new
    return Phi


def spectral_radius(cut: CutModel, frf: ModalFRF, speed_rad_s: float, depth_m: float, m: int,
                    points_per_period: int = 0, process_damping: bool = False) -> float:
    mm = intervals(cut, frf, speed_rad_s, m, points_per_period)
    return float(np.max(np.abs(np.linalg.eigvals(monodromy(cut, frf, speed_rad_s, depth_m, mm, process_damping)))))


def limit_at_speed(cut: CutModel, frf: ModalFRF, speed_rad_s: float, *, m: int, a_max: float, n_scan: int,
                   tol_rel: float, points_per_period: int = 0, process_damping: bool = False,
                   fine_ratio: float = 1.02, fine_span: float = 2.0) -> tuple[float, float]:
    """(a_lim, dominant multiplier modulus just above it): the first loss of stability. A geometric scan of n_scan
    depths up to a_max finds the first unstable depth a_u; the span [a_u / fine_span, a_u] is then scanned with the
    ratio fine_ratio, because period-doubling instability can form a narrow lens below a_u in which two real
    multipliers collide and the cut is briefly stable again (a coarse scan can step over the lens); bisection on the
    first unstable bracket. inf if stable up to a_max."""
    m = intervals(cut, frf, speed_rad_s, m, points_per_period)

    def rho(a):
        return spectral_radius(cut, frf, speed_rad_s, a, m, 0, process_damping)

    grid = np.geomspace(a_max * 1e-4, a_max, int(n_scan))
    a_u = None
    lo = 0.0
    for a in grid:
        if rho(a) > 1.0:
            a_u = float(a)
            break
        lo = float(a)
    if a_u is None:
        return math.inf, 0.0
    # the lens can lie below the last stable coarse depth (which may sit inside its stable window): scan the whole
    # span below a_u finely, from the deeper of a_u / fine_span and the last stable depth before it
    start = a_u / fine_span
    if rho(start) > 1.0:
        start = lo
    n_fine = int(math.ceil(math.log(a_u / start) / math.log(fine_ratio))) if start > 0.0 and a_u > start else 0
    hi = a_u
    lo = start
    for a in np.geomspace(start, a_u, max(n_fine, 1) + 1)[1:]:
        if rho(a) > 1.0:
            hi = float(a)
            break
        lo = float(a)
    rho_hi = rho(hi)
    while hi - lo > tol_rel * hi:
        mid = 0.5 * (lo + hi)
        rm = rho(mid)
        if rm > 1.0:
            hi, rho_hi = mid, rm
        else:
            lo = mid
    return hi, rho_hi


def multipliers(cut: CutModel, frf: ModalFRF, speed_rad_s: float, depth_m: float, m: int) -> np.ndarray:
    """Eigenvalues of the monodromy (Floquet multipliers), largest modulus first."""
    mu = np.linalg.eigvals(monodromy(cut, frf, speed_rad_s, depth_m, m))
    return mu[np.argsort(-np.abs(mu), kind="stable")]


def bifurcation_kind(mu: complex, tol: float = 1e-6) -> str:
    """'flip' (real negative critical multiplier, period doubling), 'fold' (real positive) or 'hopf'."""
    if abs(mu.imag) <= tol * max(abs(mu), 1.0):
        return "flip" if mu.real < 0.0 else "fold"
    return "hopf"


__all__ = ["monodromy", "spectral_radius", "limit_at_speed", "multipliers", "bifurcation_kind", "intervals"]
