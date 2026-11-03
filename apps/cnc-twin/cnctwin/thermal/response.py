"""Dimensionless step response of a uniformly heated rectangle on a half space (edge model).

A flux q switched on at t = 0 over |x| < a, |y| < b on the surface of a half space (adiabatic
elsewhere) raises the surface temperature by (Carslaw & Jaeger 1959, 10.4: product of the
instantaneous plane source in depth and two strip factors)::

    theta(x, y, t) = (q / rho c) int_0^t (pi alpha tau)^-1/2 X(x, tau) Y(y, tau) dtau

With Fo = alpha t / a^2 and m = b / a this is theta = (q a / k) Phi(Fo, m), where

* centre (x = y = 0):  Phi_c = int_0^Fo (pi s)^-1/2 erf(1 / 2 sqrt s) erf(m / 2 sqrt s) ds
* area mean:           Phi_m = int_0^Fo (pi s)^-1/2 M(1, s) M(m, s) ds,
                       M(c, s) = erf(X) - (1 - exp(-X^2)) / (X sqrt pi),  X = c / sqrt s

Both tend to Phi_inf - 2 m / (pi^1.5 sqrt Fo); Phi_c,inf = (2/pi)(asinh m + m asinh(1/m)).
The tables (log10 Phi on a log10 Fo x log10 m grid) are computed once per process with
16-point Gauss-Legendre per grid interval in u = sqrt(s) (smooth integrand).
"""

from __future__ import annotations

import functools
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erf

from .abi import EdgeResponse

LOG_FO_RANGE = (-8.0, 8.0)
FO_PER_DECADE = 24
LOG_M_RANGE = (-2.5, 2.5)
M_PER_DECADE = 32
GAUSS_POINTS = 16


def _strip_mean(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    x = c / np.sqrt(s)
    return erf(x) - (1.0 - np.exp(-x * x)) / (x * math.sqrt(math.pi))


def _integrand(u: np.ndarray, m: float, which: int) -> np.ndarray:
    """(2 / sqrt pi) g(u^2): d Phi / d u with s = u^2."""
    s = u * u
    if which == 0:
        g = erf(1.0 / (2.0 * u)) * erf(m / (2.0 * u))
    else:
        g = _strip_mean(np.ones_like(s), s) * _strip_mean(np.full_like(s, m), s)
    return 2.0 / math.sqrt(math.pi) * g


def cumulative_phi(fo: np.ndarray, m: float, which: int) -> np.ndarray:
    """Phi at the ascending Fo points by cumulative Gauss-Legendre integration in u = sqrt(Fo)."""
    nodes, weights = leggauss(GAUSS_POINTS)
    u_pts = np.concatenate(([0.0], np.sqrt(fo)))
    lo = u_pts[:-1]
    hi = u_pts[1:]
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    u = mid[:, None] + half[:, None] * nodes[None, :]
    pieces = (_integrand(u, m, which) * weights[None, :]).sum(axis=1) * half
    return np.cumsum(pieces)


def phi_inf_centre(m: float) -> float:
    """Steady centre rise of a uniformly heated rectangle, in q a / k (Carslaw & Jaeger 1959)."""
    return 2.0 / math.pi * (math.asinh(m) + m * math.asinh(1.0 / m))


@functools.lru_cache(maxsize=1)
def edge_response() -> EdgeResponse:
    """The response tables used by the edge kernel (computed once per process)."""
    log_fo = np.arange(LOG_FO_RANGE[0], LOG_FO_RANGE[1] + 0.5 / FO_PER_DECADE, 1.0 / FO_PER_DECADE)
    log_m = np.arange(LOG_M_RANGE[0], LOG_M_RANGE[1] + 0.5 / M_PER_DECADE, 1.0 / M_PER_DECADE)
    fo = 10.0 ** log_fo
    tables = np.zeros((2, log_m.size, log_fo.size))
    inf = np.zeros((2, log_m.size))
    for im, lm in enumerate(log_m):
        m = 10.0 ** lm
        for which in (0, 1):
            phi = cumulative_phi(fo, m, which)
            tables[which, im] = np.log10(phi)
            inf[which, im] = phi[-1] + 2.0 * m / (math.pi ** 1.5 * math.sqrt(fo[-1]))
    return EdgeResponse(np.ascontiguousarray(log_fo), np.ascontiguousarray(log_m),
                        np.ascontiguousarray(tables[0]), np.ascontiguousarray(tables[1]),
                        np.ascontiguousarray(inf[0]), np.ascontiguousarray(inf[1]))


@functools.lru_cache(maxsize=None)
def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1] (Abramowitz & Stegun 25.4.29)."""
    nodes, weights = leggauss(int(n))
    return np.ascontiguousarray(nodes), np.ascontiguousarray(weights)


def heating_phase_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes on (0, 1) of the in-cut (heating) interval and weights summing to 1."""
    nodes, weights = gauss_legendre(n)
    return np.ascontiguousarray(0.5 * (nodes + 1.0)), np.ascontiguousarray(0.5 * weights)
