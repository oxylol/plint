"""The moving rectangle / band source (Jaeger 1942) against published values."""

import math

import numpy as np
import pytest

from cnctwin.thermal import kernels as K
from cnctwin.thermal.response import edge_response, gauss_legendre, phi_inf_centre

GLX, GLW = gauss_legendre(24)
Q = 1.0e7
KK = 40.0
ALPHA = 1.0e-5


def rise(l, w, v, t, x, z=0.0):
    return K.moving_rect_rise(Q, l, w, v, KK, ALPHA, t, x, 0.0, z, GLX, GLW)


def test_fast_band_source_mean_and_maximum():
    """Jaeger (1942), band of width 2l at Peclet L = V l / 2 alpha >= 5: mean 0.754 q l / (k sqrt L)
    (0.752 from the leading term), maximum at the trailing edge 2 / sqrt(pi) = 1.128 q l / (k sqrt L)."""
    l = 1e-4
    for peclet in (20.0, 100.0):
        v = 2.0 * ALPHA * peclet / l
        scale = Q * l / (KK * math.sqrt(peclet))
        nodes, weights = np.polynomial.legendre.leggauss(24)
        mean = 0.5 * sum(wq * rise(l, 0.0, v, 10.0, l * xq) for xq, wq in zip(nodes, weights)) / scale
        peak = max(rise(l, 0.0, v, 10.0, l * x) for x in np.linspace(-1.0, 0.0, 201)) / scale
        assert mean == pytest.approx(0.754, rel=0.03)
        assert peak == pytest.approx(2.0 / math.sqrt(math.pi), rel=0.03)
    # the approach to the high-speed limit is from below for the mean (diffusion along x)
    assert mean > 0.74


def test_stationary_square_source_centre_reaches_carslaw_jaeger_value():
    """A uniformly heated square 2a x 2a at rest: steady centre rise 1.1222 q a / k (Carslaw & Jaeger)."""
    a = 1e-3
    long_time = 1e5 * a * a / ALPHA
    centre = rise(a, a, 0.0, long_time, 0.0) / (Q * a / KK)
    assert centre == pytest.approx(phi_inf_centre(1.0), rel=5e-3)
    assert phi_inf_centre(1.0) == pytest.approx(1.1222, abs=2e-4)
    resp = edge_response()
    im = int(np.argmin(np.abs(resp.log_m)))
    assert resp.phi_inf_m[im] == pytest.approx(0.9464, abs=5e-4)       # mean over the square


def test_short_time_limit_is_the_one_dimensional_solution():
    """Wide source, short time: theta = 2 q sqrt(alpha t / pi) / k at its centre."""
    t = 1e-3
    got = rise(1e-2, 1e-2, 0.0, t, 0.0)
    assert got == pytest.approx(2 * Q * math.sqrt(ALPHA * t / math.pi) / KK, rel=1e-6)


def test_temperature_ahead_decays_and_behind_cools():
    l = 2e-4
    v = 1.0
    behind = [rise(l, 1e-3, v, 1.0, -l * f) for f in (1.0, 3.0, 10.0)]
    assert behind[0] > behind[1] > behind[2] > 0.0
    assert rise(l, 1e-3, v, 1.0, 5 * l) < 0.05 * behind[0]
    deeper = [rise(l, 1e-3, v, 1.0, -l, z) for z in (0.0, 1e-5, 5e-5)]
    assert deeper[0] > deeper[1] > deeper[2]
