"""Periodic heating and cooling of the cutting edge (Stephenson & Ali 1992)."""

import math

import numpy as np
import pytest

from cnctwin.thermal import kernels as K
from cnctwin.thermal import reference as R
from cnctwin.thermal.response import edge_response, gauss_legendre

A = 1.5e-4
B = 2.5e-4
KK = 70.0
ALPHA = 2.3e-5
GLX, GLW = gauss_legendre(3)


def pair(tau, th, period, ncyc, ndirect=4):
    resp = edge_response()
    im, wm = K.m_index(resp, B / A)
    return K.periodic_pair(resp, tau, th, period, ncyc, A, KK, ALPHA, int(im), wm, ndirect, GLX, GLW)


def unit(t):
    resp = edge_response()
    im, wm = K.m_index(resp, B / A)
    return K.unit_pair(resp, t, A, KK, ALPHA, int(im), wm)


def test_unit_response_tables_match_quadrature():
    for t in (1e-6, 1e-4, 1e-3, 0.1, 10.0):
        c, m = unit(t)
        assert c == pytest.approx(R.unit_response_ref(t, A, B, KK, ALPHA, 0), rel=2e-4)
        assert m == pytest.approx(R.unit_response_ref(t, A, B, KK, ALPHA, 1), rel=2e-4)


def test_full_duty_is_continuous_heating():
    period = 2e-3
    for ncyc in (3, 50, 2000):
        c, _ = pair(period, period, period, ncyc)
        cont, _ = unit(ncyc * period)
        assert c == pytest.approx(cont, rel=2e-3)


def test_isolated_pulse_peak_is_the_single_pulse_response():
    th = 1e-3
    c, _ = pair(th, th, 50.0, 3)
    single, _ = unit(th)
    assert c == pytest.approx(single, rel=1e-3)


def test_periodic_solution_matches_brute_force_superposition():
    th, period = 1e-3, 6e-3
    for ncyc in (5, 300):
        for tau in (0.2 * th, th, period):
            c, m = pair(tau, th, period, ncyc)
            assert c == pytest.approx(R.periodic_ref(tau, th, period, ncyc, A, B, KK, ALPHA, 0), rel=3e-3)
            assert m == pytest.approx(R.periodic_ref(tau, th, period, ncyc, A, B, KK, ALPHA, 1), rel=3e-3)


def test_cycle_mean_tends_to_duty_times_steady_rise():
    """At long times the cycle mean of the periodic field is the steady rise of the mean flux."""
    th, period = 1e-3, 4e-3
    ncyc = 1.0e7
    taus = np.linspace(0.0, period, 401)[1:]
    values = [pair(t, th, period, ncyc)[0] for t in taus]
    mean = float(np.mean(values))
    resp = edge_response()
    im, wm = K.m_index(resp, B / A)
    steady = A / KK * ((1 - wm) * resp.phi_inf_c[int(im)] + wm * resp.phi_inf_c[int(im) + 1])
    assert mean == pytest.approx(th / period * steady, rel=0.03)


def test_interrupted_cutting_is_cooler_and_amplitude_grows_with_period():
    """Stephenson & Ali: interrupted edges run cooler than continuous ones at the same flux, and the
    heating/cooling amplitude per revolution grows when the cycle is slower (same duty)."""
    duty = 0.25
    amplitudes = []
    for period in (1e-3, 4e-3, 16e-3):
        th = duty * period
        hot = pair(th, th, period, 1e6)[0]
        cold = pair(period, th, period, 1e6)[0]
        amplitudes.append(hot - cold)
        continuous = unit(1e6 * period)[0]
        assert hot < continuous
    assert amplitudes[0] < amplitudes[1] < amplitudes[2]
