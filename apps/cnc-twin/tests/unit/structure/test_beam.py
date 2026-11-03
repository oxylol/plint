"""Timoshenko beam elements: cantilever frequencies and tip deflection against Euler-Bernoulli and Timoshenko
theory, the simply supported Timoshenko frequency equation, Cowper's coefficients and the foundation matrix."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.structure import beam as B
from cnctwin.structure import modal as MO

E, NU, RHO = 210e9, 0.3, 7800.0


def chain(L, sec, n, fixed_dofs):
    z = B.subdivide([0.0, L], L / n)
    asm = B.BeamAssembly(len(z))
    for i in range(len(z) - 1):
        asm.add_element(i, i + 1, z[i + 1] - z[i], sec)
    free = np.setdiff1d(np.arange(6 * len(z)), fixed_dofs(len(z)))
    return asm, free, z


def test_cantilever_tip_deflection_is_exact():
    d, L, P = 0.01, 0.08, 120.0
    sec = B.circle(d, E, NU, RHO)
    asm, free, z = chain(L, sec, 3, lambda n: np.arange(6))
    K = asm.K[np.ix_(free, free)]
    f = np.zeros(len(free))
    tip = 6 * (len(z) - 1) - 6
    f[tip + B.UX] = P
    u = np.linalg.solve(K, f)
    I = math.pi * d ** 4 / 64.0
    A = math.pi * d * d / 4.0
    expected = P * L ** 3 / (3 * E * I) + P * L / (B.cowper_circle(NU) * B.shear_modulus(E, NU) * A)
    assert u[tip + B.UX] == pytest.approx(expected, rel=1e-12)
    # the slope at the tip: P L^2 / (2 E I) (shear does not rotate the section)
    assert u[tip + B.RY] == pytest.approx(P * L * L / (2 * E * I), rel=1e-12)


def test_cantilever_frequencies_against_euler_bernoulli():
    d, L = 0.01, 0.2                      # L / d = 20: Timoshenko correction below 0.2 %
    sec = B.circle(d, E, NU, RHO)
    asm, free, _ = chain(L, sec, 40, lambda n: np.arange(6))
    lam, _ = MO.dense_modes(asm.K[np.ix_(free, free)], asm.M[np.ix_(free, free)])
    f = MO.freq_Hz(lam)
    I = math.pi * d ** 4 / 64.0
    A = math.pi * d * d / 4.0
    c = math.sqrt(E * I / (RHO * A)) / (2 * math.pi * L * L)
    # shear and rotary inertia lower the frequency; the first-order estimate of the drop is
    # c_n = (beta_n r / L)^2 (1 + E / (k G)) / 2 (r = d / 4); the converged Timoshenko value lies within
    # 0.8 c_n .. 2.5 c_n below Euler-Bernoulli (0.14 %, 1.0 %, 2.3 % here)
    kG = B.cowper_circle(NU) * B.shear_modulus(E, NU)
    for beta in (1.875104069, 4.694091133, 7.854757438):
        eb = beta * beta * c
        k = int(np.argmin(np.abs(f - eb)))
        cn = 0.5 * (beta * d / 4.0 / L) ** 2 * (1.0 + E / kG)
        assert 1.0 - 2.5 * cn < f[k] / eb < 1.0 - 0.8 * cn
    assert f[0] == pytest.approx(f[1], rel=1e-8)   # x and y pair (degenerate within the eigen solver precision)


def test_simply_supported_timoshenko_frequency_equation():
    d, L = 0.01, 0.05                     # stubby: L / d = 5
    sec = B.circle(d, E, NU, RHO)
    asm, free, z = chain(L, sec, 40, lambda n: [0, 1, 2, 5, 6 * (n - 1), 6 * (n - 1) + 1])
    lam, _ = MO.dense_modes(asm.K[np.ix_(free, free)], asm.M[np.ix_(free, free)])
    f = MO.freq_Hz(lam)
    I = math.pi * d ** 4 / 64.0
    A = math.pi * d * d / 4.0
    G = B.shear_modulus(E, NU)
    kGA = B.cowper_circle(NU) * G * A
    kn = math.pi / L
    # det [[rho A w2 - kGA k2, kGA k], [kGA k, rho I w2 - E I k2 - kGA]] = 0, lower root
    c2 = RHO * A * RHO * I
    c1 = -(RHO * A * (E * I * kn ** 2 + kGA) + RHO * I * kGA * kn ** 2)
    c0 = kGA * kn ** 2 * (E * I * kn ** 2 + kGA) - (kGA * kn) ** 2
    w2 = (-c1 - math.sqrt(c1 * c1 - 4 * c2 * c0)) / (2 * c2)
    assert f[0] == pytest.approx(math.sqrt(w2) / (2 * math.pi), rel=1e-4)
    assert f[0] < math.pi ** 2 / (2 * math.pi * L * L) * math.sqrt(E * I / (RHO * A))


def test_cowper_and_sections():
    assert B.cowper_circle(0.3) == pytest.approx(6 * 1.3 / (7 + 1.8))
    assert B.cowper_rectangle(0.3) == pytest.approx(13.0 / 15.3)
    tube = B.circle(0.02, E, NU, RHO, 0.01)
    assert tube.A_m2 == pytest.approx(math.pi * (0.02 ** 2 - 0.01 ** 2) / 4)
    assert tube.kx < B.cowper_circle(NU)
    rect = B.rectangle(0.02, 0.01, E, NU, RHO)
    assert rect.Ix_m4 == pytest.approx(0.02 * 0.01 ** 3 / 12) and rect.Iy_m4 == pytest.approx(0.01 * 0.02 ** 3 / 12)
    with pytest.raises(ValueError):
        B.circle(0.01, E, NU, RHO, 0.02)


def test_foundation_stiffness_of_a_rigid_translation_and_rotation():
    L, k = 0.03, 4e9
    Kf = B.foundation(L, k, 2e9, 1e3)
    u = np.zeros(12)
    u[B.UX] = u[6 + B.UX] = 1.0                       # rigid translation: total force k L
    assert float(u @ Kf @ u) == pytest.approx(k * L, rel=1e-12)
    th = 1e-3                                          # rigid rotation about the element centre: k L^3 / 12
    u = np.zeros(12)
    u[B.UX], u[6 + B.UX] = -th * L / 2, th * L / 2
    u[B.RY] = u[6 + B.RY] = th
    assert float(u @ Kf @ u) == pytest.approx(k * L ** 3 / 12 * th * th, rel=1e-12)
    assert np.allclose(Kf, Kf.T)
