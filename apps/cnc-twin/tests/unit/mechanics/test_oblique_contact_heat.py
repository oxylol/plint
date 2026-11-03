"""Oblique transformation, edge/ploughing/flank contact, heat-source solutions and partitions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from cnctwin.mechanics import abi, contact as C, heat as H, kernels as K, oblique as O


# ------------------------------------------------------------------------------ oblique
def test_baa_reduces_to_merchant_at_zero_inclination():
    tau, phi, beta, alpha = 5e8, 0.4, 0.6, 0.1
    ktc, krc, kac = O.cutting_coefficients(tau, phi, beta, alpha, 0.0)
    fc, ft = C.merchant_forces(tau, 1.0, phi, beta, alpha)
    assert float(ktc) == pytest.approx(fc, rel=1e-12)
    assert float(krc) == pytest.approx(ft, rel=1e-12)
    assert float(kac) == pytest.approx(0.0, abs=1e-6)


def test_baa_inclination_effects():
    tau, phi, beta, alpha = 5e8, 0.4, 0.6, 0.1
    k0 = O.cutting_coefficients(tau, phi, beta, alpha, 0.0)
    kp = O.cutting_coefficients(tau, phi, beta, alpha, math.radians(30))
    km = O.cutting_coefficients(tau, phi, beta, alpha, math.radians(-30))
    assert float(kp[2]) > 0.0 and float(km[2]) == pytest.approx(-float(kp[2]), rel=1e-12)
    assert float(kp[0]) == pytest.approx(float(km[0]), rel=1e-12)
    assert float(kp[1]) != pytest.approx(float(k0[1]), rel=1e-3)


def test_stabler_and_equivalent_rake():
    assert O.stabler_flow_angle(0.3) == 0.3
    assert O.equivalent_rake(0.1, 0.0, 0.0) == pytest.approx(0.1)
    i = math.radians(40)
    expect = math.asin(math.sin(i) ** 2 + math.cos(i) ** 2 * math.sin(0.1))
    assert O.equivalent_rake(0.1, i, i) == pytest.approx(expect)
    assert O.equivalent_rake(0.1, i, i) > 0.1


def test_chip_flow_direction():
    d = O.chip_flow_direction(0.0, 0.0, 0.0)
    assert np.allclose(d, [0.0, 1.0, 0.0])
    d = O.chip_flow_direction(math.radians(20), 0.0, 0.0)
    assert np.allclose(d, [-math.sin(math.radians(20)), math.cos(math.radians(20)), 0.0])
    d = O.chip_flow_direction(0.1, 0.4, 0.4)
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert d[2] < 0.0          # chip runs against the work's along-edge motion
    assert float(O.chip_velocity_ratio(0.4, 0.4, 0.4)) == pytest.approx(0.4)


# ------------------------------------------------------------------------------ edge contact
def _edge_by_quadrature(depth, r, stag, rake, p, tau):
    """Independent 2-D integration of pressure and friction over the arc (no rake face)."""
    w_d = math.acos(1 - depth / r)
    w_s = math.acos(1 - min(stag, depth) / r)

    def ft(w):
        down = w <= w_s
        return (p * math.sin(w) + (tau if down else -tau) * math.cos(w)) * r

    def fr(w):
        down = w <= w_s
        return (p * math.cos(w) - (tau if down else -tau) * math.sin(w)) * r

    pts = [w_s] if w_s < w_d else None
    return (integrate.quad(ft, 0.0, w_d, points=pts)[0], integrate.quad(fr, 0.0, w_d, points=pts)[0])


@pytest.mark.parametrize("depth,stag", [(1e-5, 1.1e-5), (5e-6, 1.1e-5), (1.5e-5, 8e-6)])
def test_edge_forces_against_quadrature(depth, stag):
    out = np.zeros(abi.N_PL)
    K.edge_contact_forces(depth, 4e-5, stag, 0.05, 0.0, 2.6e9, 5e8, out)
    ft, fr = _edge_by_quadrature(depth, 4e-5, stag, 0.05, 2.6e9, 5e8)
    assert out[abi.PL_FT] == pytest.approx(ft, rel=1e-9)
    assert out[abi.PL_FR] == pytest.approx(fr, rel=1e-9)
    assert out[abi.PL_FA] == pytest.approx(0.0, abs=1e-9)
    assert out[abi.PL_CONTACT] == pytest.approx(4e-5 * math.acos(1 - depth / 4e-5), rel=1e-12)


def test_edge_force_inclination_projection():
    r, hm, p, tau, i = 4e-5, 1.1e-5, 2.6e9, 5e8, math.radians(30)
    out = np.zeros(abi.N_PL)
    K.edge_contact_forces(hm, r, hm, 0.0, i, p, tau, out)
    ws = math.acos(1 - hm / r)
    expect_t = p * hm * math.cos(i) + tau * r * math.sin(ws) * math.cos(i) ** 2 + tau * r * ws * math.sin(i) ** 2
    expect_a = p * hm * math.sin(i) + tau * r * math.sin(i) * math.cos(i) * (math.sin(ws) - ws)
    assert out[abi.PL_FT] == pytest.approx(expect_t, rel=1e-12)
    assert out[abi.PL_FA] == pytest.approx(expect_a, rel=1e-12)


def test_ploughing_continuity_and_area(steel):
    r = 4e-5
    hm = C.min_chip_thickness(steel, r)
    assert hm == pytest.approx(0.27 * r)
    edge = C.edge_forces(r, hm, 0.0, 0.0, 5e8, 0.5)
    below = C.ploughing_forces(hm * (1 - 1e-9), r, hm, 0.0, 0.0, 5e8, 0.5)
    assert below[abi.PL_FR] == pytest.approx(edge[abi.PL_FR], rel=1e-6)
    assert C.ploughing_forces(0.0, r, hm, 0.0, 0.0, 5e8, 0.5)[abi.PL_FR] == 0.0
    w = math.acos(1 - hm / r)
    area = integrate.quad(lambda x: hm - (r - math.sqrt(r * r - x * x)), 0.0, r * math.sin(w))[0]
    assert C.interference_area(hm, r) == pytest.approx(area, rel=1e-9)
    k_sp, src = C.indentation_coefficient(steel, edge[abi.PL_FR], hm, r, 0.0)
    assert src == "model" and k_sp * area == pytest.approx(edge[abi.PL_FR], rel=1e-9)
    x, st, sr = C.ploughing_shape(r, hm, 0.0, 0.0, 5e8, 0.5)
    assert st[-1] == pytest.approx(1.0) and sr[0] == 0.0 and np.all(np.diff(sr) >= 0.0)


def test_chisel_indentation_pushes_and_cuts():
    f = C.indentation_forces(1e-4, 3e-5, 9e-6, math.radians(-55), 0.0, 5e8, 0.5)
    assert f[abi.PL_FR] > 0.0 and f[abi.PL_FT] > 0.0
    deeper = C.indentation_forces(2e-4, 3e-5, 9e-6, math.radians(-55), 0.0, 5e8, 0.5)
    assert deeper[abi.PL_FR] > f[abi.PL_FR] and deeper[abi.PL_AREA] > f[abi.PL_AREA]
    assert C.uses_indentation(4, 0.0) and C.uses_indentation(0, math.radians(-50)) and not C.uses_indentation(0, 0.0)


def test_flank_stresses():
    s, t = C.flank_stresses(5e-5, 2.5e9, 5e8, 1e-4)
    assert (s, t) == (2.5e9, 5e8)
    s, t = C.flank_stresses(3e-4, 2.5e9, 5e8, 1e-4)
    assert s == pytest.approx(2.5e9 * (1e-4 + 2e-4 / 3) / 3e-4)
    x = np.linspace(1e-4, 3e-4, 100001)
    numeric = (1e-4 * 2.5e9 + np.trapezoid(2.5e9 * ((3e-4 - x) / 2e-4) ** 2, x)) / 3e-4
    assert s == pytest.approx(numeric, rel=1e-6)
    arr_s, _ = C.flank_stresses(np.array([0.0, 5e-5, 3e-4]), 2.5e9, 5e8, 1e-4)
    assert arr_s[0] == 0.0 and arr_s[1] == 2.5e9


def test_shear_angle_relations():
    assert C.shear_angle_merchant(0.5, 0.1) == pytest.approx(0.25 * math.pi - 0.2)
    assert C.shear_angle_lee_shaffer(0.5, 0.1) == pytest.approx(0.25 * math.pi - 0.4)


# ------------------------------------------------------------------------------ heat sources
def _rect_mean_polar(m):
    """Independent numerical mean factor: I = 4 int_0^1 int_0^m (1 - u)(m - w) / sqrt(u^2 + w^2) dw du
    in polar coordinates (the 1/rho cancels the Jacobian; the radial integral is analytic)."""
    theta0 = math.atan(m)

    def radial(theta):
        c, s_ = math.cos(theta), math.sin(theta)
        r = 1.0 / c if theta < theta0 else m / s_
        return m * r - (c * m + s_) * r * r / 2.0 + c * s_ * r ** 3 / 3.0
    total = integrate.quad(radial, 0.0, theta0)[0] + integrate.quad(radial, theta0, 0.5 * math.pi)[0]
    return 4.0 * total / (2.0 * math.pi * m)


def test_rect_area_factor():
    # unit square: 4 asinh(1) - (4/3)(sqrt 2 - 1) over 2 pi
    expect = (4 * math.asinh(1.0) - 4.0 / 3.0 * (math.sqrt(2.0) - 1.0)) / (2 * math.pi)
    assert H.stationary_rect_factor(1.0) == pytest.approx(expect, rel=1e-12)
    for m in (0.5, 3.0):
        # closed form of the self-potential I(1, m) = int int 1/r over the rectangle
        i_closed = 2 * m * m * math.asinh(1 / m) + 2 * m * math.asinh(m) + 2 / 3 * (1 + m ** 3) \
            - 2 / 3 * (1 + m * m) ** 1.5
        assert H.stationary_rect_factor(m) == pytest.approx(i_closed / (2 * math.pi * m), rel=1e-12)
    assert H.stationary_rect_factor(1e4) == pytest.approx((math.log(2e4) + 0.5) / math.pi, rel=1e-3)


def test_rect_area_factor_numerical():
    for m in (0.2, 1.0, 2.0, 7.5):
        assert H.stationary_rect_factor(m) == pytest.approx(_rect_mean_polar(m), rel=1e-9)


def test_band_source_limits():
    for L in (50.0, 200.0):
        assert H.band_mean_factor(L) == pytest.approx(4.0 / (3.0 * math.sqrt(math.pi)) / math.sqrt(L), rel=2e-2)
    # small L: grows like ln(1/L) / pi (2-D stationary band)
    assert H.band_mean_factor(0.01) > H.band_mean_factor(0.1) > H.band_mean_factor(1.0)
    assert H.moving_strip_factor(1e-4, 2.0) == pytest.approx(2.0 * H.stationary_rect_factor(2.0))


def test_band_source_against_direct_line_source_superposition():
    """Mean temperature of the band from the moving line source e^{-u} K0(|u|) by a 2-D quadrature."""
    from scipy import special
    L = 2.0

    def theta(X):
        f = lambda u: math.exp(-u) * special.k0(abs(u))  # noqa: E731
        return integrate.quad(f, X - L, X + L, points=[0.0] if X - L < 0 < X + L else None, limit=200)[0] / (
            math.pi * L)
    mean = integrate.quad(theta, -L, L, limit=200)[0] / (2 * L)
    assert H.band_mean_factor(L) == pytest.approx(mean, rel=1e-5)


def test_blok_partition_limits(steel, carbide):
    assert H.blok_partition_to_tool(1e8, 1e-6, 1e-6, 600.0, 600.0) == pytest.approx(0.5)
    assert H.blok_partition_to_tool(1e8, 1e-6, 1e-12, 600.0, 600.0) == pytest.approx(1.0, abs=1e-5)
    assert H.blok_partition_to_tool(1e8, 1e-6, 1e-6, 300.0, 2000.0) == 0.0
    slow = H.flank_partition_to_tool(steel, carbide, 2e-4, 2e-3, 0.2, 5e8 * 0.2, 700.0, 500.0)
    fast = H.flank_partition_to_tool(steel, carbide, 2e-4, 2e-3, 5.0, 5e8 * 5.0, 700.0, 500.0)
    assert 0.0 <= fast < slow <= 1.0          # faster sliding: the moving work takes more
    assert H.flank_partition_to_tool(steel, None, 2e-4, 2e-3, 1.0, 1e8, 700.0, 500.0) == 0.0


def test_tool_resistance_transient_cap(carbide):
    steady = H.tool_side_resistance(3e-4, 600.0, carbide, 2e-3)
    short = H.tool_side_resistance(3e-4, 600.0, carbide, 2e-3, contact_time_s=1e-5)
    assert short < steady
    k = float(np.interp(600.0, carbide.k_T_K, carbide.k_W_mK))
    c = float(np.interp(600.0, carbide.cp_T_K, carbide.cp_J_kgK))
    assert short == pytest.approx(4 / 3 * math.sqrt(k / (carbide.rho_kg_m3 * c) * 1e-5 / math.pi) / k, rel=1e-12)
