"""Surface kernels against their plain-Python references (architecture 5.5) and one compiled signature each."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.surface import abi as A
from cnctwin.surface import kernels as K
from cnctwin.surface import reference as R


def test_every_kernel_has_one_signature():
    names = [n for n in dir(K) if hasattr(getattr(K, n), "signatures") and getattr(K, n).py_func.__module__ == K.__name__]
    assert set(names) >= {"pass_polyline", "envelope_profile", "gaussian_mean", "profile_params", "silhouette",
                          "wire_centre", "_rsm_core", "_raster_min"}
    for name in names:
        assert len(getattr(K, name).signatures) == 1, name


def _mixed_passes():
    """Outline, parabola, direct and additive passes with BUE overcut and a minimum chip thickness."""
    rng = np.random.default_rng(3)
    nose = 0.8e-3
    th = np.linspace(-1.2, 1.2, 181)
    outline = np.stack([-nose * (1.0 - np.cos(th)), nose + nose * np.sin(th)], axis=1)   # (x_T, z_T)
    bump = np.stack([np.linspace(-1e-5, 1e-5, 9), 2e-7 * (1.0 - np.linspace(-1, 1, 9) ** 2)], axis=1)
    T = A.new_templates([outline, bump])
    n = 24
    P = A.new_passes(n)
    for k in range(n):
        P.kind[k] = A.KIND_ADD if k % 7 == 6 else A.KIND_CUT
        P.mode[k] = A.MODE_DIRECT if P.kind[k] == A.KIND_ADD else (A.MODE_PARABOLA if k % 3 == 0 else A.MODE_OUTLINE)
        P.tpl[k] = 1 if P.kind[k] == A.KIND_ADD else 0
        P.s[k] = k * 1.1e-4 + rng.uniform(-2e-6, 2e-6)
        P.w[k] = rng.uniform(-1e-6, 1e-6)
        P.m[k] = rng.uniform(-1e-7, 1e-7)
        P.basis[k] = [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.05, 0.0, 1.0, 0.0, 1.0, 0.0]
        if P.mode[k] == A.MODE_PARABOLA:
            P.basis[k, A.B_VD:A.B_VM + 1] = [-1.0, 0.002, 0.0]
        P.elem_rz[k] = [0.0, nose]
        P.curv[k] = 200.0
        P.hmin[k] = 1.5e-6
        P.half[k] = np.inf if P.kind[k] == A.KIND_ADD else 4.0e-4
        P.t_max[k] = 4.0e-4
        P.bue[k] = [5e-7, 2e-5] if k % 5 == 2 else [0.0, 0.0]
    return P, T


def test_pass_polyline_matches_reference():
    P, T = _mixed_passes()
    cap = 300
    ps, pw = np.zeros(cap), np.zeros(cap)
    for k in range(len(P.s)):
        n = K.pass_polyline(P, T, k, ps, pw, 64)
        rs, rw = R.pass_polyline_ref(P, T, k, cap, 64)
        assert n == len(rs)
        np.testing.assert_allclose(ps[:n], rs, rtol=0.0, atol=1e-18)
        got = np.where(np.isnan(pw[:n]), -1.0, pw[:n])
        want = np.array([-1.0 if v is None else v for v in rw])
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-18)


@pytest.mark.parametrize("recovery", [0.0, 1.0])
def test_envelope_profile_matches_reference(recovery):
    P, T = _mixed_passes()
    u0, du, n, cap = -5e-4, 5e-7, 7000, 300
    G = A.new_grid(u0, du, n, cap)
    K.envelope_profile(P, T, G, 64, recovery)
    z_ref, hits_ref = R.envelope_profile_ref(P, T, u0, du, n, cap, 64, recovery)
    assert np.array_equal(np.isfinite(G.z), np.isfinite(z_ref))
    ok = np.isfinite(z_ref)
    np.testing.assert_allclose(G.z[ok], z_ref[ok], rtol=0.0, atol=1e-18)
    assert np.array_equal(G.hits, hits_ref)
    assert np.all(np.isinf(G.tmp)) and np.all(G.tmp_slope == 0.0)      # scratch left clean


def test_envelope_of_circles_is_the_feed_mark_profile():
    """Circles of radius r every f: peak-to-valley r - sqrt(r^2 - f^2 / 4) (-> f^2 / 8 r)."""
    r, f = 0.8e-3, 0.2e-3
    th = np.linspace(-1.4, 1.4, 2001)
    T = A.new_templates([np.stack([r * np.sin(th), r * (1.0 - np.cos(th))], axis=1)])
    P = A.new_passes(30)
    P.kind[:] = A.KIND_CUT
    P.mode[:] = A.MODE_DIRECT
    P.tpl[:] = 0
    P.s[:] = np.arange(30) * f
    P.half[:] = 1.0
    G = A.new_grid(-1e-3, 2.5e-7, 30000, 3000)
    K.envelope_profile(P, T, G, 2, 0.0)
    u = -1e-3 + np.arange(30000) * 2.5e-7
    mid = G.z[(u > 5 * f) & (u < 20 * f)]
    assert float(np.max(mid) - np.min(mid)) == pytest.approx(r - math.sqrt(r * r - f * f / 4.0), rel=1e-4)


def test_minimum_chip_thickness_leaves_the_thin_layer():
    """A pass removing less than h_min leaves the surface; with recovery 1 it removes part between h_min and 2 h_min."""
    T = A.new_templates([np.array([[-1e-3, 0.0], [1e-3, 0.0]])])
    for depth, hmin, recovery, expect in ((1e-6, 2e-6, 0.0, 0.0), (3e-6, 2e-6, 0.0, -3e-6),
                                          (3e-6, 2e-6, 1.0, -1.5e-6), (5e-6, 2e-6, 1.0, -5e-6)):
        P = A.new_passes(2)
        P.kind[:] = A.KIND_CUT
        P.mode[:] = A.MODE_DIRECT
        P.tpl[:] = 0
        P.w[1] = -depth
        P.hmin[:] = hmin
        P.half[:] = 1.0
        G = A.new_grid(-5e-4, 1e-6, 1001, 10)
        K.envelope_profile(P, T, G, 2, recovery)
        assert G.z[500] == pytest.approx(expect, abs=1e-15)


def test_gaussian_mean_matches_reference_and_halves_the_cutoff_wavelength():
    du, lam = 1e-6, 2.5e-4
    u = np.arange(8000) * du
    z = np.sin(2 * math.pi * u / lam) + 0.3 * np.cos(2 * math.pi * u / 3e-5)
    out = np.empty(len(z))
    K.gaussian_mean(z, du, lam, 0.4697, 7, out)
    ref = R.gaussian_mean_ref(z, du, lam, 0.4697, 7)
    np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-12)
    # ISO 16610-21: 50 % transmission of the mean line at the cut-off wavelength
    K.gaussian_mean(np.sin(2 * math.pi * u / lam), du, lam, 0.4697, 1, out)
    centre = out[2000:6000]
    assert float(np.max(np.abs(centre))) == pytest.approx(0.5, abs=0.003)


def test_profile_params_match_reference_and_a_sine():
    du = 2.5e-7
    lsl = 3200
    u = np.arange(5 * lsl) * du
    A_ = 1e-6
    r = A_ * np.sin(2 * math.pi * u / 1e-4)
    out = np.zeros(A.N_P)
    K.profile_params(r, du, 0, lsl, 5, 0.1, 0.01, out)
    ref = R.profile_params_ref(r, du, 0, lsl, 5, 0.1, 0.01)
    lengths = [A.P_RA, A.P_RQ, A.P_RP, A.P_RV, A.P_RZ, A.P_RT, A.P_RSM]
    np.testing.assert_allclose(out[lengths], ref[lengths], rtol=1e-12, atol=1e-22)
    shapes = [A.P_RSK, A.P_RKU, A.P_RDQ, A.P_N_SL, A.P_N_ELEM]
    np.testing.assert_allclose(out[shapes], ref[shapes], rtol=1e-12, atol=1e-12)
    assert out[A.P_RA] == pytest.approx(2 * A_ / math.pi, rel=1e-4)
    assert out[A.P_RQ] == pytest.approx(A_ / math.sqrt(2), rel=1e-4)
    assert out[A.P_RZ] == pytest.approx(2 * A_, rel=1e-4)
    assert out[A.P_RSM] == pytest.approx(1e-4, rel=1e-3)
    assert out[A.P_RSK] == pytest.approx(0.0, abs=1e-6) and out[A.P_RKU] == pytest.approx(1.5, rel=1e-3)


def test_silhouette_matches_reference_and_a_sphere():
    R_ = 5e-3
    k = np.linspace(0.0, math.pi / 2, 91)
    r = np.ascontiguousarray(R_ * np.sin(k))
    z = np.ascontiguousarray(R_ * (1.0 - np.cos(k)))
    a = np.array([0.0, 0.0, 1.0])
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    tc, nn = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    out = np.full(801, np.inf)
    K.silhouette(r, z, a, e1, e2, tc, nn, 180, -6e-3, 1.5e-5, out)
    ref = R.silhouette_ref(r, z, a, e1, e2, tc, nn, 180, -6e-3, 1.5e-5, 801)
    ok = np.isfinite(ref)
    assert np.array_equal(np.isfinite(out), ok)
    np.testing.assert_allclose(out[ok], ref[ok], rtol=0.0, atol=1e-15)
    u = -6e-3 + np.arange(801) * 1.5e-5
    inner = np.abs(u) < 4e-3
    sphere = R_ - np.sqrt(R_ ** 2 - u[inner] ** 2)
    assert float(np.max(np.abs(out[inner] - sphere))) < 2e-6     # polygon of 91 x 180 samples


def test_wire_centre_matches_reference():
    pz = np.linspace(-1e-3, 1e-3, 401)
    pr = 4e-3 + np.abs(pz) * 1.7
    for zc in (0.0, 1e-4, -3e-4):
        for ext in (1, 0):
            got = K.wire_centre(pz, pr, zc, 4e-4, ext)
            assert got == pytest.approx(R.wire_centre_ref(pz, pr, zc, 4e-4, bool(ext)), abs=1e-15)
    assert math.isnan(K.wire_centre(pz, pr, 5e-3, 1e-4, 1))
