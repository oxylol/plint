"""Holding kernels: one compiled signature each, the Numba kernels against their plain-Python references, the
consistent tangent against finite differences, and the slip ratchets against closed-form solutions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.holding import abi as A
from cnctwin.holding import kernels as K
from cnctwin.holding import reference as REF


def _rect(centre, n, a1, h1, h2, k, force, mu, nodes=5):
    from cnctwin.holding.devices import simpson_weights
    n = np.asarray(n, float)
    a1 = np.asarray(a1, float)
    a2 = np.cross(n, a1)
    s = np.linspace(-1.0, 1.0, nodes)
    w = simpson_weights(nodes)
    pts, wts = [], []
    for i in range(nodes):
        for j in range(nodes):
            pts.append(np.asarray(centre, float) + s[i] * h1 * a1 + s[j] * h2 * a2)
            wts.append(w[i] * w[j])
    wts = np.asarray(wts)
    m = len(pts)
    return (np.asarray(pts), np.tile(n, (m, 1)), np.tile(a1, (m, 1)), np.tile(a2, (m, 1)), k * wts, force * wts,
            np.full(m, mu))


def vise_contacts(F=10000.0, mu=0.2, mu_b=0.19, weight=13.86, k=2e8, kt_ratio=0.82):
    """A 60 x 40 x 20 mm part between two jaws (y) on a base (z), O at its centre: the preloads of the clamped
    state (both jaws F, the base the weight)."""
    parts = [_rect([0, -0.02, 0.0], [0, 1, 0], [1, 0, 0], 0.03, 0.01, k, F, mu),
             _rect([0, 0.02, 0.0], [0, -1, 0], [1, 0, 0], 0.03, 0.01, k, F, mu),
             _rect([0, 0.0, -0.01], [0, 0, 1], [1, 0, 0], 0.03, 0.02, k, weight, mu_b)]
    r = np.vstack([p[0] for p in parts])
    n = np.vstack([p[1] for p in parts])
    t1 = np.vstack([p[2] for p in parts])
    t2 = np.vstack([p[3] for p in parts])
    kn = np.concatenate([p[4] for p in parts])
    pre_n = np.concatenate([p[5] for p in parts])
    mus = np.concatenate([p[6] for p in parts])
    nn = len(kn)
    pre = np.zeros((nn, 3))
    pre[:, 0] = pre_n
    cs = A.Contacts(r, n, t1, t2, kn, np.stack([kt_ratio * kn, kt_ratio * kn], axis=1), mus, pre, np.zeros(nn),
                    np.zeros(nn, np.int32), np.zeros((nn, 2)))
    return A.contiguous_contacts(cs)


def cfg(l_c=0.036, tol=1e-9, lam_tol=1e-4):
    return A.cfg_array(tol, 60, 1, 1e-3, 1e-2, 1e-4, l_c, 1000.0, lam_tol, 1e-2)


W0 = np.array([0.0, 0.0, -13.86, 0.0, 0.0, 0.0])


def test_every_kernel_has_one_signature():
    for kernel in K.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_solve6_matches_numpy():
    rng = np.random.default_rng(3)
    for _ in range(5):
        A6 = rng.normal(size=(6, 6)) + 6 * np.eye(6)
        b = rng.normal(size=6)
        x = np.zeros(6)
        assert K.solve6(A6, b, x) == 0
        np.testing.assert_allclose(x, np.linalg.solve(A6, b), rtol=1e-12, atol=1e-14)
        xr, singular = REF.solve6_ref(A6, b)
        assert not singular
        np.testing.assert_allclose(x, xr, rtol=1e-13, atol=1e-15)
    assert K.solve6(np.zeros((6, 6)), np.ones(6), np.zeros(6)) == 1


def _random_state(cs, rng, scale):
    delta = np.concatenate([rng.normal(scale=scale, size=3), rng.normal(scale=scale / 0.03, size=3)])
    return delta


@pytest.mark.parametrize("scale", [1e-7, 2e-6, 4e-5])
def test_contact_eval_matches_reference(scale):
    cs = vise_contacts()
    rng = np.random.default_rng(int(scale * 1e9))
    n = cs.r_m.shape[0]
    for trial in range(4):
        delta = _random_state(cs, rng, scale)
        release = rng.integers(0, 3, size=n).astype(np.int32) if trial == 3 else np.zeros(n, np.int32)
        f = np.zeros((n, 3))
        st = np.zeros(n, np.int32)
        R = np.zeros(6)
        J = np.zeros((6, 6))
        w = rng.normal(size=6)
        scale_k = K.contact_eval(cs, delta, w, 1e-4, 0.036, release, f, st, R, J)
        fr, sr, Rr, Jr, scale_r = REF.contact_eval_ref(cs, delta, w, 1e-4, 0.036, release)
        np.testing.assert_array_equal(st, sr)
        np.testing.assert_allclose(f, fr, rtol=1e-11, atol=1e-9)
        np.testing.assert_allclose(R, Rr, rtol=1e-11, atol=1e-8)
        np.testing.assert_allclose(J, Jr, rtol=1e-10, atol=1e-3)
        assert scale_k == pytest.approx(scale_r, rel=1e-12)


@pytest.mark.parametrize("scale", [1e-7, 5e-6])
def test_tangent_is_the_derivative_of_the_residual(scale):
    """Central differences of R in stick and in slip states (away from the kinks) equal J."""
    cs = vise_contacts()
    n = cs.r_m.shape[0]
    rng = np.random.default_rng(11)
    delta = _random_state(cs, rng, scale)
    rel = np.zeros(n, np.int32)
    f = np.zeros((n, 3))
    st = np.zeros(n, np.int32)
    R = np.zeros(6)
    J = np.zeros((6, 6))
    K.contact_eval(cs, delta, W0, 0.0, 0.036, rel, f, st, R, J)
    states = st.copy()
    num = np.zeros((6, 6))
    for a in range(6):
        h = (1e-11 if a < 3 else 3e-10) * max(1.0, scale / 1e-7)
        for sgn in (1.0, -1.0):
            d = delta.copy()
            d[a] += sgn * h
            Rp = np.zeros(6)
            K.contact_eval(cs, d, W0, 0.0, 0.036, rel, f, st, Rp, np.zeros((6, 6)))
            if not np.array_equal(st, states):
                pytest.skip("a contact changes state within the difference step")
            num[:, a] += sgn * Rp / (2.0 * h)
    np.testing.assert_allclose(J, num, rtol=2e-4, atol=2e-4 * np.abs(J).max())


def test_solve_equilibrium_matches_reference_and_balances():
    cs = vise_contacts()
    n = cs.r_m.shape[0]
    for w in (W0 + np.array([2000.0, 0, 0, 0, 0, 0]), W0 + np.array([500.0, 300.0, 800.0, 3.0, -2.0, 4.0])):
        delta = np.zeros(6)
        f = np.zeros((n, 3))
        st = np.zeros(n, np.int32)
        info = np.zeros(A.N_INFO)
        status = K.solve_equilibrium(cs, w, cfg(), delta, f, st, info)
        rs, rd, rf, rst, rinfo = REF.solve_equilibrium_ref(cs, w, cfg())
        assert status == rs == A.STATUS_HELD
        np.testing.assert_allclose(delta, rd, rtol=1e-8, atol=1e-15)
        np.testing.assert_array_equal(st, rst)
        # equilibrium: the contact forces balance the external wrench
        total = f.sum(axis=0)
        moment = np.cross(cs.r_m, f).sum(axis=0)
        np.testing.assert_allclose(total, -w[:3], atol=1e-6 * 2e4)
        np.testing.assert_allclose(moment, -w[3:], atol=1e-6 * 2e4 * 0.036)


def test_capacity_is_the_hand_calculation_and_matches_reference():
    """Sliding along the jaws through O: 2 mu F + mu_b W; lifting: 2 mu F + W."""
    cs = vise_contacts()
    n = cs.r_m.shape[0]
    for wv, hand in ((np.array([3000.0, 0, 0, 0, 0, 0]), 2 * 0.2 * 10000 + 0.19 * 13.86),
                     (np.array([0, 0, 3000.0, 0, 0, 0]), 2 * 0.2 * 10000 + 13.86)):
        delta = np.zeros(6)
        f = np.zeros((n, 3))
        st = np.zeros(n, np.int32)
        info = np.zeros(A.N_INFO)
        mech = np.zeros(6)
        mode = np.zeros(1)
        lam = K.capacity(cs, W0, wv, cfg(), delta, f, st, info, mech, mode)
        assert lam * 3000.0 == pytest.approx(hand, rel=2e-4)
        lam_r, _, _, _, mech_r, mode_r = REF.capacity_ref(cs, W0, wv, cfg())
        assert lam == pytest.approx(lam_r, rel=1e-12)
        assert int(mode[0]) == mode_r == A.MODE_FRICTION
        axis = int(np.argmax(np.abs(wv)))
        assert abs(mech[axis]) == pytest.approx(np.abs(mech[:3]).max())
        cos = abs(mech @ mech_r) / (np.linalg.norm(mech) * np.linalg.norm(mech_r))
        assert cos == pytest.approx(1.0, abs=1e-9)


def test_capacity_of_an_unclamped_part_is_a_separation():
    """A part resting on its base under gravity only lifts off at its weight (no friction involved)."""
    base = _rect([0, 0, -0.01], [0, 0, 1], [1, 0, 0], 0.03, 0.02, 2e8, 13.86, 0.19)
    m = len(base[4])
    pre = np.zeros((m, 3))
    pre[:, 0] = base[5]
    cs = A.contiguous_contacts(A.Contacts(base[0], base[1], base[2], base[3], base[4],
                                          np.stack([0.82 * base[4]] * 2, axis=1), base[6], pre, np.zeros(m),
                                          np.zeros(m, np.int32), np.zeros((m, 2))))
    out = REF.capacity_ref(cs, W0, np.array([0, 0, 10.0, 0, 0, 0]), cfg())
    assert out[0] * 10.0 == pytest.approx(13.86, rel=2e-4)
    assert out[5] == A.MODE_SEPARATION


def _wave(mean, amp, rate=0.0, phase=0.0, n=10, dims=1):
    w = A.empty_wave(n, dims)
    w.mean[:] = mean
    w.amp[:] = amp
    w.rate_rad_s[:] = rate
    w.phase0_rad[:] = phase + rate * 1e-3 * np.arange(n)
    return w


def test_ratchet_part_matches_reference():
    for mode in (A.MODE_FRICTION, A.MODE_SEPARATION):
        wave = _wave(80.0, 60.0, rate=2 * math.pi * 667.0)
        state = np.zeros(2)
        out = np.zeros(A.N_RP)
        K.ratchet_part(wave, 1e-3, 100.0, 1.5, mode, 32, 2048, state, out)
        ref_state, ref_out = REF.ratchet_part_ref(wave, 1e-3, 100.0, 1.5, mode, 32, 2048, np.zeros(2))
        np.testing.assert_allclose(state, ref_state, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(out, ref_out, rtol=1e-12, atol=1e-18)
        assert out[A.RP_Q_MAX] > 0.0


def test_ratchet_part_constant_excess_is_free_acceleration_then_coulomb_stop():
    """Q - Q_cap = 30 N on 1.5 kg for one tick, then 20 N below the capacity: q = a1 T^2 / 2 + v^2 / (2 a2)."""
    w = _wave(0.0, 0.0, n=3)
    w.mean[0, 0], w.mean[1:, 0] = 130.0, 80.0
    state = np.zeros(2)
    out = np.zeros(A.N_RP)
    K.ratchet_part(w, 1e-3, 100.0, 1.5, A.MODE_FRICTION, 32, 2048, state, out)
    a1, a2, T = 30.0 / 1.5, 20.0 / 1.5, 1e-3
    v = a1 * T
    assert out[A.RP_DQ] == pytest.approx(0.5 * a1 * T * T + v * v / (2 * a2), rel=1e-12)
    assert state[1] == 0.0
    assert out[A.RP_WORK_J] == pytest.approx(100.0 * out[A.RP_DQ], rel=1e-12)
    # separation: lifts off and comes back to the seat (no persistent displacement)
    state = np.zeros(2)
    w2 = _wave(0.0, 0.0, n=4)
    w2.mean[0, 0], w2.mean[1:, 0] = 130.0, 60.0
    K.ratchet_part(w2, 1e-3, 100.0, 1.5, A.MODE_SEPARATION, 32, 2048, state, out)
    assert out[A.RP_Q_MAX] == pytest.approx(0.5 * a1 * T * T + v * v / (2 * 40.0 / 1.5), rel=1e-9)
    assert state[0] == 0.0 and out[A.RP_DQ] == 0.0


def test_ratchet_tool_matches_reference():
    wave = _wave(0.0, 0.0, rate=4 * 1047.2, dims=2)
    wave.mean[:, 0], wave.mean[:, 1] = 9000.0, 8000.0
    wave.amp[:, 0], wave.amp[:, 1] = 5000.0, 6000.0
    for stop, limit in ((1, 1e9), (1, 2e-4)):
        state = np.zeros(4)
        out = np.zeros(A.N_RT)
        K.ratchet_tool(wave, 1e-3, 17000.0, 17000.0, 0.025, 0.05, stop, limit, 32, 2048, state, out)
        ref_state, ref_out = REF.ratchet_tool_ref(wave, 1e-3, 17000.0, 17000.0, 0.025, 0.05, stop, limit, 32, 2048,
                                                  np.zeros(4))
        np.testing.assert_allclose(state, ref_state, rtol=1e-11, atol=1e-15)
        np.testing.assert_allclose(out, ref_out, rtol=1e-11, atol=1e-15)
        assert out[A.RT_DX2] > 0.0 and state[1] <= limit + 1e-15


def test_ratchet_tool_walk_rate_equal_masses_is_straight_free_acceleration():
    """Equal equivalent masses, constant demand |D| > C: the slip runs straight along D with a = (|D| - C) / m,
    the walk is x2 = a T^2 / 2 * D2 / |D| over one tick; the friction work is C times the path."""
    D1, D2, C, m, T = 9000.0, 12000.0, 14000.0, 0.05, 1e-3
    wave = _wave(0.0, 0.0, n=1, dims=2)
    wave.mean[0, 0], wave.mean[0, 1] = D1, D2
    state = np.zeros(4)
    out = np.zeros(A.N_RT)
    K.ratchet_tool(wave, T, C, C, m, m, 1, 1e9, 32, 2048, state, out)
    Dm = math.hypot(D1, D2)
    path = 0.5 * (Dm - C) / m * T * T
    assert out[A.RT_DX2] == pytest.approx(path * D2 / Dm, rel=1e-9)
    assert out[A.RT_DX1] == pytest.approx(path * D1 / Dm, rel=1e-9)
    assert out[A.RT_WORK_J] == pytest.approx(C * path, rel=1e-9)
    assert out[A.RT_PEAK_RATIO] == pytest.approx(Dm / C, rel=1e-12)


def test_ratchet_tool_reduces_to_the_part_ratchet_in_one_dimension():
    """Pure axial demand: the 2-DOF implicit-friction ratchet and the 1-DOF ratchet give the same walk."""
    for per_tooth in (32, 1024):
        wt = _wave(0.0, 0.0, rate=4 * 1047.2, dims=2)
        wt.mean[:, 1], wt.amp[:, 1] = 9000.0, 9000.0
        state = np.zeros(4)
        out = np.zeros(A.N_RT)
        K.ratchet_tool(wt, 1e-3, 15000.0, 15000.0, 0.05, 0.05, 0, 1e9, per_tooth, 10 ** 6, state, out)
        wp = _wave(9000.0, 9000.0, rate=4 * 1047.2)
        sp = np.zeros(2)
        op = np.zeros(A.N_RP)
        K.ratchet_part(wp, 1e-3, 15000.0, 0.05, A.MODE_FRICTION, per_tooth, 10 ** 6, sp, op)
        assert out[A.RT_DX2] == pytest.approx(op[A.RP_DQ], rel=1e-12)
        assert out[A.RT_DX1] == 0.0


def test_ratchet_tool_converges_with_sub_steps_and_back_stop_holds():
    wave = _wave(0.0, 0.0, rate=4 * 1047.2, dims=2)
    wave.mean[:, 0], wave.mean[:, 1] = 6000.0, 5000.0
    wave.amp[:, 0], wave.amp[:, 1] = 7000.0, 7000.0
    for m1 in (0.05, 0.025):              # equal and unequal equivalent masses
        results = []
        for per_tooth in (256, 8192):
            state = np.zeros(4)
            out = np.zeros(A.N_RT)
            K.ratchet_tool(wave, 1e-3, 15000.0, 15000.0, m1, 0.05, 1, 1e9, per_tooth, 10 ** 6, state, out)
            results.append(out.copy())
        assert results[0][A.RT_DX2] == pytest.approx(results[1][A.RT_DX2], rel=2e-3)
        assert results[0][A.RT_DX1] == pytest.approx(results[1][A.RT_DX1], rel=2e-3)
    # a push into the holder (negative axial demand) is taken by the back stop: no negative pull-out
    wave.mean[:, 1] = -9000.0
    wave.amp[:, 1] = 0.0
    state = np.zeros(4)
    out = np.zeros(A.N_RT)
    K.ratchet_tool(wave, 1e-3, 15000.0, 15000.0, 0.025, 0.05, 1, 1e9, 64, 2048, state, out)
    assert state[1] == 0.0


def test_implicit_slip_solves_its_equation():
    rng = np.random.default_rng(5)
    for _ in range(20):
        p = rng.normal(size=2) * 3.0
        m = rng.uniform(0.01, 0.2, size=2)
        impulse = float(np.linalg.norm(p)) * rng.uniform(0.05, 0.98)
        lam = K.implicit_slip(p[0], p[1], m[0], m[1], impulse)
        v = p / (m + lam)
        assert lam * np.linalg.norm(v) == pytest.approx(impulse, rel=1e-12)
        assert lam == pytest.approx(REF.implicit_slip_ref(p, m, impulse), rel=1e-9)
    # equal masses: v = (p - impulse p / |p|) / m exactly
    lam = K.implicit_slip(2.25, 3.0, 0.05, 0.05, 3.5)
    np.testing.assert_allclose(np.array([2.25, 3.0]) / (0.05 + lam), np.array([3.0, 4.0]), rtol=1e-14)


def test_sub_steps_rules():
    assert K.sub_steps(0.0, 1e-3, 32, 2048) == 4
    assert K.sub_steps(2 * math.pi * 1000.0, 1e-3, 32, 2048) == 32
    assert K.sub_steps(2 * math.pi * 1e6, 1e-3, 32, 2048) == 2048
    assert REF.sub_steps_ref(2 * math.pi * 1000.0, 1e-3, 32, 2048) == 32
