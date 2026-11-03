"""The analytic models of the calibration against closed forms and fine numerical references."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from cnctwin.mdata import surrogates as S


def test_sin_power_integral_against_quadrature():
    for p in (1.0, 0.75, 0.6):
        for phi in (0.3, 1.2, math.pi / 2, 2.5, math.pi):
            reference, _ = quad(lambda x: math.sin(x) ** p, 0.0, phi, limit=200)
            assert S.sin_power_integral(p, phi) == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_kienzle_power_reduces_to_specific_energy_times_mrr():
    # mc = 0: kc constant, P = kc ae ap vf for any immersion (Altintas 2012 ch. 2)
    for ae in (0.5e-3, 2e-3, 5e-3, 10e-3):
        n, z, R, ap, fz = 10000 / 60, 4, 5e-3, 2e-3, 0.05e-3
        power = S.milling_power_kienzle(spindle_rev_s=n, teeth=z, radius_m=R, ap_m=ap, fz_m=fz, ae_m=ae,
                                        kc11_Pa=800e6, mc=0.0, h_ref_m=1e-3)
        assert power == pytest.approx(800e6 * ae * ap * z * fz * n, rel=1e-12)
        mech = S.milling_power_mechanistic(spindle_rev_s=n, teeth=z, radius_m=R, ap_m=ap, fz_m=fz, ae_m=ae,
                                           Ktc_Pa=800e6, Kte_N_per_m=0.0)
        assert mech == pytest.approx(power, rel=1e-12)


def test_kienzle_power_against_tooth_by_tooth_integration():
    n, z, R, ap, fz, ae, kc11, mc = 8000 / 60, 3, 6e-3, 3e-3, 0.08e-3, 4e-3, 1500e6, 0.25
    phi_e = math.acos(1 - ae / R)
    phi = np.linspace(0.0, phi_e, 200001)
    h = fz * np.sin(phi)
    force = kc11 * (1e-3) ** mc * np.power(np.maximum(h, 1e-300), 1 - mc) * ap + 20e3 * ap
    reference = z * n * R * np.trapezoid(force, phi)
    power = S.milling_power_kienzle(spindle_rev_s=n, teeth=z, radius_m=R, ap_m=ap, fz_m=fz, ae_m=ae, kc11_Pa=kc11,
                                    mc=mc, h_ref_m=1e-3, edge_N_per_m=20e3)
    assert power == pytest.approx(reference, rel=1e-6)


def test_following_error_exact_and_steady_state():
    t = np.linspace(0.0, 0.5, 501)
    v = np.where(t < 0.1, t / 0.1 * 0.05, 0.05)                      # ramp to 50 mm/s, then constant
    e = S.following_error(t, v, kv_1_s=40.0, feedforward=0.6)
    assert e[-1] == pytest.approx(0.4 * 0.05 / 40.0, rel=1e-6)
    fine_t = np.linspace(0.0, 0.5, 500001)
    fine_v = np.interp(fine_t, t, v)
    ref = np.zeros_like(fine_t)
    dt = fine_t[1] - fine_t[0]
    for k in range(1, len(fine_t)):                                   # trapezoidal reference, tiny step
        f0 = 0.4 * fine_v[k - 1] - 40.0 * ref[k - 1]
        pred = ref[k - 1] + dt * f0
        f1 = 0.4 * fine_v[k] - 40.0 * pred
        ref[k] = ref[k - 1] + 0.5 * dt * (f0 + f1)
    assert np.max(np.abs(e - np.interp(t, fine_t, ref))) < 1e-9


def test_first_order_filter_is_exact_for_ramps():
    t = np.array([0.0, 0.3, 1.0, 2.5])
    u = np.array([0.0, 3.0, 3.0, -1.0])
    y = S.first_order_filter(t, u, 0.5)
    fine = np.linspace(0, 2.5, 250001)
    uf = np.interp(fine, t, u)
    yf = np.zeros_like(fine)
    dt = fine[1] - fine[0]
    for k in range(1, len(fine)):
        yf[k] = yf[k - 1] + dt * ((0.5 * (uf[k] + uf[k - 1])) - yf[k - 1]) / 0.5
    assert np.max(np.abs(y - np.interp(t, fine, yf))) < 1e-4
    assert S.first_order_filter(t, u, 0.0).tolist() == u.tolist()


def test_block_queue_starvation_and_lookahead():
    start, end = S.block_queue(np.full(10, 0.2e-3), block_s=1e-3, lookahead=80)
    assert np.diff(start) == pytest.approx(np.full(9, 1e-3))          # short blocks: one per block time
    start, end = S.block_queue(np.full(10, 5e-3), block_s=1e-3, lookahead=80)
    assert np.diff(start) == pytest.approx(np.full(9, 5e-3))          # long blocks: motion-limited
    motion = np.array([0.1, 1e-4, 1e-4, 1e-4, 0.1])
    start, _ = S.block_queue(motion, block_s=1e-3, lookahead=80)
    assert start[4] - start[0] == pytest.approx(0.1 + 3e-4)          # the buffer prepared the short ones ahead
    start, _ = S.block_queue(motion, block_s=1e-3, lookahead=1)
    assert start[4] - start[0] > 0.1 + 2.5e-3                         # no look-ahead: each waits for the interpreter


def test_spindle_growth_step_and_taylor_usui():
    t = np.linspace(0.0, 7200.0, 721)
    omega = np.where(t < 3600.0, 1000.0, 0.0)
    g = S.spindle_growth(t, omega, growth_max_m=25e-6, tau_s=1500.0, omega_max_rad_s=1000.0, exponent=5 / 3)
    k = np.searchsorted(t, 1500.0)
    assert g[k] == pytest.approx(25e-6 * (1 - math.exp(-1)), rel=1e-9)
    assert g[-1] == pytest.approx(g[360] * math.exp(-3600.0 / 1500.0), rel=1e-9)
    life = S.taylor_life(np.array([200 / 60, 300 / 60]), C_m_s=400 / 60, n=0.25, T_ref_s=60.0)
    assert life == pytest.approx(60.0 * (np.array([2.0, 4 / 3]) ** 4))
    vb = S.usui_flank_wear(np.array([0.0, 60.0, 120.0]), sigma_n_Pa=np.full(3, 1e9), sliding_m_s=np.full(3, 2.0),
                           temp_K=np.full(3, 900.0), A_per_Pa=1e-12, B_K=3000.0, clearance_rad=math.radians(7),
                           rake_rad=math.radians(6), VB0_m=1e-5)
    rate = 1e-12 * 1e9 * 2.0 * math.exp(-3000.0 / 900.0) * (1 / math.tan(math.radians(7)) - math.tan(math.radians(6)))
    assert vb.tolist() == pytest.approx([1e-5, 1e-5 + 60 * rate, 1e-5 + 120 * rate])


def test_axis_force_and_part_dimension():
    force = S.axis_force(np.array([-0.1, 0.0, 0.1]), np.array([0.0, 2.0, 0.0]), mass_kg=300.0, lift_frac=0.0,
                         friction_coeff=0.005, normal_load_N=300 * 9.80665, seal_drag_N=40.0)
    friction = 0.005 * 300 * 9.80665 + 40.0
    assert force.tolist() == pytest.approx([-friction, 600.0, friction])
    vertical = S.axis_force(np.array([0.0]), np.array([0.0]), mass_kg=400.0, lift_frac=1.0, friction_coeff=0.005,
                            normal_load_N=0.0, seal_drag_N=40.0)
    assert vertical[0] == pytest.approx(400.0 * 9.80665)
    d = S.part_dimension(np.array([10e-3, 8e-3]), runout_sensitivity=np.array([2.0, -1.0]),
                         deflection_m=np.array([0.0, -20e-6]), runout_m=7e-6, runout_sim_m=5e-6, stiffness_factor=0.8)
    assert d.tolist() == pytest.approx([10e-3 + 4e-6, 8e-3 - 2e-6 - 20e-6 * 0.25])
