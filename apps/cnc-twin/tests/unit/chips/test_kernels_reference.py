"""Every chips kernel against its plain-Python reference (architecture 5.5), and one signature each."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.chips import abi, kernels, params, reference
from cnctwin.chips.schema import settings as chips_settings
from cnctwin.core import rng

import chipsfix as fx


def _clone(nt):
    return type(nt)(*(np.array(a, copy=True) for a in nt))


def _model():
    return params.chip_model(fx.project().material, chips_settings(None), params.load())


def _pile(nu=24, nv=20, cell=2e-3, seed=3):
    H = abi.pile(nu, nv, (0.0, 0.0, 0.0), np.eye(3), cell)
    r = np.random.default_rng(seed)
    H.base_m[:] = 0.03
    H.base_m[8:14, 6:12] = 0.02          # a pocket
    H.base_m[0, :] = -np.inf             # an edge with nothing below
    H.base_kind[:] = abi.SURF_PART
    H.base_kind[0, :] = abi.SURF_NONE
    H.base_kind[-3:, :] = abi.SURF_TABLE
    mass = r.uniform(0, 2e-6, (nu, nv)) * (r.uniform(size=(nu, nv)) > 0.5)
    mass[0, :] = 0.0
    H.mass_kg[:] = mass
    H.bulk_m3[:] = mass * 15.0 / 2700.0
    H.enthalpy_J[:] = mass * 1.5e5
    H.size_mass[:] = mass * 1.2e-3
    H.thick_mass[:] = mass * 1.5e-4
    H.width_mass[:] = mass * 2e-3
    H.length_mass[:] = mass * 4e-3
    H.repose_mass[:] = mass * math.tan(math.radians(40))
    H.long_kg[:] = mass * 0.3
    H.fines_kg[:] = mass * 0.1
    H.wet[:, 10:] = 1
    H.flush_Pa[:, 10:] = r.uniform(0, 400, (nu, nv - 10))
    H.film_uv_m_s[:, 10:, 0] = 0.3
    H.surf_temp_K[:] = 300.0
    return H


def _window(n_t=10, dt=1e-3, moving=True):
    W = abi.window(n_t)
    W.t0_s[0] = 0.0
    W.t_end_s[0] = n_t * dt
    W.dt_tick_s[0] = dt
    for k in range(n_t + 1):
        W.grid_origin_M[k] = (1e-4 * k if moving else 0.0, 0.0, 0.0)
    return W


def _scene(n_t=10, liquid=True):
    S = abi.scene(2, n_t, 1, 2)
    S.box_lo_M[:] = (-0.2, -0.2, -0.1)
    S.box_hi_M[:] = (0.3, 0.3, 0.4)
    S.floor[:] = (2, 0)
    for k in range(n_t + 1):
        S.cyl_base_M[0, k] = (0.024 + 2e-4 * k, 0.02, 0.025)
        S.cyl_base_M[1, k] = (0.024 + 2e-4 * k, 0.02, 0.047)
    S.cyl_axis_M[:, :, 2] = 1.0
    S.cyl_omega[:] = -1047.0
    S.cyl_radius[:] = (5e-3, 8e-3)
    S.cyl_length[:] = (22e-3, 60e-3)
    S.cyl_kind[:] = (abi.CYL_PADDLE, abi.CYL_SMOOTH)
    S.cyl_surf[:] = (abi.SURF_TOOL, abi.SURF_HOLDER)
    S.cyl_owner[:] = (0, 0)
    S.cyl_wrap[:] = (1, 1)
    S.cyl_weld[:] = (1, 0)
    S.sph_center_M[0] = (0.06, 0.02, 0.09)
    S.sph_radius[0] = 2e-3
    S.sph_active[0] = 0
    S.jet_origin_M[0] = (0.06, 0.02, 0.09)
    S.jet_dir_M[0] = (-0.5, 0.0, -0.8660254037844386)
    S.jet_speed[0] = 12.0
    S.jet_r0[0] = 1.5e-3
    S.jet_tan[0] = math.tan(math.radians(5))
    S.jet_gas[0] = 0 if liquid else 1
    S.jet_fluid[0] = (992.0, 1.2e-3, 0.55, 8.8) if liquid else (1.204, 1.81e-5, 0.0257, 0.71)
    S.jet_origin_M[1] = (-0.05, 0.02, 0.06)
    S.jet_dir_M[1] = (1.0, 0.0, 0.0)
    S.jet_speed[1] = 80.0
    S.jet_r0[1] = 1e-3
    S.jet_gas[1] = 1
    S.jet_fluid[1] = (1.204, 1.81e-5, 0.0257, 0.71)
    S.air[:] = (1.204, 1.81e-5, 0.0257, 0.71, 293.15)
    S.liquid[:] = (992.0, 1.2e-3, 0.55, 8.8, 300.0, 5e-5, 0.035)
    S.jet_const[:] = (5.8, 0.094)
    return S


def _particles(n=24, seed=11):
    P = abi.particles(n + 4)
    r = np.random.default_rng(seed)
    pr = params.load()
    t = r.uniform(5e-5, 3e-4, n)
    w = r.uniform(1e-3, 3e-3, n)
    length = r.uniform(2e-3, 60e-3, n)
    m_chip, dv, area, surface, psi = params.chip_shape(t, w, length, 2700.0)
    a, b, c, d = params.hl_coefficients(pr.aero, psi)
    P.pos_M_m[:n, 0] = r.uniform(0.0, 0.045, n)
    P.pos_M_m[:n, 1] = r.uniform(0.0, 0.04, n)
    P.pos_M_m[:n, 2] = r.uniform(0.032, 0.06, n)
    P.vel_M_m_s[:n] = r.normal(0, 4.0, (n, 3))
    P.time_s[:n] = r.uniform(0, 5e-3, n)
    P.mass_kg[:n] = r.uniform(1e-7, 1e-6, n)
    P.temp_K[:n] = r.uniform(320, 700, n)
    P.thickness_m[:n] = t
    P.width_m[:n] = w
    P.length_m[:n] = length
    P.dv_m[:n] = dv
    P.chip_mass_kg[:n] = m_chip
    P.area_m2[:n] = area
    P.surface_m2[:n] = surface
    P.hl_a[:n], P.hl_b[:n], P.hl_c[:n], P.hl_d[:n] = a, b, c, d
    P.form[:n] = r.choice([11, 21, 42, 62, 70], n)
    P.status[:n] = abi.FLYING
    P.status[3] = abi.ON_TOOL
    P.status[5] = abi.FREE
    # aimed particles: weld (hot, at the flutes), wrap (long ribbon), paddle bounce, nozzle hook, pan, inside the
    # pile, enclosure wall
    aims = {0: ((0.024, 0.032, 0.035), (0.0, -6.0, 0.0), 700.0, 62, 4e-3),
            1: ((0.024, 0.032, 0.036), (0.0, -6.0, 0.0), 320.0, 11, 60e-3),
            2: ((0.012, 0.02, 0.040), (6.0, 0.0, 0.0), 320.0, 62, 4e-3),
            4: ((0.06, 0.02, 0.080), (0.0, 0.0, 5.0), 320.0, 62, 12e-3),
            6: ((0.25, 0.25, -0.090), (0.0, 0.0, -5.0), 320.0, 70, 2e-3),
            7: ((0.019, 0.015, 0.0201), (0.0, 0.0, 0.0), 320.0, 70, 2e-3),
            8: ((0.29, 0.10, 0.10), (8.0, 0.0, 0.0), 320.0, 70, 2e-3)}
    for i, (pos, vel, temp, form, length_i) in aims.items():
        P.pos_M_m[i] = pos
        P.vel_M_m_s[i] = vel
        P.temp_K[i] = temp
        P.form[i] = form
        P.time_s[i] = 0.0
        P.length_m[i] = length_i
        m_i, dv_i, a_i, s_i, psi_i = params.chip_shape(t[i], w[i], length_i, 2700.0)
        P.chip_mass_kg[i], P.dv_m[i], P.area_m2[i], P.surface_m2[i] = m_i, dv_i, a_i, s_i
        ha, hb, hc, hd = params.hl_coefficients(pr.aero, psi_i)
        P.hl_a[i], P.hl_b[i], P.hl_c[i], P.hl_d[i] = ha, hb, hc, hd
        P.bounces[i] = 3                       # chips coming back to the tool, not leaving its flute
    P.slot[:n] = 0
    P.pid[:n] = np.arange(n)
    P.n[0] = n
    return P


def _compare(a, b, rtol=1e-9, atol=1e-15):
    for name, x, y in zip(type(a)._fields, a, b):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=name)


def test_every_kernel_has_one_signature():
    for kernel in kernels.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


@pytest.mark.parametrize("liquid", [True, False])
def test_advance_particles_matches_reference(liquid):
    C = _model()
    P, H, W, S = _particles(), _pile(), _window(), _scene(liquid=liquid)
    out_k, ev_k = abi.step_out(64)
    P2, H2, W2, S2 = _clone(P), _clone(H), _clone(W), _clone(S)
    out_r, ev_r = abi.step_out(64)
    kernels.advance_particles(P, H, W, S, C, out_k, ev_k)
    reference.advance_particles(P2, H2, W2, S2, C, out_r, ev_r)
    _compare(P, P2, rtol=1e-8, atol=1e-12)
    _compare(H, H2, rtol=1e-8, atol=1e-15)
    np.testing.assert_allclose(out_k, out_r, rtol=1e-8, atol=1e-14)
    assert ev_k.n[0] == ev_r.n[0]
    np.testing.assert_allclose(ev_k.rec, ev_r.rec, rtol=1e-8, atol=1e-12)
    assert out_k[abi.OUT_HITS] > 0 and out_k[abi.OUT_SUBSTEPS] > 0
    kinds = set(ev_k.rec[:ev_k.n[0], 0].astype(int).tolist())
    assert {abi.EV_WELD, abi.EV_WRAP, abi.EV_NOZZLE} <= kinds
    assert P.status[0] == abi.ON_TOOL and P.status[1] == abi.WRAPPED and P.status[4] == abi.ON_NOZZLE
    assert P.status[6] == abi.FREE and out_k[abi.OUT_PAN_MASS_KG] > 0       # reached the chip pan
    assert P.status[7] == abi.FREE and out_k[abi.OUT_DEPOSIT_KG] > 0        # settled in the pocket


def test_relax_pile_matches_reference():
    H = _pile()
    H.bulk_m3[12, 10] += 2e-7          # a heap that must avalanche
    H.mass_kg[12, 10] += 2e-7 * 2700 / 15
    H.repose_mass[12, 10] += 2e-7 * 2700 / 15 * math.tan(math.radians(40))
    H2 = _clone(H)
    out_k = np.zeros(abi.N_OUT)
    out_r = np.zeros(abi.N_OUT)
    kernels.relax_pile(H, 30, 1e-15, out_k)
    reference.relax_pile(H2, 30, 1e-15, out_r)
    _compare(H, H2, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(out_k, out_r, rtol=1e-10, atol=1e-20)


def test_fill_occupancy_matches_reference():
    H = _pile()
    P = _particles()
    P.vel_M_m_s[:10] *= 1e-3
    R = np.eye(3)
    o = np.array([1e-3, 0.0, 0.0])
    occ = abi.occupancy(30, 26, 20, (-2e-3, -2e-3, 0.018), 2e-3)
    occ2 = _clone(occ)
    kernels.fill_occupancy(H, P, R, o, 0.5, occ)
    reference.fill_occupancy(H, P, R, o, 0.5, occ2)
    _compare(occ, occ2, rtol=1e-10, atol=1e-18)
    assert occ.mass_kg.sum() > 0


@pytest.mark.parametrize("capillary", [0.0, 0.05])
def test_flush_pile_matches_reference(capillary):
    H = _pile()
    H2 = _clone(H)
    fp = np.array([0.006, 0.0495, 3.97, 1.5, capillary, 0.035, 0.6, 2700.0])
    R = np.eye(3)
    rec, n = np.zeros((2000, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    rec2, n2 = np.zeros((2000, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    kernels.flush_pile(H, fp, R, 992.0, 9.80665, 0.01, rec, n)
    reference.flush_pile(H2, fp, R, 992.0, 9.80665, 0.01, rec2, n2)
    _compare(H, H2, rtol=1e-10, atol=1e-18)
    assert n[0] == n2[0] and n[0] > 0
    np.testing.assert_allclose(rec[:n[0]], rec2[:n2[0]], rtol=1e-10, atol=1e-18)


def test_pile_heat_matches_reference():
    C = _model()
    H = _pile()
    H2 = _clone(H)
    hp = np.array([200.0, 10.0, 2000.0, 293.15, 300.0])
    q1, q2 = np.zeros(H.base_m.shape), np.zeros(H.base_m.shape)
    o1, o2 = np.zeros(2), np.zeros(2)
    kernels.pile_heat(H, C, hp, 0.01, q1, o1)
    reference.pile_heat(H2, C, hp, 0.01, q2, o2)
    _compare(H, H2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(q1, q2, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(o1, o2, rtol=1e-10, atol=1e-14)


def test_sweep_cylinders_matches_reference():
    C = _model()
    H, W, S = _pile(), _window(), _scene()
    S.cyl_base_M[0, :, 2] = 0.021          # tool tip inside the pile layer
    H2, W2, S2 = _clone(H), _clone(W), _clone(S)
    rec, n = np.zeros((4000, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    rec2, n2 = np.zeros((4000, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    kernels.sweep_cylinders(H, W, S, C, rec, n)
    reference.sweep_cylinders(H2, W2, S2, C, rec2, n2)
    _compare(H, H2, rtol=1e-10, atol=1e-18)
    assert n[0] == n2[0] and n[0] > 0
    np.testing.assert_allclose(rec[:n[0]], rec2[:n2[0]], rtol=1e-10, atol=1e-15)


def test_recut_take_matches_reference():
    H = _pile()
    H2 = _clone(H)
    req = np.array([[10, 8, 3e-6, 1.0, 2.0, 0.0, 0, 0.004, 0.021, 0.017, 0.025],
                    [3, 3, 5e-7, 0.0, 0.0, 1.0, 0, 0.006, 0.007, 0.007, 0.035],
                    [50, 50, 1e-6, 0.0, 0.0, 0.0, 0, 0.007, 0.0, 0.0, 0.0]])
    rec, n = np.zeros((100, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    rec2, n2 = np.zeros((100, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    missing = kernels.recut_take(H, req, rec, n)
    missing2 = reference.recut_take(H2, req, rec2, n2)
    assert math.isclose(missing, missing2, rel_tol=1e-12, abs_tol=1e-20)
    assert missing >= 1e-6 * (1 - 1e-12)      # the third request is outside the grid
    _compare(H, H2, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(rec[:n[0]], rec2[:n2[0]], rtol=1e-12, atol=1e-20)


def test_drop_particles_matches_reference():
    C = _model()
    P, H, W = _particles(), _pile(), _window()
    P2, H2, W2 = _clone(P), _clone(H), _clone(W)
    o1, o2 = np.zeros(abi.N_OUT), np.zeros(abi.N_OUT)
    kernels.drop_particles(P, H, W, C, o1)
    reference.drop_particles(P2, H2, W2, C, o2)
    _compare(P, P2, rtol=1e-12, atol=1e-20)
    _compare(H, H2, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-20)


def test_hash_draws_match_reference_and_core():
    keys = np.arange(-3, 40, dtype=np.int64)
    u1, n1 = np.zeros((len(keys), 3)), np.zeros((len(keys), 2))
    u2, n2 = np.zeros_like(u1), np.zeros_like(n1)
    kernels.hash_draws(99, rng.stream_key("chips.scatter"), keys, 5, 7, u1, n1)
    reference.hash_draws(99, rng.stream_key("chips.scatter"), keys, 5, 7, u2, n2)
    np.testing.assert_array_equal(u1, u2)
    np.testing.assert_allclose(n1, n2, rtol=1e-14, atol=1e-15)
    assert u1[4, 1] == rng.hash_uniform(99, rng.stream_key("chips.scatter"), int(keys[4]), 5, 8)


def test_write_emission_matches_reference():
    rec1, n1 = np.zeros((2, abi.EMISSION_WIDTH)), np.zeros(2, np.int64)
    rec2, n2 = np.zeros((2, abi.EMISSION_WIDTH)), np.zeros(2, np.int64)
    values = (123, 1, 2, 3, 1e-7, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 500.0, 1e-4, 2e-3, 5e-3, 40.0, 0.1, 0.5, 1, 4)
    for _ in range(3):
        kernels.write_emission(rec1, n1, *values)
        reference.write_emission(rec2, n2, *values)
    np.testing.assert_array_equal(rec1, rec2)
    np.testing.assert_array_equal(n1, n2)
    assert n1[1] == 1 and n1[0] == 2


def test_small_numerics_match_reference():
    C = _model()
    for z in (0.0, 1e-9, 5e-4, 2e-3, 0.7, 30.0):
        assert np.allclose(kernels.phi12(z), reference.phi12(z), rtol=1e-14, atol=0)
    for temp in (200.0, 293.15, 350.0, 480.0, 900.0):
        h = kernels.enthalpy(temp, C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
        assert math.isclose(h, reference.enthalpy(temp, C.cp_T_K, C.cp_J_kgK, C.h_J_kg), rel_tol=1e-13, abs_tol=1e-9)
        assert math.isclose(kernels.temperature(h, C.cp_T_K, C.cp_J_kgK, C.h_J_kg), temp, rel_tol=1e-12)
    assert abs(kernels.enthalpy(293.15, C.cp_T_K, C.cp_J_kgK, C.h_J_kg)) < 1e-9
    v = np.array([3.0, -2.0, -5.0])
    s = np.array([0.5, 0.0, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    assert np.allclose(kernels.impulse(*v, *s, *n, 0.4, 0.3), reference.impulse(v, s, n, 0.4, 0.3), rtol=1e-14)


def test_film_field_matches_reference():
    r = np.random.default_rng(4)
    q = np.ascontiguousarray(r.uniform(0, 500, (17, 13)) * (r.uniform(size=(17, 13)) > 0.2))
    f1 = np.zeros((17, 13, 2))
    f2 = np.zeros((17, 13, 2))
    kernels.film_field(q, 2e-3, 992.0, f1)
    reference.film_field(q, 2e-3, 992.0, f2)
    np.testing.assert_allclose(f1, f2, rtol=1e-12, atol=1e-14)
    assert np.any(f1 != 0)


def test_full_mobilise_buffer_keeps_the_chips_in_place():
    H = _pile()
    m0 = float(H.mass_kg.sum())
    fp = np.array([0.006, 0.0495, 3.97, 1.5, 0.0, 0.035, 0.6, 2700.0])
    rec, n = np.zeros((2, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    kernels.flush_pile(H, fp, np.eye(3), 992.0, 9.80665, 0.01, rec, n)
    assert n[0] == 2 and n[1] == 1
    moved = float(rec[:2, abi.MOB_COLUMNS.index("mass_kg")].sum())
    assert math.isclose(float(H.mass_kg.sum()) + moved, m0, rel_tol=1e-13)
