"""Flight of a chip: ballistic trajectory, drag terminal velocity, lumped cooling against analytic solutions."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from cnctwin.chips import abi, kernels, params
from cnctwin.chips.schema import settings as chips_settings
from cnctwin.units import STANDARD_GRAVITY_M_S2

import chipsfix as fx

G = np.array([0.0, 0.0, -STANDARD_GRAVITY_M_S2])


def _model(cp_constant: float | None = None, emissivity: float | None = None):
    C = params.chip_model(fx.project().material, chips_settings(None), params.load())
    if cp_constant is not None:
        cp_t = np.array([293.15])
        cp_v = np.array([cp_constant])
        C = C._replace(cp_T_K=cp_t, cp_J_kgK=cp_v, h_J_kg=np.ascontiguousarray(params.enthalpy_table(cp_t, cp_v)))
    if emissivity is not None:
        C.emissivity[0] = emissivity
    return C


def _one_particle(pos, vel, t, w, length, temp=293.15):
    P = abi.particles(1)
    m, dv, area, surface, psi = params.chip_shape(t, w, length, 2700.0)
    a, b, c, d = params.hl_coefficients(params.load().aero, psi)
    P.pos_M_m[0] = pos
    P.vel_M_m_s[0] = vel
    P.mass_kg[0] = float(m) * 10.0
    P.temp_K[0] = temp
    P.thickness_m[0], P.width_m[0], P.length_m[0] = t, w, length
    P.dv_m[0], P.chip_mass_kg[0], P.area_m2[0], P.surface_m2[0] = dv, m, area, surface
    P.hl_a[0], P.hl_b[0], P.hl_c[0], P.hl_d[0] = a, b, c, d
    P.form[0] = 70
    P.status[0] = abi.FLYING
    P.n[0] = 1
    return P, float(psi), float(dv), float(area), float(m), float(surface)


def _open_world(t_end, n_t=10, air_rho=1.204, gravity=G):
    H = abi.pile(1, 1, (1e3, 1e3, 0.0), np.eye(3), 1e-3)          # no surface anywhere near
    W = abi.window(n_t)
    W.t_end_s[0] = t_end
    W.dt_tick_s[0] = t_end / n_t
    W.gravity_M[:] = gravity
    S = abi.scene(0, n_t, 0, 0)
    S.box_lo_M[:] = -1e4
    S.box_hi_M[:] = 1e4
    S.air[:] = (air_rho, 1.81e-5, 0.0257, 0.71, 293.15)
    S.jet_const[:] = (5.8, 0.094)
    return H, W, S


def test_ballistic_trajectory_is_exact_without_air():
    C = _model()
    x0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([3.0, -1.5, 4.0])
    P, *_ = _one_particle(x0, v0, 1e-4, 2e-3, 4e-3)
    t_total = 0.0
    for _ in range(20):
        H, W, S = _open_world(0.01, air_rho=0.0)
        W.t0_s[0] = t_total
        W.t_end_s[0] = t_total + 0.01
        out, ev = abi.step_out(4)
        kernels.advance_particles(P, H, W, S, C, out, ev)
        t_total += 0.01
    expected_x = x0 + v0 * t_total + 0.5 * G * t_total ** 2
    expected_v = v0 + G * t_total
    np.testing.assert_allclose(P.pos_M_m[0], expected_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(P.vel_M_m_s[0], expected_v, rtol=1e-12, atol=1e-12)
    assert P.time_s[0] == t_total


def _hl_cd(re, psi, aero):
    a, b, c, d = params.hl_coefficients(aero, psi)
    return 24.0 / re * (1 + a * re ** b) + c / (1 + d / re)


def test_terminal_velocity_matches_haider_levenspiel():
    C = _model()
    aero = params.load().aero
    side = 1e-3
    P, psi, dv, area, m, _ = _one_particle([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], side, side, side)
    rho, mu = 1.204, 1.81e-5
    # analytic: m g = 1/2 rho Cd(Re) A v^2 with the volume-equivalent sphere area
    f = lambda v: m * STANDARD_GRAVITY_M_S2 - 0.5 * rho * _hl_cd(rho * v * dv / mu, psi, aero) * area * v * v
    v_t = brentq(f, 1e-6, 100.0, xtol=1e-14)
    # v(t) -> v_t like tanh(g t / v_t): after 10 s (17 time constants) the remaining deficit is < 1e-14
    H, W, S = _open_world(10.0, n_t=100)
    out, ev = abi.step_out(4)
    kernels.advance_particles(P, H, W, S, C, out, ev)
    assert math.isclose(-P.vel_M_m_s[0, 2], v_t, rel_tol=1e-6)
    assert abs(psi - math.pi ** (1 / 3) * 6 ** (2 / 3) / 6) < 1e-12        # a cube: 0.806
    # a flat flake falls far slower than a cube of the same mass
    P2, *_ = _one_particle([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1e-4, 3.16e-3, 3.16e-3)
    H, W, S = _open_world(3.0, n_t=30)
    kernels.advance_particles(P2, H, W, S, C, *abi.step_out(4))
    assert -P2.vel_M_m_s[0, 2] < 0.5 * v_t


def test_liquid_jet_entrains_a_chip_to_the_jet_speed():
    C = _model()
    P, *_ = _one_particle([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1e-4, 1e-3, 2e-3)
    H, W, S = _open_world(0.002, n_t=2, gravity=np.zeros(3))
    S2 = abi.scene(0, 2, 0, 1)
    for name in ("box_lo_M", "box_hi_M", "air", "jet_const"):
        getattr(S2, name)[:] = getattr(S, name)
    S2.jet_origin_M[0] = (-0.01, 0.0, 0.0)
    S2.jet_dir_M[0] = (1.0, 0.0, 0.0)
    S2.jet_speed[0] = 15.0
    S2.jet_r0[0] = 2e-3
    S2.jet_tan[0] = 0.0
    S2.jet_gas[0] = 0
    S2.jet_fluid[0] = (998.0, 1.0e-3, 0.6, 7.0)
    S2.liquid[:] = (998.0, 1.0e-3, 0.6, 7.0, 293.15, 5e-5, 0.035)
    out, ev = abi.step_out(4)
    kernels.advance_particles(P, H, W, S2, C, out, ev)
    # coherent jet (alpha = 1): the chip leaves at nearly the jet speed (momentum transfer), it was at rest
    assert 0.9 * 15.0 < P.vel_M_m_s[0, 0] <= 15.0 + 1e-9


def test_lumped_cooling_matches_the_exponential_solution():
    cp = 900.0
    C = _model(cp_constant=cp, emissivity=0.0)
    t, w, length = 2e-4, 2e-3, 5e-3
    P, _, dv, _, m_chip, surface = _one_particle([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], t, w, length, temp=700.0)
    H, W, S = _open_world(0.5, n_t=50, gravity=np.zeros(3))
    out, ev = abi.step_out(4)
    kernels.advance_particles(P, H, W, S, C, out, ev)
    h = 0.0257 / dv * 2.0                                   # Ranz-Marshall at Re = 0: Nu = 2
    expected = 293.15 + (700.0 - 293.15) * math.exp(-h * surface * 0.5 / (m_chip * cp))
    assert math.isclose(P.temp_K[0], expected, rel_tol=1e-10)
    q = P.mass_kg[0] * cp * (700.0 - expected)
    assert math.isclose(out[abi.OUT_HEAT_AIR_J], q, rel_tol=1e-9)
    assert out[abi.OUT_HEAT_COOLANT_J] == 0.0


def test_settles_on_a_moving_table_with_the_table():
    """A chip dropped on the table of a moving grid frame settles in the cell it lands in (the table moves
    with the X axis under it)."""
    C = _model()
    H = abi.pile(20, 20, (0.0, 0.0, 0.0), np.eye(3), 2e-3)
    H.base_m[:] = 0.0
    H.base_kind[:] = abi.SURF_TABLE
    W = abi.window(10)
    W.t_end_s[0] = 0.2
    W.dt_tick_s[0] = 0.02
    for k in range(11):
        W.grid_origin_M[k] = (0.1 * 0.02 * k, 0.0, 0.0)        # table at 0.1 m/s along +x
    S = abi.scene(0, 10, 0, 0)
    S.box_lo_M[:] = -1.0
    S.box_hi_M[:] = 1.0
    S.air[:] = (1.204, 1.81e-5, 0.0257, 0.71, 293.15)
    P, *_ = _one_particle([0.02, 0.02, 0.005], [0.0, 0.0, 0.0], 1e-4, 2e-3, 3e-3)
    out, ev = abi.step_out(4)
    kernels.advance_particles(P, H, W, S, C, out, ev)
    assert P.status[0] == abi.FREE
    assert math.isclose(H.mass_kg.sum(), out[abi.OUT_DEPOSIT_KG], rel_tol=1e-15)
    iu, iv = np.unravel_index(np.argmax(H.mass_kg), H.mass_kg.shape)
    # fell about 32 ms (5 mm) while the table moved about 3 mm: the cell under it at contact, in G
    assert iv == 10 and iu in (8, 9)
