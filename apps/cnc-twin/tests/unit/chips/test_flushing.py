"""Flushing of settled chips by the coolant flow: threshold and bed-load rate against the hand calculation
(Shields; Wong & Parker 2006), capillary adhesion, and flushing through the domain."""

from __future__ import annotations

import math

import numpy as np

from cnctwin.chips import abi, kernels, params
from cnctwin.chips.domain import ChipsDomain
from cnctwin.chips.schema import settings as chips_settings
from cnctwin.core.bus import CoolantFeedback, Jet

import chipsfix as fx

G = 9.80665


def _cell(mass=2e-6, d=1.2e-3, q=0.0):
    C = params.chip_model(fx.project().material, chips_settings(None), params.load())
    H = abi.pile(3, 3, (0.0, 0.0, 0.0), np.eye(3), 2e-3)
    H.base_m[:] = 0.0
    H.base_kind[:] = abi.SURF_PART
    kernels.deposit(H, C, 1, 1, mass, 0.0, d, 1.5e-4, 2e-3, 3e-3, 62)
    H.flush_Pa[1, 1] = q
    H.film_uv_m_s[1, 1] = (0.5, 0.0)
    return H


def _fp(cap=0.0):
    fl = params.load().flush
    return np.array([fl["skin_friction_coefficient"], fl["critical_shields"], fl["bedload_coefficient"],
                     fl["bedload_exponent"], cap, 0.035, fl["bed_friction"], 2700.0])


def _hand(q, d, dt, cell=2e-3, rho_f=992.0, rho_s=2700.0):
    fl = params.load().flush
    theta = fl["skin_friction_coefficient"] * q / ((rho_s - rho_f) * G * d)
    if theta <= fl["critical_shields"]:
        return 0.0
    qs = fl["bedload_coefficient"] * (theta - fl["critical_shields"]) ** fl["bedload_exponent"] \
        * math.sqrt((rho_s / rho_f - 1) * G * d ** 3)
    return rho_s * qs * cell * dt


def test_bed_load_rate_matches_the_hand_calculation():
    d = 1.2e-3
    for q in (100.0, 500.0, 2000.0):
        H = _cell(mass=1e-3, d=d, q=q)
        rec, n = np.zeros((10, abi.MOB_WIDTH)), np.zeros(2, np.int64)
        kernels.flush_pile(H, _fp(), np.eye(3), 992.0, G, 0.01, rec, n)
        moved = float(rec[:n[0], abi.MOB_COLUMNS.index("mass_kg")].sum())
        assert math.isclose(moved, _hand(q, d, 0.01), rel_tol=1e-12, abs_tol=1e-20), q
        assert math.isclose(moved + H.mass_kg.sum(), 1e-3, rel_tol=1e-13)


def test_no_flushing_below_the_shields_threshold_and_capillary_holds_wet_beds():
    d = 1.2e-3
    fl = params.load().flush
    q_c = fl["critical_shields"] * (2700 - 992) * G * d / fl["skin_friction_coefficient"]
    H = _cell(d=d, q=0.99 * q_c)
    rec, n = np.zeros((10, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    kernels.flush_pile(H, _fp(), np.eye(3), 992.0, G, 0.01, rec, n)
    assert n[0] == 0
    H = _cell(d=d, q=3 * q_c)
    kernels.flush_pile(H, _fp(), np.eye(3), 992.0, G, 0.01, rec, n)
    assert n[0] == 1
    rec2, n2 = np.zeros((10, abi.MOB_WIDTH)), np.zeros(2, np.int64)
    H = _cell(d=d, q=3 * q_c)
    kernels.flush_pile(H, _fp(cap=8.0 * 0.9), np.eye(3), 992.0, G, 0.01, rec2, n2)     # MQL: not submerged
    assert n2[0] == 0


def test_domain_flushes_chips_off_the_part():
    proj = fx.project()
    ctx = fx.context(proj)
    state = fx.sim_state(1)
    dom = ChipsDomain(formation=fx.FakeFormation(), geometry=fx.scene_box())
    dom.init(ctx, state)
    cs = state.chips
    H = cs.pile
    C = dom.model
    for iu in range(20, 30):
        kernels.deposit(H, C, iu, 25, 2e-6, 2e-6 * 1e5, 1.2e-3, 1.5e-4, 2e-3, 3e-3, 62)
    cs.emitted[0] += 20e-6
    cs.emitted[1] += 20e-6 * 1e5
    shape = H.base_m.shape
    total0 = cs.emitted[0]
    for step in range(30):
        state.bus_next.motion = fx.motion(step, tip_M=[0.05, 0.04, 0.2], omega=0.0)
        state.bus_next.coolant = CoolantFeedback(
            on=True, kind="flood", jets=[Jet(nozzle_id="n1", kind="flood", origin_M_m=np.array([0.0, 0.05, 0.1]),
                                             direction_M=np.array([0.0, 0.0, -1.0]), speed_m_s=5.0, flow_m3_s=1e-5)],
            fluid_temp_K=295.0, wetted=np.ones(shape, bool), flush_pressure_Pa=np.full(shape, 1500.0))
        dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
        state.swap_bus()
        assert math.isclose(sum(cs.mass_terms().values()), total0, rel_tol=1e-9)
    on_part = float(H.mass_kg[H.base_kind == abi.SURF_PART].sum())
    assert on_part < 0.5 * 20e-6
    assert state.bus_prev.chips.heat_to_coolant_J > 0
