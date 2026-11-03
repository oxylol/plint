"""Surface integrity of the machined layer: mechanical (line-contact yield and shakedown) and thermal residual
stress, depths, hardness, white layer / burn flags, and their assignment to the generated patches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.surface import surffix as fx
from cnctwin.surface import integrity as IN
from cnctwin.surface import params as PRM
from cnctwin.surface.domain import SurfaceDomain
from cnctwin.surface.store import PF

MAT = PRM.material_props(fx.MATERIAL)
P = PRM.integrity()
AMB = 293.15
K_SHEAR = 505e6 / math.sqrt(3.0)


def _force_for(p0, l_c):
    return p0 * l_c * math.pi / 4.0


def test_no_load_no_stress():
    res = IN.evaluate(0.0, 0.0, 0.0, 2.0, AMB, AMB, MAT, P)
    assert res.sigma_surface_Pa == 0.0 and res.affected_depth_m == 0.0 and res.hardness_change_frac == 0.0


def test_mechanical_stress_between_yield_and_shakedown():
    l_c = 40e-6
    mu = 0.2                                                    # shakedown 4 k (frictionless limit governs)
    below = IN.evaluate(_force_for(3.0 * K_SHEAR, l_c), 20e-6, 20e-6, 2.0, AMB, AMB, MAT, P, mu)
    assert below.sigma_mech_Pa == 0.0
    mid = IN.evaluate(_force_for(3.55 * K_SHEAR, l_c), 20e-6, 20e-6, 2.0, AMB, AMB, MAT, P, mu)
    assert mid.sigma_mech_Pa == pytest.approx(-505e6 * (3.55 - 3.1) / (4.0 - 3.1), rel=1e-9)
    assert mid.depth_mech_m == pytest.approx(2.0 * 0.5 * l_c)
    full = IN.evaluate(_force_for(6.0 * K_SHEAR, l_c), 20e-6, 20e-6, 2.0, AMB, AMB, MAT, P, mu)
    assert full.sigma_mech_Pa == pytest.approx(-505e6)
    # high friction lowers the shakedown limit to k / mu
    rough = IN.evaluate(_force_for(3.2 * K_SHEAR, l_c), 20e-6, 20e-6, 2.0, AMB, AMB, MAT, P, 0.3)
    assert rough.sigma_mech_Pa == pytest.approx(-505e6 * (3.2 - 3.1) / (1.0 / 0.3 - 3.1), rel=1e-9)


def test_thermal_stress_is_the_excess_over_yield_on_cooling():
    E, a, nu, sy = 210e9, 11.5e-6, 0.29, 505e6
    dT = (sy + 100e6) * (1.0 - nu) / (E * a)
    res = IN.evaluate(0.0, 20e-6, 20e-6, 2.0, AMB + dT, AMB, MAT, P)
    assert res.sigma_thermal_Pa == pytest.approx(100e6, rel=1e-9)
    kappa = 49.8 / (7850.0 * 486.0)
    assert res.depth_thermal_m == pytest.approx(math.sqrt(kappa * 40e-6 / 2.0), rel=1e-12)
    capped = IN.evaluate(0.0, 20e-6, 20e-6, 2.0, AMB + 5 * dT, AMB, MAT, P)
    assert capped.sigma_thermal_Pa == pytest.approx(sy)
    mild = IN.evaluate(0.0, 20e-6, 20e-6, 2.0, AMB + 0.5 * dT, AMB, MAT, P)
    assert mild.sigma_thermal_Pa == 0.0


def test_combined_profile_and_affected_depth():
    l_c = 40e-6
    E, a, nu, sy = 210e9, 11.5e-6, 0.29, 505e6
    dT = (sy + 200e6) * (1.0 - nu) / (E * a)
    res = IN.evaluate(_force_for(3.55 * K_SHEAR, l_c), 20e-6, 20e-6, 2.0, AMB + dT, AMB, MAT, P, 0.2)
    assert res.sigma_surface_Pa == pytest.approx(res.sigma_mech_Pa + res.sigma_thermal_Pa)
    z = np.array([0.0, res.affected_depth_m])
    prof = res.stress_at(z)
    assert prof[0] == pytest.approx(res.sigma_surface_Pa)
    assert abs(prof[1]) <= 0.1 * max(abs(res.sigma_mech_Pa), abs(res.sigma_thermal_Pa)) + 1.0


def test_hardness_white_layer_and_burn():
    l_c = 40e-6
    load = IN.evaluate(_force_for(4.0 * K_SHEAR, l_c), 20e-6, 20e-6, 2.0, AMB, AMB, MAT, P, 0.2)
    assert load.hardness_change_frac == pytest.approx(0.25, rel=1e-9)        # the database increase at full load
    assert load.hardened_depth_m == pytest.approx(60e-6 * l_c / 20e-6, rel=1e-9)
    hot = IN.evaluate(0.0, 20e-6, 20e-6, 2.0, 1100.0, AMB, MAT, P)
    assert hot.white_layer and hot.burn and hot.hardness_change_frac == pytest.approx(0.3)
    warm = IN.evaluate(0.0, 20e-6, 20e-6, 2.0, 700.0, AMB, MAT, P)
    assert warm.burn and not warm.white_layer


def test_patches_carry_the_integrity_of_their_last_generation():
    edge, outline = fx.endmill(5e-3, 4)
    ctx = fx.context(fx.project())
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    omega = -2.0 * math.pi * 10000.0 / 60.0
    ev, v = fx.side_wall(5e-3, 4, 0.05e-3, omega, 20, [12], edge, edge_r=20e-6)
    ev[:, 11] = 20e-6
    for t0, rows in fx.chunks(ev, 10 * fx.IPO):
        tk = fx.step_ticks(t0, 10 * fx.IPO)
        tip = np.stack([v * (tk - 1000) * fx.TICK_S, 0 * tk, 0 * tk], axis=1)
        fr = fx.frame(tk, tip, omega)
        c = fx.contact(rows, ploughing_J=0.05, in_cut_s=0.004, speed=5.0, edge_len=4e-3)
        b = fx.bus([fr], [c], thermal={0: 1100.0})
        dom.macro_step(ctx, state, b, fx.bus([fr], []))
    pf = state.surface.store.pf[0]
    # P_t = 0.05 / (0.004 * 5 * 4e-3) = 625 N/m; without coefficients the normal share is 1 / mu (0.4)
    expect = IN.evaluate(625.0 / 0.4, 20e-6, 20e-6, 5.0, 1100.0, AMB, MAT, P, 0.4)
    assert pf[PF["sigma"]] == pytest.approx(expect.sigma_surface_Pa)
    assert pf[PF["white"]] == 1.0 and pf[PF["T_peak"]] == 1100.0
