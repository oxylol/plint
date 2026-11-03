"""Supply: Darcy-Weisbach against hand calculations, pump curves, flow split, relief valve, air."""

import math

import numpy as np
import pytest

from cnctwin.coolant import fluid as F
from cnctwin.coolant import supply as S
from cnctwin.db.simap import SIMap

import coolfix as CF

WATER = F.Props(rho_kg_m3=998.0, mu_Pa_s=1.0e-3, k_W_mK=0.6, cp_J_kgK=4180.0, sigma_N_per_m=0.072)
TOL, ITER = 1e-12, 200


def colebrook_fixed_point(re, rr, n=200):
    """Hand calculation: plain fixed-point iteration of Colebrook-White from f = 0.02."""
    f = 0.02
    for _ in range(n):
        f = (-2.0 * math.log10(rr / 3.7 + 2.51 / (re * math.sqrt(f)))) ** -2
    return f


def test_laminar_friction_is_hagen_poiseuille():
    for re in (10.0, 500.0, 1999.0):
        assert S.darcy(re, 1e-3, TOL, ITER) == pytest.approx(64.0 / re, rel=1e-15)


@pytest.mark.parametrize("re, rr", [(4.0e3, 0.0), (1.0e5, 0.0), (1.0e5, 1.0e-3), (1.0e6, 1.0e-4), (5.0e7, 1e-2)])
def test_colebrook_matches_fixed_point_hand_calc(re, rr):
    assert S.darcy(re, rr, TOL, ITER) == pytest.approx(colebrook_fixed_point(re, rr), rel=1e-10)


def test_moody_chart_values():
    # Moody (1944): smooth pipe at Re = 1e5 reads f = 0.018; eps/D = 0.001 at Re = 1e5 reads 0.022
    assert S.darcy(1e5, 0.0, TOL, ITER) == pytest.approx(0.0180, abs=2e-4)
    assert S.darcy(1e5, 1e-3, TOL, ITER) == pytest.approx(0.0222, abs=3e-4)


def test_transition_is_continuous_and_monotone():
    p = S.P()
    lo, hi = p["laminar_re_hi"], p["turbulent_re_lo"]
    assert S.darcy(lo * (1 + 1e-12), 0.0, TOL, ITER) == pytest.approx(64.0 / lo, rel=1e-9)
    assert S.darcy(hi * (1 - 1e-12), 0.0, TOL, ITER) == pytest.approx(S.colebrook(hi, 0.0, TOL, ITER), rel=1e-9)
    pipe = S.Pipe("p", 0.01, 2.0, 1.5e-6)
    flows = np.linspace(1e-6, 5e-4, 400)
    dps = [pipe.dp(q, WATER, TOL, ITER) for q in flows]
    assert np.all(np.diff(dps) > 0.0)


def test_pipe_pressure_drop_hand_calc():
    # 10 l/min of water through 2 m of 10 mm hose (eps 1.5 um) with K = 3 of fittings
    d, L, eps, K = 0.010, 2.0, 1.5e-6, 3.0
    q = 10.0 / 60000.0
    v = q / (math.pi * d * d / 4.0)
    re = 998.0 * v * d / 1.0e-3
    f = colebrook_fixed_point(re, eps / d)
    hand = (f * L / d + K) * 0.5 * 998.0 * v * v
    pipe = S.Pipe("hose", d, L, eps, K)
    assert pipe.dp(q, WATER, TOL, ITER) == pytest.approx(hand, rel=1e-10)
    # worked by hand: v = 2.122 m/s, Re = 21180, f = 0.02587, dp = (0.02587 * 200 + 3) * 2247 Pa
    assert hand == pytest.approx(18.37e3, rel=2e-3)


def test_orifice_includes_jet_energy_and_approach_velocity():
    tip = S.Orifice("o", 0.004, 0.9, 0.00635)
    q = 2.0e-4
    beta = 0.004 / 0.00635
    hand = (1 - beta ** 4) * 0.5 * 998.0 * (q / (0.9 * math.pi * 0.004 ** 2 / 4)) ** 2
    assert tip.dp(q, WATER, TOL, ITER) == pytest.approx(hand, rel=1e-14)
    assert tip.exit_speed(q) == pytest.approx(q / (math.pi * 0.004 ** 2 / 4), rel=1e-14)


def test_branch_inverse_flow():
    branch = S.Branch("n", (S.Pipe("h", 0.00635, 0.3, 1.5e-6, 3.0), S.Orifice("o", 0.004, 0.9, 0.00635)))
    for dp in (1e4, 2e5, 3e6):
        q = branch.flow_at(dp, WATER, TOL, ITER)
        assert branch.dp(q, WATER, TOL, ITER) == pytest.approx(dp, rel=1e-9)


def test_centrifugal_operating_point_and_equal_split():
    pump = S.PumpModel("flood", "flood", False, 2.0e5, 45.0 / 60000.0)
    line = (S.Pipe("line", 0.016, 2.5, 1.5e-6, 4.0),)
    nozzle = lambda name: S.Branch(name, (S.Pipe("h", 0.00635, 0.3, 1.5e-6, 3.0),
                                          S.Orifice("o", 0.004, 0.9, 0.00635)))
    static = 998.0 * 9.80665 * 1.2
    c = S.solve_circuit(pump, line, [nozzle("a"), nozzle("b")], WATER, static, TOL, ITER)
    qa, qb = c.branch_flow_m3_s["a"], c.branch_flow_m3_s["b"]
    assert qa == pytest.approx(qb, rel=1e-12)
    assert qa + qb == pytest.approx(c.flow_m3_s, rel=1e-9)
    line_dp = sum(e.dp(c.flow_m3_s, WATER, TOL, ITER) for e in line) + static
    assert pump.pressure(c.flow_m3_s) == pytest.approx(c.manifold_Pa + line_dp, rel=1e-9)
    # the shut-off ratio gives p0 = 1.25 p_r at Q = 0 and the rated point lies on the curve
    assert pump.pressure(0.0) == pytest.approx(1.25 * 2.0e5)
    assert pump.pressure(45.0 / 60000.0) == pytest.approx(2.0e5)
    # a third nozzle draws more flow at a lower manifold pressure
    c3 = S.solve_circuit(pump, line, [nozzle("a"), nozzle("b"), nozzle("c")], WATER, static, TOL, ITER)
    assert c3.flow_m3_s > c.flow_m3_s and c3.manifold_Pa < c.manifold_Pa


def _drill_shape(diameter_mm=1.0):
    from cnctwin.coolant.protocols import ToolShape
    return ToolShape(slot=0, number=1, type="twist_drill", rotating=True, radius_m=0.004, flute_length_m=0.045,
                     z_gauge_m=0.115, holder_length_m=0.08, tool_class="carbide", hole_count=2,
                     hole_diameter_m=diameter_mm * 1e-3, hole_length_m=0.09, hole_exit_kind="tip",
                     hole_pos_T_m=np.array([[0.0, 0.002, 0.0005], [0.0, -0.002, 0.0005]]),
                     hole_dir_T=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]))


def _model(machine=None, fluid="emulsion-8pct"):
    return S.SupplyModel(SIMap(machine or CF.machine_raw()), F.fluid_from_db(CF.fluid_db(fluid)), TOL, ITER)


def test_displacement_pump_relief_valve_with_small_holes():
    model = _model()
    res = model.solve({"tsc": True}, False, (), _drill_shape(1.0), 295.0, 293.15)
    c = res.circuits["tsc"]
    assert c.relief_open
    assert c.pressure_Pa == pytest.approx(21e5)
    assert 0.0 < c.flow_m3_s < 11.0 / 60000.0
    assert c.bypass_m3_s == pytest.approx(11.0 / 60000.0 - c.flow_m3_s, rel=1e-12)
    eta = S.P()["pump_efficiency_displacement"]
    assert c.shaft_power_W == pytest.approx(21e5 * 11.0 / 60000.0 / eta, rel=1e-12)
    holes = [o for o in res.outlets if o.through_tool]
    assert len(holes) == 2 and holes[0].flow_m3_s == pytest.approx(c.flow_m3_s / 2)
    # the jet speed is the hole velocity
    assert holes[0].speed_m_s == pytest.approx(c.flow_m3_s / (2 * math.pi * 1e-6 / 4), rel=1e-12)
    assert any(k.startswith("relief:tsc") for k, _ in res.notes)


def test_displacement_pump_delivers_rated_flow_through_large_holes():
    model = _model()
    res = model.solve({"tsc": True}, False, (), _drill_shape(3.0), 295.0, 293.15)
    c = res.circuits["tsc"]
    assert not c.relief_open
    assert c.flow_m3_s == pytest.approx(11.0 / 60000.0)
    assert c.pressure_Pa < 21e5


def test_through_spindle_without_holes_delivers_nothing():
    model = _model()
    shape = _drill_shape()
    from dataclasses import replace
    res = model.solve({"tsc": True}, False, (), replace(shape, hole_count=0), 295.0, 293.15)
    assert not res.outlets
    assert any(k.startswith("no_holes") for k, _ in res.notes)
    assert res.circuits["tsc"].relief_open and res.shaft_power_W > 0.0


def test_blocked_nozzle_gets_no_flow():
    model = _model()
    open_ = model.solve({"flood": True}, False, (), None, 295.0, 293.15)
    blocked = model.solve({"flood": True}, False, ("n1",), None, 295.0, 293.15)
    assert [o.jet_id for o in open_.outlets] == ["n1"]
    assert not blocked.outlets


def test_choked_air_orifice_hand_calc():
    # orifice-limited: a 2 mm nozzle at 6.9 bar gauge cannot pass 1000 l/min
    gamma, R, T0 = 1.4, 287.05, 293.15
    area = math.pi * 0.002 ** 2 / 4
    jet = S.air_outlets(area, 6.9e5, 1.0, T0, TOL, ITER)
    p0 = 6.9e5 + 101325.0
    cd = S.P()["air_nozzle_discharge_coefficient"]
    hand = cd * area * p0 / math.sqrt(R * T0) * math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    assert jet.choked and jet.mass_kg_s == pytest.approx(hand, rel=1e-12)
    # fully expanded velocity
    te = T0 * (101325.0 / p0) ** ((gamma - 1) / gamma)
    assert jet.speed_m_s == pytest.approx(math.sqrt(2 * 1005.0 * (T0 - te)), rel=1e-12)
    # supply-limited: a large orifice passes exactly the supply's mass flow at a lower pressure
    big = S.air_outlets(math.pi * 0.01 ** 2 / 4, 6.9e5, 170.0 / 60000.0, T0, TOL, ITER)
    assert big.mass_kg_s == pytest.approx(170.0 / 60000.0 * 101325.0 / (R * T0), rel=1e-9)
    assert big.stagnation_Pa < p0


def test_mql_fluid_turns_outlets_into_aerosol():
    model = _model(fluid="mql-ester")
    res = model.solve({"flood": True}, False, (), None, 293.15, 293.15)
    assert res.outlets and all(o.phase == S.AEROSOL and o.kind == "mql" for o in res.outlets)
    assert res.mql_oil_m3_s == pytest.approx(20e-6 / 3600.0, rel=1e-9)
    assert res.air_free_m3_s > 0.0


def test_air_blast_uses_flood_nozzles_without_air_nozzles():
    model = _model()
    res = model.solve({}, True, (), None, 295.0, 293.15)
    assert [o.kind for o in res.outlets] == ["air"]
    # the 4 mm nozzle could pass ~20 g/s at 6.9 bar: the 170 l/min supply limits the flow
    assert res.air_free_m3_s == pytest.approx(170.0 / 60000.0, rel=1e-9)
