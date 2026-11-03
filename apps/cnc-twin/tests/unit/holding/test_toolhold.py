"""Tool holders: ER collet torque against a hand calculation of the screw-and-wedge chain, the REGO-FIX table
lookups, catalogue and interference fits, thermal loss of a shrink fit, side-lock walk, the pull-out walk rate
through the domain, and the adapter from cnctwin.tooling's ToolGeometry."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.holding import holdfix as fx
from cnctwin.core.findings import Category, Severity
from cnctwin.holding import toolhold as TH
from cnctwin.holding.domain import HoldingDomain

FT_LBF = 1.3558179483314


def hand_er(d_shank, size_thread_d, pitch, front_d, torque_Nm, mu_t=0.12, mu_c=0.12, collar=0.45, mu_k=0.10,
            mu_s=0.15, alpha_deg=8.0):
    """Power screw with collar friction (Budynas & Nisbett eq. 8-5/8-6), wedge of the 8 deg collet cone."""
    d2 = size_thread_d - 0.649519 * pitch
    lam = math.atan(pitch / (math.pi * d2))
    rho = math.atan(mu_t / math.cos(math.radians(30.0)))
    lever = d2 / 2 * math.tan(lam + rho) + mu_c * collar * front_d
    axial = torque_Nm / lever
    radial = axial / math.tan(math.radians(alpha_deg) + math.atan(mu_k))
    return axial, radial, mu_s * radial * d_shank / 2, mu_s * radial


def test_er32_collet_torque_matches_the_hand_calculation():
    index = TH.er_index("ER32", 0.020, 0.020)
    torque = TH.er_nut_torque_Nm(index, 0.020)
    assert torque == pytest.approx(100 * FT_LBF)                 # REGO-FIX 14.03: 100 ft-lbs for 8-20 mm
    er = TH.er_collet_clamp(0.020, torque, index)
    axial, radial, t_cap, f_ret = hand_er(0.020, 0.040, 0.0015, 0.033, torque)
    assert er["axial_force_N"] == pytest.approx(axial, rel=1e-12)
    assert er["radial_force_N"] == pytest.approx(radial, rel=1e-12)
    assert er["torque_cap_Nm"] == pytest.approx(t_cap, rel=1e-12)
    assert er["retention_N"] == pytest.approx(f_ret, rel=1e-12)
    # the numbers themselves: ~28.7 kN nut force, ~117 kN on the shank, ~176 Nm on 20 mm
    assert axial == pytest.approx(28.67e3, rel=2e-3)
    assert t_cap == pytest.approx(176.3, rel=2e-3)


def test_er25_on_12mm_against_the_rego_fix_measurement():
    """REGO-FIX measured 89 Nm transferable torque on a 12 mm carbide rod (collet size and nut torque not stated):
    the chain at the recommended ER 25 torque gives 105 Nm, +18 % (the model's stated error band, 25 %)."""
    td = fx.tool_data(shank_d_mm=12.0, collet_size="ER25", bore_max_m=0.016)
    clamp = TH.build_clamp(td)
    assert clamp.model == "friction" and clamp.detail["size"] == 25
    assert clamp.detail["nut_torque_Nm"] == pytest.approx(80 * FT_LBF)
    assert clamp.torque_cap_Nm == pytest.approx(89.0, rel=0.25)
    assert clamp.torque_cap_Nm > 89.0


def test_collet_size_from_the_bore_and_torque_rows():
    assert TH.er_index("", 0.020, 0.010) == TH.er_index("ER32", None, 0.01)
    assert TH.er_index("ER 16", None, 0.006) == TH.er_index("", 0.010, 0.006)
    # a diameter between two rows takes the nearer one (ER32: 2.5 and 3.0 mm rows)
    i32 = TH.er_index("ER32", None, 0.010)
    assert TH.er_nut_torque_Nm(i32, 0.0026) == pytest.approx(20 * FT_LBF)
    assert TH.er_nut_torque_Nm(i32, 0.0029) == pytest.approx(100 * FT_LBF)


def test_catalogue_torque_gives_the_retention_of_the_same_friction():
    td = fx.tool_data(holder_type="hydraulic", shank_d_mm=12.0, clamping_torque_Nm=110.0, clamp_length_m=0.046)
    clamp = TH.build_clamp(td)
    assert clamp.torque_cap_Nm == 110.0
    assert clamp.retention_N == pytest.approx(2 * 110.0 / 0.012)        # F = 2 T / d on one friction circle
    assert clamp.pressure_Pa == pytest.approx(2 * 110.0 / (0.15 * math.pi * 0.012 ** 2 * 0.046))


def test_shrink_fit_lame_pressure_and_thermal_loss():
    td = fx.tool_data(holder_type="shrink_fit", shank_d_mm=12.0, nose_d_m=0.024, clamp_length_m=0.036,
                      shank_E_Pa=580e9, shank_poisson=0.22, shank_expansion_1_K=5.5e-6)
    clamp = TH.build_clamp(td)
    d, D = 0.012, 0.024
    delta = 1.3e-3 * d
    p = delta / (d * ((D * D + d * d) / (D * D - d * d) + 0.292) / 207e9 + d * (1 - 0.22) / 580e9)
    assert clamp.pressure_Pa == pytest.approx(p, rel=1e-12)
    assert clamp.torque_cap_Nm == pytest.approx(0.15 * p * math.pi * d * 0.036 * d / 2, rel=1e-12)
    assert clamp.torque_cap_Nm == pytest.approx(150.0, rel=0.03)      # calibrated on the CELSIO 12 mm entry
    # the holder 40 K warmer than the shank loses d (alpha_h 40 K) of its interference
    ref = 293.15
    factor = TH.pressure_factor(clamp, ref + 40.0, ref, ref)
    assert factor == pytest.approx(1.0 - d * 11.7e-6 * 40.0 / delta, rel=1e-12)
    assert TH.pressure_factor(clamp, ref + 400.0, ref, ref) == 0.0      # shrink-fit release temperature
    assert TH.pressure_factor(TH.build_clamp(fx.tool_data()), ref + 40.0, ref, ref) == 1.0   # collets: none


def test_side_lock_is_positive_in_torque_and_walks_to_its_flat():
    clamp = TH.build_clamp(fx.tool_data(holder_type="side_lock", shank_d_mm=20.0))
    assert clamp.model == "side_lock" and not math.isfinite(clamp.torque_cap_Nm)
    d_screw = 0.8 * 0.020
    preload = 227e6 * 0.76 * math.pi * d_screw ** 2 / 4
    assert clamp.retention_N == pytest.approx(2 * 0.15 * preload, rel=1e-12)
    assert clamp.walk_limit_m == pytest.approx(0.2e-3)


def test_positive_tapping_and_non_rotating_holders():
    assert TH.build_clamp(fx.tool_data(holder_type="arbor")).model == "positive"
    assert TH.build_clamp(fx.tool_data(holder_type="tapping_floating", float_travel_m=3e-3)).model == "tapping"
    assert TH.build_clamp(fx.tool_data(holder_type="turret_vdi", rotating=False)).model == "non_rotating"
    clamp = TH.build_clamp(fx.tool_data(holder_type=""))
    assert clamp.model == "none" and clamp.notes
    with pytest.raises(Exception):
        TH.build_clamp(fx.tool_data(holder_type="boring_bar_clamp"))


def _domain(td, n_ticks=fx.N_TICKS):
    proj = fx.project([fx.entry(fx.VISE, translate_mm=(50, 0, 0), input_torque_Nm=80, parallels_mm=10)])
    ctx = fx.context(proj)
    state = fx.sim_state()
    dom = HoldingDomain(tools={0: td})
    dom.init(ctx, state)
    return dom, ctx, state


def test_pullout_walk_rate_under_a_steady_axial_pull_above_the_retention():
    """One tick of a steady axial pull D above the retention C: free acceleration of the tool mass,
    walk = (D - C) / m dt^2 / 2 (exact sub-stepping), per revolution = walk / (omega dt / 2 pi)."""
    td = fx.tool_data(shank_d_mm=10.0)
    dom, ctx, state = _domain(td, n_ticks=1)
    clamp = dom._rt["clamps"][0]
    C = clamp.retention_N
    D = C + 1.0                                   # 10 um walk: below the 0.02 mm error threshold
    omega = -1047.2
    bus = fx.bus_with(work_mean=np.zeros(6), tool_mean=[0, 0, -D, 0, 0, 0], omega=omega, n=1)
    rec = fx.run(dom, ctx, state, bus)
    t = rec.tools[0]
    dt = ctx.ipo_s
    walk = 0.5 * (D - C) / td.tool_mass_kg * dt * dt
    assert t.pullout_m == pytest.approx(walk, rel=1e-9)
    assert t.walk_per_rev_m == pytest.approx(walk / (abs(omega) * dt / (2 * math.pi)), rel=1e-9)
    assert t.holding_margin == pytest.approx(C / D, rel=1e-12)
    assert t.slip_rad == 0.0
    assert t.slip_friction_J == pytest.approx(C * walk, rel=1e-9)
    pulls = [f for f in state.findings if f.category == Category.TOOL_PULLOUT]
    assert pulls and pulls[0].severity == Severity.WARNING
    assert pulls[0].quantities["depth_error_m"] == pytest.approx(walk, rel=1e-9)
    # a larger excess walks beyond the depth tolerance: an error
    bus = fx.bus_with(work_mean=np.zeros(6), tool_mean=[0, 0, -(C + 500.0), 0, 0, 0], omega=omega, n=1, tick0=1000)
    fx.run(dom, ctx, state, bus)
    assert any(f.severity == Severity.ERROR for f in state.findings if f.category == Category.TOOL_PULLOUT)


def test_pullout_below_capacity_is_zero_and_torque_slip_drags_the_tool_out():
    td = fx.tool_data(shank_d_mm=10.0)
    dom, ctx, state = _domain(td)
    clamp = dom._rt["clamps"][0]
    T_cap = clamp.torque_cap_Nm
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), tool_mean=[0, 0, -200, 0, 0, 0.4 * T_cap],
                                              tool_peak=[0, 0, -400, 0, 0, 0.8 * T_cap]))
    assert rec.tools[0].pullout_m == 0.0 and rec.tools[0].slip_rad == 0.0
    assert rec.tools[0].holding_margin == pytest.approx(1.0 / math.hypot(0.8, 400.0 / clamp.retention_N), rel=1e-6)
    # torque above the friction circle: the tool turns in the holder and the small axial pull walks it out
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), tool_mean=[0, 0, -200, 0, 0, 0.9 * T_cap],
                                              tool_peak=[0, 0, -400, 0, 0, 1.2 * T_cap]))
    t = rec.tools[0]
    assert t.slip_rad > 0.0 and t.pullout_m > 0.0 and t.holding_margin < 1.0
    assert any(f.category == Category.TOOL_PULLOUT and "turns" in f.message for f in state.findings)


def test_pullout_stops_at_the_clamped_length_with_a_fatal_finding():
    td = fx.tool_data(holder_type="hydraulic", shank_d_mm=10.0, clamping_torque_Nm=50.0, clamp_length_m=0.004)
    dom, ctx, state = _domain(td)
    clamp = dom._rt["clamps"][0]
    for step in range(6):
        rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), tool_mean=[0, 0, -3 * clamp.retention_N,
                                                                                      0, 0, 0], tick0=step * 10000))
    assert rec.tools[0].pullout_m == pytest.approx(0.004)
    fatal = [f for f in state.findings if f.severity == Severity.FATAL]
    assert len(fatal) == 1 and fatal[0].category == Category.TOOL_PULLOUT


def test_side_lock_walk_ends_at_its_positive_stop():
    td = fx.tool_data(holder_type="side_lock", shank_d_mm=20.0)
    dom, ctx, state = _domain(td)
    for step in range(4):
        rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), tool_mean=[0, 0, -12000, 0, 0, 80],
                                                  tool_peak=[0, 0, -16000, 0, 0, 120], tick0=step * 10000))
    t = rec.tools[0]
    assert t.pullout_m == pytest.approx(0.2e-3) and t.slip_rad == 0.0
    assert not math.isfinite(t.torque_capacity_Nm)


def test_floating_tapping_holder_at_the_end_of_its_travel():
    td = fx.tool_data(holder_type="tapping_floating", float_travel_m=0.003)
    dom, ctx, state = _domain(td)
    bus = fx.bus_with(work_mean=np.zeros(6), tool_mean=[0, 0, 100, 0, 0, 5])
    bus.cutting.contacts[0].float_u_max_m = 0.0031
    rec = fx.run(dom, ctx, state, bus)
    assert rec.tools[0].float_bottomed
    assert any(f.category == Category.TOOL_PULLOUT and "compensation" in f.message for f in state.findings)


def test_tool_data_from_a_tooling_geometry():
    mass = SimpleNamespace(mass_kg=0.06, com_z_m=0.03, I_axial_kgm2=8e-7, I_trans_kgm2=2e-5)
    hmass = SimpleNamespace(mass_kg=1.3, com_z_m=0.11, I_axial_kgm2=1e-3, I_trans_kgm2=4e-3)
    props = SimpleNamespace(flutes=3, tool_mass=mass, holder_mass=hmass,
                            balance_tool=None, balance_holder=SimpleNamespace(grade_m_s=0.0025, speed_rad_s=2618.0),
                            max_rad_s_tool=4000.0, max_rad_s_holder=2618.0, stickout_m=0.035)
    from cnctwin.db.simap import SIMap
    raw = SIMap({"id": "h", "type": "er_collet", "collet_size": "ER32", "bore_mm": [1.5, 20.0],
                 "axial_float_mm": None})
    holder = SimpleNamespace(type="er_collet", clamp_length_m=None, nose_diameter_m=0.05, clamping_torque_Nm=None,
                             bore_m=0.02, raw=raw)
    edge = SimpleNamespace(runout_T_m=np.array([2e-6, 0, 0]), tilt_rad=np.array([0.0, 1e-5]))
    geo = SimpleNamespace(slot=2, number=7, tool_id="t", holder_id="h", rotating=True, hand_sign=-1.0, props=props,
                          holder=holder, spec=SimpleNamespace(tool_material_class="carbide", tool_material_id=None,
                                                              geometry={}),
                          layout={"d_shank": 0.010}, edge=edge)
    td = TH.tool_data(geo)
    assert (td.slot, td.number, td.flutes, td.holder_type, td.collet_size) == (2, 7, 3, "er_collet", "ER32")
    assert td.shank_d_m == 0.010 and td.bore_max_m == pytest.approx(0.020)
    assert td.holder_grade == (0.0025, 2618.0) and td.tool_grade is None
    np.testing.assert_allclose(td.runout_T_m, [2e-6, 0, 0])
    assert TH.build_clamp(td).torque_cap_Nm == pytest.approx(88.1, rel=2e-3)
