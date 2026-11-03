"""Flute fill against a hand calculation, packing, peck reset, and the drilling chain through the domain."""

from __future__ import annotations

import math

from cnctwin.chips import flutes, params
from cnctwin.chips.domain import ChipsDomain
from cnctwin.chips.protocols import ToolInfo
from cnctwin.chips.state import FL_FILL, FL_LENGTH_E_M, FL_MASS_KG, FL_TORQUE_NM
from cnctwin.core.findings import Category, Severity

import chipsfix as fx


def _drill(helix_deg=30.0, through=False):
    d = 0.01
    return ToolInfo(slot=0, kind="twist_drill", radius_m=d / 2, flute_length_m=0.05, n_flutes=2,
                    helix_rad=math.radians(helix_deg), gullet_area_m2=0.25 * math.pi * d * d / 4,
                    holder_radius_m=0.016, holder_length_m=0.06, through_coolant=through, rotating=True)


def test_steady_fill_matches_the_hand_calculation():
    fp = params.load().flute
    tool = _drill()
    omega = -2 * math.pi * 2000 / 60
    length = 0.02
    q_in = 1.57e-6
    mu = float(fp["flute_friction"])
    delta = tool.helix_rad
    # hand calculation (docs/models/chips.md): v_s = |w| R cos d (sin d - mu cos d), attenuated exp(-k L/D)
    v_s = abs(omega) * tool.radius_m * math.cos(delta) * (math.sin(delta) - mu * math.cos(delta))
    v_eff = v_s * math.exp(-float(fp["depth_attenuation_per_diameter"]) * length / tool.diameter_m)
    f_hand = q_in / (tool.gullet_area_m2 * v_eff)
    assert math.isclose(flutes.steady_fill(tool, q_in, length, omega, 1.0, 0.0, fp), f_hand, rel_tol=1e-13)
    # transient: dV/dt = Q - V/tau integrated exactly step by step
    volume, dt = 0.0, 0.01
    tau = length / v_eff
    for k in range(1, 201):
        step = flutes.advance(tool, volume, q_in, length, omega, 0.0, dt, 3.0, 1.0, 0.0, 276e6, fp)
        volume = step.volume_m3
        expected = q_in * tau * (1 - math.exp(-k * dt / tau))
        assert math.isclose(volume, expected, rel_tol=1e-10)
    assert math.isclose(step.fill, f_hand, rel_tol=1e-6)
    assert step.torque_Nm == 0.0                     # below the critical fill: loose chips


def test_mass_balance_of_one_step_is_exact():
    fp = params.load().flute
    tool = _drill()
    v0 = 3e-7
    step = flutes.advance(tool, v0, 2e-6, 0.03, -300.0, 0.0, 0.01, 3.0, 1.0, 0.0, 276e6, fp)
    assert math.isclose(v0 + 2e-6 * 0.01, step.volume_m3 + step.out_m3, rel_tol=1e-14)


def test_peck_retract_empties_the_flutes():
    fp = params.load().flute
    step = flutes.advance(_drill(), 4e-7, 1e-6, 0.0, -300.0, 0.0, 0.01, 3.0, 1.0, 0.0, 276e6, fp)
    assert step.volume_m3 == 0.0 and math.isclose(step.out_m3, 4e-7 + 1e-8, rel_tol=1e-14)


def test_packing_stress_follows_kawakita_above_the_critical_fill():
    fp = params.load().flute
    sy = 276e6
    assert flutes.compaction_stress(0.9, 30.0, sy, fp) == 0.0
    f = 1.5
    c = 1 - float(fp["critical_fill"]) / f
    a = 1 - 1 / 30.0
    expected = float(fp["kawakita_half_stress_ratio"]) * sy * c / (a - c)
    assert math.isclose(flutes.compaction_stress(f, 30.0, sy, fp), expected, rel_tol=1e-14)
    assert flutes.compaction_stress(1e6, 30.0, sy, fp) == float(fp["max_stress_yield_ratio"]) * sy
    # a shallow helix barely conveys: the flutes clog and the packing torque appears
    tool = _drill(helix_deg=18.0)
    volume = 0.0
    for _ in range(400):
        step = flutes.advance(tool, volume, 2e-6, 0.03, -209.4, 1e-3, 0.01, 3.0, 1.0, 0.0, sy, fp)
        volume = step.volume_m3
    assert step.fill > 1.0 and step.torque_Nm > 0.0 and step.force_N > 0.0
    wall = math.pi * tool.diameter_m * 0.03
    assert math.isclose(step.torque_Nm, float(fp["wall_friction"]) * step.stress_Pa * wall * tool.radius_m,
                        rel_tol=1e-14)
    assert math.isclose(step.power_W, step.torque_Nm * 209.4 + step.force_N * 1e-3, rel_tol=1e-14)


def test_through_tool_coolant_and_helix_raise_the_evacuation_speed():
    fp = params.load().flute
    base = flutes.effective_velocity(_drill(30.0), -300.0, 0.02, 1.0, 0.0, fp)
    assert flutes.effective_velocity(_drill(40.0), -300.0, 0.02, 1.0, 0.0, fp) > base
    assert flutes.effective_velocity(_drill(30.0, through=True), -300.0, 0.02, 1.0, 1e-4, fp) > base
    assert flutes.effective_velocity(_drill(30.0), -300.0, 0.08, 1.0, 0.0, fp) < base     # deeper: slower
    straight = ToolInfo(0, "tap_cutting", 0.004, 0.02, 3, 0.0, 1e-5, 0.008, 0.05, False, True, "straight")
    assert flutes.conveying_velocity(straight, -300.0, 0.3) == 0.0


def _drill_project(helix=30.0):
    tool = fx.tool_raw(kind="twist_drill", flutes=2, helix=helix, cutting_length_mm=50.0, diameter_mm=10.0)
    return fx.project(tools=[tool], physics={"particles_per_tooth": 1.0, "chips": {"pile_cell_mm": 2.0,
                                                                                    "pile_margin_mm": 20.0}})


def _run_drilling(helix, steps, retract_at=None):
    proj = _drill_project(helix)
    ctx = fx.context(proj)
    state = fx.sim_state(1)
    geometry = fx.scene_box()
    depth = 0.02
    geometry.remove_box((0.0449, 0.0349, 0.03 - depth), (0.0551, 0.0451, 0.031))
    dom = ChipsDomain(formation=fx.FakeFormation(form=62, length_m=3e-3), geometry=geometry)
    dom.init(ctx, state)
    rho = 2700.0
    omega = -2 * math.pi * 2000 / 60
    mass_step = 1.4e-5
    removed = []
    for step in range(steps):
        tip = [0.05, 0.04, 0.03 - depth] if retract_at is None or step < retract_at else [0.05, 0.04, 0.05]
        state.bus_next.motion = fx.motion(step, tip_M=tip, omega=omega)
        cutting = retract_at is None or step < retract_at
        if cutting:
            ki = dom.kernel_inputs(ctx, state, state.bus_prev)
            rows = fx.emission_rows(step, 4, mass_step / 4, pos=tip, vel=[0.0, 0.5, 0.2])
            ki.emission.rec[:4, :rows.shape[1]] = rows
            ki.emission.n[0] = 4
            dom.absorb(ctx, state, ki, state.bus_next)
        volume = mass_step / rho if cutting else 0.0
        state.bus_next.removal = fx.removal(volume, point=tip)
        removed.append(volume * rho)
        dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
        state.swap_bus()
        terms = state.chips.mass_terms()
        assert math.isclose(sum(terms.values()), state.chips.emitted[0], rel_tol=1e-9, abs_tol=1e-18)
    return dom, ctx, state, mass_step, rho, omega


def test_drilling_fills_the_flutes_to_the_hand_calculated_steady_state():
    dom, ctx, state, mass_step, rho, omega = _run_drilling(30.0, 160)
    fp = params.load().flute
    fl = state.chips.flute[0]
    tool = dom.tools.tool(0)
    length = fl[FL_LENGTH_E_M]
    assert math.isclose(length, 0.02, rel_tol=0.1)            # the hole depth (cell resolution)
    q_in = mass_step * 3.0 / rho / 0.01                        # R = 3 for form 62, all chips confined
    f_hand = flutes.steady_fill(tool, q_in, length, omega, 1.0, 0.0, fp)
    assert math.isclose(fl[FL_FILL], f_hand, rel_tol=1e-3)
    assert fl[FL_MASS_KG] > 0.0
    assert state.chips.particles.n[0] > 0                      # chips leave the hole mouth as particles


def test_retract_empties_the_flutes_and_shallow_helix_raises_packing():
    dom, ctx, state, *_ = _run_drilling(30.0, 60, retract_at=40)
    assert state.chips.flute[0, FL_MASS_KG] == 0.0
    dom2, ctx2, state2, *_ = _run_drilling(17.5, 120)
    assert state2.chips.flute[0, FL_FILL] > 1.0 and state2.chips.flute[0, FL_TORQUE_NM] > 0.0
    packing = [f for f in state2.findings if f.category == Category.CHIP_PACKING]
    assert packing and packing[-1].severity == Severity.ERROR
    assert packing[-1].cause_chain                                   # traceable to the flute state
    record = state2.bus_prev.chips
    assert record.packing_torque_Nm[0] > 0 and record.flute_fill[0] > 1.0
