"""CoolantDomain with a fake bus and state: ownership, checkpoint/resume, determinism, dry / MQL / air, tank
bookkeeping, the pile grid, blocked and programmable nozzles, operations and the report."""

import copy
import json
import math

import numpy as np
import pytest

from cnctwin.coolant.channels import NAMES
from cnctwin.coolant.state import CoolantState
from cnctwin.core.bus import Bus, CoolantFeedback

import coolfix as CF


def fields_equal(a, b):
    for name in a.__dataclass_fields__:
        x, y = getattr(a, name), getattr(b, name)
        if name == "jets":
            assert len(x) == len(y)
            for j1, j2 in zip(x, y):
                fields_equal(j1, j2)
            continue
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y), name
        elif isinstance(x, dict):
            assert sorted(x) == sorted(y), name
            for k in x:
                if isinstance(x[k], np.ndarray):
                    assert np.array_equal(x[k], y[k]), (name, k)
                else:
                    assert x[k] == y[k], (name, k)
        else:
            assert x == y, name


def test_writes_only_its_own_record():
    dom, ctx, state = CF.make_domain()
    outs = CF.run(dom, ctx, state, 3)
    default = Bus()
    for name in Bus.WRITERS:
        if name != "coolant":
            assert repr(getattr(outs[-1], name)) == repr(getattr(default, name)), name
    assert isinstance(outs[-1].coolant, CoolantFeedback)
    assert dom.kernel_inputs(ctx, state, outs[-1]) is None


def test_checkpoint_resume_is_bit_identical():
    dom, ctx, state = CF.make_domain()
    CF.run(dom, ctx, state, 4)
    arrays, meta = state.coolant.to_arrays()
    meta = json.loads(json.dumps(meta))
    arrays = {k: np.array(v) for k, v in arrays.items()}
    resumed = copy.copy(state)
    resumed.coolant = CoolantState.from_arrays(arrays, meta)
    dom2, ctx2, _ = CF.make_domain()
    a = CF.run(dom, ctx, state, 4, start=4, chips_heat_J=3.0, thermal_heat_J=20.0, emitted_kg=1e-3)
    b = CF.run(dom2, ctx2, resumed, 4, start=4, chips_heat_J=3.0, thermal_heat_J=20.0, emitted_kg=1e-3)
    for x, y in zip(a, b):
        fields_equal(x.coolant, y.coolant)
    assert state.coolant.tank.temp_K == resumed.coolant.tank.temp_K


def test_two_runs_are_identical():
    runs = []
    for _ in range(2):
        dom, ctx, state = CF.make_domain()
        runs.append(CF.run(dom, ctx, state, 5))
    for x, y in zip(*runs):
        fields_equal(x.coolant, y.coolant)


def test_dry_when_the_pumps_are_off():
    dom, ctx, state = CF.make_domain()
    rec = CF.run(dom, ctx, state, 3, pumps={"flood": False})[-1].coolant
    assert not rec.on and rec.kind == "dry"
    assert rec.reach_fraction == {} and rec.h_flank_W_m2K == {} and rec.jets == []
    assert rec.friction_factor == {0: 1.0}
    assert rec.h_work_far_W_m2K == 0.0 and rec.through_tool_flow_m3_s == 0.0


def test_mql_and_air_fluids():
    dom, ctx, state = CF.make_domain(fluid="mql-ester")
    rec = CF.run(dom, ctx, state, 3)[-1].coolant
    assert rec.on and rec.kind == "mql"
    assert rec.mql_oil_m3_s > 0.0 and rec.air_flow_m3_s > 0.0
    assert all(j.kind == "mql" for j in rec.jets)
    assert not np.asarray(rec.wetted).any()                   # an MQL aerosol does not flood the part
    assert rec.tank_volume_m3 == 0.0 and rec.fluid_temp_K == pytest.approx(293.15)
    assert 0.0 < rec.friction_factor[0] < 1.0
    dom, ctx, state = CF.make_domain(fluid="air")
    rec = CF.run(dom, ctx, state, 3)[-1].coolant
    assert rec.kind == "air" and rec.friction_factor[0] == pytest.approx(1.0)
    assert 0.0 < rec.h_flank_W_m2K[0] < 5e3                  # air cools far less than a liquid


def test_tank_books_heat_and_drag_out_once():
    dom, ctx, state = CF.make_domain()
    cs = state.coolant
    E0 = cs.tank.energy_J
    outs = CF.run(dom, ctx, state, 5, thermal_heat_J=50.0, chips_heat_J=10.0, emitted_kg=0.002, booked=123)
    assert cs.totals["chips_heat_J"] == pytest.approx(10.0)       # one chips window, booked once
    assert cs.totals["thermal_heat_J"] == pytest.approx(250.0)
    assert cs.totals["chip_mass_kg"] == pytest.approx(0.002)
    assert cs.totals["drag_out_m3"] == pytest.approx(0.002 * dom.fluid.drag_out_m3_per_kg)
    heat_in = sum(o.coolant.heat_in_J for o in outs)
    heat_out = sum(o.coolant.heat_out_J for o in outs)
    mass_out = cs.tank.totals["mass_out_J"]
    assert cs.tank.energy_J - E0 == pytest.approx(heat_in - heat_out - mass_out, rel=1e-9)
    assert heat_in == pytest.approx(260.0 + cs.totals["pump_energy_J"], rel=1e-12)
    # residual: round-off of C T with C ~ 0.75 MJ/K
    assert abs(sum(o.coolant.tank_residual_J for o in outs)) < 1e-6 * heat_in
    # the supply is warmer than the tank by the pump heat
    assert outs[-1].coolant.fluid_temp_K > outs[-1].coolant.tank_temp_K


def test_pile_grid_carries_wetted_and_flush_maps():
    nu, nv, cell = 30, 25, 4e-3
    origin = np.array([-0.01, -0.01, 0.0])
    base = np.full((nu, nv), 0.03)
    height = np.zeros((nu, nv))
    height[12:15, 10:13] = 0.004
    pile = (origin, np.eye(3), cell, base, height)
    dom, ctx, state = CF.make_domain()
    rec = CF.run(dom, ctx, state, 3, pile=pile)[-1].coolant
    assert rec.wetted.shape == (nu, nv) and rec.flush_pressure_Pa.shape == (nu, nv)
    assert np.array_equal(rec.wetted_origin_S_m, origin) and rec.wetted_cell_m == cell
    assert np.array_equal(rec.wetted_axes_S, np.eye(3))
    # the flood film spreads over the flat part and percolates through the chip heap on it
    assert rec.wetted.all() and rec.flush_pressure_Pa.max() > 1e4


def test_chips_blocking_a_nozzle_stop_its_flow():
    dom, ctx, state = CF.make_domain()
    rec = CF.run(dom, ctx, state, 3, blocked=("n1",))[-1].coolant
    assert not rec.on and rec.jets == []
    # chips name the published jet ids: one orifice of a multi-orifice nozzle blocks the nozzle
    rec = CF.run(dom, ctx, state, 2, start=3, blocked=("n1#1",))[-1].coolant
    assert not rec.on
    rec = CF.run(dom, ctx, state, 2, start=5, blocked=("tsc#0",))[-1].coolant
    assert rec.on and [j.nozzle_id for j in rec.jets] == ["n1"]


def test_programmable_nozzle_positions():
    raw = CF.machine_raw(nozzles=[{"id": "pcool", "kind": "flood", "attached_to": "head", "pump": "flood",
                                   "position_mm": [90, 0, -40], "diameter_mm": 5,
                                   "programmable": {"code": "M34/M35", "positions": 34, "step_deg": 1.5,
                                                    "home_aim": [-0.7, 0, -0.7]}}])
    dom, ctx, state = CF.make_domain(proj=CF.project(machine=raw, nozzle_positions={1: 12}))
    rec = CF.run(dom, ctx, state, 2)[-1].coolant
    jet = rec.jets[0]
    home = np.array([-0.7, 0.0, -0.7]) / math.sqrt(0.98)
    angle = math.degrees(math.acos(float(np.clip(jet.direction_M @ home, -1, 1))))
    assert angle == pytest.approx(11 * 1.5, rel=1e-9)        # the operation's table: T1 at position 12
    rec = CF.run(dom, ctx, state, 2, start=2, nozzles={"pcool": 3})[-1].coolant
    angle = math.degrees(math.acos(float(np.clip(rec.jets[0].direction_M @ home, -1, 1))))
    assert angle == pytest.approx(2 * 1.5, rel=1e-9)         # motion's nozzle setting (M34/M35) wins
    events = [e.event for e in ctx.trace._memory]
    assert events.count("coolant:nozzle_position") == 2


def test_end_operation_idles_the_tank_and_finalize_reports():
    dom, ctx, state = CF.make_domain(proj=CF.project(coolant_degC=35.0, wait_min=30.0))
    CF.run(dom, ctx, state, 3)
    before = state.coolant.tank.temp_K
    dom.end_operation(ctx, state)
    after = state.coolant.tank.temp_K
    assert 293.15 < after < before                            # cooled toward the shop air during the wait
    report = dom.finalize(ctx, state)
    json.dumps(report)
    assert report["per_operation"] and report["tank"]["temp_end_K"] == after
    assert report["totals"]["pumped_m3"] > 0.0


def test_channels_and_macro_values():
    dom, ctx, state = CF.make_domain()
    outs = CF.run(dom, ctx, state, 3)
    values = dom.macro_values(state, outs[-1].coolant)
    assert set(values) == set(NAMES)
    assert {c.name for c in dom.channels()} == set(NAMES)
    assert values["coolant.reach_fraction"].shape == (4,)
    assert values["coolant.heat_in_W"] > 0.0


def test_hand_set_nozzles_are_aimed_at_the_tool_in_the_spindle():
    """A flex nozzle on the head (not programmable, not through-tool) is set by the operator to wet the cut of the tool
    in the spindle: aimed at the tool's zone it reaches the cut and lubricates the contact, whatever the machine file's
    aim (one setting for one tool length). With ``aim_hand_nozzles: 0`` the file's aim stays: a nozzle set for another
    tool length misses the cut, the contact gets no reach and dry friction (the D10 side step of docs/convergence.md:
    friction factor 1.0 at every step, the jet 26.6 mm from the cut)."""
    off = CF.machine_raw(nozzles=[{"id": "n1", "kind": "flood", "attached_to": "head", "position_mm": [70, 0, -60],
                                   "aim": [-70, 0, -250], "diameter_mm": 4, "pump": "flood"}])
    physics = {"every_n_macro": 1, "latency_macro": 1, "rays_per_jet": 32}
    dom, ctx, state = CF.make_domain(CF.project(machine=off, physics=dict(physics, aim_hand_nozzles=0)))
    fixed = CF.run(dom, ctx, state, 3)[-1].coolant
    assert fixed.on and fixed.reach_fraction[0] == 0.0 and fixed.friction_factor[0] == 1.0
    dom, ctx, state = CF.make_domain(CF.project(machine=off, physics=physics))
    aimed = CF.run(dom, ctx, state, 3)[-1].coolant
    assert aimed.reach_fraction[0] > 0.5 and aimed.friction_factor[0] < 1.0
    # the jet leaves the nozzle toward the tool's zone centre (one tool radius above the tip on its axis)
    jet = [j for j in aimed.jets if j.nozzle_id == "n1"][0]
    jet0 = [j for j in fixed.jets if j.nozzle_id == "n1"][0]
    assert np.linalg.norm(np.asarray(jet.direction_M) - np.asarray(jet0.direction_M)) > 0.1
