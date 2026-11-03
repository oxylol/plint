"""HoldingDomain with a fake bus and state: channels, settings and the physics schema, the published records,
bit-identical continuation across a checkpoint, a part released by removal, table acceleration as an inertial
load, a setup without workholding and the results document."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.holding import holdfix as fx
from cnctwin.core.bus import Bus, HoldingFeedback, PartHolding, ToolHolding
from cnctwin.core.channels import Tier
from cnctwin.core.errors import InputError
from cnctwin.core.findings import Category, Severity
from cnctwin.db.schema import Ctx
from cnctwin.holding.channels import NAMES, channels
from cnctwin.holding.domain import HoldingDomain, gravity_M
from cnctwin.holding.schema import PHYSICS_SCHEMA, settings
from cnctwin.holding.state import HoldingState

CENTROID = np.array([0.05, 0.03, 0.015])


def at(point, force):
    f = np.asarray(force, float)
    return np.concatenate([f, np.cross(np.asarray(point, float), f)])


def vise_domain(physics=None):
    proj = fx.project([fx.entry(fx.VISE, translate_mm=(50, 0, 0), input_torque_Nm=80, parallels_mm=10)],
                      physics=physics)
    ctx = fx.context(proj)
    state = fx.sim_state()
    dom = HoldingDomain(tools={0: fx.tool_data()})
    dom.init(ctx, state)
    return dom, ctx, state


def test_channels_settings_and_schema():
    chans = channels()
    assert {c.name for c in chans} == set(NAMES) and all(c.tier == Tier.MACRO and c.domain == "holding"
                                                            for c in chans)
    assert all(c.name.startswith("holding.") for c in chans)
    st = settings({"patch_nodes": 3, "newton_tol": 1e-8})
    assert st.patch_nodes == 3 and st.newton_tol == 1e-8 and st.every_n_macro == 1
    with pytest.raises(InputError):
        settings({"patch_nodes": 4})
    with pytest.raises(InputError):
        settings({"every_n_macro": 2})
    ctx = Ctx(None, obj="project", file="p")
    PHYSICS_SCHEMA.check({"patch_nodes": 5, "max_disp_m": 0.002}, ("physics", "holding"), None, ctx)
    assert not ctx.problems
    PHYSICS_SCHEMA.check({"unknown_setting": 1}, ("physics", "holding"), None, ctx)
    assert ctx.problems


def test_gravity_convention():
    np.testing.assert_allclose(gravity_M(None), [0, 0, -9.80665])
    np.testing.assert_allclose(gravity_M({"type": "lathe"}), [-9.80665, 0, 0])
    np.testing.assert_allclose(gravity_M({"type": "vertical_mill", "kinematics": {"gravity_direction": [0, -2, 0]}}),
                               [0, -9.80665, 0])


def test_the_records_and_their_additive_fields():
    dom, ctx, state = vise_domain()
    w = at(CENTROID, [1500.0, 0, 0])
    nxt = Bus()
    state.bus_prev = fx.bus_with(work_mean=w, tool_mean=[0, 0, -100, 0, 0, 5])
    dom.macro_step(ctx, state, state.bus_prev, nxt)
    rec = nxt.holding
    assert isinstance(rec, HoldingFeedback) and set(rec.tools) == {0} and set(rec.parts) == {0}
    tool, part = rec.tools[0], rec.parts[0]
    assert isinstance(tool, ToolHolding) and isinstance(part, PartHolding)
    assert tool.unbalance_static_kg_m.shape == (3,) and tool.unbalance_couple_kg_m2.shape == (3,)
    assert tool.torque_capacity_Nm > 0 and tool.axial_retention_N > 0 and tool.max_rad_s > 0
    assert part.pose_delta.shape == (4, 4) and part.support_stiffness_S.shape == (6, 6)
    assert part.margin == pytest.approx(8002.6 / 1500.0, rel=0.1) and part.seated and not part.free
    assert part.supports_version == 1 and part.supports
    # nothing else on the bus was written
    assert nxt.cutting.contacts == [] and nxt.motion.paths == []
    vals = dom.macro_values(state)
    assert set(vals) == set(NAMES)
    assert vals["holding.part_margin"].shape == (8,) and vals["holding.pullout_slot_m"].shape == (32,)


def _run(dom, ctx, state, loads):
    out = []
    for k, w in enumerate(loads):
        bus = fx.bus_with(work_mean=0.6 * w, work_peak=w, tool_mean=[0, 0, -100, 0, 0, 5],
                          tool_peak=[0, 0, -300, 0, 0, 20], tick0=k * 10000)
        rec = fx.run(dom, ctx, state, bus)
        p = rec.parts[0]
        out.append((p.pose_delta.copy(), p.margin, p.slip_m, rec.tools[0].holding_margin,
                    [s["force_S_N"].copy() for s in p.supports]))
    return out


def test_bit_identical_continuation_across_a_checkpoint():
    loads = [at(CENTROID, [F, 0, 0]) for F in (3000.0, 8800.0, 8800.0, 4000.0, 8900.0, 2000.0)]
    dom, ctx, state = vise_domain()
    straight = _run(dom, ctx, state, loads)
    dom2, ctx2, state2 = vise_domain()
    first = _run(dom2, ctx2, state2, loads[:2])
    arrays, meta = state2.holding.to_arrays()
    import json
    meta = json.loads(json.dumps(meta))                     # the checkpoint writes JSON
    state3 = fx.sim_state()
    state3.holding = HoldingState.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta)
    state3.clock = state2.clock
    dom3 = HoldingDomain(tools={0: fx.tool_data()})          # a new process: runtime rebuilt from the state
    resumed = first + _run_from(dom3, ctx2, state3, loads, 2)
    assert len(resumed) == len(straight)
    for a, b in zip(straight, resumed):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
        for fa, fb in zip(a[4], b[4]):
            np.testing.assert_array_equal(fa, fb)
    assert straight[1][2] > 0.0                              # the scenario slips (the slip pose is exercised)


def _run_from(dom, ctx, state, loads, start):
    out = []
    for k in range(start, len(loads)):
        w = loads[k]
        bus = fx.bus_with(work_mean=0.6 * w, work_peak=w, tool_mean=[0, 0, -100, 0, 0, 5],
                          tool_peak=[0, 0, -300, 0, 0, 20], tick0=k * 10000)
        rec = fx.run(dom, ctx, state, bus)
        p = rec.parts[0]
        out.append((p.pose_delta.copy(), p.margin, p.slip_m, rec.tools[0].holding_margin,
                    [s["force_S_N"].copy() for s in p.supports]))
    return out


def test_a_large_body_released_by_removal_is_the_part_coming_free():
    dom, ctx, state = vise_domain()
    bus = fx.bus_with(work_mean=np.zeros(6))
    bus.removal.loose_bodies = [{"workpiece": 0, "volume_m3": 6e-5, "mass_kg": 0.47, "centroid_S_m": [0.05, 0.03, 0.02],
                                 "tick": 5000, "kind": "part_free"}]
    rec = fx.run(dom, ctx, state, bus)
    assert rec.parts[0].free and "removal" in rec.parts[0].free_cause and rec.parts[0].supports == []
    found = [f for f in state.findings if f.category == Category.PART_FREE]
    assert found and found[0].severity == Severity.ERROR and found[0].tick == 5000
    # a slug is not the part
    dom, ctx, state = vise_domain()
    bus.removal.loose_bodies = [{"workpiece": 0, "volume_m3": 1e-8, "mass_kg": 8e-5, "tick": 5000}]
    assert not fx.run(dom, ctx, state, bus).parts[0].free


def test_table_acceleration_is_an_inertial_load_on_the_part():
    """The table accelerating along x at 5 g: the part sees -m a; with the light 20 kN vise grip the margin is
    the slip capacity over m a."""
    dom, ctx, state = vise_domain()
    mass = 0.1 * 0.06 * 0.03 * 7850.0
    a = 5 * 9.80665
    dt = ctx.ipo_s
    for step in range(3):
        bus = fx.bus_with(work_mean=np.zeros(6), tick0=step * 10000, contact=False)
        frame = bus.motion.paths[0]
        t = (np.arange(10) + 1 + step * 10) * dt
        frame.tip_M_m = np.stack([0.5 * a * t * t, np.zeros(10), np.zeros(10)], axis=1)    # tool fixed in S
        frame.tip_S_m = np.zeros((10, 3))
        rec = fx.run(dom, ctx, state, bus)
    capacity = 2 * 0.2 * 20000.0 + 0.19 * mass * 9.80665
    assert rec.parts[0].margin == pytest.approx(min(capacity / (mass * a), 1000.0), rel=0.1)
    assert rec.parts[0].margin < 1000.0


def test_a_setup_without_workholding_is_noted_not_judged():
    proj = fx.project([])
    ctx = fx.context(proj)
    state = fx.sim_state()
    dom = HoldingDomain(tools={0: fx.tool_data()})
    dom.init(ctx, state)
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=at(CENTROID, [500.0, 0, 0])))
    assert not rec.parts[0].free and rec.parts[0].supports == []
    assert any("no workholding" in n for n in state.holding.notes)
    assert not [f for f in state.findings if f.category in (Category.PART_FREE, Category.PART_SLIP)]


def test_results_document():
    dom, ctx, state = vise_domain()
    fx.run(dom, ctx, state, fx.bus_with(work_mean=at(CENTROID, [8800.0, 0, 0])))
    doc = dom.finalize(ctx, state)
    assert doc["tools"][0]["clamp"]["model"] == "friction"
    assert doc["tools"][0]["unbalance"]["required_grade_m_s"] == 0.0025
    assert doc["setups"] and doc["setups"][0]["max_slip_m"] > 0.0
    assert doc["setups"][0]["summary"]["devices"][0]["type"] == "vise"
    assert any(e["kind"] == "part_motion" for e in doc["events"])
    assert doc["protocols"] == {"geometry": "StockGeometry", "tabs": "NominalTabs", "burrs": "NoBurrs"}
    import json
    json.dumps(doc)                                          # JSON-serialisable


def test_mesh_stock_mass_properties_and_contact_shape():
    from cnctwin.holding.protocols import box_mass_properties, mesh_mass_properties, mesh_shape
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.06, 0.03])
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    # 12 outward triangles of the box (vertex index = 4 ix + 2 iy + iz)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    vol, cen, J = mesh_mass_properties(V, F)
    v_ref, c_ref, J_ref = box_mass_properties(lo, hi)
    assert vol == pytest.approx(v_ref, rel=1e-12)
    np.testing.assert_allclose(cen, c_ref, rtol=1e-12)
    np.testing.assert_allclose(J, J_ref, rtol=1e-10, atol=1e-18)
    assert mesh_shape([-0.0225, -0.0225, 0.0], [0.0225, 0.0225, 0.23]).kind == "cylinder"
    assert mesh_shape([0.0, 0.0, 0.0], [0.1, 0.06, 0.03]).kind == "box"
