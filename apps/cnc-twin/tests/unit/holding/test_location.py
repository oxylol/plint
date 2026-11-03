"""Location error: a chip under one locator of a 3-2-1 setup against the rigid-body hand calculation, a chip
under a vise parallel (the part not seated), and the hand-over with chips: the next setup's locating faces in S
(LocatorFaces) and the debris chips measures on them moving the part in that setup."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.holding import holdfix as fx
from cnctwin.core.findings import Category, Severity
from cnctwin.holding import setup as SU
from cnctwin.holding.domain import HoldingDomain
from cnctwin.holding.locators import HoldingLocators
from cnctwin.holding.protocols import NominalTabs, StockGeometry
from cnctwin.holding.schema import settings

H = 5e-5            # a 0.05 mm chip


def plate_321():
    """Block 100 x 60 x 30 on a plate (primary), two pins on the -x face at y = 15 and 45 (secondary), a stop on
    the -y face at x = 50 (tertiary), clamps down and toward the locators."""
    return [fx.entry(fx.PLATE, translate_mm=(50, 30, 0)),
            fx.entry(fx.PLATE, translate_mm=(0, 15, 15), label="locator x1"),
            fx.entry(fx.PLATE, translate_mm=(0, 45, 15), label="locator x2"),
            fx.entry(fx.PLATE, translate_mm=(50, 0, 15), rotate_deg=(0, 0, 90), label="stop y"),
            fx.entry(fx.CLAMP, translate_mm=(30, 30, 30)), fx.entry(fx.CLAMP, translate_mm=(70, 30, 30)),
            fx.entry(fx.CLAMP, translate_mm=(100, 30, 15), rotate_deg=(0, 90, 0)),
            fx.entry(fx.CLAMP, translate_mm=(50, 60, 15), rotate_deg=(-90, 0, 0))]


def build(entries, debris=None):
    proj = fx.project(entries)
    geo = StockGeometry.from_stock(proj.stock)
    return SU.build_setup(proj, 0, geo, NominalTabs(), settings(None), debris=debris or {}), proj


def test_chip_under_one_locator_rotates_the_part_about_the_other():
    """The -x face rests on x1 (raised by h) and x2: it turns by h / 30 mm about z, the centre of mass (y = 30,
    half way) moves h / 2 along x; the y stop sits at x = 50 under the centre: no y motion."""
    s0, _ = build(plate_321())
    main0 = s0.group("main")
    assert main0.held and main0.seated
    np.testing.assert_allclose(main0.location, 0.0, atol=1e-15)
    roles = sorted(p.role for p in main0.patches)
    assert roles.count("locator") == 3 and "plate" in roles and roles.count("clamp") == 4
    s1, _ = build(plate_321(), {"OP10/1:locator x1": H})
    loc = s1.group("main").location
    assert loc[0] == pytest.approx(H / 2, rel=1e-2)
    assert loc[5] == pytest.approx(H / 0.030, rel=1e-2)
    assert abs(loc[1]) < 1e-2 * H and abs(loc[2]) < 1e-2 * H
    assert abs(loc[3]) < 1e-2 * H / 0.03 and abs(loc[4]) < 1e-2 * H / 0.03
    assert s1.group("main").debris_node == {"OP10/1:locator x1": (0, H)}


def test_the_domain_reports_the_mislocation_with_its_cause():
    proj = fx.project(plate_321())
    ctx = fx.context(proj)
    state = fx.sim_state()
    dom = HoldingDomain(tools={0: fx.tool_data()})
    dom.init(ctx, state)
    bus = fx.bus_with(work_mean=np.zeros(6))
    bus.chips.locator_debris_m = {"OP10/1:locator x1": H}
    rec = fx.run(dom, ctx, state, bus)
    p = rec.parts[0]
    np.testing.assert_allclose(p.location_error_S_m, [H / 2, 0.0, 0.0], rtol=1e-2, atol=1e-2 * H)
    np.testing.assert_allclose(p.pose_delta, p.slip_pose_delta @ p.location_pose_delta)
    assert p.location_pose_delta[1, 0] == pytest.approx(H / 0.030, rel=1e-2)      # rotation about z
    found = [f for f in state.findings if f.category == Category.MISLOCATION]
    assert found and found[0].severity == Severity.ERROR        # 25 um + L_c x 1.7 mrad = 0.125 mm > 0.05 mm
    assert found[0].quantities["location_error_m"] == pytest.approx(H / 2 + 0.06021 * H / 0.03, rel=2e-2)
    events = [ctx.trace.get(i) for i in found[0].cause_chain]
    assert any(e.event == "holding:debris_under_locator" for e in events)


def test_chip_under_a_vise_parallel_tilts_the_part_off_its_seat():
    entry = fx.entry(fx.VISE, translate_mm=(50, 0, 0), input_torque_Nm=80, parallels_mm=10)
    s0, _ = build([entry])
    assert s0.group("main").seated
    s1, _ = build([entry], {"OP10/0:test-vise/parallel_fixed": 1e-4})
    g = s1.group("main")
    assert not g.seated and g.seat_gap_m > 5e-6
    assert np.linalg.norm(g.location[3:]) > 0.0 and g.location[2] > 0.0
    node, h = g.debris_node["OP10/0:test-vise/parallel_fixed"]
    assert h == 1e-4 and 0 <= node < 25
    # the chip sits under the patch node whose lift moves the part most: no other node gives a larger error
    l_c = g.body.l_c_m

    def err(location):
        return float(np.linalg.norm(location[:3]) + l_c * np.linalg.norm(location[3:]))

    chosen = err(g.location)
    for corner in (0, 4, 20, 24):
        proj = fx.project([entry])
        geo = StockGeometry.from_stock(proj.stock)
        trial = SU.build_setup(proj, 0, geo, NominalTabs(), settings(None))
        tg = trial.group("main")
        SU.assemble(tg, {}, {}, set())                 # back to the assembled (unclamped) contact set
        gap = np.zeros(tg.n)
        pi = [i for i, p in enumerate(tg.patches) if p.id == "OP10/0:test-vise/parallel_fixed"][0]
        gap[np.flatnonzero(tg.node_patch == pi)[corner]] = h
        base = SU.clamp_sequence(tg, np.zeros(tg.n), tg.w_ref, tg.cfg)[1]
        moved = SU.clamp_sequence(tg, gap, tg.w_ref, tg.cfg)[1]
        assert err(moved - base) <= chosen * (1.0 + 1e-6)


def test_next_setup_faces_in_S_and_debris_hand_over_between_operations():
    vise = fx.entry(fx.VISE, translate_mm=(50, 0, 0), input_torque_Nm=80, parallels_mm=10)
    proj = fx.project([[vise], plate_321()], ops=2)
    geo = StockGeometry.from_stock(proj.stock)
    faces = HoldingLocators(proj, geo, settings(None)).faces_S(0)
    ids = [f[0] for f in faces]
    assert "OP20/1:locator x1" in ids and all(i.startswith("OP20/") for i in ids)
    x1 = faces[ids.index("OP20/1:locator x1")]
    np.testing.assert_allclose(x1[1], [0.0, 0.015, 0.015], atol=1e-12)       # on the -x face
    np.testing.assert_allclose(x1[2], [-1.0, 0.0, 0.0], atol=1e-12)          # outward normal of that face
    assert x1[3].shape == (2,)
    plate = [f for f in faces if f[0].endswith("/plate")][0]
    np.testing.assert_allclose(plate[3], [0.05, 0.03], atol=1e-12)           # half extents along S x, y
    assert HoldingLocators(proj, geo, settings(None)).faces_S(1) == []
    # the run: operation 1 in the vise, chips leave a chip on x1, operation 2 is mislocated by it
    ctx = fx.context(proj)
    state = fx.sim_state()
    dom = HoldingDomain(tools={0: fx.tool_data()})
    dom.init(ctx, state)
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6)))
    assert np.allclose(rec.parts[0].location_error_S_m, 0.0)
    dom.end_operation(ctx, state)
    state.clock.op_index = 1
    bus = fx.bus_with(work_mean=np.zeros(6), tick0=10000)
    bus.chips.locator_debris_m = {"OP20/1:locator x1": H}
    rec = fx.run(dom, ctx, state, bus)
    np.testing.assert_allclose(rec.parts[0].location_error_S_m, [H / 2, 0.0, 0.0], rtol=1e-2, atol=1e-2 * H)
    summary = dom.finalize(ctx, state)
    assert [s["op_index"] for s in summary["setups"]] == [0, 1]
