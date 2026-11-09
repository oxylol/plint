"""Recut round trip: chips lying in a pocket appear in the frozen occupancy grid, the hot loop's recut rows
remove them and they come back as fragments the next step, mass conserved; tools sweeping through a pile
throw the chips out."""

from __future__ import annotations

import math

import numpy as np

from cnctwin.chips import abi, kernels
from cnctwin.chips.domain import ChipsDomain
from cnctwin.core.bus import CuttingContact, CuttingFrame
from cnctwin.core.findings import Category

import chipsfix as fx


def _setup():
    proj = fx.project()
    ctx = fx.context(proj)
    state = fx.sim_state(1)
    geometry = fx.scene_box()
    geometry.remove_box((0.02, 0.02, 0.015), (0.08, 0.06, 0.031))
    dom = ChipsDomain(formation=fx.FakeFormation(), geometry=geometry)
    dom.init(ctx, state)
    cs = state.chips
    H = cs.pile
    # chips settled on the pocket floor around (0.05, 0.04): cells of 2 mm, grid origin at -20 mm
    for iu in range(33, 38):
        for iv in range(28, 33):
            kernels.deposit(H, dom.model, iu, iv, 1e-5, 1e-5 * 1.2e5, 1.2e-3, 1.5e-4, 2e-3, 3e-3, 62)
    cs.emitted[0] += 25e-5
    cs.emitted[1] += 25e-5 * 1.2e5
    return dom, ctx, state


def _step(dom, ctx, state, step, tip, cutting=None, removal=None):
    """The hot loop of window ``step`` wrote these records (bus_prev at the chips macro step)."""
    state.bus_prev.motion = fx.motion(step, tip_M=tip, omega=-1047.0)
    state.bus_prev.cutting = cutting or CuttingFrame()
    if removal is not None:
        state.bus_prev.removal = removal
    dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
    state.swap_bus()


def test_occupancy_grid_holds_the_settled_chips():
    dom, ctx, state = _setup()
    # the tool hovers 2 mm above the chips: the grid covers what it can sweep before the next grid
    _step(dom, ctx, state, 0, [0.05, 0.04, 0.020])
    rec = state.bus_prev.chips
    assert math.isclose(rec.occupancy_mass_kg.sum(), state.chips.pile.mass_kg.sum(), rel_tol=1e-12)
    assert rec.occupancy_cell_m == state.chips.pile.cell_m[0]
    # voxel of (0.05, 0.04, floor 0.015 + half a pile layer)
    point = np.array([0.0705 - 0.02, 0.0605 - 0.02, 0.0151])
    idx = np.floor((point - rec.occupancy_origin_S_m) / rec.occupancy_cell_m).astype(int)
    assert rec.occupancy_mass_kg[tuple(idx)] > 0
    # the hot loop's kernel input is the same frozen grid
    ki = dom.kernel_inputs(ctx, state, state.bus_prev)
    assert np.array_equal(ki.occupancy.mass_kg, rec.occupancy_mass_kg)
    assert abi.is_contiguous_layout(ki.occupancy) and abi.is_contiguous_layout(ki.packing)
    # far from any tool the grid is empty (nothing can be recut there in the next step)
    _step(dom, ctx, state, 1, [0.09, 0.07, 0.10])
    assert state.bus_prev.chips.occupancy_mass_kg.size == 0


def test_the_occupancy_grid_is_bounded_by_what_the_machine_can_sweep():
    """A record's feed velocity faster than the machine can move the tool (a tool change's tip jump booked as one
    tick's velocity was 102 m/s, docs/convergence.md 9 item 10) does not widen the frozen grid: the sweep over the
    lag is at most the machine's top tool speed (scene.tool_speed_max)."""
    dom, ctx, state = _setup()
    dom.tool_speed_max = 1.0                                  # m/s, a VMC's three rapids together
    state.bus_prev.motion = fx.motion(0, tip_M=[0.05, 0.04, 0.020], omega=-1047.0, feed=(0.0, 0.0, 10.0))
    dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
    rec = state.bus_next.chips
    assert rec.occupancy_mass_kg.size > 0                     # the chips under the tool are in it
    cell = float(rec.occupancy_cell_m)
    tool = dom.tools.tool(0)
    lag = dom.settings.slow_lag_steps * ctx.macro_s
    reach = tool.radius_m + dom.tool_speed_max * lag + cell
    extent = 2.0 * reach + tool.flute_length_m + tool.holder_length_m + 2.0 * cell
    assert all(n * cell <= extent + 1e-12 for n in rec.occupancy_mass_kg.shape)


def test_tool_speed_max_of_the_machine_files():
    from cnctwin.chips.scene import tool_speed_max
    from cnctwin.db import Database
    db = Database.load()
    vf2 = db.get("machine", "haas-vf2ss")                   # three 35.6 m/min rapids
    assert math.isclose(tool_speed_max(vf2), math.sqrt(3.0) * 35.6 / 60.0, rel_tol=1e-12)
    umc = db.get("machine", "haas-umc750")                  # plus B and C turning the far end of the envelope
    assert tool_speed_max(umc) > math.sqrt(3.0) * float(umc.axes["X"].rapid_m_s)
    assert math.isinf(tool_speed_max(None))


def test_recut_rows_remove_chips_and_re_emit_fragments():
    dom, ctx, state = _setup()
    cs = state.chips
    _step(dom, ctx, state, 0, [0.05, 0.04, 0.10])
    before = float(cs.pile.mass_kg.sum())
    recut_mass = 3e-6
    rows = np.array([[15000, 0, 0, recut_mass, 0.05, 0.04]])        # tick, tooth, elem, kg, x_M, y_M
    cutting = CuttingFrame(contacts=[CuttingContact(contact=0, tool_slot=0, recut=rows)])
    _step(dom, ctx, state, 1, [0.05, 0.04, 0.10], cutting=cutting)
    record = state.bus_prev.chips
    assert math.isclose(record.recut_mass_kg, recut_mass, rel_tol=1e-12)
    terms = cs.mass_terms()
    assert math.isclose(sum(terms.values()), cs.emitted[0], rel_tol=1e-12)
    assert cs.pile.mass_kg.sum() < before
    # the fragments flew (or already landed again): their mass is no longer in the recut cell
    P = cs.particles
    n = int(P.n[0])
    assert cs.next_pid >= 1 and (terms["in_flight_kg"] > 0 or cs.pile.mass_kg.sum() + cs.pan[0] >= before - 1e-18)
    assert record.occupancy_mass_kg.sum() <= before - recut_mass + record.in_flight_kg + 1e-15
    del n


def test_recut_of_chips_that_are_gone_is_counted_as_missing():
    dom, ctx, state = _setup()
    cs = state.chips
    rows = np.array([[5000, 0, 0, 1e-6, 0.0, 0.0]])                # far from any chips
    state.bus_prev.cutting = CuttingFrame(contacts=[CuttingContact(contact=0, tool_slot=0, recut=rows)])
    state.bus_prev.motion = fx.motion(0, tip_M=[0.0, 0.0, 0.1], omega=-1047.0)
    dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
    assert math.isclose(cs.recut_missing_kg, 1e-6, rel_tol=1e-12)


def test_tool_sweeping_through_the_pile_throws_chips_and_reports_nested_chips():
    dom, ctx, state = _setup()
    cs = state.chips
    before = float(cs.pile.mass_kg.sum())
    # the tool enters the pocket where the chips lie (first contact of the slot)
    _step(dom, ctx, state, 0, [0.05, 0.04, 0.0152], removal=fx.removal(1e-9, point=[0.05, 0.04, 0.0155]))
    _step(dom, ctx, state, 1, [0.05, 0.04, 0.0152], removal=fx.removal(1e-9, point=[0.05, 0.04, 0.0155]))
    assert cs.pile.mass_kg.sum() < before
    nested = [f for f in state.findings if f.category == Category.CHIP_RECUTTING]
    assert nested and "enters" in nested[0].message
    assert math.isclose(sum(cs.mass_terms().values()), cs.emitted[0], rel_tol=1e-9)
