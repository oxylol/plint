"""StructureDomain with a fake bus and state: channels, settings and the physics schema, the published records
and their additive fields, fixture compliance, the fixed refresh latency, skipped jobs, a new setup, tool
refreshes, unclamping, kernel inputs, and bit-identical continuation across a checkpoint."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tests.unit.structure import structfix as fx
from tests.unit.tooling.fixtures import HOLDER, TOOLS
from cnctwin.core.bus import Bus, HoldingFeedback, PartHolding, ToolHolding
from cnctwin.core.channels import Tier
from cnctwin.core.errors import InputError
from cnctwin.db.schema import Ctx
from cnctwin.structure import fixture as FX
from cnctwin.structure import kernels as K
from cnctwin.structure.channels import NAMES, channels
from cnctwin.structure.domain import StructureDomain
from cnctwin.structure.protocols import BoxGeometry
from cnctwin.structure.schema import PHYSICS_SCHEMA, REFRESH_LATENCY_MACRO, settings
from cnctwin.structure.state import StructureState
from cnctwin.tooling import build_tool

SIZE = (0.04, 0.03, 0.012)
PHYS = {"max_dof": 2500, "every_n_macro": 5, "work_modes": 4, "work_residual_modes": 4}


def make(physics=None, slots=1):
    proj = fx.project(stock={"kind": "block", "size_mm": [40, 30, 12], "form": "rolled_plate"},
                      physics=physics or PHYS, machine=fx.MACHINE)
    ctx = fx.context(proj)
    state = fx.sim_state(slots)
    geo = BoxGeometry([0, 0, 0], list(SIZE))
    tool = build_tool(TOOLS["square_endmill"], HOLDER, {"number": 1, "stickout_mm": 35})
    dom = StructureDomain(geometry=geo, tools={0: tool})
    dom.init(ctx, state)
    return dom, ctx, state, geo


def bus(version=1, thermal=True, pullout=0.0, free=False):
    th = fx.thermal_grid([0, 0, 0], list(SIZE), 0.002, lambda c: 293.15 + 5.0 + 0.0 * c[..., 0]) if thermal else None
    b = fx.bus_with(fx.vise_supports(SIZE), version=version, thermal=th, free=free, slot=0)
    b.holding.tools[0] = ToolHolding(pullout_m=pullout)
    return b


def test_channels_settings_and_schema():
    chans = channels()
    assert {c.name for c in chans} == set(NAMES) and all(c.tier == Tier.MACRO and c.domain == "structure"
                                                            for c in chans)
    st = settings({"min_cell_mm": 2.0, "max_dof": 9000, "tool_element_mm": 1.0})
    assert st.min_cell_m == pytest.approx(2e-3) and st.max_dof == 9000 and st.tool_element_m == pytest.approx(1e-3)
    assert settings(None).every_n_macro == 50 and REFRESH_LATENCY_MACRO == 50
    with pytest.raises(InputError):
        settings({"fill_threshold": 1.5})
    with pytest.raises(InputError):
        settings({"work_modes": 2.5})
    ctx = Ctx(None, obj="project", file="p")
    PHYSICS_SCHEMA.check({"max_dof": 20000, "min_cell_mm": 1.5, "every_n_macro": 25}, ("physics", "structure"), None,
                         ctx)
    assert not ctx.problems
    PHYSICS_SCHEMA.check({"no_such_setting": 1}, ("physics", "structure"), None, ctx)
    assert ctx.problems


def test_records_fixture_and_additive_fields():
    dom, ctx, state, geo = make()
    rec = fx.run(dom, ctx, state, bus(), 1)
    assert set(rec.tools) == {0} and set(rec.work_modes) == {0} and set(rec.displacement) == {0}
    tm = rec.tools[0]
    assert tm.freq_Hz[0] < 950 and tm.tip_compliance_m_per_N.shape == (3, 3) and 0.0 <= tm.fit_error < 0.05
    wm, d = rec.work_modes[0], rec.displacement[0]
    st = state.structure
    g = st.works[0].grid
    assert wm.shape.shape == (len(wm.freq_Hz),) + g.shape + (3,) and wm.static_compliance.shape == g.shape + (3, 3)
    assert wm.model == "voxel" and wm.version == 1 and d.clamped and d.version == 1
    assert d.total_m.shape == g.node_shape + (3,)
    assert np.allclose(d.total_m, d.thermal_m + d.clamp_m + d.residual_m)
    assert np.max(np.abs(d.thermal_m)) > 0 and np.max(np.abs(d.clamp_m)) > 0
    assert not np.any(d.residual_m)                         # the material has no residual-stress profile
    K6, c = fx.support_stiffness(fx.vise_supports(SIZE))
    C6 = FX.compliance6(K6)
    assert np.allclose(rec.fixture_compliance_6_m_per_N[0], C6)
    assert np.allclose(rec.fixture_compliance_m_per_N[0], FX.point_compliance(C6, c, np.zeros(3)))
    assert np.allclose(rec.fixture_point_S_m[0], 0.0) and np.allclose(rec.fixture_centroid_S_m[0], c)
    vals = dom.macro_values(state)
    assert set(vals) == set(NAMES) and vals["structure.work_dof"][0] > 0 and vals["structure.tool_f1_Hz"] > 0
    # nothing else on the bus was written
    assert state.bus_next.cutting.contacts == [] and state.bus_next.holding.parts == {}


def test_fixed_latency_skip_and_new_setup():
    dom, ctx, state, geo = make()
    b = bus()
    fx.run(dom, ctx, state, b, 1)                           # macro 0: setup, solved at once
    ws = state.structure.works[0]
    v0 = ws.solution.modes.version
    fx.run(dom, ctx, state, b, 4)                           # macros 1-4: nothing due
    geo.remove_box([0.01, 0.008, 0.008], [0.03, 0.022, 0.013])
    fx.run(dom, ctx, state, b, 1)                           # macro 5: a job starts
    assert len(ws.pending) == 1 and ws.pending[0][0] == 5 + REFRESH_LATENCY_MACRO
    assert ws.solution.modes.version == v0
    fx.run(dom, ctx, state, b, REFRESH_LATENCY_MACRO - 1)   # macros 6-54: published modes unchanged
    assert ws.solution.modes.version == v0
    rec = fx.run(dom, ctx, state, b, 1)                      # macro 55: applied
    assert ws.solution.modes.version > v0 and rec.work_modes[0].version == ws.solution.modes.version
    assert ws.counters.get("skipped", 0) >= 1                # unchanged inputs (geometry, temperatures, supports)
    # holding places the supports again: solved at once
    n = len(ws.history)
    fx.run(dom, ctx, state, bus(version=2), 1)
    assert ws.history[-1]["why"] == "setup" and len(ws.history) == n + 1 and not ws.pending


def test_tool_refresh_on_pullout():
    dom, ctx, state, geo = make()
    fx.run(dom, ctx, state, bus(), 1)
    v0 = state.structure.tools[0].modes.version
    fx.run(dom, ctx, state, bus(pullout=1e-6), 1)            # below pullout_refresh (20 um)
    assert state.structure.tools[0].modes.version == v0
    rec = fx.run(dom, ctx, state, bus(pullout=2e-4), 1)
    assert rec.tools[0].version == v0 + 1 and state.structure.tools[0].pullout_m == pytest.approx(2e-4)


def test_unclamping_and_kernel_inputs():
    dom, ctx, state, geo = make()
    b = bus()
    fx.run(dom, ctx, state, b, 1)
    ki = dom.kernel_inputs(ctx, state, state.bus_next)
    d = state.structure.works[0].solution.displacement
    assert np.array_equal(ki.total_m[0], d.total_m) and ki.cell_m[0] == d.cell_m
    x = np.array([[0.02, 0.015, 0.012]])
    X = np.zeros((1, 3))
    K.material_points(ki.origin_S_m[0], float(ki.cell_m[0]), ki.total_m[0], x, 1, X)
    dx = np.zeros((1, 3))
    K.sample_node_grid(ki.origin_S_m[0], float(ki.cell_m[0]), ki.total_m[0], x, dx)
    assert np.allclose(X, x - dx)
    dom.end_operation(ctx, state)
    rec = fx.run(dom, ctx, state, b, 1)
    d = rec.displacement[0]
    assert not d.clamped and not np.any(d.clamp_m)
    unload = state.structure.works[0].unload
    assert len(unload) == 1 and unload[0]["thermal_max_m"] > 0
    out = dom.finalize(ctx, state)
    json.dumps(out, allow_nan=False)
    assert out["tools"][0]["freq_Hz"][0] == pytest.approx(rec.tools[0].freq_Hz[0])


def _snapshot(rec):
    return (rec.version, rec.tools[0].freq_Hz.copy(), rec.work_modes[0].shape.copy(),
            rec.work_modes[0].static_compliance.copy(), rec.displacement[0].total_m.copy())


def test_checkpoint_continuation_is_bit_identical():
    dom, ctx, state, geo = make()
    b = bus()
    fx.run(dom, ctx, state, b, 3)
    geo.remove_box([0.01, 0.008, 0.008], [0.03, 0.022, 0.013])
    fx.run(dom, ctx, state, b, 3)                            # a job is pending (started at macro 5)
    assert state.structure.works[0].pending
    arrays, meta = state.structure.to_arrays()
    json.dumps(meta, allow_nan=False)
    clone_state = copy.copy(state)
    clone_state.clock = copy.deepcopy(state.clock)
    clone_state.structure = StructureState.from_arrays({k: np.array(v) for k, v in arrays.items()},
                                                       json.loads(json.dumps(meta)))
    clone_dom = StructureDomain(geometry=geo, tools=dom.tools)
    steps = REFRESH_LATENCY_MACRO
    a = _snapshot(fx.run(dom, ctx, state, b, steps))
    c = _snapshot(fx.run(clone_dom, ctx, clone_state, b, steps))
    assert a[0] == c[0]
    for x, y in zip(a[1:], c[1:]):
        assert np.array_equal(x, y)
    assert state.structure.works[0].solution.modes.version == clone_state.structure.works[0].solution.modes.version


def test_free_part_uses_a_minimal_support():
    dom, ctx, state, geo = make()
    rec = fx.run(dom, ctx, state, bus(free=True), 1)
    ws = state.structure.works[0]
    assert not ws.clamped and not rec.displacement[0].clamped and not np.any(rec.displacement[0].clamp_m)
    assert rec.work_modes[0].freq_Hz[0] > 0


def test_residual_stress_release_through_the_domain():
    """The as-supplied plate is undistorted (d_residual = 0 before any cut); a pocket releases its stress, and the
    released part (end of operation, 3-2-1 support) carries the distortion."""
    profile = [[0.0, -40.0], [0.25, 20.0], [0.5, 30.0], [0.75, 20.0], [1.0, -40.0]]
    proj = fx.project(stock={"kind": "block", "size_mm": [40, 30, 12], "form": "rolled_plate"}, physics=PHYS,
                      mat=fx.material(profile=profile, thickness_mm=12.0))
    ctx = fx.context(proj)
    state = fx.sim_state(0)
    geo = BoxGeometry([0, 0, 0], list(SIZE))
    dom = StructureDomain(geometry=geo)
    dom.init(ctx, state)
    b = bus()
    rec = fx.run(dom, ctx, state, b, 1)
    assert np.max(np.abs(rec.displacement[0].residual_m)) < 1e-15
    geo.remove_box([0.004, 0.004, 0.006], [0.036, 0.026, 0.013])
    dom.end_operation(ctx, state)
    rec = fx.run(dom, ctx, state, b, 1)
    d = rec.displacement[0]
    assert not d.clamped and np.max(np.abs(d.residual_m)) > 1e-8
    assert state.structure.works[0].unload[0]["residual_max_m"] > 1e-8
