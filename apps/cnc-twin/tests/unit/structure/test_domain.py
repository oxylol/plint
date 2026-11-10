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
from cnctwin.structure.domain import StructureDomain, refresh_due
from cnctwin.structure.protocols import BoxGeometry
from cnctwin.structure.schema import PHYSICS_SCHEMA, REFRESH_LATENCY_MACRO, REFRESH_LATENCY_S, settings
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
    assert settings(None).every_n_macro == 0 and settings(None).interval_s == pytest.approx(0.5)
    assert REFRESH_LATENCY_MACRO == 50 and REFRESH_LATENCY_S == pytest.approx(0.5)
    # the former key: every_n_macro steps of the reference 10 ms (the cadence it was stated for), at any macro step
    assert settings({"every_n_macro": 50}).interval_s == pytest.approx(0.5)
    assert settings({"refresh_interval_ms": 250}).interval_s == pytest.approx(0.25)
    assert settings({"refresh_interval_s": 0.25}).interval_s == pytest.approx(0.25)
    with pytest.raises(InputError):
        settings({"fill_threshold": 1.5})
    with pytest.raises(InputError):
        settings({"work_modes": 2.5})
    ctx = Ctx(None, obj="project", file="p")
    PHYSICS_SCHEMA.check({"max_dof": 20000, "min_cell_mm": 1.5, "every_n_macro": 25}, ("physics", "structure"), None,
                         ctx)
    PHYSICS_SCHEMA.check({"refresh_interval_ms": 500}, ("physics", "structure"), None, ctx)
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


def test_refresh_schedule_is_in_run_time():
    """The workpiece refresh starts and applies at the same run times at every macro step that divides the interval
    (docs/convergence.md 9: counted in macro steps, the thermal expansion at the contact moved with macro_ipo)."""
    tick_s = 1e-6
    starts = {}
    for macro_ms in (2, 5, 10, 20, 50):
        mt = macro_ms * 1000
        starts[macro_ms] = [t for t in range(0, 3_000_001, mt) if refresh_due(t, mt, 0.5, tick_s)]
    assert all(v == [500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000] for v in starts.values())
    # a step that does not divide the interval starts its job at the first step at or after each multiple
    assert [t for t in range(0, 1_100_001, 30_000) if refresh_due(t, 30_000, 0.5, tick_s)] == [510_000, 1_020_000]
    assert not refresh_due(0, 10_000, 0.5, tick_s)


def test_fixed_latency_skip_and_new_setup():
    dom, ctx, state, geo = make()
    b = bus()
    fx.run(dom, ctx, state, b, 1)                           # macro 0: setup, solved at once
    ws = state.structure.works[0]
    v0 = ws.solution.modes.version
    fx.run(dom, ctx, state, b, 4)                           # macros 1-4: nothing due
    geo.remove_box([0.01, 0.008, 0.008], [0.03, 0.022, 0.013])
    fx.run(dom, ctx, state, b, 1)                           # macro 5: a job starts
    # the job applies REFRESH_LATENCY_S of run time after the step it started in (macro 5 starts at tick 50 000)
    assert len(ws.pending) == 1 and ws.pending[0][0] == 50_000 + int(round(REFRESH_LATENCY_S / ctx.tick_s))
    assert ws.solution.modes.version == v0
    fx.run(dom, ctx, state, b, REFRESH_LATENCY_MACRO - 1)   # macros 6-54: published modes unchanged
    assert ws.solution.modes.version == v0
    rec = fx.run(dom, ctx, state, b, 1)                      # macro 55: applied
    assert ws.solution.modes.version > v0 and rec.work_modes[0].version == ws.solution.modes.version
    assert ws.counters.get("skipped", 0) >= 1                # unchanged inputs (geometry, temperatures, supports)
    # holding places the same supports again (a new supports_version): the model is kept (support_change_rel)
    n = len(ws.history)
    fx.run(dom, ctx, state, bus(version=2), 1)
    assert len(ws.history) == n and ws.counters.get("supports_kept", 0) == 1 and ws.supports_version == 2
    # placed again with a materially different support (stiffer contacts): a new setup, solved at once
    b3 = bus(version=3)
    for sup in b3.holding.parts[0].supports:
        sup["stiffness_N_per_m"] = np.asarray(sup["stiffness_N_per_m"], float) * 1.5
    fx.run(dom, ctx, state, b3, 1)
    assert ws.history[-1]["why"] == "setup" and len(ws.history) == n + 1 and not ws.pending


def test_an_unchanged_model_reuses_its_modes_and_solves_only_the_temperatures():
    """A refresh whose model is the latest solution's (the same cells, supports, clamped state: model_key) keeps its
    eigenpairs, clamping and residual-stress displacements and solves the thermal displacement alone, with the held
    system kept between jobs: the same modes bit for bit, the thermal grid of a full computation of the same inputs
    bit for bit. Material that removes cells makes the next refresh a Ritz solve again."""
    from cnctwin.structure import work as WK
    dom, ctx, state, geo = make()
    fx.run(dom, ctx, state, bus(), 5)                         # macro 0: setup; macros 1-4: nothing due
    ws = state.structure.works[0]
    setup = ws.solution
    warm = fx.thermal_grid([0, 0, 0], list(SIZE), 0.002, lambda c: 293.15 + 5.0 + 400.0 * c[..., 0])
    b = fx.bus_with(fx.vise_supports(SIZE), version=1, thermal=warm, slot=0)
    b.holding.tools[0] = ToolHolding(pullout_m=0.0)
    fx.run(dom, ctx, state, b, 1)                             # macro 5: the temperatures changed: a job
    job = ws.latest
    assert job.stats["solve"] == "reuse" and job.stats["model_key"] == setup.stats["model_key"]
    assert np.array_equal(job.lam, setup.lam) and np.array_equal(job.modes.shape, setup.modes.shape)
    assert np.array_equal(job.modes.static_compliance, setup.modes.static_compliance)
    assert np.array_equal(job.displacement.clamp_m, setup.displacement.clamp_m)
    assert np.array_equal(job.displacement.residual_m, setup.displacement.residual_m)
    assert not np.array_equal(job.displacement.thermal_m, setup.displacement.thermal_m)
    full = WK.compute(dom._inputs(ctx, state.structure, ws, b), None, ws.reference)
    assert full.stats["solve"] == "full"
    assert np.array_equal(job.displacement.thermal_m, full.displacement.thermal_m)
    assert np.array_equal(job.displacement.total_m, full.displacement.total_m)
    assert np.allclose(job.modes.freq_Hz, full.modes.freq_Hz, rtol=1e-9, atol=0.0)
    # a pocket removes cells: the model changed, the next refresh is a Ritz solve on the latest modes
    geo.remove_box([0.01, 0.008, 0.006], [0.03, 0.022, 0.013])
    fx.run(dom, ctx, state, b, 5)
    assert ws.latest.stats["solve"] == "ritz" and ws.latest.stats["model_key"] != setup.stats["model_key"]


def test_the_first_step_waits_for_holdings_record_of_the_part():
    """At the run's first macro step holding has not yet published the part's supports (bus_prev is the initial
    bus): the setup waits one step instead of solving the free part and the clamped part one step apart."""
    dom, ctx, state, geo = make()
    empty = fx.bus_with(None, slot=0)
    empty.holding.tools[0] = ToolHolding(pullout_m=0.0)
    rec = fx.run(dom, ctx, state, empty, 1)                   # macro 0: no record of part 0 yet
    ws = state.structure.works[0]
    assert ws.solution is None and 0 not in rec.work_modes and ws.counters["setup_waited"] == 1
    fx.run(dom, ctx, state, bus(), 1)                         # macro 1: holding's supports: the setup, clamped
    assert ws.solution is not None and ws.clamped and ws.history[-1]["why"] == "setup" and len(ws.history) == 1
    # without a record later on (no holding at all) the part is still set up, free on a 3-2-1 support
    dom2, ctx2, state2, _ = make()
    state2.clock.macro = 1
    fx.run(dom2, ctx2, state2, empty, 1)
    assert state2.structure.works[0].solution is not None and not state2.structure.works[0].clamped


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


def test_supports_placed_again_without_a_material_change_keep_the_work_model():
    """holding places the supports again (supports_version) at every rebuild: structure sets its work model up
    again only for a material change (structure settings support_change_rel; a chuck's grip during a spindle
    run-up moved by less than that, examples/lathe-shaft-4140 set the model up 55 times in 2.4 s)."""
    import numpy as np

    from cnctwin.structure import lathe as LA
    from cnctwin.structure import voxel as VX
    from cnctwin.structure.domain import supports_unchanged
    rng = np.random.default_rng(3)
    n = 6
    k = np.einsum("n,ij->nij", rng.uniform(1e8, 1e9, n), np.eye(3))
    a = VX.Supports(rng.uniform(0.0, 0.03, (n, 3)), k, 0.5 * k, 0.1 * k, rng.uniform(100.0, 900.0, (n, 3)))
    b = VX.Supports(a.points + 1e-7, a.k_held * 1.005, a.k_locate.copy(), a.k_locked.copy(), a.preload_N * 1.01)
    lathe_a = LA.LatheSupports({"gripping_force_N": 20000.0, "jaws": 3, "type": "chuck"}, None, [])
    lathe_b = LA.LatheSupports({"gripping_force_N": 19900.0, "jaws": 3, "type": "chuck"}, None, [])
    assert supports_unchanged(a, b, lathe_a, lathe_b, 0.02, 1e-3)
    c = VX.Supports(a.points, a.k_held * 1.05, a.k_locate, a.k_locked, a.preload_N)
    assert not supports_unchanged(a, c, lathe_a, lathe_a, 0.02, 1e-3)
    d = VX.Supports(a.points + 1e-4, a.k_held, a.k_locate, a.k_locked, a.preload_N)
    assert not supports_unchanged(a, d, lathe_a, lathe_a, 0.02, 1e-3)
    lathe_c = LA.LatheSupports({"gripping_force_N": 15000.0, "jaws": 3, "type": "chuck"}, None, [])
    assert not supports_unchanged(a, a, lathe_a, lathe_c, 0.02, 1e-3)
    assert not supports_unchanged(a, VX.Supports.empty(), lathe_a, lathe_a, 0.02, 1e-3)


def test_the_next_part_of_a_batch_gets_its_own_setup():
    """A one-operation batch runs every part as operation 0. After the first part is unclamped (released state
    published), the next part clamped in the same supports gets a new setup at once. Before, the first part's
    released state stayed published for every later part (the released check keyed on the operation only), and with
    the supports placed as before the kept-model rule would have kept the machined first part's modes."""
    dom, ctx, state, geo = make()
    b = bus()
    fx.run(dom, ctx, state, b, 1)                            # part 0: setup
    ws = state.structure.works[0]
    geo.remove_box([0.01, 0.008, 0.008], [0.03, 0.022, 0.013])      # part 0 machined
    dom.end_operation(ctx, state)                            # unclamped: the released state
    assert ws.history[-1]["why"] == "release"
    rec = fx.run(dom, ctx, state, b, 1)
    assert not rec.displacement[0].clamped
    # the next part: the loop counts the part on; holding places the same supports again
    state.clock.part_index = 1
    n = len(ws.history)
    rec = fx.run(dom, ctx, state, bus(version=2), 1)
    assert len(ws.history) == n + 1 and ws.history[-1]["why"] == "setup" and ws.part_index == 1
    assert rec.displacement[0].clamped
    assert ws.counters.get("supports_kept", 0) == 0
