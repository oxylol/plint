"""RemovalDomain lifecycle with a fake context and bus: init, kernel_inputs, a simulated hot loop (tooth_pass,
l1_contact), absorb into the bus record and the ledger, macro_step, checkpoint continuation, end_operation
with a flip, finalize, channels, and the protocol views for thermal and chips."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cnctwin.core.abi import EngagedIntervals
from cnctwin.core.bus import CuttingContact, PathFrame
from cnctwin.core.channels import ChannelRegistry
from cnctwin.core.domain import RunContext
from cnctwin.core.findings import Category
from cnctwin.core.state import SimState
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_drive as KD
from cnctwin.removal import profile as P
from cnctwin.removal.domain import RemovalDomain, build_shape
from cnctwin.removal.state import RemovalState, ToolShape

R0 = 4e-3


def _project(n_ops=2):
    op = lambda name, fp: SimpleNamespace(name=name, setup=SimpleNamespace(workholding=[]), stock_pose_W_S=np.eye(4),
                                         from_previous=fp)
    ops = [op("OP10", None)] + [op("OP20", {"flip": {"axis": "X", "angle_deg": 180}})][: n_ops - 1]
    return SimpleNamespace(stock=SimpleNamespace(kind="block", size_m=[20e-3, 16e-3, 8e-3], mesh=None),
                           material=SimpleNamespace(density_kg_m3=2700.0),
                           physics=SimpleNamespace(lattice_m=0.25e-3, refine=[], polar_theta_bins=720, removal=None),
                           operations=ops, machine=SimpleNamespace(type="vertical_mill"), cad=None, tools=None,
                           stock_pose_W_S=np.eye(4))


def _shape():
    prof = P.profile_from_points([0.0, 20e-3], [R0, R0])
    slabs = A.ToothSlabs(np.array([[0.0, 10e-3, 20e-3]]), np.array([[0.0, 0.0]]), np.array([2], np.int32))
    return ToolShape(slot=0, kind=A.TK_ENVELOPE, rotating=True, centre_cutting=True, hand_sign=-1.0, cut=prof,
                     whole=prof, insert=A.empty_insert(), helix=A.empty_helix(), slabs=slabs, radius_m=R0)


def _ctx():
    return RunContext(project=_project(), db=None, run_dir=None, seed=1, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10,
                      channels=ChannelRegistry())


def _hot_loop(inputs, xs, tick0):
    """Stand-in for core.hotloop.run_macro: one tooth pass per x position and one L1 tick each."""
    f = inputs.fields[0]
    prof = inputs.cut[0]
    acc = inputs.accum[0]
    st_z = np.array([1e-3, 3e-3, 5e-3])
    st_r = np.array([R0, R0, R0])
    eng = EngagedIntervals(np.zeros((3, 4)), np.zeros((3, 4)), np.zeros(3, np.int32), np.zeros(1, np.int64))
    spec = A.subtract_spec([0, 0, 1], np.zeros((2, 3)), [0.0, 10e-3, 20e-3], [0.0, 0.0], [0] * 3, [0] * 3)
    gate = np.array([0.0, 1e-9, 0.0, 0.25e-3])
    stats = np.zeros(4)
    w = np.array([0.0, 0.0, 1.0])
    status = 0
    for k, x in enumerate(xs):
        tip = np.array([x, 8e-3, 5e-3])
        spec.ivals[A.SI_TICK] = tick0 + 1000 * k
        spec.ivals[A.SI_EVENT_ID] = inputs.iparams[A.RI_PATH0 + A.RI_PSTRIDE * 1]
        inputs.iparams[A.RI_PATH0 + A.RI_PSTRIDE * 1] += 1
        slab_c = np.ascontiguousarray(np.stack([tip, tip]))
        status |= KD.tooth_pass(f, prof, inputs.slabs[0], 0, slab_c, tip, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                                w, st_z, st_r, np.ones(3, np.int8), eng, np.full(3, 0.5e-3), 1.0, 0.0, gate, spec,
                                inputs.targets[0], acc, inputs.events, inputs.scratch, stats)
        KD.l1_contact(f, prof, tip, w, 0.0, True, tick0 + 1000 * k, 1e-6, acc)
    return status


def test_lifecycle_bus_ledger_checkpoint_and_transfer():
    ctx = _ctx()
    state = SimState()
    dom = RemovalDomain(tool_shapes={0: _shape()}, n_paths=1)
    dom.init(ctx, state)
    rs = state.removal
    v0 = F.volume_m3(rs.workpieces[0].field, 0, 2)
    assert v0 == pytest.approx(20e-3 * 16e-3 * 8e-3)
    from cnctwin.removal import ledger as LG
    v0_ledger = LG.volume_m3(rs.workpieces[0])
    assert v0_ledger == pytest.approx(v0, rel=1e-12)       # the block's faces lie on cell faces
    state.bus_prev.motion.paths = [PathFrame(tool_slot=0, workpiece=0)]
    inputs = dom.kernel_inputs(ctx, state, state.bus_prev)
    assert isinstance(inputs, A.RemovalInputs) and inputs.iparams[A.RI_NPATHS] == 1
    xs = np.arange(-2e-3, 6e-3, 0.1e-3)
    assert _hot_loop(inputs, xs, 0) == 0
    dom.absorb(ctx, state, None, state.bus_next)
    rec = state.bus_next.removal
    c = rec.contacts[0]
    # the booked removal is the field's ledger volume change (removal.md 1.1), exactly
    removed = v0_ledger - LG.volume_m3(rs.workpieces[0])
    assert c.removed_volume_m3 == pytest.approx(removed, rel=1e-12)
    # the slot swept by the last pass at x = 5.9 mm: (5.9 x 8 + pi 4^2 / 2) mm^2 x 3 mm; the ledger volume within
    # 0.05 % of it, the z family 0.23 % (whole columns of the end's semicircle)
    swept = (5.9e-3 * 8e-3 + 0.5 * math.pi * R0 ** 2) * 3e-3
    assert removed == pytest.approx(swept, rel=5e-4)
    assert c.removed_volume_xyz_m3[2] == pytest.approx(v0 - F.volume_m3(rs.workpieces[0].field, 0, 2), rel=1e-12)
    # the core Ledger books the removed volume and mass from this record (removal writes only its record)
    from cnctwin.core.ledger import Ledger
    Ledger(tol_mass=1e-6, tol_energy=1e-3, density_kg_m3=2700.0, domains={"removal"}).book(state, 0)
    assert state.ledger.removed_volume_m3 == pytest.approx(removed, rel=1e-12)
    assert state.ledger.removed_mass_kg == pytest.approx(2700.0 * removed, rel=1e-12)
    assert c.geometry.in_cut and c.geometry.ap_m == pytest.approx(3e-3, abs=1e-9)   # z 5 .. 8 mm of the block
    assert c.geometry.ae_m == pytest.approx(2 * R0, rel=0.02)
    assert c.surface_ids[1] > c.surface_ids[0]
    assert rec.field_version[0] > 0
    # macro steps: the consistency published in step n is that of step n - 1 (cutting's analytic volume of
    # step n - 1 is read from prev)
    dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
    assert state.bus_next.removal.contacts[0].volume_consistency == pytest.approx(1.0)   # nothing before
    state.bus_next.cutting.contacts = [CuttingContact(contact=0, tool_slot=0, analytic_chip_volume_m3=removed)]
    state.swap_bus()
    state.bus_prev.motion.paths = [PathFrame(tool_slot=0, workpiece=0)]
    dom.kernel_inputs(ctx, state, state.bus_prev)
    dom.absorb(ctx, state, None, state.bus_next)
    dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
    assert state.bus_next.removal.contacts[0].volume_consistency == pytest.approx(1.0, rel=1e-12)
    # checkpoint continuation is bit-identical
    arrays, meta = rs.to_arrays()
    arrays = {k: np.array(v) for k, v in arrays.items()}     # the checkpoint file's copy (to_arrays does not copy)
    rs2 = RemovalState.from_arrays(arrays, meta)
    state2 = SimState()
    state2.removal = rs2
    state2.bus_prev.motion.paths = [PathFrame(tool_slot=0, workpiece=0)]
    in1 = dom.kernel_inputs(ctx, state, state.bus_prev)
    in2 = dom.kernel_inputs(ctx, state2, state2.bus_prev)
    _hot_loop(in1, xs + 6e-3, 10 ** 6)
    _hot_loop(in2, xs + 6e-3, 10 ** 6)
    assert F.fields_equal(state.removal.workpieces[0].field, state2.removal.workpieces[0].field)
    dom.absorb(ctx, state, None, state.bus_next)
    # channels
    vals = dom.channel_values(ctx, state, state.bus_next)
    names = {ch.name for ch in dom.channels()}
    assert names == set(vals)
    # end of the first operation: flip about X; volume and material kept
    v_before = F.volume_m3(state.removal.workpieces[0].field, 0, 2)
    l_before = LG.volume_m3(state.removal.workpieces[0])
    state.clock.op_index = 0
    dom.end_operation(ctx, state)
    f2 = state.removal.workpieces[0].field
    assert F.volume_m3(f2, 0, 2) == pytest.approx(v_before, rel=1e-12)
    # the slot was cut from the top (z 5..8 mm): after the flip it is at the bottom
    from cnctwin.removal import k_query as KQ

    h = np.zeros(1)
    kind = np.zeros(1, np.int64)
    KQ.surface_height(f2, 2, -1, np.array([3e-3]), np.array([8e-3]), h, kind)
    assert -h[0] == pytest.approx(3e-3, abs=1e-8)
    out = dom.finalize(ctx, state)
    assert out["workpieces"][0]["volume_z_m3"] == pytest.approx(v_before, rel=1e-12)
    # the flip is a lattice symmetry: dual cells map onto dual cells, the ledger volume is kept exactly
    assert out["workpieces"][0]["volume_m3"] == pytest.approx(l_before, rel=1e-12)


def test_events_become_findings_once_per_episode():
    ctx = _ctx()
    state = SimState()
    dom = RemovalDomain(tool_shapes={0: _shape()}, n_paths=1)
    dom.init(ctx, state)
    state.bus_prev.motion.paths = [PathFrame(tool_slot=0, workpiece=0)]
    for step in range(3):
        inputs = dom.kernel_inputs(ctx, state, state.bus_prev)
        from cnctwin.removal.k_envelope import push_event

        push_event(inputs.events, A.EV_RAPID_INTO_MATERIAL, 100 + step, 0, 1e-3, 2e-3, 3e-3, 0.5e-3, 1e-9, -1, 5, 0,
                   2)
        dom.absorb(ctx, state, None, state.bus_next)
        assert state.bus_next.removal.events[0]["kind"] == "rapid_into_material"
    found = [f for f in state.findings if f.category == Category.RAPID_INTO_MATERIAL]
    assert len(found) == 1


def test_a_rapid_that_only_grazes_the_material_is_no_crash():
    """A rapid whose envelope enters the material by no more than the collision tolerance (a G00 retract lifting the
    tool while a feed axis's servo lag catches up: microns, or 0.000 mm as the finding read before) is traced, not
    reported as rapid_into_material (an error)."""
    ctx = _ctx()
    state = SimState()
    dom = RemovalDomain(tool_shapes={0: _shape()}, n_paths=1)
    dom.init(ctx, state)
    state.bus_prev.motion.paths = [PathFrame(tool_slot=0, workpiece=0)]
    for k, depth in enumerate((0.0, 3e-6)):
        inputs = dom.kernel_inputs(ctx, state, state.bus_prev)
        from cnctwin.removal.k_envelope import push_event

        push_event(inputs.events, A.EV_RAPID_INTO_MATERIAL, 100 + 2000 * k, 0, 1e-3, 2e-3, 3e-3, depth, 1e-12, -1, 5,
                   0, 2)
        dom.absorb(ctx, state, None, state.bus_next)
    assert not [f for f in state.findings if f.category == Category.RAPID_INTO_MATERIAL]


def test_views_follow_the_protocols():
    from cnctwin.chips.protocols import ChipGeometry
    from cnctwin.thermal.protocols import WorkGeometry

    ctx = _ctx()
    state = SimState()
    dom = RemovalDomain(tool_shapes={0: _shape()}, n_paths=1)
    dom.init(ctx, state)
    g = dom.geometry(state)
    assert isinstance(g, WorkGeometry) and isinstance(g, ChipGeometry)
    lo, hi = g.stock_bounds_S(0)
    assert np.allclose(hi - lo, [20e-3, 16e-3, 8e-3])
    frac = g.fill_fraction(0, [0.0, 0.0, 0.0], 4e-3, (5, 4, 2))
    assert np.allclose(frac, 1.0)
    assert g.changed_box_S(0) is not None and g.changed_box_S(0) is None
    h, kind = g.surface_height(0, 2, 1, [5e-3, 30e-3], [5e-3, 5e-3])
    assert h[0] == pytest.approx(8e-3) and list(kind) == [1, 0]
    assert g.version(0) >= 1


def test_build_shape_from_tooling_geometry():
    from cnctwin.tooling import build_tool
    from tests.unit.tooling.fixtures import HOLDER, TOOLS, instance

    geo = build_tool(TOOLS["square_endmill"], HOLDER, instance(runout_um=6, runout_phase_deg=0))
    ts = build_shape(geo, geo.edge, 0, 0.5e-3)
    assert ts.kind == A.TK_ENVELOPE and ts.rotating and ts.centre_cutting
    assert P.radius_at(ts.cut, 5e-3) == pytest.approx(5e-3, rel=1e-9)
    assert ts.whole.info[A.PI_ZMAX] > ts.cut.info[A.PI_ZMAX]
    nt = geo.edge.elem_r_m.shape[0]
    assert ts.slabs.z.shape[0] == nt and np.all(ts.slabs.n >= 1)
    # runout 6 um TIR: the teeth's radius offsets spread by about the TIR
    spread = ts.slabs.dr[:, 0].max() - ts.slabs.dr[:, 0].min()
    assert 1e-6 < spread < 8e-6


def test_polar_overflow_is_replayed_with_grown_inline_segments():
    import math as _m
    from cnctwin.removal import k_polar as KP
    from cnctwin.removal import polar as PO
    from cnctwin.tooling.abi import InsertProfile
    ctx = _ctx()
    state = SimState()
    dom = RemovalDomain(tool_shapes={0: _shape()}, n_paths=1)
    dom.init(ctx, state)
    wp = state.removal.workpieces[0]
    L = 8e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 360, 0.2e-3, 0.4e-3, 6e-3)
    wp.polar = PO.polar_tube(6e-3, 0.0, 0.0, L, n_th, dz, dr, kp=1)
    assert not dom.needs_replay(state)
    # a face groove splits radial rays into two segments: one inline segment overflows
    groove = np.array([[0.0, -0.4e-3], [3e-3, -0.4e-3], [3e-3, 0.0], [0.0, 0.0]])
    reg = PO.insert_region(InsertProfile(groove, np.ones(4, np.int8), np.zeros(2), np.array([0.0])))
    ps = PO.polar_step(L - 2e-3, 2.5e-3, L - 2e-3, 2.5e-3, 0.0, -2 * _m.pi, 1.0, -1.0, 0.0)
    assert KP.polar_cut(wp.polar, reg, ps, 4, A.new_accum(1)) == 1
    assert dom.needs_replay(state)
    dom.prepare_replay(ctx, state)
    assert int(wp.polar.info[A.PI_KP]) == 2 and not dom.needs_replay(state)
    acc = A.new_accum(1)
    assert KP.polar_cut(wp.polar, reg, ps, 4, acc) == 0 and acc.ivals[A.IA_POLAR_Q2] > 0
    assert any("polar inline segments" in n for n in wp.notes)


def test_a_lathe_turns_a_mesh_stock_through_its_polar_field(tmp_path):
    """A mesh stock on a lathe (the lathe-shaft example's bar with its centre hole is a STEP mesh) gets a polar
    field like a cylinder stock does; without one every turning pass cut air."""
    n, r, L = 48, 4e-3, 10e-3
    ang = 2 * math.pi * np.arange(n) / n
    V = np.vstack([np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(n, z)]) for z in (0.0, L)]
                  + [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]])])
    o0, o1 = np.arange(n), n + np.arange(n)
    nx = lambda a: (a - a[0] + 1) % n + a[0]
    Fc = np.vstack([np.column_stack([o0, nx(o0), nx(o1)]), np.column_stack([o0, nx(o1), o1]),
                    np.column_stack([np.full(n, 2 * n + 1), o1, nx(o1)]), np.column_stack([np.full(n, 2 * n), nx(o0), o0])])
    path = tmp_path / "bar.obj"
    path.write_text("".join(f"v {x * 1e3:.9f} {y * 1e3:.9f} {z * 1e3:.9f}\n" for x, y, z in V)
                    + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in Fc))
    ctx = _ctx()
    ctx.project.stock = SimpleNamespace(kind="mesh", size_m=None, mesh=str(path))
    ctx.project.machine = SimpleNamespace(type="lathe")
    state = SimState()
    RemovalDomain(tool_shapes={0: _shape()}, n_paths=1).init(ctx, state)
    wp = state.removal.workpieces[0]
    assert wp.polar is not None
    from cnctwin.removal import polar as PO
    vol_mesh = 0.5 * n * r * r * math.sin(2 * math.pi / n) * L
    assert PO.volume_m3(wp.polar) == pytest.approx(vol_mesh, rel=2e-3)
    assert float(wp.polar.geom[A.PG_Z0]) == 0.0
    assert int(wp.polar.geom[A.PG_NZ]) * float(wp.polar.geom[A.PG_DZ]) >= L


def test_a_slow_tools_pass_spans_the_steps_between_its_tooth_subtractions():
    """A tooth subtracts once per tooth period, which for a slow tool spans several macro steps (the spot drill of
    examples/drill-tap-plate-1045: 36 ms against 10 ms); the pass lasts while cutting reports chips, so its field
    volume is compared with the chips of all its steps (docs/models/removal.md "Volume consistency")."""
    ctx = _ctx()
    state = SimState()
    dom = RemovalDomain(tool_shapes={0: _shape()}, n_paths=1)
    dom.init(ctx, state)
    from cnctwin.removal import ledger as LG
    v0 = LG.volume_m3(state.removal.workpieces[0])

    def step(cut: bool, chips_prev: float):
        state.bus_prev.motion.paths = [PathFrame(tool_slot=0, workpiece=0)]
        state.bus_prev.cutting.contacts = [CuttingContact(contact=0, tool_slot=0, analytic_chip_volume_m3=chips_prev)]
        inputs = dom.kernel_inputs(ctx, state, state.bus_prev)
        if cut:
            _hot_loop(inputs, np.arange(-2e-3, 6e-3, 0.1e-3), int(state.clock.tick))
        dom.absorb(ctx, state, None, state.bus_next)
        dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
        state.swap_bus()
        state.clock.tick += 10000

    step(True, 0.0)                     # one tooth's subtraction in the first step of the cut
    removed = v0 - LG.volume_m3(state.removal.workpieces[0])
    for _ in range(3):                  # the chips of the cut arrive one step late, a third per step
        step(False, removed / 3.0)
    from cnctwin.removal.state import PS_ACTIVE, PS_V_ANALYTIC, PS_V_FIELD
    ps = state.removal.pass_stats[0]
    assert ps[PS_ACTIVE] == 1.0                                   # still in the pass
    assert ps[PS_V_FIELD] == pytest.approx(removed, rel=1e-12)
    assert ps[PS_V_ANALYTIC] == pytest.approx(removed, rel=1e-12)
    step(False, 0.0)                    # no subtraction, no chips: the pass ends, field and chips agree
    assert state.removal.pass_stats[0, PS_ACTIVE] == 0.0
    assert not [f for f in state.findings if f.category == Category.MASS_BALANCE]
