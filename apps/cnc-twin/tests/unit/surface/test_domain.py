"""SurfaceDomain with a fake bus and state: the Domain protocol, settings and schema, channels, the checkpoint
round trip (bit-identical continuation), determinism, features against roughness specifications and their
findings, the surface.json document, per-brick attribute arrays, and the bus records it leaves alone."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tests.unit.surface import surffix as fx
from cnctwin.core.bus import Bus
from cnctwin.core.channels import Tier
from cnctwin.core.domain import Domain
from cnctwin.core.errors import InputError
from cnctwin.core.findings import Category
from cnctwin.core.state import SimState
from cnctwin.surface.channels import NAMES, channels
from cnctwin.surface.domain import SurfaceDomain
from cnctwin.surface.protocols import LatticeBricks, parse_brick_key
from cnctwin.surface.schema import PHYSICS_SCHEMA, settings
from cnctwin.surface.state import SurfaceState

R = 5e-3


def _floor_tool():
    edge, outline = fx.endmill(R, 4)
    return edge, outline, list(range(9))


def test_the_domain_satisfies_the_protocol_and_writes_no_bus_record():
    dom = SurfaceDomain()
    assert isinstance(dom, Domain) and dom.name == "surface"
    assert dom.kernel_inputs(None, None, None) is None and dom.absorb(None, None, None, None) is None
    edge, outline = fx.endmill(R, 4)
    ctx = fx.context(fx.project())
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    ev, v = fx.side_wall(R, 4, 0.05e-3, -1047.2, 20, [12], edge)
    tk = fx.step_ticks(0, 10 * fx.IPO)
    fr = fx.frame(tk, np.stack([v * (tk - 1000) * fx.TICK_S, 0 * tk, 0 * tk], axis=1), -1047.2)
    nxt = Bus()
    before = json.dumps({k: repr(getattr(nxt, k)) for k in Bus.WRITERS})
    dom.macro_step(ctx, state, fx.bus([fr], [fx.contact(ev[ev[:, 0] <= 10000])]), nxt)
    assert json.dumps({k: repr(getattr(nxt, k)) for k in Bus.WRITERS}) == before
    assert "surface" not in Bus.WRITERS


def test_settings_schema_and_ranges():
    st = settings(None)
    assert st.face_angle_rad == pytest.approx(math.radians(10.0)) and st.lines_per_pass == 1
    raw = settings({"face_angle_deg": 20, "burr_limit_mm": 0.05, "mark_limit_um": 5, "dwell_speed_mm_min": 2})
    assert raw.face_angle_rad == pytest.approx(math.radians(20.0)) and raw.burr_limit_m == pytest.approx(5e-5)
    assert raw.mark_limit_m == pytest.approx(5e-6) and raw.dwell_speed_m_s == pytest.approx(2e-3 / 60.0)
    with pytest.raises(InputError):
        settings({"face_angle_deg": 80})
    from cnctwin.db.schema import Struct
    assert isinstance(PHYSICS_SCHEMA, Struct) and "burr_limit_mm" in PHYSICS_SCHEMA.fields
    from cnctwin.project.schema import DomainPhysics
    assert DomainPhysics("surface").declared() is PHYSICS_SCHEMA


def test_channels_and_values():
    chans = channels()
    assert [c.name for c in chans] == list(NAMES) and all(c.tier == Tier.MACRO and c.domain == "surface" for c in chans)
    edge, outline, elems = _floor_tool()
    dom, ctx, state = fx.run_floor(edge, outline, elems)
    values = dom.channel_values(ctx, state)
    assert set(values) == set(NAMES)
    assert values["surface.patches"] >= 1 and math.isfinite(values["surface.last_Ra_m"])


def test_checkpoint_round_trip_continues_bit_identically():
    edge, outline, elems = _floor_tool()
    passes = [(0.0, 0.0, 0.0), (6e-3, -2e-6, 0.5e-6)]
    ref_dom, ref_ctx, ref_state = fx.run_floor(edge, outline, elems, passes=passes, teeth_passes=80)
    reference = json.dumps(ref_dom.finalize(ref_ctx, ref_state), sort_keys=True)
    # run part way, checkpoint, restore into a fresh state and domain, continue
    dom, ctx, state = fx.run_floor(edge, outline, elems, passes=passes, teeth_passes=80, stop_after=(0, 7))
    arrays, meta = state.surface.to_arrays()
    meta = json.loads(json.dumps(meta))
    restored = SurfaceState.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta)
    a2, m2 = restored.to_arrays()
    assert set(a2) == set(arrays) and all(np.array_equal(a2[k], arrays[k]) for k in arrays)
    assert json.dumps(m2, sort_keys=True) == json.dumps(meta, sort_keys=True)
    state2 = SimState()
    state2.tools = state.tools
    state2.surface = restored
    dom2 = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom2, _, state2 = fx.run_floor(edge, outline, elems, passes=passes, teeth_passes=80, stop_after=(7, 10 ** 9),
                                   dom=dom2, ctx=ctx, state=state2)
    assert json.dumps(dom2.finalize(ctx, state2), sort_keys=True) == reference


def test_store_reads_a_checkpoint_written_before_the_pass_hmin_column():
    from cnctwin.surface.store import QF, QF_NAMES, Store
    st = Store()
    q = st.new_pass(0, 0, 0, 100, 1, 0)
    st.qf[q, QF["count"]] = 3.0
    st.qf[q, QF["hmin_sum"]] = 9e-6
    arrays, meta = st.to_arrays()
    arrays["store.qf"] = arrays["store.qf"][:, :QF["hmin_sum"]]          # the layout before the column existed
    old = Store.from_arrays(arrays, meta)
    assert old.qf.shape == (1, len(QF_NAMES))
    assert old.qf[0, QF["count"]] == 3.0 and old.qf[0, QF["hmin_sum"]] == 0.0


def test_identical_runs_give_identical_documents():
    edge, outline, elems = _floor_tool()
    docs = []
    for _ in range(2):
        dom, ctx, state = fx.run_floor(edge, outline, elems)
        docs.append(json.dumps(dom.finalize(ctx, state), sort_keys=True))
    assert docs[0] == docs[1]


def test_feature_against_its_specification_and_the_finish_finding():
    edge, outline, elems = _floor_tool()
    tol = [{"id": "floor-finish", "kind": "roughness", "Ra_um": 0.01,
            "feature": {"face": "plane", "normal": [0, 0, 1], "near_mm": [1.0, 0.0, 0.0]}},
           {"id": "loose", "kind": "roughness", "Ra_um": 10.0,
            "feature": {"face": "plane", "normal": [0, 0, 1], "near_mm": [1.0, 0.0, 0.0]}},
           {"id": "elsewhere", "kind": "roughness", "Rz_um": 1.0,
            "feature": {"face": "plane", "normal": [1, 0, 0], "near_mm": [30.0, 0.0, 0.0]}}]
    dom, ctx, state = fx.run_floor(edge, outline, elems, tolerances=tol)
    doc = dom.finalize(ctx, state)
    feats = {f["id"]: f for f in doc["features"]}
    assert feats["floor-finish"]["pass"] is False and feats["loose"]["pass"] is True
    assert feats["elsewhere"]["Ra_m"] is None and feats["elsewhere"]["pass"] is None
    f = feats["floor-finish"]
    for key in ("id", "op", "tool", "lines", "Ra_m", "Rz_m", "spec_Ra_m", "spec_Rz_m", "pass"):
        assert key in f
    assert f["tool"] == 1 and f["op"] == "OP10" and f["lines"] == [10]
    assert f["spec_Ra_m"] == pytest.approx(1e-8) and f["lay"] in ("parallel", "multidirectional")
    found = [x for x in state.findings if x.category == Category.FINISH_OUT_OF_SPEC]
    assert len(found) == 1 and "floor-finish" in found[0].message and found[0].line == 10
    assert doc["format"] == "cnctwin-surface" and doc["summary"]["patches"] >= 1
    assert isinstance(doc["patches"], list) and doc["faces"] == []


def test_events_without_a_motion_frame_are_counted_not_profiled():
    edge, outline = fx.endmill(R, 4)
    ctx = fx.context(fx.project())
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    ev, _ = fx.side_wall(R, 4, 0.05e-3, -1047.2, 20, [12], edge)
    b = fx.bus([fx.frame(fx.step_ticks(0, 10000), np.zeros((10, 3)), -1047.2, slot=5)], [fx.contact(ev)])
    dom.macro_step(ctx, state, b, Bus())
    c = state.surface.counters
    assert c["events"] == len(ev) and c["unprofiled"] == len(ev) and c["profiled"] == 0
    assert state.surface.store.n_patch == 1


def test_brick_attributes_are_written_and_merged(tmp_path):
    edge, outline, elems = _floor_tool()
    bricks = LatticeBricks.for_stock("block", [0.06, 0.04, 0.03], 2.5e-4)
    key = bricks.brick_key(0, int(bricks.brick_of(0, np.array([[1e-3, 0.0, 0.0]]))[0]))
    # a brick mesh: two triangles on the floor (normal +z) and one on a side face (normal +x)
    V = np.array([[0.0, -1e-3, 0.0], [2e-3, -1e-3, 0.0], [2e-3, 1e-3, 0.0], [0.0, 1e-3, 0.0],
                  [3e-3, 0.0, 1e-3], [3e-3, 1e-3, 1e-3], [3e-3, 0.0, 2e-3]])
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])

    class Meshes:
        def final_bricks(self):
            return {key: (V, F)}

    folder = tmp_path / "keyframes" / "attr"
    folder.mkdir(parents=True)
    np.savez(folder / f"brick_{key}.npz", deviation_m=np.arange(len(V), dtype=np.float32))
    dom, ctx, state = fx.run_floor(edge, outline, elems, run_dir=tmp_path, meshes=Meshes())
    doc = dom.finalize(ctx, state)
    assert doc["attributes"] == [f"keyframes/attr/brick_{key}.npz"]
    assert (tmp_path / "surface.json").is_file()
    with np.load(folder / f"brick_{key}.npz") as arrays:
        assert np.array_equal(arrays["deviation_m"], np.arange(len(V), dtype=np.float32))   # kept
        ra = arrays["Ra_m"]
        assert np.all(np.isfinite(ra[:4])) and np.all(np.isnan(ra[4:]))
        assert ra[0] == pytest.approx(state.surface.values["last_Ra_m"], rel=0.5)
    assert parse_brick_key("w2.17") == (2, 17) and parse_brick_key("5") == (0, 5)


def test_a_batchs_next_part_opens_its_own_patches():
    """The same wall finished on two blanks one after the other (a batch: the next part's OP10 in the same place,
    the domain's operation count one further): the second part's events open their own patches, so each part's
    step-over profile holds its own passes. Before, the second part joined the first part's patches and the two
    parts' tracks were compared as overlapping passes (witness marks on every second part of a batch)."""
    from cnctwin.surface.store import PI, QI
    edge, outline = fx.endmill(R, 4)
    omega = -2.0 * math.pi * 10000.0 / 60.0
    ev, v = fx.side_wall(R, 4, 0.05e-3, omega, 200, [12], edge, edge_r=0.0)
    ctx = fx.context(fx.project())
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    step = 10 * fx.IPO
    offset = 0
    for part in range(2):
        rows_all = ev.copy()
        rows_all[:, 0] += offset
        for t0, rows in fx.chunks(rows_all, step):
            tk = fx.step_ticks(t0, step)
            tt = (tk - 1000 - offset) * fx.TICK_S
            tip = np.stack([v * tt, 0 * tt, 0 * tt], axis=1)
            fr = fx.frame(tk, tip, omega)
            b = fx.bus([fr], [fx.contact(rows)], removal=[fx.removal_contact(ap=10e-3)])
            dom.macro_step(ctx, state, b, fx.bus([fr], []))
        dom.end_operation(ctx, state)
        offset += 2_000_000                       # the next blank two seconds later
    store = state.surface.store
    ops = store.pi[:store.n_patch, PI["op"]]
    assert set(ops.tolist()) == {0, 1}
    assert int(np.sum(ops == 0)) == int(np.sum(ops == 1))          # the same faces, once per part
    for q in range(store.n_pass):
        assert int(store.pi[int(store.qi[q, QI["patch"]]), PI["op"]]) == int(store.qi[q, QI["op"]])
    # each part's patches hold only its own rows: the second part's finish equals the first part's (identical cuts)
    first = [p for p in range(store.n_patch) if int(ops[p]) == 0]
    second = [p for p in range(store.n_patch) if int(ops[p]) == 1]
    for p0, p1 in zip(first, second):
        a, b = fx.patch_values(state, p0), fx.patch_values(state, p1)
        assert set(a) == set(b)
        for key in a:
            if isinstance(a[key], (int, float)):
                assert b[key] == pytest.approx(a[key], rel=1e-9, abs=1e-12)
            else:
                assert b[key] == a[key]
    assert not [m for m in state.surface.marks if m.get("kind") == "witness"]
