"""InspectDomain end to end with fixtures: two operations with a flip, inspection.json in the keys the report reads,
findings, part files, the next stock, the checkpoint participant protocol."""

import json

import numpy as np
import pytest

from cnctwin.core.domain import Domain, RunContext
from cnctwin.core.findings import Category, Severity
from cnctwin.core.state import SimState
from cnctwin.inspect.domain import InspectDomain
from cnctwin.inspect.protocols import ColumnsStockBuilder, StaticResults, SurfaceJson
from cnctwin.inspect.result import DisplacementGrid, ResultPart
from cnctwin.report.sections_part import _judged, limits
from tests.unit.inspect import inspfix as fx

LO, HI = [0.0, 0.0, 0.0], [0.040, 0.030, 0.015]
PLO, PHI = [0.01225, 0.00925], [0.02825, 0.02125]
FLOOR = {"face": "plane", "normal": [0, 0, 1], "near_mm": [20, 15, 11]}
TOP = {"face": "plane", "normal": [0, 0, 1], "near_mm": [5, 4, 15]}


def _project():
    return {
        "cad": {"cad_pose_S_C": np.eye(4),
                "datums": [{"label": "A", "feature": {"face": "plane", "normal": [0, 0, -1], "near_mm": [20, 15, 0]}},
                           {"label": "B", "feature": {"face": "plane", "normal": [-1, 0, 0], "near_mm": [0, 15, 7]}},
                           {"label": "C", "feature": {"face": "plane", "normal": [0, -1, 0], "near_mm": [20, 0, 7]}}],
                "tolerances": [
                    {"id": "pocket-depth", "kind": "distance", "between": [TOP, FLOOR], "nominal_mm": 4,
                     "plus_mm": 0.02, "minus_mm": 0.02},
                    {"id": "floor-flatness", "kind": "flatness", "feature": FLOOR, "tolerance_mm": 0.01},
                    {"id": "wall-perpendicularity", "kind": "perpendicularity", "tolerance_mm": 0.01, "datums": ["A"],
                     "feature": {"face": "plane", "normal": [1, 0, 0], "near_mm": [12.25, 15, 13]}},
                    {"id": "floor-finish", "kind": "roughness", "feature": FLOOR, "Ra_um": 1.6},
                    {"id": "bore", "kind": "diameter", "feature": {"face": "cylinder"},
                     "diameter_mm": {"nominal": 8, "fit": "H7"}}]},
        "operations": [{"name": "OP10", "stock_pose_W_S": np.eye(4)},
                       {"name": "OP20", "stock_pose_W_S": np.eye(4),
                        "from_previous": {"op": "OP10", "flip": {"axis": "X", "angle_deg": 180}}}],
        "stock": {"kind": "block"}, "physics": {"lattice_m": 5e-4}}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    model = fx.model(fx.pocket_block(LO, HI, PLO, PHI, 0.004), fx.pocket_specs(LO, HI, PLO, PHI, 0.004))
    op10 = fx.subdivide(fx.pocket_block(LO, HI, PLO, PHI, 0.00395), 2)          # 0.05 mm left on the floor
    spring = DisplacementGrid([0, 0, 0], 0.01, np.zeros((5, 4, 3, 3)))
    results = StaticResults()
    results.add(ResultPart("OP10", op10, {"final": spring}, op_index=0, unload_time_s=30.0, unload_mean_temp_K=296.0,
                           unload_max_temp_K=299.5))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    flip[1, 3], flip[2, 3] = HI[1], HI[2]                                         # removal's box convention
    results.add(ResultPart("OP20", op10.transformed(flip), {}, op_index=1))
    surface = SurfaceJson({"features": [{"id": "floor-finish", "Ra_m": 0.8e-6, "pass": True, "op": "OP10", "tool": 1,
                                         "lines": [11]}], "threads": []})
    dom = InspectDomain(results=results, surface=surface, cad=model, stock_builder=ColumnsStockBuilder(5e-4))
    ctx = RunContext(project=_project(), db=None, run_dir=run_dir, seed=1, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10)
    state = SimState()
    dom.init(ctx, state)
    for k in range(2):
        state.clock.op_index = k
        state.clock.tick = 1000000 * (k + 1)
        dom.end_operation(ctx, state)
    doc = dom.finalize(ctx, state)
    return dom, ctx, state, doc, run_dir


def test_is_a_domain_and_takes_no_part_in_the_loop(run):
    dom, ctx, state, _, _ = run
    assert isinstance(dom, Domain)
    assert dom.channels() == [] and dom.kernel_inputs(ctx, state, state.bus_prev) is None
    assert dom.macro_step(ctx, state, state.bus_prev, state.bus_next) is None


def test_inspection_json_in_the_report_keys(run):
    _, _, _, doc, run_dir = run
    on_disk = json.loads((run_dir / "inspection.json").read_text())
    assert on_disk["format"] == "cnctwin-inspection" and on_disk["temperature_K"] == pytest.approx(293.15)
    tol = {t["id"]: t for t in on_disk["tolerances"]}
    depth = tol["pocket-depth"]
    assert depth["measured_m"] == pytest.approx(0.00395, abs=1e-12) and depth["pass"] is False
    assert limits(depth) == (pytest.approx(0.004), pytest.approx(0.00398), pytest.approx(0.00402))
    assert _judged(depth) is False
    assert tol["floor-flatness"]["pass"] is True and tol["floor-flatness"]["tolerance_m"] == pytest.approx(1e-5)
    assert tol["wall-perpendicularity"]["pass"] is True
    assert tol["floor-finish"]["measured_m"] == pytest.approx(0.8e-6) and tol["floor-finish"]["pass"] is True
    assert tol["bore"]["status"] == "not evaluated" and "cylinder" in tol["bore"]["reason"]
    assert on_disk["summary"] == {"checked": 4, "failed": 1, "not_evaluated": 1, "cmm_disagrees": 0}
    assert tol["pocket-depth"]["cmm"]["measured"] == pytest.approx(0.00395, abs=1e-9)
    assert len(tol["floor-flatness"]["points"]) >= 3 and "deviation_m" in tol["floor-flatness"]["points"][0]
    assert on_disk["unload"][0] == {"op": "OP10", "workpiece": 0, "time_s": 30.0, "mean_temp_K": 296.0,
                                    "max_temp_K": 299.5}
    assert [r["kind"] for r in on_disk["cmm_report"]].count("distance") == 1
    # the final part is OP20's: the CAD followed the flip, the floor material is still there
    assert [r["op"] for r in on_disk["operations"]] == ["OP10", "OP20"]
    rem = on_disk["remaining"]
    assert len(rem) == 1 and rem[0]["volume_m3"] == pytest.approx(0.016 * 0.012 * 5e-5, rel=1e-9)


def test_findings_files_and_next_stock(run):
    dom, _, state, doc, run_dir = run
    found = {(f.category, f.severity) for f in state.findings}
    assert (Category.OUT_OF_TOLERANCE, Severity.ERROR) in found
    assert (Category.OUT_OF_TOLERANCE, Severity.INFO) in found
    assert (Category.REMAINING_MATERIAL, Severity.WARNING) in found
    assert not any(f.category == Category.GOUGE for f in state.findings)
    for name in ("part_OP10.glb", "part_OP10.stl", "part_OP10.3mf", "part_OP10_unload.stl", "part_OP20.stl"):
        assert (run_dir / name).is_file()
    ns = dom.next_stock(1, 0)
    assert ns is not None and ns.method == "rasterised"
    assert doc["next_stock"][0]["for_op"] == "OP20"
    np.testing.assert_allclose(dom.cad_poses[0][:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-15)


def test_checkpoint_participant_round_trip(run):
    dom, ctx, state, _, _ = run
    cut = dom.checkpoint_cut()
    saved = json.loads(json.dumps(dom.checkpoint_state()))
    other = InspectDomain()
    other.init.__func__  # the protocol is plain methods
    other.base_pose = np.eye(4)
    other.restore_state(saved)
    assert other.state.records == dom.state.records
    np.testing.assert_allclose(other.cad_poses[0], dom.cad_poses[0], atol=0)
    other.truncate_outputs({"records": 1, "next_stocks": 0, "finding_keys": 0, "files": 0})
    assert len(other.state.records) == 1 and other.cad_poses == {}
    assert cut["records"] == 2


def test_inspection_json_is_deterministic(run, tmp_path):
    dom, ctx, state, doc, run_dir = run
    model = dom.model
    again_dir = tmp_path / "again"
    again_dir.mkdir()
    dom2 = InspectDomain(results=dom.results, surface=dom.surface, cad=model, stock_builder=ColumnsStockBuilder(5e-4))
    ctx2 = RunContext(project=_project(), db=None, run_dir=again_dir, seed=1, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10)
    state2 = SimState()
    dom2.init(ctx2, state2)
    for k in range(2):
        state2.clock.op_index = k
        state2.clock.tick = 1000000 * (k + 1)
        dom2.end_operation(ctx2, state2)
    dom2.finalize(ctx2, state2)
    for name in ("inspection.json", "part_OP10.glb", "part_OP10.3mf", "part_OP20_unload.stl"):
        assert (again_dir / name).read_bytes() == (run_dir / name).read_bytes(), name
    assert [f.to_dict() for f in state2.findings] == [f.to_dict() for f in state.findings]

