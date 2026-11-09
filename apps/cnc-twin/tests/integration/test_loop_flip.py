"""Two operations with a flip (docs/formats.md "from_previous and stock_pose"; docs/models/loop.md "Operations"):
OP10 cuts a side step along the blank's front top edge, every domain ends the operation (cool-down, unclamp,
release, removal's transfer of the part turned 180 deg about X), OP20 cuts a side step along the front top edge
of the turned part (the old back bottom edge). The part keeps its material across the setups, the two steps end
up on opposite faces, the timeline and the ledgers span both operations, and each operation leaves its part
files and an operation-end checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cnctwin.core.loop import run_project
from cnctwin.core.rundir import read_frames
from tests.integration import loopfix as LF


@pytest.fixture(scope="module")
def flipped(tmp_path_factory):
    base = tmp_path_factory.mktemp("loop-flip")
    project = LF.flip_project(base / "project")
    result = run_project(project, base / "run", {"outputs": False}, None)
    return base / "run", result


def test_both_operations_ran_and_the_ledgers_pass(flipped):
    run_dir, result = flipped
    run = LF.read(run_dir, "run.json")
    assert result["complete"] and run["summary"]["operations"] == ["OP10", "OP20"]
    assert LF.conservation_ok(run) == {"mass": True, "energy": True, "time": True, "final_mass": True}
    blocks = LF.read(run_dir, "timeline.json")["blocks"]
    ops = [b["op"] for b in blocks]
    assert ops.index("OP20") > ops.index("OP10") and set(ops) == {"OP10", "OP20"}
    assert [b["index"] for b in blocks] == list(range(len(blocks)))
    first20 = next(b for b in blocks if b["op"] == "OP20")
    last10 = [b for b in blocks if b["op"] == "OP10"][-1]
    assert first20["tick0"] >= last10["tick1"]


def test_each_step_removed_its_volume(flipped):
    run_dir, _ = flipped
    run = LF.read(run_dir, "run.json")
    blocks = LF.read(run_dir, "timeline.json")["blocks"]
    # each operation's step from the path its tool took (frames.bin, the operation's own S: the blank's box
    # 80 x 60 with the top at z 25 mm in both setups)
    swept = []
    for op in ("OP10", "OP20"):
        own = [b for b in blocks if b["op"] == op]
        ticks = (min(b["tick0"] for b in own), max(b["tick1"] for b in own))
        swept.append(LF.swept_volume_m3(run_dir, 25e-3, ((0.0, 0.0), (80e-3, 60e-3)), ticks=ticks))
    assert all(4.0e-9 < v < 6.5e-9 for v in swept), swept
    removed = run["summary"]["removed_volume_m3"]
    assert removed == pytest.approx(sum(swept), rel=0.03)
    removal = run["domains"]["removal"]["workpieces"][0]
    assert removal["volume_m3"] == pytest.approx(removal["initial_volume_m3"] - removed, rel=1e-6)


def _floor_area_m2(mesh, z_m: float, normal_z: float, y_lo_m: float, y_hi_m: float) -> float:
    """Area of the mesh's faces facing ``normal_z`` (+1 up, -1 down) at height ``z_m`` (0.05 mm) whose centres
    lie between y_lo and y_hi."""
    V = np.asarray(mesh.vertices, float)
    T = V[np.asarray(mesh.faces)]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    nz = n[:, 2] / np.maximum(2.0 * area, 1e-30)
    c = T.mean(axis=1)
    take = (np.abs(c[:, 2] - z_m) < 0.05e-3) & (nz * normal_z > 0.9) & (c[:, 1] > y_lo_m) & (c[:, 1] < y_hi_m)
    return float(area[take].sum())


def test_the_part_was_transferred_and_cut_on_the_other_face(flipped):
    run_dir, _ = flipped
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    transfers = [e for e in trace if e["event"] == "removal:transfer"]
    assert len(transfers) == 1
    from cnctwin.geom.io import read_mesh
    part10 = read_mesh(run_dir / "part_OP10.stl")
    part20 = read_mesh(run_dir / "part_OP20.stl")
    for mesh in (part10, part20):
        V = np.asarray(mesh.vertices)
        assert float(V[:, 2].min()) == pytest.approx(0.0, abs=0.3e-3)
        assert float(V[:, 2].max()) == pytest.approx(25e-3, abs=0.3e-3)
    # the step floors from the tool tip's height in each setup (frames.bin): OP10's faces up at the blank's front
    # top edge (y 0 .. 0.5 mm); after the turn about X (S = (x, 60 - y, 25 - z)) it faces down at the back bottom
    # edge of OP20's S, and OP20's own floor faces up at its front top edge
    records, _ = read_frames(run_dir)
    blocks = LF.read(run_dir, "timeline.json")["blocks"]
    floors = {}
    for op in ("OP10", "OP20"):
        own = [b for b in blocks if b["op"] == op]
        take = (records["tick"] >= min(b["tick0"] for b in own)) & (records["tick"] < max(b["tick1"] for b in own))
        z = np.asarray(records["tip_S_m"][take, 0, 2], float)
        floors[op] = float(np.median(z[z < 25e-3 - 1e-5]))
    assert floors["OP10"] == pytest.approx(24e-3, abs=0.05e-3) and floors["OP20"] == pytest.approx(24e-3, abs=0.05e-3)
    assert _floor_area_m2(part10, floors["OP10"], +1.0, 0.0, 0.6e-3) > 1e-6
    assert _floor_area_m2(part20, floors["OP20"], +1.0, 0.0, 0.6e-3) > 1e-6
    assert _floor_area_m2(part20, 25e-3 - floors["OP10"], -1.0, 59.4e-3, 60e-3) > 1e-6
    assert _floor_area_m2(part20, floors["OP20"], +1.0, 59.4e-3, 60e-3) == 0.0


def test_operation_ends_leave_files_and_checkpoints(flipped):
    run_dir, _ = flipped
    for name in ("part_OP10.stl", "part_OP10_unload.stl", "part_OP20.stl", "part_OP20_unload.stl"):
        assert (run_dir / name).is_file(), name
    markers = [json.loads(p.read_text()) for p in (run_dir / "checkpoints").glob("ck_*.json")]
    ends = sorted((m["op"], m["tick"]) for m in markers if m["op_end"])
    assert [op for op, _ in ends] == ["OP10", "OP20"]
    holding = LF.read(run_dir, "run.json")["domains"]["holding"]
    assert len(holding["setups"]) >= 2
