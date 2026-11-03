"""Run directory (cnctwin.core.rundir): ids, identity, writers, timeline, frames.bin, scene, result set."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cnctwin.core import rundir as D
from cnctwin.core.bus import MotionFrame, PathFrame
from cnctwin.core.findings import Category, Finding, FindingList, Severity

ROOT = Path(__file__).resolve().parents[3]


def test_run_ids_are_deterministic(tmp_path):
    a = D.run_id("ab" * 32, {"x": 1, "y": [1, 2]}, 7, "Pocket, Al 6061")
    assert a == D.run_id("ab" * 32, {"y": [1, 2], "x": 1}, 7, "Pocket, Al 6061")
    assert a.startswith("00007-pocket-al-6061-") and len(a.split("-")[-1]) == 8
    assert D.run_id("ab" * 32, {"x": 2}, 7) != D.run_id("ab" * 32, {"x": 1}, 7)
    assert D.run_id("ab" * 32, {"x": 1, "resume_from": "/a"}, 7) == D.run_id("ab" * 32, {"x": 1}, 7)
    assert D.next_counter(tmp_path) == 1
    first = D.allocate_run_dir(tmp_path, "cd" * 32, {}, "p")
    (tmp_path / "00041-other").mkdir()
    second = D.allocate_run_dir(tmp_path, "cd" * 32, {}, "p")
    assert first.name.startswith("00001-p-") and second.name.startswith("00042-p-")


def test_numba_cache_key_matches_bin_tsafe():
    script = ("grep -rl --include='*.py' -e '@njit' -e '@numba' -e 'cuda.jit' cnctwin 2>/dev/null | LC_ALL=C sort "
              "| xargs -r sha256sum; python -c 'import numba, llvmlite; print(numba.__version__, llvmlite.__version__)'"
              " 2>/dev/null; grep -m1 'model name' /proc/cpuinfo")
    shell = subprocess.run(["bash", "-c", f"({script}) | sha256sum | cut -c1-16"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
    assert D.numba_cache_key() == shell
    from cnctwin.api.jobs import numba_cache_key
    assert numba_cache_key() == shell
    assert "cnctwin/core/recorder.py" in D.kernel_sources()


def test_build_id_and_versions():
    build = D.build_id()
    assert len(build) == 64 and build == D.build_id()
    v = D.versions()
    assert v["build"] == build and v["numpy"] == np.__version__ and v["format"] == 1
    assert D.thread_settings({"OMP_NUM_THREADS": "1"})["OMP_NUM_THREADS"] == "1"


def test_json_helpers_and_atomic_writes(tmp_path):
    value = {"a": np.float64(1.5), "b": np.array([1, 2]), "c": float("nan"), "d": (1, Path("/x")), "e": Severity.ERROR}
    assert D.jsonable(value) == {"a": 1.5, "b": [1, 2], "c": None, "d": [1, "/x"], "e": "error"}
    D.write_json_atomic(tmp_path / "x.json", value, fsync=True)
    assert json.loads((tmp_path / "x.json").read_text())["c"] is None
    assert not list(tmp_path.glob(".*.tmp"))
    one = D.npz_bytes({"b": np.arange(3), "a": np.array(2.5), "s": np.array(["x", "yz"])})
    two = D.npz_bytes({"s": np.array(["x", "yz"]), "a": np.array(2.5), "b": np.arange(3)})
    assert one == two
    D.write_npz(tmp_path / "x.npz", {"a": np.array(2.5), "f": np.asfortranarray(np.ones((2, 3)))})
    with np.load(tmp_path / "x.npz", allow_pickle=False) as archive:
        assert archive["a"].shape == () and archive["f"].shape == (2, 3)
    with pytest.raises(TypeError):
        D.npz_bytes({"o": np.array([object()])})
    assert D.tick_time_s(1_500_000) == 1.5 and D.ticks_to_seconds([3], 1e-6)[0] == 3e-6


def test_run_directory_files(tmp_path):
    run = D.RunDirectory(tmp_path)
    (tmp_path / "frames.bin").write_bytes(b"old")
    (tmp_path / "status.json").write_text("{}")
    (tmp_path / "project.yaml").write_text("keep")
    run.prepare()
    assert not (tmp_path / "frames.bin").exists() and not (tmp_path / "status.json").exists()
    assert (tmp_path / "project.yaml").exists() and (tmp_path / "series").is_dir()
    run.write_status("running", sim_time_s=0.5, est_total_s=2.0, wall_s=1.0, eta_s=3.0, op="OP10")
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["format"] == "cnctwin-run-status" and status["progress"]["eta_s"] == 3.0 and status["op"] == "OP10"
    with pytest.raises(ValueError):
        run.write_status("sleeping")
    findings = FindingList()
    findings.add(Finding(Category.COLLISION, Severity.ERROR, "holder hits vise", 1.0, 10, "removal", op="OP10", line=12))
    run.write_findings(findings)
    assert json.loads((tmp_path / "findings.json").read_text())["findings"][0]["category"] == "collision"
    assert D.verdict(findings) == {"safe": False, "reasons": ["collision (error) at OP10 line 12: holder hits vise"]}
    assert D.finding_counts(findings)["error"] == 1
    (tmp_path / "part_OP10.stl").write_bytes(b"solid")
    run.register_output("part_OP10.stl", 100)
    cut = run.checkpoint_cut()
    (tmp_path / "part_OP20.stl").write_bytes(b"solid")
    (tmp_path / "report.html").write_text("<html>")
    run.truncate_outputs(cut)
    assert (tmp_path / "part_OP10.stl").exists() and not (tmp_path / "part_OP20.stl").exists()
    assert not (tmp_path / "report.html").exists()
    index = {o["path"] for o in run.outputs_index()}
    assert "findings.json" in index and "status.json" not in index
    api_owned = D.RunDirectory(tmp_path / "api", own_status=False)
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "status.json").write_text('{"status": "running"}')
    api_owned.prepare()
    api_owned.write_status("done")
    assert json.loads((tmp_path / "api" / "status.json").read_text()) == {"status": "running"}


def test_timeline_and_production_reader(tmp_path):
    timeline = D.Timeline(tmp_path)
    a = timeline.open(path=0, op="OP10", program="op10.nc", line=3, block="G0 X0", kind="rapid", tick0=0, tool=1)
    timeline.close(a, 5000, ticks={"rapid": 5000})
    b = timeline.open(path=0, op="OP10", program="op10.nc", line=4, block="G1 X10 F600", kind="linear", tick0=5000)
    ref = timeline.line_ref(b)
    assert (ref.op, ref.program, ref.line, ref.block) == ("OP10", "op10.nc", 4, "G1 X10 F600")
    assert timeline.line_ref(99) is None
    timeline.upsert({"index": 2, "op": "OP10", "program": "op10.nc", "line": 1, "block": "G4 P1", "path": 1,
                     "kind": "dwell", "tick0": 0, "tick1": 7000, "ticks": {"dwell": 7000}, "final": False})
    assert [r["index"] for r in timeline.closed()] == [0]
    timeline.close(b, 12000, ticks={"cutting": 6000, "air_cutting": 1000})
    timeline.upsert({**timeline.records[2], "final": True})
    timeline.write()
    spans = timeline.path_spans(12000)
    assert spans[0]["block_ticks"] == 12000 and spans[0]["span_ticks"] == 12000 and spans[1]["block_ticks"] == 7000
    with pytest.raises(ValueError):
        timeline.close(a, 9000)
    with pytest.raises(ValueError):
        timeline.close(timeline.open(path=0, op="", program="", line=0, block="", kind="mode", tick0=500), 100)
    from cnctwin.production.timeline import read_timeline
    read = read_timeline(tmp_path)
    found = read.blocks if hasattr(read, "blocks") else read
    assert len(found) == 3 and sorted(b.duration_ticks for b in found) == [5000, 7000, 7000]
    assert {b.kind for b in found} == {"rapid", "linear", "dwell"}
    restored = D.Timeline(tmp_path)
    restored.restore_state(json.loads(json.dumps(timeline.checkpoint_state())))
    assert restored.records == timeline.records


def path_frame(tick0, n=10, ipo=1000):
    ticks = tick0 + ipo * np.arange(n, dtype=np.int64)
    angle = -1047.2 * ticks * 1e-6
    tip = np.column_stack([0.01 + 1e-3 * np.arange(n), np.full(n, 0.02), np.full(n, 0.005)])
    return PathFrame(path=0, tick0=tick0, ipo_ticks=ipo, tick=ticks, tip_S_m=tip, tip_M_m=tip - 0.3,
                     axis_T_M=np.tile([0.0, 0.0, 1.0], (n, 1)), axis_names=("X", "Y", "Z", "B"),
                     axis_actual=np.column_stack([tip - 0.3, 2.0 * math.pi * 3 + 0.25 * np.arange(n)]),
                     spindle_angle_rad=np.mod(angle, 2 * math.pi), spindle_rev=np.floor(angle / (2 * math.pi)).astype(np.int64))


def test_frames_writer_records_and_readers(tmp_path):
    frames = D.FramesWriter(tmp_path / "frames.bin", axes=[{"name": "X", "origin": -0.3}, {"name": "Y", "origin": -0.3},
                                                           {"name": "Z", "origin": -0.3},
                                                           {"name": "B", "kind": "rotary"}],
                            paths=[{"path": 0}], ipo_ticks=1000)
    frames.open()
    assert frames.append_motion(MotionFrame(paths=[path_frame(0)])) == 10
    assert frames.append_motion(MotionFrame(paths=[path_frame(10_000)])) == 10
    with pytest.raises(ValueError):
        frames.append_motion(MotionFrame(paths=[path_frame(5_000)]))
    frames.flush()
    header = frames.header()
    assert header["record_bytes"] == frames.dtype.itemsize == 8 + 16 + 16 + 4 + 4 + 12 + 12
    D.write_json_atomic(tmp_path / "scene.json", {"frames": header})
    records, extra = D.read_frames(tmp_path)
    assert len(records) == 20 and records["tick"][-1] == 19_000
    assert np.allclose(records["axis"][:3, 0], [0.01, 0.011, 0.012], atol=1e-8)
    assert np.all(records["axis_rev"][:, 3] == 3) and np.all(records["axis"][:, 3] < 2 * math.pi)
    assert np.allclose(extra["axis_abs"][:, 3], 2 * math.pi * 3 + 0.25 * (np.arange(20) % 10), atol=1e-6)
    unwrapped = extra["spindle_unwrapped_rad"][:, 0]
    assert np.allclose(unwrapped, -1047.2 * records["tick"] * 1e-6, atol=2e-6)
    from cnctwin.production.rundir import read_frames as production_frames
    from cnctwin.api.runfiles import frame_dtype
    assert frame_dtype(header) == D.frames_dtype(header)
    got = production_frames(tmp_path)
    assert got["axis"].shape == (20, 4) and got["axis_names"] == ("X", "Y", "Z", "B")
    cut = frames.checkpoint_cut()
    frames.append_motion(MotionFrame(paths=[path_frame(20_000)]))
    frames.truncate_outputs(cut)
    assert (tmp_path / "frames.bin").stat().st_size == cut["bytes"]


def test_wrap_never_reaches_two_pi():
    rng = np.random.default_rng(2)
    many = np.concatenate([2 * math.pi * np.arange(-50, 50) + d for d in (0.0, -1e-15, 1e-15, -1e-9)]
                          + [rng.uniform(-1e4, 1e4, 100_000)])
    wrapped, revs = D.FramesWriter.wrap(many)
    assert np.all(wrapped >= 0) and np.all(wrapped < np.float32(2 * math.pi))
    assert np.max(np.abs(wrapped.astype(float) + 2 * math.pi * revs - many)) < 4e-7 + 1e-15 * np.max(np.abs(many))
    near = np.array([2 * math.pi - 1e-9, -1e-12, 4 * math.pi - 1e-10, 1.0])
    wrapped, revs = D.FramesWriter.wrap(near)
    assert np.all(wrapped < np.float32(2 * math.pi)) and np.all(wrapped >= 0)
    assert revs.tolist() == [1, 0, 2, 0]
    assert np.allclose(wrapped.astype(float) + 2 * math.pi * revs, near, atol=1e-6)


def test_scene_writer_uses_geom_glb(tmp_path):
    from cnctwin.geom.io.gltf import read_glb
    scene = D.SceneWriter(tmp_path)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * 0.1
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    relative = scene.add_mesh("table", (v, f), color_rgba=(0.5, 0.5, 0.5, 1.0))
    assert relative == "scene/table.glb"
    assert len(read_glb(tmp_path / relative).mesh().vertices) == 4
    with pytest.raises(ValueError):
        scene.add_mesh("../evil", (v, f))
    scene.write({"machine": {"components": [{"id": "table", "mesh": relative}]}}, frames={"record_bytes": 8, "fields": []})
    document = json.loads((tmp_path / "scene.json").read_text())
    assert document["format"] == "cnctwin-scene" and document["meshes"] == {"table": "scene/table.glb"}


def test_run_document_and_result_set(tmp_path):
    findings = FindingList()
    doc = D.run_document(run_id="00001-x", project={"name": "toy", "seed": 3}, options={"a": 1, "resume_from": "/c"},
                         clock={"tick_s": 1e-6, "ipo_ticks": 1000}, verdict_=D.verdict(findings),
                         summary={"cycle_time_s": 1.25, "bad": float("inf")}, outputs=[])
    assert doc["format"] == "cnctwin-run" and doc["options"] == {"a": 1} and doc["seed"] == 3
    assert doc["verdict"]["safe"] and doc["summary"]["bad"] is None
    assert doc["numba_cache_key"] == D.numba_cache_key() and doc["versions"]["build"] == D.build_id()
    text = D.dumps(doc)
    assert "wall" not in text and "time.time" not in text
    for name in ("findings.json", "trace.jsonl", "frames.bin", "series/tick/part-0.parquet", "keyframes/kf_0.glb",
                 "part_OP10.stl", "run.json", "status.json", "series/.part-1.parquet.tmp"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(name.encode())
    digest = D.result_set_digest(tmp_path)
    assert set(digest) == {"findings.json", "trace.jsonl", "frames.bin", "series/tick/part-0.parquet",
                           "keyframes/kf_0.glb", "part_OP10.stl"}
