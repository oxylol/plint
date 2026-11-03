"""The synthetic run directory generator (tests/fixtures/make_synthetic_run.py): every file of
architecture 16, deterministic from a seed, identical across a checkpoint and resume."""

from __future__ import annotations

import json
import struct
import zipfile

import numpy as np
import pyarrow.parquet as pq
from api_support import synthetic


ARCHITECTURE_16 = ["run.json", "status.json", "findings.json", "trace.jsonl", "channels.json", "timeline.json",
                   "series/index.jsonl", "scene.json", "frames.bin", "keyframes/bricks.idx", "keyframes/index.jsonl",
                   "inspection.json", "surface.json", "tools.json", "production.json", "sld/T1.json", "report.html",
                   "report.pdf", "setup_sheet.html", "setup_sheet.pdf", "video.mp4", "audio.wav",
                   "part_OP10.glb", "part_OP10.stl", "part_OP10.3mf", "part_OP10_unload.stl"]
PATTERNS = ["series/fast/part-*.parquet", "series/tick/part-*.parquet", "series/macro/part-*.parquet",
            "scene/*.glb", "keyframes/kf_*.glb", "keyframes/kf_*.npz", "keyframes/attr/*.npz", "chips/kf_*.npz",
            "images/*.png", "checkpoints/ck_*.npz", "checkpoints/ck_*.json"]


def test_every_file_of_the_run_directory(synthetic_runs):
    run = synthetic_runs / "runs" / "00001-done"
    for name in ARCHITECTURE_16:
        assert (run / name).is_file(), name
    for pattern in PATTERNS:
        assert list(run.glob(pattern)), pattern


def test_contents_are_plausible(synthetic_runs):
    run = synthetic_runs / "runs" / "00001-done"
    document = json.loads((run / "run.json").read_text())
    assert document["format"] == "cnctwin-run" and document["verdict"]["safe"] is False
    outputs = {o["path"] for o in document["outputs"]}
    assert "findings.json" in outputs and "run.json" not in outputs and "status.json" not in outputs
    findings = json.loads((run / "findings.json").read_text())["findings"]
    assert [f["id"] for f in findings] == list(range(len(findings)))
    assert all(f["cause_chain"] and f["line"] > 0 for f in findings)
    trace = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
    assert [e["id"] for e in trace] == list(range(len(trace)))
    scene = json.loads((run / "scene.json").read_text())
    header = scene["frames"]
    assert (run / "frames.bin").stat().st_size % header["record_bytes"] == 0
    dtype = np.dtype({"names": [f["name"] for f in header["fields"]],
                      "formats": [(f["dtype"], tuple(f["shape"])) if f["shape"] else f["dtype"] for f in header["fields"]],
                      "offsets": [f["offset"] for f in header["fields"]], "itemsize": header["record_bytes"]})
    frames = np.fromfile(run / "frames.bin", dtype=dtype)
    assert np.all(np.diff(frames["tick"]) == header["ipo_ticks"])
    assert np.all((frames["spindle_angle_rad"] >= 0) & (frames["spindle_angle_rad"] < 2 * np.pi))
    table = pq.read_table(run / "series" / "tick" / "part-0.parquet")
    assert table.column_names[:2] == ["tick", "time_s"] and "cutting.force_N[0].max" in table.column_names
    assert np.isnan(table.column("cutting.chip_thickness_m[15].mean").to_numpy()).all()
    glb = (run / "keyframes" / "kf_0.glb").read_bytes()
    assert glb[:4] == b"glTF" and struct.unpack("<I", glb[8:12])[0] == len(glb)
    index = json.loads((run / "keyframes" / "bricks.idx").read_text())
    k, name, offset, length = index["bricks"]["0"][0]
    record = (run / "keyframes" / name).read_bytes()[offset:offset + length]
    magic, vertices, indices = struct.unpack("<4sII", record[:12])
    assert magic == b"BRK1" and length == 28 + vertices * 8 + indices * 4
    with zipfile.ZipFile(run / "keyframes" / "kf_1.npz") as archive:
        assert {i.date_time for i in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    assert (run / "video.mp4").read_bytes()[4:8] == b"ftyp"
    assert (run / "audio.wav").read_bytes()[:4] == b"RIFF"
    assert (run / "report.pdf").read_bytes()[:5] == b"%PDF-"


def test_same_seed_same_bytes_and_seeds_differ(tmp_path):
    a = synthetic.make_synthetic_run(tmp_path / "a", seed=5, media=False)
    synthetic.make_synthetic_run(tmp_path / "b", seed=5, media=False)
    synthetic.make_synthetic_run(tmp_path / "c", seed=6, media=False)
    digest_a = synthetic.result_set_digest(tmp_path / "a")
    assert digest_a == synthetic.result_set_digest(tmp_path / "b")
    assert digest_a != synthetic.result_set_digest(tmp_path / "c")
    assert a["finished"] and a["verdict"]["safe"] is True     # odd seed: no error finding


def test_resume_from_a_checkpoint_is_bit_identical(tmp_path):
    synthetic.make_synthetic_run(tmp_path / "straight", seed=8, media=False)
    first = synthetic.SyntheticRun(tmp_path / "resumed", seed=8, media=False)
    first.begin()
    first.run_all(until_s=1.6)
    second = synthetic.SyntheticRun(tmp_path / "resumed", seed=8, media=False)
    second.resume(tmp_path / "resumed" / "checkpoints" / "ck_1000000.json")
    second.run_all()
    assert synthetic.result_set_digest(tmp_path / "straight") == synthetic.result_set_digest(tmp_path / "resumed")


def test_partial_run_has_no_results_yet(synthetic_runs):
    run = synthetic_runs / "runs" / "00002-partial"
    assert json.loads((run / "status.json").read_text())["status"] == "running"
    assert not (run / "run.json").exists() and not (run / "inspection.json").exists()
    assert (run / "checkpoints" / "ck_1000000.json").exists()


def test_project_folder_and_cli(tmp_path):
    project = synthetic.write_synthetic_project(tmp_path / "p", seed=11)
    assert synthetic.seed_from_project(project) == 11
    assert (tmp_path / "p" / "op10.nc").read_text().splitlines()[16] == "G01 X0."
    assert synthetic.main([str(tmp_path / "cli"), "--seed", "3", "--until", "0.3", "--no-media"]) == 0
    assert json.loads((tmp_path / "cli" / "status.json").read_text())["status"] == "running"
