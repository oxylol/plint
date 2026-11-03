"""The synthetic run's 3D view additions (tests/fixtures/make_synthetic_run.py, view3d=True): deterministic,
also across a checkpoint and resume; off by default (the files of the other tests are the same with and
without the option); the carved bricks are written with cnctwin.core.keyframes' writers, closed and
consistent in volume; only changed records are indexed; the 3-D grids, the pile on its own grid, the jets
with paths and a blocked head-mounted jet, and final attributes on the last record of every brick."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import make_synthetic_run as synthetic  # noqa: E402
from cnctwin.core.keyframes import KeyframeReader  # noqa: E402


def test_option_off_writes_the_default_run(tmp_path):
    plain = synthetic.make_synthetic_run(tmp_path / "plain", seed=3, media=False)
    off = synthetic.make_synthetic_run(tmp_path / "off", seed=3, media=False, view3d=False)
    assert synthetic.result_set_digest(tmp_path / "plain") == synthetic.result_set_digest(tmp_path / "off")
    assert plain["files"] == off["files"]
    fields = np.load(tmp_path / "plain" / "keyframes" / "kf_2.npz")
    assert "work_temp_K" not in fields.files and len(fields["jet_kind"]) == 1


def test_view3d_run_is_deterministic_and_resumes(tmp_path):
    synthetic.make_synthetic_run(tmp_path / "a", seed=6, media=False, view3d=True)
    synthetic.make_synthetic_run(tmp_path / "b", seed=6, media=False, view3d=True)
    digest = synthetic.result_set_digest(tmp_path / "a")
    assert digest == synthetic.result_set_digest(tmp_path / "b")
    first = synthetic.SyntheticRun(tmp_path / "r", seed=6, media=False, view3d=True)
    first.begin()
    first.run_all(until_s=1.0)
    second = synthetic.SyntheticRun(tmp_path / "r", seed=6, media=False, view3d=True)
    second.resume(tmp_path / "r" / "checkpoints" / "ck_1000000.json")
    second.run_all()
    assert synthetic.result_set_digest(tmp_path / "r") == digest


def test_carved_bricks_fields_jets_attributes(tmp_path):
    run = tmp_path / "run"
    synthetic.make_synthetic_run(run, seed=4, media=False, view3d=True)
    reader = KeyframeReader(run)
    stock = np.prod(synthetic.STOCK_SIZE_M)
    volumes = []
    for t in (0.0, 1.2, 2.3):
        total = 0.0
        for key, (v, f) in reader.scene_at(time_s=t).items():
            total += np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0
        volumes.append(total)
    assert abs(volumes[0] - stock) < 1e-6 * stock          # float32 quantisation of the brick records
    assert volumes[0] > volumes[1] > volumes[2] > 0.9 * stock
    entries = [json.loads(line) for line in (run / "keyframes" / "index.jsonl").read_text().splitlines()]
    index = json.loads((run / "keyframes" / "bricks.idx").read_text())["bricks"]
    assert entries[1]["bricks"] == [] and all(isinstance(b, str) for e in entries for b in e["bricks"])
    assert len(index["2"]) == 1 and len(index["0"]) >= 2
    blocked = 0
    for entry in entries:
        fields = np.load(run / entry["fields"])
        temp = fields["work_temp_K"]
        assert temp.dtype == np.float16 and temp.shape == (10, 8, 4)
        assert fields["work_displacement_m"].shape == (10, 8, 4, 3)
        assert fields["pile_height_m"].shape == synthetic.VIEW3D_PILE_CELLS and fields["wetted"].shape == synthetic.VIEW3D_PILE_CELLS
        if len(fields["jet_kind"]):
            assert fields["jet_path_M_m"].shape == (2, synthetic.VIEW3D_PATH_POINTS, 3)
            blocked += int(fields["jet_blocked_by"][1] == "workpiece 0")
    assert blocked >= 1
    for key, versions in index.items():
        vertices, _ = reader.brick(key)
        attr = np.load(run / "keyframes" / "attr" / f"brick_{key}.npz")
        assert attr["deviation_m"].shape == (len(vertices),)
    floor = np.load(run / "keyframes" / "attr" / "brick_0.npz")["deviation_m"]
    assert np.nanmax(floor) > 5e-5
    scene = json.loads((run / "scene.json").read_text())
    assert [n["id"] for n in scene["nozzles"]] == ["n1", "n2"] and scene["nozzles"][1]["attached_to"] == "head"
