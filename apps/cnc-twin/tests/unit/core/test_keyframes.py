"""Keyframes (cnctwin.core.keyframes): brick records, the GLB container, the bricks index, fields, reader."""

import json
import struct

import numpy as np
import pytest

from cnctwin.core import keyframes as K
from cnctwin.geom.io.gltf import read_glb


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def test_quantisation_error_bound():
    rng = np.random.default_rng(1)
    for scale, offset in ((0.008, 0.29), (0.008, -0.31), (1e-4, 0.0), (0.2, 1.7)):
        v = offset + scale * rng.random((500, 3))
        q, origin, step = K.quantise(v)
        back = origin.astype(np.float64) + q[:, :3].astype(np.float64) * step
        assert q.dtype == np.uint16 and np.all(q[:, 3] == 0)
        assert np.max(np.abs(back - v)) <= 0.5 * step * (1 + 1e-12)
        assert step <= scale / K.QUANT_MAX * (1 + 1e-6) + 1e-12
    # an 8 mm brick at 0.25 mm lattice: within 0.061 um
    v, f = box((0.296, 0.192, 0.088), (0.304, 0.2, 0.096))
    _, _, step = K.quantise(v)
    assert 0.5 * step < 6.2e-8


def test_record_round_trip_and_empty():
    v, f = box((0.01, 0.02, 0.0), (0.018, 0.028, 0.004))
    record = K.brick_record(v, f)
    assert record[:4] == b"BRK1" and len(record) == 28 + 8 * 8 + 12 * 3 * 4
    back, faces, header = K.decode_record(record)
    assert np.array_equal(faces, f) and header["n_vertices"] == 8
    assert np.max(np.abs(back - v)) <= 0.5 * header["step_m"] * (1 + 1e-12)
    empty = K.brick_record(None, None)
    assert len(empty) == 28 and K.decode_record(empty)[2]["n_vertices"] == 0
    with pytest.raises(ValueError):
        K.brick_record(v, f + 10)
    with pytest.raises(ValueError):
        K.decode_record(b"XXXX" + record[4:])


def test_brick_glb_is_valid_gltf_and_records_are_ranges():
    meshes = {"0": box((0, 0, 0), (0.008, 0.008, 0.008)), "1": box((0.008, 0, 0), (0.016, 0.008, 0.005))}
    records = [(key, K.brick_record(*mesh)) for key, mesh in meshes.items()] + [("w1.7", K.brick_record(None, None))]
    data, spans = K.brick_glb(records, name="kf_0")
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert (magic, version, length) == (K.GLB_MAGIC, 2, len(data))
    for key, record in records:
        offset, size = spans[key]
        assert data[offset:offset + size] == record and offset % 4 == 0
    scene = read_glb(data)
    nodes = {n["name"]: n for n in scene.nodes}
    assert set(nodes) == {"brick_0", "brick_1", "brick_w1.7"} and nodes["brick_w1.7"]["mesh"] is None
    for key, (v, f) in meshes.items():
        node = nodes[f"brick_{key}"]
        world = node["mesh"].transformed(node["world"]).vertices
        decoded, _, header = K.decode_record(dict(records)[key])
        assert np.max(np.abs(world - decoded)) < 1e-9
        assert np.max(np.abs(world - v)) <= 0.5 * header["step_m"] + 1e-9
    doc = scene.json
    assert doc["extensionsRequired"] == ["KHR_mesh_quantization"]
    assert all(a["componentType"] in (5123, 5125) for a in doc["accessors"])
    empty, _ = K.brick_glb([])
    length, kind = struct.unpack_from("<II", empty, 12)
    document = json.loads(empty[20:20 + length])
    assert kind == K.GLB_JSON and len(empty) == 20 + length
    assert document["scenes"] == [{"name": ""}] and "nodes" not in document and "buffers" not in document


def test_coarsen():
    values = np.arange(5 * 4 * 3, dtype=float).reshape(5, 4, 3)
    values[0, 0, 0] = np.nan
    coarse, factor = K.coarsen(values, 0.002, 0.008)
    assert factor == 4 and coarse.shape == (2, 1, 1)
    assert coarse[0, 0, 0] == pytest.approx(np.nanmean(values[0:4, 0:4, 0:3]))
    assert coarse[1, 0, 0] == pytest.approx(np.mean(values[4:5, 0:4, 0:3]))
    same, one = K.coarsen(values, 0.01, 0.008)
    assert one == 1 and np.array_equal(same, values, equal_nan=True)
    allnan, _ = K.coarsen(np.full((4, 4), np.nan), 0.002, 0.008)
    assert np.isnan(allnan).all()


def test_writer_index_full_snapshots_and_reader(tmp_path):
    kf = K.KeyframeWriter(tmp_path, period_ticks=1000, full_every=3, lattice_m=0.00025, origin_S_m=[0, 0, 0], grid=[2, 1, 1])
    heights = {0: 0.008, 1: 0.008}

    def meshes():
        return {b: box((0.008 * b, 0, 0), (0.008 * (b + 1), 0.008, h)) if h > 0 else None for b, h in heights.items()}

    assert kf.due(0)
    kf.write(0, meshes(), fields={"surface_temp_K": np.full((2, 2), 300.0), "part_pose": np.eye(4)})
    heights[0] = 0.006
    entry = kf.write(1000, {0: meshes()[0]}, chips={"mass_kg": np.ones(3)})
    assert entry["bricks"] == ["0"] and not entry["full"] and entry["chips"] == "chips/kf_1.npz"
    same = kf.write(2000, {1: meshes()[1]}, changed=[1])   # brick 1 re-meshed but unchanged: nothing new
    assert same["bricks"] == [] and same["empty"] == []
    heights[1] = 0.0
    full = kf.write(3000, meshes())                        # full snapshot: brick 1 is empty now
    assert full["full"] and full["bricks"] == ["0", "1"] and full["empty"] == ["1"]
    index = json.loads((tmp_path / "keyframes" / "bricks.idx").read_text())
    assert [e[0] for e in index["bricks"]["0"]] == [0, 1]   # the full snapshot repeated brick 0: no new entry
    assert [e[0] for e in index["bricks"]["1"]] == [0, 3]
    assert index["format"] == "cnctwin-bricks-index" and index["cells"] == 32
    reader = K.KeyframeReader(tmp_path)
    assert [e["k"] for e in reader.entries()] == [0, 1, 2, 3]
    v0, _ = reader.brick(0, tick=500)
    v1, _ = reader.brick(0, tick=1500)
    assert v0[:, 2].max() == pytest.approx(0.008, abs=1e-7) and v1[:, 2].max() == pytest.approx(0.006, abs=1e-7)
    assert reader.brick(1, tick=2999) is not None and reader.brick(1, tick=3000) is None
    assert set(reader.scene_at(tick=2500)) == {"0", "1"} and set(reader.scene_at()) == {"0"}
    assert reader.playback(tick=2500) == [0, 1, 2] and reader.playback(tick=3000) == [3]
    fields = reader.fields(0)
    assert fields["surface_temp_K"].dtype == np.float16 and fields["tick"] == 0
    assert reader.chips(1)["mass_kg"].tolist() == [1.0, 1.0, 1.0]
    assert kf.write_attributes({0: {"deviation_m": np.zeros(8, np.float32)}}) == ["keyframes/attr/brick_0.npz"]


def test_grids_are_coarsened_to_float16(tmp_path):
    kf = K.KeyframeWriter(tmp_path, period_ticks=10)
    grid = K.FieldGrid(np.full((9, 8, 2), 310.0), np.array([0.0, 0.0, -0.01]), 0.002)
    kf.write(0, {}, grids={"work_temp_K": grid, "flush_Pa": K.FieldGrid(np.ones((4, 4)), np.zeros(3), 0.004)})
    fields = K.KeyframeReader(tmp_path).fields(0)
    assert fields["work_temp_K"].dtype == np.float16 and fields["work_temp_K"].shape == (3, 2, 1)
    assert fields["work_temp_K_cell_m"].shape == () and float(fields["work_temp_K_cell_m"]) == pytest.approx(0.008)
    assert fields["tick"].shape == () and fields["time_s"].shape == ()
    assert fields["flush_Pa"].dtype == np.float32 and fields["flush_Pa"].shape == (2, 2)


def test_npz_files_are_deterministic(tmp_path):
    for name in ("a", "b"):
        kf = K.KeyframeWriter(tmp_path / name, period_ticks=10)
        kf.write(0, {0: box((0, 0, 0), (0.008, 0.008, 0.008))}, fields={"x_K": np.linspace(290, 300, 7)})
    for relative in ("keyframes/kf_0.npz", "keyframes/kf_0.glb", "keyframes/bricks.idx", "keyframes/index.jsonl"):
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()


def test_truncate_to_a_cut(tmp_path):
    kf = K.KeyframeWriter(tmp_path, period_ticks=10, full_every=50)
    kf.write(0, {0: box((0, 0, 0), (0.008, 0.008, 0.008))})
    cut = kf.checkpoint_cut()
    state = json.loads(json.dumps(kf.checkpoint_state()))
    kf.write(10, {0: box((0, 0, 0), (0.008, 0.008, 0.004))}, chips={"m": np.ones(1)})
    kf.write_attributes({0: {"a": np.ones(2)}})
    fresh = K.KeyframeWriter(tmp_path, period_ticks=10, full_every=50)
    fresh.restore_state(state)
    fresh.truncate_outputs(cut)
    assert not (tmp_path / "keyframes" / "kf_1.glb").exists() and not (tmp_path / "chips" / "kf_1.npz").exists()
    assert not (tmp_path / "keyframes" / "attr").exists()
    assert len((tmp_path / "keyframes" / "index.jsonl").read_text().splitlines()) == 1
    assert [e[0] for e in json.loads((tmp_path / "keyframes" / "bricks.idx").read_text())["bricks"]["0"]] == [0]
    entry = fresh.write(10, {0: box((0, 0, 0), (0.008, 0.008, 0.004))})
    assert entry["k"] == 1


def test_brick_keys():
    assert K.brick_key(5) == "5" and K.brick_key((2, 5)) == "w2.5" and K.brick_key("w1.3") == "w1.3"
    assert sorted(["w1.2", "10", "2", "w1.1"], key=K.key_order) == ["2", "10", "w1.1", "w1.2"]
    with pytest.raises(ValueError):
        K.brick_key("../x")
