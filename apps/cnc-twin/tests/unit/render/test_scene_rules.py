"""Scene rules on a run directory made by hand: coolant streams stop at the first obstacle (the coolant shadow),
objects in M are placed with pose_M_S, the tool follows frames.bin, glowing chips become sparks, hot wet surfaces
steam."""

import json
import math

import numpy as np
import pytest

from cnctwin.geom.io.gltf import GlbNode, glb_bytes
from cnctwin.geom.primitives import box_mesh, cylinder_mesh
from cnctwin.render.compose import FrameComposer, FrameOptions
from cnctwin.render.rundir import RunData

FRAME_DTYPE = np.dtype([("tick", "<i8"), ("axis", "<f4", (3,)), ("spindle_angle_rad", "<f4", (1,)),
                        ("spindle_rev", "<i4", (1,)), ("tip_S_m", "<f4", (1, 3)), ("axis_T_S", "<f4", (1, 3))])


def header():
    fields = []
    for name in FRAME_DTYPE.names:
        sub, offset = FRAME_DTYPE.fields[name][:2]
        base = sub.base if sub.shape else sub
        fields.append({"name": name, "dtype": base.str, "shape": list(sub.shape), "offset": int(offset)})
    return {"record_bytes": FRAME_DTYPE.itemsize, "fields": fields, "tick_s": 1e-6, "ipo_ticks": 1000,
            "axes": [{"name": n, "kind": "linear"} for n in "XYZ"]}


def write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture()
def hand_run(tmp_path):
    run = tmp_path / "hand"
    (run / "scene").mkdir(parents=True)
    (run / "scene" / "stock.glb").write_bytes(glb_bytes([GlbNode("stock", box_mesh([0, 0, 0], [0.04, 0.03, 0.02]))]))
    (run / "scene" / "tool.glb").write_bytes(glb_bytes([GlbNode("tool", cylinder_mesh([0, 0, 0], [0, 0, 0.03], 0.004))]))
    (run / "scene" / "cover.glb").write_bytes(glb_bytes([GlbNode("cover", box_mesh([-0.1, -0.1, -0.1], [0.1, 0.1, -0.09]))]))
    scene = {"format": "cnctwin-scene", "stock": {"mesh": "scene/stock.glb", "pose_W_S": np.eye(4).tolist()},
             "tools": [{"number": 3, "envelope": "scene/tool.glb", "diameter_m": 0.008, "length_m": 0.03, "teeth": 3}],
             "machine": {"components": [{"id": "cover", "mesh": "scene/cover.glb", "frame": "M", "kind": "machine"}]},
             "frames": header()}
    (run / "scene.json").write_text(json.dumps(scene))
    frames = np.zeros(3, FRAME_DTYPE)
    frames["tick"] = [0, 1000, 2000]
    frames["tip_S_m"][:, 0, :] = [[0.02, 0.015, 0.03], [0.02, 0.015, 0.025], [0.02, 0.015, 0.021]]
    frames["axis_T_S"][:, 0, :] = [0.0, 0.0, 1.0]
    (run / "frames.bin").write_bytes(frames.tobytes())
    (run / "timeline.json").write_text(json.dumps({"blocks": [{"index": 0, "tick0": 0, "tick1": 3000, "tool": 3,
                                                               "line": 1, "block": "G01 Z21.", "op": "OP1"}]}))
    (run / "keyframes").mkdir()
    (run / "keyframes" / "index.jsonl").write_text(json.dumps({"k": 0, "tick": 0, "time_s": 0.0}) + "\n")
    pose_M_S = np.eye(4)
    pose_M_S[:3, 3] = [0.5, 0.0, 0.0]                      # S sits 0.5 m along +x of M
    write_npz(run / "keyframes" / "kf_0.npz", {
        "pose_M_S": pose_M_S,
        "jet_nozzle_id": np.array(["n1"]), "jet_kind": np.array(["flood"]),
        "jet_origin_M_m": np.array([[0.52, 0.015, 0.12]]),  # 0.1 m above the stock top at x_S = 0.02
        "jet_direction_M": np.array([[0.0, 0.0, -1.0]]), "jet_speed_m_s": np.array([3.0]),
        "jet_flow_m3_s": np.array([1e-4]), "jet_half_angle_rad": np.array([0.05]),
        "surface_origin_S_m": np.array([0.0, 0.0, 0.02]), "surface_cell_m": np.array(0.01),
        "surface_temp_K": np.full((3, 4), 420.0)})
    write_npz(run / "chips" / "kf_0.npz", {"pos_M_m": np.array([[0.53, 0.0, 0.05], [0.53, 0.01, 0.05]]),
                                           "vel_M_m_s": np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
                                           "temp_K": np.array([1100.0, 350.0]), "mass_kg": np.array([1e-6, 1e-6]),
                                           "colour_rgb": np.array([[200, 200, 210], [90, 60, 200]], np.uint8)})
    (run / "run.json").write_text(json.dumps({"project": {"name": "hand", "material": {"family": "steel"},
                                                          "coolant": {"fluid": {"kind": "emulsion"}}},
                                              "clock": {"tick_s": 1e-6, "ipo_ticks": 1000}, "seed": 5}))
    return run


def test_coolant_stream_stops_at_the_tool_or_part_it_hits(hand_run):
    composer = FrameComposer(RunData(hand_run))
    for tick, top in ((0, 0.03), (2000, 0.021)):
        scene, _, info = composer.compose(tick, FrameOptions(camera="setup"))
        jet = next(m for m in scene.meshes if m.name == "jet_0")
        end = jet.vertices[len(jet.vertices) // 2:]           # the far ring
        # the stream ends on the tool tip (the tool sits right under the nozzle), never inside it
        assert end[:, 2].mean() == pytest.approx(top + 0.03, abs=2e-4)
        assert info.counts["jets"] == 1
    assert jet.transparent and np.all(jet.vertex_alpha <= 1.0)


def test_objects_in_M_follow_pose_M_S_and_the_tool_follows_the_frames(hand_run):
    composer = FrameComposer(RunData(hand_run))
    scene, _, _ = composer.compose(1500, FrameOptions(camera="setup"))
    cover = next(m for m in scene.meshes if m.name == "cover")
    assert cover.vertices[:, 0].min() == pytest.approx(-0.6)            # M x -0.1 is S x -0.6
    tool = next(m for m in scene.meshes if m.name == "tool")
    assert tool.vertices[:, 2].min() == pytest.approx(0.025, abs=1e-6)  # frame at tick 1000 (zero-order hold)
    chips = next(m for m in scene.meshes if m.name == "chips")
    assert chips.vertices[:, 0].mean() == pytest.approx(0.03, abs=1e-3)
    sparks = [s for s in scene.sprites if s.name == "sparks"]
    assert len(sparks) == 1 and len(sparks[0]) == 1                     # only the 1100 K chip glows
    glow = chips.face_emission.reshape(2, 8, 3)
    assert glow[0].max() > 0.0 and glow[1].max() == 0.0


def test_hot_wet_surfaces_steam_and_dry_runs_do_not(hand_run):
    composer = FrameComposer(RunData(hand_run))
    scene, _, _ = composer.compose(0, FrameOptions(camera="setup"))
    steam = [s for s in scene.sprites if s.name == "steam"]
    assert steam and len(steam[0]) > 0                                   # 420 K wet surface: above 373 K boiling
    assert all(s.name != "smoke" for s in scene.sprites)                 # 420 K is below the oil smoke point
    data = RunData(hand_run)
    data.run["project"]["coolant"]["fluid"]["kind"] = "dry"
    dry = FrameComposer(data)
    scene, _, _ = dry.compose(0, FrameOptions(camera="setup"))
    assert all(s.name not in ("steam", "smoke") for s in scene.sprites)


def test_runs_without_pose_M_S_leave_M_objects_out_and_say_so(hand_run):
    fields = dict(np.load(hand_run / "keyframes" / "kf_0.npz"))
    del fields["pose_M_S"]
    write_npz(hand_run / "keyframes" / "kf_0.npz", fields)
    composer = FrameComposer(RunData(hand_run))
    scene, _, info = composer.compose(0, FrameOptions(camera="setup"))
    names = {m.name for m in scene.meshes}
    assert "cover" not in names and "chips" not in names and "jet_0" not in names
    assert any("pose_M_S" in note for note in info.notes)
    assert {"part", "tool"} <= names
