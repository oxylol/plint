"""Pictures and video of a run directory: the readers, PNG and MP4 determinism, a 2-second video, the standard
picture set and render.json."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from cnctwin.render import output as O
from cnctwin.render import pipeline
from cnctwin.render.compose import FrameComposer, FrameOptions
from cnctwin.render.rundir import RunData

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]


def test_run_reader_reads_the_synthetic_run(run_dir):
    run = RunData(run_dir)
    assert len(run.frames) == 2320 and run.tick_s == 1e-6 and run.ipo_ticks == 1000
    tips = run.tip_S()
    assert tips.shape == (2320, 3) and np.allclose(run.axis_T_S()[0], [0.0, 0.0, 1.0])
    angle = run.spindle_angle()
    assert np.all(np.diff(angle[1000:1100]) < 0.0)                       # M03: negative about +z_T, unwrapped
    ipw0, ipw4 = run.ipw(0), run.ipw(4)
    assert ipw0.faces.shape == (72, 3) and set(ipw4.attrs) == {"Ra_m", "deviation_m", "generating_VB_m"}
    assert np.all(np.isfinite(ipw4.attrs["deviation_m"]))
    assert ipw4.vertices[:, 2].max() == pytest.approx(0.015, abs=1e-6)
    lo = ipw4.vertices.min(axis=0)
    assert np.allclose(lo, [0.0, 0.0, 0.0], atol=1e-6)
    pose, known = run.pose_M_S(1_200_000)
    assert known and np.allclose(pose[:3, 3], [-0.391, -0.2122, -0.4214])
    assert run.material_family() == "aluminium"
    coolant = run.coolant()
    assert coolant["kind"] == "emulsion" and coolant["boiling_K"] == pytest.approx(373.15)
    kinds = {c["id"]: (c["frame"], c["kind"]) for c in run.components()}
    assert kinds["spindle"] == ("T", "spindle") and kinds["lang-makro-grip-77"] == ("S", "fixture")
    assert kinds["door"] == ("S", "door")
    data = run.series("macro", ["dynamics.chatter_freq_Hz", "nope"], 1_100_000, 1_400_000)
    assert "nope" not in data and data["dynamics.chatter_freq_Hz"].max() > 1000.0
    assert run.tool_number_at(1_000) is None and run.tool_number_at(1_000_000) == 1


def test_png_bytes_are_deterministic_within_and_across_processes(run_dir, tmp_path):
    options = FrameOptions(camera="follow", layer="temperature")
    first, _, _ = pipeline.render_frame(run_dir, 1.2, options, width=160, height=90, ssaa=2)
    second, _, _ = pipeline.render_frame(run_dir, 1.2, options, width=160, height=90, ssaa=2)
    assert np.array_equal(first, second)
    assert O.png_bytes(first) == O.png_bytes(second)
    digest = hashlib.sha256(O.png_bytes(first)).hexdigest()
    code = ("import hashlib, sys; from cnctwin.render import pipeline, output; "
            "from cnctwin.render.compose import FrameOptions; "
            f"img, _, _ = pipeline.render_frame({str(run_dir)!r}, 1.2, FrameOptions(camera='follow', "
            "layer='temperature'), width=160, height=90, ssaa=2); "
            "print(hashlib.sha256(output.png_bytes(img)).hexdigest())")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), check=True)
    assert result.stdout.strip() == digest
    assert len(np.unique(first.reshape(-1, 3), axis=0)) > 100                 # a picture, not a blank frame


def test_two_second_video_is_fifty_frames_and_repeatable(run_dir, tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    info = pipeline.render_video(run_dir, a, speed=1.0, fps=25.0, t0_s=0.2, t1_s=2.2, width=192, height=108, ssaa=1)
    assert info["frames"] == 50 and info["duration_s"] == pytest.approx(2.0)
    assert a.read_bytes()[4:8] == b"ftyp" and O.probe_frames(a) == 50
    pipeline.render_video(run_dir, b, speed=1.0, fps=25.0, t0_s=0.2, t1_s=2.2, width=192, height=108, ssaa=1)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    fast = pipeline.render_video(run_dir, tmp_path / "fast.mp4", speed=10.0, fps=25.0, width=96, height=54, ssaa=1)
    assert fast["frames"] == 6 and fast["speed"] == 10.0                     # 2.32 s simulated in 0.24 s of video


def test_frame_times_and_video_settings(run_dir):
    assert len(pipeline.frame_times(0.0, 2.0, 1.0, 25.0)) == 50
    assert len(pipeline.frame_times(0.0, 0.0, 1.0, 25.0)) == 1
    run = RunData(run_dir)
    assert pipeline.video_settings(run, None, None) == (10.0, 25.0)            # render.yaml defaults
    assert pipeline.video_settings(run, 2.0, 30.0) == (2.0, 30.0)
    with pytest.raises(ValueError):
        pipeline.video_settings(run, 0.0, 25.0)


def test_standard_pictures_and_render_json(run_dir, tmp_path):
    manifest = pipeline.render_run(run_dir, images=True, video=False, image_size=(256, 144, 1))
    names = {entry["file"].split("/")[-1] for entry in manifest["images"]}
    assert {"setup.png", "cut.png", "part_OP10.png", "part_OP10_section.png", "temperature.png", "chip_pile.png",
            "coolant.png", "deflection.png", "deviation.png", "finish.png", "wear.png"} <= names
    document = json.loads((run_dir / "render.json").read_text())
    assert document["format"] == "cnctwin-render" and len(document["images"]) == len(manifest["images"])
    from PIL import Image

    section = np.asarray(Image.open(run_dir / "images" / "part_OP10_section.png"))
    cap = (section[..., 0] > 150) & (section[..., 1] < 110) & (section[..., 2] < 110)
    assert cap.sum() > 50                                                     # the cut fixture shows its cap colour
    deviation = next(e for e in manifest["images"] if e["file"].endswith("deviation.png"))
    assert deviation["layer"] == "deviation" and deviation["camera"] == "part"
    for entry in manifest["images"]:
        picture = np.asarray(Image.open(run_dir / entry["file"]))
        assert picture.shape == (144, 256, 3) and picture.std() > 5.0
    setup = next(e for e in manifest["images"] if e["file"].endswith("setup.png"))
    assert setup["tool"] == 1                                                 # after the first tool change


def test_layers_paint_the_part_and_the_tool(run_dir):
    run = RunData(run_dir)
    composer = FrameComposer(run)
    scene, cam, info = composer.compose(1_500_000, FrameOptions(camera="follow", layer="temperature"))
    part = next(m for m in scene.meshes if m.name == "part")
    tool = next(m for m in scene.meshes if m.name == "tool")
    assert part.vertex_srgb is not None and part.vertex_srgb.shape == (len(part.vertices), 3)
    assert scene.colorbars and scene.colorbars[0].unit == "degC"
    assert tool.material.srgb != pytest.approx(tuple(composer.view["tool_srgb"]))   # coloured by edge temperature
    assert info.keyframe == 3 and info.tool_number == 1 and info.counts["chips"] == 24
    scene, _, info = composer.compose(2_319_000, FrameOptions(camera="part", layer="deviation", chips=False))
    bar = scene.colorbars[0]
    assert bar.vmin == pytest.approx(-bar.vmax) and bar.unit == "um"
    assert any(label.text.startswith("error: pocket-depth") for label in scene.labels)


def test_orthographic_pictures_and_the_media_entry_point(run_dir):
    image, info, cam = pipeline.render_frame(run_dir, 2.3, FrameOptions(camera="part", projection="orthographic"),
                                             width=128, height=72, ssaa=1)
    assert cam.projection == "orthographic" and image.std() > 5.0
    _, _, follow = pipeline.render_frame(run_dir, 1.2, FrameOptions(camera="follow", projection="orthographic"),
                                         width=128, height=72, ssaa=1)
    assert follow.projection == "orthographic" and follow.view_height_m == pytest.approx(0.07)
    with pytest.raises(ValueError):
        pipeline.render_frame(run_dir, 1.0, FrameOptions(camera="drone"), width=64, height=36, ssaa=1)
    result = pipeline.write_media(run_dir, images=False, video=False, audio=True)
    assert result["render"] is None and result["audio"]["sample_rate_Hz"] == 44100.0
    assert (run_dir / "audio.wav").read_bytes()[:4] == b"RIFF" and (run_dir / "audio.json").exists()
