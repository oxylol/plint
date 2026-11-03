"""Pictures and video of a run directory: ``images/*.png``, ``video.mp4`` and ``render.json``
(docs/models/render.md "Pictures of a run").

* :func:`render_frame` one picture at one simulated time (array), :func:`render_image` the same into a PNG.
* :func:`render_images` the standard set: setup, the cut, the finished part (whole and sectioned) and one
  picture per colour layer the run has data for (temperature, chip pile, coolant reach, surface finish,
  deviation, tool deflection, wear), each at the time that shows it best.
* :func:`render_video` the run (or a time window) at a speed-up, the camera following the tool.
* :func:`render_run` both, plus ``render.json`` (what each file shows: time, camera, layer, colour limits).

All of it is a deterministic function of the run directory and the options.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import layers as L
from . import output as O
from . import params as P
from .compose import CAMERA_MODES, FrameComposer, FrameOptions
from .raster import render
from .rundir import RunData

FRAME_EPS = 1e-9          # a frame starting this close to t1 (in frame periods) is not part of [t0, t1)


def _size(width, height, ssaa, kind: str) -> tuple[int, int, int]:
    p = P.view()
    w = int(p[f"{kind}_width"] if width is None else width)
    h = int(p[f"{kind}_height"] if height is None else height)
    s = int(p[f"{kind}_ssaa"] if ssaa is None else ssaa)
    return w, h, s


def _run(run) -> RunData:
    return run if isinstance(run, RunData) else RunData(run)


def render_frame(run, time_s: float, options: FrameOptions | None = None, width: int | None = None,
                 height: int | None = None, ssaa: int | None = None, composer: FrameComposer | None = None):
    """``(image uint8 [H, W, 3], FrameInfo, Camera)`` of the run at ``time_s``."""
    data = _run(run)
    w, h, s = _size(width, height, ssaa, "image")
    o = replace(options or FrameOptions(), aspect=w / h, height_px=h * s)
    comp = composer or FrameComposer(data)
    tick = int(round(time_s / data.tick_s))
    scene, cam, info = comp.compose(tick, o)
    result = render(scene, cam, w, h, ssaa=s, shadow_focus=info.shadow_focus)
    info.counts.update(result.stats)
    return result.image, info, cam


def render_image(run, time_s: float, out, options: FrameOptions | None = None, width: int | None = None,
                 height: int | None = None, ssaa: int | None = None, composer: FrameComposer | None = None) -> dict:
    image, info, cam = render_frame(run, time_s, options, width, height, ssaa, composer)
    O.write_png(out, image)
    o = options or FrameOptions()
    return {"time_s": info.time_s, "tick": info.tick, "keyframe": info.keyframe, "camera": o.camera,
            "layer": o.layer, "section": None if o.section is None else [list(map(float, o.section[0])),
                                                                         list(map(float, o.section[1]))],
            "tool": info.tool_number, "notes": list(info.notes), "counts": dict(info.counts),
            "camera_detail": cam.to_dict(), "width": int(image.shape[1]), "height": int(image.shape[0])}


# ----------------------------------------------------------------------------- choosing times
def _end_time(data: RunData) -> float:
    return max(data.end_tick - data.ipo_ticks, 0) * data.tick_s


def _cut_time(data: RunData) -> float | None:
    """Middle of the first block that cuts for most of its time (timeline ``ticks.cutting``)."""
    for block in data.timeline:
        ticks = block.get("ticks") or {}
        duration = int(block.get("tick1", 0)) - int(block.get("tick0", 0))
        if duration > 0 and 2 * int(ticks.get("cutting", 0)) > duration:
            return 0.5 * (int(block["tick0"]) + int(block["tick1"])) * data.tick_s
    return None


def _keyframe_max(data: RunData, score) -> int | None:
    best, best_k = -math.inf, None
    for k in range(len(data.keyframes)):
        value = score(k)
        if value is not None and np.isfinite(value) and value > best:
            best, best_k = float(value), k
    return best_k


def _kf_time(data: RunData, k: int | None, default: float) -> float:
    return float(data.keyframes[k]["tick"]) * data.tick_s if k is not None else default


def _temperature_score(data: RunData):
    def score(k):
        fields = data.fields(k)
        for key in ("work_temp_K", "surface_temp_K"):
            if key in fields and np.asarray(fields[key]).size:
                return float(np.nanmax(np.asarray(fields[key], dtype=np.float64)))
        return None
    return score


def _section_through_part(composer: FrameComposer, data: RunData, tick: int):
    scene, _, _ = composer.compose(tick, FrameOptions(camera="part", labels=False, chips=False, coolant=False,
                                                      cues=False))
    part = next((m for m in scene.meshes if m.name == "part"), None)
    if part is None or part.bounds() is None:
        return None
    lo, hi = part.bounds()
    return (0.5 * (lo + hi), np.array([0.0, 1.0, 0.0]))


def standard_images(data: RunData, composer: FrameComposer) -> list[tuple[str, float, FrameOptions, str]]:
    """``(file name, time_s, options, caption)`` of the standard pictures this run has data for."""
    end = _end_time(data)
    op = (data.timeline[-1].get("op") if data.timeline else None) or ""
    part_name = f"part_{op}" if op else "part"
    first_tool = next((int(b["tick1"]) for b in data.timeline if b.get("kind") in ("tool_change", "turret_index")), 0)
    items = [("setup.png", first_tool * data.tick_s, FrameOptions(camera="setup", chips=False, coolant=False, cues=False,
                                                                  finding_hold_s=0.0),
              "Setup: stock, fixture and the first tool")]
    cut = _cut_time(data)
    if cut is not None:
        items.append(("cut.png", cut, FrameOptions(camera="follow"), "The tool in the cut"))
    items.append((f"{part_name}.png", end, FrameOptions(camera="part", chips=False, coolant=False, cues=False,
                                                        finding_hold_s=0.0), f"Finished part after {op or 'the run'}"))
    section = _section_through_part(composer, data, int(round(end / data.tick_s)))
    if section is not None:
        items.append((f"{part_name}_section.png", end, FrameOptions(camera="part", chips=False, coolant=False,
                                                                  cues=False, section=section, finding_hold_s=0.0),
                      f"Finished part after {op or 'the run'}, sectioned at its middle"))
    k_hot = _keyframe_max(data, _temperature_score(data))
    if k_hot is not None:
        items.append(("temperature.png", _kf_time(data, k_hot, cut or 0.0),
                      FrameOptions(camera="follow", layer="temperature", pile=False),
                      "Temperatures at the hottest keyframe"))
    k_pile = _keyframe_max(data, lambda k: (lambda g: float(np.nanmax(g.height_m)) if g is not None else None)(
        L.pile_grid(data.fields(k), data.chips(k))))
    if k_pile is not None:
        items.append(("chip_pile.png", _kf_time(data, k_pile, end), FrameOptions(camera="part", layer="chip_pile"),
                      "Settled chips at the keyframe with the highest pile"))
    k_wet = _keyframe_max(data, lambda k: float(np.size(data.fields(k).get("jet_origin_M_m", []))) or None)
    if k_wet is not None:
        items.append(("coolant.png", _kf_time(data, k_wet, cut or 0.0), FrameOptions(camera="setup", pile=False,
                                                                                    layer="coolant_reach"),
                      "Coolant streams and the wetted surface"))
    k_defl = _keyframe_max(data, lambda k: L.tool_deflection(data.fields(k)))
    if k_defl is not None:
        items.append(("deflection.png", _kf_time(data, k_defl, cut or 0.0), FrameOptions(camera="follow", pile=False,
                                                                                        layer="deflection"),
                      "Tool deflection at the keyframe where it is largest"))
    attrs = data.ipw(len(data.keyframes) - 1).attrs if data.keyframes else {}
    for name, layer, caption in (("deviation.png", "deviation", "Deviation of the finished surface from nominal"),
                                 ("finish.png", "surface_finish", "Surface finish Ra of the finished surface"),
                                 ("wear.png", "wear", "Flank wear of the edge that generated each surface")):
        key = {"deviation": "deviation_m", "surface_finish": "Ra_m", "wear": "generating_VB_m"}[layer]
        if key in attrs and np.any(np.isfinite(attrs[key])):
            items.append((name, end, FrameOptions(camera="part", layer=layer, chips=False, coolant=False, cues=False,
                                                  finding_hold_s=0.0), caption))
    return items


def render_images(run, out_dir=None, width: int | None = None, height: int | None = None,
                  ssaa: int | None = None) -> list[dict]:
    """Write the standard pictures to ``out_dir`` (default ``<run>/images``); returns one entry per picture."""
    data = _run(run)
    folder = Path(out_dir) if out_dir is not None else data.path / "images"
    composer = FrameComposer(data)
    entries = []
    for name, time_s, options, caption in standard_images(data, composer):
        if options.layer is not None and options.colour_range is None:
            options = replace(options, colour_range=L.run_range(data, options.layer))
        entry = render_image(data, time_s, folder / name, options, width, height, ssaa, composer)
        if entry["counts"].get("opaque_faces", 0) == 0:
            (folder / name).unlink()             # nothing of the run to show: no picture
            continue
        entry.update({"file": f"{folder.name}/{name}" if out_dir is None else str(folder / name), "caption": caption})
        entries.append(entry)
    return entries


# ----------------------------------------------------------------------------- video
def video_settings(data: RunData, speed: float | None, fps: float | None) -> tuple[float, float]:
    """Speed-up and frame rate: arguments, else the project's ``outputs.video``, else ``render.yaml``."""
    project_video = ((data.project.get("outputs") or {}).get("video") or {}) if data.project else {}
    p = P.view()
    s = float(speed if speed is not None else project_video.get("speed", p["video_speed"]))
    f = float(fps if fps is not None else project_video.get("fps", p["video_frame_rate_Hz"]))
    if s <= 0.0 or f <= 0.0:
        raise ValueError("video speed and frame rate must be positive")
    return s, f


def frame_times(t0_s: float, t1_s: float, speed: float, fps: float) -> np.ndarray:
    """Simulated times of the video frames: t0 + i speed / fps for the frames that start before t1 (at least one),
    so [t0, t1) plays for (t1 - t0) / speed seconds."""
    count = max(int(math.ceil((t1_s - t0_s) * fps / speed - FRAME_EPS)), 1)
    return t0_s + np.arange(count) * (speed / fps)


def render_video(run, out=None, speed: float | None = None, fps: float | None = None, t0_s: float = 0.0,
                 t1_s: float | None = None, options: FrameOptions | None = None, width: int | None = None,
                 height: int | None = None, ssaa: int | None = None) -> dict:
    """Write the video of ``[t0_s, t1_s]`` of simulated time at ``speed`` (simulated seconds per video second)."""
    data = _run(run)
    path = Path(out) if out is not None else data.path / "video.mp4"
    w, h, s = _size(width, height, ssaa, "video")
    speed, fps = video_settings(data, speed, fps)
    end = _end_time(data) if t1_s is None else float(t1_s)
    times = frame_times(float(t0_s), end, speed, fps)
    o = options or FrameOptions(camera="follow")
    hold = float(P.number(P.VIEW, "finding_label_hold_s")) * speed
    o = replace(o, aspect=w / h, height_px=h * s, finding_hold_s=hold if o.finding_hold_s is None else o.finding_hold_s)
    if o.layer is not None and o.colour_range is None:
        o = replace(o, colour_range=L.run_range(data, o.layer))
    composer = FrameComposer(data)
    notes: list[str] = []
    with O.VideoWriter(path, w, h, fps) as video:
        for t in times:
            scene, cam, info = composer.compose(int(round(t / data.tick_s)), o)
            video.write(render(scene, cam, w, h, ssaa=s, shadow_focus=info.shadow_focus).image)
            notes.extend(n for n in info.notes if n not in notes)
    return {"file": path.name if out is None else str(path), "frames": int(len(times)), "fps": fps, "speed": speed,
            "t0_s": float(times[0]), "t1_s": float(times[-1]), "duration_s": len(times) / fps, "width": w,
            "height": h, "camera": o.camera, "layer": o.layer,
            "colour_range": None if o.colour_range is None else [float(v) for v in o.colour_range], "notes": notes}


def render_run(run_dir, images: bool = True, video: bool = True, speed: float | None = None,
               fps: float | None = None, layer: str | None = None, image_size: tuple | None = None,
               video_size: tuple | None = None) -> dict:
    """Pictures and video of a finished run into its directory, and ``render.json`` describing them.
    ``image_size`` / ``video_size`` are (width, height, ssaa) overrides of render.yaml."""
    data = _run(run_dir)
    manifest = {"format": "cnctwin-render", "version": 1, "images": [], "video": None}
    if images:
        manifest["images"] = render_images(data, None, *(image_size or (None, None, None)))
    if video:
        manifest["video"] = render_video(data, speed=speed, fps=fps, options=FrameOptions(camera="follow",
                                                                                         layer=layer),
                                         **dict(zip(("width", "height", "ssaa"), video_size or (None, None, None))))
    target = data.path / "render.json"
    tmp = target.with_name(".render.json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return manifest


def write_media(run_dir, images: bool | None = None, video: bool | None = None, audio: bool | None = None) -> dict:
    """The media a finished run asks for in its project's ``outputs`` (``images``, ``video {fps, speed}``,
    ``audio``; arguments override): pictures, video and render.json (this package) and audio.wav + audio.json
    (:func:`cnctwin.sound.render_audio`). Returns ``{"render": manifest, "audio": summary}``."""
    data = _run(run_dir)
    outputs = (data.project.get("outputs") or {}) if data.project else {}
    want_images = bool(outputs.get("images", True)) if images is None else bool(images)
    want_video = bool(outputs.get("video", True)) if video is None else bool(video)
    want_audio = bool(outputs.get("audio", True)) if audio is None else bool(audio)
    result = {"render": None, "audio": None}
    if want_images or want_video:
        result["render"] = render_run(data, images=want_images, video=want_video)
    if want_audio:
        from ..sound import render_audio

        result["audio"] = render_audio(data.path)
    return result


__all__ = ["render_frame", "render_image", "render_images", "render_video", "render_run", "write_media",
           "standard_images", "frame_times", "video_settings", "CAMERA_MODES"]
