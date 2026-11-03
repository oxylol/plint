"""Pictures of the twin: a deterministic software rasteriser (Numba z-buffer, no GPU, no external renderer) and
the pictures and video of a run directory (docs/models/render.md).

* :mod:`~cnctwin.render.raster` :func:`render` (scene + camera -> sRGB picture): flat/Gouraud shading, shadow
  maps, section planes with capped cuts, transparency, sprites, labels, colour bars, supersampling.
* :mod:`~cnctwin.render.camera` perspective/orthographic cameras, framing, the camera following the tool.
* :mod:`~cnctwin.render.colour` sRGB, colour maps and colour bars, blackbody colours.
* :mod:`~cnctwin.render.cues` sparks, glow, smoke and steam (the visual cue model).
* :mod:`~cnctwin.render.layers` temperature, chip pile height, coolant reach, surface finish, deviation,
  deflection and wear painted on the part and the tool.
* :mod:`~cnctwin.render.rundir` the run-directory readers (scene, frames.bin, keyframes, series, timeline).
* :mod:`~cnctwin.render.pipeline` :func:`render_run`, :func:`render_images`, :func:`render_video`,
  :func:`render_image`, :func:`render_frame`.

``python -m cnctwin.render RUN_DIR`` writes ``images/*.png``, ``video.mp4`` and ``render.json``;
:func:`write_media` also writes ``audio.wav`` (cnctwin.sound) as the project's ``outputs`` ask.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "render": "raster", "RenderResult": "raster",
    "Camera": "camera", "follow": "camera", "fit": "camera",
    "Scene": "scene", "RenderMesh": "scene", "Material": "scene", "Sprites": "scene", "Label": "scene",
    "ColorBar": "scene", "section_plane": "scene",
    "RunData": "rundir",
    "FrameComposer": "compose", "FrameOptions": "compose",
    "render_run": "pipeline", "render_images": "pipeline", "render_video": "pipeline", "render_image": "pipeline",
    "render_frame": "pipeline", "write_media": "pipeline",
    "write_png": "output", "png_bytes": "output", "VideoWriter": "output",
    "LAYERS": "layers",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
