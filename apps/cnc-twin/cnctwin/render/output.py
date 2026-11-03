"""Writing pictures and videos deterministically (docs/architecture.md 9, docs/models/render.md "Files").

* PNG through Pillow: RGB, no text chunks, no time stamp, fixed compression level; the same pixels give the
  same bytes.
* MP4 (H.264, yuv420p, ``+faststart``) through FFmpeg (imageio-ffmpeg's build, else ``/usr/bin/ffmpeg``) with
  ``-fflags +bitexact -flags +bitexact -map_metadata -1``, ``SOURCE_DATE_EPOCH`` fixed and one encoder thread:
  the same frames give the same bytes with the same FFmpeg build. Frames are streamed to FFmpeg's stdin, so
  memory does not grow with the video length. Files are written to a temporary name and renamed.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from . import params as P

SOURCE_DATE_EPOCH = "315532800"          # 1980-01-01, the zip epoch used by every container of a run
PNG_COMPRESS_LEVEL = 6


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB").save(
        buffer, format="PNG", optimize=False, compress_level=PNG_COMPRESS_LEVEL)
    return buffer.getvalue()


def write_png(path, rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(png_bytes(rgb))
    os.replace(tmp, path)
    return path


def ffmpeg_exe() -> str:
    """imageio-ffmpeg's FFmpeg, else the system one."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        exe = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        if not Path(exe).exists():
            raise RuntimeError("no FFmpeg: install imageio-ffmpeg or /usr/bin/ffmpeg") from None
        return exe


class VideoWriter:
    """``with VideoWriter(path, width, height, fps) as video: video.write(frame)`` (uint8 RGB frames)."""

    def __init__(self, path, width: int, height: int, fps: float, crf: int | None = None,
                 preset: str | None = None) -> None:
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p needs even frame sizes")
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.crf = int(P.integer(P.VIEW, "video_crf") if crf is None else crf)
        self.preset = str(P.text(P.VIEW, "video_preset") if preset is None else preset)
        self.frames = 0
        self._proc = None
        self._tmp = self.path.with_name(f".{self.path.stem}.tmp.mp4")

    def command(self) -> list[str]:
        return [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:g}",
                "-i", "-", "-an", "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
                "-pix_fmt", "yuv420p", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
                "-fflags", "+bitexact", "-flags", "+bitexact", "-map_metadata", "-1",
                "-movflags", "+faststart", "-f", "mp4", "-y", str(self._tmp)]

    def __enter__(self) -> "VideoWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH)
        self._proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE, env=env)
        return self

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame shape {frame.shape} is not {(self.height, self.width, 3)}")
        self._proc.stdin.write(frame.tobytes())
        self.frames += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        proc = self._proc
        proc.stdin.close()
        error = proc.stderr.read().decode(errors="replace")
        code = proc.wait()
        proc.stderr.close()
        if exc_type is not None:
            self._tmp.unlink(missing_ok=True)
            return
        if code != 0 or self.frames == 0:
            self._tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed ({code}, {self.frames} frames): {error.strip()}")
        os.replace(self._tmp, self.path)


def probe_frames(path) -> int:
    """Number of video frames of a file (decoding with FFmpeg; for tests and checks)."""
    result = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0",
                             "-f", "null", "-"], capture_output=True, check=False)
    text = result.stderr.decode(errors="replace")
    frames = 0
    for token in text.replace("\r", "\n").split("\n"):
        if "frame=" in token:
            part = token.split("frame=")[-1].strip().split()[0]
            if part.isdigit():
                frames = int(part)
    return frames
