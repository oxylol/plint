"""Cameras: look-at view, perspective and orthographic projections (OpenGL conventions), framing of a
bounding box, and the camera that follows the tool (docs/models/render.md "Cameras").

The view frame looks along -z_view with +y_view up; clip space is ``-w <= x, y, z <= w``. A camera is
plain data (eye, target, up, projection, field of view or view height, near/far), so a frame's camera can
be written to a log and rebuilt exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import params as P

PERSPECTIVE = "perspective"
ORTHOGRAPHIC = "orthographic"
# depth range of the follow camera, in view heights: near at 2 % of the eye distance (depth resolution of the
# float64 z-buffer is no concern), far 20 view heights behind the tip (the machine table and the enclosure)
FOLLOW_NEAR_FRACTION = 0.02
FOLLOW_DEPTH_RANGE_HEIGHTS = 20.0


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length direction")
    return v / n


def look_at(eye, target, up) -> np.ndarray:
    """World -> view matrix (4 x 4)."""
    eye = np.asarray(eye, dtype=np.float64)
    f = _unit(np.asarray(target, dtype=np.float64) - eye)
    up = _unit(up)
    if abs(float(f @ up)) > 1.0 - 1e-9:                 # looking along up: pick another up
        up = _unit(np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    s = _unit(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(0.5 * fov_y_rad)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(height: float, aspect: float, near: float, far: float) -> np.ndarray:
    hh = 0.5 * height
    hw = hh * aspect
    m = np.eye(4)
    m[0, 0] = 1.0 / hw
    m[1, 1] = 1.0 / hh
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def direction(azimuth_rad: float, elevation_rad: float, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Unit vector from the target toward the eye: azimuth about ``up`` from the frame's first horizontal axis,
    elevation above the horizontal plane."""
    up = _unit(up)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(ref - (ref @ up) * up)
    e2 = np.cross(up, e1)
    ce = math.cos(elevation_rad)
    return ce * math.cos(azimuth_rad) * e1 + ce * math.sin(azimuth_rad) * e2 + math.sin(elevation_rad) * up


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    projection: str = PERSPECTIVE
    fov_y_rad: float = 0.5
    view_height_m: float = 0.1          # orthographic: full height of the view volume
    near_m: float = 1e-3
    far_m: float = 10.0

    def __post_init__(self) -> None:
        self.eye = np.asarray(self.eye, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64)
        self.up = np.asarray(self.up, dtype=np.float64)
        if self.projection not in (PERSPECTIVE, ORTHOGRAPHIC):
            raise ValueError(f"projection {self.projection!r} is not perspective or orthographic")
        if not (0.0 < self.near_m < self.far_m):
            raise ValueError("camera needs 0 < near < far")

    def view(self) -> np.ndarray:
        return look_at(self.eye, self.target, self.up)

    def proj(self, aspect: float) -> np.ndarray:
        if self.projection == PERSPECTIVE:
            return perspective(self.fov_y_rad, aspect, self.near_m, self.far_m)
        return orthographic(self.view_height_m, aspect, self.near_m, self.far_m)

    def matrix(self, aspect: float) -> np.ndarray:
        """World -> clip."""
        return self.proj(aspect) @ self.view()

    def forward(self) -> np.ndarray:
        return _unit(self.target - self.eye)

    def to_dict(self) -> dict:
        return {"eye_m": self.eye.tolist(), "target_m": self.target.tolist(), "up": self.up.tolist(),
                "projection": self.projection, "fov_y_rad": self.fov_y_rad, "view_height_m": self.view_height_m,
                "near_m": self.near_m, "far_m": self.far_m}

    def pixel_size_at(self, point, height_px: int) -> float:
        """World size of one pixel at ``point`` (for minimum on-screen sizes)."""
        if self.projection == ORTHOGRAPHIC:
            return self.view_height_m / height_px
        depth = float(np.dot(np.asarray(point, dtype=np.float64) - self.eye, self.forward()))
        return 2.0 * max(depth, self.near_m) * math.tan(0.5 * self.fov_y_rad) / height_px


def fit(lo, hi, view_dir, up=(0.0, 0.0, 1.0), projection: str = PERSPECTIVE, fov_y_rad: float | None = None,
        aspect: float = 16.0 / 9.0, margin: float | None = None) -> Camera:
    """Camera looking at the box [lo, hi] from ``view_dir`` (target -> eye) so its bounding sphere fills the view
    with ``render.framing_margin``."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    fov = float(P.number(P.VIEW, "follow_fov_rad") if fov_y_rad is None else fov_y_rad)
    margin = float(P.number(P.VIEW, "framing_margin") if margin is None else margin)
    centre = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-6) * margin
    d = _unit(view_dir)
    half = 0.5 * fov if aspect >= 1.0 else math.atan(math.tan(0.5 * fov) * aspect)
    distance = radius / math.sin(half)
    near = max(distance - 2.0 * radius, distance * 1e-3)
    far = distance + 2.0 * radius
    return Camera(centre + distance * d, centre, np.asarray(up, dtype=np.float64), projection, fov,
                  2.0 * radius, near, far)


def follow(tip, view_height_m: float | None = None, azimuth_rad: float | None = None,
           elevation_rad: float | None = None, fov_y_rad: float | None = None, up=(0.0, 0.0, 1.0),
           depth_range_m: float | None = None) -> Camera:
    """Perspective camera aimed at the tool tip, at the distance where the view is ``view_height_m`` high at the
    tip (defaults: ``render.follow_*``)."""
    p = P.view()
    height = float(p["follow_view_height_m"] if view_height_m is None else view_height_m)
    az = float(p["follow_azimuth_rad"] if azimuth_rad is None else azimuth_rad)
    el = float(p["follow_elevation_rad"] if elevation_rad is None else elevation_rad)
    fov = float(p["follow_fov_rad"] if fov_y_rad is None else fov_y_rad)
    tip = np.asarray(tip, dtype=np.float64)
    distance = 0.5 * height / math.tan(0.5 * fov)
    reach = FOLLOW_DEPTH_RANGE_HEIGHTS * height if depth_range_m is None else float(depth_range_m)
    near = distance * FOLLOW_NEAR_FRACTION
    return Camera(tip + distance * direction(az, el, up), tip, np.asarray(up, dtype=np.float64), PERSPECTIVE, fov,
                  height, near, distance + reach)


def smoothed_target(ticks: np.ndarray, tips: np.ndarray, tick: int, window_ticks: int) -> np.ndarray:
    """Mean tool tip over the window centred on ``tick`` (the follow camera's target, deterministic)."""
    if window_ticks <= 0 or len(ticks) == 0:
        index = int(np.clip(np.searchsorted(ticks, tick, side="right") - 1, 0, max(len(ticks) - 1, 0)))
        return np.asarray(tips[index], dtype=np.float64)
    lo = int(np.searchsorted(ticks, tick - window_ticks // 2, side="left"))
    hi = int(np.searchsorted(ticks, tick + window_ticks // 2, side="right"))
    lo = min(max(lo, 0), len(ticks) - 1)
    hi = max(min(hi, len(ticks)), lo + 1)
    return np.asarray(tips[lo:hi], dtype=np.float64).mean(axis=0)
