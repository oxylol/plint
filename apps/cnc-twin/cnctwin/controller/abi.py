"""Array layouts of the controller's kernels (cnctwin/controller/kernels.py).

The controller takes no part in the compiled hot loop (its ``kernel_inputs`` is None); its kernels
serve the geometry of cycles that expand into many moves (Haas G150 general pocket milling).

``Boundary`` (float64 [n, 4], C-contiguous): one row per straight segment of a closed contour in
the pocket plane, columns ``SEG_X0 SEG_Y0 SEG_X1 SEG_Y1`` (SI, program frame). Arcs are given as
chords (``boundary_from_contour`` discretises them with a sagitta bound).

Interval output: two float64 [capacity] arrays ``lo``/``hi`` of the allowed x intervals on one
raster line, and the returned count (``-1`` when the capacity was too small).
"""

from __future__ import annotations

import math

import numpy as np

SEG_X0, SEG_Y0, SEG_X1, SEG_Y1 = 0, 1, 2, 3
SEG_WIDTH = 4


def boundary(points: list[tuple[float, float]]) -> np.ndarray:
    """Closed polyline (first point not repeated) -> segment array."""
    n = len(points)
    seg = np.empty((n, SEG_WIDTH))
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        seg[i] = (x0, y0, x1, y1)
    return np.ascontiguousarray(seg)


def arc_chords(centre: tuple[float, float], start: tuple[float, float], sweep: float,
               sagitta_m: float) -> list[tuple[float, float]]:
    """Points along an arc (end point included, start excluded) with chord sagitta <= ``sagitta_m``."""
    radius = math.hypot(start[0] - centre[0], start[1] - centre[1])
    if radius <= 0.0:
        return []
    if sagitta_m >= radius:
        step = math.pi / 2.0
    else:
        step = 2.0 * math.acos(1.0 - sagitta_m / radius)
    count = max(1, int(math.ceil(abs(sweep) / step)))
    angle0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    return [(centre[0] + radius * math.cos(angle0 + sweep * k / count),
             centre[1] + radius * math.sin(angle0 + sweep * k / count)) for k in range(1, count + 1)]
