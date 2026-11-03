"""Compiled geometry kernels of the controller (architecture 5.5 rules) and their Python references.

``raster_intervals(seg, y, d, lo, hi)``: the x intervals of the line ``y`` that lie inside the closed
contour ``seg`` (abi.Boundary) and at a distance of at least ``d`` from every segment, i.e. where
the centre of a tool of radius d (plus finishing allowance) may be. Inside = even-odd crossing rule
with half-open edges; the forbidden set of each segment is the intersection of the line with the
segment's capsule (Minkowski sum with a disc of radius d), which is convex, hence one interval.
Exact for the polyline (no sampling); arcs of the contour are chords with a stated sagitta.

``segment_clearance(seg, ax, ay, bx, by)``: the smallest distance between the segment A-B and the
contour (0 when they cross), used to decide whether a zig-zag link may stay at depth.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_SIG_RASTER = "int64(float64[:, ::1], float64, float64, float64[::1], float64[::1])"
_SIG_CLEAR = "float64(float64[:, ::1], float64, float64, float64, float64)"
_MIN_LENGTH_M = 1e-12


@njit(_SIG_RASTER, cache=True, nogil=True, error_model="numpy")
def raster_intervals(seg, y, d, lo_out, hi_out):
    n = seg.shape[0]
    cross = np.empty(n)
    n_cross = 0
    for i in range(n):
        x0, y0, x1, y1 = seg[i, 0], seg[i, 1], seg[i, 2], seg[i, 3]
        if (y0 <= y < y1) or (y1 <= y < y0):
            cross[n_cross] = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            n_cross += 1
    if n_cross < 2:
        return 0
    inside = np.sort(cross[:n_cross])
    f_lo = np.empty(n)
    f_hi = np.empty(n)
    n_f = 0
    for i in range(n):
        x0, y0, x1, y1 = seg[i, 0], seg[i, 1], seg[i, 2], seg[i, 3]
        lo = np.inf
        hi = -np.inf
        for k in range(2):
            px = x0 if k == 0 else x1
            py = y0 if k == 0 else y1
            dy = y - py
            if abs(dy) <= d:
                half = math.sqrt(d * d - dy * dy)
                if px - half < lo:
                    lo = px - half
                if px + half > hi:
                    hi = px + half
        dx_s = x1 - x0
        dy_s = y1 - y0
        length = math.sqrt(dx_s * dx_s + dy_s * dy_s)
        if length > 0.0:
            ux = dx_s / length
            uy = dy_s / length
            a = -np.inf
            b = np.inf
            k_n = uy * x0 + (y - y0) * ux
            if uy == 0.0:
                if abs(k_n) > d:
                    a = np.inf
            else:
                p = (k_n - d) / uy
                q = (k_n + d) / uy
                a = max(a, min(p, q))
                b = min(b, max(p, q))
            t0 = (y - y0) * uy
            if ux == 0.0:
                if t0 < 0.0 or t0 > length:
                    a = np.inf
            else:
                p = x0 + (0.0 - t0) / ux
                q = x0 + (length - t0) / ux
                a = max(a, min(p, q))
                b = min(b, max(p, q))
            if a <= b:
                if a < lo:
                    lo = a
                if b > hi:
                    hi = b
        if lo <= hi:
            f_lo[n_f] = lo
            f_hi[n_f] = hi
            n_f += 1
    order = np.argsort(f_lo[:n_f], kind="mergesort")
    m_lo = np.empty(n_f + 1)
    m_hi = np.empty(n_f + 1)
    n_m = 0
    for j in range(n_f):
        i = order[j]
        if n_m > 0 and f_lo[i] <= m_hi[n_m - 1]:
            if f_hi[i] > m_hi[n_m - 1]:
                m_hi[n_m - 1] = f_hi[i]
        else:
            m_lo[n_m] = f_lo[i]
            m_hi[n_m] = f_hi[i]
            n_m += 1
    count = 0
    capacity = lo_out.shape[0]
    for j in range(0, n_cross - 1, 2):
        start = inside[j]
        end = inside[j + 1]
        cursor = start
        for k in range(n_m):
            if m_hi[k] <= cursor:
                continue
            if m_lo[k] >= end:
                break
            if m_lo[k] - cursor > _MIN_LENGTH_M:
                if count >= capacity:
                    return -1
                lo_out[count] = cursor
                hi_out[count] = m_lo[k]
                count += 1
            if m_hi[k] > cursor:
                cursor = m_hi[k]
            if cursor >= end:
                break
        if end - cursor > _MIN_LENGTH_M:
            if count >= capacity:
                return -1
            lo_out[count] = cursor
            hi_out[count] = end
            count += 1
    return count


@njit("float64(float64, float64, float64, float64, float64, float64)", cache=True, nogil=True,
      error_model="numpy")
def _point_segment(px, py, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    t = 0.0
    if length2 > 0.0:
        t = ((px - x0) * dx + (py - y0) * dy) / length2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    ex = x0 + t * dx - px
    ey = y0 + t * dy - py
    return math.sqrt(ex * ex + ey * ey)


@njit(_SIG_CLEAR, cache=True, nogil=True, error_model="numpy")
def segment_clearance(seg, ax, ay, bx, by):
    best = np.inf
    for i in range(seg.shape[0]):
        x0, y0, x1, y1 = seg[i, 0], seg[i, 1], seg[i, 2], seg[i, 3]
        d1 = (bx - ax) * (y0 - ay) - (by - ay) * (x0 - ax)
        d2 = (bx - ax) * (y1 - ay) - (by - ay) * (x1 - ax)
        d3 = (x1 - x0) * (ay - y0) - (y1 - y0) * (ax - x0)
        d4 = (x1 - x0) * (by - y0) - (y1 - y0) * (bx - x0)
        if ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)):
            return 0.0
        d = _point_segment(ax, ay, x0, y0, x1, y1)
        d = min(d, _point_segment(bx, by, x0, y0, x1, y1))
        d = min(d, _point_segment(x0, y0, ax, ay, bx, by))
        d = min(d, _point_segment(x1, y1, ax, ay, bx, by))
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------- plain-Python references

def raster_intervals_ref(seg: np.ndarray, y: float, d: float) -> list[tuple[float, float]]:
    crossings = []
    for x0, y0, x1, y1 in seg:
        if (y0 <= y < y1) or (y1 <= y < y0):
            crossings.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    crossings.sort()
    forbidden = []
    for x0, y0, x1, y1 in seg:
        parts = []
        for px, py in ((x0, y0), (x1, y1)):
            if abs(y - py) <= d:
                half = math.sqrt(d * d - (y - py) ** 2)
                parts.append((px - half, px + half))
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 0.0:
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            lo, hi = -math.inf, math.inf
            k_n = uy * x0 + (y - y0) * ux
            if uy == 0.0:
                if abs(k_n) > d:
                    lo = math.inf
            else:
                p, q = (k_n - d) / uy, (k_n + d) / uy
                lo, hi = max(lo, min(p, q)), min(hi, max(p, q))
            t0 = (y - y0) * uy
            if ux == 0.0:
                if t0 < 0.0 or t0 > length:
                    lo = math.inf
            else:
                p, q = x0 - t0 / ux, x0 + (length - t0) / ux
                lo, hi = max(lo, min(p, q)), min(hi, max(p, q))
            if lo <= hi:
                parts.append((lo, hi))
        if parts:
            forbidden.append((min(p[0] for p in parts), max(p[1] for p in parts)))
    forbidden.sort(key=lambda iv: iv[0])
    merged: list[list[float]] = []
    for lo, hi in forbidden:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    out = []
    for j in range(0, len(crossings) - 1, 2):
        start, end = crossings[j], crossings[j + 1]
        cursor = start
        for lo, hi in merged:
            if hi <= cursor:
                continue
            if lo >= end:
                break
            if lo - cursor > _MIN_LENGTH_M:
                out.append((cursor, lo))
            cursor = max(cursor, hi)
            if cursor >= end:
                break
        if end - cursor > _MIN_LENGTH_M:
            out.append((cursor, end))
    return out


def segment_clearance_ref(seg: np.ndarray, ax: float, ay: float, bx: float, by: float) -> float:
    def point_segment(px, py, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0.0 else min(1.0, max(0.0, ((px - x0) * dx + (py - y0) * dy) / length2))
        return math.hypot(x0 + t * dx - px, y0 + t * dy - py)

    best = math.inf
    for x0, y0, x1, y1 in seg:
        d1 = (bx - ax) * (y0 - ay) - (by - ay) * (x0 - ax)
        d2 = (bx - ax) * (y1 - ay) - (by - ay) * (x1 - ax)
        d3 = (x1 - x0) * (ay - y0) - (y1 - y0) * (ax - x0)
        d4 = (x1 - x0) * (by - y0) - (y1 - y0) * (bx - x0)
        if d1 * d2 < 0.0 and d3 * d4 < 0.0:
            return 0.0
        best = min(best, point_segment(ax, ay, x0, y0, x1, y1), point_segment(bx, by, x0, y0, x1, y1),
                   point_segment(x0, y0, ax, ay, bx, by), point_segment(x1, y1, ax, ay, bx, by))
    return best


KERNELS = (raster_intervals, segment_clearance)
