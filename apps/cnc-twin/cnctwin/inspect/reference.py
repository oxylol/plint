"""Plain-Python twins of the inspection kernels (docs/architecture.md 5.5: each kernel is compared with its
reference in tests/unit/inspect/test_kernels.py). They are written for clarity, not speed: brute-force
distances over every triangle (cnctwin.geom.reference), explicit interval arithmetic, breadth-first region
growing instead of union-find.
"""

from __future__ import annotations

import math

import numpy as np

from ..geom.mesh import Mesh
from ..geom.reference import closest_point_ref, signed_distance_ref
from . import abi as A


def trilinear_ref(P, origin, cell, grid) -> np.ndarray:
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    grid = np.asarray(grid, dtype=float)
    out = np.zeros((len(P), 3))
    if grid.size == 0 or cell <= 0.0:
        return out
    n = grid.shape[:3]
    for p, x in enumerate(P):
        base, frac = [], []
        for a in range(3):
            f = (x[a] - origin[a]) / cell
            i = min(max(int(math.floor(f)), 0), max(n[a] - 2, 0))
            base.append(i)
            frac.append(min(max(f - i, 0.0), 1.0) if n[a] > 1 else 0.0)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = (frac[0] if di else 1 - frac[0]) * (frac[1] if dj else 1 - frac[1]) * \
                        (frac[2] if dk else 1 - frac[2])
                    ii, jj, kk = (min(base[0] + di, n[0] - 1), min(base[1] + dj, n[1] - 1),
                                  min(base[2] + dk, n[2] - 1))
                    out[p] += w * grid[ii, jj, kk]
    return out


def deviation_ref(mesh: Mesh, P) -> tuple[np.ndarray, np.ndarray]:
    """(signed distance, closest point) by brute force; the sign from the winding number."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    d = np.array([signed_distance_ref(mesh, p) for p in P])
    q = np.array([closest_point_ref(mesh, p)[0] for p in P]).reshape(-1, 3)
    return d, q


def columns_from_hits_ref(offsets, t, face, face_nz, z0, merge) -> list[list[tuple[float, float]]]:
    out = []
    for r in range(len(offsets) - 1):
        events = []
        last = None
        for h in range(int(offsets[r]), int(offsets[r + 1])):
            nz = face_nz[face[h]]
            if nz == 0.0:
                continue
            s = 1 if nz < 0.0 else -1
            if last is not None and last[0] == s and t[h] - last[1] <= merge:
                continue
            last = (s, t[h])
            events.append((z0 + t[h], s))
        intervals: list[tuple[float, float]] = []
        w, lo = 0, 0.0
        for z, s in events:
            before, w = w, w + s
            if before <= 0 < w:
                if intervals and intervals[-1][1] >= z:
                    lo = intervals.pop()[0]
                else:
                    lo = z
            elif before > 0 >= w and z > lo:
                intervals.append((lo, z))
        out.append(intervals)
    return out


def columns_difference_ref(a: list, b: list, min_len: float) -> list[list[tuple[float, float]]]:
    out = []
    for ca, cb in zip(a, b):
        pieces = []
        for lo, hi in ca:
            parts = [(lo, hi)]
            for bl, bh in cb:
                nxt = []
                for pl, ph in parts:
                    if bh <= pl or bl >= ph:
                        nxt.append((pl, ph))
                        continue
                    if bl > pl:
                        nxt.append((pl, bl))
                    if bh < ph:
                        nxt.append((bh, ph))
                parts = nxt
            pieces.extend(p for p in parts if p[1] > p[0] and p[1] - p[0] >= min_len)
        out.append(sorted(pieces))
    return out


def components_ref(cols: A.Columns) -> list[set[int]]:
    """Regions as sets of interval indices (breadth-first over the adjacency)."""
    nx, ny = int(cols.dims[0]), int(cols.dims[1])
    start, seg = cols.start, cols.seg
    m = len(seg)
    col_of = np.zeros(m, np.int64)
    for c in range(nx * ny):
        col_of[start[c]:start[c + 1]] = c
    seen = np.zeros(m, bool)
    regions = []
    for s0 in range(m):
        if seen[s0]:
            continue
        region, queue = set(), [s0]
        seen[s0] = True
        while queue:
            s = queue.pop()
            region.add(s)
            c = int(col_of[s])
            i, j = divmod(c, ny)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < nx and 0 <= jj < ny):
                    continue
                c2 = ii * ny + jj
                for q in range(int(start[c2]), int(start[c2 + 1])):
                    if not seen[q] and min(seg[s, 1], seg[q, 1]) > max(seg[s, 0], seg[q, 0]):
                        seen[q] = True
                        queue.append(q)
        regions.append(region)
    return regions


def component_stats_ref(cols: A.Columns, region: set[int]) -> dict:
    ny = int(cols.dims[1])
    area = float(cols.cell_m[0] * cols.cell_m[1])
    vol = cx = cy = cz = 0.0
    thick = -1.0
    for c in range(len(cols.start) - 1):
        i, j = divmod(c, ny)
        x = cols.origin_m[0] + i * cols.cell_m[0]
        y = cols.origin_m[1] + j * cols.cell_m[1]
        for s in range(int(cols.start[c]), int(cols.start[c + 1])):
            if s not in region:
                continue
            lo, hi = cols.seg[s]
            v = (hi - lo) * area
            vol += v
            cx += v * x
            cy += v * y
            cz += v * 0.5 * (lo + hi)
            thick = max(thick, hi - lo)
    return {"volume_m3": vol, "centroid_m": np.array([cx, cy, cz]) / vol if vol > 0 else np.zeros(3),
            "thickness_m": thick}


def segment_point_ref(row, s: float) -> np.ndarray:
    row = np.asarray(row, dtype=float)
    p0, p1 = row[A.SG_P0:A.SG_P0 + 3], row[A.SG_P1:A.SG_P1 + 3]
    if row[A.SG_KIND] == A.SEG_LINE:
        return p0 + s * (p1 - p0)
    c, n, sweep = row[A.SG_C:A.SG_C + 3], row[A.SG_N:A.SG_N + 3], row[A.SG_SWEEP]
    a0, a1 = float(np.dot(p0 - c, n)), float(np.dot(p1 - c, n))
    v0, v1 = p0 - c - a0 * n, p1 - c - a1 * n
    r0, r1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
    if r0 <= 0.0:
        return p0 + s * (p1 - p0)
    e = v0 / r0
    f = np.cross(n, e)
    ang = s * sweep
    return c + (r0 + s * (r1 - r0)) * (math.cos(ang) * e + math.sin(ang) * f) + (a0 + s * (a1 - a0)) * n


def chord_error_at(mesh: Mesh, row, s: float) -> float:
    """e(s) = d(P(s)) - ((1 - s) d0 + s d1) by brute-force signed distances."""
    d0 = signed_distance_ref(mesh, segment_point_ref(row, 0.0))
    d1 = signed_distance_ref(mesh, segment_point_ref(row, 1.0))
    return signed_distance_ref(mesh, segment_point_ref(row, s)) - ((1 - s) * d0 + s * d1)


def chord_error_ref(mesh: Mesh, row, samples: int = 2001) -> tuple[float, float]:
    """(most negative, most positive) chord error by dense sampling (brute-force signed distances)."""
    d0 = signed_distance_ref(mesh, segment_point_ref(row, 0.0))
    d1 = signed_distance_ref(mesh, segment_point_ref(row, 1.0))
    lo = hi = 0.0
    for s in np.linspace(0.0, 1.0, samples):
        e = signed_distance_ref(mesh, segment_point_ref(row, s)) - ((1 - s) * d0 + s * d1)
        lo, hi = min(lo, e), max(hi, e)
    return lo, hi


def probe_ref(mesh: Mesh, origin, direction, radius: float, max_travel: float, steps: int = 200000) -> float | None:
    """Travel to first contact of the ball by marching in steps of max_travel / steps and bisecting the first
    interfering step (brute-force distances); None without contact."""
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)

    def gap(t):
        return closest_point_ref(mesh, o + t * d)[1] - radius

    if gap(0.0) <= 0.0:
        return 0.0
    dt = max_travel / steps
    t0 = 0.0
    g0 = gap(0.0)
    while t0 < max_travel:
        t1 = min(t0 + max(dt, 0.5 * g0), max_travel)
        g1 = gap(t1)
        if g1 <= 0.0:
            a, b = t0, t1
            for _ in range(80):
                mid = 0.5 * (a + b)
                if gap(mid) > 0.0:
                    a = mid
                else:
                    b = mid
            return 0.5 * (a + b)
        if t1 >= max_travel:
            return None
        t0, g0 = t1, g1
    return None


__all__ = ["trilinear_ref", "deviation_ref", "columns_from_hits_ref", "columns_difference_ref", "components_ref",
           "component_stats_ref", "segment_point_ref", "chord_error_at", "chord_error_ref", "probe_ref"]
