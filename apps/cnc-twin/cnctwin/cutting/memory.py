"""Surface memory at L2 (docs/models/cutting.md "Surface memory"): stale bins, refill planes from the
field, invalidation, and analytic stock queries for tests and validation.

The hot loop never calls Python. A memory bin that is stale (older than one revolution plus one tooth
period, written before the last rapid / tool change / operation, or never written) is refilled inside
the kernel from a **field plane** of that bin: a point q and outward normal m of the material surface
near where the station's circle passes the bin. The planes are computed here, once per macro step,
for the bins in or near the last engaged intervals that are stale or will become stale during the
step, by removal's surface-point query (:class:`SurfaceQuery`). A stale bin without a plane is
reported by the kernel as a refill request (``CutEvents.refill`` rows); the integration then computes
those planes (:func:`planes_for_requests`) and replays the macro step deterministically, exactly as
for an event-buffer overflow.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .abi import CutTool, Refill, SurfaceMemory, row_station

TWO_PI = 2.0 * math.pi


class SurfaceQuery(Protocol):
    """Removal's surface-point query (the field of one workpiece, material configuration, S)."""

    def surface_planes(self, workpiece: int, origin_S_m: np.ndarray, dir_S: np.ndarray,
                       reach_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each ray: the material boundary point nearest to the origin on the line origin + t dir
        (|t| <= reach), its outward unit normal, and whether one was found ([n, 3], [n, 3], [n] bool)."""


def station_geometry(tool: CutTool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(radius, z, n_r, n_z) of every station from its first valid element (the circle the station sweeps
    and the meridian components of its chip-thickness normal)."""
    nst = tool.st_z_m.shape[0]
    radius = np.zeros(nst)
    z = np.array(tool.st_z_m, float)
    n_r = np.zeros(nst)
    n_z = np.zeros(nst)
    for s in range(nst):
        lo, hi = int(tool.st_ptr[s]), int(tool.st_ptr[s + 1])
        if hi <= lo:
            continue
        e = int(tool.st_elem[lo])
        p = tool.p_T_m[e]
        r = math.hypot(p[0], p[1])
        radius[s] = r
        z[s] = p[2]
        if r > 0.0:
            n_r[s] = (tool.n_T[e, 0] * p[0] + tool.n_T[e, 1] * p[1]) / r
        n_z[s] = tool.n_T[e, 2]
    return radius, z, n_r, n_z


def stale_mask(mem: SurfaceMemory, tick_until: int, stale_ticks: int, stale_before: int) -> np.ndarray:
    """Bins that are stale now or will be by ``tick_until``."""
    return (mem.valid == 0) | (mem.tick < stale_before) | (tick_until - mem.tick > stale_ticks)


def candidates(tool: CutTool, mem: SurfaceMemory, lo: np.ndarray, hi: np.ndarray, count: np.ndarray, tick_until: int,
               stale_ticks: int, stale_before: int, margin_rad: float, max_queries: int) -> list[tuple[int, int]]:
    """(memory row, bin) pairs that need a field plane for the coming step: inside the last engaged intervals
    of the row's station widened by ``margin_rad`` and stale by the end of the step, and every such bin of a
    station with an element that reads the field directly (a tap's first land without a predecessor); at most
    ``max_queries`` (spread evenly)."""
    n_rows, nb = mem.valid.shape
    nst = tool.st_z_m.shape[0]
    width = TWO_PI / nb
    centres = (np.arange(nb) + 0.5) * width
    stale = stale_mask(mem, tick_until, stale_ticks, stale_before)
    field = np.zeros(max(nst, 1), bool)
    for e in np.nonzero((tool.read_row < 0) & (tool.valid != 0))[0]:
        field[int(tool.station[e])] = True
    out: list[tuple[int, int]] = []
    for r in range(n_rows):
        s = row_station(tool, r)
        if s >= count.shape[0] or count[s] <= 0:
            continue
        need = stale[r] | field[r] if r < nst else stale[r]
        if not need.any():
            continue
        near = np.zeros(nb, bool)
        for i in range(int(count[s])):
            a0 = lo[s, i] - margin_rad
            a1 = hi[s, i] + margin_rad
            d = np.mod(centres - a0, TWO_PI)
            near |= d <= (a1 - a0)
        for b in np.nonzero(near & need)[0]:
            out.append((r, int(b)))
    if max_queries > 0 and len(out) > max_queries:
        step = len(out) / max_queries
        out = [out[int(k * step)] for k in range(max_queries)]
    return out


def compute_planes(query: SurfaceQuery, workpiece: int, tool: CutTool, refill: Refill, pairs: list[tuple[int, int]],
                   R_S_T0: np.ndarray, tip_S_m: np.ndarray, theta_rad: float, reach_m: float) -> int:
    """Fill ``refill`` for the (memory row, bin) pairs with removal's query at the predicted element positions
    (the circle of the row's station at the bin angle, tool pose at the start of the step). Returns planes
    found."""
    if not pairs:
        return 0
    radius, z, n_r, n_z = station_geometry(tool)
    nb = refill.valid.shape[1]
    r_idx = np.array([p[0] for p in pairs], np.int64)
    s_idx = np.array([row_station(tool, int(p[0])) for p in pairs], np.int64)
    b_idx = np.array([p[1] for p in pairs], np.int64)
    ang = (b_idx + 0.5) * TWO_PI / nb
    c, sn = np.cos(ang), np.sin(ang)
    local = np.stack([radius[s_idx] * c, radius[s_idx] * sn, z[s_idx]], axis=-1)
    normal = np.stack([n_r[s_idx] * c, n_r[s_idx] * sn, n_z[s_idx]], axis=-1)
    R = np.asarray(R_S_T0, float)
    origin = np.asarray(tip_S_m, float)[None, :] + local @ R.T
    direction = -(normal @ R.T)
    norm = np.linalg.norm(direction, axis=1)
    direction = direction / np.where(norm > 0.0, norm, 1.0)[:, None]
    q, m, ok = query.surface_planes(workpiece, origin, direction, reach_m)
    found = 0
    for k in range(len(pairs)):
        s, b = int(r_idx[k]), int(b_idx[k])
        if ok[k]:
            refill.q_S_m[s, b] = q[k]
            refill.m_S[s, b] = m[k]
            refill.valid[s, b] = 1
            found += 1
        else:
            refill.valid[s, b] = 0
    return found


def planes_for_requests(query: SurfaceQuery, workpiece: int, refill: Refill, rows: np.ndarray, reach_m: float) -> int:
    """Planes for the kernel's refill requests (rows of abi.REFILL_COLUMNS: tick, station, bin, element
    position and normal in S at the read): the query starts at the element itself (replay)."""
    if rows.size == 0:
        return 0
    origin = rows[:, 3:6]
    direction = -rows[:, 6:9]
    q, m, ok = query.surface_planes(workpiece, origin, direction, reach_m)
    found = 0
    for k in range(rows.shape[0]):
        s, b = int(rows[k, 1]), int(rows[k, 2])
        if ok[k]:
            refill.q_S_m[s, b] = q[k]
            refill.m_S[s, b] = m[k]
            refill.valid[s, b] = 1
            found += 1
    return found


def invalidate(mem: SurfaceMemory) -> None:
    """Forget the memory (new setup): every bin is refilled from the field when read."""
    mem.valid[:] = 0


# ------------------------------------------------------------------------------- analytic stocks
class HalfSpaceStock:
    """Material {x : m . x <= d} (outward normal m): the planar face of a stock (tests, validation)."""

    def __init__(self, normal, offset_m: float):
        n = np.asarray(normal, float)
        self.m = n / np.linalg.norm(n)
        self.d = float(offset_m)

    def surface_planes(self, workpiece, origin_S_m, dir_S, reach_m):
        o = np.atleast_2d(origin_S_m)
        d = np.atleast_2d(dir_S)
        den = d @ self.m
        t = np.where(np.abs(den) > 1e-12, (self.d - o @ self.m) / np.where(np.abs(den) > 1e-12, den, 1.0), np.inf)
        ok = np.abs(t) <= reach_m
        q = o + np.where(ok, t, 0.0)[:, None] * d
        return q, np.broadcast_to(self.m, q.shape).copy(), ok


class BoxStock:
    """Axis-aligned box of material [lo, hi] (S): the boundary point nearest to the origin along each line."""

    def __init__(self, lo_m, hi_m):
        self.lo = np.asarray(lo_m, float)
        self.hi = np.asarray(hi_m, float)

    def surface_planes(self, workpiece, origin_S_m, dir_S, reach_m):
        o = np.atleast_2d(origin_S_m)
        d = np.atleast_2d(dir_S)
        n = o.shape[0]
        q = np.zeros((n, 3))
        m = np.zeros((n, 3))
        ok = np.zeros(n, bool)
        for k in range(n):
            t_in, t_out = -np.inf, np.inf
            ax_in = ax_out = 0
            s_in = s_out = 1.0
            hit = True
            for a in range(3):
                if abs(d[k, a]) < 1e-15:
                    if not (self.lo[a] <= o[k, a] <= self.hi[a]):
                        hit = False
                    continue
                t1 = (self.lo[a] - o[k, a]) / d[k, a]
                t2 = (self.hi[a] - o[k, a]) / d[k, a]
                lo_face = t1 < t2
                ta, tb = (t1, t2) if lo_face else (t2, t1)
                if ta > t_in:
                    t_in, ax_in, s_in = ta, a, (-1.0 if lo_face else 1.0)
                if tb < t_out:
                    t_out, ax_out, s_out = tb, a, (1.0 if lo_face else -1.0)
            if not hit or t_in > t_out:
                continue
            best = t_in if abs(t_in) <= abs(t_out) else t_out
            if abs(best) > reach_m:
                continue
            q[k] = o[k] + best * d[k]
            normal = np.zeros(3)
            if best == t_in:
                normal[ax_in] = s_in
            else:
                normal[ax_out] = s_out
            m[k] = normal
            ok[k] = True
        return q, m, ok


class BoreStock:
    """Material around a cylindrical bore along S z: {r >= radius, z <= z_top} with r measured from the bore
    axis through ``centre`` (a pre-drilled hole for taps, a bore for boring heads)."""

    def __init__(self, radius_m: float, centre_m=(0.0, 0.0), z_top_m: float = math.inf):
        self.R = float(radius_m)
        self.c = np.asarray(centre_m, float)[:2]
        self.z_top = float(z_top_m)

    def surface_planes(self, workpiece, origin_S_m, dir_S, reach_m):
        o = np.atleast_2d(origin_S_m)
        d = np.atleast_2d(dir_S)
        n = o.shape[0]
        q = np.zeros((n, 3))
        m = np.zeros((n, 3))
        ok = np.zeros(n, bool)
        for k in range(n):
            ox, oy = o[k, 0] - self.c[0], o[k, 1] - self.c[1]
            a = d[k, 0] ** 2 + d[k, 1] ** 2
            hits = []
            if a > 1e-24:
                b = ox * d[k, 0] + oy * d[k, 1]
                c = ox * ox + oy * oy - self.R * self.R
                disc = b * b - a * c
                if disc >= 0.0:
                    root = math.sqrt(disc)
                    for t in ((-b - root) / a, (-b + root) / a):
                        if o[k, 2] + t * d[k, 2] <= self.z_top:
                            hits.append((abs(t), t, 0))
            if math.isfinite(self.z_top) and abs(d[k, 2]) > 1e-15:
                t = (self.z_top - o[k, 2]) / d[k, 2]
                if math.hypot(ox + t * d[k, 0], oy + t * d[k, 1]) >= self.R:
                    hits.append((abs(t), t, 1))
            hits = [h for h in hits if h[0] <= reach_m]
            if not hits:
                continue
            _, t, face = min(hits)
            q[k] = o[k] + t * d[k]
            if face == 0:
                rx, ry = q[k, 0] - self.c[0], q[k, 1] - self.c[1]
                rr = math.hypot(rx, ry)
                m[k] = (-rx / rr, -ry / rr, 0.0)
            else:
                m[k] = (0.0, 0.0, 1.0)
            ok[k] = True
        return q, m, ok


__all__ = ["SurfaceQuery", "station_geometry", "stale_mask", "candidates", "compute_planes", "planes_for_requests",
           "invalidate", "HalfSpaceStock", "BoxStock", "BoreStock"]
