"""3D-DDA traversal of a uniform grid along a ray or segment (Amanatides & Woo 1987; docs/models/geom.md
"Grid traversal"), for coolant jets, chip trajectories and probe moves against the bricks of the dexel
field (removal publishes which bricks hold material; this module only walks the grid).

The grid (:class:`cnctwin.geom.abi.Grid`) has cells ``(i, j, k)`` spanning ``lo + (i, j, k) * cell`` to
``lo + (i + 1, j + 1, k + 1) * cell``. The ray ``o + t d`` is first clipped to the grid box and to
``[t0, t1]``; then every cell it passes is visited in order with its entry and exit parameter. Each
boundary crossing is computed from the cell index (``(boundary - o) / d``), never by accumulating
``t`` increments, so long walks do not drift. Where the ray crosses two or three boundaries at once
(an edge or a corner) the axes advance one at a time in x, y, z order: the cells touched only at that
edge or corner are visited with a zero-length interval (conservative for occlusion).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from .abi import GRID_TYPE, Grid, contiguous, make_grid

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
I2 = types.int64[:, ::1]
U3 = types.uint8[:, :, ::1]
B1 = types.int8[::1]
f8 = types.float64
i8 = types.int64
OPTS = dict(cache=True, nogil=True, error_model="numpy")
GRID = GRID_TYPE


@njit(i8(F1, F1, f8, f8, GRID, F1), **OPTS)
def grid_clip_nb(o, d, t0, t1, grid, out):
    """Clip [t0, t1] to the grid box: returns 1 and out = [t_enter, t_exit], or 0 when the ray misses."""
    a = t0
    b = t1
    for k in range(3):
        lo = grid.lo[k]
        hi = grid.lo[k] + grid.dims[k] * grid.cell[k]
        if d[k] == 0.0:
            if o[k] < lo or o[k] > hi:
                return 0
            continue
        ta = (lo - o[k]) / d[k]
        tb = (hi - o[k]) / d[k]
        if ta > tb:
            ta, tb = tb, ta
        if ta > a:
            a = ta
        if tb < b:
            b = tb
        if a > b:
            return 0
    out[0] = a
    out[1] = b
    return 1


@njit(i8(F1, F1, f8, f8, GRID, I2, F2), **OPTS)
def dda_cells_nb(o, d, t0, t1, grid, out_ijk, out_t):
    """Cells along the ray in order: out_ijk[n] = (i, j, k), out_t[n] = (t_in, t_out). Writes at most
    len(out_ijk) rows and returns the total number of cells (0 when the ray misses the grid)."""
    clip = np.empty(2)
    if grid_clip_nb(o, d, t0, t1, grid, clip) == 0:
        return 0
    t_cur = clip[0]
    t_exit = clip[1]
    idx = np.empty(3, np.int64)
    step = np.empty(3, np.int64)
    t_next = np.empty(3)
    for k in range(3):
        p = o[k] + t_cur * d[k]
        c = int(math.floor((p - grid.lo[k]) / grid.cell[k]))
        if c < 0:
            c = 0
        if c > grid.dims[k] - 1:
            c = grid.dims[k] - 1
        idx[k] = c
        if d[k] > 0.0:
            step[k] = 1
            t_next[k] = (grid.lo[k] + (c + 1) * grid.cell[k] - o[k]) / d[k]
        elif d[k] < 0.0:
            step[k] = -1
            t_next[k] = (grid.lo[k] + c * grid.cell[k] - o[k]) / d[k]
        else:
            step[k] = 0
            t_next[k] = np.inf
    cap = out_ijk.shape[0]
    n = 0
    while True:
        axis = 0
        if t_next[1] < t_next[axis]:
            axis = 1
        if t_next[2] < t_next[axis]:
            axis = 2
        t_end = t_next[axis]
        if t_end > t_exit:
            t_end = t_exit
        if t_end < t_cur:
            t_end = t_cur
        if n < cap:
            out_ijk[n, 0] = idx[0]
            out_ijk[n, 1] = idx[1]
            out_ijk[n, 2] = idx[2]
            out_t[n, 0] = t_cur
            out_t[n, 1] = t_end
        n += 1
        if t_next[axis] >= t_exit:
            break
        idx[axis] += step[axis]
        if idx[axis] < 0 or idx[axis] >= grid.dims[axis]:
            break
        t_cur = t_end
        if step[axis] > 0:
            t_next[axis] = (grid.lo[axis] + (idx[axis] + 1) * grid.cell[axis] - o[axis]) / d[axis]
        else:
            t_next[axis] = (grid.lo[axis] + idx[axis] * grid.cell[axis] - o[axis]) / d[axis]
    return n


@njit(i8(F1, F1, f8, f8, GRID, U3, F1), **OPTS)
def dda_first_occupied_nb(o, d, t0, t1, grid, occupancy, out):
    """First cell along the ray whose occupancy is non-zero: returns 1 and out = [i, j, k, t_in, t_out],
    or 0. ``occupancy`` has shape grid.dims (uint8)."""
    clip = np.empty(2)
    if grid_clip_nb(o, d, t0, t1, grid, clip) == 0:
        return 0
    t_cur = clip[0]
    t_exit = clip[1]
    idx = np.empty(3, np.int64)
    step = np.empty(3, np.int64)
    t_next = np.empty(3)
    for k in range(3):
        p = o[k] + t_cur * d[k]
        c = int(math.floor((p - grid.lo[k]) / grid.cell[k]))
        if c < 0:
            c = 0
        if c > grid.dims[k] - 1:
            c = grid.dims[k] - 1
        idx[k] = c
        if d[k] > 0.0:
            step[k] = 1
            t_next[k] = (grid.lo[k] + (c + 1) * grid.cell[k] - o[k]) / d[k]
        elif d[k] < 0.0:
            step[k] = -1
            t_next[k] = (grid.lo[k] + c * grid.cell[k] - o[k]) / d[k]
        else:
            step[k] = 0
            t_next[k] = np.inf
    while True:
        axis = 0
        if t_next[1] < t_next[axis]:
            axis = 1
        if t_next[2] < t_next[axis]:
            axis = 2
        t_end = t_next[axis]
        if t_end > t_exit:
            t_end = t_exit
        if t_end < t_cur:
            t_end = t_cur
        if occupancy[idx[0], idx[1], idx[2]] != 0:
            out[0] = idx[0]
            out[1] = idx[1]
            out[2] = idx[2]
            out[3] = t_cur
            out[4] = t_end
            return 1
        if t_next[axis] >= t_exit:
            return 0
        idx[axis] += step[axis]
        if idx[axis] < 0 or idx[axis] >= grid.dims[axis]:
            return 0
        t_cur = t_end
        if step[axis] > 0:
            t_next[axis] = (grid.lo[axis] + (idx[axis] + 1) * grid.cell[axis] - o[axis]) / d[axis]
        else:
            t_next[axis] = (grid.lo[axis] + idx[axis] * grid.cell[axis] - o[axis]) / d[axis]


@njit(types.void(F2, F2, F1, F1, GRID, U3, F2, B1), **OPTS)
def dda_first_occupied_batch_nb(O, D, t0, t1, grid, occupancy, out, found):
    """Batch of :func:`dda_first_occupied_nb` (rows of O, D)."""
    row = np.empty(5)
    for i in range(O.shape[0]):
        if dda_first_occupied_nb(O[i], D[i], t0[i], t1[i], grid, occupancy, row) == 1:
            found[i] = 1
            for k in range(5):
                out[i, k] = row[k]
        else:
            found[i] = 0
            for k in range(5):
                out[i, k] = np.nan


# ============================================================================ Python API

def _vec(x) -> np.ndarray:
    return contiguous(np.asarray(x, dtype=np.float64).reshape(3))


def grid(lo, cell, dims) -> Grid:
    """A uniform grid (cell may be one number or one per axis)."""
    return make_grid(lo, cell, dims)


def ray_cells(g: Grid, origin, direction, t0: float = 0.0, t1: float = math.inf, max_cells: int | None = None):
    """``(ijk [n, 3] int64, t [n, 2])`` of the cells the ray passes in [t0, t1], in order."""
    o, d = _vec(origin), _vec(direction)
    cap = int(max_cells) if max_cells is not None else int(np.sum(g.dims)) + 3
    while True:
        ijk = np.empty((cap, 3), np.int64)
        t = np.empty((cap, 2))
        n = int(dda_cells_nb(o, d, float(t0), float(t1), g, ijk, t))
        if n <= cap or max_cells is not None:
            n = min(n, cap)
            return ijk[:n].copy(), t[:n].copy()
        cap = n


def segment_cells(g: Grid, p0, p1):
    """Cells crossed by the segment p0 -> p1, with fractions in [0, 1]."""
    a, b = _vec(p0), _vec(p1)
    return ray_cells(g, a, b - a, 0.0, 1.0)


def first_occupied(g: Grid, occupancy, origin, direction, t0: float = 0.0, t1: float = math.inf):
    """``((i, j, k), t_in, t_out)`` of the first non-zero cell along the ray, or None."""
    occ = contiguous(np.asarray(occupancy, dtype=np.uint8))
    if occ.shape != tuple(int(v) for v in g.dims):
        raise ValueError(f"occupancy shape {occ.shape} does not match the grid {tuple(g.dims)}")
    out = np.zeros(5)
    if dda_first_occupied_nb(_vec(origin), _vec(direction), float(t0), float(t1), g, occ, out) == 0:
        return None
    return (int(out[0]), int(out[1]), int(out[2])), float(out[3]), float(out[4])


def first_occupied_batch(g: Grid, occupancy, origins, directions, t0=0.0, t1=math.inf):
    """``(found [n] int8, rows [n, 5] = (i, j, k, t_in, t_out), NaN where nothing was found)``."""
    occ = contiguous(np.asarray(occupancy, dtype=np.uint8))
    O = contiguous(np.asarray(origins, dtype=np.float64).reshape(-1, 3))
    D = contiguous(np.broadcast_to(np.asarray(directions, dtype=np.float64).reshape(-1, 3), O.shape))
    a = contiguous(np.broadcast_to(np.asarray(t0, dtype=np.float64), (len(O),)))
    b = contiguous(np.broadcast_to(np.asarray(t1, dtype=np.float64), (len(O),)))
    out = np.empty((len(O), 5))
    found = np.empty(len(O), np.int8)
    dda_first_occupied_batch_nb(O, D, a, b, g, occ, out, found)
    return found, out


KERNELS = (grid_clip_nb, dda_cells_nb, dda_first_occupied_nb, dda_first_occupied_batch_nb)

__all__ = ["grid", "ray_cells", "segment_cells", "first_occupied", "first_occupied_batch", "grid_clip_nb",
           "dda_cells_nb", "dda_first_occupied_nb", "dda_first_occupied_batch_nb", "Grid", "KERNELS"]
