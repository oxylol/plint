"""Compiled kernels of the structure domain (architecture 5.5; docs/models/structure.md section 8).

Hot-loop kernels (the integration composes them into core.hotloop.run_macro):

* :func:`sample_node_grid` - d(X) at points, trilinear in a node grid (Displacement layout), clamped;
* :func:`material_points` - X = x - d(X) by n fixed-point steps (architecture 4.1: one step, n = 1).

L2 kernels of the voxel finite elements:

* :func:`neighbour_table`, :func:`fill_indices` - the CSR pattern of a voxel model (abi.VoxelPattern);
* :func:`assemble_values` - one element matrix added over all active cells (scaled per cell);
* :func:`add_point_springs` - 3 x 3 spring blocks at points inside cells, spread by trilinear weights;
* :func:`scatter_cell_vectors` - element load vectors ``Ge @ coef[c]`` into a global vector;
* :func:`cell_average` - node values to cell centres (mean of the cell's model nodes).

Every kernel has a plain-Python reference in :mod:`cnctwin.structure.reference`.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

f8 = types.float64
i8 = types.int64
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
F4 = types.float64[:, :, :, ::1]
I1 = types.int64[::1]
I2 = types.int64[:, ::1]
I3 = types.int64[:, :, ::1]
B3 = types.int8[:, :, ::1]
J1 = types.int32[::1]
OPTS = dict(cache=True, nogil=True, error_model="numpy")


# ------------------------------------------------------------------------------------------------ hot loop
@njit(types.void(F1, f8, F4, F2, F2), **OPTS)
def sample_node_grid(origin, cell, grid, points, out):
    """out[p] = trilinear interpolation of grid (node values at origin + (i, j, k) cell) at points[p], clamped at
    the border; an empty grid or cell <= 0 gives zeros."""
    nx = grid.shape[0]
    ny = grid.shape[1]
    nz = grid.shape[2]
    for p in range(points.shape[0]):
        out[p, 0] = 0.0
        out[p, 1] = 0.0
        out[p, 2] = 0.0
        if nx == 0 or ny == 0 or nz == 0 or cell <= 0.0:
            continue
        fx = (points[p, 0] - origin[0]) / cell
        fy = (points[p, 1] - origin[1]) / cell
        fz = (points[p, 2] - origin[2]) / cell
        i = min(max(int(math.floor(fx)), 0), max(nx - 2, 0))
        j = min(max(int(math.floor(fy)), 0), max(ny - 2, 0))
        k = min(max(int(math.floor(fz)), 0), max(nz - 2, 0))
        ax = min(max(fx - i, 0.0), 1.0) if nx > 1 else 0.0
        ay = min(max(fy - j, 0.0), 1.0) if ny > 1 else 0.0
        az = min(max(fz - k, 0.0), 1.0) if nz > 1 else 0.0
        for di in range(2):
            ii = min(i + di, nx - 1)
            wx = ax if di == 1 else 1.0 - ax
            for dj in range(2):
                jj = min(j + dj, ny - 1)
                wy = ay if dj == 1 else 1.0 - ay
                for dk in range(2):
                    kk = min(k + dk, nz - 1)
                    wz = az if dk == 1 else 1.0 - az
                    w = wx * wy * wz
                    out[p, 0] += w * grid[ii, jj, kk, 0]
                    out[p, 1] += w * grid[ii, jj, kk, 1]
                    out[p, 2] += w * grid[ii, jj, kk, 2]


@njit(types.void(F1, f8, F4, F2, i8, F2), **OPTS)
def material_points(origin, cell, grid, x, n_iter, out):
    """out = X with X = x - d(X), by n_iter fixed-point steps from X = x (one step: X = x - d(x))."""
    n = x.shape[0]
    d = np.empty((1, 3))
    q = np.empty((1, 3))
    for p in range(n):
        out[p, 0] = x[p, 0]
        out[p, 1] = x[p, 1]
        out[p, 2] = x[p, 2]
        for _ in range(n_iter):
            q[0, 0] = out[p, 0]
            q[0, 1] = out[p, 1]
            q[0, 2] = out[p, 2]
            sample_node_grid(origin, cell, grid, q, d)
            out[p, 0] = x[p, 0] - d[0, 0]
            out[p, 1] = x[p, 1] - d[0, 1]
            out[p, 2] = x[p, 2] - d[0, 2]


# ------------------------------------------------------------------------------------------------ voxel FE
@njit(i8(B3, I3, I2, I2), **OPTS)
def neighbour_table(active, node_index, node_ijk, nbr):
    """nbr[n, o] = rank of the neighbour at offset o among node n's coupled neighbours (offsets in increasing o),
    -1 when the neighbour is not in the model or shares no active cell with n. Returns the total count."""
    nx = active.shape[0]
    ny = active.shape[1]
    nz = active.shape[2]
    total = 0
    for n in range(node_ijk.shape[0]):
        i = node_ijk[n, 0]
        j = node_ijk[n, 1]
        k = node_ijk[n, 2]
        rank = 0
        for o in range(27):
            di = o // 9 - 1
            dj = (o // 3) % 3 - 1
            dk = o % 3 - 1
            nbr[n, o] = -1
            mi = i + di
            mj = j + dj
            mk = k + dk
            if mi < 0 or mj < 0 or mk < 0 or mi > nx or mj > ny or mk > nz:
                continue
            if node_index[mi, mj, mk] < 0:
                continue
            ok = False
            for cx in range(min(i, mi) - (1 if di == 0 else 0), min(i, mi) + 1):
                if cx < 0 or cx >= nx:
                    continue
                for cy in range(min(j, mj) - (1 if dj == 0 else 0), min(j, mj) + 1):
                    if cy < 0 or cy >= ny:
                        continue
                    for cz in range(min(k, mk) - (1 if dk == 0 else 0), min(k, mk) + 1):
                        if cz < 0 or cz >= nz:
                            continue
                        if active[cx, cy, cz] != 0:
                            ok = True
            if ok:
                nbr[n, o] = rank
                rank += 1
        total += rank
    return total


@njit(types.void(I2, I3, I2, I1, J1), **OPTS)
def fill_indices(node_ijk, node_index, nbr, indptr, indices):
    """Column indices of the CSR rows (3 per node, columns 3 m + c of every coupled neighbour m, ascending)."""
    for n in range(node_ijk.shape[0]):
        i = node_ijk[n, 0]
        j = node_ijk[n, 1]
        k = node_ijk[n, 2]
        for o in range(27):
            r = nbr[n, o]
            if r < 0:
                continue
            m = node_index[i + o // 9 - 1, j + (o // 3) % 3 - 1, k + o % 3 - 1]
            for row in range(3):
                start = indptr[3 * n + row] + 3 * r
                for c in range(3):
                    indices[start + c] = 3 * m + c


@njit(types.void(I2, I3, I2, I1, F2, F1, F1), **OPTS)
def assemble_values(cells, node_index, nbr, indptr, Ke, scale, data):
    """data += scale[c] * Ke over the active cells (corner a = 4 di + 2 dj + dk), in cell order."""
    for c in range(cells.shape[0]):
        ci = cells[c, 0]
        cj = cells[c, 1]
        ck = cells[c, 2]
        s = scale[c]
        if s == 0.0:
            continue
        for a in range(8):
            ai = (a >> 2) & 1
            aj = (a >> 1) & 1
            ak = a & 1
            na = node_index[ci + ai, cj + aj, ck + ak]
            for b in range(8):
                bi = (b >> 2) & 1
                bj = (b >> 1) & 1
                bk = b & 1
                o = 9 * (bi - ai + 1) + 3 * (bj - aj + 1) + (bk - ak + 1)
                r = nbr[na, o]
                for row in range(3):
                    start = indptr[3 * na + row] + 3 * r
                    for col in range(3):
                        data[start + col] += s * Ke[3 * a + row, 3 * b + col]


@njit(types.void(I3, I2, I1, I2, F2, F3, F1), **OPTS)
def add_point_springs(node_index, nbr, indptr, point_cell, weights, kmat, data):
    """For each point p in cell point_cell[p] with corner weights weights[p, 8] (0 for corners not in the model):
    data += w_a w_b kmat[p] between corners a and b (a 3 x 3 spring to ground at the interpolated point)."""
    for p in range(point_cell.shape[0]):
        ci = point_cell[p, 0]
        cj = point_cell[p, 1]
        ck = point_cell[p, 2]
        for a in range(8):
            wa = weights[p, a]
            if wa == 0.0:
                continue
            ai = (a >> 2) & 1
            aj = (a >> 1) & 1
            ak = a & 1
            na = node_index[ci + ai, cj + aj, ck + ak]
            for b in range(8):
                wb = weights[p, b]
                if wb == 0.0:
                    continue
                bi = (b >> 2) & 1
                bj = (b >> 1) & 1
                bk = b & 1
                r = nbr[na, 9 * (bi - ai + 1) + 3 * (bj - aj + 1) + (bk - ak + 1)]
                w = wa * wb
                for row in range(3):
                    start = indptr[3 * na + row] + 3 * r
                    for col in range(3):
                        data[start + col] += w * kmat[p, row, col]


@njit(types.void(I2, I3, F2, F2, F1), **OPTS)
def scatter_cell_vectors(cells, node_index, coef, Ge, out):
    """out[dofs of cell c] += Ge @ coef[c] (Ge [24, k], coef [n_cells, k])."""
    kdim = Ge.shape[1]
    for c in range(cells.shape[0]):
        ci = cells[c, 0]
        cj = cells[c, 1]
        ck = cells[c, 2]
        for a in range(8):
            na = node_index[ci + ((a >> 2) & 1), cj + ((a >> 1) & 1), ck + (a & 1)]
            for row in range(3):
                acc = 0.0
                for q in range(kdim):
                    acc += Ge[3 * a + row, q] * coef[c, q]
                out[3 * na + row] += acc


@njit(types.void(F4, B3, F4, B3), **OPTS)
def cell_average(node_vals, node_ok, out, has):
    """out[i, j, k] = mean of the cell's corner values over the corners with node_ok; has = 1 where any."""
    nx = out.shape[0]
    ny = out.shape[1]
    nz = out.shape[2]
    m = out.shape[3]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cnt = 0
                for q in range(m):
                    out[i, j, k, q] = 0.0
                for a in range(8):
                    ii = i + ((a >> 2) & 1)
                    jj = j + ((a >> 1) & 1)
                    kk = k + (a & 1)
                    if node_ok[ii, jj, kk] != 0:
                        cnt += 1
                        for q in range(m):
                            out[i, j, k, q] += node_vals[ii, jj, kk, q]
                if cnt > 0:
                    for q in range(m):
                        out[i, j, k, q] /= cnt
                    has[i, j, k] = 1
                else:
                    has[i, j, k] = 0


KERNELS = (sample_node_grid, material_points, neighbour_table, fill_indices, assemble_values, add_point_springs,
           scatter_cell_vectors, cell_average)

__all__ = ["sample_node_grid", "material_points", "neighbour_table", "fill_indices", "assemble_values",
           "add_point_springs", "scatter_cell_vectors", "cell_average", "KERNELS"]
