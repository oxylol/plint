"""Plain-Python references of the structure kernels (cnctwin.structure.kernels), for the kernel tests."""

from __future__ import annotations

import math

import numpy as np


def sample_node_grid_ref(origin, cell, grid, points) -> np.ndarray:
    out = np.zeros((len(points), 3))
    nx, ny, nz = grid.shape[:3]
    if nx == 0 or ny == 0 or nz == 0 or cell <= 0.0:
        return out
    for p, x in enumerate(points):
        f = (np.asarray(x) - origin) / cell
        lo = [min(max(int(math.floor(f[a])), 0), max(grid.shape[a] - 2, 0)) for a in range(3)]
        t = [min(max(f[a] - lo[a], 0.0), 1.0) if grid.shape[a] > 1 else 0.0 for a in range(3)]
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = (t[0] if di else 1 - t[0]) * (t[1] if dj else 1 - t[1]) * (t[2] if dk else 1 - t[2])
                    idx = (min(lo[0] + di, nx - 1), min(lo[1] + dj, ny - 1), min(lo[2] + dk, nz - 1))
                    out[p] += w * grid[idx]
    return out


def material_points_ref(origin, cell, grid, x, n_iter) -> np.ndarray:
    X = np.array(x, dtype=float, copy=True)
    for _ in range(n_iter):
        X = np.asarray(x) - sample_node_grid_ref(origin, cell, grid, X)
    return X


def neighbour_table_ref(active, node_index, node_ijk) -> tuple[np.ndarray, int]:
    nx, ny, nz = active.shape
    nbr = np.full((len(node_ijk), 27), -1, np.int64)
    total = 0
    for n, (i, j, k) in enumerate(node_ijk):
        rank = 0
        for o in range(27):
            di, dj, dk = o // 9 - 1, (o // 3) % 3 - 1, o % 3 - 1
            m = (i + di, j + dj, k + dk)
            if min(m) < 0 or m[0] > nx or m[1] > ny or m[2] > nz or node_index[m] < 0:
                continue
            cxs = [i - 1, i] if di == 0 else [min(i, m[0])]
            cys = [j - 1, j] if dj == 0 else [min(j, m[1])]
            czs = [k - 1, k] if dk == 0 else [min(k, m[2])]
            ok = any(0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz and active[cx, cy, cz]
                     for cx in cxs for cy in cys for cz in czs)
            if ok:
                nbr[n, o] = rank
                rank += 1
        total += rank
    return nbr, total


def dense_assembly_ref(cells, node_index, n_nodes, Ke, scale) -> np.ndarray:
    """The global matrix of a voxel model, dense, by direct element loops."""
    K = np.zeros((3 * n_nodes, 3 * n_nodes))
    for c, (i, j, k) in enumerate(cells):
        dofs = []
        for a in range(8):
            na = node_index[i + ((a >> 2) & 1), j + ((a >> 1) & 1), k + (a & 1)]
            dofs.extend([3 * na, 3 * na + 1, 3 * na + 2])
        K[np.ix_(dofs, dofs)] += scale[c] * Ke
    return K


def point_springs_ref(node_index, n_nodes, point_cell, weights, kmat) -> np.ndarray:
    K = np.zeros((3 * n_nodes, 3 * n_nodes))
    for p, (i, j, k) in enumerate(point_cell):
        for a in range(8):
            for b in range(8):
                wa, wb = weights[p, a], weights[p, b]
                if wa == 0.0 or wb == 0.0:
                    continue
                na = node_index[i + ((a >> 2) & 1), j + ((a >> 1) & 1), k + (a & 1)]
                nb = node_index[i + ((b >> 2) & 1), j + ((b >> 1) & 1), k + (b & 1)]
                K[3 * na:3 * na + 3, 3 * nb:3 * nb + 3] += wa * wb * kmat[p]
    return K


def scatter_cell_vectors_ref(cells, node_index, coef, Ge, n_dof) -> np.ndarray:
    out = np.zeros(n_dof)
    for c, (i, j, k) in enumerate(cells):
        fe = Ge @ coef[c]
        for a in range(8):
            na = node_index[i + ((a >> 2) & 1), j + ((a >> 1) & 1), k + (a & 1)]
            out[3 * na:3 * na + 3] += fe[3 * a:3 * a + 3]
    return out


def cell_average_ref(node_vals, node_ok) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = node_ok.shape[0] - 1, node_ok.shape[1] - 1, node_ok.shape[2] - 1
    out = np.zeros((nx, ny, nz, node_vals.shape[3]))
    has = np.zeros((nx, ny, nz), np.int8)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                vals = [node_vals[i + di, j + dj, k + dk] for di in (0, 1) for dj in (0, 1) for dk in (0, 1)
                        if node_ok[i + di, j + dj, k + dk]]
                if vals:
                    out[i, j, k] = np.mean(vals, axis=0)
                    has[i, j, k] = 1
    return out, has


__all__ = ["sample_node_grid_ref", "material_points_ref", "neighbour_table_ref", "dense_assembly_ref",
           "point_springs_ref", "scatter_cell_vectors_ref", "cell_average_ref"]
