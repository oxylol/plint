"""Layouts of the structure kernels (architecture 5.5; docs/models/structure.md section 8).

All kernels take plain writable C-contiguous arrays and scalars (no NamedTuple is needed for their arguments);
the NamedTuples below group the arrays the way the domain keeps them.

* :class:`NodeGrid` - a vector field on the node lattice of a voxel grid, value ``values[i, j, k]`` at
  ``origin_S_m + (i, j, k) * cell_m`` (the layout of ``core.bus.Displacement`` grids, which removal samples the
  same way); trilinear in between, clamped at the border.
* :class:`VoxelPattern` - the CSR pattern of a voxel FE model: ``node_index[i, j, k]`` the model node of lattice
  node (i, j, k) (-1: not in the model), ``node_ijk`` its lattice coordinates, ``nbr[n, o]`` the position of the
  neighbour at offset ``o = 9 (di + 1) + 3 (dj + 1) + (dk + 1)`` in node n's neighbour list (-1: not coupled),
  ``indptr`` / ``indices`` the CSR arrays over DOFs ``3 n + component``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class NodeGrid(NamedTuple):
    origin_S_m: np.ndarray          # [3]
    cell_m: np.ndarray              # [1]
    values: np.ndarray              # [nx + 1, ny + 1, nz + 1, 3]


class VoxelPattern(NamedTuple):
    node_index: np.ndarray          # [nx + 1, ny + 1, nz + 1] int64
    node_ijk: np.ndarray            # [n_nodes, 3] int64
    nbr: np.ndarray                 # [n_nodes, 27] int64
    indptr: np.ndarray              # [3 n_nodes + 1] int64
    indices: np.ndarray             # [nnz] int32


class StructureInputs(NamedTuple):
    """What structure hands to the compiled loop each macro step (frozen for the step): the total displacement
    grid of every workpiece (index = workpiece), for X_e = x_e - d(x_e) (kernels.material_points). A workpiece
    without a grid has cell 0 and a 1 x 1 x 1 zero grid (the kernels return d = 0)."""

    origin_S_m: np.ndarray          # [n_work, 3]
    cell_m: np.ndarray              # [n_work]
    total_m: tuple                  # n_work arrays [nx + 1, ny + 1, nz + 1, 3]


def offset_code(di: int, dj: int, dk: int) -> int:
    return 9 * (di + 1) + 3 * (dj + 1) + (dk + 1)


def node_grid(origin_S_m, cell_m: float, values) -> NodeGrid:
    return NodeGrid(np.ascontiguousarray(np.asarray(origin_S_m, dtype=float).reshape(3)),
                    np.array([float(cell_m)]), np.ascontiguousarray(np.asarray(values, dtype=float)))


__all__ = ["NodeGrid", "VoxelPattern", "StructureInputs", "offset_code", "node_grid"]
