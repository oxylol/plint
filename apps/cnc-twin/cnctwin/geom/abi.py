"""Kernel layouts of the geometry library (docs/models/geom.md "Layouts"; rules: docs/architecture.md 5.5).

Every layout is a NamedTuple of writable, C-contiguous NumPy arrays, so Numba types it concretely
(one compiled specialisation per kernel) and the cache can store it. Owners build them in Python
(``cnctwin.geom.bvh.build_tree``) and the kernels only read them.

``MeshTree`` is an axis-aligned bounding-box tree over the triangles of one mesh, flattened in
depth-first order: an inner node ``i`` has its first child at ``i + 1`` and its second child at
``node_right[i]``; a leaf has ``node_right = -1`` and holds ``node_count`` triangles starting at slot
``node_start`` of the slot arrays (``tri_v``, ``tri_face``). Per-face data (normals, pseudo-normals)
is indexed by the mesh's own face id, so query results are face ids of the mesh that was given.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import typeof

# info[] slots of a MeshTree
INFO_NODES = 0          # number of nodes in use
INFO_DEPTH = 1          # depth of the deepest leaf (root = 0)
INFO_CLOSED = 2         # 1 when the mesh is a closed, consistently oriented 2-manifold (signed distance valid)
INFO_LEAF_SIZE = 3      # leaf size the tree was built with
N_INFO = 4

# The traversal stacks are fixed-size local arrays; the build forces a leaf at MAX_DEPTH so a
# depth-first traversal (one pending entry per level) never needs more than MAX_DEPTH + 2 slots and
# a simultaneous two-tree descent never more than 2 (MAX_DEPTH + 2).
MAX_DEPTH = 60
STACK_SIZE = 2 * (MAX_DEPTH + 2)

# closest-feature codes of the closest-point queries (Ericson 2005, 5.1.5)
FEATURE_FACE = 0
FEATURE_EDGE_0 = 1      # edge v0 -> v1
FEATURE_EDGE_1 = 2      # edge v1 -> v2
FEATURE_EDGE_2 = 3      # edge v2 -> v0
FEATURE_VERTEX_0 = 4
FEATURE_VERTEX_1 = 5
FEATURE_VERTEX_2 = 6

# point-in-mesh classification
OUTSIDE = 0
INSIDE = 1
ON_SURFACE = 2
UNDECIDED = -1


class MeshTree(NamedTuple):
    """Bounding-box tree over the triangles of one mesh (built by cnctwin.geom.bvh.build_tree)."""

    node_lo: np.ndarray         # [nn, 3] float64 box minimum
    node_hi: np.ndarray         # [nn, 3] float64 box maximum
    node_right: np.ndarray      # [nn] int32 second child (first child = node + 1); -1 for a leaf
    node_start: np.ndarray      # [nn] int32 first slot of a leaf
    node_count: np.ndarray      # [nn] int32 triangles of a leaf (0 for an inner node)
    tri_v: np.ndarray           # [m, 3, 3] float64 triangle corners in slot order
    tri_face: np.ndarray        # [m] int32 face id of each slot
    vertices: np.ndarray        # [nv, 3] float64
    faces: np.ndarray           # [m, 3] int32
    face_normal: np.ndarray     # [m, 3] float64 unit normal by face id (zero for a degenerate face)
    edge_normal: np.ndarray     # [m, 3, 3] float64 pseudo-normal of edge k (v_k -> v_k+1) of face f
    vertex_normal: np.ndarray   # [nv, 3] float64 angle-weighted pseudo-normal by vertex
    info: np.ndarray            # [N_INFO] int64, INFO_*


class Grid(NamedTuple):
    """A uniform grid of cells (bricks, voxels): cell (i, j, k) spans lo + (i, j, k) * cell ... + cell."""

    lo: np.ndarray              # [3] float64 minimum corner (m)
    cell: np.ndarray            # [3] float64 cell size per axis (m)
    dims: np.ndarray            # [3] int64 number of cells per axis


def sample_tree() -> MeshTree:
    """A one-triangle tree, used to type the kernels' explicit signatures."""
    f64 = np.zeros
    return MeshTree(f64((1, 3)), f64((1, 3)), np.full(1, -1, np.int32), np.zeros(1, np.int32),
                    np.ones(1, np.int32), f64((1, 3, 3)), np.zeros(1, np.int32), f64((3, 3)),
                    np.array([[0, 1, 2]], np.int32), f64((1, 3)), f64((1, 3, 3)), f64((3, 3)),
                    np.zeros(N_INFO, np.int64))


def sample_grid() -> Grid:
    return Grid(np.zeros(3), np.ones(3), np.ones(3, np.int64))


TREE_TYPE = typeof(sample_tree())
GRID_TYPE = typeof(sample_grid())


def contiguous(array, dtype=None) -> np.ndarray:
    """Writable C-contiguous array (a read-only one would type differently and force a second compile)."""
    out = np.ascontiguousarray(array, dtype=dtype)
    if not out.flags.writeable:
        out = out.copy()
    return out


def make_grid(lo, cell, dims) -> Grid:
    cell_arr = np.broadcast_to(np.asarray(cell, dtype=np.float64), (3,))
    if np.any(cell_arr <= 0.0):
        raise ValueError("grid cell sizes must be positive")
    dims_arr = np.asarray(dims, dtype=np.int64).reshape(3)
    if np.any(dims_arr < 1):
        raise ValueError("grid dimensions must be at least 1")
    return Grid(contiguous(np.asarray(lo, dtype=np.float64).reshape(3)), contiguous(cell_arr),
                contiguous(dims_arr))


__all__ = ["MeshTree", "Grid", "sample_tree", "sample_grid", "TREE_TYPE", "GRID_TYPE", "contiguous", "make_grid",
           "INFO_NODES", "INFO_DEPTH", "INFO_CLOSED", "INFO_LEAF_SIZE", "N_INFO", "MAX_DEPTH", "STACK_SIZE",
           "FEATURE_FACE", "FEATURE_EDGE_0", "FEATURE_EDGE_1", "FEATURE_EDGE_2", "FEATURE_VERTEX_0",
           "FEATURE_VERTEX_1", "FEATURE_VERTEX_2", "OUTSIDE", "INSIDE", "ON_SURFACE", "UNDECIDED"]
