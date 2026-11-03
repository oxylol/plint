"""Meshing API (docs/models/removal.md "Meshing"): bricks of the tri-dexel field (and of the polar field
through its exact tri-dexel rebuild) to closed triangle meshes, incremental keyframes by dirty bricks.

* :func:`mesh_bricks` - one :class:`cnctwin.removal.abi.Mesh3` per brick (keyframes: changed bricks only).
* :func:`mesh_level` - the whole level as one welded mesh (bricks joined on their shared edge ids).
* :func:`dirty_bricks` / :func:`clear_dirty` - incremental keyframes (architecture 16 ``keyframes/``).
* :func:`to_geom_mesh` - a cnctwin.geom Mesh (vertex attribute ``surface_event``) for the file writers.
"""

from __future__ import annotations

import numpy as np

from . import abi as A
from .field import n_levels


def _scratch():
    n = A.BRICK + 1
    return np.zeros((n, n, n), np.int8), np.zeros((3, n, n, n), np.int32)


def _no_disp():
    return np.zeros(3), 0.0, np.zeros((0, 0, 0, 3))


def _disp_args(displacement):
    if displacement is None:
        return _no_disp()
    origin = np.ascontiguousarray(np.asarray(displacement.origin_S_m, dtype=float).reshape(3))
    cell = float(displacement.cell_m)
    grid = getattr(displacement, "total_m", None)
    if grid is None:
        grid = displacement.grid
    grid = np.ascontiguousarray(np.asarray(grid, dtype=float))
    if grid.ndim != 4 or grid.shape[-1] != 3 or grid.size == 0 or cell <= 0.0:
        return _no_disp()
    return origin, cell, grid


class Grid:
    """A displacement grid d(X) in S (same layout as core.bus.Displacement.total_m)."""

    def __init__(self, origin_S_m, cell_m, grid):
        self.origin_S_m = np.asarray(origin_S_m, dtype=float)
        self.cell_m = float(cell_m)
        self.grid = np.asarray(grid, dtype=float)


def brick_coords(field: A.DexelField, level: int, brick: int) -> tuple[int, int, int]:
    local = int(brick) - int(field.lv_brick0[level])
    bd = field.lv_bdims[level]
    return int(local // (bd[1] * bd[2])), int((local // bd[2]) % bd[1]), int(local % bd[2])


def surface_bricks(field: A.DexelField, level: int = 0) -> np.ndarray:
    """Bricks that can hold a surface: the brick or a +x/+y/+z neighbour (whose rays carry this brick's top
    node layer) holds material."""
    bd = field.lv_bdims[level]
    b0 = int(field.lv_brick0[level])
    nb = int(np.prod(bd))
    mat = (field.bmat[b0:b0 + nb] > 0).any(axis=1).reshape(tuple(int(x) for x in bd))
    need = mat.copy()
    for ax in range(3):
        sl = [slice(None)] * 3
        sl2 = [slice(None)] * 3
        sl[ax] = slice(0, -1)
        sl2[ax] = slice(1, None)
        need[tuple(sl)] |= mat[tuple(sl2)]
    # diagonal neighbours of the far corner share nodes as well
    shifted = np.zeros_like(mat)
    shifted[:-1, :-1, :-1] = mat[1:, 1:, 1:]
    need |= shifted
    for a, b in ((0, 1), (0, 2), (1, 2)):
        s = np.zeros_like(mat)
        sl = [slice(None)] * 3
        sl2 = [slice(None)] * 3
        sl[a] = slice(0, -1)
        sl[b] = slice(0, -1)
        sl2[a] = slice(1, None)
        sl2[b] = slice(1, None)
        s[tuple(sl)] = mat[tuple(sl2)]
        need |= s
    return np.flatnonzero(need.ravel()) + b0


def dirty_bricks(field: A.DexelField, level: int = 0) -> np.ndarray:
    b0 = int(field.lv_brick0[level])
    nb = int(np.prod(field.lv_bdims[level]))
    return np.flatnonzero(field.bdirty[b0:b0 + nb]) + b0


def clear_dirty(field: A.DexelField, bricks) -> None:
    field.bdirty[np.asarray(bricks, dtype=np.int64)] = 0


def mesh_bricks(field: A.DexelField, bricks, level: int = 0, displacement=None, start_capacity: int = 20000):
    """{brick: Mesh3} for the given bricks of a level (empty meshes are left out)."""
    from .k_mesh import mesh_brick
    from .mctable import tables

    count, tri = tables()
    votes, vid = _scratch()
    d_origin, d_cell, d_grid = _disp_args(displacement)
    out = {}
    cap_v = int(start_capacity)
    cap_f = 2 * int(start_capacity)
    for brick in np.asarray(bricks, dtype=np.int64):
        bx, by, bz = brick_coords(field, level, int(brick))
        while True:
            V = np.zeros((cap_v, 3))
            Fc = np.zeros((cap_f, 3), np.int32)
            S = np.zeros(cap_v, np.int32)
            E = np.zeros(cap_v, np.int64)
            counts = np.zeros(3, np.int64)
            status = mesh_brick(field, level, bx, by, bz, count, tri, V, Fc, S, E, counts, d_origin, d_cell, d_grid,
                                votes, vid)
            if status == 0:
                break
            cap_v *= 2
            cap_f *= 2
        nv, nf = int(counts[0]), int(counts[1])
        if nf:
            out[int(brick)] = A.Mesh3(V[:nv].copy(), Fc[:nf].copy(), S[:nv].copy(), E[:nv].copy())
    return out


def weld(meshes) -> A.Mesh3:
    """Join brick meshes on their global edge ids (shared vertices become one)."""
    meshes = [m for m in meshes if len(m.faces)]
    if not meshes:
        return A.Mesh3(np.zeros((0, 3)), np.zeros((0, 3), np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64))
    eid = np.concatenate([m.edge_id for m in meshes])
    V = np.concatenate([m.vertices for m in meshes])
    S = np.concatenate([m.sev for m in meshes])
    offsets = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    Fc = np.concatenate([m.faces.astype(np.int64) + off for m, off in zip(meshes, offsets)])
    uniq, first, inverse = np.unique(eid, return_index=True, return_inverse=True)
    return A.Mesh3(np.ascontiguousarray(V[first]), np.ascontiguousarray(inverse[Fc].astype(np.int32)),
                   np.ascontiguousarray(S[first]), np.ascontiguousarray(uniq))


def mesh_level(field: A.DexelField, level: int = 0, displacement=None) -> A.Mesh3:
    """The whole level as one closed, welded mesh."""
    return weld(list(mesh_bricks(field, surface_bricks(field, level), level, displacement).values()))


def to_geom_mesh(mesh: A.Mesh3, name: str = "part"):
    from ..geom.mesh import Mesh

    return Mesh(mesh.vertices, mesh.faces, vertex_attr={"surface_event": np.asarray(mesh.sev, np.int32)}, name=name)


def mesh_volume(mesh: A.Mesh3) -> float:
    V, Fc = mesh.vertices, mesh.faces.astype(np.int64)
    if len(Fc) == 0:
        return 0.0
    a, b, c = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def open_edges(mesh: A.Mesh3) -> int:
    """Directed edges without their opposite (0 for a closed, consistently oriented mesh)."""
    Fc = mesh.faces.astype(np.int64)
    if len(Fc) == 0:
        return 0
    e = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    n = int(mesh.vertices.shape[0]) + 1
    fwd = e[:, 0] * n + e[:, 1]
    rev = e[:, 1] * n + e[:, 0]
    fwd_sorted = np.sort(fwd)
    idx = np.searchsorted(fwd_sorted, rev)
    idx = np.clip(idx, 0, len(fwd_sorted) - 1)
    missing = fwd_sorted[idx] != rev
    # each directed edge must appear exactly once and its reverse exactly once
    uniq, cnt = np.unique(fwd, return_counts=True)
    return int(missing.sum()) + int((cnt > 1).sum())


__all__ = ["mesh_bricks", "mesh_level", "weld", "dirty_bricks", "clear_dirty", "surface_bricks", "to_geom_mesh",
           "mesh_volume", "open_edges", "Grid"]
