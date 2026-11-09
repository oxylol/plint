"""Meshing API (docs/models/removal.md "Meshing"): bricks of the tri-dexel field (and of the polar field
through its exact tri-dexel rebuild) to closed triangle meshes, incremental keyframes by dirty bricks.

* :func:`mesh_bricks` - one :class:`cnctwin.removal.abi.Mesh3` per brick (keyframes: changed bricks only).
* :func:`mesh_level` - the whole level as one welded mesh (bricks joined on their shared edge ids).
* :func:`dirty_bricks` / :func:`clear_dirty` - incremental keyframes (architecture 16 ``keyframes/``).
* :func:`to_geom_mesh` - a cnctwin.geom Mesh (vertex attribute ``surface_event``) for the file writers.
* :func:`planar_patches` - a brick mesh with its flat patches as fans (``mesh_level(..., planar=True)``: the same
  surface in far fewer triangles; the end-of-operation part of a fine lattice, ``planar_wanted``).
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


def mesh_level(field: A.DexelField, level: int = 0, displacement=None, planar: bool = False) -> A.Mesh3:
    """The whole level as one closed, welded mesh. ``planar``: every brick's flat patches become fans
    (:func:`planar_patches`), brick by brick as they are meshed (the fine brick meshes are never all in memory);
    the material configuration only (no displacement)."""
    bricks = surface_bricks(field, level)
    if not planar or displacement is not None:
        return weld(list(mesh_bricks(field, bricks, level, displacement).values()))
    pieces = []
    next_id = -1
    for brick in bricks:
        one = mesh_bricks(field, [int(brick)], level)
        if not one:
            continue
        lo, hi = brick_box(field, level, int(brick))
        piece, next_id = planar_patches(one[int(brick)], lo, hi, next_id)
        pieces.append(piece)
    return weld(pieces)


PLANAR_TRIANGLE_BUDGET = 3_000_000      # a level whose surface bricks could hold more triangles is meshed planar


def planar_wanted(field: A.DexelField, level: int = 0, budget: int = PLANAR_TRIANGLE_BUDGET) -> bool:
    """Whether a whole-level mesh could exceed ``budget`` triangles (two per brick face cell of every surface
    brick): examples/pocket-islands-al6061 at 0.075 mm, about 9.5 M triangles, outgrew the 1.8 GB cap with the part
    files and the inspection of its end (docs/models/loop.md 15)."""
    return int(len(surface_bricks(field, level))) * 2 * A.BRICK * A.BRICK > int(budget)


def brick_box(field: A.DexelField, level: int, brick: int) -> tuple[np.ndarray, np.ndarray]:
    """The node box of a brick in S (its outermost node planes: the faces it shares with its neighbours)."""
    b = np.array(brick_coords(field, level, brick), dtype=np.int64) * A.BRICK
    top = np.minimum(b + A.BRICK, np.asarray(field.lv_dims[level], dtype=np.int64) - 1)
    o = np.asarray(field.lv_origin_m[level], dtype=float)
    a = float(field.lv_cell_m[level])
    return o + b * a, o + top * a


def planar_patches(mesh: A.Mesh3, lo, hi, next_id: int = -1, tol_m: float = 1e-9) -> tuple[A.Mesh3, int]:
    """(mesh, next free id) with the brick's flat patches as fans: every connected set of coplanar triangles that
    has a vertex strictly inside the brick, all of whose triangles lie in the set's plane, and whose boundary is one
    convex loop, becomes the fan of its boundary edges about the loop's centroid (on the plane to round-off). The
    vertices on the brick's faces (shared with the neighbours), on edges between planes and of any non-flat
    neighbourhood stay, so the bricks weld as before without a crack, and the surface is the same surface: a flat
    32 x 32 patch of 2048 triangles becomes 128. New vertices get the ids next_id, next_id - 1, ... (negative: no
    lattice edge) and the surface event most of the vertices they replace had."""
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    nv, nf = len(V), len(F)
    if nf < 4:
        return mesh, next_id
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ar = np.linalg.norm(cr, axis=1)
    good = ar > 0.0
    n = np.zeros_like(cr)
    n[good] = cr[good] / ar[good, None]
    d = np.einsum("ij,ij->i", n, V[F[:, 0]])
    # edges and the faces on either side
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    ef = np.tile(np.arange(nf), 3)
    key = np.minimum(e[:, 0], e[:, 1]) * nv + np.maximum(e[:, 0], e[:, 1])
    order = np.argsort(key, kind="stable")
    ks = key[order]
    pair = np.nonzero(ks[1:] == ks[:-1])[0]
    fa, fb = ef[order[pair]], ef[order[pair + 1]]
    counts = np.bincount(np.searchsorted(np.unique(ks), ks), minlength=0)
    open_edge = order[np.repeat(counts == 1, counts)] if len(counts) else np.zeros(0, np.int64)
    same = good[fa] & good[fb] & (np.linalg.norm(n[fa] - n[fb], axis=1) <= 1e-9) & \
        (np.abs(np.einsum("ij,ij->i", n[fa], V[F[fb, 0]]) - d[fa]) <= tol_m) & \
        (np.abs(np.einsum("ij,ij->i", n[fa], V[F[fb, 1]]) - d[fa]) <= tol_m) & \
        (np.abs(np.einsum("ij,ij->i", n[fa], V[F[fb, 2]]) - d[fa]) <= tol_m)
    graph = coo_matrix((np.ones(int(same.sum())), (fa[same], fb[same])), shape=(nf, nf))
    _, region = connected_components(graph, directed=False)
    region = np.where(good, region, -1 - np.arange(nf))          # a degenerate face is a region of its own
    # vertices that must stay: on the brick's faces, on the piece's open boundary, of degenerate faces, or whose
    # faces lie in more than one region (edges between planes, curved surface)
    keep = np.zeros(nv, bool)
    keep |= np.any((np.abs(V - lo) <= tol_m) | (np.abs(V - hi) <= tol_m), axis=1)
    if len(open_edge):
        keep[e[open_edge].ravel()] = True
    keep[F[~good].ravel()] = True
    vf_v = F.ravel()
    vf_r = np.repeat(region, 3)
    rmin = np.full(nv, np.iinfo(np.int64).max)
    rmax = np.full(nv, np.iinfo(np.int64).min)
    np.minimum.at(rmin, vf_v, vf_r)
    np.maximum.at(rmax, vf_v, vf_r)
    keep |= rmin != rmax
    removable_regions = np.unique(vf_r[~keep[vf_v]])
    removable_regions = removable_regions[removable_regions >= 0]
    if len(removable_regions) == 0:
        return mesh, next_id
    drop_face = np.zeros(nf, bool)
    new_v, new_s, new_f = [], [], []
    S = np.asarray(mesh.sev)
    for r in removable_regions:
        faces = np.nonzero(region == r)[0]
        sub = F[faces]
        se = np.concatenate([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]])
        k = np.minimum(se[:, 0], se[:, 1]) * nv + np.maximum(se[:, 0], se[:, 1])
        uk, cnt = np.unique(k, return_counts=True)
        bnd = se[np.isin(k, uk[cnt == 1])]
        if len(bnd) < 3 or not keep[bnd.ravel()].all():
            continue
        # one loop: each boundary vertex starts exactly one boundary edge, and following them visits all
        start = {int(a): int(b) for a, b in bnd}
        if len(start) != len(bnd):
            continue
        loop = [int(bnd[0, 0])]
        while len(loop) <= len(bnd):
            nxt = start.get(loop[-1])
            if nxt is None or nxt == loop[0]:
                break
            loop.append(nxt)
        if len(loop) != len(bnd) or start.get(loop[-1]) != loop[0]:
            continue
        P = V[loop]
        nr = n[faces[0]]
        turn = np.einsum("ij,j->i", np.cross(np.roll(P, -1, axis=0) - P, np.roll(P, -2, axis=0) - np.roll(P, -1, axis=0)),
                         nr)
        if np.any(turn < -1e-18) or not np.any(turn > 0.0):
            continue                      # not convex (or no area): left as it is
        c = P.mean(axis=0)
        inner = np.setdiff1d(np.unique(sub.ravel()), loop)
        vals, votes = np.unique(S[inner], return_counts=True) if len(inner) else (np.array([S[loop[0]]]), [1])
        idx = nv + len(new_v)
        new_v.append(c)
        new_s.append(int(vals[int(np.argmax(votes))]))
        new_f.append(np.column_stack([np.full(len(loop), idx), loop, np.roll(loop, -1)]))
        drop_face[faces] = True
    if not new_v:
        return mesh, next_id
    F2 = np.concatenate([F[~drop_face]] + new_f)
    V2 = np.concatenate([V, np.asarray(new_v)])
    S2 = np.concatenate([S, np.asarray(new_s, dtype=S.dtype)])
    E2 = np.concatenate([np.asarray(mesh.edge_id, dtype=np.int64),
                         next_id - np.arange(len(new_v), dtype=np.int64)])
    used = np.zeros(len(V2), bool)
    used[F2.ravel()] = True
    remap = np.cumsum(used) - 1
    out = A.Mesh3(np.ascontiguousarray(V2[used]), np.ascontiguousarray(remap[F2].astype(np.int32)),
                  np.ascontiguousarray(S2[used]), np.ascontiguousarray(E2[used]))
    return out, next_id - len(new_v)


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
