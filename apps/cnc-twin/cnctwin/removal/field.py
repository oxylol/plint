"""Construction, growth and bookkeeping of the tri-dexel field (Python side, L2; docs/models/removal.md
"Field").

* :func:`make_field` - the lattice over a stock box (plus a margin of empty nodes so meshes close) and
  the refine levels (``physics.refine`` boxes, finer by an integer factor); empty rays, an initial pool.
* :func:`grow_pool` - a field with more overflow blocks (deterministic: new blocks join the free list in
  index order); kernels that report ``FI_OVERFLOW`` are repeated after it (they are idempotent).
* :func:`volume_m3` - exact integer volume of a family of a level times its constant factor.
* :func:`to_arrays` / :func:`from_arrays` - checkpoint form (all arrays, bit-identical round trip).
* :func:`memory_bytes` - the field's array memory (reported against the 300 MB budget).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import abi as A


@dataclass(frozen=True)
class LevelSpec:
    lo_m: np.ndarray        # authoritative box (S)
    hi_m: np.ndarray
    cell_q: int             # lattice spacing in quanta
    margin: int             # empty node layers beyond the box
    origin_m: np.ndarray | None = None   # explicit node origin (transfers); None: from the box
    dims: np.ndarray | None = None


def cell_quanta(lattice_m: float) -> int:
    """Lattice spacing in whole quanta (the lattice must be a multiple of 10 nm; it is rounded to one)."""
    q = int(round(float(lattice_m) / A.QUANTUM_M))
    if q < 1:
        raise ValueError(f"lattice {lattice_m!r} m is below the fixed-point quantum {A.QUANTUM_M} m")
    return q


def _level_geometry(lo, hi, cell_q: int, margin: int):
    a = cell_q * A.QUANTUM_M
    size = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    if np.any(size <= 0.0):
        raise ValueError(f"empty box {np.asarray(lo).tolist()} .. {np.asarray(hi).tolist()}")
    cells = np.maximum(1, np.ceil(size / a - 1e-9).astype(np.int64))
    dims = cells + 2 * int(margin)
    # nodes at the cell centres of the box (the box faces lie half way between nodes), margin outside
    origin = np.asarray(lo, dtype=float) + 0.5 * a - margin * a
    return origin, dims


def make_field(lo_S_m, hi_S_m, lattice_m: float, refine=(), margin_nodes: int = 1, pool_blocks: int | None = None,
               workpiece: int = 0) -> A.DexelField:
    """An empty field over the box [lo, hi] (S) at ``lattice_m``, with refine levels.

    ``refine``: iterable of (lo_m [3], hi_m [3], factor) boxes in S; each becomes a level of spacing
    lattice / factor (rounded to whole quanta) over the box clipped to the stock box, snapped outward to the
    base lattice. Margin layers keep the outermost nodes in air so surfaces close (meshing, gating)."""
    lo = np.asarray(lo_S_m, dtype=float).reshape(3)
    hi = np.asarray(hi_S_m, dtype=float).reshape(3)
    cq0 = cell_quanta(lattice_m)
    a0 = cq0 * A.QUANTUM_M
    levels = [LevelSpec(lo, hi, cq0, int(margin_nodes))]
    for item in refine or ():
        r_lo, r_hi, factor = item
        factor = int(factor)
        if factor < 2:
            raise ValueError("a refine factor must be at least 2")
        b_lo = np.maximum(np.asarray(r_lo, dtype=float).reshape(3), lo)
        b_hi = np.minimum(np.asarray(r_hi, dtype=float).reshape(3), hi)
        if np.any(b_hi <= b_lo):
            raise ValueError(f"refine box {np.asarray(r_lo).tolist()} .. {np.asarray(r_hi).tolist()} "
                             f"does not overlap the stock")
        # snap outward to the base lattice cells so coarse and fine regions meet on base cell faces
        b_lo = lo + np.floor((b_lo - lo) / a0 + 1e-9) * a0
        b_hi = lo + np.ceil((b_hi - lo) / a0 - 1e-9) * a0
        b_hi = np.minimum(b_hi, lo + np.ceil((hi - lo) / a0 - 1e-9) * a0)
        cq = max(1, int(round(cq0 / factor)))
        levels.append(LevelSpec(b_lo, b_hi, cq, int(margin_nodes)))
    return _allocate(levels, pool_blocks, workpiece)


def _allocate(levels: list[LevelSpec], pool_blocks: int | None, workpiece: int) -> A.DexelField:
    nl = len(levels)
    lv_origin = np.zeros((nl, 3))
    lv_cell = np.zeros(nl)
    lv_cq = np.zeros(nl, np.int64)
    lv_dims = np.zeros((nl, 3), np.int64)
    lv_ray0 = np.zeros((nl, 3), np.int64)
    lv_bdims = np.zeros((nl, 3), np.int64)
    lv_brick0 = np.zeros(nl, np.int64)
    lv_lo = np.zeros((nl, 3))
    lv_hi = np.zeros((nl, 3))
    nray = 0
    nbrick = 0
    for l, spec in enumerate(levels):
        if spec.origin_m is not None:
            origin, dims = np.asarray(spec.origin_m, dtype=float), np.asarray(spec.dims, dtype=np.int64)
        else:
            origin, dims = _level_geometry(spec.lo_m, spec.hi_m, spec.cell_q, spec.margin)
        lv_origin[l] = origin
        lv_cq[l] = spec.cell_q
        lv_cell[l] = spec.cell_q * A.QUANTUM_M
        lv_dims[l] = dims
        lv_lo[l] = spec.lo_m
        lv_hi[l] = spec.hi_m
        for fam in range(3):
            u, v = A.TRANS_U[fam], A.TRANS_V[fam]
            lv_ray0[l, fam] = nray
            nray += int(dims[u] * dims[v])
        bd = np.maximum(1, np.ceil((dims - 1) / A.BRICK).astype(np.int64))
        lv_bdims[l] = bd
        lv_brick0[l] = nbrick
        nbrick += int(np.prod(bd))
        span = (dims - 1) * spec.cell_q
        if np.any(span >= 2 ** 31 - 1):
            raise ValueError("a level spans more than the int32 fixed-point range (21 m)")
    if pool_blocks is None:
        pool_blocks = max(1024, nray // 64)
    k = A.K_INLINE
    info = np.zeros(A.N_FI, np.int64)
    info[A.FI_NLEVELS] = nl
    info[A.FI_NRAYS] = nray
    info[A.FI_WORKPIECE] = int(workpiece)
    field = A.DexelField(
        lv_origin, lv_cell, lv_cq, lv_dims, lv_ray0, lv_bdims, lv_brick0, lv_lo, lv_hi,
        np.zeros(nray, np.int32), np.zeros((nray, k, 2), np.int32), np.zeros((nray, k, 2), np.int32),
        np.full(nray, -1, np.int32), np.zeros((0, k, 2), np.int32), np.zeros((0, k, 2), np.int32),
        np.full(0, -1, np.int32), np.zeros((nbrick, 3), np.int64), np.zeros(nbrick, np.int8),
        np.zeros(nbrick, np.int8), np.zeros(nbrick, np.int32), info)
    field.info[A.FI_FREE_HEAD] = -1
    return grow_pool(field, int(pool_blocks))


def make_field_lattice(origin_m, dims, cell_q: int, box_lo_m, box_hi_m, pool_blocks: int | None = None,
                       workpiece: int = 0) -> A.DexelField:
    """A single-level empty field with an explicit node lattice (exact transfers)."""
    spec = LevelSpec(np.asarray(box_lo_m, dtype=float), np.asarray(box_hi_m, dtype=float), int(cell_q), 0,
                     np.asarray(origin_m, dtype=float).reshape(3), np.asarray(dims, dtype=np.int64).reshape(3))
    return _allocate([spec], pool_blocks, workpiece)


def grow_pool(field: A.DexelField, add_blocks: int) -> A.DexelField:
    """The same field with ``add_blocks`` more pool blocks (appended to the free list in index order)."""
    add = int(max(0, add_blocks))
    if add == 0:
        return field
    old = field.pseg.shape[0]
    k = A.K_INLINE
    pseg = np.zeros((old + add, k, 2), np.int32)
    psev = np.zeros((old + add, k, 2), np.int32)
    pnext = np.full(old + add, -1, np.int32)
    pseg[:old] = field.pseg
    psev[:old] = field.psev
    pnext[:old] = field.pnext
    # new blocks: old .. old + add - 1 linked in order, then the old free list
    new_ids = np.arange(old, old + add, dtype=np.int64)
    pnext[new_ids[:-1]] = new_ids[1:]
    pnext[new_ids[-1]] = field.info[A.FI_FREE_HEAD]
    info = field.info.copy()
    info[A.FI_FREE_HEAD] = old
    info[A.FI_N_FREE] += add
    info[A.FI_N_BLOCKS] = old + add
    info[A.FI_OVERFLOW] = 0
    return field._replace(pseg=pseg, psev=psev, pnext=pnext, info=info)


def ensure_free(field: A.DexelField, min_free: int) -> A.DexelField:
    """Grow the pool (doubling, at least ``min_free`` free blocks) when fewer are free."""
    free = int(field.info[A.FI_N_FREE])
    if free >= min_free and not field.info[A.FI_OVERFLOW]:
        return field
    grow = max(int(min_free) - free, int(field.info[A.FI_N_BLOCKS]), 1024)
    return grow_pool(field, grow)


def n_levels(field: A.DexelField) -> int:
    return int(field.info[A.FI_NLEVELS])


def level_rays(field: A.DexelField, level: int, fam: int) -> int:
    u, v = A.TRANS_U[fam], A.TRANS_V[fam]
    return int(field.lv_dims[level, u] * field.lv_dims[level, v])


def volume_factor_m3(field: A.DexelField, level: int) -> float:
    """Volume per quantum of family length of ``level``: cell^2 x quantum."""
    a = float(field.lv_cell_m[level])
    return a * a * A.QUANTUM_M


def volume_m3(field: A.DexelField, level: int = 0, fam: int = A.FAM_Z) -> float:
    """Material volume of a level as the exact integer sum of its family's lengths times the factor."""
    from .k_ray import family_length

    return float(family_length(field, level, fam)) * volume_factor_m3(field, level)


def family_lengths(field: A.DexelField, level: int = 0) -> tuple[int, int, int]:
    from .k_ray import family_length

    return tuple(int(family_length(field, level, f)) for f in range(3))


def node_position(field: A.DexelField, level: int, i: int, j: int, k: int) -> np.ndarray:
    return field.lv_origin_m[level] + np.array([i, j, k], dtype=float) * field.lv_cell_m[level]


def stock_box(field: A.DexelField) -> tuple[np.ndarray, np.ndarray]:
    return field.lv_box_lo_m[0].copy(), field.lv_box_hi_m[0].copy()


def lattice_box(field: A.DexelField, level: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Box spanned by the nodes of a level."""
    lo = field.lv_origin_m[level].copy()
    hi = lo + (field.lv_dims[level] - 1) * field.lv_cell_m[level]
    return lo, hi


def memory_bytes(field: A.DexelField, target: A.TargetField | None = None) -> int:
    total = sum(int(np.asarray(a).nbytes) for a in field)
    if target is not None:
        total += sum(int(np.asarray(a).nbytes) for a in target)
    return total


def estimate_bytes(size_m, lattice_m: float, margin_nodes: int = 1, pool_fraction: float = 1.0 / 64.0) -> int:
    """Array memory of a base level over a box of ``size_m`` (for twin check / the budget report)."""
    a = cell_quanta(lattice_m) * A.QUANTUM_M
    dims = np.maximum(1, np.ceil(np.asarray(size_m, dtype=float) / a - 1e-9).astype(np.int64)) + 2 * margin_nodes
    nray = int(dims[1] * dims[2] + dims[0] * dims[2] + dims[0] * dims[1])
    per_ray = 4 + 2 * A.K_INLINE * 2 * 4 + 4
    pool = max(1024, int(nray * pool_fraction)) * (2 * A.K_INLINE * 2 * 4 + 4)
    bricks = int(np.prod(np.maximum(1, np.ceil((dims - 1) / A.BRICK)))) * (3 * 8 + 1 + 1 + 4)
    return nray * per_ray + pool + bricks


# --------------------------------------------------------------------------- checkpoint
# The field's own arrays, not copies: a checkpoint writes them at once (the field does not change while it is written),
# and a copy of a refined field (8.3 M rays, 0.72 GB with its target in examples/thin-wall-al7075) doubled the run's
# memory at every checkpoint. Restoring takes the arrays read from the file (fresh, writable, C-contiguous).
def to_arrays(field: A.DexelField, prefix: str = "") -> dict:
    return {prefix + name: np.asarray(getattr(field, name)) for name in A.DexelField._fields}


def from_arrays(arrays: dict, prefix: str = "") -> A.DexelField:
    return A.DexelField(*(_writable(arrays[prefix + name]) for name in A.DexelField._fields))


def target_to_arrays(target: A.TargetField, prefix: str = "") -> dict:
    return {prefix + name: np.asarray(getattr(target, name)) for name in A.TargetField._fields}


def target_from_arrays(arrays: dict, prefix: str = "") -> A.TargetField:
    return A.TargetField(*(_writable(arrays[prefix + name]) for name in A.TargetField._fields))


def _writable(array) -> np.ndarray:
    array = np.ascontiguousarray(array)
    return array if array.flags.writeable else array.copy()


def copy_field(field: A.DexelField) -> A.DexelField:
    return A.DexelField(*(np.array(a, copy=True) for a in field))


def fields_equal(a: A.DexelField, b: A.DexelField) -> bool:
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def finest_level_at(field: A.DexelField, p_S_m) -> int:
    """Finest level whose authoritative box contains the point (level 0 otherwise)."""
    p = np.asarray(p_S_m, dtype=float)
    best, best_cell = 0, float(field.lv_cell_m[0])
    for l in range(1, n_levels(field)):
        if np.all(p >= field.lv_box_lo_m[l]) and np.all(p <= field.lv_box_hi_m[l]) and field.lv_cell_m[l] < best_cell:
            best, best_cell = l, float(field.lv_cell_m[l])
    return best


def brick_box(field: A.DexelField, brick: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """(level, brick coords, node lo index, node hi index (inclusive)) of a linear brick index."""
    nl = n_levels(field)
    level = 0
    for l in range(nl):
        if field.lv_brick0[l] <= brick:
            level = l
    local = brick - int(field.lv_brick0[level])
    bd = field.lv_bdims[level]
    bz = local % bd[2]
    by = (local // bd[2]) % bd[1]
    bx = local // (bd[2] * bd[1])
    coords = np.array([bx, by, bz], np.int64)
    n_lo = coords * A.BRICK
    n_hi = np.minimum(n_lo + A.BRICK, field.lv_dims[level] - 1)
    return level, coords, n_lo, n_hi


def math_isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
