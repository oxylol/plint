"""The ledger volume of a workpiece's tri-dexel field and the removed volume booked from it (docs/models/removal.md
1.1 "Volume", 13).

The field's ledger volume is the sum over level 0's dual cells of the chord of the family most normal to the surface
through the cell (:mod:`cnctwin.removal.k_volume`), an exact integer in doubled quanta of length (``dq``; one dq is
``cell^2 QUANTUM_M / 2`` of volume). :class:`VolumeCache` keeps it per brick of level 0 and recomputes only the bricks
whose change counters moved (the writers mark a brick whose first dual cell a change reached), so the change over a
macro step is the difference of two exact sums: the removed volume the core Ledger books is the field's own change,
and the loop's exact-volume check (field change = booked removal) closes to the integer. The cache is a memo of the
field's state, never checkpointed: a resumed run rebuilds it from the same field.
"""

from __future__ import annotations

import numpy as np

from . import abi as A


def factor_m3(field: A.DexelField) -> float:
    """Volume (m^3) per doubled quantum of the ledger sum (level 0)."""
    a = float(field.lv_cell_m[0])
    return a * a * A.QUANTUM_M * 0.5


def _bricks(field: A.DexelField) -> tuple[int, tuple]:
    b0 = int(field.lv_brick0[0])
    bd = tuple(int(v) for v in field.lv_bdims[0])
    return b0, bd


def _full_or_empty(field: A.DexelField) -> tuple[np.ndarray, np.ndarray]:
    """Per level-0 brick: (full, empty) where the brick's cells are certainly all material / all air, from the brick
    material sums (``bmat``) of the brick and of its lower neighbours (whose range the cells reach by half a
    spacing). Full: every ray of the brick's nodes covers its own brick and its lower neighbour along the ray."""
    b0, bd = _bricks(field)
    n = int(np.prod(bd))
    bmat = np.asarray(field.bmat[b0:b0 + n]).reshape(bd + (3,))
    dims = np.asarray(field.lv_dims[0], np.int64)
    cq = int(field.lv_cell_q[0])
    br = A.BRICK
    nodes = []
    for ax in range(3):
        cnt = np.full(bd[ax], br, np.int64)
        cnt[-1] = dims[ax] - br * (bd[ax] - 1)
        nodes.append(cnt)
    # the material length a brick's rays would hold if they covered the bmat range [32 b, 32 b + 32) cq (the last
    # brick of an axis: up to the last node) along the family's axis
    full = np.ones(bd, bool)
    empty = np.all(bmat == 0, axis=3)
    for fam in range(3):
        u, v = A.TRANS_U[fam], A.TRANS_V[fam]
        span = np.full(bd[fam], br * cq, np.int64)
        span[-1] = (dims[fam] - 1) * cq - br * (bd[fam] - 1) * cq
        shape = [1, 1, 1]
        rays = np.ones(bd, np.int64)
        for ax in (u, v):
            sh = [1, 1, 1]
            sh[ax] = bd[ax]
            rays = rays * nodes[ax].reshape(sh)
        shape[fam] = bd[fam]
        cap = rays * span.reshape(shape)
        own = bmat[..., fam] == cap
        lower = np.ones(bd, bool)
        sl_hi = [slice(None)] * 3
        sl_lo = [slice(None)] * 3
        sl_hi[fam] = slice(1, None)
        sl_lo[fam] = slice(None, -1)
        lower[tuple(sl_hi)] = own[tuple(sl_lo)]
        idx0 = [slice(None)] * 3
        idx0[fam] = 0
        lower[tuple(idx0)] = False          # the first brick's cells reach below the first node: never full
        full &= own & lower
        lo_empty = np.ones(bd, bool)
        lo_empty[tuple(sl_hi)] = bmat[..., fam][tuple(sl_lo)] == 0
        empty &= lo_empty
    return full, empty


def _cell_count(field: A.DexelField) -> np.ndarray:
    _, bd = _bricks(field)
    dims = np.asarray(field.lv_dims[0], np.int64)
    out = np.ones(bd, np.int64)
    for ax in range(3):
        cnt = np.full(bd[ax], A.BRICK, np.int64)
        cnt[-1] = dims[ax] - A.BRICK * (bd[ax] - 1)
        sh = [1, 1, 1]
        sh[ax] = bd[ax]
        out = out * cnt.reshape(sh)
    return out


_WORK: list = []


def _work():
    """The kernel's work arrays (one pair per process: the kernel is not re-entrant across threads, nogil callers
    would need their own)."""
    if not _WORK:
        from .k_volume import WORK_CELLS
        _WORK.append((np.zeros(3 * WORK_CELLS, np.int64), np.zeros(WORK_CELLS, np.int64)))
    return _WORK[0]


def compute_bricks(field: A.DexelField, bricks: np.ndarray) -> np.ndarray:
    """Ledger volume (dq) of the given level-0 brick indices (absolute), by the kernel."""
    from .k_volume import bricks_volume_dq

    idx = np.ascontiguousarray(np.asarray(bricks, np.int64))
    out = np.zeros(len(idx), np.int64)
    if len(idx):
        L, st = _work()
        bricks_volume_dq(field, 0, idx, out, L, st)
    return out


def full_volume_dq(field: A.DexelField) -> np.ndarray:
    """Every level-0 brick's ledger volume (dq), shaped as the brick array (bricks certainly full or empty from
    their material sums, the others by the kernel)."""
    b0, bd = _bricks(field)
    full, empty = _full_or_empty(field)
    vol = np.zeros(bd, np.int64)
    cq = int(field.lv_cell_q[0])
    vol[full] = (_cell_count(field) * 2 * cq)[full]
    rest = np.flatnonzero(~(full | empty).reshape(-1))
    if len(rest):
        vol.reshape(-1)[rest] = compute_bricks(field, rest + b0)
    return vol


def field_volume_dq(field: A.DexelField) -> int:
    """The ledger volume of a field from scratch (dq, Python int)."""
    return int(full_volume_dq(field).sum())


def field_volume_m3(field: A.DexelField) -> float:
    return float(field_volume_dq(field)) * factor_m3(field)


class VolumeCache:
    """Per level-0 brick the ledger volume (dq) and the brick change counter it was computed at."""

    def __init__(self, field: A.DexelField):
        b0, bd = _bricks(field)
        n = int(np.prod(bd))
        self.shape = bd
        self.vol = full_volume_dq(field).reshape(-1)
        self.ver = np.array(field.bver[b0:b0 + n], copy=True)
        self.total = int(self.vol.sum())
        self.recomputed = 0                      # bricks recomputed since made (cost statistics)

    def matches(self, field: A.DexelField) -> bool:
        _, bd = _bricks(field)
        return bd == self.shape

    def update(self, field: A.DexelField) -> int:
        """Bring the sums to the field's state; returns the volume removed since the last update (dq, may be 0)."""
        b0, bd = _bricks(field)
        n = len(self.ver)
        ver = field.bver[b0:b0 + n]
        changed = ver != self.ver
        if not changed.any():
            return 0
        # the writers mark the next brick along a ray when a change reaches the half spacing its first dual cell
        # takes from the brick before (k_core.mark), so the changed bricks are all the cells that changed
        idx = np.flatnonzero(changed)
        # bricks certainly all material or all air (their material sums) need no kernel: the neighbours of a cut
        # are mostly bulk or air
        full, empty = _full_or_empty(field)
        full = full.reshape(-1)[idx]
        empty = empty.reshape(-1)[idx]
        new = np.zeros(len(idx), np.int64)
        if full.any():
            new[full] = (_cell_count(field).reshape(-1)[idx] * 2 * int(field.lv_cell_q[0]))[full]
        rest = ~(full | empty)
        if rest.any():
            new[rest] = compute_bricks(field, idx[rest] + b0)
        old = int(self.vol[idx].sum())
        self.vol[idx] = new
        self.ver[:] = ver
        now = int(new.sum())
        self.total += now - old
        self.recomputed += int(rest.sum())
        return old - now


def cache(wp) -> VolumeCache:
    """The workpiece's volume cache, made (or remade for a new field layout) when missing. A made cache is at the
    field's current state: whatever changed the field before it is in its total, not in a change."""
    vc = getattr(wp, "vol_cache", None)
    if vc is None or not vc.matches(wp.field):
        vc = VolumeCache(wp.field)
        wp.vol_cache = vc
    return vc


def removed_dq(wp) -> int:
    """The ledger volume the field lost since the last call (dq, Python int; negative when material was added)."""
    vc = getattr(wp, "vol_cache", None)
    if vc is None or not vc.matches(wp.field):
        raise RuntimeError("removal ledger: the volume cache was not made before the field changed")
    return vc.update(wp.field)


def resync(wp) -> None:
    """Take the field's current state as the reference without booking it (a rebuild from the polar field, a
    transfer)."""
    cache(wp).update(wp.field)


def volume_dq(wp) -> int:
    """The workpiece's ledger volume now (dq). Brings the cache to the field's state: a change not booked before
    this call is not booked by it either (the exact-volume checks then see it)."""
    vc = cache(wp)
    vc.update(wp.field)
    return vc.total


def volume_m3(wp) -> float:
    return float(volume_dq(wp)) * factor_m3(wp.field)


def split(total: int, weights) -> list[int]:
    """``total`` split in proportion to non-negative integer ``weights`` exactly (largest remainder; ties to the
    lower index); all of it to the first entry when every weight is zero."""
    w = [max(int(x), 0) for x in weights]
    if not w:
        return []
    s = sum(w)
    if s == 0:
        return [int(total)] + [0] * (len(w) - 1)
    sign = -1 if total < 0 else 1
    t = abs(int(total))
    base = [t * x // s for x in w]
    rem = t - sum(base)
    order = sorted(range(len(w)), key=lambda i: (-((t * w[i]) % s), i))
    for i in order[:rem]:
        base[i] += 1
    return [sign * b for b in base]


__all__ = ["factor_m3", "field_volume_dq", "field_volume_m3", "VolumeCache", "cache", "removed_dq", "resync",
           "volume_dq", "volume_m3", "split", "compute_bricks", "full_volume_dq"]
