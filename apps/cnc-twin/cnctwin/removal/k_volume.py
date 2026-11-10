"""The ledger volume of a tri-dexel field: per dual cell, the family most normal to the surface through it
(docs/models/removal.md 1.1 "Volume").

A family's own volume ``a^2 sum(lengths)`` is the midpoint rule of the integral of its ray lengths over the plane
across it: second order where the lengths vary smoothly, first order where the surface is parallel to the family
(a wall along z makes the z lengths jump between columns, so a finishing layer 0.3 mm thick held a whole number of
node rows: -18 % at a 0.25 mm lattice, docs/convergence.md 2). Here every node owns its dual cell, the cube of side a
centred on it, and the three rays through the node give the material length inside the cell along each axis,
``L_x, L_y, L_z`` (exact to the quantum). For a plane through the cell at signed distance d from the node (d > 0:
the node in material) ``L_f = a / 2 + d / |n_f|`` clipped to [0, a], so the family most normal to the plane (the
largest |n_f|) has the chord least clipped and exact when the plane crosses the cell's four f-edges: the shortest
chord when the node is in material, the longest when it is in air. The cell's volume is ``a^2 x`` that chord and the
field's ledger volume their sum: an exact integer (half quanta: the cell faces lie half a lattice spacing from the
nodes, a half quantum when the spacing is an odd number of quanta), exact for planes along the lattice axes and
second order otherwise (edges and corners of the material inside a cell). The stock faces lie on cell faces, so a
block or tube stock reads its exact volume.

Bricks (abi ``BRICK`` nodes per axis, the last brick of an axis takes the rest) sum their cells: the domain keeps the
sums of level 0 per brick and recomputes the bricks whose change counter (``bver``) moved. A brick's first dual cell
along an axis starts half a spacing before its first node, inside the previous brick's node range: every writer marks
the next brick along the ray as well when a changed part reaches into that last half spacing (k_core.mark,
book_piece), so the counters cover the cells.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import abi as A
from .nbt import FIELD_T, I1, OPTS, i8, void

K = A.K_INLINE
BR = A.BRICK


@njit(void(FIELD_T, i8, i8, i8, i8, i8, i8, i8, i8, I1, I1), **OPTS)
def _distribute(f, lvl, fam, i0, n0, j0, n1, k0, n2, L, st):
    """Add the family-``fam`` material of the brick's nodes (i0 + n0, j0 + n1, k0 + n2 per axis) to ``L`` (doubled
    quanta, cell-major [i, j, k] flattened, family offset fam * n0 n1 n2) and, for the z family, the node's
    material state to ``st``."""
    cq = f.lv_cell_q[lvl]
    two_cq = 2 * cq
    dims = f.lv_dims[lvl]
    r0 = f.lv_ray0[lvl, fam]
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    ncell = n0 * n1 * n2
    off = fam * ncell
    # transverse (u, v) and along-ray (w) node ranges of the brick for this family
    if fam == 0:
        u0, nu, v0, nv, w0, nw = j0, n1, k0, n2, i0, n0
        dv = dims[2]
    elif fam == 1:
        u0, nu, v0, nv, w0, nw = i0, n0, k0, n2, j0, n1
        dv = dims[2]
    else:
        u0, nu, v0, nv, w0, nw = i0, n0, j0, n1, k0, n2
        dv = dims[1]
    # the cells' extent along the ray in doubled quanta: cell w spans [2 w cq - cq, 2 w cq + cq)
    lo_all = 2 * w0 * cq - cq
    hi_all = 2 * (w0 + nw) * cq - cq
    for a in range(nu):
        for b in range(nv):
            ray = r0 + (u0 + a) * dv + (v0 + b)
            n = np.int64(cnt[ray])
            if n == 0:
                continue
            blk = np.int64(link[ray])
            s = 0
            for kk in range(n):
                if kk < K:
                    q0 = np.int64(seg[ray, kk, 0])
                    q1 = np.int64(seg[ray, kk, 1])
                else:
                    if s == K:
                        blk = np.int64(pnext[blk])
                        s = 0
                    q0 = np.int64(pseg[blk, s, 0])
                    q1 = np.int64(pseg[blk, s, 1])
                    s += 1
                d0 = 2 * q0
                d1 = 2 * q1
                if d1 <= lo_all or d0 >= hi_all:
                    if d0 >= hi_all:
                        break
                    continue
                c0 = (d0 + cq) // two_cq
                c1 = (d1 - 1 + cq) // two_cq
                if c0 < w0:
                    c0 = w0
                if c1 > w0 + nw - 1:
                    c1 = w0 + nw - 1
                for c in range(c0, c1 + 1):
                    e0 = 2 * c * cq - cq
                    e1 = e0 + two_cq
                    x0 = d0 if d0 > e0 else e0
                    x1 = d1 if d1 < e1 else e1
                    if x1 <= x0:
                        continue
                    wc = c - w0
                    if fam == 0:
                        idx = (wc * n1 + a) * n2 + b
                    elif fam == 1:
                        idx = (a * n1 + wc) * n2 + b
                    else:
                        idx = (a * n1 + b) * n2 + wc
                    L[off + idx] += x1 - x0
                    if fam == 2:
                        node = 2 * c * cq
                        if d0 <= node and node < d1:
                            st[idx] = 1


@njit(i8(FIELD_T, i8, i8, I1, I1), **OPTS)
def brick_volume_dq(f, lvl, brick, L, st):
    """The ledger volume of one brick of a level in doubled quanta of length (times cell^2 QUANTUM_M / 2: m^3): per
    dual cell of the brick's nodes, the shortest of the three chords when the node is in material, the longest when
    it is in air. ``L`` (3 n^3) and ``st`` (n^3) are work arrays, n = BRICK + 1 (the last brick of an axis takes up
    to BRICK + 1 nodes)."""
    bd = f.lv_bdims[lvl]
    dims = f.lv_dims[lvl]
    rel = brick - f.lv_brick0[lvl]
    bz = rel % bd[2]
    by = (rel // bd[2]) % bd[1]
    bx = rel // (bd[2] * bd[1])
    i0 = bx * BR
    j0 = by * BR
    k0 = bz * BR
    n0 = (dims[0] - i0) if bx == bd[0] - 1 else BR
    n1 = (dims[1] - j0) if by == bd[1] - 1 else BR
    n2 = (dims[2] - k0) if bz == bd[2] - 1 else BR
    if n0 <= 0 or n1 <= 0 or n2 <= 0:
        return 0
    ncell = n0 * n1 * n2
    for c in range(3 * ncell):
        L[c] = 0
    for c in range(ncell):
        st[c] = 0
    for fam in range(3):
        _distribute(f, lvl, fam, i0, n0, j0, n1, k0, n2, L, st)
    total = np.int64(0)
    for c in range(ncell):
        lx = L[c]
        ly = L[ncell + c]
        lz = L[2 * ncell + c]
        if st[c] == 1:
            g = lx if lx < ly else ly
            if lz < g:
                g = lz
        else:
            g = lx if lx > ly else ly
            if lz > g:
                g = lz
        total += g
    return total


WORK_CELLS = (BR + 2) * (BR + 2) * (BR + 2)


@njit(void(FIELD_T, i8, I1, I1, I1, I1), **OPTS)
def bricks_volume_dq(f, lvl, bricks, out, L, st):
    """``brick_volume_dq`` of every brick index in ``bricks`` (absolute indices of level ``lvl``) into ``out``; work
    arrays ``L`` (3 WORK_CELLS) and ``st`` (WORK_CELLS) from the caller."""
    for i in range(bricks.shape[0]):
        out[i] = brick_volume_dq(f, lvl, bricks[i], L, st)


KERNELS = (brick_volume_dq, bricks_volume_dq)
INLINED = ()
