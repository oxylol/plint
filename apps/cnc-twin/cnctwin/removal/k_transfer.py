"""Exact remap of a tri-dexel level under a lattice symmetry (docs/models/removal.md "Transfers").

A transform that maps the lattice onto itself - a signed permutation of the axes (rotations by multiples of
90 degrees about the S axes, mirror-free) plus a translation by whole cells - maps every ray onto a ray and
every endpoint onto an endpoint, so the new field is the old one re-indexed: new axis i is old axis perm[i]
times sign[i]; a ray whose axial axis is reversed has its segments mirrored (q -> L - q).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import abi as A
from . import k_core as C
from .nbt import FIELD_T, I1, OPTS, SCRATCH_T, i8


@njit(i8(FIELD_T, FIELD_T, I1, I1, SCRATCH_T), **OPTS)
def remap_level0(old, new, perm, sign, sc):
    """Fill level 0 of ``new`` (empty, dims[i] = old dims[perm[i]]) from level 0 of ``old``. Returns 0 or -1
    (pool overflow: grow new's pool and call again on a fresh field)."""
    for fam_n in range(3):
        u_n = 1 if fam_n == 0 else 0
        v_n = 2 if fam_n < 2 else 1
        fam_o = perm[fam_n]
        nu_n = new.lv_dims[0, u_n]
        nv_n = new.lv_dims[0, v_n]
        lq_n = (new.lv_dims[0, fam_n] - 1) * new.lv_cell_q[0]
        for iu_n in range(nu_n):
            for iv_n in range(nv_n):
                idx_o = np.zeros(3, np.int64)
                # node indices of the new ray in new axes -> old axes
                idx_n = np.zeros(3, np.int64)
                idx_n[u_n] = iu_n
                idx_n[v_n] = iv_n
                for i in range(3):
                    if i == fam_n:
                        continue
                    k = idx_n[i] if sign[i] > 0 else new.lv_dims[0, i] - 1 - idx_n[i]
                    idx_o[perm[i]] = k
                u_o = 1 if fam_o == 0 else 0
                v_o = 2 if fam_o < 2 else 1
                ray_o = C.rindex(old.lv_ray0, old.lv_dims, 0, fam_o, idx_o[u_o], idx_o[v_o])
                n = C.load(old.cnt, old.seg, old.sev, old.link, old.pseg, old.psev, old.pnext, ray_o, sc.lo, sc.hi,
                           sc.elo, sc.ehi)
                if n <= 0:
                    continue
                if sign[fam_n] > 0:
                    for k in range(n):
                        sc.nlo[k] = sc.lo[k]
                        sc.nhi[k] = sc.hi[k]
                        sc.nelo[k] = sc.elo[k]
                        sc.nehi[k] = sc.ehi[k]
                else:
                    for k in range(n):
                        j = n - 1 - k
                        sc.nlo[k] = lq_n - sc.hi[j]
                        sc.nhi[k] = lq_n - sc.lo[j]
                        sc.nelo[k] = sc.ehi[j]
                        sc.nehi[k] = sc.elo[j]
                ray_n = C.rindex(new.lv_ray0, new.lv_dims, 0, fam_n, iu_n, iv_n)
                if C.store(new.cnt, new.seg, new.sev, new.link, new.pseg, new.psev, new.pnext, new.info, ray_n,
                           sc.nlo, sc.nhi, sc.nelo, sc.nehi, n) < 0:
                    return -1
                for k in range(n):
                    sc.plo[k] = sc.nlo[k]
                    sc.phi[k] = sc.nhi[k]
                C.account(new.lv_cell_q, new.lv_bdims, new.lv_brick0, new.bmat, new.bdirty, new.bconn, new.bver, 0,
                          fam_n, iu_n, iv_n, sc.plo, sc.phi, n, 1)
    new.info[A.FI_VERSION] += 1
    return 0


KERNELS = (remap_level0,)
