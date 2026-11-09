"""Interpolated material boundary of the tri-dexel field (docs/models/removal.md "Boundary between rays").

For a point x and a family f, the four rays of f around x (transverse cell corners) each give the signed
distance along the ray to their nearest endpoint, positive inside material: psi_i = +-|s - e_i|.
Bilinear interpolation of the four psi_i is D_f(x). For a plane boundary crossing all four rays the
endpoints are linear in the transverse coordinates, so D_f is exactly (signed distance to the plane) /
|n . e_f| and its zero set is the plane; for a curved boundary the interpolation error of the zero set is
a^2 kappa / (8 cos^2 theta) normal to the boundary (bilinear interpolation of a function with curvature
kappa / cos^3 theta over a cell of side a, de Boor 1978 ch. 2; theta the angle between the family and the
boundary normal), so at most a^2 kappa / 4 with the most normal family (theta <= 45 deg). The family most normal to the
boundary has the smallest spread of the four endpoints (max - min <= a tan(theta), theta the angle
between n and e_f, so <= a sqrt(2) for the best family); a family is valid when all four rays have an
endpoint within ``SPREAD_CELLS`` lattice cells. The array-level work is in k_core (inlined into the
kernels); these FIELD_T versions serve Python callers and tests.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from . import k_core as C
from .nbt import F1, FIELD_T, OPTS, RAW, b1, f8, i8

Q = A.QUANTUM_M
SPREAD_CELLS = C.SPREAD_CELLS


# level_at, dexel_sd and gradient serve the hot loop's refill planes (core.hot_kernels.surface_plane, every stale
# memory bin of a step): compiled without reference counting (they allocate nothing)
@njit(i8(FIELD_T, f8, f8, f8), **RAW)
def level_at(f, x, y, z):
    """Finest level whose authoritative box contains the point (0 otherwise)."""
    return C.level_at(f.lv_box_lo_m, f.lv_box_hi_m, f.lv_cell_m, f.info[A.FI_NLEVELS], x, y, z)


@njit(f8(FIELD_T, i8, i8, f8, f8, f8, F1), **RAW)
def dexel_sd(f, lvl, fam, x, y, z, out):
    """D_f at (x, y, z) in metres (positive inside). ``out`` = [spread (m), valid (1/0), unanimous inside
    (1), unanimous outside (-1) or mixed (0), smallest |psi| (m)]."""
    return C.sd(f.lv_cell_m, f.lv_origin_m, f.lv_dims, f.lv_cell_q, f.lv_ray0, f.cnt, f.seg, f.link, f.pseg,
                f.pnext, lvl, fam, x, y, z, out)


@njit(types.Tuple((b1, f8, i8))(FIELD_T, i8, f8, f8, f8, i8, F1), **OPTS)
def classify(f, lvl, x, y, z, fam0, work):
    """(inside, D of the chosen family (m), chosen family) at a point (k_core.classify)."""
    return C.classify(f.lv_cell_m, f.lv_origin_m, f.lv_dims, f.lv_cell_q, f.lv_ray0, f.cnt, f.seg, f.link, f.pseg,
                      f.pnext, lvl, x, y, z, fam0, work)


@njit(b1(FIELD_T, f8, f8, f8), **OPTS)
def point_inside(f, x, y, z):
    """Is the point inside material (finest level containing it; certain far from boundaries, the most
    normal family's interpolated boundary near them)."""
    work = np.empty(4)
    lvl = C.level_at(f.lv_box_lo_m, f.lv_box_hi_m, f.lv_cell_m, f.info[A.FI_NLEVELS], x, y, z)
    ins, d, fam = C.classify(f.lv_cell_m, f.lv_origin_m, f.lv_dims, f.lv_cell_q, f.lv_ray0, f.cnt, f.seg, f.link,
                             f.pseg, f.pnext, lvl, x, y, z, 2, work)
    return ins


@njit(types.void(FIELD_T, f8, f8, f8, i8, f8, F1), **RAW)
def gradient(f, x, y, z, fam, h, out):
    """Central-difference gradient of D_fam (unit vector, pointing into the material). Allocation-free (the six
    D values from k_core.sd_d, the sums in registers in the same order), reference counting off."""
    lv_cell_m = f.lv_cell_m
    lv_origin = f.lv_origin_m
    lv_dims = f.lv_dims
    lv_cell_q = f.lv_cell_q
    lv_ray0 = f.lv_ray0
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    lvl = C.level_at(f.lv_box_lo_m, f.lv_box_hi_m, lv_cell_m, f.info[A.FI_NLEVELS], x, y, z)
    gx = 0.0
    gy = 0.0
    gz = 0.0
    for k in range(6):
        axis = k // 2
        sign = 1.0 if k % 2 == 0 else -1.0
        px = x + (sign * h if axis == 0 else 0.0)
        py = y + (sign * h if axis == 1 else 0.0)
        pz = z + (sign * h if axis == 2 else 0.0)
        v = sign * C.sd_d(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam,
                          px, py, pz)
        if axis == 0:
            gx += v
        elif axis == 1:
            gy += v
        else:
            gz += v
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n > 0.0:
        out[0] = gx / n
        out[1] = gy / n
        out[2] = gz / n
    else:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0


KERNELS = (level_at, dexel_sd, classify, point_inside, gradient)
