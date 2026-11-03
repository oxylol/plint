"""Numba kernels of the coolant domain (docs/models/coolant.md "Kernels"; rules: architecture 5.5).

* :func:`trace_rays` - every ray of an occlusion job in one call: the jet trajectory as a polyline
  (liquid: gravity, skin friction of the coherent column, Schiller-Naumann drag of the drops after
  break-up, explicit midpoint steps whose length keeps the chord within the sagitta tolerance; gas:
  straight, centreline decay), tested segment by segment against the rotating tool envelopes (exact
  quadratic per radius-table interval), oriented boxes (slabs), the in-process workpiece bricks
  (Amanatides & Woo 1987 3-D DDA over bricks, nested DDA over the sub-cells of partial bricks), the
  chip pile height field (2-D DDA) and the enclosure; zone entries and closest approaches are
  followed along the unobstructed trajectory up to ``s_aim`` (what the jet is aimed at).
* :func:`hit_meshes` - the stored polylines against one static mesh tree (cnctwin.geom.bvh) in its
  own frame, up to each ray's current first hit.
* :func:`film_map` - wetted cells (downhill flood fill of the draining film over the pile surface)
  and the flush (dynamic) pressure of Watson's (1964) radial film or of a gas wall jet.

Plain-Python references are in :mod:`cnctwin.coolant.reference`.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, typeof, types

from ..geom.abi import TREE_TYPE
from ..geom.bvh import ray_first_hit_nb
from . import abi
from .abi import (BRICK_FULL, BRICK_PARTIAL, FP_STEP, FP_WALL_DECAY, FP_WATSON_C, FP_WATSON_R0, HIT_BASE, HIT_BOX,
                  HIT_CHIPS, HIT_LOST, HIT_MESH, HIT_NONE, HIT_TOOL, HIT_WORK, IMP_D, IMP_FLOW, IMP_KIND, IMP_NU,
                  IMP_RADIUS, IMP_RHO, IMP_SPEED, IMP_U, IMP_V, TP_BOX_HI, TP_BOX_LO, TP_CD_NEWTON, TP_CF_COLUMN,
                  TP_DS_MAX, TP_DS_MIN, TP_GX, TP_GY, TP_GZ, TP_MU_AIR, TP_RE_NEWTON, TP_RHO_AIR, TP_S_MAX,
                  TP_SAGITTA, TP_SN_A, TP_SN_N)

OPTS = dict(cache=True, nogil=True, error_model="numpy")
f8 = types.float64
i8 = types.int64
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
I1 = types.int64[::1]
I2 = types.int64[:, ::1]
B1 = types.int8[::1]
B2 = types.int8[:, ::1]

RAYS = typeof(abi.empty_rays())
REV = typeof(abi.empty_revolved())
BOXES = typeof(abi.empty_boxes())
BRICKS = typeof(abi.empty_bricks())
HF = typeof(abi.empty_heightfield())
ZONES = typeof(abi.Zones(np.zeros((0, 3)), np.zeros(0)))
OUT = typeof(abi.trace_out(0, 0, 1))

NO_HIT = 2.0            # fraction meaning "no hit on this segment"
N_SCRATCH = 16


# ============================================================================ geometry helpers

@njit(f8(f8, f8, f8, f8, f8), **OPTS)
def first_root(a, b, c, lo, hi):
    """Smallest root of a t^2 + b t + c = 0 in (lo, hi], or NO_HIT (stable quadratic formula)."""
    if abs(a) < 1e-300:
        if abs(b) > 1e-300:
            t = -c / b
            if lo < t <= hi:
                return t
        return NO_HIT
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return NO_HIT
    sq = math.sqrt(disc)
    q = -0.5 * (b + sq) if b >= 0.0 else -0.5 * (b - sq)
    r1 = q / a
    r2 = c / q if abs(q) > 1e-300 else r1
    small = min(r1, r2)
    large = max(r1, r2)
    if lo < small <= hi:
        return small
    if lo < large <= hi:
        return large
    return NO_HIT


@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, i8, f8, F1, I1, I1), **OPTS)
def revolved_hit(ax, ay, az, bx, by, bz, z0, dz, n, r_max, radius, part, part_out):
    """First entry of the segment a -> b (tool frame T0) into the revolved solid r <= R(z) of a radius
    table (nodes z0 + k dz, linear between nodes, nothing outside [z0, z0 + (n - 1) dz]). Returns the
    fraction (NO_HIT if none); part_out[0] = part of the interval hit."""
    if n < 2:
        return NO_HIT
    z_top = z0 + (n - 1) * dz
    dx = bx - ax
    dy = by - ay
    dzs = bz - az
    if (az < z0 and bz < z0) or (az > z_top and bz > z_top):
        return NO_HIT
    A = dx * dx + dy * dy
    B = 2.0 * (ax * dx + ay * dy)
    C = ax * ax + ay * ay
    tc = -B / (2.0 * A) if A > 0.0 else 0.0
    tc = min(max(tc, 0.0), 1.0)
    if A * tc * tc + B * tc + C > r_max * r_max:
        return NO_HIT
    if abs(dzs) < 1e-300:
        k = int(math.floor((az - z0) / dz))
        if k < 0 or k > n - 2:
            return NO_HIT
        f = (az - z0) / dz - k
        R = radius[k] + f * (radius[k + 1] - radius[k])
        if C <= R * R:
            part_out[0] = part[k]
            return 0.0
        t = first_root(A, B, C - R * R, 0.0, 1.0)
        if t < NO_HIT:
            part_out[0] = part[k]
        return t
    t_lo = (z0 - az) / dzs
    t_hi = (z_top - az) / dzs
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    t_start = max(t_lo, 0.0)
    t_end = min(t_hi, 1.0)
    if t_start > t_end:
        return NO_HIT
    z_start = az + t_start * dzs
    k = int(math.floor((z_start - z0) / dz))
    k = min(max(k, 0), n - 2)
    step = 1 if dzs > 0.0 else -1
    while 0 <= k <= n - 2:
        zk = z0 + k * dz
        ta = (zk - az) / dzs
        tb = (zk + dz - az) / dzs
        if ta > tb:
            ta, tb = tb, ta
        if ta > t_end:
            break
        if tb >= t_start:
            ta = max(ta, t_start)
            tb = min(tb, t_end)
            slope = (radius[k + 1] - radius[k]) / dz
            c0 = radius[k] + slope * (az - zk)
            c1 = slope * dzs
            fa = A * ta * ta + B * ta + C - (c0 + c1 * ta) ** 2
            if fa <= 0.0:
                part_out[0] = part[k]
                return ta
            t = first_root(A - c1 * c1, B - 2.0 * c0 * c1, C - c0 * c0, ta, tb)
            if t < NO_HIT:
                part_out[0] = part[k]
                return t
            if tb >= t_end:
                break
        k += step
    return NO_HIT


@njit(f8(F1, F1, F1, F1, f8, f8), **OPTS)
def slab_clip(o, d, lo, hi, t0, t1):
    """Entry fraction of the segment o + t d (t in [t0, t1]) into the box [lo, hi] (t0 if o is inside);
    NO_HIT if it misses."""
    tmin = t0
    tmax = t1
    for a in range(3):
        if abs(d[a]) < 1e-300:
            if o[a] < lo[a] or o[a] > hi[a]:
                return NO_HIT
        else:
            inv = 1.0 / d[a]
            ta = (lo[a] - o[a]) * inv
            tb = (hi[a] - o[a]) * inv
            if ta > tb:
                ta, tb = tb, ta
            if ta > tmin:
                tmin = ta
            if tb < tmax:
                tmax = tb
            if tmin > tmax:
                return NO_HIT
    return tmin


@njit(f8(F1, F1, F1, F1), **OPTS)
def slab_exit(o, d, lo, hi):
    """Fraction at which the segment o + t d (o inside the box) leaves it; NO_HIT if it stays inside."""
    tmax = NO_HIT
    for a in range(3):
        if abs(d[a]) < 1e-300:
            continue
        inv = 1.0 / d[a]
        ta = (lo[a] - o[a]) * inv
        tb = (hi[a] - o[a]) * inv
        t = tb if tb > ta else ta
        if t < tmax:
            tmax = t
    return tmax if tmax <= 1.0 else NO_HIT


@njit(f8(F1, F1, f8, f8, f8, f8, f8, f8, i8, B1, i8), **OPTS)
def sub_dda(o, d, lox, loy, loz, cell, t0, t1, n, occ, start):
    """DDA over the n^3 sub-cells of one brick (corner lo, sub-cell edge ``cell``) between t0 and t1;
    entry fraction of the first occupied sub-cell, NO_HIT if none."""
    px = o[0] + t0 * d[0]
    py = o[1] + t0 * d[1]
    pz = o[2] + t0 * d[2]
    ix = min(max(int(math.floor((px - lox) / cell)), 0), n - 1)
    iy = min(max(int(math.floor((py - loy) / cell)), 0), n - 1)
    iz = min(max(int(math.floor((pz - loz) / cell)), 0), n - 1)
    sx = 1 if d[0] > 0.0 else (-1 if d[0] < 0.0 else 0)
    sy = 1 if d[1] > 0.0 else (-1 if d[1] < 0.0 else 0)
    sz = 1 if d[2] > 0.0 else (-1 if d[2] < 0.0 else 0)
    tx = ((lox + (ix + (1 if sx > 0 else 0)) * cell) - o[0]) / d[0] if sx != 0 else np.inf
    ty = ((loy + (iy + (1 if sy > 0 else 0)) * cell) - o[1]) / d[1] if sy != 0 else np.inf
    tz = ((loz + (iz + (1 if sz > 0 else 0)) * cell) - o[2]) / d[2] if sz != 0 else np.inf
    dtx = cell / abs(d[0]) if sx != 0 else np.inf
    dty = cell / abs(d[1]) if sy != 0 else np.inf
    dtz = cell / abs(d[2]) if sz != 0 else np.inf
    t = t0
    for _ in range(3 * n + 3):
        if occ[start + ix + n * (iy + n * iz)] != 0:
            return t
        if tx <= ty and tx <= tz:
            t = tx
            tx += dtx
            ix += sx
        elif ty <= tz:
            t = ty
            ty += dty
            iy += sy
        else:
            t = tz
            tz += dtz
            iz += sz
        if t > t1 or ix < 0 or iy < 0 or iz < 0 or ix >= n or iy >= n or iz >= n:
            return NO_HIT
    return NO_HIT


@njit(f8(F1, F1, BRICKS, i8, F1), **OPTS)
def brick_hit(o, d, G, g, scratch):
    """First entry of the segment o + t d (t in [0, 1], S frame of grid g) into material: Amanatides-Woo
    over the bricks, nested over the sub-cells of partial bricks. ``scratch`` [>= 6]."""
    b = G.brick_m[g]
    nx = G.dims[g, 0]
    ny = G.dims[g, 1]
    nz = G.dims[g, 2]
    lo = scratch[0:3]
    hi = scratch[3:6]
    for a in range(3):
        lo[a] = G.lo_S[g, a]
    hi[0] = lo[0] + nx * b
    hi[1] = lo[1] + ny * b
    hi[2] = lo[2] + nz * b
    t0 = slab_clip(o, d, lo, hi, 0.0, 1.0)
    if t0 >= NO_HIT:
        return NO_HIT
    t1 = 1.0
    for a in range(3):
        if abs(d[a]) > 1e-300:
            ta = (lo[a] - o[a]) / d[a]
            tb = (hi[a] - o[a]) / d[a]
            tm = tb if tb > ta else ta
            if tm < t1:
                t1 = tm
    px = o[0] + t0 * d[0]
    py = o[1] + t0 * d[1]
    pz = o[2] + t0 * d[2]
    ix = min(max(int(math.floor((px - lo[0]) / b)), 0), nx - 1)
    iy = min(max(int(math.floor((py - lo[1]) / b)), 0), ny - 1)
    iz = min(max(int(math.floor((pz - lo[2]) / b)), 0), nz - 1)
    sx = 1 if d[0] > 0.0 else (-1 if d[0] < 0.0 else 0)
    sy = 1 if d[1] > 0.0 else (-1 if d[1] < 0.0 else 0)
    sz = 1 if d[2] > 0.0 else (-1 if d[2] < 0.0 else 0)
    tx = ((lo[0] + (ix + (1 if sx > 0 else 0)) * b) - o[0]) / d[0] if sx != 0 else np.inf
    ty = ((lo[1] + (iy + (1 if sy > 0 else 0)) * b) - o[1]) / d[1] if sy != 0 else np.inf
    tz = ((lo[2] + (iz + (1 if sz > 0 else 0)) * b) - o[2]) / d[2] if sz != 0 else np.inf
    dtx = b / abs(d[0]) if sx != 0 else np.inf
    dty = b / abs(d[1]) if sy != 0 else np.inf
    dtz = b / abs(d[2]) if sz != 0 else np.inf
    base = G.offset[g]
    ns = G.n_sub[g]
    t = t0
    for _ in range(nx + ny + nz + 3):
        t_exit = min(min(tx, ty), min(tz, t1))
        idx = base + ix + nx * (iy + ny * iz)
        state = G.state[idx]
        if state == BRICK_FULL:
            return t
        if state == BRICK_PARTIAL and G.sub[idx] >= 0:
            th = sub_dda(o, d, lo[0] + ix * b, lo[1] + iy * b, lo[2] + iz * b, b / ns, t, t_exit, ns, G.occ,
                         G.sub[idx])
            if th < NO_HIT:
                return th
        if tx <= ty and tx <= tz:
            t = tx
            tx += dtx
            ix += sx
        elif ty <= tz:
            t = ty
            ty += dty
            iy += sy
        else:
            t = tz
            tz += dtz
            iz += sz
        if t > t1 or ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
            return NO_HIT
    return NO_HIT


@njit(f8(F1, F1, HF, I1), **OPTS)
def height_hit(o, d, H, kind):
    """First hit of the segment o + t d (t in [0, 1], G coordinates u, v, w) with the pile height field
    (columns w <= top of every cell with a surface); kind[0] = HIT_CHIPS in the chip layer, HIT_BASE on
    or under the surface below it."""
    nu = H.base_w.shape[0]
    nv = H.base_w.shape[1]
    c = H.cell[0]
    t0 = 0.0
    t1 = 1.0
    for a in range(2):
        extent = (nu if a == 0 else nv) * c
        if abs(d[a]) < 1e-300:
            if o[a] < 0.0 or o[a] > extent:
                return NO_HIT
        else:
            ta = (0.0 - o[a]) / d[a]
            tb = (extent - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
    if t0 > t1:
        return NO_HIT
    pu = o[0] + t0 * d[0]
    pv = o[1] + t0 * d[1]
    iu = min(max(int(math.floor(pu / c)), 0), nu - 1)
    iv = min(max(int(math.floor(pv / c)), 0), nv - 1)
    su = 1 if d[0] > 0.0 else (-1 if d[0] < 0.0 else 0)
    sv = 1 if d[1] > 0.0 else (-1 if d[1] < 0.0 else 0)
    tu = (((iu + (1 if su > 0 else 0)) * c) - o[0]) / d[0] if su != 0 else np.inf
    tv = (((iv + (1 if sv > 0 else 0)) * c) - o[1]) / d[1] if sv != 0 else np.inf
    dtu = c / abs(d[0]) if su != 0 else np.inf
    dtv = c / abs(d[1]) if sv != 0 else np.inf
    t = t0
    for _ in range(nu + nv + 2):
        t_out = min(min(tu, tv), t1)
        base = H.base_w[iu, iv]
        if base > -np.inf:
            top = H.top_w[iu, iv]
            w0 = o[2] + t * d[2]
            w1 = o[2] + t_out * d[2]
            if w0 <= top:
                kind[0] = HIT_CHIPS if (w0 >= base and top > base + 1e-9) else HIT_BASE
                return t
            if w1 <= top:
                kind[0] = HIT_CHIPS if top > base + 1e-9 else HIT_BASE
                return t + (top - w0) / (w1 - w0) * (t_out - t)
        if tu <= tv:
            t = tu
            tu += dtu
            iu += su
        else:
            t = tv
            tv += dtv
            iv += sv
        if t > t1 or iu < 0 or iv < 0 or iu >= nu or iv >= nv:
            return NO_HIT
    return NO_HIT


@njit(f8(F1, F1, F1, f8, F1), **OPTS)
def sphere_entry(a, b, c, r, out):
    """Entry fraction of the segment a -> b into the sphere (c, r) (0 if a is inside, NO_HIT if none);
    out[0] = closest distance of the segment to c, out[1] = its fraction."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    fx = a[0] - c[0]
    fy = a[1] - c[1]
    fz = a[2] - c[2]
    A = dx * dx + dy * dy + dz * dz
    B = 2.0 * (fx * dx + fy * dy + fz * dz)
    C = fx * fx + fy * fy + fz * fz
    tc = -B / (2.0 * A) if A > 0.0 else 0.0
    tc = min(max(tc, 0.0), 1.0)
    out[0] = math.sqrt(max(A * tc * tc + B * tc + C, 0.0))
    out[1] = tc
    if C <= r * r:
        return 0.0
    return first_root(A, B, C - r * r, 0.0, 1.0)


@njit(types.UniTuple(f8, 3)(f8, f8, f8, F1, f8, i8, f8, f8), **OPTS)
def accel(vx, vy, vz, prm, k_column, spray, drop_d, rho_l):
    """Acceleration of a liquid jet element: gravity plus air drag on the coherent column (skin
    friction, k |v| v) or on a drop (Schiller-Naumann)."""
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if spray != 0 and drop_d > 0.0 and rho_l > 0.0:
        re = prm[TP_RHO_AIR] * speed * drop_d / prm[TP_MU_AIR]
        if re < prm[TP_RE_NEWTON]:
            cd = 24.0 / max(re, 1e-12) * (1.0 + prm[TP_SN_A] * re ** prm[TP_SN_N])
        else:
            cd = prm[TP_CD_NEWTON]
        k = 0.75 * prm[TP_RHO_AIR] * cd / (rho_l * drop_d)
    else:
        k = k_column
    return (prm[TP_GX] - k * speed * vx, prm[TP_GY] - k * speed * vy, prm[TP_GZ] - k * speed * vz)


# ============================================================================ the tracer

@njit(f8(F1, F1, REV, BOXES, BRICKS, HF, i8, I1, F1, I1), **OPTS)
def segment_hit(a, b, R, X, G, H, skip, info, scratch, iscratch):
    """First occluder hit of the segment a -> b (M): fraction (NO_HIT if none); info = [kind, index,
    part]. ``scratch`` [N_SCRATCH] float, ``iscratch`` [2] int."""
    best = NO_HIT
    d = scratch[0:3]
    for j in range(3):
        d[j] = b[j] - a[j]
    part_out = iscratch[0:1]
    for r in range(R.tip_M.shape[0]):
        if r == skip:
            continue
        o0 = R.tip_M[r, 0]
        o1 = R.tip_M[r, 1]
        o2 = R.tip_M[r, 2]
        Rm = R.R_M_T[r]
        ax = (a[0] - o0) * Rm[0, 0] + (a[1] - o1) * Rm[1, 0] + (a[2] - o2) * Rm[2, 0]
        ay = (a[0] - o0) * Rm[0, 1] + (a[1] - o1) * Rm[1, 1] + (a[2] - o2) * Rm[2, 1]
        az = (a[0] - o0) * Rm[0, 2] + (a[1] - o1) * Rm[1, 2] + (a[2] - o2) * Rm[2, 2]
        bx = (b[0] - o0) * Rm[0, 0] + (b[1] - o1) * Rm[1, 0] + (b[2] - o2) * Rm[2, 0]
        by = (b[0] - o0) * Rm[0, 1] + (b[1] - o1) * Rm[1, 1] + (b[2] - o2) * Rm[2, 1]
        bz = (b[0] - o0) * Rm[0, 2] + (b[1] - o1) * Rm[1, 2] + (b[2] - o2) * Rm[2, 2]
        t = revolved_hit(ax, ay, az, bx, by, bz, R.z0[r], R.dz[r], R.n_nodes[r], R.r_max[r], R.radius[r], R.part[r],
                         part_out)
        if t < best:
            best = t
            info[0] = HIT_TOOL
            info[1] = r
            info[2] = part_out[0]
    oB = scratch[3:6]
    dB = scratch[6:9]
    lo = scratch[9:12]
    hi = scratch[12:15]
    for k in range(X.centre_M.shape[0]):
        Rb = X.R_M_B[k]
        for j in range(3):
            oB[j] = (a[0] - X.centre_M[k, 0]) * Rb[0, j] + (a[1] - X.centre_M[k, 1]) * Rb[1, j] + \
                (a[2] - X.centre_M[k, 2]) * Rb[2, j]
            dB[j] = d[0] * Rb[0, j] + d[1] * Rb[1, j] + d[2] * Rb[2, j]
            lo[j] = -X.half[k, j]
            hi[j] = X.half[k, j]
        t = slab_clip(oB, dB, lo, hi, 0.0, 1.0)
        if t < best:
            best = t
            info[0] = HIT_BOX
            info[1] = k
            info[2] = -1
    for g in range(G.lo_S.shape[0]):
        Rs = G.R_M_S[g]
        for j in range(3):
            oB[j] = (a[0] - G.o_M_S[g, 0]) * Rs[0, j] + (a[1] - G.o_M_S[g, 1]) * Rs[1, j] + \
                (a[2] - G.o_M_S[g, 2]) * Rs[2, j]
            dB[j] = d[0] * Rs[0, j] + d[1] * Rs[1, j] + d[2] * Rs[2, j]
        t = brick_hit(oB, dB, G, g, scratch[9:15])
        if t < best:
            best = t
            info[0] = HIT_WORK
            info[1] = g
            info[2] = -1
    if H.active[0] != 0:
        Rs = H.R_M_S
        for j in range(3):
            lo[j] = (a[0] - H.o_M_S[0]) * Rs[0, j] + (a[1] - H.o_M_S[1]) * Rs[1, j] + (a[2] - H.o_M_S[2]) * Rs[2, j]
            hi[j] = d[0] * Rs[0, j] + d[1] * Rs[1, j] + d[2] * Rs[2, j]
        for j in range(3):
            oB[j] = (lo[0] - H.origin_S[0]) * H.axes_S[0, j] + (lo[1] - H.origin_S[1]) * H.axes_S[1, j] + \
                (lo[2] - H.origin_S[2]) * H.axes_S[2, j]
            dB[j] = hi[0] * H.axes_S[0, j] + hi[1] * H.axes_S[1, j] + hi[2] * H.axes_S[2, j]
        kind = iscratch[1:2]
        t = height_hit(oB, dB, H, kind)
        # the surface below the pile coincides with the workpiece or fixture top: a base hit wins
        # only when clearly earlier than a hit already found
        margin = 1e-9 if kind[0] == HIT_BASE else 0.0
        if t < best - margin:
            best = t
            info[0] = kind[0]
            info[1] = 0
            info[2] = -1
    return best


@njit(types.void(RAYS, F1, REV, BOXES, BRICKS, HF, ZONES, OUT), **OPTS)
def trace_rays(rays, prm, R, X, G, H, Z, out):
    """Trace every ray (see the module docstring); results in ``out``."""
    n = rays.origin_M.shape[0]
    nz = Z.centre_M.shape[0]
    m = out.pts_M.shape[1]
    box_lo = np.empty(3)
    box_hi = np.empty(3)
    for j in range(3):
        box_lo[j] = prm[TP_BOX_LO + j]
        box_hi[j] = prm[TP_BOX_HI + j]
    info = np.zeros(3, np.int64)
    zt = np.zeros(2)
    a = np.empty(3)
    b = np.empty(3)
    dv = np.empty(3)
    scratch = np.zeros(N_SCRATCH)
    iscratch = np.zeros(2, np.int64)
    for i in range(n):
        for j in range(3):
            a[j] = rays.origin_M[i, j]
        vx = rays.vel_M[i, 0]
        vy = rays.vel_M[i, 1]
        vz = rays.vel_M[i, 2]
        speed0 = math.sqrt(vx * vx + vy * vy + vz * vz)
        out.hit_kind[i] = HIT_NONE
        out.hit_index[i] = -1
        out.hit_part[i] = -1
        out.hit_s[i] = np.inf
        for k in range(nz):
            out.zone_s[i, k] = np.inf
            out.zone_v[i, k] = 0.0
            out.zone_spray[i, k] = 0
            out.zone_dmin[i, k] = np.inf
            out.zone_smin[i, k] = 0.0
        if speed0 <= 0.0:
            out.npts[i] = 0
            continue
        gas = rays.gas[i] != 0
        ux = vx / speed0
        uy = vy / speed0
        uz = vz / speed0
        k_col = 0.0
        if not gas and rays.column_d[i] > 0.0 and rays.rho_l[i] > 0.0:
            k_col = 2.0 * prm[TP_RHO_AIR] * prm[TP_CF_COLUMN] / (rays.rho_l[i] * rays.column_d[i])
        s = 0.0
        speed = speed0
        npt = 0
        if m > 0:
            for j in range(3):
                out.pts_M[i, 0, j] = a[j]
            out.pts_s[i, 0] = 0.0
            out.pts_v[i, 0] = speed0
            npt = 1
        done = False
        for j in range(3):
            if a[j] < box_lo[j] or a[j] > box_hi[j]:
                done = True
        if done:                    # a nozzle outside the enclosure is lost at once
            out.hit_kind[i] = HIT_LOST
            out.hit_s[i] = 0.0
            for j in range(3):
                out.hit_M[i, j] = a[j]
        s_limit = max(prm[TP_S_MAX], rays.s_aim[i])
        max_steps = int(s_limit / prm[TP_DS_MIN]) + 4
        entered = 0
        for _ in range(max_steps):
            if s >= s_limit:
                break
            spray = 1 if (not gas and s >= rays.breakup_s[i]) else 0
            if gas:
                ds = prm[TP_DS_MAX]
                b[0] = a[0] + ds * ux
                b[1] = a[1] + ds * uy
                b[2] = a[2] + ds * uz
                s_next = s + ds
                speed_next = speed0 * min(1.0, rays.core_s[i] / s_next)
                nvx = ux * speed_next
                nvy = uy * speed_next
                nvz = uz * speed_next
            else:
                a1 = accel(vx, vy, vz, prm, k_col, spray, rays.drop_d[i], rays.rho_l[i])
                sp = math.sqrt(vx * vx + vy * vy + vz * vz)
                if sp <= 1e-9:
                    break
                cx = a1[1] * vz - a1[2] * vy
                cy = a1[2] * vx - a1[0] * vz
                cz = a1[0] * vy - a1[1] * vx
                kappa = math.sqrt(cx * cx + cy * cy + cz * cz) / (sp * sp * sp)
                ds = prm[TP_DS_MAX]
                if kappa > 0.0:
                    ds = min(ds, math.sqrt(8.0 * prm[TP_SAGITTA] / kappa))
                ds = max(ds, prm[TP_DS_MIN])
                if spray == 0 and s < rays.breakup_s[i] < s + ds:
                    ds = max(rays.breakup_s[i] - s, 1e-9)
                dt = ds / sp
                vmx = vx + 0.5 * dt * a1[0]
                vmy = vy + 0.5 * dt * a1[1]
                vmz = vz + 0.5 * dt * a1[2]
                a2 = accel(vmx, vmy, vmz, prm, k_col, spray, rays.drop_d[i], rays.rho_l[i])
                b[0] = a[0] + dt * vmx
                b[1] = a[1] + dt * vmy
                b[2] = a[2] + dt * vmz
                nvx = vx + dt * a2[0]
                nvy = vy + dt * a2[1]
                nvz = vz + dt * a2[2]
                s_next = s + math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
                speed_next = math.sqrt(nvx * nvx + nvy * nvy + nvz * nvz)
            seg_len = s_next - s
            if not done:
                info[0] = HIT_NONE
                info[1] = -1
                info[2] = -1
                t = segment_hit(a, b, R, X, G, H, rays.skip_tool[i], info, scratch, iscratch)
                for j in range(3):
                    dv[j] = b[j] - a[j]
                t_exit = slab_exit(a, dv, box_lo, box_hi)
                if t_exit < t:
                    t = t_exit
                    info[0] = HIT_LOST
                    info[1] = -1
                    info[2] = -1
                if t < NO_HIT and s + t * seg_len > prm[TP_S_MAX]:
                    t = NO_HIT
                if t >= NO_HIT and s_next >= prm[TP_S_MAX]:
                    t = min(max((prm[TP_S_MAX] - s) / seg_len, 0.0), 1.0) if seg_len > 0.0 else 0.0
                    info[0] = HIT_LOST
                    info[1] = -1
                    info[2] = -1
                if t < NO_HIT:
                    done = True
                    out.hit_kind[i] = info[0]
                    out.hit_index[i] = info[1]
                    out.hit_part[i] = info[2]
                    out.hit_s[i] = s + t * seg_len
                    for j in range(3):
                        out.hit_M[i, j] = a[j] + t * (b[j] - a[j])
                    out.hit_vel_M[i, 0] = vx + t * (nvx - vx)
                    out.hit_vel_M[i, 1] = vy + t * (nvy - vy)
                    out.hit_vel_M[i, 2] = vz + t * (nvz - vz)
                    if npt < m:
                        for j in range(3):
                            out.pts_M[i, npt, j] = out.hit_M[i, j]
                        out.pts_s[i, npt] = out.hit_s[i]
                        out.pts_v[i, npt] = speed + t * (speed_next - speed)
                        npt += 1
                elif npt < m:
                    for j in range(3):
                        out.pts_M[i, npt, j] = b[j]
                    out.pts_s[i, npt] = s_next
                    out.pts_v[i, npt] = speed_next
                    npt += 1
            for k in range(nz):
                te = sphere_entry(a, b, Z.centre_M[k], Z.radius[k], zt)
                if zt[0] < out.zone_dmin[i, k]:
                    out.zone_dmin[i, k] = zt[0]
                    out.zone_smin[i, k] = s + zt[1] * seg_len
                if out.zone_s[i, k] == np.inf and te < NO_HIT:
                    out.zone_s[i, k] = s + te * seg_len
                    out.zone_v[i, k] = speed + te * (speed_next - speed)
                    out.zone_spray[i, k] = spray
                    entered += 1
            for j in range(3):
                a[j] = b[j]
            vx = nvx
            vy = nvy
            vz = nvz
            speed = speed_next
            s = s_next
            if done and (s >= rays.s_aim[i] or entered >= nz):
                break
        out.npts[i] = npt
        if not done:
            out.hit_kind[i] = HIT_LOST
            out.hit_s[i] = s
            for j in range(3):
                out.hit_M[i, j] = a[j]


@njit(types.void(TREE_TYPE, F2, F1, F3, F2, I1, F1, F2, i8, I1, I1), **OPTS)
def hit_meshes(tree, R_M_L, o_M_L, pts_M, pts_s, npts, limit_s, res_M, mesh_index, res_kind, res_index):
    """Stored polylines against one mesh (local frame L: p_L = R_M_L^T (p_M - o_M_L)) up to each ray's
    current first hit ``limit_s``; an earlier hit replaces it (limit_s, res_M, res_kind = HIT_MESH,
    res_index = mesh_index)."""
    buf = np.zeros(8)
    o = np.empty(3)
    d = np.empty(3)
    for i in range(pts_M.shape[0]):
        for k in range(npts[i] - 1):
            if pts_s[i, k] >= limit_s[i]:
                break
            for j in range(3):
                o[j] = (pts_M[i, k, 0] - o_M_L[0]) * R_M_L[0, j] + (pts_M[i, k, 1] - o_M_L[1]) * R_M_L[1, j] + \
                    (pts_M[i, k, 2] - o_M_L[2]) * R_M_L[2, j]
                e = (pts_M[i, k + 1, 0] - o_M_L[0]) * R_M_L[0, j] + (pts_M[i, k + 1, 1] - o_M_L[1]) * R_M_L[1, j] + \
                    (pts_M[i, k + 1, 2] - o_M_L[2]) * R_M_L[2, j]
                d[j] = e - o[j]
            face = ray_first_hit_nb(tree, o, d, 0.0, 1.0, buf)
            if face >= 0:
                t = buf[0]
                s_hit = pts_s[i, k] + t * (pts_s[i, k + 1] - pts_s[i, k])
                if s_hit < limit_s[i]:
                    limit_s[i] = s_hit
                    for j in range(3):
                        res_M[i, j] = pts_M[i, k, j] + t * (pts_M[i, k + 1, j] - pts_M[i, k, j])
                    res_kind[i] = HIT_MESH
                    res_index[i] = mesh_index
                break


# ============================================================================ film map

@njit(f8(f8, f8, f8, f8, f8, F1), **OPTS)
def watson_speed(r, flow, speed, d, nu, fp):
    """Free-surface velocity of the radial film of a liquid jet at radius r (Watson 1964): the jet speed
    inside r0 = c0 d Re^(1/3), u = K/(r^3 + l^3) beyond with K = 27 c^2 Q^2/(8 pi^4 nu), l from the
    continuity of u at r0."""
    if flow <= 0.0 or speed <= 0.0 or d <= 0.0 or nu <= 0.0:
        return 0.0
    re = speed * d / nu
    r0 = fp[FP_WATSON_R0] * d * re ** (1.0 / 3.0)
    if r <= r0:
        return speed
    c = fp[FP_WATSON_C]
    K = 27.0 * c * c * flow * flow / (8.0 * math.pi ** 4 * nu)
    l3 = max(K / speed - r0 ** 3, 0.0)
    return K / (r ** 3 + l3)


@njit(types.void(F2, f8, F2, F1, B2, F2, I1, I2), **OPTS)
def film_map(base_w, cell, imps, fp, wet, flush, queue, stamp):
    """Wetted cells and flush pressure on the pile grid from the impingement rows (abi.IMP_*).

    Liquid: flood fill from the impinged cell to 4-neighbours whose solid surface (``base_w``, the
    surface under the chips: the fluid percolates through the porous chip bed) is at most ``FP_STEP``
    above the current one (the film drains downhill and spreads on flat surfaces); every filled cell
    is wetted and gets 1/2 rho u_s(r)^2 of Watson's film. Gas: neighbours within |dw| <= step (the
    wall jet follows the surface) while u_m = u_imp min(1, a d/r) >= 1 % of the impact speed, no
    wetting. Mist: like gas, and wets the spray footprint radius. ``flush`` keeps the largest value;
    ``queue`` [nu nv] and ``stamp`` [nu, nv] are scratch (stamp starts at 0)."""
    nu = base_w.shape[0]
    nv = base_w.shape[1]
    step = fp[FP_STEP]
    for i in range(imps.shape[0]):
        u = imps[i, IMP_U]
        v = imps[i, IMP_V]
        iu0 = int(math.floor(u / cell))
        iv0 = int(math.floor(v / cell))
        if iu0 < 0 or iv0 < 0 or iu0 >= nu or iv0 >= nv:
            continue
        if base_w[iu0, iv0] == -np.inf:
            continue
        kind = int(imps[i, IMP_KIND])
        flow = imps[i, IMP_FLOW]
        speed = imps[i, IMP_SPEED]
        d = imps[i, IMP_D]
        rho = imps[i, IMP_RHO]
        nu_k = imps[i, IMP_NU]
        radius = imps[i, IMP_RADIUS]
        mark = i + 1
        head = 0
        tail = 0
        queue[tail] = iu0 * nv + iv0
        tail += 1
        stamp[iu0, iv0] = mark
        while head < tail:
            q = queue[head]
            head += 1
            iu = q // nv
            iv = q - iu * nv
            r = math.sqrt(((iu + 0.5) * cell - u) ** 2 + ((iv + 0.5) * cell - v) ** 2)
            if kind == 0:
                us = watson_speed(r, flow, speed, d, nu_k, fp)
                wet[iu, iv] = 1
            else:
                us = speed * min(1.0, fp[FP_WALL_DECAY] * d / r) if r > 0.0 else speed
                if kind == 2 and r <= radius:
                    wet[iu, iv] = 1
            qd = 0.5 * rho * us * us
            if qd > flush[iu, iv]:
                flush[iu, iv] = qd
            if kind != 0 and us < 0.01 * speed and r > radius:
                continue
            here = base_w[iu, iv]
            for nb in range(4):
                ju = iu + (1 if nb == 0 else (-1 if nb == 1 else 0))
                jv = iv + (1 if nb == 2 else (-1 if nb == 3 else 0))
                if ju < 0 or jv < 0 or ju >= nu or jv >= nv:
                    continue
                if stamp[ju, jv] == mark or base_w[ju, jv] == -np.inf:
                    continue
                there = base_w[ju, jv]
                if kind == 0:
                    if there > here + step:
                        continue
                elif abs(there - here) > step:
                    continue
                stamp[ju, jv] = mark
                queue[tail] = ju * nv + jv
                tail += 1


KERNELS = (first_root, revolved_hit, slab_clip, slab_exit, sub_dda, brick_hit, height_hit, sphere_entry, accel,
           segment_hit, trace_rays, hit_meshes, watson_speed, film_map)
