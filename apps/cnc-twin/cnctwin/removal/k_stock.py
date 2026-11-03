"""Stock and target construction kernels (docs/models/removal.md "Stock and target").

* :func:`add_interval` - union of one interval into one ray (brick bookkeeping included).
* :func:`fill_box`, :func:`fill_tube` - analytic block and cylinder / tube stocks: exact endpoints.
* :func:`rasterize` / :func:`resolve_hits` - a closed triangle mesh on the axis-aligned rays of one
  family: each ray is a lattice line, so a hit is a 2-D point-in-triangle test in the transverse plane.
  The edge function of every edge is evaluated in a canonical vertex order (the lexicographically smaller
  end first) so the two triangles of a shared edge see bit-identical values of opposite sign, and a point
  exactly on an edge or vertex belongs to one triangle only by the top-left rule (owned edges: dv > 0, or
  dv = 0 and du < 0, on the counter-clockwise triangle) - the rasterisation rule of Pineda (1988) used by
  Woop, Benthin & Wald (2013) for watertight ray casting. Hits are sorted along the ray and a winding
  count (entering +1 where the outward normal opposes the ray, leaving -1) gives the material intervals
  (winding > 0); a ray whose winding does not return to zero is counted (open mesh).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from .k_ray import brick_account, iv_union, ray_index, ray_load, ray_overlap, ray_store
from .nbt import B1, F1, F2, FIELD_T, I1, I4_1, I4_2, OPTS, SCRATCH_T, f8, i8, void

Q = A.QUANTUM_M


@njit(i8(FIELD_T, i8, i8, i8, i8, i8, i8, i8, SCRATCH_T), **OPTS)
def add_interval(f, lvl, fam, iu, iv, q0, q1, ev, sc):
    """Unite [q0, q1) (quanta, clipped to the ray) into ray (lvl, fam, iu, iv); returns the added length,
    -1 on pool overflow, -2 on scratch overflow."""
    lq = (f.lv_dims[lvl, fam] - 1) * f.lv_cell_q[lvl]
    if q0 < 0:
        q0 = 0
    if q1 > lq:
        q1 = lq
    if q1 <= q0:
        return 0
    ray = ray_index(f, lvl, fam, iu, iv)
    sc.rlo[0] = q0
    sc.rhi[0] = q1
    n = ray_load(f, ray, sc.lo, sc.hi, sc.elo, sc.ehi)
    if n < 0:
        f.info[A.FI_SCRATCH_OVF] = 1
        return -2
    nn, npc, added = iv_union(n, sc.lo, sc.hi, sc.elo, sc.ehi, 1, sc.rlo, sc.rhi, ev, sc.nlo, sc.nhi, sc.nelo,
                              sc.nehi, sc.plo, sc.phi)
    if nn < 0:
        f.info[A.FI_SCRATCH_OVF] = 1
        return -2
    if added == 0:
        return 0
    if ray_store(f, ray, sc.nlo, sc.nhi, sc.nelo, sc.nehi, nn) < 0:
        return -1
    brick_account(f, lvl, fam, iu, iv, sc.plo, sc.phi, npc, 1)
    f.info[A.FI_VERSION] += 1
    return added


@njit(i8(FIELD_T, i8, i8, i8, i8, I1, I1, i8, i8, SCRATCH_T), **OPTS)
def replace_ray(f, lvl, fam, iu, iv, qlo, qhi, m, ev, sc):
    """Replace the segments of one ray by the sorted, disjoint intervals [qlo, qhi) (quanta, clipped to the
    ray); endpoints get ``ev``. Brick material is re-booked. Returns 0, -1 on pool overflow (unchanged)."""
    lq = (f.lv_dims[lvl, fam] - 1) * f.lv_cell_q[lvl]
    ray = ray_index(f, lvl, fam, iu, iv)
    n = ray_load(f, ray, sc.lo, sc.hi, sc.elo, sc.ehi)
    if n < 0:
        f.info[A.FI_SCRATCH_OVF] = 1
        return -2
    k = 0
    for i in range(m):
        a = qlo[i] if qlo[i] > 0 else 0
        b = qhi[i] if qhi[i] < lq else lq
        if b <= a:
            continue
        if k > 0 and a <= sc.nhi[k - 1]:
            if b > sc.nhi[k - 1]:
                sc.nhi[k - 1] = b
            continue
        if k >= sc.nlo.shape[0]:
            f.info[A.FI_SCRATCH_OVF] = 1
            return -2
        sc.nlo[k] = a
        sc.nhi[k] = b
        sc.nelo[k] = ev
        sc.nehi[k] = ev
        k += 1
    if ray_store(f, ray, sc.nlo, sc.nhi, sc.nelo, sc.nehi, k) < 0:
        return -1
    for i in range(n):
        sc.plo[i] = sc.lo[i]
        sc.phi[i] = sc.hi[i]
    brick_account(f, lvl, fam, iu, iv, sc.plo, sc.phi, n, -1)
    for i in range(k):
        sc.plo[i] = sc.nlo[i]
        sc.phi[i] = sc.nhi[i]
    brick_account(f, lvl, fam, iu, iv, sc.plo, sc.phi, k, 1)
    f.info[A.FI_VERSION] += 1
    return 0


@njit(i8(FIELD_T, i8, F1, F1, i8, SCRATCH_T), **OPTS)
def fill_box(f, lvl, lo, hi, ev, sc):
    """Unite the axis-aligned box [lo, hi) (S) into every ray of a level. Returns -1 on overflow."""
    a = f.lv_cell_m[lvl]
    for fam in range(3):
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        o = f.lv_origin_m[lvl]
        q0 = np.int64(np.floor((lo[fam] - o[fam]) / Q + 0.5))
        q1 = np.int64(np.floor((hi[fam] - o[fam]) / Q + 0.5))
        if q1 <= q0:
            continue
        for iu in range(f.lv_dims[lvl, u]):
            xu = o[u] + iu * a
            if xu < lo[u] or xu >= hi[u]:
                continue
            for iv in range(f.lv_dims[lvl, v]):
                xv = o[v] + iv * a
                if xv < lo[v] or xv >= hi[v]:
                    continue
                if add_interval(f, lvl, fam, iu, iv, q0, q1, ev, sc) == -1:
                    return -1
    return 0


@njit(i8(FIELD_T, i8, f8, f8, f8, f8, f8, f8, i8, SCRATCH_T), **OPTS)
def fill_tube(f, lvl, cx, cy, r_out, r_in, z0, z1, ev, sc):
    """Unite the tube r_in <= r < r_out, z0 <= z < z1 about the axis through (cx, cy) along S z (r_in = 0:
    a cylinder) into every ray of a level. Exact endpoints (square roots). Returns -1 on overflow."""
    a = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    # z family: r constant along the ray
    qz0 = np.int64(np.floor((z0 - o[2]) / Q + 0.5))
    qz1 = np.int64(np.floor((z1 - o[2]) / Q + 0.5))
    for i in range(f.lv_dims[lvl, 0]):
        x = o[0] + i * a - cx
        for j in range(f.lv_dims[lvl, 1]):
            y = o[1] + j * a - cy
            r2 = x * x + y * y
            if r2 < r_out * r_out and r2 >= r_in * r_in:
                if add_interval(f, lvl, 2, i, j, qz0, qz1, ev, sc) == -1:
                    return -1
    # x and y families: z constant, chord of the annulus
    for fam in range(2):
        u = 1 if fam == 0 else 0
        nu = f.lv_dims[lvl, u]
        nz = f.lv_dims[lvl, 2]
        c_along = cx if fam == 0 else cy
        c_across = cy if fam == 0 else cx
        for iu in range(nu):
            c = o[u] + iu * a - c_across
            if abs(c) >= r_out:
                continue
            ho = math.sqrt(r_out * r_out - c * c)
            hi_ = 0.0
            two = False
            if abs(c) < r_in:
                hi_ = math.sqrt(r_in * r_in - c * c)
                two = True
            for k in range(nz):
                z = o[2] + k * a
                if z < z0 or z >= z1:
                    continue
                base = c_along - o[fam]
                if two:
                    a0 = np.int64(np.floor((base - ho) / Q + 0.5))
                    a1 = np.int64(np.floor((base - hi_) / Q + 0.5))
                    b0 = np.int64(np.floor((base + hi_) / Q + 0.5))
                    b1 = np.int64(np.floor((base + ho) / Q + 0.5))
                    if add_interval(f, lvl, fam, iu, k, a0, a1, ev, sc) == -1:
                        return -1
                    if add_interval(f, lvl, fam, iu, k, b0, b1, ev, sc) == -1:
                        return -1
                else:
                    a0 = np.int64(np.floor((base - ho) / Q + 0.5))
                    b1 = np.int64(np.floor((base + ho) / Q + 0.5))
                    if add_interval(f, lvl, fam, iu, k, a0, b1, ev, sc) == -1:
                        return -1
    return 0


# --------------------------------------------------------------------------- mesh rasterisation
@njit(f8(f8, f8, f8, f8, f8, f8), **OPTS)
def _canon_edge(pu, pv, qu, qv, xu, xv):
    """Edge function of the directed edge p -> q at x, evaluated in canonical vertex order."""
    if pu < qu or (pu == qu and pv < qv):
        return (qu - pu) * (xv - pv) - (qv - pv) * (xu - pu)
    return -((pu - qu) * (xv - qv) - (pv - qv) * (xu - qu))


@njit(types.boolean(f8, f8, f8), **OPTS)
def _covers(e, du, dv):
    if e > 0.0:
        return True
    if e < 0.0:
        return False
    return dv > 0.0 or (dv == 0.0 and du < 0.0)


@njit(i8(FIELD_T, i8, i8, F2, I4_2, i8, i8, i8, i8, I1, I1, F1, B1, i8), **OPTS)
def rasterize(f, lvl, fam, V, Fc, iu0, iu1, iv0, iv1, counts, offsets, out_t, out_dir, fill):
    """Hits of the triangles (V, Fc) on the rays (iu0..iu1, iv0..iv1) of family ``fam``. fill = 0: count
    into ``counts`` (window-local ray index (iu - iu0) * (iv1 - iv0 + 1) + iv - iv0); fill = 1: write the
    axial coordinate (quanta from the level origin) and direction (+1 entering, -1 leaving) at
    ``offsets`` (+ running position kept in ``counts``, which must be zero). Returns the hit total."""
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    sign = -1.0 if fam == 1 else 1.0      # n_f = sign * (2-D signed area in (u, v))
    a = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    nv_w = iv1 - iv0 + 1
    total = 0
    for t in range(Fc.shape[0]):
        i0 = Fc[t, 0]
        i1 = Fc[t, 1]
        i2 = Fc[t, 2]
        au = V[i0, u]
        av = V[i0, v]
        bu = V[i1, u]
        bv = V[i1, v]
        cu = V[i2, u]
        cv = V[i2, v]
        af = V[i0, fam]
        bf = V[i1, fam]
        cf = V[i2, fam]
        area = _canon_edge(au, av, bu, bv, cu, cv)
        if area == 0.0:
            continue
        nf = sign * area
        direction = -1 if nf > 0.0 else 1
        if area < 0.0:
            # make counter-clockwise: swap b and c
            au, av, bu, bv, cu, cv = au, av, cu, cv, bu, bv
            af, bf, cf = af, cf, bf
            area = -area
        umin = min(au, min(bu, cu))
        umax = max(au, max(bu, cu))
        vmin = min(av, min(bv, cv))
        vmax = max(av, max(bv, cv))
        ju0 = int(math.ceil((umin - o[u]) / a))
        ju1 = int(math.floor((umax - o[u]) / a))
        jv0 = int(math.ceil((vmin - o[v]) / a))
        jv1 = int(math.floor((vmax - o[v]) / a))
        if ju0 < iu0:
            ju0 = iu0
        if ju1 > iu1:
            ju1 = iu1
        if jv0 < iv0:
            jv0 = iv0
        if jv1 > iv1:
            jv1 = iv1
        for iu in range(ju0, ju1 + 1):
            xu = o[u] + iu * a
            for iv in range(jv0, jv1 + 1):
                xv = o[v] + iv * a
                e0 = _canon_edge(bu, bv, cu, cv, xu, xv)     # opposite a
                if not _covers(e0, cu - bu, cv - bv):
                    continue
                e1 = _canon_edge(cu, cv, au, av, xu, xv)     # opposite b
                if not _covers(e1, au - cu, av - cv):
                    continue
                e2 = _canon_edge(au, av, bu, bv, xu, xv)     # opposite c
                if not _covers(e2, bu - au, bv - av):
                    continue
                w = (iu - iu0) * nv_w + (iv - iv0)
                if fill == 1:
                    pos = offsets[w] + counts[w]
                    tf = (e0 * af + e1 * bf + e2 * cf) / area
                    out_t[pos] = (tf - o[fam]) / Q
                    out_dir[pos] = direction
                counts[w] += 1
                total += 1
    return total


@njit(i8(F1, B1, i8, i8, I1, I1), **OPTS)
def _ray_intervals(ts, ds, a, b, out_lo, out_hi):
    """Sort hits a..b-1 (entering before leaving at equal t) and sweep the winding number; material where
    it is positive. Writes quantised intervals; returns their number, or -(count + 1) when the winding does
    not return to zero (open mesh along this ray)."""
    n = b - a
    # insertion sort of (t, -dir) in place
    for i in range(a + 1, b):
        tk = ts[i]
        dk = ds[i]
        j = i - 1
        while j >= a and (ts[j] > tk or (ts[j] == tk and ds[j] < dk)):
            ts[j + 1] = ts[j]
            ds[j + 1] = ds[j]
            j -= 1
        ts[j + 1] = tk
        ds[j + 1] = dk
    w = 0
    m = 0
    start = 0.0
    for i in range(a, b):
        if ds[i] > 0:
            w += 1
            if w == 1:
                start = ts[i]
        else:
            w -= 1
            if w == 0:
                q0 = np.int64(np.floor(start + 0.5))
                q1 = np.int64(np.floor(ts[i] + 0.5))
                if q1 > q0:
                    if m > 0 and q0 <= out_hi[m - 1]:
                        if q1 > out_hi[m - 1]:
                            out_hi[m - 1] = q1
                    else:
                        out_lo[m] = q0
                        out_hi[m] = q1
                        m += 1
    if w != 0 or n % 2 != 0:
        return -(m + 1)
    return m


@njit(i8(FIELD_T, i8, i8, i8, i8, i8, i8, I1, F1, B1, i8, i8, I4_2, I1, I1, F1, SCRATCH_T), **OPTS)
def resolve_hits(f, lvl, fam, iu0, iu1, iv0, iv1, offsets, ts, ds, mode, ev, seg_out, nseg_out, stats_i, stats_f, sc):
    """Turn the hits of a window into material intervals per ray.

    mode 0: unite them into the field (stock from a mesh); mode 1: write them to ``seg_out`` at the ray's
    hit offset and their number to ``nseg_out`` (target field); mode 2: overlap with the field's material
    (collision: stats_f = [overlap length (m) summed, largest overlap (m), x, y, z of its middle]).
    stats_i = [open rays, rays with intervals, pool overflow]. Returns 0 (-1: pool overflow in mode 0)."""
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    nv_w = iv1 - iv0 + 1
    a_m = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    cap = 4096
    lo = np.empty(cap, np.int64)
    hi = np.empty(cap, np.int64)
    for iu in range(iu0, iu1 + 1):
        for iv in range(iv0, iv1 + 1):
            w = (iu - iu0) * nv_w + (iv - iv0)
            a = offsets[w]
            b = offsets[w + 1]
            if b <= a:
                if mode == 1:
                    nseg_out[w] = 0
                continue
            if (b - a) > cap:
                stats_i[0] += 1
                continue
            m = _ray_intervals(ts, ds, a, b, lo, hi)
            if m < 0:
                stats_i[0] += 1
                m = -m - 1
            if m > 0:
                stats_i[1] += 1
            if mode == 0:
                for k in range(m):
                    if add_interval(f, lvl, fam, iu, iv, lo[k], hi[k], ev, sc) == -1:
                        stats_i[2] = 1
                        return -1
            elif mode == 1:
                for k in range(m):
                    seg_out[a + k, 0] = lo[k]
                    seg_out[a + k, 1] = hi[k]
                nseg_out[w] = m
            else:
                ray = ray_index(f, lvl, fam, iu, iv)
                for k in range(m):
                    length, touched = ray_overlap(f, ray, lo[k], hi[k])
                    if length > 0:
                        lm = length * Q
                        stats_f[0] += lm
                        if lm > stats_f[1]:
                            stats_f[1] = lm
                            mid = 0.5 * (lo[k] + hi[k]) * Q + o[fam]
                            p = np.zeros(3)
                            p[fam] = mid
                            p[u] = o[u] + iu * a_m
                            p[v] = o[v] + iv * a_m
                            stats_f[2] = p[0]
                            stats_f[3] = p[1]
                            stats_f[4] = p[2]
    return 0


KERNELS = (add_interval, replace_ray, fill_box, fill_tube, _canon_edge, _covers, rasterize, _ray_intervals,
           resolve_hits)
