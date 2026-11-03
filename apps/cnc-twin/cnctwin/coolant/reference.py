"""Plain-Python references of the coolant kernels (tests compare kernels.py against these).

Geometry is checked by independent, brute-force methods where possible:

* revolved envelopes: the root of rho(t) - R(z(t)) by dense sampling and bisection (no interval walk);
* bricks: the slab entry into every occupied voxel / full brick, the smallest wins (no DDA);
* height field: the slab entry into every cell column below its top (no DDA);
* meshes: Moeller-Trumbore against every triangle (no BVH).

The jet trajectory uses the kernel's integration scheme (explicit midpoint with the same step rule),
written with plain floats, so the polylines must agree to round-off.
"""

from __future__ import annotations

import math

import numpy as np

from . import abi
from .abi import (BRICK_FULL, BRICK_PARTIAL, FP_STEP, FP_WALL_DECAY, FP_WATSON_C, FP_WATSON_R0, HIT_BASE, HIT_BOX,
                  HIT_CHIPS, HIT_LOST, HIT_NONE, HIT_TOOL, HIT_WORK, IMP_D, IMP_FLOW, IMP_KIND, IMP_NU,
                  IMP_RADIUS, IMP_RHO, IMP_SPEED, IMP_U, IMP_V, TP_BOX_HI, TP_BOX_LO, TP_CD_NEWTON, TP_CF_COLUMN,
                  TP_DS_MAX, TP_DS_MIN, TP_GX, TP_MU_AIR, TP_RE_NEWTON, TP_RHO_AIR, TP_S_MAX, TP_SAGITTA, TP_SN_A,
                  TP_SN_N)

NO_HIT = 2.0


# ============================================================================ geometry (brute force)

def revolved_hit_ref(a_T, b_T, z0, dz, radius, part, samples: int = 4000):
    """(fraction, part) of the first entry of a -> b into r <= R(z): dense sampling of
    f(t) = rho(t) - R(z(t)), then bisection on the first sign change."""
    a = np.asarray(a_T, float)
    b = np.asarray(b_T, float)
    n = len(radius)
    z_top = z0 + (n - 1) * dz

    def f(t):
        p = a + t * (b - a)
        if p[2] < z0 or p[2] > z_top:
            return 1.0
        x = (p[2] - z0) / dz
        k = min(int(math.floor(x)), n - 2)
        R = radius[k] + (x - k) * (radius[k + 1] - radius[k])
        return math.hypot(p[0], p[1]) - R

    if f(0.0) <= 0.0:
        return 0.0, _part_at(a, z0, dz, part, n)
    prev_t = 0.0
    for i in range(1, samples + 1):
        t = i / samples
        ft = f(t)
        if ft <= 0.0:
            lo, hi = prev_t, t
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            return hi, _part_at(a + hi * (b - a), z0, dz, part, n)
        prev_t = t
    return NO_HIT, -1


def _part_at(p, z0, dz, part, n):
    k = min(max(int(math.floor((p[2] - z0) / dz)), 0), n - 2)
    return int(part[k])


def box_entry(o, d, lo, hi):
    """Slab entry fraction of the segment o + t d (t in [0, 1]) into [lo, hi] (0 inside, NO_HIT none)."""
    tmin, tmax = 0.0, 1.0
    for a in range(3):
        if abs(d[a]) < 1e-300:
            if o[a] < lo[a] or o[a] > hi[a]:
                return NO_HIT
            continue
        t1 = (lo[a] - o[a]) / d[a]
        t2 = (hi[a] - o[a]) / d[a]
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return NO_HIT
    return tmin


def brick_hit_ref(o, d, bricks: abi.BrickGrid, g: int) -> float:
    """Entry into the material of grid g: every full brick and every occupied sub-cell as a box."""
    best = NO_HIT
    b = bricks.brick_m[g]
    nx, ny, nz = (int(v) for v in bricks.dims[g])
    lo0 = bricks.lo_S[g]
    ns = int(bricks.n_sub[g])
    c = b / ns
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = int(bricks.offset[g]) + i + nx * (j + ny * k)
                state = bricks.state[idx]
                corner = lo0 + np.array([i, j, k]) * b
                if state == BRICK_FULL:
                    best = min(best, box_entry(o, d, corner, corner + b))
                elif state == BRICK_PARTIAL:
                    start = int(bricks.sub[idx])
                    for kk in range(ns):
                        for jj in range(ns):
                            for ii in range(ns):
                                if bricks.occ[start + ii + ns * (jj + ns * kk)]:
                                    lo = corner + np.array([ii, jj, kk]) * c
                                    best = min(best, box_entry(o, d, lo, lo + c))
    return best


def height_hit_ref(o_G, d_G, base_w, top_w, cell):
    """(fraction, kind) of the first entry into a column [u0, u1] x [v0, v1] x (-inf, top] of a cell with a
    surface; HIT_CHIPS when the entry point lies in the chip layer above the base."""
    best, kind = NO_HIT, HIT_NONE
    nu, nv = base_w.shape
    for iu in range(nu):
        for iv in range(nv):
            base = base_w[iu, iv]
            if base == -np.inf:
                continue
            top = top_w[iu, iv]
            lo = np.array([iu * cell, iv * cell, -1e30])
            hi = np.array([(iu + 1) * cell, (iv + 1) * cell, top])
            t = box_entry(o_G, d_G, lo, hi)
            if t < best:
                w = o_G[2] + t * d_G[2]
                best = t
                kind = HIT_CHIPS if (w >= base - 1e-12 and top > base + 1e-9) else HIT_BASE
    return best, kind


def sphere_entry_ref(a, b, c, r):
    """(entry fraction or NO_HIT, closest distance, its fraction) by the quadratic and the projection."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    d = b - a
    f = a - c
    A = float(d @ d)
    tc = min(max(-float(f @ d) / A, 0.0), 1.0) if A > 0 else 0.0
    dist = float(np.linalg.norm(f + tc * d))
    if float(f @ f) <= r * r:
        return 0.0, dist, tc
    B = 2.0 * float(f @ d)
    C = float(f @ f) - r * r
    disc = B * B - 4 * A * C
    if A <= 0 or disc < 0:
        return NO_HIT, dist, tc
    for t in sorted(((-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A))):
        if 0.0 < t <= 1.0:
            return t, dist, tc
    return NO_HIT, dist, tc


def triangle_hit(o, d, v0, v1, v2):
    """Moeller-Trumbore: fraction t in [0, 1] of the segment o + t d through the triangle, or NO_HIT."""
    e1 = v1 - v0
    e2 = v2 - v0
    p = np.cross(d, e2)
    det = float(e1 @ p)
    if abs(det) < 1e-300:
        return NO_HIT
    inv = 1.0 / det
    s = o - v0
    u = float(s @ p) * inv
    if u < 0.0 or u > 1.0:
        return NO_HIT
    q = np.cross(s, e1)
    v = float(d @ q) * inv
    if v < 0.0 or u + v > 1.0:
        return NO_HIT
    t = float(e2 @ q) * inv
    return t if 0.0 <= t <= 1.0 else NO_HIT


# ============================================================================ tracer

def _accel_ref(v, prm, k_column, spray, drop_d, rho_l):
    speed = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if spray and drop_d > 0.0 and rho_l > 0.0:
        re = prm[TP_RHO_AIR] * speed * drop_d / prm[TP_MU_AIR]
        cd = 24.0 / max(re, 1e-12) * (1.0 + prm[TP_SN_A] * re ** prm[TP_SN_N]) if re < prm[TP_RE_NEWTON] \
            else prm[TP_CD_NEWTON]
        k = 0.75 * prm[TP_RHO_AIR] * cd / (rho_l * drop_d)
    else:
        k = k_column
    return [prm[TP_GX + j] - k * speed * v[j] for j in range(3)]


def segment_hit_ref(a, b, R, X, G, H, skip):
    """(fraction, kind, index, part) of the first occluder hit of a -> b (brute-force geometry)."""
    best = (NO_HIT, HIT_NONE, -1, -1)
    d = b - a
    for r in range(R.tip_M.shape[0]):
        if r == skip:
            continue
        aT = R.R_M_T[r].T @ (a - R.tip_M[r])
        bT = R.R_M_T[r].T @ (b - R.tip_M[r])
        n = int(R.n_nodes[r])
        t, part = revolved_hit_ref(aT, bT, R.z0[r], R.dz[r], R.radius[r, :n], R.part[r, :n])
        if t < best[0]:
            best = (t, HIT_TOOL, r, part)
    for k in range(X.centre_M.shape[0]):
        oB = X.R_M_B[k].T @ (a - X.centre_M[k])
        dB = X.R_M_B[k].T @ d
        t = box_entry(oB, dB, -X.half[k], X.half[k])
        if t < best[0]:
            best = (t, HIT_BOX, k, -1)
    for g in range(G.lo_S.shape[0]):
        oS = G.R_M_S[g].T @ (a - G.o_M_S[g])
        dS = G.R_M_S[g].T @ d
        t = brick_hit_ref(oS, dS, G, g)
        if t < best[0]:
            best = (t, HIT_WORK, g, -1)
    if H.active[0]:
        oS = H.R_M_S.T @ (a - H.o_M_S)
        dS = H.R_M_S.T @ d
        oG = H.axes_S.T @ (oS - H.origin_S)
        dG = H.axes_S.T @ dS
        t, kind = height_hit_ref(oG, dG, H.base_w, H.top_w, float(H.cell[0]))
        margin = 1e-9 if kind == HIT_BASE else 0.0
        if t < best[0] - margin:
            best = (t, kind, 0, -1)
    return best


def trace_rays_ref(rays: abi.RaySet, prm, R, X, G, H, Z, max_pts: int) -> abi.TraceOut:
    """Reference of kernels.trace_rays (same trajectory rule, brute-force occluders)."""
    n = len(rays.weight)
    nz = Z.centre_M.shape[0]
    out = abi.trace_out(n, nz, max_pts)
    box_lo = np.asarray(prm[TP_BOX_LO:TP_BOX_LO + 3])
    box_hi = np.asarray(prm[TP_BOX_HI:TP_BOX_HI + 3])
    for i in range(n):
        a = np.array(rays.origin_M[i], float)
        v = [float(x) for x in rays.vel_M[i]]
        speed0 = math.sqrt(sum(x * x for x in v))
        if speed0 <= 0.0:
            continue
        gas = bool(rays.gas[i])
        u = [x / speed0 for x in v]
        k_col = 2.0 * prm[TP_RHO_AIR] * prm[TP_CF_COLUMN] / (rays.rho_l[i] * rays.column_d[i]) \
            if (not gas and rays.column_d[i] > 0 and rays.rho_l[i] > 0) else 0.0
        s, speed, npt = 0.0, speed0, 0
        if max_pts > 0:
            out.pts_M[i, 0] = a
            out.pts_v[i, 0] = speed0
            npt = 1
        done = bool(np.any(a < box_lo) or np.any(a > box_hi))
        if done:
            out.hit_kind[i] = HIT_LOST
            out.hit_s[i] = 0.0
            out.hit_M[i] = a
        s_max = prm[TP_S_MAX]
        s_limit = max(s_max, rays.s_aim[i])
        entered = 0
        for _ in range(int(s_limit / prm[TP_DS_MIN]) + 4):
            if s >= s_limit:
                break
            spray = (not gas) and s >= rays.breakup_s[i]
            if gas:
                ds = prm[TP_DS_MAX]
                b = a + ds * np.asarray(u)
                s_next = s + ds
                speed_next = speed0 * min(1.0, rays.core_s[i] / s_next)
                nv = [x * speed_next for x in u]
            else:
                a1 = _accel_ref(v, prm, k_col, spray, rays.drop_d[i], rays.rho_l[i])
                sp = math.sqrt(sum(x * x for x in v))
                if sp <= 1e-9:
                    break
                cr = np.cross(a1, v)
                kappa = float(np.linalg.norm(cr)) / sp ** 3
                ds = prm[TP_DS_MAX]
                if kappa > 0:
                    ds = min(ds, math.sqrt(8.0 * prm[TP_SAGITTA] / kappa))
                ds = max(ds, prm[TP_DS_MIN])
                if not spray and s < rays.breakup_s[i] < s + ds:
                    ds = max(rays.breakup_s[i] - s, 1e-9)
                dt = ds / sp
                vm = [v[j] + 0.5 * dt * a1[j] for j in range(3)]
                a2 = _accel_ref(vm, prm, k_col, spray, rays.drop_d[i], rays.rho_l[i])
                b = a + dt * np.asarray(vm)
                nv = [v[j] + dt * a2[j] for j in range(3)]
                s_next = s + float(np.linalg.norm(b - a))
                speed_next = math.sqrt(sum(x * x for x in nv))
            seg = s_next - s
            if not done:
                t, kind, idx, part = segment_hit_ref(a, b, R, X, G, H, int(rays.skip_tool[i]))
                t_exit = _exit_ref(a, b - a, box_lo, box_hi)
                if t_exit < t:
                    t, kind, idx, part = t_exit, HIT_LOST, -1, -1
                if t < NO_HIT and s + t * seg > s_max:
                    t = NO_HIT
                if t >= NO_HIT and s_next >= s_max:
                    t = min(max((s_max - s) / seg, 0.0), 1.0) if seg > 0 else 0.0
                    kind, idx, part = HIT_LOST, -1, -1
                if t < NO_HIT:
                    done = True
                    out.hit_kind[i], out.hit_index[i], out.hit_part[i] = kind, idx, part
                    out.hit_s[i] = s + t * seg
                    out.hit_M[i] = a + t * (b - a)
                    out.hit_vel_M[i] = [v[j] + t * (nv[j] - v[j]) for j in range(3)]
                    if npt < max_pts:
                        out.pts_M[i, npt] = out.hit_M[i]
                        out.pts_s[i, npt] = out.hit_s[i]
                        out.pts_v[i, npt] = speed + t * (speed_next - speed)
                        npt += 1
                elif npt < max_pts:
                    out.pts_M[i, npt] = b
                    out.pts_s[i, npt] = s_next
                    out.pts_v[i, npt] = speed_next
                    npt += 1
            for k in range(nz):
                te, dist, tc = sphere_entry_ref(a, b, Z.centre_M[k], Z.radius[k])
                if dist < out.zone_dmin[i, k]:
                    out.zone_dmin[i, k] = dist
                    out.zone_smin[i, k] = s + tc * seg
                if out.zone_s[i, k] == np.inf and te < NO_HIT:
                    out.zone_s[i, k] = s + te * seg
                    out.zone_v[i, k] = speed + te * (speed_next - speed)
                    out.zone_spray[i, k] = 1 if spray else 0
                    entered += 1
            a, v, speed, s = b, nv, speed_next, s_next
            if done and (s >= rays.s_aim[i] or entered >= nz):
                break
        out.npts[i] = npt
        if not done:
            out.hit_kind[i] = HIT_LOST
            out.hit_s[i] = s
            out.hit_M[i] = a
    return out


def _exit_ref(o, d, lo, hi):
    tmax = NO_HIT
    for a in range(3):
        if abs(d[a]) < 1e-300:
            continue
        t = max((lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a])
        tmax = min(tmax, t)
    return tmax if tmax <= 1.0 else NO_HIT


def hit_meshes_ref(vertices_L, faces, R_M_L, o_M_L, pts_M, pts_s, npts, limit_s):
    """(new limit_s, hit point, hit flag) of the polylines against a triangle list (local frame)."""
    limit = np.array(limit_s, float)
    hit_M = np.zeros((len(limit), 3))
    flag = np.zeros(len(limit), bool)
    for i in range(len(limit)):
        for k in range(int(npts[i]) - 1):
            if pts_s[i, k] >= limit[i]:
                break
            o = R_M_L.T @ (pts_M[i, k] - o_M_L)
            e = R_M_L.T @ (pts_M[i, k + 1] - o_M_L)
            d = e - o
            best = NO_HIT
            for f in faces:
                best = min(best, triangle_hit(o, d, vertices_L[f[0]], vertices_L[f[1]], vertices_L[f[2]]))
            if best < NO_HIT:
                s_hit = pts_s[i, k] + best * (pts_s[i, k + 1] - pts_s[i, k])
                if s_hit < limit[i]:
                    limit[i] = s_hit
                    hit_M[i] = pts_M[i, k] + best * (pts_M[i, k + 1] - pts_M[i, k])
                    flag[i] = True
                break
    return limit, hit_M, flag


# ============================================================================ film

def watson_speed_ref(r, flow, speed, d, nu, fp):
    if flow <= 0 or speed <= 0 or d <= 0 or nu <= 0:
        return 0.0
    re = speed * d / nu
    r0 = fp[FP_WATSON_R0] * d * re ** (1.0 / 3.0)
    if r <= r0:
        return speed
    K = 27.0 * fp[FP_WATSON_C] ** 2 * flow ** 2 / (8.0 * math.pi ** 4 * nu)
    l3 = max(K / speed - r0 ** 3, 0.0)
    return K / (r ** 3 + l3)


def film_map_ref(base_w, cell, imps, fp):
    """(wet, flush) by breadth-first search with Python sets and lists (same rules as kernels.film_map)."""
    from collections import deque
    nu, nv = base_w.shape
    wet = np.zeros((nu, nv), np.int8)
    flush = np.zeros((nu, nv))
    step = fp[FP_STEP]
    for row in imps:
        u, v = row[IMP_U], row[IMP_V]
        iu0, iv0 = int(math.floor(u / cell)), int(math.floor(v / cell))
        if not (0 <= iu0 < nu and 0 <= iv0 < nv) or base_w[iu0, iv0] == -np.inf:
            continue
        kind = int(row[IMP_KIND])
        seen = {(iu0, iv0)}
        queue = deque([(iu0, iv0)])
        while queue:
            iu, iv = queue.popleft()
            r = math.hypot((iu + 0.5) * cell - u, (iv + 0.5) * cell - v)
            if kind == 0:
                us = watson_speed_ref(r, row[IMP_FLOW], row[IMP_SPEED], row[IMP_D], row[IMP_NU], fp)
                wet[iu, iv] = 1
            else:
                us = row[IMP_SPEED] * min(1.0, fp[FP_WALL_DECAY] * row[IMP_D] / r) if r > 0 else row[IMP_SPEED]
                if kind == 2 and r <= row[IMP_RADIUS]:
                    wet[iu, iv] = 1
            flush[iu, iv] = max(flush[iu, iv], 0.5 * row[IMP_RHO] * us * us)
            if kind != 0 and us < 0.01 * row[IMP_SPEED] and r > row[IMP_RADIUS]:
                continue
            here = base_w[iu, iv]
            for ju, jv in ((iu + 1, iv), (iu - 1, iv), (iu, iv + 1), (iu, iv - 1)):
                if not (0 <= ju < nu and 0 <= jv < nv) or (ju, jv) in seen or base_w[ju, jv] == -np.inf:
                    continue
                there = base_w[ju, jv]
                if (kind == 0 and there > here + step) or (kind != 0 and abs(there - here) > step):
                    continue
                seen.add((ju, jv))
                queue.append((ju, jv))
    return wet, flush
