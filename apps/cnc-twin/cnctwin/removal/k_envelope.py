"""Ray / revolved-envelope interval mathematics (docs/models/removal.md "Subtraction").

Geometry of one ray against one slab of the envelope: the ray ``x(tau) = p0 + tau d`` (d a lattice axis,
tau measured from the ray point nearest the slab's axis point, so every coefficient is of the size of the
tool) has ``z(tau) = z0 + s tau`` along the tool axis (s = d . w) and the squared distance from the axis
``r^2(tau) = A0 + A1 tau + A2 tau^2``. Per sub-piece of the profile:

* line (cylinder, cone): ``r^2 <= (R0 + R1 tau)^2`` with ``R0 + R1 tau >= 0`` - a quadratic, solved in
  the cancellation-free form (Press et al. 2007, 5.6); exact.
* arc with centre on the axis (sphere: ball ends, rc + dr = 0): ``r^2 + (z - zc)^2 <= rho^2`` - a
  quadratic; exact.
* arc off the axis (torus: corner radii, balls with a tooth radius offset): the boundary
  ``r = rc' + sigma sqrt(rho^2 - (z - zc)^2)`` is a root of the quartic
  ``(r^2 + (z - zc)^2 + rc'^2 - rho^2)^2 - 4 rc'^2 r^2`` (the implicit torus); its real roots on the
  interval are isolated through the derivative cascade (P'' quadratic, P' and P monotone between the roots
  of the next derivative) and refined by safeguarded Newton to 1e-11 m; the inside test of each
  sub-interval evaluates the explicit form at its midpoint. Exact to the root tolerance (< 1e-9 m).
* exactly parallel (d = +-w) or perpendicular (d . w = 0) rays use closed forms (3-axis fast paths).

The small helpers are inlined into the subtraction and collision kernels (one call site each).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from . import k_core as C
from .nbt import (ACCUM_T, EVENTS_T, F1, F2, FIELD_T, I1, I4_1, INLINE, OPTS, PROFILE_T, SCRATCH_T, SPEC_T,
                  TARGET_T, f8, i8, void)

Q = A.QUANTUM_M
ROOT_TOL_M = 1.0e-11
MERGE_TOL_M = 1.0e-12
# slab bounds a subtraction leaves for _env_ray in work[16:16 + A.N_ENV] (scratch fw[FW_ENV:]; set per call)
ENV_KEY = 0                 # ENV_MAGIC when valid (the root isolation never writes work[16:])
ENV_NS = 1                  # slab count, slab 0's axis point and the slab heights' ends the bounds belong to
ENV_CX = 2
ENV_CY = 3
ENV_CZ = 4
ENV_SPREAD = 5              # largest |c_k - c_0| |w| of the slab axis points
ENV_Z0 = 6
ENV_Z1 = 7
ENV_HINV = 8                # slabs per metre of the slab heights' range (the window search's first guess)
ENV_TLO = 9                 # the ray's material in the call's range, [t_lo_m, t_hi_m] (metres from the ray origin):
ENV_THI = 10                #   intervals that cannot reach it change nothing, their slabs need not be evaluated
ENV_MAGIC = -7.25e299
ENV_EPS = 1.0e-10           # m: slab culling margin (the rounding of the heights is ~1e-18 m)
AABB_EPS = 1.0e-12          # m: margin of the envelope box (the rounding of its end points is ~1e-17 m)


# --------------------------------------------------------------------------- 1-D inequalities
@njit(i8(f8, f8, f8, f8, f8, F2, I4_1, i8, i8), **INLINE)
def quad_leq(c0, c1, c2, ta, tb, out, tags, n, tag):
    """Append the parts of [ta, tb] where c0 + c1 t + c2 t^2 <= 0; returns the new count (-1: full)."""
    if n < 0 or tb <= ta:
        return n
    cap = out.shape[0]
    if c2 == 0.0:
        if c1 == 0.0:
            if c0 <= 0.0:
                if n >= cap:
                    return -1
                out[n, 0] = ta
                out[n, 1] = tb
                tags[n] = tag
                return n + 1
            return n
        r = -c0 / c1
        a = ta
        b = tb
        if c1 > 0.0:
            if r < b:
                b = r
        else:
            if r > a:
                a = r
        if b > a:
            if n >= cap:
                return -1
            out[n, 0] = a
            out[n, 1] = b
            tags[n] = tag
            return n + 1
        return n
    disc = c1 * c1 - 4.0 * c0 * c2
    if disc < 0.0:
        if c2 < 0.0:
            if n >= cap:
                return -1
            out[n, 0] = ta
            out[n, 1] = tb
            tags[n] = tag
            return n + 1
        return n
    sq = math.sqrt(disc)
    qq = -0.5 * (c1 + sq) if c1 >= 0.0 else -0.5 * (c1 - sq)
    if qq == 0.0:
        r1 = 0.0
        r2 = 0.0
    else:
        r1 = qq / c2
        r2 = c0 / qq
    lo = r1 if r1 < r2 else r2
    hi = r2 if r1 < r2 else r1
    if c2 > 0.0:
        a = lo if lo > ta else ta
        b = hi if hi < tb else tb
        if b > a:
            if n >= cap:
                return -1
            out[n, 0] = a
            out[n, 1] = b
            tags[n] = tag
            return n + 1
        return n
    b = lo if lo < tb else tb
    if b > ta:
        if n >= cap:
            return -1
        out[n, 0] = ta
        out[n, 1] = b
        tags[n] = tag
        n += 1
    a = hi if hi > ta else ta
    if tb > a:
        if n >= cap:
            return -1
        out[n, 0] = a
        out[n, 1] = tb
        tags[n] = tag
        n += 1
    return n


@njit(f8(f8, f8, f8, f8, f8, f8), **INLINE)
def _p4(a0, a1, a2, a3, a4, x):
    return (((a4 * x + a3) * x + a2) * x + a1) * x + a0


@njit(f8(f8, f8, f8, f8, f8, f8), **INLINE)
def _dp4(a0, a1, a2, a3, a4, x):
    return ((4.0 * a4 * x + 3.0 * a3) * x + 2.0 * a2) * x + a1


@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, i8), **OPTS)
def _monotone_root(a0, a1, a2, a3, a4, lo, hi, tol, deriv):
    """Root of P (deriv 0) or P' (deriv 1) on [lo, hi] where it is monotone and changes sign."""
    if deriv == 0:
        flo = _p4(a0, a1, a2, a3, a4, lo)
    else:
        flo = _dp4(a0, a1, a2, a3, a4, lo)
    x = 0.5 * (lo + hi)
    for _ in range(200):
        if deriv == 0:
            fx = _p4(a0, a1, a2, a3, a4, x)
            dfx = _dp4(a0, a1, a2, a3, a4, x)
        else:
            fx = _dp4(a0, a1, a2, a3, a4, x)
            dfx = (12.0 * a4 * x + 6.0 * a3) * x + 2.0 * a2
        if fx == 0.0:
            return x
        if (fx > 0.0) == (flo > 0.0):
            lo = x
            flo = fx
        else:
            hi = x
        if hi - lo <= tol:
            return 0.5 * (lo + hi)
        xn = x - fx / dfx if dfx != 0.0 else 0.5 * (lo + hi)
        if not (xn > lo and xn < hi):
            xn = 0.5 * (lo + hi)
        x = xn
    return 0.5 * (lo + hi)


@njit(i8(f8, f8, f8, f8, f8, f8, f8, F1), **OPTS)
def poly4_roots(p0, p1, p2, p3, p4, ta, tb, work):
    """Real roots of p0 + p1 t + ... + p4 t^4 in (ta, tb), ascending (at most 4), by the derivative
    cascade on the interval mapped to [-1, 1]. ``work`` (>= 16 entries) returns them in work[0:n]."""
    if tb <= ta:
        return 0
    m = 0.5 * (ta + tb)
    h = 0.5 * (tb - ta)
    b4 = p4
    b3 = p3 + 4.0 * p4 * m
    b2 = p2 + (3.0 * p3 + 6.0 * p4 * m) * m
    b1 = p1 + (2.0 * p2 + (3.0 * p3 + 4.0 * p4 * m) * m) * m
    b0 = (((p4 * m + p3) * m + p2) * m + p1) * m + p0
    a4 = b4 * h * h * h * h
    a3 = b3 * h * h * h
    a2 = b2 * h * h
    a1 = b1 * h
    a0 = b0
    tol = ROOT_TOL_M / h
    # work[4:8] cuts of P'', work[8:14] cuts of P'
    nc2 = 0
    work[4 + nc2] = -1.0
    nc2 += 1
    c2 = 12.0 * a4
    c1 = 6.0 * a3
    c0 = 2.0 * a2
    if c2 != 0.0:
        disc = c1 * c1 - 4.0 * c2 * c0
        if disc > 0.0:
            sq = math.sqrt(disc)
            qq = -0.5 * (c1 + sq) if c1 >= 0.0 else -0.5 * (c1 - sq)
            ra = qq / c2
            rb = c0 / qq if qq != 0.0 else ra
            if ra > rb:
                ra, rb = rb, ra
            if -1.0 < ra < 1.0:
                work[4 + nc2] = ra
                nc2 += 1
            if -1.0 < rb < 1.0 and rb > ra:
                work[4 + nc2] = rb
                nc2 += 1
    elif c1 != 0.0:
        ra = -c0 / c1
        if -1.0 < ra < 1.0:
            work[4 + nc2] = ra
            nc2 += 1
    work[4 + nc2] = 1.0
    nc2 += 1
    nc1 = 0
    work[8 + nc1] = -1.0
    nc1 += 1
    for k in range(nc2 - 1):
        lo = work[4 + k]
        hi = work[4 + k + 1]
        flo = _dp4(a0, a1, a2, a3, a4, lo)
        fhi = _dp4(a0, a1, a2, a3, a4, hi)
        if flo == 0.0:
            if lo > work[8 + nc1 - 1]:
                work[8 + nc1] = lo
                nc1 += 1
        elif (flo < 0.0) != (fhi < 0.0) and fhi != 0.0:
            r = _monotone_root(a0, a1, a2, a3, a4, lo, hi, tol, 1)
            if r > work[8 + nc1 - 1]:
                work[8 + nc1] = r
                nc1 += 1
    if work[8 + nc1 - 1] < 1.0:
        work[8 + nc1] = 1.0
        nc1 += 1
    nr = 0
    for k in range(nc1 - 1):
        lo = work[8 + k]
        hi = work[8 + k + 1]
        flo = _p4(a0, a1, a2, a3, a4, lo)
        fhi = _p4(a0, a1, a2, a3, a4, hi)
        if flo == 0.0 and lo > -1.0:
            x = lo
        elif (flo < 0.0) != (fhi < 0.0) and fhi != 0.0:
            x = _monotone_root(a0, a1, a2, a3, a4, lo, hi, tol, 0)
        else:
            continue
        t = m + h * x
        if nr == 0 or t > work[nr - 1]:
            if nr < 4:
                work[nr] = t
                nr += 1
    return nr


@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, f8), **INLINE)
def _torus_g(A0, A1, A2, zeta0, s, rcp, rho, sigma, t):
    r2 = A0 + (A1 + A2 * t) * t
    r = math.sqrt(r2) if r2 > 0.0 else 0.0
    zz = zeta0 + s * t
    h2 = rho * rho - zz * zz
    return r - rcp - sigma * (math.sqrt(h2) if h2 > 0.0 else 0.0)


@njit(i8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, F2, I4_1, i8, i8, F1), **OPTS)
def torus_leq(A0, A1, A2, zeta0, s, rcp, rho, sigma, ta, tb, out, tags, n, tag, work):
    """Append the parts of [ta, tb] inside r <= rc' + sigma sqrt(rho^2 - zeta^2) (|zeta| <= rho there)."""
    if n < 0 or tb <= ta:
        return n
    q0 = A0 + zeta0 * zeta0 + rcp * rcp - rho * rho
    q1 = A1 + 2.0 * zeta0 * s
    q2 = A2 + s * s
    k4 = 4.0 * rcp * rcp
    nr = poly4_roots(q0 * q0 - k4 * A0, 2.0 * q0 * q1 - k4 * A1, q1 * q1 + 2.0 * q0 * q2 - k4 * A2, 2.0 * q1 * q2,
                     q2 * q2, ta, tb, work)
    cap = out.shape[0]
    prev = ta
    open_start = 0.0
    is_open = False
    for k in range(nr + 1):
        nxt = work[k] if k < nr else tb
        if nxt <= prev:
            continue
        ins = _torus_g(A0, A1, A2, zeta0, s, rcp, rho, sigma, 0.5 * (prev + nxt)) <= 0.0
        if ins and not is_open:
            open_start = prev
            is_open = True
        elif not ins and is_open:
            if n >= cap:
                return -1
            out[n, 0] = open_start
            out[n, 1] = prev
            tags[n] = tag
            n += 1
            is_open = False
        prev = nxt
    if is_open:
        if n >= cap:
            return -1
        out[n, 0] = open_start
        out[n, 1] = tb
        tags[n] = tag
        n += 1
    return n


@njit(i8(F2, I4_1, i8, f8, f8, i8), **INLINE)
def _push(out, tags, n, a, b, tag):
    if n < 0 or b <= a:
        return n
    if n >= out.shape[0]:
        return -1
    out[n, 0] = a
    out[n, 1] = b
    tags[n] = tag
    return n + 1


@njit(i8(i8, f8, f8, f8, f8, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, F2, I4_1, i8, F1), **INLINE)
def _piece(kind, c0, c1, c2, c3, tag, A0, A1, A2, z0p, s, dr, za, zb, ta, tb, out, tags, n, work):
    """Append the tau intervals inside one sub-piece (radius offset dr) for z in [za, zb], tau in [ta, tb].
    Every case reduces to one quadratic, one or two direct intervals, or the torus root isolation, each
    with a single call site (keeps the inlined code small)."""
    if n < 0:
        return n
    # a ray along the axis (s = +-1: 3-axis tools and the axis family) needs no division (x / 1 = x and
    # x / -1 = -x exactly)
    if s > 0.0:
        if s == 1.0:
            t0 = za - z0p
            t1 = zb - z0p
        else:
            t0 = (za - z0p) / s
            t1 = (zb - z0p) / s
    elif s < 0.0:
        if s == -1.0:
            t0 = -(zb - z0p)
            t1 = -(za - z0p)
        else:
            t0 = (zb - z0p) / s
            t1 = (za - z0p) / s
    else:
        if z0p < za or z0p > zb:
            return n
        t0 = ta
        t1 = tb
    if t0 < ta:
        t0 = ta
    if t1 > tb:
        t1 = tb
    if t1 <= t0:
        return n
    # mode: 0 nothing, 1 quadratic (q0, q1, q2 on [t0, t1]), 2 interval [i0, i1] (+ [j0, j1] when two), 3 torus
    mode = 0
    q0 = 0.0
    q1 = 0.0
    q2 = 0.0
    i0 = 0.0
    i1 = 0.0
    j0 = 0.0
    j1 = 0.0
    two = False
    zeta0 = 0.0
    rcp = 0.0
    rho = 0.0
    sigma = 0.0
    if kind == A.PK_LINE:
        R0 = c0 + dr + c1 * z0p
        R1 = c1 * s
        if R1 > 0.0:
            lim = -R0 / R1
            if lim > t0:
                t0 = lim
        elif R1 < 0.0:
            lim = -R0 / R1
            if lim < t1:
                t1 = lim
        elif R0 < 0.0:
            return n
        if t1 <= t0:
            return n
        mode = 1
        q0 = A0 - R0 * R0
        q1 = A1 - 2.0 * R0 * R1
        q2 = A2 - R1 * R1
    else:
        rcp = c1 + dr
        rho = c2
        sigma = c3
        zeta0 = z0p - c0
        if s == 0.0:
            h2 = rho * rho - zeta0 * zeta0
            R = rcp + sigma * (math.sqrt(h2) if h2 > 0.0 else 0.0)
            if R < 0.0:
                return n
            mode = 1
            q0 = A0 - R * R
            q1 = A1
            q2 = A2
        elif A2 == 0.0:
            r = math.sqrt(A0) if A0 > 0.0 else 0.0
            if sigma > 0.0:
                if r <= rcp:
                    mode = 2
                    i0 = t0
                    i1 = t1
                else:
                    h2 = rho * rho - (r - rcp) * (r - rcp)
                    if h2 < 0.0:
                        return n
                    h = math.sqrt(h2)
                    u0 = (-h - zeta0) / s
                    u1 = (h - zeta0) / s
                    if u0 > u1:
                        u0, u1 = u1, u0
                    mode = 2
                    i0 = max(u0, t0)
                    i1 = min(u1, t1)
            else:
                if r > rcp:
                    return n
                g = rho * rho - (rcp - r) * (rcp - r)
                mode = 2
                if g <= 0.0:
                    i0 = t0
                    i1 = t1
                else:
                    sg = math.sqrt(g)
                    ua = (-sg - zeta0) / s
                    ub = (sg - zeta0) / s
                    lo = ua if ua < ub else ub
                    hi = ub if ua < ub else ua
                    i0 = t0
                    i1 = min(lo, t1)
                    j0 = max(hi, t0)
                    j1 = t1
                    two = True
        elif sigma > 0.0 and abs(rcp) <= 1e-12 * rho:
            mode = 1
            q0 = A0 + zeta0 * zeta0 - rho * rho
            q1 = A1 + 2.0 * zeta0 * s
            q2 = A2 + s * s
        elif sigma < 0.0 and rcp <= 1e-12 * rho:
            return n
        else:
            mode = 3
    if mode == 1:
        return quad_leq(q0, q1, q2, t0, t1, out, tags, n, tag)
    if mode == 2:
        for k in range(2 if two else 1):
            a = i0 if k == 0 else j0
            b = i1 if k == 0 else j1
            n = _push(out, tags, n, a, b, tag)
        return n
    if mode == 3:
        return torus_leq(A0, A1, A2, zeta0, s, rcp, rho, sigma, t0, t1, out, tags, n, tag, work)
    return n


@njit(i8(F2, I4_1, i8), **INLINE)
def merge_intervals(out, tags, n):
    """Sort intervals by start and merge overlapping or touching ones (within MERGE_TOL_M)."""
    for i in range(1, n):
        a = out[i, 0]
        b = out[i, 1]
        t = tags[i]
        j = i - 1
        while j >= 0 and out[j, 0] > a:
            out[j + 1, 0] = out[j, 0]
            out[j + 1, 1] = out[j, 1]
            tags[j + 1] = tags[j]
            j -= 1
        out[j + 1, 0] = a
        out[j + 1, 1] = b
        tags[j + 1] = t
    m = 0
    for i in range(n):
        if m > 0 and out[i, 0] <= out[m - 1, 1] + MERGE_TOL_M:
            if out[i, 1] > out[m - 1, 1]:
                out[m - 1, 1] = out[i, 1]
        else:
            out[m, 0] = out[i, 0]
            out[m, 1] = out[i, 1]
            tags[m] = tags[i]
            m += 1
    return m


@njit(i8(F1, F1, types.int32[::1], F1, F1, F1, F1, types.int32[::1], f8, f8, f8, i8, F1, F2, F1, F1, f8, f8,
         F2, I4_1, i8, F1), **INLINE)
def _env_ray(pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, ox, oy, oz, fam, w, slab_c, slab_z, slab_dr, t_lo, t_hi,
             out, tags, merge, work):
    """Intervals (metres from the ray origin, within [t_lo, t_hi]) of the ray o + t e_fam inside the slabbed
    envelope, in slab and piece order, merged when ``merge``. Only the slabs that can matter are evaluated:
    when the caller left the slab bounds in ``work[16:16 + N_ENV]`` (subtract_envelope: the spread of the slab
    axis points, keyed by the slab arrays, and the ray's material extent), a search over the ascending slab
    heights finds the slabs the ray reaches where it has material (within a quantum); every slab is then
    skipped when the ray's height along the axis over [t_lo, t_hi] misses it by more than ENV_EPS (no piece of
    it could give an interval). The intervals are those of the full evaluation, less (with the extent) those
    that cannot reach the ray's material: the subtraction removes the same."""
    n = 0
    s = w[fam]
    ns = slab_dr.shape[0]
    m = pz0.shape[0]
    k0 = 0
    k1 = ns
    e = 16
    if (ns > 0 and work.shape[0] >= e + A.N_ENV and work[e + ENV_KEY] == ENV_MAGIC and work[e + ENV_NS] == ns
            and work[e + ENV_CX] == slab_c[0, 0] and work[e + ENV_CY] == slab_c[0, 1]
            and work[e + ENV_CZ] == slab_c[0, 2] and work[e + ENV_Z0] == slab_z[0] and work[e + ENV_Z1] == slab_z[ns]):
        # height of the ray along the axis over the part of [t_lo, t_hi] that can matter (the material extent
        # the caller gave, widened by a quantum and the margin), in slab 0's frame; slab k's frame differs by
        # (c_0 - c_k) . w, at most the spread of the axis points
        mrg = work[e + ENV_SPREAD] + ENV_EPS
        tq0 = work[e + ENV_TLO] - Q - ENV_EPS
        tq1 = work[e + ENV_THI] + Q + ENV_EPS
        if tq0 < t_lo:
            tq0 = t_lo
        if tq1 > t_hi:
            tq1 = t_hi
        if tq1 < tq0:
            return 0
        zr = (ox - slab_c[0, 0]) * w[0] + (oy - slab_c[0, 1]) * w[1] + (oz - slab_c[0, 2]) * w[2]
        za_ = zr + s * tq0
        zb_ = zr + s * tq1
        zlo = (za_ if za_ < zb_ else zb_) - mrg
        zhi = (zb_ if za_ < zb_ else za_) + mrg
        # first slab whose top reaches zlo, then the first whose bottom is above zhi (ascending heights; a guess
        # from the mean slab height, then a walk: exact for any heights)
        hinv = work[e + ENV_HINV]
        g = (zlo - slab_z[0]) * hinv
        k0 = int(g) if (g > 0.0 and g < ns - 1) else (ns - 1 if g >= ns - 1 else 0)
        while k0 > 0 and slab_z[k0] >= zlo:
            k0 -= 1
        while k0 < ns and slab_z[k0 + 1] < zlo:
            k0 += 1
        g = (zhi - slab_z[0]) * hinv + 1.0
        k1 = int(g) if (g > k0 and g < ns) else (ns if g >= ns else k0)
        while k1 > k0 and slab_z[k1 - 1] > zhi:
            k1 -= 1
        while k1 < ns and slab_z[k1] <= zhi:
            k1 += 1
    # a ray along the tool axis (the axis exactly along the family: z0p = +-0, A1 = +-0, A2 = 0): a cylinder piece's
    # quadratic is the test A0 <= R0^2 on the slab's tau range (_piece / quad_leq's own branch for these values)
    axial = (s == 1.0 or s == -1.0) and w[(fam + 1) % 3] == 0.0 and w[(fam + 2) % 3] == 0.0
    for k in range(k0, k1):
        zs0 = slab_z[k]
        zs1 = slab_z[k + 1]
        if zs1 <= zs0:
            continue
        qx = ox - slab_c[k, 0]
        qy = oy - slab_c[k, 1]
        qz = oz - slab_c[k, 2]
        if fam == 0:
            ts = -qx
            qx = 0.0
        elif fam == 1:
            ts = -qy
            qy = 0.0
        else:
            ts = -qz
            qz = 0.0
        z0p = qx * w[0] + qy * w[1] + qz * w[2]
        ta = t_lo - ts
        tb = t_hi - ts
        # the ray's height over [ta, tb] misses the slab: no piece of it gives an interval
        if s == 0.0:
            if z0p < zs0 - ENV_EPS or z0p > zs1 + ENV_EPS:
                continue
        else:
            zq0 = z0p + s * ta
            zq1 = z0p + s * tb
            if (zq1 if zq0 < zq1 else zq0) < zs0 - ENV_EPS or (zq0 if zq0 < zq1 else zq1) > zs1 + ENV_EPS:
                continue
        A0 = qx * qx + qy * qy + qz * qz - z0p * z0p
        A1 = -2.0 * z0p * s
        A2 = 1.0 - s * s
        start = n
        for p in range(m):
            if pz1[p] <= zs0 or pz0[p] >= zs1:
                continue
            za = zs0 if zs0 > pz0[p] else pz0[p]
            zb = zs1 if zs1 < pz1[p] else pz1[p]
            if zb <= za:
                continue
            if axial and pkind[p] == A.PK_LINE and pc1[p] == 0.0:
                if s > 0.0:
                    t0 = za - z0p
                    t1 = zb - z0p
                else:
                    t0 = -(zb - z0p)
                    t1 = -(za - z0p)
                if t0 < ta:
                    t0 = ta
                if t1 > tb:
                    t1 = tb
                if t1 <= t0:
                    continue
                R0 = pc0[p] + slab_dr[k] + pc1[p] * z0p
                if R0 < 0.0 or not (A0 - R0 * R0 <= 0.0):
                    continue
                n = _push(out, tags, n, t0, t1, ppart[p])
            else:
                n = _piece(pkind[p], pc0[p], pc1[p], pc2[p], pc3[p], ppart[p], A0, A1, A2, z0p, s, slab_dr[k], za,
                           zb, ta, tb, out, tags, n, work)
            if n < 0:
                return -1
        for i in range(start, n):
            out[i, 0] += ts
            out[i, 1] += ts
    if merge == 1 and n > 1:
        n = merge_intervals(out, tags, n)
    return n


# --------------------------------------------------------------------------- events
@njit(void(F2, I1, i8, i8, i8, f8, f8, f8, f8, f8, i8, i8, i8, i8), **INLINE)
def _event(rec, nn, kind, tick, contact, x, y, z, depth, volume, part, sev, level, fam):
    i = nn[0]
    if i >= rec.shape[0]:
        nn[1] = 1
        return
    rec[i, A.EC_KIND] = kind
    rec[i, A.EC_TICK] = tick
    rec[i, A.EC_CONTACT] = contact
    rec[i, A.EC_X] = x
    rec[i, A.EC_Y] = y
    rec[i, A.EC_Z] = z
    rec[i, A.EC_DEPTH] = depth
    rec[i, A.EC_VOLUME] = volume
    rec[i, A.EC_PART] = part
    rec[i, A.EC_SEV_ID] = sev
    rec[i, A.EC_LEVEL] = level
    rec[i, A.EC_FAMILY] = fam
    nn[0] = i + 1


@njit(void(EVENTS_T, i8, i8, i8, f8, f8, f8, f8, f8, i8, i8, i8, i8), **OPTS)
def push_event(ev, kind, tick, contact, x, y, z, depth, volume, part, sev, level, fam):
    _event(ev.rec, ev.n, kind, tick, contact, x, y, z, depth, volume, part, sev, level, fam)


# --------------------------------------------------------------------------- bounds
@njit(f8(F1, F1, F2, F1, F1, f8, f8, f8, f8, F1, F1), **INLINE)
def _aabb(w, slab_dr, slab_c, slab_z, unused, zmin_p, zmax_p, rmax_p, slack, lo, hi):
    ns = slab_dr.shape[0]
    zmin = zmin_p if slab_z[0] < zmin_p else slab_z[0]
    zmax = zmax_p if slab_z[ns] > zmax_p else slab_z[ns]
    drmax = -1e300
    for k in range(ns):
        if slab_dr[k] > drmax:
            drmax = slab_dr[k]
    if drmax < 0.0:
        drmax = 0.0
    spread = 0.0
    for k in range(1, ns):
        dx = slab_c[k, 0] - slab_c[0, 0]
        dy = slab_c[k, 1] - slab_c[0, 1]
        dz = slab_c[k, 2] - slab_c[0, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > spread:
            spread = d
    R = rmax_p + drmax + slack
    # per axis: the extent of the slab axis points along it (not their 3-D spread), the axial range and the radius
    # projected on it, widened by a margin far above the rounding of the envelope's own end points
    for i in range(3):
        cmin = slab_c[0, i]
        cmax = slab_c[0, i]
        for k in range(1, ns):
            if slab_c[k, i] < cmin:
                cmin = slab_c[k, i]
            if slab_c[k, i] > cmax:
                cmax = slab_c[k, i]
        za = zmin * w[i]
        zb = zmax * w[i]
        e = R * math.sqrt(max(0.0, 1.0 - w[i] * w[i]))
        lo[i] = cmin + min(za, zb) - e - AABB_EPS
        hi[i] = cmax + max(za, zb) + e + AABB_EPS
    return R + spread


@njit(f8(PROFILE_T, SPEC_T, F1, F1), **OPTS)
def envelope_aabb(prof, spec, lo, hi):
    """Box of the slabbed envelope in S (written to lo, hi); returns the quick-reject radius."""
    return _aabb(spec.axis_S, spec.slab_dr, spec.slab_c, spec.slab_z, spec.slab_dr, prof.info[A.PI_ZMIN],
                 prof.info[A.PI_ZMAX], prof.info[A.PI_RMAX], spec.fvals[A.SF_SLACK], lo, hi)


@njit(types.UniTuple(i8, 2)(f8, f8, f8, i8), **INLINE)
def node_range(lo, hi, origin, n):
    """Node indices i in [0, n - 1] with origin + i in [lo, hi] (all in lattice units)."""
    i0 = int(math.ceil(lo - origin - 1e-9))
    i1 = int(math.floor(hi - origin + 1e-9))
    if i0 < 0:
        i0 = 0
    if i1 > n - 1:
        i1 = n - 1
    return np.int64(i0), np.int64(i1)


@njit(i8(f8, f8, f8, i8, F1, f8, f8, f8, f8, f8, f8, f8, f8, F2, I4_1), **OPTS)
def _dead_cyl(ox, oy, oz, fam, w, cx, cy, cz, r_dead, z0, z1, t_lo, t_hi, out, tags):
    """Intervals (t, metres from the ray origin) of the ray inside the cylinder r <= r_dead, z in [z0, z1]."""
    qx = ox - cx
    qy = oy - cy
    qz = oz - cz
    if fam == 0:
        ts = -qx
        qx = 0.0
    elif fam == 1:
        ts = -qy
        qy = 0.0
    else:
        ts = -qz
        qz = 0.0
    s = w[fam]
    z0p = qx * w[0] + qy * w[1] + qz * w[2]
    A0 = qx * qx + qy * qy + qz * qz - z0p * z0p
    A1 = -2.0 * z0p * s
    A2 = 1.0 - s * s
    ta = t_lo - ts
    tb = t_hi - ts
    if s > 0.0:
        u0 = (z0 - z0p) / s
        u1 = (z1 - z0p) / s
    elif s < 0.0:
        u0 = (z1 - z0p) / s
        u1 = (z0 - z0p) / s
    else:
        if z0p < z0 or z0p > z1:
            return 0
        u0 = ta
        u1 = tb
    if u0 < ta:
        u0 = ta
    if u1 > tb:
        u1 = tb
    n = quad_leq(A0 - r_dead * r_dead, A1, A2, u0, u1, out, tags, 0, 0)
    if n < 0:
        n = out.shape[0]
    for i in range(n):
        out[i, 0] += ts
        out[i, 1] += ts
    return n



KERNELS = (_monotone_root, poly4_roots, torus_leq, _dead_cyl, push_event, envelope_aabb)
INLINED = (quad_leq, _p4, _dp4, _torus_g, _push, _piece, merge_intervals, _env_ray, _event, _aabb, node_range)
