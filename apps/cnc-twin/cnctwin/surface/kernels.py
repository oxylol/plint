"""Numba kernels of the surface domain (docs/models/surface.md "Kernels"; rules: architecture 5.5).

Every kernel is compiled eagerly from one explicit signature (``cache=True, nogil=True,
error_model="numpy"``, no fastmath, no parallel) and has a plain-Python twin in
:mod:`cnctwin.surface.reference` that the tests compare it with.

* :func:`pass_polyline` - the generating edge of one pass as a polyline in the line frame: the outline
  of the tool (or the work's view of the insert) moved to the element's position and projected along the
  cutting velocity onto the profile plane, lifted by the path curvature (Euler: normal curvature
  kappa_v cos^2 + silhouette), or the element's own path (a parabola) when the cutting velocity lies in
  the profile plane, or a direct (u, w) template; the built-up edge lowers it near the element.
* :func:`envelope_profile` - the kinematic profile: every pass rasterised onto the grid in time order and
  merged as the lower envelope where its chip thickness normal to the edge reaches h_min; a thinner layer is
  ploughed: pressed to the pass, its volume flowing into the free space under the pass beside it (side flow)
  or, where there is none, staying in place (Brammertz 1961); additive passes add dents and fragments.
* :func:`gaussian_mean` - the ISO 16610-21 Gaussian mean line (lambda_c or lambda_s), evaluated every
  ``stride`` samples and interpolated linearly in between, weights renormalised at the profile ends.
* :func:`profile_params` - ISO 4287 parameters of a roughness profile over whole sampling lengths.
* :func:`silhouette` - lower envelope of a revolved outline seen along the feed direction (the effective
  cutting profile of a rotating tool across the feed: step-over scallops, Vickers & Quan 1989).
* :func:`wire_centre` - radial position of a measuring wire resting on both flanks of a thread groove.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A

OPTS = dict(cache=True, nogil=True, error_model="numpy")

f8 = types.float64
i8 = types.int64
F1 = types.float64[::1]

PASSES_T = A.PASSES_T
TEMPLATES_T = A.TEMPLATES_T
GRID_T = A.GRID_T

KIND_CUT = A.KIND_CUT
MODE_DIRECT = A.MODE_DIRECT
MODE_OUTLINE = A.MODE_OUTLINE
MODE_PARABOLA = A.MODE_PARABOLA
INF = np.inf
VERTICAL_SLOPE = 1.0e6          # slope given to a segment with no extent along the profile
PLANE_V_TOL = 1.0e-6            # |v.m| below this: the cutting velocity lies in the profile plane


@njit(i8(PASSES_T, TEMPLATES_T, i8, F1, F1, i8), **OPTS)
def pass_polyline(passes, tpl, k, ps, pw, n_parab):
    """Polyline (ps, pw) of pass k in the line frame; returns the point count (pw NaN: dropped point)."""
    mode = passes.mode[k]
    sk = passes.s[k]
    wk = passes.w[k]
    mk = passes.m[k]
    b = passes.basis[k]
    vd = b[A.B_VD]
    vn = b[A.B_VN]
    vm = b[A.B_VM]
    nd = b[A.B_ND]
    nn = b[A.B_NN]
    nm = b[A.B_NM]
    curv = passes.curv[k]
    tmax = passes.t_max[k]
    delta = passes.bue[k, 0]
    bw = passes.bue[k, 1]
    cap = ps.shape[0]
    if mode == MODE_OUTLINE and abs(vm) < PLANE_V_TOL:
        mode = MODE_PARABOLA
    if mode == MODE_PARABOLA:
        npt = n_parab if n_parab < cap else cap
        if npt < 2:
            return 0
        for j in range(npt):
            t = -tmax + 2.0 * tmax * j / (npt - 1)
            lift = 0.5 * curv * t * t
            sj = sk + t * vd + lift * nd
            wj = wk + t * vn + lift * nn
            if delta > 0.0 and bw > 0.0 and abs(t) < bw:
                f = delta * (1.0 - (t / bw) * (t / bw))
                sj -= f * nd
                wj -= f * nn
            ps[j] = sj
            pw[j] = wj
        return npt
    t_index = passes.tpl[k]
    if t_index < 0 or t_index + 1 >= tpl.off.shape[0]:
        return 0
    i0 = tpl.off[t_index]
    i1 = tpl.off[t_index + 1]
    npt = i1 - i0
    if npt > cap:
        npt = cap
    if mode == MODE_DIRECT:
        for j in range(npt):
            ps[j] = sk + tpl.pts[i0 + j, 0]
            pw[j] = wk + tpl.pts[i0 + j, 1]
        return npt
    re = passes.elem_rz[k, 0]
    ze = passes.elem_rz[k, 1]
    rd = b[A.B_RD]
    rn = b[A.B_RN]
    rm = b[A.B_RM]
    ad = b[A.B_AD]
    an = b[A.B_AN]
    am = b[A.B_AM]
    for j in range(npt):
        dr = tpl.pts[i0 + j, 0] - re
        dz = tpl.pts[i0 + j, 1] - ze
        pd = dr * rd + dz * ad
        pn = dr * rn + dz * an
        pm = dr * rm + dz * am
        if delta > 0.0 and bw > 0.0:
            dist = math.sqrt(dr * dr + dz * dz)
            if dist < bw:
                f = delta * (1.0 - (dist / bw) * (dist / bw))
                pd -= f * nd
                pn -= f * nn
                pm -= f * nm
        t = -(mk + pm) / vm
        if abs(t) > tmax:
            ps[j] = sk + pd
            pw[j] = np.nan
            continue
        lift = 0.5 * curv * t * t
        ps[j] = sk + pd + t * vd + lift * nd
        pw[j] = wk + pn + t * vn + lift * nn
    return npt


@njit(types.void(GRID_T, i8, i8, i8, i8, f8), **OPTS)
def _place_ploughed(grid, a, b, lo, hi, vol):
    """Volume ``vol`` ploughed out of samples a..b (already pressed to the pass) flows into the free space
    under the pass on both sides, nearest samples first, alternating; what finds no free space stays where
    it was ploughed (spread evenly over a..b)."""
    left = a - 1
    right = b + 1
    while vol > 0.0:
        moved = False
        if lo <= left:
            t = grid.tmp[left]
            zl = grid.z[left]
            if t < INF and zl < INF and t > zl:
                room = t - zl
                if vol < room:
                    grid.z[left] = zl + vol
                    vol = 0.0
                    break
                grid.z[left] = t
                vol -= room
                left -= 1
                moved = True
        if vol > 0.0 and right <= hi:
            t = grid.tmp[right]
            zr = grid.z[right]
            if t < INF and zr < INF and t > zr:
                room = t - zr
                if vol < room:
                    grid.z[right] = zr + vol
                    vol = 0.0
                    break
                grid.z[right] = t
                vol -= room
                right += 1
                moved = True
        if not moved:
            break
    if vol > 0.0:
        share = vol / (b - a + 1)
        for j in range(a, b + 1):
            grid.z[j] += share


@njit(types.void(PASSES_T, TEMPLATES_T, GRID_T, i8, f8), **OPTS)
def envelope_profile(passes, tpl, grid, n_parab, recovery):
    """Merge every pass, in order, into grid.z (inf where nothing generated the surface yet).

    A cutting pass removes material only where it lies below the surface by a chip thickness (normal to the
    pass's edge) of at least h_min; between h_min and (1 + recovery) h_min the sample moves part of the way,
    w = (h - h_min) / (recovery h_min) (the partial-removal band of cutting's surface memory). Where the pass
    lies below the surface by less than h_min the layer is ploughed, not cut: it is pressed to the pass and,
    being plastically incompressible, flows into the free space under the pass next to the ploughed run
    (side flow: nearest samples first, both sides alternating, never into the chip); what finds no free space
    stays in place (steady plane ploughing returns the surface behind the edge: Brammertz's uncut layer,
    which a later pass then meets as a thicker layer). The thin layer of Brammertz's chip tail thus moves to
    the other side of the intersection of two passes, which keeps his R_t = f^2 / 8 r + (h_min / 2)(1 +
    r h_min / f^2), while a flat (wiper or minor) edge passing over feed-mark ridges levels them into the
    valleys instead of leaving them."""
    u0 = grid.u0[0]
    du = grid.du[0]
    inv_du = 1.0 / du
    n = grid.z.shape[0]
    ps = grid.ps
    pw = grid.pw
    for k in range(passes.s.shape[0]):
        npt = pass_polyline(passes, tpl, k, ps, pw, n_parab)
        if npt < 2:
            continue
        kind = passes.kind[k]
        lo_s = -INF
        hi_s = INF
        if kind == KIND_CUT:
            wmin = INF
            smin = 0.0
            for j in range(npt):
                if pw[j] == pw[j] and pw[j] < wmin:
                    wmin = pw[j]
                    smin = ps[j]
            if wmin == INF:
                continue
            lo_s = smin - passes.half[k]
            hi_s = smin + passes.half[k]
        ilo = n
        ihi = -1
        for j in range(npt - 1):
            sa = ps[j]
            wa = pw[j]
            sb = ps[j + 1]
            wb = pw[j + 1]
            if wa != wa or wb != wb:
                continue
            a = sa if sa < sb else sb
            c = sb if sa < sb else sa
            if c < lo_s or a > hi_s:
                continue
            if a < lo_s:
                a = lo_s
            if c > hi_s:
                c = hi_s
            ia = int(math.ceil((a - u0) * inv_du))
            ic = int(math.floor((c - u0) * inv_du))
            if ia < 0:
                ia = 0
            if ic > n - 1:
                ic = n - 1
            if ic < ia:
                continue
            ds = sb - sa
            flat = abs(ds) <= 1.0e-15
            slope = VERTICAL_SLOPE if flat else (wb - wa) / ds
            for i in range(ia, ic + 1):
                if flat:
                    wv = wa if wa < wb else wb
                else:
                    wv = wa + (u0 + i * du - sa) * slope
                if wv < grid.tmp[i]:
                    grid.tmp[i] = wv
                    grid.tmp_slope[i] = slope
            if ia < ilo:
                ilo = ia
            if ic > ihi:
                ihi = ic
        hmin = passes.hmin[k]
        if kind == KIND_CUT:
            i = ilo
            while i <= ihi:
                t = grid.tmp[i]
                if not t < INF:
                    i += 1
                    continue
                zi = grid.z[i]
                if t < zi:
                    if zi == INF:
                        grid.z[i] = t
                    else:
                        sl = grid.tmp_slope[i]
                        h = (zi - t) / math.sqrt(1.0 + sl * sl)
                        if h >= hmin:
                            if hmin > 0.0 and recovery > 0.0 and h < (1.0 + recovery) * hmin:
                                grid.z[i] = zi + (h - hmin) / (recovery * hmin) * (t - zi)
                            else:
                                grid.z[i] = t
                        else:
                            # a ploughed run: pressed to the pass, its volume placed by _place_ploughed
                            run0 = i
                            vol = 0.0
                            while i <= ihi:
                                t = grid.tmp[i]
                                zi = grid.z[i]
                                if not (t < INF and t < zi < INF):
                                    break
                                sl = grid.tmp_slope[i]
                                if (zi - t) / math.sqrt(1.0 + sl * sl) >= hmin:
                                    break
                                vol += zi - t
                                grid.z[i] = t
                                grid.hits[i] += 1
                                i += 1
                            _place_ploughed(grid, run0, i - 1, ilo, ihi, vol)
                            continue
                grid.hits[i] += 1
                i += 1
        else:
            for i in range(ilo, ihi + 1):
                t = grid.tmp[i]
                if t < INF and grid.z[i] < INF:
                    grid.z[i] += t
        for i in range(ilo, ihi + 1):
            grid.tmp[i] = INF
            grid.tmp_slope[i] = 0.0


@njit(i8(PASSES_T, TEMPLATES_T, GRID_T, i8, f8, F1), **OPTS)
def pass_valleys(passes, tpl, grid, n_parab, side, out):
    """Profile positions of the lowest points of the cutting passes into out (returns the count): the
    minimum of each pass's polyline, skipped when it falls on the polyline's first or last valid point (the
    window ends there: not a valley of the pass); side != 0 keeps the passes whose element lies on that side
    of the axis along the profile (sign of r_hat . d)."""
    count = 0
    for k in range(passes.s.shape[0]):
        if passes.kind[k] != KIND_CUT:
            continue
        if side != 0.0:
            rd = passes.basis[k, A.B_RD]
            sg = 1.0 if rd > 0.0 else (-1.0 if rd < 0.0 else 0.0)
            if sg != side:
                continue
        npt = pass_polyline(passes, tpl, k, grid.ps, grid.pw, n_parab)
        first = -1
        last = -1
        best = -1
        wmin = INF
        for j in range(npt):
            w = grid.pw[j]
            if w == w:
                if first < 0:
                    first = j
                last = j
                if w < wmin:
                    wmin = w
                    best = j
        if best < 0 or best == first or best == last:
            continue
        out[count] = grid.ps[best]
        count += 1
    return count


@njit(types.void(F1, f8, f8, f8, i8, F1), **OPTS)
def gaussian_mean(z, du, lam, alpha, stride, out):
    """Gaussian weighted mean line of z (ISO 16610-21, cut-off lam) at every sample (renormalised ends)."""
    n = z.shape[0]
    if n == 0:
        return
    if stride < 1:
        stride = 1
    half = int(lam / du) + 1
    wts = np.empty(2 * half + 1)
    al = alpha * lam
    for q in range(2 * half + 1):
        x = (q - half) * du / al
        wts[q] = math.exp(-math.pi * x * x)
    last = -1
    c = 0
    while True:
        if c > n - 1:
            c = n - 1
        lo = c - half
        hi = c + half
        if lo < 0:
            lo = 0
        if hi > n - 1:
            hi = n - 1
        acc = 0.0
        wsum = 0.0
        for j in range(lo, hi + 1):
            wq = wts[j - c + half]
            acc += wq * z[j]
            wsum += wq
        out[c] = acc / wsum
        if last >= 0 and c - last > 1:
            span = c - last
            for j in range(last + 1, c):
                f = (j - last) / span
                out[j] = out[last] + f * (out[c] - out[last])
        last = c
        if c == n - 1:
            break
        c += stride


@njit(types.UniTuple(f8, 2)(F1, i8, i8, f8, f8, f8), **OPTS)
def _rsm_core(r, i0, ne, du, h_min, w_min):
    """(RSm, element count) of r[i0:i0+ne]."""
    sg = np.zeros(ne + 1, np.int64)
    a0 = np.zeros(ne + 1, np.int64)
    a1 = np.zeros(ne + 1, np.int64)
    hh = np.zeros(ne + 1)
    nseg = 0
    cur = 0
    for i in range(i0, i0 + ne):
        v = r[i]
        sgn = 1 if v > 0.0 else (-1 if v < 0.0 else 0)
        if sgn == 0:
            continue
        if sgn != cur:
            sg[nseg] = sgn
            a0[nseg] = i
            a1[nseg] = i + 1
            hh[nseg] = abs(v)
            nseg += 1
            cur = sgn
        else:
            a1[nseg - 1] = i + 1
            if abs(v) > hh[nseg - 1]:
                hh[nseg - 1] = abs(v)
    # drop the truncated first and last segments
    first = 1 if nseg > 0 and a0[0] == i0 and r[i0] != 0.0 else 0
    last = nseg - 1 if nseg > 0 and a1[nseg - 1] == i0 + ne else nseg
    ksg = np.zeros(ne + 1, np.int64)
    k0 = np.zeros(ne + 1, np.int64)
    k1 = np.zeros(ne + 1, np.int64)
    nk = 0
    for q in range(first, last):
        width = (a1[q] - a0[q]) * du
        if hh[q] < h_min or width < w_min:
            if nk > 0:
                k1[nk - 1] = a1[q]
            continue
        if nk > 0 and ksg[nk - 1] == sg[q]:
            k1[nk - 1] = a1[q]
            continue
        ksg[nk] = sg[q]
        k0[nk] = a0[q]
        k1[nk] = a1[q]
        nk += 1
    total = 0.0
    count = 0
    q = 0
    while q + 1 < nk:
        if ksg[q] == 1 and ksg[q + 1] == -1:
            total += (k1[q + 1] - k0[q]) * du
            count += 1
            q += 2
        else:
            q += 1
    return (total / count if count > 0 else 0.0), float(count)


@njit(types.void(F1, f8, i8, i8, i8, f8, f8, F1), **OPTS)
def profile_params(r, du, i0, lsl, n_sl, hd, wd, out):
    """ISO 4287 parameters of the roughness profile r over n_sl sampling lengths of lsl samples from i0."""
    for q in range(out.shape[0]):
        out[q] = 0.0
    ne = lsl * n_sl
    if ne < 2 or n_sl < 1 or i0 + ne > r.shape[0]:
        return
    s1 = 0.0
    s2 = 0.0
    for i in range(i0, i0 + ne):
        s1 += abs(r[i])
        s2 += r[i] * r[i]
    ra = s1 / ne
    rq = math.sqrt(s2 / ne)
    s3 = 0.0
    s4 = 0.0
    sd = 0.0
    for i in range(i0, i0 + ne):
        s3 += r[i] * r[i] * r[i]
        s4 += r[i] * r[i] * r[i] * r[i]
        if i + 1 < i0 + ne:
            g = (r[i + 1] - r[i]) / du
            sd += g * g
    gmax = -INF
    gmin = INF
    sp = 0.0
    sv = 0.0
    for s in range(n_sl):
        pmax = -INF
        vmin = INF
        for i in range(i0 + s * lsl, i0 + (s + 1) * lsl):
            if r[i] > pmax:
                pmax = r[i]
            if r[i] < vmin:
                vmin = r[i]
        sp += pmax
        sv += -vmin
        if pmax > gmax:
            gmax = pmax
        if vmin < gmin:
            gmin = vmin
    out[A.P_RA] = ra
    out[A.P_RQ] = rq
    out[A.P_RSK] = (s3 / ne) / (rq * rq * rq) if rq > 0.0 else 0.0
    out[A.P_RKU] = (s4 / ne) / (rq * rq * rq * rq) if rq > 0.0 else 0.0
    out[A.P_RP] = sp / n_sl
    out[A.P_RV] = sv / n_sl
    out[A.P_RZ] = (sp + sv) / n_sl
    out[A.P_RT] = gmax - gmin
    out[A.P_RDQ] = math.sqrt(sd / (ne - 1))
    out[A.P_N_SL] = n_sl
    # RSm (ISO 4287 3.3.2): profile elements = a peak and the following valley between zero crossings;
    # segments lower than hd x Rz or narrower than wd x the sampling length are absorbed into the previous
    # significant segment, equal-sign neighbours then merge; truncated segments at either end are not counted
    rsm, count = _rsm_core(r, i0, ne, du, hd * out[A.P_RZ], wd * lsl * du)
    out[A.P_RSM] = rsm
    out[A.P_N_ELEM] = count


@njit(types.void(f8, f8, f8, f8, f8, f8, f8, F1), **OPTS)
def _raster_min(sa, wa, sb, wb, u0, inv_du, du, out):
    n = out.shape[0]
    a = sa if sa < sb else sb
    c = sb if sa < sb else sa
    ia = int(math.ceil((a - u0) * inv_du))
    ic = int(math.floor((c - u0) * inv_du))
    if ia < 0:
        ia = 0
    if ic > n - 1:
        ic = n - 1
    ds = sb - sa
    for i in range(ia, ic + 1):
        if abs(ds) <= 1.0e-15:
            wv = wa if wa < wb else wb
        else:
            wv = wa + (u0 + i * du - sa) * (wb - wa) / ds
        if wv < out[i]:
            out[i] = wv


@njit(types.void(F1, F1, F1, F1, F1, F1, F1, i8, f8, f8, F1), **OPTS)
def silhouette(r, z, a, e1, e2, tc, nn, n_phi, u0, du, out):
    """Lower envelope (along nn) of the solid of revolution of the outline (r, z) about axis a (unit
    vectors e1, e2 complete the frame), as a function of u = p . tc; segments between neighbouring
    samples in angle and along the outline are rasterised so the envelope has no gaps."""
    n = out.shape[0]
    m = r.shape[0]
    inv_du = 1.0 / du
    at = a[0] * tc[0] + a[1] * tc[1] + a[2] * tc[2]
    an = a[0] * nn[0] + a[1] * nn[1] + a[2] * nn[2]
    e1t = e1[0] * tc[0] + e1[1] * tc[1] + e1[2] * tc[2]
    e1n = e1[0] * nn[0] + e1[1] * nn[1] + e1[2] * nn[2]
    e2t = e2[0] * tc[0] + e2[1] * tc[1] + e2[2] * tc[2]
    e2n = e2[0] * nn[0] + e2[1] * nn[1] + e2[2] * nn[2]
    us = np.empty(n_phi + 1)
    ws = np.empty(n_phi + 1)
    pu = np.empty(n_phi + 1)
    pv = np.empty(n_phi + 1)
    for j in range(m):
        for q in range(n_phi + 1):
            phi = 2.0 * math.pi * q / n_phi
            c = math.cos(phi)
            s = math.sin(phi)
            us[q] = z[j] * at + r[j] * (c * e1t + s * e2t)
            ws[q] = z[j] * an + r[j] * (c * e1n + s * e2n)
        for q in range(n_phi):
            _raster_min(us[q], ws[q], us[q + 1], ws[q + 1], u0, inv_du, du, out)
        if j > 0:
            for q in range(n_phi + 1):
                _raster_min(pu[q], pv[q], us[q], ws[q], u0, inv_du, du, out)
        for q in range(n_phi + 1):
            pu[q] = us[q]
            pv[q] = ws[q]


@njit(f8(F1, F1, f8, f8, i8), **OPTS)
def wire_centre(pz, pr, zc, radius, external):
    """Radius of the centre of a wire (circle of ``radius`` in the axial plane, axial position zc) resting
    on the profile points (pz, pr): outside the material of an external thread (largest), inside the
    groove of an internal thread (smallest). NaN when no profile point lies under the wire."""
    best = -INF if external != 0 else INF
    found = False
    for i in range(pz.shape[0]):
        dz = pz[i] - zc
        if abs(dz) > radius:
            continue
        h = math.sqrt(radius * radius - dz * dz)
        if external != 0:
            c = pr[i] + h
            if c > best:
                best = c
        else:
            c = pr[i] - h
            if c < best:
                best = c
        found = True
    return best if found else np.nan


__all__ = ["pass_polyline", "envelope_profile", "pass_valleys", "gaussian_mean", "profile_params", "silhouette",
           "wire_centre"]
