"""Plain-Python references of the surface kernels (architecture 5.5: every kernel has one, the tests
compare them). Written independently of :mod:`cnctwin.surface.kernels`: loops over plain floats with
``math``, the same algorithms stated in docs/models/surface.md."""

from __future__ import annotations

import math

import numpy as np

from . import abi as A


def pass_polyline_ref(passes: A.Passes, tpl: A.Templates, k: int, cap: int, n_parab: int):
    """(s list, w list) of pass k (w None: dropped point)."""
    mode = int(passes.mode[k])
    sk, wk, mk = float(passes.s[k]), float(passes.w[k]), float(passes.m[k])
    b = [float(x) for x in passes.basis[k]]
    vd, vn, vm = b[A.B_VD], b[A.B_VN], b[A.B_VM]
    nd, nn, nm = b[A.B_ND], b[A.B_NN], b[A.B_NM]
    curv = float(passes.curv[k])
    tmax = float(passes.t_max[k])
    delta, bw = float(passes.bue[k, 0]), float(passes.bue[k, 1])
    if mode == A.MODE_OUTLINE and abs(vm) < 1.0e-6:
        mode = A.MODE_PARABOLA
    s_out, w_out = [], []
    if mode == A.MODE_PARABOLA:
        npt = min(n_parab, cap)
        if npt < 2:
            return [], []
        for j in range(npt):
            t = -tmax + 2.0 * tmax * j / (npt - 1)
            lift = 0.5 * curv * t * t
            s = sk + t * vd + lift * nd
            w = wk + t * vn + lift * nn
            if delta > 0.0 and bw > 0.0 and abs(t) < bw:
                f = delta * (1.0 - (t / bw) ** 2)
                s -= f * nd
                w -= f * nn
            s_out.append(s)
            w_out.append(w)
        return s_out, w_out
    index = int(passes.tpl[k])
    if index < 0 or index + 1 >= len(tpl.off):
        return [], []
    pts = tpl.pts[int(tpl.off[index]):int(tpl.off[index + 1])][:cap]
    if mode == A.MODE_DIRECT:
        return [sk + float(p[0]) for p in pts], [wk + float(p[1]) for p in pts]
    re, ze = float(passes.elem_rz[k, 0]), float(passes.elem_rz[k, 1])
    for p in pts:
        dr, dz = float(p[0]) - re, float(p[1]) - ze
        pd = dr * b[A.B_RD] + dz * b[A.B_AD]
        pn = dr * b[A.B_RN] + dz * b[A.B_AN]
        pm = dr * b[A.B_RM] + dz * b[A.B_AM]
        if delta > 0.0 and bw > 0.0:
            dist = math.hypot(dr, dz)
            if dist < bw:
                f = delta * (1.0 - (dist / bw) ** 2)
                pd, pn, pm = pd - f * nd, pn - f * nn, pm - f * nm
        t = -(mk + pm) / vm
        if abs(t) > tmax:
            s_out.append(sk + pd)
            w_out.append(None)
            continue
        lift = 0.5 * curv * t * t
        s_out.append(sk + pd + t * vd + lift * nd)
        w_out.append(wk + pn + t * vn + lift * nn)
    return s_out, w_out


def envelope_profile_ref(passes: A.Passes, tpl: A.Templates, u0: float, du: float, n: int, cap: int,
                         n_parab: int, recovery: float = 0.0):
    """(z, hits) of the kinematic profile on the grid u0 + i du (see kernels.envelope_profile)."""
    z = [math.inf] * n
    hits = [0] * n
    for k in range(len(passes.s)):
        ps, pw = pass_polyline_ref(passes, tpl, k, cap, n_parab)
        if len(ps) < 2:
            continue
        kind = int(passes.kind[k])
        lo_s, hi_s = -math.inf, math.inf
        if kind == A.KIND_CUT:
            valid = [(w, s) for s, w in zip(ps, pw) if w is not None]
            if not valid:
                continue
            wmin, smin = min(valid, key=lambda x: x[0]) if valid else (math.inf, 0.0)
            # the first lowest point in polyline order (ties: the kernel keeps the first)
            for s, w in zip(ps, pw):
                if w is not None and w == wmin:
                    smin = s
                    break
            lo_s, hi_s = smin - float(passes.half[k]), smin + float(passes.half[k])
        cur = {}
        for j in range(len(ps) - 1):
            sa, wa, sb, wb = ps[j], pw[j], ps[j + 1], pw[j + 1]
            if wa is None or wb is None:
                continue
            a, c = min(sa, sb), max(sa, sb)
            if c < lo_s or a > hi_s:
                continue
            a, c = max(a, lo_s), min(c, hi_s)
            ia = max(int(math.ceil((a - u0) / du)), 0)
            ic = min(int(math.floor((c - u0) / du)), n - 1)
            ds = sb - sa
            flat = abs(ds) <= 1.0e-15
            slope = 1.0e6 if flat else (wb - wa) / ds
            for i in range(ia, ic + 1):
                wv = min(wa, wb) if flat else wa + (u0 + i * du - sa) * slope
                if i not in cur or wv < cur[i][0]:
                    cur[i] = (wv, slope)
        if not cur:
            continue
        if kind != A.KIND_CUT:
            for i in sorted(cur):
                if z[i] < math.inf:
                    z[i] += cur[i][0]
            continue
        hmin = float(passes.hmin[k])
        lo, hi = min(cur), max(cur)

        def tool(i):
            return cur[i][0] if i in cur else math.inf

        def thin(i):
            """The pass ploughs sample i: below the surface by less than h_min (normal to the pass)."""
            t = tool(i)
            if not (t < math.inf and t < z[i] < math.inf):
                return False
            slope = cur[i][1]
            return (z[i] - t) / math.sqrt(1.0 + slope * slope) < hmin

        i = lo
        while i <= hi:
            t = tool(i)
            if not t < math.inf:
                i += 1
                continue
            if t < z[i]:
                if z[i] == math.inf:
                    z[i] = t
                elif not thin(i):
                    slope = cur[i][1]
                    h = (z[i] - t) / math.sqrt(1.0 + slope * slope)
                    if hmin > 0.0 and recovery > 0.0 and h < (1.0 + recovery) * hmin:
                        z[i] = z[i] + (h - hmin) / (recovery * hmin) * (t - z[i])
                    else:
                        z[i] = t
                else:
                    a, vol = i, 0.0
                    while i <= hi and thin(i):
                        vol += z[i] - tool(i)
                        z[i] = tool(i)
                        hits[i] += 1
                        i += 1
                    _place_ploughed_ref(z, tool, a, i - 1, lo, hi, vol)
                    continue
            hits[i] += 1
            i += 1
    return np.array(z), np.array(hits, dtype=np.int64)


def _place_ploughed_ref(z: list, tool, a: int, b: int, lo: int, hi: int, vol: float) -> None:
    """Twin of kernels._place_ploughed: the ploughed volume fills the free space under the pass next to the run
    (nearest first, left and right alternating); the rest stays spread over the run."""
    left, right = a - 1, b + 1

    def room(j):
        t = tool(j)
        return t - z[j] if (t < math.inf and z[j] < math.inf and t > z[j]) else None

    while vol > 0.0:
        moved = False
        if lo <= left:
            r = room(left)
            if r is not None:
                if vol < r:
                    z[left] += vol
                    vol = 0.0
                    break
                z[left] = tool(left)
                vol -= r
                left -= 1
                moved = True
        if vol > 0.0 and right <= hi:
            r = room(right)
            if r is not None:
                if vol < r:
                    z[right] += vol
                    vol = 0.0
                    break
                z[right] = tool(right)
                vol -= r
                right += 1
                moved = True
        if not moved:
            break
    if vol > 0.0:
        share = vol / (b - a + 1)
        for j in range(a, b + 1):
            z[j] += share


def pass_valleys_ref(passes: A.Passes, tpl: A.Templates, cap: int, n_parab: int, side: float = 0.0):
    out = []
    for k in range(len(passes.s)):
        if int(passes.kind[k]) != A.KIND_CUT:
            continue
        if side != 0.0:
            rd = float(passes.basis[k, A.B_RD])
            if (1.0 if rd > 0 else (-1.0 if rd < 0 else 0.0)) != side:
                continue
        ps, pw = pass_polyline_ref(passes, tpl, k, cap, n_parab)
        valid = [j for j, w in enumerate(pw) if w is not None]
        if not valid:
            continue
        best = min(valid, key=lambda j: (pw[j], j))
        if best in (valid[0], valid[-1]):
            continue
        out.append(ps[best])
    return np.array(out)


def gaussian_mean_ref(z, du: float, lam: float, alpha: float, stride: int):
    n = len(z)
    out = [0.0] * n
    half = int(lam / du) + 1
    stride = max(int(stride), 1)
    centres = list(range(0, n, stride))
    if centres[-1] != n - 1:
        centres.append(n - 1)
    for c in centres:
        acc = wsum = 0.0
        for j in range(max(c - half, 0), min(c + half, n - 1) + 1):
            x = (j - c) * du / (alpha * lam)
            wq = math.exp(-math.pi * x * x)
            acc += wq * float(z[j])
            wsum += wq
        out[c] = acc / wsum
    for a, b in zip(centres[:-1], centres[1:]):
        for j in range(a + 1, b):
            out[j] = out[a] + (j - a) / (b - a) * (out[b] - out[a])
    return np.array(out)


def rsm_ref(r, i0: int, ne: int, du: float, h_min: float, w_min: float):
    segs = []                                    # [sign, start, end, height]
    for i in range(i0, i0 + ne):
        v = float(r[i])
        sgn = 1 if v > 0 else (-1 if v < 0 else 0)
        if sgn == 0:
            continue
        if not segs or segs[-1][0] != sgn:
            segs.append([sgn, i, i + 1, abs(v)])
        else:
            segs[-1][2] = i + 1
            segs[-1][3] = max(segs[-1][3], abs(v))
    if segs and segs[0][1] == i0 and float(r[i0]) != 0.0:
        segs = segs[1:]
    if segs and segs[-1][2] == i0 + ne:
        segs = segs[:-1]
    kept = []
    for sgn, a, b, h in segs:
        if h < h_min or (b - a) * du < w_min:
            if kept:
                kept[-1][2] = b
            continue
        if kept and kept[-1][0] == sgn:
            kept[-1][2] = b
            continue
        kept.append([sgn, a, b])
    total, count, q = 0.0, 0, 0
    while q + 1 < len(kept):
        if kept[q][0] == 1 and kept[q + 1][0] == -1:
            total += (kept[q + 1][2] - kept[q][1]) * du
            count += 1
            q += 2
        else:
            q += 1
    return (total / count if count else 0.0), count


def profile_params_ref(r, du: float, i0: int, lsl: int, n_sl: int, hd: float, wd: float):
    out = [0.0] * A.N_P
    ne = lsl * n_sl
    if ne < 2 or n_sl < 1 or i0 + ne > len(r):
        return np.array(out)
    seg = [float(v) for v in r[i0:i0 + ne]]
    ra = sum(abs(v) for v in seg) / ne
    rq = math.sqrt(sum(v * v for v in seg) / ne)
    out[A.P_RA], out[A.P_RQ] = ra, rq
    out[A.P_RSK] = (sum(v ** 3 for v in seg) / ne) / rq ** 3 if rq > 0 else 0.0
    out[A.P_RKU] = (sum(v ** 4 for v in seg) / ne) / rq ** 4 if rq > 0 else 0.0
    peaks = [max(seg[s * lsl:(s + 1) * lsl]) for s in range(n_sl)]
    valleys = [-min(seg[s * lsl:(s + 1) * lsl]) for s in range(n_sl)]
    out[A.P_RP] = sum(peaks) / n_sl
    out[A.P_RV] = sum(valleys) / n_sl
    out[A.P_RZ] = (sum(peaks) + sum(valleys)) / n_sl
    out[A.P_RT] = max(seg) - min(seg)
    out[A.P_RDQ] = math.sqrt(sum(((seg[i + 1] - seg[i]) / du) ** 2 for i in range(ne - 1)) / (ne - 1))
    out[A.P_N_SL] = n_sl
    rsm, count = rsm_ref(r, i0, ne, du, hd * out[A.P_RZ], wd * lsl * du)
    out[A.P_RSM], out[A.P_N_ELEM] = rsm, count
    return np.array(out)


def silhouette_ref(r, z, a, e1, e2, tc, nn, n_phi: int, u0: float, du: float, n: int):
    out = [math.inf] * n

    def raster(sa, wa, sb, wb):
        lo, hi = min(sa, sb), max(sa, sb)
        for i in range(max(int(math.ceil((lo - u0) / du)), 0), min(int(math.floor((hi - u0) / du)), n - 1) + 1):
            if abs(sb - sa) <= 1.0e-15:
                wv = min(wa, wb)
            else:
                wv = wa + (u0 + i * du - sa) * (wb - wa) / (sb - sa)
            out[i] = min(out[i], wv)

    def dot(p, q):
        return float(p[0]) * float(q[0]) + float(p[1]) * float(q[1]) + float(p[2]) * float(q[2])

    prev = None
    for j in range(len(r)):
        cur = []
        for q in range(n_phi + 1):
            phi = 2.0 * math.pi * q / n_phi
            c, s = math.cos(phi), math.sin(phi)
            u = float(z[j]) * dot(a, tc) + float(r[j]) * (c * dot(e1, tc) + s * dot(e2, tc))
            w = float(z[j]) * dot(a, nn) + float(r[j]) * (c * dot(e1, nn) + s * dot(e2, nn))
            cur.append((u, w))
        for q in range(n_phi):
            raster(cur[q][0], cur[q][1], cur[q + 1][0], cur[q + 1][1])
        if prev is not None:
            for q in range(n_phi + 1):
                raster(prev[q][0], prev[q][1], cur[q][0], cur[q][1])
        prev = cur
    return np.array(out)


def wire_centre_ref(pz, pr, zc: float, radius: float, external: bool) -> float:
    values = []
    for z, r in zip(pz, pr):
        dz = float(z) - zc
        if abs(dz) <= radius:
            h = math.sqrt(radius * radius - dz * dz)
            values.append(float(r) + h if external else float(r) - h)
    if not values:
        return math.nan
    return max(values) if external else min(values)


__all__ = ["pass_polyline_ref", "envelope_profile_ref", "pass_valleys_ref", "gaussian_mean_ref", "profile_params_ref", "rsm_ref",
           "silhouette_ref", "wire_centre_ref"]
