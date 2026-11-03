"""Plain-Python references of the removal kernels (architecture 5.5: every kernel has one and a test compares
them). They work on the same layouts with Python lists and NumPy, by direct (slow) methods: dense sampling with
bisection where the kernels solve in closed form or by root isolation, explicit loops where the kernels keep
incremental state. Use them on small fixtures only.
"""

from __future__ import annotations

import math

import numpy as np

from . import abi as A

Q = A.QUANTUM_M
K = A.K_INLINE


# --------------------------------------------------------------------------- rays
def ray_segments(f: A.DexelField, ray: int) -> list[tuple[int, int, int, int]]:
    """(lo, hi, ev_lo, ev_hi) of a ray, inline then the pool chain."""
    n = int(f.cnt[ray])
    out = []
    for k in range(min(n, K)):
        out.append((int(f.seg[ray, k, 0]), int(f.seg[ray, k, 1]), int(f.sev[ray, k, 0]), int(f.sev[ray, k, 1])))
    b = int(f.link[ray])
    while len(out) < n and b >= 0:
        for s in range(K):
            if len(out) >= n:
                break
            out.append((int(f.pseg[b, s, 0]), int(f.pseg[b, s, 1]), int(f.psev[b, s, 0]), int(f.psev[b, s, 1])))
        b = int(f.pnext[b])
    return out


def ray_index(f, lvl, fam, iu, iv) -> int:
    v = A.TRANS_V[fam]
    return int(f.lv_ray0[lvl, fam] + iu * f.lv_dims[lvl, v] + iv)


def iv_subtract_ref(segs, removals, ev):
    """Segments [(lo, hi, elo, ehi)] minus sorted intervals [(a, b)]: (new segments, pieces, removed)."""
    out, pieces, removed = [], [], 0
    for lo, hi, elo, ehi in segs:
        cur, ce = lo, elo
        for a, b in removals:
            if b <= cur or a >= hi or b <= a:
                continue
            if a > cur:
                out.append((cur, a, ce, ev))
            p0, p1 = max(a, cur), min(b, hi)
            pieces.append((p0, p1))
            removed += p1 - p0
            if b >= hi:
                cur = hi
                break
            cur, ce = b, ev
        if cur < hi:
            out.append((cur, hi, ce, ehi))
    return out, pieces, removed


def iv_union_ref(segs, adds, ev):
    covered = []
    for lo, hi, elo, ehi in segs:
        covered.append((lo, hi, elo, ehi))
    for a, b in adds:
        if b > a:
            covered.append((a, b, ev, ev))
    covered.sort(key=lambda s: (s[0], -s[1]))
    merged = []
    for s in covered:
        if merged and s[0] <= merged[-1][1]:
            if s[1] > merged[-1][1]:
                merged[-1] = (merged[-1][0], s[1], merged[-1][2], s[3])
        else:
            merged.append(s)
    old = sum(hi - lo for lo, hi, _, _ in segs)
    new = sum(hi - lo for lo, hi, _, _ in merged)
    return merged, new - old


def family_length_ref(f, lvl, fam) -> int:
    u, v = A.TRANS_U[fam], A.TRANS_V[fam]
    total = 0
    for iu in range(int(f.lv_dims[lvl, u])):
        for iv in range(int(f.lv_dims[lvl, v])):
            total += sum(hi - lo for lo, hi, _, _ in ray_segments(f, ray_index(f, lvl, fam, iu, iv)))
    return total


def ray_nearest_ref(f, ray, q):
    segs = ray_segments(f, ray)
    if not segs:
        return 0, 0, 0
    inside = int(any(lo <= q < hi for lo, hi, _, _ in segs))
    ends = [e for lo, hi, _, _ in segs for e in (lo, hi)]
    best = min(ends, key=lambda e: (abs(q - e), e))
    return inside, best, 1


def ray_overlap_ref(f, ray, q0, q1):
    total, touched = 0, 0
    for lo, hi, _, _ in ray_segments(f, ray):
        o = min(hi, q1) - max(lo, q0)
        if o > 0:
            total += o
            touched += 1
    return total, touched


# --------------------------------------------------------------------------- envelopes
def profile_radius(prof: A.Profile, z: float) -> float | None:
    for k in range(len(prof.z0_m)):
        if prof.z0_m[k] <= z <= prof.z1_m[k]:
            if prof.kind[k] == A.PK_LINE:
                return float(prof.c0[k] + prof.c1[k] * z)
            d = prof.c2[k] ** 2 - (z - prof.c0[k]) ** 2
            return float(prof.c1[k] + prof.c3[k] * math.sqrt(max(d, 0.0)))
    return None


def inside_envelope(prof, p, w, slab_c, slab_z, slab_dr) -> bool:
    for k in range(len(slab_dr)):
        q = np.asarray(p) - slab_c[k]
        z = float(q @ w)
        if not (slab_z[k] <= z <= slab_z[k + 1]):
            continue
        R = profile_radius(prof, z)
        if R is None:
            continue
        r = float(np.linalg.norm(q - z * w))
        if r <= R + slab_dr[k]:
            return True
    return False


def envelope_ray_ref(prof, o, fam, w, slab_c, slab_z, slab_dr, t_lo, t_hi, n_samples=1500):
    """Intervals of the ray o + t e_fam inside the envelope by dense sampling and bisection to 1e-13 m."""
    e = np.zeros(3)
    e[fam] = 1.0
    o = np.asarray(o, dtype=float)

    def ins(t):
        return inside_envelope(prof, o + t * e, w, slab_c, slab_z, slab_dr)

    ts = list(np.linspace(t_lo, t_hi, n_samples + 1))
    s = float(w[fam])
    if s != 0.0:
        # the ray's crossings of every slab and sub-piece boundary plane (steps in the envelope)
        for k in range(len(slab_dr)):
            z0 = float((o - slab_c[k]) @ w)
            for zb in list(slab_z[k:k + 2]) + [float(v) for v in prof.z0_m] + [float(v) for v in prof.z1_m]:
                t = (zb - z0) / s
                for dt in (-1e-12, 1e-12):
                    if t_lo < t + dt < t_hi:
                        ts.append(t + dt)
    ts = np.array(sorted(ts))
    n_samples = len(ts) - 1
    states = [ins(t) for t in ts]
    out, start = [], (t_lo if states[0] else None)
    for k in range(n_samples):
        if states[k] == states[k + 1]:
            continue
        a, b = ts[k], ts[k + 1]
        for _ in range(60):
            m = 0.5 * (a + b)
            if ins(m) == states[k]:
                a = m
            else:
                b = m
        tb = 0.5 * (a + b)
        if states[k + 1]:
            start = tb
        else:
            out.append((start, tb))
            start = None
    if start is not None:
        out.append((start, t_hi))
    return out


def poly4_roots_ref(p, ta, tb):
    roots = np.roots([p[4], p[3], p[2], p[1], p[0]]) if p[4] != 0 else np.roots([p[3], p[2], p[1], p[0]])
    real = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 * max(1.0, abs(r)) and ta < r.real < tb)
    return real


def subtract_envelope_ref(f, prof, spec):
    """Per-ray reference of the subtraction (all levels): {ray: new segment list}, removed per (lvl, fam)."""
    w = np.asarray(spec.axis_S)
    out, removed = {}, {}
    for lvl in range(int(f.info[A.FI_NLEVELS])):
        a = float(f.lv_cell_m[lvl])
        o = f.lv_origin_m[lvl]
        for fam in range(3):
            u, v = A.TRANS_U[fam], A.TRANS_V[fam]
            lq = int((f.lv_dims[lvl, fam] - 1) * f.lv_cell_q[lvl])
            for iu in range(int(f.lv_dims[lvl, u])):
                for iv in range(int(f.lv_dims[lvl, v])):
                    p = o.copy()
                    p[u] += iu * a
                    p[v] += iv * a
                    if not (spec.box_lo[u] <= p[u] <= spec.box_hi[u] and spec.box_lo[v] <= p[v] <= spec.box_hi[v]):
                        continue
                    ray = ray_index(f, lvl, fam, iu, iv)
                    segs = ray_segments(f, ray)
                    if not segs:
                        continue
                    t_lo = max(0.0, spec.box_lo[fam] - o[fam])
                    t_hi = min(lq * Q, spec.box_hi[fam] - o[fam])
                    if t_hi <= t_lo:
                        continue
                    ivs = envelope_ray_ref(prof, p, fam, w, spec.slab_c, spec.slab_z, spec.slab_dr, t_lo, t_hi, 800)
                    rem = []
                    for t0, t1 in ivs:
                        qa, qb = max(0, math.floor(t0 / Q + 0.5)), min(lq, math.floor(t1 / Q + 0.5))
                        if qb > qa:
                            if rem and qa <= rem[-1][1]:
                                rem[-1] = (rem[-1][0], max(rem[-1][1], qb))
                            else:
                                rem.append((qa, qb))
                    new, pieces, rm = iv_subtract_ref(segs, rem, int(spec.ivals[A.SI_EVENT_ID]))
                    out[ray] = [(lo, hi) for lo, hi, _, _ in new]
                    removed[(lvl, fam)] = removed.get((lvl, fam), 0) + rm
    return out, removed


# --------------------------------------------------------------------------- boundary between rays
def dexel_sd_ref(f, lvl, fam, p):
    u, v = A.TRANS_U[fam], A.TRANS_V[fam]
    a = float(f.lv_cell_m[lvl])
    o = f.lv_origin_m[lvl]
    fu, fv = (p[u] - o[u]) / a, (p[v] - o[v]) / a
    iu = min(max(int(math.floor(fu)), 0), int(f.lv_dims[lvl, u]) - 2)
    iv = min(max(int(math.floor(fv)), 0), int(f.lv_dims[lvl, v]) - 2)
    al, be = min(max(fu - iu, 0.0), 1.0), min(max(fv - iv, 0.0), 1.0)
    q = (p[fam] - o[fam]) / Q
    far = 2.5 * float(f.lv_cell_q[lvl])
    D, ends, valid = 0.0, [], True
    for du in (0, 1):
        for dv in (0, 1):
            ray = ray_index(f, lvl, fam, iu + du, iv + dv)
            wgt = (al if du else 1 - al) * (be if dv else 1 - be)
            ins, e, has = ray_nearest_ref(f, ray, q)
            if not has:
                valid = False
                D -= wgt * far
                continue
            d = abs(q - e)
            ends.append(e)
            valid = valid and d <= far
            D += wgt * (d if ins else -d)
    spread = (max(ends) - min(ends)) if ends else far
    valid = valid and spread <= far and len(ends) == 4
    return D * Q, spread * Q, valid


def classify_ref(f, lvl, p):
    best = None
    for fam in range(3):
        d, sp, ok = dexel_sd_ref(f, lvl, fam, p)
        if ok and (best is None or sp < best[1]):
            best = (d, sp)
    if best is None:
        ds = [dexel_sd_ref(f, lvl, fam, p) for fam in range(3)]
        best = min(ds, key=lambda t: t[1])[:2]
    return best[0] > 0.0


def engaged_ref(f, c, x0, y0, r, n=3600):
    """Engaged angle intervals of a circle by dense sampling of the classification and bisection."""
    def ins(phi):
        return classify_ref(f, 0, c + r * (math.cos(phi) * x0 + math.sin(phi) * y0))

    phis = np.linspace(0.0, 2 * math.pi, n + 1)
    states = [ins(p) for p in phis[:-1]]
    states.append(states[0])
    bounds = []
    for k in range(n):
        if states[k] != states[k + 1]:
            a, b = phis[k], phis[k + 1]
            for _ in range(50):
                m = 0.5 * (a + b)
                if ins(m) == states[k]:
                    a = m
                else:
                    b = m
            bounds.append((0.5 * (a + b), states[k + 1]))
    if not bounds:
        return [(0.0, 2 * math.pi)] if states[0] else []
    first = next(i for i, (_, e) in enumerate(bounds) if e)
    out = []
    order = bounds[first:] + bounds[:first]
    for i in range(0, len(order), 2):
        lo = order[i][0]
        hi = order[(i + 1) % len(order)][0]
        if hi <= lo:
            hi += 2 * math.pi
        out.append((lo, hi))
    return out


def fill_fraction_ref(f, origin, cell, shape):
    out = np.zeros(shape)
    a = float(f.lv_cell_m[0])
    o = f.lv_origin_m[0]
    for i in range(int(f.lv_dims[0, 0])):
        for j in range(int(f.lv_dims[0, 1])):
            x, y = o[0] + i * a, o[1] + j * a
            for lo, hi, _, _ in ray_segments(f, ray_index(f, 0, 2, i, j)):
                z0, z1 = o[2] + lo * Q, o[2] + hi * Q
                for cx in range(shape[0]):
                    xa = origin[0] + cx * cell
                    lx = min(x + a / 2, xa + cell) - max(x - a / 2, xa)
                    if lx <= 0:
                        continue
                    for cy in range(shape[1]):
                        ya = origin[1] + cy * cell
                        ly = min(y + a / 2, ya + cell) - max(y - a / 2, ya)
                        if ly <= 0:
                            continue
                        for cz in range(shape[2]):
                            za = origin[2] + cz * cell
                            lz = min(z1, za + cell) - max(z0, za)
                            if lz > 0:
                                out[cx, cy, cz] += lx * ly * lz / cell ** 3
    return out


def ray_cast_ref(f, o, d, t_max, mode, step=None):
    a = float(f.lv_cell_m[0])
    step = step or a / 8
    o = np.asarray(o, dtype=float)
    d = np.asarray(d, dtype=float)
    want = mode == 0
    if classify_ref(f, 0, o) == want:
        return 0.0
    t = 0.0
    while t < t_max:
        tn = min(t + step, t_max)
        if classify_ref(f, 0, o + tn * d) == want:
            lo, hi = t, tn
            for _ in range(60):
                m = 0.5 * (lo + hi)
                if classify_ref(f, 0, o + m * d) == want:
                    hi = m
                else:
                    lo = m
            return 0.5 * (lo + hi)
        t = tn
    return None


def broad_phase_ref(f, prof, tip, w, margin):
    zmin, zmax, R = prof.info[A.PI_ZMIN], prof.info[A.PI_ZMAX], prof.info[A.PI_RMAX] + margin
    lo, hi = np.zeros(3), np.zeros(3)
    for i in range(3):
        pa, pb = tip[i] + zmin * w[i], tip[i] + zmax * w[i]
        e = R * math.sqrt(max(0.0, 1 - w[i] ** 2))
        lo[i], hi[i] = min(pa, pb) - e - margin, max(pa, pb) + e + margin
    a = float(f.lv_cell_m[0])
    o = f.lv_origin_m[0]
    bd = f.lv_bdims[0]
    k0 = [max(0, (int(math.floor((lo[i] - o[i]) / a)) - 1) // A.BRICK) for i in range(3)]
    k1 = [min(int(bd[i]) - 1, (int(math.ceil((hi[i] - o[i]) / a)) + 1) // A.BRICK) for i in range(3)]
    count = 0
    for bx in range(k0[0], k1[0] + 1):
        for by in range(k0[1], k1[1] + 1):
            for bz in range(k0[2], k1[2] + 1):
                idx = int(f.lv_brick0[0] + (bx * bd[1] + by) * bd[2] + bz)
                if np.any(f.bmat[idx] > 0):
                    count += 1
    return count


def excess_material_ref(f, tgt, tol):
    a = float(f.lv_cell_m[0])
    vol, best, rays = 0.0, 0.0, 0
    for i in range(int(f.lv_dims[0, 0])):
        for j in range(int(f.lv_dims[0, 1])):
            ray = ray_index(f, 0, 2, i, j)
            tsegs = [(int(tgt.seg[k, 0]), int(tgt.seg[k, 1]))
                     for k in range(int(tgt.start[ray]), int(tgt.start[ray + 1]))]
            ray_ex = False
            for lo, hi, _, _ in ray_segments(f, ray):
                rest, _, _ = iv_subtract_ref([(lo, hi, 0, 0)], tsegs, 0)
                for x0, x1, _, _ in rest:
                    ex = (x1 - x0) * Q
                    vol += ex * a * a
                    best = max(best, ex)
                    ray_ex = ray_ex or ex > tol
            rays += int(ray_ex)
    return vol, best, rays


# --------------------------------------------------------------------------- connectivity
def components_ref(f):
    """Components of the level-0 z segments (4-neighbour z overlap): list of (volume, centroid)."""
    nx, ny = int(f.lv_dims[0, 0]), int(f.lv_dims[0, 1])
    segs = {}
    nodes = []
    for i in range(nx):
        for j in range(ny):
            for lo, hi, _, _ in ray_segments(f, ray_index(f, 0, 2, i, j)):
                segs.setdefault((i, j), []).append(len(nodes))
                nodes.append((i, j, lo, hi))
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (i, j), ids in segs.items():
        for di, dj in ((1, 0), (0, 1)):
            for a_ in ids:
                for b_ in segs.get((i + di, j + dj), []):
                    _, _, la, ha = nodes[a_]
                    _, _, lb, hb = nodes[b_]
                    if la < hb and lb < ha:
                        ra, rb = find(a_), find(b_)
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
    a = float(f.lv_cell_m[0])
    o = f.lv_origin_m[0]
    comps = {}
    for k, (i, j, lo, hi) in enumerate(nodes):
        r = find(k)
        v = (hi - lo) * Q * a * a
        c = np.array([o[0] + i * a, o[1] + j * a, o[2] + 0.5 * (lo + hi) * Q])
        vol, mom = comps.get(r, (0.0, np.zeros(3)))
        comps[r] = (vol + v, mom + v * c)
    return sorted(((v, m / v) for v, m in comps.values()), key=lambda t: -t[0])


# --------------------------------------------------------------------------- polar
def polar_radial(pf, b, i):
    kp = int(pf.info[A.PI_KP])
    nz = int(pf.geom[A.PG_NZ])
    ray = b * nz + i
    return [(int(pf.rseg[ray, k, 0]), int(pf.rseg[ray, k, 1])) for k in range(int(pf.rcnt[ray]))][:kp]


def polar_cut_ref(pf, ins, ps):
    """Rows cut by one step: {(b, i): new radial segments} and the removed q^2 sum."""
    nth, dth = int(pf.geom[A.PG_NTHETA]), float(pf.geom[A.PG_DTHETA])
    z0f, dz, nz = float(pf.geom[A.PG_Z0]), float(pf.geom[A.PG_DZ]), int(pf.geom[A.PG_NZ])
    th0, th1 = ps[A.PS_TH0], ps[A.PS_TH1]
    a_lo, a_hi = min(th0, th1), max(th0, th1)
    out, q2 = {}, 0
    k = math.floor(a_lo / dth - 0.5) + 1
    while (k + 0.5) * dth <= a_hi:
        cb = (k + 0.5) * dth
        lam = (cb - th0) / (th1 - th0)
        zt = ps[A.PS_Z0] + lam * (ps[A.PS_Z1] - ps[A.PS_Z0])
        rt = ps[A.PS_R0] + lam * (ps[A.PS_R1] - ps[A.PS_R0])
        b = k % nth
        for i in range(nz):
            zi = z0f + (i + 0.5) * dz
            ztl = (zi - zt) * ps[A.PS_SZ]
            lo, hi = _x_range_ref(ins, ztl)
            if lo > hi:
                continue
            if ps[A.PS_SX] < 0:
                ra, rb = rt - hi, rt - lo
            else:
                ra, rb = rt + lo, rt + hi
            if rb <= 0:
                continue
            ra = max(ra, 0.0)
            yo = ps[A.PS_YOFF]
            ra, rb = math.hypot(ra, yo), math.hypot(rb, yo)
            qa, qb = math.floor(ra / Q + 0.5), math.floor(rb / Q + 0.5)
            segs = out.get((b, i), polar_radial(pf, b, i))
            new, pieces, _ = iv_subtract_ref([(lo_, hi_, 0, 0) for lo_, hi_ in segs], [(qa, qb)], 0)
            q2 += sum(p1 * p1 - p0 * p0 for p0, p1 in pieces)
            out[(b, i)] = [(x, y) for x, y, _, _ in new]
        k += 1
    return out, q2


def _chain_ref(zs, xs, z, upper):
    if len(zs) == 0 or z < zs[0] or z > zs[-1]:
        return -1e300 if upper else 1e300
    return float(np.interp(z, zs, xs))


def _x_range_ref(ins, zt):
    hi = _chain_ref(ins.zu, ins.xu, zt, True)
    lo = _chain_ref(ins.zl, ins.xl, zt, False)
    rho = float(ins.nose[2])
    if rho > 0:
        d = zt - ins.nose[0]
        if rho * rho - d * d >= 0:
            h = math.sqrt(rho * rho - d * d)
            hi = max(hi, ins.nose[1] + h)
            lo = min(lo, ins.nose[1] - h)
    return lo, hi


def polar_volume_ref(pf):
    total = 0
    for ray in range(pf.rcnt.shape[0]):
        for k in range(int(pf.rcnt[ray])):
            a, b = int(pf.rseg[ray, k, 0]), int(pf.rseg[ray, k, 1])
            total += b * b - a * a
    return total


# --------------------------------------------------------------------------- taps
def helix_inside_ref(hx, x, th_a, th_b, pa, pb, w, x0, y0):
    hand = float(hx.info[0])
    d = np.asarray(x) - pa
    alpha = math.atan2(d @ y0, d @ x0)
    lo_t, hi_t = min(th_a, th_b), max(th_a, th_b)
    best = math.inf
    for land in range(len(hx.land_phase)):
        base = alpha - hand * hx.land_phase[land]
        ts = base + math.ceil((lo_t - base) / (2 * math.pi)) * 2 * math.pi
        while ts <= hi_t:
            lam = (ts - th_a) / (th_b - th_a)
            ts += 2 * math.pi
            p = pa + lam * (pb - pa)
            q = np.asarray(x) - p
            zt = float(q @ w)
            rr = float(np.linalg.norm(q - zt * w))
            for j in range(int(hx.land_first[land]), int(hx.land_first[land + 1])):
                pts = hx.poly[j, :int(hx.n[j])]
                dmax = -math.inf
                for e in range(len(pts)):
                    a_, b_ = pts[e], pts[(e + 1) % len(pts)]
                    t = b_ - a_
                    ln = float(np.hypot(*t))
                    if ln > 0:
                        dmax = max(dmax, ((zt - a_[0]) * t[1] - (rr - a_[1]) * t[0]) / ln)
                best = min(best, dmax)
    return best


def remap_ref(old, new_dims, perm, sign):
    """{(fam_new, iu, iv): segments} of level 0 after the lattice symmetry (perm, sign)."""
    out = {}
    lq_new = {f: int((new_dims[f] - 1) * old.lv_cell_q[0]) for f in range(3)}
    for fam_n in range(3):
        u, v = A.TRANS_U[fam_n], A.TRANS_V[fam_n]
        for iu in range(int(new_dims[u])):
            for iv in range(int(new_dims[v])):
                idx_n = [0, 0, 0]
                idx_n[u], idx_n[v] = iu, iv
                idx_o = [0, 0, 0]
                for i in range(3):
                    if i == fam_n:
                        continue
                    idx_o[perm[i]] = idx_n[i] if sign[i] > 0 else int(new_dims[i]) - 1 - idx_n[i]
                fam_o = perm[fam_n]
                uo, vo = A.TRANS_U[fam_o], A.TRANS_V[fam_o]
                segs = [(lo, hi) for lo, hi, _, _ in ray_segments(old, ray_index(old, 0, fam_o, idx_o[uo], idx_o[vo]))]
                if sign[fam_n] < 0:
                    segs = [(lq_new[fam_n] - hi, lq_new[fam_n] - lo) for lo, hi in reversed(segs)]
                if segs:
                    out[(fam_n, iu, iv)] = segs
    return out
