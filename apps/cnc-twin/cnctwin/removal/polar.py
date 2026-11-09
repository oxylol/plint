"""Polar field construction and conversion (docs/models/removal.md "Polar field").

* :func:`resolution` - theta bins and row height from the rules of architecture 5.4: at least
  ``physics.polar_theta_bins`` bins, dz <= min(f_min / 8, r_nose / 16), covering the turned length; when the
  arrays would exceed ``max_bytes`` the rows are coarsened to fit and the returned note says so (the domain
  raises a warning finding with it).
* :func:`polar_tube` - a cylinder / tube stock directly; :func:`polar_from_field` - the polar field of an
  already machined tri-dexel field (mill-turn), sampled on the base lattice's z planes;
* :func:`insert_region` - tooling's InsertProfile as the chains the cut kernel reads;
* :func:`grow_kp` - more inline segments per ray after a PI_OVERFLOW;
* :func:`polar_from_mesh` - a mesh stock on a lathe (exact ray-triangle crossings, the tri-dexel tie rule);
* :func:`volume_m3` - the exact integer volume of the radial family (:func:`q2_sum`, unbounded integer).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.errors import InputError
from . import abi as A


SIZING_KP = 2      # inline segments per ray the memory budget of the rows is reckoned with


def resolution(length_m: float, theta_bins: int, f_min_m_per_rev: float, r_nose_m: float, r_max_m: float,
               kp: int = SIZING_KP, max_bytes: float = 150e6, dr_m: float | None = None, lattice_m: float = 0.25e-3):
    """(n_theta, dz, nz, dr, nr, note). The removal domain sizes the rows with SIZING_KP, not with the configured
    initial capacity physics.removal.polar_kp: a capacity is grown on demand and must not change the field's
    resolution (with polar_kp 16 the budget coarsened the rows of a d45 bar from 6.25 um to 75 um, so the pass's
    forces differed from those with polar_kp 1 from the first contact; docs/models/loop.md 15)."""
    n_theta = int(max(8, theta_bins))
    limits = []
    if f_min_m_per_rev > 0.0:
        limits.append(f_min_m_per_rev / 8.0)
    if r_nose_m > 0.0:
        limits.append(r_nose_m / 16.0)
    dz = min(limits) if limits else lattice_m / 8.0
    dz = max(round(dz / A.QUANTUM_M), 1) * A.QUANTUM_M
    nz = int(math.ceil(length_m / dz - 1e-9))
    dr = float(dr_m) if dr_m else max(dz, lattice_m / 2.0)
    nr = int(math.ceil(r_max_m / dr - 1e-9)) + 1
    per_ray = 1 + 2 * kp * 2 * 4
    need = n_theta * (nz + nr) * per_ray
    note = ""
    if need > max_bytes:
        scale = need / max_bytes
        dz_new = dz * scale * (nz / (nz + nr))
        dz_new = max(round(dz_new / A.QUANTUM_M), 1) * A.QUANTUM_M
        note = (f"polar rows coarsened from dz = {dz * 1e6:.2f} um to {dz_new * 1e6:.2f} um to keep the polar field "
                f"within {max_bytes / 1e6:.0f} MB (rule: dz <= min(f_min/8, r_nose/16))")
        dz = dz_new
        nz = int(math.ceil(length_m / dz - 1e-9))
    return n_theta, dz, nz, dr, nr, note


def make_polar(n_theta: int, z0_m: float, dz_m: float, nz: int, dr_m: float, nr: int, kp: int = 2,
               kpa: int | None = None) -> A.PolarField:
    geom = np.zeros(A.N_PG)
    geom[A.PG_NTHETA] = n_theta
    geom[A.PG_DTHETA] = 2.0 * math.pi / n_theta
    geom[A.PG_Z0] = z0_m
    geom[A.PG_DZ] = dz_m
    geom[A.PG_NZ] = nz
    geom[A.PG_R0] = 0.5 * dr_m
    geom[A.PG_DR] = dr_m
    geom[A.PG_NR] = nr
    geom[A.PG_QUANTUM] = A.QUANTUM_M
    kpa = int(kp if kpa is None else kpa)
    info = np.zeros(A.N_PINFO, np.int64)
    info[A.PI_KP] = kp
    info[A.PI_KPA] = kpa
    info[A.PI_DIRTY_Z0] = -1
    info[A.PI_DIRTY_Z1] = -1
    info[A.PI_DIRTY_R0] = -1
    info[A.PI_DIRTY_R1] = -1
    for k in (A.PI_TOUCH_Q0, A.PI_TOUCH_Q1, A.PI_OPTOUCH_Q0, A.PI_OPTOUCH_Q1):
        info[k] = -1
    nrad = n_theta * nz
    nax = n_theta * nr
    return A.PolarField(geom, np.zeros(nrad, np.int8), np.zeros((nrad, kp, 2), np.int32),
                        np.zeros((nrad, kp, 2), np.int32),
                        np.zeros(nax, np.int8), np.zeros((nax, kpa, 2), np.int32), np.zeros((nax, kpa, 2), np.int32),
                        np.zeros(n_theta * ((nz + A.POLAR_BLOCK - 1) // A.POLAR_BLOCK), np.int64), info)


def polar_tube(r_out_m: float, r_in_m: float, z_s0_m: float, z_s1_m: float, n_theta: int, dz_m: float,
               dr_m: float, z0_m: float | None = None, z1_m: float | None = None, kp: int = 2,
               ev: int = A.SEV_STOCK, kpa: int | None = None) -> A.PolarField:
    """Polar field over [z0, z1] (default the stock length) holding the tube r_in <= r < r_out, z_s0 <= z < z_s1."""
    from .k_polar import refresh_zmax

    z0 = z_s0_m if z0_m is None else z0_m
    z1 = z_s1_m if z1_m is None else z1_m
    nz = int(math.ceil((z1 - z0) / dz_m - 1e-9))
    nr = int(math.ceil(r_out_m / dr_m - 1e-9)) + 1
    pf = make_polar(n_theta, z0, dz_m, nz, dr_m, nr, kp, kpa)
    kpa_ = int(pf.info[A.PI_KPA])
    q_in = int(math.floor(r_in_m / A.QUANTUM_M + 0.5))
    q_out = int(math.floor(r_out_m / A.QUANTUM_M + 0.5))
    zc = z0 + (np.arange(nz) + 0.5) * dz_m
    rows = (zc >= z_s0_m) & (zc < z_s1_m)
    rad = pf.rcnt.reshape(n_theta, nz)
    rad[:, rows] = 1
    rseg = pf.rseg.reshape(n_theta, nz, kp, 2)
    rseg[:, rows, 0, 0] = q_in
    rseg[:, rows, 0, 1] = q_out
    rsev = pf.rsev.reshape(n_theta, nz, kp, 2)
    rsev[:, rows, 0, :] = ev
    rj = (np.arange(nr) + 0.5) * dr_m
    cols = (rj >= r_in_m) & (rj < r_out_m)
    qa = int(math.floor((z_s0_m - z0) / A.QUANTUM_M + 0.5))
    qb = int(math.floor((z_s1_m - z0) / A.QUANTUM_M + 0.5))
    ax = pf.acnt.reshape(n_theta, nr)
    ax[:, cols] = 1
    aseg = pf.aseg.reshape(n_theta, nr, kpa_, 2)
    aseg[:, cols, 0, 0] = qa
    aseg[:, cols, 0, 1] = qb
    asev = pf.asev.reshape(n_theta, nr, kpa_, 2)
    asev[:, cols, 0, :] = ev
    refresh_zmax(pf, 0)
    return pf


def polar_from_field(field: A.DexelField, n_theta: int, dz_m: float, dr_m: float, z0_m: float, z1_m: float,
                     r_max_m: float, kp: int = 2) -> A.PolarField:
    """The polar field of a tri-dexel field (material already machined: mill-turn). Each radial ray is
    sampled on the nearest base-lattice z plane with ray casts along the radius (interpolated boundary,
    k_query.ray_cast), each axial ray by a ray cast along z at its radius; KP grows when a ray needs it."""
    from .k_polar import refresh_zmax
    from .k_query import ray_cast

    nz = int(math.ceil((z1_m - z0_m) / dz_m - 1e-9))
    nr = int(math.ceil(r_max_m / dr_m - 1e-9)) + 1
    pf = make_polar(n_theta, z0_m, dz_m, nz, dr_m, nr, kp)
    a = float(field.lv_cell_m[0])
    oz = float(field.lv_origin_m[0, 2])
    nzl = int(field.lv_dims[0, 2])
    hit = np.zeros(8)
    dth = 2.0 * math.pi / n_theta

    def segments_along(ox, oy, oz_, dx, dy, dz_, length):
        segs = []
        t = 0.0
        inside = False
        start = 0.0
        while t < length:
            mode = 1 if inside else 0
            if ray_cast(field, ox + t * dx, oy + t * dy, oz_ + t * dz_, dx, dy, dz_, length - t, mode, 1e-8, hit) == 0:
                break
            tt = t + hit[0]
            if not inside:
                start = tt
                inside = True
            else:
                segs.append((start, tt))
                inside = False
            t = tt + 1e-9
        if inside:
            segs.append((start, length))
        return segs

    plane_cache: dict = {}
    for b in range(n_theta):
        th = (b + 0.5) * dth
        c, s = math.cos(th), math.sin(th)
        for i in range(nz):
            z = z0_m + (i + 0.5) * dz_m
            k = int(round((z - oz) / a))
            k = min(max(k, 0), nzl - 1)
            key = (b, k)
            if key not in plane_cache:
                zp = oz + k * a
                plane_cache[key] = segments_along(0.0, 0.0, zp, c, s, 0.0, r_max_m)
            segs = plane_cache[key]
            if len(segs) > pf.info[A.PI_KP]:
                pf = grow_kp(pf, len(segs))
            ray = b * nz + i
            pf.rcnt[ray] = len(segs)
            for q, (r0, r1) in enumerate(segs):
                pf.rseg[ray, q, 0] = int(math.floor(r0 / A.QUANTUM_M + 0.5))
                pf.rseg[ray, q, 1] = int(math.floor(r1 / A.QUANTUM_M + 0.5))
                pf.rsev[ray, q, :] = A.SEV_STOCK
        for j in range(nr):
            rj = (j + 0.5) * dr_m
            segs = segments_along(rj * c, rj * s, z0_m, 0.0, 0.0, 1.0, z1_m - z0_m)
            if len(segs) > pf.info[A.PI_KPA]:
                pf = grow_kp(pf, kpa_new=len(segs))
            ray = b * nr + j
            pf.acnt[ray] = len(segs)
            for q, (za, zb) in enumerate(segs):
                pf.aseg[ray, q, 0] = int(math.floor(za / A.QUANTUM_M + 0.5))
                pf.aseg[ray, q, 1] = int(math.floor(zb / A.QUANTUM_M + 0.5))
                pf.asev[ray, q, :] = A.SEV_STOCK
    refresh_zmax(pf, 0)
    return pf


# --------------------------------------------------------------------------- polar field of a mesh stock
def _canon(pu, pv, qu, qv, xu, xv):
    """k_stock._canon_edge on arrays: the edge function of p -> q at x evaluated in canonical vertex order, so an
    edge shared by two triangles gives exactly opposite values in both."""
    keep = (pu < qu) | ((pu == qu) & (pv < qv))
    return np.where(keep, (qu - pu) * (xv - pv) - (qv - pv) * (xu - pu),
                    -((pu - qu) * (xv - qv) - (pv - qv) * (xu - qu)))


def _covers(e, du, dv):
    """k_stock._covers: inside, or on an edge the top-left rule gives to this triangle."""
    return (e > 0.0) | ((e == 0.0) & ((dv > 0.0) | ((dv == 0.0) & (du < 0.0))))


def _ccw(au, av, bu, bv, cu, cv, fa, fb, fc):
    """(area, direction, vertices counter-clockwise): direction +1 where the ray enters the material (outward normal
    against the ray), -1 where it leaves; the 2-D frame is right-handed with the ray (k_stock.rasterize, sign +1)."""
    area = _canon(au, av, bu, bv, cu, cv)
    direction = np.where(area > 0.0, -1, 1).astype(np.int64)
    flip = area < 0.0
    bu2, bv2, fb2 = np.where(flip, cu, bu), np.where(flip, cv, bv), np.where(flip, fc, fb)
    cu2, cv2, fc2 = np.where(flip, bu, cu), np.where(flip, bv, cv), np.where(flip, fb, fc)
    return np.abs(area), direction, (au, av, bu2, bv2, cu2, cv2, fa, fb2, fc2)


def _hits(tri, xu, xv):
    """Ray hits at the 2-D points (xu, xv) of the (counter-clockwise) triangles: mask and the ray coordinate."""
    au, av, bu, bv, cu, cv, fa, fb, fc, area = tri
    e0 = _canon(bu, bv, cu, cv, xu, xv)
    e1 = _canon(cu, cv, au, av, xu, xv)
    e2 = _canon(au, av, bu, bv, xu, xv)
    hit = _covers(e0, cu - bu, cv - bv) & _covers(e1, au - cu, av - cv) & _covers(e2, bu - au, bv - av)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (e0 * fa + e1 * fb + e2 * fc) / area
    return hit, f


def _expand(lo, hi):
    """(pair index, value) for every integer lo..hi of every pair."""
    n = np.maximum(hi - lo + 1, 0)
    rep = np.repeat(np.arange(len(n)), n)
    first = np.cumsum(n) - n
    return rep, lo[rep] + (np.arange(int(n.sum())) - first[rep])


def _sweep(ray, f, d):
    """(ray, lo, hi) material intervals of the hits: k_stock._ray_intervals' winding sweep (entering before leaving
    at equal f, material where the winding is positive). Raises when a ray's winding does not return to zero."""
    if len(ray) == 0:
        return ray, f, f
    order = np.lexsort((-d, f, ray))
    ray, f, d = ray[order], f[order], d[order]
    cs = np.cumsum(d)
    first = np.r_[True, ray[1:] != ray[:-1]]
    last = np.r_[ray[1:] != ray[:-1], True]
    w = cs - (cs - d)[first][np.cumsum(first) - 1]
    if np.any(w[last] != 0):
        raise InputError(f"mesh stock is not closed: {int(np.count_nonzero(w[last]))} polar rays cross it an odd "
                         "number of times")
    s = (d > 0) & (w == 1)
    e = (d < 0) & (w == 0)
    return ray[s], f[s], f[e]


def _quantised(ray, lo, hi, q_max: int):
    """Intervals in quanta per ray, clipped to [0, q_max], empty ones dropped and touching ones united (the
    k_stock rule); sorted by ray and start. Returns (ray, q0, q1, rank within the ray)."""
    q0 = np.clip(np.floor(lo / A.QUANTUM_M + 0.5), 0, q_max).astype(np.int64)
    q1 = np.clip(np.floor(hi / A.QUANTUM_M + 0.5), 0, q_max).astype(np.int64)
    keep = q1 > q0
    ray, q0, q1 = ray[keep], q0[keep], q1[keep]
    if len(ray) == 0:
        return ray, q0, q1, q0
    order = np.lexsort((q0, ray))
    ray, q0, q1 = ray[order], q0[order], q1[order]
    big = np.int64(1) << 36
    run_end = np.maximum.accumulate(ray * big + q1) - ray * big
    first = np.r_[True, ray[1:] != ray[:-1]]
    new = first | (q0 > np.r_[0, run_end[:-1]])
    idx = np.nonzero(new)[0]
    end_idx = np.r_[idx[1:] - 1, len(ray) - 1]
    ray, q0, q1 = ray[idx], q0[idx], run_end[end_idx]
    first = np.r_[True, ray[1:] != ray[:-1]]
    start = np.nonzero(first)[0]
    rank = np.arange(len(ray)) - start[np.cumsum(first) - 1]
    return ray, q0, q1, rank


def polar_from_mesh(vertices_S_m, faces, n_theta: int, dz_m: float, dr_m: float, z0_m: float, z1_m: float,
                    r_max_m: float, kp: int = 2, ev: int = A.SEV_STOCK, lines_per_block: int = 16) -> A.PolarField:
    """The polar field of a closed triangle mesh in S (bar axis on S z): the exact crossings of every radial ray
    (bin centre angle, row centre z) and every axial ray (bin centre angle, ray radius) with the triangles. Each
    line through the axis carries the radial rays of bins b and b + n/2 (n even), so the axis needs no special case;
    a ray through a shared edge or vertex counts once by the tri-dexel rasteriser's tie rule (k_stock._canon_edge,
    _covers) and the hits become material by its winding sweep, so the polar rows and the tri-dexel stock of the
    same mesh (stock.mesh_field) describe one solid. Rows hold the mesh at their own z (a shoulder or a centre hole
    sits where the mesh has it, not on the nearest lattice plane as polar_from_field samples). Pure numpy: done once
    per stock, in blocks of lines_per_block lines."""
    from .k_polar import refresh_zmax
    from .stock import oriented_mesh

    V, Fc, _ = oriented_mesh(vertices_S_m, faces)
    nz = int(math.ceil((z1_m - z0_m) / dz_m - 1e-9))
    nr = int(math.ceil(r_max_m / dr_m - 1e-9)) + 1
    pf = make_polar(n_theta, z0_m, dz_m, nz, dr_m, nr, kp)
    sides = 2 if n_theta % 2 == 0 else 1
    n_line = n_theta // sides
    dth = 2.0 * math.pi / n_theta
    P = V[Fc]                                              # [ntri, 3 vertices, xyz]
    x, y, z = P[:, :, 0], P[:, :, 1], P[:, :, 2]
    zlo, zhi = z.min(axis=1), z.max(axis=1)
    q_r = int(math.floor(r_max_m / A.QUANTUM_M + 0.5))
    q_z = int(math.floor((z1_m - z0_m) / A.QUANTUM_M + 0.5))
    for b0 in range(0, n_line, max(1, int(lines_per_block))):
        lines = np.arange(b0, min(n_line, b0 + max(1, int(lines_per_block))))
        th = (lines + 0.5) * dth
        c, s = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
        t_all = x[None] * c + y[None] * s                  # along the line through the axis
        w_all = -x[None] * s + y[None] * c                 # across it
        li, ti = np.nonzero((w_all.min(axis=2) <= 0.0) & (w_all.max(axis=2) >= 0.0))
        tv, wv = t_all[li, ti], w_all[li, ti]
        zv = z[ti]
        # radial rays: 2-D frame (across, z), ray coordinate t; one row per z inside the triangle's z range
        area, d, (au, av, bu, bv, cu, cv, fa, fb, fc) = _ccw(wv[:, 0], zv[:, 0], wv[:, 1], zv[:, 1], wv[:, 2], zv[:, 2],
                                                             tv[:, 0], tv[:, 1], tv[:, 2])
        ok = area != 0.0
        i_lo = np.maximum(np.ceil((zlo[ti] - z0_m) / dz_m - 0.5).astype(np.int64) - 1, 0)
        i_hi = np.minimum(np.floor((zhi[ti] - z0_m) / dz_m - 0.5).astype(np.int64) + 1, nz - 1)
        i_hi = np.where(ok, i_hi, i_lo - 1)
        k, row = _expand(i_lo, i_hi)
        zr = z0_m + (row + 0.5) * dz_m
        hit, f = _hits(tuple(a[k] for a in (au, av, bu, bv, cu, cv, fa, fb, fc, area)), np.zeros_like(zr), zr)
        lid = lines[li[k]]
        ray, lo, hi = _sweep((lid * nz + row)[hit], f[hit], d[k][hit])
        line, row = ray // nz, ray % nz
        rays, los, his = [line * nz + row], [np.maximum(lo, 0.0)], [hi]
        if sides == 2:
            rays.append((line + n_line) * nz + row)
            los.append(np.maximum(-hi, 0.0))
            his.append(-lo)
        rr, q0, q1, rank = _quantised(np.concatenate(rays), np.concatenate(los), np.concatenate(his), q_r)
        if len(rank) and int(rank.max()) + 1 > int(pf.info[A.PI_KP]):
            pf = grow_kp(pf, kp_new=int(rank.max()) + 1)
        np.add.at(pf.rcnt, rr, 1)
        pf.rseg[rr, rank, 0] = q0
        pf.rseg[rr, rank, 1] = q1
        pf.rsev[rr, rank, :] = ev
        # axial rays: 2-D frame (along, across), ray coordinate z; rays at +-(j + 1/2) dr on the line
        area, d, (au, av, bu, bv, cu, cv, fa, fb, fc) = _ccw(tv[:, 0], wv[:, 0], tv[:, 1], wv[:, 1], tv[:, 2], wv[:, 2],
                                                             zv[:, 0], zv[:, 1], zv[:, 2])
        ok = area != 0.0
        tlo, thi = tv.min(axis=1), tv.max(axis=1)
        tri = (au, av, bu, bv, cu, cv, fa, fb, fc, area)
        rays, los, his = [], [], []
        for side in range(sides):
            sgn = 1.0 if side == 0 else -1.0
            a_lo, a_hi = (tlo, thi) if side == 0 else (-thi, -tlo)
            j_lo = np.maximum(np.ceil(a_lo / dr_m - 0.5).astype(np.int64) - 1, 0)
            j_hi = np.minimum(np.floor(a_hi / dr_m - 0.5).astype(np.int64) + 1, nr - 1)
            j_hi = np.where(ok, j_hi, j_lo - 1)
            k, j = _expand(j_lo, j_hi)
            tr = sgn * (j + 0.5) * dr_m
            hit, f = _hits(tuple(a[k] for a in tri), tr, np.zeros_like(tr))
            bins = lines[li[k]] + side * n_line
            ray, lo, hi = _sweep((bins * nr + j)[hit], f[hit] - z0_m, d[k][hit])
            rays.append(ray)
            los.append(lo)
            his.append(hi)
        rr, q0, q1, rank = _quantised(np.concatenate(rays), np.concatenate(los), np.concatenate(his), q_z)
        if len(rank) and int(rank.max()) + 1 > int(pf.info[A.PI_KPA]):
            pf = grow_kp(pf, kpa_new=int(rank.max()) + 1)
        np.add.at(pf.acnt, rr, 1)
        pf.aseg[rr, rank, 0] = q0
        pf.aseg[rr, rank, 1] = q1
        pf.asev[rr, rank, :] = ev
    refresh_zmax(pf, 0)
    return pf


def grow_kp(pf: A.PolarField, kp_new: int | None = None, kpa_new: int | None = None) -> A.PolarField:
    """The same polar field with more inline segments per radial (kp_new) and / or axial (kpa_new) ray;
    without arguments every family that overflowed doubles."""
    kp_old = int(pf.info[A.PI_KP])
    kpa_old = int(pf.info[A.PI_KPA])
    flags = int(pf.info[A.PI_OVERFLOW])
    if kp_new is None and kpa_new is None:
        kp_new = 2 * kp_old if flags & 1 else kp_old
        kpa_new = 2 * kpa_old if flags & 2 else kpa_old
    kp_new = int(max(kp_old, kp_new if kp_new is not None else kp_old))
    kpa_new = int(max(kpa_old, kpa_new if kpa_new is not None else kpa_old))

    def grow(arr, k_old, k_new):
        if k_new == k_old:
            return arr
        out = np.zeros((arr.shape[0], k_new, 2), np.int32)
        out[:, :k_old] = arr
        return out

    info = pf.info.copy()
    info[A.PI_KP] = kp_new
    info[A.PI_KPA] = kpa_new
    info[A.PI_OVERFLOW] = 0
    return pf._replace(rseg=grow(pf.rseg, kp_old, kp_new), rsev=grow(pf.rsev, kp_old, kp_new),
                       aseg=grow(pf.aseg, kpa_old, kpa_new), asev=grow(pf.asev, kpa_old, kpa_new), info=info)


def q2_volume_factor(pf: A.PolarField) -> float:
    """Volume per unit of the radial sum of q^2 differences: dtheta / 2 * dz * quantum^2."""
    return 0.5 * float(pf.geom[A.PG_DTHETA]) * float(pf.geom[A.PG_DZ]) * A.QUANTUM_M ** 2


def q2_sum(pf: A.PolarField, rays_per_chunk: int = 1 << 18) -> int:
    """Sum over the radial rays of q_hi^2 - q_lo^2 as an exact integer. k_polar.polar_q2_sum accumulates in int64,
    which wraps past 9.2e18: a d45 x 230 bar at 720 bins and 36 um rows sums to 2.3e19 (its volume read 20 % of
    the bar's). Each ray's sum fits int64; the chunk sums are split into 31-bit halves so they cannot wrap."""
    total = 0
    kp = pf.rseg.shape[1]
    lane = np.arange(kp)[None, :]
    low = np.int64((1 << 31) - 1)
    for a in range(0, pf.rcnt.shape[0], int(rays_per_chunk)):
        cnt = pf.rcnt[a:a + int(rays_per_chunk)].astype(np.int64)
        seg = pf.rseg[a:a + int(rays_per_chunk)].astype(np.int64)
        per_ray = ((seg[:, :, 1] * seg[:, :, 1] - seg[:, :, 0] * seg[:, :, 0]) * (lane < cnt[:, None])).sum(axis=1)
        total += (int((per_ray >> 31).sum()) << 31) + int((per_ray & low).sum())
    return total


def volume_m3(pf: A.PolarField) -> float:
    return float(q2_sum(pf)) * q2_volume_factor(pf)


def clear_dirty(pf: A.PolarField) -> None:
    for k in (A.PI_DIRTY_Z0, A.PI_DIRTY_Z1, A.PI_DIRTY_R0, A.PI_DIRTY_R1, A.PI_TOUCH_Q0, A.PI_TOUCH_Q1):
        pf.info[k] = -1


def touched_radii(pf: A.PolarField, since_made: bool = False) -> tuple[float, float] | None:
    """(r_lo, r_hi) in metres the cuts reached since the last sync (``since_made``: since the field was made), or
    None when none removed material."""
    k0, k1 = (A.PI_OPTOUCH_Q0, A.PI_OPTOUCH_Q1) if since_made else (A.PI_TOUCH_Q0, A.PI_TOUCH_Q1)
    q0, q1 = int(pf.info[k0]), int(pf.info[k1])
    if q0 < 0 or q1 < q0:
        return None
    return q0 * A.QUANTUM_M, q1 * A.QUANTUM_M


def dirty_z_range(pf: A.PolarField) -> tuple[float, float] | None:
    i0, i1 = int(pf.info[A.PI_DIRTY_Z0]), int(pf.info[A.PI_DIRTY_Z1])
    if i0 < 0:
        return None
    z0, dz = float(pf.geom[A.PG_Z0]), float(pf.geom[A.PG_DZ])
    return z0 + i0 * dz, z0 + (i1 + 1) * dz


# --------------------------------------------------------------------------- insert regions
def _chains(points: np.ndarray, axis: int):
    """Split a closed polygon at its extreme vertices along ``axis`` into the two chains between them,
    each ascending along the axis: (low chain, high chain) by the other coordinate at the middle; the
    envelopes (min / max of all edges) where a chain is not monotone."""
    other = 1 - axis
    n = len(points)
    i_min = int(np.argmin(points[:, axis] + 1e-15 * points[:, other]))
    i_max = int(np.argmax(points[:, axis] - 1e-15 * points[:, other]))
    chain_a = [points[(i_min + k) % n] for k in range((i_max - i_min) % n + 1)]
    chain_b = [points[(i_max + k) % n] for k in range((i_min - i_max) % n + 1)][::-1]

    def monotone(chain, take_max):
        pts = np.asarray(chain)
        keys = np.unique(points[:, axis])
        vals = []
        for key in keys:
            best = None
            for k in range(len(pts) - 1):
                a, b = pts[k], pts[k + 1]
                lo, hi = min(a[axis], b[axis]), max(a[axis], b[axis])
                if lo - 1e-15 <= key <= hi + 1e-15:
                    if hi - lo <= 1e-15:
                        cand = [a[other], b[other]]
                    else:
                        t = (key - a[axis]) / (b[axis] - a[axis])
                        cand = [a[other] + t * (b[other] - a[other])]
                    for c in cand:
                        best = c if best is None else (max(best, c) if take_max else min(best, c))
            vals.append(best if best is not None else np.nan)
        keys = np.asarray(keys, dtype=float)
        vals = np.asarray(vals, dtype=float)
        ok = np.isfinite(vals)
        return np.ascontiguousarray(keys[ok]), np.ascontiguousarray(vals[ok])

    mid = 0.5 * (points[i_min, axis] + points[i_max, axis])

    def value_at(chain, x):
        pts = np.asarray(chain)
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            lo, hi = min(a[axis], b[axis]), max(a[axis], b[axis])
            if lo <= x <= hi and hi > lo:
                return a[other] + (x - a[axis]) / (b[axis] - a[axis]) * (b[other] - a[other])
        return float(np.mean(pts[:, other]))

    va, vb = value_at(chain_a, mid), value_at(chain_b, mid)
    low, high = (chain_a, chain_b) if va <= vb else (chain_b, chain_a)
    return monotone(low, False), monotone(high, True)


def insert_region(insert) -> A.InsertRegion:
    """InsertRegion from tooling's InsertProfile (pts_zx_m closed polygon in (z_T, x_T), nose centre and
    radius). A region that is not monotone along z (or x) is taken as its monotone envelope (documented)."""
    pts = np.asarray(insert.pts_zx_m, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        raise InputError("insert outline needs at least three points")
    (zl, xl), (zu, xu) = _chains(pts, 0)
    (xa, za), (xb, zb) = _chains(pts, 1)
    rho = float(np.asarray(insert.nose_radius_m).reshape(-1)[0])
    centre = np.asarray(insert.nose_centre_zx_m, dtype=float).reshape(2)
    nose = np.array([centre[0], centre[1], rho if rho > 0.0 else 0.0])
    box = np.array([pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(), 0.0, 0.0])
    if rho > 0.0:
        box[0] = min(box[0], centre[0] - rho)
        box[1] = max(box[1], centre[0] + rho)
        box[2] = min(box[2], centre[1] - rho)
        box[3] = max(box[3], centre[1] + rho)
    # the upper chain rises to its highest vertex and falls after it (convex outlines): the largest x_T over
    # a z_T range is then the chain at the range point nearest that vertex (k_polar's block bound in O(1))
    if len(zu) > 0:
        k = int(np.argmax(xu))
        box[4] = zu[k]
        box[5] = 1.0 if (np.all(np.diff(xu[:k + 1]) >= 0.0) and np.all(np.diff(xu[k:]) <= 0.0)) else 0.0
    return A.InsertRegion(zl, xl, zu, xu, xa, za, xb, zb, nose, box)


def polar_step(z0, r0, z1, r1, th0, th1, sz: float = 1.0, sx: float = -1.0, yoff: float = 0.0) -> np.ndarray:
    ps = np.zeros(A.N_PS)
    ps[A.PS_Z0], ps[A.PS_R0], ps[A.PS_Z1], ps[A.PS_R1] = z0, r0, z1, r1
    ps[A.PS_TH0], ps[A.PS_TH1] = th0, th1
    ps[A.PS_SZ], ps[A.PS_SX], ps[A.PS_YOFF] = sz, sx, yoff
    return ps


def to_arrays(pf: A.PolarField, prefix: str = "") -> dict:
    return {prefix + name: np.array(getattr(pf, name)) for name in A.PolarField._fields}


def from_arrays(arrays: dict, prefix: str = "") -> A.PolarField:
    return A.PolarField(*(np.ascontiguousarray(np.array(arrays[prefix + name])) for name in A.PolarField._fields))


def empty_polar() -> A.PolarField:
    return make_polar(8, 0.0, 1e-3, 1, 1e-3, 1, 1)


def bytes_of(pf: A.PolarField) -> int:
    return sum(int(np.asarray(a).nbytes) for a in pf)
