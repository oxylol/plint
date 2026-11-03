"""Polar field construction and conversion (docs/models/removal.md "Polar field").

* :func:`resolution` - theta bins and row height from the rules of architecture 5.4: at least
  ``physics.polar_theta_bins`` bins, dz <= min(f_min / 8, r_nose / 16), covering the turned length; when the
  arrays would exceed ``max_bytes`` the rows are coarsened to fit and the returned note says so (the domain
  raises a warning finding with it).
* :func:`polar_tube` - a cylinder / tube stock directly; :func:`polar_from_field` - the polar field of an
  already machined tri-dexel field (mill-turn), sampled on the base lattice's z planes;
* :func:`insert_region` - tooling's InsertProfile as the chains the cut kernel reads;
* :func:`grow_kp` - more inline segments per ray after a PI_OVERFLOW;
* :func:`volume_m3` - the exact integer volume of the radial family.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.errors import InputError
from . import abi as A


def resolution(length_m: float, theta_bins: int, f_min_m_per_rev: float, r_nose_m: float, r_max_m: float,
               kp: int = 2, max_bytes: float = 150e6, dr_m: float | None = None, lattice_m: float = 0.25e-3):
    """(n_theta, dz, nz, dr, nr, note)."""
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


def volume_m3(pf: A.PolarField) -> float:
    from .k_polar import polar_q2_sum

    return float(polar_q2_sum(pf, 0)) * q2_volume_factor(pf)


def clear_dirty(pf: A.PolarField) -> None:
    for k in (A.PI_DIRTY_Z0, A.PI_DIRTY_Z1, A.PI_DIRTY_R0, A.PI_DIRTY_R1):
        pf.info[k] = -1


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
