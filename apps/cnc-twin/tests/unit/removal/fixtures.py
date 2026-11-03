"""Analytic fields for the removal tests: every ray's segments from an exact closed form, so the kernels
are checked against geometry, not against themselves."""

from __future__ import annotations

import math

import numpy as np

from cnctwin.removal import abi as A
from cnctwin.removal.field import make_field, n_levels

Q = A.QUANTUM_M


def ray_origin(field, lvl, fam, iu, iv):
    u, v = A.TRANS_U[fam], A.TRANS_V[fam]
    o = field.lv_origin_m[lvl].copy()
    a = field.lv_cell_m[lvl]
    o[u] += iu * a
    o[v] += iv * a
    return o


def build(lo, hi, lattice, segments_fn, refine=(), margin=1):
    """Field over the box whose every ray gets ``segments_fn(origin, fam, length_m) -> [(t0, t1), ...]``
    (metres along the ray from the level origin)."""
    from cnctwin.removal.k_stock import replace_ray

    field = make_field(lo, hi, lattice, refine, margin)
    sc = A.make_scratch()
    for lvl in range(n_levels(field)):
        for fam in range(3):
            u, v = A.TRANS_U[fam], A.TRANS_V[fam]
            length = (field.lv_dims[lvl, fam] - 1) * field.lv_cell_m[lvl]
            for iu in range(int(field.lv_dims[lvl, u])):
                for iv in range(int(field.lv_dims[lvl, v])):
                    o = ray_origin(field, lvl, fam, iu, iv)
                    segs = segments_fn(o, fam, length)
                    if not segs:
                        continue
                    qlo = np.array([math.floor(a / Q + 0.5) for a, b in segs], np.int64)
                    qhi = np.array([math.floor(b / Q + 0.5) for a, b in segs], np.int64)
                    assert replace_ray(field, lvl, fam, iu, iv, qlo, qhi, len(segs), A.SEV_STOCK, sc) == 0
    return field


def clip(t0, t1, length):
    a, b = max(t0, 0.0), min(t1, length)
    return [(a, b)] if b > a else []


def box_ray(o, fam, length, lo, hi):
    u, v = A.TRANS_U[fam], A.TRANS_V[fam]
    if not (lo[u] <= o[u] < hi[u] and lo[v] <= o[v] < hi[v]):
        return []
    return clip(lo[fam] - o[fam], hi[fam] - o[fam], length)


def halfspace_box(lo, hi, normal, d):
    """Block [lo, hi] intersected with n . x <= d."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)

    def fn(o, fam, length):
        segs = box_ray(o, fam, length, lo, hi)
        if not segs:
            return []
        t0, t1 = segs[0]
        nf = n[fam]
        no = float(np.dot(n, o))
        if nf == 0.0:
            return segs if no <= d else []
        t = (d - no) / nf
        if nf > 0.0:
            return clip(t0, min(t1, t), length)
        return clip(max(t0, t), t1, length)
    return fn


def cylinder_hole_box(lo, hi, centre_xy, radius, inside_material=False):
    """Block minus (or, inside_material: intersected with) a vertical cylinder about centre_xy."""
    cx, cy = centre_xy

    def fn(o, fam, length):
        segs = box_ray(o, fam, length, lo, hi)
        if not segs:
            return []
        t0, t1 = segs[0]
        if fam == 2:
            r2 = (o[0] - cx) ** 2 + (o[1] - cy) ** 2
            inside = r2 < radius * radius
            return segs if inside == inside_material else []
        across = o[1] - cy if fam == 0 else o[0] - cx
        along_c = (cx if fam == 0 else cy) - o[fam]
        if abs(across) >= radius:
            return [] if inside_material else segs
        h = math.sqrt(radius * radius - across * across)
        a, b = along_c - h, along_c + h
        if inside_material:
            return clip(max(t0, a), min(t1, b), length)
        return clip(t0, min(t1, a), length) + clip(max(t0, b), t1, length)
    return fn
