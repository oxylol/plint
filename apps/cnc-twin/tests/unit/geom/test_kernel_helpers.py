"""The small helper kernels against plain-Python statements of the same quantity (architecture 5.5: every
kernel has a reference and a test comparing the two)."""

import bisect
import math

import numpy as np

from cnctwin.geom import bvh, dda, mesh as gm, overlap as ov, primitives as P
from cnctwin.geom import transform as tf

RNG = np.random.default_rng(13)


def test_lex_lower_is_bisect_on_tuples():
    keys = np.unique(RNG.integers(-3, 4, (200, 3)), axis=0).astype(np.int64)
    rows = [tuple(r) for r in keys.tolist()]
    for _ in range(300):
        q = tuple(int(v) for v in RNG.integers(-4, 5, 3))
        assert gm._lex_lower(np.ascontiguousarray(keys), *q) == bisect.bisect_left(rows, q)


def test_polynomial_helpers():
    for _ in range(200):
        c = np.ascontiguousarray(RNG.normal(size=5))
        t = float(RNG.normal())
        assert math.isclose(P.poly_eval_nb(c, 4, t), float(np.polyval(c, t)), rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(P._poly_scale(c, 4, t), float(np.polyval(np.abs(c), abs(t))), rel_tol=1e-12)
    c = np.ascontiguousarray(np.poly([0.3, 2.0, -1.0]))
    root = P._bracket_root(c, 3, 0.0, 1.0, float(np.polyval(c, 0.0)), float(np.polyval(c, 1.0)))
    assert math.isclose(root, 0.3, rel_tol=1e-13)


def test_frame_of_axis_is_the_transform_frame():
    R = np.empty((3, 3))
    for _ in range(100):
        a = np.ascontiguousarray(tf.unit(RNG.normal(size=3)))
        P._frame_of_axis(a, R)
        assert np.allclose(R, tf.frame_from_z(a), atol=1e-15)


def test_box_helpers():
    for _ in range(300):
        lo = RNG.normal(size=3)
        hi = lo + RNG.uniform(0.01, 1.0, 3)
        p = RNG.normal(size=3) * 2
        d = hi - lo
        assert math.isclose(bvh._area(lo, hi), 2 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]), rel_tol=1e-14)
        assert math.isclose(bvh._box_dist2(p, lo, hi), float(np.sum((p - np.clip(p, lo, hi)) ** 2)), rel_tol=1e-13,
                            abs_tol=1e-300)
        o = RNG.normal(size=3) * 2
        v = RNG.normal(size=3)
        got = bvh.ray_box_entry_nb(np.ascontiguousarray(o), np.ascontiguousarray(v), lo, hi, 0.0, np.inf)
        ta, tb = (lo - o) / v, (hi - o) / v
        tn, tfar = max(0.0, float(np.max(np.minimum(ta, tb)))), float(np.min(np.maximum(ta, tb)))
        if tn <= tfar:
            assert math.isclose(got, tn, rel_tol=1e-12, abs_tol=1e-15)
        elif tn > tfar * (1 + 1e-12) + 1e-15:
            assert got == np.inf


def test_two_d_predicates():
    for _ in range(500):
        a, b, c, d = RNG.normal(size=(4, 2))
        # reference: parametric solve of a + s (b - a) = c + u (d - c)
        m = np.column_stack([b - a, c - d])
        if abs(np.linalg.det(m)) > 1e-9:
            s, u = np.linalg.solve(m, c - a)
            expect = 0.0 <= s <= 1.0 and 0.0 <= u <= 1.0
            assert ov._seg_seg_2d(*a, *b, *c, *d) == expect
        p = RNG.normal(size=2)
        tri = RNG.normal(size=(3, 2))
        T = np.column_stack([tri[1] - tri[0], tri[2] - tri[0]])
        if abs(np.linalg.det(T)) > 1e-9:
            w = np.linalg.solve(T, p - tri[0])
            expect = w[0] >= 0 and w[1] >= 0 and w[0] + w[1] <= 1
            assert ov._point_in_tri_2d(p[0], p[1], tri[0], tri[1], tri[2]) == expect


def test_interval_and_segment_distance():
    out = np.empty(2)
    for _ in range(300):
        vv = RNG.normal(size=3)
        dd = RNG.normal(size=3)
        if np.all(dd > 0) or np.all(dd < 0):
            continue
        assert ov._interval(*vv, *dd, out) == 1
        # reference: where the signed distances change sign along the triangle's edges
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if dd[i] * dd[j] < 0:
                pts.append(vv[i] + (vv[j] - vv[i]) * dd[i] / (dd[i] - dd[j]))
            if dd[i] == 0:
                pts.append(vv[i])
        assert math.isclose(out[0], min(pts), abs_tol=1e-12) and math.isclose(out[1], max(pts), abs_tol=1e-12)
    for _ in range(300):
        a, b = RNG.normal(size=(2, 2))
        s = np.linspace(0, 1, 100001)[:, None]
        brute = float(np.min(np.linalg.norm(a + s * (b - a), axis=1)))
        assert ov._seg_origin_dist(*a, *b) <= brute + 1e-15
        assert brute - ov._seg_origin_dist(*a, *b) <= np.linalg.norm(b - a) / 1e5


def test_transform_tri_and_grid_clip():
    T = tf.pose(tf.axis_angle([1, 2, 3], 0.4), [0.1, 0.2, 0.3])
    tri = np.ascontiguousarray(RNG.normal(size=(3, 3)))
    out = np.empty((3, 3))
    ov._transform_tri(tri, np.ascontiguousarray(T[:3, :3]), np.ascontiguousarray(T[:3, 3]), out)
    assert np.allclose(out, tf.apply(T, tri), atol=1e-15)
    g = dda.grid([0, 0, 0], [0.1, 0.2, 0.3], [3, 2, 1])
    clip = np.empty(2)
    for _ in range(300):
        o = RNG.uniform(-0.5, 0.8, 3)
        d = RNG.normal(size=3)
        hit = dda.grid_clip_nb(np.ascontiguousarray(o), np.ascontiguousarray(d), 0.0, 1.0, g, clip)
        hi = np.array([0.3, 0.4, 0.3])
        ta, tb = (0 - o) / d, (hi - o) / d
        tn, tfar = max(0.0, float(np.max(np.minimum(ta, tb)))), min(1.0, float(np.min(np.maximum(ta, tb))))
        assert hit == (1 if tn <= tfar else 0)
        if hit:
            assert math.isclose(clip[0], tn, abs_tol=1e-14) and math.isclose(clip[1], tfar, abs_tol=1e-14)


def test_bvh_module_reexports_overlap():
    assert bvh.mesh_mesh_overlap is ov.mesh_mesh_overlap
