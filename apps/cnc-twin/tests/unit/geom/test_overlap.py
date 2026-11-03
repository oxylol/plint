"""Overlap tests against independent references (edge-segment tests, clipping, triangle soups, dense sampling)
and against analytic tool-envelope clearances."""

import math

import numpy as np
import pytest

from cnctwin.geom import bvh, overlap as ov, reference as ref
from cnctwin.geom import transform as tf
from cnctwin.geom.primitives import box_mesh, cylinder_mesh

RNG = np.random.default_rng(5)


def random_pose():
    return tf.pose(tf.axis_angle(RNG.normal(size=3), RNG.uniform(-math.pi, math.pi)), RNG.uniform(-0.5, 0.5, 3))


def test_tri_tri_matches_reference():
    hits = 0
    for _ in range(3000):
        a = RNG.normal(size=(3, 3))
        b = RNG.normal(size=(3, 3)) * RNG.uniform(0.3, 1.5) + RNG.normal(scale=0.7, size=3)
        got = ov.tri_tri_overlap(a, b)
        assert got == ref.tri_tri_overlap_ref(a, b)
        hits += got
    assert 300 < hits < 2700


def test_tri_tri_coplanar_and_touching_cases():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    assert ov.tri_tri_overlap(a, a + [0.2, 0.2, 0])
    assert not ov.tri_tri_overlap(a, a + [2.0, 0, 0])
    assert ov.tri_tri_overlap(a, np.array([[0.1, 0.1, 0], [0.2, 0.1, 0], [0.1, 0.2, 0]]))     # contained
    assert ov.tri_tri_overlap(a, np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]], float))          # shared vertex
    assert ov.tri_tri_overlap(a, np.array([[0.5, 0.5, -1], [0.5, 0.5, 1], [2, 2, 0]]))         # pierces an edge
    assert not ov.tri_tri_overlap(a, np.array([[0.5, 0.5, 0.1], [0.5, 0.6, 1], [2, 2, 0.1]]))
    # a degenerate (zero-area) triangle through the other
    seg = np.array([[0.2, 0.2, -1], [0.2, 0.2, 1], [0.2, 0.2, 0.5]])
    assert ov.tri_tri_overlap(a, seg) and ov.tri_tri_overlap(seg, a)
    assert not ov.tri_tri_overlap(a, seg + [3, 0, 0])


def test_tri_box_matches_clipping_reference():
    hits = 0
    for _ in range(3000):
        tri = RNG.normal(size=(3, 3)) * 0.6
        centre = RNG.normal(scale=0.6, size=3)
        half = RNG.uniform(0.05, 0.5, 3)
        got = ov.tri_box_overlap(tri, centre, half)
        assert got == ref.tri_box_overlap_ref(tri, centre, half)
        hits += got
    assert 300 < hits < 2700


def test_box_box_matches_reference():
    hits = 0
    for _ in range(600):
        T = random_pose()
        ca, cb = RNG.normal(scale=0.3, size=3), RNG.normal(scale=0.3, size=3)
        ha, hb = RNG.uniform(0.02, 0.3, 3), RNG.uniform(0.02, 0.3, 3)
        got = ov.box_box_overlap(ca, ha, T, cb, hb)
        assert got == ref.box_box_overlap_ref(ca, ha, T, cb, hb)
        hits += got
    assert 60 < hits < 540


def test_mesh_mesh_matches_all_pairs():
    a = box_mesh([0, 0, 0], [1, 1, 1])
    b = cylinder_mesh([0, 0, 0], [0, 0, 0.6], 0.2, chordal_tol_m=0.02)
    ta, tb = bvh.build_tree(a), bvh.build_tree(b)
    for T in (tf.translation([0.5, 0.5, 0.8]), tf.pose(tf.rot_x(0.7), [1.0, 0.5, 0.5]), tf.translation([0.5, 0.5, 0.2]),
              tf.translation([0.5, 0.5, 1.01])):
        pairs = ov.mesh_mesh_overlap(ta, tb, T)
        assert {tuple(p) for p in pairs.tolist()} == ref.mesh_mesh_overlap_ref(a, b, T)
    # inside without touching: no surface pairs (containment is the point classification's job)
    assert len(ov.mesh_mesh_overlap(ta, tb, tf.translation([0.5, 0.5, 0.2]))) == 0
    first = ov.mesh_mesh_overlap(ta, tb, tf.translation([0.5, 0.5, 0.8]), first_only=True)
    assert len(first) == 1


def test_mesh_box_matches_reference():
    m = cylinder_mesh([0, 0, 0], [0, 0, 0.1], 0.05, chordal_tol_m=0.002)
    tree = bvh.build_tree(m)
    for _ in range(20):
        T = random_pose()
        T[:3, 3] *= 0.15
        half = RNG.uniform(0.01, 0.06, 3)
        got = set(ov.mesh_box_overlap(tree, T, half).tolist())
        assert got == ref.mesh_box_overlap_ref(m, T, half)
    T = tf.translation([0.0, 0.0, 0.105])
    assert len(ov.mesh_box_overlap(tree, T, [0.01, 0.01, 0.004])) == 0
    assert len(ov.mesh_box_overlap(tree, T, [0.01, 0.01, 0.004], margin=0.002)) > 0


def test_revolved_clearance_analytic_cases():
    tree = bvh.build_tree(box_mesh([0, 0, 0], [1, 1, 1]))
    z, r = [0.0, 0.0, 0.03, 0.03], [0.0, 0.005, 0.005, 0.0]     # D10 x 30 envelope, tip at the frame origin
    _, _, c, face = ov.mesh_revolved_overlap(tree, tf.translation([0.5, 0.5, 1.001]), z, r, margin=0.002)
    assert math.isclose(c, 0.001, abs_tol=1e-9) and face >= 0           # 1 mm above the top face
    _, _, c, _ = ov.mesh_revolved_overlap(tree, tf.translation([0.5, 0.5, 0.999]), z, r)
    assert math.isclose(c, -0.005, abs_tol=1e-9)                        # plunged: radial penetration = R
    faces, clear, c, _ = ov.mesh_revolved_overlap(tree, tf.translation([1.007, 0.5, 0.5]), z, r, margin=0.003)
    assert math.isclose(c, 0.002, abs_tol=1e-9) and len(faces) == 2     # 2 mm beside the +x side
    T = tf.pose(tf.rot_y(math.pi / 2), [1.001, 0.5, 0.5])               # lying along +x, 1 mm off the side
    _, _, c, _ = ov.mesh_revolved_overlap(tree, T, z, r, margin=0.002)
    assert math.isclose(c, 0.001, abs_tol=1e-9)
    faces, _, c, face = ov.mesh_revolved_overlap(tree, tf.translation([0.5, 0.5, 1.01]), z, r, margin=0.002)
    assert len(faces) == 0 and face == -1 and c == math.inf
    with pytest.raises(ValueError):
        ov.mesh_revolved_overlap(tree, np.eye(4), [0.0, 0.02, 0.01], [0.0, 0.01, 0.01])


def test_chord_distance_matches_dense_sampling():
    for _ in range(300):
        tri = np.ascontiguousarray(RNG.normal(size=(3, 3)))
        zq = RNG.uniform(tri[:, 2].min(), tri[:, 2].max())
        d = ov.chord_axis_distance_nb(tri, zq)
        # chord end points by edge interpolation, then dense sampling of the chord
        pts = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            if (a[2] - zq) * (b[2] - zq) <= 0 and a[2] != b[2]:
                pts.append(a + (zq - a[2]) / (b[2] - a[2]) * (b - a))
        p, q = pts[0], pts[-1]
        s = np.linspace(0, 1, 20001)[:, None]
        brute = np.min(np.hypot(*(p[:2] + s * (q[:2] - p[:2])).T))
        assert d <= brute + 1e-15 and brute - d <= np.linalg.norm(q - p) / 20000


def test_triangle_clearance_matches_sampling():
    z = np.array([0.0, 0.0, 0.01, 0.03, 0.03])
    r = np.array([0.0, 0.004, 0.006, 0.006, 0.0])                     # cone frustum then cylinder
    for _ in range(60):
        tri = np.ascontiguousarray(RNG.uniform([-0.01, -0.01, -0.005], [0.01, 0.01, 0.035], (3, 3)))
        got = ov.triangle_clearance_nb(tri, z, r, 0.0, 1e-12)
        expected = ref.triangle_clearance_ref(tri, z, r, samples=120)
        if expected == math.inf:
            assert got == math.inf or got >= -1e-12
            continue
        # sampling over-estimates the minimum by at most the sample spacing
        spacing = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=0), axis=1)) / 120
        assert got <= expected + 1e-12 and expected - got <= 2 * spacing


def test_each_kernel_has_one_signature():
    for kernel in ov.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_clearance_is_accurate_for_a_nearly_horizontal_face():
    """A face tilted by 1 mrad, 1 mm below a D10 tool tip: the end-zone clearance max(axial, radial) has its
    minimum (gap - R tan a) / (1 + tan a) where the two gaps are equal (derived in docs/models/geom.md)."""
    gap, R, a = 1e-3, 0.005, 1e-3
    big = 0.05
    xs = np.array([-big, big, -big, big])
    ys = np.array([-big, -big, big, big])
    zs = -gap - xs * math.tan(a)
    v = np.column_stack([xs, ys, zs])
    from cnctwin.geom.mesh import Mesh

    plane = Mesh(v, [[0, 1, 3], [0, 3, 2]])
    tree = bvh.build_tree(plane)
    z, r = [0.0, 0.0, 0.03, 0.03], [0.0, R, R, 0.0]
    _, _, c, _ = ov.mesh_revolved_overlap(tree, np.eye(4), z, r, margin=0.002)
    expected = (gap - R * math.tan(a)) / (1.0 + math.tan(a))
    assert math.isclose(c, expected, abs_tol=2e-9)
