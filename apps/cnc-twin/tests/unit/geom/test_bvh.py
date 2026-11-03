"""Bounding-volume trees: deterministic build (kernel = Python reference, bit for bit), tree invariants, rays,
closest points, signed distance and point classification against brute force and analytic answers."""

import math

import numpy as np
import pytest

from cnctwin.geom import bvh, reference as ref
from cnctwin.geom import transform as tf
from cnctwin.geom.abi import INFO_CLOSED, INFO_DEPTH, INSIDE, MAX_DEPTH, ON_SURFACE, OUTSIDE
from cnctwin.geom.mesh import Mesh, merge
from cnctwin.geom.primitives import box_mesh, sphere_mesh

RNG = np.random.default_rng(3)
SPHERE_R = 0.05
SPHERE_C = np.array([0.3, 0.1, 0.05])


@pytest.fixture(scope="module")
def scene():
    mesh = merge([box_mesh([0, 0, 0], [0.1, 0.08, 0.03]), sphere_mesh(SPHERE_C, SPHERE_R, 1e-3)], offset_groups=True)
    return mesh, bvh.build_tree(mesh)


def test_build_is_deterministic_and_equals_the_reference(scene):
    mesh, tree = scene
    again = bvh.build_tree(mesh)
    for name in tree._fields:
        assert np.array_equal(getattr(tree, name), getattr(again, name)), name
    order, lo, hi, right, start, count, depth = ref.build_tree_ref(mesh, 4, 12, 0.125)
    assert np.array_equal(tree.tri_face, order)
    assert np.array_equal(tree.node_lo, lo) and np.array_equal(tree.node_hi, hi)
    assert np.array_equal(tree.node_right, right) and np.array_equal(tree.node_count, count)
    assert np.array_equal(tree.node_start[count > 0], start[count > 0])
    assert tree.info[INFO_DEPTH] == depth


def test_tree_invariants(scene):
    mesh, tree = scene
    assert sorted(tree.tri_face.tolist()) == list(range(mesh.n_faces))
    assert np.array_equal(tree.tri_v, mesh.triangles()[tree.tri_face])
    nn = len(tree.node_right)
    for node in range(nn):
        lo, hi = tree.node_lo[node], tree.node_hi[node]
        if tree.node_right[node] < 0:
            s, c = tree.node_start[node], tree.node_count[node]
            pts = tree.tri_v[s:s + c].reshape(-1, 3)
            assert np.all(pts >= lo) and np.all(pts <= hi)
        else:
            for child in (node + 1, tree.node_right[node]):
                assert np.all(tree.node_lo[child] >= lo) and np.all(tree.node_hi[child] <= hi)
    assert tree.info[INFO_CLOSED] == 1


def test_degenerate_input_is_split_by_index_and_depth_capped():
    tri = np.array([[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0]], float)
    mesh = Mesh.from_triangles(np.repeat(tri[None], 300, axis=0), weld=False)
    tree = bvh.build_tree(mesh)
    assert tree.info[INFO_DEPTH] <= MAX_DEPTH
    t, f = bvh.ray_all_hits(tree, [2e-4, 2e-4, 1.0], [0, 0, -1])
    assert len(t) == 300 and np.array_equal(f, np.arange(300))


def test_ray_triangle_kernel_matches_moller_trumbore():
    out = np.empty(5)
    setup = np.empty(6)
    agree = 0
    for _ in range(2000):
        v = np.ascontiguousarray(RNG.normal(size=(3, 3)))
        o = np.ascontiguousarray(RNG.normal(size=3) * 2)
        b = RNG.uniform(-0.3, 1.0, 2)
        target = v[0] + b[0] * (v[1] - v[0]) + b[1] * (v[2] - v[0])
        d = np.ascontiguousarray((target - o) * RNG.choice([-1.0, 1.0]))
        bvh.ray_setup_nb(d, setup)
        hit = bvh.ray_triangle_nb(o, setup, v, -np.inf, np.inf, out)
        r = ref.ray_triangle_ref(o, d, v[0], v[1], v[2])
        if r is None:
            assert hit == 0
            continue
        assert hit == 1
        assert math.isclose(out[0], r[0], rel_tol=1e-9, abs_tol=1e-12)
        assert np.allclose(out[1:4], [1 - r[1] - r[2], r[1], r[2]], atol=1e-9)
        agree += 1
    assert agree > 500


def test_first_and_all_hits_match_brute_force(scene):
    mesh, tree = scene
    for _ in range(80):
        o = RNG.uniform([-0.1, -0.1, -0.1], [0.5, 0.25, 0.2])
        d = tf.unit(RNG.normal(size=3))
        expected = ref.ray_hits_ref(mesh, o, d)
        got = bvh.ray_first_hit(tree, o, d)
        if not expected:
            assert got is None
            continue
        assert got is not None and math.isclose(got[0], expected[0][0], rel_tol=1e-12, abs_tol=1e-15)
        t_all, f_all = bvh.ray_all_hits(tree, o, d)
        assert np.allclose(t_all, [h[0] for h in expected], rtol=1e-12, atol=1e-15)
        assert sorted(f_all.tolist()) == sorted(h[1] for h in expected)


def test_batch_rays_csr_and_segments(scene):
    _, tree = scene
    O = RNG.uniform([-0.1, -0.1, -0.1], [0.5, 0.25, 0.2], (40, 3))
    D = np.array([tf.unit(v) for v in RNG.normal(size=(40, 3))])
    offsets, t, f = bvh.rays_all_hits(tree, O, D)
    tf_, ff = bvh.rays_first_hit(tree, O, D)
    for i in range(40):
        ti, fi = bvh.ray_all_hits(tree, O[i], D[i])
        assert np.array_equal(t[offsets[i]:offsets[i + 1]], ti) and np.array_equal(f[offsets[i]:offsets[i + 1]], fi)
        if len(ti):
            assert tf_[i] == ti[0]
        else:
            assert np.isinf(tf_[i]) and ff[i] == -1
    s, face = bvh.segments_hit(tree, [[-0.05, 0.04, 0.01], [-0.05, 0.04, 0.01]], [[0.05, 0.04, 0.01], [-0.01, 0.04, 0.01]])
    assert math.isclose(s[0], 0.5, rel_tol=1e-12) and np.isinf(s[1])
    hit = bvh.segment_hit(tree, [-0.05, 0.04, 0.01], [0.05, 0.04, 0.01])
    assert np.allclose(hit[2], [0.0, 0.04, 0.01], atol=1e-15)


def test_ray_through_a_shared_edge_is_not_lost():
    box = box_mesh([0, 0, 0], [1, 1, 1])
    tree = bvh.build_tree(box)
    # the -x side is split along its diagonal; this ray meets the diagonal exactly
    t, f = bvh.ray_all_hits(tree, [-1, 0.5, 0.5], [1, 0, 0])
    assert t[0] == 1.0 and len(t) >= 2
    first = bvh.ray_first_hit(tree, [-1, 0.5, 0.5], [1, 0, 0])
    assert first[0] == 1.0 and first[1] == min(f[t == 1.0])


def test_closest_point_and_signed_distance_match_brute_force(scene):
    mesh, tree = scene
    P = RNG.uniform([-0.05, -0.05, -0.05], [0.4, 0.2, 0.15], (40, 3))
    d, f, c = bvh.signed_distance(tree, P)
    for i, p in enumerate(P):
        q, dist, _ = ref.closest_point_ref(mesh, p)
        assert math.isclose(abs(d[i]), dist, rel_tol=1e-12, abs_tol=1e-16)
        assert np.allclose(c[i], q, atol=1e-14) or math.isclose(np.linalg.norm(c[i] - p), dist, rel_tol=1e-12)
        assert math.isclose(d[i], ref.signed_distance_ref(mesh, p), rel_tol=1e-12, abs_tol=1e-16)
    res = bvh.closest_point(tree, [0.05, 0.04, 0.1])
    assert np.allclose(res[0], [0.05, 0.04, 0.03]) and math.isclose(res[1], 0.07)
    assert bvh.closest_point(tree, [0.05, 0.04, 0.1], max_dist=0.05) is None


def test_signed_distance_of_a_sphere_is_analytic_within_the_chord():
    tol = 1e-5
    mesh = sphere_mesh([0, 0, 0], 0.02, tol)
    tree = bvh.build_tree(mesh)
    P = RNG.normal(size=(200, 3))
    P = P / np.linalg.norm(P, axis=1, keepdims=True) * RNG.uniform(0.0, 0.04, (200, 1))
    d, _, _ = bvh.signed_distance(tree, P)
    exact = np.linalg.norm(P, axis=1) - 0.02
    assert np.all(np.abs(d - exact) <= 2 * tol)


def test_point_classification(scene):
    mesh, tree = scene
    P = RNG.uniform([-0.02, -0.02, -0.02], [0.4, 0.2, 0.12], (400, 3))
    cls = bvh.points_in_mesh(tree, P)
    in_box = np.all((P > 0) & (P < [0.1, 0.08, 0.03]), axis=1)
    r = np.linalg.norm(P - SPHERE_C, axis=1)
    clear = (r < SPHERE_R - 2.5e-3) | (r > SPHERE_R + 1e-6)
    in_sphere = r < SPHERE_R
    expect = np.where(in_box | in_sphere, INSIDE, OUTSIDE)
    assert np.array_equal(cls[clear], expect[clear])
    w = bvh.winding_numbers(tree, P[:50])
    for i in range(50):
        assert math.isclose(w[i], ref.winding_number_ref(mesh, P[i]), abs_tol=1e-9)
    assert np.allclose(np.round(w), (cls[:50] == INSIDE).astype(float))
    # vertices, edges and faces of the box are on the surface
    on = bvh.points_in_mesh(tree, [[0, 0, 0], [0.05, 0, 0], [0.05, 0.04, 0.03], [0.1, 0.04, 0.015]])
    assert np.all(on == ON_SURFACE)


def test_open_mesh_signed_distance_needs_permission():
    open_box = box_mesh([0, 0, 0], [1, 1, 1]).submesh(np.arange(12) < 10)
    tree = bvh.build_tree(open_box)
    assert tree.info[INFO_CLOSED] == 0
    with pytest.raises(ValueError):
        bvh.signed_distance(tree, [[0.5, 0.5, 0.5]])
    d, _, _ = bvh.signed_distance(tree, [[0.5, 0.5, 0.5]], allow_open=True)
    assert math.isclose(abs(d[0]), 0.5)
    # three of nine parity rays may leak through the open top, the majority still says inside
    assert bvh.points_in_mesh(tree, [[0.5, 0.5, 0.3]])[0] == INSIDE


def test_save_load_and_static_cache(tmp_path, scene):
    mesh, tree = scene
    bvh.save_tree(tmp_path / "t.npz", tree)
    back = bvh.load_tree(tmp_path / "t.npz")
    for name in tree._fields:
        assert np.array_equal(getattr(tree, name), getattr(back, name))
    assert bvh.ray_first_hit(back, [-1, 0.04, 0.01], [1, 0, 0])[0] == 1.0
    a = bvh.static_tree(mesh)
    assert bvh.static_tree(mesh.copy()) is a


def test_closest_point_triangle_kernel_matches_reference():
    out = np.empty(3)
    for _ in range(2000):
        v = np.ascontiguousarray(RNG.normal(size=(3, 3)))
        p = np.ascontiguousarray(RNG.normal(size=3) * 2)
        bvh.closest_point_triangle_nb(p, v, out)
        q = ref.closest_point_triangle_ref(p, v[0], v[1], v[2])
        assert math.isclose(np.linalg.norm(out - p), np.linalg.norm(q - p), rel_tol=1e-10, abs_tol=1e-13)


def test_each_kernel_has_one_signature():
    for kernel in bvh.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
