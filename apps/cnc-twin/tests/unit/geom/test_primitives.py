"""Primitives: tessellated volumes against exact polyhedral formulas and the analytic solids within the chord
bound, revolved profiles with arcs, analytic ray kernels against closed forms, the companion-matrix roots
and the tessellated meshes."""

import math

import numpy as np
import pytest

from cnctwin.geom import primitives as P
from cnctwin.geom import transform as tf
from cnctwin.geom.mesh import polyhedron_volume_cylinder

RNG = np.random.default_rng(11)
TOL = 1e-5


def test_cylinder_volume_is_the_inscribed_prism():
    r, h = 0.02, 0.1
    m = P.cylinder_mesh([0, 0, 0], [0, 0, h], r, chordal_tol_m=TOL)
    n = P.circle_segments(r, TOL)
    assert math.isclose(m.volume(), polyhedron_volume_cylinder(r, h, n), rel_tol=1e-13)
    assert 0.0 < math.pi * r * r * h - m.volume() <= 2.0 * math.pi * r * h * TOL
    assert m.is_watertight()
    assert m.group_names == {0: "cap0", 1: "side", 2: "cap1"}
    # caps are flat with outward normals
    n_faces = m.face_normals()
    assert np.allclose(n_faces[m.face_group == 0], [0, 0, -1]) and np.allclose(n_faces[m.face_group == 2], [0, 0, 1])


def test_oblique_cylinder_and_cone():
    p0, p1 = np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.25, 0.35])
    h = float(np.linalg.norm(p1 - p0))
    m = P.cylinder_mesh(p0, p1, 0.01, TOL)
    assert math.isclose(m.volume(), polyhedron_volume_cylinder(0.01, h, P.circle_segments(0.01, TOL)), rel_tol=1e-12)
    assert np.allclose(m.centroid(), 0.5 * (p0 + p1), atol=1e-15)
    k = P.cone_mesh([0, 0, 0], [0, 0, 0.1], 0.03, 0.0, TOL)
    assert k.is_watertight() and set(k.group_names) == {0, 1}
    exact = math.pi * 0.03 ** 2 * 0.1 / 3
    assert 0.0 < exact - k.volume() <= math.pi * 0.03 * math.hypot(0.03, 0.1) * 2 * TOL + math.pi * 0.03 ** 2 * TOL
    f = P.cone_mesh([0, 0, 0], [0, 0, 0.1], 0.03, 0.01, TOL)
    assert f.is_watertight() and set(f.group_names) == {0, 1, 2}


def test_sphere_and_torus_within_chord_bound():
    r = 0.05
    s = P.sphere_mesh([1, 2, 3], r, TOL)
    assert s.is_watertight()
    exact = 4.0 / 3.0 * math.pi * r ** 3
    assert 0.0 < exact - s.volume() <= 4 * math.pi * r * r * 2 * TOL
    assert np.allclose(s.centroid(), [1, 2, 3], atol=1e-12)
    t = P.torus_mesh([0, 0, 0], [0, 0, 1], 0.05, 0.01, TOL)
    assert t.is_watertight() and set(t.group_names) == {0, 1}
    exact = 2 * math.pi ** 2 * 0.05 * 0.01 ** 2
    assert 0.0 < exact - t.volume() <= 4 * math.pi ** 2 * 0.05 * 0.01 * 2 * TOL
    with pytest.raises(ValueError):
        P.torus_mesh([0, 0, 0], [0, 0, 1], 0.01, 0.02)


def test_box_mesh_groups():
    b = P.box_mesh([-1, -2, -3], [1, 2, 3])
    assert math.isclose(b.volume(), 48.0) and b.group_names[5] == "+z"
    with pytest.raises(ValueError):
        P.box_mesh([0, 0, 0], [1, 0, 1])


def ball_profile(R=0.005, L=0.03):
    """Ball-end mill envelope: tip on the axis, quarter arc to (R, R), shank to L, back to the axis."""
    return [0.0, R, L, L], [0.0, R, R, 0.0], [math.tan(-math.pi / 8), 0.0, 0.0]


def test_profile_area_and_revolved_ball_end():
    z, r, b = ball_profile()
    R, L = 0.005, 0.03
    # meridian area: rectangle (L - R) x R plus a quarter disc
    area = abs(P.profile_signed_area(z, r, b))
    assert math.isclose(area, (L - R) * R + math.pi * R * R / 4, rel_tol=1e-13)
    m = P.revolved_mesh(z, r, b, chordal_tol_m=TOL)
    assert m.is_watertight()
    exact = math.pi * R * R * (L - R) + 2.0 / 3.0 * math.pi * R ** 3
    assert 0.0 < exact - m.volume() <= (2 * math.pi * R * L + 2 * math.pi * R * R) * 2 * TOL
    assert sorted(set(m.face_group.tolist())) == [0, 1, 2]


def test_conservative_polyline_encloses_convex_arcs():
    z, r, b = ball_profile()
    zs, rs, seg = P.profile_polyline(z, r, b, chordal_tol_m=1e-4, conservative=True)
    R = 0.005
    # sample the true arc: centre (R, 0), radius R; the conservative polyline radius must be >= the arc
    for phi in np.linspace(math.pi, math.pi / 2, 50):
        zq, rq = R + R * math.cos(phi), R * math.sin(phi)
        rp = np.interp(zq, zs[:len(seg) + 1], rs[:len(seg) + 1])
        assert rp >= rq - 1e-15
    zs, rs, _ = P.profile_polyline(z, r, b, chordal_tol_m=1e-4, conservative=False)
    for zq, rq in zip(zs[1:-3], rs[1:-3]):
        assert math.isclose(math.hypot(zq - R, rq), R, rel_tol=1e-12)


def test_poly_roots_match_companion_matrix():
    out = np.empty(8)
    for _ in range(300):
        roots = np.sort(RNG.uniform(-2, 2, 4))
        c = np.ascontiguousarray(np.poly(roots) * RNG.uniform(0.5, 3.0))
        n = P.poly_roots_nb(c, 4, -10.0, 10.0, out)
        assert n == 4
        assert np.allclose(out[:4], P.poly_roots_ref(c, -10.0, 10.0), atol=1e-9)
        assert np.allclose(out[:4], roots, atol=1e-7)
    # complex pair + two real roots, infinite interval
    c = np.ascontiguousarray(np.polymul([1.0, 0.0, 1.0], [1.0, -3.0, 2.0]))
    n = P.poly_roots_nb(c, 4, -np.inf, np.inf, out)
    assert n == 2 and np.allclose(out[:2], [1.0, 2.0])
    # double root (tangency)
    c = np.ascontiguousarray(np.poly([0.5, 0.5, -1.0]))
    n = P.poly_roots_nb(c, 3, -5.0, 5.0, out)
    assert n == 2 and np.allclose(out[:2], [-1.0, 0.5], atol=1e-7)
    assert P.roots2_nb(1.0, -3.0, 2.0, 0.0, 10.0, out) == 2 and np.allclose(out[:2], [1, 2])
    assert P.roots2_nb(0.0, 2.0, -1.0, 0.0, 10.0, out) == 1 and out[0] == 0.5


def test_ray_box_sphere_cylinder_cone_analytic():
    t, n = P.ray_box([-1, 0.5, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1])
    assert t == 1.0 and np.allclose(n, [-1, 0, 0])
    t, n = P.ray_box([0.5, 0.5, 0.5], [0, 0, 2], [0, 0, 0], [1, 1, 1])
    assert t == 0.25 and np.allclose(n, [0, 0, 1])
    assert P.ray_box([-1, 2, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1]) is None
    t, n = P.ray_sphere([0, 0, -1], [0, 0, 1], [0, 0, 0], 0.5)
    assert math.isclose(t, 0.5) and np.allclose(n, [0, 0, -1])
    t, _ = P.ray_cylinder([-1, 0, 0.05], [1, 0, 0], [0, 0, 0], [0, 0, 0.1], 0.02)
    assert math.isclose(t, 0.98, rel_tol=1e-14)
    t, n = P.ray_cylinder([0.01, 0, -1], [0, 0, 1], [0, 0, 0], [0, 0, 0.1], 0.02)
    assert math.isclose(t, 1.0) and np.allclose(n, [0, 0, -1])
    t, n = P.ray_cone([-1, 0, 0.05], [1, 0, 0], [0, 0, 0], [0, 0, 0.1], 0.02, 0.0)
    assert math.isclose(t, 0.99, rel_tol=1e-14)
    assert np.allclose(n, tf.unit([-1.0, 0.0, 0.2]), atol=1e-12)
    # from inside the cylinder the hit is the exit
    t, n = P.ray_cylinder([0, 0, 0.05], [1, 0, 0], [0, 0, 0], [0, 0, 0.1], 0.02)
    assert math.isclose(t, 0.02, rel_tol=1e-13) and np.allclose(n, [1, 0, 0])


def test_ray_torus_matches_quartic_reference():
    R, r = 0.05, 0.01
    for _ in range(200):
        o = RNG.uniform(-0.1, 0.1, 3)
        d = tf.unit(RNG.normal(size=3))
        hit = P.ray_torus(o, d, [0, 0, 0], [0, 0, 1], R, r)
        # reference: the torus quartic in t by numpy.roots
        al = float(o @ d)
        be = float(o @ o) + R * R - r * r
        e = d[0] ** 2 + d[1] ** 2
        f = o[0] * d[0] + o[1] * d[1]
        g = o[0] ** 2 + o[1] ** 2
        coeffs = [1.0, 4 * al, 4 * al * al + 2 * be - 4 * R * R * e, 4 * al * be - 8 * R * R * f, be * be - 4 * R * R * g]
        ref = P.poly_roots_ref(coeffs, 0.0, np.inf)
        if len(ref) == 0:
            assert hit is None
        else:
            assert hit is not None and math.isclose(hit[0], ref[0], rel_tol=1e-9, abs_tol=1e-12)
            p = o + hit[0] * d
            rho = math.hypot(p[0], p[1])
            assert math.isclose((rho - R) ** 2 + p[2] ** 2, r * r, rel_tol=1e-7)
    t, n = P.ray_torus([0.05, 0, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], R, r)
    assert math.isclose(t, 0.99, rel_tol=1e-12) and np.allclose(n, [0, 0, 1])
    assert P.ray_torus([0, 0, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], R, r) is None


def test_ray_revolved_ball_end():
    z, r, b = ball_profile()
    R = 0.005
    for x in (0.0, 0.001, 0.003, 0.0049):
        t, n = P.ray_revolved([x, 0, -1], [0, 0, 1], z, r, b)
        assert math.isclose(t, 1.0 + R - math.sqrt(R * R - x * x), rel_tol=1e-12)
        expected = tf.unit([x, 0, -(math.sqrt(R * R - x * x))])
        assert np.allclose(n, expected, atol=1e-9)
    # placed by a pose: axis along +x, tip at (0.1, 0, 0)
    T = tf.pose(tf.rot_y(math.pi / 2), [0.1, 0, 0])
    t, n = P.ray_revolved([0, 0, 0], [1, 0, 0], z, r, b, pose=T)
    assert math.isclose(t, 0.1, rel_tol=1e-12) and np.allclose(n, [-1, 0, 0], atol=1e-12)
    # side of the shank
    t, n = P.ray_revolved([1, 0, 0.02], [-1, 0, 0], z, r, b)
    assert math.isclose(t, 1 - R, rel_tol=1e-12) and np.allclose(n, [1, 0, 0])


def test_ray_kernels_agree_with_tessellated_meshes():
    from cnctwin.geom import bvh

    tol = 2e-6
    mesh = P.torus_mesh([0, 0, 0], [0, 0, 1], 0.05, 0.01, tol)
    tree = bvh.build_tree(mesh)
    hits = 0
    for _ in range(200):
        phi = RNG.uniform(0, 2 * math.pi)
        o = np.array([0.05 * math.cos(phi), 0.05 * math.sin(phi), 0.2]) + RNG.uniform(-0.008, 0.008, 3)
        d = tf.unit(np.array([0, 0, -1.0]) + RNG.normal(scale=0.02, size=3))
        a = P.ray_torus(o, d, [0, 0, 0], [0, 0, 1], 0.05, 0.01)
        m = bvh.ray_first_hit(tree, o, d)
        if a is None or m is None:
            continue
        hits += 1
        # the mesh lies within the chord tolerance inside the surface: the hit moves by tol / cos(incidence)
        cosi = abs(float(np.dot(a[1], d)))
        if cosi > 0.3:
            assert abs(a[0] - m[0]) <= 3 * tol / cosi
    assert hits > 150


def test_mesh_from_machine_primitives():
    from cnctwin.db.simap import SIMap

    box = P.mesh_from_primitive({"box": {"min_mm": [-457, -178, -80], "max_mm": [457, 178, 0]}})
    assert math.isclose(box.volume(), 0.914 * 0.356 * 0.08, rel_tol=1e-12)
    cyl = P.mesh_from_primitive(SIMap({"cylinder": {"p0_mm": [0, 0, 508], "p1_mm": [0, 0, 700], "radius_mm": 90}}))
    lo, hi = cyl.bounds()
    assert math.isclose(lo[2], 0.508) and math.isclose(hi[2], 0.7) and cyl.is_watertight()
    cone = P.mesh_from_primitive({"cone": {"p0_mm": [0, 0, 0], "p1_mm": [0, 0, 50], "r0_mm": 20, "r1_mm": 10}})
    assert cone.is_watertight()
    sph = P.mesh_from_primitive({"sphere": {"centre_mm": [0, 0, 0], "radius_mm": 5}})
    assert sph.is_watertight()
    with pytest.raises(ValueError):
        P.mesh_from_primitive({"blob": {}})


def test_each_kernel_has_one_signature():
    for kernel in P.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
