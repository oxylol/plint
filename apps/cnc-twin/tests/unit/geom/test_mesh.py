"""Mesh: measures against analytic values, topology checks, weld (kernel vs reference), merge, transforms;
and the geometry parameter file."""

import math

import numpy as np
import pytest

from cnctwin.geom import mesh as gm
from cnctwin.geom import transform as tf
from cnctwin.geom.mesh import Mesh, merge
from cnctwin.geom.primitives import box_mesh

RNG = np.random.default_rng(7)


def tetra() -> Mesh:
    v = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    f = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    return Mesh(v, f, name="tet")


def test_box_measures():
    m = box_mesh([0.0, 0.0, 0.0], [0.1, 0.08, 0.03])
    assert math.isclose(m.volume(), 0.1 * 0.08 * 0.03, rel_tol=1e-14)
    assert math.isclose(m.area(), 2 * (0.1 * 0.08 + 0.08 * 0.03 + 0.1 * 0.03), rel_tol=1e-14)
    assert np.allclose(m.centroid(), [0.05, 0.04, 0.015], atol=1e-16)
    assert np.allclose(m.centroid("area"), [0.05, 0.04, 0.015], atol=1e-16)
    lo, hi = m.bounds()
    assert np.allclose(lo, 0.0) and np.allclose(hi, [0.1, 0.08, 0.03])
    assert m.is_watertight()
    assert sorted(set(m.face_group.tolist())) == [0, 1, 2, 3, 4, 5]
    n = m.face_normals()
    assert np.allclose(n[m.face_group == 5], [0, 0, 1])
    assert np.allclose(n[m.face_group == 0], [-1, 0, 0])


def test_tetra_volume_centroid_and_normals():
    m = tetra()
    assert math.isclose(m.volume(), 1.0 / 6.0, rel_tol=1e-14)
    assert np.allclose(m.centroid(), [0.25, 0.25, 0.25])
    vn = m.vertex_normals()
    assert np.allclose(np.linalg.norm(vn, axis=1), 1.0)
    # angle-weighted normal at the right-angle corner is the normalised sum of the three axes
    assert np.allclose(vn[0], -np.ones(3) / math.sqrt(3.0))
    assert np.allclose(m.corner_angles().sum(axis=1), math.pi)


def test_watertight_detects_holes_flips_and_non_manifold():
    m = box_mesh([0, 0, 0], [1, 1, 1])
    assert m.edge_report()["boundary_edges"] == 0
    holed = m.submesh(np.arange(m.n_faces) != 3)
    assert not holed.is_watertight() and holed.edge_report()["boundary_edges"] == 3
    flipped = m.copy()
    flipped.faces[0] = flipped.faces[0][::-1]
    report = flipped.edge_report()
    assert report["inconsistent_edges"] == 3 and not flipped.is_watertight()
    extra = Mesh(np.vstack([m.vertices, [[2.0, 2.0, 2.0]]]), np.vstack([m.faces, [[0, 1, 8]]]))
    assert extra.edge_report()["non_manifold_edges"] == 1


def test_transformed_and_mirror_keep_outward_normals():
    m = box_mesh([0, 0, 0], [1, 2, 3])
    m.vertex_attr["normal"] = m.vertex_normals()
    T = tf.pose(tf.rot_z(0.4), [1, 2, 3])
    t = m.transformed(T)
    assert math.isclose(t.volume(), 6.0, rel_tol=1e-13)
    assert np.allclose(t.vertex_attr["normal"], tf.apply_dir(T, m.vertex_attr["normal"]))
    mirror = m.transformed(np.diag([-1.0, 1.0, 1.0, 1.0]))
    assert math.isclose(mirror.volume(), 6.0, rel_tol=1e-13) and mirror.is_watertight()
    assert math.isclose(m.flipped().volume(), -6.0, rel_tol=1e-13)


def test_weld_exact_and_tolerance():
    m = box_mesh([0, 0, 0], [1, 1, 1])
    soup = Mesh.from_triangles(m.triangles(), weld=False)
    assert soup.n_vertices == 36 and not soup.is_watertight()
    welded = soup.weld(0.0)
    assert welded.n_vertices == 8 and welded.is_watertight()
    jitter = soup.copy()
    jitter.vertices = jitter.vertices + RNG.uniform(-1e-9, 1e-9, jitter.vertices.shape)
    assert jitter.weld(0.0).n_vertices > 8
    w = jitter.weld(1e-6)
    assert w.n_vertices == 8 and w.is_watertight()
    # collapsing faces are dropped
    tiny = Mesh([[0, 0, 0], [1e-9, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    assert tiny.weld(1e-6).n_faces == 0 and tiny.weld(1e-6, drop_degenerate=False).n_faces == 1


def test_weld_kernel_matches_reference():
    v = RNG.uniform(0.0, 1e-3, (300, 3))
    v = np.vstack([v, v[:100] + RNG.uniform(-2e-7, 2e-7, (100, 3))])
    tol = 5e-7
    q = np.floor(v / tol).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0])).astype(np.int64)
    rep = np.empty(len(v), np.int64)
    gm.weld_representatives_nb(np.ascontiguousarray(v), tol, np.ascontiguousarray(q), order,
                               np.ascontiguousarray(q[order]), rep)
    assert np.array_equal(rep, gm.weld_representatives_ref(v, tol))


def test_merge_attributes_and_groups():
    a = box_mesh([0, 0, 0], [1, 1, 1])
    a.vertex_attr["t"] = np.arange(a.n_vertices, dtype=np.float64)
    b = tetra()
    b.face_attr["k"] = np.arange(b.n_faces, dtype=np.int32)
    m = merge([a, b], offset_groups=True)
    assert m.n_vertices == a.n_vertices + b.n_vertices and m.n_faces == a.n_faces + b.n_faces
    assert np.isnan(m.vertex_attr["t"][-1]) and m.face_attr["k"][0] == -1
    assert math.isclose(m.volume(), 1.0 + 1.0 / 6.0, rel_tol=1e-13)
    assert m.group_names[0] == "-x"
    with pytest.raises(ValueError):
        c = box_mesh([0, 0, 0], [1, 1, 1])
        c.group_names = {0: "other"}
        merge([a, c])


def test_submesh_group_compact_and_hash():
    m = box_mesh([0, 0, 0], [1, 1, 1])
    top = m.group(5)
    assert top.n_faces == 2 and top.n_vertices == 4 and math.isclose(top.area(), 1.0)
    assert m.content_hash() == box_mesh([0, 0, 0], [1, 1, 1]).content_hash()
    assert m.content_hash() != box_mesh([0, 0, 0], [1, 1, 2]).content_hash()


def test_invalid_meshes_are_refused():
    with pytest.raises(ValueError):
        Mesh([[0, 0, 0]], [[0, 1, 2]])
    with pytest.raises(ValueError):
        Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], face_group=[0, 1])
    with pytest.raises(ValueError):
        Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], vertex_attr={"x": np.zeros(2)})


def test_geom_parameters_load_in_si():
    from cnctwin.db.models import params

    p = params("geom")
    assert math.isclose(p["tessellation_chordal_tolerance_m"], 1e-5)
    assert math.isclose(p["selector_angle_tolerance_rad"], math.radians(1.0))
    for key in ("step_chordal_tolerance_m", "step_angular_tolerance_rad", "bvh_leaf_size", "bvh_bins",
                "bvh_traversal_cost", "selector_radius_tolerance_m", "feature_merge_distance_m",
                "feature_merge_angle_rad", "clearance_tolerance_m", "point_in_mesh_rays",
                "tessellation_min_segments"):
        assert key in p


def test_each_kernel_has_one_signature():
    for kernel in gm.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
