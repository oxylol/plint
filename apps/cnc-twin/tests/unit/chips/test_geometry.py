"""Geometry providers of the pile base: the analytic block with removed boxes, and fixture meshes through
cnctwin.geom.bvh."""

from __future__ import annotations

import numpy as np

from cnctwin.chips import abi
from cnctwin.chips.protocols import BoxScene, ChipGeometry, MeshSurfaces, StockFaces
from cnctwin.geom.primitives import box_mesh


def test_box_scene_heights_kinds_and_removals():
    scene = BoxScene(np.zeros(3), np.array([0.1, 0.08, 0.03]))
    assert isinstance(scene, ChipGeometry)
    scene.remove_box((0.02, 0.02, 0.01), (0.04, 0.04, 0.031))
    h, k = scene.surface_height(0, 2, 1, np.array([0.03, 0.05, 0.2]), np.array([0.03, 0.05, 0.2]))
    assert np.allclose(h, [0.01, 0.03, 0.0]) and list(k) == [abi.SURF_PART, abi.SURF_PART, abi.SURF_TABLE]
    assert scene.version(0) == 1 and scene.changed_box_S(0) is not None and scene.changed_box_S(0) is None
    # looking along -x (sign -1): the highest surface is the face at x = 0 (height = -x)
    h, _ = scene.surface_height(0, 0, -1, np.array([0.04]), np.array([0.02]))
    assert np.allclose(h, [0.0])


def test_mesh_surfaces_add_a_vise_body_around_the_part():
    part = BoxScene(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.08, 0.03]), table_half_size_m=0.0)
    vise = box_mesh(np.array([-0.05, -0.02, -0.08]), np.array([0.15, 0.10, 0.01]))
    geo = MeshSurfaces(part, [(vise, abi.SURF_FIXTURE)])
    h, k = geo.surface_height(0, 2, 1, np.array([0.05, -0.03, 0.12]), np.array([0.04, 0.04, 0.09]))
    assert np.allclose(h, [0.03, 0.01, 0.01], atol=1e-12)
    assert list(k) == [abi.SURF_PART, abi.SURF_FIXTURE, abi.SURF_FIXTURE]
    h, k = geo.surface_height(0, 2, 1, np.array([0.3]), np.array([0.3]))
    assert h[0] == -np.inf and k[0] == abi.SURF_NONE


def test_stock_faces_cover_the_six_faces():
    faces = StockFaces(np.zeros(3), np.array([0.1, 0.08, 0.03])).faces_S(0)
    assert [f[0] for f in faces] == ["S+x", "S-x", "S+y", "S-y", "S+z", "S-z"]
    top = faces[4]
    assert np.allclose(top[1], [0.05, 0.04, 0.03]) and np.allclose(top[2], [0, 0, 1]) and np.allclose(top[3], [0.05, 0.04])
