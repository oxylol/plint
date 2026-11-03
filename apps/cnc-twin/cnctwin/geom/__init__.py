"""Geometry library: transforms and control conventions, meshes, primitives, file formats (STL, OBJ, 3MF,
PLY, GLB, STEP with PMI), bounding-volume trees and their queries, overlap tests and grid traversal
(docs/models/geom.md).

Submodules load on first use (each compiles or loads its Numba kernels when imported):

* :mod:`~cnctwin.geom.transform` poses, rotations, Euler conventions (Fanuc G68.2, Siemens CYCLE800,
  Heidenhain PLANE), quaternions and slerp
* :mod:`~cnctwin.geom.mesh` the :class:`Mesh` type
* :mod:`~cnctwin.geom.primitives` box, cylinder, cone, sphere, torus, revolved profiles; analytic rays
* :mod:`~cnctwin.geom.io` files; :mod:`~cnctwin.geom.io.step` STEP import (OpenCASCADE) and PMI
* :mod:`~cnctwin.geom.cad`, :mod:`~cnctwin.geom.selectors` CAD models and face selectors
* :mod:`~cnctwin.geom.bvh` trees, rays, closest point, signed distance, point in mesh
* :mod:`~cnctwin.geom.overlap` triangle/box/mesh/revolved-solid interference
* :mod:`~cnctwin.geom.dda` 3D-DDA grid traversal
* :mod:`~cnctwin.geom.abi` kernel layouts (MeshTree, Grid); :mod:`~cnctwin.geom.reference` plain-Python references

``from cnctwin.geom import Mesh, build_tree, ...`` works for the names listed in ``__all__``.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Mesh": "mesh", "merge": "mesh",
    "MeshTree": "abi", "Grid": "abi",
    "build_tree": "bvh", "static_tree": "bvh", "ray_first_hit": "bvh", "ray_all_hits": "bvh", "segment_hit": "bvh",
    "rays_first_hit": "bvh", "rays_all_hits": "bvh", "segments_hit": "bvh", "closest_point": "bvh",
    "signed_distance": "bvh", "points_in_mesh": "bvh", "winding_numbers": "bvh", "save_tree": "bvh", "load_tree": "bvh",
    "mesh_mesh_overlap": "overlap", "mesh_box_overlap": "overlap", "mesh_revolved_overlap": "overlap",
    "tri_tri_overlap": "overlap", "tri_box_overlap": "overlap", "box_box_overlap": "overlap",
    "box_mesh": "primitives", "cylinder_mesh": "primitives", "cone_mesh": "primitives", "sphere_mesh": "primitives",
    "torus_mesh": "primitives", "revolved_mesh": "primitives", "mesh_from_primitive": "primitives",
    "profile_polyline": "primitives",
    "read_mesh": "io", "write_mesh": "io", "read_glb": "io", "write_glb": "io", "glb_bytes": "io", "GlbNode": "io",
    "CadModel": "cad", "FaceInfo": "cad", "select_faces": "selectors", "selector_for_face": "selectors",
    "pose_from_dict": "transform", "pose_to_dict": "transform",
    "ray_cells": "dda", "segment_cells": "dda", "first_occupied": "dda",
}
_SUBMODULES = ("abi", "bvh", "cad", "dda", "io", "mesh", "overlap", "primitives", "reference", "selectors", "settings",
               "transform")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _EXPORTS:
        module = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def read_step(path, chordal_tolerance_m=None, angular_tolerance_rad=None, isolated: bool = False):
    """CAD model of a STEP file (``isolated`` runs OpenCASCADE in a child process)."""
    from .io import step

    if isolated:
        return step.read_step_isolated(path, chordal_tolerance_m, angular_tolerance_rad)
    return step.read_step(path, chordal_tolerance_m, angular_tolerance_rad)


__all__ = sorted(set(_EXPORTS) | {"read_step"})
