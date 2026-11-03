"""Mesh and CAD files: STL, OBJ, 3MF, PLY, GLB, STEP (docs/models/geom.md "Files").

:func:`read_mesh` / :func:`write_mesh` choose the format by the file suffix. Meshes are in metres; STL,
OBJ and PLY carry no unit (``unit``, default mm), 3MF and STEP carry their own, GLB is metres by
definition. STEP needs OpenCASCADE and is imported by :mod:`cnctwin.geom.io.step` (lazily, it is slow
to load).
"""

from __future__ import annotations

from pathlib import Path

from ..mesh import Mesh
from .gltf import GlbNode, GlbScene, glb_bytes, read_glb, write_glb
from .obj import obj_text, read_obj, write_obj
from .ply import ply_bytes, read_ply, write_ply
from .stl import read_stl, stl_bytes, write_stl
from .threemf import read_3mf, threemf_bytes, write_3mf
from .units import unit_name, unit_scale

MESH_SUFFIXES = (".stl", ".obj", ".3mf", ".ply", ".glb", ".step", ".stp")


def read_mesh(path, unit: str = "mm", chordal_tolerance_m: float | None = None) -> Mesh:
    """Mesh of a file by its suffix (``unit`` for the unit-less formats; STEP tessellated at
    ``chordal_tolerance_m``, B-rep face ids in ``face_group``)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return read_stl(path, unit)
    if suffix == ".obj":
        return read_obj(path, unit)
    if suffix == ".ply":
        return read_ply(path, unit)
    if suffix == ".3mf":
        return read_3mf(path, merge_items=True)
    if suffix == ".glb":
        return read_glb(path).mesh(world=True)
    if suffix in (".step", ".stp"):
        from .step import read_step

        return read_step(path, chordal_tolerance_m=chordal_tolerance_m).mesh
    raise ValueError(f"unknown mesh file type {suffix!r} (one of {', '.join(MESH_SUFFIXES)})")


def write_mesh(path, mesh: Mesh, unit: str = "mm") -> None:
    """Write a mesh by suffix: .stl (binary), .obj, .ply (binary), .3mf (``unit`` as the model unit), .glb."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        write_stl(path, mesh, unit)
    elif suffix == ".obj":
        write_obj(path, mesh, unit)
    elif suffix == ".ply":
        write_ply(path, mesh, unit)
    elif suffix == ".3mf":
        write_3mf(path, mesh, unit={"mm": "millimeter", "m": "meter", "in": "inch", "cm": "centimeter",
                                     "um": "micron", "ft": "foot"}.get(unit, unit))
    elif suffix == ".glb":
        write_glb(path, mesh)
    else:
        raise ValueError(f"cannot write {suffix!r} (stl, obj, ply, 3mf, glb)")


__all__ = ["read_mesh", "write_mesh", "read_stl", "write_stl", "stl_bytes", "read_obj", "write_obj", "obj_text",
           "read_ply", "write_ply", "ply_bytes", "read_3mf", "write_3mf", "threemf_bytes", "GlbNode", "GlbScene",
           "glb_bytes", "read_glb", "write_glb", "unit_scale", "unit_name", "MESH_SUFFIXES"]
