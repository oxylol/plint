"""Structure domain: tool-holder-spindle modes by receptance coupling of Timoshenko beams, workpiece modes and
compliance on a voxel finite-element grid (thin walls by analytic plates, long turned parts as beams),
displacement grids (thermal, clamping, residual-stress release) and fixture compliance
(docs/models/structure.md).

Names are imported on first use, so ``cnctwin.structure.schema`` (read by the project schema) does not load the
solvers or compile the kernels.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "StructureDomain": "domain", "StructureState": "state", "StructureSettings": "schema", "settings": "schema",
    "PHYSICS_SCHEMA": "schema", "REFRESH_LATENCY_MACRO": "schema",
    "tool_modes": "toolmodel", "static_deflection": "toolmodel", "section_stresses": "toolmodel",
    "tip_frf": "toolmodel", "VoxelGrid": "voxel", "VoxelModel": "voxel", "choose_grid": "voxel",
    "Plate": "thinwall", "treuting_read_curvature": "displacement", "BoxGeometry": "protocols",
    "CylinderGeometry": "protocols", "WorkGeometry": "protocols", "NodeGrid": "abi", "StructureInputs": "abi",
    "sample_node_grid": "kernels", "material_points": "kernels",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'cnctwin.structure' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
