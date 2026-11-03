"""The result of one operation as geometry: the material configuration and the configurations derived from it
(docs/models/inspect.md "Result geometry"; architecture 4.1).

removal meshes its dexel field in the material configuration X (20 degC, unclamped, before machining-induced
stress redistribution); structure publishes the displacement d(X) on node grids in S
(:class:`cnctwin.core.bus.Displacement`: thermal, clamp, residual). A :class:`ResultPart` holds the X mesh and
the named grids and builds every configuration by moving each vertex to ``X + d(X)`` (trilinear in the grid,
clamped at its border - the rule removal's meshing uses, so the meshes agree vertex for vertex):

=============  =================================================  =====================================
configuration  displacement                                       what it is
=============  =================================================  =====================================
``material``   none                                               X: what the tools left, in S
``as_cut``     thermal + clamp + residual (end of cutting)        the part in the machine, hot, clamped
``cold``       clamp + residual                                   clamped, at 20 degC
``unload``     unload_thermal + final                             unclamped at the unload temperature
                                                                  (``part_<op>_unload.stl``)
``final``      final (residual after unclamping, at 20 degC)      spring-back and distortion: what is
                                                                  inspected (``part_<op>.*``)
=============  =================================================  =====================================

A grid that is not given counts as zero and is listed in :meth:`ResultPart.assumptions` (so a report says the
inspected geometry has, for example, no spring-back). Precomputed configuration meshes (``meshes``) override
the construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geom.mesh import Mesh
from ..units import REFERENCE_TEMP_K
from . import abi as A
from .kernels import trilinear_nb

CONFIGURATIONS = ("material", "as_cut", "cold", "unload", "final")
GRIDS = ("thermal", "clamp", "residual", "unload_thermal", "final")
COMPOSITION = {"material": (), "as_cut": ("thermal", "clamp", "residual"), "cold": ("clamp", "residual"),
               "unload": ("unload_thermal", "final"), "final": ("final",)}
INSPECTED = "final"


@dataclass
class DisplacementGrid:
    """d(X) at the nodes ``origin + (i, j, k) cell`` of a grid in S (layout of core.bus.Displacement)."""

    origin_S_m: np.ndarray
    cell_m: float
    grid_m: np.ndarray                  # [nx, ny, nz, 3]

    def __post_init__(self) -> None:
        self.origin_S_m = A.contiguous(np.asarray(self.origin_S_m, dtype=np.float64).reshape(3))
        self.cell_m = float(self.cell_m)
        grid = np.asarray(self.grid_m, dtype=np.float64)
        if grid.size == 0:
            grid = np.zeros((0, 0, 0, 3))
        if grid.ndim != 4 or grid.shape[-1] != 3:
            raise ValueError(f"a displacement grid is [nx, ny, nz, 3], got {grid.shape}")
        if grid.size and self.cell_m <= 0.0:
            raise ValueError("a displacement grid needs a positive cell size")
        self.grid_m = A.contiguous(grid)

    @classmethod
    def from_bus(cls, displacement, component: str = "total_m") -> "DisplacementGrid":
        """One component (``total_m``, ``thermal_m``, ``clamp_m``, ``residual_m``) of a bus Displacement record."""
        return cls(np.asarray(displacement.origin_S_m), float(displacement.cell_m),
                   np.asarray(getattr(displacement, component)))

    @property
    def empty(self) -> bool:
        return self.grid_m.size == 0

    def at(self, points) -> np.ndarray:
        P = A.contiguous(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        out = np.zeros_like(P)
        if len(P):
            trilinear_nb(P, self.origin_S_m, self.cell_m, self.grid_m, out)
        return out

    def max_norm_m(self) -> float:
        return float(np.max(np.linalg.norm(self.grid_m, axis=-1))) if self.grid_m.size else 0.0


@dataclass
class ResultPart:
    """One operation's result of one workpiece (see the module documentation)."""

    op: str
    material: Mesh
    grids: dict = field(default_factory=dict)          # name in GRIDS -> DisplacementGrid
    op_index: int = 0
    workpiece: int = 0
    meshes: dict = field(default_factory=dict)         # configuration -> Mesh (precomputed, overrides)
    inspection_temp_K: float = REFERENCE_TEMP_K
    unload_time_s: float | None = None
    unload_mean_temp_K: float | None = None
    unload_max_temp_K: float | None = None
    stock_kind: str = "block"
    notes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.grids) - set(GRIDS))
        if unknown:
            raise ValueError(f"unknown displacement grids {unknown} (known: {', '.join(GRIDS)})")
        unknown = sorted(set(self.meshes) - set(CONFIGURATIONS))
        if unknown:
            raise ValueError(f"unknown configurations {unknown} (known: {', '.join(CONFIGURATIONS)})")

    # ------------------------------------------------------------------ displacements
    def grid(self, name: str) -> DisplacementGrid | None:
        g = self.grids.get(name)
        return None if g is None or g.empty else g

    def displacement(self, configuration: str, points) -> np.ndarray:
        """d(X) of a configuration at material points X (the sum of its grids)."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.zeros_like(P)
        for name in COMPOSITION[configuration]:
            g = self.grid(name)
            if g is not None:
                out += g.at(P)
        return out

    def component(self, name: str, points) -> np.ndarray:
        """One grid's displacement at material points (zero when absent)."""
        g = self.grid(name)
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return g.at(P) if g is not None else np.zeros_like(P)

    def configuration(self, name: str) -> Mesh:
        """The mesh of a configuration (vertex attributes of X kept; ``X_m`` holds the material positions)."""
        if name not in CONFIGURATIONS:
            raise ValueError(f"unknown configuration {name!r} (known: {', '.join(CONFIGURATIONS)})")
        if name in self.meshes:
            return self.meshes[name]
        mesh = self.material.copy()
        mesh.name = f"part_{self.op}_{name}"
        mesh.vertex_attr["X_m"] = self.material.vertices.copy()
        if COMPOSITION[name]:
            mesh.vertices = A.contiguous(self.material.vertices + self.displacement(name, self.material.vertices))
        return mesh

    def material_points(self, mesh: Mesh) -> np.ndarray:
        """X of each vertex of a configuration mesh made by :meth:`configuration` (its own vertices otherwise)."""
        X = mesh.vertex_attr.get("X_m")
        return np.asarray(X, dtype=np.float64) if X is not None else mesh.vertices

    def assumptions(self, configuration: str = INSPECTED) -> list[str]:
        if configuration in self.meshes:
            return []
        missing = [g for g in COMPOSITION[configuration] if self.grid(g) is None]
        return [f"{self.op}: no {g} displacement grid (taken as zero in the {configuration} geometry)"
                for g in missing]

    def unload_record(self) -> dict | None:
        if self.unload_mean_temp_K is None and self.unload_max_temp_K is None:
            return None
        return {"op": self.op, "workpiece": self.workpiece, "time_s": self.unload_time_s,
                "mean_temp_K": self.unload_mean_temp_K, "max_temp_K": self.unload_max_temp_K}

    def summary(self) -> dict:
        out = {"op": self.op, "op_index": self.op_index, "workpiece": self.workpiece,
               "vertices": int(self.material.n_vertices), "triangles": int(self.material.n_faces),
               "material_volume_m3": float(self.material.volume()), "grids": {}}
        for name in GRIDS:
            g = self.grid(name)
            out["grids"][name] = None if g is None else {"cell_m": g.cell_m, "shape": list(g.grid_m.shape[:3]),
                                                           "max_displacement_m": g.max_norm_m()}
        return out

    # ------------------------------------------------------------------ files
    def write_files(self, run_dir, formats=("glb", "stl", "3mf")) -> list[str]:
        """``part_<op>.<fmt>`` of the inspected (final) geometry and ``part_<op>_unload.stl`` (architecture 16),
        written atomically with deterministic bytes; returns the relative paths."""
        from ..core.rundir import write_bytes_atomic
        from ..geom.io import glb_bytes, stl_bytes, threemf_bytes

        run_dir = Path(run_dir)
        name = f"part_{self.op}" if self.workpiece == 0 else f"part_{self.op}_w{self.workpiece}"
        final = self.configuration(INSPECTED)
        plain = Mesh(final.vertices, final.faces, final.face_group, name=name)
        written = []
        for fmt in formats:
            if fmt == "glb":
                data = glb_bytes(plain)
            elif fmt == "stl":
                data = stl_bytes(plain, "mm")
            elif fmt == "3mf":
                data = threemf_bytes(plain, unit="millimeter")
            else:
                raise ValueError(f"part files are glb, stl or 3mf, not {fmt!r}")
            write_bytes_atomic(run_dir / f"{name}.{fmt}", data)
            written.append(f"{name}.{fmt}")
        unload = self.configuration("unload")
        write_bytes_atomic(run_dir / f"{name}_unload.stl",
                           stl_bytes(Mesh(unload.vertices, unload.faces, name=f"{name}_unload"), "mm"))
        written.append(f"{name}_unload.stl")
        return written


__all__ = ["ResultPart", "DisplacementGrid", "CONFIGURATIONS", "GRIDS", "COMPOSITION", "INSPECTED"]
