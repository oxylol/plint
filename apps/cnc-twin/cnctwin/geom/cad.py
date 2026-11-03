"""A CAD part after import: the tessellated mesh with B-rep face ids, each face's analytic surface, the
features (adjacent faces on one surface) and the PMI (docs/models/geom.md "STEP import").

:class:`CadModel` is plain data (no OpenCASCADE objects), so the STEP import can run in a separate process
and hand it back through :meth:`CadModel.save` / :meth:`CadModel.load` (``.npz`` for the mesh, ``.json``
for the rest). All values are SI except ``pmi``, which is in the project file's own form (mm, deg, the
``cad.datums`` / ``cad.tolerances`` shape of db/_templates/project.yaml) so it can be merged with the
hand-given entries.

Surface parameters (``FaceInfo.params``, metres and radians, CAD frame):

* plane: ``point_m``, ``normal`` (outward: out of the material)
* cylinder: ``axis_point_m``, ``axis`` (unit, sign canonical: largest component positive), ``radius_m``
* cone: ``apex_m``, ``axis`` (from the apex into the cone), ``semi_angle_rad``, ``ref_point_m``, ``ref_radius_m``
* sphere: ``centre_m``, ``radius_m``
* torus: ``centre_m``, ``axis`` (canonical sign), ``major_radius_m``, ``minor_radius_m``
* bspline / bezier: ``u_degree``, ``v_degree``, ``n_poles_u``, ``n_poles_v``, ``rational``, ``periodic_u``,
  ``periodic_v``; revolution / extrusion / offset / other: ``{}`` (the mesh carries the geometry)

``concave`` is True for a curved face whose material lies outside its surface (a hole, an inner fillet),
False for a convex one (a boss, an outer round), None for planes and freeform faces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .mesh import Mesh

ANALYTIC = ("plane", "cylinder", "cone", "sphere", "torus")
FREEFORM = ("bspline", "bezier", "revolution", "extrusion", "offset", "other")


@dataclass
class FaceInfo:
    id: int
    surface: str
    params: dict = field(default_factory=dict)
    concave: bool | None = None
    reversed: bool = False
    area_m2: float = 0.0
    point_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds_m: list = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    feature: int = -1
    name: str = ""
    n_triangles: int = 0

    @property
    def selector_type(self) -> str:
        """The face type a project selector names (``freeform`` for every non-analytic surface)."""
        return self.surface if self.surface in ANALYTIC else "freeform"


@dataclass
class CadModel:
    mesh: Mesh
    faces: list = field(default_factory=list)
    features: list = field(default_factory=list)
    pmi: dict = field(default_factory=lambda: {"datums": [], "tolerances": []})
    pmi_raw: list = field(default_factory=list)
    pmi_unmapped: list = field(default_factory=list)
    unit: dict = field(default_factory=lambda: {"file_unit": "mm", "file_scale_m": 1e-3})
    source: str = ""
    chordal_tolerance_m: float = 0.0
    angular_tolerance_rad: float = 0.0
    names: dict = field(default_factory=dict)

    def face(self, face_id: int) -> FaceInfo:
        return self.faces[int(face_id)]

    def feature_faces(self, face_id: int) -> list[int]:
        """Every face of the feature ``face_id`` belongs to (sorted)."""
        f = self.faces[int(face_id)].feature
        return list(self.features[f]) if f >= 0 else [int(face_id)]

    def face_mesh(self, face_id: int) -> Mesh:
        return self.mesh.group(int(face_id))

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for f in self.faces:
            counts[f.surface] = counts.get(f.surface, 0) + 1
        return {"faces": len(self.faces), "features": len(self.features), "surfaces": dict(sorted(counts.items())),
                "triangles": int(self.mesh.n_faces), "datums": len(self.pmi.get("datums", [])),
                "tolerances": len(self.pmi.get("tolerances", [])), "unmapped_pmi": len(self.pmi_unmapped),
                "unit": self.unit, "source": self.source}

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return {"format": "cnctwin-cad-model", "version": 1, "faces": [asdict(f) for f in self.faces],
                "features": [list(map(int, f)) for f in self.features], "pmi": self.pmi, "pmi_raw": self.pmi_raw,
                "pmi_unmapped": self.pmi_unmapped, "unit": self.unit, "source": self.source,
                "chordal_tolerance_m": self.chordal_tolerance_m, "angular_tolerance_rad": self.angular_tolerance_rad,
                "names": self.names, "mesh_name": self.mesh.name,
                "group_names": {str(k): v for k, v in self.mesh.group_names.items()}}

    def save(self, path) -> tuple[Path, Path]:
        """Write ``<path>.npz`` (vertices, faces, face_group) and ``<path>.json``; returns both paths."""
        base = Path(path)
        base = base.with_suffix("") if base.suffix in (".npz", ".json") else base
        npz = base.with_name(base.name + ".npz")
        js = base.with_name(base.name + ".json")
        np.savez(npz, vertices=self.mesh.vertices, faces=self.mesh.faces, face_group=self.mesh.face_group)
        js.write_text(json.dumps(self.to_json(), sort_keys=True, indent=1), encoding="utf-8")
        return npz, js

    @classmethod
    def load(cls, path) -> "CadModel":
        base = Path(path)
        base = base.with_suffix("") if base.suffix in (".npz", ".json") else base
        data = json.loads(base.with_name(base.name + ".json").read_text(encoding="utf-8"))
        if data.get("format") != "cnctwin-cad-model":
            raise ValueError(f"{base}: not a cnctwin CAD model")
        with np.load(base.with_name(base.name + ".npz")) as arrays:
            mesh = Mesh(arrays["vertices"], arrays["faces"], arrays["face_group"],
                        group_names={int(k): v for k, v in data.get("group_names", {}).items()},
                        name=data.get("mesh_name", ""))
        faces = [FaceInfo(**f) for f in data["faces"]]
        return cls(mesh, faces, [list(f) for f in data["features"]], data["pmi"], data["pmi_raw"],
                   data["pmi_unmapped"], data["unit"], data["source"], data["chordal_tolerance_m"],
                   data["angular_tolerance_rad"], data.get("names", {}))


__all__ = ["CadModel", "FaceInfo", "ANALYTIC", "FREEFORM"]
