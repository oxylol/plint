"""Result-surface points and the CAD faces they belong to (docs/models/inspect.md "Point assignment").

Every vertex of the result mesh is a point of the machined (or stock) surface. Its closest point on the CAD
gives the signed deviation and the B-rep face; it is **usable** for that face's fits and characteristics when

* it is within ``assign_band_m`` of the CAD (farther: stock allowance, a gouge, another feature),
* its own surface normal is within ``assign_angle_rad`` of the CAD normal at the closest point (a point of the
  neighbouring wall, a chamfer the CAD does not have, a burr), and
* its closest CAD feature is not on a face boundary (an edge or corner shared with another face).

Each vertex carries the area of its third of the adjacent triangles (``weight_m2``), the material position X
(when the mesh comes from :meth:`cnctwin.inspect.result.ResultPart.configuration`) and the surface event that
generated it (removal's ``surface_event`` vertex attribute: 0 the stock surface, >= 1 a cut). Dense
evaluations thin a face's usable points to ``max_points_per_face`` by a deterministic stride in vertex order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geom.mesh import Mesh
from .cadref import CadReference
from .schema import InspectSettings


@dataclass
class PointSet:
    """Points of one or more faces with their nominal outward normals and per-point attributes."""

    points: np.ndarray                          # [n, 3] S
    normals: np.ndarray                         # [n, 3] nominal outward normals (CAD)
    faces: np.ndarray                           # [n] B-rep face of each point
    weights: np.ndarray                         # [n] area (m^2) or 1 for probe points
    material: np.ndarray | None = None          # [n, 3] material positions X (for the error budget)
    events: np.ndarray | None = None            # [n] surface event ids (-1 unknown)
    components: dict = field(default_factory=dict)   # name -> [n, 3] displacement contributions

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, mask) -> "PointSet":
        idx = np.flatnonzero(np.asarray(mask)) if np.asarray(mask).dtype == bool else np.asarray(mask, np.int64)
        return PointSet(self.points[idx], self.normals[idx], self.faces[idx], self.weights[idx],
                        None if self.material is None else self.material[idx],
                        None if self.events is None else self.events[idx],
                        {k: v[idx] for k, v in self.components.items()})

    def shifted(self, delta: np.ndarray) -> "PointSet":
        """The same points moved by ``delta`` [n, 3] (removing one error component)."""
        return PointSet(self.points - delta, self.normals, self.faces, self.weights, self.material, self.events,
                        self.components)

    def transformed(self, T: np.ndarray) -> "PointSet":
        R, t = T[:3, :3], T[:3, 3]
        return PointSet(self.points @ R.T + t, self.normals @ R.T, self.faces, self.weights,
                        self.material, self.events, {k: v @ R.T for k, v in self.components.items()})

    @staticmethod
    def concat(sets: list) -> "PointSet":
        sets = [s for s in sets if s is not None and len(s)]
        if not sets:
            return PointSet(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0))
        names = set(sets[0].components)
        for s in sets[1:]:
            names &= set(s.components)
        mat = None if any(s.material is None for s in sets) else np.concatenate([s.material for s in sets])
        ev = None if any(s.events is None for s in sets) else np.concatenate([s.events for s in sets])
        return PointSet(np.concatenate([s.points for s in sets]), np.concatenate([s.normals for s in sets]),
                        np.concatenate([s.faces for s in sets]), np.concatenate([s.weights for s in sets]), mat, ev,
                        {k: np.concatenate([s.components[k] for s in sets]) for k in sorted(names)})


def vertex_weights(mesh: Mesh) -> np.ndarray:
    """A third of the area of every triangle at a vertex (sums to the surface area)."""
    w = np.zeros(mesh.n_vertices)
    a = mesh.face_areas() / 3.0
    for k in range(3):
        np.add.at(w, mesh.faces[:, k], a)
    return w


@dataclass
class Assignment:
    """The result mesh's vertices against the CAD (see the module documentation)."""

    points: np.ndarray                  # [n, 3]
    deviation_m: np.ndarray             # [n] signed distance to the CAD (+ excess, - missing)
    closest: np.ndarray                 # [n, 3]
    triangle: np.ndarray                # [n] CAD triangle
    face: np.ndarray                    # [n] B-rep face
    boundary: np.ndarray                # [n] bool, closest feature on a face boundary
    normal_ok: np.ndarray               # [n] bool
    usable: np.ndarray                  # [n] bool
    weights: np.ndarray                 # [n] m^2
    vertex_normals: np.ndarray          # [n, 3] of the result surface
    material: np.ndarray | None = None
    events: np.ndarray | None = None
    components: dict = field(default_factory=dict)

    def face_points(self, cad: CadReference, faces, max_points: int | None = None, usable_only: bool = True) -> PointSet:
        """The (usable) points of a set of faces, thinned to ``max_points`` per face, with their nominal normals."""
        sets = []
        for f in sorted(int(x) for x in faces):
            mask = (self.face == f) & (self.usable if usable_only else True)
            idx = np.flatnonzero(mask)
            if max_points is not None and len(idx) > max_points:
                step = int(math.ceil(len(idx) / max_points))
                idx = idx[::step]
            if not len(idx):
                continue
            _, n = cad.face_distance(f, self.points[idx])
            sets.append(PointSet(self.points[idx], n, np.full(len(idx), f, np.int64), self.weights[idx],
                                 None if self.material is None else self.material[idx],
                                 None if self.events is None else self.events[idx],
                                 {k: v[idx] for k, v in self.components.items()}))
        return PointSet.concat(sets)

    def machined_fraction(self, faces) -> float | None:
        """Share (by area) of a face set's usable points that a cut generated (None without event ids)."""
        if self.events is None:
            return None
        mask = np.isin(self.face, np.asarray(sorted(faces), np.int64)) & self.usable
        total = float(self.weights[mask].sum())
        if total <= 0.0:
            return None
        return float(self.weights[mask & (self.events >= 1)].sum()) / total


def assign(mesh: Mesh, cad: CadReference, settings: InspectSettings, material: np.ndarray | None = None,
           components: dict | None = None) -> Assignment:
    """Assign every vertex of a result mesh to the CAD (see the module documentation)."""
    P = np.asarray(mesh.vertices, dtype=np.float64)
    d, tri, q, feat = cad.closest(P)
    face = cad.face_of(tri)
    boundary = cad.is_boundary(tri, feat)
    vn = mesh.vertex_normals()
    cad_n = np.zeros_like(P)
    ok = tri >= 0
    for f in np.unique(face[ok]).tolist():
        sel = np.flatnonzero(face == f)
        _, n = cad.face_distance(int(f), P[sel])
        cad_n[sel] = n
    cos_lim = math.cos(settings.assign_angle_rad)
    normal_ok = ok & (np.einsum("ij,ij->i", vn, cad_n) >= cos_lim)
    usable = ok & normal_ok & ~boundary & (np.abs(np.nan_to_num(d, nan=np.inf)) <= settings.assign_band_m)
    events = mesh.vertex_attr.get("surface_event")
    return Assignment(P, d, q, tri, face, boundary, normal_ok, usable, vertex_weights(mesh), vn,
                      None if material is None else np.asarray(material, dtype=np.float64),
                      None if events is None else np.asarray(events, dtype=np.int64).reshape(-1),
                      dict(components or {}))


__all__ = ["PointSet", "Assignment", "assign", "vertex_weights"]
