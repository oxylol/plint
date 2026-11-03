"""The CAD part in the stock frame S: its mesh and bounding-volume tree, every B-rep face's analytic surface, the
face boundaries and adjacency, and exact distances to the nominal surfaces (docs/models/inspect.md "CAD").

The CAD model (:class:`cnctwin.geom.cad.CadModel`, CAD frame C) is placed in S by ``cad.placement``
(``cad_pose_S_C``). Everything here is in S, SI. Face selectors (project ``cad.datums`` / ``cad.tolerances`` and
the PMI) are written in C and resolved by :func:`cnctwin.geom.selectors.select_faces` on the model itself.

Signed distances are positive outside the material (the part's outward side), so a result point with
``d > 0`` lies where material should not be (excess) and ``d < 0`` where material should be (missing):

* plane ``(p - p0) . n``; cylinder ``rho - R`` (boss) or ``R - rho`` (hole); sphere alike; cone
  ``rho cos(psi) - z sin(psi)`` (sign reversed for a concave cone); torus ``|p - q| - r`` with q the nearest point
  of the tube's centre circle (reversed for a concave torus);
* freeform faces: the distance to the face's own triangles (sign from their pseudo-normals, valid near the
  surface).

A CAD given as a plain mesh (STL, 3MF, OBJ: no B-rep) becomes one freeform face per mesh group
(:func:`cad_from_mesh`): the deviation map and remaining material work, face-based characteristics need STEP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geom import transform as tf
from ..geom.abi import INFO_CLOSED
from ..geom.bvh import build_tree
from ..geom.cad import ANALYTIC, CadModel, FaceInfo
from ..geom.mesh import Mesh
from . import abi as A
from .kernels import deviation_nb


@dataclass
class CadFace:
    """One B-rep face in S."""

    id: int
    surface: str                        # plane | cylinder | cone | sphere | torus | freeform kinds
    params: dict                        # analytic parameters in S (SI): see cnctwin.geom.cad
    concave: bool | None
    area_m2: float
    feature: int
    name: str
    point_S_m: np.ndarray
    triangles: np.ndarray               # triangle indices of the face in the CAD mesh

    @property
    def analytic(self) -> bool:
        return self.surface in ANALYTIC

    @property
    def internal(self) -> bool:
        """A hole-like face (material outside the surface)."""
        return bool(self.concave)


def _vec(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(3)


def _unit(v) -> np.ndarray:
    v = _vec(v)
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def _transform_params(surface: str, params: dict, T: np.ndarray) -> dict:
    R, t = T[:3, :3], T[:3, 3]
    out = {}
    for key, value in params.items():
        if key.endswith("_m") and isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).shape == (3,):
            out[key] = R @ _vec(value) + t
        elif key in ("normal", "axis"):
            out[key] = _unit(R @ _vec(value))
        else:
            out[key] = value
    return out


class CadReference:
    """The CAD model in S with its tree, faces, boundaries and adjacency (see the module documentation)."""

    def __init__(self, model: CadModel, pose_S_C=None):
        self.model = model
        self.pose_S_C = np.eye(4) if pose_S_C is None else np.asarray(pose_S_C, dtype=np.float64)
        self.mesh = model.mesh.transformed(self.pose_S_C)
        if self.mesh.n_faces == 0:
            raise ValueError("the CAD model has no triangles")
        self.tree = build_tree(self.mesh)
        self.closed = int(self.tree.info[INFO_CLOSED]) == 1
        self.tri_normals = self.mesh.face_normals()
        self.tri_areas = self.mesh.face_areas()
        groups = self.mesh.face_group.astype(np.int64)
        order = np.argsort(groups, kind="stable")
        self.faces: list[CadFace] = []
        for info in model.faces:
            tris = order[np.searchsorted(groups[order], info.id, side="left"):
                         np.searchsorted(groups[order], info.id, side="right")]
            self.faces.append(CadFace(int(info.id), str(info.surface),
                                      _transform_params(info.surface, dict(info.params), self.pose_S_C),
                                      info.concave, float(self.tri_areas[tris].sum()) if len(tris) else info.area_m2,
                                      int(info.feature), str(info.name),
                                      self.pose_S_C[:3, :3] @ _vec(info.point_m) + self.pose_S_C[:3, 3],
                                      np.sort(tris).astype(np.int64)))
        self._boundaries()
        self._face_trees: dict[int, object] = {}

    # ------------------------------------------------------------------ topology
    def _boundaries(self) -> None:
        mesh = self.mesh
        uniq, face_edge, counts = mesh.edges()
        g = mesh.face_group.astype(np.int64)
        ne = len(uniq)
        gmin = np.full(ne, np.iinfo(np.int64).max)
        gmax = np.full(ne, np.iinfo(np.int64).min)
        for k in range(3):
            np.minimum.at(gmin, face_edge[:, k], g)
            np.maximum.at(gmax, face_edge[:, k], g)
        boundary = (gmin != gmax) | (counts != 2)
        self.boundary_edge = boundary[face_edge]                          # [m, 3] edge k = (v_k, v_k+1)
        self.boundary_vertex = np.zeros(mesh.n_vertices, bool)
        self.boundary_vertex[uniq[boundary].reshape(-1)] = True
        adjacency: dict[int, set] = {f.id: set() for f in self.faces}
        between = (gmin != gmax) & (counts == 2)
        for a, b in zip(gmin[between].tolist(), gmax[between].tolist()):
            if a >= 0 and b >= 0:
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
        self.adjacency = {k: sorted(v) for k, v in sorted(adjacency.items())}
        # the shared edges between two faces: vertex pairs and the triangle of each face at the edge
        self._edges_between: dict[tuple[int, int], np.ndarray] = {}
        self._edge_tris: dict[tuple[int, int], np.ndarray] = {}
        flat = face_edge.reshape(-1)
        order = np.argsort(flat, kind="stable")
        first = np.searchsorted(flat[order], np.arange(ne), side="left")
        tri_of = order // 3                                                 # triangle of each (sorted) edge slot
        idx = np.flatnonzero(between)
        if len(idx):
            t0 = tri_of[first[idx]]
            t1 = tri_of[first[idx] + 1]
            ga, gb = g[t0], g[t1]
            ta = np.where(ga <= gb, t0, t1)                                 # the triangle of the lower face id
            tb = np.where(ga <= gb, t1, t0)
            pairs = np.column_stack([gmin[idx], gmax[idx]])
            keys = pairs[:, 0] * (int(pairs.max()) + 1) + pairs[:, 1]
            sort = np.argsort(keys, kind="stable")
            keys, pairs = keys[sort], pairs[sort]
            edges, tris = uniq[idx][sort], np.column_stack([ta, tb])[sort]
            cuts = np.flatnonzero(np.diff(keys)) + 1
            for ce, ct, cp in zip(np.split(edges, cuts), np.split(tris, cuts), np.split(pairs, cuts)):
                key = (int(cp[0, 0]), int(cp[0, 1]))
                self._edges_between[key] = ce
                self._edge_tris[key] = ct

    def shared_edge_points(self, a: int, b: int) -> np.ndarray:
        """Mid points of the mesh edges two faces share ([k, 3], empty if they are not adjacent)."""
        key = (min(a, b), max(a, b))
        e = self._edges_between.get(key)
        if e is None:
            return np.zeros((0, 3))
        v = self.mesh.vertices
        return 0.5 * (v[e[:, 0]] + v[e[:, 1]])

    def edge_concavity(self, a: int, b: int) -> float | None:
        """Share of the shared mesh edges of faces a and b that are concave (the triangle of b beyond the edge
        rises on the outward side of a's triangle there); None when a and b are not adjacent."""
        key = (min(a, b), max(a, b))
        edges = self._edges_between.get(key)
        if edges is None:
            return None
        tris = self._edge_tris[key]
        ta, tb = (tris[:, 0], tris[:, 1]) if a <= b else (tris[:, 1], tris[:, 0])
        V = self.mesh.vertices
        F = self.mesh.faces
        concave = 0
        for e, t_a, t_b in zip(edges, ta, tb):
            other = [v for v in F[t_b] if v != e[0] and v != e[1]]
            if not other:
                continue
            mid = 0.5 * (V[e[0]] + V[e[1]])
            concave += int(float((V[other[0]] - mid) @ self.tri_normals[t_a]) > 0.0)
        return concave / len(edges)

    def is_boundary(self, triangles, features) -> np.ndarray:
        """Per closest-point query: True when the closest feature lies on a B-rep face boundary."""
        tri = np.asarray(triangles, dtype=np.int64)
        feat = np.asarray(features, dtype=np.int64)
        out = np.zeros(len(tri), bool)
        valid = tri >= 0
        edge = valid & (feat >= 1) & (feat <= 3)
        out[edge] = self.boundary_edge[tri[edge], feat[edge] - 1]
        vert = valid & (feat >= 4)
        out[vert] = self.boundary_vertex[self.mesh.faces[tri[vert], feat[vert] - 4]]
        return out

    def feature_faces(self, face_id: int) -> list[int]:
        return sorted(int(f) for f in self.model.feature_faces(int(face_id)))

    def face(self, face_id: int) -> CadFace:
        return self.faces[int(face_id)]

    # ------------------------------------------------------------------ queries
    def closest(self, points, max_dist: float = math.inf):
        """(signed distance, triangle, closest point, feature) of points to the CAD surface."""
        P = A.contiguous(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        d = np.empty(len(P))
        tri = np.empty(len(P), np.int32)
        q = np.empty((len(P), 3))
        feat = np.empty(len(P), np.int8)
        if len(P):
            deviation_nb(self.tree, P, float(max_dist), d, tri, q, feat)
        return d, tri.astype(np.int64), q, feat.astype(np.int64)

    def face_of(self, triangles) -> np.ndarray:
        tri = np.asarray(triangles, dtype=np.int64)
        out = np.full(len(tri), -1, np.int64)
        ok = tri >= 0
        out[ok] = self.mesh.face_group[tri[ok]]
        return out

    def _face_tree(self, face_id: int):
        tree = self._face_trees.get(face_id)
        if tree is None:
            tris = self.faces[face_id].triangles
            sub = Mesh(self.mesh.vertices, self.mesh.faces[tris], self.mesh.face_group[tris]).compact()
            tree = build_tree(sub)
            self._face_trees[face_id] = tree
        return tree

    def face_distance(self, face_id: int, points) -> tuple[np.ndarray, np.ndarray]:
        """(signed distance to the face's nominal surface (+ outward), outward normal there) per point."""
        f = self.faces[int(face_id)]
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        p = f.params
        sign = -1.0 if f.concave else 1.0
        if f.surface == "plane":
            n = _unit(p["normal"])
            return (P - _vec(p["point_m"])) @ n, np.broadcast_to(n, P.shape).copy()
        if f.surface in ("cylinder", "cone", "torus"):
            u = _unit(p["axis"])
            origin = _vec(p["axis_point_m"] if f.surface == "cylinder" else
                          p["apex_m"] if f.surface == "cone" else p["centre_m"])
            w = P - origin
            z = w @ u
            radial = w - z[:, None] * u
            rho = np.linalg.norm(radial, axis=1)
            er = np.divide(radial, rho[:, None], out=np.tile(tf.perpendicular(u), (len(P), 1)),
                           where=rho[:, None] > 0.0)
            if f.surface == "cylinder":
                return sign * (rho - float(p["radius_m"])), sign * er
            if f.surface == "cone":
                psi = float(p["semi_angle_rad"])
                n = math.cos(psi) * er - math.sin(psi) * u[None, :]
                return sign * (rho * math.cos(psi) - z * math.sin(psi)), sign * n
            q = origin + float(p["major_radius_m"]) * er
            v = P - q
            dist = np.linalg.norm(v, axis=1)
            n = np.divide(v, dist[:, None], out=np.zeros_like(v), where=dist[:, None] > 0.0)
            return sign * (dist - float(p["minor_radius_m"])), sign * n
        if f.surface == "sphere":
            w = P - _vec(p["centre_m"])
            rho = np.linalg.norm(w, axis=1)
            n = np.divide(w, rho[:, None], out=np.zeros_like(w), where=rho[:, None] > 0.0)
            return sign * (rho - float(p["radius_m"])), sign * n
        tree = self._face_tree(int(face_id))
        Pc = A.contiguous(P)
        d = np.empty(len(P))
        tri = np.empty(len(P), np.int32)
        q = np.empty((len(P), 3))
        feat = np.empty(len(P), np.int8)
        if len(P):
            deviation_nb(tree, Pc, math.inf, d, tri, q, feat)
        normals = tree.face_normal[np.maximum(tri, 0)]
        return d, normals

    def face_frame(self, face_id: int) -> tuple[np.ndarray, np.ndarray]:
        """(origin, 3x3 rotation whose z column is the face's normal or axis) of a face."""
        f = self.faces[int(face_id)]
        p = f.params
        if f.surface == "plane":
            return _vec(p["point_m"]), tf.frame_from_z(p["normal"])
        if f.surface in ("cylinder", "cone", "torus"):
            origin = _vec(p["axis_point_m"] if f.surface == "cylinder" else
                          p["apex_m"] if f.surface == "cone" else p["centre_m"])
            return origin, tf.frame_from_z(p["axis"])
        if f.surface == "sphere":
            return _vec(p["centre_m"]), np.eye(3)
        return f.point_S_m.copy(), tf.frame_from_z(self.tri_normals[f.triangles[0]] if len(f.triangles) else [0, 0, 1])

    def face_vertices(self, face_id: int) -> np.ndarray:
        tris = self.faces[int(face_id)].triangles
        return self.mesh.vertices[np.unique(self.mesh.faces[tris].reshape(-1))]

    def select(self, selector) -> list[int]:
        """Face ids (the whole feature) a project / PMI selector names, [] when nothing matches."""
        from ..geom.selectors import select_faces

        sel = select_faces(self.model, selector)
        return list(sel.faces) if sel.ok else []

    def select_reason(self, selector) -> str:
        from ..geom.selectors import select_faces

        return select_faces(self.model, selector).reason


def cad_from_mesh(mesh: Mesh, source: str = "", plane_tol_m: float = 1e-9) -> CadModel:
    """A CadModel of a plain mesh: one face per mesh group (group -1 becomes face 0), a plane where every
    triangle of the group lies in one plane (within ``plane_tol_m``), freeform ("other") otherwise."""
    groups = mesh.face_group.astype(np.int64)
    ids = sorted(set(groups.tolist()))
    remap = {g: i for i, g in enumerate(ids)}
    new_groups = np.array([remap[g] for g in groups.tolist()], np.int64)
    out = Mesh(mesh.vertices, mesh.faces, new_groups, name=mesh.name)
    normals = out.face_normals()
    areas = out.face_areas()
    faces = []
    for i in range(len(ids)):
        tris = np.flatnonzero(new_groups == i)
        verts = out.vertices[np.unique(out.faces[tris].reshape(-1))]
        n = (normals[tris] * areas[tris, None]).sum(axis=0)
        nn = float(np.linalg.norm(n))
        surface, params = "other", {}
        if nn > 0.0:
            n = n / nn
            c = verts.mean(axis=0)
            if float(np.max(np.abs((verts - c) @ n))) <= plane_tol_m:
                surface, params = "plane", {"point_m": c.tolist(), "normal": n.tolist()}
        big = tris[int(np.argmax(areas[tris]))]
        faces.append(FaceInfo(id=i, surface=surface, params=params, concave=None, area_m2=float(areas[tris].sum()),
                              point_m=out.triangles()[big].mean(axis=0).tolist(),
                              bounds_m=[verts.min(axis=0).tolist(), verts.max(axis=0).tolist()], feature=i,
                              name=mesh.group_names.get(ids[i], "") if ids[i] >= 0 else "",
                              n_triangles=int(len(tris))))
    return CadModel(out, faces, [[i] for i in range(len(ids))], source=source)


__all__ = ["CadReference", "CadFace", "cad_from_mesh"]
