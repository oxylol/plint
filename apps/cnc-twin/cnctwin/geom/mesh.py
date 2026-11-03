"""Triangle meshes in SI (metres): normals, area, volume, centroid, bounds, merge, transform, weld,
watertight check (docs/models/geom.md "Mesh").

A :class:`Mesh` holds ``vertices`` float64 ``[n, 3]``, ``faces`` int32 ``[m, 3]`` (counter-clockwise
seen from outside, so normals point out of a solid), ``face_group`` int32 ``[m]`` (the B-rep face id
of a STEP import, the group of an OBJ/3MF object, the patch of a primitive; -1 when none), and
attribute dicts: ``vertex_attr[name]`` arrays ``[n]`` or ``[n, k]``, ``face_attr[name]`` arrays ``[m]`` or
``[m, k]``. ``group_names`` maps a group id to a name where the source had one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, types

from . import transform as tf

F2 = types.float64[:, ::1]
I1 = types.int64[::1]
I2 = types.int64[:, ::1]
OPTS = dict(cache=True, nogil=True, error_model="numpy")


def _as_vertices(v) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1, 3))
    return arr.copy() if not arr.flags.writeable else arr


def _as_faces(f) -> np.ndarray:
    raw = np.asarray(f)
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.equal(np.mod(raw, 1), 0)):
            raise ValueError("face indices must be integers")
    arr = np.ascontiguousarray(raw.astype(np.int64).reshape(-1, 3))
    if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.int32).max):
        raise ValueError("face indices out of the int32 range")
    return arr.astype(np.int32)


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_group: np.ndarray | None = None
    vertex_attr: dict = field(default_factory=dict)
    face_attr: dict = field(default_factory=dict)
    group_names: dict = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        self.vertices = _as_vertices(self.vertices)
        self.faces = _as_faces(self.faces)
        n, m = len(self.vertices), len(self.faces)
        if m and int(self.faces.max()) >= n:
            raise ValueError(f"a face refers to vertex {int(self.faces.max())} of {n}")
        if self.face_group is None:
            self.face_group = np.full(m, -1, np.int32)
        else:
            self.face_group = np.ascontiguousarray(np.asarray(self.face_group, dtype=np.int64).reshape(-1)
                                                   ).astype(np.int32)
            if len(self.face_group) != m:
                raise ValueError(f"face_group has {len(self.face_group)} entries for {m} faces")
        self.vertex_attr = {str(k): np.ascontiguousarray(np.asarray(a)) for k, a in sorted(self.vertex_attr.items())}
        self.face_attr = {str(k): np.ascontiguousarray(np.asarray(a)) for k, a in sorted(self.face_attr.items())}
        for k, a in self.vertex_attr.items():
            if a.shape[:1] != (n,):
                raise ValueError(f"vertex attribute {k!r} has {a.shape[:1]} rows for {n} vertices")
        for k, a in self.face_attr.items():
            if a.shape[:1] != (m,):
                raise ValueError(f"face attribute {k!r} has {a.shape[:1]} rows for {m} faces")
        self.group_names = {int(k): str(v) for k, v in sorted(self.group_names.items())}

    # ------------------------------------------------------------------ basics
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy(), self.face_group.copy(),
                    {k: a.copy() for k, a in self.vertex_attr.items()},
                    {k: a.copy() for k, a in self.face_attr.items()}, dict(self.group_names), self.name)

    @classmethod
    def from_triangles(cls, triangles, weld: bool = True, **kwargs) -> "Mesh":
        """Mesh of a triangle soup ``[m, 3, 3]``; ``weld`` merges bit-identical corners."""
        tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        mesh = cls(tris.reshape(-1, 3), np.arange(3 * len(tris)).reshape(-1, 3), **kwargs)
        return mesh.weld(0.0, drop_degenerate=False) if weld else mesh

    def triangles(self) -> np.ndarray:
        """Corner coordinates ``[m, 3, 3]``."""
        return self.vertices[self.faces]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds (min, max) of the vertices used by faces (all vertices if no faces)."""
        pts = self.vertices[np.unique(self.faces)] if self.n_faces else self.vertices
        if len(pts) == 0:
            raise ValueError("an empty mesh has no bounds")
        return pts.min(axis=0), pts.max(axis=0)

    # ------------------------------------------------------------------ measures
    def face_cross(self) -> np.ndarray:
        """``(v1 - v0) x (v2 - v0)`` per face (twice the area times the unit normal)."""
        t = self.triangles()
        return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])

    def face_normals(self) -> np.ndarray:
        """Unit normals ``[m, 3]`` (zero for a degenerate face)."""
        c = self.face_cross()
        n = np.linalg.norm(c, axis=1, keepdims=True)
        return np.divide(c, n, out=np.zeros_like(c), where=n > 0.0)

    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.face_cross(), axis=1)

    def area(self) -> float:
        return float(np.sum(self.face_areas()))

    def _reference_point(self) -> np.ndarray:
        lo, hi = self.bounds()
        return 0.5 * (lo + hi)

    def volume(self) -> float:
        """Signed enclosed volume by the divergence theorem (positive for an outward-oriented closed mesh),
        summed about the bounding-box centre to limit cancellation."""
        if self.n_faces == 0:
            return 0.0
        t = self.triangles() - self._reference_point()
        return float(np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])))) / 6.0

    def centroid(self, kind: str = "volume") -> np.ndarray:
        """Centroid of the enclosed volume (``volume``, closed meshes), of the surface (``area``) or the mean
        of the used vertices (``vertex``)."""
        if kind == "vertex":
            return self.vertices[np.unique(self.faces)].mean(axis=0)
        c0 = self._reference_point()
        t = self.triangles() - c0
        if kind == "area":
            a = self.face_areas()
            total = float(np.sum(a))
            if total == 0.0:
                raise ValueError("a mesh without area has no area centroid")
            return c0 + (a[:, None] * t.mean(axis=1)).sum(axis=0) / total
        if kind != "volume":
            raise ValueError(f"centroid kind {kind!r} is not volume, area or vertex")
        vol6 = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))
        total = float(np.sum(vol6))
        if total == 0.0:
            raise ValueError("a mesh enclosing no volume has no volume centroid")
        return c0 + (vol6[:, None] * t.sum(axis=1)).sum(axis=0) / (4.0 * total)

    def corner_angles(self) -> np.ndarray:
        """Interior angle at each corner ``[m, 3]`` (rad), by atan2 of cross and dot (accurate near 0 and pi)."""
        t = self.triangles()
        out = np.zeros((self.n_faces, 3))
        for k in range(3):
            a = t[:, (k + 1) % 3] - t[:, k]
            b = t[:, (k + 2) % 3] - t[:, k]
            out[:, k] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
        return out

    def vertex_normals(self) -> np.ndarray:
        """Angle-weighted vertex normals (Thuermer & Wuethrich 1998), unit (zero for unused vertices)."""
        fn = self.face_normals()
        ang = self.corner_angles()
        acc = np.zeros((self.n_vertices, 3))
        for k in range(3):
            np.add.at(acc, self.faces[:, k], fn * ang[:, k:k + 1])
        n = np.linalg.norm(acc, axis=1, keepdims=True)
        return np.divide(acc, n, out=np.zeros_like(acc), where=n > 0.0)

    # ------------------------------------------------------------------ topology
    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(edges [e, 2] sorted vertex pairs, face_edge [m, 3] edge index of edge k = (v_k, v_k+1),
        counts [e])``; edges are in lexicographic order."""
        directed = self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int64)
        undirected = np.sort(directed, axis=1)
        if len(undirected) == 0:
            return np.zeros((0, 2), np.int64), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        uniq, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
        return uniq, inverse.reshape(-1, 3), counts

    def edge_report(self) -> dict:
        """Counts of boundary, non-manifold and inconsistently oriented edges, and degenerate faces."""
        if self.n_faces == 0:
            return {"edges": 0, "boundary_edges": 0, "non_manifold_edges": 0, "inconsistent_edges": 0,
                    "degenerate_faces": 0}
        uniq, face_edge, counts = self.edges()
        directed = self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int64)
        forward = (directed[:, 0] < directed[:, 1]).astype(np.int64)
        n_forward = np.bincount(face_edge.reshape(-1), weights=forward, minlength=len(uniq)).astype(np.int64)
        manifold = counts == 2
        inconsistent = manifold & (n_forward != 1)
        f = self.faces
        degenerate = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0]) | (self.face_areas() == 0.0)
        return {"edges": int(len(uniq)), "boundary_edges": int(np.count_nonzero(counts == 1)),
                "non_manifold_edges": int(np.count_nonzero(counts > 2)),
                "inconsistent_edges": int(np.count_nonzero(inconsistent)),
                "degenerate_faces": int(np.count_nonzero(degenerate))}

    def is_watertight(self) -> bool:
        """Closed, consistently oriented 2-manifold: every edge has exactly two faces that traverse it in
        opposite directions, and no face is degenerate."""
        if self.n_faces == 0:
            return False
        r = self.edge_report()
        return r["boundary_edges"] == 0 and r["non_manifold_edges"] == 0 and r["inconsistent_edges"] == 0 \
            and r["degenerate_faces"] == 0

    # ------------------------------------------------------------------ editing
    def transformed(self, T) -> "Mesh":
        """The mesh moved by a 4x4 pose; a ``normal`` vertex attribute is transformed as a normal, and a
        mirroring transform (det < 0) flips the winding so normals keep pointing out."""
        M = np.asarray(T, dtype=np.float64)
        out = self.copy()
        out.vertices = np.ascontiguousarray(tf.apply(M, self.vertices))
        if "normal" in out.vertex_attr:
            out.vertex_attr["normal"] = np.ascontiguousarray(tf.apply_normal(M, out.vertex_attr["normal"]))
        if np.linalg.det(M[:3, :3]) < 0.0:
            out.faces = np.ascontiguousarray(out.faces[:, ::-1])
        return out

    def flipped(self) -> "Mesh":
        out = self.copy()
        out.faces = np.ascontiguousarray(out.faces[:, ::-1])
        if "normal" in out.vertex_attr:
            out.vertex_attr["normal"] = -out.vertex_attr["normal"]
        return out

    def submesh(self, faces) -> "Mesh":
        """The faces selected by a boolean mask or index array (vertices compacted, attributes kept)."""
        sel = np.asarray(faces)
        index = np.flatnonzero(sel) if sel.dtype == bool else sel.astype(np.int64)
        out = Mesh(self.vertices, self.faces[index], self.face_group[index], dict(self.vertex_attr),
                   {k: a[index] for k, a in self.face_attr.items()}, dict(self.group_names), self.name)
        return out.compact()

    def group(self, group_id: int) -> "Mesh":
        return self.submesh(self.face_group == int(group_id))

    def compact(self) -> "Mesh":
        """Drop vertices no face uses (order of the remaining vertices kept)."""
        used = np.zeros(self.n_vertices, bool)
        used[self.faces.reshape(-1)] = True
        new_index = np.cumsum(used) - 1
        return Mesh(self.vertices[used], new_index[self.faces], self.face_group,
                    {k: a[used] for k, a in self.vertex_attr.items()}, dict(self.face_attr),
                    dict(self.group_names), self.name)

    def weld(self, tol_m: float = 0.0, drop_degenerate: bool = True) -> "Mesh":
        """Merge vertices closer than ``tol_m`` (0: bit-identical coordinates only).

        Deterministic greedy clustering in vertex order: a vertex joins the lowest-index representative
        within ``tol_m`` (Euclidean), else becomes one. Faces that collapse (repeated vertex) are dropped
        when ``drop_degenerate``; vertex attributes are those of the representative.
        """
        if self.n_vertices == 0:
            return self.copy()
        if tol_m < 0.0:
            raise ValueError("weld tolerance must not be negative")
        if tol_m == 0.0:
            rep = _exact_representatives(self.vertices)
        else:
            q = np.floor(self.vertices / tol_m).astype(np.int64)
            order = np.lexsort((q[:, 2], q[:, 1], q[:, 0])).astype(np.int64)
            keys = np.ascontiguousarray(q[order])
            rep = np.empty(self.n_vertices, np.int64)
            weld_representatives_nb(self.vertices, float(tol_m), np.ascontiguousarray(q), order, keys, rep)
        keep = rep == np.arange(self.n_vertices)
        new_index = np.cumsum(keep) - 1
        faces = new_index[rep[self.faces.astype(np.int64)]]
        face_sel = np.ones(len(faces), bool)
        if drop_degenerate and len(faces):
            face_sel = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
        return Mesh(self.vertices[keep], faces[face_sel], self.face_group[face_sel],
                    {k: a[keep] for k, a in self.vertex_attr.items()},
                    {k: a[face_sel] for k, a in self.face_attr.items()}, dict(self.group_names), self.name)

    def content_hash(self) -> str:
        """SHA-256 of the geometry (vertices, faces, groups): the key of cached static trees."""
        import hashlib

        h = hashlib.sha256()
        for arr in (self.vertices, self.faces, self.face_group):
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()


def _exact_representatives(vertices: np.ndarray) -> np.ndarray:
    """Index of the first vertex with bit-identical coordinates (signed zeros made equal)."""
    v = vertices + 0.0
    _, first, inverse = np.unique(v, axis=0, return_index=True, return_inverse=True)
    return first[inverse.reshape(-1)].astype(np.int64)


@njit(types.int64(types.int64[:, ::1], types.int64, types.int64, types.int64), **OPTS)
def _lex_lower(keys, kx, ky, kz):
    """First row of the lexicographically sorted ``keys`` that is >= (kx, ky, kz)."""
    lo = 0
    hi = keys.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        a = keys[mid, 0]
        b = keys[mid, 1]
        c = keys[mid, 2]
        less = a < kx or (a == kx and (b < ky or (b == ky and c < kz)))
        if less:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(types.void(F2, types.float64, I2, I1, I2, I1), **OPTS)
def weld_representatives_nb(v, tol, q, order, keys, rep):
    """Greedy tolerance clustering in vertex order (see Mesh.weld): rep[i] = lowest representative j <= i
    with |v_i - v_j| <= tol, found among the 27 neighbouring cells of size tol."""
    n = v.shape[0]
    tol2 = tol * tol
    for i in range(n):
        best = i
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    kx = q[i, 0] + dx
                    ky = q[i, 1] + dy
                    kz = q[i, 2] + dz
                    r = _lex_lower(keys, kx, ky, kz)
                    while r < n and keys[r, 0] == kx and keys[r, 1] == ky and keys[r, 2] == kz:
                        j = order[r]
                        if j < best and rep[j] == j:
                            ex = v[i, 0] - v[j, 0]
                            ey = v[i, 1] - v[j, 1]
                            ez = v[i, 2] - v[j, 2]
                            if ex * ex + ey * ey + ez * ez <= tol2:
                                best = j
                        r += 1
        rep[i] = best


def weld_representatives_ref(vertices: np.ndarray, tol: float) -> np.ndarray:
    """Plain-Python reference of :func:`weld_representatives_nb` (all pairs)."""
    n = len(vertices)
    rep = np.empty(n, np.int64)
    for i in range(n):
        best = i
        for j in range(i):
            if rep[j] == j and float(np.sum((vertices[i] - vertices[j]) ** 2)) <= tol * tol:
                best = j
                break
        rep[i] = best
    return rep


def merge(meshes, offset_groups: bool = False, name: str = "") -> Mesh:
    """One mesh of several. Attributes present in some meshes only are filled with NaN (float) or -1
    (integer) elsewhere. ``offset_groups`` renumbers each mesh's groups after the previous ones' (their
    names follow); otherwise group ids are kept and a name conflict raises."""
    meshes = list(meshes)
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int32), name=name)
    v_attrs = sorted({k for m in meshes for k in m.vertex_attr})
    f_attrs = sorted({k for m in meshes for k in m.face_attr})
    verts, faces, groups = [], [], []
    vattr = {k: [] for k in v_attrs}
    fattr = {k: [] for k in f_attrs}
    names: dict[int, str] = {}
    offset = 0
    group_offset = 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces.astype(np.int64) + offset)
        g = m.face_group.astype(np.int64)
        if offset_groups:
            used = g >= 0
            g = np.where(used, g + group_offset, -1)
            for gid, gname in m.group_names.items():
                names[gid + group_offset] = gname
            top = int(max([int(m.face_group.max()) if m.n_faces else -1] + list(m.group_names)))
            group_offset += top + 1
        else:
            for gid, gname in m.group_names.items():
                if names.get(gid, gname) != gname:
                    raise ValueError(f"group {gid} is named {names[gid]!r} and {gname!r}: merge with offset_groups")
                names[gid] = gname
        groups.append(g)
        for k in v_attrs:
            vattr[k].append(_attr_or_fill(m.vertex_attr, k, m.n_vertices, meshes, "vertex"))
        for k in f_attrs:
            fattr[k].append(_attr_or_fill(m.face_attr, k, m.n_faces, meshes, "face"))
        offset += m.n_vertices
    return Mesh(np.concatenate(verts), np.concatenate(faces), np.concatenate(groups),
                {k: np.concatenate(a) for k, a in vattr.items()}, {k: np.concatenate(a) for k, a in fattr.items()},
                names, name)


def _attr_or_fill(attrs: dict, key: str, rows: int, meshes, kind: str) -> np.ndarray:
    if key in attrs:
        return attrs[key]
    template = next((getattr(m, f"{kind}_attr")[key] for m in meshes if key in getattr(m, f"{kind}_attr")))
    shape = (rows,) + template.shape[1:]
    if np.issubdtype(template.dtype, np.floating):
        return np.full(shape, np.nan, dtype=template.dtype)
    return np.full(shape, -1, dtype=template.dtype)


def face_points_on_surface(mesh: Mesh) -> np.ndarray:
    """Centroid of every face ``[m, 3]``."""
    return mesh.triangles().mean(axis=1)


def polyhedron_volume_cylinder(radius: float, height: float, segments: int) -> float:
    """Exact volume of a regular ``segments``-gon prism inscribed in a cylinder (for tessellation checks)."""
    return 0.5 * segments * radius * radius * math.sin(2.0 * math.pi / segments) * height


KERNELS = (weld_representatives_nb, _lex_lower)

__all__ = ["Mesh", "merge", "weld_representatives_nb", "weld_representatives_ref", "face_points_on_surface",
           "polyhedron_volume_cylinder", "KERNELS"]
