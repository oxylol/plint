"""Virtual coordinate measuring machine: probe plans per CAD feature, touch probing of the result mesh with a
stylus ball, radius compensation, on-demand probe points (docs/models/inspect.md "Virtual CMM").

**Probing.** A ball of radius r starts at ``nominal + (r + prehit) n`` (n the nominal outward normal) and moves
along ``-n`` until it touches the result surface; the move is sphere tracing on the exact unsigned distance to the
mesh (cnctwin.inspect.kernels.probe_nb), so the ball never passes through the surface. The reported point is the
ball centre compensated by r along the nominal vector (vector-mode probing, as CMM software does for features
with a known nominal); the exact contact point is kept too. The compensation error of vector mode on a surface
tilted by theta from the nominal is ``r (1 / cos(theta) - 1) ~ r theta^2 / 2`` (0.5 nm for r = 1 mm, 1 mrad).
A probe that finds nothing within ``prehit + search`` reports missing material; one that starts in contact reports
material where the nominal surface lies at least ``prehit`` deeper (excess beyond the plan's clearance).

**Plans** (ISO 10360-6 sampling counts as defaults): a plane gets a grid of about ``probe_points_plane`` points
over its outline, kept inside the face and ``probe_radius + probe_inset`` away from its edges; a cylinder or cone
``probe_levels`` circles of ``probe_points_circle`` points over its axial extent and angular span; a sphere, a
torus and a freeform face points spread over the face. The approach vector is the nominal inward normal (a hole is
probed from its axis outward). The stylus shaft and the probe head are not modelled (no reach or collision check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geom import transform as tf
from ..geom.bvh import build_tree
from ..geom.mesh import Mesh
from . import abi as A
from .cadref import CadReference
from .kernels import probe_nb
from .sampling import PointSet
from .schema import InspectSettings

STATUS = {A.PROBE_CONTACT: "contact", A.PROBE_NO_CONTACT: "no contact", A.PROBE_START_IN_CONTACT: "start in contact",
          A.PROBE_NOT_CONVERGED: "not converged"}


@dataclass
class ProbePlan:
    faces: list
    nominal: np.ndarray          # [n, 3] nominal surface points
    normals: np.ndarray          # [n, 3] nominal outward normals
    face_of: np.ndarray          # [n]
    note: str = ""

    def __len__(self) -> int:
        return len(self.nominal)


@dataclass
class ProbeResult:
    plan: ProbePlan
    status: np.ndarray           # [n] abi.PROBE_*
    centre: np.ndarray           # [n, 3]
    measured: np.ndarray         # [n, 3] compensated along the nominal vector
    contact: np.ndarray          # [n, 3] exact contact point on the result
    deviation: np.ndarray        # [n] (measured - nominal) . n
    radius_m: float
    raw: np.ndarray = field(default_factory=lambda: np.zeros((0, A.N_PR)))

    @property
    def ok(self) -> np.ndarray:
        return self.status == A.PROBE_CONTACT

    def points(self) -> PointSet:
        ok = self.ok
        return PointSet(self.measured[ok], self.plan.normals[ok], self.plan.face_of[ok], np.ones(int(ok.sum())))

    def to_json(self) -> list[dict]:
        out = []
        for i in range(len(self.status)):
            row = {"face": int(self.plan.face_of[i]), "status": STATUS[int(self.status[i])],
                   "nominal_S_m": [float(v) for v in self.plan.nominal[i]],
                   "vector_S": [float(v) for v in self.plan.normals[i]]}
            if self.status[i] == A.PROBE_CONTACT:
                row.update({"x_S_m": float(self.measured[i, 0]), "y_S_m": float(self.measured[i, 1]),
                            "z_S_m": float(self.measured[i, 2]), "deviation_m": float(self.deviation[i]),
                            "contact_S_m": [float(v) for v in self.contact[i]]})
            out.append(row)
        return out


class VirtualCMM:
    """Touch probing of one result mesh (see the module documentation)."""

    def __init__(self, mesh: Mesh, settings: InspectSettings, tree=None):
        self.mesh = mesh
        self.settings = settings
        self.tree = build_tree(mesh) if tree is None else tree

    # ------------------------------------------------------------------ probing
    def _move(self, origins, directions, radius, travel_m):
        s = self.settings
        O = A.contiguous(np.asarray(origins, dtype=np.float64).reshape(-1, 3))
        Dn = A.contiguous(np.asarray(directions, dtype=np.float64).reshape(-1, 3))
        out = np.zeros((len(O), A.N_PR))
        if len(O):
            probe_nb(self.tree, O, Dn, float(radius), float(travel_m), float(s.probe_trace_tolerance_m),
                     int(s.probe_trace_iterations), out)
        return out

    def probe(self, point, direction, radius_m: float | None = None, vector=None, travel_m: float | None = None) -> dict:
        """One probe move from ``point`` (the ball centre's start) along ``direction`` for at most ``travel_m``
        (default: the distance from the point to the far side of the mesh's bounding box): the contact, the ball
        centre and the point compensated along ``vector`` (default: against the direction of approach)."""
        r = self.settings.probe_radius_m if radius_m is None else float(radius_m)
        d = tf.unit(direction)
        if travel_m is None:
            lo, hi = self.mesh.bounds()
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            travel_m = float(np.max(np.linalg.norm(corners - np.asarray(point, dtype=np.float64), axis=1))) + r
        row = self._move([point], [d], r, travel_m)[0]
        status = int(row[A.PR_STATUS])
        out = {"status": STATUS[status], "travel_m": float(row[A.PR_T]), "steps": int(row[A.PR_ITER])}
        if status in (A.PROBE_CONTACT, A.PROBE_START_IN_CONTACT):
            centre = row[A.PR_CX:A.PR_CZ + 1].copy()
            contact = row[A.PR_QX:A.PR_QZ + 1].copy()
            n = -d if vector is None else tf.unit(vector)
            out.update({"centre_S_m": centre.tolist(), "contact_S_m": contact.tolist(),
                        "measured_S_m": (centre - r * n).tolist(),
                        "surface_normal_S": tf.unit(centre - contact).tolist() if np.linalg.norm(centre - contact) > 0
                        else (-d).tolist()})
        return out

    def measure(self, plan: ProbePlan) -> ProbeResult:
        s = self.settings
        r = s.probe_radius_m
        starts = plan.nominal + (r + s.probe_prehit_m) * plan.normals
        raw = self._move(starts, -plan.normals, r, s.probe_prehit_m + s.probe_search_m + r)
        status = raw[:, A.PR_STATUS].astype(np.int64)
        centre = raw[:, A.PR_CX:A.PR_CZ + 1]
        measured = centre - r * plan.normals
        contact = raw[:, A.PR_QX:A.PR_QZ + 1]
        deviation = np.einsum("ij,ij->i", measured - plan.nominal, plan.normals)
        return ProbeResult(plan, status, centre, measured, contact, deviation, r, raw)


# ============================================================================ plans

def _inside_triangles_2d(P2, T2) -> np.ndarray:
    """Points [n, 2] inside any triangle [m, 3, 2] (barycentric, closed)."""
    a, b, c = T2[:, 0], T2[:, 1], T2[:, 2]
    v0 = c - a
    v1 = b - a
    out = np.zeros(len(P2), bool)
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 0
    for k in range(len(P2)):
        v2 = P2[k] - a
        d02 = np.einsum("ij,ij->i", v0, v2)
        d12 = np.einsum("ij,ij->i", v1, v2)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(ok, (d11 * d02 - d01 * d12) / den, -1.0)
            v = np.where(ok, (d00 * d12 - d01 * d02) / den, -1.0)
        out[k] = bool(np.any((u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)))
    return out


def _segment_distance_2d(P2, S0, S1) -> np.ndarray:
    """Distance of points [n, 2] to the nearest of segments [m, 2]-[m, 2]."""
    if not len(S0):
        return np.full(len(P2), np.inf)
    d = S1 - S0
    L = np.einsum("ij,ij->i", d, d)
    out = np.empty(len(P2))
    for k in range(len(P2)):
        w = P2[k] - S0
        t = np.clip(np.divide(np.einsum("ij,ij->i", w, d), L, out=np.zeros_like(L), where=L > 0), 0.0, 1.0)
        q = S0 + t[:, None] * d
        out[k] = float(np.min(np.linalg.norm(P2[k] - q, axis=1)))
    return out


def _boundary_segments(cad: CadReference, faces) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    V = cad.mesh.vertices
    for f in faces:
        tris = cad.face(f).triangles
        Fm = cad.mesh.faces[tris]
        for k in range(3):
            sel = cad.boundary_edge[tris, k]
            a.append(V[Fm[sel, k]])
            b.append(V[Fm[sel, (k + 1) % 3]])
    return (np.concatenate(a) if a else np.zeros((0, 3)), np.concatenate(b) if b else np.zeros((0, 3)))


def _plane_plan(cad, faces, n_target, inset) -> tuple[np.ndarray, str]:
    f0 = cad.face(faces[0])
    n = tf.unit(f0.params["normal"])
    R = tf.frame_from_z(n)
    p0 = np.asarray(f0.params["point_m"], float)
    tris = np.concatenate([cad.face(f).triangles for f in faces])
    T = cad.mesh.triangles()[tris]
    T2 = (T - p0) @ R[:, :2]
    lo, hi = T2.reshape(-1, 2).min(axis=0), T2.reshape(-1, 2).max(axis=0)
    sa, sb = _boundary_segments(cad, faces)
    S0, S1 = (sa - p0) @ R[:, :2], (sb - p0) @ R[:, :2]
    side = max(2, int(math.ceil(math.sqrt(n_target))))
    for attempt in range(4):
        k = side * (2 ** attempt)
        us = np.linspace(lo[0] + inset, hi[0] - inset, k) if hi[0] - lo[0] > 2 * inset else np.array([0.5 * (lo[0] + hi[0])])
        vs = np.linspace(lo[1] + inset, hi[1] - inset, k) if hi[1] - lo[1] > 2 * inset else np.array([0.5 * (lo[1] + hi[1])])
        G = np.array([[u, v] for u in us for v in vs])
        keep = _inside_triangles_2d(G, T2) & (_segment_distance_2d(G, S0, S1) >= inset)
        G = G[keep]
        if len(G) >= min(n_target, 3) or attempt == 3:
            break
    if len(G) > n_target:
        G = G[np.round(np.linspace(0, len(G) - 1, n_target)).astype(np.int64)]
    note = "" if len(G) >= 3 else f"only {len(G)} points fit on the face with an inset of {inset * 1e3:g} mm"
    return p0 + G @ R[:, :2].T, note


def _on_face(cad, faces, P, inset) -> np.ndarray:
    """Points whose closest CAD point lies on the faces and at least ``inset`` from their boundary edges."""
    d, tri, q, feat = cad.closest(P)
    face = cad.face_of(tri)
    ok = np.isin(face, np.asarray(faces)) & ~cad.is_boundary(tri, feat)
    sa, sb = _boundary_segments(cad, faces)
    if len(sa) and ok.any():
        ab = sb - sa
        L = np.einsum("ij,ij->i", ab, ab)
        for i in np.flatnonzero(ok):
            w = q[i] - sa
            t = np.clip(np.divide(np.einsum("ij,ij->i", w, ab), L, out=np.zeros_like(L), where=L > 0), 0, 1)
            if float(np.min(np.linalg.norm(q[i] - (sa + t[:, None] * ab), axis=1))) < inset:
                ok[i] = False
    return ok


def _revolved_plan(cad, faces, s: InspectSettings, inset) -> tuple[np.ndarray, str]:
    f0 = cad.face(faces[0])
    p = f0.params
    u = tf.unit(p["axis"])
    R = tf.frame_from_z(u)
    V = np.concatenate([cad.face_vertices(f) for f in faces])
    if f0.surface == "cylinder":
        a = np.asarray(p["axis_point_m"], float)
    else:
        a = np.asarray(p["apex_m"], float)
    z = (V - a) @ u
    z0, z1 = float(z.min()) + inset, float(z.max()) - inset
    levels = np.linspace(z0, z1, s.probe_levels) if z1 > z0 else np.array([0.5 * (z.min() + z.max())])
    ang = 2 * math.pi * (np.arange(s.probe_points_circle * 3) + 0.5) / (s.probe_points_circle * 3)
    pts = []
    for zl in levels:
        if f0.surface == "cylinder":
            rad = float(p["radius_m"])
        else:
            rad = zl * math.tan(float(p["semi_angle_rad"]))
        for t in ang:
            pts.append(a + zl * u + rad * (math.cos(t) * R[:, 0] + math.sin(t) * R[:, 1]))
    P = np.array(pts)
    ok = _on_face(cad, faces, P, inset)
    P = P[ok]
    per_level = []
    for zl in levels:
        sel = np.flatnonzero(np.abs((P - a) @ u - zl) < 1e-9)
        if len(sel) > s.probe_points_circle:
            sel = sel[np.round(np.linspace(0, len(sel) - 1, s.probe_points_circle)).astype(np.int64)]
        per_level.append(sel)
    P = P[np.concatenate(per_level)] if per_level else P[:0]
    return P, "" if len(P) >= 3 else "fewer than 3 points fit on the face"


def _spread_plan(cad, faces, count, inset) -> tuple[np.ndarray, str]:
    """Freeform / sphere / torus: triangle centroids projected onto the nominal surface, spread over the face."""
    tris = np.concatenate([cad.face(f).triangles for f in faces])
    C = cad.mesh.triangles()[tris].mean(axis=1)
    order = np.argsort(-cad.tri_areas[tris], kind="stable")
    C = C[order]
    ok = _on_face(cad, faces, C, inset)
    C = C[ok]
    if len(C) > count:
        # farthest-point spread from the largest triangle, deterministic
        chosen = [0]
        dist = np.linalg.norm(C - C[0], axis=1)
        while len(chosen) < count:
            k = int(np.argmax(dist))
            chosen.append(k)
            dist = np.minimum(dist, np.linalg.norm(C - C[k], axis=1))
        C = C[sorted(chosen)]
    return C, "" if len(C) >= 3 else "fewer than 3 points fit on the face"


def plan(cad: CadReference, faces, settings: InspectSettings) -> ProbePlan:
    """The probe plan of a feature (module documentation)."""
    faces = sorted(int(f) for f in faces)
    f0 = cad.face(faces[0])
    inset = settings.probe_radius_m + settings.probe_inset_m
    if f0.surface == "plane":
        P, note = _plane_plan(cad, faces, settings.probe_points_plane, inset)
    elif f0.surface in ("cylinder", "cone"):
        P, note = _revolved_plan(cad, faces, settings, inset)
    else:
        count = settings.probe_points_circle * settings.probe_levels if f0.surface in ("sphere", "torus") \
            else settings.probe_points_plane
        P, note = _spread_plan(cad, faces, count, inset)
    if not len(P):
        return ProbePlan(faces, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64), note or "no point fits")
    # project onto the nominal surfaces and take their normals
    d, tri, _, _ = cad.closest(P)
    face_of = cad.face_of(tri)
    nominal = np.zeros_like(P)
    normals = np.zeros_like(P)
    for f in np.unique(face_of).tolist():
        sel = np.flatnonzero(face_of == f)
        dd, nn = cad.face_distance(int(f), P[sel])
        nominal[sel] = P[sel] - dd[:, None] * nn
        normals[sel] = nn
    return ProbePlan(faces, nominal, normals, face_of.astype(np.int64), note)


__all__ = ["VirtualCMM", "ProbePlan", "ProbeResult", "plan", "STATUS"]
