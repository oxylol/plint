"""Collision checks with the field and with static meshes (docs/models/removal.md "Collision").

* :func:`fixture_meshes` - the workholding devices of a setup as closed box meshes in S (``body_mm`` of the
  device, placed by its pose and the stock pose).
* :func:`mesh_vs_field` / :func:`swept_mesh_vs_field` - a mesh (machine part, fixture) at a pose, or moving
  between two poses, against the material (k_collide).
* :func:`envelope_vs_mesh` - the whole tool envelope (holder and spindle nose included) at sub-sampled poses
  between two ticks against a static mesh, through cnctwin.geom's exact revolved-solid / triangle clearance
  (geom.overlap.mesh_revolved_overlap) on the mesh's static tree (geom.bvh.static_tree): the first fraction
  of the motion with a non-positive clearance.
"""

from __future__ import annotations

import math

import numpy as np

from . import abi as A
from .profile import profile_polyline


def fixture_meshes(project, op_index: int = 0) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, vertices S, faces) of every workholding device body of a setup."""
    from ..geom.primitives import box_mesh

    ops = list(getattr(project, "operations", []) or []) if project is not None else []
    if op_index >= len(ops):
        return []
    op = ops[op_index]
    setup = getattr(op, "setup", None)
    S_from_W = np.linalg.inv(np.asarray(getattr(op, "stock_pose_W_S", np.eye(4)), dtype=float))
    out = []
    for k, item in enumerate(getattr(setup, "workholding", []) or []):
        device = getattr(item, "device", None)
        body = getattr(device, "body_m", None) if device is not None else None
        if body is None:
            continue
        body = np.asarray([[float(v) for v in row] for row in body], dtype=float)
        pose = getattr(item, "pose", None)
        T = S_from_W @ (pose.matrix() if hasattr(pose, "matrix") else np.eye(4))
        m = box_mesh(body[0], body[1], name=f"fixture{k}")
        V = np.asarray(m.vertices) @ T[:3, :3].T + T[:3, 3]
        out.append((str(getattr(item, "label", "") or getattr(device, "id", f"fixture{k}")), V,
                    np.asarray(m.faces, np.int32)))
    return out


def mesh_vs_field(field: A.DexelField, vertices_S_m, faces, tol_m: float) -> dict:
    from .k_collide import mesh_field_overlap

    V = np.ascontiguousarray(np.asarray(vertices_S_m, dtype=float))
    Fc = np.ascontiguousarray(np.asarray(faces, dtype=np.int32))
    out = np.zeros(7)
    hit = mesh_field_overlap(field, V, Fc, np.ascontiguousarray(V.min(axis=0)), np.ascontiguousarray(V.max(axis=0)),
                             float(tol_m), out, A.make_scratch())
    return {"hit": bool(hit), "volume_m3": float(out[0]), "depth_m": float(out[1]), "point_S_m": out[2:5].copy(),
            "rays": int(out[5]), "open_rays": int(out[6])}


def swept_mesh_vs_field(field: A.DexelField, vertices_local_m, faces, pose0, pose1, step_m: float,
                        tol_m: float) -> dict:
    from .k_collide import swept_mesh_field

    V = np.ascontiguousarray(np.asarray(vertices_local_m, dtype=float))
    Fc = np.ascontiguousarray(np.asarray(faces, dtype=np.int32))
    P0 = np.asarray(pose0, dtype=float)
    P1 = np.asarray(pose1, dtype=float)
    out = np.zeros(7)
    lam = swept_mesh_field(field, V, Fc, np.ascontiguousarray(P0[:3, :3]), np.ascontiguousarray(P0[:3, 3]),
                           np.ascontiguousarray(P1[:3, :3]), np.ascontiguousarray(P1[:3, 3]), float(step_m),
                           float(tol_m), out, A.make_scratch())
    return {"fraction": float(lam), "hit": lam >= 0.0, "volume_m3": float(out[0]), "depth_m": float(out[1]),
            "point_S_m": out[2:5].copy()}


def _frame(axis):
    w = np.asarray(axis, dtype=float)
    w = w / np.linalg.norm(w)
    ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ w) * w
    x /= np.linalg.norm(x)
    y = np.cross(w, x)
    return x, y, w


def envelope_vs_mesh(prof_all: A.Profile, tip0_S, tip1_S, axis_S, vertices_S_m, faces, step_m: float,
                     margin_m: float = 0.0, chord_tol_m: float = 1e-6) -> dict:
    """First contact of the whole tool envelope moving from tip0 to tip1 (axis fixed) with a static mesh
    (S): sub-samples spaced at most ``step_m``; returns {fraction (-1: none), clearance_m, face}."""
    from ..geom.bvh import static_tree
    from ..geom.mesh import Mesh
    from ..geom.overlap import mesh_revolved_overlap

    z, r = profile_polyline(prof_all, chord_tol_m)
    tree = static_tree(Mesh(np.asarray(vertices_S_m, dtype=float), np.asarray(faces, dtype=np.int64)))
    t0 = np.asarray(tip0_S, dtype=float)
    t1 = np.asarray(tip1_S, dtype=float)
    n = max(1, int(math.ceil(float(np.linalg.norm(t1 - t0)) / max(step_m, 1e-12))))
    x, y, w = _frame(axis_S)
    best = {"fraction": -1.0, "clearance_m": math.inf, "face": -1}
    for k in range(n + 1):
        lam = k / n
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x, y, w
        T[:3, 3] = t0 + lam * (t1 - t0)
        faces_hit, clear, cmin, fmin = mesh_revolved_overlap(tree, T, z, r, margin=margin_m)
        if cmin < best["clearance_m"]:
            best["clearance_m"] = cmin
        if cmin <= 0.0:
            best.update(fraction=lam, clearance_m=cmin, face=int(fmin))
            return best
    return best
