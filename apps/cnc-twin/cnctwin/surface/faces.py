"""Faces, features and maps (docs/models/surface.md sections 3 and 7).

* :func:`read_specs` - the project's roughness tolerances (``cad.tolerances`` of kind roughness: Ra, Rz, Rt
  limits and a face selector in the CAD frame), mapped to S with ``cad.placement``.
* :func:`match` - the patches a selector names: a plane (normal within face_angle, plane offset within the
  tolerance, connected over neighbouring bricks to the patch nearest the selector point), a cylinder (normals
  perpendicular to its axis, distance to the axis within the tolerance of its radius), any other face kind
  (patches within one brick of the point).
* :func:`cross_profile` - the step-over profile of a patch: the passes' tracks (tool-tip position across the
  feed) and generated heights with the tool's silhouette seen along the feed (the effective cutting profile;
  Vickers & Quan 1989), the lower envelope with the passes' minimum chip thickness (the step-over acts as the
  feed across the tracks) evaluated like a feed line; plus the witness marks between overlapping passes.
* :func:`group_faces` - the remaining patches clustered into faces (normal, plane, brick adjacency).
* :func:`patch_roughness` - a patch's reported finish: the direction with the larger Ra (ISO 4288: measure
  across the lay), its Rz and Rt; :func:`judge` - ISO 4288 acceptance (the 16 % rule, or every value for a
  ``max`` specification).
"""

from __future__ import annotations

import math

import numpy as np

from . import abi as A
from . import kernels as K
from . import marks as MK
from .params import field_of, num, plain
from .roughness import IsoRules, evaluate, primary_params
from .store import PF, PI, QF, QI, Store

SIXTEEN_PERCENT = 0.16              # ISO 4288 5.2: at most 16 % of the measured values may exceed the limit


# --------------------------------------------------------------------------------------------- specs
def read_specs(project) -> list[dict]:
    """Roughness tolerances of the project in S: {id, Ra_m, Rz_m, Rt_m, max_rule, face, normal, axis, near,
    radius_m}."""
    cad = field_of(project, "cad")
    tols = field_of(cad, "tolerances") or []
    placement = field_of(cad, "placement")
    pose = np.eye(4)
    if placement is not None and hasattr(placement, "matrix"):
        pose = np.asarray(placement.matrix(), dtype=float)
    R, t = pose[:3, :3], pose[:3, 3]
    out = []
    for tol in tols:
        if str(plain(field_of(tol, "kind")) or "") != "roughness":
            continue
        feature = field_of(tol, "feature")
        near = _vec(feature, "near_m", "near_mm", 1e-3)
        normal = _vec(feature, "normal", "normal", 1.0)
        axis = _vec(feature, "axis", "axis", 1.0)
        spec = {"id": str(plain(field_of(tol, "id")) or f"roughness-{len(out)}"),
                "Ra_m": _len(tol, "Ra_m", "Ra_um"), "Rz_m": _len(tol, "Rz_m", "Rz_um"), "Rt_m": _len(tol, "Rt_m", "Rt_um"),
                "max_rule": str(plain(field_of(tol, "modifier")) or "").lower() == "max",
                "face": str(plain(field_of(feature, "face")) or "any"),
                "near": (R @ near + t) if near is not None else None,
                "normal": (R @ normal / np.linalg.norm(normal)) if normal is not None else None,
                "axis": (R @ axis / np.linalg.norm(axis)) if axis is not None else None,
                "radius_m": _len(feature, "radius_m", "radius_mm", scale=1e-3)}
        out.append(spec)
    return out


def _vec(obj, si_key, raw_key, scale):
    v = field_of(obj, si_key)
    if v is None:
        v = field_of(obj, raw_key)
        if v is None:
            return None
        v = plain(v)
        return np.asarray([float(plain(x)) for x in v], dtype=float) * scale
    v = plain(v)
    return np.asarray([float(plain(x)) for x in v], dtype=float)


def _len(obj, si_key, raw_key, scale=1e-6):
    v = num(obj, si_key)
    if v is not None:
        return v
    v = num(obj, raw_key)
    return v * scale if v is not None else None


# --------------------------------------------------------------------------------------------- matching
def _neighbours(bricks, workpiece: int, brick: int) -> set:
    coords = bricks.coords(workpiece, np.array([brick]))[0]
    out = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx or dy or dz:
                    out.add((int(coords[0]) + dx, int(coords[1]) + dy, int(coords[2]) + dz))
    return out


def match(store: Store, spec: dict, settings, bricks) -> list[int]:
    """Patches a roughness selector names (see the module doc)."""
    if store.n_patch == 0 or spec.get("near") is None:
        return []
    near = spec["near"]
    cents = np.array([store.centroid(i) for i in range(store.n_patch)])
    norms = np.array([store.normal(i) for i in range(store.n_patch)])
    size = float(bricks.brick_size_m(0))
    tol = max(settings.plane_tol_m, 0.05 * size)
    cos_tol = math.cos(settings.face_angle_rad)
    face = spec["face"]
    if face == "plane" and spec.get("normal") is not None:
        n = spec["normal"]
        cand = [i for i in range(store.n_patch) if norms[i] @ n >= cos_tol and abs((cents[i] - near) @ n) <= tol]
        if not cand:
            return []
        start = min(cand, key=lambda i: float(np.linalg.norm(cents[i] - near)))
        return _connected(store, cand, start, bricks)
    if face == "cylinder" and spec.get("axis") is not None:
        a = spec["axis"]
        rel = cents - near[None, :]
        dist = np.linalg.norm(rel - (rel @ a)[:, None] * a[None, :], axis=1)
        sin_tol = math.sin(settings.face_angle_rad)
        radial = np.abs(norms @ a) <= sin_tol
        if not radial.any():
            return []
        radius = spec.get("radius_m")
        if radius is None:
            radius = float(dist[radial][np.argmin(np.abs(np.linalg.norm(rel[radial], axis=1)))])
        return [int(i) for i in np.nonzero(radial & (np.abs(dist - radius) <= max(tol, 0.1 * size)))[0]]
    dist = np.linalg.norm(cents - near[None, :], axis=1)
    return [int(i) for i in np.nonzero(dist <= size)[0]]


def _connected(store: Store, cand: list, start: int, bricks) -> list[int]:
    by_cell = {}
    for i in cand:
        w = int(store.pi[i, PI["workpiece"]])
        c = tuple(int(v) for v in bricks.coords(w, np.array([store.pi[i, PI["brick"]]]))[0])
        by_cell.setdefault((w, c), []).append(i)
    w0 = int(store.pi[start, PI["workpiece"]])
    c0 = tuple(int(v) for v in bricks.coords(w0, np.array([store.pi[start, PI["brick"]]]))[0])
    seen = {(w0, c0)}
    queue = [(w0, c0)]
    out = []
    while queue:
        w, c = queue.pop(0)
        out.extend(by_cell.get((w, c), []))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = (w, (c[0] + dx, c[1] + dy, c[2] + dz))
                    if nb not in seen and nb in by_cell:
                        seen.add(nb)
                        queue.append(nb)
    return sorted(set(out))


def group_faces(store: Store, patches: list[int], settings, bricks) -> list[list[int]]:
    """Patches clustered into faces: same workpiece, normal within face_angle, plane within tolerance,
    neighbouring bricks (deterministic: seeds in patch order)."""
    left = sorted(patches)
    cos_tol = math.cos(settings.face_angle_rad)
    size = float(bricks.brick_size_m(0))
    tol = max(settings.plane_tol_m, 0.05 * size)
    out = []
    assigned = set()
    for seed in left:
        if seed in assigned:
            continue
        n0 = store.normal(seed)
        c0 = store.centroid(seed)
        w0 = int(store.pi[seed, PI["workpiece"]])
        cand = [i for i in left if i not in assigned and int(store.pi[i, PI["workpiece"]]) == w0
                and store.normal(i) @ n0 >= cos_tol and abs((store.centroid(i) - c0) @ n0) <= tol]
        face = _connected(store, cand, seed, bricks)
        assigned.update(face)
        out.append(face)
    return out


# --------------------------------------------------------------------------------------------- roughness
def patch_roughness(pf: np.ndarray) -> dict | None:
    """Reported finish of one patch row: the roughest direction's means and the worst values."""
    out = {}
    for tag in ("f", "c"):
        length = pf[PF[f"{tag}_len"]]
        if length > 0.0:
            out[tag] = {"Ra_m": pf[PF[f"{tag}_ra"]] / length, "Rq_m": pf[PF[f"{tag}_rq"]] / length,
                        "Rz_m": pf[PF[f"{tag}_rz"]] / length, "Rt_m": pf[PF[f"{tag}_rt_max"]],
                        "Rz_max_m": pf[PF[f"{tag}_rz_max"]], "Ra_max_m": pf[PF[f"{tag}_ra_max"]],
                        "lines": int(pf[PF[f"{tag}_lines"]]), "cutoff_m": pf[PF[f"{tag}_cutoff"]],
                        "short": int(pf[PF[f"{tag}_short"]]), "length_m": float(length)}
    if not out:
        return None
    best = max(out, key=lambda k: out[k]["Ra_m"])
    res = dict(out[best])
    res["direction"] = "feed" if best == "f" else "cross"
    res["directions"] = out
    return res


def judge(values: list, limit: float | None, max_rule: bool) -> bool | None:
    """ISO 4288 acceptance of the measured values against an upper limit (None: no limit or no values)."""
    if limit is None or not values:
        return None
    over = sum(1 for v in values if v > limit)
    if max_rule:
        return over == 0
    return over <= SIXTEEN_PERCENT * len(values)


# --------------------------------------------------------------------------------------------- cross profile
def silhouette(outline_pts: np.ndarray, a: np.ndarray, tc: np.ndarray, nn: np.ndarray, n_phi: int,
               du: float) -> tuple[np.ndarray, float]:
    """(points [m, 2] of the revolved outline's lower envelope (u, w) relative to the tool tip, its lowest w)."""
    pts = np.asarray(outline_pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    count = int(min(2000, max(16, math.ceil(arc[-1] / 1.0e-5) + 1)))
    s = np.linspace(0.0, arc[-1], count)
    r = np.ascontiguousarray(np.interp(s, arc, pts[:, 0]))
    z = np.ascontiguousarray(np.interp(s, arc, pts[:, 1]))
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    e1 = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    reach = float(np.max(np.abs(r)) + np.max(np.abs(z)))
    n = int(math.ceil(2.0 * reach / du)) + 1
    out = np.full(n, np.inf)
    K.silhouette(r, z, np.ascontiguousarray(a), np.ascontiguousarray(e1), np.ascontiguousarray(e2),
                 np.ascontiguousarray(np.asarray(tc, dtype=float)), np.ascontiguousarray(np.asarray(nn, dtype=float)),
                 int(n_phi), -reach, du, out)
    u = -reach + np.arange(n) * du
    ok = np.isfinite(out)
    if not ok.any():
        return np.zeros((2, 2)), 0.0
    poly = np.stack([u[ok], out[ok]], axis=1)
    return np.ascontiguousarray(poly), float(np.min(out[ok]))


def cross_profile(store: Store, patch: int, outline_of, settings, rules: IsoRules, cache: dict,
                  keep_profile: bool = False, recovery: float = 0.0):
    """(Roughness or None, witness marks, info) of a patch's step-over profile (see the module doc); ``recovery``
    is the partial-removal band of the minimum chip thickness (surface-roughness recovery_ratio, as the feed lines)."""
    qs = [int(q) for q in store.passes_of(patch)
          if store.qf[q, QF["count"]] >= 3 and int(store.qi[q, QI["cls"]]) in (0, 1)]
    if len(qs) < 2:
        return None, [], {}
    n = store.normal(patch)
    o = store.origin(patch)
    dirs = [store.qf[q, QF["dsx"]:QF["dsz"] + 1] for q in qs]
    ref = None
    for d in dirs:
        d = d - (d @ n) * n
        if np.linalg.norm(d) > 0.0:
            ref = d / np.linalg.norm(d)
            break
    if ref is None:
        return None, [], {}
    cos20 = math.cos(math.radians(20.0))
    group = []
    for q, d in zip(qs, dirs):
        d = d - (d @ n) * n
        if np.linalg.norm(d) > 0.0 and abs(float(d @ ref)) / np.linalg.norm(d) >= cos20:
            group.append(q)
    if len(group) < 2:
        return None, [], {}
    tc = np.cross(n, ref)
    rows = []
    for q in group:
        count = store.qf[q, QF["count"]]
        tip = store.qf[q, QF["csx"]:QF["csz"] + 1] / count
        rows.append({"q": q, "U": float((tip - o) @ tc), "H": float(store.qf[q, QF["w_min"]]),
                     "u_n": float(store.qf[q, QF["un_sum"]] / count),
                     "hmin": float(store.qf[q, QF["hmin_sum"]] / count), "tick1": int(store.qi[q, QI["tick1"]]),
                     "block": int(store.qi[q, QI["block_last"]]), "slot": int(store.qi[q, QI["slot"]]),
                     "axis": store.qf[q, QF["ax"]:QF["az"] + 1].copy()})
    du = rules.stylus_for(rules.by_ra[2][1])[1]
    tpls, P_rows = [], []
    widths = []
    for row in rows:
        out = outline_of(row["slot"])
        key = (row["slot"], tuple(np.round(row["axis"], 6)), tuple(np.round(tc, 6)), tuple(np.round(n, 6)))
        if key not in cache:
            cache[key] = silhouette(out.pts, row["axis"], tc, n, settings.silhouette_angles, du)
        poly, gmin = cache[key]
        widths.append(2.0 * float(out.tool_radius_m))
        P_rows.append((row, len(tpls), gmin))
        tpls.append(poly)
    order = sorted(range(len(P_rows)), key=lambda i: P_rows[i][0]["tick1"])
    P = A.new_passes(len(P_rows))
    for k, i in enumerate(order):
        row, t_index, gmin = P_rows[i]
        P.kind[k] = A.KIND_CUT
        P.mode[k] = A.MODE_DIRECT
        P.tpl[k] = t_index
        P.s[k] = row["U"]
        P.w[k] = row["H"] - gmin
        P.hmin[k] = row["hmin"]          # the layer a pass meets from the neighbouring track (step-over as the feed)
        P.half[k] = np.inf
    T = A.new_templates(tpls)
    U = np.array([r["U"] for r in rows])
    lo = float(np.min(U)) - max(widths)
    hi = float(np.max(U)) + max(widths)
    count = int(math.ceil((hi - lo) / du)) + 1
    cap = int(np.max(np.diff(T.off)))
    G = A.new_grid(lo, du, count, cap)
    K.envelope_profile(P, T, G, 2, float(recovery))
    i0 = int(math.ceil((float(np.min(U)) - lo) / du))
    i1 = int(math.floor((float(np.max(U)) - lo) / du))
    seg = G.z[i0:i1 + 1]
    marks = MK.witness(rows, min(widths), settings.witness_min_m)
    info = {"passes": len(rows), "track_spacing_m": float(np.median(np.diff(np.sort(U)))) if len(U) > 1 else 0.0,
            "direction_S": [float(v) for v in tc]}
    if len(seg) < 16 or not np.all(np.isfinite(seg)):
        return None, marks, info
    rough = evaluate(seg, du, rules)
    info["kinematic"] = primary_params(seg, du, rules).to_dict()
    if keep_profile:
        info["profile"] = seg.copy()
        info["du_m"] = du
    return rough, marks, info


# --------------------------------------------------------------------------------------------- brick maps
ATTRIBUTES = ("Ra_m", "Rz_m", "Rt_m", "lay_x", "lay_y", "lay_z", "burr_height_m", "residual_stress_Pa",
              "hardness_change", "white_layer", "burn", "mark_height_m")


def brick_attributes(store: Store, burr_loc: np.ndarray, burr_h: np.ndarray, meshes: dict, bricks,
                     key_to_wp_brick, settings) -> dict:
    """brick key -> per-vertex attribute arrays (NaN where no patch of the brick matches the vertex)."""
    from .protocols import vertex_normals
    cos_tol = math.cos(2.0 * settings.face_angle_rad)
    tol = max(settings.plane_tol_m, 0.05 * float(bricks.brick_size_m(0)))
    out = {}
    for key in sorted(meshes, key=lambda k: (len(k), k)):
        vertices, faces = meshes[key]
        V = np.asarray(vertices, dtype=float).reshape(-1, 3)
        Nv = vertex_normals(V, faces)
        wp, brick = key_to_wp_brick(key)
        arrays = {name: np.full(len(V), np.nan, dtype=np.float32) for name in ATTRIBUTES}
        for p in store.facets.get((wp, brick), []):
            rough = patch_roughness(store.pf[p])
            n = store.normal(p)
            sel = (Nv @ n >= cos_tol) & (np.abs((V - store.origin(p)[None, :]) @ n) <= tol)
            if not sel.any():
                continue
            if rough is not None:
                arrays["Ra_m"][sel] = rough["Ra_m"]
                arrays["Rz_m"][sel] = rough["Rz_m"]
                arrays["Rt_m"][sel] = rough["Rt_m"]
            lay = store.pf[p, PF["lx"]:PF["lz"] + 1]
            arrays["lay_x"][sel], arrays["lay_y"][sel], arrays["lay_z"][sel] = lay[0], lay[1], lay[2]
            arrays["residual_stress_Pa"][sel] = store.pf[p, PF["sigma"]]
            arrays["hardness_change"][sel] = store.pf[p, PF["hard_change"]]
            arrays["white_layer"][sel] = store.pf[p, PF["white"]]
            arrays["burn"][sel] = store.pf[p, PF["burn"]]
            arrays["mark_height_m"][sel] = store.pf[p, PF["mark_max"]]
        if len(burr_loc) and len(V):
            reach = 2.0 * settings.plane_tol_m + float(np.max(np.maximum(burr_h, 0.0)))
            lo, hi = V.min(axis=0) - reach, V.max(axis=0) + reach
            inside = np.all((burr_loc >= lo[None, :]) & (burr_loc <= hi[None, :]), axis=1)
            if inside.any():
                loc, hh = burr_loc[inside], burr_h[inside]
                best = np.full(len(V), -np.inf)
                for j in range(len(loc)):
                    d = np.linalg.norm(V - loc[j][None, :], axis=1)
                    hit = d <= 2.0 * settings.plane_tol_m + max(float(hh[j]), 0.0)
                    best = np.where(hit & (hh[j] > best), hh[j], best)
                arrays["burr_height_m"] = np.where(best > -np.inf, best, np.nan).astype(np.float32)
        out[key] = arrays
    return out


__all__ = ["read_specs", "match", "group_faces", "patch_roughness", "judge", "silhouette", "cross_profile",
           "brick_attributes", "ATTRIBUTES", "SIXTEEN_PERCENT"]
