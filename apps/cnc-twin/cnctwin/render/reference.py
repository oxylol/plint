"""Plain-Python twins of the rasteriser kernels (architecture 5.5): written independently (NumPy over each
triangle's pixel box, polygons clipped with Python lists) so the tests compare two formulations of the same
rules: pixel centres, top-left rule, near-plane clipping, z/w depth, perspective-correct attributes, strict
depth test in face order, section planes, caps, shadow-map visibility, blending and sprites."""

from __future__ import annotations

import math

import numpy as np

from . import kernels as K


def _covers(e: np.ndarray, dx: float, dy: float) -> np.ndarray:
    top_left = dy < 0.0 or (dy == 0.0 and dx > 0.0)
    return (e > 0.0) | ((e == 0.0) & top_left)


def _clip_near(points: list, attrs: list) -> tuple[list, list]:
    out_p, out_a = [], []
    for i in range(3):
        j = (i + 1) % 3
        pi, pj = points[i], points[j]
        di, dj = pi[2] + pi[3], pj[2] + pj[3]
        if di >= 0.0:
            out_p.append(pi)
            out_a.append(attrs[i])
        if (di >= 0.0) != (dj >= 0.0):
            t = di / (di - dj)
            out_p.append(pi + t * (pj - pi))
            out_a.append(attrs[i] + t * (attrs[j] - attrs[i]))
    return out_p, out_a


def _triangles(clip: np.ndarray, attrs: np.ndarray, face, width: int, height: int):
    """Screen triangles of one face after near clipping: list of (sx [3], sy [3], sz [3], iw [3], attrs [3, na],
    back)."""
    pts = [np.asarray(clip[v], dtype=np.float64) for v in face]
    ats = [np.asarray(attrs[v], dtype=np.float64) for v in face]
    poly_p, poly_a = _clip_near(pts, ats)
    if len(poly_p) < 3:
        return []
    sx = [(p[0] / p[3] + 1.0) * 0.5 * width for p in poly_p]
    sy = [(1.0 - p[1] / p[3]) * 0.5 * height for p in poly_p]
    sz = [p[2] / p[3] for p in poly_p]
    iw = [1.0 / p[3] for p in poly_p]
    out = []
    for t in range(len(poly_p) - 2):
        idx = [0, t + 1, t + 2]
        area = (sx[idx[1]] - sx[idx[0]]) * (sy[idx[2]] - sy[idx[0]]) - (sy[idx[1]] - sy[idx[0]]) * (sx[idx[2]] - sx[idx[0]])
        if area == 0.0:
            continue
        back = area > 0.0
        if not back:
            idx = [idx[0], idx[2], idx[1]]
        out.append((np.array([sx[i] for i in idx]), np.array([sy[i] for i in idx]), np.array([sz[i] for i in idx]),
                    np.array([iw[i] for i in idx]), np.array([poly_a[i] for i in idx]), back))
    return out


def _pixels(sx, sy, width: int, height: int):
    """Pixel centres of the triangle's box, the screen barycentrics and the coverage mask."""
    x0 = max(int(math.floor(sx.min())), 0)
    x1 = min(int(math.ceil(sx.max())), width - 1)
    y0 = max(int(math.floor(sy.min())), 0)
    y1 = min(int(math.ceil(sy.max())), height - 1)
    if x0 > x1 or y0 > y1:
        return None
    px, py = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    xc = px + 0.5
    yc = py + 0.5
    area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sy[1] - sy[0]) * (sx[2] - sx[0])
    edges = []
    mask = np.ones(px.shape, bool)
    for a, b in ((1, 2), (2, 0), (0, 1)):
        dx, dy = sx[b] - sx[a], sy[b] - sy[a]
        e = dx * (yc - sy[a]) - dy * (xc - sx[a])
        mask &= _covers(e, dx, dy)
        edges.append(e / area)
    return px, py, edges, mask


def raster_opaque_ref(clip, attrs, faces, flags, cap_index, cap_rgb, planes, shadow, light, shadow_par,
                      width: int, height: int):
    """(depth [H, W], colour [H, W, 3] float32, face_id [H, W]) of the opaque pass."""
    depth = np.full((height, width), np.inf)
    color = np.zeros((height, width, 3), np.float32)
    face_id = np.full((height, width), -1, np.int32)
    use_shadow = shadow_par[K.SH_ON] > 0.5
    for f, face in enumerate(faces):
        fl = int(flags[f])
        for sx, sy, sz, iw, at, back in _triangles(clip, attrs, face, width, height):
            if back and (fl & K.FLAG_CULL_BACK) and not (fl & K.FLAG_CAP):
                continue
            box = _pixels(sx, sy, width, height)
            if box is None:
                continue
            px, py, (l0, l1, l2), mask = box
            for r, c in zip(*np.nonzero(mask)):
                x, y = int(px[r, c]), int(py[r, c])
                lam = np.array([l0[r, c], l1[r, c], l2[r, c]])
                z = float(lam @ sz) + (K.CAP_DEPTH_BIAS if back else 0.0)
                if z < -1.0 or z > 1.0 or z >= depth[y, x]:
                    continue
                q = lam * iw
                q = q / q.sum()
                value = q @ at
                world = value[K.A_WORLD:K.A_WORLD + 3]
                if fl & K.FLAG_CLIP and any(float(pl[:3] @ world + pl[3]) < 0.0 for pl in planes):
                    continue
                depth[y, x] = z
                face_id[y, x] = f
                if back and fl & K.FLAG_CAP:
                    color[y, x] = cap_rgb[cap_index[f]]
                    continue
                vis = 1.0
                if use_shadow and fl & K.FLAG_RECEIVE_SHADOW:
                    vis = shadow_visibility_ref(shadow, light, shadow_par, value[K.A_SHADOW:K.A_SHADOW + 3])
                color[y, x] = value[K.A_AMB:K.A_AMB + 3] + value[K.A_DIR:K.A_DIR + 3] * vis
    return depth, color, face_id


def shadow_visibility_ref(shadow, light, par, point) -> float:
    hs, ws = shadow.shape
    p = light @ np.append(np.asarray(point, dtype=np.float64), 1.0)
    p = p[:3] / p[3]
    iu = int(math.floor((p[0] + 1.0) * 0.5 * ws))
    iv = int(math.floor((1.0 - p[1]) * 0.5 * hs))
    r = int(par[K.SH_PCF])
    total = lit = 0
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            total += 1
            a, b = iu + du, iv + dv
            if not (0 <= a < ws and 0 <= b < hs) or p[2] <= shadow[b, a] + par[K.SH_BIAS]:
                lit += 1
    return lit / total


def raster_depth_ref(clip, faces, flags, width: int, height: int) -> np.ndarray:
    depth = np.full((height, width), np.inf)
    empty = np.zeros((len(clip), 0))
    for f, face in enumerate(faces):
        if not int(flags[f]) & K.FLAG_CAST_SHADOW:
            continue
        for sx, sy, sz, iw, at, back in _triangles(clip, empty, face, width, height):
            box = _pixels(sx, sy, width, height)
            if box is None:
                continue
            px, py, (l0, l1, l2), mask = box
            z = l0 * sz[0] + l1 * sz[1] + l2 * sz[2]
            ok = mask & (z >= -1.0) & (z <= 1.0)
            for r, c in zip(*np.nonzero(ok)):
                y, x = int(py[r, c]), int(px[r, c])
                if z[r, c] < depth[y, x]:
                    depth[y, x] = z[r, c]
    return depth


def raster_blend_ref(clip, rgbaw, faces, order, flags, planes, depth, color):
    color = np.array(color, dtype=np.float32, copy=True)
    height, width = depth.shape
    for f in order:
        fl = int(flags[f])
        for sx, sy, sz, iw, at, back in _triangles(clip, rgbaw, faces[f], width, height):
            if back and fl & K.FLAG_CULL_BACK:
                continue
            box = _pixels(sx, sy, width, height)
            if box is None:
                continue
            px, py, (l0, l1, l2), mask = box
            for r, c in zip(*np.nonzero(mask)):
                x, y = int(px[r, c]), int(py[r, c])
                lam = np.array([l0[r, c], l1[r, c], l2[r, c]])
                z = float(lam @ sz)
                if z < -1.0 or z > 1.0 or z >= depth[y, x]:
                    continue
                q = lam * iw
                q = q / q.sum()
                value = q @ at
                if fl & K.FLAG_CLIP and any(float(pl[:3] @ value[K.B_WORLD:K.B_WORLD + 3] + pl[3]) < 0.0
                                            for pl in planes):
                    continue
                a = min(float(value[K.B_RGBA + 3]), 1.0)
                if a <= 0.0:
                    continue
                color[y, x] = (a * value[:3] + (1.0 - a) * color[y, x].astype(np.float64)).astype(np.float32)
    return color


def raster_sprites_ref(s0, s1, radius_px, rgba, mode, order, depth, color):
    color = np.array(color, dtype=np.float32, copy=True)
    height, width = depth.shape
    for k in order:
        r = float(radius_px[k])
        if r <= 0.0:
            continue
        a, b = np.asarray(s0[k], dtype=np.float64), np.asarray(s1[k], dtype=np.float64)
        e = b[:2] - a[:2]
        ll = float(e @ e)
        for y in range(max(int(math.floor(min(a[1], b[1]) - r)), 0), min(int(math.ceil(max(a[1], b[1]) + r)), height - 1) + 1):
            for x in range(max(int(math.floor(min(a[0], b[0]) - r)), 0), min(int(math.ceil(max(a[0], b[0]) + r)), width - 1) + 1):
                c = np.array([x + 0.5, y + 0.5])
                t = 0.0 if ll == 0.0 else min(max(float((c - a[:2]) @ e) / ll, 0.0), 1.0)
                d2 = float(np.sum((c - (a[:2] + t * e)) ** 2)) / (r * r)
                z = a[2] + t * (b[2] - a[2])
                if d2 >= 1.0 or z >= depth[y, x] or z < -1.0:
                    continue
                w = (1.0 - d2) ** 2
                if mode[k] == K.SPRITE_ADD:
                    color[y, x] = (color[y, x] + rgba[k, 3] * w * rgba[k, :3]).astype(np.float32)
                else:
                    alpha = rgba[k, 3] * w
                    color[y, x] = (alpha * rgba[k, :3] + (1.0 - alpha) * color[y, x]).astype(np.float32)
    return color


def corner_normals_ref(vertices: np.ndarray, faces: np.ndarray, crease_rad: float) -> np.ndarray:
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
    out = np.zeros((len(faces), 3, 3))
    cos_crease = math.cos(crease_rad)
    for f, face in enumerate(faces):
        for k, v in enumerate(face):
            acc = np.zeros(3)
            for g, other in enumerate(faces):
                for kk, vv in enumerate(other):
                    if vv != v or float(normals[g] @ normals[f]) < cos_crease:
                        continue
                    a = tris[g, (kk + 1) % 3] - tris[g, kk]
                    b = tris[g, (kk + 2) % 3] - tris[g, kk]
                    angle = math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b))
                    acc += angle * normals[g]
            n = float(np.linalg.norm(acc))
            out[f, k] = acc / n if n > 0.0 else normals[f]
    return out
