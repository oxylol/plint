"""Numba kernels of the software rasteriser (docs/models/render.md "Rasteriser").

Conventions
-----------
* Clip space as OpenGL: a vertex is ``(x, y, z, w)`` after ``projection @ view @ world``; the view volume
  is ``-w <= x, y, z <= w``. Triangles are clipped against the near plane ``z + w >= 0`` in clip space
  (Sutherland-Hodgman, attributes interpolated linearly in clip space, which is exact); the other planes
  are handled by the pixel bounding box and the depth range test.
* Screen: pixel ``(px, py)`` covers ``[px, px + 1) x [py, py + 1)``, y down; a pixel is covered when its
  centre ``(px + 0.5, py + 0.5)`` is inside the triangle. Edges through a centre follow the top-left rule
  (an edge is top-left when, in the orientation used here, ``dy < 0`` or ``dy == 0 and dx > 0``), so two
  triangles sharing an edge never both cover, and never both miss, a pixel on it.
* Front faces are counter-clockwise in normalised device coordinates (signed screen area < 0 with y down).
* Depth is z/w (screen-linear, so interpolated with the screen barycentrics); a fragment is kept when it is
  strictly nearer than the stored depth, so of two coincident surfaces the one drawn first stays: the
  order of the face arrays decides, never the thread schedule.
* Attributes are interpolated perspective-correctly (barycentrics weighted by 1/w).

Everything is float64 inside, sequential loops in face order (one thread, no fast-math): the same inputs
give the same pixels on every run. Plain-Python twins of every kernel are in :mod:`cnctwin.render.reference`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, types

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float32[:, :, ::1]
D3 = types.float64[:, :, ::1]
I1 = types.int32[::1]
I2 = types.int32[:, ::1]
OPTS = dict(cache=True, nogil=True, error_model="numpy")

# per-face flags
FLAG_CULL_BACK = 1          # closed solid: back faces are not drawn
FLAG_CLIP = 2               # section planes apply
FLAG_CAP = 4                # back faces seen through a section plane take the cap colour (implies no culling)
FLAG_RECEIVE_SHADOW = 8     # the direct light term is multiplied by the shadow-map visibility
FLAG_CAST_SHADOW = 16       # drawn into the shadow map

# opaque-pass vertex attributes
A_AMB = 0                   # ambient + unshadowed fill light (linear RGB, 3)
A_DIR = 3                   # shadowed key light, diffuse + specular (linear RGB, 3)
A_WORLD = 6                 # world position (3): section planes
A_SHADOW = 9                # shadow lookup position (3): world position offset along the normal (no acne)
N_ATTR = 12

# blend-pass vertex attributes
B_RGBA = 0                  # linear RGB and opacity (4)
B_WORLD = 4                 # world position (3)
N_BLEND = 7

# shadow parameters
SH_ON = 0                   # 1: shadows on
SH_BIAS = 1                 # depth bias in light NDC units
SH_PCF = 2                  # percentage-closer filter radius in texels
N_SHADOW = 4

# back faces drawn as section caps are depth-tested this much (z/w units) farther than they are, so a front face
# lying in the same plane (a fixture standing on the table) always wins over a cap: caps show only through cuts
CAP_DEPTH_BIAS = 1e-7

SPRITE_ALPHA = 0            # over-blended (smoke, steam)
SPRITE_ADD = 1              # additive (glow, sparks)


@njit(types.boolean(types.float64, types.float64, types.float64), **OPTS)
def covers(e, dx, dy):
    """Edge test with the top-left rule (``e`` the edge function at the pixel centre, (dx, dy) the edge)."""
    if e > 0.0:
        return True
    if e < 0.0:
        return False
    return dy < 0.0 or (dy == 0.0 and dx > 0.0)


@njit(types.int64(F2, F2, F2, F2), **OPTS)
def clip_near(cin, ain, cout, aout):
    """Clip the triangle ``cin [3, 4]`` (attributes ``ain [3, na]``) against ``z + w >= 0``; the polygon (0, 3 or
    4 vertices) goes to ``cout [4, 4]`` / ``aout [4, na]``; returns its vertex count."""
    n = 0
    na = ain.shape[1]
    for i in range(3):
        j = (i + 1) % 3
        di = cin[i, 2] + cin[i, 3]
        dj = cin[j, 2] + cin[j, 3]
        if di >= 0.0:
            for c in range(4):
                cout[n, c] = cin[i, c]
            for c in range(na):
                aout[n, c] = ain[i, c]
            n += 1
        if (di >= 0.0) != (dj >= 0.0):
            t = di / (di - dj)
            for c in range(4):
                cout[n, c] = cin[i, c] + t * (cin[j, c] - cin[i, c])
            for c in range(na):
                aout[n, c] = ain[i, c] + t * (ain[j, c] - ain[i, c])
            n += 1
    return n


@njit(types.int64(F2, F1, F1, F1, F1, types.int64, types.int64), **OPTS)
def to_screen(cpoly, sx, sy, sz, iw, n, size_wh):
    """Perspective divide and viewport of the clipped polygon (``size_wh = W * 65536 + H``)."""
    width = size_wh // 65536
    height = size_wh % 65536
    for k in range(n):
        inv = 1.0 / cpoly[k, 3]
        iw[k] = inv
        sx[k] = (cpoly[k, 0] * inv + 1.0) * 0.5 * width
        sy[k] = (1.0 - cpoly[k, 1] * inv) * 0.5 * height
        sz[k] = cpoly[k, 2] * inv
    return n


@njit(types.float64(F2, F2, F1, types.float64, types.float64, types.float64), **OPTS)
def shadow_visibility(shadow, light, par, x, y, z):
    """Fraction of the key light reaching world point (x, y, z): percentage-closer filtering of the shadow map
    (light clip transform ``light``, orthographic). Points outside the map are lit."""
    hs = shadow.shape[0]
    ws = shadow.shape[1]
    lx = light[0, 0] * x + light[0, 1] * y + light[0, 2] * z + light[0, 3]
    ly = light[1, 0] * x + light[1, 1] * y + light[1, 2] * z + light[1, 3]
    lz = light[2, 0] * x + light[2, 1] * y + light[2, 2] * z + light[2, 3]
    lw = light[3, 0] * x + light[3, 1] * y + light[3, 2] * z + light[3, 3]
    lx /= lw
    ly /= lw
    lz /= lw
    u = (lx + 1.0) * 0.5 * ws
    v = (1.0 - ly) * 0.5 * hs
    iu = int(np.floor(u))
    iv = int(np.floor(v))
    r = int(par[SH_PCF])
    bias = par[SH_BIAS]
    lit = 0.0
    count = 0.0
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            a = iu + du
            b = iv + dv
            count += 1.0
            if a < 0 or a >= ws or b < 0 or b >= hs:
                lit += 1.0
            elif lz <= shadow[b, a] + bias:
                lit += 1.0
    return lit / count


@njit(types.boolean(F2, types.float64, types.float64, types.float64), **OPTS)
def kept_by_planes(planes, x, y, z):
    """True when (x, y, z) is on the kept side (``a x + b y + c z + d >= 0``) of every section plane."""
    for p in range(planes.shape[0]):
        if planes[p, 0] * x + planes[p, 1] * y + planes[p, 2] * z + planes[p, 3] < 0.0:
            return False
    return True


@njit(types.int64(F2, F2, I2, I1, I1, F2, F2, F2, F2, F1, F2, F3, I2), **OPTS)
def raster_opaque(clip, attrs, faces, flags, cap_index, cap_rgb, planes, shadow, light, shadow_par,
                  depth, color, face_id):
    """Z-buffered opaque pass: per pixel colour = ambient + direct x shadow visibility (Gouraud: the per-vertex
    lit colours are interpolated; flat: the caller gives the three corners one colour). Back faces of faces
    flagged FLAG_CAP take ``cap_rgb[cap_index[f]]``. Returns the number of fragments written."""
    height = depth.shape[0]
    width = depth.shape[1]
    size_wh = width * 65536 + height
    na = attrs.shape[1]
    cin = np.empty((3, 4))
    ain = np.empty((3, na))
    cpoly = np.empty((4, 4))
    apoly = np.empty((4, na))
    sx = np.empty(4)
    sy = np.empty(4)
    sz = np.empty(4)
    iw = np.empty(4)
    written = 0
    use_shadow = shadow_par[SH_ON] > 0.5
    for f in range(faces.shape[0]):
        fl = flags[f]
        for k in range(3):
            v = faces[f, k]
            for c in range(4):
                cin[k, c] = clip[v, c]
            for c in range(na):
                ain[k, c] = attrs[v, c]
        n = clip_near(cin, ain, cpoly, apoly)
        if n < 3:
            continue
        to_screen(cpoly, sx, sy, sz, iw, n, size_wh)
        for t in range(n - 2):
            i0 = 0
            i1 = t + 1
            i2 = t + 2
            area = (sx[i1] - sx[i0]) * (sy[i2] - sy[i0]) - (sy[i1] - sy[i0]) * (sx[i2] - sx[i0])
            if area == 0.0:
                continue
            back = area > 0.0
            if not back:
                i1, i2 = i2, i1
                area = -area
            if back and (fl & FLAG_CULL_BACK) != 0 and (fl & FLAG_CAP) == 0:
                continue
            xmin = max(int(np.floor(min(sx[i0], min(sx[i1], sx[i2])))), 0)
            xmax = min(int(np.ceil(max(sx[i0], max(sx[i1], sx[i2])))), width - 1)
            ymin = max(int(np.floor(min(sy[i0], min(sy[i1], sy[i2])))), 0)
            ymax = min(int(np.ceil(max(sy[i0], max(sy[i1], sy[i2])))), height - 1)
            if xmin > xmax or ymin > ymax:
                continue
            inv_area = 1.0 / area
            dx0 = sx[i2] - sx[i1]
            dy0 = sy[i2] - sy[i1]
            dx1 = sx[i0] - sx[i2]
            dy1 = sy[i0] - sy[i2]
            dx2 = sx[i1] - sx[i0]
            dy2 = sy[i1] - sy[i0]
            for py in range(ymin, ymax + 1):
                yc = py + 0.5
                for px in range(xmin, xmax + 1):
                    xc = px + 0.5
                    e0 = dx0 * (yc - sy[i1]) - dy0 * (xc - sx[i1])
                    if not covers(e0, dx0, dy0):
                        continue
                    e1 = dx1 * (yc - sy[i2]) - dy1 * (xc - sx[i2])
                    if not covers(e1, dx1, dy1):
                        continue
                    e2 = dx2 * (yc - sy[i0]) - dy2 * (xc - sx[i0])
                    if not covers(e2, dx2, dy2):
                        continue
                    l0 = e0 * inv_area
                    l1 = e1 * inv_area
                    l2 = e2 * inv_area
                    z = l0 * sz[i0] + l1 * sz[i1] + l2 * sz[i2]
                    if back:
                        z += CAP_DEPTH_BIAS
                    if z < -1.0 or z > 1.0 or z >= depth[py, px]:
                        continue
                    q0 = l0 * iw[i0]
                    q1 = l1 * iw[i1]
                    q2 = l2 * iw[i2]
                    qs = q0 + q1 + q2
                    q0 /= qs
                    q1 /= qs
                    q2 /= qs
                    wx = q0 * apoly[i0, A_WORLD] + q1 * apoly[i1, A_WORLD] + q2 * apoly[i2, A_WORLD]
                    wy = q0 * apoly[i0, A_WORLD + 1] + q1 * apoly[i1, A_WORLD + 1] + q2 * apoly[i2, A_WORLD + 1]
                    wz = q0 * apoly[i0, A_WORLD + 2] + q1 * apoly[i1, A_WORLD + 2] + q2 * apoly[i2, A_WORLD + 2]
                    if (fl & FLAG_CLIP) != 0 and not kept_by_planes(planes, wx, wy, wz):
                        continue
                    depth[py, px] = z
                    face_id[py, px] = f
                    written += 1
                    if back and (fl & FLAG_CAP) != 0:
                        ci = cap_index[f]
                        for c in range(3):
                            color[py, px, c] = cap_rgb[ci, c]
                        continue
                    vis = 1.0
                    if use_shadow and (fl & FLAG_RECEIVE_SHADOW) != 0:
                        hx = q0 * apoly[i0, A_SHADOW] + q1 * apoly[i1, A_SHADOW] + q2 * apoly[i2, A_SHADOW]
                        hy = q0 * apoly[i0, A_SHADOW + 1] + q1 * apoly[i1, A_SHADOW + 1] + q2 * apoly[i2, A_SHADOW + 1]
                        hz = q0 * apoly[i0, A_SHADOW + 2] + q1 * apoly[i1, A_SHADOW + 2] + q2 * apoly[i2, A_SHADOW + 2]
                        vis = shadow_visibility(shadow, light, shadow_par, hx, hy, hz)
                    for c in range(3):
                        amb = q0 * apoly[i0, A_AMB + c] + q1 * apoly[i1, A_AMB + c] + q2 * apoly[i2, A_AMB + c]
                        dif = q0 * apoly[i0, A_DIR + c] + q1 * apoly[i1, A_DIR + c] + q2 * apoly[i2, A_DIR + c]
                        color[py, px, c] = amb + dif * vis
    return written


@njit(types.int64(F2, I2, I1, F2), **OPTS)
def raster_depth(clip, faces, flags, depth):
    """Depth-only pass (shadow map) of the faces flagged FLAG_CAST_SHADOW, both sides; returns fragments."""
    height = depth.shape[0]
    width = depth.shape[1]
    size_wh = width * 65536 + height
    cin = np.empty((3, 4))
    ain = np.empty((3, 0))
    cpoly = np.empty((4, 4))
    apoly = np.empty((4, 0))
    sx = np.empty(4)
    sy = np.empty(4)
    sz = np.empty(4)
    iw = np.empty(4)
    written = 0
    for f in range(faces.shape[0]):
        if (flags[f] & FLAG_CAST_SHADOW) == 0:
            continue
        for k in range(3):
            v = faces[f, k]
            for c in range(4):
                cin[k, c] = clip[v, c]
        n = clip_near(cin, ain, cpoly, apoly)
        if n < 3:
            continue
        to_screen(cpoly, sx, sy, sz, iw, n, size_wh)
        for t in range(n - 2):
            i0 = 0
            i1 = t + 1
            i2 = t + 2
            area = (sx[i1] - sx[i0]) * (sy[i2] - sy[i0]) - (sy[i1] - sy[i0]) * (sx[i2] - sx[i0])
            if area == 0.0:
                continue
            if area < 0.0:
                i1, i2 = i2, i1
                area = -area
            xmin = max(int(np.floor(min(sx[i0], min(sx[i1], sx[i2])))), 0)
            xmax = min(int(np.ceil(max(sx[i0], max(sx[i1], sx[i2])))), width - 1)
            ymin = max(int(np.floor(min(sy[i0], min(sy[i1], sy[i2])))), 0)
            ymax = min(int(np.ceil(max(sy[i0], max(sy[i1], sy[i2])))), height - 1)
            if xmin > xmax or ymin > ymax:
                continue
            inv_area = 1.0 / area
            dx0 = sx[i2] - sx[i1]
            dy0 = sy[i2] - sy[i1]
            dx1 = sx[i0] - sx[i2]
            dy1 = sy[i0] - sy[i2]
            dx2 = sx[i1] - sx[i0]
            dy2 = sy[i1] - sy[i0]
            for py in range(ymin, ymax + 1):
                yc = py + 0.5
                for px in range(xmin, xmax + 1):
                    xc = px + 0.5
                    e0 = dx0 * (yc - sy[i1]) - dy0 * (xc - sx[i1])
                    if not covers(e0, dx0, dy0):
                        continue
                    e1 = dx1 * (yc - sy[i2]) - dy1 * (xc - sx[i2])
                    if not covers(e1, dx1, dy1):
                        continue
                    e2 = dx2 * (yc - sy[i0]) - dy2 * (xc - sx[i0])
                    if not covers(e2, dx2, dy2):
                        continue
                    z = (e0 * sz[i0] + e1 * sz[i1] + e2 * sz[i2]) * inv_area
                    if z < -1.0 or z > 1.0 or z >= depth[py, px]:
                        continue
                    depth[py, px] = z
                    written += 1
    return written


@njit(types.int64(F2, F2, I2, I1, I1, F2, F2, F3), **OPTS)
def raster_blend(clip, rgbaw, faces, order, flags, planes, depth, color):
    """Transparent pass: faces in ``order`` (back to front), tested against the opaque depth (not written),
    blended ``c = a src + (1 - a) dst`` with the interpolated linear colour and opacity. Returns fragments."""
    height = depth.shape[0]
    width = depth.shape[1]
    size_wh = width * 65536 + height
    na = rgbaw.shape[1]
    cin = np.empty((3, 4))
    ain = np.empty((3, na))
    cpoly = np.empty((4, 4))
    apoly = np.empty((4, na))
    sx = np.empty(4)
    sy = np.empty(4)
    sz = np.empty(4)
    iw = np.empty(4)
    written = 0
    for oi in range(order.shape[0]):
        f = order[oi]
        fl = flags[f]
        for k in range(3):
            v = faces[f, k]
            for c in range(4):
                cin[k, c] = clip[v, c]
            for c in range(na):
                ain[k, c] = rgbaw[v, c]
        n = clip_near(cin, ain, cpoly, apoly)
        if n < 3:
            continue
        to_screen(cpoly, sx, sy, sz, iw, n, size_wh)
        for t in range(n - 2):
            i0 = 0
            i1 = t + 1
            i2 = t + 2
            area = (sx[i1] - sx[i0]) * (sy[i2] - sy[i0]) - (sy[i1] - sy[i0]) * (sx[i2] - sx[i0])
            if area == 0.0:
                continue
            back = area > 0.0
            if not back:
                i1, i2 = i2, i1
                area = -area
            if back and (fl & FLAG_CULL_BACK) != 0:
                continue
            xmin = max(int(np.floor(min(sx[i0], min(sx[i1], sx[i2])))), 0)
            xmax = min(int(np.ceil(max(sx[i0], max(sx[i1], sx[i2])))), width - 1)
            ymin = max(int(np.floor(min(sy[i0], min(sy[i1], sy[i2])))), 0)
            ymax = min(int(np.ceil(max(sy[i0], max(sy[i1], sy[i2])))), height - 1)
            if xmin > xmax or ymin > ymax:
                continue
            inv_area = 1.0 / area
            dx0 = sx[i2] - sx[i1]
            dy0 = sy[i2] - sy[i1]
            dx1 = sx[i0] - sx[i2]
            dy1 = sy[i0] - sy[i2]
            dx2 = sx[i1] - sx[i0]
            dy2 = sy[i1] - sy[i0]
            for py in range(ymin, ymax + 1):
                yc = py + 0.5
                for px in range(xmin, xmax + 1):
                    xc = px + 0.5
                    e0 = dx0 * (yc - sy[i1]) - dy0 * (xc - sx[i1])
                    if not covers(e0, dx0, dy0):
                        continue
                    e1 = dx1 * (yc - sy[i2]) - dy1 * (xc - sx[i2])
                    if not covers(e1, dx1, dy1):
                        continue
                    e2 = dx2 * (yc - sy[i0]) - dy2 * (xc - sx[i0])
                    if not covers(e2, dx2, dy2):
                        continue
                    l0 = e0 * inv_area
                    l1 = e1 * inv_area
                    l2 = e2 * inv_area
                    z = l0 * sz[i0] + l1 * sz[i1] + l2 * sz[i2]
                    if z < -1.0 or z > 1.0 or z >= depth[py, px]:
                        continue
                    q0 = l0 * iw[i0]
                    q1 = l1 * iw[i1]
                    q2 = l2 * iw[i2]
                    qs = q0 + q1 + q2
                    q0 /= qs
                    q1 /= qs
                    q2 /= qs
                    if (fl & FLAG_CLIP) != 0:
                        wx = q0 * apoly[i0, B_WORLD] + q1 * apoly[i1, B_WORLD] + q2 * apoly[i2, B_WORLD]
                        wy = q0 * apoly[i0, B_WORLD + 1] + q1 * apoly[i1, B_WORLD + 1] + q2 * apoly[i2, B_WORLD + 1]
                        wz = q0 * apoly[i0, B_WORLD + 2] + q1 * apoly[i1, B_WORLD + 2] + q2 * apoly[i2, B_WORLD + 2]
                        if not kept_by_planes(planes, wx, wy, wz):
                            continue
                    a = q0 * apoly[i0, B_RGBA + 3] + q1 * apoly[i1, B_RGBA + 3] + q2 * apoly[i2, B_RGBA + 3]
                    if a <= 0.0:
                        continue
                    if a > 1.0:
                        a = 1.0
                    for c in range(3):
                        src = q0 * apoly[i0, B_RGBA + c] + q1 * apoly[i1, B_RGBA + c] + q2 * apoly[i2, B_RGBA + c]
                        color[py, px, c] = np.float32(a * src + (1.0 - a) * color[py, px, c])
                    written += 1
    return written


@njit(types.int64(F2, F2, F1, F2, I1, I1, F2, F3), **OPTS)
def raster_sprites(s0, s1, radius_px, rgba, mode, order, depth, color):
    """Round sprites and streaks in screen space: sprite k is the capsule around the segment ``s0[k] -> s1[k]``
    (screen x, y in pixels and z/w) of radius ``radius_px[k]``, weight ``(1 - (d / r)^2)^2`` at distance d
    from the segment; depth-tested against the opaque depth (interpolated along the segment). Over-blended
    (SPRITE_ALPHA) or added (SPRITE_ADD) in ``order``. Returns fragments touched."""
    height = depth.shape[0]
    width = depth.shape[1]
    written = 0
    for oi in range(order.shape[0]):
        k = order[oi]
        r = radius_px[k]
        if r <= 0.0:
            continue
        ax = s0[k, 0]
        ay = s0[k, 1]
        bx = s1[k, 0]
        by = s1[k, 1]
        ex = bx - ax
        ey = by - ay
        ll = ex * ex + ey * ey
        xmin = max(int(np.floor(min(ax, bx) - r)), 0)
        xmax = min(int(np.ceil(max(ax, bx) + r)), width - 1)
        ymin = max(int(np.floor(min(ay, by) - r)), 0)
        ymax = min(int(np.ceil(max(ay, by) + r)), height - 1)
        if xmin > xmax or ymin > ymax:
            continue
        alpha = rgba[k, 3]
        for py in range(ymin, ymax + 1):
            yc = py + 0.5
            for px in range(xmin, xmax + 1):
                xc = px + 0.5
                t = 0.0
                if ll > 0.0:
                    t = ((xc - ax) * ex + (yc - ay) * ey) / ll
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                dxp = xc - (ax + t * ex)
                dyp = yc - (ay + t * ey)
                d2 = (dxp * dxp + dyp * dyp) / (r * r)
                if d2 >= 1.0:
                    continue
                z = s0[k, 2] + t * (s1[k, 2] - s0[k, 2])
                if z >= depth[py, px] or z < -1.0:
                    continue
                w = (1.0 - d2) * (1.0 - d2)
                if mode[k] == SPRITE_ADD:
                    for c in range(3):
                        color[py, px, c] = np.float32(color[py, px, c] + alpha * w * rgba[k, c])
                else:
                    a = alpha * w
                    for c in range(3):
                        color[py, px, c] = np.float32(a * rgba[k, c] + (1.0 - a) * color[py, px, c])
                written += 1
    return written


@njit(types.int64(F2, I2, F2, I1, I1, I1, types.float64, D3), **OPTS)
def corner_normals(face_n, faces, corner_angle, offsets, inc_face, inc_corner, cos_crease, out):
    """Crease-aware corner normals: the normal of corner k of face f is the angle-weighted sum of the normals of
    the faces around its vertex whose normal is within the crease angle of face f (``cos_crease``); faces
    without such neighbours keep their own normal. ``offsets/inc_face/inc_corner`` list, per vertex, the
    (face, corner) pairs that use it. Writes ``out [m, 3, 3]``; returns the number of smoothed corners."""
    smoothed = 0
    for f in range(faces.shape[0]):
        for k in range(3):
            v = faces[f, k]
            ax = 0.0
            ay = 0.0
            az = 0.0
            count = 0
            for j in range(offsets[v], offsets[v + 1]):
                g = inc_face[j]
                dot = face_n[g, 0] * face_n[f, 0] + face_n[g, 1] * face_n[f, 1] + face_n[g, 2] * face_n[f, 2]
                if dot >= cos_crease:
                    w = corner_angle[g, inc_corner[j]]
                    ax += w * face_n[g, 0]
                    ay += w * face_n[g, 1]
                    az += w * face_n[g, 2]
                    count += 1
            norm = np.sqrt(ax * ax + ay * ay + az * az)
            if norm > 0.0:
                out[f, k, 0] = ax / norm
                out[f, k, 1] = ay / norm
                out[f, k, 2] = az / norm
                if count > 1:
                    smoothed += 1
            else:
                out[f, k, 0] = face_n[f, 0]
                out[f, k, 1] = face_n[f, 1]
                out[f, k, 2] = face_n[f, 2]
    return smoothed


KERNELS = (raster_opaque, raster_depth, raster_blend, raster_sprites, corner_normals, clip_near, covers,
           to_screen, shadow_visibility, kept_by_planes)
