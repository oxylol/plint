"""Small synthetic CAD models and result meshes for the inspection tests (closed, outward-oriented, face groups =
B-rep face ids)."""

from __future__ import annotations

import math

import numpy as np

from cnctwin.geom.cad import CadModel, FaceInfo
from cnctwin.geom.mesh import Mesh


def _grid_quad(p0, du, dv, nu, nv):
    """Vertices [nu+1, nv+1, 3] and triangles of a parallelogram p0 + s du + t dv (counter-clockwise seen from
    du x dv)."""
    s = np.linspace(0.0, 1.0, nu + 1)
    t = np.linspace(0.0, 1.0, nv + 1)
    V = p0[None, None, :] + s[:, None, None] * du[None, None, :] + t[None, :, None] * dv[None, None, :]
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    F = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            F.append([a, b, c])
            F.append([a, c, d])
    return V.reshape(-1, 3), np.array(F, np.int64)


def box_mesh_fine(lo, hi, n: int = 8) -> Mesh:
    """Box with every side split into n x n quads; groups 0..5 = -x, +x, -y, +y, -z, +z."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    e = hi - lo
    X, Y, Z = np.eye(3)
    sides = [  # (origin, du, dv) with du x dv outward
        (lo, Z * e[2], Y * e[1]), (lo + X * e[0], Y * e[1], Z * e[2]),
        (lo, X * e[0], Z * e[2]), (lo + Y * e[1], Z * e[2], X * e[0]),
        (lo, Y * e[1], X * e[0]), (lo + Z * e[2], X * e[0], Y * e[1])]
    verts, faces, groups = [], [], []
    off = 0
    for g, (p0, du, dv) in enumerate(sides):
        V, F = _grid_quad(np.asarray(p0, float), du, dv, n, n)
        verts.append(V)
        faces.append(F + off)
        groups.append(np.full(len(F), g))
        off += len(V)
    mesh = Mesh(np.concatenate(verts), np.concatenate(faces), np.concatenate(groups))
    return mesh.weld(1e-12)


def box_faces(lo, hi) -> list:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = 0.5 * (lo + hi)
    out = []
    for g, (axis, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))):
        n = np.zeros(3)
        n[axis] = sign
        p = c.copy()
        p[axis] = lo[axis] if sign < 0 else hi[axis]
        out.append((g, "plane", {"point_m": p.tolist(), "normal": n.tolist()}, None))
    return out


def model(mesh: Mesh, specs, features=None, pmi=None) -> CadModel:
    """CadModel of a mesh whose face groups are the faces; ``specs`` [(id, surface, params, concave)]."""
    areas = mesh.face_areas()
    tris = mesh.triangles()
    faces = []
    for fid, surface, params, concave in specs:
        sel = np.flatnonzero(mesh.face_group == fid)
        big = sel[int(np.argmax(areas[sel]))]
        point = tris[big].mean(axis=0)
        verts = tris[sel].reshape(-1, 3)
        faces.append(FaceInfo(id=fid, surface=surface, params=params, concave=concave, area_m2=float(areas[sel].sum()),
                              point_m=point.tolist(), bounds_m=[verts.min(0).tolist(), verts.max(0).tolist()],
                              feature=fid, n_triangles=int(len(sel))))
    feats = features if features is not None else [[f[0]] for f in specs]
    for k, group in enumerate(feats):
        for fid in group:
            faces[fid].feature = k
    return CadModel(mesh, faces, feats, pmi=pmi or {"datums": [], "tolerances": []})


def box_model(lo, hi, n: int = 2) -> CadModel:
    return model(box_mesh_fine(lo, hi, n), box_faces(lo, hi))


def pocket_block(lo, hi, plo, phi, depth, n_div: int = 1) -> Mesh:
    """Block [lo, hi] with a rectangular blind pocket [plo, phi] (xy) of ``depth`` from the top.

    Groups: 0 -x, 1 +x, 2 -y, 3 +y, 4 bottom, 5 top (frame), 6 wall x = plo[0] (normal +x), 7 wall x = phi[0]
    (normal -x), 8 wall y = plo[1] (normal +y), 9 wall y = phi[1] (normal -y), 10 floor (normal +z)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    x = [lo[0], plo[0], phi[0], hi[0]]
    y = [lo[1], plo[1], phi[1], hi[1]]
    zt, zb, zf = hi[2], lo[2], hi[2] - depth
    tris, groups = [], []

    def quad(a, b, c, d, g):
        tris.extend([[a, b, c], [a, c, d]])
        groups.extend([g, g])

    P = lambda xx, yy, zz: np.array([xx, yy, zz], float)
    # top frame: 3x3 cells minus the centre
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            quad(P(x[i], y[j], zt), P(x[i + 1], y[j], zt), P(x[i + 1], y[j + 1], zt), P(x[i], y[j + 1], zt), 5)
    # bottom: 3x3 cells (shared vertices with the sides)
    for i in range(3):
        for j in range(3):
            quad(P(x[i], y[j], zb), P(x[i], y[j + 1], zb), P(x[i + 1], y[j + 1], zb), P(x[i + 1], y[j], zb), 4)
    # outer sides, split at the pocket lines so the vertices match the top and bottom
    for j in range(3):
        quad(P(lo[0], y[j], zb), P(lo[0], y[j], zt), P(lo[0], y[j + 1], zt), P(lo[0], y[j + 1], zb), 0)
        quad(P(hi[0], y[j], zb), P(hi[0], y[j + 1], zb), P(hi[0], y[j + 1], zt), P(hi[0], y[j], zt), 1)
    for i in range(3):
        quad(P(x[i], lo[1], zb), P(x[i + 1], lo[1], zb), P(x[i + 1], lo[1], zt), P(x[i], lo[1], zt), 2)
        quad(P(x[i], hi[1], zb), P(x[i], hi[1], zt), P(x[i + 1], hi[1], zt), P(x[i + 1], hi[1], zb), 3)
    # pocket walls (outward normals point into the pocket) and floor
    quad(P(x[1], y[1], zf), P(x[1], y[2], zf), P(x[1], y[2], zt), P(x[1], y[1], zt), 6)
    quad(P(x[2], y[1], zf), P(x[2], y[1], zt), P(x[2], y[2], zt), P(x[2], y[2], zf), 7)
    quad(P(x[1], y[1], zf), P(x[1], y[1], zt), P(x[2], y[1], zt), P(x[2], y[1], zf), 8)
    quad(P(x[1], y[2], zf), P(x[2], y[2], zf), P(x[2], y[2], zt), P(x[1], y[2], zt), 9)
    quad(P(x[1], y[1], zf), P(x[2], y[1], zf), P(x[2], y[2], zf), P(x[1], y[2], zf), 10)
    soup = np.array(tris)
    mesh = Mesh.from_triangles(soup, weld=True, face_group=np.array(groups))
    return mesh


def pocket_specs(lo, hi, plo, phi, depth) -> list:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    zf = hi[2] - depth
    specs = box_faces(lo, hi)
    specs += [(6, "plane", {"point_m": [plo[0], 0.5 * (plo[1] + phi[1]), zf], "normal": [1, 0, 0]}, None),
              (7, "plane", {"point_m": [phi[0], 0.5 * (plo[1] + phi[1]), zf], "normal": [-1, 0, 0]}, None),
              (8, "plane", {"point_m": [0.5 * (plo[0] + phi[0]), plo[1], zf], "normal": [0, 1, 0]}, None),
              (9, "plane", {"point_m": [0.5 * (plo[0] + phi[0]), phi[1], zf], "normal": [0, -1, 0]}, None),
              (10, "plane", {"point_m": [0.5 * (plo[0] + phi[0]), 0.5 * (plo[1] + phi[1]), zf], "normal": [0, 0, 1]},
               None)]
    return specs


def plate_with_hole(size, thickness, centre, radius, n: int = 64) -> Mesh:
    """Plate [0, sx] x [0, sy] x [0, t] with a cylindrical through hole; groups 0 -x, 1 +x, 2 -y, 3 +y,
    4 bottom, 5 top, 6 hole wall."""
    sx, sy = size
    cx, cy = centre
    ang = 2.0 * math.pi * np.arange(n) / n
    inner = np.column_stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)])
    # boundary loop: rays from the hole centre to the square, plus the corners in angular order
    corners = np.array([[sx, sy], [0.0, sy], [0.0, 0.0], [sx, 0.0]])
    corner_ang = np.mod(np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx), 2 * math.pi)
    outer = []
    for a in ang:
        d = np.array([math.cos(a), math.sin(a)])
        ts = []
        for k, lim in ((0, sx), (0, 0.0), (1, sy), (1, 0.0)):
            if abs(d[k]) > 1e-15:
                t = (lim - (cx if k == 0 else cy)) / d[k]
                if t > 0:
                    q = np.array([cx, cy]) + t * d
                    if -1e-12 <= q[0] <= sx + 1e-12 and -1e-12 <= q[1] <= sy + 1e-12:
                        ts.append(t)
        outer.append(np.array([cx, cy]) + min(ts) * d)
    outer = np.array(outer)
    loop = []          # (point, index of the inner ring sector it belongs to)
    for k in range(n):
        loop.append(outer[k])
        a0, a1 = ang[k], ang[k + 1] if k + 1 < n else 2 * math.pi
        for c, ca in sorted(zip(corners.tolist(), corner_ang.tolist()), key=lambda x: x[1]):
            if a0 < ca < a1:
                loop.append(np.array(c))
    tris, groups = [], []
    z0, z1 = 0.0, thickness
    P = lambda q, z: np.array([q[0], q[1], z])
    # top and bottom: per sector a fan from inner[k]
    sector_pts = []
    pos = 0
    for k in range(n):
        pts = [outer[k]]
        a0, a1 = ang[k], ang[k + 1] if k + 1 < n else 2 * math.pi
        for c, ca in sorted(zip(corners.tolist(), corner_ang.tolist()), key=lambda x: x[1]):
            if a0 < ca < a1:
                pts.append(np.array(c))
        pts.append(outer[(k + 1) % n])
        sector_pts.append(pts)
    for k in range(n):
        pts = sector_pts[k]
        i0, i1 = inner[k], inner[(k + 1) % n]
        chain = [i0] + pts + [i1]
        for a in range(1, len(chain) - 1):
            tris.append([P(chain[0], z1), P(chain[a], z1), P(chain[a + 1], z1)])
            groups.append(5)
            tris.append([P(chain[0], z0), P(chain[a + 1], z0), P(chain[a], z0)])
            groups.append(4)
    # outer sides along the boundary loop
    for k in range(len(loop)):
        a, b = loop[k], loop[(k + 1) % len(loop)]
        mid = 0.5 * (a + b)
        if abs(mid[0]) < 1e-12:
            g = 0
        elif abs(mid[0] - sx) < 1e-12:
            g = 1
        elif abs(mid[1]) < 1e-12:
            g = 2
        else:
            g = 3
        tris.append([P(a, z0), P(b, z0), P(b, z1)])
        tris.append([P(a, z0), P(b, z1), P(a, z1)])
        groups += [g, g]
    # hole wall (outward normal toward the axis)
    for k in range(n):
        a, b = inner[k], inner[(k + 1) % n]
        tris.append([P(a, z0), P(a, z1), P(b, z1)])
        tris.append([P(a, z0), P(b, z1), P(b, z0)])
        groups += [6, 6]
    return Mesh.from_triangles(np.array(tris), weld=True, face_group=np.array(groups))


def plate_specs(size, thickness, centre, radius) -> list:
    specs = box_faces([0, 0, 0], [size[0], size[1], thickness])
    specs.append((6, "cylinder", {"axis_point_m": [centre[0], centre[1], 0.0], "axis": [0, 0, 1], "radius_m": radius},
                  True))
    return specs


def jitter(points, sigma, seed, normals=None):
    rng = np.random.default_rng(seed)
    if normals is None:
        return points + rng.normal(0.0, sigma, points.shape)
    return points + rng.normal(0.0, sigma, len(points))[:, None] * normals


def subdivide(mesh: Mesh, levels: int = 1) -> Mesh:
    """Midpoint subdivision (every triangle into four), face groups kept; vertices stay on the facets."""
    V = mesh.vertices.copy()
    F = mesh.faces.astype(np.int64)
    G = mesh.face_group.copy()
    for _ in range(levels):
        edges = {}
        newV = [V]
        count = len(V)

        def mid(a, b):
            nonlocal count
            key = (a, b) if a < b else (b, a)
            if key not in edges:
                edges[key] = count
                newV.append(0.5 * (V[a] + V[b])[None, :])
                count += 1
            return edges[key]

        nf, ng = [], []
        for (a, b, c), g in zip(F.tolist(), G.tolist()):
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            ng += [g] * 4
        V = np.concatenate(newV)
        F = np.array(nf, np.int64)
        G = np.array(ng, np.int64)
    return Mesh(V, F, G)


def rigid(R_axis, angle, t):
    from cnctwin.geom import transform as tf
    T = np.eye(4)
    T[:3, :3] = tf.axis_angle(R_axis, angle)
    T[:3, 3] = t
    return T
