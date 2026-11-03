"""Meshes the renderer builds itself: instanced chip particles, coolant streams, height fields (chip piles),
tool envelopes moved by frames.bin, and rigid placement helpers. Everything returns plain
``(vertices [n, 3], faces [m, 3])`` arrays in metres."""

from __future__ import annotations

import math

import numpy as np

# unit octahedron: a chip is drawn as an octahedron scaled to (thickness, width, length) along its own axes
_OCTA_V = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0]])
_OCTA_F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])


def frame_from_axis(axis) -> np.ndarray:
    """Rotation [3, 3] whose third column is the unit ``axis`` (first column in a fixed, deterministic choice)."""
    z = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(z))
    z = np.array([0.0, 0.0, 1.0]) if norm == 0.0 else z / norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ z) * z
    x /= float(np.linalg.norm(x))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rotation_about_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tool_pose(tip_S, axis_T_S, spindle_angle_rad: float = 0.0) -> np.ndarray:
    """Pose of the rotating tool frame T in S: origin at the tip, +z_T along the tool axis, turned by the
    spindle angle (right-handed about +z_T)."""
    pose = np.eye(4)
    pose[:3, :3] = frame_from_axis(axis_T_S) @ rotation_about_z(spindle_angle_rad)
    pose[:3, 3] = np.asarray(tip_S, dtype=np.float64)
    return pose


def transform(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    return v @ pose[:3, :3].T + pose[:3, 3]


def chips(pos: np.ndarray, size_m: np.ndarray, orient_seed: np.ndarray,
          velocity: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Instanced octahedra: chip i at ``pos[i]`` with half-extents ``size_m[i]`` = (thickness, width, length)/2
    along its own axes; the length axis follows the velocity when known, else a direction drawn from the
    deterministic ``orient_seed [n, 3]`` (uniforms in [0, 1)). Returns vertices [6n, 3], faces [8n, 3]."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    size = np.asarray(size_m, dtype=np.float64).reshape(n, 3)
    seed = np.asarray(orient_seed, dtype=np.float64).reshape(n, 3)
    # length axis: velocity, else a uniform direction on the sphere from two seeds
    zc = 2.0 * seed[:, 0] - 1.0
    phi = 2.0 * math.pi * seed[:, 1]
    rr = np.sqrt(np.maximum(1.0 - zc * zc, 0.0))
    axis = np.stack([rr * np.cos(phi), rr * np.sin(phi), zc], axis=1)
    if velocity is not None:
        vel = np.asarray(velocity, dtype=np.float64).reshape(n, 3)
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        axis = np.where(speed > 0.0, vel / np.maximum(speed, 1e-300), axis)
    roll = 2.0 * math.pi * seed[:, 2]
    verts = np.empty((n, 6, 3))
    for i in range(n):
        R = frame_from_axis(axis[i]) @ rotation_about_z(float(roll[i]))
        local = _OCTA_V * size[i][[0, 1, 2]]          # x: thickness, y: width, z: length (along the axis)
        verts[i] = local @ R.T + pos[i]
    faces = (_OCTA_F[None, :, :] + 6 * np.arange(n)[:, None, None]).reshape(-1, 3)
    return verts.reshape(-1, 3), faces


def jet(origin, direction, length_m: float, radius0_m: float, radius1_m: float, sides: int) -> tuple:
    """Open cone (no caps) from ``origin`` along ``direction``: (vertices, faces, axial fraction per vertex)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / float(np.linalg.norm(d))
    R = frame_from_axis(d)
    angles = 2.0 * math.pi * np.arange(sides) / sides
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(sides)], axis=1)
    v0 = ring * radius0_m
    v1 = ring * radius1_m + np.array([0.0, 0.0, length_m])
    local = np.concatenate([v0, v1])
    vertices = local @ R.T + np.asarray(origin, dtype=np.float64)
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[i, j, sides + j], [i, sides + j, sides + i]]
    frac = np.concatenate([np.zeros(sides), np.ones(sides)])
    return vertices, np.asarray(faces, np.int64), frac


def heightfield(origin, axis_u, axis_v, axis_w, cell_m: float, base_w: np.ndarray, height_m: np.ndarray,
                min_height_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top surface of a height field over cells (iu, iv) with height > ``min_height_m``: the corner heights are
    the means of the adjacent occupied cells, the surface lies at ``base_w + height`` along ``axis_w``.
    Returns vertices, faces (two triangles per occupied cell, normal along +axis_w) and per-vertex height."""
    h = np.asarray(height_m, dtype=np.float64)
    base = np.asarray(base_w, dtype=np.float64)
    nu, nv = h.shape
    occupied = np.isfinite(h) & np.isfinite(base) & (h > min_height_m)
    if not np.any(occupied):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0)
    top = np.where(occupied, base + h, 0.0)
    hh = np.where(occupied, h, 0.0)
    weight = occupied.astype(np.float64)
    cw = np.zeros((nu + 1, nv + 1))
    ct = np.zeros((nu + 1, nv + 1))
    ch = np.zeros((nu + 1, nv + 1))
    for du in (0, 1):
        for dv in (0, 1):
            cw[du:du + nu, dv:dv + nv] += weight
            ct[du:du + nu, dv:dv + nv] += top
            ch[du:du + nu, dv:dv + nv] += hh
    corner_top = np.divide(ct, cw, out=np.zeros_like(ct), where=cw > 0)
    corner_h = np.divide(ch, cw, out=np.zeros_like(ch), where=cw > 0)
    iu, iv = np.meshgrid(np.arange(nu + 1), np.arange(nv + 1), indexing="ij")
    eu = np.asarray(axis_u, dtype=np.float64)
    ev = np.asarray(axis_v, dtype=np.float64)
    ew = np.asarray(axis_w, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    points = (o + (iu[..., None] * cell_m) * eu + (iv[..., None] * cell_m) * ev + corner_top[..., None] * ew)
    index = iu * (nv + 1) + iv
    cu, cv = np.nonzero(occupied)
    a = index[cu, cv]
    b = index[cu + 1, cv]
    c = index[cu + 1, cv + 1]
    d = index[cu, cv + 1]
    faces = np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])
    if float(np.cross(eu, ev) @ ew) < 0.0:
        faces = faces[:, ::-1]
    used = np.unique(faces)
    remap = np.full((nu + 1) * (nv + 1), -1, np.int64)
    remap[used] = np.arange(len(used))
    return points.reshape(-1, 3)[used], remap[faces], corner_h.reshape(-1)[used]


def _weighted(samples: list, weights: list) -> np.ndarray:
    """Sum of w v over the finite samples divided by the sum of their weights (NaN where no sample is finite):
    grid cells outside the material (NaN) do not darken the surface next to them."""
    total = np.zeros(len(samples[0]))
    norm = np.zeros(len(samples[0]))
    for v, w in zip(samples, weights):
        ok = np.isfinite(v)
        total += np.where(ok, w * np.where(ok, v, 0.0), 0.0)
        norm += np.where(ok, w, 0.0)
    return np.divide(total, norm, out=np.full(len(total), np.nan), where=norm > 0.0)


def bilinear_grid(values: np.ndarray, origin, cell_m: float, points: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """Values of a cell-centred 2-D grid ``values [n0, n1]`` (cell (i, j) centred at origin + (i + 1/2, j + 1/2)
    cell along the two S axes ``axes``) at ``points [k, 3]``: bilinear between centres over the finite cells,
    clamped at the edges."""
    v = np.asarray(values, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n0, n1 = v.shape
    x = np.clip((p[:, axes[0]] - o[axes[0]]) / cell_m - 0.5, 0.0, max(n0 - 1, 0))
    y = np.clip((p[:, axes[1]] - o[axes[1]]) / cell_m - 0.5, 0.0, max(n1 - 1, 0))
    i0 = np.minimum(np.floor(x).astype(np.int64), max(n0 - 2, 0))
    j0 = np.minimum(np.floor(y).astype(np.int64), max(n1 - 2, 0))
    i1 = np.minimum(i0 + 1, n0 - 1)
    j1 = np.minimum(j0 + 1, n1 - 1)
    fx = x - i0
    fy = y - j0
    return _weighted([v[i0, j0], v[i1, j0], v[i0, j1], v[i1, j1]],
                     [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])


def trilinear_grid(values: np.ndarray, origin, cell_m: float, points: np.ndarray) -> np.ndarray:
    """Values of a cell-centred 3-D grid ``values [nx, ny, nz]`` at ``points [k, 3]`` (trilinear over the finite
    cells, clamped)."""
    v = np.asarray(values, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    shape = v.shape
    idx0, frac = [], []
    for axis in range(3):
        x = np.clip((p[:, axis] - o[axis]) / cell_m - 0.5, 0.0, max(shape[axis] - 1, 0))
        i = np.minimum(np.floor(x).astype(np.int64), max(shape[axis] - 2, 0))
        idx0.append(i)
        frac.append(x - i)
    samples, weights = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weights.append((frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1])
                               * (frac[2] if dz else 1 - frac[2]))
                samples.append(v[np.minimum(idx0[0] + dx, shape[0] - 1), np.minimum(idx0[1] + dy, shape[1] - 1),
                                 np.minimum(idx0[2] + dz, shape[2] - 1)])
    return _weighted(samples, weights)
