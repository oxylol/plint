"""Stock fields and the CAD target field (docs/models/removal.md "Stock and target").

Stock frame conventions (shared with chips and thermal): a block of size (sx, sy, sz) spans
``[0, sx] x [0, sy] x [0, sz]`` in S; a cylinder [diameter, length] and a tube [od, id, length] have their
axis on S z through x = y = 0 and span ``z in [0, length]``; a mesh stock is given in S. Every endpoint of
the stock gets the surface event id ``SEV_STOCK``.

The target field is the CAD part rasterised on the same rays (``cad.placement`` maps the CAD frame into
S); the gouge, breakthrough and remaining-material checks compare the cut field with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import InputError
from . import abi as A
from .field import make_field, n_levels


def _scratch():
    return A.make_scratch()


def block_field(size_m, lattice_m: float, refine=(), lo_m=(0.0, 0.0, 0.0), margin_nodes: int = 1,
                workpiece: int = 0, pool_blocks: int | None = None) -> A.DexelField:
    from .k_stock import fill_box

    lo = np.asarray(lo_m, dtype=float).reshape(3)
    hi = lo + np.asarray(size_m, dtype=float).reshape(3)
    field = make_field(lo, hi, lattice_m, refine, margin_nodes, pool_blocks, workpiece)
    sc = _scratch()
    for lvl in range(n_levels(field)):
        if fill_box(field, lvl, np.ascontiguousarray(lo), np.ascontiguousarray(hi), A.SEV_STOCK, sc) < 0:
            raise InputError("block stock: the overflow pool is too small for the stock (internal sizing error)")
    return field


def tube_field(outer_diameter_m: float, length_m: float, lattice_m: float, inner_diameter_m: float = 0.0,
               refine=(), margin_nodes: int = 1, workpiece: int = 0, pool_blocks: int | None = None) -> A.DexelField:
    """Cylinder (inner diameter 0) or tube about S z, z in [0, length]."""
    from .k_stock import fill_tube

    r_out = 0.5 * float(outer_diameter_m)
    r_in = 0.5 * float(inner_diameter_m)
    if not 0.0 <= r_in < r_out:
        raise InputError(f"tube stock: inner diameter {inner_diameter_m} must be below the outer {outer_diameter_m}")
    lo = np.array([-r_out, -r_out, 0.0])
    hi = np.array([r_out, r_out, float(length_m)])
    field = make_field(lo, hi, lattice_m, refine, margin_nodes, pool_blocks, workpiece)
    sc = _scratch()
    for lvl in range(n_levels(field)):
        if fill_tube(field, lvl, 0.0, 0.0, r_out, r_in, 0.0, float(length_m), A.SEV_STOCK, sc) < 0:
            raise InputError("tube stock: the overflow pool is too small (internal sizing error)")
    return field


def oriented_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray, float]:
    """(vertices, faces with outward orientation, volume): a mesh whose signed volume is negative is
    turned inside out; a zero volume is an error."""
    V = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64).reshape(-1, 3))
    Fc = np.ascontiguousarray(np.asarray(faces, dtype=np.int64).reshape(-1, 3)).astype(np.int32)
    if len(Fc) == 0:
        raise InputError("mesh without faces")
    a, b, c = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    if vol == 0.0:
        raise InputError("mesh encloses no volume")
    if vol < 0.0:
        Fc = np.ascontiguousarray(Fc[:, ::-1])
        vol = -vol
    return V, Fc, vol


def _windows(field, lvl, fam):
    u, v = A.TRANS_U[fam], A.TRANS_V[fam]
    return 0, int(field.lv_dims[lvl, u]) - 1, 0, int(field.lv_dims[lvl, v]) - 1


def rasterize_family(field, lvl: int, fam: int, V, Fc, window=None):
    """Hits of a mesh on the rays of one family of a level: (window, offsets, t, dir)."""
    from .k_stock import rasterize

    iu0, iu1, iv0, iv1 = window if window is not None else _windows(field, lvl, fam)
    nw = (iu1 - iu0 + 1) * (iv1 - iv0 + 1)
    counts = np.zeros(nw, np.int64)
    dummy_off = np.zeros(1, np.int64)
    dummy_t = np.zeros(1)
    dummy_d = np.zeros(1, np.int8)
    total = int(rasterize(field, lvl, fam, V, Fc, iu0, iu1, iv0, iv1, counts, dummy_off, dummy_t, dummy_d, 0))
    offsets = np.zeros(nw + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    counts[:] = 0
    ts = np.zeros(max(total, 1))
    ds = np.zeros(max(total, 1), np.int8)
    rasterize(field, lvl, fam, V, Fc, iu0, iu1, iv0, iv1, counts, offsets, ts, ds, 1)
    return (iu0, iu1, iv0, iv1), offsets, ts, ds


def mesh_field(vertices_S_m, faces, lattice_m: float, refine=(), margin_nodes: int = 1, workpiece: int = 0,
               pool_blocks: int | None = None, box=None) -> tuple[A.DexelField, dict]:
    """Stock from a closed triangle mesh in S (exact ray crossings of the triangles)."""
    from .k_stock import resolve_hits

    V, Fc, vol = oriented_mesh(vertices_S_m, faces)
    lo = V.min(axis=0) if box is None else np.asarray(box[0], dtype=float)
    hi = V.max(axis=0) if box is None else np.asarray(box[1], dtype=float)
    field = make_field(lo, hi, lattice_m, refine, margin_nodes, pool_blocks, workpiece)
    sc = _scratch()
    open_rays = 0
    for lvl in range(n_levels(field)):
        for fam in range(3):
            window, offsets, ts, ds = rasterize_family(field, lvl, fam, V, Fc)
            stats_i = np.zeros(3, np.int64)
            stats_f = np.zeros(5)
            dummy_seg = np.zeros((1, 2), np.int32)
            dummy_n = np.zeros(1, np.int64)
            if resolve_hits(field, lvl, fam, *window, offsets, ts, ds, 0, A.SEV_STOCK, dummy_seg, dummy_n, stats_i,
                            stats_f, sc) < 0:
                raise InputError("mesh stock: the overflow pool is too small (internal sizing error)")
            open_rays += int(stats_i[0])
    if open_rays:
        raise InputError(f"mesh stock is not closed: {open_rays} lattice rays cross it an odd number of times")
    return field, {"mesh_volume_m3": vol, "triangles": int(len(Fc))}


def target_from_mesh(field: A.DexelField, vertices_S_m, faces) -> tuple[A.TargetField, dict]:
    """The CAD part on the rays of ``field`` (compressed rows per ray)."""
    from .k_stock import resolve_hits

    V, Fc, vol = oriented_mesh(vertices_S_m, faces)
    nray = int(field.info[A.FI_NRAYS])
    counts = np.zeros(nray, np.int64)
    pieces = []
    open_rays = 0
    sc = _scratch()
    for lvl in range(n_levels(field)):
        for fam in range(3):
            window, offsets, ts, ds = rasterize_family(field, lvl, fam, V, Fc)
            total = int(offsets[-1])
            seg = np.zeros((max(total, 1), 2), np.int32)
            nseg = np.zeros(len(offsets) - 1, np.int64)
            stats_i = np.zeros(3, np.int64)
            stats_f = np.zeros(5)
            resolve_hits(field, lvl, fam, *window, offsets, ts, ds, 1, A.SEV_STOCK, seg, nseg, stats_i, stats_f, sc)
            open_rays += int(stats_i[0])
            r0 = int(field.lv_ray0[lvl, fam])
            counts[r0:r0 + len(nseg)] = nseg
            idx = np.repeat(offsets[:-1], nseg) + (np.arange(int(nseg.sum())) - np.repeat(np.cumsum(nseg) - nseg, nseg))
            pieces.append((r0, seg[idx] if len(idx) else np.zeros((0, 2), np.int32)))
    if open_rays:
        raise InputError(f"CAD part mesh is not closed: {open_rays} lattice rays cross it an odd number of times")
    start = np.zeros(nray + 1, np.int64)
    np.cumsum(counts, out=start[1:])
    seg_all = np.zeros((int(start[-1]), 2), np.int32)
    for r0, seg in pieces:
        if len(seg):
            seg_all[int(start[r0]):int(start[r0]) + len(seg)] = seg
    info = np.array([1, int(start[-1]), 0, 0], np.int64)
    return A.TargetField(start, np.ascontiguousarray(seg_all), info), {"cad_volume_m3": vol, "triangles": int(len(Fc))}


def read_mesh_file(path, unit: str = "mm", chordal_tolerance_m: float | None = None):
    """(vertices m, faces) of a mesh or STEP file through cnctwin.geom.io."""
    from ..geom.io import read_mesh

    path = Path(path)
    if not path.is_file():
        raise InputError(f"mesh file {path} does not exist")
    mesh = read_mesh(path, unit=unit, chordal_tolerance_m=chordal_tolerance_m)
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32)


def transform_points(points, pose4) -> np.ndarray:
    T = np.asarray(pose4, dtype=float)
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.ascontiguousarray(P @ T[:3, :3].T + T[:3, 3])
