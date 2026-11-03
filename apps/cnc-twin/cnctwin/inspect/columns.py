"""Remaining material and gouges: the dexel field against the CAD target, as connected regions with location and
volume (docs/models/inspect.md "Remaining material and gouges").

Both solids are z-family dexel columns on the same lattice (:class:`cnctwin.inspect.abi.Columns`): the removal
field and its target directly (:class:`cnctwin.inspect.protocols.FieldColumns`, removal's own rays, so the
comparison is exact to removal's 10 nm quantum), or any closed mesh dexelised here by casting +z rays through the
column nodes (:func:`columns_from_mesh`, the crossings from cnctwin.geom.bvh.rays_all_hits).

* **remaining material** = field minus target per column, intervals thinner than ``remaining_tolerance_m``
  dropped; **gouges** = target minus field, thinner than ``gouge_tolerance_m`` dropped;
* **regions**: intervals of face-adjacent columns that overlap in z are connected (union-find, smallest index
  first: deterministic); each region gets its volume ``cx cy sum(hi - lo)`` (the same midpoint rule as removal's
  volume), centroid, box, largest thickness and where it is, and the CAD face nearest to its centroid.

The volume of a region is exact for the lattice (it is the sum the field itself holds); against the true
geometry its error is that of the z-dexel representation: walls cut the column cells, an error of up to
``perimeter x cell / 2 x height`` per region boundary in the worst case, about ``perimeter x cell^2 / 12`` on
average for walls at random positions.
"""

from __future__ import annotations

import math

import numpy as np

from ..geom.bvh import rays_all_hits
from ..geom.mesh import Mesh
from . import abi as A
from .kernels import columns_difference_nb, columns_from_hits_nb, component_stats_nb, label_components_nb
from .schema import InspectSettings


def lattice_for(lo, hi, cell_m: float, margin_cells: int = 1, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray,
                                                                                            np.ndarray]:
    """(origin, cell, dims) of a column lattice covering the xy box [lo, hi] with nodes at ``(k + offset) cell``
    plus ``margin_cells`` columns of air on each side. ``offset = 0.5`` keeps the rays off faces at round
    coordinates (a CAD's walls), which then lie on cell boundaries: their volumes come out exact."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    i0 = np.floor(lo[:2] / cell_m - offset) - margin_cells
    i1 = np.ceil(hi[:2] / cell_m - offset) + margin_cells
    return (i0 + offset) * cell_m, np.array([cell_m, cell_m]), (i1 - i0 + 1).astype(np.int64)


def columns_from_mesh(mesh: Mesh, origin_m, cell_m, dims, merge_m: float = 1e-9, tree=None) -> A.Columns:
    """z-dexel columns of a closed, outward-oriented mesh on a lattice (see the module documentation)."""
    from ..geom.bvh import build_tree

    origin = np.asarray(origin_m, dtype=np.float64).reshape(2)
    cell = np.broadcast_to(np.asarray(cell_m, dtype=np.float64), (2,)).copy()
    dims = np.asarray(dims, dtype=np.int64).reshape(2)
    tree = build_tree(mesh) if tree is None else tree
    lo, hi = mesh.bounds()
    z0 = float(lo[2]) - 1e-3 * max(1.0, float(hi[2] - lo[2]))
    ii, jj = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing="ij")
    O = np.column_stack([origin[0] + ii.reshape(-1) * cell[0], origin[1] + jj.reshape(-1) * cell[1],
                         np.full(ii.size, z0)])
    offsets, t, face = rays_all_hits(tree, O, np.array([0.0, 0.0, 1.0]))
    face_nz = A.contiguous(tree.face_normal[:, 2])
    start = np.zeros(ii.size + 1, np.int64)
    seg = np.zeros((len(t) // 2 + 1, 2))
    n = int(columns_from_hits_nb(A.contiguous(offsets, np.int64), A.contiguous(t), A.contiguous(face, np.int32),
                                 face_nz, z0, float(merge_m), start, seg))
    return A.make_columns(origin, cell, dims, start, seg[:n])


def same_lattice(a: A.Columns, b: A.Columns) -> bool:
    return bool(np.array_equal(a.dims, b.dims) and np.allclose(a.origin_m, b.origin_m, rtol=0.0, atol=1e-12)
                and np.allclose(a.cell_m, b.cell_m, rtol=0.0, atol=1e-15))


def difference(a: A.Columns, b: A.Columns, min_len_m: float = 0.0) -> A.Columns:
    """A minus B per column (intervals shorter than min_len dropped)."""
    if not same_lattice(a, b):
        raise ValueError("the two column sets lie on different lattices")
    n = int(a.dims[0] * a.dims[1])
    start = np.zeros(n + 1, np.int64)
    seg = np.zeros((len(a.seg) + len(b.seg) + 1, 2))
    m = int(columns_difference_nb(a.start, a.seg, b.start, b.seg, float(min_len_m), start, seg))
    return A.make_columns(a.origin_m, a.cell_m, a.dims, start, seg[:m])


def volume_m3(cols: A.Columns) -> float:
    if not len(cols.seg):
        return 0.0
    return float(np.sum(cols.seg[:, 1] - cols.seg[:, 0]) * cols.cell_m[0] * cols.cell_m[1])


def regions(cols: A.Columns) -> np.ndarray:
    """Rows abi.CS_* of the connected regions, ordered by volume (largest first; ties by first interval)."""
    m = len(cols.seg)
    if m == 0:
        return np.zeros((0, A.N_CS))
    parent = np.zeros(m, np.int64)
    label = np.zeros(m, np.int64)
    count = int(label_components_nb(cols.dims, cols.start, cols.seg, parent, label))
    stats = np.zeros((count, A.N_CS))
    component_stats_nb(cols.origin_m, cols.cell_m, cols.dims, cols.start, cols.seg, label, np.zeros(count, np.int64),
                       stats)
    order = np.lexsort((stats[:, A.CS_FIRST], -stats[:, A.CS_VOLUME]))
    return stats[order]


def region_records(rows: np.ndarray, kind: str, min_volume_m3: float, cad=None) -> tuple[list[dict], dict]:
    """(listed regions, totals) - regions below min_volume are counted, not listed."""
    listed = []
    small_n = 0
    small_v = 0.0
    for r in rows:
        if r[A.CS_VOLUME] < min_volume_m3:
            small_n += 1
            small_v += float(r[A.CS_VOLUME])
            continue
        rec = {"kind": kind, "volume_m3": float(r[A.CS_VOLUME]),
               "centroid_S_m": [float(r[A.CS_CX]), float(r[A.CS_CY]), float(r[A.CS_CZ])],
               "box_lo_S_m": [float(r[A.CS_XMIN]), float(r[A.CS_YMIN]), float(r[A.CS_ZMIN])],
               "box_hi_S_m": [float(r[A.CS_XMAX]), float(r[A.CS_YMAX]), float(r[A.CS_ZMAX])],
               "max_thickness_m": float(r[A.CS_THICK]),
               "max_point_S_m": [float(r[A.CS_TX]), float(r[A.CS_TY]), float(r[A.CS_TZ])],
               "columns": int(r[A.CS_COLUMNS]), "intervals": int(r[A.CS_SEGMENTS])}
        if cad is not None:
            _, tri, _, _ = cad.closest(np.array([rec["max_point_S_m"]]))
            face = int(cad.face_of(tri)[0])
            rec["nearest_face"] = face
            if face >= 0:
                rec["nearest_surface"] = cad.face(face).surface
        listed.append(rec)
    totals = {"regions": int(len(rows)), "listed": len(listed), "volume_m3": float(rows[:, A.CS_VOLUME].sum())
              if len(rows) else 0.0, "unlisted_regions": small_n, "unlisted_volume_m3": small_v,
              "max_thickness_m": float(rows[:, A.CS_THICK].max()) if len(rows) else 0.0}
    return listed, totals


def compare(field: A.Columns, target: A.Columns, settings: InspectSettings, cad=None) -> dict:
    """Remaining material (field - target) and gouges (target - field) with their regions and totals."""
    excess = difference(field, target, settings.remaining_tolerance_m)
    missing = difference(target, field, settings.gouge_tolerance_m)
    rem, rem_tot = region_records(regions(excess), "remaining_material", settings.region_min_volume_m3, cad)
    gou, gou_tot = region_records(regions(missing), "gouge", settings.region_min_volume_m3, cad)
    raw_excess = volume_m3(difference(field, target, 0.0))
    raw_missing = volume_m3(difference(target, field, 0.0))
    return {"remaining": rem, "gouges": gou,
            "totals": {"remaining": rem_tot, "gouges": gou_tot, "field_volume_m3": volume_m3(field),
                       "target_volume_m3": volume_m3(target), "excess_volume_m3": raw_excess,
                       "missing_volume_m3": raw_missing,
                       "remaining_tolerance_m": settings.remaining_tolerance_m,
                       "gouge_tolerance_m": settings.gouge_tolerance_m,
                       "cell_m": [float(v) for v in field.cell_m], "dims": [int(v) for v in field.dims]}}


def compare_meshes(result: Mesh, cad_mesh: Mesh, cell_m: float, settings: InspectSettings, cad=None,
                   cad_tree=None) -> dict:
    """:func:`compare` of two closed meshes dexelised on a common lattice of spacing ``cell_m``."""
    lo = np.minimum(result.bounds()[0], cad_mesh.bounds()[0])
    hi = np.maximum(result.bounds()[1], cad_mesh.bounds()[1])
    origin, cell, dims = lattice_for(lo, hi, cell_m, offset=0.5)
    field = columns_from_mesh(result, origin, cell, dims, settings.column_merge_m)
    target = columns_from_mesh(cad_mesh, origin, cell, dims, settings.column_merge_m, tree=cad_tree)
    out = compare(field, target, settings, cad)
    out["totals"]["source"] = "meshes dexelised at " + f"{cell_m * 1e3:g} mm"
    return out


def suggested_cell(mesh: Mesh, max_columns: int = 1_000_000, fine_m: float = 2.5e-4) -> float:
    """A column spacing for mesh comparisons: ``fine_m`` unless the part needs more than max_columns."""
    lo, hi = mesh.bounds()
    area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    return max(fine_m, math.sqrt(area / max_columns)) if area > 0 else fine_m


__all__ = ["columns_from_mesh", "lattice_for", "difference", "volume_m3", "regions", "region_records", "compare",
           "compare_meshes", "same_lattice", "suggested_cell"]
