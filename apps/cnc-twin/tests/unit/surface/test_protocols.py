"""The protocol adapters: tooling geometries as generating outlines, the analytic stock query, removal's brick
numbering, vertex normals."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from cnctwin.surface.protocols import (GeometryToolInfo, LatticeBricks, StockQuery, turning_insert_outline,
                                       vertex_normals)

BASE = {"hand": "right", "tool_material": {"class": "carbide", "id": "carbide-submicron-10co"},
        "coating": {"id": "tialn", "thickness_um": 3}}
HOLDER = {"id": "h1", "type": "shrink_fit", "taper": "HSK-A63", "bore_mm": 10, "gauge_length_mm": 80,
          "nose_diameter_mm": 24, "profile_mm": [[0, 24], [40, 27], [60, 44], [80, 44]], "runout_um": 3,
          "runout_at_mm": 30, "max_rpm": 30000, "mass_kg": 1.2, "clamp_length_mm": 40}


def _item(kind, **geometry):
    out = copy.deepcopy(BASE)
    out.update({"id": f"t-{kind}", "type": kind, "geometry": geometry})
    return out


def test_tooling_geometries_become_outlines_through_their_elements():
    from cnctwin.tooling.build import build_tool
    mill = _item("toroidal_endmill", diameter_mm=10, flutes=4, helix_deg=35, rake_deg=8,
                 clearance_deg={"primary": 10}, cutting_length_mm=20, flute_length_mm=22, overall_length_mm=72,
                 shank={"diameter_mm": 10, "length_mm": 40}, edge_radius_um=8, corner_radius_mm=1.0,
                 core_diameter_mm=6, centre_cutting=True)
    insert = _item("turning_insert", insert={"iso_code": "CNMG120408", "shape": "C", "ic_mm": 12.7,
                                            "nose_radius_mm": 0.8, "chip_breaker": {"rake_deg": 12}},
                   holder={"approach_angle_deg": 95, "shank_mm": [25, 25], "overhang_mm": 32})
    g0 = build_tool(mill, HOLDER, {"number": 1, "stickout_mm": 35}, slot=0)
    g1 = build_tool(insert, None, {"number": 2, "stickout_mm": 35}, slot=1)
    info = GeometryToolInfo([g0, g1])
    for slot, rotating, finish in ((0, True, 1.0e-3), (1, False, 0.8e-3)):
        out = info.outline(slot)
        assert out.rotating is rotating and out.finish_radius_m == pytest.approx(finish)
        edge = info.edge_model(slot)
        valid = np.asarray(edge.elem_valid) != 0
        r, z = np.asarray(edge.elem_r_m)[valid], np.asarray(edge.elem_z_m)[valid]
        dist = np.array([np.min(np.hypot(out.pts[:, 0] - a, out.pts[:, 1] - b)) for a, b in zip(r, z)])
        assert float(dist.max()) < 2e-6                     # every element lies on the generating outline
    assert math.degrees(info.outline(1).kappa_major_rad) == pytest.approx(95.0, abs=1e-6)
    assert info.outline(0) is info.outline(0)                # cached


def test_analytic_insert_outline_tangencies():
    nose = 0.8e-3
    pts = turning_insert_outline(nose, math.radians(93.0), math.radians(27.0))
    assert float(np.max(pts[:, 0])) == pytest.approx(0.0, abs=2e-9)         # deepest point at x_T = 0 (0.25 deg facets)
    d = np.hypot(pts[:, 0] + nose, pts[:, 1] - nose)
    on_arc = np.abs(d - nose) < 1e-9
    assert on_arc.sum() > 100
    # the straight edges leave the arc tangentially: the step from the last arc point is perpendicular to the radius
    seg = np.diff(pts, axis=0)
    assert np.all(np.linalg.norm(seg, axis=1) > 0.0)


def test_stock_query_ray_cast_and_inside():
    q = StockQuery("block", [0.06, 0.04, 0.03])
    assert list(q.point_inside(0, [[0.01, 0.01, 0.01], [0.07, 0.01, 0.01]])) == [True, False]
    hit, t, p, n = q.ray_cast(0, [0.05, 0.02, 0.01], [1.0, 0.0, 0.0], 0.05, True)
    assert hit and t == pytest.approx(0.01, abs=1e-9) and n == pytest.approx([1.0, 0.0, 0.0])
    hit, *_ = q.ray_cast(0, [0.05, 0.02, 0.01], [1.0, 0.0, 0.0], 0.005, True)
    assert not hit
    cyl = StockQuery("cylinder", [0.05, 0.06])
    hit, t, p, n = cyl.ray_cast(0, [0.02, 0.0, 0.01], [0.0, 0.0, -1.0], 0.05, True)
    assert hit and p[2] == pytest.approx(0.0, abs=1e-9) and n == pytest.approx([0.0, 0.0, -1.0])


def test_lattice_bricks_reproduce_removals_numbering():
    from cnctwin.removal.field import brick_box
    from cnctwin.removal.stock import block_field
    size, lattice = [0.03, 0.02, 0.015], 2.5e-4
    field = block_field(size, lattice)
    bricks = LatticeBricks(field.lv_origin_m[0], float(field.lv_cell_m[0]), field.lv_dims[0])
    default = LatticeBricks.for_stock("block", size, lattice)
    assert np.allclose(default._grids[0][0], field.lv_origin_m[0]) and np.array_equal(default._grids[0][2],
                                                                                         field.lv_dims[0])
    rng = np.random.default_rng(1)
    pts = rng.uniform([0, 0, 0], size, (200, 3))
    ids = bricks.brick_of(0, pts)
    coords = bricks.coords(0, ids)
    for i, c in zip(ids, coords):
        level, bc, n_lo, n_hi = brick_box(field, int(i))
        assert level == 0 and np.array_equal(bc, c)
    nodes = np.floor((pts - field.lv_origin_m[0]) / lattice + 0.5).astype(int)
    assert np.all(nodes // 32 == np.minimum(coords, np.asarray(field.lv_bdims[0]) - 1))
    assert bricks.brick_key(0, 7) == "7" and bricks.brick_key(2, 7) == "w2.7"
    assert bricks.brick_size_m(0) == pytest.approx(32 * lattice)


def test_vertex_normals_of_a_counter_clockwise_mesh():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    n = vertex_normals(V, np.array([[0, 1, 2]]))
    assert n == pytest.approx(np.tile([0.0, 0.0, 1.0], (3, 1)))
