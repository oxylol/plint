"""Small inline fixtures for the cutting tests: ideal tools with constant coefficients, a few tooling-built
tools (file-key items in the db/_templates/tool.yaml form) and helpers to build kernel layouts."""

from __future__ import annotations

import copy
import math

import numpy as np

from cnctwin.cutting import coefficients as CO
from cnctwin.cutting import geometry as GE

# Altintas (2012) Table 2.2-type linear coefficients (Al 7050 with a carbide end mill), N/m2 and N/m
ALTINTAS = dict(Ktc=796e6, Krc=168e6, Kac=222e6, Kte=27.7e3, Kre=30.8e3, Kae=1.8e3)

BASE = {"hand": "right", "tool_material": {"class": "carbide", "id": "carbide-submicron-10co"},
        "coating": {"id": "tialn", "thickness_um": 3}}
HOLDER = {"id": "h1", "type": "shrink_fit", "taper": "HSK-A63", "bore_mm": 10, "gauge_length_mm": 80,
          "nose_diameter_mm": 24, "profile_mm": [[0, 24], [40, 27], [60, 44], [80, 44]], "runout_um": 3,
          "runout_at_mm": 30, "max_rpm": 30000, "mass_kg": 1.2, "clamp_length_mm": 40}


def item(kind: str, **geometry) -> dict:
    out = copy.deepcopy(BASE)
    out["id"] = f"t-{kind.replace('_', '-')}"
    out["type"] = kind
    out["geometry"] = copy.deepcopy(geometry)
    return out


TOOLS = {
    "square_endmill": item("square_endmill", diameter_mm=10, flutes=4, helix_deg=35, rake_deg=8,
                           clearance_deg={"primary": 10, "secondary": 18}, cutting_length_mm=20, flute_length_mm=22,
                           overall_length_mm=72, shank={"diameter_mm": 10, "length_mm": 40}, edge_radius_um=8,
                           corner_radius_mm=0, core_diameter_mm=6, gullet_area_mm2=9.5, centre_cutting=True),
    "twist_drill": item("twist_drill", diameter_mm=10, flutes=2, point_angle_deg=118, helix_deg=30,
                        cutting_length_mm=60, overall_length_mm=120, web_thickness_mm=1.5, chisel_edge_angle_deg=125,
                        margin_width_mm=0.7, clearance_deg={"primary": 10}, shank={"diameter_mm": 10}),
    "tap_cutting": item("tap_cutting", diameter_mm=10, flutes=3,
                        thread={"pitch_mm": 1.5, "form_angle_deg": 60, "hand": "right"}, cutting_length_mm=12,
                        overall_length_mm=90, chamfer_threads=4, spiral="point", shank={"diameter_mm": 10}),
    "tap_forming": item("tap_forming", diameter_mm=8, thread={"pitch_mm": 1.25}, cutting_length_mm=10,
                        overall_length_mm=90, chamfer_threads=3, shank={"diameter_mm": 8}),
    "turning_insert": item("turning_insert",
                           insert={"iso_code": "CNMG120408", "shape": "C", "ic_mm": 12.7, "nose_radius_mm": 0.8,
                                   "chip_breaker": {"rake_deg": 12}},
                           holder={"approach_angle_deg": 95, "shank_mm": [25, 25], "overhang_mm": 32}),
    "threading_insert": item("threading_insert", insert={"nose_radius_mm": 0.1, "clearance_deg": 0},
                             thread={"pitch_mm": 1.5}, holder={"overhang_mm": 30, "inclination_deg": 1.5}),
    "face_mill": item("face_mill", diameter_mm=50, cutting_length_mm=6, overall_length_mm=40,
                      inserts={"count": 5, "ic_mm": 13, "nose_radius_mm": 0.5, "lead_angle_deg": 45,
                               "axial_rake_deg": 6, "radial_rake_deg": -6.5, "edge_length_mm": 9.4, "wiper": True,
                               "iso_code": "SNMU1307", "positions_deg": [0, 72, 144, 216, 288]}),
    "thread_mill": item("thread_mill", diameter_mm=8, flutes=3, thread={"pitch_mm": 1.5}, cutting_length_mm=12,
                        overall_length_mm=63, helix_deg=10, rake_deg=5, clearance_deg={"primary": 10},
                        shank={"diameter_mm": 8}),
    "reamer": item("reamer", diameter_mm=10, flutes=6, pitch_deg=[58, 62, 60, 58, 62, 60], cutting_length_mm=30,
                   overall_length_mm=120, rake_deg=0, clearance_deg={"primary": 8}, shank={"diameter_mm": 10}),
    "chamfer_mill": item("chamfer_mill", diameter_mm=6, flutes=4, tip_angle_deg=90, tip_diameter_mm=1.5, helix_deg=0,
                         rake_deg=0, clearance_deg={"primary": 10}, overall_length_mm=63, shank={"diameter_mm": 6},
                         centre_cutting=False, core_diameter_mm=3.3),
    "engraver": item("engraver", diameter_mm=3, flutes=1, tip_angle_deg=60, tip_width_mm=0.0, rake_deg=0,
                     clearance_deg={"primary": 10}, cutting_length_mm=2.6, overall_length_mm=38,
                     shank={"diameter_mm": 3}),
    "slitting_saw": item("slitting_saw", diameter_mm=25, width_mm=1.0, flutes=20, helix_deg=0, rake_deg=5,
                         clearance_deg={"primary": 10}, hub={"diameter_mm": 10, "width_mm": 1.0}),
    "ball_endmill": item("ball_endmill", diameter_mm=10, flutes=2, helix_deg=30, rake_deg=8,
                         clearance_deg={"primary": 10}, cutting_length_mm=20, overall_length_mm=72,
                         shank={"diameter_mm": 10}, corner_radius_mm=5, edge_radius_um=8),
    "boring_head": item("boring_head", diameter_mm=30, diameter_range_mm=[25, 50],
                        insert={"nose_radius_mm": 0.4, "shape": "C", "ic_mm": 6.35, "clearance_deg": 7},
                        holder={"approach_angle_deg": 90, "rake_deg": 0, "inclination_deg": 0},
                        overall_length_mm=80, shank={"diameter_mm": 20}),
}


def instance(number: int = 1, **changes) -> dict:
    out = {"number": number, "stickout_mm": 35, "runout_um": 0}
    out.update(changes)
    return out


def linear_layout(tool, coeffs: dict, omega: float = -1047.0, h_hi: float = 6e-4, v_min: float = 0.0,
                  VB=None, n_lo: int = 16, n_hi: int = 384, options: GE.ToolOptions | None = None, float_holder=None):
    """(CutTool, CutTable, rows) of a tool with constant coefficients (cnctwin.cutting.coefficients.LinearModel)."""
    model = CO.LinearModel(coeffs)
    rows_geom, elem_row = GE.row_plan(tool, tool.edge, omega, v_min)
    grid = CO.log_grid(5e-8, max(2e-3, h_hi), 28)
    rows = [model.rows((0, r), g, CO.RowState(speed_m_s=abs(omega) * max(g.radius_m, 1e-3)), grid)
            for r, g in enumerate(rows_geom)]
    table = CO.build_table(rows, h_hi, n_lo, n_hi, v_min)
    ct = GE.build_cut_tool(tool, tool.edge, tool.frames, rows, elem_row, VB, options or GE.ToolOptions(),
                           float_holder=float_holder)
    return ct, table, rows


def built(kind: str, **instance_changes):
    """A tooling-built tool with its effective edge (nominal) and frames, as the domain would use it."""
    from cnctwin.tooling import build_tool
    holder = None if kind in ("turning_insert", "threading_insert") else HOLDER
    g = build_tool(TOOLS[kind], holder, instance(**instance_changes))
    return g


def steady_tip(vf_m_s: float, y0: float = 0.0, z0: float = 0.0, vz_m_s: float = 0.0):
    return lambda t: np.array([vf_m_s * t, y0, z0 + vz_m_s * t])


def rev_steps(omega: float, dt: float) -> int:
    return int(round(2.0 * math.pi / abs(omega) / dt))
