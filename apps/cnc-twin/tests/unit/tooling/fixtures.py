"""Small inline tool items (file-key mappings, db/_templates/tool.yaml form) for the tooling tests."""

from __future__ import annotations

import copy

BASE = {"hand": "right", "tool_material": {"class": "carbide", "id": "carbide-submicron-10co"},
        "coating": {"id": "tialn", "thickness_um": 3}}

ENDMILL = dict(diameter_mm=10, flutes=4, helix_deg=35, rake_deg=8, clearance_deg={"primary": 10, "secondary": 18},
               cutting_length_mm=20, flute_length_mm=22, overall_length_mm=72,
               shank={"diameter_mm": 10, "length_mm": 40}, edge_radius_um=8, corner_radius_mm=0,
               core_diameter_mm=6, gullet_area_mm2=9.5, centre_cutting=True)

HOLDER = {"id": "h1", "type": "shrink_fit", "taper": "HSK-A63", "bore_mm": 10, "gauge_length_mm": 80,
          "nose_diameter_mm": 24, "profile_mm": [[0, 24], [40, 27], [60, 44], [80, 44]], "runout_um": 3,
          "runout_at_mm": 30, "max_rpm": 30000, "mass_kg": 1.2, "balance_grade": "G2.5 at 25000 rpm",
          "clamp_length_mm": 40}


def item(kind: str, **geometry) -> dict:
    out = copy.deepcopy(BASE)
    out["id"] = f"t-{kind.replace('_', '-')}"
    out["type"] = kind
    out["geometry"] = copy.deepcopy(geometry)
    return out


def endmill(kind: str = "square_endmill", **changes) -> dict:
    geometry = copy.deepcopy(ENDMILL)
    geometry.update(changes)
    return item(kind, **geometry)


TOOLS = {
    "square_endmill": endmill(),
    "ball_endmill": endmill("ball_endmill", corner_radius_mm=5, flutes=2),
    "toroidal_endmill": endmill("toroidal_endmill", corner_radius_mm=1.0, pitch_deg=[86, 94, 86, 94],
                                helix_deg=[36, 38, 36, 38]),
    "roughing_endmill": endmill("roughing_endmill", corner_radius_mm=0.5),
    "serrated_endmill": endmill("serrated_endmill", cutting_length_mm=10),
    "chamfer_mill": item("chamfer_mill", diameter_mm=6, flutes=4, tip_angle_deg=90, tip_diameter_mm=1.5, helix_deg=0,
                         rake_deg=0, clearance_deg={"primary": 10}, overall_length_mm=63, shank={"diameter_mm": 6},
                         centre_cutting=False, core_diameter_mm=3.3),
    "engraver": item("engraver", diameter_mm=3, flutes=1, tip_angle_deg=60, tip_width_mm=0.0, rake_deg=0,
                     clearance_deg={"primary": 10}, cutting_length_mm=2.6, overall_length_mm=38,
                     shank={"diameter_mm": 3}),
    "keyseat": item("keyseat", diameter_mm=9.525, width_mm=3.175, flutes=8, helix_deg=0, rake_deg=5,
                    clearance_deg={"primary": 10}, neck={"diameter_mm": 4.76, "length_mm": 14.3},
                    overall_length_mm=63.5, shank={"diameter_mm": 9.525}),
    "tslot": item("tslot", diameter_mm=19.5, width_mm=5, flutes=6, corner_radius_mm=0.4, helix_deg=0, rake_deg=5,
                  clearance_deg={"primary": 10}, neck={"diameter_mm": 12, "length_mm": 90}, overall_length_mm=95,
                  shank={"diameter_mm": 12}),
    "dovetail": item("dovetail", diameter_mm=6.35, tip_angle_deg=60, width_mm=3.175, flutes=2, corner_radius_mm=0.25,
                     helix_deg=0, rake_deg=5, clearance_deg={"primary": 10}, neck={"diameter_mm": 2.4, "length_mm": 10},
                     overall_length_mm=50.8, shank={"diameter_mm": 6.35}),
    "lollipop": item("lollipop", diameter_mm=6, flutes=4, helix_deg=30, rake_deg=5, clearance_deg={"primary": 10},
                     wrap_angle_deg=270, neck={"diameter_mm": 3.76, "length_mm": 9.5}, overall_length_mm=63.5,
                     shank={"diameter_mm": 6.35}),
    "slitting_saw": item("slitting_saw", diameter_mm=25, width_mm=1.0, flutes=20, helix_deg=0, rake_deg=5,
                         clearance_deg={"primary": 10}, hub={"diameter_mm": 10, "width_mm": 1.0}),
    "spot_drill": item("spot_drill", diameter_mm=10, flutes=2, tip_angle_deg=90, overall_length_mm=80,
                       shank={"diameter_mm": 10}, clearance_deg={"primary": 10}),
    "centre_drill": item("centre_drill", diameter_mm=2.5, point_angle_deg=118, overall_length_mm=50,
                         shank={"diameter_mm": 6.3}, cutting_length_mm=3.1, clearance_deg={"primary": 10}),
    "twist_drill": item("twist_drill", diameter_mm=10, flutes=2, point_angle_deg=118, helix_deg=30,
                        cutting_length_mm=60, overall_length_mm=120, web_thickness_mm=1.5, chisel_edge_angle_deg=125,
                        margin_width_mm=0.7, clearance_deg={"primary": 10}, shank={"diameter_mm": 10},
                        coolant_holes={"count": 2, "diameter_mm": 0.8, "exit": "tip"}),
    "indexable_drill": item("indexable_drill", diameter_mm=20, cutting_length_mm=60, overall_length_mm=140,
                            shank={"diameter_mm": 25},
                            inserts={"count": 2, "lead_angle_deg": 88, "nose_radius_mm": 0.4, "axial_rake_deg": 5,
                                     "clearance_deg": 7, "edge_length_mm": 6}),
    "gun_drill": item("gun_drill", diameter_mm=8, point_angle_deg=130, cutting_length_mm=20, overall_length_mm=400,
                      shank={"diameter_mm": 20}),
    "reamer": item("reamer", diameter_mm=10, flutes=6, pitch_deg=[58, 62, 60, 58, 62, 60], cutting_length_mm=30,
                   overall_length_mm=120, rake_deg=0, clearance_deg={"primary": 8}, shank={"diameter_mm": 10}),
    "tap_cutting": item("tap_cutting", diameter_mm=10, flutes=3,
                        thread={"pitch_mm": 1.5, "form_angle_deg": 60, "hand": "right"}, cutting_length_mm=12,
                        overall_length_mm=90, chamfer_threads=4, spiral="point", shank={"diameter_mm": 10}),
    "tap_forming": item("tap_forming", diameter_mm=8, thread={"pitch_mm": 1.25}, cutting_length_mm=10,
                        overall_length_mm=90, chamfer_threads=3, shank={"diameter_mm": 8}),
    "thread_mill": item("thread_mill", diameter_mm=8, flutes=3, thread={"pitch_mm": 1.5}, cutting_length_mm=12,
                        overall_length_mm=63, helix_deg=10, rake_deg=5, clearance_deg={"primary": 10},
                        shank={"diameter_mm": 8}),
    "face_mill": item("face_mill", diameter_mm=50, cutting_length_mm=6, overall_length_mm=40,
                      inserts={"count": 5, "ic_mm": 13, "nose_radius_mm": 0.5, "lead_angle_deg": 45,
                               "axial_rake_deg": 6, "radial_rake_deg": -6.5, "edge_length_mm": 9.4, "wiper": True,
                               "iso_code": "SNMU1307", "positions_deg": [0, 72, 144, 216, 288]}),
    "shoulder_mill": item("shoulder_mill", diameter_mm=16, cutting_length_mm=11.5, overall_length_mm=85,
                          shank={"diameter_mm": 16, "length_mm": 60},
                          inserts={"count": 2, "nose_radius_mm": 0.8, "lead_angle_deg": 90, "axial_rake_deg": 6.8,
                                   "radial_rake_deg": -10, "edge_length_mm": 11.8, "iso_code": "AVMT120408"}),
    "boring_head": item("boring_head", diameter_mm=30, diameter_range_mm=[25, 50],
                        insert={"nose_radius_mm": 0.4, "shape": "C", "ic_mm": 6.35, "clearance_deg": 7},
                        holder={"approach_angle_deg": 90, "rake_deg": 0, "inclination_deg": 0},
                        overall_length_mm=80, shank={"diameter_mm": 20}),
    "turning_insert": item("turning_insert",
                           insert={"iso_code": "CNMG120408", "shape": "C", "ic_mm": 12.7, "nose_radius_mm": 0.8,
                                   "chip_breaker": {"rake_deg": 12}},
                           holder={"approach_angle_deg": 95, "shank_mm": [25, 25], "overhang_mm": 32}),
    "boring_bar": item("boring_bar", insert={"shape": "D", "ic_mm": 9.525, "nose_radius_mm": 0.4, "clearance_deg": 7},
                       holder={"approach_angle_deg": 93},
                       bar={"diameter_mm": 20, "overhang_mm": 80, "material": "steel"}, min_bore_mm=25),
    "grooving_insert": item("grooving_insert",
                            insert={"width_mm": 3, "nose_radius_mm": 0.2, "clearance_deg": 7, "max_depth_mm": 6},
                            holder={"overhang_mm": 30, "shank_mm": [20, 20]}),
    "parting_insert": item("parting_insert", insert={"width_mm": 2, "nose_radius_mm": 0.1, "clearance_deg": 7},
                           holder={"overhang_mm": 30, "approach_angle_deg": 85}),
    "threading_insert": item("threading_insert", insert={"nose_radius_mm": 0.1, "clearance_deg": 0},
                             thread={"pitch_mm": 1.5}, holder={"overhang_mm": 30, "inclination_deg": 1.5}),
    "form_tool": item("form_tool", profile_mm=[[0, 0], [1, 0.5], [3, 0.5], [4, 0]], holder={"overhang_mm": 20}),
}

NON_ROTATING = ("turning_insert", "boring_bar", "grooving_insert", "parting_insert", "threading_insert", "form_tool")


def instance(number: int = 1, **changes) -> dict:
    out = {"number": number, "stickout_mm": 35}
    out.update(changes)
    return out
