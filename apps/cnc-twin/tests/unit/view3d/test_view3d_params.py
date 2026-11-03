"""The 3D view's settings: db/models/view3d.yaml is valid (units, sources, ranges: cnctwin.db.models), the
browser's web/js/view3d/params.js is written from it (sync_params.py), and the values it shares with other
models stay equal to them: the coolant model's cutting zone, standard gravity, and the appearance and
cameras of the run pictures (db/models/render.yaml)."""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import sync_params  # noqa: E402
from cnctwin import units  # noqa: E402
from cnctwin.db.models import params  # noqa: E402


def test_view3d_parameters_are_valid_and_si():
    p = params("view3d")
    assert math.isclose(p["fov_rad"], math.radians(30))
    assert math.isclose(p["rebuild_budget_s"], 0.008)
    assert math.isclose(p["jet_max_length_m"], 0.6)
    assert math.isclose(p["pose_chain_tolerance_m"], 1e-4)
    assert p["lod_cell_factors"] == [4, 16]
    assert set(p["material_srgb"]) >= {"aluminium", "steel", "titanium", "default"}


def test_params_js_is_current():
    assert sync_params.main(["--check"]) == 0, "web/js/view3d/params.js is stale: run tests/unit/view3d/sync_params.py"


def test_shared_values_stay_equal():
    view = params("view3d")
    zone = params("coolant-occlusion")
    assert view["zone_radius_factor"] == zone["zone_radius_factor"]
    assert view["min_zone_radius_m"] == zone["zone_min_radius_m"]
    assert view["tool_zone_height_ratio"] == zone["tool_zone_height_ratio"]
    assert view["standard_gravity_m_s2"] == units.STANDARD_GRAVITY_M_S2
    render = params("render")
    for name in ("material_srgb", "tool_srgb", "holder_srgb", "machine_srgb", "fixture_srgb", "cap_srgb",
                 "enclosure_srgb", "enclosure_opacity", "door_opacity", "jet_max_length_m", "jet_opacity",
                 "jet_end_opacity_fraction", "jet_sides", "chip_aspect", "chip_default_length_m", "framing_margin",
                 "follow_view_height_m", "key_light_azimuth_rad", "key_light_elevation_rad", "crease_angle_rad"):
        assert view[name] == render[name], name
    assert view["min_chip_size_px"] == render["chip_min_size_px"]
    assert view["min_pile_height_m"] == render["pile_min_height_m"]
    assert view["fov_rad"] == render["follow_fov_rad"]
    assert view["view_azimuth_rad"] == render["follow_azimuth_rad"]
    assert view["view_elevation_rad"] == render["follow_elevation_rad"]
