"""Small inline fixtures for the mechanics tests (file-style keys, read through cnctwin.db.simap.wrap
exactly as database objects are). Values are those of db/materials/aisi1045.yaml and
db/tool_materials/carbide-k20-6co.yaml as of 2026-11-03, copied so the tests do not depend on the
databases still being researched."""

from __future__ import annotations

import copy

import pytest

from cnctwin.db.simap import wrap
from cnctwin.mechanics.materials import ToolSide, WorkMaterial

STEEL_1045 = {
    "id": "fixture-1045", "family": "steel", "iso_group": "P",
    "physical": {
        "density_kg_m3": 7850,
        "specific_heat_J_kgK": {"table": [[20, 460], [100, 480], [200, 498], [300, 524], [400, 524], [500, 615],
                                          [600, 690], [700, 720], [800, 682], [900, 637], [1000, 602]]},
        "conductivity_W_mK": {"table": [[20, 49.77], [100, 46.76], [200, 43.24], [300, 40.29], [400, 37.87],
                                        [500, 35.96], [600, 33.18], [700, 30.52], [800, 27.96], [900, 25.92],
                                        [1000, 24.02]]},
        "youngs_modulus_GPa": 206, "poisson": 0.3, "yield_MPa": 420, "ultimate_MPa": 671,
        "solidus_degC": 1445, "liquidus_degC": 1500, "fracture_strain": 0.6,
    },
    "flow_stress": {"model": "johnson_cook", "A_MPa": 553.1, "B_MPa": 600.8, "n": 0.234, "C": 0.0134, "m": 1.0,
                    "ref_strain_rate_1_s": 1.0, "T_ref_degC": 20, "T_melt_degC": 1490},
    "cutting": {
        "kienzle": [{"tool_class": "carbide", "coating": "uncoated", "kc11_N_mm2": 1672, "mc": 0.32,
                     "kf11_N_mm2": 809, "mf": 0.65, "h_ref_mm": 1, "b_ref_mm": 1, "rake_ref_deg": 0,
                     "vc_ref_m_min": 100, "src": "klippel-2022-ck45-cutting-database",
                     "calibration": {"vc_m_min": 100, "h_mm": [0.05, 0.4], "edge_radius_um": 37.5,
                                     "coolant": "dry", "work_temp_degC": 20}}],
        "mechanistic": [],
        "min_chip_thickness_ratio": 0.27,
        "indentation_coeff_N_mm3": {"value": 2.9e4, "src": "est"},
    },
    "friction": [{"tool_class": "carbide", "coating": "uncoated", "mu_sliding": 0.5, "sticking_ratio": 0.5},
                 {"tool_class": "carbide", "coating": "TiN", "mu_sliding": 0.4}],
    "chip": {
        "tendency": "continuous",
        "segmentation": {"criterion": "critical_speed", "vc_onset_m_min": 300},
        "bue": {"interface_temp_degC": [200, 550], "vc_m_min": [10, 50]},
        "breakability": "medium",
        # rows as the database loader delivers them: [T_K, "#rrggbb", name] (the file writes degC)
        "oxide_colours": [[473.15, "#EDE4B0", "white-yellow"], [493.15, "#E3C770", "straw yellow"],
                          [513.15, "#B98745", "yellow-brown"], [533.15, "#8C3A2E", "red"],
                          [553.15, "#5E2C78", "violet"], [573.15, "#3C64C4", "cornflower blue"],
                          [633.15, "#8E8E8E", "grey"]],
    },
    "thermal_damage": {"melt_smear_degC": None},
}

POLYMER = {
    "id": "fixture-polymer", "family": "polymer", "iso_group": "O",
    "physical": {"density_kg_m3": 1140, "specific_heat_J_kgK": 1700, "conductivity_W_mK": 0.25,
                 "youngs_modulus_GPa": 3.0, "poisson": 0.4, "yield_MPa": 84, "ultimate_MPa": 90,
                 "glass_transition_degC": 47, "fracture_strain": 0.5},
    "flow_stress": None,
    "cutting": {"kienzle": [{"tool_class": "carbide", "coating": "uncoated", "kc11_N_mm2": 150, "mc": 0.2,
                             "h_ref_mm": 1, "b_ref_mm": 1, "rake_ref_deg": 0, "vc_ref_m_min": 300}],
                "min_chip_thickness_ratio": 0.45},
    "friction": [{"tool_class": "carbide", "coating": "uncoated", "mu_sliding": 0.35}],
    "chip": {"tendency": "continuous", "breakability": "poor"},
    "thermal_damage": {"melt_smear_degC": 258},
}

COMPOSITE = {
    "id": "fixture-cfrp", "family": "composite", "iso_group": "O",
    "physical": {"density_kg_m3": 1550, "specific_heat_J_kgK": 900, "conductivity_W_mK": 0.8,
                 "youngs_modulus_GPa": 60, "poisson": 0.3, "yield_MPa": 600, "ultimate_MPa": 600},
    "flow_stress": None,
    "cutting": {"kienzle": [{"tool_class": "carbide", "coating": "uncoated", "kc11_N_mm2": 300, "mc": 0.3,
                             "h_ref_mm": 1, "b_ref_mm": 1}], "min_chip_thickness_ratio": 0.5},
    "friction": [{"tool_class": "carbide", "coating": "uncoated", "mu_sliding": 0.25}],
    "chip": {"tendency": "powder", "breakability": "good"},
    "composite": {"fibre": "carbon", "fibre_volume_fraction": 0.6, "dust_mass_fraction": 1.0},
}

CARBIDE = {
    "id": "fixture-k20", "class": "carbide", "density_kg_m3": 15000,
    "conductivity_W_mK": {"table": [[20, 79], [500, 67], [1000, 55]]},
    "specific_heat_J_kgK": {"table": [[20, 195], [400, 230], [800, 255]]},
}

TIALN = {"id": "fixture-tialn", "conductivity_W_mK": 3.0, "typical_thickness_um": [1, 4]}


def work_from(raw: dict, **changes) -> WorkMaterial:
    data = copy.deepcopy(raw)
    for path, value in changes.items():
        node = data
        keys = path.split("__")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return WorkMaterial.from_db(wrap(data))


@pytest.fixture(scope="session")
def steel() -> WorkMaterial:
    return work_from(STEEL_1045)


@pytest.fixture(scope="session")
def polymer() -> WorkMaterial:
    return work_from(POLYMER)


@pytest.fixture(scope="session")
def composite() -> WorkMaterial:
    return work_from(COMPOSITE)


@pytest.fixture(scope="session")
def carbide() -> ToolSide:
    return ToolSide.from_db(None, wrap(CARBIDE), None)


@pytest.fixture(scope="session")
def coated_carbide() -> ToolSide:
    tool = {"type": "turning_insert", "coating": {"id": "tialn", "thickness_um": 3},
            "geometry": {"insert": {"nose_radius_mm": 0.8,
                                    "chip_breaker": {"land_mm": 0.15, "groove_width_mm": 1.0,
                                                     "groove_depth_mm": 0.2, "rake_deg": 15,
                                                     "window": {"f_mm": [0.1, 0.4], "ap_mm": [0.5, 4.0]}}}}}
    return ToolSide.from_db(wrap(tool), wrap(CARBIDE), wrap(TIALN))
