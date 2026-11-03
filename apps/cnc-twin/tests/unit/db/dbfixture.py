"""A small, complete database written to a temporary directory (inline fixtures for db and project tests)."""

from __future__ import annotations

import textwrap
from pathlib import Path

BIBLIOGRAPHY = """\
format: cnctwin-bibliography
version: 1
entries:
  src-a:
    type: article
    authors: "A, B."
    title: "Cutting data"
    venue: "J. Test 1, 1-2"
    year: 2020
    pages: "Table 1"
  src-b:
    type: datasheet
    title: "Machine data sheet"
    accessed: 2026-11-03
"""

BIBLIOGRAPHY_2 = """\
format: cnctwin-bibliography
version: 1
entries:
  src-a:                       # same source read again elsewhere: reading fields may differ
    type: article
    authors: "A, B."
    title: "Cutting data"
    venue: "J. Test 1, 1-2"
    year: 2020
    pages: "Table 2"
  src-c:
    type: handbook
    title: "Handbook"
    year: 1990
"""

MATERIAL = """\
format: cnctwin-material
version: 1
id: test-al
name: "Test aluminium"
family: aluminium
iso_group: N
src: src-a
physical:
  density_kg_m3: {value: 2700}
  specific_heat_J_kgK: {table: [[20, 900], [200, 1000]]}
  conductivity_W_mK: {table: [[20, 167]]}
  expansion_1_K: {table: [[20, 23.6e-6]]}
  youngs_modulus_GPa: {table: [[20, 70], [200, 60]], src: src-c}
  poisson: 0.33
  solidus_degC: {value: 582, src: est, note: "from the liquidus less the freezing range"}
  hardness: {HB: 95}
  yield_MPa: 276
  ultimate_MPa: 310
flow_stress:
  model: johnson_cook
  A_MPa: 324
  B_MPa: 114
  n: 0.42
  C: 0.002
  m: 1.34
  ref_strain_rate_1_s: 1.0
  T_ref_degC: 20
  T_melt_degC: 652
cutting:
  kienzle:
    - {tool_class: carbide, kc11_N_mm2: 700, mc: 0.25, rake_ref_deg: 6, vc_ref_m_min: 300}
friction:
  - {tool_class: carbide, mu_sliding: 0.5}
chip:
  tendency: continuous
  bue: {interface_temp_degC: {range: [150, 350]}, vc_m_min: [5, 60]}
tool_life:
  - {tool_class: carbide, C_m_min: 1200, n: 0.3, VB_limit_mm: 0.3}
residual_stress:
  - {form: rolled_plate, thickness_mm: 50, direction: L, profile: [[0.0, -20], [0.5, 15], [1.0, -20]]}
skin: {type: none, thickness_mm: {value: 0}, hardness_factor: {value: 1.0}, abrasiveness_factor: {value: 1.0}}
notes: "fixture"
"""

MACHINE = """\
format: cnctwin-machine
version: 1
id: test-vmc
maker: "Test"
model: "VMC"
type: vertical_mill
src: src-b
controller:
  dialect: haas
  model: "NGC"
  alarms: test-ctrl
  ipo_period_ms: {value: 1.0, src: est, note: "typical servo period"}
  least_input_increment_mm: 0.001
  max_feed_mm_min: 20000
  rapid_mm_min: {X: 30000, Y: 30000, Z: 24000}
axes:
  X:
    type: linear
    travel_mm: [-500, 0]
    rapid_mm_min: 30000
    max_feed_mm_min: 20000
    accel_m_s2: 5.0
    jerk_m_s3: 100
    servo: {kv_1_s: 40}
  Y:
    type: linear
    travel_mm: [-400, 0]
    rapid_mm_min: 30000
    max_feed_mm_min: 20000
    accel_m_s2: 5.0
    jerk_m_s3: 100
    servo: {kv_1_s: 40}
  Z:
    type: linear
    travel_mm: [-400, 0]
    rapid_mm_min: 24000
    max_feed_mm_min: 20000
    accel_m_s2: 5.0
    jerk_m_s3: 100
    servo: {kv_1_s: 40}
kinematics:
  workpiece_chain:
    - {link: saddle, axis: Y, direction: [0, -1, 0]}
    - {link: table, axis: X, direction: [-1, 0, 0]}
  tool_chain:
    - {link: head, axis: Z, direction: [0, 0, 1]}
    - {link: nose, fixed: {translate_mm: [250, 200, 500], rotate_deg: [0, 0, 90]}}
  home_mm: {X: 0, Y: 0, Z: 0}
  tool_axis: [0, 0, 1]
spindles:
  - id: main
    kind: milling
    nose: CAT40
    max_rpm: 12000
    gear_ranges:
      - rpm: [0, 12000]
        power_kW: [[0, 0], [2000, 22.4], [12000, 22.4]]
        torque_Nm: [[0, 122], [2000, 122], [12000, 18]]
    thermal: {growth_um_at_max_rpm: {value: 25, src: est, note: "typical"}, time_constant_min: 30}
probes:
  spindle_probe: {model: "probe", trigger_force_N: 0.5}
  spindle_probe_extra: {pretravel_um: 1, src: est, note: "pretravel assumed"}
"""

MACHINE_EXT = """\
format: cnctwin-machine
version: 1
id: test-vmc-siemens
extends: test-vmc
src: src-a
controller: {dialect: siemens840d, model: "840D sl"}
"""

TOOL_MATERIAL = """\
format: cnctwin-tool-material
version: 1
id: test-carbide
class: carbide
name: "WC-10Co"
src: src-a
density_kg_m3: 14500
youngs_modulus_GPa: 580
poisson: 0.22
hardness_HV: 1600
transverse_rupture_MPa: 4000
conductivity_W_mK: {table: [[20, 80], [800, 55]]}
specific_heat_J_kgK: 200
expansion_1_K: 5.5e-6
max_service_temp_degC: 1000
usui:
  - {work_family: steel, A: 1.0e-12, B_K: 5000, T_range_K: [null, 1150]}
"""

COATINGS = {
    "test-tialn": """\
format: cnctwin-coating
version: 1
id: test-tialn
name: "TiAlN"
process: PVD
src: src-a
hardness_HV: 3300
max_service_temp_degC: 800
conductivity_W_mK: 4.5
typical_thickness_um: [2, 4]
wear_rate_factor: 0.6
thermal_barrier_factor: 0.9
""",
}

TOOLS = """\
format: cnctwin-tool
version: 1
series_id: test-endmills
maker: "Test Tools"
series: "4-flute end mills"
src: src-a
items:
  - id: test-em-10
    type: square_endmill
    hand: right
    tool_material: {class: carbide, id: test-carbide}
    coating: {id: test-tialn, thickness_um: 3}
    geometry:
      diameter_mm: 10
      flutes: 4
      helix_deg: 35
      rake_deg: 8
      cutting_length_mm: 22
      overall_length_mm: 72
      shank: {diameter_mm: 10, length_mm: 40}
    max_rpm: 40000
  - id: test-em-6
    type: square_endmill
    hand: right
    tool_material: {class: carbide, id: test-carbide}
    coating: {id: none}
    geometry:
      diameter_mm: 6
      flutes: 4
      helix_deg: 35
      rake_deg: 8
      cutting_length_mm: 13
      overall_length_mm: 57
      shank: {diameter_mm: 6}
"""

HOLDERS = """\
format: cnctwin-holder
version: 1
series_id: test-holders
maker: "Test Holders"
series: "ER32"
src: src-a
items:
  - id: test-er32
    type: er_collet
    taper: CAT40
    bore_mm: [2, 20]
    gauge_length_mm: 70
    nose_diameter_mm: 50
    profile_mm: [[0, 50], [70, 50]]
  - id: test-unpublished
    type: hydraulic
    taper: CAT40
    gauge_length_mm: 80
    nose_diameter_mm: null
    profile_mm: null
"""

COOLANT = """\
format: cnctwin-coolant
version: 1
id: test-emulsion
name: "Emulsion 8 %"
kind: emulsion
src: src-a
concentration_pct: 8
density_kg_m3: {table: [[20, 998], [60, 985]]}
viscosity_mPa_s: {table: [[20, 1.3], [60, 0.7]]}
specific_heat_J_kgK: {table: [[20, 4050]]}
conductivity_W_mK: {table: [[20, 0.58]]}
"""

WORKHOLDING = """\
format: cnctwin-workholding
version: 1
series_id: test-vises
maker: "Test Vises"
series: "Vises"
src: src-a
jaw_friction: {value: 0.2, src: est, note: "steel on steel, dry"}
items:
  - id: test-vise
    type: vise
    jaw_width_mm: 150
    max_opening_mm: 200
    clamping_force_kN: {value: 28.5, at_handle_torque_Nm: 108}
    body_mm: [[-80, -200, -100], [80, 250, 0]]
  - id: test-vac
    type: vacuum
    pressure_kPa_abs_min: 15
"""

CONTROLLER = """\
format: cnctwin-controller
version: 1
id: test-ctrl
dialect: haas
name: "Test NGC"
src: src-b
alarms:
  - {code: "312", text: "Program End"}
timing:
  ipo_period_ms: 1.0
  lookahead_blocks: 80
words:
  G: {"00": "rapid"}
"""

MODEL = """\
format: cnctwin-model-params
version: 1
model: test-model
src: src-a
params:
  gap_m: {value: 0.5, unit: mm}
  angle_rad: {value: 90, unit: deg}
  ratio: {value: 2.0, unit: "-", range: [1.0, 10.0]}
  label: {value: "steel", unit: "-"}
  table_Pa: {value: [1.0, 2.0], unit: MPa, src: est, note: "two levels assumed"}
"""

FILES = {
    "bibliography/test.yaml": BIBLIOGRAPHY,
    "bibliography/test2.yaml": BIBLIOGRAPHY_2,
    "materials/test-al.yaml": MATERIAL,
    "machines/test-vmc.yaml": MACHINE,
    "machines/test-vmc-siemens.yaml": MACHINE_EXT,
    "tool_materials/test-carbide.yaml": TOOL_MATERIAL,
    "coatings/test-tialn.yaml": COATINGS["test-tialn"],
    "tools/test-endmills.yaml": TOOLS,
    "holders/test-holders.yaml": HOLDERS,
    "coolants/test-emulsion.yaml": COOLANT,
    "workholding/test-vises.yaml": WORKHOLDING,
    "controllers/test-ctrl.yaml": CONTROLLER,
    "models/test-model.yaml": MODEL,
}


def write_db(root: Path, extra: dict[str, str] | None = None, drop: tuple[str, ...] = ()) -> Path:
    """Write the fixture database under ``root/db`` (plus ``extra`` files); returns the db root."""
    db = Path(root) / "db"
    files = {**FILES, **(extra or {})}
    for rel, text in files.items():
        if rel in drop:
            continue
        path = db / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    template = Path("/opt/plint/apps/cnc-twin/db/_templates/project.yaml")
    (db / "_templates").mkdir(parents=True, exist_ok=True)
    (db / "_templates" / "project.yaml").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return db
