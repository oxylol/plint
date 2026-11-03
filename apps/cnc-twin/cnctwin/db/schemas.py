"""Declared schemas of every database file type (db/_templates/*.yaml; docs/formats.md "Databases").

Every field declares its unit (the last token of its key, or ``unit=`` where the key cannot end
with it); :class:`cnctwin.db.simap.SIMap` converts by that unit. Ranges are SI and deliberately
wide: they catch unit mistakes (a factor 1000) and impossible values, not unusual-but-real ones.
Rules that span fields (per tool type required geometry, min <= max, chains that use every
axis ...) are the ``checks`` of each Struct.

``null`` on an optional field means "not published"; a consumer that needs the value says so
(``twin check`` reports objects a project uses whose simulation-required values are null).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .schema import (ID_PATTERN, Any_, ByType, Count, Ctx, Flag, MapOf, Node, OneKey, OxideRows, Pairs, Pose, Qty, Raw,
                     Ref, Seq, Struct, Tagged, Text, get, has, is_number, si_number, unwrap)
from .simap import SIMap, Sourced

DEG = math.pi / 180.0
TWO_PI = 2.0 * math.pi
RAD_S_MAX = 200000 * TWO_PI / 60.0   # 200 000 rpm
SPEED_MAX = 300.0                     # m/s (cutting speeds up to 18 000 m/min)
T_MAX = 5000.0                        # K

FAMILIES = ("aluminium", "steel", "tool_steel", "stainless", "titanium", "nickel", "copper", "brass", "bronze",
            "magnesium", "cast_iron", "zinc", "cobalt", "polymer", "wood", "foam", "composite")
METAL_FAMILIES = tuple(f for f in FAMILIES if f not in ("polymer", "wood", "foam", "composite"))
ISO_GROUPS = ("P", "M", "K", "N", "S", "H", "O")
TOOL_CLASSES = ("hss", "carbide", "cermet", "ceramic", "cbn", "pcd")
STOCK_FORMS = ("rolled_plate", "sheet", "extruded_bar", "cold_drawn_bar", "hot_rolled_bar", "forged", "cast",
               "tube", "moulded", "laminate", "sintered", "additive", "board", "block", "sawn")
TOOL_TYPES = ("square_endmill", "ball_endmill", "toroidal_endmill", "roughing_endmill", "serrated_endmill",
              "face_mill", "shoulder_mill", "chamfer_mill", "engraver", "keyseat", "tslot", "dovetail", "lollipop",
              "slitting_saw", "spot_drill", "centre_drill", "twist_drill", "indexable_drill", "gun_drill", "reamer",
              "tap_cutting", "tap_forming", "thread_mill", "boring_head", "boring_bar",
              "turning_insert", "grooving_insert", "parting_insert", "threading_insert", "form_tool")
HOLDER_TYPES = ("er_collet", "hydraulic", "shrink_fit", "side_lock", "milling_chuck", "modular", "arbor",
                "boring_head", "tapping_floating", "tapping_synchro", "turret_vdi", "turret_bmt", "lathe_holder")
WORKHOLDING_TYPES = ("vise", "clamp", "fixture_plate", "soft_jaws", "vacuum", "magnetic", "adhesive", "chuck_3jaw",
                     "chuck_4jaw", "collet_chuck", "collet", "tailstock_centre", "steady_rest", "tabs")
MACHINE_TYPES = ("vertical_mill", "horizontal_mill", "five_axis_mill", "lathe", "mill_turn", "swiss")
DIALECTS = ("fanuc", "haas", "siemens840d", "heidenhain_klartext", "heidenhain_iso")
COOLANT_KINDS = ("straight_oil", "emulsion", "semi_synthetic", "synthetic", "mql_oil", "air", "water")
JET_KINDS = ("flood", "mist", "mql", "air", "through_spindle", "high_pressure", "through_tool")
VALIDATION_QUANTITIES = ("cutting_force", "torque", "tool_temperature", "chip_temperature", "taylor_life",
                         "stability_lobe", "drilling_thrust", "drilling_torque", "roughness", "wall_error",
                         "spindle_growth", "chip_form", "coolant_reach", "cycle_time", "shear_angle", "chip_ratio")
VALIDATION_OPERATIONS = ("peripheral_milling", "slotting", "face_milling", "ball_end_milling", "orthogonal_turning",
                         "oblique_turning", "longitudinal_turning", "drilling", "tapping", "boring", "grooving",
                         "threading", "machine_motion", "spindle_warmup", "coolant_jet")
BIB_TYPES = ("article", "book", "chapter", "conference", "handbook", "catalogue", "standard", "manual", "datasheet",
             "web", "thesis", "report", "patent", "software", "dataset")
BIB_DATED = ("article", "book", "chapter", "conference", "handbook", "standard", "thesis", "report", "patent")
# Usui wear constants dW/dt = A sigma_n V_s exp(-B/T): the rate-unit systems a file may state for
# A_per_MPa / A_per_Pa, and the factor that turns A into SI (dW/dt in m/s, sigma_n in Pa, V_s in m/s).
# (1e-3/60 m/s per mm/min) / (1e6 Pa per MPa * 1/60 m/s per m/min) = 1e-9;
# (1e-3 m/s per mm/s) / (1e6 Pa per MPa * 1e-3 m/s per mm/s) = 1e-6.
USUI_RATE_UNITS = {
    "mm/min with sigma_n in MPa, V_s in m/min": 1e-9,
    "mm/s with sigma_n in MPa, V_s in mm/s": 1e-6,
    "m/s with sigma_n in Pa, V_s in m/s": 1.0,
}

FORMATS = {
    "material": "cnctwin-material",
    "tool_series": "cnctwin-tool",
    "tool_material": "cnctwin-tool-material",
    "coating": "cnctwin-coating",
    "holder_series": "cnctwin-holder",
    "machine": "cnctwin-machine",
    "coolant": "cnctwin-coolant",
    "workholding_series": "cnctwin-workholding",
    "controller": "cnctwin-controller",
    "validation_case": "cnctwin-validation-case",
    "model": "cnctwin-model-params",
    "bibliography": "cnctwin-bibliography",
}
KIND_OF_FORMAT = {v: k for k, v in FORMATS.items()}
CURRENT_VERSION = {kind: 1 for kind in FORMATS}
# Folder under db/ for each kind (validation cases live in validation/cases/<quantity>/).
FOLDERS = {
    "material": "materials", "tool_series": "tools", "tool_material": "tool_materials", "coating": "coatings",
    "holder_series": "holders", "machine": "machines", "coolant": "coolants",
    "workholding_series": "workholding", "controller": "controllers", "model": "models",
    "bibliography": "bibliography",
}
KIND_OF_FOLDER = {v: k for k, v in FOLDERS.items()}
SINGLE_KINDS = ("material", "tool_material", "coating", "machine", "coolant", "controller")
SERIES_KINDS = {"tool_series": "tool", "holder_series": "holder", "workholding_series": "workholding"}
ITEM_KINDS = {v: k for k, v in SERIES_KINDS.items()}


# ----------------------------------------------------------------------------- small helpers
def angle(lo_deg=-360.0, hi_deg=360.0, **kw) -> Qty:
    return Qty(lo=lo_deg * DEG, hi=hi_deg * DEG, **kw)


def temp(**kw) -> Qty:
    return Qty(gt=0.0, hi=T_MAX, **kw)


def length(hi=100.0, *, positive=True, **kw) -> Qty:
    return Qty(gt=0.0, hi=hi, **kw) if positive else Qty(lo=0.0, hi=hi, **kw)


def fraction(**kw) -> Qty:
    return Qty(lo=0.0, hi=1.0, **kw)


def vec3(lo=-1e3, hi=1e3, **kw) -> Qty:
    return Qty(lo=lo, hi=hi, shape=3, **kw)


def text(**kw) -> Text:
    return Text(nullable=True, **kw)


def _require(mapping, keys, path, ctx: Ctx, why: str) -> None:
    for key in keys:
        node = mapping
        found = True
        for part in key.split("."):
            if not isinstance(node, Mapping) or unwrap(node.get(part)) is None:
                found = False
                break
            node = unwrap(node[part])
        if not found:
            ctx.add(path, f"missing required key {key!r} ({why})", "required")


def _order(mapping, keys, path, ctx, unit=None, what="") -> None:
    """keys in non-decreasing order (SI values)."""
    values = [(k, si_number(mapping, k, unit)) for k in keys]
    values = [(k, v) for k, v in values if is_number(v)]
    for (k0, v0), (k1, v1) in zip(values, values[1:]):
        if v1 < v0:
            ctx.add(path, f"{k0} must not exceed {k1}{what}", "rule")


def min_start_max(node: Qty) -> Struct:
    return Struct({"min": node, "start": node, "max": node},
                  checks=(lambda m, p, u, c: _order(m, ("min", "start", "max"), p, c, u),))


def number_or_struct(node: Qty) -> ByType:
    return ByType(number=node, mapping=min_start_max(node), expected="a number or {min, start, max}")


def header(kind: str, *, id_key="id") -> dict:
    fmt = FORMATS[kind]
    fields = {
        "format": Text((fmt,), sourced=False),
        "version": Count(lo=0, hi=CURRENT_VERSION[kind]),
    }
    if id_key:
        fields[id_key] = Text(pattern=ID_PATTERN, sourced=False)
    return fields


EXTENDS_HINTS = {"same_mechanics_as": "use extends: <id> and give only the keys that differ",
                 "same_as": "use extends: <id> and give only the keys that differ"}
BALANCE = ByType(text=Text(), number=Qty(gt=0, hi=4000, dimensionless=True), nullable=True)
POSE = Pose()

# ----------------------------------------------------------------------------- material
HARDNESS = Struct({
    "HB": Qty(gt=0, hi=1000, dimensionless=True), "HV": Qty(gt=0, hi=3500, dimensionless=True),
    "HRC": Qty(lo=0, hi=75, dimensionless=True), "HRB": Qty(lo=0, hi=130, dimensionless=True),
    "HRA": Qty(lo=0, hi=95, dimensionless=True), "shore_A": Qty(lo=0, hi=100, dimensionless=True),
    "shore_D": Qty(lo=0, hi=100, dimensionless=True), "janka_N": Qty(gt=0, hi=5e4),
    "range": Qty(shape="range", lo=0, dimensionless=True),
}, checks=(lambda m, p, u, c: None if any(has(m, k) for k in ("HB", "HV", "HRC", "HRB", "HRA", "shore_A",
                                                               "shore_D", "janka_N"))
           else c.add(p, "give at least one hardness scale (HB, HV, HRC, HRB, HRA, shore_A, shore_D, janka_N)",
                      "required"),))


def _material_physical_rules(m, p, u, ctx):
    _order(m, ("solidus_degC", "liquidus_degC"), p, ctx)
    _order(m, ("yield_MPa", "ultimate_MPa"), p, ctx, what=" (yield above ultimate)")


PHYSICAL = Struct({
    "density_kg_m3": Qty(lo=1.0, hi=25000.0, table=True),
    "specific_heat_J_kgK": Qty(lo=50.0, hi=10000.0, table=True),
    "conductivity_W_mK": Qty(lo=1e-3, hi=3000.0, table=True),
    "expansion_1_K": Qty(lo=-5e-5, hi=5e-4, table=True),
    "youngs_modulus_GPa": Qty(gt=1e4, hi=1.2e12, table=True),
    "shear_modulus_GPa": Qty(gt=1e4, hi=1.0e12, table=True, nullable=True),
    "poisson": Qty(lo=-1.0, hi=0.5, table=True),
    "solidus_degC": temp(nullable=True),
    "liquidus_degC": temp(nullable=True),
    "glass_transition_degC": temp(nullable=True),
    "melting_degC": temp(nullable=True),
    "decomposition_degC": temp(nullable=True),
    "hardness": HARDNESS,
    "yield_MPa": Qty(gt=0, hi=1e10, table=True),
    "ultimate_MPa": Qty(gt=0, hi=1e10, table=True),
    "elongation_pct": Qty(lo=0, hi=20, nullable=True),
    "reduction_of_area_pct": Qty(lo=0, hi=1, nullable=True),
    "fracture_strain": Qty(lo=0, hi=20, nullable=True),
    "fracture_toughness_MPa_sqrt_m": Qty(gt=0, hi=1e9, nullable=True),
    "impact_energy_J": Qty(gt=0, hi=1e4, nullable=True),
    "emissivity": Qty(lo=0, hi=1, table=True, nullable=True),
    "moisture_content_pct": Qty(lo=0, hi=2, nullable=True),
    "compressive_MPa": Qty(gt=0, hi=1e10, table=True, nullable=True),
    "flexural_MPa": Qty(gt=0, hi=1e10, nullable=True),
}, required=("density_kg_m3", "specific_heat_J_kgK", "conductivity_W_mK", "expansion_1_K", "youngs_modulus_GPa",
             "poisson", "hardness", "yield_MPa", "ultimate_MPa"), checks=(_material_physical_rules,))

JOHNSON_COOK = Struct({
    "model": Text(("johnson_cook",), sourced=False),
    "A_MPa": Qty(gt=0, hi=1e10), "B_MPa": Qty(lo=0, hi=1e11), "n": Qty(lo=0, hi=3), "C": Qty(lo=-0.5, hi=2),
    "m": Qty(gt=0, hi=10), "ref_strain_rate_1_s": Qty(gt=0, hi=1e8), "T_ref_degC": temp(), "T_melt_degC": temp(),
    "damage": Struct({"D1": Qty(lo=-10, hi=10), "D2": Qty(lo=-10, hi=10), "D3": Qty(lo=-20, hi=20),
                      "D4": Qty(lo=-10, hi=10), "D5": Qty(lo=-20, hi=20)}, nullable=True),
}, required=("model", "A_MPa", "B_MPa", "n", "C", "m", "ref_strain_rate_1_s", "T_ref_degC", "T_melt_degC"),
    checks=(lambda m, p, u, c: _order(m, ("T_ref_degC", "T_melt_degC"), p, c),))
TEMPERATURE_TABLE_FLOW = Struct({
    "model": Text(("temperature_table",), sourced=False),
    "tensile_strength_MPa": Qty(gt=0, hi=1e10, table=True), "compressive_yield_MPa": Qty(gt=0, hi=1e10, table=True),
    "T_soft_degC": temp(nullable=True),
}, required=("model", "tensile_strength_MPa"))
NO_FLOW = Struct({"model": Text(("none",), sourced=False),
                  "yield_MPa": Qty(gt=0, hi=1e10, table=True), "compressive_flow_MPa": Qty(gt=0, hi=1e10, table=True),
                  "shear_yield_MPa": Qty(gt=0, hi=1e10, table=True), "T_soft_degC": temp(nullable=True),
                  "T_melt_degC": temp(nullable=True)}, required=("model",),
                 doc="no constitutive (strain/rate) model: polymers, foams, wood, composites; forces come from "
                     "cutting.kienzle, the optional strengths and softening temperatures bound the chip model")
FLOW_STRESS = Tagged("model", {"johnson_cook": JOHNSON_COOK, "temperature_table": TEMPERATURE_TABLE_FLOW,
                               "none": NO_FLOW}, nullable=True)

PROCESSES = ("milling", "turning", "drilling", "boring", "tapping", "threading", "grooving", "orthogonal", "any")
CALIBRATION = Struct({
    "vc_m_min": Qty(shape=(None, "range"), gt=0, hi=SPEED_MAX), "h_mm": Qty(shape=(None, "range"), gt=0, hi=0.05),
    "f_mm": Qty(shape=(None, "range"), gt=0, hi=0.05), "ap_mm": Qty(shape=(None, "range"), gt=0, hi=0.5),
    "b_mm": Qty(shape=(None, "range"), gt=0, hi=0.5),
    "edge_radius_um": Qty(shape=(None, "range"), lo=0, hi=1e-3), "rake_deg": angle(-60, 60, shape=(None, "range")),
    "coolant": Text(), "work_temp_degC": temp(), "tool": Text(), "method": Text(),
}, nullable=True)
KIENZLE = Struct({
    "tool_class": Text(TOOL_CLASSES), "coating": Text(), "process": Text(PROCESSES),
    "kc11_N_mm2": Qty(gt=1e5, hi=2e10), "mc": Qty(lo=0, hi=1),
    "kf11_N_mm2": Qty(gt=0, hi=2e10, nullable=True), "mf": Qty(lo=-1, hi=1.5, nullable=True),
    "kp11_N_mm2": Qty(gt=0, hi=2e10, nullable=True), "mp": Qty(lo=-1, hi=1.5, nullable=True),
    "h_ref_mm": length(0.01), "b_ref_mm": length(0.01),
    "rake_ref_deg": angle(-45, 45, nullable=True), "vc_ref_m_min": Qty(gt=0, hi=SPEED_MAX, nullable=True),
    "calibration": CALIBRATION,
}, required=("tool_class", "kc11_N_mm2", "mc", "h_ref_mm", "b_ref_mm"),
    checks=(lambda m, p, u, c: _require(m, ("mf",), p, c, "kf11_N_mm2 is given") if has(m, "kf11_N_mm2") else None,
            lambda m, p, u, c: _require(m, ("mp",), p, c, "kp11_N_mm2 is given") if has(m, "kp11_N_mm2") else None))
MECHANISTIC = Struct({
    "tool_class": Text(TOOL_CLASSES), "coating": Text(), "process": Text(PROCESSES),
    "helix_deg": angle(-80, 80, nullable=True), "rake_deg": angle(-60, 60, nullable=True),
    "diameter_mm": length(3.0), "flutes": Count(lo=1),
    "Ktc_N_mm2": Qty(gt=0, hi=2e10), "Krc_N_mm2": Qty(lo=-2e10, hi=2e10), "Kac_N_mm2": Qty(lo=-2e10, hi=2e10),
    "Kte_N_mm": Qty(lo=-1e7, hi=1e7), "Kre_N_mm": Qty(lo=-1e7, hi=1e7), "Kae_N_mm": Qty(lo=-1e7, hi=1e7),
    "vc_m_min": Qty(shape=(None, "range"), gt=0, hi=SPEED_MAX), "fz_mm": Qty(shape=(None, "range"), gt=0, hi=0.05),
    "calibration": CALIBRATION,
}, required=("tool_class", "Ktc_N_mm2", "Krc_N_mm2", "Kac_N_mm2", "Kte_N_mm", "Kre_N_mm", "Kae_N_mm"))


def _cutting_rules(m, p, u, ctx):
    if not (get(m, "kienzle") or []) and not (get(m, "mechanistic") or []):
        ctx.add(p, "give at least one kienzle or mechanistic entry (the force model needs one)", "required")


CUTTING = Struct({
    "kienzle": Seq(KIENZLE),
    "mechanistic": Seq(MECHANISTIC, nullable=True),
    "min_chip_thickness_ratio": Qty(lo=0, hi=3),
    "indentation_coeff_N_mm3": Qty(lo=0, hi=1e17),
    "work_hardening": Struct({"layer_depth_um": length(0.01, positive=False),
                              "hardness_increase_pct": Qty(lo=0, hi=5),
                              "strain_hardening_exponent": Qty(lo=0, hi=2)}, nullable=True),
}, checks=(_cutting_rules,))

FRICTION = Seq(Struct({
    "tool_class": Text(TOOL_CLASSES), "coating": Text(),
    "mu_sliding": Qty(lo=0, hi=3), "sticking_ratio": fraction(nullable=True),
}, required=("tool_class", "mu_sliding")), min_len=1)

CHIP = Struct({
    "tendency": Text(("continuous", "segmented", "discontinuous", "powder", "molten")),
    "segmentation": Struct({"criterion": Text(("critical_speed", "thermal_instability")),
                            "vc_onset_m_min": Qty(gt=0, hi=SPEED_MAX, nullable=True),
                            "temp_onset_degC": temp(nullable=True),
                            "h_onset_mm": length(0.01, nullable=True)}, required=("criterion",), nullable=True),
    "bue": Struct({"interface_temp_degC": Qty(shape="range", lo=0.0, hi=2000.0 + 273.15, nullable=True),
                   "vc_m_min": Qty(shape="range", lo=0, hi=SPEED_MAX)},
                  required=("vc_m_min",), nullable=True),
    "breakability": Text(("poor", "medium", "good")),
    "oxide_colours": OxideRows(),
}, required=("tendency",))

TOOL_LIFE = Seq(Struct({
    "tool_class": Text(TOOL_CLASSES), "coating": Text(), "grade": Text(), "process": Text(PROCESSES),
    "C_m_min": Qty(gt=0, hi=1e3), "n": Qty(gt=0, hi=2), "a": Qty(lo=-1, hi=3), "b": Qty(lo=-1, hi=3),
    "T_ref_min": Qty(gt=0, hi=1e5), "feed_ref_mm": length(0.1), "ap_ref_mm": length(1.0),
    "VB_limit_mm": length(0.01), "vc_range_m_min": Qty(shape="range", lo=0, hi=SPEED_MAX),
}, required=("tool_class", "C_m_min", "n", "T_ref_min", "VB_limit_mm"),
    checks=(lambda m, p, u, c: _require(m, ("feed_ref_mm",), p, c, "a is given") if has(m, "a") else None,
            lambda m, p, u, c: _require(m, ("ap_ref_mm",), p, c, "b is given") if has(m, "b") else None),
    hints={"usui": "Usui constants go in the material's top-level tool_life_usui list"}), nullable=True)


def _usui_rules(m, p, u, ctx):
    given = [k for k in ("A_per_MPa", "A_per_Pa") if k in m]
    if len(given) != 1:
        ctx.add(p, "give the Usui constant as exactly one of A_per_MPa or A_per_Pa", "required")
        return
    rate = get(m, "rate_unit")
    if rate is None:
        ctx.add(p, f"{given[0]} needs rate_unit (one of: {'; '.join(USUI_RATE_UNITS)})", "required")
    elif given[0] == "A_per_Pa" and USUI_RATE_UNITS.get(rate) not in (None, 1.0):
        ctx.add(p, "A_per_Pa means SI rates: rate_unit must be 'm/s with sigma_n in Pa, V_s in m/s'", "rule")


class UsuiStruct(Struct):
    """Usui constants: the SI view has ``A_per_Pa`` in SI rates (dW/dt in m/s with sigma_n in Pa and V_s
    in m/s) whatever unit system the file states in ``rate_unit``; ``A_per_MPa`` stays as written."""

    def build(self, value, unit, src, si_key):
        built = super().build(value, unit, src, si_key)
        if not isinstance(value, Mapping) or "A_per_MPa" not in value:
            return built
        factor = USUI_RATE_UNITS.get(get(value, "rate_unit"))
        raw = built["A_per_MPa"]
        if factor is None or raw is None:
            return built
        entries = dict(built._data)
        if isinstance(raw, Sourced):
            entries["A_per_Pa"] = Sourced(float(raw) * factor, raw.raw_value, raw.src, raw.note, dict(raw.extra),
                                          "A_per_Pa")
        else:
            entries["A_per_Pa"] = float(raw) * factor
        return SIMap.from_entries(value, entries, built.src)


USUI = UsuiStruct({
    "work_family": Text(FAMILIES + ("any",)), "tool_class": Text(TOOL_CLASSES), "coating": Text(),
    "surface": Text(("rake", "flank", "both")),
    "A_per_MPa": Qty(gt=0, hi=1.0, dimensionless=True), "A_per_Pa": Qty(gt=0, hi=1.0, dimensionless=True),
    "rate_unit": Text(tuple(USUI_RATE_UNITS), sourced=False),
    "B_K": Qty(gt=0, hi=1e6), "T_range_K": Qty(shape="range", lo=0, hi=T_MAX, nullable=True, open_range=True),
}, required=("B_K",), checks=(_usui_rules,))

PROFILE = Pairs(None, x=(0.0, 1.0), y=(-1e10, 1e10))
RESIDUAL_STRESS = Seq(Struct({
    "form": Text(STOCK_FORMS), "thickness_mm": length(10.0),
    "direction": Text(("L", "LT", "ST", "hoop", "radial", "axial")),
    "profile_MPa": PROFILE,
}, required=("form", "thickness_mm", "direction", "profile_MPa")), nullable=True)


def _skin_rule(m, p, u, ctx):
    if get(m, "type") not in (None, "none"):
        _require(m, ("thickness_mm", "hardness_factor", "abrasiveness_factor"), p, ctx, "skin type is not none")


SKIN = Struct({
    "type": Text(("none", "mill_scale", "cast_skin", "sand_inclusions", "decarburised")),
    "thickness_mm": length(0.1, positive=False), "hardness_factor": Qty(gt=0, hi=10),
    "abrasiveness_factor": Qty(gt=0, hi=100),
}, required=("type",), checks=(_skin_rule,), nullable=True)

THERMAL_DAMAGE = Struct({"burn_degC": temp(nullable=True), "white_layer_degC": temp(nullable=True),
                         "melt_smear_degC": temp(nullable=True), "overtemper_degC": temp(nullable=True),
                         "char_degC": temp(nullable=True)}, nullable=True)
IGNITION = Struct({"chip_ignition_degC": temp(nullable=True), "fines_ignition_degC": temp(nullable=True),
                   "bulk_ignition_degC": temp(nullable=True), "powder_ignition_degC": temp(nullable=True),
                   "chip_hazard_degC": temp(nullable=True), "autoignition_degC": temp(nullable=True),
                   "min_ignition_energy_mJ": Qty(gt=0, hi=1e6, nullable=True),
                   "critical_chip_thickness_mm": length(0.01, nullable=True), "dust_explosible": Flag(nullable=True)},
                  nullable=True)
COMPOSITE = Struct({
    "fibre": Text(("carbon", "glass", "aramid", "natural", "basalt", "wood")), "matrix": Text(),
    "fibre_volume_fraction": fraction(), "layup": Text(),
    "delamination": Struct({"critical_thrust_N": Qty(gt=0, hi=1e6), "model": Text(("hocheng-dharan",)),
                            "G_IC_J_m2": Qty(gt=0, hi=1e6)}, nullable=True),
    "dust_mass_fraction": fraction(), "abrasive_wear_factor": Qty(gt=0, hi=1000),
}, required=("fibre",), nullable=True)


def _material_rules(m, p, u, ctx):
    family = get(m, "family")
    flow = get(m, "flow_stress")
    model = get(flow, "model") if isinstance(flow, Mapping) else None
    if family in METAL_FAMILIES and model != "johnson_cook":
        ctx.add(p, f"a {family} needs flow_stress: {{model: johnson_cook, ...}} (the chip model needs it)",
                "required")


MATERIAL = Struct({
    **header("material"), "extends": Ref("material"), "name": Text(sourced=False),
    "family": Text(FAMILIES), "iso_group": Text(ISO_GROUPS), "condition": Text(),
    "designations": MapOf(Text()), "stock_forms": Seq(Text(STOCK_FORMS)),
    "physical": PHYSICAL, "flow_stress": FLOW_STRESS, "cutting": CUTTING, "friction": FRICTION, "chip": CHIP,
    "tool_life": TOOL_LIFE, "tool_life_usui": Seq(USUI, nullable=True), "residual_stress": RESIDUAL_STRESS,
    "skin": SKIN, "thermal_damage": THERMAL_DAMAGE, "ignition": IGNITION, "composite": COMPOSITE,
    "extensions": Raw(doc="data outside the material template (non-metal property sets, process windows): "
                          "kept as written, read by no model; every number still needs a source"),
}, required=("format", "version", "id", "name", "family", "iso_group", "physical", "cutting", "friction", "chip"),
    checks=(_material_rules,), hints=EXTENDS_HINTS)

# ----------------------------------------------------------------------------- tools
CLEARANCE = ByType(number=angle(0, 60), mapping=Struct({"primary": angle(0, 60), "secondary": angle(0, 80),
                                                        "tertiary": angle(0, 80)}, required=("primary",)),
                   expected="an angle or {primary, secondary}")


def _edge_prep_rule(m, p, u, ctx):
    kind = get(m, "type")
    if kind == "chamfer":
        _require(m, ("width_um", "angle_deg"), p, ctx, "chamfer edge preparation")
    if kind == "waterfall":
        _require(m, ("ratio",), p, ctx, "waterfall hone (ratio of flank to rake hone length)")


EDGE_PREP = Struct({"type": Text(("hone", "chamfer", "sharp", "waterfall")), "width_um": length(0.005, positive=False),
                    "angle_deg": angle(0, 60), "ratio": Qty(gt=0, hi=10)}, required=("type",),
                   checks=(_edge_prep_rule,))

INSERT_SHAPES = ("A", "B", "C", "D", "E", "H", "K", "L", "M", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X")
GEOMETRY_FIELDS = {
    "diameter_mm": length(3.0), "flutes": Count(lo=1, hi=500),
    "pitch_deg": Qty(shape="list", gt=0, lt=TWO_PI),
    "helix_deg": Qty(shape=(None, "list"), lo=-80 * DEG, hi=80 * DEG),
    "rake_deg": angle(-60, 60), "clearance_deg": CLEARANCE, "edge_radius_um": length(1e-3, positive=False),
    "edge_prep": EDGE_PREP, "corner_radius_mm": length(1.5, positive=False),
    "corner_chamfer_mm": length(0.1, positive=False),
    "cutting_length_mm": length(5.0), "flute_length_mm": length(5.0),
    "neck": Struct({"diameter_mm": length(3.0), "length_mm": length(5.0)}, required=("diameter_mm", "length_mm"),
                   nullable=True),
    "shank": Struct({"diameter_mm": length(1.0), "length_mm": length(5.0),
                     "form": Text(("plain", "weldon", "whistle_notch", "threaded", "hsk", "capto", "bmt", "vdi",
                                   "modular", "arbor"))}, required=("diameter_mm",), nullable=True),
    "overall_length_mm": length(10.0), "gullet_area_mm2": Qty(lo=0, hi=1e-2), "core_diameter_mm": length(3.0),
    "centre_cutting": Flag(),
    "coolant_holes": Struct({"count": Count(lo=1, hi=100), "diameter_mm": length(0.05),
                             "exit": Text(("tip", "flute", "periphery", "face"))},
                            required=("count", "diameter_mm"), nullable=True),
    "point_angle_deg": angle(30, 180), "chisel_edge_angle_deg": angle(0, 180),
    "chisel_length_mm": length(0.1, positive=False), "web_thickness_mm": length(0.5, positive=False),
    "margin_width_mm": length(0.1, positive=False), "lip_count": Count(lo=1, hi=20),
    "thread": Struct({"standard": Text(), "pitch_mm": length(0.05), "form_angle_deg": angle(1, 179),
                      "class": Text(), "diameter_mm": length(3.0), "hand": Text(("right", "left")),
                      "starts": Count(lo=1, hi=10)}, required=("pitch_mm",), nullable=True),
    "chamfer_threads": Qty(gt=0, hi=20), "tap_kind": Text(("cutting", "forming")),
    "spiral": Text(("point", "flute", "straight")), "rows": Count(lo=1, hi=1000), "single_profile": Flag(),
    "tip_angle_deg": angle(1, 179), "tip_diameter_mm": length(3.0, positive=False),
    "tip_width_mm": length(0.1, positive=False), "width_mm": length(1.0),
    "hub": Struct({"diameter_mm": length(3.0), "width_mm": length(1.0)}, nullable=True),
    "inserts": Struct({
        "count": Count(lo=1, hi=500), "iso_code": Text(), "ic_mm": length(0.2), "nose_radius_mm": length(0.1, positive=False),
        "axial_rake_deg": angle(-60, 60), "radial_rake_deg": angle(-60, 60), "lead_angle_deg": angle(1, 179),
        "clearance_deg": angle(0, 60), "edge_length_mm": length(0.2), "wiper": Flag(),
        "positions_deg": Qty(shape="list", lo=-TWO_PI, hi=TWO_PI)}, required=("count", "lead_angle_deg"),
        nullable=True),
    "insert": Struct({
        "iso_code": Text(), "shape": Text(INSERT_SHAPES), "included_angle_deg": angle(1, 180), "ic_mm": length(0.2),
        "thickness_mm": length(0.05), "nose_radius_mm": length(0.05, positive=False), "clearance_deg": angle(0, 45),
        "edge_length_mm": length(0.2), "wiper": Flag(), "width_mm": length(0.05), "max_depth_mm": length(0.5),
        "chip_breaker": Struct({"name": Text(), "land_mm": length(0.01, positive=False),
                                "groove_width_mm": length(0.02, positive=False),
                                "groove_depth_mm": length(0.01, positive=False), "rake_deg": angle(-60, 60),
                                "window": Struct({"f_mm": Qty(shape="range", lo=0, hi=0.01),
                                                  "ap_mm": Qty(shape="range", lo=0, hi=0.1)})}, nullable=True),
    }, required=("nose_radius_mm",), nullable=True),
    "holder": Struct({"iso_code": Text(), "approach_angle_deg": angle(1, 179), "inclination_deg": angle(-60, 60),
                      "rake_deg": angle(-60, 60), "shank_mm": Qty(shape=2, gt=0, hi=0.2), "overhang_mm": length(1.0),
                      "max_depth_mm": length(1.0)}, nullable=True),
    "bar": Struct({"diameter_mm": length(0.5), "overhang_mm": length(3.0),
                   "material": Text(("steel", "carbide", "damped", "heavy_metal"))},
                  required=("diameter_mm", "overhang_mm", "material"), nullable=True),
    "min_bore_mm": length(3.0), "diameter_range_mm": Qty(shape="range", gt=0, hi=3.0),
    "profile_mm": Qty(shape=("rows", 2), lo=-10.0, hi=10.0),
}

ENDMILLS = ("square_endmill", "ball_endmill", "toroidal_endmill", "roughing_endmill", "serrated_endmill")
INSERTS = ("turning_insert", "grooving_insert", "parting_insert", "threading_insert")
_ENDMILL_GEOMETRY = ("diameter_mm", "flutes", "helix_deg", "rake_deg", "cutting_length_mm", "overall_length_mm",
                     "shank.diameter_mm")
TYPE_GEOMETRY = {
    **{t: _ENDMILL_GEOMETRY for t in ENDMILLS},
    "toroidal_endmill": _ENDMILL_GEOMETRY + ("corner_radius_mm",),
    "ball_endmill": _ENDMILL_GEOMETRY + ("corner_radius_mm",),
    "face_mill": ("diameter_mm", "inserts.count", "inserts.lead_angle_deg", "cutting_length_mm"),
    "shoulder_mill": ("diameter_mm", "inserts.count", "inserts.lead_angle_deg", "cutting_length_mm"),
    "chamfer_mill": ("diameter_mm", "flutes", "tip_angle_deg", "overall_length_mm"),
    "engraver": ("diameter_mm", "tip_angle_deg", "overall_length_mm"),
    "keyseat": ("diameter_mm", "flutes", "width_mm", "overall_length_mm"),
    "tslot": ("diameter_mm", "flutes", "width_mm", "overall_length_mm"),
    "dovetail": ("diameter_mm", "flutes", "tip_angle_deg", "width_mm", "overall_length_mm"),
    "lollipop": ("diameter_mm", "flutes", "overall_length_mm"),
    "slitting_saw": ("diameter_mm", "flutes", "width_mm"),
    "spot_drill": ("diameter_mm", "flutes", "tip_angle_deg", "overall_length_mm"),
    "centre_drill": ("diameter_mm", "point_angle_deg", "overall_length_mm"),
    "twist_drill": ("diameter_mm", "flutes", "point_angle_deg", "cutting_length_mm", "overall_length_mm"),
    "indexable_drill": ("diameter_mm", "inserts.count", "cutting_length_mm"),
    "gun_drill": ("diameter_mm", "point_angle_deg", "cutting_length_mm", "overall_length_mm"),
    "reamer": ("diameter_mm", "flutes", "cutting_length_mm", "overall_length_mm"),
    "tap_cutting": ("diameter_mm", "thread.pitch_mm", "cutting_length_mm", "overall_length_mm"),
    "tap_forming": ("diameter_mm", "thread.pitch_mm", "cutting_length_mm", "overall_length_mm"),
    "thread_mill": ("diameter_mm", "flutes", "thread.pitch_mm", "cutting_length_mm", "overall_length_mm"),
    "boring_head": ("diameter_range_mm", "insert.nose_radius_mm"),
    "boring_bar": ("bar.diameter_mm", "bar.overhang_mm", "insert.nose_radius_mm", "min_bore_mm"),
    "turning_insert": ("insert.ic_mm", "insert.nose_radius_mm", "holder.approach_angle_deg", "holder.overhang_mm"),
    "grooving_insert": ("insert.width_mm", "insert.nose_radius_mm", "holder.overhang_mm"),
    "parting_insert": ("insert.width_mm", "insert.nose_radius_mm", "holder.overhang_mm"),
    "threading_insert": ("insert.nose_radius_mm", "thread.pitch_mm", "holder.overhang_mm"),
    "form_tool": ("profile_mm",),
}


def _tool_geometry_rules(item, path, unit, ctx):
    kind = get(item, "type")
    geometry = get(item, "geometry")
    if not isinstance(geometry, Mapping) or kind not in TYPE_GEOMETRY:
        return
    gpath = path + ("geometry",)
    _require(geometry, TYPE_GEOMETRY[kind], gpath, ctx, f"type {kind}")
    insert = geometry.get("insert") if isinstance(geometry.get("insert"), Mapping) else {}
    if kind in INSERTS and not (has(insert, "shape") or has(insert, "included_angle_deg")
                                or kind in ("grooving_insert", "parting_insert")):
        ctx.add(gpath, "a turning insert needs insert.shape or insert.included_angle_deg", "required")
    diameter = si_number(geometry, "diameter_mm")
    flutes = get(geometry, "flutes")
    radius = si_number(geometry, "corner_radius_mm")
    if kind == "ball_endmill" and is_number(diameter) and is_number(radius) \
            and abs(radius - diameter / 2.0) > 0.02 * diameter / 2.0:
        ctx.add(gpath, f"a ball end mill has corner_radius_mm = diameter_mm / 2 (got r = {radius * 1e3:g} mm, "
                       f"D = {diameter * 1e3:g} mm)", "rule")
    if kind == "toroidal_endmill" and is_number(radius) and radius <= 0:
        ctx.add(gpath, "a toroidal end mill needs corner_radius_mm > 0", "rule")
    if is_number(diameter) and is_number(radius) and kind in ENDMILLS and radius > diameter / 2.0 * 1.02:
        ctx.add(gpath, "corner_radius_mm exceeds diameter_mm / 2", "rule")
    pitch = get(geometry, "pitch_deg")
    if isinstance(pitch, list) and pitch and all(is_number(v) for v in pitch):
        if isinstance(flutes, int) and len(pitch) != flutes:
            ctx.add(gpath, f"pitch_deg has {len(pitch)} entries for {flutes} flutes", "rule")
        if abs(sum(pitch) - 360.0) > 0.01:
            ctx.add(gpath, f"pitch_deg must sum to 360 (sums to {sum(pitch):g})", "rule")
    helix = get(geometry, "helix_deg")
    if isinstance(helix, list) and isinstance(flutes, int) and len(helix) != flutes:
        ctx.add(gpath, f"helix_deg has {len(helix)} entries for {flutes} flutes", "rule")
    core = si_number(geometry, "core_diameter_mm")
    if is_number(core) and is_number(diameter) and core >= diameter:
        ctx.add(gpath, "core_diameter_mm must be smaller than diameter_mm", "rule")
    overall = si_number(geometry, "overall_length_mm")
    for key in ("cutting_length_mm", "flute_length_mm"):
        value = si_number(geometry, key)
        if is_number(value) and is_number(overall) and value > overall:
            ctx.add(gpath, f"{key} exceeds overall_length_mm", "rule")
    shank = get(geometry, "shank")
    if isinstance(shank, Mapping):
        shank_length = si_number(shank, "length_mm")
        if is_number(shank_length) and is_number(overall) and shank_length > overall:
            ctx.add(gpath, "shank.length_mm exceeds overall_length_mm", "rule")
    neck = get(geometry, "neck")
    if isinstance(neck, Mapping):
        neck_d = si_number(neck, "diameter_mm")
        if is_number(neck_d) and is_number(diameter) and neck_d > diameter * 1.001:
            ctx.warn(gpath, "neck.diameter_mm is larger than diameter_mm (the neck would cut)")
    inserts = get(geometry, "inserts")
    if isinstance(inserts, Mapping):
        positions = get(inserts, "positions_deg")
        count = get(inserts, "count")
        if isinstance(positions, list) and isinstance(count, int) and len(positions) != count:
            ctx.add(gpath, f"inserts.positions_deg has {len(positions)} entries for {count} inserts", "rule")


def _coating_rule(m, p, u, ctx):
    if get(m, "id") not in (None, "none") and not has(m, "thickness_um"):
        ctx.warn(p, "coated tool without thickness_um (the coating's typical thickness will be used)")


CUTTING_DATA = Struct({
    "iso_group": Text(ISO_GROUPS), "material_hint": Text(), "material": Ref("material"),
    "vc_m_min": number_or_struct(Qty(gt=0, hi=SPEED_MAX)), "fz_mm": number_or_struct(Qty(gt=0, hi=0.05)),
    "f_mm": number_or_struct(Qty(gt=0, hi=0.05)), "ap_max_mm": length(1.0), "ap_max_xD": Qty(gt=0, hi=50),
    "ae_max_xD": Qty(gt=0, hi=1), "ae_max_mm": length(1.0),
    "operation": ByType(text=Text(pattern=r"[a-z_]+"), seq=Seq(Text(pattern=r"[a-z_]+")), expected="text or list"),
    "coolant": Text(),
}, required=("iso_group",))

TOOL_ITEM = Struct({
    "id": Text(pattern=ID_PATTERN, sourced=False), "order_code": text(), "type": Text(TOOL_TYPES),
    "maker": Text(), "series": Text(), "series_id": Text(pattern=ID_PATTERN, sourced=False),
    "hand": Text(("right", "left", "neutral")),
    "tool_material": Struct({"class": Text(TOOL_CLASSES), "grade": Text(), "id": Ref("tool_material")},
                            required=("class", "id")),
    "coating": Struct({"id": Ref("coating", special=("none",)), "thickness_um": length(1e-4, positive=False)},
                      required=("id",), checks=(_coating_rule,)),
    "geometry": Struct(GEOMETRY_FIELDS),
    "max_rpm": Qty(gt=0, hi=RAD_S_MAX, nullable=True),
    "balance_grade": BALANCE,
    "mass_g": Qty(gt=0, hi=1000.0),
    "cutting_data": Seq(CUTTING_DATA),
    "price_eur": Qty(lo=0, nullable=True), "regrinds": Count(lo=0, hi=100, nullable=True),
}, required=("id", "type", "hand", "tool_material", "coating", "geometry"), checks=(_tool_geometry_rules,))

SERIES_COMMON = {"maker": Text(), "series": Text(), "items": Seq(Any_(), min_len=1),
                 "defaults": Any_(doc="item fields every item of the series inherits unless it sets them")}
TOOL_SERIES = Struct({**header("tool_series", id_key="series_id"), **SERIES_COMMON},
                     required=("format", "version", "series_id", "maker", "series", "items"))

# ----------------------------------------------------------------------------- tool materials, coatings
TOOL_MATERIAL = Struct({
    **header("tool_material"), "extends": Ref("tool_material"), "class": Text(TOOL_CLASSES), "name": Text(),
    "composition": text(), "grain_size_um": length(1e-4, nullable=True),
    "binder_wt_pct": Qty(lo=0, hi=1, nullable=True), "binder_vol_pct": Qty(lo=0, hi=1, nullable=True),
    "cbn_content_vol_pct": Qty(lo=0, hi=1, nullable=True),
    "density_kg_m3": Qty(lo=1000.0, hi=25000.0), "youngs_modulus_GPa": Qty(gt=1e9, hi=1.3e12, table=True),
    "poisson": Qty(lo=0, hi=0.5), "hardness_HV": Qty(gt=0, hi=12000, table=True),
    "transverse_rupture_MPa": Qty(gt=0, hi=2e10, nullable=True), "compressive_MPa": Qty(gt=0, hi=2e10, nullable=True),
    "fracture_toughness_MPa_sqrt_m": Qty(gt=0, hi=1e9, nullable=True),
    "conductivity_W_mK": Qty(gt=0, hi=3000.0, table=True), "specific_heat_J_kgK": Qty(gt=0, hi=5000.0, table=True),
    "expansion_1_K": Qty(lo=-1e-5, hi=5e-5, table=True), "max_service_temp_degC": temp(),
    "weibull_modulus": Qty(gt=0, hi=200, nullable=True), "thermal_shock_parameter": Qty(gt=0, nullable=True),
    "thermal_shock_dK": Qty(gt=0, hi=5000, nullable=True),
    "usui": Seq(USUI, nullable=True),
}, required=("format", "version", "id", "class", "name", "density_kg_m3", "youngs_modulus_GPa", "poisson",
             "hardness_HV", "conductivity_W_mK", "specific_heat_J_kgK", "expansion_1_K",
             "max_service_temp_degC"), hints=EXTENDS_HINTS)

COATING = Struct({
    **header("coating"), "extends": Ref("coating"), "name": Text(),
    "process": Text(("PVD", "CVD", "PACVD", "none")), "process_temp_degC": temp(nullable=True),
    "colour": text(), "hardness_HV": Qty(gt=0, hi=12000, nullable=True),
    "hardness_HIT_GPa": Qty(gt=0, hi=2e11, nullable=True), "max_service_temp_degC": temp(nullable=True),
    "conductivity_W_mK": Qty(gt=0, hi=3000.0, table=True, nullable=True),
    "friction_vs_steel": Qty(lo=0, hi=2, nullable=True),
    "typical_thickness_um": Qty(shape="range", lo=0, hi=1e-4), "wear_rate_factor": Qty(gt=0, hi=10),
    "thermal_barrier_factor": Qty(lo=0, hi=1.5),
}, required=("format", "version", "id", "name", "process", "hardness_HV", "max_service_temp_degC",
             "conductivity_W_mK", "wear_rate_factor", "thermal_barrier_factor"), hints=EXTENDS_HINTS)

# ----------------------------------------------------------------------------- holders
def _holder_rules(m, p, u, ctx):
    profile = get(m, "profile_mm")
    if isinstance(profile, list) and all(isinstance(r, list) and len(r) == 2 and all(is_number(v) for v in r)
                                         for r in profile):
        zs = [r[0] for r in profile]
        if any(b < a for a, b in zip(zs, zs[1:])):
            ctx.add(p, "profile_mm rows must have non-decreasing z", "rule")
        if any(r[1] <= 0 for r in profile):
            ctx.add(p, "profile_mm diameters must be positive", "range")


HOLDER_ITEM = Struct({
    "id": Text(pattern=ID_PATTERN, sourced=False), "order_code": text(), "type": Text(HOLDER_TYPES),
    "maker": Text(), "series": Text(), "series_id": Text(pattern=ID_PATTERN, sourced=False),
    "taper": Text(), "taper_tolerance": text(),
    "modular_interface": ByType(text=Text(), boolean=Flag(), nullable=True, expected="text or true/false"),
    "bore_mm": Qty(shape=(None, "range"), gt=0, hi=0.5, nullable=True), "collet": text(), "collet_size": text(),
    "shank_tolerance": text(), "tap_range": text(), "adapter_size": text(), "nut": text(),
    "gauge_length_mm": length(2.0, nullable=True), "nose_diameter_mm": length(1.0, nullable=True),
    "overall_length_mm": length(2.0, nullable=True), "functional_length_mm": length(2.0, nullable=True),
    "nut_height_mm": length(0.5, nullable=True),
    "dimensions_mm": MapOf(Qty(gt=0, hi=5.0, nullable=True),
                           doc="catalogue drawing dimensions by letter (L, L1, D1 ...; dimension_key in notes)"),
    "profile_mm": Qty(shape=("rows", 2), lo=0.0, hi=5.0, nullable=True),
    "clamping_torque_Nm": Qty(gt=0, hi=1e5, nullable=True), "axial_retention_N": Qty(gt=0, hi=1e7, nullable=True),
    "radial_stiffness_N_um": Qty(gt=0, hi=1e13, nullable=True),
    "runout_um": length(1e-3, positive=False, nullable=True),
    "runout_at_mm": length(1.0, positive=False, nullable=True), "balance_grade": BALANCE,
    "max_rpm": Qty(gt=0, hi=RAD_S_MAX, nullable=True), "mass_kg": Qty(gt=0, hi=500.0, nullable=True),
    "clamp_length_mm": length(1.0, nullable=True), "price_eur": Qty(lo=0, nullable=True),
    "price_usd": Qty(lo=0, nullable=True), "coolant_through": Flag(nullable=True),
    "max_coolant_pressure_bar": Qty(gt=0, hi=1e8, nullable=True),
    "axial_float_mm": Qty(shape=(None, "range"), lo=-0.1, hi=0.1, nullable=True),
    "axial_stiffness_N_um": Qty(gt=0, hi=1e13, nullable=True),
}, required=("id", "type", "taper", "gauge_length_mm", "nose_diameter_mm", "profile_mm"), checks=(_holder_rules,))

HOLDER_SERIES = Struct({**header("holder_series", id_key="series_id"), **SERIES_COMMON},
                       required=("format", "version", "series_id", "maker", "series", "items"))

# ----------------------------------------------------------------------------- workholding
WORKHOLDING_TYPE_KEYS = {
    "vise": ("jaw_width_mm", "max_opening_mm"),
    "chuck_3jaw": ("jaws", "max_rpm", "gripping_force_kN"),
    "chuck_4jaw": ("jaws", "max_rpm", "gripping_force_kN"),
    "collet_chuck": ("range_mm",),
    "collet": ("range_mm",),
    "tailstock_centre": ("angle_deg",),
    "steady_rest": ("grip_range_mm",),
}


def _workholding_rules(m, p, u, ctx):
    kind = get(m, "type")
    if kind in WORKHOLDING_TYPE_KEYS:
        _require(m, WORKHOLDING_TYPE_KEYS[kind], p, ctx, f"type {kind}")
    body = get(m, "body_mm")
    if isinstance(body, list) and len(body) == 2 and all(isinstance(r, list) and len(r) == 3 for r in body):
        if any(b <= a for a, b in zip(body[0], body[1])):
            ctx.add(p, "body_mm must be [[xmin, ymin, zmin], [xmax, ymax, zmax]] with min < max", "rule")


CLAMP_FORCE = Qty(gt=0, hi=1e7, nullable=True, extras={"at_handle_torque_Nm": Qty(gt=0, hi=1e4),
                                                       "at_pressure_bar": Qty(gt=0, hi=1e8),
                                                       "at_torque_Nm": Qty(gt=0, hi=1e4)})
BODY = ByType(seq=Qty(shape=("rows", 3), lo=-10.0, hi=10.0),
              mapping=Struct({"outer_diameter": length(5.0), "length": length(5.0),
                              "nose_diameter": length(5.0, nullable=True), "flange_diameter": length(5.0, nullable=True),
                              "inner_diameter": length(5.0, nullable=True),
                              "spigot_diameter_H6": length(5.0, nullable=True)}, required=("outer_diameter", "length")),
              nullable=True, expected="[[xmin, ymin, zmin], [xmax, ymax, zmax]] or {outer_diameter, length}")
CENTRIFUGAL = Struct({
    "method": Text(("vdi3106", "measured", "quadratic_in_speed")),
    "gripping_force_curve_kN": Pairs("rpm", x=(0, RAD_S_MAX), y=(0, 1e7), nullable=True),
    "loss_kN": Qty(gt=0, hi=1e7), "at_rpm": Qty(gt=0, hi=RAD_S_MAX),
}, required=("method",), nullable=True,
    checks=(lambda m, p, u, c: _require(m, ("loss_kN", "at_rpm"), p, c, "method quadratic_in_speed")
            if get(m, "method") == "quadratic_in_speed" else None,
            lambda m, p, u, c: _require(m, ("gripping_force_curve_kN",), p, c, "method measured")
            if get(m, "method") == "measured" else None))

WORKHOLDING_ITEM = Struct({
    "id": Text(pattern=ID_PATTERN, sourced=False), "order_code": text(), "type": Text(WORKHOLDING_TYPES),
    "maker": Text(), "series": Text(), "series_id": Text(pattern=ID_PATTERN, sourced=False),
    "size": ByType(text=Text(), number=Count(lo=0), nullable=True), "nominal_size_in": Qty(gt=0, hi=10.0, nullable=True),
    # vises, clamps, jaws
    "jaw_width_mm": length(2.0, nullable=True), "jaw_height_mm": length(1.0, nullable=True),
    "jaw_length_mm": length(1.0, nullable=True), "max_opening_mm": length(5.0, nullable=True),
    "base_length_mm": length(5.0, nullable=True), "self_centring": Flag(nullable=True),
    "clamping_force_kN": CLAMP_FORCE, "force_per_torque_N_per_Nm": Qty(gt=0, hi=1e6, nullable=True),
    "max_input_torque_Nm": Qty(gt=0, hi=1e4, nullable=True), "recommended_torque_Nm": Qty(gt=0, hi=1e4, nullable=True),
    "max_input_force_kN": Qty(gt=0, hi=1e7, nullable=True), "pull_back_ratio": Qty(gt=0, hi=100, nullable=True),
    "lever_ratio": Qty(gt=0, hi=100, nullable=True), "stud": text(), "screw": text(),
    "stud_preload_kN": Qty(gt=0, hi=1e7, nullable=True),
    "jaw_friction": Qty(lo=0, hi=2, nullable=True), "jaw_stiffness_N_um": Qty(gt=0, hi=1e13, nullable=True),
    "safety_factor": Struct({"continuous": Qty(gt=0, hi=20), "interrupted": Qty(gt=0, hi=20)}, nullable=True),
    "body_mm": BODY, "mass_kg": Qty(gt=0, hi=1e4, nullable=True),
    "moment_of_inertia_kgm2": Qty(gt=0, hi=1e4, nullable=True),
    "accuracy": Struct({"jaw_parallelism_um": length(1e-3, positive=False),
                        "repeatability_um": length(1e-3, positive=False),
                        "runout_um": length(1e-3, positive=False), "accuracy_um": length(1e-3, positive=False)},
                       nullable=True),
    "fits": text(), "serration": text(), "mounting": text(), "mounting_bolts": text(), "material": text(),
    "form": text(), "jaws_per_set": Count(lo=1, hi=12, nullable=True),
    # chucks and collets
    "jaws": Count(lo=1, hi=12), "max_rpm": Qty(gt=0, hi=RAD_S_MAX, nullable=True), "gripping_force_kN": CLAMP_FORCE,
    "gripping_range_mm": Qty(shape="range", gt=0, hi=5.0, nullable=True),
    "jaw_stroke_mm": length(0.5, nullable=True), "plunger_stroke_mm": length(0.5, nullable=True),
    "max_drawbar_pull_kN": Qty(gt=0, hi=1e7, nullable=True), "matching_cylinder": text(),
    "standard_soft_jaw": text(), "collet": text(),
    "jaw_mass_kg": Qty(gt=0, hi=100, nullable=True), "jaw_cg_radius_mm": length(2.0, nullable=True),
    "jaw_mass_moment_kg_mm": Qty(gt=0, hi=1e3, nullable=True),
    "centrifugal_loss_coeff": Qty(gt=0, hi=10, nullable=True),
    "centrifugal_loss_model": CENTRIFUGAL,
    "grip_length_mm": length(1.0, nullable=True), "through_hole_mm": length(2.0, positive=False, nullable=True),
    "accuracy_um": length(1e-3, positive=False, nullable=True),
    "actuation": Text(("manual", "hydraulic", "pneumatic", "electric")),
    "max_actuation_pressure_bar": Qty(gt=0, hi=1e8, nullable=True),
    # fixture plates, vacuum, magnetic, adhesive
    "plate_mm": Qty(shape=(2, 3), gt=0, hi=10.0, nullable=True), "grid_mm": length(1.0, nullable=True),
    "flatness_um": length(1e-3, positive=False, nullable=True), "surface": text(),
    "min_abs_pressure_kPa": Qty(gt=0, hi=1.1e5, nullable=True), "vacuum_with_pump_kPa": Qty(gt=0, hi=1.1e5, nullable=True),
    "seal": text(), "area_utilisation": fraction(nullable=True),
    "holding_pressure_N_cm2": Qty(gt=0, hi=1e8, table=("mm",), nullable=True),
    "max_holding_pressure_N_cm2": Qty(gt=0, hi=1e8, nullable=True), "magnetisation_levels": Count(lo=1, nullable=True),
    "thin_part_version": text(),
    "pole_pitch_mm": length(0.5, nullable=True), "shear_strength_MPa": Qty(gt=0, hi=1e9, nullable=True),
    "shear_strength_aluminium_MPa": Qty(gt=0, hi=1e9, nullable=True),
    "peel_strength_N_mm": Qty(gt=0, hi=1e7, nullable=True), "cure_s": Qty(gt=0, hi=1e6, nullable=True),
    "fixture_time_s": Qty(gt=0, hi=1e6, nullable=True), "specific_gravity": Qty(gt=0, hi=30, nullable=True),
    "bond_area_utilisation": fraction(nullable=True), "hot_strength": text(),
    # collets, centres, steady rests
    "range_mm": Qty(shape="range", gt=0, hi=2.0, nullable=True), "angle_deg": angle(1, 179),
    "taper": text(), "live": Flag(nullable=True), "radial_load_N": Qty(gt=0, hi=1e7, nullable=True),
    "max_workpiece_mass_kg": Qty(gt=0, hi=1e5, nullable=True), "concentricity_mm": length(0.01, positive=False,
                                                                                            nullable=True),
    "dims_mm": MapOf(Qty(gt=0, hi=5.0, nullable=True)),
    "grip_range_mm": Qty(shape="range", gt=0, hi=5.0, nullable=True), "rollers": Count(lo=2, hi=6),
    "roller_stiffness_N_um": Qty(gt=0, hi=1e13, nullable=True), "cylinder_bore_mm": length(1.0, nullable=True),
    "pressure_bar": Qty(gt=0, hi=1e8, shape=(None, "range"), nullable=True),
    "clamp_force_per_roller_kN": Qty(gt=0, hi=1e7, shape=(None, "range"), nullable=True,
                                     extras={"at_pressure_bar": Qty(gt=0, hi=1e8)}),
    "clamp_force_per_roller_vs_pressure_kN": Pairs("bar", x=(0, 1e8), y=(0, 1e7), nullable=True),
    "centring_accuracy_mm": length(0.01, positive=False, nullable=True),
    "repeatability_mm": length(0.01, positive=False, nullable=True),
    "max_peripheral_speed_m_min": Qty(gt=0, hi=SPEED_MAX, nullable=True),
    "price_eur": Qty(lo=0, nullable=True),
}, required=("id", "type"), checks=(_workholding_rules,))

WORKHOLDING_SERIES = Struct({**header("workholding_series", id_key="series_id"), **SERIES_COMMON},
                            required=("format", "version", "series_id", "maker", "series", "items"))

# ----------------------------------------------------------------------------- coolant
def _coolant_rules(m, p, u, ctx):
    if get(m, "kind") in ("emulsion", "semi_synthetic", "synthetic"):
        _require(m, ("concentration_pct",), p, ctx, "water-mixed fluid")
    _order(m, ("boiling_degC", "leidenfrost_degC"), p, ctx)


COOLANT = Struct({
    **header("coolant"), "extends": Ref("coolant"), "name": Text(), "kind": Text(COOLANT_KINDS),
    "concentration_pct": Qty(lo=0, hi=1, nullable=True),
    "density_kg_m3": Qty(gt=0, hi=3000.0, table=True), "viscosity_mPa_s": Qty(gt=0, hi=100.0, table=True),
    "kinematic_viscosity_mm2_s": Qty(gt=0, hi=1.0, table=True, nullable=True),
    "specific_heat_J_kgK": Qty(gt=0, hi=10000.0, table=True), "conductivity_W_mK": Qty(gt=0, hi=10.0, table=True),
    "surface_tension_mN_m": Qty(gt=0, hi=1.0, nullable=True), "boiling_degC": temp(nullable=True),
    "leidenfrost_degC": temp(nullable=True), "leidenfrost_water_reference_degC": temp(nullable=True),
    "flash_point_degC": temp(nullable=True), "smoke_point_degC": temp(nullable=True),
    "pour_point_degC": temp(nullable=True),
    "friction_factor": Seq(Struct({"tool_class": Text(TOOL_CLASSES + ("any",)), "work_family": Text(FAMILIES + ("any",)),
                                   "value": Qty(gt=0, hi=2), "range": Qty(shape="range", gt=0, hi=2)},
                                  required=("tool_class", "work_family", "value"))),
    "lubricity_index": fraction(), "drag_out_ml_per_kg_chips": Qty(lo=0, hi=1e-2),
    "evaporation_l_per_h_per_m2": Qty(lo=0, hi=1e-3), "mist_loss_fraction": fraction(),
    "heat_transfer": Struct({"latent_heat_kJ_kg": Qty(gt=0, hi=1e8, nullable=True),
                             "flood_htc_W_m2K": Qty(gt=0, hi=1e7, nullable=True),
                             "jet_htc_W_m2K": Qty(gt=0, hi=1e7, nullable=True),
                             "mist_htc_W_m2K": Qty(gt=0, hi=1e7, nullable=True),
                             "air_carrier_htc_W_m2K": Qty(gt=0, hi=1e7, nullable=True),
                             "free_htc_W_m2K": Qty(gt=0, hi=1e7, nullable=True)}, nullable=True),
    "supply_pressure_bar": Qty(gt=0, hi=1e8, nullable=True), "typical_flow_l_min": Qty(gt=0, hi=1.0, nullable=True),
    "mql_flow_ml_h": Qty(gt=0, hi=1e-3, nullable=True), "dosage_ml_h": Qty(shape="range", gt=0, hi=1e-3, nullable=True),
    "dosage_default_ml_h": Qty(gt=0, hi=1e-3, nullable=True), "min_air_pressure_bar": Qty(gt=0, hi=1e8, nullable=True),
    "aerosol_droplet_um": Qty(shape="range", gt=0, hi=1e-3, nullable=True),
    "system_capacity_ml_h": Struct({"one_channel_max": Qty(gt=0, hi=1e-3), "two_channel_min_max": Qty(gt=0, hi=1e-3)},
                                   nullable=True),
    "max_noack_loss_pct": Qty(gt=0, hi=1, nullable=True),
    "cost_eur_per_l": Qty(lo=0, nullable=True),
}, required=("format", "version", "id", "name", "kind", "density_kg_m3", "viscosity_mPa_s", "specific_heat_J_kgK",
             "conductivity_W_mK"), checks=(_coolant_rules,), hints=EXTENDS_HINTS)

# ----------------------------------------------------------------------------- controller
SMOOTHING_MODE = Struct({"code": Text(), "name": Text(), "tolerance_mm": length(0.01, positive=False, nullable=True),
                         "tolerance_range_mm": Qty(shape="range", lo=0, hi=1.0),
                         "rotary_tolerance_range_deg": Qty(shape="range", lo=0, hi=TWO_PI)},
                        required=("code",))
CONTROLLER = Struct({
    **header("controller"), "extends": Ref("controller"), "dialect": Text(DIALECTS), "name": Text(),
    "alarms": Seq(Struct({"code": ByType(text=Text(), number=Count(lo=0), expected="text or an integer"),
                          "text": Text(allow_empty=True), "trigger": Text(pattern=r"[a-z][a-z0-9_]*"),
                          "cause": Text(), "remedy": Text(), "kind": Text()},
                         required=("code", "text")), unique="code"),
    "messages": MapOf(Any_()),
    "timing": Struct({"ipo_period_ms": Qty(gt=0, hi=0.1), "block_processing_ms": Qty(lo=0, hi=1.0),
                      "lookahead_blocks": Count(lo=0, hi=100000),
                      "lookahead_option_blocks": Count(lo=0, hi=100000),
                      "lookahead_limit_range_blocks": Qty(shape="range", lo=0, hi=1e6, dimensionless=True),
                      "radius_comp_lookahead_blocks": Count(lo=0, hi=100000),
                      "position_controller_cycle_ms": Qty(gt=0, hi=0.1),
                      "speed_controller_cycle_ms": Qty(gt=0, hi=0.1),
                      "current_controller_cycle_ms": Qty(gt=0, hi=0.1),
                      "hsm_max_contouring_feed_ipm": Qty(gt=0, hi=10.0),
                      "corner_rounding_default_in": length(0.1, positive=False),
                      "arc_end_point_tolerance_mm": length(0.01, positive=False),
                      "min_circle_mm": length(0.01, positive=False),
                      "input_resolution_linear_um": length(1e-3), "input_resolution_rotary_deg": angle(0, 1),
                      "smoothing": Seq(SMOOTHING_MODE)}),
    "words": MapOf(Any_()),
    "variables": Any_(),
}, required=("format", "version", "id", "dialect", "name", "alarms"), hints=EXTENDS_HINTS)

# ----------------------------------------------------------------------------- machine
AXIS_NAME = r"[A-Z][0-9]?"
SERVO = Struct({"kv_1_s": Qty(gt=0, hi=1e4), "feedforward": Qty(lo=0, hi=1.2), "velocity_bw_Hz": Qty(gt=0, hi=1e5),
                "current_bw_Hz": Qty(gt=0, hi=1e6)}, required=("kv_1_s",))
LINEAR_AXIS = Struct({
    "type": Text(("linear",), sourced=False), "travel_mm": Qty(shape="range", lo=-100.0, hi=100.0, strict=True),
    "travel_non_guide_bush_mm": length(10.0, nullable=True),
    "rapid_mm_min": Qty(gt=0, hi=10.0), "max_feed_mm_min": Qty(gt=0, hi=10.0), "accel_m_s2": Qty(gt=0, hi=200.0),
    "jerk_m_s3": Qty(gt=0, hi=1e6), "servo": SERVO, "backlash_um": length(1e-3, positive=False),
    "reversal_spike_um": length(1e-3, positive=False),
    "ballscrew": Struct({"diameter_mm": length(0.5), "pitch_mm": length(0.2), "length_mm": length(50.0),
                         "fixed_end": Text(("motor", "both", "free_end", "far_end")), "cooled_nut": Flag()},
                        nullable=True),
    "drive": Text(("ballscrew", "linear_motor", "rack_pinion", "belt")),
    "feedback": Text(("rotary_encoder", "linear_scale")), "guideways": text(),
    "positioning": Struct({"accuracy_um": length(1e-3, positive=False), "repeatability_um": length(1e-3, positive=False)}),
    "motor_force_N": Qty(gt=0, hi=1e7), "moving_mass_kg": Qty(gt=0, hi=1e6), "max_load_kg": Qty(gt=0, hi=1e6),
    "brake": Struct({"holds_on_power_loss": Flag(), "clamp_force_N": Qty(gt=0, hi=1e7), "clamp_time_s": Qty(gt=0, hi=60)},
                    nullable=True),
}, required=("type", "travel_mm", "rapid_mm_min", "max_feed_mm_min", "accel_m_s2", "jerk_m_s3", "servo"))
ROTARY_AXIS = Struct({
    "type": Text(("rotary",), sourced=False),
    "travel_deg": Qty(shape="range", lo=-100 * TWO_PI, hi=100 * TWO_PI, strict=True, nullable=True),
    "continuous": Flag(), "max_rpm": Qty(gt=0, hi=RAD_S_MAX), "rapid_deg_s": Qty(gt=0, hi=RAD_S_MAX, nullable=True),
    "index_90deg_s": Qty(gt=0, hi=60, nullable=True), "accel_rad_s2": Qty(gt=0, hi=1e5),
    "jerk_rad_s3": Qty(gt=0, hi=1e8), "servo": SERVO, "backlash_arcsec": Qty(lo=0, hi=0.01),
    "brake": Struct({"clamp_torque_Nm": Qty(gt=0, hi=1e6), "clamp_time_s": Qty(gt=0, hi=60),
                     "unclamped_stiffness_Nm_rad": Qty(gt=0, hi=1e12), "clamped_stiffness_Nm_rad": Qty(gt=0, hi=1e12),
                     "clamp_force_N": Qty(gt=0, hi=1e7), "brake_diameter_mm": length(5.0)}, nullable=True),
    "positioning": Struct({"accuracy_arcsec": Qty(lo=0, hi=0.01), "repeatability_arcsec": Qty(lo=0, hi=0.01)}),
    "motor_torque_Nm": Qty(gt=0, hi=1e6), "inertia_kgm2": Qty(gt=0, hi=1e6), "max_load_kg": Qty(gt=0, hi=1e6),
    "drive": Text(), "drive_rating_kW": Qty(gt=0, hi=1e6), "cooled_bearing": Flag(),
    "max_feed_mm_min": Qty(gt=0, hi=10.0), "rapid_mm_min": Qty(gt=0, hi=10.0),
}, required=("type", "max_rpm", "accel_rad_s2", "jerk_rad_s3", "servo"),
    checks=(lambda m, p, u, c: None if has(m, "travel_deg") or get(m, "continuous") is True
            else c.add(p, "a rotary axis needs travel_deg unless continuous: true", "required"),),
    hints={"positioning_repeatability_arcsec": "write positioning: {repeatability_arcsec: ..}"})
AXIS = Tagged("type", {"linear": LINEAR_AXIS, "rotary": ROTARY_AXIS})

LINK = Struct({
    "link": Text(pattern=r"[a-z][a-z0-9_]*", sourced=False), "axis": Text(pattern=AXIS_NAME, sourced=False),
    "direction": vec3(-1.0, 1.0, unit_vector=True), "axis_point_mm": vec3(-100.0, 100.0),
    "fixed": POSE,
}, required=("link",), checks=(
    lambda m, p, u, c: c.add(p, "a link is either an NC axis (axis + direction) or fixed, not both", "rule")
    if has(m, "axis") and has(m, "fixed") else (
        _require(m, ("direction",), p, c, "an axis link") if has(m, "axis") else
        (None if has(m, "fixed") else c.add(p, "a link needs axis (+ direction) or fixed", "required"))),))
CHAIN = Seq(LINK, min_len=1, unique="link")

KINEMATICS = Struct({
    "workpiece_chain": CHAIN, "tool_chain": CHAIN,
    "workpiece_chains": MapOf(CHAIN, key=Text(pattern=r"[a-z][a-z0-9_]*"),
                              doc="further workpiece chains by name (sub spindle ...)"),
    "tool_chains": MapOf(CHAIN, key=Text(pattern=r"[a-z][a-z0-9_]*"),
                         doc="further tool chains by name (back tool post, lower turret ...)"),
    "home_mm": MapOf(Qty(lo=-100.0, hi=100.0), key=Text(pattern=AXIS_NAME)),
    "home_deg": MapOf(Qty(lo=-100 * TWO_PI, hi=100 * TWO_PI), key=Text(pattern=AXIS_NAME)),
    "tool_axis": vec3(-1.0, 1.0, unit_vector=True),
    "singularities": Seq(Struct({"axis": Text(pattern=AXIS_NAME), "when": Text(),
                                 "behaviour": Text(("slow_down", "alarm"))}, required=("axis", "when", "behaviour")),
                         nullable=True),
    "gravity_direction": vec3(-1.0, 1.0, unit_vector=True),
    "mrzp_mm": MapOf(Qty(lo=-100.0, hi=100.0), key=Text(pattern=AXIS_NAME)),
    "turret_index_axis": ByType(text=Text(pattern=AXIS_NAME), seq=vec3(-1.0, 1.0, unit_vector=True),
                                expected="the indexing axis name or the turret's index axis direction [x, y, z]"),
}, required=("workpiece_chain", "tool_chain", "tool_axis"))

MODE = Struct({"fn_Hz": Qty(gt=0, hi=1e6), "zeta": Qty(gt=0, lt=1.0), "k_N_um": Qty(gt=0, hi=1e13),
               "k_Nm_rad": Qty(gt=0, hi=1e13), "direction": ByType(text=Text(("x", "y", "z")),
                                                                   seq=vec3(-1.0, 1.0, nonzero=True),
                                                                   expected="x, y, z or a vector"),
               "axis": Text(pattern=AXIS_NAME)},
              required=("fn_Hz", "zeta"),
              checks=(lambda m, p, u, c: None if has(m, "k_N_um") or has(m, "k_Nm_rad")
                      else c.add(p, "a mode needs its modal stiffness k_N_um (or k_Nm_rad for a rotary mode)",
                                 "required"),))

CURVE_POWER = Pairs("rpm", x=(0, RAD_S_MAX), y=(0, 1e7))
CURVE_TORQUE = Pairs("rpm", x=(0, RAD_S_MAX), y=(0, 1e6))
RATING = Struct({"rating": Text(), "scale_of_peak": Qty(gt=0, hi=1.0), "power_kW": CURVE_POWER,
                 "torque_Nm": CURVE_TORQUE},
                required=("rating",),
                checks=(lambda m, p, u, c: None if has(m, "scale_of_peak") or has(m, "power_kW") or has(m, "torque_Nm")
                        else c.add(p, "a rating needs scale_of_peak or its own power_kW / torque_Nm curves",
                                   "required"),))
GEAR_RANGE = Struct({
    "rpm": Qty(shape="range", lo=0, hi=RAD_S_MAX), "power_kW": CURVE_POWER, "torque_Nm": CURVE_TORQUE,
    "rating": Text(), "additional_ratings": Seq(RATING),
}, required=("rpm", "power_kW", "torque_Nm"))


def _spindle_rules(m, p, u, ctx):
    max_rad_s = si_number(m, "max_rpm")
    gears = get(m, "gear_ranges")
    if not gears:
        ctx.warn(p, "a spindle without gear_ranges has no power and torque curves: a project that cuts with it is "
                    "refused by its check")
    if is_number(max_rad_s) and isinstance(gears, list):
        for i, gear in enumerate(gears):
            span = si_number(gear, "rpm") if isinstance(gear, Mapping) else None
            if isinstance(span, list) and len(span) == 2 and span[1] > max_rad_s * (1 + 1e-9):
                ctx.add(p + ("gear_ranges", i), "gear range goes above the spindle max_rpm", "rule")


# max_accel_rad_s2_per_gear (template key, unit not last): one value per gear range, rad/s2 (already SI,
# so the key is kept and nothing is scaled).
POSITION_LOOP = Struct({"kv_1_s": Qty(gt=0, hi=1e4), "feedforward": Qty(lo=0, hi=1.2),
                        "encoder_counts": Count(lo=1, hi=10 ** 9),
                        "max_accel_rad_s2_per_gear": Seq(Qty(gt=0, hi=1e6)),
                        "velocity_bw_Hz": Qty(gt=0, hi=1e5)}, nullable=True)
SPINDLE = Struct({
    "id": Text(pattern=r"[a-z][a-z0-9_-]*", sourced=False), "kind": Text(("milling", "turning", "live_tool", "sub")),
    "nose": Text(), "max_rpm": Qty(gt=0, hi=RAD_S_MAX), "min_rpm": Qty(lo=0, hi=RAD_S_MAX),
    "direction_reversible": Flag(), "gear_ranges": Seq(GEAR_RANGE),
    "accel_time_s": Qty(gt=0, hi=600, nullable=True), "decel_time_s": Qty(gt=0, hi=600, nullable=True),
    "gear_change_s": Qty(lo=0, hi=600, nullable=True),
    "orient_time_s": Qty(gt=0, hi=60, nullable=True), "position_loop": POSITION_LOOP,
    "rotor_inertia_kgm2": Qty(gt=0, hi=1e4), "efficiency": Qty(gt=0, hi=1),
    "idle_power_W": Qty(lo=0, hi=1e6), "runout_um": length(1e-3, positive=False), "frf_modes": Seq(MODE),
    "thermal": Struct({"growth_at_max_speed_um": length(1e-2, positive=False), "time_constant_min": Qty(gt=0, hi=1e6),
                       "bearing_heat_at_max_speed_W": Qty(lo=0, hi=1e6), "thermal_comp": Flag(),
                       "comp_residual_fraction": fraction()}),
    "drive": Text(), "drive_power_kW": Qty(gt=0, hi=1e6, nullable=True), "bearing_lubrication": Text(),
    "cooling": Text(), "option": Flag(),
    "bar_capacity_mm": length(2.0), "bore_mm": length(2.0), "bearing_id_mm": length(2.0),
    "chuck_mm": length(3.0), "centre_height_mm": length(5.0),
    "face_to_face_mm": length(20.0), "spindle_nose_detail": Text(), "clamping_force_N": Qty(gt=0, hi=1e7),
    "max_machining_diameter_mm": length(5.0), "max_chucking_diameter_mm": length(5.0),
    "front_ejection_length_mm": length(5.0), "max_work_projection_mm": length(5.0),
    "max_drilling_mm": length(1.0), "max_tapping": Text(),
}, required=("id", "kind", "nose", "max_rpm", "gear_ranges"), checks=(_spindle_rules,))

PROBE = Struct({
    "model": Text(), "kind": Text(("contact", "laser", "touch")), "option": Flag(),
    "trigger_force_N": Qty(shape=(None, "range"), gt=0, hi=1e3), "trigger_force_z_N": Qty(gt=0, hi=1e3),
    "pretravel_um": length(1e-3, positive=False), "repeatability_um": length(1e-3, positive=False),
    "overtravel_xy_deg": angle(0, 90), "overtravel_xy_mm": length(0.1, positive=False),
    "overtravel_z_mm": length(0.1, positive=False), "stylus_ball_mm": length(0.1), "stylus_length_mm": length(1.0),
    "position_mm": vec3(-100.0, 100.0), "disc_mm": length(0.5), "beam_diameter_mm": length(0.05), "laser": Text(),
}, nullable=True)
CALIBRATION_BALL = Struct({"diameter_mm": length(0.5), "position_mm": vec3(-100.0, 100.0), "model": Text(),
                           "attached_to": Text(pattern=r"[a-z][a-z0-9_]*")}, nullable=True)

PUMP = Struct({"id": Text(pattern=r"[A-Za-z0-9_-]+", sourced=False), "kind": Text(JET_KINDS),
               "pressure_bar": Qty(gt=0, hi=1e8), "flow_l_min": Qty(gt=0, hi=1.0), "power_kW": Qty(gt=0, hi=1e6),
               "pressure_flow_bar_l_min": Pairs("bar", x=(0, 1e8), y=(0, 1.0), increasing=True), "option": Flag()},
              required=("id", "kind", "pressure_bar", "flow_l_min"))
NOZZLE = Struct({"id": Text(pattern=r"[A-Za-z0-9_-]+", sourced=False), "kind": Text(JET_KINDS),
                 "attached_to": Text(pattern=r"[a-z][a-z0-9_]*"), "position_mm": vec3(-100.0, 100.0),
                 "aim": vec3(-1e3, 1e3, nonzero=True), "diameter_mm": length(0.1), "pump": Text(),
                 "programmable": Struct({"code": Text(), "positions": Count(lo=1, hi=10000),
                                         "step_deg": angle(0, 180), "home_aim": vec3(-1e3, 1e3, nonzero=True),
                                         "step_time_s": Qty(gt=0, hi=60)},
                                        required=("code", "positions"), nullable=True),
                 "count": Count(lo=1, hi=1000), "spread_deg": angle(0, 180), "option": Flag()},
                required=("id", "kind", "attached_to", "pump"),
                checks=(lambda m, p, u, c: None if get(m, "kind") in ("through_spindle", "through_tool")
                        else _require(m, ("position_mm",), p, c, "a nozzle outside the spindle")
                        if not isinstance(get(m, "programmable"), Mapping) else None,
                        lambda m, p, u, c: None if has(m, "aim") or isinstance(get(m, "programmable"), Mapping)
                        and has(get(m, "programmable"), "home_aim")
                        else c.add(p, "a nozzle needs aim (or programmable.home_aim)", "required")))

PRIMITIVE = OneKey({
    "box": Struct({"min_mm": vec3(-100.0, 100.0), "max_mm": vec3(-100.0, 100.0)}, required=("min_mm", "max_mm")),
    "cylinder": Struct({"p0_mm": vec3(-100.0, 100.0), "p1_mm": vec3(-100.0, 100.0), "radius_mm": length(50.0)},
                       required=("p0_mm", "p1_mm", "radius_mm")),
    "cone": Struct({"p0_mm": vec3(-100.0, 100.0), "p1_mm": vec3(-100.0, 100.0), "r0_mm": length(50.0, positive=False),
                    "r1_mm": length(50.0, positive=False)}, required=("p0_mm", "p1_mm", "r0_mm", "r1_mm")),
    "sphere": Struct({"centre_mm": vec3(-100.0, 100.0), "radius_mm": length(50.0)}, required=("centre_mm", "radius_mm")),
    "mesh": Pose({"file": Text(), "scale": Qty(gt=0, hi=1e4)}, required=("file",)),
})

GEOMETRY_ITEM = Struct({"name": Text(pattern=r"[a-z][a-z0-9_-]*"), "attached_to": Text(pattern=r"[a-z][a-z0-9_]*"),
                        "primitives": Seq(PRIMITIVE, min_len=1), "collide": Flag(),
                        "colour": Text(pattern=r"#[0-9a-fA-F]{6}")},
                       required=("name", "attached_to", "primitives"))


def chain_links(kinematics) -> dict[str, list]:
    """Every chain of a machine's kinematics section by name (raw links)."""
    chains: dict[str, list] = {}
    if not isinstance(kinematics, Mapping):
        return chains
    for key in ("workpiece_chain", "tool_chain"):
        if isinstance(get(kinematics, key), list):
            chains[key] = get(kinematics, key)
    for group in ("workpiece_chains", "tool_chains"):
        extra = get(kinematics, group)
        if isinstance(extra, Mapping):
            for name, chain in extra.items():
                if isinstance(chain, list):
                    chains[f"{group}.{name}"] = chain
    return chains


def _machine_rules(m, p, u, ctx):
    axes = get(m, "axes")
    kinematics = get(m, "kinematics")
    axis_types = {}
    if isinstance(axes, Mapping):
        for name, axis in axes.items():
            if isinstance(axis, Mapping):
                axis_types[name] = unwrap(axis.get("type"))
    links = {"base"}
    used: dict[str, tuple] = {}
    kpath = p + ("kinematics",)
    for chain_name, chain in chain_links(kinematics).items():
        for i, link in enumerate(chain):
            if not isinstance(link, Mapping):
                continue
            name = unwrap(link.get("link"))
            if isinstance(name, str):
                links.add(name)
            axis = unwrap(link.get("axis"))
            where = kpath + tuple(chain_name.split(".")) + (i,)
            if axis is None:
                continue
            if axis not in axis_types:
                ctx.add(where, f"axis {axis!r} is not defined under axes", "reference")
            elif axis in used:
                ctx.add(where, f"axis {axis} moves two links", "rule")
            else:
                used[axis] = where
            if axis_types.get(axis) == "rotary" and not has(link, "axis_point_mm"):
                ctx.warn(where, f"rotary link {axis} without axis_point_mm (rotates about the link origin)")
    for turret in get(m, "turrets") or []:
        if isinstance(turret, Mapping):
            name = unwrap(turret.get("id"))
            if isinstance(name, str):
                links.add(name)
            for link in unwrap(turret.get("chain")) or []:
                if isinstance(link, Mapping) and isinstance(unwrap(link.get("link")), str):
                    links.add(unwrap(link.get("link")))
                    axis = unwrap(link.get("axis"))
                    if isinstance(axis, str):
                        used.setdefault(axis, kpath)
    if isinstance(kinematics, Mapping):
        turret_axes = get(kinematics, "turret_index_axis")
        for axis in ([turret_axes] if isinstance(turret_axes, str) else turret_axes or []):
            used.setdefault(axis, kpath)
        for name in axis_types:
            if name not in used:
                ctx.add(kpath, f"axis {name} is not in any kinematic chain", "rule")
        for key, kind in (("home_mm", "linear"), ("home_deg", "rotary")):
            home = get(kinematics, key)
            if isinstance(home, Mapping):
                for name in home:
                    if name in axis_types and axis_types[name] != kind:
                        ctx.add(kpath + (key, name), f"{kind} homes belong in {key}; axis {name} is "
                                                     f"{axis_types[name]}", "rule")
    spindle_ids = [unwrap(s.get("id")) for s in get(m, "spindles") or [] if isinstance(s, Mapping)]
    coolant = get(m, "coolant")
    if isinstance(coolant, Mapping):
        cpath = p + ("coolant",)
        pumps = [unwrap(x.get("id")) for x in (get(coolant, "pumps") or []) if isinstance(x, Mapping)]
        for i, nozzle in enumerate(get(coolant, "nozzles") or []):
            if not isinstance(nozzle, Mapping):
                continue
            pump = unwrap(nozzle.get("pump"))
            if pump is not None and pump not in pumps:
                ctx.add(cpath + ("nozzles", i, "pump"), f"pump {pump!r} is not in coolant.pumps", "reference")
            attached = unwrap(nozzle.get("attached_to"))
            if attached is not None and attached not in links:
                ctx.add(cpath + ("nozzles", i, "attached_to"), f"link {attached!r} is not in a kinematic chain",
                        "reference")
    for i, item in enumerate(get(m, "geometry") or []):
        if isinstance(item, Mapping):
            attached = unwrap(item.get("attached_to"))
            if attached is not None and attached not in links:
                ctx.add(p + ("geometry", i, "attached_to"), f"link {attached!r} is not in a kinematic chain "
                                                            f"(or base)", "reference")
    for i, turret in enumerate(get(m, "turrets") or []):
        if isinstance(turret, Mapping):
            live = unwrap(turret.get("live_spindle"))
            if live is not None and live not in spindle_ids:
                ctx.add(p + ("turrets", i, "live_spindle"), f"spindle {live!r} is not in spindles", "reference")
            for j, link in enumerate(unwrap(turret.get("chain")) or []):
                if isinstance(link, str) and link not in links:
                    ctx.add(p + ("turrets", i, "chain", j), f"link {link!r} is not in a kinematic chain", "reference")
    if get(m, "type") in ("lathe", "mill_turn", "swiss") and not get(m, "turrets") \
            and not isinstance(get(m, "tool_changer"), Mapping):
        ctx.add(p, "a lathe needs turrets (or a tool_changer)", "required")


MACHINE_CONTROLLER = Struct({
    "dialect": Text(DIALECTS), "model": Text(), "alarms": Ref("controller", nullable=True),
    "ipo_period_ms": Qty(gt=0, hi=0.1), "block_processing_ms": Qty(lo=0, hi=1.0),
    "lookahead_blocks": Count(lo=0, hi=100000), "least_input_increment_mm": length(1e-3),
    "least_input_increment_deg": angle(1e-7, 1), "max_feed_mm_min": Qty(gt=0, hi=10.0),
    "rapid_mm_min": ByType(number=Qty(gt=0, hi=10.0), mapping=MapOf(Qty(gt=0, hi=10.0), key=Text(pattern=AXIS_NAME)),
                           expected="a number or {axis: rapid}"),
    "smoothing": Seq(Struct({"code": Text(), "name": Text(), "tolerance_mm": length(0.01, positive=False)},
                            required=("code", "tolerance_mm"))),
    "corner_default": Text(("blend", "exact_stop")), "in_position_mm": length(0.01),
    "rapid_override_steps_pct": Qty(shape="list", lo=0, hi=2.0, ordered=True),
    "feed_override_range_pct": Qty(shape="range", lo=0, hi=5.0),
    "spindle_override_range_pct": Qty(shape="range", lo=0, hi=5.0),
    "tool_life_management": Flag(), "probing": Flag(), "diameter_programming_X": Flag(),
    "paths": Count(lo=1, hi=16),
    "five_axis": Struct({"tcp": Text(), "tcpc": Text(), "tcpm": Text(), "dwo": Text(), "tilted_plane": Text(),
                         "mrzp_settings": ByType(text=Text(), seq=Seq(ByType(text=Text(), number=Count(lo=0))),
                                                 expected="text or a list of setting numbers")},
                        nullable=True),
}, required=("dialect", "model", "ipo_period_ms", "least_input_increment_mm", "max_feed_mm_min", "rapid_mm_min"))

TURRET = Struct({
    "id": Text(pattern=r"[a-z][a-z0-9_-]*", sourced=False), "stations": Count(lo=1, hi=200),
    "mount": Text(), "index_time_s": Struct({"adjacent": Qty(gt=0, hi=60), "max": Qty(gt=0, hi=60)},
                                            required=("adjacent", "max")),
    "live_stations": Qty(shape="list", lo=1, hi=200, dimensionless=True),
    "live_spindle": Text(),
    "chain": ByType(seq=Seq(ByType(mapping=LINK, text=Text(pattern=r"[a-z][a-z0-9_]*"),
                                   expected="a link or the name of a link in a kinematic chain")),
                    expected="a list of links"),
    "index_angle_deg": angle(0, 360), "clearance_diameter_mm": length(10.0), "max_swing_diameter_mm": length(10.0),
    "boring_bar_rear_clearance_mm": length(10.0), "max_turning_diameter_mm": length(10.0),
    "max_turning_length_mm": length(10.0)}, required=("id", "stations", "index_time_s"))

MACHINE = Struct({
    **header("machine"), "extends": Ref("machine"), "maker": Text(), "model": Text(), "type": Text(MACHINE_TYPES),
    "controller": MACHINE_CONTROLLER,
    "axes": MapOf(AXIS, key=Text(pattern=AXIS_NAME), min_len=1),
    "kinematics": KINEMATICS,
    "spindles": Seq(SPINDLE, min_len=1, unique="id"),
    "structure": Struct({"modes": Seq(MODE), "table_modes": Seq(MODE), "rotary_modes": Seq(MODE),
                         "turret_modes": Seq(MODE)}),
    "error_map": Struct({
        "squareness_urad": MapOf(Qty(lo=-1e-2, hi=1e-2), key=Text(pattern=r"[A-Z][0-9]?[A-Z][0-9]?")),
        "straightness_um_per_m": MapOf(Qty(lo=-1e-2, hi=1e-2), key=Text(pattern=AXIS_NAME)),
        "angular_urad": MapOf(Struct({"pitch": Qty(lo=-1e-2, hi=1e-2), "yaw": Qty(lo=-1e-2, hi=1e-2),
                                      "roll": Qty(lo=-1e-2, hi=1e-2)}), key=Text(pattern=AXIS_NAME)),
        "pivot_errors": MapOf(Struct({"position_um": vec3(-1e-2, 1e-2), "tilt_urad": vec3(-1e-2, 1e-2)}),
                              key=Text(pattern=AXIS_NAME)),
        "y_axis_centre_height_um": Qty(lo=-1e-2, hi=1e-2, nullable=True),
    }, nullable=True),
    "thermal": Struct({
        "axis_drift": Struct({"expansion_1_K": Qty(lo=0, hi=1e-4), "heat_per_m_travel_J": Qty(lo=0, hi=1e6),
                              "time_constant_min": Qty(gt=0, hi=1e6), "electronic_comp": Flag(),
                              "scale_feedback": Flag(), "comp_residual_fraction": fraction()}),
        "structure_sensitivity_um_per_K": Qty(lo=0, hi=1e-3),
        "warmup": Struct({"cold_to_warm_min": Qty(gt=0, hi=1e6)}),
        "mrzp_drift_um": Qty(lo=0, hi=1e-2, nullable=True),
    }),
    "tool_changer": Struct({
        "type": Text(("umbrella", "side_mount", "chain", "drum", "rack", "matrix", "turret", "none")),
        "capacity": Count(lo=0, hi=10000), "tool_to_tool_s": Qty(gt=0, hi=600, nullable=True),
        "chip_to_chip_s": Qty(gt=0, hi=600, nullable=True),
        "max_tool_diameter_mm": length(1.0, nullable=True), "max_tool_diameter_adjacent_empty_mm": length(1.0, nullable=True),
        "max_tool_length_mm": length(2.0, nullable=True), "max_tool_mass_kg": Qty(gt=0, hi=500, nullable=True),
        "max_tool_mass_moment_Nm": Qty(gt=0, hi=1e4, nullable=True), "pocket_pitch_mm": length(1.0, nullable=True),
        "random": Flag(), "carousel_diameter_mm": length(10.0), "arm_swing_diameter_mm": length(10.0),
        "loading": Text()}, required=("type",), nullable=True),
    "turrets": Seq(TURRET, unique="id"),
    "pallet_changer": Struct({"pallets": Count(lo=1, hi=1000), "change_time_s": Qty(gt=0, hi=3600),
                              "pallet_size_mm": Qty(shape=(None, 2), gt=0, hi=10.0),
                              "pallet_top_height_mm": length(10.0, positive=False), "max_part_swing_mm": length(10.0),
                              "apc_swing_diameter_mm": length(20.0), "clamping": Text(), "pallet_pattern": Text(),
                              "pool_option": ByType(text=Text(), boolean=Flag(), expected="text or true/false")},
                             required=("pallets", "change_time_s"), nullable=True),
    "tailstock": Struct({"travel_mm": Qty(shape=(None, "range"), lo=-100.0, hi=100.0),
                         "thrust_N": Qty(shape=(None, "range"), gt=0, hi=1e7), "quill_mm": length(1.0, nullable=True),
                         "centre": Text(("live", "dead")), "centre_positions_mm": Qty(shape="list", lo=-100.0, hi=100.0),
                         "option": Flag()}, nullable=True),
    "steady_rest": Struct({"grip_range_mm": Qty(shape="range", gt=0, hi=5.0), "rollers": Count(lo=2, hi=6),
                           "roller_stiffness_N_um": Qty(gt=0, hi=1e13), "option": Flag()},
                          required=("grip_range_mm",), nullable=True),
    "guide_bushing": Struct({"type": Text(("rotary", "fixed")), "clearance_um": length(1e-3, positive=False),
                             "position_mm": Qty(lo=-100.0, hi=100.0), "switchable": Flag(),
                             "max_bar_diameter_mm": length(1.0)}, required=("type",), nullable=True),
    "doors": Struct({"count": Count(lo=0, hi=20), "auto": Flag(), "open_time_s": Qty(gt=0, hi=600, nullable=True),
                     "opening_width_mm": length(20.0), "side_door_width_mm": length(20.0)}),
    "probes": Struct({"spindle_probe": PROBE, "tool_setter": PROBE, "laser": PROBE,
                      "calibration_ball": CALIBRATION_BALL}),
    "coolant": Struct({
        "tank_l": Qty(gt=0, hi=1e3), "pumps": Seq(PUMP, unique="id"), "nozzles": Seq(NOZZLE, unique="id"),
        "air_blast": Struct({"flow_l_min": Qty(gt=0, hi=10.0), "pressure_bar": Qty(gt=0, hi=1e8)}, nullable=True),
        "chiller": Struct({"capacity_kW": Qty(gt=0, hi=1e6, nullable=True), "setpoint_degC": temp(nullable=True),
                           "cools": Text(), "kind": Text()}, nullable=True),
        "filtration_um": length(1e-2)}),
    "chip_conveyor": Struct({"type": Text(("auger", "hinged_belt", "hinge_belt", "belt", "scraper", "magnetic",
                                           "drum_filter", "none")),
                             "rate_kg_h": Qty(gt=0, hi=1e3), "discharge_height_mm": length(10.0), "position": Text(),
                             "discharge": Text(), "option": Flag()}, nullable=True),
    "enclosure": Struct({"box_mm": Qty(shape=("rows", 3), lo=-100.0, hi=100.0), "floor_slope_deg": angle(0, 90)}),
    "power": Struct({"base_W": Qty(lo=0, hi=1e7), "servo_standby_W": Qty(lo=0, hi=1e7),
                     "air_consumption_l_min": Qty(lo=0, hi=10.0), "connected_power_kW": Qty(gt=0, hi=1e8),
                     "supply_kVA": Qty(gt=0, hi=1e8), "supply_V": Qty(shape=(None, "range"), gt=0, hi=1e5),
                     "supply": Text(), "machine_mass_kg": Qty(gt=0, hi=1e7),
                     "footprint_mm": Qty(shape=(2, 3), gt=0, hi=100.0),
                     "dimensions_mm": Qty(shape=3, gt=0, hi=100.0)}),
    "workholding_interface": Struct({
        "clamping_area_mm": Qty(shape=(None, 2), gt=0, hi=100.0, nullable=True), "max_load_kg": Qty(gt=0, hi=1e6),
        "max_workpiece": Struct({"diameter_mm": length(20.0), "height_mm": length(20.0), "length_mm": length(50.0)}),
        "platter_diameter_mm": length(20.0), "max_combined_height_mm": length(20.0),
        "t_slots": Struct({"count": Count(lo=0, hi=100), "pitch_mm": length(5.0), "width_mm": length(0.2)})}),
    "capacity": Struct({"max_swing_mm": length(10.0), "max_turning_diameter_mm": length(10.0),
                        "max_turning_length_mm": length(20.0), "max_bar_diameter_mm": length(1.0),
                        "max_part_mass_kg": Qty(gt=0, hi=1e5)}, nullable=True),
    "geometry": Seq(GEOMETRY_ITEM, unique="name"),
}, required=("format", "version", "id", "maker", "model", "type", "controller", "axes", "kinematics", "spindles"),
    checks=(_machine_rules,), hints=EXTENDS_HINTS)

# ----------------------------------------------------------------------------- validation cases
MEASURE = ByType(number=Qty(lo=-1e30, hi=1e30, dimensionless=True), text=Text(), boolean=Flag(), seq=Any_(),
                 mapping=Struct({"value": Any_(), "assumed": Flag(), "src": Text(), "page": Text(), "unit": Text()}),
                 expected="a value or {value, assumed: true, note}")
VALIDATION_CASE = Struct({
    **header("validation_case"), "quantity": Text(VALIDATION_QUANTITIES),
    "reference": Struct({"src": Ref("bibliography"), "location": Text(), "digitised": Flag(),
                         "uncertainty_pct": Qty(lo=0, hi=10.0), "measured": Flag()},
                        required=("src", "location", "measured"), meta=False),
    "setup": Struct({
        "operation": Text(VALIDATION_OPERATIONS),
        "material": ByType(mapping=Struct({"db": Ref("material")}, meta=False), expected="{db: <material id>}"),
        "tool": MapOf(MEASURE), "holder": MapOf(MEASURE), "machine": MapOf(MEASURE), "coolant": MapOf(MEASURE),
        "workholding": MapOf(MEASURE), "stock": MapOf(MEASURE), "conditions": MapOf(MEASURE),
        "dynamics": MapOf(Any_(), nullable=True), "program": Text(nullable=True),
    }, required=("operation",), meta=False),
    "reference_values": Seq(Struct({"name": Text(), "value": Qty(lo=-1e30, hi=1e30, dimensionless=True),
                                    "unit": Text(allow_empty=True), "uncertainty": Qty(lo=0, dimensionless=True),
                                    "x": Qty(lo=-1e30, hi=1e30, dimensionless=True), "x_name": Text(),
                                    "x_unit": Text()}, required=("name", "value", "unit"), meta=False)),
    "reference_curves": Seq(Struct({"name": Text(), "x_name": Text(), "x_unit": Text(allow_empty=True), "y_name": Text(),
                                    "y_unit": Text(allow_empty=True),
                                    "points": Qty(shape=("rows", 2), lo=-1e30, hi=1e30, dimensionless=True),
                                    "uncertainty": Qty(lo=0, dimensionless=True)},
                                   required=("name", "x_name", "x_unit", "y_name", "y_unit", "points"), meta=False)),
}, required=("format", "version", "id", "quantity", "reference", "setup"), meta=False,
    checks=(lambda m, p, u, c: None if (get(m, "reference_values") or get(m, "reference_curves"))
            else c.add(p, "a validation case needs reference_values or reference_curves", "required"),))

# ----------------------------------------------------------------------------- bibliography, model params
def _bib_rules(m, p, u, ctx):
    kind = get(m, "type")
    if kind in BIB_DATED and not has(m, "year"):
        ctx.add(p, f"missing required key 'year' (a {kind} is identified by its year)", "required")
    if kind not in BIB_DATED and not (has(m, "year") or has(m, "accessed")):
        ctx.add(p, f"a {kind} needs year or accessed (the edition read)", "required")


BIB_ENTRY = Struct({
    "type": Text(BIB_TYPES, sourced=False), "authors": Text(), "editors": Text(), "title": Text(), "venue": Text(),
    "publisher": Text(), "edition": ByType(text=Text(), number=Count(lo=1)), "volume": ByType(text=Text(), number=Count()),
    "issue": ByType(text=Text(), number=Count()), "number": ByType(text=Text(), number=Count()),
    "year": Count(lo=1500, hi=2100, nullable=True), "doi": Text(pattern=r"10\.\d{4,9}/\S+"), "isbn": Text(),
    "issn": Text(), "url": Text(pattern=r"(https?|ftp)://\S+"), "accessed": Text(pattern=r"\d{4}-\d{2}-\d{2}"),
    "pages": ByType(text=Text(), number=Count()), "institution": Text(), "language": Text(), "original": Text(),
}, required=("type", "title"), checks=(_bib_rules,), meta=False)
BIBLIOGRAPHY = Struct({
    **header("bibliography", id_key=None),
    "entries": MapOf(BIB_ENTRY, key=Text(pattern=r"[a-z0-9][a-z0-9._-]*"), meta=False),
}, required=("format", "version", "entries"), meta=False)

PARAM_VALUE = ByType(number=Qty(lo=-1e300, hi=1e300, dimensionless=True), boolean=Flag(), text=Text(allow_empty=True),
                     seq=Any_(), mapping=Any_(), expected="a number, text, true/false or a list")


def _param_range_rule(m, p, u, ctx):
    value, rng = get(m, "value"), get(m, "range")
    if is_number(value) and isinstance(rng, list) and len(rng) == 2 and all(is_number(v) for v in rng) \
            and not rng[0] <= value <= rng[1]:
        ctx.add(p, f"value {value} lies outside its range {rng}", "rule")


MODEL_PARAM = Struct({"value": PARAM_VALUE, "unit": Text(allow_empty=True),
                      "range": Qty(shape="range", lo=-1e300, hi=1e300, dimensionless=True)},
                     required=("value", "unit"), checks=(_param_range_rule,))
MODEL = Struct({
    **header("model", id_key="model"), "description": Text(), "doc": Text(),
    "params": MapOf(MODEL_PARAM, key=Text(pattern=r"[A-Za-z_][A-Za-z0-9_]*"), min_len=1, meta=False),
}, required=("format", "version", "model", "params"))

SCHEMAS: dict[str, Node] = {
    "material": MATERIAL, "tool_series": TOOL_SERIES, "tool": TOOL_ITEM, "tool_material": TOOL_MATERIAL,
    "coating": COATING, "holder_series": HOLDER_SERIES, "holder": HOLDER_ITEM, "machine": MACHINE,
    "coolant": COOLANT, "workholding_series": WORKHOLDING_SERIES, "workholding": WORKHOLDING_ITEM,
    "controller": CONTROLLER, "validation_case": VALIDATION_CASE, "model": MODEL, "bibliography": BIBLIOGRAPHY,
}


def inline_schema(kind: str) -> Struct:
    """The schema of an object written inline in a project (no header, id optional, extends allowed)."""
    base = SCHEMAS[kind]
    fields = dict(base.fields)
    for key in ("format", "version"):
        fields[key] = base.fields.get(key, Any_())
    fields["extends"] = Ref(kind)
    if "id" in fields:
        fields["id"] = Text(pattern=ID_PATTERN, sourced=False)
    required = tuple(k for k in base.required if k not in ("format", "version", "id"))
    return Struct(fields, required=required, checks=base.checks, hints=base.hints, meta=base.meta)


def partial_schema(kind: str) -> Struct:
    """Schema of an extending object before the merge: known keys, nothing required but the header."""
    base = SCHEMAS[kind]
    fields = dict(base.fields)
    fields["extends"] = Ref(kind)
    required = tuple(k for k in ("format", "version", "id") if k in base.fields)
    return Struct(fields, required=required, hints=base.hints, meta=base.meta)


__all__ = ["SCHEMAS", "FORMATS", "KIND_OF_FORMAT", "CURRENT_VERSION", "FOLDERS", "KIND_OF_FOLDER", "SINGLE_KINDS",
           "SERIES_KINDS", "ITEM_KINDS", "USUI_RATE_UNITS", "inline_schema", "partial_schema", "chain_links",
           "TOOL_TYPES", "HOLDER_TYPES", "WORKHOLDING_TYPES", "MACHINE_TYPES", "DIALECTS", "TOOL_CLASSES", "FAMILIES",
           "POSE", "TYPE_GEOMETRY"]
