"""Harness for coolant_reach: what the coolant model does to the measured quantity.

Each case's jet (nozzle bore, distance, pressure, flow and direction as stated) is built as the coolant domain
builds jets (``cnctwin.coolant.jets.jet_spec`` from a nozzle outlet) and traced through a minimal occlusion
scene of the case's own geometry (``cnctwin.coolant.occlusion``: the turning insert, its holder and the bar;
the rotating end mill and the block) to the cutting zone: the reach and the jet's arrival. From there the
coolant acts on the cut through three mechanisms of ``cnctwin.coolant``:

* lubrication: the friction factor at that reach (``lubrication.friction``);
* convective cooling of the tool faces at the engagement: Martin's coefficient of the arriving jet with its
  speed component normal to the face (``engagement.face_h``) on the reached part of the fluted surface, in
  thermal's tool ladder (``cnctwin.thermal.toolmodel.ToolThermal``) over the pass, so the tool body under the
  contact warms dry and stays cool under the jet (milling: the contact temperature at that body
  temperature, averaged over the pass as the thermocouple averages it);
* hydraulic wedge: the jet's momentum along the rake face against the chip flow lifts the chip
  (``engagement.wedge_force_N``, ``contact_length_ratio``: Oxley's chip moment balance) and shortens the
  contact. The mechanics has no input for an imposed contact length, so the harness applies it where its
  effect enters: the rake friction force (friction factor times the contact ratio, the force scaling with the
  contact length at the interface stress) and the chip's natural up-curl (``mechanics.chipform`` with the
  shortened contact length).

Per case: the tool-work thermocouple in milling under dry / external HPC / rotary HPC (sultana-2023), and the
chip-breaking grade and cutting force against the jet direction (struzikiewicz-2023).
"""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np

from ...coolant import abi as CA
from ...coolant import engagement as E
from ...coolant import fluid as F
from ...coolant import lubrication as L
from ...coolant import occlusion as O
from ...coolant.jets import jet_spec
from ...coolant.protocols import bricks_from_occupancy, merge_bricks
from ...coolant.schema import settings as coolant_settings
from ...coolant.supply import LIQUID, Outlet
from ...db.models import params
from ...db.simap import SIMap
from ...mechanics import chipform as CF
from ...mechanics import contact as C
from ...mechanics import oxley
from ...thermal import props as TP
from ...thermal.schema import settings as thermal_settings
from ...thermal.toolmodel import ToolEnvironment, ToolThermal, tool_spec
from ...tooling.abi import PART_EDGE, PART_SHANK
from ..cases import from_si, to_si
from ..registry import CLASS, COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import common as H
from .chipform import HOLDER_INCL_DEG, HOLDER_RAKE_DEG, _Chips
from .contact import ContactModel

MILLING_FLUIDS = {"VG-68": "straight-oil", "emulsion": "emulsion-8pct"}
GRAVITY = np.array([0.0, 0.0, -9.80665])
ENCLOSURE_M = 0.5
VOXEL_M = 1.0e-3
CUT_LENGTH_M = 0.1          # documented assumption: the pass runs along the 100 mm block
STICKOUT_M = 0.04           # documented assumption: end-mill stickout (flute length + 13 mm)
LADDER_STEPS = 40           # steps of thermal's tool ladder over a pass


# ----------------------------------------------------------------------------------------- scene and jets
def _settings():
    return coolant_settings({})


def _outlet(jet_id: str, jet: E.NozzleJet, temp_K: float) -> Outlet:
    return Outlet(jet_id, jet_id, jet_id, "high_pressure", LIQUID, jet.flow_m3_s, jet.speed_m_s, jet.diameter_m,
                  jet.rho_kg_m3 * jet.flow_m3_s, jet.flow_m3_s, jet.rho_kg_m3, temp_K=temp_K)


def _empty_revolved() -> CA.Revolved:
    return CA.Revolved(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0), np.ones(0), np.zeros(0, np.int64),
                       np.zeros((0, 2)), np.zeros((0, 2), np.int64), np.zeros(0))


def _boxes(rows) -> CA.Boxes:
    n = len(rows)
    boxes = CA.Boxes(np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros(n, np.int64))
    for k, (centre, axes, half) in enumerate(rows):
        boxes.centre_M[k] = centre
        boxes.R_M_B[k] = axes
        boxes.half[k] = half
    return boxes


def _scene(revolved, rev_names, boxes, box_names, occupied, lo, props: F.Props, temp_K: float) -> O.Scene:
    st = _settings()
    bs = bricks_from_occupancy(lo, VOXEL_M, occupied, int(st.brick_sub))
    bricks = merge_bricks([(bs, np.eye(3), np.zeros(3))])
    air = F.air_props(temp_K)
    lo_box = np.full(3, -ENCLOSURE_M)
    prm = O.trace_params(GRAVITY, air.rho_kg_m3, air.mu_Pa_s, st, lo_box, -lo_box)
    hf = CA.empty_heightfield()._replace(R_M_S=CA.contiguous(np.eye(3)), o_M_S=CA.contiguous(np.zeros(3)))
    return O.Scene(revolved, rev_names, boxes, box_names, bricks, hf, [], prm, None, np.eye(3), np.zeros(3),
                   props.rho_kg_m3, props.nu_m2_s, air.rho_kg_m3, air.nu_m2_s, True)


def _arrive(scene: O.Scene, jet: E.NozzleJet, origin, direction, centre, radius: float, props: F.Props,
            temp_K: float) -> tuple[float, O.Arrival | None, dict]:
    """(reach, the jet's arrival at the zone, blocked share by occluder) of one jet at a cutting zone."""
    st = _settings()
    spec = jet_spec(_outlet("jet", jet, temp_K), np.asarray(origin, float), np.asarray(direction, float), props, None)
    zone = O.Zone(0, 0, 0, np.asarray(centre, float), float(radius), float(radius), True)
    res = O.run([spec], [zone], scene, st, 0, 0, 1)
    z = res.zones.get(0)
    if z is None:
        return 0.0, None, {}
    shadows = {k: v / z.aimed_kg_s for k, v in z.shadows.items()} if z.aimed_kg_s > 0.0 else {}
    return float(z.reach), z.arrivals.get("jet"), shadows


def _zone_radius(size_m: float) -> float:
    p = O.P()
    return float(p["zone_radius_factor"]) * max(size_m, float(p["zone_min_radius_m"]))


# ----------------------------------------------------------------------------------------- turning (jet direction)
def _turning_geometry(kappa: float, nose_m: float, ap_m: float, f_m: float):
    """Directions in the rake plane (z = 0, tool below) of a turning tool cutting at the origin (its corner): the
    feed along -x, the bar's axis along x at y = -(R - ap). Returns (corner bisector, chip flow): the chip flows
    normal to the chord of the engaged edge (Colwell 1954) from its end on the major edge at depth ap to the
    feed mark on the nose."""
    phi = math.pi - kappa                                    # major edge direction from +x
    e_major = np.array([math.cos(phi), math.sin(phi), 0.0])
    n_out = np.array([-math.sin(phi), math.cos(phi), 0.0])   # outward normal (toward the uncut material)
    centre = np.array([0.0, nose_m, 0.0])
    tangent = centre + nose_m * n_out
    s = (ap_m - tangent[1]) / e_major[1] if e_major[1] > 0.0 else 0.0
    a = tangent + max(s, 0.0) * e_major
    b = np.array([0.5 * f_m, nose_m - math.sqrt(max(nose_m * nose_m - 0.25 * f_m * f_m, 0.0)), 0.0])
    chord = b - a
    flow = np.array([-chord[1], chord[0], 0.0])
    bisector_angle = math.pi - (kappa + 0.5 * math.radians(80.0))   # C insert: 80 deg corner
    bisector = np.array([math.cos(bisector_angle), math.sin(bisector_angle), 0.0])
    if np.dot(flow, bisector) < 0.0:
        flow = -flow
    return bisector, flow / np.linalg.norm(flow)


def _turning_scene(case, kappa: float, ap: float, props: F.Props, temp_K: float, col) -> O.Scene:
    radius = 0.5 * float(case.si("stock", "diameter_mm"))
    p = params("thermal-tool")
    ic = float(p["default_insert_ic_m"])
    thick = float(p["default_insert_thickness_m"])
    section = 0.020                                           # PCLNR 2020: 20 x 20 mm shank (designation)
    bis_angle = math.pi - (kappa + 0.5 * math.radians(80.0))
    u = np.array([math.cos(bis_angle), math.sin(bis_angle), 0.0])
    v = np.array([-u[1], u[0], 0.0])
    axes = np.column_stack([u, v, [0.0, 0.0, 1.0]])
    corner_to_centre = 0.5 * ic / math.sin(math.radians(40.0))
    insert = (corner_to_centre * u + np.array([0.0, 0.0, -0.5 * thick]), axes,
              np.array([corner_to_centre, 0.25 * ic, 0.5 * thick]))
    holder = (np.array([0.5 * section, 0.0625, -thick - 0.5 * section]), np.eye(3),
              np.array([0.5 * section, 0.0625, 0.5 * section]))
    col.assume("occlusion scene", f"insert {ic * 1e3:.1f} mm IC x {thick * 1e3:.2f} mm (box below the rake face), "
               f"holder shank 20 x 20 mm, bar D {2e3 * radius:.0f} mm stepped by ap", "the case's insert, holder "
               "designation and stock; insert size from thermal's defaults (CNGG 1204: 12.7 x 4.76 mm)",
               "case, db/models/thermal-tool.yaml")
    lo = np.array([-0.015, -(2.0 * radius) - 0.002, -radius - 0.002])
    shape = (int(0.030 / VOXEL_M), int((2.0 * radius + 0.004) / VOXEL_M), int((2.0 * radius + 0.004) / VOXEL_M))
    idx = [lo[k] + (np.arange(shape[k]) + 0.5) * VOXEL_M for k in range(3)]
    x, y, z = np.meshgrid(*idx, indexing="ij")
    axis_y = -(radius - ap)
    r_here = np.where(x < 0.0, radius, radius - ap)
    occupied = (y - axis_y) ** 2 + z ** 2 <= r_here ** 2
    return _scene(_empty_revolved(), [], _boxes([insert, holder]), ["insert", "holder"], occupied, lo, props, temp_K)


def _grade(cch: float) -> str:
    """The case's grade from the breaking index: + up to 5 mm, 0 5-20 mm, - over 20 mm (Cch = 0.01 / mm)."""
    mm = cch * 100.0
    return "+" if mm <= 5.0 else ("0" if mm <= 20.0 else "-")


def _form(chips: _Chips, f_m: float, ap_m: float, v_m_s: float, friction: float, contact_ratio: float):
    """The chip form of the harness's chip-form pieces with the jet-shortened contact (natural curl)."""
    h = f_m * math.sin(chips.kappa)
    width = ap_m / math.sin(chips.kappa)
    h_min = C.min_chip_thickness(chips.work, chips.r_e)
    if h <= h_min:
        return None
    res = oxley.solve(chips.work, h - h_min, v_m_s, chips.rake, work_temp_K=chips.amb, tool=chips.tool,
                      tool_temp_K=chips.amb, friction_factor=friction, contact_width_m=width)
    if res.status not in ("oxley", "merchant"):
        return None
    state = CF.ChipState.from_orthogonal(res)
    state = dataclasses.replace(state, contact_length_m=state.contact_length_m * contact_ratio)
    cond = CF.ChipConditions(h_m=h - h_min, width_m=width, speed_m_s=v_m_s, rake_rad=chips.rake, incl_rad=chips.incl,
                             cut_length_m=None, feed_m=f_m, depth_m=ap_m, nose_radius_m=float(chips.nose),
                             process="turning")
    return CF.chip_form(chips.work, chips.tool, cond, state)


def _stated_rake(case, chips: _Chips, col) -> None:
    """The working rake from the case's rake land when the insert is not in the database: the land of the insert's
    chip-breaker profile tilted by the negative holder (the chip-form helper's -6 deg holder rake alone treats the
    insert's top as flat and ignores the stated land)."""
    land = case.leaf("tool", "rake_land_deg")
    if chips.tool.chip_breaker is not None or land is None or not isinstance(land.value, (int, float)):
        return
    rake_deg = float(land.value) + HOLDER_RAKE_DEG
    chips.rake = rake_deg * H.DEG
    col.assumptions[:] = [a for a in col.assumptions if a.input != "rake and inclination"]
    col.assume("rake", f"{rake_deg:+g} deg working", f"the case's {float(land.value):g} deg rake land of the insert "
               f"profile on a negative holder tilted {HOLDER_RAKE_DEG:g} deg (the holder's tilt not stated: negative ISO "
               "holder)", "case")
    col.assume("inclination", f"{HOLDER_INCL_DEG:g} deg", "not stated: a negative ISO holder for a negative insert", "est")


def _turning_jet(case, db, col) -> HarnessOutput:
    chips = _Chips(db, case, col, coolant=True)
    _stated_rake(case, chips, col)
    work = chips.work
    vc = case.si("conditions", "cutting_speed_m_min")
    ap = case.si("conditions", "ap_mm")
    cool = case.section("coolant")
    fluid_text = str(cool.get("fluid") or "")
    fluid_id = "emulsion-8pct"
    fluid = F.fluid_from_db(db.coolant(fluid_id))
    props = fluid.props(chips.amb)
    pressure = float(case.si("coolant", "pressure_bar"))
    bore = float(case.si("coolant", "nozzle_diameter_mm"))
    distance = float(case.si("coolant", "nozzle_distance_mm"))
    flow = case.si("coolant", "flow_l_min")
    jet = E.nozzle_jet(pressure, bore, props, float(flow) if isinstance(flow, (int, float)) else None)
    col.note(f"Coolant '{fluid_text}' as {fluid_id}; jet {jet.speed_m_s:.0f} m/s, {jet.diameter_m * 1e3:.2f} mm "
             f"(Bernoulli at {pressure / 1e5:.0f} bar, the stated flow by continuity), {distance * 1e3:.0f} mm from the "
             "corner, traced through the case's insert, holder and bar.")
    col.assume("jet direction", "alpha = elevation above the rake face; beta = azimuth in the rake plane from the "
               "insert's corner bisector, positive toward the major cutting edge", "the case defines beta from a "
               "reference line drawn only in its Fig. 1b; the bisector is the usual reference and the sign the only "
               "one that keeps the 20 mm nozzle outside the bar at alpha 30 deg", "est")
    col.assume("chip flow direction", "normal to the chord of the engaged edge (Colwell 1954)", "nose radius, "
               "entering angle, ap and feed as stated", "case")
    scene = _turning_scene(case, chips.kappa, ap, props, chips.amb, col)
    h_min = C.min_chip_thickness(work, chips.r_e)
    radius = _zone_radius(0.5 * ap)
    tests = {}
    preds = []
    models = {}
    test_angles = _test_angles(case)
    for entry in case.reference_values:
        name = str(entry["name"])
        f = float(re.search(r"_f([0-9.]+)", name).group(1)) * 1e-3
        test = _test_of(entry, name, test_angles)
        if test not in tests:
            alpha, beta = test_angles[test]
            tests[test] = _turning_test(chips, scene, jet, props, fluid, f, ap, vc, alpha, beta, distance, radius,
                                        models, col)
        t = tests[test]
        unit = str(entry.get("unit") or "")
        if name.startswith("Cch"):
            ref_grade = re.search(r"grade ([+0-])", str(entry.get("note") or ""))
            reference = ref_grade.group(1) if ref_grade else _grade(float(entry["value"]))
            form = t["form"]
            if form is None:
                preds.append(no_prediction("no chip formed", kind=CLASS, reference_class=reference))
                continue
            cch = min(1.0, 0.01 * (form.length_m + 2.0 * form.curl_radius_m) * 1e3) if math.isfinite(form.length_m) \
                else 1.0
            preds.append(Prediction(_grade(cch), COMPARED, f"f {f * 1e3:g}: {form.form_name}, Cch {cch:.2f}; "
                                    f"{t['note']}", kind=CLASS, reference_class=reference,
                                    extra={"Cch": cch, **t["extra"]}))
        else:
            preds.append(Prediction(from_si(t["force_N"], unit), COMPARED, f"f {f * 1e3:g}: {t['note']}",
                                    extra=t["extra"]))
    settings = {"fluid": fluid_id, "jet_speed_m_s": jet.speed_m_s, "jet_diameter_mm": jet.diameter_m * 1e3,
                "rake_deg": math.degrees(chips.rake), "edge_radius_um": chips.r_e * 1e6, "h_min_um": h_min * 1e6,
                "zone_radius_mm": radius * 1e3}
    return HarnessOutput(preds, [[] for _ in case.reference_curves],
                         ["cnctwin.coolant.occlusion (case scene: insert, holder, bar)",
                          "cnctwin.coolant.lubrication.friction", "cnctwin.coolant.engagement (hydraulic wedge)",
                          "cnctwin.mechanics.chipform.chip_form", "cnctwin.mechanics.coefficients (cutting force)"],
                         settings, col.assumptions, col.provenance, col.notes, chips.amb)


def _test_angles(case) -> dict:
    """test number -> (alpha, beta) in rad from the notes of the force values ('alpha 30, beta 0')."""
    out = {}
    for entry in case.reference_values:
        name = str(entry["name"])
        if name.startswith("Fc"):
            m = re.search(r"alpha ([0-9]+), beta ([0-9]+)", str(entry.get("note") or ""))
            if m:
                out[int(entry["x"])] = (math.radians(float(m.group(1))), math.radians(float(m.group(2))))
    return out


def _test_of(entry, name: str, angles: dict) -> int:
    if name.startswith("Fc"):
        return int(entry["x"])
    m = re.search(r"test ([0-9]+)", str(entry.get("note") or ""))
    if m:
        return int(m.group(1))
    alpha = float(re.search(r"alpha([0-9]+)", name).group(1))
    beta = float(entry["x"])
    for test, (a, b) in sorted(angles.items()):
        if abs(math.degrees(a) - alpha) < 0.5 and abs(math.degrees(b) - beta) < 0.5:
            return test
    raise ValueError(f"no test for {name}")


def _turning_test(chips: _Chips, scene, jet, props, fluid, f, ap, vc, alpha, beta, distance, radius, models,
                  col) -> dict:
    bisector, chip_flow = _turning_geometry(chips.kappa, float(chips.nose), ap, f)
    azimuth = math.atan2(bisector[1], bisector[0]) + beta          # positive toward the major edge
    toward_nozzle = np.array([math.cos(alpha) * math.cos(azimuth), math.cos(alpha) * math.sin(azimuth),
                              math.sin(alpha)])
    origin = distance * toward_nozzle
    direction = -toward_nozzle
    reach, arrival, shadows = _arrive(scene, jet, origin, direction, np.zeros(3), radius, props, chips.amb)
    h = f * math.sin(chips.kappa)
    width = ap / math.sin(chips.kappa)
    # the contact's own uncut thickness, as the coolant domain passes it (the capillary access depends on v_c h)
    lub = L.friction(fluid, props, reach, chips.tool.tool_class, chips.work.family, vc, h, None, width,
                     fluid.is_mql, 0.0)

    def model(factor):
        key = round(factor, 9)
        if key not in models:
            models[key] = ContactModel(chips.work, chips.tool, rake_rad=chips.rake, clearance_rad=6.0 * H.DEG,
                                       edge_radius_m=chips.r_e, work_temp_K=chips.amb, speeds_m_s=[vc],
                                       h_values_m=[h], incl_rad=chips.incl, friction_factor=factor)
        return models[key]

    st = model(lub.factor).state(h, vc, chips.amb, width)
    normal = st.force_t_N * math.cos(chips.rake) - st.force_r_N * math.sin(chips.rake)
    wedge = E.wedge_force_N(jet, direction, [0.0, 0.0, 1.0], chip_flow, reach)
    ratio = E.contact_length_ratio(wedge, normal)
    force = model(lub.factor * ratio).state(h, vc, chips.amb, width).force_t_N if ratio < 1.0 else st.force_t_N
    form = _form(chips, f, ap, vc, lub.factor, ratio)
    blocked = ", ".join(f"{k} {v:.0%}" for k, v in sorted(shadows.items()) if v > 0.005)
    note = (f"alpha {math.degrees(alpha):.0f} beta {math.degrees(beta):.0f}: reach {reach:.2f}"
            + (f" (blocked: {blocked})" if blocked else "") + f", friction factor {lub.factor:.2f}, wedge "
            f"{wedge:.2f} N on N {normal:.0f} N, contact x {ratio:.3f}")
    return {"force_N": force, "form": form, "note": note,
            "extra": {"reach": reach, "friction_factor": lub.factor, "wedge_N": wedge, "contact_ratio": ratio}}


# ----------------------------------------------------------------------------------------- milling (tool temperature)
def _endmill_scene(d: float, flute: float, overall: float, ap: float, ae: float, props, temp_K: float) -> O.Scene:
    """The rotating cutter (axis +z, tip at the origin) beside the block it cuts (x <= ae - d/2, z < ap)."""
    dz = 2.5e-4
    n = int(round(overall / dz)) + 1
    rev = CA.Revolved(np.zeros((1, 3)), np.eye(3)[None, :, :].copy(), np.zeros(1), np.full(1, dz),
                      np.full(1, n, np.int64), np.full((1, n), 0.5 * d), np.zeros((1, n), np.int64),
                      np.full(1, 0.5 * d))
    rev.part[0, :] = np.where(np.arange(n) * dz < flute, PART_EDGE, PART_SHANK)
    lo = np.array([-0.04, -0.03, -0.02])
    shape = (int(0.08 / VOXEL_M), int(0.06 / VOXEL_M), int(0.02 / VOXEL_M) + int(ap / VOXEL_M))
    idx = [lo[k] + (np.arange(shape[k]) + 0.5) * VOXEL_M for k in range(3)]
    x, y, z = np.meshgrid(*idx, indexing="ij")
    occupied = (z < ap) & (x < ae - 0.5 * d) & ~(x ** 2 + y ** 2 <= (0.5 * d) ** 2)   # block on the -x side
    return _scene(rev, [(0, 1)], _boxes([]), [], occupied, lo, props, temp_K)


def _tool_ladder(case, tm: str, amb: float, col) -> ToolThermal:
    d_mm = float(case.value("tool", "diameter_mm"))
    raw = {"type": "square_endmill", "tool_material": {"id": tm},
           "geometry": {"diameter_mm": d_mm, "flutes": int(case.value("tool", "flutes") or 4),
                        "flute_length_mm": float(case.value("tool", "flute_length_mm") or 2.0 * d_mm),
                        "overall_length_mm": float(case.value("tool", "overall_length_mm") or 6.0 * d_mm),
                        "shank": {"diameter_mm": d_mm}}}
    spec = tool_spec({"stickout_m": STICKOUT_M}, SIMap(raw), None)
    st = thermal_settings({})
    tp = params("thermal-tool")
    return ToolThermal(0, spec, TP.tool_material_table_by_id(tm), TP.material_table_by_id(str(tp["holder_material"])),
                       float(st.tool_cell_m), amb, 0.0)


def _milling_temperature(case, db, col) -> HarnessOutput:
    work, raw = H.work_material(db, case, col)
    tool, tm, cid = H.tool_side(db, case, col, kind="square_endmill")
    amb = H.ambient(case, col)
    H.material_provenance(raw, col, friction_entry=work.friction_for(tool.tool_class, tool.coating))
    d = case.si("tool", "diameter_mm")
    ae = case.si("conditions", "ae_mm")
    ap = case.si("conditions", "ap_mm")
    flutes = int(case.value("tool", "flutes") or 4)
    flute_len = float(case.si("tool", "flute_length_mm") or 2.0 * d)
    overall = float(case.si("tool", "overall_length_mm") or 6.0 * d)
    helix = (case.value("tool", "helix_deg") or 0.0) * H.DEG
    rake = H.rake_rad(case, None, col)
    clearance = H.clearance_rad(case, col)
    r_e = H.edge_radius_m(case, None, col, "solid")
    phi_e = math.acos(max(min(1.0 - 2.0 * ae / d, 1.0), -1.0))
    width = min(0.5 * d * phi_e / max(math.sin(helix), 1e-6), ap / max(math.cos(helix), 1e-6))
    col.assume("chip thickness", "mean over the engagement, fz (1 - cos phi_e) / phi_e", "a thermocouple average over "
               "the cut", "est")
    col.assume("cut length", f"{CUT_LENGTH_M * 1e3:.0f} mm", "not stated: one pass along the 100 mm block", "est")
    col.assume("end-mill stickout", f"{STICKOUT_M * 1e3:.0f} mm", "not stated: the 27 mm flutes and 13 mm of shank "
               "out of the holder; holder from thermal's defaults", "est")
    entries = case.reference_values
    speeds = sorted({to_si(float(re.search(r"vc([0-9]+)", str(e["name"])).group(1)), "m/min") for e in entries})
    h_vals = sorted({to_si(e["x"], e.get("x_unit")) * (1.0 - math.cos(phi_e)) / phi_e for e in entries})
    cool = case.section("coolant")
    fluid_text = str(cool.get("fluid") or "")
    fluid_id = next((v for k, v in MILLING_FLUIDS.items() if k.lower() in fluid_text.lower()), "straight-oil")
    fluid = F.fluid_from_db(db.coolant(fluid_id))
    props = fluid.props(amb)
    col.assume(f"coolant fluid {fluid_id}", "database entry", "the case's fluid is not in the database: the closest "
               "database fluid", "db/coolants")
    jets = _milling_jets(case, props, d, flute_len, overall, ap, ae, amb, col)
    models = {}
    preds = []
    cache = {}
    for entry in entries:
        name = str(entry["name"])
        mode = re.search(r"T_([a-z]+)_vc", name).group(1)
        vc = to_si(float(re.search(r"vc([0-9]+)", name).group(1)), "m/min")
        fz = to_si(entry["x"], entry.get("x_unit"))
        h_m = fz * (1.0 - math.cos(phi_e)) / phi_e
        jet = jets.get(mode)
        reach = jet["reach"] if jet is not None else 0.0
        if jet is None:
            factor = 1.0
        else:
            # the mean uncut thickness of the engagement, as the coolant domain passes the contact's mean h
            factor = L.friction(fluid, props, reach, tool.tool_class, work.family, vc, h_m, None, width, fluid.is_mql,
                                0.0).factor
        key = round(factor, 9)
        if key not in models:
            models[key] = ContactModel(work, tool, rake_rad=rake, clearance_rad=clearance, edge_radius_m=r_e,
                                       work_temp_K=amb, speeds_m_s=speeds, h_values_m=h_vals, incl_rad=helix,
                                       friction_factor=factor)
        omega = vc / (0.5 * d)
        contact = phi_e / omega
        ckey = (mode, round(vc, 9), round(h_m, 12))
        if ckey not in cache:
            cache[ckey] = _pass(models[key], _tool_ladder(case, tm, amb, col), h_m, vc, omega, width, contact,
                                flutes * phi_e / (2.0 * math.pi), fz * flutes * omega / (2.0 * math.pi), ap, amb,
                                jet, props)
        mean_K, last_st, body_K = cache[ckey]
        unit = str(entry.get("unit") or "degC")
        h_note = f"jet h {jet['h']:.0f} W/m2K on {jet['coverage']:.0%} of the flutes, reach {reach:.2f}" if jet else \
            "dry: air only"
        preds.append(Prediction(from_si(mean_K, unit), COMPARED,
                                f"{mode}: friction factor {factor:.2f}, h_mean {h_m * 1e6:.1f} um, contact "
                                f"{contact * 1e3:.1f} ms; {h_note}; tool body under the contact "
                                f"{body_K - 273.15:.0f} degC at the end of the pass; rake {last_st.T_interface_K - 273.15:.0f}"
                                f" degC, flank {last_st.flank_temp_K - 273.15:.0f} degC",
                                extra={"friction_factor": factor, "mode": mode, "tool_body_end_degC": body_K - 273.15,
                                       "reach": reach}))
    settings = {"tool_material": tm, "coating": cid, "fluid": fluid_id, "engagement_deg": math.degrees(phi_e),
                "edge_radius_um": r_e * 1e6, "helix_deg": math.degrees(helix), "cut_length_mm": CUT_LENGTH_M * 1e3,
                **{f"{m}_jet_h_W_m2K": j["h"] for m, j in sorted(jets.items())}}
    return HarnessOutput(preds, [[] for _ in case.reference_curves],
                         ["cnctwin.coolant.occlusion (case scene: rotating cutter, block)",
                          "cnctwin.coolant.engagement.face_h (jet cooling)", "cnctwin.coolant.lubrication.friction",
                          "cnctwin.thermal.toolmodel.ToolThermal (tool body over the pass)",
                          "cnctwin.mechanics.coefficients (contact mean temperature)"],
                         settings, col.assumptions, col.provenance, col.notes, amb)


def _milling_jets(case, props, d, flute_len, overall, ap, ae, amb, col) -> dict:
    """Per coolant mode: the jet at the cutter (reach through the case scene, h on the reached flutes)."""
    out = {}
    scene = _endmill_scene(d, flute_len, overall, ap, ae, props, amb)
    centre = np.array([-0.5 * d, 0.0, 0.5 * ap])              # the engagement: the periphery facing the block
    for mode in ("hpc", "rhpc"):
        if case.leaf("coolant", f"{mode}_pressure_bar") is None:
            continue
        pressure = float(case.si("coolant", f"{mode}_pressure_bar"))
        n = int(case.value("coolant", f"{mode}_nozzles") or 1)
        bore = float(case.si("coolant", f"{mode}_nozzle_diameter_mm"))
        flow = float(case.si("coolant", f"{mode}_flow_l_min")) / n
        distance = float(case.si("coolant", f"{mode}_nozzle_distance_mm"))
        jet = E.nozzle_jet(pressure, bore, props, flow)
        r_avg = max(2.5 * jet.diameter_m, 0.5 * d)
        if mode == "hpc":
            # one fixed external jet aimed at the engagement, normal to the tool axis, from the cut's open side
            toward = np.array([-1.0, -1.0, 0.0]) / math.sqrt(2.0)
            reach, arrival, _ = _arrive(scene, jet, centre - distance * toward, toward, centre,
                                        _zone_radius(max(0.5 * d, 0.5 * ap, 0.5 * ae)), props, amb)
            col.assume("HPC jet direction", "normal to the tool axis at mid-depth, 45 deg to the feed, from the cut's "
                       "open side", "the case shows the jet only in a schematic (Fig. 3)", "est")
            coverage = min(1.0, 2.0 * r_avg / flute_len)
        else:
            # the rotary applicator's nozzles turn with the tool, each aimed at a rake face: no occluder between
            reach = 1.0
            coverage = min(1.0, n * math.pi * r_avg * r_avg / (math.pi * d * flute_len))
            col.assume("rotary applicator", f"{n} jets turning with the tool, each onto a rake face at normal incidence",
                       "the case: nozzles at 90 deg spacing aimed at the rake faces", "case")
        h = E.face_h(jet, distance, 0.5 * math.pi, r_avg, props, 1.0)
        out[mode] = {"h": h, "reach": reach, "coverage": coverage, "jet": jet}
        col.note(f"{mode}: {n} jet(s) of {jet.speed_m_s:.0f} m/s, {jet.diameter_m * 1e3:.2f} mm (Bernoulli at "
                 f"{pressure / 1e5:.0f} bar, the stated flow by continuity), Martin h {h:.0f} W/m2K over "
                 f"{coverage:.0%} of the flutes, reach {reach:.2f}.")
    return out


def _pass(model: ContactModel, ladder: ToolThermal, h_m, vc, omega, width, contact, duty, feed_m_s, ap, amb, jet,
          props) -> tuple[float, object, float]:
    """Mean contact temperature over the pass with the tool body warming under it (thermal's ladder)."""
    duration = CUT_LENGTH_M / max(feed_m_s, 1e-9)
    dt = duration / LADDER_STEPS
    tol = float(thermal_settings({}).energy_tol)
    env = ToolEnvironment(in_spindle=True, omega_rad_s=omega, T_air_K=amb, T_end_K=amb,
                          coolant_on=jet is not None, T_fluid_K=amb,
                          h_coolant_W_m2K=jet["h"] if jet is not None else 0.0,
                          reach=(jet["coverage"] * jet["reach"]) if jet is not None else 0.0)
    temps = []
    body = amb
    st = None
    for _ in range(LADDER_STEPS):
        st = model.state(h_m, vc, body, width, contact)
        temps.append(st.contact_mean_K)
        ladder.deposit_uniform(ap, st.tool_W * duty * dt)
        ladder.step(dt, env, tol)
        body = float(ladder.body_temp_at(np.array([0.5 * ap]))[0])
    st = model.state(h_m, vc, body, width, contact)
    temps.append(st.contact_mean_K)
    mean = 0.5 * (temps[0] + temps[-1]) / LADDER_STEPS + sum(temps[1:-1]) / LADDER_STEPS
    return mean, st, body


@harness("coolant_reach", models=("cnctwin.coolant.occlusion", "cnctwin.coolant.engagement",
                                  "cnctwin.coolant.lubrication", "cnctwin.thermal.toolmodel",
                                  "cnctwin.mechanics.coefficients", "cnctwin.mechanics.chipform"),
         description="The case's jet traced through a minimal scene of its own geometry to the cut (cnctwin.coolant: "
                     "reach and arrival), then the coolant's three actions on the measured quantity: lubrication "
                     "(friction factor), convective cooling of the tool faces at the engagement (Martin, normal speed "
                     "component) in thermal's tool ladder over the pass, and the hydraulic wedge of a jet running "
                     "against the chip flow (shorter contact: friction force and natural chip curl).")
def coolant_reach(case, db) -> HarnessOutput:
    col = H.Collector()
    if case.operation in ("peripheral_milling", "slotting", "face_milling"):
        return _milling_temperature(case, db, col)
    return _turning_jet(case, db, col)


@discussion("coolant_reach")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        errs = [c.error_rel for c in r.comparisons if c.status == COMPARED and c.kind != CLASS and c.error_rel is not None]
        classes = [c for c in r.comparisons if c.kind == CLASS and c.counted]
        parts = []
        if errs:
            parts.append(f"scalar values: mean signed error {sum(errs) / len(errs) * 100:+.0f} %")
        if classes:
            hit = sum(1 for c in classes if c.error_rel == 0.0)
            parts.append(f"classes: {hit} of {len(classes)} match")
        out[r.id] = "; ".join(parts) + (". The jet reaches the cut in the case's scene; its lubrication, the cooling of "
                                        "the tool body and the hydraulic wedge are in the prediction. The remaining "
                                        "miss sits in the dry level of the mechanics (contact temperature, cutting "
                                        "force) and, for the chip grades, in the jet's push on the chip's back and the "
                                        "chip-breaker geometry, which are not modelled. The flank heat no longer "
                                        "offsets the jet: the coefficient pipeline's flank share is computed for the "
                                        "heat it is applied to (the friction on the wear land and the edge arc and the "
                                        "edge's plastic work) over the contact it is deposited on, and a cooled tool "
                                        "side of that contact runs cooler, not hotter (docs/models/mechanics.md 6); "
                                        "the jet moves the contact means little because the flank contact of a "
                                        "sharp, unworn edge is short against the rake contact.")
    out["summary"] = (H.budget_summary(q) + "The coolant now acts through the reach traced in each case's scene, the "
                      "lubrication friction factor, the convective cooling of the tool faces at the engagement and the "
                      "hydraulic wedge of a jet that runs under the chip; the dry levels of the mechanics set most of "
                      "the remaining error.")
    return out


__all__ = ["coolant_reach"]
