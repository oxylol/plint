"""Harnesses for chip_temperature and tool_temperature (cnctwin.mechanics + cnctwin.thermal).

What each experiment measured decides what is simulated:

* tempering colours of the chip (chip_temperature): the chip's oxide film is a time-temperature
  integral, read on a tempering-colour table. Simulated: Oxley's mean chip temperature during the
  contact (``oxley.solve``), then air cooling, through ``cnctwin.mechanics.colour`` (contact hold +
  exponential cooling -> Arrhenius exposure -> the equivalent temperature of the 30-min colour table),
  capped at the top row of the material's colour table as the reading was;
* tool-work (natural) thermocouple: the mean temperature of the whole tool-chip and tool-work
  contact, the rake contact (``group_coefficients`` T_interface: Oxley's rake-contact mean, partition at
  the tool body temperature) and the flank/edge contact (Blok's tool-side mean, heat.tool_side_resistance)
  weighted by their lengths; at the end of the cut for turning, at the maximum chip thickness of the
  engagement with the engagement time as contact time for milling (peak readings);
* IR window or spot, embedded thermocouple: the tool temperature at the measured location from
  thermal's tool model (``toolheat``: insert ladder + local edge field), at the end of the cut or as the
  time mean the source reports.

The sensor geometry of each case is in :data:`SENSORS` with the case text it comes from.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ...mechanics import colour as COL
from ...mechanics import contact as C
from ...mechanics import oxley
from ...thermal import convection as CV
from ..cases import from_si, to_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import assumptions as AS
from . import common as H
from .contact import ContactModel
from .toolheat import (InsertThermal, disc_mean_on_rake, insert_setup, temperature_at_point, wedge_factor,
                       window_mean)

MILLING_OPS = ("peripheral_milling", "slotting", "face_milling", "ball_end_milling")


# ------------------------------------------------------------------------------------------ chip temperature
@harness("chip_temperature", models=("cnctwin.mechanics.oxley", "cnctwin.mechanics.colour", "cnctwin.thermal.convection"),
         description="Oxley's mean chip temperature per test (t1 = h - h_min, tool at ambient), its oxide exposure "
                     "during the contact and the air cooling afterwards (cnctwin.mechanics.colour), read as the "
                     "equivalent temperature of the material's tempering-colour table and capped at its top row, "
                     "as the experimenters' reading was.")
def chip_temperature(case, db) -> HarnessOutput:
    col = H.Collector()
    work, raw = H.work_material(db, case, col)
    tool, tm, cid = H.tool_side(db, case, col)
    amb = H.ambient(case, col)
    rows = H.test_rows(case)
    friction = work.friction_for(tool.tool_class, tool.coating)
    H.material_provenance(raw, col, friction_entry=friction, extra=("chip.oxide_colours",))
    table = work.oxide_table or ()
    top_K = table[-1].temp_K if table else None
    q_over_r = COL.activation_temperature(work)
    eps = CV.emissivity(work.family)
    col.assume("chip cooling after the contact", "free convection + radiation in still air",
               f"cnctwin.thermal.convection: horizontal cylinder of the {AS.CHIP_COOLING_LENGTH}, emissivity "
               f"{eps:g} ({work.family})", "est")
    preds = []
    cache: dict = {}
    for entry in case.reference_values:
        row = H.find_test(rows, entry, ("vc_m_min",))
        if row is None:
            raise ValueError(f"{case.id}: no test matches {entry.get('name')!r}")
        if row.index not in cache:
            h = H.uncut_thickness_m(row)
            v = H.speed_m_s(row)
            rake = H.rake_rad(case, row, col)
            r_e = H.edge_radius_m(case, row, col, "insert")
            width = H.width_m(case, row) or 1e-3
            h_min = C.min_chip_thickness(work, r_e)
            if h <= h_min:
                cache[row.index] = None
            else:
                res = oxley.solve(work, h - h_min, v, rake, work_temp_K=amb, tool=tool, tool_temp_K=amb,
                                  contact_width_m=width)
                t2 = res.chip_thickness_m
                contact_time = res.contact_length_m / res.chip_speed_m_s if res.chip_speed_m_s > 0 else 0.0
                h_air = float(CV.h_horizontal_cylinder(res.T_chip_K, amb, width)) + \
                    float(CV.h_radiation(eps, res.T_chip_K, amb))
                _, name, exposure = COL.emitted_chip_colour(work, res.T_chip_K, contact_time, t2, h_air, amb)
                teq = COL.equivalent_temperature(exposure, q_over_r)
                cache[row.index] = dict(T_chip_K=res.T_chip_K, T_eq_K=teq, colour=name, h_air=h_air,
                                        contact_time_s=contact_time, v=v, r_e=r_e)
        st = cache[row.index]
        unit = str(entry.get("unit") or "degC")
        if st is None:
            preds.append(no_prediction("no chip: h <= h_min (minimum chip thickness)"))
            continue
        teq = st["T_eq_K"]
        if teq is None:
            preds.append(no_prediction(f"no oxide colour predicted (bare chip); mean chip temperature "
                                       f"{st['T_chip_K'] - 273.15:.0f} degC"))
            continue
        reading = min(teq, top_K) if top_K is not None else teq
        capped = top_K is not None and teq >= top_K
        preds.append(Prediction(from_si(reading, unit), COMPARED,
                                f"vc {st['v'] * 60:.0f} m/min: colour '{st['colour']}', equivalent "
                                f"{'>= table top' if capped else f'{teq - 273.15:.0f} degC'}; chip mean "
                                f"{st['T_chip_K'] - 273.15:.0f} degC",
                                extra={"T_chip_mean_degC": st["T_chip_K"] - 273.15,
                                       "T_equivalent_degC": (teq - 273.15) if math.isfinite(teq) else None,
                                       "h_air_W_m2K": st["h_air"], "contact_time_s": st["contact_time_s"]}))
    settings = {"tool_material": tm, "coating": cid, "oxide_activation_K": q_over_r,
                "colour_table_top_degC": (top_K - 273.15) if top_K else None, "mu_sliding": friction.mu}
    return HarnessOutput(preds, [[] for _ in case.reference_curves],
                         ["cnctwin.mechanics.oxley.solve", "cnctwin.mechanics.colour.emitted_chip_colour",
                          "cnctwin.mechanics.colour.equivalent_temperature", "cnctwin.thermal.convection"],
                         settings, col.assumptions, col.provenance, col.notes, amb)


# ------------------------------------------------------------------------------------------ tool temperature
@dataclass(frozen=True)
class Sensor:
    kind: str                       # interface | interface_peak_milling | side_window | rake_spot | point
    x_m: float = 0.0                # along the rake face from the edge
    z_m: float = 0.0                # below the rake face
    extent_x_m: float = 0.0         # window length (side_window) / spot radius (rake_spot)
    extent_z_m: float = 0.0
    reading: str = "end"            # end | time_mean | mean_at
    times_s: tuple = ()
    source: str = ""                # the case text the geometry comes from


SENSORS: dict[str, dict[str, Sensor]] = {
    "tooltemp-orth-ti64-loschner2025-ntc": {
        "*": Sensor("interface", source="natural (tool-work) thermocouple: contact-area mean interface temperature "
                                         "(case notes)")},
    "tooltemp-orth-ti64-slusarczyk2025-ir": {
        "*": Sensor("side_window", extent_x_m=AS.IR_WINDOW_ALONG_RAKE.value, extent_z_m=AS.IR_WINDOW_DOWN_FLANK.value,
                    source="'mean apparent temperature of a 2 x 1 mm area on the tool, seen from the side ... next to "
                           "the chip-tool contact' (case notes)")},
    "tooltemp-thinwall-al7050-zawada2025-ntc": {
        "*": Sensor("interface_peak_milling", source="natural thermocouple, mean of ten peak values: peak contact "
                                                      "temperature of the engagement (case notes)")},
    "tooltemp-turn-aisi1045-guimaraes2023-embedded-tc": {
        "*": Sensor("point", x_m=2.9e-3, z_m=AS.EMBEDDED_TC_DEPTH.value, reading="end",
                    source="K-type junction in the rake face 2.9 mm from the cutting edge, 0.2-0.4 mm deep "
                           "(setup.tool.thermocouple, case notes)")},
    "tooltemp-turn-aisi4140-kus2015-ir-tc": {
        "T_IR_": Sensor("rake_spot", x_m=1.0e-3, extent_x_m=0.75e-3, reading="time_mean",
                        source="pyrometer spot 1.5 mm on the rake face 1 mm from the edge beside the chip, readings "
                               "averaged over the cut (setup.conditions.ir_spot)"),
        "T_TC_transient_": Sensor("point", x_m=1.0e-3, z_m=1.0e-3, reading="end",
                                  source="K-type, 1 mm from the flank face and 1 mm below the rake face at the corner "
                                         "(setup.conditions.thermocouple); readings at the cutting times of Fig. 10"),
        "T_TC_": Sensor("point", x_m=1.0e-3, z_m=1.0e-3, reading="mean_at", times_s=(5.0, 10.0, 15.0, 20.0),
                        source="as the transient curves; Table 5 values are the mean of the readings over 5-20 s "
                               "(case notes)")},
}


def _sensor(case, name: str) -> Sensor:
    table = SENSORS.get(case.id)
    if table is None:
        raise ValueError(f"{case.id}: no sensor geometry for this case in temperature.SENSORS (add one: see "
                         "docs/validation.md)")
    for prefix in sorted(table, key=len, reverse=True):
        if prefix != "*" and name.startswith(prefix):
            return table[prefix]
    if "*" in table:
        return table["*"]
    raise ValueError(f"{case.id}: no sensor for {name!r}")


def _name_params(name: str) -> dict:
    """vc / ap / f / t values written into a curve name ('T f=0.05', 'T_IR_ap0.4_f0.05', 'Tmax_vc130')."""
    out = {}
    for key, value in re.findall(r"(?<![a-z])(vc|ap|f)=?([0-9]+(?:\.[0-9]+)?)", name):
        out.setdefault(key, float(value))
    return out


@dataclass
class _Target:
    index: int
    point: int | None
    vc_m_s: float
    h_m: float
    width_m: float
    rake_rad: float
    time_s: float
    sensor: Sensor
    unit: str
    note: str = ""


@dataclass
class _Geometry:
    rake_rad: float
    clearance_rad: float
    incl_rad: float
    kappa_rad: float
    edge_radius_m: float
    extras: dict = field(default_factory=dict)


def _geometry(case, col) -> _Geometry:
    rake = H.rake_rad(case, None, col) if case.leaf("tool", "rake_deg") is not None else None
    clearance = H.clearance_rad(case, col)
    incl = (case.value("tool", "inclination_deg") or 0.0) * H.DEG
    kappa_deg = case.value("tool", "entering_angle_deg", "cutting_edge_angle_deg", "lead_angle_deg")
    kappa = (float(kappa_deg) if isinstance(kappa_deg, (int, float)) else 90.0) * H.DEG
    kind = "solid" if case.operation in MILLING_OPS else "insert"
    r_e = H.edge_radius_m(case, None, col, kind)
    return _Geometry(rake if rake is not None else 0.0, clearance, incl, kappa, r_e)


def _cut_time(case, vc_m_s: float, col) -> tuple[float, str]:
    """Cutting time of a turning test: stated (cut_duration_s), stated per speed in the notes
    ('... after 86.2 s (vc 70)'), else the documented assumption."""
    stated = case.si("conditions", "cut_duration_s")
    if isinstance(stated, (int, float)):
        return float(stated), "case: conditions.cut_duration_s"
    times = {float(v): float(t) for t, v in re.findall(r"([0-9]+(?:\.[0-9]+)?) s \(vc ([0-9]+)\)", case.notes)}
    vc_min = vc_m_s * 60.0
    for v, t in times.items():
        if abs(v - vc_min) < 0.5:
            return t, f"case notes: maximum reached after {t:g} s at vc {v:g} m/min"
    if times:
        # pass time from the pass length, feed and the bar diameter implied by the first stated test
        length = case.si("conditions", "pass_length_mm")
        feed = case.si("conditions", "feed_mm_rev")
        if isinstance(length, (int, float)) and isinstance(feed, (int, float)):
            v0 = sorted(times)[0]
            t0 = times[v0]
            revs = length / feed
            circumference = (v0 / 60.0) * t0 / revs
            t = revs * circumference / vc_m_s
            col.assume(f"cutting time at vc {vc_min:.0f} m/min", f"{t:.1f} s",
                       f"pass length / (f n) with the bar diameter {circumference / math.pi * 1e3:.1f} mm implied by "
                       f"the {v0:g} m/min test (its maximum at {t0:g} s taken as its pass end)", "case")
            return t, "derived pass time"
    col.assume("cutting time", f"{AS.TEST_CUT_TIME.value:g} s", AS.TEST_CUT_TIME.reason, "est")
    return AS.TEST_CUT_TIME.value, "assumed"


def _targets(case, geo: _Geometry, col) -> list[_Target]:
    out = []
    rows = H.test_rows(case)
    milling = case.operation in MILLING_OPS
    for i, entry in enumerate(case.reference_values):
        row = H.find_test(rows, entry, ("vc_m_min",))
        if row is None:
            raise ValueError(f"{case.id}: no test matches {entry.get('name')!r}")
        vc = H.speed_m_s(row)
        h = H.uncut_thickness_m(row)
        rake = H.rake_rad(case, row, col)
        width = H.width_m(case, row) or 1e-3
        t, _ = _cut_time(case, vc, col)
        out.append(_Target(i, None, vc, h * math.sin(geo.kappa_rad), width / math.sin(geo.kappa_rad), rake, t,
                           _sensor(case, str(entry.get("name"))), str(entry.get("unit") or "degC"),
                           f"test {row.test_id}"))
    cond = case.section("conditions")
    for c, curve in enumerate(case.reference_curves):
        name = str(curve.get("name"))
        params = _name_params(name)
        x_name = str(curve.get("x_name"))
        sensor = _sensor(case, name)
        for k, (x, _) in enumerate(curve.get("points") or []):
            vc = to_si(x, curve.get("x_unit")) if x_name == "vc" else (params["vc"] / 60.0 if "vc" in params else
                                                                        H.speed_m_s(rows[0]))
            if vc is None and isinstance(cond.get("cutting_speed_m_min"), (int, float)):
                vc = float(cond["cutting_speed_m_min"]) / 60.0
            feed = params.get("f")
            if feed is None:
                f_leaf = case.si("conditions", "feed_mm_rev", "fz_mm_tooth")
                feed_m = f_leaf if isinstance(f_leaf, (int, float)) else None
            else:
                feed_m = feed * 1e-3
            ap = to_si(x, curve.get("x_unit")) if x_name == "ap" else (params["ap"] * 1e-3 if "ap" in params else
                                                                        (case.si("conditions", "ap_mm",
                                                                                 "width_of_cut_mm")))
            if feed_m is None or ap is None or vc is None:
                raise ValueError(f"{case.id}: curve {name}: speed, feed or depth not found")
            if x_name == "cutting_time":
                t = to_si(x, curve.get("x_unit"))
            elif milling:
                t = 0.0
            else:
                t, _ = _cut_time(case, vc, col)
            if milling:
                h_m, width = feed_m, ap
            else:
                h_m, width = feed_m * math.sin(geo.kappa_rad), ap / math.sin(geo.kappa_rad)
            out.append(_Target(c, k, vc, h_m, width, geo.rake_rad, t, sensor, str(curve.get("y_unit") or "degC"),
                               f"vc {vc * 60:.0f} m/min, f {feed_m * 1e3:g} mm, ap {ap * 1e3:g} mm"))
    return out


def _milling_peak(case, geo, contact_models, work, tool, amb, target: _Target, speeds, col) -> tuple[float, str, dict]:
    d = case.si("tool", "diameter_mm")
    ae = case.si("conditions", "ae_mm")
    helix = (case.value("tool", "helix_deg") or 0.0) * H.DEG
    phi_e = math.acos(max(min(1.0 - 2.0 * ae / d, 1.0), -1.0))
    h_max = target.h_m * math.sin(phi_e)
    omega = target.vc_m_s / (0.5 * d)
    contact_time = phi_e / omega
    width = 0.5 * d * phi_e / max(math.sin(helix), 1e-6) if helix > 0 else target.width_m
    key = (round(target.rake_rad, 9), "mill")
    if key not in contact_models:
        contact_models[key] = ContactModel(work, tool, rake_rad=target.rake_rad, clearance_rad=geo.clearance_rad,
                                           edge_radius_m=geo.edge_radius_m, work_temp_K=amb, speeds_m_s=speeds,
                                           h_values_m=[h_max], incl_rad=helix)
    st = contact_models[key].state(h_max, target.vc_m_s, amb, width, contact_time)
    reading = st.contact_mean_K
    col.assume("tool body temperature (milling)", "ambient", "one short pass per test; the engagement's transient "
               "contact (contact time = engagement angle / omega) is in the partition", "est")
    note = (f"h_max {h_max * 1e6:.1f} um (engagement {math.degrees(phi_e):.1f} deg), contact {contact_time * 1e3:.2f} "
            f"ms, edge length in cut {width * 1e3:.2f} mm")
    note += (f"; rake contact {st.T_interface_K - 273.15:.0f} degC over {st.contact_length_m * 1e6:.1f} um, flank "
             f"{st.flank_temp_K - 273.15:.0f} degC over {st.flank_contact_m * 1e6:.1f} um")
    return reading, note, {"h_max_um": h_max * 1e6, "regime": st.regime, "rake_contact_degC": st.T_interface_K - 273.15,
                           "flank_contact_degC": st.flank_temp_K - 273.15}


@harness("tool_temperature", models=("cnctwin.mechanics.coefficients", "cnctwin.thermal.toolmodel",
                                     "cnctwin.thermal.kernels.moving_rect_rise"),
         description="The contact through the coefficient pipeline (cnctwin.mechanics.coefficients: Oxley rows, "
                     "edge forces, partition at the tool body temperature) coupled to thermal's tool ladder of the "
                     "insert; readings as measured: natural thermocouple = mean interface temperature, IR window or "
                     "spot and embedded thermocouple = the tool temperature at the location (body + local edge "
                     "field, thermal's edge-model composition).")
def tool_temperature(case, db) -> HarnessOutput:
    col = H.Collector()
    work, raw = H.work_material(db, case, col)
    tool, tm, cid = H.tool_side(db, case, col)
    amb = H.ambient(case, col)
    friction = work.friction_for(tool.tool_class, tool.coating)
    H.material_provenance(raw, col, friction_entry=friction)
    if not H.dry(case):
        col.note("The case is not dry; dry contact assumed.")
    geo = _geometry(case, col)
    targets = _targets(case, geo, col)
    milling = case.operation in MILLING_OPS
    wedge = wedge_factor(geo.rake_rad, geo.clearance_rad)
    contact_models: dict = {}
    sims: dict = {}
    setup = None if milling else insert_setup(case, tm, col)
    for s in {t.sensor.source for t in targets}:
        col.note(f"Sensor: {s}.")
    results: dict = {}
    for target in targets:
        key = (target.index, target.point)
        if milling:
            temp, note, extra = _milling_peak(case, geo, contact_models, work, tool, amb, target,
                                              sorted({t.vc_m_s for t in targets}), col)
            results[key] = (temp, note, extra)
            continue
        mkey = round(target.rake_rad, 9)
        if mkey not in contact_models:
            contact_models[mkey] = ContactModel(work, tool, rake_rad=target.rake_rad, clearance_rad=geo.clearance_rad,
                                                edge_radius_m=geo.edge_radius_m, work_temp_K=amb,
                                                speeds_m_s=[t.vc_m_s for t in targets],
                                                h_values_m=[t.h_m for t in targets], incl_rad=geo.incl_rad)
        model = contact_models[mkey]
        if target.h_m <= model.h_min:
            results[key] = (None, f"no chip: h {target.h_m * 1e6:.1f} um <= h_min {model.h_min * 1e6:.1f} um", {})
            continue
        sens = target.sensor
        horizon = max(target.time_s, max(sens.times_s) if sens.times_s else 0.0)
        skey = (mkey, round(target.vc_m_s, 9), round(target.h_m, 12), round(target.width_m, 12), round(horizon, 6))
        if skey not in sims:
            sims[skey] = _simulate(model, setup, target, horizon, amb, wedge_factor(target.rake_rad,
                                                                                    geo.clearance_rad))
        results[key] = _read(sims[skey], target)
    preds_values = [None] * len(case.reference_values)
    preds_curves = [[None] * len(c.get("points") or []) for c in case.reference_curves]
    for target in targets:
        temp, note, extra = results[(target.index, target.point)]
        pred = no_prediction(note) if temp is None else Prediction(from_si(temp, target.unit), COMPARED,
                                                                     f"{target.note}; {note}", extra=extra)
        if target.point is None:
            preds_values[target.index] = pred
        else:
            preds_curves[target.index][target.point] = pred
    settings = {"tool_material": tm, "coating": cid, "mu_sliding": friction.mu, "edge_radius_um":
                geo.edge_radius_m * 1e6, "rake_deg": math.degrees(geo.rake_rad), "clearance_deg":
                math.degrees(geo.clearance_rad), "entering_angle_deg": math.degrees(geo.kappa_rad),
                "wedge_factor": wedge, "ladder_step_s": AS.LADDER_STEP.value}
    if not milling:
        col.note("Chip section of the turning cases: h = f sin(kappa_r), width = ap / sin(kappa_r) (straight edge; "
                 "the nose radius is not resolved).")
    return HarnessOutput(preds_values, preds_curves,
                         ["cnctwin.mechanics.coefficients.build_process_map/group_coefficients",
                          "thermal partition (rake/edge shares, thermal/partition.py convention)"] +
                         ([] if milling else ["cnctwin.thermal.toolmodel.ToolThermal (insert ladder)",
                                              "cnctwin.thermal.kernels.moving_rect_rise (local edge field)"]),
                         settings, col.assumptions, col.provenance, col.notes, amb)


def _simulate(model: ContactModel, setup, target: _Target, horizon_s: float, amb: float, wedge: float) -> dict:
    """Step the insert ladder through the cut; keep, at each step end, what the sensors need."""
    insert = InsertThermal(setup, amb)
    dt = AS.LADDER_STEP.value
    t = 0.0
    snaps = []
    while t < horizon_s - 1e-12:
        step = min(dt, horizon_s - t)
        st = model.state(target.h_m, target.vc_m_s, insert.tip_temperature(), target.width_m)
        insert.step(step, st.tool_W, st, wedge)
        t += step
        st_after = model.state(target.h_m, target.vc_m_s, insert.tip_temperature(), target.width_m)
        snaps.append((t, st_after, _readings(insert, st_after, target, t, wedge)))
    return {"snaps": snaps}


def _readings(insert, st, target: _Target, t: float, wedge: float) -> dict:
    sens = target.sensor
    out = {"interface": st.T_interface_K, "tip_body": insert.tip_temperature()}
    if sens.kind == "side_window":
        out["sensor"] = window_mean(insert, st, (0.0, sens.extent_x_m), (0.0, sens.extent_z_m), 0.5 * st.width_m,
                                    time_s=t, wedge=wedge, side_adiabatic=True)
    elif sens.kind == "rake_spot":
        out["sensor"] = disc_mean_on_rake(insert, st, sens.x_m, 0.5 * st.width_m + sens.extent_x_m, sens.extent_x_m,
                                          time_s=t, wedge=wedge)
    elif sens.kind == "point":
        out["sensor"] = temperature_at_point(insert, st, sens.x_m, 0.0, sens.z_m, time_s=t, wedge=wedge)
    else:
        out["sensor"] = st.contact_mean_K
    return out


def _read(sim: dict, target: _Target) -> tuple[float | None, str, dict]:
    snaps = sim["snaps"]
    sens = target.sensor
    if not snaps:
        return None, "no cutting time", {}
    if sens.reading == "time_mean":
        sel = [s for s in snaps if s[0] <= target.time_s + 1e-9]
        value = sum(s[2]["sensor"] for s in sel) / len(sel)
        how = f"mean over {target.time_s:g} s of cutting"
    elif sens.reading == "mean_at":
        vals = []
        for want in sens.times_s:
            best = min(snaps, key=lambda s: abs(s[0] - want))
            vals.append(best[2]["sensor"])
        value = sum(vals) / len(vals)
        how = f"mean of the readings at {', '.join(f'{x:g}' for x in sens.times_s)} s"
    else:
        best = min(snaps, key=lambda s: abs(s[0] - target.time_s))
        value = best[2]["sensor"]
        how = f"at {best[0]:g} s"
    last = snaps[-1]
    st = last[1]
    extra = {"rake_contact_degC": st.T_interface_K - 273.15, "flank_contact_degC": st.flank_temp_K - 273.15,
             "tip_body_degC": last[2]["tip_body"] - 273.15, "tool_heat_W": st.tool_W,
             "contact_length_um": st.contact_length_m * 1e6, "flank_contact_um": st.flank_contact_m * 1e6}
    note = (f"{sens.kind} {how}; rake contact {st.T_interface_K - 273.15:.0f} degC over "
            f"{st.contact_length_m * 1e6:.0f} um, flank contact {st.flank_temp_K - 273.15:.0f} degC over "
            f"{st.flank_contact_m * 1e6:.0f} um, body at the edge {last[2]['tip_body'] - 273.15:.0f} degC, tool heat "
            f"{st.tool_W:.1f} W")
    return value, note, extra


# ------------------------------------------------------------------------------------------ discussion
def _mean_signed(result) -> float | None:
    errs = [c.error_rel for c in result.comparisons if c.status == COMPARED and c.error_rel is not None
            and math.isfinite(c.error_rel)]
    return sum(errs) / len(errs) if errs else None


@discussion("tool_temperature")
def _discuss_tool(q) -> dict:
    out = {}
    causes = {
        "interface": "the natural thermocouple averages the whole contact; the simulated rake contact is Oxley's mean "
                     "with the database Johnson-Cook set and friction",
        "side_window": "beyond about 2 max(l_c, l_f, edge element) from the edge thermal's composition leaves only the "
                       "1-D insert body (a cross-section mean of the whole insert), so a window 0-2 mm from the edge "
                       "is read nearly at the body temperature, while the measured window sees the steep 3-D field "
                       "near the contact (and the source's assumed emissivity 0.48)",
        "rake_spot": "the same limitation 1 mm from the edge (body + local field, the local field cut at r_far); the "
                     "material is the database's 30 HRC entry for a 50 HRC bar",
        "point": "the junction sits in the insert body: the 1-D ladder heats the insert as a whole (seat contact, "
                 "holder bar and cutting time are estimates or derived)",
        "interface_peak_milling": "a 3-6 um aluminium chip at a 20 um hone (tooling default): the flank contact of the "
                                  "edge dominates the contact mean; minimum chip thickness and edge radius are estimates",
    }
    for r in q.cases:
        if r.status != "ok" or r.stats is None or r.stats.passes:
            continue
        by_kind: dict = {}
        for c in r.comparisons:
            if c.status == COMPARED and c.error_rel is not None and math.isfinite(c.error_rel):
                by_kind.setdefault(_sensor(r.case, c.name).kind, []).append(c.error_rel)
        parts = [f"{kind}: mean signed error of the rise {sum(v) / len(v) * 100:+.0f} % ({causes.get(kind, kind)})"
                 for kind, v in sorted(by_kind.items())]
        out[r.id] = "; ".join(parts) + "."
    out["summary"] = (H.budget_summary(q) + "Temperatures are compared as rises above ambient. Readings at a distance "
                      "from the contact (IR window and spot) come out low because thermal's tool model resolves the "
                      "contact flash and a 1-D insert body, not the 3-D field in the first millimetres of the insert; "
                      "readings inside the insert depend on the heat into the insert, the seat and holder contact and "
                      "the cutting time. No coefficient was adjusted; the cutting times and sensor positions the "
                      "sources leave open are listed per case.")
    return out


@discussion("chip_temperature")
def _discuss_chip(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        mean = _mean_signed(r)
        if mean is not None and not r.stats.passes:
            out[r.id] = (f"mean signed error {mean * 100:+.0f} % of the rise. The reading depends on the oxide "
                         "model (activation energy est., colour table read as 30-min holds) as much as on the chip "
                         "temperature; at the colour table's top the reference is only a lower bound.")
    out["summary"] = (H.budget_summary(q) + "One indirect reference (tempering colours, capped at 360 C). The simulated reading is the "
                      "colour-equivalent temperature of the chip's thermal history, so the oxide-growth estimates of "
                      "mechanics-colour.yaml enter directly.")
    return out


__all__ = ["chip_temperature", "tool_temperature", "SENSORS", "Sensor"]
