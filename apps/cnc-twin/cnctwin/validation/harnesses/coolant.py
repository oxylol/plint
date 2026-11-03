"""Harness for coolant_reach: what the coolant model does to the measured quantity.

The coolant enters the cut through ``cnctwin.coolant`` the way the run uses it: the jet's reach at the
contact (1 here: one free jet aimed at the cutting zone from 20 mm, nothing in its way; the occlusion
ray tracing has no scene to trace in these set-ups), the lubrication friction factor
(``cnctwin.coolant.lubrication.friction``: database factor of the fluid for the tool class and work
family, capillary access at the chip speed), applied to the mechanics (coefficient pipeline, chip
form). Per case:

* tool-work thermocouple in milling under dry / external HPC / rotary HPC (sultana-2023): the mean
  contact temperature of the engagement (rake + flank contact, ``contact.ContactState.contact_mean_K``)
  at the mean chip thickness, with the engagement time as contact time, tool body at ambient; HPC
  changes the friction factor (the jet's convective cooling of the tool body does not enter a quantity
  read during the engagement);
* chip breaking index and cutting force against the jet direction (struzikiewicz-2023): the chip form
  (``chipform`` harness pieces) and the cutting force of the coefficient pipeline with the friction
  factor; the jet direction enters the model only through the reach, which is 1 for every direction
  here, so the model predicts the same chip for all nine directions (stated in the report).
"""

from __future__ import annotations

import math
import re

from ...mechanics import contact as C
from ..cases import from_si, to_si
from ..registry import CLASS, COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import common as H
from .chipform import _Chips
from .contact import ContactModel

MILLING_FLUIDS = {"VG-68": "straight-oil", "emulsion": "emulsion-8pct"}


def _fluid(db, fluid_id: str, tool_class: str, family: str, v_m_s: float, ambient: float, col) -> float:
    from ...coolant import fluid as F
    from ...coolant import lubrication as L
    fluid = F.fluid_from_db(db.coolant(fluid_id))
    lub = L.friction(fluid, fluid.props(ambient), 1.0, tool_class, family, v_m_s, None, None, 1e-3, fluid.is_mql,
                     fluid.dosage_m3_s or 0.0)
    col.assume(f"coolant fluid {fluid_id}", "database entry", "the case's fluid is not in the database: the closest "
               "database fluid, full reach of the jet", "db/coolants")
    return lub.factor


def _milling_temperature(case, db, col) -> HarnessOutput:
    work, raw = H.work_material(db, case, col)
    tool, tm, cid = H.tool_side(db, case, col, kind="square_endmill")
    amb = H.ambient(case, col)
    H.material_provenance(raw, col, friction_entry=work.friction_for(tool.tool_class, tool.coating))
    d = case.si("tool", "diameter_mm")
    ae = case.si("conditions", "ae_mm")
    ap = case.si("conditions", "ap_mm")
    helix = (case.value("tool", "helix_deg") or 0.0) * H.DEG
    rake = H.rake_rad(case, None, col)
    clearance = H.clearance_rad(case, col)
    r_e = H.edge_radius_m(case, None, col, "solid")
    phi_e = math.acos(max(min(1.0 - 2.0 * ae / d, 1.0), -1.0))
    width = min(0.5 * d * phi_e / max(math.sin(helix), 1e-6), ap / max(math.cos(helix), 1e-6))
    col.assume("tool body temperature", "ambient", "the reading is the engagement's contact temperature; the cut's "
               "duration is not stated", "est")
    col.assume("chip thickness", "mean over the engagement, fz (1 - cos phi_e) / phi_e", "a thermocouple average over "
               "the cut", "est")
    entries = case.reference_values
    speeds = sorted({to_si(float(re.search(r"vc([0-9]+)", str(e["name"])).group(1)), "m/min") for e in entries})
    h_vals = sorted({to_si(e["x"], e.get("x_unit")) * (1.0 - math.cos(phi_e)) / phi_e for e in entries})
    models = {}
    preds = []
    fluid_text = str(case.section("coolant").get("fluid") or "")
    fluid_id = next((v for k, v in MILLING_FLUIDS.items() if k.lower() in fluid_text.lower()), "straight-oil")
    for entry in entries:
        name = str(entry["name"])
        mode = re.search(r"T_([a-z]+)_vc", name).group(1)
        vc = to_si(float(re.search(r"vc([0-9]+)", name).group(1)), "m/min")
        fz = to_si(entry["x"], entry.get("x_unit"))
        h_m = fz * (1.0 - math.cos(phi_e)) / phi_e
        factor = 1.0 if mode == "dry" else _fluid(db, fluid_id, tool.tool_class, work.family, vc, amb, col)
        key = round(factor, 9)
        if key not in models:
            models[key] = ContactModel(work, tool, rake_rad=rake, clearance_rad=clearance, edge_radius_m=r_e,
                                       work_temp_K=amb, speeds_m_s=speeds, h_values_m=h_vals, incl_rad=helix,
                                       friction_factor=factor)
        omega = vc / (0.5 * d)
        st = models[key].state(h_m, vc, amb, width, phi_e / omega)
        unit = str(entry.get("unit") or "degC")
        preds.append(Prediction(from_si(st.contact_mean_K, unit), COMPARED,
                                f"{mode}: friction factor {factor:.2f}, h_mean {h_m * 1e6:.1f} um, contact "
                                f"{phi_e / omega * 1e3:.1f} ms; rake {st.T_interface_K - 273.15:.0f} degC, flank "
                                f"{st.flank_temp_K - 273.15:.0f} degC",
                                extra={"friction_factor": factor, "mode": mode}))
    col.note("The rotary applicator (rhpc) and the single jet (hpc) differ in the model only through reach, which is 1 "
             "for both: they get the same prediction.")
    settings = {"tool_material": tm, "coating": cid, "fluid": fluid_id, "engagement_deg": math.degrees(phi_e),
                "edge_radius_um": r_e * 1e6, "helix_deg": math.degrees(helix)}
    return HarnessOutput(preds, [[] for _ in case.reference_curves],
                         ["cnctwin.coolant.lubrication.friction", "cnctwin.mechanics.coefficients (contact mean "
                          "temperature)"], settings, col.assumptions, col.provenance, col.notes, amb)


def _grade(cch: float) -> str:
    """The case's grade from the breaking index: + up to 5 mm, 0 5-20 mm, - over 20 mm (Cch = 0.01 / mm)."""
    mm = cch * 100.0
    return "+" if mm <= 5.0 else ("0" if mm <= 20.0 else "-")


def _turning_jet(case, db, col) -> HarnessOutput:
    chips = _Chips(db, case, col, coolant=True)
    work = chips.work
    vc = case.si("conditions", "cutting_speed_m_min")
    ap = case.si("conditions", "ap_mm")
    fluid_text = str(case.section("coolant").get("fluid") or "")
    fluid_id = "emulsion-8pct"
    factor = chips.set_coolant(db, fluid_id)
    col.note(f"Coolant '{fluid_text}' as {fluid_id}; reach 1 for every jet direction (one free jet 20 mm from the "
             "corner aimed at the rake face).")
    h_min = C.min_chip_thickness(work, chips.r_e)
    models = {}
    preds = []
    for entry in case.reference_values:
        name = str(entry["name"])
        f = float(re.search(r"_f([0-9.]+)", name).group(1)) * 1e-3
        unit = str(entry.get("unit") or "")
        if name.startswith("Cch"):
            form = chips.form(f, ap, vc)
            ref_grade = re.search(r"grade ([+0-])", str(entry.get("note") or ""))
            reference = ref_grade.group(1) if ref_grade else _grade(float(entry["value"]))
            if form is None:
                preds.append(no_prediction("no chip formed", kind=CLASS, reference_class=reference))
                continue
            if math.isfinite(form.length_m):
                cch = min(1.0, 0.01 * (form.length_m + 2.0 * form.curl_radius_m) * 1e3)
            else:
                cch = 1.0
            preds.append(Prediction(_grade(cch), COMPARED, f"f {f * 1e3:g}: {form.form_name}, Cch {cch:.2f}",
                                    kind=CLASS, reference_class=reference, extra={"Cch": cch}))
        else:
            key = round(f, 9)
            if key not in models:
                models[key] = ContactModel(work, chips.tool, rake_rad=chips.rake, clearance_rad=6.0 * H.DEG,
                                           edge_radius_m=chips.r_e, work_temp_K=chips.amb, speeds_m_s=[vc],
                                           h_values_m=[f * math.sin(chips.kappa)], incl_rad=chips.incl,
                                           friction_factor=factor)
            st = models[key].state(f * math.sin(chips.kappa), vc, chips.amb, ap / math.sin(chips.kappa))
            preds.append(Prediction(from_si(st.force_t_N, unit), COMPARED, f"f {f * 1e3:g}: friction factor "
                                    f"{factor:.2f}"))
    settings = {"fluid": fluid_id, "friction_factor": factor, "rake_deg": math.degrees(chips.rake),
                "edge_radius_um": chips.r_e * 1e6, "h_min_um": h_min * 1e6}
    return HarnessOutput(preds, [[] for _ in case.reference_curves],
                         ["cnctwin.coolant.lubrication.friction", "cnctwin.mechanics.chipform.chip_form",
                          "cnctwin.mechanics.coefficients (cutting force)"], settings, col.assumptions, col.provenance,
                         col.notes, chips.amb)


@harness("coolant_reach", models=("cnctwin.coolant.lubrication", "cnctwin.mechanics.coefficients",
                                  "cnctwin.mechanics.chipform"),
         description="The coolant's effect on the measured quantity through cnctwin.coolant (reach of the jet at the "
                     "cut, lubrication friction factor) applied to the mechanics: the contact temperature of a milling "
                     "engagement (tool-work thermocouple), or the chip form and the cutting force of a turning cut.")
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
        out[r.id] = "; ".join(parts) + (". The model's coolant effect is the friction factor; the jet's cooling and "
                                        "chip-breaking actions and its direction (beyond reach) do not change the "
                                        "predicted quantity.")
    out["summary"] = ("No budget is defined for coolant_reach. In these set-ups the coolant model reaches the "
                      "mechanics only through the lubrication friction factor, so measured effects of jet direction, "
                      "convective cooling during the engagement and jet-assisted chip breaking are not reproduced.")
    return out


__all__ = ["coolant_reach"]
