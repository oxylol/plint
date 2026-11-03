"""Harness for chip_form: cnctwin.mechanics chip-form classification against the case's classes.

Per tested condition: the chip state from Oxley (``oxley.solve`` at t1 = h - h_min), then
``cnctwin.mechanics.chipform.chip_form`` with the insert's chip breaker from the database when the
case's insert is there (else no breaker: natural curl), the turning chip section h = f sin(kappa_r),
b = ap / sin(kappa_r), and the coolant's friction factor (``cnctwin.coolant.lubrication``) for
high-pressure coolant. The simulated chip length is mapped to the case's own classes:

* grades of chip length (slodki-2019: + <= 20 mm, 0 20-100 mm, - > 100 mm; zebala-2020: good <= 25,
  fair 25-50, poor > 50 mm): the class of the simulated piece length (an unbroken chip is the longest);
* chip-control windows (a maker's chart): a grid of feed and depth, a point is controlled when the
  form is a breaking one (ISO 3685 classes the case names: 2.2-2.3, 3, 4.2-4.3, 5, 6, 7), and the
  window's edges are compared with the chart's.

The jet's own chip-breaking action is not modelled (a coolant model effect on chips that the mechanics
chip form does not take); noted per case.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ...mechanics import chipform as CF
from ...mechanics import contact as C
from ...mechanics import oxley
from ...mechanics.materials import ToolSide
from ..cases import from_si, to_si
from ..registry import CLASS, COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import common as H

BREAKING_CODES = (22, 23, 31, 32, 42, 43, 51, 52, 61, 62, 70)
DB_INSERTS = {"CNMG120408-TM": "tungaloy-cnmg120408-tm-t9215-pclnr2525m12e"}
HOLDER_RAKE_DEG = -6.0          # negative ISO holders (PCLNR/C6-PCMNN): rake -6, inclination -6 (not stated)
HOLDER_INCL_DEG = -6.0
FEED_GRID_M = tuple(np.round(np.geomspace(0.04e-3, 0.8e-3, 24), 9))
DEPTH_GRID_M = tuple(np.round(np.geomspace(0.25e-3, 8e-3, 24), 9))


class _Chips:
    def __init__(self, db, case, col: H.Collector, *, coolant: bool):
        self.case = case
        self.col = col
        self.work, self.raw = H.work_material(db, case, col)
        self.amb = H.ambient(case, col)
        iso = str(case.value("tool", "iso_designation") or "")
        item_id = DB_INSERTS.get(iso)
        self.item_id = item_id
        if item_id is not None:
            item = db.tool(item_id)
            tm_id = str(item.tool_material.id) if item.tool_material is not None else "carbide-p30"
            coat = item.coating.id if item.coating is not None and item.coating.id else "none"
            self.tool = ToolSide.from_db(item, db.tool_material(tm_id), None if coat == "none" else db.coating(coat))
            raw_item = H.raw_object(db, "tool", item_id)
            cb = ((raw_item.get("geometry") or {}).get("insert") or {}).get("chip_breaker") or {}
            col.used(f"tool {item_id} chip breaker window", str(raw_item.get("src") or ""), str(cb.get("page") or ""))
            self.tool_mat, self.coating = tm_id, coat
            col.note(f"Insert from the database: {item_id} (chip breaker {cb.get('name')}, window and land/rake of the "
                     "maker's catalogue).")
        else:
            self.tool, self.tool_mat, self.coating = H.tool_side(db, case, col)
            breaker = case.value("tool", "chipbreaker")
            col.assume("chip breaker geometry", "none (natural curl)", f"the {breaker} breaker's land and groove are "
                       "not in the database", "db/tools")
        friction = self.work.friction_for(self.tool.tool_class, self.tool.coating)
        H.material_provenance(self.raw, col, friction_entry=friction)
        self.rake = (self.tool.chip_breaker.rake_rad if self.tool.chip_breaker is not None and
                     self.tool.chip_breaker.rake_rad is not None else HOLDER_RAKE_DEG * H.DEG)
        if self.tool.chip_breaker is None or self.tool.chip_breaker.rake_rad is None:
            col.assume("rake and inclination", f"{HOLDER_RAKE_DEG:g} / {HOLDER_INCL_DEG:g} deg", "not stated: a "
                       "negative ISO holder for a negative insert", "est")
        self.incl = HOLDER_INCL_DEG * H.DEG
        kappa = case.value("tool", "entering_angle_deg")
        self.kappa = (float(kappa) if isinstance(kappa, (int, float)) else 95.0) * H.DEG
        if not isinstance(kappa, (int, float)):
            col.assume("entering angle", "95 deg", "not stated: the C-insert holders named (PCLNR, PCMNN) are 95 deg "
                       "holders", "est")
        self.r_e = H.edge_radius_m(case, None, col, "insert")
        self.nose = case.si("tool", "nose_radius_mm") or 0.0
        self.friction_factor = 1.0
        self.coolant_on = coolant

    def set_coolant(self, db, fluid_id: str | None) -> float:
        if fluid_id is None:
            self.friction_factor = 1.0
            return 1.0
        from ...coolant import fluid as F
        from ...coolant import lubrication as L
        fluid = F.fluid_from_db(db.coolant(fluid_id))
        lub = L.friction(fluid, fluid.props(self.amb), 1.0, self.tool.tool_class, self.work.family, 1.0, None, None,
                         1e-3, fluid.is_mql, fluid.dosage_m3_s or 0.0)
        self.friction_factor = lub.factor
        self.col.assume(f"coolant {fluid_id}", f"friction factor {lub.factor:.2f}", "the case's emulsion is not in the "
                        "database: the database emulsion, full reach of the jet at the rake face", "db/coolants")
        return lub.factor

    def form(self, f_m: float, ap_m: float, v_m_s: float):
        h = f_m * math.sin(self.kappa)
        width = ap_m / math.sin(self.kappa)
        h_min = C.min_chip_thickness(self.work, self.r_e)
        if h <= h_min:
            return None
        res = oxley.solve(self.work, h - h_min, v_m_s, self.rake, work_temp_K=self.amb, tool=self.tool,
                          tool_temp_K=self.amb, friction_factor=self.friction_factor, contact_width_m=width)
        if res.status not in ("oxley", "merchant"):
            return None
        state = CF.ChipState.from_orthogonal(res)
        cond = CF.ChipConditions(h_m=h - h_min, width_m=width, speed_m_s=v_m_s, rake_rad=self.rake,
                                 incl_rad=self.incl, cut_length_m=None, feed_m=f_m, depth_m=ap_m,
                                 nose_radius_m=float(self.nose), process="turning")
        return CF.chip_form(self.work, self.tool, cond, state)


def _length_class(length_m: float, bins_mm: tuple[float, float], labels: tuple[str, str, str]) -> str:
    mm = length_m * 1e3 if math.isfinite(length_m) else math.inf
    if mm <= bins_mm[0]:
        return labels[0]
    if mm <= bins_mm[1]:
        return labels[1]
    return labels[2]


def _grades(case, db, chips: _Chips, rule_bins, labels, ref_class) -> list[Prediction]:
    out = []
    for entry in case.reference_values:
        name = str(entry.get("name"))
        ap = float(re.search(r"ap([0-9.]+)", name).group(1)) * 1e-3
        f = to_si(entry["x"], entry.get("x_unit"))
        vc_note = re.search(r"vc ([0-9]+)", str(entry.get("note") or ""))
        vc = (float(vc_note.group(1)) if vc_note else float(case.value("conditions", "cutting_speed_m_min"))) / 60.0
        hpc = "hpc" in name or bool(case.section("coolant").get("pressure_bar"))
        if "dry" in name:
            hpc = False
        chips.set_coolant(db, "emulsion-8pct" if hpc else None)
        form = chips.form(f, ap, vc)
        reference = ref_class(entry)
        if form is None:
            out.append(no_prediction("no chip (h <= h_min) or no chip-formation solution", kind=CLASS,
                                     reference_class=reference))
            continue
        cls = _length_class(form.length_m, rule_bins, labels)
        length = "unbroken" if not math.isfinite(form.length_m) else f"{form.length_m * 1e3:.1f} mm"
        out.append(Prediction(cls, COMPARED, f"f {f * 1e3:g}, ap {ap * 1e3:g}, vc {vc * 60:.0f}"
                              f"{', HPC' if hpc else ', dry'}: {form.form_name} ({form.form_code}), piece {length}, "
                              f"curl {form.curl_radius_m * 1e3:.1f} mm ({form.curl_mode}), bend {form.bend_strain:.3f} "
                              f"vs {form.fracture_strain:.3f}", kind=CLASS, reference_class=reference,
                              extra={"form_code": form.form_code, "length_mm": form.length_m * 1e3 if
                                     math.isfinite(form.length_m) else None}))
    return out


def _window(case, db, chips: _Chips) -> tuple[list, list]:
    vc = case.si("conditions", "cutting_speed_m_min")
    chips.set_coolant(db, None)
    controlled = np.zeros((len(FEED_GRID_M), len(DEPTH_GRID_M)), dtype=bool)
    for i, f in enumerate(FEED_GRID_M):
        for j, ap in enumerate(DEPTH_GRID_M):
            form = chips.form(float(f), float(ap), float(vc))
            controlled[i, j] = form is not None and form.form_code in BREAKING_CODES
    feeds = [FEED_GRID_M[i] for i in range(len(FEED_GRID_M)) if controlled[i].any()]
    depths = [DEPTH_GRID_M[j] for j in range(len(DEPTH_GRID_M)) if controlled[:, j].any()]
    note = f"{int(controlled.sum())} of {controlled.size} grid points controlled at vc {vc * 60:.0f} m/min"
    values = []
    for entry in case.reference_values:
        name = str(entry.get("name"))
        unit = str(entry.get("unit"))
        if not feeds:
            values.append(no_prediction("no feed/depth of the grid gives a breaking chip; " + note))
            continue
        value = {"chip_control_min_feed": min(feeds), "chip_control_max_feed": max(feeds),
                 "chip_control_min_ap": min(depths), "chip_control_max_ap": max(depths)}[name]
        values.append(Prediction(from_si(value, unit), COMPARED, note))
    curves = []
    for curve in case.reference_curves:
        upper = "upper" in str(curve.get("name"))
        preds = []
        for x, _y in curve.get("points") or []:
            f = to_si(x, curve.get("x_unit"))
            i = int(np.argmin(np.abs(np.array(FEED_GRID_M) - f)))
            ok = [DEPTH_GRID_M[j] for j in range(len(DEPTH_GRID_M)) if controlled[i, j]]
            if not ok:
                preds.append(no_prediction(f"no controlled depth at f {FEED_GRID_M[i] * 1e3:.3f} mm"))
                continue
            preds.append(Prediction(from_si(max(ok) if upper else min(ok), curve.get("y_unit")), COMPARED,
                                    f"grid feed {FEED_GRID_M[i] * 1e3:.3f} mm"))
        curves.append(preds)
    return values, curves


@harness("chip_form", models=("cnctwin.mechanics.oxley", "cnctwin.mechanics.chipform", "cnctwin.coolant.lubrication"),
         description="Oxley chip state and cnctwin.mechanics.chipform (curl, breaker window, Nakayama breaking, ISO 3685 "
                     "class, piece length) per tested feed and depth; the length is mapped to the case's own classes, "
                     "or a feed-depth grid gives the chip-control window.")
def chip_form(case, db) -> HarnessOutput:
    col = H.Collector()
    chips = _Chips(db, case, col, coolant=bool(case.section("coolant")))
    if case.section("coolant"):
        col.note("High-pressure coolant enters as a friction factor; the jet's chip-breaking action is not modelled.")
    names = [str(e.get("name")) for e in case.reference_values]
    curves = [[] for _ in case.reference_curves]
    if any(n.startswith("chip_control_") for n in names):
        values, curves = _window(case, db, chips)
    elif any(n.startswith("chip_grade") for n in names):
        values = _grades(case, db, chips, (20.0, 100.0), ("+", "0", "-"),
                         lambda e: {1: "+", 0: "0", -1: "-"}[int(e["value"])])
    else:
        values = _grades(case, db, chips, (25.0, 50.0), ("good", "fair", "poor"),
                         lambda e: _length_class(float(e["value"]) * 1e-3, (25.0, 50.0), ("good", "fair", "poor")))
    settings = {"tool_material": chips.tool_mat, "coating": chips.coating, "db_insert": chips.item_id,
                "rake_deg": math.degrees(chips.rake), "entering_angle_deg": math.degrees(chips.kappa),
                "edge_radius_um": chips.r_e * 1e6, "breaking_codes": list(BREAKING_CODES)}
    return HarnessOutput(values, curves, ["cnctwin.mechanics.oxley.solve", "cnctwin.mechanics.chipform.chip_form"],
                         settings, col.assumptions, col.provenance, col.notes, chips.amb)


@discussion("chip_form")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok" or r.stats is None:
            continue
        counted = [c for c in r.comparisons if c.counted]
        if not counted:
            continue
        miss = sum(1 for c in counted if c.within_tolerance is False or c.within_uncertainty is False)
        text = (f"{miss} of {len(counted)} compared values miss. Without the breaker's groove geometry the model "
                "curls the chip with its natural radius (15 x contact length, est) and breaks it only when the bending "
                "strain t2/2R exceeds 0.1 x the bulk fracture strain (est): it predicts unbroken chips everywhere; the "
                "HPC jet's breaking action is not modelled.")
        if any(c.name.startswith("chip_control") for c in counted):
            text += (" The case counts snarled tubular and washer chips (2.3, 4.3) and long conical helices (5.1) as "
                     "controlled; the model's unbroken chips fall in these classes at every grid point, so the "
                     "predicted window is the whole grid.")
        out[r.id] = text
    out["summary"] = ("No budget is defined for chip_form. The chip-form model's breaking criterion and curl radii are "
                      "estimates (mechanics.md 13) and the case inserts' breaker geometries (land, groove width and "
                      "depth) are mostly not in the database, so the comparison tests the defaults.")
    return out


__all__ = ["chip_form"]
