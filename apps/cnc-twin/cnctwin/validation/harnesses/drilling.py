"""Harnesses for drilling_thrust and drilling_torque: the cutting domain's drilling model with the case's drill.

The twist drill of the case is built by cnctwin.tooling (lips on the web offset with the helicoid rake of the
stated helix, chisel from the web or the stated chisel length with the planar-flank chisel rake, margins), its
rows come from the coefficient pipeline (cnctwin.cutting.coefficients.MechanicsModel for the process
``drilling``: Oxley / Johnson-Cook lips, the indentation model for the chisel and for the slow centre, calibrated
to the material's measured data of the pairing where the database has them), and cut_step integrates the
steady full-diameter cut (``harnesses.cutsim.drill``: every station engaged over the full circle, the surface
memory holding the previous revolution's cone; a pilot hole removes the stations inside its radius). Thrust =
mean axial force on the tool over a revolution, torque = mean process torque.

Per case the reference names carry the condition (``Fz_D6_vc50``, ``thrust feed 60 mm/min`` with the speed
as x, ``Fz_helix20``, ``Fz_f0.05`` with the pilot diameter as x, feed averages ``*_feedavg_D8_vc80.21``),
or the value's note does (``d 3, n 300, f 0.03``); drill dimensions given as fractions of D ('0.15 D') are
read as such.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from ..cases import from_si, to_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, not_compared
from . import common as H
from . import cutsim as CSIM
from .cutforce import coolant_factor, coolant_mode

MODEL_CHAIN = ["cnctwin.mechanics.materials (databases)", "cnctwin.tooling.build_tool (twist drill: lips, chisel, "
               "margins)", "cnctwin.cutting.coefficients.MechanicsModel (process drilling: lips by Oxley, chisel and "
               "slow centre by indentation, calibration to measured data of the pairing)",
               "cnctwin.cutting.kernels.cut_step (steady full-diameter drilling, cnctwin.cutting.drive.Driver)"]
LENGTH_OF_D = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*D\b", re.I)


def _plain(value):
    return H._text(value) if isinstance(value, dict) else value


def _of_D(case, key: str, D_m: float) -> float | None:
    """A drill dimension written as a number (mm) or as a fraction of D ('0.15 D')."""
    raw = case.value("tool", key)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) * 1e-3
    m = LENGTH_OF_D.match(str(raw))
    return float(m.group(1)) * D_m if m else None


@dataclass(frozen=True)
class DrillCond:
    D_m: float
    rpm: float
    f_m: float
    helix_deg: float | None = None
    pilot_m: float = 0.0
    coolant: str = "dry"


class Drills:
    """Drill models and simulations of one case (cached per drill and condition)."""

    def __init__(self, db, case, col: H.Collector):
        self.db, self.case, self.col = db, case, col
        self.pipe = CSIM.Pipeline(db, case, col, "drilling", tool_kind="twist_drill")
        self.geos: dict = {}
        self.runs: dict = {}
        self.layouts: dict = {}

    def geometry(self, D_m: float, helix_deg: float | None):
        key = (round(D_m, 9), helix_deg)
        if key in self.geos:
            return self.geos[key]
        case, col = self.case, self.col
        g = {"diameter_mm": D_m * 1e3}
        flutes = case.value("tool", "lip_count", "flutes")
        g["flutes"] = int(flutes) if isinstance(flutes, (int, float)) else 2
        g["lip_count"] = g["flutes"]
        point = case.value("tool", "point_angle_deg")
        g["point_angle_deg"] = float(point) if isinstance(point, (int, float)) else 118.0
        if not isinstance(point, (int, float)):
            col.assume("point angle", "118 deg", "not stated: the standard point of general-purpose twist drills", "est")
        helix = helix_deg if helix_deg is not None else case.value("tool", "helix_deg")
        if isinstance(helix, (int, float)):
            g["helix_deg"] = float(helix)
        chisel = _of_D(case, "chisel_edge_length", D_m)
        web = _of_D(case, "web_thickness_mm", D_m) or _of_D(case, "web_thickness", D_m) or \
            _of_D(case, "web_diameter", D_m)
        if chisel is not None:
            g["chisel_length_mm"] = chisel * 1e3
            if web is not None:
                col.note(f"Thinned point: the chisel from the stated chisel length {chisel * 1e3:.2f} mm (the web "
                         f"{web * 1e3:.2f} mm is thinned to it; the gash is modelled as lip).")
        elif web is not None:
            g["web_thickness_mm"] = web * 1e3
        angle = case.value("tool", "chisel_edge_angle_deg")
        if isinstance(angle, (int, float)):
            g["chisel_edge_angle_deg"] = float(angle)
        clear = case.value("tool", "lip_clearance_deg", "clearance_deg")
        if isinstance(clear, (int, float)):
            g["clearance_deg"] = {"primary": float(clear)}
        margin = _of_D(case, "margin_width", D_m) or _of_D(case, "margin_width_mm", D_m)
        if margin is not None:
            g["margin_width_mm"] = margin * 1e3
        r_e = case.value("tool", "edge_radius_um")
        if isinstance(r_e, (int, float)):
            g["edge_radius_um"] = float(r_e)
        g["cutting_length_mm"] = 3.0 * D_m * 1e3 + 0.5 * D_m * 1e3 / math.tan(math.radians(g["point_angle_deg"] / 2))
        g["overall_length_mm"] = g["cutting_length_mm"] + 2.0 * D_m * 1e3
        g["shank"] = {"diameter_mm": D_m * 1e3, "length_mm": 2.0 * D_m * 1e3}
        tool_class = H._text(case.section("tool").get("tool_class")) or self.pipe.tool.tool_class
        item = CSIM.tool_item("twist_drill", tool_class, self.pipe.tool_mat, self.pipe.coating, g,
                              f"validation-{case.id}-{D_m * 1e3:g}")
        geo = CSIM.build_geometry(item, stickout_m=4.0 * D_m)
        for note in geo.notes:
            col.note(f"Drill model (D {D_m * 1e3:g} mm): {note}")
        if "edge_radius_um" not in g:
            r_e = float(np.max(np.asarray(geo.edge.elem_edge_radius_m)))
            col.assume("cutting-edge radius", f"{r_e * 1e6:.1f} um", "not stated in the case: tooling's default for "
                       "the drill's edge preparation", "db/models/tooling.yaml")
        self.geos[key] = geo
        return geo

    def run(self, c: DrillCond) -> CSIM.DrillRun:
        if c in self.runs:
            return self.runs[c]
        geo = self.geometry(c.D_m, c.helix_deg)
        omega = -2.0 * math.pi * c.rpm / 60.0
        lkey = (round(c.D_m, 9), c.helix_deg, round(c.rpm, 6), c.coolant)
        lay = self.layouts.get(lkey)
        if lay is None:
            factor = coolant_factor(self.db, self.case, self.col, c.coolant, self.pipe.work, self.pipe.tool,
                                    abs(omega) * 0.5 * c.D_m, self.pipe.ambient_K)
            # the map key leaves the speed out: MechanicsModel reuses a map while the speed stays in its grid
            lay = CSIM.layout(self.pipe, geo, omega, friction_factor=factor,
                              key=f"d{c.D_m * 1e3:g}-h{c.helix_deg}-{c.coolant}")
            self.layouts[lkey] = lay
        run = CSIM.drill(lay, omega, c.f_m, rho_kg_m3=float(self.pipe.work.rho_kg_m3), r_min_m=0.5 * c.pilot_m)
        self.runs[c] = run
        return run


def _number(name: str, tag: str) -> float | None:
    m = re.search(rf"{tag}([0-9]*\.?[0-9]+)", name)
    return float(m.group(1)) if m else None


def _rpm_of(D_m: float, vc_m_min: float) -> float:
    return to_si(vc_m_min, "m/min") / (math.pi * D_m) * 60.0


def _case_D(case) -> float | None:
    D = case.si("tool", "diameter_mm")
    return float(D) if isinstance(D, (int, float)) else None


def _case_rpm(case, D_m: float) -> float:
    rpm = case.value("conditions", "spindle_rpm")
    if isinstance(rpm, (int, float)):
        return float(rpm)
    vc = case.value("conditions", "vc_m_min")
    if isinstance(vc, (int, float)):
        return _rpm_of(D_m, float(vc))
    raise ValueError(f"{case.id}: no spindle speed")


def _conditions(case, name: str, x, x_name: str, x_unit: str, note: str) -> list[DrillCond]:
    """The condition(s) of one reference (a feed average gives several)."""
    mode = coolant_mode(case, name)
    D = _case_D(case)
    d_tag = _number(name, "_D")
    if d_tag is not None:
        D = d_tag * 1e-3
    m = re.search(r"d ([0-9.]+), n ([0-9.]+), f ([0-9.]+)", note or "")
    if m:
        D = float(m.group(1)) * 1e-3
        return [DrillCond(D, float(m.group(2)), float(m.group(3)) * 1e-3, coolant=mode)]
    if D is None:
        raise ValueError(f"{case.id}: no drill diameter for {name!r}")
    vc = _number(name, "_vc")
    rpm = _rpm_of(D, vc) if vc is not None else None
    xn = (x_name or "").lower()
    helix = None
    pilot = 0.0
    f = None
    if xn in ("f", "fn", "feed"):
        f = to_si(float(x), x_unit)
    if xn.startswith("helix"):
        helix = float(x)
    if xn.startswith("pilot"):
        pilot = to_si(float(x), x_unit)
    if xn.startswith("spindle"):
        rpm = float(x)
        rate = _number(name, "feed ")
        if rate is not None:
            f = rate / rpm * 1e-3
    if rpm is None:
        rpm = _case_rpm(case, D)
    if f is None:
        tag = _number(name, "_f")
        if tag is not None:
            f = tag * 1e-3
    if f is None:
        feed = case.value("conditions", "feed_mm_rev", "fn_mm_rev")
        if isinstance(feed, (int, float)):
            f = float(feed) * 1e-3
        elif "feedavg" in name and isinstance(feed, (list, tuple)):
            return [DrillCond(D, rpm, float(v) * 1e-3, helix, pilot, mode) for v in feed]
    if f is None:
        raise ValueError(f"{case.id}: no feed for {name!r}")
    return [DrillCond(D, rpm, f, helix, pilot, mode)]


def _drilling(case, db, what: str) -> HarnessOutput:
    col = H.Collector()
    drills = Drills(db, case, col)

    def predict(name: str, unit: str, x, x_name, x_unit, note) -> Prediction:
        conds = _conditions(case, name, x, x_name, x_unit, note)
        runs = [drills.run(c) for c in conds]
        if what == "thrust":
            value = float(np.mean([r.thrust_N for r in runs]))
        else:
            value = float(np.mean([r.torque_Nm for r in runs]))
        c = conds[0]
        text = (f"D {c.D_m * 1e3:g} mm, {c.rpm:.0f} rpm, f " + "/".join(f"{k.f_m * 1e3:g}" for k in conds) + " mm/rev"
                + (f", helix {c.helix_deg:g} deg" if c.helix_deg is not None else "")
                + (f", pilot {c.pilot_m * 1e3:g} mm" if c.pilot_m > 0 else "") + (f", {c.coolant}" if c.coolant != "dry"
                                                                                    else ""))
        other = float(np.mean([r.torque_Nm if what == "thrust" else r.thrust_N for r in runs]))
        return Prediction(from_si(value, unit), COMPARED, text,
                          extra={("torque_Nm" if what == "thrust" else "thrust_N"): other})

    values = []
    for e in case.reference_values:
        name = str(e.get("name"))
        values.append(predict(name, str(e.get("unit") or ""), e.get("x"), str(e.get("x_name") or ""),
                              str(e.get("x_unit") or ""), str(e.get("note") or "")))
    curves = []
    for c in case.reference_curves:
        name = str(c.get("name"))
        curves.append([predict(name, str(c.get("y_unit") or ""), x, str(c.get("x_name") or ""),
                               str(c.get("x_unit") or ""), "") for x, _ in c.get("points") or []])
    if "workpiece_tilt_deg" in case.section("conditions"):
        col.note("Workpiece tilt (inclined entry) is not modelled: the steady torque of the full-diameter cut is "
                 "compared (the tilt acts at the entry of a 3 D deep hole).")
    settings = drills.pipe.settings_dict()
    settings.update({"edge_element_m": CSIM.EDGE_ELEMENT_M, "statistic": "mean over one revolution of the steady cut"})
    col.assume("runout", "0", "not stated in the case: an ideal concentric drill", "est")
    return HarnessOutput(values, curves, MODEL_CHAIN, settings, col.assumptions, col.provenance, col.notes, None)


@harness("drilling_thrust", models=("cnctwin.cutting.kernels", "cnctwin.cutting.coefficients", "cnctwin.tooling"),
         description="Steady full-diameter drilling through the cutting domain's kernel with the case's twist drill "
                     "(tooling: lips with the helicoid rake of the stated helix, chisel, margins) and the coefficient "
                     "pipeline for the process drilling (indentation model at the chisel and the slow centre); "
                     "thrust = mean axial force on the drill.")
def drilling_thrust(case, db) -> HarnessOutput:
    return _drilling(case, db, "thrust")


@harness("drilling_torque", models=("cnctwin.cutting.kernels", "cnctwin.cutting.coefficients", "cnctwin.tooling"),
         description="As drilling_thrust; torque = mean process torque about the drill axis.")
def drilling_torque(case, db) -> HarnessOutput:
    return _drilling(case, db, "torque")


def _discuss(q, what: str) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        comps = [c for c in r.comparisons if c.counted and c.error_rel is not None]
        if not comps:
            continue
        mean = sum(c.error_rel for c in comps) / len(comps)
        inside = sum(1 for c in comps if c.within_tolerance)
        cal = r.output.settings.get("calibration")
        text = (f"{inside}/{len(comps)} values within tolerance, mean signed error {mean * 100:+.0f} %; coefficients "
                + (f"calibrated to {cal} data" if cal not in (None, "none") else "from the mechanics alone") + ".")
        assumed = [a.path.split(".")[-1] for a in r.case.assumed_inputs()]
        if assumed:
            text += " Drill geometry the source leaves open: " + ", ".join(sorted(set(assumed))) + "."
        if what == "thrust":
            text += (" The thrust is dominated by the chisel edge (indentation, -54 to -60 deg rake) whose model is an "
                     "upper-bound-type contact estimate.")
        out[r.id] = text
    comps = [c for r in q.cases if r.status == "ok" and not r.calibration_dependent for c in r.comparisons
             if c.counted and c.error_rel is not None]
    if comps:
        mean = sum(c.error_rel for c in comps) / len(comps)
        out["summary"] = (H.budget_summary(q) + f"Mean signed error over the independent values {mean * 100:+.0f} %. "
                          "Most sources leave the web, chisel angle, helix or point angle open (listed per case as "
                          "assumed inputs)" + ("; the thrust depends on the chisel length almost linearly (chisel "
                                               "indentation)." if what == "thrust" else
                                               "; the torque comes mostly from the lips, whose rake follows the "
                                               "helix."))
    return out


@discussion("drilling_thrust")
def _discuss_thrust(q) -> dict:
    return _discuss(q, "thrust")


@discussion("drilling_torque")
def _discuss_torque(q) -> dict:
    return _discuss(q, "torque")


__all__ = ["drilling_thrust", "drilling_torque", "Drills", "DrillCond"]
