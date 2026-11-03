"""Harness for spindle_growth: thermal's spindle model driven by the case's speed profile.

``cnctwin.thermal.machine.spindle_models`` builds the spindle body (first-order: bearing heat
P_max (omega / omega_max)^(5/3), growth alpha L_eff (T - T_ambient)) from a machine mapping; the cases'
machines are not in db/machines and their papers describe no spindle, so the mapping holds only what
the case states (a milling spindle, cold start, the test speeds, the ambient) and the model falls back
to its own defaults (db/models/thermal-machine.yaml: growth at max speed, time constant, bearing heat),
which it lists as estimated. The one input the model needs and has no default for, the maximum spindle
speed, is taken from the database's vertical machining centre (``haas-vf2ss``). Control compensation is
off (ISO 230-3 tests measure the uncompensated error). Every one of these is listed as an assumption.

Compared: the growth since the start at the reference times (curves) and the maximum (or the value at
the stated time) over the test (values). The measured Z is the whole tool-to-table loop; the model is
the spindle body only (noted).
"""

from __future__ import annotations

import math
import re

import numpy as np

from ...db.models import params
from ...thermal import machine as TM
from ...thermal.schema import settings as thermal_settings
from ..cases import from_si, to_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, not_compared
from . import common as H

REFERENCE_VMC = "haas-vf2ss"
STEP_S = 60.0


def _ambient(case, curve_name: str, col) -> float:
    """Initial ambient of a curve (stated per batch, e.g. initial_ambient_K1_degC), else the middle of the band."""
    cond = case.section("conditions")
    m = re.search(r"_(K[0-9]+)_", curve_name + "_")
    if m:
        key = f"initial_ambient_{m.group(1)}_degC"
        if key in cond:
            return float(case.si("conditions", key))
    band = case.value("conditions", "ambient_band_degC")
    if isinstance(band, (list, tuple)) and len(band) == 2:
        value = 0.5 * (float(band[0]) + float(band[1]))
        col.assume(f"ambient of {curve_name}", f"{value:.1f} degC", "initial ambient not given: middle of the case's "
                   "ambient band", "case")
        return value + 273.15
    col.assume(f"ambient of {curve_name}", "20 degC", "not stated: ISO 1 reference temperature", "est")
    return 293.15


def _model(db, ambient_K: float, col) -> TM.SpindleThermal:
    ref = H.raw_object(db, "machine", REFERENCE_VMC)
    sp = next(s for s in ref["spindles"] if (s.get("id") if not isinstance(s.get("id"), dict) else s["id"]["value"])
              == "main")
    max_rpm = sp["max_rpm"]["value"] if isinstance(sp["max_rpm"], dict) else sp["max_rpm"]
    machine = {"spindles": [{"id": "main", "kind": "milling", "max_rpm": float(max_rpm)}]}
    st = thermal_settings({})
    models = TM.spindle_models(machine, ambient_K, "cold", float(st.energy_tol), 0.0)
    model = models["main"]
    col.assume("maximum spindle speed", f"{float(max_rpm):.0f} rpm", f"not stated: the database vertical machining "
               f"centre {REFERENCE_VMC}", "db/machines")
    p = dict(params("thermal-machine"))
    for name in model.estimated:
        key = {"growth": "default_spindle_growth_m", "time_constant": "default_spindle_time_constant_s",
               "bearing_heat": "default_bearing_heat_W"}.get(name, name)
        value = p.get(key)
        col.assume(f"spindle {name.replace('_', ' ')}", f"{value:.6g} (SI)", "not stated (no spindle data in the "
                   "source): the thermal model's default", "db/models/thermal-machine.yaml")
    col.assume("thermal compensation", "off", "ISO 230-3 measures the uncompensated error", "case")
    return model


def _history(model: TM.SpindleThermal, rpm: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(t, growth since start) at STEP_S intervals; the body steps exactly (first-order)."""
    omega = np.array([rpm * 2.0 * math.pi / 60.0])
    times = [0.0]
    growth = [0.0]
    g0 = model.growth_abs_m()
    t = 0.0
    while t < duration_s - 1e-9:
        dt = min(STEP_S, duration_s - t)
        model.step(dt, omega, 0.0, 1.0, t)
        t += dt
        times.append(t)
        growth.append(model.growth_abs_m() - g0)
    return np.array(times), np.array(growth)


@harness("spindle_growth", models=("cnctwin.thermal.machine.spindle_models", "cnctwin.thermal.machine.SpindleThermal"),
         description="Thermal's spindle body (first-order, Palmgren bearing heat) at the case's constant speeds from "
                     "a cold start; the machine's spindle data are not published, so the model's own defaults and "
                     "the database VMC's maximum speed are used (listed).")
def spindle_growth(case, db) -> HarnessOutput:
    col = H.Collector()
    col.note("The measured Z error is the whole tool-to-table loop (spindle, headstock, column); the model is the "
             "spindle body only.")
    curves = []
    sims = {}
    for curve in case.reference_curves:
        name = str(curve.get("name"))
        m = re.search(r"([0-9]+)\s*rpm", name)
        rpm = float(m.group(1))
        amb = _ambient(case, name, col)
        pts = curve.get("points") or []
        horizon = max(to_si(p[0], curve.get("x_unit")) for p in pts)
        model = _model(db, amb, col)
        t, g = _history(model, rpm, horizon)
        sims[rpm] = (t, g)
        preds = []
        for x, y in pts:
            tx = to_si(x, curve.get("x_unit"))
            if tx <= 0.0:
                preds.append(not_compared("zero by definition (ISO 230-3 relative displacement)", value=0.0))
                continue
            preds.append(Prediction(from_si(float(np.interp(tx, t, g)), curve.get("y_unit")), COMPARED,
                                    f"{rpm:.0f} rpm"))
        curves.append(preds)
    values = []
    for entry in case.reference_values:
        rpm = float(entry["x"])
        name = str(entry.get("name"))
        if rpm not in sims:
            amb = _ambient(case, name, col)
            duration = case.si("conditions", "duration_min") or 7200.0
            sims[rpm] = _history(_model(db, amb, col), rpm, float(duration))
        t, g = sims[rpm]
        at = re.search(r"at_([0-9]+)min", name)
        if at:
            value = float(np.interp(float(at.group(1)) * 60.0, t, g))
            how = f"at {at.group(1)} min"
        else:
            value = float(np.max(g))
            how = f"maximum over {t[-1] / 60.0:.0f} min"
        values.append(Prediction(from_si(value, entry.get("unit")), COMPARED, f"{rpm:.0f} rpm, {how}"))
    any_model = _model(db, 293.15, H.Collector())
    settings = {"growth_at_max_speed_um": any_model.growth_max_m * 1e6, "time_constant_min":
                any_model.body.cap_J_K / any_model.body.g_W_K / 60.0, "bearing_heat_at_max_W":
                any_model.bearing_heat_max_W, "max_rpm": any_model.omega_max_rad_s * 60.0 / (2.0 * math.pi),
                "growth_length_m": any_model.length_m, "speed_exponent": any_model.exponent, "step_s": STEP_S}
    return HarnessOutput(values, curves, ["cnctwin.thermal.machine.spindle_models (first-order spindle body)"],
                         settings, col.assumptions, col.provenance, col.notes, None)


@discussion("spindle_growth")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        errs = [c.error_rel for c in r.comparisons if c.status == COMPARED and c.error_rel is not None]
        if errs:
            out[r.id] = (f"mean signed error {sum(errs) / len(errs) * 100:+.0f} %: generic spindle defaults "
                         "(20 um at the assumed 12 000 rpm, 25 min) against a measured whole-machine loop; the "
                         "heat scales as (n/n_max)^(5/3), so at 2000-6000 rpm the model's growth is a small "
                         "fraction of its default maximum.")
    out["summary"] = ("No budget is defined for spindle_growth. None of the papers describes the spindle (drive, "
                      "bearings, cooling, maximum speed), so every spindle parameter is a model default: the result "
                      "measures how far the defaults are from these two machines, not the model's physics. A case "
                      "with a named spindle and its data would test the model (GAP.md).")
    return out


__all__ = ["spindle_growth"]
