"""Harness for spindle_growth: thermal's spindle network and the machine structure in the measured loop.

``cnctwin.thermal.machine.spindle_models`` builds the spindle's thermal network (``cnctwin.thermal.spindle``:
Palmgren-Harris bearing friction from speed, preload, bearing size and lubricant viscosity, the drive motor's
losses, a network of shaft, bearing sets and housing with the head's mounting to the structure, the nose
growth from the shaft and housing temperatures and lengths) from a machine mapping. The cases' machines
are not in db/machines and their papers describe no spindle (drive, bearings, preload, lubrication,
cooling, motor), so the mapping holds what the case states (a milling spindle, a cold start, the test
speeds) and the network takes the model's 40-taper defaults (db/models/thermal-machine.yaml), every one
listed as an assumption. No cooling jacket is assumed (none is stated). The measured Z is the whole
tool-to-table loop: the spindle's growth toward the table minus the drift of the machine structure
(``structure_model``: the cast-iron frame, relaxing to the ambient and heated by the head through its
mounting; warmer, it lifts the head), with the structure sensitivity of the database's vertical machining
centre (``haas-vf2ss``). The ambient is the case's per-batch initial value (else the middle of its band),
rising linearly over the batch by the stated rise where the case gives one. Control compensation is off
(ISO 230-3 tests measure the uncompensated error).

Compared: the displacement since the start at the reference times (curves) and the maximum (or the value at
the stated time) over the test (values).
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


def _ambient(case, curve_name: str, col) -> tuple[float, float]:
    """(initial ambient K, rise over the batch K) of a curve: stated per batch (initial_ambient_K1_degC,
    ambient_rise_K1_dK), else the middle of the band and no rise."""
    cond = case.section("conditions")
    m = re.search(r"_(K[0-9]+)_", curve_name + "_")
    if m:
        key = f"initial_ambient_{m.group(1)}_degC"
        if key in cond:
            rise_key = f"ambient_rise_{m.group(1)}_dK"
            rise = float(case.si("conditions", rise_key)) if rise_key in cond else 0.0
            if rise:
                col.assume(f"ambient course of {curve_name}", f"linear rise of {rise:g} K over the batch",
                           "the case states the batch's rise, not its course", "case")
            return float(case.si("conditions", key)), rise
    band = case.value("conditions", "ambient_band_degC")
    if isinstance(band, (list, tuple)) and len(band) == 2:
        value = 0.5 * (float(band[0]) + float(band[1]))
        col.assume(f"ambient of {curve_name}", f"{value:.1f} degC, constant", "initial ambient not given: middle "
                   "of the case's ambient band; its course over the batch not given", "case")
        return value + 273.15, 0.0
    col.assume(f"ambient of {curve_name}", "20 degC, constant", "not stated: ISO 1 reference temperature", "est")
    return 293.15, 0.0


def _machine(db, col) -> dict:
    """The machine mapping of the case: a milling spindle without any construction data (none is stated)
    and the structure sensitivity of the database VMC."""
    ref = H.raw_object(db, "machine", REFERENCE_VMC)
    sens = (ref.get("thermal") or {}).get("structure_sensitivity_um_per_K")
    value = float(sens["value"] if isinstance(sens, dict) else sens)
    note = sens.get("note", "") if isinstance(sens, dict) else ""
    col.used(f"{REFERENCE_VMC} thermal.structure_sensitivity_um_per_K", sens.get("src", "est")
             if isinstance(sens, dict) else "est", note)
    return {"spindles": [{"id": "main", "kind": "milling"}],
            "thermal": {"structure_sensitivity_um_per_K": value}}


def _models(machine: dict, ambient_K: float):
    st = thermal_settings({})
    spindle = TM.spindle_models(machine, ambient_K, "cold", float(st.energy_tol), 0.0)["main"]
    structure = TM.structure_model(machine, ambient_K, "cold")
    return spindle, structure, float(st.energy_tol)


def _assumptions(spindle, structure, col) -> None:
    net = spindle.network
    d = net.describe()
    front, rear = d["bearings"][0], d["bearings"][-1]
    p = dict(params("thermal-machine"))
    col.assume("spindle bearings", f"front {front['count']} x angular-contact {front['bore_m'] * 1e3:.0f}/"
               f"{front['outer_m'] * 1e3:.0f} mm, {math.degrees(front['contact_angle_rad']):.0f} deg, preload "
               f"{front['preload_N']:.0f} N; rear {rear['count']} x {rear['bore_m'] * 1e3:.0f}/{rear['outer_m'] * 1e3:.0f}"
               f" mm; {front['lubrication']} (f0 {front['f0']:g}, base oil "
               f"{p['lubricant_viscosity_40C_m2_s'][0] * 1e6:g} mm2/s at 40 degC)",
               "not stated (no spindle data in the source): the model's 40-taper default arrangement",
               "db/models/thermal-machine.yaml")
    col.assume("spindle drive", f"{p['default_drive_type']}, {d['motor_noload_W']:.0f} W no-load loss above "
               f"{d['motor_base_rad_s'] * 60.0 / (2.0 * math.pi):.0f} rpm, {d['motor_to_housing']:.2f} of it into the head",
               "not stated: the model's default drive (a separate motor, as the instrumented spindle motor of the "
               "sources implies)", "db/models/thermal-machine.yaml")
    col.assume("spindle shaft and housing", f"shaft {d['shaft_d_m'] * 1e3:.0f} mm, nose to the fixed bearing "
               f"{d['nose_length_m'] * 1e3:.0f} mm; housing {d['housing_mass_kg']:.0f} kg cast iron, "
               f"{d['housing_area_m2']:.2f} m2 exposed, growth length {d['housing_length_m'] * 1e3:.0f} mm, "
               f"{d['g_struct_W_K']:.0f} W/K to the column",
               "not stated: the model's 40-taper defaults", "db/models/thermal-machine.yaml")
    col.assume("spindle cooling", "none", "not stated: no cooling jacket", "est")
    col.assume("machine structure", f"{structure.sensitivity_m_per_K * 1e6:g} um/K, "
               f"{structure.body.cap_J_K / 1e6:.2f} MJ/K, time constant "
               f"{structure.body.cap_J_K / structure.body.g_W_K / 3600.0:.1f} h",
               f"not stated: the structure sensitivity of the database VMC {REFERENCE_VMC}, the model's default "
               "machine mass", "db/machines, db/models/thermal-machine.yaml")
    col.assume("thermal compensation", "off", "ISO 230-3 measures the uncompensated error", "case")
    col.used("bearing friction factors f0, f1 (thermal-machine)", "harris-kotzalas-2007-bearings",
             "Palmgren's coefficients, recalled")


def _history(spindle, structure, tol: float, rpm: float, duration_s: float, amb0_K: float, rise_K: float):
    """(t, tool-to-table displacement since start, spindle part, structure part) at STEP_S intervals."""
    omega = np.array([rpm * 2.0 * math.pi / 60.0])
    g0 = spindle.growth_abs_m()
    d0 = structure.drift_m()
    times, total, parts_s, parts_c = [0.0], [0.0], [0.0], [0.0]
    t = 0.0
    while t < duration_s - 1e-9:
        dt = min(STEP_S, duration_s - t)
        amb = amb0_K + rise_K * min((t + 0.5 * dt) / duration_s, 1.0)
        spindle.T_env_K = amb
        spindle.step(dt, omega, 0.0, tol, t, T_struct_K=structure.body.T_K)
        structure.body.step(dt, 0.0, amb, spindle.last_to_structure_J, tol, t)
        t += dt
        grow = spindle.growth_abs_m() - g0
        lift = structure.drift_m() - d0
        times.append(t)
        total.append(grow - lift)
        parts_s.append(grow)
        parts_c.append(lift)
    return np.array(times), np.array(total), np.array(parts_s), np.array(parts_c)


@harness("spindle_growth", models=("cnctwin.thermal.spindle.build_network", "cnctwin.thermal.machine.spindle_models",
                                   "cnctwin.thermal.machine.structure_model"),
         description="Thermal's spindle network (Palmgren-Harris bearing friction with the lubricant viscosity at the "
                     "bearing temperature, motor losses, shaft / bearing sets / housing, growth from the shaft and "
                     "housing temperatures and lengths) at the case's constant speeds from a cold start, minus the "
                     "drift of the machine structure the head heats (the measured tool-to-table loop); the spindles "
                     "are not described in the papers, so the model's 40-taper defaults are used (listed).")
def spindle_growth(case, db) -> HarnessOutput:
    col = H.Collector()
    col.note("The measured Z error is the whole tool-to-table loop: simulated as the spindle network's growth toward "
             "the table (shaft over its nose length, housing over its length to the fixed bearings) minus the lift "
             "of the machine structure (column) warmed by the ambient and the head. The ISO 230-3 test mandrel's "
             "own growth and the table side's drift other than the structure's are not modelled.")
    machine = _machine(db, col)
    curves = []
    sims = {}
    for curve in case.reference_curves:
        name = str(curve.get("name"))
        rpm = float(re.search(r"([0-9]+)\s*rpm", name).group(1))
        amb, rise = _ambient(case, name, col)
        pts = curve.get("points") or []
        horizon = max(to_si(p[0], curve.get("x_unit")) for p in pts)
        spindle, structure, tol = _models(machine, amb)
        _assumptions(spindle, structure, col)
        sims[rpm] = _history(spindle, structure, tol, rpm, horizon, amb, rise)
        t, z, zs, zc = sims[rpm]
        preds = []
        for x, _ in pts:
            tx = to_si(x, curve.get("x_unit"))
            if tx <= 0.0:
                preds.append(not_compared("zero by definition (ISO 230-3 relative displacement)", value=0.0))
                continue
            preds.append(Prediction(from_si(float(np.interp(tx, t, z)), curve.get("y_unit")), COMPARED,
                                    f"{rpm:.0f} rpm: spindle {np.interp(tx, t, zs) * 1e6:.1f} um, structure lift "
                                    f"{np.interp(tx, t, zc) * 1e6:.1f} um"))
        curves.append(preds)
    values = []
    for entry in case.reference_values:
        rpm = float(entry["x"])
        name = str(entry.get("name"))
        if rpm not in sims:
            amb, rise = _ambient(case, name, col)
            duration = case.si("conditions", "duration_min") or 7200.0
            spindle, structure, tol = _models(machine, amb)
            _assumptions(spindle, structure, col)
            sims[rpm] = _history(spindle, structure, tol, rpm, float(duration), amb, rise)
        t, z, zs, zc = sims[rpm]
        at = re.search(r"at_([0-9]+)min", name)
        if at:
            tx = float(at.group(1)) * 60.0
            value = float(np.interp(tx, t, z))
            how = f"at {at.group(1)} min"
            k = int(np.argmin(np.abs(t - tx)))
        else:
            k = int(np.argmax(z))
            value = float(z[k])
            how = f"maximum over {t[-1] / 60.0:.0f} min (at {t[k] / 60.0:.0f} min)"
        values.append(Prediction(from_si(value, entry.get("unit")), COMPARED,
                                 f"{rpm:.0f} rpm, {how}: spindle {zs[k] * 1e6:.1f} um, structure lift "
                                 f"{zc[k] * 1e6:.1f} um"))
    spindle, structure, _ = _models(machine, 293.15)
    d = spindle.network.describe()
    settings = {"model": spindle.model, "front_bearings": f"{d['bearings'][0]['count']} x "
                f"{d['bearings'][0]['bore_m'] * 1e3:.0f}/{d['bearings'][0]['outer_m'] * 1e3:.0f} mm",
                "lubrication": d["bearings"][0]["lubrication"], "nose_length_mm": d["nose_length_m"] * 1e3,
                "housing_length_mm": d["housing_length_m"] * 1e3, "housing_mass_kg": d["housing_mass_kg"],
                "structure_sensitivity_um_per_K": structure.sensitivity_m_per_K * 1e6, "step_s": STEP_S}
    return HarnessOutput(values, curves, ["cnctwin.thermal.spindle (network: bearings, motor, shaft, housing)",
                                          "cnctwin.thermal.machine.structure_model (column drift)"],
                         settings, col.assumptions, col.provenance, col.notes, None)


@discussion("spindle_growth")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        errs = [c.error_rel for c in r.comparisons if c.status == COMPARED and c.error_rel is not None]
        if errs:
            mean = sum(errs) / len(errs)
            out[r.id] = (f"mean signed error {mean * 100:+.0f} %: the spindle is not described in the source, so "
                         "the network runs on the model's 40-taper defaults (bearing arrangement, grease and its "
                         "base-oil viscosity, housing mass, exposed area and growth length, no cooling); the bearing "
                         "friction (Palmgren's f0 for grease) and the housing's exposed area and growth length each "
                         "move the plateau by tens of per cent.")
    out["summary"] = (H.budget_summary(q) + "None of the papers describes the spindle (drive, bearings, preload, "
                      "lubrication, cooling), so the physical network runs on the model's 40-taper defaults: the "
                      "comparison tests the physics (speed and ambient dependence through the bearing friction and the "
                      "lubricant viscosity, the two time scales of shaft and housing, the structure's slow lift) with "
                      "generic construction data, not a described spindle. A case with a named spindle and its bearing "
                      "data would test the model directly (GAP.md).")
    return out


__all__ = ["spindle_growth"]
