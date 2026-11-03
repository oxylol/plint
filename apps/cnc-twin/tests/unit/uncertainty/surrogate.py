"""A fast stand-in for cnctwin.core.loop.run_project with a known response, for the study tests.

It reads the *variant project file* (so the tests also check that every sampled value reaches the file
the simulation would read) and writes the result files of a run directory:

* pocket depth (nominal 2 mm, +-0.05 mm) = 2 mm + (stock height - 15 mm) - (length offset - 120 mm)
  + 0.5 mm x (kc - 1), with kc the ratio of the variant's inline flow stress A to the base material's;
* floor flatness (zone 0.02 mm) = 4 um + 0.5 x runout + 2 um x (27 Nm / clamp torque - 1);
* an out_of_tolerance error finding at line 17 when the depth is outside its tolerance, and a chatter
  warning at line 13 when kc > 1.04;
* tool VB_max = 27 um x kc x hardness scale; energy per part 180 kJ x kc.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from cnctwin.db.yamlio import load_file

BASE_HEIGHT_MM = 15.0
BASE_LENGTH_OFFSET_MM = 120.0
BASE_TORQUE_NM = 27.0
DEPTH_M = 0.002
DEPTH_TOL_M = 5e-5
FLAT_TOL_M = 2e-5


def response(data: dict, base: dict) -> dict:
    tool = data["tools"]["list"][0]
    material = data["material"]["id"]
    kc = 1.0
    hardness = 1.0
    if isinstance(material, dict):
        if "flow_stress" in material:
            kc = float(material["flow_stress"]["A_MPa"]) / float(base["A_MPa"])
        if "physical" in material and "hardness" in material["physical"]:
            hardness = float(material["physical"]["hardness"]["HB"]) / float(base["HB"])
    torque = float(data["operations"][0]["setup"]["workholding"][0]["input_torque_Nm"])
    depth = DEPTH_M + (float(data["stock"]["size_mm"][2]) - BASE_HEIGHT_MM) * 1e-3 \
        - (float(tool["length_offset_mm"]) - BASE_LENGTH_OFFSET_MM) * 1e-3 + 5e-4 * (kc - 1.0)
    flatness = 4e-6 + 0.5 * float(tool["runout_um"]) * 1e-6 + 2e-6 * (BASE_TORQUE_NM / torque - 1.0)
    return {"depth_m": depth, "flatness_m": flatness, "kc": kc, "hardness": hardness}


@functools.lru_cache(maxsize=1)
def base_from_db() -> dict:
    """The base material's flow stress A and hardness (when the caller does not pass them)."""
    from cnctwin.db import Database
    raw = Database.load().raw("material", "al6061-t6")

    def number(value):
        return float(value["value"] if isinstance(value, dict) else value)

    return {"A_MPa": number(raw["flow_stress"]["A_MPa"]), "HB": number(raw["physical"]["hardness"]["HB"])}


def run(project_path, run_dir, options, progress_cb):
    data = load_file(project_path).data
    if options.get("fail_samples") and int(options["uncertainty"]["sample"]) in options["fail_samples"]:
        raise RuntimeError("surrogate failure on request")
    values = response(data, options.get("base") or base_from_db())
    run_dir = Path(run_dir)
    progress_cb({"sim_time_s": 1.0, "est_total_s": 1.0, "op": "OP10"})
    depth_ok = abs(values["depth_m"] - DEPTH_M) <= DEPTH_TOL_M
    findings = []
    if values["kc"] > 1.04:
        findings.append({"id": len(findings), "category": "chatter", "severity": "warning", "message": "chatter",
                         "time_s": 1.2, "tick": 1200000, "domain": "dynamics", "op": "OP10", "program": "op10.nc",
                         "line": 13, "quantities": {"chatter_index": values["kc"] - 1.0}, "cause_chain": []})
    if not depth_ok:
        findings.append({"id": len(findings), "category": "out_of_tolerance", "severity": "error",
                         "message": f"pocket-depth {values['depth_m'] * 1e3:.4f} mm", "time_s": 2.0, "tick": 2000000,
                         "domain": "inspect", "op": "OP10", "program": "op10.nc", "line": 17, "quantities": {},
                         "cause_chain": []})
    reasons = [f"out_of_tolerance (error) at OP10 line 17: {f['message']}" for f in findings
               if f["severity"] == "error"]
    files = {
        "run.json": {"format": "cnctwin-run", "version": 1, "project": {"name": data["name"]},
                     "verdict": {"safe": not reasons, "reasons": reasons}, "clock": {"tick_s": 1e-6},
                     "summary": {"cycle_time_s": 2.32, "removed_mass_kg": 2.5e-3}},
        "findings.json": {"format": "cnctwin-findings", "version": 1, "findings": findings},
        "inspection.json": {"format": "cnctwin-inspection", "version": 1, "temperature_K": 293.15, "tolerances": [
            {"id": "pocket-depth", "kind": "distance", "nominal_m": DEPTH_M, "plus_m": DEPTH_TOL_M, "minus_m": DEPTH_TOL_M,
             "measured_m": values["depth_m"], "deviation_m": values["depth_m"] - DEPTH_M, "pass": depth_ok},
            {"id": "floor-flatness", "kind": "flatness", "tolerance_m": FLAT_TOL_M, "measured_m": values["flatness_m"],
             "pass": values["flatness_m"] <= FLAT_TOL_M}]},
        "tools.json": {"format": "cnctwin-tools", "version": 1, "tools": [
            {"slot": 0, "number": 1, "tool_id": "t", "VB_max_m": 2.7e-5 * values["kc"] * values["hardness"],
             "life_used_fraction": 4e-4 * values["kc"], "cut_time_s": 1.52}]},
        "production.json": {"format": "cnctwin-production", "version": 1, "per_part": [
            {"part": 0, "energy": {"total_J": 1.8e5 * values["kc"]}, "tool_cost_eur": 0.012 * values["kc"]}]},
    }
    for name, document in files.items():
        (run_dir / name).write_text(json.dumps(document), encoding="utf-8")
    return {"verdict": files["run.json"]["verdict"]}
