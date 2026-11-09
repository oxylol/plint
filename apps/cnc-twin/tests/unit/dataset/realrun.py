"""A stand-in for cnctwin.core.loop.run_project that writes a run directory in the real formats: run.json
(cnctwin.core.rundir.run_document), timeline.json (Timeline), channels.json and series/ through the real
cnctwin.core.recorder.Recorder with the real channel names and shapes the dataset reads, findings.json,
tools.json, surface.json, inspection.json, production.json and scene.json.

The numbers come from a small analytic model of the project's program (not the twin's physics): the program's
G00/G01 lines with their F and S words, the operator overrides of ``options["operator"]`` and the stick-out of
tool 1 in project.yaml, so a test can predict every value:

* spindle speed ``omega = S / 60 * 2 pi * spindle_override``; feed ``v = F / 60000 * feed_override`` (m/s);
* a feed move below Z0 cuts a slot: ``ae = D``, ``ap = -z``, ``fz = v / (n z)``;
* tick-mean force ``|F| = K_FORCE ap fz^0.8`` along x (ae/D = 1), torque ``0.6 |F| D / 2``, power ``T omega``;
* deflection ``|F| / k`` with ``k = 3 E I / L^3`` (L the stick-out); chatter index ``ap / ap_lim - 0.5`` clipped to
  [0, 1] with ``ap_lim = 2 mm (32 mm / L)^3``; a chatter finding (warning) in every cutting block whose index
  exceeds 0.3;
* edge temperature ``293.15 + 400 (vc / 5 m/s)^0.4 (fz / 50 um)^0.2``; flank wear rate ``1e-9 vc`` per second in cut.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import yaml

TICK_S = 1e-6
IPO = 1000
MACRO_IPO = 10
MACRO = IPO * MACRO_IPO
RAPID_M_S = 0.5
TOOL_CHANGE_S = 0.2
D = 0.010
Z = 3
K_FORCE = 5.0e8
E_TOOL = 600e9
ROOM_K = 293.15

# the real channels the dataset reads, with their real tiers and shapes (docs/models/*.md "Channels")
SHAPES = {
    "cutting.force_x_N": ("tick", (4,)), "cutting.force_y_N": ("tick", (4,)), "cutting.force_z_N": ("tick", (4,)),
    "cutting.torque_Nm": ("tick", (4,)), "dynamics.tool_deflection_x_m": ("tick", (4,)),
    "dynamics.tool_deflection_y_m": ("tick", (4,)), "motion.path_speed_m_s": ("tick", (4,)),
    "motion.spindle_omega_rad_s": ("tick", (4,)),
    "cutting.force_peak_N": ("macro", (4,)), "cutting.spindle_power_W": ("macro", (4,)),
    "cutting.spindle_load_fraction": ("macro", (4,)), "cutting.spindle_thermal_load": ("macro", (4,)),
    "cutting.chip_volume_rate_m3_s": ("macro", (4,)), "cutting.in_cut_fraction": ("macro", (4,)),
    "removal.ae_m": ("macro", (16,)), "removal.ap_m": ("macro", (16,)), "removal.engaged_angle_rad": ("macro", (16,)),
    "dynamics.vibration_pp_path_m": ("macro", (4,)), "dynamics.chatter_index_path": ("macro", (4,)),
    "dynamics.stability_margin": ("macro", (4,)),
    "thermal.tool_edge_max_K": ("macro", ()), "thermal.tool_flank_max_K": ("macro", ()), "chips.max_temp_K": ("macro", ()),
    "coolant.on": ("macro", ()), "coolant.flow_m3_s": ("macro", ()), "coolant.tank_temp_K": ("macro", ()),
    "wear.VB_max_m": ("macro", ()), "holding.tool_margin": ("macro", ()),
    "thermal.work_bulk_max_K": ("macro", (2,)), "thermal.work_surface_K": ("macro", (2,)),
    "thermal.spindle_temp_K": ("macro", (4,)), "holding.part_slip_m": ("macro", (8,)),
    "wear.life_used_slot": ("macro", (32,)),
}

TOOL = {"id": "helical-59766-mh35al-025-31000", "type": "square_endmill",
        "tool_material": {"class": "carbide", "id": "carbide-submicron-10co"}, "coating": {"id": "none"},
        "geometry": {"diameter_m": D, "flutes": Z, "helix_rad": math.radians(35.0), "rake_rad": math.radians(15.0),
                     "corner_radius_m": 0.0}}


def _operator(options: dict, name: str, number: int = 1) -> float:
    table = ((options or {}).get("operator") or {}).get(name) or {}
    if not isinstance(table, dict):
        return float(table)
    return float(table.get(f"T{number}", table.get("*", 1.0)))


def _project(project_path: Path) -> tuple[dict, str, float]:
    data = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    op = data["operations"][0]
    program = Path(op["program"])
    program = program if program.is_absolute() else project_path.parent / program
    stickout_mm = float(data["tools"]["list"][0].get("stickout_mm") or 32.0)
    return data, program.read_text(encoding="utf-8"), stickout_mm / 1000.0


def blocks_of(text: str, options: dict) -> list[dict]:
    """The program's blocks with their timing and cut: [{line, block, kind, tick0, tick1, v, omega, ap, cutting}]."""
    pos = np.array([0.0, 0.0, 50.0])
    feed = 0.0
    speed = 0.0
    mode = "G00"
    tool = None
    tick = 0
    out = []
    f_over = _operator(options, "feed_override")
    s_over = _operator(options, "spindle_override")
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("(")[0].strip().upper()
        if not line or line.startswith("%") or line.startswith("O"):
            continue
        words = dict(re.findall(r"([A-Z])(-?\d+\.?\d*)", line))
        if "G00" in line or re.search(r"\bG0\b", line):
            mode = "G00"
        if "G01" in line or re.search(r"\bG1\b", line):
            mode = "G01"
        if "F" in words:
            feed = float(words["F"]) / 60000.0 * f_over
        if "S" in words:
            speed = float(words["S"]) / 60.0 * 2.0 * math.pi * s_over
        if "M06" in line or "M6" in line.split():
            tool = int(words.get("T", 1))
            duration, kind = TOOL_CHANGE_S, "tool_change"
            target = pos
        elif any(axis in words for axis in "XYZ") and "G53" not in line and "G43" not in line or \
                ("G43" in line and "Z" in words):
            target = pos.copy()
            for i, axis in enumerate("XYZ"):
                if axis in words:
                    target[i] = float(words[axis])
            v = RAPID_M_S if mode == "G00" else max(feed, 1e-6)
            duration = float(np.linalg.norm(target - pos)) * 1e-3 / v
            kind = "rapid" if mode == "G00" else "linear"
        else:
            duration, kind, target = 0.001, "mode", pos
        ticks = max(1, int(math.ceil(duration / (IPO * TICK_S)))) * IPO
        depth = -min(pos[2], target[2]) * 1e-3
        cutting = kind == "linear" and depth > 0.0
        out.append({"line": number, "block": raw.strip(), "kind": kind, "tick0": tick, "tick1": tick + ticks,
                    "v": feed if kind == "linear" else (RAPID_M_S if kind == "rapid" else 0.0), "omega": speed,
                    "ap": depth if cutting else 0.0, "cutting": cutting, "tool": tool})
        tick += ticks
        pos = target
    out[-1]["tick1"] = int(math.ceil(out[-1]["tick1"] / MACRO)) * MACRO
    return out


def physics(block: dict, stickout_m: float) -> dict:
    omega, v, ap = block["omega"], block["v"], block["ap"]
    n = omega / (2.0 * math.pi)
    fz = v / (n * Z) if n > 0 and block["cutting"] else 0.0
    force = K_FORCE * ap * fz ** 0.8 if block["cutting"] else 0.0
    torque = 0.6 * force * D / 2.0
    stiffness = 3.0 * E_TOOL * (math.pi * D ** 4 / 64.0) / stickout_m ** 3
    ap_lim = 2e-3 * (0.032 / stickout_m) ** 3
    chatter = min(max(ap / ap_lim - 0.5, 0.0), 1.0) if block["cutting"] else 0.0
    vc = omega * D / 2.0
    edge = ROOM_K + (400.0 * (vc / 5.0) ** 0.4 * (fz / 50e-6) ** 0.2 if block["cutting"] else 0.0)
    return {"fz": fz, "force": force, "torque": torque, "power": torque * omega, "deflection": force / stiffness,
            "chatter": chatter, "vc": vc, "edge": edge, "mrr": D * ap * v if block["cutting"] else 0.0,
            "margin": ap_lim / ap if ap > 0 else math.nan, "wear_rate": 1e-9 * vc if block["cutting"] else 0.0}


def write_run(project_path, run_dir, options=None, progress_cb=None) -> dict:
    from cnctwin import cache
    from cnctwin.core.channels import Channel, ChannelRegistry, Tier
    from cnctwin.core.recorder import Recorder
    from cnctwin.core.rundir import (RunDirectory, Timeline, run_document, verdict, write_json_atomic)
    options = dict(options or {})
    project_path = Path(project_path)
    project_path = project_path / "project.yaml" if project_path.is_dir() else project_path
    data, text, stickout = _project(project_path)
    run_dir = Path(run_dir)
    rd = RunDirectory(run_dir, own_status=False)
    rd.prepare()
    blocks = blocks_of(text, options)
    phys = [physics(b, stickout) for b in blocks]
    end = blocks[-1]["tick1"]
    registry = ChannelRegistry()
    for name, (tier, shape) in SHAPES.items():
        registry.register(Channel(name, "-", Tier(tier), name.split(".")[0], name, shape))
    recorder = Recorder(run_dir, registry, l0_ticks_min=8, ipo_ticks=IPO, macro_ipo=MACRO_IPO, recording={"fast": "none"})
    recorder.write_channels()
    starts = np.array([b["tick0"] for b in blocks])
    vb = 20e-6
    vb_path = []
    for step in range(end // MACRO):
        t0 = step * MACRO
        ticks = np.arange(t0, t0 + MACRO, IPO, dtype=np.int64)
        idx = np.searchsorted(starts, ticks, side="right") - 1
        per = {"cutting.force_x_N": np.zeros((len(ticks), 1)), "cutting.force_y_N": np.zeros((len(ticks), 1)),
               "cutting.force_z_N": np.zeros((len(ticks), 1)), "cutting.torque_Nm": np.zeros((len(ticks), 1)),
               "dynamics.tool_deflection_x_m": np.zeros((len(ticks), 1)),
               "dynamics.tool_deflection_y_m": np.zeros((len(ticks), 1)),
               "motion.path_speed_m_s": np.zeros((len(ticks), 1)), "motion.spindle_omega_rad_s": np.zeros((len(ticks), 1))}
        for row, k in enumerate(idx):
            p = phys[k]
            per["cutting.force_x_N"][row, 0] = p["force"]
            per["cutting.torque_Nm"][row, 0] = p["torque"]
            per["dynamics.tool_deflection_x_m"][row, 0] = p["deflection"]
            per["motion.path_speed_m_s"][row, 0] = blocks[k]["v"] if blocks[k]["kind"] in ("linear", "rapid") else 0.0
            per["motion.spindle_omega_rad_s"][row, 0] = -blocks[k]["omega"]
        recorder.record_tick(ticks, per)
        ks = sorted(set(int(k) for k in idx))
        cut = [k for k in ks if blocks[k]["cutting"]]
        share = float(np.mean([blocks[k]["cutting"] for k in idx]))
        vb += sum(phys[k]["wear_rate"] for k in idx) * IPO * TICK_S
        vb_path.append(vb)
        macro = {"cutting.force_peak_N": [1.3 * max([phys[k]["force"] for k in ks])],
                 "cutting.spindle_power_W": [float(np.mean([phys[k]["power"] for k in idx]))],
                 "cutting.spindle_load_fraction": [max(phys[k]["power"] for k in ks) / 15e3],
                 "cutting.spindle_thermal_load": [0.1], "cutting.chip_volume_rate_m3_s": [float(np.mean([phys[k]["mrr"] for k in idx]))],
                 "cutting.in_cut_fraction": [share], "removal.ae_m": [D if cut else 0.0],
                 "removal.ap_m": [max([blocks[k]["ap"] for k in ks])],
                 "removal.engaged_angle_rad": [math.pi if cut else 0.0],
                 "dynamics.vibration_pp_path_m": [2.0 * max(phys[k]["deflection"] for k in ks) * 0.1],
                 "dynamics.chatter_index_path": [max(phys[k]["chatter"] for k in ks)],
                 "dynamics.stability_margin": [min([phys[k]["margin"] for k in cut]) if cut else math.nan],
                 "thermal.tool_edge_max_K": max(phys[k]["edge"] for k in ks), "thermal.tool_flank_max_K": ROOM_K + 50.0,
                 "chips.max_temp_K": ROOM_K + 100.0, "coolant.on": 1.0, "coolant.flow_m3_s": 7.5e-4,
                 "coolant.tank_temp_K": ROOM_K + 2.0, "wear.VB_max_m": vb, "holding.tool_margin": 5.0,
                 "thermal.work_bulk_max_K": [ROOM_K + 3.0], "thermal.work_surface_K": [ROOM_K + 30.0 * bool(cut)],
                 "thermal.spindle_temp_K": [ROOM_K + 5.0], "holding.part_slip_m": [0.0],
                 "wear.life_used_slot": [vb / 300e-6]}
        recorder.record_macro(t0 + MACRO, macro)
        recorder.end_step(t0, t0 + MACRO)
        if progress_cb is not None:
            progress_cb({"sim_time_s": (t0 + MACRO) * TICK_S, "est_total_s": end * TICK_S, "op": "OP10"})
    recorder.finish()
    timeline = Timeline(run_dir)
    program = Path(data["operations"][0]["program"]).name
    for b in blocks:
        i = timeline.open(path=0, op="OP10", program=program, line=b["line"], block=b["block"], kind=b["kind"],
                          tick0=b["tick0"], tool=b["tool"], part=0)
        duration = b["tick1"] - b["tick0"]
        category = {"rapid": "rapid", "tool_change": "tool_change", "mode": "auxiliary"}.get(b["kind"])
        ticks = {"cutting": duration} if b["cutting"] else ({"air_cutting": duration} if b["kind"] == "linear"
                                                            else {category: duration})
        timeline.close(i, b["tick1"], ticks=ticks)
    timeline.write()
    findings = []
    for i, (b, p) in enumerate(zip(blocks, phys)):
        if b["cutting"] and p["chatter"] > 0.3:
            findings.append({"id": len(findings), "time_s": (b["tick0"] + IPO) * TICK_S, "tick": b["tick0"] + IPO,
                             "op": "OP10", "program": program, "line": b["line"], "block": b["block"],
                             "severity": "warning", "category": "chatter",
                             "message": f"chatter index {p['chatter']:.2f}", "quantities": {"chatter_index": p["chatter"]},
                             "cause_chain": [], "domain": "dynamics"})
    write_json_atomic(run_dir / "findings.json", {"format": "cnctwin-findings", "version": 1, "findings": findings})
    cut_lines = [b["line"] for b in blocks if b["cutting"]]
    worst = max((p["fz"] for p in phys), default=0.0)
    surface = {"format": "cnctwin-surface", "version": 1,
               "features": [{"id": "slot-floor", "op": "OP10", "tool": 1, "lines": cut_lines,
                             "Ra_m": 0.25 * (worst * 1e3) ** 2 * 1e-3 / (2 * 0.0004) if worst else None,
                             "Rz_m": 4e-6, "pass": True}],
               "faces": [{"id": "face-0", "Ra_m": 1.1e-6}, {"id": "face-1", "Ra_m": 0.9e-6}],
               "burrs": [{"edge": "w0.b1.+x", "height_m": 2e-5 + 1e-3 * worst, "block": i}
                         for i, b in enumerate(blocks) if b["cutting"]][-1:]}
    write_json_atomic(run_dir / "surface.json", surface)
    depth = max((b["ap"] for b in blocks), default=0.0)
    deviation = max((p["deflection"] for p in phys), default=0.0)
    inspection = {"format": "cnctwin-inspection", "version": 1, "tolerances": [
        {"id": "slot-depth", "kind": "distance", "nominal_m": 0.0015, "plus_m": 2e-5, "minus_m": 2e-5,
         "measured_m": depth - deviation * 0.1, "deviation_m": depth - deviation * 0.1 - 0.0015,
         "pass": abs(depth - deviation * 0.1 - 0.0015) <= 2e-5}]}
    write_json_atomic(run_dir / "inspection.json", inspection)
    cut_s = sum(b["tick1"] - b["tick0"] for b in blocks if b["cutting"]) * TICK_S
    write_json_atomic(run_dir / "tools.json", {"tools": [{"slot": 0, "number": 1, "VB_m": [vb] * Z, "VB_max_m": vb,
                                                          "KT_max_m": 1e-6, "life_used_fraction": vb / 300e-6,
                                                          "cut_time_s": cut_s, "broken": False,
                                                          "max_edge_temp_K": max(p["edge"] for p in phys)}]})
    write_json_atomic(run_dir / "production.json", {"format": "cnctwin-production", "version": 1,
                                                    "cycle_time_s": end * TICK_S, "per_part_s": end * TICK_S + 69.0,
                                                    "per_part": [{"simulated": True, "energy": {"total_J": 2.0e4},
                                                                  "tool_cost_eur": 0.05}]})
    write_json_atomic(run_dir / "scene.json", {"format": "cnctwin-scene", "version": 1,
                                               "frames": {"paths": [{"path": 0, "spindle": "main"}]}})
    removed = sum(D * b["ap"] * b["v"] * (b["tick1"] - b["tick0"]) * TICK_S for b in blocks if b["cutting"])
    shaft = sum(p["power"] * (b["tick1"] - b["tick0"]) * TICK_S for b, p in zip(blocks, phys))
    project = {"name": data.get("name"), "seed": data.get("seed"),
               "machine": {"id": data.get("machine") if isinstance(data.get("machine"), str) else "machine",
                           "spindles": [{"id": "main", "max_rad_s": 1256.6370614359173}]},
               "material": {"id": "al6061-t6", "db_id": "al6061-t6", "physical": {"hardness": {"HB": 95}}},
               "tools": {"list": [{"number": 1, "tool": TOOL, "stickout_m": stickout, "runout_m": 5e-6}]}}
    summary = {"sim_time_s": end * TICK_S, "cycle_time_s": end * TICK_S, "removed_volume_m3": removed,
               "removed_mass_kg": removed * 2700.0, "chips_kg": {"chips_settled_kg": removed * 2700.0},
               "shaft_work_J": shaft, "feed_work_J": 0.01 * shaft,
               "conservation": {"mass": {"ok": True}, "energy": {"ok": True}}}
    document = run_document(run_id=run_dir.name, project=project, options=options,
                            clock={"tick_s": TICK_S, "ipo_ticks": IPO, "macro_ipo": MACRO_IPO}, verdict_=verdict(findings),
                            summary=summary, outputs=rd.outputs_index(), seed=data.get("seed"))
    document["project_hash"] = "f" * 64
    document["versions"]["build"] = cache.build_id()
    rd.write_run_json(document)
    return {"blocks": len(blocks), "end_tick": end, "verdict": document["verdict"]}


def run_project(project_path, run_dir, options, progress_cb):
    return write_run(project_path, run_dir, options, progress_cb)


def project(directory, *, stickout_mm: float = 32.0, program: str | None = None) -> Path:
    """A minimal project file this writer reads (not a resolvable project: the dataset tests need none)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "op.nc").write_text(program or SLOT, encoding="utf-8")
    data = {"format": "cnctwin-project", "version": 1, "name": "fake slot", "seed": 7, "machine": "haas-vf2ss",
            "tools": {"list": [{"number": 1, "stickout_mm": stickout_mm}]},
            "operations": [{"name": "OP10", "program": "op.nc"}]}
    (directory / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return directory / "project.yaml"


SLOT = """%
O09001 (SLOT)
G90 G17 G21 G40 G49 G80 G94 G54
T1 M06
G90 G54 G00 X-8. Y10. S10000 M03
G43 H01 Z25. M08
Z2.
G01 Z-1.5 F1000.
G01 X6. F1600.
G01 X20. F2400.
G00 Z25.
M30
%
"""
