"""Rows of the dataset tables from one run directory (cnctwin.dataset.schema names every column's source)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .schema import SEVERITIES
from .series import reduce_series

NAN = math.nan


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(value) -> float:
    if isinstance(value, bool) or value is None:
        return NAN
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NAN
    return number if math.isfinite(number) else NAN


def _flag(value) -> int:
    return 1 if value is True else 0 if value is False else -1


def _get(mapping, *path):
    node = mapping
    for key in path:
        if isinstance(node, dict):
            node = node.get(key)
        elif isinstance(node, list) and isinstance(key, int) and -len(node) <= key < len(node):
            node = node[key]
        else:
            return None
    return node


@dataclass
class Source:
    """One run to export: its folder and where it comes from (a sweep point, or given directly)."""

    run_dir: Path
    sweep: str = ""
    point: int = -1
    key: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class Extracted:
    run: dict
    blocks: list[dict]
    tools: list[str]
    characteristics: list[str]
    features: list[str]


def _tool_entries(project: dict) -> dict[tuple[int, int], dict]:
    out = {}
    for entry in _get(project, "tools", "list") or []:
        if isinstance(entry, dict) and isinstance(entry.get("number"), int):
            out[(int(entry.get("path") or 1), int(entry["number"]))] = entry
    return out


def _tool_label(number: int, path: int) -> str:
    return f"T{number}" if path == 1 else f"T{number}@{path}"


def _operator_factor(operator: dict, name: str, number, path: int) -> float:
    if not isinstance(operator, dict):
        return 1.0
    table = operator.get(name)
    if not isinstance(table, dict):
        return _num(table) if table is not None else 1.0
    if number is not None and number >= 0:
        key = _tool_label(int(number), path)
        if key in table:
            return _num(table[key])
    return _num(table.get("*", 1.0))


def _spindle_of_path(run_dir: Path, project: dict) -> dict[int, int]:
    """path -> index of its spindle in the machine's spindle order (motion.spindle_omega_rad_s is per spindle)."""
    ids = [str(s.get("id")) for s in (_get(project, "machine", "spindles") or []) if isinstance(s, dict)]
    scene = read_json(run_dir / "scene.json") or {}
    out = {}
    for entry in (_get(scene, "frames", "paths") or []):
        if isinstance(entry, dict) and isinstance(entry.get("path"), int):
            spindle = str(entry.get("spindle"))
            out[int(entry["path"])] = ids.index(spindle) if spindle in ids else 0
    return out


def _findings_by_block(findings: list[dict], blocks: list[dict]) -> list[list[dict]]:
    """The findings raised in each block: its tick in the block's [tick0, tick1) (a zero-length block: at its
    tick), the block's op, and the block's line when the finding names one (a finding without a line goes to the
    first such block of its op; one that matches no block is not attributed)."""
    import bisect
    out = [[] for _ in blocks]
    by_path: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        by_path.setdefault(int(b.get("path", 0) or 0), []).append(i)
    index = {}
    for path, members in by_path.items():
        members.sort(key=lambda i: (int(blocks[i]["tick0"]), i))
        index[path] = ([int(blocks[i]["tick0"]) for i in members], members)
    for finding in findings:
        tick = finding.get("tick")
        if not isinstance(tick, int) or isinstance(tick, bool):
            continue
        candidates = []
        for starts, members in index.values():
            k = bisect.bisect_right(starts, tick) - 1
            while k >= 0:
                i = members[k]
                t0, t1 = int(blocks[i]["tick0"]), int(blocks[i]["tick1"])
                if t0 <= tick < t1 or t0 == t1 == tick:
                    candidates.append(i)
                if t0 < tick:
                    break
                k -= 1
        op, line = finding.get("op"), finding.get("line")
        chosen = None
        for i in sorted(candidates):
            block = blocks[i]
            if op and block.get("op") and op != block.get("op"):
                continue
            if line and int(line) != int(block.get("line") or 0):
                continue
            chosen = i
            break
        if chosen is not None:
            out[chosen].append(finding)
    return out


def extract(source: Source, *, blocks: bool = True) -> Extracted:
    """The run row and the block rows of one finished run (run.json must exist)."""
    run_dir = Path(source.run_dir)
    document = read_json(run_dir / "run.json") or {}
    project = document.get("project") if isinstance(document.get("project"), dict) else {}
    summary = document.get("summary") if isinstance(document.get("summary"), dict) else {}
    options = document.get("options") if isinstance(document.get("options"), dict) else {}
    operator = options.get("operator") if isinstance(options.get("operator"), dict) else {}
    clock = document.get("clock") if isinstance(document.get("clock"), dict) else {}
    findings_doc = read_json(run_dir / "findings.json")
    findings = findings_doc.get("findings", []) if isinstance(findings_doc, dict) else (findings_doc or [])
    findings = [f for f in findings if isinstance(f, dict)]
    timeline = read_json(run_dir / "timeline.json")
    records = timeline.get("blocks", []) if isinstance(timeline, dict) else (timeline or [])
    records = [b for b in records if isinstance(b, dict) and isinstance(b.get("tick0"), int)
               and isinstance(b.get("tick1"), int) and b["tick1"] >= 0]
    records.sort(key=lambda b: (int(b.get("index", 0)), int(b["tick0"])))
    channels = read_json(run_dir / "channels.json") or {}
    tick_s = _num(clock.get("tick_s")) if clock.get("tick_s") else _num((timeline or {}).get("tick_s")
                                                                          if isinstance(timeline, dict) else None)
    tick_s = tick_s if math.isfinite(tick_s) else 1e-6
    ipo = int(clock.get("ipo_ticks") or channels.get("ipo_ticks") or 1000)
    macro_ipo = int(clock.get("macro_ipo") or channels.get("macro_ipo") or 10)
    tools = _tool_entries(project)
    slots = {ident: slot for slot, ident in enumerate(tools)}
    slot_of_block = []
    for b in records:
        number = b.get("tool")
        slot_of_block.append(slots.get((int(b.get("path", 0) or 0) + 1, int(number)), -1)
                             if isinstance(number, int) else -1)
    spindle_of_path = _spindle_of_path(run_dir, project)
    paths = sorted({int(b.get("path", 0) or 0) for b in records} | {0})
    run_stats, block_stats = reduce_series(run_dir, records, paths=paths, spindle_of_path=spindle_of_path,
                                           slot_of_block=slot_of_block, macro_ticks=ipo * macro_ipo)
    tools_doc = read_json(run_dir / "tools.json") or {}
    inspection = read_json(run_dir / "inspection.json") or {}
    surface = read_json(run_dir / "surface.json") or {}
    production = read_json(run_dir / "production.json") or {}
    material = _get(project, "material", "db_id") or _get(project, "material", "id")
    material = material if isinstance(material, str) else ""
    row = {
        "sweep": source.sweep, "point": int(source.point), "run_id": str(document.get("run_id") or run_dir.name),
        "cache_key": source.key or "", "project_name": str(project.get("name") or ""),
        "project_hash": str(document.get("project_hash") or ""), "build": str(_get(document, "versions", "build") or ""),
        "seed": int(document["seed"]) if isinstance(document.get("seed"), int) else -1,
        "machine": str(_get(project, "machine", "id") or ""), "material": material,
        "complete": _flag(document.get("complete")), "safe": _flag(_get(document, "verdict", "safe")),
        "feed_override": _operator_factor(operator, "feed_override", None, 1),
        "spindle_override": _operator_factor(operator, "spindle_override", None, 1),
        "rapid_override": _num(operator.get("rapid_override", 1.0)),
        "cycle_time_s": _num(summary.get("cycle_time_s")), "sim_time_s": _num(summary.get("sim_time_s")),
        "takt_s": _num(production.get("per_part_s")), "removed_mass_kg": _num(summary.get("removed_mass_kg")),
        "removed_volume_m3": _num(summary.get("removed_volume_m3")),
        "chip_mass_kg": float(sum(_num(v) for v in summary["chips_kg"].values()))
        if isinstance(summary.get("chips_kg"), dict) and summary["chips_kg"] else NAN,
        "cutting_work_J": _num(summary.get("shaft_work_J")) + _num(summary.get("feed_work_J"))
        if summary.get("shaft_work_J") is not None else NAN,
    }
    parts = [p for p in (production.get("per_part") or []) if isinstance(p, dict) and p.get("simulated", True)]
    energies = [_num(_get(p, "energy", "total_J")) for p in parts]
    energies = [e for e in energies if math.isfinite(e)]
    row["energy_per_part_J"] = sum(energies) / len(energies) if energies else NAN
    costs = [_num(p.get("tool_cost_eur")) for p in parts]
    costs = [c for c in costs if math.isfinite(c)]
    row["tool_cost_per_part_eur"] = sum(costs) / len(costs) if costs else NAN
    g = run_stats.get
    row.update({
        "force_peak_max_N": g("force_peak", "max"), "force_max_N": g("force", "max"), "force_mean_N": g("force_cut", "mean"),
        "torque_max_Nm": g("torque", "max"), "torque_mean_Nm": g("torque_cut", "mean"),
        "spindle_power_max_W": g("power", "max"), "spindle_power_mean_W": g("power_cut", "mean"),
        "spindle_load_max": g("load", "max"), "spindle_thermal_load_max": g("thermal_load", "max"),
        "mrr_mean_m3_s": g("mrr_cut", "mean"), "tool_deflection_max_m": g("deflection", "max"),
        "vibration_pp_max_m": g("vibration", "max"), "chatter_index_max": g("chatter", "max"),
        "stability_margin_min": g("margin", "min"), "tool_edge_temp_max_K": g("edge", "max"),
        "tool_flank_temp_max_K": g("flank", "max"), "work_temp_max_K": g("work", "max"),
        "work_surface_temp_max_K": g("surface_temp", "max"), "spindle_temp_max_K": g("spindle_temp", "max"),
        "chip_temp_max_K": g("chip_temp", "max"), "coolant_tank_temp_max_K": g("tank", "max"),
        "holder_margin_min": g("holder_margin", "min"), "part_slip_max_m": g("slip", "max"),
    })
    burrs = [b for b in (surface.get("burrs") or []) if isinstance(b, dict)]
    heights = [_num(b.get("height_m")) for b in burrs]
    heights = [h for h in heights if math.isfinite(h)]
    row["burr_max_m"] = max(heights) if heights else (0.0 if surface else NAN)
    row["burr_count"] = len(burrs)
    faces = [_num(f.get("Ra_m")) for f in (surface.get("faces") or []) if isinstance(f, dict)]
    faces = [f for f in faces if math.isfinite(f)]
    row["face_Ra_max_m"] = max(faces) if faces else NAN
    row["face_Ra_mean_m"] = sum(faces) / len(faces) if faces else NAN
    conservation = summary.get("conservation") if isinstance(summary.get("conservation"), dict) else {}
    row["mass_balance_ok"] = _flag(_get(conservation, "mass", "ok"))
    row["energy_balance_ok"] = _flag(_get(conservation, "energy", "ok"))
    row["findings_count"] = len(findings)
    for severity in SEVERITIES:
        row[f"findings[{severity}]"] = sum(1 for f in findings if f.get("severity") == severity)
    for finding in findings:
        key = f"findings[{finding.get('category')}]"
        row[key] = row.get(key, 0) + 1
    for name, value in source.params.items():
        row[f"param.{name}"] = value
    tool_labels = []
    for entry in (tools_doc.get("tools") or []):
        if not isinstance(entry, dict) or not isinstance(entry.get("number"), int):
            continue
        label = _tool_label(int(entry["number"]), int(entry.get("path") or 1))
        tool_labels.append(label)
        row[f"VB_max_m[{label}]"] = _num(entry.get("VB_max_m"))
        row[f"KT_max_m[{label}]"] = _num(entry.get("KT_max_m"))
        row[f"life_used_fraction[{label}]"] = _num(entry.get("life_used_fraction"))
        row[f"cut_time_s[{label}]"] = _num(entry.get("cut_time_s"))
        row[f"edge_temp_max_K[{label}]"] = _num(entry.get("max_edge_temp_K"))
        row[f"broken[{label}]"] = _flag(entry.get("broken"))
    characteristics = []
    for item in (inspection.get("tolerances") or []):
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        ident = str(item["id"])
        characteristics.append(ident)
        row[f"measured_m[{ident}]"] = _num(item.get("measured_m"))
        row[f"deviation_m[{ident}]"] = _num(item.get("deviation_m"))
        row[f"pass[{ident}]"] = _flag(item.get("pass"))
    features = []
    for item in (surface.get("features") or []):
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        ident = str(item["id"])
        features.append(ident)
        row[f"Ra_m[{ident}]"] = _num(item.get("Ra_m"))
        row[f"Rz_m[{ident}]"] = _num(item.get("Rz_m"))
    block_rows = _block_rows(source, document, project, records, findings, surface, block_stats, tick_s, operator,
                             tools) if blocks else []
    return Extracted(row, block_rows, tool_labels, characteristics, features)


def _block_rows(source: Source, document, project, records, findings, surface, stats, tick_s: float, operator,
                tools) -> list[dict]:
    run_id = str(document.get("run_id") or Path(source.run_dir).name)
    material = _get(project, "material", "db_id") or _get(project, "material", "id")
    hardness = _num(_get(project, "material", "physical", "hardness", "HB"))
    by_block = _findings_by_block(findings, records)
    burrs = {}
    for burr in (surface.get("burrs") or []):
        if isinstance(burr, dict) and isinstance(burr.get("block"), int):
            height = _num(burr.get("height_m"))
            if math.isfinite(height):
                burrs[burr["block"]] = max(burrs.get(burr["block"], -math.inf), height)
    named = [f for f in (surface.get("features") or []) if isinstance(f, dict)]
    out = []
    for i, b in enumerate(records):
        path = int(b.get("path", 0) or 0)
        number = b.get("tool") if isinstance(b.get("tool"), int) else None
        entry = tools.get((path + 1, number), {}) if number is not None else {}
        tool = entry.get("tool") if isinstance(entry.get("tool"), dict) else {}
        geometry = tool.get("geometry") if isinstance(tool.get("geometry"), dict) else {}
        flutes = geometry.get("flutes")
        flutes = int(flutes) if isinstance(flutes, int) and not isinstance(flutes, bool) else -1
        diameter = _num(geometry.get("diameter_m"))
        ticks = b.get("ticks") if isinstance(b.get("ticks"), dict) else {}
        omega = float(stats.get("omega", "mean", i))
        speed_cut = float(stats.get("speed_cut", "mean", i))
        rev_s = omega / (2.0 * math.pi) if math.isfinite(omega) else NAN
        fz = speed_cut / (rev_s * flutes) if flutes > 0 and math.isfinite(rev_s) and rev_s > 0 else NAN
        ra = [_num(f.get("Ra_m")) for f in named if (not f.get("op") or f.get("op") == b.get("op"))
              and int(b.get("line") or -1) in [int(x) for x in (f.get("lines") or []) if isinstance(x, int)]]
        ra = [r for r in ra if math.isfinite(r)]
        found = by_block[i]
        vb = stats.items.get("vb")
        life = stats.items.get("life")
        row = {
            "sweep": source.sweep, "point": int(source.point), "run_id": run_id, "block": int(b.get("index", i)),
            "op": str(b.get("op") or ""), "program": Path(str(b.get("program") or "")).name,
            "line": int(b.get("line") or 0),
            "path": path, "part": int(b.get("part") or 0), "kind": str(b.get("kind") or ""),
            "tool": int(number) if number is not None else -1,
            "t0_s": int(b["tick0"]) * tick_s, "duration_s": (int(b["tick1"]) - int(b["tick0"])) * tick_s,
            "cutting_s": int(ticks.get("cutting", 0) or 0) * tick_s,
            "air_cutting_s": int(ticks.get("air_cutting", 0) or 0) * tick_s,
            "tool_type": str(tool.get("type") or ""), "tool_diameter_m": diameter, "tool_flutes": flutes,
            "tool_corner_radius_m": _num(geometry.get("corner_radius_m")), "tool_helix_rad": _num(geometry.get("helix_rad")),
            "tool_rake_rad": _num(geometry.get("rake_rad")),
            "tool_material_class": str(_get(tool, "tool_material", "class") or ""),
            "tool_coating": str(_get(tool, "coating", "id") or ""), "tool_stickout_m": _num(entry.get("stickout_m")),
            "tool_runout_m": _num(entry.get("runout_m")), "material": material if isinstance(material, str) else "",
            "material_hardness_HB": hardness,
            "feed_override": _operator_factor(operator, "feed_override", number, path + 1),
            "spindle_override": _operator_factor(operator, "spindle_override", number, path + 1),
            "spindle_speed_rad_s": omega, "feed_speed_m_s": stats.get("speed", "mean", i), "feed_speed_cut_m_s": speed_cut,
            "feed_per_tooth_m": fz,
            "cutting_speed_m_s": omega * diameter / 2.0 if math.isfinite(diameter) else NAN,
            "ae_m": stats.get("ae", "max", i), "ap_m": stats.get("ap", "max", i),
            "engaged_angle_rad": stats.get("angle", "max", i), "mrr_m3_s": stats.get("mrr_cut", "mean", i),
            "coolant_on": stats.get("coolant_on", "mean", i), "coolant_flow_m3_s": stats.get("coolant_flow", "mean", i),
            "force_mean_N": stats.get("force_cut", "mean", i), "force_max_N": stats.get("force", "max", i),
            "force_peak_N": stats.get("force_peak", "max", i), "torque_mean_Nm": stats.get("torque_cut", "mean", i),
            "torque_max_Nm": stats.get("torque", "max", i), "spindle_power_mean_W": stats.get("power_cut", "mean", i),
            "spindle_power_max_W": stats.get("power", "max", i), "spindle_load_max": stats.get("load", "max", i),
            "tool_edge_temp_max_K": stats.get("edge", "max", i), "work_temp_max_K": stats.get("work", "max", i),
            "VB_increment_m": float(vb.max[i] - vb.min[i]) if vb is not None and vb.count[i] else NAN,
            "life_used_increment": float(life.max[i] - life.min[i]) if life is not None and life.count[i] else NAN,
            "deflection_max_m": stats.get("deflection", "max", i), "vibration_pp_max_m": stats.get("vibration", "max", i),
            "chatter_index_max": stats.get("chatter", "max", i), "stability_margin_min": stats.get("margin", "min", i),
            "Ra_m": max(ra) if ra else NAN,
            "burr_max_m": burrs.get(int(b.get("index", i)), NAN),
            "findings_count": len(found),
            "findings_severe_count": sum(1 for f in found if f.get("severity") in ("error", "fatal")),
            "findings_categories": ";".join(sorted({str(f.get("category")) for f in found})),
        }
        out.append(row)
    return out


__all__ = ["Source", "Extracted", "extract", "read_json"]
