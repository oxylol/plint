"""Report sections from production.json and tools.json: cycle time, energy, consumables and cost per
part, tool wear and remaining life (docs/models/production.md for the meanings of every number)."""

from __future__ import annotations

import numpy as np

from . import fmt, plots
from .context import Context
from .html import Raw, Section, code, figure, join, kv, note, para, table, ul
from .rundir import split_column

TOP_LINES = 30     # presentation: the longest program lines listed in the cycle-time section


# ----------------------------------------------------------------------------- cycle time

def _breakdown_rows(breakdown: dict, tick_s: float) -> tuple[list, int]:
    ticks = breakdown.get("ticks") or {}
    total = int(breakdown.get("total_ticks") or sum(int(v) for v in ticks.values()))
    rows = []
    for category, count in sorted(ticks.items(), key=lambda kv_: (-int(kv_[1]), kv_[0])):
        rows.append([category, fmt.seconds(int(count) * tick_s), fmt.pct(int(count) / total if total else None),
                     fmt.integer(count)])
    return rows, total


def _cycle_from_timeline(ctx: Context) -> dict | None:
    """The first part's cycle computed from timeline.json with cnctwin.production (when production.json
    is absent): the same exact integer bookkeeping."""
    blocks = ctx.run.timeline_blocks
    if not blocks:
        return None
    from ..production.cycle import cycle_time, cycle_to_dict, lines_to_dict
    part = min(int(b.part) for b in blocks)
    cycle = cycle_time(blocks, part)
    return {"cycle": cycle_to_dict(cycle, ctx.run.tick_s), "lines": lines_to_dict(cycle, ctx.run.tick_s)}


def cycle_section(ctx: Context) -> Section:
    run = ctx.run
    production = run.production or {}
    first = production.get("first_part") or {}
    source = "production.json"
    if not first:
        first = _cycle_from_timeline(ctx) or {}
        source = "timeline.json (production.json absent)"
    cycle = first.get("cycle") or {}
    breakdown = cycle.get("breakdown") or {}
    tick_s = float(production.get("tick_s") or run.tick_s)
    if not breakdown:
        return Section("cycle", "Cycle time", note("Neither production.json nor timeline.json gives the block "
                                                    "times: no cycle-time breakdown."))
    rows, total = _breakdown_rows(breakdown, tick_s)
    exact = sum(int(v) for v in (breakdown.get("ticks") or {}).values()) == total
    body = para(f"Machine time of the part: {fmt.duration(total * tick_s)} ({fmt.integer(total)} base ticks). "
                + ("The categories sum to it exactly (integer ticks)." if exact else
                   "The categories do not sum to the total: see production.json."))
    body += table(["category", "time (s)", "share (%)", "ticks"], rows, numeric={1, 2, 3},
                  caption=f"Breakdown of the first part ({source})")
    labels = [r[0] for r in rows]
    values = [int((breakdown.get("ticks") or {})[c]) * tick_s for c in labels]
    body += figure(plots.hbar(labels, values, "s", "Cycle time by category",
                              value_labels=[f"{v:.3f} s" for v in values]),
                   "cycle time by category", "Where the machine time goes (same numbers as the table).")
    body += _operations_table(cycle, tick_s) + _tools_table(cycle, tick_s) + _lines_table(ctx, first, tick_s)
    body += _batch(production)
    stops = cycle.get("stops") or {}
    if stops:
        body += para("Program stops: " + ", ".join(f"{k} × {v}" for k, v in sorted(stops.items())))
    return Section("cycle", "Cycle time", body, lead="Exact per category, operation, tool and program line.")


def _operations_table(cycle: dict, tick_s: float) -> str:
    rows = []
    for op in cycle.get("operations") or []:
        ticks = (op.get("breakdown") or {}).get("ticks") or {}
        rows.append([op.get("op", ""), fmt.seconds(int(op.get("span_ticks") or 0) * tick_s),
                     f"path {op.get('critical_path', 0)}", fmt.seconds(int(ticks.get("cutting", 0)) * tick_s),
                     fmt.seconds(int(ticks.get("air_cutting", 0)) * tick_s), fmt.seconds(int(ticks.get("rapid", 0)) * tick_s),
                     len(op.get("paths") or [])])
    if not rows:
        return ""
    return "<h4>Per operation</h4>" + table(["operation", "span (s)", "critical", "cutting (s)", "air cutting (s)",
                                             "rapid (s)", "paths"], rows, numeric={1, 3, 4, 5, 6})


def _tools_table(cycle: dict, tick_s: float) -> str:
    rows = []
    for entry in cycle.get("tools") or []:
        breakdown = entry.get("breakdown") or {}
        ticks = breakdown.get("ticks") or {}
        total = int(breakdown.get("total_ticks") or 0)
        tool = entry.get("tool")
        rows.append(["no tool" if tool is None else f"T{tool}", fmt.seconds(total * tick_s),
                     fmt.seconds(int(ticks.get("cutting", 0)) * tick_s),
                     fmt.pct(int(ticks.get("cutting", 0)) / total if total else None)])
    if not rows:
        return ""
    return "<h4>Per tool</h4>" + table(["tool", "time (s)", "cutting (s)", "in cut (%)"], rows, numeric={1, 2, 3})


def _lines_table(ctx: Context, first: dict, tick_s: float) -> str:
    lines = [entry for entry in first.get("lines") or [] if isinstance(entry, dict)]
    if not lines:
        return ""
    ranked = sorted(lines, key=lambda e: (-int((e.get("breakdown") or {}).get("total_ticks") or 0),
                                          str(e.get("op")), int(e.get("line") or 0)))
    rows = []
    for entry in ranked[:TOP_LINES]:
        breakdown = entry.get("breakdown") or {}
        ticks = breakdown.get("ticks") or {}
        main = ", ".join(f"{k} {fmt.seconds(int(v) * tick_s)}" for k, v in
                         sorted(ticks.items(), key=lambda kv_: -int(kv_[1]))[:2])
        rows.append([ctx.line_link(entry.get("op", ""), entry.get("program", ""), int(entry.get("line") or 0)),
                     code(entry.get("block", "")), fmt.integer(entry.get("executions")),
                     fmt.seconds(int(breakdown.get("total_ticks") or 0) * tick_s), main])
    caption = (f"The {min(TOP_LINES, len(lines))} longest of {len(lines)} program lines"
               if len(lines) > TOP_LINES else f"All {len(lines)} program lines, longest first")
    return "<h4>Per program line</h4>" + table(["line", "block", "runs", "time (s)", "main categories"], rows,
                                               numeric={2, 3}, caption=caption)


def _batch(production: dict) -> str:
    batch = production.get("batch") or {}
    if not batch:
        return ""
    per_part = (production.get("per_part") or [{}])[0]
    operator = per_part.get("operator") or {}
    pairs = [("parts planned / simulated / projected",
              f"{production.get('parts_planned', fmt.DASH)} / {production.get('parts_simulated', fmt.DASH)} / "
              f"{production.get('parts_projected', fmt.DASH)}"),
             ("takt per part (mean)", fmt.duration(batch.get("takt_mean_s"))),
             ("parts per hour", fmt.num(batch.get("parts_per_hour"), 4)),
             ("machine utilisation", f"{fmt.pct(batch.get('machine_utilisation_frac'))} %"),
             ("batch time", fmt.duration(batch.get("total_s")))]
    for name, value in sorted((operator.get("components_s") or {}).items()):
        pairs.append((f"operator: {name.replace('_', ' ')}", fmt.duration(value)))
    if operator:
        pairs.append(("machine waiting for the operator", fmt.duration(operator.get("machine_wait_s"))))
    return "<h4>Batch and takt</h4>" + kv(pairs)


# ----------------------------------------------------------------------------- energy, consumables, cost

CONSUMERS = ("spindle", "axes", "servo_standby", "pumps", "conveyor", "base", "compressed_air")


def _consumers(energy: dict) -> dict[str, float]:
    pumps = sum(float((p or {}).get("electrical_J") or 0.0) for p in (energy.get("pumps") or {}).values())
    return {"spindle": float((energy.get("spindle") or {}).get("electrical_J") or 0.0),
            "axes": float((energy.get("axes") or {}).get("electrical_J") or 0.0),
            "servo_standby": float(energy.get("servo_standby_J") or 0.0), "pumps": pumps,
            "conveyor": float((energy.get("conveyor") or {}).get("electrical_J") or 0.0),
            "base": float(energy.get("base_J") or 0.0), "compressed_air": float(energy.get("air_J") or 0.0)}


def energy_section(ctx: Context) -> Section:
    production = ctx.run.production
    if not production or not production.get("per_part"):
        return Section("energy", "Energy, consumables and cost per part",
                       note("production.json is not in the run directory: no energy, consumables or cost."))
    parts = production["per_part"]
    body = ""
    for record in parts[: min(len(parts), 3)]:
        body += _part_energy(record)
    if len(parts) > 3:
        body += note(f"{len(parts) - 3} more parts in production.json (per_part).")
    batch = production.get("batch") or {}
    if batch:
        pairs = [("energy per part (batch mean)", f"{fmt.kwh(batch.get('energy_per_part_J'))} kWh"),
                 ("energy of the batch", f"{fmt.kwh(batch.get('energy_J'))} kWh"),
                 ("tool cost per part (batch)", f"{fmt.eur(batch.get('tool_cost_per_part_eur'))} EUR"),
                 ("tool changes in the batch", fmt.integer(batch.get("tool_changes"))),
                 ("coolant concentrate cost (batch)", f"{fmt.eur((batch.get('coolant') or {}).get('cost_eur'))} EUR")]
        body += "<h4>Batch</h4>" + kv(pairs)
    body += para("Electricity cost is not given: the run directory holds no electricity price, so energy is "
                 "reported in kWh.")
    assumptions = production.get("assumptions") or []
    if assumptions:
        body += "<h4>Assumptions of the production metrics</h4>" + ul([str(a) for a in assumptions])
    return Section("energy", "Energy, consumables and cost per part", body,
                   lead="Electrical energy by consumer, coolant and air used, and the cost of tool edges and "
                        "coolant per part.")


def _part_energy(record: dict) -> str:
    energy = record.get("energy") or {}
    consumers = _consumers(energy)
    total = float(energy.get("total_J") or sum(consumers.values()))
    rows = [[name.replace("_", " "), fmt.kj(value), fmt.kwh(value), fmt.pct(value / total if total else None)]
            for name, value in sorted(consumers.items(), key=lambda kv_: -kv_[1])]
    rows.append([Raw("<b>total</b>"), fmt.kj(total), fmt.kwh(total), "100.0"])
    label = f"Part {int(record.get('part', 0)) + 1}" + ("" if record.get("simulated", True) else " (projected)")
    out = f"<h3>{label}</h3>" + table(["consumer", "energy (kJ)", "energy (kWh)", "share (%)"], rows,
                                      numeric={1, 2, 3}, caption="Electrical energy per part")
    ordered = sorted(consumers.items(), key=lambda kv_: -kv_[1])
    out += figure(plots.hbar([k.replace("_", " ") for k, _ in ordered], [v / 1e3 for _, v in ordered], "kJ",
                             "Electrical energy by consumer",
                             value_labels=[f"{v / 1e3:.1f} kJ" for _, v in ordered]),
                  "energy by consumer", "The same numbers as the table above.")
    spindle = energy.get("spindle") or {}
    if spindle:
        out += table(["spindle term", "energy (kJ)"],
                     [[k.replace("_J", "").replace("_", " "), fmt.kj(v)] for k, v in spindle.items()], numeric={1},
                     caption="Spindle: shaft work, run-up, braking, motor loss, idle, recovered")
    if energy.get("missing"):
        out += note("Not included (production.json energy.missing): " + "; ".join(str(m) for m in energy["missing"]))
    out += _consumables(record)
    out += _costs(record)
    return out


def _consumables(record: dict) -> str:
    consumables = record.get("consumables") or {}
    coolant = consumables.get("coolant") or {}
    air = consumables.get("air") or {}
    pairs = []
    if coolant:
        pairs += [("coolant kind", str(coolant.get("kind", fmt.DASH))),
                  ("pumped", f"{fmt.litres(coolant.get('pumped_m3'))} l"),
                  ("drag-out on chips", f"{fmt.litres(coolant.get('drag_out_m3'))} l"),
                  ("mist loss", f"{fmt.litres(coolant.get('mist_m3'))} l"),
                  ("water evaporated", f"{fmt.litres(coolant.get('evaporated_water_m3'))} l"),
                  ("concentrate used", f"{fmt.litres(coolant.get('concentrate_m3'))} l"),
                  ("make-up water", f"{fmt.litres(coolant.get('water_m3'))} l"),
                  ("oil used", f"{fmt.litres(coolant.get('oil_m3'))} l"),
                  ("coolant cost", f"{fmt.eur(coolant.get('cost_eur'))} EUR")]
    if air:
        pairs += [("compressed air, machine", f"{fmt.litres(air.get('machine_m3'))} l (free air)"),
                  ("compressed air, air blast", f"{fmt.litres(air.get('air_blast_m3'))} l"),
                  ("compressed air, MQL", f"{fmt.litres(air.get('mql_m3'))} l"),
                  ("compressed air, total", f"{fmt.litres(air.get('total_m3'))} l")]
    for pump, seconds in sorted((consumables.get("pump_on_s") or {}).items()):
        pairs.append((f"pump {pump} running", fmt.duration(seconds)))
    if record.get("removed_mass_kg") is not None:
        pairs.append(("chip mass (removed material)", f"{fmt.grams(record.get('removed_mass_kg'))} g"))
    if not pairs:
        return note("production.json gives no consumables for this part.")
    out = "<h4>Consumables</h4>" + kv(pairs)
    if coolant.get("missing"):
        out += note("Coolant terms not included: " + "; ".join(str(m) for m in coolant["missing"]))
    return out


def _costs(record: dict) -> str:
    rows = []
    for tool in record.get("tools") or []:
        cost = tool.get("cost") or {}
        rows.append([f"T{tool.get('number')}", tool.get("tool_id", ""), fmt.eur(cost.get("price_eur")),
                     fmt.integer(cost.get("regrinds")), fmt.eur(cost.get("cost_per_edge_eur")),
                     fmt.num(cost.get("parts_per_edge"), 4), fmt.eur(cost.get("cost_per_part_eur"), 4),
                     "; ".join(str(m) for m in cost.get("missing") or []) or fmt.DASH])
    coolant_cost = ((record.get("consumables") or {}).get("coolant") or {}).get("cost_eur")
    out = "<h4>Cost per part</h4>" + table(["tool", "id", "price (EUR)", "regrinds", "per edge (EUR)",
                                            "parts per edge", "per part (EUR)", "missing"], rows,
                                           numeric={2, 3, 4, 5, 6}, empty="No tool costs in production.json.")
    tool_cost = record.get("tool_cost_eur")
    total = sum(v for v in (tool_cost, coolant_cost) if isinstance(v, (int, float)))
    out += kv([("tool edges", f"{fmt.eur(tool_cost)} EUR"), ("coolant concentrate", f"{fmt.eur(coolant_cost)} EUR"),
               ("consumables total (without electricity)", f"{fmt.eur(total)} EUR")])
    return out


# ----------------------------------------------------------------------------- tools

def tools_section(ctx: Context) -> Section:
    run = ctx.run
    document = run.tools
    if not document or not document.get("tools"):
        return Section("tools", "Tool wear and remaining life",
                       note("tools.json is not in the run directory: no wear or tool life."))
    life_rates = _life_rates(run.production or {})
    rows, charts = [], ""
    for tool in sorted(document["tools"], key=lambda t: (int(t.get("number", 0)), int(t.get("slot", 0)))):
        number = tool.get("number")
        used = fmt.as_float(tool.get("life_used_fraction"))
        remaining_frac = None if used is None else max(0.0, 1.0 - used)
        remaining_s, remaining_parts = _remaining(tool, remaining_frac, life_rates.get(number))
        vb = [v for v in (tool.get("VB_m") or []) if fmt.as_float(v) is not None]
        rows.append([f"T{number}", tool.get("tool_id", ""), tool.get("holder_id", fmt.DASH),
                     fmt.seconds(tool.get("cut_time_s")), fmt.um(tool.get("VB_max_m")),
                     fmt.um(tool.get("VB_limit_m")), fmt.pct(used, 2), fmt.pct(remaining_frac, 2),
                     fmt.duration(remaining_s), fmt.num(remaining_parts, 4),
                     "BROKEN" if tool.get("broken") else "no", fmt.degc(tool.get("max_edge_temp_K"))])
        if vb:
            limit = fmt.as_float(tool.get("VB_limit_m"))
            charts += figure(plots.tooth_bars([v * fmt.UM_PER_M for v in vb],
                                              None if limit is None else limit * fmt.UM_PER_M,
                                              f"T{number} flank wear VB per tooth"),
                             f"T{number} wear per tooth", f"T{number}: wear land width at the end of the run.")
    body = table(["tool", "id", "holder", "cut (s)", "VB max (µm)", "VB limit (µm)", "life used (%)",
                  "life left (%)", "cutting left", "parts left", "broken", "max edge (°C)"], rows,
                 numeric={3, 4, 5, 6, 7, 9, 11})
    body += note("Life left in cutting time and parts is the remaining edge life at this run's rate of life use "
                 "(tools.json remaining_life_s when the wear model gives it); a new edge or regrind restarts it.")
    body += charts + _wear_series(ctx)
    flagged = [f for f in run.findings if f.get("category") in ("tool_end_of_life", "tool_breakage", "tool_pullout")]
    if flagged:
        body += "<h4>Tool findings</h4>" + ul([join([ctx.finding_link(f["id"]), " ", f.get("message", "")])
                                                for f in flagged])
    return Section("tools", "Tool wear and remaining life", body,
                   lead="Flank wear per tooth, life used by this run and what is left of the edge.")


def _life_rates(production: dict) -> dict:
    """Per tool number: (life used per part, cutting seconds per part) of the first simulated part."""
    rates = {}
    for record in (production.get("per_part") or [])[:1]:
        for tool in record.get("tools") or []:
            rates[tool.get("number")] = (fmt.as_float(tool.get("life_used_part_frac")),
                                         fmt.as_float(tool.get("cut_time_s")))
    return rates


def _remaining(tool: dict, remaining_frac, rate) -> tuple[float | None, float | None]:
    remaining_s = fmt.as_float(tool.get("remaining_life_s"))
    per_part, cut_s = rate if rate else (None, None)
    if remaining_s is None and remaining_frac is not None and per_part and cut_s:
        remaining_s = remaining_frac * cut_s / per_part
    remaining_parts = remaining_frac / per_part if remaining_frac is not None and per_part else None
    return remaining_s, remaining_parts


def _column_order(column: str):
    name, index, stat = split_column(column)
    numbers = [int(n) for n in index.strip("[]").split("][") if n] if index else []
    return name, numbers, stat


def _wear_series(ctx: Context) -> str:
    macro = ctx.tier("macro")
    if macro is None:
        return ""
    panels = []
    for channel in ctx.run.channels:
        if channel.get("domain") != "wear" or channel.get("tier") != "macro":
            continue
        columns = sorted((c for c in macro.columns if split_column(c)[0] == channel["name"]
                          and macro.columns[c].finite), key=_column_order)
        if not columns:
            continue
        shown = fmt.display_unit(channel.get("unit", ""), max(macro.columns[c].max_abs for c in columns))
        lines = []
        for column in columns[:plots.MAX_SERIES]:
            t_s, mean, _, _ = macro.curve(column)
            lines.append(plots.Line(column[len(channel["name"]):] or channel["name"], t_s, shown(mean)))
        panels.append(plots.Panel(channel["name"], shown.label, lines,
                                  note=f"first {plots.MAX_SERIES} of {len(columns)} columns"
                                  if len(columns) > plots.MAX_SERIES else ""))
    png = plots.series_panels(panels)
    return figure(png, "wear over time", "Wear channels over the run (macro tier).") if png else ""


__all__ = ["cycle_section", "energy_section", "tools_section"]
