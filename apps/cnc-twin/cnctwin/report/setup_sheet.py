"""The setup sheet: what the operator needs at the machine, from the run directory only.

Machine and control, stock and material, per operation the workholding with its positions, the
work offsets, the program (file, checksum, title) and the operator notes (project and operation
descriptions, program stops and messages, the findings that ask for attention at the machine), the
tool list with stick-outs and offsets, the coolant, and pictures (rendered setup images when the
run has them, and the tool path over the stock drawn from frames.bin).
"""

from __future__ import annotations

import re
from pathlib import Path

from . import fmt
from .context import Context
from .html import Raw, Section, code, document, esc, figure, kv, note, table, ul
from .projectview import get, pose, ref_id, si, unwrap
from .rundir import RunData
from .sections_refs import toolpath_figure

WATCH_CATEGORIES = ("chip_packing", "chip_hazard", "coolant_shadowing", "part_slip", "part_free", "mislocation",
                    "tool_end_of_life", "tool_pullout", "unbalance", "collision", "overtravel", "burr",
                    "thermal_error")
_MESSAGE = re.compile(r"\((?P<text>[^()]*)\)")
_STOP = re.compile(r"\bM0?[01]\b")


def setup_sheet_html(run_dir_or_run) -> str:
    run = run_dir_or_run if isinstance(run_dir_or_run, RunData) else RunData(run_dir_or_run)
    ctx = Context(run)
    project = run.project
    sections = [_machine(ctx), _stock(ctx)]
    for index, op in enumerate(run.operations):
        sections.append(_operation(ctx, index, op))
    sections.append(_tools(ctx))
    sections.append(_coolant(ctx))
    sections.append(_pictures(ctx))
    header = Raw(f"<h1>Setup sheet: {esc(run.name)}</h1>"
                 f"<p class='lead'>{esc(str(project.get('description') or ''))}</p>"
                 + kv([("run", run.id), ("operations", ", ".join(str(o.get('name', '')) for o in run.operations)
                                         or fmt.DASH),
                       ("verdict", "safe to run" if run.verdict.get("safe") else "NOT SAFE TO RUN: see the report")]))
    return document(f"Setup sheet: {run.name}", sections, header=header, running_title=f"Setup sheet · {run.name}",
                    with_toc=False)


def _machine(ctx: Context) -> Section:
    machine = ctx.run.project.get("machine")
    pairs = [("machine", ref_id(machine))]
    if isinstance(machine, dict):
        for key in ("name", "maker", "model", "type"):
            if machine.get(key):
                pairs.append((key, str(machine[key])))
        controller = machine.get("controller") if isinstance(machine.get("controller"), dict) else {}
        if controller:
            pairs.append(("control", " ".join(str(controller.get(k)) for k in ("dialect", "model", "name")
                                              if controller.get(k)) or fmt.DASH))
        for spindle in machine.get("spindles") or []:
            if isinstance(spindle, dict):
                top = si(spindle, "max", "rad_s")
                pairs.append((f"spindle {spindle.get('id', '')}", f"max {fmt.rpm(top)} rpm" if top else fmt.DASH))
        for axis, spec in sorted((machine.get("axes") or {}).items()):
            travel = si(spec, "travel", "m") if isinstance(spec, dict) else None
            if isinstance(travel, list) and len(travel) == 2:
                pairs.append((f"axis {axis} travel", f"{fmt.mm(travel[0], 1)} to {fmt.mm(travel[1], 1)} mm"))
    overrides = ctx.run.project.get("controller_overrides")
    if overrides:
        pairs.append(("controller overrides", ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))))
    return Section("machine", "Machine", kv(pairs))


def _stock(ctx: Context) -> Section:
    project = ctx.run.project
    stock = project.get("stock") or {}
    size = si(stock, "size", "m") or []
    tolerance = si(stock, "tolerance", "m") or []
    sizes = " × ".join(fmt.mm(v, 2) for v in size) if size else fmt.DASH
    pairs = [("kind", str(stock.get("kind", fmt.DASH))), ("size (mm)", sizes),
             ("tolerance (± mm)", " / ".join(fmt.mm(v, 3) for v in tolerance) if tolerance else fmt.DASH),
             ("form", str(stock.get("form") or fmt.DASH))]
    if stock.get("mesh"):
        pairs.append(("stock mesh", str(stock["mesh"])))
    material = project.get("material") or {}
    material_obj = material["id"] if isinstance(material, dict) and isinstance(material.get("id"), dict) else material
    pairs.append(("material", ref_id(material_obj)))
    if isinstance(material_obj, dict):
        for key in ("name", "condition"):
            if material_obj.get(key):
                pairs.append((f"material {key}", str(material_obj[key])))
        hardness = get(material_obj, "physical", "hardness")
        if isinstance(hardness, dict):
            pairs.append(("hardness", ", ".join(f"{fmt.num(unwrap_number(v))} {k}" for k, v in hardness.items()
                                               if k not in ("src", "note", "notes") and unwrap_number(v) is not None)))
    return Section("stock", "Stock and material", kv(pairs))


def unwrap_number(value):
    return fmt.as_float(unwrap(value))


def _pose_text(mapping) -> str:
    found = pose(mapping)
    if found is None:
        return fmt.DASH
    translate, rotate, order = found
    return (f"at ({', '.join(fmt.mm(v, 3) for v in translate)}) mm, rotated "
            f"({', '.join(fmt.fixed(v * fmt.DEG_PER_RAD, 3) for v in rotate)}) deg, order {order}")


def _operation(ctx: Context, index: int, op: dict) -> Section:
    name = str(op.get("name", f"operation {index + 1}"))
    setup = op.get("setup") or {}
    holding_rows = []
    for j, entry in enumerate(setup.get("workholding") or []):
        if not isinstance(entry, dict):
            continue
        device = entry.get("device")
        details_ = [f"{k.replace('_', ' ')} {v}" for k, v in (
            ("jaw opening", _mm(si(entry, "jaw_opening", "m"))), ("input torque", _nm(si(entry, "input_torque", "Nm"))),
            ("grip depth", _mm(si(entry, "grip_depth", "m"))), ("parallels", _mm(si(entry, "parallels", "m"))),
            ("pressure", _bar(si(entry, "pressure", "Pa")))) if v]
        kind = device.get("type") if isinstance(device, dict) else None
        holding_rows.append([j + 1, ref_id(device), kind or fmt.DASH, entry.get("label") or entry.get("jaws") or fmt.DASH,
                             _pose_text(entry.get("pose")), "; ".join(details_) or fmt.DASH])
    body = "<h3>Workholding</h3>" + table(["#", "device", "type", "label", "position (workholding frame in W)",
                                           "settings"], holding_rows, numeric={0},
                                          empty="The project names no workholding for this operation.")
    pairs = [("stock pose (stock in workholding)", _pose_text(setup.get("stock_pose"))),
             ("fixture pose (workholding on the table)", _pose_text(setup.get("fixture_pose")))]
    if op.get("from_previous"):
        previous = op["from_previous"]
        pairs.append(("from the previous operation", ", ".join(f"{k}: {v}" for k, v in sorted(previous.items()))
                      if isinstance(previous, dict) else str(previous)))
    body += kv(pairs)
    offsets = si(setup, "work_offsets", "m") or {}
    rows = [[key, *[fmt.mm(v, 4) for v in (list(values) + [None] * 3)[:3]],
             ", ".join(fmt.mm(v, 4) for v in list(values)[3:]) or fmt.DASH]
            for key, values in sorted(offsets.items()) if isinstance(values, (list, tuple))]
    body += "<h3>Work offsets</h3>" + table(["offset", "X (mm)", "Y (mm)", "Z (mm)", "further axes"], rows,
                                            numeric={1, 2, 3}, caption="Machine coordinates of the work zero, as set "
                                                                       "on the machine.",
                                            empty="The project gives no work offsets for this operation.")
    body += _program_and_notes(ctx, op)
    return Section(f"op-{index}", f"Operation {name}", body)


def _mm(value) -> str:
    return f"{fmt.mm(value, 2)} mm" if fmt.as_float(value) is not None else ""


def _nm(value) -> str:
    return f"{fmt.num(value, 4)} Nm" if fmt.as_float(value) is not None else ""


def _bar(value) -> str:
    number = fmt.as_float(value)
    return f"{fmt.num(number * fmt.BAR_PER_PA, 4)} bar" if number is not None else ""


def _program_and_notes(ctx: Context, op: dict) -> str:
    name = str(op.get("name", ""))
    rows, notes = [], []
    for op_name, program, text in ctx.run.programs():
        if op_name != name:
            continue
        lines = text.splitlines() if text else []
        title = next((m.group("text").strip() for line in lines[:5] for m in _MESSAGE.finditer(line)), "")
        rows.append([program, fmt.integer(len(lines)) if text else fmt.DASH, title or fmt.DASH,
                     code(str(op.get("program_sha256", ""))[:16] or fmt.DASH)])
        for number, line in enumerate(lines, start=1):
            if _STOP.search(line.split("(")[0].upper()):
                comment = " ".join(m.group("text").strip() for m in _MESSAGE.finditer(line))
                notes.append(f"{program}:{number} {line.strip()} — program stop" + (f": {comment}" if comment else ""))
            elif "MSG" in line.upper() or "OPERATOR" in line.upper():
                notes.append(f"{program}:{number} {line.strip()}")
    out = "<h3>Program</h3>" + table(["file", "lines", "title", "sha256"], rows, numeric={1},
                                     empty="run.json records no program for this operation.")
    if op.get("description"):
        notes.insert(0, str(op["description"]))
    project_notes = ctx.run.project.get("notes")
    if project_notes:
        notes.insert(0, str(project_notes))
    for finding in ctx.run.findings:
        if finding.get("op") == name and finding.get("category") in WATCH_CATEGORIES:
            notes.append(f"Watch: {finding.get('category')} at line {finding.get('line')} "
                         f"(t = {fmt.seconds(finding.get('time_s'))} s): {finding.get('message', '')}")
    out += "<h3>Operator notes</h3>" + (ul(notes) if notes else note("No operator notes: the program has no stops "
                                                                     "or messages and no finding asks for attention."))
    return out


def _tools(ctx: Context) -> Section:
    project = ctx.run.project
    entries = [t for t in get(project, "tools", "list", default=[]) if isinstance(t, dict)]
    used = _tools_used(ctx)
    rows = []
    for entry in sorted(entries, key=lambda t: (int(t.get("path") or 0), int(t.get("number") or 0))):
        tool = entry.get("tool")
        geometry = tool.get("geometry") if isinstance(tool, dict) and isinstance(tool.get("geometry"), dict) else {}
        wear = si(entry, "wear_offsets", "m") or {}
        life = entry.get("life") if isinstance(entry.get("life"), dict) else {}
        number = entry.get("number")
        rows.append([f"T{number}", ref_id(tool), (tool.get("type") if isinstance(tool, dict) else None) or fmt.DASH,
                     fmt.mm(si(geometry, "diameter", "m"), 3), fmt.integer(geometry.get("flutes")),
                     ref_id(entry.get("holder")) or fmt.DASH, fmt.mm(si(entry, "stickout", "m"), 2),
                     fmt.mm(si(entry, "length_offset", "m"), 3), fmt.mm(si(entry, "radius_offset", "m"), 3),
                     " / ".join(fmt.mm(wear.get(k), 3) for k in ("length", "radius")) if wear else fmt.DASH,
                     fmt.um(si(entry, "runout", "m"), 1),
                     _life_text(life),
                     ", ".join(sorted(used.get(number, []))) or fmt.DASH, entry.get("comment") or fmt.DASH])
    body = table(["tool", "id", "type", "Ø (mm)", "flutes", "holder", "stick-out (mm)", "length offset (mm)",
                  "radius offset (mm)", "wear L / R (mm)", "runout (µm)", "life used", "used in", "comment"],
                 rows, numeric={3, 4, 6, 7, 8, 10}, empty="run.json lists no tools.")
    return Section("tool-list", "Tool list", body,
                   lead="Stick-out from the holder nose to the tip; offsets as set on the machine.")


def _life_text(life: dict) -> str:
    if not life:
        return fmt.DASH
    used, limit = si(life, "used", "s"), si(life, "limit", "s")
    minutes = [None if v is None else fmt.as_float(v) * fmt.MIN_PER_S for v in (used, limit)]
    return f"{fmt.num(minutes[0], 3)} of {fmt.num(minutes[1], 3)} min"


def _tools_used(ctx: Context) -> dict:
    used: dict = {}
    for block in ctx.run.timeline_blocks:
        if block.tool is not None:
            used.setdefault(block.tool, set()).add(block.op)
    return used


def _coolant(ctx: Context) -> Section:
    coolant = ctx.run.project.get("coolant") or {}
    pairs = [("fluid", ref_id(coolant.get("fluid")) or fmt.DASH)]
    concentration = si(coolant, "concentration", "frac")
    if concentration is not None:
        pairs.append(("concentration", f"{fmt.pct(concentration)} %"))
    if coolant.get("tank_fill_fraction") is not None:
        pairs.append(("tank fill", f"{fmt.pct(coolant['tank_fill_fraction'])} %"))
    for op in ctx.run.operations:
        positions = op.get("nozzle_positions")
        if positions:
            pairs.append((f"{op.get('name')}: nozzle positions",
                          ", ".join(f"T{k}: {v}" for k, v in sorted(positions.items(), key=lambda kv_: str(kv_[0])))))
    return Section("coolant-setup", "Coolant", kv(pairs))


def _pictures(ctx: Context) -> Section:
    body = toolpath_figure(ctx, "Tool path over the stock (check the work zero and the clamps)")
    images = ctx.run.images
    setup_images = [i for i in images if "setup" in Path(i[0]).name.lower()] or images
    for rel, data, mime in setup_images:
        body += figure(data, Path(rel).stem, rel, mime=mime)
    if not body:
        body = note("No pictures: the run directory has neither frames.bin nor images/.")
    return Section("setup-pictures", "Pictures", body)


__all__ = ["setup_sheet_html"]
