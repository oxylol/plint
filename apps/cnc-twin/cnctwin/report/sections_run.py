"""Report sections about the run as a whole: the verdict, the findings with their chains of cause,
the program listing the findings link to, the run's identity, and what the run directory lacks."""

from __future__ import annotations

from pathlib import Path

from . import fmt
from .context import CONTEXT_LINES, FULL_LISTING_MAX_LINES, SEVERITY_RANK, Context, program_key
from .html import (Raw, Section, anchor, code, details, esc, join, kv, link, note, para, severity_badge, table,
                   tiles, ul)


# ----------------------------------------------------------------------------- verdict

def verdict_header(ctx: Context) -> Raw:
    """The first thing on the page: safe to run or not, and why, with the headline numbers."""
    run = ctx.run
    verdict = run.verdict
    safe = bool(verdict.get("safe"))
    reasons = [str(r) for r in verdict.get("reasons", [])]
    status = run.status.get("status") if run.status else None
    unfinished = not run.finished
    title = "SAFE TO RUN" if safe else "NOT SAFE TO RUN"
    icon = "✓" if safe else "✕"
    errors = [f for f in run.findings if f.get("severity") in ("error", "fatal")]
    warnings = [f for f in run.findings if f.get("severity") == "warning"]
    why = []
    if unfinished:
        why.append(para(f"The run has not finished (status {status or 'unknown'}): there is no run.json, so "
                         "this verdict covers only what was simulated so far."))
    if safe:
        why.append(para("No finding reaches severity error or fatal (architecture 11)."
                        + (f" {len(warnings)} warning(s) to review below." if warnings else "")))
    else:
        items = []
        by_reason = {_reason_key(f): f for f in errors}
        for reason in reasons or [_reason_text(f) for f in errors]:
            finding = by_reason.get(reason)
            items.append(join([ctx.finding_link(finding["id"], f"#{finding['id']}"), " ", reason])
                         if finding is not None else reason)
        why.append(para("Why:"))
        why.append(ul(items))
    banner = Raw(f'<div class="verdict {"safe" if safe else "unsafe"}"><div class="verdict-title">'
                 f'<span class="verdict-icon">{icon}</span>{esc(title)}</div>{"".join(why)}</div>')
    project = run.project
    subtitle = " · ".join(str(x) for x in (
        f"run {run.id}", _ref_id(project.get("machine")), _ref_id((project.get("material") or {}).get("id")
                                                                  if isinstance(project.get("material"), dict)
                                                                  else project.get("material")),
        f"seed {run.run.get('seed', project.get('seed', fmt.DASH))}") if x)
    head = Raw(f"<h1>{esc(run.name)}</h1><p class='lead'>{esc(subtitle)}</p>")
    return Raw(head + banner + _headline_tiles(ctx, errors, warnings))


def _reason_text(finding: dict) -> str:
    return (f"{finding['category']} ({finding['severity']}) at {finding.get('op', '')} line {finding.get('line', 0)}: "
            f"{finding['message']}")


def _reason_key(finding: dict) -> str:
    return _reason_text(finding)


def _ref_id(value) -> str:
    if isinstance(value, dict):
        return str(value.get("id") or value.get("name") or "")
    return str(value or "")


def _headline_tiles(ctx: Context, errors: list, warnings: list) -> Raw:
    run = ctx.run
    production = run.production or {}
    per_part = (production.get("per_part") or [{}])[0] if production.get("per_part") else {}
    energy = (per_part.get("energy") or {}).get("total_J")
    tool_cost = per_part.get("tool_cost_eur")
    items = [("cycle time", fmt.duration(run.cycle_time_s), "machine time of the part"),
             ("findings", str(len(run.findings)), f"{len(errors)} error, {len(warnings)} warning")]
    if production.get("batch"):
        items.append(("takt", fmt.duration(production["batch"].get("takt_mean_s")), "with operator time"))
    if energy is not None:
        items.append(("energy per part", f"{fmt.kwh(energy)} kWh", "electrical, all consumers"))
    if tool_cost is not None:
        items.append(("tool cost per part", f"{fmt.eur(tool_cost)} EUR", "edges and regrinds"))
    inspection = run.inspection or {}
    tolerances = [t for t in inspection.get("tolerances", []) if isinstance(t, dict)]
    if tolerances:
        failed = sum(1 for t in tolerances if t.get("pass") is False)
        items.append(("characteristics", f"{len(tolerances) - failed}/{len(tolerances)} pass",
                      "inspected at 20 °C"))
    return tiles(items)


# ----------------------------------------------------------------------------- findings

def findings_section(ctx: Context) -> Section:
    run = ctx.run
    program = program_section(ctx)          # first: it decides which program lines can be linked to
    findings = ctx.shown_findings
    rows, classes, ids = [], [], []
    for finding in findings:
        where = ctx.line_link(finding.get("op", ""), finding.get("program", ""), int(finding.get("line") or 0),
                              _where_text(finding))
        quantities = "; ".join(fmt.quantity(k, v) for k, v in sorted((finding.get("quantities") or {}).items()))
        position = finding.get("pos_W_m")
        rows.append([finding.get("id"), severity_badge(finding.get("severity", "info")), finding.get("category", ""),
                     fmt.seconds(finding.get("time_s")), where, code(finding.get("block", "")),
                     finding.get("message", ""), quantities,
                     " ".join(fmt.mm(v, 3) for v in position) if position else fmt.DASH,
                     link(f"#{anchor('C', finding.get('id'))}", f"{len(finding.get('cause_chain') or [])} events")])
        classes.append(f"hit-{finding.get('severity', 'info')}")
        ids.append(ctx.finding_anchor(finding.get("id")))
    counts = {s: sum(1 for f in run.findings if f.get("severity") == s) for s in ("fatal", "error", "warning", "info")}
    lead = ", ".join(f"{n} {s}" for s, n in counts.items()) if run.findings else "the run raised no finding"
    body = _category_table(run.findings)
    body += table(["#", "severity", "category", "t (s)", "where", "block", "message", "quantities", "tip in W (mm)",
                   "cause"], rows, numeric={0, 3}, row_classes=classes, row_ids=ids,
                  caption="Worst first; a line links to the program listing, the cause column to the chain.",
                  empty="The run raised no finding.")
    hidden = len(run.findings) - len(findings)
    if hidden > 0:
        body += note(f"{hidden} more findings (the least severe, latest) are counted in the table by category "
                     f"above and listed in findings.json; this table shows the worst {len(findings)}.")
    chains = [_chain(ctx, f) for f in findings]
    children = []
    if findings:
        children.append(Section("cause-chains", "Chains of cause", "".join(chains), level=3,
                                lead="From each finding back to its root causes (trace.jsonl, breadth first)."))
    children.append(program)
    return Section("findings", "Findings", body, children=children, lead=lead)


def _category_table(findings: list[dict]) -> str:
    if not findings:
        return ""
    groups: dict[str, dict] = {}
    for finding in findings:
        record = groups.setdefault(str(finding.get("category", "")), {"fatal": 0, "error": 0, "warning": 0, "info": 0,
                                                                      "first": finding})
        record[finding.get("severity", "info")] = record.get(finding.get("severity", "info"), 0) + 1
        if float(finding.get("time_s", 0.0)) < float(record["first"].get("time_s", 0.0)):
            record["first"] = finding
    rows = [[category, r["fatal"], r["error"], r["warning"], r["info"], fmt.seconds(r["first"].get("time_s")),
             _where_text(r["first"])] for category, r in sorted(groups.items(),
                                                                key=lambda kv_: (-max(SEVERITY_RANK.get(s, 0) for s in
                                                                                      ("fatal", "error", "warning", "info")
                                                                                      if kv_[1][s]), kv_[0]))]
    return table(["category", "fatal", "error", "warning", "info", "first at (s)", "first where"], rows,
                 numeric={1, 2, 3, 4, 5}, caption="Every finding, counted by category and severity.")


def _where_text(finding: dict) -> str:
    program = Path(str(finding.get("program") or "")).name
    line = int(finding.get("line") or 0)
    op = str(finding.get("op") or "")
    if line <= 0:
        return f"{op} (no program line)" if op else "no program line"
    return f"{op} {program}:{line}".strip()


def _chain(ctx: Context, finding: dict) -> str:
    ids = [int(i) for i in (finding.get("cause_chain") or [])]
    events = ctx.run.trace.events(ids)
    found = {int(e.get("id", -1)) for e in events}
    rows = []
    for event in events:
        line = int(event.get("line") or 0)
        where = ctx.line_link(finding.get("op", ""), finding.get("program", ""), line, f"line {line}") \
            if line > 0 else fmt.DASH
        quantities = "; ".join(fmt.quantity(k, v) for k, v in sorted((event.get("quantities") or {}).items()))
        rows.append([event.get("id"), fmt.seconds(event.get("time_s")), event.get("domain", ""),
                     event.get("event", ""), where, quantities,
                     ", ".join(str(p) for p in event.get("parents") or []) or "root"])
    lost = [i for i in ids if i not in found]
    body = table(["event", "t (s)", "domain", "event", "line", "quantities", "parents"], rows, numeric={0, 1},
                 empty="The finding names no trace events.")
    if lost:
        body += note(f"Trace events not found in trace.jsonl: {', '.join(str(i) for i in lost)}.")
    summary = join([f"#{finding.get('id')} ", finding.get("category", ""), f" at {_where_text(finding)}: ",
                    f"{len(events)} events"])
    return f'<div id="{esc(anchor("C", finding.get("id")))}" class="chain">{details(summary, body)}</div>'


# ----------------------------------------------------------------------------- program listing

def program_section(ctx: Context) -> Section:
    parts = []
    hits = ctx.findings_by_line
    for op, program, text in ctx.run.programs():
        title = f"{op} · {program}" if program else op
        if text is None:
            parts.append(f"<h4>{esc(title)}</h4>" + note("The program text is not recorded in run.json."))
            continue
        lines = text.splitlines()
        marked = {line for (o, p, line) in hits if program_key(o, p) == program_key(op, program)}
        shown = _shown_lines(len(lines), marked)
        out = []
        previous = 0
        for number in shown:
            if number > previous + 1:
                out.append(f'<span class="gap">… lines {previous + 1}–{number - 1} …</span>')
            line_hits = hits.get((*program_key(op, program), number), [])
            cls = f"ln hit-{ctx.worst(line_hits)}" if line_hits else "ln"
            target = ctx.line_anchor(op, program, number)
            ctx.listed.add(target)
            markers = "".join(" " + ctx.finding_link(f["id"], f"\u25c0 #{f['id']} {f.get('category', '')}")
                              for f in line_hits)
            out.append(f'<span class="{cls}" id="{esc(target)}"><span class="no">{number}</span>'
                       f'{esc(lines[number - 1])}{markers}</span>')
            previous = number
        if previous < len(lines):
            out.append(f'<span class="gap">… lines {previous + 1}–{len(lines)} …</span>')
        scope = "whole program" if len(shown) == len(lines) else \
            f"{len(shown)} of {len(lines)} lines: the lines with findings and {CONTEXT_LINES} lines around them"
        parts.append(f"<h4>{esc(title)}</h4>" + note(f"{len(lines)} lines ({scope}).")
                     + f'<div class="listing">{"".join(out)}</div>')
    if not parts:
        parts.append(note("run.json lists no operations: there is no program to show."))
    return Section("program", "Program", "".join(parts), level=3,
                   lead="Lines with findings are highlighted (worst severity) and link back to their findings.")


def _shown_lines(count: int, marked: set[int]) -> list[int]:
    if count <= FULL_LISTING_MAX_LINES:
        return list(range(1, count + 1))
    keep = set(range(1, min(count, 5) + 1))
    for line in marked:
        keep.update(range(max(1, line - CONTEXT_LINES), min(count, line + CONTEXT_LINES) + 1))
    return sorted(keep)


# ----------------------------------------------------------------------------- identity and missing data

def identity_section(ctx: Context) -> Section:
    run = ctx.run
    document = run.run
    clock = document.get("clock") or {}
    pairs = [("run directory", run.id), ("project", run.name), ("project file", document.get("project_file", fmt.DASH)),
             ("seed", document.get("seed", run.project.get("seed", fmt.DASH))),
             ("format", f"{document.get('format', fmt.DASH)} v{document.get('version', fmt.DASH)}"),
             ("base tick", f"{fmt.num(run.tick_s)} s"),
             ("IPO period", f"{clock.get('ipo_ticks', fmt.DASH)} ticks"),
             ("macro step", f"{clock.get('macro_ipo', fmt.DASH)} IPO ticks"),
             ("L0 stride", ", ".join(f"{op}: {v} ticks" for op, v in sorted((clock.get("l0_ticks") or {}).items()))
              or fmt.DASH),
             ("numba cache key", document.get("numba_cache_key", fmt.DASH))]
    for key, value in sorted((document.get("versions") or {}).items()):
        pairs.append((f"version: {key}", str(value)))
    for key, value in sorted((document.get("threads") or {}).items()):
        pairs.append((f"thread setting {key}", str(value)))
    if run.status:
        pairs.append(("status", str(run.status.get("status", fmt.DASH))))
    inputs = document.get("inputs") or {}
    rows = [[name, fmt.integer(meta.get("bytes")), code(str(meta.get("sha256", "")))]
            for name, meta in sorted(inputs.items()) if isinstance(meta, dict)]
    body = kv(pairs) + "<h4>Input files</h4>" + table(["file", "bytes", "sha256"], rows, numeric={1},
                                                      empty="run.json lists no input files.")
    outputs = document.get("outputs") or []
    if outputs:
        body += para(f"run.json indexes {len(outputs)} output files with their sha256 (the result set of "
                     "architecture 9 is among them).")
    return Section("identity", "Run identity", body,
                   lead="What was run, with which build and settings: enough to reproduce it bit for bit.")


def missing_section(ctx: Context) -> Section:
    rows = [[entry["item"], entry["reason"]] for entry in sorted(ctx.run.missing, key=lambda e: e["item"])]
    body = table(["item", "what that means for this report"], rows,
                 empty="Every file and record the report reads was present.")
    return Section("not-in-run", "Not in this run directory", body,
                   lead="Data the report looked for and did not find; the sections above say what they could "
                        "not show.")


__all__ = ["verdict_header", "findings_section", "program_section", "identity_section", "missing_section"]
