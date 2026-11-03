"""Comparison of two finished runs (docs/report.md "Comparing two runs").

What changed from run A to run B: the verdict, the findings (matched pairs, new in B, resolved in B),
every inspected characteristic, the cycle time per category, program line and tool, tool wear and
life, energy and cost per part, surface finish, and which input files differ.

Findings are matched in three passes (each finding at most once):

1. same category and the same *execution* of the same program line (the k-th time the line runs, from
   timeline.json): robust to any change of feed or speed;
2. same category and the same line (or the same inspected feature for findings without a line),
   paired by program progress (time as a fraction of each run's cycle time) within
   ``finding_match_progress_frac`` by an optimal assignment (Hungarian method, scipy);
3. what is left is "only in A" (resolved) or "only in B" (new).

Thresholds come from db/models/report-compare.yaml. The comparison is a deterministic function of
the two run directories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import fmt
from .context import SEVERITY_RANK
from .html import Raw, Section, badge, code, document, note, pass_badge, table, tiles
from .pdf import write_atomic
from .rundir import RunData
from .sections_part import limits, utilisation
from .sections_production import _consumers

FORMAT = "cnctwin-compare"
VERSION = 1


def parameters() -> dict:
    from ..db.models import params
    return params("report-compare")


# ----------------------------------------------------------------------------- findings

def _executions(run: RunData) -> dict[tuple, list[tuple[int, int]]]:
    out: dict[tuple, list[tuple[int, int]]] = {}
    for block in sorted(run.timeline, key=lambda b: (int(b["tick0"]), int(b.get("index", 0)))):
        key = (str(block.get("op", "")), Path(str(block.get("program", ""))).name, int(block.get("line") or 0))
        out.setdefault(key, []).append((int(block["tick0"]), int(block["tick1"])))
    return out


def _feature_ids(run: RunData) -> set[str]:
    ids = {str(t.get("id")) for t in (run.inspection or {}).get("tolerances", []) if isinstance(t, dict)}
    ids |= {str(f.get("id")) for f in (run.surface or {}).get("features", []) if isinstance(f, dict)}
    return {i for i in ids if i and i != "None"}


def _keyed(run: RunData, features: set[str]) -> list[dict]:
    executions = _executions(run)
    cycle = run.cycle_time_s or 0.0
    out = []
    for finding in run.findings:
        op = str(finding.get("op", ""))
        program = Path(str(finding.get("program", ""))).name
        line = int(finding.get("line") or 0)
        tick = int(finding.get("tick") or 0)
        k = None
        for number, (t0, t1) in enumerate(executions.get((op, program, line), [])):
            if t0 <= tick < t1 or (tick == t1 and number == len(executions[(op, program, line)]) - 1):
                k = number
                break
        message = str(finding.get("message", ""))
        first = message.split(" ", 1)[0].rstrip(":,") if message else ""
        feature = first if first in features else None
        time_s = float(finding.get("time_s") or 0.0)
        out.append({"finding": finding, "category": str(finding.get("category", "")), "op": op, "program": program,
                    "line": line, "execution": k, "feature": feature, "time_s": time_s,
                    "progress": time_s / cycle if cycle > 0 else 0.0})
    return out


def _location(item: dict) -> tuple:
    if item["line"] > 0:
        return ("line", item["category"], item["op"], item["program"], item["line"])
    if item["feature"] is not None:
        return ("feature", item["category"], item["feature"])
    return ("op", item["category"], item["op"])


def _summary(item: dict) -> dict:
    finding = item["finding"]
    return {"id": finding.get("id"), "time_s": finding.get("time_s"), "severity": finding.get("severity"),
            "message": finding.get("message", ""), "quantities": finding.get("quantities") or {},
            "execution": item["execution"]}


def _pair(a: dict, b: dict, how: str) -> dict:
    rank_a = SEVERITY_RANK.get(a["finding"].get("severity"), 0)
    rank_b = SEVERITY_RANK.get(b["finding"].get("severity"), 0)
    quantities = {}
    qa, qb = a["finding"].get("quantities") or {}, b["finding"].get("quantities") or {}
    for key in sorted(set(qa) & set(qb)):
        va, vb = fmt.as_float(qa[key]), fmt.as_float(qb[key])
        if va is not None and vb is not None:
            quantities[key] = {"a": va, "b": vb, "delta": vb - va}
    return {"category": a["category"], "op": a["op"], "program": a["program"], "line": a["line"],
            "feature": a["feature"], "matched_by": how, "a": _summary(a), "b": _summary(b),
            "dt_s": b["time_s"] - a["time_s"],
            "severity_change": "same" if rank_a == rank_b else ("worse" if rank_b > rank_a else "better"),
            "quantities": quantities}


def match_findings(run_a: RunData, run_b: RunData, window_frac: float) -> dict:
    from scipy.optimize import linear_sum_assignment
    features = _feature_ids(run_a) | _feature_ids(run_b)
    items_a, items_b = _keyed(run_a, features), _keyed(run_b, features)
    used_a, used_b = set(), set()
    pairs = []
    exact: dict[tuple, list[int]] = {}
    for j, item in enumerate(items_b):
        if item["execution"] is not None:
            exact.setdefault((item["category"], item["op"], item["program"], item["line"], item["execution"]),
                             []).append(j)
    for i, item in enumerate(items_a):
        if item["execution"] is None:
            continue
        candidates = [j for j in exact.get((item["category"], item["op"], item["program"], item["line"],
                                            item["execution"]), []) if j not in used_b]
        if candidates:
            j = min(candidates, key=lambda c: (abs(items_b[c]["time_s"] - item["time_s"]), c))
            used_a.add(i)
            used_b.add(j)
            pairs.append(_pair(item, items_b[j], "same execution of the line"))
    groups: dict[tuple, tuple[list[int], list[int]]] = {}
    for i, item in enumerate(items_a):
        if i not in used_a:
            groups.setdefault(_location(item), ([], []))[0].append(i)
    for j, item in enumerate(items_b):
        if j not in used_b:
            groups.setdefault(_location(item), ([], []))[1].append(j)
    for location in sorted(groups, key=str):
        left, right = groups[location]
        if not left or not right:
            continue
        cost = np.array([[abs(items_a[i]["progress"] - items_b[j]["progress"]) for j in right] for i in left])
        allowed = cost <= window_frac
        penalised = np.where(allowed, cost, 1e6)
        rows, cols = linear_sum_assignment(penalised)
        for r, c in sorted(zip(rows.tolist(), cols.tolist())):
            if allowed[r, c]:
                used_a.add(left[r])
                used_b.add(right[c])
                pairs.append(_pair(items_a[left[r]], items_b[right[c]], "same line or feature, nearest progress"))
    only_a = [dict(_summary(items_a[i]), category=items_a[i]["category"], op=items_a[i]["op"],
                   program=items_a[i]["program"], line=items_a[i]["line"]) for i in range(len(items_a)) if i not in used_a]
    only_b = [dict(_summary(items_b[j]), category=items_b[j]["category"], op=items_b[j]["op"],
                   program=items_b[j]["program"], line=items_b[j]["line"]) for j in range(len(items_b)) if j not in used_b]
    pairs.sort(key=lambda p: (float(p["a"]["time_s"] or 0.0), int(p["a"]["id"] or 0)))
    return {"matched": pairs, "only_a": only_a, "only_b": only_b,
            "counts": {"a": len(items_a), "b": len(items_b), "matched": len(pairs), "resolved": len(only_a),
                       "new": len(only_b), "worse": sum(1 for p in pairs if p["severity_change"] == "worse"),
                       "better": sum(1 for p in pairs if p["severity_change"] == "better")}}


# ----------------------------------------------------------------------------- dimensions and the rest

def _band(item: dict) -> float | None:
    nominal, lower, upper = limits(item)
    if lower is not None and upper is not None:
        return upper - lower
    return None


def compare_dimensions(run_a: RunData, run_b: RunData, frac: float) -> list[dict]:
    by_a = {str(t.get("id")): t for t in (run_a.inspection or {}).get("tolerances", []) if isinstance(t, dict)}
    by_b = {str(t.get("id")): t for t in (run_b.inspection or {}).get("tolerances", []) if isinstance(t, dict)}
    out = []
    for ident in sorted(set(by_a) | set(by_b)):
        a, b = by_a.get(ident), by_b.get(ident)
        ref = a or b
        nominal, lower, upper = limits(ref)
        ma = fmt.as_float((a or {}).get("measured_m"))
        mb = fmt.as_float((b or {}).get("measured_m"))
        delta = mb - ma if ma is not None and mb is not None else None
        band = _band(ref)
        pa, pb = (a or {}).get("pass"), (b or {}).get("pass")
        ua, ub = (utilisation(a) if a else None), (utilisation(b) if b else None)
        significant = delta is not None and band is not None and abs(delta) > frac * band
        if a is None or b is None:
            change = "only in A" if b is None else "only in B"
        elif pa is True and pb is False:
            change = "worse"
        elif pa is False and pb is True:
            change = "better"
        elif significant and ua is not None and ub is not None:
            change = "worse" if ub > ua else "better"
        else:
            change = "same"
        out.append({"id": ident, "kind": ref.get("kind"), "nominal_m": nominal, "lower_m": lower, "upper_m": upper,
                    "a_measured_m": ma, "b_measured_m": mb, "delta_m": delta, "a_pass": pa, "b_pass": pb,
                    "a_utilisation": ua, "b_utilisation": ub, "significant": bool(significant), "change": change})
    return out


def _relative(a, b) -> float | None:
    va, vb = fmt.as_float(a), fmt.as_float(b)
    if va is None or vb is None:
        return None
    scale = max(abs(va), abs(vb))
    return (vb - va) / scale if scale > 0 else 0.0


def _row(name: str, a, b, threshold: float, lower_is_better: bool = True) -> dict:
    va, vb = fmt.as_float(a), fmt.as_float(b)
    rel = _relative(va, vb)
    if va is None and vb is not None:
        change = "only in B"
    elif vb is None and va is not None:
        change = "only in A"
    elif rel is None or abs(rel) <= threshold:
        change = "same"
    else:
        change = "better" if (rel < 0) == lower_is_better else "worse"
    return {"name": name, "a": va, "b": vb, "delta": None if va is None or vb is None else vb - va,
            "relative": rel, "change": change}


def _cycle(run: RunData) -> dict:
    production = run.production or {}
    first = production.get("first_part") or {}
    if not first:
        from .context import Context
        from .sections_production import _cycle_from_timeline
        first = _cycle_from_timeline(Context(run)) or {}
    return first


def compare_cycle(run_a: RunData, run_b: RunData, threshold: float) -> dict:
    first_a, first_b = _cycle(run_a), _cycle(run_b)
    tick_a, tick_b = run_a.tick_s, run_b.tick_s
    ba = ((first_a.get("cycle") or {}).get("breakdown") or {})
    bb = ((first_b.get("cycle") or {}).get("breakdown") or {})
    total = _row("machine time (s)", (ba.get("total_ticks") or 0) * tick_a if ba else run_a.cycle_time_s,
                 (bb.get("total_ticks") or 0) * tick_b if bb else run_b.cycle_time_s, threshold)
    categories = []
    for name in sorted(set(ba.get("ticks") or {}) | set(bb.get("ticks") or {})):
        categories.append(_row(name, int((ba.get("ticks") or {}).get(name, 0)) * tick_a,
                               int((bb.get("ticks") or {}).get(name, 0)) * tick_b, threshold))
    lines_a = {(e.get("op"), Path(str(e.get("program", ""))).name, int(e.get("line") or 0)): e
               for e in first_a.get("lines") or []}
    lines_b = {(e.get("op"), Path(str(e.get("program", ""))).name, int(e.get("line") or 0)): e
               for e in first_b.get("lines") or []}
    lines = []
    for key in sorted(set(lines_a) | set(lines_b), key=lambda k: (str(k[0]), str(k[1]), k[2])):
        ea, eb = lines_a.get(key) or {}, lines_b.get(key) or {}
        row = _row(f"{key[0]} {key[1]}:{key[2]}",
                   int((ea.get("breakdown") or {}).get("total_ticks") or 0) * tick_a if ea else None,
                   int((eb.get("breakdown") or {}).get("total_ticks") or 0) * tick_b if eb else None, threshold)
        row.update({"op": key[0], "program": key[1], "line": key[2], "block": (eb or ea).get("block", ""),
                    "a_executions": ea.get("executions"), "b_executions": eb.get("executions")})
        lines.append(row)
    tools_a = {e.get("tool"): e for e in (first_a.get("cycle") or {}).get("tools") or []}
    tools_b = {e.get("tool"): e for e in (first_b.get("cycle") or {}).get("tools") or []}
    tools = []
    for tool in sorted(set(tools_a) | set(tools_b), key=lambda t: (t is None, t or 0)):
        ta = int(((tools_a.get(tool) or {}).get("breakdown") or {}).get("total_ticks") or 0) * tick_a \
            if tool in tools_a else None
        tb = int(((tools_b.get(tool) or {}).get("breakdown") or {}).get("total_ticks") or 0) * tick_b \
            if tool in tools_b else None
        row = _row("no tool" if tool is None else f"T{tool}", ta, tb, threshold)
        row["tool"] = tool
        tools.append(row)
    return {"total": total, "categories": categories, "lines": lines, "tools": tools}


def compare_tools(run_a: RunData, run_b: RunData, threshold: float) -> list[dict]:
    def by_number(run):
        return {t.get("number"): t for t in (run.tools or {}).get("tools", []) if isinstance(t, dict)}

    def costs(run):
        record = ((run.production or {}).get("per_part") or [{}])[0]
        return {t.get("number"): (t.get("cost") or {}).get("cost_per_part_eur") for t in record.get("tools") or []}

    tools_a, tools_b, cost_a, cost_b = by_number(run_a), by_number(run_b), costs(run_a), costs(run_b)
    out = []
    for number in sorted(set(tools_a) | set(tools_b), key=lambda n: (n is None, n or 0)):
        a, b = tools_a.get(number) or {}, tools_b.get(number) or {}
        out.append({"number": number, "tool_id": b.get("tool_id") or a.get("tool_id"),
                    "VB_max_m": _row("VB max", a.get("VB_max_m"), b.get("VB_max_m"), threshold),
                    "life_used_fraction": _row("life used", a.get("life_used_fraction"), b.get("life_used_fraction"),
                                               threshold),
                    "cut_time_s": _row("cutting time", a.get("cut_time_s"), b.get("cut_time_s"), threshold),
                    "max_edge_temp_K": _row("hottest edge", a.get("max_edge_temp_K"), b.get("max_edge_temp_K"),
                                            threshold),
                    "cost_per_part_eur": _row("tool cost per part", cost_a.get(number), cost_b.get(number), threshold),
                    "broken": {"a": a.get("broken"), "b": b.get("broken")}})
    return out


def compare_energy(run_a: RunData, run_b: RunData, threshold: float) -> dict:
    def first(run):
        return ((run.production or {}).get("per_part") or [{}])[0]

    pa, pb = first(run_a), first(run_b)
    ea, eb = pa.get("energy") or {}, pb.get("energy") or {}
    ca, cb = _consumers(ea) if ea else {}, _consumers(eb) if eb else {}
    consumers = [_row(name, ca.get(name) if ca else None, cb.get(name) if cb else None, threshold)
                 for name in sorted(set(ca) | set(cb))]
    coolant_a = ((pa.get("consumables") or {}).get("coolant") or {}).get("cost_eur")
    coolant_b = ((pb.get("consumables") or {}).get("coolant") or {}).get("cost_eur")
    batch_a = (run_a.production or {}).get("batch") or {}
    batch_b = (run_b.production or {}).get("batch") or {}
    return {"total_J": _row("energy per part (J)", ea.get("total_J"), eb.get("total_J"), threshold),
            "consumers": consumers,
            "tool_cost_eur": _row("tool cost per part (EUR)", pa.get("tool_cost_eur"), pb.get("tool_cost_eur"), threshold),
            "coolant_cost_eur": _row("coolant cost per part (EUR)", coolant_a, coolant_b, threshold),
            "takt_s": _row("takt (s)", batch_a.get("takt_mean_s"), batch_b.get("takt_mean_s"), threshold),
            "parts_per_hour": _row("parts per hour", batch_a.get("parts_per_hour"), batch_b.get("parts_per_hour"),
                                   threshold, lower_is_better=False)}


def compare_surface(run_a: RunData, run_b: RunData, threshold: float) -> list[dict]:
    fa = {str(f.get("id")): f for f in (run_a.surface or {}).get("features", []) if isinstance(f, dict)}
    fb = {str(f.get("id")): f for f in (run_b.surface or {}).get("features", []) if isinstance(f, dict)}
    out = []
    for ident in sorted(set(fa) | set(fb)):
        a, b = fa.get(ident) or {}, fb.get(ident) or {}
        row = _row(ident, a.get("Ra_m"), b.get("Ra_m"), threshold)
        row.update({"a_pass": a.get("pass"), "b_pass": b.get("pass"), "spec_Ra_m": b.get("spec_Ra_m") or a.get("spec_Ra_m")})
        out.append(row)
    return out


def compare_inputs(run_a: RunData, run_b: RunData) -> list[dict]:
    ia, ib = run_a.run.get("inputs") or {}, run_b.run.get("inputs") or {}
    out = []
    for name in sorted(set(ia) | set(ib)):
        ha = (ia.get(name) or {}).get("sha256")
        hb = (ib.get(name) or {}).get("sha256")
        state = "only in A" if name not in ib else "only in B" if name not in ia else \
            ("same" if ha == hb else "changed")
        out.append({"file": name, "a_sha256": ha, "b_sha256": hb, "state": state})
    return out


def _side(run: RunData) -> dict:
    return {"run": run.id, "path": str(run.path), "name": run.name, "seed": run.run.get("seed"),
            "finished": run.finished, "safe": bool(run.verdict.get("safe")),
            "reasons": [str(r) for r in run.verdict.get("reasons", [])], "cycle_time_s": run.cycle_time_s}


def compare(run_a, run_b, params: dict | None = None) -> dict:
    """The comparison document of two run directories (or RunData)."""
    a = run_a if isinstance(run_a, RunData) else RunData(run_a)
    b = run_b if isinstance(run_b, RunData) else RunData(run_b)
    settings = dict(parameters())
    settings.update(params or {})
    threshold = float(settings["relative_change_significant"])
    findings = match_findings(a, b, float(settings["finding_match_progress_frac"]))
    dimensions = compare_dimensions(a, b, float(settings["dimension_change_tolerance_frac"]))
    cycle = compare_cycle(a, b, threshold)
    tools = compare_tools(a, b, threshold)
    energy = compare_energy(a, b, threshold)
    surface = compare_surface(a, b, threshold)
    inputs = compare_inputs(a, b)
    document_ = {"format": FORMAT, "version": VERSION, "a": _side(a), "b": _side(b),
                 "verdict": {"a_safe": _side(a)["safe"], "b_safe": _side(b)["safe"],
                             "changed": _side(a)["safe"] != _side(b)["safe"]},
                 "findings": findings, "dimensions": dimensions, "cycle": cycle, "tools": tools, "energy": energy,
                 "surface": surface, "inputs": inputs, "parameters": {k: settings[k] for k in sorted(settings)},
                 "missing": {"a": list(a.missing), "b": list(b.missing)}}
    document_["summary"] = _headlines(document_)
    return _clean(document_)


def _headlines(doc: dict) -> list[str]:
    out = []
    verdict = doc["verdict"]
    state = {True: "safe to run", False: "not safe to run"}
    if verdict["changed"]:
        out.append(f"verdict changed: {state[verdict['a_safe']]} → {state[verdict['b_safe']]}")
    else:
        out.append(f"verdict unchanged: {state[verdict['b_safe']]}")
    total = doc["cycle"]["total"]
    if total["relative"] is not None:
        out.append(f"cycle time {fmt.signed(total['relative'] * 100.0, 1)} % ({fmt.seconds(total['a'])} s → "
                   f"{fmt.seconds(total['b'])} s)")
    counts = doc["findings"]["counts"]
    out.append(f"findings: {counts['new']} new, {counts['resolved']} resolved, {counts['matched']} matched "
               f"({counts['worse']} worse, {counts['better']} better)")
    dims = doc["dimensions"]
    if dims:
        out.append(f"characteristics: {sum(d['change'] == 'worse' for d in dims)} worse, "
                   f"{sum(d['change'] == 'better' for d in dims)} better, {sum(d['change'] == 'same' for d in dims)} "
                   "unchanged")
    energy = doc["energy"]["total_J"]
    if energy["relative"] is not None:
        out.append(f"energy per part {fmt.signed(energy['relative'] * 100.0, 1)} %")
    cost = doc["energy"]["tool_cost_eur"]
    if cost["relative"] is not None:
        out.append(f"tool cost per part {fmt.signed(cost['relative'] * 100.0, 1)} %")
    changed = [i["file"] for i in doc["inputs"] if i["state"] != "same"]
    out.append("inputs changed: " + (", ".join(changed) if changed else "none (same input files)"))
    return out


def _clean(value):
    """JSON-safe: non-finite numbers become None, numpy scalars plain."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ----------------------------------------------------------------------------- HTML

def _change(kind: str) -> Raw:
    labels = {"worse": ("worse", "worse"), "better": ("better", "better"), "same": ("same", "same"),
              "only in A": ("resolved", "better"), "only in B": ("new", "worse")}
    text, css = labels.get(kind, (kind, "same"))
    return badge(text, css)


def _value_row(row: dict, render) -> list:
    rel = row.get("relative")
    return [row["name"], render(row.get("a")), render(row.get("b")), render(row.get("delta")),
            fmt.signed(rel * 100.0, 1) if rel is not None else fmt.DASH, _change(row.get("change", "same"))]


def compare_html(doc: dict) -> str:
    a, b = doc["a"], doc["b"]
    state = {True: "SAFE TO RUN", False: "NOT SAFE TO RUN"}
    header = Raw(f"<h1>Comparison: {_e(a['name'])} → {_e(b['name'])}</h1>"
                 f"<p class='lead'>A = run {_e(a['run'])} (seed {_e(a['seed'])}), B = run {_e(b['run'])} "
                 f"(seed {_e(b['seed'])})</p>")
    header += Raw(f'<div class="verdict {"safe" if b["safe"] else "unsafe"}"><div class="verdict-title">'
                  f'<span class="verdict-icon">{"✓" if b["safe"] else "✕"}</span>B: {state[b["safe"]]}</div>'
                  f'<p>A: {state[a["safe"]]}.</p></div>')
    header += tiles([(line.split(":", 1)[0] if ":" in line else "change", line.split(":", 1)[-1].strip(), "")
                     for line in doc["summary"]])
    sections = [_findings_html(doc), _dimensions_html(doc), _cycle_html(doc), _tools_html(doc), _energy_html(doc),
                _surface_html(doc), _inputs_html(doc)]
    return document(f"CNC Twin comparison: {a['run']} vs {b['run']}", sections, header=header,
                    running_title=f"CNC Twin comparison · {a['run']} → {b['run']}")


def _e(value) -> str:
    from .html import esc
    return esc(value)


def _findings_html(doc: dict) -> Section:
    findings = doc["findings"]
    rows = []
    for pair in findings["matched"]:
        quantities = "; ".join(f"{fmt.quantity(k, v['a'])} → {fmt.quantity(k, v['b'])}"
                               for k, v in sorted(pair["quantities"].items()))
        rows.append([pair["category"], _where(pair), f"#{pair['a']['id']} → #{pair['b']['id']}",
                     f"{pair['a']['severity']} → {pair['b']['severity']}", fmt.signed(pair["dt_s"], 3),
                     pair["matched_by"], quantities or fmt.DASH, _change(pair["severity_change"])])
    body = table(["category", "where", "finding A → B", "severity", "Δt (s)", "matched by",
                  "quantities A → B", "change"], rows, numeric={4}, empty="No finding of A matches one of B.")
    for title, items, kind in (("New in B", findings["only_b"], "only in B"),
                               ("Resolved in B (only in A)", findings["only_a"], "only in A")):
        rows = [[item["category"], _where(item), f"#{item['id']}", item["severity"], fmt.seconds(item["time_s"]),
                 item["message"], _change(kind)] for item in items]
        body += f"<h3>{title}</h3>" + table(["category", "where", "finding", "severity", "t (s)", "message", ""],
                                            rows, numeric={4}, empty="none")
    counts = findings["counts"]
    lead = ", ".join(f"{counts[k]} {k}" for k in ("a", "b", "matched", "resolved", "new", "worse", "better"))
    return Section("findings", "Findings", body, lead=lead)


def _where(item: dict) -> str:
    if item.get("line"):
        return f"{item.get('op', '')} {item.get('program', '')}:{item['line']}".strip()
    if item.get("feature"):
        return f"feature {item['feature']}"
    return f"{item.get('op', '')} (no line)".strip()


def _dimensions_html(doc: dict) -> Section:
    rows, classes = [], []
    for d in doc["dimensions"]:
        rows.append([d["id"], d.get("kind") or fmt.DASH, fmt.mm(d["nominal_m"], 4),
                     f"{fmt.mm(d['lower_m'], 4)} … {fmt.mm(d['upper_m'], 4)}", fmt.mm(d["a_measured_m"], 4),
                     fmt.mm(d["b_measured_m"], 4), fmt.signed_um(d["delta_m"]),
                     f"{fmt.pct(d['a_utilisation'], 0)} → {fmt.pct(d['b_utilisation'], 0)}",
                     Raw(pass_badge(d["a_pass"]) + " → " + pass_badge(d["b_pass"])), _change(d["change"])])
        classes.append("changed" if d["change"] not in ("same",) else "")
    body = table(["characteristic", "kind", "nominal (mm)", "limits (mm)", "A (mm)", "B (mm)", "B − A (µm)",
                  "tol. used A → B (%)", "result", "change"], rows, numeric={2, 3, 4, 5, 6}, row_classes=classes,
                 caption=f"Changed: more than {fmt.pct(doc['parameters']['dimension_change_tolerance_frac'], 0)} % "
                         "of the tolerance band, or pass/fail changed.",
                 empty="Neither run has inspected characteristics.")
    return Section("dimensions", "Inspected characteristics", body)


def _cycle_html(doc: dict) -> Section:
    cycle = doc["cycle"]
    seconds = lambda v: fmt.seconds(v)  # noqa: E731
    body = table(["", "A", "B", "B − A", "relative (%)", "change"], [_value_row(cycle["total"], seconds)],
                 numeric={1, 2, 3, 4})
    body += "<h3>Per category</h3>" + table(["category", "A (s)", "B (s)", "B − A (s)", "relative (%)", "change"],
                                            [_value_row(r, seconds) for r in cycle["categories"]], numeric={1, 2, 3, 4})
    body += "<h3>Per tool</h3>" + table(["tool", "A (s)", "B (s)", "B − A (s)", "relative (%)", "change"],
                                        [_value_row(r, seconds) for r in cycle["tools"]], numeric={1, 2, 3, 4})
    changed = sorted((r for r in cycle["lines"] if r["delta"] is None or abs(r["delta"]) > 0),
                     key=lambda r: (-(abs(r["delta"]) if r["delta"] is not None else math.inf), r["name"]))
    rows = [[r["name"], code(r.get("block", "")), seconds(r["a"]), seconds(r["b"]), seconds(r["delta"]),
             f"{r.get('a_executions', fmt.DASH)} / {r.get('b_executions', fmt.DASH)}", _change(r["change"])]
            for r in changed]
    body += "<h3>Per program line</h3>" + table(["line", "block", "A (s)", "B (s)", "B − A (s)", "runs A / B",
                                                 "change"], rows, numeric={2, 3, 4},
                                                caption="Lines whose time changed, largest change first.",
                                                empty="Every program line takes exactly as long in both runs.")
    return Section("cycle", "Cycle time", body)


def _tools_html(doc: dict) -> Section:
    rows = []
    for tool in doc["tools"]:
        for key, render in (("VB_max_m", lambda v: fmt.um(v)), ("life_used_fraction", lambda v: fmt.pct(v, 3)),
                            ("cut_time_s", lambda v: fmt.seconds(v)), ("max_edge_temp_K", lambda v: fmt.degc(v)),
                            ("cost_per_part_eur", lambda v: fmt.eur(v, 4))):
            row = tool[key]
            rows.append([f"T{tool['number']}", *_value_row(row, render)])
    body = table(["tool", "quantity", "A", "B", "B − A", "relative (%)", "change"], rows, numeric={2, 3, 4, 5},
                 caption="VB in µm, life used in %, cutting time in s, edge temperature in °C, cost in EUR.",
                 empty="Neither run has tools.json.")
    return Section("tools", "Tool wear and life", body)


def _energy_html(doc: dict) -> Section:
    energy = doc["energy"]
    kj = lambda v: fmt.kj(v)  # noqa: E731
    rows = [_value_row(energy["total_J"], kj)] + [_value_row(r, kj) for r in energy["consumers"]]
    body = table(["energy", "A (kJ)", "B (kJ)", "B − A (kJ)", "relative (%)", "change"], rows, numeric={1, 2, 3, 4})
    rows = [_value_row(energy[k], (lambda v: fmt.eur(v, 4)) if "eur" in k else (lambda v: fmt.num(v, 5)))
            for k in ("tool_cost_eur", "coolant_cost_eur", "takt_s", "parts_per_hour")]
    body += table(["", "A", "B", "B − A", "relative (%)", "change"], rows, numeric={1, 2, 3, 4})
    return Section("energy", "Energy and cost per part", body)


def _surface_html(doc: dict) -> Section:
    rows = [[*_value_row(r, lambda v: fmt.um(v, 2)), Raw(pass_badge(r["a_pass"]) + " → " + pass_badge(r["b_pass"]))]
            for r in doc["surface"]]
    body = table(["feature", "Ra A (µm)", "Ra B (µm)", "B − A (µm)", "relative (%)", "change", "result"],
                 rows, numeric={1, 2, 3, 4}, empty="Neither run has surface.json features.")
    return Section("surface", "Surface finish", body)


def _inputs_html(doc: dict) -> Section:
    rows = [[i["file"], i["state"], code(str(i["a_sha256"] or fmt.DASH)[:16]), code(str(i["b_sha256"] or fmt.DASH)[:16])]
            for i in doc["inputs"]]
    body = table(["input file", "state", "sha256 A", "sha256 B"], rows, empty="run.json lists no inputs.")
    missing = [f"A: {m['item']}" for m in doc["missing"]["a"]] + [f"B: {m['item']}" for m in doc["missing"]["b"]]
    if missing:
        body += note("Not in the run directories (those parts are compared as absent): " + "; ".join(missing))
    return Section("inputs", "Inputs", body, lead="Which input files differ between the two runs.")


# ----------------------------------------------------------------------------- entry points

def write_comparison(run_a, run_b, out_dir, params: dict | None = None) -> dict:
    doc = compare(run_a, run_b, params)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "compare.json", (json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n").encode())
    write_atomic(out / "compare.html", compare_html(doc).encode("utf-8"))
    doc["files"] = [str(out / "compare.html"), str(out / "compare.json")]
    return doc


def compare_runs(request: dict, out_dir, progress_cb=None) -> dict:
    """API job entry (docs/api.md): ``request`` {run_a, run_b, run_a_dir, run_b_dir, options}."""
    if progress_cb is not None:
        progress_cb(phase="comparing", message=f"{request.get('run_a')} vs {request.get('run_b')}")
    options = dict(request.get("options") or {})
    params = {k: options[k] for k in ("finding_match_progress_frac", "dimension_change_tolerance_frac",
                                      "relative_change_significant") if k in options}
    return write_comparison(request["run_a_dir"], request["run_b_dir"], out_dir, params)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.report.compare",
                                     description="Compare two finished runs: compare.html and compare.json.")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--out", default=".", help="folder for compare.html and compare.json (default: here)")
    parser.add_argument("--json", action="store_true", help="print the comparison as JSON")
    args = parser.parse_args(argv)
    for path in (args.run_a, args.run_b):
        if not Path(path).is_dir():
            print(f"error: {path} is not a run directory", file=sys.stderr)
            return 1
    doc = write_comparison(args.run_a, args.run_b, args.out)
    if args.json:
        print(json.dumps(doc, indent=1, sort_keys=True))
    else:
        print("\n".join(doc["summary"]))
        print("wrote " + ", ".join(doc["files"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compare", "compare_html", "compare_runs", "write_comparison", "match_findings", "compare_dimensions",
           "main", "FORMAT"]
