"""What the regression test runs of each example and which of its EXPECTED.md behaviours it checks
(docs/models/loop.md "Regression"; no tests here).

``BUDGET_S``: simulated seconds run of a long example (``options.max_sim_s``; an example not listed runs in full).
``CHECKS[name](run_dir, run) -> [item]``: the EXPECTED.md items the run can show, each
``{item, expected, band, value, status: passed | failed | not reached, note}``; ``not reached`` when the part of
the program that shows it lies beyond the budget. :func:`record` builds a test's record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# simulated seconds per example: the set-up, the first tool change and the first cut of each program (the whole
# loop costs 7-60 CPU s per simulated second on these projects, docs/models/loop.md 13-14)
BUDGET_S = {
    "pocket-islands-al6061": 8.0,
    "drill-tap-plate-1045": 8.0,
    "thin-wall-al7075": 8.0,
    "two-setup-flip": 8.0,
    "lathe-shaft-4140": 8.0,
    "batch-50": 8.0,
    "die-h13-ballnose": 8.0,
    "blade-5axis-ti": 8.0,
    "ti-slot-throughtool": 8.0,
    "sanity-mistakes": 8.0,           # stops at its first alarm (line 16) well before; the budget is a guard
}


# variants run in addition to the example as written: the probing section removed and the work offset set to the
# value the probing finds (until 2026-11-09 the hot loop had no stylus model and programs that probe stopped at their
# first probing alarm; the variants now compare the probed and the hand-set runs)
def _without_probe(project_dir: Path, program: str, start: str, end_marker: str, offset: list, parts=None) -> None:
    import yaml
    text = (project_dir / program).read_text().splitlines()
    i0 = next(i for i, line in enumerate(text) if line.startswith(start))
    i1 = next(i for i in range(i0 + 1, len(text)) if text[i].startswith(end_marker))
    (project_dir / program).write_text("\n".join(text[:i0] + text[i1:]) + "\n")
    data = yaml.safe_load((project_dir / "project.yaml").read_text())
    data["operations"][0]["setup"]["work_offsets_mm"] = {"G54": list(offset)}
    if parts is not None:
        data["production"]["parts"] = int(parts)
    (project_dir / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


# simulated seconds of a variant when it differs from the example's budget (none: the carry-over of a second part
# of batch-50, 27 s per part by the controller's dry run, is beyond these budgets)
VARIANT_BUDGET_S: dict = {}

VARIANTS = {
    # N10 measures the top 0.200 mm below the hand-set G54 Z (EXPECTED.md 1): G54 Z -480.000
    "pocket-islands-al6061": ("no-probe", lambda d: _without_probe(d, "op10.nc", "N10", "N20",
                                                                   [-431.0, -243.2, -480.0])),
    # N10 moves G54 to the measured corner (-411.000, -228.000, -496.000) (EXPECTED.md 1); three parts of the 50
    "batch-50": ("no-probe", lambda d: _without_probe(d, "bracket.nc", "N10", "N20", [-411.0, -228.0, -496.0], 3)),
}


# =============================================================================== reading a run

def _json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def series(run_dir: Path, tier: str, column: str):
    """(time_s, values) of one column of a tier (empty arrays when the run did not record it)."""
    from cnctwin.core.recorder import iter_series, series_columns
    run_dir = Path(run_dir)
    if not (run_dir / "series" / "index.jsonl").is_file() or column not in series_columns(run_dir, tier):
        return np.zeros(0), np.zeros(0)
    ticks, values = [], []
    for t, cols in iter_series(run_dir, tier, [column]):
        ticks.append(t)
        values.append(cols[column])
    if not ticks:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ticks) * 1e-6, np.concatenate(values)


def findings(run_dir: Path) -> list[dict]:
    doc = _json(Path(run_dir) / "findings.json") or {}
    return list(doc.get("findings", []))


def item(name: str, expected, band, value, ok, note: str = "") -> dict:
    if value is None:
        status = "not reached"
    else:
        status = "passed" if ok else "failed"
    return {"item": name, "expected": expected, "band": band, "value": value, "status": status, "note": note}


def band_item(name: str, expected, lo: float, hi: float, value, unit: str = "", note: str = "") -> dict:
    ok = value is not None and math.isfinite(value) and lo <= value <= hi
    return item(name, f"{expected} {unit}".strip(), f"{lo}-{hi} {unit}".strip(),
                None if value is None else float(value), ok, note)


def peak(run_dir: Path, tier: str, column: str, t0: float = 0.0, t1: float = math.inf, scale: float = 1.0):
    """Largest |value| of a column in [t0, t1]; None (not reached) when the run part never made it non-zero (the
    tool had not cut yet within the budget)."""
    t, v = series(run_dir, tier, column)
    sel = (t >= t0) & (t <= t1) & np.isfinite(v)
    if not np.any(sel):
        return None
    top = float(np.max(np.abs(v[sel])))
    return top * scale if top > 0.0 else None


def mean_where(run_dir: Path, tier: str, column: str, mask_column: str | None = None, threshold: float = 0.0,
               scale: float = 1.0):
    t, v = series(run_dir, tier, column)
    if mask_column is not None:
        tm, m = series(run_dir, tier, mask_column)
        if len(m) != len(v):
            return None
        sel = np.isfinite(v) & np.isfinite(m) & (np.abs(m) > threshold)
    else:
        sel = np.isfinite(v) & (np.abs(v) > threshold)
    return float(np.mean(np.abs(v[sel])) * scale) if np.any(sel) else None


def no_finding(run_dir: Path, category: str) -> tuple[bool, int]:
    rows = [f for f in findings(run_dir) if f["category"] == category]
    return not rows, len(rows)


# =============================================================================== checks per example

def _common(run_dir, run) -> list[dict]:
    out = []
    ok, n = no_finding(run_dir, "spindle_overload")
    out.append(item("no spindle overload finding", "none", "none", n, ok))
    return out


def check_pocket(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    probes = [f for f in findings(run_dir) if f["severity"] in ("error", "fatal") and
              (f["category"] == "alarm" or any(w in f["message"].lower() for w in ("probe", "g31", "skip")))]
    out.append(item("N10 probe Z set: G31 triggers without alarm", "2 triggering G31", "exact",
                    len(probes), not probes, "alarms and probe / G31 / skip errors counted: " +
                    "; ".join(f["message"][:80] for f in probes[:2])))
    p = peak(run_dir, "macro", "cutting.spindle_power_W[0]", scale=1e-3)
    out.append(band_item("peak spindle cutting power in the trochoid loops", 2.7, 1.8, 3.6, p, "kW",
                         "largest macro-step mean of the run part"))
    return out


def check_drill_tap(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    t = peak(run_dir, "tick", "cutting.torque_Nm[0]")
    out.append(band_item("drilling torque, D6.8", 3.3, 2.6, 4.6, t, "N m", "peak tick-mean torque of the run part"))
    f = peak(run_dir, "tick", "cutting.force_z_N[0]", scale=1e-3)
    out.append(band_item("drilling thrust, D6.8", 0.91, 0.8, 1.4, f, "kN", "peak tick-mean axial force"))
    return out


def check_thin_wall(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    p = peak(run_dir, "macro", "cutting.spindle_power_W[0]", scale=1e-3)
    out.append(band_item("peel cutting power (T1)", 3.0, 2.0, 4.0, p, "kW"))
    return out


def _trace_events(run_dir: Path, names: tuple) -> list[dict]:
    path = Path(run_dir) / "trace.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if any(f'"{n}"' in line for n in names):
            out.append(json.loads(line))
    return out


def check_flip(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    events = _trace_events(run_dir, ("loop:operation_start", "removal:transfer"))
    started = any(e["event"] == "loop:operation_start" and int(e["quantities"].get("op_index", 0)) == 1
                  for e in events)
    transfers = sum(1 for e in events if e["event"] == "removal:transfer")
    out.append(item("OP20 runs on the OP10 part turned about X", "transfer, OP20 started", "exact",
                    transfers if started else None, started and transfers == 1,
                    "OP20 not reached: the run ended in OP10" if not started else ""))
    return out


def check_lathe(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    p = peak(run_dir, "macro", "cutting.spindle_power_W[0]", scale=1e-3)
    out.append(band_item("roughing spindle shaft power", "5.4-6.8", 4.5, 8.0, p, "kW"))
    return out


def check_batch(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    p = peak(run_dir, "macro", "cutting.spindle_power_W[0]", scale=1e-3)
    out.append(band_item("T1 roughing power", 13.1, 9.0, 15.0, p, "kW"))
    return out


def check_sanity(run_dir, run) -> list[dict]:
    import yaml
    from cnctwin.paths import EXAMPLES
    expected = yaml.safe_load((Path(EXAMPLES) / "sanity-mistakes" / "expected_findings.yaml").read_text())
    want = sorted((f["rule"], int(f["line"]), f["severity"]) for f in expected["findings"])
    got = sorted((f["message"].split(":", 1)[0], int(f["line"]), f["severity"]) for f in findings(run_dir)
                 if f["category"] == "sanity" and f["domain"] == "sanity")
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    out = [item("the 23 sanity findings of expected_findings.yaml", len(want), "exact", len(got),
                not missing and not extra, f"missing {missing}, extra {extra}")]
    alarms = [f for f in findings(run_dir) if f["category"] == "alarm"]
    first = alarms[0] if alarms else None
    out.append(item("the controller stops with alarm 320 at line 16", "line 16", "exact",
                    first["line"] if first else None, bool(first) and int(first["line"]) == 16,
                    first["message"] if first else ""))
    return out


def _no(run_dir, category: str, label: str) -> dict:
    ok, n = no_finding(run_dir, category)
    return item(label, "none", "none", n, ok)


def check_die(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    out.append(_no(run_dir, "overtravel", "travel inside the limits (Z margin 6 mm): no overtravel finding"))
    p = peak(run_dir, "macro", "cutting.spindle_power_W[0]")
    out.append(band_item("T1 roughing cutting power", 11, 5, 25, p, "W", "largest macro-step mean"))
    f = peak(run_dir, "tick", "cutting.force_y_N[0]")
    out.append(band_item("T1 mean tangential force", 10, 5, 20, f, "N", "largest tick-mean force along y_T0"))
    return out


def check_blade(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    out.append(_no(run_dir, "overtravel", "machine travel inside the UMC-750 limits: no overtravel finding"))
    out.append(_no(run_dir, "singularity", "no pass through B = 0 (no C singularity finding)"))
    return out


def check_ti_slot(run_dir, run) -> list[dict]:
    out = _common(run_dir, run)
    t = peak(run_dir, "tick", "cutting.torque_Nm[0]")
    out.append(band_item("slotting spindle torque", 19.6, 10, 24, t, "N m", "peak tick-mean torque"))
    p = peak(run_dir, "macro", "cutting.spindle_power_W[0]", scale=1e-3)
    out.append(band_item("slotting cutting power", 1.18, 0.6, 1.4, p, "kW"))
    sanity = [f for f in findings(run_dir) if f["category"] == "sanity" and f["message"].startswith("cutting-data")]
    out.append(item("cutting-data sanity rule reports nothing for T1", "none", "none", len(sanity), not sanity))
    return out


CHECKS = {"pocket-islands-al6061": check_pocket, "drill-tap-plate-1045": check_drill_tap,
          "thin-wall-al7075": check_thin_wall, "two-setup-flip": check_flip, "lathe-shaft-4140": check_lathe,
          "batch-50": check_batch, "sanity-mistakes": check_sanity, "die-h13-ballnose": check_die,
          "blade-5axis-ti": check_blade, "ti-slot-throughtool": check_ti_slot}


# =============================================================================== the record

def record(name: str, run_dir: Path, options: dict, error, wall_s: float, cpu_s: float) -> dict:
    run = _json(Path(run_dir) / "run.json")
    rows = findings(run_dir)
    counts: dict = {}
    by_category: dict = {}
    for f in rows:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
        by_category[f["category"]] = by_category.get(f["category"], 0) + 1
    conservation = {}
    summary = (run or {}).get("summary", {})
    if run is not None:
        cons = summary.get("conservation", {})
        conservation = {k: bool(cons.get(k, {}).get("ok")) for k in ("mass", "energy", "time")}
        conservation["final_mass"] = bool((summary.get("final_mass") or {}).get("ok"))
    checks = []
    # a run that stopped on an error still leaves its findings and closed series: the items it reached count
    if run is not None or (Path(run_dir) / "series" / "index.jsonl").is_file():
        try:
            checks = CHECKS[name](run_dir, run or {})
        except (OSError, ValueError, KeyError) as failure:
            checks = [item("EXPECTED checks", "", "", None, False, f"checks could not read the run: {failure}")]
    if error is not None or run is None:
        # the run stopped on an error: a band the stopped part misses was not reached (the cut had just begun)
        for c in checks:
            if c["status"] == "failed" and c["value"] is not None and "-" in str(c.get("band", "")):
                c["note"] = (c.get("note") or "") + f" (value {c['value']:.4g} when the run stopped)"
                c["status"] = "not reached"
    return {"example": name, "options": options, "error": None if error is None else
            f"{type(error).__name__}: {error}", "completed_run": run is not None, "complete": bool((run or {}).get(
                "complete")), "truncated": summary.get("truncated"), "stopped": summary.get("stopped"),
            "wall_s": wall_s, "cpu_s": cpu_s,
            "sim_time_s": summary.get("sim_time_s"), "cycle_time_s": summary.get("cycle_time_s"),
            "verdict": (run or {}).get("verdict"),
            "findings": {"severity": counts, "category": by_category,
                         "errors": [f"{f['category']} line {f['line']}: {f['message']}"[:200] for f in rows
                                    if f["severity"] in ("error", "fatal")][:12]},
            "conservation": conservation, "removed_volume_m3": summary.get("removed_volume_m3"),
            "expected": checks,
            "expected_counts": {s: sum(1 for c in checks if c["status"] == s) for s in ("passed", "failed",
                                                                                        "not reached")}}
