"""Harness for cycle_time: the case's toolpath through the controller executor and the motion planner.

The program is written from the case's path description in the ISO (FANUC) dialect the controller
implements (the Siemens and Heidenhain dialects of the sources are not available; the path, feed,
tolerance and the control's kinematic limits are what the experiment varied). It runs through
``cnctwin.controller.Controller`` (executor, look-ahead buffer, block-time model) and
``cnctwin.motion`` (look-ahead jerk-limited planner, corner blending within the programmed tolerance,
feed drives) with ``cnctwin.motion.driver.MotionRunner``: no cutting, no other domain.

Machine: a three-axis machine with the case's interpolation period and axis acceleration and jerk
limits; everything the case does not state (drive gains, masses, rapid rates, look-ahead depth, block
processing time) is taken from the database machine with the same control family
(``dmg-dmu50-3rd-siemens``, a SINUMERIK 840D sl machine), listed as assumptions.

Cycle time: from the start of the first path block to the end of the last one (the dwell and the
positioning before the path are outside the window, as the case's notes suggest).
"""

from __future__ import annotations

import copy
import math
import re

from ..cases import from_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import common as H

TEMPLATE_MACHINE = "dmg-dmu50-3rd-siemens"
CONTROLLER_DB = "fanuc-31i-b"
TICK_S = 1e-6
MACRO_IPO = 10
CAM_CHORD_TOL_M = 5e-6          # chordal deviation of the G01 polygon the CAM system wrote (not stated)
DRIVE_FORCE_MARGIN = 10.0       # motor force limit / (moving mass x stated acceleration): not binding


def _plain(value):
    if isinstance(value, dict) and "value" in value and set(value) <= {"value", "src", "note", "page", "range"}:
        return value["value"]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _machine(db, ipo_ms: float, accel: float, jerk: float, tol_m: float, col: H.Collector) -> dict:
    raw = _plain(H.raw_object(db, "machine", TEMPLATE_MACHINE))
    axes = {}
    for name in ("X", "Y", "Z"):
        axis = copy.deepcopy(raw["axes"][name])
        axis["accel_m_s2"] = accel
        axis["jerk_m_s3"] = jerk
        axis["motor_force_N"] = max(float(axis["motor_force_N"]),
                                    DRIVE_FORCE_MARGIN * float(axis["moving_mass_kg"]) * accel)
        axes[name] = axis
    ctrl = copy.deepcopy(raw["controller"])
    ctrl.update({"dialect": "fanuc", "alarms": CONTROLLER_DB, "ipo_period_ms": ipo_ms,
                 "smoothing": [{"code": "G05.1 Q1", "name": "case", "tolerance_mm": tol_m * 1e3}],
                 "corner_default": "blend",
                 "rapid_mm_min": {k: raw["controller"]["rapid_mm_min"][k] for k in ("X", "Y", "Z")}})
    col.assume("drive data (gains, masses, rapid, max feed)", TEMPLATE_MACHINE,
               "not stated: the database machine of the same control family", "db/machines")
    col.assume("motor force limit", f"{DRIVE_FORCE_MARGIN:g} x moving mass x the stated acceleration",
               "the measured machine follows its control's limits; the template's own motor forces saturate at the "
               "case's jerk (excess following error). The cycle time does not change once the force is not "
               "binding (4x and 10x give the same time)", "est")
    col.assume("corner mode", "continuous path (blend within the tolerance)",
               "the programs run in G64 / CYCLE832 and CYCL DEF 32 HSC mode (case settings); the template control's "
               "exact-stop default is not used", "case")
    col.assume("look-ahead depth and block processing time",
               f"{ctrl.get('lookahead_blocks')} blocks, {ctrl.get('block_processing_ms')} ms",
               f"not stated: {TEMPLATE_MACHINE} controller entries (est)", "db/machines")
    machine = {"format": "cnctwin-machine", "version": 1, "id": "validation-3axis", "maker": "validation",
               "model": "three-axis motion test machine", "type": "vertical_mill", "controller": ctrl, "axes": axes,
               "kinematics": {"workpiece_chain": [{"link": "saddle", "axis": "Y", "direction": [0, -1, 0]},
                                                  {"link": "table", "axis": "X", "direction": [-1, 0, 0]}],
                              "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]},
                                             {"link": "nose", "fixed": {"translate_mm": [0, 0, 400]}}],
                              "home_mm": {"X": 0, "Y": 0, "Z": 0}, "tool_axis": [0, 0, 1]},
               "spindles": [copy.deepcopy(raw["spindles"][0])]}
    return machine


def _offset(machine: dict) -> dict:
    """G54 at the middle of the X/Y travels, Z 50 mm below the top of its travel (machine coordinates, m)."""
    out = {}
    for name in ("X", "Y", "Z"):
        lo, hi = machine["axes"][name]["travel_mm"]
        out[name] = (0.5 * (lo + hi) if name != "Z" else hi - 50.0) * 1e-3 - (0.04 if name == "X" else 0.0)
    return out


def _fmt(v_m: float) -> str:
    return f"{v_m * 1e3:.4f}"


def trochoid_blocks(starts_x_m: list[float], radius_m: float, end_x_m: float, centre_y_m: float, as_g01: bool,
                    chord_tol_m: float) -> list[str]:
    """Path blocks: from (0, 0) a line to the bottom of each circle, the full CCW circle, ... the end line."""
    lines = []
    for x in starts_x_m:
        lines.append(f"G01 X{_fmt(x)} Y0.")
        if as_g01:
            n = max(8, int(math.ceil(2.0 * math.pi / (2.0 * math.acos(1.0 - chord_tol_m / radius_m)))))
            for k in range(1, n + 1):
                a = -0.5 * math.pi + 2.0 * math.pi * k / n
                lines.append(f"G01 X{_fmt(x + radius_m * math.cos(a))} Y{_fmt(centre_y_m + radius_m * math.sin(a))}")
        else:
            lines.append(f"G03 X{_fmt(x)} Y0. I0. J{_fmt(radius_m)}")
    lines.append(f"G01 X{_fmt(end_x_m)} Y0.")
    return lines


def program(blocks: list[str], feed_mm_min: float) -> tuple[str, int, int]:
    """(text, first path line, last path line)."""
    head = ["%", "O0100", "G21 G90 G17 G40 G80 G94", "G54 G00 X0. Y0. Z0.", "G04 P200", "G05.1 Q1",
            f"G01 F{feed_mm_min:.0f}."]
    tail = ["G05.1 Q0", "M30", "%"]
    first = len(head) + 1
    last = len(head) + len(blocks)
    return "\n".join(head + blocks + tail), first, last


def run_program(machine: dict, text: str, first_line: int, last_line: int, max_s: float = 120.0) -> dict:
    from ...controller import Controller, ControllerSetup, ProgramSource
    from ...controller.schema import ControllerSettings
    from ...core.domain import RunContext
    from ...core.state import SimState
    from ...motion.domain import MotionDomain
    from ...motion.driver import MotionRunner
    ipo_ticks = int(round(float(machine["controller"]["ipo_period_ms"]) * 1e-3 / TICK_S))
    setup = ControllerSetup(copy.deepcopy(machine), [ProgramSource("trochoid.nc", text)], ControllerSettings(),
                            work_offsets_m=[{"G54": [_offset(machine)[k] for k in ("X", "Y", "Z")]}, {}, {}, {}],
                            tick_s=TICK_S)
    ctrl = Controller(setup)
    dom = MotionDomain(copy.deepcopy(machine), tool_lengths={0: 0.1}, rotating={0: True}, tool_numbers={},
                       tick_s=TICK_S, ipo_ticks=ipo_ticks, macro_ipo=MACRO_IPO)
    ctx = RunContext(project=None, db=None, run_dir=None, seed=0, tick_s=TICK_S, ipo_ticks=ipo_ticks,
                     macro_ipo=MACRO_IPO)
    result = MotionRunner(ctrl, dom, ctx=ctx, state=SimState()).run(max_s)
    records = dom.timeline.records
    path = [r for r in records if first_line <= int(r.line) <= last_line]
    alarms = [p.alarm for p in ctrl.state.paths if p.alarm is not None]
    if not path:
        return {"error": f"no path block executed (stopped: {result.stopped}; alarms {alarms})"}
    ticks = path[-1].tick1 - path[0].tick0
    ideal = sum(int(r.ideal_ticks) for r in path if getattr(r, "ideal_ticks", None))
    return {"cycle_s": ticks * TICK_S, "ideal_s": ideal * TICK_S if ideal else None, "blocks": len(path),
            "alarms": alarms, "stopped": result.stopped, "done": ctrl.done()}


def _tests(case) -> list[dict]:
    """Per reference value: controller, feed, tolerance, radius (the case's per-test lists)."""
    cond = case.section("conditions")
    mach = case.section("machine")
    out = []
    feeds = cond.get("feed_mm_min_per_test")
    tols = cond.get("tcp_tolerance_um_per_test")
    for entry in case.reference_values:
        name = str(entry.get("name"))
        idx = int(entry.get("x")) - 1
        note = str(entry.get("note") or "")
        if feeds:
            feed = float(feeds[idx])
            tol = float(tols[idx]) * 1e-6
            radius = 10e-3
        else:
            m_f = re.search(r"F\s*([0-9]+)", note)
            m_r = re.search(r"R\s*([0-9]+)", note)
            m_t = re.search(r"([0-9]+)\s*um", note)
            feed = float(m_f.group(1))
            radius = float(m_r.group(1)) * 1e-3
            tol = float(m_t.group(1)) * 1e-6
        ctrl = "840d" if name.endswith("840d") else "tnc640"
        prefix = f"interpolator_{ctrl}_"
        ipo = mach.get(prefix + "sampling_ms")
        if ipo is None and mach.get(prefix + "sampling_frequency_Hz") is not None:
            ipo = 1e3 / float(mach[prefix + "sampling_frequency_Hz"])
        out.append({"controller": ctrl, "feed_mm_min": feed, "tol_m": tol, "radius_m": radius, "ipo_ms": float(ipo),
                    "accel": float(mach[prefix + "max_axis_acceleration_m_s2"]),
                    "jerk": float(mach[prefix + "max_axis_jerk_m_s3"])})
    return out


def _path(case, radius_m: float) -> tuple[list[float], float, float]:
    """Circle bottoms, end x and circle-centre y from the case's program description."""
    text = str(case.setup.get("program") or "")
    m = re.search(r"centres at Y = ([0-9.]+) mm and X = ([0-9., ]+) mm", text)
    if m:
        xs = [float(v) * 1e-3 for v in re.findall(r"[0-9]+(?:\.[0-9]+)?", m.group(2))]
        end = re.search(r"ending on Y = 0 at about X = ([0-9.]+) mm", text)
        return xs, float(end.group(1)) * 1e-3 if end else xs[-1], float(m.group(1)) * 1e-3
    # step-over pattern: five circles of R, centres every R/2 from x = 0
    step = 0.5 * radius_m
    xs = [k * step for k in range(5)]
    return xs, xs[-1], radius_m


@harness("cycle_time", models=("cnctwin.controller.Controller", "cnctwin.motion.MotionDomain",
                               "cnctwin.motion.driver.MotionRunner"),
         description="The case's toolpath (ISO dialect) through the controller executor and the motion planner and "
                     "feed drives (MotionRunner, no cutting) on a three-axis machine with the stated interpolation "
                     "period, acceleration, jerk, feed and contour tolerance; time from the first to the last path "
                     "block.")
def cycle_time(case, db) -> HarnessOutput:
    col = H.Collector()
    tests = _tests(case)
    g01 = "G01 segments" in str(case.setup.get("program") or "")
    col.note("Program written in the ISO (FANUC) dialect from the case's path description; the Siemens/Heidenhain "
             "syntax of the sources is not implemented by cnctwin.controller.")
    if g01:
        col.assume("G01 polygon of the 840D path", f"chordal deviation {CAM_CHORD_TOL_M * 1e6:g} um",
                   "segment length not stated (CAM output)", "est")
    preds = []
    cache = {}
    for entry, t in zip(case.reference_values, tests):
        key = (t["controller"], t["feed_mm_min"], t["tol_m"], t["radius_m"])
        if key not in cache:
            xs, end_x, cy = _path(case, t["radius_m"])
            as_g01 = g01 and t["controller"] == "840d"
            blocks = trochoid_blocks(xs, t["radius_m"], end_x, cy, as_g01, CAM_CHORD_TOL_M)
            text, first, last = program(blocks, t["feed_mm_min"])
            machine = _machine(db, t["ipo_ms"], t["accel"], t["jerk"], t["tol_m"], col)
            cache[key] = run_program(machine, text, first, last)
        res = cache[key]
        unit = str(entry.get("unit") or "s")
        if "error" in res or res.get("alarms"):
            preds.append(no_prediction(res.get("error") or f"alarm {res['alarms']}"))
            continue
        ideal = f", ideal {res['ideal_s']:.2f} s" if res.get("ideal_s") else ""
        preds.append(Prediction(from_si(res["cycle_s"], unit), COMPARED,
                                f"{t['controller']}: F {t['feed_mm_min']:.0f}, tol {t['tol_m'] * 1e6:.0f} um, R "
                                f"{t['radius_m'] * 1e3:g} mm, IPO {t['ipo_ms']:g} ms, A {t['accel']:g}, J {t['jerk']:g}; "
                                f"{res['blocks']} blocks{ideal}"))
    return HarnessOutput(preds, [[] for _ in case.reference_curves],
                         ["cnctwin.controller.Controller (FANUC dialect)", "cnctwin.motion.MotionDomain planner + drives",
                          "cnctwin.motion.driver.MotionRunner"],
                         {"template_machine": TEMPLATE_MACHINE, "controller_db": CONTROLLER_DB, "tick_s": TICK_S},
                         col.assumptions, col.provenance, col.notes, None)


@discussion("cycle_time")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        errs = [c.error_rel for c in r.comparisons if c.status == COMPARED and c.error_rel is not None]
        if errs:
            out[r.id] = (f"errors {min(errs) * 100:+.0f} to {max(errs) * 100:+.0f} %. The planner is a time-optimal "
                         "jerk-limited profile with tolerance-band corner blends; the controls' own smoothing "
                         "filters (CYCLE832/COMPCAD, cycle 32 HSC) slow the feed on the circle transitions and "
                         "short segments more than the bounds alone (motion.md 9).")
    out["summary"] = ("No budget is defined for cycle_time. The program is the case's path in ISO syntax; the "
                      "differences between the controls are carried only by their stated interpolation period, "
                      "acceleration and jerk limits.")
    return out


__all__ = ["cycle_time", "trochoid_blocks", "program", "run_program"]
