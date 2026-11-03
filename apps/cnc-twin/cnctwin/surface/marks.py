"""Marks on the finished surface (docs/models/surface.md section 4).

* **Dwell and rub marks**: a tool that stays in the cut while the feed stops (G04 dwell in the cut, feed hold,
  program stop, data starvation) keeps rubbing the same spot for at least one revolution. The cutting load
  falls to the rubbing load, the deflected tool springs back toward the surface by its deflection normal to
  the surface at the last generation (u_rel . n > 0: the tool stood off the surface) and cuts that much
  deeper: depth = max(0, u_rel . n). The feed-mark peaks under the edge are removed too (the mark is smooth).
* **Spindle stopping in the cut**: the spindle speed falls below ``spindle_stop_ratio`` of its running speed
  while the tool cuts: the edge stops in contact (deflection relief as above, plus a stop mark where the
  edge rests) - a finding, since the restart also risks the edge.
* **Servo marks at reversals**: when a linear axis reverses during a contour, friction makes the following
  error jump (quadrant glitch). The tool-position error relative to the commanded path, -sum_axis e_axis
  u_axis (u_axis the ISO 841 direction of the axis in S), projected on the outward normal of the surface
  being generated, gives the mark height: d_n(peak) - median(d_n) within ``reversal_window`` (> 0: the tool
  stood off, material is left: a bump; < 0: a dig).
* **Witness marks** at step-overs and re-entries: two passes of one tool over one face patch whose tracks
  overlap leave a step equal to the difference of their generated heights (deflection, thermal growth, wear
  between the passes); the deflection share is the difference of their mean u_rel . n.
* **Chatter marks**: dynamics' chatter index above ``chatter_index`` while a patch is generated (the
  vibration at generation is already in every event position, so its height is in the roughness).
"""

from __future__ import annotations

import math

import numpy as np

from . import kinematics as KIN

KIND_DWELL = "dwell"
KIND_FEED_STOP = "feed_stop"
KIND_PROGRAM_STOP = "program_stop"
KIND_SPINDLE_STOP = "spindle_stop_in_cut"
KIND_SERVO = "servo_reversal"
KIND_WITNESS = "witness"


def block_kind(timeline, block: int) -> str:
    if timeline is None or block is None or block < 0:
        return ""
    ref = timeline.line_ref(int(block)) if hasattr(timeline, "line_ref") else None
    return str(getattr(ref, "kind", "") or "") if ref is not None else ""


def dwell_and_stop(frame, ps: dict, cutting: bool, last_event: dict | None, settings, tick_s: float,
                   timeline) -> list[dict]:
    """Dwell / feed-stop and spindle-stop marks of one path over the step (ps: the path's mark state)."""
    marks = []
    ticks = np.asarray(getattr(frame, "tick", np.zeros(0)), dtype=np.int64)
    if len(ticks) == 0:
        return marks
    feed = np.asarray(getattr(frame, "feed_vel_M_m_s", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    speed = np.linalg.norm(feed, axis=1) if len(feed) == len(ticks) else np.zeros(len(ticks))
    omega = np.abs(np.asarray(getattr(frame, "spindle_omega_rad_s", np.zeros(len(ticks))), dtype=float))
    if len(omega) != len(ticks):
        omega = np.zeros(len(ticks))
    blocks = np.asarray(getattr(frame, "block_index", np.full(len(ticks), -1)), dtype=np.int64)
    if len(blocks) != len(ticks):
        blocks = np.full(len(ticks), -1, np.int64)
    running = float(ps.get("running_omega", 0.0))
    for k in range(len(ticks)):
        w = float(omega[k])
        if w > running:
            running = w
        stopped = running > 0.0 and w < settings.spindle_stop_ratio * running
        dwelling = cutting and speed[k] < settings.dwell_speed_m_s and not stopped and w > 0.0
        if dwelling and ps.get("dwell_tick", -1) < 0:
            ps["dwell_tick"] = int(ticks[k])
            ps["dwell_block"] = int(blocks[k])
            ps["dwell_omega"] = w
        elif not dwelling and ps.get("dwell_tick", -1) >= 0:
            start = int(ps["dwell_tick"])
            duration = (int(ticks[k]) - start) * tick_s
            revs = float(ps.get("dwell_omega", 0.0)) * duration / (2.0 * math.pi)
            if revs >= 1.0 and last_event is not None:
                kind = block_kind(timeline, int(ps["dwell_block"]))
                kind = KIND_DWELL if kind == "dwell" else (KIND_PROGRAM_STOP if kind == "stop" else KIND_FEED_STOP)
                marks.append(_mark(kind, last_event, start, int(ps["dwell_block"]),
                                   depth=max(0.0, float(last_event["u_n"])), duration_s=duration, revolutions=revs))
            ps["dwell_tick"] = -1
        was_running = bool(ps.get("spindle_on", False))
        if cutting and was_running and stopped and last_event is not None:
            marks.append(_mark(KIND_SPINDLE_STOP, last_event, int(ticks[k]), int(blocks[k]),
                               depth=max(0.0, float(last_event["u_n"])), duration_s=0.0, revolutions=0.0))
        ps["spindle_on"] = (not stopped) and w > 0.0
        if stopped and w == 0.0:
            running = 0.0
    ps["running_omega"] = running
    return marks


def servo_marks(frame, ps: dict, events: tuple | None, settings, tick_s: float) -> list[dict]:
    """Reversal marks of the linear axes of one path (events: (tick, X, N, u_n, patch) of the path's tool)."""
    marks = []
    ticks = np.asarray(getattr(frame, "tick", np.zeros(0)), dtype=np.int64)
    cmd = np.asarray(getattr(frame, "axis_cmd", np.zeros((0, 0))), dtype=float)
    err = np.asarray(getattr(frame, "following_error", np.zeros((0, 0))), dtype=float)
    if len(ticks) < 2 or cmd.shape[0] != len(ticks) or err.shape != cmd.shape:
        ps["last_cmd"] = cmd[-1].tolist() if cmd.size else ps.get("last_cmd")
        return marks
    dirs = KIN.linear_axis_dirs_S(frame)
    last = ps.get("last_cmd")
    prev = np.asarray(last, dtype=float) if last is not None and len(last) == cmd.shape[1] else None
    full = np.vstack([prev[None, :], cmd]) if prev is not None else cmd
    offset = 1 if prev is not None else 0
    ps["last_cmd"] = cmd[-1].tolist()
    if not dirs or events is None or len(events[0]) == 0:
        return marks
    ev_tick, ev_X, ev_N, ev_un, ev_patch = events
    disp = np.zeros((len(ticks), 3))
    for name, (col, u) in dirs.items():
        disp -= err[:, col:col + 1] * u[None, :]
    window = max(1, int(round(settings.reversal_window_s / (tick_s * max(int(ticks[1] - ticks[0]), 1)))))
    dt = (ticks[1] - ticks[0]) * tick_s
    for name, (col, _u) in dirs.items():
        v = np.diff(full[:, col]) / dt
        eps = 1e-9 / dt
        sign_prev = int(ps.get(f"sign_{name}", 0))
        for k in range(len(v)):
            s = 1 if v[k] > eps else (-1 if v[k] < -eps else 0)
            if s == 0:
                continue
            if sign_prev != 0 and s != sign_prev:
                row = k + 1 - offset                     # tick row of the reversal in this step
                row = min(max(row, 0), len(ticks) - 1)
                t_rev = int(ticks[row])
                near = int(np.argmin(np.abs(ev_tick - t_rev)))
                if abs(int(ev_tick[near]) - t_rev) <= window * int(ticks[1] - ticks[0]):
                    n = ev_N[near]
                    d_n = disp @ n
                    lo, hi = max(row - window, 0), min(row + window + 1, len(ticks))
                    ref = float(np.median(d_n))
                    seg = d_n[lo:hi] - ref
                    j = int(np.argmax(np.abs(seg)))
                    height = float(seg[j])
                    if abs(height) >= settings.servo_min_m:
                        ev = {"X": ev_X[near].tolist(), "n": n.tolist(), "u_n": float(ev_un[near]),
                              "patch": int(ev_patch[near]), "tick": int(ev_tick[near]), "block": -1}
                        blocks = np.asarray(getattr(frame, "block_index", np.zeros(0)), dtype=np.int64)
                        blk = int(blocks[row]) if len(blocks) == len(ticks) else -1
                        marks.append(_mark(KIND_SERVO, ev, t_rev, blk, depth=height, axis=str(name),
                                           duration_s=0.0, revolutions=0.0))
            sign_prev = s
        ps[f"sign_{name}"] = sign_prev
    return marks


def witness(passes: list[dict], tool_width_m: float, min_height_m: float) -> list[dict]:
    """Witness marks between overlapping passes of one patch (passes: {U, H, u_n, tick1, block, slot})."""
    out = []
    ordered = sorted(passes, key=lambda q: (q["U"], q["tick1"]))
    for a, b in zip(ordered, ordered[1:]):
        if abs(b["U"] - a["U"]) >= tool_width_m:
            continue
        height = b["H"] - a["H"]
        if abs(height) < min_height_m:
            continue
        later = a if a["tick1"] > b["tick1"] else b
        out.append({"kind": KIND_WITNESS, "height_m": float(abs(height)), "step_m": float(height),
                    "deflection_m": float(b["u_n"] - a["u_n"]), "other_m": float(height - (b["u_n"] - a["u_n"])),
                    "tick": int(later["tick1"]), "block": int(later["block"]), "slots": [a["slot"], b["slot"]],
                    "track_spacing_m": float(abs(b["U"] - a["U"]))})
    return out


def _mark(kind, ev, tick, block, depth, duration_s, revolutions, axis=None) -> dict:
    out = {"kind": kind, "height_m": float(depth), "tick": int(tick), "block": int(block),
           "location_S_m": [float(v) for v in ev["X"]], "normal_S": [float(v) for v in ev["n"]],
           "patch": int(ev["patch"]), "duration_s": float(duration_s), "revolutions": float(revolutions)}
    if axis is not None:
        out["axis"] = axis
    return out


__all__ = ["dwell_and_stop", "servo_marks", "witness", "block_kind", "KIND_DWELL", "KIND_FEED_STOP",
           "KIND_PROGRAM_STOP", "KIND_SPINDLE_STOP", "KIND_SERVO", "KIND_WITNESS"]
