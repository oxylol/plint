"""The motion domain's SimState field (checkpointable: to_arrays / from_arrays).

Everything for a bit-identical continuation is here: drive, spindle and loop states (the kernel
NamedTuples), every path's queue of commands (serialised IR commands), pieces, plan state and look-ahead
data, the committed spindle commands, the timeline (records and the open record of each path), the
ledger, the step buffers of the PathFrames being filled, pending controller actions and the finding
episodes. Floats go through JSON as their shortest round-trip text (exact); non-finite values as text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from ..controller.ir import (ArcSpec, Command, Feed, FeedMode, Feedback, Kind, LineRef, ProbeResult, SetterResult,
                             Smoothing, Space)
from .abi import LoopState, ServoState, SpindleState


# ------------------------------------------------------------------------------- JSON-safe values

def jf(x):
    """A float (or nested list/tuple/dict of them) with non-finite values as text."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isfinite(x):
            return x
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, np.ndarray):
        return [jf(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [jf(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jf(v) for k, v in x.items()}
    return x


def uf(x):
    """Inverse of jf for floats."""
    if isinstance(x, str) and x in ("nan", "inf", "-inf"):
        return float(x)
    if isinstance(x, list):
        return [uf(v) for v in x]
    if isinstance(x, dict):
        return {k: uf(v) for k, v in x.items()}
    return x


# ------------------------------------------------------------------------------- IR commands

def command_to_dict(cmd: Command) -> dict:
    d = {"kind": cmd.kind.value, "ref": {"program": cmd.ref.program, "line": cmd.ref.line, "block": cmd.ref.block,
                                         "n_number": cmd.ref.n_number, "call_stack": list(cmd.ref.call_stack)},
         "space": cmd.space.value, "target": jf(dict(cmd.target or {})),
         "tip_target_m": jf(cmd.tip_target_m) if cmd.tip_target_m is not None else None,
         "tip_axis": jf(cmd.tip_axis) if cmd.tip_axis is not None else None,
         "arc": None, "feed": None, "smoothing": None,
         "spindle": jf(cmd.spindle), "coolant": jf(cmd.coolant), "tool": jf(cmd.tool), "probe": jf(cmd.probe),
         "setter": jf(cmd.setter), "dwell_s": jf(cmd.dwell_s), "dwell_rev": jf(cmd.dwell_rev), "stop": cmd.stop,
         "message": cmd.message, "alarm": jf(cmd.alarm), "modal": jf(cmd.modal), "ready_tick": int(cmd.ready_tick),
         "path": int(cmd.path), "block_seq": int(cmd.block_seq), "sync": jf(cmd.sync)}
    if cmd.arc is not None:
        a = cmd.arc
        d["arc"] = {"plane": a.plane, "centre": jf(a.centre), "normal": jf(a.normal), "sweep_rad": jf(a.sweep_rad),
                    "radius_start_m": jf(a.radius_start_m), "radius_end_m": jf(a.radius_end_m)}
    if cmd.feed is not None:
        d["feed"] = {"mode": cmd.feed.mode.value, "value": jf(cmd.feed.value),
                     "override_allowed": bool(cmd.feed.override_allowed)}
    if cmd.smoothing is not None:
        s = cmd.smoothing
        d["smoothing"] = {"corner": s.corner, "tolerance_m": jf(s.tolerance_m), "mode_code": s.mode_code,
                          "lookahead": bool(s.lookahead), "tolerance_kind": str(s.tolerance_kind)}
    return d


def command_from_dict(d: dict) -> Command:
    r = d["ref"]
    ref = LineRef(r["program"], int(r["line"]), r["block"], r.get("n_number"), tuple(r.get("call_stack") or ()))
    arc = None
    if d.get("arc") is not None:
        a = d["arc"]
        arc = ArcSpec(a["plane"], tuple(uf(a["centre"])), tuple(uf(a["normal"])), uf(a["sweep_rad"]),
                      uf(a["radius_start_m"]), uf(a["radius_end_m"]))
    feed = None
    if d.get("feed") is not None:
        f = d["feed"]
        feed = Feed(FeedMode(f["mode"]), uf(f["value"]), bool(f["override_allowed"]))
    smoothing = None
    if d.get("smoothing") is not None:
        s = d["smoothing"]
        smoothing = Smoothing(s["corner"], uf(s["tolerance_m"]), s["mode_code"], bool(s["lookahead"]),
                              str(s.get("tolerance_kind", "deviation")))
    return Command(Kind(d["kind"]), ref, Space(d["space"]), uf(d["target"]),
                   tuple(uf(d["tip_target_m"])) if d.get("tip_target_m") is not None else None,
                   tuple(uf(d["tip_axis"])) if d.get("tip_axis") is not None else None, arc, feed, smoothing,
                   uf(d.get("spindle")), uf(d.get("coolant")), uf(d.get("tool")), uf(d.get("probe")),
                   uf(d.get("setter")), uf(d.get("dwell_s", 0.0)), uf(d.get("dwell_rev", 0.0)), d.get("stop", ""),
                   d.get("message", ""), uf(d.get("alarm")), uf(d.get("modal") or {}), int(d.get("ready_tick", 0)),
                   int(d.get("path", 0)), int(d.get("block_seq", -1)), uf(d.get("sync")))


def feedback_to_dict(fb: Feedback | None) -> dict | None:
    if fb is None:
        return None
    d = {"time_s": jf(fb.time_s), "machine_pos": jf(fb.machine_pos), "probe": None, "setter": None,
         "alarm": jf(fb.alarm), "tool_life_expired": list(fb.tool_life_expired),
         "spindle_load_fraction": jf(fb.spindle_load_fraction)}
    if fb.probe is not None:
        d["probe"] = {"triggered": bool(fb.probe.triggered), "machine_pos": jf(fb.probe.machine_pos),
                      "time_s": jf(fb.probe.time_s)}
    if fb.setter is not None:
        s = fb.setter
        d["setter"] = {"tool": int(s.tool), "length_m": jf(s.length_m), "diameter_m": jf(s.diameter_m),
                       "broken": bool(s.broken), "time_s": jf(s.time_s)}
    return d


def feedback_from_dict(d: dict | None) -> Feedback | None:
    if d is None:
        return None
    fb = Feedback(uf(d["time_s"]), uf(d.get("machine_pos") or {}))
    if d.get("probe") is not None:
        p = d["probe"]
        fb.probe = ProbeResult(bool(p["triggered"]), uf(p["machine_pos"]), uf(p["time_s"]))
    if d.get("setter") is not None:
        s = d["setter"]
        fb.setter = SetterResult(int(s["tool"]), uf(s["length_m"]), uf(s["diameter_m"]), bool(s["broken"]),
                                 uf(s["time_s"]))
    fb.alarm = uf(d.get("alarm"))
    fb.tool_life_expired = list(d.get("tool_life_expired") or [])
    fb.spindle_load_fraction = uf(d.get("spindle_load_fraction", 0.0))
    return fb


# ------------------------------------------------------------------------------- pieces

def piece_to_dict(pc) -> dict:
    d = {}
    for f in fields(pc):
        v = getattr(pc, f.name)
        if f.name == "tip":
            d["tip"] = None if v is None else {g.name: jf(getattr(v, g.name)) for g in fields(v)}
        elif f.name == "samples_q":
            d["samples_q"] = None if v is None else jf(v)
        else:
            d[f.name] = jf(v)
    return d


def piece_from_dict(d: dict):
    from .geometry import Piece, TipPath
    tip = None
    if d.get("tip") is not None:
        t = {k: uf(v) for k, v in d["tip"].items()}
        tip = TipPath(**{k: (v if k in ("mode", "rot_axes") else (np.array(v, dtype=float) if isinstance(v, list)
                                                                   else v)) for k, v in t.items()})
    arrays = ("vec", "par", "q0", "q1", "t0", "t1")
    kw = {}
    for k, v in d.items():
        if k == "tip":
            continue
        if k == "samples_q":
            kw[k] = None if v is None else np.array(uf(v), dtype=float)
        elif k in arrays:
            kw[k] = np.array(uf(v), dtype=float)
        else:
            kw[k] = uf(v)
    kw["tip"] = tip
    return Piece(**kw)


# ------------------------------------------------------------------------------- the state

def _tuple_to_arrays(prefix: str, nt) -> dict:
    return {f"{prefix}.{name}": np.array(getattr(nt, name)) for name in nt._fields}


def _tuple_from_arrays(cls, prefix: str, arrays: dict):
    """A kernel NamedTuple from its arrays; a field added since the checkpoint was written starts at zero
    (shaped like the first field)."""
    first = np.ascontiguousarray(arrays[f"{prefix}.{cls._fields[0]}"])
    return cls(*(np.ascontiguousarray(arrays[f"{prefix}.{name}"]) if f"{prefix}.{name}" in arrays
                 else np.zeros_like(first) for name in cls._fields))


@dataclass
class MotionState:
    servo: ServoState
    spindle: SpindleState
    loop: LoopState
    paths: list = field(default_factory=list)        # per path: planner data (dict, JSON-safe)
    spindle_cmd: list = field(default_factory=list)  # committed SpindleCmd dicts
    timeline: list = field(default_factory=list)     # Record.state_dict() of every record
    open_record: list = field(default_factory=list)  # per path: index of the open record (-1)
    idle_from: list = field(default_factory=list)    # per path: first base tick not yet in a record (-1)
    ledger: dict = field(default_factory=dict)       # per path span and totals
    step: dict = field(default_factory=dict)         # buffers of the macro step being filled
    actions: list = field(default_factory=list)      # controller actions not yet taken by the loop
    episodes: dict = field(default_factory=dict)     # finding episodes (overload per axis ...)
    meta: dict = field(default_factory=dict)         # counters (macro, tick of the step), op index ...

    def to_arrays(self) -> tuple[dict, dict]:
        hook = getattr(self, "snapshot_hook", None)
        if hook is not None:
            hook()
        arrays = {}
        arrays.update(_tuple_to_arrays("servo", self.servo))
        arrays.update(_tuple_to_arrays("spindle", self.spindle))
        arrays.update(_tuple_to_arrays("loop", self.loop))
        meta = {"paths": jf(self.paths), "spindle_cmd": jf(self.spindle_cmd), "timeline": jf(self.timeline),
                "open_record": list(self.open_record), "idle_from": list(self.idle_from), "ledger": jf(self.ledger),
                "step": jf(self.step), "actions": jf(getattr(self, "actions_json", self.actions)),
                "episodes": jf(self.episodes),
                "meta": jf(self.meta)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "MotionState":
        return cls(_tuple_from_arrays(ServoState, "servo", arrays), _tuple_from_arrays(SpindleState, "spindle", arrays),
                   _tuple_from_arrays(LoopState, "loop", arrays), uf(meta["paths"]), uf(meta["spindle_cmd"]),
                   uf(meta["timeline"]), list(meta["open_record"]), list(meta["idle_from"]), uf(meta["ledger"]),
                   uf(meta["step"]), uf(meta["actions"]), uf(meta["episodes"]), uf(meta["meta"]))
