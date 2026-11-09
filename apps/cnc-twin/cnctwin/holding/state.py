"""The holding domain's SimState field (checkpointable: to_arrays / from_arrays).

Everything a bit-identical continuation needs: per tool slot the clamp and unbalance data, pull-out, slip,
the ratchet state and the previous spindle speed; per workpiece the persistent slip pose, the setup's
location pose, free / seated state, the ratchet state of each contact group, failed bond nodes, broken
vacuum seals, the debris the setup was built with, the part's last two origins in M (acceleration) and
counters. Setups (contact sets) are runtime objects rebuilt deterministically from the project and this
state after a resume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .schema import HoldingSettings


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else ("inf" if value > 0 else ("-inf" if value < 0 else None))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _json_load(value):
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if isinstance(value, dict):
        return {k: _json_load(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_load(v) for v in value]
    return value


@dataclass
class ToolHoldState:
    slot: int
    number: int
    clamp: dict                              # ToolClamp.to_dict()
    unbalance: dict                          # Unbalance.to_dict()
    pullout_m: float = 0.0
    slip_rad: float = 0.0
    ratchet: np.ndarray = field(default_factory=lambda: np.zeros(4))   # [x1, x2, v1, v2]
    omega_prev: float = math.nan
    friction_J: float = 0.0
    min_margin: float = math.inf
    max_pullout_rate_m_per_rev: float = 0.0
    counters: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = {f"{prefix}ratchet": np.array(self.ratchet, dtype=float)}
        meta = {"slot": self.slot, "number": self.number, "clamp": _json_safe(self.clamp),
                "unbalance": _json_safe(self.unbalance), "pullout_m": self.pullout_m, "slip_rad": self.slip_rad,
                "omega_prev": _json_safe(self.omega_prev), "friction_J": self.friction_J,
                "min_margin": _json_safe(self.min_margin), "max_rate": self.max_pullout_rate_m_per_rev,
                "counters": _json_safe(self.counters), "flags": _json_safe(self.flags)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str) -> "ToolHoldState":
        omega = meta["omega_prev"]
        return cls(int(meta["slot"]), int(meta["number"]), _json_load(meta["clamp"]), _json_load(meta["unbalance"]),
                   float(meta["pullout_m"]), float(meta["slip_rad"]),
                   np.ascontiguousarray(arrays[f"{prefix}ratchet"], dtype=float).copy(),
                   math.nan if omega is None else float(_json_load(omega)), float(meta["friction_J"]),
                   float(_json_load(meta["min_margin"])), float(meta["max_rate"]), dict(_json_load(meta["counters"])),
                   dict(_json_load(meta["flags"])))


@dataclass
class PartHoldState:
    workpiece: int
    op_index: int = -1                       # operation whose setup holds the part (-1: none built yet)
    slip_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    location_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    location_error_S: np.ndarray = field(default_factory=lambda: np.zeros(3))
    free: bool = False
    free_cause: str = ""
    free_tick: int = -1
    seated: bool = True
    seat_gap_m: float = 0.0
    ratchet: dict = field(default_factory=dict)       # group name -> [q, v, restoring force of the last lift]
    bond_failed: dict = field(default_factory=dict)   # patch id -> bool mask
    vacuum_lost: list = field(default_factory=list)   # device ids
    debris: dict = field(default_factory=dict)        # patch id -> height the setup was built with
    o_hist: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    o_hist_n: int = 0
    omega_prev: float = math.nan
    removed_m3: float = 0.0
    slip_since_build_m: float = 0.0
    supports_version: int = 0
    build_pose: np.ndarray = field(default_factory=lambda: np.eye(4))     # slip pose the setup was built at
    build_gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # gravity in H at that build
    build_omega: float = 0.0                                              # work-spindle speed at that build
    build_volume: float = 0.0                                             # part volume at that build
    friction_J: float = 0.0
    min_margin: float = math.inf
    max_slip_m: float = 0.0
    max_lift_m: float = 0.0
    counters: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    tip_prev: np.ndarray | None = None                # tool tip in S at the last tick (the cut's feed speed)

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = {f"{prefix}slip_pose": np.array(self.slip_pose, dtype=float),
                  f"{prefix}location_pose": np.array(self.location_pose, dtype=float),
                  f"{prefix}location_error_S": np.array(self.location_error_S, dtype=float),
                  f"{prefix}o_hist": np.array(self.o_hist, dtype=float),
                  f"{prefix}build_pose": np.array(self.build_pose, dtype=float),
                  f"{prefix}build_gravity": np.array(self.build_gravity, dtype=float)}
        for name, value in sorted(self.ratchet.items()):
            arrays[f"{prefix}ratchet.{name}"] = np.array(value, dtype=float)
        if self.tip_prev is not None:
            arrays[f"{prefix}tip_prev"] = np.array(self.tip_prev, dtype=float)
        bonds = sorted(self.bond_failed)
        for i, pid in enumerate(bonds):
            arrays[f"{prefix}bond.{i}"] = np.array(self.bond_failed[pid], dtype=bool)
        meta = {"workpiece": self.workpiece, "op_index": self.op_index, "free": self.free,
                "free_cause": self.free_cause, "free_tick": self.free_tick, "seated": self.seated,
                "seat_gap_m": self.seat_gap_m, "ratchet": sorted(self.ratchet), "bonds": bonds,
                "vacuum_lost": list(self.vacuum_lost), "debris": _json_safe(dict(sorted(self.debris.items()))),
                "o_hist_n": self.o_hist_n, "omega_prev": _json_safe(self.omega_prev), "removed_m3": self.removed_m3,
                "slip_since_build_m": self.slip_since_build_m, "supports_version": self.supports_version,
                "build_omega": self.build_omega, "build_volume": self.build_volume,
                "friction_J": self.friction_J, "min_margin": _json_safe(self.min_margin),
                "max_slip_m": self.max_slip_m, "max_lift_m": self.max_lift_m,
                "counters": _json_safe(self.counters), "flags": _json_safe(self.flags)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str) -> "PartHoldState":
        def arr(name):
            return np.ascontiguousarray(arrays[f"{prefix}{name}"]).copy()

        omega = meta["omega_prev"]
        obj = cls(int(meta["workpiece"]), int(meta["op_index"]), arr("slip_pose"), arr("location_pose"),
                  arr("location_error_S"), bool(meta["free"]), str(meta["free_cause"]), int(meta["free_tick"]),
                  bool(meta["seated"]), float(meta["seat_gap_m"]))
        obj.ratchet = {name: arr(f"ratchet.{name}") for name in meta["ratchet"]}
        obj.bond_failed = {pid: arr(f"bond.{i}") for i, pid in enumerate(meta["bonds"])}
        obj.vacuum_lost = list(meta["vacuum_lost"])
        obj.debris = {k: float(v) for k, v in _json_load(meta["debris"]).items()}
        obj.o_hist = arr("o_hist")
        obj.o_hist_n = int(meta["o_hist_n"])
        obj.omega_prev = math.nan if omega is None else float(_json_load(omega))
        obj.removed_m3 = float(meta["removed_m3"])
        obj.slip_since_build_m = float(meta["slip_since_build_m"])
        obj.supports_version = int(meta["supports_version"])
        obj.build_pose = arr("build_pose")
        obj.build_gravity = arr("build_gravity")
        obj.build_omega = float(meta["build_omega"])
        obj.build_volume = float(meta["build_volume"])
        obj.friction_J = float(meta["friction_J"])
        obj.min_margin = float(_json_load(meta["min_margin"]))
        obj.max_slip_m = float(meta["max_slip_m"])
        obj.max_lift_m = float(meta["max_lift_m"])
        obj.tip_prev = arr("tip_prev") if f"{prefix}tip_prev" in arrays else None
        obj.counters = dict(_json_load(meta["counters"]))
        obj.flags = dict(_json_load(meta["flags"]))
        return obj


@dataclass
class HoldingState:
    settings: HoldingSettings
    seed: int
    op_index: int = 0
    part_index: int = 0
    tools: dict = field(default_factory=dict)        # slot -> ToolHoldState
    parts: dict = field(default_factory=dict)        # workpiece -> PartHoldState
    values: dict = field(default_factory=dict)       # channel values of the last step
    notes: list = field(default_factory=list)        # estimates and assumptions in force (report)
    trace_notes: dict = field(default_factory=dict)  # once-only trace events (key -> id)
    setups: list = field(default_factory=list)       # per built setup: summary (finalize)
    events: list = field(default_factory=list)       # slip / pull-out / free events (finalize)

    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict = {}
        meta = {"settings": dict(self.settings.__dict__), "seed": int(self.seed), "op_index": self.op_index,
                "part_index": self.part_index,
                "values": _json_safe({k: v for k, v in sorted(self.values.items())}),
                "notes": list(self.notes), "trace_notes": dict(sorted(self.trace_notes.items())),
                "setups": _json_safe(self.setups), "events": _json_safe(self.events), "tools": [], "parts": []}
        for slot, ts in sorted(self.tools.items()):
            a, m = ts.to_arrays(f"tool.{slot}.")
            arrays.update(a)
            meta["tools"].append(m)
        for w, ps in sorted(self.parts.items()):
            a, m = ps.to_arrays(f"part.{w}.")
            arrays.update(a)
            meta["parts"].append(m)
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "HoldingState":
        obj = cls(HoldingSettings(**meta["settings"]), int(meta["seed"]), int(meta["op_index"]),
                  int(meta["part_index"]))
        values = _json_load(meta["values"])
        obj.values = {k: (np.array(v, dtype=float) if isinstance(v, list) else (math.nan if v is None else v))
                      for k, v in values.items()}
        obj.notes = list(meta["notes"])
        obj.trace_notes = dict(meta["trace_notes"])
        obj.setups = list(_json_load(meta["setups"]))
        obj.events = list(_json_load(meta["events"]))
        for m in meta["tools"]:
            slot = int(m["slot"])
            obj.tools[slot] = ToolHoldState.from_arrays(arrays, m, f"tool.{slot}.")
        for m in meta["parts"]:
            w = int(m["workpiece"])
            obj.parts[w] = PartHoldState.from_arrays(arrays, m, f"part.{w}.")
        return obj


__all__ = ["HoldingState", "ToolHoldState", "PartHoldState"]
