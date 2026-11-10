"""The dynamics domain's SimState field (checkpointable: to_arrays / from_arrays).

Everything a bit-identical continuation needs: per path the system's modal basis (the Mode list), the carried
kernel state (modal coordinates, force samples at the current node, the last wrenches, node tick, flags), the
chatter analysis window and episode state, the stability diagrams and operating points recorded per tool, the
finding counters and the cumulative energy terms. The per-step kernel arrays (discretisation, shapes, relaxation)
are rebuilt from the bus and this state each macro step.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .modal import Mode

FORMAT_VERSION = 1


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else ("inf" if value > 0 else ("-inf" if value < 0 else "nan"))
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
        return {"__array__": _json_safe(value.tolist()), "dtype": str(value.dtype), "shape": list(value.shape)}
    return value


def _json_load(value):
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if value == "nan":
        return math.nan
    if isinstance(value, dict):
        if "__array__" in value:
            return np.array(_json_load(value["__array__"]), dtype=value["dtype"]).reshape(value["shape"])
        return {k: _json_load(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_load(v) for v in value]
    return value


@dataclass
class SystemRecord:
    """Carried state of one path's system."""

    path: int
    slot: int
    workpiece: int
    structure_version: int
    modes: list                                  # Mode
    nst: int
    ref_station: int
    l0_ticks: int
    rotating: bool
    n_teeth: int
    x: np.ndarray                                # [nm, 2]
    f_prev: np.ndarray                           # [nm]
    f_cut: np.ndarray
    f_cut_prev: np.ndarray
    f_unb: np.ndarray
    f_base: np.ndarray
    st_force: np.ndarray                         # [nst, 6]
    work_force_S: np.ndarray                     # [3]
    node_tick: int
    flags: np.ndarray                            # [N_FL] int64
    energy_start_J: float = 0.0                  # modal energy at kernel_inputs (after a basis change)
    basis_change_J: float = 0.0                  # booked at the last kernel_inputs
    mode_signature: tuple = ()
    work_shape: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # work shapes at the contact (S)
    carry_terms: np.ndarray = field(default_factory=lambda: np.zeros(4))      # catch-up W_unb, W_base, D, dE
    f_contact_Hz: float = 0.0                    # highest contact-stiffened mode frequency (modal.f_contact_Hz)

    def arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = {f"{prefix}x": self.x.copy(), f"{prefix}f_prev": self.f_prev.copy(), f"{prefix}f_cut": self.f_cut.copy(),
                  f"{prefix}f_cut_prev": self.f_cut_prev.copy(), f"{prefix}f_unb": self.f_unb.copy(),
                  f"{prefix}f_base": self.f_base.copy(), f"{prefix}st_force": self.st_force.copy(),
                  f"{prefix}work_force_S": self.work_force_S.copy(), f"{prefix}flags": self.flags.copy(),
                  f"{prefix}work_shape": np.asarray(self.work_shape, float).copy(),
                  f"{prefix}carry_terms": np.asarray(self.carry_terms, float).copy()}
        modes_meta = []
        for i, m in enumerate(self.modes):
            entry = {"kind": m.kind, "omega_rad_s": m.omega_rad_s, "zeta": m.zeta, "source": m.source, "label": m.label,
                     "mass_kg": m.mass_kg, "work_row": m.work_row, "key": _json_safe(list(m.key))}
            if m.stations is not None:
                arrays[f"{prefix}m{i}.stations"] = np.asarray(m.stations, float).copy()
            if m.at_unbalance is not None:
                arrays[f"{prefix}m{i}.at_unbalance"] = np.asarray(m.at_unbalance, float).copy()
            arrays[f"{prefix}m{i}.base"] = np.asarray(m.base, float).copy()
            modes_meta.append(entry)
        meta = {"path": self.path, "slot": self.slot, "workpiece": self.workpiece,
                "structure_version": self.structure_version, "nst": self.nst, "ref_station": self.ref_station,
                "l0_ticks": self.l0_ticks, "rotating": self.rotating, "n_teeth": self.n_teeth,
                "node_tick": int(self.node_tick), "energy_start_J": _json_safe(self.energy_start_J),
                "basis_change_J": _json_safe(self.basis_change_J), "modes": modes_meta,
                "mode_signature": _json_safe(list(self.mode_signature)), "f_contact_Hz": float(self.f_contact_Hz)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str) -> "SystemRecord":
        modes = []
        for i, e in enumerate(meta["modes"]):
            key = tuple(_to_tuple(v) for v in _json_load(e["key"]))
            modes.append(Mode(int(e["kind"]), float(e["omega_rad_s"]), float(e["zeta"]), str(e["source"]),
                              str(e["label"]), mass_kg=float(e["mass_kg"]),
                              stations=arrays.get(f"{prefix}m{i}.stations"),
                              at_unbalance=arrays.get(f"{prefix}m{i}.at_unbalance"), work_row=int(e["work_row"]),
                              base=np.array(arrays[f"{prefix}m{i}.base"], float), key=key))
        return cls(int(meta["path"]), int(meta["slot"]), int(meta["workpiece"]), int(meta["structure_version"]), modes,
                   int(meta["nst"]), int(meta["ref_station"]), int(meta["l0_ticks"]), bool(meta["rotating"]),
                   int(meta["n_teeth"]), np.array(arrays[f"{prefix}x"]), np.array(arrays[f"{prefix}f_prev"]),
                   np.array(arrays[f"{prefix}f_cut"]), np.array(arrays[f"{prefix}f_cut_prev"]),
                   np.array(arrays[f"{prefix}f_unb"]), np.array(arrays[f"{prefix}f_base"]),
                   np.array(arrays[f"{prefix}st_force"]), np.array(arrays[f"{prefix}work_force_S"]),
                   int(meta["node_tick"]), np.array(arrays[f"{prefix}flags"]),
                   float(_json_load(meta["energy_start_J"])), float(_json_load(meta["basis_change_J"])),
                   tuple(_to_tuple(v) for v in _json_load(meta["mode_signature"])),
                   np.array(arrays[f"{prefix}work_shape"], float), np.array(arrays[f"{prefix}carry_terms"], float),
                   float(meta.get("f_contact_Hz", 0.0)))


def _to_tuple(value):
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value


@dataclass
class ChatterWindow:
    """Vibration samples of the last revolutions of one path (L0 nodes in contact)."""

    tick: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    u: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def append(self, tick: np.ndarray, u: np.ndarray, w: np.ndarray, keep_s: float, tick_s: float) -> None:
        if len(tick) == 0:
            return
        self.tick = np.concatenate([self.tick, np.asarray(tick, np.int64)])
        self.u = np.concatenate([self.u, np.asarray(u, float).reshape(-1, 3)])
        self.w = np.concatenate([self.w, np.asarray(w, float)])
        cut = int(self.tick[-1]) - int(round(keep_s / tick_s))
        start = int(np.searchsorted(self.tick, cut, side="left"))
        if start > 0:
            self.tick = self.tick[start:].copy()
            self.u = self.u[start:].copy()
            self.w = self.w[start:].copy()

    def clear(self) -> None:
        self.tick = np.zeros(0, np.int64)
        self.u = np.zeros((0, 3))
        self.w = np.zeros(0)


@dataclass
class DynamicsState:
    systems: dict = field(default_factory=dict)          # path -> SystemRecord
    windows: dict = field(default_factory=dict)          # path -> ChatterWindow
    episodes: dict = field(default_factory=dict)         # path -> {active, onset_tick, clear, finding, slot, ...}
    diagrams: dict = field(default_factory=dict)         # tool slot -> [diagram dict]
    points: dict = field(default_factory=dict)           # tool slot -> [operating point dict]
    last_point: dict = field(default_factory=dict)       # path -> {signature, tick, depth_limit_m, ...}
    pending: dict = field(default_factory=dict)          # str(slot) -> {class key: linearised cut, FRF, ...} of the
    #                                                      operation, for its end-of-operation lobe diagrams
    counters: dict = field(default_factory=dict)         # "category:slot:op" -> findings raised
    totals: dict = field(default_factory=dict)           # cumulative energy terms (J)
    values: dict = field(default_factory=dict)           # last macro channel values
    meta: dict = field(default_factory=dict)

    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict[str, np.ndarray] = {}
        systems = {}
        for path, rec in sorted(self.systems.items()):
            a, m = rec.arrays(f"s{path}.")
            arrays.update(a)
            systems[str(path)] = m
        windows = []
        for path, win in sorted(self.windows.items()):
            arrays[f"w{path}.tick"] = win.tick.copy()
            arrays[f"w{path}.u"] = win.u.copy()
            arrays[f"w{path}.w"] = win.w.copy()
            windows.append(int(path))
        meta = {"format": FORMAT_VERSION, "systems": systems, "windows": windows,
                "episodes": _json_safe({str(k): v for k, v in sorted(self.episodes.items())}),
                "diagrams": _json_safe({str(k): v for k, v in sorted(self.diagrams.items())}),
                "points": _json_safe({str(k): v for k, v in sorted(self.points.items())}),
                "last_point": _json_safe({str(k): v for k, v in sorted(self.last_point.items())}),
                "pending": _json_safe({str(k): v for k, v in sorted(self.pending.items())}),
                "counters": _json_safe(dict(sorted(self.counters.items()))),
                "totals": _json_safe(dict(sorted(self.totals.items()))),
                "values": _json_safe(dict(sorted(self.values.items()))),
                "meta": _json_safe(self.meta)}
        json.dumps(meta)
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "DynamicsState":
        out = cls()
        for key, m in meta["systems"].items():
            out.systems[int(key)] = SystemRecord.from_arrays(arrays, m, f"s{int(key)}.")
        for path in meta["windows"]:
            out.windows[int(path)] = ChatterWindow(np.array(arrays[f"w{path}.tick"], np.int64),
                                                   np.array(arrays[f"w{path}.u"], float),
                                                   np.array(arrays[f"w{path}.w"], float))
        out.episodes = {int(k): v for k, v in _json_load(meta["episodes"]).items()}
        out.diagrams = {int(k): v for k, v in _json_load(meta["diagrams"]).items()}
        out.points = {int(k): v for k, v in _json_load(meta["points"]).items()}
        out.last_point = {int(k): v for k, v in _json_load(meta["last_point"]).items()}
        out.pending = {str(k): v for k, v in _json_load(meta.get("pending", {})).items()}
        out.counters = dict(_json_load(meta["counters"]))
        out.totals = dict(_json_load(meta["totals"]))
        out.values = dict(_json_load(meta["values"]))
        out.meta = dict(_json_load(meta["meta"]))
        return out


__all__ = ["DynamicsState", "SystemRecord", "ChatterWindow", "FORMAT_VERSION"]
