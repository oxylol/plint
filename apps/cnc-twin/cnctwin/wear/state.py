"""The wear domain's SimState field (checkpointable: to_arrays / from_arrays).

Everything a bit-identical continuation needs is here: per tool slot the element state (wear depths,
coating, crater, notch, chips, BUE, thermal damage and cracks, draw counters), the derived element
geometry, the wear law in force (published or calibrated Usui branches, hardness tables), the
chipping law, the critical sections with their instance strengths and fatigue sums, the fallback
response table, usage counters, the life record and per-part rows. The EdgeModel itself and the
protocol providers are not state: WearDomain attaches them again after a resume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .abi import ChipLaw, ElemGeo, ElemState, Law
from .schema import WearSettings

SECTION_ARRAYS = ("z_m", "Zb_m3", "Zp_m3", "d_m", "volume_b_m3", "volume_t_m3", "strength_b_Pa", "strength_t_Pa",
                  "fatigue")
FALLBACK_ARRAYS = ("speed_m_s", "flank_T_K", "rake_T_K", "flank_sigma_Pa", "rake_sigma_Pa", "chip_ratio")


def _json_safe(value):
    """Floats that JSON can hold (inf -> None, NaN -> None), recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass
class SlotWear:
    slot: int
    number: int
    tool_id: str
    holder_id: str
    tool_class: str
    rotating: bool
    elem: ElemState
    geo: ElemGeo
    law: Law
    chip: ChipLaw
    sections: dict                      # name -> [nsec] arrays (SECTION_ARRAYS)
    section_names: list
    fallback: dict                      # FALLBACK_ARRAYS -> arrays (empty arrays: no table)
    finishes: np.ndarray                # [nt, ne] int8: the element generated final surface
    props: dict = field(default_factory=dict)      # scalar material and law data (JSON)
    info: dict = field(default_factory=dict)       # sources, calibration report, notes (JSON)
    counters: dict = field(default_factory=dict)
    life: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    per_part: list = field(default_factory=list)
    part_start: dict = field(default_factory=dict)

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = {}
        for group, nt in (("elem", self.elem), ("geo", self.geo), ("law", self.law), ("chip", self.chip)):
            for name in nt._fields:
                arrays[f"{prefix}{group}.{name}"] = np.array(getattr(nt, name))
        for name in SECTION_ARRAYS:
            arrays[f"{prefix}sec.{name}"] = np.array(self.sections[name])
        for name in FALLBACK_ARRAYS:
            arrays[f"{prefix}fb.{name}"] = np.array(self.fallback[name])
        arrays[f"{prefix}finishes"] = np.array(self.finishes)
        meta = {"slot": self.slot, "number": self.number, "tool_id": self.tool_id, "holder_id": self.holder_id,
                "tool_class": self.tool_class, "rotating": self.rotating, "section_names": list(self.section_names),
                "props": _json_safe(self.props), "info": _json_safe(self.info), "counters": _json_safe(self.counters),
                "life": _json_safe(self.life), "flags": _json_safe(self.flags), "per_part": _json_safe(self.per_part),
                "part_start": _json_safe(self.part_start)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str) -> "SlotWear":
        def load(group, layout):
            def array(name):
                key = f"{prefix}{group}.{name}"
                if key not in arrays and group == "elem":
                    # an element array added after the checkpoint was written (edge_fat, 2026-11-09): it starts empty
                    return np.zeros_like(np.ascontiguousarray(arrays[f"{prefix}elem.vn"]), dtype=np.float64)
                return np.ascontiguousarray(arrays[key]).copy()
            return layout(*(array(name) for name in layout._fields))

        life = dict(meta["life"])
        for key in ("remaining_s", "replacement_s"):
            if key in life and life[key] is None:
                life[key] = math.inf
        return cls(slot=int(meta["slot"]), number=int(meta["number"]), tool_id=str(meta["tool_id"]),
                   holder_id=str(meta["holder_id"]), tool_class=str(meta["tool_class"]),
                   rotating=bool(meta["rotating"]), elem=load("elem", ElemState), geo=load("geo", ElemGeo),
                   law=load("law", Law), chip=load("chip", ChipLaw),
                   sections={name: np.ascontiguousarray(arrays[f"{prefix}sec.{name}"]).copy()
                             for name in SECTION_ARRAYS},
                   section_names=list(meta["section_names"]),
                   fallback={name: np.ascontiguousarray(arrays[f"{prefix}fb.{name}"]).copy() for name in FALLBACK_ARRAYS},
                   finishes=np.ascontiguousarray(arrays[f"{prefix}finishes"]).copy(), props=dict(meta["props"]),
                   info=dict(meta["info"]), counters=dict(meta["counters"]), life=life, flags=dict(meta["flags"]),
                   per_part=list(meta["per_part"]), part_start=dict(meta["part_start"]))


@dataclass
class WearState:
    settings: WearSettings
    seed: int
    slots: dict = field(default_factory=dict)          # slot -> SlotWear
    part_index: int = 0
    op_index: int = 0
    time_s: float = 0.0
    path_slot: dict = field(default_factory=dict)      # path -> tool slot seen in the last step
    values: dict = field(default_factory=dict)         # channel values of the last step
    estimated: list = field(default_factory=list)      # model parameters / fallbacks in use (report)
    trace_notes: dict = field(default_factory=dict)    # once-only trace events already written (key -> id)

    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict = {}
        meta = {"settings": dict(self.settings.__dict__), "seed": int(self.seed), "part_index": self.part_index,
                "op_index": self.op_index, "time_s": self.time_s,
                "path_slot": [[int(k), int(v)] for k, v in sorted(self.path_slot.items())],
                "values": _json_safe({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                      for k, v in sorted(self.values.items())}),
                "estimated": list(self.estimated), "trace_notes": dict(sorted(self.trace_notes.items())), "slots": []}
        for slot, sw in sorted(self.slots.items()):
            a, m = sw.to_arrays(f"slot.{slot}.")
            arrays.update(a)
            meta["slots"].append(m)
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "WearState":
        obj = cls(settings=WearSettings(**meta["settings"]), seed=int(meta["seed"]))
        obj.part_index = int(meta["part_index"])
        obj.op_index = int(meta["op_index"])
        obj.time_s = float(meta["time_s"])
        obj.path_slot = {int(k): int(v) for k, v in meta["path_slot"]}
        obj.values = {k: (np.array(v, dtype=float) if isinstance(v, list) else (math.nan if v is None else v))
                      for k, v in meta["values"].items()}
        obj.estimated = list(meta["estimated"])
        obj.trace_notes = dict(meta["trace_notes"])
        for m in meta["slots"]:
            slot = int(m["slot"])
            obj.slots[slot] = SlotWear.from_arrays(arrays, m, f"slot.{slot}.")
        return obj


__all__ = ["SlotWear", "WearState", "SECTION_ARRAYS", "FALLBACK_ARRAYS"]
