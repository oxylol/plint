"""SimState field of tooling: the static geometry of every tool slot and its effective edge.

``ToolingState.tools[slot]`` is the :class:`cnctwin.tooling.build.ToolGeometry` (nominal edges,
envelope, profiles, properties); ``effective[slot]`` the EdgeModel the hot loop uses this step
(nominal + composition inputs ``edge_state[slot]``), with its angle table and element frames;
``version[slot]`` counts the changes of the effective edge. ``to_arrays``/``from_arrays`` store
all of it (arrays by name, the rest as JSON metadata), so a resumed run continues bit for bit
without the project.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EdgeModel
from .abi import EdgeFrameT, EdgeState, EdgeTrig, Envelope, GroupTable, InsertProfile, ThreadProfile
from .edges import ElementArrays
from .props import Balance, CoolantHoles, MassProps, Section, ToolProps

FORMAT_VERSION = 1


@dataclass
class ToolingState:
    tools: list = field(default_factory=list)            # ToolGeometry per slot
    effective: list = field(default_factory=list)        # EdgeModel per slot
    edge_state: list = field(default_factory=list)       # EdgeState per slot
    trig: list = field(default_factory=list)             # EdgeTrig of the effective edge
    frames: list = field(default_factory=list)           # EdgeFrameT of the effective edge
    version: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    initial_edge_radius_m: np.ndarray = field(default_factory=lambda: np.zeros(0))   # NaN: none (instance)

    # ------------------------------------------------------------------ access
    def geometry(self, slot: int):
        return self.tools[slot]

    def nominal(self, slot: int) -> EdgeModel:
        return self.tools[slot].edge

    def edge_model(self, slot: int) -> EdgeModel:
        """Effective EdgeModel of a slot (thermal's ToolEdges protocol)."""
        return self.effective[slot]

    def envelope(self, slot: int) -> Envelope:
        return self.tools[slot].envelope

    def __len__(self) -> int:
        return len(self.tools)

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict[str, np.ndarray] = {"version": self.version.copy(),
                                         "initial_edge_radius_m": self.initial_edge_radius_m.copy()}
        slots_meta = []
        for slot, geo in enumerate(self.tools):
            prefix = f"s{slot}."
            _put(arrays, prefix + "nom.", geo.edge)
            _put(arrays, prefix + "eff.", self.effective[slot])
            _put(arrays, prefix + "st.", self.edge_state[slot])
            _put(arrays, prefix + "trig.", self.trig[slot])
            _put(arrays, prefix + "frm.", self.frames[slot])
            _put(arrays, prefix + "nfrm.", geo.frames)
            _put(arrays, prefix + "ntrig.", geo.trig)
            _put(arrays, prefix + "grp.", geo.groups)
            _put(arrays, prefix + "env.", geo.envelope)
            for name in ("tangent_C", "normal_C", "cutdir_C", "rake_normal_C"):
                arrays[prefix + "el." + name] = np.array(getattr(geo.elements, name))
            if geo.thread is not None:
                _put(arrays, prefix + "thr.", geo.thread)
            if geo.insert is not None:
                _put(arrays, prefix + "ins.", geo.insert)
            slots_meta.append({
                "slot": geo.slot, "number": geo.number, "tool_id": geo.tool_id, "holder_id": geo.holder_id,
                "type": geo.type, "hand": geo.hand, "hand_sign": geo.hand_sign, "rotating": geo.rotating,
                "layout": _json(geo.layout), "info": _json(geo.info), "notes": list(geo.notes),
                "props": props_to_dict(geo.props), "runout_tir_m": geo.runout_tir_m,
                "runout_source": geo.runout_source, "has_thread": geo.thread is not None,
                "has_insert": geo.insert is not None})
        return arrays, {"format": FORMAT_VERSION, "slots": slots_meta}

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "ToolingState":
        from .build import ToolGeometry
        state = cls()
        state.version = np.array(arrays["version"], dtype=np.int64)
        state.initial_edge_radius_m = np.array(arrays["initial_edge_radius_m"], dtype=float)
        for entry in meta["slots"]:
            slot = int(entry["slot"])
            prefix = f"s{slot}."
            nominal = _get(arrays, prefix + "nom.", EdgeModel)
            elements = ElementArrays(**{name: None for name in ElementArrays.__dataclass_fields__})
            for name in ("tangent_C", "normal_C", "cutdir_C", "rake_normal_C"):
                setattr(elements, name, np.array(arrays[prefix + "el." + name]))
            geo = ToolGeometry(
                slot=slot, number=int(entry["number"]), tool_id=entry["tool_id"], holder_id=entry["holder_id"],
                type=entry["type"], hand=entry["hand"], hand_sign=float(entry["hand_sign"]),
                rotating=bool(entry["rotating"]), edge=nominal, groups=_get(arrays, prefix + "grp.", GroupTable),
                trig=_get(arrays, prefix + "ntrig.", EdgeTrig), frames=_get(arrays, prefix + "nfrm.", EdgeFrameT),
                elements=elements, envelope=_get(arrays, prefix + "env.", Envelope), layout=dict(entry["layout"]),
                thread=_get(arrays, prefix + "thr.", ThreadProfile) if entry["has_thread"] else None,
                insert=_get(arrays, prefix + "ins.", InsertProfile) if entry["has_insert"] else None,
                props=props_from_dict(entry["props"]), info=dict(entry["info"]), notes=list(entry["notes"]),
                runout_tir_m=float(entry["runout_tir_m"]), runout_source=entry["runout_source"])
            state.tools.append(geo)
            state.effective.append(_get(arrays, prefix + "eff.", EdgeModel))
            state.edge_state.append(_get(arrays, prefix + "st.", EdgeState))
            state.trig.append(_get(arrays, prefix + "trig.", EdgeTrig))
            state.frames.append(_get(arrays, prefix + "frm.", EdgeFrameT))
        return state


def _put(arrays: dict, prefix: str, record) -> None:
    for name in record._fields:
        arrays[prefix + name] = np.array(getattr(record, name))


def _get(arrays: dict, prefix: str, cls):
    return cls(**{name: np.ascontiguousarray(np.array(arrays[prefix + name])) for name in cls._fields})


def _json(value):
    """JSON-safe copy (numbers, text, lists, dicts); arrays become lists, other objects are dropped."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            converted = _json(item)
            if converted is not _DROP:
                out[str(key)] = converted
        return out
    if isinstance(value, (list, tuple)):
        items = [_json(v) for v in value]
        return [v for v in items if v is not _DROP]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _DROP


_DROP = object()


def props_to_dict(props: ToolProps) -> dict:
    out = {}
    for f in dataclasses.fields(props):
        value = getattr(props, f.name)
        if dataclasses.is_dataclass(value):
            out[f.name] = {k: _json(v) for k, v in dataclasses.asdict(value).items()}
        elif isinstance(value, list) and value and dataclasses.is_dataclass(value[0]):
            out[f.name] = [{k: _json(v) for k, v in dataclasses.asdict(item).items()} for item in value]
        else:
            converted = _json(value)
            out[f.name] = converted if converted is not _DROP else None
    return out


def props_from_dict(data: dict) -> ToolProps:
    def obj(cls, value):
        return cls(**value) if value is not None else None

    coolant = data.get("coolant")
    if coolant is not None:
        coolant = dict(coolant)
        coolant["exit_pos_T_m"] = np.array(coolant["exit_pos_T_m"])
        coolant["exit_dir_T"] = np.array(coolant["exit_dir_T"])
    values = dict(data)
    values.update({
        "fluted": obj(Section, data.get("fluted")),
        "sections": [Section(**s) for s in data.get("sections") or []],
        "tool_mass": obj(MassProps, data.get("tool_mass")), "holder_mass": obj(MassProps, data.get("holder_mass")),
        "assembly_mass": obj(MassProps, data.get("assembly_mass")),
        "balance_tool": obj(Balance, data.get("balance_tool")), "balance_holder": obj(Balance, data.get("balance_holder")),
        "coolant": obj(CoolantHoles, coolant), "shank_dir_T": tuple(data["shank_dir_T"])})
    return ToolProps(**values)
