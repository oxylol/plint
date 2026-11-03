"""Tool items, holders and project tool instances as SI values (docs/models/tooling.md "Inputs").

The inputs are SIMap views (cnctwin.db / cnctwin.project, architecture 18) or plain mappings
with file keys (``diameter_mm``), which are wrapped into SIMap here. Sourced values
(``{value, src}``) read as their value. Nothing here invents geometry: defaults for values a
catalogue does not state come from db/models/tooling.yaml in the builders, and each use of a
default is recorded in ``ToolSpec.notes``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from ..db.simap import SIMap, Sourced, Table, wrap

ROTATING_TYPES = ("square_endmill", "ball_endmill", "toroidal_endmill", "roughing_endmill", "serrated_endmill",
                  "face_mill", "shoulder_mill", "chamfer_mill", "engraver", "keyseat", "tslot", "dovetail",
                  "lollipop", "slitting_saw", "spot_drill", "centre_drill", "twist_drill", "indexable_drill",
                  "gun_drill", "reamer", "tap_cutting", "tap_forming", "thread_mill", "boring_head")
TURNING_TYPES = ("boring_bar", "turning_insert", "grooving_insert", "parting_insert", "threading_insert",
                 "form_tool")
ALL_TYPES = ROTATING_TYPES + TURNING_TYPES


def as_simap(value) -> SIMap | None:
    if value is None:
        return None
    if isinstance(value, SIMap):
        return value
    if isinstance(value, Mapping):
        return wrap(dict(value))
    raise InputError(f"expected a mapping, got {type(value).__name__}")


def plain(value):
    """The bare value of a sourced entry (text, number, list)."""
    if isinstance(value, Sourced):
        return value.value
    if isinstance(value, Table):
        return float(value)
    return value


def get(mapping, key):
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return getattr(mapping, key, None)


def num(mapping, key, default=None):
    """A number from a mapping (sourced or plain); ``default`` when absent or null."""
    value = plain(get(mapping, key))
    if value is None or isinstance(value, (str, bool, list, tuple, Mapping)):
        return default
    return float(value)


def numlist(mapping, key) -> list[float] | None:
    value = plain(get(mapping, key))
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(plain(v)) for v in value]
    return None


def text(mapping, key, default=None):
    value = plain(get(mapping, key))
    if value is None:
        return default
    return str(value)


def flag(mapping, key, default=False) -> bool:
    value = plain(get(mapping, key))
    if value is None:
        return default
    return bool(value)


def sub(mapping, key):
    value = get(mapping, key)
    if isinstance(value, Sourced):
        value = value.value
    return value if isinstance(value, Mapping) else None


# --------------------------------------------------------------------------- holder
@dataclass
class HolderSpec:
    holder_id: str
    type: str
    taper: str
    gauge_length_m: float
    nose_diameter_m: float
    profile_m: list[tuple[float, float]]     # (z from the nose toward the spindle, diameter)
    bore_m: float | None
    clamp_length_m: float | None
    clamping_torque_Nm: float | None
    runout_tir_m: float | None
    runout_at_m: float | None
    balance_grade: str | None
    max_rad_s: float | None
    mass_kg: float | None
    radial_stiffness_N_per_m: float | None
    axial_float_m: float | None
    raw: SIMap | None = None
    notes: list[str] = field(default_factory=list)


def read_holder(holder, holder_id: str = "") -> HolderSpec | None:
    m = as_simap(holder)
    if m is None:
        return None
    hid = text(m, "id", holder_id) or holder_id
    profile = plain(get(m, "profile_m"))
    rows: list[tuple[float, float]] = []
    if isinstance(profile, (list, tuple)):
        for row in profile:
            row = plain(row)
            if isinstance(row, (list, tuple)) and len(row) == 2:
                rows.append((float(plain(row[0])), float(plain(row[1]))))
    gauge = num(m, "gauge_length_m")
    nose_d = num(m, "nose_diameter_m")
    bore = plain(get(m, "bore_m"))
    bore_m = None
    if isinstance(bore, (int, float)) and not isinstance(bore, bool):
        bore_m = float(bore)
    elif isinstance(bore, (list, tuple)) and bore:
        bore_m = float(plain(bore[-1]))
    notes: list[str] = []
    if nose_d is None and rows:
        nose_d = rows[0][1]
    if nose_d is None and bore_m is not None:
        from . import params as P
        nose_d = P.pf("holder_nose_to_bore_ratio") * bore_m
        notes.append(f"holder {hid}: no nose diameter or profile: outline estimated as a cylinder of "
                     f"{nose_d * 1e3:.1f} mm (holder_nose_to_bore_ratio x bore)")
    if gauge is None or nose_d is None:
        raise InputError(f"holder {hid}: gauge_length_mm and nose_diameter_mm (or profile_mm, or bore_mm) are "
                         f"required")
    if not rows:
        rows = [(0.0, nose_d), (gauge, nose_d)]
    stiffness = num(m, "radial_stiffness_N_per_m")
    return HolderSpec(
        holder_id=hid, type=text(m, "type", ""), taper=text(m, "taper", ""), gauge_length_m=gauge,
        nose_diameter_m=nose_d, profile_m=rows, bore_m=bore_m, clamp_length_m=num(m, "clamp_length_m"),
        clamping_torque_Nm=num(m, "clamping_torque_Nm"), runout_tir_m=num(m, "runout_m"),
        runout_at_m=num(m, "runout_at_m"), balance_grade=text(m, "balance_grade"), max_rad_s=num(m, "max_rad_s"),
        mass_kg=num(m, "mass_kg"), radial_stiffness_N_per_m=stiffness, axial_float_m=num(m, "axial_float_m"),
        raw=m, notes=notes)


# --------------------------------------------------------------------------- project instance
@dataclass
class InstanceSpec:
    """One entry of the project's ``tools.list`` (the physical tool as set up on the machine)."""

    number: int = 0
    path: int = 0
    spindle_id: str | None = None
    stickout_m: float | None = None
    length_offset_m: float | None = None
    radius_offset_m: float | None = None
    wear_length_m: float = 0.0
    wear_radius_m: float = 0.0
    runout_tir_m: float | None = None
    runout_phase_rad: float = 0.0
    tilt_rad: tuple[float, ...] | None = None
    life_group: int | None = None
    sister_of: int | None = None
    life_limit_s: float | None = None
    life_used_s: float | None = None
    initial_VB_m: float | None = None
    initial_edge_radius_m: float | None = None
    comment: str = ""


def read_instance(entry) -> InstanceSpec:
    m = as_simap(entry) if entry is not None else None
    if m is None:
        return InstanceSpec()
    wear = sub(m, "wear_offsets_m")
    life = sub(m, "life")
    initial = sub(m, "initial_wear")
    tilt = plain(get(m, "tilt_rad"))
    tilt_value = None
    if isinstance(tilt, (int, float)) and not isinstance(tilt, bool):
        tilt_value = (float(tilt),)
    elif isinstance(tilt, (list, tuple)) and len(tilt) == 2:
        tilt_value = (float(plain(tilt[0])), float(plain(tilt[1])))
    group = plain(get(life, "group")) if life is not None else None
    sister = plain(get(life, "sister_of")) if life is not None else None
    number = plain(get(m, "number"))
    if number is None:
        raise InputError("a tools.list entry needs its T number")
    path = plain(get(m, "path"))
    return InstanceSpec(
        number=int(number), path=int(path) if path is not None else 0, spindle_id=text(m, "spindle"),
        stickout_m=num(m, "stickout_m"), length_offset_m=num(m, "length_offset_m"),
        radius_offset_m=num(m, "radius_offset_m"),
        wear_length_m=num(wear, "length", 0.0) if wear is not None else 0.0,
        wear_radius_m=num(wear, "radius", 0.0) if wear is not None else 0.0,
        runout_tir_m=num(m, "runout_m"), runout_phase_rad=num(m, "runout_phase_rad", 0.0), tilt_rad=tilt_value,
        life_group=int(group) if group is not None else None, sister_of=int(sister) if sister is not None else None,
        life_limit_s=num(life, "limit_s") if life is not None else None,
        life_used_s=num(life, "used_s") if life is not None else None,
        initial_VB_m=num(initial, "VB_m") if initial is not None else None,
        initial_edge_radius_m=num(initial, "edge_radius_m") if initial is not None else None,
        comment=text(m, "comment", "") or "")


# --------------------------------------------------------------------------- tool item
@dataclass
class ToolSpec:
    tool_id: str
    type: str
    hand: str
    hand_sign: float
    geometry: SIMap
    tool_material_class: str
    tool_material_id: str | None
    coating_id: str | None
    coating_thickness_m: float | None
    max_rad_s: float | None
    balance_grade: str | None
    mass_kg: float | None
    raw: SIMap
    notes: list[str] = field(default_factory=list)
    set_diameter_m: float | None = None      # boring heads: diameter set on the head (from the instance)

    @property
    def rotating(self) -> bool:
        return self.type in ROTATING_TYPES

    # geometry shortcuts -------------------------------------------------
    def g(self, key, default=None):
        return num(self.geometry, key, default)

    def need(self, key) -> float:
        value = num(self.geometry, key)
        if value is None:
            raise InputError(f"tool {self.tool_id} ({self.type}): geometry needs {key}")
        return value

    def default(self, key, value: float, why: str) -> float:
        """The geometry value, or ``value`` (recorded in notes) when the item does not give it."""
        given = num(self.geometry, key)
        if given is not None:
            return given
        self.notes.append(f"{key} not given: {why}")
        return value


def read_tool(tool, tool_id: str = "") -> ToolSpec:
    m = as_simap(tool)
    if m is None:
        raise InputError("tool item missing")
    tid = text(m, "id", tool_id) or tool_id or "inline"
    kind = text(m, "type")
    if kind not in ALL_TYPES:
        raise InputError(f"tool {tid}: unknown type {kind!r}")
    hand = text(m, "hand", "right") or "right"
    notes = []
    if hand not in ("right", "left", "neutral"):
        raise InputError(f"tool {tid}: hand must be right, left or neutral (got {hand!r})")
    if hand == "neutral":
        notes.append("hand neutral: the rake face is taken for the right-hand (M03) cutting direction")
    geometry = sub(m, "geometry")
    if geometry is None:
        raise InputError(f"tool {tid}: geometry missing")
    material = sub(m, "tool_material")
    coating = sub(m, "coating")
    mass = num(m, "mass_kg")
    return ToolSpec(
        tool_id=tid, type=kind, hand=hand, hand_sign=1.0 if hand == "left" else -1.0, geometry=geometry,
        tool_material_class=text(material, "class", "carbide") if material is not None else "carbide",
        tool_material_id=text(material, "id") if material is not None else None,
        coating_id=text(coating, "id") if coating is not None else None,
        coating_thickness_m=num(coating, "thickness_m") if coating is not None else None,
        max_rad_s=num(m, "max_rad_s"), balance_grade=text(m, "balance_grade"), mass_kg=mass, raw=m, notes=notes)


# --------------------------------------------------------------------------- small parsers
_GRADE = re.compile(r"G\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:at|@)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:rpm|1/min|min-1))?",
                    re.IGNORECASE)


def parse_balance_grade(grade) -> tuple[float, float | None] | None:
    """ISO 21940-11 grade text ("G2.5 at 25000 rpm", "G6.3") or number -> (G in m/s, speed in rad/s
    or None)."""
    grade = plain(grade)
    if grade is None:
        return None
    if isinstance(grade, (int, float)) and not isinstance(grade, bool):
        return float(grade) * 1e-3, None
    match = _GRADE.search(str(grade))
    if not match:
        return None
    g_mm_s = float(match.group(1))
    speed = float(match.group(2)) * 2.0 * math.pi / 60.0 if match.group(2) else None
    return g_mm_s * 1e-3, speed


def clearance_primary(geometry, default: float | None = None) -> float | None:
    value = get(geometry, "clearance_rad")
    value_plain = plain(value)
    if isinstance(value_plain, (int, float)) and not isinstance(value_plain, bool):
        return float(value_plain)
    if isinstance(value, Mapping):
        return num(value, "primary", default)
    return default


def as_list(value, count: int, name: str, tool_id: str) -> list[float]:
    """A per-flute list: one number for all, or exactly ``count`` numbers."""
    if value is None:
        raise InputError(f"tool {tool_id}: {name} missing")
    if len(value) == 1:
        return [float(value[0])] * count
    if len(value) != count:
        raise InputError(f"tool {tool_id}: {name} has {len(value)} entries for {count} flutes")
    return [float(v) for v in value]


def vec(values) -> np.ndarray:
    return np.asarray(values, dtype=float)
