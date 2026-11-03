"""``physics.motion`` project settings (numerical settings of the motion domain).

The declaration :data:`FIELDS` is the single source: :func:`settings` reads a project's
``physics.motion`` (an SI view or the raw mapping) into :class:`MotionSettings` and rejects
out-of-range values; :func:`schema_node` builds the :mod:`cnctwin.db.schema` node for the project
schema (imported lazily).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

# name -> (file key, kind, default, lo, hi, doc)
FIELDS: dict[str, tuple[str, str, float, float, float, str]] = {
    "every_n_macro": ("every_n_macro", "count", 1, 1, 1,
                      "motion plans and absorbs every macro step (the L1 loop is compiled; the value is fixed)"),
    "servo_omega_h": ("servo_omega_h", "number", 0.25, 0.01, 1.0,
                      "drive and spindle sub-steps per L1 tick: omega_bandwidth x sub-step <= this (RK4)"),
    "friction_h": ("friction_h", "number", 1.0, 0.05, 2.5,
                   "sub-step bound from the regularised friction: h F_c / (m v_s) <= this (RK4 stability)"),
    "operator_stops": ("operator_stops", "flag", 1, 0, 1,
                       "1: M00/M01/single-block/message stops wait for the operator model; 0: restart at once"),
    "contact_from_loop": ("contact_from_loop", "flag", 1, 0, 1,
                          "1: split feed ticks into cutting/air cutting by the hot loop's per-tick contact flag "
                          "when it sets one, else by removal's ToolContact of the step"),
}


@dataclass(frozen=True)
class MotionSettings:
    every_n_macro: int = 1
    servo_omega_h: float = 0.25
    friction_h: float = 1.0
    operator_stops: bool = True
    contact_from_loop: bool = True


def _lookup(section, name: str, file_key: str):
    if section is None:
        return None
    for key in (name, file_key):
        value = section.get(key) if isinstance(section, Mapping) else getattr(section, key, None)
        if value is None:
            continue
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if hasattr(value, "value") and hasattr(value, "src"):
            value = value.value
        return float(value)
    return None


def settings(section) -> MotionSettings:
    """Settings from ``physics.motion`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(MotionSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        value = _lookup(section, item.name, file_key)
        if value is None:
            value = default
        if not lo <= value <= hi:
            raise InputError(f"physics.motion.{file_key} = {value} is outside [{lo}, {hi}]")
        if kind == "count":
            values[item.name] = int(round(value))
        elif kind == "flag":
            values[item.name] = bool(round(value))
        else:
            values[item.name] = float(value)
    return MotionSettings(**values)


def schema_node():
    """The ``physics.motion`` node of the project schema (cnctwin.db.schema)."""
    from ..db.schema import Count, Flag, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        elif kind == "flag":
            nodes[file_key] = Flag(doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc)
    return Struct(nodes, doc="motion settings (docs/models/motion.md)")
