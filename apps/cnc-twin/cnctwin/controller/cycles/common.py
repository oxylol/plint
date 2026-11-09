"""Shared helpers of the cycle, compensation, probing, tool-life, restart and TCP providers.

* :func:`params` - the resolved parameters of these functions: ``db/models/controller-cycles.yaml``
  (each one standing for the FANUC/Haas parameter named in its note) overridden by the project's
  ``physics.controller.cycle_parameters`` (keys with their unit suffix, converted to SI).
* word access for :class:`~cnctwin.controller.hooks.CycleCall` data (the exact literal of a word with
  the dialect's decimal-point rules), feed helpers, plane axes.
* :func:`register_alarms` - condition keys -> alarm numbers and built-in texts per dialect
  (``alarms.register_alarm_codes``; a controller database entry with the same ``kind`` still wins).

Nothing here keeps state: caches hang on the executor object and are derived from its immutable
context only (they are rebuilt identically on resume).
"""

from __future__ import annotations

import math

from ...db import keyunits
from ...db.models import params as model_params
from ..alarms import ControllerAlarm, register_alarm_codes
from ..pipeline import PLANES

MODEL = "controller-cycles"


def resolve(overrides: dict | None) -> dict:
    """Model defaults overridden by ``cycle_parameters`` (keys with their unit suffix, converted to SI)."""
    values = dict(model_params(MODEL))
    for key, value in (overrides or {}).items():
        name = str(key)
        if name in values:
            values[name] = value
            continue
        si_name, si_value = keyunits.to_si(name, value)
        if si_name not in values:
            raise ControllerAlarm("unsupported_function", f"unknown cycle parameter {name}")
        values[si_name] = si_value
    return values


def params(ex) -> dict:
    """Resolved parameters (SI) for one executor: model defaults + ``physics.controller.cycle_parameters``."""
    cached = getattr(ex, "_cycle_params", None)
    if cached is not None:
        return cached
    values = resolve(ex.p.get("cycle_parameters"))
    ex._cycle_params = values
    return values


def word(call, letter: str):
    """The block's word ``letter`` as a WordValue (literal form kept), or None."""
    from ..dialects.fanuc import WordValue

    raw = call.extra.get("word_values", {}).get(letter)
    if raw is None:
        return None
    return WordValue(letter, float(raw[0]), bool(raw[1]), "", bool(raw[2]))


def lin(ex, call, letter: str, default: float | None = None) -> float | None:
    """A length word in SI by the dialect's decimal-point rule."""
    w = word(call, letter)
    return ex.handler.lin(w) if w is not None else default


def integer(ex, call, letter: str, default: int | None = None) -> int | None:
    w = word(call, letter)
    return ex.handler.integer(w) if w is not None else default


def plane_axes(plane: str) -> tuple[str, str, str]:
    return PLANES[plane]


def feed_of(ex, call) -> float:
    """The cycle's cutting feed in SI of the current feed mode (F of the block, else modal F)."""
    feed = call.data.get("F")
    if feed is None and ex.s.feed_set:
        feed = ex.s.feed_si
    if feed is None or feed <= 0.0:
        raise ControllerAlarm("feed_zero", f"{call.code} without F")
    return float(feed)


def spindle_payload(ex, direction: int, rpm: float | None = None) -> dict:
    """A SPINDLE command payload like the handler's (M03 -> omega < 0), without changing the modal state."""
    s = ex.s
    machine = ex.machine
    spindle_id = machine.path_spindle[ex.ctx.path]
    info = machine.spindles.get(spindle_id)
    top = info.max_rad_s if info is not None else 0.0
    speed = (s.spindle.s_value if rpm is None else rpm) * 2.0 * math.pi / 60.0
    if top > 0.0:
        speed = min(speed, top)
    return {"id": spindle_id, "omega_rad_s": direction * speed, "css_m_s": None, "max_rad_s": top,
            "gear": s.spindle.gear, "orient_rad": None, "direction": direction, "css_axis": None}


def drop_null_moves(steps: list, start: dict) -> list:
    """Remove rapid / linear steps that would not move any axis (the tool is already there)."""
    here = dict(start)
    out = []
    for step in steps:
        if step.kind in ("rapid", "linear") and not step.machine:
            if all(abs(here.get(l, v) - v) <= 0.0 for l, v in step.target.items()):
                continue
            here.update(step.target)
        elif step.kind == "arc":
            here.update(step.target)
        out.append(step)
    return out


def register_alarms(fanuc: dict[str, tuple[str, str]], haas: dict[str, tuple[str, str]]) -> None:
    """``key -> (code, built-in text)`` for the FANUC and Haas dialects."""
    for dialect, table in (("fanuc", fanuc), ("haas", haas)):
        register_alarm_codes(dialect, {k: v[0] for k, v in table.items()}, {v[0]: v[1] for v in table.values()})


def signed_axis(text: str) -> tuple[str, float]:
    """"+X" / "-Y" -> ("X", +1.0)."""
    t = str(text).strip().upper()
    if len(t) == 2 and t[0] in "+-" and t[1] in "XYZ":
        return t[1], (1.0 if t[0] == "+" else -1.0)
    if len(t) == 1 and t in "XYZ":
        return t, 1.0
    raise ControllerAlarm("unsupported_function", f"axis direction {text!r} (write +X, -Y ...)")


def program_delta(ex, work_delta: dict[str, float]) -> dict[str, float]:
    """A displacement given in work-frame axes (machine directions) expressed in the program frame
    (through the inverse of the linear part of mirror/scaling/rotation/feature transforms)."""
    transform = ex.transform()
    if transform.identity:
        return dict(work_delta)
    vec = [work_delta.get(l, 0.0) for l in "XYZ"]
    lin_part = transform.linear
    det = (lin_part[0][0] * (lin_part[1][1] * lin_part[2][2] - lin_part[1][2] * lin_part[2][1])
           - lin_part[0][1] * (lin_part[1][0] * lin_part[2][2] - lin_part[1][2] * lin_part[2][0])
           + lin_part[0][2] * (lin_part[1][0] * lin_part[2][1] - lin_part[1][1] * lin_part[2][0]))
    import numpy as np

    out = np.linalg.solve(np.array(lin_part, dtype=float), np.array(vec, dtype=float)) if det != 0.0 else vec
    return {l: float(out[i]) for i, l in enumerate("XYZ") if l in ex.by_letter}


FANUC_CYCLE_ALARMS = {
    "drill_plane_change": ("PS0549", "CANNOT CHANGE PLANE(DRILL MODE)"),
    "rigid_s_illegal": ("PS0200", "ILLEGAL S CODE COMMAND"),
    "rigid_feed_missing": ("PS0201", "FEEDRATE NOT FOUND IN RIGID TAP"),
    "rigid_program_miss": ("PS0203", "PROGRAM MISS AT RIGID TAPPING"),
    "rigid_axis_operation": ("PS0204", "ILLEGAL AXIS OPERATION"),
    "rigid_plane_change": ("PS0206", "CAN NOT CHANGE PLANE (RIGID TAP)"),
    "box_axis": ("PS0330", "ILLEGAL AXIS COMMAND IS IN THE TURNING CANNED CYCLE"),
    "mrc_pq_missing": ("PS0061", "P OR Q COMMAND IS NOT IN THE MULTIPLE REPETIVE CYCLES BLOCK"),
    "mrc_cut_amount": ("PS0062", "THE CUTTING AMOUNT IS ILLEGAL IN THE ROUGH CUTTING CYCLE"),
    "mrc_sequence_missing": ("PS0063", "THE BLOCK OF A SPECIFIED SEQUENCE NUMBER IS NOT FOUND"),
    "mrc_not_monotone_1": ("PS0064", "THE FINISHING SHAPE IS NOT A MONOTONOUS CHANGE(FIRST AXES)"),
    "mrc_first_block": ("PS0065", "G00/G01 IS NOT IN THE FIRST BLOCK OF SHAPE PROGRAM"),
    "mrc_unavailable_block": ("PS0066", "UNAVAILABLE COMMAND IS IN THE MULTIPLE REPETIVE CYCLES BLOCK"),
    "mrc_last_block": ("PS0069", "LAST BLOCK OF SHAPE PROGRAM IS AN ILLEGAL COMMAND"),
    "thread_nose_angle": ("PS0315", "ILLEGAL NOSE ANGLE COMMAND IS IN THE THREAD CUTTING CYCLE"),
    "thread_cut_amount": ("PS0316", "ILLEGAL CUTTING AMOUNT IS IN THE THREAD CUTTING CYCLE"),
    "thread_command": ("PS0317", "ILLEGAL THREAD COMMAND IS IN THE THREAD CUTTING CYCLE"),
    "groove_relief": ("PS0318", "ILLEGAL RELIEF AMOUNT IS IN THE DRILLING CYCLE"),
    "groove_end_point": ("PS0319", "THE END POINT COMMAND IS ILLEGAL IN THE DRILLING CYCLE"),
    "groove_amount": ("PS0320", "ILLEGAL MOVEMENT AMOUNT/CUTTING AMOUNT IS IN THE DRILLING CYCLE"),
    "pattern_repeat": ("PS0321", "ILLEGAL REPEATED TIME IS IN THE PATTERN REPEATING CYCLE"),
    "mrc_over_start": ("PS0322", "FINISHING SHAPE WHICH OVER OF STARTING POINT"),
    "mrc_shape_command": ("PS0325", "UNAVAILABLE COMMAND IS IN SHAPE PROGRAM"),
    "mrc_not_monotone_2": ("PS0329", "THE FINISHING SHAPE IS NOT A MONOTONOUS CHANGE(SECOND AXES)"),
    "mrc_offset_not_cancelled": ("PS0538", "OFFSET IS NOT CANCELED"),
    "thread_lead": ("PS0313", "ILLEGAL LEAD COMMAND"),
}
HAAS_CYCLE_ALARMS = {key: ("TW0030", "CANNED CYCLE DATA ERROR") for key in FANUC_CYCLE_ALARMS}
HAAS_CYCLE_ALARMS["thread_nose_angle"] = ("605", "Invalid Tool Nose Angle")
# G70/G71/G72 bolt hole patterns (cycles.patterns): the manual names no alarm numbers for these conditions
HAAS_CYCLE_ALARMS["pattern_format"] = ("TW0030", "CANNED CYCLE DATA ERROR")
HAAS_CYCLE_ALARMS["pattern_without_cycle"] = ("TW0037", "BOLT HOLE PATTERN WITHOUT AN ACTIVE CANNED CYCLE")

__all__ = ["drop_null_moves", "params", "word", "lin", "integer", "plane_axes", "feed_of", "spindle_payload", "register_alarms",
           "signed_axis", "program_delta", "FANUC_CYCLE_ALARMS", "HAAS_CYCLE_ALARMS"]
