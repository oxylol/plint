"""Probing and tool setting for the controller core (docs/dialects.md "Cycles and compensation", probing).

* **G31 skip** (FANUC M/T, Haas M/T; replaces the core's G31 through a ``probing:G31`` G-code handler):
  ``G31 [P1-P4] IP_ F_`` is one buffer-stopping skip move (fanuc-30ibplus-om-2019 4.19-4.20,
  pp. 132-135): PS0035 in tool radius / nose radius compensation, PS0005 without an axis word, PS0011
  without a feed, P outside 1-4 -> TW0034, the continuous high-speed skip P90 and the torque-limit skip
  P98/P99 -> TW0012 (not modelled). Feedrate override, dry run and automatic acceleration are disabled
  in the move unless parameter 6200#7 SKF is set (``skip_feed_override``): the command carries
  ``Feed.override_allowed = False``. The skip number wired to the tool setter
  (``setter_skip_signal``, P2 assumed) makes the move a ``TOOL_SETTER`` command (``setter`` payload
  ``{"measure": "skip", "skip": P, "tool", "offset"}``) on machines with a setter; every other G31 is a
  ``PROBE`` command ``{"skip": P, "stylus": "spindle_probe", "expect": "trigger"}``. #5061-#5066 hold the
  latched position the feedback reports (encoder positions at the skip signal, converted to the work
  coordinates of the moment), or the end point when the signal did not come (core executor). Haas has
  no G31 P; the twin accepts it there too so the built-in setter macros run unchanged.
* **setter skip feedback**: a ``SetterResult`` without latched positions (an automatic setter that
  reports the length only) still gives #5061-: the contact is where the tool tip reaches the setter's
  trigger plane, machine tool-axis value ``z_setter + L`` (``setter_position_m``); a contact outside
  the move, or ``broken``, is no signal. The last result is kept in ``PathState.sys["setter_result"]``.
* **G37 / G36** (``SetterMeasure``, replaces the core's ToolMeasure): the core behaviour (measuring
  window epsilon, PS0080/PS0081/PS0082/PS0083, offset += error from latched positions, length offset =
  ``SetterResult.length_m`` - wear) plus: a reported length further than epsilon from the stored
  length is PS0080 like a signal outside the window, and a ``broken`` result without a latch is PS0080
  (no measuring signal). On lathes the X/Z offsets are measured from latched positions only.
* **built-in macros**: O9810-O9814 (spindle probe) and O9851/O9853 (tool setter) from
  :mod:`cnctwin.controller.macrolib`, registered as the ``probing`` library.

Assumptions and their consequences: one tool setter per path with a plane trigger perpendicular to the
tool axis (as the motion domain models it): a setter skip contact is computed on the tool axis only; the
setter position is the G53 frame of the gauge line (a machine whose base frame differs needs
``setter_position_mm``). Error bound: latched positions are whole counts (1 increment); the derived
contact is exact to half a count of the reported length. No randomness; no bus fields.
"""

from __future__ import annotations

from . import hooks
from .alarms import ControllerAlarm, register_alarm_codes
from .cycles.common import params as cycle_params
from .dialects.base import GSpec
from .expr import round_half_away
from .hooks import Step
from .macrolib import LIBRARY, setter_position
from .measure import ToolMeasure

HANDLER_KEY = "probing:G31"
UNMODELLED_SKIPS = (90, 98, 99)

FANUC_ALARMS = {"skip_p_illegal": "TW0034"}
HAAS_ALARMS = {"skip_p_illegal": "TW0034"}
TEXTS = {"TW0034": "G31 P OUT OF RANGE (P1 TO P4)"}


def _setter_dict(setter) -> dict:
    return {"tool": int(setter.tool), "length_m": None if setter.length_m is None else float(setter.length_m),
            "diameter_m": None if setter.diameter_m is None else float(setter.diameter_m),
            "broken": bool(setter.broken)}


class SkipMove:
    """G31 [P_] IP_ F_ as a buffer-stopping PROBE (or TOOL_SETTER) command, and its setter feedback."""

    def execute(self, ex, code: str, spec: GSpec, bw) -> bool:
        s, h = ex.s, ex.handler
        if s.modal.get("7", "G40") != "G40":
            raise ControllerAlarm("g31_in_comp", "G31 in tool radius compensation")
        word = bw.take("P")
        signal = 1
        if word is not None:
            signal = h.integer(word)
            if signal in UNMODELLED_SKIPS:
                raise ControllerAlarm("unsupported_function", f"G31 P{signal} (continuous high-speed / torque limit skip)")
            if not 1 <= signal <= 4:
                raise ControllerAlarm("skip_p_illegal", f"G31 P{signal}")
        target = h.axis_targets(bw)
        if not target:
            raise ControllerAlarm("no_data_after_address", "G31 without axis words")
        if not s.feed_set or s.feed_si <= 0.0:
            raise ControllerAlarm("feed_zero", "G31 without F")
        p = cycle_params(ex)
        override = bool(p["skip_feed_override"])
        goal = {letter: ex.si(letter, counts) for letter, counts in target.items()}
        setter = word is not None and signal == int(round(float(p["setter_skip_signal"]))) \
            and ex.machine.tool_setter is not None
        if not setter:
            ex.run_step(Step("probe", goal, override=override,
                             probe={"skip": signal, "stylus": "spindle_probe", "expect": "trigger"}))
            return True
        payload = {"measure": "skip", "skip": signal, "tool": int(s.tool), "offset": int(s.h_number)}
        ex.run_step(Step("setter", goal, override=override, provider=HANDLER_KEY,
                         data={"signal": signal, "start": dict(s.mach)}, probe=payload))
        waiting = s.waiting
        if waiting is not None and waiting.get("kind") == "setter":
            waiting.setdefault("data", {})["end"] = dict(waiting.get("item_end", s.mach))
        return True

    def resume(self, ex, data: dict, feedback) -> list[Step]:
        """Setter skip feedback: a SetterResult without a latch gives the contact from the length."""
        s = ex.s
        setter = feedback.setter if feedback is not None else None
        if setter is None:
            return []
        s.sys["setter_result"] = _setter_dict(setter)
        probe = feedback.probe
        if probe is not None and probe.triggered and probe.machine_pos and not setter.broken:
            return []                                    # the latched positions are already in #5061-
        start, end = data.get("start", {}), data.get("end", {})
        contact = None if setter.broken or setter.length_m is None else self._contact(ex, start, end,
                                                                                    float(setter.length_m))
        if contact is None:
            s.skip = ex.program_from_machine(dict(end))
            s.skip_triggered = False
            return []
        s.skip = ex.program_from_machine(contact)
        s.skip_triggered = True
        s.mach = dict(contact)                           # the axes stop at the contact
        s.pos = ex.program_from_machine(contact)
        return []

    def _contact(self, ex, start: dict, end: dict, length_m: float) -> dict | None:
        position = setter_position(ex.machine, cycle_params(ex))
        letter = ex.s.tool_axis_letter or "Z"
        if position is None or letter not in ex.by_letter or letter not in start or letter not in end:
            return None
        q = ex.by_letter[letter].quantum
        z_contact = int(round_half_away((position["XYZ".index(letter)] + length_m) / q)) if letter in "XYZ" else None
        if z_contact is None:
            return None
        z0, z1 = int(start[letter]), int(end[letter])
        if z0 == z1 or not min(z0, z1) <= z_contact <= max(z0, z1):
            return None
        t = (z_contact - z0) / (z1 - z0)
        out = {}
        for axis in ex.axes:
            a0, a1 = int(start.get(axis.letter, 0)), int(end.get(axis.letter, 0))
            out[axis.letter] = int(round_half_away(a0 + (a1 - a0) * t))
        out[letter] = z_contact
        return out


class SetterMeasure(ToolMeasure):
    """FANUC/Haas G37 (and lathe G36/G37) on the tool setter: the core measurement plus the window check
    of a reported length and the broken-tool result."""

    def resume(self, ex, data: dict, feedback) -> list[Step]:
        setter = feedback.setter if feedback is not None else None
        if setter is not None:
            ex.s.sys["setter_result"] = _setter_dict(setter)
            latched = feedback.probe is not None and feedback.probe.triggered
            if setter.broken and not latched:
                raise ControllerAlarm("g37_signal", "no measuring signal: the tool did not reach the setter")
            if setter.length_m is not None and data["axis"] == "Z" and not data["lathe"]:
                row = ex.s.tool_offsets.get(str(data["offset"]), [0] * 9)
                stored = (row[0] + row[1]) * ex.machine.lin_increment_m
                error = float(setter.length_m) - stored
                if abs(error) > float(data["epsilon"]):
                    raise ControllerAlarm("g37_signal", f"measured length {error * 1e3:+.3f} mm from the offset, "
                                                        "outside the measuring window")
        return super().resume(ex, data, feedback)


SKIP = SkipMove()


def register(replace: bool = True) -> None:
    """(Re-)register G31 for FANUC and Haas, G36/G37, the macro library and the alarm numbers."""
    spec = GSpec(0, "ext", HANDLER_KEY, axis_words=True)
    for dialect in ("fanuc", "haas"):
        for variant in ("M", "T"):
            hooks.register_gcode("G31", spec, variant=variant, dialect=dialect, replace=replace)
    hooks.register_gcode_handler(HANDLER_KEY, SKIP, replace=replace)
    hooks.register_resumer(HANDLER_KEY, SKIP, replace=replace)
    measure = SetterMeasure()
    hooks.register_probe("G37", measure, replace=True)
    hooks.register_probe("G36", measure, replace=True)
    hooks.register_library("probing", LIBRARY, replace=replace)
    register_alarm_codes("fanuc", FANUC_ALARMS, TEXTS)
    register_alarm_codes("haas", HAAS_ALARMS, TEXTS)


register()

__all__ = ["SkipMove", "SetterMeasure", "SKIP", "register", "HANDLER_KEY"]
