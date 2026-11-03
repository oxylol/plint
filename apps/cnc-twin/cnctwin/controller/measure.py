"""Automatic tool length / tool offset measurement with the tool setter (FANUC G37, lathe G36/G37,
Haas G37): the core's default probing provider (the probing agent may replace it with
``register_probe(..., replace=True)``).

FANUC G37 IP_ (FANUC 0i-F operator's manual, "automatic tool length measurement"): the tool moves
at rapid to the point gamma before the measuring position IP (parameter 6251), then at the
measuring feed (6241) toward it until the setter's measuring-position-reached signal; the
difference e between the position where the signal came and the commanded measuring position is
added to the active offset (H for the M series, the X or Z offset of the T code on lathes). The
signal must come within the window epsilon (6254) around the measuring position, otherwise
PS0080; no active offset -> PS0081; an H word in the G37 block -> PS0082 (M series). Haas G37
names the offset with H and the expected position with Z, and updates that offset.

The step is buffer-stopping (TOOL_SETTER command); the feedback carries either the latched axis
positions (``ProbeResult``) or the measured tool length (``SetterResult.length_m``).
"""

from __future__ import annotations

from .alarms import ControllerAlarm
from .hooks import CycleCall, Step, register_probe


class ToolMeasure:
    def start(self, ex, call: CycleCall) -> list[Step]:
        s = ex.s
        lathe = ex.handler.T
        haas = call.dialect == "haas"
        axis = None
        for letter in ("Z", "X") if lathe else ("Z",):
            if letter in call.words:
                axis = letter
        if lathe and call.code == "G36":
            axis = "X" if "X" in call.words else None
        if axis is None:
            raise ControllerAlarm("g37_axis", f"{call.code} needs {'X' if call.code == 'G36' else 'Z'}")
        if "H" in call.words and not haas:
            raise ControllerAlarm("g37_with_h", call.code)
        if haas and "H" in call.words:
            offset = int(call.words["H"])
        else:
            offset = s.offset_number if lathe else s.h_number
        if offset <= 0:
            raise ControllerAlarm("g37_offset_number", f"{call.code} without an active offset")
        target = float(call.data[axis])
        current = call.current.get(axis, target)
        direction = 1.0 if current >= target else -1.0
        gamma = float(ex.p["g37_approach_m"])
        epsilon = float(ex.p["g37_tolerance_m"])
        approach = target + direction * gamma
        end = target - direction * epsilon
        data = {"axis": axis, "offset": offset, "target": target, "epsilon": epsilon, "lathe": lathe}
        probe = {"measure": "length" if axis == "Z" else "x", "tool": s.tool, "offset": offset,
                 "tolerance_m": epsilon, "expected_m": target}
        return [Step("rapid", {axis: approach}),
                Step("setter", {axis: end}, feed=float(ex.p["g37_feed_m_s"]), feed_mode="per_min",
                     provider=call.code if call.code in ("G36", "G37") else "G37", data=data, probe=probe)]

    def resume(self, ex, data: dict, feedback) -> list[Step]:
        s = ex.s
        axis = data["axis"]
        row = s.tool_offsets.setdefault(str(data["offset"]), [0] * 9)
        inc = ex.machine.lin_increment_m
        if feedback is not None and feedback.setter is not None and feedback.setter.length_m is not None and axis == "Z" \
                and not data["lathe"]:
            row[0] = int(round(feedback.setter.length_m / inc)) - row[1]
            return []
        if feedback is None or feedback.probe is None or not feedback.probe.triggered:
            raise ControllerAlarm("g37_signal", "no measuring signal")
        measured = ex.si(axis, s.skip.get(axis, 0))
        error = measured - float(data["target"])
        if abs(error) > float(data["epsilon"]):
            raise ControllerAlarm("g37_signal", f"signal {error * 1e3:.3f} mm from the measuring position")
        if data["lathe"]:
            column = 4 if axis == "X" else 5
            quantum = ex.by_letter[axis].quantum
            row[column] += int(round(error / quantum))
        else:
            row[0] += int(round(error / inc))
        return []


register_probe("G37", ToolMeasure(), replace=True)
register_probe("G36", ToolMeasure(), replace=True)
