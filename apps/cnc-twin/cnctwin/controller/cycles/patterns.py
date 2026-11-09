"""Haas mill bolt hole patterns G70 / G71 / G72 (haas-ngc-mill-om-2020 p. 348-349, Group 00).

Each is a one-shot block that runs the active canned cycle (G73, G74, G76, G77, G81-G89; "a canned cycle
must be active so that at each position, a drill or tap function is performed") at every position of the
pattern, in the active plane, about the tool's position when the block is read (the manual's examples
position the tool at the pattern centre and define the cycle with L0, so no hole is made there):

* G70 bolt hole circle: I radius, J starting angle (degrees counter-clockwise from the 3 o'clock
  direction, default 0), L holes evenly spaced on the circle: hole k at J + k 360 / L.
* G71 bolt hole arc: I radius, J starting angle, K angular spacing (+ or -), L holes: hole k at J + k K.
* G72 bolt holes along an angle: I distance between holes, J angle of the line, L holes in a straight line.
  The manual does not say where the first hole lies; the model parameter ``haas_g72_first_hole_at_start``
  (db/models/controller-cycles.yaml) chooses the tool's position (hole k at k I) or one spacing from it
  (hole k at (k + 1) I).

Angles are written in degrees (Haas reads addresses in whole units). Every hole is one execution of the
active cycle's provider with the cycle's modal data (R, Z, Q, P, I/J/K ...) and the G98/G99 return; the
tool goes from hole to hole the way the cycle positions it. A pattern block without an active canned
cycle stops with TW0037; I or L missing, L < 1, or axis words in the block stop with the cycle data alarm.
"""

from __future__ import annotations

import math

from .. import hooks
from ..alarms import ControllerAlarm
from ..hooks import CycleCall, Step
from ..pipeline import PLANES
from . import common


def _degrees(call, letter: str, default: float = 0.0) -> float:
    word = common.word(call, letter)
    return float(word.value) if word is not None else default


class BoltPattern:
    def __init__(self, code: str):
        self.code = code

    def positions(self, ex, call) -> list[tuple[float, float]]:
        """In-plane positions (program frame, SI) of the holes, in drilling order."""
        a, b, _ = PLANES[call.plane]
        for letter in ex.letters:
            if letter in call.words and ex.by_letter[letter].kind == "linear":
                raise ControllerAlarm("pattern_format", f"{self.code}: axis word {letter} (the pattern is about the "
                                                        "current position)")
        if common.word(call, "I") is None or common.word(call, "L") is None:
            raise ControllerAlarm("pattern_format", f"{self.code} needs I and L")
        if self.code == "G71" and common.word(call, "K") is None:
            raise ControllerAlarm("pattern_format", "G71 needs K (angular spacing)")
        count = int(round(float(common.word(call, "L").value)))
        if count < 1:
            raise ControllerAlarm("pattern_format", f"{self.code} L{count}")
        size = float(common.lin(ex, call, "I"))
        start = math.radians(_degrees(call, "J"))
        centre = (float(call.current.get(a, 0.0)), float(call.current.get(b, 0.0)))
        out = []
        for k in range(count):
            if self.code == "G70":
                angle = start + 2.0 * math.pi * k / count
                out.append((centre[0] + size * math.cos(angle), centre[1] + size * math.sin(angle)))
            elif self.code == "G71":
                angle = start + math.radians(_degrees(call, "K")) * k
                out.append((centre[0] + size * math.cos(angle), centre[1] + size * math.sin(angle)))
            else:
                first = bool(common.params(ex)["haas_g72_first_hole_at_start"])
                distance = size * (k if first else k + 1)
                out.append((centre[0] + distance * math.cos(start), centre[1] + distance * math.sin(start)))
        return out

    def expand(self, ex, call) -> list[Step]:
        s = ex.s
        cycle = s.cycle
        if cycle is None or not cycle.get("code"):
            raise ControllerAlarm("pattern_without_cycle", f"{self.code}: no canned cycle is active")
        code = str(cycle["code"])
        provider = hooks.find(hooks.CYCLES, ex.handler.d.name, code)
        if provider is None:
            raise ControllerAlarm("cycle_unavailable", code)
        a, b, _ = PLANES[call.plane]
        drill = ex.handler.drilling_axis(code)
        data = dict(cycle.get("data", {}))
        current = dict(call.current)
        steps: list[Step] = []
        for k, (pa, pb) in enumerate(self.positions(ex, call)):
            position = dict(current)
            position[a], position[b] = pa, pb
            hole = CycleCall(code=code, dialect=call.dialect, plane=call.plane, absolute=call.absolute,
                             position=position, current=dict(current), initial_level=s.cycle_initial,
                             return_to_initial=call.return_to_initial, data=dict(data), words={}, repeat_index=k,
                             feed_mode=call.feed_mode, units=call.units,
                             extra={"word_values": {}, "drilling_axis": drill, "pattern": self.code})
            hole_steps = provider.expand(ex, hole)
            steps.extend(hole_steps)
            for step in hole_steps:
                if step.kind in ("rapid", "linear", "arc") and not step.machine:
                    current.update(step.target)
        return steps


__all__ = ["BoltPattern"]
