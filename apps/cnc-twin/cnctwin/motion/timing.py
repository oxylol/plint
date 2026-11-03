"""Durations of non-motion commands (docs/models/motion.md "Auxiliary functions").

* **Tool change (mill ATC)**: spindle stop (the predicted deceleration from the current speed), then
  orientation (``orient_time_s``) while the tool-axis slide goes to its tool-change position (home,
  as a rapid), then the exchange ``tool_to_tool_s``; the spindle restarts with the next spindle
  command. Chip-to-chip time is the sum with the approach and the run-up of the next tool, and is
  compared with the machine's published ``chip_to_chip_s`` in the tests.
* **Turret index**: ``index_time_s.adjacent`` for one station, ``index_time_s.max`` for half the
  stations (the farthest a bidirectional turret indexes), linear in the number of steps between.
* **M functions**: completion time ``m_function_s``; through-spindle coolant adds the pressure delay;
  chuck/collet/tailstock/steady ``chuck_s``; doors ``doors.open_time_s``; pallets
  ``pallet_changer.change_time_s``.
* **Dwell**: seconds, or revolutions at the actual speed of the path's spindle.
* **Program stops** (operator model): M00, and M01 when the optional-stop switch is on, wait for
  ``stop_s + 2 door_s`` of the production operator model (project ``production.operator`` over
  db/models/production-operator.yaml), single-block stops ``single_block_s``, message stops
  ``message_stop_s``.
"""

from __future__ import annotations

import math

from ..db.models import params as model_params
from .machine import num, plain


def ticks_of(seconds: float, ipo_s: float) -> int:
    """Whole L1 ticks for a duration (a positive duration takes at least one tick)."""
    if seconds <= 0.0:
        return 0
    return max(1, int(math.ceil(seconds / ipo_s - 1e-9)))


class Timing:
    def __init__(self, model, project=None):
        self.model = model
        self.p = model.p
        op = model_params("production-operator")
        values = dict(op)
        operator = None
        if project is not None:
            production = getattr(project, "production", None)
            operator = getattr(production, "operator", None) if production is not None else None
        if operator is not None:
            for key in ("stop_s", "door_s"):
                v = num(getattr(operator, key, None))
                if v is not None:
                    values[key] = v
        self.stop_s = float(values.get("stop_s", 30.0))
        self.door_s = float(values.get("door_s", 4.0))

    # ------------------------------------------------------------------ tool change
    def tool_to_tool_s(self) -> float:
        tc = self.model.tool_changer
        value = num(getattr(tc, "tool_to_tool_s", None)) if tc is not None else None
        if value is None:
            return float(self.p["default_tool_change_s"])
        return value

    def chip_to_chip_s(self) -> float | None:
        tc = self.model.tool_changer
        return num(getattr(tc, "chip_to_chip_s", None)) if tc is not None else None

    def turret_index_s(self, from_station: int | None, to_station: int | None, turret=None) -> float:
        turrets = self.model.turrets
        t = turret if turret is not None else (turrets[0] if turrets else None)
        if t is None:
            return float(self.p["default_turret_index_s"])
        times = getattr(t, "index_time_s", None)
        adjacent = num(getattr(times, "adjacent", None)) if times is not None else None
        longest = num(getattr(times, "max", None)) if times is not None else None
        if adjacent is None:
            adjacent = float(self.p["default_turret_index_s"])
        if longest is None:
            longest = adjacent
        stations = int(num(getattr(t, "stations", None), 1) or 1)
        if from_station is None or to_station is None or stations <= 1:
            return adjacent
        steps = abs(int(to_station) - int(from_station)) % stations
        steps = min(steps, stations - steps)
        if steps == 0:
            return 0.0
        half = max(1, stations // 2)
        if half <= 1:
            return adjacent
        return adjacent + (longest - adjacent) * (steps - 1) / (half - 1)

    # ------------------------------------------------------------------ auxiliary functions
    def m_function_s(self) -> float:
        return float(self.p["m_function_s"])

    def coolant_s(self, pumps_on: dict, pump_kinds: dict) -> float:
        """M function time, plus the pressure delay when a through-spindle/tool pump is switched on."""
        delay = self.m_function_s()
        for pid, on in pumps_on.items():
            if on and pump_kinds.get(pid) in ("through_spindle", "through_tool", "high_pressure"):
                delay += float(self.p["tsc_pressure_delay_s"])
                break
        return delay

    def door_s_auto(self) -> float:
        doors = self.model.doors
        value = num(getattr(doors, "open_time_s", None)) if doors is not None else None
        return value if value is not None else float(self.p["default_door_s"])

    def pallet_s(self) -> float:
        pc = self.model.pallet_changer
        value = num(getattr(pc, "change_time_s", None)) if pc is not None else None
        return value if value is not None else float(self.p["default_pallet_change_s"])

    def chuck_s(self) -> float:
        return float(self.p["chuck_s"])

    # ------------------------------------------------------------------ operator
    def operator_stop_s(self, stop: str) -> float:
        if stop in ("M00", "M01", "M0", "M1"):
            return self.stop_s + 2.0 * self.door_s
        if stop == "single_block":
            return float(self.p["single_block_s"])
        return float(self.p["message_stop_s"])

    def dwell_s(self, cmd, omega_rad_s: float) -> float:
        seconds = float(cmd.dwell_s or 0.0)
        revs = float(cmd.dwell_rev or 0.0)
        if revs > 0.0:
            if abs(omega_rad_s) > 1e-9:
                seconds += revs * 2.0 * math.pi / abs(omega_rad_s)
        return seconds


def pump_kinds(model) -> dict:
    coolant = model.coolant
    out = {}
    if coolant is not None:
        for pump in getattr(coolant, "pumps", None) or []:
            out[str(plain(pump.id))] = str(plain(pump.kind))
    return out
