"""FANUC custom macro B variables: locals, commons and the system variables the executor models.

Values are in program units (mm or inch by G21/G20, degrees, diameter for an X axis programmed
in diameter), as the control reads and writes them. Live machine values (#5021-, #5041-, #5101-)
stop the buffer: reading them without fresh feedback for the current block raises NeedLive, the
executor emits a SYNC, and the statement is evaluated again with the encoder positions motion
reports (commanded minus following error). Variables the control has but this build does not
model raise ``not_modelled_variable`` (never a silent 0); numbers the control does not have
raise ``variable_out_of_range`` (PS0115).
"""

from __future__ import annotations

import datetime as _dt
import math

from ..alarms import ControllerAlarm
from ..expr import NeedLive, round_half_away
from .fanuc_tables import ARGUMENT_LETTER, LINEAR_ARGUMENTS, MODAL_LETTERS, ROTARY_ARGUMENTS

INCH_M = 0.0254


class Units:
    """Program-unit conversions of one executor (current G20/G21 mode)."""

    @staticmethod
    def linear_unit_m(ex) -> float:
        return INCH_M if ex.s.units == "inch" else 1e-3

    @staticmethod
    def to_program(ex, letter: str, counts: int) -> float:
        axis = ex.by_letter[letter]
        si = counts * axis.quantum
        if axis.kind == "rotary":
            return math.degrees(si)
        if axis.diameter:
            si *= 2.0
        return _clean(si / Units.linear_unit_m(ex), ex, axis)

    @staticmethod
    def from_program(ex, letter: str, value: float) -> int:
        axis = ex.by_letter[letter]
        if axis.kind == "rotary":
            return int(round_half_away(math.radians(value) / axis.quantum))
        si = value * Units.linear_unit_m(ex)
        if axis.diameter:
            si /= 2.0
        return int(round_half_away(si / axis.quantum))

    @staticmethod
    def length_to_program(ex, si: float) -> float:
        return si / Units.linear_unit_m(ex)

    @staticmethod
    def length_from_program(ex, value: float) -> float:
        return value * Units.linear_unit_m(ex)


def _clean(value: float, ex, axis) -> float:
    """Round a position read to 12 significant digits so 0.1 mm reads as 0.1 (not 0.09999999999999999)."""
    if value == 0.0:
        return 0.0
    return float(f"{value:.12g}")


class FanucVars:
    """Variable map of FANUC 0i-F M (variant "M") and T (variant "T")."""

    def __init__(self, variant: str):
        self.variant = variant
        self.lathe = variant == "T"

    # ------------------------------------------------------------------ dispatch
    def read(self, ex, n: int):
        s = ex.s
        if n == 0:
            return None
        if 1 <= n <= 33:
            return s.locals[ex.local_level()][n - 1]
        common = self._common(ex, n)
        if common is not None:
            table, index = common
            return table[index]
        value = self.read_system(ex, n)
        return value

    def write(self, ex, n: int, value) -> None:
        s = ex.s
        if n == 0:
            raise ControllerAlarm("write_protected", "#0")
        if 1 <= n <= 33:
            s.locals[ex.local_level()][n - 1] = value
            return
        common = self._common(ex, n)
        if common is not None:
            table, index = common
            table[index] = value
            return
        self.write_system(ex, n, value)

    def adp(self, ex, n: int, value):
        """ADP[#n]: the argument as if it had been passed with a decimal point."""
        if value is None:
            return 0.0
        letter = ARGUMENT_LETTER.get(n) if n <= 26 else "IJK"[(n - 4) % 3] if 4 <= n <= 33 else None
        if letter in LINEAR_ARGUMENTS:
            return value / ex.handler.linear_increment_units()
        if letter in ROTARY_ARGUMENTS:
            return value / ex.handler.rotary_increment_deg()
        return value

    def _common(self, ex, n: int):
        s = ex.s
        p = ex.p
        if 100 <= n <= 199:
            if n - 100 < p.get("shared_common_100", 0):
                return ex.root.shared["c100"], n - 100
            return s.variables["c100"], n - 100
        if 500 <= n <= 999:
            if n - 500 < p.get("shared_common_500", 0):
                return ex.root.shared["c500"], n - 500
            return s.variables["c500"], n - 500
        return None

    # ------------------------------------------------------------------ system variables (read)
    def read_system(self, ex, n: int):
        s = ex.s
        if 1000 <= n <= 1035 or 1100 <= n <= 1135:
            inputs = ex.p.get("interface_inputs", {})
            if n in inputs:
                return float(inputs[n])
            raise ControllerAlarm("interface_signal", f"#{n} (set physics.controller.interface_inputs)")
        if 2001 <= n <= 2999 or 10001 <= n <= 19999 and not (14001 <= n <= 19980):
            hit = self.tool_offset_var(ex, n)
            if hit is not None:
                key, column, kind = hit
                counts = s.tool_offsets.get(key, [0] * 9)[column]
                return float(counts) if kind == "int" else self._offset_to_program(ex, counts, kind)
        if n in (3000, 3006):
            raise ControllerAlarm("improper_command", f"#{n} can only be written")
        if n == 3001:
            return float(int((s.interp_tick * ex.tick_s) * 1000.0 + s.sys.get("3001_offset", 0.0)))
        if n == 3002:
            return s.interp_tick * ex.tick_s / 3600.0 + s.sys.get("3002_offset", 0.0)
        if n == 3003:
            return float(s.sys.get("3003", 0))
        if n == 3004:
            return float(s.sys.get("3004", 0))
        if n == 3007:
            return 0.0
        if n in (3011, 3012):
            start = _dt.datetime.fromisoformat(ex.p.get("clock_start", "2026-01-05T07:00:00"))
            now = start + _dt.timedelta(seconds=s.interp_tick * ex.tick_s)
            return float(now.strftime("%Y%m%d") if n == 3011 else now.strftime("%H%M%S"))
        if n == 3901:
            return float(s.sys.get("parts", 0))
        if n == 3902:
            return float(s.sys.get("parts_required", 0))
        if 4001 <= n <= 4030:
            return ex.handler.modal_group_value(n - 4000)
        if 4101 <= n <= 4130:
            return self.modal_letter(ex, n - 4100)
        if 4201 <= n <= 4330:
            raise ControllerAlarm("not_modelled_variable", f"#{n} (modal data of the executing block)")
        axes = ex.axes
        if 5001 <= n <= 5120:
            group, index = divmod(n - 5001, 20)
            if index >= len(axes):
                raise ControllerAlarm("variable_out_of_range", f"#{n}: no axis {index + 1}")
            letter = axes[index].letter
            if group == 0:
                return self._pos_read(ex, letter, s.pos.get(letter, 0))
            if group == 1:
                return self._machine_live(ex, letter)
            if group == 2:
                return self._work_live(ex, letter)
            if group == 3:
                return self._pos_read(ex, letter, s.skip.get(letter, s.pos.get(letter, 0)))
            if group == 4:
                return Units.to_program(ex, letter, s.applied.get(letter, 0))
            if group == 5:
                axis = axes[index]
                actual = ex.live(axis.name)
                deviation = ex.live_commanded(axis.name) - actual
                return Units.to_program(ex, letter, int(round_half_away(deviation / axis.quantum)))
        hit = self.work_offset_var(ex, n)
        if hit is not None:
            key, index = hit
            return Units.to_program(ex, axes[index].letter, s.offsets.get(key, [0] * len(axes))[index])
        return self.read_more(ex, n)

    def read_more(self, ex, n: int):
        raise ControllerAlarm("variable_out_of_range", f"#{n}")

    def _pos_read(self, ex, letter: str, counts: int) -> float:
        value = Units.to_program(ex, letter, counts)
        axis = ex.by_letter[letter]
        if axis.kind == "rotary" and axis.continuous:
            value = math.fmod(value, 360.0)
            if value < 0.0:
                value += 360.0
        return value

    def _machine_live(self, ex, letter: str) -> float:
        axis = ex.by_letter[letter]
        si = ex.live(axis.name)
        return Units.to_program(ex, letter, int(round_half_away(si / axis.quantum)))

    def _work_live(self, ex, letter: str) -> float:
        machine = {a.letter: int(round_half_away(ex.live(a.name) / a.quantum)) for a in ex.axes}
        program = ex.program_from_machine(machine)
        return self._pos_read(ex, letter, program.get(letter, 0))

    def modal_letter(self, ex, offset: int):
        s = ex.s
        letter = next((k for k, v in MODAL_LETTERS.items() if v == offset), None)
        if letter is None:
            raise ControllerAlarm("variable_out_of_range", f"#{4100 + offset}")
        if letter == "F":
            return s.feed_prog
        if letter == "S":
            return s.spindle.s_value
        if letter == "T":
            return float(s.tool if not self.lathe else s.tool * 100 + s.offset_number)
        if letter == "H":
            return float(s.h_number)
        if letter == "D":
            return float(s.d_number)
        if letter == "N":
            return float(s.last_n)
        if letter == "O":
            program = ex.store.programs[s.frames[-1].program]
            return float(program.number or 0)
        if letter == "P":
            return float(s.wcs[1:]) if s.wcs.startswith("P") else 0.0
        return s.last_words.get(letter)

    # ------------------------------------------------------------------ system variables (write)
    def write_system(self, ex, n: int, value) -> None:
        s = ex.s
        number = 0.0 if value is None else float(value)
        if n == 3000:
            ex.handler.macro_alarm(number)
            return
        if n == 3006:
            ex.handler.message_stop(number)
            return
        if n == 3001:
            s.sys["3001_offset"] = number - (s.interp_tick * ex.tick_s) * 1000.0
            return
        if n == 3002:
            s.sys["3002_offset"] = number - s.interp_tick * ex.tick_s / 3600.0
            return
        if n in (3003, 3004):
            s.sys[str(n)] = int(round_half_away(number))
            if n == 3003:
                s.sys["single_block_suppressed"] = bool(int(number) & 1)
            return
        if n == 3901:
            s.sys["parts"] = int(round_half_away(number))
            return
        if n == 3902:
            s.sys["parts_required"] = int(round_half_away(number))
            return
        if 1100 <= n <= 1135:
            s.outputs[str(n)] = bool(number)
            return
        if 2001 <= n <= 2999 or 10001 <= n <= 13999:
            hit = self.tool_offset_var(ex, n)
            if hit is not None:
                key, column, kind = hit
                row = s.tool_offsets.setdefault(key, [0] * 9)
                row[column] = int(round_half_away(number)) if kind == "int" else self._offset_from_program(ex, number, kind)
                return
        hit = self.work_offset_var(ex, n)
        if hit is not None:
            key, index = hit
            row = s.offsets.setdefault(key, [0] * len(ex.axes))
            row[index] = Units.from_program(ex, ex.axes[index].letter, number)
            ex.resync_program_position()
            return
        if 4001 <= n <= 5120 or n in (3007, 3011, 3012) or 1000 <= n <= 1035:
            raise ControllerAlarm("write_protected", f"#{n}")
        self.write_more(ex, n, value)

    def write_more(self, ex, n: int, value) -> None:
        raise ControllerAlarm("variable_out_of_range", f"#{n}")

    # ------------------------------------------------------------------ offset maps
    def work_offset_var(self, ex, n: int):
        """(offset key, axis index) of a work-offset variable."""
        n_axes = len(ex.axes)
        if 5201 <= n <= 5340:
            system, index = divmod(n - 5201, 20)
            if index >= n_axes:
                return None
            return ("EXT", "G54", "G55", "G56", "G57", "G58", "G59")[system], index
        if 7001 <= n <= 7960:
            system, index = divmod(n - 7001, 20)
            return (f"P{system + 1}", index) if index < n_axes else None
        if 14001 <= n <= 19980:
            system, index = divmod(n - 14001, 20)
            return (f"P{system + 1}", index) if index < n_axes else None
        return None

    def tool_offset_var(self, ex, n: int):
        """(tool key, column, kind) of a tool-offset variable; kind "len" | "x" | "int"."""
        if self.lathe:
            layout = ((2001, 0, "x"), (2101, 1, "len"), (2201, 2, "len"), (2301, 3, "int"),
                      (2701, 4, "x"), (2801, 5, "len"), (2901, 6, "len"), (2401, 7, "len"), (2451, 8, "len"))
            for base, column, kind in layout:
                width = 49 if base in (2401, 2451) else 64
                if base <= n < base + width:
                    return str(n - base + 1), column, kind
            return None
        if 2001 <= n <= 2800:
            block, index = divmod(n - 2001, 200)
            return str(index + 1), (1, 0, 3, 2)[block], "len"
        if 10001 <= n <= 13999:
            block, index = divmod(n - 10001, 1000)
            if index == 999:
                return None
            return str(index + 1), (1, 0, 3, 2)[block], "len"
        return None

    def _offset_to_program(self, ex, counts: int, kind: str) -> float:
        if kind == "x" and "X" in ex.by_letter:
            return Units.to_program(ex, "X", counts)
        return Units.length_to_program(ex, counts * ex.machine.lin_increment_m)

    def _offset_from_program(self, ex, value: float, kind: str) -> int:
        if kind == "x" and "X" in ex.by_letter:
            return Units.from_program(ex, "X", value)
        return int(round_half_away(Units.length_from_program(ex, value) / ex.machine.lin_increment_m))


def live_variable(n: int) -> bool:
    return 5021 <= n <= 5060 or 5101 <= n <= 5120


__all__ = ["FanucVars", "Units", "live_variable", "NeedLive"]
