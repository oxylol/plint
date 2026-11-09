"""Haas NGC (mill and lathe): FANUC-compatible word address with Haas codes, alarms and variables.

Differences from FANUC handled here (docs/dialects.md "Haas NGC"): one M code per block, every
M code acts at the end of the block; two G codes of one group or the same address twice in a
block alarm; F is as written (no least-increment scaling); G04 P with a decimal point is seconds,
without one milliseconds; G68 centre in A/B; G100/G101 mirror; G103 look-ahead limit; G110-G129 and
G154 P1-P99 work offsets; G187 smoothing; G254/G255 dynamic work offsets and G234 tool centre
point control on table-type five-axis machines (UMC, EC-400 with a B table); G268/G269 feature
coordinate systems; M97 local subprograms; M88/M89 through-spindle coolant, M83/M84 air,
M34/M35 programmable nozzle, M31/M33 conveyor, M50 pallet change, M80/M81 doors; #3000 alarms
are 1000 + n; NGC variable numbers (#10000+) with the legacy aliases.
"""

from __future__ import annotations

import math

import numpy as np

from ... import units
from .. import hooks
from ..alarms import AlarmTable, ControllerAlarm
from ..expr import round_half_away
from ..ir import Command, Kind, Space
from ..lexer import LexerRules
from ..pipeline import euler_matrix
from .base import Dialect, GSpec, register_dialect, word_set
from .fanuc import BlockWords, FanucHandler, WordValue, fanuc_params
from .fanuc_tables import LETTERS, M_GCODES, M_POWER_ON, MAX_DIGITS, T_GCODES, T_POWER_ON
from .fanuc_vars import FanucVars, Units

HAAS_EXTRA_M = {
    "G12": GSpec(0, "cycle", "G12", cycle=True), "G13": GSpec(0, "cycle", "G13", cycle=True),
    "G47": GSpec(0, "cycle", "G47", cycle=True), "G150": GSpec(0, "cycle", "G150", cycle=True),
    "G70": GSpec(0, "cycle", "G70", cycle=True), "G71": GSpec(0, "cycle", "G71", cycle=True),
    "G72": GSpec(0, "cycle", "G72", cycle=True), "G77": GSpec(9, "cycle", "G77", cycle=True),
    "G153": GSpec(9, "cycle", "G153", cycle=True), "G156": GSpec(9, "cycle", "G156", cycle=True),
    "G174": GSpec(0, "cycle", "G174", cycle=True), "G184": GSpec(0, "cycle", "G184", cycle=True),
    "G35": GSpec(0, "probe", "haas:G35", axis_words=True), "G36": GSpec(0, "probe", "haas:G36", axis_words=True),
    "G37": GSpec(0, "probe", "G37", axis_words=True), "G136": GSpec(0, "probe", "haas:G136", axis_words=True),
    "G100": GSpec(22, "mirror", "off"), "G101": GSpec(22, "mirror", "on"),
    "G103": GSpec(0, "lookahead"), "G107": GSpec(0, "tcp", "haas:G107"),
    "G141": GSpec(7, "comp3d", "G141"), "G143": GSpec(8, "tcp", "haas:G143"),
    "G154": GSpec(14, "wcs", "P"), "G187": GSpec(0, "smoothing"),
    "G234": GSpec(8, "tcp", "haas:G234"), "G253": GSpec(0, "tcp", "haas:G253"),
    "G254": GSpec(23, "dwo", "on"), "G255": GSpec(23, "dwo", "off"),
    "G268": GSpec(31, "feature", "on"), "G269": GSpec(31, "feature", "off"),
    "G266": GSpec(0, "unsupported"), "G167": GSpec(0, "unsupported"),
}
for _n in range(110, 130):
    HAAS_EXTRA_M[f"G{_n}"] = GSpec(14, "wcs", f"P{_n - 109}")

HAAS_EXTRA_T = {
    "G14": GSpec(0, "unsupported"), "G15": GSpec(0, "unsupported"), "G61": GSpec(15, "exact", "exact_stop"),
    "G64": GSpec(15, "exact", "blend"), "G95": GSpec(0, "cycle", "G95", cycle=True),
    "G100": GSpec(22, "mirror", "off"), "G101": GSpec(22, "mirror", "on"), "G103": GSpec(0, "lookahead"),
    "G105": GSpec(0, "unsupported"), "G107": GSpec(0, "tcp", "haas:G107"),
    "G112": GSpec(21, "tcp", "haas:G112"), "G113": GSpec(21, "tcp", "off"),
    "G154": GSpec(14, "wcs", "P"), "G156": GSpec(0, "cycle", "G156", cycle=True),
    "G184": GSpec(10, "cycle", "G184", cycle=True), "G186": GSpec(0, "cycle", "G186", cycle=True),
    # face drilling cycles of the Haas lathe besides FANUC T's G83-G89 (haas-ngc-lathe-om-2020 G81-G89)
    "G81": GSpec(10, "cycle", "G81", cycle=True), "G82": GSpec(10, "cycle", "G82", cycle=True),
    "G86": GSpec(10, "cycle", "G86", cycle=True),
    "G187": GSpec(0, "smoothing"), "G195": GSpec(0, "cycle", "G195", cycle=True),
    "G196": GSpec(0, "cycle", "G196", cycle=True), "G198": GSpec(0, "unsupported"),
    "G199": GSpec(0, "unsupported"), "G200": GSpec(0, "unsupported"),
    "G211": GSpec(0, "probe", "haas:G211", axis_words=True), "G212": GSpec(0, "probe", "haas:G212", axis_words=True),
    "G241": GSpec(10, "cycle", "G241", cycle=True), "G242": GSpec(10, "cycle", "G242", cycle=True),
    "G243": GSpec(10, "cycle", "G243", cycle=True), "G245": GSpec(10, "cycle", "G245", cycle=True),
    "G246": GSpec(10, "cycle", "G246", cycle=True), "G249": GSpec(10, "cycle", "G249", cycle=True),
    "G266": GSpec(0, "unsupported"), "G167": GSpec(0, "unsupported"),
}
for _n in list(range(110, 112)) + list(range(114, 130)):
    HAAS_EXTRA_T[f"G{_n}"] = GSpec(14, "wcs", f"P{_n - 109}")

ALARM_CODES = {
    "too_many_digits": "311", "no_data_after_address": "311", "illegal_minus": "311",
    "illegal_decimal_point": "311", "improper_address": "311", "improper_g_code": "310",
    "feed_zero": "320", "radius_tolerance": "304", "radius_tolerance_r": "302", "illegal_plane": "303",
    "arc_no_radius": "338", "illegal_radius": "302", "illegal_offset_value": "308",
    "illegal_offset_number": "308", "g10_illegal_p": "315", "g10_illegal_value": "331", "g31_in_comp": "376",
    "plane_change_in_comp": "344", "ref_return_in_cycle": "311", "cycle_q_missing": "305",
    "illegal_reference_return": "311", "program_not_found": "314", "missing_p": "313", "invalid_p": "315",
    "nesting": "307", "sequence_number_not_found": "337", "sequence_out_of_range": "337",
    "ref_check_error": "311", "overflow_integer": "331", "overflow_float": "331", "zero_divide": "331",
    "improper_command": "331", "expression_format": "311", "variable_out_of_range": "331",
    "write_protected": "331", "bracket_nesting": "311", "argument_out_of_range": "331",
    "modal_call_nesting": "307", "missing_end": "311", "macro_format": "311", "illegal_loop_number": "331",
    "nc_macro_same_block": "311", "g_as_argument": "311", "illegal_scale": "331", "end_of_record": "312",
    "sub_without_m99": "312", "dwell_range": "324", "dwell_missing_p": "326", "multiple_codes": "339",
    "undefined_m_code": "329", "skip_found": "378", "skip_not_found": "379", "pocket_definition": "370",
    "arc_under_scaling": "467", "wait_mismatch": "339", "wait_illegal_p": "315", "g37_signal": "379",
    "g37_offset_number": "308", "g37_with_h": "311", "g37_axis": "311", "tool_too_big": "369",
}
# Built-in texts (Haas classic/NGC alarm list as generally documented); db/controllers/haas-ngc.yaml wins.
ALARM_TEXTS = {
    "302": "Invalid R In G02 Or G03", "303": "Invalid X, Y, Or Z In G02 Or G03",
    "304": "Invalid I, J, Or K In G02 Or G03", "305": "Invalid Q In Canned Cycle",
    "307": "Subroutine Nesting Too Deep", "308": "Invalid Tool Offset", "310": "Invalid G Code",
    "311": "Unknown Code", "312": "Program End", "313": "No P Code In M97, M98, Or G65",
    "314": "Subprogram Not In Memory", "315": "Invalid P Code In M97, M98 Or M99", "320": "No Feed Rate Specified",
    "324": "Delay Time Range Error", "326": "G04 Without P Code", "329": "Undefined M Code",
    "331": "Range Error", "337": "GOTO Or P Line Not Found", "338": "Invalid IJK And XYZ In G02 Or G03",
    "339": "Multiple Codes", "344": "Cutter Comp With G18 Or G19", "370": "Pocket Definition Error",
    "376": "No Cutter Comp In Skip", "378": "Skip Signal Found", "379": "Skip Signal Not Found",
    "467": "Arc Not Allowed With Scaling Settings", "369": "Tool Too Big",
}

# Haas lathe cycles whose D word is cycle data (HaasHandler.cycle_data_letters)
HAAS_LATHE_D_CYCLES = ("G71", "G72", "G73", "G74", "G75", "G76")
# Haas lathe radial drilling cycles (drill along X); the face cycles drill along Z
HAAS_RADIAL = ("G241", "G242", "G243", "G245", "G246", "G249")

HAAS_M_MILL = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 19, 30, 31, 33, 34, 35, 36, 41, 42, 50, 51, 52, 53,
               54, 55, 59, 61, 62, 63, 64, 65, 69, 70, 71, 73, 74, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 97, 98, 99,
               104, 105, 109, 130, 131, 138, 139, 158, 159, 199, 21, 22, 23, 24, 25, 29, 39, 46, 48, 75, 90, 91, 95,
               96, 160, 161, 162, 163}
HAAS_M_LATHE = {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 30, 31, 33, 35, 36, 37,
                38, 39, 41, 42, 51, 52, 53, 54, 55, 56, 59, 61, 62, 63, 64, 65, 66, 69, 78, 79, 85, 86, 88, 89, 90, 91,
                95, 96, 97, 98, 99, 104, 105, 109, 110, 111, 119, 121, 122, 123, 124, 125, 126, 129, 130, 131, 133,
                134, 135, 143, 144, 145, 146, 147, 219}


class HaasVars(FanucVars):
    """Haas NGC variable numbers (db/controllers/haas-ngc.yaml 'variables'; legacy #100-#999 alias #10100-)."""

    def _common(self, ex, n: int):
        s = ex.s
        if 10000 <= n <= 10999:
            return s.variables["c10000"], n - 10000
        if 100 <= n <= 999:
            if 700 <= n <= 749:
                raise ControllerAlarm("write_protected", f"#{n} is reserved for the control")
            return s.variables["c10000"], n
        return None

    def tool_offset_var(self, ex, n: int):
        if self.lathe:
            layout = ((2001, 4, "x"), (2051, 8, "len"), (2101, 5, "len"), (2201, 6, "len"), (2301, 3, "int"),
                      (2701, 0, "x"), (2751, 7, "len"), (2801, 1, "len"), (2901, 2, "len"))
            for base, column, kind in layout:
                if base <= n < base + 50:
                    return str(n - base + 1), column, kind
            return None
        if 2001 <= n <= 2800:
            block, index = divmod(n - 2001, 200)
            return str(index + 1), block, "dia" if block >= 2 else "len"
        return None

    def _offset_to_program(self, ex, counts: int, kind: str) -> float:
        if kind == "dia":
            factor = 2.0 if ex.p.get("offsets_diameter") else 1.0
            return Units.length_to_program(ex, counts * ex.machine.lin_increment_m * factor)
        return super()._offset_to_program(ex, counts, kind)

    def _offset_from_program(self, ex, value: float, kind: str) -> int:
        if kind == "dia":
            factor = 2.0 if ex.p.get("offsets_diameter") else 1.0
            return int(round_half_away(Units.length_from_program(ex, value) / factor / ex.machine.lin_increment_m))
        return super()._offset_from_program(ex, value, kind)

    def work_offset_var(self, ex, n: int):
        n_axes = len(ex.axes)
        if 5201 <= n <= 5206:
            index = n - 5201
            return ("G52", index) if index < n_axes else None
        if 5221 <= n <= 5340:
            system, index = divmod(n - 5221, 20)
            return (("G54", "G55", "G56", "G57", "G58", "G59")[system], index) if index < n_axes else None
        if 7001 <= n <= 7386:
            system, index = divmod(n - 7001, 20)
            return (f"P{system + 1}", index) if index < n_axes else None
        if 14001 <= n <= 15966:
            system, index = divmod(n - 14001, 20)
            return (f"P{system + 1}", index) if index < n_axes else None
        return None

    def read_system(self, ex, n: int):
        s = ex.s
        if 1000 <= n <= 1063 or 11000 <= n <= 11063:
            inputs = ex.p.get("interface_inputs", {})
            if n in inputs:
                return float(inputs[n])
            raise ControllerAlarm("interface_signal", f"#{n} (set physics.controller.interface_inputs)")
        if 5101 <= n <= 5120 or 4201 <= n <= 4330 or 1035 < n < 1100:
            raise ControllerAlarm("variable_out_of_range", f"#{n}")
        if 4101 <= n <= 4126:
            return self.modal_letter(ex, n - 4100)
        hit = self.work_offset_var(ex, n)
        if hit is not None and hit[0] == "G52":
            return Units.to_program(ex, ex.axes[hit[1]].letter, s.g52.get(ex.axes[hit[1]].letter, 0))
        if n == 3020 or n == 3021:
            return s.interp_tick * ex.tick_s
        if n == 3023:
            return s.interp_tick * ex.tick_s - float(s.sys.get("part_start_s", 0.0))
        if n == 3024:
            return float(s.sys.get("last_part_s", 0.0))
        if n == 3025:
            return float(s.sys.get("previous_part_s", 0.0))
        if n == 3026:
            return float(s.tool)
        if n == 3027:
            return float(abs(s.spindle.s_value) if s.spindle.direction != 0 else 0.0)
        if n == 3030:
            return 1.0 if s.operator.single_block else 0.0
        if n == 3032:
            return 1.0 if s.operator.block_delete[0] else 0.0
        if n == 3033:
            return 1.0 if s.operator.optional_stop else 0.0
        if n == 3901 or n == 3902:
            return float(s.sys.get("m30_count_1" if n == 3901 else "m30_count_2", 0))
        if 3401 <= n <= 3600:
            return float(s.sys.get("pcool", {}).get(str(n - 3400), 0))
        if n == 6198:
            return 1000000.0
        return super().read_system(ex, n)

    def modal_letter(self, ex, offset: int):
        s = ex.s
        letter = chr(ord("A") + offset - 1)
        if letter == "F":
            return s.feed_prog
        if letter == "S":
            return s.spindle.s_value
        if letter == "T":
            return float(s.tool)
        if letter == "H":
            return float(s.h_number)
        if letter == "D":
            return float(s.d_number)
        return s.last_words.get(letter)

    def read_more(self, ex, n: int):
        modelled_elsewhere = (1064 <= n <= 1098 or 1601 <= n <= 2000 or 3201 <= n <= 3400 or 5401 <= n <= 6000
                              or 8500 <= n <= 8564 or 20000 <= n <= 39999 or n in (3022, 3028, 3034))
        if modelled_elsewhere:
            raise ControllerAlarm("not_modelled_variable", f"#{n}")
        raise ControllerAlarm("variable_out_of_range", f"#{n}")

    def write_system(self, ex, n: int, value) -> None:
        s = ex.s
        number = 0.0 if value is None else float(value)
        hit = self.work_offset_var(ex, n)
        if hit is not None and hit[0] == "G52":
            letter = ex.axes[hit[1]].letter
            s.g52[letter] = Units.from_program(ex, letter, number)
            ex.resync_program_position()
            return
        if n == 3030:
            s.operator.single_block = bool(number)
            return
        if n in (3901, 3902):
            s.sys["m30_count_1" if n == 3901 else "m30_count_2"] = int(round_half_away(number))
            return
        if 3401 <= n <= 3600:
            s.sys.setdefault("pcool", {})[str(n - 3400)] = int(round_half_away(number))
            return
        if 11000 <= n <= 11063 or 1000 <= n <= 1063 or n in (3020, 3021, 3023, 3024, 3025, 3026, 3027, 6198):
            raise ControllerAlarm("write_protected", f"#{n}")
        super().write_system(ex, n, value)

    def write_more(self, ex, n: int, value) -> None:
        self.read_more(ex, n)


class HaasHandler(FanucHandler):
    m_after_motion_only = True
    gate_m_by_database = True
    repeat_letters = ("L",)             # Haas canned cycles repeat with L; K is the G73/G83 minimum peck

    def init_state(self) -> None:
        super().init_state()
        s = self.s
        s.variables = {"c10000": [None] * 1000}
        if not self.T:
            s.modal["23"] = "G255"
            s.modal["31"] = "G269"
            self.default_smoothing()
        s.sys["pcool"] = {str(k): int(v) for k, v in self.ex.ctx.nozzle_positions.items()}

    def on_reset(self) -> None:
        super().on_reset()
        s = self.s
        if not self.T:
            s.modal["23"] = "G255"
            s.modal["31"] = "G269"
            if s.tcp in ("haas:G254", "haas:G234"):
                s.tcp = ""
            self.default_smoothing()          # G187 is cancelled by RESET, M30, M02 (p. 393)
        s.lookahead_limit = 0

    # numbers ------------------------------------------------------------------
    def feed_no_point_scale(self, mode: str, inch: bool) -> float:
        return 1.0

    def spindle_number(self, word: WordValue) -> float:
        if word.value < 0:
            raise ControllerAlarm("illegal_minus", f"S{word.text}")
        return float(round_half_away(word.value))

    def g04(self, bw: BlockWords) -> None:
        word = bw.take("P")
        if word is None:
            raise ControllerAlarm("dwell_missing_p", "G04 without P")
        seconds = word.value if (word.point or word.from_variable) else word.value / 1000.0
        if seconds < 0.0:
            raise ControllerAlarm("dwell_range", f"P{word.text}")
        self.ex.dwell(seconds)

    def rotation_centre_letters(self, a: str, b: str):
        return ((a, "A"), (b, "B"))

    def max_offset(self) -> int:
        return 50 if self.T else 200

    def m_table(self) -> set[int]:
        return set(HAAS_M_LATHE if self.T else HAAS_M_MILL)

    def drilling_axis(self, code: str) -> str:
        """Haas lathes: the face cycles (G81-G89, G184) drill along Z, the radial cycles G241-G249 along X."""
        if self.T:
            return "X" if code in HAAS_RADIAL else "Z"
        return super().drilling_axis(code)

    def cycle_data_letters(self, oneshots: list) -> set[str]:
        """Haas lathes: D in a G71/G72 block is the depth of cut, in G73 the number of passes, in G74/G75
        the tool clearance, in G76 the first pass depth (haas-ngc-lathe-om-2020, G71-G76); a lathe takes
        its offsets from the T word, so D is never an offset number there."""
        if self.T and any(code in HAAS_LATHE_D_CYCLES for code, _ in oneshots):
            return {"D"}
        return set()

    def m_data_letters(self, m: int) -> set[str]:
        if m in (97,):
            return {"P", "L"}
        if m in (133, 134, 143, 144):
            return {"P"}
        if m in (46, 96):
            return {"P", "Q"}
        return set()

    def count_part(self, m: int) -> None:
        s = self.s
        if m == 30:
            s.sys["m30_count_1"] = int(s.sys.get("m30_count_1", 0)) + 1
            s.sys["m30_count_2"] = int(s.sys.get("m30_count_2", 0)) + 1
            now = s.interp_tick * self.ex.tick_s
            s.sys["previous_part_s"] = float(s.sys.get("last_part_s", 0.0))
            s.sys["last_part_s"] = now - float(s.sys.get("part_start_s", 0.0))
            s.sys["part_start_s"] = now

    # modal --------------------------------------------------------------------
    def modal_extra(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        if spec.kind == "dwo":
            s.modal["23"] = code
            if spec.arg == "on":
                if not self.machine.table_kinematics_supported():
                    raise ControllerAlarm("transform_unavailable", "G254 needs rotary table axes")
                s.tcp = "haas:G254"
                s.sys["tcp_group"] = 23
            elif s.tcp == "haas:G254":
                s.tcp = ""
                s.sys.pop("tcp_group", None)
            return
        if spec.kind == "feature":
            s.modal["31"] = code
            if spec.arg == "off":
                s.feature = None
            else:
                origin = []
                for letter in "XYZ":
                    axis = ex.by_letter.get(letter)
                    word = bw.take(letter)
                    origin.append(self.axis_counts(axis, word) if (word is not None and axis is not None) else 0)
                angles = []
                for letter in "IJK":
                    word = bw.take(letter)
                    angles.append(math.degrees(self.rot(word)) if word is not None else 0.0)
                q = bw.take("Q")
                order = str(self.integer(q)) if q is not None else "321"
                if sorted(order) != ["1", "2", "3"]:
                    raise ControllerAlarm("improper_address", f"G268 Q{order}")
                matrix = euler_matrix(tuple(angles), order)
                s.feature = {"origin": origin, "matrix": [v for row in matrix for v in row]}
            ex.invalidate_transform()
            ex.resync_program_position()
            return
        if spec.kind == "comp3d":
            ex.start_comp("3d")
            s.modal["7"] = code
            return
        raise ControllerAlarm("unsupported_function", code)

    def oneshot_extra(self, code: str, spec: GSpec, bw: BlockWords) -> bool:
        ex, s = self.ex, self.s
        if spec.kind == "lookahead":
            word = bw.take("P")
            limit = self.integer(word) if word is not None else 0
            if not 0 <= limit <= 15:
                raise ControllerAlarm("invalid_p", f"G103 P{limit}")
            s.lookahead_limit = limit
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"lookahead_blocks": limit}))
            return False
        if spec.kind == "smoothing":
            self.g187(bw)
            return False
        raise ControllerAlarm("unsupported_function", code)

    def g187(self, bw: BlockWords) -> None:
        """G187 Pn Ennnn (haas-ngc-mill-om-2020 p. 393, Settings 85 and 191 p. 456, 463): P sets the smoothness
        level and leaves an active E alone, E sets the maximum corner rounding and leaves P alone, G187 alone
        cancels E and returns to the defaults: Setting 191 (medium) and the corner rounding of Setting 85."""
        ex, s = self.ex, self.s
        p = bw.take("P")
        e = bw.take("E")
        if p is None and e is None:
            self.default_smoothing()
        if p is not None:
            level = self.integer(p)
            if level not in (1, 2, 3):
                raise ControllerAlarm("invalid_p", f"G187 P{level}")
            s.smoothing["level"] = level
            s.smoothing["mode_code"] = f"G187 P{level}"
            if e is None and not s.smoothing.get("e_active"):
                s.smoothing["tolerance_m"] = self.level_rounding_m(level)
        if e is not None:
            s.smoothing["tolerance_m"] = self.lin(e)
            s.smoothing["e_active"] = True
            if p is None:
                s.smoothing["mode_code"] = (s.smoothing.get("mode_code") or "G187") + " E"
        ex.emit(Command(Kind.MODE, ex.line_ref(), smoothing=ex.smoothing_spec(),
                        modal={"smoothing_level": s.smoothing.get("level", 2)}))

    def level_rounding_m(self, level: int) -> float:
        """Maximum corner rounding without an E word at smoothness level P<level>: the machine file's smoothing
        entry for "G187 P<level>" (Setting 85 of that machine: the P level sets the smoothness, not the corner
        tolerance), else Setting 85 as the machine file gives it for the default level."""
        table = {entry["code"]: float(entry["tolerance_m"]) for entry in self.machine.smoothing}
        value = table.get(f"G187 P{level}")
        return float(value) if value is not None else self.setting85_m()

    def setting85_m(self) -> float:
        """Setting 85 Maximum Corner Rounding as the machine file gives it: the smoothing entry of the default
        smoothness level (Setting 191, factory default MEDIUM = "G187 P2"), else the first G187 entry; without a
        smoothing table on a mill the controller database's Setting 85 factory default (0.0250 in, mill manual
        p. 456); 0 when neither gives one (the motion planner's default contour tolerance then applies)."""
        table = [(entry["code"], float(entry["tolerance_m"])) for entry in self.machine.smoothing]
        for code, value in table:
            if code == "G187 P2":
                return value
        for code, value in table:
            if code.startswith("G187"):
                return value
        if self.T:
            return 0.0
        default = ((self.d.db or {}).get("timing") or {}).get("corner_rounding_default_in")
        value = default.get("value") if isinstance(default, dict) else default
        return float(value) * units.INCH_M if value is not None else 0.0

    def default_smoothing(self) -> None:
        """Power-on, RESET, M30 and G187 alone: Setting 191 (medium) and the Setting 85 corner rounding, no E."""
        s = self.s
        s.smoothing.update({"level": 2, "mode_code": "", "tolerance_m": self.setting85_m(), "e_active": False})

    def select_wcs(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        if code == "G154":
            word = bw.take("P")
            number = self.integer(word) if word is not None else 1
            if not 1 <= number <= 99:
                raise ControllerAlarm("invalid_p", f"G154 P{number}")
            spec = GSpec(14, "wcs", f"P{number}")
        s = self.s
        key = spec.arg
        s.modal["14"] = code
        if key != s.wcs:
            s.wcs = key
            s.offsets.setdefault(key, [0] * len(self.ex.axes))
            self.ex.resync_program_position()

    def g10_work_key(self, l: int, p: int | None) -> str:
        if l == 2:
            if p is None or not 0 <= p <= 6:
                raise ControllerAlarm("g10_illegal_p", f"G10 L2 P{p}")
            return ("G52", "G54", "G55", "G56", "G57", "G58", "G59")[p]
        if p is None or not 1 <= p <= 99:
            raise ControllerAlarm("g10_illegal_p", f"G10 L20 P{p}")
        return f"P{p}"

    def g10(self, bw: BlockWords) -> None:
        l_word = bw.peek("L")
        p_word = bw.peek("P")
        if l_word is not None and p_word is not None and self.integer(l_word) == 2 and self.integer(p_word) == 0:
            ex, s = self.ex, self.s
            bw.take("L")
            bw.take("P")
            for axis in ex.axes:
                word = bw.take(axis.letter)
                if word is not None:
                    value = self.axis_counts(axis, word)
                    s.g52[axis.letter] = (s.g52.get(axis.letter, 0) + value) if not self.absolute() else value
            ex.resync_program_position()
            return
        if not self.T and l_word is not None and self.integer(l_word) in (12, 13) and self.p.get("offsets_diameter"):
            r = bw.peek("R")
            if r is not None:
                bw.addr["R"] = WordValue("R", r.value / 2.0, True, r.text, r.from_variable)
        super().g10(bw)

    def arc(self, bw: BlockWords, target: dict[str, int], direction: int, corner: dict | None = None) -> None:
        if self.s.scale is not None and not self.ex.transform().scales_uniformly(self.plane()):
            raise ControllerAlarm("arc_under_scaling", "G02/G03 with per-axis scaling")
        if corner:
            raise ControllerAlarm("chamfer_not_g01", ",C/,R on a G02/G03 block (Haas: between two G01 blocks)")
        super().arc(bw, target, direction, corner)

    # M codes ------------------------------------------------------------------
    def m_extra(self, m: int, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        machine = self.machine
        ref = ex.line_ref()
        pumps = machine.pumps
        if m == 16:
            self.tool_change(bw)
        elif m == 97:
            word = bw.take("P")
            if word is None:
                raise ControllerAlarm("missing_p", "M97 without P")
            n = self.integer(word)
            repeat_word = bw.take("L")
            repeat = self.integer(repeat_word) if repeat_word is not None else 1
            frame = s.frames[-1]
            program = ex.store.programs[frame.program]
            target = program.find_n(n, 0)
            if target is None:
                raise ControllerAlarm("sequence_number_not_found", f"M97 P{n}")
            ex.request_call("local", frame.program, None, repeat, entry_pc=target)
        elif m in (88, 89):
            tsc = [pid for pid, kind in pumps.items() if kind == "through_spindle"]
            if not tsc:
                raise ControllerAlarm("machine_feature_missing", f"M{m}: no through-spindle coolant")
            self.coolant({tsc[0]: m == 88} if m == 88 else {pid: False for pid in tsc}, f"M{m}")
        elif m == 7 and not self.T:
            shower = [pid for pid, kind in pumps.items() if kind in ("shower", "mist")]
            if not shower:
                raise ControllerAlarm("machine_feature_missing", "M07: no shower coolant")
            self.coolant({pid: True for pid in shower}, "M07")
        elif m == 9:
            self.coolant({pid: False for pid, kind in pumps.items() if kind not in ("through_spindle",)}, "M09")
        elif (m in (83, 84) and not self.T) or (m in (12, 13) and self.T):
            if not machine.air_blast:
                raise ControllerAlarm("machine_feature_missing", f"M{m}: no air blast")
            self.coolant({}, f"M{m}", air=m in (83, 12))
        elif m in (34, 35):
            nozzle = next((nid for nid, spec in machine.nozzles.items() if "M34" in spec["code"]), None)
            if nozzle is None:
                raise ControllerAlarm("machine_feature_missing", f"M{m}: no programmable coolant nozzle")
            positions = machine.nozzles[nozzle]["positions"] or 1
            current = s.nozzles.get(nozzle, 1)
            new = min(positions, current + 1) if m == 34 else max(1, current - 1)
            self.coolant({}, f"M{m}", nozzles={nozzle: new})
        elif m in (31, 33):
            if not machine.conveyor:
                raise ControllerAlarm("machine_feature_missing", f"M{m}: no chip conveyor")
            ex.emit(Command(Kind.CONVEYOR, ref, modal={"device": {"kind": "conveyor", "code": f"M{m}",
                                                                  "action": "forward" if m == 31 else "stop"}}))
        elif m == 50 or (m == 199 and machine.pallets):
            if not machine.pallets:
                raise ControllerAlarm("machine_feature_missing", "M50: no pallet changer")
            ex.emit(Command(Kind.PALLET, ref, modal={"device": {"kind": "pallet", "code": f"M{m}", "action": "change"}}))
        elif m == 199:
            self.count_part(30)
            ex.end_program("M30")
        elif m == 36:
            ex.emit(Command(Kind.PALLET, ref, modal={"device": {"kind": "pallet", "code": "M36", "action": "part_ready"}}))
        elif (m in (80, 81) and not self.T) or (m in (85, 86) and self.T):
            if not machine.door_auto:
                raise ControllerAlarm("machine_feature_missing", f"M{m}: no automatic door")
            ex.emit(Command(Kind.DOOR, ref, modal={"device": {"kind": "door", "code": f"M{m}",
                                                              "action": "open" if m in (80, 85) else "close"}}))
        elif m in (78, 79):
            if m == 78 and s.skip_triggered:
                raise ControllerAlarm("skip_found", "M78 after a G31 that triggered")
            if m == 79 and not s.skip_triggered:
                raise ControllerAlarm("skip_not_found", "M79 after a G31 that did not trigger")
        elif m in (10, 11, 12, 13) and not self.T:
            rotaries = [a for a in ex.axes if a.kind == "rotary"]
            index = 0 if m in (10, 11) else 1
            if len(rotaries) <= index:
                raise ControllerAlarm("machine_feature_missing", f"M{m}: no {('4th', '5th')[index]} axis")
            ex.emit(Command(Kind.MODE, ref, modal={"device": {"kind": "brake", "axis": rotaries[index].name,
                                                              "engaged": m in (10, 12), "code": f"M{m}"}}))
        elif m in (10, 11, 110, 111) and self.T:
            ex.emit(Command(Kind.CHUCK, ref, modal={"device": {"kind": "chuck", "code": f"M{m}",
                                                               "spindle": "sub" if m >= 110 else "main",
                                                               "action": "clamp" if m in (10, 110) else "unclamp"}}))
        elif m in (21, 22, 146, 147, 36, 37) and self.T:
            device = {21: ("tailstock", "advance"), 22: ("tailstock", "retract"), 146: ("steady_rest", "clamp"),
                      147: ("steady_rest", "unclamp"), 36: ("parts_catcher", "on"), 37: ("parts_catcher", "off")}[m]
            ex.emit(Command(Kind.CHUCK, ref, modal={"device": {"kind": device[0], "action": device[1], "code": f"M{m}"}}))
        elif m in (133, 134, 135, 143, 144, 145) and self.T:
            self.auxiliary_spindle(m, bw)
        elif m in (41, 42):
            spindle = machine.spindles.get(machine.path_spindle[ex.ctx.path])
            if spindle is None or spindle.gears < 2:
                ex.note("info", f"M{m}: this spindle has no gearbox, the code has no effect (Haas)")
            else:
                s.spindle.gear = 1 if m == 41 else 2
                self.emit_spindle()
        elif m in (51, 52, 53, 54, 55, 56, 59, 61, 62, 63, 64, 65, 66, 69):
            on = m < 60
            ex.emit(Command(Kind.MODE, ref, modal={"device": {"kind": "output", "code": f"M{m}", "on": on}}))
        elif m in (73, 74, 82, 86, 90, 91, 104, 105, 138, 139, 158, 159, 38, 39, 23, 24, 14, 15, 17, 18, 119, 219) and \
                not (m in (86,) and self.T):
            ex.emit(Command(Kind.MODE, ref, modal={"device": {"kind": "machine", "code": f"M{m}"}}))
        elif m in (130, 131):
            ex.note("info", f"M{m}: media display has no machine effect")
        elif m in (109,):
            raise ControllerAlarm("operator_input", "M109 interactive user input")
        elif m in (21, 22, 23, 24, 25, 29, 121, 122, 123, 124, 125, 126, 129):
            raise ControllerAlarm("interface_signal", f"M{m} waits for an M-FIN signal")
        elif m in (46, 48, 96, 95, 39, 75, 160, 161, 162, 163):
            raise ControllerAlarm("unsupported_function", f"M{m}")
        else:
            raise ControllerAlarm("undefined_m_code", f"M{m}")

    def auxiliary_spindle(self, m: int, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        machine = self.machine
        kind = "live_tool" if m in (133, 134, 135) else "sub"
        spindle = next((sp.id for sp in machine.spindles.values() if sp.kind == kind), None)
        if spindle is None:
            raise ControllerAlarm("machine_feature_missing", f"M{m}: no {kind.replace('_', ' ')} spindle")
        word = bw.take("P")
        if word is not None:
            s.live.s_value = float(self.integer(word))
        direction = {133: -1, 134: 1, 135: 0, 143: -1, 144: 1, 145: 0}[m]
        s.live.direction = direction
        omega = direction * s.live.s_value * 2.0 * math.pi / 60.0
        ex.emit(Command(Kind.SPINDLE, ex.line_ref(), spindle={"id": spindle, "omega_rad_s": omega, "css_m_s": None,
                                                              "max_rad_s": self.spindle_max(spindle), "gear": 1,
                                                              "orient_rad": None, "direction": direction}))

    def orient(self, bw: BlockWords) -> None:
        s = self.s
        angle = 0.0
        p = bw.take("P")
        r = bw.take("R")
        if r is not None:
            angle = self.rot(r) if r.point or r.from_variable else math.radians(r.value)
        elif p is not None:
            angle = math.radians(self.integer(p))
        s.spindle.direction = 0
        s.spindle.oriented = True
        self.emit_spindle(orient=angle)

    def after_tool_change(self, number: int) -> None:
        s = self.s
        position = s.sys.get("pcool", {}).get(str(number))
        nozzle = next((nid for nid, spec in self.machine.nozzles.items() if "M34" in spec["code"]), None)
        if position and nozzle is not None:
            self.coolant({}, "P-COOL", nozzles={nozzle: int(position)})


class TableTcp:
    """Haas G234 (TCPC: tip path kept during rotary moves) and G254 (DWO: end points only) on machines
    whose rotary axes all move the table. The command carries the commanded machine axes (inverse
    kinematics of the table chain), the tool tip in the work frame at rotary zero and the tool axis there."""

    def __init__(self, mode: str):
        self.mode = mode

    def _parts(self, ex, item):
        machine = ex.machine
        if not machine.table_kinematics_supported():
            raise ControllerAlarm("transform_unavailable", f"{item.transform}: needs table-type rotary axes")
        rotary = {}
        linear_end = {}
        for i, axis in enumerate(ex.axes):
            counts = int(round_half_away(item.end[i] / axis.quantum)) + item.offset[i]
            if axis.kind == "rotary":
                rotary[axis.name] = counts * axis.quantum
            else:
                linear_end[axis.letter] = item.end[i]
        length = float(item.modal.get("length_offset_m", 0.0))
        work_origin = np.zeros(3)
        tip = np.zeros(3)
        for k, letter in enumerate("XYZ"):
            if letter in ex.by_letter:
                index = ex.letters.index(letter)
                axis = ex.axes[index]
                tool = length if letter == "Z" else 0.0
                work_origin[k] = item.offset[index] * axis.quantum - tool
                tip[k] = linear_end.get(letter, 0.0)
        return rotary, work_origin, tip, length

    def machine_counts(self, ex, item) -> list[int]:
        rotary, work_origin, tip, length = self._parts(ex, item)
        transform = ex.machine.table_transform(rotary)
        point = transform @ np.append(work_origin + tip, 1.0)
        axis_dir = np.array(ex.machine.tool_axis)
        gauge = point[:3] + length * axis_dir
        counts = []
        for i, axis in enumerate(ex.axes):
            if axis.kind == "rotary":
                counts.append(int(round_half_away(item.end[i] / axis.quantum)) + item.offset[i])
            else:
                k = "XYZ".index(axis.letter) if axis.letter in "XYZ" else None
                value = gauge[k] if k is not None else item.end[i] + item.offset[i] * axis.quantum
                counts.append(int(round_half_away(value / axis.quantum)))
        return counts

    def to_command(self, ex, item) -> Command:
        rotary, work_origin, tip, length = self._parts(ex, item)
        counts = self.machine_counts(ex, item)
        target = {a.name: counts[i] * a.quantum for i, a in enumerate(ex.axes)}
        rotation = ex.machine.table_transform(rotary)[:3, :3]
        axis_w = rotation.T @ np.array(ex.machine.tool_axis)
        kind = {"rapid": Kind.RAPID, "linear": Kind.LINEAR, "arc": Kind.ARC, "probe": Kind.PROBE}[item.kind]
        arc = None
        if item.kind == "arc":
            from ..ir import ArcSpec
            arc = ArcSpec(item.plane, tuple(item.centre), tuple(item.normal), item.sweep_rad, item.r_start, item.r_end)
        modal = dict(item.modal)
        modal["tip_interpolation"] = "tip" if self.mode == "tcpc" else "joint"
        modal["work_origin_M0_m"] = [float(v) for v in work_origin]
        modal["tip_start_m"] = [float(item.start[ex.letters.index(l)]) if (l in ex.letters and len(item.start)
                                == len(ex.letters)) else 0.0 for l in "XYZ"]
        cmd = Command(kind, item.ref, Space.TIP, target, tuple(float(v) for v in tip),
                      tuple(float(v) for v in axis_w), arc, item.feed, item.smoothing, modal=modal)
        if item.kind == "probe":
            cmd.probe = dict(item.probe or {})
        return cmd

    def work_from_machine(self, ex, machine_counts: dict, applied: dict) -> dict:
        """Work-frame position (SI per letter) of the tool tip for machine axis counts."""
        rotary = {a.name: machine_counts.get(a.letter, 0) * a.quantum for a in ex.axes if a.kind == "rotary"}
        length = sum(applied.values()) * ex.machine.lin_increment_m
        base = ex.offset_counts(letters_tool=False)
        gauge = np.zeros(3)
        origin = np.zeros(3)
        for k, letter in enumerate("XYZ"):
            if letter in ex.by_letter:
                index = ex.letters.index(letter)
                axis = ex.axes[index]
                gauge[k] = machine_counts.get(letter, 0) * axis.quantum
                origin[k] = base[index] * axis.quantum
        point = gauge - length * np.array(ex.machine.tool_axis)
        inverse = np.linalg.inv(ex.machine.table_transform(rotary))
        p0 = (inverse @ np.append(point, 1.0))[:3]
        work = {}
        for index, axis in enumerate(ex.axes):
            if axis.letter in "XYZ":
                work[axis.letter] = float(p0["XYZ".index(axis.letter)] - origin["XYZ".index(axis.letter)])
            else:
                work[axis.letter] = (machine_counts.get(axis.letter, 0) - base[index]) * axis.quantum
        return work


hooks.register_transform("haas:G234", TableTcp("tcpc"), replace=True)
hooks.register_transform("haas:G254", TableTcp("dwo"), replace=True)


def build_haas(machine, settings, controller_db, db_id: str = "") -> Dialect:
    variant = machine.variant
    gcodes = dict(T_GCODES if variant == "T" else M_GCODES)
    gcodes.update(HAAS_EXTRA_T if variant == "T" else HAAS_EXTRA_M)
    gcodes.update(hooks.extra_gcodes("haas", variant))
    power_on = dict(T_POWER_ON if variant == "T" else M_POWER_ON)
    if variant == "M":
        power_on[23] = "G255"
        power_on[31] = "G269"
    words = (controller_db or {}).get("words") or {}
    table = (words.get("lathe") or {}) if variant == "T" else words
    supported_g = word_set(table.get("G"), "G") if table else None
    supported_m = word_set(table.get("M"), "M") if table else None
    alarms = AlarmTable("haas", ALARM_CODES, ALARM_TEXTS, controller_db, db_id)
    # ,C chamfer and ,R corner rounding between two G01 blocks (haas-ngc-mill-om-2020 p. 309-310)
    lexer = LexerRules(letters=frozenset(LETTERS), max_digits=dict(MAX_DIGITS), block_delete_switches=1,
                       max_o_digits=5, comma_letters=frozenset("CR"))
    params = fanuc_params(settings, "controller-haas")
    return Dialect("haas", variant, lexer, gcodes, power_on, alarms, HaasVars(variant), HaasHandler, params,
                   controller_db, db_id, supported_g, supported_m, frozenset(LETTERS | {",C", ",R"}))


register_dialect("haas", build_haas)

from .. import haas_cycles  # noqa: E402,F401  (registers the Haas cycles G12/G13/G47/G150)

__all__ = ["HaasHandler", "HaasVars", "TableTcp", "build_haas"]
