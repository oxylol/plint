"""FANUC Series 0i-F (0i-MF machining centres, 0i-TF lathes in G code system A), custom macro B.

Order of execution inside one block (docs/dialects.md "Fanuc"):

1. macro statement (a block is either a macro statement or NC words: PS0127 otherwise)
2. word values (literals by the decimal-point rules, expressions with <vacant> semantics; a
   vacant lone variable drops its address) - before anything changes, so a live read can stop
   the buffer and re-run the block
3. G and M code validation (unknown: PS0010 / not in the machine: TW0005), same-group rule
   (FANUC: the last G code of a group wins), macro calls by code (G65, settings macro_call_codes)
4. modal groups: units, plane, distance, feed mode, work offset, exact stop, polar, return level,
   CSS, length compensation (H), cutter compensation (D), scaling, mirror, rotation, modal call,
   stroke check, canned cycle
5. F, S, T words; M codes that start something (M03, M04, M07, M08, M29 ...)
6. one-shot codes (G04, G10, G27-G31, G37, G52, G53, G60, G92, G92.1)
7. motion of group 01 or one canned-cycle execution; the G66 call after a move
8. the other M codes (M05, M09, M06, M19, M00/M01/M02/M30, path waits, M98/M99)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import hooks
from ..alarms import AlarmTable, ControllerAlarm
from ..expr import round_half_away
from ..hooks import CycleCall, Step
from ..ir import Command, Kind
from ..lexer import LexerRules
from ..pipeline import PLANE_OF, PLANES
from ..state import new_locals
from .base import Dialect, GSpec, g_from_value, normalize_g, register_dialect, word_set
from .fanuc_tables import (ALARM_CODES, ALARM_TEXTS, ARGUMENTS_I, LETTERS, LINEAR_ARGUMENTS, M_GCODES,
                           M_POWER_ON, MAX_DIGITS, ROTARY_ARGUMENTS, T_GCODES, T_POWER_ON)
from .fanuc_vars import FanucVars, Units

INC_LETTERS = {"U": "X", "W": "Z", "V": "Y", "H": "C"}       # G code system A incremental addresses
# group 08 codes that apply the H tool length offset in the + sense (TCP codes include length compensation)
LENGTH_PLUS = ("G43", "G43.4", "G43.5", "G234", "G143")
FEED_MODE_OF = {"G93": "inverse_time", "G94": "per_min", "G95": "per_rev", "G98": "per_min", "G99": "per_rev"}
START_M = (3, 4, 7, 8, 13, 14, 29, 88, 83, 73, 133, 134, 143, 144)


@dataclass
class WordValue:
    letter: str
    value: float                    # program units, sign included
    point: bool                     # decimal point given (values from variables count as given)
    text: str = ""
    from_variable: bool = False


class BlockWords:
    """The evaluated words of one NC block."""

    def __init__(self):
        self.g: list[str] = []
        self.m: list[int] = []
        self.addr: dict[str, WordValue] = {}
        self.order: list[WordValue] = []
        self.used: set[str] = set()
        self.duplicates: list[str] = []

    def has(self, letter: str) -> bool:
        return letter in self.addr and letter not in self.used

    def take(self, letter: str) -> WordValue | None:
        if letter in self.addr and letter not in self.used:
            self.used.add(letter)
            return self.addr[letter]
        return None

    def peek(self, letter: str) -> WordValue | None:
        return self.addr.get(letter) if letter not in self.used else None


class FanucHandler:
    """Block semantics of FANUC 0i-F (M and T variants). Subclassed by the Haas dialect."""

    macro_alarm_key = "macro_alarm"
    m_after_motion_only = False
    repeat_letters = ("K", "L")         # canned-cycle repetition addresses (Haas: L only)

    def __init__(self, ex, dialect: Dialect):
        self.ex = ex
        self.d = dialect
        self.s = ex.s
        self.machine = ex.machine
        self.p = ex.p
        self.T = dialect.variant == "T"
        self.groups = {"plane": 16 if self.T else 2, "feed": 5, "css": 2 if self.T else 13,
                       "cycle": 10 if self.T else 9}

    # ================================================================== state set-up and reset
    def init_state(self) -> None:
        """Power-on state of a fresh path (not called on resume)."""
        ex, s = self.ex, self.s
        s.modal = {str(g): code for g, code in self.d.power_on.items()}
        for group, code in self.p.get("power_on", {}).items():
            s.modal[str(group)] = code
        units = self.p.get("power_on_units") or "mm"
        s.units = units
        s.modal["6"] = "G20" if units == "inch" else "G21"
        if self.machine.corner_default == "exact_stop" and not self.T:
            s.modal["15"] = "G61"
        s.smoothing = {"corner": "exact_stop" if s.modal.get("15") == "G61" else "blend",
                       "tolerance_m": 0.0, "mode_code": "", "lookahead": True}
        s.pos = {a.letter: a.home for a in ex.axes}
        s.mach = dict(s.pos)
        s.applied = {a.letter: 0 for a in ex.axes}
        n_axes = len(ex.axes)
        for key in ("EXT", "G54", "G55", "G56", "G57", "G58", "G59"):
            s.offsets[key] = [0] * n_axes
        for key, values in (ex.ctx.work_offsets_m or {}).items():
            s.offsets[str(key)] = self._offset_row(values)
        for tool in ex.ctx.tools:
            self.set_tool_offsets(tool)
        s.pumps = {pid: False for pid in self.machine.pumps}
        s.nozzles = {nid: 1 for nid in self.machine.nozzles}
        s.variables = {"c100": [None] * 100, "c500": [None] * 500}
        s.pos = ex.program_from_machine(s.mach)

    def _offset_row(self, values) -> list[int]:
        ex = self.ex
        row = [0] * len(ex.axes)
        if isinstance(values, dict):
            items = [(str(k), v) for k, v in values.items()]
        else:
            items = [(ex.axes[i].letter, v) for i, v in enumerate(list(values)[:len(ex.axes)])]
        for letter, value in items:
            if letter in ex.by_letter:
                axis = ex.by_letter[letter]
                si = float(value) / 2.0 if (axis.diameter and self.p.get("lathe_offsets_in_diameter")) else float(value)
                row[ex.letters.index(letter)] = int(round_half_away(si / axis.quantum))
        return row

    def set_tool_offsets(self, tool) -> None:
        """Initial offsets of one tool from the project (SI): length/radius geometry and wear."""
        inc = self.machine.lin_increment_m
        row = [0] * 9
        if self.T:
            x_axis = self.ex.by_letter.get("X")
            xq = x_axis.quantum if x_axis is not None else inc
            if x_axis is not None and x_axis.diameter and self.p.get("lathe_offsets_in_diameter"):
                xq *= 2.0           # X offsets are set (and read) as diameters
            row[4] = int(round_half_away(getattr(tool, "x_m", 0.0) / xq))
            row[5] = int(round_half_away(getattr(tool, "z_m", 0.0) / inc))
            row[6] = int(round_half_away(getattr(tool, "nose_radius_m", 0.0) / inc))
            row[3] = int(getattr(tool, "tip", 0) or 0)
            row[0] = int(round_half_away(getattr(tool, "x_wear_m", 0.0) / xq))
            row[1] = int(round_half_away(getattr(tool, "z_wear_m", 0.0) / inc))
            row[2] = int(round_half_away(getattr(tool, "radius_wear_m", 0.0) / inc))
        else:
            row[0] = int(round_half_away(getattr(tool, "length_m", 0.0) / inc))
            row[1] = int(round_half_away(getattr(tool, "length_wear_m", 0.0) / inc))
            row[2] = int(round_half_away(getattr(tool, "radius_m", 0.0) / inc))
            row[3] = int(round_half_away(getattr(tool, "radius_wear_m", 0.0) / inc))
        self.s.tool_offsets[str(int(tool.number))] = row

    def on_reset(self) -> None:
        """RESET / program end: cancel what FANUC cancels on reset (comp, cycle, modal call, transforms)."""
        s = self.s
        if s.cycle is not None or s.sys.get("rigid_tap"):
            self.cycle_cancelled((s.cycle or {}).get("code") or "G84")
        for group, code in (("7", "G40"), (str(self.groups["cycle"]), "G80"), ("12", "G67")):
            if group in s.modal:
                s.modal[group] = code
        if not self.T:
            s.modal["16"] = "G69"
            s.modal["11"] = "G50"
            s.modal["22"] = "G50.1"
            s.modal["17"] = "G15"
        s.rotation = None
        s.scale = None
        s.mirror = {}
        s.feature = None
        if s.sys.get("tcp_group") == 16:          # a tilted working plane (G68.2 ...) ends with the reset
            s.tcp = ""
            s.sys.pop("tcp_group", None)
        s.sys.pop("tilted", None)
        s.cycle = None
        s.cycle_initial = None
        s.sys.pop("modal_calls", None)
        s.locals = [new_locals()]
        self.ex.invalidate_transform()
        provider = hooks.TOOL_LIFE.get("default")
        if provider is not None and hasattr(provider, "reset"):
            provider.reset(self.ex)

    # ================================================================== numbers
    def linear_increment_units(self) -> float:
        """Least input increment in the current input unit (0.001 mm, 0.0001 inch for IS-B)."""
        mm = self.machine.lin_increment_m * 1000.0
        return mm / 10.0 if self.s.units == "inch" else mm

    def rotary_increment_deg(self) -> float:
        return math.degrees(self.machine.rot_increment_rad)

    def lin(self, w: WordValue) -> float:
        """A length word in SI (decimal-point rule, rounding to the least input increment)."""
        inc = self.linear_increment_units()
        value = w.value if (w.point or self.p.get("pocket_calculator")) else w.value * inc
        n = round_half_away(value / inc)
        return n * inc * (Units.linear_unit_m(self.ex))

    def rot(self, w: WordValue) -> float:
        inc = self.rotary_increment_deg()
        value = w.value if (w.point or self.p.get("pocket_calculator")) else w.value * inc
        return math.radians(round_half_away(value / inc) * inc)

    def integer(self, w: WordValue) -> int:
        """An address that takes no decimal point (M, T, H, D, P, L, N ...)."""
        if w.point and not w.from_variable:
            raise ControllerAlarm("illegal_decimal_point", f"{w.letter}{w.text}")
        return int(round_half_away(w.value))

    def factor(self, w: WordValue, unit: float = 0.001) -> float:
        """A scaling factor: with a decimal point as written, without one in units of ``unit``."""
        return w.value if w.point else w.value * unit

    def seconds(self, w: WordValue, letter_ms: bool) -> float:
        """Dwell time: P in ms (integer), X/U in s (without point: least increment 0.001 s)."""
        if letter_ms:
            return self.integer(w) / 1000.0
        return w.value if w.point else w.value * 0.001

    def feed_value(self, w: WordValue, mode: str) -> tuple[float, float]:
        """(SI value, programmed number) of an F word in feed ``mode``."""
        inch = self.s.units == "inch"
        value = w.value
        if not w.point and not self.p.get("pocket_calculator"):
            value = value * self.feed_no_point_scale(mode, inch)
        if mode == "per_min":
            si = value * (0.0254 if inch else 1e-3) / 60.0
        elif mode == "per_rev":
            si = value * (0.0254 if inch else 1e-3)
        else:
            si = value / 60.0
        return si, w.value

    def feed_no_point_scale(self, mode: str, inch: bool) -> float:
        if mode == "per_min":
            return 0.01 if inch else 1.0
        if mode == "per_rev":
            return 0.0001 if inch else 0.01
        return 1.0

    def axis_counts(self, axis, w: WordValue, diameter_value: bool = True) -> int:
        if axis.kind == "rotary":
            return int(round_half_away(self.rot(w) / axis.quantum))
        si = self.lin(w)
        if axis.diameter and diameter_value:
            si /= 2.0
        return int(round_half_away(si / axis.quantum))

    # ================================================================== queries used by the core
    def plane(self) -> str:
        return PLANE_OF.get(self.s.modal.get(str(self.groups["plane"]), "G18" if self.T else "G17"), "XY")

    def feed_mode(self) -> str:
        return FEED_MODE_OF.get(self.s.modal.get(str(self.groups["feed"]), "G99" if self.T else "G94"), "per_min")

    def absolute(self) -> bool:
        return True if self.T else self.s.modal.get("3", "G90") == "G90"

    def tool_offset_counts(self, letter: str) -> int:
        s = self.s
        if self.T:
            if s.offset_number <= 0:
                return 0
            row = s.tool_offsets.get(str(s.offset_number))
            if row is None:
                return 0
            if letter == "X":
                return row[4] + row[0]
            if letter == "Z":
                return row[5] + row[1]
            if letter == "Y":
                return row[8] + row[7]
            return 0
        length = s.modal.get("8", "G49")
        if length not in LENGTH_PLUS and length != "G44" or letter != s.tool_axis_letter or s.h_number <= 0:
            return 0
        row = s.tool_offsets.get(str(s.h_number))
        if row is None:
            return 0
        value = row[0] + row[1]
        return -value if length == "G44" else value

    def comp_radius(self) -> float:
        s = self.s
        inc = self.machine.lin_increment_m
        if self.T:
            row = s.tool_offsets.get(str(s.offset_number))
            return (row[6] + row[2]) * inc if row else 0.0
        row = s.tool_offsets.get(str(s.d_number)) if s.d_number > 0 else None
        return (row[2] + row[3]) * inc if row else 0.0

    def modal_group_value(self, group: int) -> float:
        code = self.s.modal.get(str(group))
        if code is None:
            return 0.0
        return float(code[1:])

    def modal_snapshot(self) -> dict:
        s = self.s
        ex = self.ex
        offsets = ex.offset_counts(letters_tool=False)
        snap = {
            "plane": self.plane(), "units": s.units, "distance": "abs" if self.absolute() else "inc",
            "feed_mode": self.feed_mode(), "work_offset": s.wcs,
            "work_offset_m": [offsets[i] * a.quantum for i, a in enumerate(ex.axes)],
            "tool": s.tool, "h": s.h_number, "d": s.d_number,
            "length_offset_m": self.tool_offset_counts(s.tool_axis_letter) * self.machine.lin_increment_m,
            "radius_offset_m": self.comp_radius(), "comp": s.modal.get("7", "G40"),
            "motion": s.modal.get("1", "G00"), "cycle": (s.cycle or {}).get("code"),
            "css": s.spindle.css, "exact_stop": s.smoothing.get("corner") == "exact_stop",
        }
        if self.T:
            snap["offset_number"] = s.offset_number
            # work + tool offsets per axis (path order): program = machine - program_offset_m (X as radius);
            # the constant surface speed is computed at the program X (the tip's radius about the spindle)
            snap["program_offset_m"] = [c * a.quantum for c, a in zip(ex.offset_counts(), ex.axes)]
        if s.tool_axis_letter != "Z":
            snap["tool_axis_letter"] = s.tool_axis_letter
        if s.sys.get("3004"):
            snap["feed_hold_inhibit"] = bool(int(s.sys["3004"]) & 1)
        if s.sys.get("rigid_tap"):
            snap["rigid_tap"] = True
        return snap

    def apply_mode(self, step: Step) -> None:
        """A cycle 'mode' step: data in step.data applied to the modal state (and announced)."""
        for key, value in step.data.items():
            self.s.sys[key] = value
        self.ex.emit(Command(Kind.MODE, self.ex.line_ref(), modal=dict(step.data)))

    # ================================================================== block execution
    def execute(self, block) -> None:
        ex, s = self.ex, self.s
        if block.error is not None:
            raise ControllerAlarm(*block.error)
        if block.delete and s.operator.block_delete[block.delete - 1]:
            return
        if block.lowercase:
            self.lowercase(block)
        record = self.log_block(block)
        if block.stmt is not None:
            self.statement(block.stmt)
            return
        if not block.words:
            if self.box_cycle_mode():
                self.nc_block(BlockWords())         # FANUC: an EOB-only block in G90/G92/G94 mode repeats the cycle
            return
        words = block.words
        if block.mid_delete_at >= 0 and s.operator.block_delete[block.mid_delete - 1]:
            words = words[:block.mid_delete_at]
        bw = self.evaluate(words)
        if record is not None:
            record["words"] = {letter: float(w.value) for letter, w in bw.addr.items()}
        self.nc_block(bw)
        s.sys.pop("g09", None)

    def log_block(self, block) -> dict | None:
        """The executor's analysis log entry of this block (when the log is on): its execution counter,
        call depth and program line; ``words`` are added once evaluated."""
        log = self.ex.block_log
        if log is None:
            return None
        ref = self.ex.line_ref()
        seq = int(self.s.block_seq)
        if log and log[-1]["seq"] == seq and log[-1]["line"] == int(block.line) and log[-1]["program"] == ref.program:
            return log[-1]                  # the same block again after a live read (buffer stop)
        record = {"seq": seq, "depth": len(self.s.frames), "program": ref.program, "line": int(block.line),
                  "words": {}}
        log.append(record)
        return record

    def lowercase(self, block) -> None:
        self.ex.note("warning", "lower-case address letters read as upper case")

    def evaluate(self, words) -> BlockWords:
        ex = self.ex
        bw = BlockWords()
        for word in words:
            if word.expr is not None:
                unit = self.linear_increment_units() if word.letter in "XYZUVWIJKRQ" else 1.0
                value = ex.ev.address_value(word.expr, unit)
                if value is None:
                    continue
                wv = WordValue(word.letter, float(value), True, word.text, True)
            else:
                wv = WordValue(word.letter, float(word.value), word.point, word.text, False)
            if wv.letter == "G":
                bw.g.append(g_from_value(wv.value) if wv.from_variable else normalize_g(wv.text))
            elif wv.letter == "M":
                bw.m.append(self.integer(wv))
            else:
                if wv.letter in bw.addr and wv.letter not in ("I", "J", "K"):
                    bw.duplicates.append(wv.letter)
                bw.addr[wv.letter] = wv
                bw.order.append(wv)
        self.s.last_words = {k: v.value for k, v in bw.addr.items()}
        return bw

    # ------------------------------------------------------------------ macro statements
    def statement(self, stmt: tuple) -> None:
        ex = self.ex
        kind = stmt[0]
        if kind == "assign":
            self.assign(stmt[1], stmt[2])
        elif kind == "if_goto":
            if ex.ev.truth(stmt[1]):
                ex.goto(self.sequence_number(stmt[2]))
        elif kind == "if_then":
            if ex.ev.truth(stmt[1]):
                self.assign(stmt[2][1], stmt[2][2])
        elif kind == "goto":
            ex.goto(self.sequence_number(stmt[1]))
        elif kind == "while":
            ex.loop_while(ex.ev.truth(stmt[1]), stmt[2])
        elif kind == "do":
            ex.loop_while(True, stmt[1])
        elif kind == "end":
            ex.loop_end(stmt[1])
        elif kind == "popen":
            self.s.sys["dprnt_open"] = True
        elif kind == "pclos":
            self.s.sys["dprnt_open"] = False
        elif kind == "dprnt":
            self.dprnt(stmt[1])
        else:
            raise ControllerAlarm("unsupported_function", kind.upper())

    def assign(self, target: tuple, node: tuple) -> None:
        ex = self.ex
        number = target[1] if target[0] == "var" else ex.ev.index(target[1])
        value = ex.ev.value(node)
        ex.write(number, value)

    def sequence_number(self, node: tuple) -> int:
        value = self.ex.ev.value(node)
        if value is None:
            raise ControllerAlarm("sequence_out_of_range", "GOTO <vacant>")
        return int(round_half_away(value))

    def dprnt(self, items: tuple) -> None:
        parts = []
        for item in items:
            if item[0] == "text":
                parts.append(item[1].replace("*", " "))
            else:
                value = self.ex.ev.value(item[1])
                digits, fraction = item[2], item[3]
                if value is None:
                    parts.append(" " * (digits + fraction + 1))
                else:
                    parts.append(f"{value:{digits + fraction + 1}.{fraction}f}")
        self.s.printed.append("".join(parts))

    # ------------------------------------------------------------------ NC block
    def gspec(self, code: str) -> GSpec:
        spec = self.d.gcodes.get(code)
        supported = self.d.supported_g
        if supported is not None and code not in supported:
            raise ControllerAlarm("improper_g_code", code)
        if spec is None:
            if supported is not None and code in supported:
                raise ControllerAlarm("unsupported_function", f"{code} (listed by the control, not modelled)")
            raise ControllerAlarm("improper_g_code", code)
        if spec.kind == "unsupported":
            raise ControllerAlarm("unsupported_function", code)
        return spec

    def nc_block(self, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        mode = s.sys.get("data_mode")
        if mode:
            provider = hooks.DATA_MODES[mode]
            if "G11" in bw.g:
                s.sys.pop("data_mode", None)
                provider.end(ex)
                return
            provider.block(ex, bw)
            return
        for code in bw.g:
            if code in self.p.get("macro_call_codes", {}) and not self._inside_code_call(code):
                self.code_call(code, bw)
                return
        specs = [(code, self.gspec(code)) for code in bw.g]
        if any(spec.kind == "macro_call" for _, spec in specs):
            self.g65(bw, [c for c, sp in specs if sp.kind != "macro_call"])
            return
        modal: dict[int, tuple[str, GSpec]] = {}
        oneshots: list[tuple[str, GSpec]] = []
        for code, spec in specs:
            if spec.group == 0:
                oneshots.append((code, spec))
                continue
            previous = modal.get(spec.group)
            if previous is not None and previous[0] != code:
                self.same_group(previous[0], code)
            modal[spec.group] = (code, spec)
        if sum(1 for _, spec in oneshots if spec.axis_words) > 1:
            raise ControllerAlarm("improper_g_code", " ".join(c for c, sp in oneshots if sp.axis_words))
        if self.T and any(code != "G04" and not spec.cycle for code, spec in oneshots):
            s.sys.pop("box_cycle", None)        # modal data of G90/G92/G94 cleared by a one-shot G code
        if bw.duplicates:
            self.duplicate_address(bw.duplicates)
        if len(bw.m) > self.p["max_m_codes_per_block"]:
            raise ControllerAlarm("multiple_codes", " ".join(f"M{m}" for m in bw.m))
        for m in bw.m:
            if not self.m_known(m):
                raise ControllerAlarm("undefined_m_code", f"M{m:02d}")
        reserved = self.m_letters(bw.m)
        if 6 in modal:
            self.set_units(modal[6][0])
        for group in sorted(modal):
            if group == 6:
                continue
            code, spec = modal[group]
            self.modal_code(code, spec, bw)
        self.f_word(bw)
        self.s_word(bw)
        self.t_word(bw)
        cycle_letters = self.cycle_data_letters(oneshots)
        for kind in ("H", "D"):
            if kind in cycle_letters:
                continue                        # the cycle reads the word as its data (oneshot_cycle_data)
            word = bw.take(kind)
            if word is not None:
                number = self.offset_word(kind, word)
                if number < 0 or number > self.max_offset():
                    raise ControllerAlarm("illegal_offset_number", f"{kind}{number}")
                if kind == "H":
                    s.h_number = number
                else:
                    s.d_number = number
        after: list[int] = []
        for m in bw.m:
            if m in START_M and not self.m_after_motion_only:
                self.m_code(m, bw)
            else:
                after.append(m)
        moved = False
        for code, spec in oneshots:
            if self.oneshot(code, spec, bw):
                moved = True
        if not any(spec.axis_words or spec.cycle for _, spec in oneshots):
            moved = self.motion(bw, modal) or moved
        self.leftover(bw, reserved)
        if moved:
            self.modal_call_after_move()
        if after and s.pending:
            words = {k: [v.value, v.point, v.from_variable] for k, v in bw.addr.items() if k in reserved}
            ex.queue_steps([Step("mcodes", data={"m": after, "words": words})])
            return
        for m in after:
            self.m_code(m, bw)

    def deferred_m(self, step: Step) -> None:
        """M codes of a cycle block, run after the cycle's steps."""
        bw = BlockWords()
        for letter, (value, point, from_variable) in step.data.get("words", {}).items():
            word = WordValue(letter, float(value), bool(point), "", bool(from_variable))
            bw.addr[letter] = word
            bw.order.append(word)
        for m in step.data["m"]:
            self.m_code(int(m), bw)

    def m_letters(self, codes: list[int]) -> set[str]:
        """Address letters the block's M codes take as their own data."""
        letters: set[str] = set()
        for m in codes:
            if m in (98, 97):
                letters |= {"P", "L"}
            elif m == 99 or self.is_wait(m):
                letters.add("P")
            elif m == 19:
                letters |= {"R", "P"}
            else:
                custom = self.p.get("m_codes", {}).get(f"M{m:02d}") or self.p.get("m_codes", {}).get(f"M{m}")
                if custom is not None and custom.get("kind") == "spindle":
                    letters |= {"S", "P"}
                letters |= self.m_data_letters(m)
        return letters

    def m_data_letters(self, m: int) -> set[str]:
        return set()

    def cycle_data_letters(self, oneshots: list) -> set[str]:
        """Offset addresses (H, D) that a one-shot cycle of the block takes as its own data instead of an
        offset number. FANUC: none (the T series cycles use U/W/R/P/Q/F, the M series cycles take D as the
        offset number)."""
        return set()

    def max_offset(self) -> int:
        return 64 if self.T else 999

    def offset_word(self, kind: str, word: WordValue) -> int:
        """The H or D number of a word, after tool life management (H99 / D99 select the offsets registered
        for the tool whose life is counted)."""
        number = self.integer(word)
        provider = hooks.TOOL_LIFE.get("default")
        if provider is not None and hasattr(provider, "offset_code"):
            number = int(provider.offset_code(self.ex, kind, number))
        return number

    def same_group(self, first: str, second: str) -> None:
        if self.p.get("same_group_alarm"):
            raise ControllerAlarm("multiple_codes", f"{first} and {second} in one block")
        self.ex.note("warning", f"{first} and {second} of one group in one block: {second} is used (FANUC rule)")

    def duplicate_address(self, letters: list[str]) -> None:
        if self.p.get("same_group_alarm"):
            raise ControllerAlarm("multiple_codes", "address given twice: " + " ".join(letters))
        self.ex.note("warning", "address given twice in one block, the last value is used: " + " ".join(letters))

    def leftover(self, bw: BlockWords, reserved: set[str]) -> None:
        """Words no function of the block used: an alarm, never silently ignored (an axis address the
        path does not have ends here too)."""
        for letter in bw.addr:
            if letter in bw.used or letter in reserved:
                continue
            word = bw.addr[letter]
            detail = f"{letter}{word.text}" if word.text else letter
            if letter in "XYZABCUVW" and letter not in self.ex.by_letter:
                raise ControllerAlarm("improper_address", f"{detail}: no such axis on this path")
            raise ControllerAlarm("improper_address", f"{detail} has no use in this block")

    # ------------------------------------------------------------------ modal groups
    def set_units(self, code: str) -> None:
        s = self.s
        units = "inch" if code == "G20" else "mm"
        s.modal["6"] = code
        s.units = units

    def modal_code(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        group = str(spec.group)
        kind = spec.kind
        if kind == "plane":
            if s.modal.get("7", "G40") != "G40" and s.modal.get(group) != code:
                raise ControllerAlarm("plane_change_in_comp", code)
            s.modal[group] = code
        elif kind == "distance":
            s.modal[group] = code
        elif kind == "feed":
            if s.modal.get(group) != code and self.p.get("feed_mode_change_clears_f") and not bw.has("F"):
                s.feed_si, s.feed_set = 0.0, False
            s.modal[group] = code
        elif kind == "wcs":
            self.select_wcs(code, spec, bw)
        elif kind == "exact":
            s.modal[group] = code
            corner = {"exact_stop": "exact_stop", "blend": "blend", "tapping": "tapping",
                      "corner_override": "blend"}[spec.arg]
            if s.smoothing.get("corner") != corner:
                s.smoothing["corner"] = corner
                ex.emit(Command(Kind.MODE, ex.line_ref(), smoothing=ex.smoothing_spec(), modal={"exact": code}))
        elif kind == "polar":
            s.modal[group] = code
            if code == "G16":
                s.polar_angle_deg = 0.0
        elif kind == "return":
            s.modal[group] = code
        elif kind == "css":
            s.modal[group] = code
            s.spindle.css = code == "G96"
        elif kind == "length":
            self.length_comp(code, bw)
        elif kind == "comp":
            self.cutter_comp(code, bw)
        elif kind == "scale":
            self.scaling(code, bw)
        elif kind == "mirror":
            self.mirror(code, bw)
        elif kind == "rotation":
            self.rotation(code, bw)
        elif kind == "modal_call":
            self.g66(code, bw)
        elif kind == "stroke":
            s.modal[group] = code
            limits = {}
            if code == "G22":
                for letter in list(bw.addr):
                    if letter in "XYZIJK":
                        limits[letter] = self.lin(bw.take(letter))
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"stroke_check": code == "G22", "stroke_limits_m": limits}))
        elif kind == "fluctuation":
            s.modal[group] = code
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"speed_fluctuation_detection": code == "G26"}))
        elif kind == "cycle":
            self.cycle_mode(code, spec, bw)
        elif kind == "motion":
            if s.cycle is not None:
                previous = s.cycle.get("code")
                s.cycle = None
                s.modal[str(self.groups["cycle"])] = "G80"
                self.cycle_cancelled(previous)
            s.modal[group] = code
        elif kind == "tcp":
            self.tcp_mode(code, spec, bw)
        elif kind == "ext":
            self.ext_code(code, spec, bw)
        else:
            self.modal_extra(code, spec, bw)

    def modal_extra(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        raise ControllerAlarm("unsupported_function", code)

    def ext_code(self, code: str, spec: GSpec, bw: BlockWords) -> bool:
        """A G code of kind ``ext``: executed by the provider registered under ``spec.arg``
        (hooks.register_gcode_handler); returns True when it moved the axes."""
        handler = hooks.GCODE_HANDLERS.get(spec.arg)
        if handler is None:
            raise ControllerAlarm("unsupported_function", code)
        return bool(handler.execute(self.ex, code, spec, bw))

    def cycle_cancelled(self, code: str | None) -> None:
        """The canned-cycle mode of ``code`` ended (G80, a group 01 code, reset): the cycle provider's
        optional ``cancel(ex, code)`` releases what the mode held (rigid tapping, ...)."""
        provider = hooks.find(hooks.CYCLES, self.d.name, code) if code else None
        if provider is not None and hasattr(provider, "cancel"):
            provider.cancel(self.ex, code)

    def tcp_mode(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        s = self.s
        group = str(spec.group)
        if spec.arg == "off":
            s.modal[group] = code
            if s.sys.get("tcp_group") in (spec.group, None):
                s.tcp = ""
                s.sys.pop("tcp_group", None)
            return
        if hooks.TRANSFORMS.get(spec.arg) is None:
            raise ControllerAlarm("transform_unavailable", code)
        if spec.group == 8:
            word = bw.take("H")
            if word is not None:
                number = self.offset_word("H", word)
                if number < 0 or number > self.max_offset():
                    raise ControllerAlarm("illegal_offset_number", f"H{number}")
                s.h_number = number
            s.tool_axis_letter = "Z"
        s.modal[group] = code
        s.tcp = spec.arg
        s.sys["tcp_group"] = spec.group

    def select_wcs(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        s = self.s
        if spec.arg == "P":
            word = bw.take("P")
            number = self.integer(word) if word is not None else 1
            if not 1 <= number <= 300:
                raise ControllerAlarm("illegal_offset_number", f"{code} P{number}")
            key = f"P{number}"
        else:
            key = spec.arg
        s.modal["14"] = code
        if key != s.wcs:
            s.wcs = key
            s.offsets.setdefault(key, [0] * len(self.ex.axes))
            self.ex.resync_program_position()

    def length_comp(self, code: str, bw: BlockWords) -> None:
        s = self.s
        word = bw.take("H")
        if word is not None:
            number = self.offset_word("H", word)
            if number < 0 or number > self.max_offset():
                raise ControllerAlarm("illegal_offset_number", f"H{number}")
            s.h_number = number
        s.modal["8"] = code
        if s.sys.get("tcp_group") == 8:
            s.tcp = ""
            s.sys.pop("tcp_group", None)
        s.tool_axis_letter = "Z"
        if self.p.get("tool_length_axis") == "C" and code != "G49":
            for letter in ("X", "Y", "Z"):
                if bw.has(letter):
                    s.tool_axis_letter = letter
                    break

    def cutter_comp(self, code: str, bw: BlockWords) -> None:
        s, ex = self.s, self.ex
        if "G150" in bw.g:
            s.sys["g150_side"] = code
            return
        word = bw.take("D")
        if word is not None:
            number = self.offset_word("D", word)
            if number < 0 or number > self.max_offset():
                raise ControllerAlarm("illegal_offset_number", f"D{number}")
            s.d_number = number
        active = s.comp_state is not None and s.comp_kind in ("cutter", "nose") \
            and s.modal.get("7", "G40") in ("G41", "G42")
        if active and code == "G40":
            self.comp_marker({"cancel": "G40"})
        elif active and code in ("G41", "G42"):
            self.comp_marker({"respecified": code})
        if code in ("G41", "G42"):
            ex.start_comp("nose" if self.T else "cutter")
        s.modal["7"] = code

    def comp_marker(self, data: dict) -> None:
        """Tell the active compensation filter about a group 07 command of this block (G40 or G41/G42
        given again in offset mode) before the block's motion; the filter consumes the marker."""
        ex = self.ex
        marker = dict(data, seq=self.s.block_seq)
        ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"comp_marker": marker}))

    def scaling(self, code: str, bw: BlockWords) -> None:
        s, ex = self.s, self.ex
        s.modal["11"] = code
        if code == "G50":
            s.scale = None
            ex.invalidate_transform()
            ex.resync_program_position()
            return
        centre: dict[str, int] = {}
        factor: dict[str, float] = {}
        common = None
        p = bw.take("P")
        if p is not None:
            common = self.factor(p)
        for letter, axis_letter in (("I", "X"), ("J", "Y"), ("K", "Z")):
            word = bw.take(letter)
            if word is not None:
                factor[axis_letter] = self.factor(word)
        for axis in ex.axes:
            if axis.kind != "linear":
                continue
            word = bw.take(axis.letter)
            centre[axis.letter] = self.axis_counts(axis, word) if word is not None else s.pos.get(axis.letter, 0)
            if axis.letter not in factor:
                factor[axis.letter] = common if common is not None else 1.0
        for letter, value in factor.items():
            if value == 0.0:
                raise ControllerAlarm("illegal_scale", f"factor 0 for {letter}")
        s.scale = {"centre": centre, "factor": factor}
        ex.invalidate_transform()
        ex.resync_program_position()

    def mirror(self, code: str, bw: BlockWords) -> None:
        s, ex = self.s, self.ex
        s.modal["22"] = code
        letters = [a.letter for a in ex.axes if bw.has(a.letter)]
        if code in ("G50.1", "G100"):
            if letters:
                for letter in letters:
                    bw.take(letter)
                    s.mirror.pop(letter, None)
            else:
                s.mirror = {}
        else:
            for letter in letters:
                axis = ex.by_letter[letter]
                s.mirror[letter] = self.axis_counts(axis, bw.take(letter))
        ex.invalidate_transform()
        ex.resync_program_position()

    def rotation(self, code: str, bw: BlockWords) -> None:
        s, ex = self.s, self.ex
        s.modal["16"] = code
        if code == "G69":
            s.rotation = None
            ex.invalidate_transform()
            ex.resync_program_position()
            return
        plane = self.plane()
        a, b, _ = PLANES[plane]
        centre = []
        for letter, centre_letter in self.rotation_centre_letters(a, b):
            word = bw.take(centre_letter)
            axis = ex.by_letter.get(letter)
            if word is not None and axis is not None:
                value = self.axis_counts(axis, word)
                if not self.absolute():
                    value += s.pos.get(letter, 0)
                centre.append(value)
            else:
                centre.append(s.pos.get(letter, 0))
        word = bw.take("R")
        angle = math.degrees(self.rot(word)) if word is not None else 0.0
        s.rotation = {"plane": plane, "centre": centre, "angle_deg": angle}
        ex.invalidate_transform()
        ex.resync_program_position()

    def rotation_centre_letters(self, a: str, b: str):
        return ((a, a), (b, b))

    def g66(self, code: str, bw: BlockWords) -> None:
        s, ex = self.s, self.ex
        s.modal["12"] = code
        calls = s.sys.setdefault("modal_calls", [])
        if code == "G67":
            if calls:
                calls.pop()
            return
        word = bw.take("P")
        if word is None:
            raise ControllerAlarm("missing_p", "G66 without P")
        program = self.find_program(self.integer(word))
        repeat_word = bw.take("L")
        repeat = self.integer(repeat_word) if repeat_word is not None else 1
        args = self.arguments(bw)
        if len(calls) >= 4:
            raise ControllerAlarm("modal_call_nesting", "G66 nested more than 4 deep")
        calls.append({"program": program, "args": args, "repeat": repeat, "depth": len(s.frames)})

    def modal_call_after_move(self) -> None:
        s = self.s
        calls = s.sys.get("modal_calls")
        if not calls:
            return
        call = calls[-1]
        depth = call["depth"]
        if any(f.kind == "modal" for f in s.frames[depth:]):
            return
        if len(s.frames) < depth:
            return
        self.ex.request_call("modal", call["program"], list(call["args"]), call["repeat"])

    def cycle_mode(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        s = self.s
        group = str(spec.group)
        if spec.arg == "off":
            previous = (s.cycle or {}).get("code") or ("G84" if s.sys.get("rigid_tap") else None)
            s.modal[group] = code
            s.cycle = None
            s.cycle_initial = None
            self.cycle_cancelled(previous)
            return
        if self.T and spec.group in (0, 1):
            s.modal[group] = code if spec.group == 1 else s.modal.get(group, "")
            return
        drill = self.drilling_axis(code)
        if s.cycle is None or s.cycle.get("code") != code:
            data = dict(s.cycle["data"]) if s.cycle is not None else {}
            if s.cycle is None:
                s.cycle_initial = self.ex.si(drill, s.pos.get(drill, 0)) if drill in self.ex.by_letter else 0.0
            s.cycle = {"code": code, "data": data}
        s.modal[group] = code

    # ------------------------------------------------------------------ F S T
    def f_word(self, bw: BlockWords) -> None:
        s = self.s
        word = bw.take("F")
        if word is None:
            return
        mode = self.feed_mode()
        si, programmed = self.feed_value(word, mode)
        if si < 0.0:
            raise ControllerAlarm("illegal_minus", f"F{word.text}")
        max_feed = self.machine.max_feed_m_s
        if mode == "per_min" and max_feed > 0.0 and si > max_feed:
            self.ex.note("warning", f"F {si * 60000:.1f} mm/min clamped to the maximum cutting feed "
                                    f"{max_feed * 60000:.1f} mm/min")
            si = max_feed
        s.feed_si, s.feed_prog, s.feed_set = si, programmed, True

    def s_word(self, bw: BlockWords) -> None:
        s = self.s
        if ("G50" in bw.g and self.T) or ("G92" in bw.g and not self.T):
            return                  # S of G50 (T) / G92 (M) is the maximum spindle speed, read by that code
        word = bw.take("S")
        if word is None:
            return
        value = self.spindle_number(word)
        s.spindle.s_value = value
        if s.spindle.direction != 0:
            self.emit_spindle()

    def spindle_number(self, word: WordValue) -> float:
        if word.point and not word.from_variable:
            raise ControllerAlarm("illegal_decimal_point", f"S{word.text}")
        if word.value < 0:
            raise ControllerAlarm("illegal_minus", f"S{word.text}")
        return float(round_half_away(word.value))

    def t_word(self, bw: BlockWords) -> None:
        s, ex = self.s, self.ex
        word = bw.take("T")
        if word is None:
            return
        number = self.integer(word)
        if number < 0:
            raise ControllerAlarm("illegal_minus", f"T{number}")
        if "T" in self.p.get("macro_call_codes", {}) and not self._inside_code_call("T"):
            ex.write(149, float(number))
            self.code_call("T", None)
            return
        provider = hooks.TOOL_LIFE.get("default")
        if provider is not None and hasattr(provider, "t_code"):      # tool life group calls (T99nn, T199 ...)
            number = int(provider.t_code(ex, number))
        if self.T:
            self.lathe_t(number)
            return
        capacity = self.machine.tool_capacity
        if capacity and number > capacity:
            raise ControllerAlarm("tool_not_in_magazine", f"T{number} (capacity {capacity})")
        if s.preselect != number:
            s.preselect = number
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"tool_preselect": number}))

    def lathe_t(self, number: int) -> None:
        s, ex = self.s, self.ex
        station, offset = divmod(number, 100)
        stations = max([t["stations"] for t in self.machine.turrets] or [0])
        if stations and station > stations:
            raise ControllerAlarm("tool_not_in_magazine", f"T{number:04d}: station {station} of {stations}")
        if offset > self.max_offset():
            raise ControllerAlarm("illegal_offset_number", f"T{number:04d}")
        if station != 0 and station != s.tool:
            station = self.tool_life_select(station)
            slot = self.tool_slot(station)
            ex.emit(Command(Kind.TOOL_CHANGE, ex.line_ref(),
                            tool={"number": station, "station": station, "slot": slot,
                                  "turret": self.machine.turrets[0]["id"] if self.machine.turrets else "",
                                  "offset": offset}, modal=dict(ex.snapshot())))
            s.tool = station
        s.offset_number = offset

    def tool_life_select(self, number: int) -> int:
        provider = hooks.TOOL_LIFE.get("default")
        if provider is not None:
            return int(provider.select(self.ex, number))
        if number in self.s.expired_tools:
            self.ex.note("warning", f"T{number} called after its tool life expired (no tool life management)")
        return number

    def probe_name(self, number: int | None) -> str | None:
        """The probe a T number calls (a ``kind: probe`` entry of the project's tool list), else None."""
        for tool in self.ex.ctx.tools:
            if int(tool.number) == number and getattr(tool, "probe", None):
                return str(tool.probe)
        return None

    def tool_slot(self, number: int) -> int:
        if self.probe_name(number) is not None:
            return -1                       # a probe: not a cutting tool of the tool table, but a listed T number
        table = self.ex.ctx.tool_table
        if table is None:
            return -1
        slot = table.by_number(number, self.ex.ctx.path)
        if slot is None:
            self.ex.note("warning", f"T{number} is not in the project's tool list")
            return -1
        return slot.slot

    # ------------------------------------------------------------------ M codes
    gate_m_by_database = False

    def m_known(self, m: int) -> bool:
        """An M code is executed only if the dialect (with this machine's devices), the project's
        machine-builder table, the path-wait range or a macro-call code defines it."""
        if f"M{m:02d}" in self.p.get("m_codes", {}) or f"M{m}" in self.p.get("m_codes", {}):
            return True
        if self.is_wait(m) or f"M{m:02d}" in self.p.get("macro_call_codes", {}):
            return True
        if m not in self.m_table():
            return False
        supported = self.d.supported_m
        return not (self.gate_m_by_database and supported is not None and f"M{m}" not in supported)

    def m_table(self) -> set[int]:
        codes = {0, 1, 2, 3, 4, 5, 6, 8, 9, 19, 29, 30, 98, 99}
        if any(kind in ("mist", "mql") for kind in self.machine.pumps.values()):
            codes.add(7)
        spindle = self.machine.spindles.get(self.machine.path_spindle[self.ex.ctx.path])
        if spindle is not None and spindle.gears > 1:
            codes |= set(range(41, 41 + spindle.gears))
        return codes

    def is_wait(self, m: int) -> bool:
        return len(self.machine.paths) > 1 and self.p["first_wait_code"] <= m <= self.p["last_wait_code"]

    def m_code(self, m: int, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        key = f"M{m:02d}"
        custom = self.p.get("m_codes", {}).get(key) or self.p.get("m_codes", {}).get(f"M{m}")
        if key in self.p.get("macro_call_codes", {}) and not self._inside_code_call(key):
            self.code_call(key, bw)
            return
        if custom is not None:
            self.custom_m(m, dict(custom), bw)
            return
        if self.is_wait(m):
            self.path_wait(m, bw)
            return
        if m in (3, 4):
            s.spindle.direction = -1 if m == 3 else 1
            s.sys["spindle_commanded"] = True
            self.emit_spindle()
        elif m == 5:
            s.spindle.direction = 0
            s.sys["spindle_commanded"] = True
            self.emit_spindle()
        elif m == 19:
            self.orient(bw)
        elif m == 6:
            self.tool_change(bw)
        elif m in (7, 8):
            kinds = ("flood",) if m == 8 else ("mist", "mql")
            self.coolant({pid: True for pid, kind in self.machine.pumps.items() if kind in kinds}, f"M{m:02d}")
        elif m == 9:
            self.coolant({pid: False for pid in self.machine.pumps}, "M09")
        elif m == 29:
            s.sys["rigid_tap"] = True
            s.sys["rigid_tap_at"] = {"pos": dict(s.pos), "s": s.spindle.s_value, "seq": s.block_seq}
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"rigid_tap": True}))
        elif 41 <= m <= 44:
            s.spindle.gear = m - 40
            self.emit_spindle()
        elif m in (0, 1):
            self.program_stop(m)
        elif m in (2, 30):
            self.count_part(m)
            ex.end_program(f"M{m:02d}")
        elif m == 98:
            self.m98(bw)
        elif m == 99:
            word = bw.take("P")
            ex.ret(self.integer(word) if word is not None else None)
        else:
            self.m_extra(m, bw)

    def m_extra(self, m: int, bw: BlockWords) -> None:
        raise ControllerAlarm("undefined_m_code", f"M{m:02d}")

    def custom_m(self, m: int, spec: dict, bw: BlockWords) -> None:
        """A machine-builder M code from physics.controller.m_codes."""
        ex, s = self.ex, self.s
        kind = spec.get("kind")
        ref = ex.line_ref()
        if kind == "coolant":
            pump = spec.get("pump")
            if pump not in self.machine.pumps:
                raise ControllerAlarm("machine_feature_missing", f"M{m:02d}: pump {pump}")
            self.coolant({pump: bool(spec.get("on", True))}, f"M{m:02d}")
        elif kind == "spindle":
            target = spec.get("spindle") or self.machine.path_spindle[ex.ctx.path]
            direction = {"cw": -1, "ccw": 1, "stop": 0}[spec.get("direction", "stop")]
            word = bw.take("S") or bw.take("P")
            speed = self.spindle_number(word) if word is not None else s.live.s_value
            s.live.direction, s.live.s_value = direction, speed
            omega = direction * speed * 2.0 * math.pi / 60.0
            ex.emit(Command(Kind.SPINDLE, ref, spindle={"id": target, "omega_rad_s": omega, "css_m_s": None,
                                                        "max_rad_s": self.spindle_max(target), "gear": 1,
                                                        "orient_rad": None, "direction": direction}))
        elif kind in ("pallet", "door", "conveyor", "chuck", "output", "brake", "mode"):
            device = {k: v for k, v in spec.items() if k != "kind"}
            cmd_kind = {"pallet": Kind.PALLET, "door": Kind.DOOR, "conveyor": Kind.CONVEYOR,
                        "chuck": Kind.CHUCK}.get(kind, Kind.MODE)
            ex.emit(Command(cmd_kind, ref, modal={"device": {"kind": kind, "code": f"M{m:02d}", **device}}))
        elif kind == "tool_change":
            self.tool_change(bw)
        elif kind == "stop":
            self.program_stop(0)
        elif kind == "ignore":
            ex.note("info", f"M{m:02d}: {spec.get('note', 'no action on this machine')}")
        else:
            raise ControllerAlarm("unsupported_function", f"M{m:02d} kind {kind}")

    def program_stop(self, m: int) -> None:
        ex, s = self.ex, self.s
        if m == 1 and not s.operator.optional_stop:
            return
        ex.flush_comp("program stop")
        ex.emit(Command(Kind.STOP, ex.line_ref(), stop=f"M{m:02d}", message=ex.block.comment if ex.block else ""))
        ex.wait("stop", reason=f"M{m:02d}")

    def count_part(self, m: int) -> None:
        s = self.s
        codes = self.p.get("part_count_codes") or [2, 30]
        if m in codes:
            s.sys["parts"] = int(s.sys.get("parts", 0)) + 1

    def message_stop(self, number: float) -> None:
        ex = self.ex
        text = ex.block.comment if ex.block is not None else ""
        ex.flush_comp("program stop")
        ex.emit(Command(Kind.MESSAGE, ex.line_ref(), stop="#3006", message=text))
        ex.wait("stop", reason="#3006")

    def macro_alarm(self, number: float) -> None:
        ex = self.ex
        n = int(round_half_away(number))
        if not 0 <= n <= self.p["max_macro_alarm"]:
            raise ControllerAlarm("argument_out_of_range", f"#3000 = {n}")
        text = ex.block.comment if ex.block is not None else ""
        info = self.d.alarms.program_alarm(n, text, int(self.p["macro_alarm_base"]))
        raise ControllerAlarm("macro_alarm", text, info.code, info.text)

    def coolant(self, changes: dict[str, bool], code: str, air: bool | None = None,
                nozzles: dict[str, int] | None = None) -> None:
        ex, s = self.ex, self.s
        if not changes and air is None and not nozzles:
            raise ControllerAlarm("machine_feature_missing", f"{code}: no such coolant on this machine")
        s.pumps.update(changes)
        payload = {"pumps": dict(s.pumps), "nozzles": dict(s.nozzles), "code": code}
        if air is not None:
            s.sys["air"] = air
        if "air" in s.sys:
            payload["air"] = bool(s.sys["air"])
        if nozzles:
            s.nozzles.update(nozzles)
            payload["nozzles"] = dict(s.nozzles)
        ex.emit(Command(Kind.COOLANT, ex.line_ref(), coolant=payload))

    def spindle_max(self, spindle_id: str) -> float:
        spindle = self.machine.spindles.get(spindle_id)
        return spindle.max_rad_s if spindle is not None else 0.0

    def emit_spindle(self, orient: float | None = None) -> None:
        ex, s = self.ex, self.s
        spindle_id = self.machine.path_spindle[ex.ctx.path]
        top = self.spindle_max(spindle_id)
        clamp = s.spindle.clamp_rad_s if s.spindle.clamp_rad_s > 0.0 else top
        if top > 0.0:
            clamp = min(clamp, top)
        css = None
        if s.spindle.css:
            css = s.spindle.s_value * (0.3048 if s.units == "inch" else 1.0) / 60.0
            radius = abs(ex.si("X", s.pos.get("X", 0))) if "X" in ex.by_letter else 0.0
            speed = css / radius if radius > 0.0 else clamp
            speed = min(speed, clamp) if clamp > 0.0 else speed
        else:
            speed = s.spindle.s_value * 2.0 * math.pi / 60.0
            if top > 0.0 and speed > top:
                ex.note("warning", f"S{s.spindle.s_value:g} exceeds the spindle maximum; clamped to "
                                   f"{top * 60.0 / (2.0 * math.pi):.0f} rpm")
                speed = top
        omega = s.spindle.direction * speed
        ex.emit(Command(Kind.SPINDLE, ex.line_ref(),
                        spindle={"id": spindle_id, "omega_rad_s": omega, "css_m_s": css, "max_rad_s": clamp,
                                 "gear": s.spindle.gear, "orient_rad": orient, "direction": s.spindle.direction,
                                 "css_axis": "X" if css is not None else None}))

    def orient(self, bw: BlockWords) -> None:
        s = self.s
        angle = 0.0
        word = bw.take("R")
        if word is not None:
            angle = self.rot(word)
        s.spindle.direction = 0
        s.spindle.oriented = True
        self.emit_spindle(orient=angle)

    def tool_change(self, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        number = s.preselect if s.preselect is not None else s.tool
        if number is None or number <= 0:
            raise ControllerAlarm("tool_not_in_magazine", "M06 without a selected tool")
        if number == s.tool and s.preselect is None:
            ex.note("info", f"M06: T{number} is already in the spindle, no tool change")
            return
        number = self.tool_life_select(number)
        ex.flush_comp("tool change")
        slot = self.tool_slot(number)
        payload = {"number": number, "pocket": number, "slot": slot, "length_offset": s.h_number,
                   "radius_offset": s.d_number}
        probe = self.probe_name(number)
        if probe is not None:
            payload["probe"] = probe        # additive: the spindle now carries this probe
        ex.emit(Command(Kind.TOOL_CHANGE, ex.line_ref(), tool=payload, modal=dict(ex.snapshot())))
        s.tool = number
        s.preselect = None
        self.after_tool_change(number)

    def after_tool_change(self, number: int) -> None:
        return None

    def path_wait(self, m: int, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        n_paths = len(self.machine.paths)
        word = bw.take("P")
        if word is not None:
            digits = str(self.integer(word))
            paths = sorted({int(c) - 1 for c in digits})
            if any(p < 0 or p >= n_paths for p in paths) or ex.ctx.path not in paths:
                raise ControllerAlarm("wait_illegal_p", f"M{m} P{digits}")
        else:
            paths = list(range(n_paths))
        ex.flush_comp("path wait")
        wait_id = ex.root.wait_seq
        ex.root.wait_seq += 1
        ex.emit(Command(Kind.SYNC, ex.line_ref(), sync={"reason": "path_wait", "code": f"M{m}", "paths": paths,
                                                        "id": wait_id}))
        ex.wait("path_wait", code=m, paths=paths)

    def m98(self, bw: BlockWords) -> None:
        ex = self.ex
        word = bw.take("P")
        if word is None:
            raise ControllerAlarm("missing_p", "M98 without P")
        p = self.integer(word)
        repeat_word = bw.take("L")
        repeat = self.integer(repeat_word) if repeat_word is not None else 1
        if repeat_word is None and p > 9999:
            repeat, p = divmod(p, 10000)
        ex.request_call("sub", self.find_program(p), None, repeat)

    def find_program(self, number: int) -> int:
        index = self.ex.store.by_number(number)
        if index is None:
            raise ControllerAlarm("program_not_found", f"O{number:04d}")
        return index

    def arguments(self, bw: BlockWords, skip: tuple = ("P", "L")) -> list:
        """Local variables #1-#33 from the block's words (argument specification I and II)."""
        args = new_locals()
        ijk_set, last = 0, 3
        for w in bw.order:
            letter = w.letter
            if letter in skip or letter in bw.used:
                continue
            if letter in ("G", "N", "O"):
                raise ControllerAlarm("g_as_argument" if letter == "G" else "improper_address", letter)
            if letter in ("I", "J", "K"):
                position = "IJK".index(letter)
                if position <= last:
                    ijk_set += 1
                last = position
                if ijk_set > 10:
                    raise ControllerAlarm("improper_address", "more than 10 I, J, K argument sets")
                number = 3 * ijk_set + 1 + position
            else:
                number = ARGUMENTS_I.get(letter)
                if number is None:
                    raise ControllerAlarm("improper_address", f"{letter} cannot be an argument")
            args[number - 1] = self.argument_value(w)
        for w in bw.order:
            bw.used.add(w.letter)
        return args

    def argument_value(self, w: WordValue) -> float:
        if w.point or w.from_variable or self.p.get("pocket_calculator"):
            return w.value
        if w.letter in LINEAR_ARGUMENTS:
            return round_half_away(w.value) * self.linear_increment_units()
        if w.letter in ROTARY_ARGUMENTS:
            return round_half_away(w.value) * self.rotary_increment_deg()
        return w.value

    def g65(self, bw: BlockWords, other_codes: list[str]) -> None:
        if other_codes:
            raise ControllerAlarm("g_as_argument", " ".join(other_codes))
        word = bw.take("P")
        if word is None:
            raise ControllerAlarm("missing_p", "G65 without P")
        program = self.find_program(self.integer(word))
        repeat_word = bw.take("L")
        repeat = self.integer(repeat_word) if repeat_word is not None else 1
        if bw.m:
            raise ControllerAlarm("improper_address", "M code in a G65 block")
        args = self.arguments(bw)
        self.ex.request_call("macro", program, args, repeat)

    def _inside_code_call(self, code: str) -> bool:
        active = self.s.sys.get("code_calls", {})
        depth = active.get(code)
        return depth is not None and depth <= len(self.s.frames)

    def code_call(self, code: str, bw: BlockWords | None) -> None:
        """A G, M or T code the machine builder made a macro/subprogram call (FANUC parameters 6050-6089)."""
        s = self.s
        number = int(self.p["macro_call_codes"][code])
        program = self.find_program(number)
        calls = {k: v for k, v in s.sys.get("code_calls", {}).items() if v <= len(s.frames)}
        calls[code] = len(s.frames) + 1
        s.sys["code_calls"] = calls
        macro = code.startswith("G") or (code.startswith("M") and 9020 <= number <= 9029)
        if macro and bw is not None:
            for g in list(bw.g):
                if g == code:
                    bw.g.remove(g)
            args = self.arguments(bw)
            self.ex.request_call("macro", program, args, 1)
        else:
            self.ex.request_call("sub", program, None, 1)

    # ------------------------------------------------------------------ one-shot codes
    def oneshot(self, code: str, spec: GSpec, bw: BlockWords) -> bool:
        """Execute a group-00 code; True when it moved the axes."""
        ex, s = self.ex, self.s
        kind = spec.kind
        if kind == "dwell":
            self.g04(bw)
        elif kind == "exact_once":
            s.sys["g09"] = True
        elif kind == "data":
            self.g10(bw)
        elif kind == "data_end":
            ex.note("info", "G11: no programmable data input mode is active")
        elif kind in ("ref_return", "ref_return_n"):
            return self.g28(bw, code)
        elif kind == "from_ref":
            return self.g29(bw)
        elif kind == "ref_check":
            return self.g27(bw)
        elif kind == "skip":
            self.g31(bw)
            return True
        elif kind == "probe":
            self.probe_code(code, spec, bw)
            return True
        elif kind == "local":
            self.g52(bw)
        elif kind == "machine":
            return self.g53(bw)
        elif kind == "coord_set":
            self.g92(bw)
        elif kind == "coord_or_clamp":
            self.g50_lathe(bw)
        elif kind == "coord_preset":
            self.g92_1(bw)
        elif kind == "single_direction":
            return self.g60(bw)
        elif kind == "ai_contour":
            self.smoothing_code(code, bw, "Q")
        elif kind == "preview":
            self.smoothing_code(code, bw, "P")
        elif kind == "comp_corner":
            if s.modal.get("7", "G40") == "G40":
                raise ControllerAlarm("improper_g_code", "G39 outside cutter compensation")
            ex.emit_item(hooks.PathItem("command", command=Command(Kind.MODE, ex.line_ref(), modal={"comp_corner": True}),
                                        ref=ex.line_ref(), block_seq=s.block_seq, ready_tick=s.interp_tick,
                                        corner={"code": "G39", "words": {k: v.value for k, v in bw.addr.items()
                                                                         if k in "IJK"}}))
            for letter in "IJK":
                bw.take(letter)
        elif kind == "tcp":
            self.tcp_mode(code, spec, bw)
        elif kind == "ext":
            return self.ext_code(code, spec, bw)
        elif spec.cycle:
            self.cycle_block(code, bw, oneshot=True)
            return True
        else:
            return self.oneshot_extra(code, spec, bw)
        return False

    def oneshot_extra(self, code: str, spec: GSpec, bw: BlockWords) -> bool:
        raise ControllerAlarm("unsupported_function", code)

    def smoothing_code(self, code: str, bw: BlockWords, letter: str) -> None:
        ex, s = self.ex, self.s
        word = bw.take(letter)
        on = word is None or self.integer(word) != 0
        tolerance = 0.0
        for entry in self.machine.smoothing:
            if code in entry["code"]:
                tolerance = entry["tolerance_m"]
        mode = f"{code} {letter}{1 if on else 0}"
        s.smoothing.update({"mode_code": mode if on else "", "tolerance_m": tolerance if on else 0.0})
        ex.emit(Command(Kind.MODE, ex.line_ref(), smoothing=ex.smoothing_spec(), modal={"smoothing": mode}))

    def g04(self, bw: BlockWords) -> None:
        p = bw.take("P")
        x = bw.take("X") or (bw.take("U") if self.T else None)
        if p is not None:
            seconds = self.seconds(p, True)
        elif x is not None:
            seconds = self.seconds(x, False)
        else:
            self.ex.emit(Command(Kind.SYNC, self.ex.line_ref(), sync={"reason": "exact_stop_check"}))
            return
        if seconds < 0.0:
            raise ControllerAlarm("illegal_minus", "negative dwell")
        self.ex.dwell(seconds)

    def g10(self, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        l_word = bw.take("L")
        p_word = bw.take("P")
        if l_word is None and not self.T:
            raise ControllerAlarm("g10_illegal_p", "G10 without L")
        l = self.integer(l_word) if l_word is not None else 0
        p = self.integer(p_word) if p_word is not None else None
        incremental = not self.absolute()
        if l in (2, 20):
            key = self.g10_work_key(l, p)
            row = s.offsets.setdefault(key, [0] * len(ex.axes))
            for i, axis in enumerate(ex.axes):
                word = bw.take(axis.letter)
                inc_word = None
                if self.T:
                    inc_letter = next((k for k, v in INC_LETTERS.items() if v == axis.letter), None)
                    inc_word = bw.take(inc_letter) if inc_letter else None
                if word is not None:
                    value = self.axis_counts(axis, word)
                    row[i] = row[i] + value if incremental else value
                if inc_word is not None:
                    row[i] += self.axis_counts(axis, inc_word)
            ex.resync_program_position()
            return
        if l == 3:
            provider = hooks.DATA_MODES.get("G10L3")
            if provider is None:
                raise ControllerAlarm("unsupported_function", "G10 L3 (tool life management data)")
            provider.begin(ex, p, bw)
            s.sys["data_mode"] = "G10L3"
            return
        if self.T and l in (0, 1, 10, 11):
            self.g10_lathe_tool(l, p, bw)
            return
        if l in (10, 11, 12, 13) or (l == 1 and not self.T):
            if p is None or not 1 <= p <= self.max_offset():
                raise ControllerAlarm("g10_illegal_p", f"G10 L{l} P{p}")
            r = bw.take("R")
            if r is None:
                raise ControllerAlarm("g10_illegal_value", f"G10 L{l} without R")
            value = int(round_half_away(self.lin(r) / self.machine.lin_increment_m))
            column = {10: 0, 11: 1, 12: 2, 13: 3, 1: 0}[l]
            row = s.tool_offsets.setdefault(str(p), [0] * 9)
            row[column] = row[column] + value if incremental else value
            return
        raise ControllerAlarm("g10_illegal_p", f"G10 L{l} is not supported")

    def g10_work_key(self, l: int, p: int | None) -> str:
        if l == 2:
            if p is None or not 0 <= p <= 6:
                raise ControllerAlarm("g10_illegal_p", f"G10 L2 P{p}")
            return ("EXT", "G54", "G55", "G56", "G57", "G58", "G59")[p]
        if p is None or not 1 <= p <= 300:
            raise ControllerAlarm("g10_illegal_p", f"G10 L20 P{p}")
        return f"P{p}"

    def g10_lathe_tool(self, l: int, p: int | None, bw: BlockWords) -> None:
        """G10 P X Z R Q (P 1-64 wear, 10001-10064 geometry; L1 = geometry, L11 = wear)."""
        ex, s = self.ex, self.s
        if p is None:
            raise ControllerAlarm("g10_illegal_p", "G10 without P")
        geometry = l == 1 or l == 10 or p >= 10000
        number = p - 10000 if p >= 10000 else p
        if not 1 <= number <= self.max_offset():
            raise ControllerAlarm("g10_illegal_p", f"G10 P{p}")
        row = s.tool_offsets.setdefault(str(number), [0] * 9)
        columns = {"X": 4 if geometry else 0, "Z": 5 if geometry else 1, "Y": 8 if geometry else 7}
        for letter, column in columns.items():
            axis = ex.by_letter.get(letter)
            word = bw.take(letter)
            inc_letter = next(k for k, v in INC_LETTERS.items() if v == letter)
            inc_word = bw.take(inc_letter) if letter in ex.by_letter or inc_letter in bw.addr else None
            if axis is None and (word is not None or inc_word is not None):
                raise ControllerAlarm("improper_address", letter)
            if word is not None:
                row[column] = self.axis_counts(axis, word) if letter == "X" else self.length_counts(word)
            if inc_word is not None:
                row[column] += self.axis_counts(axis, inc_word) if letter == "X" else self.length_counts(inc_word)
        r = bw.take("R")
        if r is not None:
            row[6 if geometry else 2] = self.length_counts(r)
        c = bw.take("C")
        if c is not None:
            row[6 if geometry else 2] += self.length_counts(c)
        q = bw.take("Q")
        if q is not None:
            row[3] = self.integer(q)

    def length_counts(self, w: WordValue) -> int:
        return int(round_half_away(self.lin(w) / self.machine.lin_increment_m))

    def axis_targets(self, bw: BlockWords, absolute: bool | None = None) -> dict[str, int]:
        """Program-frame targets (counts) of the block's axis words."""
        ex, s = self.ex, self.s
        absolute = self.absolute() if absolute is None else absolute
        out: dict[str, int] = {}
        if not self.T and s.modal.get("17") == "G16":
            out.update(self.polar_targets(bw, absolute))
        for axis in ex.axes:
            letter = axis.letter
            if letter in out:
                continue
            word = bw.take(letter)
            if word is not None:
                value = self.axis_counts(axis, word)
                if axis.kind == "rotary" and axis.continuous and absolute:
                    out[letter] = self.shortest(axis, s.pos.get(letter, 0), value)
                else:
                    out[letter] = value if absolute else s.pos.get(letter, 0) + value
            if self.T:
                inc_letter = next((k for k, v in INC_LETTERS.items() if v == letter), None)
                inc_word = bw.take(inc_letter) if inc_letter else None
                if inc_word is not None:
                    out[letter] = out.get(letter, s.pos.get(letter, 0)) + self.axis_counts(axis, inc_word)
        return out

    def shortest(self, axis, current: int, commanded: int) -> int:
        full = int(round_half_away(2.0 * math.pi / axis.quantum))
        delta = (commanded % full - current % full) % full
        if delta > full // 2:
            delta -= full
        return current + delta

    def polar_targets(self, bw: BlockWords, absolute: bool) -> dict[str, int]:
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        radius_word = bw.take(a)
        angle_word = bw.take(b)
        if radius_word is None and angle_word is None:
            return {}
        axis_a, axis_b = ex.by_letter[a], ex.by_letter[b]
        if absolute:
            origin = (0.0, 0.0)
            current = (ex.si(a, s.pos[a]), ex.si(b, s.pos[b]))
            radius = self.lin(radius_word) if radius_word is not None else math.hypot(*current)
            angle = math.degrees(self.rot(angle_word)) if angle_word is not None else s.polar_angle_deg
        else:
            origin = (ex.si(a, s.pos[a]), ex.si(b, s.pos[b]))
            radius = self.lin(radius_word) if radius_word is not None else 0.0
            angle = s.polar_angle_deg + (math.degrees(self.rot(angle_word)) if angle_word is not None else 0.0)
        s.polar_angle_deg = angle
        x = origin[0] + radius * math.cos(math.radians(angle))
        y = origin[1] + radius * math.sin(math.radians(angle))
        return {a: int(round_half_away(x / axis_a.quantum)), b: int(round_half_away(y / axis_b.quantum))}

    def g28(self, bw: BlockWords, code: str) -> bool:
        ex, s = self.ex, self.s
        if s.cycle is not None and not self.T:
            raise ControllerAlarm("ref_return_in_cycle", code)
        n = 1
        if code == "G30":
            word = bw.take("P")
            n = self.integer(word) if word is not None else 2
            if not 2 <= n <= 4:
                raise ControllerAlarm("illegal_reference_return", f"G30 P{n}")
        mid = self.axis_targets(bw)
        if not mid:
            ex.note("info", f"{code} without axis words: no motion")
            return False
        ex.flush_comp("reference return")
        s.g28_mid = dict(mid)
        ex.move("rapid", mid, ex.feed_spec(True), comp=False, bypass=True)
        target = {}
        for letter in mid:
            axis = ex.by_letter[letter]
            if n == 1:
                target[letter] = axis.home
            else:
                point = self.p.get("reference_points_m", {}).get(n, {})
                target[letter] = int(round_half_away(float(point.get(axis.name, point.get(letter, 0.0))) / axis.quantum))
        ex.machine_move("rapid", target, ex.feed_spec(True), modal={"reference_point": n})
        return True

    def g29(self, bw: BlockWords) -> bool:
        ex, s = self.ex, self.s
        if not s.g28_mid:
            raise ControllerAlarm("illegal_reference_return", "G29 without a G28 intermediate point")
        target = self.axis_targets(bw)
        if not target:
            return False
        mid = {l: s.g28_mid[l] for l in target if l in s.g28_mid}
        ex.move("rapid", mid, ex.feed_spec(True), comp=False, bypass=True)
        ex.move("rapid", target, ex.feed_spec(True), comp=False, bypass=True)
        return True

    def g27(self, bw: BlockWords) -> bool:
        ex, s = self.ex, self.s
        target = self.axis_targets(bw)
        if not target:
            return False
        ex.move("rapid", target, ex.feed_spec(True), comp=False, bypass=True)
        wrong = [l for l in target if s.mach.get(l) != ex.by_letter[l].home]
        if wrong:
            raise ControllerAlarm("ref_check_error", "not at the reference position: " + " ".join(wrong))
        return True

    def g31(self, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        if s.modal.get("7", "G40") != "G40":
            raise ControllerAlarm("g31_in_comp", "G31 in cutter compensation")
        word = bw.take("P")
        signal = self.integer(word) if word is not None else 1
        target = self.axis_targets(bw)
        if not target:
            raise ControllerAlarm("no_data_after_address", "G31 without axis words")
        if not s.feed_set or s.feed_si <= 0.0:
            raise ControllerAlarm("feed_zero", "G31 without F")
        probe = {"skip": signal, "stylus": self.probe_name(s.tool) or "spindle_probe", "expect": "trigger"}
        step = Step("probe", {l: ex.si(l, c) for l, c in target.items()}, probe=probe)
        ex.run_step(step)

    def probe_code(self, code: str, spec: GSpec, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        provider = hooks.find(hooks.PROBES, self.d.name, spec.arg or code)
        if provider is None:
            raise ControllerAlarm("probe_unavailable", code)
        call = self.cycle_call(code, bw, self.oneshot_cycle_data(code, bw))
        ex.queue_steps(provider.start(ex, call), front=True)

    def g52(self, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        absolute = self.absolute()
        for axis in ex.axes:
            word = bw.take(axis.letter)
            if word is not None:
                value = self.axis_counts(axis, word)
                s.g52[axis.letter] = value if absolute else s.g52.get(axis.letter, 0) + value
        ex.resync_program_position()

    def g53(self, bw: BlockWords) -> bool:
        ex = self.ex
        if not self.absolute():
            ex.note("warning", "G53 in incremental mode is ignored (FANUC)")
            for axis in ex.axes:
                bw.take(axis.letter)
            return False
        target = {}
        for axis in ex.axes:
            word = bw.take(axis.letter)
            if word is not None:
                target[axis.letter] = self.axis_counts(axis, word)
        if not target:
            return False
        ex.flush_comp("G53")
        ex.machine_move("rapid", target, ex.feed_spec(True), modal={"machine_coordinates": True})
        return True

    def g92(self, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        word = bw.take("S")
        if word is not None:
            s.spindle.clamp_rad_s = self.spindle_number(word) * 2.0 * math.pi / 60.0
        letters = [a for a in ex.axes if bw.has(a.letter)]
        if not letters:
            return
        if not ex.transform().identity:
            raise ControllerAlarm("improper_g_code", "G92 with scaling, rotation or mirror image active")
        base = ex.offset_counts(letters_tool=False)
        cancel_local = bool(self.p.get("g92_cancels_g52"))
        for axis in letters:
            value = self.axis_counts(axis, bw.take(axis.letter))
            i = ex.letters.index(axis.letter)
            shift = s.mach[axis.letter] - s.applied.get(axis.letter, 0) - value - (base[i] - s.g92.get(axis.letter, 0))
            if cancel_local:
                shift += s.g52.pop(axis.letter, 0)
            s.g92[axis.letter] = shift
            s.pos[axis.letter] = value
        if cancel_local and s.g52:
            s.g52 = {}
            ex.resync_program_position()

    def g50_lathe(self, bw: BlockWords) -> None:
        s = self.s
        word = bw.take("S")
        if word is not None:
            s.spindle.clamp_rad_s = self.spindle_number(word) * 2.0 * math.pi / 60.0
            if s.spindle.direction != 0:
                self.emit_spindle()
        if any(bw.has(a.letter) for a in self.ex.axes):
            self.g92(bw)

    def g92_1(self, bw: BlockWords) -> None:
        ex, s = self.ex, self.s
        for axis in ex.axes:
            word = bw.take(axis.letter)
            if word is not None:
                if word.value != 0.0:
                    raise ControllerAlarm("improper_address", f"{axis.letter}{word.text}: preset needs 0")
                s.g92.pop(axis.letter, None)
                s.g52.pop(axis.letter, None)
        ex.resync_program_position()

    def g60(self, bw: BlockWords) -> bool:
        ex = self.ex
        target = self.axis_targets(bw)
        if not target:
            return False
        overrun = self.p.get("g60_overrun_m", {})
        approach = {}
        for letter, value in target.items():
            axis = ex.by_letter[letter]
            amount = overrun.get(axis.name, overrun.get(letter))
            if amount:
                approach[letter] = value - int(round_half_away(float(amount) / axis.quantum))
        if approach:
            first = dict(target)
            first.update(approach)
            ex.move("rapid", first, ex.feed_spec(True), comp=False, bypass=True)
        ex.move("rapid", target, ex.feed_spec(True), exact=True, comp=False, bypass=True)
        return True

    # ------------------------------------------------------------------ motion
    def motion(self, bw: BlockWords, modal: dict) -> bool:
        ex, s = self.ex, self.s
        if s.cycle is not None and self.cycle_trigger(bw):
            self.cycle_block(s.cycle["code"], bw)
            return True
        code = s.modal.get("1", "G00")
        spec = self.d.gcodes.get(code)
        if spec is not None and spec.cycle:
            repeat = self.box_cycle_mode() and not any(
                (self.d.gcodes.get(g) or GSpec(1, "")).group == 0 for g in bw.g)
            if self.has_motion_words(bw) or repeat:
                self.cycle_block(code, bw)
                return True
            return False
        extra = self.motion_words(code, bw)
        arc_words = [l for l in ("I", "J", "K", "R") if bw.has(l)]
        if not self.has_motion_words(bw) and not (code in ("G02", "G03") and arc_words) \
                and not extra.get("tool_vector"):
            if extra:
                ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"motion_words": extra}))
            return False
        target = self.axis_targets(bw)
        corner = None
        for letter in (",C", ",R", ",A"):
            if bw.has(letter):
                word = bw.take(letter)
                value = math.degrees(self.rot(word)) if letter == ",A" else self.lin(word)
                corner = dict(corner or {}, **{letter: value})
        if corner is not None:
            self.start_corner(code)
        if code == "G00":
            ex.move("rapid", target, ex.feed_spec(True), corner=corner, modal=extra or None)
        elif code == "G01":
            ex.move("linear", target, self.cutting_feed(bw), corner=corner, modal=extra or None)
        elif code in ("G02", "G03"):
            self.arc(bw, target, -1 if code == "G02" else 1, corner)
        elif code in ("G33", "G32"):
            self.thread(bw, target)
        else:
            raise ControllerAlarm("improper_g_code", code)
        return True

    def motion_words(self, code: str, bw: BlockWords) -> dict:
        """Words of a motion block that a provider takes as its own data (hooks.MOTION_WORDS: the I J K
        vector of cutter compensation in G00/G01 blocks, the tool vector of G43.5 ...)."""
        extra: dict = {}
        for provider in hooks.MOTION_WORDS.values():
            extra.update(provider.motion_words(self.ex, code, bw) or {})
        return extra

    def box_cycle_mode(self) -> bool:
        """T series: a G90/G92/G94 box cycle is modal and its data is set (blocks without a move command
        repeat it, FANUC 0i-F / 30i lathe manual 4.1.6)."""
        if not self.T:
            return False
        code = self.s.modal.get("1", "")
        box = self.s.sys.get("box_cycle")
        return code in ("G90", "G92", "G94") and bool(box) and box.get("code") == code

    def start_corner(self, code: str) -> None:
        """,C / ,R / ,A need the next block (look-ahead): the item carries ``corner`` and goes through the
        active compensation filter, or through the "corner" filter started here (it ends itself with
        ``ex.end_comp()`` once the corner is resolved)."""
        if code not in ("G01", "G02", "G03"):
            raise ControllerAlarm("improper_address", f",C/,R with {code}")
        if self.s.comp_state is None:
            if hooks.COMPENSATION.get("corner") is None:
                raise ControllerAlarm("comp_unavailable", "chamfer / corner rounding (,C ,R ,A)")
            self.ex.start_comp("corner")

    def has_motion_words(self, bw: BlockWords) -> bool:
        for letter in self.ex.letters:
            if bw.has(letter):
                return True
        if self.T:
            return any(bw.has(k) for k, v in INC_LETTERS.items() if v in self.ex.by_letter)
        return False

    def cutting_feed(self, bw: BlockWords):
        s = self.s
        if self.feed_mode() == "inverse_time" and "F" not in bw.used:
            raise ControllerAlarm("feed_zero", "G93 needs F in every cutting block")
        if not s.feed_set or s.feed_si <= 0.0:
            raise ControllerAlarm("feed_zero", "cutting feed is zero")
        return self.ex.feed_spec(False)

    def arc(self, bw: BlockWords, target: dict[str, int], direction: int, corner: dict | None = None) -> None:
        ex, s = self.ex, self.s
        plane = self.plane()
        a, b, n = PLANES[plane]
        offsets = {"X": "I", "Y": "J", "Z": "K"}
        centre = None
        radius = None
        r_word = bw.take("R")
        given = {}
        for letter in ("I", "J", "K"):
            word = bw.take(letter)
            if word is not None:
                given[letter] = word
        for letter in given:
            axis_letter = {"I": "X", "J": "Y", "K": "Z"}[letter]
            if axis_letter not in (a, b):
                raise ControllerAlarm("illegal_plane", f"{letter} is not in the {plane} plane")
        if given:
            centre = {}
            for axis_letter in (a, b):
                word = given.get(offsets[axis_letter])
                centre[axis_letter] = self.lin(word) if word is not None else 0.0
        elif r_word is not None:
            radius = self.lin(r_word)
        feed = self.cutting_feed(bw)
        if centre is not None and r_word is not None:
            ex.note("warning", "R and I/J/K in one block: R is used (FANUC)")
            centre = None
            radius = self.lin(r_word)
        if centre is None and radius is None:
            raise ControllerAlarm("arc_no_radius", "G02/G03 without R or I, J, K")
        ex.arc_move(target, direction, centre, radius, feed, corner=corner)

    def thread(self, bw: BlockWords, target: dict[str, int]) -> None:
        """G32 (T) / G33 (M): F is the lead per revolution whatever the feed mode."""
        ex, s = self.ex, self.s
        f_word = bw.addr.get("F")
        if f_word is not None:
            lead = self.feed_value(f_word, "per_rev")[0]
        elif self.feed_mode() == "per_rev" and s.feed_set:
            lead = s.feed_si
        else:
            lead = 0.0
        if lead <= 0.0:
            raise ControllerAlarm("feed_zero", "thread without lead F")
        q = bw.take("Q")
        start = math.radians((q.value if q.point else q.value * 0.001)) if q is not None else 0.0
        # the lead is along the axis with the longer travel (taper threads, FANUC 3.1); the path advances
        # lead * length / major travel per revolution
        deltas = {l: ex.si(l, c - s.pos.get(l, 0)) for l, c in target.items() if ex.by_letter[l].kind == "linear"}
        major = max(deltas, key=lambda l: abs(deltas[l])) if deltas else None
        length = math.sqrt(sum(v * v for v in deltas.values()))
        path_lead = lead * length / abs(deltas[major]) if major is not None and deltas[major] != 0.0 else lead
        feed = ex.feed_spec(False, override=False, feed_si=path_lead, mode="per_rev")
        ex.move("linear", target, feed, modal={"thread": {"lead_m": path_lead, "lead_major_m": lead,
                                                          "major_axis": major, "start_angle_rad": start}})

    # ------------------------------------------------------------------ cycles
    def cycle_trigger(self, bw: BlockWords) -> bool:
        if self.has_motion_words(bw):
            return True
        return bw.has("R")

    def cycle_data(self, code: str, bw: BlockWords) -> dict:
        """Cycle data words in SI (R, Z/drilling axis, Q, P (s), F, I, J, K/L (repeats))."""
        s = self.s
        data = dict(s.cycle["data"]) if (s.cycle is not None and s.cycle.get("code") == code) else {}
        for letter in ("R", "Q", "I", "J", "E"):
            word = bw.take(letter)
            if word is not None:
                data[letter] = self.lin(word)
        word = bw.take("P")
        if word is not None:
            data["P"] = self.seconds(word, True) if not word.point else word.value
        for letter in ("K", "L"):
            word = bw.take(letter)
            if word is not None:
                if letter in self.repeat_letters:
                    data[letter] = float(self.integer(word))
                else:
                    data[letter] = self.lin(word)          # Haas: K is a length (G73/G83 minimum peck)
        if s.feed_set:
            data["F"] = s.feed_si
        if s.cycle is not None:
            s.cycle["data"] = data
        return data

    def cycle_call(self, code: str, bw: BlockWords, data: dict) -> CycleCall:
        ex, s = self.ex, self.s
        position = {l: ex.si(l, c) for l, c in s.pos.items()}
        current = dict(position)
        raw = {k: v.value for k, v in bw.addr.items()}
        call = CycleCall(code=code, dialect=self.d.name, plane=self.plane(), absolute=self.absolute(),
                         position=position, current=current, initial_level=s.cycle_initial,
                         return_to_initial=s.modal.get("10", "G98") == "G98", data=data, words=raw,
                         feed_mode=self.feed_mode(), units=s.units)
        call.extra["word_values"] = {k: [v.value, v.point, v.from_variable] for k, v in bw.addr.items()}
        call.extra["comment"] = ex.block.comment if ex.block is not None else ""
        return call

    def drilling_axis(self, code: str) -> str:
        """The axis a hole cycle drills along: the plane normal (FANUC parameter FXY = 1); on lathes Z for
        the face cycles (G83-G85) and X for the side cycles (G87-G89)."""
        if self.T:
            return "X" if code in ("G87", "G88", "G89") else "Z"
        return PLANES[self.plane()][2]

    def cycle_block(self, code: str, bw: BlockWords, oneshot: bool = False) -> None:
        """One cycle execution: K/L repetitions of a canned cycle (each at the next incremental position
        in G91, at the same position in G90), or one one-shot cycle (G70-G76, G12/G13/G47/G150 ...)."""
        ex, s = self.ex, self.s
        provider = hooks.find(hooks.CYCLES, self.d.name, code)
        if provider is None:
            raise ControllerAlarm("cycle_unavailable", code)
        if oneshot or s.cycle is None:
            data = self.oneshot_cycle_data(code, bw)
            spec = self.d.gcodes.get(code)
            if spec is not None and spec.group == 1:
                previous = s.sys.get("box_cycle")
                merged = dict(previous["data"]) if previous and previous.get("code") == code else {}
                merged.update(data)
                s.sys["box_cycle"] = {"code": code, "data": merged}
                data = dict(merged)
            call = self.cycle_call(code, bw, data)
            ex.queue_steps(provider.expand(ex, call))
            return
        data = self.cycle_data(code, bw)
        absolute = self.absolute()
        drill = self.drilling_axis(code)
        if drill in ex.by_letter:
            word = bw.take(drill)
            if word is not None:
                data[drill] = ex.si(drill, self.axis_counts(ex.by_letter[drill], word))
                data["absolute"] = 1.0 if absolute else 0.0
        repeats = int(next((data[l] for l in self.repeat_letters if l in data), 1.0))
        steps_words = {}
        for axis in ex.axes:
            if axis.letter == drill:
                continue
            word = bw.take(axis.letter)
            if word is not None:
                steps_words[axis.letter] = self.axis_counts(axis, word)
        s.cycle["data"] = data
        current = {l: c for l, c in s.pos.items()}
        for k in range(max(repeats, 0)):
            position = dict(current)
            for letter, value in steps_words.items():
                axis = ex.by_letter[letter]
                if absolute:
                    position[letter] = self.shortest(axis, current[letter], value) \
                        if (axis.kind == "rotary" and axis.continuous) else value
                else:
                    position[letter] = current[letter] + value
            call = self.cycle_call(code, bw, dict(data))
            call.repeat_index = k
            call.current = {l: ex.si(l, c) for l, c in current.items()}
            call.position = {l: ex.si(l, c) for l, c in position.items()}
            call.extra["drilling_axis"] = drill
            ex.queue_steps(provider.expand(ex, call))
            current = position

    def oneshot_cycle_data(self, code: str, bw: BlockWords) -> dict:
        """One-shot cycles (lathe G70-G76, Haas G12/G13/G47/G150 ...): every remaining word in SI where it
        is a length, raw otherwise; the provider reads ``call.extra['word_values']`` for exact rules."""
        ex = self.ex
        data = {}
        for letter, word in list(bw.addr.items()):
            if letter in bw.used:
                continue
            axis = ex.by_letter.get(letter)
            if axis is not None:
                data[letter] = ex.si(letter, self.axis_counts(axis, word))
            elif letter in "IJKRQEUVW":
                data[letter] = self.lin(word)
            elif letter == "F":
                data["F"] = self.feed_value(word, self.feed_mode())[0]
            else:
                data[letter] = word.value
            bw.used.add(letter)
        if "F" not in data and self.s.feed_set:
            data["F"] = self.s.feed_si
        return data


# ====================================================================== dialect builder

def fanuc_params(settings, model: str = "controller-fanuc") -> dict:
    from ...db.models import params as model_params

    values = dict(model_params(model))
    timing = model_params("controller-block-time")
    values.update(timing)
    resolved = {
        "pocket_calculator": False, "nat": False, "macro_nesting_max": int(values["macro_nesting_levels"]),
        "sub_nesting_max": int(values["sub_nesting_levels"]),
        "max_m_codes_per_block": int(values["max_m_codes_per_block"]),
        "first_wait_code": int(values.get("first_wait_code", 100)),
        "last_wait_code": int(values.get("last_wait_code", 999)),
        "macro_alarm_base": int(values["macro_alarm_base"]), "max_macro_alarm": int(values["max_macro_alarm"]),
        "arc_tolerance_m": float(values["arc_tolerance_m"]),
        "same_group_alarm": bool(values["same_group_alarm"]),
        "feed_mode_change_clears_f": bool(values["feed_mode_change_clears_f"]),
        "g92_cancels_g52": bool(values["g92_cancels_g52"]),
        "lathe_offsets_in_diameter": bool(values["lathe_offsets_in_diameter"]),
        "g37_feed_m_s": float(values["g37_feed_m_s"]), "g37_approach_m": float(values["g37_approach_m"]),
        "g37_tolerance_m": float(values["g37_tolerance_m"]),
        "macro_statement_s": float(values["macro_statement_s"]), "macro_term_s": float(values["macro_term_s"]),
        "call_s": float(values["call_s"]), "goto_scan_s": float(values["goto_scan_s"]),
        "min_block_s": float(values["min_block_s"]), "horizon_s": float(values["horizon_s"]),
        "offsets_diameter": bool(values.get("offsets_diameter", False)),
        "m_codes": {}, "macro_call_codes": {}, "reference_points_m": {}, "g60_overrun_m": {}, "power_on": {},
        "interface_inputs": {}, "clock_start": "2026-01-05T07:00:00", "shared_common_100": 0,
        "shared_common_500": 0, "tool_length_axis": "A", "power_on_units": "mm",
    }
    if settings is not None:
        mapping = {"wait_m_range": None}
        for name, value in vars(settings).items():
            if value is None or name in mapping:
                continue
            if isinstance(value, dict) and not value:
                continue
            key = {"macro_nesting_max": "macro_nesting_max", "sub_nesting_max": "sub_nesting_max",
                   "same_group": "same_group_alarm", "haas_offsets_diameter": "offsets_diameter"}.get(name, name)
            if name == "same_group":
                value = value == "alarm"
            resolved[key] = value
        if settings.wait_m_range:
            resolved["first_wait_code"], resolved["last_wait_code"] = (int(v) for v in settings.wait_m_range)
    return resolved


def build_fanuc(machine, settings, controller_db, db_id: str = "") -> Dialect:
    variant = machine.variant
    gcodes = dict(T_GCODES if variant == "T" else M_GCODES)
    gcodes.update(hooks.extra_gcodes("fanuc", variant))
    power_on = dict(T_POWER_ON if variant == "T" else M_POWER_ON)
    words = (controller_db or {}).get("words") or {}
    supported_g = word_set(words.get("G"), "G") if words else None
    supported_m = word_set(words.get("M"), "M") if words else None
    alarms = AlarmTable("fanuc", ALARM_CODES, ALARM_TEXTS, controller_db, db_id)
    letters = frozenset(LETTERS | {",C", ",R"})
    lexer = LexerRules(letters=frozenset(LETTERS), max_digits=dict(MAX_DIGITS), max_o_digits=4,
                       comma_letters=frozenset("CRA"))
    return Dialect("fanuc", variant, lexer, gcodes, power_on, alarms, FanucVars(variant), FanucHandler,
                   fanuc_params(settings), controller_db, db_id, supported_g, supported_m, letters)


register_dialect("fanuc", build_fanuc)

from .. import measure  # noqa: E402,F401  (registers the default G36/G37 tool measurement)

__all__ = ["FanucHandler", "BlockWords", "WordValue", "build_fanuc", "fanuc_params", "INC_LETTERS"]
