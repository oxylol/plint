"""The sanity checks' own modal scan of a Fanuc/Haas word-address program.

It executes the program statically the way the control would (modal groups of ISO 6983-1 as the
Fanuc and Haas mill and lathe manuals define them), follows M97/M98/M99 subprograms, expands
canned cycles into their moves, and emits :class:`cnctwin.controller.ir.Command` objects in
TIP space (tool tip in program coordinates of the active work offset, the *intended* position:
compensations are assumed right; the rules check them separately) with a modal snapshot each.
Machine-axis targets are added where work offset and length offset are known.

What it does not evaluate (macro expressions and statements, G65/G66 macro calls, rotation,
scaling, polar coordinates, Haas G150/G12/G13/G70-G72 pattern cycles, lathe multiple repetitive
cycles' passes) is listed in :attr:`ScanResult.notes`, so the rules can report it.
Block-delete (``/``) blocks are executed (block delete switch off).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from ..controller.ir import ArcSpec, Command, Feed, FeedMode, Kind, LineRef, Space
from ..units import FOOT_M, INCH_M
from .context import SanityContext, normalise_offset
from .program import Line, code_text, lex

SUPPORTED_DIALECTS = ("fanuc", "haas")
NAN3 = (math.nan, math.nan, math.nan)
TWO_PI = 2.0 * math.pi
RPM_TO_RAD_S = TWO_PI / 60.0

CYCLES_MILL = ("G73", "G74", "G76", "G81", "G82", "G83", "G84", "G84.2", "G84.3", "G85", "G86", "G87", "G88", "G89")
TAP_CYCLES = ("G84", "G74", "G84.2", "G84.3")
WORK_OFFSETS = tuple(f"G{n}" for n in range(54, 60))
HAAS_EXTRA_OFFSETS = tuple(f"G{n}" for n in range(110, 130))
# Coolant codes per dialect: code -> (what it needs, off code)
COOLANT_ON = {
    "fanuc": {"M7": "mist", "M8": "flood"},
    "haas": {"M7": "shower", "M8": "flood", "M88": "through_spindle", "M73": "through_tool_air", "M83": "air_blast"},
}
COOLANT_OFF = {
    "fanuc": {"M9": ("M7", "M8")},
    "haas": {"M9": ("M7", "M8"), "M89": ("M88",), "M74": ("M73",), "M84": ("M83",)},
}
UNEVALUATED_G = {
    "G15": "polar coordinates off", "G16": "polar coordinates", "G50": "scaling off", "G51": "scaling",
    "G50.1": "programmable mirror off", "G51.1": "programmable mirror", "G68": "coordinate rotation",
    "G69": "coordinate rotation off", "G66": "modal macro call", "G67": "modal macro call off",
    "G43.4": "tool centre point control", "G43.5": "tool centre point control", "G234": "tool centre point control",
    "G254": "dynamic work offset", "G255": "dynamic work offset off", "G68.2": "tilted working plane",
    "G101": "mirror image", "G100": "mirror image off",
}
HAAS_PATTERN_G = {"G12": "circular pocket", "G13": "circular pocket", "G150": "general pocket",
                  "G70": "bolt hole circle", "G71": "bolt hole arc", "G72": "bolt holes along an angle",
                  "G47": "text engraving"}
LATHE_CYCLE_G = {"G70": "finishing cycle", "G71": "O.D./I.D. stock removal", "G72": "end face stock removal",
                 "G73": "pattern repeating", "G74": "end face grooving / peck drilling", "G75": "O.D./I.D. grooving",
                 "G76": "threading cycle", "G90": "O.D./I.D. turning cycle", "G92": "threading cycle",
                 "G94": "end facing cycle"}


@dataclass
class Hole:
    """One hole a canned cycle (or a drill's plunge) makes, in program coordinates (SI)."""

    command: int
    line: int
    program: str
    tool: int | None
    code: str
    x_m: float
    y_m: float
    z_bottom_m: float
    z_r_m: float
    z_initial_m: float
    retract_z_m: float
    work_offset: str
    section: int


@dataclass
class Section:
    """The part of the program run with one tool (from one tool change to the next)."""

    index: int
    tool: int | None
    command: int                 # index of the tool-change command (-1: before any tool change)
    line: int
    restart_line: int
    program: str


@dataclass
class Note:
    line: int
    program: str
    kind: str                    # macro | flow | call | unevaluated | unknown_code | end
    message: str


@dataclass
class ScanResult:
    commands: list[Command]
    tip_start: np.ndarray
    tip_end: np.ndarray
    holes: list[Hole] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    g28: list[dict] = field(default_factory=list)            # {command, line, distance, axes, intermediate_W_m}
    offsets_set: dict[str, tuple[int, int]] = field(default_factory=dict)      # work offset -> (line, command) by G10
    length_measured: dict[int, tuple[int, int]] = field(default_factory=dict)  # H -> (line, command) by G10 L10/G37
    radius_set: dict[int, tuple[int, int]] = field(default_factory=dict)       # D -> (line, command) by G10 L12
    macro_calls: list[tuple[int, int]] = field(default_factory=list)   # (command index at call, line)
    end_code: str | None = None
    end_line: int = 0
    end_state: dict = field(default_factory=dict)
    lines: dict[str, list[Line]] = field(default_factory=dict)


@dataclass
class _State:
    units: str = "mm"
    distance: str = "absolute"
    plane: str = "XY"
    motion: str = "G0"
    feed_mode: str = FeedMode.PER_MIN.value
    feed: float | None = None
    feed_line: int = 0
    feed_tool: int | None = None
    feed_section: int = -1
    spindle_rpm: float = 0.0
    spindle_dir: str = "M5"
    spindle_line: int = 0
    css_m_s: float | None = None
    max_rpm: float | None = None
    coolant: tuple[str, ...] = ()
    tool: int | None = None
    tool_line: int = 0
    next_tool: int | None = None
    length_comp: str | None = None
    length_offset: int | None = None
    length_line: int = 0
    length_section: int = -1
    radius_comp: str = "G40"
    radius_offset: int | None = None
    work_offset: str = "G54"
    work_offset_line: int = 0
    cycle: str | None = None
    cycle_retract: str = "G98"
    cycle_r: float | None = None
    cycle_z: float | None = None
    cycle_initial_z: float | None = None
    cycle_line: int = 0
    rigid_tap: bool = False
    section: int = 0
    call_depth: int = 0

    def snapshot(self, machine: str, dialect: str) -> dict:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["coolant"] = list(self.coolant)
        data["length_comp"] = self.length_comp is not None
        data["length_comp_code"] = self.length_comp
        data["spindle_on"] = self.spindle_dir in ("M3", "M4") and (self.spindle_rpm > 0 or self.css_m_s is not None)
        rpm = self.spindle_rpm if self.css_m_s is None else (self.max_rpm or self.spindle_rpm)
        data["spindle_rad_s"] = spindle_omega(self.spindle_dir, rpm)
        data["machine"] = machine
        data["dialect"] = dialect
        return data


def spindle_omega(direction: str, rpm: float) -> float:
    """Signed spindle speed about +z_T: M03 turns negative (docs/architecture.md 4)."""
    if direction == "M3":
        return -abs(rpm) * RPM_TO_RAD_S
    if direction == "M4":
        return abs(rpm) * RPM_TO_RAD_S
    return 0.0


@dataclass
class _Frame:
    program: str
    index: int
    return_to: tuple[str, int] | None = None
    repeat_left: int = 0
    start_index: int = 0


class Scanner:
    """One static execution of a program (see the module docstring)."""

    def __init__(self, text: str, name: str, ctx: SanityContext, *, machine_kind: str | None = None,
                 subprograms: dict[str, str] | None = None):
        self.ctx = ctx
        self.name = name or "program"
        self.dialect = ctx.dialect
        self.texts: dict[str, list[Line]] = {self.name: lex(text)}
        for key, body in (subprograms or {}).items():
            self.texts[str(key)] = lex(body)
        self.machine = machine_kind or _decide_kind(ctx, self.texts[self.name])
        self.labels = self._labels()
        self.state = _State()
        self.state.units = str(ctx.p("default_units"))
        self.state.tool = ctx.initial_tool
        self.state.next_tool = None
        if self.machine == "lathe":
            self.state.plane = "ZX"
            self.state.feed_mode = FeedMode.PER_REV.value
        self.work_offsets = dict(ctx.work_offsets_m)
        self.length_values = {n: t.length_offset_m for n, t in ctx.tools.items()}
        self.pos = np.array(NAN3)
        self.shift = np.zeros(3)
        self.result = ScanResult([], np.zeros((0, 3)), np.zeros((0, 3)), lines=self.texts)
        self._starts: list[tuple] = []
        self._ends: list[tuple] = []
        self._snapshot_key = None
        self._snapshot = None
        self._lathe_offset: int | None = None
        self._g10_note = False
        self._flow_noted = False
        self.result.sections.append(Section(0, self.state.tool, -1, 0, 0, self.name))
        if self.machine != "lathe":
            self.pos = self._home_W(self.pos, "XYZ")

    # ------------------------------------------------------------------ driver
    def run(self) -> ScanResult:
        frames = [_Frame(self.name, 0)]
        seen_code = False
        max_depth = int(self.ctx.p("max_call_depth"))
        steps = 0
        step_limit = 50 * sum(len(lines) for lines in self.texts.values()) + 10000
        while frames:
            steps += 1
            if steps > step_limit:
                self.note(0, "flow", "the scan stopped: the program loops (M99 without an end, or a repeat count "
                                     "the scan cannot bound)")
                break
            frame = frames[-1]
            lines = self.texts[frame.program]
            if frame.index >= len(lines):
                if not self._end_of_file(frames):
                    break
                continue
            line = lines[frame.index]
            frame.index += 1
            if line.percent:
                if seen_code and len(frames) == 1:
                    break
                continue
            if line.program_number is not None and seen_code and frame.index - 1 > frame.start_index:
                if len(frames) == 1:
                    break
                self.note(line.number, "call", f"subprogram {frame.program} runs into {line.program_number} "
                                               "without M99", frame.program)
                frames.pop()
                continue
            if line.words or line.macro:
                seen_code = True
            action = self.block(line, frame.program, len(frames) - 1)
            if action is None:
                continue
            kind, payload = action
            if kind == "end":
                break
            if kind == "call":
                target, count = payload
                if len(frames) > max_depth:
                    self.note(line.number, "call", f"subprogram nesting deeper than {max_depth} levels is not "
                                                   "followed", frame.program)
                    continue
                location = self.labels.get(target)
                if location is None:
                    self.note(line.number, "call", f"subprogram {target} is not available to the scan: its blocks "
                                                   "are not checked", frame.program)
                    continue
                program, index = location
                if count <= 0:
                    continue
                frames.append(_Frame(program, index, (frame.program, frame.index), count - 1, index))
                self.state.call_depth = len(frames) - 1
            elif kind == "return":
                if len(frames) == 1:
                    self.result.end_code = "M99"
                    self.result.end_line = line.number
                    break
                if frame.repeat_left > 0:
                    frame.repeat_left -= 1
                    frame.index = frame.start_index
                    continue
                frames.pop()
                self.state.call_depth = len(frames) - 1
        self.result.tip_start = np.array(self._starts, dtype=float).reshape(-1, 3)
        self.result.tip_end = np.array(self._ends, dtype=float).reshape(-1, 3)
        self.result.end_state = self.state.snapshot(self.machine, self.dialect)
        return self.result

    def _end_of_file(self, frames: list[_Frame]) -> bool:
        frame = frames[-1]
        if len(frames) == 1:
            return False
        self.note(0, "call", f"subprogram {frame.program} ends without M99", frame.program)
        frames.pop()
        return True

    def _labels(self) -> dict[str, tuple[str, int]]:
        labels: dict[str, tuple[str, int]] = {}
        for program, lines in self.texts.items():
            for index, line in enumerate(lines):
                if line.program_number is not None:
                    labels.setdefault(line.program_number, (program, index))
                if program == self.name and line.n_number is not None:
                    labels.setdefault(f"N{line.n_number}", (program, index))
            key = program.upper()
            if key.startswith("O") and key[1:].isdigit():
                labels.setdefault(f"O{int(key[1:])}", (program, 0))
        return labels

    def note(self, line: int, kind: str, message: str, program: str | None = None) -> None:
        self.result.notes.append(Note(line, program or self.name, kind, message))

    # ------------------------------------------------------------------ one block
    def block(self, line: Line, program: str, depth: int):
        if line.macro:
            if any(w.value is None for w in line.words) or not line.words:
                kind = "flow" if not line.words else "macro"
                if kind == "flow" and not self._flow_noted:
                    self._flow_noted = True
                    self.note(line.number, "flow", "macro statements (variables, IF/GOTO/WHILE) are not evaluated: "
                                                   "the scan reads the program as written, top to bottom", program)
                elif kind == "macro":
                    self.note(line.number, "macro", "address with a macro expression: its value is unknown to the "
                                                    "scan", program)
            if not line.words:
                return None
        words: dict[str, float | None] = {}
        g_codes: list[str] = []
        m_codes: list[str] = []
        for word in line.words:
            if word.letter == "G":
                g_codes.append(word.code)
            elif word.letter == "M":
                m_codes.append(word.code)
            else:
                words.setdefault(word.letter, None if word.value is None else self._scaled(word))
        raw: dict = {}
        for word in line.words:
            if word.letter not in "GM":
                raw.setdefault(word.letter, word)
        ref = LineRef(program, line.number, line.text, line.n_number)
        self.line_ref = ref
        self.raw = raw
        if "G20" in g_codes:
            self.state.units = "inch"
            words = {k: (None if raw[k].value is None else self._scaled(raw[k])) for k in words}
        if "G21" in g_codes:
            self.state.units = "mm"
            words = {k: (None if raw[k].value is None else self._scaled(raw[k])) for k in words}
        if self.machine == "lathe":
            return self._lathe_block(line, program, ref, words, g_codes, m_codes)
        return self._mill_block(line, program, ref, words, g_codes, m_codes)

    def _scaled(self, word) -> float:
        """A length word in metres (Fanuc: integers are least input increments)."""
        value = float(word.value)
        if self.dialect == "fanuc" and not word.has_point and word.letter in "XYZUVWIJKRQ":
            step = self.ctx.least_increment_m if self.state.units == "mm" else float(self.ctx.p("fanuc_inch_increment_m"))
            return value * step
        return value * self._unit()

    def _unit(self) -> float:
        return 1e-3 if self.state.units == "mm" else INCH_M

    # ------------------------------------------------------------------ mill
    def _mill_block(self, line, program, ref, words, g_codes, m_codes):
        state = self.state
        number = line.number
        one_shot = None
        motion_code = None
        for code in g_codes:
            if code in ("G0", "G1", "G2", "G3"):
                motion_code = code
            elif code in CYCLES_MILL:
                motion_code = code
            elif code == "G80":
                motion_code = "G80"
            elif code in ("G17", "G18", "G19"):
                state.plane = {"G17": "XY", "G18": "ZX", "G19": "YZ"}[code]
            elif code in ("G90", "G91"):
                state.distance = "absolute" if code == "G90" else "incremental"
            elif code in ("G93", "G94", "G95"):
                state.feed_mode = {"G93": FeedMode.INVERSE_TIME, "G94": FeedMode.PER_MIN,
                                   "G95": FeedMode.PER_REV}[code].value
            elif code in ("G40", "G41", "G42"):
                state.radius_comp = code
                if code != "G40":
                    d = self.raw.get("D")
                    state.radius_offset = int(d.value) if d is not None and d.value is not None else None
            elif code in ("G43", "G44"):
                h = self.raw.get("H")
                self._set_length(code, int(h.value) if h is not None and h.value is not None else None, number)
            elif code == "G49":
                self._set_length(None, None, number)
            elif code in WORK_OFFSETS or code in HAAS_EXTRA_OFFSETS:
                self._set_work_offset(code, number)
            elif code in ("G54.1", "G154"):
                p = self.raw.get("P")
                self._set_work_offset(f"{code}P{int(p.value)}" if p is not None and p.value is not None else code,
                                      number)
            elif code in ("G98", "G99"):
                state.cycle_retract = code
            elif code in ("G4", "G9", "G10", "G28", "G29", "G30", "G31", "G37", "G52", "G53", "G65", "G92"):
                one_shot = code
            elif code in ("G20", "G21", "G61", "G64", "G187", "G103", "G5.1", "G8", "G5", "G7.1", "G96", "G97"):
                if code in ("G61", "G64", "G187", "G103", "G5.1", "G8"):
                    self.emit(Kind.MODE, ref, modal_change=True)
            elif code in UNEVALUATED_G:
                self.note(number, "unevaluated", f"{code} ({UNEVALUATED_G[code]}) is not evaluated: positions after "
                                                 "it may be wrong", program)
            elif self.dialect == "haas" and code in HAAS_PATTERN_G:
                self.note(number, "unevaluated", f"{code} ({HAAS_PATTERN_G[code]}) is not expanded by the scan", program)
                motion_code = motion_code or "skip"
            else:
                self.note(number, "unknown_code", f"{code} is not interpreted by the scan", program)
        if "F" in words and words["F"] is not None:
            self._set_feed(self.raw["F"].value, number)
        if "S" in words and self.raw["S"].value is not None:
            state.spindle_rpm = abs(float(self.raw["S"].value))
            if state.spindle_dir in ("M3", "M4"):
                self.emit(Kind.SPINDLE, ref, spindle=self._spindle_payload())
        t_word = self.raw.get("T")
        if t_word is not None and t_word.value is not None and one_shot != "G10":
            state.next_tool = int(t_word.value)
            self.emit(Kind.MODE, ref, tool={"number": state.next_tool, "preselect": True})
        pre = [m for m in m_codes if m in ("M3", "M4", "M6", "M19", "M29") or m in COOLANT_ON.get(self.dialect, {})]
        post = [m for m in m_codes if m not in pre]
        for code in pre:
            self._m_code(code, ref, number, program)
        if one_shot is not None:
            result = self._one_shot(one_shot, ref, words, number, program)
            if result is not None:
                return result
        elif motion_code == "G80":
            self._cancel_cycle()
            if any(k in words for k in "XYZ"):
                self._linear(ref, words, rapid=state.motion == "G0")
        elif motion_code == "skip":
            pass
        else:
            if motion_code in ("G0", "G1", "G2", "G3"):
                state.motion = motion_code
                if state.cycle is not None:
                    self._cancel_cycle()
            if motion_code in CYCLES_MILL:
                self._start_cycle(motion_code, words, number)
            if state.cycle is not None and (motion_code in CYCLES_MILL or any(k in words for k in "XY")
                                            or ("Z" in words and motion_code is None) or "R" in words):
                self._cycle_holes(ref, words, number, program)
            elif any(k in words for k in "XYZ") or (state.motion in ("G2", "G3") and any(k in words for k in "IJKR")):
                if state.motion in ("G2", "G3"):
                    self._arc(ref, words, state.motion)
                else:
                    self._linear(ref, words, rapid=state.motion == "G0")
        for code in post:
            result = self._m_code(code, ref, number, program)
            if result is not None:
                return result
        return None

    def _set_feed(self, value: float, number: int) -> None:
        state = self.state
        unit = self._unit()
        if state.feed_mode == FeedMode.PER_MIN.value:
            state.feed = float(value) * unit / 60.0
        elif state.feed_mode == FeedMode.PER_REV.value:
            state.feed = float(value) * unit
        else:
            state.feed = float(value) / 60.0
        state.feed_line = number
        state.feed_tool = state.tool
        state.feed_section = state.section

    def _spindle_payload(self) -> dict:
        state = self.state
        return {"id": "main", "omega_rad_s": spindle_omega(state.spindle_dir, state.spindle_rpm),
                "css_m_s": state.css_m_s, "max_rad_s": None if state.max_rpm is None else state.max_rpm * RPM_TO_RAD_S}

    def _m_code(self, code: str, ref, number: int, program: str):
        state = self.state
        dialect = self.dialect
        if code in ("M3", "M4", "M5"):
            state.spindle_dir = code
            state.spindle_line = number
            self.emit(Kind.SPINDLE, ref, spindle=self._spindle_payload())
        elif code == "M6":
            self._tool_change(ref, number, program)
        elif code == "M19":
            state.spindle_dir = "M5"
            self.emit(Kind.SPINDLE, ref, spindle={"id": "main", "omega_rad_s": 0.0, "orient_rad": 0.0})
        elif code == "M29":
            state.rigid_tap = True
        elif code in COOLANT_ON.get(dialect, {}):
            if code not in state.coolant:
                state.coolant = tuple(sorted(state.coolant + (code,)))
            self.emit(Kind.COOLANT, ref, coolant={"code": code, "needs": COOLANT_ON[dialect][code], "on": True})
        elif code in COOLANT_OFF.get(dialect, {}):
            off = COOLANT_OFF[dialect][code]
            state.coolant = tuple(c for c in state.coolant if c not in off)
            self.emit(Kind.COOLANT, ref, coolant={"code": code, "on": False})
        elif dialect == "haas" and code in ("M34", "M35"):
            self.emit(Kind.COOLANT, ref, coolant={"code": code, "needs": "programmable_nozzle", "on": None})
        elif code in ("M0", "M1"):
            self.emit(Kind.STOP, ref, stop=f"M0{code[1:]}")
        elif code in ("M2", "M30"):
            self.emit(Kind.STOP, ref, stop="M02" if code == "M2" else "M30")
            self.result.end_code = code
            self.result.end_line = number
            return ("end", None)
        elif code == "M98":
            return ("call", self._call_target(program))
        elif code == "M97" and dialect == "haas":
            p = self.raw.get("P")
            count = self._repeat_count()
            if p is None or p.value is None:
                self.note(number, "call", "M97 without P", program)
                return None
            return ("call", (f"N{int(p.value)}", count))
        elif code == "M99":
            return ("return", None)
        elif dialect == "haas" and code in ("M31", "M33"):
            self.emit(Kind.CONVEYOR, ref)
        elif dialect == "haas" and code in ("M80", "M81"):
            self.emit(Kind.DOOR, ref)
        elif dialect == "haas" and code == "M50":
            self.emit(Kind.PALLET, ref)
        elif dialect == "haas" and code in ("M10", "M11", "M12", "M13", "M41", "M42", "M70", "M71", "M90", "M91",
                                            "M95", "M104", "M105", "M138", "M139", "M158", "M159", "M16"):
            if code == "M16":
                self._tool_change(ref, number, program)
            else:
                self.emit(Kind.MODE, ref)
        else:
            self.note(number, "unknown_code", f"{code} is not interpreted by the scan", program)
        return None

    def _call_target(self, program: str) -> tuple[str, int]:
        p = self.raw.get("P")
        count = self._repeat_count()
        if p is None or p.value is None:
            return ("O?", 0)
        value = int(p.value)
        if self.dialect == "fanuc" and value > 9999 and "L" not in self.raw and "K" not in self.raw:
            count = value // 10000
            value = value % 10000
        return (f"O{value}", count)

    def _repeat_count(self) -> int:
        for letter in ("L", "K"):
            word = self.raw.get(letter)
            if word is not None and word.value is not None:
                return int(word.value)
        return 1

    def _tool_change(self, ref, number: int, program: str) -> None:
        state = self.state
        new_tool = state.next_tool
        state.tool = new_tool
        state.tool_line = number
        state.spindle_dir = "M5"
        state.coolant = ()
        state.section += 1
        index = self.emit(Kind.TOOL_CHANGE, ref, tool={"number": new_tool, "pocket": new_tool,
                                                       "length_offset": state.length_offset})
        self.pos = self._home_W(self.pos, "Z")
        self._ends[-1] = tuple(self.pos)
        restart = self._restart_line(program, number)
        self.result.sections.append(Section(state.section, new_tool, index, number, restart, program))

    def _restart_line(self, program: str, number: int) -> int:
        """The N-numbered line an operator restarts this tool at: the nearest one at or before the
        tool change among the lines since the previous motion (else the tool-change line)."""
        lines = self.texts[program]
        last_motion_line = 0
        for command in reversed(self.result.commands):
            if command.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC) and command.ref.program == program:
                last_motion_line = command.ref.line
                break
        for index in range(number - 1, max(last_motion_line, 0) - 1, -1):
            if index < 0:
                break
            candidate = lines[index]
            if candidate.number <= last_motion_line:
                break
            if candidate.n_number is not None:
                return candidate.number
        return number

    # ------------------------------------------------------------------ one-shot G codes
    def _one_shot(self, code, ref, words, number, program):
        state = self.state
        if code == "G4":
            seconds = self.raw.get("P")
            dwell = 0.0
            if seconds is not None and seconds.value is not None:
                dwell = float(seconds.value) / (1000.0 if not seconds.has_point else 1.0)
            elif self.raw.get("X") is not None and self.raw["X"].value is not None:
                dwell = float(self.raw["X"].value)
            self.emit(Kind.DWELL, ref, dwell_s=dwell)
        elif code == "G10":
            self._g10(number, program)
        elif code in ("G28", "G30"):
            self._reference_return(code, ref, words, number, program)
        elif code == "G53":
            self._machine_move(ref, words)
        elif code == "G52":
            for i, axis in enumerate("XYZ"):
                if axis in words and words[axis] is not None:
                    self.shift[i] = words[axis]
        elif code == "G92":
            for i, axis in enumerate("XYZ"):
                if axis in words and words[axis] is not None and not math.isnan(self.pos[i]):
                    self.shift[i] = self.pos[i] - words[axis]
            self.note(number, "unevaluated", "G92 shifts the coordinate system from the current position; positions "
                                             "after it rely on where the tool stood", program)
        elif code == "G31":
            self._linear(ref, words, rapid=False, kind=Kind.PROBE)
        elif code == "G37":
            h = self.raw.get("H")
            register = int(h.value) if h is not None and h.value is not None else state.tool
            if register is not None:
                self.result.length_measured[register] = (number, len(self.result.commands))
            self.emit(Kind.TOOL_SETTER, ref, setter={"measure": "length", "tool": state.tool})
        elif code == "G65":
            self.result.macro_calls.append((len(self.result.commands), number))
            self.note(number, "call", "G65 macro call is not evaluated (arguments are macro variables)", program)
        elif code == "G29":
            self.note(number, "unevaluated", "G29 return from the reference point is not evaluated", program)
        return None

    def _g10(self, number: int, program: str) -> None:
        l_word = self.raw.get("L")
        p_word = self.raw.get("P")
        if l_word is None or p_word is None or l_word.value is None or p_word.value is None:
            self.note(number, "unevaluated", "G10 without L and P is not evaluated", program)
            return
        kind, register = int(l_word.value), int(p_word.value)
        if kind == 2:
            if 1 <= register <= 6:
                self._set_offset(f"G{53 + register}", number)
            else:
                self.note(number, "unevaluated", f"G10 L2 P{register} (external/common offset) is not evaluated",
                          program)
        elif kind == 20:
            self._set_offset(f"G54.1P{register}", number)
        elif kind in (10, 11):
            self.result.length_measured[register] = (number, len(self.result.commands))
            r = self.raw.get("R")
            if kind == 10 and r is not None and r.value is not None:
                self.length_values[register] = self._scaled(r)
        elif kind in (12, 13):
            self.result.radius_set[register] = (number, len(self.result.commands))
        else:
            self.note(number, "unevaluated", f"G10 L{kind} is not evaluated", program)

    def _set_offset(self, code: str | None, number: int) -> None:
        if code is None:
            return
        code = normalise_offset(code)
        self.result.offsets_set[code] = (number, len(self.result.commands))
        values = [self.raw.get(a) for a in "XYZ"]
        old = self.work_offsets.get(code, (math.nan, math.nan, math.nan))
        new = []
        for i, word in enumerate(values):
            if word is None or word.value is None:
                new.append(old[i])
            elif self.state.distance == "incremental":
                new.append(old[i] + self._scaled(word))
            else:
                new.append(self._scaled(word))
        self.work_offsets[code] = tuple(new)

    def _reference_return(self, code, ref, words, number, program):
        state = self.state
        axes = [a for a in "XYZ" if a in words]
        intermediate = self.pos.copy()
        for i, axis in enumerate("XYZ"):
            if axis in words:
                value = words[axis]
                if value is None:
                    intermediate[i] = math.nan
                elif state.distance == "incremental":
                    intermediate[i] = self.pos[i] + value
                else:
                    intermediate[i] = value + self.shift[i]
        self.result.g28.append({"command": len(self.result.commands), "line": number, "program": program,
                                "code": code, "distance": state.distance, "axes": axes,
                                "intermediate_W_m": intermediate.tolist(), "start_W_m": self.pos.tolist()})
        self._move_to(Kind.RAPID, ref, intermediate)
        home = self.pos.copy()
        for i, axis in enumerate("XYZ"):
            if axis in axes:
                home[i] = self._machine_to_W(axis, self.ctx.home_m.get(axis, math.nan)) if code == "G28" else math.nan
        if code == "G30":
            self.note(number, "unevaluated", "G30 second reference point position is a machine parameter: unknown to "
                                             "the scan", program)
        self._move_to(Kind.RAPID, ref, home)

    def _machine_move(self, ref, words):
        target = self.pos.copy()
        for i, axis in enumerate("XYZ"):
            if axis in words:
                target[i] = math.nan if words[axis] is None else self._machine_to_W(axis, words[axis])
        self._move_to(Kind.RAPID, ref, target)

    def _machine_to_W(self, axis: str, value: float) -> float:
        """Program coordinate (tool tip, active work offset) of a machine-axis value."""
        offset = self.work_offsets.get(normalise_offset(self.state.work_offset))
        i = "XYZ".index(axis)
        if offset is None or value is None or math.isnan(value):
            return math.nan
        position = value - offset[i] - self.shift[i]
        if axis == "Z":
            position -= self._active_length()
        return position

    def _active_length(self) -> float:
        state = self.state
        if state.length_comp is None or state.length_offset is None or state.length_offset == 0:
            return 0.0
        value = self.length_values.get(state.length_offset)
        if value is None:
            return math.nan
        return value if state.length_comp == "G43" else -value

    def _home_W(self, position: np.ndarray, axes: str) -> np.ndarray:
        """``position`` with ``axes`` moved to the machine home (reference point), in program coordinates."""
        moved = position.copy()
        for axis in axes:
            moved["XYZ".index(axis)] = self._machine_to_W(axis, self.ctx.home_m.get(axis, math.nan))
        return moved

    def _set_work_offset(self, code: str, number: int) -> None:
        """Select a work offset: the tool stays where it is, so its program coordinates change."""
        state = self.state
        old = self.work_offsets.get(normalise_offset(state.work_offset))
        new = self.work_offsets.get(normalise_offset(code))
        if self.machine != "lathe":
            if old is not None and new is not None:
                self.pos = self.pos + np.asarray(old) - np.asarray(new)
            elif normalise_offset(code) != normalise_offset(state.work_offset):
                self.pos = np.array(NAN3)
        state.work_offset = code
        state.work_offset_line = number

    def _set_length(self, comp: str | None, register: int | None, number: int) -> None:
        """Length compensation changes the program Z of the (unmoved) tool by the offset change."""
        state = self.state
        before = self._active_length()
        state.length_comp = comp
        state.length_offset = register
        state.length_line = number
        state.length_section = state.section
        after = self._active_length()
        self.pos[2] = self.pos[2] - (after - before)

    # ------------------------------------------------------------------ motion
    def _target(self, words) -> np.ndarray:
        state = self.state
        target = self.pos.copy()
        for i, axis in enumerate("XYZ"):
            if axis not in words:
                continue
            value = words[axis]
            if value is None:
                target[i] = math.nan
            elif state.distance == "incremental":
                target[i] = self.pos[i] + value
            else:
                target[i] = value + self.shift[i]
        return target

    def _linear(self, ref, words, rapid: bool, kind: Kind | None = None) -> None:
        target = self._target(words)
        self._move_to(kind or (Kind.RAPID if rapid else Kind.LINEAR), ref, target)

    def _move_to(self, kind: Kind, ref, target: np.ndarray, arc: ArcSpec | None = None, cycle: dict | None = None):
        feed = None
        if kind in (Kind.LINEAR, Kind.ARC, Kind.PROBE):
            feed = Feed(FeedMode(self.state.feed_mode), 0.0 if self.state.feed is None else self.state.feed)
        elif kind == Kind.RAPID:
            feed = Feed(FeedMode.RAPID, 0.0)
        index = self.emit(kind, ref, target_W=target, arc=arc, feed=feed, cycle=cycle)
        self.pos = target.copy()
        return index

    def _arc(self, ref, words, code: str, end: np.ndarray | None = None) -> None:
        state = self.state
        start = self.pos.copy()
        end = self._target(words) if end is None else end
        plane = state.plane
        a, b, c = {"XY": (0, 1, 2), "ZX": (2, 0, 1), "YZ": (1, 2, 0)}[plane]
        offsets = {"XY": ("I", "J"), "ZX": ("K", "I"), "YZ": ("J", "K")}[plane]
        normal = np.zeros(3)
        normal[c] = 1.0
        sign = -1.0 if code == "G2" else 1.0
        radius_word = words.get("R")
        centre = np.array(NAN3)
        r_start = r_end = math.nan
        chord_too_long = False
        if radius_word is not None and not any(o in words for o in offsets):
            chord = math.hypot(end[a] - start[a], end[b] - start[b])
            radius = abs(radius_word)
            r_start = r_end = radius
            if chord > 2.0 * radius:
                chord_too_long = True
                half = 0.0
            else:
                half = math.sqrt(max(radius * radius - 0.25 * chord * chord, 0.0))
            mid_a, mid_b = 0.5 * (start[a] + end[a]), 0.5 * (start[b] + end[b])
            if chord > 0:
                da, db = (end[a] - start[a]) / chord, (end[b] - start[b]) / chord
                side = sign * (1.0 if radius_word > 0 else -1.0)
                centre[a] = mid_a - side * db * half
                centre[b] = mid_b + side * da * half
            centre[c] = start[c]
        else:
            offs = [words.get(o) for o in offsets]
            centre[a] = start[a] + (offs[0] or 0.0)
            centre[b] = start[b] + (offs[1] or 0.0)
            centre[c] = start[c]
            r_start = math.hypot(start[a] - centre[a], start[b] - centre[b])
            r_end = math.hypot(end[a] - centre[a], end[b] - centre[b])
        sweep = math.nan
        if not math.isnan(centre[a] + centre[b] + start[a] + start[b] + end[a] + end[b]):
            ang0 = math.atan2(start[b] - centre[b], start[a] - centre[a])
            ang1 = math.atan2(end[b] - centre[b], end[a] - centre[a])
            sweep = ang1 - ang0
            if sign < 0:
                while sweep >= 0:
                    sweep -= TWO_PI
            else:
                while sweep <= 0:
                    sweep += TWO_PI
        spec = ArcSpec(plane=plane, centre=tuple(float(v) for v in centre), normal=tuple(normal), sweep_rad=sweep,
                       radius_start_m=float(r_start), radius_end_m=float(r_end))
        index = self._move_to(Kind.ARC, ref, end, arc=spec)
        if chord_too_long:
            self.result.commands[index].modal = dict(self.result.commands[index].modal, arc_chord_too_long=True)

    # ------------------------------------------------------------------ canned cycles
    def _cancel_cycle(self) -> None:
        state = self.state
        state.cycle = None
        state.cycle_initial_z = None
        state.rigid_tap = False

    def _start_cycle(self, code: str, words, number: int) -> None:
        state = self.state
        if state.cycle is None:
            state.cycle_initial_z = self.pos[2]
        state.cycle = code
        state.cycle_line = number
        if state.motion in ("G2", "G3"):
            state.motion = "G1"

    def _cycle_holes(self, ref, words, number, program) -> None:
        state = self.state
        incremental = state.distance == "incremental"
        if state.cycle_initial_z is None:
            state.cycle_initial_z = self.pos[2]
        initial = state.cycle_initial_z
        if "R" in words:
            r = words["R"]
            state.cycle_r = None if r is None else ((initial + r) if incremental else r + self.shift[2])
        if "Z" in words:
            z = words["Z"]
            base = state.cycle_r if state.cycle_r is not None else initial
            state.cycle_z = None if z is None else ((base + z) if incremental else z + self.shift[2])
        count = self._repeat_count()
        if "K" not in self.raw and "L" not in self.raw:
            count = 1
        xy_words = {k: v for k, v in words.items() if k in "XY"}
        for repeat in range(max(count, 0)):
            target = self.pos.copy()
            for i, axis in enumerate("XY"):
                if axis in xy_words:
                    value = xy_words[axis]
                    if value is None:
                        target[i] = math.nan
                    elif incremental:
                        target[i] = self.pos[i] + value
                    else:
                        target[i] = value + self.shift[i]
            self._hole(ref, target[0], target[1], number, program)

    def _hole(self, ref, x, y, number, program) -> None:
        state = self.state
        initial = state.cycle_initial_z if state.cycle_initial_z is not None else self.pos[2]
        r_level = state.cycle_r if state.cycle_r is not None else initial
        bottom = state.cycle_z if state.cycle_z is not None else math.nan
        retract = r_level if state.cycle_retract == "G99" else initial
        cycle = {"code": state.cycle, "r_W_m": r_level, "z_W_m": bottom, "initial_W_m": initial,
                 "retract_W_m": retract, "retract_mode": state.cycle_retract}
        here = self.pos.copy()
        if not (here[0] == x and here[1] == y):
            self._move_to(Kind.RAPID, ref, np.array([x, y, here[2]]), cycle=dict(cycle, phase="position"))
        if not (self.pos[2] == r_level):
            self._move_to(Kind.RAPID, ref, np.array([x, y, r_level]), cycle=dict(cycle, phase="to_r"))
        feed_index = self._move_to(Kind.LINEAR, ref, np.array([x, y, bottom]), cycle=dict(cycle, phase="feed"))
        if state.cycle in ("G82", "G89", "G76", "G88"):
            self.emit(Kind.DWELL, ref, dwell_s=self._cycle_dwell(), cycle=dict(cycle, phase="dwell"))
        out_kind = Kind.LINEAR if state.cycle in TAP_CYCLES or state.cycle in ("G85", "G89") else Kind.RAPID
        self._move_to(out_kind, ref, np.array([x, y, retract]), cycle=dict(cycle, phase="retract"))
        self.result.holes.append(Hole(feed_index, number, program, state.tool, state.cycle, float(x), float(y),
                                      float(bottom), float(r_level), float(initial), float(retract),
                                      state.work_offset, state.section))

    def _cycle_dwell(self) -> float:
        p = self.raw.get("P")
        if p is None or p.value is None:
            return 0.0
        return float(p.value) / (1000.0 if not p.has_point else 1.0)

    # ------------------------------------------------------------------ lathe
    def _lathe_block(self, line, program, ref, words, g_codes, m_codes):
        state = self.state
        number = line.number
        lathe_words = dict(words)
        if "X" in lathe_words and lathe_words["X"] is not None:
            lathe_words["X"] = 0.5 * lathe_words["X"]
        incremental = {}
        if "U" in words:
            incremental["X"] = None if words["U"] is None else 0.5 * words["U"]
        if "W" in words:
            incremental["Z"] = words["W"]
        if "V" in words:
            incremental["Y"] = words["V"]
        motion_code = None
        one_shot = None
        for code in g_codes:
            if code in ("G0", "G1", "G2", "G3", "G32"):
                motion_code = code
            elif code in ("G17", "G18", "G19"):
                state.plane = {"G17": "XY", "G18": "ZX", "G19": "YZ"}[code]
            elif code in ("G98", "G99"):
                state.feed_mode = (FeedMode.PER_MIN if code == "G98" else FeedMode.PER_REV).value
            elif code == "G96":
                s = self.raw.get("S")
                if s is not None and s.value is not None:
                    state.css_m_s = float(s.value) * (1.0 if state.units == "mm" else FOOT_M) / 60.0
            elif code == "G97":
                state.css_m_s = None
            elif code == "G50":
                s = self.raw.get("S")
                if s is not None and s.value is not None:
                    state.max_rpm = float(s.value)
                if any(a in words for a in "XZ"):
                    self.note(number, "unevaluated", "G50 coordinate setting is not evaluated", program)
            elif code in ("G40", "G41", "G42"):
                state.radius_comp = code
            elif code in WORK_OFFSETS or code in HAAS_EXTRA_OFFSETS:
                self._set_work_offset(code, number)
            elif code in ("G4", "G28", "G53", "G10", "G31"):
                one_shot = code
            elif code in LATHE_CYCLE_G:
                motion_code = "lathe_cycle:" + code
            elif code in ("G20", "G21", "G9", "G61", "G64", "G187", "G103"):
                continue
            else:
                self.note(number, "unknown_code", f"{code} is not interpreted by the scan", program)
        if "F" in words and self.raw["F"].value is not None:
            self._set_feed(self.raw["F"].value, number)
        if "S" in words and self.raw["S"].value is not None and "G96" not in g_codes and "G50" not in g_codes:
            if state.css_m_s is not None:
                state.css_m_s = float(self.raw["S"].value) * (1.0 if state.units == "mm" else FOOT_M) / 60.0
            else:
                state.spindle_rpm = abs(float(self.raw["S"].value))
        t_word = self.raw.get("T")
        if t_word is not None and t_word.value is not None and one_shot != "G10":
            self._lathe_tool(int(t_word.value), ref, number, program)
        pre = [m for m in m_codes if m in ("M3", "M4") or m in COOLANT_ON.get(self.dialect, {})]
        post = [m for m in m_codes if m not in pre]
        for code in pre:
            self._m_code(code, ref, number, program)
        if one_shot in ("G4", "G10", "G53", "G31"):
            self._one_shot(one_shot, ref, words, number, program)
        elif one_shot == "G28":
            self._lathe_reference(ref, words, incremental, number, program)
        elif motion_code is not None and motion_code.startswith("lathe_cycle:"):
            code = motion_code.split(":", 1)[1]
            self.note(number, "unevaluated", f"{code} ({LATHE_CYCLE_G[code]}) passes are not expanded by the scan; "
                                             "it is checked as one feed move to its end point", program)
            target = self._lathe_target(lathe_words, incremental)
            if any(not math.isnan(v) for v in target) and any(k in words for k in "XZUW"):
                self._move_to(Kind.LINEAR, ref, target)
        else:
            if motion_code is not None:
                state.motion = "G1" if motion_code == "G32" else motion_code
            if any(k in words for k in "XZYUWV"):
                target = self._lathe_target(lathe_words, incremental)
                if state.motion in ("G2", "G3"):
                    self._arc(ref, {k: v for k, v in words.items() if k in "IJKR"}, state.motion, end=target)
                else:
                    self._move_to(Kind.RAPID if state.motion == "G0" else Kind.LINEAR, ref, target)
        for code in post:
            result = self._m_code(code, ref, number, program)
            if result is not None:
                return result
        return None

    def _lathe_target(self, words, incremental) -> np.ndarray:
        target = self.pos.copy()
        for i, axis in enumerate("XYZ"):
            if axis in words:
                value = words[axis]
                target[i] = math.nan if value is None else value + self.shift[i]
            if axis in incremental:
                value = incremental[axis]
                target[i] = math.nan if value is None else self.pos[i] + value
        if math.isnan(target[1]):
            target[1] = 0.0
        return target

    def _lathe_tool(self, value: int, ref, number: int, program: str) -> None:
        state = self.state
        tool, offset = (value // 100, value % 100) if value >= 100 else (value, value)
        if tool == 0 and offset == 0:
            state.length_offset = 0
            self._lathe_offset = 0
            self.emit(Kind.MODE, ref, tool={"number": state.tool, "offset": 0})
            return
        if tool == state.tool and tool != 0:
            state.length_offset = offset
            self._lathe_offset = offset
            state.length_comp = "T" if offset else None
            self.emit(Kind.MODE, ref, tool={"number": tool, "offset": offset})
            return
        state.next_tool = tool
        state.length_offset = offset
        state.length_comp = "T" if offset else None
        state.length_line = number
        state.length_section = state.section + 1
        self._lathe_offset = offset
        state.tool = tool
        state.tool_line = number
        state.section += 1
        index = self.emit(Kind.TOOL_CHANGE, ref, tool={"number": tool, "turret": "T1", "station": tool,
                                                       "length_offset": offset})
        restart = self._restart_line(program, number)
        self.result.sections.append(Section(state.section, tool, index, number, restart, program))

    def _lathe_reference(self, ref, words, incremental, number, program) -> None:
        state = self.state
        axes = [a for a in "XZ" if a in words] + [{"U": "X", "W": "Z"}[k] for k in "UW" if k in words]
        distance = "incremental" if any(k in words for k in "UW") and not any(k in words for k in "XZ") else "absolute"
        intermediate = self._lathe_target({k: (None if words[k] is None else (0.5 * words[k] if k == "X" else words[k]))
                                           for k in "XZ" if k in words}, incremental)
        self.result.g28.append({"command": len(self.result.commands), "line": number, "program": program,
                                "code": "G28", "distance": distance, "axes": axes,
                                "intermediate_W_m": intermediate.tolist(), "start_W_m": self.pos.tolist()})
        self._move_to(Kind.RAPID, ref, intermediate)
        home = self.pos.copy()
        for axis in axes:
            home["XYZ".index(axis)] = math.nan
        self._move_to(Kind.RAPID, ref, home)

    # ------------------------------------------------------------------ emission
    def emit(self, kind: Kind, ref: LineRef, *, target_W=None, arc=None, feed=None, spindle=None, coolant=None,
             tool=None, dwell_s=0.0, stop="", cycle=None, setter=None, modal_change=False) -> int:
        start = self.pos.copy()
        end = start if target_W is None else np.asarray(target_W, dtype=float)
        modal = self._modal(cycle)
        machine_target = self._machine_target(end) if target_W is not None else {}
        command = Command(kind=kind, ref=ref, space=Space.TIP, target=machine_target,
                          tip_target_m=tuple(float(v) for v in end) if target_W is not None else None,
                          tip_axis=(0.0, 0.0, 1.0) if self.machine != "lathe" else None,
                          arc=arc, feed=feed, spindle=spindle, coolant=coolant, tool=tool, dwell_s=float(dwell_s),
                          stop=stop, setter=setter, modal=modal)
        self.result.commands.append(command)
        self._starts.append(tuple(start))
        self._ends.append(tuple(end))
        return len(self.result.commands) - 1

    def _modal(self, cycle: dict | None) -> dict:
        key = (tuple(getattr(self.state, f.name) for f in fields(self.state)), None if cycle is None else
               tuple(sorted((k, v) for k, v in cycle.items() if not isinstance(v, float) or not math.isnan(v))))
        if key == self._snapshot_key:
            return self._snapshot
        snapshot = self.state.snapshot(self.machine, self.dialect)
        if cycle is not None:
            snapshot["cycle_info"] = dict(cycle)
        self._snapshot_key = key
        self._snapshot = snapshot
        return snapshot

    def _machine_target(self, end: np.ndarray) -> dict:
        offset = self.work_offsets.get(normalise_offset(self.state.work_offset))
        if offset is None or self.machine == "lathe":
            return {}
        target = {}
        for i, axis in enumerate("XYZ"):
            value = end[i] + offset[i]
            if axis == "Z":
                value += self._active_length()
            if not math.isnan(value):
                target[axis] = float(value)
        return target


def _decide_kind(ctx: SanityContext, lines: list[Line]) -> str:
    if ctx.machine_kind == "mill_turn":
        has_m6 = any("M6" in line.codes("M") for line in lines)
        return "mill" if has_m6 else "lathe"
    return "lathe" if ctx.machine_kind == "lathe" else "mill"


def scan(text: str, name: str, ctx: SanityContext, *, machine_kind: str | None = None,
         subprograms: dict[str, str] | None = None) -> ScanResult:
    """Run the static modal scan of one program (Fanuc or Haas dialect)."""
    return Scanner(text, name, ctx, machine_kind=machine_kind, subprograms=subprograms).run()


def code_of(letter: str, value: float | None) -> str:
    return code_text(letter, value)
