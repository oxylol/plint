"""Block semantics of the HEIDENHAIN TNC 640, Klartext and ISO (docs/dialects.md "Heidenhain TNC").

One handler executes both syntaxes (the ISO reader produces the same block kinds, plus ``iso`` blocks
whose modal motion is resolved here). Order inside a positioning block: coordinates and values are
evaluated (an undefined Q parameter as a lone coordinate drops the axis), radius compensation (RL /
RR / R0 / R+ / R-), feed (F modal, FMAX for the block, FAUTO, FU), the M functions that act at the block
start (M3 M4 M8 M13 M14), the motion (with RND / CHF of the next block as its corner), then the M
functions at the block end (M5 M9 M0 M1 M2 M30, M99 / M89 cycle calls, M128 / M129).

Coordinate transformations act in the TNC's order of effect whatever the programming order: the
tilted plane (PLANE) about the active datum, the datum shift (cycle 7, in the system active when it was
defined), mirroring (8), rotation (10) and scaling (11 / 26):
``x_machine = preset + O + R_tilt (M R(rot) S (x - c) + c)``, mapped onto the core's mirror -> scale ->
rotation -> feature stages exactly (M R = R' M with the rotation sense reversed by an odd number of
mirrored plane axes, M S(c) = S(Mc) M).
"""

from __future__ import annotations

import math

import numpy as np

from ...expr import round_half_away
from ...ir import Command, Kind
from ...pipeline import PLANES
from ..siemens840d.kit import Alarm, NativeHandler, arc_end_tangent, tangent_arc_centre
from . import contour, cycles
from .formula import Values
from .tables import M_EXECUTED, M_LISTED, M_START

TOOL_AXIS_PLANE = {"Z": "G17", "Y": "G18", "X": "G19"}
PLANE_TOOL_AXIS = {"XY": "Z", "ZX": "Y", "YZ": "X"}
AXIS_INDEX = {1: "X", 2: "Y", 3: "Z", 4: "A", 5: "B", 6: "C", 7: "U", 8: "V", 9: "W"}
ISO_MOTION = {"G00": "rapid", "G01": "linear", "G02": "cw", "G03": "ccw", "G05": "circle", "G06": "ct",
              "G10": "polar_rapid", "G11": "polar_linear", "G12": "polar_cw", "G13": "polar_ccw",
              "G15": "polar_circle", "G16": "polar_ct"}


class HeidenhainHandler(NativeHandler):
    dialect_key = "heidenhain"

    def __init__(self, ex, dialect):
        super().__init__(ex, dialect)
        self.q = Values(self)

    @property
    def iso(self) -> bool:
        """The current program is an ISO program (its own syntax, not the machine's dialect name)."""
        frames = self.s.frames
        if not frames:
            return self.d.name == "heidenhain_iso"
        return self.ex.store.info[frames[-1].program].syntax == "iso"

    # ================================================================== state
    def init_dialect(self) -> None:
        s = self.s
        s.variables["Q"] = [0.0] * 2000
        s.variables["QR"] = [0.0] * 500
        s.variables["QL1"] = [0.0] * 500
        s.sys["qs"] = {}
        self.reset_transforms()
        s.tool_axis_letter = "X" if self.T else "Z"
        start = "P1" if "P1" in s.offsets else "P0"
        s.wcs = start
        s.offsets.setdefault("P0", [0] * len(self.ex.axes))

    def offset_keys(self) -> list[str]:
        return ["P0"]

    def offset_key(self, key: str) -> str:
        if key.upper().startswith("G5") and key[1:].isdigit() and 54 <= int(key[1:]) <= 59:
            return f"P{int(key[1:]) - 53}"
        return key.upper() if key.upper().startswith("P") else key

    def reset_transforms(self) -> None:
        s = self.s
        s.sys["hh"] = {"T": [0.0, 0.0, 0.0], "O": [0.0, 0.0, 0.0], "TR": {}, "R": list(_IDENTITY9),
                       "mirror": [0, 0, 0], "rot": 0.0, "scale": [1.0, 1.0, 1.0], "centre": [0.0, 0.0, 0.0]}
        s.sys.pop("plane", None)
        s.sys.pop("pole", None)
        s.sys.pop("tangent", None)

    def on_reset(self) -> None:
        s = self.s
        s.modal = dict(self.d.power_on)
        s.sys.pop("cycle_def", None)
        s.sys.pop("pattern", None)
        s.sys.pop("m89", None)
        s.sys.pop("rep", None)
        s.sys.pop("fk", None)
        for key in [k for k in s.variables if k.startswith("QL") and k != "QL1"]:
            del s.variables[key]
        self.reset_transforms()
        s.tcp = ""
        s.feature = None
        s.mirror = {}
        s.scale = None
        s.rotation = None
        s.g52 = {}
        self.ex.invalidate_transform()

    def ql_scope(self) -> list:
        """QL parameters of the current program (each CALL PGM level has its own)."""
        level = sum(1 for f in self.s.frames if f.kind in ("main", "sub"))
        key = f"QL{level}"
        variables = self.s.variables
        for stale in [k for k in variables if k.startswith("QL") and int(k[2:]) > level]:
            del variables[stale]
        if key not in variables:
            variables[key] = [0.0] * 500
        return variables[key]

    # ================================================================== units and values
    def length_si(self, value: float) -> float:
        return float(value) * (0.0254 if self.s.units == "inch" else 1e-3)

    def feed_unit_m(self) -> float:
        return 0.0254 if self.s.units == "inch" else 1e-3

    def value(self, node, lone: bool = False):
        return self.q.value(node, lone)

    def number(self, node) -> float:
        return float(self.q.value(node))

    def axis_si(self, letter: str, value: float) -> float:
        axis = self.ex.by_letter[letter]
        if axis.kind == "rotary":
            return math.radians(value)
        si = self.length_si(value)
        return si / 2.0 if axis.diameter else si

    def position_value(self, letter: str, counts: int) -> float:
        axis = self.ex.by_letter[letter]
        si = counts * axis.quantum
        if axis.kind == "rotary":
            return _clean(math.degrees(si))
        if axis.diameter:
            si *= 2.0
        return _clean(si / (0.0254 if self.s.units == "inch" else 1e-3))

    def tool_axis_letter(self) -> str:
        return self.s.tool_axis_letter or "Z"

    def plane(self) -> str:
        if self.iso and self.s.modal.get("plane") in ("G17", "G18", "G19"):
            return {"G17": "XY", "G18": "ZX", "G19": "YZ"}[self.s.modal["plane"]]
        return {"Z": "XY", "Y": "ZX", "X": "YZ"}.get(self.tool_axis_letter(), "XY")

    def absolute(self) -> bool:
        return self.s.modal.get("distance", "G90") == "G90"

    def preassigned(self, index: int):
        s = self.s
        unit = 0.0254 if s.units == "inch" else 1e-3
        if index == 108:
            return self.comp_radius() / unit
        if index == 109:
            return float({"X": 0, "Y": 1, "Z": 2}.get(self.tool_axis_letter(), -1))
        if index == 110:
            d = s.spindle.direction
            last = s.sys.get("last_spindle_m", 0)
            return float({-1: 0, 1: 1}.get(d, 2 if last == 3 else 3 if last == 4 else -1))
        if index == 111:
            return 1.0 if any(s.pumps.values()) else 0.0
        if index == 113:
            return 1.0 if s.units == "inch" else 0.0
        if index == 114:
            return self.tool_offset_counts(self.tool_axis_letter()) * self.machine.lin_increment_m / unit
        if index in (115, 116, 117, 118, 119):
            point = s.sys.get("touch_point")
            if point is None:
                raise Alarm("not_modelled_variable", f"Q{index} (no probing cycle has run)")
            return float(point[index - 115])
        if index in (120, 121, 122):
            return float(s.sys.get("plane_angles", {}).get("ABC"[index - 120], 0.0))
        raise Alarm("not_modelled_variable", f"Q{index}")

    # ================================================================== block execution
    def run_block(self, block) -> None:
        s = self.s
        if block.error is not None:
            raise Alarm(block.error[0], block.error[1])
        hb = block.extra
        method = getattr(self, "do_" + hb.kind, None)
        if method is None:
            raise Alarm("syntax", hb.kind)
        method(hb.f)

    def do_empty(self, f) -> None:
        return None

    def do_comment(self, f) -> None:
        return None

    def do_eof(self, f) -> None:
        if len(self.s.frames) > 1 and self.s.frames[-1].kind == "sub":
            self.ex.ret()
            return
        raise Alarm("end_of_record", "end of the program without END PGM")

    def do_begin(self, f) -> None:
        unit = str(f.get("unit", "MM")).upper()
        if unit not in ("MM", "INCH"):
            raise Alarm("syntax", f"BEGIN PGM unit {unit}")
        self.s.units = "inch" if unit == "INCH" else "mm"

    def do_end_pgm(self, f) -> None:
        ex, s = self.ex, self.s
        if len(s.frames) > 1:
            if s.frames[-1].kind == "local":
                raise Alarm("label_not_found", "END PGM inside a label subprogram (LBL 0 missing)")
            ex.ret()
            return
        ex.end_program("END PGM")

    def do_blk(self, f) -> None:
        s = self.s
        values = {}
        for letter, inc, node in f["coords"]:
            values[("I" if inc else "") + letter] = self.number(node)
        blank = s.sys.setdefault("blank", {})
        blank[f["form"]] = values
        if f["axis"]:
            blank["axis"] = f["axis"]
        if f["form"] == "0.2" and "0.1" in blank:
            low, high = blank["0.1"], values
            for letter in "XYZ":
                if letter in high and letter in low and high[letter] <= low[letter]:
                    raise Alarm("blank_contradictory", f"BLK FORM {letter}")
            self.ex.note("info", f"workpiece blank {low} - {high}")

    # ------------------------------------------------------------------ tools
    def do_tool_def(self, f) -> None:
        s = self.s
        number = self.tool_number(f["tool"])
        keys = f["keys"]
        if "L" in keys or "R" in keys:
            row = s.tool_offsets.setdefault(str(number), [0] * 9)
            inc = self.machine.lin_increment_m
            if "L" in keys:
                row[0] = int(round_half_away(self.length_si(self.number(keys["L"])) / inc))
            if "R" in keys:
                row[2] = int(round_half_away(self.length_si(self.number(keys["R"])) / inc))
            return
        if s.preselect != number:
            s.preselect = number
            self.ex.emit(Command(Kind.MODE, self.ex.line_ref(), modal={"tool_preselect": number}))

    def tool_number(self, node) -> int:
        if node is None:
            raise Alarm("tool_undefined", "no tool number")
        if node[0] == "name":
            table = self.ex.ctx.tool_table
            for slot in (table.slots if table is not None else ()):
                if slot.tool_id.upper() == node[1] and slot.path == self.ex.ctx.path:
                    return int(slot.number)
            raise Alarm("tool_undefined", f'"{node[1]}"')
        value = self.number(node)
        if value < 0 or value != int(value):
            raise Alarm("tool_undefined", f"T{value:g}")
        return int(value)

    def do_tool_call(self, f) -> None:
        ex, s = self.ex, self.s
        keys = f["keys"]
        if f["tool"] is not None:
            number = self.tool_number(f["tool"])
            if s.comp_state is not None and s.comp_kind in ("cutter", "nose"):
                raise Alarm("comp_tool_change", "TOOL CALL with radius compensation active")
            if number != 0 and str(number) not in s.tool_offsets:
                raise Alarm("tool_undefined", f"T{number}")
            capacity = self.machine.tool_capacity
            if capacity and number > capacity:
                raise Alarm("tool_not_in_magazine", f"T{number}")
            axis = f["axis"] or ("Z" if not self.T else "X")
            if axis not in ex.by_letter:
                raise Alarm("tool_axis_missing", f"tool axis {axis}")
            number = self.tool_life_select(number) if number > 0 else number
            if number != s.tool:
                self.emit_tool_change(number)
            s.tool = number
            s.tool_axis_letter = axis
            if self.iso:
                s.modal["plane"] = TOOL_AXIS_PLANE[axis]
            s.sys["delta_length"] = 0
            s.sys["delta_radius"] = 0
            s.d_number = number
            s.h_number = number
            s.offset_number = number if self.T else 0
            s.sys["active_offset"] = list(s.tool_offsets[str(number)]) if number != 0 else None
            s.sys.pop("tangent", None)
        inc = self.machine.lin_increment_m
        if "DL" in keys:
            s.sys["delta_length"] = int(round_half_away(self.length_si(self.number(keys["DL"])) / inc))
        if "DR" in keys:
            s.sys["delta_radius"] = int(round_half_away(self.length_si(self.number(keys["DR"])) / inc))
        if "DR2" in keys:
            s.sys["delta_radius2"] = self.number(keys["DR2"])
        if f["tool"] is not None or "DL" in keys:
            self.offset_changed()
        if "F" in keys:
            # the TOOL CALL feed is the current feed until a positioning block programs another one, and the
            # value FAUTO recalls
            value = self.number(keys["F"])
            if value < 0.0:
                raise Alarm("feed_zero", f"F{value:g}")
            s.sys["feed_auto"] = value * self.feed_unit_m() / 60.0
            s.feed_si, s.feed_set, s.feed_prog = s.sys["feed_auto"], value > 0.0, value
            s.modal["feed_mode"] = "per_min"
        if "S" in keys:
            rpm = self.number(keys["S"])
            if rpm < 0.0:
                raise Alarm("spindle_rpm", f"S{rpm:g}")
            s.spindle.s_value = rpm
            if s.spindle.direction != 0:
                self.emit_spindle()

    # ------------------------------------------------------------------ positions
    def targets(self, coords, iso_absolute: bool | None = None) -> dict[str, int]:
        ex, s = self.ex, self.s
        out = {}
        for letter, inc, node in coords:
            axis = ex.by_letter.get(letter)
            if axis is None:
                raise Alarm("illegal_axis", f"axis {letter}")
            value = self.value(node, lone=True)
            if value is None:
                continue
            incremental = inc if inc is not None else not (iso_absolute if iso_absolute is not None else self.absolute())
            counts = ex.counts(letter, self.axis_si(letter, value))
            if incremental:
                out[letter] = s.pos.get(letter, 0) + counts
            elif axis.kind == "rotary" and axis.continuous:
                # an endless (modulo) rotary axis shows 0..360 deg: an absolute value moves from the shown
                # position to it, with M126 the shortest way (at most half a turn, +180 deg on a tie)
                turn = ex.counts(letter, 2.0 * math.pi)
                here = s.pos.get(letter, 0)
                if s.sys.get("m126"):
                    delta = (counts - here) % turn
                    if delta > turn // 2:
                        delta -= turn
                else:
                    delta = counts - here % turn
                out[letter] = here + delta
            else:
                out[letter] = counts
        return out

    def plane_point(self, target: dict[str, int] | None = None) -> tuple[float, float]:
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        pos = dict(s.pos)
        if target:
            pos.update(target)
        return ex.si(a, pos.get(a, 0)), ex.si(b, pos.get(b, 0))

    def pole(self) -> tuple[float, float]:
        pole = self.s.sys.get("pole")
        if pole is None:
            raise Alarm("pole_missing", "CC not defined")
        return float(pole[0]), float(pole[1])

    def polar_target(self, polar: dict, target: dict[str, int]) -> dict[str, int]:
        """PR / PA (IPR / IPA) about the pole CC: the plane axes of the end point."""
        ex = self.ex
        a, b, _ = PLANES[self.plane()]
        pole = self.pole()
        here = self.plane_point()
        radius = math.hypot(here[0] - pole[0], here[1] - pole[1])
        angle = math.degrees(math.atan2(here[1] - pole[1], here[0] - pole[0]))
        if "PR" in polar:
            inc, node = polar["PR"]
            value = self.length_si(self.number(node))
            radius = radius + value if inc else value
        if "PA" in polar:
            inc, node = polar["PA"]
            value = self.number(node)
            angle = angle + value if inc else value
        if radius < 0.0:
            raise Alarm("pole_illegal", "negative polar radius")
        out = dict(target)
        out[a] = ex.counts(a, pole[0] + radius * math.cos(math.radians(angle)))
        out[b] = ex.counts(b, pole[1] + radius * math.sin(math.radians(angle)))
        self.s.sys["polar_angle"] = angle
        return out

    # ------------------------------------------------------------------ feed, compensation, M functions
    def feed_for(self, f) -> tuple[bool, object]:
        """(rapid, Feed) of a block: FMAX for this block, F / FU modal, FAUTO from TOOL CALL."""
        ex, s = self.ex, self.s
        feed = f.get("feed")
        if feed is not None:
            kind = feed[0]
            if kind == "FMAX":
                return True, ex.feed_spec(True)
            if kind == "FAUTO":
                value = float(s.sys.get("feed_auto", 0.0))
                if value <= 0.0:
                    raise Alarm("feed_zero", "FAUTO without F in TOOL CALL")
                s.feed_si, s.feed_set, s.feed_prog = value, True, value * 60.0 / self.feed_unit_m()
                s.modal["feed_mode"] = "per_min"
            elif kind == "FZ":
                raise Alarm("feed_tooth", "FZ needs the number of teeth (CUT) of the tool table")
            else:
                value = self.number(feed[1])
                if value < 0.0:
                    raise Alarm("feed_zero", f"F{value:g}")
                if kind == "FU":
                    s.modal["feed_mode"] = "per_rev"
                    s.feed_si = value * self.feed_unit_m()
                else:
                    s.modal["feed_mode"] = "per_min"
                    s.feed_si = value * self.feed_unit_m() / 60.0
                    top = self.machine.max_feed_m_s
                    if top > 0.0 and s.feed_si > top:
                        ex.note("warning", f"F{value:g} clamped to the maximum feed {top * 60000.0:.0f} mm/min")
                        s.feed_si = top
                s.feed_prog, s.feed_set = value, value > 0.0
        return False, None

    def cutting_feed(self):
        s = self.s
        if not s.feed_set or s.feed_si <= 0.0:
            raise Alarm("feed_zero", "no feed rate")
        return self.ex.feed_spec(False)

    def compensation(self, code: str | None, element: str, moves: bool) -> None:
        """R0 / RL / RR (modal) at a positioning block; RL / RR may start only on a straight line."""
        ex, s = self.ex, self.s
        if code is None or code in ("R+", "R-"):
            return
        target = {"R0": "G40", "RL": "G41", "RR": "G42"}[code]
        current = s.modal.get("7", "G40")
        if target == current:
            return
        if element not in ("L", "LP", "APPR", "DEP", "FL", "iso_line"):
            raise Alarm("comp_start" if target != "G40" else "comp_end", f"{code} in a {element} block")
        active = s.comp_state is not None and s.comp_kind in ("cutter", "nose") and current in ("G41", "G42")
        if target == "G40":
            if active:
                ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"comp_marker": {"cancel": "G40", "seq": s.block_seq}}))
            s.modal["7"] = "G40"
            return
        if s.tool <= 0 or self.comp_radius() <= 0.0:
            ex.note("info", f"{code} with tool radius 0: the path is not offset")
        if active:
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"comp_marker": {"respecified": target, "seq": s.block_seq}}))
        ex.start_comp("nose" if self.T else "cutter")
        s.modal["7"] = target

    def m_split(self, codes: list[int]) -> tuple[list[int], list[int]]:
        if len(codes) > int(self.p["max_m_codes_per_block"]):
            raise Alarm("multiple_codes", " ".join(f"M{m}" for m in codes))
        if 3 in codes and 4 in codes or (8 in codes and 9 in codes) or (128 in codes and 129 in codes):
            raise Alarm("multiple_codes", " ".join(f"M{m}" for m in codes))
        before, after = [], []
        for m in codes:
            custom = (self.p.get("m_codes") or {}).get(f"M{m}") or (self.p.get("m_codes") or {}).get(f"M{m:02d}")
            if m not in M_EXECUTED and custom is None:
                if m in M_LISTED:
                    raise Alarm("unsupported_function", f"M{m}")
                raise Alarm("undefined_m_code", f"M{m}")
            if m in (91, 92):
                continue
            (before if m in M_START else after).append(m)
        return before, after

    def m_function(self, m: int, words) -> None:
        ex, s = self.ex, self.s
        custom = (self.p.get("m_codes") or {}).get(f"M{m}") or (self.p.get("m_codes") or {}).get(f"M{m:02d}")
        if custom is not None:
            self.custom_m(m, dict(custom))
            return
        if m in (3, 4, 13, 14):
            s.spindle.direction = -1 if m in (3, 13) else 1
            s.sys["last_spindle_m"] = 3 if m in (3, 13) else 4
            self.emit_spindle()
            if m in (13, 14):
                self.coolant_on(("flood",), f"M{m}")
        elif m == 5:
            s.spindle.direction = 0
            self.emit_spindle()
        elif m == 8:
            self.coolant_on(("flood",), "M8")
        elif m == 9:
            self.coolant_off("M9")
        elif m in (0, 1):
            if m == 1 and not s.operator.optional_stop:
                return
            if s.spindle.direction != 0:
                s.spindle.direction = 0
                self.emit_spindle()
            if any(s.pumps.values()):
                self.coolant_off(f"M{m}")
            self.program_stop(f"M{m}")
        elif m in (2, 30):
            if len(s.frames) > 1:
                raise Alarm("m2_in_subprogram", f"M{m}")
            if s.spindle.direction != 0:
                s.spindle.direction = 0
                self.emit_spindle()
            if any(s.pumps.values()):
                self.coolant_off(f"M{m}")
            s.sys["parts"] = int(s.sys.get("parts", 0)) + 1
            ex.end_program(f"M{m}")
        elif m == 99:
            cycles.call_defined(self, None)
        elif m == 89:
            s.sys["m89"] = True
            cycles.call_defined(self, None)
        elif m == 128:
            self.tcpm(True)
        elif m == 129:
            self.tcpm(False)
        elif m == 140:
            self.retract_tool_axis(words if isinstance(words, dict) else {})
        elif m in (126, 127):
            if m == 126:
                s.sys["m126"] = True
            else:
                s.sys.pop("m126", None)
        else:
            raise Alarm("undefined_m_code", f"M{m}")

    def retract_tool_axis(self, f: dict) -> None:
        """M140 MB d: d along the tool axis away from the workpiece; MB MAX: to the end of the traverse
        range. The tool stays parallel to the spindle on table kinematics, so this is the machine axis of
        the TOOL CALL axis (positive direction), in machine coordinates, at F of the M140 or else rapid."""
        ex, s = self.ex, self.s
        spec = f.get("m140")
        if spec is None:
            raise Alarm("syntax", "M140 without MB")
        letter = s.tool_axis_letter or ("X" if self.T else "Z")
        axis = ex.by_letter.get(letter)
        if axis is None:
            raise Alarm("tool_axis_missing", f"tool axis {letter}")
        current = s.mach.get(letter, 0) * axis.quantum
        if spec[0] == "MAX":
            if axis.travel is None:
                raise Alarm("unsupported_function", f"M140 MB MAX without a travel range of {letter}")
            target = float(axis.travel[1])
        else:
            distance = self.length_si(self.number(spec[1]))
            if distance < 0.0:
                raise Alarm("syntax", "M140 MB must be positive")
            target = current + distance
        if target <= current:
            return
        if axis.travel is not None and target > axis.travel[1] + 0.5 * axis.quantum:
            raise Alarm("software_limit", f"M140: {letter} {target * 1e3:.3f} beyond {axis.travel[1] * 1e3:g}",
                        axis.name, "+")
        counts = dict(s.mach)
        counts[letter] = int(round_half_away(target / axis.quantum))
        if "m140_feed" in f:
            value = self.number(f["m140_feed"])
            if value <= 0.0:
                raise Alarm("feed_zero", f"M140 F{value:g}")
            feed = ex.feed_spec(False, feed_si=value * self.feed_unit_m() / 60.0, mode="per_min")
            kind = "linear"
        else:
            feed, kind = ex.feed_spec(True), "rapid"
        ex.flush_comp("M140")
        s.applied = {l: self.tool_offset_counts(l) for l in ex.letters}      # the called tool's tip afterwards
        ex.machine_move(kind, {letter: counts[letter]}, feed, modal={"machine_coordinates": True, "m140": True})
        s.sys["last_element"] = "machine"

    def custom_m(self, m: int, spec: dict) -> None:
        ex = self.ex
        kind = spec.get("kind")
        if kind == "coolant":
            pump = spec.get("pump")
            if pump not in self.machine.pumps:
                raise Alarm("machine_feature_missing", f"M{m}: pump {pump}")
            self.coolant({pump: bool(spec.get("on", True))}, f"M{m}")
        elif kind in ("pallet", "door", "conveyor", "chuck", "output", "brake", "mode"):
            device = {k: v for k, v in spec.items() if k != "kind"}
            cmd_kind = {"pallet": Kind.PALLET, "door": Kind.DOOR, "conveyor": Kind.CONVEYOR,
                        "chuck": Kind.CHUCK}.get(kind, Kind.MODE)
            ex.emit(Command(cmd_kind, ex.line_ref(), modal={"device": {"kind": kind, "code": f"M{m}", **device}}))
        elif kind == "ignore":
            ex.note("info", f"M{m}: {spec.get('note', 'no action on this machine')}")
        else:
            raise Alarm("unsupported_function", f"M{m} kind {kind}")

    # ------------------------------------------------------------------ contour elements
    def corner_for(self) -> dict | None:
        """RND / CHF in the next block rounds or chamfers the corner at the end of this element."""
        ex, s = self.ex, self.s
        frame = s.frames[-1]
        program = ex.store.programs[frame.program]
        k = frame.pc + 1
        while k < len(program.blocks) and program.blocks[k].extra.kind in ("comment", "empty"):
            k += 1
        if k >= len(program.blocks) or program.blocks[k].error is not None:
            return None
        nxt = program.blocks[k].extra
        if nxt.kind == "RND":
            radius = self.length_si(self.number(nxt.f["keys"]["R"]))
            corner = {",R": abs(radius)}
        elif nxt.kind == "CHF":
            corner = {",C": abs(self.length_si(self.number(nxt.f["length"])))}
        elif nxt.kind == "iso" and any(g in nxt.f["g"] for g in ("G24", "G25")) and "R" in nxt.f["keys"]:
            value = abs(self.length_si(self.number(nxt.f["keys"]["R"])))
            corner = {",R" if "G25" in nxt.f["g"] else ",C": value}
        else:
            return None
        if s.comp_state is None:
            ex.start_comp("corner")
        return corner

    def do_RND(self, f) -> None:
        self.transition_block(f, "RND")

    def do_CHF(self, f) -> None:
        self.transition_block(f, "CHF")

    def transition_block(self, f, name: str) -> None:
        """RND / CHF act through the element before them (corner_for); their own feed has no element to
        apply to in the corner filter, so it is noted."""
        if f.get("feed") is not None:
            self.ex.note("warning", f"{name} F: the transition uses the feed of the element before it")
        prev_kind = self.s.sys.get("last_element")
        if prev_kind not in ("line", "arc"):
            raise Alarm("rounding_failed" if name == "RND" else "chamfer_failed", f"{name} without a contour element")

    def move_line(self, target: dict[str, int], rapid: bool, corner=None, machine: bool = False) -> None:
        ex = self.ex
        if not target:
            return
        if machine:
            # M91 / M92: the coordinates are the machine axes' own (machine datum, or the additional machine
            # datum for M92); "radius compensation remains the same in blocks that are programmed with M91 or
            # M92. The tool length, however, is not compensated" (TNC 640 Klartext manual, M91/M92): L Z-10
            # M91 puts the spindle 10 mm below the machine datum whatever tool is in it. The length of the called
            # tool stays active: the program position after the block is that tool's tip (machine - length),
            # so the next block programmed in the workpiece system starts from the tip
            ex.flush_comp("M91")
            counts = dict(target)
            self.s.applied = {l: self.tool_offset_counts(l) for l in ex.letters}
            ex.machine_move("rapid" if rapid else "linear", counts, ex.feed_spec(True) if rapid else self.cutting_feed(),
                            modal={"machine_coordinates": True})
            self.s.sys["last_element"] = "machine"
            return
        before = self.plane_point()
        feed = ex.feed_spec(True) if rapid else self.cutting_feed()
        self.check_travel(target)
        ex.move("rapid" if rapid else "linear", target, feed, corner=corner)
        after = self.plane_point()
        if after != before:
            self.s.sys["tangent"] = [after[0] - before[0], after[1] - before[1]]
        self.s.sys["last_element"] = "line"

    def move_arc(self, target: dict[str, int], centre: tuple[float, float], direction: int, *, turns: int = 0,
                 corner=None) -> None:
        feed = self.cutting_feed()
        tol = float(self.p["arc_end_point_tolerance_m"])
        self.arc_by_centre(target, direction, centre, feed, turns=turns, tolerance=(tol, 0.0), alarm_key="arc_end_point",
                           corner=corner)
        end = self.plane_point()
        self.s.sys["tangent"] = list(arc_end_tangent(centre, end, direction))
        self.s.sys["last_element"] = "arc"

    def positioning(self, f, element: str, body) -> None:
        """Common frame of a positioning block: compensation, feed, M at start, motion, M at end."""
        ex, s = self.ex, self.s
        machine = 91 in f.get("m", []) or 92 in f.get("m", [])
        before, after = self.m_split(f.get("m", []))
        rapid, _ = self.feed_for(f)
        self.compensation(f.get("comp"), element, True)
        for m in before:
            self.m_function(m, None)
        moved = body(rapid, machine)
        for m in after:
            self.m_function(m, f)
        if moved and s.sys.get("m89") and 99 not in after and 89 not in after and not self.inside_program_cycle():
            cycles.call_defined(self, None)

    def inside_program_cycle(self) -> bool:
        """True while the program of a program cycle (CYCL DEF 12 / SEL CYCLE) runs: its own positioning
        blocks do not repeat the modal M89 call."""
        cycle = self.s.sys.get("cycle_def")
        if not cycle or cycle.get("number") != 12:
            return False
        index = self.program_names().get(cycle["program"])
        return any(frame.program == index for frame in self.s.frames)

    def do_L(self, f) -> None:
        def body(rapid, machine):
            target = self.targets(f["coords"])
            if f.get("polar"):
                raise Alarm("syntax", "PR/PA in an L block (use LP)")
            if f.get("comp") in ("R+", "R-"):
                target = self.paraxial(target, f["comp"])
            if 92 in f.get("m", []):
                target = {k: v + self.ex.by_letter[k].home for k, v in target.items()}
            corner = None if (machine or rapid) else self.corner_for()
            self.move_line(target, rapid, corner, machine)
            return bool(target)
        self.positioning(f, "L", body)

    def paraxial(self, target: dict[str, int], code: str) -> dict[str, int]:
        ex, s = self.ex, self.s
        moving = [l for l in target if target[l] != s.pos.get(l, 0)]
        if len(moving) != 1:
            raise Alarm("syntax", f"{code} needs a move in one axis")
        letter = moving[0]
        sign = 1 if target[letter] > s.pos.get(letter, 0) else -1
        delta = ex.counts(letter, self.comp_radius())
        out = dict(target)
        out[letter] += sign * delta if code == "R+" else -sign * delta
        return out

    def do_LP(self, f) -> None:
        def body(rapid, machine):
            target = self.polar_target(f["polar"], self.targets(f["coords"]))
            self.move_line(target, rapid, None if rapid else self.corner_for())
            return True
        self.positioning(f, "LP", body)

    def do_CC(self, f) -> None:
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        here = self.plane_point()
        if not f["coords"]:
            s.sys["pole"] = [here[0], here[1]]
            return
        pole = list(s.sys.get("pole") or here)
        for letter, inc, node in f["coords"]:
            if letter not in (a, b):
                raise Alarm("pole_illegal", f"CC with axis {letter}")
            value = self.axis_si(letter, self.number(node))
            k = 0 if letter == a else 1
            pole[k] = here[k] + value if inc else value
        s.sys["pole"] = [float(pole[0]), float(pole[1])]

    def circle_feed(self, f) -> None:
        feed = f.get("feed")
        if feed is not None and feed[0] == "FMAX":
            raise Alarm("circle_illegal", "FMAX in a circular block")

    def do_C(self, f) -> None:
        def body(rapid, machine):
            self.circle_feed(f)
            direction = f.get("dr")
            if direction is None:
                raise Alarm("arc_direction", "C without DR+ / DR-")
            target = self.targets(f["coords"])
            self.move_arc(target, self.pole(), direction, corner=self.corner_for())
            self.s.sys["dr_last"] = direction
            return True
        self.positioning(f, "C", body)

    def do_CR(self, f) -> None:
        def body(rapid, machine):
            self.circle_feed(f)
            direction = f.get("dr")
            if direction is None:
                raise Alarm("arc_direction", "CR without DR+ / DR-")
            if "R" not in f["keys"]:
                raise Alarm("arc_no_radius", "CR without R")
            target = self.targets(f["coords"])
            radius = self.length_si(self.number(f["keys"]["R"]))
            centre = self.radius_centre(self.plane_point(), self.plane_point(target), radius, direction)
            self.move_arc(target, centre, direction, corner=self.corner_for())
            self.s.sys["dr_last"] = direction
            return True
        self.positioning(f, "CR", body)

    def radius_centre(self, start, end, radius: float, direction: int):
        dx, dy = end[0] - start[0], end[1] - start[1]
        chord = math.hypot(dx, dy)
        if chord == 0.0:
            raise Alarm("arc_no_end", "CR to its own start point")
        half = chord / 2.0
        r = abs(radius)
        tol = float(self.p["arc_end_point_tolerance_m"])
        if r < half:
            if half - r > tol:
                raise Alarm("illegal_radius", f"R {r * 1e3:.4f} mm < half chord {half * 1e3:.4f} mm")
            r = half
        h = math.sqrt(max(r * r - half * half, 0.0))
        side = direction * (1.0 if radius > 0.0 else -1.0)
        mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
        return mx + side * h * (-dy / chord), my + side * h * (dx / chord)

    def do_CT(self, f) -> None:
        def body(rapid, machine):
            self.circle_feed(f)
            target = self.targets(f["coords"])
            self.tangential(target)
            return True
        self.positioning(f, "CT", body)

    def tangential(self, target: dict[str, int]) -> None:
        tangent = self.s.sys.get("tangent")
        if tangent is None:
            raise Alarm("circle_illegal", "CT without a preceding contour element")
        found = tangent_arc_centre(self.plane_point(), self.plane_point(target), tangent)
        if found is None:
            raise Alarm("circle_illegal", "CT end point on the tangent")
        centre, direction = found
        self.move_arc(target, centre, direction, corner=self.corner_for())

    def do_CP(self, f) -> None:
        def body(rapid, machine):
            self.circle_feed(f)
            direction = f.get("dr")
            if direction is None:
                raise Alarm("arc_direction", "CP without DR+ / DR-")
            pole = self.pole()
            here = self.plane_point()
            start_angle = math.degrees(math.atan2(here[1] - pole[1], here[0] - pole[0]))
            if "PA" not in f["polar"]:
                raise Alarm("arc_no_end", "CP without PA")
            inc, node = f["polar"]["PA"]
            angle = self.number(node)
            sweep = angle if inc else None
            target = self.polar_target({"PA": (inc, node)}, self.targets(f["coords"]))
            turns = 0
            if sweep is not None and abs(sweep) >= 360.0:
                if (sweep > 0) != (direction > 0):
                    raise Alarm("helix_illegal", "IPA against DR")
                turns = int(abs(sweep) // 360.0)
                if abs(abs(sweep) - 360.0 * turns) < 1e-9:
                    turns -= 1
            self.move_arc(target, pole, direction, turns=turns, corner=self.corner_for())
            self.s.sys["dr_last"] = direction
            del start_angle
            return True
        self.positioning(f, "CP", body)

    def do_CTP(self, f) -> None:
        def body(rapid, machine):
            self.circle_feed(f)
            target = self.polar_target(f["polar"], self.targets(f["coords"]))
            self.tangential(target)
            return True
        self.positioning(f, "CTP", body)

    def do_APPR(self, f) -> None:
        contour.approach(self, f)

    def do_DEP(self, f) -> None:
        contour.departure(self, f)

    def do_FPOL(self, f) -> None:
        contour.fk_pole(self, f)

    def do_FK(self, f) -> None:
        contour.fk_element(self, f)

    def do_FSELECT(self, f) -> None:
        """The choice was read by the FK element before it (contour.fk_element looks ahead)."""
        self.s.sys.pop("fk_selected", None)

    def do_m_only(self, f) -> None:
        def body(rapid, machine):
            return False
        self.positioning(f, "M", body)

    def do_stop(self, f) -> None:
        before, after = self.m_split(f.get("m", []))
        for m in before + after:
            self.m_function(m, None)
        if not any(m in (0, 1, 2, 30) for m in after):
            self.program_stop("STOP")

    # ------------------------------------------------------------------ cycles and patterns
    def do_cycl_def(self, f) -> None:
        cycles.define(self, f)

    def do_cycl_call(self, f) -> None:
        cycles.cycl_call(self, f)

    def do_pattern(self, f) -> None:
        cycles.pattern_def(self, f)

    # ------------------------------------------------------------------ labels and programs
    def label_value(self, label):
        if isinstance(label, list):
            if label[0] == "qs":
                return self.q.get_string(label[1]).upper()
            value = self.q.get(label[1], label[2])
            if value is None:
                raise Alarm("q_undefined", f"{label[1]}{label[2]}")
            return int(round_half_away(value))
        return label

    def do_lbl(self, f) -> None:
        ex, s = self.ex, self.s
        label = self.label_value(f["label"])
        if label == 0:
            if len(s.frames) > 1 and s.frames[-1].kind == "local":
                ex.ret()
            return

    def label_index(self, label) -> int:
        frame = self.s.frames[-1]
        info = self.ex.store.info[frame.program]
        if label == 0:
            raise Alarm("label_zero", "CALL LBL 0")
        index = info.labels.get(label)
        if index is None:
            raise Alarm("label_not_found", f"LBL {label}")
        return index

    def do_call_lbl(self, f) -> None:
        ex, s = self.ex, self.s
        label = self.label_value(f["label"])
        index = self.label_index(label)
        frame = s.frames[-1]
        here = frame.pc
        rep = int(round_half_away(self.number(f["rep"]))) if f.get("rep") is not None else 0
        if rep < 0:
            raise Alarm("syntax", f"REP {rep}")
        if index < here and f.get("rep") is not None:
            key = f"{frame.program}:{here}:{len(s.frames)}"
            counters = s.sys.setdefault("rep", {})
            left = counters.get(key, rep)
            if left > 0:
                counters[key] = left - 1
                ex.advance((here - index) * self.p["goto_scan_s"])
                frame.pc = index
            else:
                counters.pop(key, None)
            return
        for other in s.frames:
            if other.kind == "local" and other.program == frame.program and other.entry_pc == index + 1:
                raise Alarm("recursive_label", f"LBL {label}")
        if len(s.frames) >= int(self.p["sub_nesting_max"]):
            raise Alarm("nesting", f"more than {self.p['sub_nesting_max']} levels")
        ex.request_call("local", frame.program, None, rep + 1, entry_pc=index + 1)

    def do_sel_pgm(self, f) -> None:
        self.find_program(f["name"])
        self.s.sys["selected_pgm"] = f["name"]

    def do_call_selected(self, f) -> None:
        name = self.s.sys.get("selected_pgm")
        if name is None:
            raise Alarm("program_not_found", "CALL SELECTED PGM without SEL PGM")
        self.do_call_pgm({"name": name})

    def do_sel_cycle(self, f) -> None:
        cycles.program_cycle(self, f["name"])

    def do_call_pgm(self, f) -> None:
        ex, s = self.ex, self.s
        index = self.find_program(f["name"])
        if any(fr.program == index for fr in s.frames):
            raise Alarm("recursive_program", f["name"])
        if len(s.frames) >= int(self.p["sub_nesting_max"]):
            raise Alarm("nesting", f"more than {self.p['sub_nesting_max']} levels")
        if ex.store.info[index].unit != ex.store.info[s.frames[-1].program].unit:
            ex.note("warning", f"CALL PGM {f['name']}: unit of measure differs from the calling program")
        ex.request_call("sub", index, None, 1)

    # ------------------------------------------------------------------ Q parameters
    def do_formula(self, f) -> None:
        target = f["target"]
        if target[0] == "qs":
            self.q.set_string(target[1], self.string_value(f))
            return
        if f.get("undefined"):
            self.q.set(target[1], target[2], None)
            return
        node = f["expr"]
        if node[0] == "ang":
            a, b = self.number(node[1]), self.number(node[2])
            if a == 0.0 and b == 0.0:
                raise Alarm("zero_divide", "FN 13 with 0 ANG 0")
            value = math.degrees(math.atan2(b, a)) % 360.0
        else:
            value = self.number(node)
        self.q.set(target[1], target[2], value)

    def string_value(self, f) -> str:
        if "string" in f:
            return f["string"]
        if "copy" in f:
            return self.q.get_string(f["copy"])
        if "concat" in f:
            return "".join(text if kind == "text" else self.q.get_string(text) for kind, text in f["concat"])
        if "tochar" in f:
            args = f["tochar"]
            decimals = int(round_half_away(self.number(args["DECIMALS"]))) if "DECIMALS" in args else 3
            if not 0 <= decimals <= 9:
                raise Alarm("q_invalid", f"TOCHAR DECIMALS{decimals}")
            return f"{self.number(args['DAT']):.{decimals}f}"
        args = f["substr"]
        source = self.q.get_string(args["SRC"])
        begin = int(round_half_away(self.number(args["BEG"])))
        length = int(round_half_away(self.number(args["LEN"])))
        if begin < 0 or length < 0 or begin + length > len(source):
            raise Alarm("q_invalid", f"SUBSTR BEG{begin} LEN{length} of a string of {len(source)}")
        return source[begin:begin + length]

    def do_fn_jump(self, f) -> None:
        ex, s = self.ex, self.s
        a = self.q.value(f["a"], lone=True)
        op = f["op"]
        if op in ("ISDEFINED", "ISUNDEFINED"):
            result = (a is not None) == (op == "ISDEFINED")
        else:
            if a is None:
                raise Alarm("q_undefined", "FN jump with an undefined parameter")
            b = self.number(f["b"])
            result = {"EQU": a == b, "NE": a != b, "GT": a > b, "LT": a < b}[op]
        if not result:
            return
        label = self.label_value(f["label"])
        index = self.label_index(label)
        frame = s.frames[-1]
        ex.advance(abs(index - frame.pc) * self.p["goto_scan_s"])
        frame.pc = index

    def do_fn14(self, f) -> None:
        number = int(round_half_away(self.number(f["value"])))
        if number < 0:
            raise Alarm("fn14_no_text", f"FN 14: ERROR = {number}")
        code = None
        if 1000 <= number <= 1199:
            candidate = f"280-{0x3E8 + number - 1000:04X}"
            entry = self.d.alarms.by_code.get(candidate)
            if entry is not None:
                code, text = candidate, str(entry["text"])
        if code is None:
            entry = self.d.alarms.by_code.get("280-0064")
            template = str(entry["text"]) if entry is not None else "FN 14: error code %1"
            code, text = "280-0064", template.replace("%1", str(number))
        from ...alarms import ControllerAlarm

        raise ControllerAlarm("program_alarm", f"FN 14: ERROR = {number}", code, text)

    def do_fn16(self, f) -> None:
        ex, s = self.ex, self.s
        text = f["text"]
        s.printed.append(text)
        if "SCREEN:" in text.upper():
            self.message_stop(text, "FN 16")

    def do_fn17(self, f) -> None:
        s = self.s
        value = self.number(f["value"])
        if f["id"] == 50 and f["nr"] in (1, 2, 4, 5):
            tool = int(round_half_away(self.number(f["idx"]))) if f["idx"] is not None else s.tool
            row = s.tool_offsets.setdefault(str(tool), [0] * 9)
            column = {1: 0, 2: 2, 4: 1, 5: 3}[f["nr"]]
            row[column] = int(round_half_away(self.length_si(value) / self.machine.lin_increment_m))
            return
        raise Alarm("not_modelled_variable", f"FN 17: SYSWRITE ID{f['id']} NR{f['nr']}")

    def do_fn18(self, f) -> None:
        target = f["target"]
        value = self.sysread(f["id"], f["nr"], self.number(f["idx"]) if f["idx"] is not None else None)
        self.q.set(target[1], target[2], value)

    def sysread(self, ident: int, nr: int, idx) -> float:
        """FN 18 system data (db/controllers/heidenhain-tnc640.yaml variables.fn18_sysread); values metric."""
        ex, s = self.ex, self.s
        mm = 1e3
        if ident == 10 and nr == 3:
            return float((s.sys.get("cycle_def") or {}).get("number", 0))
        if ident == 20:
            if nr == 1:
                return float(s.tool)
            if nr == 2:
                return float(s.preselect or 0)
            if nr == 3:
                return float({"X": 0, "Y": 1, "Z": 2}.get(self.tool_axis_letter(), -1))
            if nr == 4:
                return float(s.spindle.s_value)
            if nr == 5:
                return self.preassigned(110)
            if nr == 8:
                return 1.0 if any(s.pumps.values()) else 0.0
            if nr == 9:
                return s.feed_si * 60.0 * mm if s.feed_set else 0.0
        if ident == 30:
            cycle = s.sys.get("cycle_def") or {}
            names = {1: "Q200", 2: "Q201", 3: "Q202", 4: "Q206", 5: "Q218", 6: "Q219", 8: "Q207", 10: "Q211",
                     11: "Q239", 12: "Q368"}
            name = names.get(nr)
            if name is None:
                raise Alarm("fn18_number", f"ID30 NR{nr}")
            return float(cycle.get("params", {}).get(name, 0.0))
        if ident == 35:
            if nr == 1:
                return 0.0 if self.absolute() else 1.0
            if nr == 2:
                return 0.0 if s.modal.get("7", "G40") == "G40" else 1.0
        if ident == 50:
            tool = int(round_half_away(idx)) if idx is not None else s.tool
            row = s.tool_offsets.get(str(tool))
            if row is None:
                raise Alarm("tool_undefined", f"T{tool}")
            column = {1: 0, 2: 2, 4: 1, 5: 3}.get(nr)
            if column is None:
                raise Alarm("not_modelled_variable", f"FN 18 ID50 NR{nr}")
            return row[column] * self.machine.lin_increment_m * mm
        if ident == 60:
            values = {1: float(s.tool), 2: float({"X": 0, "Y": 1, "Z": 2}.get(self.tool_axis_letter(), -1)),
                      3: float(s.spindle.s_value), 4: s.sys.get("delta_length", 0) * self.machine.lin_increment_m * mm,
                      5: s.sys.get("delta_radius", 0) * self.machine.lin_increment_m * mm}
            if nr in values:
                return values[nr]
        if ident == 210:
            hh = s.sys["hh"]
            if nr == 2:
                return float(hh["rot"])
            if nr == 3:
                return float(sum(bit << k for k, bit in enumerate(hh["mirror"])))
            if nr == 4:
                axis = int(idx or 1)
                return float(hh["scale"][axis - 1]) if 1 <= axis <= 3 else 1.0
            if nr == 6:
                return -1.0 if s.sys.get("plane") else 0.0
        if ident in (220, 240, 270, 230):
            axis_number = int(round_half_away(idx)) if idx is not None else 0
            letter = AXIS_INDEX.get(axis_number)
            if letter is None or letter not in ex.by_letter:
                raise Alarm("fn18_axis", f"IDX{axis_number}")
            axis = ex.by_letter[letter]
            if ident == 220 and nr == 2:
                k = "XYZ".index(letter) if letter in "XYZ" else None
                hh = s.sys["hh"]
                return float(hh["T"][k] * mm) if k is not None else math.degrees(hh["TR"].get(letter, 0.0))
            if ident == 240 and nr == 1:
                return self.position_si_value(axis, s.mach.get(letter, 0)) * (mm if axis.kind == "linear" else 1.0)
            if ident == 270 and nr == 1:
                return self.position_si_value(axis, s.pos.get(letter, 0)) * (mm if axis.kind == "linear" else 1.0)
            if ident == 230 and nr in (2, 3):
                if axis.travel is None:
                    raise Alarm("not_modelled_variable", f"travel of {letter}")
                value = axis.travel[0 if nr == 2 else 1]
                return value * mm if axis.kind == "linear" else math.degrees(value)
        if ident == 280 and nr == 1:
            return -1.0 if s.tcp == "heidenhain:TCPM" else 0.0
        raise Alarm("not_modelled_variable", f"FN 18: SYSREAD ID{ident} NR{nr}")

    def position_si_value(self, axis, counts: int) -> float:
        si = counts * axis.quantum
        return math.degrees(si) if axis.kind == "rotary" else si

    # ------------------------------------------------------------------ transformations
    def update_transform(self) -> None:
        """The TNC chain (preset + O + R_tilt (M R S (x - c) + c)) onto mirror -> scale -> rotation -> feature."""
        ex, s = self.ex, self.s
        hh = s.sys["hh"]
        letters = [l for l in "XYZ" if l in ex.by_letter]
        mirror = hh["mirror"]
        s.mirror = {l: 0 for i, l in enumerate("XYZ") if mirror[i] and l in ex.by_letter}
        factors = hh["scale"]
        if any(f != 1.0 for f in factors):
            centre = [(-c if mirror[i] else c) for i, c in enumerate(hh["centre"])]
            s.scale = {"centre": {l: ex.counts(l, centre["XYZ".index(l)]) for l in letters},
                       "factor": {l: float(factors["XYZ".index(l)]) for l in letters}}
        else:
            s.scale = None
        plane = self.plane()
        a, b, _ = PLANES[plane]
        flips = sum(1 for axis in (a, b) if mirror["XYZ".index(axis)])
        angle = float(hh["rot"]) * (-1.0 if flips == 1 else 1.0)
        s.rotation = {"plane": plane, "centre": [0, 0], "angle_deg": angle} if angle else None
        origin = [ex.counts(l, hh["O"][i]) if l in ex.by_letter else 0 for i, l in enumerate("XYZ")]
        if any(origin) or hh["R"] != _IDENTITY9:
            s.feature = {"origin": origin, "matrix": list(hh["R"])}
        else:
            s.feature = None
        for axis in ex.axes:
            if axis.kind == "rotary":
                shift = hh["TR"].get(axis.letter, 0.0)
                if shift:
                    s.g52[axis.letter] = ex.counts(axis.letter, shift)
                else:
                    s.g52.pop(axis.letter, None)
        s.sys.pop("tangent", None)
        ex.invalidate_transform()
        ex.resync_program_position()

    def datum_shift(self, values: dict[str, tuple[bool, float]]) -> None:
        """Cycle 7 / ISO G54: per axis absolute or incremental to the previous shift, in the system active
        when it is defined (tilted when a PLANE is active)."""
        hh = self.s.sys["hh"]
        shift = list(hh["T"])
        for letter, (inc, value) in values.items():
            if letter in "XYZ":
                k = "XYZ".index(letter)
                si = self.length_si(value)
                shift[k] = shift[k] + si if inc else si
            else:
                si = math.radians(value)
                hh["TR"][letter] = hh["TR"].get(letter, 0.0) + si if inc else si
        hh["T"] = shift
        rotation = np.asarray(hh["R"], dtype=float).reshape(3, 3)
        hh["O"] = [float(v) for v in rotation @ np.asarray(shift)]
        self.update_transform()

    def tcpm(self, on: bool) -> None:
        ex, s = self.ex, self.s
        if on:
            if not self.machine.table_kinematics_supported():
                raise Alarm("transform_unavailable", "M128 / FUNCTION TCPM needs rotary table axes")
            if s.modal.get("7", "G40") != "G40":
                raise Alarm("tcp_not_allowed", "TCPM with radius compensation active")
            s.tcp = "heidenhain:TCPM"
        else:
            s.tcp = "heidenhain:PLANE" if s.sys.get("plane") else ""
        ex.resync_program_position()

    # FUNCTION TCPM options the twin follows: the feed on the tool tip, rotary axes by their positions,
    # axis interpolation of the orientation, the reference points tip-tip (the TNC's defaults)
    TCPM_OPTIONS = {"F": ("TCP",), "AXIS": ("POS",), "PATHCTRL": ("AXIS",), "REFPNT": ("TIP-TIP",)}

    def do_tcpm(self, f) -> None:
        options = list(f.get("options") or [])
        k = 0
        while k < len(options):
            name = options[k]
            value = options[k + 1] if k + 1 < len(options) else ""
            if name not in self.TCPM_OPTIONS:
                raise Alarm("syntax", f"FUNCTION TCPM {name}")
            if value not in self.TCPM_OPTIONS[name]:
                raise Alarm("unsupported_function", f"FUNCTION TCPM {name} {value or '(no value)'}")
            k += 2
        self.tcpm(bool(f["on"]))

    def do_plane(self, f) -> None:
        from . import plane

        plane.execute(self, f)

    # ================================================================== ISO blocks
    def do_iso(self, f) -> None:
        from . import isoexec

        isoexec.execute(self, f)


_IDENTITY9 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _clean(value: float) -> float:
    if value == 0.0:
        return 0.0
    return float(f"{value:.12g}")


__all__ = ["HeidenhainHandler"]
