"""Runtime shared by the dialects that bring their own syntax (Siemens 840D sl, Heidenhain TNC).

Both dialects parse their programs themselves (``Dialect.parser``) and execute blocks with a handler
that implements the methods the controller core calls (docs/dialects.md "Adding a dialect").
:class:`NativeHandler` is that common part:

* the core interface: ``init_state``, ``plane``, ``feed_mode``, ``absolute``, ``tool_offset_counts``,
  ``comp_radius``, ``modal_snapshot``, ``apply_mode``, ``deferred_m`` and the value helpers;
* emission of the non-motion commands (spindle, coolant, tool change, stops, path waits);
* arcs from an explicit centre with the dialect's end-point tolerance (``arc_by_centre``);
* the frame stack of these controls mapped onto the core's coordinate pipeline (``apply_frame``):
  ``x_parent = T + R S M x`` (translation, rotation, scale, mirror, the Siemens component order;
  the Heidenhain chain is brought into the same form by its handler);
* the inverse kinematics of two rotary table axes (``table_solutions``) for tilted planes and TCP;
* alarm texts with their run-time parameters (``BoundAlarms``).

Nothing here keeps state: everything lives in ``ex.s`` (PathState); the handler object only holds
the immutable context and per-block scratch values.
"""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np

from ... import hooks
from ...alarms import ControllerAlarm
from ...expr import round_half_away
from ...ir import Command, Kind
from ...pipeline import PLANE_OF, PLANES

PLANE_CODE = {"XY": "G17", "ZX": "G18", "YZ": "G19"}
TOOL_AXIS_OF_PLANE = {"XY": "Z", "ZX": "Y", "YZ": "X"}
FEED_SCALE = {"mm": 1e-3, "inch": 0.0254}
IDENTITY9 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# Offset rows (counts) in the layouts of the core's FANUC handler, which the compensation providers read:
# mill [length, length wear, radius, radius wear, ...]; lathe [X wear, Z wear, nose wear, tip, X, Z, nose
# radius, Y wear, Y] (X and Z as radius values).
ROW_LEN, ROW_LEN_WEAR, ROW_RAD, ROW_RAD_WEAR = 0, 1, 2, 3
ROW_X_WEAR, ROW_Z_WEAR, ROW_NOSE_WEAR, ROW_TIP, ROW_X, ROW_Z, ROW_NOSE = 0, 1, 2, 3, 4, 5, 6
ROW_SIZE = 9


class Alarm(ControllerAlarm):
    """A dialect alarm with run-time parameters for the control's text (%3, %4 ...)."""

    def __init__(self, key: str, detail: str = "", *params):
        super().__init__(key, detail)
        self.params = [str(p) for p in params]


class BoundAlarms:
    """The dialect's AlarmTable seen from one path: ``resolve`` fills the text's parameters
    (channel, block, the alarm's own values) the way the control displays them."""

    def __init__(self, table, handler):
        self._table = table
        self._handler = handler

    def __getattr__(self, name):
        return getattr(self._table, name)

    def resolve(self, key: str, detail: str = ""):
        info = self._table.resolve(key, detail)
        return dataclasses.replace(info, text=self._handler.alarm_text(info.text, detail))

    def program_alarm(self, number: int, message: str, base: int):
        return self._table.program_alarm(number, message, base)

    def has_key(self, key: str) -> bool:
        return self._table.has_key(key)


class WordValue:
    """Duck type of the FANUC WordValue for providers written against the core value helpers."""

    def __init__(self, letter: str, value: float, point: bool = True, text: str = "", from_variable: bool = False):
        self.letter = letter
        self.value = value
        self.point = point
        self.text = text
        self.from_variable = from_variable


# ====================================================================== geometry helpers

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(direction, angle: float) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    x, y, z = d
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def plane_rotation(plane: str, angle_rad: float) -> np.ndarray:
    """Rotation by ``angle`` about the normal of ``plane`` (G17 about +Z, G18 about +Y, G19 about +X):
    positive turns the first plane axis toward the second."""
    return {"XY": rot_z, "ZX": rot_y, "YZ": rot_x}[plane](angle_rad)


def rpy_matrix(a_deg: float, b_deg: float, c_deg: float) -> np.ndarray:
    """R = Rz(c) Ry(b) Rx(a): rotations about Z, then the new Y, then the new X."""
    return rot_z(math.radians(c_deg)) @ rot_y(math.radians(b_deg)) @ rot_x(math.radians(a_deg))


def rpy_angles(r) -> tuple[float, float, float]:
    """(a, b, c) in degrees with R = Rz(c) Ry(b) Rx(a), b in [-90, 90]."""
    m = np.asarray(r, dtype=float).reshape(3, 3)
    b = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(b)) > 1e-12:
        a = math.atan2(m[2, 1], m[2, 2])
        c = math.atan2(m[1, 0], m[0, 0])
    else:
        a = 0.0
        c = math.atan2(-m[0, 1], m[1, 1])
    return _clean_deg(math.degrees(a)), _clean_deg(math.degrees(b)), _clean_deg(math.degrees(c))


def _clean_deg(value: float) -> float:
    rounded = round(value, 9)
    return 0.0 if rounded == 0.0 else rounded


def circle_through(p0, p1, p2):
    """Centre of the circle through three plane points, or None when they are collinear."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    scale = max(abs(bx - ax), abs(by - ay), abs(cx - ax), abs(cy - ay), 1e-300)
    if abs(d) <= 1e-12 * scale * scale:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy


def tangent_arc_centre(start, end, tangent):
    """(centre, direction) of the arc that leaves ``start`` along ``tangent`` and ends at ``end``;
    None when ``end`` lies on the tangent line (no arc)."""
    tx, ty = tangent
    norm = math.hypot(tx, ty)
    if norm == 0.0:
        return None
    tx, ty = tx / norm, ty / norm
    dx, dy = end[0] - start[0], end[1] - start[1]
    left = (-ty, tx)
    side = dx * left[0] + dy * left[1]
    chord2 = dx * dx + dy * dy
    if chord2 == 0.0 or abs(side) <= 1e-12 * math.sqrt(chord2):
        return None
    rho = chord2 / (2.0 * side)          # signed: > 0 centre to the left (counter-clockwise)
    centre = (start[0] + rho * left[0], start[1] + rho * left[1])
    return centre, (1 if rho > 0.0 else -1)


def arc_end_tangent(centre, end, direction):
    """Unit tangent at ``end`` of an arc about ``centre`` turning in ``direction`` (+1 CCW)."""
    rx, ry = end[0] - centre[0], end[1] - centre[1]
    r = math.hypot(rx, ry)
    if r == 0.0:
        return (0.0, 0.0)
    return (-ry / r * direction, rx / r * direction)


def wrap_deg(value: float) -> float:
    out = math.fmod(value, 360.0)
    return out + 360.0 if out < 0.0 else out


# ====================================================================== the handler base

class NativeHandler:
    """Core interface shared by the Siemens and Heidenhain handlers."""

    dialect_key = ""                    # "siemens840d" | "heidenhain": prefix of hook keys
    default_plane = "G17"

    def __init__(self, ex, dialect):
        self.ex = ex
        bound = dataclasses.replace(dialect, alarms=BoundAlarms(dialect.alarms, self))
        ex.dialect = bound
        self.d = bound
        self.s = ex.s
        self.machine = ex.machine
        self.p = ex.p
        self.T = dialect.variant == "T"
        self._alarm_params: list[str] | None = None
        self._names = None

    # ------------------------------------------------------------------ alarm texts
    def alarm_text(self, text: str, detail: str) -> str:
        params = list(self._alarm_params) if self._alarm_params else ([detail] if detail else [])
        self._alarm_params = None
        if "%" not in text:
            return text
        multi = len(self.machine.paths) > 1
        channel = f"Channel {self.ex.ctx.path + 1}: " if multi else ""
        text = text.replace("[Channel %1: ] ", channel).replace("[Channel %1: ]", channel.strip())
        match = re.search(r"Block\s*%(\d)", text)
        if match:
            text = text.replace(f"%{match.group(1)}", self.block_label(), 1)
        for number in sorted(set(re.findall(r"%(\d)", text))):
            value = params.pop(0) if params else "?"
            text = text.replace(f"%{number}", value)
        return text

    def block_label(self) -> str:
        ref = self.ex.line_ref()
        if ref.n_number is not None:
            return f"N{ref.n_number}"
        label = getattr(self.ex.block.extra, "label", "") if self.ex.block is not None and self.ex.block.extra else ""
        return label or f"line {ref.line}"

    def execute(self, block) -> None:
        self._alarm_params = None
        try:
            self.run_block(block)
        except ControllerAlarm as alarm:
            params = getattr(alarm, "params", None)
            self._alarm_params = list(params) if params else None
            raise

    # ------------------------------------------------------------------ program names
    def program_names(self) -> dict[str, int]:
        """Program name (upper case) -> index in the store (immutable context, cached)."""
        if self._names is None:
            names: dict[str, int] = {}
            for index, program in enumerate(self.ex.store.programs):
                names.setdefault(program.name.upper(), index)
            self._names = names
        return self._names

    def find_program(self, number) -> int:
        key = str(number).upper()
        index = self.program_names().get(key)
        if index is None:
            raise Alarm("program_not_found", key, key)
        return index

    # ------------------------------------------------------------------ state set-up
    def init_state(self) -> None:
        ex, s = self.ex, self.s
        s.modal = dict(self.d.power_on)
        for key, code in (self.p.get("power_on") or {}).items():
            s.modal[str(key)] = code
        s.units = self.p.get("power_on_units") or "mm"
        exact = self.machine.corner_default == "exact_stop"
        s.smoothing = {"corner": "exact_stop" if exact else "blend", "tolerance_m": 0.0, "mode_code": "",
                       "lookahead": True}
        s.pos = {a.letter: a.home for a in ex.axes}
        s.mach = dict(s.pos)
        s.applied = {a.letter: 0 for a in ex.axes}
        for key in self.offset_keys():
            s.offsets[key] = [0] * len(ex.axes)
        for key, values in (ex.ctx.work_offsets_m or {}).items():
            s.offsets[self.offset_key(str(key))] = self.offset_row(values)
        for tool in ex.ctx.tools:
            self.set_tool_offsets(tool)
        s.pumps = {pid: False for pid in self.machine.pumps}
        s.nozzles = {nid: 1 for nid in self.machine.nozzles}
        s.variables = {}
        s.sys["frame"] = identity_frame()
        self.init_dialect()
        s.pos = ex.program_from_machine(s.mach)

    def init_dialect(self) -> None:
        return None

    def offset_keys(self) -> list[str]:
        return []

    def offset_key(self, key: str) -> str:
        return key

    def offset_row(self, values) -> list[int]:
        ex = self.ex
        row = [0] * len(ex.axes)
        items = ([(str(k), v) for k, v in values.items()] if isinstance(values, dict)
                 else [(ex.axes[i].letter, v) for i, v in enumerate(list(values)[:len(ex.axes)])])
        for letter, value in items:
            if letter in ex.by_letter:
                axis = ex.by_letter[letter]
                row[ex.letters.index(letter)] = int(round_half_away(float(value) / axis.quantum))
        return row

    def set_tool_offsets(self, tool) -> None:
        """Offsets of one tool from the project (SI): mill length/radius (+ wear); lathe X/Z (radius values),
        nose radius, tip and wear. A lathe tool's ``radius_m`` is kept as half its width (grooving)."""
        inc = self.machine.lin_increment_m
        row = [0] * ROW_SIZE

        def counts(value) -> int:
            return int(round_half_away(float(value or 0.0) / inc))

        if self.T:
            row[ROW_X] = counts(getattr(tool, "x_m", 0.0))
            row[ROW_Z] = counts(getattr(tool, "z_m", 0.0))
            row[ROW_NOSE] = counts(getattr(tool, "nose_radius_m", 0.0))
            row[ROW_TIP] = int(getattr(tool, "tip", 0) or 0)
            row[ROW_X_WEAR] = counts(getattr(tool, "x_wear_m", 0.0))
            row[ROW_Z_WEAR] = counts(getattr(tool, "z_wear_m", 0.0))
            row[ROW_NOSE_WEAR] = counts(getattr(tool, "radius_wear_m", 0.0))
            width = 2.0 * float(getattr(tool, "radius_m", 0.0) or 0.0)
            if width > 0.0:
                self.s.sys.setdefault("tool_width", {})[str(int(tool.number))] = counts(width)
        else:
            row[ROW_LEN] = counts(getattr(tool, "length_m", 0.0))
            row[ROW_LEN_WEAR] = counts(getattr(tool, "length_wear_m", 0.0))
            row[ROW_RAD] = counts(getattr(tool, "radius_m", 0.0))
            row[ROW_RAD_WEAR] = counts(getattr(tool, "radius_wear_m", 0.0))
        self.s.tool_offsets[str(int(tool.number))] = row

    # ------------------------------------------------------------------ queries used by the core
    def plane(self) -> str:
        return PLANE_OF.get(self.s.modal.get("plane", self.default_plane), "XY")

    def feed_mode(self) -> str:
        return self.s.modal.get("feed_mode", "per_min")

    def absolute(self) -> bool:
        return self.s.modal.get("distance", "G90") == "G90"

    def active_offset(self) -> list[int] | None:
        """The offset row in effect (copied when the offset was selected)."""
        return self.s.sys.get("active_offset")

    def tool_axis_letter(self) -> str:
        return self.s.tool_axis_letter or "Z"

    def tool_offset_counts(self, letter: str) -> int:
        row = self.active_offset()
        if not row:
            return 0
        if self.T:
            if letter == "X":
                return row[ROW_X] + row[ROW_X_WEAR]
            if letter == "Z":
                return row[ROW_Z] + row[ROW_Z_WEAR]
            return 0
        if letter != self.tool_axis_letter():
            return 0
        return row[ROW_LEN] + row[ROW_LEN_WEAR] + int(self.s.sys.get("delta_length", 0))

    def comp_radius(self) -> float:
        row = self.active_offset()
        if not row:
            return 0.0
        inc = self.machine.lin_increment_m
        if self.T:
            return (row[ROW_NOSE] + row[ROW_NOSE_WEAR]) * inc
        return (row[ROW_RAD] + row[ROW_RAD_WEAR] + int(self.s.sys.get("delta_radius", 0))) * inc

    def modal_group_value(self, group: int) -> float:
        code = self.s.modal.get(str(group))
        try:
            return float(str(code).lstrip("G"))
        except (TypeError, ValueError):
            return 0.0

    def modal_snapshot(self) -> dict:
        s, ex = self.s, self.ex
        offsets = ex.offset_counts(letters_tool=False)
        snap = {
            "plane": self.plane(), "units": s.units, "distance": "abs" if self.absolute() else "inc",
            "feed_mode": self.feed_mode(), "work_offset": s.wcs,
            "work_offset_m": [offsets[i] * a.quantum for i, a in enumerate(ex.axes)],
            "tool": s.tool, "h": s.h_number, "d": s.d_number,
            "length_offset_m": self.tool_offset_counts(self.tool_axis_letter()) * self.machine.lin_increment_m,
            "radius_offset_m": self.comp_radius(), "comp": s.modal.get("7", "G40"),
            "motion": s.modal.get("motion", "G0"), "cycle": s.sys.get("cycle_name"),
            "css": s.spindle.css, "exact_stop": s.smoothing.get("corner") == "exact_stop",
            "dialect": self.d.name,
        }
        if self.T:
            snap["offset_number"] = s.offset_number
        snap.update(self.snapshot_extra())
        return snap

    def snapshot_extra(self) -> dict:
        return {}

    def apply_mode(self, step) -> None:
        for key, value in step.data.items():
            self.s.sys[key] = value
        self.ex.emit(Command(Kind.MODE, self.ex.line_ref(), modal=dict(step.data)))

    def deferred_m(self, step) -> None:
        for code in step.data.get("m", []):
            self.m_function(int(code), {})

    # value helpers for providers written against the FANUC-style interface
    def linear_increment_units(self) -> float:
        mm = self.machine.lin_increment_m * 1000.0
        return mm / 25.4 if self.s.units == "inch" else mm

    def unit_m(self) -> float:
        return FEED_SCALE["inch" if self.s.units == "inch" else "mm"]

    def lin(self, w) -> float:
        return float(w.value) * self.unit_m()

    def rot(self, w) -> float:
        return math.radians(float(w.value))

    def integer(self, w) -> int:
        return int(round_half_away(float(w.value)))

    def feed_value(self, w, mode: str) -> tuple[float, float]:
        value = float(w.value)
        if mode == "per_min":
            return value * self.unit_m() / 60.0, value
        if mode == "per_rev":
            return value * self.unit_m(), value
        return value / 60.0, value

    def evaluate(self, words):
        raise Alarm("unsupported_function", f"{self.d.name}: FANUC word evaluation is not part of this dialect")

    def offset_changed(self) -> None:
        """A new tool offset is selected: it takes effect with the next motion of each axis concerned (the
        program position is re-read through the new offset, the machine position stays)."""
        ex, s = self.ex, self.s
        applied = {letter: self.tool_offset_counts(letter) for letter in ex.letters}
        s.pos = ex.program_from_machine(s.mach, applied)

    # ================================================================== emission
    def spindle_id(self) -> str:
        return self.machine.path_spindle[self.ex.ctx.path]

    def spindle_max(self, spindle_id: str) -> float:
        spindle = self.machine.spindles.get(spindle_id)
        return spindle.max_rad_s if spindle is not None else 0.0

    def emit_spindle(self, orient: float | None = None) -> None:
        ex, s = self.ex, self.s
        spindle_id = self.spindle_id()
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

    def coolant(self, changes: dict[str, bool], code: str) -> None:
        ex, s = self.ex, self.s
        if not changes:
            raise Alarm("machine_feature_missing", f"{code}: no such coolant on this machine")
        s.pumps.update(changes)
        ex.emit(Command(Kind.COOLANT, ex.line_ref(), coolant={"pumps": dict(s.pumps), "nozzles": dict(s.nozzles),
                                                             "code": code}))

    def coolant_on(self, kinds: tuple[str, ...], code: str) -> None:
        self.coolant({pid: True for pid, kind in self.machine.pumps.items() if kind in kinds}, code)

    def coolant_off(self, code: str) -> None:
        self.coolant({pid: False for pid in self.machine.pumps}, code)

    def tool_slot(self, number: int) -> int:
        table = self.ex.ctx.tool_table
        if table is None:
            return -1
        slot = table.by_number(number, self.ex.ctx.path)
        if slot is None:
            self.ex.note("warning", f"T{number} is not in the project's tool list")
            return -1
        return slot.slot

    def tool_life_select(self, number: int) -> int:
        provider = hooks.TOOL_LIFE.get("default")
        if provider is not None:
            return int(provider.select(self.ex, number))
        if number in self.s.expired_tools:
            self.ex.note("warning", f"T{number} called after its tool life expired (no tool life management)")
        return number

    def emit_tool_change(self, number: int, extra: dict | None = None) -> None:
        ex, s = self.ex, self.s
        ex.flush_comp("tool change")
        slot = self.tool_slot(number) if number > 0 else -1
        tool = {"number": number, "pocket": number, "slot": slot, "length_offset": s.d_number,
                "radius_offset": s.d_number}
        if self.T and self.machine.turrets:
            tool.update({"station": number, "turret": self.machine.turrets[0]["id"]})
        tool.update(extra or {})
        ex.emit(Command(Kind.TOOL_CHANGE, ex.line_ref(), tool=tool, modal=dict(ex.snapshot())))
        s.tool = number
        s.preselect = None

    def program_stop(self, code: str, message: str = "") -> None:
        ex = self.ex
        ex.flush_comp("program stop")
        ex.emit(Command(Kind.STOP, ex.line_ref(), stop=code, message=message))
        ex.wait("stop", reason=code)

    def message_stop(self, text: str, code: str) -> None:
        ex = self.ex
        ex.flush_comp("program stop")
        ex.emit(Command(Kind.MESSAGE, ex.line_ref(), stop=code, message=text))
        ex.wait("stop", reason=code)

    def preprocessing_stop(self, code: str) -> None:
        """The interpreter waits until motion has executed every buffered block (Siemens STOPRE,
        Heidenhain FN 20 WAIT FOR SYNC)."""
        ex = self.ex
        ex.flush_comp("buffer stop")
        ex.emit(Command(Kind.SYNC, ex.line_ref(), sync={"reason": "exact_stop_check", "code": code}))
        ex.wait("sync", code=code)

    def path_wait(self, marker: int, paths: list[int], code: str) -> None:
        ex = self.ex
        ex.flush_comp("path wait")
        wait_id = ex.root.wait_seq
        ex.root.wait_seq += 1
        ex.emit(Command(Kind.SYNC, ex.line_ref(), sync={"reason": "path_wait", "code": code, "paths": list(paths),
                                                        "id": wait_id}))
        ex.wait("path_wait", code=marker, paths=list(paths))

    # ================================================================== motion helpers
    def check_travel(self, target: dict[str, int]) -> None:
        """Software limit switches at block preparation (Siemens 10720, TNC 1A1-0010): the block's end point
        in machine coordinates against each axis' travel. Skipped under a TCP / tilted-plane stage, whose
        provider computes the machine axes."""
        ex, s = self.ex, self.s
        if s.tcp:
            return
        position = dict(s.pos)
        position.update(target)
        work = ex.work_from_program(position)
        offsets = ex.offset_counts()
        for i, axis in enumerate(ex.axes):
            if axis.travel is None or (axis.kind == "rotary" and axis.continuous):
                continue
            value = (int(round_half_away(work[i] / axis.quantum)) + offsets[i]) * axis.quantum
            lo, hi = axis.travel
            tolerance = 0.5 * axis.quantum
            if value > hi + tolerance or value < lo - tolerance:
                raise Alarm("software_limit", f"{axis.name} {value * (1e3 if axis.kind == 'linear' else 57.29577951308232):.3f} "
                                              f"outside {lo:g}..{hi:g}", axis.name, "+" if value > hi else "-")

    def unequal_scale(self, plane: str | None = None) -> bool:
        """True when the active scale factors of the two axes of ``plane`` differ: a circle there would be
        an ellipse after scaling, which circular interpolation cannot follow."""
        scale = self.s.scale
        if not scale:
            return False
        a, b, _ = PLANES[plane or self.plane()]
        factors = scale.get("factor", {})
        return abs(abs(float(factors.get(a, 1.0))) - abs(float(factors.get(b, 1.0)))) > 1e-12

    def check_circle_scale(self, plane: str | None = None, key: str = "arc_under_scaling") -> None:
        if self.unequal_scale(plane):
            raise Alarm(key, "circle with different scale factors in its plane (the scaled contour is an ellipse)")

    def check_steps_scale(self, steps, key: str = "arc_under_scaling") -> None:
        """Cycle expansions: their circles must not meet unequal scale factors either."""
        for step in steps:
            if step.kind == "arc":
                self.check_circle_scale(step.plane or None, key)

    def arc_by_centre(self, target: dict[str, int], direction: int, centre: tuple[float, float], feed, *,
                      plane: str | None = None, turns: int = 0, tolerance: tuple[float, float] = (0.0, 0.0),
                      alarm_key: str = "arc_end_point", comp: bool = True, modal: dict | None = None,
                      corner: dict | None = None):
        """G2/G3 to ``target`` (program counts) about the program-frame ``centre`` (SI, plane axes).

        The end point must lie on the start circle within ``max(tol_const, tol_factor * r)``; within
        it the centre is moved onto the perpendicular bisector of start and end (the nearest point
        equidistant from both), so the arc is a true circle through both points."""
        ex, s = self.ex, self.s
        plane = plane or self.plane()
        a, b, _ = PLANES[plane]
        if a not in ex.by_letter or b not in ex.by_letter:
            raise Alarm("illegal_plane", f"plane {plane}", a if a not in ex.by_letter else b)
        self.check_circle_scale(plane)
        end = dict(s.pos)
        end.update(target)
        sa, sb = ex.si(a, s.pos[a]), ex.si(b, s.pos[b])
        ea, eb = ex.si(a, end[a]), ex.si(b, end[b])
        ca, cb = centre
        r0 = math.hypot(sa - ca, sb - cb)
        if r0 == 0.0:
            raise Alarm(alarm_key, "centre on the start point")
        same = s.pos[a] == end[a] and s.pos[b] == end[b]
        if not same:
            r1 = math.hypot(ea - ca, eb - cb)
            limit = max(tolerance[0], tolerance[1] * r0)
            if abs(r1 - r0) > limit:
                raise Alarm(alarm_key, f"start radius {r0 * 1e3:.4f} mm, end radius {r1 * 1e3:.4f} mm "
                                       f"(tolerance {limit * 1e3:.4f} mm)")
            if r1 != r0:
                mx, my = 0.5 * (sa + ea), 0.5 * (sb + eb)
                ux, uy = ea - sa, eb - sb
                length = math.hypot(ux, uy)
                ux, uy = ux / length, uy / length
                along = (ca - mx) * ux + (cb - my) * uy
                ca, cb = ca - along * ux, cb - along * uy
        offset = {a: ca - sa, b: cb - sb}
        self.check_travel(target)
        return ex.arc_move(target, direction, offset, None, feed, comp=comp, turns=turns, plane=plane, modal=modal,
                           corner=corner)

    # ================================================================== frames onto the core pipeline
    def apply_frame(self, rotation9, translation_si, scale3, mirror3, rotary_shift: dict | None = None) -> None:
        """Set the core's mirror -> scale -> feature stages to ``x_parent = T + R S M x`` (T: XYZ in SI,
        R a rotation (row-major 9), S per-axis factors, M per-axis mirror flags), and translations of
        rotary axes as local shifts; then the program position follows."""
        ex, s = self.ex, self.s
        letters = [l for l in "XYZ" if l in ex.by_letter]
        s.mirror = {l: 0 for i, l in enumerate("XYZ") if mirror3[i] and l in ex.by_letter}
        if any(abs(float(f) - 1.0) > 0.0 for f in scale3):
            s.scale = {"centre": {l: 0 for l in letters},
                       "factor": {l: float(scale3["XYZ".index(l)]) for l in letters}}
        else:
            s.scale = None
        s.rotation = None
        rotation = [float(v) for v in rotation9]
        origin = [ex.counts(l, float(translation_si[i])) if l in ex.by_letter else 0 for i, l in enumerate("XYZ")]
        if any(origin) or any(abs(rotation[k] - IDENTITY9[k]) > 1e-15 for k in range(9)):
            s.feature = {"origin": origin, "matrix": rotation}
        else:
            s.feature = None
        for axis in ex.axes:
            if axis.kind == "rotary":
                shift = (rotary_shift or {}).get(axis.letter, 0.0)
                if shift:
                    s.g52[axis.letter] = ex.counts(axis.letter, float(shift))
                else:
                    s.g52.pop(axis.letter, None)
        ex.invalidate_transform()
        ex.resync_program_position()

    # ================================================================== rotary table kinematics
    def table_solutions(self, normal_part) -> list[dict[str, float]]:
        """Rotary positions (rad, by machine axis name) of a two-axis table that turn the part-frame
        direction ``normal_part`` onto the machine tool axis. Product of exponentials of the table
        links: R = R_1(q1) R_2(q2), solved in closed form (Paden-Kahan subproblem 2)."""
        machine = self.machine
        links = machine.work_rotaries
        if len(links) != 2 or machine.tool_rotaries:
            raise Alarm("transform_unavailable", "tilted plane / TCP needs two rotary table axes")
        a1 = np.array(links[0].direction, dtype=float)
        a2 = np.array(links[1].direction, dtype=float)
        n = np.asarray(normal_part, dtype=float)
        n = n / np.linalg.norm(n)
        t = np.array(machine.tool_axis, dtype=float)
        t = t / np.linalg.norm(t)
        coeff_a = float(n @ a1 - (a2 @ n) * (a2 @ a1))
        coeff_b = float(np.cross(a2, n) @ a1)
        coeff_c = float((a2 @ n) * (a2 @ a1))
        target = float(t @ a1) - coeff_c
        amplitude = math.hypot(coeff_a, coeff_b)
        candidates: list[float] = []
        current = self.rotary_values()
        if amplitude < 1e-12:
            if abs(target) > 1e-9:
                return []
            candidates = [current.get(links[1].axis, 0.0)]
        else:
            ratio = target / amplitude
            if abs(ratio) > 1.0 + 1e-12:
                return []
            ratio = max(-1.0, min(1.0, ratio))
            phi = math.atan2(coeff_b, coeff_a)
            delta = math.acos(ratio)
            candidates = [phi + delta, phi - delta] if delta > 1e-12 else [phi]
        out = []
        for q2 in candidates:
            u = rot_axis(a2, q2) @ n
            u_perp = u - (u @ a1) * a1
            t_perp = t - (t @ a1) * a1
            if np.linalg.norm(u_perp) < 1e-12 or np.linalg.norm(t_perp) < 1e-12:
                q1 = current.get(links[0].axis, 0.0)
            else:
                q1 = math.atan2(float(np.cross(u_perp, t_perp) @ a1), float(u_perp @ t_perp))
            out.append({links[0].axis: q1, links[1].axis: q2})
        return out

    def rotary_values(self) -> dict[str, float]:
        """Current commanded rotary positions (rad, machine names)."""
        ex = self.ex
        return {a.name: ex.s.mach.get(a.letter, 0) * a.quantum for a in ex.axes if a.kind == "rotary"}

    def place_rotary(self, name: str, value: float) -> float:
        """A rotary solution brought into the axis' range: continuous axes to the turn nearest the current
        position, limited axes into their travel (None when impossible)."""
        axis = next((a for a in self.ex.axes if a.name == name), None)
        if axis is None:
            raise Alarm("transform_unavailable", f"rotary axis {name} is not on this path")
        current = self.ex.s.mach.get(axis.letter, 0) * axis.quantum
        if axis.continuous:
            turns = round((current - value) / (2.0 * math.pi))
            return value + 2.0 * math.pi * turns
        if axis.travel is not None:
            lo, hi = axis.travel
            for shift in (0.0, 2.0 * math.pi, -2.0 * math.pi):
                if lo - 1e-9 <= value + shift <= hi + 1e-9:
                    return value + shift
            return None
        return value

    def choose_table_solution(self, normal_part, master_sign: int = 0, prefer: str = "") -> dict[str, float]:
        """One reachable solution: ``master_sign`` +1/-1 picks the sign of the first rotary axis (Heidenhain
        SEQ+/SEQ-, Siemens _DIR), otherwise the one with the smallest rotary motion."""
        solutions = []
        for raw in self.table_solutions(normal_part):
            placed = {}
            for name, value in raw.items():
                value = self.place_rotary(name, value)
                if value is None:
                    break
                placed[name] = value
            else:
                solutions.append(placed)
        if not solutions:
            raise Alarm("tilt_no_solution", "no rotary position reaches the plane")
        master = self.machine.work_rotaries[0].axis
        if master_sign:
            signed = [sol for sol in solutions if master_sign * sol[master] >= -1e-12]
            solutions = signed or solutions
        current = self.rotary_values()
        return min(solutions, key=lambda sol: (sum(abs(sol[k] - current.get(k, 0.0)) for k in sorted(sol)),
                                               [sol[k] for k in sorted(sol)]))


def identity_frame() -> dict:
    """A frame in component form: translation T (SI, XYZ), rotation R (row-major 9), scale S, mirror M,
    and translations of rotary axes (SI by letter)."""
    return {"T": [0.0, 0.0, 0.0], "R": list(IDENTITY9), "S": [1.0, 1.0, 1.0], "M": [0, 0, 0], "TR": {}}


def frame_is_identity(frame: dict) -> bool:
    return (not any(frame["T"]) and frame["R"] == IDENTITY9 and frame["S"] == [1.0, 1.0, 1.0]
            and not any(frame["M"]) and not any(frame["TR"].values()))


def frame_chain(a: dict, b: dict) -> dict:
    """``b`` applied in the coordinate system of ``a`` (Siemens component rules: ATRANS by R S M of a,
    AROT, ASCALE, AMIRROR)."""
    out = copy_frame(a)
    linear = np.asarray(a["R"], dtype=float).reshape(3, 3) @ np.diag(
        [a["S"][i] * (-1.0 if a["M"][i] else 1.0) for i in range(3)])
    shift = linear @ np.asarray(b["T"], dtype=float)
    out["T"] = [float(a["T"][i] + shift[i]) for i in range(3)]
    rotation = np.asarray(a["R"], dtype=float).reshape(3, 3) @ np.asarray(b["R"], dtype=float).reshape(3, 3)
    out["R"] = clean_matrix(rotation)
    out["S"] = [float(a["S"][i] * b["S"][i]) for i in range(3)]
    out["M"] = [int(bool(a["M"][i]) != bool(b["M"][i])) for i in range(3)]
    rotary = dict(a["TR"])
    for letter, value in b["TR"].items():
        rotary[letter] = float(rotary.get(letter, 0.0) + value)
    out["TR"] = rotary
    return out


def copy_frame(frame: dict) -> dict:
    return {"T": list(frame["T"]), "R": list(frame["R"]), "S": list(frame["S"]), "M": list(frame["M"]),
            "TR": dict(frame["TR"])}


def clean_matrix(matrix) -> list[float]:
    """Row-major 9 floats; entries within 1e-15 of 0 or +-1 snapped (exact for multiples of 90 deg)."""
    out = []
    for value in np.asarray(matrix, dtype=float).reshape(9):
        v = float(value)
        for snap in (0.0, 1.0, -1.0):
            if abs(v - snap) < 1e-15:
                v = snap
        out.append(v)
    return out


def register_tcp_stages() -> None:
    """Final-stage transforms for TCP and tilted planes on table-type machines (the Haas stage TableTcp:
    machine axes by the inverse kinematics of the table chain, Space.TIP commands)."""
    from ..haas import TableTcp

    hooks.register_transform("siemens840d:TRAORI", TableTcp("tcpc"), replace=True)
    hooks.register_transform("siemens840d:CYCLE800", TableTcp("dwo"), replace=True)
    hooks.register_transform("heidenhain:TCPM", TableTcp("tcpc"), replace=True)
    hooks.register_transform("heidenhain:PLANE", TableTcp("dwo"), replace=True)


__all__ = ["Alarm", "BoundAlarms", "NativeHandler", "WordValue", "identity_frame", "frame_chain", "copy_frame",
           "frame_is_identity", "clean_matrix", "rpy_matrix", "rpy_angles", "plane_rotation", "rot_axis", "rot_x",
           "rot_y", "rot_z", "circle_through", "tangent_arc_centre", "arc_end_tangent", "wrap_deg",
           "register_tcp_stages", "PLANE_CODE", "TOOL_AXIS_OF_PLANE", "ROW_SIZE", "ROW_LEN", "ROW_LEN_WEAR",
           "ROW_RAD", "ROW_RAD_WEAR", "ROW_X", "ROW_Z", "ROW_NOSE", "ROW_NOSE_WEAR", "ROW_TIP", "ROW_X_WEAR",
           "ROW_Z_WEAR"]
