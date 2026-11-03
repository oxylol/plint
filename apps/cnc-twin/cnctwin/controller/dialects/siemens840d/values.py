"""SINUMERIK 840D sl values: expressions, R parameters, local user data (DEF), system variables, frames.

Types: REAL and INT are Python floats (INT rounded half away from zero on assignment), BOOL is 1.0 /
0.0, STRING and AXIS are str, FRAME is a component dict (kit.identity_frame). Trigonometry is in
degrees; ``ATAN2(a, b)`` is the angle of the vector (b, a); ``POT(x)`` = x^2; ``TRUNC`` toward zero;
``ROUND`` half away from zero; ``DIV`` integer division (toward zero); ``MOD`` the remainder with the
sign of the dividend. Operator priority (highest first): NOT B_NOT, * / DIV MOD, + -, B_AND, B_XOR,
B_OR, AND, XOR, OR, << (string concatenation), == <> > < >= <= (so a combined condition needs
brackets, as on the control), ':' (frame chain). Comparisons are exact.

A block is evaluated left to right against an overlay (``Env``): assignments are visible to the
items after them, and nothing reaches the path state until the block has been read completely
(``commit``). A main-run variable ($AA_IW, $AA_IM) raises NeedLive before anything is committed,
so the core can stop the buffer and run the block again with the encoder positions.

Storage (all in PathState, checkpointed): R parameters in ``variables["R"]``; local user data in
``sys["lud"]``, one scope per call level ``{"sig": [...], "vars": {NAME: [type, value, ref]}}``; a
VAR parameter's ``ref`` names the caller's variable it stands for.
"""

from __future__ import annotations

import datetime as _dt
import math

from ...expr import round_half_away
from ...pipeline import PLANES
from .kit import (Alarm, clean_matrix, copy_frame, frame_chain, identity_frame, rpy_angles, rpy_matrix)

CONSTANTS = {"_OFF": 0.0, "_FINISH": 1.0, "_SEMIFIN": 2.0, "_ROUGH": 3.0, "_ORI_FINISH": 11.0,
             "_ORI_SEMIFIN": 12.0, "_ORI_ROUGH": 13.0, "_TOP_SURFACE_SMOOTH_OFF": 1000000.0,
             "_TOP_SURFACE_SMOOTH_ON": 2000000.0}
NOT_MODELLED_PREFIX = ("$MN_", "$MC_", "$MA_", "$SN_", "$SC_", "$SA_", "$AC_", "$AA_", "$A_", "$P_", "$TC_",
                       "$VA_", "$VC_", "$R")
TOOL_DATA = {"$TC_DP3": "len", "$TC_DP4": "len2", "$TC_DP5": "len3", "$TC_DP6": "rad", "$TC_DP12": "len_wear",
             "$TC_DP15": "rad_wear"}


def _number(value, what: str = "value") -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    raise Alarm("type_mismatch", f"{what} is not a number", str(value))


def truthy(value) -> bool:
    if isinstance(value, str):
        raise Alarm("type_mismatch", "a string is not a condition", value)
    if isinstance(value, dict):
        raise Alarm("type_mismatch", "a frame is not a condition", "FRAME")
    return _number(value) != 0.0


def convert(kind: str, value, length: int = 0):
    """A value assigned to a variable of type ``kind`` (DEF types)."""
    if kind == "INT":
        return float(round_half_away(_number(value)))
    if kind == "REAL":
        return _number(value)
    if kind == "BOOL":
        return 1.0 if truthy(value) else 0.0
    if kind == "CHAR":
        if isinstance(value, str):
            return float(ord(value[0])) if value else 0.0
        return float(round_half_away(_number(value)))
    if kind == "STRING":
        if not isinstance(value, str):
            raise Alarm("illegal_type_conversion", "a number assigned to a STRING (use << or TOSTR)")
        if length and len(value) > length:
            raise Alarm("string_too_long", f"{len(value)} characters for STRING[{length}]", value[:20])
        return value
    if kind == "AXIS":
        if not isinstance(value, str):
            raise Alarm("illegal_type_conversion", "AXIS needs an axis name")
        return value
    if kind == "FRAME":
        if not isinstance(value, dict):
            raise Alarm("illegal_type_conversion", "FRAME needs a frame value")
        return copy_frame(value)
    raise Alarm("syntax", f"data type {kind}", kind)


def default_value(kind: str):
    if kind in ("STRING", "AXIS"):
        return ""
    if kind == "FRAME":
        return identity_frame()
    return 0.0


class Env:
    """Variables of one path as one block sees them (reads through pending writes)."""

    def __init__(self, handler):
        self.h = handler
        self.ex = handler.ex
        self.s = handler.s
        self.pending: dict = {}
        self.sys_writes: list = []

    # ------------------------------------------------------------------ scopes
    def scope(self) -> dict:
        return self.h.scope()

    def lookup(self, name: str):
        """(key, type, length) of a local variable, following VAR references; None if undefined."""
        scope = self.scope()
        entry = scope["vars"].get(name)
        if entry is None:
            return None
        if entry[2] is not None:
            return self.resolve_ref(entry[2]) + (entry[0], entry[3] if len(entry) > 3 else 0)
        return ("L", len(self.s.frames) - 1, name), entry[0], entry[3] if len(entry) > 3 else 0

    def resolve_ref(self, ref) -> tuple:
        if ref[0] == "R":
            return (("R", int(ref[1])),)
        depth, name = int(ref[1]), ref[2]
        scopes = self.s.sys.get("lud", [])
        entry = scopes[depth]["vars"].get(name) if depth < len(scopes) else None
        if entry is not None and entry[2] is not None:
            return self.resolve_ref(entry[2])
        return (("L", depth, name),)

    def get(self, key):
        if key in self.pending:
            return self.pending[key]
        if key[0] == "R":
            return self.s.variables["R"][key[1]]
        scopes = self.s.sys["lud"]
        return scopes[key[1]]["vars"][key[2]][1]

    def put(self, key, value) -> None:
        self.pending[key] = value

    def commit(self) -> None:
        for key, value in self.pending.items():
            if key[0] == "R":
                self.s.variables["R"][key[1]] = value
            else:
                self.s.sys["lud"][key[1]]["vars"][key[2]][1] = value
        self.pending = {}
        writes, self.sys_writes = self.sys_writes, []
        for name, index, value in writes:
            self.h.write_sys(name, index, value)

    # ------------------------------------------------------------------ evaluation
    def r_key(self, node) -> tuple:
        index = _number(self.value(node), "R index")
        n = int(round_half_away(index))
        if n != index or not 0 <= n < len(self.s.variables["R"]):
            raise Alarm("undefined_name", f"R{index:g}", f"R{index:g}")
        return ("R", n)

    def value(self, node):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "str":
            return node[1]
        if kind == "r":
            return self.get(self.r_key(node[1]))
        if kind == "name":
            return self.name_value(node[1])
        if kind == "sys":
            return self.h.read_sys(node[1], tuple(self.index_value(i) for i in node[2]), self)
        if kind == "idx":
            return self.indexed(node[1], node[2])
        if kind == "neg":
            return -_number(self.value(node[1]))
        if kind == "not":
            return 0.0 if truthy(self.value(node[1])) else 1.0
        if kind == "bnot":
            return float(~int(round_half_away(_number(self.value(node[1])))))
        if kind == "bin":
            return self.binary(node[1], node[2], node[3])
        if kind == "fn":
            return self.function(node[1], node[2])
        if kind == "const":
            return node[1]
        raise Alarm("syntax", f"expression {kind}", kind)

    def number(self, node, what: str = "value") -> float:
        return _number(self.value(node), what)

    def index_value(self, node):
        if node is None:
            return None
        if node[0] == "const":
            return node[1]
        return self.value(node)

    def name_value(self, name: str):
        found = self.lookup(name)
        if found is not None:
            return self.get(found[0])
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name in self.ex.by_letter:
            return name
        raise Alarm("undefined_name", f"name {name}", name)

    def indexed(self, name: str, indices):
        found = self.lookup(name)
        if found is None:
            raise Alarm("undefined_name", f"name {name}", name)
        value = self.get(found[0])
        if isinstance(value, dict) and len(indices) == 2:
            axis = self.index_value(indices[0])
            part = self.index_value(indices[1])
            return frame_component(self.h, value, str(axis), str(part))
        raise Alarm("unsupported_function", f"array access {name}[...]", name)

    def binary(self, op: str, a_node, b_node):
        if op == "AND":
            return 1.0 if (truthy(self.value(a_node)) and truthy(self.value(b_node))) else 0.0
        if op == "OR":
            return 1.0 if (truthy(self.value(a_node)) or truthy(self.value(b_node))) else 0.0
        a = self.value(a_node)
        b = self.value(b_node)
        if op == "XOR":
            return 1.0 if truthy(a) != truthy(b) else 0.0
        if op == "<<":
            return _text(a) + _text(b)
        if op == ":":
            if not (isinstance(a, dict) and isinstance(b, dict)):
                raise Alarm("type_mismatch", "':' chains frames", ":")
            return frame_chain(a, b)
        if op in ("==", "<>"):
            if isinstance(a, str) or isinstance(b, str):
                equal = str(a) == str(b)
            else:
                equal = _number(a) == _number(b)
            return 1.0 if (equal if op == "==" else not equal) else 0.0
        x, y = _number(a), _number(b)
        if op == "+":
            return _finite(x + y)
        if op == "-":
            return _finite(x - y)
        if op == "*":
            return _finite(x * y)
        if op == "/":
            if y == 0.0:
                raise Alarm("zero_divide", "division by zero")
            return _finite(x / y)
        if op == "DIV":
            if y == 0.0:
                raise Alarm("zero_divide", "DIV by zero")
            return float(math.trunc(x / y))
        if op == "MOD":
            if y == 0.0:
                raise Alarm("zero_divide", "MOD by zero")
            return math.fmod(x, y)
        if op == ">":
            return 1.0 if x > y else 0.0
        if op == "<":
            return 1.0 if x < y else 0.0
        if op == ">=":
            return 1.0 if x >= y else 0.0
        if op == "<=":
            return 1.0 if x <= y else 0.0
        ia, ib = int(round_half_away(x)), int(round_half_away(y))
        if op == "B_AND":
            return float(ia & ib)
        if op == "B_OR":
            return float(ia | ib)
        if op == "B_XOR":
            return float(ia ^ ib)
        raise Alarm("syntax", f"operator {op}", op)

    def function(self, name: str, args):
        if name in ("CTRANS", "CROT", "CSCALE", "CMIRROR"):
            return self.frame_function(name, args)
        values = [self.value(a) if a is not None else None for a in args]
        if name == "TOSTR":
            return _text(values[0])
        if name == "STRLEN":
            return float(len(str(values[0])))
        if name == "NUMBER":
            try:
                return float(str(values[0]).strip())
            except ValueError:
                raise Alarm("no_number", f"NUMBER({values[0]!r})") from None
        nums = [_number(v) if v is not None else 0.0 for v in values]
        if name in ("SIN", "COS", "TAN", "ASIN", "ACOS", "SQRT", "POT", "ABS", "TRUNC", "ROUND", "ROUNDUP", "LN",
                    "EXP") and len(nums) != 1:
            raise Alarm("call_parameters", f"{name} takes one argument", name)
        x = nums[0] if nums else 0.0
        if name == "SIN":
            return math.sin(math.radians(x))
        if name == "COS":
            return math.cos(math.radians(x))
        if name == "TAN":
            return _finite(math.tan(math.radians(x)))
        if name in ("ASIN", "ACOS"):
            if not -1.0 <= x <= 1.0:
                raise Alarm("argument_out_of_range", f"{name}({x:g})")
            return math.degrees(math.asin(x) if name == "ASIN" else math.acos(x))
        if name == "ATAN2":
            if len(nums) != 2:
                raise Alarm("call_parameters", "ATAN2 takes two arguments", name)
            if nums[0] == 0.0 and nums[1] == 0.0:
                raise Alarm("argument_out_of_range", "ATAN2(0, 0)")
            return math.degrees(math.atan2(nums[0], nums[1]))
        if name == "SQRT":
            if x < 0.0:
                raise Alarm("argument_out_of_range", f"SQRT({x:g})")
            return math.sqrt(x)
        if name == "POT":
            return _finite(x * x)
        if name == "ABS":
            return abs(x)
        if name == "TRUNC":
            return float(math.trunc(x))
        if name == "ROUND":
            return float(round_half_away(x))
        if name == "ROUNDUP":
            return float(math.ceil(x)) if x >= 0.0 else float(-math.ceil(-x))
        if name == "LN":
            if x <= 0.0:
                raise Alarm("argument_out_of_range", f"LN({x:g})")
            return math.log(x)
        if name == "EXP":
            try:
                return _finite(math.exp(x))
            except OverflowError:
                raise Alarm("overflow_float", f"EXP({x:g})") from None
        if name == "MINVAL":
            return min(nums[0], nums[1])
        if name == "MAXVAL":
            return max(nums[0], nums[1])
        if name == "BOUND":
            lo, hi, v = nums[0], nums[1], nums[2]
            return min(max(v, lo), hi)
        raise Alarm("undefined_name", f"function {name}", name)

    def frame_function(self, name: str, args):
        frame = identity_frame()
        values = [self.value(a) for a in args if a is not None]
        if name == "CMIRROR":
            for axis in values:
                if not isinstance(axis, str) or axis not in "XYZ":
                    raise Alarm("frame_illegal", f"CMIRROR({axis})", "CMIRROR")
                frame["M"]["XYZ".index(axis)] = 1
            return frame
        if len(values) % 2:
            raise Alarm("call_parameters", f"{name} needs axis, value pairs", name)
        angles = [0.0, 0.0, 0.0]
        for k in range(0, len(values), 2):
            axis, value = values[k], _number(values[k + 1])
            if not isinstance(axis, str) or axis not in self.ex.by_letter:
                raise Alarm("frame_illegal", f"{name}: {axis} is not an axis", str(axis))
            if name == "CTRANS":
                si = self.h.frame_length(axis, value)
                if axis in "XYZ":
                    frame["T"]["XYZ".index(axis)] = si
                else:
                    frame["TR"][axis] = si
            elif name == "CROT":
                if axis not in "XYZ":
                    raise Alarm("frame_rotation", f"CROT about {axis}")
                angles["XYZ".index(axis)] = value
            else:
                if axis not in "XYZ":
                    raise Alarm("frame_illegal", f"CSCALE of {axis}", axis)
                frame["S"]["XYZ".index(axis)] = value
        if name == "CROT":
            frame["R"] = clean_matrix(rpy_matrix(*angles))
        return frame


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raise Alarm("type_mismatch", "a frame in a string", "FRAME")
    number = _number(value)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.6f}".rstrip("0")


def _finite(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        raise Alarm("overflow_float", "result out of range")
    return value


def frame_component(handler, frame: dict, axis: str, part: str) -> float:
    """FRAME[axis, TR|RT|SC|MI] in program units (degrees for RT)."""
    if part == "TR":
        if axis in "XYZ" and axis:
            return handler.frame_length_back(axis, frame["T"]["XYZ".index(axis)])
        return math.degrees(frame["TR"].get(axis, 0.0))
    if axis not in "XYZ" or not axis:
        raise Alarm("frame_illegal", f"frame component {part} of {axis}", axis)
    k = "XYZ".index(axis)
    if part == "RT":
        return rpy_angles(frame["R"])[k]
    if part == "SC":
        return float(frame["S"][k])
    if part == "MI":
        return float(frame["M"][k])
    raise Alarm("frame_illegal", f"frame component {part}", part)


# ====================================================================== system variables

def clock(handler) -> _dt.datetime:
    start = handler.p.get("clock_start") or "2026-01-05T07:00:00"
    base = _dt.datetime.fromisoformat(str(start))
    return base + _dt.timedelta(microseconds=int(handler.s.interp_tick * handler.ex.tick_s * 1e6))


def read_system(handler, name: str, index: tuple, env: Env):
    """The value of a $ variable (program units)."""
    ex, s = handler.ex, handler.s
    first = index[0] if index else None
    if name == "$P_EP":
        letter = _axis_index(handler, first)
        return handler.position_value(letter, s.pos.get(letter, 0))
    if name in ("$AA_IW", "$AA_IM"):
        letter = _axis_index(handler, first)
        machine = {a.letter: ex.counts(a.letter, ex.live(a.name)) for a in ex.axes}
        if name == "$AA_IM":
            return handler.position_value(letter, machine[letter], machine_frame=True)
        program = ex.program_from_machine(machine)
        return handler.position_value(letter, program[letter])
    if name == "$P_F":
        return float(s.feed_prog)
    if name == "$P_S":
        return float(s.spindle.s_value)
    if name == "$P_TOOLNO":
        return float(s.tool)
    if name == "$P_TOOL":
        return float(s.d_number)
    if name == "$P_TOOLP":
        return float(s.preselect or 0)
    if name == "$P_DRYRUN":
        return 1.0 if s.operator.dry_run else 0.0
    if name == "$P_SEARCH":
        return 1.0 if s.suppress else 0.0
    if name == "$P_GG":
        return handler.group_index(int(round_half_away(_number(first))))
    if name in ("$P_AXN1", "$P_AXN2", "$P_AXN3"):
        a, b, n = PLANES[handler.plane()]
        return {"$P_AXN1": a, "$P_AXN2": b, "$P_AXN3": n}[name]
    if name == "$P_PFRAME":
        return copy_frame(s.sys["frame"])
    if name in ("$P_IFRAME", "$P_UIFR"):
        key = s.wcs if name == "$P_IFRAME" else handler.uifr_key(int(round_half_away(_number(first))))
        return handler.settable_frame(key)
    if name == "$P_ACTFRAME":
        return frame_chain(handler.settable_frame(s.wcs), s.sys["frame"])
    if name in ("$A_YEAR", "$A_MONTH", "$A_DAY", "$A_HOUR", "$A_MINUTE", "$A_SECOND"):
        now = clock(handler)
        return float({"$A_YEAR": now.year, "$A_MONTH": now.month, "$A_DAY": now.day, "$A_HOUR": now.hour,
                      "$A_MINUTE": now.minute, "$A_SECOND": now.second}[name])
    if name == "$TC_DP1":
        tool, edge = _tool_edge(handler, index)
        return 500.0 if handler.T else 120.0
    if name in TOOL_DATA:
        tool, edge = _tool_edge(handler, index)
        return handler.tool_data(tool, TOOL_DATA[name])
    if name.startswith(NOT_MODELLED_PREFIX):
        raise Alarm("not_modelled_variable", name)
    raise Alarm("undefined_name", f"name {name}", name)


def write_system(handler, name: str, index: tuple, value) -> None:
    s = handler.s
    if name in TOOL_DATA:
        tool, edge = _tool_edge(handler, index)
        handler.set_tool_data(tool, TOOL_DATA[name], _number(value))
        return
    if name == "$P_PFRAME":
        if not isinstance(value, dict):
            raise Alarm("illegal_type_conversion", "$P_PFRAME needs a frame")
        s.sys["frame"] = copy_frame(value)
        handler.update_frames()
        return
    if name == "$P_UIFR":
        if not isinstance(value, dict):
            raise Alarm("illegal_type_conversion", "$P_UIFR needs a frame")
        handler.set_settable_frame(handler.uifr_key(int(round_half_away(_number(index[0] if index else 0)))), value)
        return
    if name.startswith(("$P_", "$AA_", "$A_", "$AC_")) or name == "$TC_DP1":
        raise Alarm("write_protected", name)
    if name.startswith(NOT_MODELLED_PREFIX):
        raise Alarm("not_modelled_variable", name)
    raise Alarm("undefined_name", f"name {name}", name)


def _axis_index(handler, value) -> str:
    if not isinstance(value, str) or value not in handler.ex.by_letter:
        raise Alarm("illegal_value", f"axis index {value}")
    return value


def _tool_edge(handler, index: tuple) -> tuple[int, int]:
    if len(index) != 2:
        raise Alarm("call_parameters", "$TC_DP needs [tool, edge]", "$TC_DP")
    tool = int(round_half_away(_number(index[0])))
    edge = int(round_half_away(_number(index[1])))
    if str(tool) not in handler.s.tool_offsets or edge != 1:
        raise Alarm("illegal_d", f"T{tool} D{edge}", str(tool), str(edge))
    return tool, edge


__all__ = ["Env", "convert", "default_value", "truthy", "read_system", "write_system", "frame_component", "clock",
           "CONSTANTS"]
