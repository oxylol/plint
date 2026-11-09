"""Block semantics of the SINUMERIK 840D sl (docs/dialects.md "Siemens 840D sl").

Order inside an NC block: the items are evaluated left to right (assignments visible to later items),
committed, checked (address twice 12000, G group twice 12060, unknown G 12470), then executed:
modal G functions and keywords; F; S, LIMS, T, D, SPOS and the M functions that start something
(M3 M4 M7 M8); a frame instruction, pole, G4/G74/G75 or the motion of group 1 (with G41/G42/G40,
CHR=/CHF=/RND=); the modal call (MCALL) after a path motion; subprogram and cycle calls; MSG, SETAL,
WAITM, STOPRE, TRAORI/TRAFOOF; the remaining M functions (M5 M9 M6 M0 M1 M2 M30 M17); a jump.

All state is in ``ex.s``; this object holds only the immutable context.
"""

from __future__ import annotations

import math

from ...alarms import ControllerAlarm
from ...expr import round_half_away
from ...hooks import CycleCall
from ...ir import Command, Kind
from ...pipeline import PLANES
from . import values as V
from .kit import (TOOL_AXIS_OF_PLANE, Alarm, NativeHandler, arc_end_tangent, circle_through, clean_matrix,
                  copy_frame, frame_chain, identity_frame, plane_rotation, rpy_matrix, tangent_arc_centre)
from .tables import G_FUNCTIONS, STANDARD_CYCLES

START_M = (3, 4, 7, 8)
FEED_TYPE = {"G93": "inverse_time", "G94": "per_min", "G95": "per_rev", "G96": "per_rev", "G97": "per_rev"}
GROUP_ORDER = {
    1: ("G0", "G1", "G2", "G3", "CIP", "ASPLINE", "BSPLINE", "CSPLINE", "POLY", "G33", "G331", "G332", "OEMIPO1",
        "OEMIPO2", "CT", "G34", "G35"),
    6: ("G17", "G18", "G19"), 7: ("G40", "G41", "G42"), 10: ("G60", "G64", "G641", "G642", "G643", "G644", "G645"),
    13: ("G70", "G71", "G700", "G710"), 14: ("G90", "G91"), 15: ("G93", "G94", "G95", "G96", "G97"),
}
MODAL_KEY = {1: "motion", 6: "plane", 7: "7", 8: "wcs", 10: "exact", 12: "stop_window", 13: "units", 14: "distance",
             15: "feed", 16: "feed_reference", 17: "approach", 18: "corner", 21: "accel", 24: "feedforward",
             28: "walim", 29: "diameter", 30: "compressor"}


class Words:
    """The evaluated items of one block."""

    def __init__(self):
        self.g: list[str] = []
        self.axes: dict[str, tuple[str, object]] = {}
        self.axes1: dict[str, tuple[str, object]] = {}
        self.addr: dict[str, tuple[str, object]] = {}
        self.m: list[tuple[int, str]] = []
        self.kw: list[str] = []
        self.frame: str | None = None
        self.calls: list[tuple] = []
        self.mcall = None
        self.mcall_given = False
        self.goto = None
        self.unmodelled: list[str] = []
        self.used: set[str] = set()

    def take(self, name: str):
        if name in self.addr and name not in self.used:
            self.used.add(name)
            return self.addr[name]
        return None

    def has(self, name: str) -> bool:
        return name in self.addr and name not in self.used


class SiemensHandler(NativeHandler):
    dialect_key = "siemens840d"

    def __init__(self, ex, dialect):
        super().__init__(ex, dialect)
        self.default_plane = "G18" if self.T else "G17"

    # ================================================================== state
    def init_dialect(self) -> None:
        s = self.s
        s.variables["R"] = [0.0] * int(self.p["r_parameters"])
        s.sys["lud"] = [self.new_scope(None)]
        s.sys["pole"] = [0.0, 0.0]
        s.sys["polar"] = [0.0, 0.0]
        s.wcs = s.modal.get("wcs", "G500")
        s.offsets.setdefault(s.wcs, [0] * len(self.ex.axes))
        s.tool_axis_letter = TOOL_AXIS_OF_PLANE[self.plane()]
        self.apply_corner_default()

    def apply_corner_default(self) -> None:
        s = self.s
        if self.machine.corner_default != "exact_stop":
            s.modal["exact"] = "G64"
        s.smoothing["corner"] = "exact_stop" if s.modal.get("exact") == "G60" else "blend"

    def offset_keys(self) -> list[str]:
        return ["G500", "G54", "G55", "G56", "G57"]

    def new_scope(self, pending: dict | None) -> dict:
        frame = self.s.frames[-1] if self.s.frames else None
        sig = [frame.program, frame.entry_pc, frame.call_line] if frame is not None else [-1, 0, 0]
        scope = {"sig": sig, "vars": {}, "args": [], "save": None, "proc": "", "sblof": False}
        if pending:
            scope.update({k: v for k, v in pending.items() if k in ("args", "save", "proc", "sblof")})
        return scope

    def scope(self) -> dict:
        self.sync_scopes()
        return self.s.sys["lud"][len(self.s.frames) - 1]

    def sync_scopes(self) -> None:
        s = self.s
        scopes = s.sys.setdefault("lud", [])
        depth = len(s.frames)
        while len(scopes) > depth:
            scopes.pop()
        while len(scopes) < depth:
            pending = s.sys.pop("lud_pending", None) if len(scopes) == depth - 1 else None
            scopes.append(self.new_scope(pending))
        s.sys["single_block_suppressed"] = any(scope.get("sblof") for scope in scopes)

    def on_reset(self) -> None:
        s = self.s
        keep = {k: s.modal.get(k) for k in ("wcs",)}
        s.modal = dict(self.d.power_on)
        for key, code in (self.p.get("power_on") or {}).items():
            s.modal[str(key)] = code
        s.modal.update({k: v for k, v in keep.items() if v is not None and k != "wcs"})
        s.units = "inch" if s.modal.get("units") in ("G70", "G700") else "mm"
        self.apply_corner_default()
        s.sys["frame"] = identity_frame()
        s.sys.pop("swivel", None)
        s.sys.pop("mcall", None)
        s.sys.pop("for", None)
        s.sys.pop("sections", None)
        s.sys.pop("lud_pending", None)
        s.sys["lud"] = [self.new_scope(None)]
        s.sys["pole"] = [0.0, 0.0]
        s.sys["polar"] = [0.0, 0.0]
        s.sys.pop("tangent", None)
        s.tcp = ""
        s.feature = None
        s.mirror = {}
        s.scale = None
        s.rotation = None
        s.g52 = {}
        s.wcs = s.modal.get("wcs", "G500")
        s.tool_axis_letter = TOOL_AXIS_OF_PLANE[self.plane()]
        self.ex.invalidate_transform()

    # ================================================================== units and positions
    def length_si(self, value: float) -> float:
        return float(value) * (0.0254 if self.s.units == "inch" else 1e-3)

    def feed_unit_m(self) -> float:
        return 0.0254 if self.s.modal.get("units") == "G700" else 1e-3

    def diameter_applies(self, absolute: bool) -> bool:
        mode = self.s.modal.get("diameter", "DIAMOF")
        return mode == "DIAMON" or (mode == "DIAM90" and absolute)

    def axis_si(self, letter: str, value: float, absolute: bool) -> float:
        axis = self.ex.by_letter[letter]
        if axis.kind == "rotary":
            return math.radians(float(value))
        si = self.length_si(value)
        if axis.diameter and self.diameter_applies(absolute):
            si /= 2.0
        return si

    def position_value(self, letter: str, counts: int, machine_frame: bool = False) -> float:
        axis = self.ex.by_letter[letter]
        si = counts * axis.quantum
        if axis.kind == "rotary":
            return _clean(math.degrees(si))
        if axis.diameter and self.diameter_applies(True):
            si *= 2.0
        return _clean(si / (0.0254 if self.s.units == "inch" else 1e-3))

    def frame_length(self, axis: str, value: float) -> float:
        info = self.ex.by_letter.get(axis)
        if info is not None and info.kind == "rotary":
            return math.radians(value)
        return self.length_si(value)

    def frame_length_back(self, axis: str, si: float) -> float:
        return _clean(si / (0.0254 if self.s.units == "inch" else 1e-3))

    def tool_axis_letter(self) -> str:
        if self.T:
            return "Z"
        return TOOL_AXIS_OF_PLANE[self.plane()]

    # ================================================================== system variables
    def read_sys(self, name: str, index: tuple, env):
        return V.read_system(self, name, index, env)

    def write_sys(self, name: str, index: tuple, value) -> None:
        V.write_system(self, name, index, value)

    def group_index(self, group: int) -> float:
        order = GROUP_ORDER.get(group)
        if group == 8:
            wcs = self.s.wcs
            if wcs == "G500":
                return 1.0
            if wcs in ("G54", "G55", "G56", "G57"):
                return float(("G54", "G55", "G56", "G57").index(wcs) + 2)
            return float(int(wcs[1:]) - 505 + 6)
        if order is None:
            raise Alarm("not_modelled_variable", f"$P_GG[{group}]")
        code = self.s.modal.get(MODAL_KEY[group])
        return float(order.index(code) + 1) if code in order else 0.0

    def uifr_key(self, n: int) -> str:
        if n == 0:
            return "G500"
        if 1 <= n <= 4:
            return ("G54", "G55", "G56", "G57")[n - 1]
        if 5 <= n <= 99:
            return f"G{500 + n}"
        raise Alarm("variable_out_of_range", f"$P_UIFR[{n}]")

    def settable_frame(self, key: str) -> dict:
        ex = self.ex
        row = self.s.offsets.get(key) or [0] * len(ex.axes)
        frame = identity_frame()
        for i, axis in enumerate(ex.axes):
            si = row[i] * axis.quantum
            if axis.letter in "XYZ" and axis.kind == "linear":
                frame["T"]["XYZ".index(axis.letter)] = si
            elif si:
                frame["TR"][axis.letter] = si
        return frame

    def set_settable_frame(self, key: str, frame: dict) -> None:
        ex = self.ex
        if frame["R"] != identity_frame()["R"] or frame["S"] != [1.0, 1.0, 1.0] or any(frame["M"]):
            raise Alarm("unsupported_function", "settable frames with rotation, scale or mirror (only translations)")
        row = [0] * len(ex.axes)
        for i, axis in enumerate(ex.axes):
            if axis.letter in "XYZ" and axis.kind == "linear":
                row[i] = ex.counts(axis.letter, frame["T"]["XYZ".index(axis.letter)])
            else:
                row[i] = ex.counts(axis.letter, frame["TR"].get(axis.letter, 0.0))
        self.s.offsets[key] = row
        ex.resync_program_position()

    def tool_data(self, tool: int, field: str) -> float:
        row = self.s.tool_offsets[str(tool)]
        inc = self.machine.lin_increment_m
        unit = 0.0254 if self.s.units == "inch" else 1e-3
        index = self.tool_field(field)
        return _clean(row[index] * inc / unit) if index is not None else 0.0

    def tool_field(self, field: str) -> int | None:
        from .kit import ROW_LEN, ROW_LEN_WEAR, ROW_NOSE, ROW_NOSE_WEAR, ROW_RAD, ROW_RAD_WEAR, ROW_X, ROW_X_WEAR, ROW_Z
        if self.T:
            return {"len": ROW_X, "len2": ROW_Z, "rad": ROW_NOSE, "len_wear": ROW_X_WEAR, "rad_wear": ROW_NOSE_WEAR,
                    "len3": None}.get(field)
        return {"len": ROW_LEN, "rad": ROW_RAD, "len_wear": ROW_LEN_WEAR, "rad_wear": ROW_RAD_WEAR, "len2": None,
                "len3": None}.get(field)

    def set_tool_data(self, tool: int, field: str, value: float) -> None:
        index = self.tool_field(field)
        if index is None:
            raise Alarm("not_modelled_variable", f"$TC_DP {field} of this tool type")
        row = self.s.tool_offsets[str(tool)]
        row[index] = int(round_half_away(self.length_si(value) / self.machine.lin_increment_m))

    # ================================================================== block execution
    def run_block(self, block) -> None:
        s = self.s
        if s.sys.get("sections") and self.section_end():
            return
        if block.error is not None:
            params = block.extra.params if block.extra is not None else ()
            raise Alarm(block.error[0], block.error[1], *params)
        if block.delete and s.operator.block_delete[block.delete - 1]:
            return
        self.sync_scopes()
        sb = block.extra
        if sb.kind == "eof":
            raise Alarm("end_of_record", "end of file without M30, M02 or M17")
        if sb.kind == "stmt":
            self.statement(sb.stmt, sb)
            return
        if sb.kind == "empty":
            return
        self.nc_block(sb)

    # ------------------------------------------------------------------ jumps and structures
    def jump_to(self, index: int) -> None:
        """Continue at block ``index`` of the current program (the core adds one after the block)."""
        frame = self.s.frames[-1]
        scanned = abs(index - frame.pc)
        self.ex.advance(scanned * self.p["goto_scan_s"])
        frame.pc = index - 1

    def partner(self, sb, role: str) -> int:
        for name, index in sb.match:
            if name == role:
                return index
        raise Alarm("structure_nesting", f"no {role} partner")

    def statement(self, stmt: tuple, sb) -> None:
        kind = stmt[0]
        env = V.Env(self)
        if kind == "if":
            if not V.truthy(env.value(stmt[1])):
                target = next((i for role, i in sb.match if role == "else"), None)
                self.jump_to((target if target is not None else self.partner(sb, "end")) + 1)
        elif kind == "else":
            self.jump_to(self.partner(sb, "end") + 1)
        elif kind == "while":
            if not V.truthy(env.value(stmt[1])):
                self.jump_to(self.partner(sb, "end") + 1)
        elif kind == "endwhile":
            self.jump_to(self.partner(sb, "start"))
        elif kind == "for":
            self.for_start(stmt, sb, env)
        elif kind == "endfor":
            self.for_end(sb)
        elif kind == "endloop":
            self.jump_to(self.partner(sb, "start") + 1)
        elif kind == "until":
            if not V.truthy(env.value(stmt[1])):
                self.jump_to(self.partner(sb, "start") + 1)
        elif kind in ("endif", "loop", "repeat", "extern"):
            return
        elif kind == "if_goto":
            if V.truthy(env.value(stmt[1])):
                self.goto(stmt[2], env)
        elif kind == "repeat_section":
            self.section_start(stmt, env)
        elif kind == "def":
            self.define(stmt, env)
        elif kind == "proc":
            self.bind_parameters(stmt)
        else:
            raise Alarm("syntax", kind, kind)

    # ------------------------------------------------------------------ program section repetition
    def section_start(self, stmt: tuple, env) -> None:
        """REPEAT / REPEATB: run the section P times (default 1), then continue after the statement. The
        section's frame, first and last block, the passes left and the return block are kept in
        sys["sections"] (innermost last); :meth:`section_end` acts when control leaves the last block."""
        _, word, first, second, count_node = stmt
        s = self.s
        frame = s.frames[-1]
        program = self.ex.store.programs[frame.program]
        labels = self.ex.store.info[frame.program].labels
        here = frame.pc
        count = 1 if count_node is None else int(round_half_away(env.number(count_node)))
        if count < 0:
            raise Alarm("pass_count", f"{word} P={count}")
        start = labels.get(first)
        if start is None:
            raise Alarm("jump_not_found", f"{word} {first}", first)
        if word == "REPEATB":
            end = start
        elif second is not None:
            end = labels.get(second)
            if end is None:
                raise Alarm("jump_not_found", f"REPEAT {first} {second}", second)
            if end < start:
                raise Alarm("jump_not_found", f"REPEAT {first} {second}: the end label precedes the start", second)
        else:
            stop = here if start < here else len(program.blocks)
            ends = [k for k in range(start + 1, stop) if getattr(program.blocks[k].extra, "label", "") == "ENDLABEL"]
            if ends:
                end = ends[0]
            elif start < here:
                end = here - 1
            else:
                raise Alarm("jump_not_found", f"REPEAT {first}: no ENDLABEL after the label", "ENDLABEL")
        if start <= here <= end:
            raise Alarm("structure_nesting", f"{word} {first}: the section contains its REPEAT")
        if count == 0:
            return
        sections = s.sys.setdefault("sections", [])
        if len(sections) >= int(self.p["structure_nesting_max"]):
            raise Alarm("structure_depth", f"more than {len(sections)} nested section repetitions")
        sections.append({"depth": len(s.frames), "program": frame.program, "start": start, "end": end,
                         "left": count, "back": here + 1})
        self.jump_to(start)

    def section_end(self) -> bool:
        """Before a block runs: if control has just left the last block of the innermost section (the
        block after it is next in the section's own frame), run the section again or return after the
        REPEAT. A section whose frame has ended is dropped. True when the block must not run now."""
        s = self.s
        sections = s.sys["sections"]
        while sections and sections[-1]["depth"] > len(s.frames):
            sections.pop()
        if not sections:
            s.sys.pop("sections", None)
            return False
        top = sections[-1]
        frame = s.frames[-1]
        if top["depth"] != len(s.frames) or top["program"] != frame.program or frame.pc != top["end"] + 1:
            return False
        top["left"] -= 1
        if top["left"] > 0:
            self.jump_to(top["start"])
        else:
            sections.pop()
            if not sections:
                s.sys.pop("sections", None)
            self.jump_to(top["back"])
        return True

    def for_key(self, start_index: int) -> str:
        return f"{self.s.frames[-1].program}:{start_index}:{len(self.s.frames)}"

    def for_target(self, env, target):
        if target[0] == "r":
            return env.r_key(target[1]), "REAL"
        found = env.lookup(target[1])
        if found is None:
            raise Alarm("undefined_name", f"name {target[1]}", target[1])
        if found[1] not in ("INT", "REAL"):
            raise Alarm("type_mismatch", f"FOR variable {target[1]} is {found[1]}", target[1])
        return found[0], found[1]

    def for_start(self, stmt: tuple, sb, env) -> None:
        key, kind = self.for_target(env, stmt[1])
        first = V.convert(kind, env.number(stmt[2]))
        last = env.number(stmt[3])
        env.put(key, first)
        env.commit()
        here = self.s.frames[-1].pc
        self.s.sys.setdefault("for", {})[self.for_key(here)] = last
        if first > last:
            self.jump_to(self.partner(sb, "end") + 1)

    def for_end(self, sb) -> None:
        start = self.partner(sb, "start")
        program = self.ex.store.programs[self.s.frames[-1].program]
        stmt = program.blocks[start].extra.stmt
        env = V.Env(self)
        key, kind = self.for_target(env, stmt[1])
        last = self.s.sys.get("for", {}).get(self.for_key(start))
        if last is None:
            raise Alarm("structure_nesting", "ENDFOR without an active FOR")
        value = env.get(key) + 1.0
        env.put(key, value)
        env.commit()
        if value <= last:
            self.jump_to(start + 1)
        else:
            self.s.sys["for"].pop(self.for_key(start), None)

    def goto(self, item: tuple, env) -> None:
        kind, target = item[1], item[2]
        frame = self.s.frames[-1]
        program = self.ex.store.programs[frame.program]
        info = self.ex.store.info[frame.program]
        pc = frame.pc
        candidates: list[int] = []
        shown = ""
        if target[0] == "label":
            name = target[1]
            found = env.lookup(name)
            if found is not None and found[1] == "STRING":
                name = str(env.get(found[0]))
            shown = name
            index = info.labels.get(name)
            candidates = [index] if index is not None else []
        elif target[0] == "n":
            shown = f"N{target[1]}"
            candidates = list(program.n_index.get(target[1], ()))
        else:
            name = str(env.value(target[1]))
            shown = name
            index = info.labels.get(name)
            candidates = [index] if index is not None else []
        forward = [i for i in candidates if i > pc]
        backward = [i for i in candidates if i < pc]
        if kind == "GOTOF":
            choice = forward[0] if forward else None
        elif kind == "GOTOB":
            choice = backward[-1] if backward else None
        else:
            choice = forward[0] if forward else (backward[-1] if backward else None)
        if choice is None:
            if kind == "GOTOC":
                self.ex.advance(len(program.blocks) * self.p["goto_scan_s"])
                return
            raise Alarm("jump_not_found", f"{kind} {shown}", shown)
        self.jump_to(choice)

    # ------------------------------------------------------------------ definitions and calls
    def define(self, stmt: tuple, env) -> None:
        _, kind, length, names = stmt
        scope = self.scope()
        for name, init in names:
            if name in scope["vars"]:
                raise Alarm("name_defined_twice", name, name)
            value = V.convert(kind, env.value(init), length) if init is not None else V.default_value(kind)
            scope["vars"][name] = [kind, value, None, length]

    def bind_parameters(self, stmt: tuple) -> None:
        _, name, params, flags = stmt
        scope = self.scope()
        args = scope.get("args") or []
        if len(args) > len(params):
            raise Alarm("call_parameters", f"{name}: {len(args)} parameters for {len(params)}")
        for k, (kind, pname, by_ref, length) in enumerate(params):
            given = args[k] if k < len(args) else None
            if by_ref:
                if given is None or given[0] != "ref":
                    raise Alarm("call_parameters", f"{name}: VAR parameter {pname} needs a variable")
                scope["vars"][pname] = [kind, None, given[1], length]
            else:
                value = V.convert(kind, given[1], length) if given is not None else V.default_value(kind)
                scope["vars"][pname] = [kind, value, None, length]

    def call_arguments(self, args, env) -> list:
        out = []
        for node in args:
            if node is None:
                out.append(None)
                continue
            ref = None
            if node[0] == "r":
                ref = list(env.r_key(node[1]))
            elif node[0] == "name":
                found = env.lookup(node[1])
                if found is not None:
                    key = found[0]
                    ref = list(key) if key[0] == "R" else ["L", key[1], key[2]]
            out.append(("val", env.value(node), ref))
        return out

    def call_program(self, name: str, args: list, repeat: int) -> None:
        s = self.s
        index = self.find_program(name)
        proc = self.ex.store.info[index].proc
        params = proc[2] if proc else ()
        if args and proc is None:
            raise Alarm("proc_missing", name)
        if len(args) > len(params) and proc is not None:
            raise Alarm("call_parameters", f"{name}: {len(args)} parameters for {len(params)}")
        if repeat < 1:
            raise Alarm("pass_count", f"P{repeat}")
        binding = []
        for k, arg in enumerate(args):
            if arg is None:
                binding.append(None)
                continue
            by_ref = params[k][2] if k < len(params) else False
            if by_ref:
                if arg[2] is None:
                    raise Alarm("param_not_variable", f"{name}: argument {k + 1} for VAR {params[k][1]}",
                                params[k][1])
                binding.append(["ref", list(arg[2])])
            else:
                binding.append(["val", arg[1]])
        flags = proc[3] if proc else ()
        save = self.save_snapshot() if "SAVE" in flags else None
        s.sys["lud_pending"] = {"args": binding, "save": save, "proc": name, "sblof": "SBLOF" in flags}
        self.ex.request_call("sub", index, None, repeat)

    def save_snapshot(self) -> dict:
        s = self.s
        return {"modal": dict(s.modal), "frame": copy_frame(s.sys["frame"]), "wcs": s.wcs, "units": s.units,
                "feed_si": s.feed_si, "feed_set": s.feed_set, "feed_prog": s.feed_prog}

    def sub_end(self, code: str) -> None:
        ex, s = self.ex, self.s
        if len(s.frames) == 1:
            ex.end_program(code)
            return
        frame = s.frames[-1]
        scope = self.scope()
        if frame.repeat_left > 0:
            scope["vars"] = {}
            ex.ret()
            return
        save = scope.get("save")
        ex.ret()
        self.sync_scopes()
        if save:
            plane_before = self.plane()
            s.modal.update(save["modal"])
            s.units = save["units"]
            s.feed_si, s.feed_set, s.feed_prog = save["feed_si"], save["feed_set"], save["feed_prog"]
            if save["wcs"] != s.wcs:
                s.wcs = save["wcs"]
                s.offsets.setdefault(s.wcs, [0] * len(ex.axes))
            s.sys["frame"] = save["frame"]
            if self.plane() != plane_before:
                s.tool_axis_letter = TOOL_AXIS_OF_PLANE[self.plane()]
            s.smoothing["corner"] = "exact_stop" if s.modal.get("exact") == "G60" else "blend"
            self.update_frames()

    # ================================================================== NC block
    def nc_block(self, sb) -> None:
        env = V.Env(self)
        w = self.read_items(sb.items, env)
        env.commit()
        self.check_block(w)
        self.execute_words(w, env)

    def read_items(self, items, env) -> Words:
        w = Words()
        for item in items:
            kind = item[0]
            if kind == "assign":
                self.assign(item[1], item[2], env)
            elif kind == "g":
                w.g.append(item[1])
            elif kind in ("axis", "axis1"):
                table = w.axes if kind == "axis" else w.axes1
                if item[1] in table:
                    raise Alarm("axis_twice", item[1], item[1])
                table[item[1]] = (item[2], env.value(item[3]))
            elif kind == "axis_ind":
                axis = env.value(item[1])
                if not isinstance(axis, str) or axis not in self.ex.by_letter:
                    raise Alarm("illegal_value", f"AX[{axis}]")
                if axis in w.axes:
                    raise Alarm("axis_twice", axis, axis)
                w.axes[axis] = (item[2], env.value(item[3]))
            elif kind == "addr":
                name = item[1]
                if name in w.addr:
                    raise Alarm("duplicate_address", name, name)
                w.addr[name] = (item[2], env.value(item[3]))
            elif kind == "m":
                w.m.append((self.m_number(env.value(item[1])), item[2]))
            elif kind == "kw":
                w.kw.append(item[1])
            elif kind == "frame":
                if w.frame is not None:
                    raise Alarm("syntax", "two frame instructions in one block", item[1])
                w.frame = item[1]
            elif kind == "call":
                w.calls.append(("call", item[1], self.call_arguments(item[2], env), item[3]))
            elif kind == "callname":
                w.calls.append(("call", str(env.value(item[1])).upper(), [], item[2]))
            elif kind == "mcall":
                w.mcall_given = True
                if item[1] is not None:
                    w.mcall = ("call", item[1][1], self.call_arguments(item[1][2], env), item[1][3])
            elif kind == "goto":
                w.goto = item
            elif kind == "unmodelled":
                w.unmodelled.append(item[1])
            else:
                raise Alarm("syntax", kind, kind)
        for name in list(w.addr):
            if name.startswith(("S", "M")) and name[1:].isdigit():
                self.spindle_address(name, w)
        return w

    def m_number(self, value) -> int:
        number = V._number(value, "M")
        if number < 0 or number != int(number):
            raise Alarm("multiple_codes", f"M{number:g}", f"M{number:g}")
        return int(number)

    def spindle_address(self, name: str, w: Words) -> None:
        """S1= / M1= address the spindle by number: 1 is this path's spindle."""
        number = int(name[1:])
        if number != 1:
            raise Alarm("unsupported_function", f"{name}: only spindle 1 of the channel is modelled")
        mod, value = w.addr.pop(name)
        if name[0] == "S":
            if "S" in w.addr:
                raise Alarm("duplicate_address", "S", "S")
            w.addr["S"] = (mod, value)
        else:
            w.m.append((self.m_number(value), "1"))

    def assign(self, target, node, env) -> None:
        value = env.value(node)
        kind = target[0]
        if kind == "r":
            env.put(env.r_key(target[1]), V._number(value, "R parameter"))
            return
        if kind == "sys":
            index = tuple(env.index_value(i) for i in target[2])
            env.sys_writes.append((target[1], index, value))
            return
        if kind == "idx":
            raise Alarm("unsupported_function", f"array element {target[1]}[...]", target[1])
        found = env.lookup(target[1])
        if found is None:
            raise Alarm("undefined_name", f"name {target[1]}", target[1])
        key, vtype, length = found
        env.put(key, V.convert(vtype, value, length))

    def check_block(self, w: Words) -> None:
        if w.unmodelled:
            raise Alarm("unsupported_function", " ".join(w.unmodelled))
        groups: dict[int, str] = {}
        for code in w.g:
            spec = G_FUNCTIONS.get(code)
            if spec is None:
                from .tables import UNMODELLED

                if code in UNMODELLED:
                    raise Alarm("unsupported_function", code)
                raise Alarm("improper_g_code", code, code)
            supported = self.d.supported_g
            if supported is not None and code.startswith("G") and code not in supported:
                raise Alarm("improper_g_code", code, code)
            group = spec[0]
            if group in groups and groups[group] != code:
                raise Alarm("same_g_group", f"{groups[group]} and {code}")
            groups[group] = code
        if len(w.m) > int(self.p["max_m_codes_per_block"]):
            raise Alarm("too_many_m", " ".join(f"M{m}" for m, _ in w.m))
        spindle = [m for m, _ in w.m if m in (3, 4, 5, 19, 70)]
        if len(spindle) > 1:
            raise Alarm("m_group_twice", " ".join(f"M{m}" for m in spindle))

    def g_of(self, w: Words, group: int) -> str | None:
        for code in w.g:
            if G_FUNCTIONS[code][0] == group:
                return code
        return None

    # ------------------------------------------------------------------ the block's actions
    def execute_words(self, w: Words, env) -> None:
        ex, s = self.ex, self.s
        nonmodal = {G_FUNCTIONS[c][0]: c for c in w.g if G_FUNCTIONS[c][0] in (2, 3, 9, 11)}
        for code in w.g:
            group, kind, arg = G_FUNCTIONS[code]
            if group in (2, 3, 9, 11):
                continue
            self.modal_function(code, group, kind, arg, w)
        for name in w.kw:
            if name in ("SAVE", "DISPLOF", "SBLOF"):
                raise Alarm("syntax", f"{name} outside PROC", name)
        if "G4" in w.g:
            self.dwell(w)
        else:
            self.f_word(w)
        self.spindle_words(w, dwell=("G4" in w.g))
        self.t_and_d(w)
        after: list[int] = []
        codes = [m for m, _ in w.m]
        if 6 in codes:
            self.m_function(6, w)
        for m in codes:
            if m == 6:
                continue
            if m in START_M:
                self.m_function(m, w)
            else:
                after.append(m)
        if "G9" in w.g:
            s.sys["g09"] = True
        moved = False
        if w.frame is not None:
            self.frame_instruction(w.frame, w)
        elif 3 in nonmodal and G_FUNCTIONS[nonmodal[3]][1] == "pole":
            self.pole(nonmodal[3], w)
        elif 3 in nonmodal and G_FUNCTIONS[nonmodal[3]][1] in ("lower_limit", "upper_limit"):
            self.limits(nonmodal[3], w)
        elif 3 in nonmodal and G_FUNCTIONS[nonmodal[3]][1] == "axial_offset":
            self.axial_offset(nonmodal[3], w)
        elif 2 in nonmodal and nonmodal[2] in ("G74", "G75"):
            moved = self.reference(nonmodal[2], w)
        elif 9 in nonmodal:
            moved = self.machine_block(nonmodal[9], w)
        elif "G4" not in w.g:
            moved = self.motion(w, tapping=("G63" in w.g))
        s.sys.pop("g09", None)
        self.leftover(w)
        if moved:
            self.modal_call_after_move()
        if w.mcall_given:
            self.set_mcall(w.mcall)
        for kw in w.kw:
            self.keyword_action(kw)
        for call in w.calls:
            self.call(call, env)
        for m in after:
            self.m_function(m, w)
        if w.goto is not None:
            self.goto(w.goto, V.Env(self))

    def leftover(self, w: Words) -> None:
        for name in w.addr:
            if name not in w.used:
                raise Alarm("unexpected_parameter", f"{name} has no use in this block", name)
        for letter in w.axes:
            if letter not in w.used and letter not in self.ex.by_letter:
                raise Alarm("illegal_plane", f"axis {letter}", letter)

    # ------------------------------------------------------------------ modal G functions
    def modal_function(self, code: str, group: int, kind: str, arg: str, w: Words) -> None:
        ex, s = self.ex, self.s
        key = MODAL_KEY.get(group, str(group))
        if kind == "plane":
            if s.modal.get(key) != code and s.modal.get("7", "G40") != "G40":
                raise Alarm("plane_change_in_comp", code)
            if s.modal.get(key) != code:
                s.sys.pop("tangent", None)
            s.modal[key] = code
            s.tool_axis_letter = TOOL_AXIS_OF_PLANE[arg] if not self.T else "Z"
        elif kind == "units":
            s.modal[key] = code
            s.units = "inch" if arg == "inch" else "mm"
        elif kind == "feed":
            previous = s.modal.get(key)
            if arg == "css":
                s.modal[key] = code
                s.spindle.css = True
                s.modal["feed_mode"] = "per_rev"
            elif arg == "css_off":
                s.modal[key] = code
                s.spindle.css = False
                s.modal["feed_mode"] = "per_rev"
                if s.spindle.direction != 0:
                    self.emit_spindle()
            else:
                s.modal[key] = code
                s.spindle.css = False if previous == "G96" else s.spindle.css
                s.modal["feed_mode"] = arg
            if FEED_TYPE.get(previous) != FEED_TYPE.get(code) and not w.has("F"):
                s.feed_si, s.feed_set = 0.0, False
        elif kind == "wcs":
            s.modal[key] = code
            if s.wcs != code:
                s.wcs = code
                s.offsets.setdefault(code, [0] * len(ex.axes))
                ex.resync_program_position()
        elif kind == "exact":
            s.modal[key] = code
            corner = "exact_stop" if arg == "exact_stop" else "blend"
            s.smoothing["corner"] = corner
            tolerance = 0.0
            if arg == "blend_adis":
                adis = w.take("ADIS")
                adispos = w.take("ADISPOS")
                if adis is not None:
                    s.sys["adis_m"] = self.length_si(V._number(adis[1], "ADIS"))
                if adispos is not None:
                    s.sys["adispos_m"] = self.length_si(V._number(adispos[1], "ADISPOS"))
                tolerance = float(s.sys.get("adis_m", 0.0))
                # G641 smooths by a distance criterion: ADIS is the rounding clearance along the blocks from the
                # transition, not a deviation (db/controllers/siemens-840d-sl.yaml; docs/models/motion.md 2.2)
                s.smoothing["tolerance_kind"] = "distance"
            elif s.smoothing.get("tolerance_kind") == "distance":
                # leaving G641: the contour tolerance of CYCLE832 (or the control default) is in force again
                s.smoothing["tolerance_m"] = float(s.sys.get("contour_tol_m", 0.0))
                s.smoothing["tolerance_kind"] = "deviation"
            s.smoothing["tolerance_m"] = tolerance if arg == "blend_adis" else s.smoothing.get("tolerance_m", 0.0)
            ex.emit(Command(Kind.MODE, ex.line_ref(), smoothing=ex.smoothing_spec(), modal={"exact": code}))
        elif kind == "comp":
            return
        elif kind == "motion":
            s.modal[key] = code
            s.sys.pop("cycle_name", None)
        elif kind == "diameter":
            if s.modal.get(key) != code:
                s.modal[key] = code
                ex.resync_program_position()
        elif kind in ("distance", "stop_window", "approach", "corner", "feed_reference"):
            s.modal[key] = code
        else:
            if s.modal.get(key) != code:
                s.modal[key] = code
                ex.emit(Command(Kind.MODE, ex.line_ref(), modal={kind: arg, "code": code}))

    # ------------------------------------------------------------------ F S T D M
    def dwell(self, w: Words) -> None:
        f = w.take("F")
        sw = w.take("S")
        if (f is None) == (sw is None):
            raise Alarm("syntax", "G4 needs F (s) or S (revolutions)", "G4")
        if f is not None:
            seconds = V._number(f[1], "F")
            if seconds < 0.0:
                raise Alarm("value_range", f"G4 F{seconds:g}", f"F{seconds:g}")
            self.ex.dwell(seconds)
        else:
            revolutions = V._number(sw[1], "S")
            if revolutions < 0.0:
                raise Alarm("value_range", f"G4 S{revolutions:g}", f"S{revolutions:g}")
            self.ex.dwell(0.0, revolutions)

    def f_word(self, w: Words) -> None:
        s = self.s
        word = w.take("F")
        if word is None:
            return
        value = V._number(word[1], "F")
        if value <= 0.0:
            raise Alarm("feed_negative", f"F{value:g}")
        mode = self.feed_mode()
        unit = self.feed_unit_m()
        if mode == "per_min":
            si = value * unit / 60.0
            top = self.machine.max_feed_m_s
            if top > 0.0 and si > top:
                self.ex.note("warning", f"F{value:g} clamped to the maximum feed {top * 60000.0:.1f} mm/min")
                si = top
        elif mode == "per_rev":
            si = value * unit
        else:
            si = value / 60.0
        s.feed_si, s.feed_prog, s.feed_set = si, value, True

    def spindle_words(self, w: Words, dwell: bool) -> None:
        s = self.s
        lims = w.take("LIMS")
        if lims is not None:
            rpm = V._number(lims[1], "LIMS")
            if rpm < 0.0:
                raise Alarm("css_negative_lims", f"LIMS={rpm:g}")
            s.spindle.clamp_rad_s = rpm * 2.0 * math.pi / 60.0
        word = None if dwell else w.take("S")
        if word is not None:
            value = V._number(word[1], "S")
            if value < 0.0:
                raise Alarm("spindle_speed_illegal", f"S{value:g}", "S")
            s.spindle.s_value = float(value)
            if s.spindle.direction != 0 and not any(m in (3, 4, 5) for m, _ in w.m):
                self.emit_spindle()
        elif lims is not None and s.spindle.direction != 0:
            self.emit_spindle()
        if s.spindle.css and s.spindle.s_value <= 0.0 and any(m in (3, 4) for m, _ in w.m):
            raise Alarm("css_no_s", "G96 without S")
        spos = w.take("SPOS")
        if spos is not None:
            angle = V._number(spos[1], "SPOS")
            s.spindle.direction = 0
            s.spindle.oriented = True
            self.emit_spindle(orient=math.radians(angle))

    def t_and_d(self, w: Words) -> None:
        ex, s = self.ex, self.s
        word = w.take("T")
        if word is not None:
            number = self.tool_number(word[1])
            capacity = self.machine.tool_capacity
            if capacity and number > capacity:
                raise Alarm("tool_not_in_magazine", f"T{number}", str(number))
            if self.T:
                number = self.tool_life_select(number) if number > 0 else number
                self.emit_tool_change(number)
                if not w.has("D"):
                    self.select_d(int(self.p["cutting_edge_default"]) if number > 0 else 0)
            elif s.preselect != number:
                s.preselect = number
                ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"tool_preselect": number}))
        word = w.take("D")
        if word is not None:
            number = V._number(word[1], "D")
            if number != int(number) or number < 0:
                raise Alarm("illegal_offset_number", f"D{number:g}")
            if 6 in [m for m, _ in w.m]:
                s.sys["d_after_m6"] = int(number)
            else:
                self.select_d(int(number))

    def tool_number(self, value) -> int:
        if isinstance(value, str):
            table = self.ex.ctx.tool_table
            for slot in (table.slots if table is not None else ()):
                if slot.tool_id.upper() == value.upper() and slot.path == self.ex.ctx.path:
                    return int(slot.number)
            raise Alarm("tool_not_in_magazine", f'T="{value}"', value)
        number = V._number(value, "T")
        if number < 0 or number != int(number):
            raise Alarm("tool_not_in_magazine", f"T{number:g}", f"{number:g}")
        return int(number)

    def select_d(self, number: int) -> None:
        s = self.s
        if number == 0:
            s.d_number = 0
            s.h_number = 0
            s.offset_number = 0
            s.sys["active_offset"] = None
            self.offset_changed()
            return
        if s.tool <= 0:
            raise Alarm("no_tool", f"D{number} without a tool")
        if number > 9 or str(s.tool) not in s.tool_offsets or number != 1:
            raise Alarm("illegal_d", f"T{s.tool} D{number}", str(s.tool), str(number))
        s.d_number = number
        s.h_number = number
        s.offset_number = s.tool if self.T else 0
        s.sys["active_offset"] = list(s.tool_offsets[str(s.tool)])
        self.offset_changed()

    def m_function(self, m: int, w) -> None:
        ex, s = self.ex, self.s
        custom = (self.p.get("m_codes") or {}).get(f"M{m}") or (self.p.get("m_codes") or {}).get(f"M{m:02d}")
        if custom is not None:
            self.custom_m(m, dict(custom))
            return
        if m in (3, 4):
            s.spindle.direction = -1 if m == 3 else 1
            if s.spindle.css and s.spindle.s_value <= 0.0:
                raise Alarm("css_no_s", "G96 without S")
            self.emit_spindle()
        elif m == 5:
            s.spindle.direction = 0
            self.emit_spindle()
        elif m == 19:
            s.spindle.direction = 0
            s.spindle.oriented = True
            self.emit_spindle(orient=0.0)
        elif m == 6:
            self.m6()
        elif m == 7:
            kinds = ("mist", "mql")
            if not any(k in kinds for k in self.machine.pumps.values()):
                raise Alarm("undefined_m_code", "M7: no mist coolant on this machine")
            self.coolant_on(kinds, "M7")
        elif m == 8:
            self.coolant_on(("flood",), "M8")
        elif m == 9:
            self.coolant_off("M9")
        elif m in (0, 1):
            if m == 1 and not s.operator.optional_stop:
                return
            self.program_stop(f"M{m}", ex.block.comment if ex.block is not None else "")
        elif m in (2, 30):
            if len(s.frames) > 1:
                self.sub_end(f"M{m}")
            else:
                s.sys["parts"] = int(s.sys.get("parts", 0)) + 1
                ex.end_program(f"M{m}")
        elif m == 17:
            self.sub_end("M17")
        elif 40 <= m <= 45:
            spindle = self.machine.spindles.get(self.spindle_id())
            if m == 40 or spindle is None or spindle.gears <= 1:
                ex.note("info", f"M{m}: automatic / single gear stage, no gear change")
                return
            if m - 40 > spindle.gears:
                raise Alarm("undefined_m_code", f"M{m}: the spindle has {spindle.gears} gear stages")
            s.spindle.gear = m - 40
            self.emit_spindle()
        else:
            raise Alarm("undefined_m_code", f"M{m}")

    def custom_m(self, m: int, spec: dict) -> None:
        ex = self.ex
        kind = spec.get("kind")
        ref = ex.line_ref()
        if kind == "coolant":
            pump = spec.get("pump")
            if pump not in self.machine.pumps:
                raise Alarm("machine_feature_missing", f"M{m}: pump {pump}")
            self.coolant({pump: bool(spec.get("on", True))}, f"M{m}")
        elif kind in ("pallet", "door", "conveyor", "chuck", "output", "brake", "mode"):
            device = {k: v for k, v in spec.items() if k != "kind"}
            cmd_kind = {"pallet": Kind.PALLET, "door": Kind.DOOR, "conveyor": Kind.CONVEYOR,
                        "chuck": Kind.CHUCK}.get(kind, Kind.MODE)
            ex.emit(Command(cmd_kind, ref, modal={"device": {"kind": kind, "code": f"M{m}", **device}}))
        elif kind == "tool_change":
            self.m6()
        elif kind == "stop":
            self.program_stop(f"M{m}")
        elif kind == "ignore":
            ex.note("info", f"M{m}: {spec.get('note', 'no action on this machine')}")
        else:
            raise Alarm("unsupported_function", f"M{m} kind {kind}")

    def m6(self) -> None:
        s = self.s
        number = s.preselect if s.preselect is not None else s.tool
        if number is None or number < 0 or (s.preselect is None and s.tool <= 0):
            raise Alarm("no_tool", "M6 without a selected tool")
        if s.preselect is None:
            self.ex.note("info", f"M6: T{number} is already in the spindle")
            return
        number = self.tool_life_select(number) if number > 0 else number
        self.emit_tool_change(number)
        d = s.sys.pop("d_after_m6", None)
        self.select_d(d if d is not None else (int(self.p["cutting_edge_default"]) if number > 0 else 0))

    # ------------------------------------------------------------------ frames, pole, limits
    def frame_values(self, w: Words) -> dict[str, float]:
        out = {}
        for letter, (mod, value) in w.axes.items():
            if letter not in self.ex.by_letter:
                raise Alarm("illegal_plane", f"axis {letter}", letter)
            if mod:
                raise Alarm("frame_illegal", f"{letter}={mod}() in a frame instruction", letter)
            out[letter] = V._number(value, letter)
            w.used.add(letter)
        return out

    def frame_instruction(self, name: str, w: Words) -> None:
        s = self.s
        values = self.frame_values(w)
        rpl = w.take("RPL")
        delta = identity_frame()
        if name in ("TRANS", "ATRANS"):
            for letter, value in values.items():
                si = self.frame_length(letter, value)
                if letter in "XYZ":
                    delta["T"]["XYZ".index(letter)] = si
                else:
                    delta["TR"][letter] = si
        elif name in ("ROT", "AROT"):
            if rpl is not None:
                if values:
                    raise Alarm("frame_rotation", "RPL= together with axis angles")
                delta["R"] = clean_matrix(plane_rotation(self.plane(), math.radians(V._number(rpl[1], "RPL"))))
            elif values:
                angles = [0.0, 0.0, 0.0]
                for letter, value in values.items():
                    if letter not in "XYZ":
                        raise Alarm("frame_rotation", f"rotation about {letter}")
                    angles["XYZ".index(letter)] = value
                delta["R"] = clean_matrix(rpy_matrix(*angles))
        elif name in ("SCALE", "ASCALE"):
            for letter, value in values.items():
                if letter not in "XYZ":
                    raise Alarm("frame_illegal", f"scale of {letter}", letter)
                if value == 0.0:
                    raise Alarm("illegal_scale", f"{letter}0", f"{letter}0")
                delta["S"]["XYZ".index(letter)] = value
        elif name in ("MIRROR", "AMIRROR"):
            for letter in values:
                if letter not in "XYZ":
                    raise Alarm("frame_illegal", f"mirror of {letter}", letter)
                delta["M"]["XYZ".index(letter)] = 1
        if rpl is not None and name not in ("ROT", "AROT"):
            raise Alarm("unexpected_parameter", "RPL", "RPL")
        s.sys["frame"] = delta if not name.startswith("A") else frame_chain(s.sys["frame"], delta)
        s.sys.pop("tangent", None)
        self.update_frames()

    def axial_offset(self, code: str, w: Words) -> None:
        s = self.s
        frame = copy_frame(s.sys["frame"])
        for letter, value in self.frame_values(w).items():
            si = self.frame_length(letter, value)
            if letter in "XYZ":
                k = "XYZ".index(letter)
                frame["T"][k] = si if code == "G58" else frame["T"][k] + si
            else:
                frame["TR"][letter] = si if code == "G58" else frame["TR"].get(letter, 0.0) + si
        s.sys["frame"] = frame
        self.update_frames()

    def update_frames(self) -> None:
        """Swivel frame (CYCLE800) and programmable frame onto the core stages; the settable frame is the
        core's work offset."""
        import numpy as np

        s = self.s
        frame = s.sys["frame"]
        swivel = s.sys.get("swivel")
        rotation = np.asarray(frame["R"], dtype=float).reshape(3, 3)
        translation = np.asarray(frame["T"], dtype=float)
        if swivel is not None:
            r_sw = np.asarray(swivel["R"], dtype=float).reshape(3, 3)
            translation = np.asarray(swivel["X0"], dtype=float) + r_sw @ (np.asarray(swivel["X1"], dtype=float)
                                                                           + translation)
            rotation = r_sw @ rotation
        self.apply_frame(clean_matrix(rotation), [float(v) for v in translation], frame["S"], frame["M"],
                         frame["TR"])

    def pole(self, code: str, w: Words) -> None:
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        current = (ex.si(a, s.pos.get(a, 0)), ex.si(b, s.pos.get(b, 0)))
        reference = {"G110": current, "G111": (0.0, 0.0), "G112": tuple(s.sys["pole"])}[code]
        ap, rp = w.take("AP"), w.take("RP")
        values = {letter: self.axis_si(letter, V._number(v[1], letter), True) for letter, v in w.axes.items()
                  if letter in (a, b)}
        for letter in w.axes:
            if letter not in (a, b):
                raise Alarm("polar_pole", f"{code} with axis {letter}")
            w.used.add(letter)
        if (ap is not None or rp is not None) and values:
            raise Alarm("polar_pole", f"{code} with Cartesian and polar values")
        if ap is not None or rp is not None:
            angle = math.radians(V._number(ap[1], "AP")) if ap is not None else 0.0
            radius = self.length_si(V._number(rp[1], "RP")) if rp is not None else 0.0
            if radius < 0.0:
                raise Alarm("polar_radius", f"RP={radius:g}")
            pole = (reference[0] + radius * math.cos(angle), reference[1] + radius * math.sin(angle))
        else:
            pole = (reference[0] + values.get(a, 0.0), reference[1] + values.get(b, 0.0))
        s.sys["pole"] = [float(pole[0]), float(pole[1])]
        dx, dy = current[0] - pole[0], current[1] - pole[1]
        s.sys["polar"] = [math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0.0, math.hypot(dx, dy)]

    def limits(self, code: str, w: Words) -> None:
        ex, s = self.ex, self.s
        word = w.take("S")
        if word is not None:
            rpm = V._number(word[1], "S")
            if code == "G26":
                s.spindle.clamp_rad_s = rpm * 2.0 * math.pi / 60.0
            else:
                s.sys["min_rpm"] = rpm
            if s.spindle.direction != 0:
                self.emit_spindle()
        limits = {}
        for letter, (mod, value) in w.axes.items():
            limits[letter] = self.axis_si(letter, V._number(value, letter), True)
            w.used.add(letter)
        if limits:
            key = "walim_min" if code == "G25" else "walim_max"
            stored = dict(s.sys.get(key) or {})
            stored.update({letter: float(v) for letter, v in limits.items()})
            s.sys[key] = stored
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"working_area_" + ("min" if code == "G25" else "max"):
                                                             limits}))

    def check_travel(self, target: dict[str, int]) -> None:
        super().check_travel(target)
        self.check_working_area(target)

    def check_working_area(self, target: dict[str, int]) -> None:
        """G25 / G26 working area limitation (WALIMON, the default) at block preparation, alarm 10730: the
        tool tip's end point in the basic coordinate system (work point + work offsets, without the tool
        offsets) against the programmed limits."""
        ex, s = self.ex, self.s
        lows, highs = s.sys.get("walim_min") or {}, s.sys.get("walim_max") or {}
        if (not lows and not highs) or s.modal.get("walim", "WALIMON") != "WALIMON" or s.tcp:
            return
        position = dict(s.pos)
        position.update(target)
        work = ex.work_from_program(position)
        offsets = ex.offset_counts(letters_tool=False)
        for i, axis in enumerate(ex.axes):
            lo, hi = lows.get(axis.letter), highs.get(axis.letter)
            if lo is None and hi is None:
                continue
            value = (int(round_half_away(work[i] / axis.quantum)) + offsets[i]) * axis.quantum
            tolerance = 0.5 * axis.quantum
            unit = 1e3 if axis.kind == "linear" else 57.29577951308232
            if hi is not None and value > hi + tolerance:
                raise Alarm("working_area", f"{axis.name} {value * unit:.3f} above G26 {hi * unit:g}", axis.name, "+")
            if lo is not None and value < lo - tolerance:
                raise Alarm("working_area", f"{axis.name} {value * unit:.3f} below G25 {lo * unit:g}", axis.name, "-")

    # ------------------------------------------------------------------ motion
    def target(self, w: Words) -> dict[str, int]:
        ex, s = self.ex, self.s
        out: dict[str, int] = {}
        for letter, (mod, value) in w.axes.items():
            axis = ex.by_letter.get(letter)
            if axis is None:
                raise Alarm("illegal_plane", f"axis {letter}", letter)
            w.used.add(letter)
            number = V._number(value, letter)
            absolute = {"AC": True, "IC": False}.get(mod, True if mod in ("DC", "ACP", "ACN") else self.absolute())
            current = s.pos.get(letter, 0)
            if axis.kind == "rotary":
                counts = ex.counts(letter, math.radians(number))
                if mod in ("DC", "ACP", "ACN") and not axis.continuous:
                    raise Alarm("unsupported_function", f"{letter}={mod}() needs a modulo rotary axis")
                if axis.continuous and absolute:
                    out[letter] = self.modulo_target(axis, current, counts, mod)
                else:
                    out[letter] = counts if absolute else current + counts
            else:
                if mod in ("DC", "ACP", "ACN"):
                    raise Alarm("unsupported_function", f"{letter}={mod}() on a linear axis")
                counts = ex.counts(letter, self.axis_si(letter, number, absolute))
                out[letter] = counts if absolute else current + counts
        return out

    def modulo_target(self, axis, current: int, commanded: int, mod: str) -> int:
        full = int(round_half_away(2.0 * math.pi / axis.quantum))
        target = commanded % full
        here = current % full
        if mod == "DC":
            delta = (target - here) % full
            if delta > full // 2:
                delta -= full
        elif mod == "ACP":
            delta = (target - here) % full
        elif mod == "ACN":
            delta = -((here - target) % full)
        else:
            delta = target - here
        return current + delta

    def polar_target(self, w: Words, out: dict[str, int]) -> bool:
        ex, s = self.ex, self.s
        ap, rp = w.take("AP"), w.take("RP")
        if ap is None and rp is None:
            return False
        a, b, _ = PLANES[self.plane()]
        if a in out or b in out:
            raise Alarm("polar_format", "AP/RP together with plane axis words")
        angle, radius = s.sys["polar"]
        if ap is not None:
            value = V._number(ap[1], "AP")
            angle = angle + value if ap[0] == "IC" else value
        if rp is not None:
            value = self.length_si(V._number(rp[1], "RP"))
            radius = radius + value if rp[0] == "IC" else value
        if radius < 0.0:
            raise Alarm("polar_radius", f"RP={radius * 1e3:g}")
        if abs(angle) >= 360.0 * 1e6:
            raise Alarm("polar_angle", f"AP={angle:g}")
        s.sys["polar"] = [angle, radius]
        pole = s.sys["pole"]
        out[a] = ex.counts(a, pole[0] + radius * math.cos(math.radians(angle)))
        out[b] = ex.counts(b, pole[1] + radius * math.sin(math.radians(angle)))
        return True

    def cutting_feed(self):
        s = self.s
        if not s.feed_set or s.feed_si <= 0.0:
            raise Alarm("feed_zero", "no feed programmed")
        return self.ex.feed_spec(False)

    def motion(self, w: Words, tapping: bool = False) -> bool:
        ex, s = self.ex, self.s
        code = s.modal.get("motion", "G0")
        comp_code = self.g_of(w, 7)
        target = self.target(w)
        polar = self.polar_target(w, target)
        arc_words = [n for n in ("I", "J", "K", "CR", "AR", "I1", "J1", "K1", "TURN") if w.has(n)]
        moves = bool(target) or (code in ("G2", "G3") and bool(arc_words))
        if comp_code is not None:
            self.comp_mode(comp_code, code, moves, w)
        if not moves:
            # CHF= CHR= RND= act on the corner at the end of this block's motion; RNDM= is modal and may stand
            # alone
            if any(w.has(name) for name in ("CHF", "CHR", "RND")):
                raise Alarm("chamfer_missing_move", "CHF/CHR/RND without a traversing movement")
            self.modal_rounding(w)
            return False
        corner = self.corner_request(w, code)
        if s.tcp == "siemens840d:TRAORI":
            self.orientation_words(w, target)
        modal = {"tapping": True} if tapping else None
        if code in ("G0", "G1", "G33", "G331", "G332"):
            self.check_travel(target)
        if code == "G0":
            ex.move("rapid", target, ex.feed_spec(True), corner=corner, modal=modal)
            self.remember_line(target)
        elif code == "G1":
            feed = self.cutting_feed()
            if tapping:
                feed = ex.feed_spec(False, override=False)
            ex.move("linear", target, feed, corner=corner, modal=modal)
            self.remember_line(target)
        elif code in ("G2", "G3"):
            self.arc(w, target, -1 if code == "G2" else 1, polar, corner)
        elif code == "CIP":
            self.cip(w, target, corner)
        elif code == "CT":
            self.ct(target, corner)
        elif code == "G33":
            self.thread(w, target)
        elif code in ("G331", "G332"):
            self.tap(w, target, code)
        else:
            raise Alarm("improper_g_code", code, code)
        return True

    def comp_mode(self, code: str, motion: str, moves: bool, w: Words) -> None:
        ex, s = self.ex, self.s
        if motion not in ("G0", "G1"):
            raise Alarm("comp_start_not_linear" if code != "G40" else "comp_end_not_linear", code)
        active = s.comp_state is not None and s.comp_kind in ("cutter", "nose") and s.modal.get("7") in ("G41", "G42")
        if code == "G40":
            if active:
                ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"comp_marker": {"cancel": "G40",
                                                                                  "seq": s.block_seq}}))
            s.modal["7"] = "G40"
            return
        if s.d_number == 0:
            raise Alarm("comp_without_tool", code)
        if active and s.modal.get("7") != code:
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"comp_marker": {"respecified": code,
                                                                              "seq": s.block_seq}}))
        ex.start_comp("nose" if self.T else "cutter")
        s.modal["7"] = code

    def corner_request(self, w: Words, code: str) -> dict | None:
        s = self.s
        corner = None
        rnd = w.take("RND")
        chr_ = w.take("CHR")
        chf = w.take("CHF")
        self.modal_rounding(w)
        given = [x for x in (rnd, chr_, chf) if x is not None]
        if len(given) > 1:
            raise Alarm("syntax", "CHF/CHR/RND together", "CHF")
        if rnd is not None:
            corner = {",R": self.length_si(V._number(rnd[1], "RND"))}
        elif chr_ is not None:
            corner = {",C": self.length_si(V._number(chr_[1], "CHR"))}
        elif chf is not None:
            corner = {",C": self.chamfer_leg(self.length_si(V._number(chf[1], "CHF")), w)}
        elif s.sys.get("rndm_m"):
            corner = {",R": float(s.sys["rndm_m"])}
        if corner is None:
            return None
        if any(v < 0.0 for v in corner.values()):
            raise Alarm("chamfer_negative", "CHF/CHR/RND")
        if not any(v > 0.0 for v in corner.values()):
            return None
        if code not in ("G1", "G2", "G3", "CIP", "CT"):
            raise Alarm("chamfer_not_g01", f"chamfer/rounding with {code}")
        if s.comp_state is None:
            self.ex.start_comp("corner")
        return corner

    def modal_rounding(self, w: Words) -> None:
        """RNDM= value: modal rounding of every following contour corner (0 ends it)."""
        rndm = w.take("RNDM")
        if rndm is not None:
            value = self.length_si(V._number(rndm[1], "RNDM"))
            if value < 0.0:
                raise Alarm("chamfer_negative", "RNDM")
            self.s.sys["rndm_m"] = value

    def chamfer_leg(self, length: float, w: Words) -> float:
        """CHF= is the chamfer length (hypotenuse): the leg along each element is CHF / (2 cos(d/2)) with d the
        change of direction at the corner, from this block's end tangent and the next block's start tangent."""
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        target = dict(s.pos)
        target.update({k: v for k, v in self.peek_target(w).items()})
        start = (ex.si(a, s.pos[a]), ex.si(b, s.pos[b]))
        end = (ex.si(a, target[a]), ex.si(b, target[b]))
        nxt = self.next_line_end(end)
        if nxt is None:
            raise Alarm("chamfer_not_g01", "CHF= needs straight lines on both sides of the corner")
        d1 = (end[0] - start[0], end[1] - start[1])
        d2 = (nxt[0] - end[0], nxt[1] - end[1])
        n1, n2 = math.hypot(*d1), math.hypot(*d2)
        if n1 == 0.0 or n2 == 0.0:
            raise Alarm("chamfer_missing_move", "CHF= at a block without travel")
        cos_turn = (d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
        half = math.acos(max(-1.0, min(1.0, cos_turn))) / 2.0
        if math.cos(half) <= 1e-9:
            raise Alarm("chamfer_too_large", "CHF= at a reversal")
        return length / (2.0 * math.cos(half))

    def peek_target(self, w: Words) -> dict[str, int]:
        saved = set(w.used)
        try:
            return self.target(w)
        finally:
            w.used = saved

    def next_line_end(self, end) -> tuple[float, float] | None:
        """End point (plane, program frame SI) of the next block when it is a straight move in the plane."""
        ex, s = self.ex, self.s
        frame = s.frames[-1]
        program = ex.store.programs[frame.program]
        a, b, _ = PLANES[self.plane()]
        for k in range(frame.pc + 1, len(program.blocks)):
            block = program.blocks[k]
            sb = block.extra
            if sb is None or sb.kind in ("empty",):
                continue
            if sb.kind != "nc" or block.error is not None:
                return None
            items = sb.items
            codes = [i[1] for i in items if i[0] == "g"]
            motion = next((c for c in codes if G_FUNCTIONS.get(c, (0,))[0] == 1), s.modal.get("motion"))
            if motion != "G1":
                return None
            env = V.Env(self)
            axes = {i[1]: (i[2], env.value(i[3])) for i in items if i[0] == "axis"}
            if a not in axes and b not in axes:
                return None
            absolute = not ("G91" in codes or (s.modal.get("distance") == "G91" and "G90" not in codes))
            out = []
            for letter, base in ((a, end[0]), (b, end[1])):
                if letter in axes:
                    mod, value = axes[letter]
                    is_abs = {"AC": True, "IC": False}.get(mod, absolute)
                    si = self.axis_si(letter, V._number(value), is_abs)
                    out.append(si if is_abs else base + si)
                else:
                    out.append(base)
            return out[0], out[1]
        return None

    def remember_line(self, target: dict[str, int]) -> None:
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        if a not in ex.by_letter or b not in ex.by_letter:
            return
        last = s.sys.get("last_pos")
        current = (s.pos[a], s.pos[b])
        if last is not None and tuple(last) != current:
            dx, dy = ex.si(a, current[0] - last[0]), ex.si(b, current[1] - last[1])
            s.sys["tangent"] = [dx, dy]
        s.sys["last_pos"] = [current[0], current[1]]

    def before_move(self) -> tuple[float, float]:
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        s.sys["last_pos"] = [s.pos[a], s.pos[b]]
        return ex.si(a, s.pos[a]), ex.si(b, s.pos[b])

    def tolerance(self) -> tuple[float, float]:
        return float(self.p["circle_error_const_m"]), float(self.p["circle_error_factor"])

    def arc(self, w: Words, target: dict[str, int], direction: int, polar: bool, corner) -> None:
        ex, s = self.ex, self.s
        plane = self.plane()
        a, b, n = PLANES[plane]
        if a not in ex.by_letter or b not in ex.by_letter:
            raise Alarm("illegal_plane", f"plane {plane}", a)
        feed = self.cutting_feed()
        start = (ex.si(a, s.pos[a]), ex.si(b, s.pos[b]))
        centre_letters = {"X": "I", "Y": "J", "Z": "K"}
        centre_words = {letter: w.take(centre_letters[letter]) for letter in (a, b)}
        for letter, name in centre_letters.items():
            if letter not in (a, b) and w.has(name):
                raise Alarm("illegal_plane", f"{name} is not in the {plane} plane", name)
        cr, ar = w.take("CR"), w.take("AR")
        turns_word = w.take("TURN")
        turns = int(round_half_away(V._number(turns_word[1], "TURN"))) if turns_word is not None else 0
        if turns < 0:
            raise Alarm("arc_turns", f"TURN={turns}")
        has_centre = any(v is not None for v in centre_words.values())
        end_given = a in target or b in target
        if cr is not None and (has_centre or ar is not None):
            raise Alarm("arc_overdetermined", "CR= with I/J/K or AR=")
        if has_centre:
            centre = []
            for k, letter in enumerate((a, b)):
                word = centre_words[letter]
                if word is None:
                    centre.append(start[k])
                    continue
                value = V._number(word[1], letter)
                if word[0] == "AC":
                    centre.append(self.axis_si(letter, value, True))
                else:
                    centre.append(start[k] + self.length_si(value))
            centre = tuple(centre)
            if ar is not None:
                if end_given:
                    raise Alarm("arc_overdetermined", "AR= with centre and end point")
                angle = math.radians(V._number(ar[1], "AR"))
                if not 0.0 < angle < 2.0 * math.pi:
                    raise Alarm("arc_angle", f"AR={math.degrees(angle):g}")
                cos_a, sin_a = math.cos(direction * angle), math.sin(direction * angle)
                rx, ry = start[0] - centre[0], start[1] - centre[1]
                end = (centre[0] + cos_a * rx - sin_a * ry, centre[1] + sin_a * rx + cos_a * ry)
                target[a] = ex.counts(a, end[0])
                target[b] = ex.counts(b, end[1])
        elif cr is not None or ar is not None:
            end = (ex.si(a, target.get(a, s.pos[a])), ex.si(b, target.get(b, s.pos[b])))
            if end == start:
                raise Alarm("arc_end_point", "full circle with CR=/AR= (needs the centre)")
            if cr is not None:
                radius = self.length_si(V._number(cr[1], "CR"))
                centre = self.radius_centre(start, end, radius, direction)
            else:
                angle = math.radians(V._number(ar[1], "AR"))
                if not 0.0 < angle < 2.0 * math.pi:
                    raise Alarm("arc_angle", f"AR={math.degrees(angle):g}")
                chord = math.hypot(end[0] - start[0], end[1] - start[1])
                radius = chord / (2.0 * math.sin(angle / 2.0))
                offset = radius * math.cos(angle / 2.0)
                mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
                lx, ly = -(end[1] - start[1]) / chord, (end[0] - start[0]) / chord
                centre = (mx + direction * offset * lx, my + direction * offset * ly)
        elif polar:
            centre = tuple(s.sys["pole"])
        else:
            raise Alarm("arc_no_radius", "G2/G3 without centre, CR=, AR= or polar end point")
        self.before_move()
        item = self.arc_by_centre(target, direction, centre, feed, plane=plane, turns=turns,
                                  tolerance=self.tolerance(), corner=corner)
        self.remember_arc(centre, direction)
        return item

    def radius_centre(self, start, end, radius: float, direction: int):
        tol_const, tol_factor = self.tolerance()
        dx, dy = end[0] - start[0], end[1] - start[1]
        chord = math.hypot(dx, dy)
        half = chord / 2.0
        r = abs(radius)
        if r < half:
            if half - r > max(tol_const, tol_factor * half):
                raise Alarm("arc_radius_small", f"CR={r * 1e3:.4f} mm, half chord {half * 1e3:.4f} mm")
            r = half
        h = math.sqrt(max(r * r - half * half, 0.0))
        side = direction * (1.0 if radius > 0.0 else -1.0)
        mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
        return mx + side * h * (-dy / chord), my + side * h * (dx / chord)

    def remember_arc(self, centre, direction: int) -> None:
        ex, s = self.ex, self.s
        a, b, _ = PLANES[self.plane()]
        end = (ex.si(a, s.pos[a]), ex.si(b, s.pos[b]))
        s.sys["tangent"] = list(arc_end_tangent(centre, end, direction))
        s.sys["last_pos"] = [s.pos[a], s.pos[b]]

    def cip(self, w: Words, target: dict[str, int], corner) -> None:
        ex, s = self.ex, self.s
        plane = self.plane()
        a, b, n = PLANES[plane]
        feed = self.cutting_feed()
        start = (ex.si(a, s.pos[a]), ex.si(b, s.pos[b]))
        names = {"X": "I1", "Y": "J1", "Z": "K1"}
        mid = []
        mid_normal = None
        for letter in (a, b, n):
            word = w.take(names[letter])
            if word is None:
                if letter in (a, b):
                    raise Alarm("cip_collinear", f"CIP without {names[letter]}")
                continue
            absolute = {"AC": True, "IC": False}.get(word[0], self.absolute())
            si = self.axis_si(letter, V._number(word[1], names[letter]), absolute)
            current = ex.si(letter, s.pos.get(letter, 0)) if letter in ex.by_letter else 0.0
            value = si if absolute else current + si
            if letter == n:
                mid_normal = value
            else:
                mid.append(value)
        end = (ex.si(a, target.get(a, s.pos[a])), ex.si(b, target.get(b, s.pos[b])))
        centre = circle_through(start, tuple(mid), end)
        if centre is None:
            raise Alarm("cip_collinear", "start, intermediate and end point on a line")
        if mid_normal is not None and n in ex.by_letter:
            z0, z1 = ex.si(n, s.pos[n]), ex.si(n, target.get(n, s.pos[n]))
            sweep_mid = _angle(centre, start, mid)
            sweep_end = _angle(centre, start, end)
            expect = z0 + (z1 - z0) * (sweep_mid / sweep_end if sweep_end else 0.0)
            if abs(expect - mid_normal) > self.tolerance()[0]:
                raise Alarm("unsupported_function", "CIP in a plane inclined to the working plane")
        cross = (mid[0] - start[0]) * (end[1] - mid[1]) - (mid[1] - start[1]) * (end[0] - mid[0])
        direction = 1 if cross > 0.0 else -1
        self.before_move()
        self.arc_by_centre(target, direction, centre, feed, plane=plane, tolerance=self.tolerance(), corner=corner)
        self.remember_arc(centre, direction)

    def ct(self, target: dict[str, int], corner) -> None:
        ex, s = self.ex, self.s
        plane = self.plane()
        a, b, _ = PLANES[plane]
        tangent = s.sys.get("tangent")
        if tangent is None:
            raise Alarm("arc_tangential", "CT without a preceding contour element in this plane")
        feed = self.cutting_feed()
        start = (ex.si(a, s.pos[a]), ex.si(b, s.pos[b]))
        end = (ex.si(a, target.get(a, s.pos[a])), ex.si(b, target.get(b, s.pos[b])))
        found = tangent_arc_centre(start, end, tangent)
        if found is None:
            raise Alarm("arc_tangential", "CT end point on the tangent")
        centre, direction = found
        self.before_move()
        self.arc_by_centre(target, direction, centre, feed, plane=plane, tolerance=self.tolerance(), corner=corner)
        self.remember_arc(centre, direction)

    def lead_word(self, w: Words, names=("I", "J", "K")) -> float:
        leads = {name: w.take(name) for name in names}
        given = [(name, word) for name, word in leads.items() if word is not None]
        if not given:
            raise Alarm("thread_pitch_zero", "no pitch (I, J or K)", "?")
        if len(given) > 1:
            raise Alarm("unsupported_function", "taper thread with two pitches")
        name, word = given[0]
        lead = self.length_si(V._number(word[1], name))
        if lead == 0.0:
            raise Alarm("thread_pitch_zero", f"{name}=0", name)
        return lead

    def thread(self, w: Words, target: dict[str, int]) -> None:
        ex = self.ex
        lead = abs(self.lead_word(w))
        sf = w.take("SF")
        start = math.radians(V._number(sf[1], "SF")) if sf is not None else 0.0
        feed = ex.feed_spec(False, override=False, feed_si=lead, mode="per_rev")
        ex.move("linear", target, feed, modal={"thread": {"lead_m": lead, "start_angle_rad": start}})
        self.remember_line(target)

    def tap(self, w: Words, target: dict[str, int], code: str) -> None:
        ex, s = self.ex, self.s
        lead = self.lead_word(w)
        if code == "G331" and w.has("S"):
            word = w.take("S")
            s.spindle.s_value = V._number(word[1], "S")
        hand = 1 if lead > 0.0 else -1
        feed = ex.feed_spec(False, override=False, feed_si=abs(lead), mode="per_rev")
        phase = "in" if code == "G331" else "out"
        ex.move("linear", target, feed, modal={"rigid_tap": True,
                                               "tap": {"lead_m": abs(lead), "direction": hand, "phase": phase}})
        self.remember_line(target)

    def orientation_words(self, w: Words, target: dict[str, int]) -> None:
        """A3= B3= C3= (tool direction in the WCS) under TRAORI: the rotary axes of the table solution."""
        names = [n for n in ("A3", "B3", "C3") if w.has(n)]
        if not names:
            return
        vector = [V._number(w.take(n)[1], n) if n in names else 0.0 for n in ("A3", "B3", "C3")]
        if not any(vector):
            raise Alarm("illegal_value", "zero tool direction")
        solution = self.choose_table_solution(vector)
        for axis in self.ex.axes:
            if axis.name in solution:
                target[axis.letter] = self.ex.counts(axis.letter, solution[axis.name])

    def reference(self, code: str, w: Words) -> bool:
        ex = self.ex
        axes = dict(w.axes1)
        axes.update(w.axes)
        if not axes:
            raise Alarm("syntax", f"{code} without axes", code)
        target = {}
        point = None
        if code == "G75":
            fp = w.take("FP")
            n = int(round_half_away(V._number(fp[1], "FP"))) if fp is not None else 1
            point = (self.p.get("reference_points_m") or {}).get(n) if n > 1 else None
            if n > 1 and point is None:
                raise Alarm("illegal_value", f"G75 FP={n}: fixed point not set (physics.controller.reference_points_mm)")
        for letter in axes:
            axis = ex.by_letter.get(letter)
            if axis is None:
                raise Alarm("illegal_plane", f"axis {letter}", letter)
            if point is not None:
                value = float(point.get(axis.name, point.get(letter, 0.0)))
                target[letter] = int(round_half_away(value / axis.quantum))
            else:
                target[letter] = axis.home
            w.used.add(letter)
        ex.flush_comp(code)
        ex.machine_move("rapid", target, ex.feed_spec(True), modal={"reference_point": 1 if code == "G74" else 2})
        return True

    def machine_block(self, code: str, w: Words) -> bool:
        """G53 / G153 / SUPA: the block's positions in machine coordinates (tool offset kept)."""
        ex, s = self.ex, self.s
        target = {}
        for letter, (mod, value) in w.axes.items():
            axis = ex.by_letter.get(letter)
            if axis is None:
                raise Alarm("illegal_plane", f"axis {letter}", letter)
            w.used.add(letter)
            absolute = {"AC": True, "IC": False}.get(mod, self.absolute())
            counts = ex.counts(letter, self.axis_si(letter, V._number(value, letter), absolute))
            tool = self.tool_offset_counts(letter)
            target[letter] = counts + tool if absolute else s.mach.get(letter, 0) + counts
        if not target:
            return False
        motion = s.modal.get("motion", "G0")
        ex.flush_comp(code)
        if motion == "G1":
            ex.machine_move("linear", target, self.cutting_feed(), modal={"machine_coordinates": True})
        else:
            ex.machine_move("rapid", target, ex.feed_spec(True), modal={"machine_coordinates": True})
        return True

    # ------------------------------------------------------------------ calls, modal calls, keywords
    def set_mcall(self, call) -> None:
        s = self.s
        if call is None:
            s.sys.pop("mcall", None)
            return
        _, name, args, repeat = call
        from . import cycles

        if name not in cycles.CYCLES:
            self.find_program(name)
        s.sys["mcall"] = {"name": name, "args": [list(a) if a is not None else None for a in args],
                          "depth": len(s.frames)}

    def modal_call_after_move(self) -> None:
        s = self.s
        mcall = s.sys.get("mcall")
        if not mcall or len(s.frames) != mcall["depth"]:
            return
        args = [tuple(a) if a is not None else None for a in mcall["args"]]
        self.call(("call", mcall["name"], args, None), None, modal=True)

    def call(self, call: tuple, env, modal: bool = False) -> None:
        _, name, args, repeat_node = call
        from . import cycles

        repeat = 1
        if repeat_node is not None:
            value = V.Env(self).number(repeat_node, "P") if env is None else env.number(repeat_node, "P")
            repeat = int(round_half_away(value))
            if repeat != value:
                raise Alarm("pass_count", f"P{value:g}")
        if name in cycles.CYCLES:
            values = [a[1] if a is not None else None for a in args]
            for _ in range(repeat):
                cycles.run(self, name, values)
            return
        builtin = BUILTIN_PROCEDURES.get(name)
        if builtin is not None:
            if repeat_node is not None:
                raise Alarm("unexpected_parameter", f"P with {name}", "P")
            builtin(self, [a[1] if a is not None else None for a in args])
            return
        if name in STANDARD_CYCLES and name not in self.program_names():
            raise Alarm("unsupported_function", f"{name}: a standard 840D sl cycle this build does not model")
        self.call_program(name, args, repeat)

    def keyword_action(self, kw: str) -> None:
        ex, s = self.ex, self.s
        if kw == "STOPRE":
            self.preprocessing_stop("STOPRE")
        elif kw == "TRAORI":
            if s.modal.get("7", "G40") != "G40":
                raise Alarm("transform_active", "TRAORI with tool radius compensation")
            if not self.machine.table_kinematics_supported():
                raise Alarm("transform_unavailable", "TRAORI needs a five-axis machine with rotary table axes")
            s.tcp = "siemens840d:TRAORI"
            ex.resync_program_position()
        elif kw == "TRAFOOF":
            if s.modal.get("7", "G40") != "G40":
                raise Alarm("transform_active", "TRAFOOF with tool radius compensation")
            s.tcp = "siemens840d:CYCLE800" if s.sys.get("swivel") is not None else ""
            ex.resync_program_position()
        elif kw == "RET":
            self.sub_end("RET")

    def cycle_call(self, code: str, data: dict) -> CycleCall:
        """A CycleCall describing the current state (for providers that need it)."""
        ex, s = self.ex, self.s
        position = {l: ex.si(l, c) for l, c in s.pos.items()}
        return CycleCall(code=code, dialect=self.d.name, plane=self.plane(), absolute=self.absolute(),
                         position=position, current=dict(position), initial_level=None, return_to_initial=True,
                         data=data, words={}, feed_mode=self.feed_mode(), units=s.units)


def _clean(value: float) -> float:
    if value == 0.0:
        return 0.0
    return float(f"{value:.12g}")


def _angle(centre, start, point) -> float:
    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(point[1] - centre[1], point[0] - centre[0])
    return (a1 - a0) % (2.0 * math.pi)


# ====================================================================== built-in procedures

def _msg(handler, args) -> None:
    s = handler.s
    text = "" if not args or args[0] is None else str(args[0])
    s.sys["msg"] = text
    if text:
        handler.ex.note("info", f"MSG: {text}")


def _setal(handler, args) -> None:
    if not args or args[0] is None:
        raise Alarm("call_parameters", "SETAL needs an alarm number")
    number = int(round_half_away(V._number(args[0], "SETAL")))
    if not 60000 <= number <= 69999:
        raise Alarm("call_parameters", f"SETAL({number}): cycle alarms are 60000-69999")
    entry = handler.d.alarms.by_code.get(str(number))
    text = str(entry["text"]) if entry is not None else f"Cycle alarm {number}"
    text = handler.alarm_text(text, " ".join(str(a) for a in args[1:] if a is not None))
    raise ControllerAlarm("program_alarm", f"SETAL({number})", str(number), text)


def _waitm(handler, args) -> None:
    ex = handler.ex
    values = [int(round_half_away(V._number(a, "WAITM"))) for a in args if a is not None]
    if len(values) < 2:
        raise Alarm("call_parameters", "WAITM(marker, channel, ...)")
    marker, channels = values[0], values[1:]
    n_paths = len(handler.machine.paths)
    if any(c < 1 or c > n_paths for c in channels):
        raise Alarm("wait_illegal_p", f"WAITM({', '.join(str(v) for v in values)}): channel not on this machine")
    paths = sorted({c - 1 for c in channels} | {ex.ctx.path})
    if len(paths) == 1:
        ex.note("info", f"WAITM({marker}): no other channel named, no wait")
        return
    handler.path_wait(marker, paths, f"WAITM({marker})")


BUILTIN_PROCEDURES = {"MSG": _msg, "SETAL": _setal, "WAITM": _waitm}

__all__ = ["SiemensHandler", "BUILTIN_PROCEDURES"]
