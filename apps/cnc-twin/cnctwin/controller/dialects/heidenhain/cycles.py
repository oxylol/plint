"""HEIDENHAIN TNC 640 fixed cycles (heidenhain-tnc640-cycles-2023; db words.cycles).

CYCL DEF 12 PGM CALL (12.1 PGM name) and SEL CYCLE name make a program the CALL-active cycle: CYCL CALL, M99
and M89 call it like CALL PGM at the current position (CYCL CALL POS / PAT of it: TW0012).

DEF-active (take effect when defined): 7 DATUM SHIFT (7.1 X+.. / IX.., per axis, absolute or added to the
previous shift), 8 MIRRORING (8.1 axes; none = reset), 9 DWELL TIME (9.1 DWELL t), 10 ROTATION (10.1 ROT /
IROT), 11 SCALING (11.1 SCL f), 13 ORIENTATION (13.1 ANGLE a), 26 AXIS-SPECIFIC SCALING (26.1 X.. Y.. Z.. CCX
CCY CCZ), 32 TOLERANCE (32.1 T, 32.2 HSC-MODE TA), 247 PRESETTING (Q339).

CALL-active (CYCL CALL, CYCL CALL POS, CYCL CALL PAT, M99 per block, M89 modal until the next CYCL DEF):
200 DRILLING, 201 REAMING, 202 BORING, 203 UNIVERSAL DRILLING, 204 BACK BORING, 205 UNIVERSAL PECKING, 206
TAPPING (floating holder), 207 RIGID TAPPING, 208 BORE MILLING, 209 TAPPING W/ CHIP BRKG, 240 CENTERING, 251
RECTANGULAR POCKET, 252 CIRCULAR POCKET, 253 SLOT MILLING. The tool is at the cycle's position in the working plane
when it is called; along the tool axis: rapid to the set-up clearance Q203 + Q200 (Q201 depth < 0 into the
material; > 0 is error 280-0435; 0 does nothing), the machining, then rapid to Q203 + Q200 or, when larger,
to the 2nd set-up clearance Q203 + Q204. Feeds Q206 / Q207 / Q208 / Q385 in mm/min (FMAX = rapid where the
manual allows it). CYCL CALL POS X Y Z: rapid to the position (first in the plane when the tool is above
Q203, else first up to the clearance), Z adds to the surface coordinate. CYCL CALL PAT: every point of the
PATTERN DEF at the 2nd set-up clearance, its Z adding to Q203. With POS and PAT the cycle is expanded when
the tool has reached each position (cycles that work about the hole centre take it from there).
"""

from __future__ import annotations

import math

from ... import hooks
from ...expr import round_half_away
from ...hooks import Step
from ...ir import Command, Kind
from ...pipeline import PLANES
from ..siemens840d import milling as M
from ..siemens840d.kit import Alarm

ACTION = "heidenhain:action"
CALL_ACTIVE = (200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 240, 251, 252, 253)
DEF_ACTIVE = (7, 8, 9, 10, 11, 13, 26, 32, 247)


class Action:
    """Spindle / stop actions inside a cycle expansion, in sequence with the moves."""

    def resume(self, ex, data: dict, feedback) -> list:
        h = ex.handler
        s = ex.s
        kind = data["action"]
        if kind == "spindle":
            s.spindle.direction = int(data["direction"])
            if data.get("rpm") is not None:
                s.spindle.s_value = float(data["rpm"])
            h.emit_spindle()
        elif kind == "orient":
            s.spindle.direction = 0
            s.spindle.oriented = True
            h.emit_spindle(orient=math.radians(float(data["angle_deg"])))
        elif kind == "cycle":
            # a cycle of CYCL CALL POS / PAT, expanded when the tool has reached its position
            return expand(h, data["cycle"], float(data["shift"]), shift_surface=bool(data["surface"]))
        return []


hooks.register_resumer(ACTION, Action(), replace=True)


def deferred(cycle: dict, surface_shift: float, shift_surface: bool) -> Step:
    return M.action(ACTION, action="cycle", cycle={"number": int(cycle["number"]), "params": dict(cycle["params"])},
                    shift=float(surface_shift), surface=bool(shift_surface))


def q(params: dict, name: str, default: float | None = None) -> float:
    value = params.get(name, default)
    if value is None:
        raise Alarm("cycle_incomplete", f"{name} missing")
    return float(value)


# ====================================================================== definition

def define(h, f) -> None:
    s = h.s
    number, sub = f["number"], f["sub"]
    if number in CALL_ACTIVE:
        params = {name: h.number(node) for name, node in f["params"].items()}
        for name, value in params.items():
            h.q.set("Q", int(name[1:]), value)
        s.sys["cycle_def"] = {"number": number, "params": params}
        s.sys["cycle_name"] = f"CYCL {number}"
        s.sys.pop("m89", None)
        return
    if number == 12:
        # 12.0 PGM CALL, 12.1 PGM <name>: the program becomes the CALL-active cycle
        if sub is None or sub == 0:
            s.sys["def_cycle"] = {"number": 12, "words": []}
            return
        words = [w for w in f["words"] if w != "PGM"]
        if s.sys.get("def_cycle", {}).get("number") != 12 or len(words) != 1:
            raise Alarm("cycle_incomplete", "CYCL DEF 12.1 PGM <name> after 12.0")
        from .klartext import program_name

        program_cycle(h, program_name(words[0]))
        return
    if number not in DEF_ACTIVE:
        from .tables import TNC640_CYCLES

        if number in TNC640_CYCLES:
            raise Alarm("unsupported_function", f"CYCL DEF {number} (a TNC 640 cycle this build does not model)")
        raise Alarm("cycle_undefined", f"CYCL DEF {number}")
    if number == 247:
        preset = int(round_half_away(h.number(f["params"].get("Q339", ("num", 0.0)))))
        select_preset(h, preset)
        return
    if sub is None or sub == 0:
        s.sys["def_cycle"] = {"number": number, "words": []}
        if number == 8:
            s.sys["hh"]["mirror"] = [0, 0, 0]
            h.update_transform()
        return
    current = s.sys.get("def_cycle")
    if current is None or current["number"] != number:
        raise Alarm("cycle_incomplete", f"CYCL DEF {number}.{sub} without {number}.0")
    current["words"] = list(current["words"]) + list(f["words"])
    apply_def_cycle(h, number, current["words"])


def words_map(h, words: list[str]) -> list[tuple[str, bool, float | None]]:
    import re

    from .formula import value_node

    out: list = []
    number = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+|Q[LRS]?\d+)$")
    for word in words:
        if word.startswith("#"):
            raise Alarm("unsupported_function", f"datum / preset table entry {word} (CYCL DEF 7 #n, 247 from a "
                        "table): only values in the program and the presets of work_offsets_m are modelled")
        if number.match(word) and out and out[-1][2] is None:
            name, inc, _ = out[-1]
            out[-1] = (name, inc, h.number(value_node(word)))
            continue
        match = re.match(r"^(I?)([A-Z]+(?:-[A-Z]+)?:?)([+-]?(?:\d+\.?\d*|\.\d+|Q[LRS]?\d+))?$", word)
        if match is None:
            raise Alarm("cycle_incomplete", f"cycle word {word}")
        value = h.number(value_node(match.group(3))) if match.group(3) else None
        out.append((match.group(2), match.group(1) == "I", value))
    return out


def apply_def_cycle(h, number: int, words: list[str]) -> None:
    s = h.s
    hh = s.sys["hh"]
    items = words_map(h, words)
    if number == 7:
        values = {name: (inc, value) for name, inc, value in items if value is not None}
        for name in values:
            if name not in h.ex.by_letter:
                raise Alarm("illegal_axis", f"datum shift axis {name}")
        h.datum_shift(values)
    elif number == 8:
        mirror = [0, 0, 0]
        for name, inc, value in items:
            if name not in "XYZ" or len(name) != 1:
                raise Alarm("transform_illegal", f"mirroring {name}")
            mirror["XYZ".index(name)] = 1
        hh["mirror"] = mirror
        h.update_transform()
    elif number == 10:
        for name, inc, value in items:
            if name != "ROT" or value is None:
                raise Alarm("rotation_not_permitted", f"cycle 10 {name}")
            hh["rot"] = float(hh["rot"] + value if inc else value)
        h.update_transform()
    elif number == 11:
        pending = None
        for name, inc, value in items:
            if name == "SCL":
                pending = value
            elif pending is None and value is not None and name.isdigit():
                pending = value
        if pending is None:
            values = [v for n, _, v in items if v is not None]
            pending = values[-1] if values else None
        if pending is None or pending <= 0.0:
            raise Alarm("illegal_scale", f"SCL {pending}")
        hh["scale"] = [pending] * 3
        hh["centre"] = [0.0, 0.0, 0.0]
        h.update_transform()
    elif number == 26:
        scale = [1.0, 1.0, 1.0]
        centre = [0.0, 0.0, 0.0]
        for name, inc, value in items:
            if name in ("X", "Y", "Z"):
                if value is None or value <= 0.0:
                    raise Alarm("illegal_scale", f"{name}{value}")
                scale["XYZ".index(name)] = value
            elif name in ("CCX", "CCY", "CCZ"):
                centre["XYZ".index(name[2])] = h.length_si(value or 0.0)
            else:
                raise Alarm("illegal_scale", f"cycle 26 {name}")
        hh["scale"] = scale
        hh["centre"] = centre
        h.update_transform()
    elif number == 9:
        for name, inc, value in items:
            if name == "DWELL" and value is not None:
                if value < 0.0:
                    raise Alarm("cycle_incomplete", "negative dwell")
                h.ex.dwell(value)
    elif number == 13:
        for name, inc, value in items:
            if name == "ANGLE" and value is not None:
                s.spindle.direction = 0
                s.spindle.oriented = True
                h.emit_spindle(orient=math.radians(value))
    elif number == 32:
        tolerance = None
        mode = None
        for name, inc, value in items:
            if name == "T":
                tolerance = value
            elif name == "HSC-MODE:":
                mode = value
        if tolerance is not None:
            tol = h.length_si(tolerance)
            s.smoothing.update({"corner": "blend", "tolerance_m": tol, "mode_code": f"CYCL DEF 32 T{tolerance:g}"})
            h.ex.emit(Command(Kind.MODE, h.ex.line_ref(), smoothing=h.ex.smoothing_spec(),
                              modal={"tolerance_cycle": {"tolerance_m": tol, "hsc_mode": mode}}))


def select_preset(h, number: int) -> None:
    ex, s = h.ex, h.s
    key = f"P{number}"
    if key not in s.offsets:
        raise Alarm("shift_not_permitted", f"preset {number} not in the preset table (work_offsets_m)")
    if s.wcs != key:
        s.wcs = key
        ex.resync_program_position()


# ====================================================================== patterns

def pattern_def(h, f) -> None:
    points = []
    for kind, number, values in f["patterns"]:
        v = {name: h.number(node) for name, node in values.items()}
        z = h.length_si(v.get("Z", 0.0))
        if kind == "POS":
            # incremental IX / IY / IZ refer to the previous position of the pattern (the first one to the
            # tool's position in the working plane and to the surface 0)
            last = points[-1] if points else (*h.plane_point(), 0.0)
            point = []
            for k, axis in enumerate("XYZ"):
                if "I" + axis in v:
                    point.append(last[k] + h.length_si(v["I" + axis]))
                else:
                    point.append(h.length_si(v.get(axis, 0.0)))
            points.append(tuple(point))
        elif kind == "ROW":
            angle = math.radians(v.get("ROT", 0.0))
            count = int(round_half_away(v.get("NUM", 0.0)))
            step = h.length_si(v.get("D", 0.0))
            x0, y0 = h.length_si(v.get("X", 0.0)), h.length_si(v.get("Y", 0.0))
            points += [(x0 + k * step * math.cos(angle), y0 + k * step * math.sin(angle), z) for k in range(count)]
        elif kind in ("PAT", "FRAME"):
            # ROT turns the whole pattern; ROTX / ROTY turn only its main / secondary axis further
            # (a sheared grid): column direction at ROT + ROTX, row direction at 90 deg + ROT + ROTY
            main = math.radians(v.get("ROT", 0.0) + v.get("ROTX", 0.0))
            secondary = math.radians(v.get("ROT", 0.0) + v.get("ROTY", 0.0))
            nx, ny = int(round_half_away(v.get("NUMX", 0.0))), int(round_half_away(v.get("NUMY", 0.0)))
            dx, dy = h.length_si(v.get("DX", 0.0)), h.length_si(v.get("DY", 0.0))
            x0, y0 = h.length_si(v.get("X", 0.0)), h.length_si(v.get("Y", 0.0))
            ux, uy = math.cos(main), math.sin(main)
            wx, wy = -math.sin(secondary), math.cos(secondary)
            for j in range(ny):
                for i in range(nx):
                    if kind == "FRAME" and 0 < i < nx - 1 and 0 < j < ny - 1:
                        continue
                    u, w = i * dx, j * dy
                    points.append((x0 + ux * u + wx * w, y0 + uy * u + wy * w, z))
        elif kind in ("CIRC", "PITCHCIRC"):
            count = int(round_half_away(v.get("NUM", 0.0)))
            radius = h.length_si(v.get("D", 0.0)) / 2.0
            start = v.get("START", 0.0)
            step = v.get("STEP", 0.0) if kind == "PITCHCIRC" else (360.0 / count if count else 0.0)
            x0, y0 = h.length_si(v.get("X", 0.0)), h.length_si(v.get("Y", 0.0))
            points += [(x0 + radius * math.cos(math.radians(start + k * step)),
                        y0 + radius * math.sin(math.radians(start + k * step)), z) for k in range(count)]
        else:
            raise Alarm("pattern_missing", kind)
    if not points:
        raise Alarm("pattern_missing", "empty pattern")
    h.s.sys["pattern"] = [[float(x), float(y), float(z)] for x, y, z in points]


# ====================================================================== calls

def cycle_plane(h):
    a, b, n = PLANES[h.plane()]
    for letter in (a, b, n):
        if letter not in h.ex.by_letter:
            raise Alarm("illegal_axis", f"axis {letter}")
    return M.Plane(a, b, n, 1.0)


def program_cycle(h, name: str) -> None:
    """CYCL DEF 12 PGM CALL / SEL CYCLE: a program called like a subprogram by CYCL CALL, M99 and M89."""
    h.find_program(name)
    h.s.sys["cycle_def"] = {"number": 12, "params": {}, "program": name}
    h.s.sys["cycle_name"] = f"PGM {name}"
    h.s.sys.pop("m89", None)
    h.s.sys.pop("def_cycle", None)


def call_defined(h, f) -> None:
    """CYCL CALL / M99 / M89: the defined CALL-active cycle at the current position."""
    cycle = h.s.sys.get("cycle_def")
    if cycle is None:
        raise Alarm("cycle_missing", "no CALL-active cycle defined")
    if cycle["number"] == 12:
        h.do_call_pgm({"name": cycle["program"]})
        return
    h.ex.queue_steps(expand(h, cycle, 0.0))


def cycl_call(h, f) -> None:
    ex, s = h.ex, h.s
    cycle = s.sys.get("cycle_def")
    if cycle is None:
        raise Alarm("cycle_missing", "CYCL CALL without CYCL DEF")
    before, after = h.m_split(f.get("m", []))
    rapid, _ = h.feed_for(f)
    for m in before:
        h.m_function(m, None)
    variant = f["variant"]
    if cycle["number"] == 12:
        if variant:
            raise Alarm("unsupported_function", f"CYCL CALL {variant} of a program cycle (CYCL DEF 12 / SEL CYCLE)")
        call_defined(h, f)
        for m in after:
            h.m_function(m, None)
        return
    plane = cycle_plane(h)
    steps: list[Step] = []
    if variant == "":
        steps = expand(h, cycle, 0.0)
    elif variant == "POS":
        target, _ = _pos_target(h, f)
        params = cycle["params"]
        surface = h.length_si(q(params, "Q203", 0.0))
        steps = _position(h, plane, target, surface + h.length_si(q(params, "Q204", 0.0)))
        steps.append(deferred(cycle, target.get(plane.n, 0.0), True))
    elif variant == "PAT":
        points = s.sys.get("pattern")
        if not points:
            raise Alarm("pattern_missing", "CYCL CALL PAT without PATTERN DEF")
        params = cycle["params"]
        clearance = h.length_si(q(params, "Q203", 0.0)) + max(h.length_si(q(params, "Q204", 0.0)),
                                                             h.length_si(q(params, "Q200", 0.0)))
        for x, y, z in points:
            steps.append(M.rapid(plane, n=clearance + z))
            steps.append(M.rapid(plane, a=x, b=y))
            steps.append(deferred(cycle, z, True))
    else:
        raise Alarm("syntax", f"CYCL CALL {variant}")
    ex.queue_steps(steps)
    for m in after:
        h.m_function(m, None)


def _pos_target(h, f):
    plane = cycle_plane(h)
    values = {}
    for letter, inc, node in f["coords"]:
        value = h.value(node, lone=True)
        if value is not None:
            si = h.axis_si(letter, value)
            current = h.ex.si(letter, h.ex.s.pos.get(letter, 0))
            values[letter] = current + si if inc else si
    missing = [l for l in (plane.a, plane.b, plane.n) if l not in values]
    if missing:
        raise Alarm("cycle_incomplete", "CYCL CALL POS needs the three coordinates")
    return values, plane


def _position(h, plane, target: dict, clearance: float) -> list[Step]:
    """Positioning logic of CYCL CALL POS: above the surface move in the plane first, else up first."""
    ex = h.ex
    z_now = ex.si(plane.n, ex.s.pos.get(plane.n, 0))
    if z_now >= clearance:
        return [M.rapid(plane, a=target[plane.a], b=target[plane.b]), M.rapid(plane, n=max(target[plane.n], clearance))]
    return [M.rapid(plane, n=clearance), M.rapid(plane, a=target[plane.a], b=target[plane.b])]


def expand(h, cycle: dict, surface_shift: float, shift_surface: bool = False) -> list[Step]:
    number = cycle["number"]
    params = dict(cycle["params"])
    plane = cycle_plane(h)
    if shift_surface:
        params["Q203"] = params.get("Q203", 0.0) + surface_shift / h.length_si(1.0)
    function = {200: drill_200, 201: ream_201, 202: bore_202, 203: drill_203, 204: back_bore_204, 205: peck_205,
                206: tap_206,
                207: tap_207, 208: bore_mill_208, 209: tap_209, 240: centre_240, 251: pocket_251, 252: pocket_252,
                253: slot_253}[number]
    steps = function(h, plane, params)
    h.check_steps_scale(steps)
    return steps


# ====================================================================== drilling cycles

def levels(h, params, depth_name: str = "Q201"):
    surface = h.length_si(q(params, "Q203", 0.0))
    clearance = h.length_si(abs(q(params, "Q200", 0.0)))
    relative = h.length_si(q(params, depth_name))
    second = h.length_si(q(params, "Q204", 0.0))
    if relative > 0.0:
        raise Alarm("depth_sign", f"{depth_name} > 0")
    retract = surface + max(clearance, second)
    depth = surface + relative if relative != 0.0 else 0.0
    return surface, clearance, depth, retract


def feed(h, params, name: str) -> float:
    value = params.get(name)
    if value is None:
        raise Alarm("cycle_incomplete", f"{name} missing")
    if value <= 0.0:
        raise Alarm("feed_zero", f"{name}")
    return float(value) * h.feed_unit_m() / 60.0


def dwell(seconds: float) -> list[Step]:
    return M.dwell(seconds=seconds) if seconds > 0.0 else []


def depth_reference(params) -> None:
    if params.get("Q395", 0.0):
        raise Alarm("unsupported_function", "Q395 = 1: depth to the full diameter needs the tool point angle")


def drill_200(h, plane, params) -> list[Step]:
    depth_reference(params)
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    step = h.length_si(abs(q(params, "Q202")))
    if step <= 0.0:
        raise Alarm("q202_missing", "Q202")
    top_dwell, bottom_dwell = q(params, "Q210", 0.0), q(params, "Q211", 0.0)
    start = surface + clearance
    out = [M.rapid(plane, n=start)]
    reached = surface
    while reached > depth + 1e-12:
        level = max(depth, reached - step)
        if reached < surface:
            out.append(M.rapid(plane, n=reached + clearance))
        out.append(M.feed(plane, f, n=level))
        reached = level
        if reached > depth + 1e-12:
            out.append(M.rapid(plane, n=start))
            out += dwell(top_dwell)
    out += dwell(bottom_dwell)
    out.append(M.rapid(plane, n=retract))
    return out


def ream_201(h, plane, params) -> list[Step]:
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    back = feed(h, params, "Q208") if params.get("Q208") else f
    out = [M.rapid(plane, n=surface + clearance), M.feed(plane, f, n=depth)] + dwell(q(params, "Q211", 0.0))
    out.append(M.feed(plane, back, n=surface + clearance))
    if retract > surface + clearance:
        out.append(M.rapid(plane, n=retract))
    return out


DISENGAGE = {1: ("a", -1.0), 2: ("b", -1.0), 3: ("a", 1.0), 4: ("b", 1.0)}


def bore_202(h, plane, params) -> list[Step]:
    ex = h.ex
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    back = feed(h, params, "Q208") if params.get("Q208") else f
    direction = int(q(params, "Q214", 0.0))
    angle = q(params, "Q336", 0.0)
    if direction not in (0, 1, 2, 3, 4):
        raise Alarm("cycle_incomplete", "Q214")
    spindle = ex.s.spindle.direction
    centre = (ex.si(plane.a, ex.s.pos[plane.a]), ex.si(plane.b, ex.s.pos[plane.b]))
    out = [M.rapid(plane, n=surface + clearance), M.feed(plane, f, n=depth)] + dwell(q(params, "Q211", 0.0))
    out.append(M.action(ACTION, action="orient", angle_deg=angle))
    if direction:
        axis, sign = DISENGAGE[direction]
        shift = sign * float(h.p["boring_disengage_m"])
        point = {"a": centre[0], "b": centre[1]}
        point[axis] += shift
        out.append(M.rapid(plane, a=point["a"], b=point["b"]))
    out.append(M.feed(plane, back, n=surface + clearance))
    if direction:
        out.append(M.rapid(plane, a=centre[0], b=centre[1]))
    if retract > surface + clearance:
        out.append(M.rapid(plane, n=retract))
    if spindle:
        out.append(M.action(ACTION, action="spindle", direction=spindle))
    return out


def back_bore_204(h, plane, params) -> list[Step]:
    """Counterbore on the underside: rapid to Q203 + Q200, oriented stop at Q336, the tool off centre by
    Q251 toward Q214 (1 -a, 2 -b, 3 +a, 4 +b), down at Q253 until the cutting edge (Q252 above the tool tip)
    is Q200 below the underside Q203 - Q250, back to the centre, the spindle on again, up at Q254 until the
    edge is Q249 above the underside, dwell Q255, oriented stop, off centre, out at Q253 to Q203 + Q200,
    rapid to the 2nd set-up clearance, back to the centre, the spindle as before."""
    ex = h.ex
    surface = h.length_si(q(params, "Q203", 0.0))
    clearance = h.length_si(abs(q(params, "Q200", 0.0)))
    second = h.length_si(q(params, "Q204", 0.0))
    depth = h.length_si(q(params, "Q249"))
    if depth < 0.0:
        raise Alarm("depth_sign", "Q249 < 0 (back boring works toward +tool axis)")
    if depth == 0.0:
        return []
    thickness = h.length_si(q(params, "Q250"))
    if thickness <= 0.0:
        raise Alarm("cycle_incomplete", "Q250 material thickness <= 0")
    offset = h.length_si(abs(q(params, "Q251")))
    edge = h.length_si(abs(q(params, "Q252")))
    pre, cut = feed(h, params, "Q253"), feed(h, params, "Q254")
    direction = int(q(params, "Q214", 0.0))
    if direction not in DISENGAGE:
        raise Alarm("cycle_incomplete", "Q214 must be 1 to 4 in cycle 204")
    spindle = ex.s.spindle.direction
    if spindle == 0:
        raise Alarm("spindle_turning", "cycle 204 with the spindle stopped")
    angle = q(params, "Q336", 0.0)
    centre = (ex.si(plane.a, ex.s.pos[plane.a]), ex.si(plane.b, ex.s.pos[plane.b]))
    axis, sign = DISENGAGE[direction]
    aside = {"a": centre[0], "b": centre[1]}
    aside[axis] += sign * offset
    underside = surface - thickness
    top = surface + clearance
    out = [M.rapid(plane, n=top), M.action(ACTION, action="orient", angle_deg=angle),
           M.rapid(plane, a=aside["a"], b=aside["b"]),
           M.feed(plane, pre, n=underside - clearance - edge),
           M.feed(plane, pre, a=centre[0], b=centre[1]),
           M.action(ACTION, action="spindle", direction=spindle),
           M.feed(plane, cut, n=underside + depth - edge)]
    out += dwell(q(params, "Q255", 0.0))
    out += [M.action(ACTION, action="orient", angle_deg=angle), M.feed(plane, pre, a=aside["a"], b=aside["b"]),
            M.feed(plane, pre, n=top)]
    if surface + second > top:
        out.append(M.rapid(plane, n=surface + second))
    out += [M.rapid(plane, a=centre[0], b=centre[1]), M.action(ACTION, action="spindle", direction=spindle)]
    return out


def drill_203(h, plane, params) -> list[Step]:
    depth_reference(params)
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    back = feed(h, params, "Q208") if params.get("Q208") else None
    step = h.length_si(abs(q(params, "Q202")))
    decrement = h.length_si(abs(q(params, "Q212", 0.0)))
    breaks = int(q(params, "Q213", 0.0))
    minimum = h.length_si(abs(q(params, "Q205", 0.0)))
    chip = h.length_si(abs(q(params, "Q256", 0.0))) or float(h.p["chip_break_retract_m"])
    top_dwell, bottom_dwell = q(params, "Q210", 0.0), q(params, "Q211", 0.0)
    if step <= 0.0:
        raise Alarm("q202_missing", "Q202")
    start = surface + clearance
    out = [M.rapid(plane, n=start)]
    reached = surface
    count = 0
    current = step
    while reached > depth + 1e-12:
        level = max(depth, reached - current)
        out.append(M.feed(plane, f, n=level))
        reached = level
        current = max(current - decrement, minimum if minimum > 0.0 else 1e-9)
        if reached <= depth + 1e-12:
            break
        count += 1
        if breaks and count < breaks:
            out.append(M.feed(plane, back or f, n=reached + chip) if back else M.rapid(plane, n=reached + chip))
        else:
            count = 0
            out.append(M.feed(plane, back, n=start) if back else M.rapid(plane, n=start))
            out += dwell(top_dwell)
            out.append(M.rapid(plane, n=reached + clearance))
    out += dwell(bottom_dwell)
    out.append(M.feed(plane, back, n=start) if back else M.rapid(plane, n=start))
    if retract > start:
        out.append(M.rapid(plane, n=retract))
    return out


def peck_205(h, plane, params) -> list[Step]:
    depth_reference(params)
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    back = feed(h, params, "Q208") if params.get("Q208") else None
    step = h.length_si(abs(q(params, "Q202")))
    decrement = h.length_si(abs(q(params, "Q212", 0.0)))
    minimum = h.length_si(abs(q(params, "Q205", 0.0)))
    upper = h.length_si(abs(q(params, "Q258", 0.0)))
    lower = h.length_si(abs(q(params, "Q259", 0.0))) or upper
    chip_depth = h.length_si(abs(q(params, "Q257", 0.0)))
    chip = h.length_si(abs(q(params, "Q256", 0.0))) or float(h.p["chip_break_retract_m"])
    deepened = h.length_si(abs(q(params, "Q379", 0.0)))
    pre_feed = feed(h, params, "Q253") if params.get("Q253") else None
    if step <= 0.0:
        raise Alarm("q202_missing", "Q202")
    start = surface + clearance
    out = [M.rapid(plane, n=start)]
    reached = surface
    if deepened > 0.0:
        out.append(M.feed(plane, pre_feed or f, n=surface - deepened + upper))
        reached = surface - deepened
    current = step
    total = abs(depth - surface)
    while reached > depth + 1e-12:
        level = max(depth, reached - current)
        if chip_depth > 0.0:
            mark = reached
            while mark - chip_depth > level + 1e-12:
                mark -= chip_depth
                out.append(M.feed(plane, f, n=mark))
                out.append(M.rapid(plane, n=mark + chip))
        out.append(M.feed(plane, f, n=level))
        reached = level
        current = max(current - decrement, minimum if minimum > 0.0 else 1e-9)
        if reached <= depth + 1e-12:
            break
        out.append(M.feed(plane, back, n=start) if back else M.rapid(plane, n=start))
        fraction = abs(reached - surface) / total if total > 0.0 else 0.0
        advance = upper + (lower - upper) * fraction
        out.append(M.rapid(plane, n=reached + advance))
    out += dwell(q(params, "Q211", 0.0))
    out.append(M.feed(plane, back, n=start) if back else M.rapid(plane, n=start))
    if retract > start:
        out.append(M.rapid(plane, n=retract))
    return out


def tap_206(h, plane, params) -> list[Step]:
    ex = h.ex
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    spindle = ex.s.spindle.direction
    if spindle == 0:
        raise Alarm("spindle_turning", "cycle 206 with the spindle stopped")
    if ex.s.modal.get("7", "G40") != "G40":
        raise Alarm("tap_in_comp", "cycle 206")
    out = [M.rapid(plane, n=surface + clearance),
           M.feed(plane, f, override=False, n=depth, modal={"tapping": True})] + dwell(q(params, "Q211", 0.0))
    out.append(M.action(ACTION, action="spindle", direction=-spindle))
    out.append(M.feed(plane, f, override=False, n=surface + clearance, modal={"tapping": True}))
    out.append(M.action(ACTION, action="spindle", direction=spindle))
    if retract > surface + clearance:
        out.append(M.rapid(plane, n=retract))
    return out


def _rigid(h, plane, params, peck: float, chip: float, retract_factor: float) -> list[Step]:
    ex = h.ex
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    pitch = h.length_si(q(params, "Q239"))
    if pitch == 0.0:
        raise Alarm("cycle_incomplete", "Q239 = 0")
    if ex.s.spindle.s_value <= 0.0:
        raise Alarm("tap_no_spindle", "rigid tapping without S")
    if ex.s.modal.get("7", "G40") != "G40":
        raise Alarm("tap_in_comp", "rigid tapping")
    hand = 1 if pitch > 0.0 else -1
    lead = abs(pitch)
    rpm = ex.s.spindle.s_value
    tap = {"lead_m": lead, "direction": hand}
    start = surface + clearance
    out = [M.rapid(plane, n=start)]
    if params.get("Q336"):
        out.append(M.action(ACTION, action="orient", angle_deg=q(params, "Q336")))
    out.append(M.action(ACTION, action="spindle", direction=-hand, rpm=rpm))
    reached = surface
    while reached > depth + 1e-12:
        level = max(depth, reached - peck) if peck > 0.0 else depth
        out.append(M.feed(plane, lead, "per_rev", override=False, n=level,
                          modal={"rigid_tap": True, "tap": dict(tap, phase="in", rpm=rpm)}))
        reached = level
        if reached <= depth + 1e-12:
            break
        back = start if chip <= 0.0 else reached + chip
        out.append(M.feed(plane, lead, "per_rev", override=False, n=back,
                          modal={"rigid_tap": True, "tap": dict(tap, phase="out", rpm=rpm * retract_factor)}))
    out.append(M.feed(plane, lead, "per_rev", override=False, n=start,
                      modal={"rigid_tap": True, "tap": dict(tap, phase="out", rpm=rpm * retract_factor)}))
    if retract > start:
        out.append(M.rapid(plane, n=retract))
    return out


def tap_207(h, plane, params) -> list[Step]:
    return _rigid(h, plane, params, 0.0, 0.0, 1.0)


def tap_209(h, plane, params) -> list[Step]:
    peck = h.length_si(abs(q(params, "Q257", 0.0)))
    chip = h.length_si(abs(q(params, "Q256", 0.0)))
    factor = q(params, "Q403", 1.0) or 1.0
    return _rigid(h, plane, params, peck, chip, factor)


def bore_mill_208(h, plane, params) -> list[Step]:
    ex = h.ex
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    pitch = h.length_si(abs(q(params, "Q334")))
    diameter = h.length_si(q(params, "Q335"))
    radius = h.comp_radius()
    path = diameter / 2.0 - radius
    if radius <= 0.0 or path <= 0.0:
        raise Alarm("cycle_tool_radius", "tool not smaller than the bore")
    if pitch <= 0.0:
        raise Alarm("infeed_zero", "Q334")
    direction = _climb(h, int(q(params, "Q351", 1.0)))
    centre = (ex.si(plane.a, ex.s.pos[plane.a]), ex.si(plane.b, ex.s.pos[plane.b]))
    out = [M.rapid(plane, n=surface + clearance), M.feed(plane, f, a=centre[0] + path, b=centre[1])]
    out += M.helix_steps(plane, centre, path, surface + clearance, depth, pitch, f, direction)
    out.append(M.feed(plane, f, a=centre[0], b=centre[1]))
    out.append(M.rapid(plane, n=retract))
    return out


def centre_240(h, plane, params) -> list[Step]:
    surface, clearance, depth, retract = levels(h, params)
    if q(params, "Q343", 0.0):
        raise Alarm("unsupported_function", "Q343 = 1: centring to a diameter needs the tool point angle")
    if params.get("Q201", 0.0) == 0.0:
        return []
    f = feed(h, params, "Q206")
    return ([M.rapid(plane, n=surface + clearance), M.feed(plane, f, n=depth)] + dwell(q(params, "Q211", 0.0)) +
            [M.rapid(plane, n=retract)])


# ====================================================================== pockets

def _climb(h, q351: int) -> int:
    spindle = h.s.spindle.direction
    if spindle == 0:
        raise Alarm("spindle_turning", "climb / up-cut milling with the spindle stopped")
    climb = 1 if spindle < 0 else -1
    return climb if q351 >= 0 else -climb


def _pocket_job(h, plane, params, shape, place, rough: bool, finish: bool) -> list[Step]:
    surface, clearance, depth, retract = levels(h, params)
    if params.get("Q201", 0.0) == 0.0:
        raise Alarm("depth_zero", "Q201 = 0")
    radius = h.comp_radius()
    if radius <= 0.0:
        raise Alarm("cycle_tool_radius", "no tool radius")
    milling = feed(h, params, "Q207")
    plunge_feed = feed(h, params, "Q206")
    finish_feed = feed(h, params, "Q385") if params.get("Q385") else milling
    overlap = q(params, "Q370", 1.0) or 1.0
    plunge = int(q(params, "Q366", 0.0))
    side = h.length_si(abs(q(params, "Q368", 0.0)))
    bottom = h.length_si(abs(q(params, "Q369", 0.0)))
    step = h.length_si(abs(q(params, "Q202", 0.0)))
    if step <= 0.0:
        raise Alarm("q202_missing", "Q202")
    if params.get("Q439", 0.0):
        raise Alarm("unsupported_function", "Q439 feed reference other than the tool centre")
    mode = {0: "vertical", 1: "helix", 2: "oscillate"}.get(plunge)
    if mode is None:
        raise Alarm("plunge_type", f"Q366={plunge}")
    angle = float(h.p["plunge_angle_rad"])
    direction = _climb(h, int(q(params, "Q351", 1.0)))
    job = M.PocketJob(plane=plane, place=place, shape=shape, tool_radius=radius, top=surface, bottom=depth,
                      safety=clearance, retract=retract, max_depth_step=step, stepover=overlap * radius,
                      side_allowance=side if rough else 0.0, bottom_allowance=bottom if rough else 0.0,
                      feed_plane=milling, feed_depth=plunge_feed, feed_finish=finish_feed, direction=direction,
                      plunge=mode, helix_radius=float(h.p["helix_radius_fraction"]) * radius,
                      helix_pitch=2.0 * math.pi * float(h.p["helix_radius_fraction"]) * radius * math.tan(angle),
                      ramp_angle=angle, rough=rough, finish_side=finish, finish_bottom=finish and bottom > 0.0,
                      finish_depth_step=h.length_si(abs(q(params, "Q338", 0.0))))
    job.plunge_point = place.map(shape.start(shape.inner())) if mode != "helix" else place.map((0.0, 0.0))
    if mode == "oscillate" and getattr(shape, "L", 0.0) - getattr(shape, "W", 0.0) <= 0.0:
        raise Alarm("plunge_type", "reciprocating plunge needs a pocket longer than wide")
    return M.pocket_steps(job)


def _operation(params) -> tuple[bool, bool]:
    q215 = int(q(params, "Q215", 0.0))
    if q215 not in (0, 1, 2):
        raise Alarm("cycle_incomplete", "Q215")
    return q215 in (0, 1), q215 in (0, 2)


POSITION_251 = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (-1.0, 1.0), 3: (-1.0, -1.0), 4: (1.0, -1.0)}


def pocket_251(h, plane, params) -> list[Step]:
    ex = h.ex
    rough, finish = _operation(params)
    length, width = h.length_si(abs(q(params, "Q218"))), h.length_si(abs(q(params, "Q219")))
    corner = h.length_si(abs(q(params, "Q220", 0.0))) or h.comp_radius()
    if corner > min(length, width) / 2.0 + 1e-12:
        raise Alarm("q220_large", "Q220")
    angle = math.radians(q(params, "Q224", 0.0))
    position = int(q(params, "Q367", 0.0))
    if position not in POSITION_251:
        raise Alarm("cycle_incomplete", "Q367")
    here = (ex.si(plane.a, ex.s.pos[plane.a]), ex.si(plane.b, ex.s.pos[plane.b]))
    su, sv = POSITION_251[position]
    c, s_ = math.cos(angle), math.sin(angle)
    du, dv = su * length / 2.0, sv * width / 2.0
    centre = (here[0] + c * du - s_ * dv, here[1] + s_ * du + c * dv)
    swap = width > length
    shape = M.RoundedRect(max(length, width), min(length, width), corner)
    place = M.Placement(centre, angle + (math.pi / 2.0 if swap else 0.0))
    return _pocket_job(h, plane, params, shape, place, rough, finish)


def pocket_252(h, plane, params) -> list[Step]:
    ex = h.ex
    rough, finish = _operation(params)
    diameter = h.length_si(abs(q(params, "Q223")))
    if diameter <= 0.0:
        raise Alarm("pocket_small", "Q223")
    centre = (ex.si(plane.a, ex.s.pos[plane.a]), ex.si(plane.b, ex.s.pos[plane.b]))
    shape = M.RoundedRect(diameter, diameter, diameter / 2.0)
    return _pocket_job(h, plane, params, shape, M.Placement(centre, 0.0), rough, finish)


SLOT_POSITION = {0: 0.0, 1: -1.0, 2: -0.5, 3: 0.5, 4: 1.0}


def slot_253(h, plane, params) -> list[Step]:
    ex = h.ex
    rough, finish = _operation(params)
    length, width = h.length_si(abs(q(params, "Q218"))), h.length_si(abs(q(params, "Q219")))
    if width < 2.0 * h.comp_radius() - 1e-12:
        raise Alarm("slot_width", "Q219 smaller than the tool diameter")
    if length < width:
        raise Alarm("q218_q219", "Q218 < Q219")
    angle = math.radians(q(params, "Q374", 0.0))
    position = int(q(params, "Q367", 0.0))
    if position not in SLOT_POSITION:
        raise Alarm("cycle_incomplete", "Q367")
    here = (ex.si(plane.a, ex.s.pos[plane.a]), ex.si(plane.b, ex.s.pos[plane.b]))
    factor = SLOT_POSITION[position]
    offset = -factor * (length / 2.0 if position in (1, 4) else (length - width) / 2.0)
    centre = (here[0] + math.cos(angle) * offset, here[1] + math.sin(angle) * offset)
    shape = M.RoundedRect(length, width, width / 2.0)
    return _pocket_job(h, plane, params, shape, M.Placement(centre, angle), rough, finish)


__all__ = ["define", "cycl_call", "call_defined", "pattern_def", "expand", "deferred", "program_cycle",
           "CALL_ACTIVE", "DEF_ACTIVE", "Action"]
