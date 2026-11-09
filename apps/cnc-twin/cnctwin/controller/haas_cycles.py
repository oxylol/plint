"""Haas cycles implemented by the controller core: G12/G13 circular pocket, G47 engraving, G150 pocket.

Each provider turns one cycle block into program-frame :class:`~cnctwin.controller.hooks.Step`
data (registered with ``dialect="haas"``). Paths (docs/dialects.md "Haas NGC"):

* **G12 (CW) / G13 (CCW)** circular pocket at the current X Y: ``I`` first radius, ``K`` final radius,
  ``Q`` stepover (with K), ``D`` tool offset (the path radius is the pocket radius minus the tool
  radius: G12 implies G42, G13 G41), ``Z`` depth (absolute in G90, increment per loop in G91), ``L``
  loops, ``F`` feed. From the centre a semicircle (radius rho/2) enters tangentially to the first
  circle; each further radius is reached by a semicircle across the stepover, then the full circle;
  the tool feeds back to the centre at the end of every loop.
* **G47** text engraving: ``P0`` the comment text, ``P1`` the serial number in #599 (as many digits as
  '#' in the comment, then #599 + 1), ``P32``-``P126`` one ASCII character; ``X Y`` lower-left start,
  ``I`` angle (deg), ``J`` height (default 1 inch), ``R`` return plane, ``Z`` depth, ``F`` feed, ``E``
  plunge feed. The twin's single-stroke font (cnctwin.controller.haas_font).
* **G150** general pocket: the closed contour of subprogram ``P`` (G01/G02/G03 in X Y, G90/G91, from
  the start point ``X Y``), roughed by zig-zag passes (``I``: stepover along X, cuts parallel to Y;
  ``J``: along Y) clear of the walls by tool radius + ``K``, at depths from ``R`` down by ``Q`` to
  ``Z``; each depth ends with a contour pass at the tool radius (the exact inward offset, in the
  G41/G42 sense of the G150 block, default G41). Zig-zag links stay at depth when the straight link
  keeps the roughing clearance, otherwise the tool retracts to R, rapids and feeds down.
"""

from __future__ import annotations

import math

import numpy as np

from . import contour
from .abi import boundary
from .alarms import ControllerAlarm
from .hooks import CycleCall, Step, register_cycle
from .kernels import raster_intervals, segment_clearance
from .pipeline import PLANES
from .haas_font import ADVANCE, CAP_HEIGHT, strokes

CHORD_SAGITTA_M = 1e-6          # roughing boundary chords (the finishing path keeps true arcs)
LINK_TOLERANCE_M = 1e-9
INCH_M = 0.0254


def _word(call: CycleCall, letter: str):
    from .dialects.fanuc import WordValue

    raw = call.extra.get("word_values", {}).get(letter)
    if raw is None:
        return None
    return WordValue(letter, float(raw[0]), bool(raw[1]), "", bool(raw[2]))


def _feed(ex, call: CycleCall, letter: str = "F") -> float:
    word = _word(call, letter)
    if word is not None:
        return ex.handler.feed_value(word, "per_min")[0]
    if letter == "F" and ex.s.feed_set:
        return ex.s.feed_si
    raise ControllerAlarm("feed_zero", f"{call.code} without {letter}")


def _tool_radius(ex, call: CycleCall) -> float:
    word = _word(call, "D")
    number = ex.handler.offset_word("D", word) if word is not None else ex.s.d_number
    if number <= 0:
        return 0.0
    row = ex.s.tool_offsets.get(str(number))
    if row is None:
        raise ControllerAlarm("illegal_offset_number", f"D{number}")
    return (row[2] + row[3]) * ex.machine.lin_increment_m


def _xy_plane(call: CycleCall) -> None:
    if call.plane != "XY":
        raise ControllerAlarm("illegal_plane", f"{call.code} needs G17")


class CircularPocket:
    def __init__(self, direction: int):
        self.direction = direction

    def expand(self, ex, call: CycleCall) -> list[Step]:
        _xy_plane(call)
        h = ex.handler
        i_word = _word(call, "I")
        if i_word is None:
            raise ControllerAlarm("arc_no_radius", f"{call.code} needs I")
        first = h.lin(i_word)
        k_word, q_word = _word(call, "K"), _word(call, "Q")
        final = h.lin(k_word) if k_word is not None else None
        step = h.lin(q_word) if q_word is not None else None
        if final is not None and (step is None or step <= 0.0):
            raise ControllerAlarm("cycle_q_missing", f"{call.code} with K needs Q > 0")
        radii = [first]
        if final is not None:
            if final < first:
                raise ControllerAlarm("arc_no_radius", f"{call.code}: K smaller than I")
            r = first
            while r + step < final - 1e-12:
                r += step
                radii.append(r)
            if final > radii[-1]:
                radii.append(final)
        tool = _tool_radius(ex, call)
        path = [r - tool for r in radii]
        if path[0] <= 0.0:
            raise ControllerAlarm("tool_too_big", f"{call.code}: tool radius {tool * 1e3:.3f} mm >= I")
        feed = _feed(ex, call)
        cx, cy = call.position["X"], call.position["Y"]
        z_word = _word(call, "Z")
        loops_word = _word(call, "L")
        loops = h.integer(loops_word) if loops_word is not None else 1
        z0 = call.position.get("Z", 0.0)
        steps: list[Step] = []
        for loop in range(max(loops, 1)):
            if z_word is not None:
                z_value = h.lin(z_word)
                z = z_value if call.absolute else z0 + z_value * (loop + 1)
                steps.append(Step("linear", {"Z": z}, feed=feed, feed_mode="per_min"))
            previous = 0.0
            for rho in path:
                mid = cx + (previous + rho) / 2.0
                steps.append(Step("arc", {"X": cx + rho, "Y": cy}, centre={"X": mid, "Y": cy},
                                  direction=self.direction, plane="XY", feed=feed, feed_mode="per_min"))
                steps.append(Step("arc", {"X": cx + rho, "Y": cy}, centre={"X": cx, "Y": cy},
                                  direction=self.direction, plane="XY", feed=feed, feed_mode="per_min"))
                previous = rho
            steps.append(Step("linear", {"X": cx, "Y": cy}, feed=feed, feed_mode="per_min"))
        return steps


class Engraving:
    def expand(self, ex, call: CycleCall) -> list[Step]:
        _xy_plane(call)
        h = ex.handler
        p_word = _word(call, "P")
        mode = h.integer(p_word) if p_word is not None else 0
        comment = call.extra.get("comment", "")
        if mode == 0:
            text = comment
        elif mode == 1:
            digits = comment.count("#")
            if digits == 0:
                raise ControllerAlarm("improper_address", "G47 P1 needs (###...) for the digits")
            serial = ex.read(599)
            number = int(serial or 0)
            text = str(number).zfill(digits)[-digits:]
            ex.write(599, float(number + 1))
        elif 32 <= mode <= 126:
            text = chr(mode)
        else:
            raise ControllerAlarm("invalid_p", f"G47 P{mode}")
        height_word = _word(call, "J")
        height = h.lin(height_word) if height_word is not None else INCH_M
        angle_word = _word(call, "I")
        angle = h.rot(angle_word) if angle_word is not None else 0.0
        feed = _feed(ex, call)
        plunge = _feed(ex, call, "E") if _word(call, "E") is not None else feed
        z_word, r_word = _word(call, "Z"), _word(call, "R")
        if z_word is None:
            raise ControllerAlarm("no_data_after_address", "G47 needs Z")
        current_z = call.current.get("Z", 0.0)
        r_level = (h.lin(r_word) if call.absolute else current_z + h.lin(r_word)) if r_word is not None else current_z
        depth = h.lin(z_word) if call.absolute else r_level + h.lin(z_word)
        x0 = call.current["X"]
        y0 = call.current["Y"]
        x_word, y_word = _word(call, "X"), _word(call, "Y")
        if x_word is not None:
            x0 = call.data["X"] if call.absolute else x0 + call.data["X"]
        if y_word is not None:
            y0 = call.data["Y"] if call.absolute else y0 + call.data["Y"]
        scale = height / CAP_HEIGHT
        c, s = math.cos(angle), math.sin(angle)
        steps = [Step("rapid", {"Z": r_level})]
        last = None
        for k, char in enumerate(text):
            glyph = strokes(char)
            if glyph is None:
                raise ControllerAlarm("g47_character", repr(char))
            for stroke in glyph:
                points = []
                for gx, gy in stroke:
                    lx, ly = (k * ADVANCE + gx) * scale, gy * scale
                    points.append((x0 + c * lx - s * ly, y0 + s * lx + c * ly))
                if last is None or math.hypot(points[0][0] - last[0], points[0][1] - last[1]) > 1e-9:
                    if last is not None:
                        steps.append(Step("rapid", {"Z": r_level}))
                    steps.append(Step("rapid", {"X": points[0][0], "Y": points[0][1]}))
                    steps.append(Step("linear", {"Z": depth}, feed=plunge, feed_mode="per_min"))
                for px, py in points[1:]:
                    steps.append(Step("linear", {"X": px, "Y": py}, feed=feed, feed_mode="per_min"))
                last = points[-1]
        steps.append(Step("rapid", {"Z": r_level}))
        return steps


class GeneralPocket:
    def expand(self, ex, call: CycleCall) -> list[Step]:
        _xy_plane(call)
        h = ex.handler
        p_word = _word(call, "P")
        if p_word is None:
            raise ControllerAlarm("missing_p", "G150 without P")
        program = h.find_program(h.integer(p_word))
        start = (call.current["X"], call.current["Y"])
        if _word(call, "X") is not None:
            start = (call.data["X"] if call.absolute else start[0] + call.data["X"], start[1])
        if _word(call, "Y") is not None:
            start = (start[0], call.data["Y"] if call.absolute else start[1] + call.data["Y"])
        elements = read_contour(ex, program, start, call.absolute)
        tool = _tool_radius(ex, call)
        allowance = h.lin(_word(call, "K")) if _word(call, "K") is not None else 0.0
        i_word, j_word = _word(call, "I"), _word(call, "J")
        if (i_word is None) == (j_word is None):
            raise ControllerAlarm("pocket_definition", "G150 needs exactly one of I or J")
        stepover = h.lin(i_word if i_word is not None else j_word)
        if stepover <= 0.0:
            raise ControllerAlarm("pocket_definition", "G150 stepover must be positive")
        along_x = j_word is not None
        q_word, r_word, z_word = _word(call, "Q"), _word(call, "R"), _word(call, "Z")
        if z_word is None or r_word is None:
            raise ControllerAlarm("no_data_after_address", "G150 needs R and Z")
        current_z = call.current.get("Z", 0.0)
        r_level = h.lin(r_word) if call.absolute else current_z + h.lin(r_word)
        bottom = h.lin(z_word) if call.absolute else r_level + h.lin(z_word)
        per_pass = h.lin(q_word) if q_word is not None else r_level - bottom
        if per_pass <= 0.0:
            raise ControllerAlarm("pocket_definition", "G150 Q must be positive")
        feed = _feed(ex, call)
        side = ex.s.sys.pop("g150_side", "G41")
        area = contour.signed_area(elements)
        ccw = elements if area > 0.0 else contour.reverse(elements)
        points = contour.polyline(ccw, CHORD_SAGITTA_M)
        if not contour.contains(points, start):
            raise ControllerAlarm("pocket_definition", "the G150 start point is not inside the pocket")
        finish = contour.offset_inward(ccw, tool) if tool > 0.0 else list(ccw)
        if not finish or contour.self_intersects(finish, CHORD_SAGITTA_M):
            raise ControllerAlarm("pocket_definition", "the pocket is too narrow for the tool")
        if side == "G42":
            finish = contour.reverse(finish)
        seg = boundary(points if along_x else [(y, x) for x, y in points])
        passes = raster(seg, stepover, tool + allowance)
        if along_x:
            passes = [[(a, y), (b, y)] for y, a, b in passes]
        else:
            passes = [[(y, a), (y, b)] for y, a, b in passes]
        seg_xy = boundary(points)
        levels = []
        z = r_level
        while z - per_pass > bottom + 1e-12:
            z -= per_pass
            levels.append(z)
        levels.append(bottom)
        steps: list[Step] = [Step("rapid", {"Z": r_level}), Step("rapid", {"X": start[0], "Y": start[1]})]
        for level in levels:
            steps.append(Step("linear", {"Z": level}, feed=feed, feed_mode="per_min"))
            here = start
            for segment in order_passes(passes, seg_xy, tool + allowance):
                here = _link(steps, here, segment[0], seg_xy, tool + allowance, r_level, level, feed)
                steps.append(Step("linear", {"X": segment[1][0], "Y": segment[1][1]}, feed=feed, feed_mode="per_min"))
                here = segment[1]
            here = _link(steps, here, finish[0].p0, seg_xy, tool - 1e-7, r_level, level, feed)
            for element in finish:
                if element.kind == "line":
                    steps.append(Step("linear", {"X": element.p1[0], "Y": element.p1[1]}, feed=feed, feed_mode="per_min"))
                else:
                    steps.append(Step("arc", {"X": element.p1[0], "Y": element.p1[1]},
                                      centre={"X": element.centre[0], "Y": element.centre[1]},
                                      direction=1 if element.sweep > 0 else -1, plane="XY", feed=feed,
                                      feed_mode="per_min"))
            steps.append(Step("rapid", {"Z": r_level}))
            steps.append(Step("rapid", {"X": start[0], "Y": start[1]}))
        return steps


def _link(steps: list[Step], here, target, seg, clearance: float, r_level: float, level: float, feed: float):
    if math.hypot(target[0] - here[0], target[1] - here[1]) <= LINK_TOLERANCE_M:
        return target
    if segment_clearance(seg, here[0], here[1], target[0], target[1]) >= clearance - LINK_TOLERANCE_M:
        steps.append(Step("linear", {"X": target[0], "Y": target[1]}, feed=feed, feed_mode="per_min"))
    else:
        steps.append(Step("rapid", {"Z": r_level}))
        steps.append(Step("rapid", {"X": target[0], "Y": target[1]}))
        steps.append(Step("linear", {"Z": level}, feed=feed, feed_mode="per_min"))
    return target


def raster(seg: np.ndarray, stepover: float, clearance: float) -> list[tuple[float, float, float]]:
    """(y, x_lo, x_hi) passes of the zig-zag raster on lines y = const, stepover <= ``stepover``."""
    ys = seg[:, [1, 3]]
    y_lo, y_hi = float(ys.min()) + clearance, float(ys.max()) - clearance
    if y_hi < y_lo:
        return []
    count = max(1, int(math.ceil((y_hi - y_lo) / stepover - 1e-9)))
    lines = [y_lo + (y_hi - y_lo) * k / count for k in range(count + 1)] if y_hi > y_lo else [y_lo]
    capacity = 64
    lo = np.empty(capacity)
    hi = np.empty(capacity)
    out = []
    allowed = float(clearance) - LINK_TOLERANCE_M      # a centre exactly at the clearance touches the wall
    for k, y in enumerate(lines):
        n = raster_intervals(seg, float(y), allowed, lo, hi)
        if n < 0:
            lo = np.empty(capacity * 16)
            hi = np.empty(capacity * 16)
            n = raster_intervals(seg, float(y), allowed, lo, hi)
        pieces = [(float(lo[j]), float(hi[j])) for j in range(n)]
        if k % 2 == 1:
            pieces = [(b, a) for a, b in reversed(pieces)]
        out.extend((float(y), a, b) for a, b in pieces)
    return out


def order_passes(passes, seg: np.ndarray, clearance: float):
    """Zig-zag order: follow the raster, preferring the next pass reachable by a safe link."""
    remaining = list(passes)
    ordered = []
    current = None
    while remaining:
        pick = 0
        if current is not None:
            for index, candidate in enumerate(remaining):
                if segment_clearance(seg, current[0], current[1], candidate[0][0], candidate[0][1]) \
                        >= clearance - LINK_TOLERANCE_M:
                    pick = index
                    break
        segment = remaining.pop(pick)
        ordered.append(segment)
        current = segment[1]
    return ordered


def read_contour(ex, program_index: int, start: tuple[float, float], absolute: bool) -> list[contour.Element]:
    """The closed contour a G150 subprogram describes (see the module docstring)."""
    h = ex.handler
    program = ex.store.programs[program_index]
    position = start
    mode = "G01"
    absolute_mode = absolute
    moves: list[contour.Element] = []
    for block in program.blocks:
        if block.error is not None:
            raise ControllerAlarm(*block.error)
        if block.stmt is not None or not block.words:
            if block.stmt is not None:
                raise ControllerAlarm("pocket_definition", f"macro statement in the pocket subprogram (line {block.line})")
            continue
        bw = h.evaluate(block.words)
        if 99 in bw.m:
            break
        if bw.m:
            raise ControllerAlarm("pocket_definition", f"M code in the pocket subprogram (line {block.line})")
        for code in bw.g:
            if code in ("G00", "G01", "G02", "G03"):
                mode = code
            elif code in ("G90", "G91"):
                absolute_mode = code == "G90"
            else:
                raise ControllerAlarm("pocket_definition", f"{code} in the pocket subprogram (line {block.line})")
        target = list(position)
        moved = False
        for k, letter in enumerate("XY"):
            word = bw.take(letter)
            if word is not None:
                value = h.lin(word)
                target[k] = value if absolute_mode else position[k] + value
                moved = True
        for letter in list(bw.addr):
            if letter not in bw.used and letter not in "IJRF":
                raise ControllerAlarm("pocket_definition", f"{letter} in the pocket subprogram (line {block.line})")
        end = (target[0], target[1])
        if mode in ("G02", "G03"):
            direction = -1 if mode == "G02" else 1
            i_word, j_word, r_word = bw.take("I"), bw.take("J"), bw.take("R")
            from .pipeline import arc_in_plane

            offset = None
            radius = None
            if i_word is not None or j_word is not None:
                offset = (h.lin(i_word) if i_word is not None else 0.0, h.lin(j_word) if j_word is not None else 0.0)
            elif r_word is not None:
                radius = h.lin(r_word)
            same = math.hypot(end[0] - position[0], end[1] - position[1]) < 1e-12
            geometry = arc_in_plane(position, end, direction, offset, radius, ex.p["arc_tolerance_m"], same)
            if geometry["centre"] is not None:
                moves.append(contour.Element("arc", position, end, geometry["centre"], geometry["sweep"]))
                position = end
                continue
        if moved and math.hypot(end[0] - position[0], end[1] - position[1]) > 1e-12:
            moves.append(contour.Element("line", position, end))
            position = end
    if len(moves) < 2:
        raise ControllerAlarm("pocket_definition", "the pocket subprogram has fewer than two moves")
    last = moves[-1].p1
    if math.hypot(last[0] - start[0], last[1] - start[1]) <= 1e-6:
        return moves
    first_end = moves[0].p1
    if math.hypot(last[0] - first_end[0], last[1] - first_end[1]) <= 1e-6:
        return moves[1:]
    raise ControllerAlarm("pocket_definition", "the pocket contour is not closed")


register_cycle("G12", CircularPocket(-1), dialect="haas", replace=True)
register_cycle("G13", CircularPocket(1), dialect="haas", replace=True)
register_cycle("G47", Engraving(), dialect="haas", replace=True)
register_cycle("G150", GeneralPocket(), dialect="haas", replace=True)

__all__ = ["CircularPocket", "Engraving", "GeneralPocket", "read_contour", "raster", "order_passes", "PLANES"]
