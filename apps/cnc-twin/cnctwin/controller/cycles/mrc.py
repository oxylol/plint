"""Multiple repetitive cycles of the FANUC T series (system A) and Haas lathes: G71 / G72 stock removal
(types I and II), G73 pattern repeating and G70 finishing (fanuc-30ibplus-om-lathe-2019 4.2.1-4.2.4, 4.2.8).

Target figure: the blocks N(ns) ... N(nf) of the current program (P, Q), read with the handler's word
rules (X/Z absolute, U/W incremental, G00-G03 with I/K/R, ,C/,R, macro assignments executed, F S T M
G96/G97 ignored for G71-G73; other G codes, G41/G42/G40 and calls: PS0325). The first block must give
G00 or G01 (PS0065); for G71 type I it moves only the depth axis (X), type II both (FANUC) or R1 (Haas).
The figure shifted by the finishing allowance (U/2 for a diameter X, W) is the roughing boundary; under
tool nose radius compensation it is first compensated (vector 0 at A, start-up on A -> A').

G71 (cuts along the plane's first axis, depth along the second; G72 the other way round):

* type I (the figure monotone in both axes, else PS0064 / PS0329): levels from A by the depth of cut
  dd (first block U / parameter 5132) toward A'; each level: infeed (G00/G01 as the first block), cut
  to the figure, 45 deg escape by e (R / 5133), rapid back to A's column; then the infeed to A'' and the
  rough pass along the figure (5105#1 RF1 = 0) and the rapid return to A.
* type II (monotone along the cut axis only; pockets allowed): the region between A's level and the
  figure is cut in levels of dd, region by region (the region at the start first, then each pocket
  that splits off, nearest first, 30i order): each pass goes down along the entry wall of the figure,
  along the level, up along the exit wall to the previous level, escapes 45 deg by e and returns at
  rapid; after all regions the tool returns to A, makes the rough pass along the figure and returns.
* G73: d passes (R / 5137) along the figure offset from (di + du/2, dk + dw) down to (du/2, dw).
* G70: the blocks ns..nf are executed as written (their F S T M effective), then a rapid return to A
  (nonlinear positioning) and the program continues after the G70 block.

After G71-G73 the program continues after block nf. Arcs are exact where geometry is computed exactly
(line and arc intersections with the levels); levels are found on chords of at most 1 um sagitta.
"""

from __future__ import annotations

import math

from ..alarms import ControllerAlarm
from ..comp import geometry as G
from ..hooks import Step, register_resumer
from ..ir import Command, Kind
from ..pipeline import PLANES, arc_in_plane
from . import common
from .lathe import INC_OF, set_value, value
from .nose import active as nose_active, compensate, modal_side, radius_tip

SAGITTA_M = 1e-6
ALLOWED_G = ("G00", "G01", "G02", "G03", "G96", "G97", "G98", "G99", "G04")


class Shape:
    def __init__(self):
        self.elements: list[G.Seg] = []
        self.corners: list[dict | None] = []
        self.first_mode = "G01"
        self.first_axes: set[str] = set()
        self.i_ns = 0
        self.i_nf = 0


def _words_of(call):
    return call.extra.get("word_values", {})


def _int_word(ex, call, letter):
    w = common.word(call, letter)
    return None if w is None else int(round(w.value))


def read_shape(ex, call, ns: int, nf: int, a: str, b: str, start: dict, strict: bool) -> Shape:
    """The target figure from N(ns) to N(nf) as plane elements (a, b) starting at the cycle start point."""
    frame = ex.s.frames[-1]
    program = ex.store.programs[frame.program]
    i_ns = program.find_n(ns, frame.pc)
    i_nf = program.find_n(nf, i_ns if i_ns is not None else 0)
    if i_ns is None or i_nf is None or i_nf < i_ns:
        raise ControllerAlarm("mrc_sequence_missing", f"N{ns} / N{nf} in {program.name}")
    shape = Shape()
    shape.i_ns, shape.i_nf = i_ns, i_nf
    h = ex.handler
    pos = (start[a], start[b])
    mode = None
    for index in range(i_ns, i_nf + 1):
        block = program.blocks[index]
        if block.error is not None:
            raise ControllerAlarm(*block.error)
        if block.delete and ex.s.operator.block_delete[block.delete - 1]:
            continue
        if block.stmt is not None:
            if block.stmt[0] != "assign":
                raise ControllerAlarm("mrc_shape_command", f"macro statement in the shape program (line {block.line})")
            h.statement(block.stmt)
            continue
        if not block.words:
            continue
        bw = h.evaluate(block.words)
        for code in bw.g:
            if code in ("G00", "G01", "G02", "G03"):
                mode = code
            elif code in ("G41", "G42", "G40") or (code not in ALLOWED_G and strict):
                raise ControllerAlarm("mrc_shape_command", f"{code} in the shape program (line {block.line})")
        if any(m in (98, 99, 97) for m in bw.m):
            raise ControllerAlarm("mrc_shape_command", f"subprogram call in the shape program (line {block.line})")
        if index == i_ns:
            if not any(code in ("G00", "G01") for code in bw.g):
                raise ControllerAlarm("mrc_first_block", f"N{ns}")
            shape.first_mode = mode
        target = list(pos)
        moved = False
        for k, letter in enumerate((a, b)):
            axis = ex.by_letter[letter]
            word = bw.take(letter)
            inc = bw.take(INC_OF.get(letter, "")) if INC_OF.get(letter) else None
            if word is not None:
                target[k] = h.axis_counts(axis, word) * axis.quantum
                moved = True
                if index == i_ns:
                    shape.first_axes.add(letter)
            if inc is not None:
                target[k] = target[k] + h.axis_counts(axis, inc) * axis.quantum
                moved = True
                if index == i_ns:
                    shape.first_axes.add(letter)
        corner = None
        for letter in (",C", ",R"):
            word = bw.take(letter)
            if word is not None:
                corner = {letter: h.lin(word)}
        if mode is None:
            raise ControllerAlarm("mrc_first_block", f"N{ns}")
        if mode in ("G02", "G03") and moved:
            centre_words = {"X": "I", "Y": "J", "Z": "K"}
            offset = None
            rw = bw.take("R")
            given = {l: bw.take(centre_words[l]) for l in (a, b)}
            if any(v is not None for v in given.values()):
                offset = tuple(h.lin(given[l]) if given[l] is not None else 0.0 for l in (a, b))
            radius = h.lin(rw) if rw is not None else None
            same = G.dist(pos, target) < 1e-12
            geo = arc_in_plane(pos, tuple(target), 1 if mode == "G03" else -1, offset, radius,
                               ex.p["arc_tolerance_m"], same)
            if geo["centre"] is not None:
                shape.elements.append(G.Seg("arc", pos, tuple(target), tuple(geo["centre"]), geo["sweep"]))
                shape.corners.append(corner)
                pos = tuple(target)
                continue
        if moved and G.dist(pos, target) > 1e-12:
            shape.elements.append(G.Seg("line", pos, tuple(target)))
            shape.corners.append(corner)
            pos = tuple(target)
        elif index == i_ns and moved:
            shape.elements.append(G.Seg("line", pos, tuple(target)))
            shape.corners.append(corner)
    if not shape.elements:
        raise ControllerAlarm("mrc_last_block", f"N{ns}-N{nf} has no move")
    _apply_corners(shape)
    return shape


def _apply_corners(shape: Shape) -> None:
    elements, corners = shape.elements, shape.corners
    out = [elements[0]]
    for k in range(1, len(elements)):
        request = corners[k - 1]
        if request and k - 1 > 0:
            a2, insert, b2 = (G.chamfer(out[-1], elements[k], float(request[",C"])) if ",C" in request
                              else G.round_corner(out[-1], elements[k], float(request[",R"])))
            out[-1] = a2
            out.append(insert)
            out.append(b2)
        else:
            out.append(elements[k])
    shape.elements = [e for e in out if e.length() > 1e-12 or e is out[0]]


def _shifted(shape_elements: list[G.Seg], shift: tuple[float, float]) -> list[G.Seg]:
    return [G.Seg(e.kind, G.add(e.p0, shift), G.add(e.p1, shift), G.add(e.c, shift), e.sweep) for e in shape_elements]


def _polyline(elements: list[G.Seg]) -> list[tuple[float, float]]:
    pts = [elements[0].p0]
    for e in elements:
        if e.kind == "arc":
            r = e.radius
            step = 2.0 * math.acos(max(-1.0, 1.0 - SAGITTA_M / r)) if r > SAGITTA_M else math.pi / 2.0
            count = max(1, int(math.ceil(abs(e.sweep) / max(step, 1e-9))))
            a0 = math.atan2(e.p0[1] - e.c[1], e.p0[0] - e.c[0])
            for k in range(1, count):
                ang = a0 + e.sweep * k / count
                pts.append((e.c[0] + r * math.cos(ang), e.c[1] + r * math.sin(ang)))
        pts.append(e.p1)
    return pts


class StockRemoval:
    """G71 (``axis="first"``: cut along the plane's first axis) / G72 (``axis="second"``)."""

    def __init__(self, code: str, haas: bool = False):
        self.code = code
        self.haas = haas

    def expand(self, ex, call) -> list[Step]:
        a, b, _ = PLANES[call.plane]
        words = _words_of(call)
        if "P" not in words or "Q" not in words:
            if self.haas and ("P" in words) != ("Q" in words):
                raise ControllerAlarm("mrc_pq_missing", call.code)
            self._first_block(ex, call)
            return []
        c_axis, k_axis = (a, b) if self.code == "G71" else (b, a)
        start = {a: call.current[a], b: call.current[b]}
        shape = read_shape(ex, call, int(words["P"][0]), int(words["Q"][0]), a, b, start, strict=True)
        if self.haas:
            type2 = (_int_word(ex, call, "R") or 0) == 1
            depth = abs(common.lin(ex, call, "D", value(ex, "g71_depth_m")))
        else:
            need = {k_axis} if self.code == "G71" else {k_axis}
            type2 = shape.first_axes >= {a, b}
            if not type2 and shape.first_axes != need:
                raise ControllerAlarm("mrc_shape_command", f"N{int(words['P'][0])}: the first block must move "
                                                           f"{k_axis} (type I) or both axes (type II)")
            depth = float(value(ex, "g71_depth_m"))
        if depth <= 0.0:
            raise ControllerAlarm("mrc_cut_amount", f"{call.code}: depth of cut 0")
        escape = float(value(ex, "g71_escape_m"))
        elements = self._boundary(ex, call, shape, a, b, start)
        rough = list(elements)
        if self.haas:
            extra = (common.lin(ex, call, "I", 0.0) or 0.0, common.lin(ex, call, "K", 0.0) or 0.0)
            offs = {"X": extra[0], "Z": extra[1]}
            rough = _shifted(elements[1:], (offs.get(a, 0.0), offs.get(b, 0.0)))
            rough = [G.Seg("line", (start[a], start[b]), rough[0].p0)] + rough
        idx = {a: 0, b: 1}
        ci, ki = idx[c_axis], idx[k_axis]
        self._check_monotone(rough[1:], ci, ki, type2)
        p_start = (start[a], start[b])
        programmed = _polyline(shape.elements)
        sk0 = 1.0 if shape.elements[0].p1[ki] > p_start[ki] else -1.0
        self._over_start(programmed, ki, p_start, sk0)
        feed, mode = common.feed_of(ex, call), call.feed_mode
        rapid_infeed = shape.first_mode == "G00"
        steps: list[Step] = []
        p0 = (start[a], start[b])
        if type2:
            steps += self._type2(rough, p0, ci, ki, depth, escape, feed, mode, rapid_infeed, a, b)
            steps.append(Step("rapid", {a: start[a], b: start[b]}))
            if value(ex, "mrc_finish_pass"):
                steps += self._contour(elements, a, b, feed, mode, rapid_infeed)
        else:
            steps += self._type1(rough, p0, ci, ki, depth, escape, feed, mode, rapid_infeed, a, b)
            if value(ex, "mrc_finish_pass"):
                steps += self._contour(elements, a, b, feed, mode, rapid_infeed, from_current=True)
        steps.append(Step("rapid", {a: start[a], b: start[b]}))
        self._continue_after(ex, shape)
        return steps

    # ------------------------------------------------------------------ data
    def _first_block(self, ex, call) -> None:
        """G71 U(dd) R(e) / G72 W(dd) R(e): FANUC parameters 5132, 5133 (radius values)."""
        letter = "W" if self.code == "G72" and not self.haas else "U"
        dd = common.lin(ex, call, letter)
        if dd is None and self.code == "G72":
            dd = common.lin(ex, call, "U")
        e = common.lin(ex, call, "R")
        if dd is not None:
            if dd <= 0.0:
                raise ControllerAlarm("mrc_cut_amount", f"{self.code} depth of cut {dd * 1e3:.4f} mm")
            set_value(ex, "g71_depth_m", abs(dd))
        if e is not None:
            set_value(ex, "g71_escape_m", abs(e))

    def _boundary(self, ex, call, shape: Shape, a: str, b: str, start: dict) -> list[G.Seg]:
        """[A -> A'', shifted (and nose-compensated) figure ...] in plane coordinates."""
        elements = list(shape.elements)
        if nose_active(ex):
            _, tip = radius_tip(ex)
            d = modal_side(ex)
            comp = compensate(ex, elements, d, tip, cancel=False)
            elements = [seg for _, seg in comp]
        u = common.lin(ex, call, "U", 0.0) or 0.0
        w = common.lin(ex, call, "W", 0.0) or 0.0
        shift = {"X": u / 2.0 if ex.by_letter["X"].diameter else u, "Z": w}
        delta = (shift.get(a, 0.0), shift.get(b, 0.0))
        body = _shifted(elements[1:], delta) if len(elements) > 1 else []
        first_end = G.add(elements[0].p1, delta)
        return [G.Seg("line", (start[a], start[b]), first_end)] + body

    def _check_monotone(self, elements: list[G.Seg], ci: int, ki: int, type2: bool) -> None:
        pts = _polyline(elements)
        for axis, key in ((ci, "mrc_not_monotone_1"), (ki, "mrc_not_monotone_2")):
            if axis == ki and type2:
                continue
            signs = {1 if q[axis] > p[axis] + 1e-12 else -1 for p, q in zip(pts, pts[1:]) if abs(q[axis] - p[axis]) > 1e-12}
            if len(signs) > 1:
                raise ControllerAlarm(key, "the finishing shape reverses direction")

    def _continue_after(self, ex, shape: Shape) -> None:
        frame = ex.s.frames[-1]
        frame.pc = shape.i_nf + 1
        ex._jumped = True

    # ------------------------------------------------------------------ passes
    @staticmethod
    def _crossings(pts, ci, ki, level, sk):
        """Crossings of the polyline with the level k = level: (c, segment index, t) in polyline order
        (half-open rule, a vertex on the level counts once, a touch not at all)."""
        out = []
        for index, (p, q) in enumerate(zip(pts, pts[1:])):
            if ((p[ki] - level) * sk > 0.0) != ((q[ki] - level) * sk > 0.0):
                t = (level - p[ki]) / (q[ki] - p[ki])
                out.append((p[ci] + t * (q[ci] - p[ci]), index, t))
        return out

    def _type1(self, rough, p0, ci, ki, depth, escape, feed, mode, rapid_infeed, a, b) -> list[Step]:
        letters = (a, b)
        target = rough[1].p0 if len(rough) > 1 else rough[0].p1
        sk = 1.0 if target[ki] > p0[ki] else -1.0
        pts = _polyline(rough[1:]) if len(rough) > 1 else [target, target]
        sc = 1.0 if pts[-1][ci] > pts[0][ci] else -1.0
        steps: list[Step] = []
        j = 1
        while True:
            level = p0[ki] + sk * j * depth
            if (target[ki] - level) * sk <= 1e-9:
                break
            hits = self._crossings(pts, ci, ki, level, sk)
            end_c = hits[0][0] if hits else pts[-1][ci]
            infeed = {letters[ki]: level}
            steps.append(Step("rapid", infeed) if rapid_infeed else Step("linear", infeed, feed=feed, feed_mode=mode))
            steps.append(Step("linear", {letters[ci]: end_c}, feed=feed, feed_mode=mode))
            steps.append(Step("linear", {letters[ci]: end_c - sc * escape, letters[ki]: level - sk * escape},
                              feed=feed, feed_mode=mode))
            steps.append(Step("rapid", {letters[ci]: p0[ci]}))
            j += 1
        final = {letters[0]: target[0], letters[1]: target[1]}
        steps.append(Step("rapid", final) if rapid_infeed else Step("linear", final, feed=feed, feed_mode=mode))
        return steps

    @staticmethod
    def _over_start(pts, ki, p0, sk) -> None:
        """The figure may not lie beyond the start point on the outer side (PS0322)."""
        for q in pts:
            if (q[ki] - p0[ki]) * sk < -1e-6:
                raise ControllerAlarm("mrc_over_start", "the finishing shape passes outside the cycle start point")

    @staticmethod
    def _wall(pts, ci, ki, crossing, top_level, sk, forward: bool) -> list:
        """Points of the figure from a crossing up its wall to ``top_level`` (or to the wall's highest point
        when it turns down again first), ordered away from the crossing."""
        _, index, t = crossing
        p, q = pts[index], pts[index + 1]
        here = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
        out = [here]
        k = index + 1 if forward else index
        step = 1 if forward else -1
        while 0 <= k < len(pts):
            point = pts[k]
            prev = out[-1]
            if (point[ki] - prev[ki]) * sk > 1e-15:
                break
            if (point[ki] - top_level) * sk <= 0.0:
                frac = (top_level - prev[ki]) / (point[ki] - prev[ki]) if point[ki] != prev[ki] else 0.0
                out.append((prev[0] + frac * (point[0] - prev[0]), prev[1] + frac * (point[1] - prev[1])))
                break
            out.append(point)
            k += step
        return out

    def _type2(self, rough, p0, ci, ki, depth, escape, feed, mode, rapid_infeed, a, b) -> list[Step]:
        letters = (a, b)
        pts = _polyline(rough)
        sk = 1.0 if rough[0].p1[ki] > p0[ki] else -1.0
        extreme = max(q[ki] * sk for q in pts) * sk
        levels = []
        j = 1
        while (extreme - (p0[ki] + sk * j * depth)) * sk > 1e-9:
            levels.append(p0[ki] + sk * j * depth)
            j += 1
        intervals = []
        for level in levels:
            hits = self._crossings(pts, ci, ki, level, sk)
            intervals.append([(hits[k], hits[k + 1]) for k in range(0, len(hits) - 1, 2)])
        steps: list[Step] = []

        def point(pq):
            return {letters[0]: pq[0], letters[1]: pq[1]}

        for region in self._regions(levels, intervals):
            prev_level = p0[ki] if region["parent"] is None else levels[region["parent"]]
            for lv_index, (enter, leave) in region["passes"]:
                level = levels[lv_index]
                entry = list(reversed(self._wall(pts, ci, ki, enter, prev_level, sk, forward=False)))
                exit_ = self._wall(pts, ci, ki, leave, prev_level, sk, forward=True)
                top = entry[0]
                safe = prev_level - sk * escape
                if region["parent"] is None and lv_index == 0:
                    steps.append(Step("rapid", point(top)) if rapid_infeed else
                                 Step("linear", point(top), feed=feed, feed_mode=mode))
                else:
                    steps.append(Step("rapid", {letters[ki]: safe}))
                    steps.append(Step("rapid", {letters[ci]: top[ci]}))
                    steps.append(Step("linear", point(top), feed=feed, feed_mode=mode))
                for q in entry[1:]:
                    steps.append(Step("linear", point(q), feed=feed, feed_mode=mode))
                for q in exit_:
                    steps.append(Step("linear", point(q), feed=feed, feed_mode=mode))
                last = exit_[-1]
                sc = 1.0 if leave[0] >= enter[0] else -1.0
                steps.append(Step("linear", {letters[ci]: last[ci] - sc * escape, letters[ki]: last[ki] - sk * escape},
                                  feed=feed, feed_mode=mode))
                steps.append(Step("rapid", {letters[ki]: safe}))
                prev_level = level
        return steps

    @staticmethod
    def _regions(levels, intervals) -> list[dict]:
        """Regions in cutting order: a region is followed level by level through its first sub-interval;
        intervals that split off become pockets cut after the region, nearest first (FANUC 30i)."""
        regions = []

        def follow(level_index: int, interval: tuple, parent: int | None):
            region = {"parent": parent, "passes": []}
            regions.append(region)
            current = interval
            k = level_index
            while True:
                region["passes"].append((k, current))
                if k + 1 >= len(levels):
                    break
                lo, hi = min(current[0][0], current[1][0]), max(current[0][0], current[1][0])
                children = [iv for iv in intervals[k + 1]
                            if lo - 1e-12 <= min(iv[0][0], iv[1][0]) and max(iv[0][0], iv[1][0]) <= hi + 1e-12]
                if not children:
                    break
                current = children[0]
                for other in children[1:]:
                    pending.append((k + 1, other, k))
                k += 1

        pending: list = []
        for iv in intervals[0] if intervals else []:
            pending.append((0, iv, None))
        while pending:
            level_index, interval, parent = pending.pop(0)
            follow(level_index, interval, parent)
        return regions

    def _contour(self, elements, a, b, feed, mode, rapid_infeed, from_current: bool = False) -> list[Step]:
        """The rough pass along the boundary (A -> A'' in the first block's mode, then the figure)."""
        steps: list[Step] = []
        for k, e in enumerate(elements):
            if k == 0 and from_current:
                continue
            target = {a: e.p1[0], b: e.p1[1]}
            if k == 0:
                steps.append(Step("rapid", target) if rapid_infeed else Step("linear", target, feed=feed, feed_mode=mode))
            elif e.kind == "arc":
                steps.append(Step("arc", target, centre={a: e.c[0], b: e.c[1]}, direction=1 if e.sweep > 0 else -1,
                                  plane=_plane_of(a, b), feed=feed, feed_mode=mode))
            else:
                steps.append(Step("linear", target, feed=feed, feed_mode=mode))
        return steps


def _plane_of(a: str, b: str) -> str:
    for name, (pa, pb, _) in PLANES.items():
        if (pa, pb) == (a, b):
            return name
    return "ZX"


class PatternRepeat:
    """G73 pattern repeating."""

    def __init__(self, haas: bool = False):
        self.haas = haas

    def expand(self, ex, call) -> list[Step]:
        a, b, _ = PLANES[call.plane]
        words = _words_of(call)
        if "P" not in words or "Q" not in words:
            self._first_block(ex, call)
            return []
        start = {a: call.current[a], b: call.current[b]}
        shape = read_shape(ex, call, int(words["P"][0]), int(words["Q"][0]), a, b, start, strict=True)
        if self.haas:
            di, dk = common.lin(ex, call, "I", 0.0) or 0.0, common.lin(ex, call, "K", 0.0) or 0.0
            passes = int(common.integer(ex, call, "D", int(value(ex, "g73_divisions"))) or 1)
        else:
            di, dk = float(value(ex, "g73_escape_x_m")), float(value(ex, "g73_escape_z_m"))
            passes = int(value(ex, "g73_divisions"))
        if passes < 1:
            raise ControllerAlarm("pattern_repeat", f"G73: {passes} divisions")
        elements = list(shape.elements)
        if nose_active(ex):
            _, tip = radius_tip(ex)
            elements = [seg for _, seg in compensate(ex, elements, modal_side(ex), tip, cancel=False)]
        u = common.lin(ex, call, "U", 0.0) or 0.0
        w = common.lin(ex, call, "W", 0.0) or 0.0
        diam = ex.by_letter["X"].diameter
        allow = {"X": u / 2.0 if diam else u, "Z": w}
        escape = {"X": di, "Z": dk}
        feed, mode = common.feed_of(ex, call), call.feed_mode
        rapid_first = shape.first_mode == "G00"
        steps: list[Step] = []
        for n in range(1, passes + 1):
            frac = (passes - n) / (passes - 1) if passes > 1 else 0.0
            shift = tuple(allow.get(l, 0.0) + escape.get(l, 0.0) * frac for l in (a, b))
            body = _shifted(elements[1:], shift)
            first_end = G.add(elements[0].p1, shift)
            target = {a: first_end[0], b: first_end[1]}
            steps.append(Step("rapid", target) if rapid_first else Step("linear", target, feed=feed, feed_mode=mode))
            for e in body:
                t = {a: e.p1[0], b: e.p1[1]}
                if e.kind == "arc":
                    steps.append(Step("arc", t, centre={a: e.c[0], b: e.c[1]}, direction=1 if e.sweep > 0 else -1,
                                      plane=_plane_of(a, b), feed=feed, feed_mode=mode))
                else:
                    steps.append(Step("linear", t, feed=feed, feed_mode=mode))
            steps.append(Step("rapid", {a: start[a], b: start[b]}))
        frame = ex.s.frames[-1]
        frame.pc = shape.i_nf + 1
        ex._jumped = True
        return steps

    def _first_block(self, ex, call) -> None:
        """G73 U(di) W(dk) R(d): FANUC parameters 5135, 5136, 5137 (di, dk radius values with sign)."""
        u = common.lin(ex, call, "U")
        w = common.lin(ex, call, "W")
        rw = common.word(call, "R")
        if u is not None:
            set_value(ex, "g73_escape_x_m", u)
        if w is not None:
            set_value(ex, "g73_escape_z_m", w)
        if rw is not None:
            count = int(round(rw.value))
            if count < 1:
                raise ControllerAlarm("pattern_repeat", f"G73 R{count}")
            set_value(ex, "g73_divisions", count)


class Finishing:
    """G70 P(ns) Q(nf): the figure's own blocks are executed, then a rapid return to the start point."""

    def expand(self, ex, call) -> list[Step]:
        words = _words_of(call)
        if "P" not in words or "Q" not in words:
            raise ControllerAlarm("mrc_pq_missing", "G70 needs P and Q")
        frame = ex.s.frames[-1]
        program = ex.store.programs[frame.program]
        ns, nf = int(words["P"][0]), int(words["Q"][0])
        i_ns = program.find_n(ns, frame.pc)
        i_nf = program.find_n(nf, i_ns if i_ns is not None else 0)
        if i_ns is None or i_nf is None or i_nf < i_ns:
            raise ControllerAlarm("mrc_sequence_missing", f"N{ns} / N{nf} in {program.name}")
        a, b, _ = PLANES[call.plane]
        steps: list[Step] = []
        comp = nose_active(ex)
        if comp:
            steps.append(Step("command", command=Command(Kind.MODE, None, modal={"comp_marker": {"cycle": "start",
                                                                                              "seq": ex.s.block_seq}})))
        ex.s.sys["after_block"] = {"depth": len(ex.s.frames), "pc": i_nf, "program": frame.program,
                                   "provider": "cycles:G70",
                                   "data": {"return_pc": frame.pc + 1, "start": {a: call.current[a], b: call.current[b]},
                                            "comp": comp}}
        frame.pc = i_ns
        ex._jumped = True
        return steps

    def after_block(self, ex, data: dict) -> None:
        frame = ex.s.frames[-1]
        frame.pc = int(data["return_pc"])
        start = {k: float(v) for k, v in data["start"].items()}
        ex.queue_steps([Step("rapid", start)], front=True)

    def resume(self, ex, data, feedback) -> list[Step]:
        return []


register_resumer("cycles:G70", Finishing(), replace=True)

__all__ = ["StockRemoval", "PatternRepeat", "Finishing", "read_shape", "Shape"]
