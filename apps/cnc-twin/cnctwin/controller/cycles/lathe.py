"""Lathe cycles of the FANUC T series (G code system A) and Haas NGC lathes: the box cycles G90 / G92 /
G94, the grooving / peck cycles G74 / G75 and the multiple threading cycle G76
(fanuc-30ibplus-om-lathe-2019 4.1, 4.2.5-4.2.7). Plane coordinates: ``a`` is the plane's first axis
(the "length", Z in G18), ``b`` its second (the "end face", X, radius values inside).

* G90 (length) / G94 (face): A -> (b_end + R) rapid, -> end at feed (taper), -> back to A's b (G90) or a
  (G94) at feed, -> A rapid. X(U), Z(W) and R are modal across blocks; a block without a move command in
  the mode repeats the cycle (core). Under tool nose radius compensation the four moves are compensated
  with start-up from A and cancel on the return (4.1.5), the side chosen away from the material.
* G92: as G90 with operation 2 a thread (F = lead along the major axis, Q = start angle, override off),
  thread chamfering over r = parameter 5130 x L at the angle 5131, then rapid retraction.
* G74 / G75: pecks of dk (dk, then return e = parameter 5139 at rapid, then dk + e at feed) to the end,
  relief dd at the bottom, rapid back, step di to the next column; the last column at the end point.
* G76 (two blocks: P m r aa, Q dmin, R d; then X Z R(i) P(k) Q(dd) F(L)): cut n at depth
  ``d_n = dd sqrt(n)`` from the crest (constant chip area), at least dmin deeper than the previous one,
  up to k - d; then the finishing cut at k, m times. One-edge cutting: each cut starts shifted along the
  flank by ``(d_n - k) tan(a / 2)`` in the threading direction (radial infeed for a = 0). Haas G76 one
  block: K (height), D (first depth), A (angle), I (taper), P1-P4 (single/double edge, constant amount /
  constant depth), Q (start angle).
"""

from __future__ import annotations

import math

from ..alarms import ControllerAlarm
from ..hooks import Step
from ..pipeline import PLANES
from . import common
from .nose import active, compensate_polyline, nose_side

INC_OF = {"X": "U", "Z": "W", "Y": "V"}
TOOL_ANGLES = (0, 29, 30, 55, 60, 80)


def value(ex, name: str):
    """A cycle parameter, after program changes (G76 P.., G71 U.. R.. set parameters like FANUC)."""
    changed = ex.s.sys.get("cycle_param_set") or {}
    return changed[name] if name in changed else common.params(ex)[name]


def set_value(ex, name: str, val) -> None:
    ex.s.sys.setdefault("cycle_param_set", {})[name] = val


def plane_of(ex, call) -> tuple[str, str]:
    a, b, _ = PLANES[call.plane]
    for letter in (a, b):
        if letter not in ex.by_letter:
            raise ControllerAlarm("illegal_plane", f"{call.code}: plane {call.plane} needs {letter}")
    return a, b


def end_point(ex, call, a: str, b: str, modal: dict | None) -> dict[str, float]:
    """End point of a box/threading cycle from X(U) Z(W) (absolute, or incremental from the start point),
    modal X(U), Z(W) kept in ``modal`` (the core's box-cycle data) when given."""
    words = call.extra.get("word_values", {})
    out = {}
    for letter in (a, b):
        inc = INC_OF.get(letter)
        mode_key = f"{letter}_mode"
        if modal is not None:
            if letter in words:
                modal[mode_key] = "abs"
            elif inc in words:
                modal[mode_key] = "inc"
        mode = (modal or {}).get(mode_key) or ("abs" if letter in words else "inc" if inc in words else None)
        start = call.current[letter]
        if mode == "abs" and letter in call.data:
            out[letter] = float(call.data[letter])
        elif mode == "inc" and inc in call.data:
            delta = float(call.data[inc])
            if ex.by_letter[letter].diameter:
                delta /= 2.0
            out[letter] = start + delta
        else:
            out[letter] = start
    for letter in words:
        if letter in ex.by_letter and letter not in (a, b) and letter not in ("C",):
            raise ControllerAlarm("box_axis", f"{call.code}: {letter} is not in the cycle plane")
    return out


def lead_of(ex, call) -> float:
    w = common.word(call, "F")
    if w is not None:
        lead = ex.handler.feed_value(w, "per_rev")[0]
    elif ex.s.feed_set and ex.handler.feed_mode() == "per_rev":
        lead = ex.s.feed_si
    else:
        lead = 0.0
    if lead <= 0.0:
        raise ControllerAlarm("thread_lead", f"{call.code} without a lead F")
    return lead


def thread_steps(ex, a: str, b: str, start: dict, end: dict, lead: float, angle0: float, chamfer_m: float,
                 chamfer_angle: float, out_sign: float) -> list[Step]:
    """Thread move(s) from start to end (lead along the major axis; feed per revolution along the path),
    with the chamfer over the last ``chamfer_m`` of the length axis pulled out at ``chamfer_angle``."""
    da, db = end[a] - start[a], end[b] - start[b]
    major = abs(da) if abs(da) >= abs(db) else abs(db)
    steps = []

    def move(p0, p1, chamfer: bool) -> Step:
        length = math.hypot(p1[a] - p0[a], p1[b] - p0[b])
        along = abs(p1[a] - p0[a]) if abs(da) >= abs(db) else abs(p1[b] - p0[b])
        path_lead = lead * length / along if along > 0.0 else lead
        modal = {"thread": {"lead_m": path_lead, "lead_major_m": lead, "major_axis": a if abs(da) >= abs(db) else b,
                            "start_angle_rad": angle0, "chamfer": chamfer}}
        return Step("linear", {a: p1[a], b: p1[b]}, feed=path_lead, feed_mode="per_rev", override=False,
                    modal=modal)

    if chamfer_m > 0.0 and abs(da) > chamfer_m and major > 0.0:
        frac = 1.0 - chamfer_m / abs(da)
        mid = {a: start[a] + da * frac, b: start[b] + db * frac}
        steps.append(move(start, mid, False))
        pulled = {a: end[a], b: end[b] + out_sign * chamfer_m * math.tan(chamfer_angle)}
        steps.append(move(mid, pulled, True))
    else:
        steps.append(move(start, end, False))
    return steps


class BoxCycle:
    """G90 / G92 / G94 (``haas``: the taper is I for G90/G92, K for G94)."""

    def __init__(self, code: str, haas: bool = False):
        self.code = code
        self.haas = haas

    def expand(self, ex, call) -> list[Step]:
        a, b = plane_of(ex, call)
        box = (ex.s.sys.get("box_cycle") or {}).get("data")
        end = end_point(ex, call, a, b, box)
        start = {a: call.current[a], b: call.current[b]}
        taper_letter = ("K" if self.code == "G94" else "I") if self.haas else "R"
        taper = float(call.data.get(taper_letter, 0.0)) if taper_letter in call.data else 0.0
        if self.code == "G94":
            p1 = {a: end[a] + taper, b: start[b]}
            p3 = {a: start[a], b: end[b]}
        else:
            p1 = {a: start[a], b: end[b] + taper}
            p3 = {a: end[a], b: start[b]}
        p2 = dict(end)
        if self.code == "G92":
            return self._thread(ex, call, a, b, start, p1, p2)
        feed, mode = common.feed_of(ex, call), call.feed_mode
        points = [start, p1, p2, p3, start]
        kinds = ["rapid", "linear", "linear", "rapid"]
        if active(ex):
            moves = compensate_polyline(ex, a, b, points, nose_side(ex, a, b, points), cancel=True)
        else:
            moves = list(enumerate(points[1:]))
        steps = []
        for index, q in moves:
            if kinds[index] == "rapid":
                steps.append(Step("rapid", {a: q[a], b: q[b]}))
            else:
                steps.append(Step("linear", {a: q[a], b: q[b]}, feed=feed, feed_mode=mode))
        return steps

    def _thread(self, ex, call, a, b, start, p1, p2) -> list[Step]:
        lead = lead_of(ex, call)
        q = common.word(call, "Q")
        angle0 = math.radians(q.value if (q is not None and (q.point or q.from_variable)) else (q.value * 0.001 if q else 0.0))
        chamfer = float(value(ex, "thread_chamfer_leads")) * lead
        out = 1.0 if start[b] >= p2[b] else -1.0
        steps = [Step("rapid", {b: p1[b]})]
        steps += thread_steps(ex, a, b, p1, p2, lead, angle0, chamfer, float(value(ex, "thread_chamfer_angle_rad")), out)
        steps += [Step("rapid", {b: start[b]}), Step("rapid", {a: start[a]})]
        return steps


class GrooveCycle:
    """G74 (pecks along the plane's first axis, columns along the second) / G75 (the other way round)."""

    def __init__(self, code: str, haas: bool = False):
        self.code = code
        self.haas = haas

    def expand(self, ex, call) -> list[Step]:
        a, b = plane_of(ex, call)
        words = call.extra.get("word_values", {})
        moving = any(l in words for l in (a, b, INC_OF.get(a), INC_OF.get(b)))
        if not moving and not self.haas:
            r = common.lin(ex, call, "R")
            if r is None:
                raise ControllerAlarm("groove_relief", f"{call.code} without R (return amount)")
            set_value(ex, "g74_return_m", abs(r))
            return []
        peck_axis, step_axis = (a, b) if self.code == "G74" else (b, a)
        end = end_point(ex, call, a, b, None)
        start = {a: call.current[a], b: call.current[b]}
        inc = ex.machine.lin_increment_m
        if self.haas:
            peck = abs(common.lin(ex, call, "K" if self.code == "G74" else "I", 0.0) or 0.0)
            shift = abs(common.lin(ex, call, "I" if self.code == "G74" else "K", 0.0) or 0.0)
            relief = common.lin(ex, call, "D", 0.0) or 0.0
        else:
            pk = common.word(call, "Q" if self.code == "G74" else "P")
            st = common.word(call, "P" if self.code == "G74" else "Q")
            peck = abs(pk.value) * inc if pk is not None else 0.0
            shift = abs(st.value) * inc if st is not None else 0.0
            relief = common.lin(ex, call, "R", 0.0) if ("R" in words) else 0.0
        feed, mode = common.feed_of(ex, call), call.feed_mode
        e = float(value(ex, "g74_return_m"))
        total = end[peck_axis] - start[peck_axis]
        width = end[step_axis] - start[step_axis]
        if total != 0.0 and peck <= 0.0:
            raise ControllerAlarm("groove_amount", f"{call.code}: depth of cut 0")
        if width != 0.0 and shift <= 0.0:
            raise ControllerAlarm("groove_amount", f"{call.code}: travel between cuts 0")
        s_peck = 1.0 if total >= 0.0 else -1.0
        s_step = 1.0 if width >= 0.0 else -1.0
        columns = [start[step_axis]]
        if width != 0.0:
            count = int(math.ceil(abs(width) / shift - 1e-9))
            columns = [start[step_axis] + s_step * min(k * shift, abs(width)) for k in range(count + 1)]
        relief_dir = -s_step if width != 0.0 else (1.0 if relief >= 0.0 else -1.0)
        steps: list[Step] = []
        for k, column in enumerate(columns):
            if k > 0:
                steps.append(Step("rapid", {step_axis: column}))
            depth, last = 0.0, start[peck_axis]
            while depth < abs(total) - 1e-12:
                depth = min(depth + peck, abs(total))
                target = start[peck_axis] + s_peck * depth
                if last != start[peck_axis]:
                    steps.append(Step("rapid", {peck_axis: last - s_peck * min(e, abs(last - start[peck_axis]))}))
                steps.append(Step("linear", {peck_axis: target}, feed=feed, feed_mode=mode))
                last = target
            if abs(relief) > 0.0:
                steps.append(Step("linear", {step_axis: column + relief_dir * abs(relief)}, feed=feed, feed_mode=mode))
            steps.append(Step("rapid", {peck_axis: start[peck_axis]}))
        steps.append(Step("rapid", {step_axis: start[step_axis]}))
        return steps


class ThreadCycle:
    """G76 multiple threading cycle (FANUC two-block form, Haas one-block form)."""

    def __init__(self, haas: bool = False):
        self.haas = haas

    def expand(self, ex, call) -> list[Step]:
        a, b = plane_of(ex, call)
        words = call.extra.get("word_values", {})
        moving = any(l in words for l in (a, b, INC_OF.get(a), INC_OF.get(b)))
        if not moving:
            if self.haas:
                raise ControllerAlarm("thread_command", "G76 without the thread end point")
            self._first_block(ex, call)
            return []
        inc = ex.machine.lin_increment_m
        lead = lead_of(ex, call)
        if self.haas:
            height = abs(common.lin(ex, call, "K", 0.0) or 0.0)
            first = abs(common.lin(ex, call, "D", 0.0) or 0.0)
            taper = common.lin(ex, call, "I", 0.0) or 0.0
            aw = common.word(call, "A")
            angle_deg = aw.value if aw is not None else math.degrees(float(value(ex, "g76_tool_angle_rad")))
            method = int(common.integer(ex, call, "P", 1) or 1)
            if method not in (1, 2, 3, 4) or not 0.0 <= angle_deg <= 120.0:
                raise ControllerAlarm("thread_nose_angle", f"G76 P{method} A{angle_deg:g}")
            qw = common.word(call, "Q")
            angle0 = math.radians(qw.value if qw is not None and (qw.point or qw.from_variable) else
                                  (qw.value * 0.001 if qw is not None else 0.0))
        else:
            pk, qd = common.word(call, "P"), common.word(call, "Q")
            height = abs(pk.value) * inc if pk is not None else 0.0
            first = abs(qd.value) * inc if qd is not None else 0.0
            taper = common.lin(ex, call, "R", 0.0) or 0.0
            angle_deg = math.degrees(float(value(ex, "g76_tool_angle_rad")))
            method = 1
            angle0 = 0.0
        finish = float(value(ex, "g76_finish_m"))
        minimum = float(value(ex, "g76_min_depth_m"))
        repeats = max(1, int(value(ex, "g76_finish_passes")))
        if height <= 0.0 or first <= 0.0 or finish >= height:
            raise ControllerAlarm("thread_cut_amount", f"G76: height {height * 1e3:.4f} mm, first cut "
                                                       f"{first * 1e3:.4f} mm, finish {finish * 1e3:.4f} mm")
        end = end_point(ex, call, a, b, None)
        start = {a: call.current[a], b: call.current[b]}
        out = 1.0 if start[b] >= end[b] else -1.0
        s_a = 1.0 if end[a] >= start[a] else -1.0
        line0 = {a: start[a], b: end[b] + taper}
        depths = self._depths(height, first, finish, minimum, constant_depth=method in (3, 4))
        depths += [height] * repeats
        flank = method in (1, 3)
        half = math.radians(angle_deg) / 2.0
        chamfer = float(value(ex, "thread_chamfer_leads")) * lead
        chamfer_angle = float(value(ex, "thread_chamfer_angle_rad"))
        steps: list[Step] = []
        for depth in depths:
            shift = s_a * (depth - height) * math.tan(half) if flank else 0.0
            p0 = {a: line0[a] + shift, b: line0[b] + out * (height - depth)}
            p1 = {a: end[a] + shift, b: end[b] + out * (height - depth)}
            steps.append(Step("rapid", {a: p0[a], b: p0[b]}))
            steps += thread_steps(ex, a, b, p0, p1, lead, angle0, chamfer, chamfer_angle, out)
            steps.append(Step("rapid", {b: start[b]}))
            steps.append(Step("rapid", {a: start[a]}))
        return steps

    @staticmethod
    def _depths(height: float, first: float, finish: float, minimum: float, constant_depth: bool) -> list[float]:
        """Roughing depths from the crest: dd sqrt(n) (or n dd), each at least ``minimum`` deeper, the last
        at height - finish."""
        target = height - finish
        depths: list[float] = []
        n, last = 1, 0.0
        while True:
            depth = first * (n if constant_depth else math.sqrt(n))
            if depth - last < minimum:
                depth = last + minimum
            if depth >= target - 1e-12:
                depths.append(target)
                return depths
            depths.append(depth)
            last = depth
            n += 1
            if n > 100000:
                raise ControllerAlarm("thread_cut_amount", "G76: more than 100000 cuts")

    def _first_block(self, ex, call) -> None:
        """G76 P(m)(r)(aa) Q(dmin) R(d): FANUC sets parameters 5142, 5130, 5143, 5140, 5141."""
        inc = ex.machine.lin_increment_m
        pw = common.word(call, "P")
        if pw is not None:
            digits = int(round(abs(pw.value)))
            m, r, aa = digits // 10000, (digits // 100) % 100, digits % 100
            if aa not in TOOL_ANGLES:
                raise ControllerAlarm("thread_nose_angle", f"G76 P{digits:06d}: tool angle {aa}")
            set_value(ex, "g76_finish_passes", max(1, m))
            set_value(ex, "thread_chamfer_leads", r / 10.0)
            set_value(ex, "g76_tool_angle_rad", math.radians(aa))
        qw = common.word(call, "Q")
        if qw is not None:
            set_value(ex, "g76_min_depth_m", abs(qw.value) * inc if not qw.point else abs(ex.handler.lin(qw)))
        rw = common.word(call, "R")
        if rw is not None:
            set_value(ex, "g76_finish_m", abs(ex.handler.lin(rw)))


__all__ = ["BoxCycle", "GrooveCycle", "ThreadCycle", "thread_steps", "end_point", "value", "set_value", "lead_of"]
