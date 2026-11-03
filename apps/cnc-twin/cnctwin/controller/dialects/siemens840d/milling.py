"""Toolpaths of the milling cycles shared by the Siemens and Heidenhain dialects.

Every generator returns :class:`~cnctwin.controller.hooks.Step` lists in the program frame (absolute
SI targets keyed by the program letters of the plane axes ``a``, ``b`` and the tool axis ``n``), so
the core runs them through mirror/scale/rotation, cutter compensation bypass and the final stage
like any cycle (docs/dialects.md "Hooks").

Pockets and slots are milled contour-parallel from the inside out. The family of tool-centre paths of
a rounded rectangle (length L along the local u axis, width W, corner radius rc; circles and
obround slots are special cases) at wall distance e is the rounded rectangle with half sizes
L/2 - e, W/2 - e and corner radius max(rc - e, 0): every path is the exact offset of the wall (lines
shifted, arcs r - e), so the finishing pass at e = r_tool + allowance leaves the allowance exactly.
Roughing passes are evenly spaced between the innermost path (e = W/2, a line or a point) and the
roughing path, with spacing <= the stepover; passes are linked by a straight move along +u at the
pass's start point (u_max, 0). Circumferential slots use the annular-sector family (inner and outer
arcs, end semicircles). Climb milling with M3 is the counter-clockwise loop (tool left of the wall,
G41 sense); conventional the clockwise one; with M4 the senses swap.

Plunging: vertical at the plunge point, helical (radius and pitch per turn given, the last turn at
the depth is closed by a full circle), or oscillating along the u axis at a given ramp angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...hooks import Step
from .kit import Alarm


@dataclass(frozen=True)
class Plane:
    a: str                          # first plane axis letter
    b: str                          # second plane axis letter
    n: str                          # tool (drilling) axis letter
    sign: float = 1.0               # +1: the tool retracts toward +n


@dataclass
class Element:
    kind: str                       # "line" | "arc"
    end: tuple[float, float]        # local (u, v)
    centre: tuple[float, float] | None = None
    direction: int = 0              # arc: +1 counter-clockwise, -1 clockwise


# ====================================================================== step helpers

def rapid(plane: Plane, **coords) -> Step:
    return Step("rapid", {_letter(plane, k): float(v) for k, v in coords.items()})


def feed(plane: Plane, feed_si: float, mode: str = "", exact: bool = False, override: bool = True,
         modal: dict | None = None, **coords) -> Step:
    return Step("linear", {_letter(plane, k): float(v) for k, v in coords.items()}, feed=float(feed_si),
                feed_mode=mode, exact_stop=exact, override=override, modal=dict(modal or {}))


def arc(plane: Plane, end_ab: tuple[float, float], centre_ab: tuple[float, float], direction: int, feed_si: float,
        n_value: float | None = None) -> Step:
    target = {plane.a: float(end_ab[0]), plane.b: float(end_ab[1])}
    if n_value is not None:
        target[plane.n] = float(n_value)
    return Step("arc", target, centre={plane.a: float(centre_ab[0]), plane.b: float(centre_ab[1])},
                direction=int(direction), plane=plane_name(plane), feed=float(feed_si))


def dwell(seconds: float = 0.0, revolutions: float = 0.0) -> list[Step]:
    if seconds <= 0.0 and revolutions <= 0.0:
        return []
    return [Step("dwell", dwell_s=max(seconds, 0.0), dwell_rev=max(revolutions, 0.0))]


def action(provider: str, **data) -> Step:
    """A deterministic dialect action executed in sequence with the motion (spindle, stop ...)."""
    return Step("resume", provider=provider, data=dict(data))


def plane_name(plane: Plane) -> str:
    return {("X", "Y"): "XY", ("Z", "X"): "ZX", ("Y", "Z"): "YZ"}[(plane.a, plane.b)]


def _letter(plane: Plane, key: str) -> str:
    return {"a": plane.a, "b": plane.b, "n": plane.n}[key]


# ====================================================================== drilling

def drill_levels(plane: Plane, retract: float, reference: float, safety: float, depth: float):
    """Approach level (reference + safety toward the retraction side) and checks of the drilling cycles."""
    sign = plane.sign
    if (retract - reference) * sign < 0.0:
        raise Alarm("cycle_reference_plane", "retraction plane on the wrong side of the reference plane")
    approach = reference + sign * abs(safety)
    if (approach - depth) * sign <= 0.0:
        raise Alarm("cycle_reference_plane", "final depth above the reference plane")
    return approach


def peck_depths(start: float, final: float, first: float, degression: float, minimum: float,
                sign: float) -> list[float]:
    """Successive depths from ``start`` to ``final``: the first infeed ``first`` (absolute depth step), then
    each infeed reduced by ``degression`` but not below ``minimum`` (> 0), the last one the remainder."""
    total = (start - final) * sign
    if total <= 0.0:
        return [final]
    step = abs(first) if first else total
    out = []
    reached = 0.0
    while reached + 1e-12 < total:
        step = max(step, minimum) if minimum > 0.0 else step
        reached = min(total, reached + step)
        out.append(start - sign * reached)
        step = step - degression if degression > 0.0 else step
        if step <= 0.0:
            step = minimum if minimum > 0.0 else total - reached
    return out


# ====================================================================== pocket shapes

class RoundedRect:
    """Local frame: centre at (0, 0), u along the length. ``loop(e)`` is the tool-centre path at wall
    distance e (half sizes L/2 - e, W/2 - e, corner radius max(rc - e, 0)), starting and ending at
    (L/2 - e, 0)."""

    def __init__(self, length: float, width: float, corner: float):
        if length < width:
            raise ValueError("RoundedRect expects length >= width (rotate the frame)")
        self.L = float(length)
        self.W = float(width)
        self.rc = min(max(float(corner), 0.0), self.W / 2.0)

    def inner(self) -> float:
        return self.W / 2.0

    def start(self, e: float) -> tuple[float, float]:
        return (self.L / 2.0 - e, 0.0)

    def loop(self, e: float, direction: int) -> list[Element]:
        a, b = self.L / 2.0 - e, self.W / 2.0 - e
        rho = max(self.rc - e, 0.0)
        if a < -1e-12 or b < -1e-12:
            raise Alarm("cycle_tool_radius", "the tool is larger than the pocket")
        a, b = max(a, 0.0), max(b, 0.0)
        if b <= 1e-12:
            if a <= 1e-12:
                return []
            return [Element("line", (-a, 0.0)), Element("line", (a, 0.0))]
        if abs(a - b) <= 1e-12 and abs(rho - b) <= 1e-12:
            return [Element("arc", (a, 0.0), (0.0, 0.0), direction)]
        path = [Element("line", (a, b - rho))]
        corners = [((a - rho, b - rho), (a - rho, b), (-a + rho, b)),
                   ((-a + rho, b - rho), (-a, b - rho), (-a, -b + rho)),
                   ((-a + rho, -b + rho), (-a + rho, -b), (a - rho, -b)),
                   ((a - rho, -b + rho), (a, -b + rho), (a, 0.0))]
        for centre, corner_end, side_end in corners:
            if rho > 1e-12:
                path.append(Element("arc", corner_end, centre, 1))
            path.append(Element("line", side_end))
        path = [el for el in path if not (el.kind == "line" and _same(el.end, _prev_end(path, el, (a, 0.0))))]
        if direction > 0:
            return path
        return reverse_loop(path, (a, 0.0))


def _same(p, q) -> bool:
    return abs(p[0] - q[0]) <= 1e-12 and abs(p[1] - q[1]) <= 1e-12


def _prev_end(path, element, start):
    index = path.index(element)
    return path[index - 1].end if index > 0 else start


def reverse_loop(path: list[Element], start: tuple[float, float]) -> list[Element]:
    points = [start] + [el.end for el in path]
    out = []
    for k in range(len(path) - 1, -1, -1):
        el = path[k]
        target = points[k]
        if el.kind == "arc":
            out.append(Element("arc", target, el.centre, -el.direction))
        else:
            out.append(Element("line", target))
    return out


class ArcSlot:
    """Circumferential slot: centre line on the circle of radius ``radius`` from angle ``a0`` over
    ``sweep`` (rad, > 0 counter-clockwise), width ``width``. Local frame = the circle centre."""

    def __init__(self, radius: float, a0: float, sweep: float, width: float):
        self.R = float(radius)
        self.a0 = float(a0)
        self.sweep = float(sweep)
        self.W = float(width)

    def inner(self) -> float:
        return self.W / 2.0

    def point(self, radius: float, angle: float) -> tuple[float, float]:
        return (radius * math.cos(angle), radius * math.sin(angle))

    def start(self, e: float) -> tuple[float, float]:
        return self.point(self.R + self.W / 2.0 - e, self.a0)

    def loop(self, e: float, direction: int) -> list[Element]:
        w = self.W / 2.0 - e
        if w < -1e-12:
            raise Alarm("cycle_tool_radius", "the tool is wider than the slot")
        a1 = self.a0 + self.sweep
        centre = (0.0, 0.0)
        if w <= 1e-12:
            return [Element("arc", self.point(self.R, a1), centre, 1), Element("arc", self.point(self.R, self.a0), centre, -1)]
        outer, inner = self.R + w, self.R - w
        if inner < 0.0:
            raise Alarm("cycle_slot_violation", "slot width larger than its circle")
        end_cap = self.point(self.R, a1)
        start_cap = self.point(self.R, self.a0)
        path = [Element("arc", self.point(outer, a1), centre, 1),
                Element("arc", self.point(inner, a1), end_cap, 1),
                Element("arc", self.point(inner, self.a0), centre, -1),
                Element("arc", self.point(outer, self.a0), start_cap, 1)]
        if direction > 0:
            return path
        return reverse_loop(path, self.point(outer, self.a0))


def pass_offsets(inner: float, rough: float, stepover: float) -> list[float]:
    """Wall distances of the roughing passes from the innermost (``inner``) out to ``rough``, evenly
    spaced with spacing <= ``stepover``."""
    if stepover <= 0.0:
        raise Alarm("cycle_infeed_width", "stepover must be positive")
    if inner <= rough + 1e-12:
        return [rough]
    count = int(math.ceil((inner - rough) / stepover - 1e-9))
    step = (inner - rough) / count
    return [inner - k * step for k in range(count + 1)]


@dataclass
class Placement:
    """Local pocket frame -> program plane: rotation by ``angle`` (rad) about the plane normal, then the
    translation to ``origin`` (plane coordinates)."""

    origin: tuple[float, float]
    angle: float = 0.0

    def map(self, uv: tuple[float, float]) -> tuple[float, float]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return (self.origin[0] + c * uv[0] - s * uv[1], self.origin[1] + s * uv[0] + c * uv[1])


def loop_steps(plane: Plane, place: Placement, elements: list[Element], feed_si: float) -> list[Step]:
    out = []
    for el in elements:
        end = place.map(el.end)
        if el.kind == "line":
            out.append(feed(plane, feed_si, a=end[0], b=end[1]))
        else:
            out.append(arc(plane, end, place.map(el.centre), el.direction, feed_si))
    return out


def layer_steps(plane: Plane, place: Placement, shape, offsets: list[float], direction: int, feed_si: float,
                at: tuple[float, float]) -> tuple[list[Step], tuple[float, float]]:
    """All passes of one depth layer, starting from the plane position ``at``; returns the steps and the
    end point."""
    out: list[Step] = []
    here = at
    for e in offsets:
        start = place.map(shape.start(e))
        if math.hypot(start[0] - here[0], start[1] - here[1]) > 1e-12:
            out.append(feed(plane, feed_si, a=start[0], b=start[1]))
        out.extend(loop_steps(plane, place, shape.loop(e, direction), feed_si))
        here = start
    return out, here


def helix_steps(plane: Plane, centre: tuple[float, float], radius: float, top: float, bottom: float,
                pitch: float, feed_si: float, direction: int) -> list[Step]:
    """Helical plunge on a circle of ``radius`` about ``centre`` from ``top`` to ``bottom`` (n values),
    ``pitch`` depth per turn; a closing full circle at the bottom. Starts at centre + (radius, 0)."""
    if radius <= 0.0 or pitch <= 0.0:
        raise Alarm("cycle_parameter", "helix radius and pitch must be positive", "_RAD1")
    sign = 1.0 if bottom < top else -1.0
    total = abs(top - bottom)
    turns = max(1, int(math.ceil(total / pitch - 1e-9)))
    start = (centre[0] + radius, centre[1])
    out = []
    for k in range(1, turns + 1):
        level = top - sign * min(total, k * total / turns)
        out.append(arc(plane, start, centre, direction, feed_si, n_value=level))
    out.append(arc(plane, start, centre, direction, feed_si))
    return out


def oscillate_steps(plane: Plane, place: Placement, half: float, top: float, bottom: float, angle: float,
                    feed_si: float) -> list[Step]:
    """Reciprocating ramp along local u between -half and +half (starting at +half), descending at
    ``angle`` from ``top`` to ``bottom``; ends with a pass at the bottom back to the centre."""
    if half <= 0.0 or angle <= 0.0:
        raise Alarm("cycle_parameter", "oscillating insertion needs a straight section and an angle", "_RAD1")
    sign = 1.0 if bottom < top else -1.0
    drop = 2.0 * half * math.tan(angle)
    out = []
    level = top
    side = -1.0
    while abs(level - bottom) > 1e-12:
        level = level - sign * min(drop, abs(level - bottom))
        end = place.map((side * half, 0.0))
        out.append(feed(plane, feed_si, a=end[0], b=end[1], n=level))
        side = -side
    centre = place.map((0.0, 0.0))
    out.append(feed(plane, feed_si, a=centre[0], b=centre[1]))
    return out


def depth_levels(top: float, bottom: float, max_step: float) -> list[float]:
    """Equal depth steps from ``top`` (excluded) to ``bottom`` (included), each <= max_step."""
    total = abs(top - bottom)
    if total <= 1e-12:
        return [bottom]
    if max_step <= 0.0:
        return [bottom]
    count = int(math.ceil(total / max_step - 1e-9))
    sign = 1.0 if bottom < top else -1.0
    return [top - sign * total * k / count for k in range(1, count + 1)]


@dataclass
class PocketJob:
    """Everything one pocket/slot needs (SI, program frame)."""

    plane: Plane
    place: Placement
    shape: object                   # RoundedRect | ArcSlot
    tool_radius: float
    top: float                      # material surface (n value)
    bottom: float                   # final depth (n value)
    safety: float                   # set-up clearance above the surface (unsigned)
    retract: float                  # retraction plane (n value)
    max_depth_step: float           # infeed per layer (0 = the whole depth)
    stepover: float
    side_allowance: float
    bottom_allowance: float
    feed_plane: float
    feed_depth: float
    feed_finish: float
    direction: int                  # +1 counter-clockwise loops, -1 clockwise
    plunge: str = "vertical"        # vertical | vertical_rapid | helix | oscillate
    helix_radius: float = 0.0
    helix_pitch: float = 0.0
    ramp_angle: float = 0.0
    rough: bool = True
    finish_side: bool = False
    finish_bottom: bool = False
    finish_depth_step: float = 0.0
    blank_offset: float | None = None   # wall distance of a premachined pocket (passes inside it skipped)
    blank_depth: float | None = None    # n value down to which the pocket is premachined
    plunge_point: tuple[float, float] | None = None


def pocket_steps(job: PocketJob) -> list[Step]:
    """Roughing layers, then the finishing passes, then the retraction (from the current n position)."""
    plane, place, shape = job.plane, job.place, job.shape
    sign = plane.sign
    r = job.tool_radius
    e_finish = r
    e_rough = r + job.side_allowance
    inner = shape.inner()
    if e_finish > inner + 1e-12:
        raise Alarm("cycle_tool_radius", "the tool is larger than the pocket")
    if job.side_allowance > inner - r + 1e-12:
        raise Alarm("cycle_finish_allowance", "finishing allowance larger than the pocket")
    approach = job.top + sign * abs(job.safety)
    plunge = job.plunge_point if job.plunge_point is not None else place.map(shape.start(inner))
    out = [rapid(plane, n=job.retract), rapid(plane, a=plunge[0], b=plunge[1]), rapid(plane, n=approach)]
    here = plunge
    rough_bottom = job.bottom + sign * job.bottom_allowance
    level = job.top
    if job.rough:
        offsets = pass_offsets(inner, e_rough, job.stepover)
        if job.blank_offset is not None:
            offsets = [e for e in offsets if e <= job.blank_offset + 1e-12] or [e_rough]
        start_level = job.top if job.blank_depth is None else job.blank_depth
        for depth in depth_levels(start_level, rough_bottom, job.max_depth_step):
            out.extend(_plunge(job, here, level if level != job.top else approach, depth))
            here = (job.plunge_point if job.plunge_point is not None else place.map(shape.start(inner))) \
                if job.plunge != "helix" else here
            steps, here = layer_steps(plane, place, shape, offsets, job.direction, job.feed_plane, here)
            out.extend(steps)
            level = depth
            out.append(rapid(plane, n=depth + sign * abs(job.safety)))
            back = job.plunge_point if job.plunge_point is not None else place.map(shape.start(inner))
            out.append(rapid(plane, a=back[0], b=back[1]))
            here = back
            level = depth + sign * abs(job.safety)
    if job.finish_bottom:
        offsets = pass_offsets(inner, e_finish, job.stepover)
        out.append(feed(plane, job.feed_depth, n=job.bottom))
        steps, here = layer_steps(plane, place, shape, offsets, job.direction, job.feed_finish, here)
        out.extend(steps)
        out.append(rapid(plane, n=job.bottom + sign * abs(job.safety)))
    if job.finish_side:
        start = place.map(shape.start(e_finish))
        out.append(rapid(plane, a=here[0], b=here[1]))
        for depth in depth_levels(job.top, job.bottom, job.finish_depth_step):
            out.append(feed(plane, job.feed_depth, a=here[0], b=here[1], n=depth))
            out.append(feed(plane, job.feed_finish, a=start[0], b=start[1]))
            out.extend(loop_steps(plane, place, shape.loop(e_finish, job.direction), job.feed_finish))
            out.append(feed(plane, job.feed_finish, a=here[0], b=here[1]))
        out.append(rapid(plane, n=job.bottom + sign * abs(job.safety)))
    out.append(rapid(plane, n=job.retract))
    return out


def _plunge(job: PocketJob, here: tuple[float, float], start: float, depth: float) -> list[Step]:
    plane = job.plane
    if job.plunge == "helix":
        centre = here
        steps = [feed(plane, job.feed_depth, a=centre[0] + job.helix_radius, b=centre[1])]
        steps += helix_steps(plane, centre, job.helix_radius, start, depth, job.helix_pitch, job.feed_depth,
                             job.direction)
        steps.append(feed(plane, job.feed_depth, a=centre[0], b=centre[1]))
        return steps
    if job.plunge == "oscillate":
        half = getattr(job.shape, "L", 0.0) / 2.0 - getattr(job.shape, "W", 0.0) / 2.0
        return oscillate_steps(plane, job.place, half, start, depth, job.ramp_angle, job.feed_depth)
    if job.plunge == "vertical_rapid":
        return [rapid(plane, n=depth + plane.sign * abs(job.safety)), feed(plane, job.feed_depth, n=depth)]
    return [feed(plane, job.feed_depth, n=depth)]


__all__ = ["Plane", "Element", "RoundedRect", "ArcSlot", "Placement", "PocketJob", "pocket_steps", "pass_offsets",
           "layer_steps", "helix_steps", "oscillate_steps", "depth_levels", "peck_depths", "drill_levels", "rapid",
           "feed", "arc", "dwell", "action", "loop_steps", "reverse_loop"]
