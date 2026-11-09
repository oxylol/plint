"""Dry run of the control against an ideal machine (``twin check``, the program view, tests).

The ideal machine executes each command at constant speed with no acceleration and no following
error: rapids move every axis independently at its rapid rate (FANUC non-linear positioning;
time = the slowest axis), cutting moves at the programmed feed, dwells for their time, tool changes
take the machine's tool change time, everything else is instantaneous. A command starts at
max(previous end, ready_tick).

Feeds (the control's feed semantics, so the time is the program's own lower bound):

* per minute: the path length over F. The path is the combined machine axes (rotary axes count their
  degrees as millimetres, FANUC's rule for a combined feed), except under tool centre point control
  with tip interpolation (Haas G234, FANUC G43.4/G43.5: ``modal tip_interpolation`` "tip"), where F is
  the tip's speed relative to the part: the length is the tip path in the work frame (from
  ``modal tip_start_m`` to ``tip_target_m``, the arc's length for an arc); a block that only turns the
  tool about its tip moves the rotary axes at F read as degrees per minute;
* per revolution: the length over F x n. Under constant surface speed (G96: the spindle payload's
  ``css_m_s``) n follows the radius of the program X (``modal program_offset_m``) along the move,
  n = min(v / (2 pi r), n_max), integrated exactly over the move (lines in closed form, arcs over
  1 degree chords); otherwise n is the commanded spindle speed;
* inverse time (G93): 1 / F.

Buffer-stopping commands are answered at their end: live reads with the commanded position, probes
and tool setters by the ``probe``/``setter`` callbacks (no trigger when absent), M00/M01/#3006/single
block by an immediate cycle start, path waits by the controller's coordination. A probe that triggers
ends its skip move at the latch point (the move's time is cut there). :class:`BoxProbe` is an ideal
spindle probe against axis-aligned boxes (the stock as set up).

The ideal machine is a check harness, not the physics: motion (cnctwin.motion) replaces it in a
run. Its cycle time is a lower bound of the real one (no acceleration limits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .controller import Controller
from .ir import Command, Feedback, FeedMode, Kind, ProbeResult, Space

BUFFER_STOPPING = (Kind.SYNC, Kind.PROBE, Kind.TOOL_SETTER, Kind.STOP, Kind.MESSAGE)
MOTION = (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE, Kind.TOOL_SETTER)
ARC_CHORD_RAD = math.pi / 180.0          # CSS timing of arcs: chords of at most 1 degree


@dataclass
class DryRunResult:
    commands: list[list[Command]]
    start_ticks: list[list[int]]
    end_ticks: list[list[int]]
    events: list[list[dict]]
    end_tick: int
    alarms: list[dict | None]
    controller: Controller = None
    stopped: str = ""


@dataclass
class _Path:
    queue: list = field(default_factory=list)
    free_tick: int = 0
    omega: float = 0.0
    position: dict = field(default_factory=dict)
    done: bool = False
    css: dict | None = None             # {"v": m/s, "max": rad/s, "axis": letter} while G96 is active
    tip: tuple | None = None            # tip of the last TCP command (work frame)


def dry_run(controller: Controller, *, max_tick: int = 3_600_000_000, parts: int = 1, probe=None, setter=None,
            operator_delay_ticks: int = 0) -> DryRunResult:
    """Run every path to its program end (``parts`` times) or alarm."""
    n = len(controller.executors)
    paths = [_Path() for _ in range(n)]
    for p, ex in enumerate(controller.executors):
        paths[p].position = {a.name: ex.s.mach.get(a.letter, 0) * a.quantum for a in ex.axes}
        paths[p].queue = list(ex.s.buffer)            # a restored state: motion still has these to execute
    commands: list[list[Command]] = [[] for _ in range(n)]
    starts: list[list[int]] = [[] for _ in range(n)]
    ends: list[list[int]] = [[] for _ in range(n)]
    runs = [0] * n
    now = 0
    stopped = ""
    guard = 0
    while now <= max_tick:
        guard += 1
        if guard > 50_000_000:
            stopped = "iteration limit"
            break
        for p in range(n):
            if not paths[p].done:
                paths[p].queue.extend(controller.produce(p, now))
        candidates = []
        for p in range(n):
            state = paths[p]
            if state.queue:
                cmd = state.queue[0]
                start = max(state.free_tick, cmd.ready_tick)
                duration = _duration(controller, p, state, cmd)
                if duration is None:
                    stopped = f"path {p + 1}: feed per revolution with the spindle stopped (line {cmd.ref.line})"
                    break
                candidates.append((start + duration, p, start))
        if stopped:
            break
        if not candidates:
            active = [p for p in range(n) if not paths[p].done]
            if not active:
                break
            states = [controller.state.paths[p] for p in active]
            if all(s.alarm is not None for s in states):
                break
            if all(s.waiting is not None or s.alarm is not None for s in states):
                ended = [p for p in active if controller.state.paths[p].waiting is not None
                         and controller.state.paths[p].waiting["kind"] == "end"]
                if ended and len(ended) + sum(1 for s in states if s.alarm is not None) == len(active):
                    finished = True
                    for p in ended:
                        runs[p] += 1
                        if runs[p] < parts:
                            controller.cycle_start(p, now + operator_delay_ticks)
                            finished = False
                        else:
                            paths[p].done = True
                    if finished:
                        break
                    continue
                stopped = "deadlock"
                break
            interp = min(controller.state.paths[p].interp_tick for p in active)
            now = max(now + 1, interp)
            continue
        end, p, start = min(candidates)
        state = paths[p]
        cmd = state.queue.pop(0)
        latched = None
        if cmd.kind == Kind.PROBE and probe is not None:
            latched = probe(p, cmd)
            if latched is not None and latched.triggered:
                end = start + int(math.ceil((end - start) * _fraction(state.position, cmd.target,
                                                                      latched.machine_pos) - 1e-9))
                # additive payload key for whoever reads the dry run: where the skip move ended
                cmd.probe = dict(cmd.probe or {}, latched_m=dict(latched.machine_pos))
        now = end
        state.free_tick = end
        commands[p].append(cmd)
        starts[p].append(start)
        ends[p].append(end)
        _apply(state, cmd)
        controller.retire(p, 1, end)
        if cmd.kind in BUFFER_STOPPING or cmd.kind == Kind.ALARM:
            _answer(controller, p, state, cmd, end, latched, setter, operator_delay_ticks)
    else:
        stopped = "time limit"
    return DryRunResult(commands, starts, ends, [list(s.events) for s in controller.state.paths],
                        max((e[-1] for e in ends if e), default=0),
                        [s.alarm for s in controller.state.paths], controller, stopped)


def _fraction(start: dict, end: dict, latched: dict) -> float:
    """Where along the move start -> end the latched point lies (0..1)."""
    num = 0.0
    den = 0.0
    for name, value in end.items():
        delta = value - start.get(name, value)
        num += delta * (float(latched.get(name, value)) - start.get(name, value))
        den += delta * delta
    return min(max(num / den, 0.0), 1.0) if den > 0.0 else 1.0


def _apply(state: _Path, cmd: Command) -> None:
    if cmd.kind == Kind.SPINDLE and cmd.spindle is not None:
        state.omega = float(cmd.spindle.get("omega_rad_s", 0.0))
        css = cmd.spindle.get("css_m_s")
        state.css = ({"v": float(css), "max": float(cmd.spindle.get("max_rad_s") or 0.0),
                      "axis": str(cmd.spindle.get("css_axis") or "X")}
                     if css is not None and state.omega != 0.0 else None)
    if cmd.target and cmd.kind in MOTION:
        state.position.update(cmd.target)
        state.tip = tuple(cmd.tip_target_m) if (cmd.space == Space.TIP and cmd.tip_target_m is not None) else None


def _answer(controller: Controller, p: int, state: _Path, cmd: Command, tick: int, latched, setter,
            delay: int) -> None:
    waiting = controller.state.paths[p].waiting
    if waiting is None:
        return
    time_s = tick * controller.setup.tick_s
    kind = waiting["kind"]
    if kind in ("stop", "end"):
        if kind == "stop":
            controller.cycle_start(p, tick + delay)
        return
    fb = Feedback(time_s, dict(state.position))
    if kind == "probe":
        fb.probe = latched if latched is not None else ProbeResult(False, dict(state.position), time_s)
        if fb.probe.triggered:
            fb.probe = ProbeResult(True, dict(fb.probe.machine_pos), time_s)
            fb.machine_pos = dict(fb.probe.machine_pos)
            state.position.update(fb.probe.machine_pos)
    elif kind == "setter":
        result = setter(p, cmd) if setter is not None else None
        if isinstance(result, ProbeResult):
            fb.probe = result
            if result.triggered:
                fb.machine_pos = dict(result.machine_pos)
                state.position.update(result.machine_pos)
        elif result is not None:
            fb.setter = result
    controller.feedback(p, fb, tick)


def _duration(controller: Controller, p: int, state: _Path, cmd: Command) -> int | None:
    tick_s = controller.setup.tick_s
    ex = controller.executors[p]
    seconds = 0.0
    if cmd.kind in MOTION and cmd.target:
        if cmd.kind == Kind.RAPID:
            for axis in ex.axes:
                delta = abs(cmd.target.get(axis.name, 0.0) - state.position.get(axis.name, 0.0))
                if delta > 0.0:
                    rate = axis.rapid if axis.rapid > 0.0 else 1.0
                    seconds = max(seconds, delta / rate)
        else:
            feed = cmd.feed
            if feed is not None and feed.mode == FeedMode.INVERSE_TIME:
                seconds = 1.0 / feed.value if feed.value > 0.0 else 0.0
            elif feed is not None and feed.mode == FeedMode.PER_REV:
                if state.css is not None:
                    seconds = _css_seconds(ex, state, cmd, feed.value)
                    if seconds is None:
                        return None
                else:
                    length = _path_length(ex, state, cmd)
                    speed = feed.value * abs(state.omega) / (2.0 * math.pi)
                    if speed <= 0.0 and length > 0.0:
                        return None
                    seconds = length / speed if speed > 0.0 else 0.0
            elif feed is not None and feed.value > 0.0:
                seconds = _path_length(ex, state, cmd) / feed.value
    elif cmd.kind == Kind.DWELL:
        seconds = cmd.dwell_s
        if cmd.dwell_rev and state.omega:
            seconds += cmd.dwell_rev * 2.0 * math.pi / abs(state.omega)
    elif cmd.kind == Kind.TOOL_CHANGE:
        seconds = controller.machine.tool_change_s
    return int(math.ceil(seconds / tick_s - 1e-9))


def _tip_feed(cmd: Command) -> bool:
    return cmd.space == Space.TIP and cmd.tip_target_m is not None and \
        (cmd.modal or {}).get("tip_interpolation") == "tip"


def _path_length(ex, state: _Path, cmd: Command) -> float:
    if _tip_feed(cmd):
        length = _tip_length(state, cmd)
        if length is not None and length > 0.0:
            return length
        return _axes_length(ex, state, cmd, rotary_only=True)     # the tool turns about its tip: F in deg/min
    return _axes_length(ex, state, cmd)


def _tip_length(state: _Path, cmd: Command) -> float | None:
    start = (cmd.modal or {}).get("tip_start_m") or state.tip
    if start is None:
        return None
    end = cmd.tip_target_m
    if cmd.kind == Kind.ARC and cmd.arc is not None:
        normal = cmd.arc.normal
        along = sum(normal[k] * (end[k] - start[k]) for k in range(3))
        radius = 0.5 * (cmd.arc.radius_start_m + cmd.arc.radius_end_m)
        return math.hypot(radius * abs(cmd.arc.sweep_rad), along)
    return math.sqrt(sum((end[k] - start[k]) ** 2 for k in range(3)))


def _axes_length(ex, state: _Path, cmd: Command, rotary_only: bool = False) -> float:
    if cmd.kind == Kind.ARC and cmd.arc is not None and not rotary_only:
        normal = cmd.arc.normal
        along = 0.0
        for axis in ex.axes:
            if axis.letter in "XYZ":
                k = "XYZ".index(axis.letter)
                along += normal[k] * (cmd.target.get(axis.name, 0.0) - state.position.get(axis.name, 0.0))
        radius = 0.5 * (cmd.arc.radius_start_m + cmd.arc.radius_end_m)
        return math.hypot(radius * abs(cmd.arc.sweep_rad), along)
    total = 0.0
    for axis in ex.axes:
        if rotary_only and axis.kind != "rotary":
            continue
        delta = cmd.target.get(axis.name, 0.0) - state.position.get(axis.name, 0.0)
        if axis.kind == "rotary":
            delta = math.degrees(delta) * 1e-3
        total += delta * delta
    return math.sqrt(total)


# ---------------------------------------------------------------------------- constant surface speed

def _css_seconds(ex, state: _Path, cmd: Command, feed_m_rev: float) -> float | None:
    """Time of a per-revolution move under constant surface speed: the integral of dl / (f n(r)) along
    the move, n(r) = min(v / (2 pi r), n_max) at the program radius r of the CSS axis."""
    css = state.css
    axis = ex.by_letter.get(css["axis"])
    if axis is None or feed_m_rev <= 0.0:
        return None
    index = ex.letters.index(css["axis"])
    offsets = (cmd.modal or {}).get("program_offset_m")
    offset = float(offsets[index]) if offsets is not None and index < len(offsets) else 0.0
    x0 = state.position.get(axis.name, 0.0) - offset
    x1 = cmd.target.get(axis.name, state.position.get(axis.name, 0.0)) - offset
    v = css["v"]
    n_max = css["max"]
    if v <= 0.0:
        return None
    if cmd.kind == Kind.ARC and cmd.arc is not None:
        pieces = _arc_pieces(ex, state, cmd, x0, x1, offset, axis)
    else:
        pieces = [(_axes_length(ex, state, cmd), x0, x1)]
    total = 0.0
    for length, a, b in pieces:
        total += length * _mean_inverse_speed(a, b, v, n_max)
    return 2.0 * math.pi * total / feed_m_rev


def _arc_pieces(ex, state: _Path, cmd: Command, x0: float, x1: float, offset: float, axis) -> list:
    """(length, x start, x end) of the chords of an arc (each a 1/count share of the arc's length): the
    start vector turned about the arc's normal by the signed sweep (right-handed, as ArcSpec defines it)."""
    arc = cmd.arc
    length = _axes_length(ex, state, cmd)
    count = max(1, int(math.ceil(abs(arc.sweep_rad) / ARC_CHORD_RAD)))
    if axis.letter not in "XYZ":
        return [(length, x0, x1)]
    centre = [float(v) for v in arc.centre]
    start = [state.position.get(ex.by_letter[l].name, centre[k]) if l in ex.by_letter else centre[k]
             for k, l in enumerate("XYZ")]
    normal = [float(v) for v in arc.normal]
    u = [start[k] - centre[k] for k in range(3)]
    along = sum(u[k] * normal[k] for k in range(3))
    u = [u[k] - along * normal[k] for k in range(3)]
    r0 = math.sqrt(sum(v * v for v in u))
    if r0 <= 0.0:
        return [(length, x0, x1)]
    u = [v / r0 for v in u]
    w = [normal[1] * u[2] - normal[2] * u[1], normal[2] * u[0] - normal[0] * u[2], normal[0] * u[1] - normal[1] * u[0]]
    k_x = "XYZ".index(axis.letter)
    pieces = []
    previous = x0
    for k in range(1, count + 1):
        t = k / count
        radius = arc.radius_start_m + (arc.radius_end_m - arc.radius_start_m) * t
        phi = arc.sweep_rad * t
        x = centre[k_x] + radius * (math.cos(phi) * u[k_x] + math.sin(phi) * w[k_x]) - offset
        if k == count:
            x = x1
        pieces.append((length / count, previous, x))
        previous = x
    return pieces


def _mean_inverse_speed(a: float, b: float, v: float, n_max: float) -> float:
    """Mean over x in [a, b] of 1 / omega(x), omega = min(v / |x|, n_max) (rad/s)."""
    floor = 1.0 / n_max if n_max > 0.0 else 0.0
    x_c = v / n_max if n_max > 0.0 else 0.0

    def g(x: float) -> float:
        return max(abs(x) / v, floor)

    if abs(b - a) < 1e-15:
        return g(a)
    lo, hi = min(a, b), max(a, b)
    cuts = sorted({lo, hi} | {c for c in (-x_c, 0.0, x_c) if lo < c < hi})
    integral = 0.0
    for u, w in zip(cuts, cuts[1:]):
        integral += 0.5 * (g(u) + g(w)) * (w - u)               # g is linear on every piece
    return integral / (hi - lo)


# ---------------------------------------------------------------------------- an ideal probe

class BoxProbe:
    """Ideal spindle probe for the dry run: a ball stylus whose tool length (the applied length offset,
    ``modal length_offset_m``) reaches the ball's bottom along the tool axis. It triggers where the ball,
    moving from the skip move's start (``probe['start_m']``) to its target, first touches one of the
    ``boxes`` (axis-aligned, in machine coordinates of a machine whose table frame is the machine frame:
    a 3-axis mill at its rotary zero). The contact point is found to ``tolerance_m`` by bisection on the
    distance of the ball centre from the boxes along the path."""

    def __init__(self, boxes, radius_m: float, tool_axis=(0.0, 0.0, 1.0), tolerance_m: float = 1e-7,
                 axis_names=("X", "Y", "Z"), radii_by_tool: dict | None = None):
        self.boxes = [(tuple(float(v) for v in lo), tuple(float(v) for v in hi)) for lo, hi in boxes]
        self.radius = float(radius_m)
        self.radii = {int(k): float(v) for k, v in (radii_by_tool or {}).items()}
        norm = math.sqrt(sum(float(v) ** 2 for v in tool_axis)) or 1.0
        self.axis = tuple(float(v) / norm for v in tool_axis)
        self.tolerance = float(tolerance_m)
        self.names = tuple(axis_names)

    def _centre(self, machine: dict, length: float, radius: float) -> tuple:
        # machine position = tip + length * axis (gauge line); the ball centre is r above the tip
        return tuple(float(machine.get(name, 0.0)) - (length - radius) * self.axis[k]
                     for k, name in enumerate(self.names))

    def _clearance(self, centre: tuple, radius: float) -> float:
        best = math.inf
        for lo, hi in self.boxes:
            d2 = 0.0
            for k in range(3):
                if centre[k] < lo[k]:
                    d2 += (lo[k] - centre[k]) ** 2
                elif centre[k] > hi[k]:
                    d2 += (centre[k] - hi[k]) ** 2
            best = min(best, math.sqrt(d2))
        return best - radius

    def __call__(self, path: int, cmd: Command) -> ProbeResult | None:
        start = (cmd.probe or {}).get("start_m")
        if not start or not self.boxes:
            return None
        end = dict(start)
        end.update(cmd.target)
        modal = cmd.modal or {}
        length = float(modal.get("length_offset_m") or 0.0)
        radius = self.radii.get(int(modal.get("tool") or 0), self.radius)
        distance = math.sqrt(sum((end.get(n, 0.0) - start.get(n, 0.0)) ** 2 for n in self.names))
        if distance <= 0.0:
            return None

        def point(t: float) -> dict:
            return {n: start.get(n, 0.0) + t * (end.get(n, 0.0) - start.get(n, 0.0)) for n in set(start) | set(end)}

        def touching(t: float) -> bool:
            return self._clearance(self._centre(point(t), length, radius), radius) <= 0.0

        if touching(0.0):
            return ProbeResult(True, point(0.0), 0.0)
        samples = max(2, int(math.ceil(distance / max(0.25 * radius, 1e-4))))
        previous = 0.0
        for k in range(1, samples + 1):
            t = k / samples
            if touching(t):
                lo, hi = previous, t
                while (hi - lo) * distance > self.tolerance:
                    mid = 0.5 * (lo + hi)
                    if touching(mid):
                        hi = mid
                    else:
                        lo = mid
                return ProbeResult(True, point(hi), 0.0)
            previous = t
        return None
