"""Dry run of the control against an ideal machine (``twin check``, the program view, tests).

The ideal machine executes each command at constant speed with no acceleration and no following
error: rapids move every axis independently at its rapid rate (FANUC non-linear positioning;
time = the slowest axis), cutting moves at the programmed feed (per revolution with the commanded
spindle speed, inverse time as 1/F), dwells for their time, tool changes take the machine's tool
change time, everything else is instantaneous. A command starts at max(previous end, ready_tick).
Buffer-stopping commands are answered at their end: live reads with the commanded position,
probes and tool setters by the ``probe``/``setter`` callbacks (no trigger when absent), M00/M01/
#3006/single block by an immediate cycle start, path waits by the controller's coordination.

The ideal machine is a check harness, not the physics: motion (cnctwin.motion) replaces it in a
run. Its cycle time is a lower bound of the real one (no acceleration limits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .controller import Controller
from .ir import Command, Feedback, FeedMode, Kind, ProbeResult

BUFFER_STOPPING = (Kind.SYNC, Kind.PROBE, Kind.TOOL_SETTER, Kind.STOP, Kind.MESSAGE)


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
        now = end
        state.free_tick = end
        commands[p].append(cmd)
        starts[p].append(start)
        ends[p].append(end)
        _apply(state, cmd)
        controller.retire(p, 1, end)
        if cmd.kind in BUFFER_STOPPING or cmd.kind == Kind.ALARM:
            _answer(controller, p, state, cmd, end, probe, setter, operator_delay_ticks)
    else:
        stopped = "time limit"
    return DryRunResult(commands, starts, ends, [list(s.events) for s in controller.state.paths],
                        max((e[-1] for e in ends if e), default=0),
                        [s.alarm for s in controller.state.paths], controller, stopped)


def _apply(state: _Path, cmd: Command) -> None:
    if cmd.kind == Kind.SPINDLE and cmd.spindle is not None:
        state.omega = float(cmd.spindle.get("omega_rad_s", 0.0))
    if cmd.target and cmd.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE, Kind.TOOL_SETTER):
        state.position.update(cmd.target)


def _answer(controller: Controller, p: int, state: _Path, cmd: Command, tick: int, probe, setter, delay: int) -> None:
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
        result = probe(p, cmd) if probe is not None else None
        fb.probe = result if result is not None else ProbeResult(False, dict(state.position), time_s)
        if fb.probe.triggered:
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
    if cmd.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE, Kind.TOOL_SETTER) and cmd.target:
        if cmd.kind == Kind.RAPID:
            for axis in ex.axes:
                delta = abs(cmd.target.get(axis.name, 0.0) - state.position.get(axis.name, 0.0))
                if delta > 0.0:
                    rate = axis.rapid if axis.rapid > 0.0 else 1.0
                    seconds = max(seconds, delta / rate)
        else:
            length = _path_length(ex, state, cmd)
            feed = cmd.feed
            if feed is not None and feed.mode == FeedMode.PER_REV:
                speed = feed.value * abs(state.omega) / (2.0 * math.pi)
                if speed <= 0.0 and length > 0.0:
                    return None
                seconds = length / speed if speed > 0.0 else 0.0
            elif feed is not None and feed.mode == FeedMode.INVERSE_TIME:
                seconds = 1.0 / feed.value if feed.value > 0.0 else 0.0
            elif feed is not None and feed.value > 0.0:
                seconds = length / feed.value
    elif cmd.kind == Kind.DWELL:
        seconds = cmd.dwell_s
        if cmd.dwell_rev and state.omega:
            seconds += cmd.dwell_rev * 2.0 * math.pi / abs(state.omega)
    elif cmd.kind == Kind.TOOL_CHANGE:
        seconds = controller.machine.tool_change_s
    return int(math.ceil(seconds / tick_s - 1e-9))


def _path_length(ex, state: _Path, cmd: Command) -> float:
    if cmd.kind == Kind.ARC and cmd.arc is not None:
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
        delta = cmd.target.get(axis.name, 0.0) - state.position.get(axis.name, 0.0)
        if axis.kind == "rotary":
            delta = math.degrees(delta) * 1e-3
        total += delta * delta
    return math.sqrt(total)
