"""ControllerDomain: the control in a run (core.domain.Domain), and its lock-step API for the loop.

The controller writes no bus record (architecture 7): it hands :class:`~cnctwin.controller.ir.Command`
objects to motion through the lock-step API below, which ``core.loop`` calls:

    cmds = dom.produce(ctx, state, path, now_tick)    # start of each macro step and at sync ticks
    dom.retire(ctx, state, path, count, tick)          # motion finished the oldest `count` commands
    dom.feedback(ctx, state, path, Feedback(...), tick) # answer to the buffer-stopping command
    dom.cycle_start(ctx, state, path, tick)            # operator: CYCLE START (after M00, next part)
    dom.operator(state, path) -> OperatorPanel          # overrides, feed hold, single block (motion reads)
    dom.done(ctx, state) -> bool                       # every path ended or stopped by an alarm

It reads no bus record either (machine feedback arrives through ``feedback``). ``macro_step`` turns
new controller events (alarms, warnings, notes) into findings and updates the channel values
(``state.controller`` is the only state it owns).
"""

from __future__ import annotations

import re

import numpy as np

from ..core.findings import Category, Severity
from .channels import CHANNELS, MAX_PATHS, WAIT_CODES
from .controller import Controller, ControllerSetup, ToolOffsetInit
from .program import ProgramSource
from .schema import settings_from


def _get(obj, name, default=None):
    if obj is None:
        return default
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return default if value is None else value


def _plain(value):
    return value.value if hasattr(value, "value") and hasattr(value, "src") else value


def split_paths(text: str, n_paths: int) -> list[str]:
    """A multi-path program file: lines ``$1``, ``$2`` ... start each path's section (Swiss / twin-turret
    convention); a file without them belongs to path 1."""
    if n_paths <= 1:
        return [text]
    sections: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines():
        marker = line.strip()
        if marker.startswith("$") and marker[1:].isdigit():
            current = int(marker[1:]) - 1
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    if not sections:
        return [text] + [""] * (n_paths - 1)
    return ["\n".join(sections.get(p, [])) for p in range(n_paths)]


_NUMBERED_OFFSET = re.compile(r"^(?:G54\.1\s*P|G154\s*P|PRESET\s*|P)(\d+)$")
_HAAS_EXTRA_OFFSET = re.compile(r"^G1(\d\d)$")


def offset_key(name: str) -> str:
    """The control's key of a work offset as the project names it (setup.work_offsets_mm): G54-G59 as
    written; the numbered systems G54.1 Pn (FANUC), G154 Pn and G110-G129 (Haas, G110 = P1) and the
    HEIDENHAIN preset table row (PRESET n) are the executor's ``Pn``."""
    text = " ".join(str(name).upper().split())
    match = _NUMBERED_OFFSET.match(text)
    if match is not None:
        return f"P{int(match.group(1))}"
    match = _HAAS_EXTRA_OFFSET.match(text)
    if match is not None and 10 <= int(match.group(1)) <= 29:
        return f"P{int(match.group(1)) - 9}"
    return text


CARTESIAN = ("X", "Y", "Z")


def offset_row(values: list[float], letters: list[str]) -> list[float]:
    """A work offset the project writes as a list (``setup.work_offsets_mm``): the machine coordinates of the
    work zero as X, Y, Z, then the path's other axes in the control's order (docs/formats.md), on every machine
    type (sanity reads the same [x, y, z]). Returned in the control's axis order ``letters``, the order the
    executor reads a list in: the same on mills (X Y Z first), X Z Y C on lathes. A value for an axis the path
    does not have is dropped."""
    order = list(CARTESIAN) + [l for l in letters if l not in CARTESIAN]
    given = {letter: float(v) for letter, v in zip(order, values)}
    return [given.get(letter, 0.0) for letter in letters]


def setup_from_project(ctx, state, op_index: int = 0) -> ControllerSetup:
    """ControllerSetup of operation ``op_index`` from the resolved project (architecture 18)."""
    project = ctx.project
    machine = _get(project, "machine")
    physics = _get(project, "physics")
    settings = settings_from(_get(physics, "controller"))
    operation = _get(project, "operations")[op_index]
    text = str(_get(operation, "program_text", ""))
    name = str(_get(operation, "program", "program.nc"))
    from .machine import build_machine

    model = build_machine(machine, settings)
    n_paths = len(model.paths)
    texts = _get(operation, "program_texts") or split_paths(text, n_paths)
    sources = [ProgramSource(name, str(t), path=p) for p, t in enumerate(texts)]
    for item in _get(project, "program_library", []) or []:
        sources.append(ProgramSource(str(_get(item, "name")), str(_get(item, "text")), path=-1, main=False))
    setup_section = _get(operation, "setup")
    offsets = {}
    raw_offsets = _get(setup_section, "work_offsets_m") or {}
    letters = [a.letter for a in model.paths[0]] if model.paths else []
    for key in raw_offsets:
        value = raw_offsets[key]
        offsets[offset_key(str(key))] = offset_row([float(_plain(v)) for v in value], letters) \
            if isinstance(value, (list, tuple)) else {k: float(_plain(v)) for k, v in dict(value).items()}
    nozzles = {int(k): int(_plain(v)) for k, v in dict(_get(operation, "nozzle_positions", {}) or {}).items()}
    tools = [tool_offsets(entry, nozzles, probe=str(_plain(_get(entry, "kind", ""))) == "probe")
             for entry in _get(_get(project, "tools"), "list", []) or []]          # (a raw project keeps its probes here)
    tools += [tool_offsets(entry, nozzles, probe=True) for entry in _get(_get(project, "tools"), "probes", []) or []]
    return ControllerSetup(machine, sources, settings, work_offsets_m=[offsets] + [{}] * (n_paths - 1), tools=tools,
                           tool_table=getattr(state, "tools", None), tick_s=float(ctx.tick_s))


def tool_offsets(entry, nozzles: dict | None = None, *, probe: bool = False) -> ToolOffsetInit:
    """The offsets one ``tools.list`` (or ``tools.probes``) entry sets on the control (SI view of the project:
    length/radius geometry and wear; lathe X (as set, a diameter on a diameter-programmed lathe), Z, nose
    radius, tip number and X/Z wear; the file's 1-based path number). A probe's radius register defaults to
    the nominal stylus radius (half the ball) when the entry gives none."""
    number = int(_plain(_get(entry, "number")))
    wear = _get(entry, "wear_offsets_m") or {}
    radius = _get(entry, "radius_offset_m")
    name = None
    if probe:
        definition = _get(entry, "probe")
        if isinstance(_plain(definition), str):             # a raw entry naming the machine's probe
            name, ball = str(_plain(definition)), None
        else:
            name = str(_get(definition, "name") or _get(definition, "extends") or "inline")
            ball = _get(definition, "stylus_ball_m")
        if radius is None:
            radius = 0.5 * float(_plain(ball)) if ball is not None else 0.0
    path = _get(entry, "path")
    return ToolOffsetInit(
        number=number, length_m=float(_plain(_get(entry, "length_offset_m", 0.0))),
        radius_m=float(_plain(radius if radius is not None else 0.0)),
        length_wear_m=float(_plain(_get(wear, "length", 0.0))), radius_wear_m=float(_plain(_get(wear, "radius", 0.0))),
        x_m=float(_plain(_get(entry, "x_offset_m", 0.0))), z_m=float(_plain(_get(entry, "z_offset_m", 0.0))),
        nose_radius_m=float(_plain(_get(entry, "nose_radius_m", 0.0))), tip=int(_plain(_get(entry, "tip", 0))),
        x_wear_m=float(_plain(_get(wear, "x", 0.0))), z_wear_m=float(_plain(_get(wear, "z", 0.0))),
        nozzle_position=(nozzles or {}).get(number), path=int(_plain(path)) - 1 if path is not None else 0,
        probe=name)


def _pose4(operation, resolved: str, written: str) -> np.ndarray:
    """An operation's 4x4 pose: the resolver's matrix, else the setup's pose as written (SI view)."""
    matrix = _get(operation, resolved)
    if matrix is not None:
        return np.asarray(matrix, dtype=float)
    pose = _get(_get(operation, "setup"), written)
    if pose is None:
        return np.eye(4)
    if hasattr(pose, "matrix") and callable(pose.matrix):
        return np.asarray(pose.matrix(), dtype=float)
    from ..db.simap import pose_matrix

    translate = [float(_plain(v)) for v in (_get(pose, "translate_m") or [0.0, 0.0, 0.0])]
    rotate = [float(_plain(v)) for v in (_get(pose, "rotate_rad") or [0.0, 0.0, 0.0])]
    return np.asarray(pose_matrix(translate, rotate, str(_plain(_get(pose, "order", "xyz")))), dtype=float)


def _stock_box(stock):
    """The stock box in its frame S from the stock definition (a project without the resolver's
    ``stock_box_S_m``): a block from the origin, a cylinder or tube about +z from z = 0; else None."""
    size = _get(stock, "size_m")
    kind = str(_plain(_get(stock, "kind", "block")))
    values = [float(_plain(v)) for v in size] if isinstance(size, (list, tuple)) else None
    if kind == "block" and values is not None and len(values) == 3:
        return [[0.0, 0.0, 0.0], values]
    if kind in ("cylinder", "tube") and values is not None and len(values) >= 2:
        r = 0.5 * values[0]
        return [[-r, -r, 0.0], [r, r, values[-1]]]
    return None


def stock_probe(project, op_index: int = 0):
    """An ideal spindle probe (:class:`cnctwin.controller.dryrun.BoxProbe`) against the stock of operation
    ``op_index`` as set up, for dry runs of programs that probe the part: the stock box placed by
    ``fixture_pose * stock_pose`` (the table frame is the machine frame: 3-axis mills, and five-axis tables
    at rotary zero), the stylus ball of each probe T number of ``tools.probes`` (else the machine's spindle
    probe). None for lathes, an unknown stock box or no stylus size."""
    from .dryrun import BoxProbe

    machine = _get(project, "machine")
    if str(_plain(_get(machine, "type", ""))) in ("lathe", "mill_turn", "swiss"):
        return None
    operation = _get(project, "operations")[op_index]
    box = _get(operation, "stock_box_S_m") or _stock_box(_get(project, "stock"))
    if box is None:
        return None
    pose = _pose4(operation, "fixture_pose_F_H", "fixture_pose") @ _pose4(operation, "stock_pose_W_S", "stock_pose")
    lo, hi = (np.asarray([float(_plain(v)) for v in corner]) for corner in box)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    moved = corners @ pose[:3, :3].T + pose[:3, 3]
    radii = {}
    tools = _get(project, "tools")
    entries = list(_get(tools, "probes", []) or []) + [e for e in (_get(tools, "list", []) or [])
                                                        if str(_plain(_get(e, "kind", ""))) == "probe"]
    for entry in entries:
        definition = _get(entry, "probe")
        ball = _get(definition, "stylus_ball_m") if not isinstance(_plain(definition), str) else None
        if ball is not None:
            radii[int(_plain(_get(entry, "number")))] = 0.5 * float(_plain(ball))
    machine_ball = _get(_get(_get(machine, "probes"), "spindle_probe"), "stylus_ball_m")
    default = 0.5 * float(_plain(machine_ball)) if machine_ball is not None else (min(radii.values()) if radii else None)
    if default is None:
        return None
    axis = _get(_get(machine, "kinematics"), "tool_axis") or (0.0, 0.0, 1.0)
    return BoxProbe([(tuple(moved.min(axis=0)), tuple(moved.max(axis=0)))], default, radii_by_tool=radii,
                    tool_axis=tuple(float(_plain(v)) for v in axis))


class ControllerDomain:
    name = "controller"

    def __init__(self, setup: ControllerSetup | None = None):
        self._setup = setup
        self._controller: Controller | None = None
        self._last_seq: list[int] = []
        self._carry: list[dict] = []
        self.last_channels: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ Domain protocol
    def channels(self):
        return list(CHANNELS)

    def init(self, ctx, state) -> None:
        setup = self._setup or setup_from_project(ctx, state, getattr(state.clock, "op_index", 0))
        self._setup = setup
        self._controller = Controller(setup)
        state.controller = self._controller.state
        self._last_seq = [0] * len(self._controller.executors)

    def controller(self, ctx, state) -> Controller:
        """The Controller bound to ``state.controller`` (rebuilt after a checkpoint restore)."""
        if self._controller is None or self._controller.state is not state.controller:
            if self._setup is None:
                self._setup = setup_from_project(ctx, state, getattr(state.clock, "op_index", 0))
            self._controller = Controller(self._setup, state.controller)
            self._last_seq = [p.block_seq for p in state.controller.paths]
        return self._controller

    def kernel_inputs(self, ctx, state, bus):
        return None

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        return None

    def macro_step(self, ctx, state, prev, nxt) -> None:
        """Findings for new controller events; channel values of the step (``last_channels``, read by the
        recorder through :meth:`channel_values`)."""
        self.controller(ctx, state)
        self.raise_findings(ctx, state)
        self.last_channels = self.channel_values(ctx, state)

    def end_operation(self, ctx, state) -> None:
        """Keep what survives a program change on the control: commons #500-#999 (FANUC) / #10000+ (Haas)
        and the part counters; the next operation's controller starts from them."""
        controller = self.controller(ctx, state)
        carry = []
        for path in controller.state.paths:
            kept = {k: list(v) for k, v in path.variables.items() if k in ("c500", "c10000")}
            carry.append({"variables": kept, "sys": {k: v for k, v in path.sys.items()
                                                     if k in ("parts", "parts_required", "m30_count_1", "m30_count_2")}})
        self._carry = carry

    def start_operation(self, ctx, state, op_index: int) -> None:
        self._setup = setup_from_project(ctx, state, op_index)
        self._controller = Controller(self._setup)
        for path, carry in zip(self._controller.state.paths, self._carry):
            path.variables.update({k: list(v) for k, v in carry["variables"].items()})
            path.sys.update(carry["sys"])
        state.controller = self._controller.state
        self._last_seq = [0] * len(self._controller.executors)

    def finalize(self, ctx, state) -> dict:
        controller = self.controller(ctx, state)
        out = {"paths": []}
        for p, path in enumerate(controller.state.paths):
            out["paths"].append({
                "path": p + 1, "blocks": path.block_seq, "commands": path.issued, "alarm": path.alarm,
                "parts": int(path.sys.get("parts", path.sys.get("m30_count_1", 0))),
                "dprnt": list(path.printed), "events": len(path.events),
                "interp_time_s": path.interp_tick * controller.setup.tick_s})
        return out

    # ------------------------------------------------------------------ lock-step API
    def produce(self, ctx, state, path: int, now_tick: int):
        return self.controller(ctx, state).produce(path, now_tick)

    def retire(self, ctx, state, path: int, count: int, tick: int) -> None:
        self.controller(ctx, state).retire(path, count, tick)

    def feedback(self, ctx, state, path: int, fb, tick: int) -> None:
        self.controller(ctx, state).feedback(path, fb, tick)

    def cycle_start(self, ctx, state, path: int, tick: int) -> None:
        self.controller(ctx, state).cycle_start(path, tick)

    def reset(self, ctx, state, path: int, tick: int) -> None:
        self.controller(ctx, state).reset(path, tick)

    def operator(self, state, path: int):
        return state.controller.paths[path].operator

    def set_operator(self, state, path: int, **changes) -> None:
        panel = state.controller.paths[path].operator
        for key, value in changes.items():
            if not hasattr(panel, key):
                raise AttributeError(f"OperatorPanel has no {key}")
            setattr(panel, key, value)

    def done(self, ctx, state) -> bool:
        return self.controller(ctx, state).done()

    def waiting(self, state, path: int):
        return state.controller.paths[path].waiting

    # ------------------------------------------------------------------ findings and channels
    def raise_findings(self, ctx, state) -> None:
        for p, path in enumerate(state.controller.paths):
            while path.reported < len(path.events):
                event = path.events[path.reported]
                path.reported += 1
                kind = event.get("kind")
                if kind == "alarm":
                    severity, category = Severity.ERROR, Category.ALARM
                    message = f"{event['code']} {event['text']}" + (f" ({event['detail']})" if event.get("detail") else "")
                elif kind == "warning":
                    severity, category = Severity.WARNING, Category.PROGRAM_ERROR
                    message = event["message"]
                else:
                    severity, category = Severity.INFO, Category.PROGRAM_ERROR
                    message = event["message"]
                finding = ctx.finding(state, self.name, category, severity, message, int(event["tick"]), path=p,
                                      quantities={"path": float(p + 1)})
                finding.program = event.get("program", finding.program)
                finding.line = int(event.get("line", finding.line) or 0)
                finding.block = event.get("block", finding.block)

    def channel_values(self, ctx, state) -> dict[str, np.ndarray]:
        controller = self.controller(ctx, state)
        paths = controller.state.paths
        now = state.clock.tick
        fill = np.full(MAX_PATHS, np.nan)
        lead = np.full(MAX_PATHS, np.nan)
        rate = np.full(MAX_PATHS, np.nan)
        waiting = np.zeros(MAX_PATHS, np.int8)
        alarm = np.zeros(MAX_PATHS, bool)
        macro_s = ctx.macro_s
        while len(self._last_seq) < len(paths):
            self._last_seq.append(0)
        for p, path in enumerate(paths[:MAX_PATHS]):
            executor = controller.executors[p]
            fill[p] = len(path.buffer) / executor.capacity()
            lead[p] = (path.interp_tick - now) * ctx.tick_s
            rate[p] = (path.block_seq - self._last_seq[p]) / macro_s if macro_s > 0 else 0.0
            self._last_seq[p] = path.block_seq
            waiting[p] = WAIT_CODES.get(path.waiting["kind"] if path.waiting else None, 0)
            alarm[p] = path.alarm is not None
        return {"controller.buffer_fill": fill, "controller.interp_lead_s": lead, "controller.block_rate_Hz": rate,
                "controller.waiting": waiting, "controller.alarm": alarm}
