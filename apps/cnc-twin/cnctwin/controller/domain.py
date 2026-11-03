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

    n_paths = len(build_machine(machine, settings).paths)
    texts = _get(operation, "program_texts") or split_paths(text, n_paths)
    sources = [ProgramSource(name, str(t), path=p) for p, t in enumerate(texts)]
    for item in _get(project, "program_library", []) or []:
        sources.append(ProgramSource(str(_get(item, "name")), str(_get(item, "text")), path=-1, main=False))
    setup_section = _get(operation, "setup")
    offsets = {}
    raw_offsets = _get(setup_section, "work_offsets_m") or {}
    for key in raw_offsets:
        value = raw_offsets[key]
        offsets[str(key)] = [float(_plain(v)) for v in value] if isinstance(value, (list, tuple)) else \
            {k: float(_plain(v)) for k, v in dict(value).items()}
    nozzles = {int(k): int(_plain(v)) for k, v in dict(_get(operation, "nozzle_positions", {}) or {}).items()}
    tools = []
    for entry in _get(_get(project, "tools"), "list", []) or []:
        number = int(_plain(_get(entry, "number")))
        wear = _get(entry, "wear_offsets_m") or {}
        tools.append(ToolOffsetInit(
            number=number, length_m=float(_plain(_get(entry, "length_offset_m", 0.0))),
            radius_m=float(_plain(_get(entry, "radius_offset_m", 0.0))),
            length_wear_m=float(_plain(_get(wear, "length", 0.0))), radius_wear_m=float(_plain(_get(wear, "radius", 0.0))),
            x_m=float(_plain(_get(entry, "x_offset_m", 0.0))), z_m=float(_plain(_get(entry, "z_offset_m", 0.0))),
            nose_radius_m=float(_plain(_get(entry, "nose_radius_m", 0.0))), tip=int(_plain(_get(entry, "tip", 0))),
            nozzle_position=nozzles.get(number), path=int(_plain(_get(entry, "path", 0)))))
    return ControllerSetup(machine, sources, settings, work_offsets_m=[offsets] + [{}] * (n_paths - 1), tools=tools,
                           tool_table=getattr(state, "tools", None), tick_s=float(ctx.tick_s))


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
