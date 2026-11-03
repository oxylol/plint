"""The control of one machine: one executor per path, path-wait synchronisation, checkpoint binding.

``Controller(setup)`` builds a fresh power-on state; ``Controller(setup, state)`` binds a state
restored with ``ControllerState.from_arrays`` (the program texts must be the same: the digest is
checked). Everything mutable lives in ``Controller.state``.

Path waits (FANUC multi-path wait M codes, Siemens WAITM in its dialect): a path that reaches a
wait code emits a SYNC command and stops interpreting; when motion reports it in position
(``feedback``), the path has *arrived*. When every path named by the wait has arrived at the same
code, all of them are released at the latest arrival tick. Different codes -> PS0160 on every
waiting path; a participant that has ended or stopped with an alarm -> TW0008 (the real control
would wait for ever).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import hooks
from .alarms import ControllerAlarm, load_controller_db
from .dialects import build_dialect
from .executor import PathContext, PathExecutor
from .lexer import build_store
from .machine import build_machine, plain
from .program import ProgramSource
from .schema import ControllerSettings
from .state import ControllerState, Frame, PathState, new_locals


@dataclass
class ToolOffsetInit:
    """Offsets of one tool as set on the machine (SI). Mill: length/radius; lathe: X/Z/nose radius."""

    number: int
    length_m: float = 0.0
    radius_m: float = 0.0
    length_wear_m: float = 0.0
    radius_wear_m: float = 0.0
    x_m: float = 0.0
    z_m: float = 0.0
    nose_radius_m: float = 0.0
    tip: int = 0
    x_wear_m: float = 0.0
    z_wear_m: float = 0.0
    nozzle_position: int | None = None
    path: int = 0


@dataclass
class ControllerSetup:
    machine: object                         # machine definition: SIMap or plain mapping (db/machines/<id>.yaml)
    programs: list[ProgramSource]
    settings: ControllerSettings = field(default_factory=ControllerSettings)
    controller_db: dict | None = None       # plain data of db/controllers/<id>.yaml; None = load it
    db_id: str | None = None                # None = machine.controller.alarms
    load_db: bool = True
    work_offsets_m: list[dict] = field(default_factory=list)   # per path: {"G54": [SI per axis] or {letter: SI}}
    tools: list[ToolOffsetInit] = field(default_factory=list)
    tool_table: object = None
    tick_s: float = 1e-6


class Controller:
    def __init__(self, setup: ControllerSetup, state: ControllerState | None = None):
        hooks.load_plugins()
        self.setup = setup
        self.machine = build_machine(setup.machine, setup.settings)
        raw_controller = self.machine.raw.controller
        dialect_name = str(plain(raw_controller.dialect))
        db_id = setup.db_id if setup.db_id is not None else str(plain(raw_controller.alarms) or "")
        db = setup.controller_db
        if db is None and setup.load_db:
            db = load_controller_db(db_id)
        self.dialect = build_dialect(dialect_name, self.machine, setup.settings, db, db_id if db else "")
        parse = self.dialect.parser or build_store
        sources = list(setup.programs) + hooks.library_sources(self.machine, self.dialect, setup.settings,
                                                               list(setup.programs))
        self.store, self.load_problems = parse(sources, self.dialect.lexer)
        params = dict(self.dialect.params)
        tick_s = float(setup.tick_s)
        params["horizon_ticks"] = max(1, int(math.ceil(params["horizon_s"] / tick_s)))
        params["min_block_ticks"] = max(1, int(round(params["min_block_s"] / tick_s)))
        self.params = params
        fresh = state is None
        if fresh:
            state = ControllerState(paths=[PathState(path=p) for p in range(len(self.machine.paths))],
                                    digest=self.store.digest)
            state.shared = {"c100": [None] * 100, "c500": [None] * 500}
        elif state.digest != self.store.digest:
            raise ValueError("the controller state was saved with different program texts")
        self.state = state
        self.executors: list[PathExecutor] = []
        for p in range(len(self.machine.paths)):
            if p not in self.store.mains:
                raise ValueError(f"no main program for controller path {p + 1}")
            offsets = setup.work_offsets_m[p] if p < len(setup.work_offsets_m) else {}
            tools = [t for t in setup.tools if t.path == p]
            ctx = PathContext(self.dialect, self.store, self.machine, p, params, tick_s, setup.tool_table, self,
                              offsets, tools, {t.number: t.nozzle_position for t in tools
                                               if t.nozzle_position is not None})
            executor = PathExecutor(ctx, state.paths[p], state)
            if fresh:
                path_state = state.paths[p]
                path_state.frames = [Frame("main", self.store.mains[p], 0, 0)]
                path_state.locals = [new_locals()]
                executor.handler.init_state()
            self.executors.append(executor)
        if fresh:
            for problem in self.load_problems:
                self.state.paths[0].events.append({"kind": "warning", "tick": 0, "message": problem,
                                                   "program": "", "line": 0, "block": ""})

    # ------------------------------------------------------------------ lock-step
    def produce(self, path: int, now_tick: int):
        commands = self.executors[path].produce(now_tick)
        self.check_waits(now_tick)
        return commands

    def retire(self, path: int, count: int, tick: int) -> None:
        self.executors[path].retire(count, tick)

    def feedback(self, path: int, fb, tick: int) -> None:
        self.executors[path].feedback(fb, tick)

    def cycle_start(self, path: int, tick: int) -> None:
        self.executors[path].cycle_start(tick)

    def reset(self, path: int, tick: int) -> None:
        self.executors[path].reset(tick)

    def waiting(self, path: int) -> dict | None:
        return self.state.paths[path].waiting

    def done(self) -> bool:
        """Every path has ended its program (waiting for the next part) or stopped with an alarm."""
        return all(p.alarm is not None or (p.waiting is not None and p.waiting["kind"] == "end")
                   for p in self.state.paths)

    # ------------------------------------------------------------------ path waits
    def arrive(self, path: int, tick: int) -> None:
        waiting = self.state.paths[path].waiting
        if waiting is None or waiting["kind"] != "path_wait":
            return
        participants = list(waiting["paths"])
        arrivals = []
        for q in participants:
            other = self.state.paths[q].waiting
            if other is None or other.get("kind") != "path_wait" or "arrived" not in other:
                self.check_waits(tick)
                return
            if other["code"] != waiting["code"] or sorted(other["paths"]) != sorted(participants):
                self._wait_alarm(participants, "wait_mismatch",
                                 f"path {path + 1} waits at M{waiting['code']}, path {q + 1} at M{other['code']}", tick)
                return
            arrivals.append(int(other["arrived"]))
        release = max(arrivals)
        for q in participants:
            self.executors[q].release_wait(release)

    def check_waits(self, tick: int) -> None:
        """A path waiting for a path that has ended or stopped with an alarm can never go on."""
        for p, path_state in enumerate(self.state.paths):
            waiting = path_state.waiting
            if waiting is None or waiting["kind"] != "path_wait":
                continue
            for q in waiting["paths"]:
                other = self.state.paths[q]
                if q == p:
                    continue
                ended = other.waiting is not None and other.waiting["kind"] == "end"
                if other.alarm is not None or ended:
                    self._wait_alarm([p], "wait_never_released",
                                     f"path {p + 1} waits at M{waiting['code']} for path {q + 1}, which has "
                                     f"{'stopped with an alarm' if other.alarm else 'ended'}", tick)
                    break

    def _wait_alarm(self, paths: list[int], key: str, detail: str, tick: int) -> None:
        for q in paths:
            executor = self.executors[q]
            if executor.s.alarm is None:
                executor.s.waiting = None
                executor.s.interp_tick = max(executor.s.interp_tick, int(tick))
                executor._stop_alarm(ControllerAlarm(key, detail))
