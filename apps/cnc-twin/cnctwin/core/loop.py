"""The run loop (docs/models/loop.md; architecture 5.3, 7, 9, 12, 16): a whole project through every domain.

* :func:`run_project` ``(project_path, run_dir, options, progress_cb) -> dict`` runs the project's operations in
  sequence (with their setups and ``from_previous`` transfers) for every part of its production batch, writes the
  run directory (architecture 16) and makes the derived outputs (cnctwin.core.outputs.finalize_outputs). Options:
  ``resume_from`` (a committed checkpoint marker: continue that run bit-identically), ``replay`` ({tick0, tick1,
  checkpoint}: the fast rows of a window, see :func:`replay_window`), ``max_sim_s`` (stop after this much simulated
  time: the operation running is ended and the run is marked incomplete), ``parts`` (overrides
  ``production.parts``), ``checkpoint_interval_s``, ``outputs`` (cnctwin.core.outputs settings; ``false`` makes
  none), ``sanity`` (false: no static program checks at operation start).
* :func:`replay_window` ``(run_dir, tick0, tick1, options, progress_cb) -> dict`` (``twin replay --window``).
* :func:`check_project` (``twin check``) lives in :mod:`cnctwin.core.check` and is re-exported here.

One macro step (:meth:`Run.step`, architecture 5.3): the hot loop (controller -> motion -> every domain's
kernel inputs -> run_macro with sync callbacks into the controller -> absorb of motion, cutting, removal,
dynamics), chips' absorb of the step's emission rows, the slow domains' macro steps (bus_prev -> own record in
bus_next; their cadences are their own), the coupling-lag monitor, the conservation ledger, the exact-volume
check, recorder / frames / timeline / findings, the bus swap and the clock, keyframes and checkpoints when due.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .bus import Bus
from .domain import RunContext
from .errors import ConservationError, InputError, TwinError
from .findings import Category, Severity
from .state import SimState

LOOP_DOMAIN = "loop"


class CouplingMonitor:
    """The coupling-lag monitor of architecture 5.3 (settings in db/models/loop.yaml): per named quantity the last
    value, the last significant change, the count of successive significant changes that alternate in sign with a
    magnitude ratio above ``coupling_ratio`` (each change above ``coupling_rel_change`` of the value). A quantity
    is flagged once, when its count reaches ``coupling_alternations``."""

    def __init__(self, ratio: float | None = None, rel_change: float | None = None, alternations: int | None = None):
        if ratio is None or rel_change is None or alternations is None:
            from ..db.models import params
            p = params("loop")
            ratio = float(p["coupling_ratio"]) if ratio is None else ratio
            rel_change = float(p["coupling_rel_change"]) if rel_change is None else rel_change
            alternations = int(p["coupling_alternations"]) if alternations is None else alternations
        self.ratio = float(ratio)
        self.rel_change = float(rel_change)
        self.alternations = int(alternations)
        self.entries: dict = {}            # name -> [value, change, count, flagged]

    def update(self, quantities: dict) -> list[tuple]:
        """Take this step's values; returns (name, value, change, previous change, count) of each quantity flagged
        now."""
        flagged = []
        for name in sorted(quantities):
            value = float(quantities[name])
            if not math.isfinite(value):
                continue
            entry = self.entries.get(name)
            if entry is None:
                self.entries[name] = [value, 0.0, 0, False]
                continue
            last, change, count, done = entry
            new_change = value - last
            significant = abs(new_change) > self.rel_change * max(abs(value), abs(last), 1e-300)
            alternates = change != 0.0 and new_change * change < 0.0 and abs(new_change) > self.ratio * abs(change)
            if significant and alternates:
                count += 1
            elif significant:
                count = 0
            if count >= self.alternations and not done:
                flagged.append((name, value, new_change, change, count))
                done = True
            self.entries[name] = [value, new_change if significant else change, count, done]
        return flagged

    def state(self) -> dict:
        return {k: list(v) for k, v in sorted(self.entries.items())}

    def restore(self, saved: dict) -> None:
        self.entries = {str(k): [float(v[0]), float(v[1]), int(v[2]), bool(v[3])] for k, v in (saved or {}).items()}


class LoopParticipant:
    """The loop's own variables in a checkpoint (which operation and part, the feeder, the exact-volume baseline,
    the global timeline offset, the hot loop's arrays, the adapters' change tracking ...)."""

    def __init__(self, run):
        self.run = run

    def checkpoint_cut(self) -> dict:
        return {}

    def checkpoint_state(self) -> dict:
        return self.run.loop_state()

    def restore_state(self, state) -> None:
        self.run.saved_loop = dict(state)

    def truncate_outputs(self, cut) -> None:
        return None


class Run:
    """One run of a project (module docstring). ``replay``: (tick0, tick1, checkpoint marker or None) for a
    deterministic replay that writes only the fast rows of the window."""

    def __init__(self, project_path, run_dir, options=None, progress_cb=None, *, replay=None):
        from ..db import Database
        from ..project import resolve
        self.options = dict(options or {})
        self.progress_cb = progress_cb
        self.replay = replay
        self.project_path = Path(project_path)
        self.run_dir = Path(run_dir).absolute()
        db = Database.load()
        project = resolve(self.project_path, db)
        machine = project.machine
        physics = project.physics
        tick_s = float(getattr(physics, "tick_s", None) or 1e-6)
        ipo_s = float(machine.controller.ipo_period_s)
        ipo_ticks = int(round(ipo_s / tick_s))
        if abs(ipo_ticks * tick_s - ipo_s) > 1e-12 * max(ipo_s, 1.0) or ipo_ticks <= 0:
            raise InputError(f"the machine's IPO period {ipo_s} s is not a whole number of {tick_s} s base ticks")
        seed = int(getattr(project, "seed", None) or 0)
        self.ctx = RunContext(project=project, db=db, run_dir=None if replay else self.run_dir, seed=seed,
                              tick_s=tick_s, ipo_ticks=ipo_ticks, macro_ipo=int(physics.macro_ipo))
        self.state = SimState()
        self.ops = list(project.operations)
        parts = self.options.get("parts")
        if parts is None:
            prod = getattr(project, "production", None)
            parts = getattr(prod, "parts", None) if prod is not None else None
        self.parts = max(int(parts or 1), 1)
        self.domains: dict = {}
        self.motion = None
        self.hot = None
        self.feeder = None
        self.inspect = None
        self.results = None
        self.field = None
        self.recorder = self.frames = self.keyframes = self.timeline = self.ledger = None
        self.checkpointer = self.rd = None
        # loop variables (checkpointed through LoopParticipant)
        self.op_active = False
        self.done = False
        self.ops_started = 0
        self.tl_base = 0
        self.tl_first = 0
        self.v0: list = []
        self.booked: list = []
        self.part_mass: dict = {}
        self.part_mass_start = 0.0
        self.part_start_tick = 0           # base tick the current part's first operation started (SurfaceBurrs)
        self.findings_written = -1
        self.est_total_s = None
        self.truncated = None
        self.stopped = None                # {at_s, reason: alarm, op, op_index, part, alarms} when a control alarmed
        self.monitor = CouplingMonitor()
        self.identity = {"in_J": 0.0, "out_J": 0.0, "abs_J": 0.0}
        self.unregistered: set = set()
        self.notes: set = set()
        self.saved_loop: dict = {}
        self.part_files: list = []
        self.last_ended = ("", -1)
        self.l0_seen: dict = {}            # operation name -> the L0 strides its steps used
        self.cpu = {"hot_s": 0.0, "l2_s": 0.0, "record_s": 0.0, "steps": 0, "l2_by_domain_s": {}, "split": {}}

    # ================================================================== small helpers
    @property
    def project(self):
        return self.ctx.project

    def op_name(self) -> str:
        i = int(self.state.clock.op_index)
        return str(self.ops[i].name) if 0 <= i < len(self.ops) else ""

    def progress(self, **fields) -> None:
        if self.progress_cb is not None:
            self.progress_cb(fields)

    def trace(self, event: str, quantities: dict | None = None) -> int:
        tick = int(self.state.clock.tick)
        return self.ctx.trace.record(tick * self.ctx.tick_s, tick, LOOP_DOMAIN, event, quantities or {}, [])

    # ================================================================== set-up
    def _register_channels(self) -> None:
        from .hotloop import channels as hot_channels
        from .loop_wiring import RunLedger
        from ..motion.channels import channels as motion_channels
        registry = self.ctx.channels
        registry.register_all(motion_channels())
        for name in sorted(self.domains):
            registry.register_all(list(self.domains[name].channels()))
        registry.register_all(RunLedger.channels())
        registry.register_all(hot_channels())
        if self.inspect is not None:
            registry.register_all(list(self.inspect.channels()))

    def _new_motion(self, op_index: int):
        from ..motion.domain import MotionDomain
        from ..motion.machine import MachineModel
        from .drive import fixture_pose_F
        project = self.project
        machine = project.machine
        op = self.ops[op_index]
        mtype = str(getattr(machine, "type", "") or "")
        F_H = fixture_pose_F(machine, op.fixture_pose_F_H) if mtype.endswith("mill") else \
            np.asarray(op.fixture_pose_F_H, float)
        ctrl = getattr(project.physics, "controller", None)
        letters = getattr(ctrl, "paths", None) if ctrl is not None else None
        P = len(MachineModel(machine, letters).paths)
        return MotionDomain(poses_F_W=[F_H] * P, poses_W_S=[np.asarray(op.stock_pose_W_S, float)] * P)

    def _motion_meta(self) -> None:
        ms = self.state.motion
        ms.meta["op_name"] = self.op_name()
        ms.meta["part"] = int(self.state.clock.part_index)
        ms.meta["op_index"] = int(self.state.clock.op_index)

    def _init_domains(self) -> None:
        """Every domain's init in dependency order on ``self.state`` (loop_wiring.INIT_ORDER)."""
        from .loop_wiring import build_domains, inspect_domain, late_wiring
        ctx, state = self.ctx, self.state
        self.domains = build_domains(self)
        d = self.domains
        self.inspect = inspect_domain(self)
        if self.results is None:
            from .loop_wiring import LoopResults
            self.results = LoopResults(self)
        self._register_channels()
        d["tooling"].init(ctx, state)
        self._build_recorder()
        d["controller"].init(ctx, state)
        self.motion = self._new_motion(int(state.clock.op_index))
        self.motion.init(ctx, state)
        self._motion_meta()
        ctx.timeline = self.motion.timeline
        for name in ("cutting", "removal", "dynamics", "structure", "holding"):
            d[name].init(ctx, state)
        late_wiring(self)
        for name in ("thermal", "coolant", "chips", "wear", "surface"):
            d[name].init(ctx, state)
        if self.inspect is not None:
            self.inspect.init(ctx, state)

    def _build_recorder(self) -> None:
        from .check import ring_l0_ticks
        from .recorder import Recorder
        ctx = self.ctx
        l0_min = ring_l0_ticks(ctx, self.state)
        replay = (int(self.replay[0]), int(self.replay[1])) if self.replay else None
        self.recorder = Recorder(self.run_dir, ctx.channels, l0_ticks_min=l0_min, ipo_ticks=ctx.ipo_ticks,
                                 macro_ipo=ctx.macro_ipo, tick_s=ctx.tick_s,
                                 recording=getattr(self.project, "recording", None), replay=replay)
        ctx.recorder = self.recorder

    def _writers(self, fresh: bool) -> None:
        from .checkpoint import Checkpointer
        from .keyframes import KeyframeWriter
        from .loop_outputs import frames_layout
        from .loop_wiring import run_ledger
        from .rundir import FramesWriter, Timeline
        ctx = self.ctx
        rec = getattr(self.project, "recording", None)
        period_s = float(getattr(rec, "keyframe_period_s", None) or 0.5) if rec is not None else 0.5
        wp = self.state.removal.workpieces[0]
        f = wp.field
        axes, paths = frames_layout(self.motion)
        self.frames = FramesWriter(self.run_dir / "frames.bin", axes=axes, paths=paths, tick_s=ctx.tick_s,
                                   ipo_ticks=ctx.ipo_ticks)
        if fresh:
            self.frames.open()
        self.keyframes = KeyframeWriter(self.run_dir, period_ticks=max(1, int(round(period_s / ctx.tick_s))),
                                        tick_s=ctx.tick_s, lattice_m=float(f.lv_cell_m[0]),
                                        origin_S_m=np.asarray(f.lv_origin_m[0], float),
                                        grid=np.asarray(f.lv_dims[0], int))
        self.timeline = Timeline(self.run_dir, tick_s=ctx.tick_s)
        self.ledger = run_ledger(ctx, set(self.domains) | {"motion"})
        interval = self.options.get("checkpoint_interval_s")
        self.checkpointer = Checkpointer(self.run_dir, project_hash=self.project.hash, tick_s=ctx.tick_s,
                                         interval_ticks=int(round(float(interval) / ctx.tick_s)) if interval else None)

    def participants(self) -> dict:
        out = {"recorder": self.recorder, "frames": self.frames, "keyframes": self.keyframes,
               "timeline": self.timeline, "rundir": self.rd, "ledger": self.ledger, "loop": LoopParticipant(self)}
        if self.inspect is not None:
            out["inspect"] = self.inspect
        return out

    def fresh(self) -> None:
        """A new run: the run directory prepared, every domain initialised, the writers opened, scene.json and
        keyframe 0 written, the estimate of the simulated time from the controller's dry runs."""
        from .rundir import RunDirectory
        from .trace import TraceLog
        self.rd = RunDirectory(self.run_dir, own_status=False, tick_s=self.ctx.tick_s)
        self.rd.prepare()
        self.ctx.trace = TraceLog(self.run_dir / "trace.jsonl")
        self.progress(phase="init", message="initialising the domains", sim_time_s=0.0)
        self._init_domains()
        self.recorder.write_channels()
        self._writers(fresh=True)
        from .loop_outputs import write_keyframe, write_scene
        write_scene(self)
        write_keyframe(self, 0, full=True)
        self.est_total_s = self._estimate_total_s()
        self.trace("loop:start", {"operations": float(len(self.ops)), "parts": float(self.parts),
                                  "est_total_s": float(self.est_total_s or 0.0)})

    def _estimate_total_s(self):
        from .check import dry_run_operation
        total = 0.0
        for i in range(len(self.ops)):
            try:
                result = dry_run_operation(self.ctx, self.state, i)
            except TwinError:
                return None
            total += float(result["cycle_time_s"])
        return total * self.parts

    # ================================================================== operations
    def _setup_operation(self, first: bool) -> None:
        """The objects of the operation about to run. A new part starts from a fresh blank. A one-operation
        batch continues on the same control and machine: the operator loads the next blank and presses cycle
        start (the control keeps its offsets, variables and tool-life counters, the axes stay where the program
        left them, motion's timeline goes on). Otherwise the control loads the operation's program (commons and
        part counters carried) and motion is set up for the operation's fixture and stock poses."""
        ctx, state, d = self.ctx, self.state, self.domains
        op_index = int(state.clock.op_index)
        if not first:
            if op_index == 0:
                self._new_part_stock()
            if len(self.ops) == 1:
                self._wait_for_cycle_start()
                for p in range(len(state.controller.paths)):
                    d["controller"].cycle_start(ctx, state, p, int(state.clock.tick))
                self._motion_meta()
            else:
                d["controller"].start_operation(ctx, state, op_index)
                previous = state.motion
                self.motion = self._new_motion(op_index)
                self.motion.init(ctx, state)
                self._continue_axes(previous)
                self._motion_meta()
                ctx.timeline = self.motion.timeline
                if self.timeline is not None:
                    self.tl_base = len(self.timeline)
                self.tl_first = 0
        self._new_hot_loop()
        self.v0 = _volumes(state.removal)
        self.booked = [0.0 for _ in state.removal.workpieces]
        state.clock.op_start_tick = int(state.clock.tick)
        self.op_active = True
        self.ops_started += 1

    def _wait_for_cycle_start(self) -> None:
        """The next part of a one-operation batch: the time from the program's end (M30, inside the last macro
        step) to the cycle start at this macro boundary is the machine waiting for the operator. Motion's
        timeline books waits into the block that follows them (docs/models/motion.md "Timeline and ledger"): the
        wait is handed to it as ``program_stop`` from the last booked tick, so the records of the path stay
        contiguous over the parts (the time balance)."""
        ms = self.state.motion
        now = int(self.state.clock.tick)
        ends = (getattr(ms, "ledger", None) or {}).get("end") or []
        for p, end in enumerate(ends):
            if end is not None and int(end) < now and ms.idle_from[p] < 0:
                ms.idle_from[p] = int(end)
                ms.meta.setdefault("idle_cat", {})[str(p)] = "program_stop"

    def _continue_axes(self, previous) -> None:
        """A new operation's machine starts where the last program left it (the operator turns or changes the
        part; the axes do not move): the new motion's drives at rest at the axis positions the previous operation's
        motion ended with (``initial_servo_state`` at those positions: velocities zero, gravity preloads held), its
        planners starting there. The spindles start at rest as before (minutes pass between operations; a residual
        coast-down speed of 1e-15 rad/s would only make tooth periods of 1e22 ticks). Without this the axes would
        jump home in one tick, which holding reads as a table acceleration of 1e5 m/s2 (the part slips and comes
        free) and dynamics as a base excitation."""
        from ..motion.planner import PathPlanner
        from ..motion.servo import initial_servo_state
        if previous is None:
            return
        ms, motion = self.state.motion, self.motion
        old = getattr(previous, "servo", None)
        if old is None or np.shape(old.load) != np.shape(ms.servo.load):
            return
        q = np.asarray(old.load, float).copy()
        rest = initial_servo_state(motion.servo, q)
        for name, target in zip(ms.servo._fields, ms.servo):
            np.copyto(target, np.asarray(getattr(rest, name)).astype(target.dtype, copy=False))
        motion.planners = [PathPlanner(motion.env, p, q) for p in range(len(motion.model.paths))]
        ms.planners = motion.planners

    def start_operation(self) -> None:
        ctx, state, d = self.ctx, self.state, self.domains
        op_index, part = int(state.clock.op_index), int(state.clock.part_index)
        first = self.ops_started == 0
        if first:
            self.tl_base = len(self.timeline)
            self.tl_first = 0
        forwarded = [list(f) for f in self.feeder.forwarded] if (self.feeder is not None and len(self.ops) == 1) \
            else None
        self._setup_operation(first)
        if forwarded is not None and len(forwarded) == len(self.feeder.forwarded):
            self.feeder.forwarded = forwarded
        self.v0 = _volumes(state.removal)
        if op_index == 0:
            self.part_mass_start = float(state.ledger.removed_mass_kg)
            self.part_start_tick = int(state.clock.tick)
        self.trace("loop:operation_start", {"op_index": float(op_index), "part": float(part)})
        if part == 0 and self.options.get("sanity", True) is not False:
            self._sanity(op_index)
        self.progress(op=self.op_name(), part=part, phase="run", message=f"{self.op_name()} part {part + 1}")
        if not first:
            self.checkpoint(op_end=True, ended=self.last_ended)

    def _new_hot_loop(self) -> None:
        from .drive import Feeder
        from .hotloop import HotLoop
        d = self.domains
        self.hot = HotLoop(self.ctx, self.state, motion=self.motion, removal=d["removal"], cutting=d["cutting"],
                           tooling=d["tooling"], dynamics=d["dynamics"], use_dynamics=True, recorder=self.recorder)
        self.feeder = Feeder(self.ctx, self.state, d["controller"], self.motion)

    def _new_part_stock(self) -> None:
        """The next part of a batch starts from a fresh blank (removal re-initialised from the project stock; the
        field version continues past the last part's so every domain that tracks it sees the change)."""
        from ..removal import abi as RA
        ctx, state = self.ctx, self.state
        old = [int(wp.field.info[RA.FI_VERSION]) for wp in state.removal.workpieces]
        self.domains["removal"].init(ctx, state)
        for wp, version in zip(state.removal.workpieces, old):
            wp.field.info[RA.FI_VERSION] = version + 1
        self.trace("loop:new_blank", {"part": float(state.clock.part_index)})

    def _sanity(self, op_index: int) -> None:
        from ..sanity import check_operation
        state = self.state
        before = len(state.findings)
        report = check_operation(self.project, op_index, trace=self.ctx.trace, project_dir=self.project.dir)
        report.add_to(state.findings)
        start = int(state.clock.tick)
        for finding in state.findings.items[before:]:
            if finding.tick == 0 and start > 0:
                finding.tick = start
                finding.time_s = start * self.ctx.tick_s
            if not finding.op:
                finding.op = self.op_name()

    def op_finished(self) -> bool:
        return bool(self.feeder.finished())

    def end_operation(self) -> None:
        """Every domain's end_operation (loop_wiring.END_ORDER: cool-down, unclamp, spring-back, burr survival,
        inspection, transfer), the part files of the operation, then the next operation or part."""
        from .loop_wiring import END_ORDER
        ctx, state, d = self.ctx, self.state, self.domains
        alarms = self._alarms()
        unfinished = self.truncated is not None or bool(alarms)
        for name in END_ORDER:
            if name == "structure" and not unfinished:
                self.results.capture_as_cut(state)
            if name == "inspect":
                if unfinished:
                    # an operation cut short (options.max_sim_s, or a control alarm): the unfinished part is not
                    # inspected against the CAD and gets no part files (meshing a fine field: millions of triangles)
                    self.notes.add("an operation ended unfinished (options.max_sim_s or an alarm): its part was not "
                                   "inspected and no part files were written")
                    continue
                self._inspect_end()
                continue
            domain = self.motion if name == "motion" else d[name]
            domain.end_operation(ctx, state)
        self._sync_timeline(final=True)
        self.timeline.write()
        self._write_findings()
        op_index, part = int(state.clock.op_index), int(state.clock.part_index)
        if op_index == len(self.ops) - 1:
            self.part_mass[part] = float(state.ledger.removed_mass_kg) - self.part_mass_start
        self.trace("loop:operation_end", {"op_index": float(op_index), "part": float(part)})
        self.last_ended = (self.op_name(), op_index)
        self.op_active = False
        if alarms and self.stopped is None:
            # the machine stopped in alarm: nothing runs on until someone resets it (no next operation or part)
            self.stopped = {"at_s": int(state.clock.tick) * ctx.tick_s, "reason": "alarm", "op": self.op_name(),
                            "op_index": op_index, "part": part, "alarms": alarms}
            self.trace("loop:stopped_by_alarm", {"paths": float(len(alarms)), "part": float(part)})
        if self.truncated is not None or self.stopped is not None:
            self.done = True
        elif op_index + 1 < len(self.ops):
            state.clock.op_index = op_index + 1
        elif part + 1 < self.parts:
            state.clock.op_index = 0
            state.clock.part_index = part + 1
        else:
            self.done = True
        if self.done:
            self.checkpoint(op_end=True, ended=self.last_ended)

    def _alarms(self) -> list:
        """The paths whose control stopped with an alarm: [{path, alarm}]."""
        ctrl = self.domains["controller"].controller(self.ctx, self.state)
        out = []
        for p, path in enumerate(ctrl.state.paths):
            alarm = path.alarm
            if alarm is not None:
                text = f"{alarm.get('code', '')} {alarm.get('text', '')}".strip() if isinstance(alarm, dict) \
                    else str(alarm)
                out.append({"path": p, "alarm": text})
        return out

    def _inspect_end(self) -> None:
        ctx, state = self.ctx, self.state
        tick = int(state.clock.tick)
        if self.inspect is not None:
            before = len(self.inspect.state.files)
            self.inspect.end_operation(ctx, state)
            files = list(self.inspect.state.files[before:])
        else:
            files = []
            for w in self.results.workpieces(int(state.clock.op_index)):
                part = self.results.result_part(ctx, state, int(state.clock.op_index), w)
                if part is not None:
                    files.extend(part.write_files(self.run_dir))
        for path in files:
            relative = Path(path).relative_to(self.run_dir).as_posix() if Path(path).is_absolute() else str(path)
            if relative not in self.part_files:
                self.part_files.append(relative)
            self.rd.register_output(relative, tick)

    # ================================================================== the macro step
    def _apply_resolved_strides(self) -> None:
        """physics.resolved (``twin check --write``): the recorded L0 stride of the tool each path holds."""
        resolved = getattr(self.project.physics, "resolved", None)
        ops = getattr(resolved, "operations", None) if resolved is not None else None
        if not ops:
            return
        entries = ops.get(self.op_name()) if hasattr(ops, "get") else None
        frames = list(getattr(self.state.bus_prev.motion, "paths", []) or [])
        for p in range(len(frames)):
            slot = int(frames[p].tool_slot)
            number = self.state.tools.slots[slot].number if 0 <= slot < len(self.state.tools.slots) else None
            l0 = None
            for entry in entries or []:
                if int(getattr(entry, "tool", -1)) == number and int(getattr(entry, "path", 1) or 1) == p + 1:
                    l0 = int(getattr(entry, "l0_ticks"))
            if l0 is not None and l0 > 0 and self.ctx.ipo_ticks % l0 == 0:
                self.hot.l0_override[p] = l0
            else:
                self.hot.l0_override.pop(p, None)

    def step(self) -> None:
        from ..chips.abi import EMISSION_WIDTH
        from .loop_outputs import emission_buffer
        from .loop_wiring import SLOW_ORDER
        ctx, state, d = self.ctx, self.state, self.domains
        t0 = time.process_time()
        state.bus_next = Bus()
        self._apply_resolved_strides()
        head0 = int(self.recorder.ring.head[0])
        friction0 = float(state.ledger.friction_J)
        rep = self.hot.run_step(self.feeder)
        friction_by_cutting = float(state.ledger.friction_J) - friction0
        self._restamp_ring(head0)
        self.l0_seen.setdefault(self.op_name(), set()).update(int(v) for v in rep.l0_ticks.values())
        t1 = time.process_time()
        prev, nxt = state.bus_prev, state.bus_next
        by_domain = self.cpu["l2_by_domain_s"]
        d["chips"].absorb(ctx, state, emission_buffer(nxt, EMISSION_WIDTH), nxt)
        ta = time.process_time()
        by_domain["chips"] = by_domain.get("chips", 0.0) + (ta - t1)
        for name in SLOW_ORDER:
            domain = self.motion if name == "motion" else d[name]
            domain.macro_step(ctx, state, prev, nxt)
            tb = time.process_time()
            by_domain[name] = by_domain.get(name, 0.0) + (tb - ta)
            ta = tb
        self._coupling_monitor(prev, nxt, rep.tick1)
        self._book_friction(nxt, friction_by_cutting)
        values = {}
        if self.ledger is not None:
            values = self.ledger.check(ctx, state, rep.tick1)
        self._exact_volume(nxt, rep.tick1)
        self._identity(nxt)
        t2 = time.process_time()
        if not self.replay:
            self._record(rep, values)
            self.frames.append_motion(nxt.motion)
            self._sync_timeline()
            self._write_findings()
        else:
            self.recorder.end_step(rep.tick0, rep.tick1, _omega_max(nxt), None)
        state.clock.tick = rep.tick1
        state.clock.macro += 1
        state.swap_bus()
        if not self.replay and self.keyframes.due(rep.tick1):
            from .loop_outputs import write_keyframe
            write_keyframe(self, rep.tick1)
        t3 = time.process_time()
        self.cpu["hot_s"] += t1 - t0
        self.cpu["l2_s"] += t2 - t1
        self.cpu["record_s"] += t3 - t2
        self.cpu["steps"] += 1
        # the split the cost model of twin check is calibrated on (db/models/loop.yaml): steps with a tool in cut
        in_cut = any(float(getattr(c, "in_cut_s", 0.0) or 0.0) > 0.0 for c in nxt.cutting.contacts)
        key = "in_cut" if in_cut else "idle"
        split = self.cpu["split"].setdefault(key, {"steps": 0, "hot_s": 0.0, "l2_s": 0.0})
        split["steps"] += 1
        split["hot_s"] += t1 - t0
        split["l2_s"] += t2 - t1

    def _restamp_ring(self, head0: int) -> None:
        """The hot loop stamps each fast (L0) ring row with the end tick of its L0 step, so the last row of a macro
        step carries the step's end tick, which belongs to the next step for the recorder (rows of a step lie in
        [tick0, tick1)). The loop moves every row of the step to the last base tick inside its L0 step (end - 1):
        rows stay strictly rising, each in the L1 tick and the macro step it was computed in."""
        ring = self.recorder.ring
        n = int(ring.tick.shape[0])
        head = int(ring.head[0])
        for h in range(max(head0, head - n), head):
            ring.tick[h % n] -= 1

    def _record(self, rep, ledger_values: dict) -> None:
        """The step's recorder rows: tick-tier and macro-tier values of every domain and the ledger, then the
        step's close (fast windows cut from the ring, tick envelopes)."""
        from .channels import Tier
        ctx, state, d = self.ctx, self.state, self.domains
        nxt = state.bus_next
        values = {}
        values.update(self.motion.channel_values(self.motion.out, nxt.motion.paths))
        values.update(d["controller"].last_channels)
        values.update(d["tooling"].channel_values(ctx, state, nxt))
        values.update(d["removal"].channel_values(ctx, state, nxt))
        values.update(self._cutting_values(nxt))
        values.update(d["dynamics"].channel_values(ctx, state))
        values.update(d["chips"].macro_values(state))
        values.update(d["thermal"].macro_values(state))
        values.update(d["coolant"].macro_values(state, nxt.coolant))
        values.update(d["wear"].macro_values(state))
        values.update(d["holding"].macro_values(state))
        values.update(d["structure"].macro_values(state))
        values.update(d["surface"].channel_values(ctx, state, nxt))
        values.update(ledger_values)
        registry = ctx.channels
        tick_values, macro_values = {}, {}
        for name, value in values.items():
            if name not in registry:
                self.unregistered.add(name)
                continue
            tier = registry.get(name).tier
            if tier == Tier.TICK:
                tick_values[name] = value
            elif tier == Tier.MACRO:
                macro_values[name] = value
        # one row per L1 tick at the tick's start (the recorder's convention: a row covers [t, t + ipo), the fast
        # rows of the tick are its envelope); motion's PathFrame ticks are the ticks' ends
        n = int(ctx.macro_ipo)
        ticks = rep.tick0 + ctx.ipo_ticks * np.arange(n, dtype=np.int64)
        self.recorder.record_tick(ticks, tick_values)
        self.recorder.record_macro(rep.tick1, macro_values)
        self.recorder.end_step(rep.tick0, rep.tick1, _omega_max(nxt), state.findings)

    def _cutting_values(self, nxt) -> dict:
        """cutting's channel values of the step. Its macro_values reads ``CuttingContact.h_max_m`` as a number
        while the bus field is the per-element array [teeth, elements]: when that fails the loop hands it the
        contacts with the array's peak in place of the array (noted once in run.json ``summary.notes``)."""
        ctx, state, cutting = self.ctx, self.state, self.domains["cutting"]
        try:
            return cutting.channel_values(ctx, state, nxt)
        except (ValueError, TypeError):
            from ..cutting.channels import macro_values, tick_values
            self.notes.add("cutting.channel_values cannot take the per-element h_max_m array: the loop records its "
                           "peak (cutting.h_max_m)")
            contacts = [None] * len(getattr(nxt.motion, "paths", []) or [])
            for rec in nxt.cutting.contacts:
                if 0 <= rec.contact < len(contacts):
                    contacts[rec.contact] = _PeakView(rec)
            loads = []
            for rec in contacts:
                sp = cutting._spindle_of(state, rec.tool_slot) if rec is not None else None
                loads.append(state.cutting.spindle_theta.get(sp.id, 0.0) if sp is not None else np.nan)
            out = tick_values(contacts, max(int(ctx.macro_ipo), 1))
            out.update(macro_values(contacts, ctx.macro_s, loads))
            return out

    def _sync_timeline(self, final: bool = False) -> None:
        """Motion's records of the operation (local indices) into the run's timeline (global indices); records
        that are closed and final are not touched again."""
        records = self.motion.timeline.records
        base = self.tl_base
        for rec in records[self.tl_first:]:
            entry = rec.to_dict()
            entry["index"] = base + int(rec.index)
            entry["final"] = bool(final or not rec.pending_feed)
            self.timeline.upsert(entry)
        while self.tl_first < len(records) and records[self.tl_first].tick1 >= 0 and \
                not records[self.tl_first].pending_feed:
            self.tl_first += 1

    def _write_findings(self) -> None:
        n = len(self.state.findings)
        if n != self.findings_written:
            self.rd.write_findings(self.state.findings)
            self.findings_written = n

    # ------------------------------------------------------------------ checks of the step
    def _exact_volume(self, nxt, tick: int) -> None:
        """The removal records of the operation sum to the field's own volume change (integer quanta: the polar
        field of a turned workpiece, the tri-dexel z family otherwise), per workpiece, to 1e-9."""
        for c in nxt.removal.contacts:
            w = int(c.workpiece)
            if 0 <= w < len(self.booked):
                self.booked[w] += float(c.removed_volume_m3)
        now = _volumes(self.state.removal)
        for w, (a, b, booked) in enumerate(zip(self.v0, now, self.booked)):
            change = (a[1] - b[1]) if (a[1] is not None and b[1] is not None) else (a[0] - b[0])
            scale = max(abs(change), abs(booked), 1e-300)
            if abs(change - booked) > 1e-9 * scale and abs(change - booked) > 1e-18:
                message = (f"workpiece {w}: removal booked {booked!r} m3 in this operation, the field changed by "
                           f"{change!r} m3")
                self.ctx.finding(self.state, LOOP_DOMAIN, Category.MASS_BALANCE, Severity.FATAL, message, int(tick),
                                 quantities={"booked_m3": booked, "field_change_m3": change})
                self._write_findings()
                raise ConservationError("removed volume", change, booked, 1e-9, tick * self.ctx.tick_s)

    def _book_friction(self, nxt, booked: float = 0.0) -> None:
        """The work the step's chip recut impulses and chip-packing loads took from the tool (cutting's
        CuttingContact.recut_J and packing_J, terms of its own energy identity) into SimState.ledger.friction_J,
        the ledger's friction term (architecture 12: packing, recut and slip friction), unless cutting booked them
        itself while the hot loop absorbed the step (``booked``: the friction term's change over the hot loop;
        cutting's ``_ledger`` does so since 2026-11-09, earlier builds wrote no SimState.ledger field for them)."""
        work = sum(float(getattr(c, "recut_J", 0.0) or 0.0) + float(getattr(c, "packing_J", 0.0) or 0.0)
                   for c in nxt.cutting.contacts)
        if work and abs(booked) <= 1e-12 * abs(work):
            self.state.ledger.friction_J += work

    def _identity(self, nxt) -> None:
        """cutting's own energy identity of the step (shaft + feed = heat by zone + ploughing + recut + packing +
        work on the deflection rates), reported in run.json (the contract check is the ledger's)."""
        e_in = e_out = mag = 0.0
        for c in nxt.cutting.contacts:
            terms_in = (float(c.shaft_work_J), float(c.feed_work_J))
            terms_out = (float(c.heat_shear_J), float(c.heat_rake_J), float(c.heat_flank_J), float(c.ploughing_J),
                         float(getattr(c, "recut_J", 0.0)), float(getattr(c, "packing_J", 0.0)),
                         float(getattr(c, "deflection_work_J", 0.0)))
            e_in += sum(terms_in)
            e_out += sum(terms_out)
            mag += sum(abs(v) for v in terms_in + terms_out)
        self.identity["in_J"] += e_in
        self.identity["out_J"] += e_out
        self.identity["abs_J"] += mag

    def _coupling_monitor(self, prev, nxt, tick: int) -> None:
        """Architecture 5.3: the frozen quantities handed across the one-step stagger (tool rake temperatures,
        coolant friction factors, holder pull-out, spindle growth) go through the CouplingMonitor; a staggered
        instability it flags becomes a warning with its history in a ``loop:coupling_lag`` trace event."""
        quantities = {}
        for slot, heat in sorted((getattr(nxt.thermal, "tools", {}) or {}).items()):
            # the rake temperature the coefficient rows read: the in-cut-time weighted mean over the elements that cut
            # (cutting's _row_state); the largest element switched between elements from step to step and flagged
            # their difference as an alternation. A step without cut adds nothing
            value = _cut_weighted(heat.rake_temp_mean_K, nxt, int(slot))
            if value is not None:
                quantities[f"thermal.tools[{slot}].rake_temp_mean_K"] = value
        for contact, value in sorted((getattr(nxt.coolant, "friction_factor", {}) or {}).items()):
            quantities[f"coolant.friction_factor[{contact}]"] = _scalar(value)
        for slot, hold in sorted((getattr(nxt.holding, "tools", {}) or {}).items()):
            quantities[f"holding.tools[{slot}].pullout_m"] = _scalar(hold.pullout_m)
        growth = getattr(nxt.thermal, "spindle_growth_M_m", None)
        items = sorted(growth.items()) if isinstance(growth, dict) else ([("", growth)] if growth is not None else [])
        for sid, vector in items:
            vector = np.asarray(vector, float).reshape(-1)
            if vector.size:
                quantities[f"thermal.spindle_growth_M_m[{sid}]"] = float(np.linalg.norm(vector))
        for name, value, change, previous, count in self.monitor.update(quantities):
            parent = self.trace("loop:coupling_lag", {"value": value, "change": change, "previous_change": previous,
                                                      "alternations": float(count)})
            self.ctx.finding(self.state, LOOP_DOMAIN, Category.SANITY, Severity.WARNING,
                             f"coupling lag: {name} alternates in sign from one macro step to the next "
                             f"({count} times, last change {change:.4g}, before {previous:.4g}): the one-step "
                             "stagger between the hot loop and the slow domains is unstable here; halve "
                             "physics.macro_ipo (the coupling interval) to check the result",
                             int(tick), quantities={"value": value, "change": change}, parents=[parent])

    # ================================================================== checkpoints
    def loop_state(self) -> dict:
        hot_arrays, hot_meta = self.hot.to_arrays() if self.hot is not None else ({}, {})
        return {"op_active": self.op_active, "done": self.done, "stopped": self.stopped,
                "ops_started": self.ops_started,
                "tl_base": self.tl_base, "tl_first": self.tl_first, "v0": [list(v) for v in self.v0],
                "booked": list(self.booked), "part_mass": {str(k): v for k, v in self.part_mass.items()},
                "part_mass_start": self.part_mass_start, "part_start_tick": self.part_start_tick,
                "est_total_s": self.est_total_s,
                "monitor": self.monitor.state(), "identity": dict(self.identity),
                "runs": list(self.feeder.runs) if self.feeder is not None else [],
                "ended": list(self.feeder.ended) if self.feeder is not None else [],
                "forwarded": [list(f) for f in self.feeder.forwarded] if self.feeder is not None else [],
                "hot_arrays": hot_arrays, "hot_meta": hot_meta, "seen": self.field.seen_state(),
                "part_files": list(self.part_files), "carry": list(getattr(self.domains["controller"], "_carry", [])),
                "l0_seen": {op: sorted(int(v) for v in seen) for op, seen in sorted(self.l0_seen.items())},
                "last_ended": list(self.last_ended),
                "dynamics_modes_event": {str(k): int(v) for k, v in
                                         sorted(getattr(self.domains["dynamics"], "_modes_event", {}).items())},
                "notes": sorted(self.notes), "unregistered": sorted(self.unregistered)}

    def checkpoint(self, op_end: bool = False, ended: tuple | None = None) -> dict | None:
        """Commit a checkpoint at the current macro boundary. An operation-end checkpoint (``op_end``, kept by the
        pruning) names the operation that ended (``ended`` = (name, index)): it is taken when the next operation
        is set up, or after the last operation."""
        if self.replay:
            return None
        state = self.state
        self.domains["cutting"].model_checkpoint(state)
        self._sync_timeline()
        self._write_findings()
        op, op_index = ended if ended is not None else (self.op_name(), int(state.clock.op_index))
        marker = self.checkpointer.write(state, tick=int(state.clock.tick), op=op, op_index=int(op_index),
                                         op_end=op_end, participants=self.participants(), trace=self.ctx.trace)
        self.progress(checkpoint={"tick": int(marker["tick"]), "file": f"checkpoints/ck_{int(marker['tick'])}.json"})
        return marker

    def resume(self, marker) -> None:
        """Continue a run from a committed checkpoint: everything is built as for a fresh run (on a scratch state
        whose trace goes nowhere), then the checkpoint's state and every participant's are restored (outputs
        after the cut truncated) and the operation's objects are bound to them."""
        from .checkpoint import CheckpointError  # noqa: F401  (raised by resume)
        from .rundir import RunDirectory
        from .trace import TraceLog
        marker = Path(marker)
        self.rd = RunDirectory(self.run_dir, own_status=False, tick_s=self.ctx.tick_s)
        self.ctx.trace = TraceLog()
        self.progress(phase="init", message="resuming from " + marker.name)
        self._init_domains()
        self._writers(fresh=False)
        self.op_active = False
        restored = self.checkpointer.resume(marker, self.participants(), run_directory=self.rd)
        self.state = restored.state
        self.ctx.trace = restored.trace
        saved = self.saved_loop
        state = self.state
        self.op_active = bool(saved["op_active"])
        self.done = bool(saved["done"])
        self.stopped = saved.get("stopped")
        self.ops_started = int(saved["ops_started"])
        self.tl_base, self.tl_first = int(saved["tl_base"]), int(saved["tl_first"])
        self.v0 = [tuple(v) for v in saved["v0"]]
        self.booked = [float(v) for v in saved["booked"]]
        self.part_mass = {int(k): float(v) for k, v in saved["part_mass"].items()}
        self.part_mass_start = float(saved["part_mass_start"])
        self.part_start_tick = int(saved.get("part_start_tick", 0))
        self.est_total_s = saved["est_total_s"]
        self.monitor.restore(saved["monitor"])
        self.identity = dict(saved["identity"])
        self.part_files = list(saved.get("part_files", []))
        self.l0_seen = {str(op): {int(v) for v in seen} for op, seen in (saved.get("l0_seen") or {}).items()}
        self.last_ended = tuple(saved.get("last_ended", ("", -1)))
        self.notes = set(saved.get("notes", []))
        self.unregistered = set(saved.get("unregistered", []))
        self.field.restore_seen(saved.get("seen"))
        # dynamics keeps the trace id of each path's last modes event on its domain object (the parent of its
        # later events): restored so the trace continues as in an uninterrupted run
        self.domains["dynamics"]._modes_event = {int(k): int(v) for k, v in
                                                 (saved.get("dynamics_modes_event") or {}).items()}
        controller = self.domains["controller"]
        controller._carry = [dict(c) for c in saved.get("carry", [])]
        controller._setup = None
        controller.controller(self.ctx, state)
        self.motion = self._new_motion(int(state.clock.op_index))
        self.motion.attach(self.ctx, state)
        self.ctx.timeline = self.motion.timeline
        self._new_hot_loop()
        self.hot.from_arrays(saved["hot_arrays"], saved["hot_meta"])
        self.feeder.runs = [int(v) for v in saved["runs"]] or self.feeder.runs
        self.feeder.ended = [bool(v) for v in saved["ended"]] or self.feeder.ended
        self.feeder.forwarded = [[int(n) for n in f] for f in saved.get("forwarded", [])] or self.feeder.forwarded
        self.domains["cutting"].model_restore(state)
        self.findings_written = len(state.findings)
        # nothing is written to the trace here: a resumed run's result set equals an uninterrupted run's
        self.progress(phase="run", message=f"resumed at {int(state.clock.tick) * self.ctx.tick_s:.3f} s",
                      sim_time_s=int(state.clock.tick) * self.ctx.tick_s)

    # ================================================================== the run
    def execute(self) -> dict:
        marker = self.options.get("resume_from")
        try:
            if marker:
                self.resume(marker)
            else:
                self.fresh()
            limit = self.options.get("max_sim_s")
            limit_ticks = int(round(float(limit) / self.ctx.tick_s)) if limit is not None else None
            while not self.done:
                if not self.op_active:
                    self.start_operation()
                    continue
                self.step()
                tick = int(self.state.clock.tick)
                self.progress(sim_time_s=tick * self.ctx.tick_s, est_total_s=self.est_total_s, op=self.op_name(),
                              part=int(self.state.clock.part_index))
                if limit_ticks is not None and tick >= limit_ticks and not self.op_finished():
                    self.truncated = {"at_s": tick * self.ctx.tick_s, "reason": f"options.max_sim_s = {limit}",
                                      "op": self.op_name(), "op_index": int(self.state.clock.op_index),
                                      "part": int(self.state.clock.part_index)}
                    self.trace("loop:truncated", {"at_s": tick * self.ctx.tick_s})
                    self.end_operation()
                    break
                if self.op_finished():
                    self.end_operation()
                    continue
                if self.checkpointer.due(tick):
                    self.checkpoint()
            return self.finish()
        except BaseException:
            self._evidence()
            raise
        finally:
            self._close()

    def _evidence(self) -> None:
        """A failed run keeps what explains the failure: findings.json, timeline.json and the closed series."""
        try:
            if self.rd is not None:
                self._write_findings()
            if self.timeline is not None and self.motion is not None:
                self._sync_timeline(final=True)
                self.timeline.write()
            if self.recorder is not None:
                self.recorder.close_parts()
        except (OSError, ValueError, TwinError):
            pass

    def _close(self) -> None:
        if self.frames is not None:
            self.frames.close()
        trace = getattr(self.ctx, "trace", None)
        if trace is not None:
            trace.flush()

    # ================================================================== the end of the run
    def finish(self) -> dict:
        from .loop_outputs import write_keyframe
        from .rundir import verdict, write_json_atomic
        ctx, state, d = self.ctx, self.state, self.domains
        tick = int(state.clock.tick)
        self._final_mass_check(tick)
        self.progress(phase="finish", message="end of run: results")
        write_keyframe(self, tick, full=True)
        self.recorder.finish()
        self.frames.close()
        documents = {}
        if self.inspect is not None:
            documents["inspection"] = self.inspect.finalize(ctx, state)
        else:
            documents["inspection"] = _no_cad_inspection(self)
            write_json_atomic(self.run_dir / "inspection.json", documents["inspection"])
        documents["surface"] = d["surface"].finalize(ctx, state)
        tools = d["wear"].finalize(ctx, state)
        write_json_atomic(self.run_dir / "tools.json", tools)
        domains_doc = {}
        for name in ("dynamics", "thermal", "chips", "coolant", "holding", "structure", "removal", "cutting",
                     "controller", "tooling"):
            domains_doc[name] = d[name].finalize(ctx, state)
        domains_doc["motion"] = self.motion.finalize(ctx, state)
        self._sync_timeline(final=True)
        self.timeline.write()
        self._write_findings()
        production = self._production()
        for relative in ["inspection.json", "surface.json", "tools.json", "production.json"] + \
                sorted(p.relative_to(self.run_dir).as_posix() for p in (self.run_dir / "sld").glob("*.json")):
            if (self.run_dir / relative).is_file():
                self.rd.register_output(relative, tick)
        summary = self._summary(domains_doc, production)
        document = self._run_document(verdict(state.findings), summary, domains_doc)
        self.rd.write_run_json(document)
        # process CPU seconds by level (volatile: in the returned result, never in run.json)
        result = {"run_dir": str(self.run_dir), "run_id": document["run_id"], "verdict": document["verdict"],
                  "complete": document["complete"], "summary": {k: summary[k] for k in
                                                                 ("sim_time_s", "cycle_time_s", "removed_mass_kg",
                                                                  "findings") if k in summary},
                  "cpu_s": dict(self.cpu)}
        outputs = self.options.get("outputs", True)
        if outputs is not False:
            from .outputs import finalize_outputs
            self.progress(phase="outputs", message="report, pictures, video and sound")
            made = finalize_outputs(self.run_dir, {"outputs": outputs} if isinstance(outputs, dict) else None,
                                    self.progress_cb)
            result["outputs"] = {"failed": made["failed"],
                                 "generators": {k: v.get("status") for k, v in made["generators"].items()}}
        self.progress(phase="done", sim_time_s=tick * ctx.tick_s, est_total_s=self.est_total_s)
        return result

    def _final_mass_check(self, tick: int) -> None:
        """After the last operation every chip is booked (chips book their last window at each operation end):
        the removed and detached mass of the whole run equals the six chip inventories plus the displaced and
        detached mass, to physics.tol_mass."""
        from .ledger import MASS_CHIP_FIELDS
        if self.ledger is None:
            return
        cum = self.ledger.removal_cum
        chips = sum(float(getattr(self.state.ledger, name)) for name in MASS_CHIP_FIELDS)
        mass_in = cum["removed_kg"] + cum["loose_rho_kg"]
        mass_out = chips + cum["displaced_kg"] + cum["loose_kg"]
        scale = max(abs(mass_in), abs(mass_out))
        rel = abs(mass_in - mass_out) / scale if scale > 0.0 else 0.0
        self.final_mass = {"mass_in_kg": mass_in, "mass_out_kg": mass_out, "residual_rel": rel,
                           "tolerance_rel": self.ledger.tol_mass, "ok": rel <= self.ledger.tol_mass}
        if rel > self.ledger.tol_mass:
            self.ctx.finding(self.state, "ledger", Category.MASS_BALANCE, Severity.FATAL,
                             f"Mass balance of the whole run violated: removed {mass_in:.9g} kg, accounted "
                             f"{mass_out:.9g} kg (relative {rel:.3g}, tolerance {self.ledger.tol_mass:g})", tick,
                             quantities={"mass_in_kg": mass_in, "mass_out_kg": mass_out, "residual_rel": rel})
            self._write_findings()
            raise ConservationError("mass (end of run)", mass_in, mass_out, self.ledger.tol_mass,
                                    tick * self.ctx.tick_s)

    def _production(self) -> dict | None:
        from ..production.rundir import production_from_run_dir
        masses = [self.part_mass.get(k) for k in range(self.parts)]
        masses = [m for m in masses if m is not None]
        try:
            return production_from_run_dir(self.run_dir, project=self.project, removed_mass_kg=masses or None,
                                           feed_work_J=float(self.state.ledger.feed_work_J),
                                           parts_planned=self.parts, write=True)
        except InputError as error:
            self.trace("loop:production_unavailable", {})
            self.production_error = str(error)
            return None

    def _summary(self, domains_doc: dict, production) -> dict:
        from .rundir import finding_counts
        ctx, state = self.ctx, self.state
        led = state.ledger
        tick = int(state.clock.tick)
        spans: dict = {}
        for block in self.timeline.closed():
            part = int(block.get("part", 0))
            lo, hi = spans.get(part, (block["tick0"], block["tick1"]))
            spans[part] = (min(lo, block["tick0"]), max(hi, block["tick1"]))
        part_cycles = [{"part": k, "cycle_time_s": (hi - lo) * ctx.tick_s} for k, (lo, hi) in sorted(spans.items())]
        cycle = part_cycles[0]["cycle_time_s"] if part_cycles else 0.0
        identity = self.identity
        summary = {
            "sim_time_s": tick * ctx.tick_s, "cycle_time_s": cycle, "parts_cycle_time_s": part_cycles,
            "macro_steps": int(state.clock.macro),
            "operations": [str(op.name) for op in self.ops], "parts": self.parts,
            "parts_run": int(state.clock.part_index) + 1,
            "removed_volume_m3": float(led.removed_volume_m3), "removed_mass_kg": float(led.removed_mass_kg),
            "parts_removed_mass_kg": [{"part": k, "removed_mass_kg": v} for k, v in sorted(self.part_mass.items())],
            "feed_work_J": float(led.feed_work_J), "shaft_work_J": float(led.shaft_work_J),
            "energy_J": {name: float(getattr(led, name)) for name in
                         ("shaft_work_J", "feed_work_J", "heat_shear_J", "heat_rake_J", "heat_flank_J", "ploughing_J",
                          "modal_energy_change_J", "damping_J", "chip_kinetic_J", "friction_J", "unbalance_work_J",
                          "heat_to_chip_J", "heat_to_tool_J", "heat_to_work_J", "heat_to_coolant_J")},
            "chips_kg": {name: float(getattr(led, name)) for name in
                         ("chips_in_flight_kg", "chips_settled_kg", "chips_conveyed_kg", "chips_adhered_kg",
                          "chips_in_flutes_kg", "dust_kg")},
            "conservation": self.ledger.summary() if self.ledger is not None else {},
            "final_mass": getattr(self, "final_mass", None),
            "cutting_identity_rel": abs(identity["in_J"] - identity["out_J"]) / identity["abs_J"]
            if identity["abs_J"] > 0.0 else 0.0,
            "findings": finding_counts(state.findings),
            "parts_files": list(self.part_files),
            "unregistered_channels": sorted(self.unregistered),
            "notes": sorted(self.notes),
        }
        if self.truncated is not None:
            summary["truncated"] = dict(self.truncated)
        if self.stopped is not None:
            summary["stopped"] = dict(self.stopped)
        if production is not None:
            summary["production"] = {k: production.get(k) for k in ("parts_simulated", "cycle_time_s", "per_part_s",
                                                                    "energy_J", "removed_mass_kg") if k in production}
        elif getattr(self, "production_error", None):
            summary["production"] = {"error": self.production_error}
        return summary

    def _run_document(self, verdict_doc: dict, summary: dict, domains_doc: dict) -> dict:
        from .rundir import jsonable, run_document
        ctx = self.ctx
        # per operation the L0 stride its steps used (a list when tools or speeds took several), the form the
        # report, the UI and the synthetic run fixture read: {op: ticks}
        l0 = {}
        for op in [str(o.name) for o in self.ops]:
            seen = sorted(int(v) for v in self.l0_seen.get(op, ()))
            if seen:
                l0[op] = seen[0] if len(seen) == 1 else seen
        run_id = self.run_dir.name
        estimates = [{"file": e.file, "path": e.path, "value": jsonable(e.value), "note": e.note}
                     for e in _estimates(ctx, self.state)]
        return run_document(run_id=run_id, project=self.project, options=self.options,
                            clock={"tick_s": ctx.tick_s, "ipo_ticks": ctx.ipo_ticks, "macro_ipo": ctx.macro_ipo,
                                   "l0_ticks": l0}, verdict_=verdict_doc, summary=summary,
                            outputs=self.rd.outputs_index(), seed=ctx.seed, complete=self.truncated is None,
                            extra={"domains": jsonable(domains_doc), "estimates": estimates,
                                   "loop": {"lag_macro": int(getattr(self.project.physics, "lag_macro", 1) or 1),
                                            "lag_note": "the slow domains of a step run after its hot loop "
                                                        "(lag 1); lag_macro 2 is computed the same way"}})


def _estimates(ctx, state) -> list:
    """The database estimates (src: est, cnctwin.db.Database.estimates) of the files the project reads
    (``project.input_files``); in a series file (several objects) only those of the objects the run uses (the
    machine, the material, the coolant, the tools and holders of the tool table)."""
    project = ctx.project
    files = set((getattr(project, "input_files", {}) or {}).keys())
    used = {f"machine:{getattr(project.machine, 'id', '')}",
            f"material:{getattr(project.material, 'db_id', None) or getattr(project.material, 'id', '')}"}
    coolant = getattr(project, "coolant", None)
    fluid = getattr(coolant, "fluid", None) if coolant is not None else None
    if fluid is not None:
        used.add(f"coolant:{getattr(fluid, 'id', '')}")
    for slot in getattr(state.tools, "slots", ()):
        used.add(f"tool:{slot.tool_id}")
        used.add(f"holder:{slot.holder_id}")
    candidates = [e for e in ctx.db.estimates() if e.file in files]
    objects_per_file: dict = {}
    for e in candidates:
        objects_per_file.setdefault(e.file, set()).add(e.obj)
    return [e for e in candidates if len(objects_per_file[e.file]) <= 1 or e.obj in used]


def _no_cad_inspection(run) -> dict:
    return {"format": "cnctwin-inspection", "version": 1, "tolerances": [], "remaining": [], "gouges": [],
            "operations": [], "notes": ["the project has no CAD (cad.path): nothing to inspect against; the part "
                                        "meshes of every operation are written as part_<op>.*"],
            "files": list(run.part_files)}


def _volumes(rs) -> list[tuple]:
    from ..removal import polar as PO
    from ..removal import ledger as LG
    # the tri-dexel's ledger volume (docs/models/removal.md 1.1: per dual cell the family most normal to the
    # surface), the volume removal books per contact
    return [(float(LG.volume_m3(wp)), float(PO.volume_m3(wp.polar)) if wp.polar is not None else None)
            for wp in rs.workpieces]


class _PeakView:
    """A CuttingContact whose per-element ``h_max_m`` array reads as its peak (cutting's macro channel)."""

    def __init__(self, record):
        self._record = record

    def __getattr__(self, name):
        value = getattr(self._record, name)
        if name == "h_max_m":
            return _scalar(value) if np.size(value) else 0.0
        return value


def _cut_weighted(value, bus, slot: int):
    """The in-cut-time weighted mean of a per-element tool quantity ([n_teeth, n_elem]) over the elements of tool
    ``slot`` that cut in the step (cutting's contacts of ``bus``; weights summed over the teeth, as cutting's
    coefficient rows read it); None when the tool did not cut or the shapes do not match."""
    array = np.asarray(value, dtype=float)
    weight = np.zeros(array.shape)
    for c in getattr(bus.cutting, "contacts", []) or []:
        ic = np.asarray(getattr(c, "in_cut_time_s", np.zeros((0, 0))), float)
        if int(c.tool_slot) == slot and ic.shape == array.shape:
            weight += ic
    if weight.ndim == 2:
        weight = np.broadcast_to(weight.sum(axis=0, keepdims=True), array.shape)
    weight = np.where(np.isfinite(array), weight, 0.0)
    total = float(weight.sum())
    if not total > 0.0:
        return None
    return float(np.sum(weight * np.where(weight > 0.0, array, 0.0)) / total)


def _scalar(value) -> float:
    """The largest finite entry of a monitored quantity (a per-tooth array's peak), NaN when it has none."""
    array = np.asarray(value, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else math.nan


def _omega_max(bus) -> float:
    best = 0.0
    for frame in getattr(bus.motion, "paths", []) or []:
        omega = np.asarray(frame.spindle_omega_rad_s, float)
        if omega.size:
            best = max(best, float(np.max(np.abs(omega))))
    return best


# =============================================================================== entry points

def run_project(project_path, run_dir, options=None, progress_cb=None) -> dict:
    """Run a project into ``run_dir`` (module docstring); returns a JSON-able result (the API saves it as the
    job's result.json). ``options["replay"]`` = {tick0, tick1, checkpoint} replays a window instead."""
    options = dict(options or {})
    replay = options.get("replay")
    if replay:
        return _replay(project_path, run_dir, options, progress_cb)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    project_path = Path(project_path)
    if project_path.is_dir():
        project_path = project_path / "project.yaml"
    return Run(project_path, run_dir, options, progress_cb).execute()


def _replay(project_path, run_dir, options: dict, progress_cb) -> dict:
    """The fast rows of [tick0, tick1) of a finished run: the run is restored from the checkpoint (or started
    from the beginning) without touching its outputs, stepped to tick1 with every writer off except the
    recorder in replay mode, which writes series/fast/replay-<tick0>-<tick1>.parquet and its index line."""
    window = options["replay"]
    tick0, tick1 = int(window["tick0"]), int(window["tick1"])
    project_path = Path(project_path)
    if project_path.is_dir():
        project_path = project_path / "project.yaml"
    run = Run(project_path, run_dir, {k: v for k, v in options.items() if k not in ("replay", "resume_from")},
              progress_cb, replay=(tick0, tick1, window.get("checkpoint")))
    from .checkpoint import Checkpointer
    from .trace import TraceLog
    run.ctx.trace = TraceLog()
    run._init_domains()
    marker = window.get("checkpoint")
    run.ledger = None
    if marker:
        from .rundir import Timeline
        ck = Checkpointer(run.run_dir, project_hash=run.project.hash, tick_s=run.ctx.tick_s)
        scratch_timeline = Timeline(None, tick_s=run.ctx.tick_s)
        participants = {"loop": LoopParticipant(run), "timeline": scratch_timeline}
        restored = ck.load(marker, participants)
        run.state = restored.state
        saved = run.saved_loop
        run.op_active = bool(saved["op_active"])
        run.done = bool(saved["done"])
        run.ops_started = int(saved["ops_started"])
        run.v0 = [tuple(v) for v in saved["v0"]]
        run.booked = [float(v) for v in saved["booked"]]
        run.field.restore_seen(saved.get("seen"))
        run.timeline = scratch_timeline
        controller = run.domains["controller"]
        controller._carry = [dict(c) for c in saved.get("carry", [])]
        controller._setup = None
        controller.controller(run.ctx, run.state)
        run.motion = run._new_motion(int(run.state.clock.op_index))
        run.motion.attach(run.ctx, run.state)
        run.ctx.timeline = run.motion.timeline
        run._new_hot_loop()
        run.hot.from_arrays(saved["hot_arrays"], saved["hot_meta"])
        run.feeder.runs = [int(v) for v in saved["runs"]] or run.feeder.runs
        run.feeder.ended = [bool(v) for v in saved["ended"]] or run.feeder.ended
        run.feeder.forwarded = [[int(n) for n in f] for f in saved.get("forwarded", [])] or run.feeder.forwarded
        run.domains["dynamics"]._modes_event = {int(k): int(v) for k, v in
                                                (saved.get("dynamics_modes_event") or {}).items()}
        run.domains["cutting"].model_restore(run.state)
    else:
        from .rundir import Timeline
        run.timeline = Timeline(None, tick_s=run.ctx.tick_s)
    from .rundir import RunDirectory
    run.rd = RunDirectory(run.run_dir, own_status=False)
    run.rd.write_findings = lambda findings: None
    steps = 0
    while not run.done and int(run.state.clock.tick) < tick1:
        if not run.op_active:
            _replay_start_operation(run)
            continue
        run.step()
        steps += 1
        run.progress(sim_time_s=int(run.state.clock.tick) * run.ctx.tick_s, phase="replay")
        if run.op_finished():
            _replay_end_operation(run)
    entries = run.recorder.finish()
    return {"replay": [tick0, tick1], "checkpoint": marker, "steps": steps,
            "rows": int(sum(int(e.get("rows", 0)) for e in entries)), "parts": entries}


def _replay_start_operation(run) -> None:
    run._setup_operation(run.ops_started == 0)


def _replay_end_operation(run) -> None:
    from .loop_wiring import END_ORDER
    ctx, state, d = run.ctx, run.state, run.domains
    for name in END_ORDER:
        if name == "structure":
            run.results.capture_as_cut(state)
        if name == "inspect":
            if run.inspect is not None:
                run.inspect.end_operation(ctx, state)
            continue
        (run.motion if name == "motion" else d[name]).end_operation(ctx, state)
    op_index, part = int(state.clock.op_index), int(state.clock.part_index)
    run.op_active = False
    if op_index + 1 < len(run.ops):
        state.clock.op_index = op_index + 1
    elif part + 1 < run.parts:
        state.clock.op_index = 0
        state.clock.part_index = part + 1
    else:
        run.done = True


def replay_window(run_dir, tick0: int, tick1: int, options=None, progress_cb=None) -> dict:
    """``twin replay --window``: the fast rows of ``[tick0, tick1)`` of a finished run, replayed from its nearest
    committed checkpoint at or before tick0 (cnctwin.core.outputs.replay_window with :func:`run_project`)."""
    from .outputs import replay_window as replay
    return replay(run_dir, int(tick0), int(tick1), options, progress_cb, runner=run_project)


def check_project(request: dict, out_dir, progress_cb=None) -> dict:
    """``twin check`` (cnctwin.core.check.check_project)."""
    from .check import check_project as check
    return check(request, out_dir, progress_cb)


__all__ = ["run_project", "replay_window", "check_project", "Run", "LoopParticipant", "CouplingMonitor"]
