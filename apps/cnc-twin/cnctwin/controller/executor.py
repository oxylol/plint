"""The dialect-neutral executor of one controller path.

Lock-step interface (docs/dialects.md "Structure of the controller"):

* ``produce(now_tick)`` interprets blocks until the look-ahead buffer holds the path's capacity,
  a buffer-stopping block is reached (G31, live position read, M00/M01, #3006, tool setter, path
  wait, program end), an alarm stops the path, or the interpreter's own clock runs more than the
  horizon ahead of ``now_tick``. It returns the commands it added to the buffer.
* ``retire(count, tick)`` - motion finished the oldest ``count`` buffered commands at ``tick``.
* ``feedback(fb, tick)`` - the answer to the buffer-stopping command that ends the buffer.
* ``cycle_start(tick)`` - the operator presses CYCLE START (after M00/M01/#3006/single block, or for
  the next part after M02/M30/M99).

Block-time model: every processed block advances the interpreter clock ``interp_tick`` by its
cost (machine ``block_processing_ms`` for an NC block; ``macro_statement_s + terms * macro_term_s``
for a macro statement; ``call_s`` per call; ``goto_scan_s`` per block searched); a command's
``ready_tick`` is the clock after its block. A full buffer stops the interpreter until motion
retires a command; a buffer stop resumes it at the feedback tick. So short blocks at high feed
starve the planner exactly as on the control.

All state is in :class:`~cnctwin.controller.state.PathState`; this object only binds it to the
immutable context (dialect, programs, machine, parameters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import hooks
from .alarms import ControllerAlarm
from .expr import Evaluator, NeedLive, round_half_away
from .hooks import PathItem, Step
from .ir import ArcSpec, Command, Feed, FeedMode, Kind, LineRef, Smoothing, Space
from .machine import AxisInfo, MachineModel
from .pipeline import PLANES, Transform, arc_in_plane
from .program import Block, ProgramStore
from .state import Frame, LoopEntry, PathState, new_locals


@dataclass
class PathContext:
    """Immutable per-path context (rebuilt identically on resume)."""

    dialect: object                 # cnctwin.controller.dialects.base.Dialect
    store: ProgramStore
    machine: MachineModel
    path: int
    params: dict                    # resolved control parameters (dialect defaults + project settings)
    tick_s: float = 1e-6
    tool_table: object = None       # cnctwin.core.state.ToolTable
    controller: object = None       # the Controller owning all paths (path waits)
    work_offsets_m: dict = field(default_factory=dict)   # initial "G54": [SI per axis letter order]
    tools: list = field(default_factory=list)            # initial tool offsets (ToolOffsetInit)
    nozzle_positions: dict = field(default_factory=dict)  # tool number -> programmable nozzle position


class PathExecutor:
    def __init__(self, ctx: PathContext, state: PathState, root):
        self.ctx = ctx
        self.s = state
        self.root = root
        self.dialect = ctx.dialect
        self.store = ctx.store
        self.machine = ctx.machine
        self.axes: list[AxisInfo] = ctx.machine.paths[ctx.path]
        self.by_letter = {a.letter: a for a in self.axes}
        self.letters = [a.letter for a in self.axes]
        self.p = ctx.params
        self.tick_s = ctx.tick_s
        self.ev = Evaluator(self)
        self.nat = bool(self.p.get("nat", False))
        self.block: Block | None = None
        self.block_pc = -1
        self._jumped = False
        self._hold = False
        self._filter = None
        self._snapshot = None
        self._snapshot_seq = -1
        self._transform = None
        self.step_ref: LineRef | None = None
        self.handler = self.dialect.make_handler(self)

    # ================================================================== lock-step interface

    def capacity(self) -> int:
        limit = self.s.lookahead_limit
        cap = self.machine.lookahead_blocks if self.machine.lookahead_blocks > 0 else 1
        return max(1, min(cap, limit) if limit > 0 else cap)

    def start(self, tick: int) -> None:
        """Cycle start of the run (the interpreter begins at ``tick``)."""
        if not self.s.sys.get("started"):
            self.s.sys["started"] = True
            self.s.interp_tick = max(self.s.interp_tick, int(tick))

    def produce(self, now_tick: int) -> list[Command]:
        s = self.s
        self.start(now_tick)
        first = len(s.buffer)
        horizon = int(now_tick) + self.p["horizon_ticks"]
        while not (s.alarm or s.waiting):
            if len(s.buffer) >= self.capacity():
                s.blocked = True
                break
            if s.interp_tick > horizon:
                break
            try:
                self._step()
            except ControllerAlarm as alarm:
                self._stop_alarm(alarm)
            except NeedLive as need:
                self._live_sync(need)
        return s.buffer[first:]

    def retire(self, count: int, tick: int) -> None:
        s = self.s
        count = min(int(count), len(s.buffer))
        if count <= 0:
            return
        del s.buffer[:count]
        s.retired += count
        if s.blocked and len(s.buffer) < self.capacity():
            s.blocked = False
            s.interp_tick = max(s.interp_tick, int(tick))

    def feedback(self, fb, tick: int) -> None:
        s = self.s
        tick = int(tick)
        if fb is not None and fb.alarm:
            info = {"code": str(fb.alarm.get("code", "")), "text": str(fb.alarm.get("text", "MACHINE ALARM")),
                    "dialect": self.dialect.name, "key": "machine_alarm", "detail": "", "source": "machine"}
            self._set_alarm(info, tick)
            return
        if fb is not None and fb.tool_life_expired:
            for number in fb.tool_life_expired:
                if number not in s.expired_tools:
                    s.expired_tools.append(int(number))
            provider = hooks.TOOL_LIFE.get("default")
            if provider is not None:
                provider.expired(self, list(fb.tool_life_expired))
        waiting = s.waiting
        if waiting is None:
            return
        kind = waiting["kind"]
        if kind == "path_wait":
            waiting["arrived"] = tick
            self.ctx.controller.arrive(self.ctx.path, tick)
            return
        self._retire_all(tick)
        s.waiting = None
        s.interp_tick = max(s.interp_tick, tick)
        try:
            if kind == "live":
                s.live_values = {"depth": waiting["depth"], "pc": waiting["pc"],
                                 "machine": dict(fb.machine_pos) if fb is not None else {},
                                 "commanded": waiting.get("commanded", {})}
            elif kind in ("probe", "setter"):
                self._probe_feedback(waiting, fb)
            elif kind == "end":
                self._next_run(tick)
        except ControllerAlarm as alarm:
            self._stop_alarm(alarm)

    def cycle_start(self, tick: int) -> None:
        waiting = self.s.waiting
        if waiting is not None and waiting["kind"] in ("stop", "end"):
            self.feedback(None, tick)

    def release_wait(self, tick: int) -> None:
        """Called by the Controller when every path of this wait has arrived."""
        s = self.s
        self._retire_all(tick)
        s.waiting = None
        s.interp_tick = max(s.interp_tick, int(tick))

    def reset(self, tick: int) -> None:
        """RESET: clear an alarm, rewind to the start of the main program (modal kept, as on FANUC)."""
        s = self.s
        s.alarm = None
        s.waiting = None
        s.buffer.clear()
        s.pending.clear()
        s.pending_call = None
        s.comp_state = None
        self._filter = None
        self._rewind()
        s.interp_tick = max(s.interp_tick, int(tick))

    def _retire_all(self, tick: int) -> None:
        if self.s.buffer:
            self.retire(len(self.s.buffer), tick)

    # ================================================================== interpretation loop

    def _step(self) -> None:
        s = self.s
        self._snapshot = None
        self.step_ref = None
        if s.pending:
            step = s.pending.pop(0)
            try:
                self.run_step(step)
            finally:
                self.step_ref = None
            return
        if s.pending_call is not None:
            call = s.pending_call
            s.pending_call = None
            self._do_call(call)
            return
        frame = s.frames[-1]
        program = self.store.programs[frame.program]
        if frame.pc >= len(program.blocks):
            self._end_of_text(frame)
            return
        block = program.blocks[frame.pc]
        self.block = block
        self.block_pc = frame.pc
        self._jumped = False
        self._hold = False
        depth = len(s.frames)
        live = s.live_values
        retry = live is not None and live.get("depth") == depth and live.get("pc") == frame.pc
        if not retry:
            self._charge(block)
            s.block_seq += 1
            s.live_values = None
        self.handler.execute(block)
        if not self._jumped and not self._hold and s.frames and len(s.frames) >= depth:
            s.frames[depth - 1].pc += 1
        if not self._hold:
            s.live_values = None
            if block.n is not None:
                s.last_n = block.n
            if s.operator.single_block and not s.sys.get("single_block_suppressed") and self._block_emitted(block) \
                    and not (s.waiting or s.alarm):
                self.emit(Command(Kind.STOP, self.line_ref(), stop="single_block"))
                self.wait("stop", reason="single_block")

    def _block_emitted(self, block: Block) -> bool:
        return block.stmt is None and bool(block.words)

    def _charge(self, block: Block) -> None:
        p = self.p
        if block.stmt is not None:
            cost = p["macro_statement_s"] + block.terms * p["macro_term_s"]
        elif block.words:
            cost = self.machine.block_processing_s + block.terms * p["macro_term_s"]
        else:
            cost = 0.0
        self.advance(cost)

    def advance(self, seconds: float) -> None:
        ticks = int(math.ceil(seconds / self.tick_s - 1e-9)) if seconds > 0.0 else 0
        self.s.interp_tick += max(ticks, self.p["min_block_ticks"])

    def _end_of_text(self, frame: Frame) -> None:
        program = self.store.programs[frame.program]
        if len(self.s.frames) > 1:
            raise ControllerAlarm("sub_without_m99", f"end of {program.name} without M99")
        raise ControllerAlarm("end_of_record", f"end of {program.name} without M02/M30/M99")

    # ================================================================== alarms, events, waits

    def alarm(self, key: str, detail: str = "") -> ControllerAlarm:
        return ControllerAlarm(key, detail)

    def _stop_alarm(self, alarm: ControllerAlarm) -> None:
        if alarm.code is not None:
            info = {"code": alarm.code, "text": alarm.text or "", "dialect": self.dialect.name, "key": alarm.key,
                    "detail": alarm.detail, "source": "program"}
        else:
            info = self.dialect.alarms.resolve(alarm.key, alarm.detail).as_dict()
        self._set_alarm(info, self.s.interp_tick)
        cmd = Command(Kind.ALARM, self.line_ref(), alarm=dict(info))
        self._push_buffer(cmd, self.s.interp_tick)

    def _set_alarm(self, info: dict, tick: int) -> None:
        s = self.s
        s.alarm = dict(info)
        ref = self.line_ref()
        s.events.append({"kind": "alarm", "tick": int(tick), "code": info["code"], "text": info["text"],
                         "key": info.get("key", ""), "detail": info.get("detail", ""),
                         "source": info.get("source", ""), "program": ref.program, "line": ref.line,
                         "block": ref.block})

    def note(self, kind: str, message: str, **data) -> None:
        """Record a warning/info event (the domain turns it into a finding)."""
        ref = self.line_ref()
        event = {"kind": kind, "tick": self.s.interp_tick, "message": message, "program": ref.program,
                 "line": ref.line, "block": ref.block}
        event.update({k: v for k, v in data.items()})
        self.s.events.append(event)

    def wait(self, kind: str, **data) -> None:
        waiting = {"kind": kind, "since": self.s.interp_tick, "depth": len(self.s.frames),
                   "pc": self.s.frames[-1].pc if self.s.frames else 0}
        waiting.update(data)
        self.s.waiting = waiting

    def _live_sync(self, need: NeedLive) -> None:
        self.flush_comp("buffer stop")
        self._hold = True
        frame = self.s.frames[-1]
        cmd = Command(Kind.SYNC, self.line_ref(), sync={"reason": "live_read", "variable": int(need.variable)})
        self.emit(cmd)
        self.wait("live", depth=len(self.s.frames), pc=frame.pc,
                  commanded={a.name: self.s.mach.get(a.letter, 0) * a.quantum for a in self.axes})

    # ================================================================== variables

    def read(self, number: int):
        return self.dialect.sysvars.read(self, number)

    def write(self, number: int, value) -> None:
        self.dialect.sysvars.write(self, number, value)

    def adp(self, number: int, value):
        return self.dialect.sysvars.adp(self, number, value)

    def local_level(self) -> int:
        return self.s.frames[-1].level if self.s.frames else 0

    def live(self, name: str) -> float:
        """Live (encoder) machine value of an axis for a buffer-stopping read; raises NeedLive when no
        fresh feedback exists for the current block."""
        live = self.s.live_values
        frame = self.s.frames[-1]
        if live is None or live.get("depth") != len(self.s.frames) or live.get("pc") != frame.pc:
            raise NeedLive(0)
        machine = live["machine"]
        if name in machine:
            return float(machine[name])
        axis = next(a for a in self.axes if a.name == name)
        return self.s.mach.get(axis.letter, 0) * axis.quantum

    def live_commanded(self, name: str) -> float:
        live = self.s.live_values
        return float(live["commanded"].get(name, 0.0)) if live else 0.0

    # ================================================================== flow control

    def goto(self, n: int) -> None:
        if not 0 <= n <= 99999:
            raise ControllerAlarm("sequence_out_of_range", f"GOTO {n}")
        frame = self.s.frames[-1]
        program = self.store.programs[frame.program]
        target = program.find_n(n, frame.pc + 1)
        if target is None:
            raise ControllerAlarm("sequence_number_not_found", f"N{n} in {program.name}")
        scanned = (target - frame.pc) if target > frame.pc else (len(program.blocks) - frame.pc + target)
        self.advance(scanned * self.p["goto_scan_s"])
        frame.loops = [lp for lp in frame.loops if lp.do_pc <= target <= lp.end_pc]
        frame.pc = target
        self._jumped = True

    def loop_while(self, condition: bool, m: int) -> None:
        if m not in (1, 2, 3):
            raise ControllerAlarm("illegal_loop_number", f"DO {m}")
        frame = self.s.frames[-1]
        pc = frame.pc
        entry = frame.loops[-1] if frame.loops and frame.loops[-1].do_pc == pc else None
        if not condition:
            end_pc = entry.end_pc if entry is not None else self._find_end(pc, m)
            if entry is not None:
                frame.loops.pop()
            frame.pc = end_pc + 1
            self._jumped = True
            return
        if entry is None:
            if any(lp.m == m for lp in frame.loops):
                raise ControllerAlarm("illegal_loop_number", f"DO {m} nested in DO {m}")
            if len(frame.loops) >= 3:
                raise ControllerAlarm("illegal_loop_number", "more than 3 nested DO loops")
            frame.loops.append(LoopEntry(m, pc, self._find_end(pc, m)))

    def loop_end(self, m: int) -> None:
        if m not in (1, 2, 3):
            raise ControllerAlarm("illegal_loop_number", f"END {m}")
        frame = self.s.frames[-1]
        if not frame.loops or frame.loops[-1].m != m:
            raise ControllerAlarm("illegal_loop_number", f"END {m} without DO {m}")
        frame.pc = frame.loops[-1].do_pc
        self._jumped = True

    def _find_end(self, pc: int, m: int) -> int:
        program = self.store.programs[self.s.frames[-1].program]
        for k in range(pc + 1, len(program.blocks)):
            stmt = program.blocks[k].stmt
            if stmt is not None and stmt[0] == "end" and stmt[1] == m:
                return k
        raise ControllerAlarm("missing_end", f"DO {m} without END {m}")

    def request_call(self, kind: str, program: int, args: list | None = None, repeat: int = 1,
                     entry_pc: int = 0, extra: dict | None = None) -> None:
        """Call at the end of the current block (M98, M97, G65, macro-call codes)."""
        self.s.pending_call = {"kind": kind, "program": int(program), "args": args, "repeat": int(repeat),
                               "entry_pc": int(entry_pc), "line": self.block.line if self.block else 0,
                               "from": self.store.programs[self.s.frames[-1].program].name, "extra": extra or {}}

    def _do_call(self, call: dict) -> None:
        s = self.s
        kind = call["kind"]
        macro = kind in ("macro", "modal")
        subs = len(s.frames) - 1
        levels = s.frames[-1].level
        if subs + 1 > self.p["sub_nesting_max"]:
            raise ControllerAlarm("nesting", f"more than {self.p['sub_nesting_max']} call levels")
        if macro and levels + 1 > self.p["macro_nesting_max"]:
            raise ControllerAlarm("nesting", f"more than {self.p['macro_nesting_max']} macro levels")
        self.advance(self.p["call_s"])
        level = levels + 1 if macro else levels
        args = call.get("args")
        if macro:
            while len(s.locals) <= level:
                s.locals.append(new_locals())
            s.locals[level] = list(args) if args is not None else new_locals()
        if call["repeat"] <= 0:
            return
        frame = Frame(kind, call["program"], call["entry_pc"], level, call["repeat"] - 1, call["entry_pc"],
                      list(args) if (macro and args is not None) else None, [], call["line"], call["from"])
        s.frames.append(frame)
        for key, value in call.get("extra", {}).items():
            s.sys[key] = value

    def ret(self, n: int | None = None) -> None:
        """M99 (P n): end of a repetition or return to the caller (at its block after the call, or N n)."""
        s = self.s
        frame = s.frames[-1]
        self.advance(self.p["call_s"])
        if len(s.frames) == 1:
            self.flush_comp("program end")
            self.emit(Command(Kind.STOP, self.line_ref(), stop="M99-end"))
            s.sys["end_kind"] = "M99"
            s.sys["end_restart_n"] = n
            self.wait("end")
            self._jumped = True
            return
        if frame.repeat_left > 0:
            frame.repeat_left -= 1
            frame.pc = frame.entry_pc
            frame.loops = []
            if frame.args is not None:
                s.locals[frame.level] = list(frame.args)
            self._jumped = True
            return
        s.frames.pop()
        if frame.kind in ("macro", "modal") and frame.level < len(s.locals):
            s.locals[frame.level] = new_locals()
        caller = s.frames[-1]
        if n is not None:
            program = self.store.programs[caller.program]
            target = program.find_n(int(n), caller.pc)
            if target is None:
                raise ControllerAlarm("sequence_number_not_found", f"M99 P{n}: N{n} in {program.name}")
            caller.pc = target
        self._jumped = True

    def _rewind(self) -> None:
        s = self.s
        main = self.store.mains[self.ctx.path]
        s.frames = [Frame("main", main, 0, 0)]
        s.locals = [new_locals()]
        s.pending.clear()
        s.pending_call = None
        s.live_values = None
        self.handler.on_reset()

    def _next_run(self, tick: int) -> None:
        s = self.s
        restart_n = (s.sys.get("end_restart_n"))
        s.runs += 1
        if s.sys.get("end_kind") == "M99":
            frame = s.frames[0]
            frame.loops = []
            if restart_n is not None:
                program = self.store.programs[frame.program]
                target = program.find_n(int(restart_n), 0)
                if target is None:
                    raise ControllerAlarm("sequence_number_not_found", f"M99 P{restart_n}")
                frame.pc = target
            else:
                frame.pc = 0
            return
        self._rewind()

    def end_program(self, code: str) -> None:
        """M02 / M30: end of the part program; waits for CYCLE START (next part)."""
        self.flush_comp("program end")
        self.emit(Command(Kind.STOP, self.line_ref(), stop=code))
        self.s.sys["end_kind"] = code
        self.s.sys["end_restart_n"] = None
        self.wait("end")
        self._jumped = True

    # ================================================================== emission

    def line_ref(self) -> LineRef:
        s = self.s
        if self.step_ref is not None:
            return self.step_ref
        if not s.frames:
            return LineRef("", 0, "")
        frame = s.frames[-1]
        program = self.store.programs[frame.program]
        block = self.block if (self.block is not None and 0 <= self.block_pc < len(program.blocks)
                               and program.blocks[self.block_pc] is self.block) else None
        if block is None:
            pc = min(frame.pc, len(program.blocks) - 1)
            block = program.blocks[pc] if program.blocks else None
        stack = tuple(f"{f.call_program}:{f.call_line}" for f in s.frames[1:]) if len(s.frames) > 1 else ()
        if block is None:
            return LineRef(program.file, 0, "", None, stack)
        return LineRef(program.file, block.line, block.text, block.n, stack)

    def emit(self, cmd: Command) -> None:
        """A non-motion command, in program order (through the compensation queue when active)."""
        item = PathItem("command", command=cmd, ref=cmd.ref, block_seq=self.s.block_seq,
                        ready_tick=self.s.interp_tick)
        self._release(self._push_item(item))

    def emit_item(self, item: PathItem) -> None:
        self._release(self._push_item(item))

    def _push_item(self, item: PathItem) -> list[PathItem]:
        s = self.s
        if s.comp_state is None:
            return [item]
        flt = self._comp_filter()
        out = flt.push(self, item)
        if self._filter is flt:             # the provider may have ended compensation (end_comp) in push
            s.comp_state = flt.to_data()
        return out

    def _comp_filter(self):
        if self._filter is None:
            factory = hooks.COMPENSATION.get(self.s.comp_kind)
            if factory is None:
                raise ControllerAlarm("comp_unavailable", self.s.comp_kind or "cutter")
            self._filter = factory.from_data(self, self.s.comp_state) if self.s.comp_state else factory.create(self)
        return self._filter

    def start_comp(self, kind: str) -> None:
        """Compensation of ``kind`` begins (G41/G42 ...): needs a registered provider."""
        s = self.s
        if hooks.COMPENSATION.get(kind) is None:
            raise ControllerAlarm("comp_unavailable", f"{kind} compensation")
        if s.comp_state is None:
            s.comp_kind = kind
            self._filter = hooks.COMPENSATION[kind].create(self)
            s.comp_state = self._filter.to_data()

    def flush_comp(self, reason: str) -> None:
        s = self.s
        if s.comp_state is None:
            return
        flt = self._comp_filter()
        out = flt.flush(self, reason)
        if self._filter is flt:
            s.comp_state = flt.to_data()
        self._release(out)

    def end_comp(self) -> None:
        """The provider reports that compensation has fully ended (queue empty, G40 done)."""
        self.s.comp_state = None
        self.s.comp_kind = ""
        self._filter = None

    def _release(self, items: list[PathItem]) -> None:
        for item in items:
            cmd = self.finalize(item)
            self._push_buffer(cmd, item.ready_tick, item.block_seq)

    def _push_buffer(self, cmd: Command, ready_tick: int, block_seq: int | None = None) -> None:
        s = self.s
        cmd.ready_tick = int(ready_tick)
        cmd.path = self.ctx.path
        cmd.block_seq = s.block_seq if block_seq is None else block_seq
        if s.suppress:
            if cmd.kind in (Kind.SPINDLE, Kind.COOLANT, Kind.TOOL_CHANGE, Kind.MODE):
                s.restart_summary[cmd.kind.value] = cmd
            return
        s.buffer.append(cmd)
        s.issued += 1

    def finalize(self, item: PathItem) -> Command:
        if item.kind == "command":
            return item.command
        if item.transform:
            provider = hooks.TRANSFORMS.get(item.transform)
            if provider is None:
                raise ControllerAlarm("transform_unavailable", item.transform)
            return provider.to_command(self, item)
        return self.default_command(item)

    def machine_counts(self, item: PathItem) -> list[int]:
        if item.transform and item.space != "direct":
            provider = hooks.TRANSFORMS.get(item.transform)
            if provider is None:
                raise ControllerAlarm("transform_unavailable", item.transform)
            if hasattr(provider, "machine_counts"):
                return provider.machine_counts(self, item)
        if item.space == "direct":
            return [int(round_half_away(v / a.quantum)) for v, a in zip(item.end, self.axes)]
        return [int(round_half_away(v / a.quantum)) + item.offset[i] for i, (v, a) in enumerate(zip(item.end, self.axes))]

    def default_command(self, item: PathItem) -> Command:
        counts = self.machine_counts(item)
        target = {a.name: counts[i] * a.quantum for i, a in enumerate(self.axes)}
        kind = {"rapid": Kind.RAPID, "linear": Kind.LINEAR, "arc": Kind.ARC, "probe": Kind.PROBE,
                "setter": Kind.TOOL_SETTER}[item.kind]
        arc = None
        if item.kind == "arc":
            centre = list(item.centre)
            if item.space != "direct":
                for i, letter in enumerate("XYZ"):
                    axis = self.by_letter.get(letter)
                    if axis is not None:
                        centre[i] += item.offset[self.letters.index(letter)] * axis.quantum
            arc = ArcSpec(item.plane, tuple(centre), tuple(item.normal), item.sweep_rad, item.r_start, item.r_end)
        cmd = Command(kind, item.ref, Space.MACHINE, target, None, None, arc, item.feed, item.smoothing,
                      modal=dict(item.modal))
        if item.kind == "probe":
            cmd.probe = dict(item.probe or {})
        elif item.kind == "setter":
            cmd.setter = dict(item.probe or {})
        return cmd

    # ================================================================== positions and offsets

    def counts(self, letter: str, si: float) -> int:
        return int(round_half_away(si / self.by_letter[letter].quantum))

    def si(self, letter: str, counts: int) -> float:
        return counts * self.by_letter[letter].quantum

    def invalidate_transform(self) -> None:
        self._transform = None

    def transform(self) -> Transform:
        if self._transform is None:
            self._transform = self._build_transform()
        return self._transform

    def _build_transform(self) -> Transform:
        s = self.s
        mirror = {l: self.si(l, c) for l, c in s.mirror.items()}
        scale = None
        if s.scale is not None:
            scale = {"centre": {l: self.si(l, c) for l, c in s.scale["centre"].items()},
                     "factor": dict(s.scale["factor"])}
        rotation = None
        if s.rotation is not None:
            a, b, _ = PLANES[s.rotation["plane"]]
            ca, cb = s.rotation["centre"]
            rotation = {"plane": s.rotation["plane"], "centre": [self.si(a, ca) if a in self.by_letter else 0.0,
                                                                 self.si(b, cb) if b in self.by_letter else 0.0],
                        "angle_deg": s.rotation["angle_deg"]}
        feature = None
        if s.feature is not None:
            origin = [self.si(l, c) if l in self.by_letter else 0.0 for l, c in zip("XYZ", s.feature["origin"])]
            m = s.feature["matrix"]
            feature = {"origin": origin, "matrix": [m[0:3], m[3:6], m[6:9]]}
        return Transform(mirror, scale, rotation, feature)

    def offset_counts(self, letters_tool: bool = True) -> list[int]:
        """Work -> machine translation per axis (counts): work offset + EXT + G52 + G92 + tool offsets."""
        s = self.s
        wcs = s.offsets.get(s.wcs) or [0] * len(self.axes)
        ext = s.offsets.get("EXT") or [0] * len(self.axes)
        out = []
        for i, letter in enumerate(self.letters):
            value = wcs[i] + ext[i] + s.g52.get(letter, 0) + s.g92.get(letter, 0)
            if letters_tool:
                value += self.handler.tool_offset_counts(letter)
            out.append(value)
        return out

    def program_si(self) -> dict[str, float]:
        return {l: self.si(l, c) for l, c in self.s.pos.items()}

    def work_from_program(self, prog_counts: dict[str, int]) -> list[float]:
        """Work-frame SI values (path axis order) of a program-frame point given in counts."""
        prog = {l: self.si(l, c) for l, c in prog_counts.items()}
        work = self.transform().forward(prog)
        return [work[l] for l in self.letters]

    def program_from_machine(self, machine_counts: dict[str, int], applied: dict[str, int] | None = None) -> dict[str, int]:
        """Inverse pipeline: machine counts -> program-frame counts (the tool offsets contained in
        ``machine_counts`` are ``applied``, default the ones currently applied)."""
        s = self.s
        applied = s.applied if applied is None else applied
        provider = hooks.TRANSFORMS.get(s.tcp) if s.tcp else None
        if provider is not None and hasattr(provider, "work_from_machine"):
            work = provider.work_from_machine(self, machine_counts, applied)
        else:
            base = self.offset_counts(letters_tool=False)
            work = {}
            for i, letter in enumerate(self.letters):
                work[letter] = self.si(letter, machine_counts.get(letter, 0) - base[i] - applied.get(letter, 0))
        prog = self.transform().inverse(work)
        return {l: self.counts(l, v) for l, v in prog.items()}

    def resync_program_position(self) -> None:
        """After a work offset / G52 / G92 / transform change: the program position of the tool."""
        self.s.pos = self.program_from_machine(self.s.mach)

    # ================================================================== motion

    def feed_spec(self, rapid: bool, override: bool = True, feed_si: float | None = None,
                  mode: str | None = None) -> Feed:
        s = self.s
        if rapid:
            return Feed(FeedMode.RAPID, 0.0, override)
        mode = mode or self.handler.feed_mode()
        value = s.feed_si if feed_si is None else feed_si
        allowed = override and not (int(s.sys.get("3004", 0)) & 2) and s.modal.get("15") != "G63"
        return Feed(FeedMode(mode), float(value), allowed)

    def smoothing_spec(self, exact: bool = False) -> Smoothing:
        sm = self.s.smoothing
        corner = sm.get("corner", "blend")
        if exact or self.s.sys.get("g09"):
            corner = "exact_stop"
        return Smoothing(corner, float(sm.get("tolerance_m", 0.0)), str(sm.get("mode_code", "")),
                         bool(sm.get("lookahead", True)))

    def snapshot(self) -> dict:
        """Modal snapshot for the commands of the current step (a pure function of the state; cached for
        the step only, so a resumed executor computes the same values)."""
        if self._snapshot is None:
            self._snapshot = self.handler.modal_snapshot()
        return self._snapshot

    def move(self, kind: str, target: dict[str, int], feed: Feed | None = None, *, arc: dict | None = None,
             comp: bool = True, exact: bool = False, modal: dict | None = None, probe: dict | None = None,
             bypass: bool = False, corner: dict | None = None) -> PathItem:
        """Interpolated move to program-frame ``target`` (counts; axes not named keep their position)."""
        s = self.s
        start_w = self.work_from_program(s.pos)
        new_pos = dict(s.pos)
        new_pos.update(target)
        end_w = self.work_from_program(new_pos)
        offset = self.offset_counts()
        item = PathItem(kind, start_w, end_w, plane=self.handler.plane(), feed=feed,
                        smoothing=self.smoothing_spec(exact), offset=offset, comp=s.modal.get("7", "G40"),
                        comp_radius_m=self.handler.comp_radius(), transform=s.tcp, ref=self.line_ref(),
                        block_seq=s.block_seq, ready_tick=s.interp_tick, modal=dict(self.snapshot()), probe=probe,
                        bypass=bypass or not comp, corner=corner)
        if modal:
            item.modal.update(modal)
        if arc is not None:
            item.centre = arc["centre"]
            item.normal = arc["normal"]
            item.sweep_rad = arc["sweep"]
            item.r_start = arc["r_start"]
            item.r_end = arc["r_end"]
        s.pos = new_pos
        counts = self.machine_counts(item)
        s.mach = {l: counts[i] for i, l in enumerate(self.letters)}
        s.applied = {l: self.handler.tool_offset_counts(l) for l in self.letters}
        self.emit_item(item)
        return item

    def machine_move(self, kind: str, target_machine: dict[str, int], feed: Feed | None = None,
                     exact: bool = True, modal: dict | None = None) -> None:
        """Move in machine coordinates (G53, reference point legs); other axes keep their machine position."""
        s = self.s
        start = [self.si(l, s.mach.get(l, 0)) for l in self.letters]
        mach = dict(s.mach)
        mach.update(target_machine)
        end = [self.si(l, mach.get(l, 0)) for l in self.letters]
        item = PathItem(kind, start, end, plane=self.handler.plane(), feed=feed, smoothing=self.smoothing_spec(exact),
                        offset=[0] * len(self.axes), comp=s.modal.get("7", "G40"), space="direct",
                        ref=self.line_ref(), block_seq=s.block_seq, ready_tick=s.interp_tick,
                        modal=dict(self.snapshot()), bypass=True)
        if modal:
            item.modal.update(modal)
        s.mach = mach
        s.pos = self.program_from_machine(mach, s.applied)
        self.emit_item(item)

    def arc_move(self, target: dict[str, int], direction: int, centre_offset: dict[str, float] | None,
                 radius: float | None, feed: Feed, comp: bool = True, turns: int = 0,
                 plane: str | None = None, modal: dict | None = None, corner: dict | None = None) -> PathItem | None:
        """G02/G03 from the current position to ``target`` (program counts) in the active plane."""
        s = self.s
        plane = plane or self.handler.plane()
        a, b, n = PLANES[plane]
        if a not in self.by_letter or b not in self.by_letter:
            raise ControllerAlarm("illegal_plane", f"plane {plane} axes are not on this path")
        start = dict(s.pos)
        end = dict(start)
        end.update(target)
        sa, sb = self.si(a, start[a]), self.si(b, start[b])
        ea, eb = self.si(a, end[a]), self.si(b, end[b])
        same = start[a] == end[a] and start[b] == end[b]
        transform = self.transform()
        tol = self.p["arc_tolerance_m"]
        offset = None
        if centre_offset is not None:
            offset = (centre_offset.get(a, 0.0), centre_offset.get(b, 0.0))
        rad = radius
        if rad is not None and not transform.scales_uniformly(plane):
            rad = rad * transform.radius_factor(plane) / max(abs(transform.scale["factor"].get(a, 1.0)), 1e-300)
        geometry = arc_in_plane((sa, sb), (ea, eb), direction, offset, rad, tol, same)
        if geometry["centre"] is None:
            if any(start.get(l) != end.get(l) for l in end):
                self.note("info", "R arc with the end point on the start point: no circular motion (FANUC rule)")
                return self.move("linear", target, feed, comp=comp)
            self.note("info", "R arc with the end point on the start point: no motion (FANUC rule)")
            return None
        sweep = geometry["sweep"] + 2.0 * math.pi * turns * direction
        centre_prog = dict(start)
        centre_si = {l: self.si(l, c) for l, c in centre_prog.items()}
        centre_si[a], centre_si[b] = geometry["centre"]
        centre_w = transform.forward(centre_si)
        normal, flipped = transform.arc_normal(plane)
        if flipped:
            sweep = -sweep
        start_w = transform.forward({l: self.si(l, c) for l, c in start.items()})
        end_w = transform.forward({l: self.si(l, c) for l, c in end.items()})
        c3 = [centre_w.get(l, 0.0) for l in "XYZ"]
        normal_axis = "XYZ".index(n)
        c3[normal_axis] = start_w.get(n, 0.0) if n in start_w else 0.0
        r_start = _radius(start_w, c3, normal)
        r_end = _radius(end_w, c3, normal)
        arc = {"centre": c3, "normal": normal, "sweep": sweep, "r_start": r_start, "r_end": r_end}
        return self.move("arc", target, feed, arc=arc, comp=comp, modal=modal, corner=corner)

    def dwell(self, seconds: float = 0.0, revolutions: float = 0.0) -> None:
        cmd = Command(Kind.DWELL, self.line_ref(), dwell_s=float(seconds), dwell_rev=float(revolutions),
                      modal=dict(self.snapshot()))
        self.emit(cmd)

    # ================================================================== pending steps (cycles, probing)

    def queue_steps(self, steps: list[Step], front: bool = False) -> None:
        ref = self.line_ref()
        for step in steps:
            if step.ref is None:
                step.ref = ref
        if front:
            self.s.pending[0:0] = list(steps)
        else:
            self.s.pending.extend(steps)

    def run_step(self, step: Step) -> None:
        s = self.s
        kind = step.kind
        self.step_ref = step.ref
        if kind == "command":
            cmd = step.command
            cmd.ref = cmd.ref if cmd.ref is not None and cmd.ref.program else self.line_ref()
            self.emit(cmd)
            return
        if kind == "dwell":
            self.dwell(step.dwell_s, step.dwell_rev)
            return
        if kind == "mode":
            self.handler.apply_mode(step)
            return
        if kind == "mcodes":
            self.handler.deferred_m(step)
            return
        if kind == "resume":
            provider = hooks.RESUMERS.get(step.provider) or hooks.PROBES.get(step.provider)
            if provider is None:
                raise ControllerAlarm("probe_unavailable", step.provider)
            self.queue_steps(provider.resume(self, step.data, s.sys.get("last_feedback")), front=True)
            return
        target = {l: self.counts(l, v) for l, v in step.target.items() if l in self.by_letter}
        modal = dict(step.modal) if step.modal else None
        if step.machine:
            feed = self.feed_spec(kind == "rapid", step.override, step.feed, step.feed_mode or None)
            self.machine_move("rapid" if kind == "rapid" else "linear", target, feed, step.exact_stop, modal)
            return
        if kind == "rapid":
            self.move("rapid", target, self.feed_spec(True), comp=step.comp, exact=step.exact_stop, modal=modal,
                      bypass=not step.comp)
            return
        feed = self.feed_spec(False, step.override, step.feed, step.feed_mode or None)
        if feed.value <= 0.0:
            raise ControllerAlarm("feed_zero", f"{kind} in a cycle with F = 0")
        if kind == "linear":
            self.move("linear", target, feed, comp=step.comp, exact=step.exact_stop, modal=modal, bypass=not step.comp)
        elif kind == "arc":
            centre = {l: v - self.si(l, s.pos[l]) for l, v in (step.centre or {}).items() if l in s.pos}
            self.arc_move(target, step.direction, centre, None, feed, comp=step.comp, plane=step.plane or None,
                          modal=modal)
        elif kind in ("probe", "setter"):
            self.flush_comp("buffer stop")
            self.move(kind, target, feed, comp=False, exact=True, modal=modal, probe=step.probe, bypass=True)
            self.wait(kind, provider=step.provider, data=step.data, item_end=dict(s.mach), applied=dict(s.applied))
        else:
            raise ControllerAlarm("unsupported_function", f"cycle step kind {kind}")

    def _probe_feedback(self, waiting: dict, fb) -> None:
        s = self.s
        s.sys["last_feedback"] = _feedback_data(fb)
        probe = fb.probe if fb is not None else None
        end = waiting.get("item_end", dict(s.mach))
        triggered = bool(probe.triggered) if probe is not None else False
        if probe is not None and triggered and probe.machine_pos:
            latched = {a.letter: self.counts(a.letter, float(probe.machine_pos.get(a.name, end.get(a.letter, 0) * a.quantum)))
                       for a in self.axes}
        else:
            latched = dict(end)
        s.skip = self.program_from_machine(latched, waiting.get("applied"))
        s.skip_triggered = triggered
        if fb is not None and fb.machine_pos:
            actual = {a.letter: self.counts(a.letter, float(fb.machine_pos.get(a.name, latched[a.letter] * a.quantum)))
                      for a in self.axes}
        else:
            actual = latched
        s.mach = actual
        s.pos = self.program_from_machine(actual)
        provider_name = waiting.get("provider") or ""
        if provider_name:
            provider = hooks.PROBES.get(provider_name) or hooks.RESUMERS.get(provider_name)
            if provider is None:
                raise ControllerAlarm("probe_unavailable", provider_name)
            self.queue_steps(provider.resume(self, waiting.get("data", {}), fb), front=True)

    # ================================================================== restart support

    def search(self, predicate, limit_blocks: int = 10_000_000) -> int:
        """Interpret without producing commands until ``predicate(ex)`` is true (program restart):
        modal state, offsets, positions and the latest spindle/coolant/tool/mode commands
        (``restart_summary``) are kept. Returns the number of blocks interpreted."""
        s = self.s
        s.suppress = True
        count = 0
        try:
            while count < limit_blocks and s.alarm is None and not predicate(self):
                if s.waiting is not None:
                    s.waiting = None
                try:
                    self._step()
                except NeedLive:
                    s.live_values = {"depth": len(s.frames), "pc": s.frames[-1].pc, "machine": {}, "commanded": {}}
                except ControllerAlarm as alarm:
                    self._stop_alarm(alarm)
                count += 1
        finally:
            s.suppress = False
        return count


def _radius(point: dict, centre: list[float], normal: list[float]) -> float:
    v = [point.get("X", 0.0) - centre[0], point.get("Y", 0.0) - centre[1], point.get("Z", 0.0) - centre[2]]
    along = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2]
    w = [v[i] - along * normal[i] for i in range(3)]
    return math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])


def _feedback_data(fb) -> dict | None:
    if fb is None:
        return None
    data = {"time_s": float(fb.time_s), "machine_pos": {k: float(v) for k, v in fb.machine_pos.items()}}
    if fb.probe is not None:
        data["probe"] = {"triggered": bool(fb.probe.triggered),
                         "machine_pos": {k: float(v) for k, v in fb.probe.machine_pos.items()},
                         "time_s": float(fb.probe.time_s)}
    if fb.setter is not None:
        data["setter"] = {"tool": int(fb.setter.tool), "length_m": fb.setter.length_m,
                          "diameter_m": fb.setter.diameter_m, "broken": bool(fb.setter.broken)}
    return data
