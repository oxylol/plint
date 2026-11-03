"""Motion without cutting: the controller and the motion domain in lock-step (docs/models/motion.md
"Domain and loop protocol").

:class:`MotionRunner` is the reference implementation of the loop protocol the integration loop
(core.loop / core.hotloop) follows, with :func:`cnctwin.motion.kernels.run_motion` in place of the
full hot loop (no cutting load, no contact). It serves ``twin check`` cycle-time estimates, the tests
and as the worked example of the API: produce -> accept -> kernel_inputs -> run -> absorb ->
controller actions, repeated from each sync tick until the macro step is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.domain import RunContext
from ..core.state import SimState
from .abi import SYNC_NONE
from .kernels import run_motion


@dataclass
class RunResult:
    end_tick: int
    steps: int
    syncs: dict = field(default_factory=dict)
    stopped: str = ""
    frames: list = field(default_factory=list)       # MotionFrame of every step (when keep_frames)


class MotionRunner:
    def __init__(self, controller, domain, *, ctx: RunContext | None = None, state: SimState | None = None,
                 parts: int = 1, probe=None, keep_frames: bool = False, operator_delay_ticks: int = 0):
        """probe: optional callable (path, tick index, inputs) -> bool telling whether the probe stylus
        touches at that tick (stands in for removal's sweep_contact)."""
        self.controller = controller
        self.dom = domain
        self.ctx = ctx
        self.state = state if state is not None else SimState()
        self.parts = parts
        self.probe = probe
        self.keep_frames = keep_frames
        self.runs = [0] * len(controller.executors)
        self.delay = int(operator_delay_ticks)

    def run(self, max_s: float = 3600.0) -> RunResult:
        dom = self.dom
        ctx = self.ctx
        state = self.state
        dom.contact_from_loop = False
        if state.motion is None:
            dom.init(ctx, state)
        elif not getattr(dom, "bound", False) or getattr(state.motion, "planners", None) is None:
            dom.attach(ctx, state)
        n_paths = len(self.controller.executors)
        steps = 0
        syncs: dict = {}
        frames = []
        stopped = ""
        max_tick = int(max_s / dom.tick_s)
        k = 0
        while True:
            tick0 = int(state.clock.tick)
            now = tick0 + k * dom.ipo_ticks
            if now > max_tick:
                stopped = "time limit"
                break
            for p in range(n_paths):
                cmds = self.controller.produce(p, now)
                dom.accept(ctx, state, p, cmds, panel=self.controller.state.paths[p].operator)
            inputs = dom.kernel_inputs(ctx, state, state.bus_prev)
            k_from = int(inputs.k_from[0])
            if self.probe is None:
                run_motion(inputs.chain, inputs.servo, inputs.servo_state, inputs.spindle, inputs.spindle_state,
                           inputs.setpoint, inputs.loop, inputs.out, k_from, dom.macro_ipo, inputs.dist_force,
                           inputs.dist_torque, inputs.result)
            else:
                self._run_with_probe(inputs, k_from)
            code = int(inputs.result[0])
            if code != SYNC_NONE:
                syncs[code] = syncs.get(code, 0) + 1
            dom.absorb(ctx, state, inputs, state.bus_next)
            self._apply(dom.controller_actions(state))
            if dom.step_complete(state):
                dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
                if self.keep_frames:
                    frames.append(state.bus_next.motion)
                state.clock.tick = tick0 + dom.macro_ipo * dom.ipo_ticks
                state.clock.macro += 1
                state.swap_bus()
                steps += 1
                k = 0
                if self._finished():
                    break
            else:
                k = int(state.motion.meta["k_next"])
        return RunResult(int(state.clock.tick), steps, syncs, stopped, frames)

    def _run_with_probe(self, inputs, k_from: int) -> None:
        from .abi import SYNC_PROBE_TRIGGER
        from .kernels import motion_tick
        dom = self.dom
        result = inputs.result
        result[0], result[1], result[2] = SYNC_NONE, dom.macro_ipo - 1, -1
        from .abi import SYNC_QUEUE_EMPTY
        from .kernels import replan_before
        for k in range(k_from, dom.macro_ipo):
            if k > k_from:
                p_end = replan_before(inputs.setpoint, k)
                if p_end >= 0:
                    inputs.out.sync[k - 1, p_end] = SYNC_QUEUE_EMPTY
                    result[0], result[1], result[2] = SYNC_QUEUE_EMPTY, k - 1, p_end
                    return
            code = motion_tick(inputs.chain, inputs.servo, inputs.servo_state, inputs.spindle, inputs.spindle_state,
                               inputs.setpoint, inputs.loop, inputs.out, k, inputs.dist_force, inputs.dist_torque)
            for p in range(inputs.out.sync.shape[1]):
                if inputs.setpoint.probe[k, p] and self.probe(p, k, inputs) and code == SYNC_NONE:
                    inputs.out.probe_hit[k, p] = 1
                    inputs.out.sync[k, p] = SYNC_PROBE_TRIGGER
                    code = SYNC_PROBE_TRIGGER
            if code != SYNC_NONE:
                result[0], result[1] = code, k
                for p in range(inputs.out.sync.shape[1]):
                    if inputs.out.sync[k, p] != SYNC_NONE:
                        result[2] = p
                        break
                return

    def _apply(self, actions) -> None:
        ctrl = self.controller
        for kind, p, arg, tick in actions:
            if kind == "retire":
                ctrl.retire(p, int(arg), int(tick))
            elif kind == "feedback":
                ctrl.feedback(p, arg, int(tick))
            elif kind == "cycle_start":
                ctrl.cycle_start(p, int(tick) + self.delay)
            elif kind == "program_end":
                self.runs[p] += 1
                if self.runs[p] < self.parts:
                    ctrl.cycle_start(p, int(tick) + self.delay)

    def _finished(self) -> bool:
        ctrl = self.controller
        dom = self.dom
        if not ctrl.done():
            return False
        for p, planner in enumerate(dom.planners):
            if planner.queued() > 0 and planner.st.alarm is None and planner.st.cur.phase not in (5, 6):
                return False
        return all(self.runs[p] >= self.parts or ctrl.state.paths[p].alarm is not None
                   for p in range(len(ctrl.executors)))
