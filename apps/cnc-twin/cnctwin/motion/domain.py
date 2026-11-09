"""MotionDomain: planner, interpolation, feed drives, spindles, kinematics and timing of every path
(core.domain.Domain; docs/models/motion.md "Domain and loop protocol").

Per macro step (architecture 5.3), with ``n = macro_ipo`` L1 ticks starting at base tick ``tick0``:

    dom.accept(ctx, state, p, controller_commands, panel)    # every path, at the step start and at syncs
    inputs = dom.kernel_inputs(ctx, state, bus_prev)          # plans ticks k_from .. n-1 of every path
    <compiled loop: motion_tick(...) per L1 tick from inputs.k_from, interleaved with L0; returns at the
     first sync event: inputs.result = (code, tick index, path)>
    dom.absorb(ctx, state, inputs, bus_next)                  # commits ticks k_from .. k_stop
    for action in dom.controller_actions(state): <apply to the controller: retire / feedback / cycle_start>
    ... until dom.step_complete(state); then bus_next.motion holds the step's PathFrames.

The hot loop calls :func:`cnctwin.motion.kernels.motion_tick` per L1 tick with the previous tick's
cutting load per axis (N along +axis) and per spindle (N m, opposing rotation), sets
``inputs.out.in_cut[k, p]`` when the path's tool is in contact during tick k and
``inputs.out.probe_hit[k, p]`` (with ``out.sync[k, p] = SYNC_PROBE_TRIGGER``) when removal's
sweep_contact detects the probe stylus touching. :class:`cnctwin.motion.driver.MotionRunner` runs the
same protocol with :func:`cnctwin.motion.kernels.run_motion` when nothing cuts.
"""

from __future__ import annotations

import math

import numpy as np

from ..controller.ir import Feedback, Kind, ProbeResult, SetterResult
from ..core.bus import MotionFrame, PathFrame
from ..core.findings import Category, Severity
from .abi import (ALARM_OVERTRAVEL_NEG, ALARM_OVERTRAVEL_POS, CHAIN_TOOL, CHAIN_WORK, FK_M_F, FK_M_T0, FK_M_W0,
                  FK_S_T0, FK_W0_T0, MotionInputs, SP_PER_REV, SP_THREAD, SYNC_ALARM, SYNC_NONE, empty_loop_state,
                  empty_out, empty_setpoint)
from .channels import MAX_AXES, MAX_PATHS, MAX_SPINDLES, channels as motion_channels
from .geometry import PK_TIP
from .ik import PathKinematics
from .ledger import (CAT_AIR, CAT_CUT, CAT_FEED, CAT_RAPID, Record, Timeline, split_largest_remainder,
                     split_motion)
from .machine import MachineModel, num, plain
from .planner import PathPlanner, PlanState
from .schema import MotionSettings, settings as settings_from
from .servo import build_servo, build_spindles, initial_servo_state, initial_spindle_state
from .spindle import SpindleCmd, SpindleTimeline
from .state import MotionState, command_from_dict, command_to_dict, feedback_to_dict, piece_from_dict, piece_to_dict
from .timing import Timing, pump_kinds

TWO_PI = 2.0 * math.pi


def _get(obj, name, default=None):
    if obj is None:
        return default
    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return default if value is None else value


class MotionEnv:
    """Run-constant data the path planners share (rebuilt from the project after a resume)."""

    def __init__(self, dom: "MotionDomain"):
        self.dom = dom
        self.model = dom.model
        self.kin = dom.kin
        self.timing = dom.timing
        self.ipo_ticks = dom.ipo_ticks
        self.tick_s = dom.tick_s
        self.ipo_s = dom.ipo_ticks * dom.tick_s
        self.spindle_cmd = [SpindleCmd() for _ in dom.model.spindles]
        self.pump_kinds = pump_kinds(dom.model)
        self.setp_cmd = None
        self.setp_geo_capacity = 64
        self.sp_actual = None
        self.loop_theta0 = None
        self.wait_kind = None
        self._css = {}

    def slot_rotating(self, slot: int) -> bool:
        return self.dom.slot_rotating(slot)

    def slot_of(self, number: int, path: int) -> int | None:
        return self.dom.slot_of(number, path)

    def tool_change_axis(self, p: int) -> int | None:
        return self.dom.tool_change_axis(p)

    def setter_config(self, p: int, slot: int):
        return self.dom.setter_config(p, slot)

    def css_axis_info(self, s: int, p: int):
        key = (s, p)
        if key not in self._css:
            self._css[key] = self.dom.css_axis(s, p)
        return self._css[key]


class MotionDomain:
    name = "motion"

    def __init__(self, machine=None, *, tool_lengths: dict | None = None, rotating: dict | None = None,
                 tool_numbers: dict | None = None, poses_F_W=None, poses_W_S=None, settings: MotionSettings | None = None,
                 tick_s: float | None = None, ipo_ticks: int | None = None, macro_ipo: int | None = None,
                 project=None, path_letters=None):
        """Direct set-up (tests, the motion-only runner) or, with no machine, from the RunContext at init.

        tool_lengths: slot -> gauge-line-to-tip length (m); rotating: slot -> bool; tool_numbers:
        (number, path) -> slot."""
        self._direct = machine
        self._tool_lengths = dict(tool_lengths or {})
        self._rotating = dict(rotating or {})
        self._numbers = dict(tool_numbers or {})
        self._poses_F_W = poses_F_W
        self._poses_W_S = poses_W_S
        self._settings = settings
        self._tick_s = tick_s
        self._ipo_ticks = ipo_ticks
        self._macro_ipo = macro_ipo
        self._project = project
        self._path_letters = path_letters
        self.contact_from_loop = True
        self.bound = False
        self.last_channels: dict = {}

    # ================================================================== Domain protocol
    def channels(self):
        return motion_channels()

    def init(self, ctx, state) -> None:
        self._build(ctx, state)
        model = self.model
        q0 = np.array([ax.home for ax in model.axes])
        self.planners = [PathPlanner(self.env, p, q0) for p in range(len(model.paths))]
        ss = initial_servo_state(self.servo, q0)
        P = len(model.paths)
        ms = MotionState(ss, initial_spindle_state(model), empty_loop_state(P))
        ms.spindle_cmd = [c.to_dict() for c in self.env.spindle_cmd]
        ms.open_record = [-1] * P
        ms.idle_from = [-1] * P
        ms.ledger = {"start": [None] * P, "end": [None] * P, "blocks": [0] * P}
        ms.meta = {"k_next": 0, "tick0": 0, "macro": 0, "op_index": int(getattr(state.clock, "op_index", 0) or 0),
                   "part": int(getattr(state.clock, "part_index", 0) or 0)}
        ms.planners = self.planners
        ms.timeline_obj = self.timeline
        ms.snapshot_hook = lambda: self.snapshot(state)
        state.motion = ms
        self._state = state
        self._new_step_buffers(ms)
        self.bound = True

    def _build(self, ctx, state) -> None:
        project = self._project if self._project is not None else (getattr(ctx, "project", None) if ctx else None)
        machine = self._direct if self._direct is not None else _get(project, "machine")
        if machine is None:
            raise ValueError("MotionDomain needs a machine (project.machine or the constructor)")
        physics = _get(project, "physics")
        self.settings = self._settings or settings_from(_get(physics, "motion"))
        letters = self._path_letters
        if letters is None:
            ctrl = _get(physics, "controller")
            letters = _get(ctrl, "paths")
        self.model = MachineModel(machine, letters)
        model = self.model
        self.tick_s = float(self._tick_s or (ctx.tick_s if ctx is not None else 1e-6))
        ipo = self._ipo_ticks or (ctx.ipo_ticks if ctx is not None else None)
        if ipo is None:
            ipo = int(round(model.ipo_s / self.tick_s))
        self.ipo_ticks = int(ipo)
        self.macro_ipo = int(self._macro_ipo or (ctx.macro_ipo if ctx is not None else 10))
        self.dt = self.ipo_ticks * self.tick_s
        self.timing = Timing(model, project)
        self.P = len(model.paths)
        self._tool_table(ctx, state, project)
        self._poses(ctx, state, project)
        self.servo = build_servo(model, self.dt, self.settings.servo_omega_h, self.settings.friction_h)
        self.spindle = build_spindles(model, self.dt, self.settings.servo_omega_h)
        self.tool_slot = [-1] * self.P
        tips = [self._tip_N(-1) for _ in range(self.P)]
        spindles = [self._tool_spindle_index(p, -1) for p in range(self.P)]
        self.chain = model.build_chain(tips, self.poses_F_W, self.poses_W_S, spindles)
        self.chain_nominal = model.build_chain(tips, self.poses_F_W, self.poses_W_S, spindles, nominal=True)
        self.static_err = self.chain.err_const.copy()
        # the control's kinematic model (no geometric or thermal errors) plans; the loop's FK is physical
        self.kin = [PathKinematics(model, self.chain_nominal, p) for p in range(self.P)]
        self.timeline = Timeline()
        self.env = MotionEnv(self)
        NA = len(model.axes)
        S = max(1, len(model.spindles))
        G = 64
        self.setp = empty_setpoint(self.macro_ipo, NA, self.P, S, G)
        self.out = empty_out(self.macro_ipo, NA, self.P, S)
        self.env.setp_cmd = self.setp.cmd
        self.env.wait_kind = self.setp.wait_kind
        self.env.setp_geo_capacity = G
        self.plans: list = [None] * self.P
        self.item_ideal: dict = {}
        self.estimated = sorted(set(model.estimated))

    def _tool_table(self, ctx, state, project) -> None:
        self.tool_lengths = dict(self._tool_lengths)
        self.rotating = dict(self._rotating)
        self.numbers = dict(self._numbers)
        self.tool_radius = {}
        table = getattr(state, "tools", None) if state is not None else None
        entries = _get(_get(project, "tools"), "list") or []
        spindle_kinds = {s.id: s.kind for s in self.model.spindles}
        if table is not None and getattr(table, "slots", None):
            for slot in table.slots:
                self.numbers.setdefault((slot.number, slot.path), slot.slot)
                self.rotating.setdefault(slot.slot, spindle_kinds.get(slot.spindle_id, "milling") in
                                         ("milling", "live_tool"))
                entry = entries[slot.slot] if slot.slot < len(entries) else None
                if entry is not None and slot.slot not in self.tool_lengths:
                    self.tool_lengths[slot.slot] = self._entry_length(entry)
                if entry is not None:
                    r = num(_get(entry, "radius_offset_m"))
                    if r is None:
                        tool = _get(entry, "tool")
                        geo = _get(tool, "geometry")
                        d = num(_get(geo, "diameter_m"))
                        r = 0.5 * d if d is not None else None
                    if r is not None:
                        self.tool_radius[slot.slot] = r

    def _entry_length(self, entry) -> float:
        """Gauge line to tip: stickout + holder gauge length (tooling's z_gauge), else the length offset set on
        the machine (gauge-line convention), else 0."""
        stick = num(_get(entry, "stickout_m"))
        holder = _get(entry, "holder")
        gauge = num(_get(holder, "gauge_length_m")) if holder is not None and not isinstance(holder, str) else None
        if stick is not None and gauge is not None:
            return stick + gauge
        length = num(_get(entry, "length_offset_m"))
        if length is not None:
            return length
        return 0.0

    def _poses(self, ctx, state, project) -> None:
        P = len(self.model.paths)
        if self._poses_F_W is not None:
            self.poses_F_H = [np.asarray(p, float) for p in self._poses_F_W]
        else:
            self.poses_F_H = [np.eye(4) for _ in range(P)]
        if self._poses_W_S is not None:
            self.poses_W_S = [np.asarray(p, float) for p in self._poses_W_S]
        else:
            self.poses_W_S = [np.eye(4) for _ in range(P)]
        ops = _get(project, "operations") or []
        op_index = int(getattr(getattr(state, "clock", None), "op_index", 0) or 0) if state is not None else 0
        if ops and self._poses_F_W is None:
            op = ops[min(op_index, len(ops) - 1)]
            F_H = op["fixture_pose_F_H"] if "fixture_pose_F_H" in op else np.eye(4)
            W_S = op["stock_pose_W_S"] if "stock_pose_W_S" in op else np.eye(4)
            for p in range(P):
                self.poses_F_H[p] = np.asarray(F_H, float)
                self.poses_W_S[p] = np.asarray(W_S, float)
        self.poses_F_W = [m.copy() for m in self.poses_F_H]

    # ------------------------------------------------------------------ tools and spindles
    def slot_rotating(self, slot: int) -> bool:
        if slot is None or slot < 0:
            return not self.model.lathe
        return bool(self.rotating.get(slot, not self.model.lathe))

    def slot_of(self, number: int, path: int) -> int | None:
        return self.numbers.get((int(number), int(path)))

    def _tip_N(self, slot: int) -> np.ndarray:
        length = float(self.tool_lengths.get(slot, 0.0)) if slot is not None and slot >= 0 else 0.0
        return -length * self.model.tool_axis_N

    def _tool_spindle_index(self, p: int, slot: int) -> int:
        if not self.slot_rotating(slot):
            return -1
        sid = self.model.paths[p].tool_spindle
        return self.model.spindle_index.get(sid, -1) if sid is not None else -1

    def tool_change_axis(self, p: int) -> int | None:
        """The tool-chain linear axis most parallel to the tool axis at home (it goes home for a tool change)."""
        model = self.model
        best = None
        best_dot = 0.5
        poses = self.kin[p].fk(np.array([ax.home for ax in model.axes]))
        z = poses[FK_M_T0][:3, 2]
        for a in self.planners[p].nc_axes if hasattr(self, "planners") else []:
            ax = model.axes[a]
            if ax.rotary or ax.side != CHAIN_TOOL:
                continue
            d = abs(float(ax.direction_M @ z))
            if d > best_dot:
                best, best_dot = a, d
        return best

    def css_axis(self, s: int, p: int):
        """(radius axis index, its value at the spindle centre line) for CSS of spindle s on path p."""
        model = self.model
        letters = model.paths[p].letters
        name = letters.get("X")
        if name is None:
            return None
        a = model.axis_index[name]
        q = np.array([ax.home for ax in model.axes])
        poses = self.kin[p].fk(q)
        tip = poses[FK_M_T0][:3, 3]
        F = poses[FK_M_F]
        origin, zf = F[:3, 3], F[:3, 2]
        r_vec = tip - origin - zf * float((tip - origin) @ zf)
        d = model.axes[a].direction_M
        side = model.axes[a].side
        sign = 1.0 if side == CHAIN_TOOL else -1.0
        along = float(r_vec @ d) * sign
        return a, float(q[a] - along)

    def setter_config(self, p: int, slot: int):
        model = self.model
        setter = _get(model.probes, "tool_setter")
        point = _get(setter, "position_m")
        pt = np.array([float(plain(v)) for v in plain(point)]) if point is not None else np.zeros(3)
        poses = self.kin[p].fk(self.planners[p].st.cur.q)
        normal = poses[FK_M_T0][:3, 2]
        return pt, normal, 0.0, float(self.setter_extra.get(slot, 0.0)) if hasattr(self, "setter_extra") else 0.0

    # ================================================================== lock-step API
    def accept(self, ctx, state, path: int, commands, panel=None) -> None:
        self._bind(ctx, state)
        planner = self.planners[path]
        if panel is not None:
            planner.panel = panel
        if commands:
            planner.accept(list(commands))

    def controller_actions(self, state) -> list:
        ms = state.motion
        out = list(ms.actions)
        ms.actions.clear()
        return [(a["action"], a["path"], a.get("arg"), a["tick"]) if a["action"] != "feedback" else
                ("feedback", a["path"], a["fb"], a["tick"]) for a in out]

    def step_complete(self, state) -> bool:
        return int(state.motion.meta["k_next"]) == 0

    # ------------------------------------------------------------------ kernel inputs
    def kernel_inputs(self, ctx, state, bus):
        self._bind(ctx, state)
        ms = state.motion
        k_from = int(ms.meta["k_next"])
        n = self.macro_ipo
        if k_from == 0:
            ms.meta["tick0"] = int(state.clock.tick)
            self._new_step_buffers(ms)
            self._dynamic_errors(bus)
            self._disturbances(bus)
            self.out = empty_out(n, len(self.model.axes), self.P, max(1, len(self.model.spindles)))
        tick0 = int(ms.meta["tick0"])
        swaps: list = []
        self.chain_now = self._chain_for_step(swaps)     # the tools in the spindles now (planning and FK)
        for p in swaps:
            # the tool tip of path p is the new tool's from tick k_from on (the feed velocity skips the jump)
            ms.step.setdefault("tip_swap", {})[str(p)] = k_from
        setp = self.setp
        env = self.env
        env.sp_actual = ms.spindle
        env.loop_theta0 = ms.loop.theta0
        env.spindle_cmd = [SpindleCmd.from_dict(d) for d in ms.spindle_cmd]
        setp.sync_start[k_from:, :] = 0
        setp.tap[k_from:, :] = 0
        setp.probe[k_from:, :] = 0
        setp.setter[k_from:, :] = 0
        setp.rapid[k_from:, :] = 0
        timeline = SpindleTimeline([c.copy() for c in env.spindle_cmd], [])
        for p, planner in enumerate(self.planners):
            tp = planner.plan(k_from, n, tick0)
            self.plans[p] = tp
            for a, off in tp.c_offset.items():      # a spindle enters C-axis mode at k_from
                ms.servo.c_off[a] = off
            setp.mode[k_from:n, p] = tp.mode[k_from:n]
            setp.rapid[k_from:n, p] = tp.rapid[k_from:n]
            setp.tap[k_from:n, p] = tp.tap[k_from:n]
            setp.probe[k_from:n, p] = tp.probe[k_from:n]
            setp.setter[k_from:n, p] = tp.setter[k_from:n]
            setp.sync_start[k_from:n, p] = tp.sync_start[k_from:n]
            timeline.changes.extend(tp.spindle)
            self._stretch_setpoint(p, tp, planner)
            if tp.setter_cfg is not None:
                pt, nrm, rad, extra = tp.setter_cfg
                setp.setter_point[p] = pt
                setp.setter_normal[p] = nrm
                setp.setter_radius[p] = rad
                setp.setter_extra[p] = extra
        timeline.fill(setp, k_from, n)
        self._css_setpoint()
        return MotionInputs(self.chain_now, self.servo, ms.servo, self.spindle, ms.spindle, setp, ms.loop, self.out,
                            np.array([k_from], np.int32), self.dist_force, self.dist_torque,
                            np.zeros(3, np.int64))

    def _stretch_setpoint(self, p: int, tp, planner) -> None:
        setp = self.setp
        st = tp.stretch
        if st is None:
            return
        setp.sync_spindle[p] = int(st["spindle"])
        theta = st.get("theta") or []
        svals = st.get("s") or []
        T = setp.sync_theta.shape[1]
        if len(theta) > T:
            raise RuntimeError(f"path {p + 1}: synchronous table of {len(theta)} rows exceeds {T}")
        m = len(theta)
        setp.sync_theta[p, :] = 0.0
        setp.sync_s[p, :] = 0.0
        if m:
            setp.sync_theta[p, :m] = theta[:m]
            setp.sync_s[p, :m] = svals[:m]
        setp.sync_n[p] = m
        setp.sync_s_end[p] = float(st["s_end"])
        if "thread" in st:
            lead, trig, tc = st["thread"]
            setp.thread_lead[p] = lead
            setp.thread_trig[p] = trig
            setp.thread_tc[p] = tc
        pieces = st["pieces"]
        G = setp.geo_kind.shape[1]
        if len(pieces) > G:
            raise RuntimeError(f"path {p + 1}: a spindle-synchronous stretch of {len(pieces)} pieces exceeds {G}")
        setp.geo_n[p] = len(pieces)
        for g, j in enumerate(pieces):
            pc = planner.piece(j)
            setp.geo_kind[p, g] = pc.geo
            setp.geo_s0[p, g] = st["s0"][g]
            setp.geo_len[p, g] = pc.length
            setp.geo_vec[p, g] = pc.vec
            setp.geo_par[p, g] = pc.par

    def _css_setpoint(self) -> None:
        setp = self.setp
        setp.css_rmin[0] = self.model.p["css_min_radius_m"]
        for s in range(len(self.model.spindles)):
            owner = self.env.spindle_cmd[s].owner
            p = owner if owner >= 0 else 0
            info = self.env.css_axis_info(s, p)
            if info is not None:
                setp.css_axis[s] = info[0]
                setp.css_centre[s] = info[1]

    def _chain_for_step(self, swaps: list | None = None):
        chain = self.chain
        for p in range(self.P):
            slot = self.planners[p].st.tool_slot
            if slot != self.tool_slot[p]:
                if swaps is not None:
                    swaps.append(p)
                self.tool_slot[p] = slot
                tip = self._tip_N(slot)
                for ch in (chain, self.chain_nominal):
                    ch.pose_N_T0[p, :3, 3] = tip
                    ch.tool_spindle[p] = self._tool_spindle_index(p, slot)
                self.kin[p].bind(self.chain_nominal)
        return chain

    # ------------------------------------------------------------------ bus inputs
    def _dynamic_errors(self, bus) -> None:
        """Thermal drift (6-component link errors, screw strain), spindle growth, structure drift and part pose
        deltas from bus_prev, as per-link error transforms (never axis offsets)."""
        chain = self.chain
        model = self.model
        err = self.static_err.copy()
        strain = np.zeros_like(chain.strain)
        strain_ref = np.zeros_like(chain.strain_ref)
        thermal = getattr(bus, "thermal", None) if bus is not None else None
        holding = getattr(bus, "holding", None) if bus is not None else None
        q_now = self.planners[0].st.cur.q if hasattr(self, "planners") else np.zeros(len(model.axes))
        for p in range(self.P):
            for c in (CHAIN_WORK, CHAIN_TOOL):
                for i in range(int(chain.n_links[p, c])):
                    a = int(chain.axis[p, c, i])
                    if a < 0 or thermal is None:
                        continue
                    name = model.axes[a].name
                    if name in (thermal.axis_drift_strain or {}):
                        strain[p, c, i] = float(thermal.axis_drift_strain[name])
                        strain_ref[p, c, i] = float(thermal.axis_drift_ref_m.get(name, 0.0))
                        vec = thermal.axis_drift.get(name) if thermal.axis_drift else None
                        if vec is not None and len(vec) == 6:
                            err[p, c, i, 3:6] += np.asarray(vec, float)[3:6]
                    elif thermal.axis_drift and name in thermal.axis_drift:
                        vec = np.asarray(thermal.axis_drift[name], float)
                        if len(vec) == 6:
                            err[p, c, i] += vec
            if thermal is None:
                continue
            q = self.planners[p].st.cur.q if hasattr(self, "planners") else q_now
            # spindle growth: tool spindles at the nose link, work spindles at their rotor link
            for sid, g in (thermal.spindle_growth_M_m or {}).items():
                sp = model.spindle(sid)
                if sp is None:
                    continue
                g = np.asarray(g, float)
                if sp.kind in ("milling", "live_tool") and model.paths[p].tool_spindle == sid:
                    i = int(chain.n_links[p, CHAIN_TOOL]) - 1
                    R = self._parent_rotation(p, CHAIN_TOOL, i, q)
                    err[p, CHAIN_TOOL, i, 0:3] += R.T @ g
                elif sp.kind in ("turning", "sub") and int(chain.work_spindle_link[p]) >= 0 and \
                        model.paths[p].work_spindle == sid:
                    i = int(chain.work_spindle_link[p])
                    R = self._parent_rotation(p, CHAIN_WORK, i, q)
                    err[p, CHAIN_WORK, i, 0:3] += R.T @ g
            drift = np.asarray(getattr(thermal, "structure_drift_M_m", np.zeros(3)), float)
            if drift.shape == (3,) and np.any(drift != 0.0) and int(chain.n_links[p, CHAIN_TOOL]) > 0:
                err[p, CHAIN_TOOL, 0, 0:3] += drift
        chain.err_const[...] = err
        chain.strain[...] = strain
        chain.strain_ref[...] = strain_ref
        # part pose delta (holding) moves W in F; pull-out and growth only lengthen the tool seen by the setter
        self.setter_extra = {}
        for p in range(self.P):
            pose = self.poses_F_H[p]
            w = self.model.paths[p].workpiece
            part = holding.parts.get(w) if holding is not None and holding.parts else None
            if part is not None and getattr(part, "pose_delta", None) is not None:
                pose = pose @ np.asarray(part.pose_delta, float)
            chain.pose_F_W[p] = pose
        if holding is not None:
            for slot, h in (holding.tools or {}).items():
                self.setter_extra[int(slot)] = float(getattr(h, "pullout_m", 0.0) or 0.0)
        if thermal is not None:
            for slot, t in (thermal.tools or {}).items():
                self.setter_extra[int(slot)] = self.setter_extra.get(int(slot), 0.0) + \
                    float(getattr(t, "length_growth_m", 0.0) or 0.0) + float(getattr(t, "holder_growth_m", 0.0) or 0.0)

    def _parent_rotation(self, p, c, i, q) -> np.ndarray:
        from .kernels import link_motion
        R = np.eye(4)
        m = np.zeros((4, 4))
        r = np.zeros((4, 4))
        t = np.zeros((4, 4))
        for k in range(i):
            link_motion(self.chain, p, c, k, np.ascontiguousarray(q, dtype=float), -1, m, r, t)
            R = R @ m
        return R[:3, :3]

    def _disturbances(self, bus) -> None:
        """Cutting load of the previous step's last tick as the first tick's disturbance (one-tick lag)."""
        NA = len(self.model.axes)
        S = max(1, len(self.model.spindles))
        self.dist_force = np.zeros(NA)
        self.dist_torque = np.zeros(S)
        cutting = getattr(bus, "cutting", None) if bus is not None else None
        motion = getattr(bus, "motion", None) if bus is not None else None
        if cutting is None or not cutting.contacts or motion is None:
            return
        frames = {f.path: f for f in motion.paths}
        removal = getattr(bus, "removal", None)
        path_of = {}
        for rc in (removal.contacts if removal is not None else []):
            path_of[rc.contact] = rc.path
        for cc in cutting.contacts:
            p = path_of.get(cc.contact, 0)
            frame = frames.get(p)
            forces = np.asarray(cc.axis_force_N, float)
            if frame is not None and forces.ndim == 2 and forces.shape[0] > 0:
                for col, name in enumerate(frame.axis_names):
                    if col < forces.shape[1] and name in self.model.axis_index:
                        self.dist_force[self.model.axis_index[name]] += float(forces[-1, col])
            torque = np.asarray(cc.spindle_torque_Nm, float)
            if torque.size and p < self.P:
                sid = self.planners[p].cut_spindle(self.planners[p].st)
                s = self.model.spindle_index.get(sid) if sid is not None else None
                if s is not None:
                    omega = float(self._ms_spindle_omega(s))
                    self.dist_torque[s] += float(torque[-1]) * (1.0 if omega >= 0.0 else -1.0)

    def _ms_spindle_omega(self, s: int) -> float:
        return float(self._state.motion.spindle.omega[s]) if self._state is not None else 0.0

    # ------------------------------------------------------------------ absorbing executed ticks
    def absorb(self, ctx, state, outputs, bus_next) -> None:
        self._bind(ctx, state)
        ms = state.motion
        out = outputs.out
        code, k_stop, p_sync = (int(v) for v in outputs.result)
        k_from = int(ms.meta["k_next"])
        n = self.macro_ipo
        if k_stop < k_from:
            k_stop = k_from
        tick0 = int(ms.meta["tick0"])
        sync_any = code != SYNC_NONE
        for p, planner in enumerate(self.planners):
            tp = self.plans[p]
            code_p = int(out.sync[k_stop, p]) if sync_any else 0
            events = planner.commit(tp, k_stop, code_p, out)
            by_boundary: dict[int, list] = {}
            for ev in events:
                by_boundary.setdefault(int(ev[0]), []).append(ev)
            for b in range(k_from, k_stop + 2):
                if b in by_boundary:
                    self._events(ctx, state, p, by_boundary.pop(b), out, tick0, k_stop)
                if b <= k_stop:
                    self._book(ctx, state, p, tp, b, b, out, tick0)
            for b in sorted(by_boundary):
                self._events(ctx, state, p, by_boundary[b], out, tick0, k_stop)
            if code_p == SYNC_ALARM and not ms.episodes.get(f"alarm{p}"):
                ms.episodes[f"alarm{p}"] = True
                self._alarm(ctx, state, p, out, k_stop, tick0)
        ms.spindle_cmd = [c.to_dict() for c in self._committed_spindles()]
        self._collect(ms, out, k_from, k_stop)
        self._overloads(ctx, state, out, k_from, k_stop, tick0)
        ms.meta["k_next"] = k_stop + 1
        if ms.meta["k_next"] >= n:
            self._publish(ctx, state, bus_next, tick0)
            ms.meta["k_next"] = 0
            ms.meta["macro"] = int(ms.meta["macro"]) + 1

    def _committed_spindles(self) -> list:
        cmds = [SpindleCmd.from_dict(d) for d in self._state.motion.spindle_cmd]
        for planner in self.planners:
            for s in planner.st.sp_changed:
                if s < len(cmds):
                    cmds[s] = planner.st.sp[s].copy()
            planner.st.sp_changed = []
            planner.st.sp = [c.copy() for c in cmds]
        return cmds

    def _book(self, ctx, state, p: int, tp, k_from: int, k_stop: int, out, tick0: int) -> None:
        """Every executed L1 tick of path p into the timeline record of the command owning it."""
        ms = state.motion
        planner = self.planners[p]
        ipo = self.ipo_ticks
        stretch = tp.stretch
        for k in range(k_from, k_stop + 1):
            owner = int(tp.owner[k])
            cat = tp.cat[k]
            if stretch is not None and k >= stretch["i"] and tp.mode[k] in (SP_PER_REV, SP_THREAD):
                owner = self._stretch_owner(planner, stretch, float(out.sync_s[k, p]), owner)
            boundary = tick0 + k * ipo
            if ms.ledger["start"][p] is None and owner >= 0:
                ms.ledger["start"][p] = boundary
            blocks = ms.step.setdefault("block", {}).setdefault(str(p), [])
            if owner < 0:
                if ms.idle_from[p] < 0 and ms.ledger["start"][p] is not None and not planner.st.ended:
                    ms.idle_from[p] = boundary
                    ms.meta.setdefault("idle_cat", {})[str(p)] = cat
                blocks.append(int(ms.open_record[p]) if ms.open_record[p] >= 0 else len(self.timeline.records) - 1)
                continue
            rec = self._record_for(ms, p, planner, owner, boundary)
            blocks.append(int(rec.index))
            if int(tp.lim[k]) == 2 and cat in (CAT_FEED, CAT_RAPID):
                rec.starve_ticks += 1
            if cat == CAT_FEED:
                if self.contact_from_loop and self.settings.contact_from_loop:
                    rec.book(CAT_CUT if int(out.in_cut[k, p]) else CAT_AIR, ipo)
                else:
                    rec.pending_feed[int(ms.meta["macro"])] = rec.pending_feed.get(int(ms.meta["macro"]), 0) + ipo
                rec.motion_ticks += ipo
                rec.motion_cat = CAT_FEED
            else:
                rec.book(cat, ipo)
                if cat == CAT_RAPID:
                    rec.motion_ticks += ipo
                    rec.motion_cat = CAT_RAPID
            ms.ledger["end"][p] = boundary + ipo

    def _stretch_owner(self, planner, stretch, s_act: float, default: int) -> int:
        s0 = stretch["s0"]
        r = 0
        for j in range(len(stretch["pieces"])):
            if s_act >= s0[j] - 1e-15:
                r = j
        try:
            return planner.piece(stretch["pieces"][r]).item
        except IndexError:
            return default

    def _record_for(self, ms, p: int, planner, owner: int, boundary: int) -> Record:
        it = planner.item(owner)
        seq = it.cmd.block_seq if it is not None else -1
        open_index = ms.open_record[p]
        rec = self.timeline.records[open_index] if open_index >= 0 else None
        if rec is not None and rec.block_seq == seq and rec.tick1 < 0:
            self._flush_idle(ms, p, rec, boundary)
            return rec
        start = ms.idle_from[p] if ms.idle_from[p] >= 0 else boundary
        if rec is not None and rec.tick1 < 0:
            self._close(ms, rec, start)
        ref = it.cmd.ref if it is not None else None
        op = str(ms.meta.get("op_name", ""))
        new = self.timeline.open(p, seq, ref, self._block_kind(planner, it), start, op, int(ms.meta.get("part", 0)),
                                 planner.st.tool_number)
        if it is not None and it.cmd.kind == Kind.STOP:
            new.stop = it.cmd.stop
        new.pumps_on = sorted(k for k, v in planner.st.pumps.items() if v)
        ms.open_record[p] = new.index
        self._flush_idle(ms, p, new, boundary)
        ms.ledger["blocks"][p] += 1
        return new

    @staticmethod
    def _block_kind(planner, it) -> str:
        """Kind of an executed block: its motion command, else its first command that is not a mode change."""
        if it is None:
            return "mode"
        kinds = [x.cmd.kind for x in planner.items if x.cmd.block_seq == it.cmd.block_seq and x.cmd.path == it.cmd.path]
        for k in kinds:
            if k in (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE, Kind.TOOL_SETTER):
                return k.value
        for k in kinds:
            if k != Kind.MODE:
                return k.value
        return it.cmd.kind.value

    def _flush_idle(self, ms, p: int, rec: Record, boundary: int) -> None:
        if ms.idle_from[p] >= 0:
            cat = ms.meta.get("idle_cat", {}).get(str(p), "path_wait")
            rec.book(cat, boundary - ms.idle_from[p])
            ms.idle_from[p] = -1

    def _close(self, ms, rec: Record, tick1: int) -> None:
        rec.tick1 = int(tick1)
        planner = self.planners[rec.path]
        ideal = 0.0
        for it in planner.items:
            if it.cmd.block_seq == rec.block_seq and it.cmd.path == rec.path:
                ideal += self.item_ideal.pop((rec.path, it.index), it.ideal_s)
        rec.ideal_ticks = int(round(ideal / self.tick_s))
        split_motion(rec, self.ipo_ticks)
        # the categories must cover the block exactly
        total = sum(rec.ticks.values()) + sum(rec.pending_feed.values())
        span = rec.tick1 - rec.tick0
        if total != span:
            rec.book("unclassified" if total < span else "overbooked", span - total if total < span else 0)

    # ------------------------------------------------------------------ events
    def _events(self, ctx, state, p: int, events: list, out, tick0: int, k_stop: int) -> None:
        ms = state.motion
        planner = self.planners[p]
        ipo = self.ipo_ticks
        for b, kind, data in events:
            boundary = tick0 + b * ipo
            if kind == "start":
                it = planner.item(data)
                if it is not None:
                    self._record_for(ms, p, planner, it.index, boundary)
                    self._tcp_findings(ctx, state, p, it, boundary)
            elif kind == "done":
                ms.actions.append({"action": "retire", "path": p, "arg": 1, "tick": boundary})
            elif kind == "ideal":
                index, ideal = data
                self.item_ideal[(p, int(index))] = float(ideal)
            elif kind == "sync_feedback":
                it = planner.item(data)
                fb = Feedback(boundary * self.tick_s, self._machine_pos(p, out, min(b, k_stop)))
                ms.actions.append({"action": "feedback", "path": p, "fb": fb, "tick": boundary})
            elif kind == "probe_feedback":
                it = planner.item(data)
                ms.actions.append({"action": "feedback", "path": p, "fb": self._probe_feedback(p, it, out, b, tick0),
                                   "tick": boundary})
            elif kind == "cycle_start":
                ms.actions.append({"action": "cycle_start", "path": p, "arg": None, "tick": boundary})
            elif kind == "program_end":
                ms.actions.append({"action": "program_end", "path": p, "arg": None, "tick": boundary})
                ms.ledger["end"][p] = boundary
                rec = self.timeline.records[ms.open_record[p]] if ms.open_record[p] >= 0 else None
                if rec is not None and rec.tick1 < 0:
                    self._close(ms, rec, boundary)
                ms.open_record[p] = -1
            elif kind in ("tool", "coolant", "spindle", "trigger"):
                ms.step.setdefault("events", {}).setdefault(str(p), []).append(
                    {"tick": boundary, "kind": kind, **({k: v for k, v in data.items()} if isinstance(data, dict)
                                                         else {"item": data})})

    def _machine_pos(self, p: int, out, k: int) -> dict:
        return {self.model.axes[a].name: float(out.actual[k, a]) for a in self.planners[p].nc_axes}

    def _probe_feedback(self, p: int, it, out, b: int, tick0: int) -> Feedback:
        k = min(max(b - 1, 0), self.macro_ipo - 1)
        time_s = (tick0 + b * self.ipo_ticks) * self.tick_s
        pos = self._machine_pos(p, out, k)
        fb = Feedback(time_s, pos)
        latched = dict(it.latched) if it.triggered and it.latched else dict(pos)
        if it.triggered:
            pre = float(num(_get(_get(self.model.probes, "spindle_probe"), "pretravel_m"),
                            self.model.p["probe_pretravel_m"]))
            pc = it.cmd.target or {}
            # latched position includes the stylus pre-travel along the move
            if it.cmd.kind == Kind.PROBE and pre > 0.0:
                planner = self.planners[p]
                start = planner.piece(it.pieces[0]).q0 if it.pieces else None
                if start is not None:
                    d = np.array([float(pc.get(self.model.axes[a].name, start[a])) - start[a]
                                  for a in planner.nc_axes])
                    nd = float(np.linalg.norm(d))
                    if nd > 0:
                        for c, a in enumerate(planner.nc_axes):
                            latched[self.model.axes[a].name] += pre * d[c] / nd
        fb.probe = ProbeResult(bool(it.triggered), latched, time_s)
        if it.cmd.kind == Kind.TOOL_SETTER:
            slot = self.planners[p].st.tool_slot
            length = float(self.tool_lengths.get(slot, 0.0)) + float(getattr(self, "setter_extra", {}).get(slot, 0.0))
            number = self.planners[p].st.tool_number
            fb.setter = SetterResult(int(number) if number is not None else -1,
                                     length if it.triggered else None, None, not it.triggered, time_s)
        return fb

    def _alarm(self, ctx, state, p: int, out, k: int, tick0: int) -> None:
        ms = state.motion
        kind = int(out.alarm[k, p])
        a = int(out.alarm_axis[k, p])
        name = self.model.axes[a].name if a >= 0 else "?"
        tick = tick0 + (k + 1) * self.ipo_ticks
        ax = self.model.axes[a] if a >= 0 else None
        if kind in (ALARM_OVERTRAVEL_POS, ALARM_OVERTRAVEL_NEG):
            limit = ax.hi if kind == ALARM_OVERTRAVEL_POS else ax.lo
            code, text = self._alarm_text("overtravel", "+" if kind == ALARM_OVERTRAVEL_POS else "-")
            msg = f"overtravel: axis {name} commanded {out.cmd[k, a]:.6g} beyond its travel limit {limit:.6g}"
            if ctx is not None:
                ctx.finding(state, "motion", Category.OVERTRAVEL, Severity.ERROR, msg, tick, path=p,
                            quantities={"value": float(out.cmd[k, a]), "limit": float(limit),
                                        "axis_index": float(a)})
        else:
            code, text = self._alarm_text("servo", "")
            msg = f"excess following error on axis {name}: {out.ferr[k, a]:.6g}"
            if ctx is not None:
                ctx.finding(state, "motion", Category.ALARM, Severity.ERROR, msg, tick, path=p,
                            quantities={"following_error": float(out.ferr[k, a]), "axis_index": float(a)})
        ms.actions.append({"action": "feedback", "path": p,
                           "fb": Feedback(tick * self.tick_s, self._machine_pos(p, out, k),
                                          alarm={"code": code, "text": text}), "tick": tick})

    def _alarm_text(self, kind: str, sign: str) -> tuple[str, str]:
        """The machine alarm a control shows (controller database alarms with trigger overtravel / servo)."""
        db = getattr(self, "_controller_db", None)
        if db is not None:
            for alarm in getattr(db, "alarms", None) or []:
                trig = str(plain(getattr(alarm, "trigger", "")) or "")
                text = str(plain(alarm.text) or "")
                if trig == kind and (not sign or text.startswith(sign)):
                    return str(plain(alarm.code)), text
        if kind == "overtravel":
            return ("OVERTRAVEL" + sign), f"{sign} OVERTRAVEL (SOFT)"
        return "SERVO", "EXCESS ERROR"

    def _tcp_findings(self, ctx, state, p: int, it, boundary: int) -> None:
        if ctx is None or not it.pieces:
            return
        planner = self.planners[p]
        for j in it.pieces:
            pc = planner.piece(j)
            if pc.kind == PK_TIP and pc.singular:
                behaviour = "slow_down"
                for sing in self.model.singularities:
                    behaviour = sing["behaviour"]
                sev = Severity.ERROR if behaviour == "alarm" else Severity.WARNING
                ctx.finding(state, "motion", Category.SINGULARITY, sev,
                            f"TCP move passes a kinematic singularity (tool axis along a rotary axis): {behaviour}",
                            boundary, path=p, quantities={"length_m": float(pc.length)})
                if behaviour == "alarm":
                    state.motion.actions.append({"action": "feedback", "path": p,
                                                 "fb": Feedback(boundary * self.tick_s, {},
                                                                alarm={"code": "SINGULARITY",
                                                                       "text": "TCP SINGULARITY"}),
                                                 "tick": boundary})
                break

    def _overloads(self, ctx, state, out, k_from: int, k_stop: int, tick0: int) -> None:
        ms = state.motion
        if ctx is None:
            return
        rows = out.overload[k_from:k_stop + 1]
        for a in range(rows.shape[1]):
            hits = np.nonzero(rows[:, a])[0]
            key = str(a)
            active = ms.episodes.get(key, False)
            if hits.size and not active:
                k = k_from + int(hits[0])
                ax = self.model.axes[a]
                force = float(out.force[k, a])
                ctx.finding(state, "motion", Category.AXIS_OVERLOAD, Severity.WARNING,
                            f"axis {ax.name}: drive force {abs(force):.4g} demanded, limit {ax.force_max:.4g}",
                            tick0 + (k + 1) * self.ipo_ticks, path=ax.path,
                            quantities={"force_N": force, "limit_N": float(ax.force_max)})
                ms.episodes[key] = True
            elif not hits.size and active:
                ms.episodes[key] = False

    # ------------------------------------------------------------------ the step's PathFrames
    def _new_step_buffers(self, ms) -> None:
        ms.step = {"events": {}}

    def _collect(self, ms, out, k_from: int, k_stop: int) -> None:
        ms.step["k_stop"] = k_stop

    def _publish(self, ctx, state, bus_next, tick0: int) -> None:
        ms = state.motion
        out = self.out
        n = self.macro_ipo
        model = self.model
        ipo = self.ipo_ticks
        ticks = tick0 + (np.arange(n, dtype=np.int64) + 1) * ipo
        frames = []
        for p, planner in enumerate(self.planners):
            names = [model.axes[a].name for a in planner.nc_axes]
            cols = planner.nc_axes
            sid = planner.cut_spindle(planner.st)
            s = model.spindle_index.get(sid, -1) if sid is not None else -1
            angle = out.sp_angle[:, s] if s >= 0 else np.zeros(n)
            omega = out.sp_omega[:, s] if s >= 0 else np.zeros(n)
            torque = out.sp_torque[:, s] if s >= 0 else np.zeros(n)
            poses = out.poses[p]
            R_M_W0 = poses[FK_M_W0][:3, :3]
            w0 = out.tip_W0[:, p, :]
            prev = np.vstack([w0[:1], w0[:-1]])
            if int(ms.meta["macro"]) > 0 and "tip_W0_last" in ms.meta and str(p) in ms.meta["tip_W0_last"]:
                prev[0] = np.asarray(ms.meta["tip_W0_last"][str(p)], float)
            feed = ((w0 - prev) / self.dt) @ R_M_W0.T
            swap = ms.step.get("tip_swap", {}).get(str(p))
            if swap is not None:
                # a tool change moved the tip to the new tool's point between two ticks: that jump (its length
                # difference, 102 mm in one 1 ms tick = 102 m/s for the convergence case's D10) is no motion of
                # the tool, which stands at the tool-change position; the tick takes the feed velocity of the tick
                # before (velocity changes by at most A dt per tick). Before 2026-11-09 the jump went to the bus
                # and chips sized its occupancy grid by it (2 ms macro steps: 8e7 voxels, beyond 1.8 GB)
                k_s = int(swap)
                if 0 < k_s < n:
                    feed[k_s] = feed[k_s - 1]
                elif k_s == 0:
                    last = ms.meta.get("feed_last", {}).get(str(p))
                    feed[0] = np.asarray(last, float) if last is not None else 0.0
            ms.meta.setdefault("tip_W0_last", {})[str(p)] = w0[-1].tolist()
            ms.meta.setdefault("feed_last", {})[str(p)] = feed[-1].tolist()
            rot = self._tool_angle(p, out, s)
            pose_M_T = poses[FK_M_T0] @ _rz(rot)
            pose_S_T = poses[FK_S_T0] @ _rz(rot)
            pose_W0_T = poses[FK_W0_T0] @ _rz(rot)
            wl = int(self.chain.work_spindle_link[p])
            work_angle = np.zeros(n)
            if wl >= 0:
                a = int(self.chain.axis[p, CHAIN_WORK, wl])
                work_angle = out.actual[:, a].copy() if a >= 0 else work_angle
            blocks = ms.step.get("block", {}).get(str(p), [])
            block_index = np.array(blocks[:n] + [blocks[-1] if blocks else -1] * (n - len(blocks)), np.int32)
            events = list(ms.step.get("events", {}).get(str(p), []))
            frame = PathFrame(path=p, tick0=tick0, ipo_ticks=ipo, tick=ticks.copy(), block_index=block_index,
                              tip_S_m=out.tip_S[:, p, :].copy(), tip_M_m=out.tip_M[:, p, :].copy(),
                              axis_T_M=out.axis_M[:, p, :].copy(), axis_names=tuple(names),
                              axis_actual=out.actual[:, cols].copy(), axis_cmd=out.cmd[:, cols].copy(),
                              following_error=out.ferr[:, cols].copy(),
                              spindle_angle_rad=np.mod(angle, TWO_PI),
                              spindle_rev=np.floor(angle / TWO_PI).astype(np.int64), spindle_omega_rad_s=omega.copy(),
                              feed_vel_M_m_s=feed, rapid=self.setp.rapid[:, p].astype(bool).copy(),
                              tool_slot=int(planner.st.tool_slot), spindle_id=str(sid or ""),
                              workpiece=int(model.paths[p].workpiece), pose_M_T=pose_M_T, pose_S_T=pose_S_T,
                              pose_W0_T=pose_W0_T, coolant_pumps=dict(planner.st.pumps),
                              nozzle_settings=dict(planner.st.nozzles), events=events,
                              axis_inertial_force_N=out.inertial[:, cols].copy())
            # additive fields (docs/models/motion.md "Bus record")
            frame.axis_force_N = out.force[:, cols].copy()
            frame.spindle_torque_Nm = torque.copy()
            frame.work_angle_rad = work_angle
            frame.tip_W0_m = w0.copy()
            frame.phase = int(planner.st.cur.phase)
            frames.append(frame)
        bus_next.motion = MotionFrame(time_s=(tick0 + n * ipo) * self.tick_s, tick=int(tick0 + n * ipo), paths=frames)
        self._ledger_to_state(state)
        self.last_channels = self.channel_values(out, frames)

    def _tool_angle(self, p: int, out, s: int) -> float:
        if s < 0 or int(self.chain.tool_spindle[p]) < 0:
            return 0.0
        return float(out.sp_angle[-1, s])

    def _ledger_to_state(self, state) -> None:
        ms = state.motion
        led = state.ledger
        best = 0
        best_blocks = 0
        for p in range(self.P):
            start, end = ms.ledger["start"][p], ms.ledger["end"][p]
            if start is None or end is None:
                continue
            span = int(end - start)
            booked = 0
            for rec in self.timeline.records:
                if rec.path != p:
                    continue
                stop = rec.tick1 if rec.tick1 >= 0 else int(end)
                booked += stop - rec.tick0
            if span >= best:
                best, best_blocks = span, booked
        led.cycle_time_ticks = int(best)
        led.block_time_ticks = int(best_blocks)

    def channel_values(self, out, frames) -> dict:
        n = self.macro_ipo
        NA = len(self.model.axes)
        pad = lambda arr, width: np.hstack([arr, np.full((arr.shape[0], width - arr.shape[1]), np.nan)]) \
            if arr.shape[1] < width else arr[:, :width]
        values = {
            "motion.axis_actual": pad(out.actual, MAX_AXES), "motion.axis_cmd": pad(out.cmd, MAX_AXES),
            "motion.following_error": pad(out.ferr, MAX_AXES), "motion.axis_force_N": pad(out.force, MAX_AXES),
            "motion.tip_M_m": out.tip_M[:, 0, :].copy(),
            "motion.spindle_omega_rad_s": pad(out.sp_omega, MAX_SPINDLES),
            "motion.spindle_torque_Nm": pad(out.sp_torque, MAX_SPINDLES),
        }
        speed = np.full((n, MAX_PATHS), np.nan)
        for f in frames:
            if f.path < MAX_PATHS:
                speed[:, f.path] = np.linalg.norm(f.feed_vel_M_m_s, axis=1)
        values["motion.path_speed_m_s"] = speed
        queued = np.full(MAX_PATHS, np.nan)
        phase = np.zeros(MAX_PATHS, np.int8)
        cycle = np.full(MAX_PATHS, np.nan)
        ms = self._state.motion
        for p, planner in enumerate(self.planners[:MAX_PATHS]):
            queued[p] = planner.queued()
            phase[p] = planner.st.cur.phase
            if ms.ledger["start"][p] is not None and ms.ledger["end"][p] is not None:
                cycle[p] = (ms.ledger["end"][p] - ms.ledger["start"][p]) * self.tick_s
        values["motion.queued_commands"] = queued
        values["motion.phase"] = phase
        values["motion.cycle_time_s"] = cycle
        return values

    # ------------------------------------------------------------------ slow step
    def macro_step(self, ctx, state, prev, nxt) -> None:
        """Feed time of the previous step split into cutting / air cutting by removal's ToolContact (when the
        hot loop gave no per-tick contact flags)."""
        self._bind(ctx, state)
        ms = state.motion
        step = int(ms.meta["macro"]) - 1
        if step < 0 or prev is None:
            return
        removal = getattr(prev, "removal", None)
        for p in range(self.P):
            recs = [r for r in self.timeline.records if r.path == p and step in r.pending_feed]
            if not recs:
                continue
            feed_ticks = [r.pending_feed[step] for r in recs]
            total = sum(feed_ticks)
            air = total
            if removal is not None:
                contacts = [c for c in removal.contacts if c.path == p]
                if contacts:
                    in_cut = any(c.geometry.in_cut for c in contacts)
                    air_s = max(c.geometry.air_cut_time_s for c in contacts)
                    air = int(round(air_s / self.tick_s)) if in_cut else total
                    air = min(max(air, 0), total)
            shares = split_largest_remainder(air, feed_ticks)
            for rec, ticks_total, a in zip(recs, feed_ticks, shares):
                rec.book(CAT_AIR, a)
                rec.book(CAT_CUT, ticks_total - a)
                del rec.pending_feed[step]
        # records left with pending classification older than one step (no removal record came): air
        for rec in self.timeline.records:
            for s_old in [s for s in rec.pending_feed if s < step]:
                rec.book(CAT_AIR, rec.pending_feed.pop(s_old))

    def end_operation(self, ctx, state) -> None:
        self._bind(ctx, state)
        ms = state.motion
        for p in range(self.P):
            idx = ms.open_record[p]
            if idx >= 0 and self.timeline.records[idx].tick1 < 0:
                end = ms.ledger["end"][p] or self.timeline.records[idx].tick0
                self._close(ms, self.timeline.records[idx], int(end))
                ms.open_record[p] = -1
        for rec in self.timeline.records:
            for step in list(rec.pending_feed):
                rec.book(CAT_AIR, rec.pending_feed.pop(step))

    def finalize(self, ctx, state) -> dict:
        self._bind(ctx, state)
        self.end_operation(ctx, state)
        ms = state.motion
        out = {"paths": [], "estimated": list(self.estimated), "timeline_blocks": len(self.timeline.records)}
        for p in range(self.P):
            start, end = ms.ledger["start"][p], ms.ledger["end"][p]
            cats: dict = {}
            for rec in self.timeline.records:
                if rec.path == p:
                    for k, v in rec.ticks.items():
                        cats[k] = cats.get(k, 0) + v
            span = int(end - start) if start is not None and end is not None else 0
            out["paths"].append({"path": p + 1, "cycle_time_s": span * self.tick_s,
                                 "categories_s": {k: v * self.tick_s for k, v in sorted(cats.items())},
                                 "blocks": ms.ledger["blocks"][p]})
        return out

    def timeline_json(self) -> dict:
        return self.timeline.to_json(self.tick_s)

    # ------------------------------------------------------------------ checkpoints
    def attach(self, ctx, state) -> None:
        """Bind this domain object to a state built elsewhere (a resumed checkpoint)."""
        self.bound = False
        self._bind(ctx, state)

    def _bind(self, ctx, state) -> None:
        """Bind to state.motion (after a checkpoint restore the planners are rebuilt from its data)."""
        self._state = state
        ms = state.motion
        if getattr(ms, "planners", None) is not None and self.bound:
            return
        if not hasattr(self, "model"):
            self._build(ctx, state)
        self._restore(ms)
        self.bound = True

    def snapshot(self, state) -> None:
        """Write the live planner/timeline data into state.motion (before to_arrays)."""
        ms = state.motion
        ms.paths = [self._planner_dict(pl) for pl in self.planners]
        ms.timeline = [r.state_dict() for r in self.timeline.records]
        ms.meta["tool_slot"] = list(self.tool_slot)
        ms.meta["item_ideal"] = [[int(p), int(i), float(v)] for (p, i), v in sorted(self.item_ideal.items())]
        ms.meta["dist_force"] = self.dist_force.tolist() if hasattr(self, "dist_force") else None
        ms.meta["dist_torque"] = self.dist_torque.tolist() if hasattr(self, "dist_torque") else None
        ms.actions_json = [dict(a, fb=feedback_to_dict(a["fb"])) if a.get("action") == "feedback" else dict(a)
                           for a in ms.actions]

    def _planner_dict(self, pl: PathPlanner) -> dict:
        return {"items": [{"index": it.index, "cmd": command_to_dict(it.cmd), "pieces": list(it.pieces),
                           "inpos": it.inpos, "dog": [[a, q0, q1, [pr.distance, pr.Tj, pr.Ta, pr.Tv, pr.J]]
                                                       for a, q0, q1, pr in it.dog] if it.dog is not None else None,
                           "feed_kind": it.feed_kind, "ideal_s": it.ideal_s, "length_m": it.length_m,
                           "triggered": it.triggered, "latched": it.latched, "latched_tick": it.latched_tick,
                           "needs_rebuild": it.needs_rebuild, "built_slot": it.built_slot,
                           "c_spindles": list(it.c_spindles)} for it in pl.items],
                "tail_slot": pl._tail_slot, "tail_caxis": sorted(pl._tail_caxis),
                "item0": pl.item0, "piece0": pl.piece0, "pieces": [piece_to_dict(pc) for pc in pl.pieces],
                "tail_q": pl.tail_q.tolist(), "last_target": dict(pl.last_target), "expect_more": pl.expect_more,
                "st": pl.st.to_dict(), "stretch_info": pl.stretch_info}

    def _restore(self, ms) -> None:
        from .planner import Item
        from .scurve import DoubleS
        q0 = np.array([ax.home for ax in self.model.axes])
        self.planners = [PathPlanner(self.env, p, q0) for p in range(self.P)]
        for pl, d in zip(self.planners, ms.paths):
            pl.items = []
            for it in d["items"]:
                item = Item(int(it["index"]), command_from_dict(it["cmd"]), list(it["pieces"]), bool(it["inpos"]),
                            [(int(a), float(q0_), float(q1_), DoubleS(*pr)) for a, q0_, q1_, pr in it["dog"]]
                            if it["dog"] is not None else None, it["feed_kind"], float(it["ideal_s"]),
                            float(it["length_m"]), bool(it["triggered"]), it["latched"], int(it["latched_tick"]),
                            bool(it["needs_rebuild"]), int(it.get("built_slot", -1)),
                            [int(s) for s in it.get("c_spindles", [])])
                pl.items.append(item)
            pl._tail_slot = int(d.get("tail_slot", -1))
            pl._tail_caxis = {int(s) for s in d.get("tail_caxis", [])}
            pl.item0 = int(d["item0"])
            pl.piece0 = int(d["piece0"])
            pl.pieces = [piece_from_dict(x) for x in d["pieces"]]
            pl.tail_q = np.array(d["tail_q"], dtype=float)
            pl.last_target = {k: float(v) for k, v in d["last_target"].items()}
            pl.expect_more = bool(d["expect_more"])
            pl.st = PlanState.from_dict(d["st"])
            pl.stretch_info = d.get("stretch_info")
        self.timeline = Timeline()
        self.timeline.records = [Record.from_dict(r) for r in ms.timeline]
        self.env.spindle_cmd = [SpindleCmd.from_dict(c) for c in ms.spindle_cmd]
        self.tool_slot = list(ms.meta.get("tool_slot", [-1] * self.P))
        self.item_ideal = {(int(p), int(i)): float(v) for p, i, v in ms.meta.get("item_ideal", [])}
        for p in range(self.P):
            self.chain.pose_N_T0[p, :3, 3] = self._tip_N(self.tool_slot[p])
            self.chain.tool_spindle[p] = self._tool_spindle_index(p, self.tool_slot[p])
        if ms.meta.get("dist_force") is not None:
            self.dist_force = np.array(ms.meta["dist_force"], dtype=float)
            self.dist_torque = np.array(ms.meta["dist_torque"], dtype=float)
        from .state import feedback_from_dict
        ms.actions = [dict(a, fb=feedback_from_dict(a["fb"])) if a.get("action") == "feedback" else dict(a)
                      for a in ms.actions]
        ms.planners = self.planners
        ms.timeline_obj = self.timeline
        ms.snapshot_hook = lambda: self.snapshot(self._state)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
