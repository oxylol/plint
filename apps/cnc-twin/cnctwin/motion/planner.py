"""Per-path planner: commands -> pieces -> setpoints per L1 tick (docs/models/motion.md "Planner").

The planner keeps the path's queue of commands (:class:`Item`), turns motion commands into pieces
(:mod:`cnctwin.motion.geometry`) as they arrive (the corner with the previous piece is blended or
bounded then; the look-ahead never passes a block before its ready tick), and plans each macro step
tick by tick as a phase machine on a *copy* of its state:

* ``MOTION``: the compiled look-ahead generator (:func:`cnctwin.motion.kernels.plan_ticks`) over the
  run of pieces up to the next stop; setpoints from the compiled geometry (joint/blend pieces) or
  inverse kinematics per tick (TCP pieces); per-revolution runs become a spindle-synchronous stretch
  the loop evaluates from the actual spindle angle (``SP_PER_REV``), threading pieces ``SP_THREAD``,
  rigid tapping drives the spindle position from the same profile;
* ``TIMED``: non-motion commands (dwell, spindle, coolant, tool change with its Z move, turret,
  pallet, door, chuck, operator stops), dog-leg rapids (independent double-S profiles per axis) and
  the wait for spindle speed arrival before a cutting feed;
* waits: in-position (``SP_HOLD_WAIT``: the loop reports it), the controller (path-wait release,
  new commands after a buffer stop), the end of the program, a machine alarm.

Every planned tick records the command that owns it, its ledger category and a snapshot of the plan
state, so a sync event at any tick commits exactly the ticks executed (:meth:`PathPlanner.commit`):
persistent state changes only there.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from ..controller.ir import Command, FeedMode, Kind, Space
from .abi import (FK_M_W as FK_M_W_INDEX, GEO_ARC, GEO_LINE, SP_END, SP_HOLD_WAIT, SP_IDLE, SP_PER_REV, SP_THREAD, SP_TIME, SPM_CAXIS,
                  SPM_POSITION, SPM_SPEED, SYNC_ALARM, SYNC_IN_POSITION, SYNC_PATH_WAIT, SYNC_PROBE_TRIGGER, SYNC_QUEUE_EMPTY,
                  SYNC_SEGMENT_END, WAIT_INPOS, WAIT_PATH)
from .geometry import (PK_BLEND, PK_JOINT, PK_TIP, Piece, TipPath, apply_limits, arc_piece, blend_pieces, blend_radius,
                       corner_angle, corner_speed, curvature_step_speed, jump_speed, line_piece, scale_vector, tip_piece,
                       tolerance_speed, trim_line_end, trim_line_start)
from .kernels import brake_distance, plan_ticks
from .ledger import (CAT_AUX, CAT_COOLANT, CAT_DWELL, CAT_FEED, CAT_HOLD, CAT_INPOS, CAT_PROBE, CAT_RAPID,
                     CAT_SPINDLE, CAT_STARVE, CAT_STOP, CAT_TOOL, CAT_WAIT)
from .scurve import double_s
from .spindle import SpindleCmd, css_omega, evolve_speed, speed_change_s
from .timing import ticks_of

PH_IDLE = 0
PH_MOTION = 1
PH_TIMED = 2
PH_WAIT_INPOS = 3
PH_WAIT_CTRL = 4      # waiting for the controller (path-wait release, new commands after a buffer stop)
PH_ENDED = 5
PH_ALARM = 6
PH_RESYNC = 7         # the next tick is a sync (chain or mode change, stretch start): nothing more this call
PHASE_NAMES = ("idle", "motion", "timed", "wait_inpos", "wait_controller", "ended", "alarm", "resync")

MOTION_KINDS = (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE, Kind.TOOL_SETTER)
BUFFER_STOPPING = (Kind.SYNC, Kind.PROBE, Kind.TOOL_SETTER, Kind.STOP, Kind.MESSAGE)
# commands the motion before them must be in position for
INPOS_BEFORE = (Kind.DWELL, Kind.SYNC, Kind.STOP, Kind.MESSAGE, Kind.TOOL_CHANGE, Kind.TURRET_INDEX, Kind.PALLET,
                Kind.ALARM, Kind.CHUCK)
TAP_CYCLES = ("G84", "G74", "G84.2", "G84.3", "G184", "G174", "CYCLE84", "CYCL207", "CYCL209")


@dataclass
class Item:
    index: int
    cmd: Command
    pieces: list = field(default_factory=list)      # absolute piece indices
    inpos: bool = False                             # in-position wait at its end
    dog: list | None = None                         # dog-leg rapid: [(axis, q0, q1, DoubleS)]
    feed_kind: str = "per_min"
    ideal_s: float = 0.0                            # programmed-speed time of its motion (ledger)
    length_m: float = 0.0
    triggered: bool = False                         # probe / setter trigger seen (set at commit)
    latched: dict | None = None
    latched_tick: int = -1
    needs_rebuild: bool = False
    built_slot: int = -1                            # tool in the spindle when its pieces were built (TCP)
    c_spindles: list = field(default_factory=list)  # spindles whose C axis it moves (C-axis mode needed)

    @property
    def kind(self) -> Kind:
        return self.cmd.kind

    @property
    def is_motion(self) -> bool:
        return self.cmd.kind in MOTION_KINDS and self.dog is None


@dataclass
class Cursor:
    item: int = 0
    phase: int = PH_IDLE
    piece: int = 0
    s: float = 0.0
    v: float = 0.0
    a: float = 0.0
    t_start: int = 0
    t_end: int = 0
    timed_cat: str = CAT_AUX
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    in_stretch: bool = False

    def copy(self) -> "Cursor":
        c = copy.copy(self)
        c.q = self.q.copy()
        return c

    def to_dict(self) -> dict:
        return {"item": self.item, "phase": self.phase, "piece": self.piece, "s": self.s, "v": self.v, "a": self.a,
                "t_start": self.t_start, "t_end": self.t_end, "timed_cat": self.timed_cat, "q": self.q.tolist(),
                "in_stretch": self.in_stretch}

    @classmethod
    def from_dict(cls, d: dict) -> "Cursor":
        return cls(int(d["item"]), int(d["phase"]), int(d["piece"]), float(d["s"]), float(d["v"]), float(d["a"]),
                   int(d["t_start"]), int(d["t_end"]), str(d["timed_cat"]), np.array(d["q"], dtype=float),
                   bool(d.get("in_stretch", False)))


@dataclass
class PlanState:
    """Everything planning changes; committed by snapshot."""

    cur: Cursor
    tool_slot: int = -1
    tool_number: int | None = None
    station: int | None = None
    work_spindle: str | None = None
    tool_spindle: str | None = None
    rigid_tap: bool = False
    pumps: dict = field(default_factory=dict)
    nozzles: dict = field(default_factory=dict)
    sp: list = field(default_factory=list)          # SpindleCmd per spindle as this path commands them
    sp_changed: list = field(default_factory=list)  # spindle indices this path changed (committed in order)
    rt: dict = field(default_factory=dict)          # item index -> runtime data of an active item
    ended: bool = False
    alarm: dict | None = None

    def copy(self) -> "PlanState":
        return PlanState(self.cur.copy(), self.tool_slot, self.tool_number, self.station, self.work_spindle,
                         self.tool_spindle, self.rigid_tap, dict(self.pumps), dict(self.nozzles),
                         [c.copy() for c in self.sp], list(self.sp_changed), {k: dict(v) for k, v in self.rt.items()},
                         self.ended, dict(self.alarm) if self.alarm else None)

    def to_dict(self) -> dict:
        return {"cur": self.cur.to_dict(), "tool_slot": self.tool_slot, "tool_number": self.tool_number,
                "station": self.station, "work_spindle": self.work_spindle, "tool_spindle": self.tool_spindle,
                "rigid_tap": self.rigid_tap, "pumps": dict(self.pumps), "nozzles": dict(self.nozzles),
                "sp": [c.to_dict() for c in self.sp], "sp_changed": list(self.sp_changed),
                "rt": {str(k): _plain_rt(v) for k, v in self.rt.items()}, "ended": self.ended, "alarm": self.alarm}

    @classmethod
    def from_dict(cls, d: dict) -> "PlanState":
        return cls(Cursor.from_dict(d["cur"]), int(d["tool_slot"]), d["tool_number"], d["station"], d["work_spindle"],
                   d["tool_spindle"], bool(d["rigid_tap"]), dict(d["pumps"]), dict(d["nozzles"]),
                   [SpindleCmd.from_dict(c) for c in d["sp"]], list(d["sp_changed"]),
                   {int(k): _rt_from(v) for k, v in d["rt"].items()}, bool(d["ended"]), d["alarm"])


def _plain_rt(v: dict) -> dict:
    out = {}
    for k, x in v.items():
        if k == "dog":
            out[k] = [[int(a), float(q0), float(q1), [p.distance, p.Tj, p.Ta, p.Tv, p.J]] for a, q0, q1, p in x]
        else:
            out[k] = x
    return out


def _rt_from(v: dict) -> dict:
    from .scurve import DoubleS
    out = dict(v)
    if "dog" in out:
        out["dog"] = [(int(a), float(q0), float(q1), DoubleS(*prof)) for a, q0, q1, prof in out["dog"]]
    return out


@dataclass
class TickPlan:
    """What one plan() call decided for ticks k_from .. n-1 of the step."""

    k_from: int
    n: int
    mode: np.ndarray
    rapid: np.ndarray
    tap: np.ndarray
    probe: np.ndarray
    setter: np.ndarray
    sync_start: np.ndarray
    owner: np.ndarray                 # item index owning each tick (-1: none)
    cat: list                         # ledger category per tick
    lim: np.ndarray                   # generator limit flag per tick (2: held back by unprepared blocks)
    snaps: list                       # PlanState after each tick
    events: list = field(default_factory=list)          # (boundary index, kind, data)
    # number of events emitted when the snapshot of each tick was taken (-1: no snapshot): a sync at k_stop commits
    # exactly the events emitted while ticks up to k_stop were planned. The boundary alone cannot tell them apart:
    # an item that ends at the end of tick k and one that starts (and, without motion, ends) at the start of tick
    # k + 1 both carry boundary k + 1, and only the first belongs to the snapshot of tick k.
    ev_mark: np.ndarray | None = None
    spindle: list = field(default_factory=list)         # (k, spindle index, SpindleCmd) changes
    stretch: dict | None = None                         # synchronous stretch of this path
    setter_cfg: tuple | None = None                     # (point, normal, radius, extra)
    final: PlanState | None = None
    c_offset: dict = field(default_factory=dict)        # axis -> ServoState.c_off from tick k_from (C-axis engage)


class PathPlanner:
    def __init__(self, env, p: int, q0: np.ndarray):
        self.env = env
        self.p = int(p)
        model = env.model
        self.model = model
        self.axes = [ax.index for ax in model.path_axes(p)]
        self.nc_axes = [a for a in self.axes if not model.axes[a].synthetic]
        self.axes_arr = np.array(self.axes, np.int32)
        self.items: list[Item] = []
        self.item0 = 0
        self.pieces: list[Piece] = []
        self.piece0 = 0
        self.tail_q = np.asarray(q0, dtype=float).copy()
        self._tail_slot = -1
        self._tail_caxis: set[int] = set()           # spindles in C-axis mode at the end of the queue
        self.last_target: dict[str, float] = {}
        self.panel = None
        self.expect_more = True
        pd = model.paths[p]
        self.st = PlanState(Cursor(q=np.asarray(q0, dtype=float).copy()), work_spindle=pd.work_spindle,
                            tool_spindle=pd.tool_spindle, sp=[SpindleCmd() for _ in model.spindles])
        self.stretch_info: dict | None = None        # geometry of the spindle-synchronous stretch in progress
        self._tail_kin = None
        self._tail_kin_slot = -1

    def tail_kin(self):
        """Kinematics with the tool that will be in the spindle at the end of the queue (TCP pieces are built
        with it, so a block after a queued tool change uses the new tool's length)."""
        from .ik import PathKinematics
        kin_now = self.env.kin[self.p]
        if self._tail_kin is None or self._tail_kin_slot != self._tail_slot:
            chain = type(kin_now.chain)(*(np.array(a, copy=True) for a in kin_now.chain))
            chain.pose_N_T0[self.p, :3, 3] = self.env.dom._tip_N(self._tail_slot)
            self._tail_kin = PathKinematics(self.model, chain, self.p)
            self._tail_kin_slot = self._tail_slot
        return self._tail_kin

    # ================================================================== items and pieces
    def item(self, i: int) -> Item | None:
        j = i - self.item0
        return self.items[j] if 0 <= j < len(self.items) else None

    def piece(self, j: int) -> Piece:
        return self.pieces[j - self.piece0]

    def n_pieces(self) -> int:
        return self.piece0 + len(self.pieces)

    def n_items(self) -> int:
        return self.item0 + len(self.items)

    def trim(self) -> None:
        """Release completed items and pieces well behind the cursor (bounded memory)."""
        cur = self.st.cur
        drop = (cur.item - 4) - self.item0
        if drop > 512:
            self.items = self.items[drop:]
            self.item0 += drop
        drop = (cur.piece - 4) - self.piece0
        if drop > 512:
            self.pieces = self.pieces[drop:]
            self.piece0 += drop

    def queued(self) -> int:
        return self.n_items() - self.st.cur.item

    # ================================================================== accepting commands
    def accept(self, commands) -> None:
        for cmd in commands:
            self._append(cmd)
        if commands and self.st.cur.phase in (PH_WAIT_CTRL, PH_ENDED):
            self.st.cur.phase = PH_IDLE
            self.st.ended = False
        self.expect_more = True

    def _prev_item(self) -> Item | None:
        for it in reversed(self.items):
            if it.cmd.kind == Kind.MODE:
                continue
            return it
        return None

    def _append(self, cmd: Command) -> None:
        model = self.model
        item = Item(self.n_items(), cmd)
        prev = self._prev_item()
        self.items.append(item)
        if cmd.kind == Kind.MODE:
            return
        prev_open = prev is not None and prev.pieces and prev.index >= self.st.cur.item
        if prev_open and not (cmd.kind in MOTION_KINDS and self._continuous(prev, cmd)):
            last = self.piece(prev.pieces[-1])
            last.stop_end = True
            if cmd.kind in INPOS_BEFORE or cmd.kind in BUFFER_STOPPING:
                prev.inpos = True
        if cmd.kind in (Kind.TOOL_CHANGE, Kind.TURRET_INDEX):
            data = cmd.tool or {}
            slot = data.get("slot")
            if slot is None and data.get("number") is not None:
                slot = self.env.slot_of(int(data["number"]), self.p)
            self._tail_slot = int(slot) if slot is not None else -1
        if cmd.kind == Kind.TOOL_CHANGE and not self._turret_change(cmd):
            z_axis = self.env.tool_change_axis(self.p)
            if z_axis is not None:
                self.tail_q[z_axis] = float(model.axes[z_axis].home)
        if cmd.kind == Kind.SPINDLE:
            # any spindle command puts the spindle it names (or, unnamed, every spindle) back in speed control
            sid = (cmd.spindle or {}).get("id")
            if sid is not None and sid in model.spindle_index:
                self._tail_caxis.discard(model.spindle_index[sid])
            else:
                self._tail_caxis.clear()
        if cmd.kind not in MOTION_KINDS:
            return
        target = self._target(cmd)
        q0 = self.tail_q.copy()
        q1 = q0.copy()
        for a, value in target.items():
            q1[a] = value
        item.feed_kind = self._feed_kind(cmd)
        item.c_spindles = sorted({model.spindle_index[model.axes[a].spindle] for a in target
                                  if model.axes[a].spindle is not None and not model.axes[a].synthetic})
        engage = any(s not in self._tail_caxis for s in item.c_spindles)
        self._tail_caxis.update(item.c_spindles)
        if engage and prev_open:
            # the spindle enters C-axis mode at rest: the motion before stops in position
            self.piece(prev.pieces[-1]).stop_end = True
            prev.inpos = True
        self.tail_q = q1
        if cmd.kind == Kind.RAPID and model.rapid_mode() == "dogleg":
            self._dogleg(item, q0, q1)
            if prev_open:
                self.piece(prev.pieces[-1]).stop_end = True
                prev.inpos = True
            return
        item.built_slot = self._tail_slot
        pieces = self._build_pieces(item, cmd, q0, q1, item.feed_kind)
        if not pieces:
            return
        for pc in pieces:
            pc.ready = int(cmd.ready_tick)
            if pc is not pieces[0] and pc.ustart == 0.0 and pc.kind != PK_TIP:
                pc.ustart = math.inf
        if prev_open and self._continuous(prev, cmd) and not engage:
            self._join(prev, pieces)
        else:
            pieces[0].ustart = 0.0
        start = self.n_pieces()
        self.pieces.extend(pieces)
        item.pieces = list(range(start, start + len(pieces)))
        exact = cmd.smoothing is not None and cmd.smoothing.corner == "exact_stop"
        if cmd.kind in (Kind.RAPID, Kind.PROBE, Kind.TOOL_SETTER) or exact or item.feed_kind in ("tap", "thread"):
            pieces[-1].stop_end = True
            item.inpos = cmd.kind in (Kind.RAPID, Kind.PROBE, Kind.TOOL_SETTER) or exact or item.feed_kind == "tap"

    def _turret_change(self, cmd: Command) -> bool:
        model = self.model
        changer = model.tool_changer
        return cmd.kind == Kind.TURRET_INDEX or (changer is not None and str(getattr(changer, "type", "")) ==
                                                 "turret") or (model.lathe and bool(model.turrets) and changer is None)

    def _target(self, cmd: Command) -> dict[int, float]:
        """Target axis values (global index) of a motion command: continuous rotary axes by the shortest
        path (half a turn: positive), spindle C axes only when the program moves them."""
        model = self.model
        out = {}
        for name, value in (cmd.target or {}).items():
            if name not in model.axis_index:
                continue
            a = model.axis_index[name]
            ax = model.axes[a]
            prev_target = self.last_target.get(name)
            self.last_target[name] = float(value)
            if ax.spindle is not None and (prev_target is None or abs(float(value) - prev_target) < 1e-12):
                continue
            v = float(value)
            if ax.rotary and ax.continuous:
                cur = self.tail_q[a]
                d = math.remainder(v - cur, 2.0 * math.pi)
                if abs(abs(d) - math.pi) < 1e-12:
                    d = abs(d)
                v = cur + d
            out[a] = v
        return out

    def _feed_kind(self, cmd: Command) -> str:
        modal = cmd.modal or {}
        if cmd.kind == Kind.RAPID:
            return "rapid"
        if modal.get("thread"):
            return "thread"
        if (modal.get("rigid_tap") or self.st.rigid_tap) and str(modal.get("cycle") or "") in TAP_CYCLES \
                and cmd.kind == Kind.LINEAR:
            return "tap"
        feed = cmd.feed
        if feed is None:
            return "per_min"
        if feed.mode == FeedMode.PER_REV:
            return "per_rev"
        if feed.mode == FeedMode.INVERSE_TIME:
            return "inverse_time"
        if feed.mode == FeedMode.RAPID:
            return "rapid"
        return "per_min"

    def _continuous(self, prev: Item, cmd: Command) -> bool:
        """Continuous path (blend or corner bound) between prev and the motion command cmd?"""
        if prev.cmd.kind not in (Kind.LINEAR, Kind.ARC) or cmd.kind not in (Kind.LINEAR, Kind.ARC):
            return False
        if prev.dog is not None or prev.inpos:
            return False
        if prev.cmd.smoothing is not None and prev.cmd.smoothing.corner == "exact_stop":
            return False
        if self._feed_kind(cmd) != prev.feed_kind or prev.feed_kind in ("tap", "thread"):
            return False
        return True

    def _tolerance(self, cmd: Command) -> float:
        sm = cmd.smoothing
        if sm is not None and sm.tolerance_m > 0.0:
            return float(sm.tolerance_m)
        return float(self.model.p["default_contour_tolerance_m"])

    def _tolerance_limit(self, pc: Piece, tol: float) -> None:
        """Tolerance-driven speed bound of a curved feed piece (geometry.tolerance_speed): the chord between the
        interpolation points of one IPO period, and the tolerance band of the control's axis setpoint filter when the
        machine has one, keep the commanded path within the contour tolerance in force."""
        v = tolerance_speed(pc, tol, self.env.ipo_s, self.model.setpoint_filter_s)
        if v < pc.limit_v:
            pc.limit_v = v
            pc.vmax = min(pc.vmax, v)

    def _programmed_speed(self, cmd: Command, length: float, feed_kind: str) -> float:
        """Programmed path speed of a feed move (m/s of the path metric, without override); per-revolution
        feeds at the spindle command known now (refreshed when the move starts)."""
        feed = cmd.feed
        value = float(feed.value) if feed is not None else 0.0
        if feed_kind in ("per_rev", "thread") or (feed_kind == "tap" and feed is not None
                                                  and feed.mode == FeedMode.PER_REV):
            omega = abs(self.planned_omega(self.cut_spindle(self.st)))
            return value * omega / (2.0 * math.pi)
        if feed_kind == "inverse_time":
            return length * value
        return value

    def overrides(self, pc: Piece, it: Item | None, omega: float = 0.0) -> tuple[float, float]:
        """(planning speed, programmed speed with override) of a piece now: feed or rapid override, dry run,
        the machine's maximum cutting feed; the piece's axis and curvature limits cap the planning speed.
        A per-revolution piece moves its feed per revolution at the cutting spindle's speed omega (rad/s)."""
        panel = self.panel
        if pc.rapid:
            scale = float(panel.rapid_override) if panel is not None else 1.0
            v = pc.limit_v * scale
            return v, v
        feed = it.cmd.feed if it is not None else None
        override = 1.0
        if panel is not None and (feed is None or feed.override_allowed) and pc.feed not in ("tap", "thread"):
            override = float(panel.feed_override)
        v = pc.feed_v if pc.per_rev <= 0.0 else pc.per_rev * abs(omega) / (2.0 * math.pi)
        if panel is not None and panel.dry_run and pc.feed not in ("tap", "thread"):
            v = self.model.max_feed if math.isfinite(self.model.max_feed) else v
        ideal = min(v * override, self.model.max_feed)
        return min(ideal, pc.limit_v), ideal

    def planned_omega(self, sid: str | None) -> float:
        s = self.model.spindle_index.get(sid) if sid is not None else None
        if s is None:
            return 0.0
        cmd = self.env.spindle_cmd[s]
        sp = self.model.spindles[s]
        return css_omega(cmd, self.css_radius(s, self.tail_q), sp, self.model.p["css_min_radius_m"])

    def css_radius(self, s: int, q: np.ndarray) -> float:
        info = self.env.css_axis_info(s, self.p)
        if info is None:
            return math.inf
        axis, centre = info
        return abs(float(q[axis]) - centre)

    def _build_pieces(self, item: Item, cmd: Command, q0, q1, feed_kind: str, building: bool = True) -> list[Piece]:
        model = self.model
        tip_mode = cmd.space == Space.TIP and (cmd.modal or {}).get("tip_interpolation", "tip") == "tip"
        if tip_mode:
            pieces = self._tip_piece(item, cmd, q0, q1, building)
        elif cmd.kind == Kind.ARC and cmd.arc is not None:
            pieces = [arc_piece(item.index, q0, q1, cmd.arc, model, self.p)]
        else:
            pieces = [line_piece(item.index, q0, q1, model, self.nc_axes)]
        out = []
        for pc in pieces:
            if pc.length <= 1e-15:
                continue
            pc.feed = feed_kind
            pc.rapid = cmd.kind == Kind.RAPID
            if pc.rapid:
                apply_limits(pc, model, self.nc_axes, math.inf)
                pc.ideal_v = pc.limit_v
            else:
                v = self._programmed_speed(cmd, pc.length, feed_kind)
                apply_limits(pc, model, self.nc_axes, v if v > 0.0 else 1e-9)
                self._tolerance_limit(pc, self._tolerance(cmd))
                pc.ideal_v = pc.feed_v
                feed = cmd.feed
                if feed_kind in ("per_rev", "thread") or (feed_kind == "tap" and feed is not None
                                                          and feed.mode == FeedMode.PER_REV):
                    pc.per_rev = float((cmd.modal or {}).get("thread", {}).get("lead_m", feed.value)) \
                        if feed_kind == "thread" else float(feed.value)
            out.append(pc)
            item.ideal_s += pc.length / pc.ideal_v if pc.ideal_v > 0.0 else 0.0
            item.length_m += pc.length
        return out

    def _rapid_scale(self) -> float:
        panel = self.panel
        return float(panel.rapid_override) if panel is not None else 1.0

    def _tip_piece(self, item: Item, cmd: Command, q0, q1, building: bool = True) -> list[Piece]:
        """TCP pieces: the tip on a line or arc in W. With programmed rotary positions (FANUC G43.4, Haas
        G234, TRAORI with B/C, Heidenhain M128) the rotary axes move linearly in joint space and the linear
        axes keep the tip on the path; with a programmed tool vector (modal tcp_type "vector", G43.5) the
        tool axis turns on the great circle (slerp) and every tick is a full inverse-kinematics solve, a
        start at a singular orientation preceded by turning the free rotary axis there with the tip still."""
        kin = self.tail_kin() if building else self.env.kin[self.p]
        model = self.model
        tip0, a0 = kin.tip_axis_W(q0)
        tip1, a1 = kin.tip_axis_W(q1)
        vector = (cmd.modal or {}).get("tcp_type") == "vector"
        rot_axes = [j.axis for j in kin.joints]
        if cmd.kind == Kind.ARC and cmd.arc is not None and cmd.tip_target_m is not None:
            # the control's TIP frame: axes parallel to M at rotary zero, origin found from the end point
            R_W0 = kin.fk(np.zeros(len(model.axes)))[FK_M_W_INDEX][:3, :3]
            offset = tip1 - R_W0.T @ np.asarray(cmd.tip_target_m, float)
            centre = R_W0.T @ np.asarray(cmd.arc.centre, float) + offset
            nrm = R_W0.T @ np.asarray(cmd.arc.normal, float)
            nrm = nrm / np.linalg.norm(nrm)
            centre = centre + nrm * float((tip0 - centre) @ nrm)
            r0v = tip0 - centre
            r0 = float(np.linalg.norm(r0v))
            e1 = r0v / r0 if r0 > 0 else np.zeros(3)
            e2 = np.cross(nrm, e1)
            h = float((tip1 - tip0) @ nrm)
            r1 = float(np.linalg.norm(tip1 - nrm * h - centre))
            path = TipPath(GEO_ARC, tip0, tip1, centre, e1, e2, r0, r1, float(cmd.arc.sweep_rad), h * nrm, a0, a1)
        else:
            path = TipPath(GEO_LINE, tip0, tip1, a0=a0, a1=a1)
        pieces = []
        if vector:
            path.mode = "vector"
            q_start = q0
            if kin.singular_at(a0):
                # the free axis is undetermined at the start: turn it to where the slerp leaves the pole
                u_off = 1.0 / max(4, int(model.p["tcp_samples_per_piece"]))
                q_next = kin.solve(path.tip(u_off), path.axis(u_off), q0).q
                q_turn = q0.copy()
                for a in kin.singular_at(a0):
                    q_turn[a] = q_next[a]
                if np.any(q_turn[rot_axes] != q0[rot_axes]):
                    pre = TipPath(GEO_LINE, tip0, tip0.copy(), a0=a0, a1=a0.copy(), mode="rotary", rot_axes=rot_axes,
                                  rot0=q0[rot_axes].copy(), rot1=q_turn[rot_axes].copy())
                    pieces.append(self._sampled(item, q0, q_turn, pre, kin))
                    q_start = q_turn
            main = self._sampled(item, q_start, q1, path, kin)
            if pieces:
                main.ustart = 0.0          # the free axis comes to rest at the pole before the slerp starts
            pieces.append(main)
        else:
            path.rot_axes = rot_axes
            path.rot0 = q0[rot_axes].copy()
            path.rot1 = q1[rot_axes].copy()
            pieces.append(self._sampled(item, q0, q1, path, kin))
        return pieces

    def _sampled(self, item: Item, q0, q1, path: TipPath, kin) -> Piece:
        pc = tip_piece(item.index, q0, q1, path, self.model)
        if path.mode == "rotary" and np.allclose(path.p0, path.p1) and path.kind == GEO_LINE:
            # orientation only: the feed applies to the rotary axes (1 deg = 1 mm)
            pc.length = self.model.rot_len * float(np.linalg.norm(path.rot1 - path.rot0))
        n_samples = max(4, int(self.model.p["tcp_samples_per_piece"]))
        qs = np.zeros((n_samples, len(self.model.axes)))
        q = q0.copy()
        singular = False
        for k, u in enumerate(np.linspace(0.0, 1.0, n_samples)):
            q = self._tcp_q(path, u, q, kin)
            singular = singular or bool(kin.singular_at(kin.tip_axis_W(q)[1]))
            qs[k] = q
        pc.samples_q = qs
        pc.singular = singular
        return pc

    def _tcp_q(self, path: TipPath, u: float, q_ref: np.ndarray, kin=None) -> np.ndarray:
        kin = kin if kin is not None else self.env.kin[self.p]
        if path.mode == "rotary":
            q = q_ref.copy()
            for c, a in enumerate(path.rot_axes):
                q[a] = path.rot0[c] + u * (path.rot1[c] - path.rot0[c])
            return kin.solve_position(path.tip(u), q).q
        return kin.solve(path.tip(u), path.axis(u), q_ref).q

    def _dogleg(self, item: Item, q0, q1) -> None:
        model = self.model
        scale = max(self._rapid_scale(), 1e-6)
        dog = []
        ideal = 0.0
        length = 0.0
        sc = scale_vector(model)
        for a in self.nc_axes:
            d = float(q1[a] - q0[a])
            if d == 0.0:
                continue
            ax = model.axes[a]
            prof = double_s(abs(d), ax.rapid * scale, ax.accel, ax.jerk)
            dog.append((a, float(q0[a]), float(q1[a]), prof))
            ideal = max(ideal, abs(d) / (ax.rapid * scale))
            length += (sc[a] * d) ** 2
        item.dog = dog
        item.inpos = True
        item.feed_kind = "rapid"
        item.ideal_s = ideal
        item.length_m = math.sqrt(length)

    def _join(self, prev: Item, pieces: list[Piece]) -> None:
        """Corner between the last piece of prev and the first of pieces: a blend arc or a node bound."""
        model = self.model
        last_index = prev.pieces[-1]
        last = self.piece(last_index)
        first = pieces[0]
        cur = self.st.cur
        if cur.piece > last_index or (cur.piece == last_index and cur.phase == PH_MOTION
                                     and cur.s >= last.length - 1e-12):
            first.ustart = 0.0
            last.stop_end = True
            return
        theta = corner_angle(last, first)
        filter_s = model.setpoint_filter_s
        if theta <= model.p["min_corner_angle_rad"]:
            # tangent: no corner, but the curvature may change (a line meeting an arc, two arcs)
            first.ustart = curvature_step_speed(last, first, model, self.nc_axes, filter_s)
            return
        tol = self._tolerance(prev.cmd)
        done = cur.s if (cur.piece == last_index and cur.phase == PH_MOTION) else 0.0
        frac = model.p["blend_length_frac"]
        blend = blend_pieces(last, first, tol, model, self.nc_axes, done, frac)
        if blend is not None:
            new_len, h1, h2, cut = blend
            trim_line_end(last, new_len, model)
            trim_line_start(first, cut, model)
            for half in (h1, h2):
                self._tolerance_limit(half, tol)
            h1.ustart = h1.limit_v
            h2.ustart = h2.limit_v
            first.ustart = h2.limit_v
            h1.ready = last.ready
            h2.ready = first.ready
            idx = self.n_pieces()
            self.pieces.append(h1)
            prev.pieces.append(idx)
            pieces.insert(0, h2)
            return
        L1 = last.length
        L2 = first.length
        R, _ = blend_radius(tol, theta, frac * min(max(L1 - done, 0.0), L2))
        # not blended: the commanded velocity of each axis steps at the corner; it may change by at most what
        # the axis' acceleration achieves in one interpolation period, and the tolerance band bounds the speed
        first.ustart = min(corner_speed(R, last, first, model, self.nc_axes),
                           jump_speed(last, first, model, self.nc_axes, self.env.ipo_s),
                           curvature_step_speed(last, first, model, self.nc_axes, filter_s))

    # ================================================================== helpers on a plan state
    def cut_spindle(self, w: PlanState) -> str | None:
        rotating = self.env.slot_rotating(w.tool_slot)
        if rotating and w.tool_spindle is not None:
            return w.tool_spindle
        return w.work_spindle or w.tool_spindle

    def _omega_at(self, s: int, boundary: int, w: PlanState) -> float:
        """Predicted spindle speed at a boundary: the actual speed at the plan start evolved toward the
        commanded target (ramp and torque limits)."""
        env = self.env
        omega0 = float(env.sp_actual.omega[s])
        dt = max(0.0, (boundary - self.plan_tick) * env.tick_s)
        cmd = w.sp[s]
        sp = self.model.spindles[s]
        target = css_omega(cmd, self.css_radius(s, w.cur.q), sp, self.model.p["css_min_radius_m"]) \
            if cmd.mode == SPM_SPEED else 0.0
        return evolve_speed(sp, omega0, target, cmd.gear, dt)

    def _angle_at(self, s: int, boundary: int) -> float:
        env = self.env
        dt = max(0.0, (boundary - self.plan_tick) * env.tick_s)
        return float(env.sp_actual.angle[s]) + float(env.sp_actual.omega[s]) * dt

    def _set_spindle(self, w: PlanState, s: int, cmd: SpindleCmd, k: int) -> None:
        cmd.owner = self.p
        w.sp[s] = cmd
        if s not in w.sp_changed:
            w.sp_changed.append(s)
        self.tp.spindle.append((k, s, cmd.copy()))

    # ================================================================== planning
    def plan(self, k_from: int, n: int, tick0: int) -> TickPlan:
        env = self.env
        self.tick0 = tick0
        self.plan_tick = tick0 + k_from * env.ipo_ticks
        tp = TickPlan(k_from, n, np.full(n, SP_IDLE, np.int8), np.zeros(n, np.int8), np.zeros(n, np.int8),
                      np.zeros(n, np.int8), np.zeros(n, np.int8), np.zeros(n, np.int8), np.full(n, -1, np.int32),
                      [CAT_WAIT] * n, np.zeros(n, np.int8), [None] * n, ev_mark=np.full(n, -1, np.int64))
        self.tp = tp
        self.rows = env.setp_cmd
        w = self.st.copy()
        w.sp = [c.copy() for c in env.spindle_cmd]
        for s in w.sp_changed:
            w.sp[s] = self.st.sp[s].copy()
        cur = w.cur
        if cur.phase == PH_RESYNC:
            # this plan call is the re-plan the resync asked for
            self._after_resync(w)
        i = k_from
        guard = 0
        while i < n:
            guard += 1
            if guard > 200000:
                raise RuntimeError(f"motion planner of path {self.p + 1} made no progress at tick {i}")
            boundary = tick0 + i * env.ipo_ticks
            if cur.phase == PH_IDLE:
                it = self.item(cur.item)
                if it is None:
                    mode = SP_END if self.expect_more else SP_IDLE
                    for k in range(i, n):
                        self._hold(tp, k, w, mode, -1, CAT_STARVE if self.expect_more else CAT_WAIT)
                    break
                if it.cmd.ready_tick > boundary:
                    self._hold(tp, i, w, SP_TIME, it.index, CAT_STARVE)
                    i += 1
                    continue
                self._start(it, w, boundary, i)
                continue
            if cur.phase == PH_MOTION:
                i += self._run_motion(w, tp, i, n)
                continue
            if cur.phase == PH_TIMED:
                i += self._run_timed(w, tp, i, n)
                continue
            if cur.phase == PH_WAIT_INPOS:
                it = self.item(cur.item)
                path_wait = it is not None and it.cmd.kind == Kind.SYNC and \
                    (it.cmd.sync or {}).get("reason") == "path_wait"
                env.wait_kind[self.p] = WAIT_PATH if path_wait else WAIT_INPOS
                for k in range(i, n):
                    self._hold(tp, k, w, SP_HOLD_WAIT, cur.item, CAT_INPOS)
                break
            if cur.phase == PH_RESYNC:
                for k in range(i, n):
                    self._hold(tp, k, w, SP_END, cur.item if self.item(cur.item) else -1, CAT_AUX)
                break
            if cur.phase == PH_WAIT_CTRL:
                for k in range(i, n):
                    self._hold(tp, k, w, SP_IDLE, cur.item if self.item(cur.item) else -1, CAT_WAIT)
                break
            for k in range(i, n):
                self._hold(tp, k, w, SP_IDLE, -1, CAT_STOP)
            break
        tp.final = w
        return tp

    def _after_resync(self, w: PlanState) -> None:
        cur = w.cur
        cur.phase = PH_IDLE
        it = self.item(cur.item)
        if it is not None and w.rt.get(it.index, {}).pop("resume_motion", False):
            cur.phase = PH_MOTION

    def _snap(self, tp: TickPlan, k: int, w: PlanState) -> None:
        """The plan state after tick k, and how many events had been emitted by then (commit scope)."""
        tp.snaps[k] = w.copy()
        tp.ev_mark[k] = len(tp.events)

    def _hold(self, tp: TickPlan, k: int, w: PlanState, mode: int, owner: int, cat: str) -> None:
        tp.mode[k] = mode
        tp.owner[k] = owner
        tp.cat[k] = cat
        for a in self.axes:
            self.rows[k, a] = w.cur.q[a]
        self._snap(tp, k, w)

    # ------------------------------------------------------------------ starting items
    def _start(self, it: Item, w: PlanState, boundary: int, k: int) -> None:
        tp = self.tp
        env = self.env
        cur = w.cur
        if w.rt.get(it.index, {}).get("c_engage") == "switch":
            if k != tp.k_from:
                cur.phase = PH_RESYNC           # the switch happens at the first tick of a plan call
                return
            self._switch_c_axis(it, w, k)
        else:
            cur.item = it.index
            tp.events.append((k, "start", it.index))
            if it.c_spindles and any(w.sp[s].mode != SPM_CAXIS for s in it.c_spindles):
                self._begin_c_engage(it, w, boundary, k)
                return
        if it.dog is not None:
            ticks = max((ticks_of(prof.duration, env.ipo_s) for _, _, _, prof in it.dog), default=0)
            w.rt[it.index] = {"dog": list(it.dog)}
            self._begin_timed(cur, boundary, ticks, CAT_RAPID)
            return
        if it.is_motion:
            locked = self.panel is not None and self.panel.machine_lock
            if it.pieces and not it.needs_rebuild and not locked:
                q0 = self.piece(it.pieces[0]).q0
                axes = self._geometry_axes(it)
                if axes and float(np.max(np.abs(q0[axes] - cur.q[axes]))) > 1e-9:
                    it.needs_rebuild = True
                if self.piece(it.pieces[0]).kind == PK_TIP and it.built_slot != w.tool_slot:
                    it.needs_rebuild = True          # TCP geometry depends on the tool in the spindle
            if it.needs_rebuild:
                self._rebuild(it, w)
            if not it.pieces:
                self._finish(it, w, k)
                return
            if it.feed_kind != "rapid" and it.cmd.kind in (Kind.LINEAR, Kind.ARC):
                wait = self._arrival_wait(boundary, w)
                if wait > 0:
                    cur.piece = it.pieces[0]
                    cur.s = cur.v = cur.a = 0.0
                    self._begin_timed(cur, boundary, wait, CAT_SPINDLE)
                    return
            self._enter_motion(it, w, k)
            return
        self._start_nonmotion(it, w, boundary, k)

    def _spindle_c_axes(self, it: Item) -> list[int]:
        """Path axes that are a spindle's C axis the item does not move: outside C-axis mode they turn with the
        spindle (the commanded value is the spindle angle), so they are not part of the item's geometry."""
        model = self.model
        return [a for a in self.nc_axes if model.axes[a].spindle is not None
                and model.spindle_index.get(model.axes[a].spindle) not in it.c_spindles]

    def _geometry_axes(self, it: Item) -> list[int]:
        """Path axes an item's pieces are built over and must start from (spindle C axes it moves included)."""
        skip = set(self._spindle_c_axes(it))
        return [a for a in self.nc_axes if a not in skip]

    def _begin_c_engage(self, it: Item, w: PlanState, boundary: int, k: int) -> None:
        """C-axis mode for the spindles the item turns as C axes: each stops under speed control (deceleration
        ramp and torque limit) and is referenced (c_engage_s); then the C axis takes over at the first tick of
        the next plan call (_switch_c_axis)."""
        seconds = 0.0
        for s in it.c_spindles:
            if w.sp[s].mode == SPM_CAXIS:
                continue
            sp = self.model.spindles[s]
            omega_now = self._omega_at(s, boundary, w)
            stop = speed_change_s(sp, omega_now, 0.0, w.sp[s].gear) if abs(omega_now) > 0.0 else 0.0
            new = w.sp[s].copy()
            new.mode = SPM_SPEED
            new.omega = 0.0
            new.css_vc = 0.0
            self._set_spindle(w, s, new, k)
            seconds = max(seconds, stop + sp.c_engage_s)
        w.rt.setdefault(it.index, {})["c_engage"] = "wait"
        self._begin_timed(w.cur, boundary, max(1, ticks_of(seconds, self.env.ipo_s)), CAT_SPINDLE)

    def _switch_c_axis(self, it: Item, w: PlanState, k: int) -> None:
        """At tick k (the first of this plan call) the spindles of the item follow their C axes. The C
        coordinate continues from the spindle angle, relabelled by whole turns to lie within half a turn of
        the value the item was built from, so the queued blocks after it stay valid; the item itself is
        rebuilt from there to the target it was built for."""
        env = self.env
        model = self.model
        cur = w.cur
        w.rt.setdefault(it.index, {}).pop("c_engage", None)
        q_from, q_to = self._item_ends(it, cur.q)
        two_pi = 2.0 * math.pi
        for s in it.c_spindles:
            if w.sp[s].mode == SPM_CAXIS:
                continue                      # already a C axis: its coordinate is the commanded one
            sp = model.spindles[s]
            a = model.axis_index[sp.c_axis]
            kap = model.axes[a].kappa
            theta = float(env.sp_actual.angle[s])
            c_now = kap * theta
            off = two_pi * round((float(q_from[a]) - c_now) / two_pi)
            cur.q[a] = c_now + off
            self.tp.c_offset[a] = off
            new = w.sp[s].copy()
            new.mode = SPM_CAXIS
            new.omega = 0.0
            new.css_vc = 0.0
            new.angle = theta
            new.omega_ff = 0.0
            self._set_spindle(w, s, new, k)
        if it.dog is not None:
            self._dogleg(it, cur.q.copy(), q_to)
        else:
            self._rebuild(it, w, q_to)

    def _item_ends(self, it: Item, q_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint positions an item was built from and to (its programmed end, before any corner blend)."""
        if it.dog is not None:
            q0 = q_now.copy()
            q1 = q_now.copy()
            for a, a0, a1, _ in it.dog:
                q0[a] = a0
                q1[a] = a1
            return q0, q1
        q0 = self.piece(it.pieces[0]).q0.copy() if it.pieces else q_now.copy()
        q1 = q0.copy()
        c_axes = {self.model.axis_index[self.model.spindles[s].c_axis] for s in it.c_spindles}
        for name, value in (it.cmd.target or {}).items():
            a = self.model.axis_index.get(name)
            if a is None:
                continue
            ax = self.model.axes[a]
            if ax.spindle is not None and a not in c_axes:
                continue
            v = float(value)
            if ax.rotary and ax.continuous:
                d = math.remainder(v - q0[a], 2.0 * math.pi)
                if abs(abs(d) - math.pi) < 1e-12:
                    d = abs(d)
                v = q0[a] + d
            q1[a] = v
        return q0, q1

    def _feed_omega(self, w: PlanState) -> float:
        """Commanded speed of the path's cutting spindle at the current position (CSS evaluated there): the
        speed per-revolution feeds are planned at (the loop then follows the actual spindle angle)."""
        sid = self.cut_spindle(w)
        s = self.model.spindle_index.get(sid) if sid is not None else None
        if s is None:
            return 0.0
        return css_omega(w.sp[s], self.css_radius(s, w.cur.q), self.model.spindles[s],
                         self.model.p["css_min_radius_m"])

    def _record_ideal(self, it: Item, w: PlanState, k: int, omega: float) -> None:
        """The programmed-speed time of an item's motion (the ledger's reference for acceleration and data
        starvation): its feed with the override in force, per-revolution feeds at the spindle speed omega,
        when the planner first reaches the item."""
        ideal = 0.0
        for j in it.pieces:
            pc = self.piece(j)
            _, v = self.overrides(pc, it, omega)
            ideal += pc.length / v if v > 0.0 else 0.0
        w.rt.setdefault(it.index, {})["ideal_s"] = ideal
        self.tp.events.append((k, "ideal", (it.index, ideal)))

    def _tap_limits(self, it: Item, w: PlanState, omega: float) -> None:
        """Rigid tapping: the spindle turns with the feed axis, so the path speed and acceleration are also
        bounded by the spindle's speed and its position-loop acceleration in the gear (pitch / 2 pi per rad)."""
        sid = self.cut_spindle(w)
        s = self.model.spindle_index.get(sid) if sid is not None else None
        if s is None:
            return
        sp = self.model.spindles[s]
        feed = it.cmd.feed
        if feed is not None and feed.mode == FeedMode.PER_REV:
            pitch = float(feed.value)
        elif feed is not None and omega != 0.0:
            pitch = float(feed.value) * 2.0 * math.pi / abs(omega)
        else:
            return
        per_rad = pitch / (2.0 * math.pi)
        gear = w.sp[s].gear
        acc = sp.pos_max_accel[min(gear, len(sp.pos_max_accel) - 1)] if sp.pos_max_accel else None
        for j in it.pieces:
            pc = self.piece(j)
            if sp.max_omega > 0.0:
                pc.limit_v = min(pc.limit_v, sp.max_omega * per_rad)
            if acc:
                pc.amax = min(pc.amax, acc * per_rad)

    def _enter_motion(self, it: Item, w: PlanState, k: int) -> None:
        cur = w.cur
        if cur.piece < it.pieces[0] or cur.piece > it.pieces[-1] or (cur.piece == it.pieces[0] and cur.s == 0.0):
            omega = self._feed_omega(w)
            if it.feed_kind == "tap":
                self._tap_limits(it, w, omega)
            self._record_ideal(it, w, k, omega)
        cur.phase = PH_MOTION
        if cur.piece < it.pieces[0] or cur.piece > it.pieces[-1]:
            cur.piece = it.pieces[0]
            cur.s = cur.v = cur.a = 0.0
        if self.piece(cur.piece).feed in ("per_rev", "thread") and not cur.in_stretch and k > self.tp.k_from:
            # a stretch starts at the first tick of a plan call (the loop latches the actual angle there)
            cur.phase = PH_RESYNC
            w.rt.setdefault(it.index, {})["resume_motion"] = True

    def _arrival_wait(self, boundary: int, w: PlanState) -> int:
        """L1 ticks a cutting feed waits for the spindle speed arrival (predicted)."""
        sid = self.cut_spindle(w)
        s = self.model.spindle_index.get(sid) if sid is not None else None
        if s is None:
            return 0
        cmd = w.sp[s]
        if cmd.mode != SPM_SPEED or cmd.omega == 0.0:
            return 0
        sp = self.model.spindles[s]
        target = css_omega(cmd, self.css_radius(s, w.cur.q), sp, self.model.p["css_min_radius_m"])
        omega_now = self._omega_at(s, boundary, w)
        t = speed_change_s(sp, omega_now, target, cmd.gear, self.model.p["spindle_arrival_frac"])
        if not math.isfinite(t):
            return 0
        return ticks_of(t, self.env.ipo_s)

    def _begin_timed(self, cur: Cursor, boundary: int, ticks: int, cat: str) -> None:
        cur.phase = PH_TIMED
        cur.t_start = boundary
        cur.t_end = boundary + ticks * self.env.ipo_ticks
        cur.timed_cat = cat

    def _start_nonmotion(self, it: Item, w: PlanState, boundary: int, k: int) -> None:
        env = self.env
        cmd = it.cmd
        kind = cmd.kind
        timing = env.timing
        ipo_s = env.ipo_s
        tp = self.tp
        cur = w.cur
        if kind == Kind.MODE:
            modal = cmd.modal or {}
            if "rigid_tap" in modal:
                w.rigid_tap = bool(modal["rigid_tap"])
                if not w.rigid_tap:
                    # leaving rigid tapping: the spindle returns to speed control, stopped
                    for s, c in enumerate(w.sp):
                        if c.mode == SPM_POSITION and c.owner == self.p:
                            new = c.copy()
                            new.mode = SPM_SPEED
                            new.omega = 0.0
                            self._set_spindle(w, s, new, k)
            self._finish(it, w, k)
            return
        if kind == Kind.DWELL:
            sid = self.cut_spindle(w)
            s = self.model.spindle_index.get(sid) if sid is not None else None
            omega = self._omega_at(s, boundary, w) if s is not None else 0.0
            self._begin_timed(cur, boundary, ticks_of(timing.dwell_s(cmd, omega), ipo_s), CAT_DWELL)
            return
        if kind == Kind.SPINDLE:
            ticks = self._spindle_command(cmd, w, boundary, k, w.rt.setdefault(it.index, {}))
            self._begin_timed(cur, boundary, ticks, CAT_SPINDLE)
            return
        if kind == Kind.COOLANT:
            data = dict(cmd.coolant or {})
            pumps = dict(data.get("pumps") or {})
            turned_on = {pid: on for pid, on in pumps.items() if on and not w.pumps.get(pid, False)}
            w.pumps.update({str(k_): bool(v) for k_, v in pumps.items()})
            w.nozzles.update({str(k_): float(v) for k_, v in (data.get("nozzles") or {}).items()})
            tp.events.append((k, "coolant", {"pumps": dict(w.pumps), "nozzles": dict(w.nozzles)}))
            self._begin_timed(cur, boundary, ticks_of(timing.coolant_s(turned_on, env.pump_kinds), ipo_s),
                              CAT_COOLANT)
            return
        if kind in (Kind.TOOL_CHANGE, Kind.TURRET_INDEX):
            self._begin_timed(cur, boundary, self._tool_change(it, w, boundary, k), CAT_TOOL)
            return
        if kind == Kind.PALLET:
            self._begin_timed(cur, boundary, ticks_of(timing.pallet_s(), ipo_s), CAT_AUX)
            return
        if kind == Kind.DOOR:
            self._begin_timed(cur, boundary, ticks_of(timing.door_s_auto(), ipo_s), CAT_AUX)
            return
        if kind == Kind.CHUCK:
            self._begin_timed(cur, boundary, ticks_of(timing.chuck_s(), ipo_s), CAT_AUX)
            return
        if kind in (Kind.CONVEYOR,):
            self._begin_timed(cur, boundary, ticks_of(timing.m_function_s(), ipo_s), CAT_AUX)
            return
        if kind in (Kind.STOP, Kind.MESSAGE):
            stop = cmd.stop or ""
            if kind == Kind.STOP and stop in ("M02", "M30", "M99-end", "M2"):
                tp.events.append((k, "program_end", it.index))
                self._finish(it, w, k)
                cur.phase = PH_ENDED
                w.ended = True
                return
            seconds = timing.operator_stop_s(stop if kind == Kind.STOP else "message")
            self._begin_timed(cur, boundary, ticks_of(seconds, ipo_s), CAT_STOP)
            return
        if kind == Kind.SYNC:
            reason = (cmd.sync or {}).get("reason", "")
            if reason == "path_wait":
                # the motion before it is in position (inpos of the previous motion item): arrive now
                cur.phase = PH_WAIT_INPOS
                return
            tp.events.append((k, "sync_feedback", it.index))
            self._finish(it, w, k)
            cur.phase = PH_RESYNC
            return
        if kind == Kind.ALARM:
            w.alarm = dict(cmd.alarm or {})
            self._finish(it, w, k)
            cur.phase = PH_ALARM
            return
        self._begin_timed(cur, boundary, ticks_of(timing.m_function_s(), ipo_s), CAT_AUX)

    def _spindle_command(self, cmd: Command, w: PlanState, boundary: int, k: int, rt: dict | None = None) -> int:
        """Apply a SPINDLE command at boundary k; return its duration in L1 ticks. An orientation stops the
        spindle under speed control, then the position loop turns it to the orientation angle within the
        orientation time (rt["orient"]: spindle, angle, switch tick, duration; _orient_tick)."""
        env = self.env
        model = self.model
        data = cmd.spindle or {}
        sid = str(data.get("id") or self.cut_spindle(w) or "main")
        s = model.spindle_index.get(sid)
        if s is None:
            return ticks_of(env.timing.m_function_s(), env.ipo_s)
        sp = model.spindles[s]
        old = w.sp[s].copy()
        new = old.copy()
        omega = data.get("omega_rad_s")
        orient = data.get("orient_rad")
        seconds = env.timing.m_function_s()
        if data.get("gear") is not None and int(data["gear"]) >= 1 and sp.gears:
            gear = min(int(data["gear"]) - 1, len(sp.gears) - 1)
        else:
            gear = sp.gear_for(float(omega or 0.0))
        omega_now = self._omega_at(s, boundary, w)
        if orient is not None:
            stop = speed_change_s(sp, omega_now, 0.0, old.gear)
            new.mode = SPM_SPEED
            new.omega = 0.0
            new.css_vc = 0.0
            seconds = max(seconds, stop + sp.orient_s)
            if rt is not None and sp.orient_s > 0.0:
                switch = boundary + ticks_of(stop, env.ipo_s) * env.ipo_ticks
                rt["orient"] = [int(s), float(orient), int(switch), float(sp.orient_s)]
        else:
            new.mode = SPM_SPEED
            new.omega = float(omega or 0.0)
            new.css_vc = float(data.get("css_m_s") or 0.0)
            clamp = data.get("max_rad_s")
            new.clamp = float(clamp) if clamp else math.inf
            if gear != old.gear and len(sp.gears) > 1:
                stop = speed_change_s(sp, omega_now, 0.0, old.gear)
                seconds = max(seconds, stop + sp.gear_change_s)
            new.gear = gear
            if w.rigid_tap and sp.kind in ("milling", "live_tool", "turning"):
                # rigid tapping mode (M29): the speed is the tapping speed; the spindle stays position-controlled
                # at rest until the tap turns it in step with the feed axis
                new.mode = SPM_POSITION
                new.angle = self._angle_at(s, boundary)
                new.omega_ff = 0.0
        if sp.kind in ("turning", "sub"):
            w.work_spindle = sid
        elif sp.kind in ("milling", "live_tool"):
            w.tool_spindle = sid
        self._set_spindle(w, s, new, k)
        self.tp.events.append((k, "spindle", {"spindle": sid, "omega_rad_s": new.omega, "css_m_s": new.css_vc,
                                              "gear": new.gear, "orient": orient is not None}))
        return ticks_of(seconds, env.ipo_s)

    def _tool_change(self, it: Item, w: PlanState, boundary: int, k: int) -> int:
        env = self.env
        model = self.model
        cmd = it.cmd
        data = cmd.tool or {}
        number = data.get("number")
        slot = data.get("slot")
        if slot is None and number is not None:
            slot = env.slot_of(int(number), self.p)
        rt = w.rt.setdefault(it.index, {})
        rt["slot"] = int(slot) if slot is not None else -1
        rt["number"] = int(number) if number is not None else None
        timing = env.timing
        if self._turret_change(cmd):
            station = data.get("station") or number
            rt["station"] = station
            return ticks_of(timing.turret_index_s(w.station, station), env.ipo_s)
        stop_s = 0.0
        if w.tool_spindle is not None and w.tool_spindle in model.spindle_index:
            s = model.spindle_index[w.tool_spindle]
            sp = model.spindles[s]
            omega_now = self._omega_at(s, boundary, w)
            if w.sp[s].omega != 0.0 or abs(omega_now) > 0.0:
                stop_s = speed_change_s(sp, omega_now, 0.0, w.sp[s].gear) + sp.orient_s
            new = w.sp[s].copy()
            new.omega = 0.0
            new.css_vc = 0.0
            new.mode = SPM_SPEED
            self._set_spindle(w, s, new, k)
        z_axis = env.tool_change_axis(self.p)
        dog = []
        move_s = 0.0
        if z_axis is not None:
            ax = model.axes[z_axis]
            q0 = float(w.cur.q[z_axis])
            q1 = float(ax.home)
            if q1 != q0:
                prof = double_s(abs(q1 - q0), ax.rapid * max(self._rapid_scale(), 1e-6), ax.accel, ax.jerk)
                dog.append((z_axis, q0, q1, prof))
                move_s = prof.duration
        rt["dog"] = dog
        rt["prep_s"] = max(stop_s, move_s)
        return max(1, ticks_of(max(stop_s, move_s) + timing.tool_to_tool_s(), env.ipo_s))

    # ------------------------------------------------------------------ timed phases
    def _run_timed(self, w: PlanState, tp: TickPlan, i: int, n: int) -> int:
        env = self.env
        cur = w.cur
        it = self.item(cur.item)
        if cur.t_end <= self.tick0 + i * env.ipo_ticks:
            self._end_timed(it, w, i)
            return 0
        consumed = 0
        k = i
        rt = w.rt.get(cur.item, {})
        dog = rt.get("dog") if (it is not None and cur.timed_cat in (CAT_RAPID, CAT_TOOL)) else None
        if self.panel is not None and self.panel.machine_lock:
            dog = None                      # machine lock: the axes stand still, the program is still timed
        while k < n:
            boundary_end = self.tick0 + (k + 1) * env.ipo_ticks
            if boundary_end > cur.t_end:
                break
            if dog:
                t = (boundary_end - cur.t_start) * env.tick_s
                for a, q0, q1, prof in dog:
                    sgn = 1.0 if q1 >= q0 else -1.0
                    cur.q[a] = q0 + sgn * prof.position(t)
                if cur.timed_cat == CAT_RAPID:
                    tp.rapid[k] = 1
            tp.mode[k] = SP_TIME
            tp.owner[k] = cur.item
            tp.cat[k] = cur.timed_cat
            for a in self.axes:
                self.rows[k, a] = cur.q[a]
            if "orient" in rt:
                self._orient_tick(w, tp, k, boundary_end, rt)
            consumed += 1
            if boundary_end == cur.t_end:
                self._end_timed(it, w, k + 1)
            self._snap(tp, k, w)
            k += 1
            if cur.phase != PH_TIMED:
                break
        return consumed

    def _orient_tick(self, w: PlanState, tp: TickPlan, k: int, boundary_end: int, rt: dict) -> None:
        """Spindle orientation after the stop: the position command moves from the angle at the switch to the
        nearest angle equal to the orientation angle (mod 2 pi) on a quintic of zero end speed and
        acceleration over the orientation time; the spindle then holds it (position control)."""
        data = rt["orient"]
        s, angle, switch, duration = int(data[0]), float(data[1]), int(data[2]), float(data[3])
        if boundary_end <= switch:
            return
        if len(data) < 6:
            theta0 = self._angle_at(s, switch)
            rt["orient"] = data = [s, angle, switch, duration, theta0, math.remainder(angle - theta0, 2.0 * math.pi)]
        theta0, delta = float(data[4]), float(data[5])
        tau = min(1.0, (boundary_end - switch) * self.env.tick_s / duration)
        shape = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)
        rate = 30.0 * tau * tau * (1.0 - tau) ** 2 / duration
        old = w.sp[s]
        new = SpindleCmd(SPM_POSITION, 0.0, 0.0, old.clamp, old.gear, theta0 + delta * shape, delta * rate, self.p)
        tp.spindle.append((k, s, new))
        w.sp[s] = new
        if s not in w.sp_changed:
            w.sp_changed.append(s)

    def _end_timed(self, it: Item | None, w: PlanState, b: int) -> None:
        """A timed phase ended at boundary b."""
        cur = w.cur
        if it is None:
            cur.phase = PH_IDLE
            return
        rt = w.rt.get(it.index, {})
        if rt.get("c_engage") == "wait":
            rt["c_engage"] = "switch"
            cur.phase = PH_RESYNC
            return
        if it.is_motion and cur.timed_cat == CAT_SPINDLE:
            self._enter_motion(it, w, b)
            return
        kind = it.cmd.kind
        if it.dog is not None:
            if it.inpos:
                cur.phase = PH_WAIT_INPOS
                return
            self._finish(it, w, b)
            return
        if kind in (Kind.TOOL_CHANGE, Kind.TURRET_INDEX):
            rt = w.rt.get(it.index, {})
            self.tp.events.append((b, "tool", {"slot": rt.get("slot", -1), "number": rt.get("number"),
                                               "station": rt.get("station")}))
            w.tool_slot = rt.get("slot", -1)
            w.tool_number = rt.get("number")
            if rt.get("station") is not None:
                w.station = rt.get("station")
            self._finish(it, w, b)
            cur.phase = PH_RESYNC
            return
        if kind in (Kind.STOP, Kind.MESSAGE):
            self.tp.events.append((b, "cycle_start", it.index))
        self._finish(it, w, b)

    def _finish(self, it: Item, w: PlanState, b: int) -> None:
        """Item done at boundary b; the cursor moves to the next item."""
        self.tp.events.append((b, "done", it.index))
        w.rt.pop(it.index, None)
        w.cur.item = it.index + 1
        w.cur.phase = PH_IDLE

    # ------------------------------------------------------------------ motion runs
    def _run_end(self, first: int) -> int:
        j = first
        top = self.n_pieces() - 1
        kind0 = self.piece(first).feed
        while j < top:
            if self.piece(j).stop_end or self.piece(j + 1).feed != kind0:
                break
            j += 1
        return j

    def _run_motion(self, w: PlanState, tp: TickPlan, i: int, n: int) -> int:
        env = self.env
        cur = w.cur
        first = cur.piece
        if first >= self.n_pieces():
            cur.phase = PH_IDLE
            return 0
        last = self._run_end(first)
        pieces = [self.piece(j) for j in range(first, last + 1)]
        feed = pieces[0].feed
        if feed == "thread":
            return self._run_thread(w, tp, i, n, first)
        m = len(pieces)
        omega = self._feed_omega(w)          # per-revolution pieces move at the spindle's commanded speed
        hold = self.panel is not None and self.panel.feed_hold
        L = np.array([pc.length for pc in pieces])
        vmax = np.array([0.0 if hold else self.overrides(pc, self.item(pc.item), omega)[0] for pc in pieces])
        amax = np.array([pc.amax for pc in pieces])
        jmax = np.array([pc.jmax for pc in pieces])
        ustart = np.array([min(pc.ustart, vmax[j], vmax[j - 1]) if j > 0 else 0.0 for j, pc in enumerate(pieces)])
        ready = np.array([pc.ready for pc in pieces], np.int64)
        stop_end = np.array([1 if pc.stop_end else 0 for pc in pieces], np.int8)
        gi = np.zeros(1, np.int64)
        gf = np.array([cur.s, cur.v, cur.a])
        count_max = n - i
        op = np.zeros(count_max, np.int32)
        osv = np.zeros(count_max)
        ov = np.zeros(count_max)
        oa = np.zeros(count_max)
        olim = np.zeros(count_max, np.int8)
        oc = np.zeros(1, np.int32)
        t_first = self.tick0 + (i + 1) * env.ipo_ticks
        status = plan_ticks(L, vmax, amax, jmax, ustart, ready, stop_end, m, int(self.model.lookahead), gi, gf,
                            t_first, env.ipo_ticks, env.ipo_s, count_max, op, osv, ov, oa, olim, oc)
        count = int(oc[0])
        synchronous = feed == "per_rev"
        tap = feed == "tap"
        if synchronous:
            self._stretch(w, tp, i, count, first, last, osv, op)
            cur.in_stretch = True
        q_prev = cur.q.copy()
        locked = self.panel is not None and self.panel.machine_lock
        prev_j = first - 1
        for t in range(count):
            k = i + t
            j = first + int(op[t])
            pc = self.piece(j)
            for jj in range(prev_j + 1, j + 1):
                # items the run reaches (a blended or joined item never starts on its own): their ideal time
                nxt = self.item(self.piece(jj).item)
                if nxt is not None and nxt.index >= cur.item and "ideal_s" not in w.rt.get(nxt.index, {}):
                    self._record_ideal(nxt, w, k, omega)
            prev_j = max(prev_j, j)
            s = float(osv[t])
            v = float(ov[t])
            q = self._eval(pc, s, q_prev)
            q_prev = q
            if not locked:
                cur.q[:] = q
            for a in self.axes:
                self.rows[k, a] = cur.q[a]
            tp.mode[k] = SP_PER_REV if synchronous else SP_TIME
            tp.rapid[k] = 1 if pc.rapid else 0
            item = self.item(pc.item)
            is_probe = item is not None and item.cmd.kind == Kind.PROBE
            is_setter = item is not None and item.cmd.kind == Kind.TOOL_SETTER
            tp.probe[k] = 1 if is_probe and not item.triggered else 0
            tp.setter[k] = 1 if is_setter and not item.triggered else 0
            if is_setter and tp.setter_cfg is None:
                tp.setter_cfg = env.setter_config(self.p, w.tool_slot)
            if tap:
                tp.tap[k] = 1
                self._tap_angle(w, tp, k, pc, s, v)
            # items whose last piece ended by this tick (a synchronous stretch: decided at commit from the
            # actual progress)
            final_stop = status == 1 and t == count - 1
            for jj in range(cur.piece if cur.piece >= first else first, j + 1 if not synchronous else first):
                ppc = self.piece(jj)
                if jj < j or s >= ppc.length - 1e-12:
                    it = self.item(ppc.item)
                    if it is not None and it.index >= cur.item and it.pieces and it.pieces[-1] == jj:
                        if it.inpos and final_stop and jj == last:
                            continue
                        self.tp.events.append((k + 1, "done", it.index))
                        w.rt.pop(it.index, None)
                        cur.item = it.index + 1
            owner = pc.item
            cat = CAT_RAPID if pc.rapid else (CAT_PROBE if (is_probe or is_setter) else CAT_FEED)
            standing = v == 0.0 and j == cur.piece and s == cur.s
            if standing and not final_stop and not (status == 2 and t == count - 1):
                # the generator holds: feed hold, or the next block is not ready yet (data starvation)
                if hold:
                    cat = CAT_HOLD
                else:
                    cat = CAT_STARVE
                    if s >= pc.length - 1e-12 and cur.item > pc.item:
                        owner = cur.item if self.item(cur.item) is not None else -1
            tp.owner[k] = owner
            tp.cat[k] = cat
            tp.lim[k] = int(olim[t])
            cur.piece = j
            cur.s = s
            cur.v = v
            cur.a = float(oa[t])
            if t == count - 1 and status in (1, 2) and not synchronous:
                self._run_finished(w, last, k + 1)
            self._snap(tp, k, w)
        if synchronous:
            # the rest of the step stays in the stretch: the loop ends it at the actual end
            for k in range(i + count, n):
                tp.mode[k] = SP_PER_REV
                tp.owner[k] = self.piece(last).item
                tp.cat[k] = CAT_FEED
                for a in self.axes:
                    self.rows[k, a] = cur.q[a]
                self._snap(tp, k, w)
            cur.in_stretch = True
            return n - i
        if count == 0 and status in (1, 2):
            self._run_finished(w, last, i)
        return count

    def _run_finished(self, w: PlanState, last: int, b: int) -> None:
        """The run stopped at the end of piece ``last`` at boundary b."""
        cur = w.cur
        pc = self.piece(last)
        it = self.item(pc.item)
        cur.piece = last + 1
        cur.s = cur.v = cur.a = 0.0
        cur.in_stretch = False
        if it is None:
            cur.phase = PH_IDLE
            return
        if it.index < cur.item:
            cur.phase = PH_IDLE
            return
        cur.item = it.index
        if it.inpos:
            cur.phase = PH_WAIT_INPOS
            return
        self._finish(it, w, b)

    def _eval(self, pc: Piece, s: float, q_ref: np.ndarray) -> np.ndarray:
        if pc.kind == PK_TIP:
            u = s / pc.length if pc.length > 0 else 1.0
            return self._tcp_q(pc.tip, u, q_ref)
        out = q_ref.copy()
        pc.eval1(s, self.axes_arr, out)
        return out

    def _tap_angle(self, w: PlanState, tp: TickPlan, k: int, pc: Piece, s: float, v: float) -> None:
        """Rigid tapping: the spindle angle follows the tap's path position (pitch per revolution), turning
        the programmed way going in and reversed coming out (the same profile drives both)."""
        it = self.item(pc.item)
        sid = self.cut_spindle(w)
        if sid is None or it is None or sid not in self.model.spindle_index:
            return
        s_idx = self.model.spindle_index[sid]
        cmd = w.sp[s_idx]
        feed = it.cmd.feed
        pitch = 0.0
        if feed is not None and feed.mode == FeedMode.PER_REV:
            pitch = float(feed.value)
        elif feed is not None and cmd.omega != 0.0:
            pitch = float(feed.value) * 2.0 * math.pi / abs(cmd.omega)
        if pitch <= 0.0:
            return
        rt = w.rt.setdefault(it.index, {})
        if "tap_theta0" not in rt:
            rt["tap_theta0"] = self._angle_at(s_idx, self.tick0 + k * self.env.ipo_ticks)
            rt["tap_dir"] = self._tap_direction(it, cmd)
            rt["tap_s0"] = 0.0
        theta = rt["tap_theta0"] + rt["tap_dir"] * 2.0 * math.pi * s / pitch
        omega_ff = rt["tap_dir"] * 2.0 * math.pi * v / pitch
        new = SpindleCmd(SPM_POSITION, cmd.omega, 0.0, cmd.clamp, cmd.gear, theta, omega_ff, self.p)
        tp.spindle.append((k, s_idx, new))
        w.sp[s_idx] = new
        if s_idx not in w.sp_changed:
            w.sp_changed.append(s_idx)

    def _tap_direction(self, it: Item, cmd: SpindleCmd) -> float:
        kin = self.env.kin[self.p]
        q_from = self.piece(it.pieces[0]).q0
        q_to = self.piece(it.pieces[-1]).q1
        tip0, axis0 = kin.tip_axis_W(q_from)
        tip1, _ = kin.tip_axis_W(q_to)
        inward = float((tip1 - tip0) @ axis0) < 0.0
        sign = 1.0 if cmd.omega >= 0.0 else -1.0
        return sign if inward else -sign

    def _stretch(self, w: PlanState, tp: TickPlan, i: int, count: int, first: int, last: int, s_out, p_out) -> None:
        """Spindle-synchronous stretch: (theta, s) table at the nominal speed and the geometry of the pieces the
        loop can reach in this step (positions s along the stretch, from its first piece)."""
        cur = w.cur
        env = self.env
        sid = self.cut_spindle(w)
        s_idx = self.model.spindle_index.get(sid) if sid is not None else None
        omega = abs(self._feed_omega(w)) or 1e-9
        continuing = cur.in_stretch and self.stretch_info is not None
        if continuing:
            start, end = self.stretch_info["first"], self.stretch_info["last"]
        else:
            start, end = cur.piece, last
        base = {}
        acc = 0.0
        for j in range(start, end + 1):
            base[j] = acc
            acc += self.piece(j).length
        s_end = acc
        s_now = base.get(cur.piece, 0.0) + cur.s
        theta_now = abs(float(env.sp_actual.angle[s_idx]) - float(env.loop_theta0[self.p])) \
            if (continuing and s_idx is not None) else 0.0
        theta = [theta_now]
        svals = [s_now]
        reach = cur.piece
        for t in range(count):
            j = first + int(p_out[t])
            reach = max(reach, j)
            theta.append(theta_now + omega * (t + 1) * env.ipo_s)
            svals.append(base.get(j, 0.0) + float(s_out[t]))
        G = env.setp_geo_capacity
        top = min(end, reach + 8, cur.piece + G - 1)
        pieces = list(range(cur.piece, top + 1))
        s0 = [base[j] for j in pieces] + [base[top] + self.piece(top).length]
        tp.stretch = {"spindle": -1 if s_idx is None else int(s_idx), "pieces": pieces, "s0": s0, "s_end": s_end,
                      "theta": theta, "s": svals, "start": not continuing, "i": i}
        if not continuing:
            tp.sync_start[i] = 1
        self.stretch_info = {"first": start, "last": end}

    def _run_thread(self, w: PlanState, tp: TickPlan, i: int, n: int, j: int) -> int:
        cur = w.cur
        pc = self.piece(j)
        it = self.item(pc.item)
        thread = (it.cmd.modal or {}).get("thread", {}) if it is not None else {}
        lead = float(thread.get("lead_m", 0.0))
        trig = float(thread.get("start_angle_rad", 0.0)) % (2.0 * math.pi)
        sid = self.cut_spindle(w)
        s_idx = self.model.spindle_index.get(sid) if sid is not None else None
        tp.stretch = {"spindle": -1 if s_idx is None else int(s_idx), "pieces": [j], "s0": [0.0, pc.length],
                      "s_end": pc.length, "theta": [], "s": [], "start": not cur.in_stretch, "i": i,
                      "thread": (lead, trig, float(self.model.p["thread_filter_tc_s"]))}
        if not cur.in_stretch:
            tp.sync_start[i] = 1
        cur.in_stretch = True
        for k in range(i, n):
            tp.mode[k] = SP_THREAD
            tp.owner[k] = pc.item
            tp.cat[k] = CAT_FEED
            for a in self.axes:
                self.rows[k, a] = cur.q[a]
            self._snap(tp, k, w)
        return n - i

    def _rebuild(self, it: Item, w: PlanState, q_to: np.ndarray | None = None) -> None:
        """Rebuild the pieces of a queued motion item from the current commanded position (and the tool now
        in the spindle) to q_to, or to its programmed targets. An item that ends in a corner blend with the
        next block is rebuilt up to the blend, which stays (the next block keeps its start); the corner where
        the new piece meets the blend is bounded by the axes' velocity step. The new pieces take the place of
        the old ones (path order)."""
        cur = w.cur
        it.needs_rebuild = False
        cmd = it.cmd
        q0 = cur.q.copy()
        old = list(it.pieces)
        last = self.piece(old[-1]) if old else None
        tail = last if (last is not None and last.kind == PK_BLEND and last.blend == 1 and len(old) >= 2) else None
        if tail is not None:
            q1 = tail.q0.copy()
            for a in self._spindle_c_axes(it):
                q1[a] = q0[a]                 # a spindle's C axis outside C-axis mode turns with the spindle
        elif q_to is not None:
            q1 = np.asarray(q_to, dtype=float).copy()
        else:
            q1 = q0.copy()
            c_axes = {self.model.axis_index[self.model.spindles[s].c_axis] for s in it.c_spindles}
            for name, value in (cmd.target or {}).items():
                a = self.model.axis_index.get(name)
                if a is None:
                    continue
                v = float(value)
                ax = self.model.axes[a]
                if ax.spindle is not None and a not in c_axes:
                    continue                  # a spindle's C axis moves only in the blocks that program it
                if ax.rotary and ax.continuous:
                    v = q0[a] + math.remainder(v - q0[a], 2.0 * math.pi)
                q1[a] = v
        it.ideal_s = 0.0
        it.length_m = 0.0
        it.built_slot = w.tool_slot
        pieces = self._build_pieces(it, cmd, q0, q1, it.feed_kind, building=False)
        for pc in pieces:
            pc.ready = int(cmd.ready_tick)
        if tail is not None:
            if pieces:
                tail.ustart = min(tail.ustart, jump_speed(pieces[-1], tail, self.model, self.nc_axes,
                                                          self.env.ipo_s))
            it.length_m += tail.length
            it.ideal_s += tail.length / tail.ideal_v if tail.ideal_v > 0.0 else 0.0
            pieces.append(tail)
        elif pieces:
            pieces[-1].stop_end = True
        self._put_pieces(it, pieces, w)

    def _put_pieces(self, it: Item, pieces: list[Piece], w: PlanState) -> None:
        """An item's rebuilt pieces take the place of its old ones, so the pieces stay in path order; the
        piece indices of the items and cursors after them shift by the difference in count."""
        old = list(it.pieces)
        if old:
            at = old[0]
            n_old = old[-1] - old[0] + 1
        else:
            later = [x.pieces[0] for x in self.items if x.index > it.index and x.pieces]
            at = min(later) if later else self.n_pieces()
            n_old = 0
        lo = at - self.piece0
        end_old = at + n_old
        self.pieces[lo:lo + n_old] = pieces
        it.pieces = list(range(at, at + len(pieces)))
        delta = len(pieces) - n_old
        if delta == 0:
            return
        for other in self.items:
            if other is not it and other.pieces and other.pieces[0] >= end_old:
                other.pieces = [j + delta for j in other.pieces]
        for c in (w.cur, self.st.cur):
            if c.piece >= end_old:
                c.piece += delta
        if self.stretch_info is not None:
            self.stretch_info = {k: (v + delta if v >= end_old else v) for k, v in self.stretch_info.items()}

    # ================================================================== committing executed ticks
    def commit(self, tp: TickPlan, k_stop: int, code: int, out) -> list:
        """Ticks tp.k_from .. k_stop were executed; code is this path's sync at k_stop (0 none).
        Returns the committed events (boundary, kind, data) in order."""
        snap = tp.snaps[k_stop]
        if snap is None:
            snap = tp.final
        self.st = snap.copy()
        mark = int(tp.ev_mark[k_stop]) if (tp.ev_mark is not None and tp.snaps[k_stop] is not None) else -1
        if mark >= 0:
            events = list(tp.events[:mark])
        else:
            events = [e for e in tp.events if (e[0] <= k_stop if e[1] == "start" else e[0] <= k_stop + 1)]
        cur = self.st.cur
        if tp.stretch is not None and k_stop >= tp.stretch["i"] and cur.in_stretch:
            events.extend(self._commit_stretch(tp, k_stop, out, code))
        if code == SYNC_IN_POSITION or code == SYNC_PATH_WAIT:
            it = self.item(cur.item)
            if it is not None and cur.phase == PH_WAIT_INPOS:
                if it.cmd.kind == Kind.SYNC:
                    events.append((k_stop + 1, "sync_feedback", it.index))
                    events.append((k_stop + 1, "done", it.index))
                    cur.item = it.index + 1
                    cur.phase = PH_WAIT_CTRL if code == SYNC_PATH_WAIT else PH_RESYNC
                else:
                    if it.cmd.kind in (Kind.PROBE, Kind.TOOL_SETTER):
                        events.append((k_stop + 1, "probe_feedback", it.index))
                    events.append((k_stop + 1, "done", it.index))
                    cur.item = it.index + 1
                    cur.phase = PH_RESYNC if it.cmd.kind in (Kind.PROBE, Kind.TOOL_SETTER) else PH_IDLE
                self.st.rt.pop(it.index, None)
        elif code == SYNC_QUEUE_EMPTY:
            if cur.phase == PH_RESYNC:
                self._after_resync(self.st)
        elif code == SYNC_PROBE_TRIGGER:
            owner = int(tp.owner[k_stop])
            it = self.item(owner) if owner >= 0 else None
            if it is not None and it.cmd.kind in (Kind.PROBE, Kind.TOOL_SETTER) and not it.triggered:
                it.triggered = True
                it.latched_tick = k_stop
                it.latched = {self.model.axes[a].name: float(out.actual[k_stop, a]) for a in self.nc_axes}
                self._skip_stop(it)
                events.append((k_stop + 1, "trigger", it.index))
        elif code == SYNC_ALARM:
            cur.phase = PH_ALARM
            cur.q = out.actual[k_stop].copy()
            cur.v = cur.a = 0.0
            self.st.alarm = {"tick_index": int(k_stop)}
            self.drop_queue()
        if cur.phase == PH_ENDED:
            self.expect_more = False
        self.trim()
        return events

    def _skip_stop(self, it: Item) -> None:
        """Skip (G31) or setter trigger: the rest of the move is cancelled; the axes brake to rest along it."""
        cur = self.st.cur
        j = cur.piece
        if j >= self.n_pieces():
            return
        pc = self.piece(j)
        d = brake_distance(max(cur.v, 0.0), cur.a, 0.0, pc.amax, pc.jmax)
        new_len = min(pc.length, cur.s + d)
        if pc.geo == GEO_LINE and pc.kind == PK_JOINT and pc.length > 0:
            frac = new_len / pc.length
            pc.q1 = pc.q0 + (pc.q1 - pc.q0) * frac
            pc.vec[1] = pc.q1 - pc.q0
            pc.length = new_len
        pc.stop_end = True
        it.inpos = True
        it.pieces = [x for x in it.pieces if x <= j]
        if self.items and self.items[-1] is it:
            # nothing is queued past a skip move until its feedback (BUFFER_STOPPING): the next move starts where
            # the axes come to rest, not at the block's programmed end point (the command used to jump there)
            self.tail_q = np.asarray(pc.q1, dtype=float).copy()

    def _commit_stretch(self, tp: TickPlan, k_stop: int, out, code: int) -> list:
        info = tp.stretch
        events = []
        s_act = float(out.sync_s[k_stop, self.p])
        pieces = info["pieces"]
        s0 = info["s0"]
        cur = self.st.cur
        r = 0
        for j in range(len(pieces)):
            if s_act >= s0[j] - 1e-15:
                r = j
        cur.piece = pieces[r]
        cur.s = min(s_act - s0[r], self.piece(pieces[r]).length)
        cur.q = out.cmd[k_stop].copy()
        last_index = (self.stretch_info or {}).get("last", pieces[-1])
        for j in range(len(pieces)):
            if s_act >= s0[j + 1] - 1e-12:
                pc = self.piece(pieces[j])
                it = self.item(pc.item)
                if it is not None and it.index >= cur.item and it.pieces and it.pieces[-1] == pieces[j] and \
                        pieces[j] != last_index:
                    events.append((k_stop + 1, "done", it.index))
                    cur.item = it.index + 1
        if code == SYNC_SEGMENT_END or s_act >= info["s_end"] - 1e-12:
            last = self.piece(last_index)
            it = self.item(last.item)
            cur.piece = last_index + 1
            cur.s = cur.v = cur.a = 0.0
            cur.in_stretch = False
            self.stretch_info = None
            if it is not None and it.inpos:
                cur.item = it.index
                cur.phase = PH_WAIT_INPOS
            elif it is not None:
                events.append((k_stop + 1, "done", it.index))
                cur.item = it.index + 1
                cur.phase = PH_IDLE
        else:
            cur.phase = PH_MOTION
            if k_stop >= 1:
                cur.v = max(0.0, (float(out.sync_s[k_stop, self.p]) - float(out.sync_s[k_stop - 1, self.p]))
                            / self.env.ipo_s)
            cur.a = 0.0
        return events

    def drop_queue(self) -> None:
        """An alarm stopped the path: nothing queued will run."""
        self.st.cur.item = self.n_items()
        self.expect_more = False
