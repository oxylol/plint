"""Inline fixtures for the motion tests: small machine definitions, a scripted controller and runners."""

from __future__ import annotations

import copy
import math

import numpy as np

from cnctwin.controller.ir import ArcSpec, Command, Feed, FeedMode, Kind, LineRef, Smoothing, Space
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState

TICK_S = 1e-6
IPO = 1000
MACRO = 10


def _axis(travel, rapid=30000, feed=20000, accel=5.0, jerk=200.0, kv=40.0, ff=0.9, bw=80.0, mass=300.0, force=10000.0,
          backlash=0.0, spike=0.0):
    return {"type": "linear", "travel_mm": travel, "rapid_mm_min": rapid, "max_feed_mm_min": feed,
            "accel_m_s2": accel, "jerk_m_s3": jerk, "servo": {"kv_1_s": kv, "feedforward": ff, "velocity_bw_Hz": bw},
            "moving_mass_kg": mass, "motor_force_N": force, "backlash_um": backlash, "reversal_spike_um": spike}


def _rot(travel=None, continuous=False, rpm=20.0, accel=10.0, jerk=500.0, kv=30.0, ff=0.9, bw=40.0):
    d = {"type": "rotary", "max_rpm": rpm, "accel_rad_s2": accel, "jerk_rad_s3": jerk,
         "servo": {"kv_1_s": kv, "feedforward": ff, "velocity_bw_Hz": bw}, "backlash_arcsec": 0.0,
         "motor_torque_Nm": 2000.0, "inertia_kgm2": 20.0}
    if continuous:
        d["continuous"] = True
        d["travel_deg"] = [0, 360]
    else:
        d["travel_deg"] = travel
    return d


SPINDLE = {"id": "main", "kind": "milling", "nose": "CAT40", "max_rpm": 12000,
           "gear_ranges": [{"rpm": [0, 12000], "power_kW": [[0, 0], [1500, 15], [12000, 15]],
                            "torque_Nm": [[0, 100], [1500, 100], [12000, 12]]}],
           "accel_time_s": 1.0, "decel_time_s": 1.0, "orient_time_s": 0.5, "rotor_inertia_kgm2": 0.02,
           "idle_power_W": 300, "position_loop": {"kv_1_s": 30, "feedforward": 0.9, "velocity_bw_Hz": 60}}

CONTROLLER = {"dialect": "fanuc", "model": "test", "alarms": "fixture", "ipo_period_ms": 1.0, "block_processing_ms": 1.0,
              "lookahead_blocks": 40, "least_input_increment_mm": 0.001, "least_input_increment_deg": 0.001,
              "max_feed_mm_min": 20000, "rapid_mm_min": {"X": 30000, "Y": 30000, "Z": 30000},
              "smoothing": [{"code": "G05.1 Q1", "name": "ai", "tolerance_mm": 0.01}], "in_position_mm": 0.005}

MILL = {
    "format": "cnctwin-machine", "version": 1, "id": "t-mill", "maker": "T", "model": "T", "type": "vertical_mill",
    "controller": copy.deepcopy(CONTROLLER),
    "axes": {"X": _axis([-800, 0]), "Y": _axis([-500, 0]), "Z": _axis([-500, 0])},
    "kinematics": {"workpiece_chain": [{"link": "saddle", "axis": "Y", "direction": [0, -1, 0]},
                                       {"link": "table", "axis": "X", "direction": [-1, 0, 0]}],
                   "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]},
                                  {"link": "nose", "fixed": {"translate_mm": [0, 0, 600]}}],
                   "home_mm": {"X": 0, "Y": 0, "Z": 0}, "tool_axis": [0, 0, 1]},
    "spindles": [copy.deepcopy(SPINDLE)],
    "tool_changer": {"type": "side_mount", "capacity": 24, "tool_to_tool_s": 2.0, "chip_to_chip_s": 3.0},
    "coolant": {"pumps": [{"id": "flood", "kind": "flood"}, {"id": "tsc", "kind": "through_spindle"}]},
    "probes": {"tool_setter": {"model": "t", "position_mm": [-400, -300, 100]}},
}


def five_axis(kind: str) -> dict:
    """Five-axis variants: table_bc, table_ac, head_bc, head_ab, head_table_bc."""
    m = copy.deepcopy(MILL)
    m["id"] = f"t-{kind}"
    m["type"] = "five_axis_mill"
    lin_tool = [{"link": "column", "axis": "X", "direction": [1, 0, 0]},
                {"link": "ram", "axis": "Y", "direction": [0, 1, 0]},
                {"link": "head", "axis": "Z", "direction": [0, 0, 1]}]
    if kind == "table_bc":
        m["axes"]["B"] = _rot([-35, 120])
        m["axes"]["C"] = _rot(continuous=True)
        work = [{"link": "trunnion", "axis": "B", "axis_point_mm": [0, 0, -50], "direction": [0, -1, 0]},
                {"link": "platter", "axis": "C", "axis_point_mm": [0, 0, 0], "direction": [0, 0, -1]}]
        tool = lin_tool + [{"link": "nose", "fixed": {"translate_mm": [0, 0, 400]}}]
        m["kinematics"]["singularities"] = [{"axis": "C", "when": "B = 0", "behaviour": "slow_down"}]
    elif kind == "table_ac":
        m["axes"]["A"] = _rot([-120, 30])
        m["axes"]["C"] = _rot(continuous=True)
        work = [{"link": "cradle", "axis": "A", "axis_point_mm": [0, 0, -80], "direction": [-1, 0, 0]},
                {"link": "platter", "axis": "C", "axis_point_mm": [0, 0, 0], "direction": [0, 0, -1]}]
        tool = lin_tool + [{"link": "nose", "fixed": {"translate_mm": [0, 0, 400]}}]
    elif kind == "head_bc":
        m["axes"]["C"] = _rot([-360, 360])
        m["axes"]["B"] = _rot([-110, 110])
        work = [{"link": "table", "fixed": {"translate_mm": [0, 0, 0]}}]
        tool = lin_tool + [{"link": "c_head", "axis": "C", "axis_point_mm": [0, 0, 500], "direction": [0, 0, 1]},
                           {"link": "b_fork", "axis": "B", "axis_point_mm": [0, 0, 300], "direction": [0, 1, 0]},
                           {"link": "nose", "fixed": {"translate_mm": [0, 0, 150]}}]
    elif kind == "head_ab":
        m["axes"]["A"] = _rot([-100, 100])
        m["axes"]["B"] = _rot([-100, 100])
        work = [{"link": "table", "fixed": {"translate_mm": [0, 0, 0]}}]
        tool = lin_tool + [{"link": "a_head", "axis": "A", "axis_point_mm": [0, 0, 400], "direction": [1, 0, 0]},
                           {"link": "b_head", "axis": "B", "axis_point_mm": [0, 0, 300], "direction": [0, 1, 0]},
                           {"link": "nose", "fixed": {"translate_mm": [0, 0, 150]}}]
    elif kind == "head_table_bc":
        m["axes"]["B"] = _rot([-110, 110])
        m["axes"]["C"] = _rot(continuous=True)
        work = [{"link": "rotary_table", "axis": "C", "axis_point_mm": [0, 0, 0], "direction": [0, 0, -1]}]
        tool = lin_tool + [{"link": "b_head", "axis": "B", "axis_point_mm": [0, 0, 350], "direction": [0, 1, 0]},
                           {"link": "nose", "fixed": {"translate_mm": [0, 0, 150]}}]
    else:
        raise ValueError(kind)
    m["kinematics"] = {"workpiece_chain": work, "tool_chain": tool,
                       "home_mm": {"X": 0, "Y": 0, "Z": 0}, "home_deg": {n: 0 for n in m["axes"] if
                                                                        m["axes"][n]["type"] == "rotary"},
                       "tool_axis": [0, 0, 1], "singularities": m["kinematics"].get("singularities")}
    m["axes"]["X"]["travel_mm"] = [-800, 800]
    m["axes"]["Y"]["travel_mm"] = [-800, 800]
    m["axes"]["Z"]["travel_mm"] = [-800, 800]
    m["controller"]["rapid_mm_min"] = {k: 30000 for k in m["axes"]}
    return m


LATHE = {
    "format": "cnctwin-machine", "version": 1, "id": "t-lathe", "maker": "T", "model": "T", "type": "lathe",
    "controller": dict(copy.deepcopy(CONTROLLER), rapid_mm_min={"X": 20000, "Z": 24000, "C": 1}, smoothing=[]),
    "axes": {"X": _axis([-200, 50], rapid=20000, feed=12000), "Z": _axis([-500, 0], rapid=24000, feed=12000),
             "C": _rot(continuous=True, rpm=60.0)},
    "kinematics": {"workpiece_chain": [{"link": "spindle", "axis": "C", "direction": [0, 0, -1]},
                                       {"link": "chuck_face", "fixed": {"translate_mm": [0, 0, 0]}}],
                   "tool_chain": [{"link": "carriage", "axis": "Z", "direction": [0, 0, 1]},
                                  {"link": "slide", "axis": "X", "direction": [1, 0, 0]},
                                  {"link": "tip_ref", "fixed": {"translate_mm": [150, 0, 300]}}],
                   "home_mm": {"X": 0, "Z": 0}, "home_deg": {"C": 0}, "tool_axis": [0, 0, 1]},
    "spindles": [{"id": "main", "kind": "turning", "nose": "A2-6", "max_rpm": 4000,
                  "gear_ranges": [{"rpm": [0, 4000], "power_kW": [[0, 0], [800, 15], [4000, 15]],
                                   "torque_Nm": [[0, 180], [800, 180], [4000, 36]]}],
                  "accel_time_s": 1.5, "decel_time_s": 1.5, "rotor_inertia_kgm2": 0.2, "idle_power_W": 500,
                  "position_loop": {"kv_1_s": 30, "feedforward": 0.9, "velocity_bw_Hz": 40}}],
    "turrets": [{"id": "t1", "stations": 12, "index_time_s": {"adjacent": 0.4, "max": 1.0}}],
    "coolant": {"pumps": [{"id": "flood", "kind": "flood"}]},
}


def ref(line: int = 1, text: str = "") -> LineRef:
    return LineRef("t.nc", line, text)


class Script:
    """Builds commands with ready ticks and block numbers."""

    def __init__(self, ready_step_ticks: int = 0, path: int = 0):
        self.cmds: list[Command] = []
        self.ready = 0
        self.step = ready_step_ticks
        self.path = path
        self.seq = 0

    def _add(self, cmd: Command, same_block: bool = False) -> Command:
        if not same_block:
            self.seq += 1
            self.ready += self.step
        cmd.ready_tick = self.ready
        cmd.path = self.path
        cmd.block_seq = self.seq
        if cmd.ref.line == 1 and cmd.ref.block == "":
            cmd.ref = ref(self.seq, f"block {self.seq}")
        self.cmds.append(cmd)
        return cmd

    def rapid(self, **target):
        return self._add(Command(Kind.RAPID, ref(), Space.MACHINE, _si(target), feed=Feed(FeedMode.RAPID, 0.0)))

    def line(self, feed_mm_min: float, tol_mm: float = 0.0, exact: bool = False, per_rev_mm: float | None = None,
             modal=None, tol_kind: str = "deviation", **target):
        if per_rev_mm is not None:
            feed = Feed(FeedMode.PER_REV, per_rev_mm * 1e-3)
        else:
            feed = Feed(FeedMode.PER_MIN, feed_mm_min * 1e-3 / 60.0)
        sm = Smoothing("exact_stop" if exact else "blend", tol_mm * 1e-3, tolerance_kind=tol_kind)
        return self._add(Command(Kind.LINEAR, ref(), Space.MACHINE, _si(target), feed=feed, smoothing=sm,
                                 modal=dict(modal or {})))

    def arc(self, feed_mm_min: float, centre_mm, sweep_rad: float, radius_mm: float, normal=(0, 0, 1), tol_mm=0.0,
            **target):
        a = ArcSpec("XY", tuple(c * 1e-3 for c in centre_mm), tuple(normal), sweep_rad, radius_mm * 1e-3,
                    radius_mm * 1e-3)
        return self._add(Command(Kind.ARC, ref(), Space.MACHINE, _si(target), arc=a,
                                 feed=Feed(FeedMode.PER_MIN, feed_mm_min * 1e-3 / 60.0),
                                 smoothing=Smoothing("blend", tol_mm * 1e-3)))

    def tcp_line(self, feed_mm_min: float, **target):
        return self._add(Command(Kind.LINEAR, ref(), Space.TIP, _si(target),
                                 feed=Feed(FeedMode.PER_MIN, feed_mm_min * 1e-3 / 60.0), smoothing=Smoothing(),
                                 modal={"tip_interpolation": "tip"}))

    def spindle(self, rpm: float, sid: str = "main", css_m_min: float | None = None, clamp_rpm: float | None = None):
        omega = -rpm * 2 * math.pi / 60.0          # M03
        data = {"id": sid, "omega_rad_s": omega, "css_m_s": (css_m_min / 60.0) if css_m_min else None,
                "max_rad_s": clamp_rpm * 2 * math.pi / 60.0 if clamp_rpm else None, "gear": None, "orient_rad": None}
        return self._add(Command(Kind.SPINDLE, ref(), spindle=data))

    def orient(self, sid: str = "main", angle_deg: float = 0.0):
        return self._add(Command(Kind.SPINDLE, ref(), spindle={"id": sid, "omega_rad_s": 0.0,
                                                               "orient_rad": math.radians(angle_deg)}))

    def mode(self, **modal):
        return self._add(Command(Kind.MODE, ref(), modal=dict(modal)))

    def tap(self, pitch_mm: float, **target):
        return self._add(Command(Kind.LINEAR, ref(), Space.MACHINE, _si(target),
                                 feed=Feed(FeedMode.PER_REV, pitch_mm * 1e-3, override_allowed=False),
                                 smoothing=Smoothing("exact_stop"), modal={"rigid_tap": True, "cycle": "G84"}))

    def turret(self, number: int, slot: int | None = None):
        return self._add(Command(Kind.TOOL_CHANGE, ref(), tool={"number": number, "slot": slot, "station": number}))

    def dwell(self, s: float = 0.0, rev: float = 0.0):
        return self._add(Command(Kind.DWELL, ref(), dwell_s=s, dwell_rev=rev))

    def coolant(self, **pumps):
        return self._add(Command(Kind.COOLANT, ref(), coolant={"pumps": dict(pumps)}))

    def tool_change(self, number: int, slot: int | None = None):
        return self._add(Command(Kind.TOOL_CHANGE, ref(), tool={"number": number, "slot": slot}))

    def stop(self, code: str = "M30"):
        return self._add(Command(Kind.STOP, ref(), stop=code))

    def sync(self, reason: str = "live_read", paths=(0, 1), code="M100"):
        return self._add(Command(Kind.SYNC, ref(), sync={"reason": reason, "paths": list(paths), "code": code}))

    def probe(self, feed_mm_min: float, **target):
        return self._add(Command(Kind.PROBE, ref(), Space.MACHINE, _si(target),
                                 feed=Feed(FeedMode.PER_MIN, feed_mm_min * 1e-3 / 60.0), probe={"skip": 1}))


def _si(target: dict) -> dict:
    out = {}
    for k, v in target.items():
        if k.endswith("_deg"):
            out[k[:-4]] = math.radians(v)
        else:
            out[k] = v * 1e-3
    return out


class ScriptedController:
    """The controller lock-step API over fixed command lists (one per path).

    With ``block_ticks`` the interpreter clock is emulated as on the control: each block costs block_ticks,
    its ready tick is the clock after it; the look-ahead buffer holds ``capacity`` commands; a full buffer
    stops the interpreter until motion retires one (the clock resumes at the retire tick)."""

    def __init__(self, scripts: list[Script], block_ticks: int = 0, capacity: int = 1000, horizon_ticks: int = 10000):
        self.queues = [list(s.cmds) for s in scripts]
        self.given = [0 for _ in scripts]
        self.retired = [0 for _ in scripts]
        self.feedbacks: list = [[] for _ in scripts]
        self.starts: list = [[] for _ in scripts]
        self.retire_ticks: list = [[] for _ in scripts]
        self.executors = [None for _ in scripts]
        self.waiting = [False for _ in scripts]

        class _P:
            def __init__(self):
                from cnctwin.controller.ir import OperatorPanel
                self.operator = OperatorPanel()
                self.alarm = None
        self.state = type("S", (), {})()
        self.state.paths = [_P() for _ in scripts]
        self.capacity = capacity
        self.block_ticks = block_ticks
        self.horizon = horizon_ticks
        self.interp = [0 for _ in scripts]
        self.blocked = [False for _ in scripts]
        self.last_seq = [None for _ in scripts]
        self.arrived: dict = {}
        self.releases: list = []
        self.released_at: dict = {}

    def produce(self, p: int, now: int):
        out = []
        while self.given[p] < len(self.queues[p]) and not self.waiting[p]:
            cmd = self.queues[p][self.given[p]]
            if self.given[p] - self.retired[p] >= self.capacity:
                self.blocked[p] = True
                break
            if self.block_ticks:
                if self.interp[p] > now + self.horizon:
                    break
                if cmd.block_seq != self.last_seq[p]:
                    self.interp[p] += self.block_ticks
                    self.last_seq[p] = cmd.block_seq
                cmd.ready_tick = self.interp[p]
            self.given[p] += 1
            out.append(cmd)
            if cmd.kind in (Kind.SYNC, Kind.PROBE, Kind.TOOL_SETTER) or (cmd.kind == Kind.STOP and cmd.stop in
                                                                        ("M00", "M01")):
                self.waiting[p] = True
                break
        return out

    def retire(self, p: int, count: int, tick: int):
        for _ in range(count):
            self.retire_ticks[p].append(tick)
        self.retired[p] += count
        if self.blocked[p]:
            self.blocked[p] = False
            self.interp[p] = max(self.interp[p], tick)

    def feedback(self, p: int, fb, tick: int):
        self.feedbacks[p].append((tick, fb))
        if fb is not None and fb.alarm:
            self.state.paths[p].alarm = dict(fb.alarm)
        cmd = self.queues[p][self.given[p] - 1] if self.given[p] else None
        if cmd is not None and cmd.kind == Kind.SYNC and (cmd.sync or {}).get("reason") == "path_wait":
            # path wait: released when every path named has arrived, at the latest arrival
            self.arrived[p] = tick
            paths = list((cmd.sync or {}).get("paths") or [])
            if all(q in self.arrived for q in paths):
                release = max(self.arrived[q] for q in paths)
                self.releases.append((tuple(paths), release))
                for q in paths:
                    self.arrived.pop(q, None)
                    self.waiting[q] = False
                    self.released_at[q] = release
            return
        self.waiting[p] = False

    def cycle_start(self, p: int, tick: int):
        self.starts[p].append(tick)
        self.waiting[p] = False

    def done(self) -> bool:
        return all(self.given[p] >= len(self.queues[p]) and self.retired[p] >= len(self.queues[p])
                   or self.state.paths[p].alarm is not None for p in range(len(self.queues)))


def context() -> RunContext:
    return RunContext(project=None, db=None, run_dir=None, seed=0, tick_s=TICK_S, ipo_ticks=IPO, macro_ipo=MACRO)


def domain(machine: dict, tool_length_m: float = 0.1, **kw):
    from cnctwin.motion.domain import MotionDomain
    return MotionDomain(copy.deepcopy(machine), tool_lengths={0: tool_length_m, 1: tool_length_m},
                        rotating={0: machine["type"] not in ("lathe",), 1: machine["type"] not in ("lathe",)},
                        tool_numbers={(1, 0): 0, (2, 0): 1}, tick_s=TICK_S, ipo_ticks=IPO, macro_ipo=MACRO, **kw)


def run(machine: dict, scripts, max_s: float = 60.0, keep_frames: bool = True, dom=None, probe=None, parts=1,
        block_ticks: int = 0, capacity: int = 1000):
    from cnctwin.motion.driver import MotionRunner
    if isinstance(scripts, Script):
        scripts = [scripts]
    ctrl = ScriptedController(scripts, block_ticks=block_ticks, capacity=capacity)
    dom = dom or domain(machine)
    ctx = context()
    state = SimState()
    runner = MotionRunner(ctrl, dom, ctx=ctx, state=state, keep_frames=keep_frames, probe=probe, parts=parts)
    result = runner.run(max_s=max_s)
    return result, dom, state, ctrl


def stack(frames, field: str, p: int = 0) -> np.ndarray:
    return np.concatenate([getattr(f.paths[p], field) for f in frames], axis=0)
