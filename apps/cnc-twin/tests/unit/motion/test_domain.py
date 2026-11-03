"""MotionDomain with a fake bus: thermal and holding error transforms, cutting load, findings,
PathFrame contents, checkpoint round trip, settings and channels."""

import copy
import json
import math

import numpy as np
import pytest

from cnctwin.core.bus import CuttingContact, PartHolding, RemovalContact, ThermalFeedback
from cnctwin.core.channels import ChannelRegistry
from cnctwin.core.findings import Category
from cnctwin.core.state import SimState
from cnctwin.motion.driver import MotionRunner
from cnctwin.motion.schema import schema_node, settings
from cnctwin.motion.state import MotionState
from mfix import MILL, Script, ScriptedController, context, domain, run, stack


def _hold_program(x=-100.0, z=-150.0):
    s = Script()
    s.rapid(X=x, Y=-100, Z=z)
    s.dwell(s=0.3)
    s.stop()
    return s


class BusRunner(MotionRunner):
    """MotionRunner that writes a fixed bus_prev record before every step (stands in for other domains)."""

    def __init__(self, *a, inject=None, **kw):
        super().__init__(*a, **kw)
        self.inject = inject

    def run(self, max_s=60.0):
        dom = self.dom
        orig = dom.kernel_inputs

        def kernel_inputs(ctx, state, bus):
            if self.inject is not None and int(state.motion.meta["k_next"]) == 0:
                self.inject(state.bus_prev, state)
            return orig(ctx, state, bus)

        dom.kernel_inputs = kernel_inputs
        return super().run(max_s)


def _run_with(inject, machine=MILL, program=None):
    ctrl = ScriptedController([program or _hold_program()])
    dom = domain(machine)
    state = SimState()
    res = BusRunner(ctrl, dom, ctx=context(), state=state, keep_frames=True, inject=inject).run(60)
    return res, dom, state


def test_thermal_screw_strain_moves_the_tip_by_the_error_transform():
    strain = 2e-5

    def inject(bus, state):
        th = ThermalFeedback()
        th.axis_drift_strain = {"X": strain}
        th.axis_drift_ref_m = {"X": 0.0}
        th.axis_drift = {"X": np.zeros(6)}
        bus.thermal = th

    res0, dom0, _ = _run_with(None)
    res1, dom1, _ = _run_with(inject)
    tip0 = stack(res0.frames, "tip_S_m")[-1]
    tip1 = stack(res1.frames, "tip_S_m")[-1]
    # the table (X link, direction -x) is displaced by strain * (X - X_ref) along its direction: the tool tip
    # relative to the work moves the other way
    x = -0.1
    assert tip1 - tip0 == pytest.approx(np.array([strain * x, 0.0, 0.0]), abs=1e-12)
    # the commanded axes are the same: drift is an error transform, never an axis offset
    assert np.array_equal(stack(res0.frames, "axis_cmd"), stack(res1.frames, "axis_cmd"))


def test_spindle_growth_and_part_pose_delta():
    growth = np.array([0.0, 0.0, -25e-6])          # nose toward the work

    def inject(bus, state):
        th = ThermalFeedback()
        th.spindle_growth_M_m = {"main": growth.copy()}
        bus.thermal = th
        pose = np.eye(4)
        pose[:3, 3] = [40e-6, 0.0, 0.0]
        bus.holding.parts = {0: PartHolding(pose_delta=pose)}

    res0, _, _ = _run_with(None)
    res1, _, _ = _run_with(inject)
    tM0 = stack(res0.frames, "tip_M_m")[-1]
    tM1 = stack(res1.frames, "tip_M_m")[-1]
    tS0 = stack(res0.frames, "tip_S_m")[-1]
    tS1 = stack(res1.frames, "tip_S_m")[-1]
    assert tM1 - tM0 == pytest.approx(growth, abs=1e-12)
    # the part slipped +40 um in x: the tool tip relative to the stock moves -40 um (and the growth)
    assert tS1 - tS0 == pytest.approx(np.array([-40e-6, 0.0, -25e-6]), abs=1e-12)


def test_cutting_load_enters_the_drive_as_a_disturbance():
    force = 600.0

    def inject(bus, state):
        frame = bus.motion.paths[0] if bus.motion.paths else None
        n = len(frame.axis_names) if frame is not None else 3
        cc = CuttingContact(contact=0, tool_slot=0)
        cc.axis_force_N = np.tile([force, 0.0, 0.0][:n], (10, 1))
        bus.cutting.contacts = [cc]
        bus.removal.contacts = [RemovalContact(contact=0, tool_slot=0, path=0)]

    res, dom, state = _run_with(inject)
    f = stack(res.frames, "axis_force_N")[:, 0]
    # at rest the drive holds the load: the demanded force is the reaction to it
    assert f[-1] == pytest.approx(-force, rel=1e-6)
    e = stack(res.frames, "following_error")[:, 0]
    assert np.max(np.abs(e[-100:])) < 1e-6        # the velocity integrator removes the static deflection


def test_overtravel_raises_a_finding_and_a_controller_alarm():
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    s.line(3000, X=-100, Y=-100, Z=-520)            # Z travel is [-500, 0] mm
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    found = [f for f in state.findings if f.category == Category.OVERTRAVEL]
    assert len(found) == 1
    alarms = [fb for _, fb in ctrl.feedbacks[0] if fb is not None and fb.alarm]
    assert alarms and "OVERTRAVEL" in alarms[0].alarm["text"]
    z = stack(res.frames, "axis_cmd")[:, 2]
    # the stroke check stops the axis just past the limit and holds it there
    assert z.min() > -0.5 - 0.01
    assert abs(z[-1] - z[-2]) < 1e-12


def test_axis_overload_finding():
    m = copy.deepcopy(MILL)
    m["axes"]["X"]["motor_force_N"] = 800.0          # 300 kg at 5 m/s2 needs 1500 N
    s = Script()
    s.rapid(X=-300, Y=0, Z=0)
    s.stop()
    res, dom, state, ctrl = run(m, s)
    found = [f for f in state.findings if f.category == Category.AXIS_OVERLOAD]
    assert found and abs(found[0].quantities["force_N"]) > 800.0


def test_pathframe_contents():
    res, dom, state, ctrl = run(MILL, _hold_program())
    f = res.frames[3].paths[0]
    n = 10
    assert f.tick.shape == (n,) and f.tick.dtype == np.int64
    assert f.axis_names == ("X", "Y", "Z")
    assert f.axis_actual.shape == (n, 3) and f.axis_cmd.shape == (n, 3) and f.following_error.shape == (n, 3)
    assert f.tip_S_m.shape == (n, 3) and f.tip_M_m.shape == (n, 3) and f.axis_T_M.shape == (n, 3)
    assert np.allclose(np.linalg.norm(f.axis_T_M, axis=1), 1.0)
    assert np.all((f.spindle_angle_rad >= 0) & (f.spindle_angle_rad < 2 * math.pi))
    assert f.pose_M_T.shape == (4, 4) and f.pose_S_T.shape == (4, 4)
    assert f.axis_force_N.shape == (n, 3) and f.axis_inertial_force_N.shape == (n, 3)
    assert np.all(np.diff(stack(res.frames, "tick")) == 1000)
    # the chain: X and Y move the table (-x, -y for +X, +Y), Z the head with the nose at 600 mm; no tool is in
    # the spindle in this program, so the control point is the gauge line
    last_M = stack(res.frames, "tip_M_m")[-1]
    last_S = stack(res.frames, "tip_S_m")[-1]
    assert last_M == pytest.approx([0.0, 0.0, 0.6 - 0.15], abs=1e-9)
    assert last_S == pytest.approx([-0.1, -0.1, 0.6 - 0.15], abs=1e-9)      # the tool relative to the work


def _copy_state(state):
    arrays, meta = state.motion.to_arrays()
    meta = json.loads(json.dumps(meta, allow_nan=False))
    s2 = SimState()
    s2.clock = copy.deepcopy(state.clock)
    s2.ledger = copy.deepcopy(state.ledger)
    s2.bus_prev = copy.deepcopy(state.bus_prev)
    s2.motion = MotionState.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta)
    return s2


def test_checkpoint_round_trip_continues_bit_identically():
    prog = Script(ready_step_ticks=500)
    prog.spindle(3000)
    prog.rapid(X=-100, Y=-100, Z=-150)
    prog.line(1200, X=-80, Y=-100, Z=-150)
    prog.arc(1500, centre_mm=(-80, -90, -150), sweep_rad=math.pi, radius_mm=10, X=-80, Y=-80, Z=-150)
    prog.line(1200, X=-100, Y=-80, Z=-150)
    prog.stop()
    ctrl = ScriptedController([prog])
    dom = domain(MILL)
    state = SimState()
    runner = MotionRunner(ctrl, dom, ctx=context(), state=state, keep_frames=True)
    runner.run(max_s=0.9)                       # stops at a step boundary inside the cut
    assert int(state.motion.meta["k_next"]) == 0
    s2 = _copy_state(state)
    ctrl2 = copy.deepcopy(ctrl)
    dom2 = domain(MILL)
    r1 = MotionRunner(ctrl, dom, ctx=context(), state=state, keep_frames=True).run(60)
    r2 = MotionRunner(ctrl2, dom2, ctx=context(), state=s2, keep_frames=True).run(60)
    assert r1.end_tick == r2.end_tick
    for name in ("axis_actual", "axis_cmd", "tip_S_m", "spindle_omega_rad_s", "block_index"):
        assert np.array_equal(stack(r1.frames, name), stack(r2.frames, name)), name
    a = [r.to_dict() for r in dom.timeline.records]
    b = [r.to_dict() for r in dom2.timeline.records]
    assert a == b


def _lathe_c_program():
    prog = Script(ready_step_ticks=500)
    prog.rapid(X=-50, Z=-100, C_deg=0)
    prog.spindle(1000)
    prog.dwell(s=1.0)
    prog.rapid(C_deg=90)                 # C-axis engage, then the C rapid
    prog.line(3600, C_deg=180)
    prog.orient(angle_deg=30)            # leaves C-axis mode: orientation to 30 degrees
    prog.spindle(500)
    prog.dwell(s=0.2)
    prog.stop()
    return prog


@pytest.mark.parametrize("cut_s", [1.7, 2.6, 4.5, 5.8])   # engage wait, C rapid, C feed, orientation
def test_checkpoint_round_trip_through_c_axis_engage_and_orientation(cut_s):
    from mfix import LATHE
    ctrl = ScriptedController([_lathe_c_program()])
    dom = domain(LATHE)
    state = SimState()
    MotionRunner(ctrl, dom, ctx=context(), state=state, keep_frames=True).run(max_s=cut_s)
    assert int(state.motion.meta["k_next"]) == 0
    s2 = _copy_state(state)
    ctrl2 = copy.deepcopy(ctrl)
    dom2 = domain(LATHE)
    r1 = MotionRunner(ctrl, dom, ctx=context(), state=state, keep_frames=True).run(60)
    r2 = MotionRunner(ctrl2, dom2, ctx=context(), state=s2, keep_frames=True).run(60)
    assert ctrl.done() and ctrl2.done()
    assert r1.end_tick == r2.end_tick
    for name in ("axis_actual", "axis_cmd", "spindle_angle_rad", "spindle_omega_rad_s", "block_index"):
        assert np.array_equal(stack(r1.frames, name), stack(r2.frames, name)), name
    assert [r.to_dict() for r in dom.timeline.records] == [r.to_dict() for r in dom2.timeline.records]


def test_settings_schema_and_channels():
    st = settings({"servo_omega_h": 0.1, "operator_stops": False})
    assert st.servo_omega_h == 0.1 and st.operator_stops is False
    with pytest.raises(Exception):
        settings({"servo_omega_h": 5.0})
    assert schema_node() is not None
    reg = ChannelRegistry()
    dom = domain(MILL)
    reg.register_all(dom.channels())
    assert "motion.following_error" in reg and "motion.cycle_time_s" in reg
