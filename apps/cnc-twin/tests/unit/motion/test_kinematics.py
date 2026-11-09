"""Kinematics: FK/IK round trips for every 5-axis type and a lathe, singularities, tilted planes, TCP."""

import copy
import math

import numpy as np
import pytest

from cnctwin.motion.abi import FK_M_T0, FK_S_T0, FK_W0_T0
from cnctwin.motion.ik import PathKinematics, sp2
from cnctwin.motion.machine import MachineModel
from mfix import LATHE, Script, five_axis, run, stack

TYPES = ["table_bc", "table_ac", "head_bc", "head_ab", "head_table_bc"]


def _kin(machine, tip=0.12):
    model = MachineModel(copy.deepcopy(machine))
    chain = model.build_chain([np.array([0.0, 0.0, -tip])], [np.eye(4)], [np.eye(4)], [-1], nominal=True)
    return model, PathKinematics(model, chain, 0)


def _random_q(model, rng, margin=0.15):
    q = np.array([ax.home for ax in model.axes])
    for ax in model.axes:
        if ax.synthetic:
            continue
        lo = ax.lo if math.isfinite(ax.lo) else -math.pi
        hi = ax.hi if math.isfinite(ax.hi) else math.pi
        if not ax.rotary:
            lo, hi = -0.3, 0.3
        q[ax.index] = lo + (hi - lo) * rng.uniform(margin, 1 - margin)
    return q


@pytest.mark.parametrize("kind", TYPES)
def test_fk_ik_round_trip_for_each_five_axis_type(kind):
    model, kin = _kin(five_axis(kind))
    assert len(kin.joints) == 2
    rng = np.random.default_rng(5)
    for _ in range(25):
        q = _random_q(model, rng)
        tip, axis = kin.tip_axis_W(q)
        ref = q.copy()
        for j in kin.joints:
            ref[j.axis] += 0.04
        res = kin.solve(tip, axis, ref)
        tip2, axis2 = kin.tip_axis_W(res.q)
        assert np.linalg.norm(tip2 - tip) < 1e-10
        assert np.linalg.norm(axis2 - axis) < 1e-10
        # the nearest solution is the one we started from (no 180 deg flip, no extra turn)
        for j in kin.joints:
            assert res.q[j.axis] == pytest.approx(q[j.axis], abs=1e-8)


def test_sp2_gives_both_solutions():
    w1 = np.array([0.0, 0.0, 1.0])
    w2 = np.array([0.0, 1.0, 0.0])
    p = np.array([0.0, 0.0, 1.0])
    q = np.array([math.sin(0.5) * math.cos(0.3), math.sin(0.5) * math.sin(0.3), math.cos(0.5)])
    sols = sp2(w1, w2, p, q, 1e-12)
    assert len(sols) == 2
    from cnctwin.motion.reference import axis_rotation
    for phi1, phi2, d1, d2 in sols:
        r = axis_rotation(w1, phi1) @ axis_rotation(w2, phi2) @ p
        assert np.allclose(r, q, atol=1e-12)


def test_lathe_fk_ik_and_rotating_workpiece_frames():
    model, kin = _kin(LATHE, tip=0.0)
    q = np.array([ax.home for ax in model.axes])
    ix, iz, ic = (model.axis_index[n] for n in ("X", "Z", "C"))
    q[ix], q[iz] = -0.12, -0.25
    poses = kin.fk(q)
    # chuck face at the origin, spindle along z: the tool tip radius is |150 mm + X|
    tip_M = poses[FK_M_T0][:3, 3]
    assert tip_M[0] == pytest.approx(0.15 - 0.12) and tip_M[2] == pytest.approx(0.3 - 0.25)
    # turning the spindle (C) leaves the tool in W0 and rotates it in S: pose_S_T = R_C^-1 pose_W0_T
    q2 = q.copy()
    q2[ic] = 0.7
    p2 = kin.fk(q2)
    assert np.allclose(p2[FK_W0_T0], poses[FK_W0_T0], atol=1e-14)
    from cnctwin.motion.reference import axis_rotation
    Rc = np.eye(4)
    Rc[:3, :3] = axis_rotation(np.array([0.0, 0.0, -1.0]), 0.7)
    assert np.allclose(p2[FK_S_T0], np.linalg.inv(Rc) @ poses[FK_W0_T0], atol=1e-14)
    # IK of the tip (orientation fixed on a lathe): X, Z found, C kept
    tip, axis = kin.tip_axis_W(q2)
    res = kin.solve(tip, axis, np.array([ax.home for ax in model.axes]) + np.r_[0.0, 0.0, 0.7])
    t2, _ = kin.tip_axis_W(res.q)
    assert np.linalg.norm(t2 - tip) < 1e-9


def test_singularity_detection_table_bc():
    model, kin = _kin(five_axis("table_bc"))
    # tool axis along the C axis (B = 0): singular, C keeps its reference value
    q = np.array([ax.home for ax in model.axes])
    q[model.axis_index["C"]] = 1.234
    tip, axis = kin.tip_axis_W(q)
    res = kin.solve(tip, axis, q)
    assert res.singular
    assert res.q[model.axis_index["C"]] == pytest.approx(1.234)
    assert kin.singular_at(axis) == [model.axis_index["C"]]
    # away from it: regular
    q[model.axis_index["B"]] = 0.3
    tip, axis = kin.tip_axis_W(q)
    assert not kin.solve(tip, axis, q).singular
    assert kin.singular_at(axis) == []


@pytest.mark.parametrize("rule,sign", [("positive", 1.0), ("negative", -1.0)])
def test_tilted_plane_rule_selects_the_master_axis_sign(rule, sign):
    model, kin = _kin(five_axis("table_bc"))
    b = model.axis_index["B"]
    q = np.array([ax.home for ax in model.axes])
    # a plane tilted 30 deg about W's y axis: two rotary solutions (B = +-30 deg with C 180 deg apart)
    a = math.radians(30.0)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    res = kin.tilted_plane(R, q, rule)
    _, axis = kin.tip_axis_W(res.q)
    assert np.allclose(axis, R[:, 2], atol=1e-10)
    if rule == "negative":
        pytest.skip("the B travel [-35, 120] deg allows -30 deg: checked below") if False else None
    assert math.copysign(1.0, res.q[b]) == sign or abs(res.q[b]) < 1e-12


@pytest.mark.parametrize("tcp_type", ["rotary", "vector"])
def test_tcp_motion_keeps_the_tip_on_the_programmed_line(tcp_type):
    m = five_axis("table_bc")
    s = Script()
    s.rapid(X=-50, Y=-40, Z=-200, B_deg=0.0, C_deg=0.0)
    # machine targets of the block end; the control (TCP) moves the tip on a straight line in W, starting
    # at the B = 0 singularity (C undetermined)
    cmd = s.tcp_line(2000, X=-30, Y=-20, Z=-190, B_deg=25.0, C_deg=40.0)
    if tcp_type == "vector":
        cmd.modal["tcp_type"] = "vector"
    s.stop()
    res, dom, state, ctrl = run(m, s)
    kin = dom.kin[0]
    q = stack(res.frames, "axis_cmd")
    blk = stack(res.frames, "block_index")
    names = res.frames[0].paths[0].axis_names
    idx = [dom.model.axis_index[n] for n in names]
    full = np.zeros((len(q), len(dom.model.axes)))
    full[:, idx] = q
    sel = np.nonzero(blk == 1)[0]
    pl = dom.planners[0]
    tp = pl.piece(pl.item(1).pieces[-1]).tip
    assert dom.timeline.records[1].tick1 > 0 and not state.findings.items or all(
        f.category.value == "singularity" for f in state.findings)
    tips = np.array([kin.tip_axis_W(full[k])[0] for k in sel])
    # every setpoint's tip on the segment between the end tips (TCP), within the IK tolerance
    d = tp.p1 - tp.p0
    u = (tips - tp.p0) @ d / (d @ d)
    off = tips - (tp.p0 + u[:, None] * d)
    assert np.max(np.linalg.norm(off, axis=1)) < 1e-9
    assert np.all(np.diff(u) >= -1e-12) and u[-1] == pytest.approx(1.0, abs=1e-9)
    # rotary axis speeds stay within the machine's limit (the singular start slows the move down)
    c = full[sel, dom.model.axis_index["C"]]
    assert np.max(np.abs(np.diff(c))) / 1e-3 <= dom.model.axis("C").rapid * 1.0001
    # while the rotary axes turn (a joint move would swing the tip far off the line)
    assert abs(full[sel[-1], dom.model.axis_index["B"]] - math.radians(25.0)) < 1e-9


def test_source_keys_beside_the_axes_are_not_axes():
    """A sourced axes mapping carries src / note / notes / page beside the axis entries (the SIEMENS DMU 50 file
    gets a src there from its same_mechanics_as source): the motion model reads only the axis definitions."""
    m = five_axis("table_bc")
    m["axes"]["src"] = "dmgmori-dmu50-3rd-2018"
    m["axes"]["note"] = "all axes as the HEIDENHAIN version"
    m["axes"]["notes"] = ["direct measuring"]
    m["axes"]["page"] = "p. 11"
    model = MachineModel(copy.deepcopy(m))
    assert [ax.name for ax in model.axes] == ["X", "Y", "Z", "B", "C"]
    assert model.paths[0].axes == ["X", "Y", "Z", "B", "C"]
    clean = MachineModel(copy.deepcopy(five_axis("table_bc")))
    assert [(a.name, a.rapid, a.accel) for a in model.axes] == [(a.name, a.rapid, a.accel) for a in clean.axes]
    # and the database machine that showed it builds a whole motion domain
    from cnctwin.db import Database
    from cnctwin.motion.domain import MotionDomain
    from cnctwin.core.state import SimState
    from mfix import context
    machine = Database.load().get("machine", "dmg-dmu50-3rd-siemens")
    assert "src" in list(machine.axes.keys())
    dom = MotionDomain(machine, tool_lengths={0: 0.1}, rotating={0: True}, tick_s=1e-6, ipo_ticks=2000, macro_ipo=10)
    dom.init(context(), SimState())
    assert [ax.name for ax in dom.model.axes] == ["X", "Y", "Z", "B", "C"]
