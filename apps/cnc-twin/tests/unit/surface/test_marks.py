"""Marks: dwell (feed stops in the cut), spindle stopping in the cut, servo marks at axis reversals, recut dents,
chatter flags - recorded with their depth, location and program line, findings above the mark limit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.surface import surffix as fx
from cnctwin.core.findings import Category
from cnctwin.surface.domain import SurfaceDomain
from cnctwin.surface.store import PF

R = 5e-3
FZ = 0.05e-3
OMEGA = -2.0 * math.pi * 10000.0 / 60.0


def _wall_run(step_hook, n_steps=12, u_n=3e-6, kinds=("linear", "dwell"), physics=None, tolerances=None):
    """Wall finishing in steps of 10 ms; ``step_hook(k)`` -> (moving, omega, in_cut, recut rows) per step.
    Block 0 is the cutting move, block 1 the dwell (G04)."""
    edge, outline = fx.endmill(R, 4)
    ctx = fx.context(fx.project(physics=physics, tolerances=tolerances), tl=fx.timeline(kinds=kinds, lines=[20, 21]))
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    period = 2.0 * math.pi / (4 * abs(OMEGA))
    v = FZ / period
    x = 0.0
    q = 0
    step = 10 * fx.IPO
    for k in range(n_steps):
        moving, omega, in_cut, recut, block = step_hook(k)
        tk = fx.step_ticks(k * step, step)
        rows = []
        tips = []
        for t in tk:
            if moving:
                x += v * fx.IPO * fx.TICK_S
            tips.append((x, 0.0, 0.0))
        if moving:
            t_next = k * step
            while True:
                t_ev = 1000 + int(round(q * period / fx.TICK_S))
                if t_ev > (k + 1) * step:
                    break
                if t_ev > k * step:
                    x_ev = np.interp(t_ev, tk, np.array(tips)[:, 0])
                    rows.append(fx.event_row(t_ev, q % 4, 12, (x_ev, R, 0.01), (0.0, -1.0, 0.0),
                                             u_rel=(0.0, -u_n, 0.0), edge_r=0.0))
                q += 1
            del t_next
        feed = np.tile([v if moving else 0.0, 0.0, 0.0], (len(tk), 1))
        fr = fx.frame(tk, np.array(tips), omega, feed=feed, blocks=np.full(len(tk), block))
        c = fx.contact(np.array(rows).reshape(-1, 16), in_cut_s=0.01 if in_cut else 0.0, recut=recut)
        dom.macro_step(ctx, state, fx.bus([fr], [c], removal=[fx.removal_contact()]), fx.bus([fr], []))
    dom.end_operation(ctx, state)
    return dom, ctx, state


def test_dwell_in_the_cut_leaves_a_mark_as_deep_as_the_deflection():
    hook = lambda k: (k not in (4, 5, 6), OMEGA, True, None, 1 if k in (4, 5, 6) else 0)  # noqa: E731
    _, _, state = _wall_run(hook)
    marks = [m for m in state.surface.marks if m["kind"] == "dwell"]
    assert len(marks) == 1
    assert marks[0]["height_m"] == pytest.approx(3e-6, rel=1e-9)
    assert marks[0]["duration_s"] == pytest.approx(0.03, rel=0.05)
    assert marks[0]["revolutions"] > 1.0
    found = [f for f in state.findings if f.category == Category.FINISH_OUT_OF_SPEC]
    assert len(found) == 1 and found[0].line == 21                 # the G04 block
    assert state.surface.store.pf[marks[0]["patch"], PF["mark_max"]] == pytest.approx(3e-6)


def test_the_mark_limit_follows_the_faces_rz_specification():
    """A 3 um dwell mark on a wall specified Rz 5 um stays below the face's own limit: no finding."""
    hook = lambda k: (k not in (4, 5, 6), OMEGA, True, None, 1 if k in (4, 5, 6) else 0)  # noqa: E731
    tol = [{"id": "wall", "kind": "roughness", "Rz_um": 5.0,
            "feature": {"face": "plane", "normal": [0, -1, 0], "near_mm": [1.0, 5.0, 10.0]}}]
    _, _, state = _wall_run(hook, tolerances=tol)
    assert [m["kind"] for m in state.surface.marks] == ["dwell"]
    assert not [f for f in state.findings if f.category == Category.FINISH_OUT_OF_SPEC and "mark" in f.message]
    assert state.surface.store.pf[state.surface.marks[0]["patch"], PF["spec_rz"]] == pytest.approx(5e-6)


def test_a_shallow_dwell_is_recorded_without_a_finding():
    hook = lambda k: (k not in (4, 5, 6), OMEGA, True, None, 1 if k in (4, 5, 6) else 0)  # noqa: E731
    _, _, state = _wall_run(hook, u_n=1e-6)
    assert [m["kind"] for m in state.surface.marks] == ["dwell"]
    assert not [f for f in state.findings if f.category == Category.FINISH_OUT_OF_SPEC]


def test_feed_stop_outside_a_dwell_block_and_no_mark_out_of_the_cut():
    hook = lambda k: (k not in (4, 5, 6), OMEGA, True, None, 0)  # noqa: E731
    _, _, state = _wall_run(hook)
    assert [m["kind"] for m in state.surface.marks] == ["feed_stop"]
    hook = lambda k: (k not in (4, 5, 6), OMEGA, k not in (4, 5, 6), None, 0)  # noqa: E731
    _, _, state = _wall_run(hook)
    assert state.surface.marks == []


def test_spindle_stopping_in_the_cut():
    hook = lambda k: (k < 5, OMEGA if k < 5 else 0.0, True, None, 0)  # noqa: E731
    _, _, state = _wall_run(hook)
    marks = [m for m in state.surface.marks if m["kind"] == "spindle_stop_in_cut"]
    assert len(marks) == 1
    assert [f for f in state.findings if "spindle stopped" in f.message]


def test_servo_mark_at_an_axis_reversal():
    """X reverses in the middle of the step while the tool finishes a wall facing -x; a following-error spike of
    4 um on X at the reversal becomes a 4 um mark normal to the wall."""
    edge, outline = fx.endmill(R, 4)
    ctx = fx.context(fx.project())
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    step = 10 * fx.IPO
    tk = fx.step_ticks(0, step)
    cmd = np.zeros((len(tk), 3))
    cmd[:, 0] = -np.abs(np.arange(len(tk)) - 4.5) * 1e-5        # X goes up then down (reversal near tick 5)
    cmd[:, 1] = np.arange(len(tk)) * 1e-5
    ferr = np.zeros((len(tk), 3))
    ferr[5, 0] = 4e-6
    tips = cmd.copy()
    rows = [fx.event_row(int(t), 0, 12, (R, 0.0, 0.01), (-1.0, 0.0, 0.0), edge_r=0.0) for t in tk]
    fr = fx.frame(tk, tips, OMEGA, axis_cmd=cmd, ferr=ferr)
    dom.macro_step(ctx, state, fx.bus([fr], [fx.contact(np.array(rows))]), fx.bus([fr], []))
    marks = [m for m in state.surface.marks if m["kind"] == "servo_reversal"]
    assert len(marks) == 1
    assert marks[0]["axis"] == "X" and marks[0]["height_m"] == pytest.approx(4e-6, rel=1e-9)


def test_recut_chips_dent_the_finished_surface():
    def hook(k):
        recut = [[1000 + k * 10000 + 1500, 1, 12, 1e-7, 0.0, 0.0]] if k in (3, 6, 9) else None
        return True, OMEGA, True, recut, 0
    _, _, dented = _wall_run(hook, u_n=0.0)
    _, _, clean = _wall_run(lambda k: (True, OMEGA, True, None, 0), u_n=0.0)
    pf = dented.surface.store.pf
    assert pf[:, PF["recut"]].sum() == 3
    depth = 5e-5 * (1.0 - 1.0 / 1.6)                                # h_max 50 um, chip ratio unknown -> t = h
    assert pf[:, PF["recut_max"]].max() == pytest.approx(depth, rel=1e-9)
    assert fx.patch_values(dented)["Rt_m"] > fx.patch_values(clean)["Rt_m"] + 0.5 * depth


def test_chatter_marks_flag_the_patches_generated_meanwhile():
    edge, outline = fx.endmill(R, 4)
    ctx = fx.context(fx.project())
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    ev, v = fx.side_wall(R, 4, FZ, OMEGA, 20, [12], edge, edge_r=0.0)
    for t0, rows in fx.chunks(ev, 10 * fx.IPO):
        tk = fx.step_ticks(t0, 10 * fx.IPO)
        tip = np.stack([v * (tk - 1000) * fx.TICK_S, 0 * tk, 0 * tk], axis=1)
        fr = fx.frame(tk, tip, OMEGA)
        b = fx.bus([fr], [fx.contact(rows)], dynamics={"chatter_index": 0.8, "chatter_freq_Hz": 1234.0})
        dom.macro_step(ctx, state, b, fx.bus([fr], []))
    pf = state.surface.store.pf
    assert pf[0, PF["chatter"]] == 1.0 and pf[0, PF["chatter_freq"]] == 1234.0
