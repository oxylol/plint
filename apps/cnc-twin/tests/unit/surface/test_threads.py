"""Thread profile accuracy: the three-wire pitch diameter on ideal ISO 68-1 profiles against ISO 724
(d2 = d - 0.649519 P), pitch and flank-angle errors, and a threading pass through SurfaceDomain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.surface import surffix as fx
from cnctwin.surface import threads as TH
from cnctwin.surface.domain import SurfaceDomain
from cnctwin.surface.protocols import ToolOutline, turning_insert_outline


def _feature(pitch, alpha=math.radians(60.0), rotating=False):
    return TH.ThreadFeature(slot=0, op=0, pitch_m=pitch, lead_m=pitch, form_angle_rad=alpha, hand=1.0, starts=1,
                            axis_point=np.zeros(3), axis_dir=np.array([0.0, 0.0, 1.0]), rotating=rotating)


@pytest.mark.parametrize("d,P", [(6e-3, 1.0e-3), (10e-3, 1.5e-3), (20e-3, 2.5e-3)])
@pytest.mark.parametrize("external", [True, False])
def test_three_wire_pitch_diameter_of_an_ideal_profile_is_the_iso_value(d, P, external):
    X, N = TH.ideal_iso_profile(d, P, external=external)
    f = _feature(P)
    f.add(X, N, 200000)
    m = TH.measure(f)
    d2 = d - 0.649519 * P
    assert m["kind"] == ("external" if external else "internal")
    assert m["pitch_diameter_3wire_m"] == pytest.approx(d2, abs=5e-8)
    assert m["simple_pitch_diameter_m"] == pytest.approx(d2, abs=5e-7)
    assert m["wire_diameter_m"] == pytest.approx(P / (2.0 * math.cos(math.radians(30.0))), rel=1e-12)
    assert m["pitch_m"] == pytest.approx(P, rel=1e-6)
    assert m["flank_half_angles_rad"] == pytest.approx([math.radians(30.0)] * 2, abs=1e-6)
    assert m["major_diameter_m"] == pytest.approx(d, abs=1e-9)
    assert m["iso724_pitch_diameter_m"] == pytest.approx(d2, abs=1e-9)


def test_pitch_error_is_measured():
    X, N = TH.ideal_iso_profile(10e-3, 1.503e-3, external=True)
    f = _feature(1.5e-3)
    f.add(X, N, 200000)
    m = TH.measure(f)
    assert m["pitch_m"] == pytest.approx(1.503e-3, abs=2e-7)
    assert m["pitch_error_m"] == pytest.approx(3e-6, abs=2e-7)


def test_flank_angle_error_is_measured():
    X, N = TH.ideal_iso_profile(10e-3, 1.5e-3, external=True, alpha=math.radians(58.0))
    f = _feature(1.5e-3)
    f.add(X, N, 200000)
    m = TH.measure(f)
    assert math.degrees(m["included_angle_rad"]) == pytest.approx(58.0, abs=0.05)
    assert math.degrees(m["flank_angle_error_rad"]) == pytest.approx(-2.0, abs=0.05)


def test_tapped_or_milled_thread_axis_from_the_tool_positions():
    """Rotating threading tools: the thread axis passes through the mean tool position (the tap's axis, the
    centre of a thread mill's orbit), here 12 mm off the S origin."""
    X, N = TH.ideal_iso_profile(10e-3, 1.5e-3, external=False)
    offset = np.array([0.012, -0.004, 0.0])
    f = _feature(1.5e-3, rotating=True)
    f.add(X + offset, N, 200000)
    f.tip_sum = offset * 50.0
    f.tip_n = 50
    m = TH.measure(f)
    assert m["kind"] == "internal"
    assert m["pitch_diameter_3wire_m"] == pytest.approx(10e-3 - 0.649519 * 1.5e-3, abs=5e-8)


def test_thinning_keeps_the_point_budget():
    X, N = TH.ideal_iso_profile(10e-3, 1.5e-3, external=True, n_turns=12, per_turn=1440)
    f = _feature(1.5e-3)
    for k in range(0, len(X), 500):
        f.add(X[k:k + 500], N[k:k + 500], 4000)
    assert len(f.pts) <= 4000 and f.keep_every >= 4
    again = _feature(1.5e-3)
    for k in range(0, len(X), 500):
        again.add(X[k:k + 500], N[k:k + 500], 4000)
    assert np.array_equal(f.pts, again.pts)                      # deterministic (counter-based draws)
    m = TH.measure(f)
    assert m["pitch_diameter_3wire_m"] == pytest.approx(10e-3 - 0.649519 * 1.5e-3, abs=2e-6)


def test_turned_thread_through_the_domain():
    """A threading insert's events (the generated flank points) become a thread feature in surface.json."""
    from types import SimpleNamespace
    P, d = 1.5e-3, 10e-3
    edge, _ = fx.insert(0.1e-3)
    thread = SimpleNamespace(pitch_m=np.array([P]), lead_m=np.array([P]), starts=np.array([1]),
                             thread_hand=np.array([1.0]), form_angle_rad=np.array([math.radians(60.0)]))
    outline = ToolOutline(False, turning_insert_outline(0.1e-3, math.radians(60.0), math.radians(60.0)), 1, 0.1e-3,
                          0.0, "threading_insert", thread=thread)
    proj = fx.project(stock={"kind": "cylinder", "size_mm": [12, 40]})
    ctx = fx.context(proj)
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    X, N = TH.ideal_iso_profile(d, P, external=True, n_turns=6, per_turn=720)
    rows = np.array([fx.event_row(1000 + 10 * k, 0, 0, X[k], N[k]) for k in range(len(X))])
    for t0, chunk in fx.chunks(rows, 10 * fx.IPO):
        tk = fx.step_ticks(t0, 10 * fx.IPO)
        fr = fx.frame(tk, np.zeros((len(tk), 3)), -100.0, pose_W0_T=fx.turning_pose())
        dom.macro_step(ctx, state, fx.bus([fr], [fx.contact(chunk)]), fx.bus([fr], []))
    doc = dom.finalize(ctx, state)
    assert len(doc["threads"]) == 1
    t = doc["threads"][0]
    assert t["kind"] == "external" and t["id"] == "thread-T1-op0"
    assert t["pitch_diameter_3wire_m"] == pytest.approx(d - 0.649519 * P, abs=5e-8)
    assert t["line"] == 10
