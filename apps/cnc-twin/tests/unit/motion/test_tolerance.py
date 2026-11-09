"""Tolerance-driven feed limits of the planner (docs/models/motion.md 2.1-2.2): the chord error of the linearised
path at the interpolation period, the tolerance band of an axis setpoint filter, and the bound at junctions where
the path curvature changes."""

import copy
import math

import numpy as np
import pytest

from mfix import MILL, Script, run, stack
from cnctwin.motion.geometry import chord_speed, filter_band_speed

# the generator's last jerk phase toward a speed limit ends within one tick: the speed may pass the limit by up to
# J dt^2 / 8 (MILL: J 200 m/s^3, dt 1 ms)
RAMP = 200.0 * 1e-3 ** 2 / 8.0


def _arc_program(radius_mm: float, feed: float, tol_mm: float) -> Script:
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    s.line(feed, tol_mm=tol_mm, X=-100 + radius_mm, Y=-100, Z=-100)
    s.arc(feed, centre_mm=(-100, -100, -100), sweep_rad=2 * math.pi, radius_mm=radius_mm, tol_mm=tol_mm,
          X=-100 + radius_mm, Y=-100, Z=-100)
    s.stop()
    return s


def _arc_speeds(res, block: int = 2):
    cmd = stack(res.frames, "axis_cmd")
    blk = stack(res.frames, "block_index")
    v = np.r_[0.0, np.linalg.norm(np.diff(cmd[:, :2], axis=0), axis=1) / 1e-3]
    return v[blk == block], cmd[blk == block, :2]


def test_chord_formula_is_the_sagitta_of_one_interpolation_period():
    R, e, T = 2e-3, 0.5e-6, 1e-3
    v = chord_speed(R, e, T)
    half = 0.5 * v * T
    assert R - math.sqrt(R * R - half * half) == pytest.approx(e, rel=1e-12)
    assert v == pytest.approx(math.sqrt(8 * R * e) / T, rel=1e-3)          # the small-angle form
    assert chord_speed(math.inf, e, T) == math.inf and chord_speed(R, 0.0, T) == math.inf
    assert chord_speed(R, 10 * R, T) == pytest.approx(2 * R / T)          # never a chord longer than the diameter
    assert filter_band_speed(5e-3, 10e-6, 0.02) == pytest.approx(math.sqrt(24 * 5e-3 * 10e-6) / 0.02)
    assert filter_band_speed(5e-3, 10e-6, 0.0) == math.inf


def test_chord_error_at_the_ipo_period_limits_a_small_arc_with_a_tight_tolerance():
    # R 1 mm, tolerance 0.2 um, IPO 1 ms: the chord limit 2 sqrt(2 R e - e^2) / T = 40 mm/s lies below the
    # centripetal (sqrt(A R) = 70.7 mm/s) and circular-jerk ((J R^2)^(1/3) = 58.5 mm/s) bounds
    R, tol = 1e-3, 0.2e-6
    res, dom, state, ctrl = run(MILL, _arc_program(1.0, 6000.0, 0.0002))
    v, pts = _arc_speeds(res)
    bound = chord_speed(R, tol, 1e-3)
    assert bound < min(math.sqrt(5.0 * R), (200.0 * R * R) ** (1 / 3))
    assert v.max() <= bound + RAMP
    assert v.max() >= bound * (1 - 5e-3)
    # every chord between two commanded points stays within the tolerance of the circle
    mid = 0.5 * (pts[1:] + pts[:-1])
    sag = R - np.hypot(mid[:, 0] + 0.1, mid[:, 1] + 0.1)
    assert sag.max() <= tol * (1 + 1e-3)
    # with a looser tolerance the chord is no limit: the circle runs at the centripetal / jerk clamp as before
    res2, _, _, _ = run(MILL, _arc_program(1.0, 6000.0, 0.01))
    v2, _ = _arc_speeds(res2)
    assert v2.max() == pytest.approx(min(math.sqrt(5.0 * R), (200.0 * R * R) ** (1 / 3)), rel=5e-3)


def test_setpoint_filter_band_slows_arcs_to_keep_the_radius_error_in_the_tolerance():
    # a 20 ms moving-average setpoint filter pulls a circle in by v^2 T^2 / (24 R): within 10 um on R 5 mm the
    # control holds v <= sqrt(24 R e) / T = 54.8 mm/s; without a filter the arc runs at the 100 mm/s feed
    m = copy.deepcopy(MILL)
    m["controller"]["setpoint_filter_ms"] = 20.0
    R, tol, Tf = 5e-3, 10e-6, 0.02
    res, dom, state, ctrl = run(m, _arc_program(5.0, 6000.0, 0.01))
    assert dom.model.setpoint_filter_s == pytest.approx(Tf)
    v, _ = _arc_speeds(res)
    bound = math.sqrt(24 * R * tol) / Tf
    assert v.max() <= bound + RAMP and v.max() >= bound * (1 - 5e-3)
    assert v.max() ** 2 * Tf ** 2 / (24 * R) <= tol * (1 + 2e-3)
    res0, dom0, _, _ = run(MILL, _arc_program(5.0, 6000.0, 0.01))
    assert dom0.model.setpoint_filter_s == 0.0
    v0, _ = _arc_speeds(res0)
    assert v0.max() == pytest.approx(0.1, rel=5e-3)
    assert res.end_tick > res0.end_tick


def test_curvature_reversal_between_tangent_arcs_bounds_the_junction_speed():
    # two tangent quarter circles of R 20 mm turning opposite ways (an S): at the junction the X acceleration
    # steps from -v^2/R to +v^2/R, so v <= sqrt(A R / 2) = 224 mm/s there, while each arc allows sqrt(A R) = 316 mm/s
    R, A = 20.0, 5.0
    s = Script()
    s.rapid(X=-200, Y=-200, Z=-100)
    s.arc(20000, centre_mm=(-200, -200 + R, -100), sweep_rad=math.pi / 2, radius_mm=R, tol_mm=0.01,
          X=-200 + R, Y=-200 + R, Z=-100)
    s.arc(20000, centre_mm=(-200 + 2 * R, -200 + R, -100), sweep_rad=-math.pi / 2, radius_mm=R, tol_mm=0.01,
          X=-200 + 2 * R, Y=-200 + 2 * R, Z=-100)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    cmd = stack(res.frames, "axis_cmd")
    blk = stack(res.frames, "block_index")
    v = np.r_[0.0, np.linalg.norm(np.diff(cmd[:, :2], axis=0), axis=1) / 1e-3]
    first = np.nonzero(blk == 2)[0][0]                 # the first tick of the second arc straddles the junction
    junction = math.sqrt(A * R * 1e-3 / 2.0)
    # the profile's slowest point is the junction: the tick that contains it moves at most at the bound
    assert min(v[first - 1], v[first]) <= junction + RAMP
    on_arcs = v[np.isin(blk, [1, 2])]
    assert on_arcs.max() > 1.2 * junction               # the arcs themselves run faster than the junction
    assert on_arcs.max() <= math.sqrt(A * R * 1e-3) + RAMP
    # the commanded path stays on the circles
    c1 = np.hypot(cmd[blk == 1, 0] + 0.2, cmd[blk == 1, 1] + 0.2 - R * 1e-3)
    assert np.max(np.abs(c1 - R * 1e-3)) < 1e-9
