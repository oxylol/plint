"""Alignment of measured data with a simulated run: markers, cross-correlation, overlays, metrics."""

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.mdata.align import (Alignment, align, align_markers, align_xcorr, error_metrics, masked_ncc, overlay,
                                 reduce_m4, robust_line)
from cnctwin.mdata.log import Channel, MachineLog
from cnctwin.mdata.simrun import Marker, SimulatedRun

import mdatakit

OFFSET = -1.234            # t_sim = OFFSET + t_meas: the log starts 1.234 s before the program


def _measured(run_dir, **kwargs):
    data = mdatakit.measured_spindle(run_dir, offset_s=OFFSET, eta=0.85, idle_W=612.0, kc_scale=1.12, **kwargs)
    log = MachineLog(epoch_ns=mdatakit.EPOCH_NS)
    log.add("spindle.power_W", data["t_meas"], data["power_W"])
    log.add("spindle.speed_rad_s", data["t_meas"], data["speed_rad_s"])
    log.add("program.line", data["t_meas"], data["line"], kind="event")
    return log, data


def test_simulated_run_as_log(synthetic_run):
    run = SimulatedRun(synthetic_run)
    log = run.to_log()
    for name in ("spindle.speed_rad_s", "spindle.shaft_power_W", "spindle.load_frac", "path.feed_m_s",
                 "axis.X.following_error_m", "axis.Z.pos_stock_m", "program.line", "program.block", "tool.number"):
        assert name in log, name
    speed = log.channels["spindle.speed_rad_s"]
    assert speed.values.max() == pytest.approx(abs(run.run["summary"]["spindle_speed_rad_s"]), rel=1e-4)
    markers = run.markers()
    assert markers[0].line == 2 and markers[-1].line == 22
    assert run.cycle_time_s() == pytest.approx(2.32)


def test_markers_recover_the_offset(synthetic_run):
    log, _ = _measured(synthetic_run)
    run = SimulatedRun(synthetic_run)
    found = align_markers(run.markers(), log)
    assert found.method == "markers" and found.quality["key"] == "line"
    assert found.quality["poll_interval_s"] == pytest.approx(0.01)
    assert found.offset_s == pytest.approx(OFFSET, abs=1.5e-3)       # polled at 10 ms: about dt / sqrt(12 n)
    assert found.scale == 1.0                                         # 2.3 s is too short to fit a scale


def test_markers_with_shifted_line_numbering(synthetic_run):
    log, data = _measured(synthetic_run)
    shifted = log.channels["program.line"]
    log.channels["program.line"] = type(shifted)(shifted.name, shifted.t_s, shifted.values + 1.0, "event")
    found = align_markers(SimulatedRun(synthetic_run).markers(), log)          # the control counts the % line
    assert found.quality["line_shift"] == 1 and found.quality["matched"] >= 15
    assert found.offset_s == pytest.approx(OFFSET, abs=1.5e-3)


def test_markers_recover_offset_and_scale_over_a_long_log():
    rng = np.random.default_rng(3)
    lines = np.tile(np.arange(10, 60), 12)                            # a loop: every line repeats 12 times
    durations = rng.uniform(0.2, 2.0, len(lines))
    starts = np.concatenate([[0.0], np.cumsum(durations[:-1])])
    markers = [Marker(float(t), 0, i, int(l), None, f"N{l}", "o.nc", "linear") for i, (t, l) in enumerate(zip(starts, lines))]
    scale, offset = 1.0 + 250e-6, 37.25                               # machine clock 250 ppm slow, log started earlier
    t_meas = (starts - offset) / scale
    poll = 0.05
    observed = np.ceil(t_meas / poll) * poll                          # polled at 20 Hz
    keep = rng.uniform(size=len(lines)) > 0.1                         # 10 % of the changes missed
    log = MachineLog()
    log.add("program.line", observed[keep], lines[keep].astype(float), kind="event")
    found = align_markers(markers, log, poll_interval_s=poll)
    assert found.quality["scale_fitted"]
    assert found.scale == pytest.approx(scale, abs=2e-6)
    assert found.offset_s == pytest.approx(offset, abs=0.01)
    assert found.quality["inliers"] >= 0.95 * keep.sum()


def test_xcorr_recovers_the_offset(synthetic_run):
    log, _ = _measured(synthetic_run)
    sim = SimulatedRun(synthetic_run).to_log()
    found = align_xcorr(sim.channels["spindle.shaft_power_W"], log.channels["spindle.power_W"])
    assert found.method == "xcorr" and not found.quality["ambiguous"]
    assert found.offset_s == pytest.approx(OFFSET, abs=3e-3)
    assert found.quality["peak"] > 0.9
    auto = align(sim, MachineLog(channels={"spindle.power_W": log.channels["spindle.power_W"]}), [])
    assert auto.method == "xcorr" and auto.offset_s == pytest.approx(OFFSET, abs=3e-3)


def test_xcorr_scale_and_ambiguity():
    t = np.arange(0.0, 600.0, 0.05)
    rng = np.random.default_rng(5)
    steps = np.repeat(rng.uniform(0, 1, 120), len(t) // 120 + 1)[: len(t)]
    sim = Channel("spindle.shaft_power_W", t, steps * 1000.0)
    scale, offset = 1.0 + 400e-6, 12.5
    t_meas = np.arange(-20.0, 600.0, 0.1)
    measured = Channel("spindle.power_W", t_meas, np.interp(offset + scale * t_meas, t, steps * 1000.0, left=0, right=0)
                       / 0.8 + 300 + rng.normal(0, 5, len(t_meas)))
    found = align_xcorr(sim, measured)
    assert found.quality["scale_fitted"] and found.scale == pytest.approx(scale, abs=5e-5)
    assert found.offset_s == pytest.approx(offset, abs=0.05)
    periodic = Channel("spindle.shaft_power_W", t, 500 + 400 * np.sign(np.sin(2 * np.pi * t / 10.0)))
    shifted = Channel("spindle.power_W", t, np.interp(t + 3.0, t, periodic.values))
    result = align_xcorr(periodic, shifted, scale_search=False)
    assert result.quality["ambiguous"]                                # every 10 s fits as well
    with pytest.raises(InputError, match="do not show the same process"):
        align_xcorr(sim, Channel("spindle.power_W", t_meas, rng.normal(0, 1, len(t_meas))))


def test_masked_ncc_matches_brute_force():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = rng.normal(size=45)
    x[[3, 17]] = np.nan
    y[5] = np.nan
    lags, c = masked_ncc(x, y, 10)
    for lag in (-30, -5, 0, 7, 40):
        k = int(np.where(lags == lag)[0][0])
        pairs = [(x[i], y[i - lag]) for i in range(len(x)) if 0 <= i - lag < len(y)
                 and np.isfinite(x[i]) and np.isfinite(y[i - lag])]
        a, b = np.array(pairs).T
        assert c[k] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-9)


def test_robust_line_rejects_mismatched_markers():
    x = np.linspace(0, 100, 50)
    y = 2.0 + 1.0001 * x + np.random.default_rng(2).normal(0, 0.001, 50)
    y[[7, 30]] += 5.0                                                  # two wrong matches
    fit = robust_line(x, y, fit_slope=True)
    assert fit["a"] == pytest.approx(2.0, abs=2e-3) and fit["b"] == pytest.approx(1.0001, abs=2e-5)
    assert not fit["inliers"][7] and not fit["inliers"][30] and fit["inliers"].sum() == 48


def test_error_metrics_and_m4():
    m = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    q = np.array([1.5, 2.0, 2.0, 9.0, 5.0])
    metrics = error_metrics(m, q)
    assert metrics["n"] == 4 and metrics["bias"] == pytest.approx(-0.125)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 1 + 0) / 4))
    assert metrics["max_abs"] == pytest.approx(1.0) and metrics["nrmse"] == pytest.approx(metrics["rmse"] / 4.0)
    t = np.arange(10000.0)
    v = np.sin(t / 50.0)
    v[5000] = 7.0
    t_r, cols, reduced = reduce_m4(t, {"y": v}, 400)
    assert reduced and len(t_r) <= 400 and cols["y"].max() == 7.0


def test_overlay_document(synthetic_run):
    log, _ = _measured(synthetic_run)
    run = SimulatedRun(synthetic_run)
    sim = run.to_log()
    alignment = Alignment(OFFSET, 1.0, "given")
    doc = overlay(sim, log, alignment, [{"name": "spindle speed", "sim": "spindle.speed_rad_s",
                                         "measured": "spindle.speed_rad_s", "unit": "rad/s"}],
                  sim_markers=run.markers(), simulated_cycle_s=run.cycle_time_s())
    series = doc["series"][0]
    assert doc["format"] == "cnctwin-overlay" and series["reduced"] == "none"
    assert len(series["t_s"]) == len(series["measured"]) == len(series["simulated"])
    metrics = doc["metrics"]["spindle speed"]["simulated"]
    assert metrics["n"] > 100 and metrics["r"] > 0.99
    assert doc["cycle_time"]["simulated_s"] == pytest.approx(2.32)


def test_program_start_alignment(synthetic_run):
    run = SimulatedRun(synthetic_run)
    log = MachineLog()
    poll = np.arange(0.0, 5.0, 0.02)
    state = np.where((poll >= 1.234) & (poll < 1.234 + 2.32), "ACTIVE", "READY").astype(object)
    log.add("program.execution", poll, state, kind="event")
    log.add("spindle.power_W", poll, np.zeros(len(poll)))
    found = align(run.to_log(), log, run.markers())
    assert found.method == "program_start"
    assert found.offset_s == pytest.approx(-1.234, abs=0.011)          # within half the 20 ms poll
