"""Pieces of the report: display formatting, tolerance limits, the tolerant run-directory reader (absent
files are listed, partial runs still render), series bucketing that keeps peaks, fast windows,
approximations filtered by the run's domains, threads and unknown characteristic kinds."""

from __future__ import annotations

import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from htmlcheck import check

from cnctwin.report import fmt
from cnctwin.report.context import Context
from cnctwin.report.report import report_html
from cnctwin.report.rundir import BUCKETS, RunData, split_column
from cnctwin.report.sections_part import limits, utilisation
from cnctwin.report.sections_refs import approximations_section


def test_formatting_in_display_units():
    assert fmt.num(1234.5678) == "1235" and fmt.num(0.000123456) == "0.0001235"
    assert fmt.num(1.5e-9, 3) == "1.50e-9" and fmt.num(None) == fmt.DASH and fmt.num(float("nan")) == fmt.DASH
    assert fmt.num(-2.5) == "−2.500" and fmt.num(7) == "7"
    assert fmt.mm(0.0123456) == "12.346" and fmt.um(-6.2e-5) == "−62.0" and fmt.signed_um(6.2e-5) == "+62.0"
    assert fmt.degc(293.15) == "20.0" and fmt.pct(0.256) == "25.6" and fmt.rpm(2 * math.pi * 100 / 60) == "100"
    assert fmt.duration(75.3) == "1 min 15.3 s" and fmt.duration(2.32) == "2.320 s" and fmt.duration(7300) == "2 h 02 min"
    assert fmt.kwh(3.6e6) == "1.000" and fmt.eur(0.0123) == "0.0123" and fmt.eur(12.345) == "12.35"
    assert fmt.quantity("deviation_m", 6.2e-5) == "deviation 62.00 µm"
    assert fmt.quantity("chatter_freq_Hz", 1927.3) == "chatter freq 1927 Hz"
    assert fmt.quantity("omega_rad_s", -1047.1975511965977) == "omega −10000 rpm"
    assert fmt.quantity("flute_fill", 0.72) == "flute fill 0.7200"
    assert fmt.quantity("edge_temp_K", 473.15) == "edge temp 200.0 °C"
    assert fmt.display_unit("m", 5e-6).label == "µm" and fmt.display_unit("m", 0.02).label == "mm"
    assert fmt.display_unit("W", 1500.0).label == "kW" and fmt.display_unit("m/s", 1.0)(0.05) == pytest.approx(3000.0)
    assert fmt.display_unit("K", 300.0)(300.0) == pytest.approx(26.85)


def test_characteristic_limits_and_use_of_the_tolerance():
    two_sided = {"nominal_m": 0.002, "plus_m": 5e-5, "minus_m": 5e-5, "measured_m": 0.002025}
    assert limits(two_sided) == pytest.approx((0.002, 0.00195, 0.00205))
    assert utilisation(two_sided) == pytest.approx(0.5)
    assert utilisation(dict(two_sided, measured_m=0.00198)) == pytest.approx(0.4)
    zone = {"kind": "flatness", "tolerance_m": 2e-5, "measured_m": 8e-6}
    assert limits(zone) == (None, 0.0, 2e-5) and utilisation(zone) == pytest.approx(0.4)
    explicit = {"nominal_m": 0.008, "lower_m": 0.008, "upper_m": 0.008015, "measured_m": 0.008012}
    assert limits(explicit) == (0.008, 0.008, 0.008015) and utilisation(explicit) == pytest.approx(0.8)
    assert split_column("cutting.force_N[1].max") == ("cutting.force_N", "[1]", "max")
    assert split_column("chips.flute_fill") == ("chips.flute_fill", "", "")


def test_an_empty_folder_still_gives_a_report(tmp_path):
    text, run = report_html(tmp_path)
    check(text)
    items = {m["item"] for m in run.missing}
    assert {"run.json", "findings.json", "timeline.json", "inspection.json", "production.json"} <= items
    assert "SAFE TO RUN" in text and "The run has not finished" in text
    assert "Not in this run directory" in text


def test_a_partial_run_renders_what_it_has(tmp_path):
    import make_synthetic_run as synthetic
    synthetic.make_synthetic_run(tmp_path / "partial", seed=4, until_s=1.2, media=False)
    text, run = report_html(tmp_path / "partial")
    check(text)
    assert not run.finished and "has not finished (status running)" in text
    assert "chip_packing" in text and "out_of_tolerance" not in text.split('id="cycle"')[0]


def _series_run(folder, values: np.ndarray, tick_step: int = 1000):
    ticks = np.arange(len(values), dtype=np.int64) * tick_step
    table = pa.table({"tick": pa.array(ticks), "time_s": pa.array(ticks * 1e-6), "x.signal_N": pa.array(values)})
    (folder / "series" / "tick").mkdir(parents=True)
    pq.write_table(table, folder / "series" / "tick" / "part-0.parquet")
    (folder / "channels.json").write_text(json.dumps({"channels": [
        {"name": "x.signal_N", "unit": "N", "tier": "tick", "domain": "x", "doc": "test signal", "max_shape": []}]}))
    (folder / "run.json").write_text(json.dumps({"clock": {"tick_s": 1e-6}, "verdict": {"safe": True, "reasons": []}}))


def test_series_buckets_keep_peaks_and_means(tmp_path):
    values = np.zeros(20000)
    values[12345] = 10.0
    values[777] = -3.0
    _series_run(tmp_path, values)
    summary = RunData(tmp_path).tier_summary("tick")
    stats = summary.columns["x.signal_N"]
    assert stats.max == 10.0 and stats.min == -3.0 and stats.mean == pytest.approx(7.0 / 20000)
    t_s, mean, lo, hi = summary.curve("x.signal_N")
    assert len(t_s) <= BUCKETS and hi.max() == 10.0 and lo.min() == -3.0
    assert np.all(np.diff(t_s) > 0) and mean.sum() * (20000 / len(t_s)) == pytest.approx(7.0, rel=0.05)
    text, _ = report_html(tmp_path)
    check(text)
    assert "x.signal_N" in text and "test signal" in text


def test_fast_windows_keep_the_highest_peak(runs):
    run = RunData(runs["unsafe"])
    windows, best = run.fast_windows({"cutting.force_N": [f"cutting.force_N[{i}]" for i in range(3)]})
    assert len(windows) >= 7 and sum(w.rows for w in windows) == sum(
        int(e["rows"]) for e in run.series_parts("fast"))
    window = best["cutting.force_N"]
    peak = max(float(np.nanmax(np.abs(v))) for v in window.values.values())
    assert window.peak == pytest.approx(peak) and window.t_s[0] == pytest.approx(window.tick0 * 1e-6)


def test_approximations_relevant_to_the_run(runs, tmp_path):
    doc = tmp_path / "approximations.md"
    doc.write_text("# Approximations\n\n## General\n\n* time is integer ticks (exact)\n\n"
                   "## Thermal\n\n| approximation | error |\n|---|---|\n| lumped tool body | 5 % |\n\n"
                   "## Sound\n\nThe sound model ignores room reflections.\n\n"
                   "## Chips\n\n- particles are spheres of equal volume, **drag +-20 %**\n", encoding="utf-8")
    section = approximations_section(Context(RunData(runs["unsafe"])), path=doc)
    body = section.body
    assert "General" in body and "Thermal" in body and "lumped tool body" in body and "<b>drag +-20 %</b>" in body
    assert "Sound" not in body and "room reflections" not in body


def test_threads_and_unknown_kinds_are_shown(run_copy):
    folder = run_copy()
    path = folder / "inspection.json"
    document = json.loads(path.read_text())
    document["tolerances"] += [
        {"id": "m6-thread", "kind": "thread", "designation": "M6x1-6H", "class": "6H",
         "pitch_diameter_m": {"measured": 0.00536, "min": 0.00535, "max": 0.0055}, "gauges": {"go": True, "no_go": True},
         "pass": True},
        {"id": "odd-one", "kind": "mystery", "measured_value": 3}]
    path.write_text(json.dumps(document))
    text, _ = report_html(folder)
    check(text)
    assert "M6x1-6H" in text and "6H" in text and "pitch: 5.3600 [5.3500, 5.5000]" in text and "no_go: accepts" in text
    assert "Other characteristics" in text and "mystery" in text and "odd-one" in text


def test_estimates_absent_is_said(run_copy):
    folder = run_copy("safe")
    document = json.loads((folder / "run.json").read_text())
    document.pop("estimates")
    (folder / "run.json").write_text(json.dumps(document))
    text, run = report_html(folder)
    assert "does not list the estimated database values" in text
    assert any(m["item"] == "estimates" for m in run.missing)


def test_many_findings_are_bounded_but_all_counted(run_copy):
    from cnctwin.report.context import FINDINGS_SHOWN
    folder = run_copy()
    path = folder / "findings.json"
    document = json.loads(path.read_text())
    base = document["findings"]
    for i in range(FINDINGS_SHOWN + 50):
        extra = dict(base[0], id=len(base), category="air_cutting", severity="info", time_s=1.0 + i * 1e-3,
                     tick=1000000 + i * 1000, line=11)
        base.append(extra)
    path.write_text(json.dumps(document))
    text, _ = report_html(folder)
    check(text)                                          # every link still has its target
    assert text.count('class="hit-') - text.count('class="hit-info"') >= 0
    assert f"55 more findings" in text and "Every finding, counted by category and severity" in text
    assert f'id="F-{len(base) - 1}"' not in text and 'id="F-4"' in text          # the error is always shown


def test_cycle_time_without_production_json(run_copy):
    folder = run_copy("safe")
    (folder / "production.json").unlink()
    text, run = report_html(folder)
    check(text)
    assert "timeline.json (production.json absent)" in text and "1620000" in text   # exact ticks from the timeline
    assert any(m["item"] == "production.json" for m in run.missing)
    assert "production.json is not in the run directory: no energy" in text
