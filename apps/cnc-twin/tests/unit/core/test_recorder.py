"""Recorder (cnctwin.core.recorder): ring kernels, tiers, windows, Parquet parts, readers, export."""

import json
import math

import numpy as np
import pyarrow.parquet as pq
import pytest

from cnctwin.core.abi import ring_buffer
from cnctwin.core.channels import Channel, ChannelRegistry, Tier
from cnctwin.core.findings import Category, Finding, FindingList, Severity
from cnctwin.core import recorder as R

IPO = 1000
MACRO_IPO = 10
MACRO = IPO * MACRO_IPO
L0 = 100
OMEGA = -1047.1975511965977          # 10 000 rpm: one revolution 6 ms
SETTINGS = {"fast_min_spacing_s": 0.05, "fast_max_windows_per_second": 3, "fast_window_min_s": 0.002,
            "fast_window_max_s": 0.1, "series_buffer_max_bytes": 4096}


def registry():
    reg = ChannelRegistry()
    reg.register_all([
        Channel("c.force_N", "N", Tier.FAST, "cutting", "force", (3,)),
        Channel("c.h_m", "m", Tier.FAST, "cutting", "chip thickness"),
        Channel("m.feed_m_s", "m/s", Tier.TICK, "motion", "feed"),
        Channel("m.err_m", "m", Tier.TICK, "motion", "following error", (4,)),
        Channel("m.mode", "-", Tier.TICK, "motion", "mode", (), "int32"),
        Channel("r.volume_m3", "m3", Tier.MACRO, "removal", "volume"),
        Channel("r.flags", "-", Tier.MACRO, "removal", "flags", (3,), "bool"),
        Channel("r.count", "-", Tier.MACRO, "removal", "count", (2,), "int64"),
    ])
    return reg


def make(tmp_path, fast="windows", **settings):
    recording = {"fast": fast, "fast_window_revs": 2, "fast_period_s": 0.5, "keyframe_period_s": 0.5}
    return R.Recorder(tmp_path, registry(), l0_ticks_min=L0, ipo_ticks=IPO, macro_ipo=MACRO_IPO, recording=recording,
                      settings={**SETTINGS, **settings})


def value(tick, j):
    return math.sin(1e-4 * tick + j) * 100.0


def run_steps(rec, steps, contact, findings=None):
    """Push ring rows for every L0 step of the steps in contact, then close each step."""
    f0, _ = rec.fast_columns("c.force_N")
    h0, _ = rec.fast_columns("c.h_m")
    for n in steps:
        tick0 = n * MACRO
        if contact(n):
            for tick in range(tick0, tick0 + MACRO, L0):
                slot = R.ring_begin(rec.ring, tick)
                for j in range(3):
                    rec.ring.data[slot, f0 + j] = value(tick, j)
                rec.ring.data[slot, h0] = 1e-5 * (tick % 7)
        ticks = tick0 + IPO * np.arange(MACRO_IPO)
        rec.record_tick(ticks, {"m.feed_m_s": 0.01 * np.ones(MACRO_IPO), "m.err_m": np.ones((MACRO_IPO, 2)) * n,
                                "m.mode": np.full(MACRO_IPO, n % 3)})
        rec.record_macro(tick0 + MACRO, {"r.volume_m3": 1e-9 * n, "r.flags": np.array([True, n % 2 == 0]),
                                         "r.count": np.array([n])})
        rec.end_step(tick0, tick0 + MACRO, OMEGA, findings)


# ----------------------------------------------------------------------------- kernels

def test_kernels_have_one_signature():
    for kernel in (R.ring_begin, R.ring_put, R.tick_envelope, R.ring_envelope):
        assert len(kernel.signatures) == 1


def test_ring_envelope_matches_reference_across_the_wrap():
    rng = np.random.default_rng(11)
    ring = ring_buffer(64, 3)
    for k in range(150):                                  # wraps the 64-row ring twice
        values = rng.normal(size=3)
        values[rng.random(3) < 0.25] = np.nan
        R.ring_put(ring, 1000 + 37 * k, values)
    head = int(ring.head[0])
    out, ref = np.empty((4, 9)), np.empty((4, 9))
    R.ring_envelope(ring, head - 60, head, 1000 + 37 * 90, 500, out)
    R.ring_envelope_reference(ring, head - 60, head, 1000 + 37 * 90, 500, ref)
    assert np.array_equal(out, ref, equal_nan=True)
    assert np.isfinite(out[0, 1])


def test_ring_kernels_match_reference():
    a, b = ring_buffer(5, 3), ring_buffer(5, 3)
    rng = np.random.default_rng(3)
    for k in range(13):
        values = rng.normal(size=3)
        if k % 2:
            assert R.ring_put(a, 100 * k, values) == R.ring_put_reference(b, 100 * k, values)
        else:
            sa, sb = R.ring_begin(a, 100 * k), R.ring_begin_reference(b, 100 * k)
            assert sa == sb
            a.data[sa, 1] = values[1]
            b.data[sb, 1] = values[1]
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)
    assert int(a.head[0]) == 13


def test_tick_envelope_matches_reference():
    rng = np.random.default_rng(5)
    ticks = np.sort(rng.choice(np.arange(0, 12000, 50), size=150, replace=False)).astype(np.int64)
    rows = rng.normal(size=(150, 4))
    rows[rng.random((150, 4)) < 0.2] = np.nan
    rows[3, :] = np.inf
    out = [np.empty((11, 4)) for _ in range(3)]
    ref = [np.empty((11, 4)) for _ in range(3)]
    R.tick_envelope(ticks, rows, 500, 1000, *out)
    R.tick_envelope_reference(ticks, rows, 500, 1000, *ref)
    for x, y in zip(out, ref):
        assert np.array_equal(x, y, equal_nan=True)
    empty = [np.empty((2, 4)) for _ in range(3)]
    R.tick_envelope(ticks, rows, 10 ** 6, 1000, *empty)
    assert np.all(np.isnan(empty[0]))


# ----------------------------------------------------------------------------- layout and policy

def test_layout_ring_size_and_policy(tmp_path):
    rec = make(tmp_path)
    assert rec.fast_columns("c.force_N") == (0, 3) and rec.fast_columns("c.h_m") == (3, 1)
    assert rec.fast_column_names() == ["c.force_N[0]", "c.force_N[1]", "c.force_N[2]", "c.h_m"]
    assert len(rec.ring.tick) == 5000                      # 0.5 s at 100 us
    assert rec.window_ticks(OMEGA) == 12000                # two revolutions at 10 000 rpm
    assert rec.window_ticks(0.0) == 2000                   # stopped: the minimum window
    assert rec.window_ticks(1.0) == 100000                 # slow: capped
    small = make(tmp_path / "small", ring_max_bytes=40 * 1200)
    assert len(small.ring.tick) == 1200                    # shortened by the byte budget
    with pytest.raises(R.InputError):
        make(tmp_path / "tiny", ring_max_bytes=40 * 50)    # cannot hold one macro step
    doc = rec.channels_document()
    assert doc["format"] == "cnctwin-channels" and doc["fast"]["ring_rows"] == 5000
    names = {c["name"]: c for c in doc["channels"]}
    assert names["r.flags"]["storage"] == "int8" and names["m.mode"]["storage"] == "int32"


def test_fast_all_size_estimate(tmp_path):
    # 10 000 rpm, 8 us stride: 125 000 rows/s of 40 float32 columns + tick and time
    assert R.Recorder.fast_all_bytes_per_second(40, 8) == pytest.approx(125_000 * (160 + 16))
    full = make(tmp_path, fast="all")
    run_steps(full, range(0, 2), lambda n: True)
    full.close_parts()
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(tmp_path / "series" / "fast" / "part-0.parquet").metadata
    raw = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
    assert raw <= R.Recorder.fast_all_bytes_per_second(4, L0) * 0.02 * 1.5


def test_template_defaults_when_project_has_no_recording(tmp_path):
    policy = R.RecordingPolicy.load(None)
    template = R.template_recording()
    assert policy.fast == template["fast"] and policy.window_revs == template["fast_window_revs"]
    with pytest.raises(R.InputError):
        R.RecordingPolicy.load({"fast": "sometimes"})


def test_block_of_pads_and_checks_shapes():
    reg = registry()
    err = reg.get("m.err_m")
    block = R.block_of(err, np.ones((5, 2)), 5, per_row=True)
    assert block.shape == (5, 4) and np.all(block[:, :2] == 1) and np.all(np.isnan(block[:, 2:]))
    assert np.all(R.block_of(reg.get("m.feed_m_s"), 2.0, 3, per_row=True) == 2.0)
    with pytest.raises(R.RecorderError):
        R.block_of(err, np.ones((5, 5)), 5, per_row=True)
    with pytest.raises(R.RecorderError):
        R.block_of(err, np.ones((4, 2)), 5, per_row=True)


# ----------------------------------------------------------------------------- a recorded run

def test_tiers_windows_and_parts(tmp_path):
    rec = make(tmp_path)
    findings = FindingList()
    contact = lambda n: 2 <= n < 30 or 40 <= n < 45      # noqa: E731  two engagements
    run_steps(rec, range(0, 20), contact, findings)
    findings.add(Finding(Category.CHATTER, Severity.WARNING, "x", 0.25, 250_000, "cutting"))
    run_steps(rec, range(20, 60), contact, findings)
    entries = rec.close_parts()
    assert {e["tier"] for e in entries} == {"fast", "tick", "macro"}
    index = [json.loads(line) for line in (tmp_path / "series" / "index.jsonl").read_text().splitlines()]
    assert [e["tier"] for e in index] == ["fast", "tick", "macro"]
    tick = pq.read_table(tmp_path / index[1]["file"])
    assert tick.num_rows == 600 and tick.column("tick").to_pylist()[:3] == [0, 1000, 2000]
    assert tick.schema.field("m.mode").type == "int32"
    err = tick.column("m.err_m[1]").to_numpy()
    assert np.all(err == np.repeat(np.arange(60), 10))
    assert np.all(np.isnan(tick.column("m.err_m[3]").to_numpy()))
    # envelopes are min/mean/max of the ring rows of each L1 tick
    t = 25 * MACRO + 3 * IPO
    row = tick.column("tick").to_pylist().index(t)
    samples = np.array([value(x, 1) for x in range(t, t + IPO, L0)])
    assert tick.column("c.force_N[1].min")[row].as_py() == samples.min()
    assert tick.column("c.force_N[1].max")[row].as_py() == samples.max()
    assert tick.column("c.force_N[1].mean")[row].as_py() == pytest.approx(samples.mean(), rel=1e-15)
    out = tick.column("tick").to_pylist().index(35 * MACRO)
    assert tick.column("c.force_N[0].mean")[out].as_py() is not None
    assert math.isnan(tick.column("c.force_N[0].mean")[out].as_py())
    # fast windows: engagement starts at 20 000 and 400 000, the finding at 250 000, periodic at 0 and 500 000
    fast = pq.read_table(tmp_path / index[0]["file"]).column("tick").to_numpy()
    assert np.all(np.diff(fast) > 0)
    assert fast[0] == 20_000 and 20_000 + 12_000 - L0 in fast and 20_000 + 12_000 not in fast
    assert 250_000 - 6000 in fast and 250_000 + 6000 - L0 in fast and 250_000 + 6000 not in fast
    assert 400_000 in fast and 500_000 not in fast        # periodic window at 0.5 s: no contact then
    assert rec.window_count["engagement"] == 2 and rec.window_count["finding"] == 1
    macro = pq.read_table(tmp_path / index[2]["file"])
    assert macro.schema.field("r.flags[0]").type == "int8"
    assert macro.column("r.flags[2]").null_count == 60 and macro.column("r.count[1]").null_count == 60
    assert macro.column("r.count[0]").to_pylist()[:3] == [0, 1, 2]


def test_engagement_caps(tmp_path):
    rec = make(tmp_path, fast_min_spacing_s=0.02, fast_max_windows_per_second=2)
    assert rec.engagement(0, OMEGA) and not rec.engagement(10_000, OMEGA)     # too close
    assert rec.engagement(30_000, OMEGA) and not rec.engagement(60_000, OMEGA)  # third in the second
    assert rec.engagement(1_000_000, OMEGA)                                   # a new second
    assert rec.window_count["skipped"] == 2


def test_all_and_none_modes(tmp_path):
    full = make(tmp_path / "all", fast="all")
    run_steps(full, range(0, 6), lambda n: n >= 1)
    full.close_parts()
    table = pq.read_table(tmp_path / "all" / "series" / "fast" / "part-0.parquet")
    assert table.num_rows == 5 * MACRO // L0 and table.schema.field("c.force_N[0]").type == "float"
    none = make(tmp_path / "none", fast="none")
    run_steps(none, range(0, 6), lambda n: n >= 1)
    none.close_parts()
    assert not (tmp_path / "none" / "series" / "fast").exists()
    assert (tmp_path / "none" / "series" / "tick" / "part-0.parquet").exists()


def test_contract_violations(tmp_path):
    rec = make(tmp_path)
    for tick in range(0, 2 * MACRO, L0):                  # rows of two steps before one end_step
        R.ring_begin(rec.ring, tick)
    with pytest.raises(R.RecorderError):
        rec.end_step(0, MACRO, OMEGA)
    rec2 = make(tmp_path / "b")
    rec2.record_tick(np.arange(MACRO_IPO) * IPO, {})
    with pytest.raises(R.RecorderError):
        rec2.record_tick(np.arange(MACRO_IPO) * IPO, {})
    with pytest.raises(R.RecorderError):
        rec2.record_macro(MACRO, {"m.feed_m_s": 1.0})      # a tick-tier channel
    with pytest.raises(R.RecorderError):
        rec2.record_macro(MACRO, {"nope": 1.0})
    small = make(tmp_path / "c", ring_max_bytes=40 * 110)
    assert len(small.ring.tick) == 110
    for tick in range(0, 2 * MACRO, L0 // 2):              # 400 rows into a 110-row ring
        R.ring_begin(small.ring, tick)
    with pytest.raises(R.RecorderError, match="overwritten"):
        small.end_step(0, 2 * MACRO, OMEGA)


def test_state_round_trip_continues_identically(tmp_path):
    contact = lambda n: n % 9 < 6                          # noqa: E731
    one = make(tmp_path / "one")
    run_steps(one, range(0, 30), contact)
    one.close_parts()
    two = make(tmp_path / "two")
    run_steps(two, range(0, 15), contact)
    cut = two.checkpoint_cut()
    saved = {k: (np.array(v) if isinstance(v, np.ndarray) else json.loads(json.dumps(v))) for k, v in two.checkpoint_state().items()}
    three = make(tmp_path / "two")
    three.restore_state(saved)
    three.truncate_outputs(cut)
    run_steps(three, range(15, 30), contact)
    three.close_parts()
    for tier in ("fast", "tick", "macro"):
        a = [pq.read_table(tmp_path / "one" / p["file"]) for p in R.series_parts(tmp_path / "one", tier)]
        b = [pq.read_table(tmp_path / "two" / p["file"]) for p in R.series_parts(tmp_path / "two", tier)]
        assert sum(t.num_rows for t in a) == sum(t.num_rows for t in b) > 0
        assert_same_rows(a, b)


def assert_same_rows(a, b):
    import pyarrow as pa
    left, right = pa.concat_tables(a), pa.concat_tables(b)
    assert left.schema.names == right.schema.names
    for name in left.schema.names:
        x, y = left.column(name), right.column(name)
        assert x.null_count == y.null_count, name
        xa = np.asarray(x.to_numpy(zero_copy_only=False), dtype=np.float64)
        ya = np.asarray(y.to_numpy(zero_copy_only=False), dtype=np.float64)
        assert np.array_equal(xa, ya, equal_nan=True), name


# ----------------------------------------------------------------------------- reading and export

def test_iter_series_read_series_and_m4(tmp_path):
    rec = make(tmp_path)
    run_steps(rec, range(0, 40), lambda n: n >= 1)
    rec.close_parts()
    run_steps(rec, range(40, 50), lambda n: n >= 1)
    rec.close_parts()
    chunks = list(R.iter_series(tmp_path, "tick", ["m.err_m[0]", "absent"], 100_000, 449_000))
    ticks = np.concatenate([c[0] for c in chunks])
    assert ticks[0] == 100_000 and ticks[-1] == 449_000 and np.all(np.diff(ticks) == IPO)
    assert np.all(np.isnan(np.concatenate([c[1]["absent"] for c in chunks])))
    assert R.select_columns(R.series_columns(tmp_path, "tick"), ["c.force_N[2]"]) == \
        ["c.force_N[2].min", "c.force_N[2].mean", "c.force_N[2].max"]
    raw = R.read_series(tmp_path, "tick", ["c.force_N[0].max"], max_points=10 ** 6)
    small = R.read_series(tmp_path, "tick", ["c.force_N[0].max"], max_points=40)
    assert small["downsampled"] and not raw["downsampled"]
    full = raw["columns"]["c.force_N[0].max"]["v"]
    down = small["columns"]["c.force_N[0].max"]["v"]
    assert np.nanmax(down) == np.nanmax(full) and np.nanmin(down) == np.nanmin(full)
    assert len(down) <= 40 and set(down[np.isfinite(down)]).issubset(set(full[np.isfinite(full)]))


def test_m4_buckets():
    acc = R.M4(0, 99, 4)
    ticks = np.arange(100)
    values = np.sin(ticks * 0.3)
    values[50:75] = np.nan
    acc.add(ticks[:60], values[:60])
    acc.add(ticks[60:], values[60:])
    t, v = acc.result()
    expected = {}
    for lo in (0, 25, 75):
        block = np.arange(lo, lo + 25)
        for tick in (block[0], block[-1], block[np.argmin(values[block])], block[np.argmax(values[block])]):
            expected[int(tick)] = values[tick]
    expected[50] = np.nan                                  # the all-NaN bucket: one NaN point
    assert t.tolist() == sorted(expected)
    assert np.array_equal(v, np.array([expected[k] for k in sorted(expected)]), equal_nan=True)


def test_export_csv_and_parquet(tmp_path, capsysbinary):
    rec = make(tmp_path)
    run_steps(rec, range(0, 5), lambda n: n >= 1)
    rec.close_parts()
    out = tmp_path / "tick.csv"
    rows = R.export(tmp_path, "tick", ["m.feed_m_s", "m.err_m[3]"], "csv", out, t0_s=0.01, t1_s=0.02)
    lines = out.read_text().splitlines()
    assert rows == 11 and lines[0] == '"tick","time_s","m.feed_m_s","m.err_m[3]"'
    assert lines[1].startswith("10000,0.01,0.01,") and lines[1].endswith(",")
    assert R.export_main([str(tmp_path), "--tier", "macro", "--format", "parquet", "--out", str(tmp_path / "m.parquet")]) == 0
    table = pq.read_table(tmp_path / "m.parquet")
    assert table.num_rows == 5 and table.column("r.flags[2]").null_count == 5
    assert R.export_main([str(tmp_path), "--tier", "tick", "--channels", "m.mode", "--t1", "0.002"]) == 0
    text = capsysbinary.readouterr().out.decode()
    assert text.splitlines()[0] == '"tick","time_s","m.mode"' and len(text.splitlines()) == 4


def test_replay_writes_only_its_window(tmp_path):
    rec = make(tmp_path)
    run_steps(rec, range(0, 10), lambda n: True)
    rec.close_parts()
    replay = R.Recorder(tmp_path, registry(), l0_ticks_min=L0, ipo_ticks=IPO, macro_ipo=MACRO_IPO,
                        recording={"fast": "windows"}, settings=SETTINGS, replay=(30_000, 52_000))
    run_steps(replay, range(0, 10), lambda n: True)
    entries = replay.close_parts()
    assert len(entries) == 1 and entries[0]["replay"] and entries[0]["file"] == "series/fast/replay-30000-52000.parquet"
    table = pq.read_table(tmp_path / entries[0]["file"])
    assert table.column("tick").to_pylist()[0] == 30_000 and table.column("tick").to_pylist()[-1] == 51_900
    merged = np.concatenate([c[0] for c in R.iter_series(tmp_path, "fast", ["c.h_m"])])
    assert np.all(np.diff(merged) > 0)
