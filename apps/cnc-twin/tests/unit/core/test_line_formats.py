"""Formats of the run directory's JSON-lines files: series/index.jsonl and keyframes/index.jsonl start with a header
line {format, version} (and are read the same without one, as written before); trace.jsonl keeps id = line number and
names its format in the sidecar trace.json."""

from __future__ import annotations

import json

import numpy as np

from cnctwin.core import keyframes as K
from cnctwin.core import recorder as R
from cnctwin.core.channels import Channel, ChannelRegistry, Tier
from cnctwin.core.rundir import is_jsonl_header, jsonl_header
from cnctwin.core.trace import TraceLog, read_sidecar, sidecar_path


def _recorder(tmp_path):
    reg = ChannelRegistry()
    reg.register(Channel("m.x_m", "m", Tier.MACRO, "motion", "x"))
    return R.Recorder(tmp_path, reg, l0_ticks_min=100, ipo_ticks=1000, macro_ipo=10, recording={"fast": "none"})


def test_the_series_index_has_a_header_line_and_a_resume_keeps_it(tmp_path):
    rec = _recorder(tmp_path)
    for n in range(2):
        rec.record_macro((n + 1) * 10_000, {"m.x_m": 0.5 * n})
        rec.end_step(n * 10_000, (n + 1) * 10_000)
    cut = rec.checkpoint_cut()
    index = tmp_path / "series" / "index.jsonl"
    lines = index.read_bytes().splitlines(keepends=True)
    assert lines[0] == jsonl_header("cnctwin-series-index") and json.loads(lines[0]) == {
        "format": "cnctwin-series-index", "version": 1}
    assert cut["index_bytes"] == index.stat().st_size and cut["index_lines"] == len(lines) - 1 >= 1
    state = rec.checkpoint_state()
    rec.record_macro(30_000, {"m.x_m": 1.0})
    rec.end_step(20_000, 30_000)
    rec.close_parts()
    assert len(index.read_bytes().splitlines()) > len(lines)
    rec.truncate_outputs(cut)                                    # a resume at the cut keeps the header
    assert index.read_bytes().splitlines(keepends=True) == lines
    parts = R.series_parts(tmp_path, "macro")
    assert [p["part"] for p in parts] == [0] and not any(is_jsonl_header(p) for p in parts)
    assert state["index_bytes"] == cut["index_bytes"]


def test_an_index_without_a_header_reads_the_same(tmp_path):
    rec = _recorder(tmp_path)
    rec.record_macro(10_000, {"m.x_m": 0.25})
    rec.end_step(0, 10_000)
    rec.close_parts()
    index = tmp_path / "series" / "index.jsonl"
    with_header = R.series_parts(tmp_path, "macro")
    index.write_bytes(b"".join(index.read_bytes().splitlines(keepends=True)[1:]))     # as written before the header
    assert R.series_parts(tmp_path, "macro") == with_header
    ticks, values = next(R.iter_series(tmp_path, "macro", ["m.x_m"]))
    assert list(ticks) == [10_000] and values["m.x_m"][0] == 0.25
    rec2 = _recorder(tmp_path)                                    # appending to a headerless index adds no header
    rec2.parts["macro"] = 1
    rec2.record_macro(20_000, {"m.x_m": 0.5})
    rec2.end_step(10_000, 20_000)
    rec2.close_parts()
    records = [json.loads(line) for line in index.read_text().splitlines()]
    assert len(records) > 2 and not any(is_jsonl_header(r) for r in records)


def test_the_keyframe_index_has_a_header_line_and_reads_without_one(tmp_path):
    kf = K.KeyframeWriter(tmp_path, period_ticks=10)
    kf.write(0, {}, fields={"x_K": np.linspace(290, 300, 7)})
    kf.write(10, {}, fields={"x_K": np.linspace(290, 300, 7)})
    index = tmp_path / "keyframes" / "index.jsonl"
    lines = index.read_bytes().splitlines(keepends=True)
    assert json.loads(lines[0]) == {"format": "cnctwin-keyframes-index", "version": 1} and len(lines) == 3
    assert kf.index_bytes == index.stat().st_size
    assert [e["k"] for e in K.KeyframeReader(tmp_path).entries()] == [0, 1]
    index.write_bytes(b"".join(lines[1:]))
    assert [e["k"] for e in K.KeyframeReader(tmp_path).entries()] == [0, 1]


def test_the_trace_names_its_format_in_a_sidecar_and_keeps_ids_as_line_numbers(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = TraceLog(path)
    first = trace.record(0.1, 100_000, "motion", "spindle_on", {"omega_rad_s": 10.0})
    second = trace.record(0.2, 200_000, "cutting", "force_peak", {"force_N": 5.0}, [first])
    cut = trace.byte_length()
    trace.record(0.3, 300_000, "cutting", "force_peak", {"force_N": 6.0}, [second])
    trace.close()
    assert sidecar_path(path) == tmp_path / "trace.json"
    header = read_sidecar(path)
    assert header["format"] == "cnctwin-trace" and header["version"] == 1 and header["file"] == "trace.jsonl"
    assert "implied" not in header and "id = line number" in header["lines"]
    lines = path.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == [0, 1, 2]       # no header line: id = line number
    resumed = TraceLog.resume(path, cut)
    assert resumed.record(0.4, 400_000, "removal", "x") == 2 and resumed.chain_ids(1) == [1, 0]
    resumed.close()
    sidecar_path(path).unlink()                                           # a trace written before the sidecar
    old = read_sidecar(path)
    assert old["implied"] and old["format"] == "cnctwin-trace" and old["version"] == 1
    sidecar_path(path).write_text(json.dumps({"format": "cnctwin-trace", "version": 9}))
    try:
        read_sidecar(path)
    except ValueError as error:
        assert "newer" in str(error)
    else:
        raise AssertionError("a newer trace format was read")
