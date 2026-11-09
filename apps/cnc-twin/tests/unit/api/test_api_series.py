"""Series queries (column selection, time windows, M4 downsampling) and exports (CSV, Parquet)."""

from __future__ import annotations

import io
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from api_support import DONE_RUN, PARTIAL_RUN

RUN = f"/api/runs/{DONE_RUN}"


def full_column(data_dir, tier, column):
    tables = [pq.read_table(p, columns=["tick", column]) for p in
              sorted((data_dir / "runs" / DONE_RUN / "series" / tier).glob("part-*.parquet"),
                     key=lambda p: int(p.stem.split("-")[1]))]
    table = pa.concat_tables(tables)
    return table.column("tick").to_numpy(), table.column(column).to_numpy()


def test_raw_when_small(client, data_dir):
    body = client.get(f"{RUN}/series", params={"tier": "macro", "channels": "thermal.edge_temp_K",
                                               "max_points": 5000}).json()
    assert body["method"] == "raw" and body["downsampled"] is False and body["rows"] == 232
    series = body["series"][0]
    ticks, values = full_column(data_dir, "macro", "thermal.edge_temp_K")
    assert series["n"] == 232 and series["unit"] == "K" and series["v"] == values.tolist()
    assert series["t_s"][:3] == [0.01, 0.02, 0.03] and series["t_s"][-1] == 2.32


def test_m4_keeps_the_extremes(client, data_dir):
    column = "cutting.force_N[0].max"
    body = client.get(f"{RUN}/series", params={"tier": "tick", "channels": column, "max_points": 200}).json()
    assert body["method"] == "m4" and body["buckets"] == 50 and body["rows"] == 2320
    series = body["series"][0]
    ticks, values = full_column(data_dir, "tick", column)
    assert series["n"] <= 200
    got = np.array([math.nan if v is None else v for v in series["v"]])
    assert np.nanmax(got) == np.nanmax(values) and np.nanmin(got) == np.nanmin(values)
    peak_time = ticks[np.nanargmax(values)] / 1e6
    assert series["t_s"][int(np.nanargmax(got))] == peak_time
    assert all(b >= a for a, b in zip(series["t_s"], series["t_s"][1:]))


def test_channel_expansion_and_exact_columns(client):
    body = client.get(f"{RUN}/series", params={"tier": "tick", "channels": "cutting.force_N,motion.feed_m_s"}).json()
    columns = [s["column"] for s in body["series"]]
    assert columns[:3] == ["cutting.force_N[0].min", "cutting.force_N[0].mean", "cutting.force_N[0].max"]
    assert len(columns) == 10 and columns[-1] == "motion.feed_m_s"
    assert {s["channel"] for s in body["series"]} == {"cutting.force_N", "motion.feed_m_s"}
    exact = client.get(f"{RUN}/series", params={"tier": "tick", "channels": "cutting.force_N[1].mean"}).json()
    assert [s["column"] for s in exact["series"]] == ["cutting.force_N[1].mean"]
    vector = client.get(f"{RUN}/series", params={"tier": "macro", "channels": "wear.VB_m"}).json()
    assert len(vector["series"]) == 16 and all(v is None for v in vector["series"][15]["v"])


def test_time_window(client):
    body = client.get(f"{RUN}/series", params={"tier": "tick", "channels": "motion.feed_m_s", "t0": 1.0,
                                               "t1": 1.0995, "max_points": 1000}).json()
    series = body["series"][0]
    assert body["rows"] == 100 and series["t_s"][0] == 1.0 and series["t_s"][-1] == 1.099
    fast = client.get(f"{RUN}/series", params={"tier": "fast", "channels": "cutting.torque_Nm", "t0": 0.642,
                                               "t1": 0.65}).json()
    assert fast["rows"] == 1001 and fast["series"][0]["unit"] == "Nm"
    assert client.get(f"{RUN}/series", params={"tier": "tick", "channels": "motion.feed_m_s", "t0": 2,
                                               "t1": 1}).status_code == 400


def test_errors(client):
    assert client.get(f"{RUN}/series", params={"tier": "tick"}).status_code == 400
    unknown = client.get(f"{RUN}/series", params={"tier": "tick", "channels": "cutting.forse_N"})
    assert unknown.status_code == 404 and "cutting.force_N" in unknown.json()["detail"]
    assert client.get(f"{RUN}/series", params={"tier": "slow", "channels": "x"}).status_code == 422
    assert client.get(f"{RUN}/series", params={"tier": "tick", "channels": "x", "max_points": 1}).status_code == 422
    assert client.get(f"/api/runs/{PARTIAL_RUN}/series", params={"tier": "fast", "channels": "cutting.torque_Nm"}
                      ).json()["rows"] > 0


def test_partial_run_series_and_missing_tier(client, data_dir):
    body = client.get(f"/api/runs/{PARTIAL_RUN}/series", params={"tier": "macro", "channels": "removal.removed_volume_m3"}).json()
    assert body["rows"] == 100                                 # one closed part (the first second)
    (data_dir / "runs" / PARTIAL_RUN / "series" / "index.jsonl").unlink()
    body = client.get(f"/api/runs/{PARTIAL_RUN}/series", params={"tier": "macro", "channels": "removal.removed_volume_m3"}).json()
    assert body["rows"] == 100                                 # parts found without the index
    import shutil
    shutil.rmtree(data_dir / "runs" / PARTIAL_RUN / "series" / "fast")
    assert client.get(f"/api/runs/{PARTIAL_RUN}/series", params={"tier": "fast", "channels": "x"}).status_code == 404


def test_export_csv(client, data_dir):
    response = client.get(f"{RUN}/export", params={"format": "csv", "tier": "macro",
                                                   "channels": "thermal.edge_temp_K,wear.VB_m"})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/csv")
    assert f'filename="{DONE_RUN}-macro.csv"' in response.headers["content-disposition"]
    lines = response.text.splitlines()
    header = lines[0].replace('"', "").split(",")
    assert header[:3] == ["tick", "time_s", "thermal.edge_temp_K"] and len(header) == 19 and len(lines) == 233
    first = lines[1].split(",")
    assert first[0] == "10000" and first[1] == "0.01" and first[-1] == ""      # wear.VB_m[15] is NaN: empty
    ticks, values = full_column(data_dir, "macro", "thermal.edge_temp_K")
    assert [float(line.split(",")[2]) for line in lines[1:]] == values.tolist()
    window = client.get(f"{RUN}/export", params={"tier": "tick", "channels": "motion.feed_m_s", "t0": 1, "t1": 1.009})
    assert len(window.text.splitlines()) == 11
    empty = client.get(f"{RUN}/export", params={"tier": "tick", "channels": "motion.feed_m_s", "t0": 50, "t1": 51})
    assert empty.text.strip() == '"tick","time_s","motion.feed_m_s"'


def test_export_parquet_everything(client, data_dir):
    response = client.get(f"{RUN}/export", params={"format": "parquet", "tier": "tick"})
    assert response.status_code == 200 and response.content[:4] == b"PAR1"
    table = pq.read_table(io.BytesIO(response.content))
    assert table.num_rows == 2320 and table.column_names[:2] == ["tick", "time_s"] and len(table.column_names) == 74
    ticks, values = full_column(data_dir, "tick", "cutting.spindle_power_W")
    np.testing.assert_array_equal(table.column("cutting.spindle_power_W").to_numpy(), values)
    assert table.column("cutting.chip_thickness_m[15].max").null_count == 2320
    assert not list((data_dir / "tmp").glob("export-*"))      # the temporary file is gone


def test_indexes_with_their_header_line_read_the_same(client, data_dir):
    """The core writes series/index.jsonl and keyframes/index.jsonl with a header line {format, version}; the
    synthetic runs have none (the older form): both read the same."""
    before_series = client.get(f"{RUN}/series", params={"tier": "macro", "channels": "thermal.edge_temp_K"}).json()
    before_keyframes = client.get(f"{RUN}/keyframes").json()
    run = data_dir / "runs" / DONE_RUN
    for relative, fmt in (("series/index.jsonl", "cnctwin-series-index"),
                          ("keyframes/index.jsonl", "cnctwin-keyframes-index")):
        path = run / relative
        path.write_bytes(('{"format": "%s", "version": 1}\n' % fmt).encode() + path.read_bytes())
    assert client.get(f"{RUN}/series", params={"tier": "macro", "channels": "thermal.edge_temp_K"}).json() \
        == before_series
    assert client.get(f"{RUN}/keyframes").json() == before_keyframes
