"""Time series of a run: Parquet parts per tier (architecture 10), queried by channel and time.

Layout: ``series/<tier>/part-<n>.parquet`` (tiers ``fast``, ``tick``, ``macro``) with an int64
``tick`` column (base ticks; time = tick * tick_s, never accumulated) and one column per channel
column (``<name>``, ``<name>[i]``; the tick tier also ``<column>.min/.mean/.max`` of the fast
channels), listed in ``series/index.jsonl`` (``{tier, part, file, tick0, tick1, rows}`` per closed
part; without an index the part files are read in part-number order).

A ``channels=`` token selects a column exactly, or every column of a channel: ``cutting.force_N``
gives ``cutting.force_N[0]`` ... and, in the tick tier, their ``.min/.mean/.max``.

Reading goes through ``cnctwin.core.recorder.iter_series(run_dir, tier, columns, tick0, tick1)``
when the recorder provides it (yielding ``(ticks, {column: values})`` chunks in time order), else
through PyArrow directly; /api/health says which.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .downsample import M4
from .runfiles import RunDir
from .safety import ApiError, resolve_inside

TIERS = ("fast", "tick", "macro")
_PART = re.compile(r"^part-(\d+)\.parquet$")
_SUFFIX = re.compile(r"(\[\d+\])*(\.(min|mean|max))?$")
MAX_COLUMNS = 256

_backend_cache: dict = {}


def series_backend():
    """(name, iter_series callable or None): the recorder's reader when the module provides one."""
    if "value" in _backend_cache:
        return _backend_cache["value"]
    from .entry import describe
    value = ("pyarrow", None)
    if describe("cnctwin.core.recorder:iter_series")["module_found"]:
        try:
            from ..core import recorder
            function = getattr(recorder, "iter_series", None)
            if callable(function):
                value = ("cnctwin.core.recorder.iter_series", function)
        except Exception as error:   # a broken recorder must not take the API down; reported in health
            value = (f"pyarrow (cnctwin.core.recorder failed to import: {type(error).__name__}: {error})", None)
    _backend_cache["value"] = value
    return value


def ticks_to_seconds(ticks: np.ndarray, tick_s: float) -> np.ndarray:
    """tick * tick_s, computed as tick / (1 / tick_s) when that is an integer (exact decimal times)."""
    per_second = 1.0 / tick_s
    if abs(per_second - round(per_second)) < 1e-6:
        return np.asarray(ticks, np.float64) / float(round(per_second))
    return np.asarray(ticks, np.float64) * tick_s


def seconds_to_tick(seconds: float, tick_s: float) -> int:
    return int(round(seconds / tick_s))


class SeriesReader:
    def __init__(self, run: RunDir) -> None:
        self.run = run

    def parts(self, tier: str) -> list[dict]:
        if tier not in TIERS:
            raise ApiError(400, f"unknown tier {tier!r}: use {', '.join(TIERS)}")
        index = self.run.path / "series" / "index.jsonl"
        parts: list[dict] = []
        if index.is_file():
            from .jsonio import read_jsonl
            records, _ = read_jsonl(index)
            for record in records:
                if record.get("tier") == tier and isinstance(record.get("file"), str):
                    parts.append(record)
            parts.sort(key=lambda r: (int(r.get("part", 0)), r["file"]))
        else:
            folder = self.run.path / "series" / tier
            if folder.is_dir():
                for path in folder.iterdir():
                    match = _PART.match(path.name)
                    if match:
                        parts.append({"tier": tier, "part": int(match.group(1)),
                                      "file": f"series/{tier}/{path.name}"})
                parts.sort(key=lambda r: r["part"])
        return parts

    def part_path(self, part: dict) -> Path:
        return resolve_inside(self.run.path, part["file"])

    def columns(self, tier: str) -> list[str]:
        import pyarrow.parquet as pq
        names: list[str] = []
        seen: set[str] = set()
        for part in self.parts(tier):
            for name in pq.read_schema(self.part_path(part)).names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def tick_range(self, tier: str) -> tuple[int, int] | None:
        import pyarrow.parquet as pq
        lo, hi = None, None
        for part in self.parts(tier):
            if isinstance(part.get("tick0"), int) and isinstance(part.get("tick1"), int):
                a, b = part["tick0"], part["tick1"]
            else:
                meta = pq.ParquetFile(self.part_path(part)).metadata
                index = meta.schema.names.index("tick")
                stats = [meta.row_group(i).column(index).statistics for i in range(meta.num_row_groups)]
                stats = [s for s in stats if s is not None and s.has_min_max]
                if not stats:
                    continue
                a, b = min(int(s.min) for s in stats), max(int(s.max) for s in stats)
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        return None if lo is None else (lo, hi)

    def iter_chunks(self, tier: str, columns: list[str], tick0: int | None,
                    tick1: int | None) -> Iterator[tuple[np.ndarray, dict]]:
        name, function = series_backend()
        if function is not None:
            yield from function(self.run.path, tier, list(columns), tick0, tick1)
            return
        import pyarrow.parquet as pq
        for part in self.parts(tier):
            if tick0 is not None and isinstance(part.get("tick1"), int) and part["tick1"] < tick0:
                continue
            if tick1 is not None and isinstance(part.get("tick0"), int) and part["tick0"] > tick1:
                continue
            path = self.part_path(part)
            present = set(pq.read_schema(path).names)
            if "tick" not in present:
                raise ApiError(500, f"run {self.run.id!r}: {part['file']} has no 'tick' column")
            wanted = [c for c in columns if c in present]
            filters = []
            if tick0 is not None:
                filters.append(("tick", ">=", int(tick0)))
            if tick1 is not None:
                filters.append(("tick", "<=", int(tick1)))
            table = pq.read_table(path, columns=["tick"] + wanted, filters=filters or None, use_threads=False)
            if table.num_rows == 0:
                continue
            ticks = table.column("tick").to_numpy()
            data = {}
            for column in columns:
                if column in present:
                    data[column] = np.asarray(table.column(column).to_numpy(zero_copy_only=False), np.float64)
                else:
                    data[column] = np.full(len(ticks), np.nan)
            yield ticks, data


def base_channel(column: str) -> str:
    return _SUFFIX.sub("", column)


def select_columns(available: list[str], tokens: list[str], tier: str) -> list[str]:
    selected: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token in ("tick", "time_s"):
            continue
        pattern = re.compile(re.escape(token) + _SUFFIX.pattern)
        matches = [c for c in available if c not in ("tick", "time_s") and pattern.match(c)]
        if not matches:
            close = sorted({base_channel(c) for c in available if token.split(".")[0] in c})[:8]
            hint = f" (channels like: {', '.join(close)})" if close else ""
            raise ApiError(404, f"the {tier} tier has no channel or column {token!r}{hint}")
        for column in matches:
            if column not in selected:
                selected.append(column)
    if len(selected) > MAX_COLUMNS:
        raise ApiError(400, f"{len(selected)} columns selected; at most {MAX_COLUMNS} per request")
    return selected


def query(run: RunDir, tier: str, tokens: list[str], t0_s: float | None, t1_s: float | None,
          max_points: int) -> dict:
    """Columns of one tier between t0 and t1, M4-downsampled when there are more than max_points rows."""
    reader = SeriesReader(run)
    parts = reader.parts(tier)
    if not parts:
        raise ApiError(404, f"run {run.id!r} has no {tier} series (yet)")
    if not tokens:
        raise ApiError(400, "name the channels: channels=a,b (GET /api/runs/{id}/channels lists them)")
    columns = select_columns(reader.columns(tier), tokens, tier)
    tick_s = run.tick_s()
    tick0 = seconds_to_tick(t0_s, tick_s) if t0_s is not None else None
    tick1 = seconds_to_tick(t1_s, tick_s) if t1_s is not None else None
    if tick0 is not None and tick1 is not None and tick1 < tick0:
        raise ApiError(400, "t1 is before t0")
    buckets = max(1, max_points // 4)
    raw_ticks: list[np.ndarray] = []
    raw: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    accumulators: dict[str, M4] | None = None
    rows = 0
    for ticks, data in reader.iter_chunks(tier, columns, tick0, tick1):
        rows += len(ticks)
        if accumulators is None and rows <= max_points:
            raw_ticks.append(ticks)
            for column in columns:
                raw[column].append(data[column])
            continue
        if accumulators is None:
            lo, hi = _bucket_range(reader, tier, tick0, tick1, raw_ticks + [ticks])
            accumulators = {c: M4(lo, hi, buckets) for c in columns}
            for index, chunk in enumerate(raw_ticks):
                for column in columns:
                    accumulators[column].add(chunk, raw[column][index])
            raw_ticks, raw = [], {}
        for column in columns:
            accumulators[column].add(ticks, data[column])
    units = run.channel_units()
    series = []
    for column in columns:
        if accumulators is None:
            ticks = np.concatenate(raw_ticks) if raw_ticks else np.zeros(0, np.int64)
            values = np.concatenate(raw[column]) if raw_ticks else np.zeros(0)
        else:
            ticks, values = accumulators[column].result()
        channel = base_channel(column)
        series.append({"column": column, "channel": channel, "unit": units.get(channel, ""), "n": int(len(ticks)),
                       "t_s": ticks_to_seconds(ticks, tick_s), "v": values})
    return {"run_id": run.id, "tier": tier, "tick_s": tick_s, "t0_s": t0_s, "t1_s": t1_s, "rows": rows,
            "max_points": max_points, "downsampled": accumulators is not None,
            "method": "m4" if accumulators is not None else "raw",
            "buckets": buckets if accumulators is not None else None, "series": series}


def _bucket_range(reader: SeriesReader, tier: str, tick0: int | None, tick1: int | None,
                  seen: list[np.ndarray]) -> tuple[int, int]:
    """The ticks the M4 buckets span: the query's t0/t1, else the tier's whole range (from the
    index or the Parquet statistics), else what has been read so far."""
    if tick0 is not None and tick1 is not None:
        return tick0, tick1
    span = reader.tick_range(tier)
    seen_lo = min(int(t.min()) for t in seen if t.size)
    seen_hi = max(int(t.max()) for t in seen if t.size)
    lo = tick0 if tick0 is not None else (min(span[0], seen_lo) if span else seen_lo)
    hi = tick1 if tick1 is not None else (max(span[1], seen_hi) if span else seen_hi)
    return lo, hi


def export_columns(run: RunDir, tier: str, tokens: list[str]) -> list[str]:
    reader = SeriesReader(run)
    if not reader.parts(tier):
        raise ApiError(404, f"run {run.id!r} has no {tier} series (yet)")
    available = reader.columns(tier)
    if tokens:
        return select_columns(available, tokens, tier)
    return [c for c in available if c not in ("tick", "time_s")]


def _table(ticks: np.ndarray, data: dict, columns: list[str], tick_s: float):
    import pyarrow as pa
    import pyarrow.compute as pc
    arrays = {"tick": pa.array(ticks, pa.int64()), "time_s": pa.array(ticks_to_seconds(ticks, tick_s), pa.float64())}
    for column in columns:
        values = pa.array(data[column], pa.float64())
        arrays[column] = pc.if_else(pc.is_nan(values), pa.scalar(None, pa.float64()), values)
    return pa.table(arrays)


def export_csv(run: RunDir, tier: str, columns: list[str], t0_s: float | None, t1_s: float | None) -> Iterator[bytes]:
    """CSV text, part by part (header first): tick, time_s, columns; NaN written as empty."""
    import pyarrow.csv as pacsv
    reader = SeriesReader(run)
    tick_s = run.tick_s()
    tick0 = seconds_to_tick(t0_s, tick_s) if t0_s is not None else None
    tick1 = seconds_to_tick(t1_s, tick_s) if t1_s is not None else None
    header = True
    for ticks, data in reader.iter_chunks(tier, columns, tick0, tick1):
        buffer = io.BytesIO()
        pacsv.write_csv(_table(ticks, data, columns, tick_s), buffer,
                        write_options=pacsv.WriteOptions(include_header=header))
        header = False
        yield buffer.getvalue()
    if header:
        names = ",".join(f'"{c}"' for c in ["tick", "time_s"] + columns)
        yield (names + "\n").encode("utf-8")


def export_parquet(run: RunDir, tier: str, columns: list[str], t0_s: float | None, t1_s: float | None,
                   target: Path) -> int:
    """One Parquet file (tick, time_s, columns) written part by part; returns the row count."""
    import pyarrow.parquet as pq
    reader = SeriesReader(run)
    tick_s = run.tick_s()
    tick0 = seconds_to_tick(t0_s, tick_s) if t0_s is not None else None
    tick1 = seconds_to_tick(t1_s, tick_s) if t1_s is not None else None
    writer = None
    rows = 0
    try:
        for ticks, data in reader.iter_chunks(tier, columns, tick0, tick1):
            table = _table(ticks, data, columns, tick_s)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
        if writer is None:
            empty = _table(np.zeros(0, np.int64), {c: np.zeros(0) for c in columns}, columns, tick_s)
            writer = pq.ParquetWriter(target, empty.schema, compression="zstd")
            writer.write_table(empty)
    finally:
        if writer is not None:
            writer.close()
    return rows
