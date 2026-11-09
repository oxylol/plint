"""Deterministic writers of the dataset tables: Parquet (zstd, no dictionary encoding, statistics, the schema as key-
value metadata, no timestamps) and CSV (UTF-8, one header row, shortest round-trip floats, NaN as an empty cell,
``\\n`` line ends). The same rows give the same bytes."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

from .schema import Column

DEFAULTS = {"float64": math.nan, "int64": -1, "int8": -1, "string": ""}


def default_of(column: Column):
    if column.dtype == "int64" and column.name.startswith("findings["):
        return 0                       # a run without a finding of that category has 0 of them
    return DEFAULTS[column.dtype]


def _cell(column: Column, value):
    if value is None:
        return default_of(column)
    if column.dtype == "float64":
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan
    if column.dtype in ("int64", "int8"):
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default_of(column)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parquet_bytes(rows: list[dict], columns: list[Column], table: str) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq
    types = {"float64": pa.float64(), "int64": pa.int64(), "int8": pa.int8(), "string": pa.string()}
    arrays = [pa.array([_cell(c, row.get(c.name)) for row in rows], type=types[c.dtype]) for c in columns]
    fields = [pa.field(c.name, types[c.dtype], nullable=False,
                       metadata={"unit": c.unit, "provenance": c.provenance, "doc": c.doc}) for c in columns]
    meta = {"cnctwin": json.dumps({"format": "cnctwin-dataset-table", "version": 1, "table": table}, sort_keys=True)}
    data = pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=meta))
    buffer = io.BytesIO()
    pq.write_table(data, buffer, compression="zstd", use_dictionary=False, write_statistics=True)
    return buffer.getvalue()


def csv_text(rows: list[dict], columns: list[Column]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([c.name for c in columns])
    for row in rows:
        cells = []
        for c in columns:
            value = _cell(c, row.get(c.name))
            if c.dtype == "float64":
                cells.append("" if not math.isfinite(value) else repr(value))
            else:
                cells.append(value)
        writer.writerow(cells)
    return out.getvalue()


def write_table(rows: list[dict], columns: list[Column], out_dir, name: str, formats=("parquet", "csv")) -> list[str]:
    out = Path(out_dir)
    written = []
    if "parquet" in formats:
        _atomic(out / f"{name}.parquet", parquet_bytes(rows, columns, name))
        written.append(f"{name}.parquet")
    if "csv" in formats:
        _atomic(out / f"{name}.csv", csv_text(rows, columns).encode("utf-8"))
        written.append(f"{name}.csv")
    return written


def write_json(path: Path, document) -> None:
    _atomic(Path(path), (json.dumps(document, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n")
            .encode("utf-8"))


__all__ = ["write_table", "parquet_bytes", "csv_text", "write_json", "default_of"]
