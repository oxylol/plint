"""Generic CSV machine logs with a column mapping, into a MachineLog.

``mapping`` (the project's ``machine_data.mapping``; values are text): ``<channel>: "<column>
[unit]"`` for each channel to read, ``time: "<column> [<time unit>]"`` for the time column, and
the options

* ``csv.delimiter`` (default: the most frequent of ``,`` ``;`` tab in the header line),
* ``csv.decimal`` (``.`` default, ``,`` for decimal commas),
* ``csv.kind.<channel>`` (``sample`` or ``event``, default from the channel table),
* ``csv.text.<channel>`` (``true``: the column holds text, such as a block or a program name).

Time units: ``iso`` (ISO 8601 date-times; the default when the first cell is not a number),
``s``, ``ms``, ``us``, ``ns``, ``min``, ``h`` (relative), ``epoch_s``, ``epoch_ms``, ``epoch_us``,
``epoch_ns`` (since 1970-01-01 UTC). A column unit converts to the channel's SI unit (``pct`` to a
``_frac`` channel, ``mm`` to ``_m``, ``rpm`` to ``_rad_s``, ``degC`` to ``_K``, ``kW``, ``mm/min``
...); without a unit the column is taken to be in that SI unit already. Empty cells and ``nan``
are gaps. Lines starting with ``#`` are comments. Unmapped columns are listed in
``log.meta["unmapped_columns"]``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ..core.errors import InputError
from .log import NUMERIC_TIME_UNITS, NS_PER_S, LogBuilder, MachineLog, conversion, default_kind, parse_time_ns
from .mtconnect import split_unit
from .params import import_params

TIME_UNITS = ("iso",) + tuple(NUMERIC_TIME_UNITS) + tuple(f"epoch_{u}" for u in ("s", "ms", "us", "ns"))


def _number(text: str, decimal: str) -> float | None:
    cell = text.strip()
    if cell == "" or cell.lower() in ("nan", "null", "none", "unavailable", "-"):
        return None
    if decimal != ".":
        cell = cell.replace(".", "").replace(decimal, ".")
    try:
        return float(cell)
    except ValueError:
        raise ValueError(f"{text!r} is not a number") from None


def read_csv_log(path, mapping: dict, *, base_dir=None, start_ns: int | None = None,
                 end_ns: int | None = None) -> MachineLog:
    """A MachineLog of the mapped columns of a CSV file."""
    path = Path(path)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise InputError(f"cannot read: {error.strerror}", str(path)) from None
    except UnicodeDecodeError as error:
        raise InputError(f"not UTF-8 text: {error}", str(path)) from None
    if not mapping or "time" not in mapping:
        raise InputError("a CSV log needs a mapping with time: \"<column> [unit]\" and one entry per channel",
                         str(path))
    options = {k: str(v) for k, v in mapping.items() if k.startswith("csv.")}
    columns = {k: split_unit(str(v)) for k, v in mapping.items() if not k.startswith("csv.") and k not in ("note", "notes")}
    lines = [(i, line) for i, line in enumerate(text.splitlines(), start=1)
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise InputError("the CSV file has no rows", str(path))
    header_line = lines[0][1]
    delimiter = options.get("csv.delimiter")
    if delimiter in ("tab", "\\t"):
        delimiter = "\t"
    if not delimiter:
        delimiter = max((",", ";", "\t"), key=header_line.count)
    decimal = options.get("csv.decimal", ".")
    if decimal not in (".", ","):
        raise InputError(f"csv.decimal must be '.' or ',', not {decimal!r}", str(path))
    if decimal == "," and delimiter == ",":
        raise InputError("csv.decimal ',' needs another delimiter (csv.delimiter: ';')", str(path))
    reader = csv.reader(io.StringIO("\n".join(line for _, line in lines)), delimiter=delimiter)
    rows = list(reader)
    header = [h.strip() for h in rows[0]]
    index = {name: i for i, name in enumerate(header)}
    for channel, (column, _) in columns.items():
        if column not in index:
            raise InputError(f"mapping {channel}: no column {column!r} (columns: {', '.join(header)})", str(path))
    time_column, time_unit = columns.pop("time")
    if time_unit is None:
        first = rows[1][index[time_column]].strip() if len(rows) > 1 else ""
        try:
            float(first.replace(",", ".") if decimal == "," else first)
            time_unit = "s"
        except ValueError:
            time_unit = "iso"
    if time_unit not in TIME_UNITS:
        raise InputError(f"time unit {time_unit!r}: use one of {', '.join(TIME_UNITS)}", str(path))
    builder = LogBuilder({"kind": "csv", "path": str(path), "columns": {c: v[0] for c, v in sorted(columns.items())}})
    plan = []
    for channel in sorted(columns):
        column, unit = columns[channel]
        text_column = options.get(f"csv.text.{channel}", "false").lower() in ("true", "1", "yes")
        factor, offset = (1.0, 0.0) if text_column else conversion(unit, channel)
        kind = options.get(f"csv.kind.{channel}", default_kind(channel))
        if kind not in ("sample", "event"):
            raise InputError(f"csv.kind.{channel}: {kind!r} is not sample or event", str(path))
        plan.append((channel, index[column], factor, offset, kind, text_column))
    limit = int(import_params()["max_observations"])
    zone_missing = False
    for row_number, row in enumerate(rows[1:], start=1):
        line = lines[row_number][0]
        if len(row) != len(header):
            raise InputError(f"{len(row)} cells, the header has {len(header)}", str(path), line)
        cell = row[index[time_column]].strip()
        try:
            if time_unit == "iso":
                t, zone = parse_time_ns(cell)
                zone_missing |= not zone
            else:
                value = _number(cell, decimal)
                if value is None:
                    raise ValueError("empty time")
                if time_unit.startswith("epoch_"):
                    t = int(round(value * NUMERIC_TIME_UNITS[time_unit[len("epoch_"):]] * NS_PER_S))
                else:
                    t = value * NUMERIC_TIME_UNITS[time_unit]
        except (ValueError, InputError) as error:
            raise InputError(f"time {cell!r}: {error}", str(path), line) from None
        if isinstance(t, int) and ((start_ns is not None and t < start_ns) or (end_ns is not None and t > end_ns)):
            continue
        for channel, column, factor, offset, kind, text_column in plan:
            raw = row[column]
            if text_column:
                value = raw.strip() or None
                builder.add(channel, t, value, kind=kind, source=header[column], text=True)
                continue
            try:
                number = _number(raw, decimal)
            except ValueError as error:
                raise InputError(f"column {header[column]}: {error}", str(path), line) from None
            builder.add(channel, t, None if number is None else number * factor + offset, kind=kind,
                        source=header[column])
        if builder.count > limit:
            raise InputError(f"more than {limit} observations (mdata-import max_observations): import a time "
                             f"window", str(path), line)
    if zone_missing:
        builder.meta["timezone_assumed"] = "UTC"
    log = builder.build(limit)
    used = {c for c, _ in columns.values()} | {time_column}
    log.meta["unmapped_columns"] = [h for h in header if h not in used]
    return log


__all__ = ["read_csv_log", "TIME_UNITS"]
