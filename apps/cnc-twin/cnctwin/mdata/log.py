"""MachineLog: measured machine data as time-stamped channels in SI (docs/models/calibration.md "MachineLog").

Every importer (MTConnect, OPC UA, FOCAS collector logs, generic CSV) produces one
:class:`MachineLog`. A channel has a canonical name whose last token is its SI unit
(``spindle.load_frac``, ``axis.X.pos_machine_m``, ``temp.spindle_K``; table :data:`CHANNELS`),
a kind (``sample``: a continuous quantity, interpolated linearly; ``event``: a value that holds
until the next one, such as the program line or the tool number) and its observations
``(t_s, value)`` in time order. Text values (program name, block text, execution state) are
event channels with ``object`` values. Alarms, messages and conditions are :class:`MachineEvent`
records in ``log.events``.

Time: ``t_s`` is float64 seconds since the log's epoch; ``epoch_ns`` is that epoch as integer
nanoseconds since 1970-01-01T00:00:00Z (UTC), or None for a source with relative times only.
Absolute times are parsed by :func:`parse_time_ns` (ISO 8601 with up to nine fraction digits, a
``Z`` or ``+hh:mm`` zone; a time without zone is read as UTC and the log says so in
``meta["timezone_assumed"]``).

Files: :meth:`MachineLog.save` writes ``machine_log.json`` (format ``cnctwin-machine-log``: source,
epoch, channel list, events, meta) and ``machine_log.parquet`` (long table ``channel, t_s, value,
text``), both deterministic; :meth:`MachineLog.load` reads them back exactly.
"""

from __future__ import annotations

import calendar
import json
import math
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import InputError
from ..db import keyunits

FORMAT = "cnctwin-machine-log"
FORMAT_VERSION = 1
KINDS = ("sample", "event")

# SI unit tokens a channel name may end with, and how they print.
SI_UNITS = {
    "m": "m", "rad": "rad", "m_s": "m/s", "rad_s": "rad/s", "m_s2": "m/s2", "W": "W", "N": "N", "Nm": "N m",
    "K": "K", "s": "s", "Hz": "Hz", "A": "A", "V": "V", "frac": "-", "Pa": "Pa", "J": "J", "m3_s": "m3/s",
}

# Canonical channels (name -> kind, doc). Per-axis channels are ``axis.<NAME>.<quantity>`` with the
# quantities of AXIS_QUANTITIES; temperatures are ``temp.<where>_K``; diagnostics ``diag.<n>``.
CHANNELS = {
    "spindle.speed_rad_s": ("sample", "spindle speed magnitude (unsigned; the direction is spindle.direction)"),
    "spindle.speed_cmd_rad_s": ("event", "commanded spindle speed magnitude"),
    "spindle.direction": ("event", "+1 M03 (clockwise seen from the spindle), -1 M04, 0 stopped"),
    "spindle.load_frac": ("sample", "spindle load meter: motor output over its reference (rated) output"),
    "spindle.power_W": ("sample", "electrical input power of the spindle drive"),
    "spindle.torque_Nm": ("sample", "spindle motor torque"),
    "spindle.current_A": ("sample", "spindle motor current"),
    "spindle.override_frac": ("event", "spindle speed override"),
    "spindle.growth_m": ("sample", "axial spindle (tool tip) growth measured along the tool axis"),
    "spindle.shaft_power_W": ("sample", "shaft power of cutting (simulated runs: cutting.spindle_power_W)"),
    "path.feed_m_s": ("sample", "actual path feed rate"),
    "path.feed_cmd_m_s": ("event", "programmed / commanded path feed rate"),
    "path.feed_override_frac": ("event", "feed override"),
    "path.rapid_override_frac": ("event", "rapid override"),
    "path.x_work_m": ("sample", "tool control point x in the work coordinate system"),
    "path.y_work_m": ("sample", "tool control point y in the work coordinate system"),
    "path.z_work_m": ("sample", "tool control point z in the work coordinate system"),
    "program.name": ("event", "running program (O number or file name)"),
    "program.main_name": ("event", "main program"),
    "program.line": ("event", "line number of the executing block in its program file"),
    "program.sequence": ("event", "sequence number (N word) of the executing block"),
    "program.block_count": ("event", "count of executed blocks since the program start"),
    "program.block": ("event", "text of the executing block"),
    "program.execution": ("event", "execution state: ACTIVE, READY, STOPPED, FEED_HOLD, INTERRUPTED, "
                                   "PROGRAM_STOPPED, PROGRAM_COMPLETED, OPTIONAL_STOP"),
    "program.part_count": ("event", "parts counter"),
    "controller.mode": ("event", "AUTOMATIC, MANUAL, MANUAL_DATA_INPUT, SEMI_AUTOMATIC, EDIT"),
    "controller.estop": ("event", "emergency stop ARMED / TRIGGERED"),
    "tool.number": ("event", "active tool number"),
    "alarm.count": ("event", "number of active alarms (from alarm list polls)"),
}
AXIS_QUANTITIES = {
    "pos_machine": ("sample", "actual position in machine coordinates"),
    "pos_work": ("sample", "actual position in the work (absolute) coordinate system"),
    "pos_relative": ("sample", "relative position display"),
    "pos_cmd": ("sample", "commanded position (machine coordinates)"),
    "pos_stock": ("sample", "axis value relative to the stock origin (simulated runs: frames.bin)"),
    "distance_to_go": ("sample", "distance to go of the block"),
    "following_error": ("sample", "commanded minus actual position (servo delay)"),
    "load": ("sample", "servo load meter: motor current over its rated current"),
    "feed": ("sample", "axis feed rate"),
    "force": ("sample", "axis drive force"),
    "torque": ("sample", "axis motor torque"),
    "current": ("sample", "axis motor current"),
}
_AXIS = re.compile(r"^axis\.([A-Za-z][A-Za-z0-9]*)\.([a-z_]+?)_(m|rad|frac|m_s|rad_s|N|Nm|A)$")
_TEMP = re.compile(r"^temp\.[a-z0-9_]+_K$")
_GENERIC = re.compile(r"^[a-z][a-z0-9_]*(\.[A-Za-z0-9_]+)+$")


def channel_unit(name: str) -> str:
    """SI unit token of a channel name (its last ``_`` token when that is an SI token), '' if none."""
    last = name.rsplit(".", 1)[-1]
    for token in sorted(SI_UNITS, key=len, reverse=True):
        if last.endswith("_" + token):
            return token
    return ""


def default_kind(name: str) -> str:
    """The kind a channel has unless its source says otherwise."""
    if name in CHANNELS:
        return CHANNELS[name][0]
    match = _AXIS.match(name)
    if match and match.group(2) in AXIS_QUANTITIES:
        return AXIS_QUANTITIES[match.group(2)][0]
    if name.startswith(("temp.", "diag.", "path.", "spindle")) or channel_unit(name):
        return "sample"
    return "event"


def check_channel_name(name: str) -> None:
    """Raise InputError for a name that is not a canonical channel name (dotted, unit last)."""
    if name in CHANNELS or _TEMP.match(name):
        return
    match = _AXIS.match(name)
    if match:
        if match.group(2) not in AXIS_QUANTITIES:
            raise InputError(f"channel {name!r}: axis quantity {match.group(2)!r} is not one of "
                             f"{', '.join(sorted(AXIS_QUANTITIES))}")
        return
    if _GENERIC.match(name):
        return
    raise InputError(f"channel {name!r} is not a channel name: write dotted lower-case names with the SI unit "
                     f"last (spindle.load_frac, axis.X.pos_machine_m, temp.spindle_K)")


# ----------------------------------------------------------------------------- units of source values

_EXTRA_SOURCE_UNITS = {
    "A": keyunits.Unit("A", "A"), "mA": keyunits.Unit("mA", "A", 1e-3), "kA": keyunits.Unit("kA", "A", 1e3),
    "frac": keyunits.Unit("frac", "frac"), "rad": keyunits.Unit("rad", "rad"),
    "m_s2": keyunits.Unit("m_s2", "m_s2"), "Nm": keyunits.Unit("Nm", "Nm"), "N_m": keyunits.Unit("N_m", "Nm"),
    "J": keyunits.Unit("J", "J"), "mm_s": keyunits.Unit("mm_s", "m_s", 1e-3), "m_s": keyunits.Unit("m_s", "m_s"),
    "inch": keyunits.Unit("inch", "m", keyunits.UNITS["in"].factor),
}


def source_unit(text: str) -> keyunits.Unit | None:
    """The unit a source states (``mm``, ``pct``, ``%``, ``rpm``, ``degC``, ``kW``, ``mm/min``, ``A``,
    ``-``): None for dimensionless. Raises InputError for a unit that cannot be converted."""
    raw = str(text).strip()
    norm = raw.replace("/", "_").replace("%", "pct").replace(" ", "").replace("·", "")
    if norm in _EXTRA_SOURCE_UNITS:
        return _EXTRA_SOURCE_UNITS[norm]
    try:
        return keyunits.parse_unit_text(raw)
    except ValueError as error:
        raise InputError(str(error)) from None


def conversion(unit_text: str | None, channel: str) -> tuple[float, float]:
    """(factor, offset) taking a source value in ``unit_text`` to the SI unit of ``channel``.

    The unit must convert to the channel's SI token (``pct`` to ``frac``, ``mm`` to ``m``, ``rpm`` to
    ``rad_s``, ``degC`` to ``K``); no unit (or ``-``) means the value is already in that unit."""
    target = channel_unit(channel)
    if unit_text is None or str(unit_text).strip() in ("", "-"):
        return 1.0, 0.0
    unit = source_unit(unit_text)
    if unit is None:
        if target in ("", "frac"):
            return 1.0, 0.0
        raise InputError(f"channel {channel}: unit {unit_text!r} is dimensionless, the channel is in {target}")
    si = unit.si_token
    if si != target and not (si == "dK" and target == "K"):
        raise InputError(f"channel {channel}: unit {unit_text!r} converts to {si}, the channel is in "
                         f"{target or 'no unit'}")
    return float(unit.factor), float(unit.offset)


# ----------------------------------------------------------------------------- time

_ISO = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,9}))?)?"
                  r"\s*(Z|z|UTC|[+-]\d{2}(?::?\d{2})?)?\s*$")
NS_PER_S = 1_000_000_000


def parse_time_ns(text: str) -> tuple[int, bool]:
    """(nanoseconds since 1970-01-01T00:00:00Z, zone_given) of an ISO 8601 date-time."""
    match = _ISO.match(str(text))
    if not match:
        raise InputError(f"time {text!r} is not an ISO 8601 date-time (YYYY-MM-DDTHH:MM:SS[.f][Z|+hh:mm])")
    year, month, day, hour, minute = (int(match.group(i)) for i in range(1, 6))
    second = int(match.group(6) or 0)
    fraction = match.group(7) or ""
    zone = match.group(8)
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59 and second <= 60):
        raise InputError(f"time {text!r} has a field out of range")
    seconds = calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0))
    nanos = int(fraction.ljust(9, "0")) if fraction else 0
    offset_s = 0
    if zone and zone not in ("Z", "z", "UTC"):
        sign = -1 if zone[0] == "-" else 1
        digits = zone[1:].replace(":", "")
        offset_s = sign * (int(digits[:2]) * 3600 + (int(digits[2:4]) * 60 if len(digits) >= 4 else 0))
    return (seconds - offset_s) * NS_PER_S + nanos, zone is not None


def format_time_ns(ns: int) -> str:
    """ISO 8601 UTC text of integer nanoseconds (nine fraction digits only when needed)."""
    seconds, nanos = divmod(int(ns), NS_PER_S)
    base = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime(seconds))
    if nanos == 0:
        return base + "Z"
    text = f"{nanos:09d}".rstrip("0")
    return f"{base}.{text}Z"


NUMERIC_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "min": 60.0, "h": 3600.0}


# ----------------------------------------------------------------------------- the log

@dataclass
class Channel:
    name: str
    t_s: np.ndarray                  # float64, non-decreasing
    values: np.ndarray               # float64 (numeric) or object (text)
    kind: str = "sample"
    source: str = ""                 # item id / node id / column the values came from

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise InputError(f"channel {self.name}: kind {self.kind!r} is not one of {KINDS}")
        self.t_s = np.asarray(self.t_s, dtype=np.float64)
        if not isinstance(self.values, np.ndarray) or self.values.dtype.kind in "US":
            text = _is_text(list(self.values) if not isinstance(self.values, np.ndarray) else self.values)
            self.values = np.asarray(self.values, dtype=object if text else np.float64)
        elif self.values.dtype != object:
            self.values = np.asarray(self.values, dtype=np.float64)
        if self.t_s.shape != self.values.shape or self.t_s.ndim != 1:
            raise InputError(f"channel {self.name}: {len(self.t_s)} times but {len(self.values)} values")

    @property
    def unit(self) -> str:
        return SI_UNITS.get(channel_unit(self.name), "")

    @property
    def numeric(self) -> bool:
        return self.values.dtype != object

    def __len__(self) -> int:
        return len(self.t_s)

    def finite(self) -> "Channel":
        """The observations with a value (numeric NaN and text None are gaps)."""
        if self.numeric:
            keep = np.isfinite(self.values)
        else:
            keep = np.array([v is not None for v in self.values], dtype=bool)
        return Channel(self.name, self.t_s[keep], self.values[keep], self.kind, self.source)

    def at(self, t_s, *, max_gap_s: float | None = None) -> np.ndarray:
        """Values at times ``t_s``: linear interpolation for samples, the last value for events (NaN /
        None before the first observation, and for samples outside the data or across gaps longer
        than ``max_gap_s``)."""
        t = np.asarray(t_s, dtype=np.float64)
        data = self.finite()
        if data.kind == "event" or not data.numeric:
            index = np.searchsorted(data.t_s, t, side="right") - 1
            if data.numeric:
                out = np.full(t.shape, np.nan)
                ok = index >= 0
                out[ok] = data.values[index[ok]]
                return out
            out = np.empty(t.shape, dtype=object)
            for i, k in enumerate(index.ravel()):
                out.ravel()[i] = data.values[k] if k >= 0 else None
            return out
        out = np.full(t.shape, np.nan)
        if len(data) == 0:
            return out
        if len(data) == 1:
            out[t == data.t_s[0]] = data.values[0]
            return out
        inside = (t >= data.t_s[0]) & (t <= data.t_s[-1])
        out[inside] = np.interp(t[inside], data.t_s, data.values)
        if max_gap_s is not None:
            right = np.clip(np.searchsorted(data.t_s, t, side="right"), 1, len(data) - 1)
            gap = data.t_s[right] - data.t_s[right - 1]
            out[inside & (gap > max_gap_s)] = np.nan
        return out

    def median_dt_s(self) -> float:
        data = self.finite()
        if len(data) < 2:
            return math.nan
        steps = np.diff(data.t_s)
        steps = steps[steps > 0]
        return float(np.median(steps)) if len(steps) else math.nan

    def resolution(self) -> float:
        """Smallest non-zero step between successive values (the quantisation of the reading)."""
        data = self.finite()
        if not data.numeric or len(data) < 2:
            return 0.0
        steps = np.abs(np.diff(data.values))
        steps = steps[steps > 0]
        return float(steps.min()) if len(steps) else 0.0

    def window(self, t0: float, t1: float) -> "Channel":
        keep = (self.t_s >= t0) & (self.t_s <= t1)
        return Channel(self.name, self.t_s[keep], self.values[keep], self.kind, self.source)


@dataclass
class MachineEvent:
    t_s: float
    kind: str                        # alarm | alarm_cleared | message | condition
    code: str = ""                   # native code ("PS0010", "SV0401") or condition type
    text: str = ""
    severity: str = ""               # fault | warning | normal | unavailable | info
    source: str = ""                 # component / data item / FOCAS function

    def to_dict(self) -> dict:
        return {"t_s": self.t_s, "kind": self.kind, "code": self.code, "text": self.text, "severity": self.severity,
                "source": self.source}


@dataclass
class MachineLog:
    channels: dict[str, Channel] = field(default_factory=dict)
    events: list[MachineEvent] = field(default_factory=list)
    epoch_ns: int | None = None
    source: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    def __contains__(self, name: str) -> bool:
        return name in self.channels and len(self.channels[name].finite()) > 0

    def channel(self, name: str) -> Channel:
        if name not in self.channels:
            raise InputError(f"the machine log has no channel {name!r} (it has: {', '.join(self.names()) or 'none'})")
        return self.channels[name]

    def names(self) -> list[str]:
        return sorted(self.channels)

    def first(self, *names: str) -> Channel | None:
        """The first of ``names`` the log has data for."""
        for name in names:
            if name in self:
                return self.channels[name]
        return None

    def axes(self) -> list[str]:
        found = set()
        for name in self.channels:
            match = _AXIS.match(name)
            if match:
                found.add(match.group(1))
        return sorted(found)

    def span_s(self) -> tuple[float, float]:
        starts = [c.t_s[0] for c in self.channels.values() if len(c)]
        ends = [c.t_s[-1] for c in self.channels.values() if len(c)]
        times = [e.t_s for e in self.events]
        if not starts and not times:
            return 0.0, 0.0
        return float(min(starts + times)), float(max(ends + times))

    def count(self) -> int:
        return sum(len(c) for c in self.channels.values()) + len(self.events)

    def add(self, name: str, t_s, values, *, kind: str | None = None, source: str = "") -> Channel:
        """Add (or extend) a channel; observations are kept in time order (stable)."""
        check_channel_name(name)
        t = np.asarray(t_s, dtype=np.float64)
        v = np.asarray(values, dtype=object) if _is_text(values) else np.asarray(values, dtype=np.float64)
        if name in self.channels:
            old = self.channels[name]
            if old.numeric != (v.dtype != object):
                raise InputError(f"channel {name}: numeric and text values mixed")
            t = np.concatenate([old.t_s, t])
            v = np.concatenate([old.values, v])
            kind = kind or old.kind
            source = old.source if not source or source == old.source else f"{old.source}, {source}"
        order = np.argsort(t, kind="stable")
        channel = Channel(name, t[order], v[order], kind or default_kind(name), source)
        self.channels[name] = channel
        return channel

    def shifted(self, offset_s: float, scale: float = 1.0) -> "MachineLog":
        """A copy with times mapped t' = offset + scale * t (epoch dropped: the times are another base)."""
        out = MachineLog(events=[MachineEvent(offset_s + scale * e.t_s, e.kind, e.code, e.text, e.severity, e.source)
                                 for e in self.events], epoch_ns=None, source=dict(self.source), meta=dict(self.meta))
        for name, c in self.channels.items():
            out.channels[name] = Channel(name, offset_s + scale * c.t_s, c.values.copy(), c.kind, c.source)
        return out

    def window(self, t0: float, t1: float) -> "MachineLog":
        out = MachineLog(events=[e for e in self.events if t0 <= e.t_s <= t1], epoch_ns=self.epoch_ns,
                         source=dict(self.source), meta=dict(self.meta))
        for name, c in self.channels.items():
            out.channels[name] = c.window(t0, t1)
        return out

    def summary(self) -> dict:
        t0, t1 = self.span_s()
        return {"source": self.source, "epoch": format_time_ns(self.epoch_ns) if self.epoch_ns is not None else None,
                "span_s": [t0, t1], "observations": self.count(), "events": len(self.events),
                "channels": [{"name": n, "unit": c.unit, "kind": c.kind, "count": len(c), "source": c.source,
                              "median_dt_s": _json_float(c.median_dt_s())}
                             for n, c in sorted(self.channels.items())],
                "meta": self.meta}

    # ------------------------------------------------------------------ files
    def save(self, directory) -> list[Path]:
        """machine_log.json + machine_log.parquet in ``directory`` (deterministic bytes)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        names, times, numbers, texts = [], [], [], []
        for name in sorted(self.channels):
            c = self.channels[name]
            names.extend([name] * len(c))
            times.append(c.t_s)
            if c.numeric:
                numbers.append(c.values)
                texts.extend([None] * len(c))
            else:
                numbers.append(np.full(len(c), np.nan))
                texts.extend(None if v is None else str(v) for v in c.values)
        table = pa.table({"channel": pa.array(names, pa.string()),
                          "t_s": pa.array(np.concatenate(times) if times else np.zeros(0), pa.float64()),
                          "value": pa.array(np.concatenate(numbers) if numbers else np.zeros(0), pa.float64()),
                          "text": pa.array(texts, pa.string())})
        parquet = directory / "machine_log.parquet"
        tmp = parquet.with_name(".machine_log.parquet.tmp")
        pq.write_table(table, tmp, compression="zstd", row_group_size=1 << 16)
        tmp.replace(parquet)
        document = {"format": FORMAT, "version": FORMAT_VERSION, "source": self.source,
                    "epoch_ns": self.epoch_ns,
                    "epoch": format_time_ns(self.epoch_ns) if self.epoch_ns is not None else None,
                    "channels": [{"name": n, "kind": c.kind, "unit": c.unit, "source": c.source,
                                  "numeric": c.numeric, "count": len(c)} for n, c in sorted(self.channels.items())],
                    "events": [e.to_dict() for e in self.events], "meta": self.meta,
                    "data": parquet.name}
        path = directory / "machine_log.json"
        tmp_json = path.with_name(".machine_log.json.tmp")
        tmp_json.write_text(json.dumps(_plain(document), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp_json.replace(path)
        return [path, parquet]

    @classmethod
    def load(cls, directory) -> "MachineLog":
        import pyarrow.parquet as pq

        directory = Path(directory)
        path = directory / "machine_log.json" if directory.is_dir() else directory
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise InputError(f"cannot read the machine log: {error}", str(path)) from None
        if document.get("format") != FORMAT:
            raise InputError(f"not a {FORMAT} file", str(path))
        table = pq.read_table(path.parent / document.get("data", "machine_log.parquet"))
        names = np.asarray(table.column("channel").to_pylist(), dtype=object)
        times = table.column("t_s").to_numpy()
        values = table.column("value").to_numpy()
        texts = np.asarray(table.column("text").to_pylist(), dtype=object)
        log = cls(epoch_ns=document.get("epoch_ns"), source=document.get("source") or {},
                  meta=document.get("meta") or {})
        for entry in document.get("channels", []):
            rows = names == entry["name"]
            data = values[rows] if entry.get("numeric", True) else texts[rows]
            log.channels[entry["name"]] = Channel(entry["name"], times[rows], data, entry.get("kind", "sample"),
                                                  entry.get("source", ""))
        log.events = [MachineEvent(**e) for e in document.get("events", [])]
        return log


def _is_text(values) -> bool:
    if isinstance(values, np.ndarray):
        return values.dtype == object or values.dtype.kind in "US"
    return any(isinstance(v, str) for v in values)


def _json_float(value):
    return None if value is None or (isinstance(value, float) and not math.isfinite(value)) else value


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return _json_float(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


class LogBuilder:
    """Collects observations with absolute (ns) or relative (s) times and builds a MachineLog.

    Channels are built in name order with observations sorted by time (stable, so equal times keep
    the order they were added in); the epoch is the earliest absolute time seen.
    """

    def __init__(self, source: dict | None = None) -> None:
        self.source = dict(source or {})
        self.rows: dict[str, list] = {}
        self.text: set[str] = set()
        self.kinds: dict[str, str] = {}
        self.sources: dict[str, set] = {}
        self.events: list[tuple[int | float, MachineEvent]] = []
        self.absolute: bool | None = None
        self.meta: dict = {}
        self.count = 0

    def _time(self, t) -> int | float:
        is_abs = isinstance(t, (int, np.integer)) and not isinstance(t, bool)
        if self.absolute is None:
            self.absolute = is_abs
        elif self.absolute != is_abs:
            raise InputError("the source mixes absolute and relative times")
        return int(t) if is_abs else float(t)

    def add(self, name: str, t, value, *, kind: str | None = None, source: str = "", text: bool = False) -> None:
        """``t``: int nanoseconds since the Unix epoch, or float seconds (relative source); ``text`` marks a
        text channel (its gaps are None)."""
        if name not in self.rows:
            check_channel_name(name)
        self.rows.setdefault(name, []).append((self._time(t), value))
        if text or isinstance(value, str):
            self.text.add(name)
        if kind:
            self.kinds[name] = kind
        if source:
            self.sources.setdefault(name, set()).add(source)
        self.count += 1

    def event(self, t, event: MachineEvent) -> None:
        self.events.append((self._time(t), event))
        self.count += 1

    def build(self, max_observations: int | None = None) -> MachineLog:
        if max_observations is not None and self.count > max_observations:
            raise InputError(f"the machine data holds {self.count} observations, more than the {max_observations} "
                             f"a log may hold (mdata-import max_observations): import a time window")
        times = [t for rows in self.rows.values() for t, _ in rows] + [t for t, _ in self.events]
        epoch = min(times) if (times and self.absolute) else None
        log = MachineLog(epoch_ns=int(epoch) if epoch is not None else None, source=self.source, meta=dict(self.meta))

        def rel(t):
            return (t - epoch) / NS_PER_S if epoch is not None else float(t)

        for name in sorted(self.rows):
            rows = self.rows[name]
            t = np.array([rel(r[0]) for r in rows], dtype=np.float64)
            raw = [r[1] for r in rows]
            if name in self.text:
                values = np.empty(len(raw), dtype=object)
                values[:] = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in raw]
            else:
                values = np.array([np.nan if v is None else float(v) for v in raw], dtype=np.float64)
            order = np.argsort(t, kind="stable")
            log.channels[name] = Channel(name, t[order], values[order], self.kinds.get(name, default_kind(name)),
                                         ", ".join(sorted(self.sources.get(name, ()))))
        ordered = sorted(range(len(self.events)), key=lambda i: (self.events[i][0], i))
        log.events = []
        for i in ordered:
            t, event = self.events[i]
            event.t_s = rel(t)
            log.events.append(event)
        return log


def merge(logs: list[MachineLog]) -> MachineLog:
    """One log from several sources of the same machine (absolute times put on one epoch)."""
    if not logs:
        return MachineLog()
    if len(logs) == 1:
        return logs[0]
    epochs = [log.epoch_ns for log in logs]
    if any(e is None for e in epochs) and not all(e is None for e in epochs):
        raise InputError("cannot merge logs with absolute and relative times")
    base = min(epochs) if epochs[0] is not None else None
    out = MachineLog(epoch_ns=base, source={"merged": [log.source for log in logs]}, meta={})
    for log in logs:
        shift = (log.epoch_ns - base) / NS_PER_S if base is not None else 0.0
        for name, c in log.channels.items():
            out.add(name, c.t_s + shift, c.values, kind=c.kind, source=c.source)
        out.events.extend(MachineEvent(e.t_s + shift, e.kind, e.code, e.text, e.severity, e.source) for e in log.events)
        out.meta.update(log.meta)
    out.events.sort(key=lambda e: e.t_s)
    return out


__all__ = ["MachineLog", "Channel", "MachineEvent", "LogBuilder", "CHANNELS", "AXIS_QUANTITIES", "SI_UNITS",
           "channel_unit", "default_kind", "check_channel_name", "conversion", "source_unit", "parse_time_ns",
           "format_time_ns", "merge", "NUMERIC_TIME_UNITS", "FORMAT"]
