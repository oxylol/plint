"""FANUC FOCAS collector logs (CSV or JSON) into a MachineLog.

FOCAS is FANUC's proprietary CNC data library; CNC Twin does not use it. Shop data collectors
built on it poll functions such as ``cnc_rdspmeter`` and write what they return; this module
reads those records (docs/models/calibration.md "FOCAS collector logs" has the full column
reference, fanuc-focas2-reference for the structures).

**CSV** (UTF-8, one header row, ``#`` comment lines, comma or semicolon): one row per returned
element, columns

``time``      ISO 8601 date-time, or seconds (relative, or since the Unix epoch with ``time_unit``)
``function``  the FOCAS function (``cnc_rdspmeter`` ...)
``item``      which output of the function (``load`` / ``speed`` for cnc_rdspmeter; ``absolute``,
              ``machine``, ``relative``, ``distance`` for cnc_rdposition; ``run`` / ``aut`` for
              cnc_statinfo; the address letter for cnc_modal; the diagnosis number for cnc_diagnoss)
``index``     1-based axis / spindle number as FOCAS counts them
``name``      axis or spindle name (``X``, ``S1``; cnc_rdaxisname / cnc_rdspdlname), optional
``value``     the value as returned (an integer with ``dec`` decimals, or a decimal number)
``dec``       decimal places of ``value`` (LOADELM / POSELM ``dec``; empty: ``value`` is decimal)
``unit``      FOCAS unit code (POSELM 0 mm, 1 inch, 2 deg; LOADELM 0 %, 1 rpm) or a unit text
``code``      alarm number (cnc_rdalmmsg ``alm_no``)
``type``      alarm type (0 SW, 1 PW, 2 IO, 3 PS, 4 OT, 5 OH, 6 SV, 7 SR, 8 MC, 9 SP, 10 DS, 11 IE,
              12 BG, 13 SN, 15 EX, 19 PC) or the cnc_rdalmmsg poll marker ``poll`` (no alarm active)
``text``      alarm message / program name

**JSON**: ``{"format": "cnctwin-focas-log", "version": 1, "records": [...]}``, a bare list, or JSON
Lines; a record is a flat object with the CSV columns, or ``{"time", "function", "data": [element,
...]}`` with the per-element columns in each element (a cnc_rdalmmsg poll with an empty ``data``
list means no alarm is active).

Functions and channels: ``cnc_absolute`` -> ``axis.<n>.pos_work_m``; ``cnc_machine`` ->
``pos_machine_m``; ``cnc_relative`` -> ``pos_relative_m``; ``cnc_distance`` ->
``distance_to_go_m`` (rotary axes ``_rad``); ``cnc_rdposition`` by item; ``cnc_rdspload`` and
``cnc_rdspmeter`` load -> ``spindle.load_frac`` (spindle 2 ``spindle2.``...); ``cnc_rdspmeter``
speed and ``cnc_acts`` -> ``spindle.speed_rad_s``; ``cnc_actf`` -> ``path.feed_m_s`` (mm/min);
``cnc_rdsvmeter`` -> ``axis.<n>.load_frac``; ``cnc_srvdelay`` -> ``axis.<n>.following_error_m``
(detection units: give ``dec`` or ``unit``); ``cnc_rdseqnum`` -> ``program.sequence``;
``cnc_rdblkcount`` -> ``program.block_count``; ``cnc_rdprgnum`` -> ``program.name`` ("O1000"; item
``main`` -> ``program.main_name``); ``cnc_modal`` item T -> ``tool.number`` (other letters
``modal.<letter>``); ``cnc_statinfo`` run / aut -> ``program.execution`` / ``controller.mode``;
``cnc_rdalmmsg`` / ``cnc_rdalmmsg2`` -> alarm events (raised when a code appears in a poll, cleared
when it is gone) and ``alarm.count``; ``cnc_diagnoss`` 300 -> following error (detection units),
308 / 309 -> ``temp.axis_<n>_motor_K`` / ``temp.axis_<n>_encoder_K``, 403 ->
``temp.spindle_motor_K`` (degC), other numbers ``diag.<number>`` (``.<axis>``) as written. The
mapping ``{"<function>[:<item>]": "<channel> [unit]"}`` overrides any of these.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path

from ..core.errors import InputError
from ..units import INCH_M, ZERO_DEGC_K, rpm_to_rad_s
from .log import NUMERIC_TIME_UNITS, NS_PER_S, LogBuilder, MachineEvent, MachineLog, conversion, parse_time_ns
from .mtconnect import split_unit
from .params import import_params

MM = 1e-3
DEG = math.pi / 180.0
PCT = 1e-2
RPM = rpm_to_rad_s(1.0)

POSITION_FUNCTIONS = {"cnc_absolute": "pos_work", "cnc_absolute2": "pos_work", "cnc_machine": "pos_machine",
                      "cnc_relative": "pos_relative", "cnc_relative2": "pos_relative", "cnc_distance": "distance_to_go"}
POSITION_ITEMS = {"absolute": "pos_work", "machine": "pos_machine", "relative": "pos_relative",
                  "distance": "distance_to_go"}
POS_UNITS = {"0": "mm", "1": "inch", "2": "deg", "mm": "mm", "inch": "inch", "in": "inch", "deg": "deg"}
ALARM_TYPES = {0: "SW", 1: "PW", 2: "IO", 3: "PS", 4: "OT", 5: "OH", 6: "SV", 7: "SR", 8: "MC", 9: "SP", 10: "DS",
               11: "IE", 12: "BG", 13: "SN", 15: "EX", 19: "PC"}
RUN_STATES = {0: "RESET", 1: "STOPPED", 2: "FEED_HOLD", 3: "ACTIVE", 4: "MSTR"}       # ODBST.run
AUT_MODES = {0: "MANUAL_DATA_INPUT", 1: "AUTOMATIC", 2: "NONE", 3: "EDIT", 4: "MANUAL", 5: "MANUAL", 6: "MANUAL",
             7: "MANUAL", 8: "MANUAL", 9: "MANUAL", 10: "AUTOMATIC"}                    # ODBST.aut (10: remote)
COLUMNS = ("time", "function", "item", "index", "name", "value", "dec", "unit", "code", "type", "text")


def _rotary(name: str) -> bool:
    return bool(name) and name[0].upper() in "ABC"


def _axis_name(element: dict) -> str:
    name = str(element.get("name") or "").strip()
    if name:
        return re.sub(r"[^A-Za-z0-9]", "", name) or "A"
    index = str(element.get("index") or "").strip()
    if not index:
        raise InputError(f"{element.get('function')}: an axis element needs name or index")
    return f"axis{index}"


def _spindle_prefix(element: dict) -> str:
    index = str(element.get("index") or "1").strip() or "1"
    return "spindle" if index in ("1", "0") else f"spindle{index}"


def _scaled(element: dict) -> float | None:
    raw = element.get("value")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InputError(f"{element.get('function')}: value {raw!r} is not a number") from None
    dec = element.get("dec")
    if dec not in (None, ""):
        value = value / (10.0 ** int(float(dec)))
    return value


def _length_factor(element: dict, rotary: bool) -> tuple[float, str]:
    unit = str(element.get("unit") if element.get("unit") is not None else "").strip()
    kind = POS_UNITS.get(unit.lower() if unit else ("2" if rotary else "0"))
    if kind is None:
        factor, _ = conversion(unit, "x.pos_rad" if rotary else "x.pos_m") if unit else (1.0, 0.0)
        return factor, "rad" if rotary else "m"
    if kind == "mm":
        return MM, "m"
    if kind == "inch":
        return INCH_M, "m"
    return DEG, "rad"


class _Reader:
    def __init__(self, mapping: dict | None, source: dict) -> None:
        self.overrides = {}
        self.time_unit = None
        for key, value in (mapping or {}).items():
            if key == "focas.time_unit":
                self.time_unit = str(value)
                continue
            if key.startswith("focas.") or key in ("note", "notes"):
                continue
            self.overrides[str(key)] = split_unit(str(value))
        self.builder = LogBuilder(source)
        self.active_alarms: dict[str, MachineEvent] = {}
        self.pending_alarms: dict[tuple, dict[str, tuple]] = {}
        self.unmapped: dict[str, int] = {}

    # ------------------------------------------------------------------ time
    def time(self, raw):
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raise InputError("a FOCAS record has no time")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return self._numeric_time(float(raw))
        text = str(raw).strip()
        try:
            number = float(text)
        except ValueError:
            ns, zone = parse_time_ns(text)
            if not zone:
                self.builder.meta["timezone_assumed"] = "UTC"
            return ns
        return self._numeric_time(number)

    def _numeric_time(self, number: float):
        unit = self.time_unit or "s"
        if unit.startswith("epoch_"):
            factor = NUMERIC_TIME_UNITS.get(unit[len("epoch_"):])
            if factor is None:
                raise InputError(f"focas.time_unit {unit!r}: use s, ms, us, ns, epoch_s, epoch_ms, epoch_us, epoch_ns")
            return int(round(number * factor * NS_PER_S))
        factor = NUMERIC_TIME_UNITS.get(unit)
        if factor is None:
            raise InputError(f"focas.time_unit {unit!r}: use s, ms, us, ns, epoch_s, epoch_ms, epoch_us, epoch_ns")
        return number * factor

    # ------------------------------------------------------------------ one element
    def element(self, t, element: dict) -> None:
        function = str(element.get("function") or "").strip().lower()
        item = str(element.get("item") or "").strip()
        for key in (f"{function}:{item}" if item else None, function):
            if key and key in self.overrides:
                channel, unit = self.overrides[key]
                value = _scaled(element)
                factor, offset = conversion(unit, channel)
                self.builder.add(channel, t, None if value is None else value * factor + offset, source=function)
                return
        if function in POSITION_FUNCTIONS or (function == "cnc_rdposition" and item.lower() in POSITION_ITEMS):
            quantity = POSITION_FUNCTIONS.get(function) or POSITION_ITEMS[item.lower()]
            name = _axis_name(element)
            factor, unit = _length_factor(element, _rotary(name))
            value = _scaled(element)
            self.builder.add(f"axis.{name}.{quantity}_{unit}", t, None if value is None else value * factor,
                             source=function)
        elif function == "cnc_rdspload" or (function == "cnc_rdspmeter" and item.lower() in ("load", "0", "")):
            value = _scaled(element)
            self.builder.add(f"{_spindle_prefix(element)}.load_frac", t, None if value is None else value * PCT,
                             source=function)
        elif (function == "cnc_rdspmeter" and item.lower() in ("speed", "1")) or function == "cnc_acts":
            value = _scaled(element)
            self.builder.add(f"{_spindle_prefix(element)}.speed_rad_s", t, None if value is None else abs(value) * RPM,
                             source=function)
        elif function == "cnc_actf":
            value = _scaled(element)
            unit = str(element.get("unit") or "mm/min")
            factor, _ = conversion(unit, "path.feed_m_s")
            self.builder.add("path.feed_m_s", t, None if value is None else value * factor, source=function)
        elif function == "cnc_rdsvmeter":
            value = _scaled(element)
            self.builder.add(f"axis.{_axis_name(element)}.load_frac", t, None if value is None else value * PCT,
                             source=function)
        elif function == "cnc_srvdelay":
            self._following_error(t, element, function)
        elif function == "cnc_rdseqnum":
            self.builder.add("program.sequence", t, _scaled(element), source=function)
        elif function == "cnc_rdblkcount":
            self.builder.add("program.block_count", t, _scaled(element), source=function)
        elif function == "cnc_rdprgnum":
            value = _scaled(element)
            channel = "program.main_name" if item.lower() in ("main", "mdata") else "program.name"
            text = element.get("text") or (f"O{int(value):04d}" if value is not None else None)
            self.builder.add(channel, t, text, source=function, text=True)
        elif function == "cnc_modal":
            letter = item.upper()
            if not re.fullmatch(r"[A-Z]", letter):
                raise InputError(f"cnc_modal: item must be the address letter (T, S, F, ...), not {item!r}")
            channel = "tool.number" if letter == "T" else f"modal.{letter.lower()}"
            self.builder.add(channel, t, _scaled(element), source=function)
        elif function == "cnc_statinfo":
            value = _scaled(element)
            if item.lower() == "run":
                self.builder.add("program.execution", t, RUN_STATES.get(int(value)) if value is not None else None,
                                 source=function, text=True)
            elif item.lower() == "aut":
                self.builder.add("controller.mode", t, AUT_MODES.get(int(value)) if value is not None else None,
                                 source=function, text=True)
            else:
                self._unmapped(f"{function}:{item}")
        elif function in ("cnc_rdalmmsg", "cnc_rdalmmsg2"):
            self._alarm(t, element)
        elif function == "cnc_diagnoss":
            self._diagnosis(t, element, item)
        else:
            self._unmapped(f"{function}:{item}" if item else function)

    def _unmapped(self, key: str) -> None:
        self.unmapped[key] = self.unmapped.get(key, 0) + 1

    def _following_error(self, t, element: dict, function: str) -> None:
        value = _scaled(element)
        name = _axis_name(element)
        unit = element.get("unit")
        if element.get("dec") in (None, "") and unit in (None, ""):
            raise InputError(f"{function} {name}: servo delay is in detection units; give dec or unit")
        factor, si = _length_factor(element, _rotary(name)) if unit not in (None, "") else ((DEG if _rotary(name) else MM),
                                                                                            "rad" if _rotary(name) else "m")
        self.builder.add(f"axis.{name}.following_error_{si}", t, None if value is None else value * factor,
                         source=function)

    def _diagnosis(self, t, element: dict, item: str) -> None:
        number = int(float(item or element.get("code") or 0))
        value = _scaled(element)
        if number == 300:
            self._following_error(t, element, "cnc_diagnoss:300")
        elif number in (308, 309):
            what = "motor" if number == 308 else "encoder"
            name = _axis_name(element).lower()
            self.builder.add(f"temp.axis_{name}_{what}_K", t, None if value is None else value + ZERO_DEGC_K,
                             source=f"cnc_diagnoss:{number}")
        elif number == 403:
            prefix = _spindle_prefix(element)
            self.builder.add(f"temp.{prefix}_motor_K", t, None if value is None else value + ZERO_DEGC_K,
                             source="cnc_diagnoss:403")
        else:
            axis = str(element.get("name") or element.get("index") or "").strip()
            suffix = f".{re.sub(r'[^A-Za-z0-9]', '', axis)}" if axis else ""
            self.builder.add(f"diag.d{number}{suffix}", t, value, source=f"cnc_diagnoss:{number}")

    # ------------------------------------------------------------------ alarms (poll snapshots)
    def _alarm(self, t, element: dict) -> None:
        key = (t,)
        snapshot = self.pending_alarms.setdefault(key, {})
        kind = str(element.get("type") if element.get("type") is not None else "").strip()
        if kind.lower() in ("poll", ""):
            if element.get("code") in (None, ""):
                return
        code = str(element.get("code") or "").strip()
        if not code:
            raise InputError("cnc_rdalmmsg: an alarm element needs code (alm_no)")
        prefix = ""
        if kind and kind.lower() != "poll":
            try:
                prefix = ALARM_TYPES.get(int(float(kind)), f"T{kind}")
            except ValueError:
                prefix = kind.upper()
        number = int(float(code))
        label = f"{prefix}{number:04d}" if prefix else str(number)
        axis = str(element.get("index") or element.get("name") or "").strip()
        snapshot[label] = (str(element.get("text") or "").strip(), axis)

    def close_poll(self, t) -> None:
        """An alarm poll at time t is complete: raise new codes, clear vanished ones."""
        snapshot = self.pending_alarms.pop((t,), None)
        if snapshot is None:
            return
        for code, (text, axis) in sorted(snapshot.items()):
            if code not in self.active_alarms:
                event = MachineEvent(0.0, "alarm", code, text, "fault", f"cnc_rdalmmsg{(' axis ' + axis) if axis else ''}")
                self.active_alarms[code] = event
                self.builder.event(t, event)
        for code in sorted(set(self.active_alarms) - set(snapshot)):
            old = self.active_alarms.pop(code)
            self.builder.event(t, MachineEvent(0.0, "alarm_cleared", code, old.text, "normal", "cnc_rdalmmsg"))
        self.builder.add("alarm.count", t, float(len(snapshot)), source="cnc_rdalmmsg")


def _elements(record: dict) -> list[dict]:
    data = record.get("data")
    if data is None:
        return [record]
    if not isinstance(data, list):
        raise InputError(f"{record.get('function')}: data must be a list of elements")
    base = {k: v for k, v in record.items() if k != "data"}
    if not data and str(base.get("function", "")).lower() in ("cnc_rdalmmsg", "cnc_rdalmmsg2"):
        return [{**base, "type": "poll"}]
    return [{**base, **element} for element in data]


def _records_from_text(text: str, where: str) -> list[dict]:
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            document = json.loads(text)
        except ValueError:
            records = []
            for number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except ValueError as error:
                        raise InputError(f"not JSON (or JSON Lines): {error}", where, number) from None
            return records
        if isinstance(document, dict):
            if document.get("format") not in (None, "cnctwin-focas-log"):
                raise InputError(f"format {document.get('format')!r} is not cnctwin-focas-log", where)
            document = document.get("records")
        if not isinstance(document, list):
            raise InputError("a FOCAS JSON log is a list of records (or {records: [...]})", where)
        return document
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return []
    dialect = ";" if lines[0].count(";") > lines[0].count(",") else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=dialect)
    header = [h.strip().lower() for h in (reader.fieldnames or [])]
    unknown = [h for h in header if h not in COLUMNS]
    if unknown or "time" not in header or "function" not in header:
        raise InputError(f"FOCAS CSV columns are {', '.join(COLUMNS)} (time and function required); "
                         f"unknown: {', '.join(unknown) or 'none'}", where)
    records = []
    for row in reader:
        records.append({k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k})
    return records


def read_focas(path, *, base_dir=None, mapping: dict | None = None, start_ns: int | None = None,
               end_ns: int | None = None) -> MachineLog:
    """A MachineLog from a FOCAS collector log (CSV, JSON or JSON Lines; a directory: every file in name order)."""
    path = Path(path)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    if not path.exists():
        raise InputError("no such file or directory", str(path))
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".csv", ".json", ".jsonl", ".txt")) \
        if path.is_dir() else [path]
    reader = _Reader(mapping, {"kind": "focas", "path": str(path), "files": [f.name for f in files]})
    limit = int(import_params()["max_observations"])
    for file in files:
        try:
            text = file.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as error:
            raise InputError(f"cannot read: {error}", str(file)) from None
        records = _records_from_text(text, str(file))
        last_alarm_time = None
        for number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise InputError("a record must be an object", str(file), number)
            try:
                t = reader.time(record.get("time"))
                if isinstance(t, int) and ((start_ns is not None and t < start_ns) or (end_ns is not None and t > end_ns)):
                    continue
                function = str(record.get("function") or "").lower()
                if last_alarm_time is not None and (function not in ("cnc_rdalmmsg", "cnc_rdalmmsg2")
                                                    or t != last_alarm_time):
                    reader.close_poll(last_alarm_time)
                    last_alarm_time = None
                for element in _elements(record):
                    reader.element(t, element)
                if function in ("cnc_rdalmmsg", "cnc_rdalmmsg2"):
                    last_alarm_time = t
            except InputError as error:
                if error.path:
                    raise
                raise InputError(str(error), str(file), number) from None
            if reader.builder.count > limit:
                raise InputError(f"more than {limit} observations (mdata-import max_observations): import a "
                                 f"time window", str(file))
        if last_alarm_time is not None:
            reader.close_poll(last_alarm_time)
    log = reader.builder.build(limit)
    log.meta["unmapped"] = dict(sorted(reader.unmapped.items()))
    return log


__all__ = ["read_focas", "COLUMNS", "ALARM_TYPES", "RUN_STATES", "AUT_MODES"]
