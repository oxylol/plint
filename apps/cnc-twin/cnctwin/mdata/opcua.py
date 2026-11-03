"""OPC UA: node histories from a server (asyncua) or from an offline export, into a MachineLog.

Server (``opc.tcp://host:4840``): the mapping names one node per channel,
``{<channel>: "<node id> [unit]"}`` (``"spindle.load_frac": "ns=2;s=Spindle1.Load [pct]"``); the
unit is optional when the node has an ``EngineeringUnits`` property (EUInformation, the unit
decoded from its UNECE code, opc-10000-8-2022 5.6.3). :func:`read_history` reads every node's raw
history over ``[start, end]`` (HistoryRead with ReadRawModifiedDetails; continuation points are
followed by asyncua, opc-10000-11-2022); :func:`read_current` reads the current value of each node
once; :func:`browse` lists the variables below a node (to find the node ids). Values whose status
is not Good are gaps (counted in ``log.meta["bad_status"]``). The time of a value is its
SourceTimestamp (the ServerTimestamp when the source gives none).

Offline export (what :func:`write_export` writes and :func:`read_export` reads):

* JSON ``{"format": "cnctwin-opcua-history", "version": 1, "server": url, "nodes": [{"channel",
  "node_id", "unit", "values": [[time, value, status], ...]}]}``, time ISO 8601, status the
  StatusCode name (``Good``) or number (0 is Good);
* CSV (long): columns ``node_id, time, value, status`` (``status`` optional, ``channel`` optional)
  with the mapping giving the channel and unit of each node id.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.errors import InputError
from .log import LogBuilder, MachineLog, conversion, format_time_ns, parse_time_ns
from .mtconnect import split_unit
from .params import import_params

FORMAT = "cnctwin-opcua-history"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# UNECE Recommendation 20 common codes (unece-rec20-2021) -> unit text this package converts.
UNECE_UNITS = {
    "MMT": "mm", "MTR": "m", "CMT": "cm", "INH": "in", "MMQ": "mm3", "CEL": "degC", "KEL": "K", "WTT": "W", "KWT": "kW",
    "P1": "pct", "C62": "-", "RPM": "rpm", "NEW": "N", "KN": "kN", "NU": "Nm", "MTS": "m_s", "C16": "mm_s",
    "2X": "m_min", "AMP": "A", "4K": "mA", "VLT": "V", "SEC": "s", "C26": "ms", "MIN": "min", "HUR": "h", "DD": "deg",
    "C81": "rad", "HTZ": "Hz", "KHZ": "kHz", "BAR": "bar", "PAL": "Pa", "KPA": "kPa", "MPA": "MPa", "JOU": "J",
}


@dataclass(frozen=True)
class NodeSpec:
    channel: str
    node_id: str
    unit: str | None = None


def node_specs(mapping: dict | None) -> list[NodeSpec]:
    """NodeSpecs of a mapping ``{channel: "node id [unit]"}`` (keys starting with ``opcua.`` are options)."""
    specs = []
    for channel, value in sorted((mapping or {}).items()):
        if channel.startswith("opcua.") or channel in ("note", "notes"):
            continue
        node_id, unit = split_unit(str(value))
        if not node_id:
            raise InputError(f"mapping {channel}: no node id")
        specs.append(NodeSpec(channel, node_id, unit))
    if not specs:
        raise InputError("an OPC UA source needs a mapping {<channel>: \"<node id> [unit]\"}")
    return specs


def decode_unece(unit_id: int) -> str:
    """The UNECE common code an EUInformation.unitId encodes (opc-10000-8-2022 5.6.3)."""
    code = []
    value = int(unit_id)
    while value > 0:
        code.append(chr(value & 0xFF))
        value >>= 8
    return "".join(reversed(code))


def encode_unece(code: str) -> int:
    value = 0
    for char in code:
        value = (value << 8) | ord(char)
    return value


def unit_of_unece(code: str, where: str) -> str:
    if code not in UNECE_UNITS:
        raise InputError(f"{where}: EngineeringUnits {code!r} is not a unit this importer converts "
                         f"(state the unit in the mapping: \"<node id> [unit]\")")
    return UNECE_UNITS[code]


def datetime_ns(moment: datetime, picoseconds: int | None = None) -> int:
    """Integer nanoseconds since the Unix epoch of a (UTC) datetime; naive datetimes are UTC
    (asyncua's convention). ``picoseconds`` is the OPC UA 10 ps count."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = moment - _EPOCH
    ns = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
    if picoseconds:
        ns += int(picoseconds) // 100
    return ns


def ns_datetime(ns: int) -> datetime:
    seconds, rest = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=rest // 1000)


def _value(variant_value):
    if isinstance(variant_value, bool):
        return float(variant_value)
    if isinstance(variant_value, (int, float)):
        return float(variant_value)
    if isinstance(variant_value, str):
        return variant_value
    if variant_value is None:
        return None
    raise InputError(f"OPC UA value {variant_value!r} of type {type(variant_value).__name__} is not a number or text")


def _add(builder: LogBuilder, spec: NodeSpec, unit: str | None, t_ns: int, value, good: bool, stats: dict) -> None:
    if not good:
        stats[spec.node_id] = stats.get(spec.node_id, 0) + 1
        value = None
    if isinstance(value, str):
        builder.add(spec.channel, t_ns, value, source=spec.node_id, text=True)
        return
    factor, offset = conversion(unit, spec.channel)
    builder.add(spec.channel, t_ns, None if value is None else value * factor + offset, source=spec.node_id)


# ----------------------------------------------------------------------------- server

async def _node_unit(node, spec: NodeSpec) -> str | None:
    if spec.unit is not None:
        return spec.unit
    from asyncua import ua

    try:
        child = await node.get_child("0:EngineeringUnits")
        info = await child.read_value()
    except (ua.UaError, ua.uaerrors.BadNoMatch, asyncio.TimeoutError, KeyError):
        return None
    if info is None:
        return None
    return unit_of_unece(decode_unece(info.UnitId), spec.node_id)


async def read_history_async(url: str, specs: list[NodeSpec], start_ns: int | None, end_ns: int | None, *,
                             timeout_s: float | None = None) -> MachineLog:
    """Raw history of each node over [start, end] (None: as far as the server holds)."""
    from asyncua import Client

    p = import_params()
    builder = LogBuilder({"kind": "opcua", "url": url, "nodes": {s.channel: s.node_id for s in specs},
                          "start": format_time_ns(start_ns) if start_ns is not None else None,
                          "end": format_time_ns(end_ns) if end_ns is not None else None})
    stats: dict[str, int] = {}
    limit = int(p["max_observations"])
    start = ns_datetime(start_ns) if start_ns is not None else None
    end = ns_datetime(end_ns) if end_ns is not None else None
    try:
        async with Client(url, timeout=float(timeout_s if timeout_s is not None else p["opcua_timeout_s"])) as client:
            for spec in specs:
                node = client.get_node(spec.node_id)
                unit = await _node_unit(node, spec)
                values = await node.read_raw_history(start, end, numvalues=0, return_bounds=False)
                for dv in values:
                    stamp = dv.SourceTimestamp or dv.ServerTimestamp
                    if stamp is None:
                        raise InputError(f"OPC UA node {spec.node_id}: a history value has no timestamp")
                    pico = dv.SourcePicoseconds if dv.SourceTimestamp is not None else dv.ServerPicoseconds
                    good = dv.StatusCode is None or dv.StatusCode.is_good()
                    _add(builder, spec, unit, datetime_ns(stamp, pico), _value(dv.Value.Value if dv.Value else None),
                         good, stats)
                if builder.count > limit:
                    raise InputError(f"more than {limit} observations (mdata-import max_observations): read a "
                                     f"shorter time window")
    except (OSError, asyncio.TimeoutError) as error:
        raise InputError(f"OPC UA server {url}: {type(error).__name__}: {error}") from None
    log = builder.build(limit)
    log.meta["bad_status"] = dict(sorted(stats.items()))
    return log


async def read_current_async(url: str, specs: list[NodeSpec], *, timeout_s: float | None = None) -> MachineLog:
    """The current value of each node (one observation per channel)."""
    from asyncua import Client

    p = import_params()
    builder = LogBuilder({"kind": "opcua", "url": url, "nodes": {s.channel: s.node_id for s in specs}, "read": "current"})
    stats: dict[str, int] = {}
    try:
        async with Client(url, timeout=float(timeout_s if timeout_s is not None else p["opcua_timeout_s"])) as client:
            for spec in specs:
                node = client.get_node(spec.node_id)
                unit = await _node_unit(node, spec)
                dv = await node.read_data_value()
                stamp = dv.SourceTimestamp or dv.ServerTimestamp
                if stamp is None:
                    raise InputError(f"OPC UA node {spec.node_id}: the value has no timestamp")
                good = dv.StatusCode is None or dv.StatusCode.is_good()
                _add(builder, spec, unit, datetime_ns(stamp), _value(dv.Value.Value if dv.Value else None), good, stats)
    except (OSError, asyncio.TimeoutError) as error:
        raise InputError(f"OPC UA server {url}: {type(error).__name__}: {error}") from None
    log = builder.build()
    log.meta["bad_status"] = dict(sorted(stats.items()))
    return log


async def browse_async(url: str, root: str = "i=85", *, max_depth: int | None = None, max_nodes: int | None = None,
                       timeout_s: float | None = None) -> list[dict]:
    """Variables below ``root`` (default the Objects folder): node id, browse path, data type, value and
    engineering unit, depth first in browse order."""
    from asyncua import Client, ua

    p = import_params()
    depth_limit = int(max_depth if max_depth is not None else p["opcua_browse_max_depth"])
    node_limit = int(max_nodes if max_nodes is not None else p["opcua_browse_max_nodes"])
    found: list[dict] = []
    try:
        async with Client(url, timeout=float(timeout_s if timeout_s is not None else p["opcua_timeout_s"])) as client:
            stack = [(client.get_node(root), [], 0)]
            while stack and len(found) < node_limit:
                node, path, depth = stack.pop()
                children = await node.get_children()
                entries = []
                for child in children:
                    name = (await child.read_browse_name()).to_string()
                    entries.append((child, name))
                for child, name in reversed(entries):
                    node_class = await child.read_node_class()
                    here = path + [name]
                    if node_class == ua.NodeClass.Variable:
                        if name == "0:EngineeringUnits":
                            continue
                        record = {"node_id": child.nodeid.to_string(), "path": "/".join(here)}
                        try:
                            dv = await child.read_data_value()
                            value = dv.Value.Value if dv.Value else None
                            record["value"] = value if isinstance(value, (int, float, str, bool)) or value is None \
                                else str(value)
                            record["data_type"] = dv.Value.VariantType.name if dv.Value else None
                        except ua.UaError as error:
                            record["error"] = str(error)
                        try:
                            eu = await (await child.get_child("0:EngineeringUnits")).read_value()
                            record["unece"] = decode_unece(eu.UnitId)
                            record["unit"] = UNECE_UNITS.get(record["unece"])
                        except (ua.UaError, ua.uaerrors.BadNoMatch, KeyError):
                            pass
                        found.append(record)
                        if len(found) >= node_limit:
                            break
                    if node_class in (ua.NodeClass.Object, ua.NodeClass.Variable) and depth + 1 < depth_limit:
                        stack.append((child, here, depth + 1))
    except (OSError, asyncio.TimeoutError) as error:
        raise InputError(f"OPC UA server {url}: {type(error).__name__}: {error}") from None
    return found


def _run(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise InputError("the synchronous OPC UA functions cannot run inside an event loop: await the *_async ones")


def read_history(url: str, specs: list[NodeSpec], start_ns: int | None = None, end_ns: int | None = None,
                 **kwargs) -> MachineLog:
    return _run(read_history_async(url, specs, start_ns, end_ns, **kwargs))


def read_current(url: str, specs: list[NodeSpec], **kwargs) -> MachineLog:
    return _run(read_current_async(url, specs, **kwargs))


def browse(url: str, root: str = "i=85", **kwargs) -> list[dict]:
    return _run(browse_async(url, root, **kwargs))


# ----------------------------------------------------------------------------- offline export

def write_export(path, log: MachineLog, specs: list[NodeSpec], server: str = "") -> Path:
    """The node histories of a log read from a server, as the offline JSON export (values in SI, unit '-'
    per channel unit; deterministic)."""
    if log.epoch_ns is None:
        raise InputError("an OPC UA export needs absolute times")
    nodes = []
    for spec in specs:
        if spec.channel not in log.channels:
            continue
        channel = log.channels[spec.channel]
        values = []
        for t, v in zip(channel.t_s, channel.values):
            t_ns = log.epoch_ns + int(round(float(t) * 1e9))
            if channel.numeric:
                values.append([format_time_ns(t_ns), None if not math.isfinite(v) else float(v),
                               "Good" if math.isfinite(v) else "Bad"])
            else:
                values.append([format_time_ns(t_ns), v, "Good" if v is not None else "Bad"])
        nodes.append({"channel": spec.channel, "node_id": spec.node_id, "unit": None, "values": values})
    document = {"format": FORMAT, "version": 1, "server": server, "nodes": nodes}
    path = Path(path)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _good(status) -> bool:
    if status is None or status == "":
        return True
    if isinstance(status, (int, float)):
        return int(status) & 0xC0000000 == 0
    text = str(status).strip()
    try:
        return int(text, 0) & 0xC0000000 == 0
    except ValueError:
        return text.lower().startswith("good")


def read_export(path, *, base_dir=None, mapping: dict | None = None, start_ns: int | None = None,
                end_ns: int | None = None) -> MachineLog:
    """A MachineLog from an OPC UA history export (JSON, or long CSV with the mapping)."""
    path = Path(path)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise InputError(f"cannot read: {error.strerror}", str(path)) from None
    by_node = {s.node_id: s for s in node_specs(mapping)} if mapping else {}
    builder = LogBuilder({"kind": "opcua", "path": str(path)})
    stats: dict[str, int] = {}
    zone_missing = False

    def take(spec: NodeSpec, stamp, value, status) -> None:
        nonlocal zone_missing
        t_ns, zone = parse_time_ns(str(stamp))
        zone_missing |= not zone
        if (start_ns is not None and t_ns < start_ns) or (end_ns is not None and t_ns > end_ns):
            return
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        _add(builder, spec, spec.unit, t_ns, value if value != "" else None, _good(status), stats)

    if text.lstrip().startswith("{"):
        try:
            document = json.loads(text)
        except ValueError as error:
            raise InputError(f"not JSON: {error}", str(path)) from None
        if document.get("format") != FORMAT:
            raise InputError(f"format {document.get('format')!r} is not {FORMAT}", str(path))
        builder.source["server"] = document.get("server")
        for node in document.get("nodes", []):
            node_id = node.get("node_id")
            spec = by_node.get(node_id) or (NodeSpec(node["channel"], node_id, node.get("unit"))
                                            if node.get("channel") else None)
            if spec is None:
                raise InputError(f"node {node_id}: no channel (give it in the export or in the mapping)", str(path))
            for row in node.get("values", []):
                if not isinstance(row, list) or len(row) not in (2, 3):
                    raise InputError(f"node {node_id}: a value row is [time, value, status]", str(path))
                take(spec, row[0], row[1], row[2] if len(row) == 3 else None)
    else:
        rows = list(csv.DictReader(io.StringIO("\n".join(l for l in text.splitlines()
                                                        if l.strip() and not l.lstrip().startswith("#")))))
        for number, row in enumerate(rows, start=2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            node_id = row.get("node_id")
            if not node_id or "time" not in row or "value" not in row:
                raise InputError("OPC UA CSV columns are node_id, time, value[, status][, channel]", str(path), number)
            spec = by_node.get(node_id) or (NodeSpec(row["channel"], node_id, None) if row.get("channel") else None)
            if spec is None:
                raise InputError(f"node {node_id}: no channel (give it in the mapping)", str(path), number)
            try:
                take(spec, row["time"], row["value"], row.get("status"))
            except InputError as error:
                raise InputError(str(error), str(path), number) from None
    if zone_missing:
        builder.meta["timezone_assumed"] = "UTC"
    log = builder.build(int(import_params()["max_observations"]))
    log.meta["bad_status"] = dict(sorted(stats.items()))
    return log


def read_opcua(source: str, *, base_dir=None, mapping: dict | None = None, start_ns: int | None = None,
               end_ns: int | None = None) -> MachineLog:
    """A server URL (``opc.tcp://``: raw history of the mapped nodes) or an export file."""
    if str(source).startswith("opc.tcp://"):
        return read_history(str(source), node_specs(mapping), start_ns, end_ns)
    return read_export(source, base_dir=base_dir, mapping=mapping, start_ns=start_ns, end_ns=end_ns)


__all__ = ["read_opcua", "read_history", "read_current", "browse", "read_history_async", "read_current_async",
           "browse_async", "read_export", "write_export", "NodeSpec", "node_specs", "decode_unece", "encode_unece",
           "UNECE_UNITS", "datetime_ns"]
