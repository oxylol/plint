"""MTConnect: MTConnectStreams documents from files or an agent, into a MachineLog.

Sources (docs/models/calibration.md "MTConnect"):

* **files**: one MTConnectStreams document (a ``/current`` or ``/sample`` response), several
  concatenated in one file (a recorder appending responses), or a directory of ``*.xml`` files
  (name order). Observations are de-duplicated by ``(instanceId, sequence)`` and ordered by time.
* **an agent** (``http://host:5000[/device]``): :class:`AgentClient` reads ``/current`` (buffer
  state: ``instanceId``, ``firstSequence``, ``lastSequence``, ``nextSequence``) and then chains
  ``/sample?from=<n>&count=<c>`` requests by each response's ``nextSequence`` until it passes the
  ``lastSequence`` seen at the start (or ``until``). A ``from`` that has fallen out of the agent's
  buffer (``OUT_OF_RANGE``, or a response whose ``firstSequence`` is past ``from``) is a recorded
  **gap** and the collection continues at the agent's ``firstSequence``; an agent restart (a new
  ``instanceId``) is recorded and the collection continues from the new buffer's start
  (ansi-mtc1-4-2018 Part 1, protocol).

Mapping (MTConnect Part 3 types to channels; values in the standard units the agent reports):

=====================================  ======================================  ==============
component / element (subType)          channel                                 unit in stream
=====================================  ======================================  ==============
Linear ``Position`` (ACTUAL, none)     ``axis.<name>.pos_machine_m``           MILLIMETER
Linear ``Position`` (COMMANDED)        ``axis.<name>.pos_cmd_m``               MILLIMETER
Linear ``Load`` / ``AxisFeedrate``     ``axis.<name>.load_frac`` / ``feed_m_s``  PERCENT, MM/S
Rotary ``Angle`` (ACTUAL)              ``axis.<name>.pos_machine_rad``         DEGREE
Rotary (spindle) ``RotaryVelocity``    ``spindle.speed_rad_s`` (COMMANDED,     REVOLUTION/MINUTE
  / ``SpindleSpeed``                     PROGRAMMED: ``spindle.speed_cmd_rad_s``)
Rotary (spindle) ``Load``              ``spindle.load_frac``                   PERCENT
Rotary (spindle) ``Wattage``,          ``spindle.power_W``, ``spindle.torque_Nm``,  WATT, NEWTON_METER,
  ``Torque``, ``Amperage``               ``spindle.current_A``                 AMPERE
Path ``PathFeedrate`` (ACTUAL)         ``path.feed_m_s`` (PROGRAMMED/COMMANDED: ``path.feed_cmd_m_s``)  MM/S
Path ``PathPosition``                  ``path.x_work_m``, ``.y_``, ``.z_``     MILLIMETER_3D
Path ``Block`` / ``Line``,             ``program.block`` / ``program.line`` /  text / number
  ``LineNumber`` / ``LineLabel``         ``program.sequence``
Path ``Program``, ``ToolNumber``,      ``program.name``, ``tool.number``,      text, number
  ``Execution``, ``ControllerMode``      ``program.execution``, ``controller.mode``
``PathFeedrateOverride`` (RAPID: rapid) ``path.feed_override_frac`` / ``path.rapid_override_frac``  PERCENT
``RotaryVelocityOverride``             ``spindle.override_frac``               PERCENT
``Temperature`` of any component       ``temp.<component>_K``                  CELSIUS
``Condition`` Normal/Warning/Fault     MachineEvent (kind condition, code = nativeCode or type)
=====================================  ======================================  ==============

A Rotary component is a spindle when it reports ``RotaryVelocity``/``SpindleSpeed`` or
``RotaryMode`` SPINDLE (the first one is ``spindle``, further ones ``spindle2`` ...; ``mapping:
{spindle: <component name>}`` chooses the main one); the first Path is ``path``/``program``/
``tool``, further ones get a number. ``UNAVAILABLE`` is a gap (NaN / None). Time series
(``...TimeSeries``) are expanded with ``sampleRate``, the timestamp being that of the last sample
(mtconnect-2-2-2023). ``mapping: {<dataItemId or name>: "<channel> [unit]"}`` maps any other data
item; items neither mapped nor in the table are counted in ``log.meta["unmapped"]``.
"""

from __future__ import annotations

import io
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import InputError
from ..units import ZERO_DEGC_K, rpm_to_rad_s
from .log import LogBuilder, MachineEvent, MachineLog, conversion, parse_time_ns
from .params import import_params

MM = 1e-3                                     # MILLIMETER -> m
DEG = math.pi / 180.0                         # DEGREE -> rad
PCT = 1e-2                                    # PERCENT -> fraction
RPM = rpm_to_rad_s(1.0)                       # REVOLUTION/MINUTE -> rad/s

# element -> (unit factor, offset) of the standard unit
_SPINDLE = {"RotaryVelocity": "speed", "SpindleSpeed": "speed", "Load": "load_frac", "Wattage": "power_W",
            "Torque": "torque_Nm", "Amperage": "current_A"}


@dataclass
class Observation:
    instance: str
    sequence: int
    time_ns: int
    zone: bool
    device: str
    component: str              # component type (Linear, Rotary, Path, Controller ...)
    component_name: str
    category: str               # SAMPLE | EVENT | CONDITION
    element: str                # type element name (Position, LoadTimeSeries, Fault ...)
    data_item: str
    name: str
    sub_type: str
    text: str
    attrs: dict = field(default_factory=dict)


@dataclass
class Header:
    instance: str = ""
    first: int = 0
    last: int = 0
    next: int = 0
    creation: str = ""
    version: str = ""


@dataclass
class StreamsDoc:
    header: Header
    observations: list[Observation]
    errors: list[tuple[str, str]] = field(default_factory=list)   # (errorCode, text) of an MTConnectError


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _guard(data: bytes, where: str) -> None:
    head = data[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in data.upper():
        raise InputError("MTConnect documents carry no DTD or entities; this file declares one (refused)", where)


def split_documents(data: bytes) -> list[bytes]:
    """The XML documents of a file that may hold several responses one after the other."""
    starts = [m.start() for m in re.finditer(rb"<\?xml", data)]
    if len(starts) <= 1:
        return [data] if data.strip() else []
    starts.append(len(data))
    return [data[a:b] for a, b in zip(starts[:-1], starts[1:]) if data[a:b].strip()]


def parse_streams(data: bytes, where: str = "") -> StreamsDoc:
    """One MTConnectStreams (or MTConnectError) document."""
    _guard(data, where)
    header = Header()
    observations: list[Observation] = []
    errors: list[tuple[str, str]] = []
    stack: list[str] = []
    device = component = component_name = category = ""
    try:
        for event, element in ET.iterparse(io.BytesIO(data), events=("start", "end")):
            tag = _local(element.tag)
            if event == "start":
                stack.append(tag)
                if tag == "DeviceStream":
                    device = element.get("name") or element.get("uuid") or ""
                elif tag == "ComponentStream":
                    component = element.get("component") or ""
                    component_name = element.get("name") or element.get("componentId") or component
                elif tag in ("Samples", "Events", "Condition") and len(stack) >= 2 and stack[-2] == "ComponentStream":
                    category = {"Samples": "SAMPLE", "Events": "EVENT", "Condition": "CONDITION"}[tag]
                continue
            stack.pop()
            if tag == "Header":
                header = Header(element.get("instanceId", ""), _int(element.get("firstSequence")),
                                _int(element.get("lastSequence")), _int(element.get("nextSequence")),
                                element.get("creationTime", ""), element.get("version", ""))
            elif tag == "Error":
                errors.append((element.get("errorCode", ""), (element.text or "").strip()))
            elif stack and stack[-1] in ("Samples", "Events", "Condition"):
                timestamp = element.get("timestamp")
                if timestamp is None:
                    raise InputError(f"{tag} {element.get('dataItemId', '')}: no timestamp", where)
                time_ns, zone = parse_time_ns(timestamp)
                attrs = {k: v for k, v in element.attrib.items()
                         if k not in ("timestamp", "sequence", "dataItemId", "name", "subType")}
                observations.append(Observation(header.instance, _int(element.get("sequence")), time_ns, zone,
                                                device, component, component_name, category, tag,
                                                element.get("dataItemId", ""), element.get("name", ""),
                                                element.get("subType", ""), (element.text or "").strip(), attrs))
                element.clear()
            elif tag in ("ComponentStream", "DeviceStream"):
                element.clear()
    except ET.ParseError as error:
        raise InputError(f"not a well-formed MTConnect document: {error}", where) from None
    for obs in observations:
        obs.instance = header.instance
    return StreamsDoc(header, observations, errors)


def _int(text) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


# ----------------------------------------------------------------------------- mapping

def _parse_mapping(mapping: dict | None) -> tuple[dict, str | None]:
    """{item id or name: (channel, unit)} and the chosen main spindle component."""
    items, spindle = {}, None
    for key, value in (mapping or {}).items():
        if key == "spindle":
            spindle = str(value)
            continue
        if key.startswith("mtconnect.") or key in ("note", "notes"):
            continue
        channel, unit = split_unit(str(value))
        items[str(key)] = (channel, unit)
    return items, spindle


def split_unit(text: str) -> tuple[str, str | None]:
    """``"name [unit]"`` -> (name, unit); without brackets the unit is None."""
    match = re.match(r"^\s*(.*?)\s*(?:\[\s*([^\]]*?)\s*\])?\s*$", text)
    name, unit = match.group(1), match.group(2)
    return name, (unit if unit else None)


class _Mapper:
    def __init__(self, observations: list[Observation], mapping: dict | None) -> None:
        self.items, main = _parse_mapping(mapping)
        rotary = {}
        for obs in observations:
            if obs.component.lower() != "rotary":
                continue
            info = rotary.setdefault(obs.component_name, set())
            if obs.element in ("RotaryVelocity", "SpindleSpeed", "RotaryVelocityTimeSeries"):
                info.add("velocity")
            if obs.element == "RotaryMode" and obs.text.upper() == "SPINDLE":
                info.add("velocity")
        spindles = [name for name, info in rotary.items() if "velocity" in info]
        if main is not None:
            if main not in spindles:
                raise InputError(f"mapping spindle: {main!r} is not a spindle component of these documents "
                                 f"(spindles: {', '.join(spindles) or 'none'})")
            spindles.remove(main)
            spindles.insert(0, main)
        self.spindle_prefix = {name: ("spindle" if i == 0 else f"spindle{i + 1}") for i, name in enumerate(spindles)}
        paths = []
        for obs in observations:
            if obs.component.lower() == "path" and obs.component_name not in paths:
                paths.append(obs.component_name)
        self.path_suffix = {name: ("" if i == 0 else str(i + 1)) for i, name in enumerate(paths)}
        self.unmapped: dict[str, int] = {}

    def channels(self, obs: Observation) -> list[tuple[str, float, float, str]]:
        """[(channel, factor, offset, kind)] for one observation (several for PathPosition)."""
        for key in (obs.data_item, obs.name):
            if key and key in self.items:
                channel, unit = self.items[key]
                factor, offset = conversion(unit, channel)
                return [(channel, factor, offset, "")]
        element = obs.element[: -len("TimeSeries")] if obs.element.endswith("TimeSeries") else obs.element
        comp = obs.component.lower()
        sub = obs.sub_type.upper()
        name = _safe(obs.component_name)
        if element == "Temperature":
            where = self.spindle_prefix.get(obs.component_name) if comp == "rotary" else None
            if where is None:
                where = (f"axis_{name.lower()}" if comp in ("linear", "rotary") else name.lower())
            return [(f"temp.{where}_K", 1.0, ZERO_DEGC_K, "sample")]
        if comp == "linear":
            if element == "Position" and sub in ("", "ACTUAL"):
                return [(f"axis.{name}.pos_machine_m", MM, 0.0, "sample")]
            if element == "Position" and sub == "COMMANDED":
                return [(f"axis.{name}.pos_cmd_m", MM, 0.0, "sample")]
            if element == "Load":
                return [(f"axis.{name}.load_frac", PCT, 0.0, "sample")]
            if element == "AxisFeedrate" and sub in ("", "ACTUAL"):
                return [(f"axis.{name}.feed_m_s", MM, 0.0, "sample")]
        if comp == "rotary":
            prefix = self.spindle_prefix.get(obs.component_name)
            if prefix is not None and element in _SPINDLE:
                what = _SPINDLE[element]
                if what == "speed":
                    if sub in ("", "ACTUAL"):
                        return [(f"{prefix}.speed_rad_s", RPM, 0.0, "sample")]
                    if sub in ("COMMANDED", "PROGRAMMED"):
                        return [(f"{prefix}.speed_cmd_rad_s", RPM, 0.0, "event")]
                else:
                    factor = PCT if what == "load_frac" else 1.0
                    return [(f"{prefix}.{what}", factor, 0.0, "sample")]
            if element == "Angle" and sub in ("", "ACTUAL"):
                return [(f"axis.{name}.pos_machine_rad", DEG, 0.0, "sample")]
            if element == "Load" and prefix is None:
                return [(f"axis.{name}.load_frac", PCT, 0.0, "sample")]
            if element == "RotaryVelocityOverride" and prefix is not None:
                return [(f"{prefix}.override_frac", PCT, 0.0, "event")]
        suffix = self.path_suffix.get(obs.component_name, "") if comp == "path" else ""
        if comp in ("path", "controller", "device"):
            path, program, tool = f"path{suffix}", f"program{suffix}", f"tool{suffix}"
            if element == "PathFeedrate":
                if sub in ("", "ACTUAL"):
                    return [(f"{path}.feed_m_s", MM, 0.0, "sample")]
                if sub in ("PROGRAMMED", "COMMANDED"):
                    return [(f"{path}.feed_cmd_m_s", MM, 0.0, "event")]
            if element == "PathPosition" and sub in ("", "ACTUAL"):
                return [(f"{path}.{c}_work_m", MM, 0.0, "sample") for c in "xyz"]
            if element in ("Line", "LineNumber"):
                return [(f"{program}.line", 1.0, 0.0, "event")]
            if element == "LineLabel":
                return [(f"{program}.sequence", 1.0, 0.0, "event")]
            if element == "Block":
                return [(f"{program}.block", 1.0, 0.0, "event")]
            if element == "Program":
                return [(f"{program}.name", 1.0, 0.0, "event")]
            if element in ("ToolNumber", "ToolAssetId"):
                return [(f"{tool}.number", 1.0, 0.0, "event")]
            if element == "Execution":
                return [(f"{program}.execution", 1.0, 0.0, "event")]
            if element == "PartCount":
                return [(f"{program}.part_count", 1.0, 0.0, "event")]
            if element == "ControllerMode":
                return [("controller.mode", 1.0, 0.0, "event")]
            if element == "EmergencyStop":
                return [("controller.estop", 1.0, 0.0, "event")]
            if element == "PathFeedrateOverride":
                target = "rapid_override_frac" if sub == "RAPID" else "feed_override_frac"
                return [(f"{path}.{target}", PCT, 0.0, "event")]
            if element == "RotaryVelocityOverride":
                return [("spindle.override_frac", PCT, 0.0, "event")]
        key = f"{obs.component or '?'}/{obs.element}" + (f"[{obs.sub_type}]" if obs.sub_type else "")
        self.unmapped[key] = self.unmapped.get(key, 0) + 1
        return []


_TEXT_CHANNELS = (".block", ".name", ".execution", "controller.mode", "controller.estop")


def _safe(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", name or "")
    return cleaned if cleaned and cleaned[0].isalpha() else f"c{cleaned}"


def _number(text: str) -> float | None:
    if text.upper() == "UNAVAILABLE" or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        digits = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(digits.group(0)) if digits else None


def observations_to_log(observations: list[Observation], *, mapping: dict | None = None, source: dict | None = None,
                        gaps: list | None = None, start_ns: int | None = None, end_ns: int | None = None) -> MachineLog:
    """Map de-duplicated observations to a MachineLog (see the module table)."""
    unique: dict[tuple, Observation] = {}
    for obs in observations:
        key = (obs.instance, obs.sequence, obs.data_item, obs.element) if obs.sequence else \
            (obs.instance, obs.time_ns, obs.data_item, obs.element, obs.text)
        unique.setdefault(key, obs)
    ordered = sorted(unique.values(), key=lambda o: (o.time_ns, o.instance, o.sequence))
    mapper = _Mapper(ordered, mapping)           # component roles from every observation, before any window
    if start_ns is not None or end_ns is not None:
        ordered = [o for o in ordered if (start_ns is None or o.time_ns >= start_ns)
                   and (end_ns is None or o.time_ns <= end_ns)]
    builder = LogBuilder(source)
    if ordered and not all(o.zone for o in ordered):
        builder.meta["timezone_assumed"] = "UTC"
    limit = int(import_params()["max_observations"])
    for obs in ordered:
        if obs.category == "CONDITION":
            severity = obs.element.lower()
            builder.event(obs.time_ns, MachineEvent(0.0, "condition", obs.attrs.get("nativeCode") or obs.attrs.get("type", ""),
                                                    obs.text, severity, f"{obs.component_name}/{obs.data_item}"))
            continue
        if obs.element == "Message":
            builder.event(obs.time_ns, MachineEvent(0.0, "message", obs.attrs.get("nativeCode", ""), obs.text, "info",
                                                    f"{obs.component_name}/{obs.data_item}"))
            continue
        targets = mapper.channels(obs)
        if not targets:
            continue
        if obs.element.endswith("TimeSeries"):
            _time_series(builder, obs, targets)
            continue
        if builder.count >= limit:
            raise InputError(f"the MTConnect data holds more than {limit} observations (mdata-import "
                             f"max_observations): import a time window")
        if targets[0][0].endswith(_TEXT_CHANNELS):
            value = None if obs.text.upper() == "UNAVAILABLE" else obs.text
            builder.add(targets[0][0], obs.time_ns, value, kind=targets[0][3] or None, source=obs.data_item or obs.name,
                        text=True)
            continue
        if len(targets) == 3 and obs.element == "PathPosition":
            parts = obs.text.split()
            numbers = [None] * 3 if obs.text.upper() == "UNAVAILABLE" or len(parts) != 3 else [_number(p) for p in parts]
            for (channel, factor, offset, kind), value in zip(targets, numbers):
                builder.add(channel, obs.time_ns, None if value is None else value * factor + offset,
                            kind=kind or None, source=obs.data_item or obs.name)
            continue
        channel, factor, offset, kind = targets[0]
        value = _number(obs.text)
        builder.add(channel, obs.time_ns, None if value is None else value * factor + offset, kind=kind or None,
                    source=obs.data_item or obs.name)
    log = builder.build(limit)
    log.meta["unmapped"] = dict(sorted(mapper.unmapped.items()))
    log.meta["gaps"] = list(gaps or [])
    log.meta["instances"] = sorted({o.instance for o in ordered if o.instance})
    return log


def _time_series(builder: LogBuilder, obs: Observation, targets) -> None:
    channel, factor, offset, kind = targets[0]
    values = [] if obs.text.upper() == "UNAVAILABLE" else [_number(v) for v in obs.text.split()]
    if not values:
        builder.add(channel, obs.time_ns, None, kind=kind or None, source=obs.data_item)
        return
    rate = obs.attrs.get("sampleRate")
    if rate is None or float(rate) <= 0:
        raise InputError(f"time series {obs.data_item or obs.element} has no sampleRate (cannot time its samples)")
    step_ns = 1e9 / float(rate)
    n = len(values)
    for i, value in enumerate(values):
        t = obs.time_ns - int(round((n - 1 - i) * step_ns))
        builder.add(channel, t, None if value is None else value * factor + offset, kind=kind or None,
                    source=obs.data_item)


# ----------------------------------------------------------------------------- files

def read_files(path, *, mapping: dict | None = None, start_ns: int | None = None, end_ns: int | None = None) -> MachineLog:
    """A MachineLog from an MTConnectStreams file, a file of several documents, or a directory of them."""
    path = Path(path)
    files = sorted(path.glob("*.xml")) if path.is_dir() else [path]
    if not files:
        raise InputError("no *.xml files", str(path))
    observations: list[Observation] = []
    for file in files:
        try:
            data = file.read_bytes()
        except OSError as error:
            raise InputError(f"cannot read: {error.strerror}", str(file)) from None
        for chunk in split_documents(data):
            doc = parse_streams(chunk, str(file))
            if doc.errors:
                raise InputError(f"the file holds an MTConnectError: {'; '.join(f'{c}: {t}' for c, t in doc.errors)}",
                                 str(file))
            observations.extend(doc.observations)
    return observations_to_log(observations, mapping=mapping, source={"kind": "mtconnect", "path": str(path),
                                                                      "files": [f.name for f in files]},
                               start_ns=start_ns, end_ns=end_ns)


# ----------------------------------------------------------------------------- agent

class AgentClient:
    """HTTP client of an MTConnect agent (``base_url`` is the agent, with the device path if any)."""

    def __init__(self, base_url: str, *, count: int | None = None, timeout_s: float | None = None,
                 transport=None) -> None:
        import httpx

        p = import_params()
        self.base = base_url.rstrip("/")
        self.count = int(count if count is not None else p["mtconnect_count"])
        self.max_requests = int(p["mtconnect_max_requests"])
        self.client = httpx.Client(timeout=float(timeout_s if timeout_s is not None else p["http_timeout_s"]),
                                   transport=transport)
        self.requests = 0

    def close(self) -> None:
        self.client.close()

    def _get(self, endpoint: str, params: dict | None = None) -> StreamsDoc:
        import httpx

        url = f"{self.base}/{endpoint}"
        try:
            response = self.client.get(url, params=params)
        except httpx.HTTPError as error:
            raise InputError(f"MTConnect agent {url}: {error}") from None
        self.requests += 1
        if response.status_code not in (200, 400, 404):
            raise InputError(f"MTConnect agent {url}: HTTP {response.status_code}")
        doc = parse_streams(response.content, url)
        if response.status_code != 200 and not doc.errors:
            raise InputError(f"MTConnect agent {url}: HTTP {response.status_code}")
        return doc

    def current(self) -> StreamsDoc:
        doc = self._get("current")
        if doc.errors:
            raise InputError(f"MTConnect agent {self.base}/current: {doc.errors[0][0]}: {doc.errors[0][1]}")
        return doc

    def sample(self, start: int, count: int | None = None) -> StreamsDoc:
        return self._get("sample", {"from": int(start), "count": int(count or self.count)})

    def collect(self, *, start: int | None = None, until: int | None = None) -> tuple[list[Observation], list[dict], dict]:
        """Every observation from ``start`` (default: the oldest in the buffer) to ``until`` (default:
        the last sequence at the time of the call). Returns (observations, gaps, header info)."""
        current = self.current()
        instance = current.header.instance
        position = current.header.first if start is None else int(start)
        stop = current.header.last if until is None else int(until)
        observations: list[Observation] = []
        gaps: list[dict] = []
        while position <= stop:
            if self.requests > self.max_requests:
                raise InputError(f"MTConnect agent {self.base}: more than {self.max_requests} requests "
                                 f"(mdata-import mtconnect_max_requests) without reaching sequence {stop}")
            doc = self.sample(position)
            if doc.errors:
                codes = {code for code, _ in doc.errors}
                if "OUT_OF_RANGE" in codes and doc.header.first > position:
                    gaps.append({"from": position, "to": doc.header.first - 1, "reason": "out of the agent buffer"})
                    position = doc.header.first
                    continue
                raise InputError(f"MTConnect agent {self.base}/sample?from={position}: "
                                 f"{'; '.join(f'{c}: {t}' for c, t in doc.errors)}")
            if doc.header.instance != instance:
                gaps.append({"from": position, "to": None, "reason": f"agent restarted (instance {instance} -> "
                                                                     f"{doc.header.instance})"})
                instance = doc.header.instance
                position = doc.header.first
                stop = doc.header.last if until is None else stop
                continue
            if doc.header.first > position:
                gaps.append({"from": position, "to": doc.header.first - 1, "reason": "out of the agent buffer"})
            observations.extend(doc.observations)
            if doc.header.next <= position:
                break
            position = doc.header.next
        return observations, gaps, {"instance": instance, "first": current.header.first, "last": stop,
                                    "requests": self.requests}


def read_agent(url: str, *, mapping: dict | None = None, start_ns: int | None = None, end_ns: int | None = None,
               start_sequence: int | None = None, until_sequence: int | None = None, transport=None) -> MachineLog:
    """A MachineLog of everything an agent's buffer holds (optionally a sequence or time window)."""
    client = AgentClient(url, transport=transport)
    try:
        observations, gaps, info = client.collect(start=start_sequence, until=until_sequence)
    finally:
        client.close()
    log = observations_to_log(observations, mapping=mapping, source={"kind": "mtconnect", "url": url, **info},
                              gaps=gaps, start_ns=start_ns, end_ns=end_ns)
    return log


def read_mtconnect(source: str, *, base_dir=None, mapping: dict | None = None, start_ns: int | None = None,
                   end_ns: int | None = None) -> MachineLog:
    """Files (path relative to ``base_dir``) or an agent URL (``http://`` / ``https://``)."""
    if re.match(r"^https?://", str(source)):
        return read_agent(str(source), mapping=mapping, start_ns=start_ns, end_ns=end_ns)
    path = Path(source)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    if not path.exists():
        raise InputError("no such file or directory", str(path))
    return read_files(path, mapping=mapping, start_ns=start_ns, end_ns=end_ns)


__all__ = ["read_mtconnect", "read_files", "read_agent", "AgentClient", "parse_streams", "split_documents",
           "observations_to_log", "Observation", "StreamsDoc", "Header", "split_unit"]
