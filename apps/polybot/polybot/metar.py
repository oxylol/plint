"""METAR reports from the Aviation Weather Center, read the way the temperature markets' rules read them.

A market's description says: the qualifying value of a routine observation is
the hourly temperature remark (T group, tenths of °C) when it carries one, else
the temperature group in the body (whole °C); SPECI reports do not count; the
value is rounded to the nearest whole degree with halves up, converted to °F
first (× 9/5 + 32) for US cities; an observation belongs to the date its time
stamp falls on in the station's local time. The resolution source is this
same API (`format=raw`), so what it shows is what counts.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger("polybot.metar")
API = "https://aviationweather.gov/api/data/metar"
T_GROUP = re.compile(r"^T([01])(\d{3})([01])(\d{3})$")
BODY_TEMP = re.compile(r"^(M?)(\d{2})/(?:M?\d{2}|//)?$")


@dataclass(frozen=True)
class Report:
    at: datetime         # the observation's own time stamp, UTC
    tenths: int          # qualifying temperature in tenths of °C
    raw: str
    special: bool = False

    def local(self, tz) -> datetime:
        return self.at.astimezone(tz)


def parse_temperature(raw: str) -> int | None:
    """The qualifying temperature of one report in tenths of °C, or None when it reports none."""
    tokens = raw.split()
    for t in tokens:
        m = T_GROUP.match(t)
        if m:
            return (-1 if m.group(1) == "1" else 1) * int(m.group(2))
    body = tokens[:tokens.index("RMK")] if "RMK" in tokens else tokens
    for t in body:
        m = BODY_TEMP.match(t)
        if m:
            return (-1 if m.group(1) else 1) * int(m.group(2)) * 10
    return None


def is_special(raw: str, kind: str | None = None) -> bool:
    return raw.startswith("SPECI") or kind == "SPECI"


def whole_degrees(tenths: int, unit: str) -> int:
    """Round to the nearest whole degree, halves up (−8.5 → −8); °F markets convert first."""
    if unit == "F":
        return (9 * tenths + 1625) // 50     # (9t/50 + 32) + ½, floored
    return (tenths + 5) // 10


def fetch_reports(icaos: list[str], hours: int = 30, timeout: int = 25) -> dict[str, list[Report]]:
    """The last `hours` of reports for each station, oldest first; one per observation time (the latest received wins)."""
    url = API + "?" + urllib.parse.urlencode({"ids": ",".join(icaos), "format": "json", "hours": hours})
    req = urllib.request.Request(url, headers={"User-Agent": "plint-polybot/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    out: dict[str, dict[datetime, tuple[str, Report]]] = {i: {} for i in icaos}
    for d in data if isinstance(data, list) else []:
        icao = d.get("icaoId")
        raw = (d.get("rawOb") or "").strip()
        t = d.get("obsTime")
        if icao not in out or not raw or t is None:
            continue
        tenths = parse_temperature(raw)
        if tenths is None:
            continue
        at = datetime.fromtimestamp(int(t), timezone.utc)
        rec = (str(d.get("receiptTime") or ""), Report(at, tenths, raw, is_special(raw, d.get("metarType"))))
        prev = out[icao].get(at)
        if prev is None or rec[0] >= prev[0]:
            out[icao][at] = rec
    return {i: sorted((r for _, r in v.values()), key=lambda r: r.at) for i, v in out.items()}


IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def fetch_history(icao: str, days: int = 45, timeout: int = 60) -> list[Report]:
    """The station's reports for the last `days` from the IEM ASOS archive (routine and special, temperature in °F),
    for the afternoon strategy's record of when each city's high lands. Not the resolution source: just climatology."""
    from datetime import timedelta
    end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=days + 1)
    url = IEM + "?" + urllib.parse.urlencode({"station": icao, "data": "tmpf", "year1": start.year, "month1": start.month, "day1": start.day,
                                              "year2": end.year, "month2": end.month, "day2": end.day, "tz": "Etc/UTC", "format": "onlycomma",
                                              "latlon": "no", "missing": "M", "trace": "T", "direct": "no"}) + "&report_type=3&report_type=4"
    req = urllib.request.Request(url, headers={"User-Agent": "plint-polybot/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    out = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", "", "T"):
            continue
        try:
            at = datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            f = float(parts[2])
        except ValueError:
            continue
        out.append(Report(at, int(round((f - 32) * 50 / 9)), ""))
    out.sort(key=lambda r: r.at)
    return out
