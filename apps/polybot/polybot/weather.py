"""Temperature markets and forecasts: parsing the questions, finding the station, fetching the models.

Polymarket's daily temperature events have one market per degree bucket
("…be 24°C on October 30?", "…be 26°C or higher…", "…between 84-85°F…") and
resolve on the METAR observations of a named airport. Forecasts come from
Open-Meteo (free, several weather models) for that airport's coordinates.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger("polybot.weather")

QUESTION = re.compile(r"^Will the (highest|lowest) temperature in (.+?) be (.+?) on ([A-Z][a-z]+ \d{1,2})\?$")
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
STATION = re.compile(r"\(([A-Z]{4})\)")
FALLBACK_STATIONS = {"Hong Kong": "VHHH"}


@dataclass(frozen=True)
class Bucket:
    """One market of a temperature event."""
    kind: str            # highest | lowest
    city: str
    unit: str            # C | F
    lo: float | None     # None = open-ended below
    hi: float | None     # None = open-ended above
    day: date
    question: str

    def contains(self, value: float) -> bool:
        return (self.lo is None or value >= self.lo) and (self.hi is None or value <= self.hi)

    @property
    def label(self) -> str:
        if self.lo is None:
            return f"{self.hi:g}°{self.unit} or below"
        if self.hi is None:
            return f"{self.lo:g}°{self.unit} or higher"
        if self.lo == self.hi:
            return f"{self.lo:g}°{self.unit}"
        return f"{self.lo:g}–{self.hi:g}°{self.unit}"


def parse_question(question: str, year: int) -> Bucket | None:
    m = QUESTION.match(question or "")
    if not m:
        return None
    kind, city, middle, when = m.groups()
    month, dnum = when.split()
    if month not in MONTHS:
        return None
    day = date(year, MONTHS[month], int(dnum))
    mm = re.match(r"^(-?\d+)°([CF]) or below$", middle)
    if mm:
        return Bucket(kind, city, mm.group(2), None, float(mm.group(1)), day, question)
    mm = re.match(r"^(-?\d+)°([CF]) or higher$", middle)
    if mm:
        return Bucket(kind, city, mm.group(2), float(mm.group(1)), None, day, question)
    mm = re.match(r"^between (-?\d+)-(-?\d+)°([CF])$", middle)
    if mm:
        return Bucket(kind, city, mm.group(3), float(mm.group(1)), float(mm.group(2)), day, question)
    mm = re.match(r"^(-?\d+)°([CF])$", middle)
    if mm:
        return Bucket(kind, city, mm.group(2), float(mm.group(1)), float(mm.group(1)), day, question)
    return None


def parse_station(description: str | None, city: str) -> str | None:
    m = STATION.search(description or "")
    return m.group(1) if m else FALLBACK_STATIONS.get(city)


def to_unit(celsius: float, unit: str) -> float:
    return celsius * 9 / 5 + 32 if unit == "F" else celsius


def pick_bucket(buckets: list[Bucket], value: float) -> Bucket | None:
    """The bucket a reported temperature `value` (in the buckets' unit) would resolve to."""
    t = math.floor(value + 0.5)   # markets read whole degrees
    for b in buckets:
        if b.contains(t):
            return b
    return None


# ── stations and forecasts ───────────────────────────────────
_airports = None


def station_info(icao: str) -> dict | None:
    """(lat, lon, tz, name) of an airport from the offline database."""
    global _airports
    if _airports is None:
        try:
            import airportsdata
            _airports = airportsdata.load("ICAO")
        except Exception as e:
            log.warning("airport database unavailable: %s", e)
            _airports = {}
    r = _airports.get(icao)
    return {"lat": r["lat"], "lon": r["lon"], "tz": r["tz"], "name": r["name"]} if r else None


def geocode(city: str) -> dict | None:
    """Fallback when there is no station: Open-Meteo's geocoder."""
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
        res = (d.get("results") or [None])[0]
        return {"lat": res["latitude"], "lon": res["longitude"], "tz": res.get("timezone", "UTC"), "name": res["name"]} if res else None
    except Exception as e:
        log.warning("geocoding %s failed: %s", city, e)
        return None


@dataclass
class Forecast:
    day: date
    highs: dict[str, float] = field(default_factory=dict)   # model → max °C
    lows: dict[str, float] = field(default_factory=dict)
    fetched_at: datetime | None = None

    def agg(self, kind: str, how: str) -> float | None:
        vals = list((self.highs if kind == "highest" else self.lows).values())
        if not vals:
            return None
        if how == "median":
            vals = sorted(vals)
            return vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
        if how == "mean":
            return sum(vals) / len(vals)
        return max(vals) if kind == "highest" else min(vals)   # "the highest expected from the data" (lowest for lows)

    def spread(self, kind: str) -> float | None:
        vals = list((self.highs if kind == "highest" else self.lows).values())
        return (max(vals) - min(vals)) if vals else None


def fetch_forecast(lat: float, lon: float, day: date, models: list[str], days: int = 4) -> Forecast:
    """Daily max/min from each model for `day`, in the station's local time."""
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto", "forecast_days": days, "models": ",".join(models)})
    with urllib.request.urlopen(url, timeout=25) as r:
        d = json.load(r)
    daily = d.get("daily") or {}
    times = daily.get("time") or []
    fc = Forecast(day=day, fetched_at=datetime.now(timezone.utc))
    if day.isoformat() not in times:
        return fc
    i = times.index(day.isoformat())
    for key, vals in daily.items():
        if key == "time" or not isinstance(vals, list) or i >= len(vals) or vals[i] is None:
            continue
        if key.startswith("temperature_2m_max"):
            fc.highs[key.replace("temperature_2m_max", "").strip("_") or models[0]] = float(vals[i])
        elif key.startswith("temperature_2m_min"):
            fc.lows[key.replace("temperature_2m_min", "").strip("_") or models[0]] = float(vals[i])
    return fc


def local_now(tz: str, now: datetime) -> datetime:
    try:
        return now.astimezone(ZoneInfo(tz))
    except Exception:
        return now


def target_value(forecast_c: float, plus_c: float, unit: str) -> float:
    """Forecast plus the margin, in the market's unit (the margin is in °C)."""
    return to_unit(forecast_c + plus_c, unit)
