"""The collector's rules: when is a day's temperature final, and which tokens then pay $1.

Pure functions over the station's reports (polybot/metar.py), tested without the network.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .metar import Report, whole_degrees
from .weather import Bucket

RESOLUTION_URL = re.compile(r"aviationweather\.gov/api/data/metar\?[^\s]*ids=([A-Z0-9]{4})[^\s]*")
READ_AT = re.compile(r"[?&]date=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z")
STATION = re.compile(r"\(([A-Z]{4})\)")


def resolution_source(description: str | None) -> tuple[str | None, datetime | None, bool]:
    """(station, the time the resolution read is taken, whether the market uses the METAR rule).

    The rule is the one in the description: the hourly temperature remark of routine observations. Cities that
    resolve on another source (Hong Kong: the Observatory's daily extract) come back as not using it.
    """
    d = description or ""
    metar_rule = "hourly temperature remark" in d
    m = RESOLUTION_URL.search(d)
    icao = m.group(1) if m else None
    if icao is None:
        m2 = STATION.search(d)
        icao = m2.group(1) if m2 else None
    r = READ_AT.search(d)
    read_at = datetime.fromisoformat(r.group(1)).replace(tzinfo=__import__("datetime").timezone.utc) if r else None
    return icao, read_at, metar_rule


@dataclass
class DaySummary:
    day: date
    n: int = 0
    max_tenths: int | None = None
    min_tenths: int | None = None
    max_at: datetime | None = None     # local time of the first report at the high
    min_at: datetime | None = None
    latest_tenths: int | None = None
    latest_at: datetime | None = None
    gap_min: float = 0.0               # the longest stretch without a routine report, midnight to `until`

    def value(self, kind: str) -> int | None:
        return self.max_tenths if kind == "highest" else self.min_tenths

    def value_at(self, kind: str) -> datetime | None:
        return self.max_at if kind == "highest" else self.min_at


def day_summary(reports: list[Report], tz, day: date, until: datetime) -> DaySummary:
    """The routine reports whose time stamp falls on `day` in the station's clock, summarised up to `until` (local)."""
    s = DaySummary(day=day)
    midnight = datetime.combine(day, datetime.min.time(), tzinfo=tz)
    end = min(until, midnight + timedelta(days=1))
    prev = midnight
    gap = 0.0
    for r in reports:
        if r.special:
            continue
        lt = r.local(tz)
        if lt.date() != day:
            continue
        s.n += 1
        gap = max(gap, (lt - prev).total_seconds() / 60)
        prev = lt
        if s.max_tenths is None or r.tenths > s.max_tenths:
            s.max_tenths, s.max_at = r.tenths, lt
        if s.min_tenths is None or r.tenths < s.min_tenths:
            s.min_tenths, s.min_at = r.tenths, lt
        s.latest_tenths, s.latest_at = r.tenths, lt
    if end > prev:
        gap = max(gap, (end - prev).total_seconds() / 60)
    s.gap_min = gap
    return s


def lock_state(kind: str, s: DaySummary, local_now: datetime, lock_hour: int, margin_tenths: int, max_gap_min: float) -> tuple[bool, str]:
    """Is the day's value final enough to buy on? Highs lock from `lock_hour` once the reading has fallen `margin`
    below the high; lows only once the day is over. A gap in the reports (a report may still turn up) blocks it."""
    if s.n == 0 or s.value(kind) is None:
        return False, "no reports yet"
    day_over = local_now.date() > s.day
    if s.gap_min > max_gap_min:
        return False, f"{s.gap_min:.0f} min without a report"
    if day_over:
        return True, "day over"
    if kind == "lowest":
        return False, "lows wait for the day to end"
    if local_now.hour < lock_hour:
        return False, f"waiting for {lock_hour:02d}:00"
    drop = s.max_tenths - (s.latest_tenths if s.latest_tenths is not None else s.max_tenths)
    if drop < margin_tenths:
        return False, f"still within {margin_tenths / 10:g}°C of the high"
    return True, f"high {s.max_tenths / 10:g}°C at {s.max_at:%H:%M}, now {drop / 10:g}°C below"


def winning_bucket(buckets: list[Bucket], tenths: int, unit: str) -> Bucket | None:
    v = whole_degrees(tenths, unit)
    for b in buckets:
        if b.contains(v):
            return b
    return None


def daily_max_hours(reports: list[Report], tz, before: date, min_reports: int = 12) -> list[tuple[date, float]]:
    """For each complete day before `before` (station clock): the hour, as a decimal, when the day's high was first reached."""
    by_day: dict[date, list[tuple[datetime, int]]] = {}
    for r in reports:
        lt = r.local(tz)
        if lt.date() < before:
            by_day.setdefault(lt.date(), []).append((lt, r.tenths))
    out = []
    for d, rs in sorted(by_day.items()):
        if len(rs) < min_reports:
            continue
        hi = max(t for _, t in rs)
        first = next(lt for lt, t in rs if t == hi)
        out.append((d, first.hour + first.minute / 60))
    return out


def city_hour(max_hours: list[float], q: float, lo: int = 11, hi: int = 19) -> int:
    """The whole hour by which the high had been reached on a share q of the days, clamped to lo–hi."""
    if not max_hours:
        raise ValueError("no history")
    xs = sorted(max_hours)
    k = max(0, min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1))
    return min(max(int(math.ceil(xs[k])), lo), hi)


def in_window(local: datetime, hour: int) -> int:
    """-1 before the hour, 0 during it (hour:00 to hour:59), +1 after."""
    return -1 if local.hour < hour else 0 if local.hour == hour else 1


def payers(buckets: list[Bucket], winner: Bucket) -> list[tuple[Bucket, str]]:
    """Every token worth $1 once `winner` is settled: its Yes and each other bucket's No."""
    return [(winner, "Yes")] + [(b, "No") for b in buckets if b is not winner]
