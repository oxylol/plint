"""The New York Stock Exchange's days, by rule.

History takes its trading days from the price table itself (a day is a day the index fund traded). This
file is for the future: when to run next. If the rule is wrong for an unusual closure, the evening run
finds no new prices and says so; nothing trades on a guess."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SPECIAL_CLOSURES = {date(2018, 12, 5), date(2025, 1, 9)}   # days of mourning


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def easter(year: int) -> date:
    """Western Easter Sunday (the anonymous Gregorian computus)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date | None:
    """A holiday on Sunday is kept on Monday, one on Saturday on Friday."""
    if d.weekday() == 6:
        return d + timedelta(days=1)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def holidays(year: int) -> frozenset[date]:
    out = {
        _nth_weekday(year, 1, 0, 3),                 # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),                 # Washington's Birthday
        easter(year) - timedelta(days=2),            # Good Friday
        _last_weekday(year, 5, 0),                   # Memorial Day
        _nth_weekday(year, 9, 0, 1),                 # Labor Day
        _nth_weekday(year, 11, 3, 4),                # Thanksgiving
        _observed(date(year, 7, 4)),
        _observed(date(year, 12, 25)),
    }
    ny = date(year, 1, 1)
    if ny.weekday() != 5:                            # on a Saturday it is simply not kept (no Friday closure)
        out.add(_observed(ny))
    if year >= 2022:
        out.add(_observed(date(year, 6, 19)))        # Juneteenth
    return frozenset({d for d in out if d and d.year == year} | {d for d in SPECIAL_CLOSURES if d.year == year})


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def days_left_in_month(d: date) -> int:
    """Trading days from d to the end of its month, d included when it is one."""
    last = (date(d.year + (d.month == 12), d.month % 12 + 1, 1) - timedelta(days=1))
    return len(trading_days(d, last))


def next_run(now: datetime, run_at: str) -> datetime:
    """The next evening run: `run_at` New York time on a trading day, strictly after `now`."""
    hh, mm = (int(x) for x in run_at.split(":"))
    ny_now = now.astimezone(NY)
    d = ny_now.date()
    while True:
        if is_trading_day(d):
            t = datetime.combine(d, time(hh, mm), NY)
            if t > ny_now:
                return t
        d += timedelta(days=1)


def last_close(now: datetime, run_at: str) -> date:
    """The latest trading day whose evening run time has passed."""
    hh, mm = (int(x) for x in run_at.split(":"))
    ny_now = now.astimezone(NY)
    d = ny_now.date()
    if not (is_trading_day(d) and ny_now.time() >= time(hh, mm)):
        d -= timedelta(days=1)
        while not is_trading_day(d):
            d -= timedelta(days=1)
    return d
