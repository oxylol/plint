"""A paper book: cash and positions, moved through each day by the price table.

A position is kept as dollars, not shares. Each day it takes the overnight total return (previous close to
this open), is traded at the open, then takes the session's move to the close. Splits and dividends are
inside the overnight return, so the book never needs to know about them; a dividend is in effect put back
into the stock that paid it. What a trade costs is charged at the open, per dollar traded (Costs).

Orders are filled at the day's opening price as the consolidated tape reports it. A real market-on-open
order gets the opening auction's price, which is close to it but not the same; the cost model's slip is
there for that."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import lag, rstd
from .panel import Panel
from .types import BookView, Holding, Order
from .util import quiet

BAND = 0.002            # a change smaller than this share of the book is not worth a trade
STALE_DAYS = 10         # no price for this long: the position is closed, and the page says how
COLLAPSE = -0.30        # a fall this deep over its last ten days of trading, then silence: written off, not cashed in


class Costs:
    """What a dollar traded at the open costs: half the spread, a slip, and the push of the order itself.

    The spread is read off how much money trades in the stock (the median day of the last month, known the
    evening before). The push follows the square-root rule: daily volatility x sqrt(order / a day's trading).
    At $10,000 a book the push is nothing; it is here so a larger book is not flattered."""

    HALF_SPREAD_BPS = ((1e9, 1.0), (2e8, 1.5), (5e7, 2.5), (2e7, 4.0), (5e6, 8.0), (0.0, 20.0))

    def __init__(self, panel: Panel, slip_bps: float = 1.0, impact: float = 1.0, scale: float = 1.0):
        self.slip, self.impact, self.scale = slip_bps, impact, scale
        with quiet():
            import pandas as pd
            adv = pd.DataFrame(panel.dv).rolling(21, min_periods=5).median().to_numpy(dtype=np.float32)
        self.adv = lag(adv, 1)                  # row t: known at the close of t-1
        self.sig = lag(rstd(panel.ret, 21, 5), 1)

    def rate(self, t: int, j: int, usd: float) -> float:
        adv = self.adv[t, j]
        if not np.isfinite(adv) or adv <= 0:
            return self.scale * (20.0 + self.slip) / 1e4
        half = next(bps for floor, bps in self.HALF_SPREAD_BPS if adv >= floor)
        sig = self.sig[t, j] if np.isfinite(self.sig[t, j]) else 0.02
        push = self.impact * sig * np.sqrt(max(usd, 0.0) / adv)
        return self.scale * ((half + self.slip) / 1e4 + push)


@dataclass
class Pos:
    value: float            # dollars, at the last mark
    basis: float            # dollars paid for what is held now, costs included
    since: int              # the day it was opened
    days: int = 0           # trading days held
    why: str = ""
    stale: int = 0          # days in a row without a price


@dataclass
class Fill:
    d: int
    sym: str
    side: str               # "buy" | "sell"
    usd: float              # value traded
    price: float            # the open it was filled at
    cost: float             # dollars
    why: str
    ret: float | None = None    # on a sell: what that part of the position made, costs in


@dataclass
class Book:
    cash: float
    pos: dict[str, Pos] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    traded: float = 0.0     # dollars traded so far (both ways)
    costs: float = 0.0
    notes: list[str] = field(default_factory=list)
    bought_out: frozenset = frozenset()     # companies known to have been taken over: their last price was the deal

    def equity(self) -> float:
        return self.cash + sum(p.value for p in self.pos.values())

    def _gone_quiet(self, panel: Panel, t: int, s: str, p: Pos, j: int | None) -> None:
        """A holding with no price for STALE_DAYS. A company that was bought out stopped at the deal price, so
        it is cashed in at its last value. One that fell hard and then went silent was most likely seized or
        went under (free data does not say which): it is written off. Anything else is cashed in at its last
        value, which is right for a move off the exchange and kind to whatever else it may have been."""
        d = int(panel.days[t])
        fall = last_ten_days(panel, t, j)
        if s not in self.bought_out and np.isfinite(fall) and fall <= COLLAPSE:
            why = f"stopped trading after falling {abs(fall):.0%} in its last ten days: written off"
            self.fills.append(Fill(d, s, "sell", 0.0, float("nan"), 0.0, why, -1.0))
        else:
            why = f"no price for {STALE_DAYS} days: closed at its last value"
            self.cash += p.value
            self.fills.append(Fill(d, s, "sell", p.value, float("nan"), 0.0, why, p.value / p.basis - 1.0 if p.basis > 0 else None))
        self.notes.append(f"{s}: {why} ({_iso(d)})")
        del self.pos[s]

    def view(self) -> BookView:
        eq = self.equity()
        return BookView(equity=eq, cash_weight=self.cash / eq if eq > 0 else 1.0, holdings={
            s: Holding(s, p.value / eq if eq > 0 else 0.0, p.days, _iso(p.since), p.value / p.basis - 1.0 if p.basis > 0 else 0.0)
            for s, p in self.pos.items()})

    def step(self, panel: Panel, t: int, orders: list[Order], costs: Costs) -> list[Fill]:
        """Day t: the night, the trades at the open, the session. Returns the day's fills."""
        d = int(panel.days[t])
        # 1. the night
        for s, p in list(self.pos.items()):
            j = panel.ix.get(s)
            has_bar = j is not None and np.isfinite(panel.c[t, j]) and panel.o[t, j] > 0
            if not has_bar:
                p.stale += 1
                p.days += 1
                if p.stale >= STALE_DAYS:
                    self._gone_quiet(panel, t, s, p, j)
                continue
            p.stale = 0
            p.days += 1
            ovn = panel.ovn[t, j]
            if np.isfinite(ovn):
                p.value *= 1.0 + float(ovn)
        # 2. the open
        day_fills: list[Fill] = []
        eq = self.equity()
        sells, buys = [], []
        for o in orders:
            j = panel.ix.get(o.symbol)
            if j is None or not np.isfinite(panel.c[t, j]) or not panel.o[t, j] > 0:
                self.notes.append(f"{o.symbol}: no opening price on {_iso(d)}, order not filled")
                continue
            have = self.pos[o.symbol].value if o.symbol in self.pos else 0.0
            want = max(0.0, float(o.weight)) * eq
            delta = want - have
            if want > 0 and have > 0 and abs(delta) < BAND * eq:
                continue
            if delta < 0 and have > 0:
                sells.append((o, j, -delta if want > 0 else have))
            elif delta > 0:
                buys.append((o, j, delta))
        for o, j, usd in sells:
            p = self.pos[o.symbol]
            usd = min(usd, p.value)
            cost = usd * costs.rate(t, j, usd)
            part = usd / p.value if p.value > 0 else 1.0
            basis_part = p.basis * part
            day_fills.append(Fill(d, o.symbol, "sell", usd, float(panel.o[t, j]), cost, o.why,
                                  (usd - cost) / basis_part - 1.0 if basis_part > 0 else None))
            self.cash += usd - cost
            p.value -= usd
            p.basis -= basis_part
            if p.value <= 1e-6 or o.weight <= 0:
                del self.pos[o.symbol]
        if buys:
            need = sum(usd * (1.0 + costs.rate(t, j, usd)) for _, j, usd in buys)
            scale = min(1.0, max(self.cash, 0.0) / need) if need > 0 else 0.0     # never more than the cash there is
            for o, j, usd in buys:
                usd *= scale
                if usd < 1.0:
                    continue
                cost = usd * costs.rate(t, j, usd)
                self.cash -= usd + cost
                p = self.pos.get(o.symbol)
                if p is None:
                    self.pos[o.symbol] = Pos(usd, usd + cost, d, 0, o.why)
                else:
                    p.value += usd
                    p.basis += usd + cost
                day_fills.append(Fill(d, o.symbol, "buy", usd, float(panel.o[t, j]), cost, o.why))
        for f in day_fills:
            self.traded += f.usd
            self.costs += f.cost
        self.fills += day_fills
        # 3. the session
        for s, p in self.pos.items():
            j = panel.ix.get(s)
            if j is not None and p.stale == 0:
                intr = panel.intr[t, j]
                if np.isfinite(intr):
                    p.value *= 1.0 + float(intr)
        return day_fills


def _iso(d: int) -> str:
    d = int(d)
    return f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"


def last_ten_days(panel: Panel, t: int, j: int | None) -> float:
    """What the stock did over its last ten days of trading before it went quiet (NaN if not known)."""
    if j is None:
        return float("nan")
    rows = np.flatnonzero(np.isfinite(panel.c[: t + 1, j]))
    if len(rows) < 2:
        return float("nan")
    tail = panel.ret[rows[-10:], j]
    return float(np.prod(1.0 + tail[np.isfinite(tail)]) - 1.0)


@dataclass
class Record:
    """A book's history: one row per day, and every fill."""
    days: list[int] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    cash: list[float] = field(default_factory=list)
    n_pos: list[int] = field(default_factory=list)
    traded: list[float] = field(default_factory=list)       # that day, both ways
    cost: list[float] = field(default_factory=list)
    book: Book | None = None

    def mark(self, d: int, book: Book, fills: list[Fill]) -> None:
        self.days.append(int(d))
        self.equity.append(book.equity())
        self.cash.append(book.cash)
        self.n_pos.append(len(book.pos))
        self.traded.append(sum(f.usd for f in fills))
        self.cost.append(sum(f.cost for f in fills))
        self.book = book


def replay(panel: Panel, orders_by_day: dict[int, list[Order]], start_cash: float, costs: Costs,
           first_day: int, bought_out: frozenset = frozenset()) -> Record:
    """A book from decisions already made. `orders_by_day[d]` was decided at the close of day d and is
    filled at the next trading day's open. The record starts at the close of `first_day` (all cash)."""
    book, rec = Book(start_cash, bought_out=bought_out), Record()
    t0 = panel.t_of(first_day)
    rec.mark(panel.days[t0], book, [])
    for t in range(t0 + 1, panel.T):
        fills = book.step(panel, t, orders_by_day.get(int(panel.days[t - 1]), []), costs)
        rec.mark(panel.days[t], book, fills)
    return rec
