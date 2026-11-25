"""What a person's copy book holds, evening by evening (DESIGN.md, "Copy trades").

A plan is worked out from a person's trades alone, in the order they became public, over a calendar of
trading days. It says, for each evening on which something changes, what share of the book each copied
stock should have from then on (the rest waits in the index fund), and why. Nothing in it can see past the
evening it is for: a trade enters on its `day` and a stake lapses a fixed number of trading days later.

Whether a stock could be copied (a price that evening, $1 million a day traded) is asked of `ok(day, sym)`
on the evening of the purchase. A purchase that could not be copied adds nothing; a sale of something not
held changes nothing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

HOUSE_LAPSE = 756           # trading days after its last purchase that a House stake with no sale seen lapses (3 years)
INSIDER_HOLD = 126          # trading days an insider's purchase is held (6 months)
CAP = 0.25                  # no stock above this share of the book
MIN_ADV = 1e6               # dollars a day (median of 21 days) for a stock to be copied


@dataclass
class Plan:
    pid: str
    days: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))     # evenings the shares change
    targets: list = field(default_factory=list)       # per evening: {sym: share of the book}, index fund not included
    why: list = field(default_factory=list)           # per evening: {sym: what happened}
    opened: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # a stake begun, by day
    copied: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # every trade acted on, by day

    def at(self, day: int) -> dict:
        """The shares held from the evening of `day` on (empty before the first purchase)."""
        i = int(np.searchsorted(self.days, day, side="right")) - 1
        return self.targets[i] if i >= 0 else {}

    def note(self, day: int) -> dict:
        i = int(np.searchsorted(self.days, day))
        return self.why[i] if i < len(self.days) and self.days[i] == day else {}

    def count(self, which: str, lo: int, hi: int) -> int:
        """How many positions were opened ('opened') or trades acted on ('copied') with lo < day <= hi."""
        a = getattr(self, which)
        return int(np.searchsorted(a, hi, side="right") - np.searchsorted(a, lo, side="right"))


def shares(stakes: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in stakes.values() if v > 0)
    if total <= 0:
        return {}
    return {s: round(min(CAP, v / total), 6) for s, v in sorted(stakes.items()) if v > 0}


def make(pid: str, trades: pd.DataFrame, calendar: np.ndarray, ok: Callable[[int, str], bool], insider: bool) -> Plan:
    """`trades`: one person's rows (sym, side, usd, day, filed), any order. `calendar`: trading days,
    ascending, as YYYYMMDD; trades after its last day are not yet known and are left out."""
    plan = Plan(pid)
    if trades is None or not len(trades):
        return plan
    cal = np.asarray(calendar, dtype=np.int64)
    t = trades.sort_values(["day", "filed"], kind="stable")
    idx = np.searchsorted(cal, t["day"].to_numpy(dtype=np.int64))          # a day the market was shut maps forward
    syms, sides = t["sym"].to_numpy(), t["side"].to_numpy()
    usds, fileds = t["usd"].to_numpy(dtype=float), t["filed"].to_numpy(dtype=np.int64)
    keep = idx < len(cal)
    stakes: dict[str, float] = {}               # House: dollars at stake per stock
    last_buy: dict[str, int] = {}               # House: calendar index of the last purchase
    lots: dict[str, list[list[float]]] = {}     # insiders: [dollars, calendar index it lapses at]
    days, targets, whys, opened, copied = [], [], [], [], []
    prev: dict[str, float] = {}

    def current() -> dict[str, float]:
        if insider:
            return {s: sum(v for v, _ in ls) for s, ls in lots.items() if ls}
        return dict(stakes)

    def emit(i: int, why: dict[str, str]) -> None:
        nonlocal prev
        now = shares(current())
        if now != prev or (why and days and days[-1] == int(cal[i])):
            if days and days[-1] == int(cal[i]):
                targets[-1], whys[-1] = now, {**whys[-1], **why}
            else:
                days.append(int(cal[i]))
                targets.append(now)
                whys.append(why)
            prev = now

    def lapse_until(i: int) -> None:
        """Every stake that lapses on or before calendar index i, each on its own evening."""
        while True:
            due = []
            if insider:
                due = [(lot[1], s) for s, ls in lots.items() for lot in ls if lot[1] <= i]
            else:
                due = [(last_buy[s] + HOUSE_LAPSE, s) for s in stakes if last_buy[s] + HOUSE_LAPSE <= i]
            if not due:
                return
            when = int(min(d for d, _ in due))
            gone = {}
            for d, s in due:
                if d != when:
                    continue
                if insider:
                    lots[s] = [lot for lot in lots[s] if lot[1] != when]
                    if not lots[s]:
                        del lots[s]
                        gone[s] = "held six months: lapsed"
                else:
                    del stakes[s], last_buy[s]
                    gone[s] = "three years with no sale seen: lapsed"
            emit(when, gone)

    order = np.flatnonzero(keep)
    k = 0
    while k < len(order):
        i = int(idx[order[k]])
        lapse_until(i - 1)
        why: dict[str, str] = {}
        while k < len(order) and int(idx[order[k]]) == i:
            r = order[k]
            s, side, usd, filed = str(syms[r]), str(sides[r]), float(usds[r]), int(fileds[r])
            when = f"{filed // 10000:04d}-{filed // 100 % 100:02d}-{filed % 100:02d}"
            held = s in (lots if insider else stakes)
            if side == "buy":
                if usd > 0 and ok(int(cal[i]), s):
                    if insider:
                        lots.setdefault(s, []).append([usd, i + INSIDER_HOLD])
                    else:
                        stakes[s] = stakes.get(s, 0.0) + usd
                        last_buy[s] = i
                    if not held:
                        opened.append(int(cal[i]))
                    copied.append(int(cal[i]))
                    why[s] = f"bought ${usd:,.0f}, reported {when}"
            elif held and not insider:
                if side == "sell" or stakes[s] - usd <= 0:
                    del stakes[s], last_buy[s]
                    why[s] = f"sold, reported {when}"
                else:
                    stakes[s] -= usd
                    why[s] = f"sold ${usd:,.0f} of it, reported {when}"
                copied.append(int(cal[i]))
            k += 1
        lapse_until(i)
        if why:
            emit(i, why)
    lapse_until(len(cal) - 1)
    plan.days = np.asarray(days, dtype=np.int64)
    plan.targets, plan.why = targets, whys
    plan.opened, plan.copied = np.asarray(sorted(opened), dtype=np.int64), np.asarray(sorted(copied), dtype=np.int64)
    return plan


def liquid(panel, adv: np.ndarray) -> Callable[[int, str], bool]:
    """ok(day, sym) on a panel: a price that evening and $1 million a day traded (the cost model's median,
    known the evening before)."""
    def ok(day: int, sym: str) -> bool:
        j = panel.ix.get(sym)
        if j is None:
            return False
        t = int(np.searchsorted(panel.days, day))
        if t >= panel.T or panel.days[t] != day:
            return False
        return bool(np.isfinite(panel.c[t, j]) and np.isfinite(adv[t, j]) and adv[t, j] >= MIN_ADV)
    return ok
