"""The copy books as models of the lab (copytrade/ is the rest of copy trades; this part lives with the models so
the two packages never import each other half-built): one decide(), one Book, the same costs as every other model.

`CopyBook` follows a schedule of periods. Each period starts on an evening and names the plans it follows
and the share of the book each one gets: one person with all of it, or the combo's ten with a tenth each.
On an evening when any of them changes, the book is set to the sum of their shares; a holding already
within a quarter of its share is left alone. Whatever is not in copied stocks waits in the index fund.

`Copy` is the live combo (id "copy"): its periods are the picks the study writes (study.py), and its plans
are worked out every evening from the trades known by then."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..book import Costs
from ..copytrade.plan import Plan, liquid, make
from ..panel import Panel
from ..types import BookView, Decision, Order
from .base import Env, Model

FULL = 0.995            # the share of the book put to work; the rest pays for trading
BAND = 0.25             # a holding this close to its share (as a part of that share) is left alone
SLACK = 0.03            # the index holding is left alone until it is this far from where it should be


@dataclass
class Period:
    start: int                          # YYYYMMDD: the first evening it decides on
    plans: list                         # [Plan]
    names: dict                         # pid -> a name for the page
    share: float = 1.0                  # of the book, for each plan


class CopyBook(Model):
    kind = "copy"
    horizon = None                      # no ranking or stated chance to score: judged by its book alone

    def __init__(self, mid: str, title: str, periods: list[Period], what: str = ""):
        self.id, self.title, self.periods, self.what = mid, title, sorted(periods, key=lambda p: p.start), what

    # the schedule
    def period_at(self, day: int) -> Period | None:
        cur = None
        for p in self.periods:
            if p.start <= day:
                cur = p
            else:
                break
        return cur

    def target(self, day: int) -> tuple[dict[str, float], dict[str, str]]:
        """The book's shares from the evening of `day` on, and what moved tonight, by stock."""
        p = self.period_at(day)
        if p is None:
            return {}, {}
        out: dict[str, float] = {}
        why: dict[str, str] = {}
        for plan in p.plans:
            for s, w in plan.at(day).items():
                out[s] = out.get(s, 0.0) + w * p.share
            for s, text in plan.note(day).items():
                who = p.names.get(plan.pid, plan.pid)
                why[s] = f"{who}: {text}" if len(p.plans) > 1 else text
        return {s: round(w, 6) for s, w in out.items() if w > 0}, why

    def prepare(self, panel: Panel, env: Env) -> None:
        super().prepare(panel, env)
        days = set()
        for p in self.periods:
            days.add(p.start)
            for plan in p.plans:
                days.update(int(d) for d in plan.days if d >= p.start)
        rows = np.searchsorted(panel.days, sorted(days))
        self.change = {int(t) for t in rows if t < panel.T}

    def decide(self, t: int, view: BookView) -> Decision:
        d = self._decision(t)
        idx = self.env.index
        first = not view.holdings
        if t not in self.change and not first:
            d.note = "nothing new made public tonight"
            return d
        day = int(self.panel.days[t])
        target, why = self.target(day)
        target.pop(idx, None)                   # someone who buys the index fund itself: that share is the index holding
        held = {s: h for s, h in view.holdings.items() if s != idx}
        orders, used = [], 0.0
        for s in held:
            if s not in target:
                orders.append(Order(s, 0.0, why.get(s, "no longer held by who it copies")))
        for s, w in sorted(target.items()):
            want = round(w * FULL, 4)
            have = held[s].weight if s in held else 0.0
            if s not in held:
                orders.append(Order(s, want, why.get(s, "held by who it copies")))
                used += want
            elif abs(have - want) > BAND * want:
                orders.append(Order(s, want, why.get(s, "back to its share")))
                used += want
            else:
                used += have
        park = max(0.0, FULL - used)
        have_idx = view.holdings[idx].weight if idx in view.holdings else 0.0
        if first or orders or abs(park - have_idx) > SLACK:
            orders.insert(0, Order(idx, round(park, 4) if park >= 0.01 else 0.0, "money not in use waits in the index fund"))
        d.orders = orders
        buys = sum(o.weight > 0 and o.symbol != idx and o.symbol not in held for o in orders)
        sells = sum(o.weight == 0 and o.symbol != idx for o in orders)
        d.note = f"{len(target)} stocks copied; {buys} to buy, {sells} to sell at the next open" if target else \
            "nothing copied yet: all in the index fund"
        return d


# ── the live combo ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CopyData:
    """What the live combo needs: the trades, the picks the study made, and the study's word on which
    purchases could be copied up to the last evening it saw (`flags_to`); after that the evening's own
    panel decides."""
    trades: pd.DataFrame
    picks: list                         # [{"start": YYYYMMDD, "people": [pid, …]}], oldest first
    names: dict                         # pid -> name
    calendar: np.ndarray                # the trading days the study saw
    flags: dict                         # (day, sym) -> bool
    flags_to: int

    def trades_of(self, pid: str) -> pd.DataFrame:
        if not hasattr(self, "_by"):
            self._by = {k: g for k, g in self.trades.groupby("pid", sort=False)} if len(self.trades) else {}
        return self._by.get(pid, self.trades.iloc[:0])


class Copy(CopyBook):
    """The fair combo, live. Its picks come from the study (made before each January and July); its plans
    are worked out every evening from the trades known by then."""
    id, title = "copy", "Copy the best ten"
    what = ("Copies the ten House members and company insiders whose disclosed trades did best against the index over the "
            "three years before, re-picked every January and July. Each trade is copied the day it was made public.")

    def __init__(self):
        super().__init__(self.id, self.title, [], self.what)
        self._wait = ""

    def waiting(self) -> str:
        return self._wait

    def prepare(self, panel: Panel, env: Env) -> None:
        data = getattr(env, "copy", None)
        self.periods, self._wait = [], ""
        if data is None or not data.picks:
            self._wait = "waiting for the copy study"
            Model.prepare(self, panel, env)
            self.change = set()
            return
        on_panel = liquid(panel, Costs(panel).adv)

        def ok(day: int, sym: str) -> bool:
            return bool(data.flags.get((day, sym), False)) if day <= data.flags_to else on_panel(day, sym)

        cal = np.union1d(np.asarray(data.calendar, dtype=np.int64), panel.days.astype(np.int64))
        cal = cal[cal <= int(panel.days[-1])]
        plans: dict[str, Plan] = {}
        for pick in data.picks:
            for pid in pick["people"]:
                if pid not in plans:
                    plans[pid] = make(pid, data.trades_of(pid), cal, ok, pid.startswith("in:"))
        self.periods = sorted((Period(int(p["start"]), [plans[pid] for pid in p["people"]], data.names,
                                      1.0 / max(len(p["people"]), 1)) for p in data.picks), key=lambda p: p.start)
        super().prepare(panel, env)
