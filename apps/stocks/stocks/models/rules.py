"""The models that follow a fixed rule: the index, the trend filter, momentum and reversal.

Each is a published effect in its plainest form, with the numbers set here before any data was seen:
- momentum: Jegadeesh & Titman (1993), the past year's winners, skipping the last month.
- reversal: Lehmann (1990), Jegadeesh (1990), last week's losers bounce.
- trend: Faber (2007), hold the index only above its 200-day average.
They are here as honest rivals and as a check on the machinery: if momentum's backtest looked nothing like
the literature, the fault would be in the backtest."""
from __future__ import annotations

import numpy as np

from ..features import Base, lag, rmean, rstd, rsum
from ..panel import Panel
from ..portfolio import TopK
from ..types import BookView, Decision, Order
from .base import Env, Model, ScoreModel
from ..util import quiet


class Index(Model):
    id, title, kind = "index", "Index fund", "index"
    what = "Buys the S&P 500 fund once and holds it. The bar the others must beat."

    def decide(self, t: int, view: BookView) -> Decision:
        d = self._decision(t)
        if self.env.index not in view.holdings:
            d.orders = [Order(self.env.index, 1.0, "the index, bought once and held")]
        d.note = "holds the index"
        return d


class Trend(Model):
    id, title, kind = "trend", "Trend filter", "rule"
    what = "Holds the index fund while it is above its 200-day average, otherwise Treasury bills."
    BAND = 0.01                 # it must cross by this much: stops it flipping daily around the line

    def prepare(self, panel: Panel, env: Env) -> None:
        super().prepare(panel, env)
        j = panel.ix[env.index]
        with quiet():
            line = np.exp(np.cumsum(np.nan_to_num(np.log1p(panel.ret[:, j:j + 1])), axis=0)).astype(np.float32)
        self.gap = (line / rmean(line, 200, 200) - 1.0)[:, 0]       # how far above its average, NaN until 200 days exist

    def decide(self, t: int, view: BookView) -> Decision:
        d = self._decision(t)
        idx, cash, gap = self.env.index, self.env.cash, self.gap[t]
        in_index, in_cash = idx in view.holdings, cash in view.holdings
        if not np.isfinite(gap):
            want = "index"
        elif in_index:
            want = "cash" if gap < -self.BAND else "index"
        elif in_cash:
            want = "index" if gap > self.BAND else "cash"
        else:
            want = "index" if gap > 0 else "cash"
        where = f"{abs(gap):.1%} {'above' if gap >= 0 else 'below'} its 200-day average" if np.isfinite(gap) else "no 200 days of history yet"
        if want == "index" and not in_index:
            d.orders = [Order(cash, 0.0, f"index {where}")] * in_cash + [Order(idx, 1.0, f"index {where}")]
        elif want == "cash" and not in_cash:
            d.orders = [Order(idx, 0.0, f"index {where}")] * in_index + [Order(cash, 1.0, f"index {where}")]
        d.note = f"index {where}: in {'the index' if want == 'index' else 'Treasury bills'}"
        return d


class Momentum(ScoreModel):
    id, title, kind = "momentum", "Momentum", "rule"
    what = "Holds the 40 stocks that rose most over the past year, skipping the last month; re-picked monthly."
    top = TopK(k=40, hold_frac=0.30, max_new=40, every="month")
    horizon = 20

    def score_table(self) -> np.ndarray:
        b = Base(self.panel, self.env.index)
        s = lag(rsum(b.lr, 231), 21)
        s[~b.valid] = np.nan
        return s


class Reversal(ScoreModel):
    id, title, kind = "reversal", "Reversal", "rule"
    what = "Buys last week's hardest-hit stocks, measured against their own usual swings, and holds them about a week."
    top = TopK(k=40, hold_frac=0.50, max_new=10, every="day", max_hold=7)
    horizon = 5

    def score_table(self) -> np.ndarray:
        b = Base(self.panel, self.env.index)
        with quiet():
            s = -rsum(b.lr, 5) / (rstd(b.r, 63) * np.sqrt(5.0))
        s[~b.valid] = np.nan
        ev = self.env.events
        if ev is not None:                                   # a fall on results is news, not noise: it does not bounce
            s[ev.earnings_within(self.panel, 5)] = np.nan
        return s
