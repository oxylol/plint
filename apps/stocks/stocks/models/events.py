"""The models that wait for something to happen: a strong reaction to results, a cluster of insider buying.

Both are slow effects with a long paper trail, the kind a small account can still reach:
- after-earnings drift: Ball & Brown (1968), Bernard & Thomas (1989). Here in the form that needs no analyst
  estimates: the size of the price reaction on the day itself (Chan, Jegadeesh & Lakonishok 1996).
- insider buying: Lakonishok & Lee (2001), Cohen, Malloy & Pomorski (2012): purchases, by several insiders
  at once, are the informative ones; sales mostly are not.
Each buys at the next open and holds for a fixed stretch. Every qualifying event is logged as a forecast,
bought or not, so the ledger can score the rule and not just the book."""
from __future__ import annotations

import numpy as np

from ..features import Base, lag, rmean, rstd
from ..panel import Panel
from ..types import BookView, Decision, Order
from .base import Env, Model
from ..util import quiet


class EventModel(Model):
    kind = "rule"
    K = 40                  # most positions; each gets 1/K of the book
    HOLD = 40               # trading days
    horizon = 20
    P = 0.55                # the chance logged with each event: the rule's standing claim, not a fitted number

    def prepare(self, panel: Panel, env: Env) -> None:
        super().prepare(panel, env)
        self._wait = ""
        self.fire = np.zeros((panel.T, panel.N), dtype=np.float32)
        if env.events is None:
            self._wait = "waiting for filings: SEC_CONTACT is not set"
            return
        self.fire = self.fire_table()

    def waiting(self) -> str:
        return self._wait

    def fire_table(self) -> np.ndarray:
        """[T, N]: above zero on the evening a stock qualifies; larger is stronger."""
        raise NotImplementedError

    def why(self, t: int, j: int) -> str:
        raise NotImplementedError

    def decide(self, t: int, view: BookView) -> Decision:
        d = self._decision(t)
        if self._wait:
            d.note = self._wait
            return d
        p = self.panel
        held = view.holdings
        # `days` counts the closes since the day it was bought, so days + 1 is the sessions it has been held
        d.orders = [Order(s, 0.0, f"held {h.days + 1} days") for s, h in held.items() if h.days + 1 >= self.HOLD]
        free = self.K - (len(held) - len(d.orders))
        row = self.fire[t]
        cols = np.flatnonzero((row > 0) & np.isfinite(p.c[t]) & (p.member[t] if p.member is not None else True))
        cols = cols[np.argsort(-row[cols], kind="stable")]
        signals = []
        for j in cols:
            sym, why = p.syms[j], self.why(t, j)
            signals.append({"symbol": sym, "horizon": self.horizon, "p": self.P, "note": why})
            if sym not in held and free > 0:
                d.orders.append(Order(sym, 1.0 / self.K, why))
                free -= 1
        d.meta["signals"] = signals
        d.note = f"{len(cols)} new today; {len(held)} held" if len(cols) else f"nothing new today; {len(held)} held"
        return d


class Drift(EventModel):
    id, title = "drift", "After-earnings drift"
    what = "Buys a stock the day after a strong price reaction to its results and holds it eight weeks."
    MIN_MOVE, MIN_SIGMAS, MIN_SURGE = 0.04, 2.0, 1.5

    def fire_table(self) -> np.ndarray:
        p, ev = self.panel, self.env.events
        if ev.earnings.empty:
            self._wait = "waiting for the first results filings"
            return self.fire
        b = Base(p, self.env.index)
        with quiet():
            self.rel = (b.r - b.mkt[:, None]).astype(np.float32)             # the day's move over the market's
            sig = self.rel / lag(rstd(b.r, 63), 1)                           # in the stock's own usual days
            self.surge = (p.v / lag(rmean(p.v, 21), 1)).astype(np.float32)
            ok = ev.earnings_table(p) & (self.rel >= self.MIN_MOVE) & (sig >= self.MIN_SIGMAS) & (self.surge >= self.MIN_SURGE)
        return np.where(ok, sig, 0.0).astype(np.float32)

    def why(self, t: int, j: int) -> str:
        return f"{self.rel[t, j]:+.1%} over the market on results, {self.surge[t, j]:.1f}x volume"


class Insiders(EventModel):
    id, title = "insiders", "Insider buying"
    what = "Buys when two or more of a company's own officers or directors buy its shares on the open market within a month."
    HOLD = 60
    WINDOW, MIN_BUYERS, MIN_USD = 21, 2, 100_000.0

    def fire_table(self) -> np.ndarray:
        ev = self.env.events
        if ev.insiders.empty:
            self._wait = "waiting for the first insider filings"
            return self.fire
        return ev.insider_clusters(self.panel, self.WINDOW, self.MIN_BUYERS, self.MIN_USD)

    def why(self, t: int, j: int) -> str:
        return f"insiders bought ${self.fire[t, j]:,.0f} of it within a month"
