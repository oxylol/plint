"""What a model is.

`prepare(panel, env)` may compute anything from the panel it is handed, and the panel only ever holds days
up to the evening being decided (live) or is read row by row with each row computed from the rows before it
(backtest). tests/test_leak.py runs every model both ways and requires the same decisions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..bars import d2s
from ..panel import Panel
from ..portfolio import TopK
from ..types import BookView, Decision


@dataclass
class Env:
    index: str = "SPY"
    cash: str = "BIL"
    names: dict[str, str] = field(default_factory=dict)
    events: object | None = None            # earnings days and insider buys, when filings are switched on
    copy: object | None = None              # the copy study's picks and the trades they follow (models.copy.CopyData)
    fits: object | None = None              # the learned models' fitted parameters (learn.fit.Fits)
    scores: dict[str, np.ndarray] = field(default_factory=dict)   # research: score tables handed in ready-made
    peers: dict[str, "Model"] = field(default_factory=dict)       # the other models, for the blend
    cache: dict = field(default_factory=dict)                     # today's inputs, shared by the learned models


class Model:
    id = "model"
    title = ""
    kind = "rule"                           # "index" | "rule" | "learned" | "agent"
    what = ""                               # one sentence for the page
    horizon: int | None = None              # trading days its scores speak about (how the ledger settles them)

    def prepare(self, panel: Panel, env: Env) -> None:
        self.panel, self.env = panel, env

    def waiting(self) -> str:
        """Why it cannot decide yet ('' when it can)."""
        return ""

    def decide(self, t: int, view: BookView) -> Decision:
        raise NotImplementedError

    def _decision(self, t: int) -> Decision:
        return Decision(self.id, d2s(self.panel.days[t]))


class ScoreModel(Model):
    """A model that gives every stock a score each evening and holds the best."""
    top = TopK()
    horizon = 5

    def prepare(self, panel: Panel, env: Env) -> None:
        super().prepare(panel, env)
        ready = env.scores.get(self.id)
        self.S = ready if ready is not None else self.score_table()     # [T, N], NaN = no opinion

    def score_table(self) -> np.ndarray:
        raise NotImplementedError

    def ranked(self, t: int) -> tuple[list[str], dict[str, float]]:
        """Today's stocks best first, and their scores."""
        cols = self.panel.members(t)
        s = self.S[t, cols]
        ok = np.isfinite(s) & np.isfinite(self.panel.c[t, cols])
        cols, s = cols[ok], s[ok]
        order = np.argsort(-s, kind="stable")
        syms = [self.panel.syms[j] for j in cols[order]]
        return syms, dict(zip(syms, s[order].astype(float).tolist()))

    def decide(self, t: int, view: BookView) -> Decision:
        d = self._decision(t)
        ranked, scores = self.ranked(t)
        if not ranked:
            d.note = self.waiting() or "nothing to rank today"
            return d
        d.scores = scores
        prev = int(self.panel.days[t - 1]) if t > 0 else None
        if self.top.due(int(self.panel.days[t]), prev, not view.holdings):
            d.orders = self.top.orders(ranked, view)
            buys = sum(o.weight > 0 and o.symbol not in view.holdings for o in d.orders)
            sells = sum(o.weight == 0 for o in d.orders)
            d.note = f"{len(ranked)} stocks ranked; {buys} to buy, {sells} to sell at the next open"
        else:
            d.note = f"{len(ranked)} stocks ranked; it trades once a {self.top.every}"
        return d
