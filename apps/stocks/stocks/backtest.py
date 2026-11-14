"""History replayed: every model walked through the same days, each deciding on what it could know.

The loop is the live one. On each day a model's book takes the night, fills yesterday's orders at the open,
takes the session; then the model is shown its book and decides for tomorrow. The only thing a backtest
adds is that the score tables of the learned models were filled in advance, each row by a fit made before
that row (learn.fit.walk_forward)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .book import Book, Costs, Record
from .ledger import decile_spread, nw_t, table_ic
from .learn.dataset import forward, relative
from .metrics import by_year, summarize
from .models.base import Env, Model, ScoreModel
from .panel import Panel
from .util import quiet


@dataclass
class Outcome:
    model: Model
    record: Record
    stats: dict = field(default_factory=dict)
    years: list = field(default_factory=list)
    ic: dict = field(default_factory=dict)      # horizon → {"ic", "t", "n", "spread"}


def run(panel: Panel, models: list[Model], env: Env, t0: int, t1: int, start_cash: float = 10_000.0,
        slip_bps: float = 1.0, impact: float = 1.0, cost_scale: float = 1.0, extra_bps: float = 0.0,
        prepare: bool = True) -> dict[str, Outcome]:
    """Every model from the close of row t0 (all cash) to the close of row t1."""
    costs = Costs(panel, slip_bps + extra_bps / max(cost_scale, 1e-9), impact, cost_scale)
    if prepare:
        for m in models:                         # the blend reads the others, so it goes last
            env.peers[m.id] = m
        for m in sorted(models, key=lambda m: m.id == "blend"):
            m.prepare(panel, env)
    out = {}
    for m in models:
        book, rec, pending = Book(start_cash), Record(), []
        for t in range(t0, t1 + 1):
            fills = book.step(panel, t, pending, costs) if t > t0 else []
            rec.mark(panel.days[t], book, fills)
            pending = m.decide(t, book.view()).orders
        out[m.id] = Outcome(m, rec)
    return out


def score(panel: Panel, outcomes: dict[str, Outcome], index_id: str, cash_symbol: str, t0: int, t1: int,
          horizons: tuple[int, ...] = (1, 5, 20)) -> None:
    """Fill in each outcome's figures: the book against the index's book, and how the rankings held up."""
    index_eq = np.array(outcomes[index_id].record.equity)
    jc = panel.ix.get(cash_symbol)
    cash_eq = None
    if jc is not None:
        with quiet():
            cash_eq = np.r_[1.0, np.cumprod(1.0 + np.nan_to_num(panel.ret[t0 + 1:t1 + 1, jc]))]
    fwd = {}
    for o in outcomes.values():
        rec = o.record
        o.stats = summarize(np.array(rec.equity), index_eq, np.array(rec.traded), np.array(rec.cost), cash_eq)
        o.stats["trades"] = len(rec.book.fills) if rec.book else 0
        o.years = by_year(np.array(rec.days), np.array(rec.equity), index_eq)
        if isinstance(o.model, ScoreModel) and panel.member is not None:
            for h in horizons:
                if h not in fwd:
                    fwd[h] = relative(forward(panel, h), panel.member)
                ics = table_ic(o.model.S, fwd[h], panel.member, t0, t1 + 1)
                mean, _se, t, n = nw_t(ics, h - 1)
                o.ic[h] = {"ic": mean, "t": t, "n": n,
                           "spread": decile_spread(o.model.S, fwd[h], panel.member, t0, t1 + 1)}
