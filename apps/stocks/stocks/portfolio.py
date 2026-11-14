"""From a ranking to orders: hold the best K, swap a few a day, leave the rest alone.

One rule for every model that ranks stocks, declared before any data was seen (DESIGN.md), so the models
differ in what they know and not in how cleverly their book was tuned. A holding stays while it still
ranks well (`hold_frac` is looser than the entry bar: that gap keeps turnover, and so costs, down) and for
`max_hold` days at most unless it is still one of the top K; at most `max_new` names change per decision;
a new name gets 1/K of the book."""
from __future__ import annotations

from dataclasses import dataclass

from .types import BookView, Order


@dataclass(frozen=True)
class TopK:
    k: int = 40
    hold_frac: float = 0.20       # a holding is kept while it ranks in this top share of the day's stocks
    max_new: int = 8              # names swapped per decision (the first fill of an empty book is exempt)
    every: str = "day"            # "day" | "week" | "month": decide on the first trading day of each
    max_hold: int | None = None   # trading days after which a holding is sold, unless it is still in the top K
    trim_at: float = 2.0          # a holding grown past this many times its share is cut back to its share

    def due(self, day: int, prev_day: int | None, empty: bool) -> bool:
        if empty or self.every == "day" or prev_day is None:
            return True
        if self.every == "month":
            return day // 100 != prev_day // 100
        return _week(day) != _week(prev_day)

    def orders(self, ranked: list[str], view: BookView) -> list[Order]:
        """`ranked`: today's stocks, best first. Returns sells, then trims, then buys."""
        n = len(ranked)
        hold_rank = max(self.k, round(self.hold_frac * n))
        rank = {s: i + 1 for i, s in enumerate(ranked)}
        held = view.holdings
        share = 1.0 / self.k
        empty = not held
        must, tired = [], []
        for s, h in held.items():
            r = rank.get(s)
            if r is None:
                must.append((s, "no longer ranked (left the universe or did not trade)"))
            elif self.max_hold is not None and h.days + 1 >= self.max_hold and r > self.k:
                tired.append((r, s, f"held {h.days + 1} days and no longer a top pick"))    # sessions held, today's included
            elif r > hold_rank:
                tired.append((r, s, f"fell to rank {r} of {n}"))
        tired.sort(reverse=True)                                    # worst rank first
        room = max(0, self.max_new - len(must))                     # a whole book bought on one day does not leave on one day
        out = must + [(s, why) for _, s, why in tired[:room]]
        orders = [Order(s, 0.0, why) for s, why in out]
        gone = {s for s, _ in out}
        kept = [s for s in held if s not in gone]
        for s in kept:
            if held[s].weight > self.trim_at * share:
                orders.append(Order(s, share, "grown to twice its share: cut back"))
        slots = self.k - len(kept)
        budget = self.k if empty else self.max_new
        picks = [s for s in ranked[: 2 * self.k] if s not in held][: max(0, min(slots, budget))]
        orders += [Order(s, share, f"rank {rank[s]} of {n}") for s in picks]
        return orders


def _week(day: int) -> tuple[int, int]:
    from datetime import date
    y, w, _ = date(day // 10000, day // 100 % 100, day % 100).isocalendar()
    return y, w
