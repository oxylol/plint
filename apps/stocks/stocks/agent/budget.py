"""The agent's money. Every call is priced twice: before it is made, at the most it could cost, so the
month's cap is never passed by a call the desk chose to make; and after, from the usage the API reports,
so the meter shows what was billed. The month's cap is hard. The day's allowance is soft: it spreads what
is left over the trading days that remain, and the desk gives up searches, then reading, before it gives
up the decision."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..store import Store, now_iso
from . import prices

CHARS_PER_TOKEN = 3.0           # older tokenizers ran near 4; the 5.5 models count about 30% more, so 3 errs high
REQUEST_OVERHEAD = 64           # framing the text does not show
SEARCH_RESULT_TOKENS = 10_000   # what one web search may add to the context, assumed large
DAY_FACTOR = 1.5                # a busy evening may use one and a half average days

_DATED = re.compile(r"^(.+?)[-@]\d{8}(?:\D.*)?$")


def tokens(*texts: str) -> int:
    """A deliberately high guess at how many tokens this text is, made without a call."""
    return sum(math.ceil(len(t) / CHARS_PER_TOKEN) for t in texts if t) + REQUEST_OVERHEAD


def price_key(model: str) -> str:
    """The row of the price table for a model as a response names it. A response may name a dated
    snapshot; unmatched, that would be billed as the dearest model in the table."""
    if model in prices.PER_MTOK:
        return model
    m = _DATED.match(model)
    return m[1] if m and m[1] in prices.PER_MTOK else model


@dataclass(frozen=True)
class Bill:
    """What one model did in one call, as the API's usage reports it."""
    model: str
    in_tok: int = 0
    out_tok: int = 0
    cache_read: int = 0
    cache_write: int = 0
    searches: int = 0

    @property
    def usd(self) -> float:
        return prices.cost_usd(price_key(self.model), self.in_tok, self.out_tok, self.cache_read, self.cache_write,
                               self.searches)


def worst_case(model: str, in_tok: int, max_tokens: int, searches: int = 0) -> float:
    """The most a call could cost: every input token at the cache-write rate (the dearest an input token
    can be), the whole of max_tokens written, every search made. A search turn reads its prompt and the
    results so far again at each step, so its input is counted once per step as if nothing were cached."""
    if searches:
        in_tok = (searches + 2) * (in_tok + searches * SEARCH_RESULT_TOKENS)
    return prices.cost_usd(price_key(model), out_tok=max_tokens, cache_write=in_tok, searches=searches)


class Budget:
    """One run's view of the money. What was spent before the run is read from the store once; the run's
    own calls are added as they are recorded, so the gates see exactly what was written."""

    def __init__(self, store: Store, cap_usd: float, days_left: int, now: str | None = None,
                 days_in_month: int | None = None):
        now = now or now_iso()
        self.store = store
        self.cap = cap_usd
        self.month_before = store.spend_month(now[:7])
        self.day_before = store.spend_day(now[:10])
        self.spent = 0.0            # this run
        per_day = max(0.0, cap_usd - self.month_before) / max(1, days_left)
        if days_in_month:           # the cap is a ceiling, not a target: a month begun late, or a quiet first
            per_day = min(per_day, cap_usd / days_in_month)     # half, is not made up for by spending faster
        self.allowance = per_day * DAY_FACTOR

    @property
    def month(self) -> float:
        return self.month_before + self.spent

    @property
    def tonight(self) -> float:
        return self.day_before + self.spent

    def fits(self, estimate: float) -> bool:
        """Whether a call that could cost `estimate` leaves the month within its cap."""
        return self.month + estimate <= self.cap

    def record(self, stage: str, top: Bill, parts: list[Bill] | None = None) -> float:
        """Books a finished call and returns its cost. `top` is the response's own usage; `parts` are its
        per-attempt entries when the API gives them (a refused first attempt and the fallback that
        answered are different models at different prices). Whichever reading is dearer is booked, one
        row per model, so the meter can only read high."""
        rows = list(parts) if parts and sum(p.usd for p in parts) >= top.usd else [top]
        for b in rows:
            self.store.add_spend(stage=stage, model=price_key(b.model), in_tok=b.in_tok, out_tok=b.out_tok,
                                 cache_read=b.cache_read, cache_write=b.cache_write, searches=b.searches, usd=b.usd)
        usd = sum(b.usd for b in rows)
        self.spent += usd
        return usd
