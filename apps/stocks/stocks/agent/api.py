"""What the desk is given and what it hands back.

The desk (desk.py) is the only part that talks to the language model. Everything it knows about the
market comes through DeskData, which the pipeline builds from data stamped no later than tonight's close;
everything it wants done goes back as orders that the book checks against the limits below. The model
proposes, the book disposes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import BookView, Event, Order, Signal


@dataclass(frozen=True)
class DeskLimits:
    """Enforced in code after the model answers; the prompt states them too."""
    max_positions: int = 15
    max_weight: float = 0.12        # one position, share of the book
    min_weight: float = 0.03        # smaller than this is not worth a trade
    max_new_per_day: int = 4
    max_invested: float = 1.0       # no borrowing
    allow_index: bool = True        # it may park money in the index fund instead of cash
    new_needs_read: bool = True     # a new name is opened only in a company read in depth that evening


class DeskData(Protocol):
    """Tonight's view of the market. Nothing in it is later than `date`'s close."""
    date: str                       # YYYY-MM-DD, the close just finished
    universe: list[str]             # the stocks it may buy
    index_symbol: str
    days_left: int                  # trading days left this month, today included (for the day's allowance)
    days_in_month: int              # trading days in the whole month: a day's allowance is never more than 1.5 even days

    def profile(self, symbol: str) -> dict:
        """{symbol, name, exchange, price, ret: {1,5,21,63,252} (total return over that many trading days),
        vs_index: {same keys}, vol21 (daily, as a fraction), dollar_volume_rank (1 = most traded),
        days_since_earnings | None, days_to_earnings | None}"""

    def history(self, symbol: str, days: int = 120) -> list[dict]:
        """Daily bars, oldest first: {d, o, h, l, c, v}."""

    def movers(self, n: int = 40) -> list[dict]:
        """Today's largest moves against the index in units of each stock's own daily volatility:
        {symbol, name, ret, vs_index, sigmas, volume_ratio}."""

    def market(self) -> dict:
        """{index: {ret: {1,5,21,63,252}, vol21}, breadth_50d (share of stocks above their 50-day mean),
        dispersion (cross-section spread of today's returns)}"""

    def events(self, since: str) -> list[Event]:
        """Filings and news about universe companies that became public after `since` (ISO time), oldest first."""

    def fetch(self, url: str, max_chars: int = 60_000) -> str:
        """The text of a filing document. sec.gov only; anything else raises ValueError."""


@dataclass
class DeskResult:
    orders: list[Order] = field(default_factory=list)
    memo: str = ""                  # markdown for the page: what it did tonight and why, about 200 words
    signals: list[Signal] = field(default_factory=list)   # each deep read's stated chances (5 and 20 days)
    reads: list[dict] = field(default_factory=list)       # the deep reads as stored and shown:
    #   {symbol, thesis, facts: [str], priced_in: str, p5, p20, conviction (0..1), risks: str,
    #    sources: [{title, url}], usd}
    usd: float = 0.0                # spent on this run
    stopped: str | None = None      # why it ended early: "month cap" | "day allowance" | "no key" | "refusal" | "error" | None
    triaged: int = 0                # how many events the small model looked at


class DeskLike(Protocol):
    def run(self, data: DeskData, book: BookView) -> DeskResult: ...
    def weekly_review(self, data: DeskData, closed: list[dict], scored: list[dict]) -> str: ...
