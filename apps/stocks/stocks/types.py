"""The few shapes every part shares."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Order:
    """Decided after a close, filled at the next open. `weight` is the share of the book's value the
    position should have at that open; 0 sells it all. A symbol with no order is left alone."""
    symbol: str
    weight: float
    why: str = ""

    def __post_init__(self):
        object.__setattr__(self, "weight", float(self.weight))     # a numpy number would not go into the record

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    """What one model decided on one evening. Written once, never changed."""
    model: str
    date: str                                   # the close it was decided on, YYYY-MM-DD
    orders: list[Order] = field(default_factory=list)
    note: str = ""                              # one line for the page
    scores: dict[str, float] | None = None      # the model's score per symbol that day, for the ledger
    meta: dict = field(default_factory=dict)


@dataclass
class Holding:
    """One position as the model sees it when it decides."""
    symbol: str
    weight: float          # share of the book at the close
    days: int              # trading days held
    since: str
    ret: float             # return on what was paid, costs in


@dataclass
class BookView:
    """A model's own book at the close it decides on."""
    equity: float
    cash_weight: float
    holdings: dict[str, Holding] = field(default_factory=dict)


@dataclass
class Signal:
    """A forecast that can be scored later without a book: `p` is the stated chance that the stock beats
    the index over `horizon` trading days from the next open."""
    model: str
    date: str
    symbol: str
    horizon: int
    p: float
    note: str = ""


@dataclass
class Event:
    """Something that became public about a company, with the time it did."""
    id: str                # accession number, news id: stable, so nothing is read twice
    kind: str              # "8-K" | "10-Q" | "10-K" | "4" | "13D" | "news" | "move" | "earnings"
    symbol: str
    at: str                # ISO time (UTC) it became public
    title: str
    url: str | None = None
    text: str | None = None        # the body, if already fetched
    facts: dict = field(default_factory=dict)   # small structured extras, e.g. {"items": "2.02,9.01"}
