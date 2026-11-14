"""The contestants. Each is a Model: given the market up to an evening and its own book, it says what to
hold from the next open."""
from __future__ import annotations

from .base import Env, Model, ScoreModel
from .events import Drift, Insiders
from .learned import Blend, Net, NetFast, Ridge, Trees
from .rules import Index, Momentum, Reversal, Trend

# the order the page lists them in when nothing else decides
CLASSES = (Index, Net, NetFast, Trees, Ridge, Blend, Momentum, Reversal, Drift, Insiders, Trend)
AGENT = {"id": "agent", "title": "API agent", "kind": "agent",
         "what": "Claude reads the day's filings and news, writes a memo and trades at most four names a day, on $200 a month."}
ORDER = ("index", "net", "netfast", "trees", "ridge", "blend", "agent", "momentum", "reversal", "drift", "insiders", "trend")


def build(enabled: tuple[str, ...] = ()) -> list[Model]:
    """Fresh instances of the models that run (all of them when `enabled` is empty). The agent is not one of
    these: it decides through the desk (agent/), and the pipeline gives it a book like the rest."""
    return [c() for c in CLASSES if not enabled or c.id in enabled]


__all__ = ["Env", "Model", "ScoreModel", "build", "CLASSES", "AGENT", "ORDER"]
