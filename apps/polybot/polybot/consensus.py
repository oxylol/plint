"""Pure consensus logic: turn the leaders' recent trades into signals.

For every market a leader touched in the lookback window we work out their
stance on each outcome: net USD flow into that outcome (buys minus sells)
minus their net flow into the other outcome of the same market. Someone who
buys both sides, as market makers do, is flat. A leader agrees with buying an
outcome when the stance is at least min_leader_usd, and is against it when
the stance is at most minus that. A signal's score is agree minus against.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Iterable


@dataclass(frozen=True)
class LeaderTrade:
    wallet: str
    condition_id: str
    asset_id: str
    side: str            # BUY | SELL
    size: Decimal        # shares
    price: Decimal       # USD per share
    timestamp: datetime  # aware, UTC
    title: str = ""
    slug: str = ""
    event_slug: str = ""
    outcome: str = ""
    outcome_index: int | None = None
    tx: str = ""

    @property
    def usd(self) -> Decimal:
        return self.size * self.price

    @property
    def key(self) -> tuple:
        return (self.tx, self.wallet, self.asset_id, self.side, str(self.size), str(self.price))


@dataclass
class Signal:
    side: str            # BUY | SELL
    condition_id: str
    asset_id: str
    outcome: str
    outcome_index: int | None
    title: str
    slug: str
    event_slug: str
    agree: list[str]
    disagree: list[str]
    avg_price: Decimal | None  # what the agreeing leaders paid (or got) for this outcome
    net_usd: Decimal           # the agreeing leaders' combined stance
    last_at: datetime          # most recent agreeing trade

    @property
    def score(self) -> int:
        return len(self.agree) - len(self.disagree)

    @property
    def key(self) -> str:
        return f"{self.side}:{self.asset_id}"


def _flow() -> dict:
    return {"net": Decimal(0), "buy_usd": Decimal(0), "buy_shares": Decimal(0),
            "sell_usd": Decimal(0), "sell_shares": Decimal(0), "last": None}


def _held(held: Iterable, condition_id: str, asset_id: str, outcome: str) -> bool:
    return asset_id in held or (condition_id, outcome.lower()) in held


def find_signals(trades: Iterable[LeaderTrade], *, min_leader_usd: Decimal, min_show: int = 2,
                 held: Iterable = frozenset()) -> list[Signal]:
    """Outcomes at least `min_show` leaders agree on buying, strongest first.

    SELL signals are only produced for outcomes in `held` (asset ids, or
    (condition_id, lower-cased outcome) pairs), since selling only matters for
    something we own. The caller decides what is enough to trade.
    """
    held = set(held)
    flows: dict[tuple[str, str], dict[str, dict]] = defaultdict(lambda: defaultdict(_flow))
    meta: dict[tuple[str, str], LeaderTrade] = {}
    for t in trades:
        k = (t.condition_id, t.asset_id)
        f = flows[k][t.wallet]
        if t.side == "BUY":
            f["net"] += t.usd
            f["buy_usd"] += t.usd
            f["buy_shares"] += t.size
        else:
            f["net"] -= t.usd
            f["sell_usd"] += t.usd
            f["sell_shares"] += t.size
        if f["last"] is None or t.timestamp > f["last"]:
            f["last"] = t.timestamp
        cur = meta.get(k)  # keep the most descriptive, then the newest, trade as the label
        if cur is None or (not cur.outcome and t.outcome) or (bool(t.outcome) == bool(cur.outcome) and t.timestamp > cur.timestamp):
            meta[k] = t

    assets_by_condition: dict[str, list[str]] = defaultdict(list)
    for cid, aid in flows:
        assets_by_condition[cid].append(aid)

    signals: list[Signal] = []
    for (cid, aid), wallets in flows.items():
        others = [a for a in assets_by_condition[cid] if a != aid]
        stance: dict[str, Decimal] = {}
        for w, f in wallets.items():
            stance[w] = f["net"]
        for a in others:
            for w, f in flows[(cid, a)].items():
                stance[w] = stance.get(w, Decimal(0)) - f["net"]
        agree = sorted(w for w, v in stance.items() if v >= min_leader_usd)
        against = sorted(w for w, v in stance.items() if v <= -min_leader_usd)
        m = meta[(cid, aid)]

        def make(side: str, yes: list[str], no: list[str]) -> Signal:
            usd_key, shares_key = ("buy_usd", "buy_shares") if side == "BUY" else ("sell_usd", "sell_shares")
            shares = sum((wallets[w][shares_key] for w in yes if w in wallets), Decimal(0))
            usd = sum((wallets[w][usd_key] for w in yes if w in wallets), Decimal(0))
            avg = (usd / shares) if shares else None
            net = sum((stance[w] for w in yes), Decimal(0))
            last = max(wallets[w]["last"] for w in yes if w in wallets) if any(w in wallets for w in yes) else max(
                flows[(cid, a)][w]["last"] for a in others for w in yes if w in flows[(cid, a)])
            return Signal(side, cid, aid, m.outcome, m.outcome_index, m.title, m.slug, m.event_slug,
                          yes, no, avg, net if side == "BUY" else -net, last)

        if len(agree) >= min_show:
            signals.append(make("BUY", agree, against))
        if len(against) >= min_show and _held(held, cid, aid, m.outcome):
            signals.append(make("SELL", against, agree))

    signals.sort(key=lambda s: (-s.score, -abs(s.net_usd), s.title))
    return signals
