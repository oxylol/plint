"""Features for the round-3 model, shared by the research builder and the bot so both compute them identically.

A trader's *record* is built from their finished positions scored at resolution
(the token's final price, 0 or 1), never at their own exit: a follower holds to
the end. It uses both the CLOSED and the REDEEMABLE sets (the API's CLOSED set
omits resolved losses), drops "No" positions in negative-risk markets (their
average price is not a usable cost basis) and positions under $1, and looks at
the newest RECORD_N positions whose market had resolved before the moment of
the decision. A *signal* is one leader's fresh net buy of an outcome, seen at
the moment the follower could act on it; its vector adds what the follower
would pay, how far the price has already moved from the leader's, the size of
the bet for this leader, the time to the market's end, the market type and
whether a game is in play.

Research (research/build3.py) and the bot (strategy_consensus.py) both call
`record()` and `vector()`; tests/test_features.py proves that a position
described by the API and the same position described by the research data give
the same numbers.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime

RECORD_N = 300          # newest resolved positions in a record (the bot fetches 500 CLOSED + 500 REDEEMABLE)
MIN_COST = 1.0          # positions with less money in them are noise
CATS = ["sports", "politics", "geopolitics", "crypto", "culture", "economics", "tech", "weather", "other"]
DAY = 86400.0


@dataclass(frozen=True)
class Pos:
    """One position, reduced to what a record needs. Times are UTC epoch seconds."""
    cost: float              # money in: total size × average price
    avg: float               # average price paid
    final: float | None      # the token's price at resolution (1 or 0); None while unresolved
    category: str
    first: float | None      # first entry
    last: float | None       # last activity (sale, redemption or last fill)
    resolved: float | None   # when the market resolved, if known (research); None = "already resolved" (bot, as of now)
    is_no: bool = False
    neg_risk: bool = False


def final_from_price(price: float | None) -> float | None:
    """A resolved token trades at 1 or 0; anything in between is not resolved yet."""
    if price is None:
        return None
    if price >= 0.99:
        return 1.0
    if price <= 0.01:
        return 0.0
    return None


def usable(p: Pos) -> bool:
    return p.final is not None and p.cost >= MIN_COST and 0 < p.avg < 1 and not (p.neg_risk and p.is_no)


def when(p: Pos) -> float | None:
    """When a position became part of the record: its resolution if known (research), else its last activity (bot, as of now)."""
    return p.resolved if p.resolved is not None else p.last


def record(positions: list[Pos], as_of: float, n_max: int = RECORD_N) -> dict:
    """A trader's record as of `as_of` (epoch seconds): positions resolved before then, newest `n_max` by last activity."""
    done = [p for p in positions if usable(p) and when(p) is not None and when(p) < as_of]
    done.sort(key=lambda p: -when(p))
    done = done[:n_max]
    n = len(done)
    out = {"n": n}
    if not n:
        return out
    wins = sum(p.final for p in done)
    pnl = [p.cost * (p.final / p.avg - 1.0) for p in done]
    edge = [p.final - p.avg for p in done]
    gw = sum(x for x in pnl if x > 0)
    gl = -sum(x for x in pnl if x < 0)
    cost = sum(p.cost for p in done)
    m_edge = sum(edge) / n
    sd = statistics.pstdev(edge) if n > 1 else 0.0
    holds = [max(0.0, (p.last - p.first) / 3600.0) for p in done if p.last is not None and p.first is not None]
    recent = [(p, x) for p, x in zip(done, pnl) if when(p) >= as_of - 30 * DAY]
    cats: dict[str, list] = {}
    for p in done:
        c = cats.setdefault(p.category, [0, 0.0, 0.0])
        c[0] += 1
        c[1] += p.final
        c[2] += p.final - p.avg
    out.update({
        "wins": wins, "wr": wins / n, "wr_shrunk": (wins + 2.0) / (n + 4.0),
        "pf": min(10.0, gw / gl) if gl > 0 else (10.0 if gw > 0 else 1.0),
        "roi": max(-1.0, min(3.0, sum(pnl) / cost)) if cost > 0 else 0.0,
        "edge": m_edge, "edge_t": max(-10.0, min(10.0, m_edge / (sd / math.sqrt(n)))) if sd > 0 else 0.0,
        "median_cost": statistics.median(p.cost for p in done),
        "median_price": statistics.median(p.avg for p in done),
        "median_hold_h": statistics.median(holds) if holds else None,
        "short_hold_share": (sum(1 for h in holds if h < 3) / len(holds)) if holds else None,
        "longshot_share": sum(1 for p in done if p.avg < 0.2) / n,
        "fav_share": sum(1 for p in done if p.avg > 0.8) / n,
        "n30": len(recent), "pnl30": sum(x for _, x in recent),
        "roi30": (sum(x for _, x in recent) / sum(p.cost for p, _ in recent)) if recent else 0.0,
        "cats": {k: tuple(v) for k, v in cats.items()},
    })
    return out


def _logit(p: float) -> float:
    p = min(0.995, max(0.005, p))
    return math.log(p / (1 - p))


def vector(rec: dict, *, price: float, leader_price: float | None, stance_usd: float, hours_to_end: float | None,
           category: str, is_no: bool, neg_risk: bool, n_fills: int, minutes_since_first: float,
           game_minutes: float | None, age_days: float | None, taker_share: float | None) -> dict[str, float]:
    """The model's inputs for one (leader, signal) at the moment the follower can act.

    price: what the follower pays now (the ask for the stake); leader_price: the leader's average buy price
    in the window; stance_usd: their net buy of this outcome minus the other side, in the window;
    game_minutes: minutes since a game started (negative before kick-off), None when not a game.
    """
    return {**price_terms(price, leader_price),
            **context(rec, leader_price=leader_price if leader_price is not None else price, stance_usd=stance_usd,
                      hours_to_end=hours_to_end, category=category, is_no=is_no, neg_risk=neg_risk, n_fills=n_fills,
                      minutes_since_first=minutes_since_first, game_minutes=game_minutes, age_days=age_days, taker_share=taker_share)}


def price_terms(price: float, leader_price: float | None) -> dict[str, float]:
    """The inputs that depend on what the follower pays (the research joins prices after building the rest)."""
    lp = leader_price if leader_price is not None else price
    return {"logit_p": _logit(price), "price": price, "drift": price - lp}


def context(rec: dict, *, leader_price: float, stance_usd: float, hours_to_end: float | None, category: str, is_no: bool,
            neg_risk: bool, n_fills: int, minutes_since_first: float, game_minutes: float | None, age_days: float | None,
            taker_share: float | None) -> dict[str, float]:
    """Everything else: the leader's bet, the market and the leader's record."""
    n = rec.get("n", 0)
    cats = rec.get("cats", {})
    cn, cw, ce = cats.get(category, (0, 0.0, 0.0))
    med_cost = rec.get("median_cost") or 0.0
    conv = stance_usd / med_cost if med_cost > 0 else 1.0
    x = {
        "leader_price": leader_price,
        "stance_log": math.log1p(max(0.0, stance_usd)), "conviction_log": math.log(max(0.05, min(20.0, conv))),
        "fills_log": math.log1p(max(0, n_fills)), "since_first_log": math.log1p(max(0.0, minutes_since_first)),
        "hours_log": math.log1p(max(0.0, hours_to_end)) if hours_to_end is not None else math.log1p(24 * 7),
        "is_no": 1.0 if is_no else 0.0, "neg_risk": 1.0 if neg_risk else 0.0,
        "in_game": 1.0 if (game_minutes is not None and game_minutes >= 0) else 0.0,
        "pre_game_h_log": math.log1p(max(0.0, -game_minutes) / 60) if (game_minutes is not None and game_minutes < 0) else 0.0,
        "age_log": math.log1p(max(0.0, age_days)) if age_days is not None else math.log1p(180),
        "taker_share": taker_share if taker_share is not None else 0.5,
        # the record
        "rec_n_log": math.log1p(n), "rec_wr": rec.get("wr_shrunk", 0.5), "rec_pf_log": math.log(max(0.1, rec.get("pf", 1.0))),
        "rec_roi": rec.get("roi", 0.0), "rec_edge": rec.get("edge", 0.0), "rec_edge_t": rec.get("edge_t", 0.0),
        "rec_price": rec.get("median_price", 0.5) if n else 0.5,
        "rec_hold_log": math.log1p(rec["median_hold_h"]) if rec.get("median_hold_h") is not None else math.log1p(24),
        "rec_short_hold": rec.get("short_hold_share") if rec.get("short_hold_share") is not None else 0.5,
        "rec_longshot": rec.get("longshot_share", 0.3) if n else 0.3, "rec_fav": rec.get("fav_share", 0.3) if n else 0.3,
        "rec_n30_log": math.log1p(rec.get("n30", 0)), "rec_roi30": max(-1.0, min(3.0, rec.get("roi30", 0.0))),
        "rec_pnl30_slog": math.copysign(math.log1p(abs(rec.get("pnl30", 0.0))), rec.get("pnl30", 0.0)),
        "rec_cost_log": math.log1p(med_cost),
        "cat_n_log": math.log1p(cn), "cat_share": cn / n if n else 0.0,
        "cat_wr": (cw + 1.0) / (cn + 2.0), "cat_edge": ce / cn if cn else 0.0,
    }
    for c in CATS[1:]:
        x[f"cat_{c}"] = 1.0 if category == c else 0.0
    return x


FEATURES = list(vector({}, price=0.5, leader_price=0.5, stance_usd=10, hours_to_end=1, category="other", is_no=False, neg_risk=False,
                       n_fills=1, minutes_since_first=0, game_minutes=None, age_days=1, taker_share=0.5))


LEAGUES = {"nba": "nba", "wnba": "nba", "nfl": "nfl", "cfb": "cfb", "ncaaf": "cfb", "cbb": "cbb", "ncaab": "cbb", "mlb": "mlb", "kbo": "mlb",
           "npb": "mlb", "nhl": "nhl", "ahl": "nhl", "atp": "tennis", "wta": "tennis", "lol": "esports", "cs2": "esports", "val": "esports",
           "dota2": "esports", "hok": "esports", "ufc": "fight", "mma": "fight", "box": "fight"}
SOCCER = {"epl", "ucl", "uel", "lal", "sea", "bun", "fl1", "mls", "ere", "el1", "elc", "por", "ned", "sco", "arg", "bra", "tur", "bel",
          "mex", "nor", "jap", "ja2", "kor", "chi", "col1", "usl", "spl", "acle", "itsb", "azec", "gtm"}
LEAGUE_KEYS = ["nba", "nfl", "cfb", "cbb", "mlb", "nhl", "soccer", "tennis", "esports", "fight"]


def league(slug: str) -> str:
    head = (slug or "").split("-")[0].lower()
    return "soccer" if head in SOCCER else LEAGUES.get(head, "other")


def market_kind(title: str, sports_type: str | None) -> str:
    st = (sports_type or "").lower()
    t = title or ""
    if "spread" in st or t.startswith("Spread"):
        return "spread"
    if "total" in st or "O/U" in t:
        return "total"
    return "moneyline" if (st == "moneyline" or " vs" in t) else "other"


def consensus_context(*, score: int, n_agree: int, n_against: int, agree_usd: float, leader_price: float | None, price: float,
                      hours_to_end: float | None, game_minutes: float | None, slug: str, title: str, sports_type: str | None) -> dict[str, float]:
    """Inputs for the consensus model: how strong the agreement is, what the follower pays against the leaders, the game's
    clock, the league and the kind of market. Research (consensus_model.py) and the bot (strategy_consensus.py) both call it."""
    lp = leader_price if leader_price is not None else price
    lg_, kind = league(slug), market_kind(title, sports_type)
    x = {
        "logit_p": _logit(price), "price": price, "drift": price - lp, "fav": 1.0 if price > 0.5 else 0.0,
        "score": float(score), "n_agree": float(n_agree), "n_against": float(n_against),
        "agree_usd_log": math.log1p(max(0.0, agree_usd)),
        "hours_log": math.log1p(max(0.0, hours_to_end)) if hours_to_end is not None else math.log1p(24 * 7),
        "in_play": 1.0 if (game_minutes is not None and game_minutes >= 0) else 0.0,
        "pre_game_h_log": math.log1p(max(0.0, -game_minutes) / 60) if (game_minutes is not None and game_minutes < 0) else 0.0,
        "spread": 1.0 if kind == "spread" else 0.0, "total": 1.0 if kind == "total" else 0.0,
    }
    for k in LEAGUE_KEYS:
        x[f"lg_{k}"] = 1.0 if lg_ == k else 0.0
    return x


def agree_quality(recs: list[dict]) -> dict[str, float]:
    """Who agrees: the agreeing leaders' records (features.record outputs), averaged over those with 5+ resolved
    positions. Research (consensus_model.py) and the bot compute it from the same records."""
    known = [r for r in recs if r.get("n", 0) >= 5]
    k = len(known)

    def mean(f, default):
        return sum(f(r) for r in known) / k if k else default

    def sports_wr(r):
        cn, cw, _ = r.get("cats", {}).get("sports", (0, 0.0, 0.0))
        return (cw + 1.0) / (cn + 2.0)

    return {"aq_known": float(k), "aq_edge": mean(lambda r: r.get("edge", 0.0), 0.0), "aq_edge_t": mean(lambda r: r.get("edge_t", 0.0), 0.0),
            "aq_best_edge_t": max((r.get("edge_t", 0.0) for r in known), default=0.0), "aq_wr": mean(lambda r: r.get("wr_shrunk", 0.5), 0.5),
            "aq_roi30": mean(lambda r: max(-1.0, min(3.0, r.get("roi30", 0.0))), 0.0), "aq_sports_wr": mean(sports_wr, 0.5)}


def leader_stats(fills, asset: str, now: float, lookback: float = 48 * 3600.0, min_usd: float = 10.0) -> dict:
    """One leader's bet on `asset`, from their fills in its market over (now − lookback, now].

    fills: (time, asset, side, size, price) with side +1 for a buy and −1 for a sale, for every outcome of the market
    (any order). stance is money into `asset` minus money into the other outcome(s), so buying both sides nets out;
    crossed_at is when the running stance first reached min_usd (the moment the bot's signal appears)."""
    win = sorted((f for f in fills if now - lookback < f[0] <= now), key=lambda f: f[0])
    stance = buy_usd = buy_sh = 0.0
    crossed = None
    for t, a, side, size, price in win:
        stance += side * size * price * (1 if a == asset else -1)
        if a == asset and side == 1:
            buy_usd += size * price
            buy_sh += size
        if crossed is None and stance >= min_usd:
            crossed = t
    return {"stance": stance, "leader_price": buy_usd / buy_sh if buy_sh else None, "n_fills": len(win),
            "minutes_since_first": (now - win[0][0]) / 60 if win else 0.0, "crossed_at": crossed}


def epoch(d: datetime | None) -> float | None:
    return d.timestamp() if d is not None else None
