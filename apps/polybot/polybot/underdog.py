"""Pure rules for the underdog strategy: which side to buy, and when to get out."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_UP, ROUND_DOWN
from typing import Any


@dataclass
class Candidate:
    condition_id: str
    slug: str
    event_slug: str
    title: str
    sport: str
    game_start: datetime
    asset_id: str        # trading id of the underdog outcome
    outcome: str
    price: Decimal       # Gamma's current price for the underdog
    liquidity: Decimal
    kind: str            # "teams" (two named sides) or "yesno" (one team's own market)
    event_id: str = ""   # groups the Yes/No markets of one match
    is_draw: bool = False


def sport_of(slug: str) -> str:
    return (slug or "").split("-")[0].lower()


WILL_WIN = re.compile(r"^will (.+?) win\b", re.IGNORECASE)


def team_from_question(question: str) -> str:
    """'Will Boyacá Chicó FC win on 2026-10-28?' → 'Boyacá Chicó FC'."""
    m = WILL_WIN.match(question or "")
    return m.group(1).strip() if m else ""


def pick_underdog(m: Any, *, min_price: Decimal, max_price: Decimal, trading_id) -> Candidate | None:
    """The cheaper side of a moneyline market, if it is priced like an underdog.

    Team-labelled markets (Lakers / Trail Blazers) use the cheaper side. Yes/No
    markets ("Will X win?", "Draw?") use the Yes side only, so we buy a team,
    never "not the favourite".
    """
    sports = getattr(m, "sports", None)
    if sports is None or sports.sports_market_type != "moneyline" or sports.game_start_time is None:
        return None
    yes, no = m.outcomes.yes, m.outcomes.no
    if yes.price is None or no.price is None:
        return None
    labels = {(yes.label or "").lower(), (no.label or "").lower()}
    if labels & {"yes", "no"}:
        kind, dog = "yesno", yes
    else:
        kind, dog = "teams", (yes if Decimal(str(yes.price)) <= Decimal(str(no.price)) else no)
    price = Decimal(str(dog.price))
    if not (min_price <= price <= max_price):
        return None
    asset = trading_id(m, dog)
    if not asset:
        return None
    event = m.events[0] if getattr(m, "events", None) else None
    title = (event.title if event and event.title else None) or m.question or m.slug or ""
    outcome = dog.label or ""
    is_draw = False
    if kind == "yesno":
        q = m.question or ""
        team = team_from_question(q)
        is_draw = "draw" in q.lower() or (m.slug or "").endswith("-draw")
        outcome = "Draw" if is_draw else (team or "Yes")
        if not (event and event.title):
            title = q or title
    return Candidate(
        condition_id=str(m.condition_id), slug=m.slug or "", event_slug=(event.slug if event else "") or "",
        title=title, sport=sport_of(m.slug), game_start=sports.game_start_time, asset_id=str(asset),
        outcome=outcome, price=price,
        liquidity=Decimal(str(m.metrics.liquidity_num or m.metrics.liquidity or 0)), kind=kind,
        event_id=str(event.id) if event and getattr(event, "id", None) else (event.slug if event else "") or "",
        is_draw=is_draw,
    )


def one_per_match(cands: list[Candidate]) -> list[Candidate]:
    """Soccer-style matches come as one Yes/No market per team plus a draw market.

    Keep the cheaper team only when it is strictly cheaper than the other team
    (an even match has no underdog), never the draw. Team-labelled markets are
    one per match already.
    """
    out: list[Candidate] = []
    by_match: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.kind != "yesno":
            out.append(c)
        elif not c.is_draw:
            by_match.setdefault(c.event_id or c.event_slug or c.slug, []).append(c)
    for group in by_match.values():
        teams = sorted(group, key=lambda c: c.price)
        if len(teams) == 1:
            out.append(teams[0])           # the other team's market is not in band: this one is the dog
        elif teams[0].price < teams[1].price:
            out.append(teams[0])
    return out


def in_buy_window(game_start: datetime, now: datetime, before_min: float, until_min: float) -> bool:
    return game_start - timedelta(minutes=before_min) <= now <= game_start + timedelta(minutes=until_min)


def target_price(entry: Decimal, take_profit_pct: Decimal, tick: Decimal) -> Decimal:
    """Entry plus the take-profit margin, rounded up to the market's tick."""
    raw = entry * (1 + take_profit_pct / 100)
    steps = (raw / tick).to_integral_value(rounding=ROUND_UP)
    return min(Decimal("0.99"), steps * tick)


def stop_price(entry: Decimal, stop_loss_pct: Decimal, tick: Decimal) -> Decimal:
    raw = entry * (1 - stop_loss_pct / 100)
    steps = (raw / tick).to_integral_value(rounding=ROUND_DOWN)
    return max(tick, steps * tick)


def exit_reason(pos: dict, bid: Decimal | None, now: datetime) -> str | None:
    """Why an underdog position should be sold now, or None to keep holding."""
    deadline = pos.get("deadline")
    if deadline and now >= deadline:
        return "time"
    if bid is None:
        return None
    if bid >= Decimal(str(pos["target"])):
        return "target"
    if bid <= Decimal(str(pos["stop"])):
        return "stop"
    return None
