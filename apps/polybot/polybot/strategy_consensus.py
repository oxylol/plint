"""Strategy 1, "consensus": buy what the leaderboard's top traders agree on.

Several *variants* of it can run at once (POLY_VARIANTS): a different
leaderboard window, threshold, size of the pack, and so on. They share one
LeaderFeed, so a trader who is on two boards is fetched once, and each keeps
its own paper book so the page can compare them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Iterator

from .config import Config, WINDOWS
from .consensus import LeaderTrade, Signal, find_signals
from .core import CENT, Core, aware, pretty_name, short
from .store import iso

log = logging.getLogger("polybot.consensus")


@dataclass(frozen=True)
class ConsensusParams:
    name: str
    label: str
    window: str
    category: str
    top_n: int
    min_agree: int
    lookback_hours: float
    min_leader_usd: Decimal
    stake_usd: Decimal
    max_price: Decimal
    min_price: Decimal
    max_above_leaders: Decimal
    max_open: int
    daily_budget_usd: Decimal
    cooldown_hours: float
    min_hours_to_end: float
    follow_sells: bool
    poll_seconds: int
    # credible-pack options (pack=1 setups)
    pack: bool = False           # take the leaders from the credible pack instead of the raw board
    pool: int = 80               # how many of the board the pack measures
    min_weight: float = 0.0      # credibility × conviction, agree minus against, needed to act (0 = ignore)
    fresh_only: bool = False     # only act when an agreeing leader's position is new
    adaptive_trust: bool = False # trust a leader more or less as their copied trades resolve
    max_hold_days: float = 0.0   # sell what is still open after this long (0 = hold to resolution)

    @classmethod
    def base(cls, cfg: Config) -> "ConsensusParams":
        return cls(name="consensus", label=f"Consensus · {cfg.window}", window=cfg.window, category=cfg.category,
                   top_n=cfg.top_n, min_agree=cfg.min_agree, lookback_hours=cfg.lookback_hours,
                   min_leader_usd=cfg.min_leader_usd, stake_usd=cfg.stake_usd, max_price=cfg.max_price,
                   min_price=cfg.min_price, max_above_leaders=cfg.max_above_leaders, max_open=cfg.max_open,
                   daily_budget_usd=cfg.daily_budget_usd, cooldown_hours=cfg.cooldown_hours,
                   min_hours_to_end=cfg.min_hours_to_end, follow_sells=cfg.follow_sells, poll_seconds=cfg.poll_seconds,
                   pool=cfg.cr_pool)

    def variant(self, name: str, overrides: dict[str, str]) -> "ConsensusParams":
        """A copy with some settings changed (values arrive as strings from .env)."""
        types = {f.name: f.type for f in fields(self)}
        changes: dict = {}
        for key, raw in overrides.items():
            if key == "label":
                changes["label"] = raw
                continue
            t = types[key]
            if t in ("int",):
                changes[key] = int(raw)
            elif t in ("float",):
                changes[key] = float(raw)
            elif t in ("Decimal",):
                changes[key] = Decimal(raw)
            elif t in ("bool",):
                changes[key] = raw.strip().lower() in ("1", "true", "yes", "on")
            else:
                changes[key] = raw.strip().lower()
        p = replace(self, name=name, **changes)
        if "label" not in overrides:
            bits = [("credible " if p.pack else "") + p.window]
            if p.category and p.category != "overall":
                bits.append(p.category)
            if p.top_n != self.top_n:
                bits.append(f"top {p.top_n}")
            if p.min_agree != self.min_agree:
                bits.append(f"{p.min_agree} agree")
            if p.lookback_hours != self.lookback_hours:
                bits.append(f"{p.lookback_hours:g} h")
            p = replace(p, label="Consensus · " + " · ".join(bits))
        p.validate()
        return p

    def validate(self) -> None:
        if self.window not in WINDOWS:
            raise ValueError(f"variant {self.name}: window must be one of {WINDOWS}, not {self.window!r}")
        if not (1 <= self.top_n <= 1000) or self.min_agree < 1 or self.stake_usd < 1:
            raise ValueError(f"variant {self.name}: top_n 1–1000, min_agree ≥ 1, stake ≥ 1")
        if not (0 < self.min_price < self.max_price <= 1):
            raise ValueError(f"variant {self.name}: need 0 < min_price < max_price <= 1")
        if self.poll_seconds < 30:
            raise ValueError(f"variant {self.name}: poll_seconds must be at least 30")

    def public(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = float(v) if isinstance(v, Decimal) else v
        return out


@dataclass
class Leader:
    wallet: str
    name: str
    rank: int
    pnl: Decimal
    volume: Decimal


class LeaderFeed:
    """Leaderboards and the leaders' fills, fetched once and shared by every variant."""
    PAGE_SIZE = 500
    MAX_PAGES_PER_LEADER = 3      # 1500 fills per leader per refresh is plenty

    def __init__(self, core: Core, keep_hours: float, board_ttl: int):
        self.core = core
        self.keep = timedelta(hours=keep_hours)
        self.board_ttl = timedelta(seconds=board_ttl)
        self._boards: dict[tuple, tuple[datetime, list[Leader]]] = {}
        self._trades: dict[str, dict[tuple, LeaderTrade]] = {}
        self._fetched_at: dict[str, datetime] = {}
        self._wanted_at: dict[str, datetime] = {}

    def leaders(self, window: str, category: str, top_n: int, now: datetime) -> list[Leader]:
        key = (window, category, top_n)
        hit = self._boards.get(key)
        if hit and now - hit[0] < self.board_ttl:
            return hit[1]
        kwargs: dict = {"window": window, "sort_by": "PNL", "page_size": min(top_n, 1000)}
        if category and category != "overall":
            kwargs["category"] = category
        page = self.core.call(lambda: self.core.public.list_trader_leaderboard(**kwargs).first_page())
        leaders = []
        for e in page.items[:top_n]:
            w = str(e.wallet).lower()
            leaders.append(Leader(w, pretty_name(e.user_name, w), int(e.rank), Decimal(str(e.pnl)), Decimal(str(e.volume))))
        self._boards[key] = (now, leaders)
        return leaders

    @staticmethod
    def _to_trade(t) -> LeaderTrade:
        return LeaderTrade(
            wallet=str(t.wallet).lower(), condition_id=str(t.condition_id), asset_id=str(t.asset_id),
            side=str(t.side).upper(), size=Decimal(str(t.size)), price=Decimal(str(t.price)),
            timestamp=aware(t.timestamp), title=t.title or "", slug=t.slug or "", event_slug=t.event_slug or "",
            outcome=t.outcome or "", outcome_index=t.outcome_index, tx=str(t.transaction_hash or ""),
        )

    def _wallet_trades(self, wallet: str, start: datetime) -> Iterator[LeaderTrade]:
        pages = self.core.public.list_trades(user=wallet, start=start, taker_only=False, page_size=self.PAGE_SIZE)
        page = self.core.call(pages.first_page)
        n = 0
        while True:
            for t in page.items:
                yield self._to_trade(t)
            n += 1
            if not page.next_cursor or n >= self.MAX_PAGES_PER_LEADER:
                return
            page = self.core.call(pages.from_cursor(page.next_cursor).first_page)

    def trades(self, leaders: list[Leader], since: datetime, now: datetime) -> dict[str, list[LeaderTrade]]:
        """Each leader's fills since `since`, fetching only what is new."""
        keep_since = now - self.keep
        out: dict[str, list[LeaderTrade]] = {}
        for l in leaders:
            self._wanted_at[l.wallet] = now
            bucket = self._trades.setdefault(l.wallet, {})
            start = keep_since
            last = self._fetched_at.get(l.wallet)
            if last:  # overlap a little: fills show up in the data API with some delay
                start = max(keep_since, last - timedelta(minutes=15))
            if last is None or now - last >= timedelta(seconds=60):
                try:
                    for t in self._wallet_trades(l.wallet, start):
                        if t.timestamp >= keep_since:
                            bucket[t.key] = t
                    self._fetched_at[l.wallet] = now
                except Exception as e:
                    self.core.warn(f"trades:{l.wallet}", f"could not fetch trades for {l.name}: {e}")
            for k, t in list(bucket.items()):
                if t.timestamp < keep_since:
                    del bucket[k]
            out[l.wallet] = [t for t in bucket.values() if t.timestamp >= since]
        # Forget wallets no board has wanted for a while.
        for w, at in list(self._wanted_at.items()):
            if now - at > timedelta(hours=1):
                self._wanted_at.pop(w, None)
                self._trades.pop(w, None)
                self._fetched_at.pop(w, None)
        return out


class ConsensusStrategy:
    kind = "consensus"
    PRICED_SIGNALS = 15           # how many signals get a live price on the page
    MUTED_TRUST = 0.5             # adaptive trust below this silences a leader

    def __init__(self, core: Core, params: ConsensusParams, feed: LeaderFeed, pack=None):
        self.core = core
        self.p = params
        self.store = core.store
        self.feed = feed
        self.pack = pack          # a CrediblePack when params.pack
        self.decisions: dict[str, str] = {}
        self._prev_decisions: dict[str, str] = {}
        self.leaders: list[Leader] = []
        self.signals: list[Signal] = []
        self.fills = 0
        self.weights: dict[str, float] = {}
        self.muted: list[str] = []
        self.note: str = ""
        self._trades_by_wallet: dict[str, list[LeaderTrade]] = {}

    @property
    def name(self) -> str:
        return self.p.name

    @property
    def label(self) -> str:
        return self.p.label

    @property
    def interval(self) -> int:
        return self.p.poll_seconds

    # ── one cycle ────────────────────────────────────────────
    def pick_leaders(self, now: datetime) -> tuple[list[Leader], dict[str, float] | None, dict[str, float] | None]:
        """The raw board, or the credible pack with vote weights and typical sizes."""
        p = self.p
        if self.pack is None:
            return self.feed.leaders(p.window, p.category, p.top_n, now), None, None
        self.pack.ensure_fresh(now)
        members, weights, sizes = self.pack.snapshot()
        leaders = [Leader(m.wallet, m.name, m.rank, Decimal(str(m.record.pnl)), Decimal(str(m.record.volume))) for m in members]
        self.muted = []
        if p.adaptive_trust:
            weights = {w: self.store.trust(self.name, w, default) for w, default in weights.items()}
            self.muted = [w for w, t in weights.items() if t < self.MUTED_TRUST]
            leaders = [l for l in leaders if l.wallet not in self.muted]
        self.weights = weights
        return leaders, weights, sizes

    def enough(self, sig: Signal) -> str | None:
        """Why a signal is not strong enough yet, or None when it is."""
        p = self.p
        if sig.score < p.min_agree:
            return f"needs {p.min_agree - sig.score} more"
        if p.min_weight > 0 and sig.weight < p.min_weight:
            return f"weight {sig.weight:.1f} of {p.min_weight:g}"
        return None

    def run(self, now: datetime) -> None:
        p = self.p
        since = now - timedelta(hours=p.lookback_hours)
        self.leaders, weights, sizes = self.pick_leaders(now)
        if self.pack is not None and not self.leaders:
            self.note = "measuring the leaderboard, back in a few minutes" if self.pack.building or self.pack.refreshed_at is None else "nobody on the board passed the credibility rules"
            log.info("[%s] %s", self.name, self.note)
            self.signals, self.fills = [], 0
            self.expire_positions(now)
            return
        self.note = ""
        self._trades_by_wallet = self.feed.trades(self.leaders, since, now)
        all_trades = [t for ts in self._trades_by_wallet.values() for t in ts]
        self.fills = len(all_trades)
        held: set = set()
        for pos in self.store.open_positions(self.name):
            held.add(pos["asset_id"])
            held.add((pos.get("condition_id", ""), str(pos.get("outcome", "")).lower()))
        self.signals = find_signals(all_trades, min_leader_usd=p.min_leader_usd, held=held, weights=weights, sizes=sizes)
        log.info("[%s] %d leaders · %d fills in the last %gh · %d outcomes with 2+ agreeing",
                 self.name, len(self.leaders), self.fills, p.lookback_hours, len(self.signals))

        self.expire_positions(now)
        self._prev_decisions = self.decisions
        self.decisions = {}
        for sig in self.signals:
            why = self.enough(sig)
            try:
                if sig.side == "BUY":
                    if why:
                        self.decisions[sig.key] = why
                    else:
                        self.consider_buy(sig, now)
                else:
                    pos = self.store.position(sig.asset_id, self.name) or self.store.find_position(sig.condition_id, sig.outcome, self.name)
                    if pos is None:
                        self.decisions[sig.key] = "not holding"
                    elif not p.follow_sells:
                        self.decisions[sig.key] = "following sells is off"
                    elif why:
                        self.decisions[sig.key] = why
                    else:
                        self.consider_sell(sig, pos, now)
            except Exception as e:  # one bad market must not stop the others
                log.exception("[%s] could not act on %s · %s", self.name, sig.title, sig.outcome)
                self.decisions[sig.key] = f"error: {e}"
                self.core.record_error(f"{self.name}: {sig.side} {sig.title} · {sig.outcome}: {e}")

    def expire_positions(self, now: datetime) -> None:
        """Sell what has been open longer than max_hold_days (0 = hold to resolution)."""
        if self.p.max_hold_days <= 0:
            return
        from .store import parse_iso
        for pos in self.store.open_positions(self.name):
            opened = parse_iso(pos["opened_at"]) if pos.get("opened_at") else now
            if now - opened < timedelta(days=self.p.max_hold_days):
                continue
            shares = Decimal(str(pos.get("shares", 0)))
            est = self.core.estimate(pos["asset_id"], "SELL", shares=shares)
            if est is None:
                continue
            m = self.core.market(pos.get("condition_id", ""), pos.get("slug", ""))
            fill = self.core.sell(pos["asset_id"], shares, est, f"{pos.get('title')} · {pos.get('outcome')}",
                                  fee_rate=self.core.fee_rate(m) if m else Decimal(0))
            if fill:
                rec = self.store.close_position(pos["asset_id"], Decimal(str(fill["price"])), Decimal(str(fill["proceeds"])),
                                                "time's up", now, mode=self.core.mode, strategy=self.name)
                log.info("[%s] sold %s after %g days, P&L $%.2f", self.name, pos.get("title"), self.p.max_hold_days, rec["pnl"])

    def fresh_voters(self, sig: Signal) -> list[str]:
        """Agreeing leaders whose position in this outcome is (mostly) new in the window."""
        fresh = []
        for w in sig.agree:
            bought = Decimal(str(sig.stance.get(w, 0)))
            if bought <= 0:
                continue
            cost_now = self.core.position_cost(w, sig.condition_id, sig.asset_id)
            if cost_now is None:
                continue
            if cost_now - bought <= bought * Decimal("0.25"):  # at most a quarter was there before
                fresh.append(w)
        return fresh

    def _skip(self, sig: Signal, what: str, reason: str) -> None:
        self.decisions[sig.key] = reason
        if self._prev_decisions.get(sig.key) != reason:  # say it once, not every five minutes
            log.info("[%s] skip %s %s: %s", self.name, sig.side, what, reason)

    def consider_buy(self, sig: Signal, now: datetime) -> dict | None:
        p = self.p
        core = self.core
        what = f"{sig.title} · {sig.outcome}"
        skip = lambda reason: self._skip(sig, what, reason)  # noqa: E731

        if self.store.position(sig.asset_id, self.name):
            return skip("already holding")
        last = self.store.acted_at(sig.asset_id, self.name)
        if last and now - last < timedelta(hours=p.cooldown_hours):
            return skip("cooling down after the last trade")
        if len(self.store.open_positions(self.name)) >= p.max_open:
            return skip("max open positions reached")
        if self.store.spent_since(now - timedelta(hours=24), self.name) + p.stake_usd > p.daily_budget_usd:
            return skip("daily budget used up")
        m = core.market(sig.condition_id, sig.slug)
        if m is None:
            return skip("market not found")
        why = core.not_tradeable(m, now, p.min_hours_to_end)
        if why:
            return skip(why)
        asset_id, label = core.resolve_asset(m, sig.asset_id, sig.outcome_index, sig.outcome)
        if self.store.position(asset_id, self.name):
            return skip("already holding")
        est = core.estimate(asset_id, "BUY", amount=p.stake_usd)
        if est is None:
            return skip("not enough liquidity")
        if est > p.max_price:
            return skip(f"price {est:.2f} is above the max {p.max_price}")
        if est < p.min_price:
            return skip(f"price {est:.2f} is below the min {p.min_price}")
        if sig.avg_price and est > sig.avg_price + p.max_above_leaders:
            return skip(f"too late: {est:.2f} now, leaders paid {sig.avg_price:.2f}")
        fresh: list[str] = []
        if p.fresh_only:
            fresh = self.fresh_voters(sig)
            if not fresh:
                return skip("no fresh entry: the leaders held this already")
        amount = p.stake_usd
        shares = (amount / est).quantize(CENT, rounding=ROUND_DOWN)
        min_size = Decimal(str(m.trading.minimum_order_size or 0))
        if shares < min_size:  # the book wants at least this many shares
            amount = (min_size * est * Decimal("1.02")).quantize(CENT, rounding=ROUND_UP)
            if self.store.spent_since(now - timedelta(hours=24), self.name) + amount > p.daily_budget_usd:
                return skip(f"minimum order (${amount}) does not fit the daily budget")
        if core.mode == "live" and core.balance is not None and core.balance < amount:
            core.warn("funds", f"not enough pUSD for a ${amount} order (balance ${core.balance:.2f})")
            return skip("not enough pUSD in the wallet")

        fill = core.buy(asset_id, amount, est, what, fee_rate=core.fee_rate(m))
        if fill is None:
            return skip("order rejected")
        rec = {
            "strategy": self.name, "side": "BUY", "asset_id": asset_id, "condition_id": sig.condition_id,
            "title": sig.title or m.question or "", "outcome": label or sig.outcome,
            "slug": sig.slug or m.slug or "", "event_slug": sig.event_slug or "",
            "agree": len(sig.agree), "disagree": len(sig.disagree), "weight": sig.weight,
            "agree_wallets": list(sig.agree), "agree_names": {w: self.name_of(w) for w in sig.agree},
            "fresh": [self.name_of(w) for w in fresh],
            "leaders_price": float(sig.avg_price) if sig.avg_price is not None else None,
            "mode": core.mode, **fill,
        }
        self.store.open_position(rec, now)
        verb = "Bought" if core.mode == "live" else "Paper buy"
        self.decisions[sig.key] = "bought" if core.mode == "live" else "paper bought"
        log.info("[%s] %s %s: %s shares at %.2f ($%s) · %d agree, %d against, weight %.1f",
                 self.name, verb, what, rec["shares"], rec["price"], rec["cost"], len(sig.agree), len(sig.disagree), sig.weight)
        core.notifier.send(f"Polybot: {verb.lower()}",
                           f"[{self.label}] {what}\n{rec['shares']:.2f} shares at {rec['price']:.2f} (${rec['cost']:.2f})\n"
                           f"{len(sig.agree)} leaders agree, {len(sig.disagree)} against",
                           tags="chart_with_upwards_trend")
        return rec

    def consider_sell(self, sig: Signal, pos: dict, now: datetime) -> dict | None:
        core = self.core
        asset_id = pos["asset_id"]
        what = f"{pos.get('title')} · {pos.get('outcome')}"
        skip = lambda reason: self._skip(sig, what, reason)  # noqa: E731

        shares = Decimal(str(pos.get("shares", 0)))
        if shares <= 0:
            return skip("nothing to sell")
        est = core.estimate(asset_id, "SELL", shares=shares)
        if est is None:
            return skip("not enough liquidity")
        m = core.market(pos.get("condition_id", ""), pos.get("slug", ""))
        fill = core.sell(asset_id, shares, est, what, fee_rate=core.fee_rate(m) if m else Decimal(0))
        if fill is None:
            return skip("order rejected")
        rec = self.store.close_position(asset_id, Decimal(str(fill["price"])), Decimal(str(fill["proceeds"])), "leaders sold", now,
                                        mode=core.mode, strategy=self.name)
        verb = "Sold" if core.mode == "live" else "Paper sell"
        self.decisions[sig.key] = "sold" if core.mode == "live" else "paper sold"
        log.info("[%s] %s %s: %s shares at %.2f, P&L $%.2f", self.name, verb, what, shares, fill["price"], rec["pnl"])
        core.notifier.send(f"Polybot: {verb.lower()}",
                           f"[{self.label}] {what}\n{shares:.2f} shares at {fill['price']:.2f}, P&L ${rec['pnl']:+.2f}\n"
                           f"{len(sig.agree)} leaders sold, {len(sig.disagree)} against",
                           tags="chart_with_downwards_trend")
        return rec

    def name_of(self, wallet: str) -> str:
        for l in self.leaders:
            if l.wallet == wallet:
                return l.name
        return short(wallet)

    # ── the page ─────────────────────────────────────────────
    def status(self) -> dict:
        core = self.core
        by_wallet = {l.wallet: l.name for l in self.leaders}
        rows = []
        priced = 0
        for s in self.signals[:40]:
            decision = self.decisions.get(s.key, "")
            price = None
            closed = False
            try:
                m = core.market(s.condition_id, s.slug)
            except Exception as e:
                log.warning("could not look up %s: %s", s.title, e)
                m = None
            if m is not None and (m.state.closed or m.state.archived):
                closed = True
                for o in (m.outcomes.yes, m.outcomes.no):
                    if s.asset_id in {str(o.token_id or ""), str(o.position_id or "")} and o.price is not None:
                        price = Decimal(str(o.price))
                if not decision or decision.startswith("needs"):
                    decision = "market closed"
            elif priced < self.PRICED_SIGNALS:
                priced += 1
                price = core.midpoint(s.asset_id)
            rows.append({
                "side": s.side, "title": s.title, "outcome": s.outcome, "slug": s.slug, "event_slug": s.event_slug,
                "condition_id": s.condition_id, "asset_id": s.asset_id,
                "agree": [by_wallet.get(w, short(w)) for w in s.agree],
                "disagree": [by_wallet.get(w, short(w)) for w in s.disagree],
                "score": s.score, "weight": s.weight, "avg_price": float(s.avg_price) if s.avg_price is not None else None,
                "net_usd": float(s.net_usd), "last_at": iso(s.last_at),
                "price": float(price) if price is not None else None,
                "decision": decision, "closed": closed,
                "holding": self.store.position(s.asset_id, self.name) is not None,
            })
        rows.sort(key=lambda r: (r["closed"], -r["score"], -r["weight"]))
        leaders = [{"rank": l.rank, "name": l.name, "wallet": l.wallet, "pnl": float(l.pnl), "volume": float(l.volume),
                    "fills": len(self._trades_by_wallet.get(l.wallet, [])),
                    "weight": round(self.weights.get(l.wallet, 1.0), 2) if self.weights else None} for l in self.leaders]
        out = {
            "name": self.name, "label": self.label, "params": self.p.public(), "note": self.note,
            "leaders": leaders, "fills": self.fills, "signals": rows,
            "followed": self.store.leaders_followed(self.name)[:15],
        }
        if self.pack is not None:
            out["pack"] = self.pack.key
            out["muted"] = [self.name_of(w) for w in self.muted]
        return out
