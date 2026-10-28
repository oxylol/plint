"""Strategy 1, "consensus": buy what the leaderboard's top traders agree on."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Iterator

from .consensus import LeaderTrade, Signal, find_signals
from .core import CENT, Core, aware, short
from .store import iso

log = logging.getLogger("polybot.consensus")
AUTO_NAME = re.compile(r"^0x[0-9a-fA-F]{40}(-\d+)?$")  # Polymarket's placeholder for traders without a name


@dataclass
class Leader:
    wallet: str
    name: str
    rank: int
    pnl: Decimal
    volume: Decimal


class ConsensusStrategy:
    name = "consensus"
    PAGE_SIZE = 500
    MAX_PAGES_PER_LEADER = 3      # 1500 fills per leader per refresh is plenty
    PRICED_SIGNALS = 15           # how many signals get a live price on the page

    def __init__(self, core: Core):
        self.core = core
        self.cfg = core.cfg
        self.store = core.store
        self._trades: dict[str, dict[tuple, LeaderTrade]] = {}
        self._fetched_at: dict[str, datetime] = {}
        self.decisions: dict[str, str] = {}
        self.leaders: list[Leader] = []
        self.signals: list[Signal] = []
        self.fills = 0

    @property
    def interval(self) -> int:
        return self.cfg.poll_seconds

    # ── leaders and their trades ─────────────────────────────
    def fetch_leaders(self) -> list[Leader]:
        kwargs: dict = {"window": self.cfg.window, "sort_by": "PNL", "page_size": min(self.cfg.top_n, 1000)}
        if self.cfg.category and self.cfg.category != "overall":
            kwargs["category"] = self.cfg.category
        page = self.core.call(lambda: self.core.public.list_trader_leaderboard(**kwargs).first_page())
        leaders = []
        for e in page.items[: self.cfg.top_n]:
            w = str(e.wallet).lower()
            name = e.user_name if e.user_name and not AUTO_NAME.match(e.user_name) else short(w)
            leaders.append(Leader(w, name, int(e.rank), Decimal(str(e.pnl)), Decimal(str(e.volume))))
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

    def refresh_trades(self, leaders: list[Leader], since: datetime, now: datetime) -> dict[str, list[LeaderTrade]]:
        """Keep every leader's fills inside the lookback window, fetching only what's new."""
        keep = {l.wallet for l in leaders}
        for w in list(self._trades):
            if w not in keep:
                self._trades.pop(w, None)
                self._fetched_at.pop(w, None)
        for l in leaders:
            bucket = self._trades.setdefault(l.wallet, {})
            start = since
            last = self._fetched_at.get(l.wallet)
            if last:  # overlap a little: fills show up in the data API with some delay
                start = max(since, last - timedelta(minutes=15))
            try:
                for t in self._wallet_trades(l.wallet, start):
                    if t.timestamp >= since:
                        bucket[t.key] = t
                self._fetched_at[l.wallet] = now
            except Exception as e:
                self.core.warn(f"trades:{l.wallet}", f"could not fetch trades for {l.name}: {e}")
            for k, t in list(bucket.items()):
                if t.timestamp < since:
                    del bucket[k]
        return {w: list(b.values()) for w, b in self._trades.items()}

    # ── one cycle ────────────────────────────────────────────
    def run(self, now: datetime) -> None:
        cfg = self.cfg
        since = now - timedelta(hours=cfg.lookback_hours)
        self.leaders = self.fetch_leaders()
        trades_by_wallet = self.refresh_trades(self.leaders, since, now)
        self._trades_by_wallet = trades_by_wallet
        all_trades = [t for ts in trades_by_wallet.values() for t in ts]
        self.fills = len(all_trades)
        held: set = set()
        for p in self.store.open_positions(self.name):
            held.add(p["asset_id"])
            held.add((p.get("condition_id", ""), str(p.get("outcome", "")).lower()))
        self.signals = find_signals(all_trades, min_leader_usd=cfg.min_leader_usd, held=held)
        log.info("%d leaders · %d fills in the last %gh · %d outcomes with 2+ agreeing",
                 len(self.leaders), self.fills, cfg.lookback_hours, len(self.signals))

        self.decisions = {}
        for sig in self.signals:
            short_by = cfg.min_agree - sig.score
            try:
                if sig.side == "BUY":
                    if short_by > 0:
                        self.decisions[sig.key] = f"needs {short_by} more"
                    else:
                        self.consider_buy(sig, now)
                else:
                    pos = self.store.position(sig.asset_id) or self.store.find_position(sig.condition_id, sig.outcome)
                    if pos is None or pos.get("strategy", "consensus") != self.name:
                        self.decisions[sig.key] = "not holding"
                    elif not cfg.follow_sells:
                        self.decisions[sig.key] = "following sells is off"
                    elif short_by > 0:
                        self.decisions[sig.key] = f"needs {short_by} more"
                    else:
                        self.consider_sell(sig, pos, now)
            except Exception as e:  # one bad market must not stop the others
                log.exception("could not act on %s · %s", sig.title, sig.outcome)
                self.decisions[sig.key] = f"error: {e}"
                self.core.record_error(f"{sig.side} {sig.title} · {sig.outcome}: {e}")

    def consider_buy(self, sig: Signal, now: datetime) -> dict | None:
        cfg = self.cfg
        core = self.core
        what = f"{sig.title} · {sig.outcome}"

        def skip(reason: str) -> None:
            self.decisions[sig.key] = reason
            log.info("skip BUY %s: %s", what, reason)

        if self.store.position(sig.asset_id):
            return skip("already holding")
        last = self.store.acted_at(sig.asset_id)
        if last and now - last < timedelta(hours=cfg.cooldown_hours):
            return skip("cooling down after the last trade")
        if len(self.store.open_positions(self.name)) >= cfg.max_open:
            return skip("max open positions reached")
        if self.store.spent_since(now - timedelta(hours=24), self.name) + cfg.stake_usd > cfg.daily_budget_usd:
            return skip("daily budget used up")
        m = core.market(sig.condition_id, sig.slug)
        if m is None:
            return skip("market not found")
        why = core.not_tradeable(m, now, cfg.min_hours_to_end)
        if why:
            return skip(why)
        asset_id, label = core.resolve_asset(m, sig.asset_id, sig.outcome_index, sig.outcome)
        if self.store.position(asset_id):
            return skip("already holding")
        est = core.estimate(asset_id, "BUY", amount=cfg.stake_usd)
        if est is None:
            return skip("not enough liquidity")
        if est > cfg.max_price:
            return skip(f"price {est:.2f} is above the max {cfg.max_price}")
        if est < cfg.min_price:
            return skip(f"price {est:.2f} is below the min {cfg.min_price}")
        if sig.avg_price and est > sig.avg_price + cfg.max_above_leaders:
            return skip(f"too late: {est:.2f} now, leaders paid {sig.avg_price:.2f}")
        amount = cfg.stake_usd
        shares = (amount / est).quantize(CENT, rounding=ROUND_DOWN)
        min_size = Decimal(str(m.trading.minimum_order_size or 0))
        if shares < min_size:  # the book wants at least this many shares
            amount = (min_size * est * Decimal("1.02")).quantize(CENT, rounding=ROUND_UP)
            if self.store.spent_since(now - timedelta(hours=24), self.name) + amount > cfg.daily_budget_usd:
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
            "agree": len(sig.agree), "disagree": len(sig.disagree),
            "leaders_price": float(sig.avg_price) if sig.avg_price is not None else None,
            "mode": core.mode, **fill,
        }
        self.store.open_position(rec, now)
        verb = "Bought" if core.mode == "live" else "Paper buy"
        self.decisions[sig.key] = "bought" if core.mode == "live" else "paper bought"
        log.info("%s %s: %s shares at %.2f ($%s) · %d agree, %d against",
                 verb, what, rec["shares"], rec["price"], rec["cost"], len(sig.agree), len(sig.disagree))
        core.notifier.send(f"Polybot: {verb.lower()}",
                           f"[consensus] {what}\n{rec['shares']:.2f} shares at {rec['price']:.2f} (${rec['cost']:.2f})\n"
                           f"{len(sig.agree)} leaders agree, {len(sig.disagree)} against",
                           tags="chart_with_upwards_trend")
        return rec

    def consider_sell(self, sig: Signal, pos: dict, now: datetime) -> dict | None:
        core = self.core
        asset_id = pos["asset_id"]
        what = f"{pos.get('title')} · {pos.get('outcome')}"

        def skip(reason: str) -> None:
            self.decisions[sig.key] = reason
            log.info("skip SELL %s: %s", what, reason)

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
        rec = self.store.close_position(asset_id, Decimal(str(fill["price"])), Decimal(str(fill["proceeds"])), "leaders sold", now, mode=core.mode)
        verb = "Sold" if core.mode == "live" else "Paper sell"
        self.decisions[sig.key] = "sold" if core.mode == "live" else "paper sold"
        log.info("%s %s: %s shares at %.2f, P&L $%.2f", verb, what, shares, fill["price"], rec["pnl"])
        core.notifier.send(f"Polybot: {verb.lower()}",
                           f"[consensus] {what}\n{shares:.2f} shares at {fill['price']:.2f}, P&L ${rec['pnl']:+.2f}\n"
                           f"{len(sig.agree)} leaders sold, {len(sig.disagree)} against",
                           tags="chart_with_downwards_trend")
        return rec

    # ── the page ─────────────────────────────────────────────
    def status(self) -> dict:
        core = self.core
        names = {l.name: l.name for l in self.leaders}
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
                "score": s.score, "avg_price": float(s.avg_price) if s.avg_price is not None else None,
                "net_usd": float(s.net_usd), "last_at": iso(s.last_at),
                "price": float(price) if price is not None else None,
                "decision": decision, "closed": closed,
                "holding": self.store.position(s.asset_id) is not None,
            })
        rows.sort(key=lambda r: (r["closed"], -r["score"]))
        trades_by_wallet = getattr(self, "_trades_by_wallet", {})
        return {
            "leaders": [{"rank": l.rank, "name": l.name, "wallet": l.wallet, "pnl": float(l.pnl), "volume": float(l.volume),
                         "fills": len(trades_by_wallet.get(l.wallet, []))} for l in self.leaders],
            "fills": self.fills,
            "signals": rows,
        }
