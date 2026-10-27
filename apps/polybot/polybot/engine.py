"""The engine: leaders → their recent trades → consensus → (paper) trades.

Everything that talks to Polymarket lives here. Public data comes from
PublicClient (no key needed); orders go through SecureClient when a key is
configured and POLY_LIVE=1. Without that, the same decisions are paper trades.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Iterator

from polymarket import (
    InsufficientLiquidityError,
    PolymarketError,
    PublicClient,
    RateLimitError,
    RelayerApiKey,
    SecureClient,
    TransportError,
)

from .config import Config
from .consensus import LeaderTrade, Signal, find_signals
from .notify import Notifier
from .store import Store, iso, utcnow

log = logging.getLogger("polybot")
CENT = Decimal("0.01")


@dataclass
class Leader:
    wallet: str
    name: str
    rank: int
    pnl: Decimal
    volume: Decimal


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _short(wallet: str) -> str:
    return wallet[:6] + "…" + wallet[-4:] if len(wallet) > 12 else wallet


class Engine:
    PAGE_SIZE = 500
    MAX_PAGES_PER_LEADER = 3      # 1500 fills per leader per refresh is plenty
    MARKET_TTL = timedelta(minutes=10)
    PRICED_SIGNALS = 15           # how many signals get a live price on the page

    def __init__(self, cfg: Config, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.public = PublicClient()
        self.secure: SecureClient | None = None
        self.account: dict = {}
        self._trades: dict[str, dict[tuple, LeaderTrade]] = {}
        self._fetched_at: dict[str, datetime] = {}
        self._markets: dict[str, tuple[datetime, object]] = {}
        self._warned: dict[str, datetime] = {}
        self.decisions: dict[str, str] = {}
        self.errors: list[dict] = []
        self.balance: Decimal | None = None

    # ── plumbing ─────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return "live" if (self.cfg.live and self.secure is not None) else "paper"

    def _call(self, fn, *args, **kwargs):
        """Call the SDK with a little patience for rate limits and hiccups."""
        delay = 5
        for _ in range(3):
            try:
                return fn(*args, **kwargs)
            except RateLimitError as e:
                log.warning("rate limited (%s); waiting %ss", e, delay)
            except TransportError as e:
                log.warning("transport error (%s); retrying in %ss", e, delay)
            time.sleep(delay)
            delay *= 2
        return fn(*args, **kwargs)

    def _warn(self, key: str, msg: str, every: timedelta = timedelta(hours=6)) -> None:
        """Log every time, record and notify at most once per `every`."""
        log.warning(msg)
        last = self._warned.get(key)
        if last and utcnow() - last < every:
            return
        self._warned[key] = utcnow()
        self.record_error(msg, notify=True)

    def record_error(self, msg: str, notify: bool = False) -> None:
        self.errors.append({"at": iso(utcnow()), "message": msg})
        self.errors = self.errors[-10:]
        if notify:
            self.notifier.send("Polybot problem", msg, priority="high", tags="warning")

    def connect(self) -> None:
        """Open the trading client when a key is configured (any mode)."""
        if not self.cfg.can_trade:
            return
        try:
            kwargs = {}
            if self.cfg.relayer_key and self.cfg.relayer_address:
                kwargs["api_key"] = RelayerApiKey(key=self.cfg.relayer_key, address=self.cfg.relayer_address)
            self.secure = SecureClient.create(private_key=self.cfg.private_key, wallet=self.cfg.wallet, **kwargs)
            self.account = {"wallet": str(self.secure.wallet), "wallet_type": str(self.secure.wallet_type)}
            log.info("connected to Polymarket as %s (%s)", self.secure.wallet, self.secure.wallet_type)
            self.refresh_balance()
        except Exception as e:
            self.secure = None
            self.record_error(f"could not connect with the configured key: {e}", notify=True)

    def refresh_balance(self) -> Decimal | None:
        if self.secure is None:
            return None
        try:
            ba = self._call(self.secure.get_balance_allowance, asset_type="COLLATERAL")
            self.balance = Decimal(ba.balance) / Decimal(10**6)
        except Exception as e:
            self._warn("balance", f"could not read the pUSD balance: {e}")
        return self.balance

    def approvals(self) -> dict:
        """What `polybot check` prints: is the account ready to trade?"""
        out: dict = {"mode": self.mode, **self.account}
        if self.secure is None:
            out["note"] = "no key configured (paper only)" if not self.cfg.can_trade else "connection failed"
            return out
        out["balance_pusd"] = float(self.refresh_balance() or 0)
        try:
            st = self._call(self.secure.get_trading_approvals_state)
            out["approved"] = bool(st.is_fully_approved)
            if not st.is_fully_approved:
                out["missing_approvals"] = str(st.missing)
        except Exception as e:
            out["approvals_error"] = str(e)
        try:
            out["gasless_ready"] = bool(self.secure.is_gasless_ready())
        except Exception as e:
            out["gasless_ready"] = False
            out["gasless_error"] = str(e)
        return out

    # ── leaders and their trades ─────────────────────────────
    def fetch_leaders(self) -> list[Leader]:
        kwargs: dict = {"window": self.cfg.window, "sort_by": "PNL", "page_size": min(self.cfg.top_n, 1000)}
        if self.cfg.category and self.cfg.category != "overall":
            kwargs["category"] = self.cfg.category
        page = self._call(lambda: self.public.list_trader_leaderboard(**kwargs).first_page())
        leaders = []
        for e in page.items[: self.cfg.top_n]:
            w = str(e.wallet).lower()
            leaders.append(Leader(w, e.user_name or _short(w), int(e.rank), Decimal(str(e.pnl)), Decimal(str(e.volume))))
        return leaders

    @staticmethod
    def _to_trade(t) -> LeaderTrade:
        return LeaderTrade(
            wallet=str(t.wallet).lower(), condition_id=str(t.condition_id), asset_id=str(t.asset_id),
            side=str(t.side).upper(), size=Decimal(str(t.size)), price=Decimal(str(t.price)),
            timestamp=_aware(t.timestamp), title=t.title or "", slug=t.slug or "", event_slug=t.event_slug or "",
            outcome=t.outcome or "", outcome_index=t.outcome_index, tx=str(t.transaction_hash or ""),
        )

    def _wallet_trades(self, wallet: str, start: datetime) -> Iterator[LeaderTrade]:
        pages = self.public.list_trades(user=wallet, start=start, taker_only=False, page_size=self.PAGE_SIZE)
        page = self._call(pages.first_page)
        n = 0
        while True:
            for t in page.items:
                yield self._to_trade(t)
            n += 1
            if not page.next_cursor or n >= self.MAX_PAGES_PER_LEADER:
                return
            page = self._call(pages.from_cursor(page.next_cursor).first_page)

    def refresh_trades(self, leaders: list[Leader], since: datetime) -> dict[str, list[LeaderTrade]]:
        """Keep every leader's fills inside the lookback window, fetching only what's new."""
        keep = {l.wallet for l in leaders}
        for w in list(self._trades):
            if w not in keep:
                self._trades.pop(w, None)
                self._fetched_at.pop(w, None)
        now = utcnow()
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
                self._warn(f"trades:{l.wallet}", f"could not fetch trades for {l.name}: {e}")
            for k, t in list(bucket.items()):
                if t.timestamp < since:
                    del bucket[k]
        return {w: list(b.values()) for w, b in self._trades.items()}

    # ── markets ──────────────────────────────────────────────
    def market(self, condition_id: str, slug: str = ""):
        """The Gamma market, by slug first: the condition-id listing only knows open markets."""
        key = condition_id or slug
        hit = self._markets.get(key)
        if hit and utcnow() - hit[0] < self.MARKET_TTL:
            return hit[1]
        m = None
        if slug:
            try:
                m = self._call(self.public.get_market, slug=slug)
            except PolymarketError as e:
                log.debug("get_market(slug=%s): %s", slug, e)
        if m is None and condition_id:
            page = self._call(lambda: self.public.list_markets(condition_ids=[condition_id], page_size=1).first_page())
            m = page.items[0] if page.items else None
        self._markets[key] = (utcnow(), m)
        return m

    def _not_tradeable(self, m, now: datetime) -> str | None:
        st = m.state
        if st.closed or st.archived:
            return "market is closed"
        if st.active is False or st.accepting_orders is False or st.enable_order_book is False:
            return "market is not accepting orders"
        if st.end_date and _aware(st.end_date) - now < timedelta(hours=self.cfg.min_hours_to_end):
            return "market ends too soon"
        return None

    @staticmethod
    def _resolve_asset(m, sig: Signal) -> tuple[str, str]:
        """The trading identifier for the signal's outcome on this market, and its label."""
        outcomes = [m.outcomes.yes, m.outcomes.no]
        for o in outcomes:
            if sig.asset_id in {str(o.token_id or ""), str(o.position_id or "")}:
                trade_id = (o.position_id if m.version == "v2" else o.token_id) or sig.asset_id
                return str(trade_id), o.label or sig.outcome
        if sig.outcome_index in (0, 1):
            o = outcomes[sig.outcome_index]
            trade_id = (o.position_id if m.version == "v2" else o.token_id) or sig.asset_id
            return str(trade_id), o.label or sig.outcome
        return sig.asset_id, sig.outcome

    def _estimate(self, asset_id: str, side: str, *, amount: Decimal | None = None, shares: Decimal | None = None) -> Decimal | None:
        try:
            kwargs = {"amount": str(amount)} if amount is not None else {"shares": str(shares)}
            return Decimal(str(self._call(self.public.estimate_market_price, asset_id=asset_id, side=side, order_type="FAK", **kwargs)))
        except InsufficientLiquidityError:
            return None
        except PolymarketError as e:  # no order book (closed market) and the like
            log.info("no price for %s: %s", asset_id[:12], e)
            return None

    def _midpoint(self, asset_id: str) -> Decimal | None:
        try:
            return Decimal(str(self._call(self.public.get_midpoint, asset_id=asset_id)))
        except Exception:
            return None

    # ── one cycle ────────────────────────────────────────────
    def run_once(self) -> dict:
        cfg = self.cfg
        now = utcnow()
        since = now - timedelta(hours=cfg.lookback_hours)
        leaders = self.fetch_leaders()
        trades_by_wallet = self.refresh_trades(leaders, since)
        all_trades = [t for ts in trades_by_wallet.values() for t in ts]
        held: set = set()
        for p in self.store.open_positions():
            held.add(p["asset_id"])
            held.add((p.get("condition_id", ""), str(p.get("outcome", "")).lower()))
        signals = find_signals(all_trades, min_leader_usd=cfg.min_leader_usd, held=held)
        log.info("%d leaders · %d fills in the last %gh · %d outcomes with 2+ agreeing",
                 len(leaders), len(all_trades), cfg.lookback_hours, len(signals))

        self.decisions = {}
        for sig in signals:
            short = cfg.min_agree - sig.score
            try:
                if sig.side == "BUY":
                    if short > 0:
                        self.decisions[sig.key] = f"needs {short} more"
                    else:
                        self.consider_buy(sig, now)
                else:
                    pos = self.store.position(sig.asset_id) or self.store.find_position(sig.condition_id, sig.outcome)
                    if pos is None:
                        self.decisions[sig.key] = "not holding"
                    elif not cfg.follow_sells:
                        self.decisions[sig.key] = "following sells is off"
                    elif short > 0:
                        self.decisions[sig.key] = f"needs {short} more"
                    else:
                        self.consider_sell(sig, pos, now)
            except Exception as e:  # one bad market must not stop the others
                log.exception("could not act on %s · %s", sig.title, sig.outcome)
                self.decisions[sig.key] = f"error: {e}"
                self.record_error(f"{sig.side} {sig.title} · {sig.outcome}: {e}")

        self.settle_positions(now)
        self.mark_positions()
        if self.mode == "live":
            self.refresh_balance()
            if cfg.auto_redeem:
                self.redeem()
        status = self.build_status(leaders, trades_by_wallet, signals, now)
        self.store.save()
        self.store.write_status(status)
        return status

    def consider_buy(self, sig: Signal, now: datetime) -> dict | None:
        cfg = self.cfg
        what = f"{sig.title} · {sig.outcome}"

        def skip(reason: str) -> None:
            self.decisions[sig.key] = reason
            log.info("skip BUY %s: %s", what, reason)

        if self.store.position(sig.asset_id):
            return skip("already holding")
        last = self.store.acted_at(sig.asset_id)
        if last and now - last < timedelta(hours=cfg.cooldown_hours):
            return skip("cooling down after the last trade")
        if len(self.store.open_positions()) >= cfg.max_open:
            return skip("max open positions reached")
        if self.store.spent_since(now - timedelta(hours=24)) + cfg.stake_usd > cfg.daily_budget_usd:
            return skip("daily budget used up")
        m = self.market(sig.condition_id, sig.slug)
        if m is None:
            return skip("market not found")
        why = self._not_tradeable(m, now)
        if why:
            return skip(why)
        asset_id, label = self._resolve_asset(m, sig)
        if self.store.position(asset_id):
            return skip("already holding")
        est = self._estimate(asset_id, "BUY", amount=cfg.stake_usd)
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
            shares = (amount / est).quantize(CENT, rounding=ROUND_DOWN)
            if self.store.spent_since(now - timedelta(hours=24)) + amount > cfg.daily_budget_usd:
                return skip(f"minimum order (${amount}) does not fit the daily budget")
        if self.mode == "live" and self.balance is not None and self.balance < amount:
            self._warn("funds", f"not enough pUSD for a ${amount} order (balance ${self.balance:.2f})")
            return skip("not enough pUSD in the wallet")

        rec = {
            "side": "BUY", "asset_id": asset_id, "condition_id": sig.condition_id,
            "title": sig.title or m.question or "", "outcome": label or sig.outcome,
            "slug": sig.slug or m.slug or "", "event_slug": sig.event_slug or "",
            "agree": len(sig.agree), "disagree": len(sig.disagree),
            "leaders_price": float(sig.avg_price) if sig.avg_price is not None else None,
            "mode": self.mode,
        }
        if self.mode == "live":
            resp = self._call(self.secure.place_market_order, asset_id=asset_id, side="BUY",
                              amount=str(amount), max_price=est, order_type="FAK")
            if not resp.ok:
                self.record_error(f"buy rejected for {what}: {resp.code} {resp.message}", notify=True)
                return skip(f"order rejected: {resp.code}")
            cost = Decimal(str(resp.making_amount)) if resp.making_amount else amount
            got = Decimal(str(resp.taking_amount)) if resp.taking_amount else shares
            price = (cost / got).quantize(Decimal("0.0001")) if got else est
            rec.update(order_id=str(resp.order_id), status=str(resp.status), cost=float(cost),
                       shares=float(got), price=float(price))
            if resp.status == "delayed":
                rec["note"] = "delayed: amounts are estimates until it matches"
        else:
            rec.update(cost=float(amount), shares=float(shares), price=float(est))

        self.store.open_position(rec, now)
        verb = "Bought" if self.mode == "live" else "Paper buy"
        self.decisions[sig.key] = "bought" if self.mode == "live" else "paper bought"
        log.info("%s %s: %s shares at %.2f ($%s) · %d agree, %d against",
                 verb, what, rec["shares"], rec["price"], rec["cost"], len(sig.agree), len(sig.disagree))
        self.notifier.send(f"Polybot: {verb.lower()}",
                           f"{what}\n{rec['shares']:.2f} shares at {rec['price']:.2f} (${rec['cost']:.2f})\n"
                           f"{len(sig.agree)} leaders agree, {len(sig.disagree)} against",
                           tags="chart_with_upwards_trend")
        return rec

    def consider_sell(self, sig: Signal, pos: dict, now: datetime) -> dict | None:
        asset_id = pos["asset_id"]
        what = f"{pos.get('title')} · {pos.get('outcome')}"

        def skip(reason: str) -> None:
            self.decisions[sig.key] = reason
            log.info("skip SELL %s: %s", what, reason)

        shares = Decimal(str(pos.get("shares", 0)))
        if shares <= 0:
            return skip("nothing to sell")
        est = self._estimate(asset_id, "SELL", shares=shares)
        if est is None:
            return skip("not enough liquidity")
        if self.mode == "live":
            resp = self._call(self.secure.place_market_order, asset_id=asset_id, side="SELL",
                              shares=str(shares), min_price=est, order_type="FAK")
            if not resp.ok:
                self.record_error(f"sell rejected for {what}: {resp.code} {resp.message}", notify=True)
                return skip(f"order rejected: {resp.code}")
            proceeds = Decimal(str(resp.taking_amount)) if resp.taking_amount else shares * est
            price = (proceeds / shares).quantize(Decimal("0.0001"))
        else:
            price = est
            proceeds = (shares * est).quantize(CENT)
        rec = self.store.close_position(asset_id, price, proceeds, "leaders sold", now, mode=self.mode)
        verb = "Sold" if self.mode == "live" else "Paper sell"
        self.decisions[sig.key] = "sold" if self.mode == "live" else "paper sold"
        log.info("%s %s: %s shares at %.2f, P&L $%.2f", verb, what, shares, price, rec["pnl"])
        self.notifier.send(f"Polybot: {verb.lower()}",
                           f"{what}\n{shares:.2f} shares at {price:.2f}, P&L ${rec['pnl']:+.2f}\n"
                           f"{len(sig.agree)} leaders sold, {len(sig.disagree)} against",
                           tags="chart_with_downwards_trend")
        return rec

    def settle_positions(self, now: datetime) -> None:
        """Close book entries for markets that have resolved (1 for the winner, 0 for the rest)."""
        for pos in list(self.store.open_positions()):
            try:
                m = self.market(pos["condition_id"], pos.get("slug", ""))
            except Exception as e:
                log.warning("could not check %s: %s", pos.get("title"), e)
                continue
            if m is None or not m.state.closed:
                continue
            price = None
            for o in (m.outcomes.yes, m.outcomes.no):
                if pos["asset_id"] in {str(o.token_id or ""), str(o.position_id or "")}:
                    price = o.price
            if price is None:
                self.store.note_position(pos["asset_id"], "market closed, waiting for resolution")
                continue
            price = Decimal(str(price))
            if price >= Decimal("0.99"):
                final = Decimal(1)
            elif price <= Decimal("0.01"):
                final = Decimal(0)
            else:
                self.store.note_position(pos["asset_id"], "market closed, waiting for resolution")
                continue
            shares = Decimal(str(pos.get("shares", 0)))
            rec = self.store.close_position(pos["asset_id"], final, shares * final, "resolved", now)
            won = final == 1
            self.notifier.send("Polybot: market resolved",
                               f"{pos.get('title')} · {pos.get('outcome')}\n"
                               f"{'Won' if won else 'Lost'}: P&L ${rec['pnl']:+.2f}"
                               + ("\nWinnings are claimed automatically." if self.mode == "live" and self.cfg.auto_redeem and self.cfg.relayer_key else
                                  ("\nClaim them on polymarket.com." if self.mode == "live" and won else "")),
                               tags="tada" if won else "x")

    def mark_positions(self) -> None:
        for pos in self.store.open_positions():
            mid = self._midpoint(pos["asset_id"])
            if mid is not None:
                self.store.update_price(pos["asset_id"], mid)

    def redeem(self) -> None:
        """Claim winnings for resolved markets (needs a Relayer API key for gasless calls)."""
        if self.secure is None:
            return
        if not (self.cfg.relayer_key and self.cfg.relayer_address):
            self._warn("redeem", "there are winnings to claim but no Relayer API key is set; claim them on polymarket.com",
                       every=timedelta(days=1)) if self._has_redeemable() else None
            return
        try:
            page = self._call(lambda: self.secure.list_positions(user=self.cfg.wallet, status="REDEEMABLE", page_size=50).first_page())
        except Exception as e:
            self._warn("redeem-list", f"could not list redeemable positions: {e}")
            return
        done = set()
        for p in page.items:
            cid = str(p.condition_id)
            if cid in done:
                continue
            done.add(cid)
            try:
                tx = self._call(self.secure.redeem_positions, condition_id=cid)
                tx.wait()
                log.info("redeemed %s · %s", p.title, p.outcome)
                self.notifier.send("Polybot: winnings claimed", f"{p.title} · {p.outcome}\n${float(p.current_value or 0):.2f} back in the wallet", tags="moneybag")
            except Exception as e:
                self._warn(f"redeem:{cid}", f"could not redeem {p.title}: {e}")

    def _has_redeemable(self) -> bool:
        try:
            page = self._call(lambda: self.secure.list_positions(user=self.cfg.wallet, status="REDEEMABLE", page_size=1).first_page())
            return bool(page.items)
        except Exception:
            return False

    # ── the page ─────────────────────────────────────────────
    def build_status(self, leaders: list[Leader], trades_by_wallet: dict[str, list[LeaderTrade]],
                     signals: list[Signal], now: datetime) -> dict:
        cfg = self.cfg
        names = {l.wallet: l.name for l in leaders}
        rows = []
        priced = 0
        for s in signals[:40]:
            decision = self.decisions.get(s.key, "")
            price = None
            closed = False
            try:
                m = self.market(s.condition_id, s.slug)
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
                price = self._midpoint(s.asset_id)
            rows.append({
                "side": s.side, "title": s.title, "outcome": s.outcome, "slug": s.slug, "event_slug": s.event_slug,
                "condition_id": s.condition_id, "asset_id": s.asset_id,
                "agree": [names.get(w, _short(w)) for w in s.agree],
                "disagree": [names.get(w, _short(w)) for w in s.disagree],
                "score": s.score, "avg_price": float(s.avg_price) if s.avg_price is not None else None, "net_usd": float(s.net_usd),
                "last_at": iso(s.last_at), "price": float(price) if price is not None else None,
                "decision": decision, "closed": closed,
                "holding": self.store.position(s.asset_id) is not None,
            })
        rows.sort(key=lambda r: (r["closed"], -r["score"]))
        positions = []
        for p in self.store.open_positions():
            shares = Decimal(str(p.get("shares", 0)))
            cost = Decimal(str(p.get("cost", 0)))
            cur = Decimal(str(p.get("current_price") or p.get("price") or 0))
            value = shares * cur
            positions.append({**p, "value": float(value), "pnl": float(value - cost)})
        positions.sort(key=lambda p: p.get("opened_at", ""), reverse=True)
        return {
            "updated_at": iso(now),
            "next_run_at": iso(now + timedelta(seconds=cfg.poll_seconds)),
            "mode": self.mode,
            "config": cfg.public(),
            "account": {**self.account, "balance": float(self.balance) if self.balance is not None else None},
            "leaders": [{"rank": l.rank, "name": l.name, "wallet": l.wallet, "pnl": float(l.pnl), "volume": float(l.volume),
                         "fills": len(trades_by_wallet.get(l.wallet, []))} for l in leaders],
            "fills": sum(len(v) for v in trades_by_wallet.values()),
            "signals": rows,
            "positions": positions,
            "trades": self.store.recent_trades(30),
            "totals": self.store.totals(now),
            "errors": self.errors[-5:],
        }
