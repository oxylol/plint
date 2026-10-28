"""Everything that talks to Polymarket, shared by the strategies.

Public data comes from PublicClient (no key needed); orders go through
SecureClient when a key is configured and POLY_LIVE=1. Without that the same
calls become paper trades at the prices the order book shows right now.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN

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
from .notify import Notifier
from .store import Store, iso, utcnow

log = logging.getLogger("polybot.core")
CENT = Decimal("0.01")


def aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def short(wallet: str) -> str:
    return wallet[:6] + "…" + wallet[-4:] if len(wallet) > 12 else wallet


AUTO_NAME = re.compile(r"^0x[0-9a-fA-F]{40}(-\d+)?$")  # Polymarket's placeholder for traders without a name


def pretty_name(user_name: str | None, wallet: str) -> str:
    return user_name if user_name and not AUTO_NAME.match(user_name) else short(wallet)


class Core:
    MARKET_TTL = timedelta(minutes=10)

    def __init__(self, cfg: Config, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.public = PublicClient()
        self.secure: SecureClient | None = None
        self.account: dict = {}
        self.balance: Decimal | None = None
        self._markets: dict[str, tuple[datetime, object]] = {}
        self._warned: dict[str, datetime] = {}
        self.errors: list[dict] = []

    # ── plumbing ─────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return "live" if (self.cfg.live and self.secure is not None) else "paper"

    def call(self, fn, *args, **kwargs):
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

    def warn(self, key: str, msg: str, every: timedelta = timedelta(hours=6)) -> None:
        """Log every time; record and notify at most once per `every`."""
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
            ba = self.call(self.secure.get_balance_allowance, asset_type="COLLATERAL")
            self.balance = Decimal(ba.balance) / Decimal(10**6)
        except Exception as e:
            self.warn("balance", f"could not read the pUSD balance: {e}")
        return self.balance

    def approvals(self) -> dict:
        """What `polybot check` prints: is the account ready to trade?"""
        out: dict = {"mode": self.mode, **self.account}
        if self.secure is None:
            out["note"] = "no key configured (paper only)" if not self.cfg.can_trade else "connection failed"
            return out
        out["balance_pusd"] = float(self.refresh_balance() or 0)
        try:
            st = self.call(self.secure.get_trading_approvals_state)
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

    # ── markets and prices ───────────────────────────────────
    def market(self, condition_id: str, slug: str = ""):
        """The Gamma market, by slug first: the condition-id listing only knows open markets."""
        key = condition_id or slug
        hit = self._markets.get(key)
        if hit and utcnow() - hit[0] < self.MARKET_TTL:
            return hit[1]
        m = None
        if slug:
            try:
                m = self.call(self.public.get_market, slug=slug)
            except PolymarketError as e:
                log.debug("get_market(slug=%s): %s", slug, e)
        if m is None and condition_id:
            page = self.call(lambda: self.public.list_markets(condition_ids=[condition_id], page_size=1).first_page())
            m = page.items[0] if page.items else None
        self._markets[key] = (utcnow(), m)
        return m

    def remember_market(self, m) -> None:
        if m is not None and m.condition_id:
            self._markets[str(m.condition_id)] = (utcnow(), m)

    @staticmethod
    def not_tradeable(m, now: datetime, min_hours_to_end: float = 0) -> str | None:
        st = m.state
        if st.closed or st.archived:
            return "market is closed"
        if st.active is False or st.accepting_orders is False or st.enable_order_book is False:
            return "market is not accepting orders"
        if min_hours_to_end and st.end_date and aware(st.end_date) - now < timedelta(hours=min_hours_to_end):
            return "market ends too soon"
        return None

    @staticmethod
    def trading_id(m, outcome) -> str | None:
        tid = outcome.position_id if m.version == "v2" else outcome.token_id
        return str(tid) if tid else None

    @classmethod
    def resolve_asset(cls, m, asset_id: str, outcome_index: int | None = None, outcome: str = "") -> tuple[str, str]:
        """The trading identifier for an outcome on this market, and its label."""
        outcomes = [m.outcomes.yes, m.outcomes.no]
        for o in outcomes:
            if asset_id in {str(o.token_id or ""), str(o.position_id or "")}:
                return cls.trading_id(m, o) or asset_id, o.label or outcome
        if outcome_index in (0, 1):
            o = outcomes[outcome_index]
            return cls.trading_id(m, o) or asset_id, o.label or outcome
        for o in outcomes:
            if outcome and o.label and o.label.lower() == outcome.lower():
                return cls.trading_id(m, o) or asset_id, o.label
        return asset_id, outcome

    def estimate(self, asset_id: str, side: str, *, amount: Decimal | None = None, shares: Decimal | None = None) -> Decimal | None:
        """The price a market order would reach right now, or None when the book can't take it."""
        try:
            kwargs = {"amount": str(amount)} if amount is not None else {"shares": str(shares)}
            return Decimal(str(self.call(self.public.estimate_market_price, asset_id=asset_id, side=side, order_type="FAK", **kwargs)))
        except InsufficientLiquidityError:
            return None
        except PolymarketError as e:  # no order book (closed market) and the like
            log.info("no price for %s: %s", asset_id[:12], e)
            return None

    def midpoint(self, asset_id: str) -> Decimal | None:
        try:
            return Decimal(str(self.call(self.public.get_midpoint, asset_id=asset_id)))
        except Exception:
            return None

    def book(self, asset_id: str) -> dict | None:
        """Best bid and ask, the tick size and the minimum order size."""
        try:
            ob = self.call(self.public.get_order_book, asset_id=asset_id)
        except Exception as e:
            log.info("no order book for %s: %s", asset_id[:12], e)
            return None
        bids = [Decimal(str(l.price)) for l in ob.bids]
        asks = [Decimal(str(l.price)) for l in ob.asks]
        return {
            "bid": max(bids) if bids else None, "ask": min(asks) if asks else None,
            "tick": Decimal(str(ob.tick_size)), "min_size": Decimal(str(ob.min_order_size)),
        }

    def price_high_since(self, asset_id: str, since: datetime, until: datetime) -> Decimal | None:
        """The highest traded price in a window (minute buckets); None when there were no trades."""
        try:
            page = self.call(lambda: self.public.list_price_history(asset_id=asset_id, start=since, end=until, bucket_seconds=60).first_page())
        except Exception as e:
            log.info("no price history for %s: %s", asset_id[:12], e)
            return None
        prices = [Decimal(str(p.price)) for p in page.items if aware(p.timestamp) >= since]
        return max(prices) if prices else None

    # ── orders (paper or live) ───────────────────────────────
    @staticmethod
    def fee_rate(m) -> Decimal:
        """The market's taker fee rate (0 when it charges none)."""
        try:
            tr = m.trading
            if not tr.fees_enabled or tr.fee_schedule is None:
                return Decimal(0)
            return Decimal(str(tr.fee_schedule.rate or 0))
        except Exception:
            return Decimal(0)

    @staticmethod
    def taker_fee(shares: Decimal, price: Decimal, rate: Decimal) -> Decimal:
        """Polymarket's taker fee: shares × rate × p × (1 − p), in USD. Makers pay nothing."""
        if rate <= 0:
            return Decimal(0)
        return (shares * rate * price * (1 - price)).quantize(Decimal("0.0001"))

    def buy(self, asset_id: str, amount: Decimal, est: Decimal, what: str, fee_rate: Decimal = Decimal(0)) -> dict | None:
        """Spend `amount` on an outcome at or below `est`. Returns the fill, or None.

        Paper fills pay the market's taker fee too, so paper P&L is not rosier
        than the real thing.
        """
        shares = (amount / est).quantize(CENT, rounding=ROUND_DOWN)
        if self.mode != "live":
            fee = self.taker_fee(shares, est, fee_rate)
            if fee:
                shares = ((amount - fee) / est).quantize(CENT, rounding=ROUND_DOWN)
            return {"cost": float(amount), "shares": float(shares), "price": float(est), "fee": float(fee)}
        resp = self.call(self.secure.place_market_order, asset_id=asset_id, side="BUY",
                         amount=str(amount), max_price=est, order_type="FAK")
        if not resp.ok:
            self.record_error(f"buy rejected for {what}: {resp.code} {resp.message}", notify=True)
            return None
        cost = Decimal(str(resp.making_amount)) if resp.making_amount else amount
        got = Decimal(str(resp.taking_amount)) if resp.taking_amount else shares
        price = (cost / got).quantize(Decimal("0.0001")) if got else est
        fill = {"cost": float(cost), "shares": float(got), "price": float(price),
                "order_id": str(resp.order_id), "status": str(resp.status)}
        if resp.status == "delayed":
            fill["note"] = "delayed: amounts are estimates until it matches"
        return fill

    def sell(self, asset_id: str, shares: Decimal, est: Decimal, what: str, fee_rate: Decimal = Decimal(0)) -> dict | None:
        """Sell `shares` at or above `est` with a market order. Returns proceeds and price, or None."""
        if self.mode != "live":
            fee = self.taker_fee(shares, est, fee_rate)
            return {"proceeds": float((shares * est - fee).quantize(CENT)), "price": float(est), "fee": float(fee)}
        resp = self.call(self.secure.place_market_order, asset_id=asset_id, side="SELL",
                         shares=str(shares), min_price=est, order_type="FAK")
        if not resp.ok:
            self.record_error(f"sell rejected for {what}: {resp.code} {resp.message}", notify=True)
            return None
        proceeds = Decimal(str(resp.taking_amount)) if resp.taking_amount else shares * est
        price = (proceeds / shares).quantize(Decimal("0.0001")) if shares else est
        return {"proceeds": float(proceeds), "price": float(price), "order_id": str(resp.order_id), "status": str(resp.status)}

    def limit_sell(self, asset_id: str, price: Decimal, size: Decimal, what: str) -> dict | None:
        """Rest a sell order on the book (live only). Returns the order id and status."""
        if self.mode != "live":
            return None
        resp = self.call(self.secure.place_limit_order, asset_id=asset_id, side="SELL", price=str(price), size=str(size))
        if not resp.ok:
            self.record_error(f"resting sell rejected for {what}: {resp.code} {resp.message}", notify=True)
            return None
        return {"order_id": str(resp.order_id), "status": str(resp.status),
                "matched_shares": float(resp.making_amount or 0), "matched_usd": float(resp.taking_amount or 0)}

    def order_state(self, order_id: str) -> dict | None:
        """How much of a resting order has matched; None when the CLOB no longer knows it."""
        if self.secure is None:
            return None
        try:
            o = self.call(self.secure.get_order, order_id=order_id)
        except PolymarketError as e:
            log.info("order %s: %s", order_id[:10], e)
            return None
        return {"matched": Decimal(str(o.size_matched)), "original": Decimal(str(o.original_size)),
                "price": Decimal(str(o.price)), "status": str(o.status)}

    def cancel(self, order_id: str) -> bool:
        if self.secure is None:
            return False
        try:
            resp = self.call(self.secure.cancel_order, order_id=order_id)
            return order_id in set(resp.canceled) or not resp.not_canceled
        except PolymarketError as e:
            log.warning("cancel %s: %s", order_id[:10], e)
            return False

    def held_shares(self, condition_id: str, asset_id: str) -> Decimal | None:
        """What the wallet actually holds of an outcome (live accounts)."""
        if not self.cfg.wallet:
            return None
        try:
            page = self.call(lambda: self.public.list_positions(user=self.cfg.wallet, condition_id=[condition_id], page_size=10).first_page())
        except Exception as e:
            log.info("positions for %s: %s", condition_id[:10], e)
            return None
        for p in page.items:
            if str(p.asset_id) == asset_id:
                return Decimal(str(p.current_size))
        return Decimal(0)

    def position_cost(self, wallet: str, condition_id: str, asset_id: str) -> Decimal | None:
        """What a trader currently has in an outcome, at cost (USD); None when unknown."""
        try:
            page = self.call(lambda: self.public.list_positions(user=wallet, condition_id=[condition_id], page_size=10).first_page())
        except Exception as e:
            log.info("positions of %s in %s: %s", short(wallet), condition_id[:10], e)
            return None
        for p in page.items:
            if str(p.asset_id) == asset_id:
                return Decimal(str(p.total_cost_usdc or p.entry_cost_usdc or 0))
        return Decimal(0)

    # ── bookkeeping shared by every strategy ─────────────────
    def settle(self, now: datetime, positions: list[dict]) -> None:
        """Close book entries for markets that have resolved (1 for the winner, 0 for the rest)."""
        for pos in positions:
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
            strategy = pos.get("strategy", "consensus")
            if price is None:
                self.store.note_position(pos["asset_id"], "market closed, waiting for resolution", strategy)
                continue
            price = Decimal(str(price))
            if price >= Decimal("0.99"):
                final = Decimal(1)
            elif price <= Decimal("0.01"):
                final = Decimal(0)
            else:
                self.store.note_position(pos["asset_id"], "market closed, waiting for resolution", strategy)
                continue
            shares = Decimal(str(pos.get("shares", 0)))
            rec = self.store.close_position(pos["asset_id"], final, shares * final, "resolved", now, strategy=strategy)
            if rec is None:
                continue
            won = final == 1
            claim = ""
            if self.mode == "live" and won:
                claim = "\nWinnings are claimed automatically." if (self.cfg.auto_redeem and self.cfg.relayer_key) else "\nClaim them on polymarket.com."
            self.notifier.send("Polybot: market resolved",
                               f"[{pos.get('strategy', 'consensus')}] {pos.get('title')} · {pos.get('outcome')}\n"
                               f"{'Won' if won else 'Lost'}: P&L ${rec['pnl']:+.2f}{claim}",
                               tags="tada" if won else "x")

    def mark(self, positions: list[dict]) -> None:
        done: set[str] = set()
        for pos in positions:
            if pos["asset_id"] in done:
                continue
            done.add(pos["asset_id"])
            mid = self.midpoint(pos["asset_id"])
            if mid is not None:
                self.store.update_price(pos["asset_id"], mid)

    def redeem(self) -> None:
        """Claim winnings for resolved markets (needs a Relayer API key for gasless calls)."""
        if self.secure is None:
            return
        if not (self.cfg.relayer_key and self.cfg.relayer_address):
            if self._has_redeemable():
                self.warn("redeem", "there are winnings to claim but no Relayer API key is set; claim them on polymarket.com",
                          every=timedelta(days=1))
            return
        try:
            page = self.call(lambda: self.secure.list_positions(user=self.cfg.wallet, status="REDEEMABLE", page_size=50).first_page())
        except Exception as e:
            self.warn("redeem-list", f"could not list redeemable positions: {e}")
            return
        done = set()
        for p in page.items:
            cid = str(p.condition_id)
            if cid in done:
                continue
            done.add(cid)
            try:
                tx = self.call(self.secure.redeem_positions, condition_id=cid)
                tx.wait()
                log.info("redeemed %s · %s", p.title, p.outcome)
                self.notifier.send("Polybot: winnings claimed", f"{p.title} · {p.outcome}\n${float(p.current_value or 0):.2f} back in the wallet", tags="moneybag")
            except Exception as e:
                self.warn(f"redeem:{cid}", f"could not redeem {p.title}: {e}")

    def _has_redeemable(self) -> bool:
        try:
            page = self.call(lambda: self.secure.list_positions(user=self.cfg.wallet, status="REDEEMABLE", page_size=1).first_page())
            return bool(page.items)
        except Exception:
            return False
