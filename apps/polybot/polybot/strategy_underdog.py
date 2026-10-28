"""Strategy 2, "underdog": buy the cheaper side of a game at kickoff and sell into the early bounce.

Filip noticed underdogs tend to get a boost once a match starts. The bot buys
the underdog in the minutes around the start, rests a sell order at the
take-profit price (live) or watches for that price (paper), and otherwise
sells at the stop or when the holding time runs out.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from .core import CENT, Core, aware
from .store import iso, parse_iso
from .underdog import Candidate, exit_reason, in_buy_window, one_per_match, pick_underdog, stop_price, target_price

log = logging.getLogger("polybot.underdog")


class UnderdogStrategy:
    name = "underdog"
    SCHEDULE_TTL = timedelta(minutes=5)
    PAGE_SIZE = 100
    MAX_PAGES = 12

    def __init__(self, core: Core):
        self.core = core
        self.cfg = core.cfg
        self.store = core.store
        self.schedule: dict[str, Candidate] = {}   # condition id → candidate
        self.decisions: dict[str, str] = {}
        self._refreshed_at: datetime | None = None
        self._checked_at: dict[str, datetime] = {}  # asset id → last paper price check

    @property
    def interval(self) -> int:
        return self.cfg.ud_tick_seconds

    # ── the schedule ─────────────────────────────────────────
    def refresh_schedule(self, now: datetime) -> None:
        cfg = self.cfg
        core = self.core
        horizon = now + timedelta(hours=cfg.ud_lookahead_hours)
        pages = core.public.list_markets(sports_market_types=["moneyline"], closed=False,
                                         end_date_min=now - timedelta(hours=6), end_date_max=now + timedelta(hours=cfg.ud_lookahead_hours + 36),
                                         page_size=self.PAGE_SIZE)
        page = core.call(pages.first_page)
        cands: list[Candidate] = []
        n = 0
        while True:
            for m in page.items:
                gs = m.sports.game_start_time if m.sports else None
                if gs is None:
                    continue
                gs = aware(gs)
                if gs < now - timedelta(minutes=cfg.ud_buy_until_min + 5) or gs > horizon:
                    continue
                # The band is applied again later; here every side of a match must be seen
                # so the cheaper team can be picked, hence the wide limits.
                cand = pick_underdog(m, min_price=Decimal("0.001"), max_price=Decimal("0.999"), trading_id=core.trading_id)
                if cand is None:
                    continue
                cand.game_start = gs
                cands.append(cand)
                core.remember_market(m)
            n += 1
            if not page.next_cursor or n >= self.MAX_PAGES:
                break
            page = core.call(pages.from_cursor(page.next_cursor).first_page)
        found: dict[str, Candidate] = {}
        for cand in one_per_match(cands):
            if cfg.ud_min_price <= cand.price <= cfg.ud_max_price:
                found[cand.condition_id] = cand
        self.schedule = found
        self._refreshed_at = now
        log.info("%d games with an underdog in band in the next %gh", len(found), cfg.ud_lookahead_hours)

    def wanted_sport(self, sport: str) -> bool:
        wanted = [x for x in self.cfg.ud_sports.split(",") if x]
        return not wanted or sport in wanted

    # ── one tick ─────────────────────────────────────────────
    def run(self, now: datetime) -> None:
        if not self.cfg.ud_enabled:
            return
        if self._refreshed_at is None or now - self._refreshed_at >= self.SCHEDULE_TTL:
            try:
                self.refresh_schedule(now)
            except Exception as e:
                self.core.warn("ud-schedule", f"could not load upcoming games: {e}")
        for pos in self.store.open_positions(self.name):
            try:
                self.manage(pos, now)
            except Exception as e:
                log.exception("could not manage %s", pos.get("title"))
                self.core.record_error(f"underdog {pos.get('title')}: {e}")
        for cand in sorted(self.schedule.values(), key=lambda c: c.game_start):
            if not in_buy_window(cand.game_start, now, self.cfg.ud_buy_before_min, self.cfg.ud_buy_until_min):
                continue
            try:
                self.consider_entry(cand, now)
            except Exception as e:
                log.exception("could not act on %s", cand.title)
                self.decisions[cand.condition_id] = f"error: {e}"
                self.core.record_error(f"underdog {cand.title}: {e}")

    def consider_entry(self, cand: Candidate, now: datetime) -> dict | None:
        cfg = self.cfg
        core = self.core
        what = f"{cand.title} · {cand.outcome}"

        def skip(reason: str, final: bool = False) -> None:
            self.decisions[cand.condition_id] = reason
            if final:
                self.store.state["acted"][cand.asset_id] = iso(now)
            log.info("skip %s: %s", what, reason)

        if self.store.position(cand.asset_id) or self.store.acted_at(cand.asset_id):
            return None  # bought already, or decided for good
        if not self.wanted_sport(cand.sport):
            return skip(f"{cand.sport}: not in UD_SPORTS", final=True)
        if cand.liquidity < cfg.ud_min_liquidity:
            return skip(f"too thin (${cand.liquidity:,.0f} liquidity)", final=True)
        if len(self.store.open_positions(self.name)) >= cfg.ud_max_open:
            return skip("max open positions reached")
        if self.store.spent_since(now - timedelta(hours=24), self.name) + cfg.ud_stake_usd > cfg.ud_daily_budget_usd:
            return skip("daily budget used up")
        m = core.market(cand.condition_id, cand.slug)
        if m is None:
            return skip("market not found", final=True)
        why = core.not_tradeable(m, now)
        if why:
            return skip(why, final=True)
        book = core.book(cand.asset_id)
        if book is None or book["ask"] is None or book["bid"] is None:
            return skip("no order book")
        spread = book["ask"] - book["bid"]
        if spread > cfg.ud_max_spread:
            return skip(f"spread {spread:.2f} too wide")
        est = core.estimate(cand.asset_id, "BUY", amount=cfg.ud_stake_usd)
        if est is None:
            return skip("not enough liquidity")
        if est > cfg.ud_max_price:
            return skip(f"price {est:.2f} is above the max {cfg.ud_max_price}", final=True)
        if est < cfg.ud_min_price:
            return skip(f"price {est:.2f} is below the min {cfg.ud_min_price}", final=True)
        amount = cfg.ud_stake_usd
        shares = (amount / est).quantize(CENT, rounding=ROUND_DOWN)
        if shares < book["min_size"]:
            amount = (book["min_size"] * est * Decimal("1.02")).quantize(CENT, rounding=ROUND_UP)
        if core.mode == "live" and core.balance is not None and core.balance < amount:
            core.warn("funds", f"not enough pUSD for a ${amount} order (balance ${core.balance:.2f})")
            return skip("not enough pUSD in the wallet")

        fee_rate = core.fee_rate(m)
        fill = core.buy(cand.asset_id, amount, est, what, fee_rate=fee_rate)
        if fill is None:
            return skip("order rejected", final=True)
        entry = Decimal(str(fill["price"]))
        tick = book["tick"]
        target = target_price(entry, cfg.ud_take_profit_pct, tick)
        stop = stop_price(entry, cfg.ud_stop_loss_pct, tick)
        deadline = cand.game_start + timedelta(minutes=cfg.ud_max_hold_min)
        rec = {
            "strategy": self.name, "side": "BUY", "asset_id": cand.asset_id, "condition_id": cand.condition_id,
            "title": cand.title, "outcome": cand.outcome, "slug": cand.slug, "event_slug": cand.event_slug,
            "sport": cand.sport, "game_start": iso(cand.game_start), "target": float(target), "stop": float(stop),
            "deadline": iso(deadline), "fee_rate": float(fee_rate), "mode": core.mode, **fill,
        }
        got = Decimal(str(fill["shares"]))
        if core.mode == "live" and got > 0:
            resting = core.limit_sell(cand.asset_id, target, got, what)
            if resting:
                rec["order_id"] = resting["order_id"]
                rec["order_status"] = resting["status"]
        self.store.open_position(rec, now)
        self._checked_at[cand.asset_id] = now
        verb = "Bought" if core.mode == "live" else "Paper buy"
        self.decisions[cand.condition_id] = "bought" if core.mode == "live" else "paper bought"
        log.info("%s %s: %s shares at %.2f ($%s), target %.2f stop %.2f until %s",
                 verb, what, rec["shares"], rec["price"], rec["cost"], target, stop, deadline.strftime("%H:%M"))
        core.notifier.send(f"Polybot: {verb.lower()}",
                           f"[underdog] {what}\n{rec['shares']:.2f} shares at {rec['price']:.2f} (${rec['cost']:.2f})\n"
                           f"sell at {target:.2f}, stop {stop:.2f}, out by {deadline.astimezone().strftime('%H:%M')}",
                           tags="chart_with_upwards_trend")
        return rec

    # ── holding ──────────────────────────────────────────────
    def manage(self, pos: dict, now: datetime) -> None:
        core = self.core
        asset_id = pos["asset_id"]
        what = f"{pos.get('title')} · {pos.get('outcome')}"
        shares = Decimal(str(pos.get("shares", 0)))
        target = Decimal(str(pos["target"]))
        deadline = parse_iso(pos["deadline"]) if pos.get("deadline") else None
        pos_view = {"target": pos["target"], "stop": pos["stop"], "deadline": deadline}

        # Live: a resting sell may have filled since the last tick.
        if core.mode == "live" and pos.get("order_id"):
            state = core.order_state(pos["order_id"])
            if state is None:
                held = core.held_shares(pos["condition_id"], asset_id)
                if held is not None and held <= Decimal("0.01"):
                    return self.close(pos, target, shares * target, "target", now)
            else:
                matched = state["matched"]
                if matched >= state["original"] - Decimal("0.01"):
                    return self.close(pos, state["price"], matched * state["price"], "target", now)
                if matched > 0 and float(matched) != float(pos.get("matched", 0)):
                    self.store.update_position(asset_id, matched=float(matched))

        bid = core.estimate(asset_id, "SELL", shares=shares)
        if bid is not None:
            self.store.update_price(asset_id, bid)
        reason = exit_reason(pos_view, bid, now)

        # Paper: a resting sell at the target would have been hit by any trade at or above it.
        if core.mode != "live" and reason != "target":
            since = self._checked_at.get(asset_id) or parse_iso(pos["opened_at"])
            high = core.price_high_since(asset_id, since, now)
            self._checked_at[asset_id] = now
            if high is not None and high >= target:
                reason = "target"
        if reason is None:
            return
        if reason == "target" and core.mode != "live":
            return self.close(pos, target, shares * target, "target", now)  # maker fill: no fee

        # Stop or time (or a live target seen on the bid): sell what is left with a market order.
        if core.mode == "live" and pos.get("order_id"):
            core.cancel(pos["order_id"])
            remaining = shares - Decimal(str(pos.get("matched", 0)))
        else:
            remaining = shares
        if remaining <= 0:
            return self.close(pos, target, shares * target, "target", now)
        est = core.estimate(asset_id, "SELL", shares=remaining)
        if est is None:
            self.store.note_position(asset_id, f"wanted to sell ({reason}) but found no bid")
            return
        m = core.market(pos["condition_id"], pos.get("slug", ""))
        fill = core.sell(asset_id, remaining, est, what, fee_rate=core.fee_rate(m) if m else Decimal(str(pos.get("fee_rate", 0))))
        if fill is None:
            return
        matched = Decimal(str(pos.get("matched", 0)))
        proceeds = Decimal(str(fill["proceeds"])) + matched * target
        price = (proceeds / shares) if shares else Decimal(str(fill["price"]))
        self.close(pos, price, proceeds, reason, now)

    def close(self, pos: dict, price: Decimal, proceeds: Decimal, reason: str, now: datetime) -> None:
        core = self.core
        label = {"target": "target hit", "stop": "stop loss", "time": "time's up"}.get(reason, reason)
        rec = self.store.close_position(pos["asset_id"], price, proceeds, label, now, mode=core.mode)
        if rec is None:
            return
        self._checked_at.pop(pos["asset_id"], None)
        verb = "Sold" if core.mode == "live" else "Paper sell"
        log.info("%s %s: %s shares at %.2f (%s), P&L $%.2f", verb, pos.get("title"), pos.get("shares"), price, label, rec["pnl"])
        core.notifier.send(f"Polybot: {verb.lower()}",
                           f"[underdog] {pos.get('title')} · {pos.get('outcome')}\n"
                           f"{float(pos.get('shares', 0)):.2f} shares at {price:.2f} ({label}), P&L ${rec['pnl']:+.2f}",
                           tags="chart_with_upwards_trend" if rec["pnl"] >= 0 else "chart_with_downwards_trend")

    # ── the page ─────────────────────────────────────────────
    def status(self, now: datetime) -> dict:
        rows = []
        for c in sorted(self.schedule.values(), key=lambda c: c.game_start):
            pos = self.store.position(c.asset_id)
            rows.append({
                "condition_id": c.condition_id, "title": c.title, "outcome": c.outcome, "sport": c.sport,
                "slug": c.slug, "event_slug": c.event_slug, "game_start": iso(c.game_start),
                "minutes": round((c.game_start - now).total_seconds() / 60),
                "price": float(c.price), "liquidity": float(c.liquidity),
                "decision": self.decisions.get(c.condition_id, "") or ("holding" if pos else ""),
                "wanted": self.wanted_sport(c.sport),
            })
        return {"enabled": self.cfg.ud_enabled, "games": rows[:40], "games_total": len(rows),
                "refreshed_at": iso(self._refreshed_at) if self._refreshed_at else None}
