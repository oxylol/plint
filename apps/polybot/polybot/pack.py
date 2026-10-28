"""The credible pack: measure the traders on a leaderboard and keep the ones worth copying.

Measuring a trader takes a handful of API calls (closed positions, a week of
fills, account stats), so a pool of 80 takes a minute or two. It runs in a
background thread every few hours; strategies use the last finished snapshot.
"""
from __future__ import annotations

import logging
import statistics
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

from .categories import category
from .core import Core, aware, pretty_name
from .credibility import Judgement, Record, Rules, judge
from .store import iso, utcnow

log = logging.getLogger("polybot.pack")


@dataclass
class Member:
    wallet: str
    name: str
    rank: int
    record: Record
    judgement: Judgement


class CrediblePack:
    def __init__(self, core: Core, rules: Rules, *, window: str, category: str, pool: int, size: int, refresh_hours: float):
        self.core = core
        self.rules = rules
        self.window = window
        self.category = category
        self.pool = pool
        self.size = size
        self.refresh = timedelta(hours=refresh_hours)
        self.members: list[Member] = []       # the pack, best first
        self.judged: list[Member] = []        # everyone measured
        self.refreshed_at: datetime | None = None
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def key(self) -> tuple:
        return (self.window, self.category, self.pool)

    @property
    def building(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ensure_fresh(self, now: datetime) -> None:
        if self.building:
            return
        if self.refreshed_at is None or now - self.refreshed_at >= self.refresh:
            self._thread = threading.Thread(target=self._build, name=f"pack-{self.window}", daemon=True)
            self._thread.start()

    # ── measuring ────────────────────────────────────────────
    def _build(self) -> None:
        started = utcnow()
        try:
            kwargs: dict = {"window": self.window, "sort_by": "PNL", "page_size": min(self.pool, 1000)}
            if self.category and self.category != "overall":
                kwargs["category"] = self.category
            board = self.core.call(lambda: self.core.public.list_trader_leaderboard(**kwargs).first_page()).items[: self.pool]
            judged: list[Member] = []
            for e in board:
                w = str(e.wallet).lower()
                name = pretty_name(e.user_name, w)
                try:
                    rec = self.measure(w, name, int(e.rank), float(e.pnl), float(e.volume))
                except Exception as ex:
                    log.warning("could not measure %s: %s", name, ex)
                    continue
                judged.append(Member(w, name, int(e.rank), rec, judge(rec, self.rules)))
            ok = sorted((m for m in judged if m.judgement.ok), key=lambda m: -m.judgement.score)
            with self._lock:
                self.judged = judged
                self.members = ok[: self.size]
                self.refreshed_at = utcnow()
                self.error = None
            log.info("[pack %s] measured %d traders in %.0fs: %d credible, keeping %d: %s", self.window, len(judged),
                     (utcnow() - started).total_seconds(), len(ok), len(self.members),
                     ", ".join(f"{m.name} {m.judgement.score:.2f}" for m in self.members))
        except Exception as ex:
            log.exception("[pack %s] build failed", self.window)
            with self._lock:
                self.error = str(ex)
                self.refreshed_at = utcnow() - self.refresh + timedelta(minutes=15)  # try again in a quarter hour

    def measure(self, wallet: str, name: str, rank: int, pnl: float, volume: float) -> Record:
        core = self.core
        now = utcnow()
        rec = Record(wallet=wallet, name=name, rank=rank, pnl=pnl, volume=volume)
        # Closed positions: the track record (up to 200, newest first).
        closed = []
        pages = core.public.list_positions(user=wallet, status="CLOSED", sort_by="TIMESTAMP", sort_direction="DESC", page_size=100)
        page = core.call(pages.first_page)
        closed.extend(page.items)
        if page.next_cursor:
            page = core.call(pages.from_cursor(page.next_cursor).first_page)
            closed.extend(page.items)
        pnls, costs, holds, cats = [], [], [], {}
        sides: dict[str, set] = {}
        for p in closed:
            # Closed rows carry a ~0 residual cost; what was really put in is total size × average price.
            cost = float(p.total_size or 0) * float(p.avg_price or 0) or float(p.total_cost_usdc or p.entry_cost_usdc or 0)
            if cost < 1:
                continue
            x = float(p.realized_pnl)
            pnls.append(x)
            costs.append(cost)
            if p.last_event_at and p.first_entry_at:
                holds.append((aware(p.last_event_at) - aware(p.first_entry_at)).total_seconds() / 3600)
            k = (p.event_slug or p.slug or "").split("-")[0]
            cats[k] = cats.get(k, 0) + 1
            c = category(p.title, p.event_slug or p.slug)
            cn, cw = rec.cat_stats.get(c, (0, 0))
            rec.cat_stats[c] = (cn + 1, cw + (1 if x > 0 else 0))
            sides.setdefault(str(p.condition_id), set()).add(p.outcome_index if p.outcome_index is not None else p.outcome)
        both = sum(1 for v in sides.values() if len(v) > 1)
        rec.two_sided_share = (both / len(sides)) if sides else 0.0
        rec.closed_n = len(pnls)
        rec.wins = sum(1 for x in pnls if x > 0)
        rec.losses = sum(1 for x in pnls if x < 0)
        rec.gross_win = sum(x for x in pnls if x > 0)
        rec.gross_loss = -sum(x for x in pnls if x < 0)
        rec.sum_pnl = sum(pnls)
        rec.sum_cost = sum(costs)
        rec.top_pnl = max(pnls) if pnls else 0.0
        rec.median_hold_hours = statistics.median(holds) if holds else None
        rec.median_cost = statistics.median(costs) if costs else None
        if cats:
            k, n = max(cats.items(), key=lambda kv: kv[1])
            rec.top_category, rec.top_category_share = k, n / max(1, len(closed))
        # A week of fills: are they active, and do they take or make?
        since = now - timedelta(days=7)
        fills = core.call(lambda: core.public.list_trades(user=wallet, start=since, taker_only=False, page_size=500).first_page()).items
        taker = core.call(lambda: core.public.list_trades(user=wallet, start=since, taker_only=True, page_size=500).first_page()).items
        rec.fills_7d = len(fills)
        rec.taker_share = (len(taker) / len(fills)) if fills else None
        last = max((aware(t.timestamp) for t in fills), default=None)
        rec.hours_since_fill = (now - last).total_seconds() / 3600 if last else None
        # Account age.
        try:
            stats = core.call(core.public.get_user_stats, user=wallet)
            if stats is not None and stats.join_date:
                rec.age_days = (now - aware(stats.join_date)).total_seconds() / 86400
        except Exception as ex:
            log.debug("stats for %s: %s", name, ex)
        return rec

    # ── what strategies use ──────────────────────────────────
    def records(self) -> dict[str, Record]:
        """Everyone measured on the board (credible or not), by wallet."""
        with self._lock:
            return {m.wallet: m.record for m in self.judged}

    def everyone(self) -> list[Member]:
        """Everyone measured, board order (for setups that judge with the data model instead)."""
        with self._lock:
            return sorted(self.judged, key=lambda m: m.rank)

    def snapshot(self) -> tuple[list[Member], dict[str, float], dict[str, float]]:
        """(members, vote weights, typical position sizes) of the last finished build."""
        with self._lock:
            members = list(self.members)
        weights = {m.wallet: m.judgement.weight for m in members}
        sizes = {m.wallet: (m.record.median_cost or 0.0) for m in members}
        return members, weights, sizes

    def status(self) -> dict:
        with self._lock:
            members, judged, at, err = list(self.members), list(self.judged), self.refreshed_at, self.error
        def row(m: Member) -> dict:
            r = m.record
            return {"wallet": m.wallet, "name": m.name, "rank": m.rank, "ok": m.judgement.ok, "score": m.judgement.score,
                    "weight": round(m.judgement.weight, 2), "reasons": m.judgement.reasons,
                    "closed": r.closed_n, "win_rate": round(r.win_rate, 3), "profit_factor": round(min(r.profit_factor, 99), 2),
                    "return_on_cost": round(r.return_on_cost, 3), "top_share": round(min(r.top_share, 9.99), 2),
                    "two_sided": round(r.two_sided_share, 2),
                    "hold_hours": round(r.median_hold_hours, 1) if r.median_hold_hours is not None else None,
                    "median_cost": round(r.median_cost, 0) if r.median_cost is not None else None,
                    "fills_7d": r.fills_7d, "taker_share": round(r.taker_share, 2) if r.taker_share is not None else None,
                    "hours_since_fill": round(r.hours_since_fill, 1) if r.hours_since_fill is not None else None,
                    "age_days": round(r.age_days) if r.age_days is not None else None,
                    "edge": round(r.edge, 4), "pnl": r.pnl, "volume": r.volume,
                    "category": r.top_category, "category_share": round(r.top_category_share, 2)}
        return {"window": self.window, "category": self.category, "pool": self.pool, "size": self.size,
                "refreshed_at": iso(at) if at else None, "building": self.building, "error": err,
                "members": [row(m) for m in members], "judged": [row(m) for m in sorted(judged, key=lambda m: m.rank)]}
