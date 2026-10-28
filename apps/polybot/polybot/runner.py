"""Runs every strategy on its own clock and writes the page's JSON."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal

from .config import Config, parse_variants
from .core import Core
from .notify import Notifier
from .pack import CrediblePack
from .store import Store, iso
from .strategy_consensus import ConsensusParams, ConsensusStrategy, LeaderFeed
from .strategy_underdog import UnderdogStrategy

log = logging.getLogger("polybot")


class Runner:
    def __init__(self, cfg: Config, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.core = Core(cfg, store, notifier)
        base = ConsensusParams.base(cfg)
        params = [base] + [base.variant(name, overrides) for name, overrides in parse_variants(cfg.variants)]
        self.feed = LeaderFeed(self.core, keep_hours=max(p.lookback_hours for p in params),
                               board_ttl=min(p.poll_seconds for p in params))
        self.packs: dict[tuple, CrediblePack] = {}
        self.consensus = []
        for p in params:
            pack = None
            if p.pack:
                key = (p.window, p.category, p.pool)
                pack = self.packs.get(key)
                if pack is None:
                    pack = self.packs[key] = CrediblePack(self.core, cfg.rules(), window=p.window, category=p.category,
                                                          pool=p.pool, size=cfg.cr_pack_size, refresh_hours=cfg.cr_refresh_hours)
            self.consensus.append(ConsensusStrategy(self.core, p, self.feed, pack))
        self.underdog = UnderdogStrategy(self.core)
        self.strategies = [*self.consensus, self.underdog]
        self.last_run: dict[str, datetime] = {}
        self._consensus_status: dict[str, dict] = {}

    def connect(self) -> None:
        self.core.connect()

    def enabled(self, strategy) -> bool:
        return strategy.name != "underdog" or self.cfg.ud_enabled

    def due(self, strategy, now: datetime) -> bool:
        last = self.last_run.get(strategy.name)
        return last is None or (now - last).total_seconds() >= strategy.interval

    def next_due(self, now: datetime) -> datetime:
        times = []
        for s in self.strategies:
            if not self.enabled(s):
                continue
            last = self.last_run.get(s.name)
            times.append(now if last is None else last + timedelta(seconds=s.interval))
        return min(times) if times else now + timedelta(seconds=self.cfg.poll_seconds)

    def run_due(self, now: datetime, force: bool = False) -> list[str]:
        ran: list[str] = []
        timings: list[str] = []
        for s in self.strategies:
            if not force and not self.due(s, now):
                continue
            self.last_run[s.name] = now
            if not self.enabled(s):
                continue
            t0 = time.monotonic()
            try:
                s.run(now)
            except Exception as e:
                log.exception("%s cycle failed", s.name)
                self.core.record_error(f"{s.name} cycle failed: {e}")
            timings.append(f"{s.name} {time.monotonic() - t0:.0f}s")
            ran.append(s.name)
        if not ran:
            return ran
        t0 = time.monotonic()
        consensus_ran = [s for s in self.consensus if s.name in ran]
        try:
            self.core.settle(now, self.store.open_positions())
            if consensus_ran:
                self.core.mark([p for s in consensus_ran for p in self.store.open_positions(s.name)])
                if self.core.mode == "live":
                    self.core.refresh_balance()
                    if self.cfg.auto_redeem:
                        self.core.redeem()
                for s in consensus_ran:
                    self._consensus_status[s.name] = s.status()
        except Exception as e:
            log.exception("bookkeeping failed")
            self.core.record_error(f"bookkeeping failed: {e}")
        self.store.save()
        self.store.write_status(self.status(now))
        if consensus_ran:
            log.info("cycle: %s · bookkeeping and page %.0fs", ", ".join(timings), time.monotonic() - t0)
        return ran

    # ── the page ─────────────────────────────────────────────
    def status(self, now: datetime) -> dict:
        cfg = self.cfg
        core = self.core
        positions = []
        for p in self.store.open_positions():
            shares = Decimal(str(p.get("shares", 0)))
            cost = Decimal(str(p.get("cost", 0)))
            cur = Decimal(str(p.get("current_price") or p.get("price") or 0))
            value = shares * cur
            positions.append({**p, "strategy": p.get("strategy", "consensus"), "value": float(value), "pnl": float(value - cost)})
        positions.sort(key=lambda p: p.get("opened_at", ""), reverse=True)
        labels = {s.name: s.label for s in self.consensus}
        labels["underdog"] = "Underdog"
        strategies = []
        for s in self.strategies:
            strategies.append({**self.store.totals(now, s.name), "name": s.name, "label": s.label if hasattr(s, "label") else "Underdog",
                               "kind": getattr(s, "kind", "underdog"), "enabled": self.enabled(s), "since": self.store.since(s.name),
                               "last_run": iso(self.last_run[s.name]) if s.name in self.last_run else None})
        return {
            "updated_at": iso(now),
            "next_run_at": iso(self.next_due(now)),
            "mode": core.mode,
            "config": cfg.public(),
            "account": {**core.account, "balance": float(core.balance) if core.balance is not None else None},
            "variants": [self._consensus_status.get(s.name, {"name": s.name, "label": s.label, "params": s.p.public(), "note": s.note,
                                                              "leaders": [], "fills": 0, "signals": [], "followed": [],
                                                              **({"pack": s.pack.key} if s.pack else {})}) for s in self.consensus],
            "underdog": self.underdog.status(now),
            "positions": positions,
            "trades": self.store.recent_trades(40),
            "totals": self.store.totals(now),
            "strategies": strategies,
            "labels": labels,
            "packs": {"|".join(map(str, k)): v.status() for k, v in self.packs.items()},
            "errors": core.errors[-5:],
        }
