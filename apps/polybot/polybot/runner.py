"""Runs both strategies on their own clocks and writes the page's JSON."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal

from .config import Config
from .core import Core
from .notify import Notifier
from .store import Store, iso
from .strategy_consensus import ConsensusStrategy
from .strategy_underdog import UnderdogStrategy

log = logging.getLogger("polybot")
LABELS = {"consensus": "Consensus", "underdog": "Underdog"}


class Runner:
    def __init__(self, cfg: Config, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.core = Core(cfg, store, notifier)
        self.consensus = ConsensusStrategy(self.core)
        self.underdog = UnderdogStrategy(self.core)
        self.strategies = [self.consensus, self.underdog]
        self.last_run: dict[str, datetime] = {}
        self._consensus_status: dict = {"leaders": [], "fills": 0, "signals": []}

    def connect(self) -> None:
        self.core.connect()

    def due(self, strategy, now: datetime) -> bool:
        last = self.last_run.get(strategy.name)
        return last is None or (now - last).total_seconds() >= strategy.interval

    def next_due(self, now: datetime) -> datetime:
        times = []
        for s in self.strategies:
            if s.name == "underdog" and not self.cfg.ud_enabled:
                continue
            last = self.last_run.get(s.name)
            times.append(now if last is None else last + timedelta(seconds=s.interval))
        return min(times) if times else now + timedelta(seconds=self.cfg.poll_seconds)

    def run_due(self, now: datetime, force: bool = False) -> list[str]:
        ran: list[str] = []
        for s in self.strategies:
            if not force and not self.due(s, now):
                continue
            self.last_run[s.name] = now
            if s.name == "underdog" and not self.cfg.ud_enabled:
                continue
            try:
                s.run(now)
            except Exception as e:
                log.exception("%s cycle failed", s.name)
                self.core.record_error(f"{s.name} cycle failed: {e}")
            ran.append(s.name)
        if not ran:
            return ran
        try:
            self.core.settle(now, self.store.open_positions())
            if "consensus" in ran:
                self.core.mark(self.store.open_positions("consensus"))
                if self.core.mode == "live":
                    self.core.refresh_balance()
                    if self.cfg.auto_redeem:
                        self.core.redeem()
                self._consensus_status = self.consensus.status()
        except Exception as e:
            log.exception("bookkeeping failed")
            self.core.record_error(f"bookkeeping failed: {e}")
        self.store.save()
        self.store.write_status(self.status(now))
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
        strategies = {}
        for name in ("consensus", "underdog"):
            strategies[name] = {**self.store.totals(now, name), "label": LABELS[name],
                                "enabled": True if name == "consensus" else cfg.ud_enabled,
                                "last_run": iso(self.last_run[name]) if name in self.last_run else None}
        return {
            "updated_at": iso(now),
            "next_run_at": iso(self.next_due(now)),
            "mode": core.mode,
            "config": cfg.public(),
            "account": {**core.account, "balance": float(core.balance) if core.balance is not None else None},
            **self._consensus_status,
            "underdog": self.underdog.status(now),
            "positions": positions,
            "trades": self.store.recent_trades(40),
            "totals": self.store.totals(now),
            "strategies": strategies,
            "errors": core.errors[-5:],
        }
