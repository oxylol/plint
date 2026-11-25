"""The clock: one evening run per trading day, the heavy jobs between them.

Every job is a child process (`python -m stocks run|research|fit|review`), one at a time. The parent only
serves the page and keeps time, so it stays small, a runaway fit is the child's problem, and whatever a
job held in memory is given back when it ends."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone

from . import cal
from .config import Config
from .store import Store

log = logging.getLogger("stocks.runner")


class Runner:
    def __init__(self, cfg: Config, store: Store, stop: threading.Event):
        self.cfg, self.store, self.stop = cfg, store, stop

    def child(self, *args: str, nice: int = 0) -> bool:
        """Run `python -m stocks <args>` and wait for it. Ends it if the container is stopping."""
        log.info("starting: stocks %s", " ".join(args))
        try:
            p = subprocess.Popen([sys.executable, "-m", "stocks", *args],
                                 preexec_fn=(lambda: os.nice(nice)) if nice else None)
        except OSError as e:
            self.store.log(f"could not start `stocks {' '.join(args)}`: {e}", "warn")
            return False
        while p.poll() is None:
            if self.stop.wait(2.0):
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()
                return False
        if p.returncode != 0:
            self.store.log(f"`stocks {' '.join(args)}` ended with an error (code {p.returncode}); see the container's log", "warn")
        return p.returncode == 0

    def heavy(self, now: datetime) -> None:
        """What is due besides the evening run: the first research, then a refit every week."""
        store = self.store
        if not store.get("history_done"):
            return
        today = now.strftime("%Y-%m-%d")
        if store.read_json("reports/index.json") is None:
            if store.get("research_tried") != today:           # once a day at most: a broken run must not spin
                store.put("research_tried", today)
                store.log("first backtests starting: several hours; nothing but the index trades until they are written")
                if self.child("research", nice=15) and self.child("fit", nice=15):
                    store.put("fitted_at", today)
                    store.log("backtests written and models fitted: they start at the next evening run")
                    self.child("run")                           # so the page shows the reports now
            return
        if store.get("filings_backfilled") and store.read_json("reports/drift.json") is None \
                and store.get("events_research_tried") != today:      # filings arrived after the first backtests
            store.put("events_research_tried", today)
            if self.child("research", "--only", "drift,insiders", nice=15):
                self.child("run")
        if store.read_json("reports/index.json") is not None and store.get("copy_tried") != today \
                and (not self.cfg.models or "copy" in self.cfg.models):
            from .copytrade.live import due
            from .bars import BarStore
            if due(store, BarStore(self.cfg.cache_dir).last_day(self.cfg.index_symbol)):
                store.put("copy_tried", today)          # once a day at most: a broken run must not spin
                store.log("copy study starting: House reports, every person's copy book, the picks (the first time, hours)")
                if self.child("copy", nice=15):
                    self.child("run")
        last = store.get("fitted_at")
        week_old = not last or (now.date() - datetime.fromisoformat(last).date()).days >= 7
        if week_old and now.astimezone(cal.NY).weekday() >= 5:   # at the weekend, when no run is waiting
            if self.child("fit", nice=15):
                store.put("fitted_at", today)
            if self.cfg.has_agent:
                self.child("review")

    def loop(self) -> None:
        while not self.stop.is_set():
            self.child("run")
            if self.stop.is_set():
                return
            self.heavy(datetime.now(timezone.utc))
            nxt = cal.next_run(datetime.now(timezone.utc), self.cfg.run_at)
            log.info("next run %s New York", nxt.strftime("%Y-%m-%d %H:%M"))
            checked = None
            while not self.stop.is_set():
                now = datetime.now(timezone.utc)
                if now >= nxt:
                    break
                hour = now.strftime("%Y-%m-%d %H")
                if hour != checked and now.astimezone(cal.NY).weekday() >= 5:   # the weekly refit comes due between runs
                    checked = hour
                    self.heavy(now)
                self.stop.wait(min(60.0, max(1.0, (nxt - now).total_seconds())))
