"""Entry point: `python -m polybot [run|once|check]`."""
from __future__ import annotations

import json
import logging
import signal
import sys
import threading

from .config import Config
from .notify import Notifier
from .runner import Runner
from .store import Store, utcnow
from . import web

TICK = 5  # seconds between looks at the strategies' clocks


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = Config.from_env()
    store = Store(cfg.data_dir)
    notifier = Notifier(cfg.ntfy_url, cfg.ntfy_token)
    runner = Runner(cfg, store, notifier)
    cmd = argv[1] if len(argv) > 1 else "run"

    if cmd == "check":
        runner.connect()
        print(json.dumps(runner.core.approvals(), indent=2, default=str))
        return 0
    if cmd == "once":
        runner.connect()
        runner.run_due(utcnow(), force=True)
        status = store.read_status() or {}
        print(json.dumps({k: status.get(k) for k in ("mode", "fills", "strategies")}, indent=2))
        print("\nconsensus:")
        for s in status.get("signals", [])[:12]:
            print(f"  {s['score']:+3d}  {s['side']:4s} {s['title'][:56]} · {s['outcome']}  agree {len(s['agree'])} / against {len(s['disagree'])}"
                  f"  leaders {s['avg_price'] if s['avg_price'] is not None else '?'}  now {s['price'] if s['price'] is not None else '?'}  → {s['decision']}")
        ud = status.get("underdog", {})
        print(f"\nunderdog: {ud.get('games_total', 0)} games with an underdog in band in the window")
        for g in ud.get("games", [])[:12]:
            print(f"  {g['minutes']:+5d} min  {g['sport']:6s} {g['title'][:50]} · {g['outcome']} at {g['price']:.2f}  → {g['decision']}")
        return 0
    if cmd != "run":
        print("usage: python -m polybot [run|once|check]", file=sys.stderr)
        return 2

    web.serve(cfg.port, store, cfg.mode)
    runner.connect()
    log = logging.getLogger("polybot")
    log.info("Polybot in %s mode: consensus (top %d, %s, %d must agree, $%s) · underdog (%s, $%s, +%s%% target)",
             runner.core.mode, cfg.top_n, cfg.window, cfg.min_agree, cfg.stake_usd,
             "on" if cfg.ud_enabled else "off", cfg.ud_stake_usd, cfg.ud_take_profit_pct)
    notifier.send("Polybot started",
                  f"{runner.core.mode} mode · consensus: top {cfg.top_n} ({cfg.window}), {cfg.min_agree} must agree, ${cfg.stake_usd} a trade"
                  f" · underdog: {'on' if cfg.ud_enabled else 'off'}, ${cfg.ud_stake_usd} a game, +{cfg.ud_take_profit_pct}% target",
                  priority="low", tags="robot")

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        try:
            runner.run_due(utcnow())
        except Exception as e:  # keep going; the page shows the error
            log.exception("tick failed")
            runner.core.record_error(f"tick failed: {e}")
        stop.wait(TICK)
    log.info("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
