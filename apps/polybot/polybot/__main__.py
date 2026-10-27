"""Entry point: `python -m polybot [run|once|check]`."""
from __future__ import annotations

import json
import logging
import signal
import sys
import threading

from .config import Config
from .engine import Engine
from .notify import Notifier
from .store import Store
from . import web


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = Config.from_env()
    store = Store(cfg.data_dir)
    notifier = Notifier(cfg.ntfy_url, cfg.ntfy_token)
    engine = Engine(cfg, store, notifier)
    cmd = argv[1] if len(argv) > 1 else "run"

    if cmd == "check":
        engine.connect()
        print(json.dumps(engine.approvals(), indent=2, default=str))
        return 0
    if cmd == "once":
        engine.connect()
        status = engine.run_once()
        print(json.dumps({k: status[k] for k in ("mode", "fills", "totals")}, indent=2))
        for s in status["signals"][:15]:
            print(f"{s['score']:+3d}  {s['side']:4s} {s['title'][:60]} · {s['outcome']}  "
                  f"agree {len(s['agree'])} / against {len(s['disagree'])}  leaders {s['avg_price']:.2f}"
                  f"  now {s['price'] if s['price'] is not None else '?'}  → {s['decision']}")
        return 0
    if cmd != "run":
        print("usage: python -m polybot [run|once|check]", file=sys.stderr)
        return 2

    web.serve(cfg.port, store, cfg.mode)
    engine.connect()
    log = logging.getLogger("polybot")
    log.info("Polybot in %s mode: top %d (%s), %d must agree, $%s a trade",
             engine.mode, cfg.top_n, cfg.window, cfg.min_agree, cfg.stake_usd)
    notifier.send("Polybot started",
                  f"{engine.mode} mode · watching the top {cfg.top_n} ({cfg.window}) · {cfg.min_agree} must agree · ${cfg.stake_usd} a trade",
                  priority="low", tags="robot")

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        try:
            engine.run_once()
        except Exception as e:  # keep going; the page shows the error
            log.exception("cycle failed")
            engine.record_error(f"cycle failed: {e}")
            try:
                store.write_status({**(store.read_status() or {}), "errors": engine.errors[-5:], "mode": engine.mode})
            except Exception:
                pass
        stop.wait(cfg.poll_seconds)
    log.info("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
