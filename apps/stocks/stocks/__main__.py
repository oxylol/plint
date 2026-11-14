"""Stocks: stock-trading models on paper, side by side.

    python -m stocks                 the page and the clock (what the container runs)
    python -m stocks run             one evening run now, with its reasoning in the log
    python -m stocks research        fit walk-forward, replay history, write the reports (design stretch)
    python -m stocks research --exam the stretch held back; counted every time it is run
    python -m stocks fit             fit the learned models on everything known now
    python -m stocks review          the agent's weekly look at its own record
    python -m stocks universe        this month's stocks, and what the name filter kept out
    python -m stocks status          keys, data, books, spend
    python -m stocks reset <model>   start a model's record over (after its rules changed)
    python -m stocks variant "what"  write down a change made to the design after seeing backtest figures
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading

from .config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stocks", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve", "run", "research", "fit", "review", "universe", "status", "reset", "variant"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--exam", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%m-%d %H:%M:%S")
    cfg = Config.from_env()
    if a.command == "serve":                       # the parent stays small: the page and the clock, nothing heavy
        from .runner import Runner
        from .store import Store
        from .web import serve
        store = Store(cfg.data_dir)
        serve(cfg, store)
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        threading.Thread(target=Runner(cfg, store, stop).loop, name="runner", daemon=True).start()
        stop.wait()
        return 0

    from .cal import next_run
    from .pipeline import Lab, evening
    lab = Lab(cfg)
    if a.command == "run":
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        try:
            print(evening(lab, now, next_run(now, cfg.run_at)))
        except Exception as e:
            lab.store.put("fault", f"The evening run stopped: {e}")
            lab.say(f"evening run stopped: {e}", "warn")
            lab.doing(None)
            raise
        return 0
    if a.command == "research":
        from . import research
        out = research.run(lab, exam=a.exam, say=lambda t: (print(t, flush=True), lab.doing(t)))
        lab.doing(None)
        print(json.dumps(out, indent=1))
        return 0
    if a.command == "fit":
        from . import research
        print(research.fit_live(lab, say=lambda t: (print(t, flush=True), lab.doing(t))))
        lab.doing(None)
        return 0
    if a.command == "review":
        from .review import weekly
        print(weekly(lab))
        return 0
    if a.command == "universe":
        from .universe import month_of
        last = lab.bars.last_day(cfg.index_symbol)
        if not last:
            print("no prices yet")
            return 1
        names = lab.assets.names()
        members = lab.universe.members(last)
        print(f"{len(members)} stocks for {month_of(last)}; the 40 most traded:")
        for s in members[:40]:
            print(f"  {s:6s} {names.get(s, '')}")
        frame = lab.assets.frame()
        out = frame[~frame["common"] & frame["active"] & frame["exchange"].isin(["NYSE", "NASDAQ", "AMEX"])]
        print(f"\nkept out by name on the main exchanges: {len(out)}; a sample to check by eye:")
        for r in out.sample(min(60, len(out)), random_state=1).itertuples():
            print(f"  {r.sym:8s} {r.name[:90]}")
        return 0
    if a.command == "status":
        st = lab.store.read_json("state.json") or {}
        print(json.dumps({"keys": cfg.keys(), "phase": st.get("phase"), "asof": st.get("asof"),
                          "last_price_day": lab.bars.last_day(cfg.index_symbol) if cfg.has_market_data or lab.bars.years() else None,
                          "universe_months": len(lab.universe.months()), "spend": st.get("spend"),
                          "models": {m["id"]: [m["state"], m.get("state_text"), m.get("vs_index")] for m in st.get("models", [])},
                          "notices": st.get("notices")}, indent=1))
        return 0
    if a.command == "reset":
        if not a.arg:
            print("which model?")
            return 2
        lab.store.reset(a.arg)
        lab.say(f"{a.arg}: record started over")
        return 0
    if a.command == "variant":
        if not a.arg:
            print("say what was changed")
            return 2
        lab.store.put("variants", (lab.store.get("variants") or []) + [a.arg])
        print(f"{len(lab.store.get('variants'))} design changes on record")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
