"""The copy study's part of the lab's evenings and clock.

- `refresh`: the House reports filed since the last look (the current year's index, fetched again once a
  day), then the people and trades tables built again with the insider purchases read tonight.
- `load`: what the live `copy` model needs (model.CopyData), or None before the first study.
- `due`: whether the study must run: never run yet, or the next trading day starts a half year it has not
  picked for (the picks are made from records up to the half year's last close)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import cal as calmod
from . import trades as tt
from .house import House, Stopped
from .model import CopyData

log = logging.getLogger("stocks.copytrade")


def folder(lab) -> Path:
    return Path(lab.cfg.cache_dir) / "copytrade"


def refresh(lab) -> dict | None:
    """Tonight's House reports and the tables built again. Only once the study has read the history."""
    if lab.store.read_json("copy/picks.json") is None:
        return None
    house = House(lab.cfg.cache_dir)
    year = datetime.now(timezone.utc).year
    years = [year - 1, year] if datetime.now(timezone.utc).month == 1 else [year]
    try:
        got = house.sync(years, log.info)
    except Stopped as e:
        lab.say(f"House reports not read tonight: {e}", "warn")
        got = {"stopped": str(e)}
    people, trades, _ = tt.build(house, lab.events.insiders, lab.assets.aliases())
    tt.save(folder(lab), people, trades)
    if got.get("trades"):
        lab.say(f"{got['reports']} new House trade report(s), {got['trades']} trades in them")
    return got


def load(lab) -> CopyData | None:
    picks = lab.store.read_json("copy/picks.json")
    if not picks or not picks.get("picks"):
        return None
    people, trades = tt.load(folder(lab))
    path = folder(lab) / "flags.parquet"
    fl = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["day", "sym", "ok"])
    flags = {(int(d), str(s)): bool(o) for d, s, o in zip(fl["day"], fl["sym"], fl["ok"])}
    to = int(str(picks.get("to") or "0000-00-00").replace("-", ""))
    cpath = folder(lab) / "calendar.npy"
    cal = np.load(cpath) if cpath.exists() else np.zeros(0, dtype=np.int64)
    wanted = {pid for p in picks["picks"] for pid in p["people"]}
    return CopyData(trades=trades[trades["pid"].isin(wanted)].reset_index(drop=True),
                    picks=[{"start": int(p["start"]), "people": list(p["people"])} for p in picks["picks"]],
                    names=dict(zip(people["pid"], people["name"])), calendar=cal, flags=flags, flags_to=to)


def symbols(data: CopyData | None) -> set[str]:
    """Every stock the live combo's people have traded: the evening's panel must hold them."""
    return set(data.trades["sym"]) if data is not None and len(data.trades) else set()


def _half(d: int) -> int:
    return d // 10000 * 10000 + (101 if d % 10000 < 701 else 701)


def due(store, last_day: int | None) -> bool:
    if store.read_json("reports/copy.json") is None or last_day is None:
        return True
    picks = (store.read_json("copy/picks.json") or {}).get("picks") or []
    nxt = calmod.next_trading_day(date(last_day // 10000, last_day // 100 % 100, last_day % 100))
    want = _half(nxt.year * 10000 + nxt.month * 100 + nxt.day)
    return not picks or max(int(p.get("half") or _half(int(p["start"]))) for p in picks) < want


__all__ = ["refresh", "load", "symbols", "due"]
