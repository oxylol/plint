"""Check the fill model against the bot's own paper fills: what each live buy paid above the midpoint at that moment.

    python validate3_fills.py DATA_DIR TRADES_JSONL

Paper fills walk the live order book for the stake (core.estimate), which is exactly what the research simulates
with "mid + cost". For every BUY in the bot's trade log this fetches the 5-minute midpoint history around it and
compares (paid − mid) with the fill model's cost for the same type, price and time to the end.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from polymarket import PublicClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fillmodel3 import FillModel  # noqa: E402
from polybot.categories import category  # noqa: E402

DATA = Path(sys.argv[1])
TRADES = Path(sys.argv[2])
c = PublicClient()
fm = FillModel(DATA / "fill_model.json")
rows = [json.loads(l) for l in TRADES.read_text().splitlines() if l.strip()]
buys = [r for r in rows if r.get("side") == "BUY" and r.get("price") and r.get("strategy") not in ("underdog",)]
out = []
for r in buys:
    t = dt.datetime.fromisoformat(r["at"])
    try:
        pts = c.list_price_history(asset_id=r["asset_id"], start=t - dt.timedelta(minutes=15), end=t + dt.timedelta(minutes=15),
                                   bucket_seconds=60).first_page().items
    except Exception:
        continue
    before = [p for p in pts if p.timestamp <= t]
    if not before:
        continue
    mid = float(before[-1].price)
    cat = category(r.get("title"), r.get("event_slug") or r.get("slug"))
    cost = fm.cost(cat, mid, 24.0, False)
    out.append({"strategy": r["strategy"], "cat": cat, "paid": r["price"], "mid": mid, "over_mid": r["price"] - mid, "model_cost": cost})
(DATA / "live" / "fill_check.json").write_text(json.dumps(out))
g = defaultdict(list)
for o in out:
    g["weather" if o["cat"] == "weather" else "sports" if o["cat"] == "sports" else "other"].append(o)
print(f"{len(out)} of {len(buys)} paper buys matched to a 1-minute midpoint")
for k, v in g.items():
    om = np.array([x["over_mid"] for x in v])
    mc = np.array([x["model_cost"] for x in v])
    print(f"  {k:8s} n {len(v):3d}  paid above mid: median {100 * np.median(om):+.1f}¢, mean {100 * om.mean():+.1f}¢, p75 {100 * np.quantile(om, .75):+.1f}¢ · "
          f"fill model says {100 * np.median(mc):+.1f}¢ (median)")
