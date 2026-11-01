"""Live check: in upcoming sports markets, how does the price history's latest value compare with the real order book?

    python kickoff_books_now.py DATA_DIR [max_markets=400]

For open moneyline / spread / total markets in markets.jsonl.gz whose game starts in the next 12 hours, fetches the
first outcome's latest 5-minute price and both order books now, and reports the underdog side's history price
against the book's midpoint and best ask, split by volume so far. If thin markets' asks sit far above their history
price, the history's "cheap underdogs" cannot be bought.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from polymarket import PublicClient

DATA = Path(sys.argv[1])
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 400
c = PublicClient()
now = time.time()


def ts(x):
    return dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp() if x else None


cand = []
with gzip.open(DATA / "markets" / "markets.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        g = ts(d.get("game_start"))
        if d.get("sports_type") in ("moneyline", "spreads", "totals") and g and now + 600 <= g <= now + 12 * 3600 and not d.get("closed") and d.get("tokens"):
            cand.append(d)
rng = np.random.default_rng(0)
rng.shuffle(cand)
cand = cand[:MAX]
print(f"{len(cand)} open markets starting in the next 12 hours")
out = []
for i in range(0, len(cand), 20):
    chunk = cand[i:i + 20]
    try:
        books = c.get_order_books(asset_ids=[m["tokens"][0] for m in chunk])
    except Exception as e:
        print("books failed", type(e).__name__)
        continue
    bk = {str(b.asset_id): b for b in books}
    for m in chunk:
        a = m["tokens"][0]
        try:
            pts = c.list_price_history(asset_id=a, start=int(now - 3600), end=int(now), bucket_seconds=300).first_page().items
        except Exception:
            continue
        if not pts or a not in bk:
            continue
        h = float(pts[-1].price)
        b = bk[a]
        bids = [float(x.price) for x in (b.bids or [])]
        asks = [float(x.price) for x in (b.asks or [])]
        if not bids or not asks:
            continue
        bid, ask = max(bids), min(asks)
        under0 = h <= 0.5
        # underdog side: token 0 if its history ≤ 50¢, else token 1 (its ask = 1 − token 0's best bid)
        hu, au, mu = (h, ask, (bid + ask) / 2) if under0 else (1 - h, 1 - bid, 1 - (bid + ask) / 2)
        out.append((hu, mu, au, ask - bid, float(m.get("vol") or 0), m["sports_type"]))
    time.sleep(0.5)
X = np.array([o[:5] for o in out])
print(f"{len(X)} with a history price and a two-sided book")
print(f"{'volume so far':>14s} {'n':>4s} | {'underdog history':>16s} {'book mid':>9s} {'best ask':>9s} | {'ask − history':>13s} {'spread':>7s} | ask > history + 3¢")
for lo, hi in ((0, 1e3), (1e3, 5e3), (5e3, 2e4), (2e4, 1e5), (1e5, 1e12)):
    s = X[(X[:, 4] >= lo) & (X[:, 4] < hi)]
    if len(s) < 5:
        continue
    print(f"${lo:>12,.0f}+ {len(s):4d} | {100 * s[:, 0].mean():15.1f}¢ {100 * s[:, 1].mean():8.1f}¢ {100 * s[:, 2].mean():8.1f}¢ | "
          f"{100 * np.median(s[:, 2] - s[:, 0]):+11.1f}¢ {100 * np.median(s[:, 3]):6.1f}¢ | {100 * (s[:, 2] > s[:, 0] + 0.03).mean():5.0f}%")
cheap = X[X[:, 0] < 0.2]
if len(cheap):
    print(f"underdogs the history prices under 20¢: {len(cheap)}, history {100 * cheap[:, 0].mean():.1f}¢, best ask {100 * cheap[:, 2].mean():.1f}¢ "
          f"(median ask/history {np.median(cheap[:, 2] / cheap[:, 0]):.2f}×)")
