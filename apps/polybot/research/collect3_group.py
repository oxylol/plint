"""Every fill of a fixed group of wallets since a date (no 6,000-fill cap), for replaying a consensus setup.

    python collect3_group.py DATA_DIR WALLETS.json SINCE [rate] [max_pages]

Pages each wallet's trades (maker and taker, as the bot's LeaderFeed sees them) back from the newest until the
oldest fill is before SINCE. Writes DATA_DIR/group/<wallet>.json.gz: [[t, asset, cid, ±1, size, price, outcome
index, slug, event slug, title, outcome], ...]. Resumable per wallet.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from polymarket import PublicClient

DATA = Path(sys.argv[1])
WALLETS = json.loads(Path(sys.argv[2]).read_text())
SINCE = dt.datetime.fromisoformat(sys.argv[3]).replace(tzinfo=dt.timezone.utc).timestamp()
RATE = float(sys.argv[4]) if len(sys.argv) > 4 else 18.0
MAX_PAGES = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
OUT = DATA / "group"
OUT.mkdir(exist_ok=True)
c = PublicClient()
lock = threading.Lock()
nxt = [time.monotonic()]


def pace():
    with lock:
        t = max(time.monotonic(), nxt[0])
        nxt[0] = t + 1.0 / RATE
    time.sleep(max(0.0, t - time.monotonic()))


def call(fn):
    delay = 2
    for i in range(6):
        pace()
        try:
            return fn()
        except Exception:
            if i == 5:
                raise
            time.sleep(delay)
            delay = min(60, delay * 2)


def one(w):
    path = OUT / f"{w}.json.gz"
    if path.exists():
        return w, -1
    rows = []
    pages = c.list_trades(user=w, taker_only=False, page_size=500)
    page = call(pages.first_page)
    n = 1
    while True:
        for t in page.items:
            rows.append([int(t.timestamp.timestamp()), str(t.asset_id), str(t.condition_id), 1 if str(t.side).upper() == "BUY" else -1,
                         float(t.size), float(t.price), t.outcome_index, t.slug or "", t.event_slug or "", (t.title or "")[:90], t.outcome or ""])
        oldest = min((r[0] for r in rows), default=SINCE)
        if not page.next_cursor or oldest < SINCE or n >= MAX_PAGES:
            break
        cur = page.next_cursor
        page = call(lambda: pages.from_cursor(cur).first_page())
        n += 1
    rows = [r for r in rows if r[0] >= SINCE]
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump({"wallet": w, "pages": n, "complete": oldest < SINCE or not page.next_cursor, "fills": rows}, f)
    tmp.replace(path)
    return w, len(rows)


t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    for w, n in ex.map(one, WALLETS):
        print(f"{dt.datetime.now():%H:%M:%S} {w[:10]} {n} fills ({time.time() - t0:.0f}s)", flush=True)
print("group done", flush=True)
