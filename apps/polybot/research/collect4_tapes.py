"""Taker tapes for a manifest of markets (jsonl.gz with cid, tok): rows [ts, tok(0 yes/1 no), side(+1 buy/-1 sell), size, price].
    python tapes_all.py OUT_DIR MANIFEST [rate=15] [max_pages=20]
"""
import gzip, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from polymarket import PublicClient

import os
OUT = Path(sys.argv[1]) / os.environ.get("TAPEDIR", "tapes")
OUT.mkdir(exist_ok=True)
MAN = sys.argv[2]
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
MAXP = int(sys.argv[4]) if len(sys.argv) > 4 else 20
c = PublicClient()
lock = threading.Lock()
nxt = [time.monotonic()]


def pace():
    with lock:
        w = max(time.monotonic(), nxt[0]); nxt[0] = w + 1 / RATE
    time.sleep(max(0.0, w - time.monotonic()))


def call(fn):
    for i in range(6):
        pace()
        try:
            return fn()
        except Exception:
            if i == 5:
                raise
            time.sleep(min(40, 2 ** (i + 1)))


def one(m):
    f = OUT / f"{m['cid']}.json.gz"
    if f.exists():
        return 0
    rows = []
    try:
        pages = c.list_trades(condition_id=m["cid"], full_history=True, page_size=500)
        page = call(pages.first_page)
        k = 1
        while True:
            rows += [[int(t.timestamp.timestamp()), 0 if str(t.asset_id) == m["tok"][0] else 1, 1 if str(t.side).upper() == "BUY" else -1,
                      float(t.size), float(t.price)] for t in page.items]
            if not page.next_cursor or not page.items or k >= MAXP:
                break
            cur = page.next_cursor
            page = call(lambda: pages.from_cursor(cur).first_page())
            k += 1
    except Exception as e:
        print("failed", m["cid"], str(e)[:80], flush=True)
        return 0
    tmp = OUT / f"{m['cid']}.tmp"
    with gzip.open(tmp, "wt") as fh:
        json.dump(rows, fh)
    tmp.replace(f)
    return len(rows)


todo = [json.loads(l) for l in gzip.open(MAN, "rt")]
todo = [m for m in todo if not (OUT / f"{m['cid']}.json.gz").exists()]
print(len(todo), "tapes to fetch", flush=True)
n = tot = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for k in ex.map(one, todo):
        n += 1; tot += k
        if n % 1000 == 0:
            print(n, tot, round(time.time() - t0), flush=True)
print("done", n, tot, flush=True)
