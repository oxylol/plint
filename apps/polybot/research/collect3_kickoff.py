"""Price paths of every liquid sports market around kick-off, for a model of the price itself (no leaders).

    python collect3_kickoff.py DATA_DIR [min_volume] [rate]

Picks resolved moneyline, child-moneyline, spread and total markets whose game started in the last 58 days (while
Polymarket still keeps 5-minute buckets) with at least `min_volume` traded, and fetches the first outcome's midpoint
in 5-minute buckets from 3 hours before kick-off to 4 hours after. Writes DATA_DIR/kickoff/markets.json (the markets)
and DATA_DIR/kickoff/prices.jsonl (a line per market; resumes from what is there).
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
MIN_VOL = float(sys.argv[2]) if len(sys.argv) > 2 else 5000.0
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
TYPES = ("moneyline", "child_moneyline", "spreads", "totals")
OUT = DATA / "kickoff"
OUT.mkdir(exist_ok=True)
c = PublicClient()
lock = threading.Lock()
nxt = [time.monotonic()]


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def pick():
    now = time.time()
    rows = {}
    with gzip.open(DATA / "markets" / "markets.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            g = ts(d.get("game_start"))
            if d.get("sports_type") not in TYPES or not g or not now - 58 * 86400 <= g <= now - 86400:
                continue
            if not d.get("closed") or not d.get("prices") or max(d["prices"]) < 0.99 or (d.get("vol") or 0) < MIN_VOL:
                continue
            rows[d["cid"]] = {k: d.get(k) for k in ("cid", "slug", "q", "labels", "prices", "tokens", "game_start", "end", "closed_time",
                                                    "sports_type", "ev", "ev_title", "vol", "fee_rate", "tick")}
    return rows


def fetch(m):
    g = ts(m["game_start"])
    a = m["tokens"][0]
    for i in range(5):
        with lock:
            w = max(time.monotonic(), nxt[0])
            nxt[0] = w + 1 / RATE
        time.sleep(max(0.0, w - time.monotonic()))
        try:
            pts = c.list_price_history(asset_id=a, start=int(g - 3 * 3600), end=int(g + 4 * 3600), bucket_seconds=300).first_page().items
            return m["cid"], [[int(p.timestamp.timestamp()), float(p.price)] for p in pts]
        except Exception:
            time.sleep(2 * (i + 1))
    return m["cid"], None


def main():
    mp = OUT / "markets.json"
    if mp.exists():
        mk = json.loads(mp.read_text())
    else:
        mk = pick()
        mp.write_text(json.dumps(mk))
    pp = OUT / "prices.jsonl"
    done = set()
    if pp.exists():
        for line in pp.open():
            try:
                done.add(json.loads(line)["cid"])
            except ValueError:
                pass
    todo = sorted((m for m in mk.values() if m["cid"] not in done), key=lambda m: m["game_start"])   # oldest first: they expire first
    print(f"{len(mk)} markets, {len(done)} done, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    with pp.open("a") as out, ThreadPoolExecutor(max_workers=8) as ex:
        for n, (cid, pts) in enumerate(ex.map(fetch, todo), 1):
            if pts is not None:
                out.write(json.dumps({"cid": cid, "p": pts}) + "\n")
            if n % 2000 == 0:
                out.flush()
                print(f"{dt.datetime.now():%H:%M:%S} {n}/{len(todo)} ({n / (time.time() - t0):.1f}/s)", flush=True)
    print("kickoff done", flush=True)


main()
