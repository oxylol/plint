"""Every closed sports market of a window, straight from Gamma: a universe that does not depend on who traded it.

    python collect3_universe.py DATA_DIR [sample=12000] [rate=15]

markets.jsonl.gz was built from the collected wallets' positions, and the positions API leaves out resolved losing
positions, so a thin market is likelier to be on file when someone's bet in it won. This lists the closed moneyline,
child-moneyline, spread and total markets ending Sep 3 → Oct 30 day by day from Gamma (whatever their volume), writes
DATA_DIR/kickoff/universe.jsonl.gz, then draws a random sample of the resolved ones and fetches their kick-off price
paths like collect3_kickoff.py into markets_uni.json / prices_uni.jsonl.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from polymarket import PublicClient

DATA = Path(sys.argv[1])
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
TYPES = ["moneyline", "child_moneyline", "spreads", "totals"]
U = dt.timezone.utc
OUT = DATA / "kickoff"
c = PublicClient()
lock = threading.Lock()
nxt = [time.monotonic()]


def pace():
    with lock:
        w = max(time.monotonic(), nxt[0])
        nxt[0] = w + 1 / RATE
    time.sleep(max(0.0, w - time.monotonic()))


def call(fn):
    for i in range(6):
        pace()
        try:
            return fn()
        except Exception:
            if i == 5:
                raise
            time.sleep(2 * (i + 1))


def row(m):
    yes, no = m.outcomes.yes, m.outcomes.no
    fs = m.trading.fee_schedule if m.trading is not None else None
    return {"cid": str(m.condition_id), "slug": m.slug, "prices": [float(yes.price) if yes.price is not None else None,
                                                                    float(no.price) if no.price is not None else None],
            "tokens": [str(yes.token_id or ""), str(no.token_id or "")], "closed": bool(m.state.closed),
            "game_start": m.sports.game_start_time.isoformat() if (m.sports and m.sports.game_start_time) else None,
            "sports_type": m.sports.sports_market_type if m.sports else None, "vol": float(m.metrics.volume_num or 0),
            "fee_rate": float(fs.rate or 0) if fs is not None else 0.0}


def day_markets(day):
    out = []
    pages = c.list_markets(closed=True, end_date_min=day, end_date_max=day + dt.timedelta(days=1), sports_market_types=TYPES, page_size=100)
    page = call(pages.first_page)
    while True:
        out += [row(m) for m in page.items]
        if not page.next_cursor or not page.items:
            break
        cur = page.next_cursor
        page = call(lambda: pages.from_cursor(cur).first_page())
    return day, out


def main():
    up = OUT / "universe.jsonl.gz"
    if not up.exists():
        days = []
        d = dt.datetime(2026, 9, 3, tzinfo=U)
        while d < dt.datetime(2026, 10, 31, tzinfo=U):
            days.append(d)
            d += dt.timedelta(days=1)
        allm = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for day, ms in ex.map(day_markets, days):
                for m in ms:
                    allm[m["cid"]] = m
                print(f"{day:%m-%d} {len(ms)} markets ({len(allm)} so far)", flush=True)
        with gzip.open(up, "wt") as f:
            for m in allm.values():
                f.write(json.dumps(m) + "\n")
    allm = [json.loads(line) for line in gzip.open(up, "rt")]
    now = time.time()

    def ok(m):
        g = m["game_start"] and dt.datetime.fromisoformat(m["game_start"]).timestamp()
        return (m["sports_type"] in TYPES and g and now - 58 * 86400 <= g <= now - 86400 and m["closed"] and m["prices"][0] is not None
                and max(m["prices"]) >= 0.99 and m["tokens"][0])
    res = sorted((m for m in allm if ok(m)), key=lambda m: m["cid"])
    print(f"universe: {len(allm)} markets, {len(res)} resolved with a game start in the price window", flush=True)
    mp = OUT / "markets_uni.json"
    if not mp.exists():
        random.Random(11).shuffle(res)
        mp.write_text(json.dumps({"n_universe": len(res), **{m["cid"]: m for m in res[:SAMPLE]}}))
    mk = {k: v for k, v in json.loads(mp.read_text()).items() if k != "n_universe"}
    pp = OUT / "prices_uni.jsonl"
    done = set()
    if pp.exists():
        done = {json.loads(line)["cid"] for line in pp.open() if line.strip()}
    todo = sorted((m for m in mk.values() if m["cid"] not in done), key=lambda m: m["game_start"])

    def fetch(m):
        g = dt.datetime.fromisoformat(m["game_start"]).timestamp()
        try:
            pts = call(lambda: c.list_price_history(asset_id=m["tokens"][0], start=int(g - 3 * 3600), end=int(g + 4 * 3600),
                                                    bucket_seconds=300).first_page().items)
        except Exception:
            return m["cid"], None
        return m["cid"], [[int(p.timestamp.timestamp()), float(p.price)] for p in pts]
    print(f"{len(todo)} price paths to fetch", flush=True)
    with pp.open("a") as out, ThreadPoolExecutor(max_workers=8) as ex:
        for n, (cid, pts) in enumerate(ex.map(fetch, todo), 1):
            if pts is not None:
                out.write(json.dumps({"cid": cid, "p": pts}) + "\n")
            if n % 2000 == 0:
                out.flush()
                print(f"{dt.datetime.now():%H:%M:%S} {n}/{len(todo)}", flush=True)
    print("universe done", flush=True)


main()
