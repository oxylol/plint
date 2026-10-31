"""The real taker trades of thin markets after each signal: what a follower could actually have bought at.

    python collect3_tapes.py DATA_DIR [workers] [max_volume_usd=250000] [pages=6] [rate=6]

A midpoint means little in a thin book (an empty 1¢/99¢ book has a "mid" of 50¢), so for
markets whose lifetime volume is under max_volume_usd (and every temperature market) this
pages each market's trade tape back from the newest trade until it reaches the earliest
signal (or `pages` × 500 trades). The data API ignores start/end for a market's trades,
which is why busy markets, where the midpoint is reliable anyway, are left out.
Writes DATA_DIR/tapes/<condition id>.json.gz: {"complete_after": t, "trades": [[t, outcome 0|1, ±1, size, price]]}.
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

import pandas as pd
from polymarket import PublicClient

DATA = Path(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_VOL = float(sys.argv[3]) if len(sys.argv) > 3 else 250000
PAGES = int(sys.argv[4]) if len(sys.argv) > 4 else 6
RATE = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
OUT = DATA / "tapes"
OUT.mkdir(exist_ok=True)
c = PublicClient()
lock = threading.Lock()
nxt = [time.monotonic()]


def pace():
    with lock:
        t = max(time.monotonic(), nxt[0])
        nxt[0] = t + 1.0 / RATE
    time.sleep(max(0.0, t - time.monotonic()))


def log(*a):
    with lock:
        print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def markets(wanted):
    meta = {}
    with gzip.open(DATA / "markets" / "markets.jsonl.gz", "rt") as f:
        for line in f:
            j = line.find('"cid": "')
            if j < 0 or line[j + 8: j + 8 + 66] not in wanted:
                continue
            m = json.loads(line)
            if not m.get("missing"):
                meta[m["cid"]] = (m.get("vol") or 0.0, m.get("tokens") or ["", ""], m.get("pos_ids") or ["", ""], m.get("q") or "")
    return meta


def main():
    ev = pd.read_parquet(DATA / "events", columns=["cid", "wallet", "t_d", "final"],
                         filters=[("has_market", "=", True), ("rec_n", ">=", 5), ("leader_px", ">", 0.03), ("leader_px", "<", 0.97),
                                  ("t_d", ">=", time.time() - 62 * 86400)])
    ev = ev[ev.final.notna()]
    first = ev.groupby("cid").t_d.min()
    # markets the bot's pool (the rebuilt 30-day board's top 300) signalled in come first
    wd = pd.read_parquet(DATA / "wallet_days.parquet")
    wd["rank30"] = wd.groupby("day").pnl30.rank(ascending=False, method="first")
    top = set(zip(wd.wallet[wd.rank30 <= 300], wd.day[wd.rank30 <= 300]))
    day = (ev.t_d // 86400).astype(int)
    pool_cids = set(ev.cid.values[[(w, d) in top for w, d in zip(ev.wallet.values, day.values)]])
    del ev
    meta = markets(set(first.index))
    todo = []
    for cid, t0 in first.items():
        m = meta.get(cid)
        if m is None or (OUT / f"{cid}.json.gz").exists():
            continue
        if m[0] < MAX_VOL or "temperature in" in m[3]:
            todo.append((cid, float(t0), m))
    todo.sort(key=lambda x: (x[0] not in pool_cids, -x[1]))
    log(f"{sum(1 for x in todo if x[0] in pool_cids)} of them carry a pool signal (fetched first)")
    log(f"{len(todo)} thin markets to fetch (of {len(first)} with signals)")

    def one(item):
        cid, t0, (vol, toks, pids, q) = item
        ids = {str(toks[0]): 0, str(toks[1]): 1, str(pids[0]): 0, str(pids[1]): 1}
        rows = []
        oldest = None
        try:
            pages = c.list_trades(condition_id=cid, page_size=500)
            pace()
            page = pages.first_page()
            k = 1
            while True:
                for t in page.items:
                    i = ids.get(str(t.asset_id))
                    if i is None:
                        continue
                    ts = int(t.timestamp.timestamp())
                    rows.append([ts, i, 1 if str(t.side).upper() == "BUY" else -1, float(t.size), float(t.price)])
                    oldest = ts if oldest is None else min(oldest, ts)
                if not page.next_cursor or k >= PAGES or (oldest is not None and oldest < t0 - 60):
                    break
                cur = page.next_cursor
                pace()
                page = pages.from_cursor(cur).first_page()
                k += 1
            complete = 0 if not page.next_cursor else (oldest or 0)
        except Exception as e:
            log("failed", cid[:10], type(e).__name__, str(e)[:60])
            return 0
        tmp = OUT / f"{cid}.tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump({"complete_after": complete if complete else 1, "trades": rows}, f)
        tmp.replace(OUT / f"{cid}.json.gz")
        return len(rows)

    n = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(one, todo):
            n += 1
            if n % 1000 == 0:
                log(f"{n}/{len(todo)} tapes, {time.time() - t_start:.0f}s")
    log("tapes done")


if __name__ == "__main__":
    main()
