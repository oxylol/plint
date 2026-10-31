"""Price paths around every signal: what the market looked like when the follower could act.

    python collect3_prices.py DATA_DIR [workers] [min_t]

For each outcome token in DATA_DIR/events (signals on resolved markets), fetch the
CLOB price history (the midpoint) in 5-minute buckets from 10 minutes before a
signal to the market's resolution, in spans of at most ~15 days (the API's limit),
so both the entry and the path to the end are known.
Polymarket keeps 5-minute buckets for about 60 days and hourly ones a little
longer, so older signals get hourly buckets. Writes DATA_DIR/prices/<aa>.jsonl.gz
(append-only; a line per request) and resumes from what is there.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from polymarket import PublicClient

DATA = Path(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MIN_T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
SAMPLE = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0   # share of the other signals to price (pool and unbiased wallets always)
U = dt.timezone.utc
FIVE_MIN_SINCE = time.time() - 59 * 86400      # 5-minute buckets exist for about 60 days
SPAN = 14.8 * 86400
c = PublicClient()
lock = threading.Lock()
OUT = DATA / "prices"
OUT.mkdir(exist_ok=True)


class Pace:
    def __init__(self, rate):
        self.gap, self.next, self.lock = 1.0 / rate, time.monotonic(), threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


PACE = Pace(25)


def log(*a):
    with lock:
        print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def windows(have):
    """Per token: spans of at most ~15 days (the API's limit) from 10 minutes before a signal to the market's resolution,
    for the signals not already inside a span fetched earlier."""
    ev = pd.read_parquet(DATA / "events", columns=["asset", "wallet", "t_d", "final", "resolved_at", "leader_px", "rec_n", "has_market",
                                                    "pnl_boards"])
    ev = ev[ev.final.notna() & ev.has_market & (ev.t_d >= MIN_T) & (ev.leader_px > 0.03) & (ev.leader_px < 0.97) & (ev.rec_n >= 5)]
    if SAMPLE < 1.0:
        # every signal the bot's pool could have sent (the 30-day board's top 300, rebuilt from resolved P&L), twice the
        # share of the signals of wallets no profit board picked, a fixed random share of the rest (by token: spans stay whole)
        wd = pd.read_parquet(DATA / "wallet_days.parquet")
        wd["rank30"] = wd.groupby("day").pnl30.rank(ascending=False, method="first")
        top = set(zip(wd.wallet[wd.rank30 <= 300], wd.day[wd.rank30 <= 300]))
        day = (ev.t_d // 86400).astype(int)
        keep = np.array([(w, d) in top for w, d in zip(ev.wallet.values, day.values)])
        u = np.array([(hash(a) % 1000) / 1000.0 for a in ev.asset.values])
        ev = ev[keep | ((ev.pnl_boards.values == 0) & (u < 2 * SAMPLE)) | (u < SAMPLE)]
    spans = defaultdict(list)
    for asset, g in ev.groupby("asset"):
        got = have.get(asset, [])
        g = g.sort_values("t_d")
        lo = hi = None
        for t, res in zip(g.t_d, g.resolved_at):
            if any(a <= t - 300 and t + 1800 <= b for a, b in got):
                continue
            end = min(res if res == res else t + SPAN, t + SPAN)
            if lo is None:
                lo, hi = t - 600, end
            elif t <= hi and max(hi, end) - lo <= SPAN:
                hi = max(hi, end)
            else:
                spans[asset].append((lo, hi))
                lo, hi = t - 600, end
        if lo is not None:
            spans[asset].append((lo, max(hi, lo + 3600)))
    return spans


def done_spans():
    have = defaultdict(list)
    for p in OUT.glob("*.jsonl.gz"):
        try:
            with gzip.open(p, "rt") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        have[j["a"]].append((j["lo"], j["hi"]))
                    except Exception:
                        pass
        except EOFError:
            pass
    return have


FILES: dict[str, object] = {}


def write(asset, row):
    k = asset[-2:]
    with lock:
        f = FILES.get(k)
        if f is None:
            f = FILES[k] = gzip.open(OUT / f"{k}.jsonl.gz", "at")
        f.write(json.dumps(row) + "\n")


def fetch(item):
    asset, lo, hi = item
    bucket = 300 if lo >= FIVE_MIN_SINCE else 3600
    for i in range(5):
        PACE.wait()
        try:
            pts = c.list_price_history(asset_id=asset, start=int(lo), end=int(hi), bucket_seconds=bucket).first_page().items
            write(asset, {"a": asset, "lo": int(lo), "hi": int(hi), "b": bucket,
                          "p": [[int(p.timestamp.timestamp()), float(p.price)] for p in pts]})
            return len(pts)
        except Exception as e:
            if i == 4:
                log("failed", asset[:10], type(e).__name__, str(e)[:60])
                return -1
            time.sleep(2 * (i + 1))


def main():
    spans = windows(done_spans())
    todo = [(a, lo, hi) for a, ss in spans.items() for lo, hi in ss]
    todo.sort(key=lambda x: -x[1])     # newest first: the 5-minute history expires oldest-first
    log(f"{len(todo)} spans over {len(spans)} tokens to fetch")
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(fetch, todo):
            n += 1
            if n % 2000 == 0:
                rate = n / (time.time() - t0)
                log(f"{n}/{len(todo)} spans, {rate:.1f}/s, eta {(len(todo) - n) / rate / 60:.0f} min")
                with lock:
                    for f in FILES.values():
                        f.flush()
    for f in FILES.values():
        f.close()
    log("prices done")


if __name__ == "__main__":
    main()
