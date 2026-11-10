"""Every closed market Gamma lists for a window of end dates, whatever its type or volume: one gz file per day.

    python universe_all.py OUT_DIR 2026-07-01 2026-11-08 [rate=12]
"""
import datetime as dt, gzip, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from polymarket import PublicClient

OUT = Path(sys.argv[1]) / "universe"
A, B = (dt.datetime.fromisoformat(x).replace(tzinfo=dt.timezone.utc) for x in sys.argv[2:4])
RATE = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
c = PublicClient()
lock = threading.Lock()
nxt = [time.monotonic()]


def pace():
    with lock:
        w = max(time.monotonic(), nxt[0]); nxt[0] = w + 1 / RATE
    time.sleep(max(0.0, w - time.monotonic()))


def call(fn):
    for i in range(7):
        pace()
        try:
            return fn()
        except Exception as e:
            if i == 6:
                raise
            time.sleep(min(60, 2 ** (i + 1)))


def iso(x):
    return x.isoformat() if x is not None else None


def row(m):
    yes, no = m.outcomes.yes, m.outcomes.no
    tr = m.trading
    fs = tr.fee_schedule if tr is not None else None
    ev = (m.events[0] if getattr(m, "events", None) else None)
    return {"cid": str(m.condition_id), "slug": m.slug, "q": m.question,
            "p": [float(yes.price) if yes.price is not None else None, float(no.price) if no.price is not None else None],
            "tok": [str(yes.token_id or ""), str(no.token_id or "")],
            "start": iso(m.state.start_date), "end": iso(m.state.end_date), "closed_time": iso(m.state.closed_time),
            "nr": bool(m.state.neg_risk), "vol": float(m.metrics.volume_num or 0),
            "fee_type": (tr.fee_type if tr is not None else None), "fee": float(fs.rate or 0) if fs is not None else 0.0,
            "sport": (m.sports.sports_market_type if m.sports else None),
            "ev": (getattr(ev, "slug", None) if ev is not None else None),
            "uma": (m.resolution.uma_resolution_status if m.resolution else None)}


def day_markets(day):
    f = OUT / f"{day:%Y%m%d}.jsonl.gz"
    if f.exists():
        return day, -1
    out = []
    pages = c.list_markets(closed=True, end_date_min=day, end_date_max=day + dt.timedelta(days=1), page_size=100)
    page = call(pages.first_page)
    while True:
        out += [row(m) for m in page.items]
        if not page.next_cursor or not page.items:
            break
        cur = page.next_cursor
        page = call(lambda: pages.from_cursor(cur).first_page())
    tmp = OUT / f"{day:%Y%m%d}.tmp"
    with gzip.open(tmp, "wt") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(f)
    return day, len(out)


days = []
d = A
while d < B:
    days.append(d); d += dt.timedelta(days=1)
days.reverse()
with ThreadPoolExecutor(max_workers=6) as ex:
    for day, n in ex.map(day_markets, days):
        print(f"{day:%m-%d} {n}", flush=True)
