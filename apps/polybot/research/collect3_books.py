"""Order-book depth snapshots of the live-tradeable universe, for the fill model.

    python collect3_books.py OUT_DIR [minutes_between=20] [hours=48]

Every few minutes: the busiest open markets (by 24h volume and by liquidity),
every daily temperature market of the next three days and every market a
leader in the bot's page is trading, both outcomes each. For every book it
keeps the best bid and ask, the top 12 levels a side and what buying (or
selling) $1, $5, $10, $20, $50 and $100 at market would cost, so the research
can ask "what does a $5 or $20 follower really pay above the mid here?".
Writes OUT_DIR/books/<UTC hour>.jsonl.gz, one line per book. Safe to restart.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

from polymarket import PublicClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polybot.categories import category  # noqa: E402

OUT = Path(sys.argv[1]) / "books"
EVERY = float(sys.argv[2]) if len(sys.argv) > 2 else 20
HOURS = float(sys.argv[3]) if len(sys.argv) > 3 else 48
STATUS = Path("/pb/status.json")   # the bot's page JSON (data/polybot mounted read-only at /pb), when available
OUT.mkdir(parents=True, exist_ok=True)
c = PublicClient()
SIZES = (1, 5, 10, 20, 50, 100)


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def call(fn, tries=4):
    delay = 2
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def walk(levels, usd, buy=True):
    """VWAP for spending `usd` (buy) or for selling `usd` worth at the touch (sell); None when the book is too thin."""
    spent = got = 0.0
    for p, s in levels:
        room = p * s
        take = min(room, usd - spent)
        spent += take
        got += take / p
        if spent >= usd - 1e-9:
            return spent / got
    return None


def universe(now):
    markets = {}
    for order in ("volume24hr", "liquidity"):
        pages = c.list_markets(closed=False, order=order, ascending=False, page_size=100)
        page = call(pages.first_page)
        n = 0
        while True:
            for m in page.items:
                if m.condition_id:
                    markets[str(m.condition_id)] = m
            n += 1
            if not page.next_cursor or n >= (30 if order == "volume24hr" else 5):
                break
            page = call(lambda: pages.from_cursor(page.next_cursor).first_page())
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(3):   # temperature markets end at noon UTC on their day
        lo = start + dt.timedelta(days=d, hours=11, minutes=58)
        pages = c.list_markets(closed=False, end_date_min=lo, end_date_max=lo + dt.timedelta(minutes=4), page_size=100)
        page = call(pages.first_page)
        n = 0
        while True:
            for m in page.items:
                if m.condition_id and "temperature in" in (m.question or ""):
                    markets[str(m.condition_id)] = m
            n += 1
            if not page.next_cursor or n >= 40:
                break
            page = call(lambda: pages.from_cursor(page.next_cursor).first_page())
    # markets the bot's leaders are trading right now
    try:
        st = json.loads(STATUS.read_text())
        slugs = {s.get("slug") for v in st.get("variants", []) for s in v.get("signals", []) if s.get("slug")}
        for slug in list(slugs)[:300]:
            try:
                m = call(lambda: c.get_market(slug=slug))
                if m is not None and m.condition_id and not m.state.closed:
                    markets[str(m.condition_id)] = m
            except Exception:
                pass
    except Exception:
        pass
    return markets


def snapshot(markets, now, out):
    tokens = []
    for cid, m in markets.items():
        for i, o in enumerate((m.outcomes.yes, m.outcomes.no)):
            tid = o.token_id if m.version != "v2" else (o.position_id or o.token_id)
            if tid:
                tokens.append((str(tid), cid, i, m))
    n = 0
    for k in range(0, len(tokens), 50):
        chunk = tokens[k:k + 50]
        try:
            books = call(lambda: c.get_order_books(asset_ids=[t[0] for t in chunk]))
        except Exception as e:
            log("books failed:", str(e)[:80])
            continue
        by_id = {str(b.asset_id): b for b in books}
        for tid, cid, i, m in chunk:
            b = by_id.get(tid)
            if b is None:
                continue
            asks = sorted(((float(l.price), float(l.size)) for l in b.asks), key=lambda x: x[0])
            bids = sorted(((float(l.price), float(l.size)) for l in b.bids), key=lambda x: -x[0])
            tr = m.trading
            rate = float(tr.fee_schedule.rate or 0) if (tr is not None and tr.fees_enabled and tr.fee_schedule is not None) else 0.0
            row = {
                "t": now.isoformat(timespec="seconds"), "asset_id": tid, "condition_id": cid, "idx": i,
                "slug": m.slug, "event_slug": m.events[0].slug if m.events else None, "q": (m.question or "")[:120],
                "cat": category(m.question, (m.events[0].slug if m.events else None) or m.slug),
                "sports_type": m.sports.sports_market_type if m.sports else None,
                "game_start": m.sports.game_start_time.isoformat() if (m.sports and m.sports.game_start_time) else None,
                "end": m.state.end_date.isoformat() if m.state.end_date else None, "neg_risk": bool(m.state.neg_risk),
                "fee": rate, "liq": float(m.metrics.liquidity_num or 0), "vol24": float(m.metrics.volume_24hr or 0),
                "tick": float(b.tick_size) if getattr(b, "tick_size", None) else None,
                "bid": bids[0][0] if bids else None, "ask": asks[0][0] if asks else None,
                "asks": asks[:12], "bids": bids[:12],
                "buy": {str(s): walk(asks, s) for s in SIZES},
                "sell": {str(s): walk(bids, s) for s in SIZES},
            }
            out.write(json.dumps(row) + "\n")
            n += 1
    return n


t_end = time.time() + HOURS * 3600
while time.time() < t_end:
    t0 = time.time()
    now = dt.datetime.now(dt.timezone.utc)
    try:
        mk = universe(now)
        with gzip.open(OUT / f"{now:%Y%m%d%H}.jsonl.gz", "at") as out:
            n = snapshot(mk, now, out)
        log(f"{len(mk)} markets, {n} books in {time.time() - t0:.0f}s")
    except Exception as e:
        log("cycle failed:", type(e).__name__, str(e)[:120])
    time.sleep(max(30, EVERY * 60 - (time.time() - t0)))
log("done")
