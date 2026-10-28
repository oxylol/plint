"""Collect every closed market's resolution so positions can be scored at resolution, not at the trader's exit.

    python collect_markets.py OUT_DIR [since YYYY-MM-DD]

Writes OUT_DIR/markets.jsonl: condition id, slug, outcome labels and final prices, end date,
negative risk flag, tags. Pages the closed markets week by week by end date.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

from polymarket import PublicClient

OUT = Path(sys.argv[1])
SINCE = dt.datetime.fromisoformat(sys.argv[2]).replace(tzinfo=dt.timezone.utc) if len(sys.argv) > 2 else dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
OUT.mkdir(parents=True, exist_ok=True)
c = PublicClient()


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def call(fn, tries=5):
    delay = 3
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


seen = set()
path = OUT / "markets.jsonl"
if path.exists():
    for line in path.open():
        try:
            seen.add(json.loads(line)["condition_id"])
        except Exception:
            pass
log(f"{len(seen)} markets already collected")
t0 = time.time()
total = 0
with path.open("a") as out:
    start = SINCE
    end_all = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=60)
    while start < end_all:
        stop = start + dt.timedelta(days=1)
        pages = c.list_markets(closed=True, end_date_min=start, end_date_max=stop, page_size=100)
        try:
            page = call(pages.first_page)
        except Exception as e:
            log("window", start.date(), "failed:", str(e)[:80])
            start = stop
            continue
        n = 0
        wk = 0
        while True:
            for m in page.items:
                cid = str(m.condition_id or "")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                try:
                    yes, no = m.outcomes.yes, m.outcomes.no
                    out.write(json.dumps({
                        "condition_id": cid, "slug": m.slug, "question": m.question, "version": m.version,
                        "labels": [yes.label, no.label], "prices": [float(yes.price) if yes.price is not None else None, float(no.price) if no.price is not None else None],
                        "token_ids": [str(yes.token_id or ""), str(no.token_id or "")], "position_ids": [str(yes.position_id or ""), str(no.position_id or "")],
                        "end_date": m.state.end_date.isoformat() if m.state.end_date else None,
                        "closed_time": m.state.closed_time.isoformat() if m.state.closed_time else None,
                        "neg_risk": bool(m.state.neg_risk), "category": m.category, "tags": [t.label for t in (m.tags or []) if getattr(t, "label", None)],
                        "event_slug": m.events[0].slug if m.events else None, "sports_type": m.sports.sports_market_type if m.sports else None,
                        "game_start": m.sports.game_start_time.isoformat() if (m.sports and m.sports.game_start_time) else None,
                        "liquidity": float(m.metrics.liquidity_num or 0), "volume": float(m.metrics.volume_num or 0),
                    }) + "\n")
                    wk += 1
                except Exception as e:
                    log("  row failed", m.slug, str(e)[:60])
            n += 1
            if not page.next_cursor or n >= 80:
                break
            try:
                page = call(pages.from_cursor(page.next_cursor).first_page)
            except Exception as e:
                log("  page failed", start.date(), str(e)[:60])
                break
        out.flush()
        total += wk
        log(f"day {start.date()}: {wk} markets ({n} pages) · total {total} · {time.time() - t0:.0f}s")
        start = stop
log("done")
