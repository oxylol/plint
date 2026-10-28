"""Second pass: positions that resolved but were never redeemed (mostly losses).

    python collect_settled.py OUT_DIR

The Data API's CLOSED set is exited positions: redeemed wins and early sales.
A position that lost at resolution is never redeemed, so it stays under
REDEEMABLE forever and is missing from CLOSED. This appends those rows to
positions.jsonl with "settled": true so the analysis sees every finished position.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

from polymarket import PublicClient

OUT = Path(sys.argv[1])
PAGES = 5
c = PublicClient()


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def call(fn, tries=4):
    delay = 3
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            kind = type(e).__name__
            if i == tries - 1 or not any(k in kind for k in ("Transport", "RateLimit", "Timeout", "Connect", "ReadError")):
                raise
            time.sleep(delay)
            delay *= 2


traders = json.loads((OUT / "traders.json").read_text())
pos_path = OUT / "positions.jsonl"
done = set()
for line in pos_path.open():
    try:
        r = json.loads(line)
        if r.get("settled_meta"):
            done.add(r["wallet"])
    except Exception:
        pass
log(f"{len(traders)} traders, {len(done)} already have settled rows")
t0 = time.time()
with pos_path.open("a") as out:
    for i, w in enumerate(traders):
        if w in done:
            continue
        try:
            rows = []
            pages = c.list_positions(user=w, status="REDEEMABLE", sort_by="TIMESTAMP", sort_direction="DESC", page_size=100)
            page = call(pages.first_page)
            n = 0
            while True:
                for p in page.items:
                    rows.append({
                        "wallet": w, "settled": True, "condition_id": str(p.condition_id), "asset_id": str(p.asset_id), "title": p.title,
                        "slug": p.slug, "event_slug": p.event_slug, "outcome": p.outcome, "outcome_index": p.outcome_index,
                        "avg_price": float(p.avg_price), "total_size": float(p.total_size), "current_size": float(p.current_size),
                        "current_price": float(p.current_price), "realized_pnl": float(p.realized_pnl), "total_pnl": float(p.total_pnl),
                        "first_entry_at": p.first_entry_at.isoformat() if p.first_entry_at else None,
                        "last_event_at": p.last_event_at.isoformat() if p.last_event_at else None,
                        "end_date": p.end_date.isoformat() if p.end_date else None, "negative_risk": p.negative_risk,
                    })
                n += 1
                if not page.next_cursor or n >= PAGES:
                    break
                page = call(pages.from_cursor(page.next_cursor).first_page)
            out.write(json.dumps({"wallet": w, "settled_meta": True, "settled_fetched": len(rows)}) + "\n")
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
        except Exception as e:
            log("  failed", w[:10], type(e).__name__, str(e)[:80])
        if (i + 1) % 50 == 0:
            log(f"{i + 1}/{len(traders)}, {time.time() - t0:.0f}s")
log("done")
