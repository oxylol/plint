"""Collect a dataset of traders and their closed positions from Polymarket's public APIs.

    python collect.py OUT_DIR [max_traders]

Writes OUT_DIR/traders.json (who, from which boards) and OUT_DIR/positions.jsonl
(one closed position per line). Resumable: wallets already in positions.jsonl are skipped.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

from polymarket import PublicClient

OUT = Path(sys.argv[1])
MAX_TRADERS = int(sys.argv[2]) if len(sys.argv) > 2 else 700
PAGES = 5  # 500 closed positions per trader at most
OUT.mkdir(parents=True, exist_ok=True)
c = PublicClient()
now = dt.datetime.now(dt.timezone.utc)


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
            log("  retry:", kind, str(e)[:80])
            time.sleep(delay)
            delay *= 2


# ── 1. traders: every board we can read, plus ordinary holders of busy markets ──
traders: dict[str, dict] = {}
boards = [("month", None, 250), ("week", None, 150), ("day", None, 100), ("all", None, 150),
          ("month", "politics", 100), ("month", "sports", 100), ("month", "crypto", 60), ("month", "culture", 60),
          ("all", "politics", 60), ("week", "politics", 60)]
for window, cat, n in boards:
    try:
        kw = {"window": window, "sort_by": "PNL", "page_size": n}
        if cat:
            kw["category"] = cat
        items = call(lambda: c.list_trader_leaderboard(**kw).first_page()).items
    except Exception as e:
        log("board", window, cat, "failed:", str(e)[:80])
        continue
    for e in items[:n]:
        w = str(e.wallet).lower()
        t = traders.setdefault(w, {"wallet": w, "name": e.user_name or "", "boards": [], "source": "board"})
        t["boards"].append({"window": window, "category": cat or "overall", "rank": int(e.rank), "pnl": float(e.pnl), "volume": float(e.volume)})
    log(f"board {window}/{cat or 'overall'}: {len(items)} → {len(traders)} traders so far")

# Control group: holders of the busiest open markets (not selected for winning).
try:
    markets = call(lambda: c.list_markets(closed=False, order="volume24hr", ascending=False, page_size=40).first_page()).items
    holders_added = 0
    for m in markets:
        if not m.condition_id:
            continue
        try:
            page = call(lambda: c.list_market_holders(condition_ids=[str(m.condition_id)], page_size=100).first_page())
        except Exception as e:
            log("holders failed", m.slug, str(e)[:60])
            continue
        for item in page.items:
            inner = getattr(item, "holders", None)
            for h in (inner if inner is not None else [item]):
                w = str(getattr(h, "wallet", "") or "").lower()
                if not w or w in traders:
                    continue
                traders[w] = {"wallet": w, "name": getattr(h, "name", "") or getattr(h, "pseudonym", "") or "", "boards": [], "source": "holder"}
                holders_added += 1
    log(f"holders: +{holders_added} → {len(traders)} traders")
except Exception as e:
    log("control group failed:", str(e)[:100])

# Keep the boards' traders first, then a random sample of holders, up to MAX_TRADERS.
board_w = [w for w, t in traders.items() if t["source"] == "board"]
holder_w = [w for w, t in traders.items() if t["source"] == "holder"]
random.seed(7)
random.shuffle(holder_w)
keep = board_w[:MAX_TRADERS] + holder_w[: max(0, MAX_TRADERS - len(board_w))]
if len(board_w) > MAX_TRADERS:
    keep = board_w[:MAX_TRADERS - 150] + holder_w[:150]
traders = {w: traders[w] for w in keep}
(OUT / "traders.json").write_text(json.dumps(traders))
log(f"collecting {len(traders)} traders ({sum(1 for t in traders.values() if t['source']=='board')} from boards)")

# ── 2. closed positions, fills, account age ──
pos_path = OUT / "positions.jsonl"
done = set()
if pos_path.exists():
    for line in pos_path.open():
        try:
            done.add(json.loads(line)["wallet"])
        except Exception:
            pass
log(f"{len(done)} traders already collected")
t0 = time.time()
with pos_path.open("a") as out:
    for i, (w, t) in enumerate(traders.items()):
        if w in done:
            continue
        try:
            rows = []
            pages = c.list_positions(user=w, status="CLOSED", sort_by="TIMESTAMP", sort_direction="DESC", page_size=100)
            page = call(pages.first_page)
            n = 0
            while True:
                for p in page.items:
                    rows.append({
                        "wallet": w, "condition_id": str(p.condition_id), "asset_id": str(p.asset_id), "title": p.title, "slug": p.slug,
                        "event_slug": p.event_slug, "outcome": p.outcome, "outcome_index": p.outcome_index,
                        "avg_price": float(p.avg_price), "total_size": float(p.total_size), "realized_pnl": float(p.realized_pnl),
                        "first_entry_at": p.first_entry_at.isoformat() if p.first_entry_at else None,
                        "last_event_at": p.last_event_at.isoformat() if p.last_event_at else None,
                        "end_date": p.end_date.isoformat() if p.end_date else None, "negative_risk": p.negative_risk,
                    })
                n += 1
                if not page.next_cursor or n >= PAGES:
                    break
                page = call(pages.from_cursor(page.next_cursor).first_page)
            since = now - dt.timedelta(days=30)
            fills = call(lambda: c.list_trades(user=w, start=since, taker_only=False, page_size=500).first_page()).items
            taker = call(lambda: c.list_trades(user=w, start=since, taker_only=True, page_size=500).first_page()).items
            age = None
            try:
                st = call(lambda: c.get_user_stats(user=w))
                if st is not None and st.join_date:
                    age = (now - st.join_date.replace(tzinfo=st.join_date.tzinfo or dt.timezone.utc)).days
            except Exception:
                pass
            meta = {"wallet": w, "meta": True, "fills_30d": len(fills), "taker_30d": len(taker), "age_days": age,
                    "last_fill": max((x.timestamp.isoformat() for x in fills), default=None), "closed_fetched": len(rows)}
            out.write(json.dumps(meta) + "\n")
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
        except Exception as e:
            log("  failed", w[:10], type(e).__name__, str(e)[:80])
        if (i + 1) % 25 == 0:
            log(f"{i + 1}/{len(traders)} traders, {time.time() - t0:.0f}s")
log("done")
