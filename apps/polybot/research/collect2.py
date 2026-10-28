"""Collect a large dataset of traders and their finished positions (exited + resolved-unclaimed).

    python collect2.py OUT_DIR [max_traders] [workers]

Writes OUT_DIR/traders.json and OUT_DIR/positions.jsonl. Resumable per trader.
Each trader: up to PAGES×100 closed positions, up to PAGES×100 resolved-unclaimed
(REDEEMABLE) positions, 30 days of fills (all and taker-only) and account age.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from polymarket import PublicClient

OUT = Path(sys.argv[1])
MAX_TRADERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PAGES = 8
CONTROL = 400
OUT.mkdir(parents=True, exist_ok=True)
c = PublicClient()
now = dt.datetime.now(dt.timezone.utc)
lock = threading.Lock()


def log(*a):
    with lock:
        print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def call(fn, tries=5):
    delay = 3
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            kind = type(e).__name__
            if i == tries - 1 or not any(k in kind for k in ("Transport", "RateLimit", "Timeout", "Connect", "ReadError", "Remote")):
                raise
            time.sleep(delay)
            delay *= 2


# ── 1. traders ───────────────────────────────────────────────
traders: dict[str, dict] = {}
if (OUT / "traders.json").exists():
    traders = json.loads((OUT / "traders.json").read_text())
    log(f"reusing {len(traders)} traders from traders.json")
else:
    boards = [("month", None, 1000), ("week", None, 600), ("day", None, 300), ("all", None, 600)]
    for cat in ("politics", "sports", "crypto", "culture", "geopolitics", "economics", "tech", "science", "finance"):
        boards += [("month", cat, 300), ("all", cat, 200), ("week", cat, 150)]
    for window, cat, n in boards:
        try:
            kw = {"window": window, "sort_by": "PNL", "page_size": min(n, 1000)}
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
    # Control group: holders of busy markets, across several sort orders so it is not just one niche.
    seen_markets = set()
    holders: dict[str, dict] = {}
    for order in ("volume24hr", "volume", "liquidity"):
        try:
            markets = call(lambda: c.list_markets(closed=False, order=order, ascending=False, page_size=60).first_page()).items
        except Exception as e:
            log("markets failed", order, str(e)[:60])
            continue
        for m in markets:
            if not m.condition_id or m.condition_id in seen_markets:
                continue
            seen_markets.add(m.condition_id)
            try:
                page = call(lambda: c.list_market_holders(condition_ids=[str(m.condition_id)], page_size=100).first_page())
            except Exception:
                continue
            for item in page.items:
                inner = getattr(item, "holders", None)
                for h in (inner if inner is not None else [item]):
                    w = str(getattr(h, "wallet", "") or "").lower()
                    if w and w not in traders and w not in holders:
                        holders[w] = {"wallet": w, "name": getattr(h, "name", "") or getattr(h, "pseudonym", "") or "", "boards": [], "source": "holder"}
    log(f"holders: {len(holders)} candidates")
    board_w = list(traders)
    random.seed(11)
    hw = list(holders)
    random.shuffle(hw)
    keep_boards = board_w[: max(0, MAX_TRADERS - CONTROL)]
    keep = {w: traders[w] for w in keep_boards}
    for w in hw[:CONTROL]:
        keep[w] = holders[w]
    traders = keep
    (OUT / "traders.json").write_text(json.dumps(traders))
log(f"collecting {len(traders)} traders ({sum(1 for t in traders.values() if t['source'] == 'board')} from boards)")

# ── 2. positions ─────────────────────────────────────────────
pos_path = OUT / "positions.jsonl"
done = set()
if pos_path.exists():
    for line in pos_path.open():
        try:
            r = json.loads(line)
            if r.get("meta"):
                done.add(r["wallet"])
        except Exception:
            pass
log(f"{len(done)} traders already collected")
out = pos_path.open("a")


def position_rows(w: str, status: str) -> list[dict]:
    rows = []
    pages = c.list_positions(user=w, status=status, sort_by="TIMESTAMP", sort_direction="DESC", page_size=100)
    page = call(pages.first_page)
    n = 0
    while True:
        for p in page.items:
            rows.append({
                "wallet": w, "settled": status == "REDEEMABLE", "condition_id": str(p.condition_id), "asset_id": str(p.asset_id),
                "title": p.title, "slug": p.slug, "event_slug": p.event_slug, "outcome": p.outcome, "outcome_index": p.outcome_index,
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
    return rows


def collect(w: str) -> str:
    try:
        closed = position_rows(w, "CLOSED")
        settled = position_rows(w, "REDEEMABLE")
        since = now - dt.timedelta(days=30)
        fills = call(lambda: c.list_trades(user=w, start=since, taker_only=False, page_size=500).first_page()).items
        taker = call(lambda: c.list_trades(user=w, start=since, taker_only=True, page_size=500).first_page()).items
        age = None
        try:
            st = call(lambda: c.get_user_stats(user=w))
            if st is not None and st.join_date:
                jd = st.join_date if st.join_date.tzinfo else st.join_date.replace(tzinfo=dt.timezone.utc)
                age = (now - jd).days
        except Exception:
            pass
        meta = {"wallet": w, "meta": True, "settled_meta": True, "fills_30d": len(fills), "taker_30d": len(taker), "age_days": age,
                "last_fill": max((x.timestamp.isoformat() for x in fills), default=None), "closed_fetched": len(closed), "settled_fetched": len(settled)}
        with lock:
            for r in closed + settled:
                out.write(json.dumps(r) + "\n")
            out.write(json.dumps(meta) + "\n")
            out.flush()
        return "ok"
    except Exception as e:
        log("  failed", w[:10], type(e).__name__, str(e)[:80])
        return "fail"


todo = [w for w in traders if w not in done]
t0 = time.time()
n_done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for res in ex.map(collect, todo):
        n_done += 1
        if n_done % 100 == 0:
            log(f"{n_done}/{len(todo)} traders, {time.time() - t0:.0f}s")
out.close()
log("done")
