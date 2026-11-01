"""Live scan: do one-winner events (negative-risk: soccer win/draw/win, temperature buckets, …) ever sell all their
outcomes for less than $1 after fees?

    python arb_scan.py DATA_DIR [hours_ahead=48] [rounds=1] [minutes_between=10]

Lists open negative-risk markets ending in the next `hours_ahead` hours from Gamma, groups them by event, reads every
outcome's Yes book and works out the cost of buying one share of each outcome at the best asks (plus each market's
taker fee rate × p × (1 − p)) and how many shares that price is good for. A set costing under $1 is a riskless profit
(exactly one outcome pays $1); the scan shows how often and how big. Appends DATA_DIR/arb/scan.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from polymarket import PublicClient

DATA = Path(sys.argv[1])
HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 48
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
GAP = float(sys.argv[4]) if len(sys.argv) > 4 else 10
U = dt.timezone.utc
c = PublicClient()


def scan():
    now = dt.datetime.now(U)
    ev = defaultdict(list)
    pages = c.list_markets(closed=False, end_date_min=now, end_date_max=now + dt.timedelta(hours=HOURS), page_size=100)
    page = pages.first_page()
    while True:
        for m in page.items:
            if m.state.neg_risk and m.events and m.outcomes.yes.token_id and m.state.active and not m.state.closed:
                fs = m.trading.fee_schedule if m.trading is not None else None
                ev[m.events[0].slug].append((str(m.outcomes.yes.token_id), float(fs.rate or 0) if fs is not None else 0.0, m.slug))
        if not page.next_cursor or not page.items:
            break
        cur = page.next_cursor
        page = pages.from_cursor(cur).first_page()
        time.sleep(0.1)
    toks = [t for v in ev.values() for t, _, _ in v]
    books = {}
    for i in range(0, len(toks), 40):
        try:
            for b in c.get_order_books(asset_ids=toks[i:i + 40]):
                books[str(b.asset_id)] = b
        except Exception:
            pass
        time.sleep(0.1)
    out = []
    for slug, legs in ev.items():
        if len(legs) < 2:
            continue
        cost = 0.0
        size = float("inf")
        ok = True
        for tok, rate, _ in legs:
            b = books.get(tok)
            asks = sorted((float(a.price), float(a.size)) for a in (b.asks or [])) if b else []
            if not asks:
                ok = False
                break
            p, s = asks[0]
            cost += p + rate * p * (1 - p)
            size = min(size, s)
        if ok:
            out.append({"t": now.isoformat(), "event": slug, "legs": len(legs), "cost": round(cost, 4), "shares": size})
    return out


with (DATA / "arb" / "scan.jsonl").open("a") as f:
    for r in range(ROUNDS):
        res = scan()
        for x in res:
            f.write(json.dumps(x) + "\n")
        f.flush()
        under = [x for x in res if x["cost"] < 1.0]
        costs = sorted(x["cost"] for x in res)
        print(f"{dt.datetime.now(U):%H:%M} {len(res)} complete one-winner events; cost of a full set: min {costs[0] if costs else 0:.3f}, "
              f"median {costs[len(costs) // 2] if costs else 0:.3f}; under $1: {len(under)}", flush=True)
        for x in sorted(under, key=lambda x: x["cost"])[:10]:
            print(f"   {x['event'][:60]:60s} {x['legs']:2d} legs  ${x['cost']:.3f} a set, {x['shares']:.0f} shares at that price", flush=True)
        if r < ROUNDS - 1:
            time.sleep(GAP * 60)
