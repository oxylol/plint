"""What the carry rule would hold today: open geopolitics markets, 10-45 days to the end, favourite's ask 90-98.5c,
favourite's hourly mid >= 90c for the last 72 h."""
import datetime as dt, json, re, sys, time
from polymarket import PublicClient
c = PublicClient(); U = dt.timezone.utc; now = dt.datetime.now(U)
TAGS = ("geopolitics", "world", "middle-east")
seen = {}; nev = 0
for tag in TAGS:
    pages = c.list_events(closed=False, tag_slug=tag, end_date_min=now + dt.timedelta(days=1), end_date_max=now + dt.timedelta(days=120), page_size=50)
    page = pages.first_page()
    while True:
        for e in page.items:
            nev += 1
            for m in (e.markets or []):
                if m.state.closed or not m.state.accepting_orders or not m.state.end_date: continue
                seen.setdefault(str(m.condition_id), (e, m))
        if not page.next_cursor or not page.items: break
        page = pages.from_cursor(page.next_cursor).first_page(); time.sleep(0.1)
rows = []; band = 0
for cid, (e, m) in seen.items():
    left = (m.state.end_date - now).total_seconds() / 86400
    bb, ba = m.prices.best_bid, m.prices.best_ask
    if bb is None or ba is None or not (10 <= left < 45): continue
    bb, ba = float(bb), float(ba)
    mid = (bb + ba) / 2
    fav_yes = mid >= 0.5
    ask = ba if fav_yes else 1 - bb; bid = bb if fav_yes else 1 - ba
    if not (0.90 <= ask < 0.985) or ask - bid > 0.03: continue
    band += 1
    tok = str(m.outcomes.yes.token_id)
    try:
        pts = c.list_price_history(asset_id=tok, start=now - dt.timedelta(hours=96), end=now, bucket_seconds=3600).first_page().items
    except Exception:
        continue
    time.sleep(0.1)
    y = [(p.timestamp, float(p.price)) for p in pts if p.timestamp >= now - dt.timedelta(hours=72)]
    if len(y) < 48 or (pts[0].timestamp > now - dt.timedelta(hours=80)): continue
    fv = [v if fav_yes else 1 - v for _, v in y]
    if min(fv) < 0.90: continue
    rows.append((left, ask, bid, "Yes" if fav_yes else "No", float(m.metrics.volume_num or 0), e.slug, m.question))
print(f"{nev} open events, {len(seen)} open markets under the tags; {band} have a favourite at 90-98.5c with 10-45 days left; {len(rows)} also stable for 72 h")
series = lambda s: re.sub(r"january|february|march|april|may|june|july|august|september|october|november|december|\d+", "#", s)
print("distinct series:", len({series(r[5]) for r in rows}), "| mean ask", round(sum(r[1] for r in rows) / max(len(rows), 1), 3), "| mean days left", round(sum(r[0] for r in rows) / max(len(rows), 1), 1))
for r in sorted(rows, key=lambda r: -r[4])[:12]:
    print(f"  {r[3]:3s} at {r[1]:.3f} (bid {r[2]:.3f}), {r[0]:4.1f} d left, vol ${r[4]:,.0f}: {r[6][:80]}")
