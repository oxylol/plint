"""Closed events by tag over a long window (the markets come inside each event): OUT_DIR/events_long/<tag>.jsonl.gz
    python events_long.py OUT_DIR 2024-11-01 2026-09-20 geopolitics world middle-east pop-culture tech economy politics
"""
import datetime as dt, gzip, json, sys, time
from pathlib import Path
from polymarket import PublicClient

OUT = Path(sys.argv[1]) / "events_long"; OUT.mkdir(exist_ok=True)
A, B = (dt.datetime.fromisoformat(x).replace(tzinfo=dt.timezone.utc) for x in sys.argv[2:4])
c = PublicClient()


def call(fn):
    for i in range(7):
        try:
            time.sleep(0.12)
            return fn()
        except Exception:
            if i == 6:
                raise
            time.sleep(min(60, 2 ** (i + 1)))


iso = lambda x: x.isoformat() if x is not None else None
for tag in sys.argv[4:]:
    f = OUT / f"{tag}.jsonl.gz"
    if f.exists():
        continue
    n = ne = 0
    with gzip.open(OUT / f"{tag}.tmp", "wt") as fh:
        d = A
        while d < B:
            d2 = min(B, d + dt.timedelta(days=7))
            pages = c.list_events(closed=True, tag_slug=tag, end_date_min=d, end_date_max=d2, page_size=50)
            page = call(pages.first_page)
            while True:
                for e in page.items:
                    ne += 1
                    for m in (e.markets or []):
                        try:
                            yes, no = m.outcomes.yes, m.outcomes.no
                            row = {"cid": str(m.condition_id), "slug": m.slug, "q": m.question, "ev": e.slug, "tag": tag,
                                   "p": [float(yes.price) if yes.price is not None else None, float(no.price) if no.price is not None else None],
                                   "tok": [str(yes.token_id or ""), str(no.token_id or "")], "start": iso(m.state.start_date), "end": iso(m.state.end_date),
                                   "closed_time": iso(m.state.closed_time), "vol": float(m.metrics.volume_num or 0), "nr": bool(m.state.neg_risk),
                                   "fee_type": (m.trading.fee_type if m.trading is not None else None), "ev_end": iso(getattr(e, "end_date", None))}
                            fh.write(json.dumps(row) + "\n"); n += 1
                        except Exception as ex:
                            pass
                if not page.next_cursor or not page.items:
                    break
                cur = page.next_cursor
                page = call(lambda: pages.from_cursor(cur).first_page())
            d = d2
    (OUT / f"{tag}.tmp").replace(f)
    print(tag, "events", ne, "markets", n, flush=True)
