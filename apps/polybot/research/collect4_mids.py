"""12-hour midpoint history (Yes token) for the last 60 days of each market in a manifest: OUT_DIR/mids12h.jsonl.gz"""
import datetime as dt, gzip, json, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from polymarket import PublicClient
R = Path(sys.argv[1]); MAN = sys.argv[2]
c = PublicClient(); lock = threading.Lock()
ends = {}
for f in (R / "events_long").glob("*.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        r = json.loads(l); ends[r["cid"]] = r["end"]
todo = [json.loads(l) for l in gzip.open(MAN, "rt")]
out = gzip.open(R / "mids12h.jsonl.gz", "at")
def one(m):
    e = ends.get(m["cid"])
    if not e: return 0
    end = dt.datetime.fromisoformat(e); pts = []
    for k in range(4):
        a, b = end - dt.timedelta(days=15 * (k + 1)), end - dt.timedelta(days=15 * k)
        for i in range(4):
            try:
                time.sleep(0.15)
                pg = c.list_price_history(asset_id=m["tok"][0], start=a, end=b, bucket_seconds=43200).first_page().items
                pts += [[int(p.timestamp.timestamp()), float(p.price)] for p in pg]
                break
            except Exception:
                time.sleep(2 * (i + 1))
    with lock:
        out.write(json.dumps({"cid": m["cid"], "p": sorted(pts)}) + "\n")
    return len(pts)
n = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    for k in ex.map(one, todo):
        n += 1
        if n % 100 == 0: print(n, flush=True)
out.close(); print("done", n)
