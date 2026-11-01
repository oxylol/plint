# How far the price moves between a pre-game consensus buy and kick-off (closing-line value), and what selling there earns.
import json, gzip, sys
from collections import defaultdict
import numpy as np
sys.argv = ["clv", "/data"]; sys.path.insert(0, "/app/research"); sys.path.insert(0, "/app")
from fillmodel3 import FillModel
C = "/data/group_cache"
rows = json.load(open(f"{C}/consensus_rows_copied29.json"))
mk = json.load(open(f"{C}/markets.json"))
gsd = {}
for line in gzip.open("/data/markets/markets.jsonl.gz", "rt"):
    j = line.find('"game_start": "')
    if j >= 0:
        m = json.loads(line); gsd[m["cid"]] = m.get("game_start")
for k, v in mk.items():
    gsd.setdefault(k, v.get("game_start"))
paths = defaultdict(list)
for line in open(f"{C}/prices.jsonl"):
    j = json.loads(line); paths[j["a"]].extend(j["p"])
PP = {a: (np.array([x[0] for x in sorted(v)], float), np.array([x[1] for x in sorted(v)])) for a, v in paths.items() if v}
fm = FillModel("/data/fill_model.json")
import datetime as dt
def ts(s):
    if not s: return None
    d = dt.datetime.fromisoformat(s); return (d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)).timestamp()
def mid_before(a, t):
    if a not in PP: return None
    tt, pp = PP[a]; k = np.searchsorted(tt, t, side="right") - 1
    return float(pp[k]) if k >= 0 and t - tt[k] <= 900 else None
out = []
for r in rows:
    if r["score"] < 4 or r["in_play"]:
        continue
    gs = ts(gsd.get(r["cid"]))
    if gs is None or gs <= r["t"] + 600 or gs - r["t"] > 3 * 86400:
        continue
    m = mid_before(r["asset"], gs - 300)
    if m is None:
        continue
    px = max(0.001, m - fm.cost("sports", m, 2.0, False))
    fee = r["fee"]
    clv = px * (1 - fee * (1 - px)) / r["est"] - 1 - fee * (1 - r["est"])
    out.append((r["t"], r["ret"], clv, m - r["est"], (gs - r["t"]) / 3600))
o = np.array(out)
print(f"{len(o)} pre-game consensus buys (score >= 4, kick-off 10 min to 3 days ahead)")
print(f"hold to the end: mean {100*o[:,1].mean():+.1f}%, sd {100*o[:,1].std():.0f}%")
print(f"sell 5 min before kick-off: mean {100*o[:,2].mean():+.1f}%, sd {100*o[:,2].std():.0f}%, positive {100*(o[:,2]>0).mean():.0f}%")
print(f"mid moved from entry price to kick-off: mean {100*o[:,3].mean():+.2f} cents, median {100*np.median(o[:,3]):+.2f}")
for lo, hi in ((0, 2), (2, 8), (8, 24), (24, 72)):
    s = (o[:, 4] >= lo) & (o[:, 4] < hi)
    if s.sum() > 20:
        print(f"  bought {lo}-{hi} h before kick-off: n {s.sum():4d}, hold {100*o[s,1].mean():+.1f}%, sell at kick-off {100*o[s,2].mean():+.1f}% (sd {100*o[s,2].std():.0f}%)")
