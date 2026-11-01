# Does entering sooner after a consensus forms pay? Entry at the 5-minute mid before the poll, at the poll, and later.
import json, sys, datetime as dt
from collections import defaultdict
import numpy as np
sys.argv = ["speed", "/data"]; sys.path.insert(0, "/app/research"); sys.path.insert(0, "/app")
from fillmodel3 import FillModel
C = "/data/group_cache"; U = dt.timezone.utc
fm = FillModel("/data/fill_model.json")
paths = defaultdict(list)
for line in open(f"{C}/prices.jsonl"):
    j = json.loads(line); paths[j["a"]].extend(j["p"])
PP = {a: (np.array([x[0] for x in sorted(v)], float), np.array([x[1] for x in sorted(v)])) for a, v in paths.items() if v}
T0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp(); SE = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
rng = np.random.default_rng(8)
def mid_before(a, t):
    ts_, ps = PP[a]; k = np.searchsorted(ts_, t, side="right") - 1
    return ps[k] if k >= 0 and t - ts_[k] <= 600 else None
for label in ("copied29", "sports144"):
    rows = json.load(open(f"{C}/consensus_rows_{label}.json"))
    tt = np.array([r["t"] for r in rows]); est = np.array([r["est"] for r in rows]); score = np.array([r["score"] for r in rows])
    avg = np.array([r["avg"] for r in rows]); asset = np.array([r["asset"] for r in rows])
    qual = (score >= 4) & (est >= .05) & (est <= .9) & (est <= avg + .05) & (tt >= T0)
    last = {}; pop = []
    for i in np.argsort(tt, kind="stable"):
        if qual[i] and not (asset[i] in last and tt[i] - last[asset[i]] < 72 * 3600):
            last[asset[i]] = tt[i]; pop.append(i)
    res = defaultdict(list)
    for i in pop:
        r = rows[i]; a = r["asset"]
        if a not in PP: continue
        ok = True; vals = {}
        for name, dtm in (("5 min sooner", -300), ("as simulated", 0), ("5 min later", 300), ("15 min later", 900), ("1 h later", 3600)):
            m = mid_before(a, r["t"] + dtm)
            if m is None: ok = False; break
            p = min(.99, m + fm.cost("sports", m, 6, r["in_play"]))
            vals[name] = r["final"] / p - 1 - r["fee"] * (1 - p)
        if ok:
            for k, v in vals.items(): res[k].append((r["t"], v))
    print(f"\n{label}: {len(res['as simulated'])} signals with prices at every delay")
    base = np.array([v for _, v in res["as simulated"]])
    for k, lst in res.items():
        v = np.array([x[1] for x in lst]); t_ = np.array([x[0] for x in lst])
        d = v - base
        dr = rng.integers(0, len(d), (4000, len(d)))
        lo, hi = np.quantile(d[dr].mean(1), [.05, .95])
        print(f"  enter {k:13s} {100*v.mean():+6.1f}% a signal (select {100*v[t_ < SE].mean():+5.1f}%, holdout {100*v[t_ >= SE].mean():+5.1f}%)  "
              f"minus as simulated {100*d.mean():+5.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]")
