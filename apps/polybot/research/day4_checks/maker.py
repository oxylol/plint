# Taker (market order at the ask, fee) against maker entry (a resting bid, no fee) for day4's qualifying signals.
import json, sys, datetime as dt
from collections import defaultdict
import numpy as np
sys.argv = ["maker", "/data"]; sys.path.insert(0, "/app/research"); sys.path.insert(0, "/app")
from fillmodel3 import FillModel
C = "/data/group_cache"; U = dt.timezone.utc
LABEL = sys.argv_label if hasattr(sys, "argv_label") else None
fm = FillModel("/data/fill_model.json")
paths = defaultdict(list)
for line in open(f"{C}/prices.jsonl"):
    j = json.loads(line); paths[j["a"]].extend(j["p"])
PP = {a: (np.array([x[0] for x in sorted(v)], float), np.array([x[1] for x in sorted(v)])) for a, v in paths.items() if v}
T0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp(); SE = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
rng = np.random.default_rng(5)
for label in ("copied29", "sports144"):
    rows = json.load(open(f"{C}/consensus_rows_{label}.json"))
    tt = np.array([r["t"] for r in rows]); est = np.array([r["est"] for r in rows]); score = np.array([r["score"] for r in rows])
    avg = np.array([r["avg"] for r in rows]); asset = np.array([r["asset"] for r in rows])
    qual = (score >= 4) & (est >= .05) & (est <= .9) & (est <= avg + .05) & (tt >= T0)
    last = {}; pop = []
    for i in np.argsort(tt, kind="stable"):
        if qual[i] and not (asset[i] in last and tt[i] - last[asset[i]] < 72 * 3600):
            last[asset[i]] = tt[i]; pop.append(i)
    print(f"\n{label}: {len(pop)} day4-qualifying signals")
    for W in (30, 120):
        for delta in (0.0, 0.01):
            out = []
            for i in pop:
                r = rows[i]; a = r["asset"]
                if a not in PP: continue
                ts_, ps = PP[a]
                cost = r["est"] - 0  # taker price paid in the simulation
                k = np.searchsorted(ts_, r["t"], side="right") - 1
                if k < 0: continue
                mid = ps[k]
                L = round(mid - 0.005 - delta, 3)          # rest a bid half a cent under the mid (at the bid), or a cent lower
                j0, j1 = np.searchsorted(ts_, r["t"] + 1), np.searchsorted(ts_, r["t"] + W * 60, side="right")
                filled = j1 > j0 and ps[j0:j1].min() <= L - 0.005   # the market traded through our price within W minutes
                fee = r["fee"]; fin = r["final"]
                taker = r["ret"]
                if filled:
                    maker = fin / L - 1                          # no taker fee for a resting order
                else:
                    maker = None                                 # not filled: no trade
                    # or chase: buy at the ask when the window ends
                    k2 = min(len(ps) - 1, j1)
                    m2 = ps[k2]; p2 = min(.99, m2 + fm.cost("sports", m2, 6, False))
                    chase = fin / p2 - 1 - fee * (1 - p2)
                out.append((r["t"], taker, maker if filled else np.nan, maker if filled else chase, filled))
            o = np.array(out, dtype=float)
            f_ = o[:, 4] > 0
            sel = o[:, 0] < SE
            def m(x): return 100 * np.nanmean(x)
            print(f"  bid {100*(0.005+delta):.1f}c under the mid, wait {W:3d} min: filled {100*f_.mean():3.0f}% | "
                  f"taker (all) {m(o[:,1]):+5.1f}%  maker (filled only) {m(o[f_,2]):+5.1f}% vs taker on the same {m(o[f_,1]):+5.1f}%  | "
                  f"maker-or-chase (all) {m(o[:,3]):+5.1f}% [select {m(o[sel,3]):+5.1f}%, holdout {m(o[~sel,3]):+5.1f}%] vs taker [{m(o[sel,1]):+5.1f}%, {m(o[~sel,1]):+5.1f}%]")
