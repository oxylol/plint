"""Post-hoc check on the saved out-of-sample predictions: does adding the live `learned` setup's two filters
(a second pool leader agrees; the follower pays at most 5¢ above the leaders) to the policy grid help?

    python posthoc3.py DATA_DIR

Labelled post hoc in REPORT3.md: the idea came from seeing learned_now's result. The same nested walk-forward as
analyze3.py picks each fold's policy on the folds before it, so the filters are only used where they earned it.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
DAY = 86400
rng = np.random.default_rng(5)
P = pd.read_parquet(DATA / "panel.parquet")
P = P[(P.p_f >= 0.02) & (P.p_f <= 0.98) & np.isfinite(P.f_logit_p)].sort_values("t_d").reset_index(drop=True)
P["hours"] = ((P.end - P.t_d) / 3600).clip(lower=0)
pr = pd.read_parquet(DATA / "oos_preds.parquet")
assert len(pr) == len(P)
fold_of = pr.fold.values
pf = P.p_f.values
SELECT_END = pd.Timestamp("2026-10-20", tz="UTC").timestamp()
pool = P.in_pool.values == 1
FOLD_START = {}
for f in sorted(set(fold_of[fold_of >= 0])):
    FOLD_START[f] = P.t_d.values[fold_of == f].min()


def qualify(idx, er, pol, st):
    lo, hi = pol["band"]
    ok = (er[idx] >= pol["thr"]) & (pf[idx] >= lo) & (pf[idx] <= hi) & (P.hours.values[idx] >= pol["min_h"]) & (P.hours.values[idx] <= pol["max_h"])
    if not pol["in_play"]:
        ok &= P.f_in_game.values[idx] < 0.5
    if pol.get("crowd"):
        ok &= P.crowd.values[idx] >= 1
    if pol.get("drift") is not None:
        ok &= P.f_drift.values[idx] <= pol["drift"]
    cand = idx[ok]
    cand = cand[np.argsort(P.t_d.values[cand], kind="stable")]
    last, busy = st.setdefault("last", {}), st.setdefault("busy", {})
    out = []
    for i in cand:
        a, e, t = P.asset.values[i], P.ev.values[i], P.t_d.values[i]
        if (a in last and t - last[a] < 72 * 3600) or busy.get(e, -1) > t:
            continue
        busy[e] = P.resolved_at.values[i]
        last[a] = t
        out.append((i, P.ret.values[i], int(t // DAY)))
    return out


def summarize(tr):
    if not tr:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "pnl_week": 0.0, "mean_c": np.nan}
    rr = np.array([x[1] for x in tr])
    dd = np.array([x[2] for x in tr])
    ud = np.unique(dd)
    sums = np.array([rr[dd == d].sum() for d in ud])
    cnts = np.array([(dd == d).sum() for d in ud])
    dr = rng.integers(0, len(ud), size=(2000, len(ud)))
    bm = sums[dr].sum(1) / cnts[dr].sum(1)
    span = max(7.0, dd.max() - dd.min() + 1)
    return {"n": len(rr), "mean": rr.mean(), "mean_c": np.minimum(rr, 3).mean(), "lo": np.quantile(bm, .05), "hi": np.quantile(bm, .95),
            "pnl_week": 5 * 7 * rr.mean() * min(10, len(rr) / span)}


GRID = []
for thr in (0.0, 0.04, 0.08, 0.12, 0.2, 0.3):
    for band in ((0.05, 0.95), (0.15, 0.85), (0.3, 0.7)):
        for min_h in (0.0, 1.0, 6.0, 24.0):
            for max_h in (72.0, 336.0, 1e9):
                if max_h <= min_h:
                    continue
                for inplay in (True, False):
                    for crowd in (False, True):
                        for drift in (None, 0.05):
                            GRID.append({"thr": thr, "band": band, "min_h": min_h, "max_h": max_h, "in_play": inplay, "crowd": crowd, "drift": drift})


def walk(er):
    folds = sorted(FOLD_START)
    per = defaultdict(dict)
    for f in folds:
        idx = np.where((fold_of == f) & pool & np.isfinite(er))[0]
        for k, pol in enumerate(GRID):
            per[f][k] = qualify(idx, er, pol, {})
    picks = []
    for j, f in enumerate(folds):
        if j < 2:
            continue
        prev = folds[:j]
        days = 7.0 * len(prev)

        def score(k):
            tr = [x for g in prev for x in per[g][k]]
            return np.mean([min(x[1], 3) for x in tr]) * min(10, len(tr) / days) if len(tr) >= 30 else -1e9
        picks.append((f, max(range(len(GRID)), key=score)))
    st, tr = {}, []
    for f, k in picks:
        idx = np.where((fold_of == f) & pool & np.isfinite(er))[0]
        tr += qualify(idx, er, GRID[k], st)
    return tr, picks


out = {}
for fam in ("logit_anchor", "gbm_anchor", "old_r2"):
    p = pr[f"p_{fam}"].values.astype(float)
    er = p / pf - 1 - P.fee_rate.values * (1 - pf)
    tr, picks = walk(er)
    a = [x for x in tr if P.t_d.values[x[0]] < SELECT_END]
    b = [x for x in tr if P.t_d.values[x[0]] >= SELECT_END]
    s, sa, sb = summarize(tr), summarize(a), summarize(b)
    used = [GRID[k] for _, k in picks]
    out[fam] = {"all": s, "select": sa, "holdout": sb, "crowd_folds": sum(u["crowd"] for u in used), "drift_folds": sum(u["drift"] is not None for u in used),
                "folds": len(used), "last": used[-1]}
    print(f"{fam:13s} {s['n']:4d} signals, mean {100 * s['mean']:+.1f}% [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}] (capped {100 * s['mean_c']:+.1f}%), "
          f"select {100 * sa['mean']:+.1f}% / holdout {100 * sb['mean']:+.1f}% [{100 * sb['lo']:+.1f}, {100 * sb['hi']:+.1f}]; "
          f"${s['pnl_week']:+.2f}/week; folds using 'second leader': {out[fam]['crowd_folds']}/{len(used)}, 'within 5¢': {out[fam]['drift_folds']}/{len(used)}; "
          f"last policy {used[-1]}")
(DATA / "posthoc3.json").write_text(json.dumps(out, default=float, indent=1))
