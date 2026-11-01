"""Which model filters day4's signals best? Every qualifying signal, paired comparisons, weekly refits.

    python consensus_filter.py DATA_DIR [label]

Uses consensus_model.py's saved candidate table. The population is every day4-qualifying signal (4+ more leaders
agreeing than against, the $5 buy priced 5–90¢ and at most 5¢ above the leaders, held to resolution), the first
qualifying moment per outcome per 72 hours, Sep 15 → Oct 31. Each model is refitted every week on candidates resolved
before that week and drops the signals whose expected return is below a cut, exactly as day4model does. Models are
compared on the same signals (paired by day); the family is picked on Sep 15 → Oct 19 and judged on Oct 20 → 31.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
LABEL = sys.argv[2] if len(sys.argv) > 2 else "copied29"
C = DATA / "group_cache"
U = dt.timezone.utc
rng = np.random.default_rng(21)
FEATS = json.loads((C / f"consensus_{LABEL}.json").read_text())["feats"]
X = np.load(C / f"consensus_X_{LABEL}.npy")
rows = json.loads((C / f"consensus_rows_{LABEL}.json").read_text())
y = np.array([r["final"] for r in rows], float)
est = np.array([r["est"] for r in rows], float)
tt = np.array([r["t"] for r in rows], float)
res = np.array([r["res"] for r in rows], float)
ret = np.array([r["ret"] for r in rows], float)
fee = np.array([r["fee"] for r in rows], float)
score = np.array([r["score"] for r in rows])
avg = np.array([r["avg"] for r in rows], float)
asset = np.array([r["asset"] for r in rows])
BASE = [f for f in FEATS if not f.startswith("aq_") and f != "mom15"]
QUAL = [f for f in FEATS if f.startswith("aq_")]
TEST0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp()
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()


def lgt(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(idx, cols, lam, w8=None):
    Xi = X[idx][:, cols]
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-6
    Z = np.hstack([(Xi - mu) / sd, np.ones((len(idx), 1))])
    off = lgt(est[idx])
    sw = np.ones(len(idx)) if w8 is None else w8
    w = np.zeros(Z.shape[1])
    I = np.eye(Z.shape[1])
    I[-1, -1] = 0
    for _ in range(30):
        p = sig(Z @ w + off)
        w -= np.linalg.solve(Z.T @ (Z * (sw * p * (1 - p))[:, None]) + lam * I, Z.T @ (sw * (p - y[idx])) + lam * I @ w)
    return lambda j: sig(np.hstack([(X[j][:, cols] - mu) / sd, np.ones((len(j), 1))]) @ w + lgt(est[j]))


def logit_model(lam, feats=None, min_score=2, decay=None):
    cols = [FEATS.index(f) for f in (feats or BASE)]

    def fit(tr, w0):
        tr = tr[score[tr] >= min_score]
        w8 = 0.5 ** ((w0 - tt[tr]) / (decay * 86400)) if decay else None
        return fit_logit(tr, cols, lam, w8)
    return fit


def trees(cfg):
    def fit(tr, w0):
        import lightgbm as lgb
        m = lgb.LGBMClassifier(objective="binary", verbose=-1, num_threads=3, **cfg)
        cols = [FEATS.index(f) for f in BASE]
        m.fit(X[tr][:, cols], y[tr], init_score=lgt(est[tr]))
        return lambda j: sig(m.predict(X[j][:, cols], raw_score=True) + lgt(est[j]))
    return fit


def ensemble(*fits):
    def fit(tr, w0):
        ms = [f(tr, w0) for f in fits]
        return lambda j: sig(np.mean([lgt(m(j)) for m in ms], axis=0))
    return fit


MODELS = {
    "shipped (l1000)": logit_model(1000.0),
    "l3000": logit_model(3000.0), "l300": logit_model(300.0), "l10000": logit_model(10000.0),
    "l1000, trained on score>=3": logit_model(1000.0, min_score=3),
    "l1000, trained on score>=4": logit_model(1000.0, min_score=4),
    "l1000, 21-day half-life": logit_model(1000.0, decay=21),
    "l1000 + who agrees": logit_model(1000.0, BASE + QUAL),
    "ensemble l300/l1000/l3000": ensemble(logit_model(300.0), logit_model(1000.0), logit_model(3000.0)),
    "trees (anchored, small)": trees(dict(num_leaves=4, learning_rate=0.03, n_estimators=150, min_child_samples=300, reg_lambda=30.0)),
}

# the population: day4-qualifying signals, the first qualifying moment per outcome per 72 hours
order = np.argsort(tt, kind="stable")
qual = (score >= 4) & (est >= 0.05) & (est <= 0.90) & (est <= avg + 0.05) & (tt >= TEST0)
last = {}
keep = []
for i in order:
    if not qual[i]:
        continue
    if asset[i] in last and tt[i] - last[asset[i]] < 72 * 3600:
        continue
    last[asset[i]] = tt[i]
    keep.append(i)
pop = np.array(keep)
print(f"{len(pop)} day4-qualifying signals from Sep 15 (all {len(rows)} candidates are training data)")

weeks = []
s = TEST0
while s < tt.max():
    weeks.append((s, s + 7 * 86400))
    s += 7 * 86400
ER = {k: np.full(len(rows), np.nan) for k in MODELS}
for w0, w1 in weeks:
    tr = np.where((tt < w0) & (res < w0))[0]
    te = pop[(tt[pop] >= w0) & (tt[pop] < w1)]
    if not len(te):
        continue
    for k, mk in MODELS.items():
        p = mk(tr, w0)(te)
        ER[k][te] = p / est[te] - 1 - fee[te] * (1 - est[te])
days = (tt // 86400).astype(int)
dlist = np.arange(days[pop].min(), days[pop].max() + 1)


def daily(sel):
    """$ per day at $5 a signal (no budget) for the selected signals."""
    return np.array([ret[sel][days[sel] == d].sum() for d in dlist]) * 5


def stats(sel):
    r = ret[sel]
    dd = days[sel]
    ud = np.unique(dd)
    sums = np.array([r[dd == x].sum() for x in ud])
    cnts = np.array([(dd == x).sum() for x in ud])
    dr = rng.integers(0, len(ud), size=(4000, len(ud)))
    bm = sums[dr].sum(1) / cnts[dr].sum(1)
    return len(r), r.mean(), np.quantile(bm, .05), np.quantile(bm, .95)


def paired(sel_a, sel_b):
    a, b = daily(sel_a), daily(sel_b)
    dr = rng.integers(0, len(dlist), size=(10000, len(dlist)))
    g = 7 * (a[dr].mean(1) - b[dr].mean(1))
    return np.median(g), np.quantile(g, .05), np.quantile(g, .95), (g > 0).mean()


sel_part = tt[pop] < SELECT_END
n, m, lo, hi = stats(pop)
print(f"\nday4's rule, every signal: {n} signals, {100 * m:+.1f}% a signal [{100 * lo:+.1f}, {100 * hi:+.1f}], ${7 * daily(pop).mean():+.1f}/week at $5 (no budget)")
print(f"\n{'model (keeps expected return ≥ cut)':38s} {'cut':>5s} {'kept':>5s} {'mean':>7s} {'90% CI':>16s} | {'select':>7s} {'holdout':>8s} | vs shipped $/wk [90% CI] P(better)")
results = {}
for k in MODELS:
    for cut in (-0.02, 0.0, 0.02):
        ok = np.isfinite(ER[k][pop]) & (ER[k][pop] >= cut)
        sel = pop[ok]
        n, m, lo, hi = stats(sel)
        ms = ret[sel[tt[sel] < SELECT_END]].mean()
        mh = ret[sel[tt[sel] >= SELECT_END]].mean()
        base = pop[np.isfinite(ER["shipped (l1000)"][pop]) & (ER["shipped (l1000)"][pop] >= -0.02)]
        g = paired(sel, base) if k != "shipped (l1000)" or cut != -0.02 else (0, 0, 0, 0)
        sel_week = 7 * daily(sel[tt[sel] < SELECT_END]).sum() / max(1, len(dlist[dlist < SELECT_END // 86400]))
        results[(k, cut)] = {"n": n, "mean": m, "lo": lo, "hi": hi, "select": ms, "holdout": mh, "select_week": sel_week, "gap": g}
        print(f"{k:38s} {cut:+5.2f} {n:5d} {100 * m:+6.1f}% [{100 * lo:+6.1f}, {100 * hi:+6.1f}] | {100 * ms:+6.1f}% {100 * mh:+7.1f}% | "
              f"{g[0]:+6.1f} [{g[1]:+6.1f}, {g[2]:+6.1f}] {100 * g[3]:3.0f}%")
best = max(results, key=lambda k: results[k]["select_week"])
b = results[best]
print(f"\npicked on Sep 15 → Oct 19 by $ a week: {best[0]} at cut {best[1]:+.2f} (select ${b['select_week']:+.1f}/wk); "
      f"holdout {100 * b['holdout']:+.1f}% a signal against the shipped model's {100 * results[('shipped (l1000)', -0.02)]['holdout']:+.1f}%")
json.dump({f"{k[0]}|{k[1]}": {kk: (list(map(float, vv)) if isinstance(vv, tuple) else float(vv)) for kk, vv in v.items()} for k, v in results.items()},
          open(C / f"consensus_filter_{LABEL}.json", "w"), indent=1)
