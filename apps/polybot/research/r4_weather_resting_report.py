import numpy as np
a = np.load('/opt/plint/data/polybot/research4/wx_resting.npy')
posted = np.load('/opt/plint/data/polybot/research4/wx_resting_posted.npy')
di, k, L, fin, phase, h, day, city, rk, spread, T, mi = a.T
ret = (fin - L) / L
days = np.unique(day)
rng = np.random.default_rng(2)
DEL = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
def stat(mask):
    n = int(mask.sum())
    if n < 80: return None
    r, d = ret[mask], day[mask]
    idx = np.searchsorted(days, d)
    num = np.bincount(idx, r, len(days)); den = np.bincount(idx, minlength=len(days)).astype(float)
    bs = [num[s].sum() / max(den[s].sum(), 1) for s in (rng.integers(0, len(days), len(days)) for _ in range(400))]
    lo, hi = np.percentile(bs, [5, 95])
    half = np.median(days)
    m1 = r[d <= half].mean() if (d <= half).any() else float('nan'); m2 = r[d > half].mean() if (d > half).any() else float('nan')
    return r.mean(), lo, hi, n, m1, m2, fin[mask].mean(), L[mask].mean()
def show(name, mask):
    s = stat(mask)
    if s: print(f"{name:50s} {100*s[0]:+7.2f}% [{100*s[1]:+7.2f},{100*s[2]:+7.2f}] n={s[3]:7d} ({s[3]/len(days):6.0f}/day) halves {100*s[4]:+6.1f}/{100*s[5]:+6.1f}  win {100*s[6]:5.1f}% paid {100*s[7]:5.1f}c")
print("days", len(days))
pn = ["before", "in play", "after"]; kn = ["Yes", "No"]
for p in (0, 1):
    for kk in (0, 1):
        for d in range(6):
            m = (phase == p) & (k == kk) & (di == d)
            fr = m.sum() / max(posted[d, kk, p], 1)
            show(f"{pn[p]} bid {kn[kk]} at ref-{DEL[d]*100:.0f}c (fill {100*fr:4.1f}%)", m)
print("\n-- by bid level, before the day")
edges = [0.01, .03, .06, .1, .2, .35, .5, .65, .8, .9, .95, 1.0]
for kk in (0, 1):
    for d in (1, 3, 4):
        for lo, hi in zip(edges[:-1], edges[1:]):
            show(f"before {kn[kk]} ref-{DEL[d]*100:.0f}c L {lo:.2f}-{hi:.2f}", (phase == 0) & (k == kk) & (di == d) & (L >= lo) & (L < hi))
print("\n-- by bid level, in play")
for kk in (0, 1):
    for d in (1, 3, 4):
        for lo, hi in zip(edges[:-1], edges[1:]):
            show(f"in play {kn[kk]} ref-{DEL[d]*100:.0f}c L {lo:.2f}-{hi:.2f}", (phase == 1) & (k == kk) & (di == d) & (L >= lo) & (L < hi))
