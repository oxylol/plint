import numpy as np, sys
a = np.load('/opt/plint/data/polybot/research4/wx_trades.npy')
q, win, usd, phase, ts_, day, ev, h, mtok, size = a.T
ret = (win - q) / q
w5 = np.minimum(usd, 5.0)
rng = np.random.default_rng(1)
days = np.unique(day)
def stat(mask, w):
    if mask.sum() < 50: return None
    r, ww, d = ret[mask], w[mask], day[mask]
    m = np.average(r, weights=ww)
    # cluster bootstrap by day
    idx = np.searchsorted(days, d)
    num = np.bincount(idx, r * ww, len(days)); den = np.bincount(idx, ww, len(days))
    bs = []
    for _ in range(300):
        s = rng.integers(0, len(days), len(days))
        bs.append(num[s].sum() / max(den[s].sum(), 1e-9))
    lo, hi = np.percentile(bs, [5, 95])
    return m, lo, hi, mask.sum(), ww.sum()
def show(name, mask):
    for wn, w in (("$5/trade", w5), ("dollar", usd)):
        s = stat(mask, w)
        if s: print(f"{name:42s} {wn:9s} {100*s[0]:+6.2f}% [{100*s[1]:+6.2f},{100*s[2]:+6.2f}] n={int(s[3]):8d} ${s[4]/1e3:9.0f}k")
print("days", len(days), "span", int(days.min()), int(days.max()))
show("all", q > 0)
for p, pn in ((0, "before day"), (1, "in play"), (2, "after day")):
    show(pn, phase == p)
tn = ["taker buys Yes", "taker buys No", "taker sells Yes", "taker sells No"]
for p, pn in ((0, "before"), (1, "in play")):
    for t in range(4):
        show(f"{pn}: {tn[t]}", (phase == p) & (ts_ == t))
edges = [0, .02, .05, .1, .2, .35, .5, .65, .8, .9, .95, .98, 1.0]
for p, pn in ((0, "before"), (1, "in play")):
    for mt, mn in ((0, "maker holds Yes"), (1, "maker holds No")):
        for lo, hi in zip(edges[:-1], edges[1:]):
            show(f"{pn}: {mn} q {lo:.2f}-{hi:.2f}", (phase == p) & (mtok == mt) & (q >= lo) & (q < hi))
