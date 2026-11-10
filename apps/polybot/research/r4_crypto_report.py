import numpy as np, sys
A = np.load('/opt/plint/data/polybot/research4/crypto_daily.npy')
ts, yp, tdir, usd, ln1, em1, ln3, em3, out, tauh, asset, mi, eday, size = A.T
days = np.unique(eday); wk = (eday // 7).astype(int); wks = np.unique(wk)
rng = np.random.default_rng(5)
print("trades", len(A), "markets", len(np.unique(mi)), "expiry days", len(days), "weeks", len(wks), "taker $M", round(usd.sum() / 1e6, 1))
def ll(p, y): p = np.clip(p, .002, .998); return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
def boot(r, c, cl):
    idx = np.searchsorted(cl, c); num = np.bincount(idx, r, len(cl)); den = np.bincount(idx, minlength=len(cl)).astype(float)
    bs = [num[s].sum() / max(den[s].sum(), 1) for s in (rng.integers(0, len(cl), len(cl)) for _ in range(500))]
    return np.percentile(bs, [5, 95])
for nm, m in (("all", tauh > 0), (">24h", tauh > 24), ("6-24h", (tauh > 6) & (tauh <= 24)), ("<6h", tauh <= 6)):
    print(f"log loss {nm:6s} n={m.sum():6d}: trade price {ll(yp[m], out[m]):.4f} | lognormal {ll(ln1[m], out[m]):.4f} | empirical {ll(em1[m], out[m]):.4f}")
def lg(x): x = np.clip(x, .002, .998); return np.log(x / (1 - x))
def fit(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(40):
        p = 1 / (1 + np.exp(-X @ w)); w -= np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (p - out_))
    return w
out_ = out
print("outcome ~ c + a*logit(price) + b*logit(empirical anchor): ", np.round(fit(np.column_stack([np.ones(len(A)), lg(yp), lg(em1)]), out), 3))
# calibration of both in the tails, as Yes probabilities folded to the cheap side
print("\ncheap side of each trade (the side priced under 50c): how often it won")
for src, P in (("price", yp), ("empirical anchor", em1), ("lognormal", ln1)):
    cp = np.minimum(P, 1 - P); cw = np.where(P < 0.5, out, 1 - out)
    s = "  " + f"{src:17s}"
    for lo, hi in ((.002, .01), (.01, .03), (.03, .06), (.06, .10), (.10, .20), (.20, .35), (.35, .5)):
        m = (cp >= lo) & (cp < hi); s += f" {100*lo:.1f}-{100*hi:.0f}c: {100*cp[m].mean():4.1f}->{100*cw[m].mean():4.1f}% (n={m.sum()//1000}k) |"
    print(s)
# anchored maker
mq = np.where(tdir < 0, yp, 1 - yp); mtok = np.where(tdir < 0, 0, 1); mwin = np.where(mtok == 0, out, 1 - out)
print("\nAnchored maker: bid = fair - margin on the given fair band, filled only by a through-print, one fill per market/side/hour, held to the end, no fee")
print(f"{'anchor':10s} {'lag':3s} {'fair band':10s} {'time left':9s} {'margin':6s} {'fills/day':>9s} {'ret/fill':>8s} {'90% CI by day':>16s} {'by week':>16s} {'win':>6s} {'paid':>6s} {'worst day':>9s}")
for an, lagn, P in (("empirical", "1m", em1), ("empirical", "3m", em3), ("lognormal", "1m", ln1)):
    fair = np.where(mtok == 0, P, 1 - P)
    for flo, fhi in ((.5, .8), (.8, .9), (.9, .96), (.96, .995), (.8, .995)):
        for tn, tm in (("all", tauh > 0.5), (">24h", tauh > 24), ("2-24h", (tauh > 2) & (tauh <= 24))):
            for marg in (0.0, .01, .02, .03, .05):
                if an != "empirical" and (tn != "all" or (flo, fhi) != (.8, .995)): continue
                if lagn == "3m" and (tn != "all" or (flo, fhi) != (.8, .995)): continue
                L = np.floor((fair - marg) * 100 + 1e-9) / 100
                f = tm & (fair >= flo) & (fair < fhi) & (L >= 0.02) & (mq < L - 1e-9)
                if f.sum() < 40: continue
                key = mi[f] * 1e8 + mtok[f] * 1e7 + (ts[f] // 3600 % 1e7)
                _, first = np.unique(key, return_index=True)
                ix = np.nonzero(f)[0][first]
                r = (mwin[ix] - L[ix]) / L[ix]
                d1 = boot(r, eday[ix], days); d2 = boot(r, wk[ix], wks)
                pd = np.array([r[eday[ix] == d].sum() for d in days])
                print(f"{an:10s} {lagn:3s} {flo:.2f}-{fhi:.3f} {tn:9s} {100*marg:4.0f}c {len(ix)/len(days):9.1f} {100*r.mean():+7.2f}% [{100*d1[0]:+6.2f},{100*d1[1]:+6.2f}] [{100*d2[0]:+6.2f},{100*d2[1]:+6.2f}] {100*mwin[ix].mean():5.1f}% {100*L[ix].mean():5.1f}c {5*pd.min():+8.0f}$")
