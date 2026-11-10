import numpy as np, json, sys
R = '/opt/plint/data/polybot/research4'; TAG = sys.argv[1] if len(sys.argv) > 1 else 'other'
A = np.load(f'{R}/{TAG}_A.npy'); B = np.load(f'{R}/{TAG}_B.npy'); meta = json.load(open(f'{R}/{TAG}_meta.json'))
fams = {v: k for k, v in meta['fams'].items()}
rng = np.random.default_rng(7)
def boot(r, w, c):
    u, idx = np.unique(c, return_inverse=True)
    num = np.bincount(idx, r * w); den = np.bincount(idx, w)
    if len(u) < 8: return (np.nan, np.nan), len(u)
    bs = [num[s].sum() / max(den[s].sum(), 1e-9) for s in (rng.integers(0, len(u), len(u)) for _ in range(400))]
    return np.percentile(bs, [5, 95]), len(u)
q, win, usd, mtok, hte, fam, ev, htc, ts = A.T.astype(np.float64)
ret = (win - q) / q; w5 = np.minimum(usd, 5)
print("A. Every taker trade seen from the maker's side ($5 of each trade, no fee). Bands = price the maker paid for the token it ended up holding")
bands = ((.005, .05), (.05, .2), (.2, .5), (.5, .8), (.8, .95), (.95, .995))
print(f"{'family':26s} {'trades':>8s} {'$M':>6s} {'events':>6s} | " + " | ".join(f"{100*lo:g}-{100*hi:g}c".center(22) for lo, hi in bands) + " | all")
for f in sorted(fams, key=lambda f: -(fam == f).sum()):
    m0 = (fam == f) & (htc > 0.25)
    if m0.sum() < 2000: continue
    s = f"{fams[f]:26s} {m0.sum():8d} {usd[m0].sum()/1e6:6.1f} {len(np.unique(ev[m0])):6d} | "
    for lo, hi in bands:
        m = m0 & (q >= lo) & (q < hi)
        if m.sum() < 300: s += " " * 22 + " | "; continue
        (l, h), n = boot(ret[m], w5[m], ev[m])
        s += f"{100*np.average(ret[m], weights=w5[m]):+6.1f} [{100*l:+5.1f},{100*h:+5.1f}]".ljust(22) + " | "
    (l, h), n = boot(ret[m0], w5[m0], ev[m0])
    print(s + f"{100*np.average(ret[m0], weights=w5[m0]):+5.1f} [{100*l:+5.1f},{100*h:+5.1f}]")
di, k, L, fin, bf, bev, bh, rk, sp, T = B.T
r = (fin - L) / L
print("\nB. Resting $5 bids at (last two-sided mid - d), filled only by through-prints within the hour, held to resolution, no fee")
print(f"{'family':26s} {'bid band':9s} " + " ".join(f"{'d='+str(d)+'c':>30s}" for d in (0, 1, 3, 5)))
for f in sorted(fams, key=lambda f: -(bf == f).sum()):
    for lo, hi in ((.8, .995), (.2, .8), (.01, .2)):
        s = f"{fams[f]:26s} {lo:.2f}-{hi:.2f} "; any_ = False
        for d in (0, 1, 3, 4):
            m = (bf == f) & (di == d) & (L >= lo) & (L < hi)
            if m.sum() < 150: s += " " * 31; continue
            (l, h), n = boot(r[m], np.ones(m.sum()), bev[m]); any_ = True
            s += f" {100*r[m].mean():+6.1f} [{100*l:+5.1f},{100*h:+5.1f}] n={m.sum():<6d}".rjust(31)
        if any_: print(s)
