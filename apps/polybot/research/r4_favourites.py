import numpy as np, json, sys, time
R = '/opt/plint/data/polybot/research4'; TAG = sys.argv[1]; FAMS = sys.argv[2].split(',') if len(sys.argv) > 2 else None
A = np.load(f'{R}/{TAG}_A.npy').astype(np.float64); meta = json.load(open(f'{R}/{TAG}_meta.json'))
B = np.load(f'{R}/{TAG}_B.npy')
fid = [i for n, i in meta['fams'].items() if (FAMS is None or n in FAMS)]
sname = {v: k for k, v in meta['sers'].items()}
q, win, usd, mtok, hte, fam, ev, htc, ts, ser = A.T
base = np.isin(fam, fid) & (htc > 0.25)
rng = np.random.default_rng(11)
def boot(r, c, w):
    u, idx = np.unique(c, return_inverse=True)
    if len(u) < 6: return (np.nan, np.nan), len(u)
    num = np.bincount(idx, r * w); den = np.bincount(idx, w)
    bs = [num[s].sum() / max(den[s].sum(), 1e-9) for s in (rng.integers(0, len(u), len(u)) for _ in range(500))]
    return np.percentile(bs, [5, 95]), len(u)
f_price = np.where(q >= 0.5, q, 1 - q); f_win = np.where(q >= 0.5, win, 1 - win); how = np.where(q >= 0.5, 0, 1)
ret = (f_win - f_price) / f_price
w5 = np.minimum(usd * np.where(how == 0, 1, f_price / np.maximum(q, 1e-9)), 5)
qtr = np.array([time.strftime('%Y', time.gmtime(t)) + 'Q' + str((time.gmtime(t).tm_mon - 1) // 3 + 1) for t in ts[::1]]) if False else None
tm = ts.astype('datetime64[s]'); yq = (tm.astype('datetime64[M]').astype(int) // 3)  # quarter index since 1970
print(f"{TAG} {FAMS}: trades {base.sum()}, events {len(np.unique(ev[base]))}, series {len(np.unique(ser[base]))}, {str(tm[base].min())[:10]} -> {str(tm[base].max())[:10]}")
print(f"{'band':11s} {'how':6s} {'trades':>7s} {'series':>6s} {'paid':>6s} {'won':>6s} {'ret/trade':>9s} {'90% CI (series)':>17s} {'mean of series':>14s} {'left(h)':>7s} {'fav $/day':>9s}")
nd = (ts[base].max() - ts[base].min()) / 86400
for lo, hi in ((.70, .80), (.80, .90), (.90, .95), (.95, .98), (.98, .995)):
    for h, hn in ((0, "maker"), (1, "taker")):
        m = base & (f_price >= lo) & (f_price < hi) & (how == h)
        if m.sum() < 200: continue
        (l, u), ns = boot(ret[m], ser[m], w5[m])
        us, idx = np.unique(ser[m], return_inverse=True)
        sm = np.bincount(idx, ret[m] * w5[m]) / np.bincount(idx, w5[m])
        fu = (usd[m] * np.where(h == 0, 1, f_price[m] / np.maximum(q[m], 1e-9))).sum()
        print(f"{lo:.2f}-{hi:.3f} {hn:6s} {m.sum():7d} {ns:6d} {100*np.average(f_price[m], weights=w5[m]):5.1f}c {100*np.average(f_win[m], weights=w5[m]):5.1f}% {100*np.average(ret[m], weights=w5[m]):+8.2f}% [{100*l:+6.2f},{100*u:+6.2f}]   {100*sm.mean():+8.2f}%   {np.median(htc[m]):7.0f} {fu/nd:9.0f}")
m = base & (f_price >= .90) & (f_price < .995)
print("favourite 90-99.5c by quarter (both ways | maker only | mean of series, maker):")
for qq in np.unique(yq[m]):
    mm = m & (yq == qq); mk_ = mm & (how == 0)
    us, idx = np.unique(ser[mk_], return_inverse=True)
    sm = (np.bincount(idx, ret[mk_] * w5[mk_]) / np.bincount(idx, w5[mk_])).mean() if mk_.sum() else float('nan')
    print(f"  {1970 + qq // 4}Q{qq % 4 + 1}: n={mm.sum():7d} series={len(np.unique(ser[mm])):4d}  {100*np.average(ret[mm], weights=w5[mm]):+6.2f}% | {100*np.average(ret[mk_], weights=w5[mk_]) if mk_.sum() else float('nan'):+6.2f}% | {100*sm:+6.2f}%   won {100*np.average(f_win[mm], weights=w5[mm]):5.1f}% paid {100*np.average(f_price[mm], weights=w5[mm]):5.1f}c")
us, idx = np.unique(ser[m], return_inverse=True)
cnt = np.bincount(idx, w5[m]); pnl = np.bincount(idx, ret[m] * w5[m])
o5 = np.argsort(-cnt)[:5]
print("top-5 series weight share", round(100 * cnt[o5].sum() / cnt.sum(), 1), "% | return without them", f"{100*pnl[np.setdiff1d(np.arange(len(cnt)), o5)].sum()/cnt[np.setdiff1d(np.arange(len(cnt)), o5)].sum():+.2f}%")
print("worst series:", "; ".join(f"{sname[int(us[o])][:38]} {100*pnl[o]/cnt[o]:+.0f}% (w {100*cnt[o]/cnt.sum():.1f}%)" for o in np.argsort(pnl)[:5]))
di, k, L, fin, bf, bev, bh, rk, sp, T = B.T
r = (fin - L) / L; bm = np.isin(bf, fid)
print("resting bid at the favourite's side of the mid, through-print fills only:")
for lo, hi in ((.80, .90), (.90, .96), (.96, .995)):
    s = f"  bid {lo:.2f}-{hi:.3f}: "
    for d, dn in ((0, "at mid"), (1, "-1c"), (3, "-3c")):
        mm = bm & (di == d) & (L >= lo) & (L < hi)
        if mm.sum() < 100: continue
        (l, u), n = boot(r[mm], bev[mm], np.ones(mm.sum()))
        s += f"{dn}: {100*r[mm].mean():+5.2f}% [{100*l:+5.2f},{100*u:+5.2f}] n={mm.sum()} ev={n} | "
    print(s)
