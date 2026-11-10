"""Carry rule with a stop: after entry, sell the first time the favourite's bid is at or below STOP.
Prints leg: hourly rows of the same market (bid = last print on the bid side); a market that leaves the 70c+ table
without such a row is a gap and is held to the end. Busy leg: 12-hour mids, exit at mid - 1c."""
import gzip, json, re, sys, datetime as dt, numpy as np
R = '/opt/plint/data/polybot/research4'
LO, HI, S, D0, D1 = 0.90, 0.985, 72, 10, 45
MON = re.compile(r"january|february|march|april|may|june|july|august|september|october|november|december|\d+")
meta = json.load(open(f'{R}/geo_meta.json')); C = np.load(f'{R}/geo_C.npy')
C = C[np.lexsort((C[:, 1], C[:, 0]))]
mi, T, kf, mid, bid, ask, fwin, hte, ser, ntape, htc, age, famc, n72, usd72 = C.T
sname = {v: k for k, v in meta['sers'].items()}
new = np.r_[True, (mi[1:] != mi[:-1]) | (kf[1:] != kf[:-1])]
below = mid < LO
idx = np.maximum.accumulate(np.where(new | below, np.arange(len(T)), 0))
stable = np.where(below, 0, (T - T[idx]) / 3600)
c = (ask >= LO) & (ask < HI) & (hte >= D0 * 24) & (hte < D1 * 24) & (age <= 2) & (htc > 1) & (stable >= S) & (ask > bid - 0.02) & (ntape < 3000)
ix = np.nonzero(c)[0]; _, first = np.unique(mi[ix], return_index=True); ix = ix[first]
mstart = np.searchsorted(mi, mi[ix], side='left'); mend = np.searchsorted(mi, mi[ix], side='right')
info = {}
for tag in ('geopolitics', 'world', 'middle-east'):
    for l in gzip.open(f'{R}/events_long/{tag}.jsonl.gz', 'rt'):
        r = json.loads(l); info.setdefault(r['cid'], r)
mids = []; f = gzip.open(f'{R}/mids12h.jsonl.gz', 'rt'); done = set()
while True:
    try: l = f.readline()
    except EOFError: break
    if not l: break
    try: j = json.loads(l)
    except Exception: continue
    m = info.get(j['cid'])
    if not m or not j['p'] or not m['end'] or j['cid'] in done: continue
    done.add(j['cid']); mids.append((m, np.array(j['p'])))
rng = np.random.default_rng(3)
def boot(r_, s_):
    u, ii = np.unique(s_, return_inverse=True); num = np.bincount(ii, r_); den = np.bincount(ii).astype(float)
    bs = [num[k].sum() / max(den[k].sum(), 1) for k in (rng.integers(0, len(u), len(u)) for _ in range(1000))]
    return np.percentile(bs, [5, 95])
print(f"{'stop':>6s} {'n':>5s} {'ret/pos':>8s} {'90% CI':>15s} {'prints leg':>10s} {'busy leg':>9s} {'stopped':>7s} {'of them would have won':>22s} {'full losses':>11s} {'$5 book, %/y on capital':>24s}")
for STOP in (None, 0.80, 0.85, 0.88):
    rr, ss, legs, t0s, t1s, stopped, stopped_won, full = [], [], [], [], [], 0, 0, 0
    for n_, i in enumerate(ix):
        a = ask[i]; out = fwin[i]; ex = None; texit = T[i] + htc[i] * 3600
        if STOP is not None:
            j0, j1 = i + 1, mend[n_]
            for j in range(j0, j1):
                if T[j] <= T[i]: continue
                ob = bid[j] if kf[j] == kf[i] else 1 - ask[j]
                if ob <= STOP:
                    ex = max(ob, 0.001); texit = T[j]; break
        if ex is not None:
            stopped += 1; stopped_won += out; r = (ex - a) / a
        else:
            r = (out - a) / a; full += (out == 0)
        rr.append(r); ss.append(sname[int(ser[i])]); legs.append(0); t0s.append(T[i]); t1s.append(texit)
    for m, P in mids:
        Eend = dt.datetime.fromisoformat(m['end']).timestamp(); Tcl = dt.datetime.fromisoformat(m['closed_time']).timestamp() if m['closed_time'] else Eend
        t = P[:, 0]; y = P[:, 1]; fv = np.maximum(y, 1 - y); side = (y < 0.5).astype(int)
        for i in range(len(t)):
            left = (Eend - t[i]) / 86400
            if not (D0 <= left < D1) or t[i] >= Tcl - 3600: continue
            w = (t >= t[i] - S * 3600 - 1) & (t <= t[i])
            if w.sum() < 6 or (side[w] != side[i]).any() or fv[w].min() < LO: continue
            a = fv[i] + 0.005
            if not (LO <= a < HI): continue
            out = m['p'][side[i]]; ex = None; texit = Tcl
            if STOP is not None:
                ours = np.where(side[i] == 0, y, 1 - y)
                for k in range(i + 1, len(t)):
                    if t[k] >= Tcl: break
                    if ours[k] - 0.005 <= STOP:
                        ex = max(ours[k] - 0.01, 0.001); texit = t[k]; break
            if ex is not None:
                stopped += 1; stopped_won += out; r = (ex - a) / a
            else:
                r = (out - a) / a; full += (out == 0)
            rr.append(r); ss.append(MON.sub('#', m['ev'] or m['slug'])); legs.append(1); t0s.append(t[i]); t1s.append(texit); break
    rr = np.array(rr); ss = np.array(ss); legs = np.array(legs); t0s = np.array(t0s); t1s = np.array(t1s)
    l, u = boot(rr, ss)
    o = np.argsort(t0s); open_ = []; dn = {}; tk = []
    for x, y_, z, s in zip(t0s[o], t1s[o], rr[o], ss[o]):
        open_ = [(e, s2) for e, s2 in open_ if e > x]; d = int(x // 86400)
        if sum(1 for _, s2 in open_ if s2 == s) >= 3 or dn.get(d, 0) >= 10: continue
        open_.append((y_, s)); dn[d] = dn.get(d, 0) + 1; tk.append((x, y_, z))
    tk = np.array(tk); pnl = 5 * tk[:, 2]; grid = np.arange(tk[:, 0].min(), tk[:, 1].max(), 86400)
    capu = np.array([5 * ((tk[:, 0] <= g) & (tk[:, 1] > g)).sum() for g in grid]); yrs = (tk[:, 1].max() - tk[:, 0].min()) / (365 * 86400)
    eq = np.cumsum(pnl[np.argsort(tk[:, 1])]); dd = (np.maximum.accumulate(eq) - eq).max()
    print(f"{str(STOP):>6s} {len(rr):5d} {100*rr.mean():+7.2f}% [{100*l:+6.2f},{100*u:+6.2f}] {100*rr[legs == 0].mean():+9.2f}% {100*rr[legs == 1].mean():+8.2f}% {stopped:7d} {int(stopped_won):22d} {int(full):11d}   {pnl.sum():+5.0f}$ = {100*pnl.sum()/capu.mean()/yrs:+4.0f}%/y, max drawdown {dd:.0f}$")
