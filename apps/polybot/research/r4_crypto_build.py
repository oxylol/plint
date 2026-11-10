"""Daily 'coin above K' ladders: every taker trade with two anchors known at the trade (spot LAG minutes stale):
lognormal and the empirical standardised-move table (anchor.json, fitted on history before the test window)."""
import gzip, json, re, sys, glob, os, datetime as dt
import numpy as np
from math import erf, sqrt

R = sys.argv[1]
SYM = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT", "xrp": "XRPUSDT"}
AN = json.load(open(f"{R}/anchor.json"))
HRS = np.array(AN["hours"], dtype=float)
TAB = [(np.array(AN["table"][str(h)]["x"]), np.array(AN["table"][str(h)]["q"])) for h in AN["hours"]]
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
K = {}
for a, s in SYM.items():
    k = np.load(f"{R}/klines/{s}.npy")
    t = k[:, 0].astype(np.int64); cl = k[:, 4]
    t15 = t[::15]; c15 = cl[::15]
    r2 = np.concatenate([[0], np.cumsum(np.diff(np.log(c15)) ** 2)])
    K[a] = (t, cl, t15, r2)


def vol_at(a, ts):
    _, _, t15, r2 = K[a]
    i = np.clip(np.searchsorted(t15, ts, side="right") - 1, 1, len(r2) - 1)
    rv = lambda nb: (r2[i] - r2[np.clip(i - nb, 0, None)]) / np.maximum(i - np.clip(i - nb, 0, None), 1)
    return np.sqrt((0.5 * rv(96) + 0.5 * rv(672)) * 4 * 24 * 365)


def spot_at(a, ts, lag):
    t, cl, _, _ = K[a]
    return cl[np.clip(np.searchsorted(t, ts - 60 - 60 * lag, side="right") - 1, 0, len(cl) - 1)]


def emp_above(z, tau_h):
    """P(x > z) from the table, linear in log-horizon between buckets"""
    lh = np.log(np.clip(tau_h, HRS[0], HRS[-1]))
    j = np.clip(np.searchsorted(np.log(HRS), lh, side="right") - 1, 0, len(HRS) - 2)
    w = (lh - np.log(HRS[j])) / (np.log(HRS[j + 1]) - np.log(HRS[j]))
    out = np.zeros(len(z))
    for b in range(len(HRS)):
        x, q = TAB[b]
        F = np.interp(z, x, q, left=q[0] / 2, right=1 - (1 - q[-1]) / 2)
        out += np.where(j == b, (1 - w) * (1 - F), 0) + np.where(j + 1 == b, w * (1 - F), 0)
    return out


rows = []; mi = 0; mism = [0, 0]
for f in sorted(glob.glob(f"{R}/universe/*.jsonl.gz")):
    for l in gzip.open(f, "rt"):
        if '-above-' not in l:
            continue
        m = json.loads(l)
        g = re.match(r"(bitcoin|ethereum|solana|xrp)-above-", m["slug"] or "")
        if not g or re.search(r"(am|pm)-et$", m["slug"]) or m["p"][0] not in (0.0, 1.0):
            continue
        q = re.search(r"above \$?([\d,]+(?:\.\d+)?)", m["q"])
        p = f"{R}/tapes/{m['cid']}.json.gz"
        if not q or not os.path.exists(p):
            continue
        tape = json.load(gzip.open(p, "rt"))
        if not tape:
            continue
        asset = g.group(1); Kk = float(q.group(1).replace(",", "")); T = dt.datetime.fromisoformat(m["end"]).timestamp()
        a = np.array(tape, dtype=np.float64)
        ts, tok, side, size, pr = a.T
        ok = (pr >= 0.001) & (pr <= 0.999) & (ts < T - 300)
        if not ok.any():
            continue
        ts, tok, side, size, pr = ts[ok], tok[ok], side[ok], size[ok], pr[ok]
        t, cl, _, _ = K[asset]
        i = np.searchsorted(t, T, side="left")
        if i >= len(t) or t[i] != T or ts.min() < t[0] + 8 * 86400:
            continue
        mism[0] += 1; mism[1] += ((1.0 if cl[i] > Kk else 0.0) != m["p"][0])
        yp = np.where(tok == 0, pr, 1 - pr); tdir = np.where(tok == 0, side, -side)
        tau_h = (T - ts) / 3600.0; tau = tau_h / (365 * 24)
        sig = vol_at(asset, ts - 60)
        cols = [ts, yp, tdir, size * pr]
        for lag in (1, 3):
            S = spot_at(asset, ts, lag)
            z = np.log(Kk / S) / (sig * np.sqrt(tau))
            cols += [Phi(-z - sig * np.sqrt(tau) / 2), emp_above(z, tau_h)]
        n = len(ts)
        cols += [np.full(n, m["p"][0]), tau_h, np.full(n, list(SYM).index(asset)), np.full(n, mi), np.full(n, int(T // 86400)), size]
        rows.append(np.column_stack(cols)); mi += 1
A = np.vstack(rows)
np.save(f"{R}/crypto_daily.npy", A)
print("markets", mi, "trades", len(A), "mismatch", mism, "expiry days", len(np.unique(A[:, 12])))
