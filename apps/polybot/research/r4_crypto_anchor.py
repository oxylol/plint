"""Empirical anchor for 'coin above K at T': the distribution of the standardised move
x = ln(S_T/S_t) / (sigma_t * sqrt(tau)), sigma_t = trailing realised vol (15-min returns, half 24 h + half 7 d),
by horizon bucket, from Binance history. P(S_T > K) = 1 - F_tau(ln(K/S_t)/(sigma_t sqrt(tau))).
    python anchor_fit.py KLINES_DIR OUT.json [fit_until=2026-08-31]
"""
import json, sys, time, calendar
import numpy as np
from math import erf, sqrt

D, OUT = sys.argv[1], sys.argv[2]
UNTIL = calendar.timegm(time.strptime(sys.argv[3] if len(sys.argv) > 3 else "2026-08-31", "%Y-%m-%d"))
HOURS = [1, 2, 4, 8, 16, 24, 48, 96, 168]
YEAR = 365 * 24 * 4          # 15-min bars a year
xs = {h: [] for h in HOURS}; yrs = {h: [] for h in HOURS}
for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
    a = np.load(f"{D}/{sym}_15m.npy")
    t, c = a[:, 0], a[:, 1]
    keep = t <= UNTIL
    t, c = t[keep], c[keep]
    lr = np.diff(np.log(c)); r2 = np.concatenate([[0], np.cumsum(lr * lr)])
    i = np.arange(672, len(c))
    var = 0.5 * (r2[i] - r2[i - 96]) / 96 + 0.5 * (r2[i] - r2[i - 672]) / 672
    sig = np.sqrt(var * YEAR)
    for h in HOURS:
        n = h * 4
        j = i[i + n < len(c)]
        x = np.log(c[j + n] / c[j]) / (sig[: len(j)] * np.sqrt(h / (365 * 24)))
        ok = np.isfinite(x) & (t[j + n] - t[j] == n * 900)
        xs[h].append(x[ok]); yrs[h].append(time.gmtime(0).tm_year + (t[j][ok] // (365.25 * 86400)).astype(int))
N = lambda z: 0.5 * (1 - erf(z / sqrt(2)))
qs = np.concatenate([np.linspace(0.0005, 0.02, 40), np.linspace(0.025, 0.975, 191), np.linspace(0.98, 0.9995, 40)])
table = {}
print("P(move beyond z sigmas), up tail / down tail, vs normal")
print(f"{'horizon':>8s} " + " ".join(f"z={z:<14.1f}" for z in (1, 1.5, 2, 2.5, 3, 4)))
print(f"{'normal':>8s} " + " ".join(f"{100*N(z):5.2f}%{'':10s}" for z in (1, 1.5, 2, 2.5, 3, 4)))
for h in HOURS:
    x = np.concatenate(xs[h]); y = np.concatenate(yrs[h])
    table[str(h)] = {"q": qs.tolist(), "x": np.quantile(x, qs).tolist()}
    print(f"{h:7d}h " + " ".join(f"{100*(x > z).mean():5.2f}/{100*(x < -z).mean():5.2f}%{'':4s}" for z in (1, 1.5, 2, 2.5, 3, 4)))
    if h in (24, 96):
        for yy in sorted(set(y)):
            m = y == yy
            print(f"   {yy}: " + " ".join(f"{100*(x[m] > z).mean():5.2f}/{100*(x[m] < -z).mean():5.2f}%{'':4s}" for z in (1, 1.5, 2, 2.5, 3, 4)))
json.dump({"hours": HOURS, "table": table, "sigma": "15-min returns, 0.5*24h + 0.5*7d, annualised", "fit_until": UNTIL}, open(OUT, "w"))
