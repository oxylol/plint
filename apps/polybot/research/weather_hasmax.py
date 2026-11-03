"""Has the day's high happened yet? A model of that, hour by hour, against the market's own price.

    python weather_hasmax.py DATA_DIR [--report OUT.md]

For every finished temperature day (cities whose resolution follows the station) and every local hour from 09:00
to 23:00: the state a trader could see (the running high, the latest reading and its trend, hours since the high,
what the day-before forecast curve still had in store) and whether the running high's bucket ended up winning.
Three walk-forward models (a logistic model on the state; the same anchored on the market's price; boosted trees)
are compared with the market's last trade for calibration, then traded: buy Yes on the running bucket when the
model's probability beats the tape's next ask by a threshold after the fee, or No when it is the other way round.
"""
from __future__ import annotations

import bisect
import datetime as dt
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

DATA = Path(sys.argv[1]) / "weather"
REPORT = Path(sys.argv[sys.argv.index("--report") + 1]) if "--report" in sys.argv else None
U = dt.timezone.utc
BAD = {"Hong Kong", "Shenzhen", "Denver", "Moscow", "Seoul"}
MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
HOURS = list(range(9, 24))
rng = np.random.default_rng(7)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


rnd = lambda x: math.floor(x + 0.5)  # noqa: E731

mk = [json.loads(l) for l in gzip.open(DATA / "markets.jsonl.gz", "rt")]
st = json.loads((DATA / "stations.json").read_text())
city_st = {}
for icao, s in st.items():
    city_st.setdefault(s["city"], icao)
obs = defaultdict(list)
for icao, s in st.items():
    p = DATA / "obs" / f"{icao}.csv"
    if not p.exists():
        continue
    tz = ZoneInfo(s["tz"])
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", ""):
            continue
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U).astimezone(tz)
            f = float(parts[2])
        except ValueError:
            continue
        obs[(icao, t.date().isoformat())].append((t, (f - 32) * 5 / 9))
for k in obs:
    obs[k].sort()
fc: dict[tuple, dict] = {}
for icao, s in st.items():
    p = DATA / "forecasts" / f"{icao}.json"
    if not p.exists():
        continue
    h = (json.loads(p.read_text()).get("temperature_2m_previous_day1") or {})
    times = h.get("time") or []
    per_day: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [None] * 24))
    for m in MODELS:
        vals = h.get(f"temperature_2m_previous_day1_{m}") or []
        for t, v in zip(times, vals):
            if v is not None:
                per_day[t[:10]][m][int(t[11:13])] = v
    for day, bym in per_day.items():
        curves = [np.array(v, dtype=float) for v in bym.values() if all(x is not None for x in v)]
        if len(curves) >= 3:
            arr = np.vstack(curves)
            fc[(icao, day)] = {"mean": arr.mean(0), "hi": arr.max(0), "max_models": arr.max(1)}

events = defaultdict(list)
for m in mk:
    if m["closed"] and m["kind"] == "highest" and m["city"] not in BAD:
        pr = m.get("prices") or [None, None]
        if pr[0] is not None and (pr[0] >= 0.99 or pr[0] <= 0.01):
            m["final"] = 1.0 if pr[0] >= 0.99 else 0.0
            events[(m["city"], m["day"])].append(m)


def bucket_of(ms, v):
    for m in ms:
        if (m["lo"] is None or v >= m["lo"]) and (m["hi"] is None or v <= m["hi"]):
            return m
    return None


_tapes: dict[str, tuple] = {}


def tape(cid):
    if cid in _tapes:
        return _tapes[cid]
    p = DATA / "tapes" / f"{cid}.json.gz"
    out = None
    if p.exists():
        try:
            rows = sorted(json.load(gzip.open(p, "rt")), key=lambda r: r[0])
            ts = [r[0] for r in rows]
            px = [(r[4] if r[1] == 0 else 1 - r[4]) for r in rows]
            by = [((r[1] == 0 and r[2] == 1) or (r[1] == 1 and r[2] == -1)) for r in rows]
            out = (ts, px, by)
        except Exception:
            out = None
    _tapes[cid] = out
    return out


def quotes(cid, t, window=3600):
    """Last trade before t (Yes terms, within 24 h) and the next prices takers paid for Yes and for No after t."""
    tp = tape(cid)
    if tp is None:
        return None, None, None
    ts, px, by = tp
    i = bisect.bisect_right(ts, t)
    last = px[i - 1] if i > 0 and ts[i - 1] >= t - 86400 else None
    quotes.age = (t - ts[i - 1]) / 3600 if last is not None else None
    ask_y = ask_n = None
    for j in range(i, len(ts)):
        if ts[j] > t + window:
            break
        if by[j] and ask_y is None:
            ask_y = px[j]
        if not by[j] and ask_n is None:
            ask_n = 1 - px[j]
        if ask_y is not None and ask_n is not None:
            break
    return last, ask_y, ask_n


# ── rows ─────────────────────────────────────────────────────
rows = []
max_hours = []
n_events = 0
for (city, day), ms in sorted(events.items(), key=lambda kv: kv[0][1]):
    icao = city_st.get(city)
    o = obs.get((icao, day)) if icao else None
    f = fc.get((icao, day)) if icao else None
    if not o or len(o) < 12 or f is None:
        continue
    unit = ms[0]["unit"]
    conv = (lambda c: c * 9 / 5 + 32) if unit == "F" else (lambda c: c)
    final_max = max(c for _, c in o)
    final_bucket = bucket_of(ms, rnd(conv(final_max)))
    resolved = [m for m in ms if m["final"] == 1.0]
    if final_bucket is None or len(resolved) != 1 or resolved[0]["cid"] != final_bucket["cid"]:
        continue
    n_events += 1
    tz = ZoneInfo(st[icao]["tz"])
    first_at_max = next(t for t, c in o if c == final_max)
    max_hours.append(first_at_max.hour + first_at_max.minute / 60)
    fmax_mean, fmax_hi = float(f["mean"].max()), float(f["hi"].max())
    fmax_hour = int(np.argmax(f["mean"]))
    spread = float(f["max_models"].max() - f["max_models"].min())
    for H in HOURS:
        t_dec = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(H, 0), tzinfo=tz)
        seen = [(t, c) for t, c in o if t <= t_dec]
        if not seen:
            continue
        M = max(c for _, c in seen)
        T = seen[-1][1]
        t_max = next(t for t, c in seen if c == M)
        before = [c for t, c in seen if t <= t_dec - dt.timedelta(hours=1)]
        trend = T - before[-1] if before else 0.0
        wb = bucket_of(ms, rnd(conv(M)))
        if wb is None:
            continue
        rem = float(f["mean"][H:].max()) if H < 24 else float(f["mean"][23])
        rem_hi = float(f["hi"][H:].max())
        last, ask_y, ask_n = quotes(wb["cid"], t_dec.timestamp())
        last_age = quotes.age if last is not None else None
        rows.append({"day": day, "city": city, "unit": unit, "H": H, "y": 1.0 if wb["cid"] == final_bucket["cid"] else 0.0,
                     "drop": M - T, "trend": trend, "hsm": (t_dec - t_max).total_seconds() / 3600, "fmax_gap": fmax_mean - M, "fmax_hi_gap": fmax_hi - M,
                     "rem_gap": rem - M, "rem_hi_gap": rem_hi - M, "fmax_hour": fmax_hour, "spread": spread, "unit_f": 1.0 if unit == "F" else 0.0,
                     "last": last, "last_age": last_age, "ask_y": ask_y, "ask_n": ask_n, "fee": wb.get("fee_rate", 0.05), "ev": (city, day),
                     "stale": (t_dec - seen[-1][0]).total_seconds() / 3600})
say(f"{n_events} finished days (cities whose resolution follows the station, with a forecast curve), {len(rows)} hourly states")

mh = np.array(max_hours)
say("\n## When the day's high is first reached (share of days by local hour)")
say("  " + " · ".join(f"by {h}:00 {100 * (mh < h).mean():.0f}%" for h in (11, 12, 13, 14, 15, 16, 17, 18, 20)) + f" · median {np.median(mh):.1f}")

Y = np.array([r["y"] for r in rows])
Hh = np.array([r["H"] for r in rows])
drop = np.array([r["drop"] for r in rows])
say("\n## Share of states whose running high held, by hour and by how far the reading had dropped (°C)")
say("  hour   all    drop<0.5  0.5–1.5  1.5–3   ≥3     market's last price (mean) where traded")
L = np.array([r["last"] if r["last"] is not None else np.nan for r in rows])
for H in (10, 12, 13, 14, 15, 16, 17, 18, 20, 22):
    m = Hh == H
    bins = [(m & (drop < 0.5)), (m & (drop >= 0.5) & (drop < 1.5)), (m & (drop >= 1.5) & (drop < 3)), (m & (drop >= 3))]
    say(f"  {H:02d}:00  {100 * Y[m].mean():4.0f}%   " + "  ".join(f"{100 * Y[b].mean():5.0f}% " if b.sum() >= 30 else "   –   " for b in bins)
        + f"   {100 * np.nanmean(L[m]):.0f}¢ (n {int((~np.isnan(L[m])).sum())})")

# ── walk-forward models ──────────────────────────────────────
FEATS = ["H", "drop", "trend", "hsm", "fmax_gap", "fmax_hi_gap", "rem_gap", "rem_hi_gap", "fmax_hour", "spread", "unit_f", "stale"]
X = np.array([[float(r[k]) for k in FEATS] for r in rows])
X = np.column_stack([X, X[:, 0] ** 2, X[:, 1] * X[:, 0], np.minimum(X[:, 6], 0), np.maximum(X[:, 6], 0)])   # hour², hour×drop, remaining gap split
days = np.array([r["day"] for r in rows])
udays = sorted(set(days))
week_of = {d: dt.date.fromisoformat(d).isocalendar()[1] + 100 * dt.date.fromisoformat(d).isocalendar()[0] for d in udays}
W = np.array([week_of[d] for d in days])
has_px = ~np.isnan(L)


def lg(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    import lightgbm as lgb
except Exception:
    lgb = None

p_state = np.full(len(rows), np.nan)
p_anch = np.full(len(rows), np.nan)
p_tree = np.full(len(rows), np.nan)
for w in sorted(set(W)):
    te = W == w
    first_day = min(days[te])
    tr = days < (dt.date.fromisoformat(first_day) - dt.timedelta(days=1)).isoformat()
    if tr.sum() < 3000:
        continue
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[tr]), Y[tr])
    p_state[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    trp = tr & has_px
    if trp.sum() >= 2000:
        Xa = np.column_stack([X, np.where(has_px, lg(np.nan_to_num(L, nan=0.5)), 0.0)])
        sca = StandardScaler().fit(Xa[trp])
        lra = LogisticRegression(C=1.0, max_iter=500).fit(sca.transform(Xa[trp]), Y[trp])
        tep = te & has_px
        p_anch[tep] = lra.predict_proba(sca.transform(Xa[tep]))[:, 1]
    if lgb is not None:
        bst = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=100, verbose=-1).fit(X[tr], Y[tr])
        p_tree[te] = bst.predict_proba(X[te])[:, 1]

ok = ~np.isnan(p_state) & has_px


def calib(p, y, name):
    bins = [0, .05, .1, .2, .35, .5, .65, .8, .9, .95, .98, 1.0001]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 50:
            out.append(f"{100 * lo:.0f}–{100 * min(hi, 1):.0f}%: {100 * p[m].mean():.0f}→{100 * y[m].mean():.0f} (n {m.sum()})")
    ll = -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
    say(f"  {name:28s} log loss {ll:.4f} · predicted→actual " + " · ".join(out))
    return ll


say(f"\n## Calibration on the {int(ok.sum())} states with a model and a market price (walk-forward, from {min(days[ok])})")
calib(L[ok], Y[ok], "market's last trade")
calib(p_state[ok], Y[ok], "state model (no price)")
oka = ok & ~np.isnan(p_anch)
calib(p_anch[oka], Y[oka], "state model + market price")
if lgb is not None:
    calib(p_tree[ok], Y[ok], "boosted trees (no price)")
say("\n  the same, 13:00–17:00 only (where the question is live):")
mid = ok & (Hh >= 13) & (Hh <= 17)
calib(L[mid], Y[mid], "market's last trade")
AGE = np.array([r["last_age"] if r["last_age"] is not None else np.nan for r in rows])
fresh = mid & (AGE <= 0.5)
calib(L[fresh], Y[fresh], f"last trade ≤30 min old (n {int(fresh.sum())})")
AYc = np.array([r["ask_y"] if r["ask_y"] is not None else np.nan for r in rows])
asked = mid & ~np.isnan(AYc)
calib(AYc[asked], Y[asked], f"next price takers paid for Yes (n {int(asked.sum())})")
calib(p_state[fresh], Y[fresh], "state model, same fresh states")
oka2 = fresh & ~np.isnan(p_anch)
calib(p_anch[oka2], Y[oka2], "state + price, same fresh states")
calib(p_state[mid], Y[mid], "state model (no price)")
if lgb is not None:
    calib(p_tree[mid], Y[mid], "boosted trees (no price)")

# ── trading ──────────────────────────────────────────────────
AY = np.array([r["ask_y"] if r["ask_y"] is not None else np.nan for r in rows])
AN = np.array([r["ask_n"] if r["ask_n"] is not None else np.nan for r in rows])
FEE = np.array([r["fee"] for r in rows])
EV = [r["ev"] for r in rows]


AGE_ALL = None


def simulate(P, name, thr, known=False, fresh_only=False):
    """One trade per day and city: the first hour the model's edge over the ask clears thr after the fee."""
    taken = {}
    for i in np.argsort(Hh, kind="stable"):
        if np.isnan(P[i]) or EV[i] in taken:
            continue
        if known:
            if np.isnan(L[i]) or (fresh_only and not (AGE[i] <= 0.5)):
                continue
            ay, an = min(0.999, L[i] + 0.02), min(0.999, 1 - L[i] + 0.02)
        else:
            ay, an = AY[i], AN[i]
        ey = P[i] - ay - FEE[i] * ay * (1 - ay) if not np.isnan(ay) else -9
        en = (1 - P[i]) - an - FEE[i] * an * (1 - an) if not np.isnan(an) else -9
        if max(ey, en) < thr:
            continue
        yes = ey >= en
        price = ay if yes else an
        if not (0.03 <= price <= 0.97):
            continue
        won = Y[i] if yes else 1 - Y[i]
        taken[EV[i]] = (days[i], won / price - 1 - FEE[i] * (1 - price), price, yes, Hh[i])
    if len(taken) < 20:
        say(f"  {name:46s} thr {100 * thr:2.0f}¢: {len(taken)} trades")
        return
    r = np.array([v[1] for v in taken.values()])
    dd = np.array([v[0] for v in taken.values()])
    ud = sorted(set(dd))
    sums = np.array([r[dd == x].sum() for x in ud])
    cnts = np.array([(dd == x).sum() for x in ud])
    draws = rng.integers(0, len(ud), size=(2000, len(ud)))
    bm = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.quantile(bm, [0.05, 0.95])
    yes_share = np.mean([v[3] for v in taken.values()])
    say(f"  {name:46s} thr {100 * thr:2.0f}¢: {len(r):5d} trades on {len(ud):3d} days · {100 * r.mean():+6.1f}% per trade [90% CI {100 * lo:+.1f}, {100 * hi:+.1f}] · "
        f"win {100 * (r > 0).mean():.0f}% · paid {100 * np.mean([v[2] for v in taken.values()]):.0f}¢ · {100 * yes_share:.0f}% Yes · at {np.mean([v[4] for v in taken.values()]):.1f}h avg · ${5 * r.sum() / len(ud) * 7:+.1f}/week at $5")


say("\n## Trading, $5 a trade, one per city and day, held to resolution, taker fee paid")
say("  priced at the tape's next taker trade on that side within the hour (optimistic: someone else did buy then)")
for name, P in (("state model", p_state), ("state model + market price", p_anch), ("boosted trees", p_tree)):
    if np.all(np.isnan(P)):
        continue
    for thr in (0.03, 0.05, 0.08, 0.12):
        simulate(P, name, thr)
say("  priced at the last trade before the decision + 2¢ (known at the decision; stale prints flatter it)")
for name, P in (("state model", p_state), ("state model + market price", p_anch), ("boosted trees", p_tree)):
    if np.all(np.isnan(P)):
        continue
    for thr in (0.05, 0.08, 0.12):
        simulate(P, name, thr, known=True)
say("  the same, only when that last trade is at most 30 minutes old")
for name, P in (("state model + market price", p_anch), ("boosted trees", p_tree)):
    for thr in (0.05, 0.08, 0.12):
        simulate(P, name, thr, known=True, fresh_only=True)
say("\n## The plainest rule: buy Yes on the running bucket at the next real ask at a fixed hour, when the ask is 20–90¢")
for H0 in (13, 14, 15, 16, 17):
    for lo_, hi_ in ((0.20, 0.90), (0.40, 0.80)):
        m = (Hh == H0) & ~np.isnan(AY) & (AY >= lo_) & (AY <= hi_)
        if m.sum() < 30:
            continue
        r = Y[m] / AY[m] - 1 - FEE[m] * (1 - AY[m])
        dd = days[m]
        ud = sorted(set(dd))
        sums = np.array([r[dd == x].sum() for x in ud]); cnts = np.array([(dd == x).sum() for x in ud])
        draws = rng.integers(0, len(ud), size=(2000, len(ud)))
        bm = sums[draws].sum(1) / cnts[draws].sum(1)
        lo, hi = np.quantile(bm, [0.05, 0.95])
        say(f"  at {H0:02d}:00, ask {100 * lo_:.0f}–{100 * hi_:.0f}¢: {int(m.sum()):5d} trades on {len(ud):3d} days · {100 * r.mean():+5.1f}% per trade [90% CI {100 * lo:+.1f}, {100 * hi:+.1f}] · "
            f"paid {100 * AY[m].mean():.0f}¢ · held {100 * Y[m].mean():.0f}% · ${5 * r.sum() / len(ud) * 7:+.1f}/week at $5")
say("  and the mirror: buy No on the running bucket at the next real ask (same bands, in No terms)")
for H0 in (12, 13, 14, 15):
    m = (Hh == H0) & ~np.isnan(AN) & (AN >= 0.20) & (AN <= 0.90)
    if m.sum() < 30:
        continue
    r = (1 - Y[m]) / AN[m] - 1 - FEE[m] * (1 - AN[m])
    dd = days[m]; ud = sorted(set(dd))
    sums = np.array([r[dd == x].sum() for x in ud]); cnts = np.array([(dd == x).sum() for x in ud])
    draws = rng.integers(0, len(ud), size=(2000, len(ud)))
    bm = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.quantile(bm, [0.05, 0.95])
    say(f"  at {H0:02d}:00, No ask 20–90¢: {int(m.sum()):5d} trades on {len(ud):3d} days · {100 * r.mean():+5.1f}% per trade [90% CI {100 * lo:+.1f}, {100 * hi:+.1f}] · paid {100 * AN[m].mean():.0f}¢")
if REPORT:
    REPORT.write_text("\n".join(lines) + "\n")
