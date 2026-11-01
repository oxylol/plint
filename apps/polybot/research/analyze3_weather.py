"""Daily temperature markets as a probabilistic forecast, against the +2°C rule and the market itself.

    python analyze3_weather.py DATA_DIR [--report OUT.md] [--model OUT.json]

For every (city, day, highest|lowest) event: the official value from the
station's METAR reports (IEM archive; the markets resolve on Weather
Underground's daily observations table, i.e. those reports in whole degrees),
the previous-day runs of the bot's five Open-Meteo models, and the market's
midpoint at the decision time. A forecast distribution (ensemble mean plus a
walk-forward bias, Student-t errors scaled by the city's recent error spread)
gives each degree bucket a probability; a logistic blend with the market price
calibrates it; the policy buys the side whose edge after the half-spread and
the taker fee clears a threshold. Everything is fitted on days before the one
being traded. Compared with the bot's rule (No on the bucket at the highest
model forecast + 2°C, No priced 50–95¢).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats

DATA = Path(sys.argv[1]) / "weather"
REPORT = MODEL_OUT = None
a = sys.argv[2:]
while a:
    k = a.pop(0)
    if k == "--report":
        REPORT = Path(a.pop(0))
    elif k == "--model":
        MODEL_OUT = Path(a.pop(0))
    elif k in ("--mids", "--no-regress", "--day2", "--late", "--entry-last"):
        pass
U = dt.timezone.utc
MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
USE_TAPES = "--mids" not in sys.argv     # default: prices from the real taker tape; --mids uses the midpoint history
# --entry-last: buy at the last trade before the decision plus the book's typical $5 cost, known at the decision (the
# default prices a buy at the next taker trade on that side within 3 h, which also skips markets nobody traded after)
ENTRY_LAST = "--entry-last" in sys.argv
REGRESS = "--no-regress" not in sys.argv  # per-city ridge on the five models (else ensemble mean + bias)
# previous_day1: the run from about a day before each hour (for afternoons in the Americas that can be later than the
# decision at 12:00 UTC the day before); previous_day2 (--day2) is a day older still, certainly known in time
LEAD = "previous_day2" if "--day2" in sys.argv else "previous_day1"
LATE = "--late" in sys.argv
rng = np.random.default_rng(3)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def rnd(x):
    return math.floor(x + 0.5)


# ── markets ──────────────────────────────────────────────────
mk = [json.loads(l) for l in gzip.open(DATA / "markets.jsonl.gz", "rt")]
st = json.loads((DATA / "stations.json").read_text())
city_st = {}
for icao, s in st.items():
    city_st.setdefault(s["city"], icao)
events = defaultdict(list)
for m in mk:
    if not m["closed"]:
        continue
    pr = m.get("prices") or [None, None]
    fy = 1.0 if (pr[0] is not None and pr[0] >= 0.99) else 0.0 if (pr[0] is not None and pr[0] <= 0.01) else None
    if fy is None:
        continue
    m["final"] = fy
    events[(m["city"], m["day"], m["kind"])].append(m)
say(f"{len(mk)} temperature markets, {sum(len(v) for v in events.values())} resolved in {len(events)} events, {len(city_st)} cities with a station")

# ── observations: daily max / min in the station's local day, whole degrees in the market's unit ──
obs_day: dict[tuple, dict] = {}
for icao, s in st.items():
    p = DATA / "obs" / f"{icao}.csv"
    if not p.exists():
        continue
    tz = ZoneInfo(s["tz"])
    vals = defaultdict(list)
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", ""):
            continue
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U).astimezone(tz)
            f = float(parts[2])
        except ValueError:
            continue
        v = rnd(f) if s["unit"] == "F" else rnd((f - 32) * 5 / 9)
        vals[t.date().isoformat()].append(v)
    for d, vs in vals.items():
        if len(vs) >= 12:
            obs_day[(icao, d)] = {"highest": max(vs), "lowest": min(vs), "n": len(vs)}

# ── forecasts: previous-day runs, daily max / min per model ──
fc_day: dict[tuple, dict] = {}
for icao, s in st.items():
    p = DATA / "forecasts" / f"{icao}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    h = d.get(f"temperature_2m_{LEAD}") or {}
    times = h.get("time") or []
    for m in MODELS:
        vals = h.get(f"temperature_2m_{LEAD}_{m}") or []
        by = defaultdict(list)
        for t, v in zip(times, vals):
            if v is not None:
                by[t[:10]].append(v)
        for day, vs in by.items():
            if len(vs) >= 20:
                conv = (lambda c: c * 9 / 5 + 32) if s["unit"] == "F" else (lambda c: c)
                e = fc_day.setdefault((icao, day), {"highest": {}, "lowest": {}})
                e["highest"][m] = conv(max(vs))
                e["lowest"][m] = conv(min(vs))

# ── prices at the decision time ──
px: dict[str, list] = {}
pp = DATA / "prices.jsonl.gz"
if pp.exists() and not USE_TAPES:
    try:   # the collector may still be appending: read what is complete
        for line in gzip.open(pp, "rt"):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if j["p"]:
                px[j["cid"]] = j["p"]
    except EOFError:
        pass


TAPE_DIR = DATA / "tapes"


def tape_quotes(cid, t, ask_window=3 * 3600):
    """From the market's taker trades: the last trade before t (in Yes terms) and the prices a taker actually paid for
    Yes and for No in the `ask_window` after t (a Yes buy at p, or a No sale at q, which means Yes was offered at 1 − q)."""
    pth = TAPE_DIR / f"{cid}.json.gz"
    if not pth.exists():
        return None
    try:
        rows = json.load(gzip.open(pth, "rt"))
    except Exception:
        return None
    rows.sort(key=lambda r: r[0])
    last = ask_y = ask_n = None
    for ts_, idx, side, size, price, _w in rows:
        yes_px = price if idx == 0 else 1 - price
        if ts_ <= t:
            if ts_ >= t - 86400:
                last = yes_px
            continue
        if ts_ > t + ask_window:
            break
        buys_yes = (idx == 0 and side == 1) or (idx == 1 and side == -1)
        if buys_yes and ask_y is None:
            ask_y = yes_px
        if not buys_yes and ask_n is None:
            ask_n = 1 - yes_px
        if ask_y is not None and ask_n is not None:
            break
    return {"last": last, "ask_yes": ask_y, "ask_no": ask_n, "n": len(rows)}


def price_at(cid, t):
    pts = px.get(cid)
    if not pts:
        return None
    best = None
    for tt, p in pts:
        if tt <= t:
            best = p
        else:
            break
    return best


# half-spread on temperature books, from the live snapshots (median ask − mid by price band)
HS = {0: 0.01, 1: 0.01, 2: 0.01}
try:
    import glob
    diffs = defaultdict(list)
    for f in glob.glob(str(DATA.parent / "books" / "*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            b = json.loads(line)
            if "temperature in" not in b.get("q", "") or b.get("bid") is None or b.get("ask") is None:
                continue
            mid = (b["bid"] + b["ask"]) / 2
            if b["buy"].get("5") is not None:
                diffs[0 if mid < 0.15 else 1 if mid < 0.85 else 2].append(b["buy"]["5"] - mid)
    for k, v in diffs.items():
        if len(v) > 30:
            HS[k] = float(np.median(v))
    say(f"cost of a $5 buy above the mid on temperature books (median, from {sum(len(v) for v in diffs.values())} snapshots): "
        + ", ".join(f"{['<15¢', '15–85¢', '>85¢'][k]} {100 * HS[k]:.1f}¢" for k in sorted(HS)))
except Exception as e:
    say(f"no book snapshots ({e}); half-spread 1¢")


def cost_above_mid(p):
    return HS[0 if p < 0.15 else 1 if p < 0.85 else 2]


# ── assemble events ──────────────────────────────────────────
rows = []
mismatch = matched = 0
for (city, day, kind), ms in events.items():
    icao = city_st.get(city)
    if not icao:
        continue
    o = obs_day.get((icao, day))
    f = fc_day.get((icao, day))
    winners = [m for m in ms if m["final"] == 1.0]
    if o is None or f is None or len(winners) != 1 or len(f[kind]) < 3:
        continue
    w = winners[0]
    val = o[kind]
    if (w["lo"] is None or val >= w["lo"]) and (w["hi"] is None or val <= w["hi"]):
        matched += 1
    else:
        mismatch += 1
    tz = ZoneInfo(st[icao]["tz"])
    d0 = dt.datetime.fromisoformat(day).replace(tzinfo=tz)
    if LATE:   # the bot's own timing: just before the city's day starts (the previous-day run is out well before)
        t_dec = d0.astimezone(U) - dt.timedelta(hours=1)
    else:      # 12:00 UTC the day before, everywhere
        t_dec = min(d0.astimezone(U) - dt.timedelta(hours=1), dt.datetime.fromisoformat(day).replace(tzinfo=U) - dt.timedelta(hours=12))
    rows.append({"city": city, "day": day, "kind": kind, "unit": ms[0]["unit"], "val": val, "fc": f[kind], "markets": ms,
                 "t_dec": t_dec.timestamp(), "official_lo": w["lo"], "official_hi": w["hi"]})
say(f"{len(rows)} events with observations, forecasts and one winning bucket; METAR max/min falls in the winning bucket in "
    f"{100 * matched / max(1, matched + mismatch):.1f}% of them ({mismatch} disagree)")
rows.sort(key=lambda r: r["day"])

# ── the forecast distribution, walk-forward ──────────────────
hist: dict[tuple, list] = defaultdict(list)     # (city, kind) → [(day, residual)]
fhist: dict[tuple, list] = defaultdict(list)    # (city, kind) → [(day, [model forecasts], actual)] for the regression
DF = 4


def bucket_prob(mu, sd, lo, hi):
    up = 1.0 if hi is None else stats.t.cdf((hi + 0.5 - mu) / sd, DF)
    dn = 0.0 if lo is None else stats.t.cdf((lo - 0.5 - mu) / sd, DF)
    return max(1e-4, min(1 - 1e-4, up - dn))


trades = []          # one row per market with a price: features for the policy
for r in rows:
    key = (r["city"], r["kind"])
    ens = float(np.mean(list(r["fc"].values())))
    cutoff = (dt.date.fromisoformat(r["day"]) - dt.timedelta(days=1)).isoformat()
    past = [x for d, x in hist[key] if d < cutoff][-45:]
    unit_scale = 1.8 if r["unit"] == "F" else 1.0
    fx = [r["fc"].get(m, ens) for m in MODELS]
    if len(past) >= 10:
        bias = float(np.mean(past))
        sd = max(0.6 * unit_scale, float(np.std(past)) * math.sqrt((DF - 2) / DF))
        spread = float(np.std(list(r["fc"].values())))
        sd = math.sqrt(sd ** 2 + (0.5 * spread) ** 2)
        mu = ens + bias
        # per-city ridge on the five models (60 past days), when it has enough history: weights the better models
        fp = [(f, v) for d, f, v in fhist[key] if d < cutoff][-60:]
        if REGRESS and len(fp) >= 30:
            Xr = np.array([[1.0] + f for f, _ in fp])
            yr = np.array([v for _, v in fp])
            lam = np.diag([0.0] + [5.0] * len(MODELS))
            wr = np.linalg.solve(Xr.T @ Xr + lam, Xr.T @ yr)
            res = yr - Xr @ wr
            mu = float(np.array([1.0] + fx) @ wr)
            sd = max(0.6 * unit_scale, float(np.std(res)) * math.sqrt(len(fp) / max(1, len(fp) - 6)) * math.sqrt((DF - 2) / DF))
        hi_max = max(r["fc"].values())
        for m in r["markets"]:
            tq = tape_quotes(m["cid"], r["t_dec"]) if USE_TAPES else None
            if USE_TAPES:
                if tq is None or tq["last"] is None:
                    continue
                mid = tq["last"]
            else:
                mid = price_at(m["cid"], r["t_dec"])
            if mid is None or not (0.005 < mid < 0.995):
                continue
            pm = bucket_prob(mu, sd, m["lo"], m["hi"])
            trades.append({"day": r["day"], "city": r["city"], "kind": r["kind"], "unit": r["unit"], "cid": m["cid"], "lo": m["lo"], "hi": m["hi"],
                           "mid": mid, "ask_yes": tq["ask_yes"] if tq else None, "ask_no": tq["ask_no"] if tq else None,
                           "p_model": pm, "final": m["final"], "fee": m.get("fee_rate", 0.05),
                           "rule_target": rnd(max(r["fc"].values()) + 2 * unit_scale) if r["kind"] == "highest" else None,
                           "mu": mu, "sd": sd, "ev": (r["city"], r["day"], r["kind"])})
    hist[key].append((r["day"], r["val"] - ens))
    fhist[key].append((r["day"], fx, r["val"]))
say(f"{len(trades)} priced markets with a walk-forward forecast distribution "
    f"({len({t['ev'] for t in trades})} events, {min(t['day'] for t in trades) if trades else '-'} → {max(t['day'] for t in trades) if trades else '-'})")
if not trades:
    sys.exit(0)

T = {k: np.array([t[k] for t in trades]) for k in ("mid", "p_model", "final", "fee")}
day = np.array([t["day"] for t in trades])
udays = sorted(set(day))


# ── calibration of the raw forecast probability and the market ──
def calib(p, y, name):
    bins = [0, .02, .05, .1, .2, .35, .5, .65, .8, .9, .95, .98, 1.0001]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            out.append(f"{100 * lo:.0f}–{100 * min(hi, 1):.0f}%: {100 * p[m].mean():.1f}→{100 * y[m].mean():.1f} (n {m.sum()})")
    ll = -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
    br = np.mean((p - y) ** 2)
    say(f"  {name}: log loss {ll:.4f}, Brier {br:.4f}; predicted→actual " + " · ".join(out))
    return ll


say("\n## Calibration (every priced bucket)")
ll_mkt = calib(T["mid"], T["final"], "market mid")
ll_fc = calib(T["p_model"], T["final"], "forecast")


# walk-forward logistic blend: y ~ a·logit(mid) + b·logit(p_model) + c, fitted on days before
def lg(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit_logit(X, y, lam=1.0):
    w = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


X = np.column_stack([lg(T["mid"]), lg(T["p_model"]), np.ones(len(T["mid"]))])
p_blend = np.full(len(T["mid"]), np.nan)
weights = {}
day_i = {d: i for i, d in enumerate(udays)}
di = np.array([day_i[d] for d in day])
for i, d in enumerate(udays):
    tr = di < i - 1           # outcomes of the day before are not in yet
    te = di == i
    if tr.sum() < 3000:
        continue
    w = fit_logit(X[tr], T["final"][tr])
    weights[d] = w
    p_blend[te] = 1 / (1 + np.exp(-X[te] @ w))
ok = ~np.isnan(p_blend)
say(f"\nblend fitted walk-forward on {ok.sum()} buckets (from {udays[int(np.argmax([d in weights for d in udays]))] if weights else '-'}); "
    f"last weights: market {list(weights.values())[-1][0]:+.2f}, forecast {list(weights.values())[-1][1]:+.2f}, const {list(weights.values())[-1][2]:+.2f}")
calib(T["mid"][ok], T["final"][ok], "market mid (same buckets)")
calib(T["p_model"][ok], T["final"][ok], "forecast (same buckets)")
calib(p_blend[ok], T["final"][ok], "blend")


# ── trading ──────────────────────────────────────────────────
def entry_price(i, side_yes):
    """What buying costs: the tape's next taker price (tapes), else the mid plus the book's typical $5 cost."""
    t = trades[i]
    if USE_TAPES and not ENTRY_LAST:
        return t["ask_yes"] if side_yes else t["ask_no"]
    m = t["mid"] if side_yes else 1 - t["mid"]
    return m + cost_above_mid(m)


def pnl_for(side_yes, mid, final, fee, ii=None):
    """Return per $ for buying Yes (or No) at the entry price, held to resolution, after the taker fee."""
    if ii is not None:
        p = np.array([entry_price(i, s) for i, s in zip(ii, side_yes)], dtype=float)
    else:
        p = np.where(side_yes, mid, 1 - mid)
        p = p + np.array([cost_above_mid(x) for x in p])
    p = np.clip(p, 0.01, 0.995)
    won = np.where(side_yes, final, 1 - final)
    return won / p - 1 - fee * (1 - p), p   # fee per $ = rate × (1 − p)


def simulate(choose, name, per_day=10, per_event=2):
    """choose(i) → (side_yes, edge) or None; take the best `per_day` a day, at most `per_event` per event."""
    picks = []
    for d in udays:
        idx = np.where((day == d) & ok)[0]
        cand = []
        for i in idx:
            c = choose(i)
            if c is not None:
                cand.append((c[1], i, c[0]))
        cand.sort(reverse=True)
        used = defaultdict(int)
        n = 0
        for e, i, side in cand:
            ev = trades[i]["ev"]
            if used[ev] >= per_event:
                continue
            used[ev] += 1
            picks.append((d, i, side))
            n += 1
            if n >= per_day:
                break
    if not picks:
        say(f"  {name}: no trades")
        return None
    ii = np.array([p[1] for p in picks])
    side = np.array([p[2] for p in picks])
    r, paid = pnl_for(side, T["mid"][ii], T["final"][ii], T["fee"][ii], ii)
    dd = np.array([p[0] for p in picks])
    ud = sorted(set(dd))
    sums = np.array([r[dd == x].sum() for x in ud])
    cnts = np.array([(dd == x).sum() for x in ud])
    draws = rng.integers(0, len(ud), size=(2000, len(ud)))
    bm = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.quantile(bm, [0.05, 0.95])
    say(f"  {name:44s} {len(r):5d} trades on {len(ud):3d} days · mean {100 * r.mean():+6.1f}% per trade [90% CI {100 * lo:+.1f}, {100 * hi:+.1f}] · "
        f"win {100 * (r > 0).mean():.0f}% · paid {100 * paid.mean():.0f}¢ avg · ${5 * r.sum() / len(ud) * 7:+.1f}/week at $5")
    return {"name": name, "n": int(len(r)), "mean": float(r.mean()), "lo": float(lo), "hi": float(hi), "win": float((r > 0).mean()),
            "per_week": float(5 * r.sum() / len(ud) * 7), "days": len(ud), "ret": r, "dd": dd}


say("\n## Trading, $5 a trade, at most 10 a day, held to resolution, half-spread and fee paid")
res = {}


def rule(i):
    t = trades[i]
    if t["kind"] != "highest" or t["rule_target"] is None:
        return None
    if (t["lo"] is None or t["rule_target"] >= t["lo"]) and (t["hi"] is None or t["rule_target"] <= t["hi"]):
        no_px = entry_price(i, False)
        if no_px is not None and 0.50 <= no_px <= 0.95:
            return (False, no_px)
    return None


res["rule"] = simulate(rule, "bot rule: No at max forecast +2°C", per_event=1)
if res["rule"]:
    rr, dd = res["rule"]["ret"], np.array(res["rule"]["dd"])
    for lo_, hi_ in (("2026-08-15", "2026-09-15"), ("2026-09-15", "2026-10-20"), ("2026-10-20", "2026-11-01")):
        s_ = (dd >= lo_) & (dd < hi_)
        if s_.sum():
            say(f"    {lo_} → {hi_}: {s_.sum():4d} trades, {100 * rr[s_].mean():+5.1f}% per trade, win {100 * (rr[s_] > 0).mean():.0f}%")

for src, P in (("forecast", T["p_model"]), ("blend", p_blend)):
    for thr in (0.03, 0.05, 0.08, 0.12):
        def ch(i, P=P, thr=thr):
            if np.isnan(P[i]):
                return None
            ay, an = entry_price(i, True), entry_price(i, False)
            ey = P[i] - ay - T["fee"][i] * ay * (1 - ay) if ay is not None else -9
            en = (1 - P[i]) - an - T["fee"][i] * an * (1 - an) if an is not None else -9
            if max(ey, en) < thr:
                return None
            return (True, ey) if ey >= en else (False, en)
        res[f"{src}_{thr}"] = simulate(ch, f"{src}: best edge ≥ {100 * thr:.0f}¢")
    for thr in (0.03, 0.05, 0.08):
        def ch_no(i, P=P, thr=thr):
            if np.isnan(P[i]):
                return None
            an = entry_price(i, False)
            if an is None:
                return None
            en = (1 - P[i]) - an - T["fee"][i] * an * (1 - an)
            return (False, en) if en >= thr and 0.5 <= an <= 0.97 else None
        res[f"{src}_no_{thr}"] = simulate(ch_no, f"{src}: No only, edge ≥ {100 * thr:.0f}¢, No 50–97¢")

best = max((v for k, v in res.items() if v and k != "rule" and v["n"] >= 100), key=lambda v: v["lo"], default=None)
if best and res.get("rule"):
    b, r0 = best, res["rule"]
    # paired by day: difference of daily mean returns
    ud = sorted(set(b["dd"]) & set(r0["dd"]))
    diff = np.array([b["ret"][b["dd"] == x].mean() - r0["ret"][r0["dd"] == x].mean() for x in ud])
    dr = rng.integers(0, len(ud), size=(2000, len(ud)))
    q = np.quantile(diff[dr].mean(1), [0.05, 0.5, 0.95])
    say(f"\nbest ({b['name']}) minus the bot rule, per trade, paired by day over {len(ud)} common days: median {100 * q[1]:+.1f} points, 90% CI [{100 * q[0]:+.1f}, {100 * q[2]:+.1f}]")

if REPORT:
    REPORT.write_text("\n".join(lines) + "\n")
if MODEL_OUT and weights:
    w = list(weights.values())[-1]
    MODEL_OUT.write_text(json.dumps({"kind": "weather_blend", "df": DF, "window_days": 45, "w_market": float(w[0]), "w_forecast": float(w[1]),
                                     "const": float(w[2]), "half_spread": HS, "models": MODELS}, indent=1))
