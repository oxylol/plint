"""Exit rules for the weather copy setups (weather5 / weather10): sell when the station's reports reach the bucket.

    python analyze3_weather_exit_copy.py DATA_DIR [--all] [--report OUT.md]

Replays the setups' buys from the round-3 panel's leader signals on temperature markets: the monthly-board pool,
a leader's net buy of at least $10, the round-2 model's score at the follower's price above its threshold, price
15–75¢, at least 12 h to the market's end, one position per token, five a day (as weather5 is configured), No
positions only (the setups have only ever bought No). --all drops the model and the pool and takes every leader
No signal on a temperature market with a tape, for a bigger sample. Then the same question as for the forecast
rule: hold to resolution, or sell at the first taker bid after the first hourly report at the bucket (plus a lag).
Positions whose bucket the reports had already reached before the buy are shown separately.
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polybot.model import Model  # noqa: E402

DATA = Path(sys.argv[1])
W = DATA / "weather"
ALL = "--all" in sys.argv
REPORT = None
a = sys.argv[2:]
while a:
    k = a.pop(0)
    if k == "--report":
        REPORT = Path(a.pop(0))
U = dt.timezone.utc
LAGS = [5, 20, 60]
WINDOW = 3 * 3600
rng = np.random.default_rng(3)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def rnd(x):
    return math.floor(x + 0.5)


mk = {}
for l in gzip.open(W / "markets.jsonl.gz", "rt"):
    m = json.loads(l)
    mk[m["cid"]] = m
st = json.loads((W / "stations.json").read_text())
city_st = {}
for icao, s in st.items():
    city_st.setdefault(s["city"], icao)

obs: dict[tuple, list] = defaultdict(list)
for icao, s in st.items():
    p_ = W / "obs" / f"{icao}.csv"
    if not p_.exists():
        continue
    tz = ZoneInfo(s["tz"])
    for line in p_.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", ""):
            continue
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U)
            f = float(parts[2])
        except ValueError:
            continue
        v = rnd(f) if s["unit"] == "F" else rnd((f - 32) * 5 / 9)
        obs[(icao, t.astimezone(tz).date().isoformat())].append((t.timestamp(), v))
for k in obs:
    obs[k].sort()

_tapes: dict[str, list | None] = {}


def tape(cid):
    if cid not in _tapes:
        pth = W / "tapes" / f"{cid}.json.gz"
        rows = None
        if pth.exists():
            try:
                rows = json.load(gzip.open(pth, "rt"))
                rows.sort(key=lambda r: r[0])
            except Exception:
                rows = None
        _tapes[cid] = rows
    return _tapes[cid]


def first_no_bid(rows, t, window=WINDOW):
    """What a taker got for No first after t: a No sale, or a Yes buy (the No bid was 1 − p)."""
    for ts_, idx, side, size, price, _w in rows:
        if ts_ <= t:
            continue
        if ts_ > t + window:
            return None
        if (idx == 1 and side == -1) or (idx == 0 and side == 1):
            return price if idx == 1 else 1 - price
    return None


# ── the setups' buys from the panel ──────────────────────────
cols = ["wallet", "cid", "asset", "cat", "stance", "f_is_no", "final", "t_d", "p_f", "ret", "fee_rate", "in_pool", "in_pool1000", "end", "nr",
        "o_prior_pf", "o_prior_wr", "o_prior_n_log", "o_prior_roc", "o_conviction_log", "o_cat_share", "o_cat_wr", "o_cat_n_log",
        "o_taker_share", "o_prior_hold_log", "o_prior_price", "o_prior_sports"]
p = pd.read_parquet(DATA / "panel.parquet", columns=cols)
p = p[p["cat"] == "weather"]
cids = pd.read_parquet(DATA / "panel_cids.parquet")
p["hex"] = p["cid"].map(dict(zip(cids["code"], cids["cid"])))
p = p[p["hex"].isin(mk.keys())]
n0 = len(p)
p = p[p["f_is_no"] == 1.0]
say(f"{n0} leader signals on temperature markets, {len(p)} of them No")
p = p[(p["stance"] >= 10) & (p["p_f"] >= 0.15) & (p["p_f"] <= 0.75) & (p["end"] - p["t_d"] >= 12 * 3600)]
say(f"{len(p)} with a $10 stance, No at 15–75¢ and 12 h to the end")
model = Model(Path(__file__).resolve().parents[1] / "polybot" / "model_r2.json")
assert model.ok


def xrow(r):
    days = max(0.0, (r.end - r.t_d) / 86400)
    x = {"prior_pf": r.o_prior_pf, "prior_wr": r.o_prior_wr, "prior_n_log": r.o_prior_n_log, "prior_roc": r.o_prior_roc,
         "conviction_log": r.o_conviction_log, "price": r.p_f, "price_sq": r.p_f ** 2, "days_log": math.log1p(days), "is_sports": 0.0,
         "cat_share": r.o_cat_share, "cat_wr": r.o_cat_wr, "cat_n_log": r.o_cat_n_log, "crowd_log": 0.0, "taker_share": r.o_taker_share,
         "prior_hold_log": r.o_prior_hold_log, "prior_price": r.o_prior_price, "prior_sports": r.o_prior_sports, "on_month": 1.0,
         "on_all": 0.0, "on_week": 0.0, "rank_log": math.log1p(100), "is_no": 1.0, "neg_risk": 1.0 if r.nr else 0.0}
    for c in ("politics", "geopolitics", "crypto", "culture", "economics", "tech", "weather", "other"):
        x[f"cat_{c}"] = 1.0 if c == "weather" else 0.0
    return x


if not ALL:
    p = p[p["in_pool1000"] == 1] if "--pool1000" in sys.argv else p[p["in_pool"] == 1]
    say(f"{len(p)} from leaders on the monthly board" + (" (top 1000)" if "--pool1000" in sys.argv else ""))
    p = p.copy()
    p["score"] = [model.score(xrow(r), float(r.p_f)) for r in p.itertuples()]
    p = p[p["score"] >= model.policy["threshold"]]
    say(f"{len(p)} pass the round-2 model (score ≥ {model.policy['threshold']:.3f})")
p = p[p["hex"].map(lambda h: (W / "tapes" / f"{h}.json.gz").exists())]
say(f"{len(p)} in markets with a trade tape")
p = p.sort_values("t_d").drop_duplicates("asset", keep="first")
if not ALL:
    p = p.groupby(p["t_d"] // 86400).head(5)
say(f"{len(p)} positions (one per token{'' if ALL else ', five a day'})")

trades = []
skipped = defaultdict(int)
for r in p.itertuples():
    m = mk[r.hex]
    icao = city_st.get(m["city"])
    if not icao:
        skipped["no station"] += 1
        continue
    s = st[icao]
    tz = ZoneInfo(s["tz"])
    o = obs.get((icao, m["day"]))
    if not o or len(o) < 12:
        skipped["no reports"] += 1
        continue
    if m["kind"] == "highest":
        if m["lo"] is None:
            skipped["open-ended low bucket"] += 1
            continue
        hits = [(ts_, v) for ts_, v in o if v >= m["lo"]]
    else:
        if m["hi"] is None:
            skipped["open-ended high bucket"] += 1
            continue
        hits = [(ts_, v) for ts_, v in o if v <= m["hi"]]
    rows = tape(r.hex)
    if not rows:
        skipped["no tape"] += 1
        continue
    pre = [h for h in hits if h[0] <= r.t_d]
    post = [h for h in hits if h[0] > r.t_d]
    fee = float(r.fee_rate) if not np.isnan(r.fee_rate) else 0.05
    tr = {"day": m["day"], "city": m["city"], "kind": m["kind"], "p": float(r.p_f), "fee": fee, "hold": float(r.ret), "won": float(r.final) == 1.0,
          "top": (m["hi"] is None) if m["kind"] == "highest" else (m["lo"] is None), "pre": bool(pre), "hit": bool(post),
          "entered_before_day": dt.datetime.fromtimestamp(r.t_d, tz).date().isoformat() < m["day"], "sell": {}}
    if post:
        ts_, v = post[0]
        tr["hit_hour"] = dt.datetime.fromtimestamp(ts_, tz).hour
        for lag in LAGS:
            tr["sell"][lag] = first_no_bid(rows, ts_ + 60 * lag)
    trades.append(tr)
say(f"{len(trades)} replayed; skipped {dict(skipped)}")
if not trades:
    sys.exit(0)

hold = np.array([t["hold"] for t in trades])
dd = np.array([t["day"] for t in trades])
won = np.array([t["won"] for t in trades])
hit = np.array([t["hit"] for t in trades])
pre = np.array([t["pre"] for t in trades])
top = np.array([t["top"] for t in trades])
before = np.array([t["entered_before_day"] for t in trades])
udays = sorted(set(dd))
say(f"{min(udays)} → {max(udays)}, {len(udays)} market days\n")


def ci_mean(r, days):
    ud = sorted(set(days))
    sums = np.array([r[days == x].sum() for x in ud])
    cnts = np.array([(days == x).sum() for x in ud])
    draws = rng.integers(0, len(ud), size=(2000, len(ud)))
    bm = sums[draws].sum(1) / cnts[draws].sum(1)
    return np.quantile(bm, [0.05, 0.95])


def line(name, r, extra=""):
    lo, hi = ci_mean(r, dd)
    d = r - hold
    dlo, dhi = ci_mean(d, dd)
    say(f"  {name:50s} mean {100 * r.mean():+5.1f}% per trade [90% CI {100 * lo:+.1f}, {100 * hi:+.1f}] · win {100 * (r > 0).mean():.0f}% · "
        f"${5 * r.sum() / len(udays) * 7:+.1f}/week at $5 · vs hold {100 * d.mean():+.1f} points [{100 * dlo:+.1f}, {100 * dhi:+.1f}]{extra}")


say("## The positions")
say(f"  won {100 * won.mean():.0f}% · paid {100 * np.mean([t['p'] for t in trades]):.0f}¢ avg · open-ended buckets {100 * top.mean():.0f}% · "
    f"bought before the city's day started {100 * before.mean():.0f}%")
say(f"  already reached when bought: {int(pre.sum())} ({100 * pre.mean():.0f}%) · won {100 * won[pre].mean() if pre.sum() else 0:.0f}% · hold {100 * hold[pre].mean() if pre.sum() else 0:+.1f}% per trade")
k = ~pre
say(f"  not yet reached when bought: {int(k.sum())} · reached later {int((k & hit).sum())} ({100 * hit[k].mean():.0f}%) · of those the No still won "
    f"{100 * won[k & hit].mean() if (k & hit).sum() else 0:.0f}% · never reached won {100 * won[k & ~hit].mean() if (k & ~hit).sum() else 0:.0f}%")
hrs = np.array([t.get("hit_hour", -1) for t in trades])
say("\n  by local hour of the first report at the bucket (not reached at entry): hold still won / bid 20 min later")
for lo_, hi_, lab in ((0, 12, "before noon"), (12, 15, "12–15"), (15, 18, "15–18"), (18, 24, "after 18")):
    mm = k & hit & (hrs >= lo_) & (hrs < hi_)
    if mm.sum():
        qs = [t["sell"].get(20) for t, f in zip(trades, mm) if f and t["sell"].get(20) is not None]
        say(f"    {lab:12s} {int(mm.sum()):4d} reached · hold won {100 * won[mm].mean():3.0f}% · No bid 20 min later {100 * np.mean(qs) if qs else float('nan'):.0f}¢ avg ({len(qs)} with a trade)")


def sell_rule(name, cond):
    for lag in LAGS:
        r = hold.copy()
        sold = nosale = 0
        prices = []
        for i, t in enumerate(trades):
            if not cond(t):
                continue
            q = t["sell"].get(lag)
            if q is None:
                nosale += 1
                continue
            sold += 1
            prices.append(q)
            r[i] = q / t["p"] - 1 - t["fee"] * (1 - t["p"]) - t["fee"] * q * (1 - q) / t["p"]
        line(f"{name}, {lag} min lag", r, f" · sold {sold} at {100 * np.mean(prices) if prices else 0:.0f}¢ avg, {nosale} had no buyer")


say("\n## Hold vs sell, $5 a trade, taker fee on both legs")
line("hold to resolution (the setups today)", hold)
sell_rule("A sell when the bucket is reached", lambda t: t["hit"] and not t["pre"])
sell_rule("D sell when reached after local noon", lambda t: t["hit"] and not t["pre"] and t.get("hit_hour", 0) >= 12)
r = np.where(pre, 0.0, hold)
line("E hold, but never buy a bucket already reached", r)

q = np.array([t["sell"].get(20) if t["sell"].get(20) is not None else np.nan for t in trades])
kk = k & hit & ~np.isnan(q)
if kk.sum():
    say(f"\n  reached after the buy, with a bid 20 min later: {int(kk.sum())} trades · bid {100 * np.nanmean(q[kk]):.0f}¢ avg · hold paid out "
        f"{100 * won[kk].mean():.0f}¢ avg · selling there {'gave up' if won[kk].mean() > np.nanmean(q[kk]) else 'saved'} {100 * abs(won[kk].mean() - np.nanmean(q[kk])):.0f}¢ a share before fees")
if REPORT:
    REPORT.write_text("\n".join(lines) + "\n")
