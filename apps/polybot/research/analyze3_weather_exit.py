"""Exit rules for the forecast strategy: sell when the station's reports reach the bucket we bet against.

    python analyze3_weather_exit.py DATA_DIR [--report OUT.md]

Replays the bot's rule as analyze3_weather.py --late does (No on the bucket at the highest model forecast + 2°C,
bought just before the city's day starts at the tape's next No price, 50–95¢, one per event, ten a day) and then
follows each trade through the day: the first hourly METAR report (IEM archive, whole degrees as the markets read
them) at or above the bucket, and what a No seller got from the tape after that moment plus a lag (the report's
publication and the bot's tick). Holding to resolution is compared with
  A  sell as soon as the bucket is reached,
  B  sell when it is reached and the bucket is the open-ended top one (the day's high is then at least that),
  C  sell one degree early (the first report at bucket − 1°),
  D  sell when reached, but only if the report is after local noon (what climbs in the morning tends to climb on).
Returns are per $ staked after the taker fee on both legs; "could not sell" (no taker trade on the bid side within
three hours of the moment) holds to resolution and is counted.
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

DATA = Path(sys.argv[1]) / "weather"
REPORT = None
a = sys.argv[2:]
while a:
    k = a.pop(0)
    if k == "--report":
        REPORT = Path(a.pop(0))
U = dt.timezone.utc
MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
LEAD = "previous_day1"
LAGS = [5, 20, 60]            # minutes between the report's valid time and the sale
WINDOW = 3 * 3600             # seconds to wait for a taker trade on the right side
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
    if not m["closed"] or m["kind"] != "highest":
        continue
    pr = m.get("prices") or [None, None]
    fy = 1.0 if (pr[0] is not None and pr[0] >= 0.99) else 0.0 if (pr[0] is not None and pr[0] <= 0.01) else None
    if fy is None:
        continue
    m["final"] = fy
    events[(m["city"], m["day"])].append(m)

# ── observations: every report with its UTC time and whole-degree value in the market's unit ──
obs: dict[tuple, list] = defaultdict(list)      # (icao, local day) → [(ts, value)]
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
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U)
            f = float(parts[2])
        except ValueError:
            continue
        v = rnd(f) if s["unit"] == "F" else rnd((f - 32) * 5 / 9)
        obs[(icao, t.astimezone(tz).date().isoformat())].append((t.timestamp(), v))
for k in obs:
    obs[k].sort()

# ── forecasts: previous-day runs, daily max per model ──
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
                c = max(vs)
                fc_day.setdefault((icao, day), {})[m] = c * 9 / 5 + 32 if s["unit"] == "F" else c

# ── tapes ──
TAPE_DIR = DATA / "tapes"
_tapes: dict[str, list | None] = {}


def tape(cid):
    if cid not in _tapes:
        pth = TAPE_DIR / f"{cid}.json.gz"
        rows = None
        if pth.exists():
            try:
                rows = json.load(gzip.open(pth, "rt"))
                rows.sort(key=lambda r: r[0])
            except Exception:
                rows = None
        _tapes[cid] = rows
    return _tapes[cid]


def no_price(idx, price):
    return price if idx == 1 else 1 - price


def first_no_ask(rows, t, window=WINDOW):
    """What a taker paid for No first after t: a No buy, or a Yes sale (No was offered at 1 − p)."""
    for ts_, idx, side, size, price, _w in rows:
        if ts_ <= t:
            continue
        if ts_ > t + window:
            return None
        if (idx == 1 and side == 1) or (idx == 0 and side == -1):
            return no_price(idx, price)
    return None


def first_no_bid(rows, t, window=WINDOW):
    """What a taker got for No first after t: a No sale, or a Yes buy (the No bid was 1 − p)."""
    for ts_, idx, side, size, price, _w in rows:
        if ts_ <= t:
            continue
        if ts_ > t + window:
            return None
        if (idx == 1 and side == -1) or (idx == 0 and side == 1):
            return no_price(idx, price)
    return None


# ── the bot's trades, with the day's reports ──────────────────
trades = []
no_obs = no_fc = no_bucket = no_price_ = 0
for (city, day), ms in events.items():
    icao = city_st.get(city)
    if not icao:
        continue
    o = obs.get((icao, day))
    f = fc_day.get((icao, day))
    winners = [m for m in ms if m["final"] == 1.0]
    if not o or len(o) < 12:
        no_obs += 1
        continue
    if f is None or len(f) < 3 or len(winners) != 1:
        no_fc += 1
        continue
    s = st[icao]
    tz = ZoneInfo(s["tz"])
    d0 = dt.datetime.fromisoformat(day).replace(tzinfo=tz)
    t_dec = (d0.astimezone(U) - dt.timedelta(hours=1)).timestamp()
    unit_scale = 1.8 if s["unit"] == "F" else 1.0
    target = rnd(max(f.values()) + 2 * unit_scale)
    b = next((m for m in ms if (m["lo"] is None or target >= m["lo"]) and (m["hi"] is None or target <= m["hi"])), None)
    if b is None or b["lo"] is None:
        no_bucket += 1
        continue
    rows = tape(b["cid"])
    if not rows:
        no_price_ += 1
        continue
    p = first_no_ask(rows, t_dec)
    if p is None or not (0.50 <= p <= 0.95):
        no_price_ += 1
        continue
    fee = b.get("fee_rate", 0.05)
    final_no = 1.0 - b["final"]
    hold = final_no / p - 1 - fee * (1 - p)
    day_max = max(v for _, v in o)
    hit = next(((ts_, v) for ts_, v in o if v >= b["lo"]), None)
    near = next(((ts_, v) for ts_, v in o if v >= b["lo"] - 1), None)
    tr = {"city": city, "day": day, "cid": b["cid"], "lo": b["lo"], "top": b["hi"] is None, "p": p, "fee": fee, "hold": hold,
          "won": final_no == 1.0, "day_max": day_max, "hit": hit is not None, "near": near is not None, "sell": {}}
    for name, h in (("hit", hit), ("near", near)):
        if h is None:
            continue
        ts_, v = h
        tr[f"{name}_hour"] = dt.datetime.fromtimestamp(ts_, tz).hour
        for lag in LAGS:
            q = first_no_bid(rows, ts_ + 60 * lag)
            tr["sell"][(name, lag)] = q
    trades.append(tr)

say(f"{len(trades)} bot trades replayed (just before the city's day starts, No at max forecast +2°C, 50–95¢); "
    f"skipped: {no_obs} events without reports, {no_fc} without forecasts/one winner, {no_bucket} without a bucket, {no_price_} without a No price")
# the study's budget: one per event (already), ten a day, the most expensive No first
by_day = defaultdict(list)
for t in trades:
    by_day[t["day"]].append(t)
trades = [t for d in sorted(by_day) for t in sorted(by_day[d], key=lambda x: -x["p"])[:10]]
udays = sorted(by_day)
say(f"{len(trades)} after the study's ten-a-day budget, {min(udays)} → {max(udays)}, {len(udays)} days\n")

hold = np.array([t["hold"] for t in trades])
dd = np.array([t["day"] for t in trades])
won = np.array([t["won"] for t in trades])
hit = np.array([t["hit"] for t in trades])
top = np.array([t["top"] for t in trades])


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
    say(f"  {name:52s} mean {100 * r.mean():+5.1f}% per trade [90% CI {100 * lo:+.1f}, {100 * hi:+.1f}] · win {100 * (r > 0).mean():.0f}% · "
        f"${5 * r.sum() / len(udays) * 7:+.1f}/week at $5 · vs hold {100 * d.mean():+.1f} points [{100 * dlo:+.1f}, {100 * dhi:+.1f}]{extra}")


say("## What the day did to the bot's No positions")
say(f"  won {100 * won.mean():.0f}% · paid {100 * np.mean([t['p'] for t in trades]):.0f}¢ avg · top (open-ended) buckets {100 * top.mean():.0f}%")
say(f"  the reports reached the bucket in {100 * hit.mean():.0f}% of trades ({hit.sum()}); of those, the position still won {100 * won[hit].mean():.0f}% "
    f"(the high went past a one-degree bucket); among never-reached trades it won {100 * won[~hit].mean():.0f}%")
say(f"  top buckets reached: {int((hit & top).sum())}, won anyway {int((hit & top & won).sum())} (report vs resolution disagreements); "
    f"one-degree buckets reached: {int((hit & ~top).sum())}, won anyway {int((hit & ~top & won).sum())}")
hrs = np.array([t.get("hit_hour", -1) for t in trades])
say("\n  by local hour of the first report at the bucket (one-degree buckets): hold still won / reached")
for lo_, hi_, lab in ((0, 12, "before noon"), (12, 15, "12–15"), (15, 18, "15–18"), (18, 24, "after 18")):
    m = hit & ~top & (hrs >= lo_) & (hrs < hi_)
    if m.sum():
        q = [t["sell"].get(("hit", 20)) for t in trades]
        qs = [x for x, mm in zip(q, m) if mm and x is not None]
        say(f"    {lab:12s} {int(m.sum()):4d} reached · hold won {100 * won[m].mean():3.0f}% · No bid 20 min later {100 * np.mean(qs) if qs else float('nan'):.0f}¢ avg ({len(qs)} with a trade)")
m = hit & top
if m.sum():
    q = [t["sell"].get(("hit", 20)) for t in trades]
    qs = [x for x, mm in zip(q, m) if mm and x is not None]
    say(f"    top buckets  {int(m.sum()):4d} reached · hold won {100 * won[m].mean():3.0f}% · No bid 20 min later {100 * np.mean(qs) if qs else float('nan'):.0f}¢ avg ({len(qs)} with a trade)")


def sell_rule(name, cond, key):
    """Return per trade when selling at the first bid after the moment `key` for trades where cond(t) holds."""
    for lag in LAGS:
        r = hold.copy()
        sold = nosale = 0
        prices = []
        for i, t in enumerate(trades):
            if not cond(t):
                continue
            q = t["sell"].get((key, lag))
            if q is None:
                nosale += 1
                continue
            sold += 1
            prices.append(q)
            r[i] = q / t["p"] - 1 - t["fee"] * (1 - t["p"]) - t["fee"] * q * (1 - q) / t["p"]
        line(f"{name}, {lag} min lag", r, f" · sold {sold} at {100 * np.mean(prices) if prices else 0:.0f}¢ avg, {nosale} had no buyer")


say("\n## Hold vs sell, $5 a trade, taker fee on both legs")
line("hold to resolution (the bot today)", hold)
sell_rule("A sell when the bucket is reached", lambda t: t["hit"], "hit")
sell_rule("B sell when reached, top buckets only", lambda t: t["hit"] and t["top"], "hit")
sell_rule("C sell one degree early", lambda t: t["near"], "near")
sell_rule("D sell when reached after local noon", lambda t: t["hit"] and t.get("hit_hour", 0) >= 12, "hit")

# what the market knew: among reached one-degree buckets, the bid 20 min later vs how often hold still won
m = hit & ~top
q = np.array([t["sell"].get(("hit", 20)) if t["sell"].get(("hit", 20)) is not None else np.nan for t in trades])
k = m & ~np.isnan(q)
if k.sum():
    say(f"\n  one-degree buckets reached, with a bid 20 min later: {int(k.sum())} trades · bid {100 * np.nanmean(q[k]):.0f}¢ avg · "
        f"hold paid out {100 * won[k].mean():.0f}¢ avg (win rate) · so selling there {'gave up' if won[k].mean() > np.nanmean(q[k]) else 'saved'} "
        f"{100 * abs(won[k].mean() - np.nanmean(q[k])):.0f}¢ a share before fees")
k = hit & top & ~np.isnan(q)
if k.sum():
    say(f"  top buckets reached, with a bid 20 min later: {int(k.sum())} trades · bid {100 * np.nanmean(q[k]):.0f}¢ avg · hold paid out {100 * won[k].mean():.0f}¢ avg")

if REPORT:
    REPORT.write_text("\n".join(lines) + "\n")
