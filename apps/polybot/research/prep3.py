"""Join the signals with what a follower would really have paid, and with who was on the boards at the time.

    python prep3.py DATA_DIR [latency_minutes=5]

Reads DATA_DIR/events (build3.py), DATA_DIR/prices (collect3_prices.py, midpoint history),
DATA_DIR/fill_model.json (fillmodel3.py), DATA_DIR/wallet_days.parquet and, when present,
DATA_DIR/tapes (collect3_tapes.py: the real taker trades of thin markets). Writes
DATA_DIR/panel.parquet: one row per resolved signal with

  mid_pre   midpoint before the leader's fill        mid_d    midpoint when the follower can act
  mid_30/60/240  midpoints 30, 60 and 240 minutes later (latency tests)
  cost      what a $5 market buy pays above the mid (book snapshots: type, price, time to end, in play)
  p_f       the follower's price: the tape's first taker buy after the signal in thin markets, else mid_d + cost
  exit_mid  midpoint when the follower could copy the leader's sale (exit-on-sell policy)
  ret       return per $ held to resolution after the taker fee (shares × rate × p × (1 − p))
  in_pool   the wallet was in the top 200 of the 30-day board (rebuilt from resolved P&L) the day before
  in_day25  in the top 25 of the previous day's P&L (the day board)
  crowd     other pool wallets that signalled the same outcome in the 48 hours before
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fillmodel3 import FillModel  # noqa: E402
from polybot import features as F  # noqa: E402

DATA = Path(sys.argv[1])
LAT = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 0.0   # extra latency on top of build3's 5 minutes
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


import pyarrow.parquet as pq  # noqa: E402

cols_all = pq.ParquetDataset(DATA / "events").schema.names
cols = [c for c in cols_all if c not in ("title", "slug", "outcome", "boards")]
ev = pd.read_parquet(DATA / "events", columns=cols, filters=[("has_market", "=", True), ("rec_n", ">=", 5),
                                                              ("leader_px", ">", 0.03), ("leader_px", "<", 0.97)])
ev = ev[ev.final.notna()].reset_index(drop=True)
for c in ev.columns:
    if ev[c].dtype == np.float64 and (c.startswith("o_") or c in ("stance", "lsell_px", "fee_rate")):
        ev[c] = ev[c].astype(np.float32)
ev["t_d"] = ev.t_d + LAT
log(f"{len(ev)} resolved signals with a record, {ev.wallet.nunique()} wallets")

# ── midpoints from the price history, streamed (the paths do not fit in memory) ──
idx = defaultdict(list)
for i, a in enumerate(ev.asset.values):
    idx[a].append(i)
n = len(ev)
cols = {k: np.full(n, np.nan) for k in ("mid_pre", "mid_d", "mid_30", "mid_60", "mid_240", "exit_mid", "bucket")}
t_d = ev.t_d.values
t_fill = ev.t_fill.values
t_ls = ev.t_lsell.values
seen = 0
for p in sorted((DATA / "prices").glob("*.jsonl.gz")):
    try:
        f = gzip.open(p, "rt")
        for line in f:
            try:
                j = json.loads(line)
            except ValueError:
                continue
            rows = idx.get(j["a"])
            if not rows or not j["p"]:
                continue
            seen += 1
            tt = np.array([x[0] for x in j["p"]], dtype=np.float64)
            pp = np.array([x[1] for x in j["p"]], dtype=np.float64)
            b = j["b"]
            for i in rows:
                if not (j["lo"] <= t_d[i] - 300 and t_d[i] + 1800 <= j["hi"]) and not np.isnan(cols["mid_d"][i]):
                    continue

                def after(t, within):
                    k = np.searchsorted(tt, t)
                    return pp[k] if k < len(tt) and tt[k] - t <= within else np.nan

                def before(t, within):
                    k = np.searchsorted(tt, t, side="right") - 1
                    return pp[k] if k >= 0 and t - tt[k] <= within else np.nan

                w = 900 if b <= 300 else 5400
                v = after(t_d[i], w)
                if np.isnan(v):
                    continue
                cols["mid_d"][i] = v
                cols["bucket"][i] = b
                cols["mid_pre"][i] = before(t_fill[i], w)
                cols["mid_30"][i] = after(t_d[i] + 1800, w)
                cols["mid_60"][i] = after(t_d[i] + 3600, w)
                cols["mid_240"][i] = after(t_d[i] + 4 * 3600, w)
                if t_ls[i] == t_ls[i]:
                    cols["exit_mid"][i] = after(t_ls[i] + 300, w)
    except EOFError:
        pass
for k, v in cols.items():
    ev[k] = v
del cols, idx
ev = ev[np.isfinite(ev.mid_d.values)].reset_index(drop=True)      # only signals a follower could have priced
n = len(ev)
t_d = ev.t_d.values
log(f"{seen} price spans read; {np.isfinite(ev.mid_d).sum()} signals have a midpoint when the follower acts "
    f"({(ev.bucket == 300).sum()} at 5-minute, {(ev.bucket == 3600).sum()} at hourly resolution)")

# ── the real tape where we have it (thin markets) ──
tape_px = np.full(n, np.nan)
tape_dt = np.full(n, np.nan)
td = DATA / "tapes"
if td.exists():
    by_cid = defaultdict(list)
    for i, c in enumerate(ev.cid.values):
        by_cid[c].append(i)
    k = 0
    for c, rows in by_cid.items():
        pth = td / f"{c}.json.gz"
        if not pth.exists():
            continue
        try:
            with gzip.open(pth, "rt") as f:
                tape = json.load(f)
        except Exception:
            continue
        if not tape.get("complete_after"):
            continue
        trades = sorted(tape["trades"], key=lambda x: x[0])
        tt = np.array([x[0] for x in trades])
        for i in rows:
            if t_d[i] < tape["complete_after"]:
                continue
            want = ev.oi.values[i]
            j0 = np.searchsorted(tt, t_d[i])
            for x in trades[j0: j0 + 400]:
                if x[0] - t_d[i] > 7200:
                    break
                # a taker buying our outcome at p, or selling the other one at q (our ask ≤ 1 − q)
                if x[1] == want and x[2] == 1:
                    tape_px[i], tape_dt[i] = x[4], x[0] - t_d[i]
                    break
                if x[1] == 1 - want and x[2] == -1:
                    tape_px[i], tape_dt[i] = 1 - x[4], x[0] - t_d[i]
                    break
            k += 1
    log(f"tape prices for {np.isfinite(tape_px).sum()} of {k} signals in markets with a complete tape")
ev["tape_px"] = tape_px
ev["tape_dt"] = tape_dt

# ── the follower's price ──
fm = FillModel(DATA / "fill_model.json")
hours = ((ev.end - ev.t_d) / 3600).clip(lower=0).fillna(24 * 30).values
in_game = ev.f_in_game.values > 0.5
cats = ev.cat.values
mid = ev.mid_d.values
cost = np.array([fm.cost(c, m, h, g) if m == m else np.nan for c, m, h, g in zip(cats, mid, hours, in_game)])
cost75 = np.array([fm.cost(c, m, h, g, q="p75") if m == m else np.nan for c, m, h, g in zip(cats, mid, hours, in_game)])
ev["cost"] = cost
ev["cost75"] = cost75
p_model = mid + cost
ev["p_f"] = np.where(np.isfinite(tape_px), np.maximum(tape_px, mid - 0.0), p_model)
ev["p_f"] = ev.p_f.clip(0.01, 0.99)
ev["p_f75"] = np.where(np.isfinite(tape_px), ev.p_f, (mid + cost75)).clip(0.01, 0.99)
fee = ev.fee_rate.values


def ret_at(p):
    return ev.final.values / p - 1.0 - fee * (1.0 - p)


ev["ret"] = ret_at(ev.p_f.values)
ev["ret75"] = ret_at(ev.p_f75.values)
ev["ret_leader"] = ret_at(np.clip(ev.leader_px.values, 0.01, 0.99))
ev["ret_pre"] = ret_at(np.clip(ev.mid_pre.values + cost, 0.01, 0.99))
for k in ("30", "60", "240"):
    ev[f"ret_{k}"] = ret_at(np.clip(ev[f"mid_{k}"].values + cost, 0.01, 0.99))
# exit on the leader's sale: sell at the mid minus the same cost, else hold to the end
ex = ev.exit_mid.values - cost
ev["ret_exit"] = np.where(np.isfinite(ex) & (ev.t_lsell.values < ev.resolved_at.values),
                          np.clip(ex, 0.0, 1.0) / ev.p_f.values - 1.0 - fee * (1 - ev.p_f.values) - fee * np.clip(ex, 0, 1) * (1 - np.clip(ex, 0, 1)) / ev.p_f.values,
                          ev.ret.values)
# the price-dependent inputs, with the bot's own function (features.price_terms), row by row
out = {k: np.empty(n, dtype=np.float32) for k in ("logit_p", "price", "drift")}
for i, (p, lp) in enumerate(zip(ev.p_f.values, ev.f_leader_price.values)):
    x = F.price_terms(float(p), float(lp))
    out["logit_p"][i], out["price"][i], out["drift"][i] = x["logit_p"], x["price"], x["drift"]
for k, v in out.items():
    ev[f"f_{k}"] = v
del out

# ── boards, point in time ──
wd = pd.read_parquet(DATA / "wallet_days.parquet")
wd["rank30"] = wd.groupby("day").pnl30.rank(ascending=False, method="first")
wd["rank1"] = wd.groupby("day").pnl1.rank(ascending=False, method="first")
ev["day"] = (ev.t_d // 86400).astype(int)
look = wd.set_index(["wallet", "day"])[["rank30", "rank1", "pnl30"]]
key = pd.MultiIndex.from_arrays([ev.wallet.values, ev.day.values])          # the board as of the start of the day
got = look.reindex(key)
ev["rank30"] = got.rank30.values
ev["rank1"] = got.rank1.values
ev["in_pool"] = (ev.rank30 <= 200).astype(np.int8)
ev["in_pool1000"] = (ev.rank30 <= 1000).astype(np.int8)
ev["in_day25"] = ((ev.rank1 <= 25) & (got.pnl30.values > 0)).astype(np.int8)

# crowd: other pool wallets signalling the same outcome in the 48 hours before
ev = ev.sort_values("t_d").reset_index(drop=True)
crowd = np.zeros(len(ev), dtype=np.int16)
for a, g in ev[ev.in_pool == 1].groupby("asset"):
    tt = g.t_d.values
    ww = g.wallet.values
    for k, i in enumerate(g.index.values):
        j0 = np.searchsorted(tt, tt[k] - 48 * 3600)
        crowd[i] = len(set(ww[j0:k]) - {ww[k]})
ev["crowd"] = crowd
keep = ev
keep.to_parquet(DATA / "panel.parquet", index=False)
log(f"panel: {len(keep)} signals with a follower price ({keep.wallet.nunique()} wallets, {keep.cid.nunique()} markets), "
    f"{int(keep.in_pool.sum())} from the rebuilt monthly top 200; "
    f"{pd.to_datetime(keep.t_d.min(), unit='s'):%Y-%m-%d} → {pd.to_datetime(keep.t_d.max(), unit='s'):%Y-%m-%d}")
