"""Round 3 dataset: one row per copyable leader signal, with the features the bot would have computed.

    python build3.py DATA_DIR [workers]

A *signal* is the moment a wallet's net buy of an outcome over the bot's 48-hour
lookback (money into this outcome minus money into the other one, so a market
maker buying both sides stays flat) first reaches MIN_USD, at most once per
outcome per 72 hours (the bot's cooldown). The follower sees it LATENCY seconds
later. Every feature is computed with polybot/features.py from what existed at
that moment: the leader's fills up to then, and their record as of the bot's
last 6-hourly pack refresh, from positions whose market had resolved by then.
The label is the outcome's final price (1 or 0) from the market data.

Writes DATA_DIR/events/part-*.parquet and DATA_DIR/wallet_days.parquet (each wallet's
30-day P&L at resolution per day, to rebuild the monthly board point in time).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import sys
import time
from collections import defaultdict, deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polybot.categories import category  # noqa: E402
from polybot import features as F  # noqa: E402
from common3 import LOOKBACK, MIN_USD, M, market_final, market_resolved_at, positions_of, signal_context, ts  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
U = dt.timezone.utc
FILLS_SINCE = dt.datetime(2026, 6, 1, tzinfo=U).timestamp()
COOLDOWN = 72 * 3600
LATENCY = 300            # seconds from the leader's fill to the follower's order (5-minute polls at best)
REFRESH = 6 * 3600       # the bot re-measures its pack every 6 hours
MARKETS: dict = {}


def load_markets():
    out = {}
    path = DATA / "markets" / "markets.jsonl.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            for line in f:
                m = json.loads(line)
                if not m.get("missing"):
                    out[m["cid"]] = M(m)
    return out


def old_record(closed_rows, as_of):
    """The record the live bot computes today (pack.measure): CLOSED positions only, newest 200, a win = realized
    profit. Returns the model.json (round-2) inputs that come from the record."""
    rows = [r for r in closed_rows if r.get("_last") is not None and r["_last"] < as_of][:200]
    pnls, costs, holds, prices = [], [], [], []
    cat_n: dict[str, int] = defaultdict(int)
    cat_w: dict[str, int] = defaultdict(int)
    for r in rows:
        cost = r["size"] * r["avg"]
        if cost < 1:
            continue
        x = r["rpnl"]
        pnls.append(x)
        costs.append(cost)
        prices.append(r["avg"])
        if r.get("_first") is not None:
            holds.append((r["_last"] - r["_first"]) / 3600)
        cat_n[r["_cat"]] += 1
        cat_w[r["_cat"]] += 1 if x > 0 else 0
    n = len(pnls)
    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)
    gw = sum(x for x in pnls if x > 0)
    gl = -sum(x for x in pnls if x < 0)
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    return {"n": n, "wr": wins / (wins + losses) if (wins + losses) else 0.0, "pf": pf,
            "roc": sum(pnls) / sum(costs) if costs else 0.0, "median_cost": float(np.median(costs)) if costs else None,
            "median_hold": float(np.median(holds)) if holds else None, "median_price": float(np.median(prices)) if prices else None,
            "cat_n": dict(cat_n), "cat_w": dict(cat_w)}


def old_vector(o, cat, stance, taker):
    """model.json round-2 features from an old-style record (polybot/model.py Model.vector), minus price, days and crowd."""
    n = o["n"]
    cn, cw = o["cat_n"].get(cat, 0), o["cat_w"].get(cat, 0)
    conv = min(2.0, max(0.5, abs(stance) / o["median_cost"])) if o.get("median_cost") else 1.0
    return {"prior_pf": min(o["pf"], 5.0) if n else 1.0, "prior_wr": o["wr"] if n else 0.5, "prior_n_log": math.log1p(n),
            "prior_roc": max(-1.0, min(o["roc"] if n else 0.0, 1.0)), "conviction_log": math.log(max(0.05, min(conv, 20.0))),
            "cat_share": cn / n if n else 0.3, "cat_wr": cw / cn if cn else 0.5, "cat_n_log": math.log1p(cn),
            "taker_share": taker if taker is not None else 0.5, "prior_hold_log": math.log1p(o["median_hold"] or 24.0),
            "prior_price": o["median_price"] if o["median_price"] is not None else 0.5,
            "prior_sports": o["cat_n"].get("sports", 0) / n if n else 0.5}


def taker_share_at(w, t):
    """Share of the wallet's fills in the 7 days before t that took liquidity, the way pack.py measures it: the newest
    500 fills against the newest 500 taker fills (one page of each)."""
    fills = w.get("_fill_ts")
    tk = w.get("_taker_ts")
    if fills is None or tk is None:
        return None
    a, b = np.searchsorted(fills, [t - 7 * 86400, t])
    if b - a == 0:
        return None
    c0, c1 = np.searchsorted(tk, [t - 7 * 86400, t])
    return min(1.0, min(500, c1 - c0) / min(500, b - a))


def wallet_events(path: Path):
    try:
        with gzip.open(path, "rt") as f:
            w = json.load(f)
    except Exception:
        return [], []
    wallet = w["wallet"]
    info = w.get("info", {})
    fills = sorted(w.get("fills") or [], key=lambda x: x[0])
    trunc = w.get("truncated", {})
    coverage = (fills[0][0] if trunc.get("fills") and fills else FILLS_SINCE) + LOOKBACK
    w["_fill_ts"] = np.array([x[0] for x in fills], dtype=np.int64)
    w["_taker_ts"] = np.array(sorted(x[0] for x in (w.get("taker") or [])), dtype=np.int64) if w.get("taker") is not None and not trunc.get("taker") else None
    joined = ts(w.get("joined"))
    pos = positions_of(w, MARKETS)
    rec_cache: dict[int, dict] = {}
    old_cache: dict[int, dict] = {}
    closed_rows = []
    for r in w.get("closed", []):
        r = dict(r)
        r["_last"], r["_first"] = ts(r.get("last")), ts(r.get("first"))
        r["_cat"] = category(r.get("title"), r.get("ev") or r.get("slug"))
        closed_rows.append(r)
    closed_rows.sort(key=lambda r: -(r["_last"] or 0))

    def old_at(t):
        slot = int(t // REFRESH)
        if slot not in old_cache:
            old_cache[slot] = old_record(closed_rows, slot * REFRESH)
        return old_cache[slot]

    def rec_at(t):
        slot = int(t // REFRESH)
        if slot not in rec_cache:
            rec_cache[slot] = F.record(pos, slot * REFRESH)
        return rec_cache[slot]

    # daily 30-day P&L at resolution, for the point-in-time monthly board
    days = []
    done = sorted(((F.when(p), p.cost * (p.final / p.avg - 1.0)) for p in pos if F.usable(p) and F.when(p) is not None), key=lambda x: x[0])
    if done:
        tt = np.array([x[0] for x in done])
        cum = np.cumsum([x[1] for x in done])
        for d in range(int(dt.datetime(2026, 6, 1, tzinfo=U).timestamp() // 86400), int(time.time() // 86400) + 1):
            t1 = d * 86400.0
            i1 = np.searchsorted(tt, t1)
            i0 = np.searchsorted(tt, t1 - 30 * 86400)
            if i1 > i0:
                j0 = np.searchsorted(tt, t1 - 86400)
                pnl1 = float(cum[i1 - 1] - (cum[j0 - 1] if j0 > 0 else 0.0)) if i1 > j0 else 0.0
                days.append((wallet, d, float(cum[i1 - 1] - (cum[i0 - 1] if i0 > 0 else 0.0)), int(i1 - i0), pnl1))

    by_cid: dict[str, list] = defaultdict(list)
    for x in fills:
        by_cid[x[2]].append(x)
    events = []
    for cid, fl in by_cid.items():
        m = MARKETS.get(cid)
        win: deque = deque()
        net: dict[str, float] = defaultdict(float)       # asset → net USD in the window
        last_event: dict[str, float] = {}
        for j, x in enumerate(fl):
            t, asset, side, size, price = x[0], x[1], x[3], x[4], x[5]
            win.append(x)
            net[asset] += side * size * price
            while win and win[0][0] <= t - LOOKBACK:
                y = win.popleft()
                net[y[1]] -= y[3] * y[4] * y[5]
            if side != 1 or t < coverage:
                continue
            total = sum(net.values())
            stance = 2 * net[asset] - total             # this outcome minus every other outcome of the market
            if stance < MIN_USD or (asset in last_event and t - last_event[asset] < COOLDOWN):
                continue
            last_event[asset] = t
            td = t + LATENCY
            # what the bot sees at td: the leader's fills in this market over the lookback (features.leader_stats)
            seen = [(y[0], y[1], y[3], y[4], y[5]) for y in fl[max(0, j - 2000): j + 200] if td - LOOKBACK < y[0] <= td]
            sh_b = sum(y[3] for y in seen if y[1] == asset and y[2] == 1)
            rec = rec_at(td)
            ts_td = taker_share_at(w, td)
            vec, ls, cat = signal_context(rec, seen, asset, td, m, x[9], x[7], x[8], x[10], joined, ts_td, price)
            st_td, lpx = ls["stance"], vec["leader_price"]
            # when the leader sold half of what they had bought in the window (the bot's "follow sells" moment)
            t_ls = ls_px = None
            if sh_b > 0:
                sold = usd_s = 0.0
                for y in fl[j + 1:]:
                    if y[0] <= td or y[1] != asset or y[3] != -1:
                        continue
                    sold += y[4]
                    usd_s += y[4] * y[5]
                    if sold >= 0.5 * sh_b:
                        t_ls, ls_px = y[0], usd_s / sold
                        break
            oi = x[6]
            final = market_final(m, oi)
            title, slug, ev = x[9], x[7], x[8]
            end = m.end if m else None
            ov = old_vector(old_at(td), cat, st_td, ts_td)
            events.append({
                "wallet": wallet, "cid": cid, "asset": asset, "oi": oi if oi in (0, 1) else -1, "t_fill": t, "t_d": td,
                "final": final if final is not None else np.nan, "resolved_at": market_resolved_at(m) if final is not None else np.nan,
                "end": end if end else np.nan, "ev": (m.ev if m else None) or ev or slug, "cat": cat, "title": title[:80],
                "slug": slug, "outcome": x[10], "fee_rate": m.fee if m else 0.0,
                "has_market": m is not None, "nr": m.nr if m else False,
                "leader_px": lpx, "fill_px": price, "stance": st_td, "rec_n": rec.get("n", 0),
                "t_lsell": t_ls if t_ls else np.nan, "lsell_px": ls_px if ls_px else np.nan, "n_window": len(seen),
                "control": bool(info.get("control")), "boards": len(info.get("boards", [])),
                "pnl_boards": sum(1 for b in info.get("boards", []) if b[2] == "PNL"),
                **{f"f_{k}": v for k, v in vec.items()},
                **{f"o_{k}": v for k, v in ov.items()}, "o_end_date": (end // 86400) * 86400 if end else np.nan,
            })
    return events, days


def main():
    global MARKETS
    t0 = time.time()
    MARKETS = load_markets()     # set before the pool forks, so workers share it instead of each getting a pickled copy
    print(f"{len(MARKETS)} markets loaded in {time.time() - t0:.0f}s", flush=True)
    paths = sorted((DATA / "wallets").glob("*/*.json.gz"))
    print(f"{len(paths)} wallets", flush=True)
    out = DATA / "events"
    out.mkdir(exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()
    buf, days = [], []
    n = n_ev = part = 0

    def flush():
        nonlocal buf, part
        if not buf:
            return
        df = pd.DataFrame(buf)
        for c in df.columns:
            if df[c].dtype == np.float64 and c.startswith("f_"):
                df[c] = df[c].astype(np.float32)
        df.to_parquet(out / f"part-{part:04d}.parquet", index=False)
        part += 1
        buf = []

    with Pool(WORKERS) as pool:
        for ev, dd in pool.imap_unordered(wallet_events, paths, chunksize=4):
            buf += ev
            n_ev += len(ev)
            days += dd
            n += 1
            if len(buf) > 40000:
                flush()
            if n % 500 == 0:
                print(f"  {n}/{len(paths)} wallets, {n_ev} signals, {time.time() - t0:.0f}s", flush=True)
    flush()
    pd.DataFrame(days, columns=["wallet", "day", "pnl30", "n30", "pnl1"]).to_parquet(DATA / "wallet_days.parquet", index=False)
    print(f"{n_ev} signals from {n} wallets in {part} parts; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
