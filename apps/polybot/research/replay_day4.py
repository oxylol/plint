"""Replay the live `day4` setup on past weeks from its leaders' complete fills.

    python replay_day4.py DATA_DIR GROUP.json [from YYYY-MM-DD] [label] [noinplay]

The bot's rules, on its own 5-minute clock: every leader's fills over the last 24 hours (seen a minute late); a
leader's stance on an outcome is money into it minus money into the other outcome; agree at ≥ $25, against at ≤ −$25;
a buy signal needs 2+ agreeing and agree − against ≥ 4; strongest signals first, at most 5 buys a cycle, 72 hours
between trades of an outcome, 20 open, $50 (ten $5 buys) per rolling 24 hours, ≥ 1 hour to the market's end, the $5
buy priced 5–90¢ and at most 5¢ above the agreeing leaders' average; sell when 4+ more leaders net-sell than buy
(follow sells), else hold to resolution. The price is the 5-minute midpoint plus the book's cost for $5 (sells: minus),
taker fee on both. Fetches what it needs (market data, price paths) and caches it under DATA_DIR/group_cache.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from polymarket import PublicClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fillmodel3 import FillModel  # noqa: E402
from polybot.categories import category  # noqa: E402

DATA = Path(sys.argv[1])
GROUP = json.loads(Path(sys.argv[2]).read_text())
FROM = dt.datetime.fromisoformat(sys.argv[3] if len(sys.argv) > 3 else "2026-09-15").replace(tzinfo=dt.timezone.utc).timestamp()
LABEL = sys.argv[4] if len(sys.argv) > 4 else "group"
NO_IN_PLAY = len(sys.argv) > 5 and sys.argv[5] == "noinplay"   # variant: skip games that have started
U = dt.timezone.utc
POLL, LAG, LOOK = 300, 60, 24 * 3600
MIN_USD, MIN_AGREE, STAKE = 25.0, 4, 5.0
CACHE = DATA / "group_cache"
CACHE.mkdir(exist_ok=True)
c = PublicClient()
rng = np.random.default_rng(9)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def ts(s):
    if not s:
        return None
    d = dt.datetime.fromisoformat(s)
    return (d if d.tzinfo else d.replace(tzinfo=U)).timestamp()


# ── the leaders' fills ───────────────────────────────────────
fills = []
for w in GROUP:
    p = DATA / "group" / f"{w}.json.gz"
    if not p.exists():
        continue
    d = json.load(gzip.open(p, "rt"))
    for r in d["fills"]:
        if r[0] >= FROM - LOOK - 3600:
            fills.append((r[0], w, r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[9]))
log(f"{len(fills)} fills from {len({f[1] for f in fills})} of {len(GROUP)} wallets")
by_cid = defaultdict(list)
for f in fills:
    by_cid[f[3]].append(f)
by_cid = {k: sorted(v) for k, v in by_cid.items() if len({x[1] for x in v}) >= MIN_AGREE}
log(f"{len(by_cid)} markets with {MIN_AGREE}+ of these wallets trading")

# ── signal states at every poll ──────────────────────────────
states = []        # (t, cid, asset, side, score, avg_price, net, title, slug, ev)
for cid, fl in by_cid.items():
    t_arr = np.array([f[0] for f in fl], dtype=float)
    wallets = sorted({f[1] for f in fl})
    assets = sorted({f[2] for f in fl})
    wi = {w: i for i, w in enumerate(wallets)}
    ai = {a: i for i, a in enumerate(assets)}
    usd = np.zeros((len(fl), len(wallets), len(assets)))
    bu = np.zeros_like(usd)   # buy usd
    bs = np.zeros_like(usd)   # buy shares
    for k, f in enumerate(fl):
        usd[k, wi[f[1]], ai[f[2]]] = f[4] * f[5] * f[6]
        if f[4] == 1:
            bu[k, wi[f[1]], ai[f[2]]] = f[5] * f[6]
            bs[k, wi[f[1]], ai[f[2]]] = f[5]
    cu = np.vstack([np.zeros((1, len(wallets), len(assets))), np.cumsum(usd, 0)])
    cbu = np.vstack([np.zeros((1, len(wallets), len(assets))), np.cumsum(bu, 0)])
    cbs = np.vstack([np.zeros((1, len(wallets), len(assets))), np.cumsum(bs, 0)])
    start = max(FROM, (t_arr[0] // POLL + 1) * POLL)
    polls = np.arange(start, t_arr[-1] + LOOK + POLL, POLL)
    if len(polls) == 0:
        continue
    hi = np.searchsorted(t_arr, polls - LAG, side="right")
    lo = np.searchsorted(t_arr, polls - LAG - LOOK, side="right")
    net = cu[hi] - cu[lo]                       # polls × wallets × assets
    tot = net.sum(2, keepdims=True)
    stance = 2 * net - tot                      # this outcome minus the other(s)
    wbu = cbu[hi] - cbu[lo]
    wbs = cbs[hi] - cbs[lo]
    meta = fl[-1]
    for a, j in ai.items():
        ag = stance[:, :, j] >= MIN_USD
        ng = stance[:, :, j] <= -MIN_USD
        na, nn = ag.sum(1), ng.sum(1)
        buy = (na >= 2) & (na - nn >= MIN_AGREE)
        sell = (nn >= 2) & (nn - na >= MIN_AGREE)
        for k in np.where(buy | sell)[0]:
            if buy[k]:
                sh = (wbs[k, :, j] * ag[k]).sum()
                avg = (wbu[k, :, j] * ag[k]).sum() / sh if sh > 0 else None
                states.append((polls[k], cid, a, "BUY", int(na[k] - nn[k]), avg, float((stance[k, :, j] * ag[k]).sum()), meta[9], meta[7], meta[8]))
            else:
                states.append((polls[k], cid, a, "SELL", int(nn[k] - na[k]), None, 0.0, meta[9], meta[7], meta[8]))
states.sort(key=lambda s: (s[0], s[3] != "SELL", -s[4], -s[6]))
log(f"{len(states)} signal-polls ({sum(1 for s in states if s[3] == 'BUY')} buy, {sum(1 for s in states if s[3] == 'SELL')} sell) "
    f"on {len({s[2] for s in states})} outcomes")

# ── markets ──────────────────────────────────────────────────
need = {s[1] for s in states}
mk = {}
mp = DATA / "markets" / "markets.jsonl.gz"
with gzip.open(mp, "rt") as f:
    for line in f:
        j = line.find('"cid": "')
        if j >= 0 and line[j + 8: j + 74] in need:
            m = json.loads(line)
            if not m.get("missing"):
                mk[m["cid"]] = m
cache_m = CACHE / "markets.json"
extra = json.loads(cache_m.read_text()) if cache_m.exists() else {}
mk.update(extra)
missing = [x for x in need if x not in mk]
for k in range(0, len(missing), 40):
    ids = missing[k:k + 40]
    for closed in (True, None):
        try:
            ms = c.list_markets(condition_ids=ids, page_size=len(ids), **({"closed": True} if closed else {})).first_page().items
        except Exception:
            continue
        for m in ms:
            yes, no = m.outcomes.yes, m.outcomes.no
            tr = m.trading
            fs = tr.fee_schedule if tr is not None else None
            mk[str(m.condition_id)] = extra[str(m.condition_id)] = {
                "cid": str(m.condition_id), "closed": bool(m.state.closed), "end": m.state.end_date.isoformat() if m.state.end_date else None,
                "closed_time": m.state.closed_time.isoformat() if m.state.closed_time else None,
                "prices": [float(yes.price) if yes.price is not None else None, float(no.price) if no.price is not None else None],
                "tokens": [str(yes.token_id or ""), str(no.token_id or "")], "fee_on": bool(tr.fees_enabled) if tr else False,
                "fee_rate": float(fs.rate or 0) if fs else 0.0,
                "game_start": m.sports.game_start_time.isoformat() if (m.sports and m.sports.game_start_time) else None}
cache_m.write_text(json.dumps(extra))
log(f"market data for {sum(1 for x in need if x in mk)} of {len(need)} markets")


def outcome_index(m, asset):
    toks = m.get("tokens") or []
    pos = m.get("pos_ids") or []
    for i in (0, 1):
        if (i < len(toks) and toks[i] == asset) or (i < len(pos) and pos[i] == asset):
            return i
    return None


# ── price paths: 5-minute midpoints from the first buy signal to resolution ──
first_sig = {}
for s in states:
    if s[3] == "BUY":
        first_sig.setdefault(s[2], (s[0], s[1]))
cache_p = CACHE / "prices.jsonl"
paths = {}
if cache_p.exists():
    for line in cache_p.open():
        j = json.loads(line)
        paths.setdefault(j["a"], []).extend(j["p"])
todo = [a for a in first_sig if a not in paths]
log(f"price paths: {len(paths)} cached, {len(todo)} to fetch")
with cache_p.open("a") as out:
    for n, a in enumerate(todo):
        t0, cid = first_sig[a]
        m = mk.get(cid, {})
        end = ts(m.get("closed_time")) or ts(m.get("end")) or t0 + 3 * 86400
        lo_, hi_ = t0 - 900, min(max(end, t0 + 3600) + 3600, t0 + 14.8 * 86400)
        pts = []
        for i in range(4):
            try:
                pts = [[int(p.timestamp.timestamp()), float(p.price)] for p in
                       c.list_price_history(asset_id=a, start=int(lo_), end=int(hi_), bucket_seconds=300).first_page().items]
                break
            except Exception:
                time.sleep(2 * (i + 1))
        paths[a] = pts
        out.write(json.dumps({"a": a, "p": pts}) + "\n")
        if n % 200 == 0:
            log(f"  {n}/{len(todo)}")
        time.sleep(0.05)
P = {a: (np.array([x[0] for x in sorted(v)], dtype=float), np.array([x[1] for x in sorted(v)])) for a, v in paths.items() if v}


def mid_at(a, t):
    if a not in P:
        return None
    tt, pp = P[a]
    k = np.searchsorted(tt, t, side="right") - 1
    if k >= 0 and t - tt[k] <= 600:
        return float(pp[k])
    k += 1
    if k < len(tt) and tt[k] - t <= 300:
        return float(pp[k])
    return None


# ── the book ─────────────────────────────────────────────────
fm = FillModel(DATA / "fill_model.json")
open_pos = {}      # asset → dict
acted = {}
buys = []
trades = []
cycle = None
cycle_n = 0
for s in states:
    t, cid, a, side, score, avg, netv, title, slug, ev = s
    m = mk.get(cid)
    # settle what has resolved by now
    for aa in list(open_pos):
        p = open_pos[aa]
        if p["res"] is not None and p["res"] <= t:
            pay = p["shares"] * p["final"]
            trades.append({**p, "exit_t": p["res"], "how": "resolved", "ret": pay / p["cost"] - 1})
            acted[aa] = p["res"]
            del open_pos[aa]
    if m is None:
        continue
    oi = outcome_index(m, a)
    final = (m.get("prices") or [None, None])[oi] if oi is not None and m.get("closed") else None
    final = 1.0 if (final is not None and final >= 0.99) else 0.0 if (final is not None and final <= 0.01) else None
    res = ts(m.get("closed_time")) if final is not None else None
    end = ts(m.get("end"))
    gs = ts(m.get("game_start"))
    cat = category(title, ev or slug)
    rate = float(m.get("fee_rate") or 0) if m.get("fee_on") else 0.0
    if t // POLL != cycle:
        cycle, cycle_n = t // POLL, 0
    if side == "SELL":
        if a in open_pos:
            mid = mid_at(a, t)
            if mid is None:
                continue
            px = max(0.001, mid - fm.cost(cat, mid, max(0, (end - t) / 3600) if end else 720, gs is not None and t >= gs))
            p = open_pos.pop(a)
            proceeds = p["shares"] * px - p["shares"] * rate * px * (1 - px)
            trades.append({**p, "exit_t": t, "how": "leaders sold", "ret": proceeds / p["cost"] - 1})
            acted[a] = t
        continue
    if a in open_pos or cycle_n >= 5 or (a in acted and t - acted[a] < 72 * 3600) or len(open_pos) >= 20:
        continue
    buys[:] = [b for b in buys if b > t - 86400]
    if (len(buys) + 1) * STAKE > 50:
        continue
    if final is None and m.get("closed"):
        continue
    if (res is not None and res <= t) or (end is not None and end - t < 3600):
        continue
    if NO_IN_PLAY and gs is not None and t >= gs:
        continue
    mid = mid_at(a, t)
    if mid is None:
        continue
    est = mid + fm.cost(cat, mid, max(0, (end - t) / 3600) if end else 720, gs is not None and t >= gs)
    if est > 0.90 or est < 0.05 or (avg is not None and est > avg + 0.05):
        continue
    if final is None:
        continue          # unresolved by now: outcome unknown, cannot score (counted below)
    fee = (STAKE / est) * rate * est * (1 - est)
    shares = (STAKE - fee) / est
    open_pos[a] = {"t": t, "asset": a, "cid": cid, "title": title, "cat": cat, "price": est, "leaders": avg, "score": score, "shares": shares,
                   "cost": STAKE, "final": final, "res": res, "in_game": gs is not None and t >= gs}
    acted[a] = t
    buys.append(t)
    cycle_n += 1
for aa, p in open_pos.items():
    trades.append({**p, "exit_t": p["res"], "how": "resolved", "ret": p["shares"] * p["final"] / p["cost"] - 1})
trades.sort(key=lambda x: x["t"])
json.dump(trades, open(CACHE / f"trades_{LABEL}.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
log(f"{len(trades)} trades")


def show(label, tr):
    if len(tr) < 3:
        print(f"  {label:36s} {len(tr):4d} trades")
        return
    r = np.array([x["ret"] for x in tr])
    d = np.array([int(x["t"] // 86400) for x in tr])
    ud = np.unique(d)
    sums = np.array([r[d == x].sum() for x in ud])
    cnts = np.array([(d == x).sum() for x in ud])
    dr = rng.integers(0, len(ud), size=(4000, len(ud)))
    bm = sums[dr].sum(1) / cnts[dr].sum(1)
    span = max(1.0, (max(x["t"] for x in tr) - min(x["t"] for x in tr)) / 86400)
    print(f"  {label:36s} {len(r):4d} trades on {len(ud):2d} days · mean {100 * r.mean():+6.1f}% [90% CI {100 * np.quantile(bm, .05):+6.1f}, "
          f"{100 * np.quantile(bm, .95):+6.1f}] · won {100 * np.mean([x['final'] for x in tr]):.0f}% at {100 * np.mean([x['price'] for x in tr]):.0f}¢ · "
          f"${STAKE * r.sum():+.2f} ({len(r) / span:.1f} a day)")


live0 = dt.datetime(2026, 10, 28, tzinfo=U).timestamp()
print(f"\nday4 replay ({LABEL}: {len(GROUP)} wallets), from {dt.datetime.fromtimestamp(FROM, U):%b %d}")
show("all", trades)
show("before the live run (→ Oct 27)", [x for x in trades if x["t"] < live0])
show("live paper period (Oct 28 →)", [x for x in trades if x["t"] >= live0])
show("sports", [x for x in trades if x["cat"] == "sports"])
show("not sports", [x for x in trades if x["cat"] != "sports"])
show("bought in play", [x for x in trades if x["in_game"]])
show("bought before the game", [x for x in trades if not x["in_game"] and x["cat"] == "sports"])
show("paid 40–60¢", [x for x in trades if 0.4 <= x["price"] <= 0.6])
show("exited when leaders sold", [x for x in trades if x["how"] == "leaders sold"])
s0 = FROM
wk = 0
while s0 + wk * 7 * 86400 < time.time():
    show(f"week of {dt.datetime.fromtimestamp(s0 + wk * 7 * 86400, U):%b %d}", [x for x in trades if s0 + wk * 7 * 86400 <= x["t"] < s0 + (wk + 1) * 7 * 86400])
    wk += 1
