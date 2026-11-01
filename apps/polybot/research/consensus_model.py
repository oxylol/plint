"""A model on top of day4's consensus signals, judged against day4 itself on the same replay.

    python consensus_model.py DATA_DIR GROUP.json [label]

Every moment a consensus forms or strengthens among the group (2+ leaders net-buying an outcome over 24 hours,
agree − against ≥ 2, seen a minute late on the bot's 5-minute clock) is a candidate, with the price a $5 buy would
pay then (5-minute midpoint + book cost) and polybot/features.consensus_context's inputs. Models (anchored on that
price) give each candidate a win probability; the policy (threshold, smallest consensus, price band, before the
game only, at most 5¢ above the leaders) is chosen per week on the weeks before it, and the bot's book (10 buys a
day, 20 open, 5 a cycle, 72 hours per outcome) decides what is bought. day4's rule runs through the same book.
Families and policies are chosen on Sep 15 → Oct 19; Oct 20 → 31 is the holdout.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from polymarket import PublicClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common3 import consensus_row  # noqa: E402
from fillmodel3 import FillModel  # noqa: E402
from polybot import features as F  # noqa: E402
from polybot.categories import category  # noqa: E402

DATA = Path(sys.argv[1])
GROUP = json.loads(Path(sys.argv[2]).read_text())
LABEL = sys.argv[3] if len(sys.argv) > 3 else "copied29"
MIN_CAND = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--min-score=")), 2))   # smallest consensus kept as a candidate
NO_QUALITY = "--no-quality" in sys.argv
U = dt.timezone.utc
POLL, LAG, LOOK, MIN_USD = 300, 60, 24 * 3600, 25.0
MIN_AGREE_CAND = 2      # markets need at least this many of the group trading them
BOARD = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--board=")), 0))   # >0: only that day's top-N of the group count
BOARD_COL = next((a.split("=")[1] for a in sys.argv if a.startswith("--board-col=")), "pnl1")  # pnl1: the day before; pnl30: 30 days
TRAIN0 = dt.datetime(2026, 9, 2, tzinfo=U).timestamp()
TEST0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp()
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
LIVE0 = dt.datetime(2026, 10, 28, tzinfo=U).timestamp()
CACHE = DATA / "group_cache"
c = PublicClient()
rng = np.random.default_rng(13)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def ts(s):
    if not s:
        return None
    d = dt.datetime.fromisoformat(s)
    return (d if d.tzinfo else d.replace(tzinfo=U)).timestamp()


# ── fills → consensus states (score ≥ 2) ─────────────────────
# markets that 2+ of the group trade, written to shards on disk, then processed shard by shard (bounded memory)
NSH = 16
who = defaultdict(set)
for w in GROUP:
    p = DATA / "group" / f"{w}.json.gz"
    if p.exists():
        for r in json.load(gzip.open(p, "rt"))["fills"]:
            if r[0] >= TRAIN0 - LOOK - 3600:
                who[r[2]].add(w)
keep_cids = {c_ for c_, ws in who.items() if len(ws) >= MIN_AGREE_CAND}
del who
shard_dir = CACHE / f"shards_{LABEL}"
shard_dir.mkdir(exist_ok=True)
outs = [gzip.open(shard_dir / f"{k:02d}.jsonl.gz", "wt") for k in range(NSH)]
n_f = 0
for w in GROUP:
    p = DATA / "group" / f"{w}.json.gz"
    if p.exists():
        for r in json.load(gzip.open(p, "rt"))["fills"]:
            if r[0] >= TRAIN0 - LOOK - 3600 and r[2] in keep_cids:
                outs[int(r[2][2:10], 16) % NSH].write(json.dumps([r[0], w, r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[9]]) + "\n")
                n_f += 1
for o in outs:
    o.close()
log(f"{n_f} fills in {len(keep_cids)} markets with {MIN_AGREE_CAND}+ of the group, in {NSH} shards")


def shard_markets():
    for k in range(NSH):
        bc = defaultdict(list)
        for line in gzip.open(shard_dir / f"{k:02d}.jsonl.gz", "rt"):
            r = json.loads(line)
            bc[r[3]].append(tuple(r))
        for cid_, fl_ in bc.items():
            yield cid_, sorted(fl_)


ON_BOARD = None
if BOARD:
    # the day board, point in time: each day the BOARD wallets of the group with the best resolved P&L of the day before
    import pandas as pd  # noqa: E402
    wd = pd.read_parquet(DATA / "wallet_days.parquet", columns=["wallet", "day", BOARD_COL]).rename(columns={BOARD_COL: "pnl"})
    wd = wd[wd.wallet.isin(set(GROUP)) & (wd.pnl > 0)]
    wd["r"] = wd.groupby("day").pnl.rank(ascending=False, method="first")
    ON_BOARD = defaultdict(set)
    for w_, d_ in zip(wd.wallet[wd.r <= BOARD], wd.day[wd.r <= BOARD]):
        ON_BOARD[w_].add(int(d_))
    ON_BOARD = {w_: np.array(sorted(v)) for w_, v in ON_BOARD.items()}
    per_day = wd[wd.r <= BOARD].groupby("day").size()
    log(f"rebuilt board: top {BOARD} of {len(GROUP)} by {BOARD_COL} (resolved P&L before that day), {per_day.mean():.1f} wallets a day on average")

events = []      # candidates: (t, cid, asset, score, n_agree, n_against, avg, usd, title, slug, ev)
sells = defaultdict(list)    # asset → times when 4+ more leaders net-sell (day4's exit)
for cid, fl in shard_markets():
    t_arr = np.array([f[0] for f in fl], dtype=float)
    wallets = sorted({f[1] for f in fl})
    assets = sorted({f[2] for f in fl})
    wi = {w: i for i, w in enumerate(wallets)}
    ai = {a: i for i, a in enumerate(assets)}
    shp = (len(fl), len(wallets), len(assets))
    usd, bu, bs = np.zeros(shp, np.float32), np.zeros(shp, np.float32), np.zeros(shp, np.float32)
    for k, f in enumerate(fl):
        usd[k, wi[f[1]], ai[f[2]]] = f[4] * f[5] * f[6]
        if f[4] == 1:
            bu[k, wi[f[1]], ai[f[2]]] = f[5] * f[6]
            bs[k, wi[f[1]], ai[f[2]]] = f[5]
    z = np.zeros((1,) + shp[1:])
    cu, cbu, cbs = (np.vstack([z, np.cumsum(x, 0, dtype=np.float64)]) for x in (usd, bu, bs))
    del usd, bu, bs
    start = max(TRAIN0, (t_arr[0] // POLL + 1) * POLL)
    polls = np.arange(start, t_arr[-1] + LOOK + POLL, POLL)
    if not len(polls):
        continue
    hi = np.searchsorted(t_arr, polls - LAG, side="right")
    lo = np.searchsorted(t_arr, polls - LAG - LOOK, side="right")
    net = cu[hi] - cu[lo]
    stance = 2 * net - net.sum(2, keepdims=True)
    wbu, wbs = cbu[hi] - cbu[lo], cbs[hi] - cbs[lo]
    if ON_BOARD is not None:     # only leaders on that day's board count
        pd_ = (polls // 86400).astype(int)
        onb = np.column_stack([np.isin(pd_, ON_BOARD.get(w, np.array([], int))) for w in wallets])
        stance = np.where(onb[:, :, None], stance, 0.0)
    meta = fl[-1]
    for a, j in ai.items():
        ag, ng = stance[:, :, j] >= MIN_USD, stance[:, :, j] <= -MIN_USD
        na, nn = ag.sum(1), ng.sum(1)
        score = na - nn
        for k in np.where((nn >= 2) & (nn - na >= 4))[0]:
            sells[a].append(polls[k])
        best, prev_k = -1, -10
        for k in np.where((na >= 2) & (score >= MIN_CAND))[0]:
            if k != prev_k + 1:
                best = -1                      # a new episode
            prev_k = k
            if score[k] > best:                # the consensus formed or got stronger: a candidate
                best = score[k]
                sh = (wbs[k, :, j] * ag[k]).sum()
                events.append((polls[k], cid, a, int(score[k]), int(na[k]), int(nn[k]), (wbu[k, :, j] * ag[k]).sum() / sh if sh > 0 else None,
                               float((stance[k, :, j] * ag[k]).sum()), meta[9], meta[7], meta[8],
                               tuple(wallets[i] for i in np.where(ag[k])[0])))
events.sort(key=lambda e: e[0])
log(f"{len(events)} candidate moments on {len({e[2] for e in events})} outcomes "
    f"({sum(1 for e in events if e[3] >= 4)} at day4's level, score ≥ 4)")

# ── market data (cached by replay_day4.py, topped up here) ──
need = {e[1] for e in events}
mk = {}
with gzip.open(DATA / "markets" / "markets.jsonl.gz", "rt") as f:
    for line in f:
        j = line.find('"cid": "')
        if j >= 0 and line[j + 8: j + 74] in need:
            m = json.loads(line)
            if not m.get("missing"):
                mk[m["cid"]] = m
cache_m = CACHE / "markets.json"
extra = json.loads(cache_m.read_text()) if cache_m.exists() else {}
mk.update({k: v for k, v in extra.items() if k not in mk})
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
                "fee_rate": float(fs.rate or 0) if fs else 0.0, "sports_type": m.sports.sports_market_type if m.sports else None,
                "game_start": m.sports.game_start_time.isoformat() if (m.sports and m.sports.game_start_time) else None}
cache_m.write_text(json.dumps(extra))
log(f"market data for {sum(1 for x in need if x in mk)} of {len(need)} markets")


def oindex(m, a):
    for key in ("tokens", "pos_ids"):
        v = m.get(key) or []
        for i in (0, 1):
            if i < len(v) and v[i] == a:
                return i
    return None


# ── price paths ──────────────────────────────────────────────
first = {}
for e in events:
    first.setdefault(e[2], (e[0], e[1]))
cache_p = CACHE / "prices.jsonl"
paths = defaultdict(list)
if cache_p.exists():
    for line in cache_p.open():
        j = json.loads(line)
        paths[j["a"]].extend(j["p"])
todo = [a for a in first if a not in paths and first[a][1] in mk and mk[first[a][1]].get("closed")]
log(f"price paths: {len(paths)} cached, {len(todo)} to fetch")
import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
lock = threading.Lock()
nxt = [time.monotonic()]


def fetch(a):
    t0, cid = first[a]
    m = mk[cid]
    end = ts(m.get("closed_time")) or ts(m.get("end")) or t0 + 3 * 86400
    lo_, hi_ = t0 - 900, min(max(end, t0 + 3600) + 3600, t0 + 14.8 * 86400)
    for i in range(4):
        with lock:                       # ~18 requests a second, under the data API's 20
            w = max(time.monotonic(), nxt[0])
            nxt[0] = w + 1 / 18
        time.sleep(max(0.0, w - time.monotonic()))
        try:
            return a, [[int(p.timestamp.timestamp()), float(p.price)] for p in
                       c.list_price_history(asset_id=a, start=int(lo_), end=int(hi_), bucket_seconds=300).first_page().items]
        except Exception:
            time.sleep(2 * (i + 1))
    return a, []


with cache_p.open("a") as out, ThreadPoolExecutor(max_workers=8) as ex:
    for n, (a, pts) in enumerate(ex.map(fetch, todo)):
        paths[a] = pts
        out.write(json.dumps({"a": a, "p": pts}) + "\n")
        if n % 2000 == 0:
            out.flush()
            log(f"  {n}/{len(todo)}")
PP = {a: (np.array([x[0] for x in sorted(v)], float), np.array([x[1] for x in sorted(v)])) for a, v in paths.items() if v}


def mid_at(a, t):
    if a not in PP:
        return None
    tt, pp = PP[a]
    k = np.searchsorted(tt, t, side="right") - 1
    if k >= 0 and t - tt[k] <= 600:
        return float(pp[k])
    k += 1
    return float(pp[k]) if (k < len(tt) and tt[k] - t <= 300) else None


# ── the candidate table ──────────────────────────────────────
fm = FillModel(DATA / "fill_model.json")
rows = []
for (t, cid, a, score, na, nn, avg, usdv, title, slug, ev, agree_w) in events:
    m = mk.get(cid)
    if m is None or not m.get("closed"):
        continue
    oi = oindex(m, a)
    pr = (m.get("prices") or [None, None])[oi] if oi is not None else None
    final = F.final_from_price(pr)
    res = ts(m.get("closed_time"))
    end, gs = ts(m.get("end")), ts(m.get("game_start"))
    if final is None or res is None or res <= t or (end is not None and end - t < 3600):
        continue
    mid = mid_at(a, t)
    if mid is None:
        continue
    cat = category(title, ev or slug)
    est = mid + fm.cost(cat, mid, max(0, (end - t) / 3600) if end else 720, gs is not None and t >= gs)
    if not (0.02 <= est <= 0.98):
        continue
    x = consensus_row(score=score, n_agree=na, n_against=nn, avg=avg, agree_usd=usdv, price=est, end=end, game_start=gs, t=t,
                      slug=slug, ev=ev, title=title, sports_type=m.get("sports_type"))
    rate = float(m.get("fee_rate") or 0) if m.get("fee_on") else 0.0
    sl = [s for s in sells.get(a, []) if s > t]
    m15, m60 = mid_at(a, t - 900), mid_at(a, t - 3600)
    rows.append({"agree_w": agree_w, "mom15": (mid - m15) if m15 is not None else None, "mom60": (mid - m60) if m60 is not None else None,
                 "t": t, "asset": a, "cid": cid, "ev": ev or slug, "score": score, "avg": avg if avg else est, "est": est, "final": final,
                 "res": res, "fee": rate, "ret": final / est - 1 - rate * (1 - est), "in_play": gs is not None and t >= gs,
                 "sell_t": sl[0] if sl and sl[0] < res else None, "x": x, "title": title})
log(f"{len(rows)} priced, resolved candidates ({sum(1 for r in rows if r['t'] >= TEST0)} from Sep 15)")
# who agrees: each agreeing leader's resolved record as of the last 6-hourly refresh (features.record over CLOSED +
# REDEEMABLE, as the bot's pack measures it), averaged over the agreeing leaders (features.agree_quality)
from common3 import M as _M, positions_of  # noqa: E402
need_w = set() if NO_QUALITY else {w for r in rows for w in r["agree_w"]}
pos_by_w, cids = {}, set()
raw = {}
for w in need_w:
    pth = DATA / "wallets" / w[2:4] / f"{w}.json.gz"
    if pth.exists():
        raw[w] = json.load(gzip.open(pth, "rt"))
        for k in ("closed", "redeemable", "open"):
            cids.update(x["cid"] for x in raw[w].get(k, []))
mm = {}
with gzip.open(DATA / "markets" / "markets.jsonl.gz", "rt") as f:
    for line in f:
        j = line.find('"cid": "')
        if j >= 0 and line[j + 8: j + 74] in cids:
            m = json.loads(line)
            if not m.get("missing"):
                mm[m["cid"]] = _M(m)
for w, d in raw.items():
    pos_by_w[w] = positions_of(d, mm)
del raw, mm
rec_cache = {}


def rec_of(w, t):
    slot = int(t // (6 * 3600))
    key = (w, slot)
    if key not in rec_cache:
        rec_cache[key] = F.record(pos_by_w[w], slot * 6 * 3600) if w in pos_by_w else {"n": 0}
    return rec_cache[key]


for r in rows:
    r["x"].update(F.agree_quality([rec_of(w, r["t"]) for w in r["agree_w"]]))
    r["x"]["mom15"] = r["mom15"] if r["mom15"] is not None else 0.0
log(f"leader records for {len(pos_by_w)} of {len(need_w)} agreeing wallets")
BASE_FEATS = [k for k in rows[0]["x"] if not k.startswith("aq_") and k != "mom15"]
FEATS = list(rows[0]["x"])
X = np.array([[r["x"][k] for k in FEATS] for r in rows], float)
y = np.array([r["final"] for r in rows], float)
est = np.array([r["est"] for r in rows])
tt = np.array([r["t"] for r in rows])
res = np.array([r["res"] for r in rows])


def lgt(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class Anchored:
    """L2 logistic regression with the price's log-odds as offset (the shipped kind 'consensus')."""
    def __init__(self, lam, feats=None):
        self.lam = lam
        self.cols = [FEATS.index(f) for f in (feats or BASE_FEATS)]

    def fit(self, idx):
        Xi = X[idx][:, self.cols]
        self.mu, self.sd = Xi.mean(0), Xi.std(0) + 1e-6
        Z = np.hstack([(Xi - self.mu) / self.sd, np.ones((len(idx), 1))])
        off = lgt(est[idx])
        w = np.zeros(Z.shape[1])
        I = np.eye(Z.shape[1])
        I[-1, -1] = 0
        for _ in range(30):
            p = sig(Z @ w + off)
            g = Z.T @ (p - y[idx]) + self.lam * I @ w
            H = Z.T @ (Z * (p * (1 - p))[:, None]) + self.lam * I
            w -= np.linalg.solve(H, g)
        self.w = w
        return self

    def p(self, idx):
        Z = np.hstack([(X[idx][:, self.cols] - self.mu) / self.sd, np.ones((len(idx), 1))])
        return sig(Z @ self.w + lgt(est[idx]))


class Trees:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, idx):
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(objective="binary", verbose=-1, num_threads=3, **self.cfg)
        self.m.fit(X[idx], y[idx], init_score=lgt(est[idx]))
        return self

    def p(self, idx):
        return sig(self.m.predict(X[idx], raw_score=True) + lgt(est[idx]))


class Price:
    def fit(self, idx):
        return self

    def p(self, idx):
        return est[idx]


QUAL = [f for f in FEATS if f.startswith("aq_")]
FAMILIES = {"day4_rule": lambda: Price(), "anchored_l1000": lambda: Anchored(1000.0),
            "anchored_l3000": lambda: Anchored(3000.0), "anchored_l300": lambda: Anchored(300.0)}
if "--features" in sys.argv:
    FAMILIES.update({"l1000+quality": lambda: Anchored(1000.0, BASE_FEATS + QUAL), "l1000+momentum": lambda: Anchored(1000.0, BASE_FEATS + ["mom15"]),
                     "l1000+both": lambda: Anchored(1000.0, BASE_FEATS + QUAL + ["mom15"])})
SHIPPED_GRID = "--old-grid" in sys.argv

weeks = []
s = TEST0
while s < tt.max():
    weeks.append((s, s + 7 * 86400))
    s += 7 * 86400
P = {k: np.full(len(rows), np.nan) for k in FAMILIES}
for (w0, w1) in weeks:
    tr = np.where((tt < w0) & (res < w0))[0]
    te = np.where((tt >= w0) & (tt < w1))[0]
    if len(tr) < 300 or not len(te):
        continue
    for k, mkf in FAMILIES.items():
        P[k][te] = mkf().fit(tr).p(te)
log("weekly fits done: " + ", ".join(f"{dt.datetime.fromtimestamp(w0, U):%b %d}" for w0, _ in weeks))


# ── the bot's book ───────────────────────────────────────────
def book(sel, er, exit_on_sell=False):
    """Candidates in time order (strongest first within a 5-minute cycle) through day4's book."""
    order = sorted(sel, key=lambda i: (int(tt[i] // POLL), -er[i]))
    open_ = {}
    acted, buys, out = {}, [], []
    cyc, cn = None, 0
    for i in order:
        t = tt[i]
        for a in [a for a, v in open_.items() if v <= t]:
            del open_[a]
        if int(t // POLL) != cyc:
            cyc, cn = int(t // POLL), 0
        a = rows[i]["asset"]
        if a in open_ or cn >= 5 or len(open_) >= 20 or (a in acted and t - acted[a] < 72 * 3600):
            continue
        buys[:] = [b for b in buys if b > t - 86400]
        if len(buys) >= 10:
            continue
        r = rows[i]
        ret = r["ret"]
        close = r["res"]
        if exit_on_sell and r["sell_t"]:
            px = mid_at(a, r["sell_t"])
            if px is not None:
                px = max(0.001, px - 0.01)
                ret = px * (1 - r["fee"]) / r["est"] - 1 - r["fee"] * (1 - r["est"])
                close = r["sell_t"]
        open_[a] = close
        acted[a] = t
        buys.append(t)
        cn += 1
        out.append((i, ret, int(t // 86400)))
    return out


def summ(tr):
    if len(tr) < 3:
        return {"n": len(tr), "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "week": 0.0}
    r = np.array([x[1] for x in tr])
    d = np.array([x[2] for x in tr])
    ud = np.unique(d)
    sums = np.array([r[d == x].sum() for x in ud])
    cnts = np.array([(d == x).sum() for x in ud])
    dr = rng.integers(0, len(ud), size=(4000, len(ud)))
    bm = sums[dr].sum(1) / cnts[dr].sum(1)
    span = max(7.0, d.max() - d.min() + 1)
    return {"n": len(r), "mean": r.mean(), "lo": np.quantile(bm, .05), "hi": np.quantile(bm, .95), "week": 5 * 7 * r.sum() / span}


GRID = []
for thr in (-0.02, 0.0, 0.02, 0.05, 0.1):
    for smin in (2, 3, 4, 5, 6):
        for band in ((0.05, 0.90), (0.30, 0.70), (0.40, 0.65)):
            for pre in (False, True):
                for drift in (None, 0.05):
                    for kick in (None, 3.0, 12.0):     # buy pre-game only within this many hours of kick-off (in play unaffected)
                        GRID.append({"thr": thr, "smin": smin, "band": band, "pre": pre, "drift": drift, "kick": kick})
DAY4 = {"thr": -9.0, "smin": 4, "band": (0.05, 0.90), "pre": False, "drift": 0.05, "kick": None}
score_a = np.array([r["score"] for r in rows])
avg_a = np.array([r["avg"] for r in rows])
inplay = np.array([r["in_play"] for r in rows])
pre_h = np.array([np.expm1(r["x"]["pre_game_h_log"]) if r["x"]["pre_game_h_log"] > 0 else (0.0 if r["in_play"] else 999.0) for r in rows])


def candidates(k, pol, w0, w1):
    er = P[k] / est - 1 - np.array([r["fee"] for r in rows]) * (1 - est)
    ok = (tt >= w0) & (tt < w1) & np.isfinite(er) & (score_a >= pol["smin"]) & (est >= pol["band"][0]) & (est <= pol["band"][1])
    if k != "day4_rule":
        ok &= er >= pol["thr"]
    if pol["pre"]:
        ok &= ~inplay
    if pol["drift"] is not None:
        ok &= est <= avg_a + pol["drift"]
    if pol.get("kick") is not None:
        ok &= inplay | (pre_h <= pol["kick"])
    return list(np.where(ok)[0]), er


def walk(k):
    """Nested: each week's policy is the one that earned most in the book over the weeks before."""
    per = {}
    for wi_, (w0, w1) in enumerate(weeks):
        for gi, pol in enumerate(GRID):
            sel, er = candidates(k, pol, w0, w1)
            per[(wi_, gi)] = book(sel, er)
    picks, out = [], []
    for wi_ in range(len(weeks)):
        if wi_ < 2:
            continue
        best = max(range(len(GRID)), key=lambda gi: sum(x[1] for w in range(wi_) for x in per[(w, gi)]) if sum(len(per[(w, gi)]) for w in range(wi_)) >= 30 else -1e9)
        picks.append((wi_, best))
    st_sel = []
    for wi_, gi in picks:
        w0, w1 = weeks[wi_]
        sel, er = candidates(k, GRID[gi], w0, w1)
        st_sel += sel
    er_all = P[k] / est - 1 - np.array([r["fee"] for r in rows]) * (1 - est)
    out = book(st_sel, er_all)
    final = max(range(len(GRID)), key=lambda gi: sum(x[1] for w in range(len(weeks)) if weeks[w][0] < SELECT_END for x in per[(w, gi)]))
    return out, picks, GRID[final]


first_scored = weeks[2][0]
print(f"\n## day4's consensus, with and without a model (book: 10 buys a day, 20 open, 5 a cycle; scored from "
      f"{dt.datetime.fromtimestamp(first_scored, U):%b %d})\n")
res_all = {}
base = []
for (w0, w1) in weeks:
    if w0 < first_scored:
        continue
    sel, er = candidates("day4_rule", DAY4, w0, w1)
    base += sel
er0 = np.zeros(len(rows))
for label, tr in (("day4 rule (hold to the end)", book(base, er0)), ("day4 rule (follow sells)", book(base, er0, exit_on_sell=True))):
    sa, sb, s_ = summ([x for x in tr if tt[x[0]] < SELECT_END]), summ([x for x in tr if tt[x[0]] >= SELECT_END]), summ(tr)
    res_all[label] = tr
    print(f"{label:34s} {s_['n']:4d} trades {100 * s_['mean']:+6.1f}% [{100 * s_['lo']:+6.1f}, {100 * s_['hi']:+6.1f}] ${s_['week']:+7.2f}/wk | "
          f"select {sa['n']:4d} {100 * sa['mean']:+6.1f}% ${sa['week']:+6.2f}/wk | holdout {sb['n']:3d} {100 * sb['mean']:+6.1f}% [{100 * sb['lo']:+6.1f}, {100 * sb['hi']:+6.1f}] ${sb['week']:+6.2f}/wk")
for k in FAMILIES:
    tr, picks, final = walk(k)
    res_all[k] = tr
    sa, sb, s_ = summ([x for x in tr if tt[x[0]] < SELECT_END]), summ([x for x in tr if tt[x[0]] >= SELECT_END]), summ(tr)
    print(f"{k + ' (nested policy)':34s} {s_['n']:4d} trades {100 * s_['mean']:+6.1f}% [{100 * s_['lo']:+6.1f}, {100 * s_['hi']:+6.1f}] ${s_['week']:+7.2f}/wk | "
          f"select {sa['n']:4d} {100 * sa['mean']:+6.1f}% ${sa['week']:+6.2f}/wk | holdout {sb['n']:3d} {100 * sb['mean']:+6.1f}% [{100 * sb['lo']:+6.1f}, {100 * sb['hi']:+6.1f}] ${sb['week']:+6.2f}/wk"
          f"  final policy {final}")


def gap(a, b):
    days = np.arange(int(first_scored // 86400), int(tt.max() // 86400) + 1)
    da = np.array([sum(x[1] for x in a if x[2] == d) for d in days]) * 5
    db = np.array([sum(x[1] for x in b if x[2] == d) for d in days]) * 5
    dr = rng.integers(0, len(days), size=(10000, len(days)))
    g = 7 * (da[dr].mean(1) - db[dr].mean(1))
    return np.median(g), np.quantile(g, .05), np.quantile(g, .95), (g > 0).mean()


print("\nEach model minus the day4 rule (hold to the end), and minus the shipped day4model (anchored_l1000), $ a week, bootstrap over days:")
for k in FAMILIES:
    if k == "day4_rule":
        continue
    m_, lo_, hi_, pw = gap(res_all[k], res_all["day4 rule (hold to the end)"])
    m2, lo2, hi2, pw2 = gap(res_all[k], res_all["anchored_l1000"])
    print(f"  {k:16s} vs day4 {m_:+6.2f} [{lo_:+6.2f}, {hi_:+6.2f}] P {100 * pw:.0f}%  |  vs day4model {m2:+6.2f} [{lo2:+6.2f}, {hi2:+6.2f}] P {100 * pw2:.0f}%")
json.dump({k: [(int(i), float(r), int(d)) for i, r, d in v] for k, v in res_all.items()}, open(CACHE / f"consensus_trades_{LABEL}.json", "w"))
json.dump({"feats": FEATS, "n": len(rows)}, open(CACHE / f"consensus_{LABEL}.json", "w"))
np.save(CACHE / f"consensus_X_{LABEL}.npy", X)
json.dump([{k: v for k, v in r.items() if k not in ("x", "agree_w")} for r in rows], open(CACHE / f"consensus_rows_{LABEL}.json", "w"),
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
