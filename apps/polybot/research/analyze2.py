"""Tournament: which model and trading rule would have made a follower the most, judged honestly.

    python analyze2.py DATA_DIR [--model OUT.json] [--report OUT.md] [--boot 400]

Walk-forward: for several dates, fit on positions entered and finished before
the date, then simulate a follower trading the positions entered in the next
two weeks: filter (price band, sports or not, days to the market's end), rank
by the model, take the top K per day, pay slippage and the taker fee, book the
position's return. Every policy is scored by the bootstrap 5th percentile of
its mean return per trade across folds (resampling days), so a rule has to be
good in the bad draws too, not just on average. Thousands of such simulations
pick the winner, which is exported with its policy for the bot.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polybot.categories import category as _category  # noqa: E402

DATA = Path(sys.argv[1])
MODEL_OUT = REPORT_OUT = None
PAYOFF = None       # --payoff MODEL.json: score the recent weeks with a saved model and report the money, no tournament
PAYOFF_DAYS = 28
STAKE = 5.0
BOOT = 400
args = sys.argv[2:]
while args:
    a = args.pop(0)
    if a == "--model":
        MODEL_OUT = Path(args.pop(0))
    elif a == "--report":
        REPORT_OUT = Path(args.pop(0))
    elif a == "--boot":
        BOOT = int(args.pop(0))
    elif a == "--payoff":
        PAYOFF = Path(args.pop(0))
    elif a == "--days":
        PAYOFF_DAYS = int(args.pop(0))
    elif a == "--stake":
        STAKE = float(args.pop(0))
rng = np.random.default_rng(7)
T_START = time.time()
lines: list[str] = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def parse(s):
    if not s:
        return None
    d = dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


CATS = ["sports", "politics", "geopolitics", "crypto", "culture", "economics", "tech", "weather", "other"]
FEE = {"sports": 0.05, "crypto": 0.07, "politics": 0.04, "economics": 0.05, "tech": 0.04, "culture": 0.05, "weather": 0.05, "geopolitics": 0.0, "other": 0.05}

# ── markets (resolutions, tags) — only for markets our positions touch, kept tiny ──
TAG_CATS = [("sports", ("sports", "nba", "nfl", "nhl", "mlb", "soccer", "tennis", "esports", "ufc", "mma", "cricket", "golf", "f1", "boxing", "rugby", "games")),
            ("politics", ("politics", "elections", "election", "congress", "senate", "governor", "midterms", "trump", "world elections")),
            ("geopolitics", ("geopolitics", "world", "middle east", "ukraine", "israel", "iran", "russia", "china", "war", "military")),
            ("crypto", ("crypto", "bitcoin", "ethereum", "solana", "memecoins", "defi")),
            ("economics", ("economics", "economy", "fed", "finance", "business", "stocks", "macro", "inflation", "earnings")),
            ("tech", ("tech", "ai", "science", "space", "openai")),
            ("weather", ("weather", "climate", "temperature")),
            ("culture", ("pop culture", "culture", "entertainment", "movies", "music", "tv", "celebrities", "awards", "mentions", "tweets"))]


def tag_category(m: dict) -> str | None:
    tags = [t.lower() for t in (m.get("tags") or [])] + [str(m.get("category") or "").lower()]
    if m.get("sports_type"):
        return "sports"
    for name, words in TAG_CATS:
        if any(any(w == t or w in t for w in words) for t in tags if t):
            return name
    return None


traders = json.loads((DATA / "traders.json").read_text())
COLLECTED_AT = dt.datetime.fromtimestamp((DATA / "positions.jsonl").stat().st_mtime, dt.timezone.utc)
wanted_cids: set[str] = set()
with (DATA / "positions.jsonl").open() as f:
    for line in f:
        j = line.find('"condition_id": "')
        if j >= 0:
            wanted_cids.add(line[j + 17: j + 17 + 66])
markets: dict[str, tuple] = {}   # cid → (category, final yes, final no, labels)
for mp in (DATA / "markets.jsonl", DATA.parent / "research2" / "markets.jsonl"):
    if mp.exists():
        with mp.open() as f:
            for line in f:
                j = line.find('"condition_id": "')
                cid = line[j + 17: j + 17 + 66] if j >= 0 else ""
                if cid not in wanted_cids:
                    continue
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                pr = m.get("prices") or [None, None]
                markets[cid] = (tag_category(m), pr[0], pr[1], tuple(str(x or "").lower() for x in (m.get("labels") or [])))
        break
say(f"{len(wanted_cids)} markets in the positions, {len(markets)} of them with tags/resolutions on file")

# ── load positions (compact rows) ────────────────────────────
meta: dict[str, dict] = {}
rows: list[dict] = []
settled_wallets: set[str] = set()
KEEP = ("wallet", "condition_id", "outcome", "outcome_index", "avg_price", "total_size", "current_price", "realized_pnl", "total_pnl",
        "first_entry_at", "last_event_at", "end_date", "negative_risk", "settled")
with (DATA / "positions.jsonl").open() as f:
    for line in f:
        r0 = json.loads(line)
        if r0.get("meta"):
            meta[r0["wallet"]] = {k: r0.get(k) for k in ("fills_30d", "taker_30d", "age_days")}
            if r0.get("settled_meta"):
                settled_wallets.add(r0["wallet"])
            continue
        if r0.get("settled_meta"):
            settled_wallets.add(r0["wallet"])
            continue
        cost = r0["total_size"] * r0["avg_price"]
        if cost < 1 or not r0.get("first_entry_at"):
            continue
        r = {k: r0.get(k) for k in KEEP}
        r["title"] = (r0.get("title") or "")[:70]
        r["slug"] = r0.get("event_slug") or r0.get("slug") or ""
        r["entry"] = parse(r["first_entry_at"])
        r["end"] = parse(r["end_date"] + "T00:00:00+00:00") if r.get("end_date") and "T" not in r["end_date"] else parse(r.get("end_date"))
        if r.get("settled"):
            pnl = r["total_pnl"]
            r["exit"] = min(r["end"], COLLECTED_AT) if r["end"] else (parse(r.get("last_event_at")) or r["entry"])
        else:
            pnl = r["realized_pnl"]
            if not r.get("last_event_at"):
                continue
            r["exit"] = parse(r["last_event_at"])
        r["pnl"] = pnl
        r["cost"] = cost
        r["ret_exit"] = max(-1.0, min(pnl / cost, 20.0))     # what the trader actually made
        r["hold_h"] = max(0.0, (r["exit"] - r["entry"]).total_seconds() / 3600)
        r["days_to_end"] = ((r["end"] - r["entry"]).total_seconds() / 86400) if r["end"] else None
        m = markets.get(r["condition_id"])
        r["cat"] = (m[0] if m else None) or _category(r["title"], r["slug"])
        # What a follower makes: the position held to the market's resolution. The row's own current
        # price is the token's final price once the market has resolved (0 or 1); the markets file
        # confirms it when we have it.
        final = None
        cp = r.get("current_price")
        resolved_by_date = r["end"] is not None and r["end"] < COLLECTED_AT - dt.timedelta(hours=6)
        if cp is not None and (r.get("settled") or resolved_by_date):
            if cp >= 0.99:
                final = 1.0
            elif cp <= 0.01:
                final = 0.0
        if m is not None:
            idx = r["outcome_index"] if r.get("outcome_index") in (0, 1) else (m[3].index(str(r["outcome"]).lower()) if r.get("outcome") and str(r["outcome"]).lower() in m[3] else None)
            if idx is not None and m[1 + idx] is not None:
                pr = m[1 + idx]
                if pr >= 0.99:
                    final = 1.0
                elif pr <= 0.01:
                    final = 0.0
        if final is None:
            continue   # unresolved (or unknown): a follower's outcome is not known
        if r.get("negative_risk") and str(r.get("outcome") or "").lower() == "no":
            continue   # the API's cost basis for these is not trustworthy
        r["ret"] = max(-1.0, min(final / r["avg_price"] - 1.0, 20.0))
        r["win"] = 1 if final >= 1.0 else 0
        for k in ("first_entry_at", "last_event_at", "end_date", "total_pnl", "realized_pnl", "current_price", "total_size"):
            r.pop(k, None)
        rows.append(r)
del wanted_cids
rows = [r for r in rows if r["wallet"] in settled_wallets]
say(f"{len(rows)} resolved positions from {len({r['wallet'] for r in rows})} traders "
    f"({sum(1 for r in rows if r.get('settled'))} resolved-unclaimed; returns are at resolution), loaded in {time.time() - T_START:.0f}s")

# ── trader priors (expanding window) ─────────────────────────
by_wallet: dict[str, list[dict]] = defaultdict(list)
for r in rows:
    by_wallet[r["wallet"]].append(r)
for w, lst in by_wallet.items():
    src = traders.get(w, {})
    windows = {b["window"] for b in src.get("boards", [])}
    best_rank = min((b["rank"] for b in src.get("boards", [])), default=2000)
    m = meta.get(w, {})
    taker_share = (m["taker_30d"] / m["fills_30d"]) if m.get("fills_30d") else None
    by_exit = sorted(lst, key=lambda r: r["exit"])
    # running sums over positions in exit order; for each position (in entry order) use those exited before entry
    order = sorted(lst, key=lambda r: r["entry"])
    j = 0
    n = wins = 0
    gw = gl = spnl = scost = 0.0
    costs: list[float] = []
    holds: list[float] = []
    prices: list[float] = []
    cat_n: dict[str, int] = defaultdict(int)
    cat_w: dict[str, int] = defaultdict(int)
    sports_n = 0
    for r in order:
        while j < len(by_exit) and by_exit[j]["exit"] < r["entry"]:
            q = by_exit[j]
            n += 1
            wins += q["win"]
            if q["pnl"] > 0:
                gw += q["pnl"]
            else:
                gl -= q["pnl"]
            spnl += q["pnl"]
            scost += q["cost"]
            costs.append(q["cost"])
            holds.append(q["hold_h"])
            prices.append(q["avg_price"])
            cat_n[q["cat"]] += 1
            cat_w[q["cat"]] += q["win"]
            sports_n += 1 if q["cat"] == "sports" else 0
            j += 1
        r["prior_n"] = n
        r["prior_wr"] = wins / n if n else None
        r["prior_pf"] = (gw / gl) if gl > 0 else (5.0 if gw > 0 else None)
        r["prior_roc"] = spnl / scost if scost > 0 else None
        r["prior_median_cost"] = statistics.median(costs) if costs else None
        r["prior_hold"] = statistics.median(holds) if holds else None
        r["prior_price"] = statistics.median(prices) if prices else None
        r["prior_sports"] = sports_n / n if n else None
        cn = cat_n.get(r["cat"], 0)
        r["prior_cat_share"] = cn / n if n else None
        r["prior_cat_wr"] = cat_w.get(r["cat"], 0) / cn if cn else None
        r["prior_cat_n"] = cn
        r["conviction"] = (r["cost"] / r["prior_median_cost"]) if r["prior_median_cost"] else None
        r["on_month"] = 1 if "month" in windows else 0
        r["on_all"] = 1 if "all" in windows else 0
        r["on_week"] = 1 if "week" in windows else 0
        r["best_rank"] = best_rank
        r["from_board"] = 1 if src.get("source") == "board" else 0
        r["taker_share"] = taker_share
        r["age_days"] = m.get("age_days")

by_side: dict[tuple, list[dict]] = defaultdict(list)
for r in rows:
    by_side[(r["condition_id"], r["outcome"])].append(r)
for lst in by_side.values():
    lst.sort(key=lambda r: r["entry"])
    for i, r in enumerate(lst):
        r["crowd_before"] = sum(1 for q in lst[:i] if q["wallet"] != r["wallet"] and (r["entry"] - q["entry"]).total_seconds() <= 48 * 3600)
say(f"features built in {time.time() - T_START:.0f}s")

# No board membership or rank: those are known today, not at the time of the old positions (they would leak).
FEATURES = ["prior_pf", "prior_wr", "prior_n_log", "prior_roc", "conviction_log", "price", "price_sq", "days_log", "is_sports",
            "cat_share", "cat_wr", "cat_n_log", "crowd_log", "taker_share", "prior_hold_log", "prior_price", "prior_sports",
            "is_no", "neg_risk"] + [f"cat_{c}" for c in CATS[1:]]


def vec(r):
    pf = min(r["prior_pf"], 5.0) if r["prior_pf"] is not None else 1.0
    return [pf, r["prior_wr"] if r["prior_wr"] is not None else 0.5, math.log1p(r["prior_n"]),
            max(-1, min(r["prior_roc"] if r["prior_roc"] is not None else 0.0, 1.0)),
            math.log(max(0.05, min(r["conviction"] if r["conviction"] else 1.0, 20.0))),
            r["avg_price"], r["avg_price"] ** 2, math.log1p(max(0.0, r["days_to_end"] if r["days_to_end"] is not None else 7.0)),
            1.0 if r["cat"] == "sports" else 0.0, r["prior_cat_share"] if r["prior_cat_share"] is not None else 0.3,
            r["prior_cat_wr"] if r["prior_cat_wr"] is not None else 0.5, math.log1p(r["prior_cat_n"]), math.log1p(r["crowd_before"]),
            r["taker_share"] if r["taker_share"] is not None else 0.5, math.log1p(r["prior_hold"] if r["prior_hold"] else 24.0),
            r["prior_price"] if r["prior_price"] is not None else 0.5, r["prior_sports"] if r["prior_sports"] is not None else 0.5,
            1.0 if (r.get("outcome") or "").lower() == "no" else 0.0, 1.0 if r.get("negative_risk") else 0.0] + \
           [1.0 if r["cat"] == c else 0.0 for c in CATS[1:]]


usable = [r for r in rows if r["prior_n"] >= 5 and r["days_to_end"] is not None]
X_all = np.array([vec(r) for r in usable], dtype=float)
rank_log_all = np.array([math.log1p(r["best_rank"]) for r in usable])
y_ret = np.array([max(-1.0, min(r["ret"], 3.0)) for r in usable])
y_win = np.array([r["win"] for r in usable], dtype=float)
entry = np.array([r["entry"].timestamp() for r in usable])
exit_ = np.array([r["exit"].timestamp() for r in usable])
price = np.array([r["avg_price"] for r in usable])
cat = np.array([r["cat"] for r in usable])
days = np.array([r["days_to_end"] for r in usable])
ret_raw = np.array([r["ret"] for r in usable])
day_idx = np.array([int(r["entry"].timestamp() // 86400) for r in usable])
fee_rate = np.array([FEE[c] for c in cat])
say(f"{len(usable)} positions with 5+ prior positions for the models")

# ── payoff of a saved model under the live policy ────────────
if PAYOFF is not None:
    mdl = json.loads(PAYOFF.read_text())
    feats = mdl["features"]
    col = {f: i for i, f in enumerate(FEATURES)}
    Xp = X_all[:, [col[f] for f in feats]]
    if mdl["kind"] == "gbm":
        total = np.full(len(Xp), float(mdl["init"]))
        for t in mdl["trees"]:
            left, right, feat, thr, val = (np.array(t[k]) for k in ("left", "right", "feature", "threshold", "value"))
            node = np.zeros(len(Xp), dtype=int)
            for _ in range(12):
                leaf = left[node] == -1
                if leaf.all():
                    break
                go_left = Xp[np.arange(len(Xp)), np.maximum(feat[node], 0)] <= thr[node]
                node = np.where(leaf, node, np.where(go_left, left[node], right[node]))
            total += float(mdl["lr"]) * val[node]
        if mdl.get("target") == "gbm_win":
            sc = 1 / (1 + np.exp(-np.clip(total, -30, 30))) - price
        else:
            sc = total
    else:
        Z = (Xp - np.array(mdl["mean"])) / np.array(mdl["std"])
        sc = np.hstack([Z, np.ones((len(Z), 1))]) @ np.array(mdl["w"])
        if mdl.get("target") == "logit":
            sc = 1 / (1 + np.exp(-np.clip(sc, -30, 30))) - price
    pol = mdl["policy"]
    since = entry.max() - PAYOFF_DAYS * 86400
    ok = (entry >= since) & (sc >= pol["threshold"]) & (price >= pol["min_price"]) & (price <= pol["max_price"]) & (days >= pol["min_days"])
    if not pol.get("sports", True):
        ok &= cat != "sports"
    sel = np.where(ok)[0]
    byday: dict[int, list] = defaultdict(list)
    for i in sel:
        byday[day_idx[i]].append(i)
    chosen = []
    for dd, items in sorted(byday.items()):
        items.sort(key=lambda i: -sc[i])
        chosen.extend(items[: int(pol["per_day"])])
    ch = np.array(chosen)
    pb = price[ch] + 0.01
    fr_cap = np.minimum(3.0, (1 + ret_raw[ch]) * price[ch] / pb - 1 - fee_rate[ch] * (1 - pb))
    fr_raw = (1 + ret_raw[ch]) * price[ch] / pb - 1 - fee_rate[ch] * (1 - pb)
    hold = np.array([max(0.0, (usable[i]["exit"] - usable[i]["entry"]).total_seconds() / 86400) for i in ch])
    ndays = len(set(day_idx[ch]))
    span = (entry[ch].max() - entry[ch].min()) / 86400 + 1
    say(f"\n=== Payoff of {PAYOFF.name} under its policy, last {PAYOFF_DAYS} days of entries, ${STAKE:.0f} a trade ===")
    say(f"policy: score ≥ {pol['threshold']:+.3f}, {pol['per_day']} a day, {'any market' if pol.get('sports', True) else 'no sports'}, "
        f"{100*pol['min_price']:.0f}–{100*pol['max_price']:.0f}¢, ≥ {pol['min_days']:g} days to the end  (note: the saved model was fit on this period too; "
        f"the tournament's out-of-sample numbers are the honest ones)")
    say(f"{len(ch)} trades over {span:.0f} days ({len(ch)/span:.1f} a day; {ndays} days had a trade) · win {100*(fr_cap>0).mean():.0f}% · "
        f"mean return per trade {100*fr_cap.mean():+.0f}% capped, {100*fr_raw.mean():+.0f}% uncapped, median {100*np.median(fr_cap):+.0f}%")
    say(f"per $5 trade: {STAKE*fr_cap.mean():+.2f} $ on average → {STAKE*fr_cap.mean()*len(ch)/span:+.2f} $ a day, {30*STAKE*fr_cap.mean()*len(ch)/span:+.0f} $ a month, "
        f"on {STAKE*len(ch)/span*30:.0f} $ staked a month")
    say(f"hold until resolution: median {np.median(hold):.1f} days, mean {hold.mean():.1f}, 90th pct {np.quantile(hold, .9):.0f} days")
    # capital tied up: stakes open on each day
    t_open = entry[ch]; t_close = t_open + hold * 86400
    grid = np.arange(t_open.min(), t_close.max() + 86400, 86400)
    open_now = np.array([((t_open <= g) & (t_close > g)).sum() for g in grid])
    say(f"money tied up at once: average {STAKE*open_now.mean():.0f} $, peak {STAKE*open_now.max():.0f} $ "
        f"→ about {100*30*fr_cap.mean()*len(ch)/span/max(1, open_now.mean()):+.0f}% a month on the average capital tied up")
    cats_c = defaultdict(list)
    for j, i in enumerate(ch):
        cats_c[cat[i]].append(fr_cap[j])
    say("by market type: " + " · ".join(f"{k} {len(v)} ({100*np.mean(v):+.0f}%, win {100*np.mean(np.array(v)>0):.0f}%)" for k, v in sorted(cats_c.items(), key=lambda kv: -len(kv[1]))))
    pbk = defaultdict(list)
    for j, i in enumerate(ch):
        pbk["<30¢" if price[i] < .3 else "30–50¢" if price[i] < .5 else "50–75¢"].append(fr_cap[j])
    say("by price: " + " · ".join(f"{k} {len(v)} ({100*np.mean(v):+.0f}%, win {100*np.mean(np.array(v)>0):.0f}%)" for k, v in pbk.items()))
    # week by week
    wk = ((entry[ch] - since) // (7 * 86400)).astype(int)
    say("week by week: " + " · ".join(f"w{w+1}: {int((wk==w).sum())} trades {100*fr_cap[wk==w].mean():+.0f}%" for w in sorted(set(wk))))
    if REPORT_OUT:
        REPORT_OUT.write_text("\n".join(lines))
    sys.exit(0)

# ── models ───────────────────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    HAVE_SK = True
except Exception:
    HAVE_SK = False
    say("scikit-learn not available: linear models only")


class Linear:
    def __init__(self, kind, lam):
        self.kind, self.lam = kind, lam

    def fit(self, X, yr, yw):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        I = np.eye(Z.shape[1]); I[-1, -1] = 0
        if self.kind == "ridge":
            self.w = np.linalg.solve(Z.T @ Z + self.lam * I, Z.T @ yr)
        else:
            w = np.zeros(Z.shape[1])
            for _ in range(30):
                p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
                g = Z.T @ (p - yw) + self.lam * (I @ w)
                H = Z.T @ (Z * (p * (1 - p))[:, None]) + self.lam * I
                w -= np.linalg.solve(H, g)
            self.w = w
        return self

    def score(self, X, price):
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        s = Z @ self.w
        if self.kind == "ridge":
            return s
        p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
        return p - price  # mispricing: predicted win probability above the price paid

    def export(self):
        return {"kind": "linear", "target": self.kind, "features": FEATURES, "mean": self.mu.tolist(), "std": self.sd.tolist(), "w": self.w.tolist()}


class Boosted:
    def __init__(self, kind, n=150, depth=3, lr=0.05):
        self.kind, self.n, self.depth, self.lr = kind, n, depth, lr

    MAX_TRAIN = 250000

    def fit(self, X, yr, yw):
        if len(X) > self.MAX_TRAIN:
            keep = rng.choice(len(X), self.MAX_TRAIN, replace=False)
            X, yr, yw = X[keep], yr[keep], yw[keep]
        if self.kind == "gbm":
            self.m = GradientBoostingRegressor(n_estimators=self.n, max_depth=self.depth, learning_rate=self.lr, subsample=0.8, min_samples_leaf=50, random_state=1).fit(X, yr)
        else:
            self.m = GradientBoostingClassifier(n_estimators=self.n, max_depth=self.depth, learning_rate=self.lr, subsample=0.8, min_samples_leaf=50, random_state=1).fit(X, yw)
        return self

    def score(self, X, price):
        if self.kind == "gbm":
            return self.m.predict(X)
        return self.m.predict_proba(X)[:, 1] - price

    def export(self):
        trees = []
        for est in self.m.estimators_[:, 0]:
            t = est.tree_
            trees.append({"left": t.children_left.tolist(), "right": t.children_right.tolist(), "feature": t.feature.tolist(),
                          "threshold": t.threshold.tolist(), "value": t.value[:, 0, 0].tolist()})
        if hasattr(self.m.init_, "constant_"):
            init = float(self.m.init_.constant_[0, 0])
        else:
            p1 = float(self.m.init_.class_prior_[1])
            init = math.log(p1 / (1 - p1))
        return {"kind": "gbm", "target": self.kind, "features": FEATURES, "lr": self.lr, "init": init, "trees": trees}


class Rule:
    """Hand rules, to see whether a model beats them."""
    def __init__(self, name):
        self.name = name

    def fit(self, X, yr, yw):
        return self

    def score(self, X, price, idx=None):
        f = {k: X[:, i] for i, k in enumerate(FEATURES)}
        if self.name == "leaderboard":
            return -rank_log_all[idx] if idx is not None else np.zeros(len(X))
        if self.name == "credible":
            ok = (f["prior_wr"] >= 0.55) & (np.expm1(f["prior_n_log"]) >= 30) & (f["prior_pf"] >= 1.3)
            return np.where(ok, f["prior_wr"] + 0.1 * f["prior_pf"], -9.0)
        raise ValueError(self.name)

    def export(self):
        return {"kind": "rule", "name": self.name, "features": FEATURES}


MODELS = {"ridge": lambda: Linear("ridge", 10.0), "logit": lambda: Linear("logit", 10.0),
          "rule:leaderboard": lambda: Rule("leaderboard"), "rule:credible": lambda: Rule("credible")}
if HAVE_SK:
    MODELS["gbm"] = lambda: Boosted("gbm")
    MODELS["gbm_win"] = lambda: Boosted("gbm_win")

# ── folds ────────────────────────────────────────────────────
last = dt.datetime.fromtimestamp(entry.max(), dt.timezone.utc)
fold_ends = []
d = last - dt.timedelta(days=14)
while len(fold_ends) < 7 and d.timestamp() > entry.min() + 45 * 86400:
    fold_ends.append(d)
    d -= dt.timedelta(days=14)
fold_ends = sorted(fold_ends)
say(f"folds (train before, test the next 14 days): {', '.join(x.strftime('%b %d') for x in fold_ends)}")

# ── policies ─────────────────────────────────────────────────
POLICIES = []
for q in (0.5, 0.75, 0.9):
    for k in (2, 3, 5):
        for sports in (True, False):
            for band in ((0.05, 0.95), (0.15, 0.75), (0.25, 0.70)):
                for min_days in (0.5, 2.0):
                    POLICIES.append({"q": q, "k": k, "sports": sports, "band": band, "min_days": min_days})
say(f"{len(MODELS)} models × {len(POLICIES)} policies × {len(fold_ends)} folds, bootstrap {BOOT} per fold")


CAP = 3.0   # a single win counts for at most +300% in the ranking; lottery tickets must not carry a rule


def follower_returns(ret, p, fee):
    """What a follower makes: buys 1¢ above the trader's average price, pays the taker fee on entry."""
    pb = p + 0.01
    value = (1 + ret) * p
    return np.minimum(CAP, value / pb - 1 - fee * (1 - pb))


def simulate(scores, thr, idx, pol):
    """Trades a policy on a test fold; returns per-trade follower returns and their day ids."""
    lo, hi = pol["band"]
    ok = (scores >= thr) & (price[idx] >= lo) & (price[idx] <= hi) & (days[idx] >= pol["min_days"])
    if not pol["sports"]:
        ok &= cat[idx] != "sports"
    sel = np.where(ok)[0]
    if len(sel) == 0:
        return np.array([]), np.array([])
    chosen = []
    byday: dict[int, list] = defaultdict(list)
    for i in sel:
        byday[day_idx[idx[i]]].append(i)
    for dday, items in byday.items():
        items.sort(key=lambda i: -scores[i])
        chosen.extend(items[: pol["k"]])
    chosen = np.array(chosen)
    fr = follower_returns(ret_raw[idx[chosen]], price[idx[chosen]], fee_rate[idx[chosen]])
    return fr, day_idx[idx[chosen]]


def bootstrap_low(fr, dids, boot):
    """5th percentile of the mean per-trade return when days are resampled."""
    if len(fr) == 0:
        return float("nan")
    udays = np.unique(dids)
    if len(udays) < 3:
        return float(fr.mean())
    groups = [fr[dids == dd] for dd in udays]
    sums = np.array([g.sum() for g in groups]); cnts = np.array([len(g) for g in groups])
    draws = rng.integers(0, len(udays), size=(boot, len(udays)))
    means = sums[draws].sum(1) / np.maximum(1, cnts[draws].sum(1))
    return float(np.quantile(means, 0.05))


results: dict[tuple, list] = defaultdict(list)   # (model, policy index) → per-fold metrics
fitted: dict[tuple, object] = {}
t_models = time.time()
for fi, fend in enumerate(fold_ends):
    fe = fend.timestamp(); te = (fend + dt.timedelta(days=14)).timestamp()
    tr = np.where((entry < fe) & (exit_ < fe))[0]
    te_idx = np.where((entry >= fe) & (entry < te))[0]
    if len(tr) < 2000 or len(te_idx) < 100:
        say(f"  fold {fend:%b %d}: skipped (train {len(tr)}, test {len(te_idx)})")
        continue
    for mname, mk in MODELS.items():
        m = mk().fit(X_all[tr], y_ret[tr], y_win[tr])
        fitted[(mname, fi)] = m
        if isinstance(m, Rule):
            s_tr = m.score(X_all[tr], price[tr], tr)
            s_te = m.score(X_all[te_idx], price[te_idx], te_idx)
        else:
            s_tr = m.score(X_all[tr], price[tr])
            s_te = m.score(X_all[te_idx], price[te_idx])
        for pi, pol in enumerate(POLICIES):
            thr = float(np.quantile(s_tr, pol["q"]))
            fr, dids = simulate(s_te, thr, te_idx, pol)
            results[(mname, pi)].append({
                "fold": fi, "n": int(len(fr)), "mean": float(fr.mean()) if len(fr) else float("nan"),
                "win": float((fr > 0).mean()) if len(fr) else float("nan"), "low": bootstrap_low(fr, dids, BOOT), "thr": thr,
            })
    say(f"  fold {fend:%b %d}: train {len(tr)}, test {len(te_idx)} ({time.time() - t_models:.0f}s)")

# ── ranking ──────────────────────────────────────────────────
table = []
for (mname, pi), folds in results.items():
    ns = [f["n"] for f in folds]
    if len(folds) < 3 or statistics.mean(ns) < 8:
        continue
    means = [f["mean"] for f in folds if f["n"] > 0]
    lows = [f["low"] for f in folds if f["n"] > 0 and not math.isnan(f["low"])]
    wins = [f["win"] for f in folds if f["n"] > 0]
    if len(means) < 3:
        continue
    total_n = sum(ns)
    wmean = sum(f["mean"] * f["n"] for f in folds if f["n"] > 0) / max(1, total_n)
    table.append({"model": mname, "policy": POLICIES[pi], "pi": pi, "folds": len(means), "trades": total_n, "per_fold": statistics.mean(ns),
                  "mean": wmean, "low": statistics.mean(lows), "worst_fold": min(means), "win": statistics.mean(wins),
                  "score": statistics.mean(lows) if min(means) > -0.5 else -9})
table.sort(key=lambda t: -t["score"])
say(f"\n{len(table)} (model, policy) pairs had enough trades; ranked by the average bootstrap 5th percentile of the mean return per trade\n")
say(f"{'model':16s} {'q':>4s} {'k':>2s} {'sports':>6s} {'band':>11s} {'days':>4s} | {'trades':>6s} {'/fold':>5s} {'mean':>7s} {'5th pct':>7s} {'worst':>7s} {'win':>4s}")
for t in table[:30]:
    p = t["policy"]
    say(f"{t['model']:16s} {p['q']:4.2f} {p['k']:2d} {'yes' if p['sports'] else 'no':>6s} {p['band'][0]:.2f}–{p['band'][1]:.2f} {p['min_days']:4.1f} | "
        f"{t['trades']:6d} {t['per_fold']:5.0f} {100*t['mean']:+6.1f}% {100*t['low']:+6.1f}% {100*t['worst_fold']:+6.1f}% {100*t['win']:3.0f}%")
say("\nBest per model:")
best_by_model = {}
for t in table:
    if t["model"] not in best_by_model:
        best_by_model[t["model"]] = t
for mname, t in best_by_model.items():
    p = t["policy"]
    say(f"  {mname:16s} mean {100*t['mean']:+6.1f}%  5th pct {100*t['low']:+6.1f}%  win {100*t['win']:3.0f}%  trades/fold {t['per_fold']:.0f}  "
        f"(q {p['q']}, k {p['k']}, sports {'yes' if p['sports'] else 'no'}, band {p['band'][0]}–{p['band'][1]}, days ≥ {p['min_days']})")

# The baseline everybody uses: copy the leaderboard, any market, top 3 a day.
base = [t for t in table if t["model"] == "rule:leaderboard" and t["policy"]["sports"] and t["policy"]["k"] == 3 and t["policy"]["band"] == (0.05, 0.95) and t["policy"]["min_days"] == 0.5 and t["policy"]["q"] == 0.5]
if base:
    say(f"\nFor scale, 'copy the leaderboard, anything, 3 a day': mean {100*base[0]['mean']:+.1f}% per trade, 5th pct {100*base[0]['low']:+.1f}%, win {100*base[0]['win']:.0f}%")

winner = table[0] if table else None
if winner:
    mname, pol = winner["model"], winner["policy"]
    say(f"\nWhat the winner trades (last three folds):")
    for fi in range(max(0, len(fold_ends) - 3), len(fold_ends)):
        m = fitted.get((mname, fi))
        if m is None:
            continue
        fend = fold_ends[fi]; fe = fend.timestamp(); te = (fend + dt.timedelta(days=14)).timestamp()
        tr = np.where((entry < fe) & (exit_ < fe))[0]; te_idx = np.where((entry >= fe) & (entry < te))[0]
        s_tr = m.score(X_all[tr], price[tr], tr) if isinstance(m, Rule) else m.score(X_all[tr], price[tr])
        s_te = m.score(X_all[te_idx], price[te_idx], te_idx) if isinstance(m, Rule) else m.score(X_all[te_idx], price[te_idx])
        thr = float(np.quantile(s_tr, pol["q"]))
        lo, hi = pol["band"]
        ok = (s_te >= thr) & (price[te_idx] >= lo) & (price[te_idx] <= hi) & (days[te_idx] >= pol["min_days"])
        if not pol["sports"]:
            ok &= cat[te_idx] != "sports"
        sel = np.where(ok)[0]
        byday = defaultdict(list)
        for i in sel:
            byday[day_idx[te_idx[i]]].append(i)
        chosen = []
        for dd, items in byday.items():
            items.sort(key=lambda i: -s_te[i]); chosen.extend(items[: pol["k"]])
        if not chosen:
            continue
        ch = np.array(chosen)
        fr = follower_returns(ret_raw[te_idx[ch]], price[te_idx[ch]], fee_rate[te_idx[ch]])
        cats_c = defaultdict(int)
        for i in ch:
            cats_c[cat[te_idx[i]]] += 1
        pb = price[te_idx[ch]]
        say(f"  fold {fend:%b %d}: {len(ch)} trades, mean {100*fr.mean():+.0f}%, win {100*(fr>0).mean():.0f}% · types {dict(cats_c)} · "
            f"price <30¢ {100*(pb<0.3).mean():.0f}%, 30–70¢ {100*((pb>=0.3)&(pb<=0.7)).mean():.0f}%, >70¢ {100*(pb>0.7).mean():.0f}%")
        top = sorted(range(len(ch)), key=lambda j: -fr[j])[:4]
        for j in top:
            r = usable[te_idx[ch[j]]]
            say(f"     {100*fr[j]:+5.0f}%  {r['cat']:11s} {r['avg_price']:.2f}  {(r.get('title') or '')[:48]} · {r.get('outcome')}  (trader prior wr {r['prior_wr']:.0%}, n {r['prior_n']})")
if winner and MODEL_OUT:
    # Refit the winning model on everything that has finished, with the winning policy's threshold from its scores.
    mname = winner["model"]
    tr = np.where(exit_ < exit_.max() - 86400)[0]
    m = MODELS[mname]().fit(X_all[tr], y_ret[tr], y_win[tr])
    s_tr = m.score(X_all[tr], price[tr])
    pol = winner["policy"]
    thr = float(np.quantile(s_tr, pol["q"]))
    out = m.export()
    out.update({"policy": {"threshold": thr, "per_day": pol["k"], "sports": pol["sports"], "min_price": pol["band"][0], "max_price": pol["band"][1],
                           "min_days": pol["min_days"]},
                "tournament": {"models": list(MODELS), "policies": len(POLICIES), "folds": len(fold_ends), "bootstrap": BOOT,
                               "mean_per_trade": winner["mean"], "fifth_pct": winner["low"], "win": winner["win"], "trades_per_fold": winner["per_fold"]},
                "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "rows": int(len(tr))})
    MODEL_OUT.write_text(json.dumps(out))
    say(f"\nwinner: {mname} with q={pol['q']}, k={pol['k']}, sports={'yes' if pol['sports'] else 'no'}, band {pol['band']}, days ≥ {pol['min_days']} "
        f"→ {MODEL_OUT} (threshold {thr:+.4f}, {len(tr)} rows)")
say(f"\ntotal {time.time() - T_START:.0f}s")
if REPORT_OUT:
    REPORT_OUT.write_text("\n".join(lines))
