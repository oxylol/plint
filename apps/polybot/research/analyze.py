"""Learn, from collected positions, what predicts that a trader's position was worth following.

    python analyze.py DATA_DIR [--model OUT.json] [--split-days 14]

Features are what a follower could know at entry time: the trader's track
record *before* that position (no peeking), how big the bet is for them, the
entry price, how long until the market ends, the market type, and how many
other traders in the pool took the same side within two days. The target is
the position's realized return on cost. The model is a small ridge/logistic
regression fitted on positions that closed before the split and judged on the
ones after. numpy only; the result is written as model.json for the bot.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
MODEL_OUT = None
SPLIT_DAYS = 14
MIN_DAYS = None   # keep only positions entered this many days or more before the market's end (what a follower can copy)
args = sys.argv[2:]
while args:
    a = args.pop(0)
    if a == "--model":
        MODEL_OUT = Path(args.pop(0))
    elif a == "--split-days":
        SPLIT_DAYS = int(args.pop(0))
    elif a == "--min-days":
        MIN_DAYS = float(args.pop(0))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polybot.categories import category as _category  # noqa: E402


def parse(s):
    if not s:
        return None
    d = dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def category(row):
    return _category(row.get("title"), row.get("event_slug") or row.get("slug"))


# ── load ─────────────────────────────────────────────────────
traders = json.loads((DATA / "traders.json").read_text())
meta: dict[str, dict] = {}
rows: list[dict] = []
settled_wallets = set()
n_settled = 0
for line in (DATA / "positions.jsonl").open():
    r = json.loads(line)
    if r.get("meta"):
        meta[r["wallet"]] = r
        continue
    if r.get("settled_meta"):
        settled_wallets.add(r["wallet"])
        continue
    cost = r["total_size"] * r["avg_price"]
    if cost < 1 or r["first_entry_at"] is None:
        continue
    r["entry"] = parse(r["first_entry_at"])
    r["end"] = parse(r["end_date"] + "T00:00:00+00:00") if r.get("end_date") and "T" not in r["end_date"] else parse(r.get("end_date"))
    if r.get("settled"):
        # Resolved but never redeemed: the tokens are worth current_price (0 for a loss, 1 for an unclaimed win).
        pnl = r["total_pnl"]
        r["exit"] = r["end"] or parse(r["last_event_at"]) or r["entry"]
        n_settled += 1
    else:
        pnl = r["realized_pnl"]
        if r["last_event_at"] is None:
            continue
        r["exit"] = parse(r["last_event_at"])
    r["pnl"] = pnl
    r["realized_pnl"] = pnl
    r["cost"] = cost
    r["ret"] = max(-1.0, min(pnl / cost, 20.0))
    r["win"] = 1 if pnl > 0 else 0
    r["hold_h"] = max(0.0, (r["exit"] - r["entry"]).total_seconds() / 3600)
    r["days_to_end"] = ((r["end"] - r["entry"]).total_seconds() / 86400) if r["end"] else None
    r["cat"] = category(r)
    rows.append(r)
# Only traders with both passes give a complete picture of finished positions.
if settled_wallets:
    rows = [r for r in rows if r["wallet"] in settled_wallets]
print(f"{len(rows)} finished positions ({n_settled} resolved-unclaimed, the rest exited) from {len({r['wallet'] for r in rows})} traders "
      f"({len(traders)} collected, {len(settled_wallets)} with both passes)")
if MIN_DAYS is not None:
    rows = [r for r in rows if r["days_to_end"] is not None and r["days_to_end"] >= MIN_DAYS]
    print(f"keeping {len(rows)} positions entered at least {MIN_DAYS:g} days before the market's end")
if not rows:
    sys.exit(0)

# ── trader-level priors: only positions that closed before this one was opened ──
by_wallet: dict[str, list[dict]] = defaultdict(list)
for r in rows:
    by_wallet[r["wallet"]].append(r)
for w, lst in by_wallet.items():
    lst.sort(key=lambda r: r["exit"])
    src = traders.get(w, {})
    board_windows = {b["window"] for b in src.get("boards", [])}
    for r in lst:
        prior = [q for q in lst if q["exit"] < r["entry"]]
        n = len(prior)
        wins = sum(q["win"] for q in prior)
        gw = sum(q["realized_pnl"] for q in prior if q["realized_pnl"] > 0)
        gl = -sum(q["realized_pnl"] for q in prior if q["realized_pnl"] < 0)
        same_cat = [q for q in prior if q["cat"] == r["cat"]]
        r["prior_n"] = n
        r["prior_wr"] = wins / n if n else None
        r["prior_pf"] = (gw / gl) if gl > 0 else (5.0 if gw > 0 else None)
        r["prior_roc"] = (sum(q["realized_pnl"] for q in prior) / sum(q["cost"] for q in prior)) if n else None
        r["prior_median_cost"] = statistics.median(q["cost"] for q in prior) if n else None
        r["prior_hold"] = statistics.median(q["hold_h"] for q in prior) if n else None
        r["prior_cat_share"] = len(same_cat) / n if n else None
        r["prior_cat_wr"] = (sum(q["win"] for q in same_cat) / len(same_cat)) if same_cat else None
        r["conviction"] = (r["cost"] / r["prior_median_cost"]) if r["prior_median_cost"] else None
        r["on_month"] = 1 if "month" in board_windows else 0
        r["on_all"] = 1 if "all" in board_windows else 0
        r["from_board"] = 1 if src.get("source") == "board" else 0
        m = meta.get(w, {})
        r["taker_share"] = (m["taker_30d"] / m["fills_30d"]) if m.get("fills_30d") else None
        r["age_days"] = m.get("age_days")

# crowd: other pool traders on the same side within ±48 h
by_side: dict[tuple, list[dict]] = defaultdict(list)
for r in rows:
    by_side[(r["condition_id"], r["outcome"])].append(r)
for lst in by_side.values():
    for r in lst:
        r["crowd"] = sum(1 for q in lst if q["wallet"] != r["wallet"] and abs((q["entry"] - r["entry"]).total_seconds()) <= 48 * 3600)
        r["crowd_before"] = sum(1 for q in lst if q["wallet"] != r["wallet"] and 0 <= (r["entry"] - q["entry"]).total_seconds() <= 48 * 3600)


# ── tables ───────────────────────────────────────────────────
def table(name, key, rows, buckets):
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is None:
            groups["(unknown)"].append(r)
            continue
        label = None
        for lo, hi, lab in buckets:
            if lo <= v < hi:
                label = lab
                break
        groups[label or "(other)"].append(r)
    print(f"\n{name:38s} {'n':>6s} {'win':>5s} {'avg ret':>8s} {'median':>7s} {'$ pnl/$1k':>10s}")
    order = [lab for _, _, lab in buckets] + ["(other)", "(unknown)"]
    for lab in order:
        g = groups.get(lab)
        if not g:
            continue
        rets = [r["ret"] for r in g]
        cost = sum(r["cost"] for r in g)
        pnl = sum(r["realized_pnl"] for r in g)
        print(f"  {lab:36s} {len(g):6d} {100*sum(r['win'] for r in g)/len(g):4.0f}% {100*statistics.mean(rets):+7.1f}% {100*statistics.median(rets):+6.1f}% {1000*pnl/cost:+9.0f}")


def cat_table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["cat"]].append(r)
    print(f"\n{'market type':38s} {'n':>6s} {'win':>5s} {'avg ret':>8s} {'median':>7s} {'$ pnl/$1k':>10s}")
    for lab, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rets = [r["ret"] for r in g]
        cost = sum(r["cost"] for r in g)
        pnl = sum(r["realized_pnl"] for r in g)
        print(f"  {lab:36s} {len(g):6d} {100*sum(r['win'] for r in g)/len(g):4.0f}% {100*statistics.mean(rets):+7.1f}% {100*statistics.median(rets):+6.1f}% {1000*pnl/cost:+9.0f}")


print("\n=== Everything (return = realized P&L / cost of the position; '$ pnl/$1k' = dollars made per $1,000 put in) ===")
cat_table(rows)
table("trader's prior profit factor", lambda r: r["prior_pf"], rows, [(0, 1, "< 1 (losing)"), (1, 1.3, "1–1.3"), (1.3, 2, "1.3–2"), (2, 4, "2–4"), (4, 99, "4+")])
table("trader's prior win rate", lambda r: r["prior_wr"], rows, [(0, .4, "< 40%"), (.4, .55, "40–55%"), (.55, .7, "55–70%"), (.7, .9, "70–90%"), (.9, 1.01, "90%+")])
table("trader's prior closed positions", lambda r: r["prior_n"], rows, [(0, 10, "< 10"), (10, 30, "10–30"), (30, 100, "30–100"), (100, 1e9, "100+")])
table("bet size vs trader's usual", lambda r: r["conviction"], rows, [(0, .5, "< ½×"), (.5, 1, "½–1×"), (1, 2, "1–2×"), (2, 5, "2–5×"), (5, 1e9, "5×+")])
table("entry price", lambda r: r["avg_price"], rows, [(0, .1, "< 10¢"), (.1, .3, "10–30¢"), (.3, .5, "30–50¢"), (.5, .7, "50–70¢"), (.7, .9, "70–90¢"), (.9, 1.01, "90¢+")])
table("days from entry to market end", lambda r: r["days_to_end"], rows, [(-1e9, 0, "after end?"), (0, 1, "< 1 day"), (1, 3, "1–3 days"), (3, 7, "3–7 days"), (7, 30, "1–4 weeks"), (30, 1e9, "1 month+")])
table("hold time (hours)", lambda r: r["hold_h"], rows, [(0, 1, "< 1 h"), (1, 6, "1–6 h"), (6, 24, "6–24 h"), (24, 168, "1–7 days"), (168, 1e9, "1 week+")])
table("trader's median hold (hours)", lambda r: r["prior_hold"], rows, [(0, 6, "< 6 h"), (6, 24, "6–24 h"), (24, 168, "1–7 days"), (168, 1e9, "1 week+")])
table("trader's specialisation in this type", lambda r: r["prior_cat_share"], rows, [(0, .2, "< 20%"), (.2, .5, "20–50%"), (.5, .8, "50–80%"), (.8, 1.01, "80%+")])
table("trader's prior win rate in this type", lambda r: r["prior_cat_wr"], rows, [(0, .4, "< 40%"), (.4, .55, "40–55%"), (.55, .7, "55–70%"), (.7, 1.01, "70%+")])
table("others in the pool on the same side ±48h", lambda r: r["crowd"], rows, [(0, 1, "none"), (1, 2, "1"), (2, 4, "2–3"), (4, 8, "4–7"), (8, 1e9, "8+")])
table("taker share (30 d)", lambda r: r["taker_share"], rows, [(0, .2, "< 20% (maker)"), (.2, .6, "20–60%"), (.6, 1.01, "60%+ (taker)")])
table("how it finished", lambda r: 1 if r.get("settled") else 0, rows, [(0, 1, "exited (sold or redeemed)"), (1, 2, "resolved, never redeemed")])
table("on a board?", lambda r: r["from_board"], rows, [(0, 1, "control: ordinary holder"), (1, 2, "leaderboard trader")])
table("on the monthly board?", lambda r: r["on_month"], rows, [(0, 1, "no"), (1, 2, "yes")])
table("entry month", lambda r: r["entry"].year * 100 + r["entry"].month, rows, [(202500, 202606, "before Jun 2026"), (202606, 202608, "Jun–Jul 2026"), (202608, 202610, "Aug–Sep 2026"), (202610, 202612, "Oct 2026")])

# ── model ────────────────────────────────────────────────────
FEATURES = ["prior_pf", "prior_wr", "prior_n_log", "prior_roc", "conviction_log", "price", "price_sq", "days_log", "is_sports",
            "cat_share", "cat_wr", "crowd_log", "taker_share", "prior_hold_log", "on_month", "on_all"]


def vec(r):
    pf = min(r["prior_pf"], 5.0) if r["prior_pf"] is not None else 1.0
    return [pf, r["prior_wr"] if r["prior_wr"] is not None else 0.5, math.log1p(r["prior_n"]),
            max(-1, min(r["prior_roc"] if r["prior_roc"] is not None else 0.0, 1.0)),
            math.log(max(0.05, min(r["conviction"] if r["conviction"] else 1.0, 20.0))),
            r["avg_price"], r["avg_price"] ** 2, math.log1p(max(0.0, r["days_to_end"] if r["days_to_end"] is not None else 7.0)),
            1.0 if r["cat"] == "sports" else 0.0, r["prior_cat_share"] if r["prior_cat_share"] is not None else 0.3,
            r["prior_cat_wr"] if r["prior_cat_wr"] is not None else 0.5, math.log1p(r["crowd"]),
            r["taker_share"] if r["taker_share"] is not None else 0.5, math.log1p(r["prior_hold"] if r["prior_hold"] else 24.0),
            float(r["on_month"]), float(r["on_all"])]


usable = [r for r in rows if r["prior_n"] >= 5]
collected_at = max(min(r["exit"], r["entry"] + dt.timedelta(days=400)) for r in rows if not r.get("settled")) if rows else dt.datetime.now(dt.timezone.utc)
split = collected_at - dt.timedelta(days=SPLIT_DAYS)
train = [r for r in usable if r["entry"] < split and r["exit"] < split]
test = [r for r in usable if r["entry"] >= split]
print(f"\n=== Model: {len(train)} train (entered and finished before {split.date()}), {len(test)} test (entered after) ===")
X = np.array([vec(r) for r in train]); y_ret = np.array([max(-1, min(r["ret"], 3.0)) for r in train]); y_win = np.array([r["win"] for r in train], dtype=float)
mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
Z = (X - mu) / sd
Z1 = np.hstack([Z, np.ones((len(Z), 1))])
lam = 3.0
w_ret = np.linalg.solve(Z1.T @ Z1 + lam * np.eye(Z1.shape[1]), Z1.T @ y_ret)  # ridge on return
# logistic regression on win by Newton steps
w_win = np.zeros(Z1.shape[1])
for _ in range(25):
    p = 1 / (1 + np.exp(-Z1 @ w_win))
    g = Z1.T @ (p - y_win) + lam * w_win
    H = Z1.T @ (Z1 * (p * (1 - p))[:, None]) + lam * np.eye(Z1.shape[1])
    w_win -= np.linalg.solve(H, g)
print(f"{'feature':16s} {'→ return':>9s} {'→ win':>7s}")
for i, f in enumerate(FEATURES):
    print(f"  {f:16s} {w_ret[i]:+8.3f} {w_win[i]:+7.3f}")

def predict(rs, w):
    if not rs:
        return np.array([])
    Xt = (np.array([vec(r) for r in rs]) - mu) / sd
    return np.hstack([Xt, np.ones((len(Xt), 1))]) @ w

def edge_score(rs):
    """Predicted win probability against the price paid: p_win / price − 1 is the expected return if held to the end."""
    if not rs:
        return np.array([])
    p = 1 / (1 + np.exp(-predict(rs, w_win)))
    price = np.array([max(0.02, r["avg_price"]) for r in rs])
    return p / price - 1

def judge(rs, label, scorer=None, name="predicted return"):
    scorer = scorer or (lambda x: predict(x, w_ret))
    if len(rs) < 50:
        print(f"{label}: too few rows ({len(rs)})")
        return
    pr = scorer(rs)
    rets = np.array([r["ret"] for r in rs]); wins = np.array([r["win"] for r in rs])
    order = np.argsort(-pr)
    n = len(rs)
    print(f"\n{label} [{name}]: {n} positions, overall avg return {100*rets.mean():+.1f}%, win {100*wins.mean():.0f}%")
    for frac, nm in ((0.1, "top 10%"), (0.25, "top 25%"), (0.5, "top 50%"), (1.0, "all")):
        k = max(1, int(n * frac)); idx = order[:k]
        cost = sum(rs[i]["cost"] for i in idx); pnl = sum(rs[i]["realized_pnl"] for i in idx)
        print(f"  {nm:12s} n={k:6d}  avg ret {100*rets[idx].mean():+6.1f}%  median {100*np.median(rets[idx]):+6.1f}%  win {100*wins[idx].mean():4.0f}%  $/$1k {1000*pnl/cost:+7.0f}  score ≥ {pr[idx].min():+.3f}")
    worst = order[-max(1, int(n * 0.25)):]
    print(f"  {'bottom 25%':12s} n={len(worst):6d}  avg ret {100*rets[worst].mean():+6.1f}%  win {100*wins[worst].mean():4.0f}%")
    rk_p = np.argsort(np.argsort(pr)); rk_r = np.argsort(np.argsort(rets))
    print(f"  rank correlation score vs actual return: {np.corrcoef(rk_p, rk_r)[0,1]:+.3f}")

def calibration(rs, label):
    if len(rs) < 200:
        return
    p = 1 / (1 + np.exp(-predict(rs, w_win)))
    wins = np.array([r["win"] for r in rs]); price = np.array([r["avg_price"] for r in rs])
    print(f"\nCalibration of the win model, {label}: predicted win → actual win (and the price paid)")
    edges = np.quantile(p, np.linspace(0, 1, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            print(f"  predicted {100*lo:3.0f}–{100*hi:3.0f}%: actual {100*wins[m].mean():3.0f}%  avg price {100*price[m].mean():3.0f}¢  n={m.sum()}")

judge(train, "In sample")
judge(test, "Out of sample (entered after the split)")
judge(test, "Out of sample (entered after the split)", edge_score, "edge = p_win/price − 1")
calibration(test, "out of sample")
pre = [r for r in usable if r["entry"] < dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)]
judge(pre, "Entered before October (before the boards that picked these traders)")
judge(pre, "Entered before October (before the boards that picked these traders)", edge_score, "edge")
ctrl = [r for r in usable if not r["from_board"]]
judge(ctrl, "Control group only (ordinary holders, never on a board)")
judge(ctrl, "Control group only (ordinary holders, never on a board)", edge_score, "edge")
if test:
    pr = predict(test, w_ret); ed = edge_score(test)
    cut10 = float(np.quantile(pr, 0.9)); cut25 = float(np.quantile(pr, 0.75))
    ecut10 = float(np.quantile(ed, 0.9)); ecut25 = float(np.quantile(ed, 0.75))
else:
    cut10 = cut25 = ecut10 = ecut25 = 0.0

if MODEL_OUT:
    MODEL_OUT.write_text(json.dumps({
        "features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
        "return": w_ret.tolist(), "win": w_win.tolist(), "cut_top10": cut10, "cut_top25": cut25,
        "edge_cut_top10": ecut10, "edge_cut_top25": ecut25,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "train_rows": len(train), "test_rows": len(test),
    }, indent=1))
    print(f"\nmodel written to {MODEL_OUT} (return cuts {cut10:+.3f}/{cut25:+.3f}, edge cuts {ecut10:+.3f}/{ecut25:+.3f})")
