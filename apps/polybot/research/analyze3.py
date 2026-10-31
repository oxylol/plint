"""Round 3 tournament: model families × trading policies × weekly walk-forward folds, judged as the bot's portfolio.

    python analyze3.py DATA_DIR [--report OUT.md] [--model OUT.json] [--boot 2000] [--quick]

Every fold trains on signals whose outcome was known before the fold's first day
and tests on the next seven days. Each model family gets the same tuning budget
(six configurations, chosen on the last 14 days of its training window by log
loss). Each model's probabilities feed the same portfolio simulator: the bot's
budgets ($5 a trade, at most 10 buys in 24 hours, 20 open, 3 per 5-minute cycle,
72-hour cooldown per outcome), the follower's real price (prep3.py) and the taker
fee, signals taken in the order they arrive. The policy (threshold, price band,
horizon, per-event and per-leader limits, exit on the leader's sale) is chosen on
the folds before the one being scored, never on it. Confidence intervals resample
days.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polybot import features as F  # noqa: E402

DATA = Path(sys.argv[1])
REPORT = MODEL_OUT = None
BOOT = 2000
QUICK = False
a = sys.argv[2:]
while a:
    k = a.pop(0)
    if k == "--report":
        REPORT = Path(a.pop(0))
    elif k == "--model":
        MODEL_OUT = Path(a.pop(0))
    elif k == "--boot":
        BOOT = int(a.pop(0))
    elif k == "--quick":
        QUICK = True
T0 = time.time()
rng = np.random.default_rng(11)
lines: list[str] = []
DAY = 86400


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ── data ─────────────────────────────────────────────────────
P = pd.read_parquet(DATA / "panel.parquet")
P = P[(P.p_f >= 0.02) & (P.p_f <= 0.98) & np.isfinite(P.f_logit_p)].sort_values("t_d").reset_index(drop=True)
FEATS = ["f_" + k for k in F.FEATURES]
for c in FEATS:
    P[c] = P[c].astype(np.float32).fillna(0.0)
P["hours"] = ((P.end - P.t_d) / 3600).clip(lower=0)
P["ret_c"] = P.ret.clip(upper=3.0)          # a single win counts for at most +300% when ranking anything
y = P.final.values.astype(float)
pf = P.p_f.values
say(f"# Round 3 tournament\n\n{len(P)} resolved signals with a follower price, {P.wallet.nunique()} wallets, {P.cid.nunique()} markets, "
    f"{pd.to_datetime(P.t_d.min(), unit='s'):%Y-%m-%d} → {pd.to_datetime(P.t_d.max(), unit='s'):%Y-%m-%d}; "
    f"{int(P.in_pool.sum())} from the rebuilt monthly top 200 (the bot's pool)")

GROUPS = {
    "price": ["f_logit_p", "f_price", "f_drift", "f_leader_price"],
    "bet": ["f_stance_log", "f_conviction_log", "f_fills_log", "f_since_first_log"],
    "market": ["f_hours_log", "f_is_no", "f_neg_risk", "f_in_game", "f_pre_game_h_log"] + [f"f_cat_{c}" for c in F.CATS[1:]],
    "style": ["f_age_log", "f_taker_share", "f_rec_price", "f_rec_hold_log", "f_rec_short_hold", "f_rec_longshot", "f_rec_fav", "f_rec_cost_log"],
    "skill": ["f_rec_n_log", "f_rec_wr", "f_rec_pf_log", "f_rec_roi", "f_rec_edge", "f_rec_edge_t", "f_rec_n30_log", "f_rec_roi30", "f_rec_pnl30_slog"],
    "cat_skill": ["f_cat_n_log", "f_cat_share", "f_cat_wr", "f_cat_edge"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATS), set(FEATS) ^ set(sum(GROUPS.values(), []))

# ── folds: weekly, from the first Monday of September to the end of the data ──
start0 = pd.Timestamp("2026-09-01", tz="UTC").timestamp()
last = P.t_d.max()
FOLDS = []
s = start0
while s < last:
    FOLDS.append((s, min(s + 7 * DAY, last + 1)))
    s += 7 * DAY
LIVE0 = pd.Timestamp("2026-10-28", tz="UTC").timestamp()
say(f"{len(FOLDS)} weekly folds from {pd.to_datetime(FOLDS[0][0], unit='s'):%b %d}; the last one holds the live paper period (Oct 28 →)")

# ── models ───────────────────────────────────────────────────
import lightgbm as lgb  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MAX_TRAIN = 350_000 if not QUICK else 120_000
LGB_GRID = [dict(num_leaves=15, learning_rate=0.05, n_estimators=200, min_child_samples=200, reg_lambda=5.0, colsample_bytree=0.8),
            dict(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=300, reg_lambda=10.0, colsample_bytree=0.7),
            dict(num_leaves=7, learning_rate=0.05, n_estimators=300, min_child_samples=500, reg_lambda=5.0, colsample_bytree=0.8),
            dict(num_leaves=15, learning_rate=0.03, n_estimators=500, min_child_samples=800, reg_lambda=20.0, colsample_bytree=0.6),
            dict(num_leaves=63, learning_rate=0.03, n_estimators=300, min_child_samples=1000, reg_lambda=20.0, colsample_bytree=0.6),
            dict(num_leaves=15, learning_rate=0.1, n_estimators=100, min_child_samples=300, reg_lambda=1.0, colsample_bytree=0.9)]
LOGIT_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def subsample(idx, w=None):
    if len(idx) <= MAX_TRAIN:
        return idx
    return np.sort(rng.choice(idx, MAX_TRAIN, replace=False))


def logloss(p, yy):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


class LGBModel:
    def __init__(self, feats, anchor=True, decay=None, objective="binary"):
        self.feats, self.anchor, self.decay, self.objective = feats, anchor, decay, objective

    def _w(self, idx, t_ref):
        if not self.decay:
            return None
        age = (t_ref - P.t_d.values[idx]) / DAY
        return 0.5 ** (age / self.decay)

    def fit(self, idx, t_ref, cfg):
        X = P[self.feats].values[idx]
        kw = dict(objective="binary", verbose=-1, num_threads=3, subsample=0.8, subsample_freq=1, **cfg)
        self.m = lgb.LGBMClassifier(**kw)
        self.m.fit(X, y[idx], sample_weight=self._w(idx, t_ref), init_score=lg(pf[idx]) if self.anchor else None)
        return self

    def predict(self, idx):
        raw = self.m.predict(P[self.feats].values[idx], raw_score=True)
        return sig(raw + (lg(pf[idx]) if self.anchor else 0.0))


class LogitAnchor:
    """L2 logistic regression with the market's log-odds as an offset: learns corrections to the price."""
    def __init__(self, feats):
        self.feats = feats

    def fit(self, idx, t_ref, lam):
        X = P[self.feats].values[idx].astype(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        off = lg(pf[idx])
        w = np.zeros(Z.shape[1])
        I = np.eye(Z.shape[1])
        I[-1, -1] = 0
        for _ in range(25):
            p = sig(Z @ w + off)
            g = Z.T @ (p - y[idx]) + lam * (I @ w)
            H = Z.T @ (Z * (p * (1 - p))[:, None]) + lam * I
            w -= np.linalg.solve(H, g)
        self.w = w
        return self

    def predict(self, idx):
        X = P[self.feats].values[idx].astype(float)
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return sig(Z @ self.w + lg(pf[idx]))


class Market:
    """Price only: an isotonic map from what the follower pays to how often it wins (the market's own calibration)."""
    def fit(self, idx, t_ref, cfg=None):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(pf[idx], y[idx])
        return self

    def predict(self, idx):
        return self.iso.predict(pf[idx])


class SkillEB:
    """Hierarchical trader skill with shrinkage: the leader's mean edge at resolution (final − price paid) over
    their record, shrunk toward zero by n / (n + k), added to the market's log-odds with one fitted slope."""
    def fit(self, idx, t_ref, k):
        self.k = k
        th = self._theta(idx)
        off = lg(pf[idx])
        b = 0.0
        for _ in range(20):
            p = sig(off + b * th)
            g = np.sum((p - y[idx]) * th)
            h = np.sum(p * (1 - p) * th * th) + 1e-6
            b -= g / h
        self.b = b
        return self

    def _theta(self, idx):
        n = np.expm1(P.f_rec_n_log.values[idx])
        return P.f_rec_edge.values[idx] * n / (n + self.k) * 4.0   # ×4: probability → rough log-odds scale

    def predict(self, idx):
        return sig(lg(pf[idx]) + self.b * self._theta(idx))


class PerCat:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, idx, t_ref, cfg):
        self.ms = {}
        grp = self._g(idx)
        for g in np.unique(grp):
            sub = idx[grp == g]
            if len(sub) >= 3000:
                self.ms[g] = LGBModel(self.feats).fit(sub, t_ref, cfg)
        self.all = LGBModel(self.feats).fit(idx, t_ref, cfg)
        return self

    @staticmethod
    def _g(idx):
        c = P.cat.values[idx]
        return np.where(c == "sports", "sports", np.where(c == "weather", "weather", np.where(np.isin(c, ["politics", "geopolitics"]), "politics",
                        np.where(c == "crypto", "crypto", "other"))))

    def predict(self, idx):
        out = self.all.predict(idx)
        grp = self._g(idx)
        for g, m in self.ms.items():
            sel = grp == g
            if sel.any():
                out[sel] = m.predict(idx[sel])
        return out


class Rank:
    """LightGBM lambdarank within each day (relevance: the signal made money), its score mapped to a probability
    by isotonic regression on the training window's last 14 days."""
    def __init__(self, feats):
        self.feats = feats

    def fit(self, idx, t_ref, cfg):
        cut = t_ref - 14 * DAY
        tr = idx[P.t_d.values[idx] < cut]
        va = idx[P.t_d.values[idx] >= cut]
        if len(va) < 500:
            tr, va = idx[: int(len(idx) * 0.8)], idx[int(len(idx) * 0.8):]
        days = (P.t_d.values[tr] // DAY).astype(int)
        order = np.argsort(days, kind="stable")
        tr = tr[order]
        _, counts = np.unique(days[order], return_counts=True)
        kw = {k: v for k, v in cfg.items()}
        self.m = lgb.LGBMRanker(objective="lambdarank", verbose=-1, num_threads=3, **kw)
        rel = (P.ret.values[tr] > 0).astype(int)
        self.m.fit(P[self.feats].values[tr], rel, group=counts)
        s = self.m.predict(P[self.feats].values[va])
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(s, P.ret_c.values[va])
        return self

    def predict(self, idx):
        # returns expected return per $ directly; converted to an equivalent probability for the shared policy code
        er = self.iso.predict(self.m.predict(P[self.feats].values[idx]))
        p = pf[idx]
        return np.clip((er + 1 + P.fee_rate.values[idx] * (1 - p)) * p, 1e-3, 1 - 1e-3)


class OldModel:
    """The shipped round-2 model.json, fed what the live bot feeds it: a CLOSED-only record, the leaders' price,
    days to midnight of the end date, the pool crowd."""
    def __init__(self, path):
        self.d = json.loads(Path(path).read_text())

    def fit(self, idx, t_ref, cfg=None):
        return self

    def predict(self, idx):
        d = self.d
        price = np.clip(P.leader_px.values[idx], 0.01, 0.99)
        days = np.maximum(0.0, (P.o_end_date.values[idx] - P.t_d.values[idx]) / DAY)
        cat = P.cat.values[idx]
        cols = {"price": price, "price_sq": price ** 2, "days_log": np.log1p(np.nan_to_num(days, nan=7.0)),
                "is_sports": (cat == "sports").astype(float), "crowd_log": np.log1p(P.crowd.values[idx]),
                "is_no": P.f_is_no.values[idx], "neg_risk": P.f_neg_risk.values[idx]}
        for c in F.CATS[1:]:
            cols[f"cat_{c}"] = (cat == c).astype(float)
        for k in ("prior_pf", "prior_wr", "prior_n_log", "prior_roc", "conviction_log", "cat_share", "cat_wr", "cat_n_log", "taker_share",
                  "prior_hold_log", "prior_price", "prior_sports"):
            cols[k] = P[f"o_{k}"].values[idx]
        X = np.column_stack([cols[f] for f in d["features"]])
        total = np.full(len(X), float(d["init"]))
        for t in d["trees"]:
            left, right, feat, thr, val = (np.array(t[k]) for k in ("left", "right", "feature", "threshold", "value"))
            node = np.zeros(len(X), dtype=int)
            for _ in range(12):
                leaf = left[node] == -1
                if leaf.all():
                    break
                go = X[np.arange(len(X)), np.maximum(feat[node], 0)] <= thr[node]
                node = np.where(leaf, node, np.where(go, left[node], right[node]))
            total += float(d["lr"]) * val[node]
        self.score = sig(total) - price         # its own decision quantity: win chance minus the leaders' price
        return sig(total)


def drop(group):
    return [f for f in FEATS if f not in GROUPS[group]]


FAMILIES = {
    "market": (lambda: Market(), [None]),
    "skill_eb": (lambda: SkillEB(), [5, 20, 50, 100, 200, 400]),
    "logit_anchor": (lambda: LogitAnchor(FEATS), LOGIT_GRID),
    "gbm_raw": (lambda: LGBModel(FEATS, anchor=False), LGB_GRID),
    "gbm_anchor": (lambda: LGBModel(FEATS), LGB_GRID),
    "gbm_decay30": (lambda: LGBModel(FEATS, decay=30), LGB_GRID),
    "gbm_percat": (lambda: PerCat(FEATS), LGB_GRID),
    "rank": (lambda: Rank(FEATS), [dict(num_leaves=15, learning_rate=0.05, n_estimators=200, min_child_samples=200),
                                   dict(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=300),
                                   dict(num_leaves=7, learning_rate=0.05, n_estimators=300, min_child_samples=500),
                                   dict(num_leaves=15, learning_rate=0.03, n_estimators=500, min_child_samples=800),
                                   dict(num_leaves=63, learning_rate=0.03, n_estimators=300, min_child_samples=1000),
                                   dict(num_leaves=15, learning_rate=0.1, n_estimators=100, min_child_samples=300)]),
}
for g in GROUPS:
    if g != "price":
        FAMILIES[f"abl_-{g}"] = (lambda g=g: LGBModel(drop(g)), "best:gbm_anchor")
old_path = Path(__file__).resolve().parent.parent / "polybot" / "model_r2.json"
if not old_path.exists():
    old_path = Path(__file__).resolve().parent.parent / "polybot" / "model.json"
if json.loads(old_path.read_text()).get("kind") == "gbm" and "prior_pf" in json.loads(old_path.read_text()).get("features", []):
    FAMILIES["old_r2"] = (lambda: OldModel(old_path), [None])
if QUICK:
    for k in list(FAMILIES):
        if k.startswith("abl_") or k in ("gbm_percat", "rank", "gbm_decay30"):
            FAMILIES.pop(k)

preds = {k: np.full(len(P), np.nan) for k in FAMILIES}
old_score = np.full(len(P), np.nan)
chosen_cfg: dict[str, list] = defaultdict(list)
metrics = defaultdict(list)
fold_of = np.full(len(P), -1)
best_cfg_last: dict[str, object] = {}
for fi, (s0, s1) in enumerate(FOLDS):
    tr = np.where((P.t_d.values < s0) & (P.resolved_at.values < s0))[0]
    te = np.where((P.t_d.values >= s0) & (P.t_d.values < s1))[0]
    if len(tr) < 20000 or len(te) < 200:
        say(f"  fold {pd.to_datetime(s0, unit='s'):%b %d}: skipped (train {len(tr)}, test {len(te)})")
        continue
    fold_of[te] = fi
    trs = subsample(tr)
    cut = s0 - 14 * DAY
    inner_tr = trs[P.t_d.values[trs] < cut]
    inner_tr = inner_tr[P.resolved_at.values[inner_tr] < cut]
    inner_va = trs[P.t_d.values[trs] >= cut]
    t1 = time.time()
    for name, (mk, grid) in FAMILIES.items():
        if isinstance(grid, str):
            grid = [best_cfg_last.get(grid.split(":")[1], LGB_GRID[0])]
        if len(grid) > 1 and len(inner_tr) > 5000 and len(inner_va) > 1000:
            scores = []
            for cfg in grid:
                try:
                    m = mk().fit(inner_tr, cut, cfg)
                    scores.append(logloss(m.predict(inner_va), y[inner_va]))
                except Exception as e:
                    scores.append(9.0)
            cfg = grid[int(np.argmin(scores))]
        else:
            cfg = grid[0]
        chosen_cfg[name].append(cfg)
        if name == "gbm_anchor":
            best_cfg_last["gbm_anchor"] = cfg
        m = mk().fit(trs, s0, cfg)
        preds[name][te] = m.predict(te)
        if name == "old_r2":
            old_score[te] = m.score
    say(f"  fold {pd.to_datetime(s0, unit='s'):%b %d}: train {len(tr)} (used {len(trs)}), test {len(te)}  ({time.time() - t1:.0f}s)")

# ── classifier metrics (out of sample, every signal and the pool's) ──
def calib_slope(p, yy):
    z = lg(p)
    X = np.column_stack([z, np.ones_like(z)])
    w = np.array([1.0, 0.0])
    for _ in range(25):
        q = sig(X @ w)
        g = X.T @ (q - yy)
        H = X.T @ (X * (q * (1 - q))[:, None]) + 1e-6 * np.eye(2)
        w -= np.linalg.solve(H, g)
    return w


say("\n## How well each model predicts (out of sample, all folds)\n")
say("Calibration slope 1 and intercept 0 are perfect; below 1 means over-confident. 'top-decile ret' is the mean return per $ "
    "of the 10% of signals with the highest expected return (P/price − 1 − fee), capped at +300%.\n")
say(f"{'model':16s} {'n':>8s} {'logloss':>8s} {'Δ vs mkt':>9s} {'Brier':>7s} {'AUC':>6s} {'slope':>6s} {'icpt':>6s} {'top-dec ret':>11s} {'pool top-dec':>12s}")
oos = fold_of >= 0
base_ll = logloss(preds["market"][oos], y[oos]) if "market" in preds else None
er_all = {}
for name, pr in preds.items():
    ok = oos & np.isfinite(pr)
    if ok.sum() < 100:
        continue
    er = pr / pf - 1 - P.fee_rate.values * (1 - pf)
    er_all[name] = er
    sl, ic = calib_slope(pr[ok], y[ok])
    top = ok & (er >= np.quantile(er[ok], 0.9))
    pool = ok & (P.in_pool.values == 1)
    ptop = pool & (er >= np.quantile(er[pool], 0.9)) if pool.sum() > 100 else pool
    ll = logloss(pr[ok], y[ok])
    say(f"{name:16s} {ok.sum():8d} {ll:8.4f} {ll - base_ll:+9.4f} {np.mean((pr[ok] - y[ok]) ** 2):7.4f} {roc_auc_score(y[ok], pr[ok]):6.3f} "
        f"{sl:6.2f} {ic:+6.2f} {100 * P.ret_c.values[top].mean():+10.1f}% {100 * P.ret_c.values[ptop].mean():+11.1f}%")
    metrics[name] = {"logloss": ll, "brier": float(np.mean((pr[ok] - y[ok]) ** 2)), "auc": float(roc_auc_score(y[ok], pr[ok])),
                     "slope": float(sl), "intercept": float(ic), "top_decile": float(P.ret_c.values[top].mean())}

# ── the portfolio simulator ──────────────────────────────────
CYCLE = 300


def simulate(idx, er, pol, budgets=(10, 20, 3)):
    """Chronological paper book over candidate signals `idx` (sorted by time). Returns per-trade (row, return, day)."""
    per_day, max_open, per_cycle = budgets
    lo, hi = pol["band"]
    p = pf[idx]
    h = P.hours.values[idx]
    ok = (er[idx] >= pol["thr"]) & (p >= lo) & (p <= hi) & (h >= pol["min_h"]) & (h <= pol["max_h"])
    if pol.get("cats"):
        ok &= np.isin(P.cat.values[idx], pol["cats"])
    if pol.get("no_cats"):
        ok &= ~np.isin(P.cat.values[idx], pol["no_cats"])
    if pol.get("max_drift") is not None:
        ok &= P.f_drift.values[idx] <= pol["max_drift"]
    cand = idx[ok]
    if len(cand) == 0:
        return []
    t = P.t_d.values
    res = P.resolved_at.values
    exit_t = np.where(np.isfinite(P.t_lsell.values) & (P.t_lsell.values < res), P.t_lsell.values + 300, res) if pol.get("exit") else res
    r = P.ret_exit.values if pol.get("exit") else P.ret.values
    open_until: list = []          # (release time, event, wallet, asset)
    buys: list = []                # buy times (rolling 24 h)
    last_buy_asset: dict = {}
    trades = []
    cyc_key, cyc_n = None, 0
    # within a 5-minute cycle the strongest signals go first
    order = sorted(range(len(cand)), key=lambda k: (int(t[cand[k]] // CYCLE), -er[cand[k]]))
    for k in order:
        i = cand[k]
        now = t[i]
        open_until = [o for o in open_until if o[0] > now]
        ck = int(now // CYCLE)
        if ck != cyc_key:
            cyc_key, cyc_n = ck, 0
        if cyc_n >= per_cycle or len(open_until) >= max_open:
            continue
        buys = [b for b in buys if b > now - DAY]
        if len(buys) >= per_day:
            continue
        a = P.asset.values[i]
        if a in last_buy_asset and now - last_buy_asset[a] < 72 * 3600:
            continue
        if any(o[3] == a for o in open_until):
            continue
        evk = P.ev.values[i]
        if pol.get("per_event", 99) < 99 and sum(1 for o in open_until if o[1] == evk) >= pol["per_event"]:
            continue
        w = P.wallet.values[i]
        if pol.get("leader_cap", 1.0) < 1.0 and open_until:
            share = (sum(1 for o in open_until if o[2] == w) + 1) / (len(open_until) + 1)
            if share > pol["leader_cap"] and len(open_until) >= 3:
                continue
        open_until.append((exit_t[i], evk, w, a))
        buys.append(now)
        last_buy_asset[a] = now
        cyc_n += 1
        trades.append((i, r[i], int(now // DAY)))
    return trades


def summarize(trades):
    if not trades:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "pnl_week": 0.0, "win": float("nan")}
    rr = np.array([x[1] for x in trades])
    rc = np.minimum(rr, 3.0)
    dd = np.array([x[2] for x in trades])
    ud = np.unique(dd)
    sums = np.array([rr[dd == d].sum() for d in ud])
    cnts = np.array([(dd == d).sum() for d in ud])
    draws = rng.integers(0, len(ud), size=(BOOT, len(ud)))
    bm = sums[draws].sum(1) / cnts[draws].sum(1)
    span_days = max(7.0, (dd.max() - dd.min() + 1))
    return {"n": len(rr), "mean": float(rr.mean()), "mean_c": float(rc.mean()), "lo": float(np.quantile(bm, 0.05)), "hi": float(np.quantile(bm, 0.95)),
            "win": float((rr > 0).mean()), "pnl_week": float(5 * rr.sum() / span_days * 7), "days": len(ud)}


POLICY_GRID = []
for thr in (0.0, 0.04, 0.08, 0.12, 0.2, 0.3):
    for band in ((0.05, 0.95), (0.15, 0.85), (0.3, 0.7)):
        for min_h, max_h in ((0.0, 1e9), (1.0, 1e9), (0.0, 72.0), (1.0, 24 * 14)):
            for per_event in (1, 99):
                for exit_ in (False, True):
                    POLICY_GRID.append({"thr": thr, "band": band, "min_h": min_h, "max_h": max_h, "per_event": per_event, "leader_cap": 0.34, "exit": exit_})
if QUICK:
    POLICY_GRID = [p for p in POLICY_GRID if not p["exit"] and p["per_event"] == 1]
say(f"\n{len(POLICY_GRID)} policies per model; each fold's policy is the one with the most P&L on the folds before it (at least 20 trades)")

pool_mask = P.in_pool.values == 1


def walk_policy(name, er, mask=pool_mask, grid=POLICY_GRID):
    """Nested walk-forward: per fold, pick the policy by P&L on earlier folds, trade it on this fold."""
    per_fold = defaultdict(dict)
    for fi in range(len(FOLDS)):
        idx = np.where((fold_of == fi) & mask & np.isfinite(er))[0]
        if len(idx) == 0:
            continue
        for pi, pol in enumerate(grid):
            per_fold[fi][pi] = simulate(idx, er, pol)
    out, picks = [], []
    folds = sorted(per_fold)
    for k, fi in enumerate(folds):
        if k < 2:
            continue           # need two earlier folds to choose on
        prev = folds[:k]
        def score(pi):
            tr = [x for f in prev for x in per_fold[f].get(pi, [])]
            return sum(x[1] for x in tr) if len(tr) >= 20 else -1e9
        best = max(range(len(grid)), key=score)
        picks.append((fi, best))
        out += per_fold[fi].get(best, [])
    allbest = max(range(len(grid)), key=lambda pi: sum(x[1] for f in folds for x in per_fold[f].get(pi, [])) if sum(len(per_fold[f].get(pi, [])) for f in folds) >= 30 else -1e9)
    return out, picks, grid[allbest], per_fold


say("\n## The bot's portfolio, out of sample (pool signals, nested policy choice, $5 a trade)\n")
say(f"{'model':16s} {'trades':>6s} {'mean/trade':>10s} {'90% CI':>17s} {'win':>5s} {'$/week':>8s}  last policy")
results = {}
for name in preds:
    er = er_all.get(name)
    if er is None:
        continue
    tr, picks, final_pol, per_fold = walk_policy(name, er)
    sm = summarize(tr)
    results[name] = {"trades": tr, "summary": sm, "picks": picks, "final_policy": final_pol, "per_fold": per_fold}
    lp = POLICY_GRID[picks[-1][1]] if picks else None
    say(f"{name:16s} {sm['n']:6d} {100 * sm['mean']:+9.1f}% [{100 * sm['lo']:+6.1f}, {100 * sm['hi']:+6.1f}] {100 * sm['win']:4.0f}% {sm['pnl_week']:+8.2f}  "
        f"{lp if lp else '-'}")

# ── rival rules on the same simulator ─────────────────────────
say("\n## Rival rules on the same book\n")
rules = {}
# copy the day board's sports bets at 40–60¢ (the live paper run's one positive setup)
m_day = (P.in_day25.values == 1) & (P.cat.values == "sports") & oos
rules["day_sports_40_60"] = simulate(np.where(m_day)[0], np.ones(len(P)), {"thr": 0.0, "band": (0.40, 0.60), "min_h": 0.0, "max_h": 1e9})
# copy every pool signal, any market, 5–95¢
rules["copy_pool_all"] = simulate(np.where(pool_mask & oos)[0], np.ones(len(P)), {"thr": 0.0, "band": (0.05, 0.95), "min_h": 0.0, "max_h": 1e9})
# the current `learned` setup: round-2 model, its policy, two pool leaders needed (find_signals' min_show), 5 a day
if "old_r2" in preds:
    od = json.loads(old_path.read_text())["policy"]
    m_old = pool_mask & oos & (P.crowd.values >= 1) & (P.leader_px.values >= od["min_price"]) & (P.leader_px.values <= od["max_price"])
    end_by_date_h = (P.o_end_date.values - P.t_d.values) / 3600
    m_old &= end_by_date_h >= 24 * od["min_days"]
    rules["learned_now"] = simulate(np.where(m_old)[0], np.nan_to_num(old_score, nan=-9),
                                    {"thr": od["threshold"], "band": (od["min_price"], od["max_price"]), "min_h": 0.0, "max_h": 1e9},
                                    budgets=(od["per_day"], 20, 3))
for k, tr in rules.items():
    sm = summarize(tr)
    results[k] = {"trades": tr, "summary": sm}
    say(f"{k:16s} {sm['n']:6d} {100 * sm['mean']:+9.1f}% [{100 * sm['lo']:+6.1f}, {100 * sm['hi']:+6.1f}] {100 * sm['win']:4.0f}% {sm['pnl_week']:+8.2f}")


# ── the winner, against each rival ───────────────────────────
def paired(a_tr, b_tr):
    """Difference in mean return per trade (a − b), bootstrap over days (unpaired trades, common day resampling)."""
    da = defaultdict(list)
    db = defaultdict(list)
    for _, r, d in a_tr:
        da[d].append(r)
    for _, r, d in b_tr:
        db[d].append(r)
    days = sorted(set(da) | set(db))
    if not days:
        return float("nan"), float("nan"), float("nan")
    A = [(sum(da.get(d, [])), len(da.get(d, []))) for d in days]
    B = [(sum(db.get(d, [])), len(db.get(d, []))) for d in days]
    A, B = np.array(A), np.array(B)
    draws = rng.integers(0, len(days), size=(BOOT, len(days)))
    ma = A[draws, 0].sum(1) / np.maximum(1, A[draws, 1].sum(1))
    mb = B[draws, 0].sum(1) / np.maximum(1, B[draws, 1].sum(1))
    diff = ma - mb
    # weekly P&L difference too
    return float(np.median(diff)), float(np.quantile(diff, 0.05)), float((diff > 0).mean())


cands = {k: v for k, v in results.items() if v["summary"]["n"] >= 30 and not k.startswith("abl_")}
model_cands = [k for k in cands if k in preds]
winner = max(model_cands, key=lambda k: cands[k]["summary"]["pnl_week"]) if model_cands else None
if winner:
    say(f"\n## Winner: {winner}\n")
    W = results[winner]
    for k, v in results.items():
        if k == winner or v["summary"]["n"] < 10:
            continue
        med, lo5, pwin = paired(W["trades"], v["trades"])
        say(f"  vs {k:16s}: +{100 * med:5.1f} points per trade (median of bootstrap), 5th pct {100 * lo5:+6.1f}, "
            f"P(winner better) {100 * pwin:.0f}%; $/week {W['summary']['pnl_week']:+.2f} vs {v['summary']['pnl_week']:+.2f}")

    def breakdown(trades, key, labels):
        g = defaultdict(list)
        for i, r, d in trades:
            g[key(i)].append(r)
        return " · ".join(f"{k} {len(g[k])} ({100 * np.mean(g[k]):+.0f}%)" for k in labels if k in g)

    tr = W["trades"]
    say("  by type: " + breakdown(tr, lambda i: P.cat.values[i], sorted(set(P.cat.values))))
    say("  by price paid: " + breakdown(tr, lambda i: "<30¢" if pf[i] < .3 else "30–50¢" if pf[i] < .5 else "50–70¢" if pf[i] < .7 else "≥70¢",
                                         ["<30¢", "30–50¢", "50–70¢", "≥70¢"]))
    say("  by time to the end: " + breakdown(tr, lambda i: "<6h" if P.hours.values[i] < 6 else "6–48h" if P.hours.values[i] < 48 else "2–14d" if P.hours.values[i] < 336 else ">14d",
                                             ["<6h", "6–48h", "2–14d", ">14d"]))
    say("  in play: " + breakdown(tr, lambda i: "in play" if P.f_in_game.values[i] > 0.5 else "not in play", ["in play", "not in play"]))
    lead = defaultdict(int)
    for i, _, _ in tr:
        lead[P.wallet.values[i]] += 1
    top = sorted(lead.values(), reverse=True)
    say(f"  leaders: {len(lead)} distinct; the busiest is behind {100 * top[0] / len(tr):.0f}% of trades, the top five {100 * sum(top[:5]) / len(tr):.0f}%")
    live = [x for x in tr if P.t_d.values[x[0]] >= LIVE0]
    sl = summarize(live)
    say(f"  live paper period (Oct 28 →): {sl['n']} trades, mean {100 * sl['mean']:+.1f}% per trade, win {100 * sl['win']:.0f}%")
    for k in ("day_sports_40_60", "learned_now"):
        if k in results:
            lv = [x for x in results[k]["trades"] if P.t_d.values[x[0]] >= LIVE0]
            s2 = summarize(lv)
            say(f"    {k}: {s2['n']} trades, mean {100 * s2['mean']:+.1f}% per trade")

if REPORT:
    REPORT.write_text("\n".join(lines) + "\n")
json.dump({k: {kk: vv for kk, vv in v["summary"].items()} for k, v in results.items()}, open(DATA / "tournament3.json", "w"), indent=1, default=float)
say(f"\ntotal {time.time() - T0:.0f}s")
