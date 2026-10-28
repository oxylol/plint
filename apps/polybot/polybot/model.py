"""The data model: scores a (trader, position) the way research/analyze2.py's winner does.

model.json is written by the tournament. It is either a linear model (ridge on
return, or logistic on winning scored as predicted win probability minus the
price), a gradient-boosted tree ensemble exported tree by tree, or a hand rule;
it also carries the trading policy that won with it (threshold, trades per day,
sports or not, price band, minimum days to the market's end).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .credibility import Record

MODEL_PATH = Path(__file__).resolve().parent / "model.json"
CATS = ["sports", "politics", "geopolitics", "crypto", "culture", "economics", "tech", "weather", "other"]


class Model:
    def __init__(self, path: Path = MODEL_PATH):
        self.ok = False
        self.kind = ""
        self.target = ""
        self.features: list[str] = []
        self.policy: dict = {}
        self.trained_at = ""
        self.tournament: dict = {}
        try:
            d = json.loads(path.read_text())
            self.features = d["features"]
            self.kind = d.get("kind", "linear")
            self.target = d.get("target", "ridge")
            self.policy = d.get("policy", {})
            self.trained_at = d.get("trained_at", "")
            self.tournament = d.get("tournament", {})
            if self.kind == "linear":
                self.mean, self.std = d["mean"], d["std"]
                self.w = d.get("w") or d.get("return")   # older files: ridge on return
            elif self.kind == "gbm":
                self.lr, self.init, self.trees = float(d["lr"]), float(d["init"]), d["trees"]
            elif self.kind == "rule":
                self.name = d["name"]
            else:
                raise ValueError(self.kind)
            self.ok = True
        except Exception:
            self.ok = False

    @property
    def threshold(self) -> float:
        return float(self.policy.get("threshold", 0.0))

    # ── features ─────────────────────────────────────────────
    @staticmethod
    def vector(rec: Record, *, conviction: float, price: float, days_to_end: float | None, is_sports: bool, category: str,
               crowd: int, on_month: bool, on_all: bool, on_week: bool = False, rank: int = 100, is_no: bool = False,
               neg_risk: bool = False) -> dict[str, float]:
        n = rec.closed_n
        pf = min(rec.profit_factor, 5.0) if n else 1.0
        cat_n, cat_wins = rec.cat_stats.get(category, (0, 0))
        sports_n = rec.cat_stats.get("sports", (0, 0))[0]
        x = {
            "prior_pf": pf, "prior_wr": rec.win_rate if n else 0.5, "prior_n_log": math.log1p(n),
            "prior_roc": max(-1.0, min(rec.return_on_cost if n else 0.0, 1.0)),
            "conviction_log": math.log(max(0.05, min(conviction or 1.0, 20.0))),
            "price": price, "price_sq": price ** 2,
            "days_log": math.log1p(max(0.0, days_to_end if days_to_end is not None else 7.0)),
            "is_sports": 1.0 if is_sports else 0.0,
            "cat_share": (cat_n / n) if n else 0.3, "cat_wr": (cat_wins / cat_n) if cat_n else 0.5, "cat_n_log": math.log1p(cat_n),
            "crowd_log": math.log1p(max(0, crowd)),
            "taker_share": rec.taker_share if rec.taker_share is not None else 0.5,
            "prior_hold_log": math.log1p(rec.median_hold_hours or 24.0),
            "prior_price": rec.median_price if rec.median_price is not None else 0.5,
            "prior_sports": (sports_n / n) if n else 0.5,
            "on_month": 1.0 if on_month else 0.0, "on_all": 1.0 if on_all else 0.0, "on_week": 1.0 if on_week else 0.0,
            "rank_log": math.log1p(max(1, rank)), "is_no": 1.0 if is_no else 0.0, "neg_risk": 1.0 if neg_risk else 0.0,
        }
        for c in CATS[1:]:
            x[f"cat_{c}"] = 1.0 if category == c else 0.0
        return x

    def row(self, x: dict[str, float]) -> list[float]:
        return [float(x.get(f, 0.0)) for f in self.features]

    # ── scoring ──────────────────────────────────────────────
    def score(self, x: dict[str, float], price: float) -> float:
        """Higher is better; compare with `threshold`. Linear/ridge and gbm give an expected return on cost;
        the win-probability models give (predicted win probability − price)."""
        v = self.row(x)
        if self.kind == "linear":
            z = [(a - m) / s for a, m, s in zip(v, self.mean, self.std)] + [1.0]
            s = sum(a * b for a, b in zip(z, self.w))
            if self.target == "logit":
                return 1 / (1 + math.exp(-max(-30, min(30, s)))) - price
            return s
        if self.kind == "gbm":
            total = self.init
            for t in self.trees:
                node = 0
                left, right, feat, thr, val = t["left"], t["right"], t["feature"], t["threshold"], t["value"]
                while left[node] != -1:
                    node = left[node] if v[feat[node]] <= thr[node] else right[node]
                total += self.lr * val[node]
            if self.target == "gbm_win":
                return 1 / (1 + math.exp(-max(-30, min(30, total)))) - price
            return total
        if self.kind == "rule":
            f = x
            if self.name == "leaderboard":
                return -f["rank_log"]
            ok = f["prior_wr"] >= 0.55 and math.expm1(f["prior_n_log"]) >= 30 and f["prior_pf"] >= 1.3
            return (f["prior_wr"] + 0.1 * f["prior_pf"]) if ok else -9.0
        return -9.0

    def describe(self, x: dict[str, float], price: float) -> str:
        s = self.score(x, price)
        if self.target in ("logit", "gbm_win"):
            return f"win chance {100 * (s + price):.0f}% vs price {100 * price:.0f}¢"
        return f"predicted {s:+.0%}"
