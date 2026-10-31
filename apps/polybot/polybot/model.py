"""The data model: scores a (trader, position) the way the research tournaments' winners do.

A model file is written by a tournament. Round 2 (research/analyze2.py, kept as
model_r2.json) is a linear model (ridge on return, or logistic on winning scored
as predicted win probability minus the price), a gradient-boosted tree ensemble
exported tree by tree, or a hand rule. Round 3 (research/analyze3.py, model.json)
is kind "lgbm": LightGBM trees over polybot/features.py's inputs, anchored on the
market: the win probability is sigmoid(logit(price the follower pays) + trees),
and its decision quantity is the expected return per dollar after the taker fee.
Each file also carries the trading policy that won with it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .credibility import Record

MODEL_PATH = Path(__file__).resolve().parent / "model.json"
MODEL_DIR = MODEL_PATH.parent
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
        self.inputs = "leader"
        self.version = 2
        try:
            d = json.loads(path.read_text())
            self.features = d["features"]
            self.kind = d.get("kind", "linear")
            self.target = d.get("target", "ridge")
            self.policy = d.get("policy", {})
            self.trained_at = d.get("trained_at", "")
            self.tournament = d.get("tournament", {})
            self.version = int(d.get("version", 2))
            self.anchor = d.get("anchor")            # "logit_p": the model corrects the market's log-odds (round 3)
            self.inputs = d.get("inputs", "leader")  # round 3: "leader" (features.vector) or "consensus" (features.consensus_context)
            if self.kind == "skill":
                self.k, self.b = float(d["k"]), float(d["b"])
            elif self.kind == "linear":
                self.mean, self.std = d["mean"], d["std"]
                self.w = d.get("w") or d.get("return")   # older files: ridge on return
            elif self.kind in ("gbm", "lgbm"):
                self.lr, self.init, self.trees = float(d.get("lr", 1.0)), float(d.get("init", 0.0)), d["trees"]
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

    @classmethod
    def load(cls, name: str = "") -> "Model":
        """A model file by name, from the package directory (empty: model.json)."""
        return cls(MODEL_DIR / name) if name else cls()

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
    def _trees(self, v: list[float]) -> float:
        total = self.init
        for t in self.trees:
            node = 0
            left, right, feat, thr, val = t["left"], t["right"], t["feature"], t["threshold"], t["value"]
            while left[node] != -1:
                node = left[node] if v[feat[node]] <= thr[node] else right[node]
            total += self.lr * val[node]
        return total

    @property
    def round3(self) -> bool:
        return self.version >= 3

    def win_probability(self, x: dict[str, float]) -> float:
        """Round-3 models: the chance the outcome wins, given everything in x (x["logit_p"] is the follower's price).
        lgbm: trees; linear: standardised inputs × weights; skill: the leader's record edge, shrunk by n / (n + k)."""
        if self.kind == "skill":
            n = math.expm1(x["rec_n_log"])
            z = self.b * x["rec_edge"] * n / (n + self.k) * 4.0
        elif self.kind == "linear":
            v = self.row(x)
            z = sum((a - m) / sd * w for a, m, sd, w in zip(v, self.mean, self.std, self.w)) + self.w[-1]
        else:
            z = self._trees(self.row(x))
        z += x["logit_p"] if self.anchor == "logit_p" else 0.0
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))

    def score(self, x: dict[str, float], price: float, fee_rate: float = 0.0) -> float:
        """Higher is better; compare with `threshold`. Linear/ridge and gbm give an expected return on cost;
        the round-2 win-probability models give (predicted win probability − price); round-3 (lgbm) models give
        the expected return per dollar at `price` after the taker fee: P/price − 1 − fee_rate × (1 − price)."""
        if self.round3:
            p = min(0.999, max(0.001, price))
            return self.win_probability(x) / p - 1.0 - fee_rate * (1.0 - p)
        v = self.row(x)
        if self.kind == "linear":
            z = [(a - m) / s for a, m, s in zip(v, self.mean, self.std)] + [1.0]
            s = sum(a * b for a, b in zip(z, self.w))
            if self.target == "logit":
                return 1 / (1 + math.exp(-max(-30, min(30, s)))) - price
            return s
        if self.kind == "gbm":
            total = self._trees(v)
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
        if self.round3:
            return f"win chance {100 * self.win_probability(x):.0f}% vs price {100 * price:.0f}¢"
        s = self.score(x, price)
        if self.target in ("logit", "gbm_win"):
            return f"win chance {100 * (s + price):.0f}% vs price {100 * price:.0f}¢"
        return f"predicted {s:+.0%}"
