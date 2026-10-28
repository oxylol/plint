"""The data model: predicts how a copied position will do, from research/analyze.py's model.json.

Features are the same as in the analysis, built from what the bot knows at
entry time: the leader's measured record (pack.py), how big this bet is for
them, the entry price, days to the market's end, the market type, how many
others in the pool took the side, and the leader's board membership.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .credibility import Record

MODEL_PATH = Path(__file__).resolve().parent / "model.json"


class Model:
    def __init__(self, path: Path = MODEL_PATH):
        self.ok = False
        self.features: list[str] = []
        self.mean: list[float] = []
        self.std: list[float] = []
        self.w_ret: list[float] = []
        self.w_win: list[float] = []
        self.cut_top10 = 0.0
        self.cut_top25 = 0.0
        self.trained_at = ""
        try:
            d = json.loads(path.read_text())
            self.features, self.mean, self.std = d["features"], d["mean"], d["std"]
            self.w_ret, self.w_win = d["return"], d["win"]
            self.cut_top10, self.cut_top25 = float(d.get("cut_top10", 0)), float(d.get("cut_top25", 0))
            self.trained_at = d.get("trained_at", "")
            self.ok = True
        except Exception:
            self.ok = False

    @staticmethod
    def vector(rec: Record, *, conviction: float, price: float, days_to_end: float | None, is_sports: bool,
               category: str, crowd: int, on_month: bool, on_all: bool) -> list[float]:
        n = rec.closed_n
        pf = min(rec.profit_factor, 5.0) if n else 1.0
        cat_n, cat_wins = rec.cat_stats.get(category, (0, 0))
        cat_share = (cat_n / n) if n else 0.3
        cat_wr = (cat_wins / cat_n) if cat_n else 0.5
        return [pf, rec.win_rate if n else 0.5, math.log1p(n), max(-1.0, min(rec.return_on_cost if n else 0.0, 1.0)),
                math.log(max(0.05, min(conviction or 1.0, 20.0))), price, price ** 2,
                math.log1p(max(0.0, days_to_end if days_to_end is not None else 7.0)), 1.0 if is_sports else 0.0,
                cat_share, cat_wr, math.log1p(max(0, crowd)),
                rec.taker_share if rec.taker_share is not None else 0.5, math.log1p(rec.median_hold_hours or 24.0),
                1.0 if on_month else 0.0, 1.0 if on_all else 0.0]

    def predict(self, x: list[float]) -> tuple[float, float]:
        """(predicted return on cost, predicted win probability)."""
        z = [(v - m) / s for v, m, s in zip(x, self.mean, self.std)] + [1.0]
        ret = sum(a * b for a, b in zip(z, self.w_ret))
        logit = sum(a * b for a, b in zip(z, self.w_win))
        return ret, 1 / (1 + math.exp(-max(-30, min(30, logit))))
