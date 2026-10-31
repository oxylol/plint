import json
import math
import tempfile
import unittest
from pathlib import Path

from polybot.credibility import Record
from polybot.model import Model

REC = Record(wallet="0xa", name="a", closed_n=100, wins=60, losses=40, gross_win=300, gross_loss=100, sum_pnl=200, sum_cost=1000,
             median_hold_hours=48, median_cost=100, median_price=0.4, taker_share=0.7, cat_stats={"politics": (50, 35), "sports": (10, 4)})
CTX = dict(conviction=2.0, price=0.4, days_to_end=10, is_sports=False, category="politics", crowd=1, on_month=True, on_all=False, rank=12, is_no=True)


def write(d, path):
    path.write_text(json.dumps(d))
    return Model(path)


class ModelTests(unittest.TestCase):
    def test_vector(self):
        x = Model.vector(REC, **CTX)
        self.assertAlmostEqual(x["prior_pf"], 3.0)
        self.assertAlmostEqual(x["cat_share"], 0.5)
        self.assertAlmostEqual(x["cat_wr"], 0.7)
        self.assertAlmostEqual(x["prior_sports"], 0.1)
        self.assertEqual((x["is_no"], x["cat_politics"], x["cat_crypto"], x["on_month"]), (1.0, 1.0, 0.0, 1.0))

    def test_linear_gbm_rule_and_policy(self):
        with tempfile.TemporaryDirectory() as d:
            feats = ["prior_pf", "price"]
            lin = write({"kind": "linear", "target": "ridge", "features": feats, "mean": [0, 0], "std": [1, 1], "w": [0.1, 0.0, 0.05],
                         "policy": {"threshold": 0.2, "per_day": 3, "sports": False, "min_price": 0.15, "max_price": 0.75, "min_days": 0.5}}, Path(d) / "a.json")
            x = Model.vector(REC, **CTX)
            self.assertAlmostEqual(lin.score(x, 0.4), 0.35)
            self.assertEqual((lin.threshold, lin.policy["per_day"]), (0.2, 3))
            logit = write({"kind": "linear", "target": "logit", "features": feats, "mean": [0, 0], "std": [1, 1], "w": [0.0, 0.0, 0.0]}, Path(d) / "b.json")
            self.assertAlmostEqual(logit.score(x, 0.4), 0.1)  # 50% predicted vs 40¢
            # one stump: price <= 0.5 → +0.3 else −0.1; init 0.05, lr 0.5
            gbm = write({"kind": "gbm", "target": "gbm", "features": feats, "lr": 0.5, "init": 0.05,
                         "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [1, -2, -2], "threshold": [0.5, -2, -2], "value": [0, 0.3, -0.1]}]}, Path(d) / "c.json")
            self.assertAlmostEqual(gbm.score(x, 0.4), 0.05 + 0.5 * 0.3)
            self.assertAlmostEqual(gbm.score(Model.vector(REC, **{**CTX, "price": 0.9}), 0.9), 0.05 - 0.5 * 0.1)
            rule = write({"kind": "rule", "name": "credible", "features": feats}, Path(d) / "d.json")
            self.assertGreater(rule.score(x, 0.4), 0)
            old = write({"features": feats, "mean": [0, 0], "std": [1, 1], "return": [0.1, 0.0, 0.0], "win": [0, 0, 0]}, Path(d) / "e.json")
            self.assertTrue(old.ok and old.kind == "linear")
            self.assertFalse(Model(Path(d) / "missing.json").ok)


class Round3ModelTests(unittest.TestCase):
    """Round-3 files are anchored on the follower's price and score the expected return per dollar after the fee."""

    def x(self, price=0.4, edge=0.05, n=99):
        from polybot import features as F
        x = {k: 0.0 for k in F.FEATURES}
        x.update(F.price_terms(price, 0.38))
        x["rec_edge"], x["rec_n_log"] = edge, math.log1p(n)
        return x

    def test_kinds(self):
        from polybot import features as F
        with tempfile.TemporaryDirectory() as d:
            x = self.x()
            # trees: one stump on rec_edge, +1 to the log-odds above 0
            lg = write({"kind": "lgbm", "version": 3, "anchor": "logit_p", "features": F.FEATURES, "init": 0.0, "lr": 1.0,
                        "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [F.FEATURES.index("rec_edge"), -2, -2],
                                   "threshold": [0.0, -2, -2], "value": [0, -1.0, 1.0]}], "policy": {"threshold": 0.05}}, Path(d) / "l.json")
            p = 1 / (1 + math.exp(-(math.log(0.4 / 0.6) + 1.0)))
            self.assertAlmostEqual(lg.win_probability(x), p)
            self.assertAlmostEqual(lg.score(x, 0.4, fee_rate=0.05), p / 0.4 - 1 - 0.05 * 0.6)
            self.assertTrue(lg.round3 and lg.threshold == 0.05)
            # linear: one standardised input
            feats = ["rec_edge"]
            li = write({"kind": "linear", "version": 3, "anchor": "logit_p", "features": feats, "mean": [0.0], "std": [0.1], "w": [2.0, -0.5]},
                       Path(d) / "i.json")
            self.assertAlmostEqual(li.win_probability(x), 1 / (1 + math.exp(-(math.log(0.4 / 0.6) + 2.0 * 0.5 - 0.5))))
            # skill: edge 0.05 over 99 positions, k 1 → shrink 0.99
            sk = write({"kind": "skill", "version": 3, "anchor": "logit_p", "features": F.FEATURES, "k": 1.0, "b": 2.0}, Path(d) / "s.json")
            self.assertAlmostEqual(sk.win_probability(x), 1 / (1 + math.exp(-(math.log(0.4 / 0.6) + 2.0 * 0.05 * 0.99 * 4))))
            self.assertIn("win chance", sk.describe(x, 0.4))


if __name__ == "__main__":
    unittest.main()
