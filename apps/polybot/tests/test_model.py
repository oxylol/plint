import json
import tempfile
import unittest
from pathlib import Path

from polybot.credibility import Record
from polybot.model import Model


class ModelTests(unittest.TestCase):
    def test_vector_and_predict(self):
        n = 16
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.json"
            path.write_text(json.dumps({"features": ["f"] * n, "mean": [0.0] * n, "std": [1.0] * n,
                                        "return": [0.1] + [0.0] * (n - 1) + [0.05], "win": [0.0] * n + [0.0],
                                        "cut_top10": 0.3, "cut_top25": 0.2, "trained_at": "2026-10-28T00:00:00+00:00"}))
            m = Model(path)
        self.assertTrue(m.ok)
        rec = Record(wallet="0xa", name="a", closed_n=100, wins=60, losses=40, gross_win=300, gross_loss=100, sum_pnl=200, sum_cost=1000,
                     median_hold_hours=48, median_cost=100, taker_share=0.7, cat_stats={"politics": (50, 35)})
        x = m.vector(rec, conviction=2.0, price=0.4, days_to_end=10, is_sports=False, category="politics", crowd=1, on_month=True, on_all=False)
        self.assertEqual(len(x), n)
        self.assertAlmostEqual(x[0], 3.0)            # profit factor
        self.assertAlmostEqual(x[9], 0.5)            # share of positions in this category
        self.assertAlmostEqual(x[10], 0.7)           # win rate in this category
        ret, win = m.predict(x)
        self.assertAlmostEqual(ret, 0.1 * 3.0 + 0.05)
        self.assertAlmostEqual(win, 0.5)
        self.assertFalse(Model(Path(d) / "missing.json").ok)


if __name__ == "__main__":
    unittest.main()
