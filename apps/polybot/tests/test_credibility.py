import unittest

from polybot.credibility import Record, Rules, conviction, judge


def good(**kw):
    base = dict(wallet="0xabc", name="good", pnl=100000, volume=1000000, closed_n=120, wins=70, losses=50, gross_win=150000, gross_loss=60000,
                sum_pnl=90000, sum_cost=400000, top_pnl=20000, median_hold_hours=100, median_cost=2000, fills_7d=60, taker_share=0.8,
                hours_since_fill=5, age_days=200)
    base.update(kw)
    return Record(**base)


class JudgeTests(unittest.TestCase):
    def test_good_trader_passes_with_a_score(self):
        j = judge(good())
        self.assertTrue(j.ok)
        self.assertEqual(j.reasons, [])
        self.assertTrue(0.5 < j.score <= 1.0)
        self.assertAlmostEqual(j.weight, 0.5 + j.score)

    def test_inactive_scalper_maker_and_whale_fail(self):
        self.assertIn("not trading right now", judge(good(hours_since_fill=200)).reasons)
        self.assertIn("not trading right now", judge(good(fills_7d=2)).reasons)
        self.assertTrue(any("in-game" in r for r in judge(good(median_hold_hours=3)).reasons))
        self.assertTrue(any("market maker" in r for r in judge(good(pnl=5000, volume=1000000)).reasons))
        self.assertTrue(any("market maker" in r for r in judge(good(taker_share=0.05, median_hold_hours=14)).reasons))
        self.assertIn("only 2 closed positions", judge(good(closed_n=2)).reasons)
        self.assertTrue(any("of the profit" in r for r in judge(good(top_pnl=80000)).reasons))
        self.assertTrue(any("profit factor" in r for r in judge(good(gross_win=50000, gross_loss=60000)).reasons))
        self.assertTrue(any("account" in r for r in judge(good(age_days=10)).reasons))
        self.assertEqual(judge(good(age_days=10)).weight, 0.0)
        self.assertTrue(any("hedger" in r for r in judge(good(two_sided_share=0.6)).reasons))

    def test_rules_are_adjustable(self):
        self.assertFalse(judge(good(median_hold_hours=3)).ok)
        self.assertTrue(judge(good(median_hold_hours=3), Rules(min_hold_hours=1)).ok)

    def test_conviction(self):
        self.assertEqual(conviction(2000, 2000), 1.0)
        self.assertEqual(conviction(10000, 2000), 2.0)
        self.assertEqual(conviction(100, 2000), 0.5)
        self.assertEqual(conviction(100, None), 1.0)


if __name__ == "__main__":
    unittest.main()
