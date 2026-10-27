import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from polybot.consensus import LeaderTrade, find_signals

T0 = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
MIN = Decimal(25)


def trade(wallet, side, asset="A", cid="c1", size="100", price="0.40", minutes=0, outcome=None):
    return LeaderTrade(wallet=wallet, condition_id=cid, asset_id=asset, side=side, size=Decimal(size),
                       price=Decimal(price), timestamp=T0 + timedelta(minutes=minutes), title="Q?",
                       slug="q", event_slug="e", outcome=outcome or ("Yes" if asset == "A" else "No"),
                       outcome_index=0 if asset == "A" else 1, tx=f"0x{wallet}{minutes}{side}{asset}")


class ConsensusTests(unittest.TestCase):
    def test_counts_distinct_net_buyers(self):
        trades = [trade("w1", "BUY"), trade("w1", "BUY", minutes=1), trade("w2", "BUY"), trade("w3", "BUY", price="0.50")]
        sigs = find_signals(trades, min_leader_usd=MIN)
        self.assertEqual(len(sigs), 1)
        s = sigs[0]
        self.assertEqual((s.side, s.score, s.agree, s.disagree), ("BUY", 3, ["w1", "w2", "w3"], []))
        # size-weighted average of the agreeing buys: (80+40+50) USD over 400 shares
        self.assertEqual(s.avg_price.quantize(Decimal("0.0001")), Decimal("0.4250"))
        self.assertEqual(s.net_usd, Decimal(170))

    def test_round_trips_are_noise(self):
        trades = [trade("w1", "BUY"), trade("w1", "SELL", minutes=5, price="0.41"), trade("w2", "BUY")]
        sigs = find_signals(trades, min_leader_usd=MIN, min_show=1)
        self.assertEqual([(s.side, s.agree) for s in sigs], [("BUY", ["w2"])])

    def test_other_outcome_counts_against(self):
        trades = [trade("w1", "BUY"), trade("w2", "BUY"), trade("w3", "BUY", asset="B", price="0.60")]
        sigs = find_signals(trades, min_leader_usd=MIN, min_show=1)
        yes = next(s for s in sigs if s.asset_id == "A")
        no = next(s for s in sigs if s.asset_id == "B")
        self.assertEqual((yes.agree, yes.disagree, yes.score), (["w1", "w2"], ["w3"], 1))
        self.assertEqual((no.agree, no.disagree, no.score), (["w3"], ["w1", "w2"], -1))

    def test_market_makers_buying_both_sides_are_flat(self):
        trades = []
        for w in ("m1", "m2", "m3", "m4"):
            trades += [trade(w, "BUY", asset="A", price="0.46"), trade(w, "BUY", asset="B", price="0.52", minutes=1)]
        trades.append(trade("w5", "BUY", asset="A", price="0.46"))
        sigs = find_signals(trades, min_leader_usd=MIN, min_show=1)
        self.assertEqual([(s.side, s.asset_id, s.agree, s.disagree) for s in sigs], [("BUY", "A", ["w5"], [])])

    def test_sell_signal_only_for_held_outcomes(self):
        trades = [trade("w1", "SELL"), trade("w2", "SELL", minutes=2), trade("w3", "BUY"), trade("w4", "BUY", asset="B", price="0.60")]
        self.assertEqual([s.side for s in find_signals(trades, min_leader_usd=MIN, min_show=1)], ["BUY", "BUY"])
        sigs = find_signals(trades, min_leader_usd=MIN, min_show=1, held={"A"})
        sell = next(s for s in sigs if s.side == "SELL")
        # w4 bought the other side, which is as good as selling A
        self.assertEqual((sell.asset_id, sell.agree, sell.disagree, sell.score), ("A", ["w1", "w2", "w4"], ["w3"], 2))
        self.assertEqual(sell.last_at, T0 + timedelta(minutes=2))
        self.assertEqual(sell.avg_price, Decimal("0.40"))
        by_label = find_signals(trades, min_leader_usd=MIN, min_show=1, held={("c1", "yes")})
        self.assertTrue(any(s.side == "SELL" for s in by_label))

    def test_small_fills_ignored_and_dedupe_key(self):
        trades = [trade("w1", "BUY", size="10"), trade("w2", "BUY", size="10")]
        self.assertEqual(find_signals(trades, min_leader_usd=MIN, min_show=1), [])
        self.assertNotEqual(trades[0].key, trades[1].key)

    def test_sorted_strongest_first(self):
        trades = [trade(f"w{i}", "BUY") for i in range(5)] + [trade(f"v{i}", "BUY", asset="C", cid="c2") for i in range(2)]
        sigs = find_signals(trades, min_leader_usd=MIN)
        self.assertEqual([s.score for s in sigs], [5, 2])


if __name__ == "__main__":
    unittest.main()
