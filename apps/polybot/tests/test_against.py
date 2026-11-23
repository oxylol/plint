"""against=1: a setup that takes the other side of every signal its board gives, at that side's own real price."""
import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace as NS

from polybot.store import Store
from polybot.strategy_consensus import ConsensusParams, ConsensusStrategy, Leader
from tests.test_round3 import NOW, FakeCore, FakeFeed, buy, market

CFG = NS(window="week", category="overall", top_n=25, min_agree=5, lookback_hours=24, min_leader_usd=Decimal(25), stake_usd=Decimal(5),
         max_price=Decimal("0.9"), min_price=Decimal("0.05"), max_above_leaders=Decimal("0.05"), max_open=20, daily_budget_usd=Decimal(50),
         cooldown_hours=72, min_hours_to_end=1, follow_sells=True, poll_seconds=300, cr_pool=200)
WALLETS = ["0xa", "0xb", "0xc", "0xd"]


class Book(FakeCore):
    """A book with its own ask per outcome (None: nobody is selling)."""
    def __init__(self, store, markets, asks):
        super().__init__(store, markets)
        self.asks = asks

    def estimate(self, asset_id, side, amount=None, shares=None):
        return self.asks.get(asset_id)


class Board(FakeFeed):
    def leaders(self, window, category, top_n, now):
        return [Leader(w, w, i + 1, Decimal(1000), Decimal(5000)) for i, w in enumerate(WALLETS)]


def setup(d, asks, trades=None, **over):
    store = Store(d)
    p = ConsensusParams.base(CFG).variant("antimonth", {"window": "month", "min_agree": "4", "against": "1", **over})
    # four leaders bought Yes at 58¢ within the hour
    trades = trades if trades is not None else [buy(w, "far", 0, 30, price=0.58) for w in WALLETS]
    s = ConsensusStrategy(Book(store, {"far": market("far", "Q far?", "other-event")}, asks), p, Board(trades), None)
    return store, s


class Against(unittest.TestCase):
    def test_buys_the_other_outcome_at_its_own_ask(self):
        with tempfile.TemporaryDirectory() as d:
            store, s = setup(d, {"far-y": Decimal("0.60"), "far-n": Decimal("0.43")})
            self.assertFalse(s.p.follow_sells)                            # held to the end, whatever the leaders do later
            s.run(NOW)
            held = store.open_positions("antimonth")
            self.assertEqual([(h["asset_id"], h["outcome"], h["price"]) for h in held], [("far-n", "No", 0.43)])
            self.assertEqual((held[0]["against"], held[0]["their_price"], held[0]["agree"]), ("Yes", 0.6, 4))
            self.assertEqual(held[0]["agree_wallets"], [])               # the leaders are not credited with this trade
            rows = store.ledger_rows()
            self.assertEqual([(r["asset_id"], r["outcome"], r["price"], r["against"], r["their_price"], r["taken"]) for r in rows],
                             [("far-n", "No", 0.43, "Yes", 0.6, True)])
            self.assertIn("paper bought the other side: No at 0.43", s.decisions["BUY:far-y"])
            self.assertTrue(s.status()["signals"][0]["holding"])
            s.run(NOW + timedelta(minutes=5))                             # the next look: nothing twice
            self.assertEqual((len(store.open_positions("antimonth")), len(store.ledger_rows())), (1, 1))
            self.assertEqual(s.decisions["BUY:far-y"], "already holding")

    def test_the_boards_own_price_rules_still_decide_what_is_a_signal(self):
        with tempfile.TemporaryDirectory() as d:
            # Yes now costs 70¢, the leaders paid 58¢: month4 would call this too late, so there is nothing to be against
            store, s = setup(d, {"far-y": Decimal("0.70"), "far-n": Decimal("0.33")})
            s.run(NOW)
            self.assertIn("too late", s.decisions["BUY:far-y"])
            self.assertEqual((store.open_positions("antimonth"), store.ledger_rows()), ([], []))

    def test_no_seller_on_the_other_side(self):
        with tempfile.TemporaryDirectory() as d:
            store, s = setup(d, {"far-y": Decimal("0.60")})
            s.run(NOW)
            self.assertEqual(s.decisions["BUY:far-y"], "not enough liquidity on the other side")
            self.assertEqual((store.open_positions("antimonth"), store.ledger_rows()), ([], []))

    def test_logged_at_the_other_sides_price_even_when_the_book_is_full(self):
        with tempfile.TemporaryDirectory() as d:
            store, s = setup(d, {"far-y": Decimal("0.60"), "far-n": Decimal("0.43")}, max_open="0")
            s.run(NOW)
            rows = store.ledger_rows()
            self.assertEqual([(r["asset_id"], r["price"], r["taken"], r["why_not"]) for r in rows], [("far-n", 0.43, False, "max open positions reached")])
            self.assertEqual(store.open_positions("antimonth"), [])
            s.run(NOW + timedelta(minutes=5))
            self.assertEqual(len(store.ledger_rows()), 1)

    def test_when_the_board_turns_nothing_is_sold_and_the_market_is_not_held_twice(self):
        with tempfile.TemporaryDirectory() as d:
            store, s = setup(d, {"far-y": Decimal("0.60"), "far-n": Decimal("0.43")})
            s.run(NOW)
            later = NOW + timedelta(hours=2)
            # all four now buy No, twice as large: the board has turned to the side this setup holds
            s.feed.t = s.feed.t + [buy(w, "far", 1, -90, usd=400.0, price=0.45) for w in WALLETS]
            s.core.asks["far-n"] = Decimal("0.47")
            s.run(later)
            self.assertEqual([h["asset_id"] for h in store.open_positions("antimonth")], ["far-n"])    # kept, and no Yes beside it
            rows = store.ledger_rows()                                    # the turned signal is still measured
            self.assertEqual([(r["asset_id"], r["against"], r["taken"]) for r in rows], [("far-n", "Yes", True), ("far-y", "No", False)])
            self.assertEqual(rows[1]["why_not"], "already holding this market's other outcome")

    def test_only_a_plain_board_setup_can_be_mirrored(self):
        for extra in ({"pack": "1"}, {"model": "1"}, {"fresh_only": "1"}):
            with self.assertRaises(ValueError, msg=str(extra)):
                ConsensusParams.base(CFG).variant("x", {"against": "1", **extra})


if __name__ == "__main__":
    unittest.main()
