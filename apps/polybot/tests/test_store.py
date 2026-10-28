import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from polybot.store import Store, pkey

T0 = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def rec(strategy, asset="A", cost=5.0, shares=10.0, price=0.5):
    return {"strategy": strategy, "side": "BUY", "asset_id": asset, "condition_id": "c1", "title": "Q?", "outcome": "Yes",
            "cost": cost, "shares": shares, "price": price, "mode": "paper"}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_strategies_can_hold_the_same_outcome(self):
        self.store.open_position(rec("consensus"), T0)
        self.store.open_position(rec("month", cost=10.0, shares=20.0), T0)
        self.assertEqual(len(self.store.open_positions()), 2)
        self.assertEqual(self.store.position("A", "month")["cost"], 10.0)
        self.store.update_price("A", Decimal("0.6"))
        self.assertEqual([p["current_price"] for p in self.store.positions_for_asset("A")], [0.6, 0.6])
        closed = self.store.close_position("A", Decimal("0.6"), Decimal("12"), "test", T0, strategy="month")
        self.assertEqual((closed["strategy"], closed["pnl"]), ("month", 2.0))
        self.assertIsNotNone(self.store.position("A", "consensus"))
        self.assertIsNone(self.store.position("A", "month"))
        self.assertEqual(self.store.totals(T0, "month")["realized"], 2.0)
        self.assertEqual(self.store.totals(T0, "consensus")["open"], 1)
        self.assertEqual(self.store.totals(T0)["unrealized"], 1.0)
        self.assertEqual(self.store.acted_at("A", "month"), T0)
        self.assertIsNone(self.store.acted_at("B", "month"))

    def test_follow_stats_and_trust(self):
        r = rec("smart"); r["agree_wallets"] = ["0xa", "0xb"]; r["agree_names"] = {"0xa": "alice"}
        self.store.open_position(r, T0)
        self.store.close_position("A", Decimal("0.7"), Decimal("7"), "test", T0, strategy="smart")
        rows = self.store.leaders_followed("smart")
        self.assertEqual([(x["wallet"], x["name"], x["follows"], x["wins"], x["pnl"], x["trust"]) for x in rows],
                         [("0xa", "alice", 1, 1, 2.0, 1.1), ("0xb", "", 1, 1, 2.0, 1.1)])
        self.assertEqual(self.store.trust("smart", "0xa", 0.9), 1.1)
        self.assertEqual(self.store.trust("smart", "0xz", 0.9), 0.9)
        r = rec("smart", asset="B"); r["agree_wallets"] = ["0xa"]
        self.store.open_position(r, T0)
        self.store.close_position("B", Decimal("0.4"), Decimal("4"), "test", T0, strategy="smart")
        self.assertAlmostEqual(self.store.trust("smart", "0xa", 1.0), 0.88)
        self.assertEqual(self.store.leaders_followed("consensus"), [])

    def test_old_state_is_migrated(self):
        path = Path(self.tmp.name) / "state.json"
        path.write_text(json.dumps({"positions": {"A": rec("underdog")}, "acted": {"A": "2026-10-28T10:00:00+00:00", "B": "2026-10-28T11:00:00+00:00"},
                                    "closed": [], "trades": []}))
        store = Store(self.tmp.name)
        self.assertIn(pkey("underdog", "A"), store.state["positions"])
        self.assertIsNotNone(store.acted_at("B", "consensus"))
        self.assertIsNotNone(store.acted_at("B", "underdog"))
        self.assertIsNone(store.acted_at("B", "month"))


if __name__ == "__main__":
    unittest.main()
