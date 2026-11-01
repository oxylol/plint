"""The ledger: every signal that passes a setup's rules is logged once per cooldown at its real price, bought or not,
and settles to a return per signal per setup."""
import json
import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS

from polybot import features as F
from polybot import ledger as L
from polybot.store import Store
from polybot.strategy_consensus import ConsensusParams, ConsensusStrategy
from tests.test_round3 import NOW, FakeCore, FakeFeed, FakePack, buy, market, member


class Pure(unittest.TestCase):
    def test_return_after_fee(self):
        self.assertAlmostEqual(L.signal_return(0.5, 1.0, 0.05), 1.0 - 0.025)
        self.assertAlmostEqual(L.signal_return(0.4, 0.0, 0.0), -1.0)

    def test_settle_and_summarize(self):
        rows = [{"t": "2026-11-01T10:00:00", "strategy": "day4", "asset_id": "a", "price": 0.5, "fee_rate": 0, "taken": True},
                {"t": "2026-11-01T11:00:00", "strategy": "day4", "asset_id": "b", "price": 0.5, "fee_rate": 0, "taken": False},
                {"t": "2026-11-01T11:00:00", "strategy": "edge", "asset_id": "b", "price": 0.25, "fee_rate": 0, "taken": True},
                {"t": "2026-11-02T11:00:00", "strategy": "edge", "asset_id": "c", "price": 0.5, "fee_rate": 0, "taken": True}]
        s = L.settle(rows, {"a": 1.0, "b": 0.0})
        self.assertEqual([r["ret"] for r in s], [1.0, -1.0, -1.0])            # c has not resolved
        out = {r["setup"]: r for r in L.summarize(s, rows, base="day4")}
        self.assertEqual((out["day4"]["logged"], out["day4"]["resolved"], out["day4"]["mean"]), (2, 2, 0.0))
        self.assertEqual((out["day4"]["taken_n"], out["day4"]["taken_mean"]), (1, 1.0))
        self.assertEqual((out["edge"]["logged"], out["edge"]["resolved"]), (2, 1))
        self.assertIn("ledger: 4 signals logged, 3 resolved", L.report(rows, {"a": 1.0, "b": 0.0}))

    def test_finals_from_markets(self):
        def mk(closed, py):
            return NS(state=NS(closed=closed), outcomes=NS(yes=NS(token_id="y", position_id=None, price=py),
                                                         no=NS(token_id="n", position_id=None, price=None if py is None else 1 - py)))
        rows = [{"condition_id": "c1", "slug": "s1", "asset_id": "y"}]
        self.assertEqual(L.finals_from_markets(rows, lambda c, s: mk(True, 1.0)), {"y": 1.0, "n": 0.0})
        self.assertEqual(L.finals_from_markets(rows, lambda c, s: mk(False, 1.0)), {})        # still open
        self.assertEqual(L.finals_from_markets(rows, lambda c, s: mk(True, 0.5)), {})         # closed, not resolved yet


class LoggedEvenWhenNotBought(unittest.TestCase):
    def test_blocked_signals_are_logged_once_with_the_reason(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "m.json"
            model.write_text(json.dumps({"kind": "lgbm", "version": 3, "anchor": "logit_p", "features": F.FEATURES, "init": 0.0, "lr": 1.0,
                                         "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [F.FEATURES.index("rec_edge"), -2, -2],
                                                    "threshold": [0.0, -2, -2], "value": [0.0, -2.0, 2.0]}],
                                         "policy": {"threshold": 0.1, "per_day": 10, "min_price": 0.05, "max_price": 0.95, "min_hours": 1,
                                                    "max_hours": 0, "per_event": 1, "leader_cap": 0.25, "max_signal_age_min": 15}}))
            store = Store(d)
            mk = {c: market(c, f"Q {c}?", "texas-senate") for c in ("rep", "dem", "far")}
            mk["far"].events = [NS(slug="other-event")]
            trades = [buy("0xa", "rep", 1, 6), buy("0xb", "dem", 1, 5), buy("0xc", "far", 1, 4)]
            cfg = NS(window="month", category="overall", top_n=25, min_agree=1, lookback_hours=48, min_leader_usd=Decimal(10),
                     stake_usd=Decimal(5), max_price=Decimal("0.9"), min_price=Decimal("0.05"), max_above_leaders=Decimal("0.05"), max_open=20,
                     daily_budget_usd=Decimal(5), cooldown_hours=72, min_hours_to_end=1, follow_sells=True, poll_seconds=300, cr_pool=200)
            p = ConsensusParams.base(cfg).variant("edge", {"pack": "1", "model": "1", "model_file": str(model), "max_per_cycle": "3"})
            core = FakeCore(store, mk)
            s = ConsensusStrategy(core, p, FakeFeed(trades), FakePack([member(w) for w in ("0xa", "0xb", "0xc")]))
            s.run(NOW)
            self.assertEqual(len(store.open_positions("edge")), 1)            # a $5 budget buys one
            rows = store.ledger_rows()
            self.assertEqual(len(rows), 3)                                   # all three passed the rules
            self.assertEqual(sum(r["taken"] for r in rows), 1)
            self.assertEqual({r["why_not"] for r in rows if not r["taken"]} - {"already holding this event", "daily budget used up"}, set())
            self.assertIn("daily budget used up", {r["why_not"] for r in rows})
            self.assertTrue(all(r["price"] == 0.4 for r in rows))
            s.run(NOW + timedelta(minutes=5))                                # the next cycle: no duplicates
            self.assertEqual(len(store.ledger_rows()), 3)

            def no_lookup(*a, **k):
                raise AssertionError("looked the market up again")
            core.market = no_lookup                                          # logged and over budget: skipped before any API call
            for sig in s.signals:
                if not store.position(sig.asset_id, "edge"):
                    self.assertIsNone(s.consider_buy(sig, NOW + timedelta(minutes=10)))
                    self.assertEqual(s.decisions[sig.key], "daily budget used up")


    def test_skipped_price_band(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(d)
            mk = {"far": market("far", "Q far?", "other-event")}
            cfg = NS(window="day", category="overall", top_n=25, min_agree=4, lookback_hours=24, min_leader_usd=Decimal(25),
                     stake_usd=Decimal(5), max_price=Decimal("0.9"), min_price=Decimal("0.05"), max_above_leaders=Decimal("0.05"), max_open=20,
                     daily_budget_usd=Decimal(50), cooldown_hours=72, min_hours_to_end=1, follow_sells=True, poll_seconds=300, cr_pool=200)
            p = ConsensusParams.base(cfg).variant("day4x", {"window": "day", "min_agree": "4", "skip_lo": "0.30", "skip_hi": "0.50"})
            self.assertEqual((p.skip_lo, p.skip_hi), (Decimal("0.30"), Decimal("0.50")))
            s = ConsensusStrategy(FakeCore(store, mk), p, FakeFeed([]), None)
            sig = NS(asset_id="far-n", condition_id="far", slug="far", title="Q far?", outcome="No", outcome_index=1, event_slug="other-event",
                     key="BUY:far-n", side="BUY", avg_price=Decimal("0.39"), agree=["0xa"] * 4, disagree=[])
            self.assertIsNone(s.consider_buy(sig, NOW))                      # the fake book prices it at 0.40
            self.assertIn("skipped 0.30–0.50 band", s.decisions["BUY:far-n"])
            self.assertEqual((store.open_positions("day4x"), store.ledger_rows()), ([], []))


if __name__ == "__main__":
    unittest.main()
