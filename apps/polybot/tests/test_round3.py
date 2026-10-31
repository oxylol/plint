"""The round-3 setup end to end on fake market data: it scores fresh single-leader bets at the price a buy would pay,
holds at most one position per event, and ignores bets that are older than its window."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS

from polybot import features as F
from polybot.consensus import LeaderTrade
from polybot.core import Core
from polybot.credibility import Judgement, Record
from polybot.pack import Member
from polybot.store import Store
from polybot.strategy_consensus import ConsensusParams, ConsensusStrategy

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def market(cid, question, event):
    yes = NS(label="Yes", token_id=f"{cid}-y", position_id=None, price=Decimal("0.6"))
    no = NS(label="No", token_id=f"{cid}-n", position_id=None, price=Decimal("0.4"))
    return NS(condition_id=cid, question=question, slug=cid, version="v1", outcomes=NS(yes=yes, no=no), events=[NS(slug=event)],
              sports=None, trading=NS(minimum_order_size=Decimal(1), fees_enabled=False, fee_schedule=None),
              state=NS(closed=False, archived=False, active=True, accepting_orders=True, enable_order_book=True, neg_risk=True,
                       end_date=NOW + timedelta(days=4)))


class FakeCore:
    mode = "paper"
    balance = None

    def __init__(self, store, markets):
        self.store, self.markets = store, markets
        self.notifier = NS(send=lambda *a, **k: None)

    def market(self, cid, slug=""):
        return self.markets.get(cid)

    def estimate(self, asset_id, side, amount=None, shares=None):
        return Decimal("0.40")

    def fee_rate(self, m):
        return Decimal(0)

    resolve_asset = Core.resolve_asset
    trading_id = Core.trading_id
    not_tradeable = Core.not_tradeable
    end_of = Core.end_of

    def buy(self, asset_id, amount, est, what, fee_rate=Decimal(0)):
        return {"cost": float(amount), "shares": float(amount / est), "price": float(est), "fee": 0.0}

    def warn(self, *a, **k):
        pass

    def record_error(self, *a, **k):
        pass

    def midpoint(self, asset_id):
        return Decimal("0.40")


class FakeFeed:
    def __init__(self, trades):
        self.t = trades

    def trades(self, leaders, since, now):
        return {l.wallet: [t for t in self.t if t.wallet == l.wallet and t.timestamp >= since] for l in leaders}


class FakePack:
    building = False
    refreshed_at = NOW
    key = ("month", "overall", 200)

    def __init__(self, members):
        self.m = members

    def ensure_fresh(self, now):
        pass

    def everyone(self):
        return self.m


def member(wallet):
    rec = Record(wallet=wallet, name=wallet, closed_n=50, fills_7d=20, age_days=200, taker_share=0.5)
    pos = [F.Pos(cost=100, avg=0.4, final=1.0 if i % 3 else 0.0, category="politics", first=NOW.timestamp() - 86400 * (i + 2),
                 last=NOW.timestamp() - 86400 * (i + 1), resolved=None) for i in range(30)]
    rec.rec3 = F.record(pos, NOW.timestamp())
    return Member(wallet, wallet, 1, rec, Judgement(True, 0.5))


def buy(wallet, cid, outcome_idx, minutes_ago, usd=200.0, price=0.38):
    asset = f"{cid}-{'y' if outcome_idx == 0 else 'n'}"
    return LeaderTrade(wallet=wallet, condition_id=cid, asset_id=asset, side="BUY", size=Decimal(str(usd / price)), price=Decimal(str(price)),
                       timestamp=NOW - timedelta(minutes=minutes_ago), title=f"Q {cid}?", slug=cid, event_slug="texas-senate",
                       outcome="Yes" if outcome_idx == 0 else "No", outcome_index=outcome_idx, tx=f"{wallet}{cid}{minutes_ago}")


class Round3Setup(unittest.TestCase):
    def test_one_position_per_event_and_fresh_bets_only(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "m.json"
            # one stump: a leader with a positive record edge adds +2 to the market's log-odds
            model.write_text(json.dumps({"kind": "lgbm", "version": 3, "anchor": "logit_p", "features": F.FEATURES, "init": 0.0, "lr": 1.0,
                                         "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [F.FEATURES.index("rec_edge"), -2, -2],
                                                    "threshold": [0.0, -2, -2], "value": [0.0, -2.0, 2.0]}],
                                         "policy": {"threshold": 0.1, "per_day": 10, "min_price": 0.05, "max_price": 0.95, "min_hours": 1,
                                                    "max_hours": 0, "per_event": 1, "leader_cap": 0.25, "max_signal_age_min": 15}}))
            store = Store(d)
            mk = {c: market(c, f"Q {c}?", "texas-senate") for c in ("rep", "dem", "old")}
            mk["old"].events = [NS(slug="other-event")]
            # 0xc's bet is two hours old; 0xd's older bet on the Democrat's Yes puts it against 0xb's No, which must not
            # veto 0xb for this setup (each leader is judged on their own)
            trades = [buy("0xa", "rep", 1, 6), buy("0xb", "dem", 1, 5), buy("0xc", "old", 0, 120), buy("0xd", "dem", 0, 120, usd=50.0, price=0.6)]
            cfg = NS(window="month", category="overall", top_n=25, min_agree=1, lookback_hours=48, min_leader_usd=Decimal(10),
                     stake_usd=Decimal(5), max_price=Decimal("0.9"), min_price=Decimal("0.05"), max_above_leaders=Decimal("0.05"), max_open=20,
                     daily_budget_usd=Decimal(50), cooldown_hours=72, min_hours_to_end=1, follow_sells=True, poll_seconds=300, cr_pool=200)
            p = ConsensusParams.base(cfg).variant("edge", {"pack": "1", "model": "1", "model_file": str(model), "max_per_cycle": "3"})
            s = ConsensusStrategy(FakeCore(store, mk), p, FakeFeed(trades), FakePack([member(w) for w in ("0xa", "0xb", "0xc", "0xd")]))
            self.assertTrue(s.round3)
            self.assertEqual((s.p.min_show, s.p.per_event, s.p.max_signal_age_min, s.p.max_above_leaders), (1, 1, 15.0, Decimal(1)))
            s.run(NOW)
            held = store.open_positions("edge")
            self.assertEqual(len(held), 1)                       # No on one side of the race, never both
            self.assertEqual(held[0]["event_key"], "texas-senate")
            self.assertIn(held[0]["leader"], ("0xa", "0xb"))
            self.assertGreater(held[0]["predicted"], 0.1)       # sigmoid(logit(0.40) + 2) / 0.40 − 1 ≈ 0.98
            self.assertIn("older than", s.decisions["BUY:old-y"])   # 0xc's bet is two hours old
            other = "BUY:dem-n" if held[0]["leader"] == "0xa" else "BUY:rep-n"
            self.assertEqual(s.decisions[other], "already holding this event")
            self.assertEqual(next(x for x in s.signals if x.key == "BUY:dem-n").disagree, ["0xd"])


class ConsensusSetup(unittest.TestCase):
    def test_policy_from_the_model_file(self):
        cfg = NS(window="day", category="overall", top_n=25, min_agree=5, lookback_hours=24, min_leader_usd=Decimal(25), stake_usd=Decimal(5),
                 max_price=Decimal("0.9"), min_price=Decimal("0.05"), max_above_leaders=Decimal("0.05"), max_open=20, daily_budget_usd=Decimal(50),
                 cooldown_hours=72, min_hours_to_end=1, follow_sells=True, poll_seconds=300, cr_pool=200)
        p = ConsensusParams.base(cfg).variant("day4model", {"window": "day", "model": "1", "model_file": "model_day4.json", "follow_sells": "0"})
        with tempfile.TemporaryDirectory() as d:
            s = ConsensusStrategy(FakeCore(Store(d), {}), p, FakeFeed([]), None)
        self.assertTrue(s.consensus3 and not s.round3)
        self.assertEqual((s.p.min_agree, s.p.min_show, s.p.follow_sells, s.p.max_above_leaders, s.p.min_price, s.p.max_price),
                         (4, 2, False, Decimal("0.05"), Decimal("0.05"), Decimal("0.9")))
        self.assertAlmostEqual(s.p.min_pred, -0.02)


if __name__ == "__main__":
    unittest.main()
