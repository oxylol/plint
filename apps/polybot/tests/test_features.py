"""Round-3 features: the record, the signal, and proof that research and bot compute them identically."""
import datetime as dt
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

from polybot import features as F  # noqa: E402
from polybot.consensus import LeaderTrade, Signal  # noqa: E402
from polybot.credibility import Record  # noqa: E402
from polybot.pack import positions_from_api  # noqa: E402
from polybot.strategy_consensus import leader_fills, signal_vector  # noqa: E402
import common3  # noqa: E402  (research/common3.py: the builder's pure parts)

U = dt.timezone.utc
NOW = dt.datetime(2026, 10, 30, 12, 0, tzinfo=U)
DAY = 86400


def pos(final, avg=0.4, cost=100.0, cat="politics", last_days=5, resolved_days=None, is_no=False, neg_risk=False):
    last = NOW.timestamp() - last_days * DAY
    return F.Pos(cost=cost, avg=avg, final=final, category=cat, first=last - 3600 * 10, last=last,
                 resolved=None if resolved_days is None else NOW.timestamp() - resolved_days * DAY, is_no=is_no, neg_risk=neg_risk)


class RecordTests(unittest.TestCase):
    def test_numbers(self):
        r = F.record([pos(1.0, 0.4), pos(0.0, 0.5), pos(1.0, 0.8, cat="sports", last_days=40)], NOW.timestamp())
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["wr"], 2 / 3)
        self.assertAlmostEqual(r["wr_shrunk"], 4 / 7)
        # P&L at resolution: +150, −100, +25
        self.assertAlmostEqual(r["pf"], 175 / 100)
        self.assertAlmostEqual(r["roi"], 75 / 300)
        self.assertAlmostEqual(r["edge"], (0.6 - 0.5 + 0.2) / 3)
        self.assertEqual(r["n30"], 2)               # the sports one is 40 days old
        self.assertAlmostEqual(r["pnl30"], 50.0)
        self.assertEqual(r["cats"]["sports"][:2], (1, 1.0))
        self.assertAlmostEqual(r["cats"]["sports"][2], 0.2)
        self.assertAlmostEqual(r["median_hold_h"], 10.0)

    def test_what_is_left_out(self):
        keep = pos(1.0)
        r = F.record([keep, pos(None), pos(0.0, cost=0.5), pos(0.0, is_no=True, neg_risk=True),
                      pos(0.0, resolved_days=-1), pos(1.0, last_days=-1)], NOW.timestamp())
        self.assertEqual(r["n"], 1)                 # unresolved, dust, negative-risk No, resolved later, later activity
        self.assertEqual(F.record([], NOW.timestamp()), {"n": 0})

    def test_newest_first(self):
        old = [pos(0.0, last_days=100 + i) for i in range(10)]
        new = [pos(1.0, last_days=1 + i) for i in range(5)]
        self.assertEqual(F.record(old + new, NOW.timestamp(), n_max=5)["wr"], 1.0)

    def test_vector_keys(self):
        x = F.vector(F.record([pos(1.0)] * 6, NOW.timestamp()), price=0.4, leader_price=0.35, stance_usd=200, hours_to_end=30,
                     category="politics", is_no=False, neg_risk=False, n_fills=3, minutes_since_first=12, game_minutes=None,
                     age_days=100, taker_share=0.6)
        self.assertEqual(list(x), F.FEATURES)
        self.assertAlmostEqual(x["drift"], 0.05)
        self.assertEqual((x["cat_politics"], x["cat_weather"], x["in_game"]), (1.0, 0.0, 0.0))

    def test_leader_stats(self):
        t = NOW.timestamp()
        fills = [(t - 600, "A", 1, 100, 0.40), (t - 500, "B", 1, 50, 0.60),    # buys both sides: B nets out $30 of A
                 (t - 400, "A", 1, 100, 0.50), (t - 60 * 60 * 49, "A", 1, 1000, 0.3)]   # the last is outside 48 h
        ls = F.leader_stats(fills, "A", t)
        self.assertAlmostEqual(ls["stance"], 40 - 30 + 50)
        self.assertAlmostEqual(ls["leader_price"], 90 / 200)
        self.assertEqual(ls["n_fills"], 3)
        self.assertAlmostEqual(ls["minutes_since_first"], 10)
        self.assertEqual(ls["crossed_at"], t - 600)


class ResearchEqualsBot(unittest.TestCase):
    """The same positions and fills, described once by the API (what the bot sees) and once by the research data
    (collect3 rows + market metadata), must give the same record and the same model inputs."""

    def setUp(self):
        d = lambda days: NOW - dt.timedelta(days=days)  # noqa: E731
        self.api_rows = []
        self.markets = {}
        spec = [  # (cid, outcome, oi, avg, size, final, status, title, slug, nr, days ago)
            ("c1", "Yes", 0, 0.40, 250, 1.0, "CLOSED", "Will X win the election?", "x-election", False, 3),
            ("c2", "No", 1, 0.30, 100, 0.0, "REDEEMABLE", "Will the highest temperature in Paris be 20°C on October 1?", "paris-temp", True, 20),
            ("c3", "Yes", 0, 0.55, 80, 0.0, "REDEEMABLE", "Lakers vs. Celtics", "nba-lal-bos", False, 9),
            ("c4", "Team A", 0, 0.62, 300, 1.0, "CLOSED", "LoL: A vs B (BO3)", "lol-a-b", False, 1),
            ("c5", "Yes", 0, 0.20, 50, None, "CLOSED", "Will Y happen by December?", "y-dec", False, 2),   # sold, not resolved
        ]
        for cid, outcome, oi, avg, size, final, status, title, slug, nr, ago in spec:
            first, last = d(ago + 1), d(ago)
            end = d(ago - 0.5) if final is not None else NOW + dt.timedelta(days=40)
            price = final if final is not None else 0.35
            self.api_rows.append(NS(asset_id=f"{cid}-{oi}", condition_id=cid, outcome=outcome, outcome_index=oi, avg_price=Decimal(str(avg)),
                                    total_size=Decimal(str(size)), current_size=Decimal(0), current_price=Decimal(str(price)),
                                    realized_pnl=Decimal(0), total_pnl=Decimal(0), first_entry_at=first, last_event_at=last, end_date=end,
                                    negative_risk=nr, title=title, slug=slug, event_slug=slug, redeemable=status == "REDEEMABLE",
                                    status=status))
            prices = [None, None]
            if final is not None:
                prices[oi], prices[1 - oi] = final, 1 - final
            self.markets[cid] = common3.M({"closed": final is not None, "prices": prices, "end": end.isoformat(),
                                           "closed_time": (end + dt.timedelta(hours=1)).isoformat() if final is not None else None,
                                           "nr": nr, "ev": slug, "fee_on": True, "fee_rate": 0.05, "game_start": None})

    def test_record(self):
        bot = F.record(positions_from_api(self.api_rows, NOW), NOW.timestamp())
        w = {"closed": [common3.pos_row(p) for p in self.api_rows if p.status == "CLOSED"],
             "redeemable": [common3.pos_row(p) for p in self.api_rows if p.status == "REDEEMABLE"], "open": [], "at": NOW.isoformat()}
        research = F.record(common3.positions_of(w, self.markets), NOW.timestamp())
        self.assertEqual(bot["n"], 3)            # the negative-risk No and the unresolved sale are left out by both
        self.assertEqual(research, bot)

    def test_signal_vector(self):
        rec = Record(wallet="0xa", name="a", age_days=321.5, taker_share=0.42)
        rec.rec3 = F.record(positions_from_api(self.api_rows, NOW), NOW.timestamp())
        t = NOW.timestamp()
        gs = NOW - dt.timedelta(minutes=20)
        end = NOW + dt.timedelta(hours=2)
        raw = [(t - 1800, "tok-yes", "BUY", 120.0, 0.44), (t - 900, "tok-no", "BUY", 30.0, 0.57), (t - 300, "tok-yes", "BUY", 80.0, 0.47),
               (t - 120, "tok-yes", "SELL", 10.0, 0.49)]
        trades = [LeaderTrade(wallet="0xa", condition_id="cg", asset_id=a, side=s, size=Decimal(str(z)), price=Decimal(str(p)),
                              timestamp=dt.datetime.fromtimestamp(ts, U), title="Lakers vs. Celtics", slug="nba-lal-bos-1",
                              event_slug="nba-lal-bos", outcome="Lakers", outcome_index=0) for ts, a, s, z, p in raw]
        sig = Signal("BUY", "cg", "tok-yes", "Lakers", 0, "Lakers vs. Celtics", "nba-lal-bos-1", "nba-lal-bos", ["0xa"], [], None,
                     Decimal(0), NOW)
        m = NS(state=NS(end_date=end, neg_risk=False), sports=NS(game_start_time=gs), events=[NS(slug="nba-lal-bos")])
        price = 0.48
        ls = F.leader_stats(leader_fills(trades, "cg"), "tok-yes", t)
        bot = signal_vector(rec, ls, sig, m, price, t)
        # research: build3's fill rows and market metadata, then prep3's price terms
        fills = [(ts, a, 1 if s == "BUY" else -1, z, p) for ts, a, s, z, p in raw]
        mk = common3.M({"closed": False, "prices": [None, None], "end": end.isoformat(), "nr": False, "ev": "nba-lal-bos",
                        "fee_on": True, "fee_rate": 0.05, "game_start": gs.isoformat()})
        joined = t - 321.5 * DAY
        ctx, _, cat = common3.signal_context(rec.rec3, fills, "tok-yes", t, mk, "Lakers vs. Celtics", "nba-lal-bos-1", "nba-lal-bos",
                                             "Lakers", joined, 0.42, 0.44)
        research = {**F.price_terms(price, ctx["leader_price"]), **ctx}
        self.assertEqual(cat, "sports")
        self.assertEqual(list(research), F.FEATURES)
        for k in F.FEATURES:
            self.assertAlmostEqual(research[k], bot[k], places=9, msg=k)
        self.assertEqual(bot["in_game"], 1.0)


if __name__ == "__main__":
    unittest.main()
