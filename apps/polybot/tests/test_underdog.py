import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace as NS

from polybot.underdog import exit_reason, in_buy_window, one_per_match, pick_underdog, stop_price, target_price, team_from_question

T0 = datetime(2026, 10, 28, 19, 0, tzinfo=timezone.utc)


def market(yes=("Lakers", "0.42"), no=("Trail Blazers", "0.58"), mtype="moneyline", start=T0, version="v1",
           slug="nba-lal-por-2026-10-28", question="Lakers vs. Trail Blazers", cid="0xc", event=("Trail Blazers vs. Lakers", "nba-por-lal-2026-10-28", "e1")):
    def outcome(label, price, i):
        return NS(label=label, price=Decimal(price) if price is not None else None, token_id=f"tok{i}", position_id=f"pos{i}")
    return NS(
        condition_id=cid, slug=slug, question=question, version=version,
        outcomes=NS(yes=outcome(*yes, 0), no=outcome(*no, 1)),
        sports=NS(sports_market_type=mtype, game_start_time=start),
        metrics=NS(liquidity_num=Decimal("12000"), liquidity=None),
        events=[NS(title=event[0], slug=event[1], id=event[2])],
    )


def soccer(team, price, draw=False, cid=None):
    q = "Draw on 2026-10-28?" if draw else f"Will {team} win on 2026-10-28?"
    return market(yes=("Yes", price), no=("No", str(1 - Decimal(price))), slug=f"epl-ars-che-2026-10-28-{'draw' if draw else team[:3].lower()}",
                  question=q, cid=cid or f"0x{team[:3].lower()}", event=("Arsenal vs. Chelsea", "epl-ars-che-2026-10-28", "e9"))


def tid(m, o):
    return o.position_id if m.version == "v2" else o.token_id


class PickTests(unittest.TestCase):
    def test_cheaper_team_is_the_dog(self):
        c = pick_underdog(market(), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid)
        self.assertEqual((c.outcome, c.asset_id, c.kind, c.sport, str(c.price)), ("Lakers", "tok0", "teams", "nba", "0.42"))
        self.assertEqual(c.title, "Trail Blazers vs. Lakers")
        self.assertEqual(c.game_start, T0)

    def test_v2_uses_position_id(self):
        c = pick_underdog(market(version="v2"), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid)
        self.assertEqual(c.asset_id, "pos0")

    def test_yes_no_market_buys_yes_only_and_names_the_team(self):
        c = pick_underdog(soccer("Chelsea", "0.30"), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid)
        self.assertEqual((c.kind, c.outcome, c.asset_id, c.sport, c.title, c.is_draw), ("yesno", "Chelsea", "tok0", "epl", "Arsenal vs. Chelsea", False))
        d = pick_underdog(soccer("", "0.25", draw=True), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid)
        self.assertEqual((d.outcome, d.is_draw), ("Draw", True))
        m = market(yes=("Yes", "0.70"), no=("No", "0.30"))
        self.assertIsNone(pick_underdog(m, min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid))
        self.assertEqual(team_from_question("Will Real Madrid CF win on 2026-09-08?"), "Real Madrid CF")

    def test_one_underdog_per_soccer_match(self):
        band = dict(min_price=Decimal("0.001"), max_price=Decimal("0.999"), trading_id=tid)
        cands = [pick_underdog(m, **band) for m in (soccer("Arsenal", "0.45"), soccer("Chelsea", "0.30"), soccer("", "0.25", draw=True), market())]
        kept = one_per_match(cands)
        self.assertEqual([(c.title, c.outcome) for c in kept], [("Trail Blazers vs. Lakers", "Lakers"), ("Arsenal vs. Chelsea", "Chelsea")])
        even = [pick_underdog(m, **band) for m in (soccer("Arsenal", "0.35"), soccer("Chelsea", "0.35"), soccer("", "0.30", draw=True))]
        self.assertEqual(one_per_match(even), [])

    def test_rejects_outside_band_and_other_market_types(self):
        self.assertIsNone(pick_underdog(market(yes=("Lakers", "0.48"), no=("Trail Blazers", "0.52")), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid))
        self.assertIsNone(pick_underdog(market(yes=("Lakers", "0.05"), no=("Trail Blazers", "0.95")), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid))
        self.assertIsNone(pick_underdog(market(mtype="spreads"), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid))
        self.assertIsNone(pick_underdog(market(start=None), min_price=Decimal("0.10"), max_price=Decimal("0.45"), trading_id=tid))


class ExitTests(unittest.TestCase):
    def test_prices(self):
        self.assertEqual(target_price(Decimal("0.42"), Decimal(15), Decimal("0.01")), Decimal("0.49"))
        self.assertEqual(stop_price(Decimal("0.42"), Decimal(30), Decimal("0.01")), Decimal("0.29"))
        self.assertEqual(target_price(Decimal("0.90"), Decimal(15), Decimal("0.01")), Decimal("0.99"))

    def test_window(self):
        self.assertTrue(in_buy_window(T0, T0 - timedelta(minutes=2), 3, 2))
        self.assertTrue(in_buy_window(T0, T0 + timedelta(minutes=2), 3, 2))
        self.assertFalse(in_buy_window(T0, T0 - timedelta(minutes=4), 3, 2))
        self.assertFalse(in_buy_window(T0, T0 + timedelta(minutes=3), 3, 2))

    def test_exit_reasons(self):
        pos = {"target": "0.49", "stop": "0.29", "deadline": T0 + timedelta(minutes=30)}
        self.assertIsNone(exit_reason(pos, Decimal("0.45"), T0))
        self.assertEqual(exit_reason(pos, Decimal("0.49"), T0), "target")
        self.assertEqual(exit_reason(pos, Decimal("0.28"), T0), "stop")
        self.assertEqual(exit_reason(pos, Decimal("0.45"), T0 + timedelta(minutes=30)), "time")
        self.assertIsNone(exit_reason(pos, None, T0))


if __name__ == "__main__":
    unittest.main()
