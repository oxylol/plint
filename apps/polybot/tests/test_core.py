"""The market cache forgets: it used to keep every market it had looked up and the bot hit its memory limit daily."""
import unittest
from datetime import timedelta

from polybot.core import Core
from polybot.store import utcnow


def core_with(markets, pruned_ago=timedelta(minutes=2)):
    core = Core.__new__(Core)
    core._markets = markets
    core._markets_pruned = utcnow() - pruned_ago
    return core


class MarketCache(unittest.TestCase):
    def test_stale_markets_go_once_the_cache_is_big(self):
        now = utcnow()
        markets = {f"old{i}": (now - timedelta(minutes=30), object()) for i in range(Core.MARKET_KEEP)}
        markets.update({f"new{i}": (now, object()) for i in range(5)})
        core = core_with(markets)
        core.forget_stale_markets()
        self.assertEqual(sorted(core._markets), [f"new{i}" for i in range(5)])

    def test_a_small_cache_is_left_alone(self):
        now = utcnow()
        core = core_with({"old": (now - timedelta(hours=3), object()), "new": (now, object())})
        core.forget_stale_markets()
        self.assertEqual(len(core._markets), 2)

    def test_not_more_than_once_a_minute(self):
        now = utcnow()
        markets = {f"old{i}": (now - timedelta(minutes=30), object()) for i in range(Core.MARKET_KEEP + 1)}
        core = core_with(markets, pruned_ago=timedelta(seconds=5))
        core.forget_stale_markets()
        self.assertEqual(len(core._markets), Core.MARKET_KEEP + 1)


if __name__ == "__main__":
    unittest.main()
