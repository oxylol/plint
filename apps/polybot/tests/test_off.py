"""POLY_OFF: the setups named there do not run; their books stay on the page and what they hold is still priced."""
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest import mock

from polybot.config import Config, parse_off, parse_variants, DEFAULT_VARIANTS
from polybot.runner import Runner
from polybot.store import Store

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


def runner(d, off):
    with mock.patch.dict(os.environ, {"POLY_OFF": off, "DATA_DIR": d}, clear=True):
        cfg = Config.from_env()
    r = Runner(cfg, Store(d), NS(send=lambda *a, **k: None))
    ran = []
    for s in r.strategies:                      # no network: note who was asked to run, and stub the bookkeeping
        s.run = (lambda name: lambda now: ran.append(name))(s.name)
    r.core.settle = lambda now, positions: None
    r.marked = []
    r.core.mark = lambda positions: r.marked.extend(p["strategy"] for p in positions)
    return r, ran


class Off(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_off(" Underdog, month4 ,,"), frozenset({"underdog", "month4"}))
        self.assertEqual(parse_off(""), frozenset())
        self.assertEqual(parse_off("none"), frozenset())

    def test_the_mirror_of_the_month_board_is_built_in(self):
        v = dict(parse_variants(DEFAULT_VARIANTS))
        self.assertEqual((v["antimonth"]["window"], v["antimonth"]["min_agree"], v["antimonth"]["against"]), ("month", "4", "1"))

    def test_named_setups_do_not_run_and_the_rest_do(self):
        with tempfile.TemporaryDirectory() as d:
            r, ran = runner(d, "underdog,afternoon,collector,consensus,week4,month,month4,credible,smart")
            self.assertEqual(r.cfg.unknown_off(), [])
            r.store.open_position({"strategy": "month4", "asset_id": "a", "condition_id": "c", "title": "Q", "outcome": "No",
                                   "cost": 5.0, "shares": 7.0, "price": 0.7}, NOW)
            r.store.open_position({"strategy": "day4", "asset_id": "b", "condition_id": "c2", "title": "Q2", "outcome": "Yes",
                                   "cost": 5.0, "shares": 10.0, "price": 0.5}, NOW)
            r.run_due(NOW, force=True)
            self.assertEqual(set(ran), {"day", "day4", "day4pre", "day4x", "day4model", "all", "all4", "learned", "weather5", "weather10",
                                        "edge", "antimonth", "forecast"})
            self.assertIn("month4", r.marked)                     # what a switched-off setup holds is still priced
            self.assertEqual(len(r.store.open_positions("month4")), 1)   # and not sold
            st = r.store.read_status()
            flags = {s["name"]: s["enabled"] for s in st["strategies"]}
            self.assertEqual({n for n, on in flags.items() if not on},
                             {"underdog", "afternoon", "collector", "consensus", "week4", "month", "month4", "credible", "smart"})
            self.assertEqual(next(s for s in st["strategies"] if s["name"] == "month4")["open"], 1)   # its book is still shown
            variants = {v["name"]: v for v in st["variants"]}
            self.assertFalse(variants["month4"]["enabled"])
            self.assertIn("Switched off", variants["month4"]["note"])
            self.assertTrue(variants["antimonth"]["enabled"] and variants["antimonth"]["params"]["against"])
            self.assertEqual((st["underdog"]["enabled"], st["afternoon"]["enabled"], st["collector"]["enabled"], st["forecast"]["enabled"]),
                             (False, False, False, True))
            self.assertEqual((st["config"]["ud_enabled"], st["config"]["fc_enabled"]), (False, True))

    def test_nothing_named_means_everything_runs(self):
        with tempfile.TemporaryDirectory() as d:
            r, ran = runner(d, "none")
            r.run_due(NOW, force=True)
            self.assertEqual(len(ran), len(r.strategies))

    def test_a_misspelt_name_shows_on_the_page(self):
        with tempfile.TemporaryDirectory() as d:
            r, ran = runner(d, "month4,undrdog")
            self.assertEqual(r.cfg.unknown_off(), ["undrdog"])
            self.assertIn("undrdog", r.core.errors[-1]["message"])
            r.run_due(NOW, force=True)
            self.assertIn("underdog", ran)


if __name__ == "__main__":
    unittest.main()
