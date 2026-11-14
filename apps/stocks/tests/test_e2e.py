"""The whole path on a made-up market: history in, universe picked, models fitted and backtested, evenings
run one after another, books replayed, the page's JSON written.

    docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev python -m unittest tests.test_e2e -v

Takes a couple of minutes: it fits the three learned models (in fast mode)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

import synth
from fake_alpaca import FakeAlpaca

FIX = Path(__file__).parent / "fixtures"


def evening_of(day: int) -> datetime:
    return datetime(day // 10000, day // 100 % 100, day % 100, 22, 0, tzinfo=timezone.utc)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="stocks-e2e-"))
        env = {"DATA_DIR": str(cls.tmp / "data"), "CACHE_DIR": str(cls.tmp / "cache"), "ALPACA_API_KEY": "k",
               "ALPACA_API_SECRET": "s", "STK_UNIVERSE": "100", "STK_FAST": "1", "STK_HISTORY_START": "2018-01-01",
               "STK_NOTIFY": "off", "STK_MIN_PRICE": "1"}
        cls.patch = mock.patch.dict(os.environ, env, clear=False)
        cls.patch.start()
        for k in ("STOCKS_ANTHROPIC_API_KEY", "SEC_CONTACT", "NTFY_URL"):
            os.environ.pop(k, None)
        from stocks import research
        from stocks.config import Config
        from stocks.pipeline import Lab, evening
        cls.frame = synth.market(150, 900, seed=3, reversal=0.25)
        cls.days = np.sort(cls.frame.loc[cls.frame["sym"] == "SPY", "d"].unique())
        cls.cfg = Config.from_env()
        cls.lab = Lab(cls.cfg)
        cls.api = cls.lab.api = FakeAlpaca(cls.frame, int(cls.days[860]))
        cls.lines = []

        def run(i: int) -> str:
            cls.api.today = int(cls.days[i])
            cls.lab = Lab(cls.cfg)                       # a fresh process each evening, as the runner does it
            cls.lab.api = cls.api
            line = evening(cls.lab, evening_of(int(cls.days[i])))
            cls.lines.append(line)
            return line

        cls.run_evening = staticmethod(run)
        run(860)                                         # first evening: the long history, the universe, the index starts
        cls.state_first = json.loads((cls.tmp / "data" / "state.json").read_text())
        cls.research = research.run(cls.lab, say=lambda t: None)
        research.fit_live(cls.lab, say=lambda t: None)
        for i in range(861, 875):
            run(i)
        cls.state = json.loads((cls.tmp / "data" / "state.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def model(self, mid: str, state=None) -> dict:
        return next(m for m in (state or self.state)["models"] if m["id"] == mid)

    def detail(self, mid: str) -> dict:
        return json.loads((self.tmp / "data" / f"model_{mid}.json").read_text())

    # ── the first evening ───────────────────────────────────────────────────────────────────────
    def test_history_and_universe(self):
        self.assertEqual(len(self.lab.bars.days("SPY")), 875)
        members = self.lab.universe.members(int(self.days[870]))
        self.assertEqual(len(members), 100)
        self.assertNotIn("SPY", members)                 # a fund is not a company
        self.assertNotIn("TQQQ", members)
        self.assertFalse(any(m.endswith("W") and len(m) == 5 for m in members))

    def test_first_evening_only_the_index_starts(self):
        s = self.state_first
        self.assertEqual(s["phase"], "research")         # the index has a book from the first evening; the rest wait for their backtests
        self.assertEqual(self.model("index", s)["state"], "running")
        for mid in ("net", "trees", "ridge", "momentum", "reversal", "trend", "blend"):
            self.assertEqual(self.model(mid, s)["state"], "waiting", mid)
            self.assertEqual(self.model(mid, s)["state_text"], "waiting for its backtest" if mid not in ("net", "trees", "ridge", "blend")
                             else self.model(mid, s)["state_text"])
        self.assertIn("backtest", self.model("momentum", s)["state_text"])
        self.assertEqual(self.state["phase"], "running")  # and once the backtests are written, the lab is running

    # ── research ────────────────────────────────────────────────────────────────────────────────
    def test_reports_written_and_shaped(self):
        for mid in ("index", "net", "netfast", "trees", "ridge", "blend", "momentum", "reversal", "trend"):
            rep = json.loads((self.tmp / "data" / "reports" / f"{mid}.json").read_text())
            bt = rep["backtest"]
            for k in ("from", "to", "ret_ann", "index_ann", "vs_index_ann", "ci", "sharpe", "max_dd", "turnover", "cost_drag", "ic", "ic_t", "exam"):
                self.assertIn(k, bt, f"{mid}.{k}")
            self.assertIsNone(bt["exam"])
            text = (self.tmp / "data" / "reports" / f"{mid}.md").read_text()
            self.assertIn("What this does not show", text)
        self.assertFalse((self.tmp / "data" / "reports" / "drift.json").exists())      # no filings, no report, no start
        pace = (self.tmp / "data" / "reports" / "netfast.md").read_text()
        self.assertIn("Does trading more often pay?", pace)
        self.assertIn("| Network, fast |", pace)
        self.assertIn("| Network, slow |", pace)                     # the old holding rule, as a yardstick only
        self.assertFalse((self.tmp / "data" / "reports" / "netslow.json").exists())

    def test_the_fast_network_trades_on_the_one_day_view_and_turns_over_more(self):
        slow, fast = self.detail("net"), self.detail("netfast")
        self.assertEqual(self.model("netfast")["verdict"]["horizon"], 1)
        self.assertGreater(self.model("netfast")["trades"], self.model("net")["trades"])
        self.assertNotEqual([p["symbol"] for p in slow["today"]["top"]], [p["symbol"] for p in fast["today"]["top"]])

    def test_planted_reversal_is_found_by_the_learned_models(self):
        got = {m["id"]: m for m in self.research["models"]}
        for mid in ("ridge", "trees", "net"):
            self.assertGreater(got[mid]["ic"], 0.008, mid)
            self.assertGreater(got[mid]["ic_t"], 2.0, mid)

    def test_index_backtest_is_the_index(self):
        got = {m["id"]: m for m in self.research["models"]}
        self.assertAlmostEqual(got["index"]["vs_index_ann"], 0.0, places=6)

    # ── the evenings after ──────────────────────────────────────────────────────────────────────
    def test_models_run_and_hold_what_their_rules_say(self):
        for mid, n in (("index", 1), ("net", 40), ("netfast", 40), ("trees", 40), ("ridge", 40), ("blend", 40), ("momentum", 40), ("reversal", 40)):
            m = self.model(mid)
            self.assertEqual(m["state"], "running", mid)
            self.assertEqual(m["n_pos"], n, mid)
            self.assertLess(m["cash"], 0.08, mid)
            self.assertIsNotNone(m["backtest"], mid)
        self.assertEqual(self.model("drift")["state"], "waiting")
        self.assertIn("SEC_CONTACT", self.model("drift")["state_text"])
        self.assertEqual(self.model("agent")["state"], "waiting")

    def test_index_book_tracks_the_index_less_one_trade(self):
        m, d = self.model("index"), self.detail("index")
        self.assertEqual(m["trades"], 1)
        self.assertEqual(d["trades"][0]["side"], "buy")
        self.assertLess(abs(m["vs_index"]), 0.01)       # the gap is the first night in cash and a cent of costs
        self.assertGreater(m["costs"], 0)
        self.assertEqual(d["curve"][0][1], 10000.0)

    def test_fills_are_at_the_next_open(self):
        d = self.detail("net")
        first = min(t["date"] for t in d["trades"])
        self.assertEqual(first, f"{self.days[862] // 10000}-{self.days[862] // 100 % 100:02d}-{self.days[862] % 100:02d}")
        t = next(x for x in d["trades"] if x["date"] == first)
        row = self.frame[(self.frame["sym"] == t["symbol"]) & (self.frame["d"] == self.days[862])].iloc[0]
        self.assertAlmostEqual(t["price"], float(row["o"]), places=2)

    def test_rankings_are_scored_once_their_outcome_is_known(self):
        d = self.detail("ridge")
        self.assertGreaterEqual(len(d["ranking_history"]), 6)       # 5-day outcomes of the first evenings are in
        self.assertEqual(self.model("ridge")["verdict"]["lamp"], "none")   # far too early for a verdict
        self.assertIn("No verdict yet", self.model("ridge")["verdict"]["text"])
        self.assertEqual(len(d["today"]["top"]), 12)
        self.assertEqual(d["today"]["top"][0]["rank"], 1)

    def test_a_second_run_of_the_same_evening_changes_nothing(self):
        before = {mid: len(self.lab.store.decisions(mid)) for mid in ("index", "net", "momentum")}
        eq = self.model("net")["equity"]
        self.run_evening(874)
        after = {mid: len(self.lab.store.decisions(mid)) for mid in before}
        self.assertEqual(before, after)
        state = json.loads((self.tmp / "data" / "state.json").read_text())
        self.assertAlmostEqual(self.model("net", state)["equity"], eq, places=2)

    # ── the page's contract ─────────────────────────────────────────────────────────────────────
    def test_state_has_the_fields_the_page_was_built_on(self):
        fx = json.loads((FIX / "state.json").read_text())
        self.assertEqual(set(self.state), set(fx))
        self.assertEqual(set(self.state["spend"]), set(fx["spend"]))
        self.assertEqual(set(self.state["clock"]), set(fx["clock"]))
        for m in self.state["models"]:
            self.assertEqual(set(m), set(fx["models"][0]), m["id"])
        fxd = json.loads((FIX / "model_net.json").read_text())
        for mid in ("net", "index", "agent", "drift"):
            self.assertEqual(set(self.detail(mid)), set(fxd), mid)
        d = self.detail("net")
        for part in ("positions", "trades", "pending", "curve"):
            self.assertTrue(d[part], part)
        self.assertEqual(set(d["positions"][0]), set(fxd["positions"][0]))
        self.assertEqual(set(d["trades"][0]), set(fxd["trades"][0]))
        self.assertEqual(set(d["pending"][0]), set(fxd["pending"][0]))
        self.assertEqual(set(d["today"]["top"][0]), set(fxd["today"]["top"][0]))
        self.assertEqual(set(d["backtest"]), set(fxd["backtest"]))

    def test_no_nan_reaches_the_page(self):
        for path in (self.tmp / "data").glob("*.json"):
            text = path.read_text()
            self.assertNotIn("NaN", text, path.name)
            self.assertNotIn("Infinity", text, path.name)


if __name__ == "__main__":
    unittest.main()
