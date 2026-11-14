"""Fetching prices: returns from the two series, late splits, corrections.

    docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev python -m unittest tests.test_ingest -v
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

import synth
from fake_alpaca import FakeAlpaca
from stocks import ingest
from stocks.alpaca import Alpaca, AlpacaError
from stocks.bars import BarStore, d2s


def as_date(d: int) -> date:
    return date(d // 10000, d // 100 % 100, d % 100)


class Ingest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synth.market(12, 460, seed=2)
        cls.days = np.sort(cls.frame.loc[cls.frame["sym"] == "SPY", "d"].unique())
        cls.syms = sorted(cls.frame["sym"].unique())
        cls.split_day = int(cls.days[400])          # SAAD splits 2-for-1 here

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="stocks-ingest-"))
        self.store = BarStore(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def truth(self, sym: str, d: int) -> float:
        row = self.frame[(self.frame["sym"] == sym) & (self.frame["d"] == d)].iloc[0]
        return float(row["ret"])

    def test_return_comes_from_the_adjusted_series_and_prices_stay_raw(self):
        api = FakeAlpaca(self.frame, int(self.days[-1]))
        df = ingest.pull(api, ["SAAD", "SAAE"], d2s(self.days[380]), d2s(self.days[420]))
        row = df[(df["sym"] == "SAAD") & (df["d"] == self.split_day)].iloc[0]
        self.assertAlmostEqual(row["ret"], self.truth("SAAD", self.split_day), places=5)   # the split is not in the return
        self.assertLess(row["raw_move"], -0.4)                                             # but it is in the raw price
        div = df[(df["sym"] == "SAAE") & (df["d"] == int(self.days[410]))].iloc[0]
        self.assertAlmostEqual(div["ret"], self.truth("SAAE", int(self.days[410])), places=5)
        self.assertTrue(np.isnan(df[df["sym"] == "SAAD"].iloc[0]["ret"]))                  # no previous close in the pull

    def test_a_split_alpaca_has_not_adjusted_yet_is_fixed_from_the_action_list(self):
        api = FakeAlpaca(self.frame, self.split_day)
        api.late_splits.add(("SAAD", self.split_day))
        raw = ingest.pull(api, ["SAAD"], d2s(self.days[390]), d2s(self.split_day))
        self.assertLess(raw.iloc[-1]["ret"], -0.4)                 # as Alpaca would serve it that evening: a fake crash
        self.assertEqual(len(ingest.split_suspects(raw)), 1)
        api.splits[("SAAD", self.split_day)] = (2.0, 1.0)
        notes = []
        fixed = ingest.apply_splits(api, raw, notes.append)
        self.assertAlmostEqual(fixed.iloc[-1]["ret"], self.truth("SAAD", self.split_day), places=4)
        self.assertEqual(len(notes), 1)

    def test_a_halving_with_no_split_on_record_stays_a_loss(self):
        api = FakeAlpaca(self.frame, self.split_day)
        api.late_splits.add(("SAAD", self.split_day))             # looks like a split, but the action list is empty
        raw = ingest.pull(api, ["SAAD"], d2s(self.days[390]), d2s(self.split_day))
        kept = ingest.apply_splits(api, raw, lambda t: None)
        self.assertLess(kept.iloc[-1]["ret"], -0.4)

    def test_backfill_then_evening_updates_append_and_correct(self):
        api = FakeAlpaca(self.frame, int(self.days[399]))
        n = ingest.backfill(api, self.store, self.syms, d2s(self.days[0]), d2s(self.days[399]), chunk=5)
        self.assertGreater(n, 399 * 10)
        self.assertEqual(self.store.last_day("SPY"), int(self.days[399]))
        self.assertEqual(sorted(self.store.years()), sorted({int(d) // 10000 for d in self.days[:400]}))
        # the split evening: Alpaca is late, nothing on record → the table holds the fake crash
        api.today = self.split_day
        api.late_splits.add(("SAAD", self.split_day))
        notes = []
        got = ingest.update(api, self.store, self.syms, as_date(self.split_day), note=notes.append)
        self.assertEqual(got["after"], self.split_day)
        stored = self.store.read(self.split_day, self.split_day, ["SAAD"]).iloc[0]
        self.assertLess(stored["ret"], -0.4)
        # the next evening the adjustment has arrived: the row is corrected and the correction is said out loud
        api.today = int(self.days[401])
        api.late_splits.clear()
        notes.clear()
        got = ingest.update(api, self.store, self.syms, as_date(int(self.days[401])), note=notes.append)
        fixed = self.store.read(self.split_day, self.split_day, ["SAAD"]).iloc[0]
        self.assertAlmostEqual(fixed["ret"], self.truth("SAAD", self.split_day), places=4)
        self.assertTrue(any("SAAD" in c["sym"] for c in got["corrections"]))
        self.assertTrue(any("data corrected: SAAD" in t for t in notes))
        # old rows keep their returns although each pull's first row has none
        all_rows = self.store.read(None, None, ["SABA"]).sort_values("d")
        self.assertEqual(int(all_rows["ret"].isna().sum()), 1)     # only its very first bar
        self.assertEqual(len(all_rows), 402)
        self.assertAlmostEqual(float(all_rows.iloc[250]["c"]),
                               float(self.frame[(self.frame["sym"] == "SABA") & (self.frame["d"] == int(self.days[250]))].iloc[0]["c"]), places=3)

    def test_a_history_fetch_stopped_halfway_picks_up_where_it_was(self):
        api = FakeAlpaca(self.frame, int(self.days[200]))
        real, seen = api.bars, {"n": 0}

        class Stopped(Exception):
            pass

        def flaky(symbols, start, end, adjustment="raw"):
            seen["n"] += 1
            if seen["n"] == 5:
                raise Stopped()
            return real(symbols, start, end, adjustment)

        api.bars = flaky
        args = (self.store, self.syms, d2s(self.days[0]), d2s(self.days[200]))
        with self.assertRaises(Stopped):
            ingest.backfill(api, *args, chunk=4)
        self.assertEqual(self.store.years(), [])                              # nothing half-written into the table
        self.assertEqual(len(list((self.store.dir / "staging").glob("*.parquet"))), 2)
        api.bars, api.calls = real, 0
        ingest.backfill(api, *args, chunk=4)
        chunks = -(-len(self.syms) // 4)
        self.assertEqual(api.calls, (chunks - 2) * 2)                         # the two finished handfuls were not fetched again
        self.assertFalse((self.store.dir / "staging").exists())
        self.assertEqual(self.store.last_day("SPY"), int(self.days[200]))
        self.assertEqual(len(self.store.read(None, None, ["SAAB"])), 201)

    def test_a_renamed_company_is_found_under_its_old_ticker(self):
        f = self.frame
        self.store.upsert(f)
        # SAAC was renamed: its bars up to day 300 are also there as OLDN (which then stops), as Alpaca serves them
        old = f[(f["sym"] == "SAAC") & (f["d"] <= int(self.days[300]))].assign(sym="OLDN")
        # and a stock that merely shares five days of bars with SAAE is nobody's old name
        few = f[f["sym"] == "SAAG"].copy()
        same = f[(f["sym"] == "SAAE") & f["d"].isin(self.days[100:105])]
        few = few[~few["d"].isin(same["d"])]
        self.store.upsert(old)
        self.store.upsert(same.assign(sym="FEWD"))
        active = set(self.syms)
        self.assertEqual(ingest.aliases(self.store, active), {"OLDN": "SAAC"})
        # neither listed any more: the shorter history is the old name
        self.assertEqual(ingest.aliases(self.store, active - {"SAAC"}), {"OLDN": "SAAC"})
        # both listed today: left alone, whatever they share
        self.assertEqual(ingest.aliases(self.store, active | {"OLDN"}), {})

    def test_a_symbol_new_to_the_store_gets_its_whole_history(self):
        api = FakeAlpaca(self.frame, int(self.days[-1]))
        rest = [s for s in self.syms if s != "SAAF"]
        ingest.backfill(api, self.store, rest, d2s(self.days[0]), d2s(self.days[-1]))
        self.assertNotIn("SAAF", self.store.symbols())
        said: list[str] = []
        ingest.update(api, self.store, self.syms, as_date(int(self.days[-1])), note=said.append, history_start=d2s(self.days[0]))
        got = self.store.days("SAAF")
        truth = self.frame[(self.frame["sym"] == "SAAF") & (self.frame["v"] > 0)]
        self.assertEqual(len(got), len(truth))                     # not the two-week tail: every day it traded
        self.assertGreater(len(got), 400)
        self.assertTrue(any("SAAF" in line for line in said))
        self.assertAlmostEqual(float(self.store.read(int(self.days[200]), int(self.days[200]), ["SAAF"]).iloc[0]["ret"]),
                               self.truth("SAAF", int(self.days[200])), places=5)

    def test_reading_filters_by_day_and_symbol(self):
        api = FakeAlpaca(self.frame, int(self.days[100]))
        ingest.backfill(api, self.store, self.syms, d2s(self.days[0]), d2s(self.days[100]))
        df = self.store.read(int(self.days[10]), int(self.days[19]), ["SAAB", "SPY"], ["c"])
        self.assertEqual(len(df), 20)
        self.assertEqual(list(df.columns), ["c"])
        self.assertEqual(list(self.store.days("SPY"))[:3], [int(d) for d in self.days[:3]])


class Client(unittest.TestCase):
    """The real client against a scripted HTTP layer: paging, the 15-minute rule, rate limits, errors."""

    def client(self, script, now=None):
        calls = []

        def fetch(url, headers, timeout=60.0):
            calls.append(url)
            return script(len(calls), url)

        import datetime as dt
        api = Alpaca("k", "s", fetch=fetch, sleep=lambda s: None,
                     now=lambda: now or dt.datetime(2026, 11, 13, 21, 30, tzinfo=dt.timezone.utc))
        return api, calls

    def test_pages_are_followed_and_the_keys_are_sent(self):
        import json
        seen = {}

        def script(n, url):
            body = {"bars": {"A": [{"t": "2026-11-12T05:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "n": 1, "vw": 1}]},
                    "next_page_token": "tok" if n == 1 else None}
            return 200, {}, json.dumps(body).encode()

        api, calls = self.client(script)
        got = list(api.bars(["A"], "2026-11-01", "2026-11-12"))
        self.assertEqual(len(got), 2)
        self.assertIn("page_token=tok", calls[1])
        self.assertIn("adjustment=raw", calls[0])
        self.assertIn("feed=sip", calls[0])
        self.assertEqual(api._headers["APCA-API-KEY-ID"], "k")

    def test_a_window_ending_today_stops_sixteen_minutes_ago(self):
        api, calls = self.client(lambda n, url: (200, {}, b'{"bars": {}, "next_page_token": null}'))
        list(api.bars(["A"], "2026-11-01", "2026-11-13"))
        self.assertIn("end=2026-11-13T21%3A14%3A00Z", calls[0])
        list(api.bars(["A"], "2026-11-01", "2026-11-10"))
        self.assertIn("end=2026-11-10", calls[1])

    def test_a_rate_limit_is_waited_out_and_a_refusal_is_raised(self):
        api, calls = self.client(lambda n, url: (429, {}, b"{}") if n == 1 else (200, {}, b"[]"))
        self.assertEqual(api.assets(), [])
        self.assertEqual(len(calls), 2)
        api, _ = self.client(lambda n, url: (403, {}, b'{"message": "forbidden."}'))
        with self.assertRaises(AlpacaError) as e:
            api.assets()
        self.assertEqual(e.exception.status, 403)

    def test_there_is_no_way_to_place_an_order(self):
        src = (Path(__file__).parent.parent / "stocks").rglob("*.py")
        for path in src:
            text = path.read_text()
            for word in ("/v2/orders", "/v2/positions", "submit_order", "api.alpaca.markets/v2/orders"):
                self.assertNotIn(word, text, f"{path.name} mentions {word}")
        self.assertFalse([n for n in dir(Alpaca) if any(w in n.lower() for w in ("order", "buy", "sell", "submit", "trade"))])
        alpaca_src = (Path(__file__).parent.parent / "stocks" / "alpaca.py").read_text()
        self.assertNotIn('method="POST"', alpaca_src)
        self.assertNotIn("data=", alpaca_src)                      # nothing is ever sent, only fetched


if __name__ == "__main__":
    unittest.main()
