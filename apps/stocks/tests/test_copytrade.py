"""Copy trades: reading the House reports, the copy plans, the books, the picks, and that none of it sees
tomorrow.

    docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev python -m unittest tests.test_copytrade -v
"""
from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import synth
from stocks.book import Costs
from stocks.copytrade import house as H
from stocks.copytrade import plan as P
from stocks.copytrade import study as ST
from stocks.copytrade import trades as TT
from stocks.copytrade.model import Copy, CopyBook, CopyData, Period
from stocks.models import Env
from stocks.types import BookView, Holding
from test_leak import scrambled_after

FIX = Path(__file__).parent / "fixtures" / "house"


def report(name: str) -> list[dict]:
    return H.parse((FIX / f"{name}.txt").read_text(encoding="utf-8"))


class Reading(unittest.TestCase):
    def test_a_2025_report_across_a_page_break(self):
        rows = report("2025-20026587")
        self.assertEqual([r["ticker"] for r in rows], ["GOOGL", "BAH", "BR", "CTAS", "CMCSA", "GD", "UNH", "VRSK"])
        self.assertEqual([r["kind"] for r in rows], ["P", "S", "P", "P", "S", "S", "S", "P"])
        self.assertTrue(all(r["typ"] == "ST" and (r["lo"], r["hi"]) == (1001, 15000) for r in rows))
        self.assertEqual((rows[0]["traded"], rows[0]["notified"]), (20250107, 20250108))
        self.assertNotIn("ID Owner", " ".join(r["notes"] for r in rows))      # the repeated header is not a note

    def test_a_range_split_over_two_lines_and_an_owner(self):
        rows = report("2022-20021335")
        self.assertEqual([(r["owner"], r["ticker"], r["kind"]) for r in rows],
                         [("SP", "IMAX", "S (partial)"), ("SP", "IMAX", "S"), ("SP", "SNY", "P"), ("SP", "WBD", "P")])
        self.assertEqual((rows[2]["lo"], rows[2]["hi"]), (15001, 50000))
        self.assertEqual([H.what_to_copy(r) for r in rows], ["part", "sell", "buy", "buy"])
        self.assertEqual(H.size(rows[2]), 32500.5)

    def test_the_old_layout_with_odd_cases(self):
        rows = report("2019-20011279")                      # "sP BF Enterprises Common stock" … "New (BFET) [sT]"
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["owner"], rows[0]["ticker"], rows[0]["typ"], rows[0]["kind"]), ("SP", "BFET", "ST", "S"))
        pelosi = report("2016-20004503")                    # no asset type at all in 2016
        self.assertEqual([(r["ticker"], r["kind"], r["typ"]) for r in pelosi], [("CRM", "S", ""), ("V", "S", ""), ("V", "S", "")])
        self.assertEqual([H.what_to_copy(r) for r in pelosi], ["sell"] * 3)

    def test_foreign_codes_and_bonds_are_not_tickers(self):
        rows = report("2016-20004516")
        self.assertEqual([(r["owner"], r["ticker"]) for r in rows], [("JT", ""), ("JT", "SPWR"), ("JT", "")])
        self.assertEqual([H.what_to_copy(r) for r in rows], ["", "buy", ""])
        allen = report("2025-20026537")                     # a stock, then Treasury notes and bills by CUSIP
        self.assertEqual([H.what_to_copy(r) for r in allen], ["buy", "", "", ""])
        self.assertEqual(allen[0]["ticker"], "ROL")
        bonds = report("2022-20022197")
        self.assertEqual([H.what_to_copy(r) for r in bonds], ["", "", ""])
        self.assertEqual((bonds[2]["lo"], bonds[2]["hi"]), (50001, 100000))

    def test_options(self):
        rows = report("2019-20012685")                      # call options on XOM, written up as [ST]
        self.assertEqual([(r["ticker"], H.what_to_copy(r)) for r in rows],
                         [("XOM", "sell"), ("XOM", "sell"), ("XOM", "sell"), ("XOM", "buy")])
        put = {"ticker": "NVDA", "typ": "OP", "kind": "P", "asset": "NVIDIA (NVDA) [OP]", "notes": "D: Purchased 10 put options"}
        self.assertEqual(H.what_to_copy(put), "")
        bare = {"ticker": "NVDA", "typ": "OP", "kind": "P", "asset": "NVIDIA (NVDA) [OP]", "notes": ""}
        self.assertEqual(H.what_to_copy(bare), "")          # an option that does not say it is a call is not copied
        call = {**bare, "notes": "D: Purchased 50 call options with a strike price of $120"}
        self.assertEqual(H.what_to_copy(call), "buy")

    def test_other_labels_and_types(self):
        rows = report("2022-20021555")                      # "S          O : …" is a label, not part of the asset
        self.assertEqual([(r["ticker"], r["kind"]) for r in rows], [("LBRDA", "S (partial)")])
        self.assertNotIn("Creation", rows[0]["asset"])
        fig = report("2025-20030925")                       # "FIG (NYSE) [OT]": another kind of asset
        self.assertEqual(H.what_to_copy(fig[0]), "")
        arlp = report("2025-20030450")
        self.assertEqual([r["ticker"] for r in arlp], ["ARLP", "HTGC"])

    def test_every_fixture_reads_without_a_lost_row(self):
        for path in sorted(FIX.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            rows = H.parse(text)
            dated = sum(1 for line in text.splitlines() if H._TX.match(H._BOXES.sub("", line).strip()))
            self.assertEqual(len(rows), dated, path.name)
            self.assertTrue(all(r["traded"] > 20100000 and r["lo"] > 0 for r in rows), path.name)

    def test_the_pdf_itself(self):
        rows = H.parse(H.pdf_text((FIX / "2022-20021335.pdf").read_bytes()))
        self.assertEqual([r["ticker"] for r in rows], ["IMAX", "IMAX", "SNY", "WBD"])

    def test_amounts(self):
        self.assertEqual(H.amount("Over $50,000,000"), (50_000_000, 0))
        self.assertEqual(H.amount("Spouse/DC Over $1,000,000"), (1_000_000, 0))
        self.assertEqual(H.amount("$1,000,001 - $5,000,000"), (1_000_001, 5_000_000))
        self.assertEqual(H.size({"lo": 1_000_000, "hi": 0}), 1_000_000)


def index_zip(rows: list[tuple]) -> bytes:
    xml = "<FinancialDisclosure>" + "".join(
        f"<Member><Prefix>Hon.</Prefix><Last>{last}</Last><First>{first}</First><Suffix></Suffix><FilingType>{ft}</FilingType>"
        f"<StateDst>{st}</StateDst><Year>{year}</Year><FilingDate>{filed}</FilingDate><DocID>{doc}</DocID></Member>"
        for last, first, ft, st, year, filed, doc in rows) + "</FinancialDisclosure>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2025FD.xml", xml)
        z.writestr("2025FD.txt", "")
    return buf.getvalue()


class Fetching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.idx = index_zip([("Pelosi", "Nancy", "P", "CA12", 2025, "1/16/2025", "20026537"),
                              ("Larsen", "Rick", "P", "WA02", 2025, "1/9/2025", "20026587"),
                              ("Smith", "Jo", "P", "TX01", 2025, "2/1/2025", "8220001"),      # paper: a scan
                              ("Smith", "Jo", "C", "TX01", 2025, "2/1/2025", "10060001")])    # not a trade report
        self.asked = []

    def tearDown(self):
        self.tmp.cleanup()

    def fetch(self, url):
        self.asked.append(url)
        if url.endswith("FD.zip"):
            return 200, self.idx
        return 200, url.rsplit("/", 1)[1].encode()

    def house(self, fetch=None):
        text = lambda body: (FIX / f"2025-{body.decode()[:-4]}.txt").read_text(encoding="utf-8")   # noqa: E731
        return H.House(Path(self.tmp.name), fetch=fetch or self.fetch, sleep=lambda s: None, text_of=text)

    def test_reads_the_electronic_reports_once(self):
        h = self.house()
        got = h.sync([2025])
        self.assertEqual((got["reports"], got["paper"], got["failed"]), (2, 1, 0))
        rows = h.rows()
        self.assertEqual(sorted(rows["doc"].unique()), ["20026537", "20026587"])
        self.assertEqual(set(rows["pid"]), {"house:pelosi:nancy:ca", "house:larsen:rick:wa"})
        self.assertEqual(int(rows.loc[rows["doc"] == "20026537", "filed"].iloc[0]), 20250116)
        n = len(self.asked)
        again = self.house().sync([2025])
        self.assertEqual(again["reports"], 0)
        self.assertEqual(len(self.asked), n)                 # nothing: the reports are read and the index is fresh

    def test_titles_are_not_names(self):
        self.assertEqual(H.person("Green", "Mark Dr", "TN07"), ("house:green:mark:tn", "Mark Green", "TN"))
        self.assertEqual(H.person("Banks", "Hon. James E", "IN03")[:2], ("house:banks:james:in", "James E Banks"))
        self.assertEqual(H.person("Comstock", "Barbara J Honorable", "VA10")[1], "Barbara J Comstock")

    def test_a_refusal_stops_the_run_and_keeps_what_was_read(self):
        calls = []

        def fetch(url):
            calls.append(url)
            if url.endswith("FD.zip"):
                return 200, self.idx
            return (200, url.rsplit("/", 1)[1].encode()) if len(calls) == 2 else (429, b"")

        h = self.house(fetch)
        with self.assertRaises(H.Stopped):
            h.sync([2025])
        self.assertEqual(list(h.docs()["doc"]), ["20026537"])

    def test_people_and_trades(self):
        h = self.house()
        h.sync([2025])
        ins = pd.DataFrame([
            {"sym": "ABT", "day": 20250303, "at": "2025-02-28T16:30:00-05:00", "owner": "WHITE MILES D", "officer": True,
             "director": True, "ten_pct": False, "title": "Chairman and CEO", "shares": 10.0, "price": 100.0, "usd": 1000.0, "accession": "a1"},
            {"sym": "ABT", "day": 20250303, "at": "2025-02-28T16:30:00-05:00", "owner": "WHITE MILES D", "officer": True,
             "director": True, "ten_pct": False, "title": "Chairman and CEO", "shares": 20.0, "price": 101.0, "usd": 2020.0, "accession": "a1"},
            {"sym": "OXY", "day": 20250310, "at": "2025-03-07T18:00:00-05:00", "owner": "BERKSHIRE HATHAWAY INC", "officer": False,
             "director": False, "ten_pct": True, "title": "", "shares": 5.0, "price": 50.0, "usd": 250.0, "accession": "a2"}])
        people, trades, counts = TT.build(h, ins, {"ROL": "ROLX"})
        self.assertEqual(counts["house_people"], 2)
        self.assertEqual(set(people.loc[people["pool"] == "insider", "name"]), {"Miles D White", "Berkshire Hathaway Inc"})
        self.assertEqual([TT.pretty_owner(x) for x in ("HOLDING FRANK B JR", "MONROE JAMES III", "ABDIEL CAPITAL ADVISORS, LP")],
                         ["Frank B Holding Jr", "James Monroe III", "Abdiel Capital Advisors, Lp"])
        abt = trades[trades["pid"] == "in:WHITE MILES D|ABT"]
        self.assertEqual((len(abt), float(abt["usd"].iloc[0])), (1, 3020.0))     # one buy for the day
        rol = trades[trades["pid"] == "house:pelosi:nancy:ca"]
        self.assertEqual(list(rol["sym"]), ["ROLX"])                              # followed to its ticker today
        self.assertEqual(int(rol["day"].iloc[0]), 20250117)                       # the trading day after filing
        larsen = trades[trades["pid"] == "house:larsen:rick:wa"]
        self.assertEqual(len(larsen), 8)

    def test_an_amended_report_counts_once(self):
        base = {"year": 2025, "pid": "house:a:b:ca", "name": "B A", "state": "CA", "owner": "SELF", "asset": "X (XX) [ST]",
                "typ": "ST", "ticker": "XX", "kind": "P", "traded": 20250102, "notified": 20250103, "lo": 1001, "hi": 15000, "notes": ""}
        h = self.house()
        h._save([{**base, "doc": "1", "filed": 20250110, "n": 0}, {**base, "doc": "1", "filed": 20250110, "n": 1},
                 {**base, "doc": "2", "filed": 20250120, "n": 0}], [])
        _, trades, counts = TT.build(h, None, {})
        self.assertEqual((len(trades), counts["repeats"]), (2, 1))               # two alike in one report both count


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sym", "side", "usd", "day", "filed"])


CAL = np.array([int(d) for d in pd.bdate_range("2020-01-01", "2026-12-31").strftime("%Y%m%d")], dtype=np.int64)
YES = lambda day, sym: True             # noqa: E731


class Plans(unittest.TestCase):
    def test_stakes_in_proportion_capped_and_the_rest_waits(self):
        p = P.make("x", frame([("A", "buy", 8000, 20200106, 20200103), ("B", "buy", 32500, 20200106, 20200103),
                               ("C", "buy", 100000, 20200107, 20200106)]), CAL, YES, insider=False)
        self.assertEqual(list(p.days[:2]), [20200106, 20200107])
        self.assertEqual(p.at(20200106), {"A": 0.197531, "B": 0.25})             # 8000 / 40500, and B capped
        self.assertEqual(p.at(20200103), {})
        self.assertEqual(p.at(20200107)["C"], 0.25)
        self.assertEqual(p.count("opened", 0, 20200107), 3)

    def test_sales(self):
        p = P.make("x", frame([("A", "buy", 10000, 20200106, 1), ("B", "buy", 10000, 20200106, 1), ("C", "buy", 20000, 20200106, 1),
                               ("A", "part", 4000, 20200110, 1), ("B", "sell", 1000, 20200113, 1), ("Z", "sell", 50, 20200114, 1),
                               ("A", "part", 9000, 20200115, 1)]), CAL, YES, insider=False)
        self.assertEqual(p.at(20200106), {"A": 0.25, "B": 0.25, "C": 0.25})
        self.assertEqual(p.at(20200110), {"A": 0.166667, "B": 0.25, "C": 0.25})   # 6,000 of 36,000
        self.assertIn("sold $4,000 of it", p.note(20200110)["A"])
        self.assertEqual(p.at(20200113), {"A": 0.230769, "C": 0.25})              # a full sale ends it whatever its size
        self.assertEqual(p.at(20200115), {"C": 0.25})                             # more sold than held: gone
        self.assertEqual(p.count("copied", 0, 20201231), 6)                       # the sale of Z was of nothing held

    def test_a_purchase_that_cannot_be_copied_adds_nothing(self):
        no_b = lambda day, sym: sym != "B"           # noqa: E731
        p = P.make("x", frame([("A", "buy", 10000, 20200106, 1), ("B", "buy", 10000, 20200106, 1)]), CAL, no_b, insider=False)
        self.assertEqual(p.at(20200106), {"A": 0.25})
        self.assertEqual(p.count("opened", 0, 20201231), 1)

    def test_lapses(self):
        h = P.make("x", frame([("A", "buy", 10000, 20200106, 1)]), CAL, YES, insider=False)
        i = int(np.searchsorted(CAL, 20200106))
        self.assertEqual(list(h.days), [20200106, int(CAL[i + P.HOUSE_LAPSE])])
        self.assertEqual(h.at(int(CAL[i + P.HOUSE_LAPSE])), {})
        self.assertIn("lapsed", h.note(int(CAL[i + P.HOUSE_LAPSE]))["A"])
        ins = P.make("in:x|A", frame([("A", "buy", 10000, 20200106, 1), ("A", "buy", 5000, 20200301, 1)]), CAL, YES, insider=True)
        j = int(np.searchsorted(CAL, 20200302))
        self.assertEqual(ins.at(int(CAL[i + P.INSIDER_HOLD])), {"A": 0.25})       # the second purchase still held
        self.assertEqual(ins.at(int(CAL[j + P.INSIDER_HOLD])), {})
        self.assertEqual(ins.count("opened", 0, 20301231), 1)

    def test_a_day_the_market_was_shut_maps_forward_and_the_future_is_left_out(self):
        p = P.make("x", frame([("A", "buy", 10000, 20200104, 1), ("B", "buy", 10000, 20990101, 1)]), CAL, YES, insider=False)
        self.assertEqual(int(p.days[0]), 20200106)                                # a Saturday's filing acts on Monday
        self.assertNotIn("B", str(p.targets))


def view(weights: dict[str, float]) -> BookView:
    return BookView(10_000.0, 1.0 - sum(weights.values()), {s: Holding(s, w, 3, "2020-01-01", 0.0) for s, w in weights.items()})


class Books(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = synth.panel(20, 300, seed=2)
        cls.env = Env(index="SPY", cash="BIL")
        cls.days = cls.p.days.astype(np.int64)

    def stocks(self):
        return [s for s in self.p.syms if s not in ("SPY", "BIL")][:2]

    def book(self, rows, insider=False):
        a, b = self.stocks()
        rows = [({"A": a, "B": b}[r[0]], *r[1:]) for r in rows]
        plan = P.make("x", frame(rows), self.days, YES, insider)
        m = CopyBook("x", "x", [Period(int(self.days[0]), [plan], {"x": "X"})])
        m.prepare(self.p, self.env)
        return m, a, b

    def test_orders(self):
        d1, d2 = int(self.days[30]), int(self.days[40])
        m, a, b = self.book([("A", "buy", 10000, d1, d1), ("B", "buy", 30000, d2, d2)])
        first = m.decide(0, view({}))
        self.assertEqual([(o.symbol, o.weight) for o in first.orders], [("SPY", 0.995)])
        self.assertEqual(m.decide(10, view({"SPY": 0.99})).orders, [])
        w = round(0.25 * 0.995, 4)                                  # a stake alone is capped at a quarter
        on = m.decide(30, view({"SPY": 0.99}))
        self.assertEqual([(o.symbol, o.weight) for o in on.orders], [("SPY", round(0.995 - w, 4)), (a, w)])
        later = m.decide(40, view({"SPY": 0.74, a: 0.24}))         # B is three quarters of the stakes: capped too
        self.assertEqual([(o.symbol, o.weight) for o in later.orders], [("SPY", round(0.995 - 0.24 - w, 4)), (b, w)])
        drift = m.decide(40, view({"SPY": 0.70, a: 0.29}))          # within a quarter of its share: left alone
        self.assertNotIn(a, [o.symbol for o in drift.orders])
        far = m.decide(40, view({"SPY": 0.60, a: 0.40}))
        self.assertIn((a, w), [(o.symbol, o.weight) for o in far.orders])

    def test_buying_the_index_fund_itself_is_one_order(self):
        d1 = int(self.days[30])
        a, _ = self.stocks()
        plan = P.make("x", frame([("SPY", "buy", 50000, d1, d1), (a, "buy", 10000, d1, d1)]), self.days, YES, False)
        m = CopyBook("x", "x", [Period(int(self.days[0]), [plan], {"x": "X"})])
        m.prepare(self.p, self.env)
        orders = m.decide(30, view({"SPY": 0.99})).orders
        self.assertEqual([o.symbol for o in orders], ["SPY", a])
        w = round(round(10000 / 60000, 6) * 0.995, 4)          # A is a sixth of the stakes; SPY's five sixths are index anyway
        self.assertEqual([o.weight for o in orders], [round(0.995 - w, 4), w])

    def test_a_group_shares_the_book(self):
        d1 = int(self.days[30])
        a, b = self.stocks()
        p1 = P.make("p1", frame([(a, "buy", 1000, d1, d1)]), self.days, YES, False)
        p2 = P.make("p2", frame([(b, "buy", 1000, d1, d1), (a, "buy", 1000, d1, d1)]), self.days, YES, False)
        m = CopyBook("c", "c", [Period(int(self.days[0]), [p1, p2], {"p1": "One", "p2": "Two"}, 0.5)])
        m.prepare(self.p, self.env)
        target, why = m.target(d1)
        self.assertEqual(target, {a: 0.25, b: 0.125})              # One: 0.5 × 0.25 cap; Two: 0.5 × (0.25, 0.25)
        self.assertTrue(why[a].startswith("Two: bought") or why[a].startswith("One: bought"))


class NoLookAhead(unittest.TestCase):
    """The live combo decides the same on a panel that ends at t as on one that runs past it, and the same
    when everything after t is replaced with noise."""

    def test_copy_decides_on_what_it_could_know(self):
        p = synth.panel(30, 420, seed=7)
        rng = np.random.default_rng(3)
        days = p.days.astype(np.int64)
        rows = []
        for k, pid in enumerate(["house:a:b:ca", "house:c:d:ny", "in:E|" + p.syms[5]]):
            for _ in range(25):
                t = int(rng.integers(40, 410))
                sym = p.syms[5] if pid.startswith("in:") else p.syms[int(rng.integers(0, 28))]
                side = "buy" if pid.startswith("in:") or rng.random() < 0.6 else ("sell" if rng.random() < 0.5 else "part")
                rows.append({"pid": pid, "sym": sym, "side": side, "usd": float(rng.choice([8000, 32500, 75000])),
                             "day": int(days[t]), "traded": 0, "filed": int(days[t - 1]), "doc": "d", "what": ""})
        trades = pd.DataFrame(rows)
        picks = [{"start": int(days[100]), "people": ["house:a:b:ca", "in:E|" + p.syms[5]]},
                 {"start": int(days[250]), "people": ["house:c:d:ny", "house:a:b:ca", "in:E|" + p.syms[5]]}]
        data = lambda: CopyData(trades, picks, {}, days[:50], {}, 0)          # noqa: E731
        full = Copy()
        full.prepare(p, Env(index="SPY", cash="BIL", copy=data()))
        seen = 0
        for t in (120, 180, 251, 300, 377, 419):
            v = view({"SPY": 0.5, p.syms[3]: 0.1})
            a = full.decide(t, v)
            live = Copy()
            live.prepare(p.upto(t), Env(index="SPY", cash="BIL", copy=data()))
            b = live.decide(t, v)
            other = Copy()
            other.prepare(scrambled_after(p, t), Env(index="SPY", cash="BIL", copy=data()))
            c = other.decide(t, v)
            self.assertEqual([o.to_json() for o in a.orders], [o.to_json() for o in b.orders], f"row {t}")
            self.assertEqual([o.to_json() for o in a.orders], [o.to_json() for o in c.orders], f"row {t}")
            seen += bool(a.orders)
        changes = sorted(full.change)
        for t in changes[:6]:
            v = view({"SPY": 0.9})
            live = Copy()
            live.prepare(p.upto(t), Env(index="SPY", cash="BIL", copy=data()))
            self.assertEqual([o.to_json() for o in full.decide(t, v).orders], [o.to_json() for o in live.decide(t, v).orders])
            seen += 1
        self.assertGreater(seen, 3)

    def test_the_study_books_and_picks_do_not_move_when_the_future_does(self):
        p = synth.panel(25, 500, seed=11)
        days = p.days.astype(np.int64)
        rng = np.random.default_rng(5)
        trades = pd.DataFrame([{"pid": f"house:p{k}:x:ca", "sym": p.syms[int(rng.integers(0, 24))], "side": "buy",
                                "usd": 15000.0, "day": int(days[int(rng.integers(30, 495))]), "filed": 0}
                               for k in range(12) for _ in range(30)])
        t = 400

        def picked(panel):
            costs = Costs(panel)
            ok = P.liquid(panel, costs.adv)
            plans = ST.plans_for(trades, panel.days.astype(np.int64), ok)
            env = Env(index="SPY", cash="BIL")
            from stocks.models.rules import Index
            ix = Index()
            ix.prepare(panel, env)
            ie, _, _ = ST.run_book(panel, ix, env, ST.WARM, costs, 10_000.0, frozenset(), {})
            books = {}
            for pid, plan in plans.items():
                m = CopyBook(pid, pid, [Period(int(days[ST.WARM]), [plan], {})])
                m.prepare(panel, env)
                books[pid] = ST.run_book(panel, m, env, ST.WARM, costs, 10_000.0, frozenset(), {})[0]
            return books, ST.picks(panel, plans, books, ie, ST.WARM, int(days[0]), int(days[t]))

        old = ST.LOOKBACK, ST.MIN_LOOK, ST.MIN_OPENED
        ST.LOOKBACK, ST.MIN_LOOK, ST.MIN_OPENED = 200, 60, 4
        try:
            b1, k1 = picked(p)
            b2, k2 = picked(scrambled_after(p, t))
        finally:
            ST.LOOKBACK, ST.MIN_LOOK, ST.MIN_OPENED = old
        for pid in b1:
            np.testing.assert_allclose(b1[pid][: t + 1], b2[pid][: t + 1], err_msg=pid)
        made = [k for k in k1 if k["start"] <= int(days[t])]
        self.assertTrue(made and all(k["people"] for k in made))
        self.assertEqual(made, [k for k in k2 if k["start"] <= int(days[t])])


class Picks(unittest.TestCase):
    def test_half_years(self):
        self.assertEqual(ST.half_years(20190102, 20200702), [20190101, 20190701, 20200101, 20200701])
        self.assertEqual(ST.half_years(20190102, 20200630), [20190101, 20190701, 20200101])

    def test_the_best_qualified_are_picked(self):
        days = CAL[:900]
        T = len(days)
        ix = np.linspace(1, 1.5, T)
        books, plans = {}, {}
        for k, (drift, opened, active) in enumerate([(0.9, 12, True), (0.5, 12, True), (2.0, 5, True), (1.5, 12, False), (0.1, 12, True)]):
            pid = f"p{k}"
            books[pid] = ix * np.linspace(1, 1 + drift, T)
            plan = P.Plan(pid)
            plan.opened = np.sort(days[np.linspace(30, 780 if active else 500, opened).astype(int)])
            plan.copied = plan.opened.copy()
            plans[pid] = plan
        old = ST.TOP
        ST.TOP = 2
        try:
            out = ST.picks(type("Pnl", (), {"days": days})(), plans, books, ix, 0, int(days[0]), int(days[-1]) + 1)
        finally:
            ST.TOP = old
        last = out[-1]
        self.assertEqual(last["people"], ["p0", "p1"])      # p2 opened too few, p3 went quiet, p4 did less
        self.assertEqual(last["eligible"], 3)

    def test_an_insider_counts_purchases(self):
        ins, house = P.Plan("in:x|A"), P.Plan("house:x:y:ca")
        for p in (ins, house):
            p.opened = CAL[[10, 400]]
            p.copied = CAL[np.arange(10, 400, 30)]
        self.assertEqual((ST.record(ins, 0, int(CAL[-1])), ST.record(house, 0, int(CAL[-1]))), (13, 2))


if __name__ == "__main__":
    unittest.main()
