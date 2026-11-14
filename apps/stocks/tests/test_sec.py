"""The SEC client: who is asked and how fast, the shapes it reads, and what it makes of them. No network: every
answer comes from tests/fixtures/sec through a stand-in for the fetch (one class talks to a server on this
machine, to try the real fetch).

    docker run --rm -v "$PWD":/app -w /app plint/stocks:dev python -m unittest tests.test_sec tests.test_filings -v

The fixtures are small files in the shapes read from sec.gov on 2026-11-14. The companies in them (Northwind
Tools, Riverbend Holdings, Gone Corp) and their people are made up; only the times in `Times` are real ones,
from Apple's file. Two shapes were not seen in the raw and are written as they are believed to be: the tables
inside a quarterly insider set (the columns are the SEC readme's) and the HTML of a filing's index page."""
from __future__ import annotations

import gzip
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from stocks.sec import (DATA, GAP, SET_COLS, SETS_PAGE, WWW, Sec, SecError, _http, _StayAtSec, ny_time, read_insider_set,
                        read_ownership, readable, sec_url)

FIX = Path(__file__).resolve().parent / "fixtures" / "sec"
TICKERS = f"{WWW}/files/company_tickers.json"
SUBS = f"{DATA}/submissions/"
ARCH = f"{WWW}/Archives/edgar/data/"
SETS = "/data/insider-transactions-data-sets/"
COVER = ARCH + "9900001/000990000126000031/nwt-20261112.htm"
INDEX = ARCH + "9900001/000990000126000031/0009900001-26-000031-index.htm"
EXHIBIT = ARCH + "9900001/000990000126000031/nwt-ex991.htm"
CONTACT = "ops@example.com"
TABLES = ("SUBMISSION", "REPORTINGOWNER", "NONDERIV_TRANS")


def fixture(name: str) -> bytes:
    return (FIX / name).read_bytes()


class Clock:
    """A clock that moves only when someone sleeps (or an answer takes its time)."""

    def __init__(self):
        self.t = 5000.0
        self.naps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.naps.append(seconds)
        self.t += max(0.0, seconds)


class Web:
    """Stands in for the network: answers from a table of address → reply and keeps what was asked, with which
    headers, and when by the clock. A reply is bytes (a 200), (status, bytes), an exception to raise, or a list
    of those to give in turn (the last one from then on). An address not in the table is a 404."""

    def __init__(self, routes: dict | None = None, clock: Clock | None = None, took: float = 0.0):
        self.routes = dict(routes or {})
        self.clock, self.took = clock, took
        self.asked: list[str] = []
        self.headers: list[dict] = []
        self.times: list[float] = []

    def __call__(self, url: str, headers: dict):
        self.asked.append(url)
        self.headers.append(dict(headers))
        if self.clock is not None:
            self.times.append(self.clock.t)
            self.clock.t += self.took
        reply = self.routes.get(url, (404, b"<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>"))
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        status, body = (200, reply) if isinstance(reply, bytes) else reply
        return status, {}, body


def insider_zip(tables: dict[str, bytes] | None = None, ext: str = ".tsv", folder: str = "") -> bytes:
    """A quarterly set as a zip: the three tables that are read (from the fixtures unless given), and
    by-standers of the kind a real one carries."""
    tables = tables if tables is not None else {t: fixture(f"form345/{t}.tsv") for t in TABLES}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in tables.items():
            z.writestr(f"{folder}{name}{ext}", body)
        z.writestr(f"{folder}FOOTNOTES{ext}", "ACCESSION_NUMBER\tFOOTNOTE_ID\tFOOTNOTE_TXT\n")
        z.writestr(f"{folder}FORM_345_readme.htm", "<html><body>readme</body></html>")
    return buf.getvalue()


def heads_only() -> dict[str, bytes]:
    """The three tables with their header rows and nothing under them: a quarter in which nobody bought."""
    return {t: fixture(f"form345/{t}.tsv").split(b"\n", 1)[0] + b"\n" for t in TABLES}


def site() -> dict:
    """The made-up corner of sec.gov the tests live in. Gone Corp (CIK 9900002) has a ticker and no file."""
    return {
        TICKERS: fixture("company_tickers.json"),
        SUBS + "CIK0009900001.json": fixture("CIK0009900001.json"),
        SUBS + "CIK0009900001-submissions-001.json": fixture("CIK0009900001-submissions-001.json"),
        SUBS + "CIK0009900001-submissions-002.json": fixture("CIK0009900001-submissions-002.json"),
        SUBS + "CIK0009900003.json": fixture("CIK0009900003.json"),
        ARCH + "9900001/000199999026000412/form4.xml": fixture("form4_mixed.xml"),
        ARCH + "9900001/000199999026000409/form4.xml": fixture("form4_two_owners.xml"),
        ARCH + "9900003/000199999026000409/form4.xml": fixture("form4_two_owners.xml"),     # the same filing, in the buyer's folder
        ARCH + "9900001/000199999026000398/form4.xml": fixture("form4_sale.xml"),
        COVER: fixture("nwt-20261112.htm"),
        INDEX: fixture("0009900001-26-000031-index.htm"),
        EXHIBIT: fixture("nwt-ex991.htm"),
        SETS_PAGE: fixture("insider_sets.html"),
        f"{WWW}/files/datastandardsinnovation{SETS}2026q3_form345.zip": insider_zip(),
        f"{WWW}/files/datastandardsinnovation{SETS}2026q2_form345.zip": b"this is not a zip",
        f"{WWW}/files/structureddata{SETS}2026q1_form345.zip": insider_zip(heads_only()),
    }


def client(web: Web, clock: Clock | None = None) -> Sec:
    clock = clock or web.clock or Clock()
    return Sec(CONTACT, fetch=web, sleep=clock.sleep, clock=clock)


class Contact(unittest.TestCase):
    def test_no_contact_no_client_and_no_request(self):
        web = Web(site())
        for nobody in ("", "   ", None, "\r\n\t"):
            with self.assertRaises(ValueError):
                Sec(nobody, fetch=web)
        for odd in ("ops@example.com\x00", "ops@exämple.com"):          # would not survive as a header
            with self.assertRaises(ValueError):
                Sec(odd, fetch=web)
        self.assertEqual(web.asked, [])

    def test_every_request_names_the_contact_and_goes_to_the_sec(self):
        web = Web(site())
        sec = client(web)
        sec.tickers()
        sec.submissions(9900001, all_pages=True)
        sec.text(COVER)
        sec.form4(9900001, "0001999990-26-000412", "xslF345X06/form4.xml")
        sec.insider_quarter(2026, 3)
        self.assertEqual(len(web.asked), 10)        # 1 + 3 + 3 (cover, index, exhibit) + 1 + 2 (the list, the zip)
        self.assertEqual({json.dumps(h, sort_keys=True) for h in web.headers}, {json.dumps({"User-Agent": "Plint stocks (ops@example.com)"})})
        for url in web.asked:
            self.assertEqual(sec_url(url), url)

    def test_a_contact_cannot_smuggle_a_header(self):
        web = Web(site())
        Sec("  ops@example.com \r\nX-Other: 1 ", fetch=web, sleep=Clock().sleep).tickers()
        self.assertEqual(web.headers, [{"User-Agent": "Plint stocks (ops@example.com X-Other: 1)"}])


class Pace(unittest.TestCase):
    def no_second_holds_more_than_eight(self, times: list[float]):
        self.assertGreater(len(times), 8)
        for i in range(len(times) - 8):
            self.assertGreater(times[i + 8] - times[i], 1.0, f"requests {i} to {i + 8} fell within one second")

    def test_never_more_than_eight_requests_in_any_second(self):
        clock = Clock()
        web = Web(site(), clock)
        sec = client(web)
        for _ in range(40):
            sec.submissions(9900003)
        self.assertEqual(len(web.times), 40)
        self.no_second_holds_more_than_eight(web.times)
        self.assertLess(web.times[-1] - web.times[0], 39 * 0.15)       # and it does not dawdle: about 7.7 a second

    def test_the_pace_holds_whatever_is_asked(self):
        clock = Clock()
        web = Web(site(), clock)
        sec = client(web)
        for _ in range(6):                                   # one call, several requests: none may slip the queue
            sec.submissions(9900001, all_pages=True)
            sec.text(COVER)
            sec.form4(9900001, "0001999990-26-000409", "xslF345X06/form4.xml")
            sec.insider_quarter(2026, 3)
        self.assertEqual(len(web.times), 6 * 8 + 1)
        self.no_second_holds_more_than_eight(web.times)
        self.assertEqual(sec.calls, len(web.times))

    def test_the_time_an_answer_takes_counts_as_waiting(self):
        clock = Clock()
        web = Web(site(), clock, took=0.5)                   # the line is slow: half a second an answer
        sec = client(web)
        for _ in range(5):
            sec.submissions(9900003)
        self.assertEqual(clock.naps, [])                     # so there was never a need to sleep on top of it
        self.assertEqual(GAP, 0.13)


class Asking(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()

    def test_a_busy_server_is_asked_again_after_longer_and_longer_waits(self):
        web = Web({TICKERS: [(503, b"busy"), (502, b"busy"), fixture("company_tickers.json")]}, self.clock)
        with self.assertLogs("stocks.sec", "WARNING") as said:
            self.assertEqual(client(web).tickers()["NWT"], 9900001)
        self.assertEqual(len(said.records), 2)
        self.assertEqual(len(web.asked), 3)
        self.assertEqual([n for n in self.clock.naps if n >= 1], [1, 2])

    def test_a_dead_line_is_tried_five_times_and_then_given_up(self):
        web = Web({TICKERS: urllib.error.URLError("no route to host")}, self.clock)
        with self.assertLogs("stocks.sec", "WARNING"), self.assertRaises(SecError) as c:
            client(web).tickers()
        self.assertEqual((c.exception.status, c.exception.refused, len(web.asked)), (0, False, 5))
        self.assertEqual([n for n in self.clock.naps if n >= 1], [1, 2, 4, 8])

    def test_what_is_not_there_is_not_asked_for_twice(self):
        web = Web(site(), self.clock)
        with self.assertRaises(SecError) as c:
            client(web).submissions(9900002)
        self.assertEqual((c.exception.status, c.exception.refused), (404, False))
        self.assertIn("NoSuchKey", str(c.exception))
        self.assertEqual(web.asked, [SUBS + "CIK0009900002.json"])

    def test_a_refusal_is_never_asked_again(self):
        for status in (403, 429):
            web = Web({TICKERS: (status, b"<html><head><title>SEC.gov | Request Rate Threshold Exceeded</title></head></html>")}, self.clock)
            with self.assertRaises(SecError) as c:
                client(web).tickers()
            self.assertTrue(c.exception.refused)
            self.assertIn("Request Rate Threshold Exceeded", str(c.exception))
            self.assertEqual(len(web.asked), 1)

    def test_an_answer_that_is_not_json_is_an_error_not_a_crash(self):
        web = Web({TICKERS: b"<html>maintenance</html>"}, self.clock)
        with self.assertRaises(SecError):
            client(web).tickers()


class Tickers(unittest.TestCase):
    def test_tickers_and_the_two_spellings_of_a_share_class(self):
        sec = client(Web(site()))
        self.assertEqual(sec.tickers(), {"NWT": 9900001, "RVR-B": 9900003, "RVR-A": 9900003, "GONE": 9900002, "ELSW": 9900009})
        self.assertEqual(sec.ciks(["NWT", "RVR.A", "RVR.B", "rvr-a", "NOPE"]),
                         {"NWT": 9900001, "RVR.A": 9900003, "RVR.B": 9900003, "rvr-a": 9900003})

    def test_the_list_is_fetched_once_a_day(self):
        clock = Clock()
        web = Web(site(), clock)
        sec = client(web)
        sec.tickers(), sec.ciks(["NWT"]), sec.tickers()
        self.assertEqual(web.asked, [TICKERS])
        clock.t += 86_401
        sec.tickers()
        self.assertEqual(web.asked, [TICKERS, TICKERS])


class Submissions(unittest.TestCase):
    def setUp(self):
        self.web = Web(site())
        self.sec = client(self.web)

    def test_the_shape_newest_first_in_new_york_time(self):
        sub = self.sec.submissions(9900001)
        self.assertEqual((sub["cik"], sub["name"], sub["sic"], sub["tickers"]), (9900001, "Northwind Tools Inc", 3540, ["NWT"]))
        self.assertEqual(len(sub["filings"]), 14)
        self.assertEqual(sub["filings"][0], {"accession": "0001999990-26-000412", "form": "4", "items": "", "filing_date": "2026-11-12",
                                             "primary": "xslF345X06/form4.xml", "description": "FORM 4", "file_number": "",
                                             "at": "2026-11-12T17:41:09-05:00"})
        by = {f["accession"]: f for f in sub["filings"]}
        results = by["0009900001-26-000031"]
        self.assertEqual((results["form"], results["items"], results["at"]), ("8-K", "2.02,9.01", "2026-11-12T16:05:11-05:00"))
        self.assertEqual(by["0009900001-26-000021"]["at"], "2026-08-06T07:30:15-04:00")          # summer has its own offset
        self.assertEqual(by["0002000001-26-000077"]["file_number"], "005-99001")
        moments = [datetime.fromisoformat(f["at"]) for f in sub["filings"]]
        self.assertEqual(moments, sorted(moments, reverse=True))
        self.assertEqual(self.web.asked, [SUBS + "CIK0009900001.json"])                          # the older pages were not asked for

    def test_the_older_pages_and_the_ones_too_old_to_want(self):
        everything = self.sec.submissions(9900001, all_pages=True)
        self.assertEqual(len(everything["filings"]), 20)
        self.assertEqual(everything["filings"][-1]["accession"], "0009900001-09-000007")
        self.assertEqual(everything["filings"][-1]["at"], "2009-03-02T16:40:00-05:00")
        self.assertEqual(len(self.web.asked), 3)
        self.web.asked.clear()
        since_2016 = self.sec.submissions(9900001, all_pages=True, since="2016-01-01")
        self.assertEqual(len(since_2016["filings"]), 18)
        self.assertEqual(self.web.asked, [SUBS + "CIK0009900001.json", SUBS + "CIK0009900001-submissions-001.json"])

    def test_a_company_without_an_industry_code(self):
        doc = json.loads(fixture("CIK0009900003.json"))
        doc["sic"] = ""
        self.web.routes[SUBS + "CIK0009900003.json"] = json.dumps(doc).encode()
        sub = self.sec.submissions(9900003)
        self.assertIsNone(sub["sic"])
        self.assertEqual(sub["tickers"], ["RVR-B", "RVR-A"])

    def test_columns_out_of_step_are_refused_not_lined_up_wrongly(self):
        doc = json.loads(fixture("CIK0009900003.json"))
        doc["filings"]["recent"]["form"].pop()
        self.web.routes[SUBS + "CIK0009900003.json"] = json.dumps(doc).encode()
        with self.assertRaises(ValueError):
            self.sec.submissions(9900003)

    def test_a_page_name_that_is_not_a_page_is_not_fetched(self):
        doc = json.loads(fixture("CIK0009900001.json"))
        doc["filings"]["files"][0]["name"] = "../../files/company_tickers.json"
        self.web.routes[SUBS + "CIK0009900001.json"] = json.dumps(doc).encode()
        with self.assertRaises(ValueError):
            self.sec.submissions(9900001, all_pages=True)
        self.assertEqual(self.web.asked, [SUBS + "CIK0009900001.json"])

    def test_where_a_document_is(self):
        self.assertEqual(Sec.filing_url(9900001, "0009900001-26-000031", "nwt-20261112.htm"), COVER)
        self.assertEqual(Sec.filing_url("0009900001", "0009900001-26-000031", ""), INDEX)
        self.assertEqual(Sec.filing_url(9900001, "0001999990-26-000412", "xslF345X06/form4.xml"),
                         ARCH + "9900001/000199999026000412/xslF345X06/form4.xml")
        for accession, primary in (("0009900001-26-31", "a.htm"), ("0009900001-26-000031", "../../x.htm"),
                                   ("0009900001-26-000031", "https://example.com/x.htm"), ("0009900001-26-000031", "a b.htm")):
            with self.assertRaises(ValueError):
                Sec.filing_url(9900001, accession, primary)


class Times(unittest.TestCase):
    """`acceptanceDateTime` is UTC. The strings on the left are as data.sec.gov/submissions/CIK0000320193.json
    gave them on 2026-11-14 for Apple's results 8-Ks, which Apple files as its press release goes out at
    16:30 New York time, summer and winter. (Read as New York digits they would say 20:30 and 21:30.)"""

    def test_apples_results_are_accepted_at_half_past_four_in_new_york_all_year(self):
        published = {
            "2026-11-02T21:30:23.000Z": "2026-11-02T16:30:23-05:00",       # its EDGAR index page: "Accepted 2026-11-02 16:30:23"
            "2026-07-30T20:30:28.000Z": "2026-07-30T16:30:28-04:00",
            "2026-01-29T21:30:33.000Z": "2026-01-29T16:30:33-05:00",
            "2024-10-31T20:30:25.000Z": "2024-10-31T16:30:25-04:00",
            "2012-01-24T21:30:34.000Z": "2012-01-24T16:30:34-05:00",       # from the older page, CIK0000320193-submissions-001.json
        }
        for stamp, new_york in published.items():
            self.assertEqual(ny_time(stamp), new_york)

    def test_a_late_evening_filing_keeps_its_new_york_date(self):
        self.assertEqual(ny_time("2026-11-06T02:54:06.000Z"), "2026-11-05T21:54:06-05:00")       # an Apple FWP, filing date the 6th

    def test_a_time_that_names_no_zone_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            ny_time("2026-11-02T16:30:23")
        with self.assertRaises(ValueError):
            ny_time("yesterday")
        self.assertEqual(ny_time(""), "")

    def test_an_offset_is_honoured(self):
        self.assertEqual(ny_time("2026-11-02T16:30:23-05:00"), "2026-11-02T16:30:23-05:00")
        self.assertEqual(ny_time("2026-11-02T22:30:23+01:00"), "2026-11-02T16:30:23-05:00")


class Text(unittest.TestCase):
    def setUp(self):
        self.web = Web(site())
        self.sec = client(self.web)

    def test_only_the_secs_own_hosts(self):
        for url in ("https://example.com/Archives/edgar/data/9900001/000990000126000031/nwt-20261112.htm",
                    "http://www.sec.gov/Archives/edgar/data/9900001/x.htm", "https://www.sec.gov.example.com/x.htm",
                    "https://www.sec.gov@example.com/x.htm", "https://example.com\\@www.sec.gov/x.htm",
                    "https://user:secret@www.sec.gov/x.htm", "https://www.sec.gov:8443/x.htm", "https://efts.sec.gov/LATEST/search-index?q=x",
                    "ftp://www.sec.gov/x.htm", "//www.sec.gov/x.htm", "file:///etc/passwd", "https://www.sec.gov\n.example.com/x.htm",
                    "www.sec.gov/x.htm", ""):
            with self.assertRaises(ValueError, msg=url):
                self.sec.text(url)
        self.assertEqual(self.web.asked, [])                 # refused before anything left this machine
        self.assertIn("Northwind", self.sec.text(COVER, exhibits=False))
        self.assertIn("Riverbend", self.sec.text(SUBS + "CIK0009900003.json"))
        self.assertEqual(len(self.web.asked), 2)

    def test_scripts_styles_and_what_is_hidden_are_left_out(self):
        text = self.sec.text(COVER, exhibits=False)
        for gone in ("SCRIPT-TEXT", "WRITTEN-BY-SCRIPT", "STYLE-TEXT", "font-family", "HIDDEN-FACT", "HIDDEN-SPAN", "nested in it",
                     "to the end", "COMMENT-MUST", "nwt-20261112", "display", "<", "&amp;", "&#160;", "\xa0"):
            self.assertNotIn(gone, text)
        for kept in ("FORM 8-K", "Date of Report: November 12, 2026", "Item 2.02 Results of Operations and Financial Condition.",
                     "A copy is furnished as Exhibit 99.1.", "Item 9.01 Financial Statements and Exhibits.",
                     "R&D spending rose — again — at the café counter.", "By: /s/ Jane Doe, Chief Financial Officer"):
            self.assertIn(kept, text)
        self.assertIn("UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nFORM 8-K\n", text)      # a line each, none blank

    def test_a_table_row_is_a_line_and_its_cells_stay_apart(self):
        cover = self.sec.text(COVER, exhibits=False).split("\n")
        self.assertIn("Delaware | 001-39901 | 00-0000000", cover)
        self.assertIn("(State of incorporation) | (Commission File Number) | (IRS Employer Identification No.)", cover)
        self.assertIn("99.1 | Press release of Northwind Tools Inc dated November 12, 2026", cover)
        release = self.sec.text(EXHIBIT, exhibits=False).split("\n")
        self.assertIn("(in millions) | Sep. 26, 2026 | Sep. 27, 2025", release)
        self.assertIn("Net sales | $412.6 | $378.4", release)                 # the $ in its own cell joins its figure
        self.assertIn("Operating loss | (12.3) | (4.1)", release)             # and so does the closing bracket
        self.assertIn("Gross margin | 41.2% | 39.8%", release)
        self.assertEqual(release[0], "Northwind Tools Reports Third Quarter 2026 Results")       # not the <title> in the head

    def test_a_long_document_is_cut_and_says_so(self):
        whole = self.sec.text(COVER, exhibits=False)
        self.assertNotIn("[cut here", whole)
        cut = self.sec.text(COVER, 200, exhibits=False)
        body, _, marker = cut.partition("\n\n[cut here")
        self.assertEqual(marker, ": only the first 200 characters are shown]")
        self.assertLessEqual(len(body), 200)
        self.assertTrue(whole.startswith(body))
        self.assertGreater(len(body), 150)

    def test_a_results_8k_brings_its_press_release_along(self):
        text = self.sec.text(COVER)
        self.assertEqual(self.web.asked, [COVER, INDEX, EXHIBIT])              # not the picture, not the XBRL schema
        self.assertIn("\n\n[exhibit EX-99.1: nwt-ex991.htm]\nNorthwind Tools Reports Third Quarter 2026 Results\n", text)
        self.assertIn("Net sales | $412.6 | $378.4", text)
        self.assertLess(text.index("Item 2.02"), text.index("[exhibit EX-99.1"))
        self.assertEqual(text.count("Item 2.02"), 1)                           # the cover is not read twice

    def test_the_press_release_is_not_fetched_when_there_is_no_room_or_no_wish(self):
        self.sec.text(COVER, 600)
        self.sec.text(COVER, exhibits=False)
        self.sec.text(INDEX)                                                   # the index itself is just a page
        self.assertEqual(self.web.asked, [COVER, COVER, INDEX])

    def test_a_filing_without_an_index_is_still_read_but_a_refusal_is_passed_on(self):
        del self.web.routes[INDEX]
        self.assertEqual(self.sec.text(COVER), self.sec.text(COVER, exhibits=False))
        self.web.routes[INDEX] = (403, b"Request Rate Threshold Exceeded")
        with self.assertRaises(SecError) as c:
            self.sec.text(COVER)
        self.assertTrue(c.exception.refused)

    def test_plain_text_keeps_its_columns_and_a_pdf_is_not_text(self):
        self.assertEqual(readable(b"Net sales      412.6\r\nNet income      31.0\r\n"), "Net sales      412.6\nNet income      31.0")
        self.assertEqual(readable(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj"), "[not a text document: 22 bytes]")
        self.assertEqual(readable("café <span>résumé</span>".encode("cp1252")), "café résumé")   # an old filing in Windows-1252


OLD_FORM_4 = b"""<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0306</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2012-05-14</periodOfReport>
    <issuer><issuerCik>0009900001</issuerCik><issuerName>Northwind Tools Inc</issuerName><issuerTradingSymbol>nwt</issuerTradingSymbol></issuer>
    <reportingOwner>
        <reportingOwnerId><rptOwnerCik>0001999006</rptOwnerCik><rptOwnerName>GRAY  ALEX</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>0</isOfficer><isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther></reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>%s</nonDerivativeTable>
</ownershipDocument>"""
LINE = b"""<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle>
<transactionDate><value>2012-05-14-05:00</value></transactionDate>
<transactionCoding><transactionFormType>4</transactionFormType><transactionCode>%s</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
<transactionAmounts><transactionShares>%s</transactionShares><transactionPricePerShare>%s</transactionPricePerShare>
<transactionAcquiredDisposedCode><value>%s</value></transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction>"""


def line(code: bytes, shares: bytes, price: bytes, side: bytes) -> bytes:
    value = lambda v: b"<value>%s</value>" % v if v else b'<footnoteId id="F1"/>'      # noqa: E731
    return LINE % (code, value(shares), value(price), side)


class Form4(unittest.TestCase):
    def setUp(self):
        self.web = Web(site())
        self.sec = client(self.web)

    def test_a_sale_an_exercise_a_grant_and_a_purchase_yield_only_the_purchase(self):
        buys = self.sec.form4(9900001, "0001999990-26-000412", "xslF345X06/form4.xml")
        self.assertEqual(buys, [{"owner": "DOE JANE", "officer": True, "director": False, "ten_pct": False, "title": "Chief Financial Officer",
                                 "shares": 5000.0, "price": 40.1, "usd": 200500.0, "date": "2026-11-11"}])
        self.assertEqual(self.web.asked, [ARCH + "9900001/000199999026000412/form4.xml"])     # the XML as filed, not the page rendered from it

    def test_the_sale_is_kept_apart_and_nothing_else_is_a_trade(self):
        doc = self.sec.ownership(9900001, "0001999990-26-000412", "xslF345X06/form4.xml")
        self.assertEqual((doc["issuer_cik"], doc["symbol"]), (9900001, "NWT"))
        self.assertEqual([(s["shares"], s["price"], s["usd"], s["date"]) for s in doc["sales"]], [(1200.0, 41.25, 49500.0, "2026-11-10")])
        self.assertEqual(len(doc["purchases"]), 1)           # of eight lines: M, S, A, F, P, G in Table I, M and a P of options in Table II

    def test_two_reporting_owners_are_one_buyer(self):
        buys = self.sec.form4(9900001, "0001999990-26-000409", "xslF345X06/form4.xml")
        self.assertEqual([(b["shares"], b["price"], b["date"]) for b in buys],
                         [(10000.0, 39.8, "2026-11-06"), (4000.0, 39.95, "2026-11-09"), (6000.0, 39.8, "2026-11-09")])
        self.assertEqual(sum(b["usd"] for b in buys), 796_600.0)               # each purchase once, not once per owner
        for b in buys:                                       # the director, though the company that sent him is listed first
            self.assertEqual((b["owner"], b["officer"], b["director"], b["ten_pct"], b["title"]), ("ROE RICHARD", False, True, True, "Director"))

    def test_a_company_that_is_the_buyer_is_not_the_company_bought(self):
        self.assertEqual(self.sec.form4(9900003, "0001999990-26-000409", "xslF345X06/form4.xml"), [])
        self.assertEqual(self.sec.ownership(9900003, "0001999990-26-000409", "xslF345X06/form4.xml")["issuer_cik"], 9900001)

    def test_the_old_spelling_of_true_and_a_date_with_a_zone(self):
        doc = read_ownership(OLD_FORM_4 % line(b"P", b"300", b"12.5", b"A"))
        self.assertEqual(doc["purchases"], [{"owner": "GRAY ALEX", "officer": False, "director": True, "ten_pct": False, "title": "Director",
                                             "shares": 300.0, "price": 12.5, "usd": 3750.0, "date": "2012-05-14"}])
        self.assertEqual(doc["symbol"], "NWT")

    def test_a_purchase_that_cannot_be_valued_or_is_no_acquisition_is_left_out(self):
        for code, shares, price, side in ((b"P", b"300", b"", b"A"), (b"P", b"300", b"0", b"A"), (b"P", b"", b"12.5", b"A"),
                                          (b"P", b"0", b"12.5", b"A"), (b"P", b"300", b"12.5", b"D"), (b"S", b"300", b"12.5", b"A"),
                                          (b"J", b"300", b"12.5", b"A"), (b"L", b"300", b"12.5", b"A")):
            doc = read_ownership(OLD_FORM_4 % line(code, shares, price, side))
            self.assertEqual((doc["purchases"], doc["sales"]), ([], []), (code, shares, price, side))
        self.assertEqual(len(read_ownership(OLD_FORM_4 % (line(b"P", b"300", b"", b"A") + line(b"P", b"100", b"12.5", b"A")))["purchases"]), 1)

    def test_what_is_not_an_ownership_document_is_refused(self):
        for body in (b"<html><body>Not found</body></html>", b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><ownershipDocument/>',
                     b"<ownershipDocument><issuer><issuerCik>1</issuerCik></issuer></ownershipDocument>"):
            with self.assertRaises(ValueError):
                read_ownership(body)

    def test_a_form_4_from_before_xml_is_empty_and_costs_no_request(self):
        self.assertEqual(self.sec.form4(9900001, "0009900001-02-000007", "form4.txt"), [])
        self.assertEqual(self.sec.form4(9900001, "0009900001-02-000007", ""), [])
        self.assertEqual(self.web.asked, [])


def bought(cik, sym, filed, accession, owner, officer, director, ten_pct, title, shares, price, usd, date):
    return dict(zip(SET_COLS, (cik, sym, filed, accession, owner, officer, director, ten_pct, title, shares, price, usd, date)))


A = "0001999990-26-000"
IN_THE_SET = [
    bought(9900001, "NWT", "2026-08-14", A + "301", "DOE JANE", True, False, False, "Chief Financial Officer", 2000.0, 35.5, 71000.0, "2026-08-12"),
    bought(9900001, "NWT", "2026-08-14", A + "301", "DOE JANE", True, False, False, "Chief Financial Officer", 1000.0, 35.75, 35750.0, "2026-08-12"),
    bought(9900001, "NWT", "2026-08-17", A + "305", "ROE RICHARD", False, True, True, "Director", 10000.0, 36.0, 360000.0, "2026-08-13"),
    bought(9900009, "ELSW", "2026-09-02", A + "315", "POE EDGAR", False, True, False, "Director", 500.0, 10.0, 5000.0, "2026-09-01"),
    bought(9900002, "GONE", "2026-09-04", A + "330", "LOW SAM", True, True, False, "Chief Executive Officer", 4000.0, 2.5, 10000.0, "2026-09-03"),
    bought(9900003, "RVR.A, RVR.B", "2026-09-09", A + "335", "VANCE IDA", False, True, False, "Director", 250.0, 410.0, 102500.0, "2026-09-08"),
]


class InsiderSets(unittest.TestCase):
    def setUp(self):
        self.web = Web(site())
        self.sec = client(self.web)

    def test_where_each_quarter_is_comes_from_the_page_that_lists_them(self):
        quarters = self.sec.insider_quarters()
        self.assertEqual(sorted(quarters), [(2015, 4), (2026, 1), (2026, 2), (2026, 3)])
        self.assertEqual(quarters[(2026, 3)], f"{WWW}/files/datastandardsinnovation{SETS}2026q3_form345.zip")      # the SEC's, not the mirror's
        self.assertEqual(quarters[(2026, 1)], f"{WWW}/files/structureddata{SETS}2026q1_form345.zip")
        self.sec.insider_quarters()
        self.assertEqual(self.web.asked, [SETS_PAGE])

    def test_a_quarter_is_reduced_to_the_purchases_on_form_4(self):
        df = self.sec.insider_quarter(2026, 3)
        self.assertEqual(list(df.columns), SET_COLS)
        self.assertEqual(df.to_dict("records"), IN_THE_SET)   # no amendment (4/A), no Form 5, no sale, award, exercise, P that disposes, or purchase without a price
        self.assertEqual([str(df[c].dtype) for c in ("cik", "officer", "director", "ten_pct", "shares", "price", "usd")],
                         ["int64", "bool", "bool", "bool", "float64", "float64", "float64"])

    def test_a_quarter_in_which_nobody_bought(self):
        df = self.sec.insider_quarter(2026, 1)
        self.assertEqual((len(df), list(df.columns)), (0, SET_COLS))

    def test_a_quarter_that_is_not_published_is_not_asked_for(self):
        with self.assertRaises(SecError) as c:
            self.sec.insider_quarter(2027, 1)
        self.assertEqual(c.exception.status, 404)
        self.assertEqual(self.web.asked, [SETS_PAGE])

    def test_when_the_list_cannot_be_read_the_two_known_folders_are_tried(self):
        del self.web.routes[SETS_PAGE]
        self.assertEqual(len(self.sec.insider_quarter(2026, 3)), 6)
        self.assertEqual(self.web.asked, [SETS_PAGE, f"{WWW}/files/structureddata{SETS}2026q3_form345.zip",
                                          f"{WWW}/files/datastandardsinnovation{SETS}2026q3_form345.zip"])
        self.sec.insider_quarter(2026, 1)
        self.assertEqual(self.web.asked.count(SETS_PAGE), 1)                   # the missing page is not asked for again

    def test_other_ways_the_same_tables_might_be_written(self):
        """What could not be seen in a real set is read loosely: the file extension and a folder inside the
        zip, an accession number without dashes, a CIK without its zeros, ISO dates, capitals in the
        relationship field."""
        def respell(name: str) -> bytes:
            s = fixture(f"form345/{name}.tsv").decode()
            s = s.replace("0001999990-26-000", "000199999026000").replace("\t0009900", "\t9900")
            for old, new in (("14-AUG-2026", "2026-08-14"), ("17-AUG-2026", "2026-08-17"), ("12-AUG-2026", "2026-08-12"), ("13-AUG-2026", "2026-08-13"),
                             ("Director,Officer", "DIRECTOR, OFFICER"), ("TenPercentOwner", "TENPERCENTOWNER"), ("\tOfficer\t", "\tOFFICER\t"),
                             ("\tDirector\t", "\tDIRECTOR\t")):
                s = s.replace(old, new)
            return s.encode()
        df = read_insider_set(insider_zip({t: respell(t) for t in TABLES}, ext=".txt", folder="2026q3_form345/"))
        self.assertEqual(df.to_dict("records"), IN_THE_SET)

    def test_a_set_without_a_table_or_a_column_says_which(self):
        tables = {t: fixture(f"form345/{t}.tsv") for t in TABLES}
        with self.assertRaisesRegex(ValueError, "REPORTINGOWNER"):
            read_insider_set(insider_zip({t: b for t, b in tables.items() if t != "REPORTINGOWNER"}))
        with self.assertRaisesRegex(ValueError, "TRANS_CODE"):
            read_insider_set(insider_zip({**tables, "NONDERIV_TRANS": tables["NONDERIV_TRANS"].replace(b"TRANS_CODE", b"TRANS_KIND")}))
        with self.assertRaises(zipfile.BadZipFile):
            read_insider_set(b"this is not a zip")


class _Local(BaseHTTPRequestHandler):
    seen: list[tuple[str, dict]] = []

    def do_GET(self):
        self.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
        if self.path == "/packed":
            body = gzip.compress(b'{"packed": "yes"}')
            self.send_response(200)
            self.send_header("content-encoding", "GZIP")
        elif self.path == "/away":
            body = b"moved"
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/elsewhere")
        elif self.path == "/elsewhere":
            body = b"you should not be here"
            self.send_response(200)
        else:
            body = b"<Error><Code>NoSuchKey</Code></Error>"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Wire(unittest.TestCase):
    """The real fetch, against a server on this machine and nowhere else."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Local)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Local.seen.clear()

    def test_the_contact_is_on_the_wire_and_gzip_is_asked_for_and_unpacked(self):
        status, _, body = _http(self.base + "/packed", {"User-Agent": "Plint stocks (ops@example.com)"})
        self.assertEqual((status, body), (200, b'{"packed": "yes"}'))
        path, headers = _Local.seen[0]
        self.assertEqual((path, headers["user-agent"], headers["accept-encoding"]), ("/packed", "Plint stocks (ops@example.com)", "gzip"))

    def test_an_error_page_comes_back_as_a_status_not_an_exception(self):
        status, _, body = _http(self.base + "/nothing", {"User-Agent": "Plint stocks (ops@example.com)"})
        self.assertEqual((status, body), (404, b"<Error><Code>NoSuchKey</Code></Error>"))

    def test_a_redirect_away_from_the_sec_is_not_followed(self):
        status, _, body = _http(self.base + "/away", {"User-Agent": "Plint stocks (ops@example.com)"})
        self.assertEqual((status, body), (302, b"moved"))
        self.assertEqual([path for path, _ in _Local.seen], ["/away"])         # the contact did not travel on

    def test_a_redirect_within_the_sec_is_followed(self):
        ask = urllib.request.Request(f"{WWW}/data/company_tickers.json", headers={"User-Agent": "Plint stocks (ops@example.com)"})
        onward = _StayAtSec().redirect_request(ask, None, 301, "Moved Permanently", {}, TICKERS)
        self.assertEqual((onward.full_url, onward.get_header("User-agent")), (TICKERS, "Plint stocks (ops@example.com)"))
        with self.assertRaises(urllib.error.HTTPError):
            _StayAtSec().redirect_request(ask, None, 301, "Moved Permanently", {}, "https://example.com/company_tickers.json")


if __name__ == "__main__":
    unittest.main()
