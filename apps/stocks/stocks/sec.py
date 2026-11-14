"""SEC EDGAR, the parts used here: which company a ticker is, what it has filed and when, the readable text
of a filing, the trades in a Form 4, and the quarterly sets of insider transactions.

Shapes were read from sec.gov on 2026-11-14; a line that leans on one that could not be read says UNVERIFIED.
The SEC's fair-access rule ("Accessing EDGAR Data") is 10 requests a second at most, each naming who asks.
This client stays under 8, sends `User-Agent: Plint stocks (<contact>)`, cannot be built without a contact,
and asks nothing of any host but the SEC's, so the contact travels nowhere else. A 403 or 429 means the SEC
is turning this address away: it is not asked again, and the callers in filings.py stop their run on it.
Make one `Sec` per process and share it: the pacing is kept per object.

Times. `acceptanceDateTime` in the submissions JSON is UTC, as its Z says. Checked on Apple: its results
8-K 0000320193-26-000024 reads 2026-11-02T21:30:23.000Z there and "Accepted 2026-11-02 16:30:23" on its
EDGAR index page, and its results 8-Ks read 20:30 in summer and 21:30 in winter back to 2012, in the recent
block and in the older pages alike. Everything leaves here as New York time with its offset."""
from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from typing import Callable, Iterable

import pandas as pd

from .cal import NY

log = logging.getLogger("stocks.sec")

WWW = "https://www.sec.gov"
DATA = "https://data.sec.gov"
HOSTS = frozenset({"www.sec.gov", "sec.gov", "data.sec.gov"})
GAP = 0.13                  # seconds between requests: 7.7 a second, so no one second ever holds more than 8
TRIES = 5
TICKERS_FOR = 86_400.0      # seconds the ticker list is kept before it is fetched again
SETS_PAGE = f"{WWW}/data-research/sec-markets-data/insider-transactions-data-sets"
SET_DIRS = ("structureddata", "datastandardsinnovation")     # /files/<dir>/data/insider-transactions-data-sets/
SET_COLS = ["cik", "sym", "filed", "accession", "owner", "officer", "director", "ten_pct", "title",
            "shares", "price", "usd", "date"]
ISSUER_COLS = ["cik", "sym", "first", "last", "n"]
EXHIBIT_ROOM = 4_000        # characters that must be left before a filing's press release is fetched as well
EXHIBITS = 2


class SecError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"SEC answered {status}: {message}" if status else f"SEC: {message}")
        self.status = status

    @property
    def refused(self) -> bool:
        """The SEC is turning this address away (asked too fast, or a User-Agent it will not have). Asking
        again only makes it last longer: whoever sees this stops."""
        return self.status in (403, 429)


def sec_url(url: str) -> str:
    """The address, if it is https on one of the SEC's hosts. Anything else is a ValueError."""
    try:
        u = urllib.parse.urlsplit(url)
        ok = (url.isascii() and not any(c <= " " for c in url) and u.scheme == "https"
              and (u.hostname or "").lower() in HOSTS and u.port in (None, 443) and not u.username and not u.password)
    except (ValueError, AttributeError, TypeError):
        ok = False
    if not ok:
        raise ValueError(f"not an SEC address: {str(url)[:120]}")
    return url


class _StayAtSec(urllib.request.HTTPRedirectHandler):
    """A redirect is followed only to another SEC address: the contact in the User-Agent goes nowhere else."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            sec_url(newurl)
        except ValueError:
            raise urllib.error.HTTPError(req.full_url, code, "a redirect away from the SEC was not followed", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_StayAtSec)


def _http(url: str, headers: dict, timeout: float = 60.0) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, headers={**headers, "Accept-Encoding": "gzip"})
    try:
        with _opener.open(req, timeout=timeout) as r:
            body, hdrs, status = r.read(), r.headers, r.status
    except urllib.error.HTTPError as e:
        body, hdrs, status = e.read(), e.headers, e.code
    if hdrs is not None and (hdrs.get("Content-Encoding") or "").lower() == "gzip":
        body = gzip.decompress(body)
    return status, dict(hdrs.items()) if hdrs is not None else {}, body


class Sec:
    def __init__(self, contact: str, fetch: Callable = _http, sleep: Callable = time.sleep,
                 clock: Callable = time.monotonic):
        contact = " ".join((contact or "").split())
        if not contact:
            raise ValueError("the SEC asks every script to say who is asking: no contact, no requests (set SEC_CONTACT)")
        if not (contact.isascii() and contact.isprintable()):
            raise ValueError("the SEC contact must be plain text, such as an e-mail address")
        self._headers = {"User-Agent": f"Plint stocks ({contact})"}
        self._fetch, self._sleep, self._clock = fetch, sleep, clock
        self._lock = threading.Lock()
        self._last = float("-inf")
        self._tickers: tuple[float, dict[str, int]] | None = None
        self._sets: dict[tuple[int, int], str] | None = None
        self.calls = 0

    # ── asking ──────────────────────────────────────────────────────────────────────────────────
    def _turn(self) -> None:
        """Waits until this request may go. Every request passes here, a second try included."""
        with self._lock:
            wait = self._last + GAP - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()

    def _get(self, url: str) -> bytes:
        sec_url(url)
        for attempt in range(TRIES):
            self._turn()
            self.calls += 1
            pause = min(60, 2 ** attempt) if attempt < TRIES - 1 else 0
            try:
                status, _, body = self._fetch(url, self._headers)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                log.warning("sec: %s (try %d)", e, attempt + 1)
                self._sleep(pause)
                continue
            if status == 200:
                return body
            if status >= 500:
                log.warning("sec: %d (try %d)", status, attempt + 1)
                self._sleep(pause)
                continue
            raise SecError(status, _gist(body) or url)          # 404, and 403/429 which must not be asked again
        raise SecError(0, f"no answer after {TRIES} tries: {url}")

    def _json(self, url: str):
        try:
            return json.loads(self._get(url))
        except ValueError:
            raise SecError(200, f"not JSON: {url}") from None

    # ── who is who ──────────────────────────────────────────────────────────────────────────────
    def tickers(self) -> dict[str, int]:
        """Ticker → CIK, tickers as the SEC spells them (BRK-B). The file is {"0": {"cik_str": 320193,
        "ticker": "AAPL", "title": "Apple Inc."}, …} (checked). Kept for a day."""
        if self._tickers is None or self._clock() - self._tickers[0] > TICKERS_FOR:
            table: dict[str, int] = {}
            for row in self._json(f"{WWW}/files/company_tickers.json").values():
                table.setdefault(str(row["ticker"]).upper(), int(row["cik_str"]))
            self._tickers = (self._clock(), table)
        return dict(self._tickers[1])

    def ciks(self, symbols: Iterable[str]) -> dict[str, int]:
        """CIK for each of these symbols that has one. Alpaca writes a share class with a dot (BRK.B), the
        SEC with a hyphen (BRK-B); both are tried. Two symbols of one company share a CIK."""
        table, out = self.tickers(), {}
        for s in symbols:
            u = s.upper()
            for t in (u, u.replace(".", "-"), u.replace(".", ""), u.replace(".", "/")):
                if t in table:
                    out[s] = table[t]
                    break
        return out

    # ── what a company has filed ────────────────────────────────────────────────────────────────
    def submissions(self, cik: int, all_pages: bool = False, since: str = "") -> dict:
        """{cik, name, sic, tickers, filings}: filings newest first, each {accession, form, items, at,
        filing_date, primary, description, file_number}. `at` is the acceptance time in New York ("" where
        the SEC gives none). `filings.recent` holds a year or the last 1,000 filings, whichever is more;
        `all_pages` adds the older pages that `filings.files` lists, those wholly before `since`
        (a filing date, YYYY-MM-DD) left unfetched.

        `file_number` tells whose filing it is where two parties share one: on a Schedule 13D/G it is filled
        (005-…) in the list of the company the stake is in and empty in the holder's; on a Form 3/4/5 it is
        empty in the issuer's list and filled in the insider's (seen in Berkshire's list; not a documented
        rule)."""
        doc = self._json(f"{DATA}/submissions/CIK{int(cik):010d}.json")
        block = doc.get("filings") or {}
        rows = _filings(block.get("recent") or {})
        if all_pages:
            for page in block.get("files") or []:
                if since and str(page.get("filingTo") or "9999") < since:
                    continue
                name = str(page.get("name") or "")
                if not _PAGE.fullmatch(name):
                    raise ValueError(f"submissions: unexpected page name {name[:80]!r}")
                rows += _filings(self._json(f"{DATA}/submissions/{name}"))      # an older page is the bare columns
        seen, out = set(), []
        for f in sorted(rows, key=lambda f: (f["stamp"] or f["filing_date"], f["accession"]), reverse=True):
            if f["accession"] in seen:
                continue
            seen.add(f["accession"])
            f["at"] = ny_time(f.pop("stamp"))
            out.append(f)
        sic = str(doc.get("sic") or "").strip()
        return {"cik": int(cik), "name": str(doc.get("name") or ""), "sic": int(sic) if sic.isdigit() else None,
                "tickers": [str(t) for t in doc.get("tickers") or []], "filings": out}

    @staticmethod
    def filing_url(cik: int, accession: str, primary: str) -> str:
        """Where a filing's document is: /Archives/edgar/data/<cik>/<accession without dashes>/<document>.
        With no document named, the filing's index page."""
        if not _ACCESSION.fullmatch(accession or ""):
            raise ValueError(f"not an accession number: {accession!r}")
        folder = f"{WWW}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
        if not primary:
            return f"{folder}/{accession}-index.htm"
        if not _DOC.fullmatch(primary) or ".." in primary:
            raise ValueError(f"not a document name: {primary!r}")
        return f"{folder}/{primary}"

    # ── reading a filing ────────────────────────────────────────────────────────────────────────
    def text(self, url: str, max_chars: int = 60_000, exhibits: bool = True) -> str:
        """The readable text of a filing document, cut at `max_chars` with a line that says so. SEC addresses
        only; anything else raises ValueError before a request is made.

        An 8-K that announces results is a cover page: the figures are in its press release, exhibit 99.1.
        So when the document leaves room and its filing has EX-99 exhibits, they follow under their own
        headings (two requests more at most; `exhibits=False` reads the one document only)."""
        out = readable(self._get(url), max_chars)
        place = _FOLDER.fullmatch(urllib.parse.urlsplit(url).path)
        if exhibits and place and max_chars - len(out) >= EXHIBIT_ROOM:
            try:
                out = self._with_exhibits(out, place, max_chars)
            except SecError as e:
                if e.refused:
                    raise
                log.debug("sec: exhibits of %s not read: %s", url, e)
            except ValueError as e:
                log.debug("sec: exhibits of %s not read: %s", url, e)
        return out

    def _with_exhibits(self, out: str, place: re.Match, max_chars: int) -> str:
        cik, folder, main = place[1], place[2], place[3]
        accession = f"{folder[:10]}-{folder[10:12]}-{folder[12:]}"
        if main.lower() == f"{accession}-index.htm":
            return out
        # UNVERIFIED in the raw: the index page was read as rendered (a table of Seq, Description, Document,
        # Type, Size, the document's name being a link). Any row with a link and a cell that says EX-99 serves.
        index = _Rows()
        index.feed(self._get(self.filing_url(int(cik), accession, "")).decode("utf-8", "replace"))
        added = 0
        for cells, links in index.rows:
            kind = next((c for c in cells if re.fullmatch(r"EX-99(\.\w+)?", c, re.I)), "")
            name = next((n for n in (h.rsplit("/", 1)[-1] for h in links) if re.search(r"\.(htm|html|txt)$", n, re.I)), "")
            room = max_chars - len(out)
            if not kind or not name or name == main or not _DOC.fullmatch(name) or room < EXHIBIT_ROOM or added >= EXHIBITS:
                continue
            head = f"\n\n[exhibit {kind.upper()}: {name}]\n"
            out += head + readable(self._get(self.filing_url(int(cik), accession, name)), room - len(head))
            added += 1
        return out

    def ownership(self, cik: int, accession: str, primary: str) -> dict:
        """One Form 3/4/5 as {issuer_cik, symbol, purchases, sales}, from its XML as filed: the primary
        document minus the `xslF345X…/` folder, which is the same thing rendered for people. Filings from
        before mid-2003 are not XML: they come back empty, with issuer_cik None."""
        raw = re.sub(r"^xsl[^/]+/", "", primary or "")
        if not raw.lower().endswith(".xml"):
            return {"issuer_cik": None, "symbol": "", "purchases": [], "sales": []}
        return read_ownership(self._get(self.filing_url(cik, accession, raw)))

    def form4(self, cik: int, accession: str, primary: str) -> list[dict]:
        """The open-market purchases of company `cik`'s shares in one Form 4, each {owner, officer, director,
        ten_pct, title, shares, price, usd, date}: non-derivative transactions coded P (purchase) with shares
        acquired (A). Nothing when `cik` is not the issuer: a company's own list also holds the Form 4s it
        files as the buyer of someone else's shares."""
        doc = self.ownership(cik, accession, primary)
        return doc["purchases"] if doc["issuer_cik"] == int(cik) else []

    # ── the quarterly sets of insider transactions ──────────────────────────────────────────────
    def insider_quarters(self) -> dict[tuple[int, int], str]:
        """The quarters published, and where each zip is: {(2026, 3): url}. Read from the page that lists
        them, because the folder changed in 2026 (…/files/structureddata/… up to 2026 Q1,
        …/files/datastandardsinnovation/… from 2026 Q2) and may again."""
        if self._sets is None:
            page = self._get(SETS_PAGE).decode("utf-8", "replace")
            found: dict[tuple[int, int], str] = {}
            for href, year, quarter in _SET_LINK.findall(page):
                try:
                    found[(int(year), int(quarter))] = sec_url(urllib.parse.urljoin(SETS_PAGE, href))
                except ValueError:
                    continue
            self._sets = found
        return dict(self._sets)

    def insider_quarter(self, year: int, quarter: int) -> pd.DataFrame:
        """The open-market purchases reported on Form 4 in one quarter's set, one row per purchase (SET_COLS):
        `cik` the issuer, `sym` its ticker as the filer wrote it, `filed` the filing date, `date` the trade's.
        The sets carry a filing date and no acceptance time."""
        return read_insider_set(self.insider_set(year, quarter))

    def insider_set(self, year: int, quarter: int) -> bytes:
        """One quarter's set as it is published: a zip, for read_insider_set and read_set_issuers."""
        try:
            listed = self.insider_quarters()
        except SecError as e:
            if e.refused:
                raise
            listed = self._sets = {}                # the page cannot be read: go by the two known folders
        if listed and (year, quarter) not in listed:
            raise SecError(404, f"{year} Q{quarter} is not among the published insider sets")
        urls = [listed[(year, quarter)]] if listed else [
            f"{WWW}/files/{d}/data/insider-transactions-data-sets/{int(year)}q{int(quarter)}_form345.zip" for d in SET_DIRS]
        for i, url in enumerate(urls):
            try:
                return self._get(url)
            except SecError as e:
                if e.status != 404 or i == len(urls) - 1:
                    raise
        raise SecError(404, f"no insider set for {year} Q{quarter}")


# ── the submissions JSON ────────────────────────────────────────────────────────────────────────
_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
_DOC = re.compile(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*")
_PAGE = re.compile(r"CIK\d{10}-submissions-\d+\.json")
_FOLDER = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})/([A-Za-z0-9_.\-]+\.(?:htm|html|txt))", re.I)
_COLUMNS = {"accession": "accessionNumber", "form": "form", "filing_date": "filingDate", "stamp": "acceptanceDateTime",
            "items": "items", "primary": "primaryDocument", "description": "primaryDocDescription",
            "file_number": "fileNumber"}
_NEEDED = ("accessionNumber", "form", "filingDate", "acceptanceDateTime")


def ny_time(stamp: str) -> str:
    """An acceptance time from the submissions JSON ("2026-11-02T21:30:23.000Z", which is UTC) as New York
    time ("2026-11-02T16:30:23-05:00"). "" stays "". A stamp that names no zone is refused, not guessed at:
    New York digits read as UTC would put every afternoon's results on the wrong trading day."""
    if not stamp:
        return ""
    t = datetime.fromisoformat(re.sub(r"[Zz]$", "+00:00", stamp.strip()))
    if t.tzinfo is None:
        raise ValueError(f"acceptance time without a zone: {stamp}")
    return t.astimezone(NY).isoformat(timespec="seconds")


def _filings(block: dict) -> list[dict]:
    """A block of column arrays (filings.recent, or an older page, which is such a block and nothing more)
    as one dict per filing."""
    n = len(block.get("accessionNumber") or [])
    cols = {}
    for key, name in _COLUMNS.items():
        col = block.get(name)
        if col is None and name not in _NEEDED:
            col = [""] * n
        if col is None or len(col) != n:
            raise ValueError(f"submissions: column {name} is missing or not as long as accessionNumber")
        cols[key] = col
    return [{key: str(col[i] or "") for key, col in cols.items()} for i in range(n)]


# ── a document as text ──────────────────────────────────────────────────────────────────────────
_VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_DROP = frozenset("script style head noscript template svg ix:header".split())      # never meant to be read
_BLOCK = frozenset("address article aside blockquote br caption center dd div dl dt figure footer form h1 h2 h3 h4 "
                   "h5 h6 header hr li main nav ol p pre section table tbody tfoot thead title tr ul".split())
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
_SPACE = re.compile(r"[\s​﻿]+")
_MARKUP = re.compile(r"<(html|body|div|td|p|span|font|br)\b", re.I)
_CLOSERS = frozenset({")", "%", ")%"})        # a table cell that only closes the figure in the cell before it


class _Enough(Exception):
    pass


class _Reader(HTMLParser):
    """HTML as lines of text. Scripts, styles, the head and anything set not to display (inline XBRL keeps
    its machine-readable facts in such a block) are left out; a table row is one line, its cells divided
    by " | ", with a currency sign or a bracket in a cell of its own joined to its figure."""

    def __init__(self, limit: int):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.room = limit + limit // 4 + 200            # counted before the spaces are tidied away
        self.skip, self.depth = "", 0
        self.cell = False                               # a new cell has begun and nothing is written in it yet
        self.fresh = True                               # nothing on this line yet
        self.tail = ""                                  # the last thing written that was not a space

    def handle_starttag(self, tag, attrs):
        if self.skip:
            self.depth += tag == self.skip and tag not in _VOID
        elif tag not in _VOID and (tag in _DROP or any(k == "style" and v and _HIDDEN.search(v) for k, v in attrs)):
            self.skip, self.depth = tag, 1
        elif tag in ("td", "th"):
            self.cell = True
        elif tag in _BLOCK:
            self._line()

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip:
                self.depth -= 1
                if self.depth <= 0:
                    self.skip = ""
        elif tag in _BLOCK:
            self._line()

    def handle_data(self, data):
        if self.skip:
            return
        s = _SPACE.sub(" ", data)
        word = s.strip()
        if not word:
            if not self.fresh:
                self._add(" ")
            return
        if self.cell and not self.fresh and word not in _CLOSERS and self.tail not in ("$", "("):
            self._add(" | ")
        self.cell, self.fresh, self.tail = False, False, word[-1]
        self._add(s)

    def _line(self):
        if not self.fresh:
            self._add("\n")
        self.fresh, self.cell, self.tail = True, False, ""

    def _add(self, s: str):
        self.parts.append(s)
        self.room -= len(s)
        if self.room < 0:
            raise _Enough

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).split("\n"))
        return "\n".join(line for line in lines if line)


class _Rows(HTMLParser):
    """The rows of the tables in a page: for each, its cells' text and the links in it."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[list[str], list[str]]] = []
        self.inside = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append(([], []))
        elif tag in ("td", "th") and self.rows:
            self.rows[-1][0].append("")
            self.inside = True
        elif tag == "a" and self.rows:
            self.rows[-1][1].extend(v for k, v in attrs if k == "href" and v)

    def handle_endtag(self, tag):
        if tag in ("td", "th", "tr"):
            self.inside = False

    def handle_data(self, data):
        if self.inside and self.rows and self.rows[-1][0]:
            cells = self.rows[-1][0]
            cells[-1] = " ".join((cells[-1] + data).split())


def readable(body: bytes, max_chars: int = 60_000) -> str:
    """A fetched document as text a person could read, at most `max_chars` of it and a line saying it was
    cut when it was. Plain text (and raw XML) is passed through as it is."""
    if body[:5] == b"%PDF-" or b"\x00" in body[:2000]:
        return f"[not a text document: {len(body):,} bytes]"
    try:
        s = body.decode("utf-8")
    except UnicodeDecodeError:
        s = body.decode("cp1252", "replace")
    if _MARKUP.search(s):
        reader = _Reader(max_chars)
        try:
            reader.feed(s)
            reader.close()
        except _Enough:
            pass
        s = reader.text()
    else:
        s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + f"\n\n[cut here: only the first {max_chars:,} characters are shown]"
    return s


def _gist(body: bytes) -> str:
    """The first words of an error page, without its markup."""
    return " ".join(re.sub(r"<[^>]*>", " ", body[:2000].decode("utf-8", "replace")).split())[:160]


# ── ownership documents (Forms 3, 4, 5) ─────────────────────────────────────────────────────────
def _true(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true")        # the schema allows both spellings


def _num(v) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def _name(v) -> str:
    """A reporting owner's name in one spelling: EDGAR keeps most in capitals, some not, and the same
    person must count as one buyer whichever file the name came from."""
    return " ".join(str(v or "").split()).upper()


def filer(owners: list[dict]) -> dict:
    """One name for a filing made by several reporting owners together. Such a filing reports each trade
    once, for all of them (a fund and the director it sent, a couple, three related partnerships): one
    purchase, one buyer, the dollars counted once. "There is no primary owner", says the specification, so
    the order they are listed in means nothing: the name used is an officer's if there is one, else a
    director's, else a ten-percent holder's, the first by alphabet among equals. The officer, director and
    ten_pct flags say whether any of them is one."""
    if not owners:
        raise ValueError("ownership document without a reporting owner")
    lead = min(owners, key=lambda o: (not o["officer"], not o["director"], not o["ten_pct"], o["owner"]))
    title = lead["title"] or ("Director" if lead["director"] else "10% owner" if lead["ten_pct"] else "")
    return {"owner": lead["owner"], "officer": any(o["officer"] for o in owners),
            "director": any(o["director"] for o in owners), "ten_pct": any(o["ten_pct"] for o in owners), "title": title}


def read_ownership(body: bytes) -> dict:
    """An ownership XML document (checked against a Form 4 filed 2026-11-12, schema X0609, and the Ownership
    XML Technical Specification 5.5) as {issuer_cik, symbol, purchases, sales}. Purchases are Table I rows
    coded P with shares acquired, sales those coded S with shares disposed of; each needs a number of shares
    and a price above zero (one without cannot be valued, and is usually a gift or an award miscoded).
    Derivatives (Table II), awards, exercises, gifts and tax withholding are not trades and are left out."""
    if b"<!DOCTYPE" in body or b"<!ENTITY" in body:
        raise ValueError("ownership document with a DTD: not read")
    root = ET.fromstring(body)
    if root.tag != "ownershipDocument":
        raise ValueError(f"not an ownership document: <{root.tag}>")
    owners = []
    for o in root.iterfind("reportingOwner"):
        rel = o.find("reportingOwnerRelationship")
        flag = lambda name: _true(rel.findtext(name)) if rel is not None else False     # noqa: E731
        owners.append({"owner": _name(o.findtext("reportingOwnerId/rptOwnerName")), "officer": flag("isOfficer"),
                       "director": flag("isDirector"), "ten_pct": flag("isTenPercentOwner"),
                       "title": " ".join((rel.findtext("officerTitle") or "").split()) if rel is not None else ""})
    who = filer(owners)
    out = {"issuer_cik": int(root.findtext("issuer/issuerCik") or 0), "purchases": [], "sales": [],
           "symbol": (root.findtext("issuer/issuerTradingSymbol") or "").strip().upper()}
    for t in root.iterfind("nonDerivativeTable/nonDerivativeTransaction"):
        code = (t.findtext("transactionCoding/transactionCode") or "").strip().upper()
        side = (t.findtext("transactionAmounts/transactionAcquiredDisposedCode/value") or "").strip().upper()
        kind = {("P", "A"): "purchases", ("S", "D"): "sales"}.get((code, side))
        shares = _num(t.findtext("transactionAmounts/transactionShares/value"))
        price = _num(t.findtext("transactionAmounts/transactionPricePerShare/value"))
        if kind and shares > 0 and price > 0:
            out[kind].append({**who, "shares": shares, "price": price, "usd": round(shares * price, 2),
                              "date": (t.findtext("transactionDate/value") or "").strip()[:10]})
    return out


# ── the quarterly sets ──────────────────────────────────────────────────────────────────────────
_SET_LINK = re.compile(r"""href=["']([^"']*?(\d{4})q([1-4])_form345\.zip)["']""", re.I)
_MONTHS = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}


def _set_date(s) -> str:
    """A date from the sets, DD-MON-YYYY by their readme ("31-MAR-2006"), as YYYY-MM-DD. An ISO date is
    taken too; anything else is ""."""
    s = str(s or "")
    m = re.fullmatch(r"\s*(\d{1,2})-([A-Za-z]{3})-(\d{4})\s*", s)
    if m and m[2].upper() in _MONTHS:
        return f"{m[3]}-{_MONTHS[m[2].upper()]:02d}-{int(m[1]):02d}"
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", s)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else ""


def _accession(s) -> str:
    """An accession number with its dashes, however it was written."""
    d = re.sub(r"\D", "", str(s or ""))
    return f"{d[:10]}-{d[10:12]}-{d[12:]}" if len(d) == 18 else str(s or "").strip()


def read_insider_set(body: bytes) -> pd.DataFrame:
    """One quarter's "Insider Transactions Data Set" (a zip of tab-separated tables) reduced to the
    open-market purchases reported on Form 4, the same ones `read_ownership` finds in the XML.

    Table and column names, the tab, UTF-8 and the DD-MON-YYYY dates are from the SEC's readme
    (insider_transactions_readme.pdf). UNVERIFIED, because a set is too large to have been opened here: the
    files' extension (matched by name alone, whatever follows the dot), how an accession number and a CIK
    are written (taken with or without dashes and leading zeros), and the exact words of
    RPTOWNER_RELATIONSHIP (the readme says OFFICER, DIRECTOR, TENPERCENTOWNER, OTHER; looked for anywhere in
    the field, in any case). Amendments (4/A) restate trades already there and Form 5 reports them a year
    late: neither is read."""
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        members = {re.sub(r"\.[^.]*$", "", n.rsplit("/", 1)[-1]).upper(): n for n in z.namelist()}

        def table(stem: str, want: list[str], keep=None) -> pd.DataFrame:
            if stem not in members:
                raise ValueError(f"insider set: no {stem} table among {sorted(members)}")
            with z.open(members[stem]) as f:
                parts = []
                for part in pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in want, quoting=csv.QUOTE_NONE,
                                        keep_default_na=False, on_bad_lines="skip", encoding="utf-8",
                                        encoding_errors="replace", chunksize=200_000):
                    missing = [c for c in want if c not in part.columns]
                    if missing:
                        raise ValueError(f"insider set: {stem} has no column {', '.join(missing)}")
                    parts.append(keep(part) if keep else part)
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=want)

        trades = table("NONDERIV_TRANS", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                          "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"],
                       keep=lambda p: p[(p["TRANS_CODE"].str.strip().str.upper() == "P")
                                        & (p["TRANS_ACQUIRED_DISP_CD"].str.strip().str.upper() == "A")])
        forms = table("SUBMISSION", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"],
                      keep=lambda p: p[p["DOCUMENT_TYPE"].str.strip() == "4"])
        people = table("REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    for t in (trades, forms, people):
        t["accession"] = t["ACCESSION_NUMBER"].map(_accession)
    df = trades.merge(forms.drop_duplicates("accession"), on="accession", how="inner")
    people = people[people["accession"].isin(set(df["accession"]))]
    who = {}
    for accession, g in people.groupby("accession"):
        rel = g["RPTOWNER_RELATIONSHIP"].str.lower()
        who[accession] = filer([{"owner": _name(n), "officer": "officer" in r, "director": "director" in r,
                                 "ten_pct": "tenpercent" in r.replace(" ", "").replace("_", ""), "title": " ".join(str(t).split())}
                                for n, r, t in zip(g["RPTOWNERNAME"], rel, g["RPTOWNER_TITLE"])])
    df = df[df["accession"].isin(who)]
    out = pd.DataFrame({
        "cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"),
        "sym": df["ISSUERTRADINGSYMBOL"].str.strip().str.upper(),
        "filed": df["FILING_DATE"].map(_set_date),
        "accession": df["accession"],
        "owner": df["accession"].map(lambda a: who[a]["owner"]),
        "officer": df["accession"].map(lambda a: who[a]["officer"]).astype(bool),
        "director": df["accession"].map(lambda a: who[a]["director"]).astype(bool),
        "ten_pct": df["accession"].map(lambda a: who[a]["ten_pct"]).astype(bool),
        "title": df["accession"].map(lambda a: who[a]["title"]),
        "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce").astype("float64"),
        "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce").astype("float64"),
        "date": df["TRANS_DATE"].map(_set_date),
    })
    out = out[(out["shares"] > 0) & (out["price"] > 0) & out["cik"].notna() & (out["filed"] != "")].copy()
    out["cik"] = out["cik"].astype("int64")
    out["usd"] = (out["shares"] * out["price"]).round(2)
    return out[SET_COLS].sort_values(["filed", "accession"], kind="stable").reset_index(drop=True)


# ── who traded under which ticker, and when ─────────────────────────────────────────────────────
_VENUE = re.compile(r"^(?:NYSE(?:\s?AMERICAN|\s?ARCA|\s?MKT)?|NASDAQ(?:GS|GM|CM)?|AMEX|OTC[A-Z]*|TSX[A-Z]*)\s*[:\-]\s*")
_TICKER = re.compile(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?")


def tickers_in(field) -> list[str]:
    """The tickers in an ISSUERTRADINGSYMBOL field, in this app's spelling (a share class after a dot).
    Filers write it by hand: "NYSE: ABC", "RVR.A, RVR.B", "BRK-B", "none". What is not a ticker is left out."""
    out = []
    for part in re.split(r"[,;]|\s+AND\s+|\s+/\s+", str(field or "").upper()):
        t = _VENUE.sub("", " ".join(part.split()).strip("()[]\"' "))
        t = re.sub(r"^([A-Z]{1,5})[\-/ ]([A-Z]{1,2})$", r"\1.\2", t)
        if _TICKER.fullmatch(t) and t not in ("NONE", "NA", "N.A") and t not in out:
            out.append(t)
    return out


def read_set_issuers(body: bytes) -> pd.DataFrame:
    """Which company its insiders filed under which ticker in one quarter's set: a row per issuer and ticker
    (ISSUER_COLS) with the first and last filing date and the number of filings, every Form 3, 4 and 5
    counted. The SEC's ticker list names only companies listed today; one that was bought or delisted is
    found here, under the ticker it had while it traded."""
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        names = {re.sub(r"\.[^.]*$", "", n.rsplit("/", 1)[-1]).upper(): n for n in z.namelist()}
        if "SUBMISSION" not in names:
            raise ValueError(f"insider set: no SUBMISSION table among {sorted(names)}")
        want = ["FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]
        with z.open(names["SUBMISSION"]) as f:
            df = pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in want, quoting=csv.QUOTE_NONE,
                             keep_default_na=False, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f"insider set: SUBMISSION has no column {', '.join(missing)}")
    df = pd.DataFrame({"cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"), "filed": df["FILING_DATE"].map(_set_date),
                       "sym": df["ISSUERTRADINGSYMBOL"].map(tickers_in)})
    df = df[df["cik"].notna() & (df["filed"] != "")].explode("sym").dropna(subset=["sym"])
    if df.empty:
        return pd.DataFrame(columns=ISSUER_COLS)
    out = df.groupby(["cik", "sym"], as_index=False).agg(first=("filed", "min"), last=("filed", "max"), n=("filed", "size"))
    out["cik"] = out["cik"].astype("int64")
    return out[ISSUER_COLS]

