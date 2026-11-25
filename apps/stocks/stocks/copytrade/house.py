"""House members' trades, from the Clerk's own files (disclosures-clerk.house.gov).

The Clerk publishes a yearly index of every financial disclosure (a zip holding an XML list) and each
periodic transaction report as a PDF. A report filed electronically has a DocID that starts with 2 and its
text can be read; one filed on paper is a scan and is not read here (about a fifth of them).

    <cache>/copytrade/house/<year>FD.zip            the index for a year (FilingType "P" is a trade report)
    <cache>/copytrade/house/pdf/<year>/<doc>.pdf    the report
    <cache>/copytrade/house/rows.parquet            every trade read from the reports, one row each
    <cache>/copytrade/house/docs.parquet            every report looked at, with how many rows it gave

A report's text comes in two layouts (before and after 2020) and its labels in a font whose letters come
out in odd cases ("FIlINg STaTuS", "(AAPl)", "F      S     : New"). `parse` reads both; the tests hold
real reports of every era (tests/fixtures/house/)."""
from __future__ import annotations

import io
import logging
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger("stocks.copytrade.house")

BASE = "https://disclosures-clerk.house.gov/public_disc"
AGENT = "plint-stocks/1.0 (personal research; paper trading only)"
PAUSE = 0.4                     # seconds between requests: the Clerk's server is a public office, not a CDN
IN_A_ROW = 8                    # failures one after another, and the run stops
ROW_COLS = ["doc", "year", "pid", "name", "state", "filed", "n", "owner", "asset", "typ", "ticker", "kind",
            "traded", "notified", "lo", "hi", "notes"]
DOC_COLS = ["doc", "year", "pid", "filed", "rows", "status"]


class Stopped(Exception):
    """The Clerk's server turned the run away or stopped answering. What was read is saved."""


# ── reading a report's text ─────────────────────────────────────────────────────────────────────
_DATE = r"\d{1,2}/\d{1,2}/\d{2,4}"
_TX = re.compile(r"^(?P<pre>(?:.*?\s)?)(?P<tx>[PSE](?:\s*\(partial\))?)\s+(?P<d1>" + _DATE + r")\s+(?P<d2>" + _DATE + r")(?P<rest>.*)$", re.I)
_AMOUNT = re.compile(r"(over\s+\$[\d,]+|\$[\d,]+(?:\s*-\s*(?:\$[\d,]+)?)?)", re.I)
_TAIL_AMOUNT = re.compile(r"\s*(\$[\d,]+)\s*$")
_OWNER = re.compile(r"^(SP|JT|DC)\s+", re.I)
_BOXES = re.compile(r"\b(?:gfedcb|gfedc|nmlkji|nmlkj)\b")      # the check boxes, as their font's letters
_LABEL = re.compile(r"^(?:[A-Za-z](?:\s+[A-Za-z])*\s*:|(?:filing status|description|subholding of|location|comments?)\s*:)", re.I)
_END = re.compile(r"^(?:\*\s*for the complete list|initial public offering|i\s+p\s+o\b|asset class details|"
                  r"certification and signature|c\s+s\s*$|i certify|yes\s+no\s*$|digitally signed)", re.I)
_BOILER = re.compile(r"^(?:filing id\s*#|id\s+owner\s+asset|type\s+date|\$200\?|p\s+t\s+r\s*$|eriodic|periodic transaction|"
                     r"clerk of the house|filer information|name\s*:|status\s*:|state/district\s*:|transactions\s*$|"
                     r"f\s+i\s*$|t\s*$)", re.I)
_TICKER = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$")


def _date(s: str) -> int:
    """mm/dd/yyyy (or yy) as YYYYMMDD; 0 when it is not a date."""
    try:
        m, d, y = (int(x) for x in s.split("/"))
        y = y + 2000 if y < 100 else y
        return date(y, m, d).year * 10000 + m * 100 + d
    except (ValueError, TypeError):
        return 0


def _dollars(s: str) -> int:
    digits = re.sub(r"[^\d]", "", s or "")
    return int(digits) if digits else 0


def amount(text: str) -> tuple[int, int]:
    """'$15,001 - $50,000' → (15001, 50000); 'Over $50,000,000' → (50000000, 0): no top."""
    t = (text or "").strip()
    if t.lower().startswith("over") or "over $" in t.lower():
        return _dollars(t.lower().split("over", 1)[1]), 0
    parts = [p for p in re.split(r"\s*-\s*", t) if p.strip()]
    lo = _dollars(parts[0]) if parts else 0
    hi = _dollars(parts[1]) if len(parts) > 1 else 0
    return lo, hi


def asset_bits(asset: str) -> tuple[str, str]:
    """The asset type in brackets ('ST', 'OP', 'GS' …, '' when the report has none) and the ticker in
    parentheses ('' when there is none: a bond's CUSIP or a foreign SEDOL is not a ticker)."""
    typ = ""
    m = re.findall(r"\[([A-Za-z]{2})\]", asset)
    if m:
        typ = m[-1].upper()
    ticker = ""
    for c in re.findall(r"\(([A-Za-z][A-Za-z0-9.\-/]{0,9})\)", asset):
        c = c.upper().replace("-", ".").replace("/", ".")
        if _TICKER.match(c):
            ticker = c
    return typ, ticker


def parse(text: str) -> list[dict]:
    """The trades in one report's text, in the order written. Each: owner ('SELF', 'SP' spouse, 'JT' joint,
    'DC' dependent), asset (as written), typ, ticker, kind ('P', 'S', 'S (partial)', 'E'), traded and
    notified (YYYYMMDD), lo and hi (dollars; hi 0 = open-ended), notes (filing status, description …)."""
    rows: list[dict] = []
    cur: dict | None = None
    mode = ""

    def close():
        nonlocal cur
        if cur is not None:
            cur["asset"] = re.sub(r"\s+", " ", cur["asset"]).strip()
            cur["typ"], cur["ticker"] = asset_bits(cur["asset"])
            cur["lo"], cur["hi"] = amount(cur.pop("amount"))
            cur["notes"] = re.sub(r"\s+", " ", cur["notes"]).strip()[:300]
            rows.append(cur)
        cur = None

    for raw in (text or "").splitlines():
        s = _BOXES.sub("", raw).replace("\x00", " ").strip()      # the label font's small letters come out as NULs
        if not s:
            continue
        if _END.match(s):
            close()
            mode = ""
            continue
        m = _TX.match(s)
        if m and _date(m["d1"]) and _date(m["d2"]):
            close()
            pre = m["pre"].strip()
            owner = "SELF"
            o = _OWNER.match(pre)
            if o:
                owner, pre = o.group(1).upper(), pre[o.end():]
            a = _AMOUNT.search(m["rest"])
            kind = "S (partial)" if "partial" in m["tx"].lower() else m["tx"][0].upper()
            cur = {"owner": owner, "asset": pre, "kind": kind, "traded": _date(m["d1"]), "notified": _date(m["d2"]),
                   "amount": a.group(1) if a else "", "notes": ""}
            mode = "asset"
            continue
        if cur is None or _BOILER.match(s):
            continue
        if _LABEL.match(s):
            mode = "notes"
            cur["notes"] += " " + s
            continue
        if mode == "asset":
            open_amount = cur["amount"].rstrip().endswith("-")
            if s.startswith("$") and open_amount:
                cur["amount"] += " " + s
                continue
            t = _TAIL_AMOUNT.search(s)
            if t and open_amount:
                cur["amount"] += " " + t.group(1)
                s = s[: t.start()]
            cur["asset"] += " " + s
        else:
            cur["notes"] += " " + s
    close()
    return rows


def what_to_copy(row: dict) -> str:
    """'buy', 'sell' (all of it), 'part' (some of it) or '' (not copied), by DESIGN.md's rule: stocks,
    funds and call options as their stock; puts, bonds, mutual funds and the rest are not."""
    if not row.get("ticker"):
        return ""
    typ = row.get("typ", "")
    words = f"{row.get('asset', '')} {row.get('notes', '')}".lower()
    is_option = typ == "OP" or "option" in words
    if is_option:
        if re.search(r"\bputs?\b", words) or not re.search(r"\bcalls?\b", words):
            return ""
    elif typ not in ("", "ST", "EF"):
        return ""
    return {"P": "buy", "S": "sell", "S (partial)": "part"}.get(row.get("kind", ""), "")


def size(row: dict) -> float:
    """The dollars a trade counts for: the middle of its range; an open-ended one ('over X') counts as X."""
    lo, hi = float(row.get("lo") or 0), float(row.get("hi") or 0)
    return (lo + hi) / 2.0 if hi > lo else lo


# ── the index ───────────────────────────────────────────────────────────────────────────────────
def _slug(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


_TITLES = re.compile(r"\b(?:dr|hon|honorable|mr|mrs|ms|rev)\b\.?", re.I)


def person(last: str, first: str, state_dst: str) -> tuple[str, str, str]:
    """(pid, the name as written for people, the state). A member is the same person across years and
    districts within a state. Titles some filers put in their first name ('Mark Dr', 'Hon. James') go."""
    first = re.sub(r"\s+", " ", _TITLES.sub(" ", first or "").strip())
    last = re.sub(r"\s+", " ", (last or "").strip())
    state = (state_dst or "")[:2].upper()
    given = first.split(" ")[0] if first else ""
    pid = f"house:{_slug(last)}:{_slug(given)}:{state.lower()}"
    return pid, f"{first} {last}".strip(), state


def read_index(body: bytes) -> list[dict]:
    """The trade reports in a year's index zip: doc, year, pid, name, state, filed (YYYYMMDD), electronic."""
    out = []
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        name = next((n for n in z.namelist() if n.lower().endswith(".xml")), None)
        if name is None:
            return out
        root = ET.fromstring(z.read(name))
    for e in root.iter("Member"):
        if (e.findtext("FilingType") or "").strip() != "P":
            continue
        doc = (e.findtext("DocID") or "").strip()
        filed = _date((e.findtext("FilingDate") or "").strip())
        year = int((e.findtext("Year") or "0").strip() or 0)
        if not doc or not filed:
            continue
        pid, nm, state = person(e.findtext("Last") or "", e.findtext("First") or "", e.findtext("StateDst") or "")
        suffix = (e.findtext("Suffix") or "").strip()
        out.append({"doc": doc, "year": year or filed // 10000, "pid": pid, "name": f"{nm} {suffix}".strip(),
                    "state": state, "filed": filed, "electronic": doc.startswith("2")})
    return out


def pdf_text(body: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(body)) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


# ── fetching ────────────────────────────────────────────────────────────────────────────────────
def _http(url: str, timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


class House:
    def __init__(self, cache_dir: Path, fetch: Callable = _http, sleep: Callable = time.sleep,
                 text_of: Callable = pdf_text):
        self.dir = Path(cache_dir) / "copytrade" / "house"
        self.fetch, self.sleep, self.text_of = fetch, sleep, text_of
        self._fails = 0

    # the tables
    def rows(self) -> pd.DataFrame:
        p = self.dir / "rows.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=ROW_COLS)

    def docs(self) -> pd.DataFrame:
        p = self.dir / "docs.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=DOC_COLS)

    def _save(self, rows: list[dict], docs: list[dict]) -> None:
        if not rows and not docs:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        if rows:
            both = pd.concat([self.rows(), pd.DataFrame(rows, columns=ROW_COLS)], ignore_index=True)
            both = both.drop_duplicates(["doc", "n"], keep="last")
            both.to_parquet(self.dir / "rows.parquet", index=False)
        if docs:
            both = pd.concat([self.docs(), pd.DataFrame(docs, columns=DOC_COLS)], ignore_index=True)
            both.drop_duplicates(["doc"], keep="last").to_parquet(self.dir / "docs.parquet", index=False)

    def _get(self, url: str) -> bytes | None:
        """The body, or None for a page the server does not have. Raises Stopped when it turns us away."""
        for attempt in range(3):
            self.sleep(PAUSE * (1 + 4 * attempt))
            try:
                status, body = self.fetch(url)
            except (OSError, urllib.error.URLError) as e:
                status, body = 0, b""
                log.debug("%s: %s", url, e)
            if status == 200:
                self._fails = 0
                return body
            if status in (403, 429):
                raise Stopped(f"the Clerk's server answered {status} for {url}")
            if status == 404:
                return None
        self._fails += 1
        if self._fails >= IN_A_ROW:
            raise Stopped(f"{IN_A_ROW} requests in a row failed (the last: {url})")
        return None

    def index(self, year: int, fresh_hours: float = 20.0) -> list[dict]:
        """A year's trade reports. The current year's index is fetched again once it is older than
        `fresh_hours` (the Clerk adds to it daily); earlier years are fetched once."""
        path = self.dir / f"{year}FD.zip"
        this_year = datetime.now(timezone.utc).year
        stale = not path.exists() or (year >= this_year - 1 and time.time() - path.stat().st_mtime > fresh_hours * 3600)
        if stale:
            body = self._get(f"{BASE}/financial-pdfs/{year}FD.zip")
            if body:
                self.dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
        return read_index(path.read_bytes()) if path.exists() else []

    def sync(self, years: list[int], say: Callable[[str], None] = log.info, limit: int | None = None) -> dict:
        """Read every electronic trade report in these years' indexes that has not been read yet. Saves as it
        goes; a Stopped run keeps what it read and the next run carries on from there."""
        seen = set(self.docs()["doc"].astype(str))
        out = {"reports": 0, "paper": 0, "trades": 0, "failed": 0, "new_docs": []}
        rows, docs = [], []
        try:
            for year in years:
                todo = []
                for f in self.index(year):
                    if not f["electronic"]:
                        out["paper"] += 1
                    elif f["doc"] not in seen:
                        todo.append(f)
                for i, f in enumerate(todo):
                    if limit is not None and out["reports"] >= limit:
                        break
                    got, status = self._read_one(f)
                    rows += got
                    docs.append({"doc": f["doc"], "year": f["year"], "pid": f["pid"], "filed": f["filed"],
                                 "rows": len(got), "status": status})
                    seen.add(f["doc"])
                    out["reports"] += 1
                    out["trades"] += len(got)
                    out["failed"] += status != "read"
                    out["new_docs"].append(f["doc"])
                    if len(docs) >= 100:
                        self._save(rows, docs)
                        rows, docs = [], []
                        say(f"House reports {year}: {i + 1} of {len(todo)} read")
        finally:
            self._save(rows, docs)
        return out

    def _read_one(self, f: dict) -> tuple[list[dict], str]:
        path = self.dir / "pdf" / str(f["year"]) / f"{f['doc']}.pdf"
        if not path.exists():
            body = self._get(f"{BASE}/ptr-pdfs/{f['year']}/{f['doc']}.pdf")
            if body is None:
                return [], "missing"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        try:
            found = parse(self.text_of(path.read_bytes()))
        except Exception as e:                      # a broken PDF is one report lost, not the run
            log.warning("report %s could not be read: %s", f["doc"], e)
            return [], "unreadable"
        return [{"doc": f["doc"], "year": f["year"], "pid": f["pid"], "name": f["name"], "state": f["state"],
                 "filed": f["filed"], "n": n, **r} for n, r in enumerate(found)], "read"

    def filers(self) -> dict[str, dict]:
        """doc → {pid, name, state}, worked out again from the indexes on disk, so a better reading of names
        reaches reports read before it (the latest spelling of a person's name wins)."""
        by_doc: dict[str, dict] = {}
        latest: dict[str, str] = {}
        for path in sorted(self.dir.glob("*FD.zip")):
            for f in read_index(path.read_bytes()):
                by_doc[f["doc"]] = {"pid": f["pid"], "name": f["name"], "state": f["state"]}
                latest[f["pid"]] = f["name"]
        for v in by_doc.values():
            v["name"] = latest[v["pid"]]
        return by_doc
