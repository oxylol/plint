"""Both sources as two tables: the people, and what each of them traded that a copy could follow.

    people   pid, name, pool ("house" | "insider"), label, first, last
    trades   pid, sym, side ("buy" | "sell" | "part"), usd, day, traded, filed, doc, what

`day` is the first trading day a copy may act on, at its close (DESIGN.md, "Copy trades"):
- House: the trading day after the report's filing date in the Clerk's index (`filings.day_after`).
- Insiders: the day the lab's insider table already carries (the first close after the Form 4 was accepted).
`traded` is the day the person traded (0 when not known); `filed` the day the report was filed.

A ticker that a company has since left is turned into the one it trades under now (`aliases`: META for FB),
because the price table follows the company. A House trade reported twice (an amended report) counts once."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..filings import day_after
from .house import House, size, what_to_copy

PEOPLE_COLS = ["pid", "name", "pool", "label", "first", "last"]
TRADE_COLS = ["pid", "sym", "side", "usd", "day", "traded", "filed", "doc", "what"]
_FIRM = re.compile(r"\b(INC|LLC|L\.?P|LP|LTD|CORP|CORPORATION|CO|COMPANY|FUND|FUNDS|TRUST|CAPITAL|PARTNERS|HOLDINGS|"
                   r"MANAGEMENT|GROUP|ADVISORS?|ADVISERS?|INVESTMENTS?|PLC|AG|SA|NV|BANK|FOUNDATION|ASSOCIATES|VENTURES|"
                   r"EQUITY|L\.?L\.?C|LIMITED|OPPORTUNITY|OPPORTUNITIES|MASTER|OFFSHORE|ONSHORE|ESTATE|GP)\b")


def _int_day(s: str) -> int:
    return int(str(s).replace("-", "")[:8])


def pretty_owner(owner: str) -> str:
    """An SEC reporting owner as people write names: 'WHITE MILES D' → 'Miles D White'. A firm keeps its
    words in their order ('BERKSHIRE HATHAWAY INC' → 'Berkshire Hathaway Inc')."""
    raw = re.sub(r"\s+", " ", str(owner or "")).strip()
    if not raw:
        return ""
    words = [_word(w) for w in raw.split(" ")]
    if _FIRM.search(raw.upper()) or len(words) < 2 or len(words) > 5:
        return " ".join(words)
    tail = [words.pop()] if len(words) > 2 and words[-1].rstrip(".,").upper() in _SUFFIX else []
    return " ".join(words[1:] + words[:1] + tail)


_SUFFIX = ("JR", "SR", "II", "III", "IV")


def _word(w: str) -> str:
    if any(c.islower() for c in w) or w.rstrip(".,") in ("II", "III", "IV", "JR", "SR") or len(w) == 1:
        return w if w.rstrip(".,") not in ("JR", "SR") else w.capitalize()
    return "-".join(part.capitalize() for part in w.split("-"))


def insider_tables(insiders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One person per (owner, company); one buy per person and day (a Form 4 lists a day's purchases line
    by line)."""
    if insiders is None or insiders.empty:
        return pd.DataFrame(columns=PEOPLE_COLS), pd.DataFrame(columns=TRADE_COLS)
    df = insiders.copy()
    df["pid"] = "in:" + df["owner"].astype(str).str.strip() + "|" + df["sym"].astype(str)
    df["filed"] = df["at"].astype(str).str[:10].str.replace("-", "").astype(int)
    g = df.groupby(["pid", "sym", "day"], as_index=False).agg(usd=("usd", "sum"), filed=("filed", "min"),
                                                                doc=("accession", "first"))
    trades = pd.DataFrame({"pid": g["pid"], "sym": g["sym"], "side": "buy", "usd": g["usd"].astype(float),
                           "day": g["day"].astype(int), "traded": 0, "filed": g["filed"].astype(int),
                           "doc": g["doc"].astype(str), "what": "open-market purchase (Form 4)"})
    rows = []
    for pid, p in df.sort_values("day").groupby("pid"):
        r = p.iloc[-1]
        role = str(r["title"] or "").strip()
        role = role.capitalize() if role.isupper() and len(role) > 4 else role      # "CHIEF EXECUTIVE OFFICER"
        if not role:
            role = "officer" if r["officer"] else "director" if r["director"] else "10 % holder" if r["ten_pct"] else "insider"
        rows.append({"pid": pid, "name": pretty_owner(r["owner"]), "pool": "insider", "label": f"{role[:40]} at {r['sym']}",
                     "first": int(p["day"].min()), "last": int(p["day"].max())})
    return pd.DataFrame(rows, columns=PEOPLE_COLS), trades


def house_tables(house: House) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The members and their copyable trades, and counts of what was left out and why."""
    rows = house.rows()
    counts = {"rows": int(len(rows)), "copyable": 0, "not_copied": 0, "repeats": 0}
    if rows.empty:
        return pd.DataFrame(columns=PEOPLE_COLS), pd.DataFrame(columns=TRADE_COLS), counts
    filers = house.filers()
    for col in ("pid", "name", "state"):
        rows[col] = [filers[d][col] if d in filers else v for d, v in zip(rows["doc"].astype(str), rows[col])]
    recs = rows.to_dict("records")
    side = [what_to_copy(r) for r in recs]
    usd = [size(r) for r in recs]
    df = rows.assign(side=side, usd=usd)
    counts["not_copied"] = int((df["side"] == "").sum())
    df = df[df["side"] != ""]
    before = len(df)
    df = df.sort_values(["filed", "doc", "n"])
    key = ["pid", "ticker", "side", "traded", "lo", "hi", "owner"]
    first_doc = df.groupby(key)["doc"].transform("first")      # a trade seen again in a later (amended) report
    df = df[df["doc"] == first_doc]                            # counts once; two alike in one report both count
    counts["repeats"] = before - len(df)
    counts["copyable"] = int(len(df))
    days = {f: day_after(f"{f // 10000:04d}-{f // 100 % 100:02d}-{f % 100:02d}") for f in df["filed"].unique()}
    trades = pd.DataFrame({"pid": df["pid"], "sym": df["ticker"], "side": df["side"], "usd": df["usd"].astype(float),
                           "day": df["filed"].map(days).astype(int), "traded": df["traded"].astype(int),
                           "filed": df["filed"].astype(int), "doc": df["doc"].astype(str),
                           "what": df["asset"].astype(str).str.slice(0, 80)})
    people = []
    for pid, g in rows.groupby("pid"):
        people.append({"pid": pid, "name": str(g["name"].iloc[-1]), "pool": "house", "label": f"House, {g['state'].iloc[-1]}",
                       "first": int(g["filed"].min()), "last": int(g["filed"].max())})
    return pd.DataFrame(people, columns=PEOPLE_COLS), trades, counts


def build(house: House, insiders: pd.DataFrame, aliases: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hp, ht, counts = house_tables(house)
    ip, it = insider_tables(insiders)
    people = pd.concat([hp, ip], ignore_index=True)
    trades = pd.concat([ht, it], ignore_index=True)
    if len(trades):
        trades["sym"] = trades["sym"].map(lambda s: aliases.get(s, s))
        trades = trades.sort_values(["day", "pid", "sym"], kind="stable").reset_index(drop=True)
    counts.update({"house_people": int(len(hp)), "insider_people": int(len(ip)), "house_trades": int(len(ht)),
                   "insider_trades": int(len(it))})
    return people, trades, counts


def save(folder: Path, people: pd.DataFrame, trades: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    people.to_parquet(folder / "people.parquet", index=False)
    trades.to_parquet(folder / "trades.parquet", index=False)


def load(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p, t = folder / "people.parquet", folder / "trades.parquet"
    if not (p.exists() and t.exists()):
        return pd.DataFrame(columns=PEOPLE_COLS), pd.DataFrame(columns=TRADE_COLS)
    return pd.read_parquet(p), pd.read_parquet(t)


def by_person(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {pid: g for pid, g in trades.groupby("pid", sort=False)} if len(trades) else {}


