"""Which validation cases depend on their own reference data (docs/validation.md "Calibration dependence").

A case is **calibration-dependent** when a database value the harness's model chain used was built
from the same work as the case's reference: the same paper supplies the Taylor constants, a flow-stress
set chosen for its agreement with the same experiments, a friction coefficient identified from the
same force data, a fit made "here" over the same tables. Such cases are reported separately from the
independent validation and do not count toward a quantity's verdict.

Two bibliography keys name the same work when (any of):

* their DOI, or their URL, is the same;
* their normalised titles and years are the same;
* the note of one entry says it is the same work, the same document, or a version (preprint, journal
  version) of the other and names the other key (the convention of db/bibliography);
* they are listed together in :data:`SAME_DATA` below (data sets the bibliography does not link:
  sections of one catalogue).

A used database value depends on the reference when its ``src`` is in the reference's group, or when
it is an estimate (``src: est`` or no source) whose note names a key of that group (the database's
convention for values "fitted here" to a source; the note of a sourced value only names
alternatives, so it does not count).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .registry import Provenance

# Sources that report the same experiments or data under keys the bibliography does not link
# (each tuple is one data set; the reason is shown in the report).
SAME_DATA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ETH Zurich dry orthogonal-cutting database of Ti6Al4V and Ck45: the arXiv report (2022) and the IJAMT article "
     "(2024) publish the same tests",
     ("klippel-2022-arxiv-orthogonal-data", "klippel-2022-ck45-cutting-database", "klippel-2024-ck45-orthogonal")),
    ("Tungaloy General Catalogue 2023-2024: the metric sections and the US turning edition carry the same "
     "chipbreaker data",
     ("tungaloy-2023-gc-insert", "tungaloy-2023-gc-turning-us", "tungaloy-2023-gc-grade",
      "tungaloy-2023-gc-ext-toolholder", "tungaloy-2023-gc-int-toolholder")),
)

SAME_PHRASES = ("same work", "same document", "journal version", "preprint", "same tests", "same experiments",
                "same data")


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, a: str) -> str:
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


@dataclass
class Works:
    """Groups of bibliography keys that name the same work or data set."""

    groups: dict[str, frozenset]
    reasons: dict[frozenset, list]

    def same(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        return b in self.groups.get(a, frozenset((a,)))

    def group(self, key: str) -> frozenset:
        return self.groups.get(key, frozenset((key,)))


def build_works(entries: dict) -> Works:
    """Works from bibliography entries ``{key: entry dict}`` (all files merged)."""
    uf = _UnionFind()
    reasons: dict[tuple, str] = {}
    by_doi: dict[str, str] = {}
    by_url: dict[str, str] = {}
    by_title: dict[str, str] = {}
    keys = sorted(entries)
    known = set(keys)
    for key in keys:
        e = entries[key] or {}
        uf.find(key)
        doi = _norm(e.get("doi"))
        if doi:
            if doi in by_doi:
                uf.union(key, by_doi[doi])
                reasons[(by_doi[doi], key)] = "same DOI"
            else:
                by_doi[doi] = key
        url = str(e.get("url") or "").strip().lower().rstrip("/")
        if url:
            if url in by_url:
                uf.union(key, by_url[url])
                reasons[(by_url[url], key)] = "same URL"
            else:
                by_url[url] = key
        title = _norm(e.get("title"))
        if title:
            tkey = f"{title}|{e.get('year')}"
            if tkey in by_title:
                uf.union(key, by_title[tkey])
                reasons[(by_title[tkey], key)] = "same title and year"
            else:
                by_title[tkey] = key
        note = str(e.get("note") or "")
        if note and any(p in note.lower() for p in SAME_PHRASES):
            for other in re.findall(r"[a-z0-9][a-z0-9._-]*[a-z0-9]", note):
                if other in known and other != key:
                    uf.union(key, other)
                    reasons[(key, other)] = "linked in the bibliography note"
    for reason, group in SAME_DATA:
        members = [k for k in group]
        for other in members[1:]:
            uf.union(members[0], other)
            reasons[(members[0], other)] = reason
    groups: dict[str, set] = {}
    for key in list(uf.parent):
        groups.setdefault(uf.find(key), set()).add(key)
    by_key = {}
    group_reasons: dict[frozenset, list] = {}
    for members in groups.values():
        frozen = frozenset(members)
        for m in members:
            by_key[m] = frozen
        group_reasons[frozen] = sorted({r for (a, b), r in reasons.items() if a in members and b in members})
    return Works(by_key, group_reasons)


def works_from_db(db) -> Works:
    """Works of the database's merged bibliography (``Database.bib``)."""
    bib = db.bib
    return build_works({key: dict(bib.get(key) or {}) for key in bib.keys()})


@dataclass(frozen=True)
class Dependency:
    what: str
    src: str
    via: str                     # 'src' | 'note'
    note: str = ""


def dependencies(reference_src: str, used: list[Provenance], works: Works) -> list[Dependency]:
    """The used database values that come from the reference's own work."""
    group = works.group(reference_src)
    out = []
    for p in used:
        if p.src and p.src in group:
            out.append(Dependency(p.what, p.src, "src", p.note))
            continue
        if p.src not in ("", "est"):
            continue                # a sourced value: keys in its note are alternatives or comments
        mentioned = [k for k in re.findall(r"[a-z0-9][a-z0-9._-]*[a-z0-9]", p.note or "") if k in group]
        if mentioned:
            out.append(Dependency(p.what, mentioned[0], "note", p.note))
    seen = set()
    unique = []
    for d in out:
        if (d.what, d.src) not in seen:
            seen.add((d.what, d.src))
            unique.append(d)
    return unique


__all__ = ["Works", "build_works", "works_from_db", "dependencies", "Dependency", "SAME_DATA"]
