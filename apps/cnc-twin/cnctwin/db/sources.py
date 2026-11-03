"""Where every number comes from: the bibliography and the source rules.

Rules (db/_templates/README.md, docs/formats.md "Databases"), checked on each file **as written**
(before key migration, so paths and lines are the ones in the file):

* every number in a database file has a source: its own ``src`` (``{value: x, src: key}``) or
  the ``src`` of the nearest enclosing mapping (section-level inheritance, through lists too);
* a ``src`` is a key of the merged bibliography (``db/bibliography/*.yaml``; the same key may be
  defined in several files only with identical entries) or ``est``;
* an ``est`` needs a ``note`` saying how the value was derived: its own, or the note written
  with the ``src: est`` of the section it inherits from. A missing note is a warning (the value
  is still listed as an estimate, with an empty note) and an error in strict mode;
* every number whose effective source is ``est`` is listed as an :class:`Estimate`;
* definitional values need no source: the values of a material ``skin`` of ``type: none``
  (thickness 0, factors 1: there is no skin) (:data:`DEFINITIONAL`).

A bibliography key may be defined in several files. It must name the same work everywhere: the
identity fields (title, year, and authors, edition, doi, isbn, issn where both definitions give
them; compared case- and punctuation-insensitively) must agree, else it is an error. The
descriptive fields (type, venue, publisher, url, volume ...) may be written differently: the first
definition (files in name order) is kept and a warning names the fields to unify. The reading
fields (``pages``, ``note``, ``accessed``) say where and when each file read the work and are merged.

Violations are problems of kind ``source``; :func:`cnctwin.db.problems.raise_for` turns them into a
:class:`~cnctwin.core.errors.SourceError` with file, line and key path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .problems import Estimate, Problem, format_path
from .schema import is_number, is_number_like

EST = "est"
# Sections whose values are definitional (no source): kind -> ((path, predicate on the mapping), ...).
DEFINITIONAL = {
    "material": ((("skin",), lambda m: isinstance(m, Mapping) and m.get("type") == "none"),),
}
IDENTITY_ALWAYS = ("title", "year")
IDENTITY_IF_BOTH = ("authors", "edition", "doi", "isbn", "issn")
READING_FIELDS = ("pages", "note", "accessed", "language", "original")
# Keys that describe a source rather than being values.
CITATION_KEYS = ("src", "note", "notes", "page", "pages", "digitised", "uncertainty_pct")
EXEMPT_TOP = ("format", "version")


@dataclass(frozen=True)
class Decl:
    """A ``src`` written at ``path`` (with the note written beside it)."""

    src: str
    note: str | None
    path: tuple


class Bibliography:
    """The merged ``db/bibliography/*.yaml`` entries."""

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.origin: dict[str, tuple[str, int]] = {}
        self.problems: list[Problem] = []

    def add(self, key: str, entry: dict, file: str, line: int) -> None:
        if key == EST:
            self.problems.append(Problem(file, line, f"entries.{key}", "'est' is reserved for estimates and "
                                         "cannot be a bibliography key", "duplicate"))
            return
        if key in self.entries:
            known = self.entries[key]
            first_file, first_line = self.origin[key]
            differs = [f for f in IDENTITY_ALWAYS if _norm(known.get(f)) != _norm(entry.get(f))]
            differs += [f for f in IDENTITY_IF_BOTH if known.get(f) is not None and entry.get(f) is not None
                        and _norm(known.get(f)) != _norm(entry.get(f))]
            if differs:
                self.problems.append(Problem(file, line, f"entries.{key}",
                                             f"bibliography key {key!r} names a different work in "
                                             f"{first_file}:{first_line} ({', '.join(differs)} differ)",
                                             "duplicate"))
                return
            described = sorted(f for f in set(known) | set(entry)
                               if f not in IDENTITY_ALWAYS and f not in READING_FIELDS
                               and _norm(known.get(f)) != _norm(entry.get(f)))
            if described:
                self.problems.append(Problem(file, line, f"entries.{key}",
                                             f"bibliography key {key!r} is described differently in "
                                             f"{first_file}:{first_line} ({', '.join(described)}); the first "
                                             f"definition is used", "duplicate", "warning"))
            merged = dict(known)
            for field_name in READING_FIELDS:
                a, b = known.get(field_name), entry.get(field_name)
                if b is not None and a != b:
                    merged[field_name] = b if a is None else f"{a} | {b}"
            self.entries[key] = merged
            return
        self.entries[key] = entry
        self.origin[key] = (file, line)

    def __contains__(self, key) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def keys(self) -> list[str]:
        return sorted(self.entries)


def _canonical(entry) -> str:
    return json.dumps(entry, sort_keys=True, default=str)


def _norm(value) -> str:
    """Case-, space- and punctuation-insensitive text of a field (None for an absent one)."""
    if value is None:
        return ""
    return re.sub(r"[^0-9a-z]+", " ", str(value).casefold()).strip()


def check_sources(data, locations, bibliography: Bibliography, *, obj: str = "", path: tuple = (),
                  inherited: Decl | None = None, exempt: tuple = EXEMPT_TOP, strict: bool = False,
                  kind: str = "") -> tuple[list[Problem], list[Estimate]]:
    """Source problems and estimates of ``data`` (one file, or one subtree of it).

    ``inherited`` is the declaration in force above ``path``. Keys in ``exempt`` at the top of
    ``data`` are not values (``format``, ``version``). ``kind`` selects the definitional sections.
    """
    walker = _Walker(locations, bibliography, obj, strict, DEFINITIONAL.get(kind, ()))
    if isinstance(data, Mapping):
        decl = walker.declaration(data, path) or inherited
        for key, value in data.items():
            if key in CITATION_KEYS or key in exempt:
                continue
            walker.walk(value, path + (key,), decl)
    else:
        walker.walk(data, path, inherited)
    return walker.problems, walker.estimates


def check_reference(src, path, locations, bibliography: Bibliography, *, obj: str = "") -> list[Problem]:
    """A single ``src`` that must name a bibliography entry (validation-case references)."""
    file, line = locations.get(path) if locations is not None else ("", 0)
    if not isinstance(src, str) or not src:
        return [Problem(file, line, format_path(path), "src must be a bibliography key", "source", "error", obj)]
    if src not in bibliography:
        return [Problem(file, line, format_path(path), f"source {src!r} is not in db/bibliography/*.yaml",
                        "source", "error", obj)]
    return []


class _Walker:
    def __init__(self, locations, bibliography, obj, strict, definitional=()):
        self.locations = locations
        self.bibliography = bibliography
        self.obj = obj
        self.strict = strict
        self.definitional = definitional
        self.problems: list[Problem] = []
        self.estimates: list[Estimate] = []
        self._missing: set = set()

    def _where(self, path):
        if self.locations is None:
            return "", 0
        return self.locations.get(path)

    def problem(self, path, message, severity="error"):
        file, line = self._where(path)
        self.problems.append(Problem(file, line, format_path(path), message, "source", severity, self.obj))

    def declaration(self, mapping, path) -> Decl | None:
        if "src" not in mapping:
            return None
        src = mapping["src"]
        where = path + ("src",)
        if not isinstance(src, str) or not src:
            self.problem(where, "src must be a bibliography key or est")
            return None
        note = mapping.get("note")
        note = note if isinstance(note, str) and note.strip() else None
        if src != EST and src not in self.bibliography:
            self.problem(where, f"source {src!r} is not in db/bibliography/*.yaml")
        return Decl(src, note, path)

    def walk(self, node, path, decl):
        if isinstance(node, Mapping):
            if any(path == where and test(node) for where, test in self.definitional):
                return
            if "value" in node or "table" in node:
                self.sourced(node, path, decl)
                return
            own = self.declaration(node, path) or decl
            for key, value in node.items():
                if key in CITATION_KEYS:
                    continue
                self.walk(value, path + (key,), own)
            return
        if isinstance(node, list):
            if node and is_number_like(node):
                self.leaf(node, path, decl)
                return
            for index, item in enumerate(node):
                self.walk(item, path + (index,), decl)
            return
        if is_number(node):
            self.leaf(node, path, decl)

    def sourced(self, mapping, path, decl):
        own = self.declaration(mapping, path)
        effective = own or decl
        value = mapping.get("table") if "table" in mapping else mapping.get("value")
        extras = {k: v for k, v in mapping.items()
                  if k not in ("value", "table", "range", "x") and k not in CITATION_KEYS}
        numeric = value is not None and (is_number_like(value) or isinstance(value, list))
        has_numbers = numeric or any(is_number_like(v) for v in extras.values())
        for key, extra in extras.items():
            if isinstance(extra, (Mapping, list)) and not is_number_like(extra):
                self.walk(extra, path + (key,), effective)
        if not has_numbers:
            return
        if effective is None:
            self.problem(path, "no source: give src: <bibliography key> here or on an enclosing section, "
                               "or src: est with a note")
            return
        if effective.src == EST:
            note = (own.note if own is not None else None) or (decl.note if own is None and decl else None)
            self.estimate(path, value if numeric else {k: v for k, v in extras.items() if is_number_like(v)},
                          note, own if own is not None else effective)

    def leaf(self, value, path, decl):
        if decl is None:
            self.problem(path, "no source: give src: <bibliography key> here or on an enclosing section, "
                               "or src: est with a note")
            return
        if decl.src == EST:
            self.estimate(path, value, decl.note, decl)

    def estimate(self, path, value, note, decl: Decl):
        file, line = self._where(path)
        if not note:
            where = decl.path + ("src",)
            key = format_path(where)
            if key not in self._missing:
                self._missing.add(key)
                self.problem(where, "src: est needs a note saying how the value was derived",
                             "error" if self.strict else "warning")
        self.estimates.append(Estimate(file, line, format_path(path), value, note or "", self.obj))


__all__ = ["Bibliography", "Decl", "EST", "check_sources", "check_reference", "CITATION_KEYS"]
