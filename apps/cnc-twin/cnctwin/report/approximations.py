"""The model approximations relevant to a run, from docs/approximations.md (when the build has it).

The generated document (cnctwin.docs_tools) is one table split into a section per domain, with the columns
``domain | approximation | consequence | error bound | evidence | source`` (:func:`read_rows`). The report keeps
the rows that concern what the run exercised (:func:`relevant_rows`): a row's domain must have been active in the
run (the run domains the report finds active, :data:`ALIASES` mapping table domains without channels of their own:
``mechanics`` with ``cutting``, ``geom`` with ``inspect``; :data:`EVERY_RUN` domains always; :data:`NEVER` never)
and the row must state an error bound. They are sorted by impact (:func:`impact`: the largest relative error the
bound states, in percent; a bound without one, an exactness statement or an absolute error, after them) and the
report shows the first :data:`MAX_ROWS` of them with a link to the full table (:data:`FULL_TABLE_URL`).

A document that is not in the generated form (no such table) is read as Markdown sections (``#`` headings): a
section is relevant when its heading or its text names an active domain, or its heading names one of the topics
every run depends on (:data:`ALWAYS`); when no section names any domain every section is listed. The text is then
rendered by a small Markdown subset (paragraphs, ``*``/``-`` bullets, ``|`` tables, ``code``, ``**bold**``), escaped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .html import Raw, esc

# The generated table (cnctwin.docs_tools Approximations): its header, its "not stated" cell, where the full
# document is served (cnctwin.api routes_docs) and how many rows the report shows.
TABLE_COLUMNS = ("domain", "approximation", "consequence", "error bound", "evidence", "source")
NOT_STATED = "not stated"
FULL_TABLE_URL = "/api/docs/approximations.md"
MAX_ROWS = 40
# table domains decided by other run domains (they have no channels, findings or trace events of their own)
ALIASES = {"mechanics": ("cutting",), "geom": ("inspect",), "hotloop": ("hotloop", "cutting")}
EVERY_RUN = ("controller", "motion", "loop")                 # every run executes its program through the loop
NEVER = ("scope", "performance", "gpu")                      # scope decisions and CPU cost, not result errors

DOMAINS = ("controller", "motion", "tooling", "removal", "cutting", "mechanics", "chips", "thermal", "coolant",
           "wear", "holding", "structure", "dynamics", "surface", "inspect", "production", "sanity", "sound",
           "render")
ALWAYS = ("general", "scope", "core", "loop", "clock", "frames", "conservation", "recording", "determinism",
          "numerics", "all runs")


@dataclass
class DocSection:
    heading: str
    level: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def names(self, words, heading_only: bool = False) -> set[str]:
        haystack = (self.heading if heading_only else self.heading + "\n" + self.text).lower()
        return {w for w in words if re.search(rf"\b{re.escape(w)}\b", haystack)}


def default_path() -> Path:
    from ..paths import DOCS
    return Path(DOCS) / "approximations.md"


def read_sections(path: Path) -> list[DocSection]:
    sections: list[DocSection] = []
    current: DocSection | None = None
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            current = DocSection(match.group(2).strip(), len(match.group(1)))
            sections.append(current)
            continue
        if current is None:
            current = DocSection("(introduction)", 1)
            sections.append(current)
        current.lines.append(line)
    return sections


@dataclass(frozen=True)
class Row:
    """One row of the generated table."""

    domain: str
    approximation: str
    consequence: str
    error_bound: str
    evidence: str
    source: str
    order: int

    @property
    def stated(self) -> bool:
        return bool(self.error_bound.strip()) and self.error_bound.strip().lower() != NOT_STATED

    @property
    def impact(self) -> float | None:
        return impact(self.error_bound) if self.stated else None


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def read_rows(path: Path) -> list[Row]:
    """The rows of every table with the generated header (:data:`TABLE_COLUMNS`), in document order; [] for a
    document without one."""
    rows: list[Row] = []
    columns: dict[str, int] | None = None
    in_code = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        if in_code or not stripped.startswith("|"):
            columns = None if not stripped.startswith("|") else columns
            continue
        cells = _cells(stripped)
        lowered = [c.lower() for c in cells]
        if all(name in lowered for name in TABLE_COLUMNS):
            columns = {name: lowered.index(name) for name in TABLE_COLUMNS}
            continue
        if columns is None or all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        get = lambda name: cells[columns[name]] if columns[name] < len(cells) else ""  # noqa: E731
        rows.append(Row(get("domain").lower(), get("approximation"), get("consequence"), get("error bound"),
                        get("evidence"), get("source"), len(rows)))
    return rows


_PERCENT = re.compile(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)(?:\s*(?:-|to)\s*([-+]?\d+(?:\.\d+)?))?\s*%")
_FACTOR = re.compile(r"\bfactors?\s+(?:of\s+)?(?:about\s+|~\s*)?(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?",
                     re.IGNORECASE)
_TIMES = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*x\b(?!\s*\d)")
_ORDER = re.compile(r"\border[- ]of[- ]magnitude\b", re.IGNORECASE)


def impact(text: str) -> float | None:
    """The largest relative error an error-bound text states, in percent: percentages (``+-50 %``, ``15-30 %``,
    ``-20 to +43 %``, a validation row's ``p90 181.6 %``), factors (``a factor 2``, ``factor 2-5``, ``3.8 x``: a factor
    f is (f - 1) x 100 %), an order of magnitude (900 %). None when it states none (exact, ulps, an absolute error)."""
    found: list[float] = []
    for match in _PERCENT.finditer(text):
        found += [abs(float(v)) for v in match.groups() if v is not None]
    for match in _FACTOR.finditer(text):
        found += [(float(v) - 1.0) * 100.0 for v in match.groups() if v is not None and float(v) > 1.0]
    for match in _TIMES.finditer(text):
        if float(match.group(1)) > 1.0:
            found.append((float(match.group(1)) - 1.0) * 100.0)
    if _ORDER.search(text):
        found.append(900.0)
    return max(found) if found else None


def exercised(table_domain: str, active: set[str]) -> bool:
    """Whether a table domain concerns a run whose active domains are ``active``."""
    domain = table_domain.strip().lower()
    if domain in NEVER:
        return False
    if domain in EVERY_RUN:
        return True
    return domain in active or any(d in active for d in ALIASES.get(domain, ()))


def relevant_rows(rows: list[Row], active: set[str]) -> list[Row]:
    """The rows of exercised domains that state an error bound, by impact (largest first; a bound without a relative
    error after them), then in document order."""
    active = {d.lower() for d in active if d}
    picked = [r for r in rows if r.stated and exercised(r.domain, active)]
    return sorted(picked, key=lambda r: (r.impact is None, -(r.impact or 0.0), r.order))


def relevant(sections: list[DocSection], domains: set[str]) -> tuple[list[DocSection], str]:
    """(sections relevant to a run that used ``domains``, how they were chosen)."""
    tagged = [s for s in sections if s.names(DOMAINS)]
    if not tagged:
        return [s for s in sections if s.text.strip()], "the document is not organised by domain: all sections"
    used = {d.lower() for d in domains if d}
    picked = [s for s in sections if s.text.strip() and (s.names(used) or s.names(ALWAYS, heading_only=True))]
    return picked, ("sections naming a domain this run used (" + ", ".join(sorted(d for d in domains if d)) +
                    ") or a topic of every run")


def _inline(text: str) -> str:
    out = esc(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return out


def render(section: DocSection) -> Raw:
    """A small Markdown subset to HTML (escaped)."""
    html: list[str] = []
    bullets: list[str] = []
    rows: list[list[str]] = []
    paragraph: list[str] = []

    def flush():
        if paragraph:
            html.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()
        if bullets:
            html.append("<ul>" + "".join(f"<li>{_inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()
        if rows:
            body = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in r if c.strip())]
            if body:
                head, *rest = body
                html.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head)
                            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r)
                                                               + "</tr>" for r in rest) + "</tbody></table></div>")
            rows.clear()

    for line in section.lines:
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("|"):
            if paragraph or bullets:
                flush()
            rows.append([c.strip() for c in stripped.strip("|").split("|")])
        elif re.match(r"^[*-]\s+", stripped):
            if paragraph or rows:
                flush()
            bullets.append(re.sub(r"^[*-]\s+", "", stripped))
        elif bullets and line.startswith((" ", "\t")):
            bullets[-1] += " " + stripped
        else:
            if bullets or rows:
                flush()
            paragraph.append(stripped)
    flush()
    return Raw("".join(html))


__all__ = ["DocSection", "Row", "read_sections", "read_rows", "relevant", "relevant_rows", "impact", "exercised", "render",
           "default_path", "DOMAINS", "ALWAYS", "ALIASES", "EVERY_RUN", "NEVER", "MAX_ROWS", "FULL_TABLE_URL"]
