"""Documentation generators (``python -m cnctwin.docs_tools``): docs/approximations.md and docs/doc-gaps.md.

``python -m cnctwin.docs_tools approximations [--root DIR] [--out PATH] [--check]``
    Regenerates docs/approximations.md: one table with the columns domain, approximation, consequence,
    error bound, evidence and source, split into one section per domain (so the run report,
    cnctwin.report.approximations, can pick the sections of the domains a run used). Rows come from

    * the model docs: docs/models/<domain>.md, and docs/dialects.md for the controller (it has no
      docs/models/controller.md). Every section whose heading matches :data:`COLLECT_PATTERNS`, and every
      section below one that does, is split into statements (a bullet with its continuation lines, a
      table row, a paragraph); outside those sections, statements that start with a label such as
      "Validity:" or "Error:" (:data:`LEAD_PATTERN`) are collected too; a doc with no matching heading is
      listed as not matching;
    * the scope decisions of docs/architecture.md (the bullets after ``**Scope decisions**``);
    * the validation medians of validation/results/*.json (the independent cases of each quantity);
    * the performance summary of docs/performance.md (its timing scatter, the ``**Status.**`` paragraph,
      the tables of the sections named "Whole runs" and "Memory", the bullets of "Open problems").

    ``--check`` writes nothing and exits 1 when the file differs from what would be written.

``python -m cnctwin.docs_tools check [--root DIR] [--out PATH] [--strict]``
    Writes docs/doc-gaps.md: for every model doc, which of the required items of docs/architecture.md
    section 14 (equations, parameters and sources, assumptions, validity range, error bound, references)
    it has under a recognised heading (:data:`REQUIRED_ITEMS`), has only as text elsewhere
    (:data:`TEXT_ONLY`), or lacks. ``--strict`` exits 1 when a doc lacks one.

The columns of a statement (heuristics, deterministic, repeated in the generated file):

* table rows: columns are given roles by their header (:data:`COLUMN_ROLES`); the first column is the
  approximation when no header names one, a validity column is appended to it, unmapped columns too;
* bullets and paragraphs: approximation = the first sentence; consequence = the text after an explicit
  ``Consequence:``, else the following sentences, else "not stated";
* error bound = the table's error column, else every clause (split at ``;`` and sentence ends) with an
  error expression (:data:`ERROR_CUES`: percentages, ``+-``, ``<=``, factors, ``1e-9``, "within",
  "exact", "order of magnitude" ...), else "not stated";
* evidence = the classes of :data:`EVIDENCE_CLASSES` whose cues appear in the statement (validation case:
  the validation suite, its quantities or case ids; convergence/test: convergence studies, refinement,
  tests against exact, analytic or reference solutions; literature: a bibliography key of db/bibliography,
  an author-year citation, a standard or catalogue; estimate: ``est``, estimated, assumed, typical,
  judgement), with the keys, case ids and test names found; "not stated" when none.

Cells are clipped to :data:`CELL_CHARS` characters and never hold ``|`` (written as ``¦``): the report's
Markdown reader splits table rows on every ``|``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .paths import ROOT

# =============================================================================== settings (presentation only)

APPROXIMATIONS_FILE = "docs/approximations.md"
GAPS_FILE = "docs/doc-gaps.md"
CELL_CHARS = {"approximation": 260, "consequence": 260, "error_bound": 200, "evidence": 160}
MIN_WORDS = 4                    # a statement needs at least this many words (skips stray labels)
MIN_EQUATIONS = 3                # equation lines a doc needs before "equations" counts as present
EVIDENCE_REFS = 3                # keys, case ids or test names named per evidence class

# A model doc's section is collected for docs/approximations.md when its heading (level 2 or deeper, the
# section number and code spans removed) matches one of these, or it lies below a heading that does.
COLLECT_PATTERNS = {
    "assumptions": r"\bassumptions?\b",
    "limitations": r"\blimitations?\b|\bknown gaps\b|\bopen problems\b",
    "validity": r"\bvalidity\b|\bvalid\b",
    "error bounds": r"(?<!location )\berror\b|\berror bounds?\b|\baccuracy\b|\bexactness\b|\btolerances\b",
    "approximations": r"\bapproximations?\b|\bsimplifications?\b",
    "consequences": r"\bconsequences?\b",
    "validation and verification": r"\bvalidation\b|\bverification\b",
}

# Roles of table columns, by a word of the header (first match in this order wins).
COLUMN_ROLES = (
    ("consequence", r"\bconsequences?\b|\bmeans\b"),
    ("validity", r"\bvalidity\b|\bvalid\b|\brange\b"),
    ("error_bound", r"\berror\b|\baccuracy\b|\bbound\b|\bresult\b|\btolerance\b|\bdeviation\b|\buncertainty\b"),
    ("evidence", r"\bbasis\b|\bestablished\b|\bevidence\b|\bhow\b|\bsource\b|\bsrc\b|\btest\b"),
    ("approximation", r"\bpart\b|\bquantity\b|\bwhat\b|\bitem\b|\bmodel\b|\bapproximation\b|\bassumption\b|"
                      r"\bcomparison\b|\bcase\b|\bcheck\b|\beffect\b|\bstage\b|\bkernel\b|\brule\b|\bquantities\b"),
)

# A clause states an error bound when it holds a strong cue, or a weak cue together with an error word.
ERROR_CUES = {
    "strong": re.compile(
        r"\+-\s*\d|±\s*\d"                                                     # +-30 %
        r"|(?:<=|≤)\s*\S"                                                      # <= r dphi^2 / 4
        r"|\b\d(?:\.\d+)?e-\d+\b"                                              # 1e-9
        r"|\bexact(?:ly)?\b|\borders? of magnitude\b|x/÷\s*\d|\bulp\b"
        r"|\bfactors?\s+(?:of\s+)?(?:~|about\s+|up\s+to\s+|of\s+about\s+)?\d", re.IGNORECASE),
    "weak": re.compile(
        r"\d(?:\.\d+)?(?:\s*(?:-|to)\s*\+?\d+(?:\.\d+)?)?\s*%"                 # 15 %, 30-95 %
        r"|(?:[<>]|~|≈)\s*\d"                                                  # < 2, ~5
        r"|\b(?:within|up to|at most|below|above|about)\s+\S*\d", re.IGNORECASE),
    "words": re.compile(
        r"\berrors?\b|\bwithin\b|\baccura\w*|\bdeviat\w*|\bbound\b|\buncertain\w*|\bscatter\w*|\boff by\b"
        r"|\bdiffer\w*|\bagree\w*|\b(?:over|under)[- ]?(?:estimat|predict)\w*"
        r"|\btoo (?:high|low|thick|thin|large|small)\b"
        r"|\bhigh\b|\blow\b|\bresidual\b|\btolerances?\b|\bprecision\b|\bresolution\b|\bmiss\w*|\brelative\b"
        r"|\bnumerical\b|\bmeasured\b|\bvs\b|\bagainst\b", re.IGNORECASE),
}

# (class, cue pattern matched case-insensitively, cue pattern matched case-sensitively or None)
EVIDENCE_CLASSES = (
    ("validation case", r"\bvalidat(?:ion|ed)\b|validation-report", None),
    ("convergence/test", r"\bconverg\w*|\brefine(?:d|ment)?\b|\bhalv\w*|\bbrute[- ]force\b|\bbit-identical\b"
                         r"|\btest(?:s|ed)?\b|\btest_\w+|\b(?:against|vs\.?)\s+(?:the\s+)?(?:exact|analytic\w*|"
                         r"closed[- ]form|reference|brute)\b|\bquadrature\b|\breproduced\b", None),
    ("literature", r"\bet al\b|\bhandbook\b|\bcatalogue\b|\bpublished\b|\breported\b|\bliterature\b|\bdatasheet\b"
                   r"|\bmanual\b",
     r"\(\d{4}\)|\b[A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)?,?\s+(?:19|20)\d{2}\b|\b(?:ISO|VDI|DIN|ANSI)\s+\d+"),
    ("estimate", r"\best\b|\bestimat\w*|\bassum\w*|\bjudgement\b|\btypical\b|\border of magnitude\b|\bplaceholder\b",
     None),
)

# Required items of a model doc (architecture 14), found by a heading (level 2 or deeper, the section number
# and code spans removed) or by the other evidence the rule names (check_doc); TEXT_ONLY finds the items a doc
# states without such a heading.
REQUIRED_ITEMS = {
    "equations": {"heading": None,
                  "rule": f"at least {MIN_EQUATIONS} equation lines: an '=' (not '==', '<=', '>=', '=>') with an "
                          "operator or a function (+ - * / ^ sqrt exp ln log sin cos tan atan min max sum) beside it, "
                          "in inline code, an untagged or math code block, or plain text"},
    "parameters and sources": {"heading": r"\bparameters?\b|\bsettings\b|\bconstants\b|\bcoefficients\b",
                               "rule": "a parameters / settings / constants / coefficients heading whose section names "
                                       "sources (a table column source/src/basis, src, est, db/models/<file>.yaml, "
                                       "db/bibliography, a bibliography key), or any table with a source column"},
    "assumptions": {"heading": r"\bassumptions?\b|\bdecisions\b|\bsimplifications?\b",
                    "rule": "a heading with assumption(s), decisions or simplification(s)"},
    "validity range": {"heading": r"\bvalidity\b|\bvalid\b",
                       "rule": "a heading with validity / valid, or a table column validity / valid / range"},
    "error bound": {"heading": r"(?<!location )\berror\b|\berror bounds?\b|\baccuracy\b|\bexactness\b|\btolerances\b",
                    "rule": "a heading with error / error bound(s) / accuracy / exactness / tolerances, or a table "
                            "column error / accuracy / bound"},
    "references": {"heading": r"^(?:references?|bibliography|literature)\b",
                   "rule": "a heading starting with references / bibliography / literature, with text below it"},
}

# Statements outside the collected sections are collected too when they start with one of these labels
# ("Validity: ...", "**Error bound**: ...").
LEAD_PATTERN = (r"^(?:\*\*)?(?:validity(?: and error)?|errors?|error bounds?|accuracy|assumptions?|limitations?|limits"
                r"|approximations?|consequences?)(?:\*\*)?\s*:")

# The domain whose model doc owns the main model of a validated quantity (docs/validation.md section 9); a
# quantity not listed here takes the cnctwin package named most often in its harness's model chain.
VALIDATION_DOMAINS = {
    "chip_form": "mechanics", "chip_ratio": "mechanics", "shear_angle": "mechanics", "chip_temperature": "mechanics",
    "cutting_force": "cutting", "torque": "cutting", "drilling_thrust": "cutting", "drilling_torque": "cutting",
    "tool_temperature": "thermal", "spindle_growth": "thermal", "taylor_life": "wear", "coolant_reach": "coolant",
    "cycle_time": "motion", "roughness": "surface", "stability_lobe": "dynamics", "wall_error": "structure",
}

CONTROLLER_DOC = "docs/dialects.md"           # the controller's model doc (no docs/models/controller.md)
VALIDATION_FORMAT = "cnctwin-validation-results"
DASH = "not stated"

# =============================================================================== Markdown reading

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)\s*([\w+-]*)")
_NUMBER = re.compile(r"^(\d+[a-z]?(?:\.\d+[a-z]?)*)\.?\s+(.*)$")
_BULLET = re.compile(r"^(\s*)(?:[*+-]|\d+[.)])\s+(.*)$")
_SEPARATOR_CELL = re.compile(r"^\s*:?-{3,}:?\s*$")


@dataclass
class Line:
    number: int                  # 1-based line of the file
    text: str
    code: bool = False           # inside a fenced code block (or the fence itself)
    fence: str = ""              # info string of the enclosing fence ("python", "" ...)


@dataclass
class Section:
    level: int                   # 1-6; 0 for the text before the first heading
    title: str                   # heading text as written
    line: int
    body: list[Line] = field(default_factory=list)
    parent: int | None = None    # index of the enclosing section

    @property
    def number(self) -> str:
        match = _NUMBER.match(self.title)
        return match.group(1) if match else ""

    @property
    def name(self) -> str:
        match = _NUMBER.match(self.title)
        return match.group(2) if match else self.title

    def label(self) -> str:
        return f"§ {self.title}" if self.level else "(introduction)"


def path_label(sections: list[Section], index: int) -> str:
    """``§ parent > section``: the headings of level 2 and deeper from the outermost to ``index`` (the
    document's title left out), so that equal headings (one per dialect) stay distinguishable."""
    titles = []
    j: int | None = index
    while j is not None and sections[j].level >= 2:
        titles.append(sections[j].title)
        j = sections[j].parent
    return "§ " + " > ".join(reversed(titles)) if titles else sections[index].label()


def parse_markdown(text: str) -> list[Section]:
    """The sections of a Markdown text: headings outside fenced code, each with its own body (until the next
    heading of any level) and the index of its enclosing section."""
    sections = [Section(0, "", 0)]
    stack: list[int] = [0]
    in_code, fence = False, ""
    for number, raw in enumerate(text.splitlines(), start=1):
        fence_match = _FENCE.match(raw)
        if fence_match:
            if not in_code:
                fence = fence_match.group(2).lower()
            in_code = not in_code
            sections[-1].body.append(Line(number, raw, True, fence))
            continue
        heading = None if in_code else _HEADING.match(raw)
        if heading:
            level = len(heading.group(1))
            while stack and sections[stack[-1]].level >= level:
                stack.pop()
            sections.append(Section(level, heading.group(2).strip(), number, parent=stack[-1] if stack else 0))
            stack.append(len(sections) - 1)
            continue
        sections[-1].body.append(Line(number, raw, in_code, fence if in_code else ""))
    return sections


def heading_matches(section: Section, pattern: str) -> bool:
    """Whether the heading (section number and code spans removed) matches ``pattern`` (case-insensitive)."""
    name = re.sub(r"`[^`]*`", " ", section.name).strip()
    return section.level > 0 and re.search(pattern, name, re.IGNORECASE) is not None


def subtree(sections: list[Section], index: int) -> list[int]:
    """``index`` and every section below it, in file order."""
    out = [index]
    level = sections[index].level
    for j in range(index + 1, len(sections)):
        if sections[j].level <= level:
            break
        out.append(j)
    return out


# =============================================================================== statements

@dataclass
class Statement:
    kind: str                    # bullet | row | paragraph
    text: str                    # the whole statement, whitespace collapsed
    line: int
    cells: dict[str, str] = field(default_factory=dict)    # table rows: header -> cell


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_statements(lines: list[Line]) -> list[Statement]:
    rows = [(line.number, _cells(line.text)) for line in lines]
    rows = [(n, cells) for n, cells in rows if not all(_SEPARATOR_CELL.match(c) or not c for c in cells)]
    if len(rows) < 2:
        return []
    header = [h if h else f"column {i + 1}" for i, h in enumerate(rows[0][1])]
    out = []
    for number, cells in rows[1:]:
        named = {header[i] if i < len(header) else f"column {i + 1}": _collapse(c) for i, c in enumerate(cells)}
        text = "; ".join(f"{k}: {v}" for k, v in named.items() if v)
        out.append(Statement("row", text, number, named))
    return out


def section_statements(section: Section) -> list[Statement]:
    """Bullets (with continuation lines and nested bullets), table rows and paragraphs of a section's own body;
    fenced and indented code is skipped."""
    out: list[Statement] = []
    kind, parts, start, table = "", [], 0, []

    def flush() -> None:
        nonlocal kind, parts, start, table
        if table:
            out.extend(_table_statements(table))
        elif parts:
            out.append(Statement(kind, _collapse(" ".join(parts)), start))
        kind, parts, start, table = "", [], 0, []

    for line in section.body:
        text = line.text
        if line.code:
            flush()
            continue
        if not text.strip():
            flush()
            continue
        if text.lstrip().startswith("|"):
            if not table:
                flush()
            table.append(line)
            continue
        if table:
            flush()
        bullet = _BULLET.match(text)
        if bullet and len(bullet.group(1)) < 2:
            flush()
            kind, parts, start = "bullet", [bullet.group(2)], line.number
        elif bullet and kind == "bullet":
            parts[-1] = parts[-1].rstrip(" .;") + "; " + bullet.group(2)
        elif kind:
            parts.append(text.strip())
        elif text.startswith("    ") or text.startswith("\t"):
            continue                                            # indented code block
        else:
            kind, parts, start = "paragraph", [text.strip()], line.number
    flush()
    return [s for s in out if len(s.text.split()) >= MIN_WORDS]


# =============================================================================== columns of a row

_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "approx.", "Fig.", "fig.", "Eq.", "eq.", "p.", "pp.", "No.", "vs.",
                  "cf.", "ca.", "resp.", "incl.", "max.", "min.", "Sec.", "sec.", "ch.", "Ch.", "Tab.", "ref.")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(`*\"'])")
_CONSEQUENCE = re.compile(r"\bconsequences?\s*:", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    protected = text
    for i, abbreviation in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbreviation, abbreviation.replace(".", f"\x00{i}\x01"))
    parts = _SENTENCE_END.split(protected)
    restore = re.compile("\x00(\\d+)\x01")
    return [restore.sub(".", p).strip() for p in parts if p.strip()]


def clauses(text: str) -> list[str]:
    out = []
    for sentence in sentences(text):
        out.extend(c.strip().rstrip(".") for c in re.split(r";\s+", sentence) if c.strip().rstrip("."))
    return out


def clip(text: str, limit: int) -> str:
    """At most ``limit`` characters, cut at a word boundary, with code spans and bold markers kept balanced."""
    text = _collapse(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    if cut.count("`") % 2:
        cut += "`"
    if cut.count("**") % 2:
        cut += "**"
    return cut + " ..."


def states_error(clause: str) -> bool:
    if ERROR_CUES["strong"].search(clause):
        return True
    return bool(ERROR_CUES["weak"].search(clause) and ERROR_CUES["words"].search(clause))


def error_bound(text: str) -> str:
    return "; ".join(c for c in clauses(text) if states_error(c))


@dataclass
class Vocabulary:
    """Words that identify evidence: bibliography keys, validation case ids and quantities."""
    bibliography: frozenset = frozenset()
    cases: frozenset = frozenset()
    quantities: frozenset = frozenset()


_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.-]*[a-z0-9]")


def evidence(text: str, vocabulary: Vocabulary) -> str:
    tokens = [t for t in _TOKEN.findall(text.lower())]
    keys = sorted({t for t in tokens if t in vocabulary.bibliography})
    cases = sorted({t for t in tokens if t in vocabulary.cases})
    quantities = sorted({t for t in tokens if t in vocabulary.quantities})
    tests = sorted(set(re.findall(r"\btest_\w+(?:\.py)?", text)))
    labels = []
    for name, pattern, case_sensitive in EVIDENCE_CLASSES:
        hit = (re.search(pattern, text, re.IGNORECASE) is not None
               or (case_sensitive is not None and re.search(case_sensitive, text) is not None))
        refs: list[str] = []
        if name == "validation case":
            hit = hit or bool(cases) or bool(quantities)
            refs = cases + quantities
        elif name == "convergence/test":
            refs = tests
        elif name == "literature":
            hit = hit or bool(keys)
            refs = keys
        if hit:
            labels.append(name + (f" ({', '.join(refs[:EVIDENCE_REFS])})" if refs else ""))
    return "; ".join(labels) if labels else DASH


def _row_roles(cells: dict[str, str]) -> dict[str, list[str]]:
    """The cells of a table row by role (COLUMN_ROLES); the first column is the approximation when its header
    names no role; a second approximation column and unmapped columns keep their header."""
    roles: dict[str, list[str]] = {"approximation": [], "validity": [], "consequence": [], "error_bound": [],
                                   "evidence": [], "other": []}
    for position, (header, value) in enumerate(cells.items()):
        if not value:
            continue
        role = next((r for r, pattern in COLUMN_ROLES if re.search(pattern, header, re.IGNORECASE)), None)
        if role is None:
            role = "approximation" if position == 0 else "other"
        if role == "other" or (role == "approximation" and roles["approximation"]):
            roles[role].append(f"{header}: {value}")
        else:
            roles[role].append(value)
    return roles


@dataclass
class Row:
    domain: str
    approximation: str
    consequence: str
    error_bound: str
    evidence: str
    source: str
    kind: str = "model"          # model | scope | validation | performance

    def cells(self) -> list[str]:
        return [self.domain, clip(self.approximation, CELL_CHARS["approximation"]) or DASH,
                clip(self.consequence, CELL_CHARS["consequence"]) or DASH,
                clip(self.error_bound, CELL_CHARS["error_bound"]) or DASH,
                clip(self.evidence, CELL_CHARS["evidence"]) or DASH, self.source]


def statement_row(statement: Statement, domain: str, source: str, vocabulary: Vocabulary, kind: str = "model") -> Row:
    """The columns of one statement (rules in the module docstring)."""
    if statement.kind == "row":
        roles = _row_roles(statement.cells)
        approximation = "; ".join(roles["approximation"] + [f"valid: {v}" for v in roles["validity"]] + roles["other"])
        error = "; ".join(roles["error_bound"]) or error_bound(statement.text)
        consequence = "; ".join(roles["consequence"])
        found = evidence(" ".join(roles["evidence"]) + " " + statement.text, vocabulary)
        return Row(domain, approximation, consequence, error, found, source, kind)
    text = statement.text
    marker = _CONSEQUENCE.search(text)
    if marker:
        approximation, consequence = text[:marker.start()].rstrip(" .;:"), text[marker.end():].strip()
    else:
        parts = sentences(text)
        approximation = parts[0] if parts else text
        consequence = " ".join(parts[1:])
    return Row(domain, approximation, consequence, error_bound(text), evidence(text, vocabulary), source, kind)


# =============================================================================== sources of rows

def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_vocabulary(root: Path) -> Vocabulary:
    """Bibliography keys (db/bibliography/*.yaml ``entries``), validation case ids and quantities (folder names
    of validation/cases)."""
    import yaml
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    keys: set[str] = set()
    for path in sorted((root / "db" / "bibliography").glob("*.yaml")):
        try:
            document = yaml.load(path.read_text(encoding="utf-8"), Loader=loader)
        except (OSError, yaml.YAMLError):
            continue
        entries = document.get("entries") if isinstance(document, dict) else None
        if isinstance(entries, dict):
            keys.update(str(k).lower() for k in entries)
    cases = {p.stem.lower() for p in (root / "validation" / "cases").glob("*/*.yaml")}
    quantities = {p.name for p in (root / "validation" / "cases").glob("*") if p.is_dir()}
    return Vocabulary(frozenset(keys), frozenset(cases), frozenset(quantities))


def model_docs(root: Path) -> list[tuple[str, Path]]:
    """(domain, path) of every model doc: docs/models/*.md by name, then the controller's docs/dialects.md."""
    docs = [(p.stem, p) for p in sorted((root / "docs" / "models").glob("*.md"))]
    controller = root / CONTROLLER_DOC
    if controller.is_file() and "controller" not in {d for d, _ in docs}:
        docs.append(("controller", controller))
    return sorted(docs)


def collected_sections(sections: list[Section]) -> list[int]:
    """Indices of the sections whose heading matches COLLECT_PATTERNS and of every section below one."""
    pattern = "|".join(f"(?:{p})" for p in COLLECT_PATTERNS.values())
    picked: list[int] = []
    for index, section in enumerate(sections):
        if section.level >= 2 and heading_matches(section, pattern) and index not in picked:
            picked.extend(j for j in subtree(sections, index) if j not in picked)
    return sorted(picked)


def doc_rows(domain: str, path: Path, root: Path, vocabulary: Vocabulary) -> tuple[list[Row], list[str], list[str]]:
    """The rows of one model doc, the headings of the sections collected (empty: the doc's structure matched
    no pattern) and the sections of the lead statements ("Validity: ...") collected outside them, in file order."""
    sections = parse_markdown(path.read_text(encoding="utf-8"))
    rows, headings, leads = [], [], []
    name = rel(path, root)
    collected = set(collected_sections(sections))
    lead = re.compile(LEAD_PATTERN, re.IGNORECASE)
    for index, section in enumerate(sections):
        label = path_label(sections, index)
        statements = section_statements(section)
        if index not in collected:
            statements = [s for s in statements if s.kind != "row" and lead.match(s.text)]
            if statements and label.removeprefix("§ ") not in leads:
                leads.append(label.removeprefix("§ "))
        elif label.removeprefix("§ ") not in headings:
            headings.append(label.removeprefix("§ "))
        for statement in statements:
            rows.append(statement_row(statement, domain, f"{name} {label}", vocabulary))
    return rows, headings, leads


def scope_rows(root: Path, vocabulary: Vocabulary) -> tuple[list[Row], list[str]]:
    """The bullets after ``**Scope decisions**`` in docs/architecture.md."""
    path = root / "docs" / "architecture.md"
    if not path.is_file():
        return [], [f"{rel(path, root)} is missing"]
    sections = parse_markdown(path.read_text(encoding="utf-8"))
    for section in sections:
        body = section.body
        start = next((i for i, line in enumerate(body) if "**Scope decisions**" in line.text), None)
        if start is None:
            continue
        lines = []
        for line in body[start + 1:]:
            if not line.text.strip() and lines:
                break
            lines.append(line)
        scope = Section(section.level, section.title, section.line, lines)
        source = f"{rel(path, root)} {section.label()} (scope decisions)"
        return [statement_row(s, "scope", source, vocabulary, "scope") for s in section_statements(scope)], []
    return [], [f"{rel(path, root)}: no '**Scope decisions**' paragraph"]


def _chain_entries(document: dict) -> list[str]:
    out: list[str] = []
    for case in document.get("cases") or []:
        for entry in case.get("model_chain") or []:
            if str(entry) not in out:
                out.append(str(entry))
    return out


def _validation_domain(quantity: str, entries: list[str], domains: set[str]) -> tuple[str, list[str]]:
    """(domain, the other domains its model chain uses): VALIDATION_DOMAINS, else the package named most often
    (ties: the first named)."""
    counts: dict[str, int] = {}
    for entry in entries:
        for package in re.findall(r"\bcnctwin\.([a-z_]+)", entry):
            if package in domains:
                counts[package] = counts.get(package, 0) + 1
    ranked = sorted(counts, key=lambda p: -counts[p])
    domain = VALIDATION_DOMAINS.get(quantity) or (ranked[0] if ranked else "validation")
    return domain, [p for p in counts if p != domain]


def _pct(value) -> str:
    return f"{100.0 * float(value):.1f} %" if isinstance(value, (int, float)) else "-"


def _budget_text(budget: dict) -> str:
    if isinstance(budget.get("pct"), (int, float)):
        return f"{budget['pct']:g} %"
    components = budget.get("components") or {}
    return ", ".join(f"{k} {v:g} %" for k, v in sorted(components.items()) if isinstance(v, (int, float))) or "none"


def validation_rows(root: Path, domains: set[str]) -> tuple[list[Row], list[str]]:
    """One row per quantity of validation/results/*.json (independent cases only)."""
    rows, notes = [], []
    folder = root / "validation" / "results"
    for path in sorted(folder.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            notes.append(f"{rel(path, root)}: not readable ({error})")
            continue
        if not isinstance(document, dict) or document.get("format") != VALIDATION_FORMAT:
            notes.append(f"{rel(path, root)}: not a {VALIDATION_FORMAT} file")
            continue
        quantity = str(document.get("quantity", path.stem))
        independent = document.get("independent") or {}
        dependent = document.get("calibration_dependent") or {}
        entries = _chain_entries(document)
        domain, others = _validation_domain(quantity, entries, domains)
        chain = next((e for e in entries if f"cnctwin.{domain}." in e and "(databases)" not in e),
                     entries[0] if entries else "")
        if independent.get("class_values"):
            error = f"{independent.get('class_matches', 0)} of {independent['class_values']} classes match"
            if isinstance(independent.get("median_abs_rel"), (int, float)):
                error += f"; median {_pct(independent['median_abs_rel'])}"
        else:
            error = f"median {_pct(independent.get('median_abs_rel'))}, p90 {_pct(independent.get('p90_abs_rel'))}"
        error += f" over {independent.get('compared', 0)} values; budget {_budget_text(document.get('budget') or {})}"
        found = (f"validation case: {independent.get('cases', 0)} independent cases, "
                 f"{independent.get('passing', 0)} within budget ({dependent.get('cases', 0)} calibration-dependent "
                 f"not counted)" + (f"; also uses {', '.join(others)}" if others else ""))
        consequence = f"verdict {document.get('verdict', '-')}: {document.get('verdict_reason', '')}"
        rows.append(Row(domain, f"{quantity} as simulated by {chain}" if chain else quantity, consequence, error,
                        found, f"{rel(path, root)} (docs/validation-report.md § {quantity})", "validation"))
    if not rows and not notes:
        notes.append(f"{rel(folder, root)} has no results: run `twin validate`")
    return rows, notes


def _section_named(sections: list[Section], pattern: str) -> int | None:
    return next((i for i, s in enumerate(sections) if heading_matches(s, pattern)), None)


def performance_rows(root: Path, vocabulary: Vocabulary) -> tuple[list[Row], list[str]]:
    """The performance summary of docs/performance.md."""
    path = root / "docs" / "performance.md"
    if not path.is_file():
        return [], [f"{rel(path, root)} is missing"]
    text = path.read_text(encoding="utf-8")
    sections = parse_markdown(text)
    name, rows, notes = rel(path, root), [], []
    introduction = " ".join(line.text for s in sections[:2] for line in s.body if not line.code)
    sentence = next((s for s in sentences(_collapse(introduction)) if re.search(r"\+-\s*\d+\s*%", s)), "")
    scatter = next((re.sub(r"^which\s+", "", part) for part in re.split(r"[,:;]\s+", sentence)
                    if re.search(r"\+-\s*\d+\s*%", part)), "")
    scatter = f"timing scatter on the shared box: {scatter}" if scatter else ""
    if not scatter:
        notes.append(f"{name}: no timing scatter (+-N %) stated before its first section")
    status = re.search(r"\*\*Status\.\*\*(.*?)(?:\n\s*\n|\Z)", text, re.DOTALL)
    if status:
        body = _collapse(status.group(1))
        met = re.search(r"\bMet:\s*(.*?)(?=\bMissed:|\Z)", body)
        missed = re.search(r"\bMissed:\s*(.*)\Z", body)
        for label, match in (("budgets met", met), ("budgets missed", missed)):
            if match:
                rows.append(Row("performance", f"Performance {label} (architecture 20)", match.group(1).strip(" ;."),
                                scatter, "measurement: process CPU time under bin/tsafe", f"{name} (Status)",
                                "performance"))
    else:
        notes.append(f"{name}: no '**Status.**' paragraph")
    for pattern, label in ((r"\bwhole runs\b", "CPU per simulated second"), (r"\bmemory\b", "peak memory")):
        index = _section_named(sections, pattern)
        if index is None:
            notes.append(f"{name}: no section named like '{pattern}'")
            continue
        for statement in section_statements(sections[index]):
            if statement.kind != "row":
                continue
            values = list(statement.cells.values())
            consequence = "; ".join(f"{k}: {v}" for k, v in list(statement.cells.items())[1:] if v)
            rows.append(Row("performance", f"{label}: {values[0]}", consequence, scatter,
                            "measurement: process CPU time / peak RSS under bin/tsafe",
                            f"{name} {sections[index].label()}", "performance"))
    index = _section_named(sections, r"\bopen problems\b")
    if index is not None:
        for statement in section_statements(sections[index]):
            rows.append(statement_row(statement, "performance", f"{name} {sections[index].label()}", vocabulary,
                                      "performance"))
    return rows, notes


# =============================================================================== docs/approximations.md

@dataclass
class Approximations:
    text: str
    rows: list[Row]
    matched: dict[str, list[str]]        # doc -> headings collected
    unmatched: list[str]                 # docs with no collected section
    notes: list[str]


HEADER = ["domain", "approximation", "consequence", "error bound", "evidence", "source"]


def _md_cell(text: str) -> str:
    return _collapse(text).replace("|", "¦")


def table(rows: list[Row]) -> list[str]:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(_md_cell(c) for c in row.cells()) + " |" for row in rows]
    return lines


def _section_title(domain: str, docs: dict[str, str]) -> str:
    if domain == "scope":
        return "Scope (docs/architecture.md)"
    if domain == "performance":
        return "Loop cost and performance (docs/performance.md)"
    if domain == "validation":
        return "Validation of quantities with no domain doc"
    title = domain.replace("hotloop", "hot loop")
    return f"{title[:1].upper()}{title[1:]} ({docs[domain]})" if domain in docs else title


def build_approximations(root: Path = ROOT) -> Approximations:
    root = Path(root)
    vocabulary = read_vocabulary(root)
    docs = model_docs(root)
    doc_names = {domain: rel(path, root) for domain, path in docs}
    rows: list[Row] = []
    matched: dict[str, list[str]] = {}
    leads: dict[str, list[str]] = {}
    unmatched: list[str] = []
    scope, notes = scope_rows(root, vocabulary)
    rows += scope
    for domain, path in docs:
        found, headings, lead_sections = doc_rows(domain, path, root, vocabulary)
        rows += found
        if headings:
            matched[doc_names[domain]] = headings
        else:
            unmatched.append(doc_names[domain])
        if lead_sections:
            leads[doc_names[domain]] = lead_sections
    validation, validation_notes = validation_rows(root, set(doc_names))
    performance, performance_notes = performance_rows(root, vocabulary)
    rows += validation + performance
    notes += validation_notes + performance_notes
    order = ["scope"] + [d for d, _ in docs] + ["validation", "performance"]
    lines = _approximations_intro(rows, validation)
    for domain in order:
        group = [r for r in rows if r.domain == domain]
        if not group:
            continue
        lines += ["", f"## {_section_title(domain, doc_names)}", ""] + table(group)
    lines += _approximations_method(matched, leads, unmatched, notes)
    return Approximations("\n".join(lines) + "\n", rows, matched, unmatched, notes)


def _approximations_intro(rows: list[Row], validation: list[Row]) -> list[str]:
    kinds = {k: sum(1 for r in rows if r.kind == k) for k in ("model", "scope", "validation", "performance")}
    with_bound = sum(1 for r in rows if r.kind == "model" and r.error_bound)
    return [
        "# Approximations",
        "",
        "Every approximation the simulator makes, what it does to a result, how large the error can be and what "
        "that bound rests on, collected from the model documentation, the validation results and the performance "
        "measurements. Generated by `bin/tsafe python -m cnctwin.docs_tools approximations` "
        "(cnctwin/docs_tools.py); do not edit it by hand: change the source documents and run the command again. "
        "The method and the documents "
        "whose structure did not match are at the end.",
        "",
        f"Rows: {kinds['model']} statements of the model docs ({with_bound} with an error expression), "
        f"{kinds['scope']} scope decisions, {kinds['validation']} validated quantities, {kinds['performance']} "
        "performance items. The table is one table split into one section per domain (the same columns "
        "throughout), so that the run report can show the sections of the domains a run used.",
        "",
        "How to read it: *approximation* is what the model does instead of the full physics; *consequence* what "
        "that does to a result; *error bound* the error expressions the source states; *evidence* what the bound "
        "rests on (validation case, convergence/test, literature, estimate); *source* the document and section "
        "to read for the whole statement. Cells are clipped; \"not stated\" means the source statement does not "
        "say it. Validation rows give the median and 90th percentile of the absolute relative error over the "
        "independent cases of validation/results/*.json, as written by the last `twin validate`.",
    ]


def _approximations_method(matched: dict[str, list[str]], leads: dict[str, list[str]], unmatched: list[str],
                           notes: list[str]) -> list[str]:
    lines = ["", "## Method (how this file is generated)", "",
             "Sections collected from a model doc: every heading (section number stripped) that matches one of "
             "these patterns (case-insensitive), and every section below it:", ""]
    lines += [f"* {name}: `{pattern.replace('|', ' or ')}`" for name, pattern in COLLECT_PATTERNS.items()]
    lines += ["",
              "Each collected section is split into statements: a bullet with its continuation lines (nested bullets "
              "joined to it), a table row, or a paragraph; code blocks are skipped. Columns: for a table row by its "
              "header (consequence, validity, error, basis/how established/source, else the first column is the "
              "approximation); for a bullet or paragraph the first sentence is the approximation, the text after "
              "'Consequence:' (else the following sentences) the consequence. The error bound is the table's error "
              "column, else the clauses with an error expression (a percentage, +-, <=, a factor, 1e-9, within, "
              "up to, exact, order of magnitude, ulp). Evidence classes: validation case (the validation suite, its "
              "quantities or case ids), convergence/test (convergence, refinement, a test against an exact, "
              "analytic, closed-form or reference solution), literature (a db/bibliography key, an author-year "
              "citation, a standard, catalogue, handbook, manual), estimate (est, estimated, assumed, typical, "
              "judgement)."]
    lines += ["", "Statements outside those sections are collected when they start with a label (pattern "
              f"`{LEAD_PATTERN.replace('|', ' or ')}`, case-insensitive), such as 'Validity:' or 'Error:'.",
              "", "Documents and the headings collected from them:", ""]
    lines += [f"* {doc}: " + "; ".join(headings) for doc, headings in sorted(matched.items())]
    lines += ["", "Sections with labelled statements ('Validity:', 'Error:' ...) collected from outside them:", ""]
    lines += [f"* {doc}: " + "; ".join(sections) for doc, sections in sorted(leads.items())] or ["* none"]
    lines += ["", "## Documents whose structure did not match", ""]
    if unmatched:
        lines += ["No heading of these matched the patterns above, so they contribute no row: give their "
                  "assumptions, validity and error bounds sections such headings.", ""]
        lines += [f"* {doc}" for doc in unmatched]
    else:
        lines += ["None: every model doc has at least one matching section."]
    if notes:
        lines += ["", "Other sources that did not match:", ""] + [f"* {note}" for note in notes]
    return lines


# =============================================================================== docs/doc-gaps.md

_EQUATION_OPERATOR = re.compile(r"[+*/^]|\s-\s|\(|\b(?:sqrt|exp|ln|log|sin|cos|tan|atan|min|max|sum|pi|integral)\b")
_EQUALS = re.compile(r"(?<![<>=!:])=(?![=>])")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_TAGGED_CODE = ("python", "py", "yaml", "yml", "json", "bash", "sh", "shell", "js", "javascript", "console", "toml")


def _is_equation(text: str) -> bool:
    match = _EQUALS.search(text)
    if not match:
        return False
    left, right = text[:match.start()], text[match.end():]
    if not left.strip() or not right.strip():
        return False
    return bool(_EQUATION_OPERATOR.search(right) or _EQUATION_OPERATOR.search(left[-40:]))


def equation_lines(sections: list[Section]) -> list[int]:
    found = []
    for section in sections:
        for line in section.body:
            if line.code:
                if line.fence in _TAGGED_CODE or _FENCE.match(line.text):
                    continue
                if _is_equation(line.text):
                    found.append(line.number)
                continue
            spans = _CODE_SPAN.findall(line.text)
            plain = _CODE_SPAN.sub(" ", line.text)
            if any(_is_equation(s) for s in spans) or any(_is_equation(c) for c in re.split(r"[;,]\s+|\.\s+", plain)):
                found.append(line.number)
    return found


def _source_evidence(section: Section, vocabulary: Vocabulary) -> str:
    text = "\n".join(line.text for line in section.body)
    for _, header in _table_headers([section]):
        if any(re.search(r"\bsources?\b|\bsrc\b|\bbasis\b", h) for h in header):
            return "a table with a source column"
    if re.search(r"\bsrc\b|\best\b|db/models/[\w.-]+\.yaml|db/bibliography", text):
        return "src / est / db/models files named"
    tokens = set(_TOKEN.findall(text.lower()))
    if tokens & vocabulary.bibliography:
        return "bibliography keys named"
    return ""


def _table_headers(sections: list[Section]) -> list[tuple[int, list[str]]]:
    out = []
    for index, section in enumerate(sections):
        previous_table = False
        for line in section.body:
            is_table = not line.code and line.text.lstrip().startswith("|")
            if is_table and not previous_table:
                out.append((index, [c.lower() for c in _cells(line.text)]))
            previous_table = is_table
    return out


@dataclass
class DocGaps:
    doc: str
    found: dict[str, str]                # item -> where it was found by the structure rules ("" = not)
    text_only: dict[str, str] = field(default_factory=dict)    # item -> where its text was found without a heading

    def state(self, item: str) -> str:
        if self.found.get(item):
            return "yes"
        return "text only" if self.text_only.get(item) else "missing"

    @property
    def missing(self) -> list[str]:
        return [item for item in REQUIRED_ITEMS if self.state(item) == "missing"]

    @property
    def unstructured(self) -> list[str]:
        return [item for item in REQUIRED_ITEMS if self.state(item) == "text only"]


# Body text that shows an item is there although no heading the rules recognise holds it ("text only").
TEXT_ONLY = {
    "parameters and sources": r"db/models/\S+|\bsrc:\s*est\b",
    "assumptions": r"^\s*(?:[*-]\s+)?(?:\*\*)?assumptions?\b",
    "validity range": r"\bvalidity\b|\bvalid (?:for|within|when|while|up to|in|above|below|only|range|between)\b",
    "error bound": r"\berror bounds?\b|\berrors? (?:of|below|within|is|are|<|<=|~|about|up to)\b|\baccurate to\b"
                   r"|\bwithin \d",
    "references": r"^\s*(?:\*\*)?references?(?:\*\*)?\s*:\s*\S",
}


def _text_only(sections: list[Section], item: str) -> str:
    pattern = re.compile(TEXT_ONLY[item], re.IGNORECASE | re.MULTILINE)
    for index, section in enumerate(sections):
        for line in section.body:
            if not line.code and pattern.search(line.text):
                return f"{path_label(sections, index)} (line {line.number})"
    return ""


def check_doc(path: Path, root: Path, vocabulary: Vocabulary) -> DocGaps:
    sections = parse_markdown(path.read_text(encoding="utf-8"))
    headers = _table_headers(sections)
    found: dict[str, str] = {}
    equations = equation_lines(sections)
    found["equations"] = (f"{len(equations)} equation lines (first at line {equations[0]})"
                          if len(equations) >= MIN_EQUATIONS else "")
    found["parameters and sources"] = _parameters(sections, headers, vocabulary)
    found["assumptions"] = _by_heading(sections, REQUIRED_ITEMS["assumptions"]["heading"])
    found["validity range"] = (_by_heading(sections, REQUIRED_ITEMS["validity range"]["heading"])
                               or _by_column(sections, headers, r"\bvalidity\b|\bvalid\b|\brange\b"))
    found["error bound"] = (_by_heading(sections, REQUIRED_ITEMS["error bound"]["heading"])
                            or _by_column(sections, headers, r"\berror\b|\baccuracy\b|\bbound\b"))
    found["references"] = _references(sections)
    text_only = {item: _text_only(sections, item) for item in TEXT_ONLY if not found.get(item)}
    return DocGaps(rel(path, root), found, {k: v for k, v in text_only.items() if v})


def _item_heading(section: Section, pattern: str) -> bool:
    return section.level >= 2 and heading_matches(section, pattern)


def _by_heading(sections: list[Section], pattern: str) -> str:
    return "; ".join(path_label(sections, i) for i, s in enumerate(sections) if _item_heading(s, pattern))


def _by_column(sections: list[Section], headers: list[tuple[int, list[str]]], pattern: str) -> str:
    for index, header in headers:
        column = next((h for h in header if re.search(pattern, h)), None)
        if column:
            return f"table column '{column}' in {path_label(sections, index)}"
    return ""


def _parameters(sections: list[Section], headers: list[tuple[int, list[str]]], vocabulary: Vocabulary) -> str:
    pattern = REQUIRED_ITEMS["parameters and sources"]["heading"]
    for index, section in enumerate(sections):
        if not _item_heading(section, pattern):
            continue
        for j in subtree(sections, index):
            how = _source_evidence(sections[j], vocabulary)
            if how:
                return f"{path_label(sections, index)} ({how})"
    for index, header in headers:
        if any(re.search(r"\bsources?\b|\bsrc\b", h) for h in header):
            return f"table with a source column in {path_label(sections, index)}"
    return ""


def _references(sections: list[Section]) -> str:
    pattern = REQUIRED_ITEMS["references"]["heading"]
    for index, section in enumerate(sections):
        if _item_heading(section, pattern):
            if any(line.text.strip() for j in subtree(sections, index) for line in sections[j].body):
                return path_label(sections, index)
    return ""


@dataclass
class GapReport:
    text: str
    docs: list[DocGaps]


def check_docs(root: Path = ROOT) -> GapReport:
    root = Path(root)
    vocabulary = read_vocabulary(root)
    docs = [check_doc(path, root, vocabulary) for _, path in model_docs(root)]
    approximations = build_approximations(root)
    return GapReport(_gaps_text(docs, approximations.unmatched), docs)


def _where(doc: DocGaps, item: str) -> str:
    state = doc.state(item)
    if state == "yes":
        return _md_cell(doc.found[item])
    if state == "text only":
        return f"text only, no heading the rules recognise: {_md_cell(doc.text_only[item])}"
    return "**missing**"


def _gaps_text(docs: list[DocGaps], unmatched: list[str]) -> str:
    items = list(REQUIRED_ITEMS)
    complete = [d for d in docs if not d.missing and not d.unstructured]
    missing = {item: sum(1 for d in docs if d.state(item) == "missing") for item in items}
    loose = {item: sum(1 for d in docs if d.state(item) == "text only") for item in items}
    lines = ["# Model documentation gaps", "",
             "Generated by `bin/tsafe python -m cnctwin.docs_tools check` (cnctwin/docs_tools.py) from "
             "docs/models/*.md and, for the controller, docs/dialects.md. A model doc must give equations, "
             "parameters and their sources, "
             "assumptions, the validity range, the error bound and references (docs/architecture.md section 14). "
             "The items are found by headings and simple text rules (listed at the end). **yes**: a section the "
             "rules recognise holds it; **text only**: the doc says it somewhere, but under no heading the rules "
             "recognise (rename or add the heading); **missing**: neither. Do not edit by hand.", "",
             "## Summary", "",
             f"{len(docs)} model docs; {len(complete)} have every item under a recognised heading; "
             f"{sum(1 for d in docs if d.missing)} lack at least one item; "
             f"{sum(1 for d in docs if d.unstructured)} have an item as text only.", ""]
    lines += [f"* {item}: missing in {missing[item]}, text only in {loose[item]}" for item in items]
    lines += ["", "| doc | " + " | ".join(items) + " | lacks |", "|" + "---|" * (len(items) + 2)]
    for doc in docs:
        marks = ["**missing**" if doc.state(i) == "missing" else doc.state(i) for i in items]
        lines.append(f"| {doc.doc} | " + " | ".join(marks) + f" | {', '.join(doc.missing) or '-'} |")
    lines += ["", "## Where each item was found", ""]
    for doc in docs:
        lines += [f"### {doc.doc}", ""]
        lines += [f"* {item}: {_where(doc, item)}" for item in items]
        lines.append("")
    lines += ["## Approximation sections (docs/approximations.md)", ""]
    if unmatched:
        lines += ["These docs have no heading that `python -m cnctwin.docs_tools approximations` collects "
                  "(assumptions, limitations, validity, error, approximations, consequences, validation), so "
                  "docs/approximations.md has no row from them:", ""] + [f"* {d}" for d in unmatched]
    else:
        lines.append("Every model doc has at least one section that docs/approximations.md collects.")
    lines += ["", "## Rules", ""]
    lines += [f"* **{item}**: {spec['rule']}." for item, spec in REQUIRED_ITEMS.items()]
    lines += [f"* **text only** for {item}: a line (outside code) matching `{pattern.replace('|', ' or ')}`."
              for item, pattern in TEXT_ONLY.items()]
    lines += ["* Section numbers (`10.`, `2.3`) are stripped from headings before matching; matching is "
              "case-insensitive; text inside fenced code blocks is not a heading."]
    return "\n".join(lines) + "\n"


# =============================================================================== command line

def _write_or_check(path: Path, text: str, check: bool) -> int:
    if check:
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text:
            print(f"{path} is not up to date: run python -m cnctwin.docs_tools", file=sys.stderr)
            return 1
        print(f"{path} is up to date")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.docs_tools",
                                     description="regenerate docs/approximations.md or docs/doc-gaps.md")
    parser.add_argument("--root", default=str(ROOT), help="the source tree (default: cnctwin.paths.ROOT)")
    sub = parser.add_subparsers(dest="command", required=True)
    approximations = sub.add_parser("approximations", help="regenerate docs/approximations.md from the model docs")
    approximations.add_argument("--out", help=f"output file (default <root>/{APPROXIMATIONS_FILE})")
    approximations.add_argument("--check", action="store_true", help="write nothing; exit 1 when the file is stale")
    check = sub.add_parser("check", help="write docs/doc-gaps.md: what each model doc lacks")
    check.add_argument("--out", help=f"output file (default <root>/{GAPS_FILE})")
    check.add_argument("--strict", action="store_true", help="exit 1 when a model doc lacks a required item")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "approximations":
        document = build_approximations(root)
        out = Path(args.out) if args.out else root / APPROXIMATIONS_FILE
        code = _write_or_check(out, document.text, args.check)
        print(f"{len(document.rows)} rows; model docs matched {len(document.matched)}, did not match "
              f"{len(document.unmatched)}" + (f" ({', '.join(document.unmatched)})" if document.unmatched else ""))
        return code
    report = check_docs(root)
    out = Path(args.out) if args.out else root / GAPS_FILE
    _write_or_check(out, report.text, False)
    lacking = [d for d in report.docs if d.missing or d.unstructured]
    for doc in lacking:
        parts = ([f"lacks {', '.join(doc.missing)}"] if doc.missing else []) + \
                ([f"text only: {', '.join(doc.unstructured)}"] if doc.unstructured else [])
        print(f"{doc.doc}: {'; '.join(parts)}")
    print(f"{len(report.docs)} model docs, {sum(1 for d in report.docs if d.missing)} lack an item, "
          f"{sum(1 for d in report.docs if d.unstructured)} have one as text only; written to {out}")
    return 1 if args.strict and any(d.missing for d in report.docs) else 0


if __name__ == "__main__":
    sys.exit(main())
