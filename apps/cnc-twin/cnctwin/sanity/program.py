"""Program view for the sanity checks: the lines as written (words, comments), the commands the
control executes for them, and the tool comments a programmer writes (``(T1 D10 ENDMILL)``).

The lexer reads the word-address format of ISO 6983-1 as the Fanuc and Haas controls write it:
single-letter addresses with a signed decimal value, comments in parentheses, ``;`` as the end
of block (text after it is kept as a comment), ``/`` block delete, ``N`` sequence numbers, ``O``
program numbers and ``%`` tape marks. A line with a macro statement (``#100 = 2``, ``IF``,
``WHILE``, ``GOTO``) or an address whose value is an expression (``X#101``, ``Z[#1 - 2]``) is
marked ``macro``: the static scan does not evaluate it (docs/models/sanity.md, "Scan").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..controller.ir import Command
from ..units import INCH_M

_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")
_KEYWORD = re.compile(r"[A-Za-z]{2,}")


@dataclass(frozen=True)
class Word:
    """One address word: letter and value as written (``value`` None for an expression)."""

    letter: str
    value: float | None
    text: str
    has_point: bool

    @property
    def code(self) -> str:
        """Canonical code text for G and M words: ``G01`` -> ``G1``, ``G54.1`` -> ``G54.1``."""
        return code_text(self.letter, self.value)


def code_text(letter: str, value: float | None) -> str:
    if value is None:
        return f"{letter}?"
    if float(value).is_integer():
        return f"{letter}{int(value)}"
    return f"{letter}{value:g}"


@dataclass(frozen=True)
class Line:
    """One line of a program file (1-based ``number``)."""

    number: int
    text: str
    words: tuple[Word, ...] = ()
    comments: tuple[str, ...] = ()
    n_number: int | None = None
    block_delete: bool = False
    macro: bool = False
    program_number: str | None = None
    percent: bool = False

    def first(self, letter: str) -> Word | None:
        for word in self.words:
            if word.letter == letter:
                return word
        return None

    def all(self, letter: str) -> list[Word]:
        return [w for w in self.words if w.letter == letter]

    def codes(self, letter: str) -> list[str]:
        return [w.code for w in self.words if w.letter == letter]

    def has(self, letter: str) -> bool:
        return any(w.letter == letter for w in self.words)

    @property
    def is_empty(self) -> bool:
        return not self.words and not self.macro


def lex(text: str) -> list[Line]:
    """Split a program into :class:`Line` objects (every physical line, 1-based)."""
    return [lex_line(raw, index + 1) for index, raw in enumerate(text.splitlines())]


def lex_line(raw: str, number: int) -> Line:
    text = raw.rstrip("\r\n")
    stripped = text.strip()
    if stripped == "%" or (stripped.startswith("%") and not stripped[1:].strip()):
        return Line(number, text, percent=True)
    words: list[Word] = []
    comments: list[str] = []
    macro = False
    block_delete = False
    n_number = None
    program_number = None
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char in " \t":
            i += 1
            continue
        if char == "(":
            close = text.find(")", i + 1)
            close = n if close < 0 else close
            comment = text[i + 1:close].strip()
            if comment:
                comments.append(comment)
            i = close + 1
            continue
        if char == ";":
            rest = text[i + 1:].strip()
            if rest:
                comments.append(rest)
            break
        if char == "/" and not words and not block_delete:
            block_delete = True
            i += 1
            while i < n and text[i].isdigit():
                i += 1
            continue
        if char == "%":
            i += 1
            continue
        if char == "#":
            macro = True
            break
        if char.isalpha():
            keyword = _KEYWORD.match(text, i)
            if keyword is not None:
                macro = True
                break
            letter = char.upper()
            i += 1
            while i < n and text[i] in " \t":
                i += 1
            if i < n and text[i] in "#[":
                start = i
                i = _skip_expression(text, i)
                macro = True
                words.append(Word(letter, None, text[start:i].strip(), False))
                continue
            if i < n and text[i] in "+-" and i + 1 < n and text[i + 1] in "#[":
                start = i
                i = _skip_expression(text, i + 1)
                macro = True
                words.append(Word(letter, None, text[start:i].strip(), False))
                continue
            match = _NUMBER.match(text, i)
            if match is None:
                words.append(Word(letter, None, "", False))
                macro = True
                continue
            literal = match.group(0)
            i = match.end()
            value = float(literal)
            if letter == "N" and not words and n_number is None:
                n_number = int(value)
                continue
            if letter == "O" and not words and program_number is None:
                program_number = f"O{int(value)}"
                continue
            words.append(Word(letter, value, literal, "." in literal))
            continue
        if char == ":" and not words and program_number is None:
            match = _NUMBER.match(text, i + 1)
            if match is not None:
                program_number = f"O{int(float(match.group(0)))}"
                i = match.end()
                continue
        macro = True
        break
    return Line(number, text, tuple(words), tuple(comments), n_number, block_delete, macro, program_number)


def _skip_expression(text: str, i: int) -> int:
    """End of a macro expression value starting at ``i`` (``#101``, ``#[#1]``, ``[#1 + 2]``)."""
    n = len(text)
    if text[i] == "#":
        i += 1
        if i < n and text[i] == "[":
            return _skip_brackets(text, i)
        while i < n and text[i].isdigit():
            i += 1
        return i
    return _skip_brackets(text, i)


def _skip_brackets(text: str, i: int) -> int:
    depth = 0
    n = len(text)
    while i < n:
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


# ----------------------------------------------------------------------------- tool comments

#: Tool kinds a comment may name, with the tool types (db/_templates/tool.yaml) each one allows.
COMMENT_KINDS: dict[str, tuple[str, ...]] = {
    "facemill": ("face_mill", "shoulder_mill"),
    "ballmill": ("ball_endmill",),
    "bullmill": ("toroidal_endmill",),
    "flatmill": ("square_endmill", "roughing_endmill", "serrated_endmill"),
    "endmill": ("square_endmill", "roughing_endmill", "serrated_endmill", "toroidal_endmill", "ball_endmill"),
    "spotdrill": ("spot_drill",),
    "centredrill": ("centre_drill",),
    "drill": ("twist_drill", "indexable_drill", "gun_drill"),
    "tap": ("tap_cutting", "tap_forming"),
    "reamer": ("reamer",),
    "chamfer": ("chamfer_mill", "spot_drill"),
    "threadmill": ("thread_mill",),
    "boring": ("boring_head", "boring_bar"),
    "engraver": ("engraver",),
}

# (pattern, kind): checked in order, the first match names the kind.
_KIND_PATTERNS: tuple[tuple[re.Pattern, str], ...] = tuple((re.compile(p), k) for p, k in (
    (r"\bFACE\s*MILL|\bFACEMILL|\bFACING\b|\bSHELL\s*MILL", "facemill"),
    (r"\bBALL\s*(?:NOSE|END)?\s*(?:MILL|EM|E/M)?\b|\bBNEM\b|\bBEM\b", "ballmill"),
    (r"\bBULL\s*(?:NOSE)?\b|\bTORUS\b|\bCORNER\s*RAD", "bullmill"),
    (r"\bFLAT\s*(?:END\s*)?(?:MILL|EM)\b|\bFEM\b", "flatmill"),
    (r"\bTHREAD\s*MILL|\bTHREADMILL", "threadmill"),
    (r"\bSPOT\s*(?:DRILL)?\b|\bSPOTDRILL", "spotdrill"),
    (r"\bCENT(?:ER|RE)\s*DRILL|\bCTR\s*DRILL", "centredrill"),
    (r"\bTAP\s*DRILL", "drill"),
    (r"\bTAP\b|\bTAPPING\b", "tap"),
    (r"\bREAM(?:ER)?\b", "reamer"),
    (r"\bCHAMF(?:ER)?\b|\bCHAMFER\s*MILL", "chamfer"),
    (r"\bBOR(?:E|ING)\b(?:\s*(?:BAR|HEAD))?", "boring"),
    (r"\bENGRAV(?:E|ER|ING)\b", "engraver"),
    (r"\bDRILL\b|\bDRL\b", "drill"),
    (r"\bEND\s*MILL|\bENDMILL|\bEM\b|\bE/M\b|\bSLOT\s*DRILL|\bMILL\b", "endmill"),
))
_T_NUMBER = re.compile(r"(?:\bT\s*0*(\d+)\b)|(?:\bTOOL\s*(?:NO\.?|#)?\s*0*(\d+)\b)")
_DIA_EXPLICIT = re.compile(r"(?:\bD(?:IA(?:M(?:ETER)?)?)?\.?\s*=?\s*|Ø\s*|\bDIA\.?\s*)(\d+(?:\.\d*)?|\.\d+)")
_DIA_TRAILING = re.compile(r"(\d+(?:\.\d*)?|\.\d+)\s*(?:MM)?\s*(?:DIA\b|Ø)")
_DIA_MM = re.compile(r"(\d+(?:\.\d*)?|\.\d+)\s*MM\b")
_FRACTION = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
_DIA_BARE = re.compile(r"(?<![A-Z0-9.])(\d+(?:\.\d*)?|\.\d+)(?![0-9.])")
_THREAD = re.compile(r"\bM\s*(\d+(?:\.\d+)?)(?:\s*[X×]\s*(\d+(?:\.\d+)?))?")


@dataclass(frozen=True)
class ToolComment:
    """What a programmer's comment says about a tool (lengths in SI, in the program's units)."""

    line: int
    text: str
    number: int | None = None
    diameter_m: float | None = None
    kind: str | None = None
    thread_major_m: float | None = None
    thread_pitch_m: float | None = None

    @property
    def describes_tool(self) -> bool:
        return self.kind is not None or self.diameter_m is not None or self.thread_major_m is not None


def parse_tool_comment(text: str, line: int, unit_m: float) -> ToolComment:
    """Read a T number, a diameter, a tool kind and a metric thread from comment text.

    ``unit_m`` is the program's length unit (1e-3 for mm, 0.0254 for inch); fractions (``1/2 EM``)
    are read in inches whatever the program unit (that is how fractional sizes are written).
    """
    upper = text.upper()
    number = None
    match = _T_NUMBER.search(upper)
    if match is not None:
        number = int(match.group(1) or match.group(2))
    kind = None
    for pattern, name in _KIND_PATTERNS:
        if pattern.search(upper):
            kind = name
            break
    thread_major = thread_pitch = None
    if kind in ("tap", "threadmill"):
        thread = _THREAD.search(upper)
        if thread is not None:
            thread_major = float(thread.group(1)) * 1e-3
            if thread.group(2):
                thread_pitch = float(thread.group(2)) * 1e-3
    diameter = _diameter(upper, unit_m, number, thread_major is not None)
    return ToolComment(line, text, number, diameter, kind, thread_major, thread_pitch)


def _diameter(upper: str, unit_m: float, number: int | None, has_thread: bool) -> float | None:
    cleaned = _T_NUMBER.sub(" ", upper)
    if has_thread:
        cleaned = _THREAD.sub(" ", cleaned)
    for pattern in (_DIA_EXPLICIT, _DIA_TRAILING):
        match = pattern.search(cleaned)
        if match is not None:
            return float(match.group(1)) * unit_m
    match = _DIA_MM.search(cleaned)
    if match is not None:
        return float(match.group(1)) * 1e-3
    match = _FRACTION.search(cleaned)
    if match is not None and float(match.group(2)) > 0:
        return float(match.group(1)) / float(match.group(2)) * INCH_M
    if has_thread:
        return None
    bare = [m for m in _DIA_BARE.finditer(cleaned)]
    if len(bare) == 1:
        return float(bare[0].group(1)) * unit_m
    return None


# ----------------------------------------------------------------------------- the view

@dataclass
class Program:
    """One program as the sanity rules see it.

    ``commands`` are :class:`cnctwin.controller.ir.Command` objects in execution order (from the
    controller, or from :mod:`cnctwin.sanity.scan`); ``tip_start_W_m``/``tip_end_W_m`` hold the
    tool-tip position (program coordinates of the active work offset, intended position) at the
    start and end of each command, NaN where it is not known.
    """

    name: str
    dialect: str
    machine_kind: str
    text: str
    lines: list[Line]
    commands: list[Command] = field(default_factory=list)
    tip_start_W_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tip_end_W_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    source: str = "scan"
    scan: object = None                       # cnctwin.sanity.scan.ScanResult (own scan)
    tool_comments: list[ToolComment] = field(default_factory=list)
    texts: dict[str, list[Line]] = field(default_factory=dict)   # every program file read (subprograms too)

    def line(self, number: int, program: str | None = None) -> Line | None:
        lines = self.texts.get(program or self.name, self.lines)
        if 1 <= number <= len(lines):
            return lines[number - 1]
        return None

    def commands_at(self, line: int, program: str | None = None) -> list[int]:
        """Indices of the commands executed for ``line`` of ``program`` (main program by default)."""
        name = program or self.name
        return [i for i, c in enumerate(self.commands) if c.ref.line == line and c.ref.program == name]

    @property
    def comments(self) -> list[tuple[int, str]]:
        return [(line.number, text) for line in self.lines for text in line.comments]
