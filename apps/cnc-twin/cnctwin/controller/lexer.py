"""ISO 6983 word-address programs: files -> programs -> blocks (Fanuc, Haas and their kin).

Rules (per dialect, :class:`LexerRules`):

* One block per line. ``( ... )`` is a comment anywhere in the block (control-out/control-in);
  ``;`` ends the block (it is the end-of-block character the control displays; text after it on
  the same line is kept as a comment). An unclosed ``(`` comments out the rest of the line.
* ``%`` marks the start and the end of the tape: when the file contains ``%`` lines, only the
  text between the first and the second is the program memory (leader and trailer are not).
* ``O1234`` starts a program (several programs per file); text before the first ``O`` is an
  unnumbered program named after the file. The ``O`` block itself is not executed.
* ``/`` (or ``/n``, n = 1..9, where the dialect has several switches) at the start of a block
  skips the block when switch n is on; in the middle of a block it skips from there to the end.
* ``Nnnnnn`` at the start of a block is its sequence number.
* A block that starts with ``#``, ``IF``, ``WHILE``, ``GOTO``, ``DO``, ``END``, ``DPRNT``, ``POPEN``,
  ``PCLOS`` or ``BPRNT`` is a macro statement (custom macro B); otherwise it is a list of words:
  a letter followed by a number, a variable (``#1``, ``-#1``, ``#[..]``) or a bracketed expression.
* Spaces outside comments are not significant. Lower-case letters are read as upper case and
  the block is flagged (the executor reports it).
* Syntax problems do not stop parsing: the block carries ``error = (key, detail)`` and the alarm
  is raised when the control reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .expr import STATEMENT_WORDS, ParseError, parse_statement, parse_word_value
from .program import Block, Program, ProgramSource, ProgramStore, Word, source_digest


@dataclass(frozen=True)
class LexerRules:
    letters: frozenset = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    max_digits: dict = field(default_factory=dict)   # letter -> max significant digits of a literal
    comment_open: str = "("
    comment_close: str = ")"
    eob_comment: str | None = ";"
    block_delete_switches: int = 9       # 9: /1../9 ; 1: only '/'
    mid_block_delete: bool = True
    program_letter: str = "O"
    max_n_digits: int = 5
    max_o_digits: int = 4
    macro: bool = True
    comma_letters: frozenset = frozenset()   # FANUC ,C ,R ,A (chamfer / corner rounding / angle)


def _split_comments(line: str, rules: LexerRules) -> tuple[str, str]:
    """(code without comments, comment text)."""
    code: list[str] = []
    comments: list[str] = []
    i = 0
    while i < len(line):
        char = line[i]
        if char == rules.comment_open:
            end = line.find(rules.comment_close, i + 1)
            if end < 0:
                comments.append(line[i + 1:].strip())
                break
            comments.append(line[i + 1:end].strip())
            code.append(" ")
            i = end + 1
            continue
        if rules.eob_comment and char == rules.eob_comment:
            rest = line[i + 1:].strip()
            if rest:
                comments.append(rest)
            break
        code.append(char)
        i += 1
    return "".join(code), " ".join(c for c in comments if c)


def _is_statement(text: str) -> bool:
    if text.startswith("#"):
        return True
    for word in STATEMENT_WORDS:
        if text.startswith(word):
            following = text[len(word):len(word) + 1]
            if word in ("DO", "END"):
                return following.isdigit()
            if word in ("IF", "WHILE", "DPRNT", "BPRNT"):
                return following == "["
            if word == "GOTO":
                return following.isdigit() or following in "#[" and following != ""
            return following == ""
    return False


def _literal(text: str, i: int) -> tuple[float, bool, str, int, str | None]:
    """Parse a literal number at text[i]: (value, has_point, literal text, end, error key)."""
    start = i
    if i < len(text) and text[i] in "+-":
        i += 1
        if i < len(text) and text[i] in "+-":
            return 0.0, False, text[start:i + 1], i + 1, "illegal_minus"
    digits_start = i
    while i < len(text) and text[i].isdigit():
        i += 1
    point = False
    if i < len(text) and text[i] == ".":
        point = True
        i += 1
        while i < len(text) and text[i].isdigit():
            i += 1
        if i < len(text) and text[i] == ".":
            return 0.0, True, text[start:i + 1], i + 1, "illegal_decimal_point"
    body = text[digits_start:i]
    if not body.replace(".", ""):
        return 0.0, point, text[start:i], i, "no_data_after_address"
    if i < len(text) and text[i] == "-":
        return 0.0, point, text[start:i + 1], i + 1, "illegal_minus"
    return float(text[start:i]), point, text[start:i], i, None


def significant_digits(literal: str) -> int:
    body = literal.lstrip("+-")
    if "." in body:
        whole, fraction = body.split(".", 1)
    else:
        whole, fraction = body, ""
    whole = whole.lstrip("0")
    fraction = fraction.rstrip("0")
    return len(whole) + len(fraction)


def parse_block(line: str, number: int, rules: LexerRules) -> Block:
    original = line.rstrip("\r\n")
    code, comment = _split_comments(original, rules)
    lowercase = any(c.islower() for c in code)
    text = "".join(code.split()).upper()
    error = None
    delete = 0
    i = 0
    if text.startswith("/"):
        delete = 1
        i = 1
        if rules.block_delete_switches > 1 and i < len(text) and text[i].isdigit() and text[i] != "0":
            delete = int(text[i])
            i += 1
    n_number = None
    if text.startswith("N", i) and rules.max_n_digits > 0:
        j = i + 1
        while j < len(text) and text[j].isdigit():
            j += 1
        if j == i + 1:
            error = ("no_data_after_address", "N without a number")
        elif j < len(text) and text[j] == ".":
            error = ("illegal_decimal_point", "decimal point in N")
        elif significant_digits(text[i + 1:j]) > rules.max_n_digits:
            error = ("too_many_digits", f"N{text[i + 1:j]}")
        else:
            n_number = int(text[i + 1:j])
        i = j
    rest = text[i:]
    if rules.program_letter and rest.startswith(rules.program_letter) and delete == 0 and n_number is None \
            and len(rest) > 1 and rest[1].isdigit():
        j = 1
        while j < len(rest) and rest[j].isdigit():
            j += 1
        if j == len(rest):
            o_text = rest[1:j]
            o_error = ("too_many_digits", f"O{o_text}") if significant_digits(o_text) > rules.max_o_digits else None
            return Block(number, original, o_number=int(o_text), comment=comment, error=o_error, lowercase=lowercase)
    if error is not None:
        return Block(number, original, n_number, delete, comment=comment, error=error, lowercase=lowercase)
    if rules.macro and _is_statement(rest):
        try:
            stmt, terms = parse_statement(rest)
        except ParseError as problem:
            return Block(number, original, n_number, delete, comment=comment,
                         error=(problem.key, problem.detail), lowercase=lowercase)
        return Block(number, original, n_number, delete, stmt=stmt, comment=comment, terms=terms,
                     lowercase=lowercase)
    words: list[Word] = []
    terms = 0
    mid_at, mid_switch = -1, 0
    j = 0
    while j < len(rest):
        char = rest[j]
        if char == "/" and rules.mid_block_delete:
            if mid_at < 0:
                mid_at, mid_switch = len(words), 1
            j += 1
            if rules.block_delete_switches > 1 and j < len(rest) and rest[j].isdigit() and rest[j] != "0":
                mid_switch = int(rest[j])
                j += 1
            continue
        if char == "," and j + 1 < len(rest) and rest[j + 1] in rules.comma_letters:
            letter = "," + rest[j + 1]
            j += 2
        elif not char.isalpha():
            error = ("improper_address", f"unexpected '{char}'")
            break
        else:
            letter = char
            if letter not in rules.letters:
                error = ("improper_address", f"address {letter}")
                break
            j += 1
        if j >= len(rest) or rest[j].isalpha() or rest[j] == "/":
            error = ("no_data_after_address", f"{letter} without a value")
            break
        if rest[j] in "#[" or (rest[j] in "+-" and j + 1 < len(rest) and rest[j + 1] in "#["):
            if not rules.macro:
                error = ("improper_address", f"{letter}{rest[j]}: macro expressions are not available")
                break
            try:
                node, end, n_terms = parse_word_value(rest, j)
            except ParseError as problem:
                error = (problem.key, problem.detail)
                break
            words.append(Word(letter, None, True, node, rest[j:end], j))
            terms += n_terms
            j = end
            continue
        value, point, literal, end, problem = _literal(rest, j)
        if problem is not None:
            error = (problem, f"{letter}{literal}")
            break
        limit = rules.max_digits.get(letter)
        if limit is not None and significant_digits(literal) > limit:
            error = ("too_many_digits", f"{letter}{literal}")
            break
        words.append(Word(letter, value, point, None, literal, j))
        j = end
    return Block(number, original, n_number, delete, tuple(words), mid_at, mid_switch, None, comment, error,
                 terms, None, lowercase)


def parse_file(name: str, text: str, rules: LexerRules) -> list[Program]:
    """Split one file into programs."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    percent = [k for k, line in enumerate(lines) if line.strip().startswith("%")]
    if percent:
        first = percent[0] + 1
        last = percent[1] if len(percent) > 1 else len(lines)
    else:
        first, last = 0, len(lines)
    programs: list[Program] = []
    blocks: list[Block] = []
    number: int | None = None
    started = False
    for k in range(first, last):
        line = lines[k]
        if not line.strip():
            continue
        block = parse_block(line, k + 1, rules)
        if block.o_number is not None:
            if started:
                programs.append(make_program(name, number, blocks))
            blocks = []
            number = block.o_number
            started = True
            continue
        started = True
        blocks.append(block)
    if started:
        programs.append(make_program(name, number, blocks))
    return programs


def make_program(file: str, number: int | None, blocks: list[Block]) -> Program:
    """A Program with its sequence-number index (also for dialects that parse their own syntax)."""
    index: dict[int, list[int]] = {}
    for k, block in enumerate(blocks):
        if block.n is not None:
            index.setdefault(block.n, []).append(k)
    name = f"O{number:04d}" if number is not None else file
    return Program(number, name, file, tuple(blocks), {n: tuple(v) for n, v in index.items()})


def build_store(sources: list[ProgramSource], rules: LexerRules) -> tuple[ProgramStore, list[str]]:
    """Parse every source. Returns the store and load problems (duplicate program numbers: the
    control keeps the first one, as it refuses to load a number already in memory)."""
    programs: list[Program] = []
    mains: dict[int, int] = {}
    problems: list[str] = []
    seen: set[int] = set()
    for source in sources:
        parsed = parse_file(source.name, source.text, rules)
        if not parsed:
            problems.append(f"{source.name}: no program in the file")
            continue
        if source.main and source.path >= 0 and source.path not in mains:
            mains[source.path] = len(programs)
        for program in parsed:
            if program.number is not None:
                if program.number in seen:
                    problems.append(f"{source.name}: program O{program.number:04d} already in memory (kept the first)")
                seen.add(program.number)
            programs.append(program)
    return ProgramStore(programs, mains, source_digest(sources)), problems
