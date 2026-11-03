"""SINUMERIK 840D sl program syntax: files -> programs -> blocks (immutable), with the structure index.

Program files: one program per file, named after the file (``PART1.MPF`` -> ``PART1``), or several in
archive form (``%_N_PART1_MPF`` / ``%_N_SUB1_SPF`` header lines), or a library file with several
``PROC NAME(...)`` programs (a ``PROC`` line after other blocks starts the next program). One line is
one block: ``[/n] [N123 | :123] [LABEL:] items [; comment]``. Items are G functions (``G1``, ``CIP``,
``G641``), addresses with a number (``X10.5``, ``F200``, ``M3``) or a value (``X=R1+2``, ``X=AC(10)``,
``CR=-5``, ``S1=1000``), assignments (``R1=5``, ``NAME=..``, ``R[R2]=..``, ``$P_PFRAME=..``), keywords
(``TRANS``, ``STOPRE``, ``TRAORI``), calls (``CYCLE81(10,0,2,-20)``, ``SUB1 P3``, ``SUB2(1,,R1)``,
``MCALL CYCLE81(..)``, ``CALL "SUB"``, ``MSG("..")``) and jumps (``GOTOF LABEL``). Control structures
(``IF``/``ELSE``/``ENDIF``, ``WHILE``, ``FOR``, ``LOOP``, ``REPEAT``/``UNTIL``), program section
repetition (``REPEAT label [label] [P=n]``, ``REPEATB label [P=n]``), ``DEF``, ``PROC`` and ``EXTERN``
are whole-block statements. Text is case-insensitive outside strings.

Parsing never raises: a block that cannot be read carries ``error = (key, detail)`` and the alarm
is raised when control reaches it. Structures are matched per program after parsing (the partners
of each IF/ELSE/ENDIF ... are stored in the block), so the executor needs no run-time loop stack.
Each program ends with a synthetic end-of-file block (alarm 14000 when reached, as on the control).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...lexer import LexerRules
from ...program import Block, Program, ProgramStore, Word, source_digest
from .tables import FRAME_WORDS, G_FUNCTIONS, UNMODELLED

STATEMENTS = ("IF", "ELSE", "ENDIF", "WHILE", "ENDWHILE", "FOR", "ENDFOR", "LOOP", "ENDLOOP", "REPEAT", "UNTIL",
              "REPEATB", "DEF", "PROC", "EXTERN")
OPEN_CLOSE = {"IF": "ENDIF", "WHILE": "ENDWHILE", "FOR": "ENDFOR", "LOOP": "ENDLOOP", "REPEAT": "UNTIL"}
CLOSERS = {v: k for k, v in OPEN_CLOSE.items()}
TYPES = ("INT", "REAL", "BOOL", "CHAR", "STRING", "AXIS", "FRAME")
MODIFIERS = ("AC", "IC", "DC", "ACP", "ACN")
REL_OPS = ("==", "<>", "<=", ">=", "<", ">")
LOGIC = (("OR",), ("XOR",), ("AND",), ("B_OR",), ("B_XOR",), ("B_AND",))
MUL_WORDS = ("DIV", "MOD")
FUNCTIONS = frozenset({"SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN2", "SQRT", "POT", "ABS", "TRUNC", "ROUND",
                       "ROUNDUP", "LN", "EXP", "MINVAL", "MAXVAL", "BOUND", "CTRANS", "CROT", "CSCALE", "CMIRROR",
                       "NUMBER", "TOSTR", "STRLEN"})
SINGLE = re.compile(r"^([A-Z])(\d+)$")
WORD_RUN = re.compile(r"^(?:[A-Z]\d+)+$")
LABEL_NAME = re.compile(r"^[A-Z_][A-Z_][A-Z0-9_]*$")
G_NAME = re.compile(r"^G0*(\d+)$")


class SyntaxProblem(Exception):
    def __init__(self, key: str, detail: str, *params):
        super().__init__(detail)
        self.key = key
        self.detail = detail
        self.params = params


@dataclass(frozen=True)
class SBlock:
    """Payload of one Siemens block (``Block.extra``)."""

    kind: str                       # "nc" | "stmt" | "eof" | "empty"
    label: str = ""
    items: tuple = ()               # NC items in program order (see the module docstring)
    stmt: tuple | None = None       # whole-block statement
    match: tuple = ()               # ((role, block index), ...) partners of a control structure
    params: tuple = ()              # alarm parameters of a parse error


@dataclass
class ProgramInfo:
    name: str
    labels: dict = field(default_factory=dict)      # label -> block index
    proc: tuple | None = None                       # ("proc", name, params, flags)


class SiemensStore(ProgramStore):
    """ProgramStore with each program's labels and PROC declaration."""

    def __init__(self, programs, mains, digest, info: list[ProgramInfo]):
        super().__init__(programs, mains, digest)
        self.info = info


# ====================================================================== expressions and items

class _Cursor:
    def __init__(self, text: str, axes: frozenset, programs: frozenset):
        self.t = text
        self.i = 0
        self.axes = axes
        self.programs = programs
        self.terms = 0

    # -- characters --------------------------------------------------------
    def ws(self) -> None:
        while self.i < len(self.t) and self.t[self.i] in " \t":
            self.i += 1

    def peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.t[j] if j < len(self.t) else ""

    def at_end(self) -> bool:
        self.ws()
        return self.i >= len(self.t)

    def take(self, token: str) -> bool:
        self.ws()
        if self.t.startswith(token, self.i):
            self.i += len(token)
            return True
        return False

    def expect(self, token: str) -> None:
        if not self.take(token):
            raise SyntaxProblem("syntax", f"'{token}' expected", self.rest())

    def rest(self) -> str:
        return self.t[self.i:].strip() or "end of block"

    def word_follows(self, word: str) -> bool:
        """``word`` as a whole identifier at the cursor."""
        self.ws()
        if not self.t.startswith(word, self.i):
            return False
        after = self.peek(len(word))
        return not (after.isalnum() or after == "_")

    def ident(self) -> str:
        self.ws()
        start = self.i
        if self.i < len(self.t) and (self.t[self.i].isalpha() or self.t[self.i] in "_$"):
            self.i += 1
            while self.i < len(self.t) and (self.t[self.i].isalnum() or self.t[self.i] == "_"):
                self.i += 1
        return self.t[start:self.i]

    def number(self) -> float:
        self.ws()
        start = self.i
        while self.peek().isdigit():
            self.i += 1
        if self.peek() == ".":
            self.i += 1
            while self.peek().isdigit():
                self.i += 1
        digits = self.t[start:self.i]
        if digits in ("", "."):
            raise SyntaxProblem("syntax", "number expected", self.rest())
        if self.t.startswith("EX", self.i):
            j = self.i + 2
            if j < len(self.t) and self.t[j] in "+-":
                j += 1
            k = j
            while k < len(self.t) and self.t[k].isdigit():
                k += 1
            if k > j:
                exponent = self.t[self.i + 2:k]
                self.i = k
                return float(digits) * 10.0 ** int(exponent)
        if self.peek() == ".":
            raise SyntaxProblem("syntax", "second decimal point", self.rest())
        return float(digits)

    def string(self) -> str:
        self.expect('"')
        out = []
        while True:
            if self.i >= len(self.t):
                raise SyntaxProblem("syntax", "string not closed", self.rest())
            char = self.t[self.i]
            self.i += 1
            if char == '"':
                if self.peek() == '"':
                    out.append('"')
                    self.i += 1
                    continue
                return "".join(out)
            out.append(char)

    # -- expressions -------------------------------------------------------
    def expression(self) -> tuple:
        node = self.relational()
        while True:
            self.ws()
            if self.peek() == ":" and self.peek(1) != "=":
                self.i += 1
                self.terms += 1
                node = ("bin", ":", node, self.relational())
            else:
                return node

    def relational(self) -> tuple:
        node = self.concat()
        while True:
            self.ws()
            op = next((o for o in REL_OPS if self.t.startswith(o, self.i)), None)
            if op is None:
                return node
            self.i += len(op)
            self.terms += 1
            node = ("bin", op, node, self.concat())

    def concat(self) -> tuple:
        node = self.logic(0)
        while self.take("<<"):
            self.terms += 1
            node = ("bin", "<<", node, self.logic(0))
        return node

    def logic(self, level: int) -> tuple:
        if level == len(LOGIC):
            return self.additive()
        node = self.logic(level + 1)
        while True:
            op = next((w for w in LOGIC[level] if self.word_follows(w)), None)
            if op is None:
                return node
            self.ws()
            self.i += len(op)
            self.terms += 1
            node = ("bin", op, node, self.logic(level + 1))

    def additive(self) -> tuple:
        node = self.multiplicative()
        while True:
            self.ws()
            char = self.peek()
            if char in "+-" and char:
                self.i += 1
                self.terms += 1
                node = ("bin", char, node, self.multiplicative())
            else:
                return node

    def multiplicative(self) -> tuple:
        node = self.unary()
        while True:
            self.ws()
            char = self.peek()
            if char in "*/" and char:
                self.i += 1
                self.terms += 1
                node = ("bin", char, node, self.unary())
                continue
            op = next((w for w in MUL_WORDS if self.word_follows(w)), None)
            if op is None:
                return node
            self.ws()
            self.i += len(op)
            self.terms += 1
            node = ("bin", op, node, self.unary())

    def unary(self) -> tuple:
        self.ws()
        if self.peek() == "-":
            self.i += 1
            self.terms += 1
            return ("neg", self.unary())
        if self.peek() == "+":
            self.i += 1
            return self.unary()
        for word in ("NOT", "B_NOT"):
            if self.word_follows(word):
                self.ws()
                self.i += len(word)
                self.terms += 1
                return ("not" if word == "NOT" else "bnot", self.unary())
        return self.primary()

    def primary(self) -> tuple:
        self.ws()
        char = self.peek()
        if char.isdigit() or char == ".":
            return ("num", self.number())
        if char == '"':
            return ("str", self.string())
        if char == "(":
            self.i += 1
            node = self.expression()
            self.expect(")")
            return node
        if char.isalpha() or char in "_$":
            name = self.ident()
            self.terms += 1
            if name in ("TRUE", "FALSE"):
                return ("num", 1.0 if name == "TRUE" else 0.0)
            self.ws()
            if self.peek() == "(" and name in FUNCTIONS:
                self.i += 1
                return ("fn", name, self.arguments(")"))
            if self.peek() == "[":
                self.i += 1
                return self.indexed(name)
            if name.startswith("R") and name[1:].isdigit():
                return ("r", ("num", float(int(name[1:]))))
            if name.startswith("$"):
                return ("sys", name, ())
            return ("name", name)
        raise SyntaxProblem("syntax", "operand expected", self.rest())

    def indexed(self, name: str) -> tuple:
        args = self.arguments("]", index=True)
        if name == "R":
            if len(args) != 1:
                raise SyntaxProblem("syntax", "R[...] takes one index", self.rest())
            return ("r", args[0])
        if name.startswith("$"):
            return ("sys", name, tuple(args))
        return ("idx", name, tuple(args))

    def arguments(self, close: str, index: bool = False) -> tuple:
        args: list = []
        self.ws()
        if self.peek() == close:
            self.i += 1
            return ()
        while True:
            self.ws()
            if self.peek() in (",", close):
                args.append(None)
            else:
                args.append(self.index_item() if index else self.expression())
            self.ws()
            if self.peek() == ",":
                self.i += 1
                continue
            self.expect(close)
            return tuple(args)

    def index_item(self) -> tuple:
        """An index: an axis name (``$P_EP[X]``), a frame component (``F1[X,TR]``) or an expression."""
        self.ws()
        start = self.i
        name = self.ident()
        if name:
            self.ws()
            if self.peek() in (",", "]") and (name in self.axes or name in ("TR", "RT", "SC", "MI", "FI")):
                return ("const", name)
            self.i = start
        return self.expression()

    # -- NC items ----------------------------------------------------------
    def value(self) -> tuple[str, tuple]:
        """The value after '=': (modifier, expression); modifier AC IC DC ACP ACN or ''."""
        self.ws()
        start = self.i
        name = self.ident()
        if name in MODIFIERS:
            self.ws()
            if self.peek() == "(":
                self.i += 1
                node = self.expression()
                self.expect(")")
                return name, node
        self.i = start
        return "", self.expression()

    def literal(self) -> tuple:
        """A literal number directly after an address letter: X10, X-5.5, F.5."""
        self.ws()
        sign = 1.0
        if self.peek() in "+-":
            sign = -1.0 if self.peek() == "-" else 1.0
            self.i += 1
        return ("num", sign * self.number())

    def items(self) -> list[tuple]:
        out: list[tuple] = []
        while not self.at_end():
            out.extend(self.item(out))
        return out

    def target(self) -> tuple:
        """A jump destination: a label, N<number> or a string expression."""
        self.ws()
        if self.peek() == '"':
            return ("expr", ("str", self.string()))
        name = self.ident()
        if not name:
            raise SyntaxProblem("syntax", "jump destination expected", self.rest())
        if name.startswith("N") and name[1:].isdigit():
            return ("n", int(name[1:]))
        return ("label", name)

    def call_repeat(self):
        """Optional ``P<n>`` / ``P=<n>`` after a subprogram name."""
        self.ws()
        if self.peek() == "P" and (self.peek(1).isdigit() or self.peek(1) == "="):
            self.i += 1
            if self.peek() == "=":
                self.i += 1
                return self.expression()
            return ("num", self.number())
        return None

    def item(self, before: list) -> list[tuple]:
        self.ws()
        char = self.peek()
        if not (char.isalpha() or char in "_$"):
            raise SyntaxProblem("syntax", f"unexpected '{char}'", self.rest())
        start = self.i
        name = self.ident()
        self.ws()
        nxt = self.peek()
        if name in ("GOTOF", "GOTOB", "GOTO", "GOTOC"):
            return [("goto", name, self.target())]
        if name in STATEMENTS:
            raise SyntaxProblem("syntax", f"{name} must begin its block", name)
        if name == "MCALL":
            if self.at_end():
                return [("mcall", None)]
            inner = self.item([])
            if len(inner) != 1 or inner[0][0] != "call":
                raise SyntaxProblem("syntax", "MCALL needs a subprogram or cycle call", self.rest())
            return [("mcall", inner[0])]
        if name == "CALL":
            return [("callname", self.expression(), self.call_repeat())]
        if name.startswith("$"):
            args = ()
            if nxt == "[":
                self.i += 1
                args = self.arguments("]", index=True)
            self.expect("=")
            return [("assign", ("sys", name, tuple(args)), self.expression())]
        if nxt == "=" and self.peek(1) != "=":
            self.i += 1
            return [self.equals_item(name)]
        if nxt == "[":
            self.i += 1
            if name == "AX":
                args = self.arguments("]", index=True)
                self.expect("=")
                mod, node = self.value()
                return [("axis_ind", args[0] if args else None, mod, node)]
            args = self.arguments("]", index=True)
            self.expect("=")
            target = ("r", args[0]) if name == "R" and len(args) == 1 else ("idx", name, tuple(args))
            return [("assign", target, self.expression())]
        if nxt == "(" and name not in FRAME_WORDS:
            self.i += 1
            args = self.arguments(")")
            return [("call", name, args, self.call_repeat())]
        code = self.g_code(name)
        if code is not None:
            return [("g", code)]
        if name in FRAME_WORDS:
            return [("frame", name)]
        if name in ("STOPRE", "TRAFOOF", "RET", "TRAORI", "SAVE", "DISPLOF", "SBLOF"):
            if name == "TRAORI" and self.peek() == "(":
                self.i += 1
                self.arguments(")")
            return [("kw", name)]
        if name in UNMODELLED:
            return [("unmodelled", name)]
        if name in self.programs and not SINGLE.match(name):
            return [("call", name, (), self.call_repeat())]
        match = SINGLE.match(name)
        if match is not None:
            return [self.address_word(match.group(1), float(int(match.group(2))), before, name)]
        if len(name) == 1 and nxt and (nxt in "+-." or nxt.isdigit()):
            return [self.address_word(name, None, before, name)]
        if len(name) == 1 and (name in self.axes or name in "FSTDHMGIJKLP"):
            raise SyntaxProblem("syntax", f"address {name} without a value", name)
        if name in self.programs:
            return [("call", name, (), self.call_repeat())]
        if WORD_RUN.match(name):
            self.i = start
            return self.word_run()
        raise SyntaxProblem("undefined_name", f"name {name} not defined", name)

    def word_run(self) -> list[tuple]:
        """Words written without spaces: G1X10Y20."""
        out = []
        while self.peek().isalpha():
            letter = self.peek()
            self.i += 1
            out.append(self.address_word(letter, None, out, letter))
        return out

    def g_code(self, name: str) -> str | None:
        match = G_NAME.match(name)
        if match is not None:
            return "G" + str(int(match.group(1)))
        if name in G_FUNCTIONS and not name.startswith("G"):
            return name
        return None

    def address_word(self, letter: str, digits: float | None, before: list, text: str) -> tuple:
        """``X10`` style word: the number continues with a fraction or follows a sign."""
        if digits is None:
            node = self.literal()
        else:
            node = ("num", digits)
            if self.peek() == ".":
                start = self.i
                self.i += 1
                while self.peek().isdigit():
                    self.i += 1
                fraction = self.t[start:self.i]
                node = ("num", float(f"{int(digits)}{fraction}" if fraction != "." else digits))
        if letter == "G":
            value = node[1]
            if value != int(value):
                raise SyntaxProblem("improper_g_code", f"G{value:g}", f"G{value:g}")
            return ("g", "G" + str(int(value)))
        if letter == "M":
            return ("m", node, "")
        if letter == "L":
            value = node[1]
            return ("call", f"L{int(value)}", (), self.call_repeat())
        if letter in self.axes:
            return ("axis", letter, "", node)
        return ("addr", letter, "", node)

    def equals_item(self, name: str) -> tuple:
        mod, node = self.value()
        if name in self.axes:
            return ("axis", name, mod, node)
        if len(name) == 2 and name[0] in self.axes and name[1] == "1":
            return ("axis1", name[0], mod, node)
        if name.startswith("R") and name[1:].isdigit():
            return ("assign", ("r", ("num", float(int(name[1:])))), node)
        from .tables import ADDRESSES

        if name in ADDRESSES or ((name[0] in "SM") and name[1:].isdigit()):
            return ("addr", name, mod, node)
        return ("assign", ("name", name), node)


# ====================================================================== statements

def _statement(cur: _Cursor, word: str) -> tuple:
    if word == "IF":
        cond = cur.expression()
        if cur.at_end():
            return ("if", cond)
        name = cur.ident()
        if name in ("GOTOF", "GOTOB", "GOTO", "GOTOC"):
            target = cur.target()
            if not cur.at_end():
                raise SyntaxProblem("syntax", "text after the jump", cur.rest())
            return ("if_goto", cond, ("goto", name, target))
        raise SyntaxProblem("syntax", "IF condition followed by text", name or cur.rest())
    if word in ("WHILE", "UNTIL"):
        cond = cur.expression()
        _end(cur)
        return (word.lower(), cond)
    if word == "FOR":
        name = cur.ident()
        if not name:
            raise SyntaxProblem("syntax", "FOR needs a variable", cur.rest())
        target = ("r", ("num", float(int(name[1:])))) if name.startswith("R") and name[1:].isdigit() \
            else ("name", name)
        cur.expect("=")
        first = cur.expression()
        if not cur.word_follows("TO"):
            raise SyntaxProblem("syntax", "FOR needs TO", cur.rest())
        cur.ws()
        cur.i += 2
        last = cur.expression()
        _end(cur)
        return ("for", target, first, last)
    if word == "REPEAT" and cur.at_end():
        return ("repeat",)
    if word in ("REPEAT", "REPEATB"):
        # program section repetition: REPEATB label [P=n] (one line), REPEAT label [P=n] (label up to
        # ENDLABEL or up to the REPEAT line), REPEAT label1 label2 [P=n] (label1 to label2)
        first = cur.ident()
        if not LABEL_NAME.match(first or ""):
            raise SyntaxProblem("syntax", f"{word} needs a label", cur.rest())
        second = None
        cur.ws()
        if word == "REPEAT" and not cur.at_end() and not (cur.peek() == "P" and (cur.peek(1) == "=" or
                                                                                 cur.peek(1).isdigit())):
            second = cur.ident()
            if not LABEL_NAME.match(second or ""):
                raise SyntaxProblem("syntax", "REPEAT end label expected", cur.rest())
        count = cur.call_repeat()
        _end(cur)
        return ("repeat_section", word, first, second, count)
    if word in ("ELSE", "ENDIF", "ENDWHILE", "ENDFOR", "LOOP", "ENDLOOP"):
        _end(cur)
        return (word.lower(),)
    if word == "DEF":
        return _definition(cur)
    if word == "PROC":
        return _proc(cur)
    if word == "EXTERN":
        name = cur.ident()
        cur.i = len(cur.t)
        return ("extern", name)
    raise SyntaxProblem("syntax", word, word)


def _end(cur: _Cursor) -> None:
    if not cur.at_end():
        raise SyntaxProblem("syntax", "unexpected text", cur.rest())


def _type(cur: _Cursor) -> tuple[str, int]:
    kind = cur.ident()
    if kind not in TYPES:
        raise SyntaxProblem("syntax", f"data type expected, found {kind or cur.rest()}", kind or cur.rest())
    length = 0
    if kind == "STRING":
        cur.expect("[")
        length = int(cur.number())
        cur.expect("]")
    return kind, length


def _definition(cur: _Cursor) -> tuple:
    kind, length = _type(cur)
    names = []
    while True:
        name = cur.ident()
        if not name or name.startswith("$"):
            raise SyntaxProblem("syntax", "variable name expected", cur.rest())
        cur.ws()
        if cur.peek() == "[":
            raise SyntaxProblem("unsupported_function", f"array variable {name}", name)
        init = None
        if cur.take("="):
            init = cur.expression()
        names.append((name, init))
        if not cur.take(","):
            break
    _end(cur)
    return ("def", kind, length, tuple(names))


def _proc(cur: _Cursor) -> tuple:
    name = cur.ident()
    if not name:
        raise SyntaxProblem("syntax", "PROC needs a name", cur.rest())
    params = []
    if cur.take("("):
        cur.ws()
        if not cur.take(")"):
            while True:
                by_ref = cur.word_follows("VAR")
                if by_ref:
                    cur.ws()
                    cur.i += 3
                kind, length = _type(cur)
                pname = cur.ident()
                if not pname:
                    raise SyntaxProblem("syntax", "parameter name expected", cur.rest())
                cur.ws()
                if cur.peek() == "[":
                    raise SyntaxProblem("unsupported_function", f"array parameter {pname}", pname)
                params.append((kind, pname, by_ref, length))
                if cur.take(","):
                    continue
                cur.expect(")")
                break
    flags = []
    while not cur.at_end():
        flag = cur.ident()
        if flag not in ("SAVE", "DISPLOF", "DISPLON", "SBLOF", "SBLON", "ICYCOF", "ACTBLOCNO", "PREVENT"):
            raise SyntaxProblem("syntax", f"PROC attribute {flag or cur.rest()}", flag or cur.rest())
        flags.append(flag)
    return ("proc", name, tuple(params), tuple(flags))


# ====================================================================== lines and files

def _split_comment(line: str) -> tuple[str, str]:
    quote = False
    for k, char in enumerate(line):
        if char == '"':
            quote = not quote
        elif char == ";" and not quote:
            return line[:k], line[k + 1:].strip()
    return line, ""


def _upper(code: str) -> str:
    out, quote = [], False
    for char in code:
        if char == '"':
            quote = not quote
        out.append(char if quote else char.upper())
    return "".join(out)


@dataclass
class _Line:
    number: int
    text: str
    n: int | None = None
    delete: int = 0
    label: str = ""
    kind: str = "empty"
    items: tuple = ()
    stmt: tuple | None = None
    error: tuple | None = None
    params: tuple = ()
    terms: int = 0
    comment: str = ""


def parse_line(number: int, original: str, axes: frozenset, programs: frozenset) -> _Line:
    code, comment = _split_comment(original.rstrip("\r\n"))
    line = _Line(number, original.rstrip("\r\n"), comment=comment)
    cur = _Cursor(_upper(code), axes, programs)
    try:
        cur.ws()
        if cur.peek() == "/":
            cur.i += 1
            level = 0
            if cur.peek().isdigit():
                level = int(cur.peek())
                cur.i += 1
            if level > 8:
                raise SyntaxProblem("unsupported_function", f"skip level /{level} (the operator panel has /0-/8)",
                                    f"/{level}")
            line.delete = level + 1
        cur.ws()
        if cur.peek() in ("N", ":") and cur.peek(1).isdigit():
            cur.i += 1
            start = cur.i
            while cur.peek().isdigit():
                cur.i += 1
            line.n = int(cur.t[start:cur.i])
        cur.ws()
        save = cur.i
        name = cur.ident()
        cur.ws()
        if name and cur.peek() == ":" and cur.peek(1) != "=" and LABEL_NAME.match(name):
            cur.i += 1
            line.label = name
        else:
            cur.i = save
        if cur.at_end():
            line.kind = "empty"
            return line
        save = cur.i
        word = cur.ident()
        if word in STATEMENTS:
            line.stmt = _statement(cur, word)
            line.kind = "stmt"
        else:
            cur.i = save
            line.items = tuple(cur.items())
            line.kind = "nc" if line.items else "empty"
            if all(item[0] in ("assign", "goto") for item in line.items):
                line.kind = "assign"
    except SyntaxProblem as problem:
        line.error = (problem.key, problem.detail)
        line.params = tuple(problem.params) or (problem.detail,)
        line.kind = "error"
    line.terms = cur.terms
    return line


def _program_name(file: str) -> str:
    stem = file.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.split(".", 1)[0]
    return stem.upper()


HEADER = re.compile(r"^%_N_([A-Z0-9_]+?)_(MPF|SPF)\s*$", re.IGNORECASE)
PROC_LINE = re.compile(r"^\s*(?:N\d+\s+)?PROC\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _split_programs(name: str, text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """(program name, [(line number, text)]) per program in one file."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    programs: list[tuple[str, list[tuple[int, str]]]] = []
    current_name = _program_name(name)
    current: list[tuple[int, str]] = []
    has_code = False
    for k, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        header = HEADER.match(stripped)
        if header is not None:
            if current and has_code:
                programs.append((current_name, current))
            current_name, current, has_code = header.group(1).upper(), [], False
            continue
        if stripped.startswith(";$PATH") or not stripped:
            continue
        proc = PROC_LINE.match(stripped)
        if proc is not None:
            if has_code:
                programs.append((current_name, current))
                current = []
            current_name = proc.group(1).upper()
        code, _ = _split_comment(stripped)
        if code.strip():
            has_code = True
        current.append((k, raw))
    if current and has_code:
        programs.append((current_name, current))
    return programs


def _structure(lines: list[_Line], max_depth: int) -> dict[int, tuple]:
    """Partners of every control-structure block; mismatches become block errors (12640 / 12641)."""
    stack: list[tuple[str, int, list]] = []
    match: dict[int, list] = {}
    for k, line in enumerate(lines):
        if line.stmt is None or line.error is not None:
            continue
        word = line.stmt[0].upper()
        if word in OPEN_CLOSE:
            if len(stack) >= max_depth:
                line.error = ("structure_depth", f"more than {max_depth} nested control structures")
                continue
            stack.append((word, k, []))
        elif word == "ELSE":
            if not stack or stack[-1][0] != "IF" or any(r == "else" for r, _ in stack[-1][2]):
                line.error = ("structure_nesting", "ELSE without IF")
                continue
            stack[-1][2].append(("else", k))
        elif word in CLOSERS:
            if not stack or stack[-1][0] != CLOSERS[word]:
                line.error = ("structure_nesting", f"{word} without {CLOSERS[word]}")
                continue
            opener, start, inner = stack.pop()
            match[start] = [("end", k)] + inner
            match[k] = [("start", start)]
            for role, index in inner:
                match[index] = [("start", start), ("end", k)]
    for opener, start, _ in stack:
        lines[start].error = ("structure_nesting", f"{opener} without {OPEN_CLOSE[opener]}")
    return {k: tuple(v) for k, v in match.items()}


def _check_definitions(lines: list[_Line]) -> None:
    """DEF only in the definition part at the start of a program (after PROC), as the control requires."""
    body = False
    for line in lines:
        if line.error is not None or line.kind == "empty":
            continue
        if line.stmt is not None and line.stmt[0] in ("def", "proc", "extern"):
            if body and line.stmt[0] == "def":
                line.error = ("def_position", "DEF after the first NC instruction")
            if body and line.stmt[0] == "proc":
                line.error = ("def_position", "PROC is not the first block of the program")
            continue
        body = True


def build_store(sources, rules: LexerRules, *, axes: frozenset = frozenset("XYZABCUVW"), max_depth: int = 16):
    """Parse every source into a SiemensStore. Returns (store, load problems)."""
    split = []
    for source in sources:
        split.append((source, _split_programs(source.name, source.text)))
    names = frozenset(name for _, programs in split for name, _ in programs)
    programs: list[Program] = []
    info: list[ProgramInfo] = []
    mains: dict[int, int] = {}
    problems: list[str] = []
    seen: set[str] = set()
    for source, parts in split:
        if not parts:
            problems.append(f"{source.name}: no program in the file")
            continue
        if source.main and source.path >= 0 and source.path not in mains:
            mains[source.path] = len(programs)
        for name, raw_lines in parts:
            if name in seen:
                problems.append(f"{source.name}: program {name} already in memory (kept the first)")
            seen.add(name)
            lines = [parse_line(k, text, axes, names) for k, text in raw_lines]
            _check_definitions(lines)
            match = _structure(lines, max_depth)
            labels: dict[str, int] = {}
            proc = None
            blocks = []
            for k, line in enumerate(lines):
                if line.label:
                    if line.label in labels:
                        line.error = line.error or ("label_twice", line.label)
                    else:
                        labels[line.label] = k
                if line.stmt is not None and line.stmt[0] == "proc" and k == _first_code(lines) and line.error is None:
                    proc = line.stmt
                blocks.append(_block(line, match.get(k, ())))
            last = raw_lines[-1][0] + 1 if raw_lines else 1
            blocks.append(Block(last, "", None, 0, (), -1, 0, None, "", None, 0, None, False, SBlock("eof")))
            index: dict[int, list[int]] = {}
            for k, block in enumerate(blocks):
                if block.n is not None:
                    index.setdefault(block.n, []).append(k)
            programs.append(Program(None, name, source.name, tuple(blocks), {n: tuple(v) for n, v in index.items()}))
            info.append(ProgramInfo(name, labels, proc))
    return SiemensStore(programs, mains, source_digest(sources), info), problems


def _first_code(lines: list[_Line]) -> int:
    for k, line in enumerate(lines):
        if line.kind != "empty":
            return k
    return -1


def _block(line: _Line, match: tuple) -> Block:
    words = tuple(Word(item[1] if isinstance(item[1], str) else item[0][:1].upper())
                  for item in line.items if item[0] in ("axis", "addr", "g", "m", "axis1", "axis_ind"))
    kind = line.kind
    stmt = None
    if kind in ("stmt", "assign"):
        stmt = line.stmt if line.stmt is not None else ("assign",)
    if kind == "nc" and not words:
        words = (Word(";"),)
    payload = SBlock(kind if kind != "error" else "nc", line.label, line.items, line.stmt, match, line.params)
    return Block(line.number, line.text, line.n, line.delete, words, -1, 0, stmt, line.comment, line.error,
                 line.terms, None, False, payload)


__all__ = ["SBlock", "SiemensStore", "ProgramInfo", "build_store", "parse_line", "SyntaxProblem", "FUNCTIONS"]
