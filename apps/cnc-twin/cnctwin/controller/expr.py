"""Custom macro B expressions and statements (Fanuc and dialects that follow it, e.g. Haas).

Parsing produces small immutable tuples (the AST is program data, never state):

    ("num", v)  ("var", n)  ("ivar", node)  ("brk", node)  ("neg", node)
    ("bin", op, a, b)  ("fn", name, (args...))

Statements: ("assign", target, node) with target ("var", n) | ("ivar", node);
("if_goto", cond, node); ("if_then", cond, assign); ("goto", node); ("while", cond, m);
("do", m); ("end", m); ("dprnt", items); ("popen",); ("pclos",).

Semantics (Fanuc Series 0i-F operator's manual, custom macro chapter; Smid 2005):

* Priority: functions > ``* / AND MOD`` > ``+ - OR XOR`` > ``EQ NE GT LT GE LE``; brackets ``[ ]``
  nest at most five deep (PS0118).
* ``<vacant>`` (a never-assigned variable, or #0): a lone variable keeps it (``#2 = #1`` leaves #2
  vacant), any operation treats it as 0 (``#1 * 5`` is 0); ``EQ``/``NE`` distinguish vacant from 0,
  the other comparisons treat it as 0.
* Trigonometry in degrees; ``ATAN[a]/[b]`` (or ``ATAN[a, b]``) is the two-argument arctangent with
  the result in 0..360 deg (parameter NAT = 0) or -180..180 deg (NAT = 1); ``ASIN`` gives 270..90 deg
  (NAT = 0) or -90..90 deg (NAT = 1).
* ``ROUND`` rounds half away from zero (to an integer in a macro statement, to the least input
  increment inside an NC address); ``FIX`` truncates toward zero; ``FUP`` raises away from zero.
* ``AND OR XOR`` work bit by bit on 32-bit integers (operands rounded); ``MOD`` rounds its operands
  to integers and keeps the sign of the dividend.
"""

from __future__ import annotations

import math

from .alarms import ControllerAlarm

MAX_BRACKET_DEPTH = 5
FUNCTIONS = ("ROUND", "ASIN", "ACOS", "ATAN", "SQRT", "SIN", "COS", "TAN", "ABS", "BIN", "BCD", "FIX",
             "FUP", "EXP", "POW", "ADP", "LN")
MUL_OPS = ("AND", "MOD")
ADD_OPS = ("XOR", "OR")
REL_OPS = ("EQ", "NE", "GT", "LT", "GE", "LE")
STATEMENT_WORDS = ("WHILE", "GOTO", "IF", "DO", "END", "DPRNT", "POPEN", "PCLOS", "BPRNT")


class ParseError(Exception):
    def __init__(self, key: str, detail: str):
        super().__init__(detail)
        self.key = key
        self.detail = detail


class NeedLive(Exception):
    """A statement read a live machine value (encoder position, servo error) that needs a buffer stop."""

    def __init__(self, variable: int):
        super().__init__(variable)
        self.variable = variable


# --------------------------------------------------------------------------- parsing

class _Parser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0
        self.depth = 0
        self.terms = 0

    def peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.s[j] if j < len(self.s) else ""

    def eof(self) -> bool:
        return self.i >= len(self.s)

    def keyword(self, words) -> str | None:
        for word in words:
            if self.s.startswith(word, self.i):
                return word
        return None

    def expect(self, char: str, what: str) -> None:
        if self.peek() != char:
            raise ParseError("expression_format", f"expected '{char}' {what} at column {self.i + 1}")
        self.i += 1

    # number -------------------------------------------------------------
    def number(self) -> float:
        start = self.i
        while self.peek().isdigit():
            self.i += 1
        if self.peek() == ".":
            self.i += 1
            while self.peek().isdigit():
                self.i += 1
        text = self.s[start:self.i]
        if text in ("", "."):
            raise ParseError("expression_format", f"number expected at column {start + 1}")
        if self.peek() == ".":
            raise ParseError("illegal_decimal_point", f"second decimal point at column {self.i + 1}")
        return float(text)

    def integer(self) -> int:
        start = self.i
        while self.peek().isdigit():
            self.i += 1
        if start == self.i:
            raise ParseError("expression_format", f"integer expected at column {start + 1}")
        if self.peek() == ".":
            raise ParseError("illegal_decimal_point", f"decimal point not allowed at column {self.i + 1}")
        return int(self.s[start:self.i])

    # expressions --------------------------------------------------------
    def bracketed(self) -> tuple:
        self.expect("[", "")
        self.depth += 1
        if self.depth > MAX_BRACKET_DEPTH:
            raise ParseError("bracket_nesting", f"more than {MAX_BRACKET_DEPTH} bracket levels")
        node = self.relational()
        self.expect("]", "closing the bracket")
        self.depth -= 1
        return ("brk", node)

    def variable(self) -> tuple:
        self.expect("#", "")
        self.terms += 1
        if self.peek() == "[":
            return ("ivar", self.bracketed())
        if self.peek() == "-":
            raise ParseError("variable_out_of_range", "negative variable number")
        if not self.peek().isdigit():
            raise ParseError("expression_format", f"variable number expected at column {self.i + 1}")
        return ("var", self.integer())

    def primary(self) -> tuple:
        char = self.peek()
        if char == "#":
            return self.variable()
        if char == "[":
            return self.bracketed()
        if char.isdigit() or char == ".":
            return ("num", self.number())
        name = self.keyword(FUNCTIONS)
        if name is not None:
            self.i += len(name)
            self.terms += 1
            if self.peek() != "[":
                raise ParseError("expression_format", f"{name} needs a bracketed argument")
            if name in ("POW", "ATAN"):
                self.expect("[", "")
                self.depth += 1
                if self.depth > MAX_BRACKET_DEPTH:
                    raise ParseError("bracket_nesting", f"more than {MAX_BRACKET_DEPTH} bracket levels")
                first = self.relational()
                if self.peek() == ",":
                    self.i += 1
                    second = self.relational()
                    self.expect("]", f"closing {name}")
                    self.depth -= 1
                    return ("fn", name, (first, second))
                self.expect("]", f"closing {name}")
                self.depth -= 1
                if name == "ATAN" and self.peek() == "/" and self.peek(1) == "[":
                    self.i += 1
                    second = self.bracketed()
                    return ("fn", name, (first, second))
                if name == "POW":
                    raise ParseError("expression_format", "POW needs two arguments: POW[a,b]")
                return ("fn", name, (first,))
            argument = self.bracketed()
            if name == "ADP" and _strip_brackets(argument)[0] != "var":
                raise ParseError("expression_format", "ADP needs a variable argument: ADP[#n]")
            return ("fn", name, (argument,))
        raise ParseError("expression_format", f"unexpected '{char or 'end of block'}' at column {self.i + 1}")

    def unary(self) -> tuple:
        if self.peek() == "-":
            self.i += 1
            self.terms += 1
            return ("neg", self.unary())
        if self.peek() == "+":
            self.i += 1
            return self.unary()
        return self.primary()

    def multiplicative(self) -> tuple:
        node = self.unary()
        while True:
            char = self.peek()
            if char in ("*", "/"):
                self.i += 1
                op = char
            else:
                op = self.keyword(MUL_OPS)
                if op is None:
                    return node
                self.i += len(op)
            self.terms += 1
            node = ("bin", op, node, self.unary())

    def additive(self) -> tuple:
        node = self.multiplicative()
        while True:
            char = self.peek()
            if char in ("+", "-"):
                self.i += 1
                op = char
            else:
                op = self.keyword(ADD_OPS)
                if op is None:
                    return node
                self.i += len(op)
            self.terms += 1
            node = ("bin", op, node, self.multiplicative())

    def relational(self) -> tuple:
        node = self.additive()
        while True:
            op = self.keyword(REL_OPS)
            if op is None:
                return node
            self.i += len(op)
            self.terms += 1
            node = ("bin", op, node, self.additive())

    # statements ---------------------------------------------------------
    def assignment(self) -> tuple:
        target = self.variable()          # writing #0 is a run-time alarm (PS0116), as on the control
        self.expect("=", "after the variable")
        if self.eof():
            raise ParseError("expression_format", "nothing after '='")
        value = self.relational()
        return ("assign", target, value)

    def condition(self) -> tuple:
        if self.peek() != "[":
            raise ParseError("macro_format", "the condition must be in brackets")
        return self.bracketed()

    def statement(self) -> tuple:
        if self.peek() == "#":
            stmt = self.assignment()
        else:
            word = self.keyword(STATEMENT_WORDS)
            if word is None:
                raise ParseError("macro_format", "not a macro statement")
            self.i += len(word)
            if word == "IF":
                cond = self.condition()
                then = self.keyword(("GOTO", "THEN"))
                if then is None:
                    raise ParseError("macro_format", "IF needs GOTO or THEN")
                self.i += len(then)
                if then == "GOTO":
                    stmt = ("if_goto", cond, self.relational())
                else:
                    if self.peek() != "#":
                        raise ParseError("macro_format", "THEN needs an assignment")
                    stmt = ("if_then", cond, self.assignment())
            elif word == "WHILE":
                cond = self.condition()
                if self.keyword(("DO",)) is None:
                    raise ParseError("macro_format", "WHILE needs DO m")
                self.i += 2
                stmt = ("while", cond, self.loop_number())
            elif word == "GOTO":
                stmt = ("goto", self.relational())
            elif word == "DO":
                stmt = ("do", self.loop_number())
            elif word == "END":
                stmt = ("end", self.loop_number())
            elif word == "POPEN":
                stmt = ("popen",)
            elif word == "PCLOS":
                stmt = ("pclos",)
            else:
                stmt = (word.lower(), self.print_items())
        if not self.eof():
            rest = self.s[self.i:]
            if rest[:1].isalpha() and stmt[0] == "assign":
                raise ParseError("nc_macro_same_block", f"NC words after a macro statement: {rest}")
            raise ParseError("macro_format", f"unexpected '{rest}' after the statement")
        return stmt

    def loop_number(self) -> int:
        if not self.peek().isdigit():
            raise ParseError("illegal_loop_number", "DO/END needs an identification number")
        return self.integer()

    def print_items(self) -> tuple:
        """DPRNT[text #var[ab] ...]: literal text and variables with a [digits fraction] format."""
        self.expect("[", "after DPRNT")
        items: list = []
        text = []
        while not self.eof() and self.peek() != "]":
            if self.peek() == "#":
                if text:
                    items.append(("text", "".join(text)))
                    text = []
                var = self.variable()
                digits, fraction = 8, 3
                if self.peek() == "[":
                    self.i += 1
                    spec = self.integer()
                    self.expect("]", "closing the format")
                    digits, fraction = spec // 10, spec % 10
                items.append(("var", var, digits, fraction))
            else:
                text.append(self.peek())
                self.i += 1
        self.expect("]", "closing DPRNT")
        if text:
            items.append(("text", "".join(text)))
        return tuple(items)


def _strip_brackets(node: tuple) -> tuple:
    while node[0] == "brk":
        node = node[1]
    return node


def parse_expression(text: str) -> tuple[tuple, int]:
    """(AST, number of terms) of an expression written without spaces, in upper case."""
    parser = _Parser(text)
    node = parser.relational()
    if not parser.eof():
        raise ParseError("expression_format", f"unexpected '{text[parser.i:]}' in expression")
    return node, parser.terms


def parse_statement(text: str) -> tuple[tuple, int]:
    """(statement AST, number of terms) of a macro statement written without spaces, upper case."""
    parser = _Parser(text)
    stmt = parser.statement()
    return stmt, parser.terms + 1


def parse_word_value(text: str, start: int) -> tuple[tuple, int, int]:
    """An address value that is an expression (``#1``, ``-#1``, ``[#1+2]``, ``#[#3]``) starting at
    ``start``: (AST, end index, terms)."""
    parser = _Parser(text)
    parser.i = start
    negative = False
    if parser.peek() in "+-" and parser.peek(1) in "#[":
        negative = parser.peek() == "-"
        parser.i += 1
    if parser.peek() == "#":
        node = parser.variable()
    elif parser.peek() == "[":
        node = parser.bracketed()
    else:
        raise ParseError("expression_format", "expression expected")
    if negative:
        node = ("neg", node)
        parser.terms += 1
    return node, parser.i, parser.terms


def is_lone_variable(node: tuple) -> bool:
    """True for ``#n``, ``#[..]``, ``[#n]`` and ``-#n`` (an address with such a value is ignored when vacant)."""
    node = _strip_brackets(node)
    if node[0] == "neg":
        node = _strip_brackets(node[1])
    return node[0] in ("var", "ivar")


# --------------------------------------------------------------------------- evaluation

def round_half_away(value: float) -> float:
    return math.copysign(math.floor(abs(value) + 0.5), value)


def _num(value):
    return 0.0 if value is None else value


def _int32(value: float) -> int:
    number = int(round_half_away(value))
    number &= 0xFFFFFFFF
    return number - (1 << 32) if number & 0x80000000 else number


def _check(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        raise ControllerAlarm("overflow_float", "result out of the floating-point range")
    return value


class Evaluator:
    """Evaluates ASTs against an environment.

    ``env.read(n) -> float | None`` reads variable n (may raise ControllerAlarm or NeedLive);
    ``env.adp(n, value) -> float`` scales an argument passed without decimal point;
    ``env.nat`` is the NAT parameter (False: 0..360 deg ranges).
    """

    def __init__(self, env):
        self.env = env

    def value(self, node: tuple, round_unit: float = 1.0):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "var":
            return self.env.read(node[1])
        if kind == "brk":
            return self.value(node[1], round_unit)
        if kind == "ivar":
            return self.env.read(self.index(node[1]))
        if kind == "neg":
            return 0.0 - _num(self.value(node[1], round_unit))
        if kind == "bin":
            return self.binary(node[1], node[2], node[3], round_unit)
        if kind == "fn":
            return self.function(node[1], node[2], round_unit)
        raise ControllerAlarm("expression_format", f"unknown node {kind}")

    def number(self, node: tuple, round_unit: float = 1.0) -> float:
        return _num(self.value(node, round_unit))

    def address_value(self, node: tuple, round_unit: float):
        """Value of an NC address expression; None (the address is ignored) when the value is a lone
        variable, or the negation of one, that is <vacant>."""
        core = _strip_brackets(node)
        if core[0] == "neg" and is_lone_variable(core):
            inner = self.value(core[1], round_unit)
            return None if inner is None else 0.0 - inner
        return self.value(node, round_unit)

    def index(self, node: tuple) -> int:
        value = self.value(node)
        if value is None:
            raise ControllerAlarm("variable_out_of_range", "variable number is <vacant>")
        number = round_half_away(value)
        if number < 0:
            raise ControllerAlarm("variable_out_of_range", f"#{number:g}")
        return int(number)

    def truth(self, node: tuple) -> bool:
        return _num(self.value(node)) != 0.0

    def binary(self, op: str, a_node: tuple, b_node: tuple, round_unit: float):
        a = self.value(a_node, round_unit)
        b = self.value(b_node, round_unit)
        if op in ("EQ", "NE"):
            if a is None or b is None:
                equal = a is None and b is None
            else:
                equal = a == b
            return 1.0 if (equal if op == "EQ" else not equal) else 0.0
        x, y = _num(a), _num(b)
        if op == "+":
            return _check(x + y)
        if op == "-":
            return _check(x - y)
        if op == "*":
            return _check(x * y)
        if op == "/":
            if y == 0.0:
                raise ControllerAlarm("zero_divide", "division by zero")
            return _check(x / y)
        if op == "GT":
            return 1.0 if x > y else 0.0
        if op == "LT":
            return 1.0 if x < y else 0.0
        if op == "GE":
            return 1.0 if x >= y else 0.0
        if op == "LE":
            return 1.0 if x <= y else 0.0
        if op == "AND":
            return float(_int32(x) & _int32(y))
        if op == "OR":
            return float(_int32(x) | _int32(y))
        if op == "XOR":
            return float(_int32(x) ^ _int32(y))
        if op == "MOD":
            dividend, divisor = round_half_away(x), round_half_away(y)
            if divisor == 0.0:
                raise ControllerAlarm("zero_divide", "MOD by zero")
            return math.fmod(dividend, divisor)
        raise ControllerAlarm("expression_format", f"unknown operator {op}")

    def function(self, name: str, args: tuple, round_unit: float):
        nat = bool(getattr(self.env, "nat", False))
        if name == "ADP":
            target = _strip_brackets(args[0])
            return self.env.adp(target[1], self.env.read(target[1]))
        x = self.number(args[0])
        if name == "SIN":
            return math.sin(math.radians(x))
        if name == "COS":
            return math.cos(math.radians(x))
        if name == "TAN":
            return _check(math.tan(math.radians(x)))
        if name == "SQRT":
            if x < 0.0:
                raise ControllerAlarm("argument_out_of_range", f"SQRT[{x:g}]")
            return math.sqrt(x)
        if name == "ABS":
            return abs(x)
        if name in ("ASIN", "ACOS"):
            if not -1.0 <= x <= 1.0:
                raise ControllerAlarm("argument_out_of_range", f"{name}[{x:g}]")
            if name == "ACOS":
                return math.degrees(math.acos(x))
            angle = math.degrees(math.asin(x))
            return angle + 360.0 if (angle < 0.0 and not nat) else angle
        if name == "ATAN":
            if len(args) == 1:
                angle = math.degrees(math.atan(x))
                return angle + 360.0 if (angle < 0.0 and not nat) else angle
            y = self.number(args[1])
            if x == 0.0 and y == 0.0:
                raise ControllerAlarm("argument_out_of_range", "ATAN[0]/[0]")
            angle = math.degrees(math.atan2(x, y))
            if not nat and angle < 0.0:
                angle += 360.0
            return angle
        if name == "ROUND":
            if round_unit == 1.0:
                return round_half_away(x)
            return round_half_away(x / round_unit) * round_unit
        if name == "FIX":
            return float(math.trunc(x))
        if name == "FUP":
            return math.copysign(float(math.ceil(abs(x))), x) if x != 0.0 else 0.0
        if name == "LN":
            if x <= 0.0:
                raise ControllerAlarm("argument_out_of_range", f"LN[{x:g}]")
            return math.log(x)
        if name == "EXP":
            try:
                return _check(math.exp(x))
            except OverflowError:
                raise ControllerAlarm("overflow_float", f"EXP[{x:g}]") from None
        if name == "POW":
            y = self.number(args[1])
            if x == 0.0 and y < 0.0:
                raise ControllerAlarm("zero_divide", f"POW[0,{y:g}]")
            if x < 0.0 and y != math.floor(y):
                raise ControllerAlarm("argument_out_of_range", f"POW[{x:g},{y:g}]")
            try:
                return _check(math.pow(x, y))
            except OverflowError:
                raise ControllerAlarm("overflow_float", f"POW[{x:g},{y:g}]") from None
        if name in ("BIN", "BCD"):
            if x < 0.0 or x != math.floor(x) or x >= 2.0 ** 31:
                raise ControllerAlarm("argument_out_of_range", f"{name}[{x:g}]")
            number = int(x)
            result = 0
            scale = 1
            if name == "BIN":
                while number:
                    digit = number & 0xF
                    if digit > 9:
                        raise ControllerAlarm("argument_out_of_range", f"BIN[{x:g}] is not BCD")
                    result += digit * scale
                    number >>= 4
                    scale *= 10
            else:
                while number:
                    result += (number % 10) * scale
                    number //= 10
                    scale *= 16
            return float(result)
        raise ControllerAlarm("expression_format", f"unknown function {name}")
