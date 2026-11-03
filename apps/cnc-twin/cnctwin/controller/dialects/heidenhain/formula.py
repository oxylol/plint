"""HEIDENHAIN Q-parameter values and formulas (Klartext and ISO).

Nodes (immutable tuples): ("num", v), ("q", table, index) with table Q | QL | QR, ("qs", index),
("neg", n), ("bin", op, a, b) with op + - * / ^ %, ("fn", name, n), ("pi",).

Formula syntax (TNC 640 Klartext manual, Q parameters, "Formula"): + - * / ^ (power), % (modulo), ( ),
the functions SIN COS TAN ASIN ACOS ATAN (degrees), SQ (square), SQRT, LN, LOG (base 10), EXP, ABS,
INT (integer part, toward zero), FRAC (fraction), SGN (-1, 0, +1), NEG, and the constant PI. Priority:
functions and unary signs, ^ (right-associative), * / %, + -. A Q parameter that is undefined (SET
UNDEFINED) in a calculation is error 160-0028; as a lone coordinate value its axis is ignored.

String functions with a numerical result ("sfn" nodes): TONUMB ( SRC_QSn ) the number a string holds,
STRLEN ( SRC_QSn ) its length (-1 for an undefined string), INSTR ( SRC_QSn SEA_QSm BEGk ) the position
(from 0) of QSm in QSn from position k on (the length of QSn when absent), STRCOMP ( SRC_QSn SEA_QSm ) 0
equal, -1 QSn sorts first, +1 QSn sorts after.
"""

from __future__ import annotations

import math
import re

from ..siemens840d.kit import Alarm

FUNCTIONS = ("SQRT", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "SQ", "LN", "LOG", "EXP", "ABS", "INT", "FRAC",
             "SGN", "NEG")
STRING_FUNCTIONS = ("TONUMB", "INSTR", "STRLEN", "STRCOMP")
Q_REF = re.compile(r"^(QL|QR|QS|Q)(\d+)$")
STRING_ARG = re.compile(r"^(?:(SRC|SEA)_QS(\d+)|(BEG|LEN|DAT|DECIMALS)\s*(.+))$")


class FormulaError(Exception):
    pass


def q_node(text: str) -> tuple | None:
    match = Q_REF.match(text)
    if match is None:
        return None
    table, index = match.group(1), int(match.group(2))
    if table == "QS":
        return ("qs", index)
    return ("q", table, index)


def value_node(text: str) -> tuple:
    """A coordinate / parameter value: +10, -5.5, .5, +Q1, -QL3, Q2."""
    t = text.strip()
    if not t:
        raise FormulaError("value missing")
    sign = 1.0
    body = t
    if t[0] in "+-":
        sign = -1.0 if t[0] == "-" else 1.0
        body = t[1:]
    node = q_node(body)
    if node is not None:
        return ("neg", node) if sign < 0 else node
    try:
        return ("num", sign * float(body))
    except ValueError:
        raise FormulaError(f"value {text!r}") from None


class _Parser:
    def __init__(self, text: str):
        self.t = text
        self.i = 0

    def ws(self):
        while self.i < len(self.t) and self.t[self.i] in " \t":
            self.i += 1

    def peek(self) -> str:
        self.ws()
        return self.t[self.i] if self.i < len(self.t) else ""

    def expr(self) -> tuple:
        node = self.term()
        while self.peek() in ("+", "-") and self.peek():
            op = self.t[self.i]
            self.i += 1
            node = ("bin", op, node, self.term())
        return node

    def term(self) -> tuple:
        node = self.power()
        while self.peek() in ("*", "/", "%") and self.peek():
            op = self.t[self.i]
            self.i += 1
            node = ("bin", op, node, self.power())
        return node

    def power(self) -> tuple:
        base = self.unary()
        if self.peek() == "^":
            self.i += 1
            return ("bin", "^", base, self.power())
        return base

    def unary(self) -> tuple:
        char = self.peek()
        if char == "-":
            self.i += 1
            return ("neg", self.unary())
        if char == "+":
            self.i += 1
            return self.unary()
        return self.primary()

    def primary(self) -> tuple:
        char = self.peek()
        if char == "(":
            self.i += 1
            node = self.expr()
            if self.peek() != ")":
                raise FormulaError("')' expected")
            self.i += 1
            return node
        if char.isdigit() or char == ".":
            start = self.i
            while self.i < len(self.t) and (self.t[self.i].isdigit() or self.t[self.i] == "."):
                self.i += 1
            try:
                return ("num", float(self.t[start:self.i]))
            except ValueError:
                raise FormulaError(self.t[start:self.i]) from None
        if char.isalpha():
            start = self.i
            while self.i < len(self.t) and (self.t[self.i].isalnum() or self.t[self.i] == "_"):
                self.i += 1
            word = self.t[start:self.i]
            if word == "PI":
                return ("pi",)
            node = q_node(word)
            if node is not None:
                return node
            if word in FUNCTIONS:
                return ("fn", word, self.unary())
            if word in STRING_FUNCTIONS:
                return ("sfn", word, string_arguments(self))
            raise FormulaError(f"unknown name {word}")
        raise FormulaError(f"unexpected {char or 'end'}")


def string_arguments(parser) -> tuple:
    """``( SRC_QSn SEA_QSm BEGk LENk DAT+value DECIMALSn )`` -> ((name, value), ...), indices as ints, the
    others as value nodes."""
    if parser.peek() != "(":
        raise FormulaError("'(' expected")
    parser.i += 1
    close = parser.t.find(")", parser.i)
    if close < 0:
        raise FormulaError("')' expected")
    body, parser.i = parser.t[parser.i:close], close + 1
    return parse_string_arguments(body)


def parse_string_arguments(body: str) -> tuple:
    out = {}
    for token in body.split():
        match = STRING_ARG.match(token)
        if match is None:
            raise FormulaError(f"string function argument {token}")
        if match.group(1):
            out[match.group(1)] = int(match.group(2))
        else:
            out[match.group(3)] = value_node(match.group(4))
    return tuple(sorted(out.items()))


def parse_formula(text: str) -> tuple:
    parser = _Parser(text)
    node = parser.expr()
    if parser.peek():
        raise FormulaError(f"unexpected {parser.t[parser.i:]}")
    return node


class Values:
    """Q parameter storage of one path (PathState) and the evaluator."""

    def __init__(self, handler):
        self.h = handler
        self.s = handler.s

    def table(self, name: str) -> list:
        if name == "QL":
            return self.h.ql_scope()
        return self.s.variables[name]

    def get(self, table: str, index: int):
        if table == "Q" and 100 <= index <= 199:
            return self.h.preassigned(index)
        values = self.table(table)
        if not 0 <= index < len(values):
            raise Alarm("q_invalid", f"{table}{index}")
        return values[index]

    def set(self, table: str, index: int, value) -> None:
        if table == "Q" and 100 <= index <= 199:
            raise Alarm("write_protected", f"Q{index} is read only")
        values = self.table(table)
        if not 0 <= index < len(values):
            raise Alarm("q_invalid", f"{table}{index}")
        values[index] = None if value is None else float(value)

    def get_string(self, index: int) -> str:
        strings = self.s.sys.setdefault("qs", {})
        if str(index) not in strings:
            raise Alarm("string_undeclared", f"QS{index}")
        return strings[str(index)]

    def set_string(self, index: int, value: str) -> None:
        if not 0 <= index <= 1999:
            raise Alarm("q_invalid", f"QS{index}")
        self.s.sys.setdefault("qs", {})[str(index)] = str(value)

    def value(self, node, lone: bool = False):
        """Number of a node; ``lone`` returns None for an undefined lone parameter (the axis is ignored)."""
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "q":
            value = self.get(node[1], node[2])
            if value is None:
                if lone:
                    return None
                raise Alarm("q_undefined", f"{node[1]}{node[2]} is undefined")
            return value
        if kind == "neg":
            inner = self.value(node[1], lone)
            return None if inner is None else -inner
        if kind == "pi":
            return math.pi
        if kind == "qs":
            raise Alarm("q_invalid", f"QS{node[1]} in a calculation")
        if kind == "bin":
            a, b = self.value(node[2]), self.value(node[3])
            return self.binary(node[1], a, b)
        if kind == "fn":
            return self.function(node[1], self.value(node[2]))
        if kind == "sfn":
            return self.string_function(node[1], dict(node[2]))
        raise Alarm("syntax", f"value {kind}")

    def string_function(self, name: str, args: dict) -> float:
        def text(key: str) -> str:
            if key not in args:
                raise Alarm("syntax", f"{name} without {key}_QS")
            return self.get_string(args[key])
        if name == "STRLEN":
            if "SRC" not in args:
                raise Alarm("syntax", "STRLEN without SRC_QS")
            strings = self.s.sys.get("qs", {})
            return float(len(strings[str(args["SRC"])])) if str(args["SRC"]) in strings else -1.0
        if name == "TONUMB":
            source = text("SRC").strip()
            try:
                return _finite(float(source))
            except ValueError:
                raise Alarm("q_invalid", f"TONUMB of {source!r}") from None
        if name == "STRCOMP":
            a, b = text("SRC"), text("SEA")
            return 0.0 if a == b else (-1.0 if a < b else 1.0)
        if name == "INSTR":
            a, b = text("SRC"), text("SEA")
            begin = int(self.value(args["BEG"])) if "BEG" in args else 0
            if begin < 0:
                raise Alarm("q_invalid", f"INSTR BEG{begin}")
            found = a.find(b, begin)
            return float(found if found >= 0 else len(a))
        raise Alarm("syntax", f"string function {name}")

    def binary(self, op: str, a: float, b: float) -> float:
        if op == "+":
            return _finite(a + b)
        if op == "-":
            return _finite(a - b)
        if op == "*":
            return _finite(a * b)
        if op == "/":
            if b == 0.0:
                raise Alarm("zero_divide", "division by 0")
            return _finite(a / b)
        if op == "%":
            if b == 0.0:
                raise Alarm("zero_divide", "modulo 0")
            return math.fmod(a, b)
        if op == "^":
            try:
                return _finite(math.pow(a, b))
            except (ValueError, OverflowError):
                raise Alarm("argument_out_of_range", f"{a:g} ^ {b:g}") from None
        raise Alarm("syntax", f"operator {op}")

    def function(self, name: str, x: float) -> float:
        if name == "SIN":
            return math.sin(math.radians(x))
        if name == "COS":
            return math.cos(math.radians(x))
        if name == "TAN":
            return _finite(math.tan(math.radians(x)))
        if name in ("ASIN", "ACOS"):
            if not -1.0 <= x <= 1.0:
                raise Alarm("argument_out_of_range", f"{name}({x:g})")
            return math.degrees(math.asin(x) if name == "ASIN" else math.acos(x))
        if name == "ATAN":
            return math.degrees(math.atan(x))
        if name == "SQ":
            return _finite(x * x)
        if name == "SQRT":
            if x < 0.0:
                raise Alarm("argument_out_of_range", f"SQRT({x:g})")
            return math.sqrt(x)
        if name in ("LN", "LOG"):
            if x <= 0.0:
                raise Alarm("argument_out_of_range", f"{name}({x:g})")
            return math.log(x) if name == "LN" else math.log10(x)
        if name == "EXP":
            try:
                return _finite(math.exp(x))
            except OverflowError:
                raise Alarm("overflow_float", f"EXP({x:g})") from None
        if name == "ABS":
            return abs(x)
        if name == "INT":
            return float(math.trunc(x))
        if name == "FRAC":
            return x - math.trunc(x)
        if name == "SGN":
            return 0.0 if x == 0.0 else math.copysign(1.0, x)
        if name == "NEG":
            return -x
        raise Alarm("syntax", f"function {name}")


def _finite(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        raise Alarm("overflow_float", "result out of range")
    return value


__all__ = ["Values", "parse_formula", "parse_string_arguments", "value_node", "q_node", "FormulaError"]
