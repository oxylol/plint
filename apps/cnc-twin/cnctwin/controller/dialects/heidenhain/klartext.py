"""HEIDENHAIN Klartext program syntax -> immutable blocks (shared block form with the ISO reader).

A Klartext file is one program: ``0 BEGIN PGM NAME MM`` ... ``n END PGM NAME MM``. Every block starts
with its block number; a line ending in ``~`` continues on the next one (cycle definitions with one Q
parameter per line); ``;`` starts a comment (also between a value and ``~``); ``*`` blocks are
structure items. Words are separated by spaces: coordinates ``X+10`` / ``IX-5`` (incremental), polar
``PR`` ``PA`` ``IPR`` ``IPA``, values may be Q parameters (``X+Q1``, ``Y-QL3``).

Each block's payload is an :class:`HBlock` ``(kind, fields)``; the handler executes it. Errors do not
stop parsing: the block carries ``error = (key, detail)`` and the alarm is raised when it is reached.
Labels (``LBL n`` / ``LBL "name"``) are indexed per program.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...program import Block, Program, ProgramStore, Word, source_digest
from .formula import FormulaError, parse_formula, parse_string_arguments, q_node, value_node

AXES = "XYZABCUVW"
COORD = re.compile(r"^(I?)([XYZABCUVW])([+-]?(?:\d+\.?\d*|\.\d+|Q[LRS]?\d+))$")
POLAR = re.compile(r"^(I?)(PR|PA)([+-]?(?:\d+\.?\d*|\.\d+|Q[LRS]?\d+))$")
NUMBERED = re.compile(r"^\s*(\d+)\s*(.*)$")
VALUE = r"[+-]?(?:\d+\.?\d*|\.\d+|Q[LRS]?\d+)"
KEYED = re.compile(rf"^([A-Z][A-Z0-9]*?)({VALUE})$")


class ParseProblem(Exception):
    def __init__(self, key: str, detail: str):
        super().__init__(detail)
        self.key = key
        self.detail = detail


@dataclass(frozen=True)
class HBlock:
    kind: str
    f: dict = field(default_factory=dict)
    label: object = None


@dataclass
class HProgramInfo:
    name: str
    labels: dict = field(default_factory=dict)      # label (int or str) -> block index
    unit: str = "MM"
    syntax: str = "klartext"


class HeidenhainStore(ProgramStore):
    def __init__(self, programs, mains, digest, info: list[HProgramInfo]):
        super().__init__(programs, mains, digest)
        self.info = info


# ====================================================================== words

def words_of(text: str) -> list[str]:
    """Split a block into words; quoted names and bracketed pattern data stay together."""
    out, current, quote, depth = [], [], False, 0
    for char in text:
        if char == '"':
            quote = not quote
        if char == "(" and not quote:
            depth += 1
        if char == ")" and not quote:
            depth -= 1
        if char in " \t" and not quote and depth == 0:
            if current:
                out.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        out.append("".join(current))
    return out


def motion_fields(words: list[str], allow: tuple = ()) -> dict:
    """Coordinates, polar values, radius compensation, feed, M functions, DR and keyed values."""
    f: dict = {"coords": [], "polar": {}, "comp": None, "feed": None, "m": [], "dr": None, "keys": {}}
    seen = set()
    words = list(words)
    k = 0
    while k < len(words):
        word = words[k]
        k += 1
        if word == "M140":
            # M140 MB <distance> | MB MAX [F<feed>]: retract along the tool axis at the end of the block
            f["m"].append(140)
            if k < len(words) and words[k].startswith("MB"):
                value = words[k][2:]
                k += 1
                if not value and k < len(words):
                    value = words[k]
                    k += 1
                if not value:
                    raise ParseProblem("syntax", "M140 MB without a distance")
                f["m140"] = ["MAX"] if value == "MAX" else ["MB", value_node(value)]
                if k < len(words) and words[k].startswith("F") and len(words[k]) > 1 and (
                        words[k][1].isdigit() or words[k][1] in "+.Q"):
                    f["m140_feed"] = value_node(words[k][1:])
                    k += 1
            continue
        match = COORD.match(word)
        if match is not None:
            letter = match.group(2)
            if letter in seen:
                raise ParseProblem("duplicate_address", f"axis {letter} twice")
            seen.add(letter)
            f["coords"].append((letter, match.group(1) == "I", value_node(match.group(3))))
            continue
        match = POLAR.match(word)
        if match is not None:
            f["polar"][match.group(2)] = (match.group(1) == "I", value_node(match.group(3)))
            continue
        if word in ("R0", "RL", "RR", "R+", "R-"):
            f["comp"] = word
            continue
        if word in ("DR+", "DR-"):
            f["dr"] = 1 if word == "DR+" else -1
            continue
        if word in ("FMAX", "FAUTO"):
            f["feed"] = (word,)
            continue
        if word.startswith("M") and word[1:].isdigit():
            f["m"].append(int(word[1:]))
            continue
        if word.startswith(("FU", "FZ")) and len(word) > 2:
            f["feed"] = (word[:2], value_node(word[2:]))
            continue
        if word.startswith("F") and len(word) > 1 and (word[1].isdigit() or word[1] in "+-.Q"):
            f["feed"] = ("F", value_node(word[1:]))
            continue
        if word in ("CLSD+", "CLSD-") and "CLSD" in allow:
            f["keys"]["CLSD"] = value_node("+1" if word.endswith("+") else "-1")
            continue
        match = KEYED.match(word)
        if match is not None and match.group(1) in allow:
            f["keys"][match.group(1)] = value_node(match.group(2))
            continue
        if word in allow:
            f["keys"][word] = None
            continue
        raise ParseProblem("syntax", f"word {word}")
    return f


# ====================================================================== blocks

FK_KEYS = ("AN", "LEN", "PDX", "PDY", "CCX", "CCY", "CCPR", "CCPA", "R", "RX", "RY", "RPR", "RPA", "RAN", "PAR",
           "DP", "IAN", "ICCX", "ICCY", "ICCPR", "ICCPA", "CLSD")
PLANE_KEYS = ("SPA", "SPB", "SPC", "PROPR", "PROMIN", "ROT", "EULPR", "EULNU", "EULROT", "BX", "BY", "BZ", "NX", "NY",
              "NZ", "P1X", "P1Y", "P1Z", "P2X", "P2Y", "P2Z", "P3X", "P3Y", "P3Z", "A", "B", "C", "DIST", "ABST", "F")


def parse_content(text: str) -> HBlock:
    words = words_of(text)
    if not words:
        return HBlock("empty")
    head = words[0]
    rest = words[1:]
    if head.startswith("*"):
        return HBlock("comment")
    if head == "BEGIN" and rest[:1] == ["PGM"]:
        return HBlock("begin", {"name": rest[1] if len(rest) > 1 else "", "unit": rest[2] if len(rest) > 2 else "MM"})
    if head == "END" and rest[:1] == ["PGM"]:
        return HBlock("end_pgm", {"name": rest[1] if len(rest) > 1 else ""})
    if head == "BLK" and rest[:1] == ["FORM"]:
        return _blank(rest[1:])
    if head == "TOOL" and rest[:1] in (["CALL"], ["DEF"]):
        return _tool(rest[0], rest[1:])
    if head in ("L", "C", "CR", "CT", "CC", "LP", "CP", "CTP"):
        f = motion_fields(rest, ("R",) if head in ("CR",) else ())
        return HBlock(head, f)
    if head == "RND":
        f = motion_fields(rest, ("R",))
        if "R" not in f["keys"]:
            raise ParseProblem("syntax", "RND needs R")
        return HBlock("RND", f)
    if head == "CHF":
        if not rest:
            raise ParseProblem("syntax", "CHF needs a length")
        f = motion_fields(rest[1:])
        f["length"] = value_node(rest[0])
        return HBlock("CHF", f)
    if head in ("APPR", "DEP"):
        if not rest:
            raise ParseProblem("syntax", f"{head} needs LT, LN, CT or LCT")
        variant = rest[0]
        if variant not in ("LT", "LN", "CT", "LCT", "PLT", "PLN", "PCT", "PLCT"):
            raise ParseProblem("syntax", f"{head} {variant}")
        f = motion_fields(rest[1:], ("LEN", "CCA", "R"))
        f["variant"] = variant
        return HBlock(head, f)
    if head in ("FL", "FLT", "FC", "FCT"):
        f = motion_fields(rest, FK_KEYS)
        return HBlock("FK", dict(f, element=head))
    if head == "FPOL":
        return HBlock("FPOL", motion_fields(rest))
    if head == "FSELECT":
        return HBlock("FSELECT", {"n": int(rest[0]) if rest else 0})
    if head == "CYCL" and rest[:1] == ["DEF"]:
        return _cycle_def(rest[1:])
    if head == "CYCL" and rest[:1] == ["CALL"]:
        variant = rest[1] if len(rest) > 1 and rest[1] in ("POS", "PAT") else ""
        f = motion_fields(rest[2:] if variant else rest[1:])
        f["variant"] = variant
        return HBlock("cycl_call", f)
    if head == "PATTERN" and rest[:1] == ["DEF"]:
        return _pattern(rest[1:])
    if head == "LBL":
        return HBlock("lbl", {"label": _label(rest)})
    if head == "CALL" and rest[:1] == ["LBL"]:
        label = _label(rest[1:2])
        rep = None
        if len(rest) >= 4 and rest[2] == "REP":
            rep = value_node(rest[3])
        elif len(rest) > 2 and not (len(rest) == 3 and rest[2] == "REP"):
            raise ParseProblem("syntax", " ".join(rest))
        return HBlock("call_lbl", {"label": label, "rep": rep})
    if head == "CALL" and rest[:1] == ["PGM"]:
        if len(rest) < 2:
            raise ParseProblem("syntax", "CALL PGM needs a name")
        return HBlock("call_pgm", {"name": program_name(rest[1])})
    if head == "SEL" and rest[:1] in (["PGM"], ["CYCLE"]):
        if len(rest) < 2:
            raise ParseProblem("syntax", f"SEL {rest[0]} needs a name")
        if q_node(rest[1].strip('"')) is not None:
            raise ParseProblem("unsupported_word", f"SEL {rest[0]} with a QS parameter")
        return HBlock("sel_pgm" if rest[0] == "PGM" else "sel_cycle", {"name": program_name(rest[1])})
    if head == "CALL" and rest == ["SELECTED", "PGM"]:
        return HBlock("call_selected", {})
    if head == "FN":
        return _fn(text)
    if q_node(head.split("=")[0]) is not None or re.match(r"^Q[LRS]?\d+\s*=", text.strip()):
        return _formula(text)
    if head == "PLANE":
        return _plane(rest)
    if head == "FUNCTION":
        return _function(rest)
    if head == "STOP":
        return HBlock("stop", motion_fields(rest))
    if head.startswith("M") and head[1:].isdigit():
        return HBlock("m_only", motion_fields(words))
    if head in ("SEL", "TRANS", "DECLARE", "FN16", "SQL", "SYSCALL", "WAIT", "CALL"):
        raise ParseProblem("unsupported_word", " ".join(words[:2]))
    raise ParseProblem("syntax", head)


def program_name(text: str) -> str:
    name = text.strip('"').replace("\\", "/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return name.split(".", 1)[0].upper()


def _label(words: list[str]):
    if not words:
        raise ParseProblem("syntax", "label missing")
    text = words[0]
    if text.startswith('"'):
        return text.strip('"').upper()
    if text.isdigit():
        return int(text)
    node = q_node(text)
    if node is not None:
        return ["q", node[1], node[2]] if node[0] == "q" else ["qs", node[1]]
    return text.upper()


def _blank(words: list[str]) -> HBlock:
    if not words:
        raise ParseProblem("syntax", "BLK FORM")
    form = words[0]
    if form not in ("0.1", "0.2"):
        raise ParseProblem("unsupported_word", f"BLK FORM {form}")
    axis = None
    rest = words[1:]
    if form == "0.1" and rest and rest[0] in ("X", "Y", "Z"):
        axis = rest[0]
        rest = rest[1:]
    f = motion_fields(rest)
    return HBlock("blk", {"form": form, "axis": axis, "coords": f["coords"]})


def _tool(kind: str, words: list[str]) -> HBlock:
    f: dict = {"tool": None, "axis": None, "keys": {}}
    k = 0
    if words and (words[0].startswith('"') or words[0][:1].isdigit() or q_node(words[0]) is not None):
        token = words[0]
        if token.startswith('"'):
            f["tool"] = ("name", token.strip('"').upper())
        else:
            f["tool"] = value_node(token)
        k = 1
    for word in words[k:]:
        if word in ("X", "Y", "Z"):
            f["axis"] = word
            continue
        match = re.match(rf"^(S|F|DL|DR2|DR|L|R)({VALUE})$", word)
        if match is None:
            raise ParseProblem("syntax", f"TOOL {kind} {word}")
        f["keys"][match.group(1)] = value_node(match.group(2))
    return HBlock("tool_call" if kind == "CALL" else "tool_def", f)


def _cycle_def(words: list[str]) -> HBlock:
    if not words:
        raise ParseProblem("cycle_undefined", "CYCL DEF without a number")
    number_text = words[0]
    match = re.match(r"^(\d+)(?:\.(\d+))?$", number_text)
    if match is None:
        raise ParseProblem("cycle_undefined", f"CYCL DEF {number_text}")
    number = int(match.group(1))
    sub = int(match.group(2)) if match.group(2) is not None else None
    params = {}
    other = []
    for word in words[1:]:
        q = re.match(rf"^Q(\d+)=({VALUE})$", word)
        if q is not None:
            params[f"Q{int(q.group(1))}"] = value_node(q.group(2))
        else:
            other.append(word)
    if sub is None:
        name_words = [w for w in other if not re.match(r"^[A-Z]+[+-]?\d", w)]
        other = [w for w in other if w not in name_words]
    return HBlock("cycl_def", {"number": number, "sub": sub, "params": params, "words": other})


def _pattern(words: list[str]) -> HBlock:
    patterns = []
    joined: list[str] = []
    for word in words:
        if word.startswith("(") and joined and not joined[-1].endswith(")"):
            joined[-1] += word
        else:
            joined.append(word)
    for word in joined:
        match = re.match(r"^(POS|ROW|PAT|FRAME|CIRC|PITCHCIRC)(\d+)\((.*)\)$", word)
        if match is None:
            raise ParseProblem("syntax", f"PATTERN DEF {word}")
        values = {}
        for item in words_of(match.group(3)):
            keyed = re.match(rf"^([A-Z]+)({VALUE})$", item)
            if keyed is None:
                raise ParseProblem("syntax", f"PATTERN DEF {item}")
            name = keyed.group(1)
            if name in ("IX", "IY", "IZ") and match.group(1) != "POS":
                raise ParseProblem("syntax", f"PATTERN DEF {match.group(1)}: incremental {name}")
            values[name] = value_node(keyed.group(2))
        patterns.append((match.group(1), int(match.group(2)), values))
    if not patterns:
        raise ParseProblem("pattern_missing", "PATTERN DEF without points")
    return HBlock("pattern", {"patterns": patterns})


def _formula(text: str) -> HBlock:
    target, _, expression = text.partition("=")
    node = q_node(target.strip())
    if node is None:
        raise ParseProblem("syntax", target)
    expression = expression.strip()
    if node[0] == "qs":
        return _string_formula(node, expression)
    if expression.replace(" ", "") == "SETUNDEFINED":
        return HBlock("formula", {"target": list(node), "undefined": True})
    try:
        return HBlock("formula", {"target": list(node), "expr": parse_formula(expression)})
    except FormulaError as problem:
        raise ParseProblem("syntax", str(problem)) from None


def _string_formula(node, expression: str) -> HBlock:
    """QS assignments: "text", QSm, STRING FORMULA a || b || ... (QS parameters and "text"), TOCHAR (
    DAT+value DECIMALSn ), SUBSTR ( SRC_QSm BEGk LENn )."""
    target = list(node)
    if len(expression) >= 2 and expression.startswith('"') and expression.endswith('"'):
        return HBlock("formula", {"target": target, "string": expression[1:-1]})
    source = q_node(expression)
    if source is not None and source[0] == "qs":
        return HBlock("formula", {"target": target, "copy": source[1]})
    head = expression.split("(", 1)[0].strip()
    try:
        if head.startswith("STRING FORMULA"):
            parts = []
            for part in expression[len("STRING FORMULA"):].split("||"):
                part = part.strip()
                ref = q_node(part)
                if len(part) >= 2 and part.startswith('"') and part.endswith('"'):
                    parts.append(["text", part[1:-1]])
                elif ref is not None and ref[0] == "qs":
                    parts.append(["qs", ref[1]])
                else:
                    raise ParseProblem("syntax", f"STRING FORMULA element {part or 'missing'}")
            return HBlock("formula", {"target": target, "concat": parts})
        if head in ("TOCHAR", "SUBSTR"):
            inner = expression.split("(", 1)[1] if "(" in expression else ""
            if not inner.rstrip().endswith(")"):
                raise ParseProblem("syntax", f"{head} ( ... ) expected")
            args = dict(parse_string_arguments(inner.rstrip()[:-1]))
            needed = ("DAT",) if head == "TOCHAR" else ("SRC", "BEG", "LEN")
            missing = [name for name in needed if name not in args]
            if missing:
                raise ParseProblem("syntax", f"{head} without {' '.join(missing)}")
            return HBlock("formula", {"target": target, head.lower(): args})
    except FormulaError as problem:
        raise ParseProblem("syntax", str(problem)) from None
    if head == "SYSSTR":
        raise ParseProblem("unsupported_word", "SYSSTR system strings")
    raise ParseProblem("syntax", f"string formula {expression}")


FN_COMPARE = {"EQU": 9, "NE": 10, "GT": 11, "LT": 12}


def _fn(text: str) -> HBlock:
    match = re.match(r"^\s*FN\s*(\d+)\s*:\s*(.*)$", text)
    if match is None:
        raise ParseProblem("syntax", text)
    number, body = int(match.group(1)), match.group(2).strip()
    if number in (0, 1, 2, 3, 4, 5, 6, 7, 8, 13):
        target, _, expression = body.partition("=")
        node = q_node(target.strip())
        if node is None or node[0] != "q":
            raise ParseProblem("syntax", f"FN {number}: target {target}")
        expression = expression.strip()
        if number == 0 and expression.replace(" ", "") == "SETUNDEFINED":
            return HBlock("formula", {"target": list(node), "undefined": True})
        parts = words_of(expression)
        try:
            if number in (0, 5, 6, 7):
                if number != 0 and parts[:1] in (["SQRT"], ["SIN"], ["COS"]):
                    parts = parts[1:]
                operands = [value_node(p) for p in parts]
                if len(operands) != 1:
                    raise ParseProblem("syntax", body)
                op = {0: None, 5: "SQRT", 6: "SIN", 7: "COS"}[number]
                expr = operands[0] if op is None else ("fn", op, operands[0])
            else:
                if len(parts) != 3:
                    raise ParseProblem("syntax", body)
                a, word, b = value_node(parts[0]), parts[1], value_node(parts[2])
                expected = {1: "+", 2: "-", 3: "*", 4: "DIV", 8: "LEN", 13: "ANG"}[number]
                if word != expected:
                    raise ParseProblem("syntax", body)
                if number == 4:
                    expr = ("bin", "/", a, b)
                elif number == 8:
                    expr = ("fn", "SQRT", ("bin", "+", ("fn", "SQ", a), ("fn", "SQ", b)))
                elif number == 13:
                    expr = ("ang", a, b)
                else:
                    expr = ("bin", expected, a, b)
        except FormulaError as problem:
            raise ParseProblem("syntax", str(problem)) from None
        return HBlock("formula", {"target": list(node), "expr": expr})
    if number in (9, 10, 11, 12):
        jump = re.match(r"^IF\s+(\S+)\s+(EQU|NE|GT|LT|IS\s+DEFINED|IS\s+UNDEFINED)\s*(\S*)\s*GOTO\s+LBL\s+(.+)$", body)
        if jump is None:
            raise ParseProblem("syntax", body)
        op = jump.group(2).replace(" ", "")
        try:
            a = value_node(jump.group(1))
            b = value_node(jump.group(3)) if jump.group(3) else None
        except FormulaError as problem:
            raise ParseProblem("syntax", str(problem)) from None
        return HBlock("fn_jump", {"number": number, "a": a, "op": op, "b": b, "label": _label([jump.group(4).strip()])})
    if number == 14:
        error = re.match(r"^ERROR\s*=\s*(\S+)$", body)
        if error is None:
            raise ParseProblem("syntax", body)
        return HBlock("fn14", {"value": value_node(error.group(1))})
    if number == 16:
        return HBlock("fn16", {"text": body})
    if number == 17:
        write = re.match(r"^SYSWRITE\s+ID\s*(\d+)\s+NR\s*(\d+)(?:\s+IDX\s*(\S+))?\s*=\s*(\S+)$", body)
        if write is None:
            raise ParseProblem("syntax", body)
        return HBlock("fn17", {"id": int(write.group(1)), "nr": int(write.group(2)),
                               "idx": value_node(write.group(3)) if write.group(3) else None,
                               "value": value_node(write.group(4))})
    if number == 18:
        read = re.match(r"^SYSREAD\s+(\S+)\s*=\s*ID\s*(\d+)\s+NR\s*(\d+)(?:\s+IDX\s*(\S+))?$", body)
        if read is None:
            raise ParseProblem("syntax", body)
        node = q_node(read.group(1))
        if node is None or node[0] != "q":
            raise ParseProblem("syntax", body)
        return HBlock("fn18", {"target": list(node), "id": int(read.group(2)), "nr": int(read.group(3)),
                               "idx": value_node(read.group(4)) if read.group(4) else None})
    if number == 15:
        raise ParseProblem("unsupported_word", "FN 15 PRINT (replaced by FN 16 on the TNC 640)")
    raise ParseProblem("unsupported_word", f"FN {number}")


def _plane(words: list[str]) -> HBlock:
    if not words:
        raise ParseProblem("syntax", "PLANE")
    variant = words[0]
    if variant not in ("SPATIAL", "PROJECTED", "EULER", "VECTOR", "POINTS", "RELATIV", "RELATIVE", "AXIAL", "RESET"):
        raise ParseProblem("syntax", f"PLANE {variant}")
    f: dict = {"variant": "RELATIV" if variant == "RELATIVE" else variant, "values": {}, "position": None,
               "seq": 0, "feed": None, "dist": None}
    for word in words[1:]:
        if word in ("MOVE", "TURN", "STAY"):
            f["position"] = word
            continue
        if word in ("SEQ+", "SEQ-"):
            f["seq"] = 1 if word == "SEQ+" else -1
            continue
        if word in ("FMAX",):
            f["feed"] = ("FMAX",)
            continue
        match = re.match(rf"^(I?)(SPA|SPB|SPC|PROPR|PROMIN|ROT|EULPR|EULNU|EULROT|BX|BY|BZ|NX|NY|NZ|P1X|P1Y|P1Z|P2X|P2Y|"
                         rf"P2Z|P3X|P3Y|P3Z|A|B|C|DIST|ABST|F)({VALUE})$", word)
        if match is None:
            if word in ("TABLE", "COORD", "ROT"):
                raise ParseProblem("unsupported_word", f"PLANE ... {word} ROT")
            raise ParseProblem("syntax", f"PLANE {word}")
        key = match.group(2)
        if key == "F":
            f["feed"] = ("F", value_node(match.group(3)))
        elif key in ("DIST", "ABST"):
            f["dist"] = value_node(match.group(3))
        else:
            f["values"][match.group(1) + key] = value_node(match.group(3))
    if f["position"] is None and variant != "RESET":
        raise ParseProblem("syntax", "PLANE needs MOVE, TURN or STAY")
    return HBlock("plane", f)


def _function(words: list[str]) -> HBlock:
    if words[:1] == ["TCPM"]:
        return HBlock("tcpm", {"on": True, "options": words[1:]})
    if words[:2] == ["RESET", "TCPM"]:
        return HBlock("tcpm", {"on": False, "options": []})
    raise ParseProblem("unsupported_word", "FUNCTION " + " ".join(words[:2]))


# ====================================================================== files

def _split_comment(line: str) -> tuple[str, str]:
    quote = False
    for k, char in enumerate(line):
        if char == '"':
            quote = not quote
        elif char == ";" and not quote:
            return line[:k], line[k + 1:].strip()
    return line, ""


def logical_blocks(text: str) -> list[tuple[int, str, str, str]]:
    """(line, block text as written, code, comment) with ``~`` continuations joined."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    k = 0
    while k < len(lines):
        raw = lines[k].rstrip()
        k += 1
        if not raw.strip():
            continue
        start = k
        written = [raw]
        stripped = raw.strip()
        cont = stripped.endswith("~")
        code, comment = _split_comment(stripped[:-1] if cont else stripped)
        codes, comments = [code], [comment] if comment else []
        while cont and k < len(lines):
            nxt = lines[k].rstrip()
            k += 1
            written.append(nxt)
            piece = nxt.strip()
            cont = piece.endswith("~")
            code, comment = _split_comment(piece[:-1] if cont else piece)
            codes.append(code)
            if comment:
                comments.append(comment)
        out.append((start, "\n".join(written), " ".join(c.strip() for c in codes), " ".join(comments)))
    return out


def parse_klartext(name: str, text: str) -> tuple[str, list[Block], HProgramInfo]:
    blocks: list[Block] = []
    info = HProgramInfo(program_name(name))
    for line, written, code, comment in logical_blocks(text):
        match = NUMBERED.match(code)
        number = None
        content = code
        if match is not None and match.group(2) != "" or (match is not None and code.strip().isdigit()):
            number = int(match.group(1))
            content = match.group(2)
        payload, error = None, None
        try:
            payload = parse_content(_upper(content))
        except ParseProblem as problem:
            error = (problem.key, problem.detail)
            payload = HBlock("error")
        if payload.kind == "begin":
            info.name = program_name(payload.f["name"]) or info.name
            info.unit = payload.f["unit"]
        if payload.kind == "lbl" and error is None:
            label = payload.f["label"]
            if not isinstance(label, list) and label != 0:
                if label in info.labels:
                    error = ("label_twice", f"LBL {label}")
                else:
                    info.labels[label] = len(blocks)
        blocks.append(make_block(line, written, number, payload, error, comment))
    return info.name, blocks, info


def make_block(line: int, written: str, number, payload: HBlock, error, comment: str) -> Block:
    kind = payload.kind
    statement = kind in ("formula", "fn_jump", "lbl", "call_lbl", "fn18", "fn17")
    words = () if statement or kind in ("empty", "comment") else (Word(kind[:1] or ";"),)
    return Block(line, written, number, 0, words, -1, 0, ("hh", kind) if statement else None, comment, error,
                 _terms(payload), None, False, payload)


def _terms(payload: HBlock) -> int:
    expr = payload.f.get("expr") if payload.f else None
    count = 0
    stack = [expr] if expr else []
    while stack:
        node = stack.pop()
        if isinstance(node, tuple):
            count += 1
            stack.extend(n for n in node[1:] if isinstance(n, tuple))
    return count


def _upper(text: str) -> str:
    out, quote = [], False
    for char in text:
        if char == '"':
            quote = not quote
        out.append(char if quote else char.upper())
    return "".join(out)


def build_store(sources, rules=None, *, syntax: str = "klartext"):
    """Parse every source into a HeidenhainStore (Klartext or ISO by ``syntax``)."""
    from .iso import parse_iso

    programs: list[Program] = []
    info: list[HProgramInfo] = []
    mains: dict[int, int] = {}
    problems: list[str] = []
    seen: set[str] = set()
    for source in sources:
        reader = parse_iso if _looks_iso(source.text, syntax) else parse_klartext
        name, blocks, pinfo = reader(source.name, source.text)
        if not blocks:
            problems.append(f"{source.name}: no program in the file")
            continue
        if name in seen:
            problems.append(f"{source.name}: program {name} already in memory (kept the first)")
        seen.add(name)
        if source.main and source.path >= 0 and source.path not in mains:
            mains[source.path] = len(programs)
        last = blocks[-1].line + 1
        blocks.append(Block(last, "", None, 0, (), -1, 0, None, "", None, 0, None, False, HBlock("eof")))
        index: dict[int, list[int]] = {}
        for k, block in enumerate(blocks):
            if block.n is not None:
                index.setdefault(block.n, []).append(k)
        programs.append(Program(None, name, source.name, tuple(blocks), {n: tuple(v) for n, v in index.items()}))
        info.append(pinfo)
    return HeidenhainStore(programs, mains, source_digest(sources), info), problems


def _looks_iso(text: str, syntax: str) -> bool:
    head = text.lstrip()[:200].upper()
    if head.startswith("%"):
        return True
    if "BEGIN PGM" in head:
        return False
    return syntax == "iso"


__all__ = ["HBlock", "HProgramInfo", "HeidenhainStore", "build_store", "parse_content", "parse_klartext",
           "motion_fields", "words_of", "ParseProblem", "program_name", "make_block", "logical_blocks"]
