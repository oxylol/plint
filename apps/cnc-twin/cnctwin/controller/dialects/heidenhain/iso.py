"""HEIDENHAIN TNC 640 ISO (DIN/ISO) program syntax -> the dialect's block form.

``%NAME G71 *`` opens the program (G71 mm, G70 inch); blocks are ``N<n> words *``; the last block is
``N<n> %NAME G71 *``. Words: G codes (several per block), X Y Z A B C (signed values or Q parameters),
I J K (circle centre / pole), R (radius; polar radius in G10-G16), H (polar angle; rotation angle with
G73), F S T M, D00-D18 with P01 P02 P03 (Q parameter functions = FN 0-18), Q formulas (``Q1 = ...``),
cycle parameters ``Q200=2`` after G200..G253, ``G98 L1`` (label), ``L1,2`` (call label 1, 2 repeats),
``%NAME`` (program call), ``G99 T L R`` (tool definition), ``G51 T`` (tool preselection).

G00-G16 are modal (group "motion"); a block without a motion G code continues the active one. The
block form keeps the G codes as written (``iso`` kind) and the handler resolves the modal state when
the block runs (the modal state depends on the path taken through jumps).
"""

from __future__ import annotations

import re

from .formula import FormulaError, q_node, value_node
from .klartext import HBlock, HProgramInfo, ParseProblem, make_block, parse_content, program_name

VALUE = r"[+-]?(?:\d+\.?\d*|\.\d+|Q[LRS]?\d+)"
ISO_MOTION = ("G00", "G01", "G02", "G03", "G05", "G06", "G10", "G11", "G12", "G13", "G15", "G16")
CYCLE_G = {200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 240, 251, 252, 253, 247}


def _g(text: str) -> str:
    return "G" + str(int(text)).zfill(2)


def parse_iso_block(text: str) -> HBlock:
    code = text.strip()
    if code.endswith("*"):
        code = code[:-1].strip()
    if not code:
        return HBlock("empty")
    if re.match(r"^Q[LRS]?\d+\s*=", code) and not re.match(r"^Q\d+=\S+(\s+Q\d+=\S+)*$", code):
        return parse_content(code)
    if code.startswith("PLANE") or code.startswith("FUNCTION") or code.startswith("TOOL") or code.startswith("CYCL"):
        return parse_content(code)
    if code.startswith("%"):
        parts = code.split()
        name = program_name(parts[0][1:])
        units = [p for p in parts[1:] if p in ("G70", "G71")]
        if len(parts) > 1 and all(p in ("G70", "G71") for p in parts[1:]):
            return HBlock("iso_header", {"name": name, "unit": "INCH" if units == ["G70"] else "MM"})
        if len(parts) == 1:
            return HBlock("call_pgm", {"name": name})
        raise ParseProblem("syntax", code)
    if re.search(r"(^|\s)D\d+(\s|$)", code):
        return _d_function(code)
    f: dict = {"g": [], "coords": [], "ijk": [], "keys": {}, "m": [], "params": {}, "label_call": None, "pgm": None}
    tokens = code.split()
    cycle = any(re.match(r"^G2\d\d$", t) for t in tokens)
    mirror = "G28" in tokens
    for token in tokens:
        if mirror and len(token) == 1 and token in "XYZ":
            f["coords"].append((token, None, ("num", 0.0)))
            continue
        param = re.match(r"^Q(\d+)=(" + VALUE + ")$", token)
        if param is not None:
            f["params"][f"Q{int(param.group(1))}"] = value_node(param.group(2))
            continue
        if token.startswith("%"):
            f["pgm"] = program_name(token[1:])
            continue
        if re.match(r"^L\d+,\d+$", token):
            label, rep = token[1:].split(",")
            f["label_call"] = (int(label), int(rep))
            continue
        for word in _split_words(token, cycle):
            keyed = re.match(rf"^([A-Z]+)({VALUE})$", word)
            if keyed is None:
                raise ParseProblem("syntax", word)
            _word(f, keyed.group(1), keyed.group(2))
    return HBlock("iso", f)


def _split_words(token: str, cycle: bool) -> list[str]:
    """Words of one token: ``G01``, ``X+10``, also run together (``G01X+10``); a cycle name is dropped."""
    words = re.findall(rf"[A-Z]+{VALUE}", token)
    if "".join(words) == token:
        return words
    if cycle and token.isalpha():
        return []
    raise ParseProblem("syntax", token)


def _word(f: dict, letter: str, value: str) -> None:
    if letter == "G":
        if value.startswith(("+", "-")) or "." in value or not value.isdigit():
            raise ParseProblem("syntax", f"G{value}")
        f["g"].append(_g(value))
    elif letter == "M":
        f["m"].append(int(float(value)))
    elif letter in "XYZABCUVW" and len(letter) == 1:
        if any(c[0] == letter for c in f["coords"]):
            raise ParseProblem("duplicate_address", f"axis {letter} twice")
        f["coords"].append((letter, None, value_node(value)))
    elif letter in ("I", "J", "K"):
        f["ijk"].append((letter, value_node(value)))
    elif letter in ("F", "S", "T", "R", "H", "L"):
        if letter in f["keys"]:
            raise ParseProblem("duplicate_address", f"{letter} twice")
        f["keys"][letter] = value_node(value)
    else:
        raise ParseProblem("syntax", f"{letter}{value}")


def _d_function(code: str) -> HBlock:
    """D00 Q1 P01 +5 / D01 Q1 P01 +Q2 P02 +3 / D09 P01 +Q1 P02 +5 P03 1 / D14 P01 1000 / D18 Q1 ID20 NR1."""
    tokens = code.split()
    try:
        number = int(next(t[1:] for t in tokens if re.match(r"^D\d+$", t)))
    except StopIteration:
        raise ParseProblem("syntax", code) from None
    target = next((t for t in tokens if q_node(t) is not None and q_node(t)[0] == "q"), None)
    operands = []
    for k, token in enumerate(tokens):
        if re.match(r"^P0?\d$", token) and k + 1 < len(tokens):
            operands.append(tokens[k + 1])
        elif re.match(r"^P0?\d[+-]", token):
            operands.append(token[3:] if token[2].isdigit() else token[2:])
    try:
        if number in (0, 1, 2, 3, 4, 5, 6, 7, 8, 13):
            if target is None:
                raise ParseProblem("syntax", code)
            values = [value_node(op) for op in operands]
            node = q_node(target)
            if number == 0:
                expr = values[0]
            elif number in (5, 6, 7):
                expr = ("fn", {5: "SQRT", 6: "SIN", 7: "COS"}[number], values[0])
            elif number == 8:
                expr = ("fn", "SQRT", ("bin", "+", ("fn", "SQ", values[0]), ("fn", "SQ", values[1])))
            elif number == 13:
                expr = ("ang", values[0], values[1])
            else:
                expr = ("bin", {1: "+", 2: "-", 3: "*", 4: "/"}[number], values[0], values[1])
            return HBlock("formula", {"target": list(node), "expr": expr})
        if number in (9, 10, 11, 12):
            op = {9: "EQU", 10: "NE", 11: "GT", 12: "LT"}[number]
            return HBlock("fn_jump", {"number": number, "a": value_node(operands[0]), "op": op,
                                      "b": value_node(operands[1]), "label": int(float(operands[2]))})
        if number == 14:
            return HBlock("fn14", {"value": value_node(operands[0])})
        if number == 18:
            ident = next(int(t[2:]) for t in tokens if re.match(r"^ID\d+$", t))
            nr = next(int(t[2:]) for t in tokens if re.match(r"^NR\d+$", t))
            idx = next((t[3:] for t in tokens if re.match(r"^IDX", t)), None)
            return HBlock("fn18", {"target": list(q_node(target)), "id": ident, "nr": nr,
                                   "idx": value_node(idx) if idx else None})
    except (IndexError, StopIteration, FormulaError):
        raise ParseProblem("syntax", code) from None
    raise ParseProblem("unsupported_word", f"D{number:02d}")


def parse_iso(name: str, text: str):
    from .klartext import _split_comment

    blocks = []
    info = HProgramInfo(program_name(name), syntax="iso")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for k, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        code, comment = _split_comment(raw.strip())
        code = code.upper()
        number = None
        match = re.match(r"^N(\d+)\s*(.*)$", code)
        if match is not None:
            number = int(match.group(1))
            code = match.group(2)
        error = None
        try:
            payload = parse_iso_block(code)
        except (ParseProblem, FormulaError) as problem:
            error = (getattr(problem, "key", "syntax"), getattr(problem, "detail", str(problem)))
            payload = HBlock("error")
        if payload.kind == "iso_header":
            if not blocks:
                info.name = payload.f["name"] or info.name
                info.unit = payload.f["unit"]
                payload = HBlock("begin", {"name": payload.f["name"], "unit": payload.f["unit"]})
            else:
                payload = HBlock("end_pgm", {"name": payload.f["name"]})
        if payload.kind == "iso" and "G98" in payload.f["g"] and "L" in payload.f["keys"]:
            label_node = payload.f["keys"]["L"]
            if label_node[0] == "num":
                label = int(label_node[1])
                payload = HBlock("lbl", {"label": label})
                if label != 0:
                    if label in info.labels:
                        error = ("label_twice", f"LBL {label}")
                    else:
                        info.labels[label] = len(blocks)
        blocks.append(make_block(k, raw.rstrip(), number, payload, error, comment))
    return info.name, blocks, info


__all__ = ["parse_iso", "parse_iso_block", "ISO_MOTION", "CYCLE_G"]
