"""Tool libraries of a project: CSV tool lists and ISO 13399 directories (docs/formats.md "Project file").

``tools.library`` names a CSV file or an ISO 13399 directory/file:

* **CSV** (``*.csv``, UTF-8, comma separated, one header row; ``#`` starts a comment line): each row
  is one ``tools.list`` entry. Columns (:data:`LIST_COLUMNS`, :data:`TOOL_COLUMNS`); empty cells are
  "not given"; an unknown column is an error:

  - list columns: ``number`` (required), ``tool`` (database or library id), ``holder``, ``path``,
    ``spindle``, ``turret``, ``station``, ``stickout_mm``, ``length_offset_mm``, ``radius_offset_mm``,
    ``wear_length_mm``, ``wear_radius_mm``, ``runout_um``, ``runout_phase_deg``, ``tilt_urad``,
    ``life_group``, ``life_limit_min``, ``life_used_min``, ``life_limit_parts``, ``sister_of``,
    ``initial_VB_mm``, ``comment``;
  - tool columns (an inline tool: with ``tool`` empty a complete item, with ``tool`` given an
    ``extends`` of it): ``type``, ``hand``, ``tool_material``, ``tool_material_class``, ``grade``,
    ``coating``, ``coating_thickness_um``, ``order_code``, ``maker``, ``diameter_mm``, ``flutes``,
    ``helix_deg``, ``rake_deg``, ``corner_radius_mm``, ``cutting_length_mm``, ``flute_length_mm``,
    ``overall_length_mm``, ``shank_diameter_mm``, ``shank_length_mm``, ``neck_diameter_mm``,
    ``neck_length_mm``, ``edge_radius_um``, ``point_angle_deg``, ``tip_angle_deg``,
    ``thread_pitch_mm``, ``centre_cutting``, ``max_rpm``, ``mass_g``.

* **ISO 13399** (a ``.p21``/``.stp``/``.step`` file or a directory of them): tool items by id
  (:mod:`cnctwin.project.iso13399`) that ``tools.list[i].tool`` may name (or ``extend`` inline, to
  add what ISO 13399 does not carry: ``type`` when it cannot be inferred, ``tool_material.id``,
  ``coating``).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ..core.errors import InputError
from ..db.problems import Problem
from . import iso13399

LIST_COLUMNS = {
    "number": ("number", "int"), "tool": ("tool", "text"), "holder": ("holder", "text"), "path": ("path", "int"),
    "spindle": ("spindle", "text"), "turret": ("turret", "text"), "station": ("station", "int"),
    "stickout_mm": ("stickout_mm", "num"), "length_offset_mm": ("length_offset_mm", "num"),
    "radius_offset_mm": ("radius_offset_mm", "num"), "wear_length_mm": ("wear_offsets_mm.length", "num"),
    "wear_radius_mm": ("wear_offsets_mm.radius", "num"), "runout_um": ("runout_um", "num"),
    "runout_phase_deg": ("runout_phase_deg", "num"), "tilt_urad": ("tilt_urad", "num"),
    "life_group": ("life.group", "int"), "life_limit_min": ("life.limit_min", "num"),
    "life_used_min": ("life.used_min", "num"), "life_limit_parts": ("life.limit_parts", "int"),
    "sister_of": ("life.sister_of", "int"), "initial_VB_mm": ("initial_wear.VB_mm", "num"),
    "comment": ("comment", "text"),
}
TOOL_COLUMNS = {
    "type": ("type", "text"), "hand": ("hand", "text"), "tool_material": ("tool_material.id", "text"),
    "tool_material_class": ("tool_material.class", "text"), "grade": ("tool_material.grade", "text"),
    "coating": ("coating.id", "text"), "coating_thickness_um": ("coating.thickness_um", "num"),
    "order_code": ("order_code", "text"), "maker": ("maker", "text"),
    "diameter_mm": ("geometry.diameter_mm", "num"), "flutes": ("geometry.flutes", "int"),
    "helix_deg": ("geometry.helix_deg", "num"), "rake_deg": ("geometry.rake_deg", "num"),
    "corner_radius_mm": ("geometry.corner_radius_mm", "num"),
    "cutting_length_mm": ("geometry.cutting_length_mm", "num"),
    "flute_length_mm": ("geometry.flute_length_mm", "num"),
    "overall_length_mm": ("geometry.overall_length_mm", "num"),
    "shank_diameter_mm": ("geometry.shank.diameter_mm", "num"),
    "shank_length_mm": ("geometry.shank.length_mm", "num"),
    "neck_diameter_mm": ("geometry.neck.diameter_mm", "num"),
    "neck_length_mm": ("geometry.neck.length_mm", "num"),
    "edge_radius_um": ("geometry.edge_radius_um", "num"), "point_angle_deg": ("geometry.point_angle_deg", "num"),
    "tip_angle_deg": ("geometry.tip_angle_deg", "num"), "thread_pitch_mm": ("geometry.thread.pitch_mm", "num"),
    "centre_cutting": ("geometry.centre_cutting", "bool"), "max_rpm": ("max_rpm", "num"),
    "mass_g": ("mass_g", "num"),
}
TRUE_WORDS = ("true", "yes", "1", "y")
FALSE_WORDS = ("false", "no", "0", "n")


def _set(target: dict, path: str, value) -> None:
    node = target
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _cell(text: str, kind: str, column: str, file: str, line: int):
    text = text.strip()
    try:
        if kind == "int":
            number = float(text)
            if not number.is_integer():
                raise ValueError
            return int(number)
        if kind == "num":
            return float(text)
        if kind == "bool":
            lowered = text.lower()
            if lowered in TRUE_WORDS:
                return True
            if lowered in FALSE_WORDS:
                return False
            raise ValueError
    except ValueError:
        expected = {"int": "an integer", "num": "a number", "bool": "true or false"}[kind]
        raise InputError(f"column {column}: {text!r} is not {expected}", file, line) from None
    return text


def read_csv(path, *, label: str | None = None) -> tuple[list[dict], dict[int, int], list[Problem]]:
    """``(entries, lines, problems)``: tools.list entries of a CSV tool list, the CSV line of each
    entry (by index), and the problems found (unknown columns, bad cells)."""
    path = Path(path)
    file = label or str(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise InputError(f"cannot read the tool list: {error}", file) from None
    rows = [(i + 1, line) for i, line in enumerate(text.splitlines()) if line.strip() and not line.lstrip().startswith("#")]
    problems: list[Problem] = []
    if not rows:
        return [], {}, [Problem(file, 1, "", "the tool list has no header row", "format")]
    header_line, header_text = rows[0]
    header = [h.strip() for h in next(csv.reader(io.StringIO(header_text)))]
    known = {**LIST_COLUMNS, **TOOL_COLUMNS}
    for column in header:
        if column not in known:
            problems.append(Problem(file, header_line, column, f"unknown column {column!r} (columns: "
                                    f"{', '.join(sorted(known))})", "unknown"))
    if "number" not in header:
        problems.append(Problem(file, header_line, "number", "the tool list needs a number column", "required"))
    if len(set(header)) != len(header):
        problems.append(Problem(file, header_line, "", "a column is named twice", "duplicate"))
    if problems:
        return [], {}, problems
    entries: list[dict] = []
    lines: dict[int, int] = {}
    for line, row_text in rows[1:]:
        cells = next(csv.reader(io.StringIO(row_text)))
        if len(cells) > len(header):
            problems.append(Problem(file, line, "", f"{len(cells)} cells for {len(header)} columns", "format"))
            continue
        entry: dict = {}
        tool: dict = {}
        try:
            for column, cell in zip(header, cells):
                if not cell.strip():
                    continue
                if column in LIST_COLUMNS:
                    target, kind = LIST_COLUMNS[column]
                    _set(entry, target, _cell(cell, kind, column, file, line))
                else:
                    target, kind = TOOL_COLUMNS[column]
                    _set(tool, target, _cell(cell, kind, column, file, line))
        except InputError as error:
            problems.append(Problem(file, line, "", str(error).split(": ", 2)[-1], "type"))
            continue
        if "number" not in entry:
            problems.append(Problem(file, line, "number", "a row needs its tool number", "required"))
            continue
        if tool:
            base = entry.get("tool")
            if base:
                tool["extends"] = base
            tool.setdefault("id", f"{path.stem}-t{entry['number']}")
            entry["tool"] = tool
        elif "tool" not in entry:
            problems.append(Problem(file, line, "tool", "a row needs a tool id or the tool columns of an inline "
                                    "tool", "required"))
            continue
        lines[len(entries)] = line
        entries.append(entry)
    return entries, lines, problems


def read_library(path, project_dir: Path | None = None) -> dict:
    """``{"kind": "csv"|"iso13399", "entries": [...], "items": {id: item}, "lines", "problems", "files"}``."""
    path = Path(path)
    if project_dir is not None and not path.is_absolute():
        path = project_dir / path
    if path.is_file() and path.suffix.lower() == ".csv":
        entries, lines, problems = read_csv(path)
        return {"kind": "csv", "entries": entries, "items": {}, "lines": lines, "problems": problems,
                "files": [path]}
    if path.is_dir() or path.suffix.lower() in iso13399.P21_SUFFIXES:
        if not path.exists():
            raise InputError("the tool library does not exist", str(path))
        items = iso13399.read_items(path)
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in iso13399.P21_SUFFIXES) \
            if path.is_dir() else [path]
        by_id: dict[str, dict] = {}
        problems = []
        for item in items:
            if item["id"] in by_id:
                problems.append(Problem(str(path), 0, item["id"], f"tool id {item['id']!r} is defined by two ISO 13399 "
                                        "files", "duplicate"))
                continue
            by_id[item["id"]] = item
        return {"kind": "iso13399", "entries": [], "items": by_id, "lines": {}, "problems": problems, "files": files}
    raise InputError("a tool library is a .csv file or an ISO 13399 file / directory", str(path))


__all__ = ["read_csv", "read_library", "LIST_COLUMNS", "TOOL_COLUMNS"]
