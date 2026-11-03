"""``twin check --write``: store the resolved L0 strides under ``physics.resolved`` in the project file.

:func:`write_resolved_physics` replaces only the ``resolved`` entry of the ``physics`` section
(adding the section or the entry when absent) and leaves every other line of the file as written,
comments included. The new block is checked against the declared schema before it is written,
the edited text is parsed again and compared with the expected data (the original data with only
``physics.resolved`` changed) before the file is replaced (written to a temporary file, flushed,
fsynced and renamed over the original, so a crash never leaves half a file).

Format of ``physics.resolved`` (docs/formats.md "Project file")::

    physics:
      resolved:
        ipo_ticks: 1000                  # base ticks per L1 tick the strides were chosen for
        macro_ipo: 10
        operations:
          OP10:                          # one entry per tool / spindle-range segment of the operation
            - {tool: 1, gear: 1, l0_ticks: 20, max_rpm: 12000, teeth: 4, rule: tooth}
"""

from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml

from ..core.errors import InputError
from ..db.problems import raise_for
from ..db.schema import Ctx
from ..db.yamlio import TwinLoader, dumps, loads
from .loader import load_project, project_path
from .schema import RESOLVED


def _normalise(resolved: Mapping) -> dict:
    """Plain data (ints for tick counts, str operation names) in a stable key order."""
    out: dict = {}
    for key in ("ipo_ticks", "macro_ipo", "build"):
        if key in resolved and resolved[key] is not None:
            out[key] = int(resolved[key]) if key != "build" else str(resolved[key])
    operations = resolved.get("operations") or {}
    out["operations"] = {}
    for name in operations:
        segments = []
        for segment in operations[name]:
            row = {}
            for key in ("tool", "gear", "path", "l0_ticks", "teeth"):
                if key in segment and segment[key] is not None:
                    row[key] = int(segment[key])
            for key in ("max_rpm", "f_max_Hz"):
                if key in segment and segment[key] is not None:
                    row[key] = float(segment[key])
            if segment.get("rule") is not None:
                row["rule"] = str(segment["rule"])
            unknown = set(segment) - set(row) - {"rule"}
            if unknown:
                raise InputError(f"physics.resolved.operations.{name}: unknown key(s) {', '.join(sorted(unknown))}")
            segments.append(row)
        out["operations"][str(name)] = segments
    return out


def _block(resolved: dict, indent: int) -> str:
    """YAML text of ``resolved: ...`` indented by ``indent`` spaces (block style, flow segments)."""
    lines = [" " * indent + "resolved:"]
    pad = " " * (indent + 2)
    for key in ("ipo_ticks", "macro_ipo", "build"):
        if key in resolved:
            lines.append(f"{pad}{key}: {dumps(resolved[key]).strip().removesuffix('...').strip()}")
    operations = resolved.get("operations") or {}
    if not operations:
        lines.append(f"{pad}operations: {{}}")
    else:
        lines.append(f"{pad}operations:")
        for name, segments in operations.items():
            lines.append(f"{pad}  {yaml.safe_dump(name).strip().removesuffix('...').strip()}:")
            for segment in segments:
                items = ", ".join(f"{k}: {_scalar(v)}" for k, v in segment.items())
                lines.append(f"{pad}    - {{{items}}}")
    return "\n".join(lines) + "\n"


def _scalar(value) -> str:
    if isinstance(value, float):
        text = repr(value)
        return text
    if isinstance(value, str):
        return yaml.safe_dump(value, default_style=None).strip().removesuffix("...").strip()
    return str(value)


def _compose(text: str):
    loader = TwinLoader(text)
    try:
        return loader.get_single_node()
    finally:
        loader.dispose()


def _find(mapping_node, key: str):
    if mapping_node is None or not isinstance(mapping_node, yaml.MappingNode):
        return None, None
    for key_node, value_node in mapping_node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return key_node, value_node
    return None, None


def _value_end_line(value_node, lines: list[str]) -> int:
    """Index of the last line (exclusive end) that belongs to ``value_node``: blank and comment-only
    lines after its last value belong to what follows."""
    end = value_node.end_mark
    stop = min(end.line if end.column == 0 else end.line + 1, len(lines))
    first = value_node.start_mark.line
    while stop - 1 > first and (not lines[stop - 1].strip() or lines[stop - 1].lstrip().startswith("#")):
        stop -= 1
    return stop


def edited_text(text: str, resolved: dict) -> str:
    """The project text with ``physics.resolved`` replaced (or added)."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    root = _compose(text)
    if root is not None and not isinstance(root, yaml.MappingNode):
        raise InputError("a project file is a YAML mapping")
    key_node, physics_node = _find(root, "physics")
    if key_node is None:
        block = "physics:\n" + _block(resolved, 2)
        return "".join(lines) + ("" if not lines or lines[-1].strip() == "" else "") + block
    if isinstance(physics_node, yaml.ScalarNode) and physics_node.value in ("", "~", "null", "Null", "NULL"):
        start = key_node.start_mark.line
        indent = key_node.start_mark.column
        replacement = " " * indent + "physics:\n" + _block(resolved, indent + 2)
        return "".join(lines[:start]) + replacement + "".join(lines[start + 1:])
    if not isinstance(physics_node, yaml.MappingNode):
        raise InputError("physics must be a mapping")
    if physics_node.flow_style:
        start = key_node.start_mark.line
        stop = _value_end_line(physics_node, lines)
        data = loads(text)[0]["physics"] or {}
        data = {k: v for k, v in data.items() if k != "resolved"}
        indent = key_node.start_mark.column
        body = dumps({"physics": data}).splitlines(keepends=True) if data else ["physics:\n"]
        body = [" " * indent + line for line in body]
        return "".join(lines[:start]) + "".join(body) + _block(resolved, indent + 2) + "".join(lines[stop:])
    res_key, res_value = _find(physics_node, "resolved")
    if res_key is not None:
        start = res_key.start_mark.line
        indent = res_key.start_mark.column
        stop = _value_end_line(res_value, lines) if not (isinstance(res_value, yaml.ScalarNode)
                                                          and res_value.start_mark.line == start) else start + 1
        if isinstance(res_value, (yaml.MappingNode, yaml.SequenceNode)) and res_value.flow_style:
            stop = res_value.end_mark.line + 1
        tail = lines[stop:]
        return "".join(lines[:start]) + _block(resolved, indent) + "".join(tail)
    entries = physics_node.value
    first_key = entries[0][0] if entries else None
    indent = first_key.start_mark.column if first_key is not None else key_node.start_mark.column + 2
    stop = _value_end_line(physics_node, lines)
    return "".join(lines[:stop]) + _block(resolved, indent) + "".join(lines[stop:])


def write_resolved_physics(project_path_or_dir, resolved: Mapping) -> Path:
    """Store ``resolved`` as ``physics.resolved`` of the project file; returns the file path."""
    path = project_path(project_path_or_dir).resolve()
    data = _normalise(resolved)
    ctx = Ctx(None, obj="project", file=str(path))
    RESOLVED.check(data, ("physics", "resolved"), None, ctx)
    ipo = data.get("ipo_ticks")
    for name, segments in data.get("operations", {}).items():
        for i, segment in enumerate(segments):
            if isinstance(ipo, int) and ipo % segment["l0_ticks"]:
                ctx.add(("physics", "resolved", "operations", name, i, "l0_ticks"),
                        f"l0_ticks {segment['l0_ticks']} is not a divisor of ipo_ticks {ipo}", "rule")
    raise_for(ctx.problems, "physics.resolved")
    project = load_project(path)
    names = {op.get("name") for op in project.data.get("operations", [])}
    unknown = sorted(set(data.get("operations", {})) - names)
    if unknown:
        raise InputError(f"physics.resolved names operations the project does not have: {', '.join(unknown)}",
                         str(path))
    text = path.read_text(encoding="utf-8")
    new_text = edited_text(text, data)
    expected = copy.deepcopy(loads(text)[0])
    expected.setdefault("physics", {})
    if expected["physics"] is None:
        expected["physics"] = {}
    expected["physics"]["resolved"] = data
    got = loads(new_text)[0]
    if got != expected:
        raise InputError("internal: editing physics.resolved would change other keys; the file was not written",
                         str(path))
    directory = path.parent
    handle, temporary = tempfile.mkstemp(prefix=".project-", suffix=".yaml.tmp", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(new_text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    load_project(path)
    return path


__all__ = ["write_resolved_physics", "edited_text"]
