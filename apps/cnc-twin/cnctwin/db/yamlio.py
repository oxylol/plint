"""YAML files with line numbers for every key (database and project files).

Scalars follow the YAML 1.2 core schema (the current YAML specification), not PyYAML's
YAML 1.1 default: ``1e-9`` and ``2.195e4`` are floats, ``0755`` is the integer 755, ``1:20`` is
text, ``yes``/``no``/``on``/``off`` are text, dates stay text (``2026-11-03``). Merge keys
(``<<: *anchor``) work as in YAML 1.1. A mapping may not repeat a key.

:func:`load_file` returns the plain data (dicts, lists, scalars) and a :class:`Locations`
map from key paths (``("axes", "X", "travel_mm")``, list items by index) to ``(file, line)``,
so every problem found later can name the file and line it comes from.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from ..core.errors import InputError

_BASE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class TwinLoader(_BASE_LOADER):
    """Safe loader with YAML 1.2 core-schema implicit types (plus merge keys)."""

    yaml_implicit_resolvers: dict = {}


def _core_schema(loader_cls) -> None:
    add = loader_cls.add_implicit_resolver
    add("tag:yaml.org,2002:null", re.compile(r"^(?:~|null|Null|NULL|)$"), ["~", "n", "N", ""])
    add("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
    add("tag:yaml.org,2002:int", re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
        list("-+0123456789"))
    add("tag:yaml.org,2002:float",
        re.compile(r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
                   r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"),
        list("-+0123456789."))
    add("tag:yaml.org,2002:merge", re.compile(r"^(?:<<)$"), ["<"])


_core_schema(TwinLoader)


def _construct_int(loader, node):
    text = loader.construct_scalar(node)
    sign = -1 if text.startswith("-") else 1
    body = text[1:] if text[:1] in "+-" else text
    if body.startswith("0o"):
        return sign * int(body[2:], 8)
    if body.startswith("0x"):
        return sign * int(body[2:], 16)
    return sign * int(body, 10)


def _construct_float(loader, node):
    text = loader.construct_scalar(node).lower()
    if text in (".inf", "+.inf"):
        return float("inf")
    if text == "-.inf":
        return float("-inf")
    if text == ".nan":
        return float("nan")
    return float(text)


TwinLoader.add_constructor("tag:yaml.org,2002:int", _construct_int)
TwinLoader.add_constructor("tag:yaml.org,2002:float", _construct_float)


class Locations:
    """Key path -> (file, line). Lookups fall back to the nearest ancestor path."""

    def __init__(self, file: str = "", entries: dict | None = None):
        self.file = file
        self.entries: dict[tuple, tuple[str, int]] = dict(entries or {})
        if () not in self.entries:
            self.entries[()] = (file, 1)

    def set(self, path: tuple, line: int, file: str | None = None) -> None:
        self.entries[tuple(path)] = (file or self.file, int(line))

    def get(self, path) -> tuple[str, int]:
        path = tuple(path)
        while True:
            found = self.entries.get(path)
            if found is not None:
                return found
            if not path:
                return (self.file, 0)
            path = path[:-1]

    def line(self, path) -> int:
        return self.get(path)[1]

    def file_of(self, path) -> str:
        return self.get(path)[0]

    def has(self, path) -> bool:
        return tuple(path) in self.entries

    def copy(self) -> "Locations":
        return Locations(self.file, self.entries)

    def overlay(self, other: "Locations", prefix: tuple = ()) -> "Locations":
        """A copy where every path ``other`` knows (moved under ``prefix``) wins."""
        merged = self.copy()
        for path, where in other.entries.items():
            if path == () and prefix == ():
                continue
            merged.entries[tuple(prefix) + path] = where
        return merged

    def subtree(self, prefix: tuple) -> "Locations":
        """Locations below ``prefix``, re-rooted at ``()``."""
        prefix = tuple(prefix)
        out = Locations(self.file_of(prefix))
        out.entries[()] = self.get(prefix)
        n = len(prefix)
        for path, where in self.entries.items():
            if len(path) > n and path[:n] == prefix:
                out.entries[path[n:]] = where
        return out

    def relocate(self, old_prefix: tuple, new_prefix: tuple) -> None:
        """Move every path under ``old_prefix`` to ``new_prefix`` (format migrations)."""
        old_prefix, new_prefix = tuple(old_prefix), tuple(new_prefix)
        n = len(old_prefix)
        moved = {}
        for path in list(self.entries):
            if len(path) >= n and path[:n] == old_prefix and path != ():
                moved[new_prefix + path[n:]] = self.entries.pop(path)
        self.entries.update(moved)


@dataclass
class Doc:
    """One parsed YAML file."""

    path: Path
    data: object
    locations: Locations
    sha256: str
    text: str


def loads(text: str, name: str = "<text>") -> tuple[object, Locations]:
    """Parse YAML text; raise :class:`InputError` (with the line) on any syntax problem."""
    locations = Locations(name)
    try:
        loader = TwinLoader(text)
        try:
            node = loader.get_single_node()
            if node is None:
                return None, locations
            data = _build(loader, node, (), locations, name, [])
        finally:
            loader.dispose()
    except yaml.MarkedYAMLError as error:
        mark = error.problem_mark or error.context_mark
        line = mark.line + 1 if mark is not None else 0
        problem = error.problem or str(error)
        if error.context:
            problem = f"{error.context}: {problem}"
        raise InputError(f"YAML: {problem}", name, line) from None
    except yaml.YAMLError as error:
        raise InputError(f"YAML: {error}", name, 0) from None
    return data, locations


def load_file(path) -> Doc:
    """Read and parse one YAML file (UTF-8)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise InputError(f"cannot read file: {error.strerror}", str(path)) from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise InputError(f"not UTF-8 text (byte {error.start})", str(path)) from None
    data, locations = loads(text, str(path))
    return Doc(path, data, locations, hashlib.sha256(raw).hexdigest(), text)


def _build(loader, node, path, locations, name, stack):
    if isinstance(node, ScalarNode):
        return loader.construct_object(node, deep=True)
    if any(node is seen for seen in stack):
        raise InputError("recursive alias (an anchor that contains itself)", name, node.start_mark.line + 1)
    stack.append(node)
    try:
        if isinstance(node, SequenceNode):
            items = []
            for index, child in enumerate(node.value):
                locations.set(path + (index,), child.start_mark.line + 1)
                items.append(_build(loader, child, path + (index,), locations, name, stack))
            return items
        if isinstance(node, MappingNode):
            return _build_mapping(loader, node, path, locations, name, stack)
    finally:
        stack.pop()
    raise InputError(f"unsupported YAML node {type(node).__name__}", name, node.start_mark.line + 1)


def _build_mapping(loader, node, path, locations, name, stack):
    own: list[tuple] = []
    merges: list = []
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            merges.extend(_merge_sources(value_node, name))
        else:
            own.append((key_node, value_node))
    result: dict = {}
    for key_node, value_node in own:
        if not isinstance(key_node, ScalarNode):
            raise InputError("a mapping key must be a plain scalar", name, key_node.start_mark.line + 1)
        key = loader.construct_object(key_node, deep=True)
        if isinstance(key, float) or key is None:
            raise InputError(f"mapping key {key_node.value!r} must be text or an integer",
                             name, key_node.start_mark.line + 1)
        if key in result:
            raise InputError(f"duplicate key {key!r} in this mapping", name, key_node.start_mark.line + 1)
        locations.set(path + (key,), key_node.start_mark.line + 1)
        result[key] = _build(loader, value_node, path + (key,), locations, name, stack)
    depth = len(path)
    for source in merges:
        merged_locations = Locations(name)
        merged = _build(loader, source, path, merged_locations, name, stack)
        for key, value in merged.items():
            if key in result:
                continue
            result[key] = value
            for sub_path, where in merged_locations.entries.items():
                if len(sub_path) > depth and sub_path[:depth + 1] == path + (key,):
                    locations.entries[sub_path] = where
    return result


def _merge_sources(value_node, name):
    if isinstance(value_node, MappingNode):
        return [value_node]
    if isinstance(value_node, SequenceNode) and all(isinstance(n, MappingNode) for n in value_node.value):
        return list(value_node.value)
    raise InputError("a merge key (<<) needs a mapping or a list of mappings", name,
                     value_node.start_mark.line + 1)


def dumps(data) -> str:
    """YAML text of plain data, block style, keys in the given order (for written files)."""
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, default_flow_style=False,
                     width=100)


class _Dumper(yaml.SafeDumper):
    """Block-style dumper without anchors/aliases (written files repeat shared values)."""

    def ignore_aliases(self, data):
        return True


def _represent_float(dumper, value):
    if value != value:
        text = ".nan"
    elif value in (float("inf"), float("-inf")):
        text = ".inf" if value > 0 else "-.inf"
    else:
        text = repr(float(value))
        if "e" in text and "." not in text.split("e")[0]:
            mantissa, exponent = text.split("e")
            text = f"{mantissa}.0e{exponent}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


_Dumper.add_representer(float, _represent_float)


def _represent_list(dumper, value):
    flow = all(not isinstance(v, (dict, list)) for v in value) and len(value) <= 12
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=flow)


_Dumper.add_representer(list, _represent_list)
