"""Older key forms migrated forward at load time (the "migrate forward" rule, docs/formats.md).

Database files written against the first templates use key forms that contract v2 forbids
(``growth_um_at_max_rpm``: unit not last; ``pressure_kPa_abs_min``: minimum as a suffix;
Kienzle entries without ``h_ref_mm``; Usui constants without their unit system ...). They are
never accepted as synonyms: :func:`migrate` rewrites a copy of the parsed file into the current
form with the documented rules of :data:`RULES` (each applied once, recorded as a
:class:`~cnctwin.db.problems.Migration` with file, line and both key paths) and the schema then
checks the result. The files themselves are not changed.

Kinds of rule:

* ``rename``: an older key name for a declared field (same value, same unit after conversion);
* ``to_notes``: descriptive text or duplicated catalogue data the schema does not model, moved
  into the ``notes`` of the same mapping as ``key: value`` text (sources were checked as written);
* ``merge`` / ``move``: a value written beside its section (``spindle_probe_extra``) moved into
  it, with the source it had where it was written pinned to it;
* ``default``: a reference value the older template left implicit (``h_ref_mm: 1`` for Kienzle,
  ``T_ref_min: 1`` for Taylor), added with the source of the entry it belongs to;
* schema-driven value forms: ``{range: [lo, hi]}`` without ``value`` on a field that *is* a
  [lo, hi] window becomes ``{value: [lo, hi]}``; ``{text: ..}`` on a text field becomes
  ``{value: ..}``.

A value that cannot be migrated mechanically (a non-zero rotary home under ``home_mm``, a text
where a number is declared ...) is moved to ``notes`` and reported as a ``migration`` warning:
it is not used by the simulator until the file is corrected.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .problems import Migration, Problem, format_path
from .schema import ByType, MapOf, Node, OneKey, Pairs, Qty, Seq, Struct, Tagged, Text, is_number

CITE = ("src", "note", "page", "pages", "digitised", "uncertainty_pct")


# --------------------------------------------------------------------------- rule records

@dataclass(frozen=True)
class Rule:
    id: str
    kinds: tuple[str, ...]
    at: str                     # pattern of the mapping the rule applies to ("" = file root)
    why: str
    action: str = "rename"      # rename | to_notes | custom
    old: str = ""
    new: str = ""
    keys: tuple[str, ...] = ()
    fn: Callable | None = field(default=None, compare=False)
    when: Callable | None = field(default=None, compare=False)   # predicate on the mapping


def _pattern(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    out: list[str] = []
    for part in text.split("."):
        while part.endswith("[]"):
            name = part[:-2]
            if name:
                out.append(name)
            out.append("[]")
            part = ""
        if part:
            out.append(part)
    return tuple(out)


def matches(pattern: tuple[str, ...], path: tuple) -> bool:
    if not pattern:
        return not path
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(matches(rest, path[i:]) for i in range(len(path) + 1))
    if not path:
        return False
    first = path[0]
    if head == "[]":
        ok = isinstance(first, int) and not isinstance(first, bool)
    elif head == "*":
        ok = isinstance(first, str)
    else:
        ok = first == head
    return ok and matches(rest, path[1:])


# --------------------------------------------------------------------------- the engine

class Migrator:
    """Applies the rules to one parsed file (a deep copy) and records what it did."""

    def __init__(self, kind: str, data, locations=None, file: str = "", obj: str = ""):
        self.kind = kind
        self.root = copy.deepcopy(data)
        self.locations = locations.copy() if locations is not None else None
        self.file = file
        self.obj = obj
        self.migrations: list[Migration] = []
        self.problems: list[Problem] = []

    # bookkeeping
    def line(self, path) -> int:
        return self.locations.get(path)[1] if self.locations is not None else 0

    def record(self, path_from, path_to, rule: str, note: str = "") -> None:
        self.migrations.append(Migration(self.file, self.line(path_from), format_path(path_from),
                                         format_path(path_to), rule, note, self.obj))

    def warn(self, path, message: str) -> None:
        self.problems.append(Problem(self.file, self.line(path), format_path(path), message, "migration",
                                     "warning", self.obj))

    def relocate(self, old, new) -> None:
        if self.locations is not None:
            self.locations.relocate(tuple(old), tuple(new))

    # primitives
    def rename(self, mapping: dict, path: tuple, old: str, new: str, rule: str, note: str = "") -> bool:
        if old not in mapping:
            return False
        if new in mapping:
            self.to_notes(mapping, path, old, rule, f"{note}; {new!r} is also given, the older {old!r} is kept as text")
            self.warn(path + (old,), f"{old!r} and its current form {new!r} are both given; {old!r} moved to notes")
            return False
        items = list(mapping.items())
        mapping.clear()
        for key, value in items:
            mapping[new if key == old else key] = value
        self.relocate(path + (old,), path + (new,))
        self.record(path + (old,), path + (new,), rule, note)
        return True

    def to_notes(self, mapping: dict, path: tuple, key: str, rule: str, note: str = "") -> None:
        if key not in mapping:
            return
        value = mapping.pop(key)
        add_note(mapping, f"{key}: {compact(value)}")
        if self.locations is not None and not self.locations.has(path + ("notes",)):
            self.locations.set(path + ("notes",), self.line(path + (key,)))
        self.record(path + (key,), path + ("notes",), rule, note)

    def pin(self, value, citation: dict):
        """``value`` carrying the citation it had where it was written."""
        if not citation:
            return value
        if isinstance(value, Mapping):
            out = dict(value)
            for k, v in citation.items():
                out.setdefault(k, v)
            return out
        out = {"value": value}
        out.update(citation)
        return out

    def citation_at(self, path: tuple) -> dict:
        """src (and its note/page) in force at ``path`` (section inheritance, as written)."""
        node = self.root
        found: dict = {}
        if isinstance(node, Mapping) and isinstance(node.get("src"), str):
            found = {k: node[k] for k in ("src", "note", "page") if k in node}
        for part in path:
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
                node = node[part]
            else:
                break
            if isinstance(node, Mapping) and isinstance(node.get("src"), str):
                found = {k: node[k] for k in ("src", "note", "page") if k in node}
        if found.get("src") != "est":
            found.pop("note", None)
        return found

    def get_path(self, path: tuple):
        node = self.root
        for part in path:
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
                node = node[part]
            else:
                return None
        return node

    # traversal
    def run(self, rules: tuple[Rule, ...], schema: Node | None, item_schema: Node | None = None) -> None:
        active = [(r, _pattern(r.at)) for r in rules if self.kind in r.kinds]
        self._walk(self.root, (), active)
        self._notes_pass(self.root, ())
        if schema is not None:
            self._schema_pass(schema, self.root, ())
        if item_schema is not None and isinstance(self.root, dict) and isinstance(self.root.get("items"), list):
            for i, item in enumerate(self.root["items"]):
                self._schema_pass(item_schema, item, ("items", i))
            if isinstance(self.root.get("defaults"), dict):
                self._schema_pass(item_schema, self.root["defaults"], ("defaults",))

    def _notes_pass(self, node, path):
        """A note written as a sourced value ``{value: text, src: ..}`` is note text naming its source."""
        if isinstance(node, dict):
            for key in ("note", "notes"):
                value = node.get(key)
                if isinstance(value, dict) and isinstance(value.get("value"), str):
                    cite = ", ".join(f"{k} {value[k]}" for k in ("src", "page") if k in value)
                    node[key] = f"{value['value']} ({cite})" if cite else value["value"]
                    self.record(path + (key,), path + (key,), "note-text", "a note is text; its source is "
                                                                         "named in the text")
            for key, value in node.items():
                if key not in ("note", "notes"):
                    self._notes_pass(value, path + (key,))
        elif isinstance(node, list):
            for i, item in enumerate(node):
                self._notes_pass(item, path + (i,))

    def _walk(self, node, path, active):
        if isinstance(node, dict):
            for rule, pattern in active:
                if matches(pattern, path) and (rule.when is None or rule.when(node)):
                    self._apply(rule, node, path)
            for key in list(node.keys()):
                if key in node and key not in ("notes", "note"):
                    self._walk(node[key], path + (key,), active)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                self._walk(item, path + (i,), active)

    def _apply(self, rule: Rule, mapping: dict, path: tuple) -> None:
        if rule.action == "rename":
            self.rename(mapping, path, rule.old, rule.new, rule.id, rule.why)
        elif rule.action == "to_notes":
            for key in rule.keys:
                self.to_notes(mapping, path, key, rule.id, rule.why)
        elif rule.action == "custom":
            rule.fn(self, mapping, path, rule)

    # schema-driven value forms
    def _schema_pass(self, node: Node, value, path):
        if value is None:
            return
        if isinstance(node, Tagged):
            chosen = node.select(value)
            if chosen is not None:
                self._schema_pass(chosen, value, path)
            return
        if isinstance(node, ByType):
            chosen = node.select(value)
            if chosen is not None:
                self._schema_pass(chosen, value, path)
            return
        if isinstance(node, Struct) and isinstance(value, dict):
            for key in list(value.keys()):
                child = node.fields.get(key)
                if child is None or key in ("notes", "note"):
                    continue
                self._fix_value(child, value, key, path)
                if key in value:
                    self._schema_pass(child, value[key], path + (key,))
            return
        if isinstance(node, MapOf) and isinstance(value, dict):
            for key in list(value.keys()):
                if key in CITE or key == "notes":
                    continue
                self._fix_value(node.value, value, key, path)
                self._schema_pass(node.value, value[key], path + (key,))
            return
        if isinstance(node, Seq) and isinstance(value, list):
            for i in range(len(value)):
                self._fix_value(node.item, value, i, path)
                self._schema_pass(node.item, value[i], path + (i,))
            return
        if isinstance(node, OneKey) and isinstance(value, dict):
            for key, child in node.cases.items():
                if key in value:
                    self._schema_pass(child, value[key], path + (key,))

    def _fix_value(self, node: Node, container, key, path):
        value = container[key]
        where = path + (key,)
        if isinstance(node, Qty) and isinstance(value, dict) and "range" in value and "value" not in value \
                and "table" not in value and "range" in node.shapes:
            items = list(value.items())
            value.clear()
            for k, v in items:
                value["value" if k == "range" else k] = v
            self.relocate(where + ("range",), where + ("value",))
            self.record(where + ("range",), where + ("value",), "range-window",
                        "a [lo, hi] window written as range: without value: is the window itself")
            return
        if isinstance(node, Text) and isinstance(value, dict) and "text" in value and "value" not in value:
            items = list(value.items())
            value.clear()
            for k, v in items:
                value["value" if k == "text" else k] = v
            self.relocate(where + ("text",), where + ("value",))
            self.record(where + ("text",), where + ("value",), "text-value", "a sourced text is {value: .., src: ..}")
            return
        if isinstance(node, (Qty, Pairs)) and not isinstance(container, list):
            inner = value.get("value") if isinstance(value, dict) else value
            if isinstance(inner, str):
                self.to_notes(container, path, key, "text-in-number",
                              "text where a number is declared: kept as text, not used")
                self.warn(where, f"{key}: text {inner!r} where a number is declared; moved to notes (not used "
                                 f"until written as a number in its unit)")
                if node.nullable:
                    container[key] = None


def add_note(mapping: dict, text: str) -> None:
    existing = mapping.get("notes")
    if existing is None or existing == "":
        mapping["notes"] = text
    elif isinstance(existing, list):
        existing.append(text)
    else:
        mapping["notes"] = f"{existing}; {text}"


def compact(value) -> str:
    """One-line text of a value as it was written (flow-style YAML-like)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str).replace('"', "")


# --------------------------------------------------------------------------- custom rules

def _kinematic_chains(mig: Migrator, kin: dict, path: tuple, rule: Rule) -> None:
    for key in list(kin.keys()):
        match_w = re.fullmatch(r"workpiece_chain_([a-z][a-z0-9_]*)", key)
        match_t = re.fullmatch(r"([a-z][a-z0-9_]*)_tool_chain", key)
        if not (match_w or match_t):
            continue
        group, name = ("workpiece_chains", match_w.group(1)) if match_w else ("tool_chains", match_t.group(1))
        target = kin.setdefault(group, {})
        if not isinstance(target, dict) or name in target:
            mig.warn(path + (key,), f"{key}: {group}.{name} exists already; not migrated")
            continue
        target[name] = kin.pop(key)
        mig.relocate(path + (key,), path + (group, name))
        if mig.locations is not None and not mig.locations.has(path + (group,)):
            mig.locations.set(path + (group,), mig.line(path + (group, name)))
        mig.record(path + (key,), path + (group, name), rule.id, rule.why)


def _rotary_homes(mig: Migrator, kin: dict, path: tuple, rule: Rule) -> None:
    home = kin.get("home_mm")
    axes = mig.root.get("axes") if isinstance(mig.root, dict) else None
    if not isinstance(home, dict) or not isinstance(axes, dict):
        return
    for name in list(home.keys()):
        axis = axes.get(name)
        kind = axis.get("type") if isinstance(axis, dict) else None
        if kind != "rotary":
            continue
        value = home[name]
        number = value.get("value") if isinstance(value, dict) else value
        if is_number(number) and number == 0:
            del home[name]
            degs = kin.setdefault("home_deg", {})
            degs[name] = value
            mig.relocate(path + ("home_mm", name), path + ("home_deg", name))
            mig.record(path + ("home_mm", name), path + ("home_deg", name), rule.id,
                       "rotary axis home 0 under home_mm moved to home_deg (0 in any unit)")
        else:
            del home[name]
            add_note(kin, f"home_mm.{name}: {compact(value)} (rotary axis written under home_mm; unit unknown)")
            mig.record(path + ("home_mm", name), path + ("notes",), rule.id, "non-zero rotary home under home_mm")
            mig.warn(path + ("home_mm", name), f"rotary axis {name} has home {number!r} under home_mm: degrees or "
                                               f"millimetres cannot be told apart; moved to notes, write it under "
                                               f"home_deg")
    if not home:
        kin["home_mm"] = {}


def _merge_extra(mig: Migrator, parent: dict, path: tuple, rule: Rule) -> None:
    for key in list(parent.keys()):
        if not key.endswith("_extra"):
            continue
        base = key[: -len("_extra")]
        extra = parent[key]
        if not isinstance(extra, dict):
            continue
        target = parent.get(base)
        if target is None:
            target = parent[base] = {}
        if not isinstance(target, dict):
            mig.warn(path + (key,), f"{key}: {base} is not a mapping; not merged")
            continue
        citation = mig.citation_at(path + (key,))
        for k in list(extra.keys()):
            if k in CITE or k == "notes":
                continue
            if k in target:
                mig.warn(path + (key, k), f"{key}.{k}: {base}.{k} is given too; moved to notes")
                add_note(target, f"{k} (from {key}): {compact(extra[k])}")
                continue
            target[k] = mig.pin(extra[k], citation)
            mig.relocate(path + (key, k), path + (base, k))
            mig.record(path + (key, k), path + (base, k), rule.id, rule.why)
        if isinstance(extra.get("notes"), str):
            add_note(target, extra["notes"])
        del parent[key]


def _positioning_flat(mig: Migrator, axis: dict, path: tuple, rule: Rule) -> None:
    for flat, inner in (("positioning_repeatability_arcsec", "repeatability_arcsec"),
                        ("positioning_accuracy_arcsec", "accuracy_arcsec"),
                        ("positioning_repeatability_um", "repeatability_um"),
                        ("positioning_accuracy_um", "accuracy_um")):
        if flat not in axis:
            continue
        group = axis.setdefault("positioning", {})
        if not isinstance(group, dict) or inner in group:
            mig.warn(path + (flat,), f"{flat}: positioning.{inner} is given too; moved to notes")
            mig.to_notes(axis, path, flat, rule.id, rule.why)
            continue
        group[inner] = mig.pin(axis.pop(flat), mig.citation_at(path + (flat,)))
        mig.relocate(path + (flat,), path + ("positioning", inner))
        mig.record(path + (flat,), path + ("positioning", inner), rule.id, rule.why)


def _two_path(mig: Migrator, ctrl: dict, path: tuple, rule: Rule) -> None:
    value = ctrl.get("two_path")
    flag = value.get("value") if isinstance(value, dict) else value
    if not isinstance(flag, bool) or "paths" in ctrl:
        mig.to_notes(ctrl, path, "two_path", rule.id, "two_path is not a boolean (or paths is given)")
        return
    entry = dict(value) if isinstance(value, dict) else {}
    entry["value"] = 2 if flag else 1
    del ctrl["two_path"]
    ctrl["paths"] = entry if len(entry) > 1 else entry["value"]
    mig.relocate(path + ("two_path",), path + ("paths",))
    mig.record(path + ("two_path",), path + ("paths",), rule.id, "two_path: true is paths: 2 (false: 1)")


def _tcpc_dwo(mig: Migrator, ctrl: dict, path: tuple, rule: Rule) -> None:
    value = ctrl.pop("tcpc_dwo")
    five = ctrl.setdefault("five_axis", {})
    if not isinstance(five, dict):
        ctrl["tcpc_dwo"] = value
        mig.to_notes(ctrl, path, "tcpc_dwo", rule.id, "five_axis is not a mapping")
        return
    for key in ("tcpc", "dwo"):
        if key in five:
            mig.warn(path + ("tcpc_dwo",), f"tcpc_dwo: five_axis.{key} is given too; kept")
            continue
        five[key] = copy.deepcopy(value)
        mig.record(path + ("tcpc_dwo",), path + ("five_axis", key), rule.id, rule.why)
    if mig.locations is not None and not mig.locations.has(path + ("five_axis",)):
        mig.locations.set(path + ("five_axis",), mig.line(path + ("tcpc_dwo",)))


def _machine_air(mig: Migrator, coolant: dict, path: tuple, rule: Rule) -> None:
    value = coolant.get("machine_air_l_min")
    power = mig.root.get("power") if isinstance(mig.root, dict) else None
    if isinstance(power, dict) and "air_consumption_l_min" not in power:
        power["air_consumption_l_min"] = mig.pin(coolant.pop("machine_air_l_min"),
                                                 mig.citation_at(path + ("machine_air_l_min",)))
        mig.relocate(path + ("machine_air_l_min",), ("power", "air_consumption_l_min"))
        mig.record(path + ("machine_air_l_min",), ("power", "air_consumption_l_min"), rule.id, rule.why)
        return
    mig.to_notes(coolant, path, "machine_air_l_min", rule.id,
                 "the machine's air consumption is power.air_consumption_l_min (given there already)")
    del value


def _nozzle_geometry(mig: Migrator, nozzle: dict, path: tuple, rule: Rule) -> None:
    geometry = nozzle.get("geometry")
    if not isinstance(geometry, dict):
        return
    citation = mig.citation_at(path + ("geometry",))
    programmable = nozzle.get("programmable")
    for key in list(geometry.keys()):
        if key in CITE or key == "notes":
            continue
        into_prog = key in ("step_deg", "home_aim", "step_time_s") and isinstance(programmable, dict)
        target, target_path = (programmable, path + ("programmable",)) if into_prog else (nozzle, path)
        if key in target:
            mig.warn(path + ("geometry", key), f"geometry.{key}: given on the nozzle too; moved to notes")
            add_note(nozzle, f"geometry.{key}: {compact(geometry[key])}")
            continue
        target[key] = mig.pin(geometry[key], citation)
        mig.relocate(path + ("geometry", key), target_path + (key,))
        mig.record(path + ("geometry", key), target_path + (key,), rule.id, rule.why)
    if isinstance(geometry.get("notes"), str):
        add_note(nozzle, geometry["notes"])
    del nozzle["geometry"]


def _kienzle_refs(mig: Migrator, entry: dict, path: tuple, rule: Rule) -> None:
    for key in ("h_ref_mm", "b_ref_mm"):
        if key not in entry:
            entry[key] = 1
            if mig.locations is not None:
                mig.locations.set(path + (key,), mig.line(path))
            mig.record(path, path + (key,), rule.id, rule.why)


def _taylor_ref(mig: Migrator, entry: dict, path: tuple, rule: Rule) -> None:
    if "T_ref_min" not in entry and "C_m_min" in entry:
        entry["T_ref_min"] = 1
        if mig.locations is not None:
            mig.locations.set(path + ("T_ref_min",), mig.line(path))
        mig.record(path, path + ("T_ref_min",), rule.id, rule.why)


def _usui_in_tool_life(mig: Migrator, material: dict, path: tuple, rule: Rule) -> None:
    life = material.get("tool_life")
    if not isinstance(life, list):
        return
    for i, entry in enumerate(life):
        if not isinstance(entry, dict) or "usui" not in entry:
            continue
        usui = entry.pop("usui")
        citation = mig.citation_at(path + ("tool_life", i, "usui"))
        items = usui if isinstance(usui, list) else [usui]
        target = material.get("tool_life_usui")
        if not isinstance(target, list):
            target = material["tool_life_usui"] = []
        for j, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            moved = mig.pin(item, citation)
            for key in ("tool_class", "coating"):
                if key in entry and key not in moved:
                    moved[key] = entry[key]
            target.append(moved)
            mig.relocate(path + ("tool_life", i, "usui", j) if isinstance(usui, list)
                         else path + ("tool_life", i, "usui"), path + ("tool_life_usui", len(target) - 1))
            mig.record(path + ("tool_life", i, "usui"), path + ("tool_life_usui", len(target) - 1), rule.id,
                       rule.why)


def _usui_plain_a(mig: Migrator, entry: dict, path: tuple, rule: Rule) -> None:
    if "A" not in entry or "A_per_MPa" in entry or "A_per_Pa" in entry:
        return
    mig.rename(entry, path, "A", "A_per_Pa", rule.id, rule.why)
    if "rate_unit" not in entry:
        entry["rate_unit"] = "m/s with sigma_n in Pa, V_s in m/s"
        if mig.locations is not None:
            mig.locations.set(path + ("rate_unit",), mig.line(path + ("A_per_Pa",)))
        mig.record(path + ("A_per_Pa",), path + ("rate_unit",), rule.id, rule.why)


def _holder_letter_lengths(mig: Migrator, item: dict, path: tuple, rule: Rule) -> None:
    for key in list(item.keys()):
        match = re.fullmatch(r"length_([A-Z][0-9]?)_mm", key)
        if not match:
            continue
        letter = match.group(1)
        dims = item.setdefault("dimensions_mm", {})
        if not isinstance(dims, dict) or letter in dims:
            mig.warn(path + (key,), f"{key}: dimensions_mm.{letter} is given too; moved to notes")
            mig.to_notes(item, path, key, rule.id, rule.why)
            continue
        dims[letter] = item.pop(key)
        mig.relocate(path + (key,), path + ("dimensions_mm", letter))
        if mig.locations is not None and not mig.locations.has(path + ("dimensions_mm",)):
            mig.locations.set(path + ("dimensions_mm",), mig.line(path + ("dimensions_mm", letter)))
        mig.record(path + (key,), path + ("dimensions_mm", letter), rule.id, rule.why)


_BODY_LETTERS = {"nose_diameter_A": "nose_diameter", "length_B": "length", "flange_diameter_I": "flange_diameter"}


def _body_letters(mig: Migrator, item: dict, path: tuple, rule: Rule) -> None:
    body = item.get("body_mm")
    if not isinstance(body, dict):
        return
    bpath = path + ("body_mm",)
    for key in list(body.keys()):
        if key in _BODY_LETTERS:
            mig.rename(body, bpath, key, _BODY_LETTERS[key], rule.id,
                       f"catalogue drawing letter {key.rsplit('_', 1)[1]} dropped from the key")
        elif re.fullmatch(r"[A-Z][0-9]?", key):
            mig.to_notes(body, bpath, key, rule.id, "catalogue drawing dimension without a name (mm)")
    if "flange_diameter" in body and "outer_diameter" not in body:
        body["outer_diameter"] = copy.deepcopy(body["flange_diameter"])
        mig.record(bpath + ("flange_diameter",), bpath + ("outer_diameter",), rule.id,
                   "the flange is the largest diameter of the body (its collision outline)")


def _series_defaults(mig: Migrator, series: dict, path: tuple, rule: Rule) -> None:
    for key in rule.keys:
        if key not in series:
            continue
        defaults = series.setdefault("defaults", {})
        if not isinstance(defaults, dict) or key in defaults:
            mig.warn(path + (key,), f"{key}: defaults.{key} is given too; moved to notes")
            mig.to_notes(series, path, key, rule.id, rule.why)
            continue
        defaults[key] = mig.pin(series.pop(key), mig.citation_at(path + (key,)))
        mig.relocate(path + (key,), path + ("defaults", key))
        if mig.locations is not None and not mig.locations.has(path + ("defaults",)):
            mig.locations.set(path + ("defaults",), mig.line(path + ("defaults", key)))
        mig.record(path + (key,), path + ("defaults", key), rule.id, rule.why)


def _centrifugal_general(mig: Migrator, series: dict, path: tuple, rule: Rule) -> None:
    general = series.get("centrifugal_loss_general")
    if not isinstance(general, dict):
        return
    citation = mig.citation_at(path + ("centrifugal_loss_general",))
    if "c" in general:
        defaults = series.setdefault("defaults", {})
        defaults["centrifugal_loss_coeff"] = mig.pin(general.pop("c"), citation)
        mig.relocate(path + ("centrifugal_loss_general", "c"), path + ("defaults", "centrifugal_loss_coeff"))
        mig.record(path + ("centrifugal_loss_general", "c"), path + ("defaults", "centrifugal_loss_coeff"), rule.id,
                   "F_loss = c M omega^2 with M = sum(m_jaw r_cg): the coefficient applies to every chuck of the series")
    rest = {k: v for k, v in general.items() if k not in CITE}
    del series["centrifugal_loss_general"]
    if rest:
        add_note(series, f"centrifugal_loss_general: {compact(rest)}")
        mig.record(path + ("centrifugal_loss_general",), path + ("notes",), rule.id, "formula text kept in notes")


def _soft_jaw_dims(mig: Migrator, item: dict, path: tuple, rule: Rule) -> None:
    for old, new in (("length_mm", "jaw_length_mm"), ("width_mm", "jaw_width_mm"), ("height_mm", "jaw_height_mm")):
        mig.rename(item, path, old, new, rule.id, rule.why)


def _holding_pressure_curve(mig: Migrator, item: dict, path: tuple, rule: Rule) -> None:
    old = "holding_pressure_N_cm2_vs_thickness_mm"
    if old not in item:
        return
    value = item[old]
    if "holding_pressure_N_cm2" in item or not isinstance(value, dict) or "table" not in value:
        mig.warn(path + (old,), f"{old}: not a table or holding_pressure_N_cm2 is given too; moved to notes")
        mig.to_notes(item, path, old, rule.id, rule.why)
        return
    value.setdefault("x", "mm")
    mig.rename(item, path, old, "holding_pressure_N_cm2", rule.id, rule.why)


def _roller_force_at_pressure(mig: Migrator, item: dict, path: tuple, rule: Rule) -> None:
    for key in list(item.keys()):
        match = re.fullmatch(r"clamp_force_per_roller_at_([0-9]+(?:\.[0-9]+)?)bar_kN", key)
        if not match:
            continue
        pressure = float(match.group(1))
        value = item.pop(key)
        number = value.get("value") if isinstance(value, dict) else value
        citation = {k: v for k, v in value.items() if k in CITE} if isinstance(value, dict) else \
            mig.citation_at(path + (key,))
        curve = item.setdefault("clamp_force_per_roller_vs_pressure_kN", {"value": []})
        rows = curve["value"] if isinstance(curve, dict) else curve
        rows.append([int(pressure) if pressure.is_integer() else pressure, number])
        rows.sort(key=lambda r: r[0])
        if isinstance(curve, dict):
            for k, v in citation.items():
                curve.setdefault(k, v)
        mig.relocate(path + (key,), path + ("clamp_force_per_roller_vs_pressure_kN",))
        mig.record(path + (key,), path + ("clamp_force_per_roller_vs_pressure_kN",), rule.id,
                   f"the force at {match.group(1)} bar is a point of the force-pressure curve [[bar, kN], ...]")


def _chiller_text(mig: Migrator, coolant: dict, path: tuple, rule: Rule) -> None:
    chiller = coolant.get("chiller")
    if isinstance(chiller, dict) and "value" in chiller and isinstance(chiller["value"], str):
        items = list(chiller.items())
        chiller.clear()
        for k, v in items:
            chiller["kind" if k == "value" else k] = v
        mig.relocate(path + ("chiller", "value"), path + ("chiller", "kind"))
        mig.record(path + ("chiller", "value"), path + ("chiller", "kind"), rule.id, rule.why)


def _tool_text_note(mig: Migrator, mapping: dict, path: tuple, rule: Rule) -> None:
    """A key whose sourced value is only a text description of a declared struct (calibration_ball ...)."""
    for key in rule.keys:
        value = mapping.get(key)
        inner = value.get("value") if isinstance(value, dict) else value
        if isinstance(inner, str):
            mig.to_notes(mapping, path, key, rule.id, rule.why)


def _pose_vector(value):
    """``[x, y, z, a, b, c]`` (mm, deg) -> ``(pose, ambiguous)``; ambiguous when two or more angles
    are non-zero (the v1 form did not say in which order the rotations apply)."""
    numbers = value.get("value") if isinstance(value, dict) else value
    if not (isinstance(numbers, list) and len(numbers) == 6 and all(is_number(v) for v in numbers)):
        return None, True
    pose = {"translate_mm": list(numbers[:3]), "rotate_deg": list(numbers[3:]), "order": "xyz"}
    if isinstance(value, dict):
        pose.update({k: v for k, v in value.items() if k in CITE})
    return pose, sum(1 for v in numbers[3:] if v != 0) > 1


def _pose_6_vectors(mig: Migrator, mapping: dict, path: tuple, rule: Rule) -> None:
    for key in list(mapping.keys()):
        if not (isinstance(key, str) and key.endswith("_mm_deg") and len(key) > len("_mm_deg")):
            continue
        new = key[: -len("_mm_deg")]
        value = mapping[key]
        if isinstance(value, list) and value and all(isinstance(v, list) for v in value):
            converted = [_pose_vector(v) for v in value]
        else:
            converted = [_pose_vector(value)]
        if any(pose is None for pose, _ in converted) or new in mapping:
            mig.warn(path + (key,), f"{key}: not a [x, y, z, a, b, c] pose (or {new} is given too); moved to notes")
            mig.to_notes(mapping, path, key, rule.id, rule.why)
            continue
        if any(ambiguous for _, ambiguous in converted):
            mig.warn(path + (key,), f"{key}: more than one rotation angle is non-zero and the older 6-vector form "
                                    f"does not say in which order they apply; moved to notes, write {new}: "
                                    f"{{translate_mm, rotate_deg, order}}")
            mig.to_notes(mapping, path, key, rule.id, "rotation order of a 6-vector pose is unknown")
            continue
        poses = [pose for pose, _ in converted]
        items = list(mapping.items())
        mapping.clear()
        for k, v in items:
            if k == key:
                mapping[new] = poses if len(poses) > 1 or isinstance(value, list) and value and \
                    isinstance(value[0], list) else poses[0]
            else:
                mapping[k] = v
        mig.relocate(path + (key,), path + (new,))
        mig.record(path + (key,), path + (new,), rule.id, rule.why)


def _is_soft_jaws(item) -> bool:
    return isinstance(item, Mapping) and item.get("type") == "soft_jaws"


# --------------------------------------------------------------------------- the table

M, MAT, TM, CO, CL = ("machine",), ("material",), ("tool_material",), ("coating",), ("coolant",)
HS, WS, TS = ("holder_series",), ("workholding_series",), ("tool_series",)

ALL_KINDS = ("machine", "material", "tool_material", "coating", "coolant", "controller", "tool_series",
             "holder_series", "workholding_series", "validation_case", "model")

RULES: tuple[Rule, ...] = (
    Rule("pose-6-vector", ALL_KINDS, "**", "a pose is {translate_mm, rotate_deg, order}: <name>_mm_deg "
         "[x, y, z, a, b, c] is <name> with order xyz (mechanical when at most one angle is non-zero)",
         action="custom", fn=_pose_6_vectors, when=lambda m: any(isinstance(k, str) and k.endswith("_mm_deg")
                                                                for k in m)),
    # machine
    Rule("m-spindle-growth", M, "spindles[].thermal", "unit last: growth_um_at_max_rpm is growth_at_max_speed_um",
         old="growth_um_at_max_rpm", new="growth_at_max_speed_um"),
    Rule("m-spindle-bearing-heat", M, "spindles[].thermal",
         "unit last: bearing_heat_W_at_max_rpm is bearing_heat_at_max_speed_W",
         old="bearing_heat_W_at_max_rpm", new="bearing_heat_at_max_speed_W"),
    Rule("m-same-mechanics", M, "", "same_mechanics_as: <id> is extends: <id> (the file's own keys override)",
         old="same_mechanics_as", new="extends"),
    Rule("m-rotation-90", M, "axes.*", "rotation_90deg_s is the 90 deg index time index_90deg_s",
         old="rotation_90deg_s", new="index_90deg_s"),
    Rule("m-drive-power", M, "spindles[]", "drive (text) and drive_kW share a stem: the rating is drive_power_kW",
         old="drive_kW", new="drive_power_kW"),
    Rule("m-guide-bush-switch", M, "guide_bushing", "switchable_to_non_guide_bush is switchable",
         old="switchable_to_non_guide_bush", new="switchable"),
    Rule("m-axis-text", M, "axes.*", "descriptive text kept in notes", action="to_notes",
         keys=("description", "travel_note", "implementation", "indexing")),
    Rule("m-spindle-text", M, "spindles[]", "descriptive or duplicated catalogue data (the curves hold the "
         "ratings) kept in notes", action="to_notes",
         keys=("optional", "max_torque_published", "max_rpm_option", "duty", "tabulated", "motor_kW")),
    Rule("m-probe-alt-trigger", M, "probes.*", "an alternative trigger-force setting (the selected one is "
         "trigger_force_N) kept in notes", action="to_notes", keys=("trigger_force_high_N",)),
    Rule("m-probe-extra", M, "probes", "<probe>_extra merged into <probe> with the source it was written with",
         action="custom", fn=_merge_extra),
    Rule("m-calibration-text", M, "probes", "a calibration ball given only as text kept in notes",
         action="custom", fn=_tool_text_note, keys=("calibration_ball",)),
    Rule("m-coolant-text", M, "coolant", "descriptive text kept in notes", action="to_notes",
         keys=("mist_separator", "pcool")),
    Rule("m-chiller-text", M, "coolant", "a chiller given as {value: text} is {kind: text}", action="custom",
         fn=_chiller_text),
    Rule("m-machine-air", M, "coolant", "the machine's air consumption is power.air_consumption_l_min",
         action="custom", fn=_machine_air, when=lambda m: "machine_air_l_min" in m),
    Rule("m-nozzle-geometry", M, "coolant.nozzles[]", "nozzle geometry written on the nozzle (position_mm, "
         "diameter_mm) and its programmable sweep (step_deg, home_aim)", action="custom", fn=_nozzle_geometry),
    Rule("m-thermal-text", M, "thermal", "descriptive text kept in notes", action="to_notes", keys=("mrzp_drift",)),
    Rule("m-envelope-text", M, "workholding_interface", "descriptive text kept in notes", action="to_notes",
         keys=("envelope_note",)),
    Rule("m-tooling-text", M, "", "tool-post capacity summary (not modelled: the project's tool list holds the "
         "tools) kept in notes", action="to_notes", keys=("tooling",)),
    Rule("m-chains", M, "kinematics", "workpiece_chain_<name> is workpiece_chains.<name>, <name>_tool_chain is "
         "tool_chains.<name>", action="custom", fn=_kinematic_chains),
    Rule("m-rotary-home", M, "kinematics", "rotary axis homes belong in home_deg", action="custom",
         fn=_rotary_homes),
    Rule("m-positioning", M, "axes.*", "positioning_<x> is positioning: {<x>}", action="custom",
         fn=_positioning_flat),
    Rule("m-two-path", M, "controller", "two_path: true is paths: 2", action="custom", fn=_two_path,
         when=lambda m: "two_path" in m),
    Rule("m-tcpc-dwo", M, "controller", "tcpc_dwo is five_axis: {tcpc, dwo}", action="custom", fn=_tcpc_dwo,
         when=lambda m: "tcpc_dwo" in m),
    # material
    Rule("mat-profile", MAT, "residual_stress[]", "v1 profile was [[z/thickness, MPa]]: the unit goes in the key",
         old="profile", new="profile_MPa"),
    Rule("mat-kienzle-ref", MAT, "cutting.kienzle[]", "kc11 is by definition the specific force at h = 1 mm and "
         "b = 1 mm: the v1 entries carry h_ref_mm: 1, b_ref_mm: 1 implicitly", action="custom", fn=_kienzle_refs),
    Rule("mat-taylor-ref", MAT, "tool_life[]", "v1 Taylor constants were for T in minutes: T_ref_min: 1",
         action="custom", fn=_taylor_ref),
    Rule("mat-usui-in-tool-life", MAT, "", "Usui constants under a tool_life entry go to tool_life_usui",
         action="custom", fn=_usui_in_tool_life),
    Rule("mat-usui-plain-a", MAT, "tool_life_usui[]", "v1 Usui A was SI (dW/dt in m/s, sigma_n in Pa, V_s in "
         "m/s): A is A_per_Pa with that rate unit", action="custom", fn=_usui_plain_a),
    # tool material, coating
    Rule("tm-usui-plain-a", TM, "usui[]", "v1 Usui A was SI (dW/dt in m/s, sigma_n in Pa, V_s in m/s): A is "
         "A_per_Pa with that rate unit", action="custom", fn=_usui_plain_a),
    Rule("tm-thermal-shock-dk", TM, "", "temperature differences are _dK (a difference in degC equals one in K)",
         old="thermal_shock_deltaT_degC", new="thermal_shock_dK"),
    Rule("co-text", CO, "", "descriptive text kept in notes", action="to_notes", keys=("process_detail",)),
    # coolant
    Rule("cl-noack-max", CL, "", "minimum/maximum are prefixes", old="noack_loss_pct_max", new="max_noack_loss_pct"),
    Rule("cl-air-min", CL, "", "minimum is a prefix (_min at the end means minutes)", old="air_pressure_bar_min",
         new="min_air_pressure_bar"),
    # holders
    Rule("h-series-text", HS, "", "series-level description kept in notes", action="to_notes",
         keys=("collet", "dimension_key", "feature")),
    Rule("h-axial-float", HS, "items[]", "axial compensation travel of a tapping holder is axial_float_mm",
         old="axial_compensation_mm", new="axial_float_mm"),
    Rule("h-coolant-pressure", HS, "items[]", "the coolant pressure a holder takes is max_coolant_pressure_bar",
         old="coolant_bar", new="max_coolant_pressure_bar"),
    Rule("h-drawing-lengths", HS, "items[]", "catalogue drawing lengths length_<X>_mm are dimensions_mm.<X>",
         action="custom", fn=_holder_letter_lengths),
    # workholding
    Rule("w-series-text", WS, "", "series-level description kept in notes", action="to_notes",
         keys=("basis", "dimension_key", "friction_example", "speed_rule", "system", "usage", "force_vs_torque")),
    Rule("w-series-defaults", WS, "", "series-level item values are defaults every item inherits",
         action="custom", fn=_series_defaults, keys=("jaw_friction", "safety_factor")),
    Rule("w-centrifugal-general", WS, "", "the series' centrifugal-loss coefficient is an item default",
         action="custom", fn=_centrifugal_general),
    Rule("w-min-abs-pressure", WS, "items[]", "minimum is a prefix: min_abs_pressure_kPa",
         old="pressure_kPa_abs_min", new="min_abs_pressure_kPa"),
    Rule("w-max-holding-pressure", WS, "items[]", "maximum is a prefix", old="holding_pressure_max_N_cm2",
         new="max_holding_pressure_N_cm2"),
    Rule("w-inertia", WS, "items[]", "kg_m2 reads as kg/m2 (as kg_m3 is kg/m3): a moment of inertia is kgm2",
         old="moment_of_inertia_kg_m2", new="moment_of_inertia_kgm2"),
    Rule("w-max-torque", WS, "items[]", "a clamp's maximum screw torque is max_input_torque_Nm",
         old="max_torque_Nm", new="max_input_torque_Nm"),
    Rule("w-centrifugal-method", WS, "items[].centrifugal_loss_model", "type is method",
         old="type", new="method"),
    Rule("w-gripping-curve", WS, "items[].centrifugal_loss_model", "unit last: [[rpm, kN]] is gripping_force_curve_kN",
         old="gripping_force_curve", new="gripping_force_curve_kN"),
    Rule("w-body-letters", WS, "items[]", "catalogue drawing letters dropped from body_mm keys", action="custom",
         fn=_body_letters),
    Rule("w-soft-jaw-dims", WS, "items[]", "the dimensions of a soft jaw are jaw_length/width/height_mm",
         action="custom", fn=_soft_jaw_dims, when=_is_soft_jaws),
    Rule("w-holding-curve", WS, "items[]", "holding pressure over part thickness is the table holding_pressure_N_cm2 "
         "with x: mm", action="custom", fn=_holding_pressure_curve),
    Rule("w-roller-force", WS, "items[]", "a roller force at a stated pressure is a point of "
         "clamp_force_per_roller_vs_pressure_kN", action="custom", fn=_roller_force_at_pressure),
)


@dataclass
class Result:
    data: object
    locations: object
    migrations: list[Migration]
    problems: list[Problem]


def migrate(kind: str, data, locations=None, *, file: str = "", obj: str = "", schema: Node | None = None,
            rules: tuple[Rule, ...] = RULES, item_schema: Node | None = None) -> Result:
    """The current form of one parsed file (a copy), with the migrations applied and recorded.
    ``schema`` (and ``item_schema`` for the items of a series file) enables the value-form rules."""
    migrator = Migrator(kind, data, locations, file, obj)
    migrator.run(rules, schema, item_schema)
    return Result(migrator.root, migrator.locations, migrator.migrations, migrator.problems)


def rule_table() -> list[dict]:
    """The migration table (for docs/formats.md and ``twin db check --rules``)."""
    rows = []
    for rule in RULES:
        if rule.action == "rename":
            change = f"{rule.old} -> {rule.new}"
        elif rule.action == "to_notes":
            change = ", ".join(rule.keys) + " -> notes"
        else:
            change = rule.fn.__name__.lstrip("_").replace("_", " ")
        rows.append({"id": rule.id, "kinds": list(rule.kinds), "at": rule.at or "(file root)", "change": change,
                     "why": rule.why})
    rows.append({"id": "range-window", "kinds": ["any"], "at": "any [lo, hi] field", "change": "{range: w} -> {value: w}",
                 "why": "a window written as range: without value: is the window itself"})
    rows.append({"id": "text-value", "kinds": ["any"], "at": "any text field", "change": "{text: t} -> {value: t}",
                 "why": "a sourced text is {value: .., src: ..}"})
    rows.append({"id": "text-in-number", "kinds": ["any"], "at": "any numeric field",
                 "change": "text -> notes (+ warning)", "why": "cannot be migrated mechanically"})
    rows.append({"id": "note-text", "kinds": ["any"], "at": "any note/notes", "change": "{value: t, src: s} -> \"t (src s)\"",
                 "why": "a note is text; its source is named in the text"})
    return rows


__all__ = ["RULES", "Rule", "Result", "migrate", "rule_table", "matches", "add_note", "compact", "Migrator"]
