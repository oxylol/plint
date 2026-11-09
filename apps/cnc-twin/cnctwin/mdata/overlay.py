"""The calibration overlay: a YAML file of calibrated values the project includes (``calibration: path``).

Format ``cnctwin-calibration`` version 1 (docs/models/calibration.md "Overlay")::

    format: cnctwin-calibration
    version: 1
    name: "spindle, haas-vf2ss, log of 2026-11-03"
    machine: haas-vf2ss                    # the machine (and material) the values belong to
    material: al6061-t6
    inputs: {machine_data: {...}, run: {...}, project: {...}}
    alignment: {offset_s, scale, method}
    set:                                   # target path -> value in the unit of the path's last key
      machine.spindles[main].idle_power_W: {value: 612.4, parameter: spindle_idle_power_W, ci: [598.1, 626.7]}
      machine.controller.block_processing_ms: {value: 1.42, parameter: block_processing_s}
      material.cutting.kienzle[tool_class=carbide,coating=any].kc11_N_mm2: {value: 845.0, parameter: kc_scale}
      tools[1].runout_um: {value: 7.1, parameter: tool_runout_m}
      models.production-energy.axis_seal_drag_N: {value: 41.2, parameter: axis_seal_drag_N}
      factors.tool_stiffness[1]: {value: 0.86, parameter: tool_stiffness_factor}
    parameters: [...]                      # the fit report of every parameter (prior, value, interval, flags)
    fit: {...}

Target paths: ``<root>.<key>[<selector>].<key>...`` with root ``machine`` / ``material`` (the
resolved database objects), ``tools[<T number>]`` (the project's tool list entry), ``models.<model>``
(a db/models parameter, the name carrying its SI unit, the value SI) or ``factors.<name>[<key>]``
(multipliers with no database key: read by the domain named in the doc). A selector is an index
(``[0]``), ``key=value`` pairs (``[tool_class=carbide,coating=any]``) or a bare text matched
against the item's ``id`` (``[main]``). A ``key=value`` selector that matches no item creates one
with those keys (a new Usui entry).

Applying (:func:`apply_to_raw`, :func:`controller_overrides`, :func:`model_overrides`,
:func:`factors`, :func:`apply_to_tools`, :func:`apply_to_project`): every value lands as written in the overlay; a
database value becomes ``{value, src: est, note: "calibrated (<overlay name>): <parameter> ..."}``
(a calibrated value is a statistical estimate; the note names its interval). Entries under
``machine.controller`` win over any override the project already has.

Two ways to run a calibrated project:

* the project includes the overlay (``calibration: calibration.yaml`` in project.yaml): cnctwin.project.resolve
  applies it at resolve time (:func:`apply_to_tools` to the tool list, the project's ``controller_overrides`` and
  then :func:`apply_to_raw` to the machine, :func:`apply_to_raw` to the material) and hashes the file as an input;
  ``models.*`` and ``factors.*`` entries are listed in ``project.calibration`` for the domains that read them;
* :func:`calibrated_project` writes a standalone project file (``project.calibrated.yaml`` of ``twin calibrate``)
  with the overlay applied through the existing project syntax (inline ``extends`` objects, absolute paths); a
  project that already includes an overlay is written with both baked in (:func:`combined`) and without the key.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..core.errors import InputError
from ..db import keyunits

FORMAT = "cnctwin-calibration"
VERSION = 1
ROOTS = ("machine", "material", "tools", "models", "factors")
_SEGMENT = re.compile(r"([A-Za-z0-9_+-]+)((?:\[[^\]]*\])?)")


@dataclass(frozen=True)
class Target:
    """Where a parameter's value goes: ``path`` (unit = its last key's unit), ``mode`` set or scale
    (``base`` x value: a multiplier on a file value, kc_scale on kc11)."""

    path: str
    mode: str = "set"
    base: float | None = None

    def file_value(self, si_value: float):
        if self.mode == "scale":
            if self.base is None:
                raise InputError(f"target {self.path}: a scale target needs base")
            return float(self.base) * float(si_value)
        if self.path.startswith(("models.", "factors.")):
            return float(si_value)
        _, unit = keyunits.parse_key(parse_path(self.path)[-1][0])
        if unit is None or unit.scale:
            return float(si_value)
        return unit.from_si(float(si_value))

    def to_dict(self) -> dict:
        out = {"path": self.path, "mode": self.mode}
        if self.base is not None:
            out["base"] = self.base
        return out


def parse_path(path: str) -> list[tuple[str, str | None]]:
    """[(key, selector or None)] of a target path; raises InputError when it is not one."""
    text = str(path).strip()
    if text.startswith("models."):
        parts = text.split(".", 2)
        if len(parts) != 3 or not parts[1] or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", parts[2]):
            raise InputError(f"target {path!r}: write models.<model>.<parameter name>")
        return [("models", None), (parts[1], None), (parts[2], None)]
    segments = []
    position = 0
    while position < len(text):
        match = _SEGMENT.match(text, position)
        if not match:
            raise InputError(f"target {path!r}: cannot read it at {text[position:]!r}")
        key, selector = match.group(1), match.group(2)
        segments.append((key, selector[1:-1] if selector else None))
        position = match.end()
        if position < len(text):
            if text[position] != ".":
                raise InputError(f"target {path!r}: expected '.' at {text[position:]!r}")
            position += 1
    if not segments or segments[0][0] not in ROOTS:
        raise InputError(f"target {path!r}: the root is one of {', '.join(ROOTS)}")
    return segments


def _select(items: list, selector: str, path: str, create: bool):
    if not isinstance(items, list):
        raise InputError(f"target {path}: [{selector}] needs a list")
    if re.fullmatch(r"-?\d+", selector):
        index = int(selector)
        if not -len(items) <= index < len(items):
            raise InputError(f"target {path}: index {index} is outside the list of {len(items)}")
        return items[index]
    if "=" in selector:
        wanted = {}
        for part in selector.split(","):
            key, _, value = part.partition("=")
            wanted[key.strip()] = value.strip()
        found = [item for item in items if isinstance(item, dict)
                 and all(str(_plain_value(item.get(k))) == v for k, v in wanted.items())]
    else:
        found = [item for item in items if isinstance(item, dict) and str(_plain_value(item.get("id"))) == selector]
        wanted = {"id": selector}
    if len(found) > 1:
        raise InputError(f"target {path}: [{selector}] matches {len(found)} items (make the selector unique)")
    if found:
        return found[0]
    if not create:
        raise InputError(f"target {path}: no item matches [{selector}]")
    item = dict(wanted)
    items.append(item)
    return item


def _plain_value(value):
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def set_path(document: dict, segments: list[tuple[str, str | None]], value, *, sourced: bool, note: str,
             path: str) -> None:
    """Set the value at ``segments`` (below the root) of a plain mapping, creating mappings as needed."""
    node = document
    for depth, (key, selector) in enumerate(segments):
        last = depth == len(segments) - 1
        if selector is None and last:
            old = node.get(key) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                raise InputError(f"target {path}: {key} is not inside a mapping")
            if sourced and isinstance(value, (int, float)) and not isinstance(value, bool):
                entry = dict(old) if isinstance(old, dict) and "value" in old else {}
                entry.update({"value": value, "src": "est", "note": note})
                for drop in ("range", "page", "digitised", "uncertainty_pct"):
                    entry.pop(drop, None)
                node[key] = entry
            else:
                node[key] = value
            return
        if not isinstance(node, dict):
            raise InputError(f"target {path}: {key} is not inside a mapping")
        child = node.get(key)
        if selector is not None:
            if child is None:
                child = []
                node[key] = child
            item = _select(child, selector, path, create="=" in selector)
            if last:
                raise InputError(f"target {path}: the path ends in a list item, not a value")
            node = item
        else:
            if child is None:
                child = {}
                node[key] = child
            if isinstance(child, dict) and "value" in child and not last:
                raise InputError(f"target {path}: {key} is a value, not a section")
            node = child


# ----------------------------------------------------------------------------- the document

def make_overlay(*, name: str, machine: str | None, material: str | None, entries: dict[str, dict],
                 parameters: list[dict], fit: dict | None = None, inputs: dict | None = None,
                 alignment: dict | None = None, notes: str = "") -> dict:
    document = {"format": FORMAT, "version": VERSION, "name": name, "machine": machine, "material": material,
                "inputs": inputs or {}, "alignment": alignment or {}, "set": dict(sorted(entries.items())),
                "parameters": parameters, "fit": fit or {}}
    if notes:
        document["notes"] = notes
    check_overlay(document)
    return document


def check_overlay(document: dict, where: str = "") -> dict:
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise InputError(f"not a {FORMAT} file", where)
    if int(document.get("version") or 0) > VERSION:
        raise InputError(f"{FORMAT} version {document.get('version')} is newer than this build reads ({VERSION})", where)
    entries = document.get("set")
    if not isinstance(entries, dict):
        raise InputError("set: expected a mapping of target paths", where)
    for path, entry in entries.items():
        parse_path(path)
        if not isinstance(entry, dict) or "value" not in entry:
            raise InputError(f"set.{path}: expected {{value: ...}}", where)
        value = entry["value"]
        if isinstance(value, float) and not math.isfinite(value):
            raise InputError(f"set.{path}: the value is not finite", where)
        if not isinstance(value, (int, float, str)) or isinstance(value, bool):
            raise InputError(f"set.{path}: the value must be a number or text", where)
    return document


def _yaml_plain(value):
    if isinstance(value, dict):
        return {str(k): _yaml_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_plain(v) for v in value]
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        value = value.item()
    if isinstance(value, float):
        return float(f"{value:.12g}") if math.isfinite(value) else None
    return value


def write_overlay(path, document: dict) -> Path:
    check_overlay(document)
    path = Path(path)
    header = ("# CNC Twin calibration overlay (docs/models/calibration.md). Include it from project.yaml with\n"
              "# `calibration: <this file>` (a path relative to the project). Values under `set:` are in the unit\n"
              "# of the last key of their path.\n")
    text = header + yaml.safe_dump(_yaml_plain(document), sort_keys=False, allow_unicode=True, width=110)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load_overlay(path) -> dict:
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InputError(f"cannot read: {error.strerror}", str(path)) from None
    except yaml.YAMLError as error:
        raise InputError(f"not YAML: {error}", str(path)) from None
    return check_overlay(document, str(path))


# ----------------------------------------------------------------------------- applying

def _note(document: dict, entry: dict) -> str:
    parts = [f"calibrated ({entry.get('overlay') or document.get('name') or 'calibration overlay'})"]
    if entry.get("parameter"):
        parts.append(f"parameter {entry['parameter']}")
    if entry.get("ci"):
        lo, hi = entry["ci"]
        parts.append(f"interval [{lo:.6g}, {hi:.6g}]")
    return ", ".join(parts)


def entries_for(document: dict, root: str) -> list[tuple[str, list, dict]]:
    out = []
    for path, entry in sorted(document.get("set", {}).items()):
        segments = parse_path(path)
        if segments[0][0] == root:
            out.append((path, segments, entry))
    return out


def apply_to_raw(kind: str, raw: dict, document: dict) -> dict:
    """A copy of a database object's raw data (machine or material) with the overlay's entries set."""
    if kind not in ("machine", "material"):
        raise InputError(f"overlay entries apply to machine or material objects, not {kind}")
    out = copy.deepcopy(dict(raw))
    owner = document.get(kind)
    if owner and out.get("id") and owner != out.get("id"):
        raise InputError(f"the calibration overlay is for {kind} {owner!r}, the project uses {out.get('id')!r}")
    for path, segments, entry in entries_for(document, kind):
        set_path(out, segments[1:], entry["value"], sourced=True, note=_note(document, entry), path=path)
    return out


def controller_overrides(document: dict) -> dict:
    """The ``machine.controller.*`` entries as a controller_overrides mapping (file units)."""
    out: dict = {}
    for path, segments, entry in entries_for(document, "machine"):
        if len(segments) >= 3 and segments[1] == ("controller", None):
            set_path(out, segments[2:], entry["value"], sourced=False, note="", path=path)
    return out


def model_overrides(document: dict) -> dict[str, dict[str, float]]:
    """{model: {parameter: SI value}} of the ``models.*`` entries (cnctwin.db.models.params names)."""
    out: dict[str, dict[str, float]] = {}
    for path, segments, entry in entries_for(document, "models"):
        out.setdefault(segments[1][0], {})[segments[2][0]] = entry["value"]
    return out


def factors(document: dict) -> dict[str, dict[str, float]]:
    """{factor name: {key: value}} of the ``factors.*`` entries (``factors.tool_stiffness[1]``)."""
    out: dict[str, dict[str, float]] = {}
    for path, segments, entry in entries_for(document, "factors"):
        if len(segments) != 2 or segments[1][1] is None:
            raise InputError(f"target {path}: write factors.<name>[<key>]")
        out.setdefault(segments[1][0], {})[segments[1][1]] = entry["value"]
    return out


def _tool_path(value) -> int:
    """The controller path of a tool list entry or selector (``path`` absent: 1, as the project file defaults it)."""
    try:
        return int(value) if value not in (None, "") else 1
    except (TypeError, ValueError):
        return -1


def apply_to_tools(project: dict, document: dict) -> dict:
    """A copy of a project document with the ``tools[<T number>[@<path>]].<key>`` entries set in its tool list (file
    units, as written in the overlay). A selector names one entry by its number and controller path (default 1)."""
    out = copy.deepcopy(project)
    tools = (out.get("tools") or {}).get("list") or []
    for path, segments, entry in entries_for(document, "tools"):
        selector = segments[0][1]
        if selector is None:
            raise InputError(f"target {path}: write tools[<T number>].<key>")
        number, _, tool_path = selector.partition("@")
        matches = [t for t in tools if isinstance(t, dict) and str(t.get("number")) == number.strip()
                   and _tool_path(t.get("path")) == _tool_path(tool_path.strip())]
        if len(matches) != 1:
            raise InputError(f"target {path}: the project has {len(matches)} tools numbered {number}"
                             + (f" on path {tool_path.strip()}" if tool_path.strip() else ""))
        set_path(matches[0], segments[1:], entry["value"], sourced=False, note="", path=path)
    return out


def apply_to_project(project: dict, document: dict) -> dict:
    """A copy of a project document with the ``tools[...]`` entries set in its tool list and the
    controller entries deep-merged into controller_overrides (so they win over the project's own)."""
    from ..db.merge import deep_merge
    out = apply_to_tools(project, document)
    overrides = controller_overrides(document)
    if overrides:
        out["controller_overrides"] = deep_merge(out.get("controller_overrides") or {}, overrides)
    return out


def combined(first: dict, second: dict) -> dict:
    """One overlay with the entries of ``first`` and ``second`` (``second`` wins on a target both set): what a
    project that includes ``first`` (``calibration: <file>``) is after ``second`` was calibrated on top of it. The
    entries of ``first`` keep their overlay's name for their notes (``overlay``)."""
    entries = {path: {**entry, "overlay": entry.get("overlay") or first.get("name") or "calibration overlay"}
               for path, entry in (first.get("set") or {}).items()}
    entries.update(copy.deepcopy(second.get("set") or {}))
    out = copy.deepcopy(second)
    out["set"] = dict(sorted(entries.items()))
    out["machine"] = second.get("machine") or first.get("machine")
    out["material"] = second.get("material") or first.get("material")
    return check_overlay(out)


def minimal_patch(raw: dict, document: dict, kind: str) -> dict:
    """The smallest ``extends`` patch of a database object carrying the overlay's entries: mappings to
    the changed keys, lists (which an extends replaces whole) repeated in full with the change."""
    patched = apply_to_raw(kind, raw, document)
    patch: dict = {}
    for path, segments, entry in entries_for(document, kind):
        if kind == "machine" and segments[1] == ("controller", None):
            continue                       # carried by controller_overrides
        source, target = patched, patch
        for depth, (key, selector) in enumerate(segments[1:]):
            last = depth == len(segments) - 2
            if selector is not None or isinstance(source.get(key), list) or last:
                target[key] = copy.deepcopy(source[key])
                break
            target = target.setdefault(key, {})
            source = source[key]
    return patch


PATH_KEYS = (("cad", "path"), ("stock", "mesh"), ("tools", "library"))


def calibrated_project(project_path, document: dict, out_path, *, machine_raw: dict | None = None,
                       material_raw: dict | None = None) -> Path:
    """Write a standalone project file (``out_path``) that is the project with the overlay applied.

    ``machine_raw`` / ``material_raw``: the raw (migrated, merged) database objects the project uses,
    needed when the overlay has entries for them (cnctwin.db.Database.raw). File paths of the
    project are made absolute so the file can live anywhere. A project that includes an overlay itself
    (``calibration: <file>``) is written with both overlays baked in (:func:`combined`: this one wins on a target
    both set) and without the ``calibration`` key, so no value is applied twice."""
    project_path = Path(project_path)
    base = project_path.parent
    try:
        data = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise InputError(f"cannot read the project: {error}", str(project_path)) from None
    included = data.pop("calibration", None) if isinstance(data, dict) else None
    if included:
        included_path = Path(included) if Path(str(included)).is_absolute() else base / str(included)
        document = combined(load_overlay(included_path), document)
    data = apply_to_project(data, document)
    if entries_for(document, "machine"):
        patch = minimal_patch(machine_raw, document, "machine") if machine_raw is not None else None
        if patch is None:
            raise InputError("the overlay sets machine values: pass the machine's raw data")
        if patch:
            ref = data.get("machine")
            if isinstance(ref, str) and not (ref.endswith((".yaml", ".yml")) or "/" in ref):
                data["machine"] = {"extends": ref, **patch}
            else:
                data["machine"] = {k: v for k, v in apply_to_raw("machine", machine_raw, document).items()
                                   if k not in ("format", "version")}
    if entries_for(document, "material"):
        if material_raw is None:
            raise InputError("the overlay sets material values: pass the material's raw data")
        patch = minimal_patch(material_raw, document, "material")
        section = dict(data.get("material") or {})
        ref = section.get("id")
        if isinstance(ref, str) and not (ref.endswith((".yaml", ".yml")) or "/" in ref):
            section["id"] = {"extends": ref, **patch}
        else:
            section["id"] = {k: v for k, v in apply_to_raw("material", material_raw, document).items()
                             if k not in ("format", "version")}
        data["material"] = section
    for keys in PATH_KEYS:
        node = data
        for key in keys[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and isinstance(node.get(keys[-1]), str):
            node[keys[-1]] = str((base / node[keys[-1]]).resolve())
    for op in data.get("operations") or []:
        if isinstance(op.get("program"), str):
            op["program"] = str((base / op["program"]).resolve())
        if isinstance(op.get("programs"), dict):
            op["programs"] = {k: str((base / v).resolve()) for k, v in op["programs"].items()}
        stock = (op.get("setup") or {}).get("stock") if isinstance(op.get("setup"), dict) else None
        if isinstance(stock, dict) and isinstance(stock.get("mesh"), str):
            stock["mesh"] = str((base / stock["mesh"]).resolve())
    out_path = Path(out_path)
    text = (f"# {project_path.name} with the calibration overlay {document.get('name')!r} applied "
            f"(cnctwin.mdata.overlay.calibrated_project)\n") + yaml.safe_dump(_yaml_plain(data), sort_keys=False,
                                                                               allow_unicode=True, width=110)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_path)
    return out_path


__all__ = ["FORMAT", "VERSION", "Target", "parse_path", "set_path", "make_overlay", "check_overlay", "write_overlay",
           "load_overlay", "apply_to_raw", "apply_to_tools", "apply_to_project", "controller_overrides",
           "model_overrides", "factors", "minimal_patch", "combined", "calibrated_project", "entries_for"]
