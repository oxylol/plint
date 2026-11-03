"""Project variants: one project file per sample, with the sampled values written where the project
file holds them, so the simulation needs nothing special to honour them.

* File paths (programs, CAD, stock mesh, tool library) become absolute, so a variant can live in its
  own folder; a CSV tool library is inlined into ``tools.list`` (so its tools' offsets and runout can
  be varied) and the library key dropped.
* Stock sizes, tool offsets (the geometry offset when the project gives one, else the wear offset),
  runout and clamping torque or pressure are replaced in place (written in the file's units).
* Material batch factors make the material an inline object that ``extends`` the database material
  and repeats only the scaled values: every stress of ``flow_stress`` (``*_MPa``, ``*_Pa``,
  ``*_N_mm2``, tables included) by kc x flow-stress scale; the Kienzle and mechanistic coefficients
  (stresses ``*_N_mm2`` and edge forces ``*_N_mm``) and ``cutting.indentation_coeff_N_mm3`` by the kc
  scale; every number of ``physical.hardness`` by the hardness scale; ``physical.conductivity_W_mK``
  by the conductivity scale. Lists (Kienzle, mechanistic) are repeated whole (``extends`` replaces
  lists), mappings merge key by key.
* Factors the project file cannot hold (``applies: options``) are passed to the run in
  ``options["uncertainty"]["factors"]`` (all sampled values are, for the record).

Every variant is checked with :func:`cnctwin.project.load_project` before it is run.
"""

from __future__ import annotations

import copy
from pathlib import Path

from ..core.errors import InputError
from ..db import keyunits
from ..db.yamlio import dumps, load_file
from ..project import load_project
from ..project.schema import is_path_ref
from .factors import Factor

STRESS_TOKENS = ("Pa",)
EDGE_TOKENS = ("N_per_m",)
HEADER_KEYS = ("format", "version")


def _set(data, path: tuple, value) -> None:
    current = data
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def absolute_paths(data: dict, project) -> None:
    for path, text, _node in project.paths:
        if path == ("tools", "library") and str(text).lower().endswith(".csv"):
            continue
        target = Path(text)
        if not target.is_absolute():
            _set(data, path, str((project.dir / text).resolve()))


def inline_library(data: dict, project_dir: Path) -> None:
    """A CSV tool library joins tools.list (the variant must be able to vary its tools)."""
    tools = data.get("tools") or {}
    library = tools.get("library")
    if not library or not str(library).lower().endswith(".csv"):
        return
    from ..project.toollib import read_csv
    path = Path(library) if Path(library).is_absolute() else project_dir / library
    entries, _, problems = read_csv(path)
    errors = [p for p in problems if p.is_error]
    if errors:
        raise InputError("the tool library cannot be inlined into the variant: " + "; ".join(str(p) for p in errors))
    tools["list"] = list(tools.get("list") or []) + entries
    tools["library"] = None


def _scaled(value, factor: float):
    """A number, sourced value {value, ...} or table {table: [[x, v]], x} times ``factor`` (sources dropped:
    the scaled value is the study's, not the source's)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value) * factor
    if isinstance(value, list):
        return [_scaled(v, factor) for v in value]
    if isinstance(value, dict):
        if "table" in value:
            out = {"table": [[row[0], float(row[1]) * factor] for row in value["table"]]}
            if "x" in value:
                out["x"] = value["x"]
            return out
        if "value" in value:
            return _scaled(value["value"], factor)
    return value


def _si_token(key: str) -> str | None:
    _, unit = keyunits.parse_key(key)
    return unit.si_token if unit is not None and not unit.scale else None


def _scale_keys(mapping: dict, tokens: tuple, factor: float) -> dict:
    """The keys of ``mapping`` whose SI unit is one of ``tokens``, scaled."""
    return {key: _scaled(value, factor) for key, value in mapping.items()
            if _si_token(str(key)) in tokens and value is not None}


def material_overrides(base: dict, scales: dict) -> dict:
    """The sections an inline material must repeat for the batch scales (kc, hardness, flow stress,
    conductivity) of the full database material ``base``."""
    kc = scales.get("kc_scale", 1.0)
    flow = kc * scales.get("flow_stress_scale", 1.0)
    out: dict = {}
    if flow != 1.0 and isinstance(base.get("flow_stress"), dict):
        stresses = _scale_keys(base["flow_stress"], STRESS_TOKENS, flow)
        if stresses:
            out["flow_stress"] = stresses
    cutting = base.get("cutting") if isinstance(base.get("cutting"), dict) else {}
    if kc != 1.0 and cutting:
        section = {}
        for name in ("kienzle", "mechanistic"):
            entries = cutting.get(name)
            if isinstance(entries, list) and entries:
                repeated = []
                for entry in entries:
                    copy_ = copy.deepcopy(entry)
                    if isinstance(copy_, dict):
                        copy_.update(_scale_keys(entry, STRESS_TOKENS + EDGE_TOKENS, kc))
                    repeated.append(copy_)
                section[name] = repeated
        if cutting.get("indentation_coeff_N_mm3") is not None:
            section["indentation_coeff_N_mm3"] = _scaled(cutting["indentation_coeff_N_mm3"], kc)
        if section:
            out["cutting"] = section
    physical = base.get("physical") if isinstance(base.get("physical"), dict) else {}
    hardness = scales.get("hardness_scale", 1.0)
    if hardness != 1.0 and isinstance(physical.get("hardness"), dict):
        scaled = {k: _scaled(v, hardness) for k, v in physical["hardness"].items()
                  if k not in ("src", "note", "notes", "page") and isinstance(_scaled(v, 1.0), (int, float))}
        if scaled:
            out.setdefault("physical", {})["hardness"] = scaled
    conductivity = scales.get("conductivity_scale", 1.0)
    if conductivity != 1.0 and physical.get("conductivity_W_mK") is not None:
        out.setdefault("physical", {})["conductivity_W_mK"] = _scaled(physical["conductivity_W_mK"], conductivity)
    return out


def _deep_update(target: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict) and "table" not in value:
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def inline_material(section: dict, scales: dict, db, project_dir: Path, note: str) -> dict:
    """The material section of a variant: an inline object extending the project's material."""
    ref = section.get("id")
    if isinstance(ref, str) and not is_path_ref(ref):
        base = db.raw("material", ref)
        inline = {"extends": ref}
    elif isinstance(ref, str):
        path = Path(ref) if Path(ref).is_absolute() else project_dir / ref
        written = dict(load_file(path).data)
        for key in HEADER_KEYS:
            written.pop(key, None)
        base = _resolved_inline(written, db)
        inline = written
    elif isinstance(ref, dict):
        inline = copy.deepcopy(ref)
        base = _resolved_inline(inline, db)
    else:
        raise InputError(f"material.id {ref!r} is neither an id, a path nor an inline material")
    _deep_update(inline, material_overrides(base, scales))
    inline["notes"] = note
    out = dict(section)
    out["id"] = inline
    return out


def _resolved_inline(inline: dict, db) -> dict:
    base_id = inline.get("extends")
    if isinstance(base_id, str):
        merged = copy.deepcopy(db.raw("material", base_id))
        _deep_update(merged, {k: v for k, v in inline.items() if k != "extends"})
        return merged
    return inline


def _tool_entry(data: dict, factor: Factor) -> dict:
    return data["tools"]["list"][factor.index["tool"]]


def apply(data: dict, factors: list[Factor], values: dict[str, float]) -> list[dict]:
    """Write the project-file factors into ``data`` (in place). Returns what was written where."""
    written = []
    for factor in factors:
        if factor.applies != "project":
            continue
        value = float(values[factor.name])
        if factor.name.startswith("stock_size["):
            data["stock"]["size_mm"][factor.index["i"]] = value / keyunits.unit("mm").factor
        elif factor.name.startswith(("length_offset[", "radius_offset[")):
            entry = _tool_entry(data, factor)
            kind = factor.index["kind"]
            key = f"{kind}_offset_mm"
            delta_mm = value / keyunits.unit("mm").factor
            if isinstance(entry.get(key), (int, float)) and not isinstance(entry.get(key), bool):
                entry[key] = float(entry[key]) + delta_mm
            else:
                wear = dict(entry.get("wear_offsets_mm") or {})
                wear[kind] = float(wear.get(kind) or 0.0) + delta_mm
                entry["wear_offsets_mm"] = wear
        elif factor.name.startswith("runout["):
            entry = _tool_entry(data, factor)
            entry["runout_um"] = value / keyunits.unit("um").factor
        elif factor.name.startswith("clamping_force["):
            op, j, key = factor.index["op"], factor.index["entry"], factor.index["key"]
            holder = data["operations"][op]["setup"]["workholding"][j]
            holder[key] = float(holder[key]) * value
        else:
            raise InputError(f"factor {factor.name} says it is written to the project file but has no writer")
        written.append({"factor": factor.name, "target": factor.target, "value": value})
    return written


def variant_data(project, factors: list[Factor], values: dict[str, float], db, *, sample: int,
                 seed: int | None = None) -> tuple[dict, list[dict]]:
    """The variant's project data and what was written."""
    data = copy.deepcopy(project.data)
    absolute_paths(data, project)
    inline_library(data, project.dir)
    written = apply(data, factors, values)
    scales = {f.name: float(values[f.name]) for f in factors if f.applies == "material"}
    if any(v != 1.0 for v in scales.values()):
        note = (f"uncertainty sample {sample}: " + ", ".join(f"{k} x{v:.6g}" for k, v in sorted(scales.items()))
                + " applied to the project's material")
        data["material"] = inline_material(dict(data["material"]), scales, db, project.dir, note)
        written += [{"factor": name, "target": "material (inline)", "value": value} for name, value in sorted(scales.items())]
    if seed is not None:
        data["seed"] = int(seed)
    data["name"] = f"{data.get('name', 'project')} (uncertainty sample {sample})"
    return data, written


def write_variant(directory: Path, data: dict, comment: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "project.yaml"
    text = "".join(f"# {line}\n" for line in comment.splitlines()) + dumps(data)
    tmp = path.with_name(".project.yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    load_project(path)
    return path


__all__ = ["variant_data", "write_variant", "material_overrides", "inline_material", "apply", "absolute_paths",
           "inline_library"]
