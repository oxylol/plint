"""Point variants of a sweep: one project file per point, holding every value the project can hold
(docs/formats.md "Sweep").

``points/p<nnnnn>/project.yaml`` is the base project with:

* file paths made absolute (cnctwin.uncertainty.variants.absolute_paths; a CSV tool library inlined into
  ``tools.list``), so the variant lives in its own folder;
* every ``project`` parameter written at its key path;
* the program of every operation a ``program_template`` parameter names: the template with its placeholders
  replaced (and the spec's fixed ``placeholders``), written as ``points/p<nnnnn>/<program file>`` (relative, so the
  project hash does not depend on where the sweep folder is); an unbound placeholder is an error;
* the program of every operation a ``generator`` produces (cnctwin.sweep.generate), written the same way;
* ``name``: the base name and the values written into the file (``[stickout=30, F1=1200]``): the same values give
  the same project hash, so a repeated point, a resumed sweep and another sweep over the same values share cached
  runs (points that differ only in overrides share one project file and differ in their options).

``override`` parameters are not in the file: they go to the point's run options as ``operator`` (normalised,
validated against the machine's override ranges and the tool list, cnctwin.sweep.operator). Every variant is
checked with :func:`cnctwin.project.load_project`.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import InputError
from .spec import Parameter, SweepSpec, set_key_path

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass
class PointPlan:
    index: int
    values: dict
    units: list | None
    project: Path
    options: dict
    project_sha256: str
    programs: dict = field(default_factory=dict)      # operation -> program file written in the point folder

    def to_dict(self, root: Path) -> dict:
        return {"index": self.index, "values": dict(self.values), "units": self.units,
                "project": self.project.relative_to(root).as_posix(), "options": copy.deepcopy(self.options),
                "project_sha256": self.project_sha256, "programs": dict(self.programs)}


def point_dir(root: Path, index: int) -> Path:
    return Path(root) / "points" / f"p{index:05d}"


def nc_number(value) -> str:
    """A value as an NC word value: integers as written, floats with a decimal point (``1200.``, ``0.5``), so a
    control without decimal-point programming does not read ``X10`` as 10 increments."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InputError(f"{value!r} cannot be written into a program")
        text = f"{value:.6f}".rstrip("0")
        return "0." if text in ("-0.", "-0") else text
    return str(value)


def format_value(p: Parameter, value) -> str:
    if p.format:
        return p.format.format(value)
    return nc_number(value)


def label(values: dict) -> str:
    parts = []
    for name, value in values.items():
        if isinstance(value, float):
            parts.append(f"{name}={value:.6g}")
        else:
            parts.append(f"{name}={value}")
    return "[" + ", ".join(parts) + "]"


def _operation_index(data: dict, name: str | None, where: str) -> int:
    operations = data.get("operations") or []
    if name is None:
        return 0
    for i, op in enumerate(operations):
        if isinstance(op, dict) and op.get("name") == name:
            return i
    raise InputError(f"{where}: the project has no operation {name!r}")


def _program_name(op: dict, where: str) -> str:
    if op.get("programs"):
        raise InputError(f"{where}: operation {op.get('name')} has several programs (programs:); a sweep replaces the "
                         f"program of single-program operations only")
    program = op.get("program")
    if not isinstance(program, str):
        raise InputError(f"{where}: operation {op.get('name')} names no program")
    return Path(program).name


class VariantWriter:
    """Writes the point variants of one spec (module docstring)."""

    def __init__(self, spec: SweepSpec, root: Path, *, db=None) -> None:
        from ..project import load_project
        from ..uncertainty.variants import absolute_paths, inline_library
        self.spec = spec
        self.root = Path(root)
        self.base = load_project(spec.base_project)
        data = copy.deepcopy(self.base.data)
        self.original_programs = {}
        for i, op in enumerate(data.get("operations") or []):
            if isinstance(op, dict) and isinstance(op.get("program"), str):
                self.original_programs[i] = (self.base.dir / op["program"]).resolve() \
                    if not Path(op["program"]).is_absolute() else Path(op["program"])
        absolute_paths(data, self.base)
        inline_library(data, self.base.dir)
        self.base_data = data
        self.db = db
        self._resolved = None
        self.templates = self._templates()
        self.generated = self._generator_ops()

    # ------------------------------------------------------------------ the base project
    @property
    def resolved(self):
        """The base project resolved against the databases (tools, machine limits for the overrides)."""
        if self._resolved is None:
            from ..cache import database
            from ..project import resolve
            self._resolved = resolve(self.base, self.db if self.db is not None else database())
        return self._resolved

    def _templates(self) -> dict:
        """operation index -> (template text, program file name, [parameters])."""
        out: dict = {}
        params = defaultdict(list)
        for p in self.spec.parameters:
            if p.kind == "program_template":
                params[_operation_index(self.base_data, p.operation, f"parameter {p.name}")].append(p)
        for index, group in sorted(params.items()):
            files = {p.template for p in group}
            if len(files) > 1:
                raise InputError(f"operation {self.base_data['operations'][index].get('name')}: its program_template "
                                 f"parameters name different templates ({', '.join(sorted(map(str, files)))})")
            template = files.pop()
            op = self.base_data["operations"][index]
            name = _program_name(op, f"parameter {group[0].name}")
            path = (self.base.dir / template).resolve() if template else self.original_programs.get(index)
            if path is None or not path.is_file():
                raise InputError(f"parameter {group[0].name}: no template file {path}")
            text = path.read_text(encoding="utf-8")
            found = set(PLACEHOLDER.findall(text))
            bound = {p.placeholder for p in group} | set(self.spec.placeholders)
            missing = sorted(found - bound)
            if missing:
                raise InputError(f"template {path}: placeholders {', '.join('{{' + m + '}}' for m in missing)} have no "
                                 f"parameter and no fixed value (placeholders:)")
            for p in group:
                if p.placeholder not in found:
                    raise InputError(f"parameter {p.name}: the template {path} has no {{{{{p.placeholder}}}}}")
            out[index] = (text, name, group)
        return out

    def _generator_ops(self) -> dict:
        """operation index -> (generator, [parameters])."""
        out: dict = {}
        by_generator = defaultdict(list)
        for p in self.spec.parameters:
            if p.kind == "generator":
                by_generator[p.generator].append(p)
        for name, group in sorted(by_generator.items()):
            generator = self.spec.generators[name]
            index = _operation_index(self.base_data, generator.operation, f"generator {name}")
            if index in out or index in self.templates:
                raise InputError(f"generator {name}: operation {self.base_data['operations'][index].get('name')} already "
                                 f"gets its program from another template or generator")
            out[index] = (generator, group)
        return out

    # ------------------------------------------------------------------ one point
    def write(self, index: int, values: dict, units=None) -> PointPlan:
        from ..uncertainty.variants import write_variant
        from .generate import generate, write_record
        from .operator import problems
        spec = self.spec
        folder = point_dir(self.root, index)
        folder.mkdir(parents=True, exist_ok=True)
        data = copy.deepcopy(self.base_data)
        where = f"point {index} {label(values)}"
        for p in spec.parameters:
            if p.kind == "project":
                try:
                    set_key_path(data, p.path, values[p.name])
                except InputError as error:
                    raise InputError(f"{where}: {error}") from None
        programs = {}
        for op_index, (text, name, group) in sorted(self.templates.items()):
            fixed = {k: nc_number(v) for k, v in spec.placeholders.items()}
            fixed.update({p.placeholder: format_value(p, values[p.name]) for p in group})
            program = PLACEHOLDER.sub(lambda m: fixed[m.group(1)], text)
            self._write_program(data, folder, op_index, name, program, programs)
        for op_index, (generator, group) in sorted(self.generated.items()):
            name = generator.output or _program_name(data["operations"][op_index], f"generator {generator.name}")
            program, record = generate(generator, self.base.dir, {p.argument: values[p.name] for p in group}, folder,
                                       output=name, skip=[self.root])
            self._write_program(data, folder, op_index, Path(name).name, program, programs)
            write_record(folder, record)
        operator = self._operator(values)
        if operator is not None:
            found = problems(operator, self.resolved)
            if found:
                raise InputError(f"{where}: " + "; ".join(found))
        written = {p.name: values[p.name] for p in spec.parameters if p.kind != "override"}
        if written:
            data["name"] = f"{data.get('name', 'project')} {label(written)}"
        comment = (f"Sweep {spec.name!r}, point {index}: {label(values)}\nbase project: {spec.base_project}\n"
                   f"written by cnctwin.sweep (docs/formats.md \"Sweep\")")
        try:
            path = write_variant(folder, data, comment)
        except InputError as error:
            raise InputError(f"{where}: the variant does not validate:\n{error}") from None
        options = copy.deepcopy(spec.options)
        if operator is not None:
            options["operator"] = operator
        return PointPlan(index=index, values=dict(values), units=units, project=path, options=options,
                         project_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), programs=programs)

    def _write_program(self, data: dict, folder: Path, op_index: int, name: str, program: str, programs: dict) -> None:
        op = data["operations"][op_index]
        taken = set(programs.values())
        if name in taken:
            name = f"{op.get('name', op_index)}-{name}"
        target = folder / name
        if not target.is_file() or target.read_text(encoding="utf-8") != program:
            tmp = target.with_name(f".{target.name}.tmp")
            tmp.write_text(program, encoding="utf-8")
            tmp.replace(target)
        op["program"] = name
        programs[str(op.get("name", op_index))] = name

    def _operator(self, values: dict) -> dict | None:
        from .operator import normalise, tool_key
        operator: dict = {}
        for p in self.spec.parameters:
            if p.kind != "override":
                continue
            value = float(values[p.name])
            if p.target == "rapid":
                operator["rapid_override"] = value
                continue
            table = operator.setdefault(f"{p.target}_override", {})
            key = "*" if p.tool is None else tool_key(p.tool, p.path or 1)
            table[key] = value
        return normalise(operator)


__all__ = ["VariantWriter", "PointPlan", "point_dir", "nc_number", "format_value", "label", "PLACEHOLDER"]
