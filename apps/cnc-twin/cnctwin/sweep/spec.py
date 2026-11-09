"""The sweep spec (format ``cnctwin-sweep``, version 1; docs/formats.md "Sweep", docs/ml.md).

A sweep is a base project, parameters and a design: every point of the design is one project variant run
through the run cache. :func:`load_spec` reads a YAML file (or takes a mapping), checks every key and returns a
:class:`SweepSpec`; every problem is listed in one InputError.

Parameter kinds:

* ``override``: ``target`` feed | spindle | rapid, optional ``tool`` (T number; ``path`` for paths other than 1):
  what the operator's override does on the machine (cnctwin.sweep.operator), a fraction (1 = 100 %);
* ``program_template``: ``{{NAME}}`` placeholders (``placeholder``, default the name) in ``template`` (a program
  file of the base project, default the operation's own program) are replaced by the value written with
  ``format`` (Python format spec, default the shortest form); ``operation`` names the operation (default the
  first);
* ``project``: any key path of the project file (``tools.list[number=1].stickout_mm``,
  ``operations[name=OP10].setup.workholding[0].input_torque_Nm``, ``coolant.fluid``, ``physics.lattice_mm``);
* ``generator``: a value for ``argument`` of a generator of the spec's ``generators`` section (a make_program.py
  style script that writes the operation's program; cnctwin.sweep.generate).

Values: ``range: [lo, hi]`` (with ``scale: linear|log``, ``integer: true``, ``step``, and ``levels`` for a grid)
or ``values: [...]`` (any YAML values; a categorical parameter), ``default`` (the value of a ``list`` point that
omits it), ``unit`` (documentation, carried into the dataset's schema).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import InputError

FORMAT = "cnctwin-sweep"
VERSION = 1
KINDS = ("override", "program_template", "project", "generator")
DESIGNS = ("grid", "lhs", "sobol", "random", "list")
OVERRIDE_TARGETS = ("feed", "spindle", "rapid")
ISOLATIONS = ("process", "inline")
MEMORY_MODES = ("auto", "systemd", "rlimit", "none")
LHS_OPTIMIZATIONS = ("none", "random-cd", "lloyd")
DEFAULT_RUNNER = "cnctwin.sweep.runner:run_project"
DEFAULT_OPTIONS = {"outputs": False}
DEFAULT_MEMORY = "1536M"
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_SIZE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)i?B?\s*$", re.IGNORECASE)
_SCALE = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}

TOP_KEYS = ("format", "version", "name", "description", "base", "seed", "design", "parameters", "generators",
            "placeholders", "options", "execution", "note", "notes")
PARAMETER_KEYS = ("name", "kind", "range", "values", "levels", "scale", "integer", "step", "default", "unit", "doc",
                  "target", "tool", "path", "placeholder", "template", "operation", "format", "generator", "argument",
                  "note", "notes")
DESIGN_KEYS = ("kind", "points", "optimization", "scramble", "list", "note", "notes")
GENERATOR_KEYS = ("script", "function", "call", "output", "operation", "constants", "timeout_s", "note", "notes")
EXECUTION_KEYS = ("workers", "memory", "memory_mode", "isolation", "runner", "retry_failed", "verify", "note", "notes")


def parse_bytes(text) -> int:
    if isinstance(text, bool):
        raise ValueError(f"not a size: {text!r}")
    if isinstance(text, (int, float)):
        return int(text)
    match = _SIZE.match(str(text))
    if not match:
        raise ValueError(f"not a size: {text!r} (use e.g. 1536M or 1.5G)")
    return int(float(match.group(1)) * _SCALE[match.group(2).upper()])


@dataclass
class Parameter:
    name: str
    kind: str
    range: tuple[float, float] | None = None
    values: list | None = None
    levels: int | None = None
    scale: str = "linear"
    integer: bool = False
    step: float | None = None
    default: object = None
    unit: str = "-"
    doc: str = ""
    target: str | None = None          # override
    tool: int | None = None            # override
    path: object = None                # override: controller path of the tool (int); project: the key path (text)
    placeholder: str | None = None     # program_template
    template: str | None = None        # program_template
    operation: str | None = None       # program_template, generator (from the generator)
    format: str | None = None          # program_template
    generator: str | None = None       # generator
    argument: str | None = None        # generator

    @property
    def categorical(self) -> bool:
        return self.values is not None

    def describe(self) -> dict:
        out = {"name": self.name, "kind": self.kind, "unit": self.unit, "doc": self.doc}
        for key in ("range", "values", "levels", "scale", "integer", "step", "default", "target", "tool", "path",
                    "placeholder", "template", "operation", "format", "generator", "argument"):
            value = getattr(self, key)
            if value is not None and value is not False and not (key == "scale" and value == "linear"):
                out[key] = list(value) if isinstance(value, tuple) else value
        return out


@dataclass
class Generator:
    name: str
    script: str
    function: str = "main"
    call: str = "constants"
    output: str | None = None
    operation: str | None = None
    constants: dict = field(default_factory=dict)
    timeout_s: float = 600.0

    def describe(self) -> dict:
        return {"name": self.name, "script": self.script, "function": self.function, "call": self.call,
                "output": self.output, "operation": self.operation, "constants": dict(self.constants),
                "timeout_s": self.timeout_s}


@dataclass
class Design:
    kind: str
    points: int | None = None
    optimization: str = "none"
    scramble: bool = True
    list: list | None = None

    def describe(self) -> dict:
        out = {"kind": self.kind}
        if self.points is not None:
            out["points"] = self.points
        if self.kind == "lhs":
            out["optimization"] = self.optimization
        if self.kind == "sobol":
            out["scramble"] = self.scramble
        if self.list is not None:
            out["list"] = copy.deepcopy(self.list)
        return out


@dataclass
class Execution:
    workers: int = 1
    memory_bytes: int = parse_bytes(DEFAULT_MEMORY)
    memory_mode: str = "auto"
    isolation: str = "process"
    runner: str = DEFAULT_RUNNER
    retry_failed: bool = False
    verify: str = "full"

    def describe(self) -> dict:
        return {"workers": self.workers, "memory_bytes": self.memory_bytes, "memory_mode": self.memory_mode,
                "isolation": self.isolation, "runner": self.runner, "retry_failed": self.retry_failed,
                "verify": self.verify}


@dataclass
class SweepSpec:
    name: str
    base: Path
    seed: int
    design: Design
    parameters: list[Parameter]
    generators: dict[str, Generator]
    placeholders: dict
    options: dict
    execution: Execution
    description: str = ""
    path: Path | None = None
    sha256: str = ""
    data: dict = field(default_factory=dict)

    @property
    def base_project(self) -> Path:
        return self.base / "project.yaml" if self.base.is_dir() else self.base

    @property
    def base_dir(self) -> Path:
        return self.base_project.parent

    def parameter(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)

    def describe(self) -> dict:
        """The spec as resolved (absolute base, defaults filled in): sweep.json's ``spec``."""
        return {"format": FORMAT, "version": VERSION, "name": self.name, "description": self.description,
                "base": str(self.base_project), "seed": self.seed, "design": self.design.describe(),
                "parameters": [p.describe() for p in self.parameters],
                "generators": {k: g.describe() for k, g in sorted(self.generators.items())},
                "placeholders": dict(self.placeholders), "options": copy.deepcopy(self.options),
                "execution": self.execution.describe()}


class _Problems:
    def __init__(self, where: str) -> None:
        self.where = where
        self.items: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.items.append(f"{path}: {message}" if path else message)

    def raise_any(self) -> None:
        if self.items:
            raise InputError(f"sweep spec {self.where}: {len(self.items)} problem(s)\n" + "\n".join(self.items))


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _unknown(mapping: dict, allowed: tuple, path: str, problems: _Problems) -> None:
    for key in mapping:
        if key not in allowed:
            problems.add(f"{path}.{key}" if path else str(key), f"unknown key (allowed: {', '.join(allowed)})")


def load_spec(source, *, base_override=None) -> SweepSpec:
    """Read and check a sweep spec: a YAML file path, or a mapping (``base`` relative to the current folder then).
    ``base_override`` replaces ``base`` (the API's project)."""
    from ..db.yamlio import load_file
    path = None
    if isinstance(source, dict):
        data = copy.deepcopy(source)
        text = json.dumps(data, sort_keys=True, default=str)
        where = "(mapping)"
    else:
        path = Path(source).resolve()
        if not path.is_file():
            raise InputError(f"no sweep spec {path}")
        doc = load_file(path)
        data = doc.data
        text = doc.text
        where = str(path)
    if not isinstance(data, dict):
        raise InputError(f"sweep spec {where}: a sweep spec is a YAML mapping")
    problems = _Problems(where)
    _unknown(data, TOP_KEYS, "", problems)
    if data.get("format", FORMAT) != FORMAT:
        problems.add("format", f"{data.get('format')!r} is not {FORMAT}")
    version = data.get("version", VERSION)
    if version != VERSION:
        problems.add("version", f"version {version!r}: this build reads version {VERSION}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.add("name", "a sweep needs a name")
        name = "sweep"
    folder = path.parent if path is not None else Path.cwd()
    base_value = base_override if base_override is not None else data.get("base")
    base = None
    if not isinstance(base_value, (str, Path)) or not str(base_value):
        problems.add("base", "the base project (a project.yaml or its folder) is required")
    else:
        base = Path(base_value)
        base = (folder / base).resolve() if not base.is_absolute() else base.resolve()
        project_file = base / "project.yaml" if base.is_dir() else base
        if not project_file.is_file():
            problems.add("base", f"no project file {project_file}")
    seed = data.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        problems.add("seed", "a non-negative integer")
        seed = 0
    generators = _generators(data.get("generators"), problems)
    parameters = _parameters(data.get("parameters"), generators, problems)
    design = _design(data.get("design"), parameters, problems)
    placeholders = data.get("placeholders") or {}
    if not isinstance(placeholders, dict):
        problems.add("placeholders", "a mapping {NAME: value} of fixed placeholder values")
        placeholders = {}
    options = data.get("options")
    if options is None:
        options = dict(DEFAULT_OPTIONS)
    elif not isinstance(options, dict):
        problems.add("options", "a mapping of run options (cnctwin.core.loop.run_project)")
        options = dict(DEFAULT_OPTIONS)
    if "operator" in options:
        problems.add("options.operator", "operator overrides are parameters of kind override, not options")
    execution = _execution(data.get("execution"), problems)
    problems.raise_any()
    return SweepSpec(name=name.strip(), base=base, seed=int(seed), design=design, parameters=parameters,
                     generators=generators, placeholders=dict(placeholders), options=dict(options), execution=execution,
                     description=str(data.get("description") or ""), path=path,
                     sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), data=data)


def _generators(section, problems: _Problems) -> dict[str, Generator]:
    out: dict[str, Generator] = {}
    if section is None:
        return out
    if not isinstance(section, dict):
        problems.add("generators", "a mapping {name: {script, function, call, output, operation, constants}}")
        return out
    for name, entry in section.items():
        where = f"generators.{name}"
        if not isinstance(name, str) or not NAME.match(name):
            problems.add(where, "a generator name is an identifier")
            continue
        if not isinstance(entry, dict):
            problems.add(where, "a mapping")
            continue
        _unknown(entry, GENERATOR_KEYS, where, problems)
        script = entry.get("script")
        if not isinstance(script, str) or not script.endswith(".py"):
            problems.add(f"{where}.script", "the generator script (a .py file of the base project's folder)")
            continue
        call = entry.get("call", "constants")
        if call not in ("constants", "kwargs"):
            problems.add(f"{where}.call", "constants (make_program.py style) or kwargs")
        function = entry.get("function", "main")
        if not isinstance(function, str) or not function.isidentifier():
            problems.add(f"{where}.function", "a function name")
            function = "main"
        constants = entry.get("constants") or {}
        if not isinstance(constants, dict):
            problems.add(f"{where}.constants", "a mapping {CONSTANT: value}")
            constants = {}
        timeout = entry.get("timeout_s", 600.0)
        if not _number(timeout) or float(timeout) <= 0:
            problems.add(f"{where}.timeout_s", "a positive number of seconds")
            timeout = 600.0
        output = entry.get("output")
        if output is not None and (not isinstance(output, str) or Path(output).is_absolute() or ".." in Path(output).parts):
            problems.add(f"{where}.output", "a file name relative to the generator's folder")
            output = None
        operation = entry.get("operation")
        if operation is not None and not isinstance(operation, str):
            problems.add(f"{where}.operation", "an operation name")
            operation = None
        out[name] = Generator(name=name, script=script, function=function, call=str(call), output=output,
                              operation=operation, constants=dict(constants), timeout_s=float(timeout))
    return out


def _parameters(section, generators: dict, problems: _Problems) -> list[Parameter]:
    out: list[Parameter] = []
    if not isinstance(section, list) or not section:
        problems.add("parameters", "a list of at least one parameter")
        return out
    names = set()
    for i, entry in enumerate(section):
        where = f"parameters[{i}]"
        if not isinstance(entry, dict):
            problems.add(where, "a mapping")
            continue
        _unknown(entry, PARAMETER_KEYS, where, problems)
        name = entry.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            problems.add(f"{where}.name", "a parameter name: letters, digits, '_', '.', '-' (starting with a letter)")
            continue
        if name in names:
            problems.add(f"{where}.name", f"parameter {name!r} is named twice")
        names.add(name)
        kind = entry.get("kind")
        if kind not in KINDS:
            problems.add(f"{where}.kind", f"one of {', '.join(KINDS)}")
            continue
        p = Parameter(name=name, kind=kind, unit=str(entry.get("unit", "-") if entry.get("unit") is not None else "-"),
                      doc=str(entry.get("doc") or ""), default=entry.get("default"))
        _values(entry, p, where, problems)
        if kind == "override":
            target = entry.get("target")
            if target not in OVERRIDE_TARGETS:
                problems.add(f"{where}.target", f"one of {', '.join(OVERRIDE_TARGETS)}")
            p.target = target
            tool = entry.get("tool")
            if tool is not None and (not isinstance(tool, int) or isinstance(tool, bool) or tool < 0):
                problems.add(f"{where}.tool", "a tool number")
            if target == "rapid" and tool is not None:
                problems.add(f"{where}.tool", "the rapid override is one key of the panel, not per tool")
            p.tool = tool
            tool_path = entry.get("path")
            if tool_path is not None and (not isinstance(tool_path, int) or isinstance(tool_path, bool) or tool_path < 1):
                problems.add(f"{where}.path", "the controller path of the tool (1, 2, ...)")
            p.path = tool_path
            if p.categorical and any(not _number(v) for v in p.values):
                problems.add(f"{where}.values", "an override is a number (1 = 100 %)")
        elif kind == "program_template":
            placeholder = entry.get("placeholder", name)
            if not isinstance(placeholder, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", placeholder):
                problems.add(f"{where}.placeholder", "an identifier ({{NAME}} in the program)")
            p.placeholder = placeholder
            template = entry.get("template")
            if template is not None and (not isinstance(template, str) or Path(template).is_absolute()
                                         or ".." in Path(template).parts):
                problems.add(f"{where}.template", "a program file relative to the base project's folder")
            p.template = template
            p.operation = entry.get("operation")
            fmt = entry.get("format")
            if fmt is not None:
                try:
                    fmt.format(1.5)
                except (AttributeError, ValueError, IndexError, KeyError):
                    problems.add(f"{where}.format", "a Python format string with one field, e.g. '{:.0f}'")
                    fmt = None
            p.format = fmt
        elif kind == "project":
            key_path = entry.get("path")
            if not isinstance(key_path, str) or not key_path.strip():
                problems.add(f"{where}.path", "a key path of the project file (tools.list[number=1].stickout_mm)")
            else:
                try:
                    parse_key_path(key_path)
                except InputError as error:
                    problems.add(f"{where}.path", str(error))
            p.path = key_path
        elif kind == "generator":
            generator = entry.get("generator")
            if generator not in generators:
                problems.add(f"{where}.generator", f"a generator of the generators section ({', '.join(sorted(generators)) or 'none'})")
            argument = entry.get("argument")
            if not isinstance(argument, str) or not argument.isidentifier():
                problems.add(f"{where}.argument", "the constant or keyword argument the value sets")
            p.generator, p.argument = generator, argument
        out.append(p)
    return out


def _values(entry: dict, p: Parameter, where: str, problems: _Problems) -> None:
    has_range, has_values = "range" in entry, "values" in entry
    if has_range == has_values:
        problems.add(where, "give range: [lo, hi] or values: [...]")
        return
    if has_values:
        values = entry["values"]
        if not isinstance(values, list) or not values:
            problems.add(f"{where}.values", "a non-empty list")
            return
        p.values = copy.deepcopy(values)
        for key in ("levels", "scale", "integer", "step"):
            if key in entry:
                problems.add(f"{where}.{key}", "only with range")
        return
    rng = entry["range"]
    if not isinstance(rng, list) or len(rng) != 2 or not all(_number(v) for v in rng) or float(rng[0]) >= float(rng[1]):
        problems.add(f"{where}.range", "[lo, hi] with lo < hi")
        return
    p.range = (float(rng[0]), float(rng[1]))
    scale = entry.get("scale", "linear")
    if scale not in ("linear", "log"):
        problems.add(f"{where}.scale", "linear or log")
    elif scale == "log" and p.range[0] <= 0.0:
        problems.add(f"{where}.scale", "a log scale needs lo > 0")
    p.scale = str(scale)
    p.integer = bool(entry.get("integer", False))
    step = entry.get("step")
    if step is not None and (not _number(step) or float(step) <= 0):
        problems.add(f"{where}.step", "a positive number")
        step = None
    p.step = float(step) if step is not None else None
    levels = entry.get("levels")
    if levels is not None and (not isinstance(levels, int) or isinstance(levels, bool) or levels < 2):
        problems.add(f"{where}.levels", "an integer of at least 2 (the grid's values over the range)")
        levels = None
    p.levels = levels


def _design(section, parameters: list[Parameter], problems: _Problems) -> Design:
    if not isinstance(section, dict):
        problems.add("design", "a mapping {kind: grid|lhs|sobol|random|list, points, ...}")
        return Design("grid")
    _unknown(section, DESIGN_KEYS, "design", problems)
    kind = section.get("kind")
    if kind not in DESIGNS:
        problems.add("design.kind", f"one of {', '.join(DESIGNS)}")
        return Design("grid")
    design = Design(kind=kind)
    points = section.get("points")
    if kind in ("lhs", "sobol", "random"):
        if not isinstance(points, int) or isinstance(points, bool) or points < 1:
            problems.add("design.points", f"the number of points of the {kind} design (at least 1)")
        else:
            design.points = points
    elif points is not None:
        problems.add("design.points", f"a {kind} design has its points from the parameters" if kind == "grid"
                     else "a list design has the points of its list")
    optimization = section.get("optimization", "none")
    if optimization not in LHS_OPTIMIZATIONS:
        problems.add("design.optimization", f"one of {', '.join(LHS_OPTIMIZATIONS)}")
    elif optimization != "none" and kind != "lhs":
        problems.add("design.optimization", "only for lhs")
    design.optimization = str(optimization)
    design.scramble = bool(section.get("scramble", True))
    if kind == "grid":
        for p in parameters:
            if not p.categorical and p.levels is None:
                problems.add(f"parameters[{p.name}]", "a grid needs values: [...] or range with levels")
    if kind == "list":
        listed = section.get("list")
        if not isinstance(listed, list) or not listed:
            problems.add("design.list", "a non-empty list of points {parameter: value}")
        else:
            names = {p.name for p in parameters}
            for i, point in enumerate(listed):
                if not isinstance(point, dict):
                    problems.add(f"design.list[{i}]", "a mapping {parameter: value}")
                    continue
                for key in point:
                    if key not in names:
                        problems.add(f"design.list[{i}].{key}", "not a parameter of this sweep")
                for p in parameters:
                    if p.name not in point and p.default is None:
                        problems.add(f"design.list[{i}]", f"gives no value for {p.name} (and it has no default)")
            design.list = copy.deepcopy(listed)
    return design


def _execution(section, problems: _Problems) -> Execution:
    out = Execution()
    if section is None:
        return out
    if not isinstance(section, dict):
        problems.add("execution", "a mapping {workers, memory, memory_mode, isolation, runner, retry_failed}")
        return out
    _unknown(section, EXECUTION_KEYS, "execution", problems)
    workers = section.get("workers", 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        problems.add("execution.workers", "at least 1")
    else:
        out.workers = workers
    if "memory" in section:
        try:
            out.memory_bytes = parse_bytes(section["memory"])
        except ValueError as error:
            problems.add("execution.memory", str(error))
    mode = section.get("memory_mode", "auto")
    if mode not in MEMORY_MODES:
        problems.add("execution.memory_mode", f"one of {', '.join(MEMORY_MODES)}")
    else:
        out.memory_mode = mode
    isolation = section.get("isolation", "process")
    if isolation not in ISOLATIONS:
        problems.add("execution.isolation", f"one of {', '.join(ISOLATIONS)}")
    else:
        out.isolation = isolation
    if out.workers > 1 and out.isolation == "inline":
        problems.add("execution.workers", "several workers need isolation: process")
    runner = section.get("runner", DEFAULT_RUNNER)
    if not isinstance(runner, str) or ":" not in runner:
        problems.add("execution.runner", "module:function or /path/file.py:function")
    else:
        out.runner = runner
    out.retry_failed = bool(section.get("retry_failed", False))
    verify = section.get("verify", "full")
    if verify not in ("full", "quick"):
        problems.add("execution.verify", "full or quick")
    else:
        out.verify = verify
    return out


# =============================================================================== key paths of the project file

_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\]]+\])*)$")
_INDEX = re.compile(r"\[([^\]]+)\]")


def parse_key_path(text: str) -> list:
    """``tools.list[number=1].stickout_mm`` -> ["tools", "list", ("number", 1), "stickout_mm"]: keys, list indexes
    (int) and list selectors (key, value) matching the one item whose key has that value."""
    out: list = []
    for part in str(text).strip().split("."):
        match = _SEGMENT.match(part)
        if not match:
            raise InputError(f"key path {text!r}: {part!r} is not key or key[index] or key[field=value]")
        out.append(match.group(1))
        for index in _INDEX.findall(match.group(2)):
            index = index.strip()
            if re.fullmatch(r"-?\d+", index):
                out.append(int(index))
            elif "=" in index:
                key, _, value = index.partition("=")
                key, value = key.strip(), value.strip()
                if not key.isidentifier():
                    raise InputError(f"key path {text!r}: selector [{index}] needs field=value")
                out.append((key, _literal(value)))
            else:
                raise InputError(f"key path {text!r}: [{index}] is neither an index nor field=value")
    return out


def _literal(text: str):
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d*(?:[eE][-+]?\d+)?", text):
        return float(text)
    return text.strip("'\"")


def set_key_path(data, key_path: str, value) -> None:
    """Set ``value`` at the key path in the project data (the last key may be new; every other step must exist)."""
    steps = parse_key_path(key_path)
    node = data
    for i, step in enumerate(steps):
        last = i == len(steps) - 1
        where = ".".join(str(s) for s in steps[:i + 1])
        if isinstance(step, tuple):
            if not isinstance(node, list):
                raise InputError(f"{key_path}: {where} selects in a list, found {type(node).__name__}")
            matches = [j for j, item in enumerate(node) if isinstance(item, dict) and item.get(step[0]) == step[1]]
            if len(matches) != 1:
                raise InputError(f"{key_path}: {len(matches)} items have {step[0]} = {step[1]!r} (exactly one must)")
            if last:
                node[matches[0]] = copy.deepcopy(value)
            else:
                node = node[matches[0]]
        elif isinstance(step, int):
            if not isinstance(node, list) or not -len(node) <= step < len(node):
                raise InputError(f"{key_path}: {where} is not an index of the list")
            if last:
                node[step] = copy.deepcopy(value)
            else:
                node = node[step]
        else:
            if not isinstance(node, dict):
                raise InputError(f"{key_path}: {where}: the value above is not a mapping")
            if last:
                node[step] = copy.deepcopy(value)
            else:
                if step not in node or node[step] is None:
                    raise InputError(f"{key_path}: the project has no {where}")
                node = node[step]


__all__ = ["SweepSpec", "Parameter", "Generator", "Design", "Execution", "load_spec", "parse_key_path",
           "set_key_path", "parse_bytes", "FORMAT", "VERSION", "KINDS", "DESIGNS", "DEFAULT_RUNNER", "DEFAULT_OPTIONS"]
