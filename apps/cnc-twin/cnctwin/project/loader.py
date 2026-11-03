"""Load, validate and resolve a project (docs/architecture.md 18, docs/formats.md "Project file").

* :func:`load_project` reads ``project.yaml`` (or a project directory) with line numbers, migrates
  older versions forward (:mod:`cnctwin.project.migrate`) and checks it against the declared schema
  (:mod:`cnctwin.project.schema`). Every problem names ``file:line`` and the key path.
* :func:`resolve` (and :meth:`ResolvedProject.load`) also resolves every database reference through
  :class:`cnctwin.db.Database` (an id, a database file relative to the project, or an inline object
  in place of the id), the tool library (CSV rows join ``tools.list``, ISO 13399 items can be named
  or extended), checks that every file exists, reads the programs, applies
  ``controller_overrides`` to the machine, and hashes every input file.

The result, :class:`ResolvedProject`, is the SI view of the project file (a
:class:`~cnctwin.db.simap.SIMap`: ``project.stock.size_m``, ``project.physics.max_deg_per_step_rad``)
with the database objects in place of their ids: ``project.machine``, ``project.material`` (the
material, plus the project's ``batch``), ``project.tools.list[i].tool`` and ``.holder``,
``project.coolant.fluid``, ``project.operations[i].setup.workholding[j].device``, and
``project.operations[i].program_text`` (``program_path``: absolute path). File paths in the view
are absolute; ``.raw`` holds them as written.

Frames: the workpiece frame W coincides with the workholding frame at setup (``part_pose`` is the
identity until a part slips), so an operation's ``stock_pose_W_S`` is its ``setup.stock_pose``
matrix (4x4, S to W) and ``fixture_pose_F_H`` its ``setup.fixture_pose``; ``cad_pose_S_C`` is
``cad.placement``. ``ResolvedProject.stock_pose_W_S`` is the first operation's (findings use it).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import InputError
from ..db import schemas as DS
from ..db.database import DEFAULT_ROOT, Database
from ..db.merge import deep_merge
from ..db.migrate import migrate as migrate_db
from ..db.problems import Problem, errors_only, format_path, raise_for, sort_problems
from ..db.schema import Ctx
from ..db.simap import PoseMap, SIMap
from ..db.sources import check_sources
from ..db.yamlio import Locations, load_file
from . import toollib
from .migrate import migrate_with_locations, version_of
from .schema import FORMAT, PROJECT, TICK_S, TOOL_ENTRY, is_path_ref

PROJECT_FILE = "project.yaml"
DS_TEMPLATE = Path(DEFAULT_ROOT) / "_templates" / "project.yaml"
# Values a used holder must give (the tool is placed and collision-checked with them).
HOLDER_NEEDS = ("gauge_length_mm", "nose_diameter_mm", "profile_mm")


@dataclass
class ProjectFile:
    """A project file read and checked (not yet resolved against the databases)."""

    path: Path
    data: dict                          # current version (migrated)
    written: dict                       # as written
    locations: Locations
    version_written: int
    migration_notes: list[str]
    sha256: str
    text: str
    problems: list[Problem] = field(default_factory=list)
    refs: list = field(default_factory=list)       # (kind, id-or-mapping, path)
    paths: list = field(default_factory=list)      # (path, text, FilePath node)

    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def rel(self) -> str:
        return str(self.path)


def project_path(path) -> Path:
    path = Path(path)
    return path / PROJECT_FILE if path.is_dir() else path


def load_project(path, *, check: bool = True) -> ProjectFile:
    """Read, migrate and check one project file; raises InputError listing every error."""
    path = project_path(path).resolve()
    doc = load_file(path)
    written = doc.data
    if not isinstance(written, dict):
        raise InputError("a project file is a YAML mapping", str(path), 1)
    if "format" in written and written["format"] != FORMAT:
        raise InputError(f"format {written['format']!r} is not {FORMAT}", str(path), doc.locations.line(("format",)))
    try:
        version = version_of(written)
    except InputError as error:
        raise InputError(str(error), str(path), doc.locations.line(("version",))) from None
    data, locations, notes = migrate_with_locations(written, doc.locations)
    project = ProjectFile(path, data, written, locations, version, notes, doc.sha256, doc.text)
    ctx = Ctx(locations, obj="project", file=str(path))
    PROJECT.check(data, (), None, ctx)
    project.problems.extend(ctx.problems)
    project.refs = list(ctx.refs)
    project.paths = list(ctx.paths)
    _cross_checks(project)
    if check:
        raise_for(project.problems, f"project {path}")
    return project


def _cross_checks(project: ProjectFile) -> None:
    data = project.data
    loc = project.locations
    add = lambda path, message, kind="rule": project.problems.append(  # noqa: E731
        Problem(str(project.path), loc.line(path), format_path(path), message, kind, "error", "project"))
    tools = data.get("tools") if isinstance(data.get("tools"), Mapping) else {}
    numbers: dict[tuple, int] = {}
    for i, entry in enumerate(tools.get("list") or []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("number"), int):
            continue
        key = (entry.get("path", 1), entry["number"])
        if key in numbers:
            add(("tools", "list", i, "number"), f"tool number {entry['number']} (path {key[0]}) is used twice "
                                               f"(also tools.list[{numbers[key]}])", "duplicate")
        else:
            numbers[key] = i
    known_numbers = {number for _, number in numbers}
    operations = data.get("operations") if isinstance(data.get("operations"), list) else []
    names = set()
    for i, op in enumerate(operations):
        if not isinstance(op, Mapping):
            continue
        names.add(op.get("name"))
        for number in (op.get("nozzle_positions") or {}) if isinstance(op.get("nozzle_positions"), Mapping) else ():
            if isinstance(number, int) and number not in known_numbers and not tools.get("library"):
                add(("operations", i, "nozzle_positions", number), f"tool {number} is not in tools.list", "reference")
        previous = op.get("from_previous")
        if isinstance(previous, Mapping) and previous.get("op") is not None:
            earlier = [o.get("name") for o in operations[:i] if isinstance(o, Mapping)]
            if previous.get("op") not in earlier:
                add(("operations", i, "from_previous", "op"), f"operation {previous.get('op')!r} is not an earlier "
                                                              f"operation", "reference")
    physics = data.get("physics") if isinstance(data.get("physics"), Mapping) else {}
    resolved = physics.get("resolved") if isinstance(physics.get("resolved"), Mapping) else {}
    for name in (resolved.get("operations") or {}) if isinstance(resolved.get("operations"), Mapping) else ():
        if name not in names:
            add(("physics", "resolved", "operations", name), f"physics.resolved names operation {name!r}, which "
                                                             f"the project does not have", "reference")


class ResolvedProject(SIMap):
    """The SI view of a project with its database objects, programs and input hashes (see module)."""

    path: Path
    db: Database

    @classmethod
    def load(cls, path, db: Database | None = None) -> "ResolvedProject":
        return resolve(path, db)

    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def stock_pose_W_S(self) -> np.ndarray:
        """The first operation's stock pose (S to W, 4x4)."""
        return self._data["operations"][0]["stock_pose_W_S"]

    def operation(self, name: str) -> SIMap:
        for op in self._data["operations"]:
            if op["name"] == name:
                return op
        raise KeyError(name)

    @property
    def hash(self) -> str:
        return project_hash(self)


def resolve(path_or_project, db: Database | None = None) -> ResolvedProject:
    """Load (if given a path) and resolve a project; raises InputError listing every error."""
    project = path_or_project if isinstance(path_or_project, ProjectFile) else load_project(path_or_project)
    raise_for(project.problems, f"project {project.path}")
    db = db if db is not None else Database.load()
    resolver = _Resolver(project, db)
    return resolver.run()


class _Resolver:
    def __init__(self, project: ProjectFile, db: Database):
        self.project = project
        self.db = db
        self.problems: list[Problem] = []
        self.objects: dict = {}                 # path -> SIMap of the resolved object
        self.raws: dict = {}                    # path -> raw data of the resolved object
        self.files: dict[str, Path] = {}        # name -> input file
        self.library_items: dict[str, dict] = {}
        self.data = copy.deepcopy(project.data)     # library rows join tools.list here, not in the file data
        self.locations = project.locations.copy()

    def error(self, path, message, kind="reference", severity="error", file=None, line=None):
        loc = self.locations
        self.problems.append(Problem(file or str(self.project.path), loc.line(path) if line is None else line,
                                     format_path(path), message, kind, severity, "project"))

    def run(self) -> ResolvedProject:
        project = self.project
        self.files[PROJECT_FILE] = project.path
        self._library()
        self._paths()
        refs = list(project.refs) + self._library_refs
        for kind, value, path in refs:
            self._resolve_ref(kind, value, path)
        if self.data.get("controller_overrides") and ("machine",) in self.raws:
            self._machine_overrides(self.raws[("machine",)], ("machine",))
        self._holder_checks()
        raise_for(self.problems, f"project {project.path}")
        view = self._view()
        resolved = ResolvedProject.from_entries(self.data, view._data, None)
        resolved.path = project.path
        resolved.db = self.db
        resolved.project_file = project
        resolved.problems = sort_problems(self.problems)
        resolved.objects = dict(self.objects)
        resolved.input_files = dict(sorted(self.files.items()))
        resolved.input_hashes = {name: _sha256(path) for name, path in sorted(self.files.items())}
        resolved.input_hashes[PROJECT_FILE] = project.sha256
        resolved.programs = {op["name"]: op["program_text"] for op in view["operations"]}
        return resolved

    # -------------------------------------------------------------- library and files
    def _library(self) -> None:
        self._library_refs = []
        tools = self.data.get("tools") or {}
        library = tools.get("library")
        if not library:
            return
        where = ("tools", "library")
        try:
            lib = toollib.read_library(library, self.project.dir)
        except InputError as error:
            self.error(where, str(error), "path")
            return
        self.problems.extend(lib["problems"])
        for file in lib["files"]:
            self.files[self._name(file)] = file
        self.library_items = lib["items"]
        entries = lib["entries"]
        if entries:
            listed = tools.setdefault("list", [])
            taken = {(e.get("path", 1), e.get("number")) for e in listed if isinstance(e, Mapping)}
            loc = self.locations
            library_file = str((self.project.dir / library).resolve())
            for i, entry in enumerate(entries):
                key = (entry.get("path", 1), entry["number"])
                line = lib["lines"].get(i, 0)
                if key in taken:
                    self.problems.append(Problem(library_file, line, "number", f"tool number {entry['number']} is "
                                                 f"also in tools.list", "duplicate", "error", "project"))
                    continue
                taken.add(key)
                index = len(listed)
                listed.append(entry)
                base = ("tools", "list", index)
                loc.set(base, line, library_file)
                for key_name in entry:
                    loc.set(base + (key_name,), line, library_file)
                ctx = Ctx(loc, obj="project", file=library_file)
                TOOL_ENTRY.check(entry, base, None, ctx)
                self.problems.extend(ctx.problems)
                self._library_refs.extend(ctx.refs)

    def _name(self, path: Path) -> str:
        """Input file name: relative to the project, else to the database's parent (db/...)."""
        path = Path(path).resolve()
        for base in (self.project.dir, self.db.root.parent):
            try:
                return str(path.relative_to(base))
            except ValueError:
                continue
        return str(path)

    def _paths(self) -> None:
        for path, text, node in self.project.paths:
            if path == ("tools", "library"):
                continue
            target = (self.project.dir / text).resolve() if not Path(text).is_absolute() else Path(text)
            if not target.exists():
                self.error(path, f"file {text!r} does not exist (relative to {self.project.dir})", "path")
                continue
            if node.exts and target.suffix.lower() not in node.exts:
                self.error(path, f"{text!r} is not one of {', '.join(node.exts)}", "path")
                continue
            if target.is_file():
                self.files[self._name(target)] = target

    # -------------------------------------------------------------- database references
    def _resolve_ref(self, kind: str, value, path: tuple) -> None:
        if isinstance(value, Mapping):
            self._inline(kind, value, path)
            return
        if kind == "tool" and value in self.library_items:
            self._inline(kind, self.library_items[value], path, label=f"tool {value} (library)")
            return
        if is_path_ref(value):
            self._external(kind, value, path)
            return
        try:
            obj = self.db.object(kind, value)
        except InputError as error:
            self.error(path, str(error))
            return
        errors = errors_only(obj.problems)
        if errors:
            self.error(path, f"{kind} {value!r} has errors in the database: " + "; ".join(str(e) for e in errors[:5]))
            return
        raw = obj.merged if obj.merged is not None else obj.raw
        self.objects[path] = self.db.get(kind, value)
        self.raws[path] = raw
        for file in self.db.files_of(kind, value):
            self.files[self._name(file)] = file
        self._dependencies(kind, raw)

    def _dependencies(self, kind: str, raw: Mapping) -> None:
        """Files of the objects an object names (tool materials, coatings, controllers)."""
        refs = []
        if kind == "tool":
            tm = raw.get("tool_material") if isinstance(raw.get("tool_material"), Mapping) else {}
            coating = raw.get("coating") if isinstance(raw.get("coating"), Mapping) else {}
            refs = [("tool_material", tm.get("id")), ("coating", coating.get("id"))]
        elif kind == "machine":
            controller = raw.get("controller") if isinstance(raw.get("controller"), Mapping) else {}
            refs = [("controller", controller.get("alarms"))]
        for ref_kind, ref_id in refs:
            if isinstance(ref_id, str) and (ref_kind, ref_id) in self.db.objects:
                for file in self.db.files_of(ref_kind, ref_id):
                    self.files[self._name(file)] = file

    def _machine_overrides(self, raw: Mapping, path: tuple) -> None:
        overrides = self.data.get("controller_overrides") or {}
        merged = deep_merge(raw, {"controller": overrides})
        ctx = Ctx(self.locations, obj="project", file=str(self.project.path))
        DS.MACHINE.check(merged, (), None, ctx)
        for problem in ctx.problems:
            if problem.is_error:
                self.error(("controller_overrides",), f"the machine with controller_overrides is invalid: "
                                                      f"{problem.path}: {problem.message}", "rule")
        if errors_only(ctx.problems):
            self.objects.pop(path, None)
            return
        self.objects[path] = DS.MACHINE.build(merged, None, None, merged.get("id"))
        self.raws[path] = merged

    def _inline(self, kind: str, value: Mapping, path: tuple, label: str | None = None) -> None:
        bases = self.library_items if kind == "tool" else {}
        raw = dict(value)
        base_id = raw.get("extends")
        if isinstance(base_id, str) and base_id in bases:
            raw.pop("extends")
            raw = deep_merge(bases[base_id], raw)
        built, merged, problems = self.db.inline(kind, raw, self.locations, path=path,
                                                 file=str(self.project.path), label=label or f"{kind} (inline)")
        self.problems.extend(problems)
        if built is not None:
            self.objects[path] = built
            self.raws[path] = merged
            if isinstance(base_id, str) and (kind, base_id) in self.db.objects:
                for file in self.db.files_of(kind, base_id):
                    self.files[self._name(file)] = file
            self._dependencies(kind, merged)

    def _external(self, kind: str, text: str, path: tuple) -> None:
        """A database file named by a path relative to the project: checked as a database file."""
        file = (self.project.dir / text).resolve()
        if not file.is_file():
            self.error(path, f"{kind} file {text!r} does not exist (relative to {self.project.dir})", "path")
            return
        try:
            doc = load_file(file)
        except InputError as error:
            self.error(path, str(error), "yaml")
            return
        data = doc.data
        expected = DS.FORMATS.get(kind) or DS.FORMATS.get(DS.ITEM_KINDS.get(kind, ""))
        if not isinstance(data, Mapping) or data.get("format") != expected:
            self.error(path, f"{text!r} is not a {expected} file", "format")
            return
        problems, _ = check_sources(data, doc.locations, self.db.bib, obj=f"{kind}:{file.stem}", kind=kind)
        db_kind = DS.ITEM_KINDS.get(kind, kind)
        result = migrate_db(db_kind, data, doc.locations, file=str(file), obj=f"{kind}:{file.stem}",
                            schema=DS.SCHEMAS[db_kind],
                            item_schema=DS.SCHEMAS[kind] if kind in DS.ITEM_KINDS else None)
        problems.extend(result.problems)
        migrated = result.data
        if kind in DS.ITEM_KINDS:
            items = migrated.get("items") or []
            if len(items) != 1:
                self.error(path, f"{text!r} holds {len(items)} items; name one of them by id instead", "rule")
                return
            raw = {k: migrated[k] for k in ("maker", "series", "series_id") if k in migrated}
            raw.update(items[0])
            if "src" not in raw and isinstance(migrated.get("src"), str):
                raw["src"] = migrated["src"]
            locations = result.locations.subtree(("items", 0))
        else:
            raw = migrated
            locations = result.locations
        if isinstance(raw.get("extends"), str):
            try:
                base = self.db.object(kind, raw["extends"])
            except InputError as error:
                self.error(path, str(error), "extends")
                return
            raw = deep_merge(base.merged if base.merged is not None else base.raw, raw)
            raw.pop("extends", None)
        ctx = Ctx(locations, obj=f"{kind}:{file.stem}", file=str(file))
        DS.SCHEMAS[kind].check(raw, (), None, ctx)
        problems.extend(ctx.problems)
        for ref_kind, ref_id, ref_path in ctx.refs:
            if (ref_kind, ref_id) not in self.db.objects and not (ref_kind == "bibliography" and ref_id in self.db.bib):
                problems.append(Problem(str(file), locations.line(ref_path), format_path(ref_path),
                                        f"{ref_kind} {ref_id!r} is not in the database", "reference"))
        self.problems.extend(problems)
        if errors_only(problems):
            return
        self.objects[path] = DS.SCHEMAS[kind].build(raw, None, None, raw.get("id") or file.stem)
        self.raws[path] = raw
        self.files[self._name(file)] = file
        self._dependencies(kind, raw)

    def _holder_checks(self) -> None:
        for path, raw in self.raws.items():
            if len(path) >= 1 and path[-1] == "holder" and isinstance(raw, Mapping):
                missing = [k for k in HOLDER_NEEDS if raw.get(k) is None or (isinstance(raw.get(k), Mapping)
                                                                               and raw[k].get("value") is None)]
                if missing:
                    self.error(path, f"holder {raw.get('id')!r} does not give {', '.join(missing)} (not published): "
                                     f"write it inline, holder: {{extends: {raw.get('id')}, "
                                     f"{missing[0]}: ...}}", "required")

    # -------------------------------------------------------------- the SI view
    def _view(self) -> SIMap:
        view = PROJECT.build(self.data, None, None, "project")
        for path, obj in sorted(self.objects.items(), key=lambda kv: len(kv[0])):
            if path == ("material", "id"):
                continue
            _set_in(view, path, obj)
        material = self.objects.get(("material", "id"))
        if material is not None:
            section = view["material"]
            entries = dict(material._data)
            entries["batch"] = section["batch"] if "batch" in section else None
            entries["db_id"] = material["id"] if "id" in material else None
            _set_in(view, ("material",), SIMap.from_entries(self.raws[("material", "id")], entries, material.src))
        for path, text, node in self.project.paths:
            if path == ("tools", "library"):
                _set_in(view, path, str((self.project.dir / text).resolve()))
            elif _has_path(view, path):
                _set_in(view, path, str((self.project.dir / text).resolve()))
        cad = view["cad"] if "cad" in view else None
        if isinstance(cad, SIMap) and isinstance(cad["placement"] if "placement" in cad else None, PoseMap):
            cad._data["cad_pose_S_C"] = cad["placement"].matrix()
        for i, op in enumerate(view["operations"]):
            raw_op = self.data["operations"][i]
            program = raw_op.get("program")
            programs = raw_op.get("programs") or {}
            op._data["program_path"] = str((self.project.dir / program).resolve()) if program else None
            op._data["program_text"] = _read_text(self.project.dir / program) if program else None
            op._data["program_texts"] = {path: _read_text(self.project.dir / file) for path, file in programs.items()}
            setup = op["setup"] if "setup" in op else None
            stock_pose = setup["stock_pose"] if isinstance(setup, SIMap) and "stock_pose" in setup else None
            fixture_pose = setup["fixture_pose"] if isinstance(setup, SIMap) and "fixture_pose" in setup else None
            op._data["stock_pose_W_S"] = stock_pose.matrix() if isinstance(stock_pose, PoseMap) else np.eye(4)
            op._data["fixture_pose_F_H"] = fixture_pose.matrix() if isinstance(fixture_pose, PoseMap) else np.eye(4)
        _set_in(view, ("physics",), physics_with_defaults(self.data.get("physics")))
        return view


def physics_defaults() -> dict:
    """The physics settings a project does not write (db/_templates/project.yaml ``physics``: the
    architecture 5.1 rules and the documented defaults), without ``resolved``."""
    template = Path(DS_TEMPLATE)
    if not template.is_file():
        return {"tick_s": TICK_S}
    data = load_file(template).data
    physics = dict(data.get("physics") or {}) if isinstance(data, Mapping) else {}
    physics.pop("resolved", None)
    return physics


def physics_with_defaults(physics) -> SIMap:
    """SI view of a project's physics section over the documented defaults."""
    from .schema import PHYSICS
    merged = deep_merge(physics_defaults(), physics or {})
    merged["tick_s"] = TICK_S
    return PHYSICS.build(merged, None, None, "physics")


def _read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise InputError(f"cannot read program: {error.strerror}", str(path)) from None


def _has_path(view, path) -> bool:
    node = view
    for part in path:
        if isinstance(node, SIMap) and part in node:
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
        else:
            return False
    return True


def _set_in(view: SIMap, path: tuple, value) -> None:
    node = view
    for part in path[:-1]:
        node = node[part]
    last = path[-1]
    if isinstance(node, SIMap):
        node._data[last] = value
    elif isinstance(node, list):
        node[last] = value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def project_hash(project) -> str:
    """sha256 of the project's content (migrated data, comments and formatting ignored) and of
    every other input file (programs, CAD, tool library, database and model files it uses)."""
    if isinstance(project, (str, Path)):
        project = resolve(project)
    if isinstance(project, ResolvedProject):
        data = project.project_file.data
        inputs = {k: v for k, v in project.input_hashes.items() if k != PROJECT_FILE}
    else:
        data = project.data
        inputs = {}
    models = DSModels.hashes()
    canonical = json.dumps({"format": FORMAT, "project": data, "inputs": inputs, "models": models},
                           sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DSModels:
    """Hashes of the model parameter files (inputs of every run)."""

    @staticmethod
    def hashes() -> dict:
        from ..db.models import ROOT
        folder = Path(ROOT) / "models"
        if not folder.is_dir():
            return {}
        return {f"db/models/{p.name}": _sha256(p) for p in sorted(folder.glob("*.yaml"))}


__all__ = ["load_project", "resolve", "ResolvedProject", "ProjectFile", "project_hash", "project_path",
           "PROJECT_FILE", "HOLDER_NEEDS"]
