"""Load, validate and resolve a project (docs/architecture.md 18, docs/formats.md "Project file").

* :func:`load_project` reads ``project.yaml`` (or a project directory) with line numbers, migrates
  older versions forward (:mod:`cnctwin.project.migrate`) and checks it against the declared schema
  (:mod:`cnctwin.project.schema`). Every problem names ``file:line`` and the key path.
* :func:`resolve` (and :meth:`ResolvedProject.load`) also resolves every database reference through
  :class:`cnctwin.db.Database` (an id, a database file relative to the project, or an inline object
  in place of the id), the tool library (CSV rows join ``tools.list``, ISO 13399 items can be named
  or extended), checks that every file exists, reads the programs, applies
  ``controller_overrides`` to the machine and the calibration overlay the project includes
  (``calibration: <file>``, cnctwin.mdata.overlay) to the tool list, the machine and the material, and
  hashes every input file (the overlay included).

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
import math
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
from .schema import FORMAT, PROJECT, RESOLVED_PROBE, TICK_S, TOOL_ENTRY, TOOL_LIST_ENTRY, is_path_ref

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
        self.overlay: dict | None = None            # the included calibration overlay (calibration: <file>)
        self.overlay_path: Path | None = None

    def error(self, path, message, kind="reference", severity="error", file=None, line=None):
        loc = self.locations
        self.problems.append(Problem(file or str(self.project.path), loc.line(path) if line is None else line,
                                     format_path(path), message, kind, severity, "project"))

    def run(self) -> ResolvedProject:
        project = self.project
        self.files[PROJECT_FILE] = project.path
        self._library()
        self._paths()
        self._calibration()
        refs = list(project.refs) + self._library_refs
        for kind, value, path in refs:
            self._resolve_ref(kind, value, path)
        if ("machine",) in self.raws and (self.data.get("controller_overrides") or self._calibrates("machine")):
            self._machine_overrides(self.raws[("machine",)], ("machine",))
        if ("material", "id") in self.raws and self._calibrates("material"):
            self._calibrated_material(self.raws[("material", "id")], ("material", "id"))
        self._holder_checks()
        calibration = self._calibration_record()
        raise_for(self.problems, f"project {project.path}")
        view = self._view()
        raise_for(self.problems, f"project {project.path}")
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
        resolved.calibration_overlay = calibration
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
        """The machine with the project's ``controller_overrides`` deep-merged over its controller and then the
        calibration overlay's ``machine.*`` entries set (so a calibrated controller value wins over an override);
        the result is checked as a machine and becomes the project's machine."""
        overrides = self.data.get("controller_overrides") or {}
        merged = deep_merge(raw, {"controller": overrides}) if overrides else copy.deepcopy(dict(raw))
        what = "controller_overrides" if overrides else ""
        if self._calibrates("machine"):
            from ..mdata import overlay
            try:
                merged = overlay.apply_to_raw("machine", merged, self.overlay)
            except InputError as error:
                self.error(("calibration",), f"calibration overlay: {error}", "calibration")
                self.objects.pop(path, None)
                return
            what = f"{what} and the calibration overlay" if what else "the calibration overlay"
        ctx = Ctx(self.locations, obj="project", file=str(self.project.path))
        DS.MACHINE.check(merged, (), None, ctx)
        for problem in ctx.problems:
            if problem.is_error:
                self.error(("controller_overrides",) if overrides else ("calibration",),
                           f"the machine with {what} is invalid: {problem.path}: {problem.message}", "rule")
        if errors_only(ctx.problems):
            self.objects.pop(path, None)
            return
        self.objects[path] = DS.MACHINE.build(merged, None, None, merged.get("id"))
        self.raws[path] = merged

    # -------------------------------------------------------------- the calibration overlay
    def _calibration(self) -> None:
        """``calibration: <file>``: the overlay (format cnctwin-calibration, cnctwin.mdata.overlay) is loaded and
        checked, and its ``tools[...]`` entries are set in the tool list (after the library rows joined it; each
        changed entry is checked again). Its ``machine.*`` and ``material.*`` entries are set in the resolved raws
        (:meth:`_machine_overrides`, :meth:`_calibrated_material`); the file itself is an input (``_paths`` hashed
        it with the other files)."""
        self.overlay = None
        self.overlay_path = None
        text = self.data.get("calibration")
        if not isinstance(text, str) or not text:
            return
        where = ("calibration",)
        path = Path(text) if Path(text).is_absolute() else (self.project.dir / text).resolve()
        if not path.is_file():
            return                      # reported by _paths
        from ..mdata import overlay
        try:
            document = overlay.load_overlay(path)
        except InputError as error:
            self.error(where, f"calibration overlay {text!r}: {error}", "calibration")
            return
        try:
            patched = overlay.apply_to_tools(self.data, document)
        except InputError as error:
            self.error(where, f"calibration overlay {text!r}: {error}", "calibration")
            return
        before = (self.data.get("tools") or {}).get("list") or []
        after = (patched.get("tools") or {}).get("list") or []
        for i, entry in enumerate(after):
            if i < len(before) and entry == before[i]:
                continue
            ctx = Ctx(self.locations, obj="project", file=str(self.project.path))
            TOOL_LIST_ENTRY.check(entry, ("tools", "list", i), None, ctx)
            for problem in ctx.problems:
                if problem.is_error:
                    self.error(where, f"calibration overlay {text!r} makes tools.list[{i}] invalid: {problem.path}: "
                                      f"{problem.message}", "calibration")
        if "tools" in patched:
            self.data["tools"] = patched["tools"]
        self.overlay = document
        self.overlay_path = path

    def _calibrates(self, kind: str) -> bool:
        if self.overlay is None:
            return False
        from ..mdata import overlay
        return bool(overlay.entries_for(self.overlay, kind))

    def _calibrated_material(self, raw: Mapping, path: tuple) -> None:
        """The material with the overlay's ``material.*`` entries set, checked as a material."""
        from ..mdata import overlay
        try:
            patched = overlay.apply_to_raw("material", raw, self.overlay)
        except InputError as error:
            self.error(("calibration",), f"calibration overlay: {error}", "calibration")
            self.objects.pop(path, None)
            return
        schema = DS.SCHEMAS["material"]
        ctx = Ctx(self.locations, obj="project", file=str(self.project.path))
        schema.check(patched, (), None, ctx)
        for problem in ctx.problems:
            if problem.is_error:
                self.error(("calibration",), f"the material with the calibration overlay is invalid: {problem.path}: "
                                             f"{problem.message}", "rule")
        if errors_only(ctx.problems):
            self.objects.pop(path, None)
            return
        self.objects[path] = schema.build(patched, None, None, patched.get("id"))
        self.raws[path] = patched

    def _calibration_record(self) -> dict | None:
        """``ResolvedProject.calibration_overlay``: what the included overlay set and where its other entries are
        (None without ``calibration``). ``models`` ({model: {parameter: SI value}}) and ``factors`` ({name: {key:
        value}}) have no database object to land in: they are listed here for the domains that read them, and
        each is reported as a warning while no domain applies it."""
        if self.overlay is None:
            return None
        from ..mdata import overlay
        document = self.overlay
        models = overlay.model_overrides(document)
        factors = overlay.factors(document)
        applied = sorted(path for root in ("machine", "material", "tools")
                         for path, _, _ in overlay.entries_for(document, root))
        record = {"file": str(self.overlay_path), "name": document.get("name"), "machine": document.get("machine"),
                  "material": document.get("material"), "applied": applied,
                  "set": {path: dict(document["set"][path]) for path in applied}, "models": models,
                  "factors": factors}
        for path, _, _ in overlay.entries_for(document, "models") + overlay.entries_for(document, "factors"):
            self.problems.append(Problem(str(self.project.path), self.locations.line(("calibration",)), "calibration",
                                         f"calibration overlay entry {path} is not applied by the simulation in this "
                                         f"build (listed in project.calibration_overlay)", "calibration", "warning",
                                         "project"))
        return record

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
        self._stock_frames(view)
        self._probes(view)
        _set_in(view, ("physics",), physics_with_defaults(self.data.get("physics")))
        return view

    # -------------------------------------------------------------- stock frames of the operations
    def _stock_frames(self, view) -> None:
        """Each operation's stock frame S (docs/formats.md "Project file", from_previous): the first
        operation's S is the stock's own; an operation with ``from_previous`` turns the previous operation's
        part (``flip`` about an S axis, ``transfer: sub_spindle`` half a turn about S y) and re-bases the turned
        stock box by the stock conventions (cnctwin.removal.transfer, the frame the material is carried in);
        without ``from_previous`` the part stays in the previous operation's S. ``stock_pose`` places that S
        in the workholding frame (the extra placement: it must not repeat the turn)."""
        ops = view["operations"]
        stock = view["stock"] if "stock" in view else None
        kind = str(stock["kind"]) if isinstance(stock, SIMap) and "kind" in stock else "block"
        needs_box = any(transfer_rotation(raw.get("from_previous")) is not None
                        for raw in self.data.get("operations") or [] if isinstance(raw, Mapping))
        box = stock_box_S(stock, self.project.dir, read_mesh=needs_box, problems=self, path=("stock",))
        frames: dict[str, tuple[np.ndarray, tuple | None]] = {}
        pose, current = np.eye(4), box
        for i, op in enumerate(ops):
            raw_op = self.data["operations"][i]
            previous = raw_op.get("from_previous") if isinstance(raw_op, Mapping) else None
            transfer = np.eye(4)
            if isinstance(previous, Mapping) and previous.get("op") in frames:
                pose, current = frames[previous["op"]]
                rotation = transfer_rotation(previous)
                if rotation is not None and current is not None:
                    lo, hi, t = turned_box(current[0], current[1], rotation, kind)
                    transfer[:3, :3] = rotation
                    transfer[:3, 3] = t
                    current = (lo, hi)
                    self._check_turn(op, raw_op, i, rotation, transfer)
                elif rotation is not None:
                    transfer = None
            pose = transfer @ pose if transfer is not None else None
            op._data["transfer_S_Sprev"] = transfer
            op._data["pose_S_S0"] = pose
            op._data["stock_box_S_m"] = [list(map(float, current[0])), list(map(float, current[1]))] \
                if current is not None else None
            op._data["stock_pose_W_S0"] = op["stock_pose_W_S"] @ pose if pose is not None else None
            frames[str(op["name"])] = (pose if pose is not None else np.eye(4), current)
            if pose is None:
                pose = np.eye(4)

    def _check_turn(self, op, raw_op, index: int, rotation: np.ndarray, transfer: np.ndarray) -> None:
        """A stock_pose whose rotation repeats or undoes the turn of from_previous is refused: the part
        would be turned twice, or not at all (the turn is already in this operation's stock frame)."""
        written = np.asarray(op["stock_pose_W_S"], dtype=float)
        r_sp = written[:3, :3]
        if np.allclose(rotation, np.eye(3), atol=1e-9):
            return
        repeats = np.allclose(r_sp, rotation, atol=1e-9)
        undoes = np.allclose(r_sp, rotation.T, atol=1e-9)
        if not (repeats or undoes):
            return
        meant = written @ np.linalg.inv(transfer)
        translate = [round(float(v) * 1e3, 6) + 0.0 for v in meant[:3, 3]]
        rotate = [round(math.degrees(a), 6) + 0.0 for a in euler_xyz(meant[:3, :3])]
        what = raw_op.get("from_previous") or {}
        turn = (f"flip {what['flip'].get('axis')} {what['flip'].get('angle_deg')} deg" if isinstance(what.get("flip"),
                Mapping) else f"transfer {what.get('transfer')}")
        self.error(("operations", index, "setup", "stock_pose"),
                   f"stock_pose {'repeats' if repeats else 'undoes'} the from_previous turn ({turn}): with "
                   f"from_previous the turn is already in this operation's stock frame and stock_pose is only the "
                   f"extra placement; if the written pose was the complete one, write stock_pose: "
                   f"{{translate_mm: {translate}, rotate_deg: {rotate}}}", "rule")

    # -------------------------------------------------------------- probes in tools.list
    def _probes(self, view) -> None:
        """``tools.list`` entries with ``kind: probe`` become ``tools.probes`` (the probe definition resolved
        against the machine's ``probes`` section); ``tools.list`` keeps the cutting tools, whose index is
        the tool-table slot every domain uses."""
        tools = view["tools"] if "tools" in view else None
        if not isinstance(tools, SIMap):
            return
        entries = list(tools["list"] or []) if "list" in tools else []
        raw_entries = (self.data.get("tools") or {}).get("list") or []
        machine = self.raws.get(("machine",)) or {}
        machine_probes = machine.get("probes") if isinstance(machine, Mapping) else None
        cutting, probes = [], []
        for i, entry in enumerate(entries):
            raw = raw_entries[i] if i < len(raw_entries) else {}
            if not (isinstance(raw, Mapping) and raw.get("kind") == "probe"):
                cutting.append(entry)
                continue
            where = ("tools", "list", i, "probe")
            spec = raw.get("probe")
            name = spec if isinstance(spec, str) else (spec.get("extends") if isinstance(spec, Mapping) else None)
            inline = {k: v for k, v in spec.items() if k != "extends"} if isinstance(spec, Mapping) else {}
            base = {}
            if name is not None:
                found = machine_probes.get(name) if isinstance(machine_probes, Mapping) else None
                if not isinstance(found, Mapping):
                    known = sorted(k for k, v in (machine_probes or {}).items() if isinstance(v, Mapping))
                    self.error(where, f"the machine has no probe {name!r} (probes: {', '.join(known) or 'none'})")
                    continue
                base = {k: v for k, v in found.items() if k in RESOLVED_PROBE.fields or k in ("src", "note", "notes")}
            merged = deep_merge(base, inline)
            if merged.get("stylus_ball_mm") is None:
                self.error(where, f"probe T{raw.get('number')} needs the stylus ball: the machine's {name!r} does not "
                                  f"give stylus_ball_mm, write probe: {{extends: {name}, stylus_ball_mm: ...}}"
                           if name else f"probe T{raw.get('number')} needs stylus_ball_mm", "required")
                continue
            ctx = Ctx(self.locations, obj="project", file=str(self.project.path))
            RESOLVED_PROBE.check(merged, where, None, ctx)
            self.problems.extend(ctx.problems)
            if errors_only(ctx.problems):
                continue
            built = RESOLVED_PROBE.build(merged, None, "project" if not base else None, "probe")
            built._data["name"] = name
            entry._data["probe"] = built
            entry._data["list_index"] = i
            probes.append(entry)
        tools._data["list"] = cutting
        tools._data["probes"] = probes


AXES = {"X": 0, "Y": 1, "Z": 2}


def _rotation(axis: int, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    rotation[i, i], rotation[i, j], rotation[j, i], rotation[j, j] = c, -s, s, c
    return rotation


def transfer_rotation(from_previous) -> np.ndarray | None:
    """3x3 turn of the part by ``from_previous`` (None: the part keeps its orientation), as
    cnctwin.removal.transfer turns the material: ``flip {axis, angle_deg}`` about that S axis, ``transfer:
    sub_spindle`` half a turn about S y; ``main_spindle``, ``pallet``, ``manual`` keep the orientation."""
    if not isinstance(from_previous, Mapping):
        return None
    flip = from_previous.get("flip")
    if isinstance(flip, Mapping) and flip.get("axis") in AXES and flip.get("angle_deg") is not None:
        angle = flip["angle_deg"]
        angle = angle.get("value") if isinstance(angle, Mapping) else angle
        return _rotation(AXES[flip["axis"]], math.radians(float(angle)))
    if from_previous.get("transfer") == "sub_spindle":
        return _rotation(1, math.pi)
    return None


def turned_box(lo, hi, rotation: np.ndarray, kind: str):
    """(lo, hi, t) of the stock box turned by ``rotation`` and placed by the stock conventions,
    x_new = R x + t: a block's box starts at the S origin; a cylinder or tube whose axis stays on z keeps it
    on S z with z from 0 (cnctwin.removal.transfer.new_box, the frame the material is carried in)."""
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    turned = corners @ rotation.T
    keeps_axis = abs(abs(rotation[2, 2]) - 1.0) < 1e-9 and abs(rotation[0, 2]) < 1e-9 and abs(rotation[1, 2]) < 1e-9
    if kind in ("cylinder", "tube") and keeps_axis:
        t = np.array([0.0, 0.0, -turned[:, 2].min()])
    else:
        t = -turned.min(axis=0)
    moved = turned + t
    return moved.min(axis=0), moved.max(axis=0), t


def stock_box_S(stock, project_dir: Path, *, read_mesh: bool = False, problems=None, path=("stock",)):
    """(lo, hi) of the stock box in its own frame S (SI): a block from the S origin, a cylinder or tube
    about +z from z = 0; a mesh stock its bounds (read only when ``read_mesh``). None when unknown."""
    if not isinstance(stock, SIMap):
        return None
    kind = str(stock["kind"]) if "kind" in stock else "block"
    size = stock["size_m"] if "size_m" in stock else None
    values = [float(v) for v in size] if isinstance(size, (list, tuple)) else None
    if kind == "block" and values is not None and len(values) == 3:
        return (np.zeros(3), np.array(values))
    if kind in ("cylinder", "tube") and values is not None and len(values) >= 2:
        r = 0.5 * values[0]
        return (np.array([-r, -r, 0.0]), np.array([r, r, values[-1]]))
    mesh_file = stock["mesh"] if "mesh" in stock else None
    if kind == "mesh" and read_mesh and mesh_file:
        try:
            from ..geom.io import read_mesh as read
            mesh = read(Path(str(mesh_file)), unit="mm")
            vertices = np.asarray(mesh.vertices, dtype=float)
            return (vertices.min(axis=0), vertices.max(axis=0))
        except Exception as error:      # noqa: BLE001  (any reader failure is reported as a problem)
            if problems is not None:
                problems.error(path + ("mesh",), f"the stock mesh could not be read to place the turned stock of "
                                                 f"from_previous: {error}", "path")
    return None


def euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
    """(a, b, c) rad with R = Rx(a) Ry(b) Rz(c) (the project's pose order xyz, cnctwin.db.simap.pose_matrix)."""
    r = np.asarray(rotation, dtype=float)
    b = math.asin(max(-1.0, min(1.0, r[0, 2])))
    if abs(abs(r[0, 2]) - 1.0) < 1e-12:
        return (math.atan2(r[2, 1], r[1, 1]), b, 0.0)
    return (math.atan2(-r[1, 2], r[2, 2]), b, math.atan2(-r[0, 1], r[0, 0]))


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


__all__ = ["load_project", "resolve", "ResolvedProject", "ProjectFile", "project_hash", "project_path", "transfer_rotation",
           "turned_box", "stock_box_S", "euler_xyz",
           "PROJECT_FILE", "HOLDER_NEEDS"]
