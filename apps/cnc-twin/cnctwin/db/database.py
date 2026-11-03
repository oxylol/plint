"""The database: every file under db/ (and validation/cases/), validated, as SI objects.

``Database.load(root)`` reads every ``db/<folder>/**/*.yaml`` (folders in
:data:`cnctwin.db.schemas.FOLDERS`; ``_templates`` and other folders are not database files) and
``validation/cases/**/*.yaml``, then for each file:

1. checks the header (``format`` names the kind of its folder, ``version`` is supported, the id is
   the file name);
2. checks the source rules on the file as written (:mod:`cnctwin.db.sources`) and collects the
   estimated values;
3. migrates older key forms forward (:mod:`cnctwin.db.migrate`, each change recorded);
4. checks the migrated file against its declared schema (:mod:`cnctwin.db.schemas`); series files
   (tools, holders, workholding) are checked per item, with the series ``defaults`` applied and
   the series ``maker``, ``series``, ``series_id`` and ``src`` carried into every item;
5. resolves ``extends: <id>`` by deep merge onto the base object (chains allowed, cycles and
   missing bases are errors) and checks the merged object against the full schema;
6. checks every reference (tool materials, coatings, controllers, materials, bibliography keys).

Problems never stop the load: each object keeps its own problems, accessing an object with
errors raises them (:class:`~cnctwin.core.errors.InputError`, or
:class:`~cnctwin.core.errors.SourceError` when all are source problems), and
:meth:`Database.validate_all` lists everything. ``strict=True`` raises at load on any error and
treats an ``est`` without a note as an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from ..core.errors import InputError
from . import schemas as S
from .merge import deep_merge, merge_pinned, pin_sources
from .migrate import migrate
from .problems import Estimate, Migration, Problem, errors_only, format_path, raise_for, sort_problems
from .schema import Ctx
from .simap import SIMap
from .sources import Bibliography, check_reference, check_sources
from .yamlio import Doc, Locations, load_file

try:
    from ..paths import DB as _DB_ROOT
except ImportError:  # pragma: no cover - paths.py is optional
    _DB_ROOT = Path(__file__).resolve().parents[2] / "db"

DEFAULT_ROOT = Path(_DB_ROOT)
KINDS = ("material", "tool", "holder", "tool_material", "coating", "machine", "coolant", "workholding",
         "controller", "model", "bibliography", "validation_case", "tool_series", "holder_series",
         "workholding_series")
# A dangling reference of these kinds is a warning (the object is still usable): a machine whose
# controller reference file is missing has no alarm texts, but every value it needs is its own.
SOFT_REFERENCES = ("controller",)


@dataclass
class Obj:
    """One database object (a file, or one item of a series file)."""

    kind: str
    id: str
    file: Path
    rel: str
    raw: dict                      # migrated (series defaults and header applied), before extends
    locations: Locations
    item_path: tuple = ()
    extends: str | None = None
    problems: list[Problem] = field(default_factory=list)
    merged: dict | None = None
    chain: list = field(default_factory=list)   # files of the extends chain, base last
    simap: SIMap | None = None
    refs: list = field(default_factory=list)    # (kind, id, path) references found by the schema

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.id)

    def errors(self) -> list[Problem]:
        return errors_only(self.problems)


@dataclass
class FileRecord:
    path: Path
    rel: str
    kind: str | None
    doc: Doc | None
    problems: list[Problem] = field(default_factory=list)
    estimates: list[Estimate] = field(default_factory=list)
    migrations: list[Migration] = field(default_factory=list)
    objects: list[tuple[str, str]] = field(default_factory=list)


class ModelParams:
    """``db.models.params(name)``: the parameter values of ``db/models/<name>.yaml``."""

    def __init__(self, database: "Database"):
        self._db = database

    def params(self, name: str) -> dict:
        """name -> SI value (converted from each parameter's written unit; see cnctwin.db.models)."""
        from .models import _to_si
        return {key: _to_si(entry) for key, entry in self.records(name).items()}

    def records(self, name: str) -> dict:
        """name -> {value, unit, src, note, range} as written."""
        obj = self._db.object("model", name)
        raise_for(obj.problems, f"model parameters {name!r}")
        out = {}
        for key, entry in (obj.raw.get("params") or {}).items():
            out[key] = dict(entry)
        return out

    def names(self) -> list[str]:
        return self._db.ids("model")


class Database:
    """All database objects, validated, with SI views (see the module docstring)."""

    def __init__(self, root: Path, validation_root: Path | None, strict: bool):
        self.root = Path(root)
        self.validation_root = Path(validation_root) if validation_root is not None else None
        self.strict = strict
        self.bib = Bibliography()
        self.files: list[FileRecord] = []
        self.objects: dict[tuple[str, str], Obj] = {}
        self.general: list[Problem] = []
        self.models = ModelParams(self)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, root=DEFAULT_ROOT, *, validation_root=None, strict: bool = False,
             raise_on_error: bool | None = None) -> "Database":
        """Read and check every file. ``strict``: an ``est`` without a note is an error, and any error
        raises (``raise_on_error=False`` keeps the strict rules but returns the database)."""
        root = Path(root)
        if validation_root is None:
            candidate = root.parent / "validation" / "cases"
            validation_root = candidate if candidate.is_dir() else None
        db = cls(root, validation_root, strict)
        db._load()
        if strict if raise_on_error is None else raise_on_error:
            raise_for(db.validate_all(), f"database {root}")
        return db

    def _rel(self, path: Path) -> str:
        for base in (self.root.parent, self.root):
            try:
                return str(path.relative_to(base))
            except ValueError:
                continue
        return str(path)

    def _discover(self) -> list[tuple[str, Path]]:
        found: list[tuple[str, Path]] = []
        if not self.root.is_dir():
            self.general.append(Problem(str(self.root), 0, "", "database root does not exist", "format"))
            return found
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if folder.name.startswith(("_", ".")):
                continue
            kind = S.KIND_OF_FOLDER.get(folder.name)
            if kind is None:
                self.general.append(Problem(self._rel(folder), 0, "", f"folder {folder.name!r} is not a database "
                                            f"kind ({', '.join(sorted(S.KIND_OF_FOLDER))})", "format", "warning"))
                continue
            for path in sorted(folder.rglob("*.yaml")) + sorted(folder.rglob("*.yml")):
                found.append((kind, path))
        if self.validation_root is not None and self.validation_root.is_dir():
            for path in sorted(self.validation_root.rglob("*.yaml")) + sorted(self.validation_root.rglob("*.yml")):
                found.append(("validation_case", path))
        return found

    def _load(self) -> None:
        found = self._discover()
        parsed: list[FileRecord] = []
        for kind, path in found:
            rel = self._rel(path)
            record = FileRecord(path, rel, kind, None)
            try:
                record.doc = load_file(path)
                record.doc.locations.file = rel
                record.doc.locations.entries = {k: (rel, line) for k, (_, line) in record.doc.locations.entries.items()}
            except InputError as error:
                record.problems.append(Problem(rel, error.line, "", str(error).split(": ", 1)[-1]
                                               if error.line else str(error), "yaml"))
            parsed.append(record)
            self.files.append(record)
        for record in parsed:
            if record.kind == "bibliography" and record.doc is not None:
                self._load_bibliography(record)
        for record in parsed:
            if record.kind != "bibliography" and record.doc is not None:
                self._load_file(record)
        self._resolve_extends()
        self._check_references()

    def _header(self, record: FileRecord, data, expected_kind: str) -> bool:
        if not isinstance(data, Mapping):
            record.problems.append(Problem(record.rel, 1, "", "a database file is a YAML mapping", "format"))
            return False
        fmt = data.get("format")
        kind = S.KIND_OF_FORMAT.get(fmt) if isinstance(fmt, str) else None
        if kind is None:
            record.problems.append(Problem(record.rel, record.doc.locations.line(("format",)), "format",
                                           f"format {fmt!r} is not a database format "
                                           f"(expected {S.FORMATS[expected_kind]})", "format"))
            return False
        if kind != expected_kind:
            record.problems.append(Problem(record.rel, record.doc.locations.line(("format",)), "format",
                                           f"a {fmt} file belongs in db/{S.FOLDERS.get(kind, kind)}/, not here",
                                           "format"))
            return False
        version = data.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1 \
                or version > S.CURRENT_VERSION[kind]:
            record.problems.append(Problem(record.rel, record.doc.locations.line(("version",)), "version",
                                           f"version {version!r} is not supported (current "
                                           f"{S.CURRENT_VERSION[kind]})", "migration"))
            return False
        return True

    def _load_bibliography(self, record: FileRecord) -> None:
        data = record.doc.data
        if not self._header(record, data, "bibliography"):
            return
        ctx = Ctx(record.doc.locations, obj=f"bibliography:{record.path.stem}", file=record.rel)
        S.BIBLIOGRAPHY.check(data, (), None, ctx)
        record.problems.extend(ctx.problems)
        entries = data.get("entries")
        if isinstance(entries, Mapping):
            for key, entry in entries.items():
                if isinstance(entry, Mapping):
                    self.bib.add(str(key), dict(entry), record.rel, record.doc.locations.line(("entries", key)))
                    obj = Obj("bibliography", str(key), record.path, record.rel, dict(entry),
                              record.doc.locations.subtree(("entries", key)))
                    self.objects.setdefault(obj.key, obj)
        record.problems.extend(self.bib.problems)
        self.bib.problems = []

    def _load_file(self, record: FileRecord) -> None:
        doc = record.doc
        data = doc.data
        if not self._header(record, data, record.kind):
            return
        kind = record.kind
        obj_name = f"{kind}:{record.path.stem}"
        if kind == "validation_case":
            reference = data.get("reference") if isinstance(data.get("reference"), Mapping) else {}
            record.problems.extend(check_reference(reference.get("src"), ("reference", "src"), doc.locations,
                                                   self.bib, obj=obj_name))
        else:
            problems, estimates = check_sources(data, doc.locations, self.bib, obj=obj_name, strict=self.strict,
                                                kind=kind)
            record.problems.extend(problems)
            record.estimates.extend(estimates)
        schema = S.SCHEMAS[kind]
        item_schema = S.SCHEMAS[S.SERIES_KINDS[kind]] if kind in S.SERIES_KINDS else None
        result = migrate(kind, data, doc.locations, file=record.rel, obj=obj_name, schema=schema,
                         item_schema=item_schema)
        record.migrations.extend(result.migrations)
        record.problems.extend(result.problems)
        migrated, locations = result.data, result.locations
        if kind in S.SERIES_KINDS:
            self._load_series(record, migrated, locations, schema)
            return
        id_key = "model" if kind == "model" else "id"
        obj_id = migrated.get(id_key)
        if isinstance(obj_id, str) and obj_id != record.path.stem:
            record.problems.append(Problem(record.rel, locations.line((id_key,)), id_key,
                                           f"{id_key} {obj_id!r} must be the file name ({record.path.stem!r})",
                                           "format"))
        if not isinstance(obj_id, str):
            obj_id = record.path.stem
        extends = migrated.get("extends") if isinstance(migrated.get("extends"), str) else None
        ctx = Ctx(locations, obj=f"{kind}:{obj_id}", file=record.rel)
        schema.check(migrated, (), None, ctx)
        if extends:
            ctx.problems = partial_problems(ctx.problems)
            ctx.refs = []
        if kind == "model" and isinstance(migrated.get("params"), Mapping):
            from .models import _check_units
            for key, entry in migrated["params"].items():
                if isinstance(entry, Mapping):
                    _check_units(str(key), entry, ("params", key), ctx)
        obj = Obj(kind, obj_id, record.path, record.rel, migrated, locations, (), extends)
        obj.problems.extend(record.problems)
        obj.problems.extend(ctx.problems)
        obj.refs = list(ctx.refs)
        self._register(record, obj)

    def _load_series(self, record: FileRecord, data, locations: Locations, schema) -> None:
        kind = S.SERIES_KINDS[record.kind]
        series_id = data.get("series_id")
        if isinstance(series_id, str) and series_id != record.path.stem:
            record.problems.append(Problem(record.rel, locations.line(("series_id",)), "series_id",
                                           f"series_id {series_id!r} must be the file name ({record.path.stem!r})",
                                           "format"))
        ctx = Ctx(locations, obj=f"{record.kind}:{record.path.stem}", file=record.rel)
        schema.check(data, (), None, ctx)
        record.problems.extend(ctx.problems)
        items = data.get("items") if isinstance(data.get("items"), list) else []
        defaults = data.get("defaults") if isinstance(data.get("defaults"), Mapping) else {}
        series_src = data.get("src") if isinstance(data.get("src"), str) else None
        header = {k: data[k] for k in ("maker", "series", "series_id") if isinstance(data.get(k), str)}
        item_schema = S.SCHEMAS[kind]
        for index, item in enumerate(items):
            path = ("items", index)
            if not isinstance(item, Mapping):
                record.problems.append(Problem(record.rel, locations.line(path), format_path(path),
                                               "an item is a mapping", "type"))
                continue
            raw = dict(item)
            if defaults:
                raw = deep_merge(pin_sources(defaults, series_src), raw)
            for key, value in header.items():
                raw.setdefault(key, value)
            if series_src is not None and "src" not in raw:
                raw["src"] = series_src
            item_id = raw.get("id") if isinstance(raw.get("id"), str) else f"{record.path.stem}#{index}"
            sub = locations.subtree(path)
            extends = raw.get("extends") if isinstance(raw.get("extends"), str) else None
            ictx = Ctx(sub, obj=f"{kind}:{item_id}", file=record.rel)
            item_schema.check(raw, (), None, ictx)
            if extends:
                ictx.problems = partial_problems(ictx.problems)
                ictx.refs = []
            obj = Obj(kind, item_id, record.path, record.rel, raw, sub, path, extends)
            prefix = format_path(path)
            obj.problems.extend(p for p in record.problems
                                if not p.path.startswith("items[") or p.path == prefix
                                or p.path.startswith(prefix + ".") or p.path.startswith(prefix + "["))
            obj.problems.extend(_reroot(ictx.problems, path))
            obj.refs = [(k, i, path + p) for k, i, p in ictx.refs]
            self._register(record, obj)

    def _register(self, record: FileRecord, obj: Obj) -> None:
        existing = self.objects.get(obj.key)
        if existing is not None:
            problem = Problem(obj.rel, obj.locations.line(("id",)), "id",
                              f"{obj.kind} id {obj.id!r} is also defined in {existing.rel}", "duplicate",
                              obj=f"{obj.kind}:{obj.id}")
            record.problems.append(problem)
            existing.problems.append(problem)
            return
        self.objects[obj.key] = obj
        record.objects.append(obj.key)

    def _resolve_extends(self) -> None:
        for obj in list(self.objects.values()):
            if obj.extends is None:
                continue
            chain, problem = self._chain(obj)
            if problem is not None:
                obj.problems.append(problem)
                continue
            merged = chain[-1].raw
            locations = chain[-1].locations
            for base in reversed(chain[:-1]):
                merged = merge_pinned(merged, base.raw)
                locations = locations.overlay(base.locations)
            merged.pop("extends", None)
            obj.merged = merged
            obj.chain = [o.file for o in chain]
            node = S.SCHEMAS[obj.kind]
            ctx = Ctx(locations, obj=f"{obj.kind}:{obj.id}", file=obj.rel)
            node.check(merged, (), None, ctx)
            seen = {(p.path, p.message) for p in obj.problems}
            obj.problems.extend(p for p in ctx.problems if (p.path, p.message) not in seen)
            obj.refs = list(obj.refs) + list(ctx.refs)

    def _chain(self, obj: Obj):
        """[obj, base, base of base, ...]; or a problem (missing base, cycle)."""
        chain = [obj]
        seen = {obj.id}
        current = obj
        while current.extends is not None:
            base = self.objects.get((obj.kind, current.extends))
            line = current.locations.line(("extends",))
            if base is None:
                return chain, Problem(current.rel, line, "extends", f"extends {current.extends!r}: no {obj.kind} "
                                      f"with that id", "extends", obj=f"{obj.kind}:{obj.id}")
            if base.id in seen:
                return chain, Problem(current.rel, line, "extends", f"extends chain is circular "
                                      f"({' -> '.join(o.id for o in chain)} -> {base.id})", "extends",
                                      obj=f"{obj.kind}:{obj.id}")
            if base.errors():
                obj.problems.append(Problem(current.rel, line, "extends", f"the base {obj.kind} {base.id!r} has "
                                            f"errors", "extends", obj=f"{obj.kind}:{obj.id}"))
            seen.add(base.id)
            chain.append(base)
            current = base
        return chain, None

    def _check_references(self) -> None:
        for obj in self.objects.values():
            for kind, ref_id, path in obj.refs:
                if kind == "bibliography":
                    exists = ref_id in self.bib
                else:
                    exists = (kind, ref_id) in self.objects
                if exists:
                    continue
                severity = "warning" if kind in SOFT_REFERENCES else "error"
                obj.problems.append(Problem(obj.rel, obj.locations.line(path), format_path(path),
                                            f"{kind} {ref_id!r} is not in the database", "reference", severity,
                                            f"{obj.kind}:{obj.id}"))

    # ------------------------------------------------------------------ access
    def object(self, kind: str, obj_id: str) -> Obj:
        kind = _kind_alias(kind)
        obj = self.objects.get((kind, obj_id))
        if obj is None:
            known = self.ids(kind)
            from difflib import get_close_matches
            close = get_close_matches(obj_id, known, n=3)
            hint = f" (did you mean {', '.join(close)}?)" if close else ""
            raise InputError(f"no {kind} {obj_id!r} in {self.root}{hint}")
        return obj

    def raw(self, kind: str, obj_id: str) -> dict:
        """The object as plain data (migrated, defaults and extends applied)."""
        obj = self.object(kind, obj_id)
        return obj.merged if obj.merged is not None else obj.raw

    def get(self, kind: str, obj_id: str) -> SIMap:
        """The SI view of an object; raises its errors if it has any."""
        obj = self.object(kind, obj_id)
        raise_for(obj.problems, f"{obj.kind} {obj.id!r}")
        if obj.simap is None:
            data = obj.merged if obj.merged is not None else obj.raw
            node = S.SCHEMAS[obj.kind]
            obj.simap = node.build(data, None, None, obj.id)
        return obj.simap

    def material(self, obj_id: str) -> SIMap:
        return self.get("material", obj_id)

    def tool(self, obj_id: str) -> SIMap:
        return self.get("tool", obj_id)

    def holder(self, obj_id: str) -> SIMap:
        return self.get("holder", obj_id)

    def tool_material(self, obj_id: str) -> SIMap:
        return self.get("tool_material", obj_id)

    def coating(self, obj_id: str) -> SIMap:
        return self.get("coating", obj_id)

    def machine(self, obj_id: str) -> SIMap:
        return self.get("machine", obj_id)

    def coolant(self, obj_id: str) -> SIMap:
        return self.get("coolant", obj_id)

    def workholding(self, obj_id: str) -> SIMap:
        return self.get("workholding", obj_id)

    def controller(self, obj_id: str) -> SIMap:
        return self.get("controller", obj_id)

    def validation_case(self, obj_id: str) -> SIMap:
        return self.get("validation_case", obj_id)

    def bibliography(self, key: str) -> dict:
        entry = self.bib.get(key)
        if entry is None:
            raise InputError(f"no bibliography entry {key!r}")
        return dict(entry)

    def ids(self, kind: str) -> list[str]:
        kind = _kind_alias(kind)
        return sorted(obj_id for (k, obj_id) in self.objects if k == kind)

    def files_of(self, kind: str, obj_id: str) -> list[Path]:
        """The files an object is read from (its own, then the files it extends)."""
        obj = self.object(kind, obj_id)
        return list(obj.chain) if obj.chain else [obj.file]

    def problems_of(self, kind: str, obj_id: str) -> list[Problem]:
        return sort_problems(self.object(kind, obj_id).problems)

    # ------------------------------------------------------------------ reports
    def estimates(self) -> list[Estimate]:
        out: list[Estimate] = []
        for record in self.files:
            out.extend(record.estimates)
        return sorted(out, key=lambda e: (e.file, e.line, e.path))

    def migrations(self) -> list[Migration]:
        out: list[Migration] = []
        for record in self.files:
            out.extend(record.migrations)
        return sorted(out, key=lambda m: (m.file, m.line, m.path_from))

    def validate_all(self) -> list[Problem]:
        """Every problem of every file and object (errors and warnings), sorted, each once."""
        problems: dict[tuple, Problem] = {}
        for problem in self.general:
            problems.setdefault(_problem_key(problem), problem)
        for record in self.files:
            for problem in record.problems:
                problems.setdefault(_problem_key(problem), problem)
        for obj in self.objects.values():
            for problem in obj.problems:
                problems.setdefault(_problem_key(problem), problem)
        return sort_problems(problems.values())

    def errors(self) -> list[Problem]:
        return errors_only(self.validate_all())

    @cached_property
    def summary(self) -> dict:
        problems = self.validate_all()
        by_kind: dict[str, int] = {}
        for kind, _ in self.objects:
            by_kind[kind] = by_kind.get(kind, 0) + 1
        return {"objects": dict(sorted(by_kind.items())), "files": len(self.files),
                "errors": sum(1 for p in problems if p.is_error),
                "warnings": sum(1 for p in problems if not p.is_error),
                "estimates": len(self.estimates()), "migrations": len(self.migrations()),
                "bibliography": len(self.bib)}

    # ------------------------------------------------------------------ inline objects (projects)
    def inline(self, kind: str, raw: Mapping, locations: Locations | None = None, *, path: tuple = (),
               file: str = "", label: str = "") -> tuple[SIMap | None, dict, list[Problem]]:
        """An object written inline in a project: ``(SIMap, merged raw, problems)``.

        Inline objects need no sources (they are the user's own data; their values report
        ``src`` "project"); ``extends: <id>`` starts from a database object.
        """
        kind = _kind_alias(kind)
        sub = locations.subtree(path) if locations is not None else Locations(file)
        data = {"src": "project", **dict(raw)} if "src" not in raw else dict(raw)
        if kind in S.ITEM_KINDS:
            # item rules are written for series files: migrate the item as the only item of a series
            wrapped = Locations(file, {("items", 0) + k: v for k, v in sub.entries.items()})
            result = migrate(S.ITEM_KINDS[kind], {"items": [data]}, wrapped, file=file, obj=label,
                             item_schema=S.inline_schema(kind))
            data = result.data["items"][0]
            problems = [Problem(p.file, p.line, p.path.replace("items[0]", "", 1).lstrip("."), p.message, p.kind,
                                p.severity, p.obj) for p in result.problems]
            problems = list(_reroot(problems, path))
        else:
            result = migrate(kind, data, sub, file=file, obj=label, schema=S.inline_schema(kind))
            data = result.data
            problems = list(_reroot(result.problems, path))
        base_id = data.get("extends") if isinstance(data.get("extends"), str) else None
        if base_id is not None:
            try:
                base = self.object(kind, base_id)
            except InputError as error:
                problems.append(Problem(file, sub.line(("extends",)), format_path(path + ("extends",)), str(error),
                                        "extends", obj=label))
                return None, data, problems
            if base.errors():
                problems.append(Problem(file, sub.line(("extends",)), format_path(path + ("extends",)),
                                        f"the base {kind} {base_id!r} has errors", "extends", obj=label))
                return None, data, problems
            base_raw = base.merged if base.merged is not None else base.raw
            data = merge_pinned(base_raw, data)
            data.pop("extends", None)
        ctx = Ctx(sub, obj=label, file=file)
        S.inline_schema(kind).check(data, (), None, ctx)
        problems.extend(_reroot(ctx.problems, path))
        for ref_kind, ref_id, ref_path in ctx.refs:
            exists = ref_id in self.bib if ref_kind == "bibliography" else (ref_kind, ref_id) in self.objects
            if not exists:
                problems.append(Problem(file, sub.line(ref_path), format_path(path + ref_path),
                                        f"{ref_kind} {ref_id!r} is not in the database", "reference",
                                        "warning" if ref_kind in SOFT_REFERENCES else "error", label))
        if errors_only(problems):
            return None, data, problems
        return S.SCHEMAS[kind].build(data, None, None, data.get("id") or label), data, problems


# Problems that an extending file can have on its own: wrong keys and wrong values. Missing keys,
# cross-field rules and references are checked on the merged object.
PARTIAL_KINDS = ("unknown", "type", "range", "format", "source", "migration", "yaml", "duplicate")


def partial_problems(problems):
    return [p for p in problems if p.kind in PARTIAL_KINDS]


def _reroot(problems, prefix: tuple):
    """Problems found on a subtree, with their key paths under ``prefix``."""
    if not prefix:
        return list(problems)
    head = format_path(prefix)
    out = []
    for p in problems:
        path = head if not p.path else (f"{head}{p.path}" if p.path.startswith("[") else f"{head}.{p.path}")
        out.append(Problem(p.file, p.line, path, p.message, p.kind, p.severity, p.obj))
    return out


def _problem_key(problem: Problem) -> tuple:
    return (problem.file, problem.line, problem.path, problem.message, problem.severity)


_ALIASES = {"tools": "tool", "holders": "holder", "materials": "material", "machines": "machine",
            "coolants": "coolant", "coatings": "coating", "tool_materials": "tool_material",
            "controllers": "controller", "models": "model", "validation_cases": "validation_case",
            "cases": "validation_case"}


def _kind_alias(kind: str) -> str:
    return _ALIASES.get(kind, kind)


__all__ = ["Database", "DEFAULT_ROOT", "KINDS", "ModelParams", "Obj"]
