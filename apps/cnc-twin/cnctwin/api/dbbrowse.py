"""Browsing the databases (db/): kinds, items and one item with its sources.

Through :class:`cnctwin.db.database.Database` when the build has it (validated objects, SI views,
problems per object, the bibliography), loaded once and again whenever a file under db/ changes.
Without it (an older build), the YAML files are read as they are (cnctwin.db.yamlio), with no
validation and no SI view; ``backend`` in every response says which.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from ..core.errors import InputError
from ..db import yamlio
from .safety import ApiError, check_id

ALIASES = {"materials": "material", "tools": "tool", "holders": "holder", "machines": "machine",
           "coolants": "coolant", "coatings": "coating", "tool_materials": "tool_material",
           "controllers": "controller", "models": "model", "validation_cases": "validation_case",
           "cases": "validation_case", "workholdings": "workholding", "bibliographies": "bibliography"}
FOLDER_OF = {"material": "materials", "tool": "tools", "holder": "holders", "machine": "machines",
             "coolant": "coolants", "coating": "coatings", "tool_material": "tool_materials",
             "controller": "controllers", "model": "models", "workholding": "workholding",
             "bibliography": "bibliography"}
SUMMARY_KEYS = ("name", "maker", "model", "series", "type", "family", "iso_group", "kind", "dialect", "order_code",
                "series_id", "title")


def canonical_kind(kind: str) -> str:
    kind = check_id(kind, "database kind")
    return ALIASES.get(kind, kind)


def _signature(root: Path) -> tuple:
    items = []
    if root.is_dir():
        for directory, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                if name.endswith((".yaml", ".yml")):
                    stat = os.stat(os.path.join(directory, name))
                    items.append((os.path.join(directory, name), stat.st_size, stat.st_mtime_ns))
    return tuple(items)


def _sources(value, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "src" and isinstance(item, str):
                found.add(item)
            else:
                _sources(item, found)
    elif isinstance(value, list):
        for item in value:
            _sources(item, found)


def _summary(kind: str, obj_id: str, raw: dict, file: str) -> dict:
    out = {"id": obj_id, "kind": kind, "file": file}
    for key in SUMMARY_KEYS:
        if isinstance(raw.get(key), (str, int, float)):
            out[key] = raw[key]
    return out


class DatabaseBrowser:
    def __init__(self, db_dir: Path) -> None:
        self.root = Path(db_dir)
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._db = None
        self._files: dict | None = None
        self.backend = self._detect()

    @staticmethod
    def _detect() -> str:
        try:
            from ..db.database import Database  # noqa: F401
            return "cnctwin.db.database"
        except ImportError:
            return "files"

    # ------------------------------------------------------------------ loading
    def _current(self):
        signature = _signature(self.root)
        with self._lock:
            if signature != self._signature:
                if self.backend == "cnctwin.db.database":
                    from ..db.database import Database
                    self._db = Database.load(self.root)
                else:
                    self._files = self._load_files()
                self._signature = signature
            return self._db if self.backend == "cnctwin.db.database" else self._files

    def _load_files(self) -> dict:
        objects: dict[tuple[str, str], dict] = {}
        bibliography: dict[str, dict] = {}
        if not self.root.is_dir():
            return {"objects": objects, "bibliography": bibliography}
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
            kind = ALIASES.get(folder.name, folder.name)
            for path in sorted(folder.rglob("*.yaml")):
                rel = path.relative_to(self.root.parent).as_posix()
                try:
                    data = yamlio.load_file(path).data
                except InputError as error:
                    objects[(kind, path.stem)] = {"raw": None, "file": rel, "error": str(error), "series": None}
                    continue
                if not isinstance(data, dict):
                    continue
                if kind == "bibliography":
                    for key, entry in (data.get("entries") or {}).items():
                        bibliography[str(key)] = dict(entry) if isinstance(entry, dict) else {"value": entry}
                    continue
                items = data.get("items")
                if isinstance(items, list):
                    series = {k: v for k, v in data.items() if k != "items"}
                    for item in items:
                        if isinstance(item, dict) and isinstance(item.get("id"), str):
                            merged = {**{k: v for k, v in series.items() if k in ("maker", "series", "series_id")},
                                      **item}
                            objects[(kind, item["id"])] = {"raw": merged, "file": rel, "error": None,
                                                           "series": series}
                else:
                    obj_id = data.get("id") if isinstance(data.get("id"), str) else path.stem
                    objects[(kind, obj_id)] = {"raw": data, "file": rel, "error": None, "series": None}
        return {"objects": objects, "bibliography": bibliography}

    # ------------------------------------------------------------------ queries
    def kinds(self) -> dict:
        current = self._current()
        counts: dict[str, int] = {}
        if self.backend == "cnctwin.db.database":
            for kind, _ in current.objects:
                counts[kind] = counts.get(kind, 0) + 1
            counts["bibliography"] = len(current.bib.keys())
        else:
            for kind, _ in current["objects"]:
                counts[kind] = counts.get(kind, 0) + 1
            counts["bibliography"] = len(current["bibliography"])
        return {"backend": self.backend,
                "kinds": [{"kind": k, "count": counts[k], "url": f"/api/db/{k}"} for k in sorted(counts)]}

    def list(self, kind: str) -> dict:
        kind = canonical_kind(kind)
        current = self._current()
        items = []
        if kind == "bibliography":
            entries = self._bibliography(current)
            for key in sorted(entries):
                entry = entries[key]
                items.append({"id": key, "kind": "bibliography",
                              **{k: entry[k] for k in ("type", "authors", "title", "year", "venue") if k in entry}})
        elif self.backend == "cnctwin.db.database":
            for obj_id in current.ids(kind):
                obj = current.object(kind, obj_id)
                raw = obj.merged if obj.merged is not None else obj.raw
                summary = _summary(kind, obj_id, raw if isinstance(raw, dict) else {}, obj.rel)
                errors = sum(1 for p in obj.problems if p.is_error)
                summary["errors"] = errors
                summary["warnings"] = len(obj.problems) - errors
                items.append(summary)
        else:
            for (k, obj_id), record in sorted(current["objects"].items()):
                if k == kind:
                    summary = _summary(kind, obj_id, record["raw"] or {}, record["file"])
                    if record["error"]:
                        summary["error"] = record["error"]
                    items.append(summary)
        if not items and kind not in self._known_kinds(current):
            raise ApiError(404, f"no database kind {kind!r} (GET /api/db lists them)")
        return {"backend": self.backend, "kind": kind, "count": len(items), "items": items}

    def get(self, kind: str, obj_id: str) -> dict:
        kind = canonical_kind(kind)
        check_id(obj_id, "database id")
        current = self._current()
        bibliography = self._bibliography(current)
        if kind == "bibliography":
            if obj_id not in bibliography:
                raise ApiError(404, f"no bibliography entry {obj_id!r}")
            return {"backend": self.backend, "kind": kind, "id": obj_id, "entry": bibliography[obj_id]}
        if self.backend == "cnctwin.db.database":
            from ..db.simap import plain
            try:
                obj = current.object(kind, obj_id)
            except InputError as error:
                raise ApiError(404, str(error)) from None
            raw = obj.merged if obj.merged is not None else obj.raw
            si = None
            si_error = None
            if not obj.errors():
                try:
                    si = plain(current.get(kind, obj_id))
                except (InputError, TypeError, ValueError, KeyError) as error:
                    si_error = str(error)
            out = {"backend": self.backend, "kind": kind, "id": obj_id, "file": obj.rel,
                   "files": [str(Path(p).relative_to(self.root.parent)) if Path(p).is_absolute() else str(p)
                             for p in current.files_of(kind, obj_id)],
                   "data": raw, "si": si, "si_error": si_error,
                   "problems": [p.to_dict() for p in current.problems_of(kind, obj_id)]}
        else:
            record = current["objects"].get((kind, obj_id))
            if record is None:
                raise ApiError(404, f"no {kind} {obj_id!r} in the database")
            raw = record["raw"]
            out = {"backend": self.backend, "kind": kind, "id": obj_id, "file": record["file"], "data": raw,
                   "series": record["series"], "si": None, "problems": [record["error"]] if record["error"] else []}
        found: set[str] = set()
        _sources(raw, found)
        out["sources"] = {key: bibliography[key] for key in sorted(found) if key in bibliography}
        out["unresolved_sources"] = sorted(k for k in found if k not in bibliography and k != "est")
        return out

    def _bibliography(self, current) -> dict:
        if self.backend == "cnctwin.db.database":
            return {key: dict(current.bib.get(key)) for key in current.bib.keys()}
        return current["bibliography"]

    def _known_kinds(self, current) -> set[str]:
        if self.backend == "cnctwin.db.database":
            return {k for k, _ in current.objects} | set(FOLDER_OF)
        return {k for k, _ in current["objects"]} | {"bibliography"}
