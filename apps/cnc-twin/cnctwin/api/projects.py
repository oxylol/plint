"""Projects: the examples shipped in ``examples/`` (always listed, read-only) and the zips uploaded
into ``DATA_DIR/projects/<id>/``.

A project is a folder with ``project.yaml`` (format ``cnctwin-project``, db/_templates/project.yaml)
and the files it names (programs, CAD, stock meshes). Ids: an example folder ``examples/<name>``
is ``example-<name>``; an upload gets a slug of the name asked for (``?name=``), else of the
project's ``name:``, else of the zip's file name, made unique with ``-2``, ``-3`` ...; upload ids
never start with ``example-``, so the two kinds cannot collide.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import InputError
from ..db import yamlio
from .safety import ID_PATTERN, ApiError, check_id, list_files, resolve_inside, slugify
from .settings import Settings
from .upload import ExtractReport, extract_zip, flatten_single_folder, remove_tree

PROJECT_FILE = "project.yaml"
PROJECT_FORMAT = "cnctwin-project"
EXAMPLE_PREFIX = "example-"


@dataclass(frozen=True)
class Project:
    id: str
    source: str            # example | upload
    dir: Path

    @property
    def file(self) -> Path:
        return self.dir / PROJECT_FILE

    @property
    def read_only(self) -> bool:
        return self.source == "example"


class ProjectStore:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.uploads_dir = settings.projects_dir
        self.examples_dir = settings.examples_dir
        self._lock = threading.Lock()
        self._parsed: dict[str, tuple[tuple, dict]] = {}

    # ------------------------------------------------------------------ lookup
    def examples(self) -> list[Project]:
        if not self.examples_dir.is_dir():
            return []
        found = []
        for path in sorted(self.examples_dir.iterdir()):
            if path.is_dir() and (path / PROJECT_FILE).is_file():
                name = path.name if ID_PATTERN.match(path.name) else slugify(path.name)
                found.append(Project(f"{EXAMPLE_PREFIX}{name}"[:128], "example", path))
        return found

    def uploads(self) -> list[Project]:
        if not self.uploads_dir.is_dir():
            return []
        return [Project(p.name, "upload", p) for p in sorted(self.uploads_dir.iterdir())
                if p.is_dir() and not p.name.startswith(".") and (p / PROJECT_FILE).is_file()]

    def all(self) -> list[Project]:
        return self.examples() + self.uploads()

    def get(self, project_id: str) -> Project:
        check_id(project_id, "project id")
        if project_id.startswith(EXAMPLE_PREFIX):
            for project in self.examples():
                if project.id == project_id:
                    return project
        else:
            path = self.uploads_dir / project_id
            if (path / PROJECT_FILE).is_file():
                return Project(project_id, "upload", path)
        raise ApiError(404, f"no project {project_id!r}")

    # ------------------------------------------------------------------ content
    def document(self, project: Project) -> tuple[dict | None, str | None]:
        """(parsed project.yaml, error text)."""
        path = project.file
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = self._parsed.get(str(path))
        if cached is not None and cached[0] == signature:
            return cached[1]["data"], cached[1]["error"]
        data, error = None, None
        try:
            parsed = yamlio.load_file(path).data
            if not isinstance(parsed, dict):
                error = f"{PROJECT_FILE} is not a mapping"
            elif parsed.get("format") != PROJECT_FORMAT:
                error = f"{PROJECT_FILE}: format is {parsed.get('format')!r}, expected {PROJECT_FORMAT!r}"
                data = parsed
            else:
                data = parsed
        except InputError as problem:
            error = str(problem).replace(str(project.dir) + os.sep, "")
        self._parsed[str(path)] = (signature, {"data": data, "error": error})
        return data, error

    def summary(self, project: Project) -> dict:
        data, error = self.document(project)
        data = data or {}
        operations = []
        for item in data.get("operations") or []:
            if isinstance(item, dict):
                program = item.get("program")
                exists = isinstance(program, str) and _exists_inside(project.dir, program)
                operations.append({"name": item.get("name"), "program": program, "program_found": exists})
        tools = data.get("tools").get("list") if isinstance(data.get("tools"), dict) else None
        return {"id": project.id, "source": project.source, "read_only": project.read_only,
                "name": data.get("name") or project.dir.name, "description": data.get("description"),
                "machine": _ref_id(data.get("machine")), "material": _ref_id(data.get("material")),
                "operations": operations, "tools": len(tools) if isinstance(tools, list) else 0,
                "error": error}

    def detail(self, project: Project) -> dict:
        data, _ = self.document(project)
        out = self.summary(project)
        out["project"] = data
        out["files"] = list_files(project.dir)
        return out

    def file_path(self, project: Project, relative: str) -> Path:
        return resolve_inside(project.dir, relative)

    # ------------------------------------------------------------------ upload
    def install_upload(self, archive: Path, *, desired: str | None, filename: str | None) -> tuple[Project, ExtractReport]:
        """Extract ``archive`` (already on disk) into a new project folder."""
        self.s.tmp_dir.mkdir(parents=True, exist_ok=True)
        staging = self.s.tmp_dir / f"extract-{uuid.uuid4().hex}"
        try:
            report = extract_zip(archive, staging, max_bytes=self.s.max_extract_bytes,
                                 max_files=self.s.max_extract_files, max_ratio=self.s.max_compression_ratio)
            flatten_single_folder(staging)
            if not (staging / PROJECT_FILE).is_file():
                yaml_files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*.yaml"))
                hint = f" (found {', '.join(yaml_files[:5])})" if yaml_files else ""
                raise ApiError(422, f"the archive has no {PROJECT_FILE} at its top level{hint}")
            try:
                parsed = yamlio.load_file(staging / PROJECT_FILE).data
            except InputError as problem:
                raise ApiError(422, str(problem).replace(str(staging) + os.sep, "")) from None
            if not isinstance(parsed, dict) or parsed.get("format") != PROJECT_FORMAT:
                raise ApiError(422, f"{PROJECT_FILE} is not a CNC Twin project (format: {PROJECT_FORMAT} missing)")
            base = desired or parsed.get("name") or (Path(filename).stem if filename else "") or "project"
            slug = slugify(str(base))
            if slug == EXAMPLE_PREFIX.rstrip("-") or slug.startswith(EXAMPLE_PREFIX):
                slug = f"p-{slug}"
            with self._lock:
                self.uploads_dir.mkdir(parents=True, exist_ok=True)
                project_id = slug
                number = 1
                while (self.uploads_dir / project_id).exists():
                    number += 1
                    project_id = f"{slug}-{number}"
                staging.rename(self.uploads_dir / project_id)
            return Project(project_id, "upload", self.uploads_dir / project_id), report
        finally:
            remove_tree(staging)

    def problems(self, project: Project) -> list[str]:
        problems = []
        summary = self.summary(project)
        if summary["error"]:
            problems.append(summary["error"])
        for item in summary["operations"]:
            if item["program"] and not item["program_found"]:
                problems.append(f"operation {item['name']}: program {item['program']!r} is not in the project")
        return problems


def _ref_id(value) -> str | None:
    """A database reference: an id, or an inline item with an id."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        return value["id"]
    return None


def _exists_inside(root: Path, relative: str) -> bool:
    try:
        resolve_inside(root, relative)
        return True
    except ApiError:
        return False
