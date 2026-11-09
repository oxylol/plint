"""The run cache (docs/ml.md "Run cache"): a finished run is reused when nothing that decides its results changed.

A run does not have to be computed in real time: it is computed once, kept, and played back. The cache finds the
run of a request again. Its **key** is the sha256 of the canonical JSON of:

* ``project_hash``: :func:`cnctwin.project.project_hash` (the migrated project data, every file it names, the
  database files of the objects it uses with their ``extends`` chains, every ``db/models`` file);
* ``databases``: the sha256 of every file under ``db/`` (a run reads database files beyond the objects its
  project names: the class representative tool materials of db/models/tooling.yaml, the template defaults of
  db/_templates, so any database edit invalidates every cached run: conservative by design);
* ``physics``: the project's resolved physics settings (SI);
* ``options``: every run option except ``resume_from`` and ``no_cache`` (``max_sim_s``, ``parts``, ``outputs``,
  ``sanity``, ``checkpoint_interval_s``, ``operator``, ``uncertainty`` ...);
* ``runner``: the simulation entry (:func:`canonical_runner`: the loop, and cnctwin.sweep.runner without
  operator overrides, are one; any other runner is its spec plus the sha256 of its source file);
* ``build``: :func:`build_id`, the same value as :func:`cnctwin.core.rundir.build_id` (every ``cnctwin/**/*.py``
  and the numeric libraries), recomputed for every key (file hashes are memoised by size and mtime);
* ``libraries``: the versions of the other libraries the run directory depends on (STEP import, pictures,
  report, YAML);
* ``platform``: machine, system and CPU model (Numba compiles for the CPU; architecture 9);
* ``environment``: the deterministic-environment contract (threads 1, PYTHONHASHSEED 0) and SOURCE_DATE_EPOCH.

A run is **registered** only when the key computed after the run equals the key computed before it (the source
tree, the databases and the inputs did not change while it ran) and it wrote run.json; a run of another build
or other inputs is never reused. A lookup returns the newest entry of a key that still **verifies**: its run.json
is there (with the entry's build, when it records one) and the digest of its result set (architecture 9; the
fast-tier replay parts ``twin replay`` adds are left out) equals the registered one. An entry that does not
verify is dropped.

**Index** (``<root>/index.jsonl``, root ``CNCTWIN_CACHE_DIR`` or ``DATA_DIR/cache``): append-only JSON lines
(``put``, ``hit``, ``drop``), each appended with one ``write`` under an exclusive ``flock`` of ``<root>/.lock``
and fsynced; a torn last line (a crash while appending) is skipped when read and terminated before the next
append. ``compact`` rewrites the file (temporary file, fsync, rename) with one ``put`` per live entry. A
``flock`` on ``<root>/locks/<key>.lock`` is held while a key is being computed, so a second request for the same
key waits and then finds the run.

**Eviction** (after every registration, and ``twin cache evict``): only runs the cache owns (made under
``<root>/runs/``) are deleted, runs elsewhere (``twin run`` into DATA_DIR/runs, the API's runs) are references
that are dropped only when they no longer verify. An owned entry not used (registered or hit) for
``max_age_days`` (``CNCTWIN_CACHE_MAX_AGE_DAYS``, default 90) is evicted; then, while the owned runs take more
than ``max_bytes`` (``CNCTWIN_CACHE_MAX_GB``, default 10 GiB), the least recently used one is (ties by key). An
entry whose key lock is held is never evicted. ``CNCTWIN_CACHE=off`` switches the cache off.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths
from .core.errors import InputError

INDEX_FORMAT = "cnctwin-cache-index"
KEY_FORMAT = "cnctwin-cache-key"
VERSION = 1
DEFAULT_RUNNER = "cnctwin.core.loop:run_project"
SWEEP_RUNNER = "cnctwin.sweep.runner:run_project"
NON_RESULT_OPTIONS = ("resume_from", "no_cache")
UNCACHED_OPTIONS = ("replay", "resume_from")
# The environment every deterministic run has (bin/twin, bin/tsafe, the API's jobs; architecture 9)
DETERMINISTIC_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "NUMBA_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
# Libraries beyond rundir.LIBRARIES that decide files of a run directory (STEP import, pictures, the report, YAML)
OUTPUT_LIBRARIES = ("cadquery-ocp", "matplotlib", "pillow", "weasyprint", "imageio-ffmpeg", "pyyaml", "pydantic")
DEFAULT_MAX_BYTES = 10 * 1024 ** 3
DEFAULT_MAX_AGE_S = 90 * 86400.0
_REPLAY_PART = re.compile(r"^series/fast/replay-\d+-\d+\.parquet$")
_KEY = re.compile(r"^[0-9a-f]{64}$")


# =============================================================================== fingerprints

_FILE_MEMO: dict[str, tuple[int, int, bytes]] = {}


def _file_sha256(path: Path) -> bytes:
    """sha256 digest of a file, memoised by (size, mtime_ns): a file is read again only when it changed."""
    st = os.stat(path)
    key = str(path)
    memo = _FILE_MEMO.get(key)
    if memo is not None and memo[0] == st.st_size and memo[1] == st.st_mtime_ns:
        return memo[2]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    value = digest.digest()
    _FILE_MEMO[key] = (st.st_size, st.st_mtime_ns, value)
    return value


def build_id(package_dir: Path | None = None) -> str:
    """:func:`cnctwin.core.rundir.build_id` computed afresh (rundir memoises it per process, a long-lived server
    or sweep driver must see a changed tree): sha256 of every ``cnctwin/**/*.py`` (path and content, C-sorted) and
    of the library versions."""
    from .core import rundir
    package_dir = Path(package_dir) if package_dir is not None else rundir._PACKAGE_DIR
    base = package_dir.parent
    files = []
    for directory, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        files.extend(Path(directory) / n for n in filenames if n.endswith(".py"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(base).as_posix().encode("utf-8")):
        digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(_file_sha256(path))
    digest.update(json.dumps(rundir.library_versions(), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _tree_files(root: Path) -> list[Path]:
    out = []
    if root.is_dir():
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
            out.extend(Path(directory) / n for n in filenames if not n.startswith("."))
    return sorted(out, key=lambda p: p.relative_to(root).as_posix().encode("utf-8"))


def databases_digest(root: Path | None = None) -> str:
    """sha256 over (relative path, sha256) of every file under ``db/``."""
    root = Path(root) if root is not None else paths.DB
    digest = hashlib.sha256()
    for path in _tree_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + _file_sha256(path))
    return digest.hexdigest()


def _tree_signature(root: Path) -> tuple:
    items = []
    for path in _tree_files(root):
        st = os.stat(path)
        items.append((path.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns))
    return tuple(items)


_DB_MEMO: dict = {"signature": None, "db": None, "root": None}
_DB_LOCK = threading.Lock()


def database(root: Path | None = None):
    """The loaded :class:`cnctwin.db.Database`, loaded again whenever a file under db/ changed (Database.load
    takes about 2 s; a server or sweep driver resolves many projects). Thread-safe: the API's request threads
    share it."""
    from .db import Database
    root = Path(root) if root is not None else paths.DB
    with _DB_LOCK:
        signature = _tree_signature(root)
        if _DB_MEMO["db"] is None or _DB_MEMO["signature"] != signature or _DB_MEMO["root"] != str(root):
            _DB_MEMO.update(db=Database.load(root), signature=signature, root=str(root))
        return _DB_MEMO["db"]


def output_library_versions() -> dict:
    import importlib.metadata
    out = {}
    for name in OUTPUT_LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def platform_info() -> dict:
    from .core import rundir
    return rundir.platform_info()


def environment_contract(source_date_epoch: str | None = None) -> dict:
    """The environment a deterministic run executes in: the thread settings of architecture 9 and the
    SOURCE_DATE_EPOCH of its derived outputs (the API passes its jobs' value)."""
    env = dict(DETERMINISTIC_ENV)
    env["SOURCE_DATE_EPOCH"] = source_date_epoch
    return env


def current_environment(environ: Mapping | None = None) -> dict:
    environ = os.environ if environ is None else environ
    env = {name: environ.get(name) for name in DETERMINISTIC_ENV}
    env["SOURCE_DATE_EPOCH"] = environ.get("SOURCE_DATE_EPOCH")
    return env


def deterministic_environment(environ: Mapping | None = None) -> bool:
    """True when this process runs under the contract (bin/twin, bin/tsafe and the API's jobs set it)."""
    environ = os.environ if environ is None else environ
    return all(environ.get(name) == value for name, value in DETERMINISTIC_ENV.items())


def canonical_options(options: Mapping | None) -> dict:
    """The options that decide a run's results, as plain JSON (keys sorted when dumped)."""
    from .core.rundir import jsonable
    plain = jsonable({k: v for k, v in (options or {}).items() if k not in NON_RESULT_OPTIONS})
    return json.loads(json.dumps(plain, sort_keys=True))


def canonical_runner(spec: str | None, options: Mapping | None = None) -> str:
    """The runner as it enters the key: the loop (``cnctwin.core.loop:run_project``, also when no runner is named),
    cnctwin.sweep.runner without ``operator`` overrides (it calls the loop unchanged), else the spec itself with
    the sha256 of its source file, so a changed test runner never matches an old run."""
    spec = (spec or DEFAULT_RUNNER).strip()
    if spec == DEFAULT_RUNNER:
        return DEFAULT_RUNNER
    if spec == SWEEP_RUNNER:
        return SWEEP_RUNNER if (options or {}).get("operator") else DEFAULT_RUNNER
    module_name = spec.rsplit(":", 1)[0]
    source = None
    if module_name.endswith(".py"):
        source = Path(module_name)
    elif module_name.split(".")[0] == "cnctwin":
        base = Path(__file__).resolve().parent.joinpath(*module_name.split(".")[1:])
        source = base.with_suffix(".py") if base.with_suffix(".py").is_file() else base / "__init__.py"
    if source is not None and source.is_file():
        return f"{spec}@{_file_sha256(source.resolve()).hex()}"
    return spec


# =============================================================================== the key

@dataclass(frozen=True)
class CacheKey:
    key: str
    components: dict
    project: object = None         # the ResolvedProject it was computed from

    def summary(self) -> dict:
        """What ``twin cache`` and the index show of the components (the options and environment in full)."""
        c = self.components
        return {"project_hash": c["project_hash"], "databases": c["databases"], "build": c["build"],
                "runner": c["runner"], "options": c["options"], "platform": c["platform"],
                "environment": c["environment"]}


def key_for(project, options: Mapping | None = None, *, runner: str | None = None, environment: Mapping | None = None,
            db=None) -> CacheKey:
    """The cache key of running ``project`` (a project.yaml, its folder or a ResolvedProject) with ``options``
    through ``runner`` in ``environment`` (default: the contract with this process's SOURCE_DATE_EPOCH).
    Raises InputError when the project does not resolve."""
    from .core.rundir import jsonable
    from .project import ResolvedProject, load_project, resolve
    if not isinstance(project, ResolvedProject):
        path = Path(project)
        path = path / "project.yaml" if path.is_dir() else path
        if not path.is_file():
            raise InputError(f"no project file {path}")
        written = load_project(path)              # a project file that does not validate fails before the database loads
        project = resolve(written, db if db is not None else database())
    physics = getattr(project, "physics", None)
    components = {
        "format": KEY_FORMAT, "version": VERSION,
        "project_hash": project.hash,
        "databases": databases_digest(),
        "physics": jsonable(physics.to_dict()) if hasattr(physics, "to_dict") else None,
        "options": canonical_options(options),
        "runner": canonical_runner(runner, options),
        "build": build_id(),
        "libraries": output_library_versions(),
        "platform": platform_info(),
        "environment": dict(environment) if environment is not None
        else environment_contract(os.environ.get("SOURCE_DATE_EPOCH")),
    }
    text = json.dumps(components, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return CacheKey(hashlib.sha256(text.encode("utf-8")).hexdigest(), components, project)


# =============================================================================== digests of run directories

def _result_files(run_dir: Path) -> list[tuple[str, Path]]:
    from .core.rundir import result_set_files
    out = []
    for path in result_set_files(run_dir):
        relative = path.relative_to(run_dir).as_posix()
        if not _REPLAY_PART.match(relative):
            out.append((relative, path))
    return out


def _index_without_replays(path: Path) -> bytes:
    return b"".join(line for line in path.read_bytes().splitlines(keepends=True)
                    if line.strip() and not _replay_line(line))


def run_digest(run_dir) -> tuple[str, int, int]:
    """(digest, files, bytes) of a run's result set (cnctwin.core.rundir.result_set_files) without the replay parts
    of ``twin replay`` (and their lines of series/index.jsonl): those add fast rows of a window that the run
    already determines, so a replay does not invalidate the run."""
    run_dir = Path(run_dir)
    digest = hashlib.sha256()
    count = size = 0
    for relative, path in _result_files(run_dir):
        if relative == "series/index.jsonl":
            data = _index_without_replays(path)
            file_hash = hashlib.sha256(data).digest()
            size += len(data)
        else:
            length = path.stat().st_size
            file_hash = hashlib.sha256(path.read_bytes()).digest() if length < (1 << 20) else _stream_sha256(path)
            size += length
        digest.update(relative.encode("utf-8") + b"\0" + file_hash + b"\n")
        count += 1
    return digest.hexdigest(), count, size


def result_size(run_dir) -> tuple[int, int]:
    """(files, bytes) of the result set as :func:`run_digest` counts them, without hashing."""
    run_dir = Path(run_dir)
    count = size = 0
    for relative, path in _result_files(run_dir):
        size += len(_index_without_replays(path)) if relative == "series/index.jsonl" else path.stat().st_size
        count += 1
    return count, size


def _replay_line(line: bytes) -> bool:
    try:
        return bool(json.loads(line).get("replay"))
    except (ValueError, AttributeError):
        return False


def _stream_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.digest()


def directory_bytes(path) -> int:
    total = 0
    for directory, _, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                continue
    return total


def _run_json(run_dir: Path) -> dict | None:
    try:
        document = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _recorded_build(document: Mapping) -> str | None:
    versions = document.get("versions") if isinstance(document.get("versions"), Mapping) else {}
    build = versions.get("build")
    return build if isinstance(build, str) else None


# =============================================================================== the index

@dataclass
class Entry:
    """A registered run of a key (the state the index lines replay to)."""

    key: str
    run_dir: str
    run_id: str
    owned: bool
    digest: str
    files: int
    result_bytes: int
    bytes: int
    build: str | None
    project_hash: str | None
    created_s: float
    used_s: float
    hits: int = 0
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RunCache:
    """The cache under ``root`` (module docstring)."""

    def __init__(self, root=None, *, max_bytes: int | None = None, max_age_s: float | None = None) -> None:
        self.root = (Path(root) if root is not None else default_root()).absolute()
        self.max_bytes = int(max_bytes) if max_bytes is not None else _env_max_bytes()
        self.max_age_s = float(max_age_s) if max_age_s is not None else _env_max_age_s()

    # ------------------------------------------------------------------ layout
    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def owns(self, run_dir) -> bool:
        try:
            Path(run_dir).resolve().relative_to(self.runs_dir.resolve())
            return True
        except ValueError:
            return False

    @contextlib.contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _lock_path(self, key: str) -> Path:
        if not _KEY.match(key):
            raise ValueError(f"not a cache key: {key!r}")
        return self.root / "locks" / f"{key}.lock"

    @contextlib.contextmanager
    def lock(self, key: str):
        """Exclusive lock of one key while it is computed (released by the kernel if the process dies)."""
        path = self._lock_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def try_lock(self, key: str):
        """The key's lock without waiting: yields True when held, False when another process computes the key."""
        path = self._lock_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def in_use(self, key: str) -> bool:
        path = self._lock_path(key)
        if not path.is_file():
            return False
        with open(path, "a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False

    # ------------------------------------------------------------------ lines
    def _append(self, records: list[dict]) -> None:
        """Append index lines: one write under the lock, a torn last line terminated first, fsynced."""
        data = b"".join((json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                        for r in records)
        with self._locked():
            fd = os.open(self.index_path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                size = os.fstat(fd).st_size
                if size == 0:
                    header = {"op": "format", "format": INDEX_FORMAT, "version": VERSION}
                    data = (json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") + data
                elif os.pread(fd, 1, size - 1) != b"\n":
                    data = b"\n" + data
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)

    def lines(self) -> list[dict]:
        """Every complete, readable line (a torn or damaged line is skipped)."""
        try:
            raw = self.index_path.read_bytes()
        except FileNotFoundError:
            return []
        out = []
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n") or not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("op"), str):
                out.append(record)
        return out

    def state(self) -> dict[tuple[str, str], Entry]:
        """(key, run_dir) -> Entry, in registration order (the index replayed)."""
        entries: dict[tuple[str, str], Entry] = {}
        for record in self.lines():
            op = record["op"]
            ident = (str(record.get("key", "")), str(record.get("run_dir", "")))
            if op == "put":
                entries.pop(ident, None)
                entries[ident] = Entry(key=ident[0], run_dir=ident[1], run_id=str(record.get("run_id", "")),
                                       owned=bool(record.get("owned")), digest=str(record.get("digest", "")),
                                       files=int(record.get("files") or 0),
                                       result_bytes=int(record.get("result_bytes") or 0),
                                       bytes=int(record.get("bytes") or 0), build=record.get("build"),
                                       project_hash=record.get("project_hash"),
                                       created_s=float(record.get("at_s") or 0.0),
                                       used_s=float(record.get("used_s") or record.get("at_s") or 0.0),
                                       hits=int(record.get("hits") or 0), components=dict(record.get("components") or {}))
            elif op == "hit" and ident in entries:
                entries[ident].used_s = max(entries[ident].used_s, float(record.get("at_s") or 0.0))
                entries[ident].hits += 1
            elif op == "drop":
                entries.pop(ident, None)
        return entries

    def entries(self, key: str | None = None) -> list[Entry]:
        return [e for (k, _), e in self.state().items() if key is None or k == key]

    # ------------------------------------------------------------------ operations
    def verify(self, entry: Entry, mode: str = "full") -> tuple[bool, str]:
        """(ok, reason) of one entry: run.json there (with the entry's build), the result set unchanged
        (``full``: its digest; ``quick``: its file count and bytes, no hashing)."""
        run_dir = Path(entry.run_dir)
        document = _run_json(run_dir)
        if document is None:
            return False, "run.json is missing or unreadable"
        recorded = _recorded_build(document)
        if recorded is not None and entry.build is not None and recorded != entry.build:
            return False, "run.json names another build"
        if mode == "quick":
            files, size = result_size(run_dir)
            if (files, size) != (entry.files, entry.result_bytes):
                return False, (f"the result set has {files} files of {size} bytes, {entry.files} files of "
                               f"{entry.result_bytes} bytes were registered")
            return True, ""
        digest, count, _ = run_digest(run_dir)
        if digest != entry.digest:
            return False, f"the result set changed ({count} files, digest {digest[:12]} != {entry.digest[:12]})"
        return True, ""

    def lookup(self, key: str, *, verify: str = "full", record_hit: bool = True, inside=None) -> Entry | None:
        """The newest entry of ``key`` that verifies (entries that do not are dropped); ``inside``: only runs in
        that folder. A hit is recorded (it keeps the entry from eviction by age and size)."""
        candidates = [e for e in self.entries(key) if inside is None or _inside(e.run_dir, inside)]
        for entry in reversed(candidates):
            ok, reason = self.verify(entry, verify)
            if not ok:
                self.drop(entry, f"does not verify: {reason}")
                continue
            if record_hit:
                self._append([{"op": "hit", "key": entry.key, "run_dir": entry.run_dir, "at_s": time.time()}])
                entry.used_s = time.time()
                entry.hits += 1
            return entry
        return None

    def register(self, key: CacheKey, run_dir, *, owned: bool | None = None) -> Entry:
        """Add a finished run (it must have run.json; a run.json of another build is refused)."""
        run_dir = Path(run_dir).resolve()
        document = _run_json(run_dir)
        if document is None:
            raise InputError(f"{run_dir} has no run.json: only finished runs are cached")
        recorded = _recorded_build(document)
        if recorded is not None and recorded != key.components["build"]:
            raise InputError(f"{run_dir}: run.json names build {recorded[:12]}, the key build "
                             f"{key.components['build'][:12]} (the source tree changed while it ran)")
        digest, files, result_bytes = run_digest(run_dir)
        now = time.time()
        owned = self.owns(run_dir) if owned is None else bool(owned)
        record = {"op": "put", "key": key.key, "run_dir": str(run_dir), "run_id": str(document.get("run_id") or run_dir.name),
                  "owned": owned, "digest": digest, "files": files, "result_bytes": result_bytes,
                  "bytes": directory_bytes(run_dir), "build": key.components["build"],
                  "project_hash": key.components["project_hash"], "at_s": now, "components": _jsonable(key.summary())}
        self._append([record])
        return Entry(key=key.key, run_dir=str(run_dir), run_id=record["run_id"], owned=owned, digest=digest, files=files,
                     result_bytes=result_bytes, bytes=record["bytes"], build=record["build"],
                     project_hash=record["project_hash"], created_s=now, used_s=now, components=record["components"])

    def drop(self, entry: Entry, reason: str) -> None:
        self._append([{"op": "drop", "key": entry.key, "run_dir": entry.run_dir, "reason": reason, "at_s": time.time()}])

    def evict(self, *, now: float | None = None, max_bytes: int | None = None, max_age_s: float | None = None,
              dry_run: bool = False, protect=()) -> list[Entry]:
        """Apply the eviction policy (module docstring); returns the evicted entries (their runs deleted unless
        ``dry_run``)."""
        now = time.time() if now is None else float(now)
        max_bytes = self.max_bytes if max_bytes is None else int(max_bytes)
        max_age_s = self.max_age_s if max_age_s is None else float(max_age_s)
        protected = set(protect)
        owned = [e for e in self.entries() if e.owned and e.key not in protected]
        victims = [e for e in owned if now - e.used_s > max_age_s]
        rest = sorted((e for e in owned if e not in victims), key=lambda e: (e.used_s, e.key, e.run_dir))
        total = sum(e.bytes for e in rest)
        for entry in list(rest):
            if total <= max_bytes:
                break
            victims.append(entry)
            total -= entry.bytes
        evicted = []
        for entry in victims:
            if self.in_use(entry.key):
                continue
            evicted.append(entry)
            if dry_run:
                continue
            reason = "evicted: not used for more than the maximum age" if now - entry.used_s > max_age_s else \
                "evicted: the cache is above its size limit (least recently used)"
            self.drop(entry, reason)
            if self.owns(entry.run_dir):
                shutil.rmtree(entry.run_dir, ignore_errors=True)
        if not dry_run:
            self._maybe_compact()
        return evicted

    def _maybe_compact(self) -> None:
        lines = self.lines()
        live = len(self.state())
        if len(lines) > 4 * live + 1000:
            self.compact()

    def compact(self) -> int:
        """Rewrite the index with one put (with its hit count and last use) per live entry; returns the line count."""
        with self._locked():
            entries = self.state()
            records = [{"op": "format", "format": INDEX_FORMAT, "version": VERSION}]
            for entry in entries.values():
                records.append({"op": "put", "key": entry.key, "run_dir": entry.run_dir, "run_id": entry.run_id,
                                "owned": entry.owned, "digest": entry.digest, "files": entry.files,
                                "result_bytes": entry.result_bytes, "bytes": entry.bytes, "build": entry.build,
                                "project_hash": entry.project_hash, "at_s": entry.created_s, "used_s": entry.used_s,
                                "hits": entry.hits, "components": entry.components})
            data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records).encode("utf-8")
            tmp = self.index_path.with_name(f".index.jsonl.{os.getpid()}.tmp")
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.index_path)
            from .core.rundir import fsync_dir
            fsync_dir(self.root)
        locks = self.root / "locks"
        if locks.is_dir():
            live = {e.key for e in entries.values()}
            for path in sorted(locks.glob("*.lock")):
                if path.stem not in live and not self.in_use(path.stem):
                    with contextlib.suppress(OSError):
                        path.unlink()
        return len(records)


def _inside(run_dir, folder) -> bool:
    try:
        Path(run_dir).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def _jsonable(value):
    from .core.rundir import jsonable
    return jsonable(value)


def default_root() -> Path:
    override = os.environ.get("CNCTWIN_CACHE_DIR")
    return Path(override) if override else paths.data_dir() / "cache"


def enabled(environ: Mapping | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return str(environ.get("CNCTWIN_CACHE", "on")).strip().lower() not in ("off", "0", "false", "no")


def _env_max_bytes() -> int:
    text = os.environ.get("CNCTWIN_CACHE_MAX_GB")
    try:
        return int(float(text) * 1024 ** 3) if text else DEFAULT_MAX_BYTES
    except ValueError:
        raise InputError(f"CNCTWIN_CACHE_MAX_GB: not a number: {text!r}") from None


def _env_max_age_s() -> float:
    text = os.environ.get("CNCTWIN_CACHE_MAX_AGE_DAYS")
    try:
        return float(text) * 86400.0 if text else DEFAULT_MAX_AGE_S
    except ValueError:
        raise InputError(f"CNCTWIN_CACHE_MAX_AGE_DAYS: not a number: {text!r}") from None


# =============================================================================== running through the cache

def resolve_runner(spec: str | None):
    """The callable of a runner spec (``module:function`` or ``/path/file.py:function``; default the loop)."""
    from .uncertainty.driver import resolve_runner as resolve
    return resolve(spec or DEFAULT_RUNNER)


def copy_run(source, target) -> list[str]:
    """Copy a finished run into ``target`` (everything but status.json, which the target's owner writes); run.json
    gets the target's run id and ``cache: {source_run_id, source_run_dir}``. Returns the copied paths."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source)
        if relative.as_posix() == "status.json" or any(part.startswith(".") for part in relative.parts):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(relative.as_posix())
    document = _run_json(target)
    if document is not None:
        from .core.rundir import write_json_atomic
        original = document.get("run_id")
        document["run_id"] = target.name
        document["cache"] = {"source_run_id": original, "source_run_dir": str(source)}
        write_json_atomic(target / "run.json", document, fsync=True)
    return copied


def run_cached(project_path, options: Mapping | None = None, runs_dir=None, progress_cb=None, *,
               cache: RunCache | None = None, runner: str | None = None, no_cache: bool = False,
               verify: str = "full", run_dir=None, strict_key: bool = True, environment: Mapping | None = None) -> dict:
    """Run a project through the cache (module docstring): the verified cached run of its key, else a new run in
    ``run_dir`` (when given) or a new folder of ``runs_dir`` (default ``<cache root>/runs``, owned by the cache),
    registered when it finished and the key did not change while it ran.

    ``runner``: the simulation's spec (default the loop); ``no_cache``: never reuse (the new run is registered
    and supersedes); ``run_dir`` given and a hit: the cached run is copied into it. ``strict_key=False``: a
    project whose key cannot be computed is run without the cache (a note says why) instead of raising.

    Returns ``{run_dir, run_id, cached, key, registered, source_run_dir, result, verdict, notes}``.
    """
    options = dict(options or {})
    project_path = Path(project_path)
    if project_path.is_dir():
        project_path = project_path / "project.yaml"
    notes: list[str] = []
    simulate = resolve_runner(runner)
    cache = cache if cache is not None else (RunCache() if enabled() else None)
    usable = cache is not None
    if cache is None:
        notes.append("the cache is switched off (CNCTWIN_CACHE=off)")
    elif any(name in options for name in UNCACHED_OPTIONS):
        usable = False
        notes.append("resumes and replays are not cached")
    elif not deterministic_environment():
        usable = False
        notes.append("this process does not run under the deterministic environment (threads 1, PYTHONHASHSEED 0): "
                     "its runs are not cached")
    key0 = None
    if usable:
        try:
            key0 = key_for(project_path, options, runner=runner, environment=environment)
        except InputError as error:
            if strict_key:
                raise
            usable = False
            notes.append(f"no cache key: {error}")
    if not usable:
        target = _target_dir(run_dir, runs_dir, cache, None, options, project_path)
        result = simulate(project_path, target, options, progress_cb)
        return _outcome(target, False, None, False, None, result, notes)
    with cache.lock(key0.key):
        if not no_cache:
            hit = cache.lookup(key0.key, verify=verify)
            if hit is not None:
                return _hit(hit, run_dir, key0, progress_cb, notes)
        else:
            notes.append("no_cache: the cache was not consulted")
        target = _target_dir(run_dir, runs_dir, cache, key0, options, project_path)
        result = simulate(project_path, target, options, progress_cb)
        registered = False
        try:
            key1 = key_for(project_path, options, runner=runner, environment=environment)
        except InputError as error:
            key1 = None
            notes.append(f"not registered: the project no longer resolves ({error})")
        if key1 is not None and key1.key != key0.key:
            notes.append("not registered: the source tree, the databases or the inputs changed while it ran")
        elif key1 is not None:
            if (target / "run.json").is_file():
                try:
                    cache.register(key0, target)
                    registered = True
                except InputError as error:
                    notes.append(f"not registered: {error}")
            else:
                notes.append("not registered: the run wrote no run.json")
    try:
        cache.evict(protect=(key0.key,))
    except OSError as error:
        notes.append(f"eviction failed: {error}")
    return _outcome(target, False, key0.key, registered, None, result, notes)


def _target_dir(run_dir, runs_dir, cache, key0, options, project_path: Path) -> Path:
    from .core import rundir
    if run_dir is not None:
        target = Path(run_dir)
        target.mkdir(parents=True, exist_ok=True)
        return target
    base = Path(runs_dir) if runs_dir is not None else (cache.runs_dir if cache is not None else paths.data_dir() / "runs")
    project_hash = key0.components["project_hash"] if key0 is not None else hashlib.sha256(
        project_path.read_bytes()).hexdigest()
    return rundir.allocate_run_dir(base, project_hash, options, slug=project_path.parent.name)


def _hit(hit: Entry, run_dir, key0: CacheKey, progress_cb, notes: list) -> dict:
    source = Path(hit.run_dir)
    target = source
    if run_dir is not None and Path(run_dir).resolve() != source.resolve():
        target = Path(run_dir)
        copy_run(source, target)
        notes.append(f"copied from the cached run {source}")
    document = _run_json(target) or {}
    summary = document.get("summary") if isinstance(document.get("summary"), Mapping) else {}
    if progress_cb is not None:
        sim = summary.get("sim_time_s")
        progress_cb({"sim_time_s": sim, "est_total_s": sim} if isinstance(sim, (int, float)) else {},
                    phase="cached", message=f"cached run {hit.run_id}")
    return _outcome(target, True, key0.key, False, str(source), None, notes)


def _outcome(run_dir: Path, cached: bool, key, registered: bool, source, result, notes) -> dict:
    document = _run_json(run_dir) or {}
    return {"run_dir": str(Path(run_dir).absolute()), "run_id": str(document.get("run_id") or Path(run_dir).name),
            "cached": cached,
            "key": key, "registered": registered, "source_run_dir": source, "result": result,
            "verdict": document.get("verdict"), "complete": document.get("complete"), "notes": list(notes)}


def register_finished(project_path, run_dir, options: Mapping | None, key0: CacheKey, *, runner: str | None = None,
                      cache: RunCache | None = None, environment: Mapping | None = None) -> dict:
    """Register a run made outside :func:`run_cached` (the API worker runs into the folder the server allocated):
    the key computed before the run, ``key0``, must equal the key now. Returns ``{registered, key, note}``; never
    raises for a cache problem."""
    try:
        cache = cache if cache is not None else RunCache()
        key1 = key_for(project_path, options, runner=runner, environment=environment)
        if key1.key != key0.key:
            return {"registered": False, "key": key0.key,
                    "note": "the source tree, the databases or the inputs changed while it ran"}
        if not (Path(run_dir) / "run.json").is_file():
            return {"registered": False, "key": key0.key, "note": "the run wrote no run.json"}
        cache.register(key0, run_dir)
        cache.evict(protect=(key0.key,))
        return {"registered": True, "key": key0.key, "note": None}
    except (InputError, OSError, ValueError) as error:
        return {"registered": False, "key": key0.key, "note": f"{type(error).__name__}: {error}"}


# =============================================================================== twin cache

def _fmt_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value} B"


def main(argv=None) -> int:
    """``twin cache list|key|verify|evict|compact`` (exit 0; 1 on an error)."""
    parser = argparse.ArgumentParser(prog="twin cache", description="the run cache (docs/ml.md \"Run cache\")")
    parser.add_argument("--root", help="cache folder (default CNCTWIN_CACHE_DIR or DATA_DIR/cache)")
    sub = parser.add_subparsers(dest="command", metavar="command")
    listing = sub.add_parser("list", help="the cached runs, most recently used last")
    listing.add_argument("--json", action="store_true")
    key = sub.add_parser("key", help="the cache key of a project with options, and its components")
    key.add_argument("project")
    key.add_argument("--set", action="append", metavar="KEY=VALUE", help="a run option (JSON value), repeatable")
    key.add_argument("--runner", help="runner spec (default the loop)")
    key.add_argument("--json", action="store_true")
    verify = sub.add_parser("verify", help="verify every entry; drop the ones that do not")
    verify.add_argument("--quick", action="store_true", help="count the result files instead of hashing them")
    evict = sub.add_parser("evict", help="apply the eviction policy now")
    evict.add_argument("--max-gb", type=float, help="size limit of the owned runs (default CNCTWIN_CACHE_MAX_GB or 10)")
    evict.add_argument("--max-age-days", type=float, help="maximum age since the last use (default 90)")
    evict.add_argument("--dry-run", action="store_true")
    sub.add_parser("compact", help="rewrite the index with one line per live entry")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    cache = RunCache(args.root)
    try:
        if args.command == "list":
            entries = sorted(cache.entries(), key=lambda e: (e.used_s, e.key))
            if args.json:
                print(json.dumps([e.to_dict() for e in entries], indent=1, sort_keys=True))
                return 0
            for e in entries:
                print(f"{e.key[:12]}  {e.run_id:40s}  {'owned' if e.owned else 'ref  '}  {_fmt_bytes(e.bytes):>10s}  "
                      f"hits {e.hits:<4d} used {time.strftime('%Y-%m-%d %H:%M', time.gmtime(e.used_s))}Z  {e.run_dir}")
            owned = sum(e.bytes for e in entries if e.owned)
            print(f"{len(entries)} entries; owned runs {_fmt_bytes(owned)} of {_fmt_bytes(cache.max_bytes)}")
            return 0
        if args.command == "key":
            from .cli import _options
            computed = key_for(args.project, _options(args.set), runner=args.runner)
            if args.json:
                print(json.dumps({"key": computed.key, "components": computed.components}, indent=1, sort_keys=True))
            else:
                print(computed.key)
                hit = cache.lookup(computed.key, record_hit=False)
                print(f"cached: {hit.run_dir}" if hit is not None else "not cached")
            return 0
        if args.command == "verify":
            dropped = 0
            for entry in cache.entries():
                ok, reason = cache.verify(entry, "quick" if args.quick else "full")
                if not ok:
                    cache.drop(entry, f"does not verify: {reason}")
                    dropped += 1
                    print(f"dropped {entry.key[:12]} {entry.run_dir}: {reason}")
            print(f"{len(cache.entries())} entries verify, {dropped} dropped")
            return 0
        if args.command == "evict":
            evicted = cache.evict(max_bytes=int(args.max_gb * 1024 ** 3) if args.max_gb is not None else None,
                                  max_age_s=args.max_age_days * 86400.0 if args.max_age_days is not None else None,
                                  dry_run=args.dry_run)
            for e in evicted:
                print(f"{'would evict' if args.dry_run else 'evicted'} {e.key[:12]} {e.run_dir} ({_fmt_bytes(e.bytes)})")
            print(f"{len(evicted)} {'would be ' if args.dry_run else ''}evicted")
            return 0
        if args.command == "compact":
            print(f"{cache.compact()} lines")
            return 0
    except (InputError, OSError, ValueError) as error:
        print(f"twin cache: error: {error}", file=sys.stderr)
        return 1
    return 2


__all__ = ["RunCache", "Entry", "CacheKey", "key_for", "run_cached", "register_finished", "copy_run", "run_digest",
           "build_id", "databases_digest", "database", "canonical_options", "canonical_runner", "environment_contract",
           "current_environment", "deterministic_environment", "default_root", "enabled", "main", "DEFAULT_RUNNER",
           "SWEEP_RUNNER", "DETERMINISTIC_ENV"]

if __name__ == "__main__":
    raise SystemExit(main())
