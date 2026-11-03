"""The run directory (docs/architecture.md 16; layouts in docs/formats.md "Run directory", "frames.bin").

Everything a run writes, and the only thing the API and the UI read:

* identity: :func:`run_id` (deterministic from the project hash, the options and a counter),
  :func:`build_id` (sha256 of the code and the numeric libraries: a checkpoint of another build is
  refused), :func:`numba_cache_key` (the key ``bin/tsafe`` and ``bin/twin`` compute), :func:`versions`,
  :func:`thread_settings`;
* atomic writers: :func:`write_json_atomic`, :func:`write_bytes_atomic` (temporary file in the same
  folder, flush, optional fsync, rename), :func:`npz_bytes` / :func:`write_npz` (deterministic
  ``.npz``: members sorted, dated 1980-01-01), :func:`jsonable` (NumPy values to JSON, non-finite
  floats to null);
* :class:`RunDirectory`: the layout, ``status.json`` (volatile: status, progress, ETA), ``run.json``
  (no volatile fields: :func:`run_document`), ``findings.json``, the outputs written at operation
  ends (registered, so a resume removes those written after its cut);
* :class:`Timeline`: the executed blocks (``timeline.json``) and ``line_ref(block_index)`` for
  :meth:`cnctwin.core.domain.RunContext.finding`;
* :class:`FramesWriter` / :func:`read_frames`: ``frames.bin`` (one packed record per L1 tick; header in
  ``scene.json["frames"]``);
* :class:`SceneWriter`: ``scene.json`` and ``scene/*.glb`` (cnctwin.geom's GLB writer);
* :func:`result_set_digest`: sha256 of every file of the declared result set (architecture 9).

Writers that hold output past a checkpoint implement the checkpoint participant protocol of
:mod:`cnctwin.core.checkpoint` (``checkpoint_cut``, ``checkpoint_state``, ``restore_state``,
``truncate_outputs``).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import FORMAT_VERSION, __version__
from .errors import InputError

TICK_S = 1e-6                                  # architecture 5.1: the base clock is fixed
ZIP_DATE = (1980, 1, 1, 0, 0, 0)               # earliest zip date: containers carry no wall time
RUN_FORMAT = "cnctwin-run"
STATUS_FORMAT = "cnctwin-run-status"
FINDINGS_FORMAT = "cnctwin-findings"
TIMELINE_FORMAT = "cnctwin-timeline"
FRAMES_FORMAT = "cnctwin-frames"
SCENE_FORMAT = "cnctwin-scene"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                    "PYTHONHASHSEED")
LIBRARIES = ("numpy", "scipy", "numba", "llvmlite", "pyarrow")
STATUSES = ("queued", "running", "cancelling", "done", "failed", "cancelled")

# What the determinism test hashes (architecture 9), and what a resume removes when it was written
# after the cut (files of operation ends and of the end of the run).
RESULT_SET = ("findings.json", "trace.jsonl", "series", "frames.bin", "keyframes", "inspection.json",
              "surface.json", "tools.json", "production.json")
RESULT_GLOBS = ("part_*",)
INCREMENTAL = ("frames.bin", "trace.jsonl", "findings.json", "timeline.json", "run.json", "channels.json",
               "scene.json")
INCREMENTAL_DIRS = ("scene", "series", "keyframes", "chips", "checkpoints")
LATE_OUTPUT_GLOBS = ("part_*", "inspection.json", "surface.json", "tools.json", "production.json", "sld/*.json",
                     "report.html", "report.pdf", "setup_sheet.html", "setup_sheet.pdf", "images/*",
                     "video.mp4", "audio.wav")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_KERNEL_PATTERN = re.compile(rb"@njit|@numba|cuda.jit")      # = bin/tsafe's grep -e patterns
_PACKAGE_DIR = Path(__file__).resolve().parents[1]


# =============================================================================== JSON and files

def jsonable(value):
    """Plain JSON value of ``value``: NumPy scalars and arrays, tuples, Paths, Enums and dataclasses
    converted; non-finite floats become ``None`` (JSON has no NaN, and the UI parses with JSON.parse)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return jsonable(value.value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if callable(getattr(value, "to_dict", None)):
            return jsonable(value.to_dict())
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k.value if isinstance(k, enum.Enum) else k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(v) for v in sorted(value, key=repr)]
    if callable(getattr(value, "to_dict", None)):
        return jsonable(value.to_dict())
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def dumps(value, indent: int | None = 1) -> str:
    """Deterministic JSON text (insertion order, UTF-8, no NaN)."""
    return json.dumps(jsonable(value), indent=indent, ensure_ascii=False, allow_nan=False)


def fsync_dir(path: Path) -> None:
    """Make a rename in ``path`` durable (POSIX; a no-op where directories cannot be opened)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_file(path: Path) -> None:
    """fsync a file written through another descriptor (Linux syncs the inode's dirty pages)."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_atomic(path, data: bytes, fsync: bool = False) -> None:
    """Write next to the target and rename over it: a reader sees the old or the new file, never half."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    os.replace(tmp, path)
    if fsync:
        fsync_dir(path.parent)


def write_json_atomic(path, value, fsync: bool = False, indent: int | None = 1) -> None:
    write_bytes_atomic(path, (dumps(value, indent) + "\n").encode("utf-8"), fsync=fsync)


def read_json(path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def npz_bytes(arrays: Mapping[str, np.ndarray], compress: bool = True) -> bytes:
    """A ``.npz`` (numpy.load reads it with ``allow_pickle=False``) whose bytes depend only on the
    arrays: members sorted by name and dated 1980-01-01 (numpy.savez stamps the wall clock)."""
    buffer = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise TypeError(f"array {name!r} has dtype object: .npz files are read without pickle")
            member = io.BytesIO()
            if not array.flags.c_contiguous:
                array = np.array(array, order="C")          # (ascontiguousarray would make 0-d arrays 1-d)
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            info.compress_type = method
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def write_npz(path, arrays: Mapping[str, np.ndarray], compress: bool = True, fsync: bool = False) -> None:
    write_bytes_atomic(path, npz_bytes(arrays, compress), fsync=fsync)


def safe_name(name: str, what: str = "name") -> str:
    """A file-name-safe id (letters, digits, '.', '_', '-'; starting with a letter or digit)."""
    if not isinstance(name, str) or not _SAFE_NAME.match(name) or name in (".", ".."):
        raise ValueError(f"invalid {what} {name!r}: use letters, digits, '.', '_' and '-' (at most 128)")
    return name


def ticks_to_seconds(ticks, tick_s: float = TICK_S):
    """``tick * tick_s`` computed as ``tick / (1 / tick_s)`` when that is an integer, so 1500000 ticks
    of 1 us read 1.5 s exactly (never accumulated, architecture 5.1)."""
    per_second = 1.0 / tick_s
    ticks = np.asarray(ticks, dtype=np.float64)
    if abs(per_second - round(per_second)) < 1e-9 * per_second:
        return ticks / float(round(per_second))
    return ticks * tick_s


def tick_time_s(tick: int, tick_s: float = TICK_S) -> float:
    return float(ticks_to_seconds(np.asarray([tick], np.int64), tick_s)[0])


# =============================================================================== identity

def kernel_sources(package_dir: Path = _PACKAGE_DIR) -> list[str]:
    """Relative paths (``cnctwin/...``) of every source file that defines a kernel, C-sorted: the files
    ``grep -rl --include='*.py' -e '@njit' -e '@numba' -e 'cuda.jit' cnctwin | LC_ALL=C sort`` lists."""
    package_dir = Path(package_dir)
    base = package_dir.parent
    found = []
    for directory, dirnames, filenames in os.walk(package_dir):
        dirnames.sort()
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            try:
                content = path.read_bytes()
            except OSError:
                continue
            if _KERNEL_PATTERN.search(content):
                found.append(path.relative_to(base).as_posix())
    return sorted(found, key=lambda s: s.encode("utf-8"))


def _cpu_model_line() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "model name" in line:
                    return line if line.endswith("\n") else line + "\n"
    except OSError:
        pass
    return ""


def numba_cache_key(package_dir: Path = _PACKAGE_DIR) -> str:
    """The key of the Numba cache directory, exactly as ``bin/tsafe`` and ``bin/twin`` compute it:
    sha256 over the ``sha256sum`` lines of the kernel sources, ``numba llvmlite`` versions and the first
    ``model name`` line of /proc/cpuinfo; the first 16 hex digits."""
    package_dir = Path(package_dir)
    base = package_dir.parent
    lines = [f"{hashlib.sha256((base / rel).read_bytes()).hexdigest()}  {rel}\n" for rel in kernel_sources(package_dir)]
    try:
        import llvmlite
        import numba
        lines.append(f"{numba.__version__} {llvmlite.__version__}\n")
    except ImportError:
        pass
    lines.append(_cpu_model_line())
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()[:16]


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


_BUILD: dict = {}


def build_id(package_dir: Path = _PACKAGE_DIR) -> str:
    """sha256 of every ``cnctwin/**/*.py`` (path and content, C-sorted) and of the library versions: two
    processes with the same build id run the same code on the same numerics (cached per process)."""
    key = str(package_dir)
    if key in _BUILD:
        return _BUILD[key]
    package_dir = Path(package_dir)
    base = package_dir.parent
    digest = hashlib.sha256()
    files = []
    for directory, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        files.extend(Path(directory) / n for n in filenames if n.endswith(".py"))
    for path in sorted(files, key=lambda p: p.relative_to(base).as_posix().encode("utf-8")):
        digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(json.dumps(library_versions(), sort_keys=True).encode("utf-8"))
    _BUILD[key] = digest.hexdigest()
    return _BUILD[key]


def versions() -> dict:
    return {"cnctwin": __version__, "format": FORMAT_VERSION, **library_versions(), "build": build_id()}


def thread_settings(environ: Mapping | None = None) -> dict:
    environ = os.environ if environ is None else environ
    return {name: environ.get(name) for name in THREAD_VARIABLES}


def platform_info() -> dict:
    model = _cpu_model_line().split(":", 1)[-1].strip() if _cpu_model_line() else platform.processor()
    return {"machine": platform.machine(), "system": platform.system(), "cpu": model}


# =============================================================================== run ids

def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return (slug or "run")[:limit].strip("-") or "run"


def canonical_options(options: Mapping | None) -> str:
    """The options as canonical JSON (sorted keys): what the run id hashes. ``resume_from`` is left
    out: a resumed run is the same run."""
    plain = {k: v for k, v in (options or {}).items() if k != "resume_from"}
    return json.dumps(jsonable(plain), sort_keys=True, separators=(",", ":"))


def run_digest(project_hash: str, options: Mapping | None = None) -> str:
    text = f"{project_hash}\n{canonical_options(options)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_id(project_hash: str, options: Mapping | None = None, counter: int = 1, slug: str = "") -> str:
    """``<nnnnn>-<slug>-<h8>``: the counter (5 digits, as the API numbers its runs), the slug of the
    project and the first 8 hex digits of sha256(project hash, canonical options). The same project,
    options and counter always give the same id."""
    if counter < 0:
        raise ValueError("the run counter is not negative")
    head = f"{int(counter):05d}"
    tail = run_digest(project_hash, options)[:8]
    middle = slugify(slug) if slug else ""
    return f"{head}-{middle}-{tail}" if middle else f"{head}-{tail}"


_NUMBERED = re.compile(r"^(\d{5,})-")


def next_counter(runs_dir) -> int:
    """One above the highest numbered run folder in ``runs_dir`` (1 for an empty folder)."""
    runs_dir = Path(runs_dir)
    highest = 0
    if runs_dir.is_dir():
        for path in runs_dir.iterdir():
            match = _NUMBERED.match(path.name)
            if match and path.is_dir():
                highest = max(highest, int(match.group(1)))
    return highest + 1


def allocate_run_dir(runs_dir, project_hash: str, options: Mapping | None = None, slug: str = "") -> Path:
    """Create ``runs_dir/<run_id>`` with the next free counter and return it."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    counter = next_counter(runs_dir)
    while True:
        path = runs_dir / run_id(project_hash, options, counter, slug)
        try:
            path.mkdir()
            return path
        except FileExistsError:
            counter += 1


# =============================================================================== verdict and run.json

def verdict(findings) -> dict:
    """``{safe, reasons}``: safe when no finding has severity error or fatal (architecture 11)."""
    reasons = []
    for finding in findings:
        record = finding.to_dict() if hasattr(finding, "to_dict") else dict(finding)
        severity = str(record.get("severity", ""))
        if severity in ("error", "fatal"):
            where = f"{record.get('op') or '-'} line {record.get('line', 0)}"
            reasons.append(f"{record.get('category')} ({severity}) at {where}: {record.get('message', '')}")
    return {"safe": not reasons, "reasons": reasons}


def finding_counts(findings) -> dict:
    counts = {"info": 0, "warning": 0, "error": 0, "fatal": 0}
    for finding in findings:
        record = finding.to_dict() if hasattr(finding, "to_dict") else dict(finding)
        severity = str(record.get("severity", ""))
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def project_document(project) -> tuple[dict, dict, str | None, str | None]:
    """(project as plain SI data, input hashes {file: {sha256, bytes}}, project dir, project hash) of a
    ResolvedProject (or of a plain mapping, which is taken as already plain)."""
    if isinstance(project, Mapping) and not hasattr(project, "to_dict"):
        return dict(project), {}, None, None
    document = jsonable(project.to_dict())
    for op in document.get("operations", []) or []:
        text = op.get("program_text")
        if isinstance(text, str):
            op["program_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    inputs = {}
    files = getattr(project, "input_files", {}) or {}
    for name, digest in sorted((getattr(project, "input_hashes", {}) or {}).items()):
        path = files.get(name)
        size = Path(path).stat().st_size if path is not None and Path(path).is_file() else None
        inputs[str(name)] = {"sha256": digest, "bytes": size}
    directory = str(project.dir) if getattr(project, "path", None) is not None else None
    return document, inputs, directory, getattr(project, "hash", None)


def run_document(*, run_id: str, project, options: Mapping | None, clock: Mapping, verdict_: Mapping,
                 summary: Mapping, outputs: Sequence[Mapping], seed: int | None = None, complete: bool = True,
                 extra: Mapping | None = None) -> dict:
    """run.json (architecture 16): resolved project, versions, input hashes, numba cache key, thread
    settings, verdict, summary, output index; nothing that changes with wall time or machine load."""
    project_doc, inputs, project_dir, project_hash = project_document(project)
    document = {
        "format": RUN_FORMAT, "version": FORMAT_VERSION, "run_id": run_id, "complete": bool(complete),
        "project": project_doc, "project_hash": project_hash, "project_dir": project_dir,
        "project_file": "project.yaml" if project_dir else None, "inputs": inputs,
        "options": jsonable({k: v for k, v in (options or {}).items() if k != "resume_from"}),
        "versions": versions(), "numba_cache_key": numba_cache_key(), "threads": thread_settings(),
        "platform": platform_info(), "clock": jsonable(dict(clock)),
        "seed": seed if seed is not None else (project_doc.get("seed") if isinstance(project_doc, dict) else None),
        "verdict": jsonable(dict(verdict_)), "summary": jsonable(dict(summary)), "outputs": jsonable(list(outputs)),
    }
    if extra:
        document.update(jsonable(dict(extra)))
    return document


# =============================================================================== the run directory

class RunDirectory:
    """``runs/<run_id>/`` (architecture 16). ``own_status=False`` when a job runner owns status.json
    (the API worker writes it from the progress callback; the loop must not)."""

    def __init__(self, path, *, own_status: bool = True, tick_s: float = TICK_S) -> None:
        self.path = Path(path)
        self.own_status = own_status
        self.tick_s = tick_s
        self.outputs: list[dict] = []          # files written at operation ends: {path, tick}

    # ------------------------------------------------------------------ layout
    def __truediv__(self, relative: str) -> Path:
        return self.path / relative

    @property
    def run_json(self) -> Path:
        return self.path / "run.json"

    @property
    def status_json(self) -> Path:
        return self.path / "status.json"

    @property
    def findings_json(self) -> Path:
        return self.path / "findings.json"

    @property
    def trace_jsonl(self) -> Path:
        return self.path / "trace.jsonl"

    @property
    def frames_bin(self) -> Path:
        return self.path / "frames.bin"

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    def prepare(self) -> None:
        """A fresh run: remove what an earlier run left here (status.json too when it is ours) and
        create the folders. Files the run does not write (the project copy, job files) stay."""
        self.path.mkdir(parents=True, exist_ok=True)
        for name in INCREMENTAL:
            target = self.path / name
            if target.is_file():
                target.unlink()
        for name in INCREMENTAL_DIRS:
            _remove_tree(self.path / name)
        for pattern in LATE_OUTPUT_GLOBS:
            for target in sorted(self.path.glob(pattern)):
                if target.is_file():
                    target.unlink()
        if self.own_status and self.status_json.is_file():
            self.status_json.unlink()
        for name in ("series", "keyframes", "chips", "checkpoints", "scene"):
            (self.path / name).mkdir(parents=True, exist_ok=True)
        self.outputs = []

    # ------------------------------------------------------------------ files
    def write_status(self, status: str, *, sim_time_s: float = 0.0, est_total_s: float | None = None,
                     wall_s: float | None = None, eta_s: float | None = None, **extra) -> None:
        """status.json (volatile: progress and timing live here, never in run.json)."""
        if not self.own_status:
            return
        if status not in STATUSES:
            raise ValueError(f"status {status!r} is not one of {', '.join(STATUSES)}")
        document = {"format": STATUS_FORMAT, "version": FORMAT_VERSION, "status": status,
                    "progress": {"sim_time_s": sim_time_s, "est_total_s": est_total_s, "wall_s": wall_s,
                                 "eta_s": eta_s}}
        for key in ("op", "phase", "message", "checkpoint", "error"):
            document[key] = extra.pop(key, None)
        document.update(extra)
        write_json_atomic(self.status_json, document)

    def write_findings(self, findings) -> None:
        items = findings.to_list() if hasattr(findings, "to_list") else [f.to_dict() for f in findings]
        write_json_atomic(self.findings_json, {"format": FINDINGS_FORMAT, "version": FORMAT_VERSION,
                                               "findings": items})

    def write_run_json(self, document: Mapping, fsync: bool = True) -> None:
        write_json_atomic(self.run_json, document, fsync=fsync)

    def register_output(self, relative: str, tick: int) -> Path:
        """An output written at an operation end or at the end of the run (``part_<op>.glb`` ...). A
        resume keeps the registered ones written before its cut and removes later ones."""
        path = self.path / relative
        if not path.is_file():
            raise FileNotFoundError(f"register_output: {relative} does not exist in {self.path}")
        self.outputs.append({"path": Path(relative).as_posix(), "tick": int(tick)})
        return path

    def outputs_index(self) -> list[dict]:
        """``[{path, bytes, sha256}]`` of every file of the run except run.json, status.json, the
        checkpoints and hidden temporary files (run.json's output index)."""
        out = []
        for path in sorted(p for p in self.path.rglob("*") if p.is_file()):
            relative = path.relative_to(self.path).as_posix()
            if relative in ("run.json", "status.json") or relative.startswith("checkpoints/") \
                    or any(part.startswith(".") for part in relative.split("/")):
                continue
            out.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        return out

    # ------------------------------------------------------------------ checkpoint participant
    def checkpoint_cut(self) -> dict:
        return {"outputs": len(self.outputs)}

    def checkpoint_state(self) -> dict:
        return {"outputs": [dict(o) for o in self.outputs]}

    def restore_state(self, state: Mapping) -> None:
        self.outputs = [dict(o) for o in state.get("outputs", [])]

    def truncate_outputs(self, cut: Mapping) -> None:
        """Remove the late outputs (operation-end and end-of-run files) not registered before the cut."""
        keep = {o["path"] for o in self.outputs[: int(cut.get("outputs", len(self.outputs)))]}
        self.outputs = self.outputs[: int(cut.get("outputs", len(self.outputs)))]
        for pattern in LATE_OUTPUT_GLOBS:
            for target in sorted(self.path.glob(pattern)):
                relative = target.relative_to(self.path).as_posix()
                if target.is_file() and relative not in keep:
                    target.unlink()
        if self.run_json.is_file():
            self.run_json.unlink()


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir() and not child.is_symlink():
            child.rmdir()
        else:
            child.unlink()
    path.rmdir()


# =============================================================================== timeline

@dataclass(frozen=True)
class BlockRef:
    """What a finding needs of the block executing at its tick (RunContext.finding reads op, program,
    line and block)."""

    index: int
    op: str
    program: str
    line: int
    block: str
    path: int = 0
    kind: str = ""
    tick0: int = 0
    n_number: int | None = None
    call_stack: tuple[str, ...] = ()


_TIMELINE_REQUIRED = ("path", "op", "program", "line", "block", "kind", "tick0")


class Timeline:
    """The executed blocks of a run (``timeline.json``): one record per executed NC block of a path,
    ``[tick0, tick1)`` in base ticks, index = position (global over paths, in order of opening).

    Records: ``{index, op, program, line, block, path, kind, tick0, tick1, part, tool?, ticks?, ...}``
    (the format cnctwin.production reads; extra keys are kept). Open records have ``tick1 = -1``;
    ``final: False`` keeps a closed record out of timeline.json until its owner finalises it (motion
    splits feed time into cutting / air cutting one step later). :meth:`line_ref` serves open and
    closed records alike.
    """

    def __init__(self, run_dir=None, tick_s: float = TICK_S) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.tick_s = tick_s
        self.records: list[dict] = []

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ building
    def open(self, *, path: int, op: str, program: str, line: int, block: str, kind: str, tick0: int,
             part: int = 0, tool: int | None = None, n_number: int | None = None, call_stack=(),
             **extra) -> int:
        record = {"index": len(self.records), "op": str(op), "program": str(program), "line": int(line),
                  "block": str(block), "path": int(path), "kind": str(kind), "tick0": int(tick0), "tick1": -1,
                  "part": int(part)}
        if tool is not None:
            record["tool"] = int(tool)
        if n_number is not None:
            record["n_number"] = int(n_number)
        if call_stack:
            record["call_stack"] = [str(c) for c in call_stack]
        record.update(extra)
        self.records.append(record)
        return record["index"]

    def open_ref(self, ref, *, path: int, op: str, kind: str, tick0: int, **extra) -> int:
        """Open a record for a controller :class:`cnctwin.controller.ir.LineRef`."""
        return self.open(path=path, op=op, program=ref.program, line=ref.line, block=ref.block, kind=kind,
                         tick0=tick0, n_number=getattr(ref, "n_number", None),
                         call_stack=tuple(getattr(ref, "call_stack", ()) or ()), **extra)

    def close(self, index: int, tick1: int, ticks: Mapping | None = None, **extra) -> dict:
        record = self._record(index)
        if record["tick1"] >= 0:
            raise ValueError(f"timeline block {index} is already closed (tick1 {record['tick1']})")
        if int(tick1) < record["tick0"]:
            raise ValueError(f"timeline block {index}: tick1 {tick1} is before tick0 {record['tick0']}")
        record["tick1"] = int(tick1)
        if ticks is not None:
            record["ticks"] = _check_ticks(index, ticks)
        record.update(extra)
        return record

    def upsert(self, record: Mapping) -> int:
        """Add or replace a whole record (as cnctwin.motion's Record.to_dict() gives it): ``index`` is the
        next position (a new record) or an existing one (a replacement)."""
        missing = [k for k in _TIMELINE_REQUIRED if k not in record]
        if missing or "index" not in record:
            raise ValueError(f"timeline record lacks {', '.join(['index'] + missing if 'index' not in record else missing)}")
        index = int(record["index"])
        clean = dict(record)
        clean["index"] = index
        clean["tick0"] = int(clean["tick0"])
        clean["tick1"] = int(clean.get("tick1", -1))
        if clean["tick1"] >= 0 and clean["tick1"] < clean["tick0"]:
            raise ValueError(f"timeline block {index}: tick1 {clean['tick1']} is before tick0 {clean['tick0']}")
        if "ticks" in clean and clean["ticks"] is not None:
            clean["ticks"] = _check_ticks(index, clean["ticks"])
        if index == len(self.records):
            self.records.append(clean)
        elif 0 <= index < len(self.records):
            self.records[index] = clean
        else:
            raise ValueError(f"timeline record index {index}: the timeline has {len(self.records)} records")
        return index

    def _record(self, index: int) -> dict:
        if not 0 <= int(index) < len(self.records):
            raise IndexError(f"no timeline block {index} (the timeline has {len(self.records)})")
        return self.records[int(index)]

    # ------------------------------------------------------------------ reading
    def line_ref(self, index: int) -> BlockRef | None:
        """The block with timeline index ``index`` (None for an index the timeline does not have)."""
        if not 0 <= int(index) < len(self.records):
            return None
        r = self.records[int(index)]
        return BlockRef(int(r["index"]), str(r.get("op", "")), str(r.get("program", "")), int(r.get("line", 0)),
                        str(r.get("block", "")), int(r.get("path", 0)), str(r.get("kind", "")),
                        int(r.get("tick0", 0)), r.get("n_number"), tuple(r.get("call_stack", ()) or ()))

    def closed(self) -> list[dict]:
        """Records that are closed and final, in index order, as timeline.json holds them."""
        out = []
        for r in self.records:
            if r["tick1"] >= 0 and r.get("final", True):
                out.append({k: v for k, v in r.items() if k != "final"})
        return out

    def open_records(self) -> list[dict]:
        return [r for r in self.records if r["tick1"] < 0]

    def path_spans(self, now_tick: int) -> dict[int, dict]:
        """Per path: ``block_ticks`` (sum of the block durations, the open block up to now),
        ``span_ticks`` (now minus the path's first tick0), ``open`` (open records) and the categories'
        sum of the closed blocks (``category_ticks``) against their durations (``closed_ticks``)."""
        out: dict[int, dict] = {}
        for r in self.records:
            p = int(r["path"])
            entry = out.setdefault(p, {"block_ticks": 0, "first_tick": r["tick0"], "open": 0, "closed_ticks": 0,
                                       "category_ticks": 0, "categorised_ticks": 0})
            entry["first_tick"] = min(entry["first_tick"], r["tick0"])
            if r["tick1"] >= 0:
                duration = r["tick1"] - r["tick0"]
                entry["block_ticks"] += duration
                entry["closed_ticks"] += duration
                if isinstance(r.get("ticks"), Mapping):
                    entry["category_ticks"] += sum(int(v) for v in r["ticks"].values())
                    entry["categorised_ticks"] += duration
            else:
                entry["block_ticks"] += max(0, int(now_tick) - r["tick0"])
                entry["open"] += 1
        for entry in out.values():
            entry["span_ticks"] = int(now_tick) - entry["first_tick"]
        return out

    def to_json(self) -> dict:
        return {"format": TIMELINE_FORMAT, "version": FORMAT_VERSION, "tick_s": self.tick_s, "blocks": self.closed()}

    def write(self, run_dir=None) -> None:
        target = Path(run_dir) if run_dir is not None else self.run_dir
        if target is None:
            raise ValueError("Timeline.write: no run directory")
        write_json_atomic(target / "timeline.json", self.to_json())

    # ------------------------------------------------------------------ checkpoint participant
    def checkpoint_cut(self) -> dict:
        if self.run_dir is not None:
            self.write()
            fsync_file(self.run_dir / "timeline.json")
        return {"blocks": len(self.records), "closed": len(self.closed())}

    def checkpoint_state(self) -> dict:
        return {"records": [dict(r) for r in self.records]}

    def restore_state(self, state: Mapping) -> None:
        self.records = [dict(r) for r in state.get("records", [])]

    def truncate_outputs(self, cut: Mapping) -> None:
        if len(self.records) != int(cut.get("blocks", len(self.records))):
            raise ValueError(f"timeline state has {len(self.records)} records, the cut {cut.get('blocks')}")
        if self.run_dir is not None:
            self.write()


def _check_ticks(index, ticks: Mapping) -> dict:
    out = {}
    for name, value in ticks.items():
        count = int(value)
        if count < 0 or count != value:
            raise ValueError(f"timeline block {index}: ticks[{name!r}] = {value!r} is not a non-negative integer")
        if count:
            out[str(name)] = count
    return out


# =============================================================================== frames.bin

class FramesWriter:
    """``frames.bin``: one packed little-endian record per L1 tick (docs/formats.md "frames.bin").

    Fields: ``tick`` int64; ``axis`` float32 [n_axes] (linear axes minus their stock-origin value,
    rotary axes wrapped to [0, 2 pi)); ``axis_rev`` int32 [n_axes] (revolutions of rotary axes, 0 for
    linear); per path ``spindle_angle_rad`` float32 (wrapped to [0, 2 pi)), ``spindle_rev`` int32,
    ``tip_S_m`` float32 [3] (tool tip in S) and ``axis_T_S`` float32 [3] (tool axis +z_T in S).
    The header (``scene.json["frames"]``) names the axes with their origin, kind and unit.
    """

    def __init__(self, path, *, axes: Sequence[Mapping], paths: Sequence[Mapping], tick_s: float = TICK_S,
                 ipo_ticks: int = 1000) -> None:
        self.path = Path(path)
        self.axes = [self._axis(a) for a in axes]
        self.paths = [{"path": int(p.get("path", i)), "spindle": str(p.get("spindle", "main"))}
                      for i, p in enumerate(paths)]
        if not self.paths:
            raise ValueError("frames.bin needs at least one path")
        self.tick_s = float(tick_s)
        self.ipo_ticks = int(ipo_ticks)
        n_axes, n_paths = len(self.axes), len(self.paths)
        self.dtype = np.dtype([("tick", "<i8"), ("axis", "<f4", (n_axes,)), ("axis_rev", "<i4", (n_axes,)),
                               ("spindle_angle_rad", "<f4", (n_paths,)), ("spindle_rev", "<i4", (n_paths,)),
                               ("tip_S_m", "<f4", (n_paths, 3)), ("axis_T_S", "<f4", (n_paths, 3))])
        self.origin = np.array([a["origin"] for a in self.axes], dtype=np.float64)
        self.rotary = np.array([a["kind"] == "rotary" for a in self.axes], dtype=bool)
        self._file = None
        self.last_tick: int | None = None

    @staticmethod
    def _axis(axis: Mapping) -> dict:
        kind = str(axis.get("kind", "linear"))
        if kind not in ("linear", "rotary"):
            raise ValueError(f"axis {axis.get('name')!r}: kind {kind!r} is not linear or rotary")
        origin = float(axis.get("origin", 0.0)) if kind == "linear" else 0.0
        return {"name": str(axis["name"]), "kind": kind, "unit": "m" if kind == "linear" else "rad", "origin": origin}

    def header(self) -> dict:
        fields = []
        units = {"tick": "tick", "axis": "m|rad", "axis_rev": "rev", "spindle_angle_rad": "rad", "spindle_rev": "rev",
                 "tip_S_m": "m", "axis_T_S": "-"}
        for name in self.dtype.names:
            sub, offset = self.dtype.fields[name][:2]
            base = sub.base if sub.shape else sub
            fields.append({"name": name, "dtype": base.str, "shape": list(sub.shape), "offset": int(offset),
                           "unit": units[name]})
        return {"format": FRAMES_FORMAT, "version": FORMAT_VERSION, "file": self.path.name,
                "record_bytes": int(self.dtype.itemsize), "byte_order": "little", "tick_s": self.tick_s,
                "ipo_ticks": self.ipo_ticks, "axes": [dict(a) for a in self.axes], "paths": [dict(p) for p in self.paths],
                "fields": fields,
                "doc": "one record per L1 tick: axis values (linear: machine value minus the axis' origin, the "
                       "value at the stock origin; rotary: wrapped to [0, 2 pi) with axis_rev revolutions), per "
                       "path the spindle angle wrapped to [0, 2 pi) with spindle_rev revolutions, the tool tip in "
                       "S and the tool axis in S"}

    def open(self, truncate_to: int | None = None) -> None:
        if truncate_to is None:
            self._file = open(self.path, "wb")
            return
        self._file = open(self.path, "r+b") if self.path.exists() else open(self.path, "w+b")
        self._file.truncate(int(truncate_to))
        self._file.seek(int(truncate_to))

    def _ensure_open(self):
        if self._file is None:
            self.open(truncate_to=self.path.stat().st_size if self.path.exists() else None)
        return self._file

    @staticmethod
    def wrap(angle_rad) -> tuple[np.ndarray, np.ndarray]:
        """(float32 angle in [0, 2 pi), int32 revolutions) of unwrapped angles (float64)."""
        angle = np.asarray(angle_rad, dtype=np.float64)
        rev = np.floor(angle / (2.0 * math.pi))
        remainder = angle - rev * (2.0 * math.pi)
        below = remainder < 0.0                         # angle / 2 pi rounded up to a whole revolution
        rev = rev - below
        remainder = np.where(below, remainder + 2.0 * math.pi, remainder)
        wrapped = remainder.astype(np.float32)
        over = wrapped >= np.float32(2.0 * math.pi)     # rounding to float32 reached 2 pi
        wrapped = np.where(over, np.float32(0.0), wrapped)
        rev = rev + over
        return wrapped.astype(np.float32), rev.astype(np.int32)

    def records(self, ticks, axis_values, spindle_angle_rad, tip_S_m, axis_T_S, spindle_rev=None) -> np.ndarray:
        """Packed records of n ticks. ``axis_values`` [n, n_axes] machine SI values (rotary axes
        unwrapped); ``spindle_angle_rad`` [n, n_paths] unwrapped, or wrapped with ``spindle_rev``;
        ``tip_S_m`` and ``axis_T_S`` [n, n_paths, 3]."""
        ticks = np.asarray(ticks, dtype=np.int64).reshape(-1)
        n = len(ticks)
        n_axes, n_paths = len(self.axes), len(self.paths)
        axis_values = np.asarray(axis_values, dtype=np.float64).reshape(n, n_axes)
        angle = np.asarray(spindle_angle_rad, dtype=np.float64).reshape(n, n_paths)
        tip = np.asarray(tip_S_m, dtype=np.float64).reshape(n, n_paths, 3)
        axis_T = np.asarray(axis_T_S, dtype=np.float64).reshape(n, n_paths, 3)
        out = np.zeros(n, self.dtype)
        out["tick"] = ticks
        linear = axis_values - self.origin[None, :]
        wrapped, revs = self.wrap(axis_values)
        out["axis"] = np.where(self.rotary[None, :], wrapped, linear.astype(np.float32))
        out["axis_rev"] = np.where(self.rotary[None, :], revs, 0)
        if spindle_rev is None:
            out["spindle_angle_rad"], out["spindle_rev"] = self.wrap(angle)
        else:
            unwrapped = angle + 2.0 * math.pi * np.asarray(spindle_rev, dtype=np.float64).reshape(n, n_paths)
            out["spindle_angle_rad"], out["spindle_rev"] = self.wrap(unwrapped)
        out["tip_S_m"] = tip.astype(np.float32)
        out["axis_T_S"] = axis_T.astype(np.float32)
        return out

    def append(self, ticks, axis_values, spindle_angle_rad, tip_S_m, axis_T_S, spindle_rev=None) -> int:
        """Append records (ticks rising, after every tick already written); returns how many."""
        records = self.records(ticks, axis_values, spindle_angle_rad, tip_S_m, axis_T_S, spindle_rev)
        if len(records) == 0:
            return 0
        if np.any(np.diff(records["tick"]) <= 0) or (self.last_tick is not None and records["tick"][0] <= self.last_tick):
            raise ValueError("frames.bin records must be in rising tick order")
        self._ensure_open().write(records.tobytes())
        self.last_tick = int(records["tick"][-1])
        return len(records)

    def append_motion(self, motion, R_S_M=None) -> int:
        """Append the ticks of one step from a :class:`cnctwin.core.bus.MotionFrame`.

        Axis values are taken from path 0's ``axis_actual`` in the header's axis order (by
        ``axis_names``); the tool axis in S per tick is ``PathFrame.axis_T_S`` when the motion record
        carries it, else ``R_S_M @ axis_T_M`` with ``R_S_M`` given per tick [n, 3, 3] or for the step
        [3, 3], else derived from ``pose_S_T @ inv(pose_M_T)`` at the end of the step (exact when the
        stock does not rotate relative to M within the step)."""
        frames = list(motion.paths)
        if not frames:
            return 0
        ticks = np.asarray(frames[0].tick, dtype=np.int64)
        n = len(ticks)
        if n == 0:
            return 0
        first = frames[0]
        names = list(first.axis_names)
        values = np.zeros((n, len(self.axes)))
        for k, axis in enumerate(self.axes):
            source = None
            for frame in frames:
                if axis["name"] in frame.axis_names:
                    source = (frame, list(frame.axis_names).index(axis["name"]))
                    break
            if source is None:
                raise ValueError(f"frames.bin axis {axis['name']!r} is not in the motion record ({', '.join(names)})")
            values[:, k] = np.asarray(source[0].axis_actual, dtype=np.float64)[:, source[1]]
        by_path = {int(f.path): f for f in frames}
        angle = np.zeros((n, len(self.paths)))
        revs = np.zeros((n, len(self.paths)))
        tip = np.zeros((n, len(self.paths), 3))
        axis_T = np.zeros((n, len(self.paths), 3))
        for j, entry in enumerate(self.paths):
            frame = by_path.get(entry["path"])
            if frame is None:
                raise ValueError(f"frames.bin path {entry['path']} is not in the motion record")
            if len(frame.tick) != n or np.any(np.asarray(frame.tick) != ticks):
                raise ValueError(f"path {entry['path']} has other L1 ticks than path {first.path}")
            angle[:, j] = np.asarray(frame.spindle_angle_rad, dtype=np.float64)
            revs[:, j] = np.asarray(frame.spindle_rev, dtype=np.float64)
            tip[:, j] = np.asarray(frame.tip_S_m, dtype=np.float64)
            axis_T[:, j] = self._axis_in_S(frame, n, R_S_M)
        return self.append(ticks, values, angle, tip, axis_T, spindle_rev=revs)

    @staticmethod
    def _axis_in_S(frame, n: int, R_S_M) -> np.ndarray:
        given = getattr(frame, "axis_T_S", None)
        if given is not None and np.shape(given) == (n, 3):
            return np.asarray(given, dtype=np.float64)
        axis_M = np.asarray(frame.axis_T_M, dtype=np.float64).reshape(n, 3)
        if R_S_M is None:
            pose_S_T = np.asarray(frame.pose_S_T, dtype=np.float64)
            pose_M_T = np.asarray(frame.pose_M_T, dtype=np.float64)
            R = (pose_S_T @ np.linalg.inv(pose_M_T))[:3, :3]
            return axis_M @ R.T
        R = np.asarray(R_S_M, dtype=np.float64)
        if R.shape == (3, 3):
            return axis_M @ R.T
        return np.einsum("nij,nj->ni", R.reshape(n, 3, 3), axis_M)

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def byte_length(self) -> int:
        if self._file is not None:
            self._file.flush()
            return int(self._file.tell())
        return self.path.stat().st_size if self.path.exists() else 0

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # ------------------------------------------------------------------ checkpoint participant
    def checkpoint_cut(self) -> dict:
        length = self.byte_length()
        if self._file is not None:
            os.fsync(self._file.fileno())
        return {"bytes": length, "records": length // self.dtype.itemsize}

    def checkpoint_state(self) -> dict:
        return {"last_tick": self.last_tick}

    def restore_state(self, state: Mapping) -> None:
        self.last_tick = state.get("last_tick")

    def truncate_outputs(self, cut: Mapping) -> None:
        length = int(cut["bytes"])
        if length % self.dtype.itemsize:
            raise ValueError(f"frames.bin cut {length} is not a whole number of {self.dtype.itemsize}-byte records")
        self.close()
        if self.path.exists() and self.path.stat().st_size < length:
            raise ValueError(f"frames.bin is shorter ({self.path.stat().st_size} bytes) than its checkpoint cut ({length})")
        self.open(truncate_to=length)


def frames_dtype(header: Mapping) -> np.dtype:
    names, formats, offsets = [], [], []
    for item in header["fields"]:
        shape = tuple(int(n) for n in (item.get("shape") or ()))
        names.append(str(item["name"]))
        formats.append((item["dtype"], shape) if shape else item["dtype"])
        offsets.append(int(item["offset"]))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(header["record_bytes"])})


def read_frames(run_dir, tick0: int | None = None, tick1: int | None = None) -> tuple[np.ndarray, dict]:
    """(records, header) of frames.bin between tick0 and tick1 (inclusive), with the axis values made
    absolute again in ``axis_abs`` ([n, n_axes] float64: linear origin added, rotary unwrapped) and
    the spindle angles unwrapped in ``spindle_unwrapped_rad`` (keys of the returned dict)."""
    run_dir = Path(run_dir)
    scene = read_json(run_dir / "scene.json")
    header = scene["frames"]
    dtype = frames_dtype(header)
    path = run_dir / header.get("file", "frames.bin")
    count = path.stat().st_size // dtype.itemsize if path.exists() else 0
    records = np.fromfile(path, dtype=dtype, count=count) if count else np.zeros(0, dtype)
    if tick0 is not None or tick1 is not None:
        lo = np.searchsorted(records["tick"], tick0, side="left") if tick0 is not None else 0
        hi = np.searchsorted(records["tick"], tick1, side="right") if tick1 is not None else len(records)
        records = records[lo:hi]
    axes = header.get("axes", [])
    origin = np.array([float(a.get("origin", 0.0)) for a in axes])
    rotary = np.array([a.get("kind") == "rotary" for a in axes], dtype=bool)
    values = np.asarray(records["axis"], dtype=np.float64)
    revs = np.asarray(records["axis_rev"], dtype=np.float64) if "axis_rev" in dtype.names else np.zeros_like(values)
    absolute = np.where(rotary[None, :], values + 2.0 * math.pi * revs, values + origin[None, :]) if len(axes) \
        else values
    unwrapped = np.asarray(records["spindle_angle_rad"], dtype=np.float64) \
        + 2.0 * math.pi * np.asarray(records["spindle_rev"], dtype=np.float64)
    return records, {"header": header, "axis_abs": absolute, "spindle_unwrapped_rad": unwrapped}


# =============================================================================== scene

class SceneWriter:
    """``scene.json`` + ``scene/<name>.glb``: the static scene (machine components and links, chain,
    axes, fixture, stock, tool envelopes, nozzles), meshes written with cnctwin.geom's GLB writer."""

    def __init__(self, run_dir) -> None:
        self.run_dir = Path(run_dir)
        self.meshes: dict[str, str] = {}

    def add_mesh(self, name: str, mesh, *, color_rgba=None, extras: Mapping | None = None,
                 quantize: bool = False) -> str:
        """Write ``scene/<name>.glb`` and return its run-relative path. ``mesh``: a cnctwin.geom Mesh, an
        object with ``vertices`` and ``faces`` (metres), or a ``(vertices, faces)`` pair."""
        from ..geom.io.gltf import GlbNode, glb_bytes
        from ..geom.mesh import Mesh

        safe_name(name, "scene mesh name")
        if not isinstance(mesh, Mesh):
            vertices, faces = (mesh if isinstance(mesh, tuple) else (mesh.vertices, mesh.faces))
            mesh = Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), name=name)
        data = glb_bytes([GlbNode(name=name, mesh=mesh, color_rgba=color_rgba, extras=dict(extras) if extras else None)],
                         quantize=quantize)
        relative = f"scene/{name}.glb"
        write_bytes_atomic(self.run_dir / relative, data)
        self.meshes[name] = relative
        return relative

    def write(self, document: Mapping, frames: FramesWriter | Mapping | None = None,
              keyframes: Mapping | None = None) -> Path:
        """scene.json: the caller's document (components referencing ``scene/<name>.glb``), the frames.bin
        header (``frames``) and the keyframe brick layout (``bricks``)."""
        scene = {"format": SCENE_FORMAT, "version": FORMAT_VERSION, "units": "m"}
        scene.update(jsonable(dict(document)))
        scene["meshes"] = dict(sorted(self.meshes.items()))
        if frames is not None:
            scene["frames"] = frames.header() if isinstance(frames, FramesWriter) else jsonable(dict(frames))
        if keyframes is not None:
            scene["bricks"] = jsonable(dict(keyframes))
        target = self.run_dir / "scene.json"
        write_json_atomic(target, scene)
        return target


# =============================================================================== result set

def result_set_files(run_dir) -> list[Path]:
    """Every file of the declared result set (architecture 9), in sorted order."""
    run_dir = Path(run_dir)
    files: list[Path] = []
    for name in RESULT_SET:
        target = run_dir / name
        if target.is_dir():
            files.extend(p for p in target.rglob("*") if p.is_file())
        elif target.is_file():
            files.append(target)
    for pattern in RESULT_GLOBS:
        files.extend(p for p in run_dir.glob(pattern) if p.is_file())
    files = [p for p in files if not any(part.startswith(".") for part in p.relative_to(run_dir).parts)]
    return sorted(set(files), key=lambda p: p.relative_to(run_dir).as_posix())


def result_set_digest(run_dir) -> dict[str, str]:
    """``{relative path: sha256}`` of the result set: two runs are bit-identical when these are equal."""
    run_dir = Path(run_dir)
    return {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in result_set_files(run_dir)}


def project_hash_of(project) -> str:
    """The project hash a checkpoint is bound to (ResolvedProject.hash, or a given hex string)."""
    if isinstance(project, str):
        return project
    value = getattr(project, "hash", None)
    if not isinstance(value, str):
        raise InputError("the project has no hash (resolve it with cnctwin.project.resolve)")
    return value


__all__ = ["RunDirectory", "Timeline", "BlockRef", "FramesWriter", "SceneWriter", "read_frames", "frames_dtype",
           "run_id", "run_digest", "next_counter", "allocate_run_dir", "slugify", "canonical_options",
           "build_id", "numba_cache_key", "kernel_sources", "versions", "library_versions", "thread_settings",
           "platform_info", "verdict", "finding_counts", "run_document", "project_document", "result_set_digest",
           "result_set_files", "jsonable", "dumps", "write_json_atomic", "write_bytes_atomic", "read_json",
           "npz_bytes", "write_npz", "sha256_file", "fsync_dir", "fsync_file", "safe_name", "ticks_to_seconds",
           "tick_time_s", "project_hash_of", "RESULT_SET", "TICK_S", "ZIP_DATE"]
