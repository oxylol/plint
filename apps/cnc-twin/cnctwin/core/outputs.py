"""Post-run outputs of a run directory, and the studies made around runs (docs/outputs.md).

* :func:`finalize_outputs` ``(run_dir, options=None, progress_cb=None) -> dict`` makes every derived output of a
  finished run directory (architecture 16): ``images/*.png`` and ``render.json`` (cnctwin.render: the standard
  pictures, the finished part with each colour layer, the findings where they were raised), ``video.mp4``
  (cnctwin.render, at the speed-up asked for), ``audio.wav`` + ``audio.json`` (cnctwin.sound), and
  ``report.html``/``.pdf`` with ``setup_sheet.html``/``.pdf`` (cnctwin.report, which embeds the pictures, so it
  runs last). Each generator runs in its own subprocess with a memory cap (a systemd scope like bin/tsafe, else
  RLIMIT_DATA) and the deterministic environment (one thread, PYTHONHASHSEED=0, SOURCE_DATE_EPOCH, TZ=UTC; FFmpeg
  writes bit-exact). Each one's files are registered in run.json's output index and its status and reason in
  run.json ``generators``; a failed generator is recorded and its partial files removed, the others go on.
* What to make comes from the project's ``outputs`` section (run.json ``project.outputs``), the run's own
  ``options.outputs`` and the caller's ``options["outputs"]`` on top (:func:`output_settings`).
* Studies around runs (each a thin, tested layer over the module that owns the method):
  :func:`uncertainty_study` (cnctwin.uncertainty with ``run_project`` as the runner, the result attached to the
  run as ``uncertainty.json`` and the report's section: :func:`attach_study`), :func:`compare_study`
  (cnctwin.report.compare of two run directories), :func:`calibration_study` (cnctwin.mdata.calibrate, whose
  loop simulator re-runs ``run_project`` with a parameter overlay) and :func:`replay_window` (``twin replay
  --window``: ``run_project`` asked to replay a window from the nearest checkpoint).
* :func:`resolve_entry` resolves ``module:function`` / ``/path/file.py:function`` (default
  ``cnctwin.core.loop:run_project``, overridden by ``CNCTWIN_ENTRY_SIMULATE``).
* ``python -m cnctwin.core.outputs RUN_DIR [--only report,images] [--no-video] [--speed S] ...``
  (:func:`main`); the generator subprocess is ``python -m cnctwin.core.outputs --generate NAME ...``.

Settings (memory caps, time limits, priority, the findings pictures): ``db/models/outputs.yaml``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from .. import paths
from .errors import InputError, TwinError

GENERATORS = ("images", "video", "audio", "report")        # in the order they run (the report embeds images)
OWNED = {"images": ("images",), "video": ("video.mp4",), "audio": ("audio.wav", "audio.json"),
         "report": ("report.html", "report.pdf", "setup_sheet.html", "setup_sheet.pdf")}
RENDER_MANIFEST = "render.json"           # what each picture and the video show (images and video write it)
UNCERTAINTY_SUMMARY = "uncertainty.json"  # the study's results attached to the run it was made around
STUDY_FOLDER = "uncertainty"              # study.json + uncertainty.html beside them (the report reads these)
SIMULATE_ENTRY = "cnctwin.core.loop:run_project"
OUTPUTS_ENTRY = "cnctwin.core.outputs:finalize_outputs"
OUTPUTS_FORMAT = "cnctwin-outputs"
SUMMARY_FORMAT = "cnctwin-uncertainty-summary"
RENDER_FORMAT = "cnctwin-render"
MEMORY_MODES = ("auto", "systemd", "rlimit", "none")
REPORT_FORMATS = ("html", "pdf")
SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "fatal": 3}
SETTING_KEYS = ("report", "setup_sheet", "images", "findings", "layers", "image_size", "video", "audio", "only",
                "memory_mode", "memory_bytes", "timeout_s")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PASSTHROUGH_ENV = ("DATA_DIR", "NUMBA_ENABLE_CUDASIM", "NUMBA_BOUNDSCHECK", "CUDA_VISIBLE_DEVICES", "XDG_CACHE_HOME",
                   "CNCTWIN_ROOT")
LOCK_FILE = ".outputs.lock"
HASH_CACHE = ".outputs-hashes.json"
PACKAGE_PARENT = Path(__file__).resolve().parents[2]       # the folder that holds the cnctwin package
SIZE_SUFFIX = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


class EntryMissing(TwinError):
    """The callable an entry point names is not in this build."""


class OutputsBusy(TwinError):
    """Another process is making the outputs of this run directory."""


# =============================================================================== entry points

def resolve_entry(spec=None, *, default: str = SIMULATE_ENTRY, env: str | None = "CNCTWIN_ENTRY_SIMULATE"):
    """The callable ``spec`` names (``module:function`` or ``/path/file.py:function``; a callable is returned as it
    is); without one ``$<env>``, else ``default``. :class:`EntryMissing` when it is not in this build."""
    if callable(spec):
        return spec
    spec = spec or (os.environ.get(env) if env else None) or default
    module_name, _, attribute = str(spec).rpartition(":")
    if not module_name or not attribute.isidentifier():
        raise EntryMissing(f"entry point {spec!r}: expected 'module:function' or '/path/file.py:function'")
    if module_name.endswith(".py"):
        path = Path(module_name)
        if not path.is_file():
            raise EntryMissing(f"{spec} is not available: no file {path}")
        name = "cnctwin_entry_" + hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        module = sys.modules.get(name)
        if module is None:
            loader_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(loader_spec)
            sys.modules[name] = module
            try:
                loader_spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
    else:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing = getattr(error, "name", None) or module_name
            if module_name == missing or module_name.startswith(missing + "."):
                raise EntryMissing(f"{module_name}.{attribute} is not available in this build (no module "
                                   f"{missing!r})") from None
            raise
    function = getattr(module, attribute, None)
    if not callable(function):
        raise EntryMissing(f"{module_name}.{attribute} is not available in this build (module {module_name!r} has "
                           f"no callable {attribute!r})")
    return function


def optional_entry(spec=None, *, default: str = SIMULATE_ENTRY, env: str | None = "CNCTWIN_ENTRY_SIMULATE"):
    """:func:`resolve_entry`, or None when the entry is not in this build."""
    try:
        return resolve_entry(spec, default=default, env=env)
    except EntryMissing:
        return None


# =============================================================================== the run directory

def _read_json(path: Path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def run_document(run_dir) -> dict:
    """run.json of a finished run (InputError when there is none)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise InputError(f"{run_dir} is not a run directory")
    path = run_dir / "run.json"
    if not path.is_file():
        raise InputError(f"{run_dir} has no run.json: the run has not finished (outputs are made from a finished "
                         "run directory)")
    try:
        document = _read_json(path)
    except (OSError, ValueError) as error:
        raise InputError(f"run.json cannot be read ({error})", str(path)) from None
    if not isinstance(document, dict):
        raise InputError("run.json does not hold a mapping", str(path))
    return document


def project_path(document: dict, run_dir=None, required: bool = False) -> Path | None:
    """The project file a run was made from (run.json ``project_dir`` / ``project_file``)."""
    folder = document.get("project_dir")
    name = document.get("project_file") or "project.yaml"
    if folder:
        candidate = Path(folder) / name
        if candidate.is_file():
            return candidate
    if required:
        where = f" ({run_dir})" if run_dir is not None else ""
        raise InputError(f"the project of this run{where} is unknown or gone: run.json names project_dir "
                         f"{folder!r} and project_file {name!r}")
    return None


def _index_skip(relative: str) -> bool:
    return relative in ("run.json", "status.json") or relative.startswith("checkpoints/") \
        or any(part.startswith(".") for part in relative.split("/"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(run_dir: Path, files: list[tuple[Path, str]]) -> dict[str, tuple[int, str]]:
    """{relative path: (bytes, sha256)}; a file whose size and modification time (ns) are those recorded in the
    hidden cache ``.outputs-hashes.json`` at its last hashing is not read again (a long run's series and frames
    are hashed once, not after every generator)."""
    cache_path = run_dir / HASH_CACHE
    try:
        cache = _read_json(cache_path) if cache_path.is_file() else {}
    except (OSError, ValueError):
        cache = {}
    cache = cache if isinstance(cache, dict) else {}
    fresh, out = {}, {}
    for file, relative in files:
        stat = file.stat()
        old = cache.get(relative)
        if isinstance(old, list) and len(old) == 3 and old[0] == stat.st_size and old[1] == stat.st_mtime_ns:
            digest = str(old[2])
        else:
            digest = _sha256(file)
        fresh[relative] = [int(stat.st_size), int(stat.st_mtime_ns), digest]
        out[relative] = (int(stat.st_size), digest)
    tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(fresh, sort_keys=True), encoding="utf-8")
    tmp.replace(cache_path)
    return out


def update_index(run_dir, records: dict | None = None) -> dict:
    """Rewrite run.json's output index ``outputs`` ``[{path, bytes, sha256}]`` (every file except run.json,
    status.json, checkpoints/ and hidden files, as cnctwin.core.rundir.RunDirectory.outputs_index) and, given
    ``records``, its ``generators`` ``{name: {status, reason, files}}``; the files of a generator carry its name
    (``generator``). Returns the new document."""
    run_dir = Path(run_dir)
    document = run_document(run_dir)
    if records is not None:
        document["generators"] = {name: records[name] for name in GENERATORS if name in records}
    owners = {}
    for name, record in (document.get("generators") or {}).items():
        for relative in (record or {}).get("files") or []:
            owners[str(relative)] = name
    files = []
    for file in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative = file.relative_to(run_dir).as_posix()
        if not _index_skip(relative):
            files.append((file, relative))
    index = []
    for relative, (size, digest) in _hashes(run_dir, files).items():
        entry = {"path": relative, "bytes": size, "sha256": digest}
        if relative in owners:
            entry["generator"] = owners[relative]
        index.append(entry)
    document["outputs"] = index
    from .rundir import write_json_atomic
    write_json_atomic(run_dir / "run.json", document, fsync=True)
    return document


@contextmanager
def _locked(run_dir: Path):
    handle = open(run_dir / LOCK_FILE, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OutputsBusy(f"another process is making the outputs of {run_dir}") from None
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _remove_owned(run_dir: Path, name: str) -> None:
    for relative in OWNED[name]:
        target = run_dir / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


# =============================================================================== settings

def _params() -> dict:
    from ..db.models import params
    return params("outputs")


def _template_outputs() -> dict:
    from ..db.yamlio import load_file
    from ..project.loader import DS_TEMPLATE
    return dict((load_file(DS_TEMPLATE).data or {}).get("outputs") or {})


def _project_outputs(document: dict):
    project = document.get("project")
    if isinstance(project, dict) and "outputs" in project:
        return project["outputs"]
    path = project_path(document)
    if path is None:
        return None
    from ..db.yamlio import load_file
    try:
        data = load_file(path).data or {}
    except InputError:
        return None
    return data.get("outputs") if isinstance(data, dict) else None


def _merge(base, over):
    if over is None:
        return base
    if isinstance(base, dict) and isinstance(over, dict):
        merged = dict(base)
        for key, value in over.items():
            merged[key] = _merge(base.get(key), value) if isinstance(value, dict) else value
        return merged
    return over


def _bytes(value, what: str) -> int:
    if isinstance(value, bool):
        raise InputError(f"outputs.{what}: {value!r} is not a size")
    if isinstance(value, (int, float)):
        number = int(value)
    else:
        text = str(value).strip().upper().removesuffix("B").removesuffix("I")
        suffix = text[-1:] if text[-1:] in SIZE_SUFFIX and not text[-1:].isdigit() else ""
        try:
            number = int(float(text[: len(text) - len(suffix)]) * SIZE_SUFFIX[suffix])
        except ValueError:
            raise InputError(f"outputs.{what}: {value!r} is not a size (bytes, or 1536M, 1.5G)") from None
    if number <= 0:
        raise InputError(f"outputs.{what}: a memory cap must be positive")
    return number


def _size(value, what: str, even: bool = False):
    if value in (None, False):
        return None
    if isinstance(value, str):
        value = [v for v in value.replace("x", ",").split(",") if v.strip()]
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise InputError(f"outputs.{what}: give [width, height] or [width, height, supersampling]")
    try:
        numbers = [int(v) for v in value]
    except (TypeError, ValueError):
        raise InputError(f"outputs.{what}: {value!r} are not whole numbers") from None
    if any(n < 1 for n in numbers):
        raise InputError(f"outputs.{what}: sizes must be positive")
    if even and (numbers[0] % 2 or numbers[1] % 2):
        raise InputError(f"outputs.{what}: H.264 video needs even width and height")
    return numbers + [None] * (3 - len(numbers))


def _formats(value) -> list[str]:
    if value in (None, False):
        return []
    if value is True:
        return list(REPORT_FORMATS)
    items = [v.strip() for v in value.split(",")] if isinstance(value, str) else list(value)
    out = []
    for item in items:
        if item not in REPORT_FORMATS:
            raise InputError(f"outputs.report: {item!r} is not a report format ({', '.join(REPORT_FORMATS)})")
        if item not in out:
            out.append(item)
    return out


def _video(value) -> dict | None:
    if value in (None, False):
        return None
    from ..render import params as RP
    view = RP.view()
    raw = {} if value is True else value
    if not isinstance(raw, dict):
        raise InputError("outputs.video: give {fps, speed} or false")
    unknown = sorted(set(raw) - {"fps", "speed", "t0_s", "t1_s", "size", "layer"})
    if unknown:
        raise InputError(f"outputs.video: unknown keys {', '.join(unknown)} (fps, speed, t0_s, t1_s, size, layer)")
    try:
        video = {"fps": float(raw.get("fps", view["video_frame_rate_Hz"])),
                 "speed": float(raw.get("speed", view["video_speed"])),
                 "t0_s": float(raw.get("t0_s") or 0.0),
                 "t1_s": None if raw.get("t1_s") is None else float(raw["t1_s"])}
    except (TypeError, ValueError):
        raise InputError("outputs.video: fps, speed, t0_s and t1_s are numbers") from None
    if not (video["fps"] > 0.0 and video["speed"] > 0.0):
        raise InputError("outputs.video: fps and speed must be positive")
    if video["t1_s"] is not None and video["t1_s"] <= video["t0_s"]:
        raise InputError("outputs.video: t1_s must come after t0_s")
    video["size"] = _size(raw.get("size"), "video.size", even=True)
    layer = raw.get("layer")
    if layer is not None:
        from ..render.layers import LAYERS
        if layer not in LAYERS:
            raise InputError(f"outputs.video.layer: {layer!r} is not one of {', '.join(LAYERS)}")
    video["layer"] = layer
    return video


def _audio(value) -> dict | None:
    if value in (None, False):
        return None
    raw = {} if value is True else value
    if not isinstance(raw, dict):
        raise InputError("outputs.audio: give true, false or {listener, sources}")
    unknown = sorted(set(raw) - {"listener", "sources"})
    if unknown:
        raise InputError(f"outputs.audio: unknown keys {', '.join(unknown)} (listener, sources)")
    sources = raw.get("sources")
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    return {"listener": raw.get("listener"), "sources": list(sources) if sources else None}


def _only(value):
    if value in (None, False):
        return None
    items = [v.strip() for v in value.split(",") if v.strip()] if isinstance(value, str) else list(value)
    unknown = [i for i in items if i not in GENERATORS]
    if unknown:
        raise InputError(f"outputs.only: {', '.join(map(str, unknown))} not among {', '.join(GENERATORS)}")
    return [g for g in GENERATORS if g in items]


def _per_generator(value, defaults: dict, convert, what: str) -> dict:
    out = dict(defaults)
    if value is None:
        return out
    if isinstance(value, dict):
        unknown = [k for k in value if k not in GENERATORS]
        if unknown:
            raise InputError(f"outputs.{what}: {', '.join(map(str, unknown))} not among {', '.join(GENERATORS)}")
        for key, item in value.items():
            out[key] = convert(item)
        return out
    return {name: convert(value) for name in GENERATORS}


def simulated_time_s(document: dict, run_dir) -> float:
    """The run's simulated length: run.json ``summary.cycle_time_s``, else the end of its timeline."""
    cycle = (document.get("summary") or {}).get("cycle_time_s")
    if isinstance(cycle, (int, float)) and math.isfinite(cycle):
        return float(cycle)
    tick_s = float((document.get("clock") or {}).get("tick_s") or 1e-6)
    try:
        timeline = _read_json(Path(run_dir) / "timeline.json")
    except (OSError, ValueError):
        return 0.0
    blocks = timeline.get("blocks", []) if isinstance(timeline, dict) else timeline
    ends = [int(b.get("tick1", 0)) for b in blocks or [] if isinstance(b, dict)]
    return max(ends, default=0) * tick_s


def output_settings(run_dir, options: dict | None = None) -> dict:
    """What finalize_outputs makes: the template's ``outputs``, under the project's (run.json
    ``project.outputs``), under the run's ``options.outputs``, under ``options["outputs"]``; normalised
    (``report`` the formats, ``video`` and ``audio`` None or their settings, sizes, caps and time limits)."""
    run_dir = Path(run_dir)
    document = run_document(run_dir)
    raw = _template_outputs()
    for layer in (_project_outputs(document), (document.get("options") or {}).get("outputs"),
                  (options or {}).get("outputs")):
        raw = _merge(raw, layer)
    if raw is False or raw is None:
        raw = {"report": [], "images": False, "video": False, "audio": False}
    if not isinstance(raw, dict):
        raise InputError("outputs: give a mapping (report, images, video, audio, ...) or false")
    unknown = sorted(set(raw) - set(SETTING_KEYS))
    if unknown:
        raise InputError(f"outputs: unknown keys {', '.join(unknown)} ({', '.join(SETTING_KEYS)})")
    p = _params()
    simulated = simulated_time_s(document, run_dir)
    timeout = float(p["timeout_base_s"]) + float(p["timeout_wall_per_sim"]) * simulated
    mode = str(raw.get("memory_mode") or "auto")
    if mode not in MEMORY_MODES:
        raise InputError(f"outputs.memory_mode: {mode!r} is not one of {', '.join(MEMORY_MODES)}")

    def seconds(value) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise InputError(f"outputs.timeout_s: {value!r} is not a number of seconds") from None
        if not number > 0.0:
            raise InputError("outputs.timeout_s: a time limit must be positive")
        return number

    return {
        "report": _formats(raw.get("report")),
        "setup_sheet": bool(raw.get("setup_sheet", True)),
        "images": bool(raw.get("images", False)),
        "findings": bool(raw.get("findings", True)),
        "layers": bool(raw.get("layers", True)),
        "image_size": _size(raw.get("image_size"), "image_size"),
        "video": _video(raw.get("video")),
        "audio": _audio(raw.get("audio")),
        "only": _only(raw.get("only")),
        "memory_mode": mode,
        "memory_bytes": _per_generator(raw.get("memory_bytes"),
                                       {n: int(p[f"{n}_memory_max_bytes"]) for n in GENERATORS},
                                       lambda v: _bytes(v, "memory_bytes"), "memory_bytes"),
        "timeout_s": _per_generator(raw.get("timeout_s"), {n: timeout for n in GENERATORS}, seconds, "timeout_s"),
        "simulated_s": simulated,
    }


def _plan(settings: dict) -> list[tuple[str, str, str | None]]:
    """(generator, action, reason): action ``run``, ``skip`` (not asked for: its files are removed) or ``keep``
    (outside ``only``: left as it is)."""
    wanted = {"images": settings["images"], "video": settings["video"] is not None,
              "audio": settings["audio"] is not None, "report": bool(settings["report"])}
    switches = {"images": "images: false", "video": "video: false", "audio": "audio: false", "report": "report: []"}
    plan = []
    for name in GENERATORS:
        if settings["only"] is not None and name not in settings["only"]:
            plan.append((name, "keep", None))
        elif wanted[name]:
            plan.append((name, "run", None))
        else:
            plan.append((name, "skip", f"not asked for (outputs.{switches[name]})"))
    return plan


# =============================================================================== the generator subprocess

_systemd: dict[str, bool] = {}


def systemd_available() -> bool:
    """systemd runs here and systemd-run may be called (as root, or through passwordless sudo)."""
    if "ok" not in _systemd:
        ok = os.path.isdir("/run/systemd/system") and shutil.which("systemd-run") is not None
        if ok and os.geteuid() != 0:
            try:
                ok = shutil.which("sudo") is not None and subprocess.run(
                    ["sudo", "-n", "systemd-run", "--version"], capture_output=True, timeout=10,
                    check=False).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
        _systemd["ok"] = ok
    return _systemd["ok"]


def memory_mode(requested: str) -> str:
    if requested == "auto":
        return "systemd" if systemd_available() else "rlimit"
    if requested == "systemd" and not systemd_available():
        raise InputError("outputs.memory_mode systemd needs systemd and root or passwordless sudo for systemd-run "
                         "(use rlimit)")
    return requested


def generator_environment() -> dict:
    """The environment of a generator, built from nothing (only HOME and a few settings pass): one thread,
    PYTHONHASHSEED=0, SOURCE_DATE_EPOCH, TZ=UTC, C.UTF-8, the keyed Numba cache, Matplotlib's Agg backend."""
    from .rundir import numba_cache_key
    p = _params()
    python_path = str(PACKAGE_PARENT)
    env = {
        "HOME": os.environ.get("HOME") or "/tmp",
        "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
        "PYTHONPATH": python_path, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR") or str(paths.ROOT / ".numba-cache" / numba_cache_key()),
        "SOURCE_DATE_EPOCH": os.environ.get("SOURCE_DATE_EPOCH") or str(int(p["source_date_epoch_s"])),
        "MPLBACKEND": "Agg",
    }
    for name in THREAD_VARIABLES:
        env[name] = "1"
    for name in PASSTHROUGH_ENV:
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def capped_command(module_args: list[str], mode: str, cap_bytes: int, unit: str | None = None,
                   env: dict | None = None) -> tuple[list[str], dict | None]:
    """``python <module_args>`` in the deterministic environment (:func:`generator_environment`), under a memory
    cap like bin/tsafe: mode ``systemd`` a scope ``sudo -n systemd-run --scope -p MemoryMax=<cap> -p
    MemorySwapMax=<swap>`` (the command then carries the environment: returns (command, None)); ``rlimit`` and
    ``none`` the plain command (the subprocess caps itself in rlimit mode) and the environment for Popen.
    The swap allowance is never more than the cap: a scope allowed more swap than memory pages its whole
    working set in and out instead of failing (an 8 MiB cap with 256 MiB of swap ran for minutes at full CPU)."""
    env = generator_environment() if env is None else env
    python = [sys.executable, *module_args]
    if mode != "systemd":
        return python, env
    swap = min(int(_params()["memory_swap_max_bytes"]), int(cap_bytes))
    command = [] if os.geteuid() == 0 else ["sudo", "-n"]
    command += ["systemd-run", "--scope", "-q", "-p", f"MemoryMax={int(cap_bytes)}", "-p", f"MemorySwapMax={swap}"]
    if unit:
        command.append(f"--unit={unit}")
    if os.geteuid() != 0:
        command += [f"--uid={os.getuid()}", f"--gid={os.getgid()}"]
    command += ["/usr/bin/env", "-i"] + [f"{k}={v}" for k, v in sorted(env.items())] + python
    return command, None


def _command(name: str, run_dir: Path, work: Path, mode: str, cap: int, unit: str | None) -> tuple[list, dict | None]:
    """The command line of the generator subprocess and its environment (None: the command carries it)."""
    return capped_command(["-m", "cnctwin.core.outputs", "--generate", name, "--run-dir", str(run_dir),
                           "--settings", str(work / "settings.json"), "--result", str(work / "result.json"),
                           "--memory-mode", mode, "--memory-bytes", str(int(cap))], mode, cap, unit)


_unit_counter = [0]


def unit_name(run_dir: Path, name: str) -> str:
    """A systemd scope name unique to this process, the run directory and the task."""
    _unit_counter[0] += 1
    tag = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"cnctwin-outputs-{tag}-{name}-{os.getpid()}-{_unit_counter[0]}"


def _signal(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop_process(proc: subprocess.Popen, unit: str | None, grace_s: float) -> None:
    """SIGTERM, then after ``grace_s`` SIGKILL (and, for a systemd scope, ``systemctl kill`` of the scope: sudo
    cannot pass SIGKILL on)."""
    if proc.poll() is not None:
        return
    _signal(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal(proc, signal.SIGKILL)
    if unit:
        command = ([] if os.geteuid() == 0 else ["sudo", "-n"]) + ["systemctl", "kill", "--signal=SIGKILL",
                                                                  f"{unit}.scope"]
        try:
            subprocess.run(command, capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.wait(timeout=max(grace_s, 1.0))
    except subprocess.TimeoutExpired:
        pass


def _tail(path: Path, limit: int) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _exit_reason(code: int | None, mode: str, cap: int) -> str:
    if code is None:
        return "the generator did not end"
    number = -code if code < 0 else (code - 128 if code > 128 else 0)
    if number:
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = f"signal {number}"
        hint = f" (it may have reached its memory cap of {cap / 2 ** 20:.0f} MiB)" \
            if number == signal.SIGKILL and mode != "none" else ""
        return f"the generator was killed by {name}{hint}"
    return f"the generator exited with status {code} without a result"


def _progress(progress_cb, **fields) -> None:
    if progress_cb is not None:
        progress_cb({}, **fields)


def run_generator(name: str, run_dir, settings: dict, progress_cb=None, label: str = "") -> dict:
    """Run one generator in its own capped subprocess and wait for it, calling ``progress_cb`` every
    ``poll_interval_s`` (an exception from it, the API's Cancelled, stops the generator and passes on).
    Returns ``{status: done|failed, reason, files, result, wall_s, cpu_s, peak_rss_bytes, memory_mode,
    memory_bytes, timeout_s, detail}``."""
    run_dir = Path(run_dir)
    p = _params()
    mode = memory_mode(settings["memory_mode"])
    cap = int(settings["memory_bytes"][name])
    timeout_s = float(settings["timeout_s"][name])
    work = Path(tempfile.mkdtemp(prefix=f".outputs-{name}-", dir=run_dir))
    unit = unit_name(run_dir, name) if mode == "systemd" else None
    outcome = {"status": "failed", "reason": None, "files": [], "result": None, "wall_s": 0.0, "cpu_s": None,
               "peak_rss_bytes": None, "peak_vm_bytes": None, "memory_mode": mode, "memory_bytes": cap,
               "timeout_s": timeout_s, "detail": ""}
    try:
        (work / "settings.json").write_text(json.dumps(settings, sort_keys=True), encoding="utf-8")
        limit = cap * int(p["rlimit_data_factor"]) if mode == "rlimit" else cap
        command, env = _command(name, run_dir, work, mode, limit, unit)
        t0 = time.perf_counter()
        timed_out = False
        with open(work / "log.txt", "wb") as log:
            proc = subprocess.Popen(command, env=env, cwd=str(run_dir), stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    try:
                        proc.wait(timeout=float(p["poll_interval_s"]))
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    elapsed = time.perf_counter() - t0
                    if elapsed > timeout_s:
                        timed_out = True
                        stop_process(proc, unit, float(p["kill_grace_s"]))
                        break
                    _progress(progress_cb, phase=f"outputs: {name}", message=f"{label or name}: {elapsed:.0f} s",
                              eta_s=None)
            except BaseException:
                stop_process(proc, unit, float(p["kill_grace_s"]))
                raise
        outcome["wall_s"] = time.perf_counter() - t0
        try:
            result = _read_json(work / "result.json")
        except (OSError, ValueError):
            result = None
        outcome["detail"] = _tail(work / "log.txt", int(p["log_tail_bytes"]))
        if isinstance(result, dict):
            used = result.get("usage") or {}
            outcome["cpu_s"], outcome["peak_rss_bytes"] = used.get("cpu_s"), used.get("peak_rss_bytes")
            outcome["peak_vm_bytes"] = used.get("peak_vm_bytes")
        if timed_out:
            outcome["reason"] = f"timed out after {timeout_s:.0f} s (outputs.timeout_s)"
        elif isinstance(result, dict) and result.get("ok") and proc.returncode == 0:
            outcome.update(status="done", files=list(result.get("files") or []), result=result.get("result"))
        elif isinstance(result, dict) and result.get("error"):
            error = result["error"]
            outcome["reason"] = f"{error.get('type')}: {error.get('message')}"
            outcome["detail"] = error.get("traceback") or outcome["detail"]
        else:
            outcome["reason"] = _exit_reason(proc.returncode, mode, cap)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return outcome


# =============================================================================== generators (in the subprocess)

def _triple(size) -> tuple:
    return tuple(size) if size else (None, None, None)


def _end_time_s(data) -> float:
    return max(data.end_tick - data.ipo_ticks, 0) * data.tick_s


def _part_name(data) -> str:
    op = (data.timeline[-1].get("op") if data.timeline else None) or ""
    return f"part_{op}" if op else "part"


def _picture_size(width, height, ssaa) -> tuple[int, int, int]:
    from ..render import params as RP
    view = RP.view()
    return (int(view["image_width"] if width is None else width), int(view["image_height"] if height is None
                                                                       else height),
            int(view["image_ssaa"] if ssaa is None else ssaa))


FINAL_LAYERS = (("temperature", "Temperatures of the finished part at the last keyframe"),
                ("chip_pile", "Chips settled on the finished part at the last keyframe"),
                ("coolant_reach", "Coolant reach on the finished part at the last keyframe"))


def _layer_has_data(layer: str, fields: dict, chips: dict) -> bool:
    import numpy as np
    from ..render import layers as L
    from ..render import params as RP
    if layer == "temperature":
        for key in ("work_temp_K", "surface_temp_K"):
            values = np.asarray(fields.get(key, []), dtype=np.float64)
            if values.size and np.any(np.isfinite(values)):
                return True
        return False
    if layer == "chip_pile":
        grid = L.pile_grid(fields, chips)
        if grid is None:
            return False
        heights = np.asarray(grid.height_m, dtype=np.float64)
        return bool(np.any(np.isfinite(heights) & (heights > float(RP.view()["pile_min_height_m"]))))
    if layer == "coolant_reach":
        wetted = np.asarray(fields.get("wetted", []), dtype=np.float64)
        return bool(wetted.size and np.any(wetted > 0.0))
    return False


def _layer_pictures(data, standard: list, size: tuple, notes: list) -> list[dict]:
    """The finished part with each colour layer the standard set shows at another time or from another camera
    (temperature, chip pile, coolant reach: their standard pictures are at the time that shows them best)."""
    from dataclasses import replace

    from ..render import layers as L
    from ..render.compose import FrameComposer, FrameOptions
    from ..render.pipeline import render_image
    if not data.keyframes:
        return []
    k = len(data.keyframes) - 1
    fields, chips = data.fields(k), data.chips(k)
    end = _end_time_s(data)
    composer = FrameComposer(data)
    prefix = _part_name(data)
    out = []
    for layer, caption in FINAL_LAYERS:
        if not _layer_has_data(layer, fields, chips):
            continue
        if any(e.get("layer") == layer and e.get("keyframe") == k and e.get("camera") == "part" for e in standard):
            continue
        pile = layer == "chip_pile"
        options = FrameOptions(camera="part", layer=layer, chips=pile, pile=pile, coolant=False, cues=False,
                               finding_hold_s=0.0)
        options = replace(options, colour_range=L.run_range(data, layer))
        name = f"{prefix}_{layer}.png"
        target = data.path / "images" / name
        try:
            entry = render_image(data, end, target, options, *size, composer)
        except Exception as error:          # one picture lost, not the set (the reason is in render.json)
            notes.append(f"{name}: {type(error).__name__}: {error}")
            target.unlink(missing_ok=True)
            continue
        if entry["counts"].get("opaque_faces", 0) == 0:
            target.unlink(missing_ok=True)
            continue
        entry.update({"file": f"images/{name}", "caption": caption})
        out.append(entry)
    return out


def _stock_pose_W_S(data, op: str | None):
    import numpy as np
    for item in (data.project.get("operations") or []):
        if isinstance(item, dict) and op is not None and item.get("name") == op and item.get("stock_pose_W_S"):
            return np.asarray(item["stock_pose_W_S"], dtype=np.float64).reshape(4, 4)
    stock = data.scene.get("stock") or {}
    return np.asarray(stock.get("pose_W_S", np.eye(4)), dtype=np.float64).reshape(4, 4)


def positioned_findings(data) -> list[dict]:
    """Findings with a finite tool position, with that position in the stock frame S of their operation
    (``pos_S = stock_pose_W_S^-1 pos_W``), worst first (severity, then time, then id)."""
    import numpy as np
    document = None
    try:
        document = _read_json(data.path / "findings.json")
    except (OSError, ValueError):
        return []
    items = document.get("findings", []) if isinstance(document, dict) else (document or [])
    out = []
    for finding in items:
        if not isinstance(finding, dict):
            continue
        position = finding.get("pos_W_m")
        try:
            pos_W = np.asarray(position, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(pos_W)):
            continue
        pose = _stock_pose_W_S(data, finding.get("op"))
        pos_S = np.linalg.solve(pose, np.append(pos_W, 1.0))[:3]
        out.append({"id": int(finding.get("id", len(out))), "severity": str(finding.get("severity", "info")),
                    "category": str(finding.get("category", "")), "op": finding.get("op"),
                    "line": finding.get("line"), "time_s": float(finding.get("time_s") or 0.0),
                    "tick": int(finding.get("tick") or 0), "message": str(finding.get("message", "")),
                    "pos_S": pos_S})
    out.sort(key=lambda f: (-SEVERITY_RANK.get(f["severity"], 0), f["time_s"], f["id"]))
    return out


def _severity_srgb(severity: str, p: dict) -> tuple:
    if severity in ("error", "fatal"):
        return tuple(float(v) for v in p["finding_error_srgb"])
    if severity == "warning":
        return tuple(float(v) for v in p["finding_warning_srgb"])
    return tuple(float(v) for v in p["finding_info_srgb"])


def _marker(name: str, centre, radius_m: float, srgb: tuple):
    """An octahedron at ``centre`` (counter-clockwise faces seen from outside), drawn over everything it is not
    hidden by, casting no shadow."""
    import numpy as np
    from ..render.scene import FLAT, Material, RenderMesh
    corners = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return RenderMesh(name, np.asarray(centre, dtype=np.float64) + radius_m * corners, faces,
                      Material(srgb=srgb, shading=FLAT, clip=False, cast_shadow=False, receive_shadow=False))


def _finding_line(number: int, finding: dict) -> str:
    where = f"{finding['op'] or '-'} line {finding['line'] if finding['line'] is not None else '-'}"
    return f"{number}  {finding['severity']}  {finding['category']}  {where}  t = {finding['time_s']:.3f} s"


def _picture_entry(name: str, caption: str, image, info, cam, options, extra: dict) -> dict:
    return {"file": f"images/{name}", "caption": caption, "time_s": info.time_s, "tick": info.tick,
            "keyframe": info.keyframe, "camera": options.camera, "layer": options.layer, "section": None,
            "tool": info.tool_number, "notes": list(info.notes), "counts": dict(info.counts),
            "camera_detail": cam.to_dict(), "width": int(image.shape[1]), "height": int(image.shape[0]), **extra}


def _findings_overview(data, marked: list[dict], size: tuple, p: dict) -> dict | None:
    """findings.png: the finished part with a numbered marker where each finding was raised (its tool position)
    and the legend under the title."""
    import numpy as np
    from dataclasses import replace

    from ..render import camera as CAM
    from ..render import output as O
    from ..render import params as RP
    from ..render.compose import FINDING_LABEL_FIRST_ROW, FrameComposer, FrameOptions
    from ..render.raster import render
    from ..render.scene import Label
    width, height, ssaa = _picture_size(*size)
    composer = FrameComposer(data)
    tick = int(round(_end_time_s(data) / data.tick_s))
    options = FrameOptions(camera="part", chips=False, pile=False, coolant=False, cues=False, labels=False,
                           finding_hold_s=0.0)
    options = replace(options, aspect=width / height, height_px=height * ssaa)
    scene, cam, info = composer.compose(tick, options)
    points = np.array([f["pos_S"] for f in marked], dtype=np.float64)
    lo, hi = points.min(axis=0), points.max(axis=0)
    part = next((m for m in scene.meshes if m.name == "part"), None)
    if part is not None and part.bounds() is not None:
        lo, hi = np.minimum(lo, part.bounds()[0]), np.maximum(hi, part.bounds()[1])
    radius = float(p["finding_marker_radius_frac"]) * max(float(np.linalg.norm(hi - lo)), 1e-6)
    view = RP.view()
    size_px = float(view["label_font_size"])
    for number, finding in enumerate(marked, start=1):
        colour = _severity_srgb(finding["severity"], p)
        scene.add(_marker(f"finding_{finding['id']}", finding["pos_S"], radius, colour))
        scene.labels.append(Label(str(number), anchor_m=finding["pos_S"], xy_px=(0.8 * size_px, -0.8 * size_px),
                                  srgb=colour))
    for row, (number, finding) in enumerate(list(enumerate(marked, start=1))[: int(p["finding_legend_rows"])]):
        scene.labels.append(Label(_finding_line(number, finding), xy_px=(size_px, (FINDING_LABEL_FIRST_ROW + 1.6 * row)
                                                                          * size_px),
                                  srgb=_severity_srgb(finding["severity"], p), scale_with_height=True))
    scene.title = f"{data.project.get('name', '')}: where the findings were raised".strip(": ")
    scene.subtitle = f"{len(marked)} findings with a position, on the part after {_part_name(data)[5:] or 'the run'}"
    direction = CAM.direction(float(view["overview_azimuth_rad"]), float(view["overview_elevation_rad"]))
    cam = CAM.fit(lo - radius, hi + radius, direction, projection=options.projection, aspect=options.aspect)
    result = render(scene, cam, width, height, ssaa=ssaa, shadow_focus=composer.focus(cam, options, scene))
    if result.stats.get("opaque_faces", 0) == 0:
        return None
    O.write_png(data.path / "images" / "findings.png", result.image)
    info.counts.update(result.stats)
    caption = (f"Where the findings were raised (the tool position at the time), numbered worst first: "
               f"{len(marked)} with a position")
    return _picture_entry("findings.png", caption, result.image, info, cam, options,
                          {"findings": [f["id"] for f in marked]})


def _finding_picture(data, finding: dict, size: tuple, p: dict, composer) -> dict | None:
    """finding_<id>.png: the moment a finding was raised, the camera following the tool, a marker at the
    position (the part camera when the position is outside the follow view)."""
    import numpy as np
    from dataclasses import replace

    from ..render import camera as CAM
    from ..render import output as O
    from ..render import params as RP
    from ..render.compose import FrameOptions
    from ..render.raster import project_points, render
    from ..render.scene import Label
    width, height, ssaa = _picture_size(*size)
    hold = float(RP.view()["finding_label_hold_s"])
    options = FrameOptions(camera="follow", finding_hold_s=hold, pile=finding["category"].startswith("chip"))
    options = replace(options, aspect=width / height, height_px=height * ssaa)
    tick = int(finding["tick"]) if finding["tick"] else int(round(finding["time_s"] / data.tick_s))
    scene, cam, info = composer.compose(tick, options)
    colour = _severity_srgb(finding["severity"], p)
    xy, w = project_points(cam.matrix(options.aspect), finding["pos_S"][None, :], width, height)
    inside = bool(w[0] > 0.0 and 0.0 <= xy[0, 0] < width and 0.0 <= xy[0, 1] < height)
    if inside:
        radius = float(p["finding_marker_radius_frac"]) * float(cam.view_height_m)
    else:
        view = RP.view()
        part = next((m for m in scene.meshes if m.name == "part"), None)
        lo = hi = finding["pos_S"]
        if part is not None and part.bounds() is not None:
            lo, hi = np.minimum(lo, part.bounds()[0]), np.maximum(hi, part.bounds()[1])
        radius = float(p["finding_marker_radius_frac"]) * max(float(np.linalg.norm(hi - lo)), 1e-6)
        direction = CAM.direction(float(view["overview_azimuth_rad"]), float(view["overview_elevation_rad"]))
        options = replace(options, camera="part")
        cam = CAM.fit(lo - radius, hi + radius, direction, projection=options.projection, aspect=options.aspect)
    scene.add(_marker(f"finding_{finding['id']}", finding["pos_S"], radius, colour))
    size_px = float(RP.view()["label_font_size"])
    scene.labels.append(Label(f"finding {finding['id']}", anchor_m=finding["pos_S"],
                              xy_px=(0.8 * size_px, 0.9 * size_px), srgb=colour))
    scene.title = f"Finding {finding['id']}: {finding['category']} ({finding['severity']})"
    result = render(scene, cam, width, height, ssaa=ssaa, shadow_focus=composer.focus(cam, options, scene))
    if result.stats.get("opaque_faces", 0) == 0:
        return None
    name = f"finding_{finding['id']}.png"
    O.write_png(data.path / "images" / name, result.image)
    info.counts.update(result.stats)
    caption = (f"Finding {finding['id']} ({finding['severity']}, {finding['category']}) at "
               f"{finding['op'] or '-'} line {finding['line']}, t = {finding['time_s']:.3f} s: {finding['message']}")
    return _picture_entry(name, caption, result.image, info, cam, options, {"findings": [finding["id"]]})


def _finding_pictures(data, size: tuple, notes: list) -> list[dict]:
    from ..render.compose import FrameComposer
    p = _params()
    marked = positioned_findings(data)
    if not marked:
        return []
    out = []
    try:
        overview = _findings_overview(data, marked[: int(p["finding_markers_max"])], size, p)
        if overview is not None:
            out.append(overview)
    except Exception as error:              # one picture lost, not the set (the reason is in render.json)
        notes.append(f"findings.png: {type(error).__name__}: {error}")
    threshold = SEVERITY_RANK.get(str(p["finding_picture_min_severity"]), 1)
    chosen = [f for f in marked if SEVERITY_RANK.get(f["severity"], 0) >= threshold][: int(p["finding_pictures_max"])]
    composer = FrameComposer(data) if chosen else None
    for finding in chosen:
        try:
            entry = _finding_picture(data, finding, size, p, composer)
        except Exception as error:
            notes.append(f"finding_{finding['id']}.png: {type(error).__name__}: {error}")
            continue
        if entry is not None:
            out.append(entry)
    return out


def generate_images(run_dir, settings: dict) -> dict:
    """images/*.png: the standard set of cnctwin.render (setup, cut, finished part whole and sectioned, one
    picture per colour layer at the time that shows it best), the finished part with the layers whose standard
    picture is taken at another time, and the findings where they were raised."""
    from ..render.pipeline import render_images
    from ..render.rundir import RunData
    data = RunData(run_dir)
    size = _triple(settings.get("image_size"))
    entries = render_images(data, None, *size)
    notes: list[str] = []
    if settings.get("layers", True):
        entries += _layer_pictures(data, entries, size, notes)
    if settings.get("findings", True):
        entries += _finding_pictures(data, size, notes)
    return {"result": {"images": entries, "notes": notes}, "files": [e["file"] for e in entries]}


def generate_video(run_dir, settings: dict) -> dict:
    from ..render.compose import FrameOptions
    from ..render.pipeline import render_video
    video = settings["video"]
    width, height, ssaa = _triple(video.get("size"))
    entry = render_video(run_dir, None, speed=video["speed"], fps=video["fps"], t0_s=video["t0_s"],
                         t1_s=video["t1_s"], options=FrameOptions(camera="follow", layer=video["layer"]),
                         width=width, height=height, ssaa=ssaa)
    return {"result": {"video": entry}, "files": ["video.mp4"]}


def generate_audio(run_dir, settings: dict) -> dict:
    from ..sound import render_audio
    audio = settings["audio"]
    summary = render_audio(run_dir, None, audio.get("sources"), audio.get("listener"))
    return {"result": {"audio": summary}, "files": ["audio.wav", "audio.json"]}


def generate_report(run_dir, settings: dict) -> dict:
    from ..report import build_report
    run_dir = Path(run_dir)
    result = build_report(run_dir, None, formats=tuple(settings["report"]), setup_sheet=settings["setup_sheet"])
    files = [Path(f).resolve().relative_to(run_dir.resolve()).as_posix() for f in result["files"]]
    keep = ("report_pdf_pages", "setup_sheet_pdf_pages", "verdict")
    return {"result": {k: result[k] for k in keep if k in result} | {"missing": len(result.get("missing") or [])},
            "files": files}


GENERATOR_FUNCTIONS = {"images": generate_images, "video": generate_video, "audio": generate_audio,
                       "report": generate_report}


def cap_memory(limit: int) -> None:
    """RLIMIT_DATA (else RLIMIT_AS) of this process (rlimit mode: before anything large is imported)."""
    import resource
    for name in ("RLIMIT_DATA", "RLIMIT_AS"):
        kind = getattr(resource, name, None)
        if kind is None:
            continue
        _, hard = resource.getrlimit(kind)
        cap = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        try:
            resource.setrlimit(kind, (cap, cap))
            return
        except (ValueError, OSError):
            continue


def _status_bytes(key: str) -> int | None:
    """A ``kB`` field of /proc/self/status (VmPeak: peak address space, VmData: data segment) in bytes."""
    try:
        with open("/proc/self/status", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def usage() -> dict:
    """CPU seconds (with finished children: FFmpeg), peak resident and address-space bytes of this process."""
    import resource
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"cpu_s": own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime,
            "peak_rss_bytes": int(max(own.ru_maxrss, children.ru_maxrss)) * 1024,
            "peak_vm_bytes": _status_bytes("VmPeak"), "data_bytes": _status_bytes("VmData")}


def warm_blas() -> None:
    """Let OpenBLAS allocate its work buffer before an rlimit cap is set. Under RLIMIT_DATA a buffer it cannot map
    makes it retry the mapping forever (a generator capped at 16 MiB spun at full CPU for half an hour instead of
    failing); a buffer mapped beforehand is reused by every later call (one thread)."""
    try:
        import numpy as np
        square = np.ones((256, 256))
        float((square @ square)[0, 0])
    except Exception:
        pass


def _traceback_text() -> str:
    """The current exception's traceback, or "" when formatting it runs out of memory itself."""
    try:
        return traceback.format_exc()
    except MemoryError:
        return ""


def generator_main(name: str, run_dir: str, settings_path: str, result_path: str, mode: str, cap: int) -> int:
    """The generator subprocess: cap the memory (rlimit mode) before anything large is imported (only NumPy, whose
    BLAS buffer :func:`warm_blas` maps first), lower the priority, run the generator, write its result (or its
    error) as JSON."""
    import resource  # noqa: F401 (usage() reads it, also after a MemoryError)
    from .rundir import write_json_atomic       # before the cap: the error record must still be writable
    if mode == "rlimit":
        warm_blas()
        cap_memory(int(cap))
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    record: dict
    try:
        nice = int(_params()["generator_nice"])
        try:
            os.nice(max(0, nice - os.nice(0)))
        except OSError:
            pass
        produced = GENERATOR_FUNCTIONS[name](run_dir, settings)
        record = {"ok": True, "result": produced.get("result"), "files": produced.get("files", [])}
    except MemoryError:
        limit = f"the RLIMIT_DATA limit of {int(cap) / 2 ** 20:.0f} MiB" if mode == "rlimit" else \
            f"the memory cap of {int(cap) / 2 ** 20:.0f} MiB"
        record = {"ok": False, "error": {"type": "MemoryError", "message": f"out of memory ({limit}; "
                                                                         "outputs.memory_bytes raises it)",
                                         "traceback": _traceback_text()}}
    except BaseException as error:          # the parent records any failure of the generator
        record = {"ok": False, "error": {"type": type(error).__name__, "message": str(error),
                                         "traceback": _traceback_text()}}
    record["usage"] = usage()
    write_json_atomic(result_path, record)
    return 0 if record["ok"] else 1


# =============================================================================== finalize_outputs

def _write_render_manifest(run_dir: Path, ran: dict) -> None:
    """render.json (cnctwin-render): the images part from the images generator, the video part from the video
    generator; a part whose generator did not run this time is kept."""
    from .rundir import jsonable
    path = run_dir / RENDER_MANIFEST
    try:
        manifest = _read_json(path) if path.is_file() else {}
    except (OSError, ValueError):
        manifest = {}
    manifest = manifest if isinstance(manifest, dict) else {}
    document = {"format": RENDER_FORMAT, "version": 1, "images": manifest.get("images") or [],
                "video": manifest.get("video"), "notes": manifest.get("notes") or []}
    if "images" in ran:
        result = ran["images"] or {}
        document["images"] = result.get("images") or []
        document["notes"] = result.get("notes") or []
    if "video" in ran:
        document["video"] = (ran["video"] or {}).get("video")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(jsonable(document), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def finalize_outputs(run_dir, options: dict | None = None, progress_cb=None) -> dict:
    """Make every derived output of a finished run directory (module docstring); returns
    ``{format, run_dir, settings, generators: {name: {status, reason, files, wall_s, cpu_s, peak_rss_bytes, ...}},
    files, failed, wall_s}``. ``options`` are the run's options: ``options["outputs"]`` overrides the project's
    outputs section (docs/outputs.md). Generator failures are recorded, never raised; an input error (no run.json,
    a bad setting) raises InputError; ``progress_cb`` is called while generators run and may raise (cancel)."""
    t0 = time.perf_counter()
    run_dir = Path(run_dir).absolute()
    settings = output_settings(run_dir, options)
    plan = _plan(settings)
    to_run = [name for name, action, _ in plan if action == "run"]
    summary = {"format": OUTPUTS_FORMAT, "version": 1, "run_dir": str(run_dir), "settings": settings,
               "generators": {}, "files": [], "failed": [], "wall_s": 0.0}
    actions = {name: action for name, action, _ in plan}
    with _locked(run_dir):
        records = dict(run_document(run_dir).get("generators") or {})
        for name, action, reason in plan:
            if action == "keep":
                continue
            _remove_owned(run_dir, name)
            records.pop(name, None)
            if action == "skip":
                records[name] = {"status": "skipped", "reason": reason, "files": []}
                summary["generators"][name] = {"status": "skipped", "reason": reason, "files": []}
        # render.json: the part of a generator that is not asked for any more is emptied
        ran: dict = {name: None for name in ("images", "video") if actions[name] == "skip"}
        if actions["images"] == "skip" and actions["video"] == "skip":
            (run_dir / RENDER_MANIFEST).unlink(missing_ok=True)
        elif ran and (run_dir / RENDER_MANIFEST).is_file():
            _write_render_manifest(run_dir, ran)
        update_index(run_dir, records)
        for position, name in enumerate(to_run):
            label = f"{name} ({position + 1} of {len(to_run)})"
            _progress(progress_cb, phase=f"outputs: {name}", message=label, outputs_done=position,
                      outputs_total=len(to_run), eta_s=None)
            outcome = run_generator(name, run_dir, settings, progress_cb, label)
            done = outcome["status"] == "done"
            if not done:
                _remove_owned(run_dir, name)
                outcome["files"] = []
                summary["failed"].append(name)
            if name in ("images", "video"):
                ran[name] = outcome["result"] if done else None
                _write_render_manifest(run_dir, ran)
                if done:
                    outcome["files"] = outcome["files"] + [RENDER_MANIFEST]
            records[name] = {"status": outcome["status"], "reason": outcome["reason"], "files": outcome["files"]}
            update_index(run_dir, records)
            summary["generators"][name] = outcome
            summary["files"].extend(f for f in outcome["files"] if f not in summary["files"])
    summary["wall_s"] = time.perf_counter() - t0
    statuses = {name: (summary["generators"].get(name) or {}).get("status", "kept") for name in GENERATORS}
    _progress(progress_cb, phase="outputs: done", message=", ".join(f"{k} {v}" for k, v in statuses.items()),
              outputs_done=len(to_run), outputs_total=len(to_run), outputs=statuses)
    return summary


# =============================================================================== studies

def _write_sorted_json(path: Path, document) -> None:
    from .rundir import jsonable
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(jsonable(document), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def study_summary(document: dict) -> dict:
    """``uncertainty.json`` of a run: the results of the study made around it, without the per-sample values
    (those stay in ``uncertainty/study.json``) and without paths, so the same study gives the same bytes."""
    results = document.get("results") or {}

    def without_samples(items):
        return [{k: v for k, v in item.items() if k != "samples"} for item in items or [] if isinstance(item, dict)]

    factors = [{k: f.get(k) for k in ("name", "label", "unit", "nominal", "distribution", "stated_by", "applies")}
               for f in document.get("factors") or [] if isinstance(f, dict)]
    sensitivity = [{"output": s.get("output"), "kind": s.get("kind"), "ranking": (s.get("ranking") or [])[:3]}
                   for s in results.get("sensitivity") or [] if isinstance(s, dict)]
    return {"format": SUMMARY_FORMAT, "version": 1, "study": f"{STUDY_FOLDER}/study.json",
            "page": f"{STUDY_FOLDER}/uncertainty.html",
            "project": {k: (document.get("project") or {}).get(k) for k in ("name", "sha256")},
            "settings": document.get("settings"), "factors": factors, "options_only": document.get("options_only"),
            "results": {"samples_ok": results.get("samples_ok"), "samples_failed": results.get("samples_failed"),
                        "failed": [{"index": f.get("index"), "error": f.get("error")}
                                   for f in results.get("failed") or []],
                        "verdict": results.get("verdict"), "dimensions": without_samples(results.get("dimensions")),
                        "scalars": without_samples(results.get("scalars")), "findings": results.get("findings"),
                        "sensitivity": sensitivity}}


def attach_study(run_dir, document: dict, *, progress_cb=None, refresh: bool = True, outputs_entry=None) -> dict:
    """Attach an uncertainty study to the run it was made around: ``uncertainty/study.json`` and
    ``uncertainty/uncertainty.html`` (what the report and the UI read), ``uncertainty.json`` (the summary), and
    with ``refresh`` the report made again (its Uncertainty section) through ``outputs_entry`` (default this
    module's finalize_outputs, only the report); run.json's output index is updated either way."""
    from ..uncertainty.driver import write_outputs
    run_dir = Path(run_dir)
    run_document(run_dir)
    write_outputs(document, run_dir / STUDY_FOLDER)
    summary = study_summary(document)
    _write_sorted_json(run_dir / UNCERTAINTY_SUMMARY, summary)
    report = None
    if refresh:
        finalize = resolve_entry(outputs_entry, default=OUTPUTS_ENTRY, env=None) if outputs_entry else finalize_outputs
        report = finalize(run_dir, {"outputs": {"only": ["report"]}}, progress_cb)
    else:
        update_index(run_dir)
    return {"summary": summary, "report": report, "files": [UNCERTAINTY_SUMMARY, f"{STUDY_FOLDER}/study.json",
                                                            f"{STUDY_FOLDER}/uncertainty.html"]}


def _study_dir(run_dir: Path, kind: str, out_dir) -> Path:
    return Path(out_dir) if out_dir is not None else run_dir.with_name(f"{run_dir.name}.{kind}")


def uncertainty_study(run_dir, options: dict | None = None, progress_cb=None, *, out_dir=None, runner=None,
                      refresh: bool = True) -> dict:
    """An uncertainty study around a finished run: cnctwin.uncertainty plans the samples from the project's
    stated tolerances, writes each variant project to its own folder and runs them one after the other through
    ``runner`` (default :func:`resolve_entry`: cnctwin.core.loop.run_project) with ``progress_cb`` passed on; the
    results are attached to the run (:func:`attach_study`). The study folder is ``out_dir`` (default
    ``<run_dir>.uncertainty`` beside the run)."""
    run_dir = Path(run_dir)
    document = run_document(run_dir)
    project = project_path(document, run_dir, required=True)
    options = dict(options or {})
    runner = resolve_entry(runner if runner is not None else options.pop("runner", None))
    from ..uncertainty.driver import study
    study_dir = _study_dir(run_dir, "uncertainty", out_dir)
    result = study(project, study_dir, options=options, runner=runner, progress_cb=progress_cb, base_run_dir=run_dir)
    attached = attach_study(run_dir, result, progress_cb=progress_cb, refresh=refresh)
    return {"study_dir": str(study_dir), "study": str(study_dir / "study.json"),
            "summary": attached["summary"], "report": attached["report"]}


def compare_study(run_a, run_b, out_dir=None, options: dict | None = None, progress_cb=None) -> dict:
    """The comparison of two finished runs (cnctwin.report.compare: compare.json and compare.html in ``out_dir``,
    default ``<run_b>.compare-<run_a name>`` beside run B); returns the comparison document."""
    from ..report.compare import compare_runs
    run_a, run_b = Path(run_a), Path(run_b)
    for run in (run_a, run_b):
        run_document(run)
    out = Path(out_dir) if out_dir is not None else run_b.with_name(f"{run_b.name}.compare-{run_a.name}")
    out.mkdir(parents=True, exist_ok=True)
    request = {"run_a": run_a.name, "run_b": run_b.name, "run_a_dir": str(run_a), "run_b_dir": str(run_b),
               "options": dict(options or {})}
    return compare_runs(request, out, progress_cb)


def calibration_study(run_dir, machine_data: dict | None = None, options: dict | None = None, progress_cb=None, *,
                      out_dir=None, runner=None) -> dict:
    """A calibration of the run's project from recorded machine data (cnctwin.mdata.calibrate). Its loop
    simulator builds ``simulate(params)`` on ``runner`` (default cnctwin.core.loop.run_project when it is in the
    build): every evaluation writes the project with the parameters as an overlay and runs it. Outputs in
    ``out_dir`` (default ``<run_dir>.calibration``); returns the job summary."""
    from ..mdata.calibrate import calibrate
    run_dir = Path(run_dir)
    document = run_document(run_dir)
    project = project_path(document, run_dir, required=True)
    options = dict(options or {})
    spec = runner if runner is not None else options.pop("runner", None)
    simulate = resolve_entry(spec) if spec is not None else optional_entry()
    request = {"project_id": None, "project_path": str(project), "project_dir": str(project.parent),
               "run_id": document.get("run_id") or run_dir.name, "run_dir": str(run_dir),
               "machine_data": machine_data, "options": options}
    out = _study_dir(run_dir, "calibration", out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return calibrate(request, out, progress_cb, runner=simulate)


def nearest_checkpoint(run_dir, tick: int) -> Path | None:
    """The newest committed checkpoint marker (``checkpoints/ck_<tick>.json`` with its ``.npz``) at or before
    ``tick``."""
    folder = Path(run_dir) / "checkpoints"
    best = None
    for marker in folder.glob("ck_*.json") if folder.is_dir() else []:
        stem = marker.stem[3:]
        if stem.isdigit() and marker.with_suffix(".npz").is_file() and int(stem) <= int(tick):
            if best is None or int(stem) > best[0]:
                best = (int(stem), marker)
    return best[1] if best else None


def _index_lines(run_dir: Path) -> list[dict]:
    path = run_dir / "series" / "index.jsonl"
    out = []
    try:
        with open(path, "rb") as handle:
            for line in handle:
                if line.endswith(b"\n"):
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


def replay_window(run_dir, tick0: int, tick1: int, options: dict | None = None, progress_cb=None, *,
                  runner=None) -> dict:
    """``twin replay --window``: the fast (L0) tier of ``[tick0, tick1)`` of a finished run, written by
    ``run_project`` replaying from the nearest committed checkpoint at or before tick0 (from the start without
    one). It is called with the run's own options (run.json ``options``) and ``options["replay"] = {tick0,
    tick1, checkpoint}`` (the marker's absolute path or null); ``options`` given here go on top. The loop writes
    ``series/fast/replay-<tick0>-<tick1>.parquet`` and its index line (``replay: true``) and nothing else
    (docs/formats.md "Fast windows"). Returns the window, the checkpoint, the new index lines and the loop's
    return value; run.json's output index is updated."""
    run_dir = Path(run_dir)
    document = run_document(run_dir)
    tick0, tick1 = int(tick0), int(tick1)
    if tick1 <= tick0 or tick0 < 0:
        raise InputError(f"replay window [{tick0}, {tick1}): the end must come after a start at or after 0")
    project = project_path(document, run_dir, required=True)
    marker = nearest_checkpoint(run_dir, tick0)
    run_options = {k: v for k, v in (document.get("options") or {}).items() if k not in ("resume_from", "replay")}
    run_options.update(options or {})
    run_options["replay"] = {"tick0": tick0, "tick1": tick1,
                             "checkpoint": str(marker.resolve()) if marker is not None else None}
    simulate = resolve_entry(runner)
    before = len(_index_lines(run_dir))
    result = simulate(project, run_dir, run_options, progress_cb)
    added = [entry for entry in _index_lines(run_dir)[before:] if entry.get("replay")]
    if not added:
        raise TwinError(f"the replay of [{tick0}, {tick1}) wrote no replay part to series/index.jsonl")
    update_index(run_dir)
    tick_s = float((document.get("clock") or {}).get("tick_s") or 1e-6)
    return {"window": [tick0, tick1], "window_s": [tick0 * tick_s, tick1 * tick_s],
            "checkpoint": marker.relative_to(run_dir).as_posix() if marker is not None else None,
            "parts": added, "result": result}


# =============================================================================== command line

def _print_summary(summary: dict, stream) -> None:
    for name in GENERATORS:
        record = summary["generators"].get(name)
        if record is None:
            stream.write(f"{name:7s} kept\n")
            continue
        line = f"{name:7s} {record['status']}"
        if record.get("wall_s"):
            line += f" in {record['wall_s']:.1f} s"
        if record.get("peak_rss_bytes"):
            line += f", peak {record['peak_rss_bytes'] / 2 ** 20:.0f} MiB"
        if record.get("files"):
            line += f": {len(record['files'])} files"
        if record.get("reason"):
            line += f" ({record['reason']})"
        stream.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.core.outputs",
                                     description="Make the report, setup sheet, pictures, video and sound of a "
                                                 "finished run directory (docs/outputs.md).")
    parser.add_argument("run_dir", nargs="?", help="the run directory (runs/<run_id>)")
    parser.add_argument("--only", help="comma-separated generators to make (images, video, audio, report); "
                                       "the others are left as they are")
    for name in GENERATORS:
        parser.add_argument(f"--no-{name}", action="store_true", help=f"do not make the {name}")
    parser.add_argument("--report", dest="report_formats", help="report formats: html, pdf or html,pdf")
    parser.add_argument("--no-setup-sheet", action="store_true")
    parser.add_argument("--speed", type=float, help="video speed-up: simulated seconds per video second")
    parser.add_argument("--fps", type=float, help="video frame rate")
    parser.add_argument("--image-size", help="WIDTH,HEIGHT[,SUPERSAMPLING] of the pictures")
    parser.add_argument("--video-size", help="WIDTH,HEIGHT[,SUPERSAMPLING] of the video (even sizes)")
    parser.add_argument("--memory-mode", choices=MEMORY_MODES)
    parser.add_argument("--timeout", type=float, help="time limit of every generator (s)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--generate", help=argparse.SUPPRESS)
    parser.add_argument("--settings", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", dest="child_run_dir", help=argparse.SUPPRESS)
    parser.add_argument("--memory-bytes", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def cli_overrides(args) -> dict:
    outputs: dict = {}
    if args.only:
        outputs["only"] = args.only
    for name in ("images", "audio"):
        if getattr(args, f"no_{name}"):
            outputs[name] = False
    if args.no_report:
        outputs["report"] = []
    elif args.report_formats:
        outputs["report"] = args.report_formats
    if args.no_setup_sheet:
        outputs["setup_sheet"] = False
    if args.no_video:
        outputs["video"] = False
    elif args.speed is not None or args.fps is not None or args.video_size:
        video = {k: v for k, v in (("speed", args.speed), ("fps", args.fps)) if v is not None}
        if args.video_size:
            video["size"] = args.video_size
        outputs["video"] = video
    if args.image_size:
        outputs["image_size"] = args.image_size
    if args.memory_mode:
        outputs["memory_mode"] = args.memory_mode
    if args.timeout is not None:
        outputs["timeout_s"] = args.timeout
    return outputs


def main(argv=None) -> int:
    """``python -m cnctwin.core.outputs RUN_DIR ...``: exit 0 when every generator asked for succeeded, 1 when one
    failed or on an input error. (``--generate`` is the generator subprocess.)"""
    args = build_parser().parse_args(argv)
    if args.generate:
        return generator_main(args.generate, args.child_run_dir, args.settings, args.result, args.memory_mode or
                              "none", args.memory_bytes)
    if not args.run_dir:
        build_parser().print_usage(sys.stderr)
        return 2

    def progress(progress=None, **fields):
        if fields.get("message") and not args.json:
            sys.stderr.write(f"{fields.get('phase', '')}: {fields['message']}\n")

    try:
        summary = finalize_outputs(args.run_dir, {"outputs": cli_overrides(args)}, progress)
    except TwinError as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=1, default=str) + "\n")
    else:
        _print_summary(summary, sys.stdout)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["finalize_outputs", "output_settings", "run_generator", "update_index", "resolve_entry", "optional_entry",
           "attach_study", "study_summary", "uncertainty_study", "compare_study", "calibration_study", "replay_window",
           "nearest_checkpoint", "positioned_findings", "generator_environment", "simulated_time_s", "run_document",
           "project_path", "main", "GENERATORS", "OWNED", "EntryMissing", "OutputsBusy"]
