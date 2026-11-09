"""Job manager: simulations, checks, comparisons, uncertainty studies and calibrations, each in its
own subprocess (``python -m cnctwin.api.worker <job_dir>``) with a memory cap and the deterministic
environment of architecture 9.

* **Queue.** Two pools: ``heavy`` (simulate, uncertainty, calibrate; default one at a time on this
  box) and ``light`` (check, compare; default one). Jobs start in submission order within a pool.
* **Memory cap**, like bin/tsafe. ``systemd``: ``sudo -n systemd-run --scope -p MemoryMax=...
  -p MemorySwapMax=...`` (a cgroup cap on the job and everything it starts; used when systemd runs
  and passwordless sudo, or root, is available). ``rlimit``: the worker caps its own data segment
  (RLIMIT_DATA) before importing anything, so an allocation beyond the cap raises MemoryError and
  the job fails with that reason (used in containers without systemd). ``none``: no cap.
* **Environment** (built from scratch, nothing inherited but HOME): OMP/OPENBLAS/MKL/NUMBA/NUMEXPR
  threads 1, PYTHONHASHSEED=0, NUMBA_CACHE_DIR keyed exactly as bin/tsafe keys it (sha256 of the
  kernel source files, the numba/llvmlite versions and the CPU model), SOURCE_DATE_EPOCH, TZ=UTC,
  LC_ALL=C.UTF-8, MPLBACKEND=Agg, nice 10.
* **Cancel.** A queued job is dropped. A running job gets a ``cancel`` file in its folder; the
  worker's progress callback then raises (the loop stops at its next callback); after
  ``cancel_grace_s`` the process group gets SIGTERM, after ``kill_grace_s`` more SIGKILL.
* **Persistence.** ``DATA_DIR/jobs/<id>/job.json`` (written by the manager) and ``worker.json``
  (written by the worker) survive a server restart: queued jobs are queued again, running jobs whose
  process is still alive are adopted, the others are marked failed (interrupted; a simulation can be
  resumed from its last checkpoint). One server per DATA_DIR (an flock on ``DATA_DIR/.api.lock``).
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cnctwin

from .jsonio import read_json, write_json_atomic
from .safety import ApiError
from .settings import JOB_POOLS, Settings

TERMINAL = ("done", "failed", "cancelled")
ACTIVE = ("queued", "running", "cancelling")
RUN_STATUS_FORMAT = "cnctwin-run-status"
JOB_FORMAT = "cnctwin-job"
RESULT_FORMAT = "cnctwin-job-result"         # result.json: {format, version, job_id, kind, result} (worker.py)
WORKER_MODULE = "cnctwin.api.worker"
_KERNEL_PATTERN = re.compile(rb"@njit|@numba|cuda.jit")
PACKAGE_DIR = Path(cnctwin.__file__).resolve().parent


# ----------------------------------------------------------------------------- environment of a job

def numba_cache_key(package_dir: Path = PACKAGE_DIR) -> str:
    """The key bin/tsafe computes: sha256 over ``sha256sum`` lines of every cnctwin source file that
    defines a kernel (C-sorted relative paths), ``numba.__version__ llvmlite.__version__`` and the
    first ``model name`` line of /proc/cpuinfo; the first 16 hex digits."""
    package_dir = Path(package_dir)
    base = package_dir.parent
    found = []
    for directory, dirnames, filenames in os.walk(package_dir):
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
    lines = []
    for relative in sorted(found, key=lambda s: s.encode("utf-8")):
        digest = hashlib.sha256((base / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    try:
        lines.append(f"{importlib.metadata.version('numba')} {importlib.metadata.version('llvmlite')}\n")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "model name" in line:
                    lines.append(line if line.endswith("\n") else line + "\n")
                    break
    except OSError:
        pass
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()[:16]


def numba_cache_dir(settings: Settings) -> str:
    """NUMBA_CACHE_DIR for a job: the server's own when it was given one (bin/tsafe sets it),
    else ``<root>/.numba-cache/<key>`` (``DATA_DIR/numba-cache/<key>`` if the root is read-only)."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return os.environ["NUMBA_CACHE_DIR"]
    key = numba_cache_key()
    base = Path(os.environ.get("CNCTWIN_NUMBA_CACHE_BASE") or (settings.root / ".numba-cache"))
    try:
        base.mkdir(parents=True, exist_ok=True)
        if os.access(base, os.W_OK):
            return str(base / key)
    except OSError:
        pass
    return str(settings.data_dir / "numba-cache" / key)


PASSTHROUGH_ENV = ("NUMBA_ENABLE_CUDASIM", "NUMBA_BOUNDSCHECK", "CUDA_VISIBLE_DEVICES", "NUMBA_CUDA_DRIVER")


def job_environment(settings: Settings, job_dir: Path) -> dict:
    env = {
        "HOME": os.environ.get("HOME") or "/tmp",
        "PATH": f"{Path(settings.python).parent}:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
        "PYTHONPATH": str(PACKAGE_DIR.parent),
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "NUMBA_CACHE_DIR": numba_cache_dir(settings),
        "NUMBA_ENABLE_CUDASIM": "0", "NUMBA_BOUNDSCHECK": "0",
        "SOURCE_DATE_EPOCH": settings.source_date_epoch,
        "MPLBACKEND": "Agg",
        "DATA_DIR": str(settings.data_dir),
        "CNCTWIN_ROOT": str(settings.root),
        "CNCTWIN_JOB_DIR": str(job_dir),
    }
    for name in PASSTHROUGH_ENV:
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


_systemd_cache: dict[str, bool] = {}


def systemd_available() -> bool:
    """systemd runs here and we may call systemd-run (as root, or through passwordless sudo)."""
    if "ok" in _systemd_cache:
        return _systemd_cache["ok"]
    ok = os.path.isdir("/run/systemd/system") and shutil.which("systemd-run") is not None
    if ok and os.geteuid() != 0:
        ok = shutil.which("sudo") is not None
        if ok:
            try:
                ok = subprocess.run(["sudo", "-n", "systemd-run", "--version"], capture_output=True, timeout=10,
                                    check=False).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
    _systemd_cache["ok"] = ok
    return ok


def resolve_memory_mode(requested: str) -> str:
    if requested == "auto":
        return "systemd" if systemd_available() else "rlimit"
    if requested == "systemd" and not systemd_available():
        raise RuntimeError("memory mode 'systemd' needs systemd and root or passwordless sudo for systemd-run; "
                           "use CNCTWIN_JOB_MEMORY_MODE=rlimit")
    return requested


def scope_unit(settings: Settings, job_id: str) -> str:
    """systemd scope name of a job (unique per DATA_DIR and job)."""
    tag = hashlib.sha256(str(settings.data_dir).encode("utf-8")).hexdigest()[:8]
    return f"cnctwin-job-{tag}-{job_id}"


def worker_command(settings: Settings, job_dir: Path, mode: str, memory_bytes: int,
                   env: dict, unit: str | None = None) -> tuple[list[str], dict | None]:
    """The command line and the environment for Popen (None: the command carries the environment)."""
    python = [settings.python, "-m", WORKER_MODULE, str(job_dir)]
    if mode != "systemd":
        return python, env
    command = [] if os.geteuid() == 0 else ["sudo", "-n"]
    command += ["systemd-run", "--scope", "-q", "-p", f"MemoryMax={memory_bytes}",
                "-p", f"MemorySwapMax={settings.swap_max_bytes}"]
    if unit:
        command.append(f"--unit={unit}")
    if os.geteuid() != 0:
        command += [f"--uid={os.getuid()}", f"--gid={os.getgid()}"]
    command += ["/usr/bin/env", "-i"] + [f"{k}={v}" for k, v in sorted(env.items())] + python
    return command, None


def write_run_status(run_dir: Path, status: str, job, *, error: dict | None = None) -> None:
    """status.json of a simulation whose worker is not (or no longer) writing it: queued, cancelled
    before it started, or ended without reporting. The last progress numbers are kept."""
    path = Path(run_dir) / "status.json"
    previous: dict = {}
    try:
        previous = read_json(path)
    except (OSError, ValueError):
        previous = {}
    last = previous.get("progress") if isinstance(previous.get("progress"), dict) else {}
    progress = dict(last)
    progress.setdefault("sim_time_s", 0.0)
    progress.setdefault("est_total_s", None)
    progress.setdefault("wall_s", job.prior_wall_s if job is not None else 0.0)
    progress["eta_s"] = None if status in TERMINAL or status == "queued" else progress.get("eta_s")
    document = {"format": RUN_STATUS_FORMAT, "version": 1, "status": status, "progress": progress,
                "job_id": job.id if job is not None else None, "attempt": job.attempt if job is not None else None,
                "error": error, "checkpoint": previous.get("checkpoint"),
                "resume_from": job.request.get("resume_from") if job is not None else None}
    write_json_atomic(path, document, fsync=False)


# ----------------------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    seq: int
    kind: str
    pool: str
    entry: str
    request: dict
    state: str = "queued"
    run_id: str | None = None
    project_id: str | None = None
    key: str | None = None
    created_s: float = 0.0
    started_s: float | None = None
    finished_s: float | None = None
    pid: int | None = None
    pgid: int | None = None
    worker_pid: int | None = None
    returncode: int | None = None
    error: dict | None = None
    progress: dict = field(default_factory=dict)
    cancel_requested_s: float | None = None
    term_sent_s: float | None = None
    kill_sent_s: float | None = None
    memory_mode: str = ""
    memory_bytes: int = 0
    attempt: int = 1
    prior_wall_s: float = 0.0
    has_result: bool = False

    def record(self) -> dict:
        data = asdict(self)
        data["format"] = JOB_FORMAT
        data["version"] = 1
        return data

    @classmethod
    def from_record(cls, data: dict) -> "Job":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.jobs_dir = settings.jobs_dir
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._logs: dict[str, object] = {}
        self._running: set[str] = set()
        self._worker_mtime: dict[str, int] = {}
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._lockfile = None
        self.memory_mode = ""
        self.started = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.memory_mode = resolve_memory_mode(self.s.memory_mode)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.s.runs_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.s.data_dir / ".api.lock"
        self._lockfile = open(lock_path, "a+")
        try:
            fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lockfile.close()
            self._lockfile = None
            raise RuntimeError(f"another CNC Twin server uses {self.s.data_dir} (lock {lock_path})") from None
        try:
            with self._wake:
                self._recover()
                self._stopping = False
            self._thread = threading.Thread(target=self._loop, name="cnctwin-jobs", daemon=True)
            self._thread.start()
        except BaseException:
            fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_UN)
            self._lockfile.close()
            self._lockfile = None
            raise
        self.started = True

    def stop(self) -> None:
        with self._wake:
            self._stopping = True
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30)
        if self.s.cancel_on_shutdown:
            self._terminate_running("the server stopped")
        with self._wake:
            for handle in self._logs.values():
                handle.close()
            self._logs.clear()
        if self._lockfile is not None:
            fcntl.flock(self._lockfile.fileno(), fcntl.LOCK_UN)
            self._lockfile.close()
            self._lockfile = None
        self.started = False

    def _terminate_running(self, reason: str) -> None:
        with self._wake:
            running = [self._jobs[j] for j in sorted(self._running)]
            for job in running:
                if job.state in ACTIVE:
                    if job.cancel_requested_s is None:
                        job.cancel_requested_s = time.time()
                    job.state = "cancelling"
                self._signal(job, signal.SIGTERM)
        deadline = time.monotonic() + self.s.kill_grace_s
        while time.monotonic() < deadline:
            with self._wake:
                self._reap()
                if not self._running:
                    return
            time.sleep(0.05)
        with self._wake:
            for job_id in sorted(self._running):
                self._signal(self._jobs[job_id], signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._wake:
                self._reap()
                if not self._running:
                    return
            time.sleep(0.05)
        with self._wake:
            for job_id in sorted(self._running):
                job = self._jobs[job_id]
                self._finalize(job, interrupted=reason)
            self._running.clear()

    # ------------------------------------------------------------------ public API
    def submit(self, kind: str, request: dict, *, run_id: str | None = None, project_id: str | None = None,
               key: str | None = None, attempt: int = 1, prior_wall_s: float = 0.0) -> Job:
        if kind not in JOB_POOLS:
            raise ApiError(400, f"unknown job kind {kind!r}")
        with self._wake:
            if not self.started:
                raise ApiError(503, "the job manager is not running")
            self._seq += 1
            pool = JOB_POOLS[kind]
            job = Job(id=f"j{self._seq:06d}", seq=self._seq, kind=kind, pool=pool, entry=self.s.entries[kind],
                      request=dict(request), run_id=run_id, project_id=project_id, key=key, created_s=time.time(),
                      memory_mode=self.memory_mode, memory_bytes=self.s.memory_bytes(pool), attempt=attempt,
                      prior_wall_s=prior_wall_s)
            job_dir = self.job_dir(job.id)
            job_dir.mkdir(parents=True, exist_ok=False)
            (job_dir / "out").mkdir()
            job.request.setdefault("out_dir", str(job_dir / "out"))
            job.request.setdefault("status_interval_s", self.s.status_interval_s)
            self._entries_for(kind, job.request)
            self._jobs[job.id] = job
            self._persist(job)
            if kind == "simulate":
                write_run_status(Path(job.request["run_dir"]), "queued", job)
            self._wake.notify_all()
            return job

    def _entries_for(self, kind: str, request: dict) -> None:
        """The other entry points a job calls: a simulation its outputs (report, pictures, video, sound) after a
        successful run; uncertainty studies and calibrations the server's simulation (each variant, each loop
        evaluation) and a study the outputs that show it on the run's report; a sweep the server's simulation (its
        points' runner when the spec names none and the server overrides the default loop)."""
        if kind == "simulate":
            request.setdefault("outputs_entry", self.s.entries.get("outputs"))
        elif kind == "sweep":
            request.setdefault("simulate_entry", self.s.entries.get("simulate"))
        elif kind in ("uncertainty", "calibrate"):
            request.setdefault("simulate_entry", self.s.entries.get("simulate"))
            if kind == "uncertainty":
                request.setdefault("outputs_entry", self.s.entries.get("outputs"))

    def get(self, job_id: str) -> Job | None:
        with self._wake:
            job = self._jobs.get(job_id)
            if job is not None and job.id in self._running:
                self._reap_one(job)
            return job

    def list(self, kind: str | None = None, state: str | None = None) -> list[Job]:
        with self._wake:
            jobs = [j for j in self._jobs.values() if (kind is None or j.kind == kind)
                    and (state is None or j.state == state)]
            return sorted(jobs, key=lambda j: j.seq)

    def jobs_for_run(self, run_id: str, kind: str | None = "simulate") -> list[Job]:
        """Jobs of a run (by default its simulation attempts), oldest first."""
        with self._wake:
            return sorted((j for j in self._jobs.values() if j.run_id == run_id and (kind is None or j.kind == kind)),
                          key=lambda j: j.seq)

    def latest_for_run(self, run_id: str, kind: str | None = "simulate") -> Job | None:
        """The newest job of a run, brought up to date with its worker's report."""
        with self._wake:
            jobs = self.jobs_for_run(run_id, kind)
            if not jobs:
                return None
            job = jobs[-1]
            if job.id in self._running:
                self._reap_one(job)
            return job

    def find_by_key(self, kind: str, key: str) -> Job | None:
        with self._wake:
            matches = [j for j in self._jobs.values() if j.kind == kind and j.key == key]
            return max(matches, key=lambda j: j.seq) if matches else None

    def queue_position(self, job_id: str) -> int | None:
        with self._wake:
            job = self._jobs.get(job_id)
            if job is None or job.state != "queued":
                return None
            ahead = [j for j in self._jobs.values() if j.pool == job.pool and j.state == "queued" and j.seq < job.seq]
            return len(ahead)

    def cancel(self, job_id: str, reason: str = "cancelled on request") -> Job:
        with self._wake:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, f"no job {job_id!r}")
            if job.state in TERMINAL:
                raise ApiError(409, f"job {job_id} has already ended ({job.state})")
            if job.state == "queued":
                job.state = "cancelled"
                job.finished_s = time.time()
                job.error = {"type": "Cancelled", "message": f"{reason} before it started"}
                if job.kind == "simulate":
                    write_run_status(Path(job.request["run_dir"]), "cancelled", job, error=job.error)
            elif job.state == "running":
                job.state = "cancelling"
                job.cancel_requested_s = time.time()
                (self.job_dir(job.id) / "cancel").write_text(reason + "\n", encoding="utf-8")
            self._persist(job)
            self._wake.notify_all()
            return job

    def wait(self, job_id: str, timeout_s: float) -> Job:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._wake:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    raise ApiError(404, f"no job {job_id!r}")
                if job.state in TERMINAL:
                    return job
                if job.id in self._running:
                    self._reap_one(job)
                if job.state in TERMINAL:
                    return job
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return job
                self._wake.wait(min(remaining, self.s.poll_s))

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def result(self, job: Job):
        """The entry's return value: result.json is ``{format: cnctwin-job-result, version, job_id, kind, result}``
        (cnctwin.api.worker); a result.json written before that header is the bare value and is returned as it is."""
        path = self.job_dir(job.id) / "result.json"
        if not path.exists():
            return None
        document = read_json(path)
        if isinstance(document, dict) and document.get("format") == RESULT_FORMAT:
            return document.get("result")
        return document

    def log_tail(self, job: Job, max_bytes: int = 4096) -> str:
        path = self.job_dir(job.id) / "log.txt"
        if not path.exists():
            return ""
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", "replace")

    def counts(self) -> dict:
        with self._wake:
            out = {"queued": 0, "running": 0, "cancelling": 0}
            for job in self._jobs.values():
                if job.state in out:
                    out[job.state] += 1
            return out

    # ------------------------------------------------------------------ the loop
    def _loop(self) -> None:
        while True:
            with self._wake:
                if self._stopping:
                    return
                try:
                    self._reap()
                    self._enforce_cancels()
                    self._launch_ready()
                except Exception as error:   # keep the manager alive; the failure is logged per job
                    self._log_manager_error(error)
                self._wake.wait(self.s.poll_s)

    def _log_manager_error(self, error: Exception) -> None:
        with open(self.jobs_dir / "manager-errors.log", "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {type(error).__name__}: {error}\n")

    def _launch_ready(self) -> None:
        for pool in ("heavy", "light"):
            while sum(1 for j in self._running if self._jobs[j].pool == pool) < self.s.max_jobs(pool):
                queued = [j for j in self._jobs.values() if j.pool == pool and j.state == "queued"]
                if not queued:
                    break
                self._launch(min(queued, key=lambda j: j.seq))

    def _launch(self, job: Job) -> None:
        job_dir = self.job_dir(job.id)
        for stale in ("cancel", "worker.json", "result.json"):
            if (job_dir / stale).exists():
                (job_dir / stale).unlink()
        job.started_s = time.time()
        job.state = "running"
        self._persist(job)
        try:
            env = job_environment(self.s, job_dir)
            command, popen_env = worker_command(self.s, job_dir, job.memory_mode, job.memory_bytes, env,
                                                unit=scope_unit(self.s, job.id))
            log = open(job_dir / "log.txt", "ab")
            process = subprocess.Popen(command, env=popen_env, cwd=str(job_dir), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            job.state = "failed"
            job.finished_s = time.time()
            job.error = {"type": "LaunchError", "message": f"could not start the job process: {error}"}
            self._persist(job)
            if job.kind == "simulate":
                write_run_status(Path(job.request["run_dir"]), "failed", job, error=job.error)
            self._wake.notify_all()
            return
        job.pid = process.pid
        job.pgid = process.pid
        self._procs[job.id] = process
        self._logs[job.id] = log
        self._running.add(job.id)
        self._persist(job)
        self._wake.notify_all()

    def _reap(self) -> None:
        for job_id in sorted(self._running):
            self._reap_one(self._jobs[job_id])

    def _reap_one(self, job: Job) -> None:
        """Take the worker's final report as soon as it is written (the job ends then; its process
        keeps its pool slot until it exits), and clean up once the process has ended. A process that
        lingers ``kill_grace_s`` after its final report is killed."""
        process = self._procs.get(job.id)
        self._refresh_from_worker(job)
        exited = False
        if process is not None:
            code = process.poll()
            if code is not None:
                job.returncode = code
                exited = True
        elif not self._alive(job):
            exited = True
        if exited:
            self._finalize(job)
        elif job.state in TERMINAL and job.finished_s and job.kill_sent_s is None \
                and time.time() - job.finished_s > self.s.kill_grace_s:
            job.kill_sent_s = time.time()
            self._signal(job, signal.SIGKILL)
            self._persist(job)

    def _refresh_from_worker(self, job: Job) -> dict | None:
        path = self.job_dir(job.id) / "worker.json"
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        try:
            worker = read_json(path)
        except (OSError, ValueError):
            return None
        if self._worker_mtime.get(job.id) != mtime:
            self._worker_mtime[job.id] = mtime
            job.worker_pid = worker.get("pid", job.worker_pid)
            if isinstance(worker.get("progress"), dict):
                job.progress = worker["progress"]
            if worker.get("final") and worker.get("state") in TERMINAL and job.state in ACTIVE:
                job.state = worker["state"]
                job.error = worker.get("error")
                job.has_result = bool(worker.get("result"))
                job.finished_s = time.time()
                self._persist(job)
                self._wake.notify_all()
        return worker

    def _finalize(self, job: Job, interrupted: str | None = None) -> None:
        """The process has ended (or is gone): its final report, or why it ended without one."""
        self._refresh_from_worker(job)
        if job.state in TERMINAL and job.kind == "simulate":
            self._reconcile_run_status(job)
        if job.state not in TERMINAL:
            cancelled = job.cancel_requested_s is not None
            job.state = "cancelled" if cancelled else "failed"
            job.error = {"type": "Cancelled" if cancelled else "WorkerDied",
                         "message": interrupted or self._exit_reason(job), "log_tail": self.log_tail(job)}
            job.finished_s = time.time()
            if job.kind == "simulate":
                write_run_status(Path(job.request["run_dir"]), job.state, job, error=job.error)
        self._running.discard(job.id)
        self._procs.pop(job.id, None)
        handle = self._logs.pop(job.id, None)
        if handle is not None:
            handle.close()
        self._persist(job)
        self._wake.notify_all()

    def _reconcile_run_status(self, job: Job) -> None:
        """A worker that reported its end but died before writing the run's last status: write it."""
        path = Path(job.request["run_dir"]) / "status.json"
        try:
            status = read_json(path).get("status")
        except (OSError, ValueError, AttributeError):
            status = None
        if status != job.state:
            write_run_status(Path(job.request["run_dir"]), job.state, job, error=job.error)

    def _exit_reason(self, job: Job) -> str:
        code = job.returncode
        if code is None:
            return "the job process ended without reporting (the server may have restarted)"
        number = -code if code < 0 else (code - 128 if code > 128 else 0)
        if number:
            try:
                name = signal.Signals(number).name
            except ValueError:
                name = f"signal {number}"
            hint = ""
            if number == signal.SIGKILL and job.kill_sent_s is None and job.memory_mode != "none":
                hint = f" (it may have reached its memory cap of {job.memory_bytes / 2**20:.0f} MiB)"
            return f"the job process was killed by {name}{hint}"
        return f"the job process exited with status {code} without reporting"

    def _enforce_cancels(self) -> None:
        now = time.time()
        for job_id in sorted(self._running):
            job = self._jobs[job_id]
            if job.state != "cancelling" or job.cancel_requested_s is None:
                continue
            if job.term_sent_s is None and now - job.cancel_requested_s >= self.s.cancel_grace_s:
                job.term_sent_s = now
                self._signal(job, signal.SIGTERM)
                self._persist(job)
            elif job.term_sent_s is not None and job.kill_sent_s is None and now - job.term_sent_s >= self.s.kill_grace_s:
                job.kill_sent_s = now
                self._signal(job, signal.SIGKILL)
                self._persist(job)

    def _signal(self, job: Job, sig: int) -> None:
        """Signal the launcher's process group and the worker's own group (the worker calls setsid),
        and in systemd mode, for SIGKILL, the whole scope."""
        self._refresh_from_worker(job)
        for group in sorted({g for g in (job.pgid, job.worker_pid) if g}):
            try:
                os.killpg(group, sig)
            except (ProcessLookupError, PermissionError):
                continue
        if sig == signal.SIGKILL and job.memory_mode == "systemd":
            command = ([] if os.geteuid() == 0 else ["sudo", "-n"]) + [
                "systemctl", "kill", "--signal=SIGKILL", f"{scope_unit(self.s, job.id)}.scope"]
            try:
                subprocess.run(command, capture_output=True, timeout=10, check=False)
            except (OSError, subprocess.SubprocessError) as error:
                self._log_manager_error(RuntimeError(f"{job.id}: systemctl kill failed ({error})"))

    def _alive(self, job: Job) -> bool:
        for pid in (job.worker_pid, job.pid):
            if not pid:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as handle:
                    cmdline = handle.read()
            except OSError:
                continue
            if WORKER_MODULE.encode() in cmdline and str(self.job_dir(job.id)).encode() in cmdline:
                return True
        return False

    # ------------------------------------------------------------------ persistence
    def _persist(self, job: Job) -> None:
        write_json_atomic(self.job_dir(job.id) / "job.json", job.record())

    def _recover(self) -> None:
        if not self.jobs_dir.exists():
            return
        for job_dir in sorted(p for p in self.jobs_dir.iterdir() if p.is_dir()):
            record_path = job_dir / "job.json"
            if not record_path.exists():
                continue
            try:
                job = Job.from_record(read_json(record_path))
            except (OSError, ValueError, TypeError) as error:
                self._log_manager_error(RuntimeError(f"{record_path}: unreadable job record ({error})"))
                continue
            self._jobs[job.id] = job
            self._seq = max(self._seq, job.seq)
            if job.state in ("running", "cancelling"):
                if self._alive(job):
                    self._running.add(job.id)
                else:
                    self._finalize(job, interrupted="the server stopped while the job was running")
