"""Running a sweep (docs/ml.md "Sweeps"): plan the points, write their variants, run each through the run cache.

The sweep folder::

    sweep.json                    the resolved spec, its sha256, the design's notes (format cnctwin-sweep-run)
    plan.json                     every point: index, values, unit draws, variant project, run options, sha256
    points/p<nnnnn>/project.yaml  the variant (+ the program a template or generator wrote)
    points/p<nnnnn>/result.json   what the point gave: status done | cached | failed, key, run_dir, verdict, error
    status.json                   volatile: progress, ETA, running points
    summary.json                  the end: counts, verdicts, every point's status, key and run (format
                                  cnctwin-sweep-summary; no wall times, so the same sweep gives the same file)

Each point: its cache key is computed here, a verified cached run is taken as it is (``cached``), a point that
failed before with the same key is not run again unless ``retry_failed`` (a failure under another build or
other inputs is), anything else runs: ``isolation: process`` (default) in a fresh ``python -m
cnctwin.sweep.point`` per point, with the deterministic environment and a memory cap like bin/tsafe
(``memory_mode``: ``systemd`` = ``sudo -n systemd-run --scope -p MemoryMax=<memory> -p MemorySwapMax=256M``,
``rlimit`` = RLIMIT_DATA, ``auto`` = systemd when available; ``none``); up to ``workers`` at once. A process
always runs the source tree as it is on disk at its start, so its run's key is that build's. ``isolation:
inline`` runs points in this process, one after the other.

A point process watches its standard input and stops when this process is gone; a cancel (an exception from the
progress callback, SIGTERM, Ctrl-C) terminates the running points (SIGTERM, then SIGKILL and ``systemctl kill``
of their scopes after a grace period), records the points that finished and re-raises. Running the same spec
into the same folder again resumes: finished points are cached runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import InputError, TwinError
from .spec import SweepSpec

RUN_FORMAT = "cnctwin-sweep-run"
PLAN_FORMAT = "cnctwin-sweep-plan"
POINT_FORMAT = "cnctwin-sweep-point"
SUMMARY_FORMAT = "cnctwin-sweep-summary"
STATUS_FORMAT = "cnctwin-sweep-status"
VERSION = 1
SWAP_MAX = "256M"
POLL_S = 0.2
PROGRESS_EVERY_S = 1.0
TERM_GRACE_S = 10.0
LOG_TAIL = 4000
PASSTHROUGH = ("HOME", "DATA_DIR", "NUMBA_ENABLE_CUDASIM", "NUMBA_BOUNDSCHECK", "CUDA_VISIBLE_DEVICES",
               "SOURCE_DATE_EPOCH")                       # and every CNCTWIN_* variable


def _write_json(path: Path, document, *, fsync: bool = False) -> None:
    from ..core.rundir import write_json_atomic
    write_json_atomic(path, document, fsync=fsync)


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# =============================================================================== the plan

def plan_sweep(spec: SweepSpec, out_dir, *, db=None, progress_cb=None) -> list:
    """Draw the design, write every point's variant, sweep.json and plan.json; returns the PointPlans."""
    from .design import design_points
    from .variants import VariantWriter
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    values, units, notes = design_points(spec)
    if not values:
        raise InputError(f"sweep {spec.name!r}: the design has no points")
    writer = VariantWriter(spec, out, db=db)
    plans = []
    for i, point in enumerate(values):
        if progress_cb is not None:
            progress_cb({"done": i, "total": len(values)}, phase="plan", message=f"writing point {i + 1} of {len(values)}")
        plans.append(writer.write(i, point, [float(u) for u in units[i]] if units is not None else None))
    _write_json(out / "sweep.json", {"format": RUN_FORMAT, "version": VERSION, "name": spec.name,
                                     "spec": spec.describe(), "spec_sha256": spec.sha256,
                                     "spec_path": str(spec.path) if spec.path else None, "points": len(plans),
                                     "notes": notes})
    _write_json(out / "plan.json", {"format": PLAN_FORMAT, "version": VERSION, "name": spec.name,
                                    "parameters": [p.describe() for p in spec.parameters],
                                    "points": [p.to_dict(out) for p in plans]})
    return plans


# =============================================================================== memory caps and environments

def point_environment(python: str | None = None) -> dict:
    """The environment of a point process: the deterministic contract (architecture 9), the Numba cache this
    process uses (or bin/tsafe's key), this source tree first on PYTHONPATH, and the settings it passes on (DATA_DIR,
    SOURCE_DATE_EPOCH, every CNCTWIN_* variable)."""
    from ..core.rundir import numba_cache_key
    python = python or sys.executable
    root = Path(__file__).resolve().parents[2]
    env = {"PATH": f"{Path(python).parent}:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "TZ": "UTC", "PYTHONPATH": str(root), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMBA_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "MPLBACKEND": "Agg",
           "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR") or str(root / ".numba-cache" / numba_cache_key())}
    for name in sorted(os.environ):
        if (name in PASSTHROUGH or name.startswith("CNCTWIN_")) and os.environ[name]:
            env[name] = os.environ[name]
    env.setdefault("HOME", "/tmp")
    return env


def resolve_memory_mode(requested: str) -> str:
    from ..api.jobs import resolve_memory_mode as resolve
    return resolve(requested)


def point_command(job_path: Path, *, mode: str, memory_bytes: int, env: dict, unit: str,
                  python: str | None = None) -> tuple[list[str], dict | None]:
    """(command, environment for Popen; None when the command carries it) of one point process."""
    python = python or sys.executable
    base = [python, "-m", "cnctwin.sweep.point", str(job_path)]
    if mode != "systemd":
        return base, env
    command = [] if os.geteuid() == 0 else ["sudo", "-n"]
    command += ["systemd-run", "--scope", "-q", "-p", f"MemoryMax={int(memory_bytes)}", "-p", f"MemorySwapMax={SWAP_MAX}",
                f"--unit={unit}"]
    if os.geteuid() != 0:
        command += [f"--uid={os.getuid()}", f"--gid={os.getgid()}"]
    command += ["/usr/bin/env", "-i"] + [f"{k}={v}" for k, v in sorted(env.items())] + base
    return command, None


# =============================================================================== running

@dataclass
class Child:
    plan: object
    key: str
    process: subprocess.Popen
    started: float
    unit: str
    log: object
    progress: dict = field(default_factory=dict)


class SweepRun:
    """One execution of a sweep into ``out_dir`` (module docstring)."""

    def __init__(self, spec: SweepSpec, out_dir, *, cache=None, runs_dir=None, workers: int | None = None,
                 isolation: str | None = None, memory_bytes: int | None = None, memory_mode: str | None = None,
                 runner: str | None = None, retry_failed: bool | None = None, no_cache: bool = False,
                 verify: str | None = None, progress_cb=None, db=None) -> None:
        from ..cache import RunCache
        ex = spec.execution
        self.spec = spec
        self.out = Path(out_dir).resolve()
        self.cache = cache if cache is not None else RunCache()
        self.runs_dir = Path(runs_dir).resolve() if runs_dir is not None else None
        self.workers = int(workers if workers is not None else ex.workers)
        self.isolation = isolation or ex.isolation
        self.memory_bytes = int(memory_bytes if memory_bytes is not None else ex.memory_bytes)
        self.memory_mode = memory_mode or ex.memory_mode
        self.runner = runner or ex.runner
        self.retry_failed = bool(ex.retry_failed if retry_failed is None else retry_failed)
        self.no_cache = bool(no_cache)
        self.verify = verify or ex.verify
        self.progress_cb = progress_cb
        self.db = db
        if self.workers < 1:
            raise InputError("a sweep needs at least one worker")
        if self.workers > 1 and self.isolation != "process":
            raise InputError("several workers need isolation: process")
        self.results: dict[int, dict] = {}
        self.children: dict[int, Child] = {}
        self.walls: list[float] = []
        self.t0 = time.monotonic()
        self.last_progress = 0.0
        self.total = 0
        self.mode = None
        self._inline_progress = None

    # ------------------------------------------------------------------ the whole sweep
    def run(self, *, plan_only: bool = False) -> dict:
        from ..cache import database
        db = self.db if self.db is not None else database()
        plans = plan_sweep(self.spec, self.out, db=db, progress_cb=self.progress_cb)
        self.total = len(plans)
        if plan_only:
            return {"format": SUMMARY_FORMAT, "version": VERSION, "name": self.spec.name, "points": len(plans),
                    "planned": True, "out_dir": str(self.out)}
        if self.isolation == "process":
            self.mode = resolve_memory_mode(self.memory_mode)
        pending = list(plans)
        self._status("running")
        try:
            while pending or self.children:
                while pending and len(self.children) < self.workers:
                    plan = pending.pop(0)
                    self._start(plan, db)
                self._poll()
                self._report(force=False, pending=len(pending))
                if self.children:
                    time.sleep(POLL_S)
        except BaseException as error:
            self._terminate()
            self._status("cancelled" if not isinstance(error, Exception) else "failed",
                         error=f"{type(error).__name__}: {error}")
            raise
        summary = self._summary(plans)
        _write_json(self.out / "summary.json", summary, fsync=True)
        self._status("done")
        self._report(force=True, pending=0)
        return summary

    # ------------------------------------------------------------------ one point
    def _start(self, plan, db) -> None:
        from ..cache import key_for
        folder = plan.project.parent
        previous = _read_json(folder / "result.json")
        try:
            key = key_for(plan.project, plan.options, runner=self.runner, db=db).key
        except InputError as error:
            self._record(plan, {"status": "failed", "key": None, "error": {"type": "InputError", "message": str(error)},
                                "notes": ["the variant does not resolve"]})
            return
        if previous and previous.get("status") == "failed" and previous.get("key") == key and not self.retry_failed:
            record = {k: previous.get(k) for k in ("status", "key", "run_dir", "run_id", "verdict", "complete", "error",
                                                    "registered")}
            record["notes"] = list(previous.get("notes") or []) + ["failed before with the same key: not run again "
                                                                    "(retry_failed runs it)"]
            self._record(plan, record)
            return
        if not self.no_cache:
            hit = self.cache.lookup(key, verify=self.verify)
            if hit is not None:
                document = _read_json(Path(hit.run_dir) / "run.json") or {}
                self._record(plan, {"status": "cached", "key": key, "run_dir": hit.run_dir, "run_id": hit.run_id,
                                    "verdict": document.get("verdict"), "complete": document.get("complete"),
                                    "registered": True, "error": None, "notes": []})
                return
        if self.isolation == "inline":
            self._inline(plan, key)
        else:
            self._spawn(plan, key)

    def _inline(self, plan, key: str) -> None:
        from ..cache import run_cached
        started = time.monotonic()
        index = plan.index

        def inner(progress=None, **fields):
            merged = dict(progress or {})
            merged.update(fields)
            self._child_progress(index, merged)
            self._report(force=False, pending=None, point=index)

        try:
            outcome = run_cached(plan.project, plan.options, self.runs_dir, inner, cache=self.cache, runner=self.runner,
                                 no_cache=self.no_cache, verify=self.verify)
        except Exception as error:
            import traceback
            self._record(plan, {"status": "failed", "key": key, "error": {
                "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=20)},
                "notes": []}, wall_s=time.monotonic() - started)
            return
        self._record(plan, _outcome_record(outcome, key), wall_s=time.monotonic() - started)

    def _spawn(self, plan, key: str) -> None:
        folder = plan.project.parent
        for stale in ("outcome.json", "progress.json"):
            if (folder / stale).exists():
                (folder / stale).unlink()
        job = {"format": "cnctwin-sweep-job", "version": VERSION, "index": plan.index, "project": str(plan.project),
               "options": plan.options, "runner": self.runner, "cache_root": str(self.cache.root),
               "cache_max_bytes": self.cache.max_bytes, "cache_max_age_s": self.cache.max_age_s,
               "runs_dir": str(self.runs_dir) if self.runs_dir is not None else None, "no_cache": self.no_cache,
               "verify": self.verify, "key": key, "watch_stdin": True, "memory_mode": self.mode,
               "memory_bytes": self.memory_bytes}
        job_path = folder / "job.json"
        _write_json(job_path, job)
        unit = f"cnctwin-sweep-{hashlib.sha256(str(self.out).encode()).hexdigest()[:8]}-p{plan.index:05d}-{os.getpid()}"
        env = point_environment()
        command, popen_env = point_command(job_path, mode=self.mode, memory_bytes=self.memory_bytes, env=env, unit=unit)
        log = open(folder / "log.txt", "ab")
        try:
            process = subprocess.Popen(command, env=popen_env, cwd=str(folder), stdin=subprocess.PIPE, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            log.close()
            self._record(plan, {"status": "failed", "key": key, "error": {
                "type": "LaunchError", "message": f"could not start the point process: {error}"}, "notes": []})
            return
        self.children[plan.index] = Child(plan, key, process, time.monotonic(), unit, log)

    def _poll(self) -> None:
        for index in sorted(self.children):
            child = self.children[index]
            folder = child.plan.project.parent
            progress = _read_json(folder / "progress.json")
            if isinstance(progress, dict):
                self._child_progress(index, progress)
            code = child.process.poll()
            if code is None:
                continue
            del self.children[index]
            child.log.close()
            if child.process.stdin is not None:
                child.process.stdin.close()
            outcome = _read_json(folder / "outcome.json")
            wall = time.monotonic() - child.started
            if isinstance(outcome, dict) and outcome.get("status") == "done":
                self._record(child.plan, _outcome_record(outcome.get("outcome") or {}, child.key), wall_s=wall)
            elif isinstance(outcome, dict):
                self._record(child.plan, {"status": "failed", "key": child.key, "error": outcome.get("error"),
                                          "notes": []}, wall_s=wall)
            else:
                self._record(child.plan, {"status": "failed", "key": child.key, "error": {
                    "type": "PointDied", "message": _exit_reason(code, self.mode, self.memory_bytes),
                    "log_tail": _tail(folder / "log.txt")}, "notes": []}, wall_s=wall)

    def _child_progress(self, index: int, progress: dict) -> None:
        child = self.children.get(index)
        if child is not None:
            child.progress = dict(progress)
        else:
            self._inline_progress = (index, dict(progress), time.monotonic())

    def _terminate(self) -> None:
        if not self.children:
            return
        for child in self.children.values():
            _signal(child, signal.SIGTERM)
        deadline = time.monotonic() + TERM_GRACE_S
        while time.monotonic() < deadline and any(c.process.poll() is None for c in self.children.values()):
            time.sleep(0.1)
        for child in self.children.values():
            if child.process.poll() is None:
                _signal(child, signal.SIGKILL)
                if self.mode == "systemd":
                    command = ([] if os.geteuid() == 0 else ["sudo", "-n"]) + [
                        "systemctl", "kill", "--signal=SIGKILL", f"{child.unit}.scope"]
                    subprocess.run(command, capture_output=True, timeout=10, check=False)
        for child in self.children.values():
            try:
                child.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            child.log.close()
            if child.process.stdin is not None:
                child.process.stdin.close()
        self.children.clear()

    # ------------------------------------------------------------------ records and progress
    def _record(self, plan, record: dict, wall_s: float | None = None) -> None:
        document = {"format": POINT_FORMAT, "version": VERSION, "index": plan.index, "values": plan.values,
                    "project": str(plan.project), "project_sha256": plan.project_sha256, "options": plan.options}
        for name in ("status", "key", "run_dir", "run_id", "verdict", "complete", "registered", "error", "notes"):
            document[name] = record.get(name)
        document["wall_s"] = wall_s
        _write_json(plan.project.parent / "result.json", document)
        self.results[plan.index] = document
        if document["status"] == "done" and wall_s is not None:
            self.walls.append(wall_s)
        self._report(force=True, pending=None)

    def counts(self) -> dict:
        out = {"done": 0, "cached": 0, "failed": 0}
        for result in self.results.values():
            out[result["status"]] = out.get(result["status"], 0) + 1
        return out

    def eta_s(self, pending: int) -> float | None:
        mean = sum(self.walls) / len(self.walls) if self.walls else None
        remaining = 0.0
        for child in self.children.values():
            elapsed = time.monotonic() - child.started
            sim, total = child.progress.get("sim_time_s"), child.progress.get("est_total_s")
            if isinstance(sim, (int, float)) and isinstance(total, (int, float)) and sim > 0:
                remaining += max(0.0, (total - sim) * elapsed / sim)
            elif mean is not None:
                remaining += max(0.0, mean - elapsed)
            else:
                return None
        if pending:
            if mean is None:
                return None
            remaining += pending * mean
        return remaining / max(1, self.workers)

    def _report(self, *, force: bool, pending: int | None, point: int | None = None) -> None:
        now = time.monotonic()
        if not force and now - self.last_progress < PROGRESS_EVERY_S:
            return
        self.last_progress = now
        if pending is None:
            pending = max(0, self.total - len(self.results) - len(self.children))
        counts = self.counts()
        finished = len(self.results)
        eta = self.eta_s(pending)
        running = sorted(self.children)
        inline = self._inline_progress
        current = {}
        if running:
            current = self.children[running[0]].progress
        elif inline is not None and point is not None:
            current = inline[1]
        progress = {"done": finished, "total": self.total, "cached": counts["cached"], "simulated": counts["done"],
                    "failed": counts["failed"], "running": running, "eta_s": eta}
        if isinstance(current.get("sim_time_s"), (int, float)):
            progress["point_sim_time_s"] = current.get("sim_time_s")
            progress["point_est_total_s"] = current.get("est_total_s")
        message = (f"{finished} of {self.total} points ({counts['cached']} cached, {counts['done']} simulated, "
                   f"{counts['failed']} failed)" + (f", running {', '.join(f'p{i:05d}' for i in running)}" if running else ""))
        self._status("running" if finished < self.total or self.children else "done", progress=progress,
                     message=message)
        if self.progress_cb is not None:
            self.progress_cb(progress, phase=f"point {min(finished + 1, self.total)}/{self.total}", message=message)

    def _status(self, status: str, *, progress: dict | None = None, message: str | None = None,
                error: str | None = None) -> None:
        counts = self.counts()
        document = {"format": STATUS_FORMAT, "version": VERSION, "name": self.spec.name, "status": status,
                    "total": self.total, "finished": len(self.results), **counts,
                    "running": sorted(self.children), "wall_s": time.monotonic() - self.t0,
                    "eta_s": (progress or {}).get("eta_s"), "message": message, "error": error}
        try:
            _write_json(self.out / "status.json", document)
        except OSError:
            pass

    def _summary(self, plans) -> dict:
        counts = self.counts()
        safe = sum(1 for r in self.results.values() if (r.get("verdict") or {}).get("safe") is True)
        unsafe = sum(1 for r in self.results.values() if (r.get("verdict") or {}).get("safe") is False)
        return {"format": SUMMARY_FORMAT, "version": VERSION, "name": self.spec.name, "spec_sha256": self.spec.sha256,
                "points": len(plans), **counts, "safe": safe, "unsafe": unsafe,
                "results": [{"index": p.index, "status": self.results[p.index]["status"],
                             "key": self.results[p.index].get("key"), "run_dir": self.results[p.index].get("run_dir"),
                             "run_id": self.results[p.index].get("run_id"),
                             "safe": (self.results[p.index].get("verdict") or {}).get("safe"),
                             "error": (self.results[p.index].get("error") or {}).get("message")}
                            for p in plans if p.index in self.results]}


def _outcome_record(outcome: dict, key: str) -> dict:
    return {"status": "cached" if outcome.get("cached") else "done", "key": outcome.get("key") or key,
            "run_dir": outcome.get("run_dir"), "run_id": outcome.get("run_id"), "verdict": outcome.get("verdict"),
            "complete": outcome.get("complete"), "registered": outcome.get("registered"), "error": None,
            "notes": list(outcome.get("notes") or [])}


def _signal(child: Child, sig: int) -> None:
    try:
        os.killpg(child.process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            child.process.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _exit_reason(code: int, mode: str | None, memory_bytes: int) -> str:
    number = -code if code < 0 else (code - 128 if code > 128 else 0)
    if number:
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = f"signal {number}"
        hint = f" (it may have reached its memory cap of {memory_bytes / 2 ** 20:.0f} MiB)" \
            if number == signal.SIGKILL and mode in ("systemd", "rlimit") else ""
        return f"the point process was killed by {name}{hint}"
    return f"the point process exited with status {code} without an outcome"


def _tail(path: Path) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - LOG_TAIL))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def read_sweep(out_dir) -> dict:
    """What a sweep folder says now: sweep.json, status.json, summary.json and every point's result (compact)."""
    out = Path(out_dir)
    if not (out / "sweep.json").is_file():
        raise InputError(f"{out} is not a sweep folder (no sweep.json)")
    sweep = _read_json(out / "sweep.json") or {}
    plan = _read_json(out / "plan.json") or {}
    points = []
    for entry in plan.get("points", []):
        result = _read_json(out / "points" / f"p{int(entry['index']):05d}" / "result.json") or {}
        points.append({"index": entry["index"], "values": entry.get("values"), "status": result.get("status", "pending"),
                       "key": result.get("key"), "run_id": result.get("run_id"), "run_dir": result.get("run_dir"),
                       "safe": (result.get("verdict") or {}).get("safe"),
                       "error": (result.get("error") or {}).get("message")})
    return {"name": sweep.get("name"), "spec": sweep.get("spec"), "notes": sweep.get("notes"),
            "status": _read_json(out / "status.json"), "summary": _read_json(out / "summary.json"), "points": points}


def clean_point_scratch(out_dir) -> None:
    """Remove the scratch files of point processes (job.json, progress.json, outcome.json, generator copies)."""
    for folder in sorted((Path(out_dir) / "points").glob("p*")):
        for name in ("progress.json",):
            if (folder / name).exists():
                (folder / name).unlink()
        for scratch in folder.glob(".generator-*"):
            shutil.rmtree(scratch, ignore_errors=True)


__all__ = ["plan_sweep", "SweepRun", "read_sweep", "point_environment", "point_command", "resolve_memory_mode",
           "clean_point_scratch"]
