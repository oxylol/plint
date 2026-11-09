"""The process of one job: ``python -m cnctwin.api.worker <job_dir>`` (started by the job manager).

It reads ``job.json``, caps its own memory in ``rlimit`` mode (before importing anything large),
lowers its priority (nice 10), starts a new process group (so a cancel reaches everything it
starts), resolves the entry point and calls it:

* ``simulate``: ``run_project(project_path, run_dir, options, progress_cb)`` (cnctwin.core.loop);
  ``options["resume_from"]`` is the checkpoint marker ``.json`` when the run is resumed. After it returns with
  run.json written, the request's ``outputs_entry`` (cnctwin.core.outputs.finalize_outputs(run_dir, options,
  progress_cb)) makes the report, setup sheet, pictures, video and sound; its summary is the result's
  ``outputs``.
* every other kind: ``entry(request, out_dir, progress_cb) -> dict`` (the request carries the
  paths and options of that kind; the returned dict is saved as the job's ``result.json``).

``progress_cb(progress: dict | None = None, **fields)`` takes any of ``sim_time_s``,
``est_total_s``, ``eta_s``, ``op``, ``phase``, ``message``, ``checkpoint`` (and other JSON-able
fields, kept as given). It returns None, and raises :class:`Cancelled` after a cancel request; that
is a BaseException, so an ``except Exception`` in the loop does not swallow it.

The worker owns ``status.json`` of a simulation while it runs: status (running, cancelling, then
done, failed or cancelled), progress (wall time measured here, summed over attempts; the eta from
the rate of this attempt when the loop gives none) and the error with its traceback. Its own state
goes to ``worker.json`` (the last write has ``final: true``).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path


class Cancelled(BaseException):
    """Raised from progress_cb (or on SIGTERM) after a cancel request."""


def _apply_memory_cap(job: dict) -> None:
    memory = job.get("memory_mode")
    limit = int(job.get("memory_bytes") or 0)
    if memory != "rlimit" or limit <= 0:
        return
    import resource
    for name in ("RLIMIT_DATA", "RLIMIT_AS"):
        kind = getattr(resource, name, None)
        if kind is None:
            continue
        soft, hard = resource.getrlimit(kind)
        cap = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        try:
            resource.setrlimit(kind, (cap, cap))
            return
        except (ValueError, OSError):
            continue


def _jsonable(value):
    from .jsonio import plain
    try:
        return plain(value)
    except TypeError:
        return str(value)


class Worker:
    def __init__(self, job_dir: Path, job: dict) -> None:
        self.dir = job_dir
        self.job = job
        self.kind = job["kind"]
        self.request = job.get("request") or {}
        self.interval_s = float(self.request.get("status_interval_s") or 0.2)
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.stop = threading.Event()
        self.finishing = False
        self.progress: dict = {}
        self.dirty = False
        self.state = "running"
        self.t0 = time.monotonic()
        self.prior_wall_s = float(job.get("prior_wall_s") or 0.0)
        self.sim_start_s: float | None = None
        self.run_dir = Path(self.request["run_dir"]) if self.kind == "simulate" else None
        self.last_write = 0.0
        self.thread: threading.Thread | None = None

    # ------------------------------------------------------------------ the run
    def run(self) -> int:
        self._write_worker(final=False)
        self._write_progress(force=True)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        self.thread = threading.Thread(target=self._watch, name="cnctwin-status", daemon=True)
        self.thread.start()
        from .entry import EntryPointMissing, resolve
        try:
            function = resolve(self.job["entry"])
            if self.kind == "simulate":
                options = dict(self.request.get("options") or {})
                if self.request.get("resume_from"):
                    options["resume_from"] = self.request["resume_from"]
                result = function(Path(self.request["project_path"]), self.run_dir, options, self.progress_cb)
                result = self._outputs(options, result)
            else:
                out_dir = Path(self.request["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                result = function(self.request, out_dir, self.progress_cb)
        except EntryPointMissing as error:
            return self._finish("failed", {"type": "EntryPointMissing", "message": str(error)})
        except Cancelled as error:
            return self._finish("cancelled", {"type": "Cancelled", "message": str(error) or "cancelled on request"})
        except MemoryError:
            cap = int(self.job.get("memory_bytes") or 0) / 2**20
            return self._finish("failed", {"type": "MemoryError",
                                           "message": f"the job ran out of memory (cap {cap:.0f} MiB)",
                                           "traceback": traceback.format_exc(limit=30)})
        except Exception as error:
            return self._finish("failed", {"type": type(error).__name__, "message": str(error),
                                           "traceback": traceback.format_exc(limit=30)})
        except (KeyboardInterrupt, SystemExit) as error:
            return self._finish("failed", {"type": type(error).__name__, "message": str(error) or "interrupted",
                                           "traceback": traceback.format_exc(limit=30)})
        return self._finish("done", None, result)

    def _outputs(self, options: dict, result):
        """After a successful run: the derived outputs (report, setup sheet, pictures, video, sound) through the
        request's ``outputs_entry`` (cnctwin.core.outputs.finalize_outputs(run_dir, options, progress_cb)), while
        status.json still says running (phase ``outputs: <generator>``). A generator that fails is recorded in
        run.json by finalize_outputs; finalize_outputs failing as a whole leaves the run done and says why in the
        job's result. A cancel request (Cancelled) passes on."""
        from .entry import resolve
        from .settings import NO_ENTRY
        spec = self.request.get("outputs_entry")
        if spec is None or str(spec).strip().lower() in NO_ENTRY or not (self.run_dir / "run.json").is_file():
            return result
        merged = dict(result) if isinstance(result, dict) else {"run": result}
        try:
            merged["outputs"] = resolve(str(spec))(self.run_dir, options, self.progress_cb)
        except Exception as error:
            merged["outputs_error"] = {"type": type(error).__name__, "message": str(error),
                                       "traceback": traceback.format_exc(limit=30)}
        return merged

    def progress_cb(self, progress: dict | None = None, **fields) -> None:
        if self.cancel.is_set():
            raise Cancelled("cancelled on request")
        data = dict(progress or {})
        data.update(fields)
        with self.lock:
            for key, value in data.items():
                self.progress[str(key)] = _jsonable(value)
            if self.sim_start_s is None and isinstance(self.progress.get("sim_time_s"), (int, float)):
                self.sim_start_s = float(self.progress["sim_time_s"])
            self.dirty = True

    def _on_sigterm(self, signum, frame) -> None:
        if self.finishing:
            return
        self.cancel.set()
        raise Cancelled("stopped by SIGTERM")

    def _watch(self) -> None:
        while not self.stop.wait(self.interval_s):
            if not self.cancel.is_set() and (self.dir / "cancel").exists():
                self.cancel.set()
                with self.lock:
                    self.state = "cancelling"
                    self.dirty = True
            self._write_progress()

    # ------------------------------------------------------------------ files
    def _wall_s(self) -> float:
        return self.prior_wall_s + (time.monotonic() - self.t0)

    def _status_progress(self) -> dict:
        progress = dict(self.progress)
        progress.setdefault("sim_time_s", 0.0)
        progress.setdefault("est_total_s", None)
        wall = self._wall_s()
        progress["wall_s"] = wall
        if "eta_s" not in self.progress:
            sim, total = progress.get("sim_time_s"), progress.get("est_total_s")
            attempt_wall = time.monotonic() - self.t0
            done = (sim or 0.0) - (self.sim_start_s or 0.0)
            if isinstance(total, (int, float)) and isinstance(sim, (int, float)) and done > 0 and attempt_wall > 0:
                progress["eta_s"] = max(0.0, (total - sim) * attempt_wall / done)
            else:
                progress["eta_s"] = None
        return progress

    def _write_progress(self, force: bool = False) -> None:
        with self.lock:
            due = force or self.dirty or (time.monotonic() - self.last_write) >= 1.0
            if not due:
                return
            self.dirty = False
            self.last_write = time.monotonic()
            state = self.state
            progress = self._status_progress()
        if self.kind == "simulate":
            self._write_status(state, progress, None)
        else:
            self._write_worker(final=False, progress=progress)

    def _write_status(self, state: str, progress: dict, error: dict | None) -> None:
        from .jsonio import write_json_atomic
        document = {"format": "cnctwin-run-status", "version": 1, "status": state, "progress": progress,
                    "job_id": self.job.get("id"), "attempt": self.job.get("attempt"), "error": error,
                    "op": progress.get("op"), "phase": progress.get("phase"), "message": progress.get("message"),
                    "checkpoint": progress.get("checkpoint"), "resume_from": self.request.get("resume_from")}
        write_json_atomic(self.run_dir / "status.json", document, fsync=False)

    def _write_worker(self, *, final: bool, progress: dict | None = None, error: dict | None = None,
                      result: bool = False) -> None:
        from .jsonio import write_json_atomic
        document = {"format": "cnctwin-worker", "version": 1, "pid": os.getpid(), "state": self.state,
                    "final": final, "progress": progress if progress is not None else self._status_progress(),
                    "error": error, "result": result, "wall_s": self._wall_s()}
        write_json_atomic(self.dir / "worker.json", document, fsync=final)

    def _finish(self, state: str, error: dict | None, result=None) -> int:
        self.finishing = True
        self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=10)
        with self.lock:
            self.state = state
            progress = self._status_progress()
        if state != "running":
            progress["eta_s"] = None if state != "done" else 0.0
        has_result = False
        if state == "done" and result is not None:
            from .jsonio import write_json_atomic
            try:
                write_json_atomic(self.dir / "result.json", result)
                has_result = True
            except TypeError as problem:
                state, error = "failed", {"type": "TypeError",
                                          "message": f"the job's result is not JSON-serialisable: {problem}"}
                self.state = state
        # the job's own record first: once status.json shows the end, the job has ended too
        self._write_worker(final=True, progress=progress, error=error, result=has_result)
        if self.kind == "simulate":
            self._write_status(state, progress, error)
        return 0 if state in ("done", "cancelled") else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m cnctwin.api.worker <job_dir>", file=sys.stderr)
        return 2
    job_dir = Path(argv[0])
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    _apply_memory_cap(job)
    try:
        os.setsid()
    except PermissionError:
        pass
    try:
        os.nice(10)
    except OSError:
        pass
    return Worker(job_dir, job).run()


if __name__ == "__main__":
    raise SystemExit(main())
