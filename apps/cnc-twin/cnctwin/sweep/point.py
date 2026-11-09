"""One sweep point in its own process: ``python -m cnctwin.sweep.point <points/p<nnnnn>/job.json>``.

Started by cnctwin.sweep.execute with the deterministic environment (and, in ``systemd`` mode, inside its own
memory-capped scope). It caps its own data segment first in ``rlimit`` mode (before importing anything large),
lowers its priority (nice 10), watches its standard input (the sweep process holds the other end: end of file
means that process is gone, and the point stops), runs the variant through the run cache
(cnctwin.cache.run_cached) and writes:

* ``progress.json`` (at most every 0.5 s; format ``cnctwin-sweep-progress``): the loop's progress
  (``sim_time_s``, ``est_total_s``, ``op``, ``phase``) and this process's wall time;
* ``outcome.json`` at the end (format ``cnctwin-sweep-outcome``): ``{status: done | failed, outcome: run_cached's
  record, error: {type, message, traceback}}``.

Both carry ``format`` and ``version`` (1); the sweep process (cnctwin.sweep.execute) also reads them without (written
by an older point process).

Exit status 0 when the run finished (safe or not), 1 when it failed.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

PROGRESS_EVERY_S = 0.5
PROGRESS_FORMAT = "cnctwin-sweep-progress"
OUTCOME_FORMAT = "cnctwin-sweep-outcome"
VERSION = 1


def _apply_memory_cap(job: dict) -> None:
    if job.get("memory_mode") != "rlimit" or not int(job.get("memory_bytes") or 0):
        return
    from ..core.outputs import cap_memory       # maps the BLAS buffer first: OpenBLAS spins on a failed map
    cap_memory(int(job["memory_bytes"]))


def _watch_stdin() -> None:
    def watch():
        try:
            while sys.stdin.buffer.read(65536):
                pass
        except (OSError, ValueError):
            pass
        os._exit(75)
    threading.Thread(target=watch, name="cnctwin-sweep-parent", daemon=True).start()


def _write(path: Path, document) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(document, default=str, allow_nan=False) if _finite(document)
                   else json.dumps(_plain(document), default=str), encoding="utf-8")
    os.replace(tmp, path)


def _finite(document) -> bool:
    try:
        json.dumps(document, default=str, allow_nan=False)
        return True
    except ValueError:
        return False


def _plain(value):
    from ..core.rundir import jsonable
    try:
        return jsonable(value)
    except TypeError:
        return str(value)


class Progress:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.state: dict = {}
        self.t0 = time.monotonic()
        self.last = 0.0

    def __call__(self, progress=None, **fields) -> None:
        self.state.update(progress or {})
        self.state.update(fields)
        now = time.monotonic()
        if now - self.last < PROGRESS_EVERY_S and fields.get("phase") not in ("done", "cached"):
            return
        self.last = now
        document = {k: v for k, v in self.state.items() if k in ("sim_time_s", "est_total_s", "eta_s", "op", "phase",
                                                                 "message", "part")}
        document["wall_s"] = now - self.t0
        try:
            _write(self.folder / "progress.json", {"format": PROGRESS_FORMAT, "version": VERSION, **_plain(document)})
        except OSError:
            pass


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m cnctwin.sweep.point <job.json>", file=sys.stderr)
        return 2
    job_path = Path(argv[0]).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    _apply_memory_cap(job)
    try:
        os.nice(10)
    except OSError:
        pass
    if job.get("watch_stdin"):
        _watch_stdin()
    folder = job_path.parent
    from ..cache import RunCache, run_cached
    cache = RunCache(job["cache_root"], max_bytes=job.get("cache_max_bytes"), max_age_s=job.get("cache_max_age_s"))
    progress = Progress(folder)
    try:
        outcome = run_cached(job["project"], job.get("options") or {}, job.get("runs_dir"), progress, cache=cache,
                             runner=job.get("runner"), no_cache=bool(job.get("no_cache")),
                             verify=job.get("verify") or "full")
        outcome = {k: v for k, v in outcome.items() if k != "result"}
        document = {"status": "done", "outcome": _plain(outcome), "error": None}
        code = 0
    except MemoryError:
        document = {"status": "failed", "outcome": None, "error": {
            "type": "MemoryError", "message": f"the point ran out of memory (cap {int(job.get('memory_bytes') or 0) / 2 ** 20:.0f} MiB)",
            "traceback": traceback.format_exc(limit=30)}}
        code = 1
    except Exception as error:
        document = {"status": "failed", "outcome": None, "error": {
            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=30)}}
        code = 1
    _write(folder / "outcome.json", {"format": OUTCOME_FORMAT, "version": VERSION, **document})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
