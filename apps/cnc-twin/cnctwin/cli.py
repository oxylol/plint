"""``twin``: the command line of CNC Twin (``bin/twin``, or ``python -m cnctwin.cli``).

Every subcommand is dispatched lazily to the module that owns it, so a part that is missing from a
build costs that subcommand only (it exits 4 and names the missing module):

=============  ==============================================  ==================================================
command        owner                                           call
=============  ==============================================  ==================================================
run            cnctwin.core.loop.run_project                   run_project(project_path, run_dir, options, progress_cb)
check          cnctwin.core.loop.check_project                 check_project(request, out_dir, progress_cb) -> dict
replay         cnctwin.core.loop.replay_window                 replay_window(run_dir, tick0, tick1, options, progress_cb)
new            cnctwin.project.new_project                     new_project(argv) -> int
export         cnctwin.core.recorder.export_main               export_main(argv) -> int
compare        cnctwin.report.compare.main                     main(argv) -> int
uncertainty    cnctwin.uncertainty.main                        main(argv) -> int
calibrate      cnctwin.mdata.calibrate.main                    main(argv) -> int
validate       cnctwin.validation.main                         main(argv) -> int
serve          cnctwin.api.serve                               serve(argv) -> int
db             cnctwin.db.check.main                           main(argv) -> int  (``twin db check ...``)
bench          cnctwin.bench.main                              main(argv) -> int
warmup         cnctwin.core.warmup.main                        main(argv) -> int
sweep          cnctwin.sweep.main                              main(argv) -> int  (parameter sweeps, docs/ml.md)
dataset        cnctwin.dataset.main                            main(argv) -> int  (training data, docs/ml.md)
cache          cnctwin.cache.main                              main(argv) -> int  (the run cache, docs/ml.md)
explain        here                                            the cause chain of a finding from trace.jsonl
=============  ==============================================  ==================================================

``twin run`` allocates ``<runs>/<run_id>`` (cnctwin.core.rundir.run_id) unless ``--out`` names the run
directory, passes ``--set KEY=VALUE`` options (JSON values, dotted keys nest) and ``resume_from`` (the
latest committed checkpoint marker for ``--resume``), and writes status.json from the progress callback
(as the API worker does for its jobs: the loop never writes it). It goes through the run cache
(cnctwin.cache): a finished run of the same key (project and its inputs, databases, options, build, platform)
that still verifies is reported instead of running again (copied into ``--out`` when given); a new run is
registered when the key did not change while it ran (a resumed run too: it is bit-identical to an uninterrupted
one). ``--no-cache`` runs regardless (the new run supersedes); a project whose key cannot be computed runs
uncached. A run whose options carry operator overrides (``--set operator.feed_override=1.2``, docs/formats.md
"Operator option") is normalised (entries at 100 % dropped; nothing left: the plain run) and runs through
cnctwin.sweep.runner, which applies them (the loop's run_project does not read them); its cache key names that
runner, as a sweep point's does. ``twin replay`` of a run made with operator overrides (run.json
``options.operator``) replays through cnctwin.sweep.runner too, which applies them again.

``twin check`` writes ``check.json`` to ``--out``, by default ``DATA_DIR/checks/<project folder>-<h8>`` (the data
directory of the runs; ``<h8>`` from the project folder's path), or ``<tmp>/cnctwin-check/...`` when that folder
cannot be written: never into the project folder (the examples are part of the repository).

Exit status: 0 success (``run``: safe to run), 1 an error, 2 a usage error, 3 ``run`` finished and the
program is not safe to run, 4 the subcommand's module is not in this build, 130 interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_UNSAFE, EXIT_MISSING, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130

# name -> (module, attribute, one-line description): function(argv) -> int, argv after the command
FORWARDED = {
    "new": ("cnctwin.project", "new_project", "create a minimal project (twin new DIR --machine ID)"),
    "compare": ("cnctwin.report.compare", "main", "compare two finished runs (compare.html, compare.json)"),
    "uncertainty": ("cnctwin.uncertainty", "main", "uncertainty study of a project over its stated tolerances"),
    "calibrate": ("cnctwin.mdata.calibrate", "main", "calibrate the machine model from recorded machine data"),
    "validate": ("cnctwin.validation", "main", "run the validation cases against their measurements"),
    "serve": ("cnctwin.api", "serve", "run the HTTP API and the UI"),
    "db": ("cnctwin.db.check", "main", "validate the databases (twin db check [--strict] [--estimates] ...)"),
    "bench": ("cnctwin.bench", "main", "performance benchmarks (CPU seconds per simulated second)"),
    "warmup": ("cnctwin.core.warmup", "main", "compile and cache every kernel (the image runs it at build time)"),
    "export": ("cnctwin.core.recorder", "export_main", "export a run's time series as CSV or Parquet"),
    "sweep": ("cnctwin.sweep", "main", "run a parameter sweep through the run cache (training data, docs/ml.md)"),
    "dataset": ("cnctwin.dataset", "main", "export run-level and per-block training tables from sweeps and runs"),
    "cache": ("cnctwin.cache", "main", "the run cache: list, key, verify, evict, compact"),
}
LOOP = "cnctwin.core.loop"
OVERRIDE_RUNNER = "cnctwin.sweep.runner"      # run_project with the operator overrides of options["operator"]


class Missing(Exception):
    """The module or function a subcommand dispatches to is not in this build."""


class UsageError(Exception):
    """A command line or an argument that names something that is not there."""


def resolve(module_name: str, attribute: str):
    """``module.attribute`` imported now; Missing when this build does not have it."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing = getattr(error, "name", None) or str(error)
        if module_name == missing or module_name.startswith(missing + "."):
            raise Missing(f"{module_name}.{attribute} is not available in this build (no module {missing!r})") from None
        raise
    function = getattr(module, attribute, None)
    if not callable(function):
        raise Missing(f"{module_name}.{attribute} is not available in this build "
                      f"(module {module_name!r} has no callable {attribute!r})")
    return function


# =============================================================================== progress

class Progress:
    """progress_cb for the loop: one line on standard error, at most every ``interval_s`` of wall time, and
    (with ``run_dir``) the run's status.json, as the API worker writes it for its jobs: the loop itself
    reports only through this callback and never writes status.json."""

    def __init__(self, quiet: bool = False, interval_s: float = 0.5, stream=None, run_dir=None) -> None:
        self.quiet = quiet
        self.interval_s = interval_s
        self.stream = stream if stream is not None else sys.stderr
        self.state: dict = {}
        self.last = 0.0
        self.started = time.monotonic()
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.run_dir = None
        if run_dir is not None:
            from .core.rundir import RunDirectory
            self.run_dir = RunDirectory(run_dir)
            self.status("running")

    def status(self, state: str, error: dict | None = None) -> None:
        if self.run_dir is None:
            return
        wall = time.monotonic() - self.started
        sim, total = self.state.get("sim_time_s"), self.state.get("est_total_s")
        eta = self.state.get("eta_s")
        if eta is None and isinstance(sim, (int, float)) and isinstance(total, (int, float)) and sim > 0:
            eta = max(0.0, (total - sim) * wall / sim)
        if state != "running":
            eta = 0.0 if state == "done" else None
        extra = {k: v for k, v in self.state.items() if k in ("op", "phase", "message", "checkpoint")}
        self.run_dir.write_status(state, sim_time_s=float(sim or 0.0),
                                  est_total_s=float(total) if isinstance(total, (int, float)) else None,
                                  wall_s=wall, eta_s=eta, error=error, **extra)

    def __call__(self, progress: dict | None = None, **fields) -> None:
        self.state.update(progress or {})
        self.state.update(fields)
        now = time.monotonic()
        if now - self.last < self.interval_s:
            return
        self.last = now
        self.status("running")
        if self.quiet:
            return
        self.stream.write(self.line() + ("\r" if self.tty else "\n"))
        self.stream.flush()

    def line(self) -> str:
        sim = self.state.get("sim_time_s")
        total = self.state.get("est_total_s")
        parts = []
        if isinstance(sim, (int, float)):
            parts.append(f"t = {sim:.3f} s" + (f" of {total:.3f} s" if isinstance(total, (int, float)) else ""))
        for key in ("op", "phase", "message"):
            if self.state.get(key):
                parts.append(str(self.state[key]))
        parts.append(f"wall {time.monotonic() - self.started:.1f} s")
        return "  ".join(parts)

    def done(self) -> None:
        if not self.quiet and self.tty:
            self.stream.write("\n")
            self.stream.flush()


# =============================================================================== run, check, replay

def _options(pairs) -> dict:
    options = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise UsageError(f"--set {pair!r}: write KEY=VALUE")
        key, text = pair.split("=", 1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        node = options
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return options


def _project_file(path: str) -> Path:
    candidate = Path(path)
    project = candidate / "project.yaml" if candidate.is_dir() else candidate
    if not project.is_file():
        raise UsageError(f"no project file {project}")
    return project.resolve()


def _cache_key(project: Path, options: dict, runner: str | None = None):
    """The run's cache key through ``runner`` (default the loop), or None (the cache is off, this process is not
    deterministic, or the project does not resolve: the run then says why itself)."""
    from . import cache
    if not cache.enabled():
        return None
    if not cache.deterministic_environment():
        print("twin run: not cached: this process does not run under the deterministic environment "
              "(OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1, PYTHONHASHSEED=0; bin/twin sets them)", file=sys.stderr)
        return None
    from .core.errors import InputError
    try:
        return cache.key_for(project, options, runner=runner or cache.DEFAULT_RUNNER)
    except (InputError, OSError):
        return None


def operator_overrides(options: dict) -> bool:
    """Normalise ``options["operator"]`` in place (cnctwin.sweep.operator.normalise: entries at 100 % dropped; nothing
    overridden: the option is removed, the run is the plain run) and say whether the run carries overrides. A value
    that is not an operator option is a UsageError."""
    if "operator" not in options:
        return False
    from .sweep.operator import normalise
    try:
        operator = normalise(options["operator"])
    except ValueError as error:
        raise UsageError(f"--set operator: {error}") from None
    if operator is None:
        options.pop("operator")
        return False
    options["operator"] = operator
    return True


def _cached_run(key, out: str | None) -> Path | None:
    """The verified cached run of ``key`` (copied into ``out`` when given), or None."""
    from . import cache
    store = cache.RunCache()
    hit = store.lookup(key.key)
    if hit is None:
        return None
    print(f"cached run: {hit.run_dir} (key {key.key[:12]})", file=sys.stderr)
    if not out:
        return Path(hit.run_dir)
    run_dir = Path(out)
    if run_dir.resolve() != Path(hit.run_dir).resolve():
        cache.copy_run(hit.run_dir, run_dir)
        from .core.rundir import RunDirectory
        RunDirectory(run_dir).write_status("done", message=f"copied from the cached run {hit.run_dir}")
    return run_dir


def cmd_run(args) -> int:
    project = _project_file(args.project)
    options = _options(args.set)
    # the loop's run_project does not read options["operator"]: a run with overrides goes through the runner that
    # applies them (and refuses overrides the machine or the tools cannot take), and its cache key names that runner
    overrides = operator_overrides(options)
    if overrides:
        from .cache import SWEEP_RUNNER
        run_project, runner = resolve(OVERRIDE_RUNNER, "run_project"), SWEEP_RUNNER
    else:
        from .cache import DEFAULT_RUNNER
        run_project, runner = resolve(LOOP, "run_project"), DEFAULT_RUNNER
    key = _cache_key(project, options, runner)
    if key is not None and not args.no_cache and args.resume is None:
        cached = _cached_run(key, args.out)
        if cached is not None:
            return _report_verdict(cached)
    if args.out:
        run_dir = Path(args.out)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        from .core import rundir
        from .paths import data_dir
        runs = Path(args.runs) if args.runs else data_dir() / "runs"
        project_hash = key.components["project_hash"] if key is not None else _project_hash(project)
        run_dir = rundir.allocate_run_dir(runs, project_hash, options, slug=project.parent.name)
    if args.resume is not None:
        if not args.out:
            raise UsageError("--resume continues an existing run: name it with --out RUN_DIR")
        marker = Path(args.resume) if args.resume else None
        if marker is None:
            found = sorted((run_dir / "checkpoints").glob("ck_*.json"),
                           key=lambda p: int(p.stem.split("_")[1])) if (run_dir / "checkpoints").is_dir() else []
            found = [m for m in found if m.with_suffix(".npz").is_file()]
            if not found:
                raise UsageError(f"{run_dir} has no committed checkpoint to resume from")
            marker = found[-1]
        elif not marker.is_file():
            raise UsageError(f"no checkpoint marker {marker}")
        options["resume_from"] = str(marker.resolve())
    print(f"run directory: {run_dir}", file=sys.stderr)
    progress = Progress(args.quiet, run_dir=run_dir)
    try:
        run_project(project, run_dir, options, progress)
    except KeyboardInterrupt:
        progress.status("cancelled", {"type": "KeyboardInterrupt", "message": "interrupted"})
        raise
    except BaseException as error:
        progress.status("failed", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        progress.done()
    progress.status("done")
    if key is not None:
        from . import cache
        registered = cache.register_finished(project, run_dir, options, key, runner=runner)
        if not registered["registered"]:
            print(f"twin run: not cached: {registered['note']}", file=sys.stderr)
    return _report_verdict(run_dir)


def _report_verdict(run_dir: Path) -> int:
    verdict = _verdict(run_dir)
    if verdict is None:
        print(f"{run_dir}: the run wrote no verdict (run.json)", file=sys.stderr)
        return EXIT_ERROR
    if verdict.get("safe"):
        print("SAFE TO RUN")
        return EXIT_OK
    print("NOT SAFE TO RUN")
    for reason in verdict.get("reasons", []):
        print(f"  {reason}")
    return EXIT_UNSAFE


def _project_hash(project: Path) -> str:
    from .project import resolve as resolve_project
    return resolve_project(project).hash


def _verdict(run_dir: Path):
    try:
        return json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("verdict")
    except (OSError, ValueError):
        return None


def default_check_dir(project: Path) -> Path:
    """Where ``twin check`` writes when ``--out`` is not given: ``DATA_DIR/checks/<project folder slug>-<h8>`` (the
    data directory of the runs, cnctwin.paths.data_dir; h8 = the first 8 hex digits of sha256(project folder path),
    so two projects in folders of the same name do not share one), else ``<tmp>/cnctwin-check/<same name>`` when that
    folder cannot be created or written. Never the project folder: the examples are part of the repository."""
    from .core.rundir import slugify
    from .paths import data_dir
    folder = Path(project).resolve().parent
    name = f"{slugify(folder.name, 40)}-{hashlib.sha256(str(folder).encode('utf-8')).hexdigest()[:8]}"
    for base in (data_dir() / "checks", Path(tempfile.gettempdir()) / "cnctwin-check"):
        target = base / name
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(target, os.W_OK | os.X_OK):
            return target
    raise UsageError(f"no writable folder for the check's files ({data_dir() / 'checks'}, "
                     f"{Path(tempfile.gettempdir()) / 'cnctwin-check'}): give --out DIR")


def cmd_check(args) -> int:
    check_project = resolve(LOOP, "check_project")
    project = _project_file(args.project)
    out_dir = Path(args.out) if args.out else default_check_dir(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.out and not args.quiet:
        print(f"check files: {out_dir}", file=sys.stderr)
    request = {"project_path": str(project), "project_dir": str(project.parent), "read_only": not args.write,
               "options": {"write": bool(args.write), **_options(args.set)}, "out_dir": str(out_dir)}
    progress = Progress(args.quiet)
    try:
        result = check_project(request, out_dir, progress)
    finally:
        progress.done()
    result = result or {}
    if args.json:
        print(json.dumps(result, indent=1, default=str))
    else:
        _print_check(result)
    if isinstance(result.get("exit_code"), int):
        return int(result["exit_code"])
    if result.get("ok") is False or result.get("errors"):
        return EXIT_ERROR
    return EXIT_OK


def _print_check(result: dict) -> None:
    for key in ("summary", "message"):
        if isinstance(result.get(key), str):
            print(result[key])
    for key in ("errors", "warnings", "problems"):
        items = result.get(key)
        if isinstance(items, list) and items:
            print(f"{key}:")
            for item in items:
                print(f"  {item if isinstance(item, str) else json.dumps(item, default=str)}")
    for key, value in result.items():
        if key not in ("summary", "message", "errors", "warnings", "problems") and isinstance(value, (int, float, str)):
            print(f"{key}: {value}")


def cmd_replay(args) -> int:
    run_dir = Path(args.run_dir)
    try:
        recorded = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("options") or {}
    except (OSError, ValueError, AttributeError):
        recorded = {}
    if isinstance(recorded, dict) and recorded.get("operator"):
        replay_window = resolve("cnctwin.sweep.runner", "replay_window")    # the run's overrides, applied again
    else:
        replay_window = resolve(LOOP, "replay_window")
    tick_s = _tick_s(run_dir)
    tick0 = int(round(args.window[0] / tick_s))
    tick1 = int(round(args.window[1] / tick_s))
    if tick1 <= tick0:
        raise UsageError("--window T0 T1 needs T1 after T0")
    progress = Progress(args.quiet)
    try:
        result = replay_window(run_dir, tick0, tick1, _options(args.set), progress)
    finally:
        progress.done()
    if result:
        print(json.dumps(result, indent=1, default=str))
    return EXIT_OK


def _tick_s(run_dir: Path) -> float:
    try:
        clock = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("clock") or {}
        if isinstance(clock.get("tick_s"), (int, float)):
            return float(clock["tick_s"])
    except (OSError, ValueError):
        pass
    return 1e-6


# =============================================================================== explain

def _first_line_at(handle, position: int):
    if position <= 0:
        handle.seek(0)
    else:
        handle.seek(position - 1)
        handle.readline()
    start = handle.tell()
    line = handle.readline()
    if not line or not line.endswith(b"\n"):
        return None
    return start, line


def trace_event(handle, size: int, event_id: int) -> dict | None:
    """The trace.jsonl line whose id is ``event_id`` (ids rise one per line: bisect the byte offsets)."""
    if event_id < 0:
        return None
    lo, hi = 0, size
    while lo < hi:
        mid = (lo + hi) // 2
        found = _first_line_at(handle, mid)
        if found is None or json.loads(found[1])["id"] >= event_id:
            hi = mid
        else:
            lo = mid + 1
    found = _first_line_at(handle, lo)
    if found is None:
        return None
    record = json.loads(found[1])
    return record if record.get("id") == event_id else None


def _findings(run_dir: Path) -> list[dict]:
    path = run_dir / "findings.json"
    if not path.is_file():
        raise UsageError(f"{run_dir} has no findings.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    items = document.get("findings", []) if isinstance(document, dict) else document
    return [f for f in items if isinstance(f, dict)]


def _format_quantities(quantities: dict) -> str:
    parts = []
    for key, value in sorted((quantities or {}).items()):
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _event_line(event: dict) -> str:
    where = f"  [line {event['line']}]" if event.get("line") else ""
    quantities = _format_quantities(event.get("quantities", {}))
    return (f"#{event['id']}  t={float(event.get('time_s', 0.0)):.6f} s  {event.get('domain', '')}  "
            f"{event.get('event', '')}" + (f"  {quantities}" if quantities else "") + where)


def cause_tree(handle, size: int, root_id: int, max_events: int = 200) -> list[tuple[int, dict, bool]]:
    """(depth, event, seen_before) in depth-first order from ``root_id`` through the parents; an event
    reached again is listed once more as a reference, its parents not repeated."""
    out: list[tuple[int, dict, bool]] = []
    seen: set[int] = set()
    stack = [(root_id, 0)]
    while stack and len(out) < max_events:
        event_id, depth = stack.pop()
        event = trace_event(handle, size, event_id)
        if event is None:
            out.append((depth, {"id": event_id, "event": "(not in trace.jsonl)", "domain": "", "time_s": 0.0}, True))
            continue
        again = event_id in seen
        out.append((depth, event, again))
        if again:
            continue
        seen.add(event_id)
        for parent in reversed([int(p) for p in event.get("parents", [])]):
            stack.append((parent, depth + 1))
    return out


def explain(run_dir, finding_id: int | None, as_json: bool = False, max_events: int = 200, out=None) -> int:
    """Print a finding and its chain of cause (trace.jsonl), or the findings list when no id is given."""
    out = out if out is not None else sys.stdout
    run_dir = Path(run_dir)
    findings = _findings(run_dir)
    if finding_id is None:
        if as_json:
            out.write(json.dumps([{k: f.get(k) for k in ("id", "category", "severity", "time_s", "op", "line",
                                                           "message")} for f in findings], indent=1) + "\n")
            return EXIT_OK
        if not findings:
            out.write("no findings\n")
        for f in findings:
            out.write(f"{f.get('id'):>4}  {f.get('severity', ''):7s}  {f.get('category', ''):24s}  "
                      f"t={float(f.get('time_s', 0.0)):.6f} s  {f.get('op') or '-'} line {f.get('line', 0)}  "
                      f"{f.get('message', '')}\n")
        return EXIT_OK
    finding = next((f for f in findings if int(f.get("id", -1)) == int(finding_id)), None)
    if finding is None:
        raise UsageError(f"{run_dir} has no finding {finding_id} (twin explain {run_dir} lists them)")
    trace_path = run_dir / "trace.jsonl"
    chain = [int(i) for i in finding.get("cause_chain", [])]
    size = trace_path.stat().st_size if trace_path.is_file() else 0
    with open(trace_path, "rb") if trace_path.is_file() else open(os.devnull, "rb") as handle:
        tree = cause_tree(handle, size, chain[0], max_events) if chain and size else []
        roots = []
        if chain and size:
            for event_id in sorted(set(chain)):
                event = trace_event(handle, size, event_id)
                if event is not None and not event.get("parents"):
                    roots.append(event)
    if as_json:
        out.write(json.dumps({"finding": finding, "chain": [dict(e, depth=d, repeated=again) for d, e, again in tree],
                              "roots": roots}, indent=1, default=str) + "\n")
        return EXIT_OK
    position = finding.get("pos_W_m")
    out.write(f"Finding {finding.get('id')}: {finding.get('category')} ({finding.get('severity')}) at "
              f"{finding.get('op') or '-'} {finding.get('program') or ''} line {finding.get('line', 0)}, "
              f"t = {float(finding.get('time_s', 0.0)):.6f} s (tick {finding.get('tick', 0)})\n")
    out.write(f"  {finding.get('message', '')}\n")
    if finding.get("block"):
        out.write(f"  block: {finding['block']}\n")
    if isinstance(position, list) and len(position) == 3:
        out.write("  tool tip in W: " + ", ".join(f"{1e3 * float(v):.4f}" for v in position) + " mm\n")
    quantities = _format_quantities(finding.get("quantities", {}))
    if quantities:
        out.write(f"  {quantities}\n")
    if not tree:
        out.write("Cause chain: none recorded\n")
        return EXIT_OK
    out.write("Cause chain (trace.jsonl, each event above the events it was derived from):\n")
    for depth, event, again in tree:
        prefix = "  " + "   " * max(0, depth - 1) + ("└─ " if depth else "")
        out.write(prefix + _event_line(event) + ("  (see above)" if again else "") + "\n")
    if roots:
        out.write("Root causes: " + "; ".join(f"#{e['id']} {e.get('domain', '')} {e.get('event', '')}" for e in roots)
                  + "\n")
    return EXIT_OK


def cmd_explain(args) -> int:
    return explain(args.run_dir, args.finding_id, args.json, args.max_events)


# =============================================================================== the parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="twin", description="CNC Twin: a physics twin of a CNC machine, run "
                                     "before the program runs on the machine.",
                                     epilog="twin <command> --help shows the options of a command.")
    parser.add_argument("--traceback", action="store_true", help="show the Python traceback of an error")
    sub = parser.add_subparsers(dest="command", metavar="command")

    run = sub.add_parser("run", help="simulate a project (writes a run directory)")
    run.add_argument("project", help="project.yaml or its folder")
    target = run.add_mutually_exclusive_group()
    target.add_argument("--out", help="the run directory (default: a new numbered one under --runs)")
    target.add_argument("--runs", help="folder of run directories (default: DATA_DIR/runs)")
    run.add_argument("--resume", nargs="?", const="", default=None, metavar="CHECKPOINT",
                     help="continue the run in --out from a checkpoint marker (default: its latest)")
    run.add_argument("--set", action="append", metavar="KEY=VALUE", help="a run option (JSON value), repeatable")
    run.add_argument("--no-cache", action="store_true",
                     help="run even when the run cache has this run (the new run is registered and supersedes it)")
    run.add_argument("--quiet", action="store_true", help="no progress lines")
    run.set_defaults(handler=cmd_run)

    check = sub.add_parser("check", help="check a project without running it (timestep, memory, size estimates)")
    check.add_argument("project", help="project.yaml or its folder")
    check.add_argument("--write", action="store_true", help="store the resolved strides under physics.resolved")
    check.add_argument("--json", action="store_true", help="print the result as JSON")
    check.add_argument("--out", help="folder for the check's files (default: DATA_DIR/checks/<project>-<h8>, "
                                     "else under the temporary folder; never the project folder)")
    check.add_argument("--set", action="append", metavar="KEY=VALUE", help="a check option (JSON value)")
    check.add_argument("--quiet", action="store_true")
    check.set_defaults(handler=cmd_check)

    replay = sub.add_parser("replay", help="replay a window of a finished run from its nearest checkpoint")
    replay.add_argument("run_dir", help="run directory")
    replay.add_argument("--window", nargs=2, type=float, required=True, metavar=("T0", "T1"),
                        help="simulated seconds; the fast tier of this window is written")
    replay.add_argument("--set", action="append", metavar="KEY=VALUE")
    replay.add_argument("--quiet", action="store_true")
    replay.set_defaults(handler=cmd_replay)

    explain_parser = sub.add_parser("explain", help="the chain of cause of a finding (or the list of findings)")
    explain_parser.add_argument("run_dir", help="run directory")
    explain_parser.add_argument("finding_id", nargs="?", type=int, default=None, help="finding id (findings.json)")
    explain_parser.add_argument("--json", action="store_true", help="print as JSON")
    explain_parser.add_argument("--max-events", type=int, default=200, help="at most this many events (default 200)")
    explain_parser.set_defaults(handler=cmd_explain)

    for name, (module, attribute, text) in FORWARDED.items():
        forwarded = sub.add_parser(name, help=text, add_help=False, prefix_chars="\x00")
        forwarded.add_argument("rest", nargs=argparse.REMAINDER)
        forwarded.set_defaults(handler=None, forward=(module, attribute))
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return EXIT_USAGE
    show_traceback = "--traceback" in argv[:1] or os.environ.get("CNCTWIN_TRACEBACK") == "1"
    if argv[:1] == ["--traceback"]:
        argv = argv[1:]
    if argv and argv[0] in FORWARDED:
        module, attribute = FORWARDED[argv[0]][:2]
        rest = argv[1:]
        if argv[0] == "db" and rest[:1] == ["check"]:
            rest = rest[1:]
        return _run(lambda: resolve(module, attribute)(rest), argv[0], show_traceback)
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is None:
        parser.print_help()
        return EXIT_USAGE
    return _run(lambda: args.handler(args), args.command, show_traceback or args.traceback)


def _run(call, command: str, show_traceback: bool) -> int:
    from .core.errors import TwinError
    try:
        code = call()
    except Missing as error:
        print(f"twin {command}: {error}", file=sys.stderr)
        return EXIT_MISSING
    except KeyboardInterrupt:
        print(f"twin {command}: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (TwinError, UsageError, FileNotFoundError) as error:
        if show_traceback:
            raise
        print(f"twin {command}: error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"twin {command}: failed (the traceback above is a bug report)", file=sys.stderr)
        return EXIT_ERROR
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else (EXIT_OK if error.code is None else EXIT_ERROR)
    return int(code) if isinstance(code, int) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
