"""Parameter sweeps: many project variants run through the run cache, as training data (docs/ml.md "Sweeps",
docs/formats.md "Sweep").

* :func:`run_sweep` (``spec, out_dir=None, *, progress_cb=None, plan_only=False, ...``) -> dict: load the spec
  (cnctwin.sweep.spec), draw the design (cnctwin.sweep.design), write every point's variant
  (cnctwin.sweep.variants; generators: cnctwin.sweep.generate), run the points (cnctwin.sweep.execute: one process
  per point, memory-capped, ``workers`` at once, cached points skipped), optionally export the dataset
  (cnctwin.dataset); returns summary.json's document.
* :func:`run_sweep_job` (``request, out_dir, progress_cb``) -> dict: the API's job entry (``POST /api/sweeps``).
* :func:`load_spec`, :func:`plan_sweep`, :func:`read_sweep`.
* :func:`main` (``argv=None``) -> exit code: ``twin sweep SPEC [--out DIR] [--plan-only] [--workers N]
  [--isolation process|inline] [--memory 1536M] [--memory-mode auto|systemd|rlimit|none] [--runner SPEC]
  [--no-cache] [--retry-failed] [--dataset [DIR]] [--json] [--quiet]``, ``twin sweep --show DIR``.

The simulation of each point is cnctwin.sweep.runner.run_project (the loop, plus the operator overrides of
``override`` parameters).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

__all__ = ["run_sweep", "run_sweep_job", "load_spec", "plan_sweep", "read_sweep", "main", "default_out_dir"]


def __getattr__(name: str):
    if name == "load_spec":
        from .spec import load_spec
        return load_spec
    if name in ("plan_sweep", "read_sweep"):
        from . import execute
        return getattr(execute, name)
    raise AttributeError(f"module 'cnctwin.sweep' has no attribute {name!r}")


def default_out_dir(name: str) -> Path:
    from ..core.rundir import slugify
    from ..paths import data_dir
    return data_dir() / "sweeps" / slugify(name)


def run_sweep(spec, out_dir=None, *, progress_cb=None, plan_only: bool = False, cache=None, runs_dir=None,
              workers: int | None = None, isolation: str | None = None, memory=None, memory_mode: str | None = None,
              runner: str | None = None, retry_failed: bool | None = None, no_cache: bool = False, dataset_dir=None,
              base_override=None, db=None) -> dict:
    """Run a sweep (module docstring). ``spec``: a spec file, a mapping or a SweepSpec."""
    from .execute import SweepRun
    from .spec import SweepSpec, load_spec, parse_bytes
    if not isinstance(spec, SweepSpec):
        spec = load_spec(spec, base_override=base_override)
    out = Path(out_dir) if out_dir is not None else default_out_dir(spec.name)
    run = SweepRun(spec, out, cache=cache, runs_dir=runs_dir, workers=workers, isolation=isolation,
                   memory_bytes=parse_bytes(memory) if memory is not None else None, memory_mode=memory_mode,
                   runner=runner, retry_failed=retry_failed, no_cache=no_cache, progress_cb=progress_cb, db=db)
    summary = run.run(plan_only=plan_only)
    summary["out_dir"] = str(out)
    if dataset_dir is not None and not plan_only:
        from ..dataset import export
        if progress_cb is not None:
            progress_cb(phase="dataset", message="exporting the dataset")
        manifest = export([out], dataset_dir)
        summary["dataset"] = {"out_dir": str(dataset_dir), "files": manifest["files"], "runs": manifest["runs"],
                              "blocks": manifest["blocks"]}
    return summary


def run_sweep_job(request: dict, out_dir, progress_cb=None) -> dict:
    """The API's sweep job (docs/api.md "Sweeps"): ``request`` {project_id, project_path, project_dir, spec,
    options, simulate_entry}; the spec's base is the project, its generators must lie in the project's folder. The
    points' runner is the spec's ``execution.runner``, else the server's simulation when it overrides the default
    loop (CNCTWIN_ENTRY_SIMULATE), else cnctwin.sweep.runner. The server runs one simulation at a time: the points
    run one after another, each in its own process inside the job's memory cap.
    The sweep folder is the job's output folder; the dataset goes to ``<out_dir>/dataset`` (``options.dataset:
    false`` skips it)."""
    from ..core.errors import InputError
    from .spec import load_spec
    project_path = request.get("project_path")
    if not project_path:
        raise InputError("the sweep request names no project")
    spec = load_spec(dict(request.get("spec") or {}), base_override=project_path)
    options = dict(request.get("options") or {})
    out = Path(out_dir)
    dataset = None if options.get("dataset") is False else out / "dataset"
    runner = None
    named = (spec.data.get("execution") or {}).get("runner") if isinstance(spec.data.get("execution"), dict) else None
    simulate = request.get("simulate_entry")
    if not named and simulate and str(simulate).strip() != "cnctwin.core.loop:run_project":
        runner = str(simulate)               # the server's own simulation (CNCTWIN_ENTRY_SIMULATE)
    isolation = options.get("isolation") if options.get("isolation") in ("process", "inline") else None
    summary = run_sweep(spec, out, progress_cb=progress_cb, workers=1, runner=runner, isolation=isolation,
                        memory_mode="none",
                        no_cache=bool(options.get("no_cache")), retry_failed=bool(options.get("retry_failed")),
                        dataset_dir=dataset)
    summary["sweep_dir"] = str(out)
    return summary


def _progress_printer(quiet: bool, stream=None):
    stream = stream if stream is not None else sys.stderr
    state = {"last": 0.0}

    def progress(progress=None, **fields):
        if quiet:
            return
        now = time.monotonic()
        if now - state["last"] < 1.0 and fields.get("phase") not in ("dataset",):
            return
        state["last"] = now
        message = fields.get("message") or ""
        eta = (progress or {}).get("eta_s")
        stream.write(message + (f"  ETA {eta / 60:.1f} min" if isinstance(eta, (int, float)) else "") + "\n")
        stream.flush()
    return progress


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="twin sweep", description="run a parameter sweep through the run cache "
                                     "(docs/ml.md); finished points are reused, so running it again resumes it")
    parser.add_argument("spec", nargs="?", help="the sweep spec (YAML, format cnctwin-sweep)")
    parser.add_argument("--show", metavar="DIR", help="print what a sweep folder says now and exit")
    parser.add_argument("--out", help="the sweep folder (default DATA_DIR/sweeps/<name>)")
    parser.add_argument("--plan-only", action="store_true", help="write the points' variants, run nothing")
    parser.add_argument("--workers", type=int, help="points at once (default the spec's, 1)")
    parser.add_argument("--isolation", choices=("process", "inline"))
    parser.add_argument("--memory", help="memory cap of each point process (default 1536M)")
    parser.add_argument("--memory-mode", choices=("auto", "systemd", "rlimit", "none"))
    parser.add_argument("--runner", help="simulation entry (default cnctwin.sweep.runner:run_project)")
    parser.add_argument("--no-cache", action="store_true", help="run every point again (new runs are registered)")
    parser.add_argument("--retry-failed", action="store_true", help="run points that failed before with the same key")
    parser.add_argument("--dataset", nargs="?", const="", default=None, metavar="DIR",
                        help="export the dataset afterwards (default folder <sweep>/dataset)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    from ..core.errors import InputError, TwinError
    if args.show:
        from .execute import read_sweep
        try:
            print(json.dumps(read_sweep(args.show), indent=1, sort_keys=True, default=str))
        except InputError as error:
            print(f"twin sweep: error: {error}", file=sys.stderr)
            return 1
        return 0
    if not args.spec:
        parser.print_help()
        return 2
    try:
        from .spec import load_spec
        spec = load_spec(args.spec)
        out = Path(args.out) if args.out else default_out_dir(spec.name)
        dataset = None if args.dataset is None else (Path(args.dataset) if args.dataset else out / "dataset")
        summary = run_sweep(spec, out, progress_cb=_progress_printer(args.quiet), plan_only=args.plan_only,
                            workers=args.workers, isolation=args.isolation, memory=args.memory,
                            memory_mode=args.memory_mode, runner=args.runner,
                            retry_failed=True if args.retry_failed else None, no_cache=args.no_cache,
                            dataset_dir=dataset)
    except (InputError, TwinError, ValueError) as error:
        print(f"twin sweep: error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    elif args.plan_only:
        print(f"planned {summary['points']} points in {summary['out_dir']}")
    else:
        print(f"{summary['points']} points: {summary['cached']} cached, {summary['done']} simulated, "
              f"{summary['failed']} failed; {summary['safe']} safe, {summary['unsafe']} not safe; {summary['out_dir']}")
        if summary.get("dataset"):
            print(f"dataset: {summary['dataset']['out_dir']} ({summary['dataset']['runs']} runs, "
                  f"{summary['dataset']['blocks']} blocks)")
    return 3 if summary.get("failed") else 0
