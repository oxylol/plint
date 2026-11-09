"""Calibration from machine data: the job entry point ``calibrate(request, out_dir, progress_cb)`` and the
command line ``python -m cnctwin.mdata.calibrate`` (docs/models/calibration.md "Running a calibration").

Steps:

1. resolve the project (``project_path``), import the machine data (``machine_data`` of the request,
   else the project's ``machine_data`` section) and read the simulated run (``run_dir``);
2. align the measured log with the run (options ``alignment``: ``{method: auto|markers|xcorr|given|none,
   offset_s, scale, key, poll_interval_s}``);
3. for each target (options ``targets``; default every target whose data are present, see
   cnctwin.mdata.targets) build the problem and fit it (cnctwin.mdata.fit). The simulator
   (options ``simulator``): ``surrogate`` (the analytic models applied to the nominal run),
   ``loop`` (every evaluation re-runs the simulation with the parameters through
   ``cnctwin.core.loop.run_project``; only for spindle, servo and block_time) or ``auto`` (the
   surrogate, then, when cnctwin.core.loop is in the build and the project is known, a refinement
   with the loop started from the surrogate's solution);
4. write into ``out_dir``: ``machine_log.json/.parquet`` (the imported data), ``alignment.json``,
   ``calibration.yaml`` (the overlay, cnctwin.mdata.overlay), ``calibration.json`` (the full report),
   ``overlay.json`` (measured vs simulated vs calibrated series and error metrics for the UI and
   the report) and ``project.calibrated.yaml`` (the project with the overlay applied) when a project
   was given and the overlay changes it.

Other options: ``target_options: {<target>: {free: [...], fixed: {name: value}, ...}}`` (see each
target), ``spindle`` (spindle id), ``name`` (overlay name), and the measurement tables
``tool_life``, ``wear``, ``part_measurements``, ``cuts`` (lists of rows, or CSV files relative to the
project folder).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from ..core.errors import InputError
from .align import Alignment, align, error_metrics, reduce_m4
from .fit import FitSettings, Param, fit
from .importers import load_machine_data
from .overlay import calibrated_project, make_overlay, write_overlay
from .params import align_params
from .simrun import SimulatedRun
from .targets import BUILDERS, LOOP_TARGETS, TARGETS, Context, Problem, Skip, spindle_signal

RESULT_FORMAT = "cnctwin-calibration-result"
RUN_PARAMS = {"spindle": ("kc_scale",), "block_time": ("block_processing_s",)}


def _progress(progress_cb):
    def report(**fields):
        if progress_cb is not None:
            progress_cb(**fields)
    return report


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loop_entry():
    """cnctwin.core.loop.run_project when the build has it, else None."""
    try:
        module = importlib.import_module("cnctwin.core.loop")
    except ModuleNotFoundError as error:
        if error.name in ("cnctwin.core.loop",):
            return None
        raise
    function = getattr(module, "run_project", None)
    return function if callable(function) else None


# ----------------------------------------------------------------------------- simulators

def surrogate_simulator(problem: Problem, ctx: Context):
    def simulate(values: dict) -> dict:
        return problem.observe(ctx.sim_log, values, True)
    return simulate


class LoopSimulator:
    """Every evaluation simulates the project with the run parameters applied (a calibrated project
    file in ``work_dir``), reads the run and applies the observation model. Runs are cached by the
    run parameters' values and deleted once read."""

    def __init__(self, problem: Problem, ctx: Context, run_project, work_dir: Path, project_path: Path,
                 progress=None, options: dict | None = None) -> None:
        self.problem = problem
        self.ctx = ctx
        self.run_project = run_project
        self.work_dir = Path(work_dir)
        self.project_path = Path(project_path)
        self.progress = progress
        self.options = dict(options or {})
        self.names = [n for n in RUN_PARAMS.get(problem.target, ()) if any(p.name == n for p in problem.params)]
        if problem.target == "servo":
            self.names = [p.name for p in problem.params]
        self.cache: dict = {}
        self.runs = 0

    def overlay_for(self, values: dict) -> dict:
        entries = {}
        for p in self.problem.params:
            if p.name in self.names:
                for target in p.targets:
                    entries[target.path] = {"value": target.file_value(values[p.name]), "parameter": p.name}
        return make_overlay(name=f"{self.problem.target} evaluation", machine=None, material=None, entries=entries,
                            parameters=[])

    def __call__(self, values: dict) -> dict:
        key = tuple(float(values[n]) for n in self.names)
        if key not in self.cache:
            self.runs += 1
            folder = self.work_dir / f"eval-{self.runs:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            project = calibrated_project(self.project_path, self.overlay_for(values), folder / "project.yaml",
                                         machine_raw=self.ctx.machine.raw if self.ctx.machine is not None else None,
                                         material_raw=self.ctx.material.raw if self.ctx.material is not None else None)
            run_dir = folder / "run"
            run_dir.mkdir(exist_ok=True)

            def forward(progress=None, **fields):
                if self.progress is not None:
                    self.progress(phase=f"{self.problem.target}: simulation {self.runs}",
                                  **{k: v for k, v in {**(progress or {}), **fields}.items() if k != "phase"})

            self.run_project(project, run_dir, dict(self.options), forward)
            self.cache[key] = SimulatedRun(run_dir).to_log()
            shutil.rmtree(folder, ignore_errors=True)
        return self.problem.observe(self.cache[key], values, False)


# ----------------------------------------------------------------------------- context

def _project(request: dict):
    path = request.get("project_path")
    if not path:
        return None, None
    from ..project import resolve
    path = Path(path)
    return resolve(path), path


def _machine_data_spec(request: dict, project) -> dict | None:
    if request.get("machine_data"):
        return dict(request["machine_data"])
    if project is not None:
        raw = getattr(project, "raw", None)
        spec = raw.get("machine_data") if isinstance(raw, dict) else None
        if spec:
            return dict(spec)
    return None


def build_context(request: dict, out_dir: Path, report) -> Context:
    options = dict(request.get("options") or {})
    report(phase="project", message="resolving the project")
    project, project_path = _project(request)
    base_dir = Path(request["project_dir"]) if request.get("project_dir") else \
        (project_path.parent if project_path is not None else (Path(options["base_dir"]) if options.get("base_dir")
                                                              else None))
    spec = _machine_data_spec(request, project)
    log = None
    if spec is not None:
        report(phase="import", message=f"reading {spec.get('kind')} machine data")
        log = load_machine_data(spec, base_dir)
        log.save(out_dir)
    run = sim_log = None
    if request.get("run_dir"):
        report(phase="run", message="reading the simulated run")
        run = SimulatedRun(request["run_dir"])
        sim_log = run.to_log()
    ctx = Context(project=project, log=log, sim_log=sim_log, run=run, options=options, base_dir=base_dir)
    if log is not None and sim_log is not None:
        report(phase="align", message="aligning the machine data with the run")
        settings = dict(options.get("alignment") or {})
        signal = spindle_signal(ctx)
        ctx.alignment = align(sim_log, log, run.markers(), method=settings.get("method", "auto"),
                              offset_s=settings.get("offset_s"), scale=settings.get("scale"), key=settings.get("key"),
                              poll_interval_s=settings.get("poll_interval_s"),
                              sim_signal=signal[0] if signal else None, measured_signal=signal[1] if signal else None)
    elif log is not None:
        ctx.alignment = Alignment(0.0, 1.0, "none", {"reason": "no simulated run"})
    return ctx


# ----------------------------------------------------------------------------- one target

def _restart(params: list[Param], values: dict) -> list[Param]:
    out = []
    for p in params:
        v = min(max(values.get(p.name, p.value), p.lower), p.upper)
        out.append(Param(p.name, v, p.lower, p.upper, p.unit, p.free, p.log, p.scale, p.targets, p.doc))
    return out


def run_target(problem: Problem, ctx: Context, simulator: str, *, out_dir: Path, project_path: Path | None,
               report, runner=None) -> dict:
    """Fit one problem; returns {problem, fit (surrogate), loop_fit (when the loop refined it), result}.
    ``runner``: the ``run_project`` of the loop simulator (default :func:`loop_entry`)."""
    kind = simulator
    entry = (runner if runner is not None else loop_entry()) if kind in ("auto", "loop") else None
    if kind == "loop" and entry is None:
        raise InputError("simulator loop: cnctwin.core.loop is not in this build")
    usable = problem.target in LOOP_TARGETS and problem.loop and project_path is not None and ctx.run is not None
    if kind == "loop" and not usable:
        kind = "surrogate"                 # no run-based prediction: the target's analytic model is its simulator
    report(phase=f"{problem.target}: fit", message=f"fitting {problem.target}")
    first = fit(problem.params, surrogate_simulator(problem, ctx), problem.observations,
                FitSettings.for_simulator("surrogate"), progress=report) if kind != "loop" else None
    final, simulator_used, loop_runs = first, "surrogate", 0
    if entry is not None and usable:
        start = _restart(problem.params, first.values) if first is not None else problem.params
        simulator = LoopSimulator(problem, ctx, entry, out_dir / "loop" / problem.target, project_path, report,
                                  ctx.options.get("loop_options"))
        final = fit(start, simulator, problem.observations, FitSettings.for_simulator("loop"), progress=report)
        simulator_used, loop_runs = ("surrogate + loop" if first is not None else "loop"), simulator.runs
        shutil.rmtree(out_dir / "loop" / problem.target, ignore_errors=True)
    return {"problem": problem, "surrogate": first, "fit": final, "simulator": simulator_used, "loop_runs": loop_runs}


def _written(record: dict) -> tuple[bool, str]:
    if not record["free"]:
        return False, "fixed"
    if not record.get("identifiable"):
        return False, "not identifiable from these data: the prior is kept"
    return True, "at a bound" if record.get("at_bound") else ""


def _summary(outcome: dict) -> dict:
    problem, result = outcome["problem"], outcome["fit"]
    params = []
    for record in result.params:
        written, why = _written(record)
        params.append({k: record.get(k) for k in ("name", "unit", "prior", "value", "ci", "std_error", "free",
                                                   "identifiable", "at_bound", "bounds", "log", "doc", "targets",
                                                   "ci_check")} | {"written": written, "remark": why})
    return {"target": problem.target, "status": "fitted", "simulator": outcome["simulator"],
            "loop_runs": outcome["loop_runs"], "params": params, "confounded": result.confounded,
            "metrics": result.metrics, "s2": result.s2, "dof": result.dof, "n_obs": result.n_obs,
            "cost_before": result.cost_before, "cost_after": result.cost_after, "nfev": result.nfev,
            "success": result.success, "message": result.message, "condition_number": result.condition_number,
            "correlation": {"names": result.names, "matrix": result.correlation}, "sigma": result.sigma,
            "notes": problem.notes, "inputs": _jsonable(problem.inputs)}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _overlay_series(outcome: dict) -> list[dict]:
    """Measured, simulated (prior) and calibrated series of each observation set, for the UI."""
    problem, result = outcome["problem"], outcome["fit"]
    max_points = int(align_params()["overlay_max_points"])
    out = []
    for obs in problem.observations:
        predictions = result.predictions[obs.name]
        t = obs.t_s if obs.t_s is not None else np.arange(len(obs.measured), dtype=np.float64)
        columns = {"measured": obs.measured, "simulated": predictions["prior"], "calibrated": predictions["fit"]}
        t_red, reduced, was_reduced = reduce_m4(np.asarray(t, dtype=np.float64), columns, max_points)
        out.append({"target": problem.target, "name": obs.name, "unit": obs.unit, "doc": obs.doc,
                    "x": "t_s" if obs.t_s is not None else "index", "t_s": t_red, **reduced,
                    "reduced": "m4" if was_reduced else "none",
                    "metrics": {"simulated": error_metrics(obs.measured, predictions["prior"]),
                                "calibrated": error_metrics(obs.measured, predictions["fit"])}})
    return out


# ----------------------------------------------------------------------------- the job

def _runner(request: dict, options: dict, simulator: str):
    """The loop simulator's ``run_project``: ``options.runner`` or the job's ``simulate_entry`` (the server's
    simulation) when named, else None (then :func:`loop_entry`). A named runner that is not in the build is an
    error for ``simulator: loop`` and leaves ``auto`` with the surrogate."""
    spec = options.get("runner") or request.get("simulate_entry")
    if not spec:
        return None
    from ..core.outputs import EntryMissing, resolve_entry
    try:
        return resolve_entry(spec)
    except EntryMissing as error:
        if simulator == "loop":
            raise InputError(f"simulator loop: {error}") from None
        return None


def calibrate(request: dict, out_dir, progress_cb=None, *, runner=None) -> dict:
    """The ``calibrate`` job (docs/api.md "Entry points"): request ``{project_id, project_path, project_dir,
    run_id, run_dir, machine_data, options}`` (the job manager adds ``simulate_entry``); returns the summary that
    becomes result.json. ``runner``: the ``run_project`` the loop simulator re-runs (default: ``options.runner``,
    the request's ``simulate_entry``, else cnctwin.core.loop when it is in the build)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _progress(progress_cb)
    ctx = build_context(request, out_dir, report)
    options = ctx.options
    requested = options.get("targets")
    if isinstance(requested, str):
        requested = [requested]
    names = list(requested) if requested else list(TARGETS)
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise InputError(f"options.targets: unknown {', '.join(unknown)} (targets: {', '.join(TARGETS)})")
    simulator = str(options.get("simulator", "auto"))
    if simulator not in ("auto", "surrogate", "loop"):
        raise InputError(f"options.simulator {simulator!r}: use auto, surrogate or loop")
    project_path = Path(request["project_path"]) if request.get("project_path") else None
    if runner is None and simulator in ("auto", "loop"):
        runner = _runner(request, options, simulator)
    outcomes, skipped = [], []
    for name in names:
        try:
            problem = BUILDERS[name](ctx)
        except Skip as reason:
            if requested:
                raise InputError(f"target {name}: {reason}") from None
            skipped.append({"target": name, "status": "skipped", "reason": str(reason)})
            continue
        outcome = run_target(problem, ctx, simulator, out_dir=out_dir, project_path=project_path, report=report,
                             runner=runner)
        outcomes.append(outcome)
        for record in outcome["fit"].params:
            if _written(record)[0]:
                ctx.calibrated[record["name"]] = record["value"]
    if not outcomes:
        reasons = "; ".join(f"{s['target']}: {s['reason']}" for s in skipped)
        raise InputError(f"nothing to calibrate from these data ({reasons})")
    report(phase="write", message="writing the overlay and the report")
    return write_outputs(ctx, outcomes, skipped, out_dir, request, project_path)


def write_outputs(ctx: Context, outcomes: list[dict], skipped: list[dict], out_dir: Path, request: dict,
                  project_path: Path | None) -> dict:
    entries, parameters = {}, []
    for outcome in outcomes:
        problem = outcome["problem"]
        for record in outcome["fit"].params:
            written, why = _written(record)
            parameters.append({"target": problem.target, "name": record["name"], "unit": record["unit"],
                               "prior": record["prior"], "value": record["value"], "ci": record.get("ci"),
                               "std_error": record.get("std_error"), "free": record["free"],
                               "identifiable": record.get("identifiable"), "at_bound": record.get("at_bound"),
                               "written": written, "remark": why})
            if not written:
                continue
            param = next(p for p in problem.params if p.name == record["name"])
            for target in param.targets:
                entry = {"value": target.file_value(record["value"]), "parameter": record["name"]}
                if record.get("ci"):
                    entry["ci"] = [target.file_value(v) for v in record["ci"]]
                entries[target.path] = entry
        if any(_written(r)[0] for r in outcome["fit"].params):
            for path, value in problem.constants.items():
                entries[path] = {"value": value, "parameter": f"{problem.target} (constant)"}
    machine_id = str(ctx.machine.id) if ctx.machine is not None and "id" in ctx.machine else None
    material_id = None
    if ctx.material is not None:
        material_id = str(ctx.material.db_id) if "db_id" in ctx.material and ctx.material.db_id else \
            (str(ctx.material.id) if "id" in ctx.material else None)
    log_source = ctx.log.source if ctx.log is not None else None
    inputs = {"machine_data": _jsonable(log_source),
              "run": {"run_id": request.get("run_id"), "run_dir": request.get("run_dir"),
                      "run_json_sha256": _sha256(Path(request["run_dir"]) / "run.json") if request.get("run_dir")
                      else None},
              "project": {"project_id": request.get("project_id"),
                          "sha256": _sha256(project_path) if project_path is not None else None}}
    name = str(ctx.options.get("name") or f"calibration of {', '.join(o['problem'].target for o in outcomes)}"
               + (f", {machine_id}" if machine_id else ""))
    fits = {o["problem"].target: {"cost_before": o["fit"].cost_before, "cost_after": o["fit"].cost_after,
                                  "n_obs": o["fit"].n_obs, "dof": o["fit"].dof, "s2": o["fit"].s2,
                                  "simulator": o["simulator"], "confounded": o["fit"].confounded,
                                  "success": o["fit"].success}
            for o in outcomes}
    alignment = ctx.alignment.to_dict() if ctx.alignment is not None else None
    if alignment is not None:
        alignment = {k: v for k, v in alignment.items() if k != "quality"} | \
            {"quality": {k: v for k, v in (alignment.get("quality") or {}).items() if k not in ("pairs", "windows")}}
    document = make_overlay(name=name, machine=machine_id, material=material_id, entries=entries,
                            parameters=_jsonable(parameters), fit=_jsonable(fits), inputs=_jsonable(inputs),
                            alignment=_jsonable(alignment))
    files = {"overlay": "calibration.yaml", "report": "calibration.json", "series": "overlay.json"}
    write_overlay(out_dir / "calibration.yaml", document)
    summaries = [_summary(o) for o in outcomes] + skipped
    report_doc = {"format": RESULT_FORMAT, "version": 1, "name": name, "machine": machine_id, "material": material_id,
                  "inputs": inputs, "alignment": ctx.alignment.to_dict() if ctx.alignment is not None else None,
                  "targets": summaries, "set": entries}
    (out_dir / "calibration.json").write_text(json.dumps(_jsonable(report_doc), indent=1, sort_keys=True) + "\n",
                                              encoding="utf-8")
    series = [s for o in outcomes for s in _overlay_series(o)]
    (out_dir / "overlay.json").write_text(json.dumps(_jsonable({
        "format": "cnctwin-calibration-overlay", "version": 1, "time_base": "simulation",
        "alignment": ctx.alignment.to_dict() if ctx.alignment is not None else None, "series": series}),
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if ctx.alignment is not None:
        (out_dir / "alignment.json").write_text(json.dumps(_jsonable(ctx.alignment.to_dict()), indent=1,
                                                           sort_keys=True) + "\n", encoding="utf-8")
        files["alignment"] = "alignment.json"
    if ctx.log is not None:
        files["machine_log"] = "machine_log.json"
    if project_path is not None and entries and any(p.split(".")[0].split("[")[0] in ("machine", "material", "tools")
                                                    for p in entries):
        calibrated_project(project_path, document, out_dir / "project.calibrated.yaml",
                           machine_raw=ctx.machine.raw if ctx.machine is not None else None,
                           material_raw=ctx.material.raw if ctx.material is not None else None)
        files["project"] = "project.calibrated.yaml"
    return _jsonable({"format": RESULT_FORMAT, "version": 1, "name": name, "out_dir": str(out_dir), "files": files,
                      "alignment": alignment, "set": entries,
                      "targets": [{k: s[k] for k in ("target", "status", "reason", "simulator", "params", "metrics",
                                                     "confounded", "notes") if k in s} for s in summaries]})


# ----------------------------------------------------------------------------- command line

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.mdata.calibrate",
                                     description="Calibrate the machine model from recorded machine data")
    parser.add_argument("--project", help="project.yaml (its machine, material and tools are calibrated)")
    parser.add_argument("--run", help="simulated run directory of the same program")
    parser.add_argument("--data", help="machine_data spec as a YAML/JSON file ({kind, path, mapping, start, end})")
    parser.add_argument("--kind", help="machine data kind (instead of --data): mtconnect, opcua, focas, csv, log")
    parser.add_argument("--path", help="machine data file, directory or URL (with --kind)")
    parser.add_argument("--map", action="append", default=[], metavar="KEY=VALUE", help="mapping entry (repeat)")
    parser.add_argument("--target", action="append", default=[], choices=TARGETS, help="target to fit (repeat)")
    parser.add_argument("--simulator", default="auto", choices=("auto", "surrogate", "loop"))
    parser.add_argument("--offset", type=float, help="given alignment offset t_sim - t_meas (s)")
    parser.add_argument("--options", help="further options as a YAML/JSON file (merged under the flags)")
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args(argv)
    try:
        options = {}
        if args.options:
            options = yaml.safe_load(Path(args.options).read_text(encoding="utf-8")) or {}
            if not isinstance(options, dict):
                raise InputError("--options must hold a mapping", args.options)
        if args.target:
            options["targets"] = args.target
        options["simulator"] = args.simulator
        if args.offset is not None:
            options.setdefault("alignment", {})["offset_s"] = args.offset
        machine_data = None
        if args.data:
            machine_data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
            if not isinstance(machine_data, dict):
                raise InputError("--data must hold a mapping {kind, path, mapping}", args.data)
            base = Path(args.data).resolve().parent
            if machine_data.get("path") and "://" not in str(machine_data["path"]) \
                    and not Path(machine_data["path"]).is_absolute() and not args.project:
                machine_data["path"] = str(base / machine_data["path"])
        elif args.kind:
            mapping = {}
            for entry in args.map:
                if "=" not in entry:
                    parser.error(f"--map {entry!r}: write KEY=VALUE")
                key, value = entry.split("=", 1)
                mapping[key.strip()] = value.strip()
            machine_data = {"kind": args.kind, "path": str(Path(args.path).resolve()) if args.path and "://" not in args.path
                            else args.path, "mapping": mapping}
        project = Path(args.project).resolve() if args.project else None
        request = {"project_path": str(project) if project else None,
                   "project_dir": str(project.parent) if project else str(Path.cwd()),
                   "run_dir": str(Path(args.run).resolve()) if args.run else None, "machine_data": machine_data,
                   "options": options}
        result = calibrate(request, Path(args.out))
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"files": result["files"], "set": result["set"],
                      "targets": [{"target": t["target"], "status": t["status"], "reason": t.get("reason")}
                                  for t in result["targets"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calibrate", "main", "build_context", "run_target", "LoopSimulator", "surrogate_simulator",
           "loop_entry", "write_outputs"]
