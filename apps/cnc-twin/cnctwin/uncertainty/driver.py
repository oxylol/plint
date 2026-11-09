"""Uncertainty studies: plan the samples, write the project variants, run them, aggregate.

Layout of a study folder (``out_dir``)::

    plan.json                       settings, factors, every sample's draws and values, variant files
    variants/v<nnnn>/project.yaml   the variant (checked with cnctwin.project.load_project)
    variants/v<nnnn>/run/           its run directory (architecture 16)
    variants/v<nnnn>/result.json    what the run said (format cnctwin-uncertainty-variant; read back when the
                                    study is resumed; an older one without the header is read too)
    study.json                      the results (format cnctwin-uncertainty)
    uncertainty.html                the results as a page (the report includes the same section)

A variant whose ``result.json`` names the same project sha256 is not run again, so a cancelled
study continues where it stopped, deterministically. The runner is ``run_project(project_path,
run_dir, options, progress_cb)`` (docs/api.md "Entry points"): ``options["runner"]``, else the
``CNCTWIN_ENTRY_SIMULATE`` override of the API, else ``cnctwin.core.loop:run_project``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from ..core.errors import InputError, TwinError
from ..core.rng import stream
from ..project import load_project
from . import sampling
from .aggregate import aggregate
from .factors import factors_for, spreads
from .results import SampleResult, read_sample
from .variants import variant_data, write_variant

FORMAT = "cnctwin-uncertainty"
PLAN_FORMAT = "cnctwin-uncertainty-plan"
VARIANT_FORMAT = "cnctwin-uncertainty-variant"     # variants/v<n>/result.json: {format, version, project_sha256, result}
VERSION = 1
DEFAULT_RUNNER = "cnctwin.core.loop:run_project"


class RunnerMissing(TwinError):
    """The simulation the study needs is not in this build (or the named runner does not exist)."""


def resolve_runner(spec: str | None = None):
    spec = spec or os.environ.get("CNCTWIN_ENTRY_SIMULATE") or DEFAULT_RUNNER
    if ":" not in spec:
        raise RunnerMissing(f"runner {spec!r}: expected 'module:function' or '/path/file.py:function'")
    module_name, attr = spec.rsplit(":", 1)
    if module_name.endswith(".py"):
        path = Path(module_name)
        if not path.is_file():
            raise RunnerMissing(f"{spec} is not available: no file {path}")
        name = "cnctwin_runner_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        spec_obj = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            raise RunnerMissing(f"{module_name}.{attr} is not available in this build (no module "
                                f"{getattr(error, 'name', None) or module_name!r}): the study cannot run its variants") from None
    function = getattr(module, attr, None)
    if not callable(function):
        raise RunnerMissing(f"{module_name}.{attr} is not available in this build (no callable {attr!r})")
    return function


def _template_uncertainty() -> dict:
    from ..db.yamlio import load_file
    from ..project.loader import DS_TEMPLATE
    return dict((load_file(DS_TEMPLATE).data or {}).get("uncertainty") or {})


def settings_for(data: dict, options: dict | None) -> dict:
    """Runs, method and seed: the options, else the project's uncertainty section, else the template's."""
    options = dict(options or {})
    section = data.get("uncertainty") if isinstance(data.get("uncertainty"), dict) else {}
    template = _template_uncertainty()

    def pick(key):
        for source in (options, section, template):
            if source.get(key) is not None:
                return source[key]
        return None

    runs = int(pick("runs"))
    method = str(pick("method"))
    if method not in sampling.METHODS:
        raise InputError(f"uncertainty method {method!r}: use lhs or monte_carlo")
    if runs < 1:
        raise InputError("an uncertainty study needs at least one run")
    seed = int(options.get("seed", data.get("seed", 0)) or 0)
    return {"runs": runs, "method": method, "seed": seed, "vary_seed": bool(options.get("vary_seed", False)),
            "keep_runs": bool(options.get("keep_runs", True))}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_study(project_path, out_dir, *, options: dict | None = None, db=None) -> dict:
    """Draw the samples and write every variant project; returns (and writes) plan.json."""
    from ..db.models import params as model_params
    project = load_project(project_path)
    out = Path(out_dir)
    params = model_params("uncertainty")
    settings = settings_for(project.data, options)
    factors, skipped = factors_for(project.data, project.dir, params, options)
    vary, origin = spreads(project.data, options)
    n, method, seed = settings["runs"], settings["method"], settings["seed"]
    units, values = sampling.plan([f.name for f in factors], [f.distribution for f in factors], n, method, seed)
    if any(f.applies == "material" for f in factors) and db is None:
        from ..db import Database
        db = Database.load()
    seeds = stream(seed, "uncertainty.variant_seed").integers(0, 2 ** 31 - 1, size=n) if settings["vary_seed"] else None
    samples = []
    for i in range(n):
        sampled = {f.name: float(values[i, j]) for j, f in enumerate(factors)}
        data, written = variant_data(project, factors, sampled, db, sample=i,
                                     seed=int(seeds[i]) if seeds is not None else None)
        comment = (f"Uncertainty sample {i} of {n} ({method}, seed {seed}) of {project.path}\n"
                   + "\n".join(f"{name} = {value!r}" for name, value in sampled.items()))
        path = write_variant(out / "variants" / f"v{i:04d}", data, comment)
        samples.append({"index": i, "units": {f.name: float(units[i, j]) for j, f in enumerate(factors)},
                        "values": sampled, "project": path.relative_to(out).as_posix(), "project_sha256": _sha256(path),
                        "written": written, "seed": data.get("seed"),
                        "options_factors": {f.name: sampled[f.name] for f in factors if f.applies == "options"}})
    plan = {"format": PLAN_FORMAT, "version": VERSION,
            "project": {"path": str(project.path), "name": project.data.get("name"), "sha256": project.sha256},
            "settings": settings, "vary": {k: vary[k] for k in sorted(vary, key=str)},
            "vary_origin": {k: origin[k] for k in sorted(origin, key=str)},
            "factors": [f.to_dict() for f in factors], "not_varied": skipped,
            "options_only": [f.name for f in factors if f.applies == "options"], "samples": samples}
    _write_json(out / "plan.json", plan)
    return plan


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(_clean(document), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def execute(plan: dict, out_dir, runner, *, progress_cb=None, simulate_options: dict | None = None) -> list[SampleResult]:
    """Run every variant (reusing results of unchanged variants); a failing variant is recorded, not fatal."""
    out = Path(out_dir)
    total = len(plan["samples"])
    results = []
    for position, sample in enumerate(plan["samples"]):
        folder = out / "variants" / f"v{int(sample['index']):04d}"
        cached = _cached(folder, sample)
        if cached is not None:
            results.append(cached)
            continue
        run_dir = folder / "run"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        options = dict(simulate_options or {})
        options["uncertainty"] = {"sample": int(sample["index"]), "samples": total, "method": plan["settings"]["method"],
                                  "seed": plan["settings"]["seed"], "factors": dict(sample["values"]),
                                  "options_only": dict(sample["options_factors"])}

        def inner(progress=None, _position=position, **fields):
            if progress_cb is None:
                return
            merged = dict(progress or {})
            merged.update(fields)
            progress_cb({k: v for k, v in merged.items() if k in ("sim_time_s", "est_total_s", "eta_s", "op")},
                        phase=f"variant {_position + 1}/{total}", done=_position, total=total,
                        message=f"uncertainty sample {_position + 1} of {total}")

        if progress_cb is not None:
            progress_cb(phase=f"variant {position + 1}/{total}", done=position, total=total,
                        message=f"uncertainty sample {position + 1} of {total}")
        try:
            runner(out / sample["project"], run_dir, options, inner)
        except Exception as error:      # a variant that fails is a result of the study, not its end
            result = SampleResult(int(sample["index"]), False, f"{type(error).__name__}: {error}", str(run_dir))
        else:
            result = read_sample(run_dir, int(sample["index"]))
        _write_json(folder / "result.json", {"format": VARIANT_FORMAT, "version": VERSION,
                                              "project_sha256": sample["project_sha256"], "result": result.to_dict()})
        if not plan["settings"].get("keep_runs", True) and result.ok:
            shutil.rmtree(run_dir)
        results.append(result)
    if progress_cb is not None:
        progress_cb(phase="aggregating", done=total, total=total, message="aggregating the samples")
    return results


def _cached(folder: Path, sample: dict) -> SampleResult | None:
    path = folder / "result.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("project_sha256") != sample["project_sha256"]:
        return None
    if record.get("format", VARIANT_FORMAT) != VARIANT_FORMAT or int(record.get("version") or VERSION) > VERSION:
        return None                     # not a variant result this build reads: the variant runs again
    return SampleResult.from_dict(record["result"])


def assemble(plan: dict, results: list[SampleResult], nominal: SampleResult | None = None) -> dict:
    from ..db.models import params as model_params
    params = model_params("uncertainty")
    return _clean({"format": FORMAT, "version": VERSION, "project": plan["project"], "settings": plan["settings"],
                   "parameters": {k: params[k] for k in sorted(params)}, "factors": plan["factors"],
                   "not_varied": plan["not_varied"], "options_only": plan["options_only"], "vary": plan["vary"],
                   "vary_origin": plan["vary_origin"], "samples": plan["samples"],
                   "results": aggregate(plan, results, params, nominal),
                   "nominal": nominal.to_dict() if nominal is not None else None,
                   "sample_results": [r.to_dict() for r in results]})


def study(project_path, out_dir, *, options: dict | None = None, runner=None, progress_cb=None, base_run_dir=None,
          db=None) -> dict:
    """Plan, run and aggregate a study; writes study.json and uncertainty.html into ``out_dir`` (and, with
    a base run, a copy into ``<run>/uncertainty/`` so the run's report shows it)."""
    options = dict(options or {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = plan_study(project_path, out, options=options, db=db)
    if options.get("plan_only"):
        return plan
    runner = runner or resolve_runner(options.get("runner"))
    results = execute(plan, out, runner, progress_cb=progress_cb, simulate_options=options.get("simulate_options"))
    nominal = None
    if base_run_dir is not None and (Path(base_run_dir) / "run.json").is_file():
        nominal = read_sample(base_run_dir, -1)
    document = assemble(plan, results, nominal)
    document["base_run"] = str(base_run_dir) if base_run_dir is not None else None
    write_outputs(document, out)
    if base_run_dir is not None and options.get("attach", True) and Path(base_run_dir).is_dir():
        write_outputs(document, Path(base_run_dir) / "uncertainty")
    return document


def write_outputs(document: dict, folder: Path) -> None:
    from .html import study_page
    _write_json(Path(folder) / "study.json", document)
    page = Path(folder) / "uncertainty.html"
    tmp = page.with_name(".uncertainty.html.tmp")
    tmp.write_text(study_page(document), encoding="utf-8")
    tmp.replace(page)


def run_study(request: dict, out_dir, progress_cb=None) -> dict:
    """API job entry (docs/api.md): ``request`` {run_id, run_dir, project_id, project_path, project_dir, options};
    the job manager adds ``simulate_entry`` (the server's simulation: the runner when the options name none) and
    ``outputs_entry`` (with it the run's report is made again with the study's section). With a run, the study
    is attached to it (cnctwin.core.outputs.attach_study: uncertainty/study.json, uncertainty.json, the output
    index of run.json)."""
    project_path = request.get("project_path")
    if not project_path:
        raise InputError("the uncertainty request names no project_path (the run was not started by this server)")
    options = dict(request.get("options") or {})
    runner = resolve_runner(request["simulate_entry"]) if request.get("simulate_entry") and not options.get("runner") \
        else None
    document = study(project_path, out_dir, options=options, runner=runner, progress_cb=progress_cb,
                     base_run_dir=request.get("run_dir"))
    run_dir = request.get("run_dir")
    attached = None
    if run_dir and options.get("attach", True) and not options.get("plan_only") and (Path(run_dir) / "run.json").is_file():
        from ..core.outputs import attach_study
        outputs_entry = request.get("outputs_entry")
        attached = attach_study(run_dir, document, progress_cb=progress_cb, outputs_entry=outputs_entry,
                                refresh=bool(outputs_entry) and str(outputs_entry).strip().lower() not in ("none", "off"))
    results = document.get("results") or {}
    report = (attached or {}).get("report")
    return {"format": document.get("format"), "study": str(Path(out_dir) / "study.json"),
            "page": str(Path(out_dir) / "uncertainty.html"), "samples_ok": results.get("samples_ok"),
            "samples_failed": results.get("samples_failed"), "verdict": results.get("verdict"),
            "factors": [f["name"] for f in document.get("factors", [])],
            "attached": attached["files"] if attached else None,
            "report": {"failed": report.get("failed"), "files": report.get("files")} if report else None}


__all__ = ["study", "plan_study", "execute", "assemble", "run_study", "resolve_runner", "settings_for",
           "write_outputs", "RunnerMissing", "FORMAT", "DEFAULT_RUNNER"]
