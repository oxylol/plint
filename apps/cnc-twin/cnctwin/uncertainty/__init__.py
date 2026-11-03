"""Uncertainty studies over a project's stated tolerances (docs/models/uncertainty.md).

* :func:`study` (``project_path, out_dir, *, options=None, runner=None, progress_cb=None,
  base_run_dir=None, db=None``) -> dict: plan (Latin hypercube or Monte Carlo, seeded with
  ``cnctwin.core.rng.stream``), write one project variant per sample, run each through the runner
  (``run_project(project_path, run_dir, options, progress_cb)``; default
  ``cnctwin.core.loop:run_project``), aggregate: distributions and Cp/Cpk of every inspected
  dimension, the probability of every finding category and location, the sensitivity ranking
  (rank correlation); writes study.json and uncertainty.html;
* :func:`run_study` (``request, out_dir, progress_cb``) -> dict: the API job entry (docs/api.md);
* :func:`plan_study`, :func:`aggregate`, :func:`html_section` (the report's section), :func:`study_page`;
* :func:`main` (``argv=None``) -> exit code: ``python -m cnctwin.uncertainty PROJECT --out DIR [--run
  RUN_DIR] [--runs N] [--method lhs|monte_carlo] [--seed S] [--runner module:function] [--plan-only]
  [--json]``.
"""

from __future__ import annotations

import argparse
import json
import sys

__all__ = ["study", "run_study", "plan_study", "aggregate", "html_section", "study_page", "resolve_runner", "main"]


def __getattr__(name: str):
    if name in ("study", "run_study", "plan_study", "resolve_runner"):
        from . import driver
        return getattr(driver, name)
    if name == "aggregate":
        from .aggregate import aggregate
        return aggregate
    if name in ("html_section", "study_page"):
        from . import html
        return getattr(html, name)
    raise AttributeError(f"module 'cnctwin.uncertainty' has no attribute {name!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.uncertainty",
                                     description="Uncertainty study of a project over its stated tolerances.")
    parser.add_argument("project", help="project.yaml (or its folder)")
    parser.add_argument("--out", required=True, help="study folder (plan.json, variants/, study.json, uncertainty.html)")
    parser.add_argument("--run", default=None, help="the run the study is made around (its report gets the section)")
    parser.add_argument("--runs", type=int, default=None, help="number of samples (default: the project's)")
    parser.add_argument("--method", choices=("lhs", "monte_carlo"), default=None)
    parser.add_argument("--seed", type=int, default=None, help="default: the project's seed")
    parser.add_argument("--runner", default=None, help="module:function or file.py:function that runs a project")
    parser.add_argument("--vary-seed", action="store_true", help="a different simulation seed in every sample")
    parser.add_argument("--plan-only", action="store_true", help="write the plan and the variants, run nothing")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    options = {k: v for k, v in (("runs", args.runs), ("method", args.method), ("seed", args.seed),
                                 ("runner", args.runner)) if v is not None}
    if args.vary_seed:
        options["vary_seed"] = True
    if args.plan_only:
        options["plan_only"] = True
    from ..core.errors import TwinError
    from .driver import study

    def progress(progress=None, **fields):
        if fields.get("message"):
            print(fields["message"], file=sys.stderr)

    try:
        document = study(args.project, args.out, options=options, progress_cb=progress, base_run_dir=args.run)
    except TwinError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(document, indent=1, sort_keys=True))
    elif args.plan_only:
        print(f"planned {len(document['samples'])} samples of {len(document['factors'])} factors in {args.out}")
    else:
        results = document["results"]
        unsafe = (results.get("verdict") or {}).get("unsafe") or {}
        print(f"{results['samples_ok']} samples ok, {results['samples_failed']} failed; "
              f"P(not safe to run) = {unsafe.get('p')}; wrote {args.out}/study.json and uncertainty.html")
    return 0
