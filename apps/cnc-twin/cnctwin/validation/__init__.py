"""Validation suite of CNC Twin: published experiments against the simulator (docs/validation.md).

``twin validate`` is :func:`main`. The suite loads the cases of validation/cases/ through cnctwin.db,
runs each through the harness registered for its quantity (cnctwin.validation.harnesses; a harness
calls the domain libraries directly at the level the experiment measured, never the run loop),
compares every reference value with the simulated one, applies the budgets of validation/budgets.yaml
and writes docs/validation-report.md and validation/results/<quantity>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..paths import DOCS, VALIDATION

REPORT_PATH = DOCS / "validation-report.md"
RESULTS_DIR = VALIDATION / "results"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="twin validate", description="Run the validation cases against their "
                                "published measurements and apply the error budgets.")
    p.add_argument("--quantity", action="append", default=[], help="only this quantity (repeatable)")
    p.add_argument("--case", action="append", default=[], help="only this case id (repeatable)")
    p.add_argument("--report", type=Path, default=None,
                   help=f"write the markdown report here (default with no filter: {REPORT_PATH.relative_to(DOCS.parent)})")
    p.add_argument("--results", type=Path, default=None,
                   help=f"write <quantity>.json here (default with no filter: {RESULTS_DIR.relative_to(VALIDATION.parent)})")
    p.add_argument("--no-write", action="store_true", help="write no file (print only)")
    p.add_argument("--json", action="store_true", help="print the results as JSON instead of the summary table")
    p.add_argument("--list", action="store_true", help="list the cases and their harnesses, run nothing")
    return p


def discussions(results) -> dict:
    from .registry import DISCUSSIONS
    out = {}
    for q in results:
        fn = DISCUSSIONS.get(q.quantity)
        if fn is not None and q.independent is not None:
            out[q.quantity] = fn(q)
    return out


def intros(results) -> dict:
    from .registry import get
    out = {}
    for q in results:
        h = get(q.quantity)
        if h is not None and h.description:
            out[q.quantity] = " ".join(h.description.split())
    return out


def write_outputs(results, report: Path | None, results_dir: Path | None) -> list[Path]:
    from . import report as R
    written = []
    if results_dir is not None:
        written += R.write_results(results, results_dir)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(R.markdown(results, discussions(results), intros(results)), encoding="utf-8")
        written.append(report)
    return written


def main(argv=None) -> int:
    """``twin validate``: exit 0 when no budgeted quantity fails, 1 when one fails or a case could not
    be simulated, 2 on a usage error."""
    args = _parser().parse_args(argv)
    from ..db import Database
    from . import report as R
    from .cases import load_cases
    from .registry import get, load_harnesses
    from .runner import FAIL, run

    db = Database.load()
    load_harnesses()
    if args.list:
        for case in load_cases(db, args.quantity or None, args.case or None):
            h = get(case.quantity)
            print(f"{case.quantity:18s} {case.id:60s} {h.function.__module__ if h else '(no harness)'}")
        return 0
    known = set(db.ids("validation_case"))
    unknown = [c for c in args.case if c not in known]
    if unknown:
        print(f"twin validate: unknown case id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    results = run(db, args.quantity or None, args.case or None)
    if not results:
        print("twin validate: no case selected", file=sys.stderr)
        return 2
    filtered = bool(args.quantity or args.case)
    report = args.report if args.report is not None else (None if filtered else REPORT_PATH)
    results_dir = args.results if args.results is not None else (None if args.case else RESULTS_DIR)
    if args.no_write:
        report = results_dir = None
    written = write_outputs(results, report, results_dir)
    if args.json:
        print(json.dumps([R.quantity_json(q) for q in results], indent=1, sort_keys=True))
    else:
        sys.stdout.write(R.summary_text(results))
        for path in written:
            print(f"wrote {path}")
    failed = any(q.verdict == FAIL for q in results) or any(r.status == "error" for q in results for r in q.cases)
    return 1 if failed else 0


__all__ = ["main", "REPORT_PATH", "RESULTS_DIR", "write_outputs", "discussions"]
