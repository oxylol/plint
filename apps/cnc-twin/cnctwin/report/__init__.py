"""Reports of a run, built from its run directory only (docs/report.md).

* :func:`build_report` (``run_dir, out_dir=None, *, formats=("html", "pdf"), setup_sheet=True,
  study=None``) -> dict: writes ``report.html``/``report.pdf`` and ``setup_sheet.html``/``.pdf``;
* :func:`report_html` (``run_dir, study=None``) -> (text, RunData) and :func:`setup_sheet_html`;
* :class:`RunData`: the tolerant reader of a run directory (what is absent is listed, not fatal);
* :mod:`cnctwin.report.compare`: the comparison of two runs (``compare_runs`` for the API job,
  ``main(argv)`` for the command line);
* :func:`main` (``argv=None``) -> exit code: ``python -m cnctwin.report RUN_DIR [--out DIR]
  [--format html,pdf] [--no-setup-sheet] [--study study.json] [--json]``.

Importing this package is light; Matplotlib and WeasyPrint load when a report is built.
"""

from __future__ import annotations

import argparse
import json
import sys

__all__ = ["build_report", "report_html", "setup_sheet_html", "RunData", "main"]


def __getattr__(name: str):
    if name in ("build_report", "report_html"):
        from . import report
        return getattr(report, name)
    if name == "setup_sheet_html":
        from .setup_sheet import setup_sheet_html
        return setup_sheet_html
    if name == "RunData":
        from .rundir import RunData
        return RunData
    raise AttributeError(f"module 'cnctwin.report' has no attribute {name!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.report",
                                     description="Write report.html/.pdf and setup_sheet.html/.pdf of a run directory.")
    parser.add_argument("run_dir", help="the run directory (runs/<run_id>)")
    parser.add_argument("--out", default=None, help="where to write (default: the run directory)")
    parser.add_argument("--format", default="html,pdf", help="html, pdf or html,pdf (default)")
    parser.add_argument("--no-setup-sheet", action="store_true", help="only the report")
    parser.add_argument("--study", default=None, help="an uncertainty study.json to include "
                                                      "(default: <run_dir>/uncertainty/study.json when present)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    from .report import build_report
    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    try:
        result = build_report(args.run_dir, args.out, formats=formats, setup_sheet=not args.no_setup_sheet,
                              study=args.study)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=1, sort_keys=True))
    else:
        verdict = "safe to run" if result["verdict"].get("safe") else "NOT SAFE TO RUN"
        print(f"{verdict}; wrote " + ", ".join(result["files"]))
        if result["missing"]:
            print(f"{len(result['missing'])} items not in the run directory (listed at the end of the report)")
    return 0
