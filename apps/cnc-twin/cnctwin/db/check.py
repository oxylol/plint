"""``twin db check``: validate every file under db/ and validation/cases/ and print the problems.

Usage::

    python -m cnctwin.db.check [--root DIR] [--validation DIR] [--strict] [--estimates]
                               [--migrations] [--rules] [--json] [--quiet]

Prints every problem as ``file:line: key.path: message`` (warnings marked), then a summary.
``--estimates`` lists every estimated value (file, key path, value, note); ``--migrations``
every older key form moved forward at load; ``--rules`` the migration table. Exit status 1 if
any error (in ``--strict`` mode an ``est`` without a note is an error too), else 0.
"""

from __future__ import annotations

import argparse
import json
import sys

from .database import DEFAULT_ROOT, Database
from .migrate import rule_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twin db check", description="validate the CNC Twin databases")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="database root (default: %(default)s)")
    parser.add_argument("--validation", default=None, help="validation cases root (default: <root>/../validation/cases)")
    parser.add_argument("--strict", action="store_true", help="an est without a note is an error")
    parser.add_argument("--estimates", action="store_true", help="list every estimated value")
    parser.add_argument("--migrations", action="store_true", help="list every key migrated at load")
    parser.add_argument("--rules", action="store_true", help="print the migration table and exit")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quiet", action="store_true", help="errors and the summary only")
    args = parser.parse_args(argv)

    if args.rules:
        for row in rule_table():
            print(f"{row['id']:24s} {','.join(row['kinds']):20s} {row['at']:32s} {row['change']}  ({row['why']})")
        return 0

    db = Database.load(args.root, validation_root=args.validation, strict=args.strict, raise_on_error=False)
    problems = db.validate_all()
    errors = [p for p in problems if p.is_error]
    if args.json:
        out = {"summary": db.summary, "problems": [p.to_dict() for p in problems]}
        if args.estimates:
            out["estimates"] = [e.to_dict() for e in db.estimates()]
        if args.migrations:
            out["migrations"] = [m.to_dict() for m in db.migrations()]
        json.dump(out, sys.stdout, indent=1, default=str)
        sys.stdout.write("\n")
        return 1 if errors else 0
    for problem in problems:
        if args.quiet and not problem.is_error:
            continue
        print(problem)
    if args.migrations:
        print("\nmigrated at load (old form -> current form):")
        for migration in db.migrations():
            print(f"  {migration}")
    if args.estimates:
        print("\nestimated values (src: est):")
        for estimate in db.estimates():
            where = f"{estimate.file}:{estimate.line}" if estimate.line else estimate.file
            note = estimate.note or "(no note)"
            print(f"  {where}: {estimate.path} = {estimate.value!r}  -- {note}")
    summary = db.summary
    objects = ", ".join(f"{n} {k}" for k, n in summary["objects"].items())
    print(f"\n{summary['files']} files ({objects}); {summary['errors']} errors, {summary['warnings']} warnings, "
          f"{summary['estimates']} estimated values, {summary['migrations']} keys migrated")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
