"""One entry point for every machine data source: ``load_machine_data(spec, base_dir)``.

``spec`` is the project's ``machine_data`` section (or the API's ``machine_data`` object):
``{kind: mtconnect | opcua | focas | csv | log, path: <file, directory or URL>, mapping: {...},
start: <ISO 8601>, end: <ISO 8601>}``. Paths are relative to ``base_dir`` (the project folder);
``path`` may be an MTConnect agent URL (``http://``) or an OPC UA server (``opc.tcp://``). ``log``
reads a saved MachineLog (``machine_log.json``). ``paths: [...]`` (several files of one kind) are
read and merged on one time base.

``python -m cnctwin.mdata.importers --kind K --path P [--map channel=source ...] [--start T]
[--end T] [--out DIR]`` converts a source into a saved MachineLog and prints its summary;
``--browse URL`` lists the variables of an OPC UA server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.errors import InputError
from .log import MachineLog, merge, parse_time_ns

KINDS = ("mtconnect", "opcua", "focas", "csv", "log")


def _time(spec: dict, key: str) -> int | None:
    value = spec.get(key)
    if value in (None, ""):
        return None
    return parse_time_ns(str(value))[0]


def load_machine_data(spec: dict, base_dir=None) -> MachineLog:
    """The MachineLog of one ``machine_data`` spec (see the module docstring)."""
    if not isinstance(spec, dict):
        raise InputError("machine_data must be a mapping {kind, path, mapping}")
    kind = str(spec.get("kind") or "").lower()
    if kind not in KINDS:
        raise InputError(f"machine_data.kind {spec.get('kind')!r}: use one of {', '.join(KINDS)}")
    paths = spec.get("paths") or ([spec["path"]] if spec.get("path") else [])
    if not paths:
        raise InputError("machine_data needs path (a file, a directory or a URL)")
    mapping = dict(spec.get("mapping") or {})
    start, end = _time(spec, "start"), _time(spec, "end")
    logs = [_load_one(kind, str(p), base_dir, mapping, start, end) for p in paths]
    log = merge(logs)
    log.source = dict(log.source) if len(logs) == 1 else log.source
    log.source.setdefault("kind", kind)
    return log


def _load_one(kind: str, source: str, base_dir, mapping: dict, start: int | None, end: int | None) -> MachineLog:
    if kind == "mtconnect":
        from .mtconnect import read_mtconnect
        return read_mtconnect(source, base_dir=base_dir, mapping=mapping, start_ns=start, end_ns=end)
    if kind == "opcua":
        from .opcua import read_opcua
        return read_opcua(source, base_dir=base_dir, mapping=mapping, start_ns=start, end_ns=end)
    if kind == "focas":
        from .focas import read_focas
        return read_focas(source, base_dir=base_dir, mapping=mapping, start_ns=start, end_ns=end)
    if kind == "csv":
        from .csvlog import read_csv_log
        return read_csv_log(source, mapping, base_dir=base_dir, start_ns=start, end_ns=end)
    path = Path(source)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    return MachineLog.load(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.mdata.importers",
                                     description="Convert machine data (MTConnect, OPC UA, FOCAS logs, CSV) into a "
                                                 "MachineLog")
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--path", help="file, directory or URL")
    parser.add_argument("--map", action="append", default=[], metavar="KEY=VALUE",
                        help="mapping entry, e.g. spindle.load_frac='Load [pct]' (repeat)")
    parser.add_argument("--start", help="ISO 8601 start of the window")
    parser.add_argument("--end", help="ISO 8601 end of the window")
    parser.add_argument("--out", help="directory to save machine_log.json/.parquet in")
    parser.add_argument("--browse", metavar="URL", help="list the variables of an OPC UA server and exit")
    parser.add_argument("--root", default="i=85", help="browse start node (default: Objects)")
    args = parser.parse_args(argv)
    try:
        if args.browse:
            from .opcua import browse
            print(json.dumps(browse(args.browse, args.root), indent=1, default=str))
            return 0
        if not args.kind or not args.path:
            parser.error("--kind and --path are required (or --browse URL)")
        mapping = {}
        for entry in args.map:
            if "=" not in entry:
                parser.error(f"--map {entry!r}: write KEY=VALUE")
            key, value = entry.split("=", 1)
            mapping[key.strip()] = value.strip()
        spec = {"kind": args.kind, "path": args.path, "mapping": mapping, "start": args.start, "end": args.end}
        log = load_machine_data(spec, Path.cwd())
        if args.out:
            log.save(args.out)
        print(json.dumps(log.summary(), indent=1))
        return 0
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_machine_data", "KINDS", "main"]
