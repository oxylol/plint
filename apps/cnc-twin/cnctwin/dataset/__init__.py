"""Training data from finished runs (docs/ml.md "Datasets"): a run-level metrics table and a per-block
feature/label table, each as Parquet and CSV, with every column's unit and provenance in schema.json.

* :func:`export` (``sources, out_dir, *, formats=("parquet", "csv"), blocks=True, progress_cb=None``) -> manifest:
  ``sources`` are sweep folders (sweep.json: every finished point, with its parameters and cache key), run
  directories (run.json) or folders of run directories; writes ``runs.parquet``/``runs.csv``,
  ``blocks.parquet``/``blocks.csv``, ``schema.json`` (cnctwin.dataset.schema) and ``manifest.json`` (the runs
  exported, the points skipped and why, the build); deterministic: the same runs give the same bytes.
* :func:`main` (``argv=None``): ``twin dataset SOURCE... --out DIR [--format parquet,csv] [--no-blocks] [--json]``.

The rows are reductions of each run's own files (cnctwin.dataset.extract, cnctwin.dataset.series); nothing is
simulated again. A model trained on them inherits the twin's errors (docs/ml.md "Limits").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MANIFEST_FORMAT = "cnctwin-dataset"
VERSION = 1

__all__ = ["export", "collect_sources", "main"]


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_sources(paths) -> tuple[list, list[dict], list[dict]]:
    """(Sources in order, the sweep parameter descriptions in first-seen order, skipped [{source, reason}])."""
    from ..core.errors import InputError
    from .extract import Source
    sources, parameters, skipped = [], [], []
    seen_params = set()
    for raw in paths:
        path = Path(raw).resolve()
        if (path / "sweep.json").is_file():
            sweep = _read(path / "sweep.json") or {}
            plan = _read(path / "plan.json") or {}
            name = str(sweep.get("name") or path.name)
            for p in plan.get("parameters") or []:
                if p.get("name") not in seen_params:
                    seen_params.add(p.get("name"))
                    parameters.append(p)
            for entry in plan.get("points") or []:
                index = int(entry["index"])
                result = _read(path / "points" / f"p{index:05d}" / "result.json") or {}
                run_dir = result.get("run_dir")
                if result.get("status") not in ("done", "cached"):
                    skipped.append({"source": str(path), "point": index,
                                    "reason": f"status {result.get('status', 'not run')}"
                                              + (f": {(result.get('error') or {}).get('message')}" if result.get("error") else "")})
                    continue
                if not run_dir or not (Path(run_dir) / "run.json").is_file():
                    skipped.append({"source": str(path), "point": index,
                                    "reason": f"its run {run_dir} is gone (evicted from the cache? run the sweep again)"})
                    continue
                sources.append(Source(Path(run_dir), name, index, str(result.get("key") or ""),
                                      dict(entry.get("values") or {})))
        elif (path / "run.json").is_file():
            sources.append(Source(path))
        elif path.is_dir():
            runs = sorted(p for p in path.iterdir() if (p / "run.json").is_file())
            if not runs:
                raise InputError(f"{path} is neither a sweep folder, a run directory nor a folder of runs")
            sources.extend(Source(p) for p in runs)
        else:
            raise InputError(f"no sweep folder or run directory {path}")
    return sources, parameters, skipped


def _tool_order(label: str) -> tuple:
    number, _, path = label[1:].partition("@")
    return (int(path or 1), int(number) if number.isdigit() else 0, label)


def export(sources, out_dir, *, formats=("parquet", "csv"), blocks: bool = True, progress_cb=None) -> dict:
    """Write the dataset of ``sources`` into ``out_dir`` (module docstring); returns manifest.json's document."""
    from .. import __version__
    from ..core.findings import Category
    from . import schema as S
    from .extract import extract
    from .write import write_json, write_table
    formats = tuple(formats)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    collected, parameters, skipped = collect_sources(sources)
    run_rows, block_rows, used = [], [], []
    tools, characteristics, features, categories = set(), [], [], set()
    for i, source in enumerate(collected):
        if progress_cb is not None:
            progress_cb({"done": i, "total": len(collected)}, phase="dataset",
                        message=f"run {i + 1} of {len(collected)}: {source.run_dir.name}")
        extracted = extract(source, blocks=blocks)
        run_rows.append(extracted.run)
        block_rows.extend(extracted.blocks)
        tools.update(extracted.tools)
        characteristics.extend(c for c in extracted.characteristics if c not in characteristics)
        features.extend(f for f in extracted.features if f not in features)
        categories.update(k[len("findings["):-1] for k in extracted.run if k.startswith("findings[")
                          and k[len("findings["):-1] not in S.SEVERITIES)
        document = _read(source.run_dir / "run.json") or {}
        used.append({"run_dir": str(source.run_dir), "run_id": extracted.run["run_id"], "sweep": source.sweep,
                     "point": source.point, "cache_key": source.key,
                     "build": (document.get("versions") or {}).get("build")})
    known = [c.value for c in Category]
    category_list = known + sorted(c for c in categories if c not in known)
    run_columns = (S.RUN_IDENTITY + S.parameter_columns(parameters) + S.RUN_OPTIONS + S.RUN_RESULTS + S.RUN_SEVERITY
                   + S.category_columns(category_list) + S.tool_columns(sorted(tools, key=_tool_order))
                   + S.inspection_columns(characteristics) + S.surface_columns(features))
    files = write_table(run_rows, run_columns, out, "runs", formats)
    tables = {"runs": run_columns}
    if blocks:
        files += write_table(block_rows, S.BLOCK_COLUMNS, out, "blocks", formats)
        tables["blocks"] = S.BLOCK_COLUMNS
    write_json(out / "schema.json", S.schema_document(tables))
    files.append("schema.json")
    manifest = {"format": MANIFEST_FORMAT, "version": VERSION, "cnctwin": __version__, "formats": list(formats),
                "runs": len(run_rows), "blocks": len(block_rows) if blocks else None,
                "files": sorted(files + ["manifest.json"]), "sources": used, "skipped": skipped,
                "builds": sorted({u["build"] for u in used if u["build"]}),
                "notes": ["a model trained on these rows inherits the twin's errors (docs/ml.md \"Limits\")"]}
    if len(manifest["builds"]) > 1:
        manifest["notes"].append("the runs come from several builds: their models may differ")
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="twin dataset", description="export training data (a run-level and a "
                                     "per-block table, Parquet and CSV, with schema.json) from sweeps and runs")
    parser.add_argument("sources", nargs="+", help="sweep folders, run directories or folders of run directories")
    parser.add_argument("--out", required=True, help="the dataset folder")
    parser.add_argument("--format", default="parquet,csv", help="parquet, csv or both (default parquet,csv)")
    parser.add_argument("--no-blocks", action="store_true", help="only the run-level table")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args(argv)
    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    if not formats or any(f not in ("parquet", "csv") for f in formats):
        print("twin dataset: --format takes parquet, csv or parquet,csv", file=sys.stderr)
        return 2
    from ..core.errors import InputError
    try:
        manifest = export(args.sources, args.out, formats=formats, blocks=not args.no_blocks)
    except (InputError, OSError) as error:
        print(f"twin dataset: error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(manifest, indent=1, sort_keys=True))
    else:
        print(f"{manifest['runs']} runs" + (f", {manifest['blocks']} blocks" if manifest["blocks"] is not None else "")
              + f" -> {args.out}" + (f" ({len(manifest['skipped'])} skipped)" if manifest["skipped"] else ""))
    return 0
