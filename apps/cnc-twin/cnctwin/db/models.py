"""Model coefficients: ``db/models/<model>.yaml`` (docs/architecture.md 3, docs/formats.md "Databases").

File format::

    format: cnctwin-model-params
    version: 1
    model: oxley                      # = file name
    description: "..."                # optional
    src: <bibkey>                     # optional section default for every parameter
    params:
      <name>: {value: x, unit: "mm", src: <bibkey|est>, note: "...", range: [lo, hi]}

``value`` is a number, a list (of numbers or rows), text or true/false. ``unit`` is the unit the
value is **written** in (a key unit token or a plain unit text: ``mm``, ``deg``, ``1/s``,
``W/(m K)``, ``pct``; ``"-"`` for dimensionless). The parameter **name carries the SI unit** of
the value :func:`params` returns (``arc_alarm_tolerance_m`` written ``{value: 0.02, unit: mm}``
reads as 2e-5); a name ending in a non-SI unit token is an error. ``range`` is in the written
unit and must contain the value. Every value needs a source like any database value (its own
``src`` or the file's; ``est`` needs a ``note``).

:func:`params` reads one model file (cached per process: model files do not change during a
run), validates it against the schema and the bibliography, and returns ``name -> SI value``
(numbers and lists of numbers converted, text and booleans as written). Problems raise
:class:`~cnctwin.core.errors.InputError` (:class:`~cnctwin.core.errors.SourceError` when all are
source problems) with file, line and key path. The ``cnctwin.db`` owner keeps this module.
"""

from __future__ import annotations

import functools
from pathlib import Path

from ..core.errors import InputError
from . import keyunits
from . import schemas as S
from .problems import Problem, raise_for
from .schema import Ctx, is_number_like
from .sources import Bibliography, check_sources
from .yamlio import load_file

try:
    from ..paths import DB as _DB_ROOT
except ImportError:  # pragma: no cover - paths.py is optional
    _DB_ROOT = Path(__file__).resolve().parents[2] / "db"

ROOT = Path(_DB_ROOT)


@functools.lru_cache(maxsize=None)
def _bibliography(root: str) -> Bibliography:
    bib = Bibliography()
    folder = Path(root) / "bibliography"
    for path in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
        try:
            doc = load_file(path)
        except InputError:
            continue
        entries = doc.data.get("entries") if isinstance(doc.data, dict) else None
        if isinstance(entries, dict):
            for key, entry in entries.items():
                if isinstance(entry, dict):
                    bib.add(str(key), dict(entry), str(path), doc.locations.line(("entries", key)))
    return bib


def check_model(data, locations, bibliography: Bibliography, *, file: str = "", name: str = "",
                strict: bool = False) -> list[Problem]:
    """Every problem of one model parameter file (schema, units, sources)."""
    ctx = Ctx(locations, obj=f"model:{name}", file=file)
    S.MODEL.check(data, (), None, ctx)
    if isinstance(data, dict):
        if name and data.get("model") != name:
            ctx.add(("model",), f"model {data.get('model')!r} must be the file name ({name!r})", "format")
        for key, entry in (data.get("params") or {}).items() if isinstance(data.get("params"), dict) else ():
            if not isinstance(entry, dict):
                continue
            _check_units(key, entry, ("params", key), ctx)
    problems = list(ctx.problems)
    if isinstance(data, dict):
        source_problems, _ = check_sources(data, locations, bibliography, obj=f"model:{name}", strict=strict)
        problems.extend(source_problems)
    return problems


def _check_units(key: str, entry: dict, path: tuple, ctx: Ctx) -> None:
    try:
        written = keyunits.parse_unit_text(entry.get("unit"))
    except ValueError as error:
        ctx.add(path + ("unit",), str(error), "type")
        return
    _, own = keyunits.parse_key(str(key))
    if own is None or own.scale:
        if written is not None and not written.scale and (written.factor != 1.0 or written.offset != 0.0) \
                and is_number_like(entry.get("value")):
            ctx.add(path, f"{key}: written in {written.token}, so the name must carry the SI unit "
                          f"(..._{written.si_token})", "rule")
        return
    if own.si_token != own.token:
        ctx.add(path, f"{key}: names carry the SI unit (params() returns SI): write {key[: -len(own.token)]}"
                      f"{own.si_token} with unit: {own.token}", "rule")
        return
    if written is not None and written.si_token != own.token and not written.scale:
        ctx.add(path + ("unit",), f"{key}: unit {entry.get('unit')!r} converts to {written.si_token}, not "
                                  f"{own.token}", "rule")


def _to_si(entry: dict):
    value = entry.get("value")
    unit = keyunits.parse_unit_text(entry.get("unit"))
    return keyunits.convert(value, unit) if is_number_like(value) else value


@functools.lru_cache(maxsize=None)
def _records(model: str, root: str) -> dict:
    path = Path(root) / "models" / f"{model}.yaml"
    if not path.is_file():
        raise InputError(f"no model parameter file for {model!r}", str(path))
    doc = load_file(path)
    problems = check_model(doc.data, doc.locations, _bibliography(root), file=str(path), name=model)
    raise_for(problems, f"model parameters {model!r}")
    return {name: dict(entry) for name, entry in (doc.data.get("params") or {}).items()}


def records(model: str, root: str | Path = ROOT) -> dict:
    """name -> {value, unit, src, note, range} as written (a copy)."""
    return {name: dict(entry) for name, entry in _records(model, str(root)).items()}


def params(model: str, root: str | Path = ROOT) -> dict:
    """name -> SI value of ``db/models/<model>.yaml`` (validated; see the module docstring)."""
    return {name: _to_si(entry) for name, entry in _records(model, str(root)).items()}


def units(model: str, root: str | Path = ROOT) -> dict:
    """name -> the unit text each parameter is written in."""
    return {name: entry.get("unit", "") for name, entry in _records(model, str(root)).items()}


def clear_cache() -> None:
    """Forget cached model files (tests that write model files)."""
    _records.cache_clear()
    _bibliography.cache_clear()


__all__ = ["params", "records", "units", "check_model", "clear_cache", "ROOT"]
