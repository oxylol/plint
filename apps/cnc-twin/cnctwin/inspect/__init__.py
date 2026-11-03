"""Result geometry and inspection (docs/models/inspect.md).

The finished geometry of every operation (:mod:`~cnctwin.inspect.result`: material configuration and the
configurations displaced by structure's grids), compared with the CAD (:mod:`~cnctwin.inspect.cadref`):

* :mod:`~cnctwin.inspect.deviation` signed deviation map with per-face statistics;
* :mod:`~cnctwin.inspect.columns` remaining material and gouges as connected regions (dexel field vs target);
* :mod:`~cnctwin.inspect.chordal` chord errors of the linearised toolpath;
* :mod:`~cnctwin.inspect.features` features from B-rep faces (planes ... tori, slots, pockets) and their measured
  size / location / orientation; :mod:`~cnctwin.inspect.fitting` least-squares and minimum-zone fits;
* :mod:`~cnctwin.inspect.datums` datum feature simulators and datum reference frames;
  :mod:`~cnctwin.inspect.gdt` every characteristic kind; :mod:`~cnctwin.inspect.fits286` ISO 286 limits;
  :mod:`~cnctwin.inspect.threads` ISO 965-1 / ASME B1.1 thread classes and gauging;
* :mod:`~cnctwin.inspect.cmm` the virtual CMM (probe plans, probing, compensation, on-demand probes);
* :mod:`~cnctwin.inspect.stock` the next operation's stock (transfer + mislocation);
* :mod:`~cnctwin.inspect.inspector` one operation's inspection and inspection.json;
  :mod:`~cnctwin.inspect.domain` the Domain object; :mod:`~cnctwin.inspect.protocols` what it needs from others.

Submodules load on first use (``from cnctwin.inspect import InspectDomain, ResultPart`` works).
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ResultPart": "result", "DisplacementGrid": "result", "CadReference": "cadref", "cad_from_mesh": "cadref",
    "Inspector": "inspector", "document": "inspector", "InspectDomain": "domain", "InspectSettings": "schema",
    "settings": "schema", "VirtualCMM": "cmm", "next_stock": "stock", "NextStock": "stock", "Columns": "abi",
    "assign": "sampling", "deviation_map": "deviation", "evaluate": "gdt", "recognise": "features",
}
_SUBMODULES = ("abi", "cadref", "channels", "chordal", "cmm", "columns", "datums", "deviation", "domain", "features",
               "fits286", "fitting", "gdt", "inspector", "kernels", "protocols", "reference", "result", "sampling",
               "schema", "specs", "state", "stock", "threads")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(set(_EXPORTS))
