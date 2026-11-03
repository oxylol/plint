"""Cutting domain: the coefficient pipeline, the surface memory and the force kernels of every process
(docs/models/cutting.md).

* :mod:`.kernels` - the compiled L0 kernels: :func:`~cnctwin.cutting.kernels.cut_step` (rotating tools),
  :func:`~cnctwin.cutting.kernels.turn_step` (turning on the polar field), deflection from the modal state;
* :mod:`.coefficients` - the L2 coefficient pipeline (cnctwin.mechanics, rebuild policy, forming lobes,
  linearised kernel table, bus ProcessCoefficients);
* :mod:`.geometry` - tooling's effective EdgeModel -> the kernel tool layout;
* :mod:`.memory` - stale bins, refill planes, analytic stocks; :mod:`.spindle` - spindle load and duty;
* :mod:`.domain` - CuttingDomain; :mod:`.state`, :mod:`.schema`, :mod:`.channels`, :mod:`.abi`;
* :mod:`.reference` - plain-Python reference of the milling kernel; :mod:`.analytic` - closed forms and
  ideal tools; :mod:`.drive` - a driver of the kernels over a prescribed path.

Names are imported on first use (the kernels compile on import of :mod:`.kernels`).
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "CuttingDomain": "domain", "ContactInputs": "domain", "CuttingInputs": "domain",
    "CuttingState": "state", "CuttingSettings": "schema", "settings": "schema",
    "MechanicsModel": "coefficients", "LinearModel": "coefficients", "build_table": "coefficients",
    "build_cut_tool": "geometry", "row_plan": "geometry", "Driver": "drive",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'cnctwin.cutting' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
