"""Machine data import, alignment with simulated runs, and calibration (docs/models/calibration.md).

* :func:`load_machine_data` - MTConnect (files or an agent), OPC UA (a server's history or an export),
  FANUC FOCAS collector logs, generic CSV, or a saved log -> :class:`MachineLog` (time-stamped SI channels).
* :class:`SimulatedRun` - a run directory (docs/architecture.md 16) as comparable channels and block markers.
* ``cnctwin.mdata.align.align`` - t_sim = offset + scale * t_meas from block markers, the program start or the
  cross-correlation of spindle load / power; ``align.overlay`` and :func:`error_metrics` for the UI and report.
* ``cnctwin.mdata.fit.fit`` - bounded least squares through a ``simulate(params)`` callback with intervals from
  the Jacobian.
* ``cnctwin.mdata.calibrate.calibrate`` - the ``calibrate`` job (cnctwin.api worker entry) and
  ``cnctwin.mdata.calibrate.main`` its command line; writes the calibration overlay (``calibration.yaml``,
  format cnctwin-calibration) a project includes (``calibration: calibration.yaml``, applied by
  cnctwin.project.resolve) and ``project.calibrated.yaml`` with it baked in.

The submodules are imported lazily, so ``import cnctwin.mdata`` stays cheap (``align``, ``fit``, ``overlay`` and
``calibrate`` are submodules: their functions of the same name are reached through them).
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "MachineLog": "log", "Channel": "log", "MachineEvent": "log", "LogBuilder": "log", "merge": "log",
    "parse_time_ns": "log", "format_time_ns": "log", "CHANNELS": "log",
    "load_machine_data": "importers",
    "read_mtconnect": "mtconnect", "read_focas": "focas", "read_csv_log": "csvlog", "read_opcua": "opcua",
    "SimulatedRun": "simrun",
    "Alignment": "align", "error_metrics": "align", "resample": "align",
    "Param": "fit", "ObservationSet": "fit", "FitSettings": "fit", "FitResult": "fit",
    "Target": "overlay", "load_overlay": "overlay", "write_overlay": "overlay", "apply_to_raw": "overlay",
    "apply_to_project": "overlay", "apply_to_tools": "overlay", "controller_overrides": "overlay",
    "model_overrides": "overlay", "factors": "overlay", "calibrated_project": "overlay", "combined": "overlay",
    "Context": "targets", "Problem": "targets", "TARGETS": "targets",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'cnctwin.mdata' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
