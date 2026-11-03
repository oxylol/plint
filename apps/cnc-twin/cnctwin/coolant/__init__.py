"""Coolant domain: supply, jets in M, occlusion, heat transfer, lubrication and the tank
(docs/models/coolant.md).

* :class:`CoolantDomain` - the Domain object (writes ``bus.coolant``; owns the tank);
* protocols the integration wires: :class:`ToolShapes`, :class:`WorkpieceBricks`, :class:`SceneMeshes`,
  :class:`MountPoses` (complete defaults: :class:`TooledShapes`, :class:`StockBricks`,
  :class:`OccupancyBricks`, :class:`MachineMeshes`, :class:`cnctwin.coolant.kinematics.ChainMounts`);
* :class:`CoolantState` (checkpointable), :class:`CoolantSettings` (``physics.coolant``).

Names are imported on first use, so ``cnctwin.coolant.schema`` (read by the project schema) does not
load the kernels.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "CoolantDomain": "domain", "CoolantState": "state", "CoolantSettings": "schema", "settings": "schema",
    "ToolShapes": "protocols", "ToolShape": "protocols", "TooledShapes": "protocols",
    "WorkpieceBricks": "protocols", "BrickSet": "protocols", "StockBricks": "protocols",
    "OccupancyBricks": "protocols", "SceneMeshes": "protocols", "SceneMesh": "protocols",
    "MachineMeshes": "protocols", "MountPoses": "protocols", "ChainMounts": "kinematics",
    "MachineChains": "kinematics", "Fluid": "fluid", "fluid_from_db": "fluid",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'cnctwin.coolant' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
