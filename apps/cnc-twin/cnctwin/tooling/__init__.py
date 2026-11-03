"""Tooling library: one owner for every tool's geometry as the physics sees it
(docs/models/tooling.md).

* :func:`build_tool` - EdgeModel (core/abi.py) of every tool type, coefficient groups, angle tables
  and element frames, revolved envelope, thread and insert profiles, properties (gullet, fluted
  section, critical sections, mass and inertia, balance, max speed, coolant holes, hand);
* :func:`build_tool_table` - the project's ToolTable (core/state.py);
* :func:`edge_state`, :func:`effective_edge` - per-macro-step composition (wear, growth,
  pull-out, slip) of the effective EdgeModel;
* :class:`ToolingDomain` - the SimState field ``tooling`` and the kernel inputs;
* kernels in :mod:`cnctwin.tooling.kernels` (edge_points_T0, edge_frames_T0, compose_effective,
  envelope_radius_at, envelope_radius_profile).

Names are imported on first use, so ``cnctwin.tooling.schema`` (read by the project schema) does
not load the builders or compile the kernels.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "EdgeFrameT": "abi", "EdgeState": "abi", "EdgeTrig": "abi", "Envelope": "abi", "GroupTable": "abi",
    "InsertProfile": "abi", "ThreadProfile": "abi", "ToolingInputs": "abi",
    "ToolGeometry": "build", "build_tool": "build", "element_positions_T": "build", "runout_vectors": "build",
    "ToolingCatalog": "catalog", "ToolingDomain": "domain", "empty_edge": "domain",
    "SpindleNose": "envelope", "spindle_nose_from_machine": "envelope",
    "PHYSICS_TOOLING": "schema", "ToolingSettings": "schema", "settings_from": "schema",
    "ToolingState": "state", "build_tool_table": "table", "edge_state": "update", "effective_edge": "update",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'cnctwin.tooling' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
