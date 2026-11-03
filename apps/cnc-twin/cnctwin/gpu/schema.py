"""``physics.gpu`` project setting: whether the GPU kernels may be used (docs/models/gpu.md "Switch").

The switch is text, ``auto | on | off`` (default ``auto``)::

    physics:
      gpu: auto

or a mapping that also tunes ``auto`` and picks the device::

    physics:
      gpu: {mode: auto, device: 0, min_rays: 2048, min_slabs: 24, min_cells: 32768, min_particles: 1024}

* ``off`` - the CPU kernels only.
* ``on``  - the GPU kernels whenever a CUDA device (or, with ``NUMBA_ENABLE_CUDASIM=1``, the CUDA
  simulator) is usable, whatever the problem size; without one the CPU kernels run and the reason
  is recorded (a run never fails for want of an accelerator).
* ``auto`` - the GPU kernels on a real CUDA device when a call is at least the size given below
  (smaller calls lose to launch latency and transfers; a dexel subtraction also needs ``min_slabs`` slabs);
  never the simulator.

``true``/``false`` read as ``on``/``off`` (a YAML 1.1 writer may produce them). The declaration
:data:`FIELDS` is the single source: :func:`settings` reads the value into :class:`GpuSettings`
and rejects anything else, :func:`schema_node` builds the :mod:`cnctwin.db.schema` node for the
project schema (imported lazily, so this module stays importable without the databases).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

MODES = ("auto", "on", "off")

# name -> (kind, default, lo, hi, doc); kind: mode | count
FIELDS: dict[str, tuple[str, object, int, int, str]] = {
    "mode": ("mode", "auto", 0, 0, "auto | on | off"),
    "device": ("count", 0, 0, 63, "CUDA device index (CUDA_VISIBLE_DEVICES order)"),
    "min_rays": ("count", 2048, 1, 10 ** 9,
                 "auto: candidate rays of one dexel subtraction from which the GPU is used"),
    "min_slabs": ("count", 24, 1, 10 ** 4,
                  "auto: axial slabs of the envelope from which the GPU dexel path is used (its host commit "
                  "costs about the same per ray at any slab count; the geometry it saves grows with them)"),
    "min_cells": ("count", 32768, 1, 10 ** 10,
                  "auto: workpiece thermal cells from which the GPU finite-volume step is used"),
    "min_particles": ("count", 1024, 1, 10 ** 9,
                      "auto: particles in use (chips) from which the GPU flight step is used"),
}

# kernel name -> the threshold field auto compares the call size with
THRESHOLD_FIELD = {"dexel_subtract": "min_rays", "thermal_work_step": "min_cells",
                   "chips_advance": "min_particles"}


@dataclass(frozen=True)
class GpuSettings:
    mode: str = "auto"
    device: int = 0
    min_rays: int = 2048
    min_slabs: int = 24
    min_cells: int = 32768
    min_particles: int = 1024

    def threshold(self, kernel: str) -> int:
        return int(getattr(self, THRESHOLD_FIELD[kernel]))


def _mode(value, where: str) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str) and value.strip().lower() in MODES:
        return value.strip().lower()
    raise InputError(f"{where} = {value!r}: expected one of {', '.join(MODES)}")


def _get(section, key):
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def settings(value) -> GpuSettings:
    """Settings from the value of ``physics.gpu`` (None: defaults; text/bool: the mode; a mapping or an
    SI view with the FIELDS keys). Unknown keys and out-of-range values raise InputError."""
    if value is None:
        return GpuSettings()
    if isinstance(value, (str, bool)):
        return GpuSettings(mode=_mode(value, "physics.gpu"))
    raw = getattr(value, "raw", None)
    section = raw if isinstance(raw, Mapping) else value
    if not isinstance(section, Mapping):
        raise InputError(f"physics.gpu = {value!r}: expected auto, on, off or a mapping of {', '.join(FIELDS)}")
    unknown = sorted(str(k) for k in section if k not in FIELDS and k not in ("note", "notes"))
    if unknown:
        raise InputError(f"physics.gpu: unknown setting(s) {', '.join(unknown)} (known: {', '.join(FIELDS)})")
    values = {}
    for item in fields(GpuSettings):
        kind, default, lo, hi, _doc = FIELDS[item.name]
        found = _get(section, item.name)
        if found is None:
            values[item.name] = default
            continue
        if kind == "mode":
            values[item.name] = _mode(found, "physics.gpu.mode")
            continue
        if isinstance(found, bool) or not isinstance(found, (int, float)) or int(found) != found:
            raise InputError(f"physics.gpu.{item.name} = {found!r} is not a whole number")
        number = int(found)
        if not lo <= number <= hi:
            raise InputError(f"physics.gpu.{item.name} = {number} is outside [{lo}, {hi}]")
        values[item.name] = number
    return GpuSettings(**values)


def from_project(project) -> GpuSettings:
    """Settings of a resolved project (``project.physics.gpu``; any of SIMap, mapping or namespace)."""
    if project is None:
        return GpuSettings()
    physics = _get(project, "physics")
    if physics is None:
        return GpuSettings()
    return settings(_get(physics, "gpu"))


def schema_node():
    """The ``physics.gpu`` node of the project schema (cnctwin.db.schema)."""
    from ..db.schema import ByType, Count, Flag, Struct, Text

    nodes = {}
    for name, (kind, _default, lo, hi, doc) in FIELDS.items():
        nodes[name] = Text(MODES, doc=doc, sourced=False) if kind == "mode" else Count(lo=lo, hi=hi, doc=doc)
    return ByType(text=Text(MODES, doc="auto | on | off", sourced=False), boolean=Flag(),
                  mapping=Struct(nodes, meta=False, doc="GPU kernels (docs/models/gpu.md)"), nullable=True,
                  expected="auto, on, off or {mode, device, min_rays, min_slabs, min_cells, min_particles}",
                  doc="GPU acceleration of the dexel, thermal and particle kernels (docs/models/gpu.md)")


PHYSICS_GPU = schema_node
