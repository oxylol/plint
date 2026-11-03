"""Optional GPU acceleration of three CPU kernels (docs/models/gpu.md).

* dexel subtraction of a tool envelope over many rays (``cnctwin.removal.k_subtract.subtract_envelope``),
  one GPU thread per ray: bit-identical to the CPU kernel;
* one workpiece thermal finite-volume step (``cnctwin.thermal.kernels.work_step``), one thread per cell and
  per lattice line: bit-identical;
* one particle flight window (``cnctwin.chips.kernels.advance_particles``), one thread per particle with an
  ordered host commit: bit-identical up to the device's transcendental functions.

The switch is ``physics.gpu: auto | on | off`` (:mod:`cnctwin.gpu.schema`); without a usable CUDA device
the CPU kernels run, automatically (:func:`available` says why). Nothing here imports numba-cuda until a
GPU path is actually used, so the package is importable on hosts without it.

Typical use from a domain (same arguments as the CPU kernel)::

    from cnctwin import gpu
    accel = gpu.for_context(ctx)          # the run's accelerator (physics.gpu of ctx.project)
    accel.work_step(g, s, mat, bnd, dt, out)
"""

from __future__ import annotations

from .accel import KERNELS, PARITY, Accelerator, KernelUsage, default, for_context
from .device import DeviceInfo, GpuError, available, info, probe
from .schema import MODES, GpuSettings, from_project, schema_node, settings


def subtract_envelope(f, prof, spec, tgt, acc, ev, sc, accel: Accelerator | None = None, backend: str | None = None):
    """removal's subtract_envelope through ``accel`` (default: the process-wide accelerator)."""
    return (accel or default()).subtract_envelope(f, prof, spec, tgt, acc, ev, sc, backend)


def work_step(g, s, mat, bnd, dt, out, accel: Accelerator | None = None, backend: str | None = None):
    """thermal's work_step through ``accel``."""
    return (accel or default()).work_step(g, s, mat, bnd, dt, out, backend)


def advance_particles(P, H, W, S, C, out, ev, accel: Accelerator | None = None, backend: str | None = None):
    """chips' advance_particles (K1) through ``accel``."""
    return (accel or default()).advance_particles(P, H, W, S, C, out, ev, backend)


def benchmark(**kwargs) -> dict:
    """CPU vs GPU times of the three kernels (:func:`cnctwin.gpu.bench.benchmark`)."""
    from .bench import benchmark as run
    return run(**kwargs)


__all__ = ["Accelerator", "DeviceInfo", "GpuError", "GpuSettings", "KERNELS", "KernelUsage", "MODES", "PARITY",
           "advance_particles", "available", "benchmark", "default", "for_context", "from_project", "info", "probe",
           "schema_node", "settings", "subtract_envelope", "work_step"]
