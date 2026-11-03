"""Backend choice, automatic fallback and the run record of the GPU kernels (docs/models/gpu.md "Switch",
"Fallback", "Run record").

An :class:`Accelerator` belongs to one run (``for_context``). Its three methods have the CPU kernels'
signatures and results:

* :meth:`Accelerator.subtract_envelope` - ``cnctwin.removal.k_subtract.subtract_envelope``
* :meth:`Accelerator.work_step` - ``cnctwin.thermal.kernels.work_step``
* :meth:`Accelerator.advance_particles` - ``cnctwin.chips.kernels.advance_particles``

Each call picks the backend (:meth:`Accelerator.choose`) from ``physics.gpu`` and the device detection,
runs the GPU path when chosen and falls back to the CPU kernel when the GPU path raises: the GPU paths
change the caller's arrays only after their device phase succeeded, so the CPU kernel then runs on
untouched inputs. After a GPU failure the accelerator stays on the CPU for the rest of the run (no
repeated failures) and records why. ``run_info()`` is what run.json stores under ``gpu`` (no wall times:
run.json holds no volatile fields).
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

import numpy as np

from . import device as D
from .schema import GpuSettings, from_project

KERNELS = ("dexel_subtract", "thermal_work_step", "chips_advance")
ITEM_UNITS = {"dexel_subtract": "rays", "thermal_work_step": "cells", "chips_advance": "particles"}
PARITY = {
    "dexel_subtract": "bit-identical: field arrays, accumulators, events and status",
    "thermal_work_step": "bit-identical: grid, scratch (line_c, line_d, gprev unspecified) and the ledger terms",
    "chips_advance": ("bit-identical in the simulator; on a device exp, expm1 and pow may differ from glibc by "
                      "<= 2 ulp per call (docs/models/gpu.md \"Tolerances\")"),
}
MAX_ERRORS = 20


@dataclass
class KernelUsage:
    gpu_calls: int = 0
    cpu_calls: int = 0
    gpu_items: int = 0
    cpu_items: int = 0
    host_items: int = 0          # items of GPU calls the host computed (spilled rays, pile conflicts, log overflows)
    fallbacks: int = 0           # GPU chosen, failed, CPU used
    last_backend: str = ""
    last_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Accelerator:
    """GPU kernels of one run with automatic CPU fallback."""

    def __init__(self, settings: GpuSettings | None = None):
        self.settings = settings or GpuSettings()
        self.usage = {name: KernelUsage() for name in KERNELS}
        self.errors: list[str] = []
        self.disabled_reason = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ choice
    def device(self) -> D.DeviceInfo:
        return D.info(self.settings.device)

    def choose(self, kernel: str, items, slabs: int | None = None) -> tuple[str, str]:
        """(backend ``gpu`` | ``cpu``, reason) for a call of ``kernel`` with ``items`` work items (an int or a
        callable returning one; only ``auto`` evaluates it); ``slabs``: the envelope's slabs (dexel)."""
        mode = self.settings.mode
        if mode == "off":
            return "cpu", "physics.gpu: off"
        if D.disabled_by_env():
            return "cpu", "disabled by CNCTWIN_GPU=off"
        if self.disabled_reason:
            return "cpu", f"GPU disabled for this run after an error: {self.disabled_reason}"
        found = self.device()
        if not found.usable:
            return "cpu", f"no usable CUDA device: {found.reason}"
        if mode == "on":
            return "gpu", f"physics.gpu: on ({found.kind})"
        if found.kind != "cuda":
            return "cpu", "auto uses real CUDA devices only (the simulator is for correctness tests)"
        if slabs is not None and slabs < self.settings.min_slabs:
            return "cpu", f"auto: {slabs} slabs < {self.settings.min_slabs}"
        count = int(items() if callable(items) else items)
        threshold = self.settings.threshold(kernel)
        if count < threshold:
            return "cpu", f"auto: {count} {ITEM_UNITS[kernel]} < {threshold}"
        return "gpu", f"auto: {count} {ITEM_UNITS[kernel]} >= {threshold}"

    def _run(self, kernel: str, items, gpu_call, cpu_call, backend: str | None = None, slabs: int | None = None):
        """Run one call on the chosen backend (or the requested one: ``backend`` gpu | cpu), falling back to
        the CPU kernel when the GPU path raises GpuError. The work-item count is taken before the call (and
        not at all under ``physics.gpu: off``)."""
        if backend is None:
            backend, reason = self.choose(kernel, items, slabs)
        elif backend == "gpu":
            ok, why = D.available(self.settings.device)
            if self.disabled_reason:
                backend, reason = "cpu", f"GPU disabled for this run after an error: {self.disabled_reason}"
            elif not ok:
                backend, reason = "cpu", f"GPU requested, no usable CUDA device: {why}"
            else:
                reason = "GPU requested"
        elif backend == "cpu":
            reason = "CPU requested"
        else:
            raise ValueError(f"backend must be 'gpu', 'cpu' or None, not {backend!r}")
        count = 0 if self.settings.mode == "off" else int(items() if callable(items) else items)
        usage = self.usage[kernel]
        if backend == "gpu":
            try:
                D.ensure_selected(self.settings.device)
                result, host_items = gpu_call()
                with self._lock:
                    usage.gpu_calls += 1
                    usage.gpu_items += count
                    usage.host_items += int(host_items)
                    usage.last_backend = "gpu"
                    usage.last_reason = reason
                return result
            except D.GpuError as exc:
                with self._lock:
                    usage.fallbacks += 1
                    self.disabled_reason = str(exc)
                    if len(self.errors) < MAX_ERRORS:
                        self.errors.append(f"{kernel}: {exc}")
                reason = f"GPU failed, CPU used: {exc}"
        result = cpu_call()
        with self._lock:
            usage.cpu_calls += 1
            usage.cpu_items += count
            usage.last_backend = "cpu"
            usage.last_reason = reason
        return result

    # ------------------------------------------------------------------ the kernels
    def subtract_envelope(self, f, prof, spec, tgt, acc, ev, sc, backend: str | None = None) -> int:
        """removal's subtract_envelope (same arguments and results)."""
        from ..removal import k_subtract

        def items():
            try:
                return max(0, gpu_module("k_dexel").candidate_bound(f, prof, spec))
            except D.GpuError:
                return 0

        def gpu():
            stats = np.zeros(3, np.int64)
            status = gpu_module("k_dexel").subtract_envelope(f, prof, spec, tgt, acc, ev, sc, stats)
            return status, int(stats[1])

        def cpu():
            return int(k_subtract.subtract_envelope(f, prof, spec, tgt, acc, ev, sc))

        return self._run("dexel_subtract", _Lazy(items), gpu, cpu, backend, slabs=int(spec.slab_dr.shape[0]))

    def work_step(self, g, s, mat, bnd, dt: float, out: np.ndarray, backend: str | None = None) -> None:
        """thermal's work_step (same arguments and results)."""
        from ..thermal import kernels as thermal_kernels

        def gpu():
            gpu_module("k_thermal").work_step(g, s, mat, bnd, dt, out)
            return None, 0

        def cpu():
            thermal_kernels.work_step(g, s, mat, bnd, float(dt), out)

        self._run("thermal_work_step", int(g.f.size), gpu, cpu, backend)

    def advance_particles(self, P, H, W, S, C, out: np.ndarray, ev, backend: str | None = None) -> None:
        """chips' advance_particles (K1; same arguments and results)."""
        from ..chips import abi as chips_abi
        from ..chips import kernels as chips_kernels

        def gpu():
            stats = gpu_module("k_particles").advance_particles(P, H, W, S, C, out, ev)
            return None, int(stats[0])

        def cpu():
            chips_kernels.advance_particles(P, H, W, S, C, out, ev)

        n = int(P.n[0])
        self._run("chips_advance", _Lazy(lambda: int(np.count_nonzero(P.status[:n] != chips_abi.FREE))), gpu, cpu,
                  backend)

    # ------------------------------------------------------------------ records
    def warmup(self) -> dict:
        """Compile the GPU kernels now (when the GPU would be used); returns {kernel: "compiled" | reason}."""
        out = {}
        backend, reason = self.choose("dexel_subtract", 1 << 62)
        if backend != "gpu":
            return {name: reason for name in KERNELS}
        for name, module, build in (("dexel_subtract", "k_dexel", "gpu_kernel"),
                                    ("thermal_work_step", "k_thermal", "kernels"),
                                    ("chips_advance", "k_particles", "gpu_kernel")):
            try:
                D.ensure_selected(self.settings.device)
                getattr(gpu_module(module), build)()
                out[name] = "compiled"
            except Exception as exc:          # recorded; the calls will fall back
                out[name] = f"{type(exc).__name__}: {exc}"
        return out

    def run_info(self, device: D.DeviceInfo | None = None) -> dict:
        """The ``gpu`` entry of run.json: switch, device, per-kernel use and parity, errors. ``device`` replaces
        the detection (synthetic runs, tests); without it the device is probed unless the GPU is off."""
        if device is not None:
            device = device.to_dict()
        elif self.settings.mode != "off" and not D.disabled_by_env():
            device = self.device().to_dict()
        else:
            device = D.DeviceInfo(reason="not probed (physics.gpu: off or CNCTWIN_GPU=off)").to_dict()
        return {"mode": self.settings.mode, "device_index": self.settings.device,
                "thresholds": {"min_rays": self.settings.min_rays, "min_slabs": self.settings.min_slabs,
                               "min_cells": self.settings.min_cells, "min_particles": self.settings.min_particles},
                "device": device,
                "kernels": {name: {**self.usage[name].to_dict(), "parity": PARITY[name]} for name in KERNELS},
                "disabled_reason": self.disabled_reason, "errors": list(self.errors)}


def gpu_module(name: str):
    """cnctwin.gpu.<name> (k_dexel, k_thermal, k_particles); any import failure (a domain kernel this module
    mirrors changed shape, numba-cuda broken) is a GpuError, so the CPU kernel runs instead."""
    import importlib

    try:
        return importlib.import_module(f"cnctwin.gpu.{name}")
    except Exception as exc:
        raise D.GpuError(f"cnctwin.gpu.{name} cannot be imported ({type(exc).__name__}: {exc})") from exc


class _Lazy:
    """A work-item count computed once, on first use (auto mode and the usage record)."""

    def __init__(self, fn):
        self._fn = fn
        self._value = None

    def __call__(self) -> int:
        if self._value is None:
            self._value = int(self._fn())
        return self._value


# ---------------------------------------------------------------------------------------------- module level
_DEFAULT: Accelerator | None = None
_DEFAULT_LOCK = threading.Lock()


def default() -> Accelerator:
    """The process-wide accelerator (``physics.gpu: auto`` defaults) for callers without a run context."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Accelerator()
        return _DEFAULT


def for_context(ctx) -> Accelerator:
    """The run's accelerator: ``ctx.gpu`` when set, else one from ``ctx.project.physics.gpu``, stored as
    ``ctx.gpu`` (RunContext is a plain dataclass; the attribute is additive)."""
    found = getattr(ctx, "gpu", None)
    if isinstance(found, Accelerator):
        return found
    accel = Accelerator(from_project(getattr(ctx, "project", None)))
    try:
        setattr(ctx, "gpu", accel)
    except AttributeError:
        pass
    return accel
