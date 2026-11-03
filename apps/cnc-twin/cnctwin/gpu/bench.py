"""CPU vs GPU times of the three kernels on the synthetic cases (docs/models/gpu.md "Benchmark").

:func:`benchmark` times each CPU kernel and, when a CUDA device is usable, its GPU path on the same inputs
(best of ``repeat`` runs on fresh copies; the first GPU call, which compiles the kernels, is reported
separately), checks that both give the same result, and says plainly when there is no device. The CUDA
simulator is never timed unless asked (``include_simulator``): it executes Python, its times mean nothing.
Wall-clock times are measurements, not results; nothing here goes into a run directory.

Sizes: ``"smoke"`` (seconds, for tests), ``"default"`` (the reference sizes of docs/models/gpu.md) or a
mapping ``{"dexel": {...dexel_case kwargs}, "thermal": {...}, "chips": {...}}``.
"""

from __future__ import annotations

import time

import numpy as np

from . import cases
from . import device as D

SIZES = {
    "smoke": {"dexel": {"size_mm": (3.0, 2.5, 2.5), "lattice_mm": 0.3, "cuts": 2, "refine": False},
              "thermal": {"shape": (6, 5, 4)}, "chips": {"n": 12}},
    "default": {"dexel": {"size_mm": (20.0, 16.0, 10.0), "lattice_mm": 0.1, "cuts": 8},
                "thermal": {"shape": (96, 64, 32), "cell_mm": 1.0}, "chips": {"n": 4096}},
}


def _best(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(max(1, int(repeat))):
        prepared = fn.prepare()
        start = time.perf_counter()
        fn.run(prepared)
        best = min(best, time.perf_counter() - start)
    return best


class _Dexel:
    def __init__(self, case, backend):
        from ..removal import k_subtract

        from . import k_dexel
        self.case = case
        self.call = k_subtract.subtract_envelope if backend == "cpu" else k_dexel.subtract_envelope

    def prepare(self):
        return self.case

    def run(self, case):
        return cases.dexel_run(case, self.call)


class _Thermal:
    def __init__(self, case, backend):
        from ..thermal import kernels

        from . import k_thermal
        self.case = case
        self.call = kernels.work_step if backend == "cpu" else k_thermal.work_step

    def prepare(self):
        g, s, mat, bnd, dt = self.case
        return cases.clone(g), cases.clone(s), mat, bnd, dt, np.zeros(16)

    def run(self, prepared):
        g, s, mat, bnd, dt, out = prepared
        self.call(g, s, mat, bnd, dt, out)
        # line_c, line_d and gprev are scratch the GPU path leaves unspecified
        return g, (s.src, s.a, s.cap, s.rx, s.ry, s.rz, s.bits), out


class _Chips:
    def __init__(self, case, backend):
        from ..chips import abi
        from ..chips import kernels

        from . import k_particles
        self.case = case
        self.abi = abi
        self.call = kernels.advance_particles if backend == "cpu" else k_particles.advance_particles

    def prepare(self):
        P, H, W, S, C = self.case
        out, ev = self.abi.step_out(4 * len(P.mass_kg) + 16)
        return cases.clone(P), cases.clone(H), W, S, C, out, ev

    def run(self, prepared):
        P, H, W, S, C, out, ev = prepared
        self.call(P, H, W, S, C, out, ev)
        return P, H, out, ev


def _equal(a, b) -> bool:
    if isinstance(a, (tuple, list)) and not hasattr(a, "_fields"):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "_fields"):
        return all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    return a == b


def _max_rel(a, b) -> float:
    """Largest relative difference over the float arrays of two results (0 when equal)."""
    worst = 0.0
    if isinstance(a, (tuple, list)) or hasattr(a, "_fields"):
        for x, y in zip(a, b):
            worst = max(worst, _max_rel(x, y))
        return worst
    x = np.asarray(a)
    y = np.asarray(b)
    if x.dtype.kind != "f" or x.size == 0:
        return worst if np.array_equal(x, y) else float("inf")
    both = np.isfinite(x) & np.isfinite(y)
    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
        return float("inf")
    scale = np.maximum(np.abs(x[both]), np.abs(y[both]))
    diff = np.abs(x[both] - y[both])
    nonzero = scale > 0.0
    return float(np.max(diff[nonzero] / scale[nonzero])) if np.any(nonzero) else 0.0


def _dexel_items(case) -> int:
    from .k_dexel import candidate_bound
    return int(sum(max(0, candidate_bound(case.field, case.profile, spec)) for spec in case.specs))


def benchmark(sizes="default", repeat: int = 3, device_index: int = 0, include_simulator: bool = False,
              kernels=("dexel_subtract", "thermal_work_step", "chips_advance")) -> dict:
    """Times per kernel: {"device", "gpu_timed", "note", "cases": {kernel: {items, cpu_s, gpu_s, gpu_first_s,
    speedup, identical, max_rel_diff}}}. ``gpu_s`` is None when no device is timed (the note says why)."""
    spec = SIZES[sizes] if isinstance(sizes, str) else dict(sizes)
    found = D.info(device_index)
    timed = found.usable and (found.kind == "cuda" or (include_simulator and found.kind == "simulator"))
    if not found.usable:
        note = f"no usable CUDA device ({found.reason}): CPU times only"
    elif found.kind == "simulator" and not include_simulator:
        note = "CUDA simulator: GPU times not measured (it executes Python); CPU times only"
    elif found.kind == "simulator":
        note = "CUDA simulator timed on request: its GPU times are Python execution times, not device times"
    else:
        note = f"CUDA device {found.index}: {found.name}"
    report = {"device": found.to_dict(), "gpu_timed": bool(timed), "note": note, "repeat": int(repeat),
              "sizes": sizes if isinstance(sizes, str) else "custom", "cases": {}}
    if timed:
        D.ensure_selected(device_index)
    builders = {
        "dexel_subtract": (lambda: cases.dexel_case(**spec.get("dexel", {})), _Dexel, _dexel_items),
        "thermal_work_step": (lambda: cases.thermal_case(**spec.get("thermal", {})), _Thermal,
                              lambda c: int(c[0].f.size)),
        "chips_advance": (lambda: cases.chips_case(**spec.get("chips", {})), _Chips, lambda c: int(c[0].n[0])),
    }
    for name in kernels:
        make, runner, items = builders[name]
        case = make()
        cpu = runner(case, "cpu")
        reference = cpu.run(cpu.prepare())
        entry = {"items": items(case), "cpu_s": _best(cpu, repeat), "gpu_s": None, "gpu_first_s": None,
                 "speedup": None, "identical": None, "max_rel_diff": None}
        if timed:
            gpu = runner(case, "gpu")
            start = time.perf_counter()
            result = gpu.run(gpu.prepare())
            entry["gpu_first_s"] = time.perf_counter() - start
            entry["gpu_s"] = _best(gpu, repeat)
            entry["speedup"] = entry["cpu_s"] / entry["gpu_s"] if entry["gpu_s"] > 0.0 else None
            entry["identical"] = _equal(reference, result)
            entry["max_rel_diff"] = 0.0 if entry["identical"] else _max_rel(reference, result)
        report["cases"][name] = entry
    return report
