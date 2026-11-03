"""Offline check that every GPU kernel of this package compiles for a real device (docs/models/gpu.md
"Checks without a device").

With no usable GPU (this box: the GTX 950's kernel driver is not loaded) the simulator proves the
kernels' arithmetic, but not that numba-cuda/NVVM accept them. :func:`compile_all` compiles every kernel to
PTX for a compute capability (default 5.2, the GTX 950) with the real NVVM of the installed CUDA toolkit,
no driver needed, and :func:`fma_hook_works` checks that :func:`cnctwin.gpu.device.exact_compile` really
turns FMA contraction off with this numba-cuda version (a pure ``a * b + c`` kernel has no ``fma.rn.f64``
under it and one without it). Must run in a process without the CUDA simulator: :func:`check` starts one.

Command line: ``python -m cnctwin.gpu.ptxcheck [major.minor [group,...]]`` prints one JSON object (groups:
thermal, particles, dexel; default all).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager

DEFAULT_CC = (5, 2)
CHECK_TIMEOUT_S = 900.0
_TAG = "CNCTWIN_PTXCHECK "


GROUPS = ("thermal", "particles", "dexel")


def kernel_sources(groups=GROUPS) -> dict:
    """name -> (python source function, signature) of the CUDA kernels of the given groups."""
    import importlib

    out = {}
    for group in groups:
        if group not in GROUPS:
            raise ValueError(f"unknown kernel group {group!r} (known: {', '.join(GROUPS)})")
        module = importlib.import_module(f"cnctwin.gpu.k_{group}")
        for name, fn in module.SOURCES.items():
            out[f"{group}.{name}"] = (fn, module.SIGNATURES[name])
    return out


@contextmanager
def _fake_device(cc):
    """numba-cuda asks the current device for its compute capability when it compiles device functions;
    answer with ``cc`` while compiling offline."""
    import numba.cuda.dispatcher as dispatcher

    class _Device:
        compute_capability = tuple(cc)

    original = dispatcher.get_current_device
    dispatcher.get_current_device = lambda: _Device()
    try:
        yield
    finally:
        dispatcher.get_current_device = original


def _count_fma(ptx: str) -> int:
    return len(re.findall(r"\bfma\.rn\.f64\b", ptx))


def fma_hook_works(cc=DEFAULT_CC) -> dict:
    """PTX of ``out = x * y + z`` with and without exact_compile: fused only without it."""
    from numba import cuda

    from .device import exact_compile

    def axpy(out, x, y, z):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = x[i] * y[i] + z[i]

    sig = "void(float64[::1], float64[::1], float64[::1], float64[::1])"
    default_ptx, _ = cuda.compile_ptx(axpy, sig, cc=tuple(cc))
    with exact_compile():
        exact_ptx, _ = cuda.compile_ptx(axpy, sig, cc=tuple(cc))
    return {"fma_default": _count_fma(default_ptx), "fma_exact": _count_fma(exact_ptx),
            "ok": _count_fma(default_ptx) > 0 and _count_fma(exact_ptx) == 0}


def compile_all(cc=DEFAULT_CC, groups=GROUPS) -> dict:
    """Compile the kernels of ``groups`` to PTX for ``cc`` (no device needed; never in the simulator)."""
    from numba import cuda

    from .device import exact_compile, simulator_mode
    from .twins import Builder

    if simulator_mode():
        raise RuntimeError("ptxcheck.compile_all needs NUMBA_ENABLE_CUDASIM=0")
    build = Builder("cuda", cuda, simulator=False)
    results = {}
    with _fake_device(cc), exact_compile():
        for name, (fn, sig) in sorted(kernel_sources(groups).items()):
            try:
                ptx, _ = cuda.compile_ptx(build.rebind(fn), sig, cc=tuple(cc))
                target = re.findall(r"\.target\s+(\S+)", ptx)
                results[name] = {"ok": True, "ptx_lines": ptx.count("\n"), "target": target[0] if target else "",
                                 "local_memory": ".local" in ptx}
            except Exception as exc:
                results[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc).strip()[:2000]}"}
    return results


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cc = tuple(int(v) for v in argv[0].split(".")) if argv else DEFAULT_CC
    groups = tuple(argv[1].split(",")) if len(argv) > 1 else GROUPS
    report = {"cc": list(cc), "groups": list(groups)}
    try:
        report["fma"] = fma_hook_works(cc)
        report["kernels"] = compile_all(cc, groups)
        report["ok"] = report["fma"]["ok"] and all(r["ok"] for r in report["kernels"].values())
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    sys.stdout.write(_TAG + json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


def check(cc=DEFAULT_CC, groups=GROUPS, timeout_s: float = CHECK_TIMEOUT_S) -> dict:
    """Run :func:`main` in a child process without the simulator (the NVVM of the CUDA toolkit must be
    installed: ``pip install 'numba-cuda[cu12]'``)."""
    from ..paths import ROOT

    env = dict(os.environ)
    env["NUMBA_ENABLE_CUDASIM"] = "0"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    args = [sys.executable, "-m", "cnctwin.gpu.ptxcheck", ".".join(str(c) for c in cc), ",".join(groups)]
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(ROOT),
                              check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the PTX check did not finish within {timeout_s:g} s"}
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(_TAG):
            return json.loads(line[len(_TAG):])
    return {"ok": False, "error": f"the PTX check failed (exit {done.returncode}): {done.stderr.strip()[-2000:]}"}


if __name__ == "__main__":
    raise SystemExit(main())
