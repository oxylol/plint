"""Compile every kernel module of a tree into its Numba cache, each module in its own process (the warmup's way:
all modules in one process outgrow a 1.8 GB cap). Usage: bin/tsafe python tools/convergence/warm.py [--root DIR]."""

from __future__ import annotations

import sys
import time

import env

root = env.setup()

from cnctwin.core.warmup import compile_modules  # noqa: E402

t0 = time.perf_counter()
done = compile_modules(stream=sys.stdout, isolated=True)
print(f"compiled {len(done)} kernel modules of {root} in {time.perf_counter() - t0:.0f} s", flush=True)
