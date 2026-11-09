"""``twin warmup`` (docs/models/loop.md "twin warmup"): compile and cache every kernel before the first real run.

Numba compiles each kernel on its first call and caches it on disk under the kernel-source key (bin/tsafe,
bin/twin, the image). A cold first run spends minutes compiling; the image runs ``twin warmup`` at build time.
:func:`main` imports every kernel module, each in its own Python process (Numba compiles a kernel with an
explicit signature when it is defined, and keeps the machine code and LLVM state of everything it compiled
resident: all modules in one process outgrow a 1.8 GB memory cap), then runs two tiny built-in projects through
the whole loop (cnctwin.core.loop.run_project, every domain on), each in its own process, so the kernels compiled
on first call are cached too: a short side step in an aluminium block on the 3-axis mill of
examples/pocket-islands-al6061 and a short OD turning pass on the lathe of examples/lathe-shaft-4140.

``twin warmup [--keep DIR] [--only mill|lathe] [--quiet] [--in-process]``: exit 0 when every module compiled and
both runs completed, 1 otherwise. ``--in-process``: no child processes (the runs asked for in this process).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# the slow domains' heavy L2 analyses on coarse settings: warmup compiles kernels, it does not need their answers
# (the process-damped stability limit is a fixed-point iteration of lobe scans: two iterations, not up to 30)
WARM_PHYSICS = {"dynamics": {"sld_n_freq": 64, "sld_n_speeds": 8, "sld_n_lobes": 2, "sd_check": 0, "sd_max_points": 0,
                             "sld_refresh_macro": 1000, "engagement_tol": 0.5, "pd_max_iter": 2, "pd_tol": 0.05},
                "cutting": {"every_n_macro": 50},
                "structure": {"max_dof": 3000, "every_n_macro": 50},
                "thermal": {"every_n_macro": 5, "cell_mm": 4.0}}

KERNEL_MODULES = ("cnctwin.core.recorder", "cnctwin.core.hot_kernels", "cnctwin.core.hotloop", "cnctwin.geom.bvh",
                  "cnctwin.geom.dda", "cnctwin.geom.overlap", "cnctwin.geom.primitives", "cnctwin.geom.transform",
                  "cnctwin.controller.kernels", "cnctwin.motion.kernels", "cnctwin.tooling.kernels",
                  "cnctwin.mechanics.kernels", "cnctwin.cutting.kernels", "cnctwin.removal.kernels",
                  "cnctwin.dynamics.kernels", "cnctwin.structure.kernels", "cnctwin.thermal.kernels",
                  "cnctwin.chips.kernels", "cnctwin.coolant.kernels", "cnctwin.wear.kernels",
                  "cnctwin.holding.kernels", "cnctwin.surface.kernels", "cnctwin.inspect.kernels",
                  "cnctwin.render.kernels")

SLOT_PROGRAM = """%
O09001 (TWIN WARMUP: SIDE STEP ALONG THE FRONT FACE)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T1 M06
G90 G54 G00 X-8. Y-4.5 S10000 M03
G43 H01 Z25. M08
Z2.
G01 Z-1. F1000.
G01 X12. F1600.
G01 Y-8.
G00 Z25.
M09
M05
G53 G49 Z0.
M30
%
"""

TURN_PROGRAM = """%
O09002 (TWIN WARMUP: SHORT OD PASS)
N10 G21 G18 G40 G80 G99
N20 G54
N30 G50 S2500
N100 G53 G00 X0
N110 T0101
N120 G97 S1400 M04
N130 M08
N140 G00 X34.0 Z2.0
N150 G00 X27.0
N160 G01 Z-3.0 F0.25
N170 G00 X34.0
N180 G53 G00 X0
N190 M05
N200 M30
%
"""


def mill_project(directory) -> Path:
    """A 30 x 20 x 10 mm block of Al 6061-T6 in the vise of examples/pocket-islands-al6061, one D10 end mill, a
    side step 0.5 mm wide and 1 mm deep along 12 mm of the front face at 10 000 rpm, led out of the cut before the
    rapid retract (no CAD). A side step, not a slot: a full-width slot at this depth is far above the D10's
    predicted stability limit (about 0.05 mm on the lobe diagram), and the simulated chatter grows until the tool
    fails."""
    from ..paths import EXAMPLES
    from .drive import project_variant
    size = (30.0, 20.0, 10.0)

    def edit(data):
        data["cad"] = None
        data["stock"]["size_mm"] = list(size)
        data["physics"].update({k: dict(v) for k, v in WARM_PHYSICS.items()})
        data["physics"]["lattice_mm"] = 0.25
        data["tools"]["list"] = [data["tools"]["list"][0]]
        data["production"]["parts"] = 1
        op = data["operations"][0]
        op["setup"]["stock_pose"] = {"translate_mm": [-size[0] / 2, -size[1] / 2, 0], "rotate_deg": [0, 0, 0]}
        op["setup"]["work_offsets_mm"] = {"G54": [-381.0 - size[0] / 2, -203.2 - size[1] / 2, -510.0 + size[2]]}
        op["setup"]["workholding"][0]["jaw_opening_mm"] = size[1]
        data["outputs"] = {"report": [], "images": False, "video": False, "audio": False}
    return project_variant(Path(EXAMPLES) / "pocket-islands-al6061", directory, program_text=SLOT_PROGRAM,
                           program_name="warmup.nc", edits=edit)


def lathe_project(directory) -> Path:
    """A d30 x 40 mm bar of 4140 in the chuck of examples/lathe-shaft-4140, the CNMG roughing tool, a 3 mm OD pass
    1.5 mm deep at 1400 rpm (M04; no CAD, no tailstock)."""
    from ..paths import EXAMPLES
    from .drive import project_variant
    d_mm, length_mm = 30.0, 40.0

    def edit(data):
        data["cad"] = None
        data["physics"].update({k: dict(v) for k, v in WARM_PHYSICS.items()})
        data["physics"]["lattice_mm"] = 0.5
        data["stock"] = {"kind": "cylinder", "size_mm": [d_mm, length_mm], "tolerance_mm": None,
                         "non_squareness_deg": None, "form": "cold_drawn_bar", "mesh": None}
        data["tools"]["list"] = [data["tools"]["list"][0]]
        data["production"]["parts"] = 1
        op = data["operations"][0]
        op["setup"]["workholding"] = op["setup"]["workholding"][:2]
        op["setup"]["stock_pose"] = {"translate_mm": [0, 0, 99], "rotate_deg": [0, 0, 0]}
        op["setup"]["work_offsets_mm"] = {"G54": [-426.0, 0.0, 99.0 + length_mm - 662.0]}
        data["outputs"] = {"report": [], "images": False, "video": False, "audio": False}
    return project_variant(Path(EXAMPLES) / "lathe-shaft-4140", directory, program_text=TURN_PROGRAM,
                           program_name="warmup.nc", edits=edit)


def _child_env() -> dict:
    """The environment of a child process: this one's (NUMBA_CACHE_DIR, thread settings), with the folder this
    cnctwin package is imported from first on PYTHONPATH so the child imports the same build."""
    import cnctwin
    env = dict(os.environ)
    root = str(Path(cnctwin.__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def compile_modules(stream=None, isolated: bool = True) -> list[str]:
    """Import every kernel module (explicit signatures compile at definition and go to the cache) and touch its
    KERNELS tuple; ``isolated``: each module in its own Python process. Returns ``"<module> (<n> kernels)"`` per
    module; raises RuntimeError naming the modules that failed."""
    done, failed = [], []
    env = _child_env() if isolated else None
    for name in KERNEL_MODULES:
        try:
            if importlib.util.find_spec(name) is None:
                continue
        except ModuleNotFoundError:
            continue
        if isolated:
            code = (f"import importlib; m = importlib.import_module({name!r}); "
                    "print(len(getattr(m, 'KERNELS', ())))")
            proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [f"exit {proc.returncode}"]
                failed.append(f"{name}: {tail[0]}")
                continue
            count = (proc.stdout.strip().splitlines() or ["0"])[-1]
        else:
            module = importlib.import_module(name)
            count = str(len(getattr(module, "KERNELS", ())))
        done.append(f"{name} ({count} kernels)")
        if stream is not None:
            stream.write(f"compiled {name}\n")
    if failed:
        raise RuntimeError("kernel modules that did not compile: " + "; ".join(failed))
    return done


def _run_kind(kind: str, base: Path, stream) -> bool:
    """One tiny project through run_project in this process; True when the run completed. A run the physics
    stops (a conservation or model error) is reported, not raised: the kernels it reached are cached."""
    from .errors import TwinError
    from .loop import run_project
    build = {"mill": mill_project, "lathe": lathe_project}[kind]
    start = time.perf_counter()
    project = build(base / f"{kind}-project")
    try:
        result = run_project(project, base / f"{kind}-run", {"outputs": False, "sanity": False}, None)
    except (TwinError, ValueError, FloatingPointError) as error:
        if stream is not None:
            stream.write(f"{kind}: stopped after {time.perf_counter() - start:.1f} s: {type(error).__name__}: "
                         f"{error}\n")
        return False
    ok = bool(result.get("complete"))
    if stream is not None:
        stream.write(f"{kind}: {'done' if ok else 'incomplete'} in {time.perf_counter() - start:.1f} s "
                     f"(verdict {'safe' if result['verdict']['safe'] else 'not safe'})\n")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="twin warmup", description="compile and cache every kernel: each kernel "
                                     "module in its own process, then two tiny built-in projects through the loop")
    parser.add_argument("--keep", help="keep the projects and run directories in this folder")
    parser.add_argument("--only", choices=("mill", "lathe"), help="run only one of the two projects")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="no child processes: compile and run here")
    parser.add_argument("--no-compile", action="store_true", help=argparse.SUPPRESS)    # a run child of main
    args = parser.parse_args(argv)
    stream = None if args.quiet else sys.stderr
    base = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="twin-warmup-"))
    base.mkdir(parents=True, exist_ok=True)
    kinds = [args.only] if args.only else ["mill", "lathe"]
    status = 0
    try:
        t0 = time.perf_counter()
        try:
            if not args.no_compile:
                compile_modules(stream, isolated=not args.in_process)
        except RuntimeError as error:
            status = 1
            if stream is not None:
                stream.write(f"{error}\n")
        for kind in kinds:
            if args.in_process:
                ok = _run_kind(kind, base, stream)
            else:
                command = [sys.executable, "-m", "cnctwin.core.warmup", "--in-process", "--no-compile", "--only", kind,
                           "--keep", str(base)] + (["--quiet"] if args.quiet else [])
                ok = subprocess.run(command, env=_child_env()).returncode == 0
            status = status or (0 if ok else 1)
        if stream is not None:
            if not args.no_compile:
                stream.write(f"warmup finished in {time.perf_counter() - t0:.1f} s\n")
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "mill_project", "lathe_project", "compile_modules", "WARM_PHYSICS", "KERNEL_MODULES"]
