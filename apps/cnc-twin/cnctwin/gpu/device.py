"""CUDA device detection, the exact-arithmetic compile setting and launch helpers (docs/models/gpu.md
"Device detection").

Detection never touches the CUDA driver in the simulation process: :func:`probe` runs a short child
process (``python -I -c PROBE_CODE``) that imports numba-cuda, initialises the driver, opens the device,
checks that the CUDA toolkit compiles for its compute capability, compiles a test kernel with FMA
contraction off and checks that its double-precision results equal the host's bit for bit. A missing
package, a driver that is not loaded (this box: GTX 950 with no ``/dev/nvidiactl``), a toolkit too new
for the card, a crash or a hang (timeout) all end as ``usable = False`` with the reason; nothing raises.
The result is cached per process and device index (:func:`info`, :func:`available`).

``NUMBA_ENABLE_CUDASIM=1`` (set before numba-cuda is first imported) selects the CUDA simulator: the
kernels run as Python on the host, for correctness tests only. ``CNCTWIN_GPU=off`` in the environment
disables the GPU path without probing (a kill switch for hosts and CI).

FMA contraction: NVVM fuses ``a * b + c`` into one fused multiply-add by default (``-fma=1``), which
rounds once where the CPU kernels round twice, so results would differ in the last bits.
:func:`exact_compile` compiles this package's kernels with ``-fma=0`` (numba-cuda passes no such
option itself; the context wraps ``numba.cuda.cudadrv.nvvm.compile_ir`` while the package's kernels
are compiled, under a lock). Division and square root stay IEEE correctly rounded (``fastmath`` off).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

PROBE_TIMEOUT_S = 120.0      # child-process probe: compile + launch of one small kernel takes 1-10 s
PROBE_TAG = "CNCTWIN_GPU_PROBE "
THREADS_PER_BLOCK = 128      # real device: occupancy-safe for register-heavy FP64 kernels on sm_5x
THREADS_PER_BLOCK_SIM = 32   # simulator: every thread of a block is a Python thread


class GpuError(RuntimeError):
    """The GPU path cannot be used (the caller falls back to the CPU kernels)."""


@dataclass(frozen=True)
class DeviceInfo:
    """What the GPU path can use in this process. ``kind``: ``cuda`` (a real device), ``simulator``
    (numba-cuda's CUDA simulator) or ``none``."""

    kind: str = "none"
    usable: bool = False
    reason: str = ""
    index: int = 0
    name: str = ""
    compute_capability: tuple[int, int] | None = None
    driver_version: str = ""
    runtime_version: str = ""
    nvvm_version: str = ""
    memory_bytes: int = 0
    multiprocessors: int = 0
    numba_cuda_version: str = ""
    fma_contraction: bool | None = None      # False: the test kernel's a*b-c was not fused
    exact_arithmetic: bool | None = None     # True: the test kernel's results equal the host's bit for bit
    supported_ccs: tuple[tuple[int, int], ...] = field(default=())

    def to_dict(self) -> dict:
        record = asdict(self)
        record["compute_capability"] = list(self.compute_capability) if self.compute_capability else None
        record["supported_ccs"] = [list(c) for c in self.supported_ccs]
        return record


# ---------------------------------------------------------------------------------------------- the probe
PROBE_CODE = r'''
import json, math, os, sys
res = {"kind": "none", "usable": False, "reason": "", "index": 0, "name": "", "compute_capability": None,
       "driver_version": "", "runtime_version": "", "nvvm_version": "", "memory_bytes": 0, "multiprocessors": 0,
       "numba_cuda_version": "", "fma_contraction": None, "exact_arithmetic": None, "supported_ccs": []}


def done(reason=""):
    if reason:
        res["reason"] = reason
    sys.stdout.write("CNCTWIN_GPU_PROBE " + json.dumps(res) + "\n")
    sys.stdout.flush()
    sys.exit(0)


index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
simulate = len(sys.argv) > 2 and sys.argv[2] == "simulate"
res["index"] = index
os.environ["NUMBA_ENABLE_CUDASIM"] = "1" if simulate else "0"
try:
    import numba_cuda
    res["numba_cuda_version"] = str(getattr(numba_cuda, "__version__", ""))
except Exception as exc:
    done("numba-cuda is not installed (%s: %s)" % (type(exc).__name__, exc))
try:
    from numba import cuda
except Exception as exc:
    done("numba.cuda cannot be imported (%s: %s)" % (type(exc).__name__, exc))
cc = None
if not simulate:
    nodes = ""
    if sys.platform.startswith("linux") and not os.path.exists("/dev/nvidiactl"):
        nodes = "; the NVIDIA kernel driver is not loaded (/dev/nvidiactl is missing)"
    try:
        from numba.cuda.cudadrv.driver import driver
        ok = driver.is_available
        err = driver.initialization_error
    except Exception as exc:
        done("CUDA driver: %s: %s%s" % (type(exc).__name__, exc, nodes))
    if not ok:
        done("CUDA driver: %s%s" % (str(err).strip(), nodes))
    try:
        res["driver_version"] = "%d.%d" % tuple(driver.get_version()[:2])
    except Exception:
        pass
try:
    count = len(list(cuda.list_devices()))
except Exception as exc:
    done("cannot list CUDA devices (%s: %s)" % (type(exc).__name__, exc))
if index >= count:
    done("CUDA device %d requested, %d present" % (index, count))
try:
    cuda.select_device(index)
    dev = cuda.get_current_device()
    name = getattr(dev, "name", "CUDA simulator")
    res["name"] = name.decode() if isinstance(name, bytes) else str(name)
    cc = tuple(int(c) for c in dev.compute_capability)
    res["compute_capability"] = list(cc)
    if not simulate:
        res["multiprocessors"] = int(dev.MULTIPROCESSOR_COUNT)
        res["memory_bytes"] = int(cuda.current_context().get_memory_info()[1])
except Exception as exc:
    done("CUDA device %d cannot be opened (%s: %s)" % (index, type(exc).__name__, exc))
if not simulate:
    try:
        from numba.cuda.cudadrv import nvrtc
        ccs = sorted(tuple(int(v) for v in c) for c in nvrtc.get_supported_ccs())
        res["supported_ccs"] = [list(c) for c in ccs]
    except Exception as exc:
        done("the CUDA toolkit (NVRTC) is not usable (%s: %s)" % (type(exc).__name__, exc))
    if ccs and cc < ccs[0]:
        done("compute capability %d.%d is below the lowest the installed CUDA toolkit compiles for (%d.%d)"
             % (cc + ccs[0]))
    try:
        from numba.cuda import runtime
        res["runtime_version"] = "%d.%d" % tuple(runtime.get_version()[:2])
    except Exception:
        pass
    try:
        from numba.cuda.cudadrv import nvvm
        res["nvvm_version"] = "%d.%d" % tuple(nvvm.NVVM().get_version()[:2])
        original = nvvm.compile_ir

        def compile_ir(llvmir, **options):
            options.setdefault("fma", False)
            return original(llvmir, **options)
        nvvm.compile_ir = compile_ir
    except Exception as exc:
        done("the CUDA toolkit (NVVM) is not usable (%s: %s)" % (type(exc).__name__, exc))


def kernel(x, y, z, out):
    i = cuda.grid(1)
    if i == 0:
        out[0] = x[0] * y[0] - z[0]
        out[1] = math.sqrt(x[1])
        out[2] = x[2] / y[2]
        out[3] = (x[3] + y[3]) + z[3]
        out[4] = x[4] * y[4] + z[4]


try:
    import numpy as np
    k = cuda.jit("void(float64[::1], float64[::1], float64[::1], float64[::1])")(kernel)
    x = np.array([1.0 + 2.0 ** -30, 2.0, 1.0, 0.1, 1.0 + 2.0 ** -27])
    y = np.array([1.0 + 2.0 ** -30, 1.0, 3.0, 0.2, 1.0 - 2.0 ** -27])
    z = np.array([1.0 + 2.0 ** -29, 0.0, 0.0, 0.3, -1.0])
    d_out = cuda.to_device(np.zeros(5))
    k[1, 32](cuda.to_device(x), cuda.to_device(y), cuda.to_device(z), d_out)
    cuda.synchronize()
    out = d_out.copy_to_host()
except Exception as exc:
    done("a test kernel failed to compile or run (%s: %s)" % (type(exc).__name__, str(exc).strip()[:400]))
expected = [float(x[0] * y[0] - z[0]), math.sqrt(float(x[1])), float(x[2] / y[2]), float((x[3] + y[3]) + z[3]),
            float(x[4] * y[4] + z[4])]
res["fma_contraction"] = bool(out[0] != 0.0 or out[4] != 0.0)
res["exact_arithmetic"] = bool(all(float(a) == b for a, b in zip(out, expected)))
res["kind"] = "simulator" if simulate else "cuda"
if not res["exact_arithmetic"]:
    done("the device's double-precision results differ from the host's (%s vs %s)"
         % ([float(v) for v in out], expected))
res["usable"] = True
done("CUDA simulator (Python execution: correctness only)" if simulate else "")
'''


def _from_record(record: dict) -> DeviceInfo:
    cc = record.get("compute_capability")
    return DeviceInfo(kind=str(record.get("kind", "none")), usable=bool(record.get("usable", False)),
                      reason=str(record.get("reason", "")), index=int(record.get("index", 0)),
                      name=str(record.get("name", "")), compute_capability=tuple(cc) if cc else None,
                      driver_version=str(record.get("driver_version", "")),
                      runtime_version=str(record.get("runtime_version", "")),
                      nvvm_version=str(record.get("nvvm_version", "")),
                      memory_bytes=int(record.get("memory_bytes", 0) or 0),
                      multiprocessors=int(record.get("multiprocessors", 0) or 0),
                      numba_cuda_version=str(record.get("numba_cuda_version", "")),
                      fma_contraction=record.get("fma_contraction"),
                      exact_arithmetic=record.get("exact_arithmetic"),
                      supported_ccs=tuple(tuple(c) for c in record.get("supported_ccs", []) or []))


def probe(index: int = 0, timeout_s: float = PROBE_TIMEOUT_S, simulate: bool = False,
          python: str | None = None) -> DeviceInfo:
    """Probe CUDA device ``index`` in a child process (never raises). ``simulate`` runs the same steps in
    the CUDA simulator (tests of the probe itself)."""
    args = [python or sys.executable, "-I", "-c", PROBE_CODE, str(int(index))] + (["simulate"] if simulate else [])
    env = dict(os.environ)
    env["NUMBA_ENABLE_CUDASIM"] = "1" if simulate else "0"
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=float(timeout_s), env=env, check=False)
    except subprocess.TimeoutExpired:
        return DeviceInfo(index=index, reason=f"the device probe did not finish within {timeout_s:g} s "
                                              f"(a hung CUDA driver?)")
    except OSError as exc:
        return DeviceInfo(index=index, reason=f"the device probe could not start ({exc})")
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(PROBE_TAG):
            try:
                return _from_record(json.loads(line[len(PROBE_TAG):]))
            except (ValueError, TypeError) as exc:
                return DeviceInfo(index=index, reason=f"the device probe printed an unreadable result ({exc})")
    tail = (done.stderr.strip().splitlines() or [""])[-1][:300]
    if done.returncode < 0:
        return DeviceInfo(index=index, reason=f"the device probe crashed (signal {-done.returncode}) {tail}".strip())
    return DeviceInfo(index=index, reason=f"the device probe failed (exit {done.returncode}) {tail}".strip())


# ---------------------------------------------------------------------------------------------- the process
_LOCK = threading.RLock()
_INFO: dict[int, DeviceInfo] = {}
_SELECTED: dict[str, int] = {}
LAUNCH_LOCK = threading.RLock()       # one kernel sequence at a time (the simulator supports nothing else)


def disabled_by_env() -> bool:
    return os.environ.get("CNCTWIN_GPU", "").strip().lower() == "off"


def simulator_mode() -> bool:
    """True when numba-cuda runs (or, once imported, will run) the CUDA simulator in this process."""
    module = sys.modules.get("numba.cuda")
    if module is not None:
        config = sys.modules.get("numba.cuda.core.config") or getattr(module, "config", None)
        flag = getattr(config, "ENABLE_CUDASIM", None)
        if flag is not None:
            return bool(flag)
    return os.environ.get("NUMBA_ENABLE_CUDASIM", "0").strip() == "1"


def _simulator_info(index: int) -> DeviceInfo:
    try:
        import numba_cuda
        version = str(getattr(numba_cuda, "__version__", ""))
    except Exception as exc:
        return DeviceInfo(index=index, reason=f"numba-cuda is not installed ({type(exc).__name__}: {exc})")
    try:
        from numba import cuda  # noqa: F401  (imports the simulator: NUMBA_ENABLE_CUDASIM=1)
    except Exception as exc:
        return DeviceInfo(index=index, reason=f"the CUDA simulator cannot be imported ({type(exc).__name__}: {exc})")
    return DeviceInfo(kind="simulator", usable=True, index=index, name="CUDA simulator",
                      reason="CUDA simulator (NUMBA_ENABLE_CUDASIM=1): Python execution, correctness only",
                      numba_cuda_version=version, fma_contraction=False, exact_arithmetic=True)


def info(index: int = 0, refresh: bool = False) -> DeviceInfo:
    """What this process can use for device ``index`` (cached; ``refresh`` probes again)."""
    index = int(index)
    if disabled_by_env():
        return DeviceInfo(index=index, reason="disabled by CNCTWIN_GPU=off")
    with _LOCK:
        if not refresh and index in _INFO:
            return _INFO[index]
        found = _simulator_info(index) if simulator_mode() else probe(index)
        _INFO[index] = found
        return found


def available(index: int = 0, refresh: bool = False) -> tuple[bool, str]:
    """(usable, reason) for the GPU kernels on device ``index`` (the CUDA simulator counts as usable; its
    reason says so). Never raises."""
    found = info(index, refresh)
    if found.usable:
        if found.kind == "simulator":
            return True, found.reason
        cc = ".".join(str(c) for c in found.compute_capability or ())
        return True, f"CUDA device {found.index}: {found.name} (compute capability {cc}, driver {found.driver_version})"
    return False, found.reason or "no CUDA device"


def forget() -> None:
    """Drop the cached detection results (tests)."""
    with _LOCK:
        _INFO.clear()


def cuda_module():
    """numba.cuda (the simulator under NUMBA_ENABLE_CUDASIM=1); raises GpuError when it cannot be imported."""
    try:
        from numba import cuda
    except Exception as exc:          # numba-cuda missing or broken
        raise GpuError(f"numba.cuda cannot be imported ({type(exc).__name__}: {exc})") from exc
    return cuda


def ensure_selected(index: int) -> None:
    """Make device ``index`` current in this process (once; real devices only)."""
    if simulator_mode():
        return
    cuda = cuda_module()
    with _LOCK:
        if _SELECTED.get("index") == index:
            return
        try:
            cuda.select_device(int(index))
        except Exception as exc:
            raise GpuError(f"CUDA device {index} cannot be selected ({type(exc).__name__}: {exc})") from exc
        _SELECTED["index"] = int(index)


@contextmanager
def exact_compile():
    """Compile this package's kernels with NVVM FMA contraction off (``-fma=0``); a no-op in the simulator."""
    if simulator_mode():
        yield
        return
    from numba.cuda.cudadrv import nvvm

    with _LOCK:
        original = nvvm.compile_ir

        def compile_ir(llvmir, **options):
            options.setdefault("fma", False)
            return original(llvmir, **options)

        nvvm.compile_ir = compile_ir
        try:
            yield
        finally:
            nvvm.compile_ir = original


def threads_per_block() -> int:
    return THREADS_PER_BLOCK_SIM if simulator_mode() else THREADS_PER_BLOCK


def grid(n: int) -> tuple[int, int]:
    """(blocks, threads per block) covering n threads (at least one block)."""
    tpb = threads_per_block()
    return max(1, (int(n) + tpb - 1) // tpb), tpb
