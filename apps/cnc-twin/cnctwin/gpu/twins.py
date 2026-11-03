"""CPU and CUDA twins of Numba functions: one Python source, two targets (docs/models/gpu.md "One source").

The GPU kernels run the CPU kernels' own code wherever the algorithm is per item: a domain's njit helper
(its Python source, ``py_func``) is compiled a second time as a CUDA device function, with every global it
calls rebound to that callee's twin, recursively (across modules: a module global whose attributes are
called gets a proxy module holding the twins). So a change to, say, removal's envelope geometry reaches the
GPU path without anyone porting it, and both targets round alike (the CUDA side is compiled with FMA
contraction off, :func:`cnctwin.gpu.device.exact_compile`).

Functions of this package written for both targets are plain Python tagged :func:`twin`; they are
compiled with ``numba.njit`` for the CPU or as CUDA device functions from the same source.

On a real device ``numpy`` is replaced inside device code by a proxy whose ``floor``, ``ceil``, ``trunc``,
``sqrt``, ``fabs``/``abs`` and ``isnan``/``isinf``/``isfinite`` are libdevice / math twins (numba-cuda does
not lower these NumPy ufuncs on scalars); each is exact (IEEE), so values are unchanged. ``cuda`` is
injected into every rebound namespace (this package imports numba.cuda lazily).
"""

from __future__ import annotations

import math
import threading
import types

import numpy as np

_TAG = "_cnctwin_twin"


def twin(fn):
    """Mark a plain Python function as compiled for both targets (left unchanged: it stays callable)."""
    setattr(fn, _TAG, True)
    return fn


def is_njit(obj) -> bool:
    """A Numba CPU dispatcher (duck-typed: numba-cuda vendors its own copy of numba.core)."""
    return callable(obj) and getattr(obj, "py_func", None) is not None and hasattr(obj, "targetoptions")


def is_tagged(obj) -> bool:
    return isinstance(obj, types.FunctionType) and bool(getattr(obj, _TAG, False))


def _source(obj):
    return obj.py_func if is_njit(obj) else obj


def names_used(code: types.CodeType) -> set[str]:
    """Every name a code object (and the code objects nested in it) looks up."""
    out = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            out |= names_used(const)
    return out


# numpy scalar functions for device code; ``libdevice`` is bound when the proxy is built (real devices only)
def _np_floor(x):
    return libdevice.floor(x)  # noqa: F821


def _np_ceil(x):
    return libdevice.ceil(x)  # noqa: F821


def _np_trunc(x):
    return libdevice.trunc(x)  # noqa: F821


def _np_sqrt(x):
    return libdevice.sqrt(x)  # noqa: F821


def _np_fabs(x):
    return libdevice.fabs(x)  # noqa: F821


def _np_isnan(x):
    return math.isnan(x)


def _np_isinf(x):
    return math.isinf(x)


def _np_isfinite(x):
    return math.isfinite(x)


def _numpy_proxy(cuda):
    """numpy for device code: the scalar ufuncs numba-cuda does not lower, as exact libdevice twins."""
    from numba.cuda import libdevice

    proxy = types.ModuleType("numpy")
    proxy.__dict__.update(np.__dict__)
    names = {"libdevice": libdevice, "math": math}
    for fn in (_np_floor, _np_ceil, _np_trunc, _np_sqrt, _np_fabs, _np_isnan, _np_isinf, _np_isfinite):
        local = types.FunctionType(fn.__code__, names, fn.__name__)
        setattr(proxy, fn.__name__[4:], cuda.jit(device=True)(local))
    proxy.abs = proxy.fabs
    proxy.absolute = proxy.fabs
    return proxy


class Builder:
    """Compiles functions and their dependency closure for one target: ``cpu`` (numba.njit; only tagged
    functions are compiled, njit callees are used as they are) or ``cuda`` (device functions; njit callees
    get twins too). ``simulator``: numba-cuda's CUDA simulator (no numpy proxy needed)."""

    def __init__(self, target: str, cuda=None, simulator: bool = False):
        if target not in ("cpu", "cuda"):
            raise ValueError(f"unknown target {target!r}")
        if target == "cuda" and cuda is None:
            raise ValueError("the cuda target needs the numba.cuda module")
        self.target = target
        self.cuda = cuda
        self.simulator = bool(simulator)
        self._compiled: dict[int, object] = {}
        self._namespaces: dict[int, dict] = {}
        self._modules: dict[int, types.ModuleType] = {}
        self._np = None

    def needs(self, obj) -> bool:
        if is_tagged(obj):
            return True
        return self.target == "cuda" and is_njit(obj)

    def namespace(self, source_globals: dict) -> dict:
        """The rebound copy of a module's globals (one per module, shared by its functions)."""
        key = id(source_globals)
        found = self._namespaces.get(key)
        if found is None:
            found = dict(source_globals)
            if self.target == "cuda":
                found["cuda"] = self.cuda
            self._namespaces[key] = found
        return found

    def _module(self, module: types.ModuleType, used: set[str]):
        if module is np and self.target == "cuda" and not self.simulator:
            if self._np is None:
                self._np = _numpy_proxy(self.cuda)
            return self._np
        attrs = [name for name in sorted(used) if self.needs(getattr(module, name, None))]
        if not attrs:
            return module
        proxy = self._modules.get(id(module))
        if proxy is None:
            proxy = types.ModuleType(module.__name__)
            proxy.__dict__.update(module.__dict__)
            self._modules[id(module)] = proxy
        for name in attrs:
            setattr(proxy, name, self.build(getattr(module, name)))
        return proxy

    def _copy(self, fn) -> types.FunctionType:
        src = _source(fn)
        copy = types.FunctionType(src.__code__, self.namespace(src.__globals__), src.__name__, src.__defaults__,
                                  src.__closure__)
        copy.__kwdefaults__ = src.__kwdefaults__
        copy.__qualname__ = src.__qualname__
        return copy

    def _populate(self, fn) -> None:
        """Point the names ``fn`` looks up at the twins of its callees (and proxies of their modules)."""
        src = _source(fn)
        ns = self.namespace(src.__globals__)
        used = names_used(src.__code__)
        for name in sorted(used):
            value = src.__globals__.get(name)
            if value is None:
                continue
            if self.needs(value):
                ns[name] = self.build(value)
            elif isinstance(value, types.ModuleType):
                ns[name] = self._module(value, used)

    def rebind(self, fn) -> types.FunctionType:
        """A copy of ``fn`` (plain function or njit dispatcher) whose globals name the twins of its callees."""
        copy = self._copy(fn)
        self._populate(fn)
        return copy

    def build(self, fn):
        """The twin of one function (device function for cuda, njit for cpu)."""
        key = id(fn)
        found = self._compiled.get(key)
        if found is not None:
            return found
        if not self.needs(fn):
            return fn
        copy = self._copy(fn)
        if self.target == "cpu":
            from numba import njit
            compiled = njit(nogil=True, error_model="numpy", cache=True)(copy)
        else:
            compiled = self.cuda.jit(device=True)(copy)
        self._compiled[key] = compiled       # before the callees: cycles end here
        self._populate(fn)
        return compiled

    def kernel(self, fn, signature):
        """A CUDA kernel (``cuda.jit(signature)``) or CPU entry (``njit(signature)``) from ``fn``'s source."""
        copy = self.rebind(fn)
        if self.target == "cpu":
            from numba import njit
            return njit(signature, nogil=True, error_model="numpy", cache=True)(copy)
        return self.cuda.jit(signature)(copy)


_BUILDERS: dict[tuple[str, bool], Builder] = {}
_LOCK = threading.RLock()


def builder(target: str, cuda=None, simulator: bool = False) -> Builder:
    """The process-wide builder of a target (twins are compiled once per process)."""
    with _LOCK:
        key = (target, bool(simulator) if target == "cuda" else False)
        found = _BUILDERS.get(key)
        if found is None:
            found = Builder(target, cuda, simulator)
            _BUILDERS[key] = found
        return found
