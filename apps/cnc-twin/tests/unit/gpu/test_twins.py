"""One source, two targets: twins of njit and tagged functions (cnctwin.gpu.twins)."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
from numba import njit

from cnctwin.gpu import device as D
from cnctwin.gpu.twins import Builder, is_njit, is_tagged, names_used, twin


@njit("float64(float64, float64)", cache=False)
def _scaled_hypot(a, b):
    return math.sqrt(a * a + b * b)


@njit("float64(float64)", cache=False)
def _outer(a):
    return 2.0 * _scaled_hypot(a, 1.0)


@twin
def _plain(x, out):
    out[0] = _outer(x) + np.floor(x)


def _kernel(xs, out):
    i = cuda.grid(1)  # noqa: F821  (bound by the builder)
    if i < xs.shape[0]:
        out[i] = _outer(xs[i]) + np.floor(xs[i])


def test_tags_and_names():
    assert is_njit(_outer) and not is_njit(_plain)
    assert is_tagged(_plain) and not is_tagged(_kernel)
    assert {"_outer", "np"} <= names_used(_plain.__code__)
    assert _plain(0.0, np.zeros(1)) is None                       # a tagged function stays plain Python


def test_cpu_twin_matches_python():
    b = Builder("cpu")
    compiled = b.build(_plain)
    out_c = np.zeros(1)
    out_p = np.zeros(1)
    compiled(1.75, out_c)
    _plain(1.75, out_p)
    assert out_c[0] == out_p[0] == 2.0 * math.sqrt(1.75 ** 2 + 1.0) + 1.0
    assert b.build(_plain) is compiled                             # once per builder
    assert b.build(_outer) is _outer                               # the cpu target keeps njit callees


def test_unknown_target():
    with pytest.raises(ValueError):
        Builder("tpu")
    with pytest.raises(ValueError):
        Builder("cuda")


@pytest.mark.gpu
def test_cuda_twins_rebind_the_whole_closure(gpu_kind):
    cuda = D.cuda_module()
    b = Builder("cuda", cuda, D.simulator_mode())
    kernel = b.kernel(_kernel, "void(float64[::1], float64[::1])")
    xs = np.array([0.0, 0.5, 1.5, -2.25])
    d_out = cuda.to_device(np.zeros(4))
    kernel[1, 32](cuda.to_device(xs), d_out)
    expected = np.array([2.0 * math.sqrt(x * x + 1.0) + math.floor(x) for x in xs])
    assert np.array_equal(d_out.copy_to_host(), expected)
    ns = b.namespace(_outer.py_func.__globals__)
    assert ns["_outer"] is not _outer and ns["_scaled_hypot"] is not _scaled_hypot
    assert ns["cuda"] is cuda


@twin
def _plain_hypot(a, b):
    return math.sqrt(a * a + b * b)


def _uses_module(a):
    return holder.inner(a, a)  # noqa: F821  (``holder`` is bound by the test)


def test_module_attribute_callees_get_proxies():
    holder = types.ModuleType("holder")
    holder.inner = _plain_hypot
    fn = twin(types.FunctionType(_uses_module.__code__, {"holder": holder, "math": math, "__name__": __name__},
                                 "_uses_module"))
    b = Builder("cpu")
    assert b.build(fn)(3.0) == math.sqrt(18.0)
    proxy = b.namespace(fn.__globals__)["holder"]
    assert proxy is not holder and proxy.inner is not _plain_hypot and holder.inner is _plain_hypot
