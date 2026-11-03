"""The benchmark says when there is no device, and checks results when it times one (cnctwin.gpu.bench)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from cnctwin.gpu import bench
from cnctwin.gpu import device as D


def test_without_a_device_it_times_the_cpu_and_says_so(monkeypatch):
    none = D.DeviceInfo(reason="the NVIDIA kernel driver is not loaded (/dev/nvidiactl is missing)")
    monkeypatch.setattr(D, "info", lambda index=0, refresh=False: none)
    report = bench.benchmark(sizes="smoke", repeat=1)
    assert report["gpu_timed"] is False and "no usable CUDA device" in report["note"]
    assert set(report["cases"]) == {"dexel_subtract", "thermal_work_step", "chips_advance"}
    for entry in report["cases"].values():
        assert entry["cpu_s"] > 0.0 and entry["items"] > 0
        assert entry["gpu_s"] is None and entry["identical"] is None
    json.dumps(report)


@pytest.mark.gpu
def test_simulator_only_on_request(gpu_kind):
    if gpu_kind != "simulator":
        pytest.skip("a real device: the default benchmark times it")
    report = bench.benchmark(sizes="smoke", repeat=1, kernels=("thermal_work_step",))
    assert report["gpu_timed"] is False and "simulator" in report["note"]
    timed = bench.benchmark(sizes="smoke", repeat=1, include_simulator=True)
    assert timed["gpu_timed"] is True
    for name, entry in timed["cases"].items():
        assert entry["identical"] is True, name
        assert entry["gpu_s"] > 0.0 and entry["max_rel_diff"] == 0.0


def test_relative_difference():
    a = (np.array([1.0, 2.0, np.nan]), np.array([0, 1]))
    assert bench._max_rel(a, (np.array([1.0, 2.0, np.nan]), np.array([0, 1]))) == 0.0
    assert math.isclose(bench._max_rel(np.array([1.0, 2.0]), np.array([1.0, 2.0 * (1 + 1e-12)])), 1e-12,
                        rel_tol=1e-3)
    assert bench._max_rel(np.array([1.0, np.nan]), np.array([1.0, 2.0])) == math.inf
    assert bench._max_rel(np.array([1, 2]), np.array([1, 3])) == math.inf
