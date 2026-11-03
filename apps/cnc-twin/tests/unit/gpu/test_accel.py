"""Backend choice, forced backends and the run record (cnctwin.gpu.accel)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cnctwin.gpu import accel as M
from cnctwin.gpu import device as D
from cnctwin.gpu.schema import GpuSettings

CUDA = D.DeviceInfo(kind="cuda", usable=True, name="GTX 950", compute_capability=(5, 2), driver_version="12.2")
SIM = D.DeviceInfo(kind="simulator", usable=True, reason="CUDA simulator")
NONE = D.DeviceInfo(kind="none", usable=False, reason="the NVIDIA kernel driver is not loaded")


def accel_with(monkeypatch, device, **settings):
    monkeypatch.setattr(D, "info", lambda index=0, refresh=False: device)
    monkeypatch.delenv("CNCTWIN_GPU", raising=False)
    return M.Accelerator(GpuSettings(**settings))


def test_off_never_probes(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("probed")

    monkeypatch.setattr(D, "info", boom)
    a = M.Accelerator(GpuSettings(mode="off"))
    assert a.choose("thermal_work_step", 10 ** 9) == ("cpu", "physics.gpu: off")
    assert a.run_info()["device"]["reason"].startswith("not probed")


def test_on_uses_any_usable_device(monkeypatch):
    assert accel_with(monkeypatch, CUDA, mode="on").choose("dexel_subtract", 1)[0] == "gpu"
    assert accel_with(monkeypatch, SIM, mode="on").choose("dexel_subtract", 1)[0] == "gpu"
    backend, reason = accel_with(monkeypatch, NONE, mode="on").choose("dexel_subtract", 1)
    assert backend == "cpu" and "kernel driver" in reason


def test_auto_uses_real_devices_above_the_thresholds(monkeypatch):
    a = accel_with(monkeypatch, CUDA, mode="auto", min_rays=100, min_cells=1000, min_particles=10)
    assert a.choose("dexel_subtract", 99)[0] == "cpu"
    assert a.choose("dexel_subtract", 100)[0] == "gpu"
    assert a.choose("thermal_work_step", lambda: 5000)[0] == "gpu"
    assert a.choose("chips_advance", 9) == ("cpu", "auto: 9 particles < 10")
    assert a.choose("dexel_subtract", 10 ** 6, slabs=3) == ("cpu", "auto: 3 slabs < 24")
    assert a.choose("dexel_subtract", 10 ** 6, slabs=24)[0] == "gpu"
    assert accel_with(monkeypatch, SIM, mode="auto").choose("chips_advance", 10 ** 6)[0] == "cpu"
    assert accel_with(monkeypatch, NONE, mode="auto").choose("chips_advance", 10 ** 6)[0] == "cpu"


def test_kill_switch_and_disabled_run(monkeypatch):
    a = accel_with(monkeypatch, CUDA, mode="on")
    monkeypatch.setenv("CNCTWIN_GPU", "off")
    assert a.choose("dexel_subtract", 1) == ("cpu", "disabled by CNCTWIN_GPU=off")
    monkeypatch.delenv("CNCTWIN_GPU")
    a.disabled_reason = "out of memory"
    backend, reason = a.choose("dexel_subtract", 1)
    assert backend == "cpu" and "out of memory" in reason


def test_forced_backends(monkeypatch):
    a = accel_with(monkeypatch, NONE, mode="auto")
    calls = []
    assert a._run("chips_advance", 3, lambda: (calls.append("gpu"), 0), lambda: calls.append("cpu") or 7,
                  backend="gpu") == 7
    assert calls == ["cpu"] and "no usable CUDA device" in a.usage["chips_advance"].last_reason
    assert a._run("chips_advance", 3, lambda: ("gpu-result", 0), lambda: "cpu-result", backend="cpu") == "cpu-result"
    with pytest.raises(ValueError):
        a._run("chips_advance", 3, lambda: (1, 0), lambda: 2, backend="tpu")


def test_gpu_failure_falls_back_once_and_is_recorded(monkeypatch):
    a = accel_with(monkeypatch, CUDA, mode="on")
    monkeypatch.setattr(D, "ensure_selected", lambda index: None)

    def failing():
        raise D.GpuError("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES")

    assert a._run("dexel_subtract", 5, failing, lambda: 1) == 1
    assert a._run("dexel_subtract", 5, lambda: (0, 0), lambda: 2) == 2     # stays on the CPU
    usage = a.usage["dexel_subtract"]
    assert (usage.fallbacks, usage.cpu_calls, usage.gpu_calls, usage.cpu_items) == (1, 2, 0, 10)
    info = a.run_info()
    assert info["errors"] == ["dexel_subtract: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"]
    json.dumps(info)


def test_run_record_shape(monkeypatch):
    a = accel_with(monkeypatch, CUDA, mode="auto")
    info = a.run_info()
    assert set(info) == {"mode", "device_index", "thresholds", "device", "kernels", "disabled_reason", "errors"}
    assert set(info["kernels"]) == set(M.KERNELS)
    for name, record in info["kernels"].items():
        assert record["parity"] == M.PARITY[name] and record["gpu_calls"] == 0


def test_for_context_attaches_one_accelerator():
    ctx = SimpleNamespace(project={"physics": {"gpu": {"mode": "off", "min_cells": 5}}})
    a = M.for_context(ctx)
    assert a is M.for_context(ctx) and ctx.gpu is a and a.settings.min_cells == 5
    assert M.for_context(SimpleNamespace(project=None)).settings == GpuSettings()
    assert M.default() is M.default()


def test_warmup_without_a_device(monkeypatch):
    a = accel_with(monkeypatch, NONE, mode="on")
    assert all("no usable CUDA device" in r for r in a.warmup().values())


def test_a_gpu_module_that_cannot_be_imported_means_the_cpu(monkeypatch):
    from cnctwin.gpu import cases
    from cnctwin.removal import k_subtract as KS

    a = accel_with(monkeypatch, CUDA, mode="on")
    monkeypatch.setattr(D, "ensure_selected", lambda index: None)

    def missing(name):
        raise D.GpuError(f"cnctwin.gpu.{name} cannot be imported (KeyError: '_env_ray')")

    monkeypatch.setattr(M, "gpu_module", missing)
    case = cases.dexel_case(cuts=2, refine=False, seed=9)
    ref = cases.dexel_run(case, KS.subtract_envelope)
    got = cases.dexel_run(case, a.subtract_envelope)
    assert all((x == y).all() for x, y in zip(ref[0], got[0])) and ref[3] == got[3]
    usage = a.usage["dexel_subtract"]
    assert usage.fallbacks == 1 and usage.cpu_calls == 2 and "_env_ray" in a.disabled_reason
