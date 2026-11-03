"""Device detection never raises and says why the GPU cannot be used (cnctwin.gpu.device)."""

from __future__ import annotations

import json
import sys

import pytest

from cnctwin.gpu import device as D


def test_probe_of_this_host_is_definite():
    found = D.probe(0)
    assert isinstance(found.usable, bool)
    if found.usable:
        assert found.kind == "cuda" and found.exact_arithmetic is True and found.fma_contraction is False
    else:
        assert found.kind == "none" and found.reason
    json.dumps(found.to_dict())


def test_probe_failures_become_reasons():
    assert "could not start" in D.probe(0, python="/nonexistent/python").reason
    assert "did not finish" in D.probe(0, timeout_s=1e-3).reason
    missing = D.probe(0, python=sys.executable)            # the real path again, but a device index nobody has
    assert isinstance(missing.reason, str)
    many = D.probe(63)
    assert not many.usable and many.reason


def test_simulated_probe_runs_every_step():
    found = D.probe(0, simulate=True)
    assert found.usable and found.kind == "simulator"
    assert found.exact_arithmetic is True and found.fma_contraction is False
    assert found.compute_capability is not None


def test_environment_kill_switch(monkeypatch):
    monkeypatch.setenv("CNCTWIN_GPU", "off")
    ok, reason = D.available(refresh=True)
    assert not ok and "CNCTWIN_GPU=off" in reason
    monkeypatch.delenv("CNCTWIN_GPU")
    D.forget()


def test_session_backend(gpu_kind):
    ok, reason = D.available()
    assert ok and reason
    assert D.simulator_mode() == (gpu_kind == "simulator")
    if gpu_kind == "simulator":
        with D.exact_compile():          # a no-op in the simulator
            pass
        assert D.threads_per_block() == D.THREADS_PER_BLOCK_SIM
    blocks, tpb = D.grid(1000)
    assert blocks * tpb >= 1000 and D.grid(0)[0] == 1


def test_device_info_round_trip():
    record = D.DeviceInfo(kind="cuda", usable=True, index=1, name="GTX 950", compute_capability=(5, 2),
                          supported_ccs=((5, 0), (5, 2)), exact_arithmetic=True, fma_contraction=False).to_dict()
    again = D._from_record(json.loads(json.dumps(record)))
    assert again.compute_capability == (5, 2) and again.supported_ccs == ((5, 0), (5, 2)) and again.index == 1


def test_unreadable_probe_output(monkeypatch):
    class Done:
        returncode = 0
        stdout = D.PROBE_TAG + "{not json"
        stderr = ""

    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: Done())
    assert "unreadable" in D.probe(0).reason

    class Crashed:
        returncode = -11
        stdout = ""
        stderr = "Segmentation fault"

    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: Crashed())
    assert "signal 11" in D.probe(0).reason
