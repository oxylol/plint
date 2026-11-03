"""run.json ``gpu``: the synthetic run directory carries the record cnctwin.gpu writes (architecture 16,
docs/models/gpu.md "Run record")."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cnctwin.gpu.accel import KERNELS, Accelerator
from cnctwin.gpu.device import DeviceInfo
from cnctwin.gpu.schema import GpuSettings

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_synthetic_run_json_has_the_gpu_record(tmp_path):
    sys.path.insert(0, str(FIXTURES))
    try:
        from make_synthetic_run import make_synthetic_run
    finally:
        sys.path.remove(str(FIXTURES))
    make_synthetic_run(tmp_path / "run", seed=3, media=False)
    document = json.loads((tmp_path / "run" / "run.json").read_text())
    record = document["gpu"]
    expected = Accelerator(GpuSettings()).run_info(device=DeviceInfo(reason="x"))
    assert set(record) == set(expected)
    assert set(record["kernels"]) == set(KERNELS)
    assert set(record["device"]) == set(expected["device"])
    assert record["device"]["usable"] is False and record["device"]["reason"]
    assert all(k["gpu_calls"] == 0 and k["parity"] for k in record["kernels"].values())
    again = tmp_path / "again"
    make_synthetic_run(again, seed=3, media=False)
    assert json.loads((again / "run.json").read_text())["gpu"] == record          # deterministic
