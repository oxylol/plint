"""A polar overflow in the middle of a turning step (docs/models/hotloop.md 6, docs/models/loop.md 15). With one
inline segment per polar ray (physics.removal.polar_kp 1) the feed marks of the first revolutions need more: the axial
rays at the height of the nose-radius scallops cross material and air in turn. Removal's polar cut then refuses
the step (PI_OVERFLOW), the wrapper grows the field and the hot loop repeats only that step's removal
(`pending_removal`) at the resume. Before 2026-11-09 the whole L0 step was repeated: its forces and work were
booked twice and dynamics was stepped by 0 base ticks (a TwinError that stopped warmup's lathe pass). The pass with
polar_kp 1 must reproduce the pass that never overflows (polar_kp 4): the same removed volume, the same cutting
energy, the same forces and modal response, to round-off. Until 2026-11-09 it did not: the polar rows were sized for
the memory budget with the configured capacity, so polar_kp 16 coarsened them (6.25 -> 75 um) and the passes
differed from the first contact (removal.polar.SIZING_KP now)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from cnctwin.core import hot_abi as H
from cnctwin.core.drive import OperationDriver
from tests.integration import hotfix as HF

D_MM, L_MM, AP_MM, Z1_MM = 45.0, 60.0, 1.5, -3.0


def _run(tmp, kp: int):
    project = HF.turn_project(tmp, D_MM, L_MM, AP_MM, Z1_MM)
    data = yaml.safe_load(project.read_text())
    data["physics"].setdefault("removal", {})["polar_kp"] = int(kp)
    project.write_text(yaml.safe_dump(data, sort_keys=False))
    drv = OperationDriver(project, use_dynamics=True)
    drv.hot.keep_reports = True
    res = HF.run_until_done(drv)
    splits = sum(int(rep.splits.get(H.SPLIT_POLAR, 0)) for rep in drv.hot.reports)
    return drv, res, splits


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("turn-split")
    return _run(tmp / "kp1", 1), _run(tmp / "kp4", 4)


def test_the_overflowing_pass_splits_and_finishes(runs):
    (drv, res, splits), _ = runs
    assert splits >= 1, "polar_kp 1 is expected to overflow during the pass"
    assert res.finished
    assert res.ledger["time"]["ok"] and res.ledger["time"]["residual_ticks"] == 0
    en = res.ledger["energy_run"]
    assert abs(en["in_J"] - en["out_J"]) <= 1e-9 * en["abs_J"]
    assert res.removed_volume_m3 == pytest.approx(res.field_volume_change_m3, rel=1e-12)


def test_the_split_pass_reproduces_the_pass_without_overflow(runs):
    (d1, r1, s1), (d2, r2, s2) = runs
    assert s2 == 0
    assert r1.removed_volume_m3 == pytest.approx(r2.removed_volume_m3, rel=1e-12)
    e1, e2 = r1.ledger["energy_run"], r2.ledger["energy_run"]
    assert e1["in_J"] == pytest.approx(e2["in_J"], rel=1e-9)
    assert len(d1.steps) == len(d2.steps)
    for a, b in zip(d1.steps, d2.steps):
        for wa, wb in zip(a.wrench_T0, b.wrench_T0):
            assert np.allclose(wa, wb, rtol=1e-9, atol=1e-9)
