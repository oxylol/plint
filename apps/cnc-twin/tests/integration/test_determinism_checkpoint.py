"""(f) Two runs of the same project are bit-identical; (g) a checkpoint between macro steps (the whole SimState
through the core codec and every domain's to_arrays / from_arrays, the hot loop's own state) restored into a
fresh driver continues bit-identically. The slot is cut with the dynamics domain live (machine and spindle modes,
no L2 lobe analysis), so the modal states, the surface memory, the field and the loop's tooth phases all cross
the cut."""

import hashlib

import numpy as np
import pytest

from cnctwin.core.drive import OperationDriver
from tests.integration import hotfix as HF

CUT_AT_S = 4.6          # inside the slot (the tool enters the stock at about 4.2 s)
AFTER_STEPS = 25


def _digest(drv) -> str:
    """Hash of what the steps produced and of the states that carry the cut."""
    h = hashlib.sha256()
    for st in drv.steps:
        for w in st.wrench_T0 + st.wrench_T0_peak + st.torque_Nm:
            h.update(np.ascontiguousarray(w).tobytes())
        h.update(np.float64([st.removed_m3, st.analytic_m3, st.in_cut_s]).tobytes())
    f = drv.state.removal.workpieces[0].field
    for name in ("cnt", "seg", "sev", "bmat"):
        h.update(np.ascontiguousarray(getattr(f, name)).tobytes())
    for cc in sorted(drv.state.cutting.contacts.values(), key=lambda c: c.key):
        h.update(cc.mem.point_S_m.tobytes())
        h.update(cc.kstate.elem_acc.tobytes())
    for p, rec in sorted(drv.state.dynamics.systems.items()):
        h.update(np.ascontiguousarray(rec.x).tobytes())
    return h.hexdigest()


def _steps_until(drv, t_s):
    while drv.state.clock.tick * 1e-6 < t_s - 1e-9:
        drv.step()


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return HF.slot_project(tmp_path_factory.mktemp("det"), 2.0, x1_mm=8.0)


def test_two_runs_are_bit_identical(project):
    digests = []
    for _ in range(2):
        drv = OperationDriver(project, use_dynamics=True, dynamics_l2=False)
        _steps_until(drv, CUT_AT_S)
        for _ in range(AFTER_STEPS):
            drv.step()
        assert any(st.in_cut_s > 0 for st in drv.steps)
        digests.append(_digest(drv))
    assert digests[0] == digests[1]


def test_checkpoint_between_macro_steps_continues_bit_identically(project):
    ref = OperationDriver(project, use_dynamics=True, dynamics_l2=False)
    _steps_until(ref, CUT_AT_S)
    assert ref.steps[-1].in_cut_s > 0.0                  # the cut lies inside the engagement
    ck = ref.checkpoint()
    n_before = len(ref.steps)
    for _ in range(AFTER_STEPS):
        ref.step()
    resumed = OperationDriver(project, use_dynamics=True, dynamics_l2=False)
    resumed.restore(ck)
    resumed.steps = list(ref.steps[:n_before])
    for _ in range(AFTER_STEPS):
        resumed.step()
    for a, b in zip(ref.steps[n_before:], resumed.steps[n_before:]):
        for wa, wb in zip(a.wrench_T0, b.wrench_T0):
            assert np.array_equal(wa, wb)
        assert a.removed_m3 == b.removed_m3 and a.analytic_m3 == b.analytic_m3
    assert _digest(ref) == _digest(resumed)
