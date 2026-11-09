"""CuttingDomain with a fake run context: init on tooling's state, coefficient tables per slot, kernel inputs
for a motion path (memory with the tap's handover rows), macro step (coefficients published), checkpoint."""

import json
import math

import numpy as np
import pytest

from cnctwin.core.bus import Bus, MotionFrame, PathFrame
from cnctwin.core.state import SimState
from cnctwin.core.domain import RunContext
from cnctwin.cutting import abi as A
from cnctwin.cutting.coefficients import LinearModel
from cnctwin.cutting.domain import CuttingDomain
from cnctwin.cutting.state import CuttingState
from cnctwin.tooling import ToolingDomain
from tests.unit.cutting.cutfix import ALTINTAS, HOLDER, TOOLS


def _project():
    return {"physics": {"edge_element_mm": 0.5},
            "machine": {"spindles": [{"id": "main", "kind": "milling"}]},
            "material": {"physical": {"density_kg_m3": 2700.0}},
            "tools": {"list": [{"number": 1, "tool": TOOLS["square_endmill"], "holder": HOLDER, "stickout_mm": 35},
                               {"number": 2, "tool": TOOLS["tap_cutting"], "holder": HOLDER, "stickout_mm": 50}]}}


@pytest.fixture(scope="module")
def run():
    ctx = RunContext(project=_project(), db=None, run_dir=None, seed=1, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10)
    state = SimState()
    ToolingDomain().init(ctx, state)
    dom = CuttingDomain(model_factory=lambda slot, geom: LinearModel({"default": dict(ALTINTAS, h_min_m=2e-6)}))
    dom.init(ctx, state)
    return dom, ctx, state


def _bus(slot, omega=-1047.0):
    n = 10
    p = PathFrame(path=0, tick0=0, ipo_ticks=1000, tick=np.arange(n, dtype=np.int64) * 1000,
                  tip_S_m=np.zeros((n, 3)), tip_M_m=np.zeros((n, 3)), axis_T_M=np.tile([0.0, 0.0, 1.0], (n, 1)),
                  spindle_angle_rad=np.mod(omega * np.arange(n) * 1e-3, 2 * math.pi),
                  spindle_omega_rad_s=np.full(n, omega), feed_vel_M_m_s=np.zeros((n, 3)),
                  rapid=np.zeros(n, bool), tool_slot=slot)
    if hasattr(p, "pose_S_T"):
        p.pose_S_T = np.eye(4)
    bus = Bus()
    bus.motion = MotionFrame(time_s=0.01, tick=n * 1000, paths=[p])
    return bus


def test_init_builds_a_table_per_slot_and_kernel_inputs(run):
    dom, ctx, state = run
    cs = state.cutting
    assert isinstance(cs, CuttingState) and set(cs.tools) == {0, 1}
    for slot in (0, 1):
        dom.update_tool(ctx, state, slot, Bus(), force=True)
        tc = cs.tools[slot]
        assert tc.table is not None and tc.tool is not None
        assert np.all(np.isfinite(tc.table.coef)) and tc.table.coef.shape[0] == len(tc.rows)
    inputs = dom.kernel_inputs(ctx, state, _bus(1))
    assert len(inputs.contacts) == 1
    ci = inputs.contacts[0]
    assert ci.slot == 1 and ci.rotating
    # a tap: one handover row per station, refill planes for every memory row
    nst = ci.tool.st_z_m.shape[0]
    assert ci.mem.point_S_m.shape[0] == A.memory_rows(ci.tool) == 2 * nst
    assert ci.refill.valid.shape[0] == ci.mem.point_S_m.shape[0]
    assert A.is_kernel_layout(ci.tool) and A.is_kernel_layout(ci.mem) and A.is_kernel_layout(ci.st)


def test_macro_step_publishes_coefficients_and_checkpoint_round_trip(run):
    dom, ctx, state = run
    prev, nxt = _bus(0), Bus()
    dom.macro_step(ctx, state, prev, nxt)
    assert 0 in nxt.cutting.coefficients or 1 in nxt.cutting.coefficients
    arrays, meta = state.cutting.to_arrays()
    back = CuttingState.from_arrays({k: np.array(v) for k, v in arrays.items()}, json.loads(json.dumps(meta)))
    for slot, tc in state.cutting.tools.items():
        if tc.table is not None:
            assert np.array_equal(back.tools[slot].table.coef, tc.table.coef)
    assert set(back.contacts) == set(state.cutting.contacts)


def test_stale_ticks_of_a_barely_turning_spindle_stay_in_int64():
    """One revolution plus one tooth period in base ticks; a spindle barely turning (the last steps of a run-down)
    takes the stopped spindle's bound instead of a tick count beyond int64 (it stopped a batch run)."""
    from types import SimpleNamespace
    ctx = SimpleNamespace(tick_s=1e-6)
    assert CuttingDomain._stale_ticks(ctx, 0.0, 4) == 1 << 60
    assert CuttingDomain._stale_ticks(ctx, 1e-30, 4) == 1 << 60
    assert CuttingDomain._stale_ticks(ctx, 5e-324, 4) == 1 << 60
    w = 2.0 * math.pi * 100.0
    assert abs(CuttingDomain._stale_ticks(ctx, w, 4) - 12500) <= 1
    istep = np.zeros(4, np.int64)
    istep[0] = CuttingDomain._stale_ticks(ctx, 1e-15, 2)      # assignable to the kernel's int64 step array
