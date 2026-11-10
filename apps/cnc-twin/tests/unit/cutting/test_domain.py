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


def test_the_quantised_slow_state_keeps_its_cell_near_a_boundary():
    """Hysteresis of the rebuild grid (docs/models/cutting.md 1): a value keeps the row's last cell while it stays
    within 0.75 cell of that cell's centre, and is snapped afresh beyond. Plain rounding switched a temperature lying
    at a cell boundary (rounding 395.1 -> 400, 394.9 -> 390 with 10 K cells) between two cells from one macro step to
    the next: with the slow feedback's one-step stagger a limit cycle (the coupling-lag monitor flagged it)."""
    from cnctwin.cutting.coefficients import RowState, Tolerances, quantised_state
    tol = Tolerances(temp_K=10.0, speed_rel=0.02, VB_m=2e-6)
    s0 = quantised_state(RowState(speed_m_s=5.0, tool_temp_K=395.1, flank_temp_K=394.9, VB_m=3.1e-6), tol)
    assert s0.tool_temp_K == 400.0 and s0.flank_temp_K == 390.0
    seq = [394.9, 395.1, 394.8, 395.3, 393.0]
    prev = s0
    cells = []
    for T in seq:
        prev = quantised_state(RowState(speed_m_s=5.0, tool_temp_K=T, flank_temp_K=T, VB_m=3.1e-6), tol, prev)
        cells.append((prev.tool_temp_K, prev.flank_temp_K))
    assert all(c == (400.0, 390.0) for c in cells)                 # within 0.75 cell: no switching at 395
    far = quantised_state(RowState(speed_m_s=5.0, tool_temp_K=392.4, flank_temp_K=398.0, VB_m=3.1e-6), tol, prev)
    assert far.tool_temp_K == 390.0 and far.flank_temp_K == 400.0  # beyond 0.75 cell: snapped afresh
    # log grids (speed) likewise
    v = quantised_state(RowState(speed_m_s=5.0), tol)
    near = quantised_state(RowState(speed_m_s=v.speed_m_s * 1.014), tol, v)
    assert near.speed_m_s == v.speed_m_s
    beyond = quantised_state(RowState(speed_m_s=v.speed_m_s * 1.016), tol, v)
    assert beyond.speed_m_s != v.speed_m_s


def test_the_flank_partition_reads_the_entry_temperature(run):
    """mechanics' flank partition takes thermal's flank temperature at the entry into the cut (flank_entry_K, the
    background of its stationary-source partition), not the in-cut flank temperature (which already holds the step's
    own rise: it was counted twice)."""
    from cnctwin.core.bus import ToolHeat
    dom, ctx, state = run
    dom.update_tool(ctx, state, 0, Bus(), force=True)
    tc = state.cutting.tools[0]
    edge = state.tooling.effective[0]
    shape = np.asarray(edge.elem_r_m).shape
    heat = ToolHeat(flank_temp_q_K=np.full(shape + (1,), 900.0), q_weights=np.ones(1),
                    rake_temp_mean_K=np.full(shape, 500.0))
    heat.flank_entry_K = np.full(shape, 450.0)
    prev = _bus(0)
    prev.thermal.tools = {0: heat}
    members = (np.asarray(edge.elem_valid) != 0) & (np.asarray(tc.elem_row) == 0)
    geom = tc.rows_geom[0]
    rs = dom._row_state(ctx, state, 0, 0, geom, prev, 1047.0, 0.0, members, True)
    assert rs.flank_temp_K == pytest.approx(450.0) and rs.tool_temp_K == pytest.approx(500.0)
    heat.flank_entry_K = np.zeros((0, 0))                          # a record without it: the in-cut mean
    rs = dom._row_state(ctx, state, 0, 0, geom, prev, 1047.0, 0.0, members, True)
    assert rs.flank_temp_K == pytest.approx(900.0)


def _monitor_bus(state, slot, rows, weight_row):
    """A previous step in which the elements at the edge positions where every tooth's element is in table row
    ``weight_row`` cut (1 ms each; the weights are summed over the teeth per position)."""
    from cnctwin.core.bus import CuttingContact
    prev = _bus(slot)
    rows = np.asarray(rows)
    ic = np.where(np.all(rows == weight_row, axis=0, keepdims=True) & (rows == weight_row), 1e-3, 0.0)
    prev.cutting.contacts = [CuttingContact(contact=0, tool_slot=slot, in_cut_s=0.01, in_cut_time_s=ic)]
    return prev


def test_the_coupling_monitor_follows_the_cut_weighted_coefficient(run):
    """The coefficient monitor takes the tangential coefficient averaged over the rows with their elements' in-cut
    time, under the loop's rule (a change counts above coupling_rel_change, 1e-3; three alternations in a row flag):
    +-0.05 % flips (the quantised slow state between two cells) and the rows the cut did not use never flag; a
    sustained +-1 % alternation does, once."""
    dom, ctx, state = run
    slot = 0
    dom.update_tool(ctx, state, slot, Bus(), force=True)
    tc = state.cutting.tools[slot]
    rows = np.asarray(tc.elem_row)
    used = int(np.bincount(rows[rows >= 0].ravel()).argmax())
    prev = _monitor_bus(state, slot, rows, used)
    coef = tc.table.coef
    base = coef.copy()
    n_rows = coef.shape[0]

    def warnings():
        return [f for f in state.findings if f.domain == "cutting" and "alternates" in f.message]

    before = len(warnings())
    tc.lag, tc.ref_coeff = [], 0.0
    for k in range(12):                      # small flips of the used row, large ones of an idle row
        coef[:] = base
        coef[used] *= 1.0 + (5e-4 if k % 2 else -5e-4)
        for r in range(n_rows):
            if r != used:
                coef[r] *= 1.0 + (0.2 if k % 2 else -0.2)
        dom._lag_monitor(ctx, state, tc, slot, prev)
    assert len(warnings()) == before
    tc.lag, tc.ref_coeff = [], 0.0
    state.cutting.raised.pop(f"lag:{slot}", None)
    for k in range(8):
        coef[:] = base
        coef[used] *= 1.0 + (1e-2 if k % 2 else -1e-2)
        dom._lag_monitor(ctx, state, tc, slot, prev)
    assert len(warnings()) == before + 1
    assert "macro_ipo" in warnings()[-1].message and "lag_macro" not in warnings()[-1].message
    coef[:] = base
    idle = Bus()                             # a step without cut leaves the monitor as it was
    lag = list(tc.lag)
    dom._lag_monitor(ctx, state, tc, slot, idle)
    assert tc.lag == lag
