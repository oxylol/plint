"""twin check's numerical-setting rules (cnctwin.core.check): the surface memory against the L0 stride and the lattice
against the radial engagement of finishing passes (docs/convergence.md 2 and 4.2)."""

import math
from types import SimpleNamespace

import numpy as np

from cnctwin.core import check as CK
from cnctwin.sanity.context import ToolFacts


def test_memory_bins_limit_is_one_bin_per_l0_step():
    omega = 10000.0 * 2.0 * math.pi / 60.0              # 10 000 rpm
    angle, limit = CK.memory_bins_limit(omega, 8, 1e-6)  # 8 us: 0.48 deg per step (docs/convergence.md 1)
    assert abs(angle - 0.48) < 1e-12
    assert limit == 750
    angle, limit = CK.memory_bins_limit(omega, 25, 1e-6)  # 1.5 deg per step: 240 bins at most
    assert abs(angle - 1.5) < 1e-12 and limit == 240
    assert CK.memory_bins_limit(0.0, 8, 1e-6) == (0.0, None)


def test_memory_bins_warning_only_above_the_limit():
    omega = 10000.0 * 2.0 * math.pi / 60.0
    angle, limit = CK.memory_bins_limit(omega, 8, 1e-6)
    entry = {"tool": 1, "angle_per_step_deg": angle, "memory_bins_max": limit, "rpm": 10000.0, "l0_run_ticks": 8}
    assert CK.memory_bins_warning("OP10", entry, 720) is None        # the template default
    assert CK.memory_bins_warning("OP10", entry, 750) is None        # exactly one bin per step
    text = CK.memory_bins_warning("OP10", entry, 2880)               # the bins_runout level that moved the forces
    assert text is not None and "T1" in text and "2880" in text and "at most 750 bins" in text
    assert CK.memory_bins_warning("OP10", {**entry, "memory_bins_max": None}, 2880) is None


def test_effective_lattice_counts_a_refine_box_only_where_it_covers_the_move():
    boxes = [(np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.02, 0.02]), 4)]
    inside = (np.array([0.01, 0.0, 0.01]), np.array([0.04, 0.01, 0.015]))
    across = (np.array([0.01, 0.0, 0.01]), np.array([0.06, 0.01, 0.015]))
    assert CK.effective_lattice(2.5e-4, inside, boxes) == 2.5e-4 / 4
    assert CK.effective_lattice(2.5e-4, across, boxes) == 2.5e-4
    assert CK.effective_lattice(2.5e-4, None, boxes) == 2.5e-4


def _program(moves):
    """A stand-in for sanity's Program: one straight move per (x0, x1, y) at z = -1 mm with T1 in G54."""
    commands = [SimpleNamespace(modal={"tool": 1, "work_offset": "G54"}, ref=SimpleNamespace(line=10 + i,
                                                                                          program="op.nc"))
                for i in range(len(moves))]
    start = np.array([[x0, y, -1e-3] for x0, _, y in moves])
    end = np.array([[x1, y, -1e-3] for _, x1, y in moves])
    return SimpleNamespace(commands=commands, tip_start_W_m=start, tip_end_W_m=end)


def test_finishing_passes_groups_breaking_moves_and_states_the_volume_bound():
    program = _program([(-8e-3, 30e-3, 0.0)] * 5)
    sctx = SimpleNamespace(tools={1: ToolFacts(1, diameter_m=0.01)}, work_offsets_m={"G54": (0.0, 0.0, 0.0)})
    # moves 0-1 rough (ae 3 mm), 2-3 a finishing pass (ae 0.3 mm), 4 rough again
    ae = np.array([3e-3, 3e-3, 0.3e-3, 0.32e-3, 3e-3])
    eng = SimpleNamespace(ae_m=ae, cut=np.ones(5, bool), ap_m=np.full(5, 1e-3))
    passes = CK.finishing_passes(program, sctx, eng, 2.5e-4, [])
    assert len(passes) == 1
    p = passes[0]
    assert (p["line_first"], p["line_last"], p["moves"]) == (12, 13, 2)
    assert p["ae_m"] == 0.3e-3 and p["lattice_m"] == 2.5e-4
    assert abs(p["required_m"] - 0.075e-3) < 1e-15
    assert p["refine_factor"] == 4                                  # 0.25 / 4 = 0.0625 mm <= 0.075 mm
    assert abs(p["volume_error_bound"] - 2.5e-4 / 0.6e-3) < 1e-12   # lattice / (2 ae): 42 %
    text = CK.pass_warning("OP10", p)
    assert "lines 12-13" in text and "0.300 mm" in text and "0.0750 mm" in text and "factor >= 4" in text
    # the lattice the rule admits: no pass breaks it
    assert CK.finishing_passes(program, sctx, eng, 0.0625e-3, []) == []


def test_a_refine_box_over_the_pass_satisfies_the_rule():
    program = _program([(-8e-3, 30e-3, 0.0)])
    sctx = SimpleNamespace(tools={1: ToolFacts(1, diameter_m=0.01)}, work_offsets_m={"G54": (0.0, 0.0, 0.0)})
    eng = SimpleNamespace(ae_m=np.array([0.3e-3]), cut=np.ones(1, bool), ap_m=np.array([1e-3]))
    box = (np.array([-0.02, -0.01, -0.01]), np.array([0.04, 0.01, 0.01]), 4)
    assert CK.finishing_passes(program, sctx, eng, 2.5e-4, [box]) == []
    short = (np.array([-0.02, -0.01, -0.01]), np.array([0.02, 0.01, 0.01]), 4)   # ends before the pass does
    assert len(CK.finishing_passes(program, sctx, eng, 2.5e-4, [short])) == 1
