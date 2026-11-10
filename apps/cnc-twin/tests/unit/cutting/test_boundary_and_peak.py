"""The element at the material's boundary along the edge and the instantaneous peak force (docs/models/cutting.md 4
"Engaged length", "Peak"; docs/convergence.md 9 items 1 and 2).

* The element at the depth-of-cut line was engaged whole or not at all: the forces erred by (engaged length - ap) / ap,
  up to half an element's height over ap. Weighted by its engaged length (CutState.st_edge, measured by the hot
  loop's edge_reach from the field; here from the analytic planes), every element size gives the same force per
  unit depth.
* A peak was the largest step mean: 5-11 % low at the default stride. The step's ends give the instantaneous force,
  so the peak of a coarse stride is the fine stride's, and the means do not change.
"""

import math

import numpy as np
import pytest

from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting.drive import Driver, straight_immersion
from tests.unit.cutting.cutfix import ALTINTAS, linear_layout

OMEGA = -1047.0                # 10 000 rpm, M03
R_TOOL = 5e-3
FLUTE = 4e-3                   # edge length of the ideal cutter (its stations above the cut stay out of material)
FZ = 0.05e-3


def _slot(n_z: int, ap: float, weighted: bool, deg_per_step: float = 0.5, immersion=(0.0, math.pi),
          revolutions: float = 1.0, helix_deg: float = 30.0):
    """Mean force of a steady slot of depth ap (or a side cut of the given immersion) over whole revolutions, the
    largest step-mean |F| and the instantaneous peak (CA_FPEAK_INST_N)."""
    tool = AN.cylinder_mill(R_TOOL, 4, FLUTE, n_z, helix_rad=math.radians(helix_deg))
    nt, ne = tool.edge.elem_r_m.shape
    ct, table, _ = linear_layout(tool, {"default": dict(ALTINTAS, h_min_m=0.0)}, OMEGA)
    drv = Driver(ct, table, nt, ne, rho_kg_m3=2700.0, n_ticks=1)
    vf = FZ * 4 * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([vf * t, 0.0, 0.0])
    lo, hi = straight_immersion(*immersion)
    engaged = [s for s in range(drv.nst) if float(ct.st_z_m[s]) < ap]
    drv.set_interval(lo, hi, stations=engaged)
    if weighted:
        drv.set_edge_plane(ap)
    drv.prefill_memory(tip, 0.0, OMEGA)
    dt = math.radians(deg_per_step) / abs(OMEGA)
    n = int(round(revolutions * 360.0 / deg_per_step))
    rec = drv.run_rotating(n, OMEGA, dt, tip)
    F = np.array(rec.wrench)[:, :3]
    return (float(np.linalg.norm(F.mean(axis=0))), float(np.max(np.linalg.norm(F, axis=1))),
            float(drv.cacc.scalars[A.CA_FPEAK_INST_N]), float(drv.acc.scalars[A.ACC_CHIP_VOLUME_M3]))


def test_the_boundary_element_is_weighted_by_its_engaged_length():
    """D10 4-flute 30 deg helix, slot ap 2.86 mm (a depth no element boundary of these sizes meets exactly) with
    elements of 1, 0.5, 0.25 and 0.125 mm: weighted, the mean force per mm of depth and the analytic chip volume agree
    to 0.2 % across the sizes; whole elements spread by several per cent (the 0.5 mm element reaches 3.0 mm: +4.9 %)."""
    ap = 2.86e-3
    sizes = (4, 8, 16, 32)
    whole = [_slot(n, ap, False) for n in sizes]
    weighted = [_slot(n, ap, True) for n in sizes]
    fw = np.array([w[0] for w in weighted])
    vw = np.array([w[3] for w in weighted])
    fu = np.array([w[0] for w in whole])
    assert np.ptp(fw) / fw.mean() < 2e-3
    assert np.ptp(vw) / vw.mean() < 2e-3
    assert np.ptp(fu) / fu.mean() > 0.02
    # the reference: the engaged length is exactly ap, so the force is the per-depth force of a fully engaged depth
    full = _slot(16, 2.0e-3, False)[0] / 2.0e-3            # 2 mm: a station boundary of the 0.25 mm elements
    assert fw.mean() / ap == pytest.approx(full, rel=2e-3)


def test_a_shallow_cut_inside_one_element():
    """A 0.7 mm deep cut with 1 mm elements: one station (z 0.5 mm) is engaged and the material ends inside its
    element: its weight is 0.7 (the engaged length over its own), not 1; 0.125 mm elements give the same force."""
    ap = 0.7e-3
    f1 = _slot(4, ap, True)[0]
    f8 = _slot(32, ap, True)[0]
    whole = _slot(4, ap, False)[0]
    assert f1 == pytest.approx(f8, rel=2e-3)
    assert f1 == pytest.approx(0.7 * whole, rel=1e-6)


def test_the_peak_is_the_instantaneous_force_and_the_means_are_unchanged():
    """Up milling to 45 deg with straight flutes: the force is largest at the exit, where the chip is thickest and the
    tooth leaves the cut within a step. At 2 deg per step the largest step mean is 1.6 % below the converged peak (the
    0.1 deg stride's), the instantaneous peak from the step's ends 0.5 %; the step means do not change."""
    ap = 2.0e-3
    up = (0.0, 0.25 * math.pi)
    mean_c, stepmax_c, inst_c, _ = _slot(16, ap, True, deg_per_step=2.0, immersion=up, helix_deg=0.0)
    mean_f, stepmax_f, inst_f, _ = _slot(16, ap, True, deg_per_step=0.1, immersion=up, helix_deg=0.0)
    assert inst_f == pytest.approx(stepmax_f, rel=1e-3)     # converged: the fine stride's peak
    assert stepmax_c < 0.99 * inst_f                       # the old peak: a step mean
    assert inst_c == pytest.approx(inst_f, rel=6e-3)
    assert inst_c >= stepmax_c
    assert mean_c == pytest.approx(mean_f, rel=1e-3)


def test_a_commensurate_stride_writes_every_bin():
    """At 0.25 or 0.05 deg per step on 0.5 deg memory bins, bin centres fall on step ends; the crossings of
    consecutive steps (computed separately) left some of them outside both by round-off, so the passing tooth did not
    write them and the next tooth read a surface three passes old (h = 3 f_z, the force doubled, the mean +2 %). Every
    stride now gives the 0.1 deg stride's mean and peak, and no chip exceeds f_z."""
    imm = (0.45 * math.pi, math.pi)
    ref = _slot(16, 2e-3, True, deg_per_step=0.1, immersion=imm, helix_deg=0.0)
    for deg in (0.25, 0.05):
        r = _slot(16, 2e-3, True, deg_per_step=deg, immersion=imm, helix_deg=0.0)
        assert r[0] == pytest.approx(ref[0], rel=1e-3)
        assert r[2] == pytest.approx(ref[2], rel=5e-3)


def test_the_end_forces_read_the_table_at_the_end_chip():
    """The tool steps back by 0.99 f_z between two steps (a slot, straight flutes): every chip falls from up to f_z to
    below h_min, where the edge forces ramp up steeply from zero. The forces at that step's start are the table's at
    the chip of that instant (half way back); extrapolating the centroid chip's segment slope there (the first
    version) gave several times the steady peak (the integrated slot's 850 N against 258 N at 1 deg per step)."""
    ap = 2.0e-3
    tool = AN.cylinder_mill(R_TOOL, 4, FLUTE, 16, helix_rad=0.0)
    nt, ne = tool.edge.elem_r_m.shape
    ct, table, _ = linear_layout(tool, {"default": dict(ALTINTAS, h_min_m=2e-6)}, OMEGA)
    assert float(table.h_min_m[0]) == pytest.approx(2e-6)
    vf = FZ * 4 * abs(OMEGA) / (2 * math.pi)
    dt = math.radians(1.0) / abs(OMEGA)
    n = 360
    t_jump = (n // 2 + 0.5) * dt

    def run(jump):
        drv = Driver(ct, table, nt, ne, rho_kg_m3=2700.0, n_ticks=1)
        tip = lambda t: np.array([vf * t - (0.99 * FZ if jump and t > t_jump else 0.0), 0.0, 0.0])
        drv.set_interval(*straight_immersion(0.0, math.pi),
                         stations=[s for s in range(drv.nst) if float(ct.st_z_m[s]) < ap])
        drv.set_edge_plane(ap)
        drv.prefill_memory(tip, 0.0, OMEGA)
        rec = drv.run_rotating(n, OMEGA, dt, tip)
        F = np.linalg.norm(np.array(rec.wrench)[:, :3], axis=1)
        return float(drv.cacc.scalars[A.CA_FPEAK_INST_N]), float(F[: n // 2].max())

    steady_inst, steady_step = run(False)
    jump_inst, jump_step = run(True)
    assert steady_inst == pytest.approx(steady_step, rel=0.05)
    assert jump_step == pytest.approx(steady_step, rel=1e-6)    # the jump only lowers the chips
    assert jump_inst <= 1.02 * steady_inst
