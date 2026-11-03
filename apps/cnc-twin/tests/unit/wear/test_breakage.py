"""Breakage of tool sections: the overload threshold (median Weibull strength of the section's effective
volume), torsion, chip-packing spikes, Basquin fatigue with Miner's sum, the tool_breakage finding with
its tick and cause chain, and a broken tool that stops cutting and is reported on the bus."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.wear import wearfix as fx
from cnctwin.core.bus import ChipFeedback
from cnctwin.core.findings import Category, Severity
from cnctwin.wear import fracture as FR
from cnctwin.wear.domain import WearDomain


def _setup():
    proj = fx.project()
    ctx = fx.context(proj)
    state = fx.sim_state()
    edge = fx.edge_model()
    dom = WearDomain(tools=fx.tool_info(edge), response=fx.AnalyticResponse(), catalog=fx.Catalog())
    dom.init(ctx, state)
    return dom, ctx, state, edge


def _run(dom, ctx, state, edge, i, moment=0.0, torque=0.0, chips=None, entries=4, duty=0.25):
    contact = fx.cutting(edge, duty=duty, entries=entries, section_moment=[moment, 0.0], torque=torque)
    prev = fx.bus(edge, [contact], fx.heat(edge), fx.path_frame(tick0=i * 10000), chips=chips)
    return fx.step(dom, ctx, state, prev)


def test_bending_overload_threshold_is_the_median_section_strength():
    dom, ctx, state, edge = _setup()
    sw = state.wear.slots[0]
    secs = sw.sections
    m = float(sw.chip.scal[0])
    sigma0 = FR.characteristic_strength_Pa(4.0e9, m)
    median = FR.strength_Pa(secs["volume_b_m3"][0], sigma0, FR.reference_volume_m3(m), m, 0.5)
    assert secs["strength_b_Pa"][0] == pytest.approx(median, rel=1e-12)
    threshold = median * secs["Zb_m3"][0]
    record = _run(dom, ctx, state, edge, 0, moment=0.98 * threshold)
    assert not record.tools[0].broken and record.broken == []
    warnings = [f for f in state.findings if f.category == Category.TOOL_BREAKAGE]
    assert warnings and warnings[0].severity == Severity.WARNING          # high fracture probability, not broken
    record = _run(dom, ctx, state, edge, 1, moment=1.02 * threshold)
    tool = record.tools[0]
    assert tool.broken and record.broken == [1] and 1 in record.expired
    assert tool.broken_section_z_m == pytest.approx(0.022) and tool.length_loss_m == pytest.approx(0.022)
    errors = [f for f in state.findings if f.severity == Severity.ERROR]
    assert len(errors) == 1 and errors[0].category == Category.TOOL_BREAKAGE
    finding = errors[0]
    assert finding.tick == tool.broken_tick == 10000 + 9000
    chain = [ctx.trace.get(i).event for i in finding.cause_chain]
    assert any(e.startswith("wear:tool_fracture:bending") for e in chain)
    assert any(e.startswith("wear:law:") for e in chain)
    assert finding.quantities["equivalent_stress_Pa"] == pytest.approx(1.02 * median, rel=1e-9)


def test_torsion_uses_the_torsion_volume_and_the_principal_stress():
    dom, ctx, state, edge = _setup()
    secs = state.wear.slots[0].sections
    assert secs["strength_t_Pa"][0] < secs["strength_b_Pa"][0]      # the whole surface carries shear: weaker
    threshold = secs["strength_t_Pa"][0] * secs["Zp_m3"][0]
    record = _run(dom, ctx, state, edge, 0, torque=0.97 * threshold)
    assert not record.tools[0].broken
    record = _run(dom, ctx, state, edge, 1, torque=1.03 * threshold)
    assert record.tools[0].broken and state.wear.slots[0].life["broken_cause"] == "torsion"


def test_a_chip_packing_spike_breaks_the_tool():
    dom, ctx, state, edge = _setup()
    secs = state.wear.slots[0].sections
    threshold = secs["strength_t_Pa"][0] * secs["Zp_m3"][0]
    chips = ChipFeedback(packing_torque_Nm={0: 1.05 * threshold}, packing_force_T0_N={0: np.zeros(3)})
    record = _run(dom, ctx, state, edge, 0, torque=0.1 * threshold, chips=chips)
    assert record.tools[0].broken and state.wear.slots[0].life["broken_cause"] == "packing"
    error = [f for f in state.findings if f.severity == Severity.ERROR][0]
    assert "chip packing" in error.message


def test_basquin_fatigue_breaks_the_tool_after_the_predicted_cycles():
    dom, ctx, state, edge = _setup()
    sw = state.wear.slots[0]
    secs = sw.sections
    sigma = 0.85 * secs["strength_b_Pa"][0]
    moment = sigma * secs["Zb_m3"][0]
    basquin = FR.basquin("carbide", 4.0e9, 5.8e11)
    assert basquin.b == pytest.approx(math.log(0.7) / math.log(2e7))
    cycles_to_failure = basquin.cycles(FR.swt_amplitude(sigma, rotating_bending=True))
    per_step = 1.0 * 0.01 * 1047.2 / (2 * math.pi)           # continuous cut: revolutions in cut per step
    expected = math.ceil(cycles_to_failure / per_step)
    assert 20 < expected < 400
    for i in range(expected + 2):
        record = _run(dom, ctx, state, edge, i, moment=moment, entries=0, duty=1.0)
        if record.tools[0].broken:
            break
    assert record.tools[0].broken and sw.life["broken_cause"] == "fatigue"
    assert i + 1 == pytest.approx(expected, abs=1)
    assert record.tools[0].section_fatigue_damage[0] >= 1.0


def test_a_broken_tool_stops_wearing_and_stays_reported():
    dom, ctx, state, edge = _setup()
    secs = state.wear.slots[0].sections
    threshold = secs["strength_b_Pa"][0] * secs["Zb_m3"][0]
    _run(dom, ctx, state, edge, 0, moment=1.1 * threshold)
    vb = state.wear.slots[0].elem.vn.copy()
    for i in range(1, 4):
        record = _run(dom, ctx, state, edge, i)
    np.testing.assert_array_equal(state.wear.slots[0].elem.vn, vb)
    assert record.tools[0].broken and record.broken == [1] and record.expired == [1]
    assert record.tools[0].life_used_fraction == 1.0 and record.tools[0].life_cause == "bending"
    assert len([f for f in state.findings if f.severity == Severity.ERROR]) == 1
