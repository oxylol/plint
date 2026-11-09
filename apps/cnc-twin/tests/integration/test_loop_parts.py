"""The loop's own pieces that need no simulation (docs/models/loop.md): the coupling-lag monitor of architecture 5.3,
the fast ring's restamping, the ledger that reports the chips' kinetic energy, the adapters' checkpointable
change tracking, the peak view cutting's macro channel gets."""

from __future__ import annotations

import types

import numpy as np
import pytest

from cnctwin.core.abi import ring_buffer
from cnctwin.core.bus import Bus, CuttingContact
from cnctwin.core.loop import CouplingMonitor, Run, _PeakView, _scalar
from cnctwin.core.loop_wiring import LiveField, RunLedger
from cnctwin.core.state import SimState


def test_the_monitor_flags_a_sustained_alternation_once():
    monitor = CouplingMonitor(ratio=0.5, rel_change=1e-3, alternations=3)
    flagged = []
    values = [500.0, 510.0, 501.0, 509.0, 502.0, 508.0, 503.0, 507.0]
    for v in values:
        flagged += monitor.update({"thermal.tools[0].rake_temp_mean_K": v})
    assert len(flagged) == 1 and flagged[0][0] == "thermal.tools[0].rake_temp_mean_K" and flagged[0][4] == 3
    # a monotone rise, a damped alternation (ratio below 0.5) and round-off never flag
    for series in ([300.0 + k for k in range(10)], [500.0 + 10.0 * (-0.3) ** k for k in range(10)],
                   [500.0 + 1e-9 * (-1) ** k for k in range(10)]):
        quiet = CouplingMonitor(ratio=0.5, rel_change=1e-3, alternations=3)
        assert not sum((quiet.update({"q": v}) for v in series), [])
    restored = CouplingMonitor(ratio=0.5, rel_change=1e-3, alternations=3)
    restored.restore(monitor.state())
    assert restored.state() == monitor.state()


def test_the_ring_rows_of_a_step_move_inside_their_l0_step():
    ring = ring_buffer(8, 2)
    run = types.SimpleNamespace(recorder=types.SimpleNamespace(ring=ring))
    for k, tick in enumerate((1008, 1016, 1024)):              # end ticks of three L0 steps of stride 8
        ring.tick[k] = tick
    ring.head[0] = 3
    Run._restamp_ring(run, 0)
    assert ring.tick[:3].tolist() == [1007, 1015, 1023]
    ring.tick[3:5] = (1032, 2000)                              # the next step: only its own rows move
    ring.head[0] = 5
    Run._restamp_ring(run, 3)
    assert ring.tick[:5].tolist() == [1007, 1015, 1023, 1031, 1999]


def test_the_peak_view_and_scalars():
    record = CuttingContact()
    record.h_max_m = np.array([[1e-5, 3e-5], [np.nan, 2e-5]])
    view = _PeakView(record)
    assert view.h_max_m == pytest.approx(3e-5) and view.tool_slot == record.tool_slot
    assert _scalar([np.nan, 2.0, -1.0]) == 2.0 and np.isnan(_scalar([]))


def _ledger():
    return RunLedger(tol_mass=1e-9, tol_energy=1e-6, density_kg_m3=2700.0, domains={"cutting", "removal", "chips",
                                                                                     "dynamics"})


def test_the_kinetic_energy_is_reported_not_balanced():
    ledger = _ledger()
    state = SimState()
    state.bus_next = Bus()
    contact = CuttingContact()
    contact.shaft_work_J, contact.heat_shear_J = 10.0, 10.0
    state.bus_next.cutting.contacts.append(contact)
    state.bus_next.chips.booked_until_tick = 10000
    state.bus_next.chips.kinetic_energy_J = 1e-3            # 1e-4 of the work: would fail the 1e-6 balance
    values = ledger.check(None, state, 10000)
    assert values["ledger.energy_residual_rel"] == 0.0
    assert values["ledger.chip_kinetic_reported_J"] == pytest.approx(1e-3)
    assert state.ledger.chip_kinetic_J == pytest.approx(1e-3)
    summary = ledger.summary()
    assert summary["energy"]["chip_kinetic_reported_J"] == pytest.approx(1e-3) and summary["energy"]["ok"]
    copy = _ledger()
    copy.restore_state(ledger.checkpoint_state())
    assert copy.kinetic_reported_J == pytest.approx(1e-3)
    assert {"ledger.chip_kinetic_reported_J", "ledger.modal_quadrature_J"} <= {c.name for c in RunLedger.channels()}


def test_the_quadrature_difference_is_its_own_term():
    ledger = _ledger()
    state = SimState()
    state.bus_next = Bus()
    contact = CuttingContact()
    # cutting: shaft 10 J = heat 9.999 J + 1.000e-3 J on the deflection rates (its quadrature);
    # the modes received 0.999e-3 J exactly (change + damping)
    contact.shaft_work_J, contact.heat_shear_J = 10.0, 9.999
    state.bus_next.cutting.contacts.append(contact)
    dyn = state.bus_next.dynamics
    dyn.modal_energy_change_J, dyn.damping_dissipation_J = 0.6e-3, 0.399e-3
    dyn.cutting_work_quadrature_J, dyn.cutting_work_J = 1.0e-3, 0.999e-3
    state.bus_next.chips.booked_until_tick = 10000
    values = ledger.check(None, state, 10000)
    assert values["ledger.energy_residual_rel"] < 1e-12
    assert ledger.summary()["energy"]["modal_quadrature_J"] == pytest.approx(1e-6)


def test_the_field_change_tracking_survives_a_checkpoint():
    field = LiveField(types.SimpleNamespace(state=None), "loop")
    field._seen = {("thermal", 0): np.array([1, 2, 3]), ("chips", 1): np.array([4])}
    saved = field.seen_state()
    other = LiveField(types.SimpleNamespace(state=None), "loop")
    other.restore_seen(saved)
    assert sorted(other._seen) == sorted(field._seen)
    assert all(np.array_equal(other._seen[k], field._seen[k]) for k in field._seen)


def test_free_vibration_round_off_is_judged_against_the_floor():
    # out of the cut: the modes ring down after a spindle start, damping and the modal energy change agree to their
    # integrator's round-off (the lathe warmup: 6.7e-18 J on 3.0e-12 J terms, relative 1.1e-6)
    def interval(ledger):
        state = SimState()
        state.bus_next = Bus()
        dyn = state.bus_next.dynamics
        dyn.damping_dissipation_J, dyn.modal_energy_change_J = 3.0348603901574923e-12, -3.0348536769629423e-12
        state.bus_next.chips.booked_until_tick = 10000
        return ledger.check(None, state, 10000)

    from cnctwin.core.errors import ConservationError
    with pytest.raises(ConservationError):
        interval(_ledger())
    floored = _ledger()
    floored.energy_scale_floor_J = 1e-6
    values = interval(floored)
    assert values["ledger.energy_residual_rel"] < 1e-10
