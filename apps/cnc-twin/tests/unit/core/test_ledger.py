"""Conservation ledger (cnctwin.core.ledger): mass, energy, time, thermal solvers, alignment, state."""

import json

import numpy as np
import pytest

from cnctwin.core.bus import Bus, ChipFeedback, CuttingContact, DynamicsFrame, RemovalContact
from cnctwin.core.domain import RunContext
from cnctwin.core.errors import ConservationError
from cnctwin.core.ledger import Ledger, LedgerError
from cnctwin.core.rundir import Timeline
from cnctwin.core.state import SimState

MACRO = 10_000
RHO = 7850.0


def ctx(timeline=None):
    return RunContext(project=None, db=None, run_dir=None, seed=1, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10,
                      timeline=timeline)


class Toy:
    """Removal, cutting, dynamics and chips records that balance (chips one step behind)."""

    def __init__(self, ledger, context, *, ke=0.005, ke_booked=None, mass_error=0.0, energy_error=0.0, loose=None,
                 displaced=0.0):
        self.ledger, self.ctx = ledger, context
        self.state = SimState()
        self.ke = ke                                          # share of the work that leaves as chip kinetic energy
        self.ke_booked = ke if ke_booked is None else ke_booked
        self.mass_error, self.energy_error, self.loose, self.displaced = mass_error, energy_error, loose, displaced
        self.inventory = 0.0
        self.n = 0

    def step(self):
        state, n = self.state, self.n
        bus = Bus()
        volume = 1e-9 * (1 + n % 3)
        total = 50.0 + n
        bus.removal.contacts = [RemovalContact(contact=0, removed_volume_m3=volume, displaced_volume_m3=self.displaced)]
        if self.loose and n == 2:
            bus.removal.loose_bodies = [dict(self.loose)]
        bus.cutting.contacts = [CuttingContact(contact=0, shaft_work_J=0.9 * total, feed_work_J=0.1 * total,
                                               heat_shear_J=0.6 * total, heat_rake_J=0.25 * total,
                                               heat_flank_J=0.1 * total, ploughing_J=0.02 * total)]
        bus.dynamics = DynamicsFrame(modal_energy_change_J=0.01 * total, damping_dissipation_J=(0.02 - self.ke) * total,
                                     process_damping_J=0.0)
        prev = state.bus_prev
        prev_volume = sum(c.removed_volume_m3 - c.displaced_volume_m3 for c in prev.removal.contacts)
        prev_total = sum(c.shaft_work_J + c.feed_work_J for c in prev.cutting.contacts)
        self.inventory += RHO * prev_volume
        bus.chips = ChipFeedback(booked_until_tick=n * MACRO if n else -1,
                                 kinetic_energy_J=(self.ke_booked + self.energy_error) * prev_total)
        state.ledger.chips_in_flight_kg = 0.4 * self.inventory
        state.ledger.chips_settled_kg = 0.6 * self.inventory * (1 + self.mass_error)
        state.bus_next = bus
        values = self.ledger.check(self.ctx, state, (n + 1) * MACRO)
        state.swap_bus()
        self.n += 1
        return values


def make(**kw):
    return Ledger(tol_mass=1e-9, tol_energy=1e-6, density_kg_m3=RHO, **kw)


def test_balanced_run_passes_and_reports():
    ledger = make()
    toy = Toy(ledger, ctx())
    for _ in range(12):
        values = toy.step()
    assert values["ledger.mass_residual_rel"] < 1e-14 and values["ledger.energy_residual_rel"] < 1e-14
    assert values["ledger.alignment_lag_s"] == pytest.approx(0.01)
    state = toy.state
    assert state.ledger.shaft_work_J == pytest.approx(sum(0.9 * (50 + n) for n in range(12)))
    assert state.ledger.removed_mass_kg == pytest.approx(RHO * state.ledger.removed_volume_m3)
    summary = ledger.summary()
    assert summary["mass"]["ok"] and summary["energy"]["ok"] and summary["mass"]["checked_steps"] == 12
    assert summary["energy"]["checked_steps"] == 11            # the first step waits for chips
    assert summary["time"]["checked_steps"] == 0 and "not checked" in summary["time"]["note"]
    assert {c.name for c in Ledger.channels()} >= set(values)


def test_displaced_and_loose_bodies_are_own_terms():
    ledger = make()
    loose = {"workpiece": 0, "volume_m3": 2e-10, "mass_kg": RHO * 2e-10}
    toy = Toy(ledger, ctx(), loose=loose, displaced=2e-10)
    for _ in range(6):
        toy.step()
    terms = ledger.summary()["mass"]["terms"]
    assert terms["loose_kg"] == pytest.approx(RHO * 2e-10) and terms["displaced_m3"] == pytest.approx(12e-10)
    assert toy.state.ledger.freed_body_mass_kg == pytest.approx(RHO * 2e-10)


def test_mass_violation_raises_and_adds_a_fatal_finding():
    ledger = make()
    context = ctx()
    toy = Toy(ledger, context, mass_error=1e-6)
    with pytest.raises(ConservationError, match="mass"):
        for _ in range(5):
            toy.step()
    finding = toy.state.findings.items[-1]
    assert finding.category.value == "mass_balance" and finding.severity.value == "fatal"
    assert context.trace.get(finding.cause_chain[0]).event == "finding:mass_balance"


def test_energy_violation_raises():
    toy = Toy(make(), ctx(), energy_error=1e-5)
    with pytest.raises(ConservationError, match="energy"):
        for _ in range(5):
            toy.step()


def test_missing_chip_kinetic_energy_is_seen():
    balanced = Toy(make(), ctx(), ke=0.0)
    for _ in range(4):
        balanced.step()
    unbalanced = Toy(make(), ctx(), ke=0.005, ke_booked=0.004)    # chips books less than the cut spent
    with pytest.raises(ConservationError, match="energy"):
        for _ in range(4):
            unbalanced.step()


def test_inactive_owners_are_not_checked():
    ledger = make(domains={"cutting"})
    toy = Toy(ledger, ctx(), ke=0.0)                      # no chips domain: no chip kinetic energy booked
    for _ in range(3):
        values = toy.step()
    assert values["ledger.energy_residual_rel"] < 1e-14
    assert np.isnan(values["ledger.mass_residual_rel"])
    assert ledger.summary()["mass"]["ok"] is False and "not checked" in ledger.summary()["mass"]["note"]


def test_chips_lagging_too_far_is_an_error():
    ledger = make(max_lag_ticks=2 * MACRO)
    state = SimState()
    for n in range(3):
        state.bus_next = Bus()
        state.bus_next.chips = ChipFeedback(booked_until_tick=-1)
        if n < 2:
            ledger.check(ctx(), state, (n + 1) * MACRO)
        else:
            with pytest.raises(LedgerError, match="behind"):
                ledger.check(ctx(), state, (n + 1) * MACRO)
        state.swap_bus()


def test_a_record_left_from_two_steps_ago_is_refused():
    ledger = make(domains={"cutting"})
    state = SimState()
    stale = CuttingContact(shaft_work_J=1.0, heat_shear_J=1.0)
    state.bus_next.cutting.contacts = [stale]
    ledger.check(ctx(), state, MACRO)
    state.swap_bus()
    state.bus_next = Bus()
    ledger.check(ctx(), state, 2 * MACRO)
    state.swap_bus()                                         # bus_next is now the first step's untouched Bus
    with pytest.raises(LedgerError, match="two macro steps"):
        ledger.check(ctx(), state, 3 * MACRO)


def test_time_balance_with_a_timeline():
    timeline = Timeline()
    ledger = make(domains=())
    state = SimState()
    a = timeline.open(path=0, op="OP10", program="p", line=1, block="G0", kind="rapid", tick0=0)
    ledger.check(ctx(timeline), state, MACRO)
    timeline.close(a, MACRO, ticks={"rapid": MACRO})
    b = timeline.open(path=0, op="OP10", program="p", line=2, block="G1", kind="linear", tick0=MACRO)
    c = timeline.open(path=1, op="OP10", program="q", line=1, block="G4", kind="dwell", tick0=0)
    timeline.close(c, 15_000, ticks={"dwell": 15_000})
    values = ledger.check(ctx(timeline), state, 2 * MACRO)
    assert values["ledger.time_residual_ticks"] == 0 and state.ledger.block_time_ticks == 2 * MACRO + 15_000
    timeline.close(b, 2 * MACRO, ticks={"cutting": MACRO})
    timeline.open(path=0, op="OP10", program="p", line=3, block="G1", kind="linear", tick0=2 * MACRO + 500)
    with pytest.raises(ConservationError, match="time"):
        ledger.check(ctx(timeline), state, 3 * MACRO)        # 500 ticks not in any block
    over = Timeline()
    d = over.open(path=0, op="OP10", program="p", line=1, block="G1", kind="linear", tick0=0)
    over.close(d, MACRO, ticks={"cutting": MACRO, "air_cutting": 10})
    with pytest.raises(ConservationError, match="categories"):
        make(domains=()).check(ctx(over), SimState(), MACRO)


def test_thermal_residuals_and_solver_hook():
    ledger = make(domains=())
    state = SimState()
    state.bus_next.thermal.ledger_residual_rel = {"tool.0": 1e-12, "work.0": 3e-13}
    values = ledger.check(ctx(), state, MACRO)
    assert values["ledger.thermal_residual_rel"] == 1e-12
    state.swap_bus()
    state.bus_next = Bus()
    state.bus_next.thermal.ledger_residual_rel = {"work.0": 2e-6}
    with pytest.raises(ConservationError, match="work.0"):
        ledger.check(ctx(), state, 2 * MACRO)
    hook = ledger.solver("spindle.main")
    assert hook.book(10.0, 12.0, 3.0, 1.0, 0.0, 0.1) == 0.0
    with pytest.raises(ConservationError, match="spindle.main"):
        hook.book(10.0, 12.5, 3.0, 1.0, 0.0, 0.1)
    assert ledger.summary()["thermal"]["solvers"]["spindle.main"] > 1e-6


def test_state_round_trip():
    one = make()
    toy = Toy(one, ctx())
    for _ in range(5):
        toy.step()
    saved = json.loads(json.dumps(one.checkpoint_state()))
    two = make()
    two.restore_state(saved)
    toy.ledger = two
    for _ in range(5):
        toy.step()
    reference = Toy(make(), ctx())
    for _ in range(10):
        reference.step()
    assert two.summary() == reference.ledger.summary()
    assert toy.state.ledger.shaft_work_J == reference.state.ledger.shaft_work_J
