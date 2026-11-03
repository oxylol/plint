"""The chips Domain object: interface, bus record, checkpoint round trip (bit-identical continuation),
determinism, and the findings with their chains of cause."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.chips import abi, kernels
from cnctwin.chips.channels import channels
from cnctwin.chips.domain import ChipsDomain
from cnctwin.chips.state import ChipsState, PAN_BULK, PAN_MASS
from cnctwin.core.bus import CoolantFeedback, CuttingContact, CuttingFrame, Jet
from cnctwin.core.channels import ChannelRegistry
from cnctwin.core.domain import Domain
from cnctwin.core.errors import InputError
from cnctwin.core.findings import Category, Severity
from cnctwin.db.simap import wrap

import chipsfix as fx

RHO = 2700.0


def _domain(proj=None, geometry=None, formation=None, n_slots=1):
    proj = proj or fx.project()
    ctx = fx.context(proj)
    state = fx.sim_state(n_slots)
    dom = ChipsDomain(formation=formation or fx.FakeFormation(), geometry=geometry or fx.scene_box())
    dom.init(ctx, state)
    return dom, ctx, state


def _run(dom, ctx, state, steps, tip, rows_fn=None, omega=-1047.0, coolant_fn=None, cutting_fn=None,
         events_fn=None, consistency=1.0, start=0):
    for step in range(start, start + steps):
        ki = dom.kernel_inputs(ctx, state, state.bus_prev)
        rows = rows_fn(step) if rows_fn is not None else None
        n = 0 if rows is None else len(rows)
        if n:
            ki.emission.rec[:n, :rows.shape[1]] = rows
            ki.emission.n[0] = n
        dom.absorb(ctx, state, ki, state.bus_next)
        tip_k = tip(step) if callable(tip) else tip
        events = events_fn(step) if events_fn is not None else None
        state.bus_next.motion = fx.motion(step, tip_M=tip_k, omega=omega, events=events)
        mass = float(rows[:, abi.EM["mass_kg"]].sum()) if n else 0.0
        state.bus_next.removal = fx.removal(mass * consistency / RHO, point=tip_k)
        state.bus_next.coolant = coolant_fn(step, state) if coolant_fn is not None else CoolantFeedback()
        state.bus_next.cutting = cutting_fn(step) if cutting_fn is not None else CuttingFrame()
        dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
        state.swap_bus()
        assert math.isclose(sum(state.chips.mass_terms().values()), state.chips.emitted[0], rel_tol=1e-9,
                            abs_tol=1e-18)


def _rows(n=4, mass=2e-7, pos=(0.05, 0.04, 0.035), vel=(6.0, 2.0, 3.0), temp=450.0):
    return lambda step: fx.emission_rows(step, n, mass, pos=pos, vel=vel, temp=temp)


def _of(state, category):
    return [f for f in state.findings if f.category == category]


# ---------------------------------------------------------------------------------------------- interface

def test_domain_interface_channels_and_kernel_inputs():
    dom, ctx, state = _domain()
    assert isinstance(dom, Domain)
    registry = ChannelRegistry()
    registry.register_all(dom.channels())
    assert len(registry.all()) == len(channels()) and all(c.domain == "chips" for c in registry.all())
    ki = dom.kernel_inputs(ctx, state, state.bus_prev)
    for layout in ki:
        assert abi.is_contiguous_layout(layout)
    assert ki.emission.rec.shape[1] == abi.EMISSION_WIDTH
    assert ki.packing.torque_Nm.shape == (1,) and ki.packing.force_T0_N.shape == (1, 3)
    # the bus record publishes the pile grid at init (coolant places its grids on it)
    assert state.bus_prev.chips.pile_height_m.shape == state.chips.pile.base_m.shape
    with pytest.raises(InputError):
        ChipsDomain(geometry=fx.scene_box()).init(ctx, fx.sim_state(1))       # no formation model wired


def test_absorb_overflow_doubles_the_buffer_and_keeps_the_rows():
    dom, ctx, state = _domain()
    ki = dom.kernel_inputs(ctx, state, state.bus_prev)
    cap = ki.emission.rec.shape[0]
    rows = fx.emission_rows(0, 3, 1e-7, pos=[0.05, 0.04, 0.035], vel=[1, 0, 0])
    ki.emission.rec[:3, :rows.shape[1]] = rows
    ki.emission.n[:] = (3, 1)
    dom.absorb(ctx, state, ki, state.bus_next)
    assert state.chips.emission_capacity == 2 * cap and len(state.chips.pending) == 3
    assert dom.kernel_inputs(ctx, state, state.bus_prev).emission.rec.shape[0] == 2 * cap


def test_bus_record_and_formation_per_contact():
    dom, ctx, state = _domain()
    _run(dom, ctx, state, 4, [0.05, 0.04, 0.035], rows_fn=_rows(), consistency=1.02)
    rec = state.bus_prev.chips
    assert rec.formation and rec.formation[0].contact == 0 and rec.formation[0].form_code == 62
    batch = rec.formation[0].emissions
    assert len(batch) > 0 and np.all(batch.tool_slot == 0)
    # the mass rule: parcels carry rho x field volume (consistency 1.02 x the analytic mass)
    assert math.isclose(float(batch.mass_kg.sum()), 4 * 2e-7 * 1.02, rel_tol=1e-12)
    assert rec.booked_until_tick == 30000 and rec.emitted_kg == state.chips.emitted[0]
    assert rec.kinetic_energy_J > 0
    assert set(dom.macro_values(state)) == {c.name for c in channels()}


# ---------------------------------------------------------------------------------------------- checkpoint

def _arrays(state):
    arrays, meta = state.chips.to_arrays()
    return arrays, meta


def test_checkpoint_round_trip_continues_bit_identically():
    dom, ctx, state = _domain()
    rows = _rows(n=5, vel=(4.0, -3.0, 2.0))
    _run(dom, ctx, state, 25, [0.05, 0.04, 0.035], rows_fn=rows)
    arrays, meta = state.chips.to_arrays()
    dom2, ctx2, state2 = _domain()
    state2.chips = ChipsState.from_arrays({k: v.copy() for k, v in arrays.items()}, meta)
    state2.bus_prev, state2.bus_next = state.bus_prev, state.bus_next
    import copy
    state2.bus_prev = copy.deepcopy(state.bus_prev)
    state2.bus_next = copy.deepcopy(state.bus_next)
    _run(dom, ctx, state, 20, [0.05, 0.04, 0.035], rows_fn=rows, start=25)
    _run(dom2, ctx2, state2, 20, [0.05, 0.04, 0.035], rows_fn=rows, start=25)
    a1, m1 = state.chips.to_arrays()
    a2, m2 = state2.chips.to_arrays()
    assert m1 == m2
    for key in a1:
        assert np.array_equal(a1[key], a2[key], equal_nan=True), key


def test_two_runs_are_identical():
    results = []
    for _ in range(2):
        dom, ctx, state = _domain()
        _run(dom, ctx, state, 30, [0.05, 0.04, 0.035], rows_fn=_rows(n=6, vel=(5.0, 1.0, 4.0)))
        results.append(state.chips.to_arrays())
    for key in results[0][0]:
        assert np.array_equal(results[0][0][key], results[1][0][key], equal_nan=True), key


# ---------------------------------------------------------------------------------------------- findings

def test_hot_chips_bouncing_back_weld_to_the_edge():
    geometry = fx.scene_box()
    geometry.remove_box((0.044, 0.0, 0.02), (0.056, 0.08, 0.031))          # 12 mm slot, D10 tool
    dom, ctx, state = _domain(geometry=geometry)
    # chips leave the flutes (M03: the flute moves -y at the +x side) toward the slot wall 1.5 mm away, bounce
    # back and strike the tool again while hotter than 0.6 x the solidus (K)
    rows = _rows(n=4, mass=1e-5, pos=(0.0545, 0.04, 0.023), vel=(4.0, -5.0, 0.0), temp=800.0)
    _run(dom, ctx, state, 8, [0.05, 0.04, 0.021], rows_fn=rows)
    welds = [f for f in _of(state, Category.CHIP_HAZARD) if "weld" in f.message]
    assert welds and welds[0].severity == Severity.WARNING
    assert state.chips.mass_terms()["adhered_kg"] > 0
    chain = [ctx.trace.get(i).event for i in welds[0].cause_chain]
    assert "chips:weld" in chain and "chips:formation" in chain


def test_stringy_chips_wrap_the_turning_part():
    raw = {"material": fx.material_raw(), "machine": {**fx.machine_raw(), "type": "lathe",
                                                        "spindles": [{"id": "main", "kind": "turning"}]},
           "stock": {"kind": "cylinder", "size_mm": [20.0, 60.0]}, "environment": {"ambient_degC": 20},
           "physics": {"particles_per_tooth": 1.0}, "tools": {"list": [fx.tool_raw(kind="turning_insert")]}}
    proj = wrap(raw)
    geometry = fx.scene_box(lo=(-0.01, -0.01, 0.0), hi=(0.01, 0.01, 0.06))
    dom, ctx, state = _domain(proj=proj, geometry=geometry,
                              formation=fx.FakeFormation(form=11, length_m=0.08, thickness_m=2e-4))
    rows = _rows(n=4, mass=5e-5, pos=(0.025, 0.0, 0.03), vel=(-6.0, 0.0, 0.0), temp=400.0)
    _run(dom, ctx, state, 8, [0.03, 0.0, 0.03], rows_fn=rows, omega=300.0)
    wraps = [f for f in _of(state, Category.CHIP_HAZARD) if "wrap" in f.message]
    assert wraps and "turning workpiece" in wraps[0].message
    assert state.bus_prev.chips.wrap_mass_kg > 0
    # the spindle stops: the wrapped chips fall off
    _run(dom, ctx, state, 3, [0.03, 0.0, 0.03], omega=0.0, start=8)
    assert state.bus_prev.chips.wrap_mass_kg == 0.0


def test_composite_dust_and_magnesium_ignition():
    raw_c = fx.material_raw(family="composite", density=1550.0)
    raw_c["composite"] = {"fibre": "carbon", "fibre_volume_fraction": 0.6, "dust_mass_fraction": 1.0}
    proj = wrap({**{k: v for k, v in dict(fx.project().raw).items()}, "material": raw_c})
    dom, ctx, state = _domain(proj=proj)
    _run(dom, ctx, state, 3, [0.05, 0.04, 0.035], rows_fn=_rows())
    assert math.isclose(state.chips.dust[0], state.chips.emitted[0], rel_tol=1e-12) and state.chips.dust[0] > 0
    assert [f for f in _of(state, Category.CHIP_HAZARD) if "dust" in f.message]
    raw_m = fx.material_raw(family="magnesium", density=1770.0)
    raw_m["ignition"] = {"chip_hazard_degC": {"value": 500, "src": "est"}}
    proj = wrap({**{k: v for k, v in dict(fx.project().raw).items()}, "material": raw_m})
    dom, ctx, state = _domain(proj=proj)
    _run(dom, ctx, state, 3, [0.05, 0.04, 0.035], rows_fn=_rows(temp=900.0))
    fire = [f for f in _of(state, Category.CHIP_HAZARD) if "ignition" in f.message]
    assert fire and fire[0].severity == Severity.ERROR


def test_chips_on_the_next_setups_locating_face_and_on_the_probe():
    dom, ctx, state = _domain()
    H = state.chips.pile
    for iu in range(20, 40):
        for iv in range(20, 30):
            kernels.deposit(H, dom.model, iu, iv, 1e-6, 0.0, 1.2e-3, 1.5e-4, 2e-3, 3e-3, 62)
    state.chips.emitted[0] += 200e-6
    probe_point = np.array([0.02 + 0.001 + 0.01 - 0.02 + 0.02, 0.01 + 0.02 + 0.001, 0.03])
    events = lambda step: [{"kind": "probe", "tick": step * 10000 + 5000, "pos_S_m": probe_point}] if step == 2 else []
    _run(dom, ctx, state, 4, [0.09, 0.07, 0.2], events_fn=events)
    probe = [f for f in _of(state, Category.CHIP_HAZARD) if "probing" in f.message]
    assert probe and state.chips.gflags[4] == 1
    dom.end_operation(ctx, state)
    debris = state.bus_prev.chips.locator_debris_m
    assert debris["S+z"] > 0.0 and debris["S-z"] == 0.0
    assert _of(state, Category.MISLOCATION)
    assert state.chips.pile.mass_kg.sum() == 0.0 and state.chips.pan[PAN_MASS] > 0       # cleaned into the pan
    assert math.isclose(sum(state.chips.mass_terms().values()), state.chips.emitted[0], rel_tol=1e-12)


def test_long_chips_hook_on_a_nozzle_and_block_it():
    dom, ctx, state = _domain(formation=fx.FakeFormation(form=11, length_m=0.05))

    def coolant(step, st):
        return CoolantFeedback(on=True, kind="flood", fluid_temp_K=295.0,
                               jets=[Jet(nozzle_id="n1", kind="flood", origin_M_m=np.array([0.05, 0.04, 0.045]),
                                         direction_M=np.array([0.0, 1.0, 0.0]), speed_m_s=0.5, flow_m3_s=1e-6)])
    rows = _rows(n=2, mass=1e-6, pos=(0.05, 0.04, 0.041), vel=(0.0, 0.0, 6.0))
    _run(dom, ctx, state, 6, [0.05, 0.0, 0.1], rows_fn=rows, coolant_fn=coolant)
    assert state.bus_prev.chips.blocked_nozzles == ["n1"]
    assert [f for f in _of(state, Category.CHIP_HAZARD) if "nozzle" in f.message]


def test_recutting_and_bird_nesting_and_pan_overflow():
    dom, ctx, state = _domain(formation=fx.FakeFormation(form=11, length_m=0.04))
    H = state.chips.pile
    for iu in range(28, 48):
        for iv in range(20, 42):
            kernels.deposit(H, dom.model, iu, iv, 3e-6, 0.0, 1.2e-3, 1.5e-4, 2e-3, 0.04, 11)
    state.chips.emitted[0] += 440 * 3e-6
    recut = lambda step: CuttingFrame(contacts=[CuttingContact(
        contact=0, tool_slot=0, recut=np.array([[step * 10000 + 5000, 0, 0, 1e-7, 0.07, 0.04]]))])
    _run(dom, ctx, state, 6, [0.075, 0.04, 0.031], rows_fn=_rows(n=2, mass=5e-7), cutting_fn=recut)
    assert [f for f in _of(state, Category.CHIP_RECUTTING) if "recuts" in f.message]
    assert [f for f in _of(state, Category.CHIP_HAZARD) if "nest" in f.message]
    state.chips.pan[PAN_MASS] += 1.0          # a full pan (booked as made, so the ledger still closes)
    state.chips.pan[PAN_BULK] += 10.0
    state.chips.emitted[0] += 1.0
    _run(dom, ctx, state, 1, [0.075, 0.04, 0.2], start=6)
    assert [f for f in _of(state, Category.CHIP_HAZARD) if "overflows" in f.message]
