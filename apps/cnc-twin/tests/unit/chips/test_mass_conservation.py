"""Mass ledger (architecture 12) over 1000 macro steps of random (hash-seeded) emissions: in flight +
settled + conveyed + adhered + in flutes + dust = emitted (1e-9), emitted = rho x removed volume of the
booked windows, and the chip enthalpy closes the same way."""

from __future__ import annotations

import math

import numpy as np

from cnctwin.core.bus import CoolantFeedback, CuttingContact, CuttingFrame, Jet
from cnctwin.core.rng import hash_uniform, stream_key
from cnctwin.chips.domain import ChipsDomain
from cnctwin.chips.state import H_AIR, H_COOLANT, H_FIXTURE, H_MACHINE, H_PART

import chipsfix as fx

SEED = 2026
KEY = stream_key("test.chips.mass")


def u(a: int, b: int, c: int = 0) -> float:
    return hash_uniform(SEED, KEY, a, b, c)


def _coolant(step: int, shape) -> CoolantFeedback:
    on = (step // 100) % 2 == 1
    jets = [Jet(nozzle_id="n1", kind="flood", origin_M_m=np.array([0.08, 0.04, 0.12]),
                direction_M=np.array([-0.6, 0.0, -0.8]), speed_m_s=8.0 if on else 0.0, flow_m3_s=2e-5 if on else 0.0)]
    flush = np.zeros(shape)
    if on:
        flush[:, :] = 50.0
        flush[20:40, 15:30] = 400.0
    return CoolantFeedback(on=on, kind="flood" if on else "dry", jets=jets, fluid_temp_K=295.0,
                           wetted=np.ones(shape, bool) if on else np.zeros(shape, bool), flush_pressure_Pa=flush)


def test_mass_ledger_closes_over_1000_random_steps():
    proj = fx.project()
    ctx = fx.context(proj, seed=SEED)
    state = fx.sim_state(1)
    geometry = fx.scene_box()
    geometry.remove_box((0.02, 0.02, 0.015), (0.08, 0.06, 0.031))          # a pocket
    dom = ChipsDomain(formation=fx.FakeFormation(form=62, length_m=3e-3), geometry=geometry)
    dom.init(ctx, state)
    cs = state.chips
    rho = 2700.0
    booked = []           # rho x removed volume per window, booked one step later
    for step in range(1000):
        x = 0.03 + 0.04 * (0.5 + 0.5 * math.sin(step / 40.0))
        tip = [x, 0.04, 0.016]
        state.bus_next.motion = fx.motion(step, tip_M=tip, omega=-1047.0, feed=(0.02, 0.0, 0.0))
        n_rows = int(u(step, 0) * 7)
        ki = dom.kernel_inputs(ctx, state, state.bus_prev)
        if n_rows:
            pos = np.array([[tip[0] + 0.005 * (u(step, 1, i) - 0.5), tip[1] + 0.005 * (u(step, 2, i) - 0.5),
                             tip[2] + 0.004 * u(step, 3, i)] for i in range(n_rows)])
            vel = np.array([[8 * (u(step, 4, i) - 0.5), 8 * (u(step, 5, i) - 0.5), 6 * u(step, 6, i)]
                            for i in range(n_rows)])
            rows = fx.emission_rows(step, n_rows, 2e-7, pos=pos, vel=vel, temp=350 + 300 * u(step, 7))
            rows[:, 4] *= 0.5 + u(step, 8)
            ki.emission.rec[:n_rows, :rows.shape[1]] = rows
            ki.emission.n[0] = n_rows
        dom.absorb(ctx, state, ki, state.bus_next)
        analytic = float(ki.emission.rec[:n_rows, 4].sum()) if n_rows else 0.0
        consistency = 1.0 + 0.04 * (u(step, 9) - 0.5)
        volume = analytic * consistency / rho
        if n_rows == 0 and u(step, 10) < 0.05:
            volume = 1e-10                                      # removed without an emission row (fallback)
        state.bus_next.removal = fx.removal(volume, point=tip)
        booked.append(rho * volume)
        state.bus_next.coolant = _coolant(step, cs.pile.base_m.shape)
        if step % 7 == 3:
            recut = np.array([[step * 10000 + 5000, 0, 0, 2e-7, tip[0], tip[1]]])
            state.bus_next.cutting = CuttingFrame(contacts=[CuttingContact(contact=0, tool_slot=0, recut=recut)])
        else:
            state.bus_next.cutting = CuttingFrame()
        dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
        state.swap_bus()
        terms = cs.mass_terms()
        total = sum(terms.values())
        assert math.isclose(total, cs.emitted[0], rel_tol=1e-9, abs_tol=1e-15), step
        expected = float(sum(booked[:-1]))
        assert math.isclose(cs.emitted[0], expected, rel_tol=1e-12, abs_tol=1e-18), step
        led = state.ledger
        assert led.chips_in_flight_kg == terms["in_flight_kg"] and led.chips_settled_kg == terms["settled_kg"]
        assert led.chips_in_flutes_kg == terms["in_flutes_kg"] and led.dust_kg == terms["dust_kg"]
        assert led.chips_adhered_kg == terms["adhered_kg"] and led.chips_conveyed_kg == terms["conveyed_kg"]
        record = state.bus_prev.chips
        assert math.isclose(record.emitted_kg, cs.emitted[0], rel_tol=0, abs_tol=0)
        if step % 50 == 49:
            enth = cs.enthalpy_terms(dom._enthalpy)
            now = sum(enth.values()) + float(cs.heat[[H_PART, H_FIXTURE, H_MACHINE, H_COOLANT, H_AIR]].sum()) \
                + float(cs.conveyed[1]) + float(cs.dust[1])
            assert math.isclose(now, cs.emitted[1], rel_tol=1e-9, abs_tol=1e-9), step
    assert cs.emitted[0] > 0
    assert terms["settled_kg"] > 0 and terms["conveyed_kg"] > 0
    assert cs.next_pid > 500
    dom.end_operation(ctx, state)
    terms = cs.mass_terms()
    assert math.isclose(sum(terms.values()), cs.emitted[0], rel_tol=1e-9)
    assert math.isclose(cs.emitted[0], float(sum(booked)), rel_tol=1e-12)
    assert terms["in_flight_kg"] == 0.0 and cs.pile.mass_kg.sum() == 0.0
