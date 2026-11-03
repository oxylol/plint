"""Energy per part against hand calculations (numbers worked out in the comments)."""

import math

import numpy as np
import pytest

from cnctwin.db.models import params
from cnctwin.production import (SpindleModel, air_specific_energy_J_m3, axis_energy, axis_models, base_energy,
                                conveyor_power_W, process_feed_energy, pump_energy, spindle_energy)
from cnctwin.units import STANDARD_ATMOSPHERE_PA, STANDARD_GRAVITY_M_S2

from production_testkit import machine

P = params("production-energy")


def test_spindle_at_constant_speed_and_power():
    model = SpindleModel.from_machine(machine())
    n = 1000
    dt = np.full(n, 0.01)                                  # 10 s
    omega = np.full(n, 6000 * 2 * math.pi / 60)            # half the 12 000 rpm of the idle-power rating
    shaft = np.full(n, 5000.0)
    terms = spindle_energy(dt, omega, shaft, model)
    # 5 kW / 0.85 for 10 s = 58 823.5 J; idle 700 W x 0.5^(5/3) = 220.5 W for 10 s = 2 205 J
    assert terms["shaft_J"] == pytest.approx(50_000.0)
    assert terms["idle_J"] == pytest.approx(700.0 * 0.5 ** P["palmgren_speed_exponent"] * 10.0)
    assert terms["electrical_J"] == pytest.approx(50_000.0 / 0.85 + terms["idle_J"])
    assert terms["accel_J"] == 0.0 and terms["motor_loss_J"] == pytest.approx(50_000.0 / 0.85 - 50_000.0)
    assert 2190.0 < terms["idle_J"] < 2220.0


def test_spindle_run_up_and_braking():
    model = SpindleModel.from_machine(machine(), extra_inertia_kgm2=0.01)
    omega = np.array([0.0, 1000.0, 1000.0, 0.0])
    dt = np.full(4, 1e-3)
    terms = spindle_energy(dt, omega, np.zeros(4), model)
    kinetic = 0.5 * 0.06 * 1000.0 ** 2                     # J = rotor 0.05 + tool and holder 0.01 kg m2
    assert terms["accel_J"] == pytest.approx(kinetic) and terms["brake_J"] == pytest.approx(kinetic)
    assert terms["recovered_J"] == 0.0                     # braking resistor (default)
    assert terms["electrical_J"] == pytest.approx(kinetic / 0.85 + terms["idle_J"])


def test_axes_friction_gravity_and_inertia():
    models = axis_models(machine(), payload_kg={"X": 50.0, "Y": 50.0})
    by_name = {m.name: m for m in models}
    x, z = by_name["X"], by_name["Z"]
    assert x.mass_kg == 350.0 and x.lift_m_s2 == 0.0 and z.lift_m_s2 == pytest.approx(STANDARD_GRAVITY_M_S2)
    assert z.normal_load_N == 0.0                          # vertical axis: no weight on the guides
    efficiency = P["ballscrew_efficiency"] * P["axis_drive_efficiency"]
    # X at 0.1 m/s for 10 s: (0.003 x 350 kg x g + 30 N) x 1 m = 40.30 J of friction work
    friction_x = P["axis_guide_friction_coeff"] * 350.0 * STANDARD_GRAVITY_M_S2 + P["axis_seal_drag_N"]
    n = 1000
    dt = np.full(n, 0.01)
    vel = np.zeros((n, 3))
    vel[:, 0] = 0.1
    terms = axis_energy(dt, vel, models)
    assert terms["axes"]["X"]["friction_J"] == pytest.approx(friction_x * 1.0)
    assert terms["axes"]["X"]["electrical_J"] == pytest.approx(friction_x / efficiency)
    # Z up 0.2 m in 4 s then down: lifting m g h = 400 x g x 0.2 = 784.5 J, drawn through eta; the
    # way down is braking (gravity works, the dissipated energy is not drawn)
    vel = np.zeros((800, 3))
    vel[:400, 2] = 0.05
    vel[400:, 2] = -0.05
    terms = axis_energy(np.full(800, 0.01), vel, models)
    lift = (400.0 * STANDARD_GRAVITY_M_S2 + P["axis_seal_drag_N"]) * 0.2
    assert terms["axes"]["Z"]["gravity_J"] == pytest.approx(0.0, abs=1e-9)
    assert terms["axes"]["Z"]["electrical_J"] == pytest.approx(lift / efficiency)
    # a start and a stop: the kinetic energy m v^2 / 2 is drawn once and returned once
    vel = np.zeros((4, 3))
    vel[1:3, 0] = 0.1
    terms = axis_energy(np.full(4, 0.01), vel, models)
    assert terms["axes"]["X"]["inertial_J"] == pytest.approx(0.0, abs=1e-12)
    # samples v = 0, 0.1, 0.1, 0 (10 ms each): work = [m v^2/2, F v dt, -m v^2/2 + F v dt, 0]; the last
    # positive sample is the start plus one sample of friction, the braking sample is negative (not drawn)
    start_work = 0.5 * 350.0 * 0.1 ** 2 + friction_x * 0.1 * 0.01
    assert terms["axes"]["X"]["electrical_J"] == pytest.approx(start_work / efficiency)


def test_pumps_conveyor_base_air_and_feed():
    data = machine()
    pumps = pump_energy({"flood": 30.0, "tsc": 0.0}, data)
    # 0.75 kW at its duty point through a 0.8 motor for 30 s
    assert pumps == {"flood": {"on_s": 30.0, "power_W": 937.5, "electrical_J": 28125.0}}
    assert conveyor_power_W(data) == P["conveyor_power_W"]["auger"]
    assert base_energy(100.0, data) == {"base_W": 900.0, "servo_standby_W": 300.0, "base_J": 90000.0,
                                        "servo_standby_J": 30000.0}
    # isothermal work of 1 m3 of free air to 6.9 bar(g): 101 325 ln(7.81) = 208.3 kJ; / 0.55 = 378.7 kJ
    expected = STANDARD_ATMOSPHERE_PA * math.log((STANDARD_ATMOSPHERE_PA + 6.9e5) / STANDARD_ATMOSPHERE_PA)
    assert air_specific_energy_J_m3(6.9e5) == pytest.approx(expected / P["compressor_isothermal_efficiency"])
    assert 3.7e5 < air_specific_energy_J_m3(6.9e5) < 3.9e5
    assert process_feed_energy(100.0) == pytest.approx(100.0 / (0.9 * 0.85))
    with pytest.raises(ValueError):
        pump_energy({"mist": 1.0}, data)
