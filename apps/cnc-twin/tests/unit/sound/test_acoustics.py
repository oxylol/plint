"""The acoustic relations against their textbook statements and limits."""

import math

import numpy as np
import pytest

from cnctwin.sound import acoustics as A
from cnctwin.sound import params as P
from cnctwin.units import AIR_DENSITY_20C_KG_M3, rpm_to_rad_s

RHO, C = AIR_DENSITY_20C_KG_M3, 343.2


def test_sphere_radiation_efficiencies_and_the_dipole_far_field():
    ka = np.array([1e-3, 0.1, 1.0, 10.0, 1e3])
    mono, dip = A.sigma_monopole(ka), A.sigma_dipole(ka)
    assert mono[0] == pytest.approx(1e-6, rel=1e-5) and dip[0] == pytest.approx(0.25e-12, rel=1e-5)
    assert mono[-1] == pytest.approx(1.0, abs=1e-6) and dip[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(mono) > 0) and np.all(np.diff(dip) > 0) and np.all(dip[:3] <= mono[:3])  # compact: dipole weaker
    # compact oscillating sphere: W from sigma equals the integral of the far-field intensity of
    # p = rho a^3 omega^2 U cos(theta) / (2 c r) (Pierce): pi rho omega^4 a^6 U^2 / (6 c^3) for amplitude U
    a, f, U = 0.01, 200.0, 0.1
    omega = 2 * math.pi * f
    w_sigma = RHO * C * (4 * math.pi * a * a / 3) * float(A.sigma_dipole(omega / C * a)) * (U * U / 2)
    w_far = math.pi * RHO * omega ** 4 * a ** 6 * U * U / (6 * C ** 3)
    assert w_sigma == pytest.approx(w_far, rel=1e-3)


def test_infinite_plate_mobility_and_the_sea_power():
    E, rho, nu, h = 210e9, 7850.0, 0.3, 0.002
    B = E * h ** 3 / (12 * (1 - nu * nu))
    assert A.plate_mobility(E, rho, nu, h) == pytest.approx(1 / (8 * math.sqrt(B * rho * h)))
    thick = A.plate_mobility(E, rho, nu, 0.004)
    assert thick == pytest.approx(A.plate_mobility(E, rho, nu, h) / 4.0)          # Y ~ 1 / h^2
    f = np.array([20.0, 40.0, 4000.0, 8000.0])
    power = A.sea_power_per_force2(f, E, rho, nu, h, 0.01, 1.0)
    assert np.all(power < A.plate_mobility(E, rho, nu, h))                        # radiates part of the input
    assert power[1] / power[0] == pytest.approx(8.0, rel=0.05)    # compact: sigma ~ (ka)^4, eta_rad ~ sigma / f ~ f^3
    assert power[3] / power[2] == pytest.approx(0.5, rel=0.05)                    # sigma ~ 1, eta_rad << eta_int: ~ 1/f
    heavily_damped = A.sea_power_per_force2(f, E, rho, nu, h, 0.1, 1.0)
    assert np.all(heavily_damped < power)


def test_room_equation_mass_law_and_enclosure():
    R = A.room_constant(340.0, 0.12)
    assert R == pytest.approx(340 * 0.12 / 0.88)
    assert A.listener_factor(1.0, 2.0, R) == pytest.approx(2 / (4 * math.pi) + 4 / R)
    tl = A.mass_law_tl([100.0, 1000.0, 2000.0], 11.775, 5.0)
    assert tl[2] - tl[1] == pytest.approx(6.0, abs=0.1)                           # +6 dB per octave
    assert tl[1] == pytest.approx(20 * math.log10(math.pi * 1000 * 11.775 / (RHO * C)) - 5.0, abs=0.1)
    tau = A.enclosure_power_transmission(np.array([100.0, 1000.0, 10000.0]))
    p = P.sound()
    floor = p["leak_area_fraction"] / p["enclosure_absorption"]
    assert np.all(np.diff(tau) < 0) and tau[-1] > floor * 0.99 and np.all(tau <= 1.0)   # leaks limit the loss


def test_a_weighting_matches_the_iec_table():
    f = 10.0 ** np.array([1.5, 2.0, 3.0, 3.6, 4.0])                               # exact base-ten frequencies
    expected = np.array([-39.4, -19.1, 0.0, 1.0, -2.5])                           # IEC 61672-1 table 3 (31.5 Hz ... 10 kHz)
    assert np.allclose(A.a_weighting_dB(f), expected, atol=0.1)


def test_empirical_motor_and_pump_levels():
    level = A.motor_level_dB(11e3, rpm_to_rad_s(1500.0))
    assert float(level) == pytest.approx(17 + 17 * math.log10(11) + 15 * math.log10(1500))
    capped = A.motor_level_dB(11e3, rpm_to_rad_s(12000.0))
    assert float(capped) == pytest.approx(17 + 17 * math.log10(11) + 15 * math.log10(3600))   # no extrapolation
    large = A.motor_level_dB(75e3, rpm_to_rad_s(1500.0))
    assert float(large) == pytest.approx(28 + 10 * math.log10(75) + 15 * math.log10(1500))
    assert np.isneginf(A.motor_level_dB(11e3, 0.0))
    w = A.power_from_level_at(80.0, 1.0)
    assert 10 * math.log10(w / 1e-12) == pytest.approx(80.0 + 10 * math.log10(2 * math.pi) +
                                                       10 * math.log10(400e-12 / (RHO * C * 1e-12)), abs=0.01)
    pump = A.pump_power_W(750.0)
    assert pump == pytest.approx(A.power_from_level_at(72 + 10 * math.log10(0.75), 1.0))


def test_jets_bubbles_impacts_bearings_unbalance_chatter():
    w1, w2 = A.lighthill_power_W(50.0, 0.002), A.lighthill_power_W(100.0, 0.002)
    assert w2 / w1 == pytest.approx(2.0 ** 8)                                    # eighth power of the speed
    assert A.minnaert_Hz(1e-3, 998.0) == pytest.approx(3280.0, rel=0.02)          # 3.26 kHz mm for water
    t1 = A.hertz_duration_s(1e-6, 5e-4, 10.0, 70e9, 0.33, 210e9, 0.3)
    t2 = A.hertz_duration_s(1e-6, 5e-4, 10.0 * 2 ** 5, 70e9, 0.33, 210e9, 0.3)
    assert float(t1 / t2) == pytest.approx(2.0)                                   # T ~ v^(-1/5)
    assert 1e-6 < float(t1) < 1e-4
    ratio = A.ball_pass_outer_ratio(20, 0.011, 0.09, math.radians(15))
    assert ratio == pytest.approx(10 * (1 - 0.011 / 0.09 * math.cos(math.radians(15))))
    force = A.unbalance_force_N(12.0, 0.0025, 1256.6, 1256.6)
    assert float(force) == pytest.approx(12.0 * 0.0025 * 1256.6)                  # m G omega at the rated speed
    mode = (1000.0, 0.05, 1e8)
    assert float(A.chatter_force_N(1e-6, 1000.0, mode)) == pytest.approx(1e-6 * 1e8 * 2 * 0.05)   # 2 zeta k x
    assert float(A.chatter_force_N(1e-6, 0.0, mode)) == pytest.approx(100.0)                     # static: k x
