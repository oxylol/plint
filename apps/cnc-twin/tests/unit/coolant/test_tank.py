"""Tank: temperature over a shift against an independent integration and its energy ledger, concentration
drift from evaporation and drag-out, chiller control, the evaporation reference conditions."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cnctwin.coolant import fluid as F
from cnctwin.coolant import tank as T

import coolfix as CF


def emulsion_tank(chiller_W=0.0, set_K=math.inf, start_K=295.15):
    fluid = F.fluid_from_db(CF.fluid_db("emulsion-8pct"))
    return T.Tank.create(fluid, 0.208, 0.9, start_K, 293.15, 0.08, chiller_W, set_K)


def test_shift_temperature_against_energy_balance():
    tank = emulsion_tank()
    E0 = tank.energy_J
    T0 = tank.temp_K
    tank_volume0 = tank.volume_m3
    C = tank.capacity_J_K
    power = 1500.0                               # cutting heat + pumps
    dt = 10.0
    n = int(8 * 3600 / dt)
    temps = [T0]
    heat_in = ambient = evap = chill = mass = residual = 0.0
    for _ in range(n):
        step = tank.step(dt, power)
        temps.append(tank.temp_K)
        heat_in += step.heat_in_J
        ambient += step.ambient_J
        evap += step.evaporation_J
        chill += step.chiller_J
        mass += step.mass_out_enthalpy_J
        residual += step.residual_J
    # the ledger closes: change of tank energy = heat in - losses - enthalpy that left with evaporated water
    assert tank.energy_J - E0 == pytest.approx(heat_in - ambient - evap - chill - mass, rel=1e-9, abs=1e-6)
    assert abs(residual) < 1e-6 * heat_in
    assert heat_in == pytest.approx(power * 8 * 3600)
    # an independent integration of C(V) dT/dt = P - Q(T, V), dV/dt = -evaporation(T) with the same loss model
    # (RK45, tight tolerance); the volume falls by evaporation, so the thermal mass shrinks during the shift
    probe = emulsion_tank()
    rho_cp = probe.rho_ref_kg_m3 * probe.cp_ref_J_kgK

    def rhs(t, y):
        probe.temp_K = float(y[0])
        probe.volume_m3 = float(y[1])
        values, _ = probe._losses(float(y[0]))
        rate, _ = probe._evaporation(float(y[0]), probe.geometry()[0])
        return [(power - sum(values)) / (rho_cp * float(y[1])), -rate]

    sol = solve_ivp(rhs, (0.0, n * dt), [T0, tank_volume0], rtol=1e-11, atol=1e-12, t_eval=np.arange(n + 1) * dt)
    assert np.max(np.abs(np.array(temps) - sol.y[0])) < 0.02
    assert tank.volume_m3 == pytest.approx(sol.y[1][-1], rel=1e-4)
    # first minute: the heat goes into the fluid, less the losses at the start (2 K above the shop air:
    # ~35 W of convection and radiation, ~55 W of evaporation)
    start = emulsion_tank()
    losses0 = sum(start._losses(T0)[0])
    assert 50.0 < losses0 < 150.0
    assert temps[6] - T0 == pytest.approx((power - losses0) * 60.0 / C, rel=2e-3)
    # 1.5 kW for 8 h into 187 l (0.75 MJ/K) would be 57 K without losses: the tank warms by ~30 K and is
    # still below its equilibrium
    assert 15.0 < temps[-1] - T0 < 45.0
    values, _ = tank._losses(tank.temp_K)
    assert sum(values) < power


def test_evaporation_raises_concentration_drag_out_keeps_it():
    tank = emulsion_tank(start_K=303.15)
    c0, v0 = tank.concentration, tank.volume_m3
    for _ in range(360):
        tank.step(10.0, 0.0)
    assert tank.volume_m3 < v0
    assert tank.concentrate_m3 == pytest.approx(v0 * c0, rel=1e-12)
    assert tank.concentration > c0
    c1 = tank.concentration
    tank.evaporation_m3_s_per_m2 = 0.0           # isolate the drag-out of this step
    tank.step(10.0, 0.0, drag_out_m3=0.002)
    assert tank.concentration == pytest.approx(c1, rel=1e-12)
    assert tank.totals["drag_out_m3"] == pytest.approx(0.002)


def test_evaporation_rate_at_the_reference_conditions():
    tank = emulsion_tank()
    p = T.P()
    tank.ambient_K = p["evaporation_ref_ambient_K"]
    surface = tank.geometry()[0]
    rate, _ = tank._evaporation(p["evaporation_ref_temp_K"], surface)
    assert rate == pytest.approx(tank.evaporation_m3_s_per_m2 * surface, rel=1e-12)
    # Magnus: 2339 Pa at 20 degC, 4246 Pa at 30 degC (saturation tables)
    assert T.saturation_pressure_Pa(293.15) == pytest.approx(2339.0, rel=5e-3)
    assert T.saturation_pressure_Pa(303.15) == pytest.approx(4246.0, rel=5e-3)


def test_chiller_holds_the_set_point():
    set_K = 298.15
    tank = emulsion_tank(chiller_W=2000.0, set_K=set_K, start_K=303.15)
    chill = 0.0
    heat = 0.0
    amb = 0.0
    for k in range(4 * 360):
        step = tank.step(10.0, 1000.0)
        if k >= 3 * 360:
            chill += step.chiller_J
            heat += step.heat_in_J
            amb += step.ambient_J + step.evaporation_J
    band = T.P()["chiller_band_dK"]
    assert set_K < tank.temp_K < set_K + band
    # at equilibrium the chiller removes what the ambient does not
    assert chill == pytest.approx(heat - amb, rel=0.02)
