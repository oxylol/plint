"""Physical spindle network (cnctwin/thermal/spindle.py): bearing friction, lubricant viscosity, exact
integration and energy ledger, growth composition, model choice, checkpoint, domain coupling to the structure."""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cnctwin.core.bus import Bus
from cnctwin.db.models import params
from cnctwin.thermal import machine as M
from cnctwin.thermal import props
from cnctwin.thermal import spindle as SP
from cnctwin.thermal.domain import ThermalDomain

import fixtures as F

RPM = 2.0 * math.pi / 60.0
NETWORK_SPINDLE = {
    "id": "main", "kind": "milling", "max_rpm": 8000, "drive_type": "belt",
    "bearings": [{"position": "front", "type": "angular_contact", "count": 3, "bore_mm": 70, "outer_mm": 110,
                  "contact_angle_deg": 15, "preload_N": 300, "static_load_rating_kN": 30, "lubrication": "grease",
                  "axially_fixed": True},
                 {"position": "rear", "type": "angular_contact", "count": 2, "bore_mm": 60, "outer_mm": 95,
                  "contact_angle_deg": 15, "preload_N": 200, "static_load_rating_kN": 22, "lubrication": "grease"}],
    "thermal": {"housing_mass_kg": 180, "housing_size_mm": [500, 500, 600], "housing_growth_length_mm": 350,
                "nose_to_fixed_bearing_mm": 84},
}


def network(spindle=None, ambient_K=293.15):
    net, estimated = SP.build_network("main", spindle or NETWORK_SPINDLE, ambient_K)
    return net, estimated


def run(net, rpm, seconds, step=60.0, T_env=293.15, T_struct=None):
    t = 0.0
    while t < seconds - 1e-9:
        dt = min(step, seconds - t)
        net.step(dt, np.array([rpm * RPM]), 0.0, 1e-9, t, T_env, T_env if T_struct is None else T_struct)
        t += dt


def test_walther_viscosity_through_its_two_points_and_slope():
    a, b = SP.walther_coefficients(22e-6, 4.3e-6)
    assert SP.viscosity_m2_s(a, b, 313.15)[0] == pytest.approx(22e-6, rel=1e-12)
    assert SP.viscosity_m2_s(a, b, 373.15)[0] == pytest.approx(4.3e-6, rel=1e-12)
    nu, slope = SP.viscosity_m2_s(a, b, 303.15)
    fd = (SP.viscosity_m2_s(a, b, 303.16)[0] - SP.viscosity_m2_s(a, b, 303.14)[0]) / 0.02
    assert slope == pytest.approx(fd, rel=1e-6) and slope < 0.0 and nu > 22e-6


def test_palmgren_moment_matches_the_hand_calculation():
    net, _ = network()
    b = net.bearings[0]
    nu = 22e-6
    moment, _ = SP.bearing_moment_Nm(b, np.array([4000.0]), nu)
    mv = 1e-7 * 2.0 * (22.0 * 4000.0) ** (2.0 / 3.0) * 90.0 ** 3          # N mm, f0 = 2 (grease, single ACBB)
    p0 = max(0.0, 0.5 * 0.0 + 0.46 * 300.0)                               # ISO 76, 15 deg: Y0 = 0.46
    f1 = 0.001 * (p0 / 30000.0) ** 0.33
    ml = f1 * 0.9 * 300.0 / math.tan(math.radians(15.0)) * 90.0
    assert moment[0] == pytest.approx((mv + ml) * 1e-3, rel=1e-12)
    low, _ = SP.bearing_moment_Nm(b, np.array([50.0]), nu)                 # nu n = 1100 < 2000: 160e-7 f0 dm^3
    assert low[0] == pytest.approx((160e-7 * 2.0 * 90.0 ** 3 + ml) * 1e-3, rel=1e-12)


def test_ledger_closes_and_steps_are_exact():
    net, _ = network()
    run(net, 4000, 1800.0, step=60.0)
    net.step(0.01, np.full(10, 4000 * RPM), 2.0, 1e-9, 1800.0, 293.15, 295.0)     # holder heat in, warm structure
    net.step(0.0, np.zeros(1), 1.5, 1e-9, 1800.0, 293.15, 293.15)
    assert net.ledger.worst_rel < 1e-12
    assert net.ledger.sources_J - net.ledger.losses_J == pytest.approx(net.energy() - net.ledger.u0_J, rel=1e-10)
    # a long step and many short ones agree (the sources are linearised per sub-step)
    a, _ = network()
    b, _ = network()
    run(a, 6000, 3600.0, step=60.0)
    run(b, 6000, 3600.0, step=1.0)
    assert a.growth_m() == pytest.approx(b.growth_m(), rel=2e-3)
    assert np.allclose(a.temps_K(), b.temps_K(), atol=0.05)


def test_steady_state_balances_heat_in_and_out():
    net, _ = network()
    run(net, 4000, 40 * 3600.0, step=600.0)
    out = sum(net.last[f"to_{k}_W"] for k in ("air", "structure", "jacket"))
    assert out == pytest.approx(net.last["bearing_W"] + net.last["motor_W"], rel=1e-4)
    assert net.housing_T_K > 293.15 and net.T_K > net.housing_T_K                # the shaft runs hotter


def test_growth_is_shaft_and_housing_strain_over_their_lengths():
    net, _ = network()
    net.set_temperatures([303.15, 310.0, 309.0, 298.15])
    p = params("thermal-machine")
    steel = props.material_table_by_id(str(p["spindle_material"]))
    iron = props.material_table_by_id(str(p["housing_material"]))
    expected = props.thermal_strain(steel, 303.15, 293.15) * 0.084 + props.thermal_strain(iron, 298.15, 293.15) * 0.35
    assert net.growth_m() == pytest.approx(expected, rel=1e-12)


def test_heat_grows_with_speed_and_with_a_colder_lubricant():
    cold, _ = network(ambient_K=283.15)
    warm, _ = network(ambient_K=303.15)
    heats = [warm.bearing_heat_W(np.array([n * RPM])) for n in (2000, 4000, 6000)]
    assert heats[0] < heats[1] < heats[2]
    assert cold.bearing_heat_W(np.array([4000 * RPM])) > warm.bearing_heat_W(np.array([4000 * RPM]))
    run(cold, 4000, 4 * 3600.0, T_env=283.15)
    run(warm, 4000, 4 * 3600.0, T_env=303.15)
    assert cold.growth_m() > warm.growth_m()                                    # cold ambient: thicker oil, more growth


def test_jacket_holds_the_housing_near_the_coolant():
    jacketed = copy.deepcopy(NETWORK_SPINDLE)
    jacketed["thermal"]["jacket"] = {"flow_l_min": 10, "channel_diameter_mm": 8, "channels": 1, "area_cm2": 1500}
    net, _ = network(jacketed)
    p = params("thermal-machine")
    flow = 10e-3 / 60.0
    mcp = p["jacket_fluid_density_kg_m3"] * flow * p["jacket_fluid_cp_J_kgK"]
    assert 0.0 < net.jacket_g_W_K < mcp                                          # effectiveness below 1
    dry, _ = network()
    run(net, 6000, 6 * 3600.0)
    run(dry, 6000, 6 * 3600.0)
    assert net.housing_T_K - 293.15 < 0.25 * (dry.housing_T_K - 293.15)
    assert net.totals["to_jacket_J"] > 0.0


def test_model_choice_from_the_machine_file():
    assert M.spindle_model_kind(NETWORK_SPINDLE) == "network"
    assert M.spindle_model_kind(F.MACHINE_RAW["spindles"][0]) == "lumped"          # lumped data only
    assert M.spindle_model_kind({"id": "main", "kind": "milling", "nose": "BT40"}) == "network"
    forced = dict(NETWORK_SPINDLE, thermal={"model": "lumped", "growth_at_max_speed_um": 20})
    assert M.spindle_model_kind(forced) == "lumped"
    net, estimated = network({"id": "main", "kind": "milling", "nose": "CAT50"})
    assert net.bearings[0].bore_m == pytest.approx(0.1)                          # 50-taper class
    assert any("bearing arrangement" in e for e in estimated)


def test_spindle_models_cold_warm_and_checkpoint_round_trip():
    machine = {"spindles": [NETWORK_SPINDLE], "thermal": {"warmup": {"cold_to_warm_min": 60}}}
    cold = M.spindle_models(machine, 293.15, "cold", 1e-9, 0.0)["main"]
    warm = M.spindle_models(machine, 293.15, "warm", 1e-9, 0.0)["main"]
    assert cold.model == "network" and warm.body.T_K > cold.body.T_K
    assert warm.growth_rel_m() == 0.0 and warm.network.ledger.worst_rel == 0.0
    for _ in range(5):
        cold.step(60.0, np.array([5000 * RPM]), 0.0, 1e-9, 0.0, T_struct_K=293.15)
    copy_ = M.SpindleThermal.from_dict(json.loads(json.dumps(cold.to_dict())))
    for model in (cold, copy_):
        for _ in range(5):
            model.step(60.0, np.array([5000 * RPM]), 0.5, 1e-9, 0.0, T_struct_K=294.0)
    assert cold.growth_abs_m() == copy_.growth_abs_m()
    assert cold.network.temps_K().tobytes() == copy_.network.temps_K().tobytes()
    assert cold.last_to_structure_J == copy_.last_to_structure_J > 0.0


def test_domain_passes_the_head_heat_to_the_structure():
    machine = copy.deepcopy(F.MACHINE_RAW)
    machine["spindles"] = [dict(NETWORK_SPINDLE, max_rpm=12000)]
    proj = F.project()
    proj = SimpleNamespace(**{**proj.__dict__, "machine": machine})
    ctx = F.context(proj)
    state = F.sim_state()
    dom = ThermalDomain(geometry=F.stock(), edges=F.Edges(F.edge_model()))
    dom.init(ctx, state)
    th = state.thermal
    assert th.spindles["main"].model == "network"
    for n in range(20):
        prev = F.bus_step(n)
        nxt = Bus()
        state.clock.tick = int(prev.motion.tick)
        dom.macro_step(ctx, state, prev, nxt)
    fb = nxt.thermal
    assert fb.spindle_growth_M_m["main"][2] < 0.0                                # toward the work
    net = th.spindles["main"].network
    assert net.totals["bearing_J"] > 0.0
    report = dom.finalize(ctx, state)
    assert report["spindles"]["main"]["model"] == "network"
    share = th.spindles["main"].uniform_growth_m_per_K()
    assert report["structure"]["spindle_share_m_per_K"] == pytest.approx(share, rel=1e-12) and share > 0.0
    assert set(report["spindles"]["main"]["node_temps_K"]) == set(net.names)
    for name, ledger in dom._solver_ledgers(th):
        assert ledger.worst_rel < 1e-10, name


def test_domain_cadence_and_resume_with_pending_spindle_steps():
    from cnctwin.thermal.state import ThermalState
    machine = copy.deepcopy(F.MACHINE_RAW)
    machine["spindles"] = [dict(NETWORK_SPINDLE, max_rpm=12000)]
    proj = SimpleNamespace(**{**F.project().__dict__, "machine": machine})
    ctx = F.context(proj)

    def advance(state, dom, n0, n1):
        out = None
        for n in range(n0, n1):
            prev = F.bus_step(n)
            out = Bus()
            state.clock.tick = int(prev.motion.tick)
            dom.macro_step(ctx, state, prev, out)
        return out

    state = F.sim_state()
    dom = ThermalDomain(geometry=F.stock(), edges=F.Edges(F.edge_model()))
    dom.init(ctx, state)
    advance(state, dom, 0, 15)
    assert state.thermal.spindle_pending["main"]["n"] == 5                       # stepped at 10, five pending
    arrays, meta = state.thermal.to_arrays()
    meta = json.loads(json.dumps(meta))
    clock = copy.deepcopy(state.clock)
    ledger = copy.deepcopy(state.ledger)
    a = advance(state, dom, 15, 30)
    state_b = F.sim_state()
    state_b.clock, state_b.ledger = clock, ledger
    state_b.thermal = ThermalState.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta)
    dom_b = ThermalDomain(geometry=F.stock(), edges=F.Edges(F.edge_model()))
    b = advance(state_b, dom_b, 15, 30)
    assert a.thermal.spindle_temp_K == b.thermal.spindle_temp_K
    assert np.array_equal(a.thermal.spindle_growth_M_m["main"], b.thermal.spindle_growth_M_m["main"])
    report = dom.finalize(ctx, state)                                           # flushes what is pending
    assert "main" not in state.thermal.spindle_pending
    assert report["spindles"]["main"]["energy_J"]["bearing_J"] > 0.0


def test_uniform_growth_per_K_is_the_whole_spindle_one_kelvin_up():
    net, _ = network()
    net.set_temperature(293.15 + 1.0)
    assert net.uniform_growth_m_per_K() == pytest.approx(net.growth_m(), rel=1e-12)
    machine = {"spindles": [NETWORK_SPINDLE]}
    model = M.spindle_models(machine, 293.15, "cold", 1e-9, 0.0)["main"]
    assert model.uniform_growth_m_per_K() == pytest.approx(net.uniform_growth_m_per_K(), rel=1e-12)


def test_structure_counts_the_spindle_head_once_under_uniform_warming():
    """The machine file's structure sensitivity is the whole loop's drift per K (the head included): with the
    spindle linked, a uniform warming of spindle and structure drifts by exactly that sensitivity."""
    machine = {"spindles": [NETWORK_SPINDLE], "thermal": {"structure_sensitivity_um_per_K": 3.0}}
    spindles = M.spindle_models(machine, 293.15, "cold", 1e-9, 0.0)
    structure = M.structure_model(machine, 293.15, "cold")
    M.link_structure(structure, spindles)
    share = spindles["main"].uniform_growth_m_per_K()
    assert structure.spindle_share_m_per_K == pytest.approx(share, rel=1e-12) and share > 4e-6
    for dT, rel in ((1.0, 1e-9), (3.0, 1e-3)):           # exact at 1 K; the mean-CTE tables bend it slightly beyond
        spindles["main"].network.set_temperature(293.15 + dT)
        structure.body.set_temperature(293.15 + dT)
        lift = structure.drift_m() - (spindles["main"].growth_abs_m() - spindles["main"].growth0_m)
        assert lift == pytest.approx(3e-6 * dT, rel=rel)
    copy_ = M.StructureThermal.from_dict(json.loads(json.dumps(structure.to_dict())))
    assert copy_.drift_m() == structure.drift_m()
    old = structure.to_dict()
    del old["spindle_share_m_per_K"]                    # a checkpoint written before the share existed
    assert M.StructureThermal.from_dict(old).spindle_share_m_per_K == 0.0
    lathe = {"spindles": [dict(NETWORK_SPINDLE, kind="turning")], "thermal": {"structure_sensitivity_um_per_K": 4.0}}
    turning = M.structure_model(lathe, 293.15, "cold")
    M.link_structure(turning, M.spindle_models(lathe, 293.15, "cold", 1e-9, 0.0))
    assert turning.spindle_share_m_per_K == 0.0          # a turning spindle grows across the radial loop
