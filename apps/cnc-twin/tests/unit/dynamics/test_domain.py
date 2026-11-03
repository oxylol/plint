"""DynamicsDomain with a fake run: systems from the bus, the hot-loop protocol driven by hand, absorb (energy identity,
bus record, ledger), catch-up of ticks the loop skipped, basis changes, checkpoint round trip, chatter findings
with their cause chain, operating points and sld/<tool>.json."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cnctwin.core.abi import EdgeModel
from cnctwin.core.bus import Bus, ProcessCoefficients, RemovalContact, ToolContact, ToolHolding, ToolModes
from cnctwin.core.findings import Category
from cnctwin.core.state import ToolSlot, ToolTable
from cnctwin.dynamics import kernels as K
from cnctwin.dynamics.domain import DynamicsDomain
from cnctwin.dynamics.state import ChatterWindow, DynamicsState
from cnctwin.tooling.abi import EdgeFrameT
from tests.unit.dynamics.dynfix import FakeCtx, bus_with, machine_doc, path_frame, sim_state

TWO_PI = 2.0 * math.pi
OMEGA = -10000.0 * TWO_PI / 60.0
IPO = 1000
MACRO = 10


def _edge(nt=2, z=(0.002, 0.006, 0.010), R=0.005):
    ne = len(z)
    shape = (nt, ne)
    f = lambda v: np.full(shape, v, float)
    edge = EdgeModel(tooth_phase_rad=np.array([TWO_PI * j / nt for j in range(nt)]),
                     elem_z_m=np.tile(np.array(z, float), (nt, 1)), elem_r_m=f(R), elem_lag_rad=f(0.0), elem_len_m=f(0.004),
                     elem_db_m=f(0.004), elem_kappa_rad=f(0.5 * math.pi), elem_rake_rad=f(0.1), elem_incl_rad=f(0.0),
                     elem_clear_rad=f(0.12), elem_edge_radius_m=f(1e-5), elem_kind=np.zeros(shape, np.int32),
                     elem_group=np.zeros(shape, np.int32), elem_station=np.tile(np.arange(ne, dtype=np.int32), (nt, 1)),
                     elem_valid=np.ones(shape, np.int8), runout_T_m=np.zeros(3), tilt_rad=np.zeros(2),
                     station_z_m=np.array(z, float), station_r_m=np.full(ne, R), hand_sign=np.array([-1.0]))
    normal = np.zeros(shape + (3,))
    cut = np.zeros(shape + (3,))
    tan = np.zeros(shape + (3,))
    for j in range(nt):
        a = -float(edge.tooth_phase_rad[j])
        normal[j, :] = [math.cos(a), math.sin(a), 0.0]
        cut[j, :] = [math.sin(a), -math.cos(a), 0.0]
        tan[j, :] = [0.0, 0.0, 1.0]
    frames = EdgeFrameT(tan, normal, cut, np.zeros(shape + (3,)))
    return edge, frames


def _run(physics_dynamics=None, spindle_modes=None, frame_modes=None):
    mach = machine_doc(frame_modes=frame_modes or [{"fn_Hz": 50.0, "zeta": 0.05, "k_N_per_m": 3e7, "direction": "x"}],
                       spindle_modes=spindle_modes or [{"fn_Hz": 950.0, "zeta": 0.03, "k_N_per_m": 2e7, "direction": "x"},
                                                       {"fn_Hz": 950.0, "zeta": 0.03, "k_N_per_m": 2e7, "direction": "y"}])
    physics = SimpleNamespace(dynamics=physics_dynamics or {"sld_n_freq": 400, "sld_n_speeds": 40, "sd_m": 20,
                                                            "sd_points_per_period": 8, "sld_n_lobes": 4},
                              max_rad_per_step=math.radians(0.5), steps_per_tooth=96)
    ctx = FakeCtx(project=SimpleNamespace(physics=physics, machine=mach, stock_pose_W_S=None), ipo_ticks=IPO,
                  macro_ipo=MACRO)
    state = sim_state()
    state.tools = ToolTable((ToolSlot(0, 1, "t", "h"),))
    edge, frames = _edge()
    state.tooling = SimpleNamespace(tools=[SimpleNamespace(rotating=True, edge=edge, frames=frames)], effective=[edge],
                                    frames=[frames])
    dom = DynamicsDomain()
    dom.init(ctx, state)
    return dom, ctx, state


def _holding():
    return ToolHolding(unbalance_static_kg_m=np.array([3e-6, 0.0, 0.0]), unbalance_z_T_m=0.05,
                       unbalance_couple_kg_m2=np.zeros(3))


def _bus(step, holding=True, contact=None, coeffs=None, structure=None, feed=(0.0, 0.0, 0.0)):
    frame = path_frame(step * IPO * MACRO, MACRO, IPO, 1e-6, OMEGA, feed=feed)
    frame.tick0 = step * IPO * MACRO
    bus = bus_with(frame, _holding() if holding else None, contact)
    bus.motion.tick = (step + 1) * IPO * MACRO
    if coeffs is not None:
        bus.cutting.coefficients[0] = coeffs
    if structure is not None:
        bus.structure.tools[0] = structure[0]
        bus.structure.version = structure[1]
    return bus


def _free_step(dom, inputs, tick0):
    """The hot loop out of contact: one free tick per L1 tick and its record."""
    for p, sys in inputs.systems.items():
        for k in range(MACRO):
            t0 = (tick0 + k * IPO) * 1e-6
            t1 = (tick0 + (k + 1) * IPO) * 1e-6
            K.modal_free(sys, OMEGA * t0, OMEGA * t1, OMEGA, OMEGA, np.zeros(7), np.zeros(7), tick0 + (k + 1) * IPO)
            K.record_tick(sys, k)


def test_free_steps_energy_and_ledger():
    dom, ctx, state = _run()
    tick0 = 0
    for step in range(3):
        bus = _bus(step - 1)
        bus.motion.tick = tick0
        inputs = dom.kernel_inputs(ctx, state, bus)
        sys = inputs.systems[0]
        assert sys.ds.node_tick[0] == tick0
        _free_step(dom, inputs, tick0)
        nxt = Bus()
        dom.absorb(ctx, state, None, nxt)
        dom.macro_step(ctx, state, bus, nxt)
        frame = nxt.dynamics
        c = frame.contacts[0]
        assert c.unbalance_work_J != 0.0 and frame.process_damping_J == 0.0
        assert abs(c.modal_energy_change_J - (c.unbalance_work_J + c.base_work_J - c.damping_dissipation_J)) <= \
            1e-9 * (abs(c.unbalance_work_J) + abs(c.damping_dissipation_J) + abs(c.modal_energy_J) + 1e-30)
        tick0 += IPO * MACRO
    # the ledger's out-side external term is minus the work done on the modes
    total = state.dynamics.totals["unbalance_work_J"] + state.dynamics.totals["base_work_J"]
    assert state.ledger.unbalance_work_J == pytest.approx(-total, rel=1e-12)
    values = dom.channel_values(ctx, state)
    assert values["dynamics.tool_deflection_x_m"].shape == (MACRO, 4)
    assert np.all(np.isfinite(values["dynamics.tool_deflection_x_m"][:, 0]))
    assert "dynamics.chatter_index" in values and "dynamics.modal_energy_J" in values


def test_unbalance_response_reaches_the_receptance():
    """Steady free running at 10000 rpm: the spindle-mode response to the unbalance (3 g mm) through the domain."""
    dom, ctx, state = _run()
    tick0 = 0
    for step in range(30):
        bus = _bus(step - 1)
        bus.motion.tick = tick0
        inputs = dom.kernel_inputs(ctx, state, bus)
        _free_step(dom, inputs, tick0)
        dom.absorb(ctx, state, None, Bus())
        tick0 += IPO * MACRO
    rec = state.dynamics.systems[0]
    k = [i for i, m in enumerate(rec.modes) if m.source == "spindle_frf" and abs(m.stations[0, 0]) > 0.0][0]
    sp = rec.modes[k]
    w = sp.omega_rad_s
    theta_end = OMEGA * tick0 * 1e-6
    H = 1.0 / complex(w * w - OMEGA ** 2, 2 * sp.zeta * w * OMEGA)
    q_ss = ((3e-6 / math.sqrt(sp.mass_kg)) * OMEGA ** 2 * H * complex(math.cos(theta_end), math.sin(theta_end))).real
    np.testing.assert_allclose(rec.x[k, 0], q_ss, rtol=1e-4, atol=1e-4 * abs(q_ss))


def test_catch_up_when_the_loop_skips_a_step():
    dom, ctx, state = _run()
    bus = _bus(-1)
    bus.motion.tick = 0
    dom.kernel_inputs(ctx, state, bus)
    dom.absorb(ctx, state, None, Bus())             # the hot loop did not run: the node stays at tick 0
    bus = _bus(0)
    inputs = dom.kernel_inputs(ctx, state, bus)
    assert state.dynamics.meta["caught_up_ticks"] == MACRO
    sys = inputs.systems[0]
    assert sys.ds.node_tick[0] == bus.motion.tick
    _free_step(dom, inputs, bus.motion.tick)
    nxt = Bus()
    dom.absorb(ctx, state, None, nxt)             # energy identity checked over both windows
    c = nxt.dynamics.contacts[0]
    assert abs(c.modal_energy_change_J - (c.unbalance_work_J + c.base_work_J - c.damping_dissipation_J)) <= \
        1e-9 * (abs(c.unbalance_work_J) + abs(c.damping_dissipation_J) + 1e-30)


def test_basis_change_and_tracking():
    dom, ctx, state = _run()
    z = np.array([0.0, 0.004, 0.012])
    shape = np.zeros((2, 3, 6))
    shape[0, :, 0] = (z + 0.05) ** 2 / 0.06 ** 2 / math.sqrt(0.1)
    shape[1, :, 1] = (z + 0.05) ** 2 / 0.06 ** 2 / math.sqrt(0.1)
    tm = ToolModes(freq_Hz=np.array([1800.0, 1850.0]), zeta=np.array([0.02, 0.02]), shape=shape, station_z_m=z,
                   static_residual_m_per_N=np.zeros((9, 9)))
    tick0 = 0
    events = []
    for step, structure in enumerate((None, (tm, 1), (tm, 2))):
        bus = _bus(step - 1, structure=structure)
        bus.motion.tick = tick0
        inputs = dom.kernel_inputs(ctx, state, bus)
        _free_step(dom, inputs, tick0)
        nxt = Bus()
        dom.absorb(ctx, state, None, nxt)
        events.append(nxt.dynamics.basis_change_J)
        tick0 += IPO * MACRO
    rec = state.dynamics.systems[0]
    assert any(m.source == "structure" for m in rec.modes)
    assert events[1] != 0.0                       # fallback spindle modes -> structure modes: projected, booked
    assert events[2] == pytest.approx(0.0, abs=1e-30)   # version 2 republishes the same modes: tracked, no change


def test_checkpoint_round_trip_continues_identically():
    dom, ctx, state = _run()
    bus = _bus(-1)
    bus.motion.tick = 0
    inputs = dom.kernel_inputs(ctx, state, bus)
    _free_step(dom, inputs, 0)
    dom.absorb(ctx, state, None, Bus())
    arrays, meta = state.dynamics.to_arrays()
    json.dumps(meta)
    restored = DynamicsState.from_arrays({k: v.copy() for k, v in arrays.items()}, json.loads(json.dumps(meta)))
    dom2, ctx2, state2 = _run()
    state2.dynamics = restored
    nxt_bus = _bus(0)
    a = dom.kernel_inputs(ctx, state, nxt_bus)
    b = dom2.kernel_inputs(ctx2, state2, nxt_bus)
    _free_step(dom, a, nxt_bus.motion.tick)
    _free_step(dom2, b, nxt_bus.motion.tick)
    np.testing.assert_array_equal(a.systems[0].state.x, b.systems[0].state.x)
    np.testing.assert_array_equal(a.systems[0].da.scal, b.systems[0].da.scal)


def test_chatter_finding_with_cause_chain():
    dom, ctx, state = _run()
    bus = _bus(-1)
    bus.motion.tick = 0
    inputs = dom.kernel_inputs(ctx, state, bus)
    _free_step(dom, inputs, 0)
    nxt = Bus()
    dom.absorb(ctx, state, None, nxt)
    # vibration samples of 25 revolutions with a dominant non-forced component at 1234 Hz
    f_s = abs(OMEGA) / TWO_PI
    n = 25000
    ticks = np.arange(n, dtype=np.int64) * 6
    t = ticks * 1e-6
    u = np.zeros((n, 3))
    u[:, 0] = 2e-6 * np.cos(TWO_PI * 2 * f_s * t) + 30e-6 * np.cos(TWO_PI * 1234.0 * t)
    win = ChatterWindow()
    win.append(ticks, u, np.full(n, abs(OMEGA)), 1.0, 1e-6)
    state.dynamics.windows[0] = win
    dom.macro_step(ctx, state, bus, nxt)
    found = [f for f in state.findings if f.category == Category.CHATTER]
    assert len(found) == 1
    f = found[0]
    assert abs(f.quantities["chatter_freq_Hz"] - 1234.0) < 3.0
    chain = [ctx.trace.get(i).event for i in f.cause_chain]
    assert "dynamics:chatter_onset" in chain and "dynamics:modes" in chain
    assert nxt.dynamics.chatter_index > 0.5 and nxt.dynamics.chatter_freq_Hz > 0.0
    # the episode stays open: no second finding for the same chatter
    dom.macro_step(ctx, state, bus, nxt)
    assert len([f for f in state.findings if f.category == Category.CHATTER]) == 1


def test_operating_point_lobes_and_sld_file(tmp_path):
    dom, ctx, state = _run()
    ctx.run_dir = tmp_path
    h = np.geomspace(1e-6, 1e-3, 8)
    coeffs = ProcessCoefficients(tool_slot=0, h_grid_m=h, Kt_Pa=np.full((1, 8), 7e8), Kr_Pa=np.full((1, 8), 2.5e8),
                                 Ka_Pa=np.zeros((1, 8)), Kte_N_per_m=np.zeros(1), Kre_N_per_m=np.zeros(1),
                                 Kae_N_per_m=np.zeros(1), h_min_m=np.zeros(1), K_sp_N_per_m3=np.zeros(1), version=3)
    R = 0.005
    centroid = np.array([R * math.cos(-math.pi / 4), R * math.sin(-math.pi / 4), 0.006])
    contact = RemovalContact(contact=0, tool_slot=0, workpiece=0, path=0,
                             geometry=ToolContact(in_cut=True, point_S_m=centroid, point_M_m=centroid, ae_m=R,
                                                  ap_m=0.012, engaged_angle_rad=0.5 * math.pi))
    feed = (0.05, 0.0, 0.0)
    tick0 = 0
    for step in range(2):
        bus = _bus(step - 1, contact=contact, coeffs=coeffs, feed=feed)
        bus.motion.tick = tick0
        inputs = dom.kernel_inputs(ctx, state, bus)
        op = dom._op[0]
        assert op is not None and op.cut is not None and len(op.cut.M) == 6
        assert op.feed_per_tooth_m == pytest.approx(0.05 * TWO_PI / (2 * abs(OMEGA)), rel=1e-9)
        _free_step(dom, inputs, tick0)
        nxt = Bus()
        dom.absorb(ctx, state, None, nxt)
        dom.macro_step(ctx, state, bus, nxt)
        tick0 += IPO * MACRO
    c = nxt.dynamics.contacts[0]
    assert math.isfinite(c.depth_limit_m) and c.depth_of_cut_m == pytest.approx(0.012)
    out = dom.finalize(ctx, state)
    doc = json.loads((tmp_path / "sld" / "T1.json").read_text())
    assert out["files"] == ["sld/T1.json"]
    assert doc["format"] == "cnctwin-sld" and doc["tool"] == "T1"
    assert len(doc["speed_rad_s"]) == len(doc["depth_limit_m"]) > 0
    pts = doc["operating_points"]
    assert len(pts) == 1 and pts[0]["stable"] in (True, False) and pts[0]["sd_stable"] in (True, False)
    assert pts[0]["stable"] == (pts[0]["margin"] > 1.0 and pts[0]["sd_stable"])
    assert doc["radial_immersion_m"] == pytest.approx(R)
    if not pts[0]["stable"]:
        assert any(f.category == Category.CHATTER for f in state.findings)
