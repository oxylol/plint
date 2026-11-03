"""Occlusion: reach with a holder blocking a nozzle (shadow, finding, cause chain), a clear nozzle, a misaimed
nozzle, the latency contract, through-tool coolant in a drilled hole, the film map and Watson's film."""

import numpy as np
import pytest

from cnctwin.coolant import kernels as K, reference as R
from cnctwin.coolant.domain import CoolantDomain
from cnctwin.coolant.protocols import OccupancyBricks, TooledShapes
from cnctwin.core.findings import Category

import coolfix as CF


def nozzles(position_mm, aim):
    return [{"id": "n1", "kind": "flood", "attached_to": "head", "position_mm": position_mm, "aim": aim,
             "diameter_mm": 4, "pump": "flood"},
            {"id": "tsc", "kind": "through_spindle", "attached_to": "spindle_nose", "position_mm": [0, 0, 0],
             "aim": [0, 0, -1], "pump": "tsc"}]


def domain_with(nozzle_list, physics=None, raws=(CF.ENDMILL,), bricks=None):
    proj = CF.project(machine=CF.machine_raw(nozzles=nozzle_list),
                      physics=physics or {"every_n_macro": 1, "latency_macro": 1, "rays_per_jet": 32})
    ctx = CF.context(proj)
    state = CF.sim_state(len(raws))
    dom = CoolantDomain(shapes=TooledShapes(CF.geometries(raws)), bricks=bricks)
    dom.init(ctx, state)
    return dom, ctx, state


def test_clear_nozzle_reaches_the_cut():
    dom, ctx, state = domain_with(nozzles([70, 0, -60], [-70, 0, -50]))
    rec = CF.run(dom, ctx, state, 3)[-1].coolant
    assert rec.reach_fraction[0] > 0.9
    assert rec.h_flank_W_m2K[0] > 1e4 and rec.h_work_W_m2K[0] > 1e4
    assert not [f for f in state.findings if f.category == Category.COOLANT_SHADOWING]
    jet = next(j for j in rec.jets if j.nozzle_id == "n1")
    # the jet's own reach: the share of its whole flow that lands in a cutting zone (its cone is wider)
    assert 0.5 < jet.reach_fraction <= rec.reach_fraction[0] and len(jet.path_M_m) >= 2
    # dynamic pressure along the coherent jet (inside its break-up length): 1/2 rho v^2 with the speed gained
    # falling toward the cut (energy conservation)
    assert len(jet.path_q_Pa) == len(jet.path_M_m) and jet.breakup_m > 0.5
    rho = dom.fluid.props(rec.fluid_temp_K).rho_kg_m3
    drop = jet.path_M_m[0, 2] - jet.path_M_m[:, 2]
    expected = 0.5 * rho * (jet.speed_m_s ** 2 + 2 * 9.80665 * drop)
    assert np.allclose(jet.path_q_Pa, expected, rtol=1e-3)


def test_holder_blocking_a_nozzle():
    # 30 mm off the axis, 10 mm below the spindle nose, aimed at the tool tip: the holder is in the way
    dom, ctx, state = domain_with(nozzles([0, 30, -10], [0, -30, -100]))
    rec = CF.run(dom, ctx, state, 3)[-1].coolant
    assert rec.reach_fraction[0] < 0.1
    shadows = rec.contact_shadows[0]
    assert max(shadows, key=shadows.get) == "holder of T1"
    assert shadows["holder of T1"] > 0.9
    assert rec.shadows["holder of T1"] > 0.9
    # the few rays that slip past still give a local coefficient (where they land); thermal applies it over the
    # reached share of the surface, so the cooling of the tool drops with the reach, and lubrication with it
    clear_dom, clear_ctx, clear_state = domain_with(nozzles([70, 0, -60], [-70, 0, -50]))
    clear = CF.run(clear_dom, clear_ctx, clear_state, 3)[-1].coolant
    assert rec.h_flank_W_m2K[0] * rec.reach_fraction[0] < 0.1 * clear.h_flank_W_m2K[0] * clear.reach_fraction[0]
    assert rec.friction_factor[0] > 0.99 > clear.friction_factor[0]
    jet = next(j for j in rec.jets if j.nozzle_id == "n1")
    assert jet.blocked_by == "holder of T1"
    found = [f for f in state.findings if f.category == Category.COOLANT_SHADOWING]
    assert len(found) == 1                                    # one finding per episode
    finding = found[0]
    assert "holder of T1" in finding.message and "T1" in finding.message
    assert finding.quantities["blocked_frac[holder of T1]"] > 0.9
    events = [ctx.trace.get(i).event for i in finding.cause_chain]
    assert events[0] == "finding:coolant_shadowing"
    assert "coolant:occlusion" in events and "coolant:jets" in events and "coolant:supply" in events


def test_misaimed_nozzle_names_the_nearest_jet():
    dom, ctx, state = domain_with(nozzles([70, 0, -60], [1, 0, 0]))
    rec = CF.run(dom, ctx, state, 3)[-1].coolant
    assert rec.reach_fraction[0] == 0.0 and rec.aimed_flow_kg_s[0] == 0.0
    found = [f for f in state.findings if f.category == Category.COOLANT_SHADOWING]
    assert found and "no coolant jet is aimed at the cut of T1" in found[0].message
    assert "n1" in found[0].message and found[0].quantities["nearest_miss_m"] > 0.0


def test_occlusion_latency_is_one_macro_step():
    dom, ctx, state = domain_with(nozzles([70, 0, -60], [-70, 0, -50]))
    outs = CF.run(dom, ctx, state, 3)
    first, second = outs[0].coolant, outs[1].coolant
    assert first.on and first.reach_fraction[0] == 0.0 and first.occlusion_tick == -1
    assert second.reach_fraction[0] > 0.9
    tick0 = int(CF.bus_step(0).motion.tick)
    assert second.occlusion_tick == tick0                    # traced at step 0, applied at step 1
    # latency 0 applies the job of the same step
    dom0, ctx0, state0 = domain_with(nozzles([70, 0, -60], [-70, 0, -50]),
                                     {"every_n_macro": 1, "latency_macro": 0, "rays_per_jet": 32})
    assert CF.run(dom0, ctx0, state0, 1)[0].coolant.reach_fraction[0] > 0.9


def test_through_tool_coolant_reaches_the_bottom_of_the_hole():
    centre = np.array([0.05, 0.04])
    hole = lambda p: ~((np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) < 0.0042) & (p[:, 2] > 0.01))
    bricks = OccupancyBricks([0.0, 0.0, 0.0], [0.1, 0.08, 0.03], 1e-3, 4, inside=hole)
    dom, ctx, state = domain_with(CF.machine_raw()["coolant"]["nozzles"], raws=(CF.DRILL,), bricks=bricks)
    tip = (0.05, 0.04, 0.012)
    outs = CF.run(dom, ctx, state, 3, tip_S=tip, pumps={"tsc": True}, contact_point_S=tip, ap=4e-3, ae=8e-3)
    rec = outs[-1].coolant
    assert rec.kind == "through_spindle"
    assert rec.through_tool_flow_m3_s > 0.0 and rec.through_tool_exit_speed_m_s > 20.0
    assert rec.reach_fraction[0] == pytest.approx(1.0)
    assert rec.h_work_W_m2K[0] > 3e4                         # jet on the hole bottom at H/d ~ 2
    holes = [j for j in rec.jets if j.through_tool]
    assert len(holes) == 2 and all(j.origin_M_m[2] < 0.0135 for j in holes)


def test_film_map_against_reference_and_terrain():
    nu, nv, cell = 40, 30, 2e-3
    base = np.zeros((nu, nv))
    base[25:, :] = 0.010                                      # a plateau 10 mm higher
    base[:, :3] = -np.inf                                     # no surface (off the part)
    imps = np.array([[0.02, 0.03, 2e-4, 10.0, 0.005, 990.0, 1.06e-6, 0.0, 0.0],
                     [0.065, 0.03, 1e-3, 150.0, 0.01, 1.2, 1.5e-5, 1.0, 0.0],
                     [0.07, 0.01, 1e-3, 100.0, 0.005, 1.2, 1.5e-5, 2.0, 0.006]])
    fp = np.array([0.1833, 1.402, 1.0, 1e-3])
    wet = np.zeros((nu, nv), np.int8)
    flush = np.zeros((nu, nv))
    K.film_map(base, cell, imps, fp, wet, flush, np.zeros(nu * nv, np.int64), np.zeros((nu, nv), np.int64))
    wet_r, flush_r = R.film_map_ref(base, cell, imps, fp)
    assert np.array_equal(wet, wet_r)
    assert np.allclose(flush, flush_r, rtol=1e-14, atol=0.0)
    # the liquid film covers the low part, does not climb onto the plateau, never wets cells without a surface
    assert wet[:25, 3:].all()
    assert not wet[26:, 3:].any() or wet[26:, 3:].sum() < 30      # only the mist footprint on the plateau
    assert not wet[:, :3].any()
    assert flush.max() == pytest.approx(0.5 * 990.0 * 100.0, rel=1e-12)


def test_watson_film_speed():
    fp = np.array([0.1833, 1.402, 1.0, 1e-3])
    flow, speed, d, nu = 2e-4, 10.0, 0.005, 1e-6
    re = speed * d / nu
    r0 = 0.1833 * d * re ** (1 / 3)
    assert K.watson_speed(0.5 * r0, flow, speed, d, nu, fp) == speed
    assert K.watson_speed(r0 * (1 + 1e-12), flow, speed, d, nu, fp) == pytest.approx(speed, rel=1e-9)
    rs = np.linspace(r0, 20 * r0, 50)
    us = [K.watson_speed(r, flow, speed, d, nu, fp) for r in rs]
    assert np.all(np.diff(us) < 0)


def test_steady_machining_does_not_flood_the_trace():
    dom, ctx, state = domain_with(nozzles([70, 0, -60], [-70, 0, -50]))
    CF.run(dom, ctx, state, 12)
    events = [e.event for e in ctx.trace._memory]
    # a job every step, but the same result: one jets event and one occlusion event per zone
    assert events.count("coolant:jets") == 1
    assert events.count("coolant:occlusion") == 2           # the contact and the tool zone
    assert events.count("coolant:supply") == 1
