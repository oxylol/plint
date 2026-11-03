"""Exit burrs: the roll-over model against the published turning case (Machado et al. 2003: the leaned exit burr
is as high as the depth of cut at kappa_r 45 deg), through the domain's exit geometry; exit angle, exit order,
ductility and breakout; the drilling burr from the plastic collapse of the remaining disc; Hocheng-Dharan
delamination of composites; burr survival and the burr finding against the drawing limit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.surface import surffix as fx
from cnctwin.core.findings import Category, Severity
from cnctwin.surface import burrs as BU
from cnctwin.surface import params as PRM
from cnctwin.surface.domain import SurfaceDomain
from cnctwin.surface.protocols import StockQuery
from cnctwin.surface.state import BF, BI

KR45 = math.radians(45.0)


def _model(material=None):
    return BU.BurrModel(PRM.burr(), PRM.material_props(material or fx.MATERIAL))


def test_rollover_height_equals_the_depth_of_cut_in_the_published_case():
    """Machado et al. 2003 (AISI 1045, SNMG 120408, kappa_r 45 deg, f 0.215 mm/rev): leaned burr height ~ ap
    over ap 0.55 ... 2.0 mm; the model's C_H comes from it, the test checks the scaling with ap and feed."""
    model = _model()
    f = 0.215e-3
    h = f * math.sin(KR45)
    for ap in (0.55e-3, 0.78e-3, 1.0e-3, 1.5e-3, 2.0e-3):
        res = model.exit_burr(h, ap, KR45, 1.0, True, 20e-6)
        assert res.kind == BU.KIND_ROLLOVER
        assert res.height_m == pytest.approx(ap, rel=0.2)
        assert res.thickness_m == pytest.approx(h + 20e-6, rel=1e-9)
    thick = [model.exit_burr(ff * math.sin(KR45), 1e-3, KR45, 1.0, True, 20e-6).thickness_m
             for ff in (0.138e-3, 0.323e-3, 0.646e-3)]
    assert thick[0] < thick[1] < thick[2]            # thickness grows with feed (Machado Fig. 4b)
    low, high = (model.exit_burr(h, 1e-3, math.radians(k), 1.0, True, 20e-6).height_m for k in (20.0, 85.0))
    assert low < 1e-3 < high                         # below ap at 20 deg, above it beyond 45 deg (Fig. 8)


def test_exit_angle_order_and_ductility():
    model = _model()
    base = model.exit_burr(1e-4, 1e-3, KR45, 1.0, True, 0.0).height_m
    assert model.exit_burr(1e-4, 1e-3, KR45, 0.5, True, 0.0).height_m == pytest.approx(0.5 * base)
    assert model.exit_burr(1e-4, 1e-3, KR45, 0.0, True, 0.0).height_m == 0.0     # tangential exit
    assert model.exit_burr(1e-4, 1e-3, KR45, 1.0, False, 0.0).height_m == pytest.approx(0.5 * base)
    semi = dict(fx.MATERIAL, physical=dict(fx.MATERIAL["physical"], elongation_pct={"value": 6}))
    semi_h = _model(semi).exit_burr(1e-4, 1e-3, KR45, 1.0, True, 0.0).height_m
    assert semi_h == pytest.approx(base * math.log(1.06) / 0.1, rel=1e-6)
    brittle = dict(fx.MATERIAL, physical=dict(fx.MATERIAL["physical"], elongation_pct={"value": 1}))
    res = _model(brittle).exit_burr(1e-4, 1e-3, KR45, 1.0, True, 0.0)
    assert res.kind == BU.KIND_BREAKOUT and res.height_m == pytest.approx(0.5 * base)


def test_composite_exits_leave_pulled_out_fibres_instead_of_burrs():
    comp = dict(fx.MATERIAL, composite={"fibre": "carbon", "delamination": {"G_IC_J_m2": {"value": 300}}})
    res = _model(comp).exit_burr(1e-4, 1e-3, KR45, 1.0, True, 15e-6)
    assert res.kind == BU.KIND_PULLOUT and res.height_m == pytest.approx(2.0 * 15e-6) and res.thickness_m == 0.0


def test_drilling_burr_from_the_plastic_collapse_of_the_remaining_disc():
    ductile = dict(fx.MATERIAL, physical=dict(fx.MATERIAL["physical"], elongation_pct={"value": 25}))
    model = _model(ductile)
    thrust = 1000.0
    sigma = 0.5 * (505e6 + 655e6)
    t_c = math.sqrt(thrust / (1.5 * math.pi * sigma))
    res = model.drill_burr(thrust)
    assert res.factors["onset_thickness_m"] == pytest.approx(t_c, rel=1e-12)
    assert res.height_m == pytest.approx(t_c, rel=1e-12) and res.kind == BU.KIND_UNIFORM
    assert model.drill_burr(4 * thrust).height_m == pytest.approx(2.0 * res.height_m, rel=1e-12)
    crown = dict(fx.MATERIAL, physical=dict(fx.MATERIAL["physical"], elongation_pct={"value": 12}))
    assert _model(crown).drill_burr(thrust).kind == BU.KIND_CROWN


def test_hocheng_dharan_critical_thrust():
    G, E, nu, h = 300.0, 70e9, 0.3, 0.5e-3
    expect = math.pi * math.sqrt(8.0 * G * E * h ** 3 / (3.0 * (1.0 - nu * nu)))
    assert BU.hocheng_dharan_N(G, E, nu, h) == pytest.approx(expect, rel=1e-12)
    assert expect == pytest.approx(275.6, abs=0.5)
    assert BU.hocheng_dharan_N(G, E, nu, 2 * h) == pytest.approx(2 ** 1.5 * expect, rel=1e-12)


def _turning_exit(ap, f=0.215e-3, limit_mm=None):
    """A turning pass along -z leaving the end face z = 0 of a cylinder at radius 25 mm - ap."""
    edge, outline = fx.insert(0.8e-3, kr=KR45, kr2=math.radians(45.0))
    physics = {"burr_limit_mm": limit_mm} if limit_mm is not None else {}
    proj = fx.project(stock={"kind": "cylinder", "size_mm": [50, 60]}, physics=physics)
    ctx = fx.context(proj, tl=fx.timeline(kinds=("linear",), lines=[42]))
    state = fx.sim_state()
    dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}),
                        query=StockQuery("cylinder", [0.05, 0.06]))
    dom.init(ctx, state)
    rho = 0.025 - ap
    omega = -2.0 * math.pi * 1000.0 / 60.0
    w = abs(omega)
    v = f * w / (2.0 * math.pi)
    tk = fx.step_ticks(0, 10 * fx.IPO)
    tt = tk * fx.TICK_S
    theta = -w * tt
    z = 2e-4 - v * tt
    tip = np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1)
    exit_tick = int(tk[-2])
    shock = [exit_tick, 0, 0, f * math.sin(KR45), 0.0, 300.0, 0.0, 2]
    fr = fx.frame(tk, tip, omega, pose_W0_T=fx.turning_pose())
    c = fx.contact(np.zeros((0, 16)), shocks=[shock])
    b = fx.bus([fr], [c], removal=[fx.removal_contact(ap=ap)])
    dom.macro_step(ctx, state, b, fx.bus([fr], []))
    dom.end_operation(ctx, state)
    return dom, ctx, state


@pytest.mark.parametrize("ap", [0.55e-3, 1.0e-3, 2.0e-3])
def test_domain_turning_exit_burr_matches_the_published_case(ap):
    _, _, state = _turning_exit(ap)
    burrs = state.surface.burrs
    assert burrs.n == 1
    assert burrs.bf[0, BF["h_max"]] == pytest.approx(ap, rel=0.2)
    n = burrs.normal(0)
    assert n == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)          # the exit face is the end face
    assert burrs.bi[0, BI["removed"]] == 0


def test_burr_above_the_drawing_limit_raises_a_finding_at_the_program_line():
    _, ctx, state = _turning_exit(1.0e-3, limit_mm=0.2)
    found = [f for f in state.findings if f.category == Category.BURR]
    assert len(found) == 1 and found[0].severity == Severity.WARNING and found[0].line == 42
    assert found[0].quantities["height_m"] == pytest.approx(1.0e-3, rel=0.2)
    doc = SurfaceDomain.document(_dom_for(state, ctx), ctx, state)
    entry = doc["burrs"][0]
    assert entry["line"] == 42 and entry["kind"] == "rollover"
    for key in ("edge", "kind", "height_m", "root_thickness_m", "op", "program", "line", "location_S_m"):
        assert key in entry


def _dom_for(state, ctx):
    dom = SurfaceDomain(tools=fx.tool_info({0: fx.insert(0.8e-3, kr=KR45, kr2=KR45)}),
                        query=StockQuery("cylinder", [0.05, 0.06]))
    dom._attach(ctx, state)
    return dom


def test_a_burr_machined_away_later_is_marked_removed():
    dom, ctx, state = _turning_exit(1.0e-3)

    class Gone:
        def point_inside(self, workpiece, points):
            return np.zeros(len(np.asarray(points).reshape(-1, 3)), bool)

        def ray_cast(self, *a, **k):
            return False, 0.0, np.zeros(3), np.zeros(3)

    dom2 = SurfaceDomain(tools=fx.tool_info({0: fx.insert(0.8e-3, kr=KR45, kr2=KR45)}), query=Gone())
    state.surface.burrs.bi[0, BI["op"]] = state.surface.op_index
    dom2.end_operation(ctx, state)
    assert state.surface.burrs.bi[0, BI["removed"]] == 1
    assert dom2.view(state).burr_height_at(0, state.surface.burrs.location(0)) == 0.0


def test_burr_height_at_a_later_locator():
    dom, ctx, state = _turning_exit(1.0e-3)
    loc = state.surface.burrs.location(0)
    view = dom.view(state)
    assert view.burr_height_at(0, loc, normal=[0.0, 0.0, -1.0]) == pytest.approx(1.0e-3, rel=0.2)
    assert view.burr_height_at(0, loc, normal=[1.0, 0.0, 0.0]) == 0.0          # another face
    assert view.burr_height_at(0, loc + np.array([0.0, 0.0, -5e-3]), radius_m=1e-3) == 0.0


def test_composite_drilling_delamination_finding():
    comp = dict(fx.MATERIAL, family="composite", composite={"fibre": "carbon", "layup": "quasi-isotropic",
                                                            "delamination": {"G_IC_J_m2": {"value": 300},
                                                                             "E_GPa": {"value": 70},
                                                                             "poisson": {"value": 0.3}}})
    edge, outline = fx.endmill(3e-3, 2)
    outline = outline.__class__(**{**outline.__dict__, "kind": "twist_drill"})
    found = {}
    for thrust in (200.0, 400.0):
        proj = fx.project(material=comp, stock={"kind": "block", "size_mm": [60, 40, 5]})
        ctx = fx.context(proj)
        state = fx.sim_state()
        dom = SurfaceDomain(tools=fx.tool_info({0: (edge, outline)}), query=StockQuery("block", [0.06, 0.04, 0.005]))
        dom.init(ctx, state)
        tk = fx.step_ticks(0, 10 * fx.IPO)
        tip = np.tile([0.03, 0.02, 0.5e-3], (len(tk), 1))
        fr = fx.frame(tk, tip, -1000.0)
        wrench = np.zeros((len(tk), 6))
        wrench[:, 2] = -thrust
        c = fx.contact(np.zeros((0, 16)), wrench=wrench)
        dom.macro_step(ctx, state, fx.bus([fr], [c], removal=[fx.removal_contact()]), fx.bus([fr], []))
        found[thrust] = [f for f in state.findings if f.category == Category.BURR]
        if thrust == 400.0:
            dmg = state.surface.damage
            assert len(dmg) == 1 and dmg[0]["uncut_thickness_m"] == pytest.approx(0.5e-3, rel=1e-3)
            assert dmg[0]["critical_thrust_N"] == pytest.approx(BU.hocheng_dharan_N(300.0, 70e9, 0.3, 0.5e-3), rel=1e-2)
    assert found[200.0] == []
    assert len(found[400.0]) == 1 and found[400.0][0].severity == Severity.ERROR
