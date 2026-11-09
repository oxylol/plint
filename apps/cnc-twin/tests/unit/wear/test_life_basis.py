"""Tool-life basis of the Taylor calibration (docs/models/wear.md section 3, "Life basis"): what a Taylor
life counts (cutting time or edge contact time), the defaults by process, the conversion with the engaged
fraction of the revolution, and whole-domain runs at the Taylor reference conditions that reach the VB
limit in the Taylor life of cutting time (a milling cut at its reference engagement, a turning cut)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.wear import wearfix as fx
from cnctwin.core.errors import InputError
from cnctwin.db.simap import SIMap
from cnctwin.wear import calibration as CAL
from cnctwin.wear import params as P
from cnctwin.wear.domain import WearDomain

ISO_8688_ARC = math.acos(1.0 - 2.0 * 0.1)          # ISO 8688-2 Table 3 I/II: ae 2.5 mm with D 25 mm
STEPS_PER_LIFE = 400
TURNING_TOOL = {"id": "test-insert", "type": "turning_insert", "hand": "right",
                "tool_material": {"class": "carbide", "id": "test-carbide"}, "coating": {"id": "none"},
                "geometry": {"nose_radius_mm": 0.8}}


def _taylor(**changes) -> dict:
    entry = dict(fx.MATERIAL["tool_life"][0])
    entry.update(changes)
    return {key: value for key, value in entry.items() if value is not None}


def _material(**changes) -> dict:
    material = dict(fx.MATERIAL)
    material["tool_life"] = [_taylor(**changes)]
    return material


def _entry(**changes) -> P.TaylorEntry:
    return P.taylor_entries(SIMap(_material(**changes)).tool_life)[0]


def test_life_basis_defaults_by_process_and_the_contact_fraction():
    turning = _entry(process="turning")
    assert (turning.life_basis, turning.contact_fraction, turning.engagement) == ("contact_time", 1.0, "contact")
    milling = _entry(process="milling")
    assert milling.life_basis == "cutting_time" and milling.engagement == "default"
    assert milling.contact_fraction == pytest.approx(ISO_8688_ARC / (2.0 * math.pi), rel=1e-4)
    stated = _entry(process="milling", engaged_arc_ref_deg=90.0)
    assert (stated.contact_fraction, stated.engagement) == (pytest.approx(0.25), "stated")
    for process in ("drilling", "tapping", "any", None):           # the edge is in the cut while the tool feeds
        entry = _entry(process=process)
        assert (entry.life_basis, entry.contact_fraction, entry.engagement) == ("cutting_time", 1.0, "continuous")
    explicit = _entry(process="milling", life_basis="contact_time")
    assert (explicit.life_basis, explicit.contact_fraction) == ("contact_time", 1.0)
    assert stated.contact_life_s(3.0, stated.feed_ref_m, stated.ap_ref_m) == \
        pytest.approx(0.25 * stated.life_s(3.0, stated.feed_ref_m, stated.ap_ref_m), rel=1e-15)
    with pytest.raises(InputError):
        _entry(life_basis="machine_time")
    with pytest.raises(InputError):
        _entry(process="milling", engaged_arc_ref_deg=400.0)


def test_the_calibration_targets_the_contact_time_of_the_reference_cut():
    model = fx.AnalyticResponse()
    geo = CAL.FlankGeometry(0.14, 0.17, 8e-6, 0.0, 1 / math.tan(0.17) - math.tan(0.14),
                            math.cos(0.17) * (1 - math.tan(0.17) * math.tan(0.14)))
    none = CAL.Abrasion(0.0, np.array([293.15]), np.array([1.57e10]))
    response = lambda v, vb: model.response(0, CAL.Condition(v, 3e-4, 2e-3, vb, 0.14, 0.17, 8e-6))
    milling = CAL.calibrate(_entry(process="milling", engaged_arc_ref_deg=90.0), response, geo, None, none, 0.0, 6)
    turning = CAL.calibrate(_entry(process="turning"), response, geo, None, none, 0.0, 6)
    rep = milling.report
    assert rep["life_basis"] == "cutting_time" and rep["contact_fraction"] == pytest.approx(0.25)
    assert rep["contact_life_s"] == pytest.approx([0.25 * t for t in rep["taylor_life_s"]], rel=1e-12)
    assert abs(rep["residual_rel"][0]) < 1e-9 and abs(rep["residual_rel"][1]) < 1e-9
    assert turning.report["contact_life_s"] == pytest.approx(turning.report["taylor_life_s"], rel=1e-15)
    # the same Taylor curve in a quarter of the contact time: a 4 x faster law at the same temperatures
    assert milling.B_K == pytest.approx(turning.B_K, rel=1e-6)
    assert milling.A_per_Pa == pytest.approx(4.0 * turning.A_per_Pa, rel=1e-6)


def _run_to_vb_limit(material, tool, edge, speed, edge_fraction, life_s, rotating=True):
    """Whole-domain run at constant conditions until the flank land VB_B reaches the Taylor VB limit; the
    flank temperatures follow the calibration response at the current land (the run then uses the
    calibration's thermal model, so only the time basis is under test). Returns the cutting time (s)."""
    project = fx.project(material=material, tool=tool)
    ctx = fx.context(project, macro_ipo=max(1, int(round(life_s / STEPS_PER_LIFE / 1e-3))))
    state = fx.sim_state()
    response = fx.AnalyticResponse()
    dom = WearDomain(tools=fx.tool_info(edge, rotating=rotating), response=response, catalog=fx.Catalog())
    dom.init(ctx, state)
    vb_lim = state.wear.slots[0].props["limits"]["vb"]
    dt = ctx.macro_s
    cut_s, vb_now = 0.0, 0.0
    for i in range(20 * STEPS_PER_LIFE):
        temps = response.temps(speed, vb_now)
        heat = fx.heat(edge, flank_K=float(temps[1]), rake_K=float(temps[1]), spread=response.spread)
        cut = fx.cutting(edge, speed=speed, duty=1.0, dt=dt, edge_fraction=edge_fraction, entries=0,
                         sigma_f=response.sigma)
        tick0 = i * ctx.ipo_ticks * ctx.macro_ipo
        record = fx.step(dom, ctx, state, fx.bus(edge, [cut], heat, fx.path_frame(tick0=tick0, n_ticks=ctx.macro_ipo),
                                                 tick=tick0 + ctx.ipo_ticks * ctx.macro_ipo))
        vb_new = float(np.max(np.mean(record.tools[0].VB_m, axis=1)))        # VB_B of the most worn tooth
        if vb_new >= vb_lim:
            return cut_s + dt * (vb_lim - vb_now) / (vb_new - vb_now), state
        cut_s += dt
        vb_now = vb_new
    raise AssertionError("the flank land never reached the VB limit")


@pytest.mark.parametrize("which", [0, 1, 2])
def test_a_milling_cut_at_the_taylor_reference_conditions_reaches_vb_lim_in_the_taylor_cutting_time(which):
    """Milling entry counted in cutting time with a 90 deg reference arc (each element in the material a
    quarter of every revolution): at the reference engagement the land reaches VB_lim after T(vc) of cutting
    time (calibration speeds: the step discretisation, 1 %; the 15-min check speed: the documented 6 %)."""
    material = _material(process="milling", engaged_arc_ref_deg=90.0)
    entry = P.taylor_entries(SIMap(material).tool_life)[0]
    speed = CAL.calibration_speeds(entry)[which]
    life = entry.life_s(speed, entry.feed_ref_m, entry.ap_ref_m)
    cut_s, state = _run_to_vb_limit(material, fx.TOOL, fx.edge_model(), speed, 0.25, life)
    tolerance = 0.01 if which < 2 else 0.06
    assert cut_s == pytest.approx(life, rel=tolerance)
    sw = state.wear.slots[0]
    calib = sw.info["law"]["flank"]["calibration"]
    assert calib["life_basis"] == "cutting_time" and calib["contact_fraction"] == pytest.approx(0.25)
    # wear followed the elements' own contact time: a quarter of the cutting time
    assert float(np.max(sw.elem.cut_s)) == pytest.approx(0.25 * sw.counters["in_cut_s"], rel=1e-12)


def test_engagement_other_than_the_reference_scales_the_cutting_time_life():
    """The law is per contact second: the same cut with the elements in contact for all of the cutting
    time (4 x the reference engagement) wears out in a quarter of the Taylor cutting time."""
    material = _material(process="milling", engaged_arc_ref_deg=90.0)
    entry = P.taylor_entries(SIMap(material).tool_life)[0]
    speed = CAL.calibration_speeds(entry)[0]
    life = entry.life_s(speed, entry.feed_ref_m, entry.ap_ref_m)
    cut_s, _ = _run_to_vb_limit(material, fx.TOOL, fx.edge_model(), speed, 1.0, 0.25 * life)
    assert cut_s == pytest.approx(0.25 * life, rel=0.01)


def test_a_milling_entry_without_reference_engagement_uses_the_iso_8688_arc():
    project = fx.project(material=_material(process="milling"))
    ctx = fx.context(project)
    state = fx.sim_state()
    dom = WearDomain(tools=fx.tool_info(), response=fx.AnalyticResponse(), catalog=fx.Catalog())
    dom.init(ctx, state)
    calib = state.wear.slots[0].info["law"]["flank"]["calibration"]
    assert calib["engagement"] == "default"
    assert calib["contact_fraction"] == pytest.approx(ISO_8688_ARC / (2.0 * math.pi), rel=1e-4)
    assert any("ISO 8688-2" in text for text in state.wear.estimated)


@pytest.mark.parametrize("which", [0, 1])
def test_a_turning_cut_reaches_vb_lim_in_the_taylor_cutting_time(which):
    """Turning entry (contact time = cutting time): a single-point tool in continuous contact reaches VB_lim
    after T(vc) of cutting time."""
    material = _material(process="turning")
    entry = P.taylor_entries(SIMap(material).tool_life)[0]
    assert entry.life_basis == "contact_time"
    speed = CAL.calibration_speeds(entry)[which]
    life = entry.life_s(speed, entry.feed_ref_m, entry.ap_ref_m)
    edge = fx.edge_model(nt=1, ne=8, length_m=2e-3, radius_m=0.02)
    cut_s, state = _run_to_vb_limit(material, TURNING_TOOL, edge, speed, 1.0, life, rotating=False)
    assert state.wear.slots[0].props["process"] == "turning"
    assert cut_s == pytest.approx(life, rel=0.01)
