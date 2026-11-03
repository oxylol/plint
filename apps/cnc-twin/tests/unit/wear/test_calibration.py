"""Usui constants calibrated to the Taylor life: two speeds reproduced, the third predicted within the
documented error, the kernel reaching the VB limit in the Taylor life, coating phase, abrasive cap,
published constants with their unit systems, and the default mechanics response."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.wear import wearfix as fx
from cnctwin.core.errors import InputError
from cnctwin.db.simap import SIMap
from cnctwin.wear import abi as A
from cnctwin.wear import build as B
from cnctwin.wear import calibration as CAL
from cnctwin.wear import kernels as K
from cnctwin.wear import params as P

GEO = CAL.FlankGeometry(rake_rad=0.14, clearance_rad=0.17, edge_radius_m=8e-6, incl_rad=0.0,
                        g_vb=1 / math.tan(0.17) - math.tan(0.14),
                        c_coat=math.cos(0.17) * (1 - math.tan(0.17) * math.tan(0.14)))
NO_ABRASION = CAL.Abrasion(0.0, np.array([293.15]), np.array([1.57e10]))


def _entry():
    return P.taylor_entries(SIMap(fx.MATERIAL).tool_life)[0]


def _response(model):
    return lambda v, vb: model.response(0, CAL.Condition(v, 3e-4, 2e-3, vb, 0.14, 0.17, 8e-6))


def _simulate(law_A, law_B, model, speed, vb_lim, coating=None, abrasion=0.0, hardness=1.57e10, steps=1500):
    """Drive advance_elements for one element at constant cutting conditions until VB reaches vb_lim;
    the temperatures follow the response at the current land. Returns the time (s)."""
    entry = _entry()
    life = entry.life_s(speed, entry.feed_ref_m, entry.ap_ref_m)
    dt = life / steps
    geo = A.new_geo(1, 1)
    geo.valid[:] = 1
    geo.sin_k[:] = 1.0
    geo.rake[:] = GEO.rake_rad
    geo.clear[:] = GEO.clearance_rad
    geo.g_vb[:] = GEO.g_vb
    geo.c_coat[:] = GEO.c_coat
    geo.beta[:] = 0.5 * math.pi - GEO.rake_rad - GEO.clearance_rad
    geo.r_e0[:] = 8e-6
    law = A.new_law(1, 1, 1, 1)
    law.fl_A[:] = law_A
    law.fl_B[:] = law_B
    law.fl_lo[:] = -math.inf
    law.rk_A[:] = law_A
    law.rk_B[:] = law_B
    law.rk_lo[:] = -math.inf
    law.hs_T[:] = 293.15
    law.hs_Pa[:] = hardness
    law.hc_T[:] = 293.15
    law.hc_Pa[:] = hardness if coating is None else 3.2e10
    law.scal[A.LS_ABR] = abrasion
    law.scal[A.LS_COAT_FACTOR] = coating.factor if coating is not None else 1.0
    law.scal[A.LS_COAT_TMAX_K] = coating.max_service_K if coating is not None else 0.0
    law.scal[A.LS_SKIN_FACTOR] = 1.0
    law.scal[A.LS_DOC_RATIO] = 0.5
    law.scal[A.LS_CRACK_GROWTH] = 1.0
    st = A.new_state(1, 1)
    if coating is not None:
        st.coat_f[:] = coating.thickness_m
    out = A.new_output(1, 1)
    inp = A.new_input(1, 1, 3)
    t = 0.0
    while True:
        vb = st.vn[0, 0] * GEO.g_vb
        if vb >= vb_lim:
            return t - (vb - vb_lim) / (out.d_vn[0, 0] * GEO.g_vb / dt)
        r = model.response(0, CAL.Condition(speed, 3e-4, 2e-3, vb, 0.14, 0.17, 8e-6))
        inp.in_cut_s[:] = dt
        inp.sv_flank[:] = r.flank_sv_Pa_m_s * dt
        inp.h_int[:] = 3e-4 * dt
        inp.flank_T[0, 0, :] = r.flank_temp_K
        inp.rake_T[0, 0, :] = r.rake_temp_K
        inp.w_q[:] = r.weights
        inp.scal[A.IS_DT_S] = dt
        K.advance_elements(geo, inp, law, st, out)
        t += dt


def test_taylor_reproduced_at_two_speeds_and_predicted_at_the_third():
    model = fx.AnalyticResponse()
    entry = _entry()
    cal = CAL.calibrate(entry, _response(model), GEO, None, NO_ABRASION, 0.0, 6)
    res = cal.report["residual_rel"]
    assert abs(res[0]) < 1e-9 and abs(res[1]) < 1e-9
    assert abs(res[2]) < 0.05                      # documented: <= 5 % at the 15-min speed (wear.md section 3)
    assert cal.report["bracketed"] and cal.B_K > 0.0 and cal.A_per_Pa > 0.0
    v1, v2, v3 = cal.report["speeds_m_s"]
    assert entry.life_s(v1, entry.feed_ref_m, entry.ap_ref_m) == pytest.approx(3600.0, rel=1e-12)
    assert entry.life_s(v2, entry.feed_ref_m, entry.ap_ref_m) == pytest.approx(300.0, rel=1e-12)


def test_the_kernel_reaches_the_vb_limit_in_the_taylor_life():
    model = fx.AnalyticResponse()
    entry = _entry()
    cal = CAL.calibrate(entry, _response(model), GEO, None, NO_ABRASION, 0.0, 6)
    for i, speed in enumerate(cal.report["speeds_m_s"]):
        life = _simulate(cal.A_per_Pa, cal.B_K, model, speed, entry.VB_limit_m)
        target = cal.report["taylor_life_s"][i]
        tolerance = 5e-3 if i < 2 else 0.05        # step discretisation; the check speed carries the model error
        assert life == pytest.approx(target, rel=tolerance), (i, life, target)


def test_coated_tool_calibration_and_simulation_agree():
    model = fx.AnalyticResponse()
    entry = _entry()
    coating = P.CoatingData("c", 0.5, 3e-6, 1100.0, np.array([293.15]), np.array([3.2e10]))
    cal = CAL.calibrate(entry, _response(model), GEO, coating, NO_ABRASION, 0.0, 6)
    plain = CAL.calibrate(entry, _response(model), GEO, None, NO_ABRASION, 0.0, 6)
    assert abs(cal.report["residual_rel"][0]) < 1e-9 and abs(cal.report["residual_rel"][1]) < 1e-9
    # the coating carries part of the life, so the substrate law must wear faster than without it
    assert cal.A_per_Pa * math.exp(-cal.B_K / 1000.0) > plain.A_per_Pa * math.exp(-plain.B_K / 1000.0)
    life = _simulate(cal.A_per_Pa, cal.B_K, model, cal.report["speeds_m_s"][0], entry.VB_limit_m, coating=coating)
    assert life == pytest.approx(cal.report["taylor_life_s"][0], rel=5e-3)


def test_abrasive_share_is_capped_at_the_calibration_speeds():
    model = fx.AnalyticResponse()
    entry = _entry()
    strong = CAL.Abrasion(1e-6, np.array([293.15]), np.array([1.57e10]))
    cal = CAL.calibrate(entry, _response(model), GEO, None, strong, 0.0, 6)
    share = P.get(P.CALIBRATION)["max_abrasive_share"]
    assert 0.0 < cal.abrasive_scale < 1.0
    nodes = CAL._nodes(_response(model), cal.report["speeds_m_s"][0], 0.0, entry.VB_limit_m, GEO, None, strong, 6)
    abrasive_life = CAL.life_s(nodes, 0.0, 1.0, 1000.0, GEO.g_vb, None, cal.abrasive_scale)
    assert cal.report["taylor_life_s"][0] / abrasive_life <= share * (1 + 1e-9)
    assert abs(cal.report["residual_rel"][0]) < 1e-9 and abs(cal.report["residual_rel"][1]) < 1e-9


def test_published_usui_constants_convert_their_unit_systems():
    def entry(**kw):
        return P.usui_entry(SIMap({"B_K": 5302, **kw}), "test")

    assert entry(A_per_MPa=1.0, rate_unit="mm/min with sigma_n in MPa, V_s in m/min").A_per_Pa == pytest.approx(1e-9)
    assert entry(A_per_MPa=7.8e-9, rate_unit="mm/s with sigma_n in MPa, V_s in mm/s").A_per_Pa == pytest.approx(7.8e-15)
    assert entry(A_per_Pa=7.8e-15, rate_unit="m/s with sigma_n in Pa, V_s in m/s").A_per_Pa == pytest.approx(7.8e-15)
    with pytest.raises(InputError):
        entry(A_per_MPa=1.0)
    with pytest.raises(InputError):
        entry(A=7.8e-15)
    with pytest.raises(InputError):
        entry(A_per_Pa=1.0, rate_unit="mm/min with sigma_n in MPa, V_s in m/min")
    # two branches of one set (Usui 1984 as quoted for P20 / steel): sorted, the first open below
    branches = P.select_usui([entry(A_per_Pa=1.198e-8, B_K=21950, T_range_K=[1150, None]),
                              entry(A_per_Pa=7.8e-15, T_range_K=[None, 1150])], "flank", "steel", "carbide", "none")
    assert [b.T_lo_K for b in branches] == [-math.inf, 1150.0]


def test_published_constants_take_precedence_over_the_taylor_calibration():
    tm = dict(fx.TOOL_MATERIAL)
    tm["usui"] = [{"work_family": "steel", "surface": "flank", "A_per_MPa": 7.8e-9,
                   "rate_unit": "mm/s with sigma_n in MPa, V_s in mm/s", "B_K": 5302},
                  {"work_family": "steel", "surface": "rake", "A_per_Pa": 4e-12,
                   "rate_unit": "m/s with sigma_n in Pa, V_s in m/s", "B_K": 10456}]
    tool_mat = P.tool_material_data(SIMap(tm))
    work = P.work_data(SIMap(fx.MATERIAL))
    estimated = []
    law, report, taylor = B.wear_law(slot=0, work=work, tool_mat=tool_mat, coating=None, coating_id="none",
                                     process="milling", fg=GEO, response=fx.AnalyticResponse(), vb0_m=0.0,
                                     settings=type("S", (), {"calib_nodes": 6})(), estimated=estimated)
    assert report["flank"]["source"] == "published" and taylor is None
    assert law.fl_A[0] == pytest.approx(7.8e-15) and law.fl_B[0] == 5302.0
    assert report["rake"]["source"] == "published" and law.rk_A[0] == pytest.approx(4e-12)


def test_default_mechanics_response_calibrates_a_steel():
    from cnctwin.mechanics.materials import ToolSide, WorkMaterial

    work = WorkMaterial.from_db(SIMap(fx.MATERIAL))
    side = ToolSide.from_db(SIMap(fx.TOOL), SIMap(fx.TOOL_MATERIAL), None)
    response = CAL.MechanicsResponse(work, side, 293.15)
    entry = _entry()
    f = lambda v, vb: response.response(0, CAL.Condition(v, entry.feed_ref_m, entry.ap_ref_m, vb, 0.1, 0.12, 2e-5))
    cal = CAL.calibrate(entry, f, CAL.FlankGeometry(0.1, 0.12, 2e-5, 0.0, 1 / math.tan(0.12) - math.tan(0.1),
                                                     math.cos(0.12) * (1 - math.tan(0.12) * math.tan(0.1))),
                        None, NO_ABRASION, 0.0, 4)
    temps = cal.report["flank_temp_K"]
    assert temps[0] < temps[1]                       # the flank runs hotter at the higher speed
    assert abs(cal.report["residual_rel"][0]) < 1e-9 and abs(cal.report["residual_rel"][1]) < 1e-9
    assert abs(cal.report["residual_rel"][2]) < 0.10      # documented bound for the default response
    assert 2e3 < cal.B_K < 1e5
