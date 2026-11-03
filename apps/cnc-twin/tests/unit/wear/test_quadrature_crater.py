"""Usui's rate is evaluated at every in-cut quadrature temperature (never at the mean temperature), and the
crater appears only once the rake is hot enough (Arrhenius onset, two-branch constants, domain event)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.wear import wearfix as fx
from cnctwin.wear import abi as A
from cnctwin.wear import kernels as K
from cnctwin.wear.domain import WearDomain


def _one_element(flank_T, rake_T, weights, fl=(1e-9, 20000.0), rk=None, lo=None):
    geo = A.new_geo(1, 1)
    geo.valid[:] = 1
    geo.sin_k[:] = 1.0
    geo.g_vb[:] = 5.0
    geo.c_coat[:] = 1.0
    law = A.new_law(len(np.atleast_1d(fl[0])), len(np.atleast_1d((rk or fl)[0])), 1, 1)
    law.fl_A[:] = fl[0]
    law.fl_B[:] = fl[1]
    law.fl_lo[:] = -math.inf if lo is None else lo
    law.rk_A[:] = (rk or fl)[0]
    law.rk_B[:] = (rk or fl)[1]
    law.rk_lo[:] = -math.inf if lo is None else lo
    law.hs_T[:] = 293.15
    law.hs_Pa[:] = 1.6e10
    law.hc_T[:] = 293.15
    law.hc_Pa[:] = 1.6e10
    law.scal[A.LS_SKIN_FACTOR] = 1.0
    law.scal[A.LS_DOC_RATIO] = 0.5
    law.scal[A.LS_CRACK_GROWTH] = 1.0
    inp = A.new_input(1, 1, len(weights))
    inp.in_cut_s[:] = 1e-3
    inp.sv_flank[:] = 1e9 * 3.0 * 1e-3
    inp.sv_rake[:] = 0.6e9 * 1.2 * 1e-3
    inp.h_int[:] = 1e-4 * 1e-3
    inp.flank_T[0, 0, :] = flank_T
    inp.rake_T[0, 0, :] = rake_T
    inp.w_q[:] = weights
    inp.scal[A.IS_DT_S] = 0.01
    st = A.new_state(1, 1)
    out = A.new_output(1, 1)
    K.advance_elements(geo, inp, law, st, out)
    return out


def test_quadrature_is_not_the_rate_of_the_mean_temperature():
    weights = np.array([0.25, 0.5, 0.25])
    temps = np.array([900.0, 1000.0, 1100.0])
    B = 20000.0
    quad = _one_element(temps, temps, weights, fl=(1e-9, B)).d_vn[0, 0]
    mean = _one_element(np.full(3, float(weights @ temps)), np.full(3, 1000.0), weights, fl=(1e-9, B)).d_vn[0, 0]
    expected = float(weights @ np.exp(-B / temps)) / math.exp(-B / float(weights @ temps))
    assert quad / mean == pytest.approx(expected, rel=1e-12)
    assert quad / mean > 1.15                      # Jensen: the hot part of the in-cut interval dominates
    # no spread: the quadrature and the mean agree
    flat = _one_element(np.full(3, 1000.0), np.full(3, 1000.0), weights, fl=(1e-9, B)).d_vn[0, 0]
    assert flat == pytest.approx(mean, rel=1e-14)


def test_crater_rate_follows_the_arrhenius_onset():
    weights = np.array([1.0])
    rates = []
    for t in (700.0, 900.0, 1100.0, 1300.0):
        rates.append(_one_element([900.0], [t], weights, fl=(1e-9, 20000.0)).d_kt[0, 0])
    rates = np.array(rates)
    assert np.all(np.diff(rates) > 0)
    assert rates[0] < 1e-3 * rates[-1]             # below ~800 K the crater does not grow
    # two published branches (P20 / steel as quoted): the diffusion branch takes over at 1150 K
    two = (np.array([7.8e-15, 1.198e-8]), np.array([5302.0, 21950.0]))
    below = _one_element([900.0], [1149.0], weights, fl=two, lo=np.array([-math.inf, 1150.0])).d_kt[0, 0]
    above = _one_element([900.0], [1151.0], weights, fl=two, lo=np.array([-math.inf, 1150.0])).d_kt[0, 0]
    assert above / below == pytest.approx(1.198e-8 * math.exp(-21950 / 1151.0) / (7.8e-15 * math.exp(-5302 / 1149.0)),
                                          rel=1e-12)


def test_domain_reports_the_crater_onset():
    proj = fx.project()
    ctx = fx.context(proj)
    state = fx.sim_state()
    edge = fx.edge_model()
    dom = WearDomain(tools=fx.tool_info(edge), response=fx.AnalyticResponse(), catalog=fx.Catalog())
    dom.init(ctx, state)
    cold = fx.heat(edge, flank_K=700.0, rake_K=750.0, spread=0.0, weights=(1.0,))
    for i in range(3):
        fx.step(dom, ctx, state, fx.bus(edge, [fx.cutting(edge, duty=1.0)], cold, fx.path_frame(tick0=i * 10000)))
    assert float(np.max(state.wear.slots[0].elem.kt)) < 1e-9
    assert not any(e.event == "wear:crater_onset" for e in ctx.trace._memory)
    hot = fx.heat(edge, flank_K=1100.0, rake_K=1500.0, spread=0.0, weights=(1.0,))
    for i in range(3, 200):
        fx.step(dom, ctx, state, fx.bus(edge, [fx.cutting(edge, duty=1.0)], hot, fx.path_frame(tick0=i * 10000)))
        if state.wear.slots[0].flags["crater_onset"]:
            break
    assert state.wear.slots[0].flags["crater_onset"]
    events = [e for e in ctx.trace._memory if e.event == "wear:crater_onset"]
    assert len(events) == 1 and events[0].quantities["KT_m"] >= 1e-6
