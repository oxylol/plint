"""The jet at the cutting engagement (cnctwin/coolant/engagement.py): nozzle jet, face coefficient with the
normal speed component, hydraulic wedge and the contact-length ratio of the chip's moment balance."""

import math

import numpy as np
import pytest

from cnctwin.coolant import engagement as E
from cnctwin.coolant import fluid as F
from cnctwin.coolant import heat as H
from cnctwin.db.models import params

import coolfix as X


def props():
    return F.fluid_from_db(X.fluid_db("emulsion-8pct")).props(293.15)


def test_nozzle_jet_speed_and_continuity():
    pr = props()
    cv = params("coolant-supply")["nozzle_discharge_coefficient"]
    jet = E.nozzle_jet(150e5, 0.7e-3, pr, flow_m3_s=2.7e-3 / 60.0)
    assert jet.speed_m_s == pytest.approx(cv * math.sqrt(2.0 * 150e5 / pr.rho_kg_m3), rel=1e-12)
    assert math.pi * jet.diameter_m ** 2 / 4.0 * jet.speed_m_s == pytest.approx(2.7e-3 / 60.0, rel=1e-12)
    bore = E.nozzle_jet(150e5, 0.7e-3, pr)
    assert bore.diameter_m == 0.7e-3 and bore.flow_m3_s == pytest.approx(bore.speed_m_s * math.pi * 0.7e-3 ** 2 / 4)
    assert bore.momentum_N == pytest.approx(pr.rho_kg_m3 * bore.flow_m3_s * bore.speed_m_s)


def test_face_coefficient_uses_the_normal_speed_and_the_reach():
    pr = props()
    jet = E.nozzle_jet(80e5, 2e-3, pr, flow_m3_s=2e-3 / 60.0)
    normal = E.face_h(jet, 0.02, 0.5 * math.pi, 6e-3, pr)
    assert normal == pytest.approx(H.jet_h(False, jet.speed_m_s, jet.diameter_m, 0.02, 6e-3, pr), rel=1e-12)
    inclined = E.face_h(jet, 0.02, math.radians(30.0), 6e-3, pr)
    assert inclined == pytest.approx(H.jet_h(False, 0.5 * jet.speed_m_s, jet.diameter_m, 0.02, 6e-3, pr), rel=1e-12)
    assert 0.0 < inclined < normal
    assert E.face_h(jet, 0.02, 0.5 * math.pi, 6e-3, pr, reach=0.4) == pytest.approx(0.4 * normal, rel=1e-12)
    assert E.face_h(jet, 0.02, 0.0, 6e-3, pr) == 0.0


def test_wedge_takes_only_the_momentum_against_the_chip_flow():
    pr = props()
    jet = E.nozzle_jet(150e5, 0.7e-3, pr)
    n = [0.0, 0.0, 1.0]
    flow = [1.0, 0.0, 0.0]
    a = math.radians(30.0)
    against = [-math.cos(a), 0.0, -math.sin(a)]               # down onto the rake, running against the chip
    assert E.wedge_force_N(jet, against, n, flow) == pytest.approx(jet.momentum_N * math.cos(a), rel=1e-12)
    along = [math.cos(a), 0.0, -math.sin(a)]                  # with the chip: no wedge
    assert E.wedge_force_N(jet, along, n, flow) == 0.0
    assert E.wedge_force_N(jet, [0.0, 0.0, -1.0], n, flow) == 0.0          # normal to the rake face
    assert E.wedge_force_N(jet, against, n, flow, reach=0.5) == pytest.approx(0.5 * jet.momentum_N * math.cos(a))


def test_contact_length_ratio_from_the_moment_balance():
    # N' + F = N and N' l'/3 + F l' = N l/3  ->  l'/l = N / (N + 2 F)
    assert E.contact_length_ratio(10.0, 100.0) == pytest.approx(100.0 / 120.0, rel=1e-12)
    assert E.contact_length_ratio(0.0, 100.0) == 1.0
    floor = params("coolant-heat")["wedge_min_contact_ratio"]
    assert E.contact_length_ratio(1e4, 100.0) == pytest.approx(floor)
    n, f, lc = 100.0, 10.0, 1.0
    lp = E.contact_length_ratio(f, n) * lc
    assert (n - f) * lp / 3.0 + f * lp == pytest.approx(n * lc / 3.0, rel=1e-12)
