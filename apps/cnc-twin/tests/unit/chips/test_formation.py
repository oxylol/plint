"""The default chip formation model: cnctwin.mechanics (Oxley + chip form) behind the FormationModel Protocol."""

from __future__ import annotations

import math

import pytest

from cnctwin.chips.domain import ChipsDomain
from cnctwin.chips.formation import MechanicsFormation
from cnctwin.chips.protocols import FormationModel, FormationQuery
from cnctwin.core.bus import CHIP_FORMS
from cnctwin.core.errors import InputError
from cnctwin.db.simap import wrap

import chipsfix as fx


def _query(h=1e-4, v=50.0, continuous=False, edge_kind=0):
    return FormationQuery(contact=0, tool_slot=0, h_m=h, b_m=2e-3, l_m=6e-3, v_m_s=v, rake_rad=0.1, incl_rad=0.5,
                          edge_kind=edge_kind, temp_K=0.0, shear_angle_rad=math.nan, chip_ratio=math.nan,
                          coolant_kind="dry", continuous=continuous)


def test_mechanics_formation_gives_a_physical_chip():
    model = MechanicsFormation(wrap(fx.material_jc_raw()))
    assert isinstance(model, FormationModel)
    f = model.formation(_query())
    assert f.form_code in CHIP_FORMS
    assert 0.1 < f.chip_ratio < 1.0 and f.thickness_m > 1e-4      # the chip is thicker than the uncut chip
    assert 0.0 < f.shear_angle_rad < math.pi / 2 and f.shear_strain > 1.0
    assert f.chip_temp_K > 293.15 and f.length_m > 0.0
    # binned and cached: a query in the same bin gives the same result, a different one solves again
    assert model.formation(_query(h=1.0001e-4)) is f
    assert model.formation(_query(h=2e-4)) is not f
    # continuous cutting with a nose: turning, long chips have no finite piece length -> 0 (one per row)
    g = model.formation(_query(continuous=True, edge_kind=9))
    assert g.length_m >= 0.0 and g.form_code in CHIP_FORMS


def test_below_the_minimum_chip_thickness_no_chip_forms():
    model = MechanicsFormation(wrap(fx.material_jc_raw()))
    f = model.formation(_query(h=0.0))
    assert f.form_code == 70 and f.chip_ratio == 1.0


def test_material_without_flow_stress_is_refused():
    with pytest.raises(InputError):
        MechanicsFormation(wrap(fx.material_raw()))


def test_domain_uses_the_mechanics_formation_by_default():
    import test_domain as td
    proj = fx.project(material=fx.material_jc_raw())
    ctx = fx.context(proj)
    state = fx.sim_state(1)
    dom = ChipsDomain(geometry=fx.scene_box())
    dom.init(ctx, state)
    assert isinstance(dom.formation_model, MechanicsFormation)
    td._run(dom, ctx, state, 3, [0.05, 0.04, 0.035], rows_fn=td._rows())
    rec = state.bus_prev.chips
    assert rec.formation and rec.formation[0].form_code in CHIP_FORMS and rec.formation[0].chip_ratio < 1.0
