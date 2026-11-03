"""physics.coolant settings, the project schema node and the model parameter files."""

import pytest

from cnctwin.core.errors import InputError
from cnctwin.coolant.schema import FIELDS, CoolantSettings, schema_node, settings
from cnctwin.db.models import params
from cnctwin.db.schema import Ctx
from cnctwin.db.simap import SIMap


def test_defaults_and_both_key_forms():
    assert settings(None) == CoolantSettings()
    raw = settings({"every_n_macro": 1, "rays_per_jet": 64, "step_max_mm": 5.0})
    assert raw.every_n_macro == 1 and raw.rays_per_jet == 64 and raw.step_max_m == pytest.approx(5e-3)
    assert settings(SIMap({"every_n_macro": 1, "rays_per_jet": 64, "step_max_mm": 5.0})) == raw
    assert CoolantSettings().latency_macro == 1           # architecture 5.1: coolant occlusion k = 1


def test_out_of_range_values_are_rejected():
    for bad in ({"rays_per_jet": 0}, {"every_n_macro": 0}, {"step_max_mm": 0.0}, {"reach_scale": 3.0},
                {"step_min_mm": 20.0, "step_max_mm": 10.0}):
        with pytest.raises(InputError):
            settings(bad)


def test_schema_node_declares_every_field_and_checks_the_template():
    node = schema_node()
    assert set(node.fields) == {spec[0] for spec in FIELDS.values()}
    ctx = Ctx(None, obj="project")
    node.check({"every_n_macro": 1, "rays_per_jet": 64}, ("physics", "coolant"), None, ctx)
    assert not [p for p in ctx.problems if p.is_error]
    from cnctwin.project.schema import physics_schema
    assert physics_schema("coolant") is not None


@pytest.mark.parametrize("model", ["coolant-fluid", "coolant-supply", "coolant-jet", "coolant-heat",
                                   "coolant-lubrication", "coolant-tank", "coolant-occlusion"])
def test_model_parameter_files_load(model):
    assert params(model)
