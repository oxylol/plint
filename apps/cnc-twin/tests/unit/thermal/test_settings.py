"""physics.thermal settings and model parameter files."""

import pytest

from cnctwin.core.errors import InputError
from cnctwin.db.models import params
from cnctwin.db.simap import SIMap
from cnctwin.thermal.schema import FIELDS, ThermalSettings, schema_node, settings


def test_defaults_and_both_key_forms():
    assert settings(None) == ThermalSettings()
    raw = settings({"every_n_macro": 7, "cell_mm": 3.0})
    assert raw.every_n_macro == 7 and raw.cell_m == pytest.approx(3e-3)
    si = settings(SIMap({"every_n_macro": 7, "cell_mm": 3.0}))
    assert si == raw


def test_out_of_range_values_are_rejected():
    with pytest.raises(InputError):
        settings({"cell_mm": 0.0})
    with pytest.raises(InputError):
        settings({"every_n_macro": 0})


def test_schema_node_declares_every_field():
    node = schema_node()
    assert set(node.fields) == {spec[0] for spec in FIELDS.values()}


@pytest.mark.parametrize("model", ["thermal-convection", "thermal-tool", "thermal-work", "thermal-machine",
                                   "thermal-partition"])
def test_model_parameter_files_load(model):
    values = params(model)
    assert values
