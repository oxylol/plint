"""physics.tooling settings: declared schema (found by the project schema) and SI reading."""

import math

import pytest

from cnctwin.db.schema import Ctx
from cnctwin.db.simap import wrap
from cnctwin.tooling.schema import PHYSICS_TOOLING, ToolingSettings, settings_from


def test_schema_accepts_declared_keys_and_rejects_unknown():
    ctx = Ctx()
    PHYSICS_TOOLING.check({"rake_class_deg": 2.5, "wave_elements": 10}, ("physics", "tooling"), None, ctx)
    assert not [p for p in ctx.problems if p.severity == "error"]
    ctx = Ctx()
    PHYSICS_TOOLING.check({"rake_klass_deg": 2.5}, ("physics", "tooling"), None, ctx)
    assert any(p.kind == "unknown" for p in ctx.problems)


def test_project_schema_finds_the_tooling_section():
    from cnctwin.project.schema import physics_schema
    node = physics_schema("tooling")
    assert node is not None and "rake_class_deg" in node.fields


def test_settings_from_si_view_and_file_keys():
    physics = wrap({"edge_element_mm": 0.25, "tooling": {"rake_class_deg": 3, "max_elements_per_tooth": 500}})
    s = settings_from(physics)
    assert s.edge_element_m == pytest.approx(0.25e-3)
    assert s.rake_class_rad == pytest.approx(math.radians(3))
    assert s.max_elements_per_tooth == 500
    assert s.incl_class_rad == ToolingSettings().incl_class_rad
    assert settings_from({"edge_element_mm": 1.0}).edge_element_m == pytest.approx(1e-3)
    assert settings_from(None) == ToolingSettings()


def test_element_guard():
    from cnctwin.tooling import build_tool
    from cnctwin.tooling.schema import ToolingSettings as S
    from tests.unit.tooling.fixtures import HOLDER, endmill, instance
    with pytest.raises(ValueError):
        build_tool(endmill(), HOLDER, instance(), S(edge_element_m=1e-6, max_elements_per_tooth=100))
