"""physics.gpu: the switch and its tuning keys (cnctwin.gpu.schema)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cnctwin.core.errors import InputError
from cnctwin.db.simap import wrap
from cnctwin.gpu.schema import FIELDS, MODES, THRESHOLD_FIELD, GpuSettings, from_project, schema_node, settings


def test_defaults_and_text_forms():
    assert settings(None) == GpuSettings()
    assert GpuSettings().mode == "auto"
    for mode in MODES:
        assert settings(mode).mode == mode
        assert settings(mode.upper()).mode == mode
    assert settings(True).mode == "on" and settings(False).mode == "off"


def test_mapping_form_and_thresholds():
    s = settings({"mode": "on", "device": 1, "min_rays": 10, "min_slabs": 5, "min_cells": 20, "min_particles": 30})
    assert (s.mode, s.device, s.min_slabs) == ("on", 1, 5)
    assert s.threshold("dexel_subtract") == 10
    assert s.threshold("thermal_work_step") == 20
    assert s.threshold("chips_advance") == 30
    assert set(THRESHOLD_FIELD.values()) <= set(FIELDS)
    partial = settings({"min_cells": 99})
    assert partial.mode == "auto" and partial.min_cells == 99 and partial.min_rays == GpuSettings().min_rays


@pytest.mark.parametrize("bad", ["fast", 3, {"mode": "sometimes"}, {"min_rays": 0}, {"min_rays": 2.5},
                                 {"device": -1}, {"min_cells": True}, {"warp": 32}])
def test_bad_values_are_rejected(bad):
    with pytest.raises(InputError):
        settings(bad)


def test_from_project_reads_every_view():
    assert from_project(None) == GpuSettings()
    assert from_project({"physics": {"gpu": "off"}}).mode == "off"
    assert from_project(SimpleNamespace(physics={"gpu": {"mode": "on"}})).mode == "on"
    assert from_project(wrap({"physics": {"gpu": {"mode": "off", "min_rays": 7}}})).min_rays == 7
    assert from_project(SimpleNamespace(physics=None)) == GpuSettings()


def test_schema_node_checks_project_values():
    from cnctwin.db.schema import Ctx

    node = schema_node()

    def problems(value):
        ctx = Ctx(file="project.yaml")
        node.check(value, ("physics", "gpu"), None, ctx)
        return [p.message for p in ctx.problems]

    assert problems("auto") == [] and problems("off") == [] and problems(True) == []
    assert problems({"mode": "on", "min_particles": 5}) == []
    assert problems("maybe")
    assert problems({"mode": "on", "threads": 4})
    assert problems({"min_rays": 0})
