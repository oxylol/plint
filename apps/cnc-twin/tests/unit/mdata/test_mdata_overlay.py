"""The calibration overlay: paths, setting values in raw objects and projects, files."""

import pytest

from cnctwin.core.errors import InputError
from cnctwin.mdata.overlay import (Target, apply_to_project, apply_to_raw, controller_overrides, factors,
                                   load_overlay, make_overlay, minimal_patch, model_overrides, parse_path,
                                   write_overlay)

MACHINE = {"id": "vmc", "controller": {"block_processing_ms": {"value": 1.0, "src": "est", "note": "x"}},
           "spindles": [{"id": "main", "efficiency": {"value": 0.85, "src": "est"}, "idle_power_W": 700,
                         "thermal": {"time_constant_min": 30}},
                        {"id": "sub", "efficiency": 0.8}],
           "axes": {"X": {"servo": {"kv_1_s": {"value": 40, "src": "est"}, "feedforward": 0.9}}}}
MATERIAL = {"id": "al", "cutting": {"kienzle": [{"tool_class": "carbide", "coating": "any", "kc11_N_mm2": 800, "mc": 0.25},
                                                {"tool_class": "hss", "kc11_N_mm2": 920, "mc": 0.25}]},
            "tool_life_usui": []}


def _overlay():
    entries = {
        "machine.spindles[main].efficiency": {"value": 0.81, "parameter": "spindle_efficiency", "ci": [0.8, 0.82]},
        "machine.spindles[main].thermal.time_constant_min": {"value": 24.5, "parameter": "spindle_thermal_tau_s"},
        "machine.controller.block_processing_ms": {"value": 1.4, "parameter": "block_processing_s"},
        "machine.axes.X.servo.kv_1_s": {"value": 36.0, "parameter": "axis_X_kv_1_s"},
        "material.cutting.kienzle[tool_class=carbide,coating=any].kc11_N_mm2": {"value": 860.0, "parameter": "kc_scale"},
        "material.tool_life_usui[tool_class=carbide,surface=flank].B_K": {"value": 4100.0, "parameter": "usui_B_K"},
        "material.tool_life_usui[tool_class=carbide,surface=flank].rate_unit": {"value": "m/s with sigma_n in Pa, V_s in m/s"},
        "tools[1].runout_um": {"value": 7.0, "parameter": "tool_runout_m"},
        "models.production-energy.axis_seal_drag_N": {"value": 41.0, "parameter": "axis_seal_drag_N"},
        "factors.tool_stiffness[1]": {"value": 0.86, "parameter": "tool_stiffness_factor"},
    }
    return make_overlay(name="test", machine="vmc", material="al", entries=entries, parameters=[])


def test_paths_and_targets():
    assert parse_path("machine.spindles[main].thermal.time_constant_min") == [
        ("machine", None), ("spindles", "main"), ("thermal", None), ("time_constant_min", None)]
    assert parse_path("models.production-energy.axis_seal_drag_N")[1] == ("production-energy", None)
    assert parse_path("material.cutting.kienzle[tool_class=carbide,coating=any].mc")[2][1] == "tool_class=carbide,coating=any"
    for bad in ("plant.x", "machine.spindles[main", "models.only", "machine..x"):
        with pytest.raises(InputError):
            parse_path(bad)
    assert Target("machine.controller.block_processing_ms").file_value(1.4e-3) == pytest.approx(1.4)
    assert Target("machine.spindles[main].thermal.time_constant_min").file_value(1470.0) == pytest.approx(24.5)
    assert Target("tools[1].runout_um").file_value(7e-6) == pytest.approx(7.0)
    assert Target("material.tool_life[0].C_m_min").file_value(30.0) == pytest.approx(1800.0)
    assert Target("material.cutting.kienzle[0].kc11_N_mm2", "scale", 800.0).file_value(1.07) == pytest.approx(856.0)
    assert Target("machine.spindles[main].efficiency").file_value(0.8) == 0.8


def test_apply_to_raw_objects():
    document = _overlay()
    machine = apply_to_raw("machine", MACHINE, document)
    main = machine["spindles"][0]
    assert main["efficiency"]["value"] == 0.81 and main["efficiency"]["src"] == "est"
    assert "interval [0.8, 0.82]" in main["efficiency"]["note"]
    assert main["thermal"]["time_constant_min"]["value"] == 24.5
    assert machine["axes"]["X"]["servo"]["kv_1_s"]["value"] == 36.0
    assert machine["spindles"][1] == MACHINE["spindles"][1]
    assert MACHINE["spindles"][0]["efficiency"]["value"] == 0.85               # the input is not changed
    material = apply_to_raw("material", MATERIAL, document)
    assert material["cutting"]["kienzle"][0]["kc11_N_mm2"]["value"] == 860.0
    assert material["cutting"]["kienzle"][1]["kc11_N_mm2"] == 920
    usui = material["tool_life_usui"][0]                                      # created by the key=value selector
    assert usui["tool_class"] == "carbide" and usui["surface"] == "flank" and usui["B_K"]["value"] == 4100.0
    assert usui["rate_unit"] == "m/s with sigma_n in Pa, V_s in m/s"
    with pytest.raises(InputError, match="is for machine"):
        apply_to_raw("machine", {**MACHINE, "id": "other"}, document)


def test_project_level_entries():
    document = _overlay()
    assert controller_overrides(document) == {"block_processing_ms": 1.4}
    assert model_overrides(document) == {"production-energy": {"axis_seal_drag_N": 41.0}}
    assert factors(document) == {"tool_stiffness": {"1": 0.86}}
    project = {"tools": {"list": [{"number": 1, "runout_um": 5}, {"number": 2}]}, "controller_overrides": {"x_mm": 1}}
    patched = apply_to_project(project, document)
    assert patched["tools"]["list"][0]["runout_um"] == 7.0
    assert patched["controller_overrides"] == {"x_mm": 1, "block_processing_ms": 1.4}
    patch = minimal_patch(MACHINE, document, "machine")
    assert set(patch) == {"spindles", "axes"}                                 # controller goes to the overrides
    assert len(patch["spindles"]) == 2 and patch["axes"] == {"X": {"servo": {"kv_1_s": {
        "value": 36.0, "src": "est", "note": "calibrated (test), parameter axis_X_kv_1_s"}}}}


def test_file_roundtrip_and_checks(tmp_path):
    document = _overlay()
    path = write_overlay(tmp_path / "calibration.yaml", document)
    assert path.read_text().startswith("# CNC Twin calibration overlay")
    back = load_overlay(path)
    assert back["set"]["tools[1].runout_um"]["value"] == 7.0
    write_overlay(tmp_path / "again.yaml", back)
    assert (tmp_path / "again.yaml").read_bytes() == path.read_bytes()
    with pytest.raises(InputError, match="number or text"):
        make_overlay(name="x", machine=None, material=None, entries={"tools[1].runout_um": {"value": [1, 2]}},
                     parameters=[])
    (tmp_path / "future.yaml").write_text("format: cnctwin-calibration\nversion: 9\nset: {}\n")
    with pytest.raises(InputError, match="newer"):
        load_overlay(tmp_path / "future.yaml")
