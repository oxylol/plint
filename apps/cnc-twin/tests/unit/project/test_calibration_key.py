"""``calibration: <file>`` in project.yaml: the overlay (cnctwin.mdata.overlay) applied at resolve time to the tool
list, the machine (after controller_overrides) and the material, hashed as an input; project.calibrated.yaml of a
project that includes an overlay bakes both in."""

import pytest
import yaml

from cnctwin.core.errors import InputError
from cnctwin.mdata.overlay import calibrated_project, make_overlay, write_overlay
from cnctwin.project import project_hash, resolve

ENTRIES = {
    "machine.axes.X.servo.kv_1_s": {"value": 36.0, "parameter": "axis_X_kv_1_s", "ci": [35.0, 37.0]},
    "machine.controller.block_processing_ms": {"value": 1.4, "parameter": "block_processing_s"},
    "material.cutting.kienzle[tool_class=carbide].kc11_N_mm2": {"value": 760.0, "parameter": "kc_scale"},
    "tools[1].runout_um": {"value": 7.0, "parameter": "tool_runout_m"},
    "models.production-energy.axis_seal_drag_N": {"value": 41.0, "parameter": "axis_seal_drag_N"},
    "factors.tool_stiffness[1]": {"value": 0.9, "parameter": "tool_stiffness_factor"},
}


def overlay(name="fixture calibration", machine="test-vmc", material="test-al", entries=None):
    return make_overlay(name=name, machine=machine, material=material, entries=dict(entries or ENTRIES),
                        parameters=[])


def include(project_dir, document, extra: str = "") -> None:
    write_overlay(project_dir / "calibration.yaml", document)
    text = (project_dir / "project.yaml").read_text(encoding="utf-8")
    (project_dir / "project.yaml").write_text(text + "calibration: calibration.yaml\n" + extra, encoding="utf-8")


def test_the_overlay_is_applied_at_resolve_time_and_hashed(project_dir, fixture_db):
    plain = resolve(project_dir, fixture_db)
    assert plain.calibration_overlay is None and float(plain.machine.axes.X.servo.kv_1_s) == 40.0
    include(project_dir, overlay(), "controller_overrides: {block_processing_ms: 2.0}\n")
    p = resolve(project_dir, fixture_db)
    kv = p.machine.axes.X.servo.kv_1_s
    assert float(kv) == pytest.approx(36.0) and kv.src == "est" and "calibrated (fixture calibration)" in kv.note
    assert "interval [35, 37]" in kv.note
    # a calibrated controller value wins over the project's own override, and keeps its note
    assert float(p.machine.controller.block_processing_s) == pytest.approx(1.4e-3)
    assert float(p.material.cutting.kienzle[0].kc11_Pa) == pytest.approx(760e6)
    assert p.material.cutting.kienzle[0].kc11_Pa.estimated
    assert p.tools.list[0].runout_m == pytest.approx(7e-6) and p.tools.list[1].runout_m != pytest.approx(7e-6)
    assert float(p.material.physical.density_kg_m3) == 2700 and p.machine.axes.Y.servo.kv_1_s == 40.0   # the rest
    assert str(p.calibration).endswith("calibration.yaml")                 # the SI view holds the absolute path
    record = p.calibration_overlay
    assert record["name"] == "fixture calibration" and record["file"].endswith("calibration.yaml")
    assert record["models"] == {"production-energy": {"axis_seal_drag_N": 41.0}}
    assert record["factors"] == {"tool_stiffness": {"1": 0.9}}
    assert "tools[1].runout_um" in record["applied"] and len(record["applied"]) == 4
    warnings = [q for q in p.problems if q.kind == "calibration"]
    assert len(warnings) == 2 and all(q.severity == "warning" and "not applied" in q.message for q in warnings)
    # an input: in the input hashes, and a changed value changes the project hash
    assert "calibration.yaml" in p.input_hashes
    first = project_hash(p)
    write_overlay(project_dir / "calibration.yaml", overlay(entries={**ENTRIES, "tools[1].runout_um": {"value": 8.0}}))
    again = resolve(project_dir, fixture_db)
    assert again.tools.list[0].runout_m == pytest.approx(8e-6) and project_hash(again) != first


def test_overlay_problems_are_project_errors(project_dir, fixture_db):
    include(project_dir, overlay(machine="another-vmc"))
    with pytest.raises(InputError, match="calibration overlay is for machine 'another-vmc'"):
        resolve(project_dir, fixture_db)
    write_overlay(project_dir / "calibration.yaml", overlay(entries={"tools[9].runout_um": {"value": 7.0}}))
    with pytest.raises(InputError, match="the project has 0 tools numbered 9"):
        resolve(project_dir, fixture_db)
    write_overlay(project_dir / "calibration.yaml", overlay(entries={"tools[1].runout_um": {"value": 5000.0}}))
    with pytest.raises(InputError, match=r"makes tools.list\[0\] invalid"):
        resolve(project_dir, fixture_db)
    (project_dir / "calibration.yaml").write_text("format: cnctwin-calibration\nversion: 9\nset: {}\n")
    with pytest.raises(InputError, match="newer than this build reads"):
        resolve(project_dir, fixture_db)
    (project_dir / "calibration.yaml").unlink()
    with pytest.raises(InputError, match="calibration.yaml' does not exist"):
        resolve(project_dir, fixture_db)


def test_calibrated_project_of_a_calibrated_project_applies_each_value_once(project_dir, fixture_db, tmp_path):
    include(project_dir, overlay())
    first = resolve(project_dir, fixture_db)
    second = overlay(name="second", entries={"machine.axes.X.servo.kv_1_s": {"value": 33.0},
                                             "machine.axes.Y.servo.kv_1_s": {"value": 31.0}})
    out = calibrated_project(project_dir / "project.yaml", second, tmp_path / "project.calibrated.yaml",
                             machine_raw=first.machine.raw, material_raw=first.material.raw)
    written = yaml.safe_load(out.read_text())
    assert "calibration" not in written
    p = resolve(out, fixture_db)
    assert float(p.machine.axes.X.servo.kv_1_s) == pytest.approx(33.0)          # the newer overlay wins
    assert float(p.machine.axes.Y.servo.kv_1_s) == pytest.approx(31.0)
    assert "calibrated (second)" in p.machine.axes.Y.servo.kv_1_s.note
    assert float(p.material.cutting.kienzle[0].kc11_Pa) == pytest.approx(760e6)  # the included one is kept
    assert "calibrated (fixture calibration)" in p.material.cutting.kienzle[0].kc11_Pa.note
    assert p.tools.list[0].runout_m == pytest.approx(7e-6)
    assert float(p.machine.controller.block_processing_s) == pytest.approx(1.4e-3)
