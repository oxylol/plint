import math

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.project import load_project, migrate, project_hash, resolve
from conftest import PROJECT, write


def test_resolve_mirrors_the_template_with_db_objects(project_dir, fixture_db):
    p = resolve(project_dir, fixture_db)
    assert p.machine.id == "test-vmc" and p.machine.axes.X.travel_m == [-0.5, 0.0]
    assert p.material.id == "test-al" and float(p.material.physical.density_kg_m3) == 2700
    assert p.material.batch.kc_scale_sigma == 0.04 and p.material.batch.hardness_sigma_frac == pytest.approx(0.03)
    t1 = p.tools.list[0]
    assert t1.tool.id == "test-em-10" and t1.holder.id == "test-er32"
    assert t1.stickout_m == pytest.approx(0.035) and t1.runout_m == pytest.approx(5e-6)
    t2 = p.tools.list[1]
    assert t2.tool.geometry.diameter_m == pytest.approx(0.00598) and t2.tool.geometry.flutes == 4
    assert p.coolant.fluid.id == "test-emulsion" and p.coolant.concentration_frac == pytest.approx(0.08)
    op = p.operations[0]
    assert op.setup.workholding[0].device.id == "test-vise"
    assert op.setup.workholding[0].input_torque_Nm == 80
    assert op.program_text.startswith("%\nO0010") and op.program_path.endswith("op10.nc")
    assert op.setup.work_offsets_m.G54 == pytest.approx([-0.25, -0.2, -0.4])
    assert p.environment.ambient_K == pytest.approx(293.15)
    assert p.stock.size_m == pytest.approx([0.1, 0.08, 0.03])
    assert p.physics.max_rad_per_step == pytest.approx(math.radians(0.5))
    assert p.physics.tick_s == 1e-6 and p.physics.thermal.cell_m == pytest.approx(0.002)
    assert p.physics.macro_ipo == 10                                  # documented default (template)
    pose = p.stock_pose_W_S
    assert isinstance(pose, np.ndarray) and pose.shape == (4, 4)
    assert pose[:3, 3] == pytest.approx([0.01, 0.02, 0.0])
    assert pose[:3, :3] @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0])
    assert p.operation("OP10").fixture_pose_F_H[2, 3] == pytest.approx(-0.46)
    assert p.programs["OP10"] == op.program_text
    assert p.raw["operations"][0]["program"] == "op10.nc"


def test_input_hashes_and_project_hash(project_dir, fixture_db):
    p = resolve(project_dir, fixture_db)
    names = set(p.input_hashes)
    assert {"project.yaml", "op10.nc", "db/machines/test-vmc.yaml", "db/materials/test-al.yaml",
            "db/tools/test-endmills.yaml", "db/holders/test-holders.yaml", "db/coolants/test-emulsion.yaml",
            "db/workholding/test-vises.yaml", "db/tool_materials/test-carbide.yaml",
            "db/coatings/test-tialn.yaml", "db/controllers/test-ctrl.yaml"} <= names
    assert all(len(h) == 64 for h in p.input_hashes.values())
    first = project_hash(p)
    assert first == project_hash(resolve(project_dir, fixture_db))
    text = (project_dir / "project.yaml").read_text()
    (project_dir / "project.yaml").write_text("# a comment changes nothing\n" + text)
    assert project_hash(resolve(project_dir, fixture_db)) == first
    (project_dir / "op10.nc").write_text("%\nO0010\nM30\n%\n")
    assert project_hash(resolve(project_dir, fixture_db)) != first


def test_validation_errors_name_file_and_line(project_dir, fixture_db):
    text = PROJECT.replace("  form: rolled_plate\n", "  form: rolled_plate\n  colour: blue\n") \
                  .replace("stickout_mm: 35", "stickout_in: 35").replace("program: op10.nc", "program: missing.nc")
    (project_dir / "project.yaml").write_text(text)
    with pytest.raises(InputError) as info:
        load_project(project_dir)
    message = str(info.value)
    assert "project.yaml:14: stock.colour: unknown key 'colour'" in message
    assert "tools.list[0].stickout_in" in message and "declared unit of 'stickout' is mm" in message
    (project_dir / "project.yaml").write_text(PROJECT.replace("program: op10.nc", "program: missing.nc")
                                              .replace("tool: test-em-10", "tool: no-such-tool"))
    with pytest.raises(InputError) as info:
        resolve(project_dir, fixture_db)
    message = str(info.value)
    assert "operations[0].program: file 'missing.nc' does not exist" in message
    assert "project.yaml:17: tools.list[0].tool: no tool 'no-such-tool'" in message


def test_duplicate_tool_numbers_and_references(project_dir, fixture_db):
    text = PROJECT.replace("    - number: 2\n", "    - number: 1\n").replace("nozzle_positions: {1: 12}",
                                                                          "nozzle_positions: {9: 12}")
    (project_dir / "project.yaml").write_text(text)
    with pytest.raises(InputError) as info:
        load_project(project_dir)
    message = str(info.value)
    assert "tool number 1 (path 1) is used twice" in message and "tool 9 is not in tools.list" in message


def test_holder_without_published_geometry_is_refused_with_a_hint(project_dir, fixture_db):
    (project_dir / "project.yaml").write_text(PROJECT.replace("holder: test-er32, stickout_mm: 35",
                                                              "holder: test-unpublished, stickout_mm: 35"))
    with pytest.raises(InputError) as info:
        resolve(project_dir, fixture_db)
    assert "does not give nose_diameter_mm, profile_mm" in str(info.value)
    assert "holder: {extends: test-unpublished" in str(info.value)
    inline = PROJECT.replace("holder: test-er32, stickout_mm: 35",
                             "holder: {extends: test-unpublished, nose_diameter_mm: 40, "
                             "profile_mm: [[0, 40], [80, 40]]}, stickout_mm: 35")
    (project_dir / "project.yaml").write_text(inline)
    p = resolve(project_dir, fixture_db)
    assert p.tools.list[0].holder.profile_m[1] == pytest.approx([0.08, 0.04])
    assert p.tools.list[0].holder.gauge_length_m == pytest.approx(0.08)


def test_controller_overrides_and_local_db_files(project_dir, fixture_db):
    machine = (fixture_db.root / "machines" / "test-vmc.yaml").read_text()
    write(project_dir / "machines" / "local.yaml", machine.replace("id: test-vmc", "id: local"))
    text = PROJECT.replace("machine: test-vmc", "machine: machines/local.yaml\ncontroller_overrides: "
                                                "{lookahead_blocks: 400, in_position_mm: 0.002}")
    (project_dir / "project.yaml").write_text(text)
    p = resolve(project_dir, fixture_db)
    assert p.machine.id == "local" and "machines/local.yaml" in p.input_hashes
    assert p.machine.controller.lookahead_blocks == 400 and p.machine.controller.in_position_m == pytest.approx(2e-6)
    text = PROJECT.replace("machine: test-vmc", "machine: test-vmc\ncontroller_overrides: {lookahead_blocks: 400}")
    (project_dir / "project.yaml").write_text(text)
    p = resolve(project_dir, fixture_db)
    assert p.machine.controller.lookahead_blocks == 400 and p.machine.controller.dialect == "haas"
    (project_dir / "project.yaml").write_text(text.replace("{lookahead_blocks: 400}", "{lookahead_blocks: -3}"))
    with pytest.raises(InputError):
        load_project(project_dir)


V0 = """\
name: "old flat project"
machine: test-vmc
material: test-al
coolant: test-emulsion
stock: {kind: block, size_mm: [100, 80, 30]}
tool_list:
  - {number: 1, tool: test-em-10, holder: test-er32, stickout_mm: 35}
program: op10.nc
setup:
  stock_pose: {translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}
"""


def test_version_0_is_migrated_forward(project_dir, fixture_db):
    from cnctwin.db.yamlio import loads
    data = migrate(loads(V0)[0])
    assert data["format"] == "cnctwin-project" and data["version"] == 1
    assert data["material"] == {"id": "test-al"} and data["coolant"] == {"fluid": "test-emulsion"}
    assert data["tools"] == {"list": [{"number": 1, "tool": "test-em-10", "holder": "test-er32", "stickout_mm": 35}]}
    assert data["operations"][0]["name"] == "OP10" and data["operations"][0]["program"] == "op10.nc"
    (project_dir / "project.yaml").write_text(V0)
    p = resolve(project_dir, fixture_db)
    assert p.tools.list[0].tool.id == "test-em-10" and p.operations[0].program_text.startswith("%")
    assert load_project(project_dir).version_written == 0
    (project_dir / "project.yaml").write_text(V0.replace("stickout_mm: 35", "stickout_mm: 35, colour: red"))
    with pytest.raises(InputError) as info:
        load_project(project_dir)
    assert "project.yaml:7: tools.list[0].colour" in str(info.value)       # line of the v0 file
    (project_dir / "project.yaml").write_text("version: 5\nname: x\n")
    with pytest.raises(InputError) as info:
        load_project(project_dir)
    assert "newer than this build" in str(info.value)


def test_physics_rules(project_dir):
    (project_dir / "project.yaml").write_text(PROJECT.replace("  max_deg_per_step: 0.5\n",
                                                              "  max_deg_per_step: 0.5\n  tick_s: 1.0e-5\n"))
    with pytest.raises(InputError) as info:
        load_project(project_dir)
    assert "tick_s is fixed" in str(info.value)
    bad = PROJECT.replace("  resolved: {}                       # written by twin check --write",
                          "  resolved: {ipo_ticks: 1000, operations: {OP10: [{tool: 1, l0_ticks: 300}], OP99: "
                          "[{tool: 1, l0_ticks: 10}]}}")
    (project_dir / "project.yaml").write_text(bad)
    with pytest.raises(InputError) as info:
        load_project(project_dir)
    message = str(info.value)
    assert "not a divisor of ipo_ticks 1000" in message and "OP99" in message
