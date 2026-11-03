import pytest

from cnctwin.core.errors import InputError
from cnctwin.db.yamlio import loads
from cnctwin.project import load_project, new_project, resolve, write_resolved_physics
from cnctwin.project.new import program_skeleton
from conftest import PROJECT

STRIDES = {"ipo_ticks": 1000, "macro_ipo": 10,
           "operations": {"OP10": [{"tool": 1, "gear": 1, "l0_ticks": 20, "max_rpm": 12000, "teeth": 4,
                                    "rule": "tooth"}, {"tool": 2, "l0_ticks": 50}]}}


def test_new_project_is_valid_for_a_machine(tmp_path, fixture_db, capsys):
    target = tmp_path / "new"
    assert new_project([str(target), "--machine", "test-vmc", "--db", str(fixture_db.root)]) == 0
    assert "created" in capsys.readouterr().out
    p = resolve(target, fixture_db)
    assert p.machine.id == "test-vmc" and p.material.id == "test-al" and p.coolant.fluid.id == "test-emulsion"
    assert p.operations[0].program_text == program_skeleton("haas")
    assert p.physics.steps_per_tooth == 96 and p.physics.macro_ipo == 10 and p.physics.tick_s == 1e-6
    assert new_project([str(target), "--machine", "test-vmc", "--db", str(fixture_db.root)]) == 2
    assert new_project([str(tmp_path / "x"), "--machine", "nope", "--db", str(fixture_db.root)]) == 1
    assert "no machine 'nope'" in capsys.readouterr().err
    target2 = tmp_path / "siemens"
    assert new_project([str(target2), "--machine", "test-vmc-siemens", "--db", str(fixture_db.root), "--force"]) == 0
    assert (target2 / "op10.mpf").read_text().endswith("M30\n")


def test_program_skeletons():
    assert "BEGIN PGM OP10 MM" in program_skeleton("heidenhain_klartext")
    assert program_skeleton("heidenhain_iso").startswith("%OP10 G71 *")
    with pytest.raises(InputError):
        program_skeleton("unknown")


def test_write_resolved_replaces_only_that_block(project_dir, fixture_db):
    before = (project_dir / "project.yaml").read_text()
    write_resolved_physics(project_dir, STRIDES)
    after = (project_dir / "project.yaml").read_text()
    data = loads(after)[0]
    assert data["physics"]["resolved"] == {"ipo_ticks": 1000, "macro_ipo": 10, "operations": {
        "OP10": [{"tool": 1, "gear": 1, "l0_ticks": 20, "teeth": 4, "max_rpm": 12000.0, "rule": "tooth"},
                 {"tool": 2, "l0_ticks": 50}]}}
    old = loads(before)[0]
    old["physics"]["resolved"] = data["physics"]["resolved"]
    assert data == old
    kept = [line for line in before.splitlines() if "resolved" not in line]
    assert all(line in after.splitlines() for line in kept)            # every other line as written
    p = resolve(project_dir, fixture_db)
    assert p.physics.resolved.operations.OP10[0].l0_ticks == 20
    write_resolved_physics(project_dir, {"ipo_ticks": 1000, "operations": {"OP10": [{"tool": 1, "l0_ticks": 25}]}})
    assert loads((project_dir / "project.yaml").read_text())[0]["physics"]["resolved"]["operations"]["OP10"] == \
        [{"tool": 1, "l0_ticks": 25}]


def test_write_resolved_adds_the_section_and_keeps_comments(project_dir):
    text = PROJECT.split("physics:")[0] + "recording: {fast: windows}\n# trailing comment\n"
    (project_dir / "project.yaml").write_text(text)
    write_resolved_physics(project_dir, STRIDES)
    after = (project_dir / "project.yaml").read_text()
    assert "# trailing comment" in after and loads(after)[0]["physics"]["resolved"]["ipo_ticks"] == 1000
    text = PROJECT.replace("  resolved: {}                       # written by twin check --write\n", "")
    text = text.replace("  thermal: {every_n_macro: 5, cell_mm: 2.0}\n",
                        "  thermal: {every_n_macro: 5, cell_mm: 2.0}\n  # comment after the last physics key\n")
    (project_dir / "project.yaml").write_text(text)
    write_resolved_physics(project_dir, STRIDES)
    after = (project_dir / "project.yaml").read_text()
    data = loads(after)[0]
    assert data["physics"]["thermal"] == {"every_n_macro": 5, "cell_mm": 2.0}
    assert data["physics"]["resolved"]["operations"]["OP10"][1]["l0_ticks"] == 50
    assert "# comment after the last physics key" in after
    load_project(project_dir)


def test_write_resolved_rejects_bad_strides(project_dir):
    with pytest.raises(InputError) as info:
        write_resolved_physics(project_dir, {"ipo_ticks": 1000, "operations": {"OP10": [{"tool": 1, "l0_ticks": 300}]}})
    assert "not a divisor" in str(info.value)
    with pytest.raises(InputError) as info:
        write_resolved_physics(project_dir, {"ipo_ticks": 1000, "operations": {"OP99": [{"tool": 1, "l0_ticks": 10}]}})
    assert "OP99" in str(info.value)
    with pytest.raises(InputError):
        write_resolved_physics(project_dir, {"ipo_ticks": 1000, "operations": {"OP10": [{"tool": 1, "l0": 10}]}})
    assert loads((project_dir / "project.yaml").read_text())[0]["physics"]["resolved"] == {}
