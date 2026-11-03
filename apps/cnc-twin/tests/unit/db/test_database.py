import math

import numpy as np
import pytest

from cnctwin.core.errors import InputError, SourceError
from cnctwin.db import Database
from cnctwin.db.check import main as check_main
from cnctwin.db.simap import Curve, PoseMap, Sourced, Table
from dbfixture import MATERIAL, TOOLS, write_db

RPM = 2 * math.pi / 60


@pytest.fixture()
def db(tmp_path):
    return Database.load(write_db(tmp_path))


def test_fixture_database_is_valid(db):
    errors = [p for p in db.validate_all() if p.is_error]
    assert errors == []
    assert db.ids("tool") == ["test-em-10", "test-em-6"]
    assert db.ids("machine") == ["test-vmc", "test-vmc-siemens"]
    assert "src-a" in db.ids("bibliography")


def test_full_machine_through_simap(db):
    m = db.machine("test-vmc")
    assert m.axes.X.travel_m == [-0.5, 0.0]
    assert math.isclose(m.axes.Z.rapid_m_s, 24000e-3 / 60)
    assert math.isclose(m.controller.rapid_m_s.Z, 0.4)
    assert isinstance(m.controller.ipo_period_s, Sourced) and m.controller.ipo_period_s.estimated
    assert math.isclose(float(m.controller.ipo_period_s), 1e-3)
    spindle = m.spindles[0]
    assert math.isclose(spindle.max_rad_s, 12000 * RPM)
    gear = spindle.gear_ranges[0]
    assert gear.rad_s == pytest.approx([0.0, 12000 * RPM])                    # the bare key rpm is rad_s in SI
    assert isinstance(gear.power_W, Curve)
    assert gear.power_W[1] == pytest.approx([2000 * RPM, 22400.0])          # [rpm, kW] by schema path
    assert gear.torque_Nm(7000 * RPM) == pytest.approx(70.0)
    # migrated forward: growth_um_at_max_rpm -> growth_at_max_speed_um; minutes -> seconds
    assert float(spindle.thermal.growth_at_max_speed_m) == pytest.approx(25e-6)
    assert spindle.thermal.time_constant_s == pytest.approx(1800.0)
    # spindle_probe_extra merged into spindle_probe with the est source it was written with
    pretravel = m.probes.spindle_probe.pretravel_m
    assert float(pretravel) == pytest.approx(1e-6) and pretravel.src == "est" and pretravel.note
    nose = m.kinematics.tool_chain[1].fixed
    assert isinstance(nose, PoseMap)
    matrix = nose.matrix()
    assert matrix[:3, 3] == pytest.approx([0.25, 0.2, 0.5])
    assert matrix[:3, :3] @ np.array([1.0, 0, 0]) == pytest.approx([0, 1.0, 0])


def test_full_material_through_simap(db):
    mat = db.material("test-al")
    density = mat.physical.density_kg_m3
    assert isinstance(density, Sourced) and float(density) == 2700 and density.src == "src-a"
    cp = mat.physical.specific_heat_J_kgK
    assert isinstance(cp, Table) and cp(293.15) == pytest.approx(900) and cp.x_unit == "K"
    assert mat.physical.youngs_modulus_Pa(473.15) == pytest.approx(60e9)
    assert mat.physical.youngs_modulus_Pa.src == "src-c"
    assert float(mat.physical.solidus_K) == pytest.approx(855.15)
    kienzle = mat.cutting.kienzle[0]
    assert kienzle.kc11_Pa == pytest.approx(7e8)
    assert kienzle.h_ref_m == pytest.approx(1e-3) and kienzle.b_ref_m == pytest.approx(1e-3)   # migrated default
    assert mat.tool_life[0].T_ref_s == pytest.approx(60.0)                                      # migrated default
    assert mat.tool_life[0].C_m_s == pytest.approx(20.0)
    assert mat.chip.bue.interface_temp_K == pytest.approx([423.15, 623.15])                   # range -> value
    profile = mat.residual_stress[0].profile_Pa                                                 # profile -> profile_MPa
    assert profile[1] == pytest.approx([0.5, 15e6])
    assert mat.flow_stress.T_melt_K == pytest.approx(925.15)


def test_series_items_carry_series_fields(db):
    tool = db.tool("test-em-10")
    assert tool.maker == "Test Tools" and tool.series == "4-flute end mills" and tool.series_id == "test-endmills"
    assert tool.src == "src-a"
    assert tool.geometry.diameter_m == pytest.approx(0.01)
    assert tool.max_rad_s == pytest.approx(40000 * RPM)
    vise = db.workholding("test-vise")
    friction = vise.jaw_friction                      # series-level value -> defaults -> every item
    assert float(friction) == pytest.approx(0.2) and friction.src == "est"
    force = vise.clamping_force_N
    assert float(force) == pytest.approx(28500) and force.at_handle_torque_Nm == 108 and force.src == "src-a"
    assert db.workholding("test-vac").min_abs_pressure_Pa == pytest.approx(15e3)


def test_extends_deep_merge(db):
    ext = db.machine("test-vmc-siemens")
    assert ext.controller.dialect == "siemens840d"
    assert ext.controller.max_feed_m_s == pytest.approx(20000e-3 / 60)       # from the base
    assert ext.axes.X.travel_m == [-0.5, 0.0]
    assert [p.name for p in db.files_of("machine", "test-vmc-siemens")] == ["test-vmc-siemens.yaml", "test-vmc.yaml"]


def test_tool_material_usui_migrated_to_si(db):
    tm = db.tool_material("test-carbide")
    entry = tm.usui[0]
    assert entry.A_per_Pa == pytest.approx(1e-12)
    assert entry.rate_unit == "m/s with sigma_n in Pa, V_s in m/s"
    assert entry.T_range_K == [None, 1150.0]


def test_models_params_si(db):
    params = db.models.params("test-model")
    assert params["gap_m"] == pytest.approx(5e-4)
    assert params["angle_rad"] == pytest.approx(math.pi / 2)
    assert params["label"] == "steel"
    assert params["table_Pa"] == pytest.approx([1e6, 2e6])


def test_estimates_listed_with_file_and_path(db):
    estimates = db.estimates()
    paths = {(e.file.split("/")[-1], e.path) for e in estimates}
    assert ("test-al.yaml", "physical.solidus_degC") in paths
    assert ("test-vmc.yaml", "probes.spindle_probe_extra.pretravel_um") in paths
    solidus = next(e for e in estimates if e.path == "physical.solidus_degC")
    assert solidus.value == 582 and "liquidus" in solidus.note and solidus.line > 0


def test_migrations_recorded(db):
    rules = {m.rule for m in db.migrations()}
    assert {"m-spindle-growth", "m-probe-extra", "mat-profile", "mat-kienzle-ref", "mat-taylor-ref",
            "range-window", "w-series-defaults", "w-min-abs-pressure", "tm-usui-plain-a"} <= rules
    growth = next(m for m in db.migrations() if m.rule == "m-spindle-growth")
    assert growth.path_from == "spindles[0].thermal.growth_um_at_max_rpm"
    assert growth.path_to == "spindles[0].thermal.growth_at_max_speed_um" and growth.line > 0


def test_unknown_key_and_wrong_unit_are_errors(tmp_path):
    bad = MATERIAL.replace("  poisson: 0.33\n", "  poisson: 0.33\n  colour: red\n").replace(
        "  yield_MPa: 276\n", "  yield_psi: 40000\n")
    db = Database.load(write_db(tmp_path, {"materials/test-al.yaml": bad}))
    messages = [p.message for p in db.problems_of("material", "test-al") if p.is_error]
    assert any("unknown key 'colour'" in m for m in messages)
    assert any("declared unit of 'yield' is MPa, not psi" in m for m in messages)
    with pytest.raises(InputError) as info:
        db.material("test-al")
    assert "test-al.yaml:" in str(info.value)


def test_missing_source_is_a_source_error(tmp_path):
    bad = MATERIAL.replace("src: src-a\nphysical:", "physical:").replace("  poisson: 0.33\n",
                                                                       "  poisson: {value: 0.33, src: nowhere}\n")
    db = Database.load(write_db(tmp_path, {"materials/test-al.yaml": bad}))
    with pytest.raises(SourceError) as info:
        db.material("test-al")
    text = str(info.value)
    assert "'nowhere' is not in db/bibliography" in text or "no source" in text
    problem = next(p for p in db.problems_of("material", "test-al") if "nowhere" in p.message)
    assert problem.path == "physical.poisson.src" and problem.line == 13 and problem.kind == "source"


def test_est_without_note_warns_and_is_an_error_in_strict_mode(tmp_path):
    bad = MATERIAL.replace('{value: 582, src: est, note: "from the liquidus less the freezing range"}',
                           "{value: 582, src: est}")
    root = write_db(tmp_path, {"materials/test-al.yaml": bad})
    db = Database.load(root)
    warning = next(p for p in db.problems_of("material", "test-al") if "needs a note" in p.message)
    assert warning.severity == "warning" and warning.path == "physical.solidus_degC.src"
    assert db.material("test-al").physical.solidus_K.estimated
    with pytest.raises(SourceError):
        Database.load(root, strict=True)


def test_bibliography_duplicates(tmp_path):
    db = Database.load(write_db(tmp_path))
    assert "Table 1" in db.bibliography("src-a")["pages"] and "Table 2" in db.bibliography("src-a")["pages"]
    conflict = """\
format: cnctwin-bibliography
version: 1
entries:
  src-a:
    type: article
    authors: "C, D."
    title: "Other"
    year: 2021
"""
    db = Database.load(write_db(tmp_path / "x", {"bibliography/conflict.yaml": conflict}))
    problem = next(p for p in db.validate_all() if p.kind == "duplicate")
    assert "names a different work" in problem.message and problem.is_error
    described = conflict.replace('authors: "C, D."\n    title: "Other"\n    year: 2021',
                                 'authors: "A, B."\n    title: "Cutting data!"\n    year: 2020\n    venue: "J. Other"')
    db = Database.load(write_db(tmp_path / "y", {"bibliography/conflict.yaml": described}))
    problem = next(p for p in db.validate_all() if p.kind == "duplicate")
    assert "described differently" in problem.message and not problem.is_error


def test_references_and_extends_problems(tmp_path):
    bad_tools = TOOLS.replace("id: test-carbide}", "id: no-such-grade}", 1)
    broken_ext = """\
format: cnctwin-machine
version: 1
id: test-loop
extends: test-loop2
maker: x
"""
    loop2 = broken_ext.replace("id: test-loop\nextends: test-loop2", "id: test-loop2\nextends: test-loop")
    db = Database.load(write_db(tmp_path, {"tools/test-endmills.yaml": bad_tools,
                                           "machines/test-loop.yaml": broken_ext,
                                           "machines/test-loop2.yaml": loop2}))
    ref = next(p for p in db.problems_of("tool", "test-em-10") if p.kind == "reference")
    assert "tool_material 'no-such-grade'" in ref.message and ref.path == "items[0].tool_material.id"
    assert not [p for p in db.problems_of("tool", "test-em-6") if p.is_error]
    loop = [p for p in db.problems_of("machine", "test-loop") if p.kind == "extends"]
    assert loop and "circular" in loop[0].message


def test_header_problems(tmp_path):
    wrong_folder = MATERIAL.replace("id: test-al", "id: test-al2")
    db = Database.load(write_db(tmp_path, {"coolants/test-al2.yaml": wrong_folder,
                                           "materials/badname.yaml": MATERIAL}))
    messages = [p.message for p in db.validate_all() if p.is_error]
    assert any("belongs in db/materials/" in m for m in messages)
    assert any("must be the file name" in m for m in messages)
    assert any("also defined in" in m for m in messages)       # test-al twice


def test_check_main_exit_codes(tmp_path, capsys):
    root = write_db(tmp_path)
    assert check_main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "0 errors" in out
    assert check_main(["--root", str(root), "--estimates", "--migrations"]) == 0
    out = capsys.readouterr().out
    assert "physical.solidus_degC = 582" in out and "growth_um_at_max_rpm" in out
    bad = write_db(tmp_path / "bad", {"materials/test-al.yaml": MATERIAL.replace("family: aluminium",
                                                                                "family: unobtainium")})
    assert check_main(["--root", str(bad)]) == 1
    assert "unobtainium" in capsys.readouterr().out
    assert check_main(["--rules"]) == 0


def test_inline_object_validation(db):
    built, merged, problems = db.inline("tool", {"extends": "test-em-10", "geometry": {"diameter_mm": 9.98}},
                                        label="inline")
    assert not [p for p in problems if p.is_error]
    assert built.geometry.diameter_m == pytest.approx(0.00998) and built.geometry.flutes == 4
    assert built.geometry.diameter_m.__class__ is float
    built, merged, problems = db.inline("tool", {"type": "square_endmill"}, label="inline")
    assert built is None and any("missing required key 'hand'" in p.message for p in problems)
