"""End to end: machine data + simulated run + project -> alignment, calibration, overlay, calibrated project."""

import json
import shutil

import pyarrow.parquet as pq
import pytest
import yaml

from cnctwin.core.errors import InputError
from cnctwin.mdata import calibrate as calibrate_module
from cnctwin.mdata.calibrate import calibrate, main
from cnctwin.mdata.overlay import load_overlay

import mdatakit

OFFSET = -1.234
TRUTH = {"kc_scale": 1.12, "idle_W": 612.0, "eta": 0.85}       # eta = the machine file's value (kept fixed)


@pytest.fixture(scope="module")
def machine_data(synthetic_run, tmp_path_factory):
    folder = tmp_path_factory.mktemp("mdata-measured")
    data = mdatakit.measured_spindle(synthetic_run, offset_s=OFFSET, eta=TRUTH["eta"], idle_W=TRUTH["idle_W"],
                                     kc_scale=TRUTH["kc_scale"])
    path = folder / "agent_dump.xml"
    path.write_bytes(mdatakit.mtconnect_xml(data))
    return path


def _request(project_file, synthetic_run, machine_data, **options):
    return {"project_id": "test", "project_path": str(project_file), "project_dir": str(project_file.parent),
            "run_id": "00001-test", "run_dir": str(synthetic_run),
            "machine_data": {"kind": "mtconnect", "path": str(machine_data)},
            "options": {"targets": ["spindle"], "simulator": "surrogate", **options}}


def test_job_recovers_spindle_values_and_writes_the_overlay(project_file, synthetic_run, machine_data, tmp_path):
    phases = []
    result = calibrate(_request(project_file, synthetic_run, machine_data), tmp_path / "out",
                       lambda progress=None, **fields: phases.append(fields.get("phase")))
    assert {"project", "import", "run", "align", "write"} <= set(phases)
    assert result["alignment"]["method"] == "markers"
    assert result["alignment"]["offset_s"] == pytest.approx(OFFSET, abs=1.5e-3)
    target = result["targets"][0]
    params = {p["name"]: p for p in target["params"]}
    for name, truth in (("kc_scale", TRUTH["kc_scale"]), ("spindle_idle_power_W", TRUTH["idle_W"])):
        lo, hi = params[name]["ci"]
        assert lo < truth < hi, (name, lo, hi)
    assert params["kc_scale"]["value"] == pytest.approx(TRUTH["kc_scale"], rel=0.01)
    assert params["spindle_efficiency"]["written"] is False and params["spindle_efficiency"]["remark"] == "fixed"
    metrics = target["metrics"]["spindle.power_W"]
    # the synthetic shaft power is a per-tick mean of the tooth-passing torque (aliased at 1 ms): sampling it
    # 0.4 ms off (the alignment's error) leaves a residual above the 15 W meter noise
    assert metrics["after"]["rmse"] < 0.8 * metrics["before"]["rmse"]
    assert metrics["after"]["r2"] > 0.9
    out = tmp_path / "out"
    for name in ("calibration.yaml", "calibration.json", "overlay.json", "alignment.json", "machine_log.json",
                 "machine_log.parquet", "project.calibrated.yaml"):
        assert (out / name).is_file(), name
    overlay = load_overlay(out / "calibration.yaml")
    assert overlay["machine"] == "haas-vf2ss" and overlay["material"] == "al6061-t6"
    kc = overlay["set"]["material.cutting.kienzle[tool_class=carbide,coating=any].kc11_N_mm2"]
    assert kc["value"] == pytest.approx(800 * params["kc_scale"]["value"])
    assert overlay["set"]["machine.spindles[main].idle_power_W"]["value"] == pytest.approx(params["spindle_idle_power_W"]["value"])
    series = json.loads((out / "overlay.json").read_text())["series"][0]
    assert series["name"] == "spindle.power_W" and len(series["t_s"]) == len(series["measured"])
    assert series["metrics"]["calibrated"]["rmse"] < series["metrics"]["simulated"]["rmse"]
    # the calibrated project runs through the project loader with the new values
    from cnctwin.project import resolve
    calibrated = resolve(out / "project.calibrated.yaml")
    spindle = calibrated.machine.spindles[0]
    assert float(spindle.idle_power_W) == pytest.approx(params["spindle_idle_power_W"]["value"], rel=1e-9)
    assert spindle.idle_power_W.estimated and "calibrated" in spindle.idle_power_W.note
    assert float(calibrated.material.cutting.kienzle[0].kc11_Pa) == pytest.approx(800e6 * params["kc_scale"]["value"],
                                                                                   rel=1e-9)
    assert float(calibrated.machine.spindles[0].efficiency) == pytest.approx(0.85)


def test_outputs_are_deterministic(project_file, synthetic_run, machine_data, tmp_path):
    for folder in ("a", "b"):
        calibrate(_request(project_file, synthetic_run, machine_data), tmp_path / folder)
    for name in ("calibration.yaml", "calibration.json", "overlay.json", "alignment.json", "machine_log.parquet",
                 "project.calibrated.yaml"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_command_line(project_file, synthetic_run, machine_data, tmp_path, capsys):
    spec = tmp_path / "data.yaml"
    spec.write_text(yaml.safe_dump({"kind": "mtconnect", "path": str(machine_data)}))
    code = main(["--project", str(project_file), "--run", str(synthetic_run), "--data", str(spec), "--target", "spindle",
                 "--simulator", "surrogate", "--out", str(tmp_path / "cli")])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["targets"][0]["status"] == "fitted" and "machine.spindles[main].idle_power_W" in printed["set"]
    code = main(["--project", str(project_file), "--run", str(synthetic_run), "--data", str(spec), "--target", "servo",
                 "--out", str(tmp_path / "cli2")])
    assert code == 1 and "target servo" in capsys.readouterr().err
    with pytest.raises(InputError, match="nothing to calibrate"):
        calibrate({"project_path": str(project_file), "options": {}}, tmp_path / "none")
    with pytest.raises(InputError, match="unknown"):
        calibrate({"project_path": str(project_file), "options": {"targets": ["coolant"]}}, tmp_path / "bad")


def _fake_run_project(nominal_run):
    """A stand-in for cnctwin.core.loop.run_project: the run's cutting power follows the kc1.1 the project
    file carries (as a simulation with those coefficients would, to first order)."""
    calls = []

    def run_project(project_path, run_dir, options, progress_cb):
        document = yaml.safe_load(project_path.read_text())
        kienzle = document["material"]["id"]["cutting"]["kienzle"]
        carbide = next(k for k in kienzle if k["tool_class"] == "carbide")
        value = carbide["kc11_N_mm2"]
        scale = (value["value"] if isinstance(value, dict) else value) / 800.0
        calls.append(scale)
        for name in ("run.json", "scene.json", "frames.bin", "timeline.json"):
            shutil.copy(nominal_run / name, run_dir / name)
        (run_dir / "series" / "tick").mkdir(parents=True)
        shutil.copy(nominal_run / "series" / "index.jsonl", run_dir / "series" / "index.jsonl")
        for part in sorted((nominal_run / "series" / "tick").glob("part-*.parquet")):
            table = pq.read_table(part)
            index = table.column_names.index("cutting.spindle_power_W")
            power = table.column(index).to_numpy() * scale
            import pyarrow as pa
            table = table.set_column(index, "cutting.spindle_power_W", pa.array(power))
            pq.write_table(table, run_dir / "series" / "tick" / part.name)
        progress_cb(sim_time_s=2.32)
    return run_project, calls


def test_loop_simulator_reruns_the_simulation(project_file, synthetic_run, machine_data, tmp_path, monkeypatch):
    run_project, calls = _fake_run_project(synthetic_run)
    monkeypatch.setattr(calibrate_module, "loop_entry", lambda: run_project)
    result = calibrate(_request(project_file, synthetic_run, machine_data, simulator="auto"), tmp_path / "loop")
    target = result["targets"][0]
    assert target["simulator"] == "surrogate + loop"
    params = {p["name"]: p for p in target["params"]}
    assert params["kc_scale"]["value"] == pytest.approx(TRUTH["kc_scale"], rel=0.01)
    assert len(calls) >= 2 and all(0.3 <= c <= 3.0 for c in calls)
    assert not (tmp_path / "loop" / "loop").exists() or not any((tmp_path / "loop" / "loop").rglob("*.parquet"))
    monkeypatch.setattr(calibrate_module, "loop_entry", lambda: None)
    with pytest.raises(InputError, match="not in this build"):
        calibrate(_request(project_file, synthetic_run, machine_data, simulator="loop"), tmp_path / "noloop")


def test_api_entry_point_is_this_function():
    from cnctwin.api.entry import resolve
    from cnctwin.api.settings import DEFAULT_ENTRIES

    assert resolve(DEFAULT_ENTRIES["calibrate"]) is calibrate
