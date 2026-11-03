"""FOCAS collector logs and generic CSV logs."""

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.mdata.csvlog import read_csv_log
from cnctwin.mdata.focas import read_focas
from cnctwin.mdata.importers import load_machine_data

from mdatakit import DATA

RPM = 2 * np.pi / 60


def test_focas_csv_records():
    log = read_focas(DATA / "focas_log.csv")
    assert log.source["kind"] == "focas"
    assert log.channels["axis.X.pos_machine_m"].values.tolist() == pytest.approx([-0.381, -0.380512])
    assert log.channels["axis.B.pos_machine_rad"].values[0] == pytest.approx(np.pi / 2)    # unit 2: degrees
    assert log.channels["axis.X.pos_work_m"].values.tolist() == [0.0]
    assert log.channels["spindle.load_frac"].values.tolist() == pytest.approx([0.085, 0.23])  # rdspmeter dec 2, rdspload
    assert log.channels["spindle.speed_rad_s"].values[0] == pytest.approx(9998 * RPM)
    assert log.channels["path.feed_m_s"].values[0] == pytest.approx(0.05)
    assert log.channels["axis.X.following_error_m"].values[0] == pytest.approx(-24e-6)
    assert log.channels["axis.Y.load_frac"].values[0] == pytest.approx(0.12)
    assert log.channels["temp.axis_x_motor_K"].values[0] == pytest.approx(314.15)
    assert log.channels["temp.spindle_motor_K"].values[0] == pytest.approx(325.15)
    assert log.channels["diag.d1000"].values[0] == 7.0
    assert log.channels["program.sequence"].values.tolist() == [120.0, 130.0]
    assert log.channels["program.block_count"].values.tolist() == [57.0]
    assert log.channels["program.name"].values.tolist() == ["O1000"]
    assert log.channels["program.execution"].values.tolist() == ["ACTIVE", "STOPPED"]
    assert log.channels["controller.mode"].values.tolist() == ["AUTOMATIC"]
    assert log.channels["tool.number"].values.tolist() == [12.0]
    # alarm polls: raised when a code appears, cleared when it is gone
    assert [(e.kind, e.code, round(e.t_s, 3)) for e in log.events] == [
        ("alarm", "OT0500", 0.1), ("alarm", "PS0010", 0.2), ("alarm_cleared", "OT0500", 0.3),
        ("alarm_cleared", "PS0010", 0.4)]
    assert log.channels["alarm.count"].values.tolist() == [0.0, 1.0, 2.0, 1.0, 0.0]


def test_focas_json_structures_and_zone():
    log = read_focas(DATA / "focas_log.json")
    assert log.epoch_ns == 1_793_689_200 * 10**9                  # 08:00 +01:00 is 07:00Z
    assert log.channels["axis.X.pos_work_m"].values[0] == pytest.approx(0.0125)
    assert log.channels["axis.Z.pos_work_m"].values[0] == pytest.approx(0.5 * 0.0254)   # dec 4, unit 1: inch
    assert log.channels["spindle2.load_frac"].values[0] == pytest.approx(0.05)
    assert log.channels["axis.X.pos_machine_m"].values[0] == pytest.approx(-0.3685)     # cnc_rdposition machine
    assert [(e.kind, e.code) for e in log.events] == [("alarm", "SV0401"), ("alarm_cleared", "SV0401")]


def test_focas_jsonl_relative_and_mapping_override(tmp_path):
    log = read_focas(DATA / "focas_log.jsonl", mapping={"focas.time_unit": "s"})
    assert log.epoch_ns is None
    assert log.channels["spindle.speed_rad_s"].values.tolist() == pytest.approx([1200 * RPM, 1500 * RPM, 1500 * RPM])
    log = read_focas(DATA / "focas_log.csv", mapping={"cnc_diagnoss:1000": "temp.coolant_K [degC]"})
    assert log.channels["temp.coolant_K"].values[0] == pytest.approx(280.15)
    bad = tmp_path / "bad.csv"
    bad.write_text("time,function,colour\n0,cnc_acts,red\n")
    with pytest.raises(InputError, match="columns"):
        read_focas(bad)
    worse = tmp_path / "worse.csv"
    worse.write_text("time,function,name,value,dec\n0,cnc_srvdelay,X,12,\n")
    with pytest.raises(InputError, match="detection units") as error:
        read_focas(worse, mapping={"focas.time_unit": "s"})
    assert error.value.line == 2                                   # the CSV line of the record


def test_generic_csv_with_mapping():
    mapping = {"time": "Zeit", "spindle.load_frac": "Spindel Last [pct]", "spindle.speed_rad_s": "Drehzahl [rpm]",
               "axis.X.pos_machine_m": "X ist [mm]", "temp.spindle_K": "Temperatur Spindel [degC]",
               "program.block": "Satz", "csv.text.program.block": "true", "csv.decimal": ","}
    log = read_csv_log(DATA / "generic_log.csv", mapping)
    assert log.meta["timezone_assumed"] == "UTC"
    assert log.meta["unmapped_columns"] == ["Programm"]
    load = log.channels["spindle.load_frac"]
    assert load.values[:2].tolist() == pytest.approx([0.125, 0.13]) and np.isnan(load.values[2])
    assert log.channels["axis.X.pos_machine_m"].values[-1] == pytest.approx(-0.379536)
    assert log.channels["spindle.speed_rad_s"].values[0] == pytest.approx(10000 * RPM)
    assert np.isnan(log.channels["temp.spindle_K"].values[-1])
    assert log.channels["program.block"].values[1] == "N20 G01 X16. F3000."
    assert log.channels["program.block"].kind == "event"


def test_generic_csv_numeric_times_and_errors(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("t,P\n0,1.5\n0.5,2.0\n1.0,2.5\n")
    log = load_machine_data({"kind": "csv", "path": "log.csv", "mapping": {"time": "t [ms]", "spindle.power_W": "P [kW]"}},
                            tmp_path)
    assert log.channels["spindle.power_W"].t_s.tolist() == pytest.approx([0.0, 5e-4, 1e-3])
    assert log.channels["spindle.power_W"].values.tolist() == pytest.approx([1500.0, 2000.0, 2500.0])
    epoch = load_machine_data({"kind": "csv", "path": str(path), "mapping": {"time": "t [epoch_s]",
                                                                            "spindle.power_W": "P [kW]"}})
    assert epoch.epoch_ns == 0
    with pytest.raises(InputError, match="no column"):
        read_csv_log(path, {"time": "t", "spindle.power_W": "Power"})
    with pytest.raises(InputError, match="converts to"):
        read_csv_log(path, {"time": "t", "spindle.power_W": "P [mm]"})
    with pytest.raises(InputError, match="time"):
        read_csv_log(path, {"spindle.power_W": "P"})
    with pytest.raises(InputError, match="kind"):
        load_machine_data({"kind": "excel", "path": str(path)})


def test_importer_command_line(tmp_path, capsys):
    import json

    from cnctwin.mdata.importers import main
    from cnctwin.mdata.log import MachineLog

    assert main(["--kind", "focas", "--path", str(DATA / "focas_log.csv"), "--out", str(tmp_path / "log")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["source"]["kind"] == "focas" and summary["events"] == 4
    saved = MachineLog.load(tmp_path / "log")
    assert saved.channels["program.sequence"].values.tolist() == [120.0, 130.0]
    assert main(["--kind", "csv", "--path", str(DATA / "generic_log.csv"), "--map", "spindle.power_W=Nope"]) == 1
    assert "time" in capsys.readouterr().err
