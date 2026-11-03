"""OPC UA: the offline export and a live asyncua server (history read, current read, browse)."""

import asyncio
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.mdata.importers import load_machine_data
from cnctwin.mdata.opcua import (NodeSpec, browse, decode_unece, encode_unece, read_current, read_export, read_history,
                                 write_export)

from mdatakit import DATA

URL = "opc.tcp://127.0.0.1:5992/cnctwin-test/"
T0 = datetime(2026, 11, 3, 9, 0, 0, tzinfo=timezone.utc)


def test_unece_codes():
    assert decode_unece(encode_unece("MMT")) == "MMT"
    assert encode_unece("CEL") == 4408652          # OPC 10000-8 example: degree Celsius
    assert decode_unece(4408652) == "CEL"


def test_export_json_and_csv():
    log = read_export(DATA / "opcua_history.json")
    load = log.channels["spindle.load_frac"]
    assert load.values[:2].tolist() == pytest.approx([0.10, 0.125]) and np.isnan(load.values[2])
    assert load.values[3] == pytest.approx(0.14)
    assert log.meta["bad_status"] == {"ns=2;s=Spindle1.Load": 1}
    assert log.channels["spindle.speed_rad_s"].values[0] == pytest.approx(6000 * 2 * np.pi / 60)
    assert log.channels["program.name"].values.tolist() == ["O1000"]
    log = load_machine_data({"kind": "opcua", "path": "opcua_history.csv",
                             "mapping": {"spindle.load_frac": "ns=2;s=Spindle1.Load [pct]",
                                         "temp.axis_x_K": "ns=2;s=Axis.X.Temp [degC]"}}, DATA)
    assert np.isnan(log.channels["spindle.load_frac"].values[2])
    assert log.channels["temp.axis_x_K"].values[0] == pytest.approx(298.65)
    with pytest.raises(InputError, match="no channel"):
        read_export(DATA / "opcua_history.csv")


@pytest.fixture(scope="module")
def server():
    """An asyncua server with three historized variables (one with EngineeringUnits) on port 5992."""
    from asyncua import Server, ua

    ready = threading.Event()
    stop = threading.Event()
    failure = []

    async def main():
        srv = Server()
        await srv.init()
        srv.set_endpoint(URL)
        ns = await srv.register_namespace("urn:cnctwin:test")
        machine = await srv.nodes.objects.add_object(ua.NodeId("Machine", ns), "Machine")
        load = await machine.add_variable(ua.NodeId("Spindle.Load", ns), "SpindleLoad", 0.0)
        speed = await machine.add_variable(ua.NodeId("Spindle.Speed", ns), "SpindleSpeed", 0.0)
        program = await machine.add_variable(ua.NodeId("Channel.Program", ns), "Program", "")
        eu = ua.EUInformation()
        eu.NamespaceUri = "http://www.opcfoundation.org/UA/units/un/cefact"
        eu.UnitId = encode_unece("RPM")
        eu.DisplayName = ua.LocalizedText("r/min")
        await speed.add_property(ua.NodeId("Spindle.Speed.EU", ns), "EngineeringUnits", eu)
        async with srv:
            await srv.historize_node_data_change([load, speed], period=None, count=0)
            storage = srv.iserver.history_manager.storage
            for i in range(10):
                stamp = T0 + timedelta(milliseconds=100 * i)
                dv = ua.DataValue(ua.Variant(10.0 + i, ua.VariantType.Double), SourceTimestamp=stamp,
                                  ServerTimestamp=stamp)
                await storage.save_node_value(load.nodeid, dv)
                dv = ua.DataValue(ua.Variant(6000.0 + i, ua.VariantType.Double), SourceTimestamp=stamp,
                                  ServerTimestamp=stamp)
                await storage.save_node_value(speed.nodeid, dv)
            await program.write_value("O1000")
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)

    def runner():
        try:
            asyncio.run(main())
        except Exception as error:          # reported to the test thread
            failure.append(error)
            ready.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    assert ready.wait(30), "the OPC UA test server did not start"
    if failure:
        raise failure[0]
    yield URL
    stop.set()
    thread.join(10)


def test_history_read_with_engineering_units(server):
    specs = [NodeSpec("spindle.load_frac", "ns=2;s=Spindle.Load", "pct"),
             NodeSpec("spindle.speed_rad_s", "ns=2;s=Spindle.Speed", None)]
    start = int((T0 - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()) * 10**9
    log = read_history(server, specs, start, start + 450_000_000)
    load = log.channels["spindle.load_frac"]
    assert load.values.tolist() == pytest.approx([0.10, 0.11, 0.12, 0.13, 0.14])
    assert load.t_s.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert log.epoch_ns == start
    speed = log.channels["spindle.speed_rad_s"]
    assert speed.values[0] == pytest.approx(6000 * 2 * np.pi / 60)       # unit from EngineeringUnits (RPM)


def test_current_read_browse_and_export(server, tmp_path):
    current = read_current(server, [NodeSpec("program.name", "ns=2;s=Channel.Program")])
    assert current.channels["program.name"].values.tolist() == ["O1000"]
    found = browse(server, "ns=2;s=Machine")
    ids = {f["node_id"]: f for f in found}
    assert "ns=2;s=Spindle.Speed" in ids and ids["ns=2;s=Spindle.Speed"]["unece"] == "RPM"
    specs = [NodeSpec("spindle.load_frac", "ns=2;s=Spindle.Load", "pct")]
    log = read_history(server, specs)
    path = write_export(tmp_path / "export.json", log, specs, server)
    back = read_export(path)
    np.testing.assert_allclose(back.channels["spindle.load_frac"].values, log.channels["spindle.load_frac"].values)
    assert back.epoch_ns == log.epoch_ns
    with pytest.raises(InputError):
        read_history("opc.tcp://127.0.0.1:5997/none", specs, timeout_s=1.0)
