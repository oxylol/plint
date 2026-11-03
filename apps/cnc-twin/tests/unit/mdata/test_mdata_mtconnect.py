"""MTConnect importer: streams documents (files) and an agent with sequence handling."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.mdata.importers import load_machine_data
from cnctwin.mdata.mtconnect import AgentClient, parse_streams, read_agent, read_files

from mdatakit import DATA


def test_streams_document_mapping_and_units():
    log = read_files(DATA / "mtconnect_sample.xml")
    assert log.epoch_ns is not None and log.source["kind"] == "mtconnect"
    x = log.channels["axis.X.pos_machine_m"]
    assert x.values[:2].tolist() == pytest.approx([-0.381, -0.379512])
    assert np.isnan(x.values[2])                                   # UNAVAILABLE is a gap
    assert log.channels["axis.X.pos_cmd_m"].values.tolist() == pytest.approx([-0.381, -0.379488])
    assert log.channels["axis.X.load_frac"].values.tolist() == pytest.approx([0.03, 0.11])
    assert log.channels["axis.C.pos_machine_rad"].values[0] == pytest.approx(np.pi / 2)
    speed = log.channels["spindle.speed_rad_s"]
    assert speed.values[-1] == pytest.approx(9998.5 * 2 * np.pi / 60)
    assert log.channels["spindle.speed_cmd_rad_s"].values[0] == pytest.approx(10000 * 2 * np.pi / 60)
    assert log.channels["spindle.power_W"].values.tolist() == [3120.0]
    assert log.channels["temp.spindle_K"].values[0] == pytest.approx(304.65)
    assert log.channels["path.feed_m_s"].values[0] == pytest.approx(0.05)
    assert log.channels["path.z_work_m"].values[0] == pytest.approx(-0.002)
    assert log.channels["program.line"].values.tolist() == [6.0, 11.0]
    assert log.channels["program.block"].values.tolist() == ["S10000 M03", "G01 X16. F3000.", None]
    assert log.channels["program.execution"].values.tolist() == ["READY", "ACTIVE"]
    assert log.channels["tool.number"].values.tolist() == [1.0]
    assert log.channels["path.rapid_override_frac"].values.tolist() == [0.5]
    load = log.channels["spindle.load_frac"]                       # Load + LoadTimeSeries (5 samples at 10 Hz)
    assert load.t_s.tolist() == pytest.approx([0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert load.values.tolist() == pytest.approx([0.045, 0.20, 0.22, 0.25, 0.24, 0.23])
    faults = [e for e in log.events if e.severity == "fault"]
    assert faults[0].code == "PS0010" and faults[0].text == "IMPROPER G-CODE"
    assert any(e.kind == "message" and e.text == "CHECK TOOL" for e in log.events)
    assert "Rotary/RotaryMode" in log.meta["unmapped"]


def test_concatenated_documents_dedupe_and_zones():
    log = read_files(DATA / "mtconnect_two_docs.xml")
    load = log.channels["spindle.load_frac"]
    # sequence 103 appears in both documents once; 10:00:01+01:00 is 09:00:01Z, an hour before the rest
    assert len(load) == 3
    assert sorted(v for v in load.values.tolist() if np.isfinite(v)) == pytest.approx([0.12, 0.15])
    assert log.channels["program.line"].values.tolist() == [5.0]
    assert log.channels["spindle.speed_rad_s"].values.tolist() == pytest.approx([0.0, 6000 * 2 * np.pi / 60])


def test_mapping_override_and_main_spindle():
    log = read_files(DATA / "mtconnect_sample.xml", mapping={"S1temp": "temp.bearing_front_K [degC]",
                                                             "Xload": "axis.X.force_N"})
    assert log.channels["temp.bearing_front_K"].values[0] == pytest.approx(304.65)
    assert log.channels["axis.X.force_N"].values.tolist() == [3.0, 11.0]
    with pytest.raises(InputError):
        read_files(DATA / "mtconnect_sample.xml", mapping={"spindle": "C"})   # C has no velocity


def test_refuses_dtd_and_reports_errors(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><MTConnectStreams/>')
    with pytest.raises(InputError, match="DTD"):
        read_files(bad)
    broken = tmp_path / "broken.xml"
    broken.write_text("<MTConnectStreams><Header")
    with pytest.raises(InputError, match="well-formed"):
        read_files(broken)
    error = tmp_path / "error.xml"
    error.write_text('<MTConnectError><Header instanceId="1"/><Errors><Error errorCode="NO_DEVICE">no device'
                     '</Error></Errors></MTConnectError>')
    with pytest.raises(InputError, match="NO_DEVICE"):
        read_files(error)


# ----------------------------------------------------------------------------- a local agent

class Agent:
    """In-memory agent: a ring buffer of observations with sequence numbers (MTConnect Part 1 semantics)."""

    def __init__(self, count=40, buffer=1000):
        self.instance = "1762160000"
        self.items = []                       # (sequence, t_ms, value)
        for i in range(count):
            self.items.append((i + 1, i * 100, 10.0 + i))
        self.buffer = buffer
        self.drop_before = None               # simulate the buffer overwriting old sequences
        self.requests = []

    def first(self):
        return self.drop_before or self.items[0][0]

    def document(self, start, count):
        first, last = self.first(), self.items[-1][0]
        if start < first:
            return 400, (f'<?xml version="1.0"?><MTConnectError><Header instanceId="{self.instance}" '
                         f'firstSequence="{first}" lastSequence="{last}" nextSequence="{last + 1}"/><Errors>'
                         f'<Error errorCode="OUT_OF_RANGE">from must be at least {first}</Error></Errors>'
                         f'</MTConnectError>')
        chosen = [i for i in self.items if i[0] >= start][:count]
        nxt = chosen[-1][0] + 1 if chosen else start
        rows = "".join(f'<Load dataItemId="S1load" timestamp="2026-11-03T10:00:{t // 1000:02d}.{t % 1000:03d}Z" '
                       f'sequence="{s}">{v}</Load>' for s, t, v in chosen)
        return 200, (f'<?xml version="1.0"?><MTConnectStreams><Header instanceId="{self.instance}" '
                     f'firstSequence="{first}" lastSequence="{last}" nextSequence="{nxt}"/><Streams>'
                     f'<DeviceStream name="M"><ComponentStream component="Rotary" name="S1"><Samples>'
                     f'<RotaryVelocity dataItemId="sp" timestamp="2026-11-03T10:00:00Z" sequence="0">1000'
                     f'</RotaryVelocity>{rows}</Samples></ComponentStream></DeviceStream></Streams>'
                     f'</MTConnectStreams>')


@pytest.fixture()
def agent():
    state = Agent()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            state.requests.append(url.path + ("?" + url.query if url.query else ""))
            if url.path.endswith("/current"):
                status, body = state.document(state.items[-1][0], 1)
            elif url.path.endswith("/sample"):
                status, body = state.document(int(query["from"][0]), int(query["count"][0]))
            else:
                status, body = 404, '<MTConnectError><Errors><Error errorCode="UNSUPPORTED">x</Error></Errors></MTConnectError>'
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/xml")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 5991), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, "http://127.0.0.1:5991/M"
    finally:
        server.shutdown()
        server.server_close()


def test_agent_chains_samples_by_next_sequence(agent):
    state, url = agent
    client = AgentClient(url, count=7)
    observations, gaps, info = client.collect()
    client.close()
    loads = [o for o in observations if o.element == "Load"]
    assert [o.sequence for o in loads] == list(range(1, 41))
    assert gaps == [] and info["requests"] == 1 + 6                 # /current + ceil(40 / 7)
    froms = [r for r in state.requests if "/sample" in r]
    assert froms[0].startswith("/M/sample?from=1&count=7") and "from=36" in froms[-1]


def test_agent_buffer_overrun_is_a_gap(agent):
    state, url = agent
    state.drop_before = 15                                          # sequences 1..14 already overwritten
    log = read_agent(url)
    load = log.channels["spindle.load_frac"]
    assert len(load) == 26 and load.values[0] == pytest.approx(0.24)
    assert log.meta["gaps"] == []                                   # the collection started at firstSequence
    client = AgentClient(url, count=10)
    observations, gaps, _ = client.collect(start=5)
    client.close()
    assert gaps and gaps[0]["from"] == 5 and gaps[0]["to"] == 14
    assert min(o.sequence for o in observations if o.element == "Load") == 15


def test_agent_through_load_machine_data(agent):
    _, url = agent
    log = load_machine_data({"kind": "mtconnect", "path": url, "start": "2026-11-03T10:00:01Z",
                             "end": "2026-11-03T10:00:02Z"})
    load = log.channels["spindle.load_frac"]
    assert len(load) == 11 and load.values[0] == pytest.approx(0.20)
    with pytest.raises(InputError):
        read_agent("http://127.0.0.1:5998/none")                    # nothing listens there


def test_parse_streams_header():
    doc = parse_streams((DATA / "mtconnect_sample.xml").read_bytes())
    assert (doc.header.instance, doc.header.first, doc.header.last, doc.header.next) == ("1762160000", 1001, 1060, 1061)
    assert len(doc.observations) == 37
