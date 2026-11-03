"""The run event stream while files change under it: new findings and keyframes arrive once,
half-written files are waited out, the stream ends on a final status, heartbeats and timeouts."""

from __future__ import annotations

import json

from api_support import PARTIAL_RUN

from cnctwin.api.sse import format_event, parse_last_event_id

URL = f"/api/runs/{PARTIAL_RUN}/events"


def next_event(lines) -> dict:
    """The next event from an iter_lines() iterator (comments skipped)."""
    event = {"data": []}
    for line in lines:
        if line.startswith("event: "):
            event["event"] = line[7:]
        elif line.startswith("id: "):
            event["id"] = line[4:]
        elif line.startswith("data: "):
            event["data"].append(line[6:])
        elif line == "" and "event" in event:
            event["data"] = json.loads("\n".join(event["data"]))
            return event
    raise AssertionError("the stream ended")


def test_stream_follows_the_files(client, data_dir):
    run = data_dir / "runs" / PARTIAL_RUN
    findings_path, status_path = run / "findings.json", run / "status.json"
    document = json.loads(findings_path.read_text())
    with client.stream("GET", URL) as stream:
        lines = stream.iter_lines()
        first = [next_event(lines) for _ in range(6)]               # progress, 2 findings, 3 keyframes
        assert [e["event"] for e in first] == ["progress", "finding", "finding", "keyframe", "keyframe", "keyframe"]
        assert first[0]["data"]["status"] == "running" and first[-1]["id"] == "f2k3"
        findings_path.write_text('{"findings": [')                  # a writer caught mid-write
        status_path.write_text("{")
        status_garbage = next_event(lines)
        assert status_garbage["event"] == "progress" and status_garbage["data"]["status"] == "unknown"
        new = dict(document["findings"][-1], id=2, message="a third finding")
        findings_path.write_text(json.dumps({"findings": document["findings"] + [new]}))
        event = next_event(lines)
        assert event["event"] == "finding" and event["data"]["id"] == 2 and event["id"] == "f3k3"
        with open(run / "keyframes" / "index.jsonl", "ab") as handle:
            handle.write(b'{"k": 3, "tick": 1500000, "time_s": 1.5}\n{"k": 4, "ti')   # second line incomplete
        event = next_event(lines)
        assert event["event"] == "keyframe" and event["data"]["k"] == 3
        status_path.write_text(json.dumps({"status": "failed", "progress": {"sim_time_s": 1.2},
                                           "error": {"type": "X", "message": "stopped"}}))
        progress = next_event(lines)
        done = next_event(lines)
        assert progress["data"]["status"] == "failed"
        assert done["event"] == "done" and done["data"] == {"status": "failed", "verdict": None,
                                                            "error": {"type": "X", "message": "stopped"},
                                                            "findings": 3, "keyframes": 4}


def test_heartbeat_and_timeout(client):
    with client.stream("GET", URL, params={"timeout_s": 1.2, "last_event_id": "f2k3"}) as stream:
        text = "".join(stream.iter_text())
    assert ": keep-alive" in text and text.rstrip().endswith(": timeout (reconnect to continue)")
    assert text.count("event: ") == 1                               # only the status: nothing else is new


def test_event_formatting():
    assert format_event("done", {"a": float("nan")}, "f1k2") == 'event: done\nid: f1k2\ndata: {"a":null}\n\n'
    assert parse_last_event_id("f12k3") == (12, 3)
    assert parse_last_event_id("garbage") == (0, 0) and parse_last_event_id(None) == (0, 0)
