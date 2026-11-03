"""Read endpoints of a run (architecture 16/17) against the synthetic run directory."""

from __future__ import annotations

import io
import json
import os
import struct

import numpy as np
import pytest
from api_support import DONE_RUN, PARTIAL_RUN

from cnctwin.core.trace import TraceLog

RUN = f"/api/runs/{DONE_RUN}"


def frame_dtype(header):
    return np.dtype({"names": [f["name"] for f in header["fields"]],
                     "formats": [(f["dtype"], tuple(f["shape"])) if f["shape"] else f["dtype"] for f in header["fields"]],
                     "offsets": [f["offset"] for f in header["fields"]], "itemsize": header["record_bytes"]})


def test_list_and_get(client):
    listing = client.get("/api/runs").json()
    assert [r["id"] for r in listing["runs"]] == [PARTIAL_RUN, DONE_RUN] and listing["total"] == 2
    done = next(r for r in listing["runs"] if r["id"] == DONE_RUN)
    assert done["status"] == "done" and done["verdict"]["safe"] is False and done["name"].startswith("Synthetic")
    assert client.get("/api/runs", params={"status": "running"}).json()["runs"][0]["id"] == PARTIAL_RUN
    assert client.get("/api/runs", params={"limit": 1, "offset": 1}).json()["runs"][0]["id"] == DONE_RUN
    record = client.get(RUN).json()
    assert record["run"]["format"] == "cnctwin-run" and record["status"]["status"] == "done"
    assert record["checkpoint"]["file"] == "checkpoints/ck_2320000.json" and "path" not in record["checkpoint"]
    partial = client.get(f"/api/runs/{PARTIAL_RUN}").json()
    assert partial["run"] is None and partial["status"]["status"] == "running" and partial["job"] is None
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2f..").status_code in (400, 404)


def test_findings_filters(client):
    body = client.get(f"{RUN}/findings").json()
    assert body["total"] == 5 and body["worst"] == "error" and body["safe"] is False
    assert [f["category"] for f in client.get(f"{RUN}/findings", params={"severity": "warning"}).json()["findings"]] \
        == ["chip_packing", "chatter", "out_of_tolerance"]
    only = client.get(f"{RUN}/findings", params={"category": "chatter,burr"}).json()
    assert only["count"] == 2 and only["total"] == 5
    assert client.get(f"{RUN}/findings", params={"op": "OP20"}).json()["count"] == 0
    assert client.get(f"{RUN}/findings", params={"severity": "loud"}).status_code == 400
    assert client.get(f"/api/runs/{PARTIAL_RUN}/findings").json()["count"] == 2


def test_trace_event_and_chain_match_the_trace_log(client, data_dir):
    findings = client.get(f"{RUN}/findings").json()["findings"]
    log = TraceLog.resume(data_dir / "runs" / DONE_RUN / "trace.jsonl",
                          (data_dir / "runs" / DONE_RUN / "trace.jsonl").stat().st_size)
    try:
        for finding in findings:
            root = finding["cause_chain"][0]
            body = client.get(f"{RUN}/trace/{root}").json()
            assert body["event"]["id"] == root and body["event"]["event"].startswith(("finding:", "air_feed"))
            assert [e["id"] for e in body["chain"]] == finding["cause_chain"] == log.chain_ids(root)
        last = len(log._offsets) - 1
        assert client.get(f"{RUN}/trace/{last}").json()["event"]["id"] == last
    finally:
        log.close()
    assert client.get(f"{RUN}/trace/0", params={"max_events": 1}).json()["chain"][0]["id"] == 0
    assert client.get(f"{RUN}/trace/100000").status_code == 404
    assert client.get(f"{RUN}/trace/-1").status_code == 404


def test_trace_ignores_a_line_still_being_written(client, data_dir):
    path = data_dir / "runs" / PARTIAL_RUN / "trace.jsonl"
    count = len(path.read_bytes().splitlines())
    with open(path, "ab") as handle:
        handle.write(b'{"id": %d, "event": "half' % count)
    assert client.get(f"/api/runs/{PARTIAL_RUN}/trace/{count}").status_code == 404
    assert client.get(f"/api/runs/{PARTIAL_RUN}/trace/{count - 1}").json()["event"]["id"] == count - 1


def test_channels_timeline_scene(client):
    channels = client.get(f"{RUN}/channels").json()
    assert channels["count"] == 15 and {c["tier"] for c in channels["channels"]} == {"fast", "tick", "macro"}
    assert client.get(f"{RUN}/channels", params={"tier": "fast"}).json()["count"] == 4
    timeline = client.get(f"{RUN}/timeline").json()
    assert timeline["count"] == 21 and timeline["blocks"][9]["line"] == 11
    window = client.get(f"{RUN}/timeline", params={"t0": 0.7, "t1": 0.8}).json()["blocks"]
    assert [b["line"] for b in window] == [11]
    assert client.get(f"{RUN}/timeline", params={"op": "OP20"}).json()["count"] == 0
    scene = client.get(f"{RUN}/scene").json()
    assert scene["format"] == "cnctwin-scene" and scene["frames"]["record_bytes"] == 52


def test_program(client):
    body = client.get(f"{RUN}/program/OP10").json()
    assert body["source"] == "run.json" and body["lines"] == 23 and body["text"].startswith("%\nO1000")
    assert client.get(f"{RUN}/program/0").json()["op"] == "OP10"
    assert client.get(f"{RUN}/program/OP99").status_code == 404
    assert client.get(f"/api/runs/{PARTIAL_RUN}/program/OP10").status_code == 404


def test_frames_binary_json_window_and_stride(client):
    scene = client.get(f"{RUN}/scene").json()
    dtype = frame_dtype(scene["frames"])
    whole = client.get(f"{RUN}/frames")
    records = np.frombuffer(whole.content, dtype=dtype)
    assert len(records) == 2320 and int(whole.headers["x-frame-count"]) == 2320
    window = client.get(f"{RUN}/frames", params={"t0": 0.5, "t1": 0.6})
    part = np.frombuffer(window.content, dtype=dtype)
    assert part["tick"][0] == 500000 and part["tick"][-1] == 600000 and len(part) == 101
    assert window.headers["x-first-tick"] == "500000" and window.headers["content-type"] == "application/octet-stream"
    strided = np.frombuffer(client.get(f"{RUN}/frames", params={"max_frames": 100}).content, dtype=dtype)
    assert len(strided) <= 100 and np.all(np.diff(strided["tick"]) == 24000)
    body = client.get(f"{RUN}/frames", params={"t0": 1.0, "t1": 1.002, "format": "json"}).json()
    assert body["count"] == 3 and body["fields"]["tick"] == [1000000, 1001000, 1002000]
    assert body["fields"]["tip_S_m"][0][0] == pytest.approx(part["tip_S_m"][0][0].tolist(), abs=0.1)
    first = body["fields"]["axis"][0]
    assert all(len(repr(v)) < 14 for v in first)                 # float32 at its own digits
    assert client.get(f"{RUN}/frames", params={"t0": 5.0}).content == b""


def test_keyframes_glb_fields_bricks_chips_jets(client, data_dir):
    frames = client.get(f"{RUN}/keyframes").json()
    assert frames["count"] == 5 and frames["keyframes"][0]["full"] is True
    glb = client.get(f"{RUN}/keyframes/0.glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF" and glb.headers["content-type"] == "model/gltf-binary"
    assert client.head(f"{RUN}/keyframes/0.glb").headers["content-length"] == str(len(glb.content))
    assert client.get(f"{RUN}/keyframes/9.glb").status_code == 404
    fields = client.get(f"{RUN}/keyframes/3/fields").json()
    assert fields["k"] == 3 and fields["time_s"] == 1.5
    surface = fields["arrays"]["surface_temp_K"]
    assert surface["dtype"] == "float16" and surface["shape"] == [8, 10] and len(surface["data"]) == 8
    assert fields["arrays"]["jet_kind"]["data"] == ["flood"] and fields["arrays"]["wetted"]["dtype"] == "bool"
    only = client.get(f"{RUN}/keyframes/3/fields", params={"names": "part_pose,tick"}).json()["arrays"]
    assert sorted(only) == ["part_pose", "tick"] and only["tick"]["data"] == 1500000
    assert client.get(f"{RUN}/keyframes/3/fields", params={"names": "nope"}).status_code == 404
    npz = client.get(f"{RUN}/keyframes/3/fields", params={"format": "npz"})
    with np.load(io.BytesIO(npz.content)) as archive:
        assert "pile_height_m" in archive.files
    index = json.loads((data_dir / "runs" / DONE_RUN / "keyframes" / "bricks.idx").read_text())["bricks"]
    k, name, offset, length = index["0"][1]                    # brick 0 changed at keyframe 2 (t = 1.0 s)
    brick = client.get(f"{RUN}/bricks/0", params={"t": 1.2})
    assert brick.headers["x-keyframe"] == str(k) == "2" and brick.headers["x-keyframe-time-s"] == "1.0"
    raw = (data_dir / "runs" / DONE_RUN / "keyframes" / name).read_bytes()[offset:offset + length]
    assert brick.content == raw and struct.unpack("<4s", brick.content[:4])[0] == b"BRK1"
    assert client.get(f"{RUN}/bricks/0", params={"t": 0.2}).headers["x-keyframe"] == "0"
    assert client.get(f"{RUN}/bricks/0").headers["x-keyframe"] == str(index["0"][-1][0])
    assert client.get(f"{RUN}/bricks/2", params={"t": 1.7}).headers["x-keyframe"] == "0"
    assert client.get(f"{RUN}/bricks/99").status_code == 404
    assert client.get(f"{RUN}/bricks/0", params={"t": -1}).status_code == 404
    chips = client.get(f"{RUN}/chips", params={"t": 1.2}).json()
    assert chips["k"] == 2 and chips["arrays"]["pos_M_m"]["shape"] == [24, 3]
    assert client.get(f"{RUN}/chips", params={"t": 0.1}).json()["arrays"]["mass_kg"]["shape"] == [0]
    jets = client.get(f"{RUN}/jets", params={"t": 1.2}).json()
    assert jets["k"] == 2 and jets["jets"][0]["nozzle_id"] == "n1" and jets["nozzles"][0]["id"] == "n1"
    assert client.get(f"{RUN}/jets", params={"t": 0.1}).json()["jets"] == []


def test_results(client):
    assert client.get(f"{RUN}/inspection").json()["tolerances"][0]["pass"] is False
    assert client.get(f"{RUN}/surface").json()["features"][0]["id"] == "floor-finish"
    assert client.get(f"{RUN}/tools").json()["tools"][0]["number"] == 1
    assert client.get(f"{RUN}/production").json()["parts"] == 1
    assert client.get(f"{RUN}/sld").json()["tools"] == ["T1"]
    assert client.get(f"{RUN}/sld/T1").json()["tool"] == "T1"
    assert client.get(f"{RUN}/sld/1").json()["tool"] == "T1"
    assert client.get(f"{RUN}/sld/T9").status_code == 404
    for name in ("inspection", "surface", "tools", "production"):
        response = client.get(f"/api/runs/{PARTIAL_RUN}/{name}")
        assert response.status_code == 404 and "yet" in response.json()["detail"]


def test_files_list_range_head_and_types(client, data_dir):
    files = client.get(f"{RUN}/files").json()
    paths = {f["path"] for f in files["files"]}
    assert {"video.mp4", "report.pdf", "scene/stock.glb", "keyframes/attr/brick_0.npz"} <= paths
    video = (data_dir / "runs" / DONE_RUN / "video.mp4").read_bytes()
    full = client.get(f"{RUN}/files/video.mp4")
    assert full.status_code == 200 and full.content == video and full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"] == "video/mp4"
    part = client.get(f"{RUN}/files/video.mp4", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == video[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(video)}"
    tail = client.get(f"{RUN}/files/video.mp4", headers={"Range": "bytes=-50"})
    assert tail.status_code == 206 and tail.content == video[-50:]
    multi = client.get(f"{RUN}/files/video.mp4", headers={"Range": "bytes=0-9,20-29"})
    assert multi.status_code == 206 and multi.headers["content-type"].startswith("multipart/byteranges")
    assert client.get(f"{RUN}/files/video.mp4", headers={"Range": f"bytes={len(video) + 10}-"}).status_code == 416
    head = client.head(f"{RUN}/files/video.mp4")
    assert head.status_code == 200 and head.headers["content-length"] == str(len(video)) and head.content == b""
    report = client.get(f"{RUN}/files/report.html")
    assert report.headers["content-security-policy"] == "sandbox" and report.headers["x-content-type-options"] == "nosniff"
    assert client.get(f"{RUN}/files/scene/stock.glb").headers["content-type"] == "model/gltf-binary"
    assert client.get(f"{RUN}/files/part_OP10.stl").headers["content-type"] == "model/stl"
    assert client.get(f"{RUN}/files/nothing.bin").status_code == 404


@pytest.mark.parametrize("path, status", [
    ("..%2F..%2F..%2Fetc%2Fpasswd", 400), ("%2E%2E/%2E%2E/etc/passwd", 400), ("scene/..%2F..%2Frun.json", 400),
    ("%2Fetc%2Fpasswd", 400), ("a%5C..%5Cb", 400), ("scene//stock.glb", 400), ("run.json%00.txt", 400)])
def test_file_paths_cannot_leave_the_run(client, path, status):
    response = client.get(f"{RUN}/files/{path}")
    assert response.status_code == status, (path, response.text)


def test_a_symbolic_link_out_of_the_run_is_refused(client, data_dir, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    os.symlink(secret, data_dir / "runs" / DONE_RUN / "leak.txt")
    assert client.get(f"{RUN}/files/leak.txt").status_code == 403
    assert "leak.txt" not in {f["path"] for f in client.get(f"{RUN}/files").json()["files"]}
