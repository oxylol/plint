"""The API over real HTTP: uvicorn on a free port in 5990-5998, SSE arriving while the job runs,
range requests on the video, an upload, and serve() argument handling."""

from __future__ import annotations

import io
import socket
import threading
import time
import zipfile
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from api_support import DONE_RUN, EXAMPLE, make_settings, synthetic

from cnctwin.api.app import create_app

PORTS = range(5990, 5999)


def free_port() -> int:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    pytest.skip("no free port in 5990-5998")


@contextmanager
def live_server(settings):
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning",
                                           timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the server did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20)


def test_sse_arrives_while_the_run_is_running(data_dir):
    with live_server(make_settings(data_dir)) as base, httpx.Client(base_url=base, timeout=60) as http:
        run_id = http.post("/api/runs", json={"project_id": EXAMPLE, "options": {"step_delay_s": 0.004}}).json()["id"]
        arrivals = []
        with http.stream("GET", f"/api/runs/{run_id}/events") as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            assert stream.headers["x-accel-buffering"] == "no"
            event = None
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                    arrivals.append((event, time.monotonic(), http.get(f"/api/runs/{run_id}").json()["status"]["status"]))
                    if event == "done":
                        break
        kinds = [a[0] for a in arrivals]
        assert kinds[-1] == "done" and kinds.count("finding") == 5 and kinds.count("keyframe") == 5
        first_finding = next(a for a in arrivals if a[0] == "finding")
        assert first_finding[2] == "running"                  # findings stream before the run ends
        assert arrivals[-1][1] - arrivals[0][1] > 0.3


def test_range_requests_over_http(data_dir):
    video = (data_dir / "runs" / DONE_RUN / "video.mp4").read_bytes()
    with live_server(make_settings(data_dir)) as base, httpx.Client(base_url=base, timeout=30) as http:
        url = f"/api/runs/{DONE_RUN}/files/video.mp4"
        head = http.head(url)
        assert head.status_code == 200 and int(head.headers["content-length"]) == len(video)
        chunks = b""
        for start in range(0, len(video), 700):
            part = http.get(url, headers={"Range": f"bytes={start}-{min(start + 699, len(video) - 1)}"})
            assert part.status_code == 206
            chunks += part.content
        assert chunks == video
        etag = head.headers["etag"]
        assert http.get(url, headers={"Range": "bytes=0-9", "If-Range": etag}).status_code == 206
        assert http.get(url, headers={"Range": "bytes=0-9", "If-Range": '"stale"'}).status_code == 200


def test_upload_over_http(data_dir, tmp_path):
    source = tmp_path / "p"
    synthetic.write_synthetic_project(source, seed=9, name="Over HTTP")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(source.iterdir()):
            archive.write(path, path.name)
    with live_server(make_settings(data_dir)) as base, httpx.Client(base_url=base, timeout=30) as http:
        response = http.post("/api/projects", files={"file": ("p.zip", buffer.getvalue(), "application/zip")})
        assert response.status_code == 201 and response.json()["id"] == "over-http"
        assert http.get("/api/projects/over-http/files/op10.nc").text.startswith("%\nO1000")
