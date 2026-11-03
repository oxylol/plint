"""Fixtures of the 3D view tests (tests/unit/view3d): the browser test server (serve_view3d.py: the real API
on synthetic runs, a copy of web/ with the test page) started once per session on port 5996 (else the next
free port in 5990-5999), the Playwright library (npm playwright@1.48.2 in /tmp), and the Playwright image.

Docker is reached directly or, under bin/tsafe (which drops supplementary groups), through `sg docker`. The
browser tests are skipped, saying why, when docker, the image mcr.microsoft.com/playwright:v1.48.2-jammy or the
npm package cannot be had; nothing else is skipped."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from v3d_support import IMAGE, NODE_DEPS, docker, docker_wrapper  # noqa: E402
PORTS = (5996, 5998, 5994, 5993, 5992, 5991, 5990, 5997, 5999)


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _playwright_deps() -> str | None:
    """Path of the Playwright library (installed into NODE_DEPS when missing), or why it is not there."""
    if (NODE_DEPS / "node_modules" / "playwright" / "index.mjs").is_file():
        return None
    npm = shutil.which("npm")
    if not npm:
        return "npm is not installed"
    NODE_DEPS.mkdir(parents=True, exist_ok=True)
    if not (NODE_DEPS / "package.json").exists():
        (NODE_DEPS / "package.json").write_text('{"private": true}\n')
    env = dict(os.environ, PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="1")
    result = subprocess.run([npm, "i", "--no-audit", "--no-fund", "playwright@1.48.2"], cwd=NODE_DEPS, env=env,
                            capture_output=True, text=True, timeout=180, check=False)
    if result.returncode != 0:
        return f"npm i playwright@1.48.2 failed: {result.stderr[-300:]}"
    return None


def _docker_problem() -> str | None:
    if docker_wrapper() is None:
        return "the docker daemon cannot be reached (not installed, or not a member of the docker group)"
    result = docker(["image", "inspect", IMAGE], capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        return f"the image {IMAGE} is not available ({result.stderr.strip()[:200]})"
    return None


@pytest.fixture(scope="session")
def v3d_server(tmp_path_factory):
    """{base, out, data}: the test server, its synthetic runs and a folder for screens and results."""
    problem = _docker_problem() or _playwright_deps()
    if problem:
        pytest.skip(f"browser tests need docker, the Playwright image and playwright@1.48.2: {problem}")
    port = next((p for p in PORTS if _free(p)), None)
    if port is None:
        pytest.skip("no free port in 5990-5999 for the test server")
    work = tmp_path_factory.mktemp("view3d-browser")
    data = work / "data"
    web = work / "web"
    out = work / "out"
    out.mkdir()
    import serve_view3d
    manifest = serve_view3d.build(data)
    log = open(work / "server.log", "w")
    process = subprocess.Popen([sys.executable, str(HERE / "serve_view3d.py"), "--data", str(data), "--web", str(web),
                                "--port", str(port)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while True:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    break
        except OSError:
            pass
        if process.poll() is not None or time.monotonic() > deadline:
            log.close()
            raise RuntimeError(f"the test server did not start: {(work / 'server.log').read_text()[-2000:]}")
        time.sleep(0.25)
    try:
        yield {"base": base, "out": out, "data": data, "manifest": manifest, "port": port}
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        log.close()
