"""Helpers of the 3D view's browser tests (tests/unit/view3d): run browser_test.mjs in the Playwright image."""

from __future__ import annotations

import functools
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "mcr.microsoft.com/playwright:v1.48.2-jammy"
NODE_DEPS = Path(os.environ.get("V3D_NODE_DEPS", "/tmp/cnctwin-v3d-node"))


@functools.lru_cache(maxsize=None)
def docker_wrapper() -> tuple[str, ...] | None:
    """How to reach the docker daemon from here: directly, or through `sg docker` (bin/tsafe runs commands
    without the user's supplementary groups, so a member of the docker group needs sg there); None if not."""
    docker = shutil.which("docker")
    if not docker:
        return None
    if subprocess.run([docker, "version", "--format", "{{.Server.Version}}"], capture_output=True, timeout=30,
                      check=False).returncode == 0:
        return ()
    sg = shutil.which("sg")
    if sg and subprocess.run([sg, "docker", "-c", "docker version --format '{{.Server.Version}}'"], capture_output=True,
                             timeout=30, check=False).returncode == 0:
        return (sg, "docker", "-c")
    return None


def docker(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run `docker <args>` with the wrapper docker_wrapper() found."""
    wrapper = docker_wrapper()
    if wrapper is None:
        raise RuntimeError("the docker daemon cannot be reached")
    command = ["docker", *args]
    if wrapper:
        return subprocess.run([*wrapper, shlex.join(command)], **kwargs)
    return subprocess.run(command, **kwargs)


def run_browser_suite(server: dict, suite: str, timeout_s: float = 150) -> dict:
    """Run browser_test.mjs (SUITE) in the Playwright image; returns its results document."""
    out = server["out"]
    args = ["run", "--rm", "--network", "host", "--user", "1002:1002", "--memory", "1g", "--cpus", "1",
            "-e", "HOME=/tmp", "-e", f"V3D_BASE={server['base']}", "-e", f"SUITE={suite}", "-e", "OUT=/out",
            "-v", f"{NODE_DEPS}:/deps:ro", "-v", f"{HERE}:/tests:ro", "-v", f"{out}:/out", IMAGE,
            "node", "/tests/browser_test.mjs"]
    result = docker(args, capture_output=True, text=True, timeout=timeout_s, check=False)
    path = out / f"view3d-{suite}-results.json"
    if not path.exists():
        raise AssertionError(f"the browser test wrote no results (exit {result.returncode}):\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}")
    document = json.loads(path.read_text())
    document["stdout"] = result.stdout
    return document
