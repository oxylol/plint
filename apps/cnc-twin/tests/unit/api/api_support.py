"""Helpers of the API tests (imported by conftest.py and the test modules)."""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"
for entry in (str(ROOT), str(FIXTURES)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import make_synthetic_run as synthetic  # noqa: E402,F401

# Starlette 1.x asks for httpx2 for its TestClient; httpx works the same here
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated")

from cnctwin.api.settings import Settings  # noqa: E402

FAKES = Path(__file__).with_name("fake_entries.py")
FAKE_FUNCTIONS = {"simulate": "run_project", "check": "check_project", "compare": "compare_runs",
                  "uncertainty": "run_study", "calibrate": "calibrate"}
DONE_RUN = "00001-done"           # seed 2: finished, one error finding (not safe)
PARTIAL_RUN = "00002-partial"     # seed 3: stopped at 1.2 s, status running, no run.json
EXAMPLE = "example-pocket"        # examples/pocket (seed 4)


def fake_entries(**overrides) -> dict:
    entries = {kind: f"{FAKES}:{name}" for kind, name in FAKE_FUNCTIONS.items()}
    entries.update(overrides)
    return entries


def make_settings(data_dir: Path, **overrides) -> Settings:
    values = dict(data_dir=data_dir, examples_dir=data_dir / "examples", web_dir=data_dir / "web",
                  db_dir=ROOT / "db", memory_mode="rlimit", entries=fake_entries(), poll_s=0.05, sse_poll_s=0.05,
                  sse_heartbeat_s=0.5, status_interval_s=0.05, cancel_grace_s=5.0, kill_grace_s=2.0,
                  cancel_on_shutdown=True)
    values.update(overrides)
    return Settings(**values)


def wait_for(predicate, timeout_s: float = 30.0, interval_s: float = 0.05):
    """Poll until predicate() is truthy; return its value (fail the test on timeout)."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout_s} s waiting for {getattr(predicate, '__name__', predicate)}")
        time.sleep(interval_s)


def parse_sse(text: str) -> list[dict]:
    """Events of an SSE body: [{event, id, data}] (comments and retry lines skipped)."""
    events = []
    for block in text.split("\n\n"):
        event, ident, data = None, None, []
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("id: "):
                ident = line[4:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if event:
            events.append({"event": event, "id": ident, "data": json.loads("\n".join(data)) if data else None})
    return events


def run_status(client, run_id: str) -> str:
    return client.get(f"/api/runs/{run_id}").json()["status"]["status"]
