"""POST /api/runs with operator overrides: normalised before the run id, the cache key and the job are made, refused
with 422 when malformed; the default simulate entry is the runner that applies them (cnctwin.sweep.runner)."""

from __future__ import annotations

import json

from api_support import EXAMPLE, run_status, wait_for

from cnctwin import cache
from cnctwin.api.settings import DEFAULT_ENTRIES, LOOP_ENTRY


def test_the_default_simulate_entry_applies_operator_overrides():
    assert DEFAULT_ENTRIES["simulate"] == cache.SWEEP_RUNNER == "cnctwin.sweep.runner:run_project"
    # without overrides its runs are the loop's runs (same cache key), with overrides the runner is in the key
    assert cache.canonical_runner(DEFAULT_ENTRIES["simulate"], {}) == cache.DEFAULT_RUNNER == LOOP_ENTRY
    assert cache.canonical_runner(DEFAULT_ENTRIES["simulate"], {"operator": {"feed_override": {"*": 1.2}}}) \
        == cache.SWEEP_RUNNER


def test_operator_options_are_normalised_and_malformed_ones_refused(client, data_dir):
    options = {"operator": {"feed_override": {"*": 1.2, "T1": 1.2}, "rapid_override": 1.0}, "step_delay_s": 0.0}
    response = client.post("/api/runs", json={"project_id": EXAMPLE, "options": options})
    assert response.status_code == 201, response.text
    record = response.json()
    job = json.loads((data_dir / "jobs" / record["job"]["id"] / "job.json").read_text())
    assert job["request"]["options"] == {"operator": {"feed_override": {"*": 1.2}}, "step_delay_s": 0.0}
    wait_for(lambda: run_status(client, record["id"]) == "done", timeout_s=40)
    plain = client.post("/api/runs", json={"project_id": EXAMPLE, "options": {"operator": {"feed_override": 1.0}},
                                           "no_cache": True})
    assert plain.status_code == 201
    job = json.loads((data_dir / "jobs" / plain.json()["job"]["id"] / "job.json").read_text())
    assert job["request"]["options"] == {}                                     # nothing overridden: the plain run
    bad = client.post("/api/runs", json={"project_id": EXAMPLE, "options": {"operator": {"torque_override": 2}}})
    assert bad.status_code == 422 and "torque_override" in bad.text
    wait_for(lambda: run_status(client, plain.json()["id"]) == "done", timeout_s=40)
