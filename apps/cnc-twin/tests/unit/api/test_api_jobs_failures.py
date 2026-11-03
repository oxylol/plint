"""Jobs that fail: entry point missing, an exception, a crash, the memory cap, a job that ignores
SIGTERM, a server restart while a job runs, two servers on one DATA_DIR."""

from __future__ import annotations

import json
import os
import signal

import pytest
from api_support import EXAMPLE, fake_entries, make_settings, parse_sse, run_status, wait_for
from fastapi.testclient import TestClient

from cnctwin.api.app import create_app


def start(client, **options):
    response = client.post("/api/runs", json={"project_id": EXAMPLE, "options": options})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_missing_run_project_fails_with_a_clear_status(data_dir):
    settings = make_settings(data_dir, entries=fake_entries(simulate="cnctwin.core.loop_not_written:run_project"))
    with TestClient(create_app(settings)) as client:
        run_id = start(client)
        events = parse_sse(client.get(f"/api/runs/{run_id}/events").text)
        assert events[-1]["event"] == "done" and events[-1]["data"]["status"] == "failed"
        error = events[-1]["data"]["error"]
        assert error["type"] == "EntryPointMissing"
        assert error["message"] == ("cnctwin.core.loop_not_written.run_project is not available in this build "
                                    "(no module 'cnctwin.core.loop_not_written')")
        status = json.loads((data_dir / "runs" / run_id / "status.json").read_text())
        assert status["status"] == "failed" and status["error"]["type"] == "EntryPointMissing"
        record = client.get(f"/api/runs/{run_id}").json()
        assert record["job"]["state"] == "failed" and record["job"]["error"]["type"] == "EntryPointMissing"
        health = client.get("/api/health").json()["entry_points"]["simulate"]
        assert health == {"entry": "cnctwin.core.loop_not_written:run_project", "module_found": False}


def test_missing_function_in_an_existing_module(data_dir):
    settings = make_settings(data_dir, entries=fake_entries(simulate="cnctwin.core.trace:run_project"))
    with TestClient(create_app(settings)) as client:
        run_id = start(client)
        wait_for(lambda: run_status(client, run_id) == "failed")
        message = client.get(f"/api/runs/{run_id}").json()["status"]["error"]["message"]
        assert message == ("cnctwin.core.trace.run_project is not available in this build "
                           "(module 'cnctwin.core.trace' has no callable 'run_project')")


def test_an_exception_fails_the_run_and_resume_continues_from_the_checkpoint(client):
    run_id = start(client, fail_at_s=1.5)
    wait_for(lambda: run_status(client, run_id) == "failed", timeout_s=40)
    record = client.get(f"/api/runs/{run_id}").json()
    error = record["status"]["error"]
    assert error["type"] == "RuntimeError" and "synthetic failure at 1.5" in error["message"]
    assert "Traceback" in error["traceback"] and "fake_entries.py" in error["traceback"]
    assert record["checkpoint"]["tick"] == 1000000
    resumed = client.post(f"/api/runs/{run_id}/resume").json()
    assert resumed["job"]["resume_from"].endswith("ck_1000000.json")
    wait_for(lambda: run_status(client, run_id) == "failed", timeout_s=40)      # the same option fails again
    assert client.get(f"/api/runs/{run_id}").json()["job"]["attempt"] == 2


def test_a_crash_is_reported_with_the_exit_status(client):
    run_id = start(client, crash_at_s=0.5)
    wait_for(lambda: run_status(client, run_id) == "failed", timeout_s=40)
    job = client.get(f"/api/runs/{run_id}").json()["job"]
    assert job["error"]["type"] == "WorkerDied"
    assert job["error"]["message"] == "the job process exited with status 3 without reporting"
    assert client.get(f"/api/runs/{run_id}").json()["status"]["error"]["type"] == "WorkerDied"


def test_the_memory_cap_turns_into_a_memory_error(data_dir):
    settings = make_settings(data_dir, heavy_memory_bytes=400 * 2**20)
    with TestClient(create_app(settings)) as client:
        run_id = start(client, hog_mb=1200)
        wait_for(lambda: run_status(client, run_id) == "failed", timeout_s=40)
        error = client.get(f"/api/runs/{run_id}").json()["status"]["error"]
        assert error["type"] == "MemoryError" and error["message"] == "the job ran out of memory (cap 400 MiB)"


def test_a_job_that_ignores_sigterm_is_killed(data_dir):
    settings = make_settings(data_dir, cancel_grace_s=0.3, kill_grace_s=0.3)
    with TestClient(create_app(settings)) as client:
        run_id = start(client, hang_at_s=0.3)
        wait_for(lambda: client.get(f"/api/runs/{run_id}").json()["status"]["progress"].get("sim_time_s", 0) >= 0.3)
        client.post(f"/api/runs/{run_id}/cancel")
        wait_for(lambda: run_status(client, run_id) == "cancelled", timeout_s=20)
        job = client.get(f"/api/runs/{run_id}").json()["job"]
        assert job["state"] == "cancelled" and job["error"]["message"] == "the job process was killed by SIGKILL"


def test_a_running_job_survives_a_server_restart(data_dir):
    settings = make_settings(data_dir, cancel_on_shutdown=False)
    with TestClient(create_app(settings)) as client:
        run_id = start(client, step_delay_s=0.01)
        wait_for(lambda: run_status(client, run_id) == "running")
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/runs/{run_id}").json()["job"]["state"] in ("running", "done")
        wait_for(lambda: run_status(client, run_id) == "done", timeout_s=40)
        wait_for(lambda: client.get(f"/api/runs/{run_id}").json()["job"]["state"] == "done")


def test_a_job_lost_while_the_server_was_down_can_be_resumed(data_dir):
    settings = make_settings(data_dir, cancel_on_shutdown=False)
    with TestClient(create_app(settings)) as client:
        run_id = start(client, step_delay_s=0.01)
        wait_for(lambda: client.get(f"/api/runs/{run_id}").json()["checkpoint"], timeout_s=40)
        job_id = client.get(f"/api/runs/{run_id}").json()["job"]["id"]
    worker = json.loads((data_dir / "jobs" / job_id / "worker.json").read_text())
    os.kill(worker["pid"], signal.SIGKILL)
    wait_for(lambda: not os.path.exists(f"/proc/{worker['pid']}/cmdline")
             or not open(f"/proc/{worker['pid']}/cmdline", "rb").read())
    with TestClient(create_app(settings)) as client:
        record = client.get(f"/api/runs/{run_id}").json()
        assert record["job"]["state"] == "failed"
        assert record["job"]["error"]["message"] == "the server stopped while the job was running"
        assert record["status"]["status"] == "failed"
        client.post(f"/api/runs/{run_id}/resume")
        wait_for(lambda: run_status(client, run_id) == "done", timeout_s=40)


def test_one_server_per_data_dir(data_dir):
    with TestClient(create_app(make_settings(data_dir))):
        with pytest.raises(RuntimeError, match="another CNC Twin server"):
            with TestClient(create_app(make_settings(data_dir))):
                pass
