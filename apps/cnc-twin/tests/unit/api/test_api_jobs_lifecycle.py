"""Simulation jobs end to end with a fake run_project in the real worker subprocess: SSE, queue,
cancel, resume from a checkpoint (bit-identical result set), Last-Event-ID."""

from __future__ import annotations

import json
from collections import Counter

from api_support import EXAMPLE, parse_sse, run_status, synthetic, wait_for


def start(client, **options):
    response = client.post("/api/runs", json={"project_id": EXAMPLE, "options": options})
    assert response.status_code == 201, response.text
    return response.json()


def test_run_streams_progress_findings_keyframes_and_done(client, data_dir):
    from cnctwin.api.runcache import project_hash
    from cnctwin.core.rundir import run_id as make_run_id
    record = start(client, step_delay_s=0.002)
    run_id = record["id"]
    # named as twin run names its runs: the counter above the old-form 00001-done and 00002-partial, the project's
    # slug and the hash of the project and the options
    expected = make_run_id(project_hash(data_dir / "examples" / "pocket" / "project.yaml"), {"step_delay_s": 0.002}, 3,
                           "example-pocket")
    assert run_id == expected and run_id.startswith("00003-example-pocket-") and len(run_id) == len("00003-example-pocket-") + 8
    assert record["status"]["status"] in ("queued", "running")
    assert record["job"]["kind"] == "simulate" and record["links"]["events"] == f"/api/runs/{run_id}/events"
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        body = "".join(stream.iter_text())
    assert body.startswith("retry: 2000\n\n")
    events = parse_sse(body)
    kinds = Counter(e["event"] for e in events)
    assert kinds["finding"] == 5 and kinds["keyframe"] == 5 and kinds["done"] == 1 and kinds["progress"] >= 2
    assert events[-1]["event"] == "done" and events[-1]["data"]["status"] == "done"
    assert events[-1]["data"]["verdict"]["safe"] is False                      # seed 4: an out-of-tolerance error
    findings = [e["data"]["id"] for e in events if e["event"] == "finding"]
    assert findings == list(range(5))
    assert events[-1]["id"] == "f5k5"
    progress = [e["data"]["progress"]["sim_time_s"] for e in events if e["event"] == "progress"
                and e["data"]["status"] == "running"]
    assert progress == sorted(progress) and progress[-1] > 1.0
    record = client.get(f"/api/runs/{run_id}").json()
    assert record["status"]["status"] == "done" and record["job"]["state"] == "done"
    assert record["status"]["progress"]["wall_s"] > 0 and record["status"]["progress"]["eta_s"] == 0.0
    assert record["run"]["verdict"]["safe"] is False and record["job"]["has_result"] is True
    job = client.get(f"/api/jobs/{record['job']['id']}").json()
    assert job["result"]["end_tick"] == 2320000
    assert client.get(f"/api/runs/{run_id}/findings").json()["count"] == 5
    replay = parse_sse(client.get(f"/api/runs/{run_id}/events").text)       # a finished run streams at once
    assert Counter(e["event"] for e in replay) == Counter({"progress": 1, "finding": 5, "keyframe": 5, "done": 1})


def test_verdict_of_a_safe_run(client, data_dir):
    synthetic.write_synthetic_project(data_dir / "examples" / "safe", seed=5)
    run = client.post("/api/runs", json={"project_id": "example-safe", "options": {}}).json()
    done = wait_for(lambda: client.get(f"/api/runs/{run['id']}").json()["run"])
    assert done["verdict"] == {"safe": True, "reasons": []}
    assert client.get(f"/api/runs/{run['id']}/findings").json()["safe"] is True


def test_last_event_id_skips_what_was_seen(client):
    run_id = start(client)["id"]
    wait_for(lambda: run_status(client, run_id) == "done")
    response = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "f2k3"})
    events = parse_sse(response.text)
    assert [e["data"]["id"] for e in events if e["event"] == "finding"] == [2, 3, 4]
    assert [e["data"]["k"] for e in events if e["event"] == "keyframe"] == [3, 4]
    again = parse_sse(client.get(f"/api/runs/{run_id}/events", params={"last_event_id": "f5k5"}).text)
    assert [e["event"] for e in again] == ["progress", "done"] and again[-1]["id"] == "f5k5"


def test_cancel_then_resume_is_bit_identical(client, data_dir, tmp_path):
    run_id = start(client, step_delay_s=0.01)["id"]
    wait_for(lambda: client.get(f"/api/runs/{run_id}").json()["checkpoint"], timeout_s=40)
    cancel = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel.status_code == 202 and cancel.json()["job"]["state"] in ("cancelling", "cancelled")
    wait_for(lambda: run_status(client, run_id) == "cancelled", timeout_s=20)
    record = client.get(f"/api/runs/{run_id}").json()
    assert record["job"]["state"] == "cancelled" and record["job"]["error"]["type"] == "Cancelled"
    assert record["run"] is None and record["checkpoint"]["tick"] == 1000000
    sim_time = record["status"]["progress"]["sim_time_s"]
    assert 1.0 <= sim_time < 2.32
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409
    resumed = client.post(f"/api/runs/{run_id}/resume")
    assert resumed.status_code == 202
    job = resumed.json()["job"]
    assert job["attempt"] == 2 and job["resume_from"].endswith("checkpoints/ck_1000000.json")
    assert client.post(f"/api/runs/{run_id}/resume").status_code == 409            # already queued/running
    wait_for(lambda: run_status(client, run_id) == "done", timeout_s=40)
    status = client.get(f"/api/runs/{run_id}").json()["status"]
    assert status["attempt"] == 2 and status["progress"]["wall_s"] > 0
    assert client.post(f"/api/runs/{run_id}/resume").status_code == 409            # finished
    synthetic.make_synthetic_run(tmp_path / "straight", seed=4, media=False)
    assert synthetic.result_set_digest(data_dir / "runs" / run_id) == synthetic.result_set_digest(tmp_path / "straight")


def test_queue_runs_one_simulation_at_a_time(client):
    first = start(client, step_delay_s=0.005)["id"]
    second = start(client)["id"]
    third = start(client)["id"]
    queued = client.get(f"/api/runs/{third}").json()
    assert queued["status"]["status"] == "queued" and queued["job"]["queue_position"] in (0, 1)
    assert queued["status"]["queue_position"] == queued["job"]["queue_position"]
    assert client.get("/api/jobs", params={"state": "queued"}).json()["total"] >= 1
    cancelled = client.post(f"/api/runs/{third}/cancel").json()
    assert cancelled["status"]["status"] == "cancelled" and cancelled["job"]["state"] == "cancelled"
    assert "before it started" in cancelled["job"]["error"]["message"]
    assert client.get("/api/health").json()["jobs"]["running"] <= 1
    wait_for(lambda: run_status(client, second) == "done", timeout_s=40)
    jobs = {j["run_id"]: j for j in client.get("/api/jobs", params={"kind": "simulate"}).json()["jobs"]}
    assert jobs[first]["finished_s"] <= jobs[second]["started_s"]
    resumed = client.post(f"/api/runs/{third}/resume").json()
    assert resumed["job"]["resume_from"] is None                                   # no checkpoint: from the start
    wait_for(lambda: run_status(client, third) == "done", timeout_s=40)
    assert client.get(f"/api/runs/{third}").json()["run"]["format"] == "cnctwin-run"


def test_run_requests_are_validated(client, data_dir):
    assert client.post("/api/runs", json={"project_id": "nope"}).status_code == 404
    assert client.post("/api/runs", json={"options": {}}).status_code == 422
    assert client.post("/api/runs", json={"project_id": EXAMPLE, "options": [1]}).status_code == 422
    assert client.post("/api/runs", json={"project_id": EXAMPLE, "extra": 1}).status_code == 422
    (data_dir / "examples" / "pocket" / "project.yaml").write_text("format: cnctwin-project\nname: [\n")
    response = client.post("/api/runs", json={"project_id": EXAMPLE})
    assert response.status_code == 422 and "project.yaml" in response.json()["detail"]
    assert client.post("/api/runs/00001-done/cancel").status_code == 409
    assert client.post("/api/runs/00001-done/resume").status_code == 409
    assert not [p for p in (data_dir / "runs").iterdir() if not any(p.iterdir())]


def test_job_events_stream(client):
    run_id = start(client)["id"]
    job_id = client.get(f"/api/runs/{run_id}").json()["job"]["id"]
    events = parse_sse(client.get(f"/api/jobs/{job_id}/events").text)
    assert events[0]["event"] == "progress" and events[-1] == {"event": "done", "id": None,
                                                               "data": {"status": "done", "error": None}}
    log = client.get(f"/api/jobs/{job_id}/log")
    assert log.status_code == 200
    assert json.loads((client.app.state.ctx.jobs.job_dir(job_id) / "job.json").read_text())["state"] == "done"
