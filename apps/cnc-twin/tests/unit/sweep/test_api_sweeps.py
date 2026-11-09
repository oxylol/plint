"""The API's run cache (POST /api/runs answers a cached run unless no_cache; the worker registers finished runs)
and the sweep endpoints (POST /api/sweeps, GET /api/sweeps, GET /api/sweeps/{id}), with the real job manager and
worker processes and tests/unit/dataset/realrun.py as the simulation."""

from __future__ import annotations

import csv
import io
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT / "tests" / "unit" / "api", ROOT / "tests" / "unit" / "dataset"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from api_support import fake_entries, make_settings, run_status, wait_for  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import realrun  # noqa: E402
from cnctwin import cache as C  # noqa: E402
from cnctwin.api.app import create_app  # noqa: E402
from cnctwin.core.warmup import mill_project  # noqa: E402

RUNNER = f"{ROOT / 'tests' / 'unit' / 'dataset' / 'realrun.py'}:run_project"
PROJECT = "example-mill"


@pytest.fixture(scope="module")
def mill_dir(tmp_path_factory) -> Path:
    return mill_project(tmp_path_factory.mktemp("mill") / "mill").parent


@pytest.fixture
def settings(tmp_path, mill_dir):
    data = tmp_path / "data"
    shutil.copytree(mill_dir, data / "examples" / "mill")
    return make_settings(data, entries=fake_entries(simulate=RUNNER, sweep="cnctwin.sweep:run_sweep_job"),
                         kill_grace_s=5.0)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def job_result(client, job_id: str) -> dict:
    return client.get(f"/api/jobs/{job_id}").json()["result"]


def test_post_runs_answers_the_cached_run(client, settings):
    build = C.build_id()
    first = client.post("/api/runs", json={"project_id": PROJECT, "options": {"outputs": False}})
    assert first.status_code == 201, first.text
    record = first.json()
    assert record["cache"]["hit"] is False and len(record["cache"]["key"]) == 64
    run_id = record["id"]
    wait_for(lambda: run_status(client, run_id) == "done", timeout_s=120)
    job = record["job"]["id"]
    wait_for(lambda: client.get(f"/api/jobs/{job}").json()["job"]["state"] == "done", timeout_s=60)
    registered = job_result(client, job)["cache"]
    if C.build_id() != build:                    # the source tree changed under other work: nothing to compare
        pytest.skip("the build changed during the test")
    assert registered["registered"] is True and registered["key"] == record["cache"]["key"]
    again = client.post("/api/runs", json={"project_id": PROJECT, "options": {"outputs": False}})
    assert again.status_code == 200 and again.json()["id"] == run_id and again.json()["cache"]["hit"] is True
    assert again.headers["location"] == f"/api/runs/{run_id}" and again.json()["run"]["verdict"] is not None
    fresh = client.post("/api/runs", json={"project_id": PROJECT, "options": {"outputs": False}, "no_cache": True})
    assert fresh.status_code == 201 and fresh.json()["id"] != run_id and fresh.json()["cache"]["hit"] is False
    other = client.post("/api/runs", json={"project_id": PROJECT, "options": {"outputs": False, "parts": 2}})
    assert other.status_code == 201 and other.json()["cache"]["key"] != record["cache"]["key"]


def test_a_cached_run_elsewhere_is_copied_into_the_runs(client, settings, tmp_path):
    options = {"outputs": False, "max_sim_s": 9.0}
    project = settings.examples_dir / "mill" / "project.yaml"
    key = C.key_for(project, options, runner=settings.entries["simulate"],
                    environment=C.environment_contract(settings.source_date_epoch))
    store = C.RunCache(settings.data_dir / "cache")
    source = store.runs_dir / "00001-mill-cached"
    realrun.write_run(project, source, options)
    if C.build_id() != key.components["build"]:
        pytest.skip("the build changed during the test")
    store.register(key, source)
    response = client.post("/api/runs", json={"project_id": PROJECT, "options": options})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cache"]["hit"] and body["cache"]["copied_from"] == "00001-mill-cached"
    assert body["id"] != "00001-mill-cached" and (settings.runs_dir / body["id"] / "run.json").is_file()
    assert body["run"]["cache"]["source_run_id"] == "00001-mill-cached" and body["status"]["status"] == "done"
    assert C.run_digest(settings.runs_dir / body["id"])[0] == C.run_digest(source)[0]
    second = client.post("/api/runs", json={"project_id": PROJECT, "options": options})
    assert second.status_code == 200 and second.json()["id"] == body["id"]          # the copy is found directly


def test_a_sweep_runs_as_a_job_with_its_dataset(client, settings):
    spec = {"name": "feeds", "design": {"kind": "grid"},     # the points run the server's simulate entry (realrun)
            "parameters": [{"name": "feed", "kind": "override", "target": "feed", "values": [0.8, 1.2]}]}
    response = client.post("/api/sweeps", json={"project_id": PROJECT, "spec": spec})
    assert response.status_code == 202, response.text
    sweep_id = response.json()["job"]["id"]
    assert response.headers["location"] == f"/api/sweeps/{sweep_id}" and response.json()["job"]["kind"] == "sweep"
    wait_for(lambda: client.get(f"/api/sweeps/{sweep_id}").json()["job"]["state"] in ("done", "failed"), timeout_s=240)
    body = client.get(f"/api/sweeps/{sweep_id}").json()
    assert body["job"]["state"] == "done", body["job"]["error"]
    assert body["name"] == "feeds" and [p["status"] for p in body["points"]] == ["done", "done"]
    assert body["summary"]["done"] == 2 and body["result"]["dataset"]["runs"] == 2
    assert [p["values"] for p in body["points"]] == [{"feed": 0.8}, {"feed": 1.2}]
    table = client.get(body["links"]["dataset"])
    assert table.status_code == 200
    rows = list(csv.DictReader(io.StringIO(table.text)))
    assert [r["param.feed"] for r in rows] == ["0.8", "1.2"]
    assert float(rows[1]["force_max_N"]) > float(rows[0]["force_max_N"]) > 0
    listing = client.get("/api/sweeps").json()
    assert listing["total"] == 1 and listing["sweeps"][0]["id"] == sweep_id
    again = client.post("/api/sweeps", json={"project_id": PROJECT, "spec": spec}).json()["job"]["id"]
    wait_for(lambda: client.get(f"/api/sweeps/{again}").json()["job"]["state"] in ("done", "failed"), timeout_s=240)
    summary = client.get(f"/api/sweeps/{again}").json()["summary"]
    assert summary["cached"] + summary["done"] == 2 and summary["failed"] == 0


def test_sweep_requests_are_checked(client):
    assert client.post("/api/sweeps", json={"project_id": "nothing-here", "spec": {}}).status_code == 404
    bad = client.post("/api/sweeps", json={"project_id": PROJECT, "spec": {"name": "x", "design": {"kind": "lhs"}}})
    assert bad.status_code == 422 and "parameters" in bad.json()["detail"]
    outside = client.post("/api/sweeps", json={"project_id": PROJECT, "spec": {
        "name": "g", "design": {"kind": "grid"}, "generators": {"g": {"script": "../../evil.py"}},
        "parameters": [{"name": "a", "kind": "generator", "generator": "g", "argument": "A", "values": [1]}]}})
    assert outside.status_code in (400, 403, 404)          # a path leaving the project folder is refused
    assert client.post("/api/sweeps", json={"project_id": PROJECT, "spec": {}, "colour": 1}).status_code == 422
    runner = client.post("/api/sweeps", json={"project_id": PROJECT, "spec": {
        "name": "r", "design": {"kind": "grid"}, "execution": {"runner": "os:system"},
        "parameters": [{"name": "feed", "kind": "override", "target": "feed", "values": [1.1]}]}})
    assert runner.status_code == 422 and "runner" in runner.json()["detail"]
    assert client.get("/api/sweeps/j999999").status_code == 404
    run = client.post("/api/runs", json={"project_id": PROJECT, "options": {"step": 1}, "no_cache": True}).json()
    assert client.get(f"/api/sweeps/{run['job']['id']}").status_code == 404


def test_uploaded_projects_may_not_run_generators(client, settings, monkeypatch):
    upload = settings.projects_dir / "my-mill"
    shutil.copytree(settings.examples_dir / "mill", upload)
    (upload / "gen.py").write_text("def main():\n    pass\n")
    spec = {"name": "g", "design": {"kind": "grid"}, "generators": {"g": {"script": "gen.py"}},
            "parameters": [{"name": "a", "kind": "generator", "generator": "g", "argument": "A", "values": [1]}]}
    refused = client.post("/api/sweeps", json={"project_id": "my-mill", "spec": spec})
    assert refused.status_code == 403 and "example projects" in refused.json()["detail"]
    monkeypatch.setenv("CNCTWIN_ALLOW_UPLOAD_GENERATORS", "1")
    accepted = client.post("/api/sweeps", json={"project_id": "my-mill", "spec": spec, "options": {"dataset": False}})
    assert accepted.status_code == 202
    client.post(f"/api/jobs/{accepted.json()['job']['id']}/cancel")
