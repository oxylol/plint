"""Health, the UI at /, unknown routes, the OpenAPI export, studies (compare, uncertainty,
calibrate), the database browser, jobs, settings, and the job environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from api_support import DONE_RUN, EXAMPLE, PARTIAL_RUN, ROOT, fake_entries, make_settings, wait_for
from fastapi.testclient import TestClient

from cnctwin.api import export_openapi
from cnctwin.api.app import create_app
from cnctwin.api.dbbrowse import DatabaseBrowser
from cnctwin.api.jobs import job_environment, numba_cache_key, systemd_available, worker_command
from cnctwin.api.openapi import render
from cnctwin.api.safety import ApiError
from cnctwin.api.server import parse_args, settings_from_args
from cnctwin.api.settings import Settings, parse_bytes


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["version"] == "1.0.0" and body["memory_mode"] == "rlimit"
    assert set(body["entry_points"]) == {"simulate", "check", "compare", "uncertainty", "calibrate"}
    assert all(e["module_found"] for e in body["entry_points"].values())
    assert body["jobs"]["max_heavy"] == 1 and body["series_reader"].startswith(("pyarrow", "cnctwin.core.recorder"))
    assert len(body["numba_cache_key"]) == 16 and body["ui"] is False


def test_ui_is_served_at_the_root(client, data_dir):
    missing = client.get("/")
    assert missing.status_code == 404 and "web/index.html" in missing.json()["detail"].replace("\\", "/")
    web = data_dir / "web"
    (web / "js").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>CNC Twin</title>")
    (web / "js" / "app.js").write_text("console.log(1)")
    index = client.get("/")
    assert index.status_code == 200 and "CNC Twin" in index.text and index.headers["cache-control"] == "no-cache"
    assert "content-security-policy" not in index.headers
    assert client.get("/js/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/runs/00001-done").text == index.text            # a client-side route
    assert client.get("/js/nothing.js").status_code == 404
    assert client.get("/..%2F..%2Fetc%2Fpasswd").status_code == 400
    assert client.head("/").status_code == 200


def test_unknown_api_routes(client):
    response = client.get("/api/nothing/here")
    assert response.status_code == 404 and response.json()["detail"] == "no API endpoint /api/nothing/here"
    wrong = client.delete("/api/runs")
    assert wrong.status_code == 405 and "GET, POST" in wrong.json()["detail"]
    assert client.put(f"/api/runs/{DONE_RUN}/files/x").status_code == 405


def test_openapi_is_exported_and_up_to_date(tmp_path, client):
    document = export_openapi(tmp_path / "openapi.json")
    assert document["info"]["title"] == "CNC Twin API" and document["openapi"].startswith("3.")
    assert (tmp_path / "openapi.json").read_text() == render(document)
    assert render(export_openapi(False)) == render(document)            # deterministic
    assert (ROOT / "docs" / "openapi.json").read_text() == render(document), \
        "docs/openapi.json is stale: run python -m cnctwin.api.openapi"
    assert client.get("/api/openapi.json").json()["paths"].keys() == document["paths"].keys()
    paths = document["paths"]
    for path in ("/api/health", "/api/projects", "/api/runs", "/api/runs/{run_id}/events", "/api/runs/{run_id}/series",
                 "/api/runs/{run_id}/keyframes/{k}.glb", "/api/runs/{run_id}/bricks/{brick}", "/api/compare",
                 "/api/calibrate", "/api/db/{kind}/{item_id}", "/api/runs/{run_id}/files/{path}"):
        assert path in paths, path
    operation_ids = [op["operationId"] for item in paths.values() for op in item.values()]
    assert len(operation_ids) == len(set(operation_ids))
    assert client.get("/docs").status_code == 404                       # no CDN-backed pages offline


def test_compare_is_a_cached_job(client, data_dir):
    import shutil
    shutil.copytree(data_dir / "runs" / DONE_RUN, data_dir / "runs" / "00003-copy")
    first = client.get("/api/compare", params={"a": DONE_RUN, "b": "00003-copy"})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["result"]["findings_delta"] == 0 and body["result"]["a"]["safe"] is False
    again = client.get("/api/compare", params={"a": DONE_RUN, "b": "00003-copy"}).json()
    assert again["job"]["id"] == body["job"]["id"]
    files = client.get(f"/api/jobs/{body['job']['id']}/files").json()
    assert files["files"] == [{"path": "compare.json", "bytes": files["files"][0]["bytes"]}]
    assert client.get(f"/api/jobs/{body['job']['id']}/files/compare.json").json()["findings_delta"] == 0
    assert client.get(f"/api/jobs/{body['job']['id']}/files/..%2Fjob.json").status_code == 400
    assert client.get("/api/compare", params={"a": DONE_RUN, "b": PARTIAL_RUN}).status_code == 409
    assert client.get("/api/compare", params={"a": DONE_RUN, "b": "nope"}).status_code == 404


def test_compare_entry_missing_is_501(data_dir):
    settings = make_settings(data_dir, entries=fake_entries(compare="cnctwin.report.compare_absent:compare_runs"))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/compare", params={"a": DONE_RUN, "b": DONE_RUN})
        assert response.status_code == 501 and "is not available in this build" in response.json()["detail"]
        retry = client.get("/api/compare", params={"a": DONE_RUN, "b": DONE_RUN, "retry": True})
        assert retry.json()["job"]["id"] != response.json()["job"]["id"]


def test_uncertainty_and_calibrate(client):
    run = client.post("/api/runs", json={"project_id": EXAMPLE}).json()["id"]
    wait_for(lambda: client.get(f"/api/runs/{run}").json()["status"]["status"] == "done", timeout_s=40)
    study = client.post(f"/api/runs/{run}/uncertainty", json={"options": {"runs": 3}})
    assert study.status_code == 202 and study.headers["location"].startswith("/api/jobs/")
    job_id = study.json()["job"]["id"]
    done = wait_for(lambda: (lambda j: j if j["job"]["state"] == "done" else None)(client.get(f"/api/jobs/{job_id}").json()))
    assert done["result"] == {"runs": 3, "base_run": run} and done["job"]["progress"]["done"] == 3
    assert client.post(f"/api/runs/{DONE_RUN}/uncertainty").status_code == 409     # not started here
    calibration = client.post("/api/calibrate", json={"run_id": run, "machine_data": {"kind": "csv", "path": "x.csv"}})
    assert calibration.status_code == 202
    job_id = calibration.json()["job"]["id"]
    done = wait_for(lambda: (lambda j: j if j["job"]["state"] == "done" else None)(client.get(f"/api/jobs/{job_id}").json()))
    assert done["result"]["project_id"] == EXAMPLE and done["result"]["machine_data"]["kind"] == "csv"
    assert client.post("/api/calibrate", json={}).status_code == 400
    assert client.post("/api/calibrate", json={"project_id": "nope"}).status_code == 404
    kinds = {j["kind"] for j in client.get("/api/jobs").json()["jobs"]}
    assert kinds == {"simulate", "uncertainty", "calibrate"}


def test_job_endpoints(client):
    assert client.get("/api/jobs/j999999").status_code == 404
    assert client.get("/api/jobs/bad%20id").status_code == 400
    response = client.post(f"/api/projects/{EXAMPLE}/check")
    job_id = response.json()["job"]["id"]
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409
    log = client.get(f"/api/jobs/{job_id}/log")
    assert log.status_code == 200 and log.headers["content-type"].startswith("text/plain")
    listing = client.get("/api/jobs", params={"kind": "check"}).json()
    assert listing["jobs"][0]["id"] == job_id and listing["jobs"][0]["links"]["log"] == f"/api/jobs/{job_id}/log"


def test_database_browse(client):
    kinds = client.get("/api/db").json()
    assert kinds["backend"] in ("cnctwin.db.database", "files")
    names = {k["kind"] for k in kinds["kinds"]}
    assert {"machine", "material", "bibliography"} <= names
    machines = client.get("/api/db/machines").json()
    assert machines["kind"] == "machine" and "haas-vf2ss" in {m["id"] for m in machines["items"]}
    assert client.get("/api/db/machine").json()["count"] == machines["count"]
    item = client.get("/api/db/materials/al6061-t6").json()
    assert item["data"]["name"] == "Aluminium 6061-T6" and item["file"].endswith("al6061-t6.yaml")
    assert "kaiser-2006-al6061-datasheet" in item["sources"]
    if kinds["backend"] == "cnctwin.db.database" and not any(p["severity"] == "error" for p in item["problems"]):
        assert item["si"]["physical"]["density_kg_m3"] == 2700.0
        assert client.get("/api/db/holders/haimer-a63-000-16").json()["si"]["bore_m"] == pytest.approx(0.016)
    holders = client.get("/api/db/holders").json()["items"]
    assert any(h["id"] == "haimer-a63-000-16" for h in holders)
    bibliography = client.get("/api/db/bibliography").json()
    assert bibliography["count"] > 10
    key = bibliography["items"][0]["id"]
    assert client.get(f"/api/db/bibliography/{key}").json()["entry"]
    assert client.get("/api/db/materials/unobtainium").status_code == 404
    assert client.get("/api/db/spaceships").status_code == 404
    assert client.get("/api/db/..").status_code in (400, 404)


def test_database_files_backend():
    browser = DatabaseBrowser(ROOT / "db")
    browser.backend = "files"
    kinds = {k["kind"]: k["count"] for k in browser.kinds()["kinds"]}
    assert kinds["material"] >= 10 and kinds["bibliography"] > 10
    item = browser.get("materials", "al6061-t6")
    assert item["backend"] == "files" and item["si"] is None and item["data"]["family"] == "aluminium"
    assert "kaiser-2006-al6061-datasheet" in item["sources"]
    holder = browser.get("holder", "haimer-a63-000-16")
    assert holder["series"]["series_id"] == "haimer-weldon" and holder["data"]["maker"] == "HAIMER"
    assert any(i["id"] == "haas-vf2ss" for i in browser.list("machines")["items"])
    with pytest.raises(ApiError):
        browser.get("material", "unobtainium")


def test_settings_from_env_and_args(tmp_path):
    env = {"DATA_DIR": str(tmp_path / "d"), "CNCTWIN_PORT": "5999", "CNCTWIN_JOB_MEMORY": "1.5G",
           "CNCTWIN_MAX_JOBS": "2", "CNCTWIN_ENTRY_SIMULATE": "/x/y.py:run", "CNCTWIN_JOB_MEMORY_MODE": "none"}
    settings = Settings.from_env(env)
    assert settings.data_dir == tmp_path / "d" and settings.port == 5999 and settings.max_heavy_jobs == 2
    assert settings.heavy_memory_bytes == int(1.5 * 2**30) and settings.entries["simulate"] == "/x/y.py:run"
    assert settings.runs_dir == tmp_path / "d" / "runs"
    args = parse_args(["--port", "5991", "--host", "0.0.0.0", "--data-dir", str(tmp_path / "e"), "--job-memory", "900M"])
    from_args = settings_from_args(args, env)
    assert from_args.port == 5991 and from_args.host == "0.0.0.0" and from_args.data_dir == tmp_path / "e"
    assert from_args.heavy_memory_bytes == 900 * 2**20 and from_args.memory_mode == "none"
    assert Settings.from_env({"PORT": "8123"}).port == 8123
    assert Settings.from_env({"PORT": "8123", "CNCTWIN_PORT": "8124"}).port == 8124
    assert parse_bytes("512K") == 512 * 1024 and parse_bytes(10) == 10
    with pytest.raises(ValueError):
        parse_bytes("lots")
    with pytest.raises(ValueError):
        Settings.from_env({"CNCTWIN_JOB_MEMORY_MODE": "cgroups"})
    with pytest.raises(ValueError, match="CNCTWIN_PORT"):
        Settings.from_env({"CNCTWIN_PORT": "http"})


def test_numba_cache_key_is_the_one_bin_tsafe_uses():
    script = ("cd /opt/plint/apps/cnc-twin && (grep -rl --include='*.py' -e '@njit' -e '@numba' -e 'cuda.jit' cnctwin "
              "2>/dev/null | LC_ALL=C sort | xargs -r sha256sum; "
              "python -c 'import numba, llvmlite; print(numba.__version__, llvmlite.__version__)' 2>/dev/null; "
              "grep -m1 'model name' /proc/cpuinfo) | sha256sum | cut -c1-16")
    expected = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()
    assert numba_cache_key() == expected


def test_job_environment_and_commands(tmp_path):
    settings = make_settings(tmp_path)
    env = job_environment(settings, tmp_path / "job")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        assert env[name] == "1"
    assert env["PYTHONHASHSEED"] == "0" and env["SOURCE_DATE_EPOCH"] == "315532800" and env["TZ"] == "UTC"
    assert env["NUMBA_CACHE_DIR"] and env["PYTHONPATH"] == str(ROOT)
    command, popen_env = worker_command(settings, tmp_path / "job", "rlimit", 2**30, env)
    assert command[1:] == ["-m", "cnctwin.api.worker", str(tmp_path / "job")] and popen_env is env
    command, popen_env = worker_command(settings, tmp_path / "job", "systemd", 2**30, env, unit="cnctwin-job-x")
    assert popen_env is None and "--scope" in command and f"MemoryMax={2**30}" in command
    assert "--unit=cnctwin-job-x" in command and "PYTHONHASHSEED=0" in command
    assert command[command.index("/usr/bin/env") + 1] == "-i"


@pytest.mark.skipif(not systemd_available(), reason="needs systemd-run (root or passwordless sudo)")
def test_a_job_in_a_systemd_scope(data_dir):
    settings = make_settings(data_dir, memory_mode="systemd")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["memory_mode"] == "systemd"
        response = client.post(f"/api/projects/{EXAMPLE}/check", params={"wait_s": 30})
        assert response.status_code == 200, response.text
        job = response.json()["job"]
        assert job["memory_mode"] == "systemd" and response.json()["result"]["ok"] is True
        worker = json.loads((data_dir / "jobs" / job["id"] / "worker.json").read_text())
        assert worker["final"] and worker["pid"] != os.getpid()


def test_environment_inherits_numba_cache_dir_from_tsafe(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "cache"))
    assert job_environment(make_settings(tmp_path), tmp_path)["NUMBA_CACHE_DIR"] == str(tmp_path / "cache")
    monkeypatch.delenv("NUMBA_CACHE_DIR")
    expected = str(Path(make_settings(tmp_path).root) / ".numba-cache" / numba_cache_key())
    assert job_environment(make_settings(tmp_path), tmp_path)["NUMBA_CACHE_DIR"] == expected
