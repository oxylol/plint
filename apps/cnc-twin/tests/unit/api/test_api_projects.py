"""Projects: examples, zipped uploads (raw and multipart), files, the check job."""

from __future__ import annotations

import io
import zipfile

from api_support import EXAMPLE, fake_entries, make_settings, synthetic
from fastapi.testclient import TestClient

from cnctwin.api.app import create_app


def project_zip(tmp_path, *, folder="pocket", name="Synthetic pocket, Al 6061-T6", extra=None, seed=4) -> bytes:
    source = tmp_path / "src" / folder
    synthetic.write_synthetic_project(source, seed=seed, name=name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(source.rglob("*")):
            archive.write(path, f"{folder}/{path.relative_to(source)}" if folder else str(path.relative_to(source)))
        for member, data in (extra or {}).items():
            archive.writestr(member, data)
    return buffer.getvalue()


def test_examples_are_listed_read_only(client):
    projects = client.get("/api/projects").json()["projects"]
    assert [p["id"] for p in projects] == [EXAMPLE]
    example = projects[0]
    assert example["source"] == "example" and example["read_only"] and example["machine"] == "haas-vf2ss"
    assert example["operations"] == [{"name": "OP10", "program": "op10.nc", "program_found": True}]
    detail = client.get(f"/api/projects/{EXAMPLE}").json()
    assert detail["project"]["seed"] == 4 and {f["path"] for f in detail["files"]} == {"project.yaml", "op10.nc"}
    program = client.get(f"/api/projects/{EXAMPLE}/files/op10.nc")
    assert program.status_code == 200 and program.headers["content-type"].startswith("text/plain")
    assert client.get(f"/api/projects/{EXAMPLE}/files/op10.nc", headers={"Range": "bytes=0-0"}).content == b"%"
    assert client.get(f"/api/projects/{EXAMPLE}/files/..%2F..%2Fruns").status_code == 400
    assert client.get("/api/projects/example-nope").status_code == 404
    assert client.get("/api/projects/bad%20id").status_code == 400


def test_upload_raw_zip_then_multipart_with_unique_ids(client, tmp_path):
    data = project_zip(tmp_path, extra={"__MACOSX/pocket/._op10.nc": "junk"})
    first = client.post("/api/projects", content=data, headers={"Content-Type": "application/zip"})
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["id"] == "synthetic-pocket-al-6061-t6" and body["source"] == "upload" and not body["read_only"]
    assert body["skipped"] == ["__MACOSX/pocket/._op10.nc"] and body["problems"] == []
    assert first.headers["location"] == f"/api/projects/{body['id']}"
    assert {f["path"] for f in body["files"]} == {"project.yaml", "op10.nc"}
    second = client.post("/api/projects", files={"file": ("pocket.zip", data, "application/zip")},
                         data={"name": "Synthetic pocket, Al 6061-T6"})
    assert second.status_code == 201 and second.json()["id"] == "synthetic-pocket-al-6061-t6-2"
    named = client.post("/api/projects", params={"name": "My Part"}, content=data,
                        headers={"Content-Type": "application/octet-stream"})
    assert named.json()["id"] == "my-part"
    reserved = client.post("/api/projects", params={"name": "example-pocket"}, content=data,
                           headers={"Content-Type": "application/zip"})
    assert reserved.json()["id"] == "p-example-pocket"
    flat = client.post("/api/projects", content=project_zip(tmp_path / "flat", folder="", name="Flat"),
                       headers={"Content-Type": "application/zip"})
    assert flat.status_code == 201 and flat.json()["id"] == "flat"
    ids = [p["id"] for p in client.get("/api/projects").json()["projects"]]
    assert ids == [EXAMPLE, "flat", "my-part", "p-example-pocket", "synthetic-pocket-al-6061-t6",
                   "synthetic-pocket-al-6061-t6-2"]


def test_upload_missing_program_is_reported(client, tmp_path):
    source = tmp_path / "noprog"
    synthetic.write_synthetic_project(source, seed=1, name="No program")
    (source / "op10.nc").unlink()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.write(source / "project.yaml", "project.yaml")
    body = client.post("/api/projects", content=buffer.getvalue(), headers={"Content-Type": "application/zip"}).json()
    assert body["problems"] == ["operation OP10: program 'op10.nc' is not in the project"]
    assert body["operations"][0]["program_found"] is False


def test_bad_uploads(client, tmp_path):
    def post(content, content_type="application/zip"):
        return client.post("/api/projects", content=content, headers={"Content-Type": content_type})
    assert post(b"").status_code == 400
    assert post(b"not a zip").status_code == 400
    assert post(project_zip(tmp_path), "text/plain").status_code == 415
    slip = io.BytesIO()
    with zipfile.ZipFile(slip, "w") as archive:
        archive.writestr("project.yaml", "format: cnctwin-project\n")
        archive.writestr("../../escape.txt", "x")
    response = post(slip.getvalue())
    assert response.status_code == 400 and "leaves the archive" in response.json()["detail"]
    assert not (tmp_path / "escape.txt").exists()
    none = io.BytesIO()
    with zipfile.ZipFile(none, "w") as archive:
        archive.writestr("readme.txt", "no project here")
        archive.writestr("deep/other.yaml", "a: 1\n")
    response = post(none.getvalue())
    assert response.status_code == 422 and "deep/other.yaml" in response.json()["detail"]
    wrong = io.BytesIO()
    with zipfile.ZipFile(wrong, "w") as archive:
        archive.writestr("project.yaml", "format: cnctwin-machine\nversion: 1\n")
    assert post(wrong.getvalue()).status_code == 422
    broken = io.BytesIO()
    with zipfile.ZipFile(broken, "w") as archive:
        archive.writestr("project.yaml", "format: cnctwin-project\nname: [unclosed\n")
    response = post(broken.getvalue())
    assert response.status_code == 422 and "project.yaml:3" in response.json()["detail"]
    assert client.post("/api/projects", files={"other": ("x.txt", b"hello", "text/plain")},
                       data={"name": "x"}).status_code in (400, 422)
    assert [p["id"] for p in client.get("/api/projects").json()["projects"]] == [EXAMPLE]
    assert not list((client.app.state.ctx.settings.tmp_dir).iterdir())


def test_upload_size_limit(data_dir, tmp_path):
    settings = make_settings(data_dir, max_upload_bytes=2000)
    with TestClient(create_app(settings)) as small:
        response = small.post("/api/projects", content=b"x" * 5000, headers={"Content-Type": "application/zip"})
        assert response.status_code == 413
        response = small.post("/api/projects", files={"file": ("p.zip", b"x" * 100000, "application/zip")})
        assert response.status_code == 413


def test_check_job(client):
    response = client.post(f"/api/projects/{EXAMPLE}/check", json={"options": {"verbose": True}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"]["ok"] is True and body["result"]["read_only"] is True
    assert body["result"]["options"] == {"verbose": True} and body["job"]["kind"] == "check"
    assert body["job"]["state"] == "done" and body["job"]["pool"] == "light"
    refused = client.post(f"/api/projects/{EXAMPLE}/check", json={"options": {"write": True}})
    assert refused.status_code == 409
    failed = client.post(f"/api/projects/{EXAMPLE}/check", json={"options": {"fail": True}})
    assert failed.status_code == 500 and "synthetic check failure" in failed.json()["detail"]
    assert failed.json()["job"]["error"]["type"] == "ValueError"
    assert client.post("/api/projects/nope/check").status_code == 404
    later = client.post(f"/api/projects/{EXAMPLE}/check", params={"wait_s": 0})
    assert later.status_code in (200, 202)
    job_id = later.json()["job"]["id"]
    done = client.get(f"/api/jobs/{job_id}").json()
    assert done["job"]["id"] == job_id


def test_job_result_json_has_a_header_and_a_bare_older_one_still_reads(client, data_dir):
    import json
    body = client.post(f"/api/projects/{EXAMPLE}/check", json={"options": {}}).json()
    job_id = body["job"]["id"]
    path = data_dir / "jobs" / job_id / "result.json"
    document = json.loads(path.read_text())
    assert document["format"] == "cnctwin-job-result" and document["version"] == 1
    assert document["job_id"] == job_id and document["kind"] == "check" and document["result"] == body["result"]
    path.write_text(json.dumps(body["result"]))                    # as an older worker wrote it: the bare value
    assert client.get(f"/api/jobs/{job_id}").json()["result"] == body["result"]


def test_check_entry_point_missing(data_dir):
    settings = make_settings(data_dir, entries=fake_entries(check="cnctwin.project.nothing_here:check_project"))
    with TestClient(create_app(settings)) as client:
        response = client.post(f"/api/projects/{EXAMPLE}/check")
        assert response.status_code == 501
        assert "cnctwin.project.nothing_here.check_project is not available" in response.json()["detail"]
