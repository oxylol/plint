"""/api/projects: the examples and uploaded projects, their files, and the project check."""

from __future__ import annotations

import uuid

import anyio
from fastapi import Body, FastAPI, Query, Request
from starlette.responses import Response

from .jsonio import json_response
from .models import BINARY, JobResponse, OptionsBody, ProjectDetail, ProjectUpload, errors
from .safety import ApiError
from .upload import save_request_body
from .views import ApiContext, file_response, job_outcome

ZIP_BODY = {"requestBody": {"required": True, "description": "The project as a zip: the archive itself, or "
                            "multipart/form-data with the archive in a field named 'file'",
                            "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}},
                                        "multipart/form-data": {"schema": {"type": "object", "properties": {
                                            "file": {"type": "string", "format": "binary"},
                                            "name": {"type": "string"}}, "required": ["file"]}}}}}


def register(app: FastAPI, ctx: ApiContext) -> None:
    store = ctx.projects

    @app.get("/api/projects", tags=["projects"])
    def list_projects():
        """Every project: the examples (always listed, read-only), then the uploads."""
        items = [store.summary(p) for p in store.all()]
        return json_response({"count": len(items), "projects": items})

    @app.post("/api/projects", tags=["projects"], status_code=201, response_model=ProjectUpload,
              openapi_extra=ZIP_BODY, responses=errors(400, 413, 415, 422))
    async def upload_project(request: Request,
                             name: str | None = Query(None, description="id to ask for (made unique)")):
        """Upload a zipped project folder (project.yaml at its top, or in its one top folder)."""
        ctx.settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        archive = ctx.settings.tmp_dir / f"upload-{uuid.uuid4().hex}.zip"
        try:
            saved = await save_request_body(request, archive, ctx.settings.max_upload_bytes)
            desired = name or saved.fields.get("name") or None
            project, report = await anyio.to_thread.run_sync(
                lambda: store.install_upload(archive, desired=desired, filename=saved.filename))
        finally:
            archive.unlink(missing_ok=True)
        detail = store.detail(project)
        detail["skipped"] = report.skipped
        detail["problems"] = store.problems(project)
        return json_response(detail, 201, headers={"Location": f"/api/projects/{project.id}"})

    @app.get("/api/projects/{project_id}", tags=["projects"], response_model=ProjectDetail, responses=errors(404))
    def get_project(project_id: str):
        """project.yaml (as written), its operations and its files."""
        return json_response(store.detail(store.get(project_id)))

    @app.get("/api/projects/{project_id}/files/{path:path}", tags=["projects"], response_class=Response,
             responses={200: {"content": BINARY}, 206: {"description": "Partial content (Range)"},
                        **errors(400, 403, 404)})
    def get_project_file(project_id: str, path: str):
        """A file of the project (programs, CAD, meshes), with HTTP range requests."""
        project = store.get(project_id)
        return file_response(store.file_path(project, path))

    @app.head("/api/projects/{project_id}/files/{path:path}", include_in_schema=False)
    def head_project_file(project_id: str, path: str):
        return get_project_file(project_id, path)

    @app.post("/api/projects/{project_id}/check", tags=["projects"], response_model=JobResponse,
              responses={202: {"model": JobResponse, "description": "still running: poll the job"},
                         **errors(404, 409, 500, 501)})
    def check_project(project_id: str, body: OptionsBody | None = Body(None),
                      wait_s: float = Query(20.0, ge=0, le=300, description="wait this long for the result")):
        """Run the project check (``twin check``) as a job: 200 with its result when it finishes
        within wait_s, else 202 with the job to poll."""
        project = store.get(project_id)
        options = dict(body.options) if body is not None else {}
        if options.get("write") and project.read_only:
            raise ApiError(409, f"{project_id} is an example (read-only): upload a copy to check it with write")
        request = {"kind": "check", "project_id": project.id, "project_path": str(project.file),
                   "project_dir": str(project.dir), "read_only": project.read_only, "options": options}
        job = ctx.jobs.submit("check", request, project_id=project.id)
        return job_outcome(ctx, job, wait_s)
