"""/api/sweeps: parameter sweeps as jobs (cnctwin.sweep, docs/ml.md "Sweeps", docs/api.md "Sweeps").

A sweep is a heavy job (one at a time with the simulations) whose output folder is the sweep folder: plan.json,
the points' variants and results, status.json, summary.json and the dataset (``dataset/``), all served by
``/api/jobs/{id}/files``. Its id is the job's id.

What a request may not do: name a runner (the points run the server's own simulation), or use a generator of an
uploaded project (a generator is Python from the project's folder; only the example projects shipped with the
server may, unless CNCTWIN_ALLOW_UPLOAD_GENERATORS=1).
"""

from __future__ import annotations

import os

from fastapi import Body, FastAPI, Query

from .jsonio import json_response
from .models import JobResponse, SweepCreate, SweepList, SweepRecord, errors
from .safety import ApiError, check_id, resolve_inside
from .views import ApiContext, public_job


def _sweep_job(ctx: ApiContext, sweep_id: str):
    check_id(sweep_id, "sweep id")
    job = ctx.jobs.get(sweep_id)
    if job is None or job.kind != "sweep":
        raise ApiError(404, f"no sweep {sweep_id!r}")
    return job


def _links(job_id: str) -> dict:
    return {"self": f"/api/sweeps/{job_id}", "job": f"/api/jobs/{job_id}", "events": f"/api/jobs/{job_id}/events",
            "files": f"/api/jobs/{job_id}/files", "dataset": f"/api/jobs/{job_id}/files/dataset/runs.csv",
            "blocks": f"/api/jobs/{job_id}/files/dataset/blocks.csv", "schema": f"/api/jobs/{job_id}/files/dataset/schema.json"}


def register(app: FastAPI, ctx: ApiContext) -> None:
    jobs = ctx.jobs

    @app.post("/api/sweeps", tags=["studies"], status_code=202, response_model=JobResponse,
              responses=errors(400, 403, 404, 422, 503))
    def create_sweep(body: SweepCreate = Body(...)):
        """Start a parameter sweep of a project (cnctwin.sweep, as a job): the spec's points run one after
        another through the run cache (cached points are not simulated again), then the dataset is exported."""
        from ..core.errors import InputError
        from ..sweep.spec import load_spec
        project = ctx.projects.get(body.project_id)
        _, error = ctx.projects.document(project)
        if error:
            raise ApiError(422, f"project {project.id}: {error}")
        spec_data = dict(body.spec)
        spec_data.pop("base", None)
        execution = spec_data.get("execution") if isinstance(spec_data.get("execution"), dict) else {}
        if execution.get("runner"):
            raise ApiError(422, "execution.runner: the server runs its own simulation (its simulate entry); a runner "
                                "of your choice is for twin sweep")
        try:
            spec = load_spec(spec_data, base_override=str(project.file))
        except InputError as problem:
            raise ApiError(422, str(problem)) from None
        if spec.generators and project.source != "example" and \
                os.environ.get("CNCTWIN_ALLOW_UPLOAD_GENERATORS", "") != "1":
            raise ApiError(403, "generators run Python from the project's folder: only the example projects shipped "
                                "with the server may use them (CNCTWIN_ALLOW_UPLOAD_GENERATORS=1 allows uploaded ones)")
        for generator in spec.generators.values():
            resolve_inside(project.dir, generator.script)
        for parameter in spec.parameters:
            if parameter.kind == "program_template" and parameter.template:
                resolve_inside(project.dir, parameter.template)
        request = {"kind": "sweep", "project_id": project.id, "project_path": str(project.file),
                   "project_dir": str(project.dir), "spec": spec_data, "options": dict(body.options)}
        job = jobs.submit("sweep", request, project_id=project.id)
        return json_response({"job": public_job(ctx, job), "result": None, "links": _links(job.id)}, 202,
                             headers={"Location": f"/api/sweeps/{job.id}"})

    @app.get("/api/sweeps", tags=["studies"], response_model=SweepList)
    def list_sweeps(limit: int = Query(100, ge=1, le=1000)):
        """Sweeps, newest first."""
        items = []
        for job in reversed(jobs.list("sweep")):
            items.append({"id": job.id, "name": (job.request.get("spec") or {}).get("name"), "state": job.state,
                          "project_id": job.project_id, "progress": job.progress, "links": _links(job.id)})
        return json_response({"total": len(items), "sweeps": items[:limit]})

    @app.get("/api/sweeps/{sweep_id}", tags=["studies"], response_model=SweepRecord, responses=errors(404))
    def get_sweep(sweep_id: str):
        """The sweep's job, progress, summary and every point (values, status, run, verdict)."""
        from ..sweep.execute import read_sweep
        job = _sweep_job(ctx, sweep_id)
        folder = jobs.job_dir(job.id) / "out"
        state = read_sweep(folder) if (folder / "sweep.json").is_file() else {}
        return json_response({"id": job.id, "name": state.get("name") or (job.request.get("spec") or {}).get("name"),
                              "project_id": job.project_id, "job": public_job(ctx, job), "status": state.get("status"),
                              "summary": state.get("summary"), "notes": state.get("notes"),
                              "points": state.get("points", []),
                              "result": jobs.result(job) if job.state == "done" else None, "links": _links(job.id)})


__all__ = ["register"]
