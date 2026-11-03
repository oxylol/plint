"""/api/health, /api/jobs and the job-backed studies: compare, uncertainty, calibrate."""

from __future__ import annotations

import hashlib

from fastapi import Body, FastAPI, Query
from starlette.responses import Response, StreamingResponse

import cnctwin

from .entry import describe
from .jobs import numba_cache_key
from .jsonio import json_response
from .models import BINARY, SSE, CalibrateRequest, FileList, Health, JobRecord, JobResponse, OptionsBody, errors
from .safety import ApiError, check_id, list_files, resolve_inside
from .series import series_backend
from .sse import job_events
from .views import ApiContext, file_response, job_outcome, public_job


def register(app: FastAPI, ctx: ApiContext) -> None:
    jobs = ctx.jobs

    @app.get("/api/health", tags=["health"], response_model=Health)
    def health():
        """Version, data folder, job counts and limits, and which optional parts this build has."""
        counts = jobs.counts()
        counts.update({"max_heavy": ctx.settings.max_heavy_jobs, "max_light": ctx.settings.max_light_jobs,
                       "heavy_memory_bytes": ctx.settings.heavy_memory_bytes,
                       "light_memory_bytes": ctx.settings.light_memory_bytes})
        return json_response({
            "status": "ok", "version": cnctwin.__version__, "format_version": cnctwin.FORMAT_VERSION,
            "data_dir": str(ctx.settings.data_dir), "memory_mode": jobs.memory_mode or ctx.settings.memory_mode,
            "jobs": counts, "entry_points": {kind: describe(spec) for kind, spec in sorted(ctx.settings.entries.items())},
            "series_reader": series_backend()[0], "db_backend": ctx.db.backend, "numba_cache_key": numba_cache_key(),
            "ui": (ctx.settings.web_dir / "index.html").is_file()})

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs", tags=["jobs"])
    def list_jobs(kind: str | None = None, state: str | None = None, limit: int = Query(200, ge=1, le=5000)):
        """Jobs, newest first."""
        items = [public_job(ctx, j) for j in reversed(jobs.list(kind, state))]
        return json_response({"total": len(items), "jobs": items[:limit]})

    @app.get("/api/jobs/{job_id}", tags=["jobs"], response_model=JobResponse, responses=errors(404))
    def get_job(job_id: str):
        job = _job(ctx, job_id)
        result = jobs.result(job) if job.state == "done" else None
        return json_response({"job": public_job(ctx, job), "result": result})

    @app.post("/api/jobs/{job_id}/cancel", tags=["jobs"], status_code=202, response_model=JobRecord,
              responses=errors(404, 409))
    def cancel_job(job_id: str):
        job = _job(ctx, job_id)
        if job.state != "cancelling":
            job = jobs.cancel(job.id)
        return json_response(public_job(ctx, job), 202)

    @app.get("/api/jobs/{job_id}/events", tags=["jobs"], response_class=StreamingResponse,
             responses={200: {"content": SSE}, **errors(404)})
    async def job_events_stream(job_id: str, timeout_s: float | None = Query(None, gt=0)):
        """SSE: ``progress`` whenever the job record changes, then ``done``."""
        _job(ctx, job_id)
        stream = job_events(jobs, job_id, lambda j: public_job(ctx, j), poll_s=ctx.settings.sse_poll_s,
                            heartbeat_s=ctx.settings.sse_heartbeat_s, timeout_s=timeout_s)
        return StreamingResponse(stream, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/files", tags=["jobs"], response_model=FileList, responses=errors(404))
    def list_job_files(job_id: str):
        """What the job wrote into its output folder."""
        job = _job(ctx, job_id)
        files = list_files(jobs.job_dir(job.id) / "out")
        return json_response({"count": len(files), "files": files})

    @app.get("/api/jobs/{job_id}/files/{path:path}", tags=["jobs"], response_class=Response,
             responses={200: {"content": BINARY}, **errors(400, 403, 404)})
    def get_job_file(job_id: str, path: str):
        job = _job(ctx, job_id)
        return file_response(resolve_inside(jobs.job_dir(job.id) / "out", path))

    @app.get("/api/jobs/{job_id}/log", tags=["jobs"], response_class=Response,
             responses={200: {"content": {"text/plain": {}}}, **errors(404)})
    def get_job_log(job_id: str):
        """The job process's standard output and error (range requests work: tail with bytes=-4096)."""
        job = _job(ctx, job_id)
        path = jobs.job_dir(job.id) / "log.txt"
        if not path.is_file():
            raise ApiError(404, f"job {job_id} has no log (it has not started)")
        return file_response(path)

    # ------------------------------------------------------------------ studies
    @app.get("/api/compare", tags=["studies"], response_model=JobResponse,
             responses={202: {"model": JobResponse}, **errors(404, 409, 500, 501)})
    def compare(a: str = Query(..., description="run id"), b: str = Query(..., description="run id"),
                wait_s: float = Query(20.0, ge=0, le=300), retry: bool = Query(False, description="re-run a failed comparison")):
        """Compare two finished runs (cnctwin.report.compare, as a job). The result is kept: asking
        again for the same pair (unchanged run.json files) returns it at once."""
        run_a, run_b = ctx.runs.get(a), ctx.runs.get(b)
        digests = []
        for run in (run_a, run_b):
            if not run.has("run.json"):
                raise ApiError(409, f"run {run.id} has not finished (no run.json)")
            digests.append(hashlib.sha256((run.path / "run.json").read_bytes()).hexdigest())
        key = hashlib.sha256(f"{a}\0{b}\0{digests[0]}\0{digests[1]}".encode()).hexdigest()
        job = jobs.find_by_key("compare", key)
        if job is None or (retry and job.state in ("failed", "cancelled")):
            request = {"kind": "compare", "run_a": a, "run_b": b, "run_a_dir": str(run_a.path),
                       "run_b_dir": str(run_b.path), "options": {}}
            job = jobs.submit("compare", request, key=key)
        return job_outcome(ctx, job, wait_s)

    @app.post("/api/runs/{run_id}/uncertainty", tags=["studies"], status_code=202, response_model=JobResponse,
              responses=errors(404, 409))
    def uncertainty(run_id: str, body: OptionsBody | None = Body(None)):
        """Start an uncertainty study around a run (cnctwin.uncertainty, as a job; the project's
        ``uncertainty:`` section with ``options`` on top)."""
        run = ctx.runs.get(run_id)
        simulation = jobs.latest_for_run(run_id)
        if simulation is None:
            raise ApiError(409, f"run {run_id} was not started by this server: its project is unknown here")
        request = {"kind": "uncertainty", "run_id": run_id, "run_dir": str(run.path),
                   "project_id": simulation.project_id, "project_path": simulation.request.get("project_path"),
                   "project_dir": simulation.request.get("project_dir"),
                   "options": dict(body.options) if body is not None else {}}
        job = jobs.submit("uncertainty", request, run_id=run_id, project_id=simulation.project_id)
        return json_response({"job": public_job(ctx, job), "result": None}, 202,
                             headers={"Location": f"/api/jobs/{job.id}"})

    @app.post("/api/calibrate", tags=["studies"], status_code=202, response_model=JobResponse,
              responses=errors(400, 404))
    def calibrate(body: CalibrateRequest):
        """Calibrate the machine model from recorded machine data (cnctwin.mdata.calibrate, as a job)."""
        if body.project_id is None and body.run_id is None:
            raise ApiError(400, "give project_id, run_id or both")
        project = ctx.projects.get(body.project_id) if body.project_id else None
        run = ctx.runs.get(body.run_id) if body.run_id else None
        if project is None and run is not None:
            simulation = jobs.latest_for_run(run.id)
            if simulation is not None and simulation.project_id:
                project = ctx.projects.get(simulation.project_id)
        request = {"kind": "calibrate", "project_id": project.id if project else None,
                   "project_path": str(project.file) if project else None,
                   "project_dir": str(project.dir) if project else None,
                   "run_id": run.id if run else None, "run_dir": str(run.path) if run else None,
                   "machine_data": body.machine_data, "options": body.options}
        job = jobs.submit("calibrate", request, run_id=run.id if run else None,
                          project_id=project.id if project else None)
        return json_response({"job": public_job(ctx, job), "result": None}, 202,
                             headers={"Location": f"/api/jobs/{job.id}"})


def _job(ctx: ApiContext, job_id: str):
    check_id(job_id, "job id")
    job = ctx.jobs.get(job_id)
    if job is None:
        raise ApiError(404, f"no job {job_id!r}")
    return job


__all__ = ["register"]
