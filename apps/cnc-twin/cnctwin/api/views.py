"""Shared pieces of the route handlers: the context, job and run records, file responses."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from starlette.responses import FileResponse

from .dbbrowse import DatabaseBrowser
from .jobs import ACTIVE, JobManager
from .jsonio import FileCache, json_response
from .projects import ProjectStore
from .runfiles import RunDir, RunStore
from .settings import Settings

MEDIA_TYPES = {
    ".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".stl": "model/stl", ".3mf": "model/3mf",
    ".obj": "model/obj", ".step": "model/step", ".stp": "model/step", ".npz": "application/octet-stream",
    ".npy": "application/octet-stream", ".parquet": "application/vnd.apache.parquet",
    ".jsonl": "application/x-ndjson", ".json": "application/json", ".wav": "audio/wav", ".mp4": "video/mp4",
    ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".pdf": "application/pdf",
    ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8", ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8", ".yaml": "application/yaml",
    ".yml": "application/yaml", ".md": "text/markdown; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".wasm": "application/wasm",
    ".bin": "application/octet-stream", ".idx": "application/octet-stream", ".zip": "application/zip",
}
# CNC program extensions of the supported dialects: shown as text
TEXT_PROGRAMS = (".nc", ".ngc", ".tap", ".cnc", ".eia", ".min", ".mpf", ".spf", ".h", ".i", ".iso", ".ptp", ".gcode")
ACTIVE_CONTENT = ("text/html", "image/svg+xml")


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in MEDIA_TYPES:
        return MEDIA_TYPES[suffix]
    if suffix in TEXT_PROGRAMS:
        return "text/plain; charset=utf-8"
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed or "application/octet-stream"


def file_response(path: Path, *, download_name: str | None = None, sandbox: bool = True) -> FileResponse:
    """A file with range support (Starlette FileResponse: single and multiple ranges, If-Range, 416).
    Files from runs, projects and jobs are data: HTML and SVG get ``Content-Security-Policy:
    sandbox`` (no scripts, no same-origin access), everything ``X-Content-Type-Options: nosniff``."""
    kind = media_type(path)
    headers = {"X-Content-Type-Options": "nosniff"}
    if sandbox and kind.split(";")[0] in ACTIVE_CONTENT:
        headers["Content-Security-Policy"] = "sandbox"
    return FileResponse(path, media_type=kind, headers=headers, filename=download_name,
                        content_disposition_type="attachment" if download_name else "inline")


@dataclass
class ApiContext:
    settings: Settings
    cache: FileCache
    projects: ProjectStore
    runs: RunStore
    jobs: JobManager
    db: DatabaseBrowser

    @classmethod
    def build(cls, settings: Settings) -> "ApiContext":
        cache = FileCache()
        return cls(settings, cache, ProjectStore(settings), RunStore(settings.runs_dir, cache), JobManager(settings),
                   DatabaseBrowser(settings.db_dir))


def public_job(ctx: ApiContext, job) -> dict:
    links = {"self": f"/api/jobs/{job.id}", "events": f"/api/jobs/{job.id}/events", "files": f"/api/jobs/{job.id}/files",
             "log": f"/api/jobs/{job.id}/log"}
    if job.kind == "simulate" and job.run_id:
        links["run"] = f"/api/runs/{job.run_id}"
    return {"id": job.id, "kind": job.kind, "state": job.state, "pool": job.pool, "run_id": job.run_id,
            "project_id": job.project_id, "queue_position": ctx.jobs.queue_position(job.id), "attempt": job.attempt,
            "created_s": job.created_s, "started_s": job.started_s, "finished_s": job.finished_s,
            "progress": job.progress, "error": job.error, "has_result": job.has_result,
            "memory_mode": job.memory_mode, "memory_bytes": job.memory_bytes,
            "resume_from": job.request.get("resume_from"), "links": links}


def public_checkpoint(checkpoint: dict | None) -> dict | None:
    if checkpoint is None:
        return None
    return {k: v for k, v in checkpoint.items() if k != "path"}


def run_status(ctx: ApiContext, run: RunDir) -> dict:
    status = dict(run.status())
    job = ctx.jobs.latest_for_run(run.id)
    if job is not None and job.state == "queued":
        status["queue_position"] = ctx.jobs.queue_position(job.id)
    return status


def run_record(ctx: ApiContext, run: RunDir) -> dict:
    job = ctx.jobs.latest_for_run(run.id)
    base = f"/api/runs/{run.id}"
    links = {name: f"{base}/{name}" for name in ("events", "findings", "channels", "series", "timeline", "scene",
                                                 "frames", "keyframes", "files", "inspection", "surface", "tools",
                                                 "production", "export")}
    links["self"] = base
    return {"id": run.id, "run": run.run_json(), "status": run_status(ctx, run),
            "job": public_job(ctx, job) if job is not None else None,
            "checkpoint": public_checkpoint(run.latest_checkpoint()), "links": links}


def run_summary(ctx: ApiContext, run: RunDir) -> dict:
    status = run.status()
    document = run.run_json() or {}
    job = ctx.jobs.latest_for_run(run.id)
    project = document.get("project") if isinstance(document.get("project"), dict) else {}
    name = project.get("name")
    project_id = job.project_id if job is not None else document.get("project_id")
    if name is None and project_id:
        try:
            name = ctx.projects.summary(ctx.projects.get(project_id))["name"]
        except Exception:
            name = None
    return {"id": run.id, "status": status.get("status", "unknown"), "progress": status.get("progress") or {},
            "project_id": project_id, "name": name, "verdict": document.get("verdict"),
            "summary": document.get("summary"), "job_id": job.id if job is not None else None,
            "active": job is not None and job.state in ACTIVE}


def job_outcome(ctx: ApiContext, job, wait_s: float):
    """200 with the result when done, 202 while queued or running, 409 if cancelled, 501 when the
    entry point is not in this build, 500 when the job failed."""
    job = ctx.jobs.wait(job.id, wait_s)
    record = public_job(ctx, job)
    if job.state == "done":
        return json_response({"job": record, "result": ctx.jobs.result(job)})
    if job.state in ACTIVE:
        return json_response({"job": record}, 202, headers={"Location": f"/api/jobs/{job.id}"})
    if job.state == "cancelled":
        return json_response({"detail": f"job {job.id} was cancelled", "job": record}, 409)
    error = job.error or {}
    code = 501 if error.get("type") == "EntryPointMissing" else 500
    return json_response({"detail": error.get("message") or f"job {job.id} failed", "job": record}, code)


def array_doc(array: np.ndarray) -> dict:
    """An array as JSON: dtype, shape and nested lists (float16/32 at their own shortest digits)."""
    if array.dtype.kind == "f" and array.dtype.itemsize < 8:
        data = array.astype(str).astype(np.float64)
    else:
        data = array
    name = "str" if array.dtype.kind == "U" else array.dtype.name
    return {"dtype": name, "shape": list(array.shape), "data": data}
