"""/api/runs: start, list, follow, cancel and resume simulations, and read everything a run wrote."""

from __future__ import annotations

import hashlib
import math
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, Query, Request
from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse

from ..core.errors import InputError
from ..db import yamlio
from . import series as series_mod
from .jsonio import json_response
from .models import (BINARY, GLB, SSE, ArraysResponse, FileList, FindingsResponse, FramesJson, JetsResponse,
                     KeyframeList, RunCreate, RunList, RunRecord, SeriesResponse, TraceResponse, errors)
from .runfiles import RunDir
from .safety import ApiError, check_id, list_files, resolve_inside, slugify
from .sse import run_events
from .views import ApiContext, array_doc, file_response, run_record, run_status, run_summary

SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "fatal": 3}
JSON_FRAME_LIMIT = 100000
FRAME_CHUNK = 65536


def register(app: FastAPI, ctx: ApiContext) -> None:
    runs = ctx.runs

    # ------------------------------------------------------------------ lifecycle
    @app.post("/api/runs", tags=["runs"], status_code=201, response_model=RunRecord,
              responses={200: {"model": RunRecord, "description": "The run cache holds this run: it is answered "
                                                                 "instead of a new simulation"},
                         **errors(400, 404, 422, 503)})
    def create_run(body: RunCreate):
        """Queue a simulation of a project; follow it with GET /api/runs/{id}/events. When the run cache holds a
        finished run of the same project, inputs, options and build (and ``no_cache`` is not set), that run is
        answered (200, ``cache.hit``) instead; a new run is registered in the cache when it has finished."""
        from . import runcache
        project = ctx.projects.get(body.project_id)
        _, error = ctx.projects.document(project)
        if error:
            raise ApiError(422, f"project {project.id}: {error}")
        cache = {"hit": False, "key": None, "note": "no_cache: the cache was not consulted"} if body.no_cache else None
        if not body.no_cache:
            cached_id, cache = runcache.lookup(ctx, project, body.options)
            if cached_id is not None:
                record = run_record(ctx, runs.get(cached_id))
                record["cache"] = cache
                return json_response(record, 200, headers={"Location": f"/api/runs/{cached_id}"})
        run_id, run_dir = runs.allocate(slugify(project.id))
        request = {"kind": "simulate", "run_id": run_id, "run_dir": str(run_dir), "project_id": project.id,
                   "project_path": str(project.file), "project_dir": str(project.dir), "options": body.options,
                   "resume_from": None, "cache": runcache.enabled()}
        try:
            ctx.jobs.submit("simulate", request, run_id=run_id, project_id=project.id)
        except Exception:
            if not any(run_dir.iterdir()):
                run_dir.rmdir()
            raise
        record = run_record(ctx, runs.get(run_id))
        record["cache"] = cache
        return json_response(record, 201, headers={"Location": f"/api/runs/{run_id}"})

    @app.get("/api/runs", tags=["runs"], response_model=RunList)
    def list_runs(project_id: str | None = None, status: str | None = None,
                  limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        """Runs, newest first (status, verdict and summary of each)."""
        items = []
        for run_id in sorted(runs.ids(), reverse=True):
            summary = run_summary(ctx, runs.get(run_id))
            if project_id is not None and summary["project_id"] != project_id:
                continue
            if status is not None and summary["status"] != status:
                continue
            items.append(summary)
        return json_response({"total": len(items), "runs": items[offset: offset + limit]})

    @app.get("/api/runs/{run_id}", tags=["runs"], response_model=RunRecord, responses=errors(404))
    def get_run(run_id: str):
        """run.json (null until finished) and status.json, with the latest job and checkpoint."""
        return json_response(run_record(ctx, runs.get(run_id)))

    @app.get("/api/runs/{run_id}/events", tags=["runs"], response_class=StreamingResponse,
             responses={200: {"content": SSE, "description": "Server-Sent Events"}, **errors(404)})
    async def run_events_stream(run_id: str, request: Request,
                                timeout_s: float | None = Query(None, gt=0, description="end the stream after this")):
        """SSE: ``progress`` (status.json changed), ``finding``, ``keyframe``, then ``done``."""
        run = runs.get(run_id)
        last = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
        stream = run_events(run, poll_s=ctx.settings.sse_poll_s, heartbeat_s=ctx.settings.sse_heartbeat_s,
                            last_event_id=last, timeout_s=timeout_s, job_status=lambda s: _with_queue(ctx, run, s))
        return StreamingResponse(stream, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/runs/{run_id}/cancel", tags=["runs"], status_code=202, response_model=RunRecord,
              responses=errors(404, 409))
    def cancel_run(run_id: str):
        """Stop a queued or running simulation (it stops at its next progress callback)."""
        run = runs.get(run_id)
        job = ctx.jobs.latest_for_run(run_id)
        if job is None:
            raise ApiError(409, f"run {run_id} was not started by this server; stop it where it runs")
        if job.state not in ("queued", "running", "cancelling"):
            raise ApiError(409, f"run {run_id} is not running ({job.state})")
        if job.state != "cancelling":
            ctx.jobs.cancel(job.id)
        return json_response(run_record(ctx, run), 202)

    @app.post("/api/runs/{run_id}/resume", tags=["runs"], status_code=202, response_model=RunRecord,
              responses=errors(404, 409))
    def resume_run(run_id: str):
        """Continue a cancelled or failed simulation from its last committed checkpoint
        (from the start when it has none)."""
        run = runs.get(run_id)
        job = ctx.jobs.latest_for_run(run_id)
        if job is None:
            raise ApiError(409, f"run {run_id} was not started by this server; resume it with twin")
        if job.state in ("queued", "running", "cancelling") and run.state() in ("done", "failed", "cancelled"):
            job = ctx.jobs.wait(job.id, ctx.settings.kill_grace_s)       # the worker is writing its last lines
        if job.state in ("queued", "running", "cancelling"):
            raise ApiError(409, f"run {run_id} is already {job.state}")
        state = run.state()
        if state == "done" or job.state == "done":
            raise ApiError(409, f"run {run_id} has finished; start a new run instead")
        checkpoint = run.latest_checkpoint()
        request = dict(job.request)
        request["resume_from"] = checkpoint["path"] if checkpoint else None
        request.pop("out_dir", None)
        progress = run.status().get("progress") or {}
        prior_wall = float(progress.get("wall_s") or 0.0)
        ctx.jobs.submit("simulate", request, run_id=run_id, project_id=job.project_id, attempt=job.attempt + 1,
                        prior_wall_s=prior_wall)
        return json_response(run_record(ctx, run), 202)

    # ------------------------------------------------------------------ findings, trace, channels
    @app.get("/api/runs/{run_id}/findings", tags=["results"], response_model=FindingsResponse, responses=errors(400, 404))
    def get_findings(run_id: str, severity: str | None = Query(None, description="minimum: info|warning|error|fatal"),
                     category: str | None = Query(None, description="comma-separated categories"),
                     op: str | None = None):
        run = runs.get(run_id)
        items = run.findings()
        worst = max((f.get("severity", "info") for f in items), key=lambda s: SEVERITY_RANK.get(s, 0), default=None)
        document = run.run_json() or {}
        verdict = document.get("verdict") if isinstance(document.get("verdict"), dict) else None
        safe = bool(verdict["safe"]) if verdict and "safe" in verdict else (
            worst is None or SEVERITY_RANK.get(worst, 0) < SEVERITY_RANK["error"])
        if severity is not None:
            if severity not in SEVERITY_RANK:
                raise ApiError(400, f"severity {severity!r}: use info, warning, error or fatal")
            items = [f for f in items if SEVERITY_RANK.get(f.get("severity"), 0) >= SEVERITY_RANK[severity]]
        if category:
            wanted = {c.strip() for c in category.split(",") if c.strip()}
            items = [f for f in items if f.get("category") in wanted]
        if op is not None:
            items = [f for f in items if f.get("op") == op]
        return json_response({"run_id": run_id, "total": len(run.findings()), "count": len(items), "worst": worst,
                              "safe": safe, "findings": items})

    @app.get("/api/runs/{run_id}/trace/{event_id}", tags=["results"], response_model=TraceResponse,
             responses=errors(404))
    def get_trace(run_id: str, event_id: int, max_events: int = Query(200, ge=1, le=5000)):
        """A trace event and its chain of causes (ancestors, breadth first)."""
        run = runs.get(run_id)
        event = run.trace_event(event_id)
        if event is None:
            raise ApiError(404, f"run {run_id} has no trace event {event_id}")
        return json_response({"event": event, "chain": run.trace_chain(event_id, max_events)})

    @app.get("/api/runs/{run_id}/channels", tags=["series"], responses=errors(404))
    def get_channels(run_id: str, tier: str | None = None):
        """channels.json: every recorded channel with unit, tier, owner, doc, max_shape and dtype."""
        items = runs.get(run_id).channels()
        if tier is not None:
            items = [c for c in items if c.get("tier") == tier]
        return json_response({"count": len(items), "channels": items})

    # ------------------------------------------------------------------ series
    @app.get("/api/runs/{run_id}/series", tags=["series"], response_model=SeriesResponse, responses=errors(400, 404))
    def get_series(run_id: str, tier: Literal["fast", "tick", "macro"] = "tick",
                   channels: str = Query("", description="comma-separated channel names or columns"),
                   t0: float | None = Query(None, description="from (s, simulated)"),
                   t1: float | None = Query(None, description="to (s, simulated, inclusive)"),
                   max_points: int = Query(2000, ge=4, le=200000, description="M4-downsample above this")):
        """Channel values over time; above max_points rows, M4 keeps each bucket's first, last,
        minimum and maximum sample (peaks are never smoothed away)."""
        run = runs.get(run_id)
        tokens = [c for c in channels.split(",") if c.strip()]
        return json_response(series_mod.query(run, tier, tokens, t0, t1, max_points))

    @app.get("/api/runs/{run_id}/export", tags=["series"],
             responses={200: {"content": {"text/csv": {}, "application/vnd.apache.parquet": {}}}, **errors(400, 404)})
    def export(run_id: str, format: Literal["csv", "parquet"] = "csv",
               tier: Literal["fast", "tick", "macro"] = "tick",
               channels: str = Query("", description="comma-separated; empty = every column"),
               t0: float | None = None, t1: float | None = None):
        """Full-resolution series of one tier as CSV (streamed; NaN as empty) or one Parquet file."""
        run = runs.get(run_id)
        tokens = [c for c in channels.split(",") if c.strip()]
        columns = series_mod.export_columns(run, tier, tokens)
        name = f"{run_id}-{tier}"
        if format == "csv":
            return StreamingResponse(series_mod.export_csv(run, tier, columns, t0, t1),
                                     media_type="text/csv; charset=utf-8",
                                     headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
        ctx.settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        target = ctx.settings.tmp_dir / f"export-{uuid.uuid4().hex}.parquet"
        try:
            series_mod.export_parquet(run, tier, columns, t0, t1, target)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        response = file_response(target, download_name=f"{name}.parquet")
        response.background = BackgroundTask(target.unlink, missing_ok=True)
        return response

    # ------------------------------------------------------------------ program, timeline, scene, frames
    @app.get("/api/runs/{run_id}/program/{op}", tags=["results"], responses=errors(404, 409))
    def get_program(run_id: str, op: str):
        """The program of an operation (by name, or index) as it was run."""
        run = runs.get(run_id)
        return json_response(_program(ctx, run, op))

    @app.get("/api/runs/{run_id}/timeline", tags=["results"], responses=errors(404))
    def get_timeline(run_id: str, op: str | None = None, t0: float | None = None, t1: float | None = None):
        """timeline.json: the executed blocks with their tick ranges (filtered by op and time)."""
        run = runs.get(run_id)
        meta, blocks = run.timeline_blocks()
        if op is not None:
            blocks = [b for b in blocks if b.get("op") == op]
        tick_s = run.tick_s()
        if t0 is not None:
            tick0 = series_mod.seconds_to_tick(t0, tick_s)
            blocks = [b for b in blocks if not isinstance(b.get("tick1"), int) or b["tick1"] > tick0]
        if t1 is not None:
            tick1 = series_mod.seconds_to_tick(t1, tick_s)
            blocks = [b for b in blocks if not isinstance(b.get("tick0"), int) or b["tick0"] <= tick1]
        return json_response({**meta, "count": len(blocks), "blocks": blocks})

    @app.get("/api/runs/{run_id}/scene", tags=["scene"], responses=errors(404))
    def get_scene(run_id: str):
        """scene.json: machine components, chain, axes, fixture, stock, tools, nozzles, frames header."""
        return json_response(runs.get(run_id).json("scene.json"))

    @app.get("/api/runs/{run_id}/frames", tags=["scene"], response_model=FramesJson,
             responses={200: {"content": BINARY, "description": "records as in scene.json frames header"},
                        **errors(400, 404)})
    def get_frames(run_id: str, t0: float | None = None, t1: float | None = None,
                   format: Literal["bin", "json"] = "bin",
                   max_frames: int | None = Query(None, ge=1, description="every n-th frame to stay below this")):
        """frames.bin records between t0 and t1 (raw bytes, or column-wise JSON)."""
        run = runs.get(run_id)
        records, header, dtype = run.frame_records()
        tick_s = float(header.get("tick_s") or run.tick_s())
        total = 0 if records is None else len(records)
        i0, i1 = 0, total
        if records is not None:
            ticks = records["tick"]
            if t0 is not None:
                i0 = int(np.searchsorted(ticks, series_mod.seconds_to_tick(t0, tick_s), "left"))
            if t1 is not None:
                i1 = int(np.searchsorted(ticks, series_mod.seconds_to_tick(t1, tick_s), "right"))
        count = max(0, i1 - i0)
        limit = max_frames if max_frames is not None else (JSON_FRAME_LIMIT if format == "json" else None)
        stride = int(math.ceil(count / limit)) if limit and count > limit else 1
        returned = (count + stride - 1) // stride if count else 0
        if format == "json":
            subset = records[i0:i1:stride] if records is not None and count else None
            fields = {name: (array_doc(np.asarray(subset[name]))["data"] if subset is not None else [])
                      for name in dtype.names}
            return json_response({"run_id": run_id, "count": returned, "stride": stride, "tick_s": tick_s,
                                  "header": header, "fields": fields})
        headers = {"X-Frame-Count": str(returned), "X-Frame-Stride": str(stride), "X-Record-Bytes": str(dtype.itemsize),
                   "X-Tick-S": repr(tick_s), "Content-Length": str(returned * dtype.itemsize)}
        if returned:
            headers["X-First-Tick"] = str(int(records["tick"][i0]))

        def chunks():
            if not returned:
                return
            step = FRAME_CHUNK * stride
            for start in range(i0, i1, step):
                yield np.ascontiguousarray(records[start:min(i1, start + step):stride]).tobytes()
        return StreamingResponse(chunks(), media_type="application/octet-stream", headers=headers)

    # ------------------------------------------------------------------ keyframes, bricks, chips, jets
    @app.get("/api/runs/{run_id}/keyframes", tags=["scene"], response_model=KeyframeList, responses=errors(404))
    def list_keyframes(run_id: str):
        frames = runs.get(run_id).keyframes()
        return json_response({"count": len(frames), "keyframes": frames})

    @app.get("/api/runs/{run_id}/keyframes/{k}.glb", tags=["scene"], response_class=Response,
             responses={200: {"content": GLB}, **errors(404)})
    def keyframe_glb(run_id: str, k: int):
        """The changed in-process-workpiece bricks of keyframe k (full snapshot every 20 keyframes)."""
        run = runs.get(run_id)
        entry = run.keyframe(k)
        return file_response(run.file(str(entry.get("glb") or f"keyframes/kf_{k}.glb")))

    @app.head("/api/runs/{run_id}/keyframes/{k}.glb", include_in_schema=False)
    def keyframe_glb_head(run_id: str, k: int):
        return keyframe_glb(run_id, k)

    @app.get("/api/runs/{run_id}/keyframes/{k}/fields", tags=["scene"], response_model=ArraysResponse,
             responses={200: {"content": BINARY}, **errors(404)})
    def keyframe_fields(run_id: str, k: int, names: str = Query("", description="comma-separated arrays; empty = all"),
                        format: Literal["json", "npz"] = "json"):
        """Coarse fields of keyframe k: surface temperatures, chip piles, wetted cells, tool deflection,
        part pose, coolant jets (kf_<k>.npz)."""
        run = runs.get(run_id)
        entry = run.keyframe(k)
        relative = str(entry.get("fields") or f"keyframes/kf_{k}.npz")
        return _arrays(run, relative, k, run.keyframe_time(k), names, format)

    @app.get("/api/runs/{run_id}/bricks/{brick}", tags=["scene"], response_class=Response,
             responses={200: {"content": BINARY, "description": "the brick's record in its keyframe file"},
                        **errors(404)})
    def get_brick(run_id: str, brick: str, t: float | None = Query(None, description="time (s); default: latest")):
        """One brick of the in-process workpiece as it was at time t: one ranged read of the
        keyframe file named by keyframes/bricks.idx (the latest entry at or before t)."""
        run = runs.get(run_id)
        check_id(brick, "brick")
        entries = run.bricks_index().get(brick)
        if not entries:
            raise ApiError(404, f"run {run_id} has no brick {brick!r}")
        if t is None:
            chosen = entries[-1]
        else:
            times = {int(e["k"]): run.keyframe_time(int(e["k"])) for e in run.keyframes()}
            earlier = [e for e in entries if times.get(e[0]) is not None and times[e[0]] <= t + 1e-12]
            if not earlier:
                raise ApiError(404, f"brick {brick} has no state at or before t = {t} s")
            chosen = earlier[-1]
        k, name, offset, length = chosen
        path = run.file(name) if name.startswith("keyframes/") else resolve_inside(run.path / "keyframes", name)
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise ApiError(500, f"run {run_id}: {name} is shorter than bricks.idx says")
        time_s = run.keyframe_time(k)
        headers = {"X-Keyframe": str(k), "X-Brick": brick, "X-Source-File": name, "X-Source-Offset": str(offset)}
        if time_s is not None:
            headers["X-Keyframe-Time-S"] = repr(time_s)
        return Response(data, media_type="application/octet-stream", headers=headers)

    @app.get("/api/runs/{run_id}/chips", tags=["scene"], response_model=ArraysResponse,
             responses={200: {"content": BINARY}, **errors(404)})
    def get_chips(run_id: str, t: float | None = Query(None, description="time (s); default: latest"),
                  names: str = "", format: Literal["json", "npz"] = "json"):
        """Chips in flight and the pile summary at the keyframe at or before t (chips/kf_<k>.npz)."""
        run = runs.get(run_id)
        entry = run.keyframe_at(t)
        k = int(entry["k"])
        return _arrays(run, str(entry.get("chips") or f"chips/kf_{k}.npz"), k, run.keyframe_time(k), names, format)

    @app.get("/api/runs/{run_id}/jets", tags=["scene"], response_model=JetsResponse, responses=errors(404))
    def get_jets(run_id: str, t: float | None = Query(None, description="time (s); default: latest")):
        """Coolant jets (in M) at the keyframe at or before t, and the nozzles of the scene."""
        run = runs.get(run_id)
        scene = run.optional_json("scene.json") or {}
        nozzles = scene.get("nozzles", []) if isinstance(scene, dict) else []
        frames = run.keyframes()
        if not frames:
            return json_response({"k": None, "time_s": None, "jets": [], "nozzles": nozzles})
        entry = run.keyframe_at(t)
        k = int(entry["k"])
        arrays = run.npz(str(entry.get("fields") or f"keyframes/kf_{k}.npz"))
        columns = {name[len("jet_"):]: value for name, value in arrays.items() if name.startswith("jet_")}
        count = min((len(v) for v in columns.values()), default=0)
        jets = [{name: value[i] for name, value in columns.items()} for i in range(count)]
        return json_response({"k": k, "time_s": run.keyframe_time(k), "jets": jets, "nozzles": nozzles})

    # ------------------------------------------------------------------ results and files
    @app.get("/api/runs/{run_id}/sld", tags=["results"], responses=errors(404))
    def list_sld(run_id: str):
        """Tools with a stability lobe diagram."""
        run = runs.get(run_id)
        folder = run.path / "sld"
        tools = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
        return json_response({"count": len(tools), "tools": tools})

    @app.get("/api/runs/{run_id}/sld/{tool}", tags=["results"], responses=errors(404))
    def get_sld(run_id: str, tool: str):
        """Stability lobe diagram of a tool (sld/<tool>.json; a bare number n also finds T<n>)."""
        run = runs.get(run_id)
        check_id(tool, "tool")
        if not run.has(f"sld/{tool}.json") and tool.isdigit() and run.has(f"sld/T{tool}.json"):
            tool = f"T{tool}"
        return json_response(run.json(f"sld/{tool}.json", missing=f"run {run_id} has no stability lobes for {tool}"))

    for name, doc in (("inspection", "Measured tolerances (inspection.json)"),
                      ("surface", "Surface finish and burrs (surface.json)"),
                      ("tools", "Tool wear and life (tools.json)"),
                      ("production", "Cycle time, cost and energy (production.json)")):
        _result_route(app, ctx, name, doc)

    @app.get("/api/runs/{run_id}/files", tags=["files"], response_model=FileList, responses=errors(404))
    def list_run_files(run_id: str):
        files = list_files(runs.get(run_id).path)
        return json_response({"count": len(files), "files": files})

    @app.get("/api/runs/{run_id}/files/{path:path}", tags=["files"], response_class=Response,
             responses={200: {"content": BINARY}, 206: {"description": "Partial content (Range)"}, **errors(400, 403, 404)})
    def get_run_file(run_id: str, path: str):
        """Any file of the run (video.mp4, report.pdf, part_<op>.glb ...), with HTTP range requests."""
        return file_response(runs.get(run_id).file(path))

    @app.head("/api/runs/{run_id}/files/{path:path}", include_in_schema=False)
    def head_run_file(run_id: str, path: str):
        return get_run_file(run_id, path)


def _result_route(app: FastAPI, ctx: ApiContext, name: str, doc: str) -> None:
    def handler(run_id: str):
        return json_response(ctx.runs.get(run_id).json(f"{name}.json"))
    handler.__name__ = f"get_{name}"
    handler.__doc__ = doc
    app.get(f"/api/runs/{{run_id}}/{name}", tags=["results"], responses=errors(404), name=f"get_{name}")(handler)


def _with_queue(ctx: ApiContext, run: RunDir, status: dict) -> dict:
    if status.get("status") == "queued":
        return run_status(ctx, run)
    return status


def _arrays(run: RunDir, relative: str, k: int, time_s, names: str, format: str):
    if format == "npz":
        return file_response(run.file(relative))
    wanted = [n.strip() for n in names.split(",") if n.strip()] or None
    arrays = run.npz(relative, wanted)
    return json_response({"k": k, "time_s": time_s, "file": relative,
                          "arrays": {name: array_doc(value) for name, value in arrays.items()}})


def _program(ctx: ApiContext, run: RunDir, op: str) -> dict:
    document = run.run_json() or {}
    project = document.get("project") if isinstance(document.get("project"), dict) else {}
    operations = project.get("operations") if isinstance(project.get("operations"), list) else None
    project_dir = document.get("project_dir")
    job = ctx.jobs.latest_for_run(run.id)
    if operations is None and job is not None:
        try:
            data = yamlio.load_file(job.request["project_path"]).data
        except InputError as error:
            raise ApiError(404, f"run {run.id}: its project cannot be read ({error})") from None
        operations = data.get("operations") if isinstance(data, dict) else None
    if project_dir is None and job is not None:
        project_dir = job.request.get("project_dir")
    if not operations:
        raise ApiError(404, f"run {run.id} records no operations (yet)")
    chosen = next((o for o in operations if isinstance(o, dict) and o.get("name") == op), None)
    if chosen is None and op.isdigit() and int(op) < len(operations):
        chosen = operations[int(op)]
    if chosen is None:
        names = ", ".join(str(o.get("name")) for o in operations if isinstance(o, dict))
        raise ApiError(404, f"run {run.id} has no operation {op!r} (it has {names})")
    program = chosen.get("program")
    inputs = document.get("inputs") if isinstance(document.get("inputs"), dict) else {}
    expected = chosen.get("program_sha256") or (inputs.get(program) or {}).get("sha256") if program else None
    text = chosen.get("program_text")
    source = "run.json"
    if not isinstance(text, str):
        if not program or not project_dir:
            raise ApiError(404, f"run {run.id}: the program of {op} was not recorded")
        path = resolve_inside(Path(project_dir), program)
        raw = path.read_bytes()
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ApiError(409, f"{program} has changed since run {run.id} (sha256 differs from run.json)")
        text = raw.decode("utf-8", "replace")
        source = "project"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"op": chosen.get("name"), "program": program, "source": source, "sha256": digest,
            "lines": len(text.splitlines()), "text": text}


__all__ = ["register"]
