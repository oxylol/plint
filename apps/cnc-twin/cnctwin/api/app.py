"""The FastAPI application (architecture 17): create_app(settings) builds it; the job manager runs
for the lifetime of the app (lifespan); the UI (web/) is served at ``/``.

Errors are JSON ``{"detail": "..."}`` with the HTTP status (400 bad input, 403 a path leaving its
folder, 404 not there (yet), 409 not in a state for this, 413 too large, 415 wrong body type,
422 a body or project that does not validate, 500/501 a job that failed or is not in this build,
503 the job manager is not running). The OpenAPI document is at /api/openapi.json (no interactive
docs pages: they load scripts from a CDN, and the box runs offline).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute

import cnctwin

from . import routes_db, routes_jobs, routes_projects, routes_runs, routes_sweeps
from .jsonio import json_response
from .safety import ApiError, resolve_inside
from .settings import Settings
from .views import ApiContext, file_response

DESCRIPTION = """\
HTTP API of CNC Twin, the one-to-one CNC machining simulator: projects, simulation runs (queued as
jobs, followed with Server-Sent Events, cancelled and resumed from checkpoints), everything a run
writes (findings and their causes, time series downsampled with M4, the scene, frames, keyframes,
chips, coolant jets, inspection, surface, tools, production, files with range requests), studies
(compare, uncertainty, calibration, parameter sweeps with their training data) and the databases. A run the run
cache holds is answered instead of simulated again. SI units throughout (names carry units).
See docs/api.md.
"""

TAGS = [
    {"name": "health", "description": "Is it up, and what this build has"},
    {"name": "projects", "description": "Example and uploaded projects"},
    {"name": "runs", "description": "Start, follow, cancel and resume simulations"},
    {"name": "results", "description": "Findings, causes, program, timeline, inspection, surface, tools, production"},
    {"name": "series", "description": "Channels, downsampled series, exports"},
    {"name": "scene", "description": "Scene, frames, keyframes, bricks, chips, jets"},
    {"name": "files", "description": "Any file of a run, with range requests"},
    {"name": "jobs", "description": "The job queue (simulations, checks, studies)"},
    {"name": "studies", "description": "Comparison, uncertainty, calibration and parameter sweeps"},
    {"name": "database", "description": "Machines, tools, holders, materials, coolants, bibliography"},
]


def _operation_id(route: APIRoute) -> str:
    return route.name


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    ctx = ApiContext.build(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await anyio.to_thread.run_sync(ctx.jobs.start)
        try:
            yield
        finally:
            await anyio.to_thread.run_sync(ctx.jobs.stop)

    app = FastAPI(title="CNC Twin API", version=cnctwin.__version__, description=DESCRIPTION,
                  openapi_url="/api/openapi.json", docs_url=None, redoc_url=None, openapi_tags=TAGS,
                  lifespan=lifespan, generate_unique_id_function=_operation_id)
    app.state.ctx = ctx

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError):
        return json_response({"detail": error.message, **error.extra}, error.status)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        return json_response({"detail": f"internal error: {type(error).__name__}: {error}"}, 500)

    routes_jobs.register(app, ctx)
    routes_projects.register(app, ctx)
    routes_runs.register(app, ctx)
    routes_sweeps.register(app, ctx)
    routes_db.register(app, ctx)
    _register_fallbacks(app, ctx)
    return app


def _register_fallbacks(app: FastAPI, ctx: ApiContext) -> None:
    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]

    @app.api_route("/api/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
                   include_in_schema=False)
    def api_not_found(rest: str, request: Request):
        path = f"/api/{rest}"
        allowed = sorted({m for r in api_routes if r.path_regex.match(path) for m in r.methods})
        if allowed:
            raise ApiError(405, f"{request.method} is not allowed on {path} (use {', '.join(allowed)})")
        raise ApiError(404, f"no API endpoint {path}")

    def ui(path: str = ""):
        """The UI (web/): files as they are, index.html for / and for client-side routes."""
        web = ctx.settings.web_dir
        index = web / "index.html"
        if not index.is_file():
            raise ApiError(404, f"the UI is not installed ({index} is missing); the API is at /api")
        target = index
        if path not in ("", "/"):
            try:
                target = resolve_inside(web, path)
            except ApiError as error:
                last = path.rstrip("/").rsplit("/", 1)[-1]
                if error.status != 404 or "." in last:
                    raise
        response = file_response(target, sandbox=False)
        response.headers["Cache-Control"] = "no-cache"
        return response

    app.add_api_route("/", ui, methods=["GET", "HEAD"], include_in_schema=False, name="ui_index")
    app.add_api_route("/{path:path}", ui, methods=["GET", "HEAD"], include_in_schema=False, name="ui_file")

