"""``cnctwin.api.serve(argv)``: the API server under uvicorn (one process: the job manager is in it).

    python -m cnctwin.api --host 127.0.0.1 --port 5210 --data-dir /data

Every option has an environment variable (the option wins): DATA_DIR, CNCTWIN_HOST, CNCTWIN_PORT (or PORT),
CNCTWIN_MAX_JOBS, CNCTWIN_JOB_MEMORY, CNCTWIN_JOB_MEMORY_MODE, CNCTWIN_LOG_LEVEL (and the others
in cnctwin.api.settings). Open SSE streams are closed 5 s after a shutdown request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .settings import MEMORY_MODES, Settings, parse_bytes

GRACEFUL_SHUTDOWN_S = 5


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.api", description="CNC Twin HTTP API server")
    parser.add_argument("--host", help="address to listen on (CNCTWIN_HOST, default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (CNCTWIN_PORT or PORT, default 5210)")
    parser.add_argument("--data-dir", type=Path, help="projects, runs and jobs (DATA_DIR)")
    parser.add_argument("--max-jobs", type=int, help="simulations at once (CNCTWIN_MAX_JOBS, default 1)")
    parser.add_argument("--job-memory", help="memory cap of a simulation, e.g. 1536M (CNCTWIN_JOB_MEMORY)")
    parser.add_argument("--memory-mode", choices=MEMORY_MODES, help="how jobs are capped (CNCTWIN_JOB_MEMORY_MODE)")
    parser.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug"),
                        help="uvicorn log level (CNCTWIN_LOG_LEVEL, default info)")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace, environ=None) -> Settings:
    overrides = {}
    if args.host:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.max_jobs is not None:
        overrides["max_heavy_jobs"] = args.max_jobs
    if args.job_memory:
        overrides["heavy_memory_bytes"] = parse_bytes(args.job_memory)
    if args.memory_mode:
        overrides["memory_mode"] = args.memory_mode
    if args.log_level:
        overrides["log_level"] = args.log_level
    return Settings.from_env(environ, **overrides)


def serve(argv=None) -> int:
    import uvicorn

    from .app import create_app
    settings = settings_from_args(parse_args(argv))
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=settings.log_level,
                workers=1, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S)
    return 0
