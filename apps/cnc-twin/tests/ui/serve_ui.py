"""The UI test server: the real API and UI (cnctwin.api.create_app) on a throwaway DATA_DIR with the
synthetic runs and example projects of ui_fixtures.py, and the job entry points of fake_jobs.py
(comparisons use the real cnctwin.report.compare). Run it under bin/tsafe on a port in 5990-5999:

    bin/tsafe python tests/ui/serve_ui.py --data /tmp/cnctwin-ui/data --port 5995 --build

Settings are passed in code, not through the environment (bin/tsafe runs through sudo, which
resets the environment). Jobs are capped with RLIMIT_DATA (memory mode rlimit).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

MiB = 1024 * 1024


def settings_for(data: Path, port: int, web: Path):
    from cnctwin.api.settings import Settings
    fakes = HERE / "fake_jobs.py"
    entries = {"simulate": f"{fakes}:run_project", "check": f"{fakes}:check_project",
               "compare": "cnctwin.report.compare:compare_runs", "uncertainty": f"{fakes}:run_study",
               "calibrate": f"{fakes}:calibrate"}
    return Settings(data_dir=data, examples_dir=data / "examples", web_dir=web, db_dir=ROOT / "db", host="127.0.0.1",
                    port=port, memory_mode="rlimit", heavy_memory_bytes=1024 * MiB, light_memory_bytes=1024 * MiB,
                    entries=entries, cancel_on_shutdown=True, poll_s=0.1, sse_poll_s=0.1, sse_heartbeat_s=5.0,
                    status_interval_s=0.1, cancel_grace_s=5.0, kill_grace_s=2.0, log_level="warning")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CNC Twin UI test server")
    parser.add_argument("--data", required=True, help="throwaway DATA_DIR")
    parser.add_argument("--port", type=int, default=5995, help="5990-5999 on this box")
    parser.add_argument("--web", default=str(ROOT / "web"), help="the UI folder (default web/)")
    parser.add_argument("--build", action="store_true", help="(re)write the synthetic data first")
    args = parser.parse_args(argv)
    if not 5990 <= args.port <= 5999:
        parser.error("use a port in 5990-5999")
    data = Path(args.data)
    if args.build or not (data / "runs").is_dir():
        import ui_fixtures
        ui_fixtures.build_data_dir(data)
    import uvicorn
    from cnctwin.api.app import create_app
    settings = settings_for(data, args.port, Path(args.web))
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="warning", workers=1,
                timeout_graceful_shutdown=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
