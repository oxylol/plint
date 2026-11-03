"""Test server of the 3D view: the real API (cnctwin.api.create_app) on a throwaway DATA_DIR with
synthetic runs (tests/fixtures/make_synthetic_run.py), serving a copy of web/ with the test page
(harness/view3d-harness.html) at its root. Run it under bin/tsafe on a port in 5990-5999:

    bin/tsafe python tests/unit/view3d/serve_view3d.py --data /tmp/cnctwin-v3d/data --web /tmp/cnctwin-v3d/web --port 5996 --build

Runs written (RUNS): the default synthetic run (seed 2), the same run with the 3D view's additions
(seed 4, view3d=True: carved pocket bricks, a 3-D temperature grid, jets with paths and a head-mounted
blocked jet, a tool displacement channel), and a run stopped at 1.2 s (status running).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for entry in (str(ROOT), str(ROOT / "tests" / "fixtures")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

RUNS = {"00001-v3d-default": {"seed": 2}, "00002-v3d-rich": {"seed": 4, "view3d": True},
        "00003-v3d-partial": {"seed": 5, "view3d": True, "until_s": 1.2}}


def build(data: Path) -> dict:
    import make_synthetic_run as synthetic
    if data.exists():
        shutil.rmtree(data)
    (data / "runs").mkdir(parents=True)
    (data / "examples").mkdir(parents=True)
    out = {}
    for run_id, spec in RUNS.items():
        kwargs = {"media": False}
        if spec.get("view3d"):
            kwargs["view3d"] = True
        if spec.get("until_s") is not None:
            kwargs["until_s"] = spec["until_s"]
        out[run_id] = synthetic.make_synthetic_run(data / "runs" / run_id, seed=spec["seed"], **kwargs)
    return out


def web_copy(web: Path) -> None:
    if web.exists():
        shutil.rmtree(web)
    shutil.copytree(ROOT / "web", web)
    for name in ("view3d-harness.html", "view3d-harness.js"):
        shutil.copy2(HERE / "harness" / name, web / name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="3D view test server")
    parser.add_argument("--data", required=True)
    parser.add_argument("--web", required=True)
    parser.add_argument("--port", type=int, default=5996)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    if not 5990 <= args.port <= 5999:
        parser.error("use a port in 5990-5999")
    data = Path(args.data)
    if args.build or not (data / "runs").is_dir():
        build(data)
    web_copy(Path(args.web))
    import uvicorn
    from cnctwin.api.app import create_app
    from cnctwin.api.settings import Settings
    settings = Settings(data_dir=data, examples_dir=data / "examples", web_dir=Path(args.web), db_dir=ROOT / "db",
                        host="127.0.0.1", port=args.port, memory_mode="none", log_level="warning")
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="warning", workers=1,
                timeout_graceful_shutdown=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
