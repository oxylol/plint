"""The 3D view in Chromium on the other runs (browser_test.mjs SUITE=runs): the default synthetic run (box
bricks, 2-D surface grids read rows = y, attribute layers with the shell's ranges), a running run read
through the contract's minimal api {get, getBinary} (no final attributes yet, refreshed index files), and a
phone with touch (pixel-ratio cap, tap to pick, 44 px buttons); no console errors."""

from v3d_support import run_browser_suite


def test_view3d_runs_and_phone_in_chromium(v3d_server):
    document = run_browser_suite(v3d_server, "runs")
    failed = [f"{r['name']}: {r['detail']}" for r in document["results"] if not r["pass"]]
    assert not failed, "\n".join(failed)
    assert document["passed"] >= 20, document["stdout"][-2000:]
