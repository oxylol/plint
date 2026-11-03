"""The 3D view in Chromium on the rich synthetic run (browser_test.mjs SUITE=desktop): mounting, the workpiece
at each time (brick versions as bricks.idx gives them), playback, the tool pose against frames.bin, every
layer with the shell's colour range, capped sections, picking, the three measurements, follow and reset,
visibility, the deflected tool and its exaggeration, coolant shadows, destroy, no console errors."""

from v3d_support import run_browser_suite


def test_view3d_contract_in_chromium(v3d_server):
    document = run_browser_suite(v3d_server, "desktop")
    failed = [f"{r['name']}: {r['detail']}" for r in document["results"] if not r["pass"]]
    assert not failed, "\n".join(failed)
    assert document["passed"] >= 70, document["stdout"][-2000:]
