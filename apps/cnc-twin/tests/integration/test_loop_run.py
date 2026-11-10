"""A short real program through cnctwin.core.loop.run_project with every domain on (docs/models/loop.md
"Tests"): the ledgers pass every step and at the end, every file of the run directory (architecture 16) is
written and indexed, the derived outputs (report, pictures, video, sound) are made, the verdict is sensible, and
the physics agrees with the run's own records (removed volume, chip mass, cutting energy, timeline)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cnctwin.core.loop import run_project
from cnctwin.core.rundir import read_frames, result_set_digest
from tests.integration import loopfix as LF


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    base = tmp_path_factory.mktemp("loop-run")
    project = LF.tiny_mill(base / "project")
    events = []

    def progress(progress=None, **fields):
        events.append(dict(progress or {}, **fields))

    result = run_project(project, base / "run", {"outputs": LF.SMALL_OUTPUTS}, progress)
    return base / "run", result, events


def test_the_run_completes_with_a_sensible_verdict(finished):
    run_dir, result, events = finished
    run = LF.read(run_dir, "run.json")
    assert result["complete"] and run["complete"]
    assert run["verdict"]["safe"] is True, run["verdict"]["reasons"]
    findings = LF.read(run_dir, "findings.json")["findings"]
    assert not [f for f in findings if f["severity"] in ("error", "fatal")]
    summary = run["summary"]
    assert summary["removed_volume_m3"] > 0.0 and summary["cycle_time_s"] > 1.0
    assert summary["sim_time_s"] >= summary["cycle_time_s"]
    assert any(e.get("phase") == "done" for e in events) and any("sim_time_s" in e for e in events)
    strides = run["clock"]["l0_ticks"]["OP10"]
    assert all(1000 % v == 0 for v in (strides if isinstance(strides, list) else [strides]))


def test_every_ledger_passes(finished):
    run_dir, _, _ = finished
    run = LF.read(run_dir, "run.json")
    assert LF.conservation_ok(run) == {"mass": True, "energy": True, "time": True, "final_mass": True}
    cons = run["summary"]["conservation"]
    assert cons["mass"]["max_residual_rel"] <= 1e-9 and cons["energy"]["max_residual_rel"] <= 1e-6
    assert cons["time"]["residual_ticks"] == 0
    assert cons["mass"]["checked_steps"] > 100 and cons["energy"]["checked_steps"] > 10
    # cutting's own identity: 5.7e-11 dry, 1.2e-9 since the flooded contact is lubricated (friction factor varying
    # from step to step; cause not isolated, docs/models/loop.md 15); the run's energy contract above is 1e-6
    assert run["summary"]["cutting_identity_rel"] < 1e-8
    # the removed mass is in the six chip inventories
    chips = sum(run["summary"]["chips_kg"].values())
    assert chips == pytest.approx(run["summary"]["removed_mass_kg"], rel=1e-9)


def test_the_step_has_the_volume_of_the_tool_path(finished):
    run_dir, _, _ = finished
    run = LF.read(run_dir, "run.json")
    # the D10 at Y-4.5 enters at the x 0 face, runs to x 12 and leaves at -y (G54 X0 Y0 at the block's corner,
    # the top at S z 10 mm): a step about 0.5 mm wide and 1 mm deep. The reference is the path the tool took
    # (frames.bin): the control rounds the corner at x 12 within its look-ahead tolerance. The field's rays are
    # 0.25 mm apart: its volume agrees to about 1 %.
    swept = LF.swept_volume_m3(run_dir, 10e-3, ((0.0, 0.0), (30e-3, 20e-3)))
    assert 5.0e-9 < swept < 7.0e-9
    assert run["summary"]["removed_volume_m3"] == pytest.approx(swept, rel=0.03)


def test_the_run_directory_is_complete(finished):
    run_dir, _, _ = finished
    for name in ("run.json", "findings.json", "trace.jsonl", "channels.json", "timeline.json", "scene.json",
                 "frames.bin", "keyframes/bricks.idx", "keyframes/index.jsonl", "series/index.jsonl",
                 "inspection.json", "surface.json", "tools.json", "production.json", "part_OP10.stl",
                 "part_OP10.glb", "part_OP10_unload.stl", "report.html", "report.pdf", "setup_sheet.html",
                 "video.mp4", "audio.wav", "render.json"):
        assert (run_dir / name).is_file(), name
    assert list((run_dir / "sld").glob("*.json")) and list((run_dir / "images").glob("*.png"))
    assert list((run_dir / "series" / "tick").glob("part-*.parquet"))
    assert list((run_dir / "series" / "macro").glob("part-*.parquet"))
    assert list((run_dir / "checkpoints").glob("ck_*.json"))
    run = LF.read(run_dir, "run.json")
    indexed = {e["path"] for e in run["outputs"]}
    for name in ("inspection.json", "tools.json", "production.json", "frames.bin", "part_OP10.stl", "report.html"):
        assert name in indexed, name
    assert all(g["status"] == "done" for g in run["generators"].values())
    assert set(result_set_digest(run_dir)) >= {"findings.json", "trace.jsonl", "frames.bin", "tools.json"}


def test_frames_timeline_and_production_agree(finished):
    run_dir, _, _ = finished
    records, header = read_frames(run_dir)
    ticks = records["tick"]
    assert len(ticks) > 100 and np.all(np.diff(ticks) == 1000)
    timeline = LF.read(run_dir, "timeline.json")["blocks"]
    assert timeline and all(b["tick1"] >= b["tick0"] for b in timeline)
    cats = {}
    for b in timeline:
        assert sum(b.get("ticks", {}).values()) == b["tick1"] - b["tick0"]
        for k, v in b.get("ticks", {}).items():
            cats[k] = cats.get(k, 0) + v
    assert cats.get("cutting", 0) > 0 and cats.get("tool_change", 0) > 0
    production = LF.read(run_dir, "production.json")
    assert production["parts_simulated"] == 1
    tools = LF.read(run_dir, "tools.json")["tools"]
    assert tools[0]["number"] == 1 and tools[0]["cut_time_s"] > 0.0


def test_every_domain_left_its_mark(finished):
    run_dir, _, _ = finished
    run = LF.read(run_dir, "run.json")
    domains = run["domains"]
    for name in ("thermal", "chips", "coolant", "holding", "structure", "dynamics", "removal", "cutting", "controller",
                 "tooling", "motion"):
        assert name in domains, name
    channels = {c["name"] for c in LF.read(run_dir, "channels.json")["channels"]}
    for prefix in ("motion.", "cutting.", "removal.", "dynamics.", "thermal.", "chips.", "coolant.", "wear.",
                   "holding.", "structure.", "surface.", "ledger.", "controller.", "tooling.", "hotloop."):
        assert any(c.startswith(prefix) for c in channels), prefix
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    events = {e["event"] for e in trace}
    assert {"loop:start", "loop:operation_start", "loop:operation_end"} <= events
    assert any(e["domain"] == "removal" for e in trace) and any(e["domain"] == "wear" for e in trace)
