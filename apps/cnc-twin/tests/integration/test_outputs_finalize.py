"""finalize_outputs on the synthetic run directory (tests/fixtures/make_synthetic_run.py): every derived output is
made in its own capped subprocess and registered in run.json, a failed or timed-out generator is recorded without
losing the others, a cancel stops the generator, settings come from the project and the options."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from cnctwin.core.errors import InputError
from cnctwin.core.outputs import (GENERATORS, finalize_outputs, main, output_settings, positioned_findings,
                                  update_index)
from tests.integration import test_outputs_fakes as fakes

SMALL = fakes.SMALL
STANDARD = ("setup.png", "cut.png", "part_OP10.png", "part_OP10_section.png", "temperature.png", "chip_pile.png",
            "coolant.png", "deflection.png", "deviation.png", "finish.png", "wear.png")
templates = fakes.templates


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def finalized(tmp_path_factory):
    runs = templates(tmp_path_factory)
    run_dir = tmp_path_factory.mktemp("outputs") / "00001-pocket"
    shutil.copytree(runs["unsafe"], run_dir)
    phases = []

    def progress(progress=None, **fields):
        phases.append((fields.get("phase"), fields.get("message")))

    summary = finalize_outputs(run_dir, {"outputs": SMALL}, progress)
    return run_dir, summary, phases


def test_every_output_is_made_and_indexed(finalized):
    run_dir, summary, phases = finalized
    assert summary["failed"] == [] and [summary["generators"][n]["status"] for n in GENERATORS] == ["done"] * 4
    images = {p.name for p in (run_dir / "images").iterdir()}
    assert set(STANDARD) <= images
    # the finished part with the layers the standard set shows at another time, and the findings
    assert {"part_OP10_temperature.png", "part_OP10_coolant_reach.png", "findings.png", "finding_4.png"} <= images
    assert {"finding_1.png", "finding_2.png"} <= images and "finding_0.png" not in images    # info: no own picture
    manifest = json.loads((run_dir / "render.json").read_text())
    captions = {e["file"]: e["caption"] for e in manifest["images"]}
    assert set(captions) == {f"images/{name}" for name in images}
    assert "Where the findings were raised" in captions["images/findings.png"]
    overview = next(e for e in manifest["images"] if e["file"] == "images/findings.png")
    assert overview["findings"] == [4, 1, 2, 0, 3]          # worst first: error, warnings by time, infos by time
    assert manifest["video"]["frames"] == 12 and manifest["video"]["speed"] == 2.0     # 2.32 s at 2x, 10 frames/s
    from cnctwin.render.output import probe_frames
    assert probe_frames(run_dir / "video.mp4") == 12
    with wave.open(str(run_dir / "audio.wav")) as audio:
        assert (audio.getframerate(), audio.getsampwidth(), audio.getnchannels()) == (44100, 2, 1)
    assert json.loads((run_dir / "audio.json").read_text())["format"] == "cnctwin-audio"
    for name in ("report.pdf", "setup_sheet.pdf"):
        assert (run_dir / name).read_bytes()[:5] == b"%PDF-"
    report = (run_dir / "report.html").read_text()
    assert "Where the findings were raised" in report and "Video: video.mp4, 12 frames" in report
    assert "Sound: audio.wav" in report and "NOT SAFE TO RUN" in report
    assert report.index("images/setup.png") < report.index("images/findings.png")   # render.json order
    document = json.loads((run_dir / "run.json").read_text())
    records = document["generators"]
    assert list(records) == list(GENERATORS) and all(r["status"] == "done" and r["reason"] is None
                                                     for r in records.values())
    index = {e["path"]: e for e in document["outputs"]}
    for name, record in records.items():
        for relative in record["files"]:
            entry = index[relative]
            assert entry["sha256"] == sha256(run_dir / relative)
            assert entry["bytes"] == (run_dir / relative).stat().st_size
            if relative != "render.json":
                assert entry["generator"] == name
    on_disk = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()
                     and not any(part.startswith(".") for part in p.relative_to(run_dir).parts)
                     and p.relative_to(run_dir).parts[0] != "checkpoints" and p.name not in ("run.json", "status.json"))
    assert sorted(index) == on_disk                       # the index lists every file, hidden ones excluded
    assert not list(run_dir.glob(".outputs-*-*"))         # the generators' work folders are gone
    assert ("outputs: images", "images (1 of 4)") in phases and phases[-1][0] == "outputs: done"
    for name in GENERATORS:
        record = summary["generators"][name]
        assert record["memory_mode"] == "rlimit" and record["wall_s"] > 0 and record["peak_rss_bytes"] > 0


def test_findings_are_placed_in_the_stock_frame(finalized):
    from cnctwin.render.rundir import RunData
    run_dir, _, _ = finalized
    marked = positioned_findings(RunData(run_dir))
    assert [f["id"] for f in marked] == [4, 1, 2, 0, 3]
    # the synthetic stock frame coincides with W (scene stock.pose_W_S is the identity)
    assert marked[0]["pos_S"].tolist() == pytest.approx([0.01005, 0.021, 0.013])


@pytest.fixture()
def run_copy(finalized, tmp_path):
    run_dir, _, _ = finalized
    target = tmp_path / "00002-copy"
    shutil.copytree(run_dir, target)
    return target


def test_a_failed_generator_does_not_lose_the_others(run_copy):
    before = {name: sha256(run_copy / name) for name in ("video.mp4", "images/findings.png")}
    summary = finalize_outputs(run_copy, {"outputs": SMALL | {"only": ["audio", "report"],
                                                               "memory_bytes": {"audio": "8M"},
                                                               "timeout_s": {"report": 0.3}}})
    assert summary["failed"] == ["audio", "report"]
    audio, report = summary["generators"]["audio"], summary["generators"]["report"]
    assert audio["status"] == "failed" and audio["reason"]
    assert report["status"] == "failed" and report["reason"].startswith("timed out after 0 s")
    for name in ("audio.wav", "audio.json", "report.html", "report.pdf", "setup_sheet.html", "setup_sheet.pdf"):
        assert not (run_copy / name).exists(), name             # partial files of a failed generator are removed
    document = json.loads((run_copy / "run.json").read_text())
    records = document["generators"]
    assert records["audio"] == {"status": "failed", "reason": audio["reason"], "files": []}
    assert records["report"]["status"] == "failed" and records["images"]["status"] == "done"
    assert records["video"]["status"] == "done"
    index = {e["path"] for e in document["outputs"]}
    assert {"video.mp4", "images/findings.png", "render.json"} <= index and "audio.wav" not in index
    assert before == {name: sha256(run_copy / name) for name in before}     # the others are untouched


def test_a_cancel_stops_the_generator(run_copy):
    class Stop(BaseException):
        pass

    def progress(progress=None, **fields):
        if fields.get("phase") == "outputs: report" and ": " in str(fields.get("message")):
            raise Stop()

    with pytest.raises(Stop):
        finalize_outputs(run_copy, {"outputs": SMALL | {"only": ["report"]}}, progress)
    assert not list(run_copy.glob(".outputs-report-*"))
    pattern = f"cnctwin.core.outputs --generate report --run-dir {run_copy}"
    left = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    assert left.stdout.strip() == ""
    summary = finalize_outputs(run_copy, {"outputs": SMALL | {"only": ["audio"]}})     # the lock was released
    assert summary["generators"]["audio"]["status"] == "done"


def test_a_generator_not_asked_for_is_removed(run_copy):
    summary = finalize_outputs(run_copy, {"outputs": SMALL | {"only": ["video"], "video": False}})
    assert summary["generators"]["video"]["status"] == "skipped" and not (run_copy / "video.mp4").exists()
    manifest = json.loads((run_copy / "render.json").read_text())
    assert manifest["video"] is None and manifest["images"]
    records = json.loads((run_copy / "run.json").read_text())["generators"]
    assert records["video"] == {"status": "skipped", "reason": "not asked for (outputs.video: false)", "files": []}
    assert records["images"]["status"] == "done"


def test_settings_come_from_the_project_and_the_options(finalized, tmp_path):
    run_dir, _, _ = finalized
    settings = output_settings(run_dir)
    # run.json of the synthetic run has no project outputs: the project file's are read (video at speed 1)
    assert settings["report"] == ["html", "pdf"] and settings["images"] and settings["audio"] == {"listener": None,
                                                                                                    "sources": None}
    assert settings["video"]["speed"] == 1.0 and settings["video"]["fps"] == 25.0 and settings["only"] is None
    assert settings["memory_bytes"]["images"] == 1610612736 and settings["simulated_s"] == pytest.approx(2.32)
    assert settings["timeout_s"]["video"] == pytest.approx(1800 + 30 * 2.32)
    merged = output_settings(run_dir, {"outputs": {"video": {"speed": 4}, "report": "html", "audio": False,
                                                   "memory_bytes": "1.5G", "only": "report,images"}})
    assert merged["video"]["speed"] == 4.0 and merged["video"]["fps"] == 25.0 and merged["report"] == ["html"]
    assert merged["audio"] is None and merged["memory_bytes"]["audio"] == int(1.5 * 2 ** 30)
    assert merged["only"] == ["images", "report"]
    assert output_settings(run_dir, {"outputs": False})["report"] == []
    for bad, message in (({"colour": 1}, "unknown keys colour"), ({"report": ["docx"]}, "not a report format"),
                         ({"video": {"size": [191, 108]}}, "even width"), ({"memory_mode": "cgroup"}, "memory_mode"),
                         ({"only": ["sound"]}, "not among"), ({"video": {"speed": 0}}, "positive"),
                         ({"timeout_s": {"report": -1}}, "positive"), ({"memory_bytes": "lots"}, "not a size")):
        with pytest.raises(InputError, match=message):
            output_settings(run_dir, {"outputs": bad})
    with pytest.raises(InputError, match="no run.json"):
        finalize_outputs(tmp_path)


def test_command_line(run_copy, capsys, tmp_path):
    assert main([str(run_copy), "--only", "audio", "--memory-mode", "rlimit", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["generators"]["audio"]["status"] == "done" and list(printed["generators"]) == ["audio"]
    assert main([str(tmp_path / "missing")]) == 1 and "not a run directory" in capsys.readouterr().err


def test_the_index_is_rewritten_when_files_change(run_copy):
    (run_copy / "inspection.json").write_text('{"changed": true}\n')
    document = update_index(run_copy)
    entry = next(e for e in document["outputs"] if e["path"] == "inspection.json")
    assert entry["sha256"] == sha256(run_copy / "inspection.json")
    assert not any(e["path"].startswith(".") for e in document["outputs"])
