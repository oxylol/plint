"""production.json from a run directory (the synthetic run of tests/fixtures/make_synthetic_run.py)."""

import json
import sys
from pathlib import Path

import pytest

from cnctwin.production import production_from_run_dir, read_frames, read_timeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

from make_synthetic_run import PRODUCTION_PROJECT, TICK_S, make_synthetic_run  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-run")
    manifest = make_synthetic_run(directory, seed=5, media=False)
    return directory, manifest


def test_breakdown_is_exact_and_energy_plausible(run):
    directory, manifest = run
    document = json.loads((directory / "production.json").read_text())
    part = document["per_part"][0]
    timeline = read_timeline(directory)
    assert all(b.duration_ticks == sum(t for _, t in b.ticks) for b in timeline.blocks)
    assert part["machine_ticks"] == manifest["end_tick"] == part["breakdown"]["total_ticks"]
    assert document["cycle_time_s"] == pytest.approx(manifest["end_tick"] * TICK_S)
    assert part["breakdown"]["ticks"]["cutting"] > 0 and part["breakdown"]["ticks"]["rapid"] > 0
    lines = {line["line"] for line in document["first_part"]["lines"]}
    assert {b.line for b in timeline.blocks} == lines
    omega = abs(manifest["omega_rad_s"])
    spindle = part["energy"]["spindle"]
    assert spindle["accel_J"] == pytest.approx(0.5 * 0.05 * omega ** 2, rel=1e-4)   # J omega^2 / 2 at run-up
    assert spindle["shaft_J"] > 0 and part["energy"]["axes"]["electrical_J"] > 0
    assert part["energy"]["total_J"] == pytest.approx(sum(document["batch"]["energy_by_consumer_J"].values()))
    assert part["tools"][0]["cost"]["cost_per_part_eur"] > 0
    for key in ("parts", "cycle_time_s", "cut_time_s", "air_time_s", "tool_change_time_s", "operator_s",
                "per_part_s", "removed_volume_m3", "removed_mass_kg", "energy_J"):
        assert key in document


def test_recomposing_gives_the_same_document(run):
    directory, _ = run
    written = json.loads((directory / "production.json").read_text())
    again = production_from_run_dir(directory, project=PRODUCTION_PROJECT, tick_s=TICK_S,
                                    removed_mass_kg=written["removed_mass_kg"], write=False)
    written.pop("removed_volume_m3")
    assert json.loads(json.dumps(again)) == written


def test_frames_are_read_with_their_header(run):
    directory, manifest = run
    frames = read_frames(directory)
    assert frames["axis_names"] == ("X", "Y", "Z") and frames["axis"].shape[1] == 3
    assert frames["tick"][0] == 0 and frames["tick"][-1] < manifest["end_tick"]
