"""Dataset export (cnctwin/dataset, docs/ml.md "Datasets"): the run-level and per-block tables from run directories
in the real formats (realrun.py writes them with the real channel names through the real Recorder, from an
analytic model whose values the tests predict), sweep folders as sources, the schema, determinism and the command."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import realrun as R
from cnctwin import dataset
from cnctwin.dataset import schema as S

HERE = Path(__file__).resolve().parent


def make_run(base: Path, name: str, *, stickout_mm: float = 32.0, options=None, program=None) -> Path:
    project = R.project(base / f"{name}-project", stickout_mm=stickout_mm, program=program)
    run = base / "runs" / name
    R.write_run(project, run, options or {})
    return run


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(text: str) -> float:
    return float(text) if text != "" else math.nan


def expected(stickout_mm=32.0, options=None):
    blocks = R.blocks_of(R.SLOT, options or {})
    return blocks, [R.physics(b, stickout_mm / 1000.0) for b in blocks]


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("dataset")


@pytest.fixture(scope="module")
def plain(base) -> Path:
    return make_run(base, "plain")


def test_the_run_table_has_the_runs_results(base, plain):
    out = base / "out-plain"
    manifest = dataset.export([plain], out)
    assert manifest["runs"] == 1 and manifest["blocks"] > 5 and not manifest["skipped"]
    assert sorted(manifest["files"]) == sorted(["blocks.csv", "blocks.parquet", "manifest.json", "runs.csv",
                                                "runs.parquet", "schema.json"])
    row = rows(out / "runs.csv")[0]
    blocks, phys = expected()
    run = json.loads((plain / "run.json").read_text())
    assert row["run_id"] == "plain" and row["point"] == "-1" and row["sweep"] == "" and row["safe"] == "1"
    assert num(row["cycle_time_s"]) == pytest.approx(blocks[-1]["tick1"] * 1e-6)
    assert num(row["removed_mass_kg"]) == pytest.approx(run["summary"]["removed_mass_kg"])
    assert num(row["chip_mass_kg"]) == pytest.approx(run["summary"]["removed_mass_kg"])
    assert num(row["cutting_work_J"]) == pytest.approx(1.01 * run["summary"]["shaft_work_J"])
    forces = [p["force"] for p in phys]
    assert num(row["force_max_N"]) == pytest.approx(max(forces), rel=1e-12)
    assert num(row["force_peak_max_N"]) == pytest.approx(1.3 * max(forces), rel=1e-12)
    cut = [(b["tick1"] - b["tick0"], p["force"]) for b, p in zip(blocks, phys) if b["cutting"]]
    mean = sum(w * f for w, f in cut) / sum(w for w, _ in cut)
    assert num(row["force_mean_N"]) == pytest.approx(mean, rel=1e-9)
    assert num(row["tool_deflection_max_m"]) == pytest.approx(max(p["deflection"] for p in phys), rel=1e-12)
    assert num(row["tool_edge_temp_max_K"]) == pytest.approx(max(p["edge"] for p in phys), rel=1e-12)
    assert num(row["chatter_index_max"]) == pytest.approx(max(p["chatter"] for p in phys))
    assert num(row["stability_margin_min"]) == pytest.approx(min(p["margin"] for p in phys if math.isfinite(p["margin"])))
    tools = json.loads((plain / "tools.json").read_text())["tools"][0]
    assert num(row["VB_max_m[T1]"]) == pytest.approx(tools["VB_max_m"]) and row["broken[T1]"] == "0"
    inspection = json.loads((plain / "inspection.json").read_text())["tolerances"][0]
    assert num(row["deviation_m[slot-depth]"]) == pytest.approx(inspection["deviation_m"])
    assert row["pass[slot-depth]"] == ("1" if inspection["pass"] else "0")
    surface = json.loads((plain / "surface.json").read_text())
    assert num(row["Ra_m[slot-floor]"]) == pytest.approx(surface["features"][0]["Ra_m"])
    assert num(row["burr_max_m"]) == pytest.approx(surface["burrs"][0]["height_m"]) and row["burr_count"] == "1"
    assert num(row["face_Ra_max_m"]) == 1.1e-6 and num(row["face_Ra_mean_m"]) == pytest.approx(1.0e-6)
    findings = json.loads((plain / "findings.json").read_text())["findings"]
    assert row["findings_count"] == str(len(findings)) and row["findings[warning]"] == str(len(findings))
    assert row["findings[chatter]"] == str(len(findings)) and row["findings[collision]"] == "0"
    assert num(row["energy_per_part_J"]) == 2.0e4 and num(row["takt_s"]) == pytest.approx(blocks[-1]["tick1"] * 1e-6 + 69)
    assert row["feed_override"] == "1.0" and row["mass_balance_ok"] == "1"


def test_the_block_table_has_features_and_labels(base, plain):
    out = base / "out-plain"
    if not (out / "blocks.csv").is_file():
        dataset.export([plain], out)
    table = rows(out / "blocks.csv")
    blocks, phys = expected()
    assert len(table) == len(blocks)
    findings = json.loads((plain / "findings.json").read_text())["findings"]
    for row, b, p in zip(table, blocks, phys):
        assert int(row["line"]) == b["line"] and row["kind"] == b["kind"] and row["op"] == "OP10"
        assert num(row["duration_s"]) == pytest.approx((b["tick1"] - b["tick0"]) * 1e-6)
        if not b["cutting"]:
            assert math.isnan(num(row["force_mean_N"])) and row["findings_count"] == "0"
            continue
        assert row["tool"] == "1" and row["tool_type"] == "square_endmill" and row["tool_flutes"] == "3"
        assert num(row["spindle_speed_rad_s"]) == pytest.approx(10000 / 60 * 2 * math.pi)
        assert num(row["feed_speed_cut_m_s"]) == pytest.approx(b["v"])
        assert num(row["feed_per_tooth_m"]) == pytest.approx(p["fz"], rel=1e-12)
        assert num(row["cutting_speed_m_s"]) == pytest.approx(p["vc"], rel=1e-12)
        assert num(row["ae_m"]) == R.D and num(row["ap_m"]) == pytest.approx(b["ap"])
        assert num(row["force_mean_N"]) == pytest.approx(p["force"], rel=1e-12)
        assert num(row["torque_max_Nm"]) == pytest.approx(p["torque"], rel=1e-12)
        assert num(row["deflection_max_m"]) == pytest.approx(p["deflection"], rel=1e-12)
        assert num(row["tool_stickout_m"]) == 0.032 and row["material"] == "al6061-t6"
        assert num(row["material_hardness_HB"]) == 95 and num(row["coolant_on"]) == 1.0
        assert num(row["VB_increment_m"]) > 0 and num(row["life_used_increment"]) > 0
        assert math.isfinite(num(row["Ra_m"]))
        expected_findings = [f for f in findings if f["line"] == b["line"]]
        assert row["findings_count"] == str(len(expected_findings))
        assert row["findings_categories"] == ("chatter" if expected_findings else "")
    cutting_rows = [r for r, b in zip(table, blocks) if b["cutting"]]
    assert math.isfinite(num(cutting_rows[-1]["burr_max_m"])) and math.isnan(num(cutting_rows[0]["burr_max_m"]))


def test_overrides_and_stickout_reach_the_rows(base, plain):
    faster = make_run(base, "faster", options={"operator": {"feed_override": {"*": 1.25}}})
    longer = make_run(base, "longer", stickout_mm=40.0)
    out = base / "out-three"
    dataset.export([plain, faster, longer], out)
    runs = {r["run_id"]: r for r in rows(out / "runs.csv")}
    assert runs["faster"]["feed_override"] == "1.25" and runs["plain"]["feed_override"] == "1.0"
    ratio = num(runs["faster"]["force_max_N"]) / num(runs["plain"]["force_max_N"])
    assert ratio == pytest.approx(1.25 ** 0.8, rel=1e-9)
    assert num(runs["faster"]["cycle_time_s"]) < num(runs["plain"]["cycle_time_s"])
    deflection = num(runs["longer"]["tool_deflection_max_m"]) / num(runs["plain"]["tool_deflection_max_m"])
    assert deflection == pytest.approx((40 / 32) ** 3, rel=1e-9)
    blocks = [r for r in rows(out / "blocks.csv") if r["run_id"] == "faster" and r["kind"] == "linear"]
    assert all(r["feed_override"] == "1.25" for r in blocks)
    plain_blocks = [r for r in rows(out / "blocks.csv") if r["run_id"] == "plain" and r["kind"] == "linear"]
    for fast, slow in zip(blocks, plain_blocks):
        assert num(fast["feed_per_tooth_m"]) == pytest.approx(1.25 * num(slow["feed_per_tooth_m"]), rel=1e-12)


def test_the_export_is_deterministic(base, plain, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    dataset.export([plain], first)
    dataset.export([plain], second)
    for name in ("runs.parquet", "runs.csv", "blocks.parquet", "blocks.csv", "schema.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_the_schema_names_every_column_with_unit_and_source(base, plain):
    out = base / "out-plain"
    if not (out / "schema.json").is_file():
        dataset.export([plain], out)
    schema = json.loads((out / "schema.json").read_text())
    assert schema["format"] == S.FORMAT
    for table in ("runs", "blocks"):
        names = [c["name"] for c in schema["tables"][table]]
        with open(out / f"{table}.csv", encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == names
        parquet = pq.read_table(out / f"{table}.parquet")
        assert parquet.column_names == names and parquet.num_rows == len(rows(out / f"{table}.csv"))
        for column in schema["tables"][table]:
            assert column["dtype"] in S.DTYPES and column["unit"] and column["provenance"] and column["doc"]
            field = parquet.schema.field(column["name"])
            assert field.metadata[b"unit"].decode() == column["unit"]
    blocks = pq.read_table(out / "blocks.parquet").to_pydict()
    assert np.isnan(blocks["force_mean_N"][0]) and blocks["tool"][0] == -1


def _sweep_folder(base: Path, runs: dict[int, Path | None], failed: set[int]) -> Path:
    folder = base / "sweep"
    points = []
    for index, run in sorted(runs.items()):
        point = folder / "points" / f"p{index:05d}"
        point.mkdir(parents=True, exist_ok=True)
        values = {"feed": 0.8 + 0.1 * index, "coolant": "emulsion-8pct" if index % 2 else "dry"}
        points.append({"index": index, "values": values})
        status = "failed" if index in failed else "done"
        (point / "result.json").write_text(json.dumps({
            "index": index, "status": status, "key": f"{index:064x}", "run_dir": str(run) if run else None,
            "error": {"message": "ConservationError: mass"} if status == "failed" else None}))
    (folder / "sweep.json").write_text(json.dumps({"format": "cnctwin-sweep-run", "name": "feeds"}))
    (folder / "plan.json").write_text(json.dumps({"format": "cnctwin-sweep-plan", "points": points, "parameters": [
        {"name": "feed", "kind": "override", "target": "feed", "unit": "-", "range": [0.8, 1.2]},
        {"name": "coolant", "kind": "project", "path": "coolant.fluid", "values": ["dry", "emulsion-8pct"]}]}))
    return folder


def test_a_sweep_folder_gives_parameters_and_skips_what_did_not_finish(base, plain, tmp_path):
    longer = base / "runs" / "longer"
    if not longer.is_dir():
        longer = make_run(base, "longer", stickout_mm=40.0)
    folder = _sweep_folder(tmp_path, {0: plain, 1: longer, 2: plain, 3: tmp_path / "evicted"}, failed={2})
    manifest = dataset.export([folder], tmp_path / "out")
    assert manifest["runs"] == 2 and [s["point"] for s in manifest["skipped"]] == [2, 3]
    assert "ConservationError" in manifest["skipped"][0]["reason"] and "gone" in manifest["skipped"][1]["reason"]
    table = rows(tmp_path / "out" / "runs.csv")
    assert [r["point"] for r in table] == ["0", "1"] and table[0]["sweep"] == "feeds"
    assert table[0]["param.feed"] == "0.8" and table[1]["param.coolant"] == "emulsion-8pct"
    assert table[0]["cache_key"] == f"{0:064x}"
    schema = json.loads((tmp_path / "out" / "schema.json").read_text())
    columns = {c["name"]: c for c in schema["tables"]["runs"]}
    assert columns["param.feed"]["dtype"] == "float64" and columns["param.coolant"]["dtype"] == "string"
    blocks = rows(tmp_path / "out" / "blocks.csv")
    assert {r["point"] for r in blocks} == {"0", "1"}


def test_runs_without_series_or_results_give_nan_not_errors(tmp_path):
    run = tmp_path / "runs" / "bare"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "bare", "verdict": {"safe": False}, "summary": {}}))
    manifest = dataset.export([tmp_path / "runs"], tmp_path / "out", formats=("csv",))
    assert manifest["runs"] == 1 and manifest["blocks"] == 0
    row = rows(tmp_path / "out" / "runs.csv")[0]
    assert row["safe"] == "0" and row["complete"] == "-1" and row["force_max_N"] == "" and row["cycle_time_s"] == ""
    assert not (tmp_path / "out" / "runs.parquet").exists()


def test_the_dataset_reads_only_real_channels():
    """Every channel the dataset reads is registered by its domain with that tier and an index of that kind."""
    import importlib
    from cnctwin.dataset.series import CHANNELS
    registry = {}
    for domain in ("motion", "cutting", "dynamics", "thermal", "wear", "removal", "coolant", "holding", "chips"):
        for channel in importlib.import_module(f"cnctwin.{domain}.channels").channels():
            registry[channel.name] = channel
    for name, (tier, index) in CHANNELS.items():
        assert name in registry, f"{name} is not a channel of the twin"
        assert registry[name].tier.value == tier, name
        assert bool(registry[name].max_shape) == (index is not None), name
        assert (R.SHAPES[name][1] == tuple(registry[name].max_shape)), f"realrun.SHAPES[{name}] differs from the twin"


def test_twin_dataset_command(base, plain, tmp_path, capsys):
    assert dataset.main([str(plain), "--out", str(tmp_path / "o"), "--format", "csv", "--no-blocks"]) == 0
    assert "1 runs" in capsys.readouterr().out
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == ["manifest.json", "runs.csv", "schema.json"]
    assert dataset.main([str(plain), "--out", str(tmp_path / "o"), "--format", "xlsx"]) == 2
    assert dataset.main([str(tmp_path / "nothing"), "--out", str(tmp_path / "o")]) == 1
    assert dataset.main([str(plain), "--out", str(tmp_path / "j"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 1


def test_docs_ml_lists_every_fixed_column():
    text = (Path(__file__).resolve().parents[3] / "docs" / "ml.md").read_text(encoding="utf-8")
    for column in S.RUN_IDENTITY + S.RUN_OPTIONS + S.RUN_RESULTS + S.RUN_SEVERITY + S.BLOCK_COLUMNS:
        assert f"| `{column.name}` | {column.dtype} | {column.unit} |" in text, column.name
