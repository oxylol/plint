"""Sweeps (cnctwin/sweep, docs/ml.md "Sweeps", docs/formats.md "Sweep"): the spec, the designs, the point variants
(project key paths, program templates, generators, operator overrides), execution inline and in point processes,
resuming from the cache, failures, cancellation, the command and the API's job entry. The simulation is
tests/unit/dataset/realrun.py (real formats, analytic values); the build id is pinned for in-process runs."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
DATASET_TESTS = ROOT / "tests" / "unit" / "dataset"
if str(DATASET_TESTS) not in sys.path:
    sys.path.insert(0, str(DATASET_TESTS))

import realrun as R  # noqa: E402
from cnctwin import cache as C  # noqa: E402
from cnctwin.core.errors import InputError  # noqa: E402
from cnctwin.core.warmup import mill_project  # noqa: E402
from cnctwin.sweep import design as D  # noqa: E402
from cnctwin.sweep import spec as S  # noqa: E402
from cnctwin.sweep.execute import SweepRun, read_sweep  # noqa: E402
from cnctwin.sweep.variants import VariantWriter, nc_number  # noqa: E402

RUNNER = f"{DATASET_TESTS / 'realrun.py'}:run_project"
FAILING = f"{Path(__file__).with_name('fake_points.py')}:failing_run_project"
SLOW = f"{Path(__file__).with_name('fake_points.py')}:slow_run_project"


@pytest.fixture(scope="module")
def mill_dir(tmp_path_factory) -> Path:
    return mill_project(tmp_path_factory.mktemp("mill") / "mill").parent


SLOT = """%
O09001 (SWEEP TEST: SHORT SLOT)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T1 M06
G90 G54 G00 X-8. Y10. S10000 M03
G43 H01 Z25. M08
Z2.
G01 Z-1.5 F1000.
G01 X6. F1600.
G00 Z25.
M09
M05
G53 G49 Z0.
M30
%
"""


@pytest.fixture
def base(tmp_path, mill_dir) -> Path:
    """A copy of the tiny mill project with this module's own program (independent of cnctwin.core.warmup's)."""
    target = tmp_path / "base"
    shutil.copytree(mill_dir, target)
    (target / "warmup.nc").write_text(SLOT, encoding="utf-8")
    return target


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(C, "build_id", lambda package_dir=None: "b" * 64)


def spec_dict(base: Path, **extra) -> dict:
    data = {"format": "cnctwin-sweep", "version": 1, "name": "feeds and stickout", "base": str(base), "seed": 3,
            "design": {"kind": "grid"},
            "parameters": [{"name": "feed", "kind": "override", "target": "feed", "values": [0.8, 1.2]},
                           {"name": "stickout", "kind": "project", "path": "tools.list[number=1].stickout_mm",
                            "values": [30.0, 40.0], "unit": "mm"}],
            "execution": {"isolation": "inline", "runner": RUNNER}}
    data.update(extra)
    return data


# ----------------------------------------------------------------------------- the spec

def test_a_spec_loads_with_defaults(base, tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec_dict(Path("base"))), encoding="utf-8")
    spec = S.load_spec(path)
    assert spec.name == "feeds and stickout" and spec.base_project == (tmp_path / "base" / "project.yaml").resolve()
    assert spec.options == {"outputs": False} and spec.execution.workers == 1 and spec.execution.runner == RUNNER
    assert spec.parameter("stickout").unit == "mm" and len(spec.sha256) == 64
    described = spec.describe()
    assert described["parameters"][0] == {"name": "feed", "kind": "override", "unit": "-", "doc": "",
                                          "values": [0.8, 1.2], "target": "feed"}


def test_every_problem_of_a_spec_is_listed(base):
    bad = spec_dict(base, design={"kind": "lhs"}, colour="red", options={"operator": {}},
                    execution={"isolation": "inline", "workers": 2})
    bad["parameters"] += [{"name": "feed", "kind": "override", "target": "torque", "range": [2, 1]},
                          {"name": "x", "kind": "project", "path": "tools.list[=1]", "values": [1]},
                          {"name": "g", "kind": "generator", "generator": "none", "argument": "A", "values": [1]},
                          {"name": "t", "kind": "program_template", "range": [1, 2], "values": [1]}]
    with pytest.raises(InputError) as error:
        S.load_spec(bad)
    text = str(error.value)
    for expected in ("colour: unknown key", "design.points", "options.operator", "named twice", "target: one of",
                     "range: [lo, hi] with lo < hi", "tools.list[=1]", "generator: a generator", "give range",
                     "several workers need isolation: process"):
        assert expected in text, expected
    with pytest.raises(InputError, match="a grid needs values"):
        S.load_spec(spec_dict(base, parameters=[{"name": "a", "kind": "override", "target": "feed", "range": [1, 2]}]))
    with pytest.raises(InputError, match="no value for stickout"):
        S.load_spec(spec_dict(base, design={"kind": "list", "list": [{"feed": 1.0}]}))


def test_key_paths_select_items_by_key():
    data = {"tools": {"list": [{"number": 1, "stickout_mm": 30}, {"number": 2, "stickout_mm": 31}]},
            "operations": [{"name": "OP10", "setup": {"workholding": [{"input_torque_Nm": 60}]}}]}
    S.set_key_path(data, "tools.list[number=2].stickout_mm", 45)
    S.set_key_path(data, "operations[name=OP10].setup.workholding[0].input_torque_Nm", 80)
    S.set_key_path(data, "coolant_extra", {"a": 1}) if False else None
    assert data["tools"]["list"][1]["stickout_mm"] == 45 and data["operations"][0]["setup"]["workholding"][0][
        "input_torque_Nm"] == 80
    assert S.parse_key_path("a.b[2].c[k=v]") == ["a", "b", 2, "c", ("k", "v")]
    with pytest.raises(InputError, match="exactly one"):
        S.set_key_path(data, "tools.list[number=9].stickout_mm", 1)
    with pytest.raises(InputError, match="has no"):
        S.set_key_path(data, "nothing.deep", 1)


# ----------------------------------------------------------------------------- designs

def test_grid_list_and_value_maps(base):
    spec = S.load_spec(spec_dict(base))
    points, units, _ = D.design_points(spec)
    assert units is None and points == [{"feed": 0.8, "stickout": 30.0}, {"feed": 0.8, "stickout": 40.0},
                                        {"feed": 1.2, "stickout": 30.0}, {"feed": 1.2, "stickout": 40.0}]
    p = S.Parameter("x", "project", range=(1.0, 100.0), scale="log", levels=3)
    assert D.levels_of(p) == [1.0, 10.000000000000002, 100.0] or D.levels_of(p)[1] == pytest.approx(10.0)
    q = S.Parameter("n", "project", range=(0.0, 10.0), step=2.5, levels=5)
    assert D.levels_of(q) == [0.0, 2.5, 5.0, 7.5, 10.0] and D.value_of(q, 0.31) == 2.5
    r = S.Parameter("i", "project", range=(1.0, 4.0), integer=True)
    assert D.value_of(r, 0.0) == 1 and D.value_of(r, 0.999) == 4 and isinstance(D.value_of(r, 0.5), int)
    c = S.Parameter("c", "project", values=["a", "b", "c"])
    assert [D.value_of(c, u) for u in (0.0, 0.34, 0.99)] == ["a", "b", "c"]
    listed = S.load_spec(spec_dict(base, design={"kind": "list", "list": [{"feed": 1.1}, {"feed": 0.9, "stickout": 33}]},
                                   parameters=[{"name": "feed", "kind": "override", "target": "feed", "range": [0.5, 1.5]},
                                               {"name": "stickout", "kind": "project", "path": "tools.list[0].stickout_mm",
                                                "range": [28, 40], "default": 32}]))
    assert D.design_points(listed)[0] == [{"feed": 1.1, "stickout": 32}, {"feed": 0.9, "stickout": 33}]


def test_stochastic_designs_are_seeded_and_space_filling(base):
    params = [{"name": "feed", "kind": "override", "target": "feed", "range": [0.5, 1.5]},
              {"name": "stickout", "kind": "project", "path": "tools.list[0].stickout_mm", "range": [28, 40]}]
    lhs = S.load_spec(spec_dict(base, design={"kind": "lhs", "points": 16}, parameters=params))
    points, units, notes = D.design_points(lhs)
    assert units.shape == (16, 2) and not notes
    for j in range(2):                                   # one point in each of the 16 strata of every factor
        assert sorted(np.floor(units[:, j] * 16).astype(int)) == list(range(16))
    again, units2, _ = D.design_points(lhs)
    assert again == points and np.array_equal(units, units2)
    other = S.load_spec(spec_dict(base, design={"kind": "lhs", "points": 16}, parameters=params, seed=4))
    assert D.design_points(other)[0] != points
    sobol = S.load_spec(spec_dict(base, design={"kind": "sobol", "points": 8}, parameters=params))
    points, units, notes = D.design_points(sobol)
    assert units.shape == (8, 2) and not notes and all(0.5 <= p["feed"] <= 1.5 for p in points)
    for j in range(2):                                   # a scrambled Sobol' net of 8 points: one per eighth
        assert sorted(np.floor(units[:, j] * 8).astype(int)) == list(range(8))
    odd = S.load_spec(spec_dict(base, design={"kind": "sobol", "points": 6}, parameters=params))
    assert "powers of two" in D.design_points(odd)[2][0]
    rnd = S.load_spec(spec_dict(base, design={"kind": "random", "points": 5}, parameters=params))
    assert D.design_points(rnd)[0] == D.design_points(rnd)[0]
    cd = S.load_spec(spec_dict(base, design={"kind": "lhs", "points": 8, "optimization": "random-cd"}, parameters=params))
    assert len(D.design_points(cd)[0]) == 8


# ----------------------------------------------------------------------------- variants

def test_project_parameters_and_overrides_make_the_variants(base, tmp_path):
    spec = S.load_spec(spec_dict(base))
    writer = VariantWriter(spec, tmp_path / "sweep")
    plan = writer.write(0, {"feed": 1.2, "stickout": 40.0})
    data = yaml.safe_load(plan.project.read_text())
    assert data["tools"]["list"][0]["stickout_mm"] == 40.0 and data["name"].endswith("[stickout=40]")
    assert Path(data["operations"][0]["program"]).is_absolute()
    assert plan.options == {"outputs": False, "operator": {"feed_override": {"*": 1.2}}}
    unit = writer.write(1, {"feed": 1.0, "stickout": 40.0})
    assert unit.options == {"outputs": False}                                   # 100 % is the plain run
    same = VariantWriter(spec, tmp_path / "other").write(7, {"feed": 0.8, "stickout": 40.0})
    from cnctwin.project import project_hash, resolve
    assert project_hash(resolve(same.project)) == project_hash(resolve(plan.project))
    with pytest.raises(InputError, match="outside the machine's override range"):
        writer.write(2, {"feed": 2.5, "stickout": 40.0})
    tool = S.load_spec(spec_dict(base, parameters=[{"name": "s", "kind": "override", "target": "spindle", "tool": 7,
                                                    "values": [0.9]}]))
    with pytest.raises(InputError, match="no cutting tool 7"):
        VariantWriter(tool, tmp_path / "t").write(0, {"s": 0.9})
    rapid = S.load_spec(spec_dict(base, parameters=[{"name": "r", "kind": "override", "target": "rapid",
                                                     "values": [0.3, 0.25]}]))
    with pytest.raises(InputError, match="not a rapid key"):
        VariantWriter(rapid, tmp_path / "r").write(0, {"r": 0.3})
    assert VariantWriter(rapid, tmp_path / "r").write(1, {"r": 0.25}).options["operator"] == {"rapid_override": 0.25}
    bad = S.load_spec(spec_dict(base, parameters=[{"name": "s", "kind": "project", "path": "stock.size_mm",
                                                   "values": ["big"]}]))
    with pytest.raises(InputError, match="does not validate"):
        VariantWriter(bad, tmp_path / "b").write(0, {"s": "big"})


def test_program_templates_replace_their_placeholders(base, tmp_path):
    program = (base / "warmup.nc").read_text()
    (base / "slot.tpl.nc").write_text(program.replace("G01 X6. F1600.", "G01 X{{XEND}} F{{F1}} (S {{S1}})"))
    spec = S.load_spec(spec_dict(base, placeholders={"S1": 10000}, parameters=[
        {"name": "F1", "kind": "program_template", "template": "slot.tpl.nc", "range": [800, 2400], "levels": 2,
         "format": "{:.0f}."},
        {"name": "x_end", "kind": "program_template", "placeholder": "XEND", "template": "slot.tpl.nc",
         "values": [6.0, 8]}]))
    plan = VariantWriter(spec, tmp_path / "s").write(3, {"F1": 2400.0, "x_end": 8})
    text = (plan.project.parent / "warmup.nc").read_text()
    assert "G01 X8 F2400. (S 10000)" in text and "{{" not in text
    data = yaml.safe_load(plan.project.read_text())
    assert data["operations"][0]["program"] == "warmup.nc" and plan.programs == {"OP10": "warmup.nc"}
    assert nc_number(6.0) == "6." and nc_number(0.25) == "0.25" and nc_number(-0.0) == "0." and nc_number(3) == "3"
    (base / "open.tpl.nc").write_text(program.replace("F1600.", "F{{F1}} S{{UNBOUND}}"))
    with pytest.raises(InputError, match="UNBOUND"):
        VariantWriter(S.load_spec(spec_dict(base, parameters=[
            {"name": "F1", "kind": "program_template", "template": "open.tpl.nc", "values": [1000]}])), tmp_path / "o")
    with pytest.raises(InputError, match="has no"):
        VariantWriter(S.load_spec(spec_dict(base, placeholders={"S1": 1, "XEND": 6.0}, parameters=[
            {"name": "F1", "kind": "program_template", "template": "slot.tpl.nc", "values": [1000]},
            {"name": "F2", "kind": "program_template", "template": "slot.tpl.nc", "values": [1000]}])), tmp_path / "p")


GENERATOR = '''
"""A make_program.py style generator: constants at the top, main() writes the program beside itself."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FEED = 1600.0
DEPTH = 1.5
FEED_PLUNGE = round(FEED * 0.625)          # derived from FEED: follows the swept value
PROGRAM = """%
O09001 (GENERATED SLOT)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T1 M06
G90 G54 G00 X-8. Y10. S10000 M03
G43 H01 Z25. M08
Z2.
G01 Z-{depth} F{plunge}.
G01 X6. F{feed}
G00 Z25.
M09
M05
G53 G49 Z0.
M30
%
"""


def main():
    text = PROGRAM.format(depth=DEPTH, plunge=FEED_PLUNGE, feed=FEED)
    (HERE / "warmup.nc").write_text(text)
    print("wrote", HERE / "warmup.nc")


def make(FEED=1600.0, DEPTH=1.5):
    return PROGRAM.format(depth=DEPTH, plunge=round(FEED * 0.625), feed=FEED)
'''


def test_generators_are_called_with_the_points_values(base, tmp_path):
    (base / "gen.py").write_text(GENERATOR)
    spec = S.load_spec(spec_dict(base, generators={"slot": {"script": "gen.py", "output": "warmup.nc"},
                                                   "slot_kw": {"script": "gen.py", "function": "make", "call": "kwargs"}},
                                 parameters=[{"name": "feed_mm_min", "kind": "generator", "generator": "slot",
                                              "argument": "FEED", "values": [1200.0, 2000.0]}]))
    writer = VariantWriter(spec, tmp_path / "g")
    plan = writer.write(0, {"feed_mm_min": 2000.0})
    text = (plan.project.parent / "warmup.nc").read_text()
    assert "G01 X6. F2000.0" in text and "F1250." in text and "Z-1.5" in text
    record = json.loads((plan.project.parent / ".generated-slot.json").read_text())
    assert "wrote" in record["log"] and record["values"] == {"FEED": 2000.0}
    assert not list(plan.project.parent.glob(".generator-*")) and not (base / "warmup.nc").read_text().count("2000")
    stamp = (plan.project.parent / "warmup.nc").stat().st_mtime_ns
    writer.write(0, {"feed_mm_min": 2000.0})                                 # unchanged request: not run again
    assert (plan.project.parent / "warmup.nc").stat().st_mtime_ns == stamp
    kwargs = S.load_spec(spec_dict(base, generators={"slot": {"script": "gen.py", "function": "make", "call": "kwargs"}},
                                   parameters=[{"name": "depth", "kind": "generator", "generator": "slot",
                                                "argument": "DEPTH", "values": [0.75]}]))
    plan = VariantWriter(kwargs, tmp_path / "k").write(0, {"depth": 0.75})
    assert "G01 Z-0.75 F1000." in (plan.project.parent / "warmup.nc").read_text()
    missing = S.load_spec(spec_dict(base, generators={"slot": {"script": "gen.py"}},
                                    parameters=[{"name": "x", "kind": "generator", "generator": "slot",
                                                 "argument": "NOT_THERE", "values": [1]}]))
    with pytest.raises(InputError, match="no top-level assignment of NOT_THERE"):
        VariantWriter(missing, tmp_path / "m").write(0, {"x": 1})


# ----------------------------------------------------------------------------- running

def test_an_inline_sweep_runs_resumes_and_exports(base, tmp_path, pinned, monkeypatch):
    calls = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_RUNNER_LOG", str(calls))
    cache = C.RunCache(tmp_path / "cache")
    spec = S.load_spec(spec_dict(base))
    events = []
    summary = SweepRun(spec, tmp_path / "sweep", cache=cache, progress_cb=lambda p=None, **f: events.append(
        dict(p or {}, **f))).run()
    assert summary["points"] == 4 and summary["done"] == 4 and summary["cached"] == 0 and summary["failed"] == 0
    assert summary["safe"] + summary["unsafe"] == 4 and len(cache.entries()) == 4
    assert events[-1]["done"] == 4 and events[-1]["total"] == 4 and any(e.get("phase") == "plan" for e in events)
    result = json.loads((tmp_path / "sweep" / "points" / "p00003" / "result.json").read_text())
    assert result["status"] == "done" and result["values"] == {"feed": 1.2, "stickout": 40.0} and result["registered"]
    run = json.loads((Path(result["run_dir"]) / "run.json").read_text())
    assert run["options"] == {"outputs": False, "operator": {"feed_override": {"*": 1.2}}}
    again = SweepRun(spec, tmp_path / "sweep", cache=cache).run()
    assert again["cached"] == 4 and again["done"] == 0
    assert json.loads((tmp_path / "sweep" / "summary.json").read_text())["cached"] == 4
    shown = read_sweep(tmp_path / "sweep")
    assert [p["status"] for p in shown["points"]] == ["cached"] * 4 and shown["status"]["status"] == "done"
    from cnctwin import dataset
    manifest = dataset.export([tmp_path / "sweep"], tmp_path / "data")
    assert manifest["runs"] == 4
    import csv
    with open(tmp_path / "data" / "runs.csv", newline="") as handle:
        table = list(csv.DictReader(handle))
    force = {(float(r["param.feed"]), float(r["param.stickout"])): float(r["force_max_N"]) for r in table}
    assert force[(1.2, 30.0)] / force[(0.8, 30.0)] == pytest.approx(1.5 ** 0.8, rel=1e-9)
    deflection = {float(r["param.stickout"]): float(r["tool_deflection_max_m"]) for r in table if r["param.feed"] == "0.8"}
    assert deflection[40.0] / deflection[30.0] == pytest.approx((40 / 30) ** 3, rel=1e-9)


def test_failed_points_are_recorded_and_retried_only_when_asked(base, tmp_path, pinned):
    cache = C.RunCache(tmp_path / "cache")
    data = spec_dict(base, execution={"isolation": "inline", "runner": FAILING})
    spec = S.load_spec(data)
    summary = SweepRun(spec, tmp_path / "sweep", cache=cache).run()
    assert summary["failed"] == 2 and summary["done"] == 2                       # stick-out 40 fails in this fake
    failed = json.loads((tmp_path / "sweep" / "points" / "p00001" / "result.json").read_text())
    assert failed["status"] == "failed" and "synthetic failure" in failed["error"]["message"]
    again = SweepRun(spec, tmp_path / "sweep", cache=cache).run()
    assert again["failed"] == 2 and again["cached"] == 2
    notes = json.loads((tmp_path / "sweep" / "points" / "p00001" / "result.json").read_text())["notes"]
    assert any("not run again" in n for n in notes)
    retried = SweepRun(spec, tmp_path / "sweep", cache=cache, retry_failed=True).run()
    assert retried["failed"] == 2 and retried["cached"] == 2


def test_point_processes_run_in_parallel_with_a_memory_cap(base, tmp_path):
    cache = C.RunCache(tmp_path / "cache")
    spec = S.load_spec(spec_dict(base, execution={"isolation": "process", "workers": 2, "memory_mode": "rlimit",
                                                  "memory": "1200M", "runner": RUNNER}))
    build = C.build_id()
    summary = SweepRun(spec, tmp_path / "sweep", cache=cache).run()
    assert summary["failed"] == 0 and summary["done"] + summary["cached"] == 4, summary
    for i in range(4):
        folder = tmp_path / "sweep" / "points" / f"p{i:05d}"
        outcome = json.loads((folder / "outcome.json").read_text())
        assert outcome["status"] == "done" and (Path(outcome["outcome"]["run_dir"]) / "run.json").is_file()
        assert json.loads((folder / "job.json").read_text())["memory_mode"] == "rlimit"
    again = SweepRun(spec, tmp_path / "sweep", cache=cache).run()
    if C.build_id() == build:                                     # the tree may change under other work
        assert again["cached"] == 4


def test_a_cancel_stops_the_running_points(base, tmp_path, monkeypatch):
    monkeypatch.setenv("CNCTWIN_TEST_DELAY_S", "60")
    spec = S.load_spec(spec_dict(base, execution={"isolation": "process", "memory_mode": "none", "runner": SLOW}))

    class Stop(BaseException):
        pass

    def progress(progress=None, **fields):
        if (progress or {}).get("running"):
            raise Stop()

    run = SweepRun(spec, tmp_path / "sweep", cache=C.RunCache(tmp_path / "cache"), progress_cb=progress)
    with pytest.raises(Stop):
        run.run()
    status = json.loads((tmp_path / "sweep" / "status.json").read_text())
    assert status["status"] == "cancelled" and not run.children
    log = (tmp_path / "sweep" / "points" / "p00000" / "log.txt")
    assert not (tmp_path / "sweep" / "points" / "p00000" / "outcome.json").exists() and log.is_file()


def test_the_sweep_command(base, tmp_path, pinned, monkeypatch, capsys):
    from cnctwin import sweep
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec_dict(base)), encoding="utf-8")
    assert sweep.main([str(path), "--plan-only", "--out", str(tmp_path / "s")]) == 0
    assert "planned 4 points" in capsys.readouterr().out
    assert sweep.main([str(path), "--out", str(tmp_path / "s"), "--quiet", "--dataset"]) == 0
    out = capsys.readouterr().out
    assert "4 points: 0 cached, 4 simulated, 0 failed" in out and "dataset:" in out
    assert (tmp_path / "s" / "dataset" / "runs.parquet").is_file()
    assert (tmp_path / "data" / "cache" / "index.jsonl").is_file()                 # DATA_DIR/cache by default
    assert sweep.main(["--show", str(tmp_path / "s")]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["done"] == 4
    assert sweep.main([str(tmp_path / "nothing.yaml")]) == 1
    assert sweep.main([]) == 2


def test_the_api_job_entry(base, tmp_path, pinned, monkeypatch):
    from cnctwin.sweep import run_sweep_job
    monkeypatch.setenv("CNCTWIN_CACHE_DIR", str(tmp_path / "cache"))
    spec = spec_dict(Path("ignored"))
    spec.pop("base")
    result = run_sweep_job({"project_path": str(base / "project.yaml"), "spec": spec,
                            "options": {"isolation": "inline"}}, tmp_path / "out", None)
    assert result["done"] == 4 and result["dataset"]["runs"] == 4 and result["sweep_dir"] == str(tmp_path / "out")
    assert (tmp_path / "out" / "dataset" / "blocks.csv").is_file()
    with pytest.raises(InputError, match="names no project"):
        run_sweep_job({"spec": spec}, tmp_path / "x", None)


def test_generated_points_resolve_and_run(base, tmp_path, pinned):
    (base / "gen.py").write_text(GENERATOR)
    spec = S.load_spec(spec_dict(base, generators={"slot": {"script": "gen.py", "output": "warmup.nc"}},
                                 parameters=[{"name": "depth", "kind": "generator", "generator": "slot",
                                              "argument": "DEPTH", "values": [0.5, 1.0]}]))
    summary = SweepRun(spec, tmp_path / "sweep", cache=C.RunCache(tmp_path / "cache")).run()
    assert summary["done"] == 2
    removed = []
    for i in range(2):
        result = json.loads((tmp_path / "sweep" / "points" / f"p{i:05d}" / "result.json").read_text())
        removed.append(json.loads((Path(result["run_dir"]) / "run.json").read_text())["summary"]["removed_volume_m3"])
    assert removed[1] == pytest.approx(2 * removed[0], rel=0.05)
