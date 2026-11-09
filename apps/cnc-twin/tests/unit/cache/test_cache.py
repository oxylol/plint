"""The run cache (cnctwin/cache.py, docs/ml.md "Run cache"): the key, reuse only of verified runs, the index,
eviction, the per-key lock and twin cache. The simulation is a fake (fake_runner.py); the source tree's build id
is pinned in most tests, because other work on the tree must not make them flaky (its own test is separate)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from cnctwin import cache as C
from cnctwin.core.errors import InputError
from cnctwin.core.warmup import mill_project

HERE = Path(__file__).resolve().parent
FAKE = HERE / "fake_runner.py"
RUNNER = f"{FAKE}:run_project"


@pytest.fixture(scope="module")
def project_dir(tmp_path_factory) -> Path:
    return mill_project(tmp_path_factory.mktemp("mill") / "mill").parent


@pytest.fixture
def project(tmp_path, project_dir) -> Path:
    """A private copy of the tiny mill project (tests edit its program)."""
    import shutil
    target = tmp_path / "mill"
    shutil.copytree(project_dir, target)
    return target / "project.yaml"


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(C, "build_id", lambda package_dir=None: "b" * 64)


@pytest.fixture
def calls(tmp_path, monkeypatch) -> Path:
    log = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_RUNNER_LOG", str(log))
    return log


def count(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.is_file() else 0


def test_build_id_is_rundirs_and_follows_the_tree(tmp_path):
    from cnctwin.core import rundir
    package = tmp_path / "pkg" / "cnctwin"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n")
    (package / "sub" / "a.py").write_text("y = 2\n")
    first = C.build_id(package)
    rundir._BUILD.pop(str(package), None)
    assert first == rundir.build_id(package)
    time.sleep(0.01)
    (package / "sub" / "a.py").write_text("y = 3\n")
    assert C.build_id(package) != first
    (package / "sub" / "a.py").write_text("y = 2\n")
    assert C.build_id(package) == first


def test_the_key_names_what_decides_the_results(project, pinned):
    base = C.key_for(project, {}, runner=RUNNER)
    assert len(base.key) == 64 and base.key == C.key_for(project, {}, runner=RUNNER).key
    assert C.key_for(project, {"resume_from": "/x.json", "no_cache": True}, runner=RUNNER).key == base.key
    assert C.key_for(project, {"max_sim_s": 0.5}, runner=RUNNER).key != base.key
    assert C.key_for(project, {}, runner=None).key != base.key
    assert C.key_for(project, {}, runner=RUNNER, environment=C.environment_contract("1")).key != base.key
    components = base.components
    assert components["project_hash"] == base.project.hash and components["build"] == "b" * 64
    assert components["environment"]["PYTHONHASHSEED"] == "0" and components["platform"]["cpu"] is not None
    assert components["runner"].startswith(RUNNER + "@")
    program = project.parent / "warmup.nc"
    program.write_text(program.read_text() + "(CHANGED)\n")
    assert C.key_for(project, {}, runner=RUNNER).key != base.key
    assert C.canonical_runner("cnctwin.sweep.runner:run_project", {}) == C.DEFAULT_RUNNER
    assert C.canonical_runner("cnctwin.sweep.runner:run_project", {"operator": {"feed_override": 1.1}}) \
        == "cnctwin.sweep.runner:run_project"


def test_a_project_that_does_not_resolve_has_no_key(tmp_path):
    bad = tmp_path / "bad" / "project.yaml"
    bad.parent.mkdir()
    bad.write_text("format: cnctwin-project\n")
    with pytest.raises(InputError):
        C.key_for(bad, {})


def test_a_verified_run_is_reused(tmp_path, project, pinned, calls):
    cache = C.RunCache(tmp_path / "cache")
    events = []
    first = C.run_cached(project, {}, progress_cb=lambda p=None, **f: events.append(dict(p or {}, **f)),
                         cache=cache, runner=RUNNER)
    assert not first["cached"] and first["registered"] and count(calls) == 1
    assert Path(first["run_dir"]).parent == cache.runs_dir and cache.owns(first["run_dir"])
    second = C.run_cached(project, {}, cache=cache, runner=RUNNER, progress_cb=lambda p=None, **f: events.append(f))
    assert second["cached"] and second["run_dir"] == first["run_dir"] and count(calls) == 1
    assert second["key"] == first["key"] and any(e.get("phase") == "cached" for e in events)
    ops = [line["op"] for line in cache.lines()]
    assert ops == ["format", "put", "hit"]
    entry = cache.entries()[0]
    assert entry.hits == 1 and entry.owned and entry.bytes > 0 and entry.files >= 8
    other = C.run_cached(project, {"max_sim_s": 1.0}, cache=cache, runner=RUNNER)
    assert not other["cached"] and count(calls) == 2


def test_a_changed_result_set_is_never_reused(tmp_path, project, pinned, calls):
    cache = C.RunCache(tmp_path / "cache")
    first = C.run_cached(project, {}, cache=cache, runner=RUNNER)
    run = Path(first["run_dir"])
    findings = json.loads((run / "findings.json").read_text())
    findings["findings"].append({"id": 0, "category": "collision", "severity": "fatal"})
    (run / "findings.json").write_text(json.dumps(findings))
    second = C.run_cached(project, {}, cache=cache, runner=RUNNER)
    assert not second["cached"] and second["run_dir"] != first["run_dir"] and count(calls) == 2
    drops = [line for line in cache.lines() if line["op"] == "drop"]
    assert len(drops) == 1 and "the result set changed" in drops[0]["reason"]
    (Path(second["run_dir"]) / "run.json").unlink()
    assert cache.lookup(second["key"]) is None


def test_replay_parts_leave_the_run_valid(tmp_path, project, pinned, calls):
    cache = C.RunCache(tmp_path / "cache")
    first = C.run_cached(project, {}, cache=cache, runner=RUNNER)
    run = Path(first["run_dir"])
    (run / "series" / "fast").mkdir(parents=True, exist_ok=True)
    (run / "series" / "fast" / "replay-0-1000.parquet").write_bytes(b"replayed rows")
    with open(run / "series" / "index.jsonl", "a") as handle:
        handle.write(json.dumps({"tier": "fast", "file": "series/fast/replay-0-1000.parquet", "replay": True}) + "\n")
    assert C.run_cached(project, {}, cache=cache, runner=RUNNER)["cached"] and count(calls) == 1


def test_no_cache_runs_again_and_the_new_run_supersedes(tmp_path, project, pinned, calls):
    cache = C.RunCache(tmp_path / "cache")
    first = C.run_cached(project, {}, cache=cache, runner=RUNNER)
    second = C.run_cached(project, {}, cache=cache, runner=RUNNER, no_cache=True)
    assert not second["cached"] and second["registered"] and count(calls) == 2
    assert "no_cache" in " ".join(second["notes"])
    third = C.run_cached(project, {}, cache=cache, runner=RUNNER)
    assert third["cached"] and third["run_dir"] == second["run_dir"] != first["run_dir"]


def test_inputs_changed_during_the_run_are_not_registered(tmp_path, project, pinned, calls):
    cache = C.RunCache(tmp_path / "cache")
    result = C.run_cached(project, {}, cache=cache, runner=f"{FAKE}:edit_program_run_project")
    assert not result["registered"] and "changed while it ran" in " ".join(result["notes"])
    assert cache.entries() == []


def test_a_run_of_another_build_is_refused(tmp_path, project, monkeypatch, calls):
    monkeypatch.setattr(C, "build_id", lambda package_dir=None: "b" * 64)
    cache = C.RunCache(tmp_path / "cache")
    key = C.key_for(project, {}, runner=RUNNER)
    run = tmp_path / "run"
    import fake_runner
    fake_runner.run_project(project, run, {}, None)
    document = json.loads((run / "run.json").read_text())
    document["versions"]["build"] = "c" * 64
    (run / "run.json").write_text(json.dumps(document))
    with pytest.raises(InputError, match="names build"):
        cache.register(key, run)
    with pytest.raises(InputError, match="no run.json"):
        cache.register(key, tmp_path)


def test_the_environment_contract_decides_caching(tmp_path, project, pinned, calls, monkeypatch):
    cache = C.RunCache(tmp_path / "cache")
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    result = C.run_cached(project, {}, cache=cache, runner=RUNNER)
    assert not result["registered"] and "deterministic environment" in result["notes"][0]
    monkeypatch.setenv("CNCTWIN_CACHE", "off")
    assert not C.enabled()


def test_a_hit_is_copied_into_a_given_run_directory(tmp_path, project, pinned, calls):
    cache = C.RunCache(tmp_path / "cache")
    first = C.run_cached(project, {}, cache=cache, runner=RUNNER)
    target = tmp_path / "api-runs" / "00007-mill"
    target.mkdir(parents=True)
    (target / "status.json").write_text('{"status": "queued"}')
    copy = C.run_cached(project, {}, cache=cache, runner=RUNNER, run_dir=target)
    assert copy["cached"] and copy["run_dir"] == str(target) and copy["source_run_dir"] == first["run_dir"]
    assert json.loads((target / "status.json").read_text()) == {"status": "queued"}
    document = json.loads((target / "run.json").read_text())
    assert document["run_id"] == "00007-mill" and document["cache"]["source_run_dir"] == first["run_dir"]
    assert C.run_digest(target)[0] == C.run_digest(first["run_dir"])[0] and count(calls) == 1


def test_torn_lines_are_skipped_and_terminated(tmp_path):
    cache = C.RunCache(tmp_path / "cache")
    cache._append([{"op": "hit", "key": "k", "run_dir": "/r", "at_s": 1.0}])
    with open(cache.index_path, "ab") as handle:
        handle.write(b'{"op": "put", "key": "torn')
    assert [line["op"] for line in cache.lines()] == ["format", "hit"]
    cache._append([{"op": "drop", "key": "k", "run_dir": "/r", "reason": "x", "at_s": 2.0}])
    assert [line["op"] for line in cache.lines()] == ["format", "hit", "drop"]
    raw = cache.index_path.read_bytes()
    assert raw.endswith(b"\n") and b'"torn\n' in raw


def _owned_run(cache: C.RunCache, name: str, size: int, key: str, used_s: float) -> None:
    run = cache.runs_dir / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": name}))
    (run / "frames.bin").write_bytes(b"x" * size)
    digest, files, result_bytes = C.run_digest(run)
    cache._append([{"op": "put", "key": key, "run_dir": str(run), "run_id": name, "owned": True, "digest": digest,
                    "files": files, "result_bytes": result_bytes, "bytes": C.directory_bytes(run), "build": None,
                    "project_hash": None, "at_s": used_s, "components": {}}])


def test_eviction_by_age_then_least_recently_used(tmp_path):
    cache = C.RunCache(tmp_path / "cache", max_bytes=2500, max_age_s=1000.0)
    now = 10_000.0
    _owned_run(cache, "old", 1000, "1" * 64, now - 2000.0)       # older than the maximum age
    _owned_run(cache, "a", 1000, "2" * 64, now - 300.0)
    _owned_run(cache, "b", 1000, "3" * 64, now - 200.0)
    _owned_run(cache, "c", 1000, "4" * 64, now - 100.0)
    elsewhere = tmp_path / "runs" / "ref"
    elsewhere.mkdir(parents=True)
    (elsewhere / "run.json").write_text("{}")
    cache._append([{"op": "put", "key": "5" * 64, "run_dir": str(elsewhere), "run_id": "ref", "owned": False,
                    "digest": C.run_digest(elsewhere)[0], "files": 0, "result_bytes": 0, "bytes": 10 ** 9,
                    "at_s": 0.0, "components": {}}])
    planned = cache.evict(now=now, dry_run=True)
    assert [e.run_id for e in planned] == ["old", "a"] and (cache.runs_dir / "old").is_dir()
    with cache.lock("3" * 64):
        evicted = cache.evict(now=now, max_bytes=1100)
    assert [e.run_id for e in evicted] == ["old", "a"]               # b is in use, c fits under the limit alone
    assert not (cache.runs_dir / "old").exists() and not (cache.runs_dir / "a").exists()
    assert (cache.runs_dir / "b").is_dir() and elsewhere.is_dir()
    assert {e.run_id for e in cache.entries()} == {"b", "c", "ref"}
    evicted = cache.evict(now=now, max_bytes=1100)
    assert [e.run_id for e in evicted] == ["b"]                        # the least recently used of b and c


def test_compact_keeps_the_state(tmp_path):
    cache = C.RunCache(tmp_path / "cache")
    _owned_run(cache, "a", 10, "2" * 64, 5.0)
    _owned_run(cache, "b", 10, "3" * 64, 6.0)
    for _ in range(5):
        cache.lookup("2" * 64)
    cache.drop(cache.entries("3" * 64)[0], "test")
    before = {(e.key, e.run_dir): (e.hits, e.used_s) for e in cache.entries()}
    assert cache.compact() == 2                                       # the format line and one put
    after = {(e.key, e.run_dir): (e.hits, e.used_s) for e in cache.entries()}
    assert after == before and len(cache.lines()) == 2


def test_one_key_is_computed_once(tmp_path, project, pinned, calls, monkeypatch):
    monkeypatch.setenv("CNCTWIN_TEST_DELAY_S", "0.6")
    cache = C.RunCache(tmp_path / "cache")
    results = []

    def request():
        results.append(C.run_cached(project, {}, cache=cache, runner=f"{FAKE}:slow_run_project"))

    threads = [threading.Thread(target=request) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert count(calls) == 1 and sorted(r["cached"] for r in results) == [False, True]
    assert results[0]["run_dir"] == results[1]["run_dir"]


def test_resumes_are_not_cached(tmp_path, project, pinned, calls):
    cache = C.RunCache(tmp_path / "cache")
    run = tmp_path / "run"
    result = C.run_cached(project, {"resume_from": str(tmp_path / "ck.json")}, cache=cache, runner=RUNNER, run_dir=run)
    assert not result["registered"] and result["run_dir"] == str(run) and "not cached" in result["notes"][0]


def test_the_synthetic_run_is_cached_whole(tmp_path, project, pinned, calls):
    """A complete synthetic run directory (keyframes, checkpoints, scene, series): registered, verified, reused."""
    cache = C.RunCache(tmp_path / "cache")
    runner = f"{FAKE}:synthetic_run_project"
    first = C.run_cached(project, {}, cache=cache, runner=runner, runs_dir=tmp_path / "runs")
    assert first["registered"] and not cache.entries()[0].owned                 # outside the cache: a reference
    entry = cache.entries()[0]
    assert entry.files > 20 and (Path(first["run_dir"]) / "keyframes" / "bricks.idx").is_file()
    assert C.run_cached(project, {}, cache=cache, runner=runner, runs_dir=tmp_path / "runs")["cached"]
    assert count(calls) == 1
    assert cache.verify(entry, "quick") == (True, "")
    with open(Path(first["run_dir"]) / "trace.jsonl", "a") as handle:
        handle.write("{}\n")
    assert cache.verify(entry, "quick")[0] is False and "bytes" in cache.verify(entry, "quick")[1]
    assert cache.verify(entry, "full")[0] is False
    assert cache.evict(max_bytes=0) == []                                       # references are never deleted


def test_twin_cache_command(tmp_path, project, pinned, calls, capsys):
    root = tmp_path / "cache"
    C.run_cached(project, {}, cache=C.RunCache(root), runner=RUNNER)
    assert C.main(["--root", str(root), "list"]) == 0
    assert "1 entries" in capsys.readouterr().out
    assert C.main(["--root", str(root), "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["owned"] is True
    assert C.main(["--root", str(root), "key", str(project), "--runner", RUNNER]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out[0]) == 64 and out[1].startswith("cached: ")
    assert C.main(["--root", str(root), "verify"]) == 0
    assert "1 entries verify, 0 dropped" in capsys.readouterr().out
    assert C.main(["--root", str(root), "evict", "--max-gb", "0", "--dry-run"]) == 0
    assert "1 would be evicted" in capsys.readouterr().out
    assert C.main(["--root", str(root), "compact"]) == 0
    assert C.main(["--root", str(root), "evict", "--max-gb", "0"]) == 0
    assert C.RunCache(root).entries() == []
    assert C.main(["--root", str(root), "key", str(tmp_path / "nothing")]) == 1
