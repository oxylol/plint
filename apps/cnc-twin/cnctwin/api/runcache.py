"""The run cache seen from the server (docs/api.md "Run cache"; cnctwin.cache), and the project hash of a run id.

``POST /api/runs`` looks the run up before it queues a simulation: its key is computed for the job the server
would start (the simulate entry as the runner, the jobs' deterministic environment and SOURCE_DATE_EPOCH), in
``DATA_DIR/cache`` (the folder the jobs register into). A verified run in the server's runs folder is answered
as it is; one elsewhere (a sweep's run in the cache's own store, a ``twin run`` into another folder) is copied
into a new run folder first (and the copy registered, so the next request finds it directly). Anything that
goes wrong here is a miss with a note: the cache never stops a run from being queued.

Run folders are named as ``twin run`` names them (cnctwin.core.rundir.run_id: ``<nnnnn>-<slug>-<h8>``), with the
project hash the key holds, or :func:`project_hash` when no key was computed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def cache_root(settings) -> Path:
    return Path(settings.data_dir) / "cache"


def project_hash(project_file) -> str:
    """The project hash of a run id when the cache computed no key: the resolved project's
    (cnctwin.project.project_hash, as ``twin run`` computes it; the server's memoised database), else, for a project
    that does not resolve, the hash of its migrated data alone, else of its file's bytes (the simulation reports why
    it does not resolve)."""
    from ..cache import database
    from ..project import load_project, resolve
    from ..project import project_hash as hash_of
    try:
        written = load_project(project_file)
    except Exception:       # noqa: BLE001  (an unreadable project still gets a run folder: the job says why)
        try:
            return hashlib.sha256(Path(project_file).read_bytes()).hexdigest()
        except OSError:
            return hashlib.sha256(str(project_file).encode("utf-8")).hexdigest()
    try:
        return resolve(written, database()).hash
    except Exception:       # noqa: BLE001
        return hash_of(written)


def lookup(ctx, project, options: dict) -> tuple[str | None, dict, object]:
    """(id of a finished run with the request's key that the API serves, or None; what the cache said; the
    cnctwin.cache.CacheKey, or None when none was computed)."""
    from ..cache import RunCache, copy_run, enabled, environment_contract, key_for
    from .safety import slugify
    if not enabled():
        return None, {"hit": False, "key": None, "note": "the run cache is off (CNCTWIN_CACHE=off)"}, None
    try:
        key = key_for(project.file, options, runner=ctx.settings.entries["simulate"],
                      environment=environment_contract(ctx.settings.source_date_epoch))
    except Exception as error:      # a project that does not resolve: the simulation reports why
        return None, {"hit": False, "key": None, "note": f"no cache key: {type(error).__name__}: {error}"}, None
    try:
        store = RunCache(cache_root(ctx.settings))
        hit = store.lookup(key.key, inside=ctx.settings.runs_dir)
        if hit is not None:
            return Path(hit.run_dir).name, {"hit": True, "key": key.key, "run_id": Path(hit.run_dir).name}, key
        with store.try_lock(key.key) as held:      # eviction leaves a locked key alone while it is copied
            if not held:                           # a sweep computes it now: never wait for it in a request
                return None, {"hit": False, "key": key.key, "note": "another process is computing this run now"}, key
            other = store.lookup(key.key)
            if other is not None:
                run_id, run_dir = ctx.runs.allocate(slugify(project.id), key.components["project_hash"], options)
                copy_run(other.run_dir, run_dir)
                store.register(key, run_dir, owned=False)
                return run_id, {"hit": True, "key": key.key, "run_id": run_id, "copied_from": other.run_id}, key
    except Exception as error:      # a broken cache costs a simulation, never the request
        return None, {"hit": False, "key": key.key, "note": f"cache lookup failed: {type(error).__name__}: {error}"}, key
    return None, {"hit": False, "key": key.key}, key


def enabled() -> bool:
    from ..cache import enabled as cache_enabled
    return cache_enabled()


__all__ = ["lookup", "cache_root", "enabled", "project_hash"]
