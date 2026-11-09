"""The run cache seen from the server (docs/api.md "Run cache"; cnctwin.cache).

``POST /api/runs`` looks the run up before it queues a simulation: its key is computed for the job the server
would start (the simulate entry as the runner, the jobs' deterministic environment and SOURCE_DATE_EPOCH), in
``DATA_DIR/cache`` (the folder the jobs register into). A verified run in the server's runs folder is answered
as it is; one elsewhere (a sweep's run in the cache's own store, a ``twin run`` into another folder) is copied
into a new run folder first (and the copy registered, so the next request finds it directly). Anything that
goes wrong here is a miss with a note: the cache never stops a run from being queued.
"""

from __future__ import annotations

from pathlib import Path


def cache_root(settings) -> Path:
    return Path(settings.data_dir) / "cache"


def lookup(ctx, project, options: dict) -> tuple[str | None, dict]:
    """(id of a finished run with the request's key that the API serves, or None; what the cache said)."""
    from ..cache import RunCache, copy_run, enabled, environment_contract, key_for
    from .safety import slugify
    if not enabled():
        return None, {"hit": False, "key": None, "note": "the run cache is off (CNCTWIN_CACHE=off)"}
    try:
        key = key_for(project.file, options, runner=ctx.settings.entries["simulate"],
                      environment=environment_contract(ctx.settings.source_date_epoch))
    except Exception as error:      # a project that does not resolve: the simulation reports why
        return None, {"hit": False, "key": None, "note": f"no cache key: {type(error).__name__}: {error}"}
    try:
        store = RunCache(cache_root(ctx.settings))
        hit = store.lookup(key.key, inside=ctx.settings.runs_dir)
        if hit is not None:
            return Path(hit.run_dir).name, {"hit": True, "key": key.key, "run_id": Path(hit.run_dir).name}
        with store.try_lock(key.key) as held:      # eviction leaves a locked key alone while it is copied
            if not held:                           # a sweep computes it now: never wait for it in a request
                return None, {"hit": False, "key": key.key, "note": "another process is computing this run now"}
            other = store.lookup(key.key)
            if other is not None:
                run_id, run_dir = ctx.runs.allocate(slugify(project.id))
                copy_run(other.run_dir, run_dir)
                store.register(key, run_dir, owned=False)
                return run_id, {"hit": True, "key": key.key, "run_id": run_id, "copied_from": other.run_id}
    except Exception as error:      # a broken cache costs a simulation, never the request
        return None, {"hit": False, "key": key.key, "note": f"cache lookup failed: {type(error).__name__}: {error}"}
    return None, {"hit": False, "key": key.key}


def enabled() -> bool:
    from ..cache import enabled as cache_enabled
    return cache_enabled()


__all__ = ["lookup", "cache_root", "enabled"]
