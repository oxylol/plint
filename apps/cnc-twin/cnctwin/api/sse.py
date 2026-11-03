"""Server-Sent Events (text/event-stream) for runs and jobs, by tailing their files.

A run's stream tails ``status.json`` (event ``progress`` whenever it changes), ``findings.json``
(event ``finding`` per new finding, in id order) and ``keyframes/index.jsonl`` (event ``keyframe``
per new keyframe), and ends with ``done`` once the status is done, failed or cancelled (after a
last read of findings and keyframes, which the run writes before its final status).

Every event carries ``id: f<findings sent>k<keyframes sent>``; a client reconnecting with
``Last-Event-ID`` (EventSource does this by itself) gets only what it has not seen. Findings and
keyframes are keyed by their own ids, so after a resume (which rewrites what came after the
checkpoint, identically) a client may see a finding again with the same id. A comment line
(``: keep-alive``) goes out after ``sse_heartbeat_s`` of silence, so proxies keep the connection.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator, Callable

import anyio

from .jsonio import dumps
from .safety import ApiError

TERMINAL = ("done", "failed", "cancelled")
_LAST_ID = re.compile(r"^f(\d+)k(\d+)$")
RETRY_MS = 2000


def format_event(event: str, data, event_id: str | None = None) -> str:
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    for line in dumps(data).splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def parse_last_event_id(value: str | None) -> tuple[int, int]:
    if not value:
        return 0, 0
    match = _LAST_ID.match(value.strip())
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


async def _read(function: Callable):
    return await anyio.to_thread.run_sync(function)


async def run_events(run, *, poll_s: float, heartbeat_s: float, last_event_id: str | None = None,
                     timeout_s: float | None = None, job_status: Callable | None = None) -> AsyncIterator[str]:
    findings_sent, keyframes_sent = parse_last_event_id(last_event_id)
    yield f"retry: {RETRY_MS}\n\n"
    last_status = None
    started = time.monotonic()
    last_output = started

    def ident() -> str:
        return f"f{findings_sent}k{keyframes_sent}"

    def read_status() -> dict:
        status = run.status()
        return job_status(status) if job_status is not None else status

    while True:
        status = await _read(read_status)        # in a thread: it may take the job manager's lock
        out: list[str] = []
        key = json.dumps(status, sort_keys=True, default=str)
        if key != last_status:
            last_status = key
            out.append(format_event("progress", status, ident()))
        try:
            findings = await _read(run.findings)
        except ApiError:
            findings = None              # unreadable for a moment: try again next round
        if findings is not None:
            if len(findings) < findings_sent:
                findings_sent = len(findings)
            for finding in findings[findings_sent:]:
                findings_sent += 1
                out.append(format_event("finding", finding, ident()))
        try:
            keyframes = await _read(run.keyframes)
        except ValueError:
            keyframes = None
        if keyframes is not None:
            if len(keyframes) < keyframes_sent:
                keyframes_sent = len(keyframes)
            for keyframe in keyframes[keyframes_sent:]:
                keyframes_sent += 1
                out.append(format_event("keyframe", keyframe, ident()))
        state = status.get("status")
        if state in TERMINAL and (findings is None or keyframes is None):
            state = None                 # finish only after a clean last read
        if state in TERMINAL:
            run_json = await _read(run.run_json)
            verdict = run_json.get("verdict") if isinstance(run_json, dict) else None
            out.append(format_event("done", {"status": state, "verdict": verdict, "error": status.get("error"),
                                             "findings": findings_sent, "keyframes": keyframes_sent}, ident()))
            for chunk in out:
                yield chunk
            return
        now = time.monotonic()
        if out:
            last_output = now
            for chunk in out:
                yield chunk
        elif now - last_output >= heartbeat_s:
            last_output = now
            yield ": keep-alive\n\n"
        if timeout_s is not None and now - started >= timeout_s:
            yield ": timeout (reconnect to continue)\n\n"
            return
        await asyncio.sleep(poll_s)


async def job_events(manager, job_id: str, public: Callable, *, poll_s: float, heartbeat_s: float,
                     timeout_s: float | None = None) -> AsyncIterator[str]:
    """Stream of a job: ``progress`` when its record changes, ``done`` when it ends."""
    yield f"retry: {RETRY_MS}\n\n"
    last = None
    started = time.monotonic()
    last_output = started
    while True:
        job = await _read(lambda: manager.get(job_id))
        if job is None:
            yield format_event("done", {"status": "unknown", "error": {"message": f"no job {job_id!r}"}})
            return
        record = await _read(lambda: public(job))
        key = json.dumps(record, sort_keys=True, default=str)
        now = time.monotonic()
        if key != last:
            last = key
            last_output = now
            yield format_event("progress", record)
        if job.state in TERMINAL:
            yield format_event("done", {"status": job.state, "error": job.error})
            return
        if now - last_output >= heartbeat_s:
            last_output = now
            yield ": keep-alive\n\n"
        if timeout_s is not None and now - started >= timeout_s:
            yield ": timeout (reconnect to continue)\n\n"
            return
        await asyncio.sleep(poll_s)
