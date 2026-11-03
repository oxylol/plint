"""Request bodies and documented response shapes (OpenAPI, docs/openapi.json).

Run files are passed through as their writers wrote them (``dict``), so response models list
the fields the API adds or guarantees and allow the rest (``extra="allow"``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorResponse(Open):
    detail: str = Field(description="What went wrong, in words")


# ----------------------------------------------------------------------------- requests

class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"project_id": "example-pocket", "options": {}}]})
    project_id: str = Field(description="A project id from GET /api/projects")
    options: dict[str, Any] = Field(default_factory=dict,
                                    description="Passed unchanged to cnctwin.core.loop.run_project")


class OptionsBody(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"options": {}}]})
    options: dict[str, Any] = Field(default_factory=dict, description="Passed unchanged to the job's entry point")


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"project_id": "example-pocket", "run_id": "00001-example-pocket",
                      "machine_data": {"kind": "csv", "path": "spindle_load.csv"}, "options": {}}]})
    project_id: str | None = Field(default=None, description="Project whose machine is calibrated")
    run_id: str | None = Field(default=None, description="Simulated run to compare the machine data with")
    machine_data: dict[str, Any] | None = Field(default=None, description="Recorded machine data "
                                                "({kind: mtconnect|opcua|focas|csv, path, mapping}); "
                                                "paths are relative to the project folder")
    options: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------------- responses

class Health(Open):
    status: Literal["ok"]
    version: str
    format_version: int
    data_dir: str
    memory_mode: str
    jobs: dict[str, Any]
    entry_points: dict[str, Any]
    series_reader: str
    db_backend: str
    numba_cache_key: str


class Operation(Open):
    name: str | None
    program: str | None
    program_found: bool


class ProjectSummary(Open):
    id: str
    source: Literal["example", "upload"]
    read_only: bool
    name: str
    description: str | None
    machine: str | None
    material: str | None
    operations: list[Operation]
    tools: int
    error: str | None = Field(description="Why project.yaml could not be read, or null")


class FileEntry(Open):
    path: str
    bytes: int


class ProjectDetail(ProjectSummary):
    project: dict[str, Any] | None = Field(description="project.yaml as written (plain YAML data)")
    files: list[FileEntry]


class ProjectUpload(ProjectDetail):
    skipped: list[str] = Field(description="Archive members dropped (macOS metadata)")
    problems: list[str] = Field(description="Things to fix before a run (programs not in the archive ...)")


class JobRecord(Open):
    id: str
    kind: Literal["simulate", "check", "compare", "uncertainty", "calibrate"]
    state: Literal["queued", "running", "cancelling", "done", "failed", "cancelled"]
    pool: Literal["heavy", "light"]
    run_id: str | None
    project_id: str | None
    queue_position: int | None
    attempt: int
    created_s: float
    started_s: float | None
    finished_s: float | None
    progress: dict[str, Any]
    error: dict[str, Any] | None
    has_result: bool
    memory_mode: str
    memory_bytes: int
    links: dict[str, str]


class JobResponse(Open):
    job: JobRecord
    result: Any = Field(default=None, description="The entry point's result once done")


class RunStatus(Open):
    status: str = Field(description="queued | running | cancelling | done | failed | cancelled | unknown")
    progress: dict[str, Any] = Field(description="sim_time_s, est_total_s, wall_s, eta_s (volatile)")


class RunRecord(Open):
    id: str
    run: dict[str, Any] | None = Field(description="run.json (null until the run has finished)")
    status: RunStatus
    job: JobRecord | None
    checkpoint: dict[str, Any] | None = Field(description="Latest committed checkpoint, or null")
    links: dict[str, str]


class RunSummary(Open):
    id: str
    status: str
    progress: dict[str, Any]
    project_id: str | None
    name: str | None
    verdict: dict[str, Any] | None
    summary: dict[str, Any] | None
    job_id: str | None


class RunList(Open):
    total: int
    runs: list[RunSummary]


class FindingsResponse(Open):
    run_id: str
    total: int
    count: int
    worst: str | None
    safe: bool
    findings: list[dict[str, Any]]


class TraceResponse(Open):
    event: dict[str, Any]
    chain: list[dict[str, Any]] = Field(description="The event and its ancestors, breadth first")


class SeriesItem(Open):
    column: str
    channel: str
    unit: str
    n: int
    t_s: list[float | None]
    v: list[float | None]


class SeriesResponse(Open):
    run_id: str
    tier: Literal["fast", "tick", "macro"]
    tick_s: float
    t0_s: float | None
    t1_s: float | None
    rows: int = Field(description="Samples in the time range before downsampling")
    max_points: int
    downsampled: bool
    method: Literal["raw", "m4"]
    buckets: int | None
    series: list[SeriesItem]


class FramesJson(Open):
    run_id: str
    count: int
    stride: int
    tick_s: float
    header: dict[str, Any]
    fields: dict[str, Any] = Field(description="Field name -> values per frame (column-wise)")


class KeyframeList(Open):
    count: int
    keyframes: list[dict[str, Any]]


class ArrayDoc(Open):
    dtype: str
    shape: list[int]
    data: Any


class ArraysResponse(Open):
    k: int
    time_s: float | None
    file: str
    arrays: dict[str, ArrayDoc]


class JetsResponse(Open):
    k: int | None
    time_s: float | None
    jets: list[dict[str, Any]]
    nozzles: list[dict[str, Any]]


class FileList(Open):
    count: int
    files: list[FileEntry]


class DbKinds(Open):
    backend: str
    kinds: list[dict[str, Any]]


class DbList(Open):
    backend: str
    kind: str
    count: int
    items: list[dict[str, Any]]


class DbItem(Open):
    backend: str
    kind: str
    id: str


BINARY = {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}
GLB = {"model/gltf-binary": {"schema": {"type": "string", "format": "binary"}}}
SSE = {"text/event-stream": {"schema": {"type": "string"},
                             "example": "event: progress\nid: f0k0\ndata: {\"status\":\"running\",...}\n\n"}}


def errors(*codes: int) -> dict:
    return {code: {"model": ErrorResponse} for code in codes}
