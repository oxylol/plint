"""Server settings: where the data lives, how many jobs run at once and how they are capped.

Every value has an environment variable (read by :meth:`Settings.from_env`) and the command line
of ``cnctwin.api.serve`` overrides host and port. Defaults suit this box (4 cores, 7.7 GB shared
with other services): one simulation at a time, capped at 1.5 GiB (architecture 10 budgets the
simulation process at 1.2 GB for the reference part).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .. import paths

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5210
MiB = 1024 * 1024
GiB = 1024 * MiB

# Job kinds: which pool they run in and the callable the worker calls ("module:attr" or
# "/path/file.py:attr"). Overridable with CNCTWIN_ENTRY_<KIND>. "outputs" is not a job kind: it is what the
# worker of a simulation calls after a successful run (the report, pictures, video and sound,
# cnctwin.core.outputs.finalize_outputs(run_dir, options, progress_cb)); CNCTWIN_ENTRY_OUTPUTS=none turns it off.
JOB_POOLS = {"simulate": "heavy", "uncertainty": "heavy", "calibrate": "heavy", "check": "light", "compare": "light"}
DEFAULT_ENTRIES = {
    "simulate": "cnctwin.core.loop:run_project",
    "check": "cnctwin.project.check:check_project",
    "compare": "cnctwin.report.compare:compare_runs",
    "uncertainty": "cnctwin.uncertainty:run_study",
    "calibrate": "cnctwin.mdata.calibrate:calibrate",
    "outputs": "cnctwin.core.outputs:finalize_outputs",
}
NO_ENTRY = ("", "none", "off")      # an entry set to one of these is switched off
MEMORY_MODES = ("auto", "systemd", "rlimit", "none")
# A fixed timestamp for every job (reports, PDF, video): 1980-01-01, the earliest a zip can hold.
DEFAULT_SOURCE_DATE_EPOCH = "315532800"

_SIZE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)i?B?\s*$", re.IGNORECASE)
_SCALE = {"": 1, "K": 1024, "M": MiB, "G": GiB, "T": 1024 * GiB}


def parse_bytes(text: str | int) -> int:
    """``"1536M"``, ``"1.5G"``, ``"900MiB"``, ``"512K"`` or a plain byte count."""
    if isinstance(text, int):
        return text
    match = _SIZE.match(str(text))
    if not match:
        raise ValueError(f"not a size: {text!r} (use e.g. 1536M or 1.5G)")
    return int(float(match.group(1)) * _SCALE[match.group(2).upper()])


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    root: Path = paths.ROOT
    examples_dir: Path = paths.EXAMPLES
    web_dir: Path = paths.WEB
    db_dir: Path = paths.DB
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_heavy_jobs: int = 1                   # simulations, uncertainty studies, calibrations
    max_light_jobs: int = 1                   # checks, comparisons
    heavy_memory_bytes: int = 1536 * MiB
    light_memory_bytes: int = 1024 * MiB
    swap_max_bytes: int = 256 * MiB           # systemd mode: MemorySwapMax (as bin/tsafe)
    memory_mode: str = "auto"                 # auto | systemd | rlimit | none
    cancel_grace_s: float = 60.0              # after a cancel request, SIGTERM if the job has not stopped
    kill_grace_s: float = 10.0                # after SIGTERM, SIGKILL
    poll_s: float = 0.2                       # job manager loop period
    max_upload_bytes: int = 256 * MiB
    max_extract_bytes: int = 1 * GiB
    max_extract_files: int = 20000
    max_compression_ratio: float = 200.0      # zip members above 1 MiB may not expand more than this
    sse_poll_s: float = 0.25
    sse_heartbeat_s: float = 15.0
    status_interval_s: float = 0.2            # worker: minimum time between status.json writes
    python: str = sys.executable
    source_date_epoch: str = DEFAULT_SOURCE_DATE_EPOCH
    entries: dict = field(default_factory=lambda: dict(DEFAULT_ENTRIES))
    cancel_on_shutdown: bool = False          # stop running jobs when the server stops (tests)
    log_level: str = "info"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def memory_bytes(self, pool: str) -> int:
        return self.heavy_memory_bytes if pool == "heavy" else self.light_memory_bytes

    def max_jobs(self, pool: str) -> int:
        return self.max_heavy_jobs if pool == "heavy" else self.max_light_jobs

    def with_(self, **changes) -> "Settings":
        return replace(self, **changes)

    @classmethod
    def from_env(cls, environ=None, **overrides) -> "Settings":
        env = os.environ if environ is None else environ
        values: dict = {"data_dir": Path(env.get("DATA_DIR") or paths.data_dir())}
        if env.get("PORT") and not env.get("CNCTWIN_PORT"):
            try:
                values["port"] = int(env["PORT"])         # the container convention (Dockerfile sets PORT)
            except ValueError:
                raise ValueError(f"PORT: not a port number: {env['PORT']!r}") from None
        simple = {
            "CNCTWIN_HOST": ("host", str), "CNCTWIN_PORT": ("port", int),
            "CNCTWIN_MAX_JOBS": ("max_heavy_jobs", int), "CNCTWIN_MAX_LIGHT_JOBS": ("max_light_jobs", int),
            "CNCTWIN_JOB_MEMORY": ("heavy_memory_bytes", parse_bytes),
            "CNCTWIN_LIGHT_JOB_MEMORY": ("light_memory_bytes", parse_bytes),
            "CNCTWIN_JOB_SWAP": ("swap_max_bytes", parse_bytes),
            "CNCTWIN_JOB_MEMORY_MODE": ("memory_mode", str),
            "CNCTWIN_CANCEL_GRACE_S": ("cancel_grace_s", float), "CNCTWIN_KILL_GRACE_S": ("kill_grace_s", float),
            "CNCTWIN_MAX_UPLOAD_MB": ("max_upload_bytes", lambda v: int(float(v) * MiB)),
            "CNCTWIN_MAX_EXTRACT_MB": ("max_extract_bytes", lambda v: int(float(v) * MiB)),
            "CNCTWIN_EXAMPLES_DIR": ("examples_dir", Path), "CNCTWIN_WEB_DIR": ("web_dir", Path),
            "CNCTWIN_DB_DIR": ("db_dir", Path), "CNCTWIN_PYTHON": ("python", str),
            "SOURCE_DATE_EPOCH": ("source_date_epoch", str), "CNCTWIN_LOG_LEVEL": ("log_level", str),
        }
        for name, (key, convert) in simple.items():
            if env.get(name):
                try:
                    values[key] = convert(env[name])
                except ValueError as error:
                    raise ValueError(f"{name}: {error}") from None
        entries = dict(DEFAULT_ENTRIES)
        for kind in DEFAULT_ENTRIES:
            override = env.get(f"CNCTWIN_ENTRY_{kind.upper()}")
            if override:
                entries[kind] = override
        values["entries"] = entries
        values.update(overrides)
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory mode {self.memory_mode!r}: use one of {', '.join(MEMORY_MODES)}")
        if self.max_heavy_jobs < 1 or self.max_light_jobs < 1:
            raise ValueError("at least one job of each pool must be allowed to run")
        for kind in JOB_POOLS:
            if kind not in self.entries:
                raise ValueError(f"no entry point for job kind {kind!r}")

    def describe(self) -> dict:
        """JSON view (for /api/health)."""
        out = {}
        for item in fields(self):
            value = getattr(self, item.name)
            out[item.name] = str(value) if isinstance(value, Path) else value
        return out
