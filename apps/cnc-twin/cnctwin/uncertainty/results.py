"""What one variant run says, read from its run directory (the same reader as the report)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..report.rundir import RunData
from ..report.sections_part import limits


@dataclass
class SampleResult:
    index: int
    ok: bool
    error: str | None = None
    run_dir: str = ""
    safe: bool | None = None
    dimensions: dict[str, dict] = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)
    scalars: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"index": self.index, "ok": self.ok, "error": self.error, "run_dir": self.run_dir, "safe": self.safe,
                "dimensions": self.dimensions, "findings": self.findings, "scalars": self.scalars}

    @classmethod
    def from_dict(cls, record: dict) -> "SampleResult":
        return cls(**{k: record.get(k) for k in ("index", "ok", "error", "run_dir", "safe")},
                   dimensions=dict(record.get("dimensions") or {}), findings=list(record.get("findings") or []),
                   scalars=dict(record.get("scalars") or {}))


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def read_sample(run_dir, index: int) -> SampleResult:
    run = RunData(run_dir)
    if not run.finished:
        return SampleResult(index, False, "the run has no run.json (it did not finish)", str(run_dir))
    result = SampleResult(index, True, None, str(Path(run_dir)), bool(run.verdict.get("safe")))
    for item in (run.inspection or {}).get("tolerances", []):
        if not isinstance(item, dict) or "measured_m" not in item:
            continue
        nominal, lower, upper = limits(item)
        one_sided = nominal is None          # a tolerance zone (flatness, position ...): 0 is a natural bound
        result.dimensions[str(item.get("id"))] = {
            "kind": str(item.get("kind", "")), "nominal_m": nominal, "lower_m": None if one_sided else lower,
            "upper_m": upper, "one_sided": one_sided, "measured_m": _number(item.get("measured_m")),
            "deviation_m": _number(item.get("deviation_m")),
            "pass": item.get("pass") if isinstance(item.get("pass"), bool) else None}
    for finding in run.findings:
        result.findings.append({"category": str(finding.get("category", "")), "severity": str(finding.get("severity", "")),
                                "op": str(finding.get("op", "")), "program": Path(str(finding.get("program", ""))).name,
                                "line": int(finding.get("line") or 0)})
    scalars = {"cycle_time_s": run.cycle_time_s, "removed_mass_kg": _number(run.summary.get("removed_mass_kg"))}
    record = ((run.production or {}).get("per_part") or [{}])[0]
    scalars["energy_per_part_J"] = _number((record.get("energy") or {}).get("total_J"))
    scalars["tool_cost_per_part_eur"] = _number(record.get("tool_cost_eur"))
    for tool in (run.tools or {}).get("tools", []):
        if isinstance(tool, dict):
            scalars[f"VB_max_m[T{tool.get('number')}]"] = _number(tool.get("VB_max_m"))
            scalars[f"life_used_fraction[T{tool.get('number')}]"] = _number(tool.get("life_used_fraction"))
    for feature in (run.surface or {}).get("features", []):
        if isinstance(feature, dict):
            scalars[f"Ra_m[{feature.get('id')}]"] = _number(feature.get("Ra_m"))
    result.scalars = {k: v for k, v in sorted(scalars.items()) if v is not None}
    return result


__all__ = ["SampleResult", "read_sample"]
