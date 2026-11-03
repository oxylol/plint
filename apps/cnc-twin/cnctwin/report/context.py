"""What the report's sections share: the run, the program-line anchors, the series summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .html import Raw, anchor, esc, link
from .rundir import RunData, TierSummary

# Presentation settings (not model coefficients): a program is listed whole up to this many lines,
# longer programs as excerpts of CONTEXT_LINES around every line with a finding.
FULL_LISTING_MAX_LINES = 400
CONTEXT_LINES = 3
SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "fatal": 3}


def program_key(op: str, program: str) -> tuple[str, str]:
    return str(op or ""), Path(str(program or "")).name


@dataclass
class Context:
    run: RunData
    study: dict | None = None
    listed: set[str] = field(default_factory=set)          # anchors of program lines in the listing
    _tiers: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ findings by line
    @property
    def findings_by_line(self) -> dict[tuple[str, str, int], list[dict]]:
        out: dict[tuple[str, str, int], list[dict]] = {}
        for finding in self.run.findings:
            line = int(finding.get("line") or 0)
            if line > 0:
                op, program = program_key(finding.get("op", ""), finding.get("program", ""))
                out.setdefault((op, program, line), []).append(finding)
        return out

    def worst(self, findings: list[dict]) -> str:
        return max((f.get("severity", "info") for f in findings), key=lambda s: SEVERITY_RANK.get(s, 0),
                   default="info")

    # ------------------------------------------------------------------ anchors and links
    @staticmethod
    def line_anchor(op: str, program: str, line: int) -> str:
        op, program = program_key(op, program)
        return anchor("L", op, program, int(line))

    def line_link(self, op: str, program: str, line: int, text=None) -> Raw:
        label = text if text is not None else f"{Path(str(program)).name or op}:{int(line)}"
        if int(line or 0) <= 0:
            return Raw(esc(label))
        target = self.line_anchor(op, program, line)
        return link(f"#{target}", label) if target in self.listed else Raw(esc(label))

    @staticmethod
    def finding_anchor(finding_id) -> str:
        return anchor("F", finding_id)

    def finding_link(self, finding_id, text=None) -> Raw:
        return link(f"#{self.finding_anchor(finding_id)}", text if text is not None else f"#{finding_id}")

    # ------------------------------------------------------------------ series
    def tier(self, name: str) -> TierSummary | None:
        if name not in self._tiers:
            self._tiers[name] = self.run.tier_summary(name)
        return self._tiers[name]

    def channel(self, name: str) -> dict | None:
        for entry in self.run.channels:
            if entry.get("name") == name:
                return entry
        return None


__all__ = ["Context", "FULL_LISTING_MAX_LINES", "CONTEXT_LINES", "SEVERITY_RANK", "program_key"]
