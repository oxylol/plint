"""Inspection's state: the operation records made at each operation end (checkpointable).

Inspection takes no part in the macro loop; what it carries from one operation end to the next and across a
checkpoint is the list of operation records (JSON documents, cnctwin.inspect.inspector), the next-stock summaries
and the keys of the findings already raised (so a resumed run does not raise them twice). ``to_arrays`` stores them
as JSON metadata (no arrays), ``from_arrays`` restores them exactly. SimState has no ``inspect`` field: the domain
keeps this object and offers the checkpoint participant protocol (cnctwin.core.checkpoint).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class InspectState:
    records: list = field(default_factory=list)          # operation records, in the order they were made
    next_stocks: list = field(default_factory=list)      # NextStock.to_json() per transfer
    finding_keys: list = field(default_factory=list)     # findings raised (dedupe on resume)
    files: list = field(default_factory=list)            # result files written (relative to the run directory)

    def to_arrays(self) -> tuple[dict, dict]:
        return {}, {"records": copy.deepcopy(self.records), "next_stocks": copy.deepcopy(self.next_stocks),
                    "finding_keys": list(self.finding_keys), "files": list(self.files)}

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "InspectState":
        return cls(copy.deepcopy(list(meta.get("records", []))), copy.deepcopy(list(meta.get("next_stocks", []))),
                   list(meta.get("finding_keys", [])), list(meta.get("files", [])))


__all__ = ["InspectState"]
