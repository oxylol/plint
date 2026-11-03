"""Findings: what the run says will happen, each with time, line, position, severity and cause."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2, Severity.FATAL: 3}


class Category(str, Enum):
    COLLISION = "collision"
    OVERTRAVEL = "overtravel"
    SPINDLE_OVERLOAD = "spindle_overload"
    AXIS_OVERLOAD = "axis_overload"
    TOOL_BREAKAGE = "tool_breakage"
    TOOL_END_OF_LIFE = "tool_end_of_life"
    TOOL_PULLOUT = "tool_pullout"
    UNBALANCE = "unbalance"
    OVERSPEED = "overspeed"
    CHATTER = "chatter"
    CHIP_PACKING = "chip_packing"
    CHIP_RECUTTING = "chip_recutting"
    CHIP_HAZARD = "chip_hazard"            # bird-nesting, wrapping, ignition risk, blocked probe/nozzle
    COOLANT_SHADOWING = "coolant_shadowing"
    PART_SLIP = "part_slip"
    PART_FREE = "part_free"
    MISLOCATION = "mislocation"
    OUT_OF_TOLERANCE = "out_of_tolerance"
    FINISH_OUT_OF_SPEC = "finish_out_of_spec"
    BURR = "burr"
    THERMAL_DAMAGE = "thermal_damage"
    THERMAL_ERROR = "thermal_error"
    GOUGE = "gouge"
    REMAINING_MATERIAL = "remaining_material"
    BREAKTHROUGH = "breakthrough"
    RAPID_INTO_MATERIAL = "rapid_into_material"
    NON_CENTRE_CUTTING_PLUNGE = "non_centre_cutting_plunge"
    AIR_CUTTING = "air_cutting"
    SINGULARITY = "singularity"
    SANITY = "sanity"
    PROGRAM_ERROR = "program_error"
    ALARM = "alarm"
    MASS_BALANCE = "mass_balance"
    ENERGY_BALANCE = "energy_balance"
    TIME_BALANCE = "time_balance"


@dataclass
class Finding:
    """One thing the run predicts, tied to the program line and the chain of physical causes."""

    category: Category
    severity: Severity
    message: str
    time_s: float
    tick: int
    domain: str
    op: str = ""
    program: str = ""
    line: int = 0                          # 1-based line in `program`, 0 if not tied to a line
    block: str = ""                        # the block text as written
    pos_W_m: tuple[float, float, float] | None = None   # tool tip in the workpiece frame
    pos_M_m: tuple[float, float, float] | None = None   # tool tip in the machine frame
    quantities: dict[str, float] = field(default_factory=dict)   # SI values, keys carry units
    cause_chain: list[int] = field(default_factory=list)         # trace ids, finding first
    id: int = -1

    def to_dict(self) -> dict:
        record = asdict(self)
        record["category"] = self.category.value
        record["severity"] = self.severity.value
        return record

    @classmethod
    def from_dict(cls, record: dict) -> "Finding":
        values = dict(record)
        values["category"] = Category(values["category"])
        values["severity"] = Severity(values["severity"])
        for key in ("pos_W_m", "pos_M_m"):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        return cls(**values)


class FindingList:
    """Findings of one run, numbered in the order they occur."""

    def __init__(self) -> None:
        self.items: list[Finding] = []

    def add(self, finding: Finding) -> Finding:
        finding.id = len(self.items)
        self.items.append(finding)
        return finding

    def worst(self) -> Severity | None:
        if not self.items:
            return None
        return max((f.severity for f in self.items), key=lambda s: s.rank)

    def safe_to_run(self) -> bool:
        """Safe when nothing reaches severity error."""
        worst = self.worst()
        return worst is None or worst.rank < Severity.ERROR.rank

    def by_category(self) -> dict[str, list[Finding]]:
        grouped: dict[str, list[Finding]] = {}
        for finding in self.items:
            grouped.setdefault(finding.category.value, []).append(finding)
        return grouped

    def to_list(self) -> list[dict]:
        return [f.to_dict() for f in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
