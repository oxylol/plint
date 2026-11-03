"""``physics.controller`` project settings: control parameters a machine builder or operator sets.

Every setting defaults to ``None`` = the dialect's default from ``db/models/controller-<dialect>.yaml``
(sourced or estimated, with notes), so a project only names what differs from the delivered
control. The schema below uses the database schema language (cnctwin.db.schema); keys carry
their unit, :func:`settings_from` reads an SIMap (SI) or a plain mapping (file units).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from ..db.schema import Any_, Count, Flag, MapOf, Qty, Seq, Struct, Text
from ..db.simap import SIMap, Sourced, wrap

SCHEMA = Struct({
    "pocket_calculator": Flag(doc="Fanuc parameter 3401#0 DPI: a value without decimal point is in mm/inch/s"),
    "arc_tolerance_mm": Qty(lo=0, hi=0.01, nullable=True, doc="arc end-point radius tolerance (Fanuc 3410)"),
    "nat": Flag(doc="Fanuc parameter 6004#0 NAT: ASIN/ATAN result ranges -90..90 / -180..180"),
    "wait_m_range": Seq(Count(lo=0, hi=99999999), min_len=2, max_len=2,
                        doc="[first, last] path-wait M codes (FANUC parameters 8110/8111)"),
    "macro_nesting_max": Count(lo=1, hi=99),
    "sub_nesting_max": Count(lo=1, hi=99),
    "max_m_codes_per_block": Count(lo=1, hi=5),
    "same_group": Text(("last", "alarm")),
    "m_codes": MapOf(Any_(), doc="machine-builder M codes: {M60: {kind: pallet}, M57: {kind: output, output: 7, on: true}}"),
    "macro_call_codes": MapOf(Count(lo=1, hi=99999999), doc="{G100: 9010, M06: 9001, T: 9000}: macro/subprogram call codes"),
    "reference_points_mm": MapOf(MapOf(Qty()), doc="{2: {X: .., Y: .., Z: ..}}: 2nd-4th reference points (machine)"),
    "g60_overrun_mm": MapOf(Qty(), doc="single-direction positioning overrun per axis (Fanuc 5440), signed"),
    "power_on_units": Text(("mm", "inch")),
    "power_on": MapOf(Text(), doc="{group number: G code} power-on modal overrides"),
    "tool_length_axis": Text(("A", "C"), doc="Fanuc 5001: A = always Z (type A), C = the axis in the G43 block"),
    "lathe_offsets_in_diameter": Flag(),
    "shared_common_100": Count(lo=0, hi=100),
    "shared_common_500": Count(lo=0, hi=500),
    "paths": Seq(MapOf(Text())),
    "interface_inputs": MapOf(Qty(dimensionless=True)),
    "clock_start": Text(pattern=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
    "macro_statement_ms": Qty(lo=0, hi=1.0, nullable=True),
    "macro_term_ms": Qty(lo=0, hi=1.0, nullable=True),
    "call_ms": Qty(lo=0, hi=1.0, nullable=True),
    "goto_scan_ms": Qty(lo=0, hi=1.0, nullable=True),
    "horizon_ms": Qty(gt=0, hi=100.0, nullable=True),
    "feed_mode_change_clears_f": Flag(),
    "g92_cancels_g52": Flag(),
    "part_count_codes": Count(lo=0, hi=9999, doc="M codes that count a part (#3901)"),
    "haas_offsets_diameter": Flag(doc="Haas setting 40: D offsets are diameters"),
    "g37_feed_mm_min": Qty(gt=0, nullable=True),
    "g37_approach_mm": Qty(lo=0, nullable=True),
    "g37_tolerance_mm": Qty(lo=0, nullable=True),
    # --- additive (cycles, compensation, probing, tool life, restart, TCP: docs/dialects.md "Cycles and
    # compensation"; defaults in db/models/controller-cycles.yaml, each named after the parameter it stands for)
    "cycle_parameters": MapOf(Any_(), doc="cycle/compensation/tool-life/restart/TCP parameters by name with unit "
                                          "suffix (g73_retract_mm: 0.5, crc_startup_type: B ...)"),
    "tool_life": Any_(doc="tool life management groups: {groups: [{group, life, count: uses|time, tools: "
                          "[{tool, h, d}]}]} (FANUC tool life management / Haas tool groups)"),
    "builtin_macros": Flag(doc="load the built-in probe macro library (O9810-O9814, O9851-O9853); default: "
                               "when the machine has a spindle probe or a tool setter"),
}, meta=False)


@dataclass
class ControllerSettings:
    """Resolved control parameters (SI). None = take the dialect default."""

    pocket_calculator: bool | None = None
    arc_tolerance_m: float | None = None
    nat: bool | None = None
    wait_m_range: list[int] | None = None
    macro_nesting_max: int | None = None
    sub_nesting_max: int | None = None
    max_m_codes_per_block: int | None = None
    same_group: str | None = None
    m_codes: dict = field(default_factory=dict)
    macro_call_codes: dict = field(default_factory=dict)
    reference_points_m: dict = field(default_factory=dict)
    g60_overrun_m: dict = field(default_factory=dict)
    power_on_units: str | None = None
    power_on: dict = field(default_factory=dict)
    tool_length_axis: str | None = None
    lathe_offsets_in_diameter: bool | None = None
    shared_common_100: int = 0
    shared_common_500: int = 0
    paths: list | None = None
    interface_inputs: dict = field(default_factory=dict)
    clock_start: str = "2026-01-05T07:00:00"
    macro_statement_s: float | None = None
    macro_term_s: float | None = None
    call_s: float | None = None
    goto_scan_s: float | None = None
    horizon_s: float | None = None
    feed_mode_change_clears_f: bool | None = None
    g92_cancels_g52: bool | None = None
    part_count_codes: list[int] | None = None
    haas_offsets_diameter: bool | None = None
    g37_feed_m_s: float | None = None
    g37_approach_m: float | None = None
    g37_tolerance_m: float | None = None
    cycle_parameters: dict = field(default_factory=dict)     # additive: cycles/comp/probing/tool life/restart/TCP
    tool_life: dict | None = None                            # additive: tool life management groups
    builtin_macros: bool | None = None                       # additive: built-in probe macro library


def _plain(value):
    if isinstance(value, Sourced):
        return value.value
    if isinstance(value, SIMap):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def settings_from(section) -> ControllerSettings:
    """ControllerSettings from ``physics.controller`` (SIMap, plain file mapping, or None)."""
    if section is None:
        return ControllerSettings()
    view = section if isinstance(section, SIMap) else wrap(dict(section))
    names = {f.name for f in fields(ControllerSettings)}
    values = {}
    for key, value in view.items():
        if key in names:
            values[key] = _plain(value)
    for key in ("m_codes", "macro_call_codes", "reference_points_m", "g60_overrun_m", "power_on",
                "interface_inputs", "cycle_parameters"):
        if values.get(key) is None:
            values.pop(key, None)
    if "reference_points_m" in values:
        values["reference_points_m"] = {int(k): {a: float(v) for a, v in dict(p).items()}
                                        for k, p in values["reference_points_m"].items()}
    if "power_on" in values:
        values["power_on"] = {int(k): str(v) for k, v in values["power_on"].items()}
    if "interface_inputs" in values:
        values["interface_inputs"] = {int(str(k).lstrip("#")): float(v) for k, v in values["interface_inputs"].items()}
    if "macro_call_codes" in values:
        values["macro_call_codes"] = {str(k): int(v) for k, v in values["macro_call_codes"].items()}
    if isinstance(values.get("part_count_codes"), int):
        values["part_count_codes"] = [values["part_count_codes"]]
    return ControllerSettings(**values)
