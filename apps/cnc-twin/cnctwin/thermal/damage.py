"""Thermal damage of the machined surface (docs/models/thermal.md "Thermal damage").

The two-scale peak temperature of the generated surface at each contact is compared with the
thresholds of the material database (``thermal_damage``): temper/burn colours (``burn_degC``),
over-tempering of hardened steels (``overtemper_degC`` where given), rehardening to a white layer
(``white_layer_degC``, Ac1) and melting/smearing (``melt_smear_degC``). Each exceedance raises one
``thermal_damage`` finding per (operation, workpiece, kind, program block), with the location in S
and the tick; repeats in the same block are counted in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.findings import Category, Severity
from ..units import ZERO_DEGC_K
from .inputs import plain, raw

KINDS = (
    # kind, database key, severity, what happens
    ("burn", "burn_degC", Severity.WARNING, "temper / oxide colours (burn marks) on the surface"),
    ("overtemper", "overtemper_degC", Severity.WARNING, "the hardened surface over-tempers (soft layer)"),
    ("white_layer", "white_layer_degC", Severity.ERROR, "the surface rehardens to a brittle white layer"),
    ("melt_smear", "melt_smear_degC", Severity.ERROR, "the surface melts or smears"),
)


@dataclass(frozen=True)
class Threshold:
    kind: str
    temp_K: float
    severity: Severity
    effect: str


def thresholds(material) -> list[Threshold]:
    """Thresholds of a material (resolved SI view or raw mapping), lowest first."""
    data = raw(material) or {}
    section = data.get("thermal_damage") if isinstance(data, dict) else None
    out = []
    if not isinstance(section, dict):
        return out
    for kind, key, severity, effect in KINDS:
        value = plain(section.get(key))
        if value is None:
            continue
        out.append(Threshold(kind, float(value) + ZERO_DEGC_K, severity, effect))
    return sorted(out, key=lambda t: t.temp_K)


class DamageMonitor:
    """Raises thermal_damage findings once per (op, workpiece, kind, block)."""

    def __init__(self, material_name: str, limits: list[Threshold]):
        self.material_name = material_name
        self.limits = list(limits)
        self.seen: set[tuple[int, int, str, int]] = set()
        self.counts: dict[str, int] = {}
        self.max_K: dict[str, float] = {}

    def check(self, ctx, state, op: int, workpiece: int, block: int, tick: int, path: int, surface_K: float,
              quantities: dict, parents: list[int]) -> list:
        raised = []
        for limit in self.limits:
            if not surface_K > limit.temp_K:
                continue
            self.counts[limit.kind] = self.counts.get(limit.kind, 0) + 1
            self.max_K[limit.kind] = max(self.max_K.get(limit.kind, 0.0), surface_K)
            key = (int(op), int(workpiece), limit.kind, int(block))
            if key in self.seen:
                continue
            self.seen.add(key)
            message = (f"surface temperature {surface_K - ZERO_DEGC_K:.0f} degC exceeds the {limit.kind.replace('_', ' ')} "
                       f"temperature {limit.temp_K - ZERO_DEGC_K:.0f} degC of {self.material_name}: {limit.effect}")
            values = dict(quantities)
            values["surface_temp_K"] = surface_K
            values["threshold_K"] = limit.temp_K
            raised.append(ctx.finding(state, "thermal", Category.THERMAL_DAMAGE, limit.severity, message, tick=tick,
                                      path=path, quantities=values, parents=parents))
        return raised

    def to_meta(self) -> dict:
        return {"seen": sorted([list(k) for k in self.seen]), "counts": dict(sorted(self.counts.items())),
                "max_K": dict(sorted(self.max_K.items())), "material_name": self.material_name,
                "limits": [[t.kind, t.temp_K, t.severity.value, t.effect] for t in self.limits]}

    @classmethod
    def from_meta(cls, meta: dict) -> "DamageMonitor":
        obj = cls(meta["material_name"], [Threshold(k, float(t), Severity(s), e) for k, t, s, e in meta["limits"]])
        obj.seen = {(int(a), int(b), str(c), int(d)) for a, b, c, d in meta["seen"]}
        obj.counts = {str(k): int(v) for k, v in meta["counts"].items()}
        obj.max_K = {str(k): float(v) for k, v in meta["max_K"].items()}
        return obj
