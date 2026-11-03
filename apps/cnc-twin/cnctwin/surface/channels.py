"""Time series the surface domain records (macro tier; values from SurfaceDomain.channel_values)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

_DEFS = (
    ("surface.events", "-", "surface events gathered this step"),
    ("surface.profiled_events", "-", "surface events that joined a profile line this step"),
    ("surface.patches", "-", "face patches generated so far"),
    ("surface.open_lines", "-", "profile lines open (memory bound)"),
    ("surface.last_Ra_m", "m", "Ra of the most recently evaluated profile line (NaN before the first)"),
    ("surface.last_Rz_m", "m", "Rz of the most recently evaluated profile line"),
    ("surface.last_Rt_m", "m", "Rt of the most recently evaluated profile line"),
    ("surface.burr_max_m", "m", "highest exit burr formed so far"),
    ("surface.marks", "-", "marks recorded so far (dwell, stop, servo, witness)"),
    ("surface.residual_stress_Pa", "Pa", "surface residual stress of the layer generated this step (last contact)"),
)


def channels() -> list[Channel]:
    return [Channel(name, unit, Tier.MACRO, "surface", doc) for name, unit, doc in _DEFS]


NAMES = tuple(d[0] for d in _DEFS)

__all__ = ["channels", "NAMES"]
