"""Time series of the removal domain (macro tier; values from RemovalDomain.channel_values each step)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_PATHS = 16

_SCALAR = (
    ("removal.removed_volume_m3", "m^3",
     "material volume removed in the step (all contacts; exact integer field sums)"),
    ("removal.removal_rate_m3_s", "m^3/s", "mean material removal rate over the step"),
    ("removal.field_volume_m3", "m^3", "material volume of workpiece 0 (level-0 z family)"),
    ("removal.freed_mass_kg", "kg", "mass of the bodies that came loose (cumulative)"),
    ("removal.pool_used_blocks", "-", "overflow pool blocks in use (workpiece 0)"),
    ("removal.events", "-", "removal events in the step (rapid into material, collision, gouge, ...)"),
    ("removal.field_version", "-", "geometry version of workpiece 0"),
    ("removal.polar_volume_m3", "m^3", "material volume of the polar field of workpiece 0 (0: not turning)"),
)

_PER_PATH = (
    ("removal.volume_consistency", "-", "field volume / analytic chip volume of the contact (previous step)"),
    ("removal.ae_m", "m", "radial engagement of the contact (largest in the step)"),
    ("removal.ap_m", "m", "axial engagement of the contact (extent of the removed material along the tool axis)"),
    ("removal.engaged_angle_rad", "rad", "largest engaged angle of a station of the contact"),
    ("removal.in_cut_fraction", "-", "fraction of the step's subtractions that removed material"),
    ("removal.air_cut_time_s", "s", "feed motion with the cutting envelope out of contact in the step"),
)


def channels() -> list[Channel]:
    out = [Channel(name=n, unit=u, tier=Tier.MACRO, domain="removal", doc=d) for n, u, d in _SCALAR]
    out += [Channel(name=n, unit=u, tier=Tier.MACRO, domain="removal", doc=d, max_shape=(MAX_PATHS,))
            for n, u, d in _PER_PATH]
    return out


NAMES = tuple(s[0] for s in _SCALAR) + tuple(s[0] for s in _PER_PATH)
