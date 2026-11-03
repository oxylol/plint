"""Time series of the chips domain (macro tier; values in ``state.chips.channel_values`` each step)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

_SPECS = (
    ("chips.in_flight_kg", "kg", "chip mass flying (in air, jets, the film, bouncing)"),
    ("chips.settled_kg", "kg", "chip mass resting on the part, fixture, table and in the chip pan"),
    ("chips.conveyed_kg", "kg", "chip mass the conveyor has removed (cumulative)"),
    ("chips.adhered_kg", "kg", "chip mass welded to edges, wrapped on tools/bars, stuck on nozzles"),
    ("chips.in_flutes_kg", "kg", "chip mass in the flutes of all tools"),
    ("chips.dust_kg", "kg", "airborne dust (composites, cast iron fines; cumulative)"),
    ("chips.emitted_kg", "kg", "chip mass booked from the removed volume (cumulative)"),
    ("chips.particles", "-", "representative particles flying or attached"),
    ("chips.substeps", "-", "particle sub-steps taken in the step (cost)"),
    ("chips.flute_fill", "-", "flute fill of the active tool of path 0 (bulk chips / gullet volume)"),
    ("chips.packing_torque_Nm", "N m", "packing torque of the active tool of path 0"),
    ("chips.packing_force_N", "N", "axial packing force of the active tool of path 0 (+z_T)"),
    ("chips.packing_power_W", "W", "friction power of the packing loads of the active tool of path 0"),
    ("chips.part_chip_mass_kg", "kg", "chip mass resting on the workpiece"),
    ("chips.pile_max_height_m", "m", "highest chip pile above its base"),
    ("chips.recut_mass_kg", "kg", "chip mass recut by the tools in the step"),
    ("chips.flushed_mass_kg", "kg", "chip mass mobilised by the coolant in the step"),
    ("chips.heat_to_part_W", "W", "heat of resting chips into the workpiece (mean over the step)"),
    ("chips.heat_to_coolant_W", "W", "chip heat into the coolant (mean over the step)"),
    ("chips.heat_to_air_W", "W", "chip heat to the air and the enclosure (mean over the step)"),
    ("chips.max_temp_K", "K", "hottest flying or resting chip"),
    ("chips.pan_kg", "kg", "chips in the chip pan / conveyor trough not yet removed"),
    ("chips.dust_rate_kg_s", "kg/s", "dust generation rate"),
)


def channels() -> list[Channel]:
    return [Channel(name=name, unit=unit, tier=Tier.MACRO, domain="chips", doc=doc) for name, unit, doc in _SPECS]


NAMES = tuple(spec[0] for spec in _SPECS)
