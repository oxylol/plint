"""Chip findings with their chains of cause (docs/models/chips.md "Findings").

Each rule raises a finding once when its condition starts (per tool slot or globally) and re-arms when it
clears; the thresholds are db/models/chips-hazards.yaml. Every finding's parents are the trace events the
chips domain recorded for its causes (the formation of the chips of the tool, the flute state, the weld or
wrap events, the pile state at the tool entry), so ``twin explain`` walks back to the chip form and the
cutting conditions.
"""

from __future__ import annotations

from ..core.findings import Category, Severity
from .state import (F_NEST, F_PACKING, F_RECUT, F_WELD, F_WRAP, G_BODY_WRAP, G_CONVEYOR, G_DUST,
                    G_FINES, G_IGNITION, G_PROBE, G_UNSCALED, T_FLUTE, T_FORMATION, T_RECUT, T_WELD, T_WRAP)

DOMAIN = "chips"


def _trace(ctx, tick: int, event: str, quantities: dict, parents: list[int]) -> int:
    return ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, event, quantities, [p for p in parents if p >= 0])


def _raise(ctx, state, category, severity, message, tick, quantities, parents, path=0):
    return ctx.finding(state, DOMAIN, category, severity, message, tick, path=path, quantities=quantities,
                       parents=[p for p in parents if p >= 0])


def packing(ctx, state, cs, hz, slot: int, tool, fill: float, quantities: dict, tick: int, path: int) -> None:
    flag = int(cs.flags[slot, F_PACKING])
    if fill < float(hz["flute_fill_rearm"]):
        cs.flags[slot, F_PACKING] = 0
        return
    level = 2 if fill >= float(hz["flute_fill_clog"]) else (1 if fill >= float(hz["flute_fill_warn"]) else 0)
    if level <= flag:
        return
    event = _trace(ctx, tick, "chips:flute_fill", {"slot": slot, "fill": fill, **quantities},
                   [int(cs.trace_ids[slot, T_FORMATION])])
    cs.trace_ids[slot, T_FLUTE] = event
    cs.flags[slot, F_PACKING] = level
    if level == 2:
        severity = Severity.ERROR if tool is not None and tool.hole_making else Severity.WARNING
        message = (f"chips packed in the flutes of tool slot {slot} (fill {fill:.2f}): packing torque "
                   f"{quantities.get('packing_torque_Nm', 0.0):.2f} N m; peck, add through-tool coolant or reduce feed")
    else:
        severity = Severity.WARNING
        message = f"chip evacuation of tool slot {slot} is marginal (flute fill {fill:.2f})"
    _raise(ctx, state, Category.CHIP_PACKING, severity, message, tick, {"fill": fill, **quantities}, [event], path)


def recut(ctx, state, cs, hz, slot: int, fraction: float, recut_kg: float, tick: int, path: int) -> None:
    warn = float(hz["recut_fraction_warn"])
    if fraction < 0.5 * warn:
        cs.flags[slot, F_RECUT] = 0
        return
    if fraction < warn or cs.flags[slot, F_RECUT]:
        return
    event = _trace(ctx, tick, "chips:recut", {"slot": slot, "recut_fraction": fraction, "recut_mass_kg": recut_kg},
                   [int(cs.trace_ids[slot, T_FORMATION]), int(cs.trace_ids[slot, T_RECUT])])
    cs.flags[slot, F_RECUT] = 1
    _raise(ctx, state, Category.CHIP_RECUTTING, Severity.WARNING,
           f"tool slot {slot} recuts its chips ({100 * fraction:.1f} % of the chip mass it makes)", tick,
           {"recut_fraction": fraction, "recut_mass_kg": recut_kg}, [event], path)


def weld(ctx, state, cs, hz, slot: int, welded_kg: float, temp_K: float, tick: int, path: int) -> None:
    if welded_kg < float(hz["weld_mass_warn_kg"]) or cs.flags[slot, F_WELD]:
        return
    cs.flags[slot, F_WELD] = 1
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.WARNING,
           f"hot chips weld to the cutting edge of tool slot {slot} ({1e3 * welded_kg:.3f} g at {temp_K - 273.15:.0f} degC): "
           "edge chipping and surface defects when they are recut", tick,
           {"welded_mass_kg": welded_kg, "chip_temp_K": temp_K},
           [int(cs.trace_ids[slot, T_WELD]), int(cs.trace_ids[slot, T_FORMATION])], path)


def nest(ctx, state, cs, hz, slot: int, fraction: float, tick: int, path: int) -> None:
    warn = float(hz["nest_fraction_warn"])
    if fraction < 0.5 * warn:
        cs.flags[slot, F_NEST] = 0
        return
    if fraction < warn or cs.flags[slot, F_NEST]:
        return
    cs.flags[slot, F_NEST] = 1
    event = _trace(ctx, tick, "chips:nest", {"slot": slot, "nest_fraction": fraction},
                   [int(cs.trace_ids[slot, T_FORMATION])])
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.WARNING,
           f"long chips nest around tool slot {slot} (bird's nest, {100 * fraction:.0f} % of the space around it)",
           tick, {"nest_fraction": fraction}, [event], path)


def wrap(ctx, state, cs, hz, slot: int, mass_kg: float, tick: int, path: int, body: bool) -> None:
    flags = cs.gflags if body else cs.flags[slot]
    key = G_BODY_WRAP if body else F_WRAP
    level = 2 if mass_kg >= float(hz["wrap_mass_error_kg"]) else (1 if mass_kg >= float(hz["wrap_mass_warn_kg"]) else 0)
    if level == 0:
        flags[key] = 0
        return
    if level <= flags[key]:
        return
    flags[key] = level
    parent = int(cs.trace_ids[slot, T_WRAP]) if (not body and slot >= 0) else -1
    what = "the turning workpiece" if body else f"tool slot {slot}"
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.ERROR if level == 2 else Severity.WARNING,
           f"stringy chips wrap around {what} ({1e3 * mass_kg:.1f} g)", tick, {"wrap_mass_kg": mass_kg},
           [parent, int(cs.trace_ids[slot, T_FORMATION]) if slot >= 0 else -1], path)


def nested_pocket(ctx, state, cs, hz, slot: int, mass_kg: float, height_m: float, tick: int, path: int) -> None:
    if mass_kg < float(hz["nested_mass_warn_kg"]):
        return
    event = _trace(ctx, tick, "chips:nested_at_entry", {"slot": slot, "mass_kg": mass_kg, "height_m": height_m}, [])
    _raise(ctx, state, Category.CHIP_RECUTTING, Severity.WARNING,
           f"tool slot {slot} enters {1e3 * mass_kg:.2f} g of chips lying in the pocket ({1e3 * height_m:.1f} mm deep)",
           tick, {"mass_kg": mass_kg, "height_m": height_m}, [event], path)


def probe(ctx, state, cs, hz, height_m: float, tick: int, path: int) -> bool:
    blocked = height_m >= float(hz["probe_debris_m"])
    if not blocked:
        cs.gflags[G_PROBE] = 0
        return False
    if not cs.gflags[G_PROBE]:
        cs.gflags[G_PROBE] = 1
        event = _trace(ctx, tick, "chips:probe_debris", {"height_m": height_m}, [])
        _raise(ctx, state, Category.CHIP_HAZARD, Severity.WARNING,
               f"chips ({1e3 * height_m:.2f} mm) lie at the probing point: false trigger or wrong measurement", tick,
               {"chip_height_m": height_m}, [event], path)
    return True


def nozzle(ctx, state, nozzle_id: str, bulk_m3: float, tick: int) -> None:
    event = _trace(ctx, tick, "chips:nozzle_blocked", {"bulk_m3": bulk_m3}, [])
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.WARNING,
           f"coolant nozzle {nozzle_id} is blocked by chips caught on it", tick, {"chip_bulk_m3": bulk_m3}, [event])


def dust(ctx, state, cs, rate_kg_s: float, tick: int) -> None:
    if rate_kg_s <= 0.0 or cs.gflags[G_DUST]:
        return
    cs.gflags[G_DUST] = 1
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.WARNING,
           f"machining makes airborne dust ({1e3 * rate_kg_s:.3f} g/s): extraction and respiratory protection needed",
           tick, {"dust_rate_kg_s": rate_kg_s}, [])


def ignition(ctx, state, cs, hz, temp_K: float, threshold_K: float, tick: int) -> None:
    if temp_K < threshold_K - float(hz["ignition_margin_K"]):
        cs.gflags[G_IGNITION] = 0
        return
    if cs.gflags[G_IGNITION]:
        return
    cs.gflags[G_IGNITION] = 1
    event = _trace(ctx, tick, "chips:ignition_risk", {"chip_temp_K": temp_K, "ignition_K": threshold_K}, [])
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.ERROR,
           f"chips reach {temp_K - 273.15:.0f} degC, above the material's chip ignition temperature "
           f"{threshold_K - 273.15:.0f} degC: fire risk", tick, {"chip_temp_K": temp_K, "ignition_K": threshold_K},
           [event])


def fines(ctx, state, cs, hz, mass_kg: float, tick: int) -> None:
    if mass_kg < float(hz["fines_mass_warn_kg"]):
        cs.gflags[G_FINES] = 0
        return
    if cs.gflags[G_FINES]:
        return
    cs.gflags[G_FINES] = 1
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.WARNING,
           f"{1e3 * mass_kg:.1f} g of fine chips of an ignitable material collect in the work zone (dry): fire risk",
           tick, {"fines_mass_kg": mass_kg}, [])


def conveyor(ctx, state, cs, backlog_m3: float, capacity_m3: float, tick: int) -> None:
    if backlog_m3 <= capacity_m3:
        cs.gflags[G_CONVEYOR] = 0
        return
    if cs.gflags[G_CONVEYOR]:
        return
    cs.gflags[G_CONVEYOR] = 1
    _raise(ctx, state, Category.CHIP_HAZARD, Severity.WARNING,
           f"the chip pan overflows ({1e3 * backlog_m3:.1f} l of chips, capacity {1e3 * capacity_m3:.1f} l): the conveyor "
           "cannot keep up", tick, {"backlog_m3": backlog_m3, "capacity_m3": capacity_m3}, [])


def locator(ctx, state, locator_id: str, height_m: float, tolerance_m: float, tick: int) -> None:
    if height_m <= tolerance_m:
        return
    event = _trace(ctx, tick, "chips:locator_debris", {"height_m": height_m}, [])
    _raise(ctx, state, Category.MISLOCATION, Severity.WARNING,
           f"chips ({1e3 * height_m:.2f} mm) lie on face {locator_id}, which the next setup clamps or locates on: "
           "clean it before loading", tick, {"debris_height_m": height_m}, [event])


def unscaled(ctx, state, cs, contact: int, mass_kg: float, tick: int) -> None:
    """Emissions without a removal record of their window: booked with their analytic mass (recorded, and
    reported once, because the mass balance against the removed volume then cannot close)."""
    _trace(ctx, tick, "chips:unscaled_emission", {"contact": contact, "mass_kg": mass_kg}, [])
    if cs.gflags[G_UNSCALED]:
        return
    cs.gflags[G_UNSCALED] = 1
    _raise(ctx, state, Category.SANITY, Severity.WARNING,
           f"chip emissions of contact {contact} arrived without a removal record of their window: booked with the "
           "analytic chip volume", tick, {"mass_kg": mass_kg}, [])
