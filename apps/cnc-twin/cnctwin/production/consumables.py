"""Coolant, MQL oil and compressed air used per part.

Water-mixed fluids (emulsion, semi-synthetic, synthetic): the mixture leaves the tank with the chips
(``drag_out_ml_per_kg_chips`` x chip mass) and as mist (``mist_loss_fraction`` x pumped volume);
the concentrate lost is that volume times the concentration; water also evaporates from the tank
surface (``evaporation_l_per_h_per_m2`` x tank volume / ``tank_depth_m`` x time) and is made up.
Straight oil: drag-out and mist are oil. MQL: total loss, dosage x time the MQL unit runs (the
fluid file's ``dosage_default_ml_h`` or db/models/production-consumables.yaml), plus its carrier
air. Compressed air: the machine's own consumption while it is on, the air blast while commanded,
the MQL carrier air. Costs from the fluid file's ``cost_eur_per_l`` (concentrate or oil price).
"""

from __future__ import annotations

from ..db.models import params as model_params
from ..sanity.si import get, num, si_view, text

WATER_MIXED = ("emulsion", "semi_synthetic", "synthetic")
M3_PER_L = 1e-3


def coolant_use(fluid, *, chip_mass_kg: float | None, pumped_m3: float, mql_on_s: float, elapsed_s: float,
                tank_m3: float | None, concentration_frac: float | None = None, params=None) -> dict:
    """Fluid consumption of one part (m3) and its cost (EUR); fields that cannot be known are None
    with the reason in ``missing``."""
    p = params if params is not None else model_params("production-consumables")
    fluid = si_view(fluid)
    kind = text(get(fluid, "kind"), "dry")
    missing: list[str] = []
    result = {"kind": kind, "drag_out_m3": 0.0, "mist_m3": 0.0, "evaporated_water_m3": 0.0, "concentrate_m3": 0.0,
              "water_m3": 0.0, "oil_m3": 0.0, "cost_eur": 0.0, "missing": missing}
    if fluid is None or kind in ("dry", "air"):
        return result
    price = num(get(fluid, "cost_eur_per_l"))
    if kind == "mql_oil":
        dosage = num(get(fluid, "dosage_default_m3_s")) or float(p["mql_dosage_m3_s"])
        result["oil_m3"] = dosage * mql_on_s
        result["cost_eur"] = result["oil_m3"] / M3_PER_L * price if price is not None else None
        if price is None:
            missing.append("cost_eur_per_l")
        return result
    drag = num(get(fluid, "drag_out_m3_per_kg"), 0.0)
    mist = num(get(fluid, "mist_loss_fraction"), 0.0)
    if chip_mass_kg is None:
        missing.append("chip mass (removed mass of the part)")
        drag_out = 0.0
    else:
        drag_out = drag * chip_mass_kg
    mist_m3 = mist * pumped_m3
    lost = drag_out + mist_m3
    result["drag_out_m3"] = drag_out
    result["mist_m3"] = mist_m3
    if kind in WATER_MIXED:
        c = concentration_frac if concentration_frac is not None else num(get(fluid, "concentration_frac"), 0.0)
        evaporation = num(get(fluid, "evaporation_m3_s_per_m2"), 0.0)
        area = tank_m3 / float(p["tank_depth_m"]) if tank_m3 else 0.0
        if not tank_m3:
            missing.append("machine coolant.tank_l (evaporation)")
        evaporated = evaporation * area * elapsed_s
        result["concentrate_m3"] = lost * c
        result["water_m3"] = lost * (1.0 - c) + evaporated
        result["evaporated_water_m3"] = evaporated
        used = result["concentrate_m3"]
    elif kind == "straight_oil":
        result["oil_m3"] = lost
        used = lost
    else:
        result["water_m3"] = lost
        used = 0.0
    if price is None and used > 0.0:
        missing.append("cost_eur_per_l")
        result["cost_eur"] = None
    else:
        result["cost_eur"] = used / M3_PER_L * (price or 0.0)
    return result


def air_use(machine, *, machine_on_s: float, blast_on_s: float, mql_on_s: float, params=None) -> dict:
    """Free-air volumes (m3) of one part."""
    p = params if params is not None else model_params("production-consumables")
    machine = si_view(machine)
    own = num(get(machine, "power", "air_consumption_m3_s"), 0.0) * machine_on_s
    blast_flow = num(get(machine, "coolant", "air_blast", "flow_m3_s"), 0.0)
    blast = blast_flow * blast_on_s
    mql = float(p["mql_air_m3_s"]) * mql_on_s
    return {"machine_m3": own, "air_blast_m3": blast, "mql_m3": mql, "total_m3": own + blast + mql}
