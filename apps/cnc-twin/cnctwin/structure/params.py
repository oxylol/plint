"""Model coefficients of the structure domain (db/models/structure-*.yaml; docs/models/structure.md 9).

``p(model, name)`` reads one SI value (cached by :func:`cnctwin.db.models.params`); the holder-type tables map
the holder ``type`` of db/_templates/holder.yaml to the interface family whose contact stiffness and loss
factor the model file gives.
"""

from __future__ import annotations

import functools

from ..core.errors import InputError
from ..db.models import params

TOOL = "structure-tool"
WORK = "structure-work"

CM2_IN_M2 = 1e-4          # unit of the per-cm2 contact stiffness keys (k_per_cm2_..._N_per_m = N/m of 1 cm2)

# holder type (db/_templates/holder.yaml) -> interface family of the contact parameters
HOLDER_FAMILY = {
    "shrink_fit": "shrink_fit", "hydraulic": "hydraulic", "milling_chuck": "milling_chuck",
    "er_collet": "er_collet", "side_lock": "side_lock", "tapping_floating": "tapping", "tapping_synchro": "tapping",
    "modular": "face", "arbor": "face", "boring_head": "face",
    "turret_vdi": "turret", "turret_bmt": "turret", "lathe_holder": "turret",
}


@functools.lru_cache(maxsize=None)
def _table(model: str) -> dict:
    return dict(params(model))


def p(model: str, name: str) -> float:
    table = _table(model)
    if name not in table:
        raise InputError(f"db/models/{model}.yaml has no parameter {name!r}")
    return float(table[name])


def family(holder_type: str | None) -> str:
    """Interface family of a holder type; unknown or missing types use the collet values (the most compliant
    common connection: conservative for chatter)."""
    if not holder_type:
        return "er_collet"
    return HOLDER_FAMILY.get(str(holder_type), "er_collet")


def contact_stiffness_N_per_m3(holder_type: str | None) -> float:
    """Normal contact stiffness per unit area of the clamping interface (N/m per m2)."""
    return p(TOOL, f"contact_k_per_cm2_{family(holder_type)}_N_per_m") / CM2_IN_M2


def contact_loss_factor(holder_type: str | None) -> float:
    return p(TOOL, f"contact_loss_factor_{family(holder_type)}")


def mindlin_ratio(nu: float) -> float:
    """Tangential / normal contact stiffness 2 (1 - nu) / (2 - nu) (Mindlin; Johnson 1985 sec. 7.2)."""
    return 2.0 * (1.0 - nu) / (2.0 - nu)


def clear() -> None:
    _table.cache_clear()


__all__ = ["TOOL", "WORK", "CM2_IN_M2", "HOLDER_FAMILY", "p", "family", "contact_stiffness_N_per_m3",
           "contact_loss_factor", "mindlin_ratio", "clear"]
