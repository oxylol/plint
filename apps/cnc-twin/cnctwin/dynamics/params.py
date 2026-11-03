"""Model coefficients of the dynamics domain (db/models/dynamics-*.yaml, SI through cnctwin.db.models.params)."""

from __future__ import annotations

from ..db.models import params as _params

CHATTER = "dynamics-chatter"
PROCESS_DAMPING = "dynamics-process-damping"
MODES = "dynamics-modes"


def p(model: str, name: str) -> float:
    """One coefficient (SI)."""
    values = _params(model)
    if name not in values:
        raise KeyError(f"db/models/{model}.yaml has no parameter {name!r}")
    return float(values[name])


__all__ = ["p", "CHATTER", "PROCESS_DAMPING", "MODES"]
