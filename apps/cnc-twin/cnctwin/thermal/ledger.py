"""Energy ledger of one thermal solver (docs/architecture.md 12).

Per step: change in internal energy = sources - losses - removed enthalpy (net of re-added
material), every term computed independently (the internal energies from the stored enthalpies,
the losses from the face/link conductances the scheme used). A relative residual above the
tolerance raises :class:`~cnctwin.core.errors.ConservationError` (the run fails).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.errors import ConservationError


@dataclass
class SolverLedger:
    name: str
    u_J: float = 0.0             # internal energy now (rel. 20 degC)
    u0_J: float = 0.0            # at the start of the run
    sources_J: float = 0.0       # cumulative
    losses_J: float = 0.0        # cumulative
    removed_J: float = 0.0       # cumulative net removed enthalpy
    step_absorbed_J: float = 0.0 # change of internal energy in the last booked step
    worst_rel: float = 0.0       # largest relative residual seen
    steps: int = 0

    def book(self, u_before: float, u_after: float, sources: float, losses: float, removed: float,
             tol: float, time_s: float) -> float:
        """Book one step; returns the relative residual (raises when it exceeds ``tol``)."""
        du = u_after - u_before
        expected = sources - losses - removed
        scale = abs(u_before) + abs(u_after) + abs(sources) + abs(losses) + abs(removed)
        rel = abs(du - expected) / scale if scale > 0.0 else 0.0
        if rel > tol:
            raise ConservationError(f"thermal energy ({self.name})", expected, du, tol, time_s)
        self.u_J = u_after
        self.sources_J += sources
        self.losses_J += losses
        self.removed_J += removed
        self.step_absorbed_J = du
        self.worst_rel = max(self.worst_rel, rel)
        self.steps += 1
        return rel

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "SolverLedger":
        return cls(**values)
