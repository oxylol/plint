"""Canned and multiple repetitive cycles for the controller core (docs/dialects.md "Cycles and
compensation"): FANUC M series drilling G73-G89 and G84.2/G84.3, T series G70-G76, G83-G89, G90/G92/G94,
and the Haas NGC variants (mill G73-G89 with G77 back bore and the G70/G71/G72 bolt hole patterns; lathe
G70-G76, G90/G92/G94, the face cycles G81-G89 and G184, the radial cycles G241-G249). The same G code means
different cycles on mills and lathes (G73, G74, G76 ...), so each code is registered once with a
:class:`Dispatch` that picks the implementation by the dialect's variant; Haas variants are registered
under ``haas:<code>``.
"""

from __future__ import annotations

from .. import hooks
from ..alarms import ControllerAlarm
from ..dialects.base import GSpec
from . import common
from .drilling import FS15_CODES, HAAS_LATHE_CODES, HAAS_MILL_CODES, LATHE_CODES, MILL_CODES, DrillCycle
from .lathe import BoxCycle, GrooveCycle, ThreadCycle
from .mrc import Finishing, PatternRepeat, StockRemoval
from .patterns import BoltPattern


class Dispatch:
    """One cycle code: the mill (M variant) or lathe (T variant) implementation."""

    def __init__(self, code: str, mill=None, lathe=None):
        self.code = code
        self.mill = mill
        self.lathe = lathe

    def _impl(self, ex):
        impl = self.lathe if getattr(ex.handler, "T", False) else self.mill
        if impl is None:
            raise ControllerAlarm("cycle_unavailable", f"{self.code} on this machine type")
        return impl

    def expand(self, ex, call):
        return common.drop_null_moves(self._impl(ex).expand(ex, call), call.current)

    def cancel(self, ex, code: str) -> None:
        impl = self.lathe if getattr(ex.handler, "T", False) else self.mill
        if impl is not None and hasattr(impl, "cancel"):
            impl.cancel(ex, code)


def _lathe(code: str, haas: bool):
    if code in ("G71", "G72"):
        return StockRemoval(code, haas)
    if code == "G73":
        return PatternRepeat(haas)
    if code == "G70":
        return Finishing()
    if code in ("G74", "G75"):
        return GrooveCycle(code, haas)
    if code == "G76":
        return ThreadCycle(haas)
    if code in ("G90", "G92", "G94"):
        return BoxCycle(code, haas)
    if haas:
        return DrillCycle(code, True) if code in HAAS_LATHE_CODES else None
    if code in LATHE_CODES:
        return DrillCycle(code, haas)
    return None


def _haas_mill(code: str):
    if code in ("G70", "G71", "G72"):
        return BoltPattern(code)
    if code in MILL_CODES or code in HAAS_MILL_CODES:
        return DrillCycle(code, haas=True)
    return None


LATHE_ONLY = ("G70", "G71", "G72", "G75", "G90", "G92", "G94")


def register(replace: bool = True) -> None:
    """(Re-)register every cycle provider, the FS15 rigid tapping G codes and the alarm numbers."""
    codes = sorted(set(MILL_CODES) | set(FS15_CODES) | set(LATHE_CODES) | set(LATHE_ONLY) | {"G73", "G74", "G76"})
    for code in codes:
        mill = DrillCycle(code) if code in MILL_CODES or code in FS15_CODES else None
        hooks.register_cycle(code, Dispatch(code, mill, _lathe(code, False)), replace=replace)
    for code in sorted(set(codes) | set(HAAS_MILL_CODES) | set(HAAS_LATHE_CODES)):
        hooks.register_cycle(code, Dispatch(code, _haas_mill(code), _lathe(code, True)), dialect="haas",
                             replace=replace)
    for code in FS15_CODES:
        hooks.register_gcode(code, GSpec(9, "cycle", code, cycle=True), variant="M", dialect="fanuc", replace=replace)
    common.register_alarms(common.FANUC_CYCLE_ALARMS, common.HAAS_CYCLE_ALARMS)


register()

__all__ = ["Dispatch", "DrillCycle", "BoxCycle", "GrooveCycle", "ThreadCycle", "StockRemoval", "PatternRepeat",
           "Finishing", "BoltPattern", "register"]
