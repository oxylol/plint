"""Cutter radius compensation (G41/G42/G40, FANUC type C), lathe tool nose radius compensation and
optional chamfering / corner rounding (,C ,R) for the controller core (docs/dialects.md "Cycles and
compensation"). Registers the ``cutter``, ``nose`` and ``corner`` compensation kinds, the provider of
the I J K words of G00/G01 blocks in compensation (IJ type vector, G40 I J K) and the alarm numbers.
"""

from __future__ import annotations

from .. import hooks
from ..cycles import common
from .cutter import CompFactory, CompFilter, Frame2D
from . import geometry

FANUC_ALARMS = {
    "crc_no_intersection": ("PS0033", "NO INTERSECTION AT CUTTER COMPENSATION"),
    "crc_arc_startup": ("PS0034", "NO CIRC ALLOWED IN STUP/EXT BLK"),
    "crc_interference": ("PS0041", "INTERFERENCE IN CUTTER COMPENSATION"),
    "crc_g39_motion": ("PS5445", "CAN NOT COMMAND MOTION IN G39"),
    "chamfer_in_thread": ("PS0050", "CHF/CNR NOT ALLOWED IN THRD BLK"),
    "chamfer_missing_move": ("PS0051", "MISSING MOVE AFTER CNR/CHF"),
    "chamfer_not_g01": ("PS0052", "CODE IS NOT G01 AFTER CHF/CNR"),
    "chamfer_too_large": ("PS0055", "MISSING MOVE VALUE IN CHF/CNR"),
}
HAAS_ALARMS = {
    "crc_no_intersection": ("TW0026", "CUTTER COMPENSATION: NO INTERSECTION"),
    "crc_arc_startup": ("TW0027", "CUTTER COMPENSATION START OR END ON AN ARC"),
    "crc_interference": ("TW0025", "CUTTER COMPENSATION INTERFERENCE"),
    "crc_g39_motion": ("TW0027", "CUTTER COMPENSATION START OR END ON AN ARC"),
    "chamfer_in_thread": ("TW0031", "CHAMFER / CORNER ROUNDING ERROR"),
    "chamfer_missing_move": ("TW0031", "CHAMFER / CORNER ROUNDING ERROR"),
    "chamfer_not_g01": ("TW0031", "CHAMFER / CORNER ROUNDING ERROR"),
    "chamfer_too_large": ("TW0031", "CHAMFER / CORNER ROUNDING ERROR"),
}


class CompWords:
    """I J K in a G00/G01 block while cutter / nose compensation is active (also in the G40 block that
    cancels it): the IJ type vector, taken in the plane of compensation (FANUC 6.6.3, 6.6.4)."""

    def motion_words(self, ex, code: str, bw) -> dict:
        s = ex.s
        if s.comp_state is None or s.comp_kind not in ("cutter", "nose") or code not in ("G00", "G01"):
            return {}
        if s.tcp == "fanuc:G43.5":
            return {}
        out = {}
        for letter in ("I", "J", "K"):
            word = bw.take(letter)
            if word is not None:
                out[letter] = ex.handler.lin(word)
        return {"comp_ijk": out} if out else {}


def register(replace: bool = True) -> None:
    """(Re-)register the providers (tests that replace them restore them with this)."""
    hooks.register_compensation("cutter", CompFactory("cutter"), replace=replace)
    hooks.register_compensation("nose", CompFactory("nose"), replace=replace)
    hooks.register_compensation("corner", CompFactory("corner"), replace=replace)
    hooks.register_motion_words("comp", CompWords(), replace=replace)
    common.register_alarms(FANUC_ALARMS, HAAS_ALARMS)


register()

__all__ = ["CompFactory", "CompFilter", "CompWords", "Frame2D", "geometry", "register"]
