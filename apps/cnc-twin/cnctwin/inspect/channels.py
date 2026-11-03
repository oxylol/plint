"""Inspection's time-series channels: none.

Inspection evaluates the result geometry at operation ends and at the end of the run; it has no value per L0,
L1 or macro step to record (a value written once per operation into a macro channel would be NaN in every other
row). Its results are inspection.json, part_<op>.* files, the per-brick ``deviation_m`` attribute
(keyframes/attr) and findings (cnctwin.inspect.domain).
"""

from __future__ import annotations

CHANNELS: tuple = ()


def channels() -> list:
    return list(CHANNELS)


__all__ = ["CHANNELS", "channels"]
