"""HEIDENHAIN TNC 640 dialects (docs/dialects.md "Heidenhain TNC (Klartext and ISO)").

Registers ``heidenhain_klartext`` and ``heidenhain_iso`` (the machine file's ``controller.dialect``; a
program file starting with ``%`` is read as ISO, one with ``BEGIN PGM`` as Klartext, whichever the
machine's dialect): parser (``klartext.build_store``), handler (``handler.HeidenhainHandler``), cycles,
PLANE / TCPM stages (registered by the Siemens package's kit for both dialects) and the NC error table
(texts from db/controllers/heidenhain-tnc640.yaml). Loaded by ``cnctwin.controller.hooks.load_plugins``.
"""

from __future__ import annotations

import functools

from ...alarms import AlarmTable
from ...lexer import LexerRules
from ..base import Dialect, register_dialect
from ..siemens840d import NoNumberedVariables, native_params
from ..siemens840d.kit import register_tcp_stages
from . import cycles  # noqa: F401  (registers the cycle step resumer)
from .handler import HeidenhainHandler
from .klartext import build_store
from .tables import ALARM_CODES, ALARM_TEXTS, WORD_LIST

POWER_ON = {"7": "G40", "distance": "G90", "feed_mode": "per_min", "plane": "G17", "iso_motion": "G00",
            "motion": "G00"}


def _builder(name: str, syntax: str):
    def build(machine, settings, controller_db, db_id: str = "") -> Dialect:
        params = native_params(settings, "controller-heidenhain")
        params["arc_tolerance_m"] = float(params["arc_end_point_tolerance_m"])
        params["sub_nesting_max"] = int(params["sub_nesting_levels"])
        params["macro_nesting_max"] = int(params["sub_nesting_levels"])
        params["max_m_codes_per_block"] = int(params["max_m_codes_per_block"])
        alarms = AlarmTable(name, ALARM_CODES, ALARM_TEXTS, controller_db, db_id)
        parser = functools.partial(build_store, syntax=syntax)
        power_on = dict(POWER_ON, plane="G18" if machine.variant == "T" else "G17")
        return Dialect(name, machine.variant, LexerRules(), {}, power_on, alarms, NoNumberedVariables(),
                       HeidenhainHandler, params, controller_db, db_id, None, None, frozenset(), {"words": WORD_LIST},
                       parser)
    return build


register_dialect("heidenhain_klartext", _builder("heidenhain_klartext", "klartext"), replace=True)
register_dialect("heidenhain_iso", _builder("heidenhain_iso", "iso"), replace=True)
register_tcp_stages()

__all__ = ["HeidenhainHandler"]
