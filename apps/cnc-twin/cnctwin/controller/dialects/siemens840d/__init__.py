"""SINUMERIK 840D sl dialect (docs/dialects.md "Siemens 840D sl").

Registers the dialect ``siemens840d`` (the machine file's ``controller.dialect``): its own parser
(``syntax.build_store``), the handler (``handler.SiemensHandler``), the technology cycles
(``cycles``, ``turning``), the alarm table (840D sl numbers, texts from db/controllers/siemens-840d-sl.yaml)
and the TCP / swivel-plane stages for table-type five-axis machines (``kit.register_tcp_stages``).
Loaded by ``cnctwin.controller.hooks.load_plugins``.
"""

from __future__ import annotations

import functools

from ...alarms import AlarmTable, ControllerAlarm
from ...lexer import LexerRules
from ..base import Dialect, register_dialect
from . import cycles  # noqa: F401  (registers the cycle step resumer)
from .handler import SiemensHandler
from .kit import Alarm, register_tcp_stages
from .syntax import build_store
from .tables import ALARM_CODES, ALARM_TEXTS, POWER_ON_LATHE, POWER_ON_MILL, WORD_LIST

SETTINGS_KEYS = ("m_codes", "reference_points_m", "power_on", "power_on_units", "clock_start", "macro_statement_s",
                 "macro_term_s", "call_s", "goto_scan_s", "horizon_s", "sub_nesting_max", "max_m_codes_per_block",
                 "interface_inputs", "cycle_parameters", "tool_life")


class NoNumberedVariables:
    """The 840D sl has no numbered macro variables (#n); providers that read them get TW0012."""

    def read(self, ex, n: int):
        raise Alarm("unsupported_function", f"#{n}: numbered variables do not exist in the Siemens dialect")

    def write(self, ex, n: int, value) -> None:
        raise Alarm("unsupported_function", f"#{n}: numbered variables do not exist in the Siemens dialect")

    def adp(self, ex, n: int, value):
        raise Alarm("unsupported_function", "ADP does not exist in the Siemens dialect")


def native_params(settings, model: str) -> dict:
    """Block-time model + the dialect's model file + project settings (SI)."""
    from ....db.models import params as model_params

    values = dict(model_params("controller-block-time"))
    values.update(model_params(model))
    resolved = dict(values)
    resolved.update({"m_codes": {}, "reference_points_m": {}, "power_on": {}, "interface_inputs": {},
                     "cycle_parameters": {}, "clock_start": "2026-01-05T07:00:00", "power_on_units": "mm",
                     "macro_nesting_max": int(values.get("sub_nesting_levels", 16)),
                     "sub_nesting_max": int(values.get("sub_nesting_levels", 16)),
                     "shared_common_100": 0, "shared_common_500": 0, "nat": False, "pocket_calculator": False})
    if settings is not None:
        for name in SETTINGS_KEYS:
            value = getattr(settings, name, None)
            if value is None or (isinstance(value, dict) and not value):
                continue
            resolved[name] = value
        if getattr(settings, "arc_tolerance_m", None) is not None:
            resolved["circle_error_const_m"] = float(settings.arc_tolerance_m)
    resolved["arc_tolerance_m"] = float(resolved.get("circle_error_const_m", resolved.get("arc_tolerance_m", 0.01e-3)))
    return resolved


def build_siemens(machine, settings, controller_db, db_id: str = "") -> Dialect:
    variant = machine.variant
    params = native_params(settings, "controller-siemens840d")
    params["r_parameters"] = int(params["r_parameters"])
    params["structure_nesting_max"] = int(params["structure_nesting_levels"])
    params["max_m_codes_per_block"] = int(params["max_m_codes_per_block"])
    axes = frozenset(a.letter for path in machine.paths for a in path)
    parser = functools.partial(build_store, axes=axes, max_depth=params["structure_nesting_max"])
    alarms = AlarmTable("siemens840d", ALARM_CODES, ALARM_TEXTS, controller_db, db_id)
    power_on = dict(POWER_ON_LATHE if variant == "T" else POWER_ON_MILL)
    return Dialect("siemens840d", variant, LexerRules(), {}, power_on, alarms, NoNumberedVariables(), SiemensHandler,
                   params, controller_db, db_id, None, None, frozenset(), {"words": WORD_LIST}, parser)


register_dialect("siemens840d", build_siemens, replace=True)
register_tcp_stages()

__all__ = ["build_siemens", "native_params", "SiemensHandler", "ControllerAlarm"]
