"""Built-in probe and tool-setter macro programs (FANUC custom macro B text), loaded as library programs
(docs/dialects.md "Cycles and compensation", probing).

The ``O98xx.nc`` files next to this module are macro programs in the FANUC macro language, written for
the twin in the style of the Renishaw Inspection Plus / tool-setting macros (the Renishaw programs
themselves are proprietary and were not used):

======  ===========================================================================================
O9810   protected positioning: a skip move that alarms PATH OBSTRUCTED when the probe triggers on the way
O9811   single surface (X, Y or Z): work offset (S) and tool length wear (T) updates, tolerance H
O9812   web / pocket width in X or Y: centre to the work offset (S), tolerance H on the width
O9814   bore / boss: four touches, centre to the work offset (S), tolerance H on the diameter
O9851   tool length on the tool setter: G37 (TOOL_SETTER, offset from the SetterResult), new tools searched
O9853   broken-tool check on the tool setter (skip moves on the setter signal), offsets unchanged
======  ===========================================================================================

The files are templates: ``{name}`` fields are filled at load time with the resolved parameters of
``db/models/controller-cycles.yaml`` (feeds, distances in mm, the alarm numbers from
``probe_alarm_base``, the setter position and skip number) and with the dialect's variable numbers
(FANUC: F #4109, H #4111, H geometry #2201-, H wear #2001-, D geometry #2601-, D wear #2401-; Haas NGC:
F #4106, H #4108, length geometry #2001-, wear #2201-, diameter geometry #2401-, wear #2601-, halved when
setting 40 keeps diameters). Values are written in mm and divided by 25.4 inside the macros under G20.
The macros run on any machining-centre variant (``machine.variant == "M"``); the stylus ball radius is
the radius (D) offset of the probe's tool number (geometry + wear), so a probe needs a calibrated D offset
(otherwise PROBE NOT CALIBRATED).

Loading (``ControllerSettings.builtin_macros``): ``None`` installs the probe macros when the machine has a
spindle probe and the setter macros when it has a tool setter with a known position; ``True`` installs the
probe macros on every machining centre (and the setter macros when the position is known); ``False``
installs none. A number the project already defines is left out (the project's program wins, as a control
keeps the program already in memory).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..program import ProgramSource

HERE = Path(__file__).resolve().parent
PROBE_PROGRAMS = ("O9810", "O9811", "O9812", "O9814")
SETTER_PROGRAMS = ("O9851", "O9853")
ALARM_OFFSETS = {"alarm_obstructed": 0, "alarm_open": 1, "alarm_calibration": 2, "alarm_format": 3,
                 "alarm_tolerance": 4, "alarm_broken": 5}


def _number(value: float) -> str:
    """A macro constant with a decimal point and no exponent (FANUC reads "2000." and "0.001")."""
    return f"{float(value):.6f}".rstrip("0")


def setter_position(machine, params: dict) -> list[float] | None:
    """Machine coordinates (SI) of the tool setter for the setter macros, or None when unknown."""
    value = params.get("setter_position_m", "machine")
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("setter_position_mm needs three values [x, y, z]")
        return [float(v) for v in value]
    if value != "machine":
        raise ValueError(f"setter_position_mm must be 'machine' or [x, y, z], not {value!r}")
    setter = machine.tool_setter
    if setter is None or setter.get("position_m") is None:
        return None
    return [float(v) for v in setter["position_m"]]


def template_values(machine, dialect, params: dict) -> dict:
    """The ``{name}`` fields of the macro templates for one machine and dialect."""
    haas = dialect.name == "haas"
    diameters = haas and bool(dialect.params.get("offsets_diameter"))
    base = int(round(float(params["probe_alarm_base"])))
    values = {
        "f_var": 4106 if haas else 4109,
        "h_var": 4108 if haas else 4111,
        "h_geom": 2000 if haas else 2200,
        "h_wear": 2200 if haas else 2000,
        "d_geom": 2400 if haas else 2600,
        "d_wear": 2600 if haas else 2400,
        "d_factor": "0.5" if diameters else "1.",
        "fast_feed_mm_min": _number(params["probe_fast_feed_m_s"] * 60e3),
        "measure_feed_mm_min": _number(params["probe_measure_feed_m_s"] * 60e3),
        "search_mm": _number(params["probe_search_m"] * 1e3),
        "backoff_mm": _number(params["probe_backoff_m"] * 1e3),
        "clearance_mm": _number(params["probe_clearance_m"] * 1e3),
        "count_mm": _number(machine.lin_increment_m * 1e3),
        "setter_search_mm": _number(params["setter_search_m"] * 1e3),
        "setter_clearance_mm": _number(params["setter_clearance_m"] * 1e3),
        "broken_tolerance_mm": _number(params["broken_tool_tolerance_m"] * 1e3),
        "setter_skip": int(round(float(params["setter_skip_signal"]))),
    }
    for key, offset in ALARM_OFFSETS.items():
        values[key] = base + offset
    position = setter_position(machine, params)
    if position is not None:
        values.update(setter_x_mm=_number(position[0] * 1e3), setter_y_mm=_number(position[1] * 1e3),
                      setter_z_mm=_number(position[2] * 1e3))
    values["setter_feed_mm_min"] = _number(params["setter_feed_m_s"] * 60e3)   # measuring feed of O9851/O9853
    return values


def program_text(name: str, values: dict) -> str:
    """One macro program with its fields filled (a missing field raises KeyError)."""
    text = (HERE / f"{name}.nc").read_text(encoding="ascii")
    if name in SETTER_PROGRAMS:
        values = dict(values, measure_feed_mm_min=values["setter_feed_mm_min"])
    return "%\n" + text.format(**values).strip() + "\n%\n"


def _defined(programs: list) -> set[int]:
    numbers: set[int] = set()
    for source in programs:
        for match in re.finditer(r"(?m)^\s*(?:O|:)0*(\d{1,5})(?!\d)", source.text):
            numbers.add(int(match.group(1)))
    return numbers


class Library:
    """The ``LIBRARIES`` provider of the built-in macros."""

    def selection(self, machine, settings) -> tuple[str, ...]:
        flag = getattr(settings, "builtin_macros", None)
        if machine.variant != "M" or flag is False:
            return ()
        probe = PROBE_PROGRAMS if flag or machine.spindle_probe else ()
        setter = SETTER_PROGRAMS if flag or machine.tool_setter is not None else ()
        return probe + setter

    def sources(self, machine, dialect, settings, programs: list) -> list[ProgramSource]:
        names = self.selection(machine, settings)
        if not names:
            return []
        from ..cycles.common import resolve

        params = resolve(getattr(settings, "cycle_parameters", None))
        values = template_values(machine, dialect, params)
        if "setter_z_mm" not in values:
            names = tuple(n for n in names if n not in SETTER_PROGRAMS)
        taken = _defined(programs)
        out = []
        for name in names:
            if int(name[1:]) in taken:
                continue
            out.append(ProgramSource(f"cnctwin-builtin/{name}.nc", program_text(name, values), path=-1, main=False))
        return out


LIBRARY = Library()

__all__ = ["Library", "LIBRARY", "PROBE_PROGRAMS", "SETTER_PROGRAMS", "ALARM_OFFSETS", "template_values",
           "program_text", "setter_position"]
