"""Inputs the harnesses assume when a case does not state them (one place, each with its reason).

These are not model coefficients (those are in db/models) and were not chosen by comparing with any
reference: they describe the test set-ups the sources leave open. Every use is listed in the report
as an assumed input of the case.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumed:
    value: float
    unit: str
    reason: str


# cutting time of an orthogonal or turning test whose duration the source does not state: long enough
# for the stable phase the sources report, short against the insert's heating time (tens of seconds)
TEST_CUT_TIME = Assumed(5.0, "s", "duration of a turning test not stated: 5 s of stable cutting assumed")

# the tool ladder is stepped with this time step (implicit; the edge partition is re-evaluated each step)
LADDER_STEP = Assumed(0.25, "s", "time step of the transient tool-body solution (backward Euler, conservative)")

# IR window of slusarczyk-2025: on the tool's side face at the lateral end of the chip contact,
# 2 mm along the rake face from the edge by 1 mm down the flank (Sect. 3.2 of the source: 2 x 1 mm)
IR_WINDOW_ALONG_RAKE = Assumed(2.0e-3, "m", "window length along the rake face (source: 2 x 1 mm window)")
IR_WINDOW_DOWN_FLANK = Assumed(1.0e-3, "m", "window height down the flank (source: 2 x 1 mm window)")

# the chip cools in still air after it leaves the tool (tempering colours): free convection around a
# horizontal cylinder of the chip width, plus radiation at the steel's emissivity
CHIP_COOLING_LENGTH = "chip width"

# depth below the rake face of the embedded thermocouple of guimaraes-2023 (the case: 'about 0.2-0.4 mm
# below the rake surface, est. from the groove depth'): the middle of that range
EMBEDDED_TC_DEPTH = Assumed(0.3e-3, "m", "case: 0.2-0.4 mm below the rake surface; the middle")

# Guimaraes et al. give the time to the maximum for 70, 130 and 160 m/min only; the 40 m/min pass time
# follows from the pass length, the feed and the bar diameter implied by the 70 m/min test
# (its maximum taken at the end of its pass): D = vc t f / (pi L)
GUIMARAES_DIAMETER_FROM = "vc 70 m/min, 86.2 s"

# Natural-thermocouple and IR readings average over the stable phase; the reported value is the
# simulated temperature at the end of the stated (or assumed) cutting time
READING_AT = "end of the cutting time"


__all__ = ["Assumed", "TEST_CUT_TIME", "LADDER_STEP", "IR_WINDOW_ALONG_RAKE", "IR_WINDOW_DOWN_FLANK",
           "CHIP_COOLING_LENGTH", "EMBEDDED_TC_DEPTH", "GUIMARAES_DIAMETER_FROM", "READING_AT"]
