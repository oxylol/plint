"""Mechanics library: chip formation, heat generation and partition, chip form and colour.

Pure functions and precomputed tables (docs/models/mechanics.md); every domain may import it.

* :mod:`.oxley` - Oxley's predictive machining theory with the Johnson-Cook flow stress
  (orthogonal), its fallbacks, and a SciPy reference solver.
* :mod:`.oblique` - Stabler's flow rule, equivalent rake, Budak-Altintas-Armarego coefficients.
* :mod:`.contact` - edge ploughing, indentation/extrusion, flank wear land, Zorev split.
* :mod:`.heat` - heat generation by zone and partition (Boothroyd, Loewen-Shaw, Blok/Jaeger).
* :mod:`.coefficients` - process maps and coefficient tables for cutting's pipeline, calibration
  to measured Kienzle/mechanistic data.
* :mod:`.chipform`, :mod:`.colour` - chip geometry, curl, breaking, ISO 3685 class, segmentation,
  BUE, ejection, oxide colour.
* :mod:`.kernels` - the Numba kernels (Oxley solver, map lookups, edge contact, oxide, BUE).
"""

from .abi import ProcessMap
from .chipform import ChipConditions, ChipForm, ChipState, bue_advance, bue_in_window, chip_form, is_segmented
from .coefficients import (Calibration, EdgeGroup, GroupCoefficients, SlowState, build_calibration,
                           build_process_map, coeff_table, flank_coefficients, group_coefficients, lookup,
                           map_matches, process_coefficients)
from .colour import chip_colour, emitted_chip_colour
from .contact import edge_forces, flank_stresses, indentation_forces, min_chip_thickness, ploughing_forces
from .materials import ToolSide, WorkMaterial
from .oblique import cutting_coefficients, equivalent_rake, stabler_flow_angle
from .oxley import OrthogonalResult, solve, solve_reference, zorev_split

__all__ = ["ProcessMap", "ChipConditions", "ChipForm", "ChipState", "bue_advance", "bue_in_window", "chip_form",
           "is_segmented", "Calibration", "EdgeGroup", "GroupCoefficients", "SlowState", "build_calibration",
           "build_process_map", "coeff_table", "flank_coefficients", "group_coefficients", "lookup", "map_matches",
           "process_coefficients", "chip_colour", "emitted_chip_colour", "edge_forces", "flank_stresses",
           "indentation_forces", "min_chip_thickness", "ploughing_forces", "ToolSide", "WorkMaterial",
           "cutting_coefficients", "equivalent_rake", "stabler_flow_angle", "OrthogonalResult", "solve",
           "solve_reference", "zorev_split"]
