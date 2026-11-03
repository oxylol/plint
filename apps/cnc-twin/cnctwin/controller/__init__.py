"""The CNC control: reads the program in its dialect and executes it the way the control would.

Public API (docs/dialects.md "Structure of the controller"):

* :class:`Controller` (``controller.py``): one executor per path; ``produce/retire/feedback/cycle_start``
* :class:`ControllerSetup`, :class:`ToolOffsetInit`: what a run gives the control
* :class:`ControllerState` (``state.py``): all executor state as data (``to_arrays``/``from_arrays``)
* :func:`dry_run` (``dryrun.py``): the program against an ideal machine
* :mod:`hooks`: extension points (cycles, compensation, transforms, probing, tool life, restart)
* :mod:`dialects`: the dialect registry (FANUC 0i-F, Haas NGC; others register themselves)
* :mod:`ir`: the commands handed to motion and the feedback it returns
"""

from . import hooks, ir
from .controller import Controller, ControllerSetup, ToolOffsetInit
from .dryrun import DryRunResult, dry_run
from .program import ProgramSource
from .schema import ControllerSettings, settings_from
from .state import ControllerState

__all__ = ["Controller", "ControllerSetup", "ToolOffsetInit", "ControllerState", "ControllerSettings",
           "settings_from", "ProgramSource", "dry_run", "DryRunResult", "hooks", "ir"]
