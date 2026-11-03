"""Motion domain: look-ahead planner, interpolation, feed drives, spindles, kinematics and timing of
every controller path (docs/models/motion.md).

* :class:`MotionDomain` - the SimState field ``motion``, kernel inputs for the compiled loop, the bus
  record ``motion`` (PathFrame per path), the timeline and cycle-time ledger;
* :class:`MachineModel` - axes, drives, paths, chains, spindles and the ISO 230-1 error map of a
  machine file; :class:`PathKinematics` - FK/IK, singularities, tilted-plane solutions;
* kernels in :mod:`cnctwin.motion.kernels` (motion_tick, run_motion, fk_poses, plan_ticks, ...),
  layouts in :mod:`cnctwin.motion.abi`;
* :class:`MotionRunner` - the controller and motion in lock-step without cutting.

Names are imported on first use, so ``cnctwin.motion.schema`` (read by the project schema) does not
compile the kernels.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "MotionDomain": "domain", "MotionEnv": "domain", "MachineModel": "machine", "PathKinematics": "ik",
    "MotionRunner": "driver", "MotionSettings": "schema", "MotionState": "state", "Timeline": "ledger",
    "double_s": "scurve", "circle_radius_ratio": "servo",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'cnctwin.motion' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
