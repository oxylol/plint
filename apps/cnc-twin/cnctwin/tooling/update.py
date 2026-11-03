"""Per-macro-step composition of the effective EdgeModel (architecture 4.1; docs/models/tooling.md
"Composition").

Inputs come from the previous step's bus: WearFeedback (radial / axial wear and worn hone radius
per element, broken), ThermalFeedback (radial growth per element, tool length growth, holder
growth) and HoldingFeedback (pull-out, slip). :func:`edge_state` turns them into the
:class:`cnctwin.tooling.abi.EdgeState` of one tool, :func:`effective_edge` applies it with the
compiled :func:`cnctwin.tooling.kernels.compose_effective`.
"""

from __future__ import annotations

import numpy as np

from ..core.abi import EdgeModel
from ..core.errors import TwinError
from .abi import (N_STATE, STATE_FREE_LENGTH_M, STATE_HOLDER_GROWTH_M, STATE_LENGTH_GROWTH_M, STATE_PULLOUT_M,
                  STATE_ROTATING, STATE_SHANK_DIR_R, STATE_SHANK_DIR_Z, STATE_SLIP_RAD, EdgeState)
from .edges import edge_trig, frames_T
from .kernels import compose_effective


def _per_element(value, shape, name: str, slot: int) -> np.ndarray:
    if value is None:
        return np.zeros(shape)
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros(shape)
    if arr.shape != shape:
        raise TwinError(f"tool slot {slot}: {name} has shape {arr.shape}, the EdgeModel is {shape}")
    return np.ascontiguousarray(arr.copy())


def free_length(geometry) -> float:
    """Length over which the tool's thermal strain acts: holder nose to tip (rotating tools), the
    holder or bar overhang (non-rotating)."""
    props = geometry.props
    if geometry.rotating:
        return float(props.stickout_m) if props.stickout_m else 0.0
    overhang = geometry.info.get("overhang")
    return float(overhang) if overhang else 0.0


def edge_state(geometry, wear=None, heat=None, holding=None, initial_edge_radius_m: float | None = None) -> EdgeState:
    """EdgeState of one tool from its ToolWear, ToolHeat and ToolHolding records (None = absent)."""
    shape = geometry.edge.elem_r_m.shape
    slot = geometry.slot
    radial_wear = _per_element(getattr(wear, "radial_wear_m", None), shape, "radial_wear_m", slot)
    axial_wear = _per_element(getattr(wear, "axial_wear_m", None), shape, "axial_wear_m", slot)
    edge_radius = _per_element(getattr(wear, "edge_radius_m", None), shape, "edge_radius_m", slot)
    if wear is None and initial_edge_radius_m:
        edge_radius = np.full(shape, float(initial_edge_radius_m))
    growth = _per_element(getattr(heat, "radial_growth_m", None), shape, "radial_growth_m", slot)
    scalars = np.zeros(N_STATE)
    scalars[STATE_LENGTH_GROWTH_M] = float(getattr(heat, "length_growth_m", 0.0) or 0.0)
    scalars[STATE_HOLDER_GROWTH_M] = float(getattr(heat, "holder_growth_m", 0.0) or 0.0)
    scalars[STATE_PULLOUT_M] = float(getattr(holding, "pullout_m", 0.0) or 0.0)
    scalars[STATE_SLIP_RAD] = float(getattr(holding, "slip_rad", 0.0) or 0.0)
    scalars[STATE_FREE_LENGTH_M] = free_length(geometry)
    direction = geometry.props.shank_dir_T
    # shank direction in (e_r at psi = 0, z_T) = (x_T, z_T) components
    scalars[STATE_SHANK_DIR_R] = float(direction[0])
    scalars[STATE_SHANK_DIR_Z] = float(direction[2])
    scalars[STATE_ROTATING] = 1.0 if geometry.rotating else 0.0
    broken = np.array([1 if bool(getattr(wear, "broken", False)) else 0], np.int8)
    return EdgeState(radial_wear, axial_wear, edge_radius, growth, scalars, broken)


def copy_edge(edge: EdgeModel) -> EdgeModel:
    return EdgeModel(*(np.ascontiguousarray(np.array(getattr(edge, name), copy=True)) for name in edge._fields))


def effective_edge(nominal: EdgeModel, state: EdgeState, out: EdgeModel | None = None) -> EdgeModel:
    out = copy_edge(nominal) if out is None else out
    compose_effective(nominal, state, out)
    return out


def rephase(geometry, edge: EdgeModel):
    """Angle table and element frames of an (effective) edge whose tooth phases may have moved."""
    return edge_trig(edge), frames_T(geometry.elements, edge.tooth_phase_rad, float(edge.hand_sign[0]))
