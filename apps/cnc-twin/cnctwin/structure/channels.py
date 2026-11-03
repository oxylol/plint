"""Time series the structure domain records (macro tier; values from StructureDomain.macro_values).

Per-tool channels describe the tool of the first path that has one (the active tool); per-slot channels cover
every tool of the table (up to MAX_SLOTS); per-part channels have one column per workpiece (up to MAX_PARTS)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_PARTS = 8
MAX_SLOTS = 32

_DEFS = (
    ("structure.tool_f1_Hz", "Hz", "first natural frequency of the active tool-holder-spindle assembly", ()),
    ("structure.tool_tip_compliance_m_per_N", "m/N", "static compliance at the active tool's tip along T0 x", ()),
    ("structure.tool_fit_error", "-", "modal vs RCSA tip receptance error of the active tool", ()),
    ("structure.tool_f1_slot_Hz", "Hz", "first natural frequency per tool slot", (MAX_SLOTS,)),
    ("structure.work_f1_Hz", "Hz", "first natural frequency of each workpiece as held", (MAX_PARTS,)),
    ("structure.work_dof", "-", "degrees of freedom of each workpiece model", (MAX_PARTS,)),
    ("structure.ritz_residual", "-", "eigen-residual of the last Ritz refresh of each workpiece", (MAX_PARTS,)),
    ("structure.displacement_max_m", "m", "largest |d| (thermal + clamp + residual) of each workpiece", (MAX_PARTS,)),
    ("structure.thermal_max_m", "m", "largest thermal displacement of each workpiece", (MAX_PARTS,)),
    ("structure.clamp_max_m", "m", "largest clamping displacement of each workpiece", (MAX_PARTS,)),
    ("structure.residual_max_m", "m", "largest residual-stress distortion of each workpiece", (MAX_PARTS,)),
    ("structure.fixture_compliance_m_per_N", "m/N", "largest fixture compliance at each part's datum", (MAX_PARTS,)),
    ("structure.jobs_pending", "-", "refresh jobs started and not yet applied", ()),
)


def channels() -> list[Channel]:
    return [Channel(name, unit, Tier.MACRO, "structure", doc, shape) for name, unit, doc, shape in _DEFS]


NAMES = tuple(d[0] for d in _DEFS)

__all__ = ["channels", "NAMES", "MAX_PARTS", "MAX_SLOTS"]
