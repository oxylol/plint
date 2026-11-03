"""Chip formation from the mechanics library (cnctwin.mechanics): the default FormationModel.

Per contact and step the mean conditions of the emission rows are binned (relative bins in h, speed and
uncut length, absolute bins in rake and inclination) and solved once per bin: Oxley's orthogonal
solution with the work material's Johnson-Cook flow stress gives the shear angle, chip ratio, strains,
contact length and temperatures (mechanics.solve); mechanics.chip_form gives the deformed chip, its curl,
breaking, ISO 3685 class, segmentation, BUE window and colour. The bin centre is solved, not the raw
query, so a result depends only on its bin (deterministic, identical after a resume).
"""

from __future__ import annotations

import math

from ..core import abi as core_abi
from ..core.errors import InputError
from ..units import REFERENCE_TEMP_K
from .protocols import Formation, FormationQuery

FORM_ELEMENTAL = 70


def _process(query: FormationQuery) -> str:
    if query.edge_kind in (core_abi.EDGE_LIP, core_abi.EDGE_CHISEL, core_abi.EDGE_MARGIN):
        return "drilling"
    if query.edge_kind == core_abi.EDGE_THREAD_FLANK:
        return "tapping"
    if query.continuous and query.edge_kind in (core_abi.EDGE_NOSE, core_abi.EDGE_INSERT, core_abi.EDGE_WIPER):
        return "turning"
    return "milling"


class MechanicsFormation:
    """FormationModel backed by cnctwin.mechanics (Oxley + chip form). ``tool_sides`` maps tool slots to
    mechanics.ToolSide (chip breaker, flute curl, tool friction); without one the flat-rake model is used."""

    def __init__(self, material, tool_sides: dict | None = None, ambient_K: float = REFERENCE_TEMP_K,
                 bin_rel: float = 0.02, bin_rad: float = 0.01):
        from ..mechanics import WorkMaterial
        try:
            self.work = WorkMaterial.from_db(material)
        except (ValueError, KeyError, TypeError) as error:
            raise InputError(f"chips: the chip formation model cannot read the work material ({error}); wire "
                             "ChipsDomain(formation=...) or complete the material file") from None
        if not self.work.has_flow_stress:
            raise InputError(f"chips: material {self.work.id!r} has no flow-stress law, which the chip formation "
                             "model needs (wire ChipsDomain(formation=...) for it)")
        self.tool_sides = dict(tool_sides or {})
        self.ambient_K = float(ambient_K)
        self.log_bin = math.log1p(float(bin_rel))
        self.bin_rad = float(bin_rad)
        self._cache: dict[tuple, Formation] = {}

    def _rel(self, x: float) -> tuple[int, float]:
        if x <= 0.0 or not math.isfinite(x):
            return 0, 0.0
        k = int(round(math.log(x) / self.log_bin))
        return k, math.exp(k * self.log_bin)

    def _abs(self, x: float) -> tuple[int, float]:
        k = int(round(x / self.bin_rad)) if math.isfinite(x) else 0
        return k, k * self.bin_rad

    def formation(self, query: FormationQuery) -> Formation:
        process = _process(query)
        kh, h = self._rel(query.h_m)
        kv, v = self._rel(query.v_m_s)
        kb, b = self._rel(query.b_m)
        kl, length = self._rel(query.l_m)
        ka, rake = self._abs(query.rake_rad)
        ki, incl = self._abs(query.incl_rad)
        key = (query.tool_slot, process, query.continuous, kh, kv, kb, kl, ka, ki)
        found = self._cache.get(key)
        if found is None:
            found = self._solve(query.tool_slot, process, query.continuous, h, v, b, length, rake, incl)
            self._cache[key] = found
        return found

    def _solve(self, slot, process, continuous, h, v, b, length, rake, incl) -> Formation:
        from ..mechanics import ChipConditions, ChipState, chip_form, solve
        from ..mechanics.chipform import NO_CHIP
        tool = self.tool_sides.get(slot)
        if h <= 0.0 or v <= 0.0:
            return self._no_chip(h, b)
        result = solve(self.work, h, v, rake, tool=tool)
        state = ChipState.from_orthogonal(result)
        cond = ChipConditions(h_m=h, width_m=b, speed_m_s=v, rake_rad=rake, incl_rad=incl,
                              cut_length_m=None if continuous else length, process=process)
        form = chip_form(self.work, tool, cond, state, ambient_K=self.ambient_K)
        if form.form_code == NO_CHIP or not (state.chip_ratio > 0.0):
            return self._no_chip(h, b)
        piece = form.length_m if math.isfinite(form.length_m) else 0.0
        curl = form.curl_radius_m if math.isfinite(form.curl_radius_m) else 0.0
        return Formation(shear_angle_rad=state.shear_angle_rad, chip_ratio=state.chip_ratio,
                         shear_strain=state.shear_strain, strain_rate_1_s=state.shear_strain_rate_1_s,
                         chip_flow_angle_rad=form.chip_flow_angle_rad, chip_temp_K=form.temp_K,
                         form_code=int(form.form_code), bue_active=bool(form.bue_window),
                         segmented=bool(form.segmented), thickness_m=form.thickness_m, width_m=form.width_m,
                         length_m=piece, curl_radius_m=curl, colour_rgb=tuple(int(c) for c in form.colour_rgb))

    def _no_chip(self, h: float, b: float) -> Formation:
        """Below the minimum chip thickness no chip leaves the edge; the removed volume the field still reports
        (ploughed-off debris) leaves as elemental fragments of the uncut thickness."""
        t = max(h, 1e-7)
        return Formation(shear_angle_rad=0.0, chip_ratio=1.0, shear_strain=0.0, strain_rate_1_s=0.0,
                         chip_flow_angle_rad=0.0, chip_temp_K=self.ambient_K, form_code=FORM_ELEMENTAL,
                         bue_active=False, segmented=False, thickness_m=t, width_m=max(b, t), length_m=max(b, t),
                         curl_radius_m=0.0, colour_rgb=(128, 128, 128))
