"""The faces of the part that a setup locates or clamps on, in S (docs/models/holding.md "Chips and burrs
under locators").

:class:`HoldingLocators` implements chips' ``LocatorFaces`` protocol: ``faces_S(op_index)`` gives the patches
of the **next** operation's setup (op_index + 1), so chips measures the chips and burrs left on them at the end
of operation ``op_index`` (``ChipFeedback.locator_debris_m``, keyed by these ids); holding reads that record
when it builds the next setup and turns the heights into the location error. The ids are holding's patch ids
(``<op>/<entry>:<device>/<role>``).
"""

from __future__ import annotations

import numpy as np

from . import devices as D
from . import params as P
from . import setup as SU


class HoldingLocators:
    """LocatorFaces over a project's setups (devices only: no clamping solve)."""

    def __init__(self, project, geometry, settings):
        self.project = project
        self.geometry = geometry
        self.settings = settings

    def faces_for_op(self, op_index: int) -> list:
        """(id, centre_S, outward normal_S, half extents [2]) of operation op_index's locating and clamping
        patches at the nominal part placement."""
        op = SU.operation(self.project, op_index)
        if op is None:
            return []
        entries = SU.setup_entries(op)
        if not entries:
            return []
        w = SU.op_workpiece(op)
        T_H_S = SU.stock_pose(op)
        shape = self.geometry.shape_S(w)
        mat = SU.material_props(P.field_of(self.project, "material"))
        bc = D.BuildCtx(op_name=P.text(op, "name", f"op{op_index}"), workpiece=w, shape=shape, T_H_S=T_H_S,
                        n_side=self.settings.patch_nodes, centre_nodes=self.settings.centre_nodes,
                        material_family=mat["family"], ultimate_Pa=mat["ultimate"], youngs_Pa=mat["youngs"],
                        poisson_work=mat["poisson"], machine=P.field_of(self.project, "machine"))
        patches, _, devices, _, _ = SU.build_patches(bc.op_name, entries, bc, {})
        patches.extend(SU.auto_base(bc.op_name, bc, devices, patches))
        out = []
        for patch in patches:
            if patch.locating or patch.law == D.LAW_LOCKED:
                pid, c, n, half = SU.patch_face_S(patch, T_H_S)
                out.append((pid, np.asarray(c, float), np.asarray(n, float), np.asarray(half, float)))
        return out

    def faces_S(self, op_index: int) -> list:
        """chips' LocatorFaces: the faces the next setup (op_index + 1) clamps or locates on."""
        return self.faces_for_op(int(op_index) + 1)


__all__ = ["HoldingLocators"]
