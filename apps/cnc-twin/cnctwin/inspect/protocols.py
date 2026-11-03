"""What inspection needs from other parts, as small Protocols the integration wires (docs/models/inspect.md
"Integration"), with complete default implementations for runs and tests that are not wired.

* :class:`ResultSource` - the :class:`~cnctwin.inspect.result.ResultPart` of an operation: removal's mesh of the
  field (material configuration, ``surface_event`` vertex attribute) and structure's displacement grids, taken
  **before** removal transfers the field to the next operation. Default :class:`StaticResults`.
* :class:`FieldColumns` - removal's level-0 z rays of the field and of the CAD target as
  :class:`~cnctwin.inspect.abi.Columns` (exact remaining-material / gouge comparison). Default
  :class:`MeshColumns` (both dexelised from the result mesh and the CAD on a common lattice).
* :class:`SurfaceResults` - surface.json ``features`` (roughness per tolerance id) and ``threads``. Default
  :class:`SurfaceJson` (a document or a run directory's surface.json).
* :class:`SurfaceEventLookup` - per surface event id the generating tool slot and the relative tool-work
  deflection at generation (cutting's SURFACE_EVENT rows ``u_rel``), for the error budget. Default
  :class:`NoEvents`.
* :class:`StockFieldBuilder` - the next operation's dexel field from this operation's result: removal's exact
  remap for a lattice symmetry, or rasterising a closed mesh. Default :class:`ColumnsStockBuilder` (z columns).
* :class:`TipResolver` and :class:`CommandSource` - tool reference points of the controller's commands and the
  command stream of an operation. Default :class:`CartesianTips` (sanity's 3-axis convention) and
  :class:`StaticCommands`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..controller.ir import Command, Space
from ..geom.mesh import Mesh


@runtime_checkable
class ResultSource(Protocol):
    def result_part(self, ctx, state, op_index: int, workpiece: int):
        """The ResultPart of an operation and workpiece (None when the workpiece has none)."""


@runtime_checkable
class FieldColumns(Protocol):
    def columns(self, workpiece: int, result_mesh: Mesh | None = None) -> tuple:
        """(field Columns, target Columns or None) of a workpiece on one lattice."""


@runtime_checkable
class SurfaceResults(Protocol):
    def features(self) -> list[dict]:
        """surface.json ``features`` (one per roughness tolerance id)."""

    def threads(self) -> list[dict]:
        """surface.json ``threads`` (measured threads)."""


@runtime_checkable
class SurfaceEventLookup(Protocol):
    def events(self, ids: np.ndarray) -> dict:
        """For surface event ids [n]: ``{"tool_slot": [n] int (-1 unknown), "u_rel_m": [n, 3]}``."""


@runtime_checkable
class StockFieldBuilder(Protocol):
    def remap(self, workpiece: int, rotation: np.ndarray, kind: str) -> Any:
        """The field turned by a lattice symmetry (exact), or None when the builder cannot remap."""

    def from_mesh(self, workpiece: int, vertices_S_m: np.ndarray, faces: np.ndarray, lattice_m: float,
                  box_lo_m: np.ndarray, box_hi_m: np.ndarray) -> Any:
        """A field rasterised from a closed mesh in the new stock frame."""


@runtime_checkable
class TipResolver(Protocol):
    def tip_W(self, command: Command, previous_W: np.ndarray | None) -> np.ndarray | None:
        """End tool-tip position of a motion command in W (None when unknown)."""

    def point_W(self, command: Command, point) -> np.ndarray:
        """A point given in the command's space (an arc centre) in W."""

    def axis_W(self, command: Command) -> np.ndarray:
        """Tool axis (+z_T) in W at the command's end."""


@runtime_checkable
class CommandSource(Protocol):
    def commands(self, op_index: int) -> list:
        """The executed controller commands of an operation (cnctwin.controller.ir.Command)."""


# ============================================================================ defaults

class StaticResults:
    """ResultParts given up front: ``{(op_index, workpiece): ResultPart}``."""

    def __init__(self, parts: dict | None = None):
        self.parts = dict(parts or {})

    def add(self, part) -> None:
        self.parts[(int(part.op_index), int(part.workpiece))] = part

    def result_part(self, ctx, state, op_index: int, workpiece: int):
        return self.parts.get((int(op_index), int(workpiece)))

    def workpieces(self, op_index: int) -> list[int]:
        return sorted(w for (o, w) in self.parts if o == op_index)


class MeshColumns:
    """Field and target columns by dexelising the result (material configuration) and the CAD meshes at
    ``cell_m`` (default: cnctwin.inspect.columns.suggested_cell)."""

    def __init__(self, cad_mesh: Mesh, cell_m: float | None = None, merge_m: float = 1e-9, cad_tree=None):
        self.cad_mesh = cad_mesh
        self.cell_m = cell_m
        self.merge_m = merge_m
        self.cad_tree = cad_tree

    def columns(self, workpiece: int, result_mesh: Mesh | None = None) -> tuple:
        from .columns import columns_from_mesh, lattice_for, suggested_cell

        if result_mesh is None:
            raise ValueError("MeshColumns needs the result mesh")
        cell = self.cell_m or suggested_cell(result_mesh)
        lo = np.minimum(result_mesh.bounds()[0], self.cad_mesh.bounds()[0])
        hi = np.maximum(result_mesh.bounds()[1], self.cad_mesh.bounds()[1])
        origin, cell2, dims = lattice_for(lo, hi, cell, offset=0.5)
        field = columns_from_mesh(result_mesh, origin, cell2, dims, self.merge_m)
        target = columns_from_mesh(self.cad_mesh, origin, cell2, dims, self.merge_m, tree=self.cad_tree)
        return field, target


class SurfaceJson:
    """surface.json content: a dict, a callable returning it (read again at every call: wire
    ``lambda: surface_domain.document(ctx, state)`` so each operation end sees the current results), a path to the
    file or a run directory."""

    def __init__(self, source=None):
        self._doc = None
        self._path = None
        self._getter = None
        if isinstance(source, dict):
            self._doc = source
        elif callable(source):
            self._getter = source
        elif source is not None:
            path = Path(source)
            self._path = path / "surface.json" if path.is_dir() else path

    def _load(self) -> dict:
        if self._getter is not None:
            return self._getter() or {}
        if self._doc is None:
            if self._path is None or not self._path.is_file():
                return {}
            self._doc = json.loads(self._path.read_text(encoding="utf-8"))
        return self._doc

    def features(self) -> list[dict]:
        return [f for f in self._load().get("features") or [] if isinstance(f, dict)]

    def threads(self) -> list[dict]:
        return [t for t in self._load().get("threads") or [] if isinstance(t, dict)]


class NoEvents:
    """No surface-event data: every point's tool unknown, no deflection component."""

    def events(self, ids: np.ndarray) -> dict:
        n = len(np.asarray(ids))
        return {"tool_slot": np.full(n, -1, np.int64), "u_rel_m": np.zeros((n, 3))}


class TableEvents:
    """Surface events from rows of cutting's SURFACE_EVENT_COLUMNS keyed by event id (the event id of a row is its
    index plus ``first_id``)."""

    def __init__(self, rows: np.ndarray, first_id: int = 1):
        from ..core.bus import SURFACE_EVENT_COLUMNS

        self.rows = np.asarray(rows, dtype=np.float64).reshape(-1, len(SURFACE_EVENT_COLUMNS))
        self.first = int(first_id)
        self._col = {name: i for i, name in enumerate(SURFACE_EVENT_COLUMNS)}

    def events(self, ids: np.ndarray) -> dict:
        ids = np.asarray(ids, dtype=np.int64)
        k = ids - self.first
        ok = (k >= 0) & (k < len(self.rows))
        slot = np.full(len(ids), -1, np.int64)
        u = np.zeros((len(ids), 3))
        c = self._col
        slot[ok] = self.rows[k[ok], c["tool_slot"]].astype(np.int64)
        u[ok] = self.rows[k[ok]][:, [c["u_rel_x_m"], c["u_rel_y_m"], c["u_rel_z_m"]]]
        return {"tool_slot": slot, "u_rel_m": u}


class ColumnsStockBuilder:
    """Next-stock 'field' as z columns of the transformed mesh (tests, and runs without removal wired)."""

    def __init__(self, lattice_m: float | None = None):
        self.lattice_m = lattice_m

    def remap(self, workpiece: int, rotation: np.ndarray, kind: str):
        """None: this builder holds no field to remap, so the next stock is always rasterised from the mesh."""
        return None

    def from_mesh(self, workpiece: int, vertices_S_m, faces, lattice_m, box_lo_m, box_hi_m):
        from .columns import columns_from_mesh, lattice_for

        cell = float(self.lattice_m or lattice_m)
        mesh = Mesh(vertices_S_m, faces)
        origin, cell2, dims = lattice_for(box_lo_m, box_hi_m, cell, margin_cells=0, offset=0.5)
        return columns_from_mesh(mesh, origin, cell2, dims)


class CartesianTips:
    """Tool tips of a 3-axis Cartesian mill (cnctwin.sanity's convention): TIP-space commands carry the tip in W;
    machine-space ``x_W = X - offset_x``, ``y_W = Y - offset_y``, ``z_W = Z - offset_z - h`` with the work offset of
    the command's ``modal["work_offset"]`` and h the active tool length (``modal["length_comp"]`` and
    ``modal["length_offset"]``; ``G44`` negates it)."""

    def __init__(self, work_offsets_m: dict, length_offsets_m: dict | None = None,
                 axis_W=(0.0, 0.0, 1.0)):
        self.work_offsets = {str(k).upper().replace(" ", ""): np.asarray(v, dtype=np.float64)[:3]
                             for k, v in (work_offsets_m or {}).items()}
        self.lengths = {int(k): float(v) for k, v in (length_offsets_m or {}).items()}
        self.axis = np.asarray(axis_W, dtype=np.float64)

    def _offset(self, command: Command) -> tuple[np.ndarray, float] | None:
        modal = command.modal or {}
        code = str(modal.get("work_offset", "G54")).upper().replace(" ", "")
        offset = self.work_offsets.get(code)
        if offset is None:
            return None
        h = 0.0
        if modal.get("length_comp") and modal.get("length_offset"):
            h = self.lengths.get(int(modal["length_offset"]), math.nan)
            if modal.get("length_comp_code") == "G44":
                h = -h
        return offset, h

    def tip_W(self, command: Command, previous_W):
        if command.space == Space.TIP and command.tip_target_m is not None:
            return np.asarray(command.tip_target_m, dtype=np.float64)
        found = self._offset(command)
        if found is None:
            return None
        offset, h = found
        tip = np.full(3, np.nan) if previous_W is None else np.asarray(previous_W, dtype=np.float64).copy()
        for k, axis in enumerate("XYZ"):
            if axis in command.target:
                tip[k] = command.target[axis] - offset[k] - (h if axis == "Z" else 0.0)
        return tip

    def point_W(self, command: Command, point):
        p = np.asarray(point, dtype=np.float64)
        if command.space == Space.TIP:
            return p
        found = self._offset(command)
        if found is None:
            return np.full(3, np.nan)
        offset, h = found
        return p - offset - np.array([0.0, 0.0, h])

    def axis_W(self, command: Command):
        if command.space == Space.TIP and command.tip_axis is not None:
            return np.asarray(command.tip_axis, dtype=np.float64)
        return self.axis


class StaticCommands:
    """Command streams given up front: ``{op_index: [Command, ...]}``."""

    def __init__(self, streams: dict | None = None):
        self.streams = dict(streams or {})

    def commands(self, op_index: int) -> list:
        return list(self.streams.get(int(op_index), []))


__all__ = ["ResultSource", "FieldColumns", "SurfaceResults", "SurfaceEventLookup", "StockFieldBuilder",
           "TipResolver", "CommandSource", "StaticResults", "MeshColumns", "SurfaceJson", "NoEvents", "TableEvents",
           "ColumnsStockBuilder", "CartesianTips", "StaticCommands"]
