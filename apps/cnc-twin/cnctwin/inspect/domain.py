"""The inspection Domain object (core.domain.Domain; docs/models/inspect.md "Domain and integration").

Inspection is a sink at operation ends: it takes no part in the macro loop (``kernel_inputs`` and ``absorb`` return
None, ``macro_step`` does nothing: nothing it computes feeds back into the physics within an operation) and writes
no bus record. It reads, through the wired protocols (cnctwin.inspect.protocols):

* at ``init``: the project's ``cad`` (path, units, placement = cad_pose_S_C, datums, tolerances), the CAD model
  (STEP with PMI through cnctwin.geom.read_step in a child process; a plain mesh becomes freeform faces), the tool
  table (slot -> T number);
* at ``end_operation``: the operation's ResultPart per workpiece (ResultSource), the field / target columns
  (FieldColumns), the command stream and tip resolver (chordal deviation), and surface results (roughness,
  threads); it inspects, writes ``part_<op>.glb|stl|3mf`` and ``part_<op>_unload.stl``, raises ``gouge`` findings,
  and builds the next operation's stock (cnctwin.inspect.stock) for the integration to hand to removal;
* at ``finalize``: writes inspection.json and the per-brick ``deviation_m`` attributes, raises
  ``out_of_tolerance`` (error) for failed characteristics, ``out_of_tolerance`` (info) for characteristics that
  could not be evaluated, and ``remaining_material`` (warning) per region of the finished part.

Its state is :class:`cnctwin.inspect.state.InspectState` on the domain object (SimState has no inspect field); the
domain implements the checkpoint participant protocol (``checkpoint_cut``, ``checkpoint_state``, ``restore_state``,
``truncate_outputs``) so a resumed run continues with the records made before the cut.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.findings import Category, Severity
from . import schema as SC
from . import specs as SP
from .cadref import CadReference, cad_from_mesh
from .channels import channels as _channels
from .inspector import Inspector, document
from .result import INSPECTED
from .state import InspectState
from .stock import next_stock

DOMAIN = "inspect"


def _get(obj, key, default=None):
    if obj is None:
        return default
    if hasattr(obj, "get"):
        try:
            value = obj.get(key)
            return default if value is None else value
        except TypeError:
            pass
    value = getattr(obj, key, None)
    return default if value is None else value


class InspectDomain:
    name = DOMAIN

    def __init__(self, results=None, columns=None, surface=None, events=None, stock_builder=None, tips=None,
                 commands=None, cad=None, settings=None, mislocation=None, bricks=None):
        self.results = results
        self.columns = columns
        self.surface = surface
        self.events = events
        self.stock_builder = stock_builder
        self.tips = tips
        self.commands = commands
        self.cad_model = cad
        self.settings = settings
        self.mislocation = mislocation
        self.bricks = bricks
        self.state = InspectState()
        self.inspector: Inspector | None = None
        self.cad: CadReference | None = None
        self.stocks: dict = {}
        self._last_results: dict = {}
        self._inspectors: dict = {}
        self.cad_poses: dict = {}                  # workpiece -> S <- C after the transfers so far
        self.last_poses: dict = {}                 # workpiece -> S <- C of its last inspected operation
        self.model = None
        self.base_pose = np.eye(4)
        self.datums: list = []
        self.tolerances: list = []
        self.tool_numbers: dict = {}

    # ------------------------------------------------------------------ Domain interface
    def channels(self) -> list:
        return _channels()

    def init(self, ctx, state) -> None:
        project = ctx.project
        cad_section = _get(project, "cad")
        model = self.cad_model if self.cad_model is not None else self._load_cad(cad_section)
        self.model = model
        self.base_pose = np.asarray(_get(cad_section, "cad_pose_S_C", np.eye(4)), dtype=np.float64)
        self.datums, self.tolerances = SP.merged(cad_section, model.pmi)
        self.settings = self.settings or SC.settings(_get(_get(project, "physics"), "inspect"))
        self.tool_numbers = {int(s.slot): int(s.number)
                             for s in getattr(getattr(state, "tools", None), "slots", ()) or ()}
        self.inspector = self._inspector_for(self.base_pose)
        self.cad = self.inspector.cad
        ctx.trace.record(0.0, 0, DOMAIN, "inspect:init", {"faces": float(len(self.cad.faces)),
                                                          "features": float(len(self.inspector.features)),
                                                          "datums": float(len(self.datums)),
                                                          "tolerances": float(len(self.tolerances))}, [])

    def _inspector_for(self, pose: np.ndarray) -> Inspector:
        """The inspector of the CAD placed by ``pose`` (S <- C): the CAD follows the part through transfers."""
        key = np.round(np.asarray(pose, dtype=np.float64), 15).tobytes()
        found = self._inspectors.get(key)
        if found is None:
            found = Inspector(CadReference(self.model, pose), self.datums, self.tolerances, self.settings,
                              self.surface, self.events, None, self.tool_numbers)
            self._inspectors[key] = found
        return found

    @staticmethod
    def _load_cad(cad_section):
        path = _get(cad_section, "path")
        if not path:
            raise ValueError("inspection needs the project's cad.path (or a CadModel handed to InspectDomain)")
        path = Path(str(path))
        if path.suffix.lower() in (".step", ".stp"):
            from ..geom import read_step

            return read_step(path, isolated=True)
        from ..geom.io import read_mesh

        return cad_from_mesh(read_mesh(path, unit=str(_get(cad_section, "units", "mm"))), source=str(path))

    def kernel_inputs(self, ctx, state, bus):
        return None

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        return None

    def macro_step(self, ctx, state, prev, nxt) -> None:
        return None

    def _workpieces(self, op_index: int, op) -> list[int]:
        lister = getattr(self.results, "workpieces", None)
        if callable(lister):
            return list(lister(op_index))
        return [int(_get(_get(op, "setup"), "workpiece", 0))]

    def end_operation(self, ctx, state) -> None:
        if self.results is None or self.inspector is None:
            return
        operations = list(_get(ctx.project, "operations", []) or [])
        op_index = int(getattr(state.clock, "op_index", 0))
        op = operations[op_index] if op_index < len(operations) else None
        final = op_index >= len(operations) - 1
        tick = int(getattr(state.clock, "tick", 0))
        for w in self._workpieces(op_index, op):
            result = self.results.result_part(ctx, state, op_index, w)
            if result is None:
                continue
            commands = self.commands.commands(op_index) if self.commands is not None else None
            pose_S_W = np.linalg.inv(np.asarray(_get(op, "stock_pose_W_S", np.eye(4)), dtype=np.float64))
            pose_S_C = self.cad_poses.get(w, self.base_pose)
            inspector = self._inspector_for(pose_S_C)
            record = inspector.inspect(result, final, commands, self.tips, pose_S_W, self.columns)
            record["cad_pose_S_C"] = [[float(v) for v in row] for row in pose_S_C]
            self.state.records.append(record)
            self._last_results[w] = result
            self.last_poses[w] = pose_S_C
            if ctx.run_dir is not None:
                self.state.files.extend(result.write_files(ctx.run_dir))
            for k, g in enumerate((record.get("material") or {}).get("gouges", [])):
                self._finding(ctx, state, f"gouge:{result.op}:{w}:{k}", Category.GOUGE, Severity.ERROR,
                              f"{result.op}: CAD material missing (gouge) of {g['volume_m3'] * 1e9:.3f} mm^3, up to "
                              f"{g['max_thickness_m'] * 1e3:.4f} mm deep at S ({', '.join(f'{v * 1e3:.3f}' for v in g['max_point_S_m'])}) mm",
                              tick, {"volume_m3": g["volume_m3"], "max_thickness_m": g["max_thickness_m"],
                                     "x_S_m": g["max_point_S_m"][0], "y_S_m": g["max_point_S_m"][1],
                                     "z_S_m": g["max_point_S_m"][2]})
            if not final:
                nxt = operations[op_index + 1]
                M = self.mislocation(op_index + 1, w) if callable(self.mislocation) else \
                    (self.mislocation or {}).get((op_index + 1, w))
                stock = next_stock(result, _get(nxt, "from_previous"), str(_get(_get(ctx.project, "stock"), "kind",
                                                                                   "block")),
                                   float(_get(_get(ctx.project, "physics"), "lattice_m", 2.5e-4)), M,
                                   self.stock_builder)
                self.stocks[(op_index + 1, w)] = stock
                self.state.next_stocks.append(stock.to_json() | {"for_op": str(_get(nxt, "name", op_index + 1))})
                # the CAD follows the nominal transfer (a mislocation displaces the material, not the drawing)
                self.cad_poses[w] = stock.nominal_transform @ pose_S_C

    def next_stock(self, op_index: int, workpiece: int = 0):
        """The NextStock built for operation ``op_index`` (None if none was built)."""
        return self.stocks.get((int(op_index), int(workpiece)))

    def finalize(self, ctx, state) -> dict:
        settings = self.settings or SC.defaults()
        doc = document(self.state.records, settings, {"next_stock": list(self.state.next_stocks),
                                                      "files": list(self.state.files)})
        tick = int(getattr(state.clock, "tick", 0))
        for t in doc["tolerances"]:
            if t.get("status") == "evaluated" and t.get("pass") is False:
                limit = t.get("allowed_m", t.get("tolerance_m"))
                text = (f"{t['measured_m'] * 1e3:.4f} mm" if t.get("measured_m") is not None else "")
                self._finding(ctx, state, f"tol:{t['id']}:{t.get('workpiece', 0)}", Category.OUT_OF_TOLERANCE,
                              Severity.ERROR, f"{t['id']} ({t['kind']}) is out of tolerance: measured {text}" +
                              (f", allowed {limit * 1e3:.4f} mm" if limit is not None else
                               f", limits {t.get('lower_m', 0) * 1e3:.4f}-{t.get('upper_m', 0) * 1e3:.4f} mm"
                               if t.get("lower_m") is not None else ""), tick,
                              {k: t[k] for k in ("measured_m", "tolerance_m", "lower_m", "upper_m", "allowed_m")
                               if isinstance(t.get(k), (int, float))})
            elif t.get("status") == "not evaluated":
                self._finding(ctx, state, f"noeval:{t['id']}:{t.get('workpiece', 0)}", Category.OUT_OF_TOLERANCE,
                              Severity.INFO, f"{t['id']} ({t['kind']}) could not be evaluated: {t.get('reason')}", tick,
                              {})
        for k, r in enumerate(doc["remaining"]):
            self._finding(ctx, state, f"remaining:{k}", Category.REMAINING_MATERIAL, Severity.WARNING,
                          f"material outside the CAD remains on the finished part: {r['volume_m3'] * 1e9:.3f} mm^3, up "
                          f"to {r['max_thickness_m'] * 1e3:.4f} mm thick", tick,
                          {"volume_m3": r["volume_m3"], "max_thickness_m": r["max_thickness_m"],
                           "x_S_m": r["max_point_S_m"][0], "y_S_m": r["max_point_S_m"][1],
                           "z_S_m": r["max_point_S_m"][2]})
        if ctx.run_dir is not None:
            from ..core.rundir import write_json_atomic

            run_dir = Path(ctx.run_dir)
            doc["attributes"] = self._write_attributes(run_dir)
            write_json_atomic(run_dir / "inspection.json", doc)
        return doc

    # ------------------------------------------------------------------ helpers
    def _finding(self, ctx, state, key, category, severity, message, tick, quantities) -> None:
        if key in self.state.finding_keys:
            return
        self.state.finding_keys.append(key)
        ctx.finding(state, DOMAIN, category, severity, message, tick,
                    quantities={k: float(v) for k, v in quantities.items()})

    def _write_attributes(self, run_dir: Path) -> list:
        from .deviation import brick_attributes, write_brick_attributes

        source = self.bricks
        if self.model is None:
            return []
        if source is None and (run_dir / "keyframes" / "bricks.idx").is_file():
            from ..core.keyframes import KeyframeReader

            class _Reader:
                def final_bricks(self_inner):
                    return KeyframeReader(run_dir).scene_at()
            source = _Reader()
        if source is None:
            return []
        bricks = source.final_bricks()
        by_w: dict[int, dict] = {}
        for key, mesh in bricks.items():
            text = str(key)
            w = int(text[1:].split(".", 1)[0]) if text.startswith("w") and "." in text else 0
            by_w.setdefault(w, {})[key] = mesh
        attrs = {}
        for w, group in sorted(by_w.items()):
            result = self._last_results.get(w)
            displace = (lambda X, r=result: r.displacement(INSPECTED, X)) if result is not None else None
            cad = self._inspector_for(self.last_poses.get(w, self.base_pose)).cad
            attrs.update(brick_attributes(cad, group, displace))
        return write_brick_attributes(run_dir, attrs)

    # ------------------------------------------------------------------ checkpoint participant
    def checkpoint_cut(self) -> dict:
        return {"records": len(self.state.records), "next_stocks": len(self.state.next_stocks),
                "finding_keys": len(self.state.finding_keys), "files": len(self.state.files)}

    def checkpoint_state(self) -> dict:
        return self.state.to_arrays()[1]

    def restore_state(self, saved) -> None:
        self.state = InspectState.from_arrays({}, dict(saved))
        self._rebuild_poses()

    def truncate_outputs(self, cut) -> None:
        for name in ("records", "next_stocks", "finding_keys", "files"):
            keep = int(cut.get(name, len(getattr(self.state, name))))
            setattr(self.state, name, getattr(self.state, name)[:keep])
        self._rebuild_poses()

    def _rebuild_poses(self) -> None:
        """The CAD poses after the transfers recorded so far (each next stock's nominal transform, in order)."""
        self.cad_poses = {}
        for ns in self.state.next_stocks:
            w = int(ns.get("workpiece", 0))
            self.cad_poses[w] = np.asarray(ns["nominal_transform"], dtype=np.float64) @ \
                self.cad_poses.get(w, self.base_pose)


__all__ = ["InspectDomain", "DOMAIN"]
