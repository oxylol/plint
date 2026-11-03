"""One operation's inspection and the inspection.json document (docs/models/inspect.md "Results").

:class:`Inspector` holds what does not change between operations (the CAD in S, the recognised features, the
merged datum and tolerance specifications, the settings and the wired sources) and :meth:`Inspector.inspect`
turns one :class:`~cnctwin.inspect.result.ResultPart` into an operation record:

* ``configurations`` (what geometry was built from which grids), ``deviation`` (per-face statistics of the
  inspected geometry), ``features`` (measured associations), ``datums`` (the frames used),
* ``tolerances`` - every characteristic on the dense result surface, with its virtual-CMM evaluation (``cmm``), the
  probe points (``points``) and the error budget (``contributions_m``: the measured value minus the value with one
  physical cause removed from the points - the spring-back / distortion, the tool deflection at generation per
  tool),
* ``remaining`` / ``gouges`` (connected regions, field against target), ``chordal`` (toolpath chord errors),
* ``cmm_report`` (feature, characteristic, nominal, tolerance, actual, deviation, pass/fail, points).

:func:`document` assembles inspection.json (format ``cnctwin-inspection``) from the operation records: the top-level
``tolerances`` are the last operation's evaluations of each workpiece (the finished part), as the report reads them.
"""

from __future__ import annotations

import math

import numpy as np

from .. import FORMAT_VERSION
from ..core.errors import InputError
from ..units import REFERENCE_TEMP_K
from . import gdt as G
from .cadref import CadReference
from .chordal import chordal_deviation, segments_from_commands
from .cmm import VirtualCMM, plan
from .columns import compare
from .datums import DatumError
from .deviation import deviation_map
from .features import measure, recognise
from .fitting import FitError
from .protocols import MeshColumns, NoEvents
from .result import COMPOSITION, INSPECTED, ResultPart
from .sampling import PointSet, assign
from .schema import InspectSettings
from .specs import ToleranceSpec

FORMAT = "cnctwin-inspection"
EVAL_ERRORS = (G.EvalError, FitError, DatumError, InputError, ValueError, np.linalg.LinAlgError)


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class Inspector:
    def __init__(self, cad: CadReference, datums: list, tolerances: list[ToleranceSpec], settings: InspectSettings,
                 surface=None, events=None, columns=None, tool_numbers: dict | None = None):
        self.cad = cad
        self.settings = settings
        self.datum_specs = {d.label: d for d in datums}
        self.tolerances = list(tolerances)
        self.surface = surface
        self.events = events if events is not None else NoEvents()
        self.columns = columns if columns is not None else MeshColumns(cad.mesh, cad_tree=cad.tree)
        self.tool_numbers = dict(tool_numbers or {})
        self.features = recognise(cad, settings)
        self.sizes = {}
        for t in self.tolerances:
            if t.kind == "diameter" and t.size is not None and t.feature is not None and not t.size.thread_class:
                faces = cad.select(t.feature)
                if faces:
                    self.sizes[frozenset(cad.feature_faces(faces[0]))] = t.size

    # ------------------------------------------------------------------ points
    def _components(self, result: ResultPart, mesh) -> dict:
        X = result.material_points(mesh)
        comps = {}
        if result.grid(INSPECTED) is not None:
            comps["residual_distortion"] = result.component(INSPECTED, X)
        ev = mesh.vertex_attr.get("surface_event")
        if ev is not None:
            data = self.events.events(np.asarray(ev, dtype=np.int64).reshape(-1))
            slots = np.asarray(data["tool_slot"], dtype=np.int64)
            u = np.asarray(data["u_rel_m"], dtype=np.float64)
            for slot in sorted(set(slots[slots >= 0].tolist())):
                if not np.any(u[slots == slot]):
                    continue
                c = np.zeros_like(u)
                c[slots == slot] = u[slots == slot]
                comps[f"tool_deflection:{slot}"] = c
        return comps

    def _cmm_points(self, cmm: VirtualCMM, cache: dict):
        def points_of(faces):
            sets = []
            groups = sorted({tuple(self.cad.feature_faces(f)) for f in faces})
            for g in groups:
                if g not in cache:
                    cache[g] = cmm.measure(plan(self.cad, list(g), self.settings))
                res = cache[g]
                ps = res.points()
                keep = np.isin(ps.faces, np.asarray(sorted(faces)))
                sets.append(ps.subset(keep))
            return PointSet.concat(sets)
        return points_of

    # ------------------------------------------------------------------ one operation
    def inspect(self, result: ResultPart, final: bool = True, commands=None, tips=None, pose_S_W=None,
                field_columns=None) -> dict:
        s = self.settings
        mesh = result.configuration(INSPECTED)
        comps = self._components(result, mesh)
        a = assign(mesh, self.cad, s, result.material_points(mesh), comps)
        dense = G.EvalContext(self.cad, lambda faces: a.face_points(self.cad, faces, s.max_points_per_face), s,
                              self.datum_specs, self.features, self.surface, self.sizes, "surface")
        cmm = VirtualCMM(mesh, s)
        cache: dict = {}
        cmm_ctx = G.EvalContext(self.cad, self._cmm_points(cmm, cache), s, self.datum_specs, self.features,
                                self.surface, self.sizes, "cmm")
        record = {"op": result.op, "op_index": result.op_index, "workpiece": result.workpiece, "final": bool(final),
                  "inspection_temp_K": result.inspection_temp_K, "configuration": INSPECTED,
                  "configurations": {name: list(grids) for name, grids in COMPOSITION.items()},
                  "result": result.summary(), "assumptions": result.assumptions(INSPECTED),
                  "deviation": deviation_map(a, self.cad, s.remaining_tolerance_m),
                  "features": [f.to_json() | {"measured": measure(f, self.cad, dense.points_of, s)}
                               for f in self.features]}
        tolerances, rows, frames = [], [], {}
        for spec in self.tolerances:
            entry = self._evaluate(spec, dense, cmm_ctx, a, result, mesh)
            tolerances.append(entry)
            if entry.get("cmm"):
                rows.append(entry["cmm"]["row"])
        record["tolerances"] = tolerances
        record["cmm_report"] = rows
        for key, (nominal, measured, T) in sorted(dense.frames.items()):
            frames["|".join(key)] = {"nominal": nominal.to_json(), "measured": measured.to_json(),
                                     "registration": [[float(v) for v in r] for r in T]}
        record["datums"] = frames
        record["material"] = self._material(result, field_columns)
        if commands is not None and tips is not None:
            segments = segments_from_commands(commands, tips, pose_S_W)
            record["chordal"] = chordal_deviation(segments, self.cad, s)
        unload = result.unload_record()
        if unload is not None:
            record["unload"] = unload
        record["probes"] = {"|".join(str(f) for f in k): v.to_json() for k, v in sorted(cache.items())}
        return json_safe(record)

    def _material(self, result: ResultPart, field_columns) -> dict:
        source = field_columns if field_columns is not None else self.columns
        try:
            field, target = source.columns(result.workpiece, result.material)
        except (ValueError, TypeError) as error:
            return {"status": f"not evaluated: {error}"}
        if target is None:
            return {"status": "not evaluated: no CAD target on the field's rays"}
        out = compare(field, target, self.settings, self.cad)
        out["status"] = "evaluated"
        out["configuration"] = "material (X: what the tools left; spring-back is in the deviation map)"
        return out

    def _tools(self, a, faces) -> list:
        return sorted({self.tool_numbers.get(int(k.split(":")[1]), int(k.split(":")[1]))
                       for k, v in a.components.items() if k.startswith("tool_deflection:")
                       and np.any(v[np.isin(a.face, np.asarray(faces))])})

    def _evaluate(self, spec: ToleranceSpec, dense, cmm_ctx, a, result: ResultPart, mesh) -> dict:
        entry = {"id": spec.id, "kind": spec.kind, "feature": spec.describe_feature(), "datums": list(spec.datums),
                 "source": spec.source, "op": result.op}
        if spec.size is not None and spec.size.fit:
            entry["fit"] = spec.size.fit
        if spec.modifier:
            entry["modifier"] = spec.modifier
        faces = []
        try:
            faces = sorted(set(f for sel in ([spec.feature] if spec.feature is not None else spec.between)
                               for f in self.cad.select(sel)))
            entry["faces"] = faces
            ev = G.evaluate(spec, dense)
        except EVAL_ERRORS as error:
            entry.update({"status": "not evaluated", "reason": str(error), "pass": None})
            if spec.tolerance_m is not None:
                entry["tolerance_m"] = spec.tolerance_m
            return entry
        entry.update(ev.record)
        entry.update({"status": "evaluated", "pass": ev.passed})
        if ev.record.get("kind") == "thread":
            entry["kind"] = "thread"
        fraction = a.machined_fraction(faces) if faces else None
        if fraction is not None:
            entry["machined_fraction"] = fraction
        tools = self._tools(a, faces)
        if tools:
            entry["tools"] = tools
        if ev.points is not None and a.components and ev.measured is not None:
            entry["contributions_m"] = self._budget(spec, dense, ev, sorted(a.components))
        try:
            evc = G.evaluate(spec, cmm_ctx)
            pts = evc.points
            cmm_points = []
            if pts is not None and len(pts) and evc.deviations is not None:
                cmm_points = [{"x_S_m": float(p[0]), "y_S_m": float(p[1]), "z_S_m": float(p[2]),
                               "deviation_m": float(d)} for p, d in zip(pts.points, evc.deviations)]
            elif pts is not None and len(pts):
                d, _ = G._nominal(cmm_ctx, pts.points, pts.faces)
                cmm_points = [{"x_S_m": float(p[0]), "y_S_m": float(p[1]), "z_S_m": float(p[2]),
                               "deviation_m": float(v)} for p, v in zip(pts.points, d)]
            entry["points"] = cmm_points
            measured = evc.record.get("measured_m", evc.record.get("measured_rad"))
            entry["cmm"] = {"measured": measured, "pass": evc.passed, "method": evc.record.get("method"),
                            "points": len(cmm_points), "agrees": evc.passed == ev.passed,
                            "row": {"characteristic": spec.id, "kind": entry["kind"], "feature": entry["feature"],
                                    "nominal_m": evc.record.get("nominal_m"), "lower_m": evc.record.get("lower_m"),
                                    "upper_m": evc.record.get("upper_m"), "tolerance_m": evc.record.get("tolerance_m",
                                                                                                         spec.tolerance_m),
                                    "actual": measured, "deviation": evc.record.get("deviation_m",
                                                                                    evc.record.get("deviation_rad")),
                                    "pass": evc.passed, "points": len(cmm_points)}}
        except EVAL_ERRORS as error:
            entry["cmm"] = {"status": "not evaluated", "reason": str(error)}
        return entry

    def _budget(self, spec, dense, ev, names) -> list[dict]:
        """Contribution of each physical cause: the measured value minus the value with that displacement removed
        from every point (datum features included)."""
        out = []
        base = ev.measured
        for name in names:
            def shifted(faces, name=name):
                ps = dense.points_of(faces)
                return ps.shifted(ps.components[name]) if name in ps.components else ps
            try:
                other = G.evaluate(spec, dense.with_points(shifted)).measured
            except EVAL_ERRORS:
                continue
            if other is None:
                continue
            source, _, slot = name.partition(":")
            row = {"source": source, "value_m": float(base - other)}
            if slot:
                row["tool"] = self.tool_numbers.get(int(slot), int(slot))
            out.append(row)
        return out


def document(records: list[dict], settings: InspectSettings, extra: dict | None = None) -> dict:
    """inspection.json from operation records (module documentation)."""
    last: dict[int, dict] = {}
    for r in records:
        last[int(r["workpiece"])] = r
    final = [last[w] for w in sorted(last)]
    tolerances = []
    for r in final:
        for t in r.get("tolerances", []):
            entry = dict(t)
            if len(final) > 1:
                entry["workpiece"] = r["workpiece"]
            tolerances.append(entry)
    checked = [t for t in tolerances if t.get("status") == "evaluated"]
    failed = [t for t in checked if t.get("pass") is False]
    doc = {"format": FORMAT, "version": FORMAT_VERSION,
           "temperature_K": final[0]["inspection_temp_K"] if final else REFERENCE_TEMP_K,
           "configuration": "final: X + d_residual (unclamped, spring-back and distortion) at 20 degC",
           "tolerances": tolerances,
           "cmm_report": [row for r in final for row in r.get("cmm_report", [])],
           "features": [f for r in final for f in r.get("features", [])],
           "datums": {f"w{r['workpiece']}:{k}" if len(final) > 1 else k: v for r in final
                      for k, v in r.get("datums", {}).items()},
           "deviation": final[0]["deviation"] if len(final) == 1 else {f"w{r['workpiece']}": r["deviation"]
                                                                        for r in final},
           "remaining": [x for r in final for x in (r.get("material") or {}).get("remaining", [])],
           "gouges": [x for r in records for x in (r.get("material") or {}).get("gouges", [])],
           "chordal": [dict(r["chordal"], op=r["op"]) for r in records if "chordal" in r],
           "unload": [r["unload"] for r in records if "unload" in r],
           "operations": records,
           "summary": {"checked": len(checked), "failed": len(failed),
                       "not_evaluated": len(tolerances) - len(checked),
                       "cmm_disagrees": sum(1 for t in checked if (t.get("cmm") or {}).get("agrees") is False)},
           "settings": settings.to_dict(),
           "models": {"association": "least squares (ISO 10360-6) and minimum zone / envelopes by sequential LP",
                      "datums": "ASME Y14.5-2018 / ISO 5459 datum feature simulators, constrained fits",
                      "cmm": "touch probing by sphere tracing, vector-mode radius compensation",
                      "threads": "ISO 965-1, ASME B1.1, gauging per ISO 1502", "fits": "ISO 286-1/-2"}}
    if extra:
        doc.update(extra)
    return json_safe(doc)


__all__ = ["Inspector", "document", "json_safe", "FORMAT", "EVAL_ERRORS"]
