"""Declared schema of the project file (db/_templates/project.yaml; docs/formats.md "Project file").

Every key of the template is declared with its unit (the same declared-unit language as the
databases, :mod:`cnctwin.db.schema`). Database references (:class:`DbRef`) are an id, a path to a
database file (relative to the project), or an inline object in place of the id (checked with
the inline schema of its kind; ``extends: <id>`` starts from a database object).

``physics.<domain>`` sections: every domain of :data:`cnctwin.core.state.SimState.DOMAIN_FIELDS`
may have one. ``every_n_macro`` (architecture 5.1) is declared for all of them; a domain declares
its other settings in ``cnctwin/<domain>/schema.py`` as ``PHYSICS_SCHEMA`` (a
:class:`cnctwin.db.schema.Struct`; also found as ``PHYSICS_<DOMAIN>``, ``SCHEMA``, or returned by
``schema_node()`` / ``physics_schema()``) and is then checked strictly; without one the section is
checked generically (identifier keys, numbers/text/booleans/lists, units by key suffix).
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping

from ..db import schemas as DS
from ..db.schema import (Any_, ByType, Count, FilePath, Flag, MapOf, Node, Pose, Qty, Seq, Struct, Text, get, has,
                         is_number, unwrap)

DEG = math.pi / 180.0
FORMAT = "cnctwin-project"
CURRENT_VERSION = 1
DOMAINS = ("controller", "motion", "tooling", "removal", "cutting", "chips", "thermal", "coolant", "wear",
           "holding", "structure", "dynamics", "surface")
TICK_S = 1e-6                       # architecture 5.1: the base clock is fixed
ID_OR_PATH = r"[A-Za-z0-9_./+-][A-Za-z0-9_./ +-]*"


def is_path_ref(text: str) -> bool:
    """A database reference that names a file (contains a / or ends with .yaml/.yml)."""
    return "/" in text or text.endswith((".yaml", ".yml"))


class DbRef(Node):
    """A database object: an id, a path to a database file, or an inline object in place of the id."""

    def __init__(self, kind: str, *, nullable=False, doc="", special=()):
        self.kind = kind
        self.nullable = nullable
        self.doc = doc
        self.special = tuple(special)

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if isinstance(value, str):
            if value in self.special:
                return
            if not value or not re.fullmatch(ID_OR_PATH, value):
                ctx.add(path, f"{value!r} is not a {self.kind} id or a path", "type")
                return
            ctx.refs.append((self.kind, value, path))
            return
        if isinstance(value, Mapping):
            ctx.refs.append((self.kind, value, path))
            return
        ctx.add(path, f"expected a {self.kind} id, a path to a {self.kind} file or an inline {self.kind}", "type")

    def build(self, value, unit, src, si_key):
        return value


POSE = Pose()
FEATURE = Struct({"face": Text(("plane", "cylinder", "cone", "sphere", "torus", "freeform", "any")),
                  "normal": DS.vec3(-1.0, 1.0, unit_vector=True), "axis": DS.vec3(-1.0, 1.0, unit_vector=True),
                  "near_mm": DS.vec3(-100.0, 100.0), "radius_mm": DS.length(50.0), "id": Text()},
                 required=("face",), meta=False)
TOLERANCE_KINDS = ("distance", "flatness", "position", "roughness", "diameter", "radius", "perpendicularity",
                   "parallelism", "angularity", "cylindricity", "circularity", "straightness", "profile_surface",
                   "profile_line", "concentricity", "symmetry", "runout", "total_runout", "angle")


def _tolerance_rules(m, p, u, ctx):
    kind = get(m, "kind")
    if kind == "distance":
        if not isinstance(get(m, "between"), list):
            ctx.add(p, "a distance tolerance needs between: [feature, feature]", "required")
    elif not has(m, "feature"):
        ctx.add(p, f"a {kind} tolerance needs feature", "required")
    if kind == "roughness" and not any(has(m, k) for k in ("Ra_um", "Rz_um", "Rt_um")):
        ctx.add(p, "a roughness tolerance needs Ra_um, Rz_um or Rt_um", "required")
    if kind in ("distance", "diameter", "radius", "angle") and not (has(m, "nominal_mm") or has(m, "nominal_deg")
                                                                     or has(m, "diameter_mm")):
        ctx.add(p, f"a {kind} tolerance needs its nominal (nominal_mm / nominal_deg / diameter_mm)", "required")
    if kind not in ("distance", "diameter", "radius", "angle", "roughness") and not has(m, "tolerance_mm"):
        ctx.add(p, f"a {kind} tolerance needs tolerance_mm", "required")


TOLERANCE = Struct({
    "id": Text(pattern=r"[A-Za-z0-9][A-Za-z0-9_.-]*"), "kind": Text(TOLERANCE_KINDS), "feature": FEATURE,
    "between": Seq(FEATURE, min_len=2, max_len=2), "nominal_mm": Qty(lo=-100.0, hi=100.0),
    "nominal_deg": DS.angle(-360, 360), "plus_mm": Qty(lo=0, hi=1.0), "minus_mm": Qty(lo=0, hi=1.0),
    "plus_deg": DS.angle(0, 90), "minus_deg": DS.angle(0, 90), "tolerance_mm": Qty(lo=0, hi=1.0),
    "datums": Seq(Text(pattern=r"[A-Z][A-Z0-9]*")), "basic_mm": Qty(shape=(2, 3), lo=-100.0, hi=100.0),
    "diameter_mm": ByType(number=DS.length(10.0), mapping=Struct({"nominal": DS.length(10.0), "fit": Text(),
                                                                  "plus": Qty(lo=0, hi=1.0), "minus": Qty(lo=0, hi=1.0)},
                                                                 required=("nominal",), meta=False),
                          expected="a diameter or {nominal, fit}"),
    "Ra_um": DS.length(1e-3), "Rz_um": DS.length(1e-2), "Rt_um": DS.length(1e-2), "modifier": Text(),
}, required=("id", "kind"), checks=(_tolerance_rules,), meta=False)


def _stock_rules(m, p, u, ctx):
    kind = get(m, "kind")
    size = get(m, "size_mm")
    expected = {"block": 3, "cylinder": 2, "tube": 3}.get(kind)
    if expected is not None:
        if not isinstance(size, list):
            ctx.add(p, f"a {kind} stock needs size_mm ({'[x, y, z]' if kind == 'block' else '[diameter, length]' if kind == 'cylinder' else '[od, id, length]'})",
                    "required")
        elif len(size) != expected:
            ctx.add(p + ("size_mm",), f"a {kind} stock has {expected} sizes, got {len(size)}", "rule")
        elif kind == "tube" and all(is_number(v) for v in size) and size[1] >= size[0]:
            ctx.add(p + ("size_mm",), "a tube's inner diameter must be smaller than its outer diameter", "rule")
    if kind == "mesh" and not get(m, "mesh"):
        ctx.add(p, "a mesh stock needs mesh: <file>", "required")


STOCK = Struct({
    "kind": Text(("block", "cylinder", "tube", "mesh")),
    "size_mm": Qty(shape=(2, 3), gt=0, hi=100.0, nullable=True),
    "tolerance_mm": Qty(shape=(2, 3), lo=0, hi=1.0, nullable=True),
    "non_squareness_deg": DS.angle(0, 10, nullable=True),
    "form": Text(DS.STOCK_FORMS, nullable=True),
    "mesh": FilePath(exts=(".stl", ".3mf", ".obj", ".step", ".stp"), nullable=True),
}, required=("kind",), checks=(_stock_rules,), meta=False)

WEAR_OFFSETS = Struct({"length": Qty(lo=-0.1, hi=0.1), "radius": Qty(lo=-0.1, hi=0.1),
                       "x": Qty(lo=-0.1, hi=0.1), "z": Qty(lo=-0.1, hi=0.1)}, meta=False,
                      doc="wear offsets as set on the control: length/radius (mills), x (diameter on a diameter-"
                          "programmed lathe) and z (lathes)")

TOOL_ENTRY = Struct({
    "number": Count(lo=0, hi=100000), "kind": Text(("tool",)), "tool": DbRef("tool"),
    "holder": DbRef("holder", nullable=True),
    "path": Count(lo=1, hi=16), "spindle": Text(pattern=r"[a-z][a-z0-9_-]*"), "turret": Text(),
    "station": Count(lo=1, hi=200),
    "stickout_mm": DS.length(2.0, nullable=True), "length_offset_mm": Qty(lo=-10.0, hi=10.0, nullable=True),
    "radius_offset_mm": Qty(lo=-1.0, hi=1.0, nullable=True),
    # lathe geometry offsets as set on the control's offset page (X in diameter on a diameter-programmed lathe)
    "x_offset_mm": Qty(lo=-10.0, hi=10.0, nullable=True), "z_offset_mm": Qty(lo=-10.0, hi=10.0, nullable=True),
    "nose_radius_mm": DS.length(0.1, positive=False, nullable=True), "tip": Count(lo=0, hi=9),
    "wear_offsets_mm": WEAR_OFFSETS,
    "runout_um": DS.length(1e-3, positive=False, nullable=True), "runout_phase_deg": DS.angle(-360, 360),
    "tilt_urad": Qty(lo=0, hi=0.1),
    "life": Struct({"group": Count(lo=0), "limit_min": Qty(gt=0, hi=1e8, nullable=True),
                    "used_min": Qty(lo=0, hi=1e8), "limit_parts": Count(lo=0, nullable=True),
                    "sister_of": Count(lo=0, nullable=True)}, meta=False, nullable=True),
    "initial_wear": Struct({"VB_mm": DS.length(0.01, positive=False), "KT_um": DS.length(1e-3, positive=False),
                            "edge_radius_um": DS.length(1e-3, positive=False)}, meta=False, nullable=True),
    "comment": Text(allow_empty=True, nullable=True),
}, required=("number", "tool"), meta=False)

PROBE_NAME = r"[a-z][a-z0-9_]*"
# The probe body (shank to stylus): a cylinder below the gauge line; the stylus: a stem with its ball.
PROBE_BODY = Struct({"diameter_mm": DS.length(0.5), "length_mm": DS.length(1.0)},
                    required=("diameter_mm", "length_mm"), meta=False)


def _inline_probe_rules(m, p, u, ctx):
    if not has(m, "extends") and not has(m, "stylus_ball_mm"):
        ctx.add(p, "an inline probe needs stylus_ball_mm (or extends: <machine probe> giving it)", "required")


INLINE_PROBE = Struct({
    "extends": Text(pattern=PROBE_NAME, doc="a probe of the machine's probes section (spindle_probe)"),
    "model": Text(), "stylus_ball_mm": DS.length(0.05), "stylus_length_mm": DS.length(0.5),
    "stylus_diameter_mm": DS.length(0.05), "body": PROBE_BODY,
    "pretravel_um": DS.length(1e-3, positive=False), "repeatability_um": DS.length(1e-3, positive=False),
    "trigger_force_N": Qty(gt=0, hi=1e3), "overtravel_z_mm": DS.length(0.1, positive=False),
}, checks=(_inline_probe_rules,), meta=False)

# A probe entry's probe once resolved: the machine's probe (db PROBE fields, with their sources) merged with
# the inline keys of the entry (stylus, body); the SI view is ``project.tools.probes[i].probe``.
RESOLVED_PROBE = Struct({**{k: v for k, v in INLINE_PROBE.fields.items() if k != "extends"}, **DS.PROBE.fields},
                        meta=True)

PROBE_ENTRY = Struct({
    "number": Count(lo=0, hi=100000), "kind": Text(("probe",)),
    "probe": ByType(text=Text(pattern=PROBE_NAME), mapping=INLINE_PROBE,
                    expected="a probe of the machine (spindle_probe) or an inline probe {stylus_ball_mm, ...}"),
    "path": Count(lo=1, hi=16), "spindle": Text(pattern=r"[a-z][a-z0-9_-]*"),
    "length_offset_mm": Qty(lo=-10.0, hi=10.0, nullable=True),
    "radius_offset_mm": DS.length(0.05, positive=False, nullable=True),
    "wear_offsets_mm": WEAR_OFFSETS, "comment": Text(allow_empty=True, nullable=True),
}, required=("number", "kind", "probe"), meta=False)


class ToolListEntry(Node):
    """A ``tools.list`` entry: a cutting tool (TOOL_ENTRY), or with ``kind: probe`` a spindle probe the
    program calls by its T number (PROBE_ENTRY)."""

    takes_unit = False

    def __init__(self):
        self.nullable = False

    @staticmethod
    def select(value):
        kind = unwrap(value.get("kind")) if isinstance(value, Mapping) else None
        return PROBE_ENTRY if kind == "probe" else TOOL_ENTRY

    def check(self, value, path, unit, ctx):
        if not isinstance(value, Mapping):
            ctx.add(path, "expected a tool entry {number, tool, ...} or a probe {number, kind: probe, probe}", "type")
            return
        self.select(value).check(value, path, None, ctx)

    def build(self, value, unit, src, si_key):
        return self.select(value).build(value, None, src, si_key) if isinstance(value, Mapping) else value


TOOL_LIST_ENTRY = ToolListEntry()


def _workholding_rules(m, p, u, ctx):
    outline = unwrap(m.get("outline_mm")) if isinstance(m, Mapping) else None
    if isinstance(outline, list) and len(outline) != 2:
        ctx.add(p + ("outline_mm",), "outline_mm is a box: [[x0, y0, z0], [x1, y1, z1]]", "rule")


WORKHOLDING_ENTRY = Struct({
    "device": DbRef("workholding"), "pose": POSE, "jaw_opening_mm": DS.length(5.0, nullable=True),
    "input_torque_Nm": Qty(gt=0, hi=1e4, nullable=True), "grip_depth_mm": DS.length(1.0, nullable=True),
    "parallels_mm": DS.length(1.0, nullable=True), "pressure_bar": Qty(gt=0, hi=1e8, nullable=True),
    # additive: the device's outline (a box in its own frame, placed by pose) and the height of a vise's seat
    # above its mounting face, for devices whose database item gives no body
    "outline_mm": Qty(shape=("rows", 3), lo=-100.0, hi=100.0, nullable=True),
    "seat_height_mm": DS.length(2.0, positive=False, nullable=True),
    "jaws": Text(), "label": Text(),
}, required=("device",), checks=(_workholding_rules,), meta=False)

OPERATION = Struct({
    "name": Text(pattern=r"[A-Za-z0-9][A-Za-z0-9_.-]*"), "description": Text(),
    "program": FilePath(exts=None),
    "programs": MapOf(FilePath(), int_keys=True, meta=False, doc="multi-path controls: program per path number"),
    "setup": Struct({
        "workholding": Seq(WORKHOLDING_ENTRY), "stock_pose": POSE,
        "work_offsets_mm": MapOf(Qty(shape=(3, 4, 5, 6), lo=-100.0, hi=100.0),
                                 key=Text(pattern=r"[A-Za-z0-9.#_ ]+"), meta=False),
        "fixture_pose": POSE, "spindle": Text(), "workpiece": Count(lo=0, hi=8),
    }, meta=False),
    "from_previous": Struct({"op": Text(), "flip": Struct({"axis": Text(("X", "Y", "Z")), "angle_deg": DS.angle(-360, 360)},
                                                          required=("axis", "angle_deg"), meta=False),
                             "transfer": Text(("sub_spindle", "main_spindle", "pallet", "manual"))},
                            required=("op",), meta=False, nullable=True),
    "nozzle_positions": MapOf(Count(lo=0, hi=10000), int_keys=True, meta=False),
}, required=("name",), checks=(lambda m, p, u, c: None if has(m, "program") or has(m, "programs")
                               else c.add(p, "an operation needs program (or programs per path)", "required"),),
    meta=False)

RESOLVED_SEGMENT = Struct({
    "tool": Count(lo=0, hi=100000), "gear": Count(lo=0, hi=16), "path": Count(lo=1, hi=16),
    "l0_ticks": Count(lo=1, hi=10 ** 9), "max_rpm": Qty(gt=0, hi=DS.RAD_S_MAX),
    "f_max_Hz": Qty(gt=0, hi=1e7), "teeth": Count(lo=1, hi=1000), "rule": Text(("angle", "mode", "tooth", "ipo")),
}, required=("tool", "l0_ticks"), meta=False)
RESOLVED = Struct({
    "ipo_ticks": Count(lo=1, hi=10 ** 9), "macro_ipo": Count(lo=1, hi=10 ** 6),
    "operations": MapOf(Seq(RESOLVED_SEGMENT, min_len=1), meta=False),
    "build": Text(doc="the build (version) that resolved the strides"),
}, meta=False, nullable=True, doc="written by 'twin check --write': the recorded L0 strides per operation")


def _resolved_rules(m, p, u, ctx):
    resolved = get(m, "resolved")
    if not isinstance(resolved, Mapping) or not resolved:
        return
    ipo = get(resolved, "ipo_ticks")
    ops = get(resolved, "operations")
    if ops and not isinstance(ipo, int):
        ctx.add(p + ("resolved",), "physics.resolved needs ipo_ticks with its operations", "required")
        return
    for name, segments in (ops or {}).items():
        for i, segment in enumerate(segments or []):
            l0 = get(segment, "l0_ticks") if isinstance(segment, Mapping) else None
            if isinstance(l0, int) and isinstance(ipo, int) and ipo % l0:
                ctx.add(p + ("resolved", "operations", name, i, "l0_ticks"),
                        f"l0_ticks {l0} is not a divisor of ipo_ticks {ipo}", "rule")


def _physics_rules(m, p, u, ctx):
    tick = unwrap(m.get("tick_s")) if isinstance(m, Mapping) else None
    if is_number(tick) and abs(tick - TICK_S) > 1e-15:
        ctx.add(p + ("tick_s",), f"tick_s is fixed at {TICK_S:g} s (architecture 5.1)", "rule")
    _resolved_rules(m, p, u, ctx)


REFINE = Struct({"box": Struct({"min_mm": DS.vec3(-100.0, 100.0), "max_mm": DS.vec3(-100.0, 100.0)},
                               required=("min_mm", "max_mm"), meta=False),
                 "factor": Count(lo=2, hi=64)}, required=("box", "factor"), meta=False)


class DomainPhysics(Node):
    """``physics.<domain>``: the domain's PHYSICS_SCHEMA if it declares one, else a generic check."""

    takes_unit = False

    def __init__(self, domain: str):
        self.domain = domain
        self.nullable = True

    # names a domain's schema module may declare its physics node under (first found wins)
    NAMES = ("PHYSICS_SCHEMA", "PHYSICS_{DOMAIN}", "SCHEMA", "schema_node", "physics_schema")

    def declared(self) -> Node | None:
        try:
            module = importlib.import_module(f"cnctwin.{self.domain}.schema")
        except ModuleNotFoundError as error:
            if error.name in (f"cnctwin.{self.domain}", f"cnctwin.{self.domain}.schema"):
                return None
            raise
        for name in self.NAMES:
            found = getattr(module, name.format(DOMAIN=self.domain.upper()), None)
            if callable(found) and not isinstance(found, Node):
                found = found()
            if isinstance(found, Node):
                return found
        return None

    def check(self, value, path, unit, ctx):
        if value is None:
            return
        node = self.declared()
        if node is not None:
            if isinstance(node, Struct) and "every_n_macro" not in node.fields:
                node = node.extend({"every_n_macro": Count(lo=1, hi=10 ** 6)})
            node.check(value, path, None, ctx)
            return
        if not isinstance(value, Mapping):
            ctx.add(path, "expected a mapping of settings", "type")
            return
        for key, item in value.items():
            where = path + (key,)
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                ctx.add(where, f"setting name {key!r} must be an identifier", "type")
                continue
            if key in ("note", "notes"):
                continue
            if key == "every_n_macro":
                Count(lo=1, hi=10 ** 6).check(item, where, None, ctx)
            elif isinstance(item, (Mapping, list, str, bool, int, float)) or item is None:
                continue
            else:
                ctx.add(where, f"setting {key} must be a number, text, true/false, a list or a mapping", "type")

    def build(self, value, unit, src, si_key):
        node = self.declared()
        if node is not None and isinstance(value, Mapping):
            return node.build(value, None, src, si_key)
        return Any_().build(value, None, src, si_key)


PHYSICS = Struct({
    # template key with the unit inside (deg per step): declared explicitly, SI key max_rad_per_step
    "tick_s": Qty(gt=0, hi=1e-3), "max_deg_per_step": DS.angle(1e-3, 45, unit="deg", si_key="max_rad_per_step"),
    "steps_per_tooth": Count(lo=4, hi=10 ** 5),
    "macro_ipo": Count(lo=1, hi=10 ** 5), "lag_macro": Count(lo=1, hi=2), "lattice_mm": DS.length(0.01),
    "refine": Seq(REFINE), "edge_element_mm": DS.length(0.01), "memory_bins": Count(lo=8, hi=10 ** 6),
    "polar_theta_bins": Count(lo=8, hi=10 ** 6), "particles_per_tooth": Qty(lo=0, hi=1e4),
    "tol_mass": Qty(gt=0, hi=0.1), "tol_energy": Qty(gt=0, hi=0.1),
    **{domain: DomainPhysics(domain) for domain in DOMAINS},
    "resolved": RESOLVED,
}, checks=(_physics_rules,), meta=False)

PROJECT = Struct({
    "format": Text((FORMAT,), sourced=False), "version": Count(lo=0, hi=CURRENT_VERSION),
    "name": Text(), "description": Text(allow_empty=True), "seed": Count(lo=0, hi=2 ** 63 - 1),
    "machine": DbRef("machine"),
    "controller_overrides": Struct(DS.MACHINE_CONTROLLER.fields, meta=False),
    "material": Struct({"id": DbRef("material"),
                        "batch": Struct({"kc_scale_sigma": Qty(lo=0, hi=1), "hardness_sigma_pct": Qty(lo=0, hi=1),
                                         "flow_stress_sigma_pct": Qty(lo=0, hi=1),
                                         "conductivity_sigma_pct": Qty(lo=0, hi=1)}, meta=False, nullable=True)},
                       required=("id",), meta=False),
    "environment": Struct({
        "ambient_degC": DS.temp(), "coolant_degC": DS.temp(),
        "machine_state": ByType(text=Text(("cold", "warm")),
                                mapping=Struct({"spindle_degC": DS.temp(), "axes_degC": DS.temp(),
                                                "structure_degC": DS.temp()}, meta=False),
                                expected="cold, warm or {spindle_degC, axes_degC, structure_degC}"),
        "time_between_ops_min": Qty(lo=0, hi=1e7)}, meta=False),
    "stock": STOCK,
    "cad": Struct({"path": FilePath(exts=(".step", ".stp", ".stl", ".3mf", ".obj")),
                   "units": Text(("mm", "inch")), "placement": POSE, "datums": Seq(Struct({"label": Text(
                       pattern=r"[A-Z][A-Z0-9]*"), "feature": FEATURE}, required=("label", "feature"), meta=False)),
                   "tolerances": Seq(TOLERANCE, unique="id")}, required=("path",), meta=False, nullable=True),
    "tools": Struct({"library": FilePath(kind="any", nullable=True), "list": Seq(TOOL_LIST_ENTRY)}, meta=False),
    "coolant": Struct({"fluid": DbRef("coolant"), "concentration_pct": Qty(lo=0, hi=1, nullable=True),
                       "tank_fill_fraction": DS.fraction()}, required=("fluid",), meta=False),
    "operations": Seq(OPERATION, min_len=1, unique="name"),
    "production": Struct({"parts": Count(lo=1, hi=10 ** 7),
                          "operator": Struct({"load_s": Qty(lo=0, hi=1e5), "unload_s": Qty(lo=0, hi=1e5),
                                              "door_s": Qty(lo=0, hi=1e4), "clean_s": Qty(lo=0, hi=1e5),
                                              "inspect_every": Count(lo=0), "inspect_s": Qty(lo=0, hi=1e5)},
                                             meta=False)}, meta=False),
    "physics": PHYSICS,
    "recording": Struct({"fast": Text(("windows", "all", "none")), "fast_window_revs": Count(lo=1, hi=10 ** 5),
                         "fast_period_s": Qty(gt=0, hi=1e5), "keyframe_period_s": Qty(gt=0, hi=1e5)}, meta=False),
    "outputs": Struct({"report": Seq(Text(("html", "pdf"))), "images": Flag(),
                       "video": ByType(mapping=Struct({"fps": Count(lo=1, hi=240), "speed": Qty(gt=0, hi=1e4)},
                                                      meta=False), boolean=Flag(), nullable=True,
                                       expected="{fps, speed} or false"),
                       "audio": Flag()}, meta=False),
    "uncertainty": Struct({"runs": Count(lo=1, hi=10 ** 5), "method": Text(("lhs", "monte_carlo")),
                           "vary": Struct({"stock_size": Flag(), "tool_offsets_um": Qty(lo=0, hi=1e-3),
                                           "runout_um": Qty(lo=0, hi=1e-3), "kc_scale_sigma": Qty(lo=0, hi=1),
                                           "clamping_force_rel": Qty(lo=0, hi=1), "hardness_sigma_pct": Qty(lo=0, hi=1),
                                           "coolant_reach_rel": Qty(lo=0, hi=1)}, meta=False)},
                          meta=False, nullable=True),
    "machine_data": Struct({"kind": Text(("mtconnect", "opcua", "focas", "csv")), "path": Text(),
                            "mapping": MapOf(Text(), meta=False)}, required=("kind",), meta=False, nullable=True),
    # a calibration overlay (format cnctwin-calibration, written by twin calibrate) applied at resolve time
    "calibration": FilePath(exts=(".yaml", ".yml"), nullable=True,
                            doc="a cnctwin-calibration overlay (twin calibrate's calibration.yaml), relative to the "
                                "project: its values are applied to the machine, material and tools at resolve time"),
}, required=("format", "version", "name", "machine", "material", "stock", "tools", "coolant", "operations"),
    meta=False, hints={"program": "write operations: [{name: OP10, program: <file>}] (or version: 0 for the flat layout)",
                       "tool_list": "write tools: {list: [...]} (or version: 0 for the flat layout)"})


def physics_schema(domain: str) -> Node | None:
    """The declared PHYSICS_SCHEMA of a domain, if it has one."""
    return DomainPhysics(domain).declared()


__all__ = ["PROJECT", "PHYSICS", "RESOLVED", "TOOL_ENTRY", "PROBE_ENTRY", "TOOL_LIST_ENTRY", "INLINE_PROBE", "RESOLVED_PROBE",
           "WORKHOLDING_ENTRY", "OPERATION", "DbRef", "FORMAT", "CURRENT_VERSION", "DOMAINS", "TICK_S", "is_path_ref",
           "physics_schema"]
