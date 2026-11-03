"""What the sanity rules know besides the program: machine, tool table, setup geometry, material.

:class:`SanityContext` is plain data in SI. :meth:`SanityContext.from_project` builds it from the
resolved project (docs/architecture.md 18) or from raw project/database mappings.

Geometry convention (docs/models/sanity.md, "Setup geometry"): for a 3-axis mill the machine
coordinates of a point fixed to the table are its coordinates in the table frame, the frame in
which ``fixture_pose`` and the work offsets are written (db/_templates/project.yaml). The stock box
in machine coordinates is ``fixture_pose * stock_pose * box_S``; program coordinates of work offset
``G5x`` are machine coordinates minus the offset. Callers that know better (the kinematics) pass
``stock_box_M`` and ``clamps_M`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..db.models import params as model_params
from ..db.simap import pose_matrix as db_pose_matrix
from .si import get, items, num, plain, si_view, text

LATHE_TYPES = ("lathe", "swiss")
MILL_TYPES = ("vertical_mill", "horizontal_mill", "five_axis_mill")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box (SI) in machine coordinates unless stated."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    name: str = ""

    def shifted(self, offset) -> "Box":
        o = np.asarray(offset, dtype=float)
        return Box(tuple(np.asarray(self.lo) - o), tuple(np.asarray(self.hi) - o), self.name)

    @property
    def top(self) -> float:
        return float(self.hi[2])


@dataclass
class CuttingData:
    """One maker's recommendation (SI): speeds m/s, feeds m (per tooth or per rev), lengths m."""

    iso_group: str
    material: str | None = None
    vc_m_s: tuple[float | None, float | None, float | None] = (None, None, None)   # min, start, max
    fz_m: tuple[float | None, float | None, float | None] = (None, None, None)
    f_m: tuple[float | None, float | None, float | None] = (None, None, None)
    ap_max_m: float | None = None
    ap_max_xD: float | None = None
    ae_max_m: float | None = None
    ae_max_xD: float | None = None
    operation: tuple[str, ...] = ()
    coolant: str | None = None
    page: str | None = None


@dataclass
class ToolFacts:
    """One tool of the project's tool table, as the sanity rules need it (SI)."""

    number: int
    tool_id: str = ""
    type: str | None = None
    hand: str | None = None
    tool_class: str | None = None
    diameter_m: float | None = None
    flutes: int | None = None
    corner_radius_m: float = 0.0
    point_angle_rad: float | None = None
    flute_length_m: float | None = None
    cutting_length_m: float | None = None
    stickout_m: float | None = None
    length_offset_m: float | None = None
    radius_offset_m: float | None = None
    coolant_holes: bool = False
    centre_cutting: bool | None = None
    thread_pitch_m: float | None = None
    chamfer_threads: float | None = None
    cutting_data: list[CuttingData] = field(default_factory=list)
    comment: str | None = None
    max_rad_s: float | None = None
    price_eur: float | None = None
    regrinds: int | None = None
    resolved: bool = True            # False when the tool item could not be read (db id only)

    @property
    def radius_m(self) -> float | None:
        return None if self.diameter_m is None else 0.5 * self.diameter_m

    @property
    def usable_length_m(self) -> float | None:
        """The length that may go into the material: flute length, else cutting length."""
        values = [v for v in (self.flute_length_m, self.cutting_length_m) if v is not None]
        return max(values) if values else None

    @property
    def is_drill(self) -> bool:
        return self.type in ("twist_drill", "indexable_drill", "gun_drill", "spot_drill", "centre_drill")

    @property
    def is_tap(self) -> bool:
        return self.type in ("tap_cutting", "tap_forming")

    @property
    def rotating(self) -> bool:
        return self.type not in (None, "turning_insert", "grooving_insert", "parting_insert", "threading_insert",
                                 "form_tool", "boring_bar")


@dataclass
class SanityContext:
    """Machine, tools, setup and material for one operation's program (SI)."""

    dialect: str = "fanuc"
    machine_kind: str = "mill"                # mill | lathe | mill_turn (decided per program)
    machine_type: str = "vertical_mill"
    least_increment_m: float = 1e-6
    travel_m: dict[str, tuple[float, float]] = field(default_factory=dict)
    home_m: dict[str, float] = field(default_factory=dict)
    pump_kinds: set[str] = field(default_factory=set)
    pump_ids: set[str] = field(default_factory=set)
    nozzle_ids: set[str] = field(default_factory=set)
    has_air_blast: bool = False
    has_programmable_nozzle: bool = False
    arc_alarm_tolerance_m: float | None = None
    tools: dict[int, ToolFacts] = field(default_factory=dict)
    stock_box_M: Box | None = None
    stock_kind: str = "block"
    stock_radius_m: float | None = None
    clamps_M: list[Box] = field(default_factory=list)
    clamps_unknown: list[str] = field(default_factory=list)
    work_offsets_m: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    material_id: str = ""
    material_iso_group: str | None = None
    material_family: str | None = None
    coolant_kind: str | None = None
    initial_tool: int | None = None
    op_name: str = ""
    program_name: str = ""
    params: object = None

    def __post_init__(self):
        if self.params is None:
            self.params = model_params("sanity-rules")

    def p(self, name: str):
        return self.params[name]

    def stock_box_W(self, work_offset: str) -> Box | None:
        offset = self.work_offsets_m.get(normalise_offset(work_offset))
        if self.stock_box_M is None or offset is None:
            return None
        return self.stock_box_M.shifted(offset)

    def clamps_W(self, work_offset: str) -> list[Box] | None:
        offset = self.work_offsets_m.get(normalise_offset(work_offset))
        if offset is None:
            return None
        return [box.shifted(offset) for box in self.clamps_M]

    @classmethod
    def from_project(cls, project, op_index: int = 0, *, stock_box_M: Box | None = None,
                     clamps_M: list[Box] | None = None, params=None) -> "SanityContext":
        """Context of operation ``op_index`` of a resolved project (or raw project mappings whose
        machine, material, tools and coolant fluid are given as mappings)."""
        project = si_view(project)
        machine = si_view(get(project, "machine"))
        material = si_view(get(project, "material"))
        operation = si_view(get(project, "operations", op_index))
        setup = si_view(get(operation, "setup"))
        ctx = cls(params=params or model_params("sanity-rules"))
        ctx.op_name = text(get(operation, "name"), "")
        ctx.program_name = text(get(operation, "program"), "")
        _read_machine(ctx, machine, project)
        _read_material(ctx, material)
        ctx.coolant_kind = text(get(si_view(get(project, "coolant")), "fluid", "kind"))
        for entry in items(get(project, "tools", "list")):
            facts = tool_facts(si_view(entry))
            ctx.tools[facts.number] = facts
        offsets = get(setup, "work_offsets_m") or {}
        for code, value in dict(offsets).items():
            vector = plain(value)
            if isinstance(vector, (list, tuple)) and len(vector) == 3 and all(v is not None for v in vector):
                ctx.work_offsets_m[normalise_offset(str(code))] = tuple(float(v) for v in vector)
        initial = get(setup, "initial_tool")
        ctx.initial_tool = int(num(initial)) if num(initial) is not None else None
        stock = si_view(get(project, "stock"))
        ctx.stock_kind = text(get(stock, "kind"), "block")
        transform = _setup_transform(setup)
        box_S = stock_box_S(stock)
        if stock_box_M is not None:
            ctx.stock_box_M = stock_box_M
        elif box_S is not None:
            ctx.stock_box_M = transform_box(transform, box_S, "stock")
        if ctx.stock_kind in ("cylinder", "tube"):
            size = plain(get(stock, "size_m")) or []
            ctx.stock_radius_m = 0.5 * float(size[0]) if size else None
        if clamps_M is not None:
            ctx.clamps_M = list(clamps_M)
        else:
            _read_clamps(ctx, setup, box_S, transform)
        return ctx


def normalise_offset(code: str) -> str:
    """``g54.1 p3`` -> ``G54.1P3``; ``G54`` stays."""
    return "".join(str(code).upper().split())


def _read_machine(ctx: SanityContext, machine, project) -> None:
    controller = get(machine, "controller")
    overrides = si_view(get(project, "controller_overrides"))
    ctx.dialect = text(get(overrides, "dialect"), None) or text(get(controller, "dialect"), "fanuc")
    ctx.machine_type = text(get(machine, "type"), "vertical_mill")
    ctx.machine_kind = "lathe" if ctx.machine_type in LATHE_TYPES else (
        "mill_turn" if ctx.machine_type == "mill_turn" else "mill")
    increment = num(get(overrides, "least_input_increment_m")) or num(get(controller, "least_input_increment_m"))
    ctx.least_increment_m = increment if increment else float(ctx.p("default_least_increment_m"))
    tolerance = num(get(overrides, "arc_radius_tolerance_m")) or num(get(controller, "arc_radius_tolerance_m"))
    ctx.arc_alarm_tolerance_m = tolerance
    axes = get(machine, "axes") or {}
    for name in axes:
        axis = get(axes, name)
        travel = plain(get(axis, "travel_m"))
        if text(get(axis, "type"), "linear") == "linear" and isinstance(travel, (list, tuple)) and len(travel) == 2:
            ctx.travel_m[str(name)] = (float(min(travel)), float(max(travel)))
    home = get(machine, "kinematics", "home_m") or {}
    for name in home:
        value = num(get(home, name))
        if value is not None:
            ctx.home_m[str(name)] = value
    coolant = get(machine, "coolant")
    for pump in items(get(coolant, "pumps")):
        kind = text(get(pump, "kind"))
        if kind:
            ctx.pump_kinds.add(kind)
        ctx.pump_ids.add(text(get(pump, "id"), ""))
    for nozzle in items(get(coolant, "nozzles")):
        ctx.nozzle_ids.add(text(get(nozzle, "id"), ""))
        kind = text(get(nozzle, "kind"))
        if kind in ("mist", "air", "mql"):
            ctx.pump_kinds.add(kind)
        if get(nozzle, "programmable") is not None:
            ctx.has_programmable_nozzle = True
    ctx.has_air_blast = get(coolant, "air_blast") is not None


def _read_material(ctx: SanityContext, material) -> None:
    ctx.material_id = text(get(material, "id"), "")
    ctx.material_iso_group = text(get(material, "iso_group"))
    ctx.material_family = text(get(material, "family"))


def tool_facts(entry) -> ToolFacts:
    """:class:`ToolFacts` of one ``tools.list`` entry (the tool item resolved or inline)."""
    number = int(num(get(entry, "number"), 0))
    item = get(entry, "tool")
    facts = ToolFacts(number=number)
    facts.stickout_m = num(get(entry, "stickout_m"))
    facts.length_offset_m = num(get(entry, "length_offset_m"))
    facts.radius_offset_m = num(get(entry, "radius_offset_m"))
    facts.comment = text(get(entry, "comment"))
    if isinstance(plain(item), str):
        facts.tool_id = plain(item)
        facts.resolved = False
        return facts
    item = si_view(item)
    if item is None:
        facts.resolved = False
        return facts
    geometry = get(item, "geometry")
    facts.tool_id = text(get(item, "id"), "inline")
    facts.type = text(get(item, "type"))
    facts.hand = text(get(item, "hand"))
    facts.tool_class = text(get(item, "tool_material", "class"))
    facts.diameter_m = num(get(geometry, "diameter_m"))
    flutes = num(get(geometry, "flutes"))
    facts.flutes = int(flutes) if flutes else None
    facts.corner_radius_m = num(get(geometry, "corner_radius_m"), 0.0)
    if facts.type == "ball_endmill" and facts.diameter_m is not None and not facts.corner_radius_m:
        facts.corner_radius_m = 0.5 * facts.diameter_m
    facts.point_angle_rad = num(get(geometry, "point_angle_rad"))
    facts.flute_length_m = num(get(geometry, "flute_length_m"))
    facts.cutting_length_m = num(get(geometry, "cutting_length_m"))
    facts.coolant_holes = get(geometry, "coolant_holes") is not None
    centre = plain(get(geometry, "centre_cutting"))
    facts.centre_cutting = None if centre is None else bool(centre)
    facts.thread_pitch_m = num(get(geometry, "thread", "pitch_m"))
    facts.chamfer_threads = num(get(geometry, "chamfer_threads"))
    max_speed = num(get(item, "max_rad_s"))
    facts.max_rad_s = max_speed
    facts.price_eur = num(get(item, "price_eur"))
    regrinds = num(get(item, "regrinds"))
    facts.regrinds = int(regrinds) if regrinds is not None else None
    facts.cutting_data = [_cutting_data(si_view(c)) for c in items(get(item, "cutting_data"))]
    return facts


def _triple(value) -> tuple[float | None, float | None, float | None]:
    if value is None:
        return (None, None, None)
    number = num(value)
    if number is not None:
        return (None, number, None)
    return (num(get(value, "min")), num(get(value, "start")), num(get(value, "max")))


def _cutting_data(entry) -> CuttingData:
    operation = plain(get(entry, "operation"))
    if isinstance(operation, str):
        operation = (operation,)
    return CuttingData(
        iso_group=text(get(entry, "iso_group"), ""), material=text(get(entry, "material")),
        vc_m_s=_triple(get(entry, "vc_m_s")), fz_m=_triple(get(entry, "fz_m")), f_m=_triple(get(entry, "f_m")),
        ap_max_m=num(get(entry, "ap_max_m")), ap_max_xD=num(get(entry, "ap_max_xD")),
        ae_max_m=num(get(entry, "ae_max_m")), ae_max_xD=num(get(entry, "ae_max_xD")),
        operation=tuple(operation or ()), coolant=text(get(entry, "coolant")), page=text(get(entry, "page")))


# ----------------------------------------------------------------------------- setup geometry

def pose_matrix(pose) -> np.ndarray:
    """4x4 of a ``{translate_m, rotate_rad, order}`` pose in the project convention
    (:func:`cnctwin.db.simap.pose_matrix`: rotations about the moving axes in the written order)."""
    if pose is None:
        return np.eye(4)
    if hasattr(pose, "matrix") and callable(pose.matrix):
        return np.asarray(pose.matrix(), dtype=float)
    translate = plain(get(pose, "translate_m")) or [0.0, 0.0, 0.0]
    rotate = plain(get(pose, "rotate_rad")) or [0.0, 0.0, 0.0]
    return db_pose_matrix(translate, rotate, text(get(pose, "order"), "xyz"))


def _setup_transform(setup) -> np.ndarray:
    return pose_matrix(get(setup, "fixture_pose")) @ pose_matrix(get(setup, "stock_pose"))


def stock_box_S(stock) -> Box | None:
    """The stock's bounding box in its own frame S: a block from the origin corner, a cylinder or
    tube about +z_S from the bottom face centre."""
    size = plain(get(stock, "size_m"))
    kind = text(get(stock, "kind"), "block")
    if not isinstance(size, (list, tuple)) or not size:
        return None
    if kind == "block" and len(size) == 3:
        return Box((0.0, 0.0, 0.0), tuple(float(v) for v in size), "stock")
    if kind in ("cylinder", "tube") and len(size) >= 2:
        radius = 0.5 * float(size[0])
        length = float(size[-1])
        return Box((-radius, -radius, 0.0), (radius, radius, length), "stock")
    return None


def transform_box(matrix: np.ndarray, box: Box, name: str = "") -> Box:
    corners = np.array([[x, y, z] for x in (box.lo[0], box.hi[0]) for y in (box.lo[1], box.hi[1])
                        for z in (box.lo[2], box.hi[2])])
    moved = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return Box(tuple(moved.min(axis=0)), tuple(moved.max(axis=0)), name or box.name)


def _read_clamps(ctx: SanityContext, setup, box_S: Box | None, transform: np.ndarray) -> None:
    for index, entry in enumerate(items(get(setup, "workholding"))):
        entry = si_view(entry)
        device = get(entry, "device")
        device_view = si_view(device) if not isinstance(plain(device), str) else None
        name = text(get(device_view, "id")) if device_view is not None else text(device, f"workholding-{index}")
        outline = plain(get(entry, "outline_m"))
        if isinstance(outline, (list, tuple)) and len(outline) == 2:
            lo = np.minimum(outline[0], outline[1])
            hi = np.maximum(outline[0], outline[1])
            ctx.clamps_M.append(transform_box(transform, Box(tuple(lo), tuple(hi)), name))
            continue
        kind = text(get(device_view, "type")) or text(get(entry, "type"))
        if kind in ("vise", "soft_jaws") and box_S is not None:
            ctx.clamps_M.extend(_vise_jaws(ctx, entry, device_view, box_S, transform, name))
        elif kind in ("vacuum", "magnetic", "adhesive", "fixture_plate", "tabs"):
            continue
        elif kind in ("chuck_3jaw", "chuck_4jaw", "collet_chuck", "collet", "tailstock_centre", "steady_rest"):
            continue
        else:
            ctx.clamps_unknown.append(name or f"workholding-{index}")


def _vise_jaws(ctx, entry, device, box_S: Box, transform, name) -> list[Box]:
    grip = num(get(entry, "grip_depth_m"))
    size_y = box_S.hi[1] - box_S.lo[1]
    top = box_S.lo[2] + (grip if grip is not None else box_S.hi[2] - box_S.lo[2])
    jaw_height = num(get(device, "jaw_height_m")) or (top - box_S.lo[2])
    jaw_width = num(get(device, "jaw_width_m")) or (box_S.hi[0] - box_S.lo[0])
    thickness = float(ctx.p("vise_jaw_thickness_m"))
    centre_x = 0.5 * (box_S.lo[0] + box_S.hi[0])
    x0, x1 = centre_x - 0.5 * jaw_width, centre_x + 0.5 * jaw_width
    z0 = top - jaw_height
    fixed = Box((x0, box_S.lo[1] - thickness, z0), (x1, box_S.lo[1], top), f"{name} fixed jaw")
    moving = Box((x0, box_S.lo[1] + size_y, z0), (x1, box_S.lo[1] + size_y + thickness, top), f"{name} moving jaw")
    return [transform_box(transform, fixed), transform_box(transform, moving)]
