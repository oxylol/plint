"""Shared pieces of the harnesses: database objects of a case, per-test conditions, provenance.

Every value a harness feeds into a model comes from the case (its stated inputs, including the ones it
marks as assumed) or from the databases; anything else is a documented assumption added to the
output (``Assumption``). Database values the model chain uses are listed as ``Provenance`` with their
sources, so the runner can tell calibration-dependent cases.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from ...db.simap import wrap
from ...units import REFERENCE_TEMP_K
from ..cases import Case, TestRow, ambient_K, find_test, per_test_table, to_si
from ..registry import Assumption, Provenance

DEG = math.pi / 180.0

# Tool grades named in the cases mapped to database tool materials (the database's own names list
# these grades as references; every use is reported as an assumption).
GRADE_MATERIALS: tuple[tuple[str, str, str], ...] = (
    ("h13a", "carbide-k20-6co", "db/tool_materials/carbide-k20-6co.yaml names Sandvik H13A (straight WC-6Co)"),
    ("wc-co", "carbide-k20-6co", "uncoated straight WC-Co: the straight WC-6Co K20 entry"),
    ("tungsten carbide", "carbide-k20-6co", "uncoated straight WC-Co: the straight WC-6Co K20 entry"),
    ("k313", "carbide-k20-6co", "Kennametal K313 is an uncoated straight WC-Co K grade"),
    ("kc5010", "carbide-k20-6co", "Kennametal KC5010: PVD TiAlN on a straight WC-Co K substrate"),
    ("1115", "carbide-submicron-10co", "Sandvik 1115: PVD-coated fine-grained S-grade substrate"),
    ("sialon", "ceramic-sialon", "Si3N4-based sialon ceramic"),
    ("sx9", "ceramic-sialon", "NTK SX9 is a sialon"),
    ("whisker", "ceramic-whisker", "SiC-whisker reinforced alumina"),
    ("wg300", "ceramic-whisker", "Greenleaf WG300 is SiC-whisker reinforced alumina"),
    ("cbn7020", "cbn-low", "Sandvik CBN7020 (57 % cBN, ceramic binder) is a low-content PCBN"),
    ("57 % cbn", "cbn-low", "57 % cBN with ceramic binder is a low-content PCBN"),
    ("t1200a", "cermet-ticn", "Sumitomo T1200A is a TiCN-based cermet"),
    ("cermet", "cermet-ticn", "TiCN-based cermet"),
    ("hss-e-pm", "hss-m42", "powder-metallurgical cobalt HSS: the HSS-Co (M42) entry"),
    ("hss", "hss-m2", "HSS: the M2 entry"),
)
CLASS_MATERIALS = {"carbide": "carbide-k20-6co", "cermet": "cermet-ticn", "ceramic": "ceramic-al2o3-tic",
                   "cbn": "cbn-low", "hss": "hss-m2", "pcd": "pcd"}

COATINGS: tuple[tuple[str, str], ...] = (
    ("tialn", "tialn"), ("altin", "altin"), ("alcrn", "alcrn"), ("ticn", "ticn"), ("tin", "tin"),
    ("al2o3", "al2o3-cvd"), ("cvd", "al2o3-cvd"), ("dlc", "dlc"), ("diamond", "diamond-cvd"),
)


def _text(value) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    return "" if value is None else str(value)


@dataclass
class Collector:
    """Assumptions, provenance and notes gathered while a harness builds its inputs."""

    assumptions: list = field(default_factory=list)
    provenance: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def assume(self, what: str, value, reason: str, source: str = "") -> None:
        text = value if isinstance(value, str) else (f"{value:.6g}" if isinstance(value, float) else str(value))
        item = Assumption(what, text, reason, source)
        if item not in self.assumptions:
            self.assumptions.append(item)

    def used(self, what: str, src: str, note: str = "") -> None:
        item = Provenance(what, src or "", note or "")
        if item not in self.provenance:
            self.provenance.append(item)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


# ------------------------------------------------------------------------------------------ database sources
def leaf_source(raw, inherited: str = "") -> tuple[str, str]:
    """(src, note) of a database leaf written as {value, src, note} or plainly (section src)."""
    if isinstance(raw, Mapping):
        return str(raw.get("src") or inherited or ""), str(raw.get("note") or "")
    return inherited, ""


def material_provenance(raw: Mapping, col: Collector, *, flow=True, friction_entry=None, chip_thickness=True,
                        physical=True, tool_life_entry=None, extra: tuple = ()) -> None:
    """Record the material values a mechanics model chain uses (with their sources)."""
    mid = raw.get("id")
    if flow and isinstance(raw.get("flow_stress"), Mapping):
        fs = raw["flow_stress"]
        col.used(f"{mid} flow stress (Johnson-Cook)", str(fs.get("src") or ""), str(fs.get("note") or ""))
    if physical and isinstance(raw.get("physical"), Mapping):
        ph = raw["physical"]
        base = str(ph.get("src") or "")
        for key in ("density_kg_m3", "specific_heat_J_kgK", "conductivity_W_mK"):
            if key in ph:
                src, note = leaf_source(ph[key], base)
                col.used(f"{mid} physical.{key}", src, note)
    if chip_thickness:
        cut = raw.get("cutting") or {}
        if "min_chip_thickness_ratio" in cut:
            src, note = leaf_source(cut["min_chip_thickness_ratio"])
            col.used(f"{mid} cutting.min_chip_thickness_ratio", src, note)
    if friction_entry is not None:
        for entry in raw.get("friction") or []:
            if str(entry.get("tool_class")) == friction_entry.tool_class and \
                    str(entry.get("coating")) == friction_entry.coating:
                src, note = leaf_source(entry.get("mu_sliding"), str(entry.get("src") or ""))
                col.used(f"{mid} friction mu_sliding ({entry.get('tool_class')}, {entry.get('coating')})", src, note)
                break
    if tool_life_entry is not None:
        items = raw.get("tool_life") or []
        if 0 <= tool_life_entry < len(items):
            e = items[tool_life_entry]
            col.used(f"{mid} tool_life[{tool_life_entry}] ({e.get('tool_class')}, {e.get('coating')})",
                     str(e.get("src") or ""), str(e.get("note") or ""))
    for path in extra:
        node = raw
        for part in path.split("."):
            node = node.get(part) if isinstance(node, Mapping) else None
        if node is not None:
            src, note = leaf_source(node)
            col.used(f"{mid} {path}", src, note)


def raw_object(db, kind: str, item_id: str) -> dict:
    obj = db.object(kind, item_id)
    merged = getattr(obj, "merged", None)
    return dict(merged if merged is not None else obj.raw)


# ------------------------------------------------------------------------------------------ work and tool
def work_material(db, case: Case, col: Collector):
    """(WorkMaterial, raw material dict) of the case's database material."""
    from ...mechanics.materials import WorkMaterial
    mid = case.material_id
    if mid is None:
        raise ValueError(f"{case.id}: the case names no database material")
    simap = db.material(mid)
    raw = raw_object(db, "material", mid)
    note = _text((case.setup.get("material") or {}).get("note"))
    if note:
        col.note(f"Material: database entry {mid}; the case says: {note}")
    return WorkMaterial.from_db(simap), raw


def tool_material_id(case: Case, col: Collector) -> str:
    """Database tool material for the case's tool (grade keywords, else the class default)."""
    tool = case.section("tool")
    words = " ".join(_text(tool.get(k)) for k in ("grade", "grades", "tool_material", "maker_tool", "designation"))
    words = words.lower()
    for key, mat, why in GRADE_MATERIALS:
        if key in words:
            col.assume("tool material", mat, why, "db/tool_materials")
            return mat
    cls = _text(tool.get("tool_class")).lower() or "carbide"
    mat = CLASS_MATERIALS.get(cls, CLASS_MATERIALS["carbide"])
    col.assume("tool material", mat, f"grade not in the database: the {cls} class default", "db/tool_materials")
    return mat


def coating_id(case: Case, col: Collector, text: str | None = None) -> str:
    """Database coating id of the case's tool coating ('none' when uncoated or not stated)."""
    raw = text if text is not None else _text(case.section("tool").get("coating"))
    low = raw.lower()
    if not low or low in ("none", "uncoated") or "uncoated" in low or "none" in low:
        if not low or "not stated" in low:
            col.assume("coating", "none", "not stated in the case: uncoated assumed", "case")
        return "none"
    if "not stated" in low:
        col.assume("coating", "none", "not stated in the case: uncoated assumed", "case")
        return "none"
    for key, cid in COATINGS:
        if key in low:
            if key != low.replace(" ", "").replace("-", ""):
                col.assume("coating", cid, f"case coating {raw!r} mapped to the database coating {cid}", "db/coatings")
            return cid
    col.assume("coating", "none", f"case coating {raw!r} has no database entry: modelled uncoated", "db/coatings")
    return "none"


def tool_side(db, case: Case, col: Collector, *, tool_mat: str | None = None, coat: str | None = None,
              insert: dict | None = None, kind: str = "turning_insert"):
    """(ToolSide, tool material id, coating id) for the mechanics models."""
    from ...mechanics.materials import ToolSide
    tm = tool_mat or tool_material_id(case, col)
    cid = coat if coat is not None else coating_id(case, col)
    tm_obj = db.tool_material(tm)
    tm_raw = raw_object(db, "tool_material", tm)
    col.used(f"tool material {tm} thermal properties", str(tm_raw.get("src") or ""),
             _text((tm_raw.get("conductivity_W_mK") or {}).get("note") if isinstance(tm_raw.get("conductivity_W_mK"),
                                                                                       Mapping) else ""))
    coat_obj = None if cid == "none" else db.coating(cid)
    item = {"type": kind, "coating": {"id": cid}, "geometry": {"insert": insert or {}}}
    return ToolSide.from_db(wrap(item), tm_obj, coat_obj), tm, cid


EDGE_DEFAULTS = {"insert": "insert_edge_radius_m", "solid": "hone_edge_radius_m", "chamfer": "chamfer_corner_radius_m",
                 "sharp": "sharp_edge_radius_m"}


def edge_radius_m(case: Case, row: TestRow | None, col: Collector, tool_kind: str = "insert") -> float:
    """Cutting-edge radius of a test: per-test list, else the tool's stated value, else the default the
    tooling model itself uses for that kind of tool (db/models/tooling.yaml). ``tool_kind`` "solid" follows
    tooling.common.edge_prep for a solid tool without a stated edge preparation: the light hone when coated,
    the ground (sharp) edge when uncoated."""
    if row is not None:
        value = row.si("edge_radius_per_test_um", "edge_radius_um")
        if isinstance(value, (int, float)):
            return float(value)
    leaf = case.leaf("tool", "edge_radius_um", "edge_radius_mm", "hone_um")
    if leaf is not None and isinstance(leaf.value, (int, float)):
        return float(leaf.si())
    if tool_kind == "solid" and coating_id(case, col) == "none":
        tool_kind = "sharp"
        col.note("Edge radius: an uncoated solid tool without a stated edge preparation has the ground (sharp) edge "
                 "in the tooling model (tooling.common.edge_prep), not the light hone of coated solid tools.")
    from ...db.models import params
    name = EDGE_DEFAULTS[tool_kind]
    value = float(params("tooling")[name])
    col.assume("cutting-edge radius", f"{value * 1e6:.0f} um", f"not stated in the case: the tooling model's default "
               f"for this tool ({name})", "db/models/tooling.yaml")
    return value


def rake_rad(case: Case, row: TestRow | None, col: Collector) -> float:
    if row is not None:
        value = row.raw("rake_per_test_deg", "rake_deg")
        if isinstance(value, (int, float)):
            return float(value) * DEG
    leaf = case.leaf("tool", "rake_deg", "rake_land_deg")
    if leaf is not None and isinstance(leaf.value, (int, float)):
        return float(leaf.value) * DEG
    raise ValueError(f"{case.id}: the case gives no rake angle")


def clearance_rad(case: Case, col: Collector, default_deg: float = 6.0) -> float:
    leaf = case.leaf("tool", "clearance_deg")
    if leaf is not None and isinstance(leaf.value, (int, float)):
        return float(leaf.value) * DEG
    col.assume("clearance angle", f"{default_deg:g} deg", "not stated in the case", "est")
    return default_deg * DEG


def speed_m_s(row: TestRow) -> float | None:
    value = row.si("vc_m_min", "cutting_speed_m_min", "vc_m_s")
    return float(value) if isinstance(value, (int, float)) else None


def uncut_thickness_m(row: TestRow) -> float | None:
    value = row.si("h_mm", "feed_mm_rev", "feed_mm", "uncut_chip_thickness_um", "f_mm", "h_um")
    return float(value) if isinstance(value, (int, float)) else None


def width_m(case: Case, row: TestRow | None) -> float | None:
    if row is not None:
        value = row.si("width_mm", "width_of_cut_mm")
        if isinstance(value, (int, float)):
            return float(value)
    leaf = case.leaf("conditions", "width_of_cut_mm", "ap_mm")
    if leaf is not None and isinstance(leaf.value, (int, float)):
        return float(leaf.si())
    return None


def dry(case: Case) -> bool:
    text = _text(case.section("conditions").get("coolant")).lower()
    return text in ("", "dry") or text.startswith("dry")


def test_rows(case: Case) -> list[TestRow]:
    return per_test_table(case)


def label(entry: Mapping) -> str:
    """The test label of a reference name: 'chip_ratio V0072 (derived)' -> 'V0072'."""
    tokens = [t for t in re.split(r"[\s(),;:]+", str(entry.get("name", ""))) if t]
    rest = [t for t in tokens[1:] if t.lower() not in ("derived",)]
    return " ".join(rest)


def ambient(case: Case, col: Collector) -> float:
    temp, assumed = ambient_K(case)
    if assumed:
        col.assume("ambient temperature", f"{temp - 273.15:.0f} degC",
                   "case states none (or marks it assumed): ISO 1 reference temperature" if temp == REFERENCE_TEMP_K
                   else "as the case assumes", "case" if temp != REFERENCE_TEMP_K else "est")
    return temp


__all__ = ["Collector", "work_material", "tool_side", "tool_material_id", "coating_id", "material_provenance",
           "edge_radius_m", "rake_rad", "clearance_rad", "speed_m_s", "uncut_thickness_m", "width_m", "dry",
           "test_rows", "find_test", "label", "ambient", "raw_object", "leaf_source", "budget_summary", "DEG", "to_si"]


def budget_summary(q) -> str:
    """Plain statement of which independent cases meet the budget, and how many values each has within
    tolerance (computed from the results, for the discussions)."""
    indep = [r for r in q.cases if r.status == "ok" and not r.calibration_dependent and r.stats is not None]
    if not indep or q.budget is None:
        return ""
    ok = [r.id for r in indep if r.stats.passes]
    parts = []
    for r in indep:
        counted = [c for c in r.comparisons if c.counted]
        inside = sum(1 for c in counted if c.within_tolerance)
        parts.append(f"{r.id} {inside}/{len(counted)}")
    head = (f"{len(ok)} of {len(indep)} independent cases within budget"
            + (f" ({', '.join(ok)})" if ok else "") + ". Values within tolerance per case: " + "; ".join(parts) + ". ")
    return head
