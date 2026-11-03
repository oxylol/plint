"""Elastic and thermal constants the structural models read (docs/models/structure.md section 9).

* workpiece: the project's material (SI view): Young's modulus at 20 degC (a temperature table is read at
  293.15 K: the FE lives in the material configuration at 20 degC, architecture 4.1), Poisson's ratio, density,
  the mean expansion coefficient from 20 degC as a function of temperature, and the residual-stress profiles of
  the stock forms;
* tool substrate: db/tool_materials/<id>.yaml of the tool item (the database object when the run has one, else
  the file), falling back to tooling's representative grade of the tool's class;
* holders, turning shanks and steel bars: db/models/structure-tool.yaml (carbon steel).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..core.errors import InputError
from ..db.simap import Curve, SIMap, Sourced, Table
from ..paths import DB
from ..units import REFERENCE_TEMP_K
from . import params as P


@dataclass(frozen=True)
class Elastic:
    E_Pa: float
    nu: float
    rho_kg_m3: float
    name: str = ""


@dataclass
class WorkMaterial:
    E_Pa: float
    nu: float
    rho_kg_m3: float
    alpha_table: object = None          # callable T_K -> mean CTE from 20 degC (1/K), or a float
    residual: list = field(default_factory=list)   # [{form, thickness_m, direction, x [n], sigma_Pa [n]}]
    name: str = ""

    def alpha_1_K(self, T_K):
        a = self.alpha_table
        if a is None:
            return 0.0 * T_K
        if callable(a):
            return a(T_K)
        return float(a) + 0.0 * T_K


def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _number(value, at_K: float = REFERENCE_TEMP_K):
    if value is None:
        return None
    if isinstance(value, Sourced):
        value = value.value
    if isinstance(value, Table):
        return float(value(at_K))
    if isinstance(value, Curve):
        return float(value(at_K))
    if isinstance(value, Mapping) and "value" in value:
        return _number(value.get("value"), at_K)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _alpha(value):
    if isinstance(value, Sourced):
        value = value.value
    if isinstance(value, (Table, Curve)):
        return value
    number = _number(value)
    return number


def _profile(entry) -> dict | None:
    prof = _get(entry, "profile_Pa")
    if isinstance(prof, Sourced):
        prof = prof.value
    if prof is None:
        return None
    rows = [list(r) for r in prof]
    if not rows:
        return None
    x = [float(_number(r[0]) if not isinstance(r[0], (int, float)) else r[0]) for r in rows]
    s = [float(_number(r[1]) if not isinstance(r[1], (int, float)) else r[1]) for r in rows]
    order = sorted(range(len(x)), key=lambda i: x[i])
    return {"form": str(_get(entry, "form") or ""), "thickness_m": _number(_get(entry, "thickness_m")),
            "direction": str(_get(entry, "direction") or ""), "x": [x[i] for i in order],
            "sigma_Pa": [s[i] for i in order]}


def work_material(material) -> WorkMaterial:
    """The project's material (SI view or mapping with a ``physical`` section)."""
    if material is None:
        raise InputError("structure: the project has no material")
    phys = _get(material, "physical")
    if phys is None:
        raise InputError("structure: the material has no physical section")
    E = _number(_get(phys, "youngs_modulus_Pa"))
    nu = _number(_get(phys, "poisson"))
    rho = _number(_get(phys, "density_kg_m3"))
    if E is None or nu is None or rho is None:
        raise InputError("structure: the material needs youngs_modulus, poisson and density_kg_m3")
    residual = []
    for entry in _get(material, "residual_stress") or []:
        prof = _profile(entry)
        if prof is not None:
            residual.append(prof)
    name = str(_get(material, "id") or _get(material, "name") or "")
    return WorkMaterial(E, nu, rho, _alpha(_get(phys, "expansion_1_K")), residual, name)


def holder_steel() -> Elastic:
    return Elastic(P.p(P.TOOL, "holder_youngs_modulus_Pa"), P.p(P.TOOL, "holder_poisson"),
                   P.p(P.TOOL, "holder_density_kg_m3"), "holder steel")


@functools.lru_cache(maxsize=None)
def _tool_material_file(material_id: str) -> SIMap | None:
    path = DB / "tool_materials" / f"{material_id}.yaml"
    if not path.is_file():
        return None
    from ..db.yamlio import load_file
    data = load_file(path).data
    return SIMap(data) if isinstance(data, Mapping) else None


def _tool_material_item(material_id: str, db):
    if db is not None and hasattr(db, "get"):
        try:
            item = db.get("tool_material", material_id)
            if item is not None:
                return item
        except Exception:  # noqa: BLE001 - a database without the item falls back to the file
            pass
    return _tool_material_file(material_id)


def tool_substrate(material_id: str | None, material_class: str | None, db=None) -> Elastic:
    """E, nu, rho of a tool substrate: its own material item, else the representative item of its class."""
    candidates = []
    if material_id:
        candidates.append(material_id)
    from ..db.models import params as model_params
    by_class = model_params("tooling").get("tool_material_by_class") or {}
    if material_class and material_class in by_class:
        candidates.append(by_class[material_class])
    for mid in candidates:
        item = _tool_material_item(str(mid), db)
        if item is None:
            continue
        E = _number(_get(item, "youngs_modulus_Pa"))
        nu = _number(_get(item, "poisson"))
        rho = _number(_get(item, "density_kg_m3"))
        if E is not None and nu is not None and rho is not None:
            return Elastic(E, nu, rho, str(mid))
    raise InputError(f"structure: no elastic constants for tool material {material_id!r} (class {material_class!r})")


__all__ = ["Elastic", "WorkMaterial", "work_material", "holder_steel", "tool_substrate"]
