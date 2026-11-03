"""Error budgets of validation/budgets.yaml and how they are applied (docs/validation.md "Budgets").

The file states its rule in its header; the suite applies exactly that rule:

* tolerance of one reference value or curve point = max(budget_pct / 100 x |reference|,
  2 x uncertainty of that reference value); for temperatures the percentage applies to the rise
  above ambient (T - T_ambient), not to T;
* a case passes when every compared value lies within its tolerance;
* a quantity passes when at least 80 % of its cases pass and no case misses by more than twice its
  tolerance.

The two numbers of the quantity rule (80 %, twice) are written in the header comment of the file, not
as data; they are the constants below, and a unit test checks that the header still states them.
A budget with ``components`` (cutting_force) needs the harness to name the component of each
prediction (:attr:`cnctwin.validation.registry.Prediction.component`).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..paths import VALIDATION

BUDGETS_FILE = VALIDATION / "budgets.yaml"
QUANTITY_PASS_FRACTION = 0.8          # "a quantity passes when at least 80 % of its cases pass"
MAX_MISS_FACTOR = 2.0                 # "... and no case misses by more than twice its tolerance"
UNCERTAINTY_FACTOR = 2.0              # "max(budget_pct / 100 * |reference|, 2 * uncertainty ...)"
RULE_TEXT = ("tolerance = max(budget_pct/100 x |reference| (for temperatures: x the rise above ambient), "
             "2 x the reference's uncertainty); a case passes when every compared value lies within its tolerance; "
             "a quantity passes when at least 80 % of its cases pass and no case misses by more than twice its "
             "tolerance")


@dataclass(frozen=True)
class Budget:
    quantity: str
    metric: str
    pct: float | None                              # single budget (budget_pct)
    components: dict = field(default_factory=dict)  # component name -> pct (budgets with components)
    src: str = ""
    note: str = ""

    def pct_for(self, component: str | None) -> float | None:
        """Budget percentage for a prediction's component (the single budget when there are none)."""
        if self.components:
            if component is None:
                return None
            return self.components.get(component)
        return self.pct

    def describe(self) -> str:
        if self.components:
            return ", ".join(f"{k} {v:g} %" for k, v in sorted(self.components.items()))
        return f"{self.pct:g} %" if self.pct is not None else "none"


def _number(entry) -> tuple[float | None, str, str]:
    if isinstance(entry, dict):
        value = entry.get("value")
        return (None if value is None else float(value)), str(entry.get("src") or ""), str(entry.get("note") or "")
    if entry is None:
        return None, "", ""
    return float(entry), "", ""


@functools.lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_budgets(path: str | Path | None = None) -> dict[str, Budget]:
    """Budgets by quantity (only the quantities the file defines)."""
    data = _load(str(path or BUDGETS_FILE))
    out: dict[str, Budget] = {}
    for quantity, entry in sorted((data.get("budgets") or {}).items()):
        if not isinstance(entry, dict):
            continue
        metric = str(entry.get("metric") or "")
        if "components" in entry:
            comps = {}
            notes = []
            for name, comp in sorted((entry.get("components") or {}).items()):
                value, src, note = _number(comp)
                if value is not None:
                    comps[name] = value
                    notes.append(f"{name}: {note}" if note else name)
            out[quantity] = Budget(quantity, metric, None, comps, "est", " | ".join(notes))
        else:
            value, src, note = _number(entry.get("budget_pct"))
            out[quantity] = Budget(quantity, metric, value, {}, src, note)
    return out


def header_text(path: str | Path | None = None) -> str:
    """The comment header of the budgets file (where the application rule is stated)."""
    lines = []
    with open(str(path or BUDGETS_FILE), encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            lines.append(line[1:].strip())
    return " ".join(lines)


def tolerance(budget_pct: float | None, scale: float, uncertainty: float | None) -> float | None:
    """Tolerance of one reference value (absolute, reference unit): ``scale`` is |reference| (or the
    rise above ambient for temperatures). None without a budget."""
    if budget_pct is None:
        return None
    tol = budget_pct / 100.0 * abs(scale)
    if uncertainty is not None:
        tol = max(tol, UNCERTAINTY_FACTOR * abs(uncertainty))
    return tol


__all__ = ["Budget", "load_budgets", "header_text", "tolerance", "BUDGETS_FILE", "QUANTITY_PASS_FRACTION",
           "MAX_MISS_FACTOR", "UNCERTAINTY_FACTOR", "RULE_TEXT"]
