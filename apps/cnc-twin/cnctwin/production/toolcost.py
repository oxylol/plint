"""Tool cost per part: ``price / (regrinds + 1)`` per edge life (one new edge plus each regrind),
times the share of that life one part uses.

``life_used_frac`` is the part's increment of the tool's life used (wear to the life criterion, as
the wear domain reports it in tools.json, or cutting time over the project's ``life.limit_min``);
``parts_per_edge = 1 / life_used_frac``. Regrinding service and tool-holder costs are not part of
the formula (the cost is the lower bound by the regrind service price).
"""

from __future__ import annotations


def tool_cost(price_eur: float | None, regrinds: int | None, life_used_frac: float | None) -> dict:
    """Cost of one part for one tool (None where price or life is unknown, with the reason;
    ``parts_per_edge`` None when the part uses no life)."""
    regrinds = int(regrinds or 0)
    lives = regrinds + 1
    result = {"price_eur": price_eur, "regrinds": regrinds, "edge_lives": lives, "life_used_frac": life_used_frac,
              "cost_per_edge_eur": None if price_eur is None else price_eur / lives,
              "parts_per_edge": None, "cost_per_part_eur": None, "missing": []}
    if price_eur is None:
        result["missing"].append("price_eur")
    if life_used_frac is None:
        result["missing"].append("life used per part")
    if life_used_frac is not None and life_used_frac > 0.0:
        result["parts_per_edge"] = 1.0 / life_used_frac
    if price_eur is not None and life_used_frac is not None:
        result["cost_per_part_eur"] = price_eur / lives * life_used_frac
    return result
