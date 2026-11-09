"""Operator overrides in a run (``options["operator"]``; docs/formats.md "Sweep", docs/ml.md "Overrides").

The feed, spindle and rapid keys of the control panel, held at a value for the whole run, globally or per tool::

    options["operator"] = {"feed_override": {"*": 1.0, "T1": 1.2, "T3@2": 0.9},   # fractions, 1 = 100 %
                           "spindle_override": {"*": 1.0, "T2": 0.8},
                           "rapid_override": 0.5}

``"*"`` is every tool not named, ``T<n>`` tool n of path 1, ``T<n>@<p>`` tool n of path p. They act where the
control applies them, on the commands the controller hands to motion (``MotionDomain.accept``), with the tool
that is active at each block (the controller's modal snapshot ``modal["tool"]``, so a resumed run applies them
identically):

* **feed**: the programmed feed of every feed move (G01/G02/G03, probing and tool-setter moves) whose ``Feed``
  allows the override (``override_allowed``: not in tapping, threading, G63, or skip moves without the skip-feed
  override) is multiplied, per minute, per revolution and inverse time alike; the planner's limits (the machine's
  maximum cutting feed, axis and curvature limits) still apply afterwards, as on the machine;
* **spindle**: the commanded speed of every spindle command (and the surface speed of constant-surface-speed
  commands) is multiplied and held within the spindle's maximum speed and the command's clamp (G50/G92 S); for
  taps and threading inserts (tool types ``tap_cutting``, ``tap_forming``, ``threading_insert``) it stays at
  100 %, as controls hold rigid tapping and threading at the programmed speed;
* **rapid**: the panel's rapid key (``OperatorPanel.rapid_override``, which motion applies to rapid moves); a value
  of the machine's ``rapid_override_steps_pct`` (or 100 %).

Overrides do not change the program or the project: the run records them in run.json ``options.operator``.
Validation (:func:`problems`) against the machine's ``feed_override_range_pct``, ``spindle_override_range_pct``
and ``rapid_override_steps_pct`` and the project's tool list happens before a sweep runs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

LOCKED_TOOL_TYPES = ("tap_cutting", "tap_forming", "threading_insert")
KEYS = ("feed_override", "spindle_override", "rapid_override")


def tool_key(number: int, path: int = 1) -> str:
    return f"T{int(number)}" if int(path) == 1 else f"T{int(number)}@{int(path)}"


def normalise(operator) -> dict | None:
    """The canonical option value: entries of 1.0 dropped, empty tables dropped, None when nothing is overridden
    (a run with all overrides at 100 % is the plain run, and has its cache key)."""
    if not operator:
        return None
    if not isinstance(operator, Mapping):
        raise ValueError("options.operator is a mapping {feed_override, spindle_override, rapid_override}")
    out: dict = {}
    for name, value in operator.items():
        if name not in KEYS:
            raise ValueError(f"options.operator.{name}: one of {', '.join(KEYS)}")
        if name == "rapid_override":
            factor = _factor(value, name)
            if factor != 1.0:
                out[name] = factor
            continue
        table = value if isinstance(value, Mapping) else {"*": value}
        kept = {}
        for key, factor in table.items():
            key = str(key)
            if key != "*" and not _valid_tool_key(key):
                raise ValueError(f"options.operator.{name}: {key!r} is '*', 'T<n>' or 'T<n>@<path>'")
            kept[key] = _factor(factor, f"{name}.{key}")
        default = kept.get("*", 1.0)
        kept = {k: v for k, v in kept.items() if v != default or k == "*"}
        if default == 1.0:
            kept.pop("*", None)
        if kept:
            out[name] = dict(sorted(kept.items()))
    return dict(sorted(out.items())) or None


def _valid_tool_key(key: str) -> bool:
    if not key.startswith("T"):
        return False
    number, _, path = key[1:].partition("@")
    return number.isdigit() and (path == "" or path.isdigit())


def _factor(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"options.operator.{where}: a fraction >= 0 (1 = 100 %), not {value!r}")
    return float(value)


@dataclasses.dataclass(frozen=True)
class Overrides:
    feed: dict
    spindle: dict
    rapid: float

    @classmethod
    def from_options(cls, options: Mapping | None) -> "Overrides | None":
        operator = normalise((options or {}).get("operator"))
        if operator is None:
            return None
        return cls(feed=dict(operator.get("feed_override") or {}), spindle=dict(operator.get("spindle_override") or {}),
                   rapid=float(operator.get("rapid_override", 1.0)))

    @staticmethod
    def lookup(table: Mapping, number, path: int) -> float:
        if number:
            key = tool_key(int(number), path)
            if key in table:
                return float(table[key])
        return float(table.get("*", 1.0))


def _tools(project) -> dict:
    """(path, number) -> tool type of the project's cutting tools."""
    out = {}
    tools = getattr(project, "tools", None)
    for entry in (getattr(tools, "list", None) or []):
        number = getattr(entry, "number", None)
        if number is None:
            continue
        tool = getattr(entry, "tool", None)
        out[(int(getattr(entry, "path", None) or 1), int(number))] = str(getattr(tool, "type", "") or "")
    return out


def _range(controller, name: str):
    value = getattr(controller, name, None) if controller is not None else None
    if value is None:
        return None
    try:
        values = [float(v) for v in value]
    except TypeError:
        return None
    return values


def problems(operator, project) -> list[str]:
    """Why an ``options.operator`` value cannot be applied to ``project`` (a ResolvedProject); empty when it can."""
    try:
        canonical = normalise(operator)
    except ValueError as error:
        return [str(error)]
    if canonical is None:
        return []
    out = []
    tools = _tools(project)
    controller = getattr(getattr(project, "machine", None), "controller", None)
    limits = {"feed_override": _range(controller, "feed_override_range_frac"),
              "spindle_override": _range(controller, "spindle_override_range_frac")}
    for name in ("feed_override", "spindle_override"):
        for key, factor in (canonical.get(name) or {}).items():
            if key != "*":
                number, _, path = key[1:].partition("@")
                ident = (int(path or 1), int(number))
                if ident not in tools:
                    out.append(f"{name}.{key}: the project has no cutting tool {number} on path {ident[0]}")
                elif name == "spindle_override" and tools[ident] in LOCKED_TOOL_TYPES and factor != 1.0:
                    out.append(f"{name}.{key}: tool {number} is a {tools[ident]}: the control holds tapping and "
                               f"threading at 100 % spindle speed")
            window = limits[name]
            if window is not None and len(window) == 2 and not window[0] - 1e-12 <= factor <= window[1] + 1e-12:
                out.append(f"{name}.{key}: {factor:g} is outside the machine's override range "
                           f"{window[0] * 100:g}-{window[1] * 100:g} %")
    if "rapid_override" in canonical:
        steps = _range(controller, "rapid_override_steps_frac")
        allowed = sorted(set((steps or []) + [1.0]))
        if steps is not None and not any(abs(canonical["rapid_override"] - s) <= 1e-9 for s in allowed):
            out.append(f"rapid_override: {canonical['rapid_override']:g} is not a rapid key of the machine "
                       f"({', '.join(f'{s * 100:g} %' for s in allowed)})")
    return out


class OperatorHook:
    """Applies :class:`Overrides` to the commands of one run (module docstring). ``install(motion)`` wraps a motion
    domain's ``accept``; ``calls`` counts the batches it saw (a run with overrides and no call means the hook was
    not reached: the runner refuses such a result)."""

    def __init__(self, overrides: Overrides, project) -> None:
        self.overrides = overrides
        self.tools = _tools(project)
        self.spindle_max = {}
        for spindle in (getattr(getattr(project, "machine", None), "spindles", None) or []):
            value = getattr(spindle, "max_rad_s", None)
            if value is not None:
                self.spindle_max[str(getattr(spindle, "id", ""))] = float(value)
        self.calls = 0
        self.scaled = {"feed": 0, "spindle": 0, "locked_spindle": 0}

    def install(self, motion) -> None:
        original = getattr(motion, "accept", None)
        if original is None or not callable(original):
            raise TypeError("the motion domain has no accept(ctx, state, path, commands, panel): operator overrides "
                            "cannot be applied")
        if getattr(original, "_operator_hook", None) is self:
            return

        def accept(ctx, state, path, commands, panel=None, **kwargs):
            return self.accept(original, ctx, state, path, commands, panel, **kwargs)

        accept._operator_hook = self
        motion.accept = accept

    def accept(self, original, ctx, state, path, commands, panel=None, **kwargs):
        self.calls += 1
        if panel is not None and hasattr(panel, "rapid_override"):
            panel.rapid_override = self.overrides.rapid
        changed = [self.command(c, int(path) + 1) for c in (commands or [])]
        return original(ctx, state, path, changed, panel=panel, **kwargs)

    def command(self, cmd, path: int):
        from ..controller.ir import FeedMode, Kind
        modal = getattr(cmd, "modal", None) or {}
        number = modal.get("tool")
        feed = getattr(cmd, "feed", None)
        if feed is not None and feed.override_allowed and feed.mode != FeedMode.RAPID and cmd.kind != Kind.RAPID:
            factor = self.overrides.lookup(self.overrides.feed, number, path)
            if factor != 1.0:
                cmd = dataclasses.replace(cmd, feed=dataclasses.replace(feed, value=float(feed.value) * factor))
                self.scaled["feed"] += 1
        spindle = getattr(cmd, "spindle", None)
        if cmd.kind == Kind.SPINDLE and isinstance(spindle, dict):
            factor = self.overrides.lookup(self.overrides.spindle, number, path)
            if factor != 1.0 and number and self.tools.get((path, int(number))) in LOCKED_TOOL_TYPES:
                self.scaled["locked_spindle"] += 1
                factor = 1.0
            if factor != 1.0 and (spindle.get("omega_rad_s") or spindle.get("css_m_s")):
                cmd = dataclasses.replace(cmd, spindle=self._spindle(spindle, factor))
                self.scaled["spindle"] += 1
        return cmd

    def _spindle(self, spindle: dict, factor: float) -> dict:
        out = dict(spindle)
        limit = self.spindle_max.get(str(out.get("id", "")), math.inf)
        clamp = out.get("max_rad_s")
        if isinstance(clamp, (int, float)) and clamp > 0:
            limit = min(limit, float(clamp))
        omega = out.get("omega_rad_s")
        if isinstance(omega, (int, float)) and omega:
            scaled = float(omega) * factor
            out["omega_rad_s"] = math.copysign(min(abs(scaled), limit), scaled)
        css = out.get("css_m_s")
        if isinstance(css, (int, float)) and css:
            out["css_m_s"] = float(css) * factor
        return out

    def summary(self) -> dict:
        return {"feed_override": dict(self.overrides.feed), "spindle_override": dict(self.overrides.spindle),
                "rapid_override": self.overrides.rapid, "batches": self.calls, "scaled": dict(self.scaled)}


__all__ = ["Overrides", "OperatorHook", "normalise", "problems", "tool_key", "LOCKED_TOOL_TYPES"]
