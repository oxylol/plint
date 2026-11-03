"""Conservation ledger: mass, energy and time balances, checked every macro step (docs/architecture.md 12).

Every term is computed independently by its owner and nothing is closed by difference. Each macro
step :meth:`Ledger.check` (after the slow domains, before the recorder) books the step's bus records
into ``SimState.ledger`` and checks:

* **mass** (kg, cumulative, tolerance ``physics.tol_mass``)::

      rho20 (V_removed + V_loose)  =  m_chips + rho20 V_displaced + m_loose

  ``V_removed`` = sum of ``removal.contacts[*].removed_volume_m3`` (the field volume the cuts took out),
  ``V_displaced`` = sum of ``displaced_volume_m3`` (formed / burr / side flow put back), ``V_loose`` and
  ``m_loose`` = ``removal.loose_bodies[*].volume_m3`` / ``mass_kg`` (bodies detached by the
  connectivity check); ``m_chips`` = the six inventories chips writes into ``SimState.ledger``
  (in flight, settled, conveyed, adhered, in flutes, dust), valid at ``chips.booked_until_tick``: the
  removal side is taken at that same step end. ``rho20`` is the workpiece density at 20 degC
  (architecture 4.1: masses are rho(20 degC) times material-configuration volumes).
* **energy** (J, per interval between two checks, tolerance ``physics.tol_energy``)::

      shaft work + feed work  =  shear + rake + flank heat + ploughing + modal energy change
                                 + structural damping + process damping + chip kinetic energy
                                 + friction (packing, recut, slip) + stored cold work + unbalance work

  cutting, dynamics and the domain-written ``friction_J``, ``stored_cold_work_J`` and
  ``unbalance_work_J`` cover the step's hot-loop window; the chip kinetic energy covers the windows
  chips has booked (``booked_until_tick``), so the check runs up to the earlier of the two. The
  residual is relative to the sum of the magnitudes of all terms of the interval.
* **time** (integer base ticks, exact): per controller path the block durations of the timeline (the
  open block up to now) sum to the path's span, and no block books more category ticks than it lasts.
* **thermal solvers**: the residuals thermal publishes (``thermal.ledger_residual_rel``) and the
  :meth:`Ledger.solver` hook (``dU = sources - losses - removed enthalpy``) against ``tol_energy``.

A violation adds a fatal finding (mass_balance / energy_balance / time_balance) and raises
:class:`~cnctwin.core.errors.ConservationError`: the run fails. The heat partition (generation by
zone of a step against what thermal distributes one step later) is reported, not enforced.
Residual channels ``ledger.*`` (macro tier) and :meth:`Ledger.summary` (run.json
``summary.conservation``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .channels import Channel, Tier
from .errors import ConservationError, TwinError
from .findings import Category, Severity
from .rundir import TICK_S, tick_time_s

MASS_CHIP_FIELDS = ("chips_in_flight_kg", "chips_settled_kg", "chips_conveyed_kg", "chips_adhered_kg",
                    "chips_in_flutes_kg", "dust_kg")
DOMAIN_ENERGY_FIELDS = ("friction_J", "stored_cold_work_J", "unbalance_work_J")
HEAT_DESTINATION_FIELDS = ("heat_to_chip_J", "heat_to_tool_J", "heat_to_work_J")
# step terms booked from the bus by the ledger (written into SimState.ledger by nobody else)
CUTTING_TERMS = ("shaft_work_J", "feed_work_J", "heat_shear_J", "heat_rake_J", "heat_flank_J", "ploughing_J")
ENERGY_IN = ("shaft_work_J", "feed_work_J")
ENERGY_OUT = ("heat_shear_J", "heat_rake_J", "heat_flank_J", "ploughing_J", "modal_energy_change_J", "damping_J",
              "process_damping_J", "chip_kinetic_J") + DOMAIN_ENERGY_FIELDS
GENERATION = ("heat_shear_J", "heat_rake_J", "heat_flank_J", "ploughing_J")


class LedgerError(TwinError):
    """A ledger input breaks its contract (a record booked twice, a window that is not a step end)."""


@dataclass
class _Step:
    tick: int
    delta: dict                 # energy terms of the step (J)
    cum: dict                   # cumulative removal terms at the end of the step


class SolverCheck:
    """Per-step energy identity of one thermal solver: ``dU = sources - losses - removed``."""

    def __init__(self, ledger: "Ledger", name: str) -> None:
        self.ledger = ledger
        self.name = name

    def book(self, u_before_J: float, u_after_J: float, sources_J: float, losses_J: float, removed_J: float,
             time_s: float) -> float:
        du = u_after_J - u_before_J
        expected = sources_J - losses_J - removed_J
        scale = abs(u_before_J) + abs(u_after_J) + abs(sources_J) + abs(losses_J) + abs(removed_J)
        rel = abs(du - expected) / scale if scale > 0.0 else 0.0
        self.ledger._solver_residual(self.name, rel)
        if rel > self.ledger.tol_energy:
            raise ConservationError(f"thermal energy ({self.name})", expected, du, self.ledger.tol_energy, time_s)
        return rel


class Ledger:
    """The run's conservation checks (module docstring); a checkpoint participant without outputs.

    ``density_kg_m3``: workpiece density at 20 degC (one material per project; a mapping workpiece ->
    density for several); ``domains``: the active domain names (a balance whose owners are absent is
    reported as not checked, never as passed)."""

    def __init__(self, *, tol_mass: float, tol_energy: float, density_kg_m3, domains=None, tick_s: float = TICK_S,
                 max_lag_ticks: int | None = None):
        self.tol_mass = float(tol_mass)
        self.tol_energy = float(tol_energy)
        self.density = density_kg_m3
        self.domains = None if domains is None else frozenset(domains)
        self.tick_s = float(tick_s)
        self.max_lag_ticks = None if max_lag_ticks is None else int(max_lag_ticks)
        self.ke_pending = 0.0                        # chip kinetic energy booked by chips, not yet checked
        self.steps: list[_Step] = []                 # since the last energy check point (pruned)
        self.removal_cum = {"removed_m3": 0.0, "displaced_m3": 0.0, "loose_m3": 0.0, "loose_kg": 0.0,
                            "removed_kg": 0.0, "loose_rho_kg": 0.0, "displaced_kg": 0.0}
        self.state_prev = {name: 0.0 for name in DOMAIN_ENERGY_FIELDS}
        self.chips_booked = -1
        self.energy_checked_until = 0
        self.worst = {"mass": 0.0, "energy": 0.0, "time": 0, "thermal": 0.0, "partition": 0.0}
        self.final = {"mass": 0.0, "energy": 0.0, "time": 0, "partition": 0.0}
        self.checked = {"mass": 0, "energy": 0, "time": 0, "partition": 0}
        self.unclassified_ticks = 0
        self.solvers: dict[str, float] = {}
        self.values: dict[str, float] = {}
        self.generation_prev = 0.0
        self._booked: list = []                      # (record objects, fingerprints) of the last two steps
        self.energy_in_cum = 0.0
        self.energy_out_cum = 0.0

    @classmethod
    def from_context(cls, ctx, domains=None) -> "Ledger":
        """Tolerances from ``ctx.project.physics``, density from ``ctx.project.material.physical``."""
        physics = ctx.project.physics
        material = ctx.project.material
        physical = getattr(material, "physical", None) if material is not None else None
        density = getattr(physical, "density_kg_m3", None) if physical is not None else None
        if density is None:
            raise TwinError("the project's material has no physical.density_kg_m3: the mass balance needs it")
        chips = getattr(physics, "chips", None)
        every = getattr(chips, "every_n_macro", None) if chips is not None else None
        lag = getattr(physics, "lag_macro", None)
        steps = int(every or 1) + int(lag or 1) + 1
        return cls(tol_mass=float(physics.tol_mass), tol_energy=float(physics.tol_energy),
                   density_kg_m3=float(density), domains=domains, tick_s=ctx.tick_s,
                   max_lag_ticks=steps * ctx.ipo_ticks * ctx.macro_ipo)

    def _rho(self, workpiece: int) -> float:
        if isinstance(self.density, Mapping):
            return float(self.density[int(workpiece)])
        return float(self.density)

    def active(self, domain: str) -> bool:
        return self.domains is None or domain in self.domains

    def solver(self, name: str) -> SolverCheck:
        return SolverCheck(self, name)

    def _solver_residual(self, name: str, rel: float) -> None:
        self.solvers[name] = max(self.solvers.get(name, 0.0), float(rel))
        self.worst["thermal"] = max(self.worst["thermal"], float(rel))

    # ------------------------------------------------------------------ booking
    def _guard(self, bus) -> None:
        """A record still in bus_next from two steps ago would be booked twice: refuse it."""
        current = []
        for name in ("cutting", "removal", "dynamics"):
            record = getattr(bus, name)
            current.append((name, record, _fingerprint(name, record)))
        if len(self._booked) == 2:
            for (name, record, print_), (_, old, old_print) in zip(current, self._booked[0]):
                if record is old and print_ == old_print and any(v != 0.0 for v in print_):
                    raise LedgerError(f"bus_next.{name} still holds the record of two macro steps ago (the loop "
                                      f"must give every step fresh records): it would be booked twice")
        self._booked = (self._booked + [current])[-2:]

    def book(self, state, tick_end: int) -> dict:
        """Book the step's bus records into ``state.ledger``; returns the step's energy terms."""
        bus = state.bus_next
        ledger = state.ledger
        self._guard(bus)
        delta = {name: 0.0 for name in ENERGY_IN + ENERGY_OUT}
        for contact in bus.cutting.contacts:
            for name in CUTTING_TERMS:
                delta[name] += float(getattr(contact, name))
        dynamics = bus.dynamics
        delta["modal_energy_change_J"] = float(dynamics.modal_energy_change_J)
        delta["damping_J"] = float(dynamics.damping_dissipation_J)
        delta["process_damping_J"] = float(dynamics.process_damping_J)
        booked = int(getattr(bus.chips, "booked_until_tick", -1)) if self.active("chips") else -1
        if booked > self.chips_booked:
            delta["chip_kinetic_J"] = float(getattr(bus.chips, "kinetic_energy_J", 0.0))
            self.ke_pending += delta["chip_kinetic_J"]
            self.chips_booked = booked
        for name in DOMAIN_ENERGY_FIELDS:
            value = float(getattr(ledger, name))
            delta[name] = value - self.state_prev[name]
            self.state_prev[name] = value
        removed = displaced = loose_v = loose_m = removed_kg = displaced_kg = loose_rho = 0.0
        for contact in bus.removal.contacts:
            rho = self._rho(contact.workpiece)
            removed += float(contact.removed_volume_m3)
            displaced += float(contact.displaced_volume_m3)
            removed_kg += rho * float(contact.removed_volume_m3)
            displaced_kg += rho * float(contact.displaced_volume_m3)
        for body in bus.removal.loose_bodies:
            rho = self._rho(int(body.get("workpiece", 0)))
            loose_v += float(body.get("volume_m3", 0.0))
            loose_m += float(body.get("mass_kg", 0.0))
            loose_rho += rho * float(body.get("volume_m3", 0.0))
        cum = self.removal_cum
        for key, value in (("removed_m3", removed), ("displaced_m3", displaced), ("loose_m3", loose_v),
                           ("loose_kg", loose_m), ("removed_kg", removed_kg), ("displaced_kg", displaced_kg),
                           ("loose_rho_kg", loose_rho)):
            cum[key] += value
        for name in CUTTING_TERMS + ("modal_energy_change_J", "damping_J", "process_damping_J", "chip_kinetic_J"):
            setattr(ledger, name, float(getattr(ledger, name)) + delta[name])
        ledger.removed_volume_m3 = cum["removed_m3"]
        ledger.removed_mass_kg = cum["removed_kg"]
        ledger.displaced_volume_m3 = cum["displaced_m3"]
        ledger.freed_body_mass_kg = cum["loose_kg"]
        self.steps.append(_Step(int(tick_end), delta, dict(cum)))
        return delta

    # ------------------------------------------------------------------ the checks
    def check(self, ctx, state, tick_end: int) -> dict:
        """Book the step and check every balance; raises ConservationError on a violation."""
        tick_end = int(tick_end)
        self.book(state, tick_end)
        values: dict[str, float] = {}
        self._check_mass(ctx, state, tick_end, values)
        self._check_energy(ctx, state, tick_end, values)
        self._check_time(ctx, state, tick_end, values)
        self._check_thermal(ctx, state, tick_end, values)
        self._partition(state, values)
        self._prune()
        self.values = values
        return values

    def _check_mass(self, ctx, state, tick_end: int, values: dict) -> None:
        if not (self.active("removal") and self.active("chips")):
            values["ledger.mass_residual_rel"] = math.nan
            return
        cut = state.bus_next.chips
        booked = int(getattr(cut, "booked_until_tick", -1))
        if booked < 0:
            side = {k: 0.0 for k in self.removal_cum}
        else:
            side = self._removal_at(booked)
        chips = sum(float(getattr(state.ledger, name)) for name in MASS_CHIP_FIELDS)
        mass_in = side["removed_kg"] + side["loose_rho_kg"]
        mass_out = chips + side["displaced_kg"] + side["loose_kg"]
        scale = max(abs(mass_in), abs(mass_out))
        rel = abs(mass_in - mass_out) / scale if scale > 0.0 else 0.0
        values.update({"ledger.mass_in_kg": mass_in, "ledger.mass_out_kg": mass_out, "ledger.mass_residual_rel": rel})
        self.checked["mass"] += 1
        self.worst["mass"] = max(self.worst["mass"], rel)
        self.final["mass"] = rel
        if rel > self.tol_mass:
            self._fail(ctx, state, tick_end, Category.MASS_BALANCE, "mass", mass_in, mass_out, self.tol_mass,
                       f"Mass balance violated: removed {mass_in:.9g} kg, accounted {mass_out:.9g} kg "
                       f"(relative {rel:.3g}, tolerance {self.tol_mass:g})",
                       {"mass_in_kg": mass_in, "mass_out_kg": mass_out, "residual_rel": rel,
                        "chips_booked_until_tick": booked})

    def _removal_at(self, tick: int) -> dict:
        if tick >= self.steps[-1].tick:
            return dict(self.removal_cum)
        for step in reversed(self.steps):
            if step.tick == tick:
                return step.cum
        raise LedgerError(f"chips booked emissions until tick {tick}, which is not a macro step end the ledger "
                          f"holds (steps {self.steps[0].tick}..{self.steps[-1].tick})")

    def _check_energy(self, ctx, state, tick_end: int, values: dict) -> None:
        if not self.active("cutting"):
            values["ledger.energy_residual_rel"] = math.nan
            return
        until = min(tick_end, self.chips_booked) if self.active("chips") else tick_end
        values["ledger.alignment_lag_s"] = (tick_end - max(until, 0)) * self.tick_s
        if self.active("chips") and self.max_lag_ticks is not None and tick_end - max(until, 0) > self.max_lag_ticks:
            raise LedgerError(f"chips has booked its windows only up to tick {until} at tick {tick_end} (more than "
                              f"{self.max_lag_ticks} ticks behind): the balances cannot be aligned")
        interval = [s for s in self.steps if self.energy_checked_until < s.tick <= until]
        if not interval:
            values.update({"ledger.energy_in_J": self.energy_in_cum, "ledger.energy_out_J": self.energy_out_cum,
                           "ledger.energy_residual_rel": 0.0, "ledger.energy_residual_J": 0.0})
            return
        totals = {name: 0.0 for name in ENERGY_IN + ENERGY_OUT}
        for step in interval:
            for name, value in step.delta.items():
                if name != "chip_kinetic_J":
                    totals[name] += value
        # the chip kinetic energy chips has booked covers its windows up to chips_booked = the interval's end
        totals["chip_kinetic_J"] = self.ke_pending
        self.ke_pending = 0.0
        energy_in = sum(totals[n] for n in ENERGY_IN)
        energy_out = sum(totals[n] for n in ENERGY_OUT)
        scale = sum(abs(v) for v in totals.values())
        rel = abs(energy_in - energy_out) / scale if scale > 0.0 else 0.0
        self.energy_in_cum += energy_in
        self.energy_out_cum += energy_out
        self.energy_checked_until = interval[-1].tick
        values.update({"ledger.energy_in_J": self.energy_in_cum, "ledger.energy_out_J": self.energy_out_cum,
                       "ledger.energy_residual_rel": rel, "ledger.energy_residual_J": energy_in - energy_out})
        self.checked["energy"] += 1
        self.worst["energy"] = max(self.worst["energy"], rel)
        self.final["energy"] = rel
        if rel > self.tol_energy:
            self._fail(ctx, state, tick_end, Category.ENERGY_BALANCE, "energy", energy_in, energy_out, self.tol_energy,
                       f"Energy balance violated: {energy_in:.9g} J in, {energy_out:.9g} J out "
                       f"(relative {rel:.3g}, tolerance {self.tol_energy:g})",
                       dict(totals) | {"residual_rel": rel})

    def _check_time(self, ctx, state, tick_end: int, values: dict) -> None:
        timeline = getattr(ctx, "timeline", None) if ctx is not None else None
        spans = _path_spans(timeline, tick_end) if timeline is not None else {}
        if not spans:
            values["ledger.time_residual_ticks"] = math.nan
            return
        worst = 0
        problems = []
        block_total = span_total = 0
        unclassified = 0
        for path, entry in sorted(spans.items()):
            residual = entry["block_ticks"] - entry["span_ticks"]
            block_total += entry["block_ticks"]
            span_total += entry["span_ticks"]
            worst = max(worst, abs(residual))
            if residual:
                problems.append(f"path {path}: blocks {entry['block_ticks']} ticks, span {entry['span_ticks']}")
            if entry["open"] > 1:
                problems.append(f"path {path}: {entry['open']} blocks open at once")
            over = entry["category_ticks"] - entry["categorised_ticks"]
            if over > 0:
                problems.append(f"path {path}: {over} ticks booked to categories beyond their blocks' durations")
            unclassified += max(0, -over)
        state.ledger.block_time_ticks = int(block_total)
        state.ledger.cycle_time_ticks = int(span_total)
        self.unclassified_ticks = int(unclassified)
        values["ledger.time_residual_ticks"] = float(worst)
        self.checked["time"] += 1
        self.worst["time"] = max(self.worst["time"], worst)
        self.final["time"] = worst
        if problems:
            self._fail(ctx, state, tick_end, Category.TIME_BALANCE, f"time ({'; '.join(problems)})", float(span_total),
                       float(block_total), 0.0,
                       "Time balance violated: " + "; ".join(problems),
                       {"block_ticks": float(block_total), "span_ticks": float(span_total)})

    def _check_thermal(self, ctx, state, tick_end: int, values: dict) -> None:
        residuals = getattr(state.bus_next.thermal, "ledger_residual_rel", {}) or {}
        worst = 0.0
        for name, rel in sorted(residuals.items()):
            self._solver_residual(str(name), float(rel))
            worst = max(worst, float(rel))
        values["ledger.thermal_residual_rel"] = worst
        if worst > self.tol_energy:
            name = max(residuals, key=lambda k: residuals[k])
            self._fail(ctx, state, tick_end, Category.ENERGY_BALANCE, f"thermal energy ({name})", 0.0, worst,
                       self.tol_energy, f"Thermal solver {name}: relative energy residual {worst:.3g} above "
                                        f"{self.tol_energy:g}", {"residual_rel": worst})

    def _partition(self, state, values: dict) -> None:
        """Heat generated by zone in a step against what thermal distributed in the next (reported)."""
        step = self.steps[-1]
        distributed = sum(float(getattr(state.ledger, n)) for n in HEAT_DESTINATION_FIELDS)
        generated = self.generation_prev
        if self.active("thermal") and (generated or distributed):
            scale = max(abs(generated), abs(distributed))
            rel = abs(generated - distributed) / scale if scale > 0.0 else 0.0
            values["ledger.partition_residual_rel"] = rel
            self.checked["partition"] += 1
            self.worst["partition"] = max(self.worst["partition"], rel)
            self.final["partition"] = rel
        else:
            values["ledger.partition_residual_rel"] = math.nan
        self.generation_prev += sum(step.delta[n] for n in GENERATION)

    def _prune(self) -> None:
        """Keep the steps a later check can still need: after the energy check point, and the step chips'
        inventories refer to (the mass check looks the removal side up there)."""
        keep_from = self.energy_checked_until + 1
        if self.active("chips") and self.chips_booked >= 0:
            keep_from = min(keep_from, self.chips_booked)
        last = self.steps[-1]
        self.steps = [s for s in self.steps if s.tick >= keep_from or s is last]

    def _fail(self, ctx, state, tick, category, quantity, expected, actual, tolerance, message, quantities) -> None:
        if ctx is not None and hasattr(ctx, "finding"):
            ctx.finding(state, "ledger", category, Severity.FATAL, message, tick,
                        quantities={k: float(v) for k, v in quantities.items() if _number(v)})
        raise ConservationError(quantity, expected, actual, tolerance, tick_time_s(tick, self.tick_s))

    # ------------------------------------------------------------------ channels and summary
    @staticmethod
    def channels() -> list[Channel]:
        doc = {"ledger.mass_in_kg": ("kg", "workpiece mass that left the field (removed and detached), at chips' booked tick"),
               "ledger.mass_out_kg": ("kg", "chips (six inventories), displaced and detached mass"),
               "ledger.mass_residual_rel": ("-", "relative mass residual"),
               "ledger.energy_in_J": ("J", "shaft and feed work checked so far"),
               "ledger.energy_out_J": ("J", "heat, ploughing, modal, damping, chip kinetic, friction, cold work, unbalance"),
               "ledger.energy_residual_rel": ("-", "relative energy residual of the interval checked this step"),
               "ledger.energy_residual_J": ("J", "energy in minus out of the interval checked this step"),
               "ledger.alignment_lag_s": ("s", "how far the energy check trails the step end (chips' booked window)"),
               "ledger.time_residual_ticks": ("tick", "largest |block time - span| over the paths"),
               "ledger.thermal_residual_rel": ("-", "largest relative residual of the thermal solvers this step"),
               "ledger.partition_residual_rel": ("-", "generated heat (previous step) against distributed heat, cumulative")}
        return [Channel(name, unit, Tier.MACRO, "ledger", text) for name, (unit, text) in doc.items()]

    def channel_values(self) -> dict:
        return dict(self.values)

    def summary(self) -> dict:
        """run.json ``summary.conservation``."""
        def entry(name, tolerance, unit_note):
            checked = self.checked[name]
            return {"max_residual_rel": self.worst[name] if checked else None,
                    "final_residual_rel": self.final[name] if checked else None, "tolerance_rel": tolerance,
                    "checked_steps": checked, "ok": checked > 0, "note": unit_note if checked else
                    f"not checked: its terms' owners were not active in this run"}
        out = {"mass": entry("mass", self.tol_mass, "rho20 (V_removed + V_loose) = chips + rho20 V_displaced + m_loose"),
               "energy": entry("energy", self.tol_energy, "shaft + feed work = heat by zone + ploughing + modal change "
                                                          "+ damping + process damping + chip kinetic + friction + cold "
                                                          "work + unbalance work"),
               "time": {"residual_ticks": self.worst["time"] if self.checked["time"] else None, "tolerance_rel": 0.0,
                        "checked_steps": self.checked["time"], "ok": self.checked["time"] > 0,
                        "unclassified_ticks": self.unclassified_ticks,
                        "note": "sum of block times equals each path's span in integer ticks" if self.checked["time"]
                        else "not checked: no timeline"},
               "thermal": {"solvers": dict(sorted(self.solvers.items())), "max_residual_rel": self.worst["thermal"],
                           "tolerance_rel": self.tol_energy},
               "partition": {"max_residual_rel": self.worst["partition"] if self.checked["partition"] else None,
                             "final_residual_rel": self.final["partition"] if self.checked["partition"] else None,
                             "enforced": False,
                             "note": "heat generated by zone (hot loop) against heat to chip + tool + work (thermal, "
                                     "one step later); reported, not enforced"}}
        out["mass"]["terms"] = {k: v for k, v in sorted(self.removal_cum.items())}
        return out

    # ------------------------------------------------------------------ checkpoint participant
    def checkpoint_cut(self) -> dict:
        return {}

    def checkpoint_state(self) -> dict:
        return {"steps": [{"tick": s.tick, "delta": dict(s.delta), "cum": dict(s.cum)} for s in self.steps],
                "removal_cum": dict(self.removal_cum), "state_prev": dict(self.state_prev),
                "chips_booked": self.chips_booked, "energy_checked_until": self.energy_checked_until,
                "worst": dict(self.worst), "final": dict(self.final), "checked": dict(self.checked),
                "unclassified_ticks": self.unclassified_ticks, "solvers": dict(self.solvers),
                "generation_prev": self.generation_prev, "energy_in_cum": self.energy_in_cum,
                "energy_out_cum": self.energy_out_cum, "ke_pending": self.ke_pending}

    def restore_state(self, state: Mapping) -> None:
        self.steps = [_Step(int(s["tick"]), dict(s["delta"]), dict(s["cum"])) for s in state["steps"]]
        self.removal_cum = dict(state["removal_cum"])
        self.state_prev = dict(state["state_prev"])
        self.chips_booked = int(state["chips_booked"])
        self.energy_checked_until = int(state["energy_checked_until"])
        self.worst = dict(state["worst"])
        self.final = dict(state["final"])
        self.checked = dict(state["checked"])
        self.unclassified_ticks = int(state["unclassified_ticks"])
        self.solvers = dict(state["solvers"])
        self.generation_prev = float(state["generation_prev"])
        self.energy_in_cum = float(state["energy_in_cum"])
        self.energy_out_cum = float(state["energy_out_cum"])
        self.ke_pending = float(state["ke_pending"])
        self._booked = []

    def truncate_outputs(self, cut: Mapping) -> None:
        return None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fingerprint(name: str, record) -> tuple:
    if name == "cutting":
        return tuple(float(getattr(c, t)) for c in record.contacts for t in CUTTING_TERMS)
    if name == "removal":
        return tuple(float(c.removed_volume_m3) for c in record.contacts) + \
            tuple(float(b.get("volume_m3", 0.0)) for b in record.loose_bodies)
    return (float(record.modal_energy_change_J), float(record.damping_dissipation_J), float(record.process_damping_J))


def _path_spans(timeline, now: int) -> dict:
    """Per path ``{block_ticks, span_ticks, open, category_ticks, categorised_ticks}`` of a timeline
    (cnctwin.core.rundir.Timeline, or anything with ``records`` of dicts or objects with path, tick0,
    tick1 and optional ticks). A path without an open block spans up to its last block's end."""
    records = getattr(timeline, "records", None)
    if records is None:
        return {}
    out: dict[int, dict] = {}
    for r in records:
        get = r.get if isinstance(r, Mapping) else (lambda k, d=None, _r=r: getattr(_r, k, d))
        path, tick0, tick1 = int(get("path", 0)), int(get("tick0")), int(get("tick1", -1))
        entry = out.setdefault(path, {"block_ticks": 0, "first": tick0, "last": tick0, "open": 0, "category_ticks": 0,
                                      "categorised_ticks": 0})
        entry["first"] = min(entry["first"], tick0)
        if tick1 >= 0:
            entry["block_ticks"] += tick1 - tick0
            entry["last"] = max(entry["last"], tick1)
            ticks = get("ticks")
            if isinstance(ticks, Mapping) and ticks:
                entry["category_ticks"] += sum(int(v) for v in ticks.values())
                entry["categorised_ticks"] += tick1 - tick0
        else:
            entry["block_ticks"] += max(0, int(now) - tick0)
            entry["open"] += 1
    for entry in out.values():
        end = int(now) if entry["open"] else entry["last"]
        entry["span_ticks"] = end - entry["first"]
    return out


__all__ = ["Ledger", "SolverCheck", "LedgerError", "ENERGY_IN", "ENERGY_OUT", "MASS_CHIP_FIELDS"]
