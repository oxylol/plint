"""Wiring of the run loop (docs/models/loop.md "Wiring"): the domain objects of a run, built in dependency order, and
the protocol adapters through which they read each other without importing each other's internals.

Every adapter reads the run's *current* state through the run object (``run.state``), never a state object captured
at construction: a checkpoint restore replaces ``run.state`` and the adapters follow it.

=====================  ==================================================  ===========================================
adapter                serves                                              reads
=====================  ==================================================  ===========================================
:class:`LiveField`     thermal / structure WorkGeometry, chips             removal's dexel field (FieldGeometry); its
                       ChipGeometry, surface WorkpieceQuery                change tracking is checkpointed
:class:`LiveEdges`     thermal ToolEdges                                   tooling's effective edges
:class:`LiveCatalog`   chips ToolCatalog                                   tooling (ToolingCatalog)
:class:`LiveBricks`    surface BrickGrid                                   removal's level-0 lattice of the moment
:class:`SurfaceBurrs`  holding BurrSource                                  surface's surviving burrs (SurfaceView)
:class:`LoopResults`   inspect ResultSource                                removal's mesh of the field + structure's
                                                                           grids (as cut, released) + thermal's
                                                                           unload temperatures
=====================  ==================================================  ===========================================
"""

from __future__ import annotations

import numpy as np

from ..removal.views import FieldGeometry
from .ledger import ENERGY_IN, ENERGY_OUT, Ledger, LedgerError

# the order domains are built (init) in: each reads only what an earlier one set up (architecture 6, 19)
INIT_ORDER = ("tooling", "controller", "motion", "cutting", "removal", "dynamics", "structure", "holding", "thermal",
              "coolant", "chips", "wear", "surface")
# the order of the slow domains' macro steps (they read bus_prev and write only their own record: any order gives
# the same result; this one is fixed so traces and findings are numbered the same on every run)
SLOW_ORDER = ("motion", "controller", "tooling", "cutting", "removal", "dynamics", "chips", "thermal", "coolant",
              "wear", "holding", "structure", "surface")
# end of an operation (inspect notes: structure before surface before inspect before removal's transfer)
END_ORDER = ("motion", "controller", "cutting", "dynamics", "chips", "coolant", "wear", "holding", "thermal",
             "structure", "surface", "inspect", "removal", "tooling")
DOMAINS = ("controller", "motion", "tooling", "removal", "cutting", "chips", "thermal", "coolant", "wear", "holding",
           "structure", "dynamics", "surface")


class LiveField(FieldGeometry):
    """removal's FieldGeometry over the run's current removal state. ``changed_box_S`` remembers, per caller and
    workpiece, the brick versions it last reported: that memory is part of the run (:meth:`seen_state`), so a
    resumed run reports the same boxes as an uninterrupted one."""

    def __init__(self, run, caller: str = "loop"):
        super().__init__(lambda: run.state.removal, caller)

    def seen_state(self) -> dict:
        return {f"{caller}|{w}": np.array(v) for (caller, w), v in sorted(self._seen.items())}

    def restore_seen(self, saved: dict) -> None:
        self._seen = {}
        for key, value in (saved or {}).items():
            caller, _, w = str(key).rpartition("|")
            self._seen[(caller, int(w))] = np.array(value)


class LiveEdges:
    """thermal's ToolEdges: the effective (worn, grown, pulled-out) edge model of a slot."""

    def __init__(self, run):
        self.run = run

    def edge_model(self, slot: int):
        return self.run.state.tooling.edge_model(int(slot))


class LiveCatalog:
    """chips' ToolCatalog over the current tooling state: tooling's ToolInfoView (gullet area = all flutes) as chips'
    own ToolInfo record (the same fields; chips also reads its ``hole_making`` property)."""

    def __init__(self, run):
        self.run = run

    def tool(self, slot: int):
        from dataclasses import fields

        from ..chips.protocols import ToolInfo
        from ..tooling.catalog import ToolingCatalog
        view = ToolingCatalog(self.run.state.tooling).tool(int(slot))
        if view is None:
            return None
        return ToolInfo(**{f.name: getattr(view, f.name) for f in fields(ToolInfo)})

    def edge_model(self, slot: int):
        return self.run.state.tooling.edge_model(int(slot))


class LiveBricks:
    """surface's BrickGrid: removal's level-0 brick numbering of the field as it is now (a transfer between
    operations moves the lattice; a fixed LatticeBricks would key the later operations' patches wrongly)."""

    def __init__(self, run):
        self.run = run

    def _grid(self, workpiece: int):
        from ..surface.protocols import LatticeBricks
        rs = self.run.state.removal
        f = rs.workpieces[min(max(int(workpiece), 0), len(rs.workpieces) - 1)].field
        return LatticeBricks(f.lv_origin_m[0], float(f.lv_cell_m[0]), f.lv_dims[0])

    def brick_of(self, workpiece: int, points_S_m):
        return self._grid(workpiece).brick_of(0, points_S_m)

    def brick_key(self, workpiece: int, brick: int) -> str:
        return str(int(brick)) if int(workpiece) == 0 else f"w{int(workpiece)}.{int(brick)}"

    def brick_size_m(self, workpiece: int) -> float:
        return self._grid(workpiece).brick_size_m(0)

    def coords(self, workpiece: int, bricks):
        return self._grid(workpiece).coords(0, bricks)


class SurfaceBurrs:
    """holding's BurrSource: the largest surviving burr surface reports on each locating patch of a setup (its
    exit face matched to the patch's outward normal)."""

    wired = True

    def __init__(self, run):
        self.run = run

    def burr_heights_m(self, op_index, patches) -> dict:
        surface = self.run.domains.get("surface")
        if surface is None or getattr(self.run.state, "surface", None) is None:
            return {}
        view = surface.view(self.run.state)
        out = {}
        for pid, centre, normal, half in patches:
            radius = float(max(np.max(np.asarray(half, float)), 1e-3))
            height = float(view.burr_height_at(0, np.asarray(centre, float), np.asarray(normal, float), radius))
            if height > 0.0:
                out[pid] = height
        return out


class LoopResults:
    """inspect's ResultSource (docs/models/inspect.md "Result geometry"): the material configuration X from
    removal's mesh of the field, structure's displacement grids at the end of cutting (``thermal``, ``clamp``,
    ``residual``: captured by the loop before structure unclamps) and of the released part (``final`` =
    residual after unclamping at 20 degC, ``unload_thermal`` = its thermal part), thermal's unload temperatures.
    Called by inspect.end_operation after structure.end_operation and before removal's transfer."""

    def __init__(self, run):
        self.run = run
        self.as_cut: dict = {}          # workpiece -> {name: DisplacementGrid} captured before the release

    def workpieces(self, op_index: int) -> list[int]:
        return list(range(len(self.run.state.removal.workpieces)))

    def capture_as_cut(self, state) -> None:
        from ..inspect.result import DisplacementGrid
        self.as_cut = {}
        st = getattr(state, "structure", None)
        for w, ws in sorted((getattr(st, "works", {}) or {}).items()):
            d = ws.solution.displacement if ws.solution is not None else None
            if d is None or float(d.cell_m) <= 0.0 or np.size(d.total_m) == 0:
                continue
            self.as_cut[int(w)] = {name: DisplacementGrid.from_bus(d, f"{name}_m")
                                   for name in ("thermal", "clamp", "residual")}

    def result_part(self, ctx, state, op_index: int, workpiece: int):
        from ..inspect.result import DisplacementGrid, ResultPart
        from ..removal import mesh as RM
        rs = state.removal
        if not 0 <= int(workpiece) < len(rs.workpieces):
            return None
        wp = rs.workpieces[int(workpiece)]
        mesh = RM.to_geom_mesh(RM.mesh_level(wp.field), name=f"part_w{workpiece}")
        if mesh.n_faces == 0:
            return None
        grids = dict(self.as_cut.get(int(workpiece), {}))
        notes = []
        st = getattr(state, "structure", None)
        ws = (getattr(st, "works", {}) or {}).get(int(workpiece)) if st is not None else None
        d = ws.solution.displacement if (ws is not None and ws.solution is not None) else None
        if d is not None and float(d.cell_m) > 0.0 and np.size(d.total_m):
            grids["final"] = DisplacementGrid.from_bus(d, "residual_m")
            grids["unload_thermal"] = DisplacementGrid.from_bus(d, "thermal_m")
        else:
            notes.append("structure published no displacement grid: the geometry is the material configuration")
        unload = _unload_record(state, op_index, int(workpiece))
        ops = list(getattr(ctx.project, "operations", []) or [])
        op_name = str(getattr(ops[op_index], "name", f"OP{op_index + 1}")) if op_index < len(ops) else f"op{op_index}"
        stock = getattr(ctx.project, "stock", None)
        return ResultPart(op=op_name, material=mesh, grids=grids, op_index=int(op_index), workpiece=int(workpiece),
                          unload_time_s=unload.get("time_s"), unload_mean_temp_K=unload.get("mean_K"),
                          unload_max_temp_K=unload.get("max_K"),
                          stock_kind=str(getattr(stock, "kind", "block") or "block"), notes=notes)


def _unload_record(state, op_index: int, workpiece: int) -> dict:
    th = getattr(state, "thermal", None)
    rows = [r for r in (getattr(th, "unload", None) or []) if int(r.get("op", -1)) == int(op_index)
            and int(r.get("workpiece", 0)) == int(workpiece)]
    return dict(rows[-1]) if rows else {}


def run_ledger(ctx, domains):
    """The run's conservation ledger (:class:`RunLedger`: the core Ledger with the two energy terms below)."""
    from ..db.models import params
    ledger = RunLedger.from_context(ctx, domains=domains)
    p = params("loop")
    ledger.kinetic_in_balance = bool(int(p["chip_kinetic_in_balance"]))
    ledger.energy_scale_floor_J = float(p["energy_scale_floor_J"])
    return ledger


class RunLedger(Ledger):
    """cnctwin.core.ledger.Ledger (architecture 12) with two energy terms of the integrated run:

    * ``modal_quadrature_J`` (out): cutting books the work of the cutting forces on the deflection rates by its own
      quadrature (impulse x predicted rate: dynamics' ``cutting_work_quadrature_J``, the same rule), the modes
      receive the exact first-order-hold work (``cutting_work_J``: in the modal energy change, damping and the
      unbalance term). The two differ by 0.1-3 % of the vibration work (docs/models/dynamics.md "Ledger note"),
      above the 1e-6 balance whenever the vibration work exceeds about 1e-4 of the shaft work (entries, chatter):
      their difference, which dynamics computes explicitly, is carried as its own term (dynamics' documented
      remedy).
    * the chips' kinetic energy at separation: an out term with ``loop.chip_kinetic_in_balance`` 1; with 0
      (db/models/loop.yaml) it is reported (``kinetic_reported_J``, channel ledger.chip_kinetic_reported_J) and
      kept out of the balance: cutting's force model has no momentum flux, so its shaft work does not contain it.

    The relative residual of an interval is taken against its terms' scale, but not below
    ``loop.energy_scale_floor_J``: out of the cut the only terms can be the modes' free vibration (picojoules of
    damping against the same modal energy change), whose integrator round-off alone is 1e-6 of them.
    """

    OUT = ENERGY_OUT + ("modal_quadrature_J",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kinetic_in_balance = False
        self.kinetic_reported_J = 0.0
        self.quadrature_J = 0.0
        self.energy_scale_floor_J = 0.0

    def book(self, state, tick_end: int) -> dict:
        delta = super().book(state, tick_end)
        dyn = state.bus_next.dynamics
        delta["modal_quadrature_J"] = float(getattr(dyn, "cutting_work_quadrature_J", 0.0) or 0.0) - \
            float(getattr(dyn, "cutting_work_J", 0.0) or 0.0)
        self.quadrature_J += delta["modal_quadrature_J"]
        return delta

    def _check_energy(self, ctx, state, tick_end: int, values: dict) -> None:
        """The core ledger's interval check (same alignment with chips' booked window, same failure) over the
        terms of this class."""
        import math
        from .findings import Category
        if not self.active("cutting"):
            values["ledger.energy_residual_rel"] = math.nan
            return
        until = min(tick_end, self.chips_booked) if self.active("chips") else tick_end
        values["ledger.alignment_lag_s"] = (tick_end - max(until, 0)) * self.tick_s
        if self.active("chips") and self.max_lag_ticks is not None and tick_end - max(until, 0) > self.max_lag_ticks:
            raise LedgerError(f"chips has booked its windows only up to tick {until} at tick {tick_end} (more than "
                              f"{self.max_lag_ticks} ticks behind): the balances cannot be aligned")
        values["ledger.chip_kinetic_reported_J"] = self.kinetic_reported_J + (0.0 if self.kinetic_in_balance
                                                                              else self.ke_pending)
        values["ledger.modal_quadrature_J"] = self.quadrature_J
        interval = [st for st in self.steps if self.energy_checked_until < st.tick <= until]
        if not interval:
            values.update({"ledger.energy_in_J": self.energy_in_cum, "ledger.energy_out_J": self.energy_out_cum,
                           "ledger.energy_residual_rel": 0.0, "ledger.energy_residual_J": 0.0})
            return
        totals = {name: 0.0 for name in ENERGY_IN + self.OUT}
        for step in interval:
            for name, value in step.delta.items():
                if name != "chip_kinetic_J" and name in totals:
                    totals[name] += value
        if self.kinetic_in_balance:
            totals["chip_kinetic_J"] = self.ke_pending
        else:
            self.kinetic_reported_J += self.ke_pending
        self.ke_pending = 0.0
        energy_in = sum(totals[n] for n in ENERGY_IN)
        energy_out = sum(totals[n] for n in self.OUT)
        scale = max(sum(abs(v) for v in totals.values()), self.energy_scale_floor_J)
        rel = abs(energy_in - energy_out) / scale if scale > 0.0 else 0.0
        self.energy_in_cum += energy_in
        self.energy_out_cum += energy_out
        self.energy_checked_until = interval[-1].tick
        values.update({"ledger.energy_in_J": self.energy_in_cum, "ledger.energy_out_J": self.energy_out_cum,
                       "ledger.energy_residual_rel": rel, "ledger.energy_residual_J": energy_in - energy_out,
                       "ledger.chip_kinetic_reported_J": self.kinetic_reported_J})
        self.checked["energy"] += 1
        self.worst["energy"] = max(self.worst["energy"], rel)
        self.final["energy"] = rel
        if rel > self.tol_energy:
            self._fail(ctx, state, tick_end, Category.ENERGY_BALANCE, "energy", energy_in, energy_out, self.tol_energy,
                       f"Energy balance violated: {energy_in:.9g} J in, {energy_out:.9g} J out "
                       f"(relative {rel:.3g}, tolerance {self.tol_energy:g})",
                       dict(totals) | {"residual_rel": rel})

    @staticmethod
    def channels() -> list:
        from .channels import Channel, Tier
        return Ledger.channels() + [
            Channel("ledger.chip_kinetic_reported_J", "J", Tier.MACRO, "ledger",
                    "chips' kinetic energy at separation booked so far, reported (not in the energy balance unless "
                    "loop.chip_kinetic_in_balance)"),
            Channel("ledger.modal_quadrature_J", "J", Tier.MACRO, "ledger",
                    "cutting's quadrature of the work on the deflection rates minus the exact modal work, so far")]

    def summary(self) -> dict:
        out = super().summary()
        if not self.checked["energy"] and self.active("cutting"):
            # no interval was checked (a run that stops in its first steps, before chips books a window): with no
            # energy booked at all there is nothing to balance; with energy booked the balance stays unproven
            booked = sum(abs(float(v)) for st in self.steps for name, v in st.delta.items()
                         if name in ENERGY_IN + self.OUT)
            if booked <= self.energy_scale_floor_J * self.tol_energy:
                out["energy"].update({"ok": True, "max_residual_rel": 0.0, "final_residual_rel": 0.0,
                                      "note": "no energy booked in the run (no cut): nothing to balance"})
        out["energy"]["chip_kinetic_reported_J"] = None if self.kinetic_in_balance else self.kinetic_reported_J
        out["energy"]["modal_quadrature_J"] = self.quadrature_J
        out["energy"]["terms_added"] = ["modal_quadrature_J (out: cutting's quadrature of the work on the deflection "
                                        "rates minus dynamics' exact modal work)"] + \
            ([] if self.kinetic_in_balance else ["chips' kinetic energy reported, not balanced (cutting's force model "
                                                 "has no momentum flux)"])
        return out

    def checkpoint_state(self) -> dict:
        out = super().checkpoint_state()
        out.update({"kinetic_reported_J": self.kinetic_reported_J, "quadrature_J": self.quadrature_J})
        return out

    def restore_state(self, state) -> None:
        super().restore_state(state)
        self.kinetic_reported_J = float(state.get("kinetic_reported_J", 0.0))
        self.quadrature_J = float(state.get("quadrature_J", 0.0))


KineticReportedLedger = RunLedger          # the name earlier wiring used


def build_domains(run) -> dict:
    """Every domain object of a run except motion (built per operation: its chain carries the operation's fixture
    and stock poses), with the adapters of the module table. Returned by name; ``inspect`` only when the project
    has a CAD to inspect against."""
    from ..chips.domain import ChipsDomain
    from ..controller.domain import ControllerDomain
    from ..coolant.domain import CoolantDomain
    from ..cutting.domain import CuttingDomain
    from ..dynamics.domain import DynamicsDomain
    from ..holding.domain import HoldingDomain
    from ..removal.domain import RemovalDomain
    from ..structure.domain import StructureDomain
    from ..surface.domain import SurfaceDomain
    from ..thermal.domain import ThermalDomain
    from ..tooling.domain import ToolingDomain
    from ..wear.domain import WearDomain
    run.field = LiveField(run, "loop")
    domains = {"tooling": ToolingDomain(), "controller": ControllerDomain(), "cutting": CuttingDomain(),
               "removal": RemovalDomain(), "dynamics": DynamicsDomain(),
               "structure": StructureDomain(geometry=run.field),
               "holding": HoldingDomain(burrs=SurfaceBurrs(run)),
               "thermal": ThermalDomain(geometry=run.field, edges=LiveEdges(run)),
               "coolant": CoolantDomain(), "chips": ChipsDomain(geometry=run.field, tools=LiveCatalog(run)),
               "wear": WearDomain(), "surface": SurfaceDomain(query=run.field, bricks=LiveBricks(run))}
    return domains


def late_wiring(run) -> None:
    """Adapters that need an initialised state: tool geometries for wear and surface, the next setups' locator
    faces for chips, the project tolerances for wear (called after tooling and holding are initialised, before
    the domains that use them)."""
    from ..surface.protocols import GeometryToolInfo as SurfaceTools
    from ..wear.protocols import GeometryToolInfo as WearTools
    from ..wear.protocols import ProjectTolerances
    d = run.domains
    geometries = list(run.state.tooling.tools)
    d["wear"].tools = WearTools(geometries)
    d["wear"].tolerances = ProjectTolerances(run.ctx.project)
    d["surface"].tools = SurfaceTools(geometries)
    d["chips"].locators = d["holding"].locator_faces(run.ctx)


def inspect_domain(run):
    """InspectDomain wired to the loop (None without a CAD: the run then writes inspection.json saying so)."""
    cad = getattr(run.ctx.project, "cad", None)
    if cad is None or not getattr(cad, "path", None):
        return None
    from ..inspect.domain import InspectDomain
    from ..inspect.protocols import SurfaceJson
    surface = run.domains["surface"]
    run.results = LoopResults(run)
    return InspectDomain(results=run.results,
                         surface=SurfaceJson(lambda: surface.document(run.ctx, run.state)))


__all__ = ["LiveField", "LiveEdges", "LiveCatalog", "LiveBricks", "SurfaceBurrs", "LoopResults", "build_domains",
           "run_ledger", "RunLedger", "KineticReportedLedger",
           "late_wiring", "inspect_domain", "INIT_ORDER", "SLOW_ORDER", "END_ORDER", "DOMAINS"]
