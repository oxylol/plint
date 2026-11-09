"""Single-operation driver: one operation of a real project through controller -> motion -> HotLoop -> absorb
(docs/models/hotloop.md "Driver").

The slow domains (thermal, chips, coolant, wear, holding, structure, surface) are not run: their bus records stay
at their defaults, so the hot loop sees a cold, unworn, rigidly held tool with no chips and dry cutting. What the
driver checks every macro step:

* the core ledger's time balance (cnctwin.core.ledger: block times sum to each path's span in integer ticks);
* the energy balance of every step from the step's own records (shaft + feed work = heat by zone + ploughing
  + the work of the cutting forces on the modes + packing / recut friction; the modes' own identity
  dE = W_cut + W_unbalance + W_base - D is dynamics'): booked per step, never by differencing cumulative
  values (the core ledger differences SimState.ledger.unbalance_work_J, which loses the step's 1e-13 J against
  the mJ of base-excitation work booked during rapids: docs/models/hotloop.md "Open problems");
* the exact removed volume: the sum of the removal records' volumes equals the field's own volume change
  (integer quanta) at every step end; the chip mass of the emission rows (rho x cutting's analytic chip
  volume per pass) is reported against rho x that volume (removal's volume consistency);

and it collects the per-tick wrenches, the removed volume, cycle time and the final field.

Frames: the project's ``fixture_pose`` is written in machine coordinates (db/_templates/project.yaml,
docs/models/sanity.md "Setup geometry convention": the axis values that put the spindle gauge line on the
point). Motion's chain wants the pose of W in the fixture mounting face F (architecture 4). For mills (machine
type ``*_mill``: their machine-coordinate axes are parallel to F at home, 5-axis machines at rotary zero) the
point under the gauge line at axis values q is ``F(0)^-1 N(0) + q``, so ``F_W = T(F(0)^-1 N(0)) fixture_pose``;
:func:`fixture_pose_F` does that and the driver hands motion the converted pose. Lathes write the chuck's pose on
the spindle face directly (examples/lathe-shaft-4140): it is used as it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bus import Bus
from .domain import RunContext
from .state import SimState


def fixture_pose_F(machine, fixture_pose_mc: np.ndarray, path: int = 0, letters=None) -> np.ndarray:
    """Pose of the workholding frame in the fixture mounting face F from a fixture pose written in machine
    coordinates (module doc)."""
    from ..motion.abi import FK_M_F, FK_M_N, eye_stack
    from ..motion.kernels import fk_poses
    from ..motion.machine import MachineModel
    model = MachineModel(machine, letters)
    P = len(model.paths)
    tips = [np.zeros(3) for _ in range(P)]
    chain = model.build_chain(tips, [np.eye(4)] * P, [np.eye(4)] * P, [-1] * P)
    q = np.ascontiguousarray([0.0] * len(model.axes), dtype=float)
    poses = eye_stack(7)
    fk_poses(chain, path, q, poses, np.zeros((6, 4, 4)))
    rel = np.linalg.inv(poses[FK_M_F]) @ poses[FK_M_N]
    T = np.eye(4)
    T[:3, 3] = rel[:3, 3]
    return T @ np.asarray(fixture_pose_mc, float)


class Feeder:
    """The controller side of the loop protocol: produce -> motion accept at the step start and every sync,
    motion's controller actions back to the controller. The tool numbers whose life the wear domain ended
    (``bus_prev.wear.expired``) reach each path's control as ``Feedback.tool_life_expired`` (its tool-life
    management then calls a sister tool): inside the next feedback motion sends, or alone when the path is not
    waiting for a feedback (a feedback would answer that wait)."""

    def __init__(self, ctx, state, controller, motion, *, parts: int = 1, operator_delay_ticks: int = 0):
        self.ctx = ctx
        self.state = state
        self.cd = controller
        self.md = motion
        self.parts = int(parts)
        self.delay = int(operator_delay_ticks)
        self.runs = [0] * motion.P
        self.ended = [False] * motion.P
        self.forwarded = [[] for _ in range(motion.P)]      # expired tool numbers each path's control knows

    def _expired(self, p: int) -> list[int]:
        wear = getattr(self.state.bus_prev, "wear", None)
        numbers = {int(n) for n in (getattr(wear, "expired", None) or [])}
        return sorted(numbers - set(self.forwarded[p]))

    def feed(self, now: int) -> None:
        ctx, state = self.ctx, self.state
        for p in range(self.md.P):
            numbers = self._expired(p)
            if numbers and self.cd.waiting(state, p) is None:
                from ..controller.ir import Feedback
                self.cd.feedback(ctx, state, p, Feedback(time_s=int(now) * float(ctx.tick_s),
                                                         tool_life_expired=numbers), int(now))
                self.forwarded[p].extend(numbers)
            cmds = self.cd.produce(ctx, state, p, int(now))
            self.md.accept(ctx, state, p, cmds, panel=self.cd.operator(state, p))

    def apply(self, actions) -> None:
        ctx, state = self.ctx, self.state
        for kind, p, arg, tick in actions:
            if kind == "retire":
                self.cd.retire(ctx, state, p, int(arg), int(tick))
            elif kind == "feedback":
                numbers = self._expired(p)
                if numbers and arg is not None:
                    arg.tool_life_expired = sorted(set(arg.tool_life_expired) | set(numbers))
                    self.forwarded[p].extend(numbers)
                self.cd.feedback(ctx, state, p, arg, int(tick))
            elif kind == "cycle_start":
                self.cd.cycle_start(ctx, state, p, int(tick) + self.delay)
            elif kind == "program_end":
                self.runs[p] += 1
                if self.runs[p] < self.parts:
                    self.cd.cycle_start(ctx, state, p, int(tick) + self.delay)
                else:
                    self.ended[p] = True

    def finished(self) -> bool:
        """Every path has run its parts or stopped with an alarm, and motion has executed what it holds. An alarm
        stops its path's motion as on the machine: commands queued before it do not execute (one of them may wait
        for the control, which answers nothing in alarm)."""
        ctx, state = self.ctx, self.state
        if not self.cd.done(ctx, state):
            return False
        ctrl = self.cd.controller(ctx, state)
        alarmed = [ctrl.state.paths[p].alarm is not None for p in range(self.md.P)]
        for p, planner in enumerate(self.md.planners):
            if alarmed[p]:
                continue
            if planner.queued() > 0 and planner.st.alarm is None and planner.st.cur.phase not in (5, 6):
                return False
        return all(self.runs[p] >= self.parts or alarmed[p] for p in range(self.md.P))


@dataclass
class StepRecord:
    tick0: int
    tick1: int
    wrench_T0: list                 # per contact [n_ticks, 6] tick means (T0)
    wrench_T0_peak: list
    torque_Nm: list                 # per contact [n_ticks]
    removed_m3: float
    analytic_m3: float
    in_cut_s: float
    slots: list
    l0_ticks: dict
    energy: dict = field(default_factory=dict)
    cpu: dict = field(default_factory=dict)     # process CPU s: step (whole driver step), hot (HotLoop.run_step),
    #                                             macro (run_macro), build, absorb (the hot loop's), l2 (domains)


@dataclass
class DriveResult:
    steps: list
    removed_volume_m3: float
    field_volume_change_m3: float
    analytic_chip_volume_m3: float
    chip_mass_kg: float
    density_kg_m3: float
    cycle_time_s: float
    end_tick: int
    finished: bool
    findings: list
    ledger: dict
    field: object                   # removal DexelField of workpiece 0 at the end
    counts: dict

    def wrenches(self, path_index: int = 0) -> np.ndarray:
        """[n_ticks, 6] tick-mean tool wrench (T0) over the run (zeros while the path was not cutting)."""
        rows = []
        for st in self.steps:
            n = st.tick1 - st.tick0
            w = st.wrench_T0[path_index] if path_index < len(st.wrench_T0) else None
            rows.append(np.asarray(w, float) if w is not None else np.zeros((0, 6)))
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, 6))


class OperationDriver:
    """Runs one operation of a resolved project (module doc). ``use_dynamics`` False: rigid tool and work;
    ``dynamics_l2`` False: dynamics' L2 analysis (chatter detection, stability limits and lobe diagrams in its
    macro_step) is not run - the hot loop's dynamics (modes, unbalance, base excitation, deflections) is
    unchanged by it; ``cutting_model_factory``: cutting's CoefficientModel per tool slot (default: the
    mechanics pipeline; cnctwin.cutting.coefficients.LinearModel for verification)."""

    def __init__(self, project, *, op_index: int = 0, db=None, use_dynamics: bool = True, recorder=None,
                 l0_override: dict | None = None, seed: int | None = None, check_energy: bool = True,
                 dynamics_l2: bool = True, cutting_model_factory=None):
        from ..cutting.domain import CuttingDomain
        from ..controller.domain import ControllerDomain
        from ..db import Database
        from ..dynamics.domain import DynamicsDomain
        from ..motion.domain import MotionDomain
        from ..project import resolve
        from ..removal.domain import RemovalDomain
        from ..tooling.domain import ToolingDomain
        from .hotloop import HotLoop
        from .ledger import Ledger
        if db is None:
            db = Database.load()
        if isinstance(project, (str, Path)):
            project = resolve(project, db)
        self.project = project
        machine = project.machine
        ipo_s = float(machine.controller.ipo_period_s)
        tick_s = float(project.physics.tick_s) if getattr(project.physics, "tick_s", None) else 1e-6
        ctx = RunContext(project=project, db=db, run_dir=None, seed=int(seed if seed is not None else
                                                                          (getattr(project, "seed", 0) or 0)),
                         tick_s=tick_s, ipo_ticks=int(round(ipo_s / tick_s)), macro_ipo=int(project.physics.macro_ipo))
        state = SimState()
        state.clock.op_index = int(op_index)
        self.ctx = ctx
        self.state = state
        op = project.operations[op_index]
        mtype = str(getattr(machine, "type", "") or "")
        F_H = fixture_pose_F(machine, op.fixture_pose_F_H) if mtype.endswith("mill") else \
            np.asarray(op.fixture_pose_F_H, float)
        self.tooling = ToolingDomain()
        self.tooling.init(ctx, state)
        self.controller = ControllerDomain()
        self.controller.init(ctx, state)
        self.motion = MotionDomain(poses_F_W=[F_H], poses_W_S=[op.stock_pose_W_S])
        self.motion.init(ctx, state)
        if self.motion.P > 1:
            self.motion._poses_F_W = [F_H] * self.motion.P
        ctx.timeline = self.motion.timeline
        self.cutting = CuttingDomain(model_factory=cutting_model_factory)
        self.cutting.init(ctx, state)
        self.removal = RemovalDomain()
        self.removal.init(ctx, state)
        self.dynamics = DynamicsDomain()
        self.dynamics.init(ctx, state)
        self.use_dynamics = bool(use_dynamics)
        self.dynamics_l2 = bool(dynamics_l2)
        if recorder is not None:
            ctx.recorder = recorder
        self.hot = HotLoop(ctx, state, motion=self.motion, removal=self.removal, cutting=self.cutting,
                           tooling=self.tooling, dynamics=self.dynamics, use_dynamics=self.use_dynamics,
                           recorder=recorder, l0_override=l0_override)
        self.feeder = Feeder(ctx, state, self.controller, self.motion)
        self.ledger = Ledger.from_context(ctx, domains={"motion", "removal"} | ({"dynamics"} if use_dynamics else set()))
        self.check_energy = bool(check_energy)
        self.tol_energy = float(getattr(project.physics, "tol_energy", 1e-6) or 1e-6)
        self.energy = {"in_J": 0.0, "out_J": 0.0, "abs_J": 0.0, "worst_step_rel": 0.0}
        from ..removal import field as FD
        rs = state.removal
        self.rho = float(rs.workpieces[0].rho_kg_m3) if hasattr(rs.workpieces[0], "rho_kg_m3") else \
            float(project.material.physical.density_kg_m3)
        self.v0 = _volumes(rs)
        self.booked = [0.0 for _ in rs.workpieces]      # removal records per workpiece (polar or tri-dexel volume)
        self.steps: list[StepRecord] = []
        self.publishers: list = []        # callables(driver) after the domains' macro_step: records of absent domains
        self.observers: list = []         # callables(driver, StepRecord) after each step (tests, the full loop)
        self.removed = 0.0
        self.analytic = 0.0
        self.ledger_values: dict = {}

    # ------------------------------------------------------------------ one macro step
    def step(self) -> StepRecord:
        import time as _time
        t_cpu = _time.process_time()
        ctx, state = self.ctx, self.state
        state.bus_next = Bus()
        rep = self.hot.run_step(self.feeder)
        t_l2 = _time.process_time()
        prev, nxt = state.bus_prev, state.bus_next
        self.motion.macro_step(ctx, state, prev, nxt)
        self.cutting.macro_step(ctx, state, prev, nxt)
        self.removal.macro_step(ctx, state, prev, nxt)
        if self.use_dynamics and self.dynamics_l2:
            self.dynamics.macro_step(ctx, state, prev, nxt)
        for publish in self.publishers:
            publish(self)
        cpu_l2 = _time.process_time() - t_l2
        tick1 = rep.tick1
        if self.ledger is not None:
            self.ledger_values = self.ledger.check(ctx, state, tick1)
        self._energy_step()
        rec = self._record(rep)
        rec.cpu = {"hot": rep.cpu_s, "macro": rep.cpu_macro_s, "build": rep.cpu_build_s, "absorb": rep.cpu_absorb_s,
                   "l2": cpu_l2, "l0_steps": float(rep.l0_steps)}
        self._exact_volume()
        rec.cpu["step"] = _time.process_time() - t_cpu
        for observe in self.observers:
            observe(self, rec)
        state.clock.tick = tick1
        state.clock.macro += 1
        state.swap_bus()
        return rec

    def _record(self, rep) -> StepRecord:
        nxt = self.state.bus_next
        removed = float(sum(c.removed_volume_m3 for c in nxt.removal.contacts))
        for c in nxt.removal.contacts:
            if 0 <= int(c.workpiece) < len(self.booked):
                self.booked[int(c.workpiece)] += float(c.removed_volume_m3)
        analytic = float(sum(c.analytic_chip_volume_m3 for c in nxt.cutting.contacts))
        self.removed += removed
        self.analytic += analytic
        rec = StepRecord(rep.tick0, rep.tick1, [np.asarray(c.wrench_tool_T0_mean, float) for c in nxt.cutting.contacts],
                         [np.asarray(c.wrench_tool_T0_peak, float) for c in nxt.cutting.contacts],
                         [np.asarray(c.spindle_torque_Nm, float) for c in nxt.cutting.contacts], removed, analytic,
                         float(sum(c.in_cut_s for c in nxt.cutting.contacts)),
                         [int(c.tool_slot) for c in nxt.cutting.contacts], dict(rep.l0_ticks),
                         {k: float(v) for k, v in self.ledger_values.items()} | dict(self.energy))
        self.steps.append(rec)
        return rec

    def _energy_step(self) -> None:
        """Energy balance of the step from its records (module doc): cutting's identity shaft + feed = heat by zone +
        ploughing + recut + packing + the work on the deflection rates (its quadrature), checked to round-off;
        the modes' own identity is dynamics' (its absorb raises); the difference between cutting's quadrature
        and dynamics' exact first-order-hold work on the modes is reported (docs/models/dynamics.md "Ledger
        note": 0.5-3 % of the vibration work). Raises when the run's cumulative residual leaves
        ``ENERGY_ROUNDOFF`` of the run's term magnitudes."""
        nxt = self.state.bus_next
        e_in = e_out = mag = w_def = 0.0
        for c in nxt.cutting.contacts:
            terms_in = (float(c.shaft_work_J), float(c.feed_work_J))
            terms_out = (float(c.heat_shear_J), float(c.heat_rake_J), float(c.heat_flank_J), float(c.ploughing_J),
                         float(getattr(c, "recut_J", 0.0)), float(getattr(c, "packing_J", 0.0)),
                         float(getattr(c, "deflection_work_J", 0.0)))
            w_def += terms_out[-1]
            e_in += sum(terms_in)
            e_out += sum(terms_out)
            mag += sum(abs(v) for v in terms_in + terms_out)
        en = self.energy
        en["in_J"] += e_in
        en["out_J"] += e_out
        en["abs_J"] += mag
        en["deflection_work_J"] = en.get("deflection_work_J", 0.0) + w_def
        if self.use_dynamics:
            en["modal_cutting_work_J"] = en.get("modal_cutting_work_J", 0.0) + float(nxt.dynamics.cutting_work_J)
        if mag > 0.0:
            en["worst_step_rel"] = max(en["worst_step_rel"], abs(e_in - e_out) / mag)
        if self.check_energy and en["abs_J"] > 0.0:
            rel = abs(en["in_J"] - en["out_J"]) / en["abs_J"]
            if rel > ENERGY_ROUNDOFF:
                raise AssertionError(f"energy balance of the run: {en['in_J']!r} J in, {en['out_J']!r} J out "
                                     f"(relative {rel:.3g} > {ENERGY_ROUNDOFF:g})")

    def _exact_volume(self) -> None:
        """The field's own volume change equals the booked removal per workpiece (integer quanta; the ledger's
        exactness): the polar field of a turned workpiece (authoritative while turning; removal books its radial
        quanta and rebuilds the tri-dexel copy from it once per polar_sync_revolutions, a resampling that is not
        booked), the tri-dexel z family otherwise."""
        for w, (change, booked) in enumerate(zip(self._changes(), self.booked)):
            scale = max(abs(change), abs(booked), 1e-300)
            if abs(change - booked) > 1e-9 * scale and abs(change - booked) > 1e-18:
                raise AssertionError(f"workpiece {w}: removed volume {booked!r} m3 booked, field changed by "
                                     f"{change!r} m3")

    def _changes(self) -> list[float]:
        now = _volumes(self.state.removal)
        return [(a[1] - b[1]) if a[1] is not None and b[1] is not None else (a[0] - b[0])
                for a, b in zip(self.v0, now)]

    # ------------------------------------------------------------------ checkpoints between macro steps
    def checkpoint(self) -> dict:
        """Everything that continues the run bit-identically (taken between macro steps): the whole SimState
        through the core checkpoint codec (domain states through their to_arrays), the hot loop's own state, the
        core ledger's and the driver's counters. A plain dict of JSON-able values and NumPy arrays."""
        from .checkpoint import encode
        document, arrays = encode(self.state)
        hot_arrays, hot_meta = self.hot.to_arrays()
        return {"document": document, "arrays": arrays, "hot": (hot_arrays, hot_meta),
                "ledger": self.ledger.checkpoint_state() if self.ledger is not None else None,
                "driver": {"removed": self.removed, "analytic": self.analytic, "v0": [list(v) for v in self.v0],
                           "booked": list(self.booked),
                           "energy": dict(self.energy), "runs": list(self.feeder.runs),
                           "ended": list(self.feeder.ended)}}

    def restore(self, ck: dict) -> None:
        """Continue from ``checkpoint()`` (this driver's domain objects are bound to the restored state, as in a
        fresh process)."""
        from .checkpoint import decode
        state = decode(ck["document"], ck["arrays"])
        self.state = state
        ctx = self.ctx
        self.motion.attach(ctx, state)
        ctx.timeline = self.motion.timeline
        self.controller.controller(ctx, state)
        self.hot.state = state
        self.hot.from_arrays(*ck["hot"])
        self.feeder.state = state
        self.feeder.runs = list(ck["driver"]["runs"])
        self.feeder.ended = list(ck["driver"]["ended"])
        if self.ledger is not None and ck["ledger"] is not None:
            self.ledger.restore_state(ck["ledger"])
        d = ck["driver"]
        self.removed = float(d["removed"])
        self.analytic = float(d["analytic"])
        self.v0 = [tuple(v) for v in d["v0"]]
        self.booked = [float(v) for v in d["booked"]]
        self.energy = dict(d["energy"])

    # ------------------------------------------------------------------ the operation
    def run(self, max_s: float | None = None, until=None) -> DriveResult:
        """Macro steps until the program ends (or ``max_s`` simulated seconds, or ``until(driver)`` is true)."""
        ctx, state = self.ctx, self.state
        limit = int(round(max_s / ctx.tick_s)) if max_s is not None else None
        finished = False
        while True:
            if limit is not None and state.clock.tick >= limit:
                break
            if until is not None and until(self):
                break
            self.step()
            if self.feeder.finished():
                finished = True
                break
        return self.result(finished)

    def result(self, finished: bool = False) -> DriveResult:
        state = self.state
        rs = state.removal
        change = float(sum(self._changes()))
        ends = [e for e in state.motion.ledger.get("end", []) if e is not None]
        end_tick = max(ends) if ends else int(state.clock.tick)
        return DriveResult(self.steps, self.removed, change, self.analytic, self.rho * self.removed, self.rho,
                           end_tick * self.ctx.tick_s, int(state.clock.tick), finished, list(state.findings),
                           (dict(self.ledger.summary()) if self.ledger is not None else {}) | {"energy_run": dict(self.energy)},
                           rs.workpieces[0].field, {k: int(v) for k, v in zip(_COUNT_NAMES, self.hot.g.count)}
                           if self.hot.g is not None else {})


def _volumes(rs) -> list[tuple]:
    """(tri-dexel z-family volume, polar field volume or None) per workpiece."""
    from ..removal import field as FD
    from ..removal import polar as PO
    return [(float(FD.volume_m3(wp.field, 0, 2)), float(PO.volume_m3(wp.polar)) if wp.polar is not None else None)
            for wp in rs.workpieces]


ENERGY_ROUNDOFF = 1e-9      # cutting's energy identity is exact by construction: this bounds accumulated round-off
_COUNT_NAMES = ("l0_steps", "ticks", "passes", "refills", "polar_cuts", "helix_calls", "free_ticks", "splits",
                "probe_sweeps", "broad_phase", "unused10", "unused11")


def project_variant(src_dir, dst_dir, *, program_text: str | None = None, program_name: str = "op.nc",
                    edits=None) -> Path:
    """Copy of a project directory with its operation 0 program replaced and the project mapping edited
    (``edits(data)`` changes the parsed project.yaml in place). Returns the new project file."""
    import shutil

    import yaml
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load((src / "project.yaml").read_text())
    for name in ("part.step",):
        if (src / name).exists() and not (dst / name).exists() and data.get("cad"):
            shutil.copy(src / name, dst / name)
    if program_text is not None:
        (dst / program_name).write_text(program_text)
        data["operations"][0]["program"] = program_name
    else:
        prog = data["operations"][0]["program"]
        shutil.copy(src / prog, dst / prog)
    if edits is not None:
        edits(data)
    out = dst / "project.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


__all__ = ["OperationDriver", "DriveResult", "StepRecord", "Feeder", "fixture_pose_F", "project_variant"]
