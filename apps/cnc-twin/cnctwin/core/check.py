"""``twin check`` (docs/models/loop.md "twin check"): what a run of a project will need, before running it.

:func:`check_project` ``(request, out_dir, progress_cb) -> dict`` per operation:

* the controller's dry run (ideal machine: the cycle-time lower bound, alarms, the tools and spindle speeds the
  program uses) and the planner's cycle time (motion's MotionRunner: acceleration, blending, block rate);
* the L0 stride of every tool / spindle segment by architecture 5.1 (the largest divisor of ipo_ticks with
  max |omega| dt <= physics.max_deg_per_step, physics.steps_per_tooth steps per tooth period and 2 pi f_max dt
  <= 0.3 for the modes carrying >= 1 % of the tip compliance; max |omega| = programmed speed x the machine's
  largest spindle override, the CSS clamp for G96). ``options.write`` stores them in ``physics.resolved``
  (cnctwin.project.write_resolved_physics), which a run uses;
* the surface-memory rule: ``physics.memory_bins`` not finer than one bin per L0 step at the stride the run takes
  (``memory_bins <= 360 deg / angle per step``; finer bins add an error, docs/convergence.md 4.2): a warning per
  tool that breaks it;
* the lattice rule (``lattice_mm <= ae / 4`` on every cutting move, or a refine box covering the moves that break
  it, at ``lattice_mm / factor``) from sanity's engagement estimate re-run on a height map of a quarter of the
  lattice (the 0.5 mm map of sanity's rules misses or widens finishing passes thinner than a cell): a warning per
  finishing pass that breaks it (docs/convergence.md 2: the ledger's removed volume of such a layer errs by up to
  lattice / (2 ae));
* the estimated CPU seconds per simulated second (cost model in db/models/loop.yaml from element, mode, step and
  tick counts) and the disk ``fast: all`` would take;
* sanity's findings of the operation.

:func:`ring_l0_ticks` and :func:`dry_run_operation` are shared with the run loop.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .errors import InputError, TwinError

FORMAT = "cnctwin-check"
ZMAP_MIN_CELL_M = 5.0e-5      # sanity-rules zmap_cell_m range: the finest height map the lattice rule asks for
LATTICE_AE_RATIO = 4.0         # architecture 5.4: lattice_mm <= ae / 4
MAX_PASS_WARNINGS = 20         # finishing passes named one by one per operation (the rest counted)


# =============================================================================== strides

def _physics(ctx, name, default):
    value = getattr(ctx.project.physics, name, None)
    return default if value is None else value


def _resolved_entry(e: dict) -> dict:
    """One physics.resolved segment (project schema RESOLVED_SEGMENT): f_max and the speed only when positive (a
    tool without modes, a non-rotating tool)."""
    out = {"tool": e["tool"], "path": e["path"], "l0_ticks": e["l0_ticks"], "teeth": e["teeth"], "rule": e["rule"]}
    if e["max_rpm"] > 0.0:
        out["max_rpm"] = round(e["max_rpm"], 3)
    if e["f_max_Hz"] > 0.0:
        out["f_max_Hz"] = round(e["f_max_Hz"], 3)
    return out


def stride(ctx, omega_max_rad_s: float, n_teeth: int, f_max_Hz: float) -> tuple[int, str]:
    """(l0_ticks, binding rule) by architecture 5.1 (dynamics' l0_ticks_rule)."""
    from ..dynamics import params as DP
    from ..dynamics.domain import l0_ticks_rule
    rad = float(_physics(ctx, "max_rad_per_step", 0.5 * math.pi / 180.0))
    spt = int(_physics(ctx, "steps_per_tooth", 96))
    omega_dt = float(DP.p(DP.MODES, "stride_omega_dt"))
    l0 = int(l0_ticks_rule(int(ctx.ipo_ticks), float(ctx.tick_s), float(abs(omega_max_rad_s)), max(int(n_teeth), 1),
                           float(f_max_Hz), rad, spt, omega_dt))
    dt = l0 * ctx.tick_s
    # the rule names of project schema's physics.resolved: angle, tooth, mode, ipo
    bounds = {"angle": rad / max(abs(omega_max_rad_s), 1e-300),
              "tooth": 2.0 * math.pi / (max(abs(omega_max_rad_s), 1e-300) * max(int(n_teeth), 1) * spt),
              "mode": omega_dt / (2.0 * math.pi * f_max_Hz) if f_max_Hz > 0.0 else math.inf}
    rule = min(bounds, key=bounds.get) if min(bounds.values()) < math.inf else "ipo"
    if min(bounds.values()) >= ctx.ipo_ticks * ctx.tick_s or dt >= ctx.ipo_ticks * ctx.tick_s:
        rule = "ipo"
    return l0, rule


def memory_bins_limit(omega_rad_s: float, l0_ticks: int, tick_s: float) -> tuple[float, int | None]:
    """(spindle angle per L0 step in degrees, the most surface-memory bins per revolution not finer than one step:
    floor(360 / angle)). An element's chip is evaluated once per L0 step, at the centroid of the angle it sweeps, so
    a memory bin narrower than that angle is written from the step's first-order motion and finer bins move the
    result away from the converged one (docs/convergence.md 4.2: 2880 bins at 0.48 deg per step, +1.5 % mean force,
    +2.5 % chip volume). None: no limit (a spindle at rest)."""
    angle = math.degrees(abs(float(omega_rad_s)) * int(l0_ticks) * float(tick_s))
    if angle <= 0.0:
        return 0.0, None
    return angle, int(math.floor(360.0 / angle + 1e-9))


def _resolved_l0(project, op_name: str, tool: int, path: int) -> int | None:
    """The L0 stride physics.resolved already stores for the tool on that path of the operation (the run uses it)."""
    resolved = getattr(project.physics, "resolved", None)
    ops = getattr(resolved, "operations", None) if resolved is not None else None
    entries = ops.get(op_name) if (ops and hasattr(ops, "get")) else None
    for entry in entries or []:
        if int(getattr(entry, "tool", -1)) == int(tool) and int(getattr(entry, "path", 1) or 1) == int(path):
            value = getattr(entry, "l0_ticks", None)
            return int(value) if value else None
    return None


def memory_bins_warning(op_name: str, entry: dict, memory_bins: int) -> str | None:
    """The warning for a stride entry whose surface memory is finer than one bin per L0 step, else None."""
    limit = entry.get("memory_bins_max")
    if limit is None or int(memory_bins) <= int(limit):
        return None
    return (f"{op_name}: T{entry['tool']}: physics.memory_bins {int(memory_bins)} is finer than one bin per L0 step "
            f"({entry['angle_per_step_deg']:.3f} deg per step at {entry['rpm']:.0f} rpm, stride {entry['l0_run_ticks']} "
            f"base ticks): at most {limit} bins; finer bins add an error to the chip thickness and the forces "
            f"(docs/convergence.md 4.2): refine the stride with the bins (max_deg_per_step <= 360 / memory_bins) or "
            f"use at most {limit} bins")


def _machine_max_omega(machine) -> float:
    best = 0.0
    for sp in getattr(machine, "spindles", None) or []:
        value = getattr(sp, "max_rad_s", None)
        if value is not None:
            best = max(best, float(value))
        for gear in getattr(sp, "gear_ranges", None) or []:
            span = getattr(gear, "rad_s", None)
            if span is not None:
                best = max(best, float(np.max(np.asarray([float(v) for v in span]))))
    return best


def _override_max(machine) -> float:
    ctrl = getattr(machine, "controller", None)
    rng = getattr(ctrl, "spindle_override_range_frac", None) if ctrl is not None else None
    if rng is None:
        return 1.0
    values = [float(v) for v in rng]
    return max(values[-1], 1.0) if values else 1.0


def ring_l0_ticks(ctx, state) -> int:
    """The smallest L0 stride a run of the project can take (it sizes the recorder's fast ring): the stride rule at
    the machine's highest spindle speed for the tool with the most teeth, with modes up to loop.ring_f_max_Hz."""
    from ..db.models import params
    omega = _machine_max_omega(ctx.project.machine) * _override_max(ctx.project.machine)
    teeth = 1
    tooling = getattr(state, "tooling", None)
    if tooling is not None:
        for slot in range(len(tooling)):
            teeth = max(teeth, int(np.asarray(tooling.effective[slot].tooth_phase_rad).shape[0]))
    l0, _ = stride(ctx, omega if omega > 0.0 else 0.0, teeth, float(params("loop")["ring_f_max_Hz"]))
    return max(1, int(l0))


def tool_f_max_Hz(ctx, geometry, share: float) -> tuple[float, int]:
    """(highest frequency among the modes carrying at least ``share`` of the tip compliance, number of modes) of a
    tool in its holder and spindle (structure's tool model, tip station, x and y) together with the machine's frame
    and table modes (1/k each)."""
    from ..structure.schema import settings as structure_settings
    from ..structure.toolmodel import tool_modes
    machine = ctx.project.machine
    settings = structure_settings(getattr(ctx.project.physics, "structure", None))
    try:
        result = tool_modes(geometry, machine, settings, db=ctx.db)
    except (TwinError, ValueError) as error:
        raise InputError(f"tool T{geometry.number}: its modes cannot be computed ({error})") from None
    tm = result.modes
    freq = np.asarray(tm.freq_Hz, float)
    shape = np.asarray(tm.shape, float)
    comp, freqs = [], []
    for k in range(len(freq)):
        w2 = (2.0 * math.pi * freq[k]) ** 2
        if shape.ndim == 3 and shape.shape[1] > 0:
            comp.append(max(shape[k, 0, 0] ** 2, shape[k, 0, 1] ** 2) / w2)
            freqs.append(float(freq[k]))
    residual = np.asarray(getattr(tm, "static_residual_m_per_N", np.zeros((3, 3))), float)
    total_extra = float(max(residual[0, 0], residual[1, 1])) if residual.shape == (3, 3) else 0.0
    structure = getattr(machine, "structure", None)
    for group in ("modes", "table_modes"):
        for md in (getattr(structure, group, None) or []) if structure is not None else []:
            fn = getattr(md, "fn_Hz", None)
            k = getattr(md, "k_N_per_m", None)
            if fn is not None and k is not None and float(k) > 0.0:
                comp.append(1.0 / float(k))
                freqs.append(float(fn))
    total = float(sum(comp)) + total_extra
    if total <= 0.0:
        return 0.0, len(freqs)
    keep = [f for f, c in zip(freqs, comp) if c >= share * total]
    return (float(max(keep)) if keep else 0.0), len(freqs)


# =============================================================================== the controller's dry run

def dry_run_operation(ctx, state, op_index: int, max_s: float | None = None) -> dict:
    """The controller's dry run of an operation (ideal machine, no acceleration: a lower bound of the cycle time):
    ``{cycle_time_s, feed_time_s, commands, segments: [{tool, slot, path, max_rad_s, css}], alarms, stopped}``."""
    from ..controller import Controller, dry_run
    from ..controller.domain import setup_from_project, stock_probe
    from ..controller.ir import Kind
    setup = setup_from_project(ctx, state, op_index)
    controller = Controller(setup)
    limit = int(round((max_s if max_s is not None else 86400.0) / ctx.tick_s))
    result = dry_run(controller, max_tick=limit, probe=stock_probe(ctx.project, op_index))
    segments: dict = {}
    feed_ticks = 0
    for p, commands in enumerate(result.commands):
        tool, slot, omega, clamp, css = None, -1, 0.0, None, False
        for k, cmd in enumerate(commands):
            if cmd.kind == Kind.TOOL_CHANGE and cmd.tool:
                tool = cmd.tool.get("number")
                slot = int(cmd.tool.get("slot", -1) if cmd.tool.get("slot") is not None else -1)
            elif cmd.kind == Kind.SPINDLE and cmd.spindle:
                omega = abs(float(cmd.spindle.get("omega_rad_s") or 0.0))
                clamp = cmd.spindle.get("max_rad_s")
                css = cmd.spindle.get("css_m_s") is not None
            if cmd.kind in (Kind.LINEAR, Kind.ARC) and tool is not None:
                key = (int(tool), p)
                seg = segments.setdefault(key, {"tool": int(tool), "slot": slot, "path": p + 1, "max_rad_s": 0.0,
                                                "css": False})
                speed = float(clamp) if (css and clamp) else omega
                seg["max_rad_s"] = max(seg["max_rad_s"], speed)
                seg["css"] = seg["css"] or css
                if k < len(result.start_ticks[p]):
                    feed_ticks += int(result.end_ticks[p][k]) - int(result.start_ticks[p][k])
    return {"cycle_time_s": int(result.end_tick) * ctx.tick_s, "feed_time_s": feed_ticks * ctx.tick_s,
            "commands": int(sum(len(c) for c in result.commands)), "segments": sorted(segments.values(),
                                                                                       key=lambda s: (s["path"],
                                                                                                      s["tool"])),
            "alarms": [a for a in result.alarms if a], "stopped": str(result.stopped or "")}


def planner_cycle_time(ctx, op_index: int, max_s: float) -> dict:
    """motion's planner over the controller (MotionRunner: acceleration, blending, block rate; no cutting)."""
    from ..controller.domain import ControllerDomain
    from ..core.state import SimState
    from ..motion.domain import MotionDomain
    from ..motion.driver import MotionRunner
    from ..tooling.domain import ToolingDomain
    from .drive import fixture_pose_F
    project = ctx.project
    state = SimState()
    state.clock.op_index = int(op_index)
    ToolingDomain().init(ctx, state)
    controller = ControllerDomain()
    controller.init(ctx, state)
    op = project.operations[op_index]
    machine = project.machine
    mtype = str(getattr(machine, "type", "") or "")
    F_H = fixture_pose_F(machine, op.fixture_pose_F_H) if mtype.endswith("mill") else np.asarray(op.fixture_pose_F_H)
    motion = MotionDomain(poses_F_W=[F_H], poses_W_S=[np.asarray(op.stock_pose_W_S, float)])
    motion.init(ctx, state)
    runner = MotionRunner(controller.controller(ctx, state), motion, ctx=ctx, state=state)
    result = runner.run(max_s)
    return {"cycle_time_s": int(result.end_tick) * ctx.tick_s, "stopped": str(result.stopped or ""),
            "steps": int(result.steps)}


# =============================================================================== twin check

def _context(project_path):
    from ..db import Database
    from ..project import resolve
    from .domain import RunContext
    db = Database.load()
    project = resolve(project_path, db)
    tick_s = float(getattr(project.physics, "tick_s", None) or 1e-6)
    ipo_ticks = int(round(float(project.machine.controller.ipo_period_s) / tick_s))
    return RunContext(project=project, db=db, run_dir=None, seed=int(getattr(project, "seed", 0) or 0),
                      tick_s=tick_s, ipo_ticks=ipo_ticks, macro_ipo=int(project.physics.macro_ipo))


def fine_engagement(report, lattice_m: float):
    """(engagement, height-map cell): sanity's engagement estimate re-run on a map of a quarter of the lattice. The
    estimate's ap and ae are within about one cell (cnctwin/sanity/geometry.py); on sanity's own 0.5 mm map a
    finishing pass thinner than a cell is found only where a cell centre happens to lie in it and its ae comes out a
    cell wide, so the lattice rule (ae against 4 x lattice) needs cells of lattice / 4 (the map grows its cell above
    zmap_max_cells, as sanity's does). The report's own estimate when its map is already that fine."""
    from dataclasses import replace

    from ..sanity.geometry import estimate
    ctx = report.context
    coarse = float(ctx.p("zmap_cell_m"))
    cell = max(min(coarse, lattice_m / LATTICE_AE_RATIO), ZMAP_MIN_CELL_M)
    if cell >= coarse or report.program is None:
        return report.engagement, coarse
    return estimate(report.program, replace(ctx, params={**ctx.params, "zmap_cell_m": cell})), cell


def _refine_boxes_M(project, op_index: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """physics.refine boxes (stock frame S, removal's levels: spacing lattice / factor) as (lo, hi, factor) boxes in
    machine coordinates of the operation's setup (sanity's frame before the work offset)."""
    from ..removal.schema import settings as removal_settings
    from ..sanity.context import Box, _setup_transform, transform_box
    from ..sanity.si import get, si_view
    boxes = []
    raw = removal_settings(project.physics).refine
    if not raw:
        return boxes
    setup = si_view(get(get(si_view(project), "operations", op_index), "setup"))
    T = _setup_transform(setup)
    for lo, hi, factor in raw:
        box = transform_box(T, Box(tuple(lo), tuple(hi), "refine"))
        boxes.append((np.asarray(box.lo, float), np.asarray(box.hi, float), int(factor)))
    return boxes


def _move_region(program, sctx, index: int, ap: float):
    """(lo, hi) in machine coordinates of the material a cutting move engages: its tool disc swept from the start
    to the end point, from the tip up to the engaged depth; None when the tool or the work offset is unknown."""
    from ..sanity.context import normalise_offset
    command = program.commands[index]
    modal = command.modal or {}
    tool = sctx.tools.get(modal.get("tool"))
    offset = sctx.work_offsets_m.get(normalise_offset(str(modal.get("work_offset", "G54"))))
    if tool is None or tool.diameter_m is None or offset is None:
        return None
    radius = 0.5 * float(tool.diameter_m)
    a = np.asarray(program.tip_start_W_m[index], float) + np.asarray(offset, float)
    b = np.asarray(program.tip_end_W_m[index], float) + np.asarray(offset, float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    lo = np.minimum(a, b) - np.array([radius, radius, 0.0])
    hi = np.maximum(a, b) + np.array([radius, radius, max(float(ap), 0.0)])
    return lo, hi


def effective_lattice(lattice_m: float, region, boxes) -> float:
    """The lattice a move cuts on: lattice / factor of the finest refine box that contains its whole region (the
    base lattice when none does, or when the region is unknown)."""
    best = float(lattice_m)
    if region is None:
        return best
    lo, hi = region
    for blo, bhi, factor in boxes:
        if np.all(lo >= blo - 1e-12) and np.all(hi <= bhi + 1e-12):
            best = min(best, float(lattice_m) / max(int(factor), 1))
    return best


def finishing_passes(program, sctx, engagement, lattice_m: float, boxes, cell_m: float = 0.0) -> list[dict]:
    """Cutting moves whose radial engagement is below 4 x the lattice they cut on, grouped into passes (consecutive
    breaking moves of one tool): lines, the smallest ae, the lattice in force, the lattice the rule asks for and the
    first-order bound of the ledger's removed-volume error of the layer, lattice / (2 ae) (docs/convergence.md 2).
    sanity's ae is the engaged cells' width plus one cell (``cell_m``), within one cell of the true width: the rule
    is decided on it, the recommendation (required lattice, refine factor, error bound) on its lower bound
    ``ae_min_m`` = ae - one cell, so it is never too coarse."""
    ae = np.asarray(engagement.ae_m, float)
    cut = np.asarray(engagement.cut, bool)
    ap = np.asarray(engagement.ap_m, float)
    passes: list[dict] = []
    current = None
    for i in range(min(len(program.commands), ae.size)):
        if not (cut[i] and np.isfinite(ae[i]) and ae[i] > 0.0):
            continue
        region = _move_region(program, sctx, i, ap[i] if np.isfinite(ap[i]) else 0.0)
        lat = effective_lattice(lattice_m, region, boxes)
        if ae[i] >= LATTICE_AE_RATIO * lat:
            current = None
            continue
        command = program.commands[i]
        tool = (command.modal or {}).get("tool")
        line = int(getattr(command.ref, "line", 0) or 0)
        if current is not None and current["tool"] == tool and current["last_index"] == i - 1:
            current["line_last"] = line
            current["ae_m"] = min(current["ae_m"], float(ae[i]))
            current["lattice_m"] = max(current["lattice_m"], lat)
            current["moves"] += 1
            current["last_index"] = i
        else:
            current = {"tool": tool, "program": str(getattr(command.ref, "program", "") or ""), "line_first": line,
                       "line_last": line, "ae_m": float(ae[i]), "lattice_m": lat, "moves": 1, "last_index": i}
            passes.append(current)
    for p in passes:
        p.pop("last_index", None)
        p["ae_min_m"] = max(p["ae_m"] - float(cell_m), min(float(cell_m), p["ae_m"])) if cell_m > 0.0 else p["ae_m"]
        p["required_m"] = p["ae_min_m"] / LATTICE_AE_RATIO
        p["refine_factor"] = int(math.ceil(lattice_m / p["required_m"] - 1e-9))
        p["volume_error_bound"] = p["lattice_m"] / (2.0 * p["ae_min_m"])
    return passes


def pass_warning(op_name: str, p: dict) -> str:
    lines = f"line {p['line_first']}" if p["line_first"] == p["line_last"] else \
        f"lines {p['line_first']}-{p['line_last']}"
    ae = f"ae {p['ae_m'] * 1e3:.3f} mm" if p.get("ae_min_m", p["ae_m"]) >= p["ae_m"] else \
        f"ae {p['ae_min_m'] * 1e3:.3f}-{p['ae_m'] * 1e3:.3f} mm"
    return (f"{op_name} {lines}: a pass of T{p['tool']} with thin radial engagement ({ae}: a finishing pass, a "
            f"trochoid's crescent) runs on a {p['lattice_m'] * 1e3:.4f} mm lattice, above ae / 4 = "
            f"{p['required_m'] * 1e3:.4f} mm: the ledger's "
            f"removed volume of the layer errs by up to lattice / (2 ae) = {100.0 * p['volume_error_bound']:.0f} %, and "
            f"the engaged arc and the peak force do not converge (docs/convergence.md 2); refine physics.lattice_mm to "
            f"{p['required_m'] * 1e3:.4f} mm or cover these lines with a refine box (physics.refine, factor "
            f">= {p['refine_factor']})")


def _lattice(ctx, op_index: int, report) -> dict:
    """The lattice rule of the operation (architecture 5.4, module doc): every finishing pass that breaks it (one
    warning each), on the fine engagement estimate, with refine boxes counted where they cover the whole pass."""
    lattice = float(getattr(ctx.project.physics, "lattice_m", 0.0) or 0.0)
    boxes = _refine_boxes_M(ctx.project, op_index)
    eng, cell = fine_engagement(report, lattice) if lattice > 0.0 else (getattr(report, "engagement", None), None)
    ae = np.asarray(getattr(eng, "ae_m", np.zeros(0)), float) if eng is not None else np.zeros(0)
    cut = np.asarray(getattr(eng, "cut", np.zeros(0)), bool) if eng is not None else np.zeros(0, bool)
    valid = ae[cut & np.isfinite(ae) & (ae > 0.0)] if ae.size and cut.size == ae.size else np.zeros(0)
    if valid.size == 0 or lattice <= 0.0:
        return {"lattice_m": lattice, "min_ae_m": None, "ok": True, "refine": len(boxes), "zmap_cell_m": cell,
                "passes": [], "note": "no cutting move with an evaluated radial engagement"}
    min_ae = float(np.min(valid))
    passes = finishing_passes(report.program, report.context, eng, lattice, boxes, float(cell or 0.0))
    finest = min([lattice / f for _, _, f in boxes] + [lattice])
    ok = not passes
    breaking = int(sum(p["moves"] for p in passes))
    return {"lattice_m": lattice, "min_ae_m": min_ae, "required_m": min_ae / LATTICE_AE_RATIO,
            "moves_breaking": breaking, "refine": len(boxes), "finest_refine_m": finest if boxes else None,
            "ok": bool(ok), "zmap_cell_m": cell, "passes": passes,
            "note": "lattice_mm <= ae / 4 on every cutting move, or a refine box over it (architecture 5.4)" if ok else
            f"{len(passes)} passes ({breaking} cutting moves) have ae below 4 x their lattice: refine the lattice to "
            f"{min_ae / LATTICE_AE_RATIO * 1e3:.4f} mm or add refine boxes over those passes"}


def check_project(request: dict, out_dir, progress_cb=None) -> dict:
    """``twin check`` (module docstring). ``request``: {project_path | project_dir, options: {write}, read_only}.
    Writes ``check.json`` into ``out_dir`` and returns the same document."""
    from ..db.models import params
    from ..sanity import check_operation
    from ..tooling.domain import ToolingDomain
    from .rundir import jsonable, write_json_atomic
    from .state import SimState

    def progress(**fields):
        if progress_cb is not None:
            progress_cb(fields)

    path = request.get("project_path") or (Path(request["project_dir"]) / "project.yaml")
    options = dict(request.get("options") or {})
    write = bool(options.get("write")) and not bool(request.get("read_only", False))
    ctx = _context(path)
    project = ctx.project
    state = SimState()
    ToolingDomain().init(ctx, state)
    p = params("loop")
    from ..dynamics import params as DP
    share = float(DP.p(DP.MODES, "stride_compliance_share"))
    override = _override_max(project.machine)
    bins = int(getattr(project.physics, "memory_bins", None) or 720)
    fast = _fast_columns(state)
    errors, warnings = [], []
    operations, resolved = [], {"ipo_ticks": int(ctx.ipo_ticks), "macro_ipo": int(ctx.macro_ipo), "operations": {}}
    f_cache: dict = {}
    total_cycle = total_planned = 0.0
    for i, op in enumerate(project.operations):
        name = str(op.name)
        progress(phase="check", op=name, message=f"{name}: controller dry run")
        state.clock.op_index = i
        try:
            dry = dry_run_operation(ctx, state, i)
        except TwinError as error:
            errors.append(f"{name}: the controller cannot run the program ({error})")
            operations.append({"op": name, "error": str(error)})
            continue
        for alarm in dry["alarms"]:
            errors.append(f"{name}: controller alarm {alarm.get('code')} {alarm.get('text')}")
        planned = None
        try:
            progress(phase="check", op=name, message=f"{name}: planner")
            planned = planner_cycle_time(ctx, i, max(10.0 * dry["cycle_time_s"] + 60.0, 60.0))
        except TwinError as error:
            warnings.append(f"{name}: the planner's cycle time is not available ({error})")
        entries = []
        cost_in_cut = 0.0
        for seg in dry["segments"]:
            slot = seg["slot"]
            if slot < 0:
                slot = next((s.slot for s in state.tools.slots
                             if s.number == seg["tool"] and s.path == seg["path"] - 1), -1)
            if slot < 0:
                warnings.append(f"{name}: T{seg['tool']} is not in the project's tool list (no stride)")
                continue
            geo = state.tooling.geometry(slot)
            rotating = bool(geo.rotating)
            teeth = int(np.asarray(state.tooling.effective[slot].tooth_phase_rad).shape[0]) if rotating else 1
            if slot not in f_cache:
                try:
                    f_cache[slot] = tool_f_max_Hz(ctx, geo, share)
                except InputError as error:
                    warnings.append(str(error))
                    f_cache[slot] = (0.0, 0)
            f_max, n_modes = f_cache[slot]
            omega = seg["max_rad_s"] * override
            l0, rule = stride(ctx, omega, teeth, f_max)
            elements = int(np.count_nonzero(np.asarray(state.tooling.effective[slot].elem_valid)))
            engaged = 0.25 * elements if rotating else float(elements)
            per_step = float(p["cost_l0_fixed_s"]) + float(p["cost_l0_element_s"]) * engaged + \
                float(p["cost_l0_mode_s"]) * n_modes
            cpu_per_s = per_step / (l0 * ctx.tick_s)
            cost_in_cut = max(cost_in_cut, cpu_per_s)
            entry = {"tool": seg["tool"], "path": seg["path"], "l0_ticks": int(l0),
                     "max_rpm": omega * 60.0 / (2.0 * math.pi), "f_max_Hz": f_max, "teeth": teeth,
                     "rule": rule, "css": bool(seg["css"]), "elements": elements, "modes": n_modes,
                     "cpu_s_per_s_in_cut": cpu_per_s,
                     "fast_all_bytes_per_s": _fast_bytes(fast, l0, ctx.tick_s)}
            if rotating:
                # the surface memory against the stride the run takes at the programmed speed: physics.resolved's
                # when the project stores one, else the stride rule at that speed (dynamics re-derives it from the
                # actual spindle speed every macro step)
                l0_run = _resolved_l0(project, name, seg["tool"], seg["path"]) or \
                    stride(ctx, seg["max_rad_s"], teeth, f_max)[0]
                angle, bins_max = memory_bins_limit(seg["max_rad_s"], l0_run, ctx.tick_s)
                entry.update({"l0_run_ticks": int(l0_run), "angle_per_step_deg": angle, "memory_bins_max": bins_max,
                              "rpm": seg["max_rad_s"] * 60.0 / (2.0 * math.pi)})
                note = memory_bins_warning(name, entry, bins)
                if note is not None:
                    warnings.append(note)
            entries.append(entry)
        resolved["operations"][name] = [_resolved_entry(e) for e in entries]
        progress(phase="check", op=name, message=f"{name}: sanity")
        try:
            report = check_operation(project, i, project_dir=project.dir)
            sanity = [{"rule": f.message.split(":", 1)[0], "severity": f.severity.value, "line": f.line,
                       "message": f.message} for f in report.findings]
            lattice = _lattice(ctx, i, report)
        except TwinError as error:
            sanity = [{"rule": "scan", "severity": "error", "line": 0, "message": str(error)}]
            lattice = {"ok": None, "note": f"no engagement estimate ({error})"}
        for f in sanity:
            if f["severity"] in ("error", "fatal"):
                errors.append(f"{name} line {f['line']}: {f['message']}")
        if lattice.get("ok") is False:
            passes = lattice.get("passes") or []
            warnings.extend(pass_warning(name, p) for p in passes[:MAX_PASS_WARNINGS])
            if len(passes) > MAX_PASS_WARNINGS:
                warnings.append(f"{name}: {len(passes) - MAX_PASS_WARNINGS} more passes break the lattice rule "
                                f"({lattice['note']})")
        cycle = float(planned["cycle_time_s"]) if planned else float(dry["cycle_time_s"])
        feed_s = min(float(dry["feed_time_s"]), cycle)
        steps = cycle / ctx.macro_s
        cpu = cost_in_cut * feed_s + float(p["cost_l1_tick_s"]) * cycle / (ctx.ipo_ticks * ctx.tick_s) + \
            float(p["cost_l2_step_s"]) * feed_s / ctx.macro_s + float(p["cost_l2_idle_step_s"]) * max(steps - feed_s /
                                                                                                    ctx.macro_s, 0.0)
        disk = sum(e["fast_all_bytes_per_s"] for e in entries[:1]) * feed_s if entries else 0.0
        total_cycle += float(dry["cycle_time_s"])
        total_planned += cycle
        operations.append({"op": name, "dry_run_cycle_time_s": dry["cycle_time_s"],
                           "planner_cycle_time_s": planned["cycle_time_s"] if planned else None,
                           "feed_time_s": dry["feed_time_s"], "commands": dry["commands"], "strides": entries,
                           "lattice": lattice, "sanity": sanity,
                           "estimated_cpu_s": cpu, "estimated_cpu_s_per_s": cpu / cycle if cycle > 0.0 else 0.0,
                           "fast_all_bytes": disk, "stopped": dry["stopped"]})
    written = None
    if write and not errors:
        from ..project import write_resolved_physics
        written = str(write_resolved_physics(path, resolved))
    elif write:
        warnings.append("physics.resolved not written: the check found errors")
    parts = int(getattr(getattr(project, "production", None), "parts", None) or 1)
    cpu_total = sum(o.get("estimated_cpu_s", 0.0) for o in operations) * parts
    lines = [f"{project.name}: {len(project.operations)} operations, {parts} part(s)"]
    for o in operations:
        if "error" in o:
            lines.append(f"  {o['op']}: {o['error']}")
            continue
        strides = ", ".join(f"T{e['tool']} l0 {e['l0_ticks']} ({e['rule']})" for e in o["strides"])
        lines.append(f"  {o['op']}: cycle {o['planner_cycle_time_s'] or o['dry_run_cycle_time_s']:.1f} s, "
                     f"{strides}; about {o['estimated_cpu_s_per_s']:.1f} CPU s per simulated s; fast: all "
                     f"{o['fast_all_bytes'] / 2 ** 20:.0f} MiB")
    lines.append(f"estimated CPU time of the run: {cpu_total:.0f} s")
    document = {"format": FORMAT, "version": 1, "project": str(project.name), "project_file": str(path),
                "ok": not errors, "errors": errors, "warnings": warnings, "operations": operations,
                "resolved": resolved, "written": written, "cycle_time_s": total_cycle,
                "planner_cycle_time_s": total_planned, "estimated_cpu_s": cpu_total, "parts": parts,
                "fast_columns": fast, "summary": "\n".join(lines), "exit_code": 0 if not errors else 1}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "check.json", jsonable(document))
    progress(phase="check", message="done")
    return jsonable(document)


def _fast_columns(state) -> int:
    """Number of fast (L0) columns a run records (hot loop and dynamics fast channels)."""
    from ..dynamics import channels as DCH
    from .channels import Tier
    from .hotloop import channels as hot_channels
    count = 0
    for channel in list(hot_channels()) + list(DCH.channels()):
        if channel.tier == Tier.FAST:
            count += len(channel.columns())
    return count


def _fast_bytes(n_columns: int, l0_ticks: int, tick_s: float) -> float:
    """Bytes per simulated second in contact that ``fast: all`` writes (the recorder's estimate)."""
    from .recorder import Recorder
    return float(Recorder.fast_all_bytes_per_second(int(n_columns), int(l0_ticks), float(tick_s)))


__all__ = ["check_project", "stride", "ring_l0_ticks", "dry_run_operation", "planner_cycle_time", "tool_f_max_Hz",
           "memory_bins_limit", "memory_bins_warning", "fine_engagement", "effective_lattice", "finishing_passes",
           "pass_warning"]
