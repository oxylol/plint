"""``twin check`` (docs/models/loop.md "twin check"): what a run of a project will need, before running it.

:func:`check_project` ``(request, out_dir, progress_cb) -> dict`` per operation:

* the controller's dry run (ideal machine: the cycle-time lower bound, alarms, the tools and spindle speeds the
  program uses) and the planner's cycle time (motion's MotionRunner: acceleration, blending, block rate);
* the L0 stride of every tool / spindle segment by architecture 5.1 (the largest divisor of ipo_ticks with
  max |omega| dt <= physics.max_deg_per_step, physics.steps_per_tooth steps per tooth period and 2 pi f_max dt
  <= 0.3 for the modes carrying >= 1 % of the tip compliance; max |omega| = programmed speed x the machine's
  largest spindle override, the CSS clamp for G96). ``options.write`` stores them in ``physics.resolved``
  (cnctwin.project.write_resolved_physics), which a run uses;
* the lattice rule (``lattice_mm <= min(ae) / 4`` on every cutting move, or a refine box covering the moves that
  break it) from sanity's engagement estimate;
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


def _lattice(ctx, op_index: int, report) -> dict:
    lattice = float(getattr(ctx.project.physics, "lattice_m", 0.0) or 0.0)
    refine = list(getattr(ctx.project.physics, "refine", None) or [])
    eng = getattr(report, "engagement", None)
    ae = np.asarray(getattr(eng, "ae_m", np.zeros(0)), float) if eng is not None else np.zeros(0)
    cut = np.asarray(getattr(eng, "cut", np.zeros(0)), bool) if eng is not None else np.zeros(0, bool)
    valid = ae[cut & np.isfinite(ae) & (ae > 0.0)] if ae.size and cut.size == ae.size else np.zeros(0)
    if valid.size == 0:
        return {"lattice_m": lattice, "min_ae_m": None, "ok": True, "refine": len(refine),
                "note": "no cutting move with an evaluated radial engagement"}
    min_ae = float(np.min(valid))
    breaking = int(np.count_nonzero(valid < 4.0 * lattice))
    finest = min([float(getattr(r, "lattice_m", lattice) or lattice) for r in refine] + [lattice])
    ok = breaking == 0 or (refine and finest <= min_ae / 4.0)
    return {"lattice_m": lattice, "min_ae_m": min_ae, "required_m": min_ae / 4.0, "moves_breaking": breaking,
            "refine": len(refine), "finest_refine_m": finest if refine else None, "ok": bool(ok),
            "note": "lattice_mm <= min(ae) / 4 on every cutting move (architecture 5.4)" if ok else
            f"{breaking} cutting moves have ae below 4 x lattice: refine the lattice to {min_ae / 4.0 * 1e3:.4f} mm or "
            f"add a refine box over the finishing passes"}


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
            entries.append({"tool": seg["tool"], "path": seg["path"], "l0_ticks": int(l0),
                            "max_rpm": omega * 60.0 / (2.0 * math.pi), "f_max_Hz": f_max, "teeth": teeth,
                            "rule": rule, "css": bool(seg["css"]), "elements": elements, "modes": n_modes,
                            "cpu_s_per_s_in_cut": cpu_per_s,
                            "fast_all_bytes_per_s": _fast_bytes(fast, l0, ctx.tick_s)})
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
            warnings.append(f"{name}: {lattice['note']}")
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


__all__ = ["check_project", "stride", "ring_l0_ticks", "dry_run_operation", "planner_cycle_time", "tool_f_max_Hz"]
