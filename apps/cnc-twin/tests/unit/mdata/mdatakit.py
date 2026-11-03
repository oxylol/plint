"""Shared helpers of the mdata tests: a project on the real database, the synthetic run, and measured
machine data generated from the run with known alignment and known machine values."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = HERE.parents[1] / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

EPOCH_NS = 1_793_700_930_000_000_000          # 2026-11-03T10:15:30Z
OMEGA_MAX = 12000 * 2 * np.pi / 60             # haas-vf2ss main spindle max_rpm 12000
PALMGREN = 1.6667                              # production-energy palmgren_speed_exponent

PROJECT = """\
format: cnctwin-project
version: 1
name: "Calibration test pocket"
seed: 7
machine: haas-vf2ss
material:
  id: al6061-t6
stock:
  kind: block
  size_mm: [40, 30, 15]
  form: rolled_plate
tools:
  list:
    - {number: 1, tool: helical-59482-mhsv-020-41000, holder: haimer-40-720-16, stickout_mm: 35,
       length_offset_mm: 120.5, runout_um: 5}
coolant: {fluid: emulsion-8pct, concentration_pct: 8}
operations:
  - name: OP10
    program: op10.nc
    setup:
      workholding:
        - {device: lang-45600, pose: {translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}}
      stock_pose: {translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}
      work_offsets_mm: {G54: [-381.0, -203.2, -406.4]}
"""


def write_project(directory: Path) -> Path:
    from make_synthetic_run import program_text

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "project.yaml").write_text(PROJECT, encoding="utf-8")
    (directory / "op10.nc").write_text(program_text(), encoding="utf-8")
    return directory / "project.yaml"


def make_run(directory: Path, seed: int = 7) -> Path:
    from make_synthetic_run import make_synthetic_run

    make_synthetic_run(directory, seed=seed, media=False)
    return directory


def measured_spindle(run_dir: Path, *, offset_s: float, eta: float, idle_W: float, kc_scale: float,
                     dt_s: float = 0.01, before_s: float = 1.234, after_s: float = 1.0, noise_W: float = 15.0,
                     seed: int = 1) -> dict:
    """Measured electrical spindle power and polled line markers of the synthetic run, with
    t_sim = offset_s + t_meas (the log starts ``before_s`` before the program)."""
    from cnctwin.mdata.simrun import SimulatedRun

    run = SimulatedRun(run_dir)
    sim = run.to_log()
    end = run.cycle_time_s()
    t_meas = np.arange(0.0, before_s + end + after_s, dt_s)
    t_sim = offset_s + t_meas
    shaft = np.nan_to_num(sim.channels["spindle.shaft_power_W"].at(t_sim))
    omega = np.nan_to_num(sim.channels["spindle.speed_rad_s"].at(t_sim))
    rng = np.random.default_rng(seed)
    power = kc_scale * shaft / eta + idle_W * (omega / OMEGA_MAX) ** PALMGREN + rng.normal(0.0, noise_W, len(t_meas))
    line = sim.channels["program.line"].at(t_sim)
    speed = omega * (1.0 + rng.normal(0.0, 1e-4, len(t_meas)))
    return {"t_meas": t_meas, "t_sim": t_sim, "power_W": power, "line": line, "speed_rad_s": speed,
            "sim": sim, "run": run}


def mtconnect_xml(data: dict, *, epoch_ns: int = EPOCH_NS, first_sequence: int = 1) -> bytes:
    """An MTConnectStreams /sample document of measured spindle power (Wattage), speed and line numbers."""
    from cnctwin.mdata.log import format_time_ns

    rows = []
    sequence = first_sequence
    last_line = None
    for t, power, speed, line in zip(data["t_meas"], data["power_W"], data["speed_rad_s"], data["line"]):
        stamp = format_time_ns(epoch_ns + int(round(t * 1e9)))
        rows.append(("s1", f'<Wattage dataItemId="S1power" timestamp="{stamp}" sequence="{sequence}">{power:.3f}</Wattage>'))
        sequence += 1
        rpm = speed * 60.0 / (2 * np.pi)
        rows.append(("s1", f'<RotaryVelocity dataItemId="S1speed" timestamp="{stamp}" sequence="{sequence}" '
                           f'subType="ACTUAL">{rpm:.2f}</RotaryVelocity>'))
        sequence += 1
        if np.isfinite(line) and line != last_line:
            rows.append(("path", f'<LineNumber dataItemId="ln" timestamp="{stamp}" sequence="{sequence}" '
                                 f'subType="ABSOLUTE">{int(line)}</LineNumber>'))
            sequence += 1
            last_line = line
    samples = "".join(r for c, r in rows if c == "s1")
    events = "".join(r for c, r in rows if c == "path")
    last = sequence - 1
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<MTConnectStreams xmlns="urn:mtconnect.org:MTConnectStreams:2.0">'
            f'<Header creationTime="2026-11-03T10:30:00Z" sender="test" instanceId="1762160000" version="2.0.0" '
            f'bufferSize="1048576" nextSequence="{last + 1}" firstSequence="{first_sequence}" lastSequence="{last}"/>'
            f'<Streams><DeviceStream name="VF2SS" uuid="vf2ss-1">'
            f'<ComponentStream component="Rotary" name="S1" componentId="s1"><Samples>{samples}</Samples>'
            f'</ComponentStream><ComponentStream component="Path" name="path" componentId="p1"><Events>{events}'
            f'</Events></ComponentStream></DeviceStream></Streams></MTConnectStreams>\n').encode("utf-8")
