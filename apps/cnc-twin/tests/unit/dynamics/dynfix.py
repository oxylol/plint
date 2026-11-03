"""Small fixtures for the dynamics tests: hand-built systems, a fake run context, bus frames."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from cnctwin.core.bus import Bus, MotionFrame, PathFrame, RemovalFrame
from cnctwin.core.channels import ChannelRegistry
from cnctwin.core.state import Ledger, SimState
from cnctwin.core.trace import TraceLog
from cnctwin.dynamics import modal as MD
from cnctwin.dynamics.abi import KIND_TOOL, KIND_WORK, WorkGrid
from cnctwin.dynamics.modal import Mode

TWO_PI = 2.0 * math.pi


def tool_mode(freq_Hz, zeta, mass_kg, direction, nst=1, z=None, rot=None, kind=KIND_TOOL, key=None, base=None):
    """A mode with the same translation at every station (and optional rotation)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    st = np.zeros((nst, 6))
    st[:, :3] = d / math.sqrt(mass_kg)
    if rot is not None:
        st[:, 3:] = np.asarray(rot, float)
    m = Mode(kind, TWO_PI * freq_Hz, zeta, "fixture", f"fixture {freq_Hz} Hz", mass_kg=mass_kg, stations=st,
             at_unbalance=st[0].copy(), key=key or ("fixture", freq_Hz, tuple(direction)))
    if base is not None:
        m.base = np.asarray(base, float)
    return m


def cantilever_mode(freq_Hz, zeta, z_st, length, mass_eff, axis=0, key=None):
    """A bending-like mode: translation (z / L)^2 shape, rotation its slope (r_y for x bending, -r_x for y)."""
    z = np.asarray(z_st, float)
    shape = np.zeros((len(z), 6))
    u = (z / length) ** 2 / math.sqrt(mass_eff)
    du = 2.0 * z / length ** 2 / math.sqrt(mass_eff)
    if axis == 0:
        shape[:, 0] = u
        shape[:, 4] = du
    else:
        shape[:, 1] = u
        shape[:, 3] = -du
    return Mode(KIND_TOOL, TWO_PI * freq_Hz, zeta, "fixture", f"bend {freq_Hz}", stations=shape,
                at_unbalance=shape[-1].copy(), key=key or ("bend", freq_Hz, axis))


def empty_grid():
    return MD.work_modes(None, 0, np.eye(3), None)[1]


def work_grid(n_modes=1, freqs=(800.0,), zeta=0.02, shape_scale=1.0 / math.sqrt(2.0), cell=0.01, dims=(3, 3, 3)):
    """Workpiece modes on a small grid whose shapes vary linearly in x (so interpolation is testable)."""
    nx, ny, nz = dims
    shape = np.zeros((n_modes, nx, ny, nz, 3))
    for k in range(n_modes):
        for i in range(nx):
            shape[k, i, :, :, k % 3] = shape_scale * (1.0 + 0.5 * i)
    comp = np.zeros((nx, ny, nz, 3, 3))
    for i in range(nx):
        comp[i, :, :] = np.eye(3) * 1e-9 * (1.0 + i)
    grid = WorkGrid(shape, comp, np.zeros(3), np.array([cell]), np.eye(3), np.zeros((3, 3)))
    modes = [Mode(KIND_WORK, TWO_PI * freqs[k % len(freqs)], zeta, "work", f"work {k}", work_row=k, key=("work", 0, k))
             for k in range(n_modes)]
    return modes, grid


def system(modes, *, nst=1, ref=0, l0_ticks=10, ipo_ticks=1000, tick_s=1e-6, n_ticks=10, capacity=4000,
           rotating=True, unbalance=None, K_c=None, residual=None, grid=None, point_S=None, R_S_T0=None):
    sys, layout = MD.build_system(modes, nst=nst, ref_station=ref, residual=residual, grid=grid or empty_grid(),
                                  point_S=np.zeros(3) if point_S is None else np.asarray(point_S, float),
                                  l0_ticks=l0_ticks, ipo_ticks=ipo_ticks, tick_s=tick_s, n_ticks=n_ticks,
                                  capacity=capacity, rotating=rotating, unbalance_T=unbalance, K_c_T0=K_c,
                                  R_S_T0=np.eye(3) if R_S_T0 is None else R_S_T0)
    return sys, layout


def clone(nt):
    """Deep copy of a (nested) NamedTuple of arrays."""
    return copy.deepcopy(nt)


def assert_close_tuple(a, b, rtol=1e-12, atol=1e-300, path="sys"):
    for name, va, vb in zip(a._fields, a, b):
        if isinstance(va, tuple):
            assert_close_tuple(va, vb, rtol, atol, f"{path}.{name}")
        else:
            np.testing.assert_allclose(va, vb, rtol=rtol, atol=atol, err_msg=f"{path}.{name}")


# ------------------------------------------------------------------------------- run context and bus
@dataclass
class FakeTimeline:
    def line_ref(self, index):
        return None


@dataclass
class FakeCtx:
    project: object = None
    db: object = None
    run_dir: object = None
    seed: int = 1
    tick_s: float = 1e-6
    ipo_ticks: int = 1000
    macro_ipo: int = 10
    channels: ChannelRegistry = field(default_factory=ChannelRegistry)
    trace: TraceLog = field(default_factory=TraceLog)
    recorder: object = None
    timeline: object = None

    @property
    def macro_s(self):
        return self.tick_s * self.ipo_ticks * self.macro_ipo

    @property
    def ipo_s(self):
        return self.tick_s * self.ipo_ticks

    def finding(self, state, domain, category, severity, message, tick, path=0, quantities=None, parents=None):
        from cnctwin.core.domain import RunContext
        return RunContext.finding(self, state, domain, category, severity, message, tick, path, quantities, parents)


def machine_doc(frame_modes=(), spindle_modes=(), rotary_modes=(), max_rpm=12000.0):
    """A minimal machine SI view (dict-based) with kinematics X, Y (workpiece) and Z (tool)."""
    return {
        "structure": {"modes": [dict(m) for m in frame_modes], "table_modes": [], "rotary_modes": [dict(m) for m in rotary_modes]},
        "spindles": [{"id": "main", "max_rad_s": max_rpm * TWO_PI / 60.0, "frf_modes": [dict(m) for m in spindle_modes]}],
        "kinematics": {"workpiece_chain": [{"link": "y", "axis": "Y", "direction": [0, -1, 0]},
                                           {"link": "x", "axis": "X", "direction": [-1, 0, 0]}],
                       "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]}]},
        "axes": {"X": {"type": "linear"}, "Y": {"type": "linear"}, "Z": {"type": "linear"}},
    }


def path_frame(tick0, n_ticks, ipo_ticks, tick_s, omega, slot=0, theta0=0.0, axis_acc=(0.0, 0.0, 0.0), feed=(0.0, 0.0, 0.0)):
    ticks = tick0 + (np.arange(n_ticks, dtype=np.int64) + 1) * ipo_ticks
    t = (ticks - 0) * tick_s
    angle = theta0 + omega * t
    dt = ipo_ticks * tick_s
    a = np.asarray(axis_acc, float)
    pos = 0.5 * a[None, :] * (t[:, None] ** 2)
    f = PathFrame(path=0, tick0=tick0, ipo_ticks=ipo_ticks, tick=ticks, block_index=np.zeros(n_ticks, np.int32),
                  tip_S_m=np.zeros((n_ticks, 3)), tip_M_m=np.zeros((n_ticks, 3)), axis_T_M=np.tile([0, 0, 1.0], (n_ticks, 1)),
                  axis_names=("X", "Y", "Z"), axis_actual=pos, axis_cmd=pos.copy(), following_error=np.zeros((n_ticks, 3)),
                  spindle_angle_rad=np.mod(angle, TWO_PI), spindle_rev=np.floor(angle / TWO_PI).astype(np.int64),
                  spindle_omega_rad_s=np.full(n_ticks, omega), feed_vel_M_m_s=np.tile(np.asarray(feed, float), (n_ticks, 1)),
                  rapid=np.zeros(n_ticks, bool), tool_slot=slot, spindle_id="main", workpiece=0)
    theta_end = float(angle[-1])
    c, s = math.cos(theta_end), math.sin(theta_end)
    pose = np.eye(4)
    pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    f.pose_M_T = pose.copy()
    f.pose_S_T = pose.copy()
    return f


def bus_with(frame, holding=None, contact=None):
    bus = Bus()
    bus.motion = MotionFrame(time_s=0.0, tick=int(frame.tick0), paths=[frame])
    if holding is not None:
        bus.holding.tools[int(frame.tool_slot)] = holding
    if contact is not None:
        bus.removal = RemovalFrame(contacts=[contact])
    return bus


def sim_state():
    s = SimState()
    s.ledger = Ledger()
    return s
