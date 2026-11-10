"""The edge temperatures follow the heat history in run time, not the macro step (docs/models/thermal.md 2.2;
docs/convergence.md 9 item 8): the periodic heating of a rotating tool's element (one pass per revolution) is
reconstructed from running sums of its passes, so the entry temperature cutting reads and the in-cut temperatures wear
reads are the same at the same instant whether the macro step is shorter or longer than a revolution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.db.simap import SIMap
from cnctwin.thermal import props
from cnctwin.thermal.partition import ContactHeat
from cnctwin.thermal.schema import ThermalSettings
from cnctwin.thermal.toolmodel import ToolEnvironment, ToolThermal, tool_spec

import fixtures as F

RPM = 10000.0
OMEGA = RPM * 2.0 * math.pi / 60.0
PERIOD = 2.0 * math.pi / OMEGA          # 6 ms
T_PASS = 0.43e-3                        # in cut per pass (a side step of 0.05 D)
RAKE_W, FLANK_W = 1.5, 0.4              # tool heat rate of an element while in cut


def make_tool():
    spec = tool_spec({"stickout_m": 0.035}, SIMap(F.TOOL_RAW), SIMap(F.HOLDER_RAW))
    mt = props.tool_material_table_by_id(spec.tool_material_id)
    mh = props.material_table_by_id(spec.holder_material_id)
    return ToolThermal(0, spec, mt, mh, 1e-3, 293.15, 0.0)


def step_data(t0: float, t1: float, nt: int, ne: int, engaged_ne: int = 3):
    """In-cut time and passes of every element in [t0, t1): tooth j passes at j T / nt + k T for T_PASS; the first
    ``engaged_ne`` elements of each tooth cut (the depth of cut), the rest stay idle."""
    in_cut = np.zeros((nt, ne))
    entries = np.zeros((nt, ne))
    for j in range(nt):
        phase = j * PERIOD / nt + 0.137e-3
        k0 = math.floor((t0 - phase - T_PASS) / PERIOD)
        k1 = math.ceil((t1 - phase) / PERIOD)
        for k in range(k0, k1 + 1):
            start = phase + k * PERIOD
            overlap = max(0.0, min(t1, start + T_PASS) - max(t0, start))
            in_cut[j, :engaged_ne] += overlap
            if t0 <= start < t1:
                entries[j, :engaged_ne] += 1.0
    return in_cut, entries


def run(dt: float, until: float, field: bool, sample_every: float):
    tool = make_tool()
    edges = F.edge_model()
    nt, ne = edges.elem_z_m.shape
    st = ThermalSettings()
    if field:
        assert tool.enable_field(st, edges, every_n_steps=st.tool_field_every_n_macro,
                                 interval_s=st.tool_field_interval_s)
    env = ToolEnvironment(in_spindle=True, omega_rad_s=OMEGA, T_air_K=293.15, T_end_K=293.15)
    samples = {}
    n = int(round(until / dt))
    every = int(round(sample_every / dt))
    history = []
    for i in range(n):
        t0, t1 = i * dt, (i + 1) * dt
        in_cut, entries = step_data(t0, t1, nt, ne)
        heat = ContactHeat(0, 0, 0)
        heat.tool_rake_elem_J = RAKE_W * in_cut
        heat.tool_flank_elem_J = FLANK_W * in_cut
        heat.h_mean_m = np.where(in_cut > 0.0, 4e-5, 0.0)
        heat.compression = np.full((nt, ne), 2.5)
        heat.flank_length_m = np.full((nt, ne), 6.4e-6)
        if field:
            tool.deposit_elements(edges, heat, None)
        else:
            tool.deposit(edges.elem_z_m, heat.tool_rake_elem_J + heat.tool_flank_elem_J)
        tool.step(dt, env, 1e-9)
        out = tool.edge_heat(edges, heat, in_cut, entries, None, dt, st, omega_rad_s=OMEGA)
        history.append(float(out.rake_temp_mean_K[0, 0]))
        if (i + 1) % every == 0:
            samples[round(t1, 9)] = out
    return samples, history


@pytest.mark.parametrize("field", [False, True])
def test_edge_temperatures_do_not_depend_on_the_macro_step(field):
    """The local rises at the entry (what cutting reads) and in the cut (what wear reads) agree within 2 % for macro
    steps of 5, 10 and 20 ms (a revolution is 6 ms; the old reconstruction took the period as the step over its
    entries and restarted an element whose step held no pass: 85, 55 and 37 K at the same instant)."""
    sample = 0.1 if field else 0.02
    runs = {dt: run(dt, 0.4, field, sample) for dt in (0.005, 0.01, 0.02)}
    times = sorted(set(runs[0.02][0]) & set(runs[0.005][0]) & set(runs[0.01][0]))
    times = [t for t in times if t >= 0.2]
    assert times
    for t in times:
        ref = runs[0.005][0][t]
        body = ref.body_temp_at_elem_K[:, :3]
        rise_entry = ref.rake_temp_mean_K[:, :3] - body
        rise_flank = ref.flank_temp_q_K[:, :3, :] - body[:, :, None]
        assert np.all(rise_flank > 1.0)
        for dt in (0.01, 0.02):
            out = runs[dt][0][t]
            b = out.body_temp_at_elem_K[:, :3]
            assert np.allclose(out.rake_temp_mean_K[:, :3] - b, rise_entry, rtol=0.02, atol=0.05)
            assert np.allclose(out.flank_temp_q_K[:, :3, :] - b[:, :, None], rise_flank, rtol=0.02, atol=0.05)
            assert np.allclose(b, body, atol=0.05 * float(np.max(body - 293.15)) + 1e-6)


def test_no_alternation_when_the_step_is_shorter_than_a_revolution():
    """At 5 ms steps about one step in six holds no pass of a given element; its entry temperature stays on the curve
    of its heat history instead of jumping to the body temperature and back (old: by its whole rise, 1.1 K here; with
    the cutting-region field 7-10 K). With the field it changes in steps only where the field is solved (every 100 ms
    of run time, the same instants at every macro step)."""
    _, history = run(0.005, 0.3, False, 0.1)
    tail = np.asarray(history[20:])
    jumps = np.diff(tail)
    assert np.max(np.abs(jumps)) < 0.05 * (np.max(tail) - 293.15)
    _, history = run(0.005, 0.4, True, 0.1)
    h = np.asarray(history)
    change = np.abs(np.diff(h))
    step = np.arange(1, h.size)
    at_solve = (step % 20 == 19) | (step % 20 == 0)       # the steps ending at, and just after, a field solve
    assert np.max(change[~at_solve][20:]) < 0.02 * (h[-1] - 293.15)


def test_field_solves_fall_on_the_same_run_times():
    tool = make_tool()
    edges = F.edge_model()
    st = ThermalSettings()
    tool.enable_field(st, edges, interval_s=0.1)
    env = ToolEnvironment(in_spindle=True, omega_rad_s=OMEGA)
    solved = []
    for i in range(80):
        tool.deposit(edges.elem_z_m, np.full(edges.elem_z_m.shape, 1e-3))
        out = tool.step(0.005, env, 1e-9)
        if out["source_J"] > 0.0:
            solved.append(round(tool.time_s, 9))
    assert solved == [0.1, 0.2, 0.3, 0.4]


def test_a_continuous_field_element_enters_at_the_field_less_its_own_rise():
    """A continuous element's field strip mean is its in-cut temperature: it holds the steady rise of the element's
    own contact heat, which mechanics' stationary-source partitions add again. Its entry temperature (what cutting's
    rows read as the tool's background) is the strip mean less that rise (2026-11-10; before, the strip mean itself:
    the entry rose with the in-cut temperature, a double count). Without the field the entry is the body
    temperature; with it the entry stays near that background while the in-cut rise is many times larger."""
    def continuous(field: bool, dt=0.01, until=0.4):
        tool = make_tool()
        edges = F.edge_model()
        nt, ne = edges.elem_z_m.shape
        st = ThermalSettings()
        if field:
            assert tool.enable_field(st, edges, every_n_steps=st.tool_field_every_n_macro,
                                     interval_s=st.tool_field_interval_s)
        env = ToolEnvironment(in_spindle=True, omega_rad_s=OMEGA, T_air_K=293.15, T_end_K=293.15)
        out = None
        for i in range(int(round(until / dt))):
            in_cut = np.zeros((nt, ne))
            in_cut[:, :3] = dt                      # in the cut the whole time: one entry, then continuous
            entries = np.zeros((nt, ne))
            if i == 0:
                entries[:, :3] = 1.0
            heat = ContactHeat(0, 0, 0)
            heat.tool_rake_elem_J = 0.25 * RAKE_W * in_cut
            heat.tool_flank_elem_J = 0.25 * FLANK_W * in_cut
            heat.h_mean_m = np.where(in_cut > 0.0, 4e-5, 0.0)
            heat.compression = np.full((nt, ne), 2.5)
            heat.flank_length_m = np.full((nt, ne), 6.4e-6)
            if field:
                tool.deposit_elements(edges, heat, None)
            else:
                tool.deposit(edges.elem_z_m, heat.tool_rake_elem_J + heat.tool_flank_elem_J)
            tool.step(dt, env, 1e-9)
            out = tool.edge_heat(edges, heat, in_cut, entries, None, dt, st, omega_rad_s=OMEGA)
        return out

    for field in (False, True):
        out = continuous(field)
        body = out.body_temp_at_elem_K[:, :3]
        in_cut_rise = out.rake_contact_mean_K[:, :3] - body
        entry_rise = out.rake_temp_mean_K[:, :3] - body
        flank_in = np.einsum("ijq,q->ij", out.flank_temp_q_K[:, :3, :], out.q_weights) - body
        flank_entry = out.flank_entry_K[:, :3] - body
        assert np.all(in_cut_rise > 5.0) and np.all(flank_in > 5.0)
        if not field:
            assert np.allclose(entry_rise, 0.0, atol=1e-9) and np.allclose(flank_entry, 0.0, atol=1e-9)
        else:
            assert np.all(np.abs(entry_rise) < 0.25 * in_cut_rise)
            assert np.all(np.abs(flank_entry) < 0.25 * flank_in)
