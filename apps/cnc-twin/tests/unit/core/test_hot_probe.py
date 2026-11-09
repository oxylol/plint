"""cnctwin.core.hot_probe: the stylus profile of a probe in a path's spindle and the probe-only contact of the hot
loop (docs/models/hotloop.md 9a)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core import hot_abi as H
from cnctwin.core.hot_probe import ProbeContacts, probe_entry, stylus_dimensions, stylus_profile
from cnctwin.removal.reference import profile_radius


def test_stylus_profile_is_the_ball_on_its_stem_below_motions_tip():
    ball, length, ratio, z0 = 6e-3, 50e-3, 0.75, -0.125
    prof = stylus_profile(ball, length, ratio, z0)
    R, rs = 3e-3, 2.25e-3
    assert prof.info[0] == pytest.approx(z0)                     # PI_ZMIN: the ball's lowest point
    assert profile_radius(prof, z0 + 1e-9) == pytest.approx(math.sqrt(2 * R * 1e-9), rel=1e-3)
    assert profile_radius(prof, z0 + R) == pytest.approx(R, rel=1e-12)
    z_stem = R + math.sqrt(R * R - rs * rs)
    assert profile_radius(prof, z0 + 0.5 * (R + z_stem)) == pytest.approx(
        math.sqrt(R * R - (0.5 * (z_stem - R)) ** 2), rel=1e-9)
    assert profile_radius(prof, z0 + 0.02) == pytest.approx(rs, rel=1e-12)
    assert prof.info[1] == pytest.approx(z0 + R + length)        # PI_ZMAX: the stem's top
    assert prof.info[2] == pytest.approx(R)                      # PI_RMAX


class _P:
    """A minimal resolved project: the tool list and the machine's probes (SI keys as SIMap gives them)."""

    def __init__(self):
        self.tools = {"list": [{"number": 1, "tool": "x"},
                               {"number": 20, "kind": "probe", "probe": "spindle_probe", "length_offset_m": 0.125}]}
        self.machine = {"probes": {"spindle_probe": {"stylus_length_m": 0.05},
                                   "spindle_probe_extra": {"stylus_ball_m": 0.006}}}


def test_probe_entry_and_contact():
    project = _P()
    assert probe_entry(project, 20) == ("spindle_probe", 0.125)
    assert probe_entry(project, 1) is None and probe_entry(project, None) is None
    assert stylus_dimensions(project.machine, "spindle_probe") == (0.006, 0.05)
    probes = ProbeContacts(project)
    found = probes.find(20, 0.0)
    name, z0, ball, length = found
    assert name == "spindle_probe" and z0 == pytest.approx(-0.125) and (ball, length) == (0.006, 0.05)
    pad = H.empty_contact(3)
    ct = probes.contact(pad, H, 0, 0, found)
    assert int(ct.ipar[H.CI_PROBE]) == 1 and int(ct.ipar[H.CI_ACTIVE]) == 0 and int(ct.ipar[H.CI_PATH]) == 0
    assert ct.work is not pad.work and ct.ipar is not pad.ipar and ct.probe_out is not pad.probe_out
    assert ct.probe.info[0] == pytest.approx(-0.125)
    assert np.all(pad.ipar == 0)


class _Resolved:
    """The resolved project's view: the loader moves ``kind: probe`` entries to ``tools.probes``, each with its
    definition resolved against the machine (SI keys, ``name`` the machine probe it extends)."""

    def __init__(self):
        self.tools = {"list": [{"number": 1, "tool": "x"}],
                      "probes": [{"number": 20, "kind": "probe", "length_offset_m": 0.125, "list_index": 1,
                                  "probe": {"name": "spindle_probe", "stylus_ball_m": 0.004,
                                            "stylus_length_m": 0.03}}]}
        self.machine = {"probes": {"spindle_probe": {"stylus_ball_m": 0.006, "stylus_length_m": 0.05}}}


def test_a_resolved_projects_probes_come_from_tools_probes_with_their_own_stylus():
    """The run loop sees the resolved project: its probes are in ``tools.probes`` (before this the hot loop only
    looked in ``tools.list``, found no probe and the G31 move ran through the block)."""
    project = _Resolved()
    assert probe_entry(project, 20) == ("spindle_probe", 0.125)
    assert probe_entry(project, 1) is None
    name, z0, ball, length = ProbeContacts(project).find(20, 0.01)
    assert name == "spindle_probe" and z0 == pytest.approx(0.01 - 0.125)
    assert (ball, length) == (0.004, 0.03)      # the entry's definition wins over the machine's probe


def test_the_stylus_sweep_finds_the_top_of_a_block():
    """removal's sweep_contact with the stylus: a ball 125 mm below motion's tip moving down onto a block's top
    face touches it where the ball's lowest point reaches the top (within the field's resolution)."""
    from cnctwin.removal.k_query import sweep_contact
    from cnctwin.removal.stock import block_field
    f = block_field([0.02, 0.02, 0.01], 2.5e-4)
    prof = stylus_profile(6e-3, 50e-3, 0.75, -0.125)
    w = np.array([0.0, 0.0, 1.0])
    x0 = np.array([1.0, 0.0, 0.0])
    c0 = np.array([0.01, 0.01, 0.01 + 0.125 + 0.002])           # ball bottom 2 mm above the top
    c1 = np.array([0.01, 0.01, 0.01 + 0.125 - 0.002])           # 2 mm below
    out = np.zeros(8)
    lam = sweep_contact(f, prof, c0, c1, w, x0, 8, 32, 1e-8, out)
    assert 0.0 <= lam <= 1.0
    z_ball_bottom = c0[2] + lam * (c1[2] - c0[2]) - 0.125
    assert z_ball_bottom == pytest.approx(0.01, abs=1e-7)
