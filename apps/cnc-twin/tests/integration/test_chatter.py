"""(e) Regenerative chatter of a 1-DOF system: a full slot (Al 6061-T6, D10 4FL, 10000 rpm, f_z 0.04 mm) on a VF-2SS
whose only flexible mode is a spindle mode in x (900 Hz, zeta 0.02, 5 N/um). The hot loop closes the loop
(cutting force -> modal dynamics -> deflection in the chip thickness -> surface memory -> next tooth); dynamics'
lobe diagram at the operating point (ZOA, variable pitch: cnctwin.dynamics.sld.limit_at_speed) gives the limit.
At 0.4 x the limit the vibration stays forced (small, chatter index ~ 0); at 2 x the limit it grows into chatter
at a frequency above the mode and off the tooth-passing harmonics.

Depths from the tool's stations (the stock top midway between two peripheral stations, docs/models/hotloop.md
"Field resolution"); the steady window excludes the entry and the exit transient (the slot end, where the last
tooth passes leave a free vibration that dynamics may flag)."""

import math

import numpy as np
import pytest

from cnctwin.core.drive import OperationDriver
from cnctwin.db import Database
from tests.integration import hotfix as HF

FN_HZ, ZETA, K_N_UM = 900.0, 0.02, 5.0
X1_MM = 20.0
STEADY_X_M = (2e-3, X1_MM * 1e-3 - 1e-3)


def _run(tmp, db, n_stations):
    probe = OperationDriver(HF.slot_project(tmp / "probe", 3.0), db=db, use_dynamics=False)
    probe.cutting.update_tool(probe.ctx, probe.state, 0, probe.state.bus_prev, force=True)
    ap_mm = HF.station_ap_mm(probe.state.cutting.tools[0].tool, n_stations)
    drv = OperationDriver(HF.chatter_project(tmp / "run", db, ap_mm, FN_HZ, ZETA, K_N_UM, x1_mm=X1_MM), db=db)
    rows = []

    def observe(d, rec):
        dyn = d.state.bus_next.dynamics
        xs = np.asarray(d.state.bus_next.motion.paths[0].tip_S_m)[:, 0]
        if dyn.contacts:
            c = dyn.contacts[0]
            rows.append({"t": rec.tick1 * 1e-6, "x0": float(xs.min()), "x1": float(xs.max()),
                         "in_cut": rec.in_cut_s, "pp": c.vibration_pp_m, "index": c.chatter_index,
                         "chatter": bool(c.chatter), "limit": c.depth_limit_m, "ap": c.depth_of_cut_m,
                         "freq": c.chatter_freq_Hz, "amp": c.chatter_amp_m})
    drv.observers.append(observe)
    res = HF.run_until_done(drv)
    # the slot between full engagement and the exit (a chattering tool leaves the cut part of each period: its
    # in-cut time per step is below the step length)
    steady = [r for r in rows if r["in_cut"] > 0.0 and STEADY_X_M[0] < r["x0"] and r["x1"] < STEADY_X_M[1]]
    for r in steady[::8]:
        print({k: (round(v, 9) if isinstance(v, float) else v) for k, v in r.items()})
    return drv, res, ap_mm * 1e-3, steady


@pytest.fixture(scope="module")
def db():
    return Database.load()


@pytest.fixture(scope="module")
def stable(tmp_path_factory, db):
    return _run(tmp_path_factory.mktemp("chatter_lo"), db, 3)


@pytest.fixture(scope="module")
def unstable(tmp_path_factory, db):
    return _run(tmp_path_factory.mktemp("chatter_hi"), db, 14)


def _limit(steady):
    lims = [r["limit"] for r in steady if math.isfinite(r["limit"]) and r["limit"] > 0.0]
    assert lims, "no lobe-diagram limit in the steady window"
    return float(np.median(lims))


def test_below_the_limit_stays_forced(stable):
    drv, res, ap, steady = stable
    assert res.finished and len(steady) >= 40
    lim = _limit(steady)
    print("ap", ap, "ZOA limit", lim)
    assert ap < 0.5 * lim
    assert not any(r["chatter"] for r in steady)
    assert max(r["index"] for r in steady) < 0.05
    assert max(r["pp"] for r in steady) < 5e-6


def test_above_the_limit_chatters(stable, unstable):
    drv, res, ap, steady = unstable
    assert res.finished and len(steady) >= 40
    lim = _limit(steady)
    print("ap", ap, "ZOA limit", lim)
    assert ap > 1.5 * lim
    # the regenerative vibration has grown within the first quarter of the steady window and stays
    late = steady[len(steady) // 4:]
    assert all(r["chatter"] for r in late)
    assert min(r["index"] for r in late) > 0.5
    pp_hi = float(np.median([r["pp"] for r in late]))
    pp_lo = float(np.median([r["pp"] for r in stable[3]]))
    print("vibration pp", pp_hi, "stable", pp_lo)
    assert pp_hi > 10.0 * pp_lo
    # chatter above the natural frequency (a 1-DOF regenerative system: Re G < 0) and off the tooth harmonics
    f = float(np.median([r["freq"] for r in late]))
    omega = abs(HF.OMEGA)
    f_tooth = 4 * omega / (2 * math.pi)
    print("chatter frequency", f, "tooth passing", f_tooth)
    assert FN_HZ < f < 1.5 * FN_HZ
    k = f / f_tooth
    assert abs(k - round(k)) > 0.1
