"""Tool-holder-spindle model: a clamped tool against stepped-cantilever theory, a boring bar against Timoshenko,
the coupled end mill (canonical x/y pairs, spindle coupling, residual flexibility, RCSA fit), pull-out and
breakage, and the static deflection and section stress helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.structure import structfix as fx
from tests.unit.tooling.fixtures import HOLDER, TOOLS
from cnctwin.db.simap import SIMap
from cnctwin.structure import beam as B
from cnctwin.structure import params as P
from cnctwin.structure import toolmodel as TM
from cnctwin.structure.materials import tool_substrate
from cnctwin.structure.schema import settings
from cnctwin.tooling import build_tool

ST = settings({})


@pytest.fixture(scope="module")
def endmill():
    return build_tool(TOOLS["square_endmill"], HOLDER, {"number": 1, "stickout_mm": 35})


@pytest.fixture(scope="module")
def coupled(endmill):
    return TM.tool_modes(endmill, SIMap(fx.MACHINE), ST)


def test_clamped_tool_against_stepped_cantilever(monkeypatch):
    """Holder-less tool clamped rigidly (stiff contact, rigid spindle): tip compliance = int z^2 / EI + int 1 / kGA
    over the fluted section and the shank up to the clamp (exact for the Timoshenko elements)."""
    monkeypatch.setattr(P, "contact_stiffness_N_per_m3", lambda t: 1e17)
    geo = build_tool(TOOLS["square_endmill"], None, {"number": 1, "stickout_mm": 35})
    res = TM.tool_modes(geo, None, ST)
    mat = tool_substrate("carbide-submicron-10co", "carbide")
    G = B.shear_modulus(mat.E_Pa, mat.nu)
    fl = geo.props.fluted
    I_f = 0.5 * (fl.I_min_m4 + fl.I_max_m4)
    zf, zn, d = geo.layout["z_flute_end"], geo.layout["stickout"], geo.layout["d_shank"]
    I_s = math.pi * d ** 4 / 64
    A_s = math.pi * d * d / 4
    k = B.cowper_circle(mat.nu)
    bend = zf ** 3 / (3 * mat.E_Pa * I_f) + (zn ** 3 - zf ** 3) / (3 * mat.E_Pa * I_s)
    shear = zf / (k * G * fl.area_m2) + (zn - zf) / (k * G * A_s)
    # the square end mill's tip region: the envelope radius equals R, so the fluted section is uniform
    assert res.tip_compliance_m_per_N[0, 0] == pytest.approx(bend + shear, rel=2e-3)
    assert res.tip_compliance_m_per_N[0, 0] == pytest.approx(res.tip_compliance_m_per_N[1, 1], rel=1e-9)


def test_boring_bar_against_timoshenko(monkeypatch):
    monkeypatch.setattr(P, "contact_stiffness_N_per_m3", lambda t: 1e17)
    geo = build_tool(TOOLS["boring_bar"], None, {"number": 1})
    res = TM.tool_modes(geo, None, ST)
    steel = P.p(P.TOOL, "holder_youngs_modulus_Pa"), P.p(P.TOOL, "holder_poisson"), P.p(P.TOOL, "holder_density_kg_m3")
    E, nu, rho = steel
    d, L = 0.02, 0.08
    I = math.pi * d ** 4 / 64
    A = math.pi * d * d / 4
    G = B.shear_modulus(E, nu)
    k = B.cowper_circle(nu)
    assert res.modes.station_z_m.tolist() == [0.0]
    assert res.tip_compliance_m_per_N[0, 0] == pytest.approx(L ** 3 / (3 * E * I) + L / (k * G * A), rel=2e-3)
    f_eb = 1.875104069 ** 2 / (2 * math.pi * L * L) * math.sqrt(E * I / (rho * A))
    assert 0.95 * f_eb < res.modes.freq_Hz[0] < f_eb      # L / d = 4: shear and rotary inertia lower it 2-4 %


def test_coupled_end_mill_modes(endmill, coupled):
    tm = coupled.modes
    f = tm.freq_Hz
    assert f[0] == pytest.approx(f[1], rel=1e-8)
    rigid = TM.tool_modes(endmill, None, ST)
    # the spindle flange mode (950 Hz) couples below both it and the tool's own first mode
    assert f[0] < 950.0 and f[0] < rigid.modes.freq_Hz[0]
    assert coupled.tip_compliance_m_per_N[0, 0] > rigid.tip_compliance_m_per_N[0, 0]
    # canonical pair: mode 0 translates along x, mode 1 along y at every station
    s0, s1 = tm.shape[0, :, :3], tm.shape[1, :, :3]
    assert np.max(np.abs(s0[:, 1])) < 1e-9 * np.max(np.abs(s0[:, 0]))
    assert np.max(np.abs(s1[:, 0])) < 1e-9 * np.max(np.abs(s1[:, 1]))
    # stations: tip, flute end, holder nose, gauge line, ascending
    z = tm.station_z_m
    assert z[0] == 0.0 and np.all(np.diff(z) > 0)
    for zk in (0.022, 0.035, 0.115):
        assert np.min(np.abs(z - zk)) < 1e-12
    # damping: the spindle mode keeps most of its 4 %, tool modes get interface and material damping
    assert 0.02 < tm.zeta[0] < 0.04 and np.all(tm.zeta > 0.001)
    # residual flexibility: symmetric, positive semi-definite, and modes + residual = static compliance
    R = tm.static_residual_m_per_N
    assert np.allclose(R, R.T) and np.min(np.linalg.eigvalsh(R)) > -1e-12 * np.max(np.abs(R))
    lam = coupled.omega_rad_s ** 2
    phi = tm.shape[:, 0, :3]
    C_tip = (phi.T / lam) @ phi + R[:3, :3]
    assert np.allclose(C_tip, coupled.tip_compliance_m_per_N, rtol=1e-9, atol=1e-18)
    assert coupled.fit_error < 0.02
    assert np.all(np.diff(tm.sections_z_m) >= 0) and np.all(tm.sections_modulus_m3 > 0)


def test_pullout_and_breakage(endmill, coupled):
    longer = TM.tool_modes(endmill, SIMap(fx.MACHINE), ST, pullout_m=1e-3)
    assert longer.tip_compliance_m_per_N[0, 0] > coupled.tip_compliance_m_per_N[0, 0]
    assert longer.modes.station_z_m[np.argmin(np.abs(longer.modes.station_z_m - 0.036))] == pytest.approx(0.036)
    broken = TM.tool_modes(endmill, SIMap(fx.MACHINE), ST, length_loss_m=5e-3)
    assert broken.modes.station_z_m[0] == pytest.approx(5e-3)
    assert broken.modes.freq_Hz[2] > coupled.modes.freq_Hz[2]


def test_static_and_stress_helpers(endmill, coupled):
    F = np.array([100.0, 0.0, 0.0])
    u = TM.static_deflection(coupled, F, 0)
    assert u[0, 0] == pytest.approx(coupled.tip_compliance_m_per_N[0, 0] * 100.0, rel=1e-9)
    assert np.all(u[1:, 0] < u[0, 0])                 # the tip moves most
    out = TM.section_stresses(endmill.props.sections, [300.0, 400.0, -50.0], 0.005, torque_Nm=2.0)
    s = out[0]
    M = 500.0 * (s["z_m"] - 0.005)
    sec = sorted(endmill.props.sections, key=lambda x: x.z_m)[0]
    assert s["moment_Nm"] == pytest.approx(M)
    sigma = M / sec.Z_bending_m3 + 50.0 / sec.area_m2
    tau = 2.0 / sec.Z_polar_m3
    assert s["sigma_vm_Pa"] == pytest.approx(math.sqrt(sigma ** 2 + 3 * tau ** 2))
    g = TM.tip_frf(coupled, [0.0])
    assert abs(g[0]) == pytest.approx(coupled.tip_compliance_m_per_N[0, 0], rel=1e-9)


def test_turning_tool_orientation_and_turret_modes():
    geo = build_tool(TOOLS["turning_insert"], None, {"number": 1})
    machine = SIMap({"structure": {"turret_modes": [{"fn_Hz": 600, "zeta": 0.05, "k_N_um": 50, "direction": "x",
                                                     "src": "est", "note": "t"}]}})
    a = TM.tool_modes(geo, machine, ST, R_M_T0=np.eye(3))
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    b = TM.tool_modes(geo, machine, ST, R_M_T0=turned)
    # the turret mode is along M x: in T0 along x for the identity, along -y for the turned block
    m_a = int(np.argmin(np.abs(a.modes.freq_Hz - 600)))
    m_b = int(np.argmin(np.abs(b.modes.freq_Hz - 600)))
    sa, sb = a.modes.shape[m_a, 0, :3], b.modes.shape[m_b, 0, :3]
    assert abs(sa[0]) > 10 * abs(sa[1]) and abs(sb[1]) > 10 * abs(sb[0])
    # the shank bends more easily along z_T (width b) than ... equal for a square shank: compliances x and z
    C = a.tip_compliance_m_per_N
    assert C[0, 0] > 0 and C[1, 1] > 0 and C[2, 2] > 0
