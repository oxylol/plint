"""Every wear kernel against its plain-Python reference (architecture 5.5), and one signature each."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core.rng import stream_key
from cnctwin.wear import abi as A
from cnctwin.wear import kernels as K
from cnctwin.wear import reference as R


def test_every_kernel_has_one_signature():
    names = ("usui_rate", "table_at", "worn_edge_radius", "wedge_tension", "wedge_volume_factor", "advance_elements",
             "element_outputs", "chip_trials")
    for name in names:
        assert len(getattr(K, name).signatures) == 1, name
    found = [n for n in dir(K) if hasattr(getattr(K, n), "signatures") and getattr(K, n).py_func.__module__ == K.__name__]
    assert sorted(found) == sorted(names)


def _clone(nt):
    return type(nt)(*(np.array(a, copy=True) for a in nt))


def _law(seed=0, branches=2):
    rng = np.random.default_rng(seed)
    law = A.new_law(branches, 1, 3, 2)
    law.fl_A[:] = [2e-12, 5e-9][:branches]
    law.fl_B[:] = [9000.0, 18000.0][:branches]
    law.fl_lo[:] = [-math.inf, 1100.0][:branches]
    law.rk_A[:] = [3e-12]
    law.rk_B[:] = [11000.0]
    law.rk_lo[:] = [-math.inf]
    law.hs_T[:] = [293.15, 1273.15, 1323.15]
    law.hs_Pa[:] = [1.6e10, 1.2e10, 4e9]
    law.hc_T[:] = [293.15, 1073.15]
    law.hc_Pa[:] = [3.2e10, 2.5e10]
    s = law.scal
    s[A.LS_ABR] = 2e-8
    s[A.LS_COAT_FACTOR] = 0.5
    s[A.LS_COAT_TMAX_K] = 1050.0
    s[A.LS_NOTCH_EXCESS] = 1.2
    s[A.LS_ROUNDING] = 1.0
    s[A.LS_BUE_PROTECT] = 0.5
    s[A.LS_PLUCK_M] = 1e-7
    s[A.LS_SKIN_FACTOR] = 2.0
    s[A.LS_FAT_SIGMA_PA] = 4e9
    s[A.LS_FAT_B] = -0.0212
    s[A.LS_CRACK_A0_M] = 5e-6
    s[A.LS_CRACK_GROWTH] = 1.0
    s[A.LS_DOC_RATIO] = 0.5
    _ = rng
    return law


def _case(seed=1, nt=3, ne=12, nq=3):
    rng = np.random.default_rng(seed)
    geo = A.new_geo(nt, ne)
    geo.valid[:] = 1
    geo.valid[1, 5] = 0
    geo.kind[:] = 0
    kappa = rng.uniform(0.2, 1.5, (nt, ne))
    geo.sin_k[:] = np.sin(kappa)
    geo.cos_k[:] = np.cos(kappa)
    geo.rake[:] = rng.uniform(-0.1, 0.2, (nt, ne))
    geo.clear[:] = rng.uniform(0.05, 0.2, (nt, ne))
    geo.g_vb[:] = 1 / np.tan(geo.clear) - np.tan(geo.rake)
    geo.c_coat[:] = np.cos(geo.clear) * (1 - np.tan(geo.clear) * np.tan(geo.rake))
    geo.beta[:] = 0.5 * math.pi - geo.rake - geo.clear
    geo.r_e0[:] = rng.uniform(5e-6, 2e-5, (nt, ne))
    geo.db[:] = 5e-4
    inp = A.new_input(nt, ne, nq)
    cut = rng.uniform(0, 1, (nt, ne)) > 0.3
    cut[0, :4] = True
    cut[0, 4] = False
    inp.in_cut_s[:] = np.where(cut, rng.uniform(1e-3, 3e-3, (nt, ne)), 0.0)
    inp.sv_flank[:] = inp.in_cut_s * rng.uniform(1e9, 5e9, (nt, ne))
    inp.sv_rake[:] = inp.in_cut_s * rng.uniform(5e8, 2e9, (nt, ne))
    inp.h_int[:] = inp.in_cut_s * rng.uniform(2e-5, 2e-4, (nt, ne))
    inp.h_int[0, :4] = inp.in_cut_s[0, :4] * 1e-4          # an even run: its end at element 3 is a DOC line
    inp.entries[:] = np.where(cut, rng.integers(0, 5, (nt, ne)), 0)
    inp.skin_frac[:] = np.where(cut, rng.uniform(0, 1, (nt, ne)), 0.0)
    inp.bue_broke[:] = np.where(cut & (rng.uniform(0, 1, (nt, ne)) > 0.8), 1.0, 0.0)
    inp.flank_T[:] = rng.uniform(800, 1250, (nt, ne, nq))
    inp.rake_T[:] = rng.uniform(900, 1350, (nt, ne, nq))
    w = rng.uniform(0.2, 1, nq)
    inp.w_q[:] = w / w.sum()
    inp.cycle_dT[:] = rng.uniform(0, 400, (nt, ne))
    inp.therm_coef[:] = 5.5e-6 * 5.8e11 / (2 * 0.78)
    inp.diffusivity[:] = 2.5e-5
    inp.scal[A.IS_DT_S] = 0.01
    st = A.new_state(nt, ne)
    st.vn[:] = rng.uniform(0, 2e-5, (nt, ne))
    st.coat_f[:] = np.where(rng.uniform(0, 1, (nt, ne)) > 0.4, rng.uniform(0, 3e-6, (nt, ne)), 0.0)
    st.coat_r[:] = rng.uniform(0, 3e-6, (nt, ne))
    st.D_th[:] = rng.uniform(0.5, 1.2, (nt, ne))
    st.bue_m[:] = np.where(rng.uniform(0, 1, (nt, ne)) > 0.7, rng.uniform(0, 1e-4, (nt, ne)), 0.0)
    st.chip_d[:] = np.where(rng.uniform(0, 1, (nt, ne)) > 0.9, 2e-5, 0.0)
    st.chipped[:] = (st.chip_d > 0).astype(np.int8)
    return geo, inp, st


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_advance_elements_matches_reference(seed):
    geo, inp, st = _case(seed)
    law = _law(seed)
    st_ref = _clone(st)
    out = A.new_output(*geo.valid.shape)
    K.advance_elements(geo, inp, law, st, out)
    ref = R.advance_elements_ref(geo, inp, law, st_ref)
    np.testing.assert_array_equal(out.doc, ref["doc"])
    for name in ("d_vn", "d_notch", "d_kt", "d_D"):
        np.testing.assert_allclose(getattr(out, name), ref[name], rtol=1e-12, atol=0.0, err_msg=name)
    np.testing.assert_allclose(out.scal, ref["scal"], rtol=1e-12)
    for name in st._fields:
        np.testing.assert_allclose(getattr(st, name), getattr(st_ref, name), rtol=1e-12, atol=1e-30, err_msg=name)
    assert out.doc[0, 3] == 1               # run [0..3] of tooth 0 ends before an element out of cut
    assert np.all(out.d_vn[~(inp.in_cut_s > 0)] == 0.0)


def test_element_outputs_match_reference():
    geo, inp, st = _case(5)
    law = _law()
    out = A.new_output(*geo.valid.shape)
    K.advance_elements(geo, inp, law, st, out)
    K.element_outputs(geo, law, st, out)
    ref = R.element_outputs_ref(geo, law, st)
    for name in ("VB", "VB_N", "radial", "axial", "edge_r"):
        np.testing.assert_allclose(getattr(out, name), ref[name], rtol=1e-13, atol=0.0, err_msg=name)


def test_helpers_match_reference():
    A_ = np.array([2e-12, 5e-9])
    B_ = np.array([9000.0, 18000.0])
    lo = np.array([-math.inf, 1100.0])
    for t in (300.0, 900.0, 1099.9, 1100.0, 1500.0):
        assert K.usui_rate(A_, B_, lo, t) == pytest.approx(R.usui_rate_ref(A_, B_, lo, t), rel=1e-15)
    assert K.usui_rate(np.zeros(0), np.zeros(0), np.zeros(0), 1000.0) == 0.0
    xs, ys = np.array([1.0, 2.0, 4.0]), np.array([10.0, 20.0, 0.0])
    for x in (0.0, 1.0, 1.5, 3.0, 4.0, 9.0):
        assert K.table_at(xs, ys, x) == pytest.approx(R.table_at_ref(xs, ys, x), abs=1e-14)
    for args in ((1e5, 2e-4, 1.0, 0.1, 0.12, 0.5), (3e5, 1e-4, 1.4, -0.1, 0.08, 0.9), (1e5, 2e-4, 0.7, 0.3, 0.1, 0.2)):
        assert np.allclose(K.wedge_tension(*args), R.wedge_tension_ref(*args), rtol=1e-13)
        s, cx, cy = K.wedge_tension(*args)
        for m in (8.0, 15.0, 25.0):
            assert K.wedge_volume_factor(cx, cy, args[2], m, 64) == pytest.approx(
                R.wedge_volume_factor_ref(cx, cy, args[2], m, 64), rel=1e-12)
    assert K.worn_edge_radius(1e-5, 2e-5, 1e-5, 1e-4, 1, 1.0) == pytest.approx(5e-5)
    assert K.worn_edge_radius(1e-5, 2e-5, 1e-5, 1e-4, 0, 1.0) == pytest.approx(4e-5)


def test_wedge_volume_factor_converges_with_nodes():
    s, cx, cy = K.wedge_tension(1e5, 2e-4, 1.1, 0.1, 0.12, 0.5)
    for m in (10.0, 20.0, 30.0):
        coarse = K.wedge_volume_factor(cx, cy, 1.1, m, 64)
        fine = K.wedge_volume_factor(cx, cy, 1.1, m, 8192)
        assert coarse == pytest.approx(fine, rel=1e-2)       # documented: 64 nodes within 1 % (strength 1 % / m)
    # flamant tension sits on the rake face for a force between the cutting direction and the bisector
    assert s > 0.0


def _rows(n, tooth=0, elem=2, force=600.0, h=1e-4, kind=1):
    rows = np.zeros((n, A.SHOCK_WIDTH))
    rows[:, A.SH_TICK] = np.arange(n) * 1000
    rows[:, A.SH_TOOTH] = tooth
    rows[:, A.SH_ELEM] = elem
    rows[:, A.SH_H] = h
    rows[:, A.SH_FORCE] = force
    rows[:, A.SH_KIND] = kind
    return rows


def _chip_law(m=15.0, sigma0=2.0e9, vref=3e-10, kic=1e7):
    chip = A.new_chip_law()
    c = chip.scal
    c[A.CL_M] = m
    c[A.CL_SIGMA0_PA] = sigma0
    c[A.CL_VREF_M3] = vref
    c[A.CL_KIC] = kic
    c[A.CL_Y] = 1.1215
    c[A.CL_LC_RATIO] = 2.0
    c[A.CL_CHIP_RATIO] = 1.0
    c[A.CL_EXIT_FACTOR] = 1.0
    c[A.CL_MU] = 0.5
    c[A.CL_COAT_SIGMA0_PA] = 4e9
    c[A.CL_COAT_M] = 6.0
    c[A.CL_BETA_MIN] = math.radians(20)
    c[A.CL_BETA_MAX] = math.radians(160)
    c[A.CL_NTHETA] = 64
    c[A.CL_ROUNDING] = 1.0
    return chip


def test_chip_trials_match_reference():
    geo, inp, st = _case(7)
    st.coat_f[:] = 2e-6
    st.crack_m[0, 2] = 5e-5
    rows = np.concatenate([_rows(40, 0, 2, 900.0), _rows(30, 2, 7, 1500.0, kind=2), _rows(5, 1, 5, 800.0)])
    b_eng = np.array([5e-3, 4e-3, 6e-3])
    keys = np.array([11, stream_key("wear.chipping"), stream_key("wear.coating"), 3], np.int64)
    chip = _chip_law(sigma0=1.5e9)
    st_ref = _clone(st)
    out = np.zeros((rows.shape[0], A.N_CO))
    K.chip_trials(geo, st, chip, keys, rows, b_eng, out)
    ref = R.chip_trials_ref(geo, st_ref, chip, keys, rows, b_eng)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0.0)
    for name in st._fields:
        np.testing.assert_array_equal(getattr(st, name), getattr(st_ref, name), err_msg=name)
    assert np.all(out[70:, A.CO_P] == 0.0)       # invalid element: no trial
    assert np.all(st.n_chip[0, 2] == 40) and st.n_chip[1, 5] == 0


def test_layouts_are_kernel_arguments():
    for layout in (A.new_geo(2, 3), A.new_input(2, 3, 4), A.new_law(), A.new_state(2, 3), A.new_output(2, 3),
                   A.new_chip_law()):
        assert A.is_kernel_layout(layout)
    keys = A.element_keys(3, 2, 4)
    assert keys.dtype == np.int64 and len(set(keys.ravel().tolist())) == 8
