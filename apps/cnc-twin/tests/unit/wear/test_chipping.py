"""Weibull chipping of the edge at entry/exit shocks: each edge has its own strength quantile (one draw per edge
and generation), so the share of edges that chip at a first load is the Weibull probability, an edge that survived
a load does not chip again at that load unless its strength degrades along the Basquin curve (Miner sum of the
load cycles); draws are reproducible and seed-dependent, the size effect and the stress exponent are Weibull's,
comb cracks raise the hazard, a chip makes a new edge, coatings spall."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core.rng import stream_key
from cnctwin.wear import abi as A
from cnctwin.wear import fracture as FR
from cnctwin.wear import kernels as K


def _geo(n):
    geo = A.new_geo(1, n)
    geo.valid[:] = 1
    geo.sin_k[:] = 1.0
    geo.rake[:] = 0.1
    geo.clear[:] = 0.12
    geo.g_vb[:] = 1 / math.tan(0.12) - math.tan(0.1)
    geo.c_coat[:] = 1.0
    geo.beta[:] = 0.5 * math.pi - 0.22
    geo.r_e0[:] = 1e-5
    geo.db[:] = 1e-3
    return geo


def _law(m=15.0, trs=2.2e9, kic=1e7):
    chip = A.new_chip_law()
    c = chip.scal
    c[A.CL_M] = m
    c[A.CL_SIGMA0_PA] = FR.characteristic_strength_Pa(trs, m)
    c[A.CL_VREF_M3] = FR.reference_volume_m3(m)
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


def _trials(force, n=4000, width=1e-3, seed=5, crack=0.0, coat=0.0, h=1e-4, law=None, st=None):
    """One entry row per element (n identical elements, each with its own strength): one load of each edge."""
    geo = _geo(n)
    if st is None:
        st = A.new_state(1, n)
        st.crack_m[:] = crack
        st.coat_f[:] = coat
        st.coat_r[:] = coat
    rows = np.zeros((n, A.SHOCK_WIDTH))
    rows[:, A.SH_TICK] = np.arange(n)
    rows[:, A.SH_ELEM] = np.arange(n)
    rows[:, A.SH_H] = h
    rows[:, A.SH_FORCE] = force
    rows[:, A.SH_KIND] = A.SHOCK_ENTRY
    keys = np.array([seed, stream_key("wear.chipping"), stream_key("wear.coating"), 0], np.int64)
    out = np.zeros((n, A.N_CO))
    K.chip_trials(geo, st, law or _law(), keys, rows, np.array([width]), out)
    return out, st


def _force_for(p_target, **kw):
    """Force at which the chipping probability is p_target (bisection on the kernel's own probability)."""
    lo, hi = 1.0, 1e5
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        p = _trials(mid, n=1, **kw)[0][0, A.CO_P]
        lo, hi = (mid, hi) if p < p_target else (lo, mid)
    return math.sqrt(lo * hi)


def test_chipping_frequency_matches_the_weibull_probability():
    force = _force_for(0.2)
    out, st = _trials(force)
    p = out[0, A.CO_P]
    assert p == pytest.approx(0.2, rel=1e-6)
    assert np.allclose(out[:, A.CO_P], p)
    frequency = float(np.mean(out[:, A.CO_CHIPPED]))
    sigma = math.sqrt(p * (1 - p) / out.shape[0])
    assert abs(frequency - p) < 4.0 * sigma
    assert np.array_equal(st.chipped[0] != 0, out[:, A.CO_CHIPPED] > 0)
    assert np.all(st.chip_d[0][st.chipped[0] != 0] > 0) and np.all(st.n_chip == 1)


def test_draws_are_reproducible_and_depend_on_the_seed():
    force = _force_for(0.3)
    a, _ = _trials(force, seed=5)
    b, _ = _trials(force, seed=5)
    c, _ = _trials(force, seed=6)
    assert np.array_equal(a[:, A.CO_CHIPPED], b[:, A.CO_CHIPPED])
    assert not np.array_equal(a[:, A.CO_CHIPPED], c[:, A.CO_CHIPPED])
    sigma = math.sqrt(0.3 * 0.7 / a.shape[0])
    assert abs(np.mean(c[:, A.CO_CHIPPED]) - 0.3) < 4.0 * sigma


def test_weibull_size_effect_and_stress_exponent():
    m = 15.0
    base, _ = _trials(800.0, n=1)
    hazard = -math.log1p(-base[0, A.CO_P])
    # twice the engaged width at the same force per width: twice the stressed volume, same stress
    wide, _ = _trials(1600.0, n=1, width=2e-3)
    assert wide[0, A.CO_SIGMA_PA] == pytest.approx(base[0, A.CO_SIGMA_PA], rel=1e-12)
    assert -math.log1p(-wide[0, A.CO_P]) == pytest.approx(2.0 * hazard, rel=1e-9)
    # 1.2 x the force: 1.2^m the hazard (the stress field scales, the effective volume does not)
    strong, _ = _trials(960.0, n=1)
    assert -math.log1p(-strong[0, A.CO_P]) == pytest.approx(1.2 ** m * hazard, rel=1e-9)


def test_comb_cracks_raise_the_chipping_probability():
    force = _force_for(0.01)
    intact, _ = _trials(force, n=1)
    sigma = intact[0, A.CO_SIGMA_PA]
    # a crack that fails at 1.1 x the wedge stress: K_IC / (Y sqrt(pi a)) = 1.1 sigma
    depth = (1e7 / (1.1215 * 1.1 * sigma)) ** 2 / math.pi
    cracked, _ = _trials(force, n=1, crack=depth)
    assert cracked[0, A.CO_P] > 10.0 * intact[0, A.CO_P]
    extra = (1.0 / 1.1) ** 15.0
    assert -math.log1p(-cracked[0, A.CO_P]) == pytest.approx(-math.log1p(-intact[0, A.CO_P]) + extra, rel=1e-9)


def test_coatings_spall_with_their_own_probability():
    out, st = _trials(_force_for(0.05, h=5e-5), n=2000, coat=3e-6, h=5e-5)
    survived = out[:, A.CO_CHIPPED] == 0
    p_coat = out[survived, A.CO_P_COAT]
    assert np.all(p_coat > 0.0) and np.allclose(p_coat, p_coat[0])
    spalled = out[:, A.CO_SPALLED] > 0
    sigma = math.sqrt(p_coat[0] * (1 - p_coat[0]) / survived.sum())
    assert abs(spalled[survived].mean() - p_coat[0]) < 4.0 * sigma + 1e-12
    assert np.all(st.coat_f[0][spalled] == 0.0) and np.all(st.coat_f[0][survived & ~spalled] == 3e-6)
    assert np.all(st.coat_f[0][~survived] == 0.0)          # a chip takes the coating with it


def test_an_edge_that_survived_a_load_survives_it_again_without_fatigue():
    """Weibull's weakest link: the strength is the edge's (its flaw population), not the load cycle's. A second
    identical load chips no further edge when the strength does not degrade (b = 0); a new draw per entry had made
    every edge chip in the end at any stress (1 - (1 - P)^N)."""
    force = _force_for(0.2)
    first, st = _trials(force)
    second, st = _trials(force, st=st)
    assert np.mean(first[:, A.CO_CHIPPED]) == pytest.approx(0.2, abs=4.0 * math.sqrt(0.16 / 4000))
    assert not np.any((second[:, A.CO_CHIPPED] > 0) & (first[:, A.CO_CHIPPED] == 0))
    assert np.all(st.n_chip == 2)


def test_load_cycles_degrade_the_edge_along_the_basquin_curve():
    """Equal rank (strength life): after N identical cycles the hazard is N^(m/p) times the static one (Miner's sum
    of (sigma_n / sigma_0)^p, p = -1/b), so the edges that chip by the N-th cycle are 1 - exp(-N^(m/p) H_1)."""
    m = 15.0
    b = math.log(0.7) / math.log(2e7)
    law = _law(m=m)
    law.scal[A.CL_FATIGUE_B] = b
    force = _force_for(0.01, law=law)
    st = None
    for cycle in range(1, 6):
        out, st = _trials(force, law=law, st=st)
    alive = st.chipped[0] == 0
    p1 = out[np.flatnonzero(alive)[0], A.CO_P]            # the static probability of the load on an intact edge
    h1 = -math.log1p(-p1)
    assert np.allclose(out[alive, A.CO_HAZARD], 5.0 ** (m * -b) * h1, rtol=1e-9, atol=0.0)
    expected = -math.expm1(-(5.0 ** (m * -b)) * h1)
    chipped = float(np.mean(st.chipped[0] != 0))
    assert abs(chipped - expected) < 4.0 * math.sqrt(expected * (1 - expected) / st.chipped.size)
    assert expected > 1.3 * p1                                            # fatigue: more edges by the 5th cycle


def test_a_chip_makes_a_new_edge():
    """A chip takes the fatigue sum with it and the edge draws the next generation's strength."""
    force = _force_for(0.5)
    out, st = _trials(force, n=400)
    hit = st.chipped[0] != 0
    assert hit.any() and np.all(st.edge_fat[0][hit] == 0.0) and np.all(st.chipped[0][hit] == 1)
    again, st2 = _trials(force, n=400, st=st)
    # the new edges chip again at the static probability of the load (their own new quantiles)
    renewed = again[hit, A.CO_CHIPPED]
    p = again[0, A.CO_P]
    assert abs(float(np.mean(renewed)) - p) < 4.0 * math.sqrt(p * (1 - p) / hit.sum())


def test_reference_strength_of_the_trs_test_piece():
    m = 15.0
    sigma0 = FR.characteristic_strength_Pa(2.2e9, m)
    assert sigma0 * math.gamma(1 + 1 / m) == pytest.approx(2.2e9)      # Weibull mean = TRS
    v = FR.reference_volume_m3(m)
    assert v == pytest.approx(6.5e-3 * 5.25e-3 * 14.5e-3 / (2 * 16 ** 2))
    # the median of a volume twice the reference is lower by 2^(-1/m)
    assert FR.strength_Pa(2 * v, sigma0, v, m) / FR.strength_Pa(v, sigma0, v, m) == pytest.approx(2 ** (-1 / m))
