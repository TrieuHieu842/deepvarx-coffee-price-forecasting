"""Unit tests for Harvey-Leybourne-Newbold (HLN) correction."""
import math
import pytest
from scipy import stats

from src.hln_correction import apply_hln_correction, compute_hln_multiplier


def test_hln_multiplier_h1():
    """For h=1, scale = sqrt((T - 1) / T)."""
    T = 100
    scale = compute_hln_multiplier(T=T, horizon=1)
    expected = math.sqrt(99.0 / 100.0)
    assert math.isclose(scale, expected, rel_tol=1e-7)


def test_hln_multiplier_multi_step():
    """For h=4, T=50: numerator = 50 + 1 - 8 + 4*3/50 = 43 + 0.24 = 43.24. Scale = sqrt(43.24 / 50)."""
    T = 50
    h = 4
    scale = compute_hln_multiplier(T=T, horizon=h)
    expected = math.sqrt((50 + 1 - 2 * 4 + 4 * 3 / 50.0) / 50.0)
    assert math.isclose(scale, expected, rel_tol=1e-7)


def test_hln_large_t_convergence():
    """As T -> infinity, HLN multiplier approaches 1.0."""
    scale = compute_hln_multiplier(T=1_000_000, horizon=1)
    assert math.isclose(scale, 1.0, rel_tol=1e-5)


def test_hln_p_value_calculation():
    """Check Student-t p-value calculation."""
    T = 20
    dm_stat = 2.0
    hln_stat, p_2s, p_1s, diag = apply_hln_correction(dm_stat, T=T, horizon=1)

    scale = math.sqrt(19.0 / 20.0)
    expected_stat = 2.0 * scale
    assert math.isclose(hln_stat, expected_stat, rel_tol=1e-6)

    # Student-t with df = 19
    expected_p_2s = 2.0 * stats.t.sf(abs(expected_stat), df=19)
    assert math.isclose(p_2s, expected_p_2s, rel_tol=1e-6)
