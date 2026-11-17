"""Unit tests for Holm-Bonferroni correction."""
import math
import numpy as np
import pytest

from src.holm_correction import holm_bonferroni_correction


def test_holm_hand_calculated_example():
    """Verify Holm step-down on a known hand-calculated sequence."""
    # Input in unsorted order: [0.01, 0.04, 0.03]
    # Sorted: p_(1)=0.01, p_(2)=0.03, p_(3)=0.04
    # Multipliers: 3, 2, 1
    # Step 1: 0.01 * 3 = 0.03
    # Step 2: 0.03 * 2 = 0.06, running_max = max(0.03, 0.06) = 0.06
    # Step 3: 0.04 * 1 = 0.04, running_max = max(0.06, 0.04) = 0.06 (monotonicity enforced!)
    # Expected output in original order: [0.03, 0.06, 0.06]
    raw = [0.01, 0.04, 0.03]
    adj = holm_bonferroni_correction(raw)

    expected = np.array([0.03, 0.06, 0.06])
    assert np.allclose(adj, expected, atol=1e-7)


def test_holm_monotonicity_and_bounds():
    """Ensure p_adj >= p_raw, p_adj <= 1.0, and ordered monotonically."""
    raw = [0.001, 0.02, 0.015, 0.5, 0.8, 0.04]
    adj = holm_bonferroni_correction(raw)

    # Upper bound
    assert np.all(adj <= 1.0)
    # p_adj >= p_raw
    assert np.all(adj >= np.array(raw) - 1e-9)

    # In sorted order of raw, adj must be non-decreasing
    sort_idx = np.argsort(raw)
    sorted_adj = adj[sort_idx]
    assert np.all(np.diff(sorted_adj) >= 0.0)


def test_holm_edge_cases():
    """Test empty, single element, and NaN handling."""
    assert len(holm_bonferroni_correction([])) == 0

    single = holm_bonferroni_correction([0.05])
    assert math.isclose(single[0], 0.05, rel_tol=1e-7)

    with_nan = holm_bonferroni_correction([0.01, np.nan, 0.02])
    assert math.isclose(with_nan[0], 0.02, rel_tol=1e-7)
    assert np.isnan(with_nan[1])
    assert math.isclose(with_nan[2], 0.02, rel_tol=1e-7)
