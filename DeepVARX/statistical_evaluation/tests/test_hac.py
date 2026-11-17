"""Unit tests for Newey-West HAC estimator."""
import math
import numpy as np
import pytest

from src.hac_estimator import estimate_hac_variance, select_hac_lag


def test_bartlett_weights_and_autocov():
    """Verify Bartlett weights formula w_j = 1 - j/(q+1)."""
    # T = 10, d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d = np.arange(1, 11, dtype=float)
    lrv, se, diag = estimate_hac_variance(d, horizon=1, hac_lag=2)

    assert diag["q"] == 2
    weights = diag["bartlett_weights"]
    # For q=2: w_1 = 1 - 1/3 = 2/3, w_2 = 1 - 2/3 = 1/3
    assert len(weights) == 2
    assert math.isclose(weights[0], 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(weights[1], 1.0 / 3.0, rel_tol=1e-6)


def test_select_hac_lag_rules():
    """Verify lag selection logic."""
    # When hac_lag is specified explicitly
    assert select_hac_lag(T=100, horizon=1, hac_lag=4) == 4
    # When horizon > 1, q must be at least horizon - 1
    assert select_hac_lag(T=100, horizon=5, hac_lag=1) == 4
    # Auto selection: T=542 => floor(4 * 5.42^(2/9)) = 5
    assert select_hac_lag(T=542, horizon=1, hac_lag=None) == 5


def test_hac_zero_variance():
    """Constant series must return zero LRV and SE."""
    d = np.full(50, 42.0)
    lrv, se, diag = estimate_hac_variance(d, horizon=1)
    assert lrv == 0.0
    assert se == 0.0


def test_hac_positive_variance():
    """Verify LRV is strictly positive on non-trivial series."""
    rng = np.random.RandomState(42)
    d = rng.normal(loc=1.0, scale=2.0, size=200)
    lrv, se, diag = estimate_hac_variance(d, horizon=1)
    assert lrv > 0.0
    assert se > 0.0
    assert not math.isnan(lrv)
