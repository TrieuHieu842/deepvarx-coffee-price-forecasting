"""Unit tests for Clark and West (2006, 2007) nested model test."""
import numpy as np
import pytest
from scipy import stats

from src.clark_west import clark_west_test


def test_clark_west_hand_calculation():
    """Verify Clark-West adjusted loss differential calculation against hand computation."""
    y = np.array([10.0, 12.0, 11.0, 13.0, 15.0])
    # Restricted model predictions (Model A)
    y_r = np.array([9.5, 11.5, 10.5, 12.5, 14.5])
    # Unrestricted model predictions (Model B)
    y_u = np.array([9.8, 12.1, 10.9, 13.2, 14.9])

    e_r = y - y_r  # [0.5, 0.5, 0.5, 0.5, 0.5]
    e_u = y - y_u  # [0.2, -0.1, 0.1, -0.2, 0.1]

    # Hand calculation:
    # adj_term = (y_r - y_u)^2
    # f_t = e_r^2 - e_u^2 + adj_term
    adj = (y_r - y_u) ** 2
    expected_f_t = (e_r ** 2) - (e_u ** 2) + adj
    expected_f_bar = float(np.mean(expected_f_t))

    cw_stat, p_val, f_bar, diag = clark_west_test(y, y_r, y_u, horizon=1, hac_lag=0)

    assert pytest.approx(f_bar, rel=1e-5) == expected_f_bar
    assert diag["T"] == 5
    assert cw_stat > 0  # Unrestricted model had lower errors
    assert 0.0 <= p_val <= 1.0


def test_clark_west_when_unrestricted_is_superior():
    """When the larger model provides genuinely superior forecasts, CW test rejects H0."""
    np.random.seed(42)
    T = 100
    y = np.random.normal(loc=50.0, scale=2.0, size=T)
    # Restricted model: constant mean (poor fit)
    y_r = np.full(T, 45.0)
    # Unrestricted model: accurate predictions
    y_u = y + np.random.normal(0, 0.2, size=T)

    cw_stat, p_val, f_bar, diag = clark_west_test(y, y_r, y_u, horizon=1)

    assert f_bar > 0
    assert cw_stat > 5.0
    assert p_val < 0.001


def test_clark_west_noise_correction():
    """Under H0 where unrestricted model is just restricted model + independent noise,
    standard MSPE of unrestricted is larger due to noise, but Clark-West adjustment compensates."""
    np.random.seed(123)
    T = 200
    y = np.random.normal(0, 1, size=T)
    y_r = np.zeros(T)
    # Unrestricted adds pure estimation noise
    noise = np.random.normal(0, 0.5, size=T)
    y_u = y_r + noise

    # Standard loss differential: e_r^2 - e_u^2 = y^2 - (y - noise)^2
    # Mean of (e_r^2 - e_u^2) is approximately -E[noise^2] = -0.25 (unrestricted looks worse)
    d_mean = np.mean((y - y_r) ** 2 - (y - y_u) ** 2)
    assert d_mean < 0  # Sample MSPE of unrestricted is higher

    # Clark-West adjusted mean f_bar should be close to 0 under H0
    cw_stat, p_val, f_bar, diag = clark_west_test(y, y_r, y_u, horizon=1)
    # Adjustment term restores the null mean
    assert f_bar > d_mean
    assert p_val > 0.05  # Does not falsely reject H0
