"""Unit tests for Diebold-Mariano core test engine."""
import math
import numpy as np
import pytest

from src.dm_test import diebold_mariano_test


def test_identical_forecasts():
    """When Model A and Model B make identical predictions, stat should be 0 and p-val 1.0."""
    y_true = np.array([100.0, 102.0, 105.0, 103.0, 108.0, 107.0, 110.0, 112.0, 111.0, 115.0])
    y_pred_a = np.array([101.0, 101.0, 104.0, 104.0, 107.0, 108.0, 109.0, 113.0, 110.0, 114.0])
    y_pred_b = y_pred_a.copy()

    res = diebold_mariano_test(y_true, y_pred_a, y_pred_b, loss_func="mse")

    assert res.is_identical is True
    assert res.dm_stat == 0.0
    assert res.hln_stat == 0.0
    assert res.dm_p_value_two_sided == 1.0
    assert res.hln_p_value_two_sided == 1.0
    assert res.mean_loss_diff == 0.0
    assert res.better_model_by_loss == "Equal"


def test_model_a_clearly_better():
    """When Model A has much lower loss than Model B: d_bar < 0, stat < 0."""
    y_true = np.array([100.0, 102.0, 105.0, 103.0, 108.0, 107.0, 110.0, 112.0, 111.0, 115.0, 120.0, 122.0])
    # A is very close to true (error = 0.5)
    y_pred_a = y_true + 0.5
    # B has large errors (error = 10.0)
    y_pred_b = y_true + 10.0

    res = diebold_mariano_test(y_true, y_pred_a, y_pred_b, loss_func="mse")

    assert res.mean_loss_diff < 0.0
    assert res.dm_stat < 0.0
    assert res.hln_stat < 0.0
    assert res.better_model_by_loss == "Model A"
    assert res.hln_p_value_two_sided < 0.001


def test_model_b_clearly_better():
    """When Model B has much lower loss than Model A: d_bar > 0, stat > 0."""
    y_true = np.array([100.0, 102.0, 105.0, 103.0, 108.0, 107.0, 110.0, 112.0, 111.0, 115.0, 120.0, 122.0])
    y_pred_a = y_true + 10.0
    y_pred_b = y_true + 0.5

    res = diebold_mariano_test(y_true, y_pred_a, y_pred_b, loss_func="mse")

    assert res.mean_loss_diff > 0.0
    assert res.dm_stat > 0.0
    assert res.hln_stat > 0.0
    assert res.better_model_by_loss == "Model B"
    assert res.hln_p_value_two_sided < 0.001


def test_sign_consistency_symmetry():
    """Swapping Model A and Model B must invert the sign of d_bar, dm_stat, and hln_stat."""
    rng = np.random.RandomState(42)
    y_true = 50000.0 + rng.randn(40) * 1000.0
    y_pred_a = y_true + rng.randn(40) * 200.0
    y_pred_b = y_true + rng.randn(40) * 300.0

    res_ab = diebold_mariano_test(y_true, y_pred_a, y_pred_b, loss_func="mse")
    res_ba = diebold_mariano_test(y_true, y_pred_b, y_pred_a, loss_func="mse")

    assert math.isclose(res_ab.mean_loss_diff, -res_ba.mean_loss_diff, rel_tol=1e-7)
    assert math.isclose(res_ab.dm_stat, -res_ba.dm_stat, rel_tol=1e-7)
    assert math.isclose(res_ab.hln_stat, -res_ba.hln_stat, rel_tol=1e-7)
    assert math.isclose(res_ab.dm_p_value_two_sided, res_ba.dm_p_value_two_sided, rel_tol=1e-7)
    assert math.isclose(res_ab.hln_p_value_two_sided, res_ba.hln_p_value_two_sided, rel_tol=1e-7)


def test_hand_calculated_benchmark():
    """Verify DM calculation against exact manual formula.
    Let T=4, d = [-2, -2, -2, -2] -> d_bar = -2, gamma_0 = 0 -> zero variance.
    Let d = [-3, -1, -3, -1] -> d_bar = -2, dev = [-1, 1, -1, 1].
    T = 4.
    gamma_0 = (1 + 1 + 1 + 1) / 4 = 1.0.
    gamma_1 = (-1*1 + 1*-1 + -1*1) / 4 = -3/4 = -0.75.
    If q = 0 (h=1, hac_lag=0):
    lrv = gamma_0 = 1.0.
    se = sqrt(1.0 / 4) = 0.5.
    DM = -2 / 0.5 = -4.0.
    HLN scale = sqrt((4-1)/4) = sqrt(3/4) = 0.866025.
    HLN stat = -4.0 * sqrt(3/4) = -3.4641016.
    """
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    # Model A error: [0, 0, 0, 0] -> loss_a = [0, 0, 0, 0]
    # Model B error: [sqrt(3), 1, sqrt(3), 1] -> loss_b = [3, 1, 3, 1]
    # d_t = loss_a - loss_b = [-3, -1, -3, -1]
    y_pred_a = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred_b = np.array([10.0 - math.sqrt(3.0), 9.0, 10.0 - math.sqrt(3.0), 9.0])

    res = diebold_mariano_test(
        y_true, y_pred_a, y_pred_b, loss_func="mse", horizon=1, hac_lag=0, min_observations=4
    )

    assert math.isclose(res.mean_loss_diff, -2.0, rel_tol=1e-6)
    assert math.isclose(res.long_run_variance, 1.0, rel_tol=1e-6)
    assert math.isclose(res.se_mean, 0.5, rel_tol=1e-6)
    assert math.isclose(res.dm_stat, -4.0, rel_tol=1e-6)
    expected_hln = -4.0 * math.sqrt(3.0 / 4.0)
    assert math.isclose(res.hln_stat, expected_hln, rel_tol=1e-5)
