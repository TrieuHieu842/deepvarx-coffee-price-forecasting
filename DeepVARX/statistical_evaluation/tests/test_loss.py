"""Unit tests for loss functions and loss differential calculation."""
import math
import numpy as np
import pytest

from src.loss_functions import (
    squared_error_loss,
    absolute_error_loss,
    gaussian_crps_loss,
    compute_loss_differential,
)


def test_squared_error_loss():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 17.0, 30.0])
    loss = squared_error_loss(y_true, y_pred)
    # (-2)^2 = 4, (3)^2 = 9, 0^2 = 0
    assert np.allclose(loss, [4.0, 9.0, 0.0])


def test_absolute_error_loss():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 17.0, 30.0])
    loss = absolute_error_loss(y_true, y_pred)
    assert np.allclose(loss, [2.0, 3.0, 0.0])


def test_crps_exact_match():
    """When true matches mean and sigma -> 0, CRPS -> 0."""
    y_true = np.array([100.0])
    y_pred = np.array([100.0])
    sigma = np.array([1e-4])
    loss = gaussian_crps_loss(y_true, y_pred, sigma)
    assert loss[0] < 1e-3


def test_loss_differential_sign_convention():
    """Verify d_t = Loss(A) - Loss(B)."""
    y_true = np.array([10.0])
    y_a = np.array([11.0])  # err_a = -1, mse_a = 1
    y_b = np.array([13.0])  # err_b = -3, mse_b = 9

    d_t, l_a, l_b = compute_loss_differential(y_true, y_a, y_b, loss_func="mse")
    assert d_t[0] == 1.0 - 9.0 == -8.0
    assert l_a[0] == 1.0
    assert l_b[0] == 9.0
