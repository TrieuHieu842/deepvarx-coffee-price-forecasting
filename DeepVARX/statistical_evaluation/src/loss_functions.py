"""Loss functions and loss differential calculations for forecast evaluation.

Loss differential definition:
    d_t = L(e_A,t) - L(e_B,t)

Interpretation:
    d_t < 0 <=> L(e_A,t) < L(e_B,t)  (Model A has lower loss than Model B)
    d_t > 0 <=> L(e_A,t) > L(e_B,t)  (Model B has lower loss than Model A)
    d_t = 0 <=> L(e_A,t) == L(e_B,t) (Both models have identical loss)
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats


def squared_error_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared Error Loss: L(e) = (y_true - y_pred)^2."""
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return e ** 2


def absolute_error_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute Error Loss: L(e) = |y_true - y_pred|."""
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return np.abs(e)


def gaussian_crps_loss(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    min_std: float = 1e-6,
) -> np.ndarray:
    """Continuous Ranked Probability Score (CRPS) for Gaussian predictive distribution.

    CRPS(mu, sigma, y) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(y_pred_mean, dtype=float)
    sigma = np.maximum(np.asarray(y_pred_std, dtype=float), min_std)

    z = (y - mu) / sigma
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    crps = sigma * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - 1.0 / math.sqrt(math.pi))
    return crps


LOSS_REGISTRY: Dict[str, Callable] = {
    "mse": squared_error_loss,
    "squared_error": squared_error_loss,
    "mae": absolute_error_loss,
    "absolute_error": absolute_error_loss,
    "crps": gaussian_crps_loss,
}


def get_loss_function(name: str) -> Callable:
    """Retrieve loss function by normalized name."""
    key = name.strip().lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[key]


def compute_loss_differential(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    loss_func: Union[str, Callable] = "mse",
    pred_std_a: Optional[np.ndarray] = None,
    pred_std_b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute loss differential d_t = L(e_A,t) - L(e_B,t).

    Returns:
        d_t: Loss differential array (loss_a - loss_b)
        loss_a: Loss array of Model A
        loss_b: Loss array of Model B
    """
    fn = get_loss_function(loss_func) if isinstance(loss_func, str) else loss_func

    if loss_func in ("crps", gaussian_crps_loss):
        if pred_std_a is None or pred_std_b is None:
            raise ValueError("CRPS requires pred_std_a and pred_std_b")
        loss_a = fn(y_true, y_pred_a, pred_std_a)
        loss_b = fn(y_true, y_pred_b, pred_std_b)
    else:
        loss_a = fn(y_true, y_pred_a)
        loss_b = fn(y_true, y_pred_b)

    d_t = loss_a - loss_b
    return d_t, loss_a, loss_b
