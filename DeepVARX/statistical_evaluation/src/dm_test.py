"""Diebold-Mariano (1995) test with Newey-West HAC and HLN finite-sample correction.

Loss differential convention:
    d_t = L(e_A,t) - L(e_B,t)

Hypotheses:
    H0: E[d_t] = 0 (Equal forecast accuracy)
    H1: E[d_t] != 0 (Different forecast accuracy)

Signs:
    d_bar < 0 => Model A has lower average loss than Model B
    d_bar > 0 => Model B has lower average loss than Model A
    stat < 0  => Model A favored
    stat > 0  => Model B favored
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats

from .hac_estimator import estimate_hac_variance
from .hln_correction import apply_hln_correction
from .loss_functions import compute_loss_differential


@dataclass
class DMTestResult:
    model_a: str
    model_b: str
    loss_function: str
    horizon: int
    n_observations: int
    mean_loss_a: float
    mean_loss_b: float
    mean_loss_diff: float
    std_loss_diff: float
    hac_lag: int
    long_run_variance: float
    se_mean: float
    dm_stat: float
    dm_p_value_two_sided: float
    dm_p_value_a_better: float
    hln_stat: float
    hln_p_value_two_sided: float
    hln_p_value_a_better: float
    degrees_of_freedom: int
    better_model_by_loss: str
    is_identical: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    loss_func: Union[str, Callable] = "mse",
    horizon: int = 1,
    hac_lag: Optional[int] = None,
    pred_std_a: Optional[np.ndarray] = None,
    pred_std_b: Optional[np.ndarray] = None,
    min_observations: int = 10,
) -> DMTestResult:
    """Execute Diebold-Mariano test between two forecast models.

    Parameters:
        y_true: Actual values array.
        y_pred_a: Model A predictions.
        y_pred_b: Model B predictions.
        model_a_name: Name of model A.
        model_b_name: Name of model B.
        loss_func: Loss function name ('mse', 'mae', 'crps') or callable.
        horizon: Forecast step h (>= 1).
        hac_lag: HAC bandwidth (None for automatic).
        pred_std_a: Predictive standard deviations for Model A (for CRPS).
        pred_std_b: Predictive standard deviations for Model B (for CRPS).
        min_observations: Minimum valid observations required.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_a = np.asarray(y_pred_a, dtype=float)
    y_b = np.asarray(y_pred_b, dtype=float)

    loss_name = loss_func if isinstance(loss_func, str) else getattr(loss_func, "__name__", "custom")
    is_crps = (loss_name.lower() in ("crps", "gaussian_crps_loss"))

    # Valid mask (no NaNs in any)
    mask = ~(np.isnan(y_t) | np.isnan(y_a) | np.isnan(y_b))
    if is_crps and pred_std_a is not None and pred_std_b is not None:
        mask = mask & ~(np.isnan(pred_std_a) | np.isnan(pred_std_b))

    y_t = y_t[mask]
    y_a = y_a[mask]
    y_b = y_b[mask]
    std_a = pred_std_a[mask] if (is_crps and pred_std_a is not None) else None
    std_b = pred_std_b[mask] if (is_crps and pred_std_b is not None) else None

    T = len(y_t)

    if T < min_observations:
        return DMTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            loss_function=loss_name,
            horizon=horizon,
            n_observations=T,
            mean_loss_a=np.nan,
            mean_loss_b=np.nan,
            mean_loss_diff=np.nan,
            std_loss_diff=np.nan,
            hac_lag=0,
            long_run_variance=np.nan,
            se_mean=np.nan,
            dm_stat=np.nan,
            dm_p_value_two_sided=np.nan,
            dm_p_value_a_better=np.nan,
            hln_stat=np.nan,
            hln_p_value_two_sided=np.nan,
            hln_p_value_a_better=np.nan,
            degrees_of_freedom=max(1, T - 1),
            better_model_by_loss="Insufficient Data",
            is_identical=False,
        )

    # Compute loss differential: d_t = Loss(A) - Loss(B)
    d_t, loss_a, loss_b = compute_loss_differential(
        y_t, y_a, y_b, loss_func=loss_func, pred_std_a=std_a, pred_std_b=std_b
    )

    mean_loss_a = float(np.mean(loss_a))
    mean_loss_b = float(np.mean(loss_b))
    d_bar = float(np.mean(d_t))
    d_std = float(np.std(d_t, ddof=1)) if T > 1 else 0.0

    # Determine better model by raw sample loss
    if abs(d_bar) < 1e-12:
        better_by_loss = "Equal"
    elif d_bar < 0:
        better_by_loss = model_a_name
    else:
        better_by_loss = model_b_name

    # Check identical forecast edge case
    max_abs_diff = float(np.max(np.abs(y_a - y_b)))
    if max_abs_diff < 1e-12 or np.allclose(loss_a, loss_b, atol=1e-12):
        return DMTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            loss_function=loss_name,
            horizon=horizon,
            n_observations=T,
            mean_loss_a=mean_loss_a,
            mean_loss_b=mean_loss_b,
            mean_loss_diff=0.0,
            std_loss_diff=0.0,
            hac_lag=0,
            long_run_variance=0.0,
            se_mean=0.0,
            dm_stat=0.0,
            dm_p_value_two_sided=1.0,
            dm_p_value_a_better=0.5,
            hln_stat=0.0,
            hln_p_value_two_sided=1.0,
            hln_p_value_a_better=0.5,
            degrees_of_freedom=T - 1,
            better_model_by_loss="Equal",
            is_identical=True,
        )

    # Estimate HAC variance
    lrv, se_mean, diag = estimate_hac_variance(d_t, horizon=horizon, hac_lag=hac_lag)
    lag_used = diag.get("q", 0)

    if se_mean <= 1e-18 or math.isnan(se_mean):
        if abs(d_bar) < 1e-12:
            dm_stat = 0.0
            p_two_sided = 1.0
            p_a_better = 0.5
        elif d_bar < 0:
            dm_stat = -999.0
            p_two_sided = 0.0
            p_a_better = 0.0
        else:
            dm_stat = 999.0
            p_two_sided = 0.0
            p_a_better = 1.0
    else:
        dm_stat = float(d_bar / se_mean)
        p_two_sided = float(2.0 * stats.norm.sf(abs(dm_stat)))
        p_a_better = float(stats.norm.cdf(dm_stat))

    # Apply Harvey-Leybourne-Newbold (HLN) correction
    hln_stat, hln_p_2s, hln_p_1s, hln_diag = apply_hln_correction(
        dm_stat, T=T, horizon=horizon
    )

    return DMTestResult(
        model_a=model_a_name,
        model_b=model_b_name,
        loss_function=loss_name,
        horizon=horizon,
        n_observations=T,
        mean_loss_a=mean_loss_a,
        mean_loss_b=mean_loss_b,
        mean_loss_diff=d_bar,
        std_loss_diff=d_std,
        hac_lag=lag_used,
        long_run_variance=lrv,
        se_mean=se_mean,
        dm_stat=dm_stat,
        dm_p_value_two_sided=p_two_sided,
        dm_p_value_a_better=p_a_better,
        hln_stat=hln_stat,
        hln_p_value_two_sided=hln_p_2s,
        hln_p_value_a_better=hln_p_1s,
        degrees_of_freedom=T - 1,
        better_model_by_loss=better_by_loss,
        is_identical=False,
    )
