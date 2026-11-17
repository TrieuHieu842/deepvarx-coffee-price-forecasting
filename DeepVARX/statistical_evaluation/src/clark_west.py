"""Clark and West (2006, 2007) test for predictive accuracy in nested models.

Papers:
    Clark, T. E., & West, K. D. (2006). Using out-of-sample mean squared prediction errors
    to test the martingale difference hypothesis. Journal of Econometrics, 135(1-2), 155-186.

    Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive
    accuracy in nested models. Journal of Econometrics, 138(1), 291-311.

Context:
    When Model A (restricted/parsimonious) is nested in Model B (unrestricted/larger),
    standard Diebold-Mariano test is undersized (conservative) because Model B introduces
    estimation noise under H0 that inflates its sample MSPE.
    Clark-West adjusts the loss differential by adding the squared difference of predictions:
        f_t = e_A,t^2 - [ e_B,t^2 - (y_pred_A,t - y_pred_B,t)^2 ]
            = (e_A,t^2 - e_B,t^2) + (y_pred_A,t - y_pred_B,t)^2
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

from .hac_estimator import estimate_hac_variance


def clark_west_test(
    y_true: np.ndarray,
    y_pred_restricted: np.ndarray,
    y_pred_unrestricted: np.ndarray,
    horizon: int = 1,
    hac_lag: Optional[int] = None,
) -> Tuple[float, float, float, Dict]:
    """Execute Clark-West test for nested models.

    Parameters:
        y_true: Ground truth array.
        y_pred_restricted: Predictions of parsimonious (restricted) model (Model A).
        y_pred_unrestricted: Predictions of larger (unrestricted) model (Model B).
        horizon: Forecast horizon h.
        hac_lag: HAC lag parameter (None for automatic).

    Returns:
        cw_stat: Clark-West test statistic (t-statistic).
        p_one_sided: One-sided p-value (H1: Unrestricted model has smaller MSPE).
        f_bar: Sample mean of adjusted loss differential.
        diagnostics: Metadata dictionary.
    """
    y = np.asarray(y_true, dtype=float)
    y_r = np.asarray(y_pred_restricted, dtype=float)
    y_u = np.asarray(y_pred_unrestricted, dtype=float)

    mask = ~(np.isnan(y) | np.isnan(y_r) | np.isnan(y_u))
    y = y[mask]
    y_r = y_r[mask]
    y_u = y_u[mask]

    T = len(y)
    if T < 5:
        return np.nan, np.nan, np.nan, {"T": T, "reason": "insufficient_observations"}

    e_r = y - y_r
    e_u = y - y_u

    # Clark-West adjustment term: (y_pred_restricted - y_pred_unrestricted)^2
    adj_term = (y_r - y_u) ** 2
    f_t = (e_r ** 2) - (e_u ** 2) + adj_term
    f_bar = float(np.mean(f_t))

    lrv, se_mean, diag = estimate_hac_variance(f_t, horizon=horizon, hac_lag=hac_lag)

    if se_mean <= 1e-18 or math.isnan(se_mean):
        cw_stat = 0.0
        p_one_sided = 0.5
    else:
        cw_stat = float(f_bar / se_mean)
        # Clark-West test is strictly a one-sided test (H1: f_bar > 0 => unrestricted is superior)
        p_one_sided = float(stats.t.sf(cw_stat, df=T - 1))

    diagnostics = {
        "T": T,
        "f_bar": f_bar,
        "se_mean": se_mean,
        "cw_stat": cw_stat,
        "p_one_sided": p_one_sided,
        "hac_lag": diag.get("q", 0),
        "mean_adj_term": float(np.mean(adj_term)),
    }

    return cw_stat, p_one_sided, f_bar, diagnostics
