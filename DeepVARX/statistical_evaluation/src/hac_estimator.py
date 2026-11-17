"""Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) estimator.

Computes the long-run variance of a stationary series d_t:
    sigma_LR^2 = gamma_0 + 2 * sum_{j=1}^q w(j, q) * gamma_j

where:
    gamma_j = (1/T) * sum_{t=j+1}^T (d_t - d_bar) * (d_{t-j} - d_bar)
    w(j, q) = 1 - j / (q + 1)  (Bartlett kernel)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def select_hac_lag(
    T: int,
    horizon: int = 1,
    hac_lag: Optional[int] = None,
) -> int:
    """Determine the lag truncation parameter (bandwidth) q for HAC estimation.

    Parameters:
        T: Sample size (number of observations).
        horizon: Forecast horizon h (>= 1).
        hac_lag: Explicit user-specified lag, or None for automatic selection.

    Logic:
        - If user supplies `hac_lag` (>= 0), respect it, but ensure q >= horizon - 1
          if horizon > 1 (due to theoretical MA(h-1) serial correlation).
        - If automatic (hac_lag is None):
            Apply Newey & West (1994) / Andrews (1991) rule-of-thumb:
                q_rule = floor( 4 * (T / 100)^(2/9) )
            Ensure q = max(horizon - 1, q_rule).
            Cap at max(1, T // 3) to prevent degrees-of-freedom exhaustion in small samples.
    """
    if T <= 2:
        return 0

    if hac_lag is not None:
        q = max(0, int(hac_lag))
        if horizon > 1 and q < horizon - 1:
            q = horizon - 1
        return min(q, max(1, T // 3))

    # Automatic selection rule
    q_rule = int(math.floor(4.0 * ((T / 100.0) ** (2.0 / 9.0))))
    q_rule = max(0, q_rule)
    q = max(horizon - 1, q_rule) if horizon > 1 else q_rule
    return min(q, max(1, T // 3))


def estimate_hac_variance(
    d: np.ndarray,
    horizon: int = 1,
    hac_lag: Optional[int] = None,
) -> Tuple[float, float, Dict]:
    """Estimate long-run variance and standard error of sample mean d_bar.

    Parameters:
        d: 1D array of loss differential series.
        horizon: Forecast horizon h.
        hac_lag: Bandwidth parameter q (None for automatic).

    Returns:
        lrv: Estimated long-run variance sigma_LR^2 (non-negative).
        se_mean: Standard error of the mean sqrt(sigma_LR^2 / T).
        diagnostics: Dictionary containing detailed estimation metadata.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)

    if T == 0:
        return np.nan, np.nan, {"T": 0, "q": 0, "gamma_0": np.nan, "reason": "empty_series"}

    d_bar = float(np.mean(d))
    dev = d - d_bar
    gamma_0 = float(np.dot(dev, dev) / T)

    if gamma_0 <= 1e-18:
        # All values identical or zero variance
        return 0.0, 0.0, {
            "T": T,
            "q": 0,
            "gamma_0": 0.0,
            "d_bar": d_bar,
            "reason": "zero_variance",
        }

    q = select_hac_lag(T, horizon=horizon, hac_lag=hac_lag)

    autocovs = []
    weights = []
    lrv = gamma_0

    for j in range(1, q + 1):
        gamma_j = float(np.dot(dev[j:], dev[:-j]) / T)
        w_j = 1.0 - (float(j) / float(q + 1))
        lrv += 2.0 * w_j * gamma_j
        autocovs.append(gamma_j)
        weights.append(w_j)

    # Guarantee non-negativity (Bartlett kernel theoretically guarantees PSD, but protect against float precision)
    if lrv < 0.0:
        lrv = gamma_0

    se_mean = math.sqrt(lrv / T) if lrv > 0 and T > 0 else 0.0

    diagnostics = {
        "T": T,
        "q": q,
        "d_bar": d_bar,
        "gamma_0": gamma_0,
        "lrv": lrv,
        "se_mean": se_mean,
        "autocovariances": autocovs,
        "bartlett_weights": weights,
    }

    return lrv, se_mean, diagnostics
