"""Harvey-Leybourne-Newbold (HLN, 1997) finite-sample correction for DM Test.

Paper:
    Harvey, D., Leybourne, S., & Newbold, P. (1997).
    Testing the equality of prediction mean squared errors.
    International Journal of Forecasting, 13(2), 281-291.

Formula:
    DM_HLN = DM * sqrt( (T + 1 - 2*h + h*(h - 1)/T) / T )
    Reference distribution: Student-t with (T - 1) degrees of freedom.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy import stats


def compute_hln_multiplier(T: int, horizon: int = 1) -> float:
    """Compute the finite-sample HLN scaling factor.

    sqrt( (T + 1 - 2*h + h*(h - 1)/T) / T )
    """
    if T <= 1:
        return 1.0

    h = float(horizon)
    T_f = float(T)
    numerator = T_f + 1.0 - 2.0 * h + (h * (h - 1.0) / T_f)
    if numerator <= 0.0:
        return 1.0

    return math.sqrt(numerator / T_f)


def apply_hln_correction(
    dm_stat: float,
    T: int,
    horizon: int = 1,
) -> Tuple[float, float, float, Dict]:
    """Apply HLN correction to raw DM test statistic.

    Parameters:
        dm_stat: Raw Diebold-Mariano test statistic.
        T: Sample size (number of observations in loss differential).
        horizon: Forecast horizon h (>= 1).

    Returns:
        hln_stat: Adjusted test statistic.
        p_two_sided: Two-sided p-value from Student-t(T - 1).
        p_one_sided_a_better: One-sided p-value testing H1: E[d] < 0 (Model A has lower loss).
        diagnostics: Metadata dict (df, scale, raw_stat).
    """
    if math.isnan(dm_stat) or T <= 1:
        return np.nan, np.nan, np.nan, {
            "df": max(1, T - 1),
            "scale": np.nan,
            "raw_stat": dm_stat,
            "reason": "invalid_input_or_sample_too_small",
        }

    scale = compute_hln_multiplier(T, horizon)
    hln_stat = dm_stat * scale
    df = T - 1

    # Student-t distribution
    p_two_sided = float(2.0 * stats.t.sf(abs(hln_stat), df=df))
    # For d_t = Loss(A) - Loss(B), A is better when d_t < 0 => left tail
    p_one_sided_a_better = float(stats.t.cdf(hln_stat, df=df))

    diagnostics = {
        "df": df,
        "hln_scale": scale,
        "raw_dm_stat": dm_stat,
        "hln_stat": hln_stat,
        "horizon": horizon,
        "T": T,
    }

    return hln_stat, p_two_sided, p_one_sided_a_better, diagnostics
