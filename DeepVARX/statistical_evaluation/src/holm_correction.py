"""Holm-Bonferroni (1979) step-down procedure for multiple hypothesis testing.

Controls the Family-Wise Error Rate (FWER) under arbitrary dependence.
Paper:
    Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np


def holm_bonferroni_correction(
    p_values: Sequence[float],
) -> np.ndarray:
    """Adjust a family of raw p-values using the Holm-Bonferroni step-down procedure.

    Algorithm:
        1. Let m = len(p_values).
        2. Sort raw p-values in ascending order: p_(1) <= p_(2) <= ... <= p_(m).
        3. Step-down adjusted p-values:
           p_adj_(k) = min( (m - k + 1) * p_(k), 1.0 ) for k = 1, ..., m
        4. Enforce monotonicity:
           p_adj_(k) = max( p_adj_(k), p_adj_(k-1) )
        5. Restore original input ordering.

    Guarantees:
        - 0.0 <= p_adj <= 1.0
        - p_adj >= p_raw for all tests
        - Output preserves original indexing
    """
    raw = np.asarray(p_values, dtype=float)
    m = len(raw)

    if m == 0:
        return np.array([], dtype=float)
    if m == 1:
        return np.clip(raw, 0.0, 1.0)

    # Handle NaNs: preserve NaN locations, adjust valid p-values
    nan_mask = np.isnan(raw)
    valid_indices = np.where(~nan_mask)[0]
    valid_m = len(valid_indices)

    if valid_m == 0:
        return raw.copy()

    valid_p = raw[valid_indices]
    sort_order = np.argsort(valid_p)
    sorted_p = valid_p[sort_order]

    adjusted_sorted = np.empty(valid_m, dtype=float)
    running_max = 0.0

    for rank, p in enumerate(sorted_p):
        multiplier = valid_m - rank
        adj = min(float(p) * multiplier, 1.0)
        running_max = max(running_max, adj)
        # Ensure it does not drop below raw p-value
        running_max = max(running_max, float(p))
        adjusted_sorted[rank] = min(running_max, 1.0)

    # Invert sorting back to original valid order
    inv_sort = np.empty(valid_m, dtype=int)
    inv_sort[sort_order] = np.arange(valid_m)
    adjusted_valid = adjusted_sorted[inv_sort]

    result = np.full(m, np.nan, dtype=float)
    result[valid_indices] = adjusted_valid
    return result
