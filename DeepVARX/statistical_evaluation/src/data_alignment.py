"""Alignment engine to synchronize prediction series between competing models.

Sync Key:
    (province, forecast_origin, target_date, horizon)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def align_model_pair(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    province: Optional[str] = None,
    horizon: int = 1,
    date_col: Optional[str] = None,
    province_col: Optional[str] = None,
    model_col: Optional[str] = None,
    actual_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    origin_col: Optional[str] = None,
    horizon_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Extract and align predictions of Model A and Model B on common observations.

    Returns:
        aligned_df: DataFrame sorted by date with columns:
            [target_date, forecast_origin, province, horizon, actual, pred_a, pred_b, error_a, error_b, (pred_std_a, pred_std_b)]
        metadata: Alignment summary
    """
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Resolve column names
    m_col = model_col or cols_lower.get("model", "model")
    p_col = province_col or cols_lower.get("province", cols_lower.get("region", "province"))
    t_col = date_col or cols_lower.get("target_date", cols_lower.get("date", "date"))
    o_col = origin_col or cols_lower.get("forecast_origin", cols_lower.get("origin", None))
    h_col = horizon_col or cols_lower.get("horizon", cols_lower.get("h", None))
    y_col = actual_col or cols_lower.get("actual", cols_lower.get("y_true", "y_true"))
    yp_col = pred_col or cols_lower.get("predicted", cols_lower.get("y_pred", "y_pred"))

    # Filter by models
    df_a = df[df[m_col] == model_a].copy()
    df_b = df[df[m_col] == model_b].copy()

    # Filter by province if provided
    if province is not None and p_col in df.columns:
        df_a = df_a[df_a[p_col] == province]
        df_b = df_b[df_b[p_col] == province]

    # Filter by horizon if column exists
    if h_col and h_col in df.columns:
        df_a = df_a[df_a[h_col] == horizon]
        df_b = df_b[df_b[h_col] == horizon]

    # Date normalization
    df_a["_dt_str"] = pd.to_datetime(df_a[t_col]).dt.strftime("%Y-%m-%d")
    df_b["_dt_str"] = pd.to_datetime(df_b[t_col]).dt.strftime("%Y-%m-%d")

    # Merge keys: (province, forecast_origin, target_date, horizon)
    merge_keys = ["_dt_str"]
    if p_col in df.columns:
        merge_keys.append(p_col)
    if h_col and h_col in df.columns:
        merge_keys.append(h_col)
    if o_col and o_col in df_a.columns and o_col in df_b.columns:
        df_a["_orig_str"] = pd.to_datetime(df_a[o_col]).dt.strftime("%Y-%m-%d")
        df_b["_orig_str"] = pd.to_datetime(df_b[o_col]).dt.strftime("%Y-%m-%d")
        merge_keys.append("_orig_str")

    merged = pd.merge(
        df_a,
        df_b,
        on=merge_keys,
        suffixes=("_a", "_b"),
        how="inner",
    )

    merged = merged.sort_values(by="_dt_str").reset_index(drop=True)

    n_common = len(merged)
    n_a_total = len(df_a)
    n_b_total = len(df_b)
    dropped_a = n_a_total - n_common
    dropped_b = n_b_total - n_common

    if dropped_a > 0 or dropped_b > 0:
        logger.warning(
            f"Alignment ({model_a} vs {model_b}, {province}, h={horizon}): "
            f"Common={n_common}, Dropped A={dropped_a}, Dropped B={dropped_b} due to non-matching keys."
        )

    # Resolve actual
    actual_a = merged[f"{y_col}_a"].to_numpy(dtype=float)
    actual_b = merged[f"{y_col}_b"].to_numpy(dtype=float)

    diff_actual = np.max(np.abs(actual_a - actual_b)) if n_common > 0 else 0.0
    if diff_actual > 1e-4:
        logger.warning(
            f"Ground truth discrepancy between {model_a} and {model_b}: max diff = {diff_actual:.4f}. Using actual_a."
        )

    actual = actual_a
    pred_a = merged[f"{yp_col}_a"].to_numpy(dtype=float)
    pred_b = merged[f"{yp_col}_b"].to_numpy(dtype=float)

    aligned_dict = {
        "date": merged["_dt_str"],
        "target_date": merged["_dt_str"],
        "forecast_origin": merged["_orig_str"] if "_orig_str" in merged.columns else (
            pd.to_datetime(merged["_dt_str"]) - pd.Timedelta(days=horizon)
        ).dt.strftime("%Y-%m-%d"),
        "province": merged[p_col] if p_col in merged.columns else (province or "all"),
        "horizon": merged[h_col] if (h_col and h_col in merged.columns) else horizon,
        "actual": actual,
        "actual_price": actual,
        "y_true": actual,  # compatibility alias
        "pred_a": pred_a,
        "pred_b": pred_b,
        "pred_price_a": pred_a,
        "pred_price_b": pred_b,
        "error_a": actual - pred_a,
        "error_b": actual - pred_b,
    }

    # Pass through research & audit metadata if available
    for meta_key in ["experiment_id", "data_frequency", "forecast_strategy", "actual_source"]:
        if f"{meta_key}_a" in merged.columns:
            aligned_dict[meta_key] = merged[f"{meta_key}_a"]

    if "model_family_a" in merged.columns:
        aligned_dict["model_a_family"] = merged["model_family_a"]
    if "model_family_b" in merged.columns:
        aligned_dict["model_b_family"] = merged["model_family_b"]
    if "model_config_a" in merged.columns:
        aligned_dict["model_a_config"] = merged["model_config_a"]
    if "model_config_b" in merged.columns:
        aligned_dict["model_b_config"] = merged["model_config_b"]
    if "prediction_type_a" in merged.columns:
        aligned_dict["model_a_type"] = merged["prediction_type_a"]
    if "prediction_type_b" in merged.columns:
        aligned_dict["model_b_type"] = merged["prediction_type_b"]

    # Add optional predictive std dev if available
    for std_alias in ["pred_std", "std", "scale", "sigma"]:
        if f"{std_alias}_a" in merged.columns and f"{std_alias}_b" in merged.columns:
            aligned_dict["pred_std_a"] = merged[f"{std_alias}_a"].to_numpy(dtype=float)
            aligned_dict["pred_std_b"] = merged[f"{std_alias}_b"].to_numpy(dtype=float)
            break

    aligned_df = pd.DataFrame(aligned_dict)

    metadata = {
        "model_a": model_a,
        "model_b": model_b,
        "province": province,
        "horizon": horizon,
        "n_common": n_common,
        "n_total_a": n_a_total,
        "n_total_b": n_b_total,
        "dropped_a": dropped_a,
        "dropped_b": dropped_b,
        "date_start": merged["_dt_str"].iloc[0] if n_common > 0 else "",
        "date_end": merged["_dt_str"].iloc[-1] if n_common > 0 else "",
    }

    return aligned_df, metadata
