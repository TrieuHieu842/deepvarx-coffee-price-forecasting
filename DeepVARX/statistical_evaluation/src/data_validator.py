"""Validation engine for forecast data integrity and compatibility."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    num_rows: int = 0
    models_found: List[str] = field(default_factory=list)
    provinces_found: List[str] = field(default_factory=list)
    horizons_found: List[int] = field(default_factory=list)
    date_range: Tuple[str, str] = ("", "")
    ground_truth_consistent: bool = True


def validate_forecast_dataframe(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    allow_missing: bool = False,
) -> ValidationReport:
    """Validate a forecast DataFrame against schema and statistical consistency rules."""
    report = ValidationReport(is_valid=True)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        report.is_valid = False
        report.errors.append("DataFrame is empty or not a valid pandas DataFrame.")
        return report

    report.num_rows = len(df)
    cols = [c.lower() for c in df.columns]

    # Required column checks (with flexible naming)
    col_mapping = {}
    for standard_col, aliases in [
        ("date", ["date", "target_date", "forecast_date", "time"]),
        ("province", ["province", "region", "entity", "location"]),
        ("model", ["model", "model_name", "method", "forecaster"]),
        ("y_true", ["y_true", "actual", "true", "ground_truth", "actual_value"]),
        ("y_pred", ["y_pred", "predicted", "forecast", "pred", "y_hat"]),
    ]:
        found = None
        for alias in aliases:
            if alias in cols:
                found = df.columns[cols.index(alias)]
                break
        if found:
            col_mapping[standard_col] = found
        else:
            report.is_valid = False
            report.errors.append(f"Missing required column: '{standard_col}' (checked aliases: {aliases})")

    if not report.is_valid:
        return report

    # Optional horizon column
    horizon_col = None
    for alias in ["horizon", "h", "step"]:
        if alias in cols:
            horizon_col = df.columns[cols.index(alias)]
            break

    # Date parsing check
    date_series = pd.to_datetime(df[col_mapping["date"]], errors="coerce")
    invalid_dates = date_series.isna().sum()
    if invalid_dates > 0:
        report.is_valid = False
        report.errors.append(f"Found {invalid_dates} rows with unparseable date values.")
    else:
        report.date_range = (
            str(date_series.min().strftime("%Y-%m-%d")),
            str(date_series.max().strftime("%Y-%m-%d")),
        )

    # Models and provinces
    models = sorted(df[col_mapping["model"]].dropna().unique().tolist())
    provinces = sorted(df[col_mapping["province"]].dropna().unique().tolist())
    horizons = sorted(df[horizon_col].dropna().unique().tolist()) if horizon_col else [1]

    report.models_found = models
    report.provinces_found = provinces
    report.horizons_found = horizons

    if len(models) < 2:
        report.warnings.append(
            f"Only {len(models)} model found ({models}). DM test requires at least 2 models."
        )

    # Duplicate check
    keys = [col_mapping["date"], col_mapping["province"], col_mapping["model"]]
    if horizon_col:
        keys.append(horizon_col)

    dups = df.duplicated(subset=keys, keep=False)
    if dups.any():
        n_dups = dups.sum()
        report.is_valid = False
        report.errors.append(
            f"Found {n_dups} duplicate entries for unique key (date, province, model, horizon)."
        )

    # Check ground truth consistency across models
    # For every (date, province, horizon), all models should have identical y_true
    group_keys = [col_mapping["date"], col_mapping["province"]]
    if horizon_col:
        group_keys.append(horizon_col)

    truth_spread = df.groupby(group_keys)[col_mapping["y_true"]].agg(lambda s: s.max() - s.min())
    max_discrepancy = float(truth_spread.max()) if not truth_spread.empty else 0.0

    if max_discrepancy > 1e-4:
        report.ground_truth_consistent = False
        report.warnings.append(
            f"Inconsistent ground truth detected! Max discrepancy between models: {max_discrepancy:.4f}. "
            f"Ensure all models evaluated on the same target series."
        )

    # Numeric check
    for col_key in ["y_true", "y_pred"]:
        col_name = col_mapping[col_key]
        non_num = pd.to_numeric(df[col_name], errors="coerce").isna().sum()
        if non_num > 0 and not allow_missing:
            report.is_valid = False
            report.errors.append(f"Column '{col_name}' contains {non_num} non-numeric/NaN values.")

    return report
